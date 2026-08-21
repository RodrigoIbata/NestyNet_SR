# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Template VarPro Adaptor.

Provides an LAProvider-compatible adaptor for templates with nonlinear parameters,
using autograd for Jacobians of the template features Φ(ψ).
"""

import os
import sys
from typing import Optional

import torch

# Add parent nestynet to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from nestynet.optimizer.registry import provider_offset

from nestynet_sr.adaptors.template_varpro_base import TemplateVarProBase


class TemplateBaseAdaptor(torch.nn.Module):
    """Autograd-based adaptor for TemplateVarProBase.

    This wraps TemplateVarProBase and provides jvp/vjp via autograd,
    similar to AutogradAdaptor but specialized for templates.

    Parameters
    ----------
    base_module : TemplateVarProBase
        The template base module
    """

    def __init__(self, base_module: TemplateVarProBase, *, global_map_offset: Optional[int] = None):
        super().__init__()
        self.base = base_module
        self._global_map_offset = int(global_map_offset) if global_map_offset is not None else None

    def named_parameters(self, *a, **k):
        return self.base.named_parameters(*a, **k)

    def named_buffers(self, *a, **k):
        return self.base.named_buffers(*a, **k)

    def num_parameters(self):
        return sum(p.numel() for p in self.base.parameters())

    def pre_block(self, block=None, *, theta=None, **kw):
        """No-op: the autograd feature path holds no per-block state to prepare."""
        return None

    def build_cache(self, data, **kw):
        """Build cache with features f = Φ(x, u, du; ψ)."""
        if isinstance(data, (tuple, list)):
            x = data[0]
            y = data[1] if len(data) > 1 else None
        else:
            x = data
            y = None

        # Evaluate features
        f = self.base(x)  # (N, num_templates)

        cache = {
            "x": x,
            "f": f,
            "y": y,
            "O": 1,  # single output dimension
            "S": 1,  # single segment (all templates)
            "Pseg": self.num_parameters(),
            "SegmentedModel": False,  # Not a segmented model
        }
        return cache

    def residuals(self, cache, data_unused=None, *, track_grad=False):
        """Residuals not used directly - VarPro will compute them."""
        raise NotImplementedError("TemplateBaseAdaptor is meant to be wrapped by VarProAdaptor")

    def jvp(self, cache, v, out_dim=None):
        """Jacobian-vector product following residual sign convention.

        Returns -(∂f/∂ψ)·v to match LAProvider residual-sign convention:
        jvp returns -(J_f · v) where J_f is the forward feature Jacobian.

        Notes
        -----
        * This implementation prefers torch.func.jvp (fast, true forward-mode).
        * Falls back to a slow but correct autograd loop if torch.func is unavailable.
        """
        x = cache["x"]
        params_list = list(self.base.parameters())

        if len(params_list) == 0:
            return torch.zeros(x.shape[0], cache["f"].shape[1], device=x.device, dtype=x.dtype)

        # Split v into parameter-shaped tensors
        v_split = []
        offset = 0
        for p in params_list:
            numel = p.numel()
            v_split.append(v[offset : offset + numel].view_as(p))
            offset += numel

        # Fast-path: torch.func.jvp over parameter PyTree
        try:
            from torch.func import functional_call
            from torch.func import jvp as func_jvp

            params = {name: p for name, p in self.base.named_parameters()}
            buffers = {name: b for name, b in self.base.named_buffers()}

            v_params = {}
            offset = 0
            for name, p in self.base.named_parameters():
                numel = p.numel()
                v_params[name] = v[offset : offset + numel].view_as(p)
                offset += numel

            def f_fn(pdict):
                return functional_call(self.base, (pdict, buffers), (x,))

            _, jvp_out = func_jvp(f_fn, (params,), (v_params,))

            # Apply residual sign convention: return -(J_f · v)
            return -jvp_out

        except Exception:
            # Fallback: slow but correct elementwise directional derivatives
            N, num_templates = cache["f"].shape
            jvp_result = torch.zeros(N, num_templates, device=x.device, dtype=x.dtype)

            with torch.enable_grad():
                f = self.base(x)
                # Compute d f[i,j] / dψ · v for every output element.
                for i in range(N):
                    for j in range(num_templates):
                        f_ij = f[i, j]
                        grads = torch.autograd.grad(
                            f_ij, params_list, retain_graph=True, allow_unused=True
                        )
                        val = None
                        for g, v_part in zip(grads, v_split):
                            if g is None:
                                continue
                            contrib = (g * v_part).sum()
                            val = contrib if val is None else (val + contrib)
                        if val is None:
                            val = f_ij.new_zeros(())
                        jvp_result[i, j] = val

            # Apply residual sign convention: return -(J_f · v)
            return -jvp_result

    def vjp(self, cache, w, out_dim=None):
        """Vector-Jacobian product following residual sign convention.

        Returns -(J_fᵀ · w) to match LAProvider residual-sign convention,
        where J_f is the forward feature Jacobian.
        """
        x = cache["x"]
        params_list = list(self.base.parameters())

        P = sum(p.numel() for p in params_list)
        if P == 0:
            return torch.zeros((0,), device=x.device, dtype=x.dtype)

        # Ensure w is (N, num_templates)
        if w.dim() == 1:
            w = w.unsqueeze(1)
        elif w.dim() > 2:
            # Handle higher dims by reshaping
            w = w.view(x.shape[0], -1)

        with torch.enable_grad():
            # Compute features
            f = self.base(x)

            # Weighted sum
            loss = (w * f).sum()

            # Backprop
            grads = torch.autograd.grad(loss, params_list, allow_unused=True)

        # Flatten - ensure always 1D
        grad_parts = []
        for g, p in zip(grads, params_list):
            if g is not None:
                grad_parts.append(g.reshape(-1))
            else:
                grad_parts.append(torch.zeros(p.numel(), device=x.device, dtype=x.dtype))

        if len(grad_parts) == 0:
            return torch.zeros((1, 0), device=x.device, dtype=x.dtype)

        grad_flat = torch.cat(grad_parts)
        # Apply residual sign convention: return -(J_fᵀ · w)
        return -grad_flat.reshape(1, -1)  # Shape (1, P) to match AutogradAdaptor

    def blocks(self, block_size=None, shuffle=False):
        """Return parameter blocks for optimizer."""
        # Use forced offset if provided, otherwise use registry offset
        off_self = (
            self._global_map_offset
            if self._global_map_offset is not None
            else provider_offset.get(id(self), 0)
        )
        P = self.num_parameters()

        # Single block for all template parameters
        yield {
            "global_map": torch.arange(P, dtype=torch.long) + off_self,
            "analytic_map": torch.arange(P, dtype=torch.long) + off_self,
            "dimension_map": torch.full((P,), -1, dtype=torch.long),  # Coupled across all
        }
