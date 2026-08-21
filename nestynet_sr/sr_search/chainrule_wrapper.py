# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Virtual y-transform wrapper using chain-rule derivatives.

The wrapper exposes :math:`g(f(x))` from an identity-space model :math:`f(x)`
without retraining for each candidate y-transform.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch
import torch.nn as nn


class ModelAPI(Protocol):
    """Minimal model interface needed by separability probes."""

    def forward(self, x): ...

    def grad(self, x, out_dim=None): ...

    def grad_grad(self, x, out_dim=None): ...

    def parameters(self, recurse: bool = True): ...


class VirtualYModel(nn.Module):
    """Wrap identity model ``f`` and expose ``g(f)`` with chain-rule derivatives."""

    def __init__(
        self,
        base_model: ModelAPI,
        y_transform,
        *,
        max_abs_deriv: Optional[float] = None,
    ):
        super().__init__()
        self._inner = base_model
        self._op = y_transform.torch_op  # g(y)
        self._d1 = y_transform.d1  # g'(y)
        self._d2 = y_transform.d2  # g''(y)
        self._check = y_transform.check_fn
        self._name = y_transform.name
        self._max_abs_deriv = None
        if max_abs_deriv is not None:
            try:
                m = float(max_abs_deriv)
                if m > 0.0:
                    self._max_abs_deriv = m
            except Exception:
                self._max_abs_deriv = None

    def _sanitize_deriv(self, t: torch.Tensor) -> torch.Tensor:
        if self._max_abs_deriv is None:
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        m = float(self._max_abs_deriv)
        t = torch.nan_to_num(t, nan=0.0, posinf=m, neginf=-m)
        return torch.clamp(t, min=-m, max=m)

    def _x_from_cache(self, cache_or_x):
        return cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x

    def forward(self, x):
        f = self._inner(x)
        return self._op(f)

    def pred(self, x):
        return self.forward(x)

    def grad(self, cache_or_x, out_dim=None):
        x = self._x_from_cache(cache_or_x)
        f = self._inner(x)
        dphi = self._sanitize_deriv(self._d1(f))

        if out_dim is not None:
            g_o = self._inner.grad(x, out_dim=out_dim)
            dphi_o = dphi[:, out_dim]
            return dphi_o.unsqueeze(-1) * g_o

        g = self._inner.grad(x)
        return dphi.unsqueeze(-1) * g

    def grad_grad(self, cache_or_x, out_dim=None):
        x = self._x_from_cache(cache_or_x)
        f = self._inner(x)
        dphi = self._sanitize_deriv(self._d1(f))
        d2phi = self._sanitize_deriv(self._d2(f))

        if out_dim is not None:
            g_o = self._inner.grad(x, out_dim=out_dim)
            h_o = self._inner.grad_grad(x, out_dim=out_dim)
            dphi_o = dphi[:, out_dim]
            d2phi_o = d2phi[:, out_dim]
            outer = torch.einsum("bi,bj->bij", g_o, g_o)
            return d2phi_o[:, None, None] * outer + dphi_o[:, None, None] * h_o

        g = self._inner.grad(x)
        h = self._inner.grad_grad(x)
        outer = g.unsqueeze(-1) * g.unsqueeze(-2)
        return d2phi.unsqueeze(-1).unsqueeze(-1) * outer + dphi.unsqueeze(-1).unsqueeze(-1) * h

    def hess_diag(self, cache_or_x, out_dim=None):
        h = self.grad_grad(cache_or_x, out_dim=out_dim)
        return torch.diagonal(h, dim1=-2, dim2=-1)

    def parameters(self, recurse=True):
        return self._inner.parameters(recurse=recurse)

    def domain_ok(self, y_data):
        if self._check is None:
            return True
        mask = self._check(y_data)
        return bool(mask.all())

    def __repr__(self):
        return f"VirtualYModel(φ={self._name}, inner={self._inner.__class__.__name__})"


class ChainRuleYModel(VirtualYModel):
    """Backward-compatible alias used by existing call sites/tests."""
