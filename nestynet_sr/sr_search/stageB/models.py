# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Model classes for Stage B subtree evaluation.

This module contains wrapper models that enable analytic gradient and Hessian
computation for AST subtrees during Stage B refinement.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AcosNode,
    AddNode,
    ArgNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    RealNode,
    SinNode,
    const_full_like,
)


class _SubtreeModel(nn.Module):
    """
    Black-box model for a single AST subtree.

    It reuses the *trained* leaf modules from the current Stage-B model,
    and evaluates the subtree purely in terms of Add/Mul/Pow/Log and
    those leaves.

    Forward signature:
        x_full : [B, Nxvars]  ->  u(x_full) : [B, 1]
    """

    def __init__(self, root: Node, atom_to_leaf: Dict[int, nn.Module]):
        super().__init__()
        self.root = root
        # Don't register leaves; just keep references.
        self._atom_to_leaf = dict(atom_to_leaf)

    # Optional: for symmetry with other models
    @property
    def O(self):  # scalar output
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, _, _ = self._value_grad_grad(x, need_gg=False)
        return v

    def grad(self, cache_or_x, out_dim=None):
        if isinstance(cache_or_x, dict):
            x = cache_or_x["x"]
        else:
            x = cache_or_x
        _, g, _ = self._value_grad_grad(x, need_gg=False)
        if out_dim is not None:
            return g[:, out_dim]
        return g

    def grad_grad(self, cache_or_x, out_dim=None):
        if isinstance(cache_or_x, dict):
            x = cache_or_x["x"]
        else:
            x = cache_or_x
        _, _, gg = self._value_grad_grad(x, need_gg=True)
        if out_dim is not None:
            return gg[:, out_dim]
        return gg

    def _value_grad_grad(self, x: torch.Tensor, *, need_gg: bool):
        """
        Compute value, gradient and optionally Hessian w.r.t. inputs.

        Returns
        -------
        (v, g, gg) where
          v  : [B, 1]
          g  : [B, 1, Nx]
          gg : [B, 1, Nx, Nx] or None
        """
        B, Nx = x.shape
        O = 1  # scalar output by design

        def eval_node(node: Node):
            # ---- Fixed constant -------------------------------------------
            if isinstance(node, ConstNode):
                v = x.new_full((B, 1), float(getattr(node, "value", 0.0)))
                g = x.new_zeros(B, O, Nx)
                gg = x.new_zeros(B, O, Nx, Nx) if need_gg else None
                return v, g, gg

            # ---- Leaf ----------------------------------------------------
            if isinstance(node, AtomNode):
                kind = str(getattr(node, "kind", "")).lower()

                # Pure variable nodes can appear inside compound input_expr trees.
                if kind in ("var", "x", "input"):
                    if len(node.var_idxs) != 1:
                        raise ValueError("Var node must have exactly 1 var_idx")
                    j = int(node.var_idxs[0])
                    v = x[:, j : j + 1]
                    g = x.new_zeros(B, O, Nx)
                    g[:, :, j] = 1.0
                    gg = x.new_zeros(B, O, Nx, Nx) if need_gg else None
                    return v, g, gg

                leaf = self._atom_to_leaf.get(id(node), None)
                if leaf is None:
                    raise KeyError(f"No leaf module found for atom {node}")

                # --- Unified leaf evaluation (compound & simple) ---
                # Build x_in and the Jacobian/Hessian of x_in w.r.t. x.
                from nestynet_sr.sr_core.bridges import get_input_exprs, is_trivial_input

                inputs = get_input_exprs(node)
                n_in = len(inputs)
                vals, g_ins, gg_ins = [], [], []
                for inp in inputs:
                    if is_trivial_input(inp):
                        j = int(inp.var_idxs[0])
                        v_i = x[:, j : j + 1]
                        g_i = x.new_zeros(B, 1, Nx)
                        g_i[:, 0, j] = 1.0
                        gg_i = x.new_zeros(B, 1, Nx, Nx) if need_gg else None
                    else:
                        v_i, g_i, gg_i = eval_node(inp)
                    vals.append(v_i)
                    g_ins.append(g_i)
                    if need_gg:
                        gg_ins.append(gg_i if gg_i is not None else x.new_zeros(B, 1, Nx, Nx))

                x_in = torch.cat(vals, dim=1)           # [B, n_in]
                g_in = torch.cat(g_ins, dim=1)           # [B, n_in, Nx]

                v_leaf = leaf(x_in)
                if v_leaf.dim() == 1:
                    v_leaf = v_leaf.view(-1, 1)

                cache_like = {"x": x_in}
                g_t = leaf.grad(cache_like, allow_unused=True)   # [B, O, n_in]
                gg_t = None
                if need_gg and hasattr(leaf, "grad_grad"):
                    gg_t = leaf.grad_grad(cache_like, allow_unused=True)  # [B, O, n_in, n_in]

                # Chain rule: g = g_t @ g_in  →  [B, O, Nx]
                g = torch.einsum('boi,biN->boN', g_t, g_in)

                gg = None
                if need_gg:
                    gg_in = torch.cat(gg_ins, dim=1)  # [B, n_in, Nx, Nx]
                    gg = x.new_zeros(B, O, Nx, Nx)
                    # J^T H_t J term
                    if gg_t is not None:
                        # gg += g_in^T @ gg_t @ g_in (full contraction)
                        for p in range(n_in):
                            for q in range(n_in):
                                Hpq = gg_t[:, :, p, q]  # [B, O]
                                outer = g_in[:, p:p+1, :].unsqueeze(-1) * g_in[:, q:q+1, :].unsqueeze(-2)  # [B,1,Nx,Nx]
                                gg += Hpq[..., None, None] * outer
                    # + sum_i g_t_i * gg_in_i
                    for i in range(n_in):
                        df_di = g_t[:, :, i:i+1]  # [B, O, 1]
                        gg += df_di[..., None] * gg_in[:, i:i+1, :, :]

                return v_leaf, g, gg

            # ---- Addition ------------------------------------------------
            if isinstance(node, AddNode):
                v1, g1, gg1 = eval_node(node.left)
                v2, g2, gg2 = eval_node(node.right)
                v = v1 + v2
                g = g1 + g2
                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    if gg2 is None:
                        gg2 = g2.new_zeros(B, O, Nx, Nx)
                    gg = gg1 + gg2
                else:
                    gg = None
                return v, g, gg

            # ---- Multiplication -----------------------------------------
            if isinstance(node, MulNode):
                v1, g1, gg1 = eval_node(node.left)
                v2, g2, gg2 = eval_node(node.right)

                v = v1 * v2  # [B,1]
                g = v2[..., None] * g1 + v1[..., None] * g2  # [B,1,Nx]

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    if gg2 is None:
                        gg2 = g2.new_zeros(B, O, Nx, Nx)

                    outer = g1.unsqueeze(-1) * g2.unsqueeze(-2)
                    outer = outer + outer.transpose(-1, -2)
                    gg = v2[..., None, None] * gg1 + v1[..., None, None] * gg2 + outer
                else:
                    gg = None
                return v, g, gg

            # ---- Power (scalar exponent) --------------------------------
            if isinstance(node, PowNode):
                v1, g1, gg1 = eval_node(node.base)
                c = float(node.exponent)

                v = v1.pow(c)
                g = c * v1.pow(c - 1.0)[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    term1 = c * v1.pow(c - 1.0)[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = c * (c - 1.0) * v1.pow(c - 2.0)[..., None, None] * outer
                    gg = term1 + term2
                else:
                    gg = None
                return v, g, gg

            # ---- Log -----------------------------------------------------
            if isinstance(node, LogNode):
                v1, g1, gg1 = eval_node(node.arg)

                v = torch.log(v1)
                inv = 1.0 / v1
                g = inv[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    term1 = inv[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = (inv.square())[..., None, None] * outer
                    gg = term1 - term2
                else:
                    gg = None
                return v, g, gg

            # ---- Exp ------------------------------------------------------
            if isinstance(node, ExpNode):
                v1, g1, gg1 = eval_node(node.arg)

                v = torch.exp(v1)
                g = v[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    term1 = v[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = v[..., None, None] * outer  # grad f ⊗ grad f
                    gg = term1 + term2  # e^f (H_f + grad f grad f^T)
                else:
                    gg = None
                return v, g, gg

            # ---- Sin -----------------------------------------------------
            if isinstance(node, SinNode):
                v1, g1, gg1 = eval_node(node.arg)

                v = torch.sin(v1)
                cos_v1 = torch.cos(v1)
                g = cos_v1[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    term1 = cos_v1[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = v[..., None, None] * outer  # v = sin(f)
                    gg = term1 - term2
                else:
                    gg = None
                return v, g, gg

            # ---- Cos -----------------------------------------------------
            if isinstance(node, CosNode):
                v1, g1, gg1 = eval_node(node.arg)

                v = torch.cos(v1)
                sin_v1 = torch.sin(v1)
                g = -sin_v1[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    term1 = -sin_v1[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = v[..., None, None] * outer  # v = cos(f)
                    gg = term1 - term2
                else:
                    gg = None
                return v, g, gg

            # ---- Inverse trig --------------------------------------------
            if isinstance(node, AsinNode):
                v1, g1, gg1 = eval_node(node.arg)
                v1c = torch.clamp(v1, min=-1.0 + 1.0e-12, max=1.0 - 1.0e-12)
                v = torch.asin(v1c)
                denom = torch.clamp(1.0 - v1c * v1c, min=1.0e-24)
                d1 = torch.rsqrt(denom)
                g = d1[..., None] * g1
                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    d2 = v1c * torch.pow(denom, -1.5)
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    gg = d1[..., None, None] * gg1 + d2[..., None, None] * outer
                else:
                    gg = None
                return v, g, gg

            if isinstance(node, AcosNode):
                v1, g1, gg1 = eval_node(node.arg)
                v1c = torch.clamp(v1, min=-1.0 + 1.0e-12, max=1.0 - 1.0e-12)
                v = torch.acos(v1c)
                denom = torch.clamp(1.0 - v1c * v1c, min=1.0e-24)
                d1 = -torch.rsqrt(denom)
                g = d1[..., None] * g1
                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    d2 = -v1c * torch.pow(denom, -1.5)
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    gg = d1[..., None, None] * gg1 + d2[..., None, None] * outer
                else:
                    gg = None
                return v, g, gg

            if isinstance(node, AtanNode):
                v1, g1, gg1 = eval_node(node.arg)
                v = torch.atan(v1)
                denom = 1.0 + v1 * v1
                d1 = 1.0 / denom
                g = d1[..., None] * g1
                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    d2 = -2.0 * v1 / (denom * denom)
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    gg = d1[..., None, None] * gg1 + d2[..., None, None] * outer
                else:
                    gg = None
                return v, g, gg

            # ---- ConstNode ------------------------------------------------
            if isinstance(node, ConstNode):
                v = const_full_like(x, (B, O), node.value)
                g = v.new_zeros(B, O, Nx)
                gg = None
                if need_gg:
                    gg = v.new_zeros(B, O, Nx, Nx)
                return v, g, gg

            # ---- Complex unary ops ----------------------------------------
            if isinstance(node, ConjNode):
                v1, g1, gg1 = eval_node(node.arg)
                v = torch.conj(v1)
                g = torch.conj(g1)
                gg = torch.conj(gg1) if gg1 is not None else None
                return v, g, gg

            if isinstance(node, RealNode):
                v1, g1, gg1 = eval_node(node.arg)
                v = v1.real if torch.is_complex(v1) else v1
                g = g1.real if torch.is_complex(g1) else g1
                gg = gg1.real if (gg1 is not None and torch.is_complex(gg1)) else gg1
                return v, g, gg

            if isinstance(node, ImagNode):
                v1, g1, gg1 = eval_node(node.arg)
                if torch.is_complex(v1):
                    v = v1.imag
                    g = g1.imag
                    gg = gg1.imag if gg1 is not None else None
                else:
                    v = x.new_zeros(v1.shape)
                    g = x.new_zeros(g1.shape)
                    gg = x.new_zeros(gg1.shape) if gg1 is not None else None
                return v, g, gg

            if isinstance(node, AbsNode):
                v1, g1, gg1 = eval_node(node.arg)
                v = torch.abs(v1)
                # Gradient: d|z|/dz = z / |z| (for z != 0)
                safe = v.abs() > 1e-12
                grad_factor = torch.where(safe, v1 / v, v1.new_zeros(v1.shape))
                if torch.is_complex(grad_factor):
                    grad_factor = grad_factor.real
                g = grad_factor[..., None] * g1
                if torch.is_complex(g):
                    g = g.real
                # Hessian approximation (ignore second-order terms for simplicity)
                gg = None
                if need_gg:
                    gg = x.new_zeros(B, O, Nx, Nx)
                return v, g, gg

            if isinstance(node, ArgNode):
                v1, g1, gg1 = eval_node(node.arg)
                v = torch.angle(v1)
                # Gradient: d arg(z)/dz = -i / z (complex derivative)
                # For real-valued output, this is Im(-i/z) * dz
                safe = v1.abs() > 1e-12
                inv_z = torch.where(safe, 1.0 / v1, v1.new_zeros(v1.shape))
                # d arg(z) = Im(-i * dz / z) = Re(dz / z) when dz is real
                grad_factor = torch.where(safe, -inv_z.imag if torch.is_complex(inv_z) else inv_z.new_zeros(inv_z.shape), inv_z.new_zeros(inv_z.shape))
                g = grad_factor[..., None] * g1
                if torch.is_complex(g):
                    g = g.real
                gg = None
                if need_gg:
                    gg = x.new_zeros(B, O, Nx, Nx)
                return v, g, gg

            raise TypeError(f"Unsupported node type in _SubtreeModel: {type(node)}")

        return eval_node(self.root)


class _OuterTransformedSubtreeModel(nn.Module):
    """
    Teacher wrapper: v(x) = T(u(x)) with analytic grad/grad_grad via chain rule.

    Supported transforms: identity, log, sqrt, square, recip, arcsin

    NOTE: This is the single canonical definition used throughout stageB.
    Do not create duplicate class definitions elsewhere.

    model_u must provide _value_grad_grad(x, need_gg=...).
    """

    def __init__(self, model_u: nn.Module, transform: str, eps: float = 1e-12):
        super().__init__()
        self.model_u = model_u
        self.transform = str(transform)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.model_u(x)
        if self.transform == "identity":
            return u
        if self.transform == "log":
            return torch.log(torch.clamp(u, min=self.eps))
        if self.transform == "sqrt":
            return torch.sqrt(torch.clamp(u, min=0.0))
        if self.transform == "square":
            return u * u
        if self.transform == "recip":
            return 1.0 / (u + 1e-20)  # gentle clamp/eps managed in domain check
        if self.transform == "arcsin":
            return torch.arcsin(torch.clamp(u, -1.0 + self.eps, 1.0 - self.eps))
        raise ValueError(f"Unsupported transform: {self.transform}")

    def grad(self, cache_or_x, out_dim=None):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        _, g, _ = self._value_grad_grad(x, need_gg=False)
        return g[:, out_dim] if out_dim is not None else g

    def grad_grad(self, cache_or_x, out_dim=None):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        _, _, gg = self._value_grad_grad(x, need_gg=True)
        return gg[:, out_dim] if out_dim is not None else gg

    def _phi_prime_second(self, u: torch.Tensor):
        t = self.transform
        if t == "identity":
            v = u
            phi1 = torch.ones_like(u)
            phi2 = torch.zeros_like(u)
            return v, phi1, phi2
        if t == "log":
            u_safe = torch.clamp(u, min=self.eps)
            v = torch.log(u_safe)
            inv = 1.0 / u_safe
            phi1 = inv
            phi2 = -(inv * inv)
            return v, phi1, phi2
        if t == "sqrt":
            u_safe = torch.clamp(u, min=0.0)
            s = torch.sqrt(u_safe + self.eps)
            v = s
            phi1 = 0.5 / s
            phi2 = -0.25 / (s * s * s)
            return v, phi1, phi2
        if t == "square":
            v = u * u
            phi1 = 2 * u
            phi2 = 2 * torch.ones_like(u)
            return v, phi1, phi2
        if t == "recip":
            u_safe = u.clone()
            # Avoid division by zero
            mask = u_safe.abs() < self.eps
            u_safe[mask] = self.eps * u_safe[mask].sign() + 1e-20  # push away
            v = 1.0 / u_safe
            phi1 = -1.0 / (u_safe * u_safe)
            phi2 = 2.0 / (u_safe * u_safe * u_safe)
            return v, phi1, phi2
        if t == "arcsin":
            # arcsin'(u) = 1/sqrt(1-u²),  arcsin''(u) = u/(1-u²)^{3/2}
            u_safe = torch.clamp(u, -1.0 + self.eps, 1.0 - self.eps)
            v = torch.arcsin(u_safe)
            denom = 1.0 - u_safe * u_safe          # 1 - u²
            sqrt_d = torch.sqrt(denom)              # sqrt(1 - u²)
            phi1 = 1.0 / sqrt_d
            phi2 = u_safe / (denom * sqrt_d)        # u / (1 - u²)^{3/2}
            return v, phi1, phi2
        raise ValueError(f"Unsupported transform: {t}")

    def _value_grad_grad(self, x: torch.Tensor, *, need_gg: bool):
        u, g_u, gg_u = self.model_u._value_grad_grad(x, need_gg=need_gg)
        v, phi1, phi2 = self._phi_prime_second(u)
        g = phi1[..., None] * g_u
        if not need_gg:
            return v, g, None
        if gg_u is None:
            gg_u = g_u.new_zeros(*g_u.shape, g_u.shape[-1])
        outer = g_u.unsqueeze(-1) * g_u.unsqueeze(-2)
        gg = phi1[..., None, None] * gg_u + phi2[..., None, None] * outer
        return v, g, gg
