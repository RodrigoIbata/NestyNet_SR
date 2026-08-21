# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared outer-wrapper transforms for the NestyNet factorized symbolic search adapter."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from nestynet_sr.sr_core.bridges import ConstNode, ExpNode, LogNode, MulNode, Node, PowNode


def _normalize_outer_wrapper_name(name: str) -> Optional[str]:
    """Normalize wrapper aliases to a small canonical set."""
    nm = str(name or "").strip().lower()
    if nm in ("recip", "inverse", "inv"):
        nm = "reciprocal"
    if nm in ("reciprocal", "log", "sqrt", "square", "exp"):
        return nm
    return None


def _outer_wrapper_forward(
    y: torch.Tensor,
    tname: str,
    *,
    eps: float = 1.0e-12,
    exp_abs_cap: float = 20.0,
    square_sign_consistency: float = 0.98,
) -> Tuple[torch.Tensor, torch.Tensor, float, str]:
    """Compute ``t = phi(y)`` with conservative domain masking."""
    y = y.view(-1).to(dtype=torch.float64)
    mask = torch.isfinite(y)
    t = torch.full_like(y, float("nan"))
    sign_hint = 1.0
    reason = "ok"

    if tname == "log":
        mask = mask & (y > float(eps))
        if bool(mask.any()):
            t[mask] = torch.log(y[mask])
        else:
            reason = "log_domain"
    elif tname == "reciprocal":
        mask = mask & (y.abs() > float(eps))
        if bool(mask.any()):
            t[mask] = torch.reciprocal(y[mask])
        else:
            reason = "reciprocal_domain"
    elif tname == "sqrt":
        mask = mask & (y >= 0.0)
        if bool(mask.any()):
            t[mask] = torch.sqrt(y[mask].clamp_min(0.0))
        else:
            reason = "sqrt_domain"
    elif tname == "square":
        if bool(mask.any()):
            ym = y[mask]
            frac_pos = float((ym > 0.0).double().mean().item())
            frac_neg = float((ym < 0.0).double().mean().item())
            frac_best = max(frac_pos, frac_neg)
            if frac_best < float(square_sign_consistency):
                mask = mask & torch.zeros_like(mask)
                reason = "square_sign_ambiguous"
            else:
                sign_hint = 1.0 if frac_pos >= frac_neg else -1.0
                t[mask] = ym * ym
        else:
            reason = "empty"
    elif tname == "exp":
        mask = mask & (y.abs() <= float(exp_abs_cap))
        if bool(mask.any()):
            t[mask] = torch.exp(y[mask])
        else:
            reason = "exp_clip"
    else:
        mask = mask & torch.zeros_like(mask)
        reason = f"unsupported:{tname}"

    return mask, t, float(sign_hint), str(reason)


def _outer_wrapper_inverse_ast(inner_expr: Node, tname: str, *, sign_hint: float = 1.0) -> Optional[Node]:
    """Wrap ``inner_expr`` with ``phi^{-1}`` for the selected wrapper transform."""
    tname = _normalize_outer_wrapper_name(tname)
    if tname is None:
        return None
    if tname == "log":
        return ExpNode(inner_expr)
    if tname == "exp":
        return LogNode(inner_expr)
    if tname == "reciprocal":
        return PowNode(inner_expr, -1.0)
    if tname == "sqrt":
        return PowNode(inner_expr, 2.0)
    if tname == "square":
        base = PowNode(inner_expr, 0.5)
        if float(sign_hint) < 0.0:
            return MulNode(ConstNode(-1.0), base)
        return base
    return None


def _is_dimless_dims(dims: Optional[Tuple[float, ...]], *, tol: float = 1.0e-12) -> bool:
    if dims is None:
        return False
    for value in dims:
        try:
            if abs(float(value)) > float(tol):
                return False
        except Exception:
            return False
    return True


def _outer_wrapper_transformed_y_dims(
    y_dims: Optional[Tuple[float, ...]],
    tname: str,
) -> Tuple[bool, Optional[Tuple[float, ...]], str]:
    """Return whether wrapper is unit-compatible and the transformed target dims."""
    tname = _normalize_outer_wrapper_name(tname)
    if tname is None:
        return False, None, "unsupported"
    if y_dims is None:
        return False, None, "unknown_target_dims"

    yd = tuple(float(v) for v in y_dims)
    if tname in ("log", "exp"):
        if not _is_dimless_dims(yd):
            return False, None, "requires_dimensionless_target"
        return True, tuple(0.0 for _ in yd), "ok"
    if tname == "reciprocal":
        return True, tuple(-v for v in yd), "ok"
    if tname == "square":
        return True, tuple(2.0 * v for v in yd), "ok"
    if tname == "sqrt":
        return True, tuple(0.5 * v for v in yd), "ok"
    return False, None, "unsupported"


__all__ = [
    "_normalize_outer_wrapper_name",
    "_outer_wrapper_forward",
    "_outer_wrapper_inverse_ast",
    "_outer_wrapper_transformed_y_dims",
]
