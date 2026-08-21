# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Fit-only output links for LM stability / conditioning.

These functions implement output transformations ``t(y)`` that are used only
*while fitting* to improve numerical conditioning of the Levenberg–Marquardt
optimizer. The symbolic regression pipeline continues to operate, compare, and
report models in the original y-space.

When a fit-link is enabled, residuals become:

    r = t(y) - t(f)

instead of:

    r = y - f

Historically we used this only for numeric conditioning (``asinh``).  We also
support a small set of *lift* links that are useful when a candidate contains
fragile outer transforms (e.g. ``sqrt(·)``, ``log(·)``, reciprocal forms).  In
those cases a short pre-fit in a lifted space can avoid singular Jacobians.

Note: links other than ``asinh`` are primarily intended for temporary use
(during a pre-fit) followed by a short refine in the original space.
"""

from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Canonical names
# ---------------------------------------------------------------------------

def canonical_fit_link_name(name: Optional[str]) -> Optional[str]:
    """Canonicalize the fit-link name (None/identity -> None)."""
    if name is None:
        return None
    n = str(name).strip().lower()
    if n in {"", "none", "identity", "id"}:
        return None
    if n in {"asinh", "arcsinh"}:
        return "asinh"
    if n in {"square", "sq", "squared", "pow2", "power2", "x2"}:
        return "square"
    if n in {"inv_square", "invsquare", "inv_sq", "recip_square", "square_recip", "1/x2"}:
        return "inv_square"
    if n in {"recip", "reciprocal", "inv", "inverse", "1/x"}:
        return "recip"
    if n in {"exp", "exponential"}:
        return "exp"
    if n in {"log", "ln"}:
        return "log"
    raise ValueError(
        f"Unknown fit_y_link '{name}'. Supported: none, asinh, square, inv_square, recip, exp, log"
    )


def describe_fit_link(name: Optional[str], scale: float = 1.0) -> str:
    """Return a human-readable description of the fit-link."""
    n = canonical_fit_link_name(name)
    if n is None:
        return "identity"
    if n == "asinh":
        return f"asinh(y/{float(scale):.6g})"
    if n == "square":
        return "y^2"
    if n == "inv_square":
        return "1/y^2"
    if n == "recip":
        return "1/y"
    if n == "exp":
        return "exp(y)"
    if n == "log":
        return "log(y)"
    return str(n)


# ---------------------------------------------------------------------------
# t(y)
# ---------------------------------------------------------------------------

def fit_link_torch(y: torch.Tensor, name: Optional[str], scale: float = 1.0) -> torch.Tensor:
    """Apply the fit-link transformation t(y)."""
    n = canonical_fit_link_name(name)
    if n is None:
        return y

    if n == "asinh":
        s = float(scale) if (scale not in (None, 0.0)) else 1.0
        return torch.asinh(y / s)

    if n == "square":
        return y * y

    if n == "inv_square":
        # Guard against y≈0. This link is mainly used to lift inv-sqrt outputs.
        eps = 1e-12
        ya = y.abs().clamp(min=eps)
        return 1.0 / (ya * ya)

    if n == "recip":
        eps = 1e-12
        sgn = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
        y_safe = torch.where(y.abs() < eps, sgn * eps, y)
        return 1.0 / y_safe

    if n == "exp":
        # Guard against overflow; exp(80) is finite for float32/float64.
        m = 80.0
        y_cl = torch.clamp(y, min=-m, max=m)
        return torch.exp(y_cl)

    if n == "log":
        eps = 1e-12
        if not y.is_complex():
            y = torch.clamp(y, min=eps)
        return torch.log(y)

    raise ValueError(f"Unhandled fit_y_link '{name}'")


# ---------------------------------------------------------------------------
# dt/dy
# ---------------------------------------------------------------------------

def fit_link_torch_d1(y: torch.Tensor, name: Optional[str], scale: float = 1.0) -> torch.Tensor:
    """First derivative dt/dy evaluated at y."""
    n = canonical_fit_link_name(name)
    if n is None:
        return torch.ones_like(y)

    if n == "asinh":
        s = float(scale) if (scale not in (None, 0.0)) else 1.0
        u = y / s
        return 1.0 / (s * torch.sqrt(1.0 + u * u))

    if n == "square":
        return 2.0 * y

    if n == "inv_square":
        eps = 1e-12
        ya = y.abs()
        m = ya >= eps
        # For |y|>eps: d/dy |y|^{-2} = -2*sign(y)*|y|^{-3}
        denom = torch.where(m, ya, torch.ones_like(ya))
        d = -2.0 * torch.sign(y) / (denom * denom * denom)
        return torch.where(m, d, torch.zeros_like(d))

    if n == "recip":
        eps = 1e-12
        m = y.abs() >= eps
        denom = torch.where(m, y, torch.ones_like(y))
        d = -1.0 / (denom * denom)
        return torch.where(m, d, torch.zeros_like(d))

    if n == "exp":
        m = 80.0
        in_range = (y >= -m) & (y <= m)
        y_cl = torch.clamp(y, min=-m, max=m)
        d = torch.exp(y_cl)
        return torch.where(in_range, d, torch.zeros_like(d))

    if n == "log":
        eps = 1e-12
        if y.is_complex():
            return 1.0 / y
        m = y > eps
        denom = torch.where(m, y, torch.ones_like(y))
        d = 1.0 / denom
        return torch.where(m, d, torch.zeros_like(d))

    raise ValueError(f"Unhandled fit_y_link '{name}'")


# ---------------------------------------------------------------------------
# d2t/dy2 (currently unused by the LM adaptor but kept for completeness)
# ---------------------------------------------------------------------------

def fit_link_torch_d2(y: torch.Tensor, name: Optional[str], scale: float = 1.0) -> torch.Tensor:
    """Second derivative d²t/dy² evaluated at y."""
    n = canonical_fit_link_name(name)
    if n is None:
        return torch.zeros_like(y)

    if n == "asinh":
        s = float(scale) if (scale not in (None, 0.0)) else 1.0
        u = y / s
        return -(y) / (s**3 * torch.pow(1.0 + u * u, 1.5))

    if n == "square":
        return torch.full_like(y, 2.0)

    if n == "inv_square":
        eps = 1e-12
        ya = y.abs()
        m = ya >= eps
        # For |y|>eps: d2/dy2 |y|^{-2} = 6*|y|^{-4}
        denom = torch.where(m, ya, torch.ones_like(ya))
        d2 = 6.0 / (denom**4)
        return torch.where(m, d2, torch.zeros_like(d2))

    if n == "recip":
        eps = 1e-12
        m = y.abs() >= eps
        denom = torch.where(m, y, torch.ones_like(y))
        d2 = 2.0 / (denom**3)
        return torch.where(m, d2, torch.zeros_like(d2))

    if n == "exp":
        m = 80.0
        in_range = (y >= -m) & (y <= m)
        y_cl = torch.clamp(y, min=-m, max=m)
        d2 = torch.exp(y_cl)
        return torch.where(in_range, d2, torch.zeros_like(d2))

    if n == "log":
        eps = 1e-12
        if y.is_complex():
            return -1.0 / (y * y)
        m = y > eps
        denom = torch.where(m, y, torch.ones_like(y))
        d2 = -1.0 / (denom * denom)
        return torch.where(m, d2, torch.zeros_like(d2))

    raise ValueError(f"Unhandled fit_y_link '{name}'")
