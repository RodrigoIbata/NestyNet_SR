# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Helpers for rendering / wrapping expressions under a y-transform inverse.

Stage A trains networks in φ(y) space. Stage B/C often produce expressions for
φ(y), and we then need to render y = φ^{-1}(·) in either string form (for logs)
or SymPy form (for simplification).

Historically, multiple modules re-implemented the same mapping from inverse
transform functions (torch) to symbolic wrappers ("exp", "sqrt", "-log", etc.).
This module centralises that mapping to avoid drift.
"""

from __future__ import annotations

from typing import Any, Optional


def _inv_names(y_op_inv: Any) -> tuple[str, str]:
    """Return (inv_name, lower_name) for a y-op inverse callable/module."""
    if y_op_inv is None:
        return "", ""
    inv_name = getattr(y_op_inv, "__name__", None)
    if inv_name is None:
        inv_name = str(y_op_inv)
    lname = str(inv_name).lower()
    return str(inv_name), lname


def wrap_phi_expr_str(
    phi_expr_str: str,
    y_op_inv: Any,
    *,
    simplify: bool = True,
    max_simplify_len: int = 200,
) -> Optional[str]:
    """Wrap a φ(y)-space expression string with the inverse y-transform.

    Parameters
    ----------
    phi_expr_str:
        Expression in φ(y) space, e.g. "log(x0) + log(x1)".
    y_op_inv:
        Inverse transform callable (torch-side), e.g. torch.exp for log.
    simplify:
        If True, attempt a small SymPy simplification pass (length-gated).

    Returns
    -------
    str | None
        Wrapped expression in y-space.
    """
    if phi_expr_str is None:
        return None

    inv_name, lname = _inv_names(y_op_inv)

    # Identity (function or module)
    if ("identity" in lname) or (lname == "<lambda>"):
        wrapped_str = phi_expr_str
    elif lname in ("sqrt", "numpy_sqrt", "torch_sqrt"):
        wrapped_str = f"sqrt({phi_expr_str})"
    elif lname in ("exp", "numpy_exp", "torch_exp"):
        wrapped_str = f"exp({phi_expr_str})"
    elif lname in ("log", "numpy_log", "torch_log"):
        wrapped_str = f"log({phi_expr_str})"
    elif lname in ("arcsin", "asin", "torch_arcsin"):
        wrapped_str = f"arcsin({phi_expr_str})"
    elif lname in ("arccos", "acos", "torch_arccos"):
        wrapped_str = f"arccos({phi_expr_str})"
    elif lname in ("arctan", "atan", "torch_arctan"):
        wrapped_str = f"arctan({phi_expr_str})"
    elif lname in ("sin", "numpy_sin", "torch_sin"):
        wrapped_str = f"sin({phi_expr_str})"
    elif lname in ("cos", "numpy_cos", "torch_cos"):
        wrapped_str = f"cos({phi_expr_str})"
    elif lname in ("tan", "numpy_tan", "torch_tan"):
        wrapped_str = f"tan({phi_expr_str})"
    elif ("logneg" in lname) or (lname in ("negexp",)) or (lname == "_torch_logneg_inv"):
        # logneg inverse: y = -exp(φ)
        wrapped_str = f"-exp({phi_expr_str})"
    elif ("expneg" in lname) or (lname in ("neglog",)) or (lname == "_torch_expneg_inv"):
        # expneg inverse: y = -log(φ)
        wrapped_str = f"-log({phi_expr_str})"
    elif ("reciprocal" in lname) or ("recip" == lname) or ("1/" in str(y_op_inv)):
        wrapped_str = f"1/({phi_expr_str})"
    elif ("sqrt1p_inv" in lname) or (lname == "_torch_sqrt1p_inv"):
        wrapped_str = f"sqrt({phi_expr_str} + 1)"
    elif lname in ("square", "torch_square"):
        wrapped_str = f"({phi_expr_str})**2"
    else:
        wrapped_str = f"{inv_name}({phi_expr_str})"

    if not simplify:
        return wrapped_str

    # Optional SymPy simplification, length-gated to avoid pathological hangs.
    if max_simplify_len is not None and len(wrapped_str) >= int(max_simplify_len):
        return wrapped_str

    try:
        import sympy as sp

        y_expr = sp.sympify(wrapped_str.replace("^", "**"))
        y_expr = sp.simplify(y_expr)
        return sp.sstr(y_expr)
    except Exception:
        return wrapped_str


def wrap_phi_expr_sympy(phi_expr, y_op_inv: Any, *, sp) -> Any:
    """Wrap a SymPy φ(y)-space expression with the inverse y-transform."""
    if phi_expr is None or y_op_inv is None:
        return phi_expr

    inv_name, lname = _inv_names(y_op_inv)

    if ("identity" in lname) or (lname == "<lambda>"):
        return phi_expr
    if lname in ("sqrt", "numpy_sqrt", "torch_sqrt"):
        return sp.sqrt(phi_expr)
    if lname in ("exp", "numpy_exp", "torch_exp"):
        return sp.exp(phi_expr)
    if lname in ("log", "numpy_log", "torch_log"):
        return sp.log(phi_expr)
    if lname in ("arcsin", "asin", "torch_arcsin"):
        return sp.asin(phi_expr)
    if lname in ("arccos", "acos", "torch_arccos"):
        return sp.acos(phi_expr)
    if lname in ("arctan", "atan", "torch_arctan"):
        return sp.atan(phi_expr)
    if lname in ("sin", "numpy_sin", "torch_sin"):
        return sp.sin(phi_expr)
    if lname in ("cos", "numpy_cos", "torch_cos"):
        return sp.cos(phi_expr)
    if lname in ("tan", "numpy_tan", "torch_tan"):
        return sp.tan(phi_expr)
    if ("logneg" in lname) or (lname in ("negexp",)) or (lname == "_torch_logneg_inv"):
        return -sp.exp(phi_expr)
    if ("expneg" in lname) or (lname in ("neglog",)) or (lname == "_torch_expneg_inv"):
        return -sp.log(phi_expr)
    if ("reciprocal" in lname) or ("recip" == lname):
        try:
            return 1 / phi_expr
        except Exception:
            return sp.Pow(phi_expr, -1)
    if ("sqrt1p_inv" in lname) or (lname == "_torch_sqrt1p_inv"):
        return sp.sqrt(phi_expr + 1)
    if lname in ("square", "torch_square"):
        return phi_expr**2

    # Unknown inverse: represent as a SymPy Function
    try:
        F = sp.Function(inv_name)
        return F(phi_expr)
    except Exception:
        return phi_expr
