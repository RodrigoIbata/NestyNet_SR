# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Small shared dimensional helpers for compound proposal builders."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import Node

try:
    from nestynet_sr.sr_core.units import eval_analytic_expr_dim, is_dimless
except Exception:  # pragma: no cover
    eval_analytic_expr_dim = None  # type: ignore
    is_dimless = None  # type: ignore


DimTuple = Tuple[Any, ...]


def dim_tuple(d: Any) -> Optional[DimTuple]:
    if d is None:
        return None
    try:
        return tuple(d)
    except Exception:
        return None


def same_dim(a: Optional[DimTuple], b: Optional[DimTuple]) -> bool:
    """Return True if dimensions are compatible, treating unknown as permissive."""

    if a is None or b is None:
        return True
    return tuple(a) == tuple(b)


def is_dimless_dim(d: Optional[DimTuple]) -> bool:
    """Return True if dimensionless, treating unknown as permissive."""

    if d is None:
        return True
    if is_dimless is None:
        return True
    try:
        return bool(is_dimless(tuple(d)))
    except Exception:
        return True


def scale_dim(d: Optional[DimTuple], p: float | Fraction) -> Optional[DimTuple]:
    if d is None:
        return None
    try:
        frac = Fraction(p).limit_denominator(128)
        return tuple(e * frac for e in d)
    except Exception:
        return None


def add_dim(a: Optional[DimTuple], b: Optional[DimTuple]) -> Optional[DimTuple]:
    if a is None or b is None:
        return None
    try:
        return tuple(x + y for x, y in zip(tuple(a), tuple(b)))
    except Exception:
        return None


def sub_dim(a: Optional[DimTuple], b: Optional[DimTuple]) -> Optional[DimTuple]:
    if a is None or b is None:
        return None
    try:
        return tuple(x - y for x, y in zip(tuple(a), tuple(b)))
    except Exception:
        return None


def eval_expr_dim(expr: Node, units_spec: Any) -> Optional[DimTuple]:
    """Evaluate an analytic expression dimension in a UnitsSpec context."""

    if units_spec is None or eval_analytic_expr_dim is None:
        return None
    try:
        return dim_tuple(
            eval_analytic_expr_dim(
                expr,
                units_spec.x_dims,
                free_const_dims=getattr(units_spec, "free_const_dims", {}) or {},
                fixed_const_dims=getattr(units_spec, "fixed_const_dims", {}) or {},
            )
        )
    except Exception:
        return None


def input_dims(inputs: Sequence[Node], units_spec: Any) -> list[Optional[DimTuple]]:
    if units_spec is None:
        return [None for _ in inputs]
    return [eval_expr_dim(expr, units_spec) for expr in inputs]
