# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Exact dimensional algebra for polynomial and rational coefficients.

For an active polynomial term ``a_i * x**alpha_i``, dimensional homogeneity
requires

``dim(a_i) + sum_j(alpha_ij * dim(x_j)) == dim(polynomial block)``.

The same equation handles unitless and unitful coefficients; unitless is just
the zero dimension vector.  A rational expression adds one relation between
its two additive blocks:

``dim(numerator block) - dim(denominator block) == dim(output)``.

This module deliberately knows nothing about Torch modules, fitted values, or
AST nodes.  Later producer integrations can therefore use it before fitting to
construct admissible supports, and use it again after fitting as an assertion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional, Sequence

from .units import Dim, add_dim, is_dimless, scale_dim, sub_dim


def _as_fraction(value: Any, *, max_denominator: int = 1024) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        raise TypeError("boolean values are not dimension exponents")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("dimension exponents must be finite")
        return Fraction.from_float(value).limit_denominator(max_denominator)
    try:
        return Fraction(str(value))
    except Exception as exc:
        raise TypeError(f"unsupported dimension exponent {value!r}") from exc


def _as_polynomial_exponent(value: Any) -> Fraction:
    """Normalize support metadata without rounding a near-integer to an integer."""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        raise TypeError("boolean values are not polynomial exponents")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("polynomial exponents must be finite")
        if not value.is_integer():
            raise ValueError(
                f"polynomial exponent {value!r} is not an exact integer"
            )
        return Fraction(int(value), 1)
    try:
        return Fraction(str(value))
    except Exception as exc:
        raise TypeError(f"unsupported polynomial exponent {value!r}") from exc


def normalize_dimension(
    dimension: Sequence[Any],
    *,
    rank: Optional[int] = None,
    label: str = "dimension",
) -> Dim:
    """Return an exact dimension tuple and optionally enforce its rank."""

    if isinstance(dimension, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of exponents")
    try:
        out = tuple(_as_fraction(value) for value in dimension)
    except TypeError:
        raise
    except Exception as exc:
        raise TypeError(f"{label} must be a sequence of exponents") from exc
    if rank is not None and len(out) != int(rank):
        raise ValueError(f"{label} has rank {len(out)}; expected {int(rank)}")
    return out


def _normalize_input_dimensions(
    input_dims: Sequence[Sequence[Any]],
    *,
    rank: Optional[int] = None,
) -> tuple[Dim, ...]:
    dims = tuple(
        normalize_dimension(dim, rank=rank, label=f"input_dims[{index}]")
        for index, dim in enumerate(input_dims)
    )
    if not dims:
        raise ValueError("at least one input dimension is required")
    inferred_rank = len(dims[0])
    if inferred_rank <= 0:
        raise ValueError("dimension rank must be positive")
    for index, dim in enumerate(dims[1:], start=1):
        if len(dim) != inferred_rank:
            raise ValueError(
                f"input_dims[{index}] has rank {len(dim)}; expected {inferred_rank}"
            )
    return dims


def _normalize_exponent(
    exponent: Sequence[Any],
    *,
    n_inputs: int,
    polynomial: bool,
    label: str,
) -> tuple[Fraction, ...]:
    if isinstance(exponent, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    normalizer = _as_polynomial_exponent if polynomial else _as_fraction
    values = tuple(normalizer(value) for value in exponent)
    if len(values) != int(n_inputs):
        raise ValueError(
            f"{label} has arity {len(values)}; expected {int(n_inputs)}"
        )
    if polynomial:
        for value in values:
            if value.denominator != 1 or value < 0:
                raise ValueError(
                    f"{label} contains non-polynomial exponent {value}; "
                    "rational-polynomial supports require non-negative integers"
                )
    return values


def _monomial_dimension_normalized(
    exponent: Sequence[Fraction],
    input_dims: Sequence[Dim],
    *,
    rank: int,
) -> Dim:
    out = tuple(Fraction(0) for _ in range(rank))
    for power, input_dim in zip(exponent, input_dims):
        out = add_dim(out, scale_dim(input_dim, Fraction(power)))
    return out


def monomial_dimension(
    exponent: Sequence[Any],
    input_dims: Sequence[Sequence[Any]],
) -> Dim:
    """Return ``dim(prod_j x_j**exponent_j)`` using exact rationals."""

    dims = _normalize_input_dimensions(input_dims)
    powers = _normalize_exponent(
        exponent,
        n_inputs=len(dims),
        polynomial=False,
        label="exponent",
    )
    return _monomial_dimension_normalized(powers, dims, rank=len(dims[0]))


def required_coefficient_dimension(
    block_dim: Sequence[Any],
    exponent: Sequence[Any],
    input_dims: Sequence[Sequence[Any]],
) -> Dim:
    """Return the coefficient dimension required for one additive term."""

    block = normalize_dimension(block_dim, label="block_dim")
    if not block:
        raise ValueError("dimension rank must be positive")
    dims = _normalize_input_dimensions(input_dims, rank=len(block))
    powers = _normalize_exponent(
        exponent,
        n_inputs=len(dims),
        polynomial=False,
        label="exponent",
    )
    monomial = _monomial_dimension_normalized(powers, dims, rank=len(block))
    return sub_dim(block, monomial)


def coefficient_dimensions_for_support(
    block_dim: Sequence[Any],
    exponents: Sequence[Sequence[Any]],
    input_dims: Sequence[Sequence[Any]],
) -> tuple[Dim, ...]:
    """Return required dimensions for coefficients aligned with a support."""

    return tuple(
        required_coefficient_dimension(block_dim, exponent, input_dims)
        for exponent in exponents
    )


def term_dimension(
    coefficient_dim: Sequence[Any],
    exponent: Sequence[Any],
    input_dims: Sequence[Sequence[Any]],
) -> Dim:
    """Return the dimension of ``coefficient * monomial``."""

    coefficient = normalize_dimension(coefficient_dim, label="coefficient_dim")
    if not coefficient:
        raise ValueError("dimension rank must be positive")
    monomial = monomial_dimension(exponent, input_dims)
    if len(monomial) != len(coefficient):
        raise ValueError(
            f"coefficient_dim has rank {len(coefficient)}; "
            f"monomial has rank {len(monomial)}"
        )
    return add_dim(coefficient, monomial)


def _dim_payload(dim: Optional[Dim]) -> Optional[list[str]]:
    if dim is None:
        return None
    return [str(value) for value in dim]


def _exponent_payload(exponent: Sequence[Fraction]) -> list[int | str]:
    return [
        int(value.numerator) if value.denominator == 1 else str(value)
        for value in exponent
    ]


@dataclass(frozen=True)
class GaugeAnchor:
    """One coefficient constraint that implies an additive block dimension."""

    side: str
    index: int
    exponent: tuple[Fraction, ...]
    monomial_dim: Dim
    coefficient_dim: Dim
    implied_block_dim: Dim
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "index": int(self.index),
            "exponent": _exponent_payload(self.exponent),
            "monomial_dim": _dim_payload(self.monomial_dim),
            "coefficient_dim": _dim_payload(self.coefficient_dim),
            "implied_block_dim": _dim_payload(self.implied_block_dim),
            "source": self.source,
        }


@dataclass(frozen=True)
class CoefficientDimensionRequirement:
    """Required dimension for one active coefficient in a solved gauge."""

    side: str
    index: int
    exponent: tuple[Fraction, ...]
    monomial_dim: Dim
    required_dim: Dim
    constraint_dim: Optional[Dim]
    constraint_source: str

    @property
    def is_dimensionless(self) -> bool:
        return is_dimless(self.required_dim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "index": int(self.index),
            "exponent": _exponent_payload(self.exponent),
            "monomial_dim": _dim_payload(self.monomial_dim),
            "required_dim": _dim_payload(self.required_dim),
            "constraint_dim": _dim_payload(self.constraint_dim),
            "constraint_source": self.constraint_source,
            "dimensionless": bool(self.is_dimensionless),
        }


@dataclass(frozen=True)
class RationalGaugeSolution:
    """Structured result of rational coefficient-dimension solving."""

    ok: bool
    code: str
    reason: str
    coefficient_policy: str
    gauge_status: str
    gauge_free: bool
    target_dim: Optional[Dim] = None
    input_dims: tuple[Dim, ...] = ()
    numerator_block_dim: Optional[Dim] = None
    denominator_block_dim: Optional[Dim] = None
    numerator: tuple[CoefficientDimensionRequirement, ...] = ()
    denominator: tuple[CoefficientDimensionRequirement, ...] = ()
    anchors: tuple[GaugeAnchor, ...] = ()
    failure_side: Optional[str] = None
    failure_index: Optional[int] = None
    expected_dim: Optional[Dim] = None
    actual_dim: Optional[Dim] = None
    solver: str = "coefficient_units_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "valid": bool(self.ok),
            "solver": self.solver,
            "code": self.code,
            "reason": self.reason,
            "coefficient_policy": self.coefficient_policy,
            "gauge_status": self.gauge_status,
            "gauge_free": bool(self.gauge_free),
            "target_dim": _dim_payload(self.target_dim),
            "input_dims": [_dim_payload(dim) for dim in self.input_dims],
            "numerator_block_dim": _dim_payload(self.numerator_block_dim),
            "denominator_block_dim": _dim_payload(self.denominator_block_dim),
            "numerator": [item.to_dict() for item in self.numerator],
            "denominator": [item.to_dict() for item in self.denominator],
            "anchors": [item.to_dict() for item in self.anchors],
            "failure_side": self.failure_side,
            "failure_index": self.failure_index,
            "expected_dim": _dim_payload(self.expected_dim),
            "actual_dim": _dim_payload(self.actual_dim),
        }


class _SolverInputError(ValueError):
    def __init__(
        self,
        code: str,
        reason: str,
        *,
        side: Optional[str] = None,
        index: Optional[int] = None,
        expected_dim: Optional[Dim] = None,
        actual_dim: Optional[Dim] = None,
    ):
        super().__init__(reason)
        self.code = str(code)
        self.reason = str(reason)
        self.side = side
        self.index = index
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim


def _normalize_support(
    exponents: Sequence[Sequence[Any]],
    *,
    n_inputs: int,
    side: str,
) -> tuple[tuple[Fraction, ...], ...]:
    rows = tuple(
        _normalize_exponent(
            exponent,
            n_inputs=n_inputs,
            polynomial=True,
            label=f"{side}_exponents[{index}]",
        )
        for index, exponent in enumerate(exponents)
    )
    if not rows:
        raise _SolverInputError(
            "empty_support",
            f"{side} support must contain at least one active monomial",
            side=side,
        )
    seen: set[tuple[Fraction, ...]] = set()
    for index, exponent in enumerate(rows):
        if exponent in seen:
            raise _SolverInputError(
                "duplicate_exponent",
                f"{side} support repeats exponent {_exponent_payload(exponent)}",
                side=side,
                index=index,
            )
        seen.add(exponent)
    return rows


def _normalize_policy(value: Any) -> str:
    raw_value = "free_const_only" if value is None else value
    policy = str(raw_value).strip().lower().replace("-", "_")
    if policy in {"free_const_only", "dimensionless", "strict"}:
        return "free_const_only"
    if policy in {"infer", "inferred", "declared_or_inferred", "unrestricted"}:
        return "infer"
    raise _SolverInputError(
        "invalid_coefficient_policy",
        f"unknown coefficient policy {value!r}; expected free_const_only or infer",
    )


def _normalize_canonical_gauge(value: Any) -> str:
    raw_value = "denominator_dimensionless" if value is None else value
    canonical = (
        str(raw_value).strip().lower().replace("-", "_")
    )
    if canonical in {"denominator_dimensionless", "denominator", "q"}:
        return "denominator_dimensionless"
    if canonical in {"numerator_dimensionless", "numerator", "p"}:
        return "numerator_dimensionless"
    raise _SolverInputError(
        "invalid_canonical_gauge",
        f"unknown canonical gauge {value!r}",
    )


def _normalize_coefficient_constraints(
    raw_dims: Optional[Sequence[Optional[Sequence[Any]]]],
    *,
    count: int,
    rank: int,
    side: str,
    policy: str,
    pivot: Optional[int],
) -> tuple[tuple[Optional[Dim], str], ...]:
    if raw_dims is None:
        raw: list[Optional[Sequence[Any]]] = [None] * int(count)
    else:
        raw = list(raw_dims)
        if len(raw) != int(count):
            raise _SolverInputError(
                "coefficient_count_mismatch",
                f"{side}_coefficient_dims has {len(raw)} entries; expected {count}",
                side=side,
            )

    pivot_index: Optional[int] = None
    if pivot is not None:
        pivot_value = pivot
        if hasattr(pivot_value, "item"):
            try:
                pivot_value = pivot_value.item()
            except Exception:
                pass
        if isinstance(pivot_value, bool) or not isinstance(pivot_value, int):
            raise _SolverInputError(
                "invalid_pivot",
                f"{side} pivot must be an integer index",
                side=side,
            )
        pivot_index = int(pivot_value)
        if pivot_index < 0 or pivot_index >= int(count):
            raise _SolverInputError(
                "invalid_pivot",
                f"{side} pivot index {pivot_index} is outside support size {count}",
                side=side,
                index=pivot_index,
            )

    zero = tuple(Fraction(0) for _ in range(rank))
    out: list[tuple[Optional[Dim], str]] = []
    for index, raw_dim in enumerate(raw):
        if raw_dim is None:
            dim = zero if policy == "free_const_only" else None
            source = (
                "anonymous_dimensionless"
                if policy == "free_const_only"
                else "inferred"
            )
        else:
            try:
                dim = normalize_dimension(
                    raw_dim,
                    rank=rank,
                    label=f"{side}_coefficient_dims[{index}]",
                )
            except Exception as exc:
                raise _SolverInputError(
                    "invalid_coefficient_dimension",
                    str(exc),
                    side=side,
                    index=index,
                ) from exc
            source = "declared"
        if index == pivot_index:
            if dim is not None and dim != zero:
                raise _SolverInputError(
                    "pivot_dimension_conflict",
                    f"{side} pivot coefficient is fixed numerically and must be dimensionless",
                    side=side,
                    index=index,
                    expected_dim=zero,
                    actual_dim=dim,
                )
            dim = zero
            source = "pivot_dimensionless"
        out.append((dim, source))
    return tuple(out)


def _anchors_for_side(
    side: str,
    support: Sequence[tuple[Fraction, ...]],
    monomial_dims: Sequence[Dim],
    constraints: Sequence[tuple[Optional[Dim], str]],
) -> tuple[GaugeAnchor, ...]:
    return tuple(
        GaugeAnchor(
            side=side,
            index=index,
            exponent=tuple(support[index]),
            monomial_dim=monomial_dims[index],
            coefficient_dim=constraint_dim,
            implied_block_dim=add_dim(constraint_dim, monomial_dims[index]),
            source=source,
        )
        for index, (constraint_dim, source) in enumerate(constraints)
        if constraint_dim is not None
    )


def _consistent_block_dim(
    anchors: Sequence[GaugeAnchor],
    *,
    side: str,
) -> Optional[Dim]:
    if not anchors:
        return None
    expected = anchors[0].implied_block_dim
    for anchor in anchors[1:]:
        if anchor.implied_block_dim != expected:
            raise _SolverInputError(
                f"{side}_block_inconsistent",
                f"{side} active terms imply incompatible additive-block dimensions",
                side=side,
                index=anchor.index,
                expected_dim=expected,
                actual_dim=anchor.implied_block_dim,
            )
    return expected


def _requirements_for_side(
    side: str,
    block_dim: Dim,
    support: Sequence[tuple[Fraction, ...]],
    monomial_dims: Sequence[Dim],
    constraints: Sequence[tuple[Optional[Dim], str]],
) -> tuple[CoefficientDimensionRequirement, ...]:
    out: list[CoefficientDimensionRequirement] = []
    for index, (exponent, monomial_dim, constraint) in enumerate(
        zip(support, monomial_dims, constraints)
    ):
        constraint_dim, source = constraint
        required_dim = sub_dim(block_dim, monomial_dim)
        if constraint_dim is not None and constraint_dim != required_dim:
            raise _SolverInputError(
                "coefficient_dimension_mismatch",
                f"{side} coefficient {index} has an incompatible declared dimension",
                side=side,
                index=index,
                expected_dim=required_dim,
                actual_dim=constraint_dim,
            )
        out.append(
            CoefficientDimensionRequirement(
                side=side,
                index=index,
                exponent=tuple(exponent),
                monomial_dim=monomial_dim,
                required_dim=required_dim,
                constraint_dim=constraint_dim,
                constraint_source=source,
            )
        )
    return tuple(out)


def solve_rational_coefficient_gauge(
    *,
    target_dim: Sequence[Any],
    input_dims: Sequence[Sequence[Any]],
    numerator_exponents: Sequence[Sequence[Any]],
    denominator_exponents: Sequence[Sequence[Any]],
    numerator_coefficient_dims: Optional[
        Sequence[Optional[Sequence[Any]]]
    ] = None,
    denominator_coefficient_dims: Optional[
        Sequence[Optional[Sequence[Any]]]
    ] = None,
    numerator_pivot: Optional[int] = None,
    denominator_pivot: Optional[int] = None,
    coefficient_policy: str = "free_const_only",
    canonical_gauge: str = "denominator_dimensionless",
) -> RationalGaugeSolution:
    """Solve coefficient dimensions for an active rational support.

    ``None`` entries in the aligned coefficient-dimension arrays represent
    anonymous coefficients.  Under ``free_const_only`` they are constrained to
    the zero dimension.  Under ``infer`` they are unconstrained and the solver
    reports the dimensions they would need; this mode is useful for planning,
    but does not itself authorize anonymous unitful constants.

    Concrete entries in those arrays are declarations: zero-dimensional and
    unitful declarations follow exactly the same equations.

    A reduced rational pivot is a literal numerical ``+1`` and is therefore a
    dimensionless coefficient.  Setting ``numerator_pivot`` or
    ``denominator_pivot`` pins the common rational gauge accordingly.
    """

    policy = "free_const_only"
    target: Optional[Dim] = None
    dims: tuple[Dim, ...] = ()
    anchors: tuple[GaugeAnchor, ...] = ()
    try:
        policy = _normalize_policy(coefficient_policy)
        canonical = _normalize_canonical_gauge(canonical_gauge)
        target = normalize_dimension(target_dim, label="target_dim")
        if not target:
            raise _SolverInputError(
                "invalid_dimension_rank",
                "target dimension rank must be positive",
            )
        rank = len(target)
        dims = _normalize_input_dimensions(input_dims, rank=rank)
        numerator_support = _normalize_support(
            numerator_exponents,
            n_inputs=len(dims),
            side="numerator",
        )
        denominator_support = _normalize_support(
            denominator_exponents,
            n_inputs=len(dims),
            side="denominator",
        )
        numerator_constraints = _normalize_coefficient_constraints(
            numerator_coefficient_dims,
            count=len(numerator_support),
            rank=rank,
            side="numerator",
            policy=policy,
            pivot=numerator_pivot,
        )
        denominator_constraints = _normalize_coefficient_constraints(
            denominator_coefficient_dims,
            count=len(denominator_support),
            rank=rank,
            side="denominator",
            policy=policy,
            pivot=denominator_pivot,
        )

        numerator_monomial_dims = tuple(
            _monomial_dimension_normalized(exponent, dims, rank=rank)
            for exponent in numerator_support
        )
        denominator_monomial_dims = tuple(
            _monomial_dimension_normalized(exponent, dims, rank=rank)
            for exponent in denominator_support
        )
        numerator_anchors = _anchors_for_side(
            "numerator",
            numerator_support,
            numerator_monomial_dims,
            numerator_constraints,
        )
        denominator_anchors = _anchors_for_side(
            "denominator",
            denominator_support,
            denominator_monomial_dims,
            denominator_constraints,
        )
        anchors = numerator_anchors + denominator_anchors
        numerator_block = _consistent_block_dim(
            numerator_anchors,
            side="numerator",
        )
        denominator_block = _consistent_block_dim(
            denominator_anchors,
            side="denominator",
        )

        if numerator_block is not None and denominator_block is not None:
            actual_target = sub_dim(numerator_block, denominator_block)
            if actual_target != target:
                raise _SolverInputError(
                    "rational_target_mismatch",
                    "numerator and denominator block dimensions do not produce the target dimension",
                    side="rational",
                    expected_dim=target,
                    actual_dim=actual_target,
                )
            gauge_status = "pinned"
            gauge_free = False
        elif numerator_block is not None:
            denominator_block = sub_dim(numerator_block, target)
            gauge_status = "pinned_by_numerator"
            gauge_free = False
        elif denominator_block is not None:
            numerator_block = add_dim(target, denominator_block)
            gauge_status = "pinned_by_denominator"
            gauge_free = False
        else:
            zero = tuple(Fraction(0) for _ in range(rank))
            if canonical == "denominator_dimensionless":
                denominator_block = zero
                numerator_block = target
                gauge_status = "free_canonical_denominator_dimensionless"
            else:
                numerator_block = zero
                denominator_block = scale_dim(target, Fraction(-1))
                gauge_status = "free_canonical_numerator_dimensionless"
            gauge_free = True

        numerator_requirements = _requirements_for_side(
            "numerator",
            numerator_block,
            numerator_support,
            numerator_monomial_dims,
            numerator_constraints,
        )
        denominator_requirements = _requirements_for_side(
            "denominator",
            denominator_block,
            denominator_support,
            denominator_monomial_dims,
            denominator_constraints,
        )
        return RationalGaugeSolution(
            ok=True,
            code="rational_gauge_solved",
            reason=(
                "active coefficient terms are dimensionally homogeneous and "
                "the rational block difference matches the target"
            ),
            coefficient_policy=policy,
            gauge_status=gauge_status,
            gauge_free=gauge_free,
            target_dim=target,
            input_dims=dims,
            numerator_block_dim=numerator_block,
            denominator_block_dim=denominator_block,
            numerator=numerator_requirements,
            denominator=denominator_requirements,
            anchors=anchors,
        )
    except _SolverInputError as exc:
        return RationalGaugeSolution(
            ok=False,
            code=exc.code,
            reason=exc.reason,
            coefficient_policy=policy,
            gauge_status="inconsistent",
            gauge_free=False,
            target_dim=target,
            input_dims=dims,
            anchors=anchors,
            failure_side=exc.side,
            failure_index=exc.index,
            expected_dim=exc.expected_dim,
            actual_dim=exc.actual_dim,
        )
    except Exception as exc:
        return RationalGaugeSolution(
            ok=False,
            code="invalid_input",
            reason=str(exc),
            coefficient_policy=policy,
            gauge_status="inconsistent",
            gauge_free=False,
            target_dim=target,
            input_dims=dims,
            anchors=anchors,
        )


# Concise alias for callers that already operate in a coefficient-units context.
solve_rational_gauge = solve_rational_coefficient_gauge


__all__ = [
    "CoefficientDimensionRequirement",
    "GaugeAnchor",
    "RationalGaugeSolution",
    "coefficient_dimensions_for_support",
    "monomial_dimension",
    "normalize_dimension",
    "required_coefficient_dimension",
    "solve_rational_coefficient_gauge",
    "solve_rational_gauge",
    "term_dimension",
]
