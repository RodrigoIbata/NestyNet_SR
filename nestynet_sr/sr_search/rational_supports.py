# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Construct exact unit-consistent supports for rational-polynomial producers.

The trainable coefficients inside the current rational leaves are anonymous
scalars.  Under the default ``free_const_only`` policy those scalars are
dimensionless, so each additive block must be assembled from monomials in one
exact dimension class.  This module performs that construction *before* a
numeric fit and then delegates every emitted support to the shared coefficient
gauge solver as a safety assertion.

There is intentionally no separate dimensionless algorithm.  Zero-dimensional
inputs and targets pass through the same grouping and solver equations as
unitful ones; all their monomials simply land in the same dimension class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from nestynet_sr.sr_core.atoms import _enumerate_exponents
from nestynet_sr.sr_core.coefficient_units import (
    RationalGaugeSolution,
    monomial_dimension,
    normalize_dimension,
    solve_rational_coefficient_gauge,
)
from nestynet_sr.sr_core.units import Dim, sub_dim


@dataclass(frozen=True)
class UnitConsistentRationalSupport:
    """One structurally admissible anonymous-coefficient rational support."""

    numerator_exponents: tuple[tuple[int, ...], ...]
    denominator_exponents: tuple[tuple[int, ...], ...]
    degree_num: int
    degree_den: int
    certificate: RationalGaugeSolution

    @property
    def complexity(self) -> int:
        return len(self.numerator_exponents) + len(self.denominator_exponents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deg_num": int(self.degree_num),
            "deg_den": int(self.degree_den),
            "exps_num_override": [list(row) for row in self.numerator_exponents],
            "exps_den_override": [list(row) for row in self.denominator_exponents],
            "complexity": int(self.complexity),
            "coefficient_unit_certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True)
class RationalSupportPlan:
    """Finite support-planning result with machine-readable diagnostics."""

    supports: tuple[UnitConsistentRationalSupport, ...]
    raw_attempted: int
    unit_rejected: int
    deduplicated: int
    truncated_by_attempt_budget: bool
    max_attempts: int
    reason_counts: tuple[tuple[str, int], ...] = ()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "raw_attempted": int(self.raw_attempted),
            "unit_rejected": int(self.unit_rejected),
            "deduplicated": int(self.deduplicated),
            "emitted": int(len(self.supports)),
            "exhausted": True,
            "exhaustion_reason": (
                "attempt_budget_exhausted"
                if self.truncated_by_attempt_budget
                else "candidate_space_exhausted"
            ),
            "truncated_by_attempt_budget": bool(self.truncated_by_attempt_budget),
            "max_attempts": int(self.max_attempts),
            "reason_counts": {key: int(value) for key, value in self.reason_counts},
        }


def _normalized_dimensions(
    target_dim: Sequence[Any],
    input_dims: Sequence[Sequence[Any]],
) -> tuple[Dim, tuple[Dim, ...]]:
    target = normalize_dimension(target_dim, label="target_dim")
    if not target:
        raise ValueError("target_dim must have positive rank")
    dims = tuple(
        normalize_dimension(dim, rank=len(target), label=f"input_dims[{index}]")
        for index, dim in enumerate(input_dims)
    )
    if not dims:
        raise ValueError("at least one input dimension is required")
    return target, dims


def _dimension_classes(
    *,
    input_dims: Sequence[Dim],
    max_degree: int,
) -> dict[Dim, tuple[tuple[int, ...], ...]]:
    classes: dict[Dim, list[tuple[int, ...]]] = {}
    for exponent in _enumerate_exponents(len(input_dims), int(max_degree)):
        row = tuple(int(value) for value in exponent)
        dim = monomial_dimension(row, input_dims)
        classes.setdefault(dim, []).append(row)
    return {dim: tuple(rows) for dim, rows in classes.items()}


def _support_up_to(
    rows: Sequence[tuple[int, ...]],
    degree: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(row for row in rows if sum(row) <= int(degree))


def dense_rational_support_kwargs(
    *,
    n_inputs: int,
    degree_num: int,
    degree_den: int,
    min_total_num: int = 0,
    min_total_den: int = 0,
) -> dict[str, Any]:
    """Return explicit dense support metadata for a fixed rational recipe.

    Static templates sometimes intentionally request a full polynomial basis.
    Making that basis explicit preserves their numerical behavior while
    allowing the exact coefficient-unit checker to accept or reject the recipe
    without reconstructing an implicit span.
    """

    n_in = int(n_inputs)
    deg_num = int(degree_num)
    deg_den = int(degree_den)
    mt_num = int(min_total_num)
    mt_den = int(min_total_den)
    if n_in <= 0:
        raise ValueError("dense rational support requires at least one input")
    if deg_num < 0 or deg_den < 0:
        raise ValueError("dense rational support degrees must be non-negative")
    if mt_num < 0 or mt_num > deg_num or mt_den < 0 or mt_den > deg_den:
        raise ValueError("dense rational support min_total values are invalid")
    return {
        "deg_num": deg_num,
        "deg_den": deg_den,
        "exps_num_override": [
            list(row)
            for row in _enumerate_exponents(n_in, deg_num, min_total=mt_num)
        ],
        "exps_den_override": [
            list(row)
            for row in _enumerate_exponents(n_in, deg_den, min_total=mt_den)
        ],
    }


def plan_unit_consistent_rational_supports(
    *,
    target_dim: Sequence[Any],
    input_dims: Sequence[Sequence[Any]],
    max_deg_num: int,
    max_deg_den: int,
    coefficient_policy: str = "free_const_only",
    max_attempts: int = 2048,
) -> RationalSupportPlan:
    """Enumerate anonymous-coefficient rational supports admitted by units.

    ``raw_attempted`` counts unique support pairs presented to the exact gauge
    solver.  The finite search is bounded independently of how many supports
    survive, and the returned diagnostics distinguish true finite-space
    exhaustion from an attempt-budget cutoff.

    The current rational leaf coefficients are anonymous.  Consequently this
    producer only authorizes ``free_const_only``.  Declared unitful constants
    can be represented in the ``input_dims`` supplied to this pure planner,
    but using them as individual rational coefficients requires an explicit
    named-leaf representation rather than silently treating a numeric
    parameter as unitful.  Evaluation of constant-bearing NN input-expression
    ASTs is a separate producer-bridge concern.
    """

    attempt_limit = max(0, int(max_attempts))
    degree_num_limit = int(max_deg_num)
    degree_den_limit = int(max_deg_den)
    if degree_num_limit < 0 or degree_den_limit < 0:
        raise ValueError("rational support degree limits must be non-negative")

    policy = str(coefficient_policy or "free_const_only").strip().lower().replace("-", "_")
    if policy not in {"free_const_only", "dimensionless", "strict"}:
        raise ValueError(
            "anonymous rational support planning requires coefficient_policy='free_const_only'"
        )
    policy = "free_const_only"

    target, dims = _normalized_dimensions(target_dim, input_dims)
    num_classes = _dimension_classes(input_dims=dims, max_degree=degree_num_limit)
    den_classes = _dimension_classes(input_dims=dims, max_degree=degree_den_limit)

    raw_attempted = 0
    unit_rejected = 0
    deduplicated = 0
    truncated = False
    reason_counts: dict[str, int] = {}
    candidates: list[UnitConsistentRationalSupport] = []
    seen: set[
        tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
    ] = set()

    stop = False
    for numerator_dim, numerator_rows in num_classes.items():
        denominator_dim = sub_dim(numerator_dim, target)
        denominator_rows = den_classes.get(denominator_dim)
        if not denominator_rows:
            continue

        # Include cutoff zero explicitly.  This is essential when a zero-
        # dimensional class also contains nonconstant monomials: starting at
        # cutoff one would silently lose constant-only numerators such as the
        # ``1`` in ``1 / (1 + x)``.  The public degree label remains at least
        # one for compatibility; the override is the source of truth.
        for degree_num in range(0, degree_num_limit + 1):
            support_num = _support_up_to(numerator_rows, degree_num)
            if not support_num:
                continue
            for degree_den in range(0, degree_den_limit + 1):
                support_den = _support_up_to(denominator_rows, degree_den)
                if not support_den:
                    continue
                key = (support_num, support_den)
                if key in seen:
                    deduplicated += 1
                    continue
                seen.add(key)
                if raw_attempted >= attempt_limit:
                    truncated = True
                    stop = True
                    break
                raw_attempted += 1
                certificate = solve_rational_coefficient_gauge(
                    target_dim=target,
                    input_dims=dims,
                    numerator_exponents=support_num,
                    denominator_exponents=support_den,
                    coefficient_policy=policy,
                )
                if not certificate.ok:
                    unit_rejected += 1
                    reason_counts[certificate.code] = reason_counts.get(certificate.code, 0) + 1
                    continue
                candidates.append(
                    UnitConsistentRationalSupport(
                        numerator_exponents=support_num,
                        denominator_exponents=support_den,
                        degree_num=max(1, max(sum(row) for row in support_num)),
                        degree_den=max(1, max(sum(row) for row in support_den)),
                        certificate=certificate,
                    )
                )
            if stop:
                break
        if stop:
            break

    candidates.sort(
        key=lambda support: (
            support.complexity,
            support.degree_num + support.degree_den,
            support.degree_num,
            support.degree_den,
            support.numerator_exponents,
            support.denominator_exponents,
        )
    )
    return RationalSupportPlan(
        supports=tuple(candidates),
        raw_attempted=raw_attempted,
        unit_rejected=unit_rejected,
        deduplicated=deduplicated,
        truncated_by_attempt_budget=truncated,
        max_attempts=attempt_limit,
        reason_counts=tuple(sorted(reason_counts.items())),
    )


__all__ = [
    "dense_rational_support_kwargs",
    "RationalSupportPlan",
    "UnitConsistentRationalSupport",
    "plan_unit_consistent_rational_supports",
]
