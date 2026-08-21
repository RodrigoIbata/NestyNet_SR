# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Unit-state contract for a certified inner coordinate and its outer map.

An inner coordinate ``z(x)`` is not itself a candidate for ``y``.  Its AST must
be internally dimensionally consistent, but ``dim(z) == dim(y)`` is deliberately
deferred until a concrete relation ``y = g(z)`` has been assembled.  This module
keeps that distinction explicit and independent of the proposal's route name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from math import comb
from typing import Any, Mapping, Sequence


CANDIDATE_ROLE_INNER_COORDINATE = "inner_coordinate"
CANDIDATE_ROLE_TARGET_EXPRESSION = "target_expression"

CARRIER_INTERNAL_UNITS_INVALID = "carrier_internal_units_invalid"
CARRIER_UNITS_DEFERRED = "carrier_units_deferred_to_outer_map"
OUTER_MAP_UNITS_VALID = "outer_map_units_valid"
OUTER_MAP_UNITS_INVALID = "outer_map_units_invalid"
STAGEA_BUCKINGHAM_DEFERRED = "stageA_buckingham_deferred_for_carrier"


class UnitDecision(str, Enum):
    """Three states needed while a carrier is waiting for its consumer."""

    VALID = "VALID"
    INVALID = "INVALID"
    DEFERRED_UNTIL_OUTER_MAP = "DEFERRED_UNTIL_OUTER_MAP"


@dataclass(frozen=True)
class CarrierUnitContext:
    """Unit-relevant facts attached to one proposal/consumer transaction."""

    role: str
    carrier_dim: tuple[float, ...] | None
    target_dim: tuple[float, ...] | None
    source: str = ""
    certified: bool = False


@dataclass(frozen=True)
class CarrierUnitResult:
    """Decision plus report-ready evidence."""

    decision: UnitDecision
    diagnostic: str
    reason: str
    carrier_dim: tuple[float, ...] | None
    target_dim: tuple[float, ...] | None
    map_family: str = ""
    assembled_dim: tuple[float, ...] | None = None

    @property
    def ok(self) -> bool:
        return self.decision is UnitDecision.VALID

    def to_metadata(self, *, context: CarrierUnitContext | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "decision": self.decision.value,
            "diagnostic": self.diagnostic,
            "reason": self.reason,
            "carrier_dim": _dim_payload(self.carrier_dim),
            "target_dim": _dim_payload(self.target_dim),
            "map_family": self.map_family,
            "assembled_dim": _dim_payload(self.assembled_dim),
        }
        if context is not None:
            row.update(
                {
                    "candidate_role": str(context.role),
                    "candidate_source": str(context.source),
                    "carrier_certified": bool(context.certified),
                }
            )
        return row


def mark_inner_coordinate_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    source: str,
    certified: bool = True,
) -> dict[str, Any]:
    """Return metadata carrying the explicit inner-coordinate contract."""

    row = dict(metadata) if isinstance(metadata, Mapping) else {}
    row["candidate_role"] = CANDIDATE_ROLE_INNER_COORDINATE
    row["candidate_source"] = str(source)
    row["carrier_certified"] = bool(certified)
    row["carrier_unit_decision"] = (
        UnitDecision.DEFERRED_UNTIL_OUTER_MAP.value
        if certified
        else UnitDecision.INVALID.value
    )
    return row


def is_certified_inner_coordinate(metadata: Mapping[str, Any] | None) -> bool:
    """Whether metadata explicitly authorizes delayed target-unit checking."""

    if not isinstance(metadata, Mapping):
        return False
    return bool(
        str(metadata.get("candidate_role", "") or "") == CANDIDATE_ROLE_INNER_COORDINATE
        and metadata.get("carrier_certified", False) is True
    )


def mark_stagea_buckingham_deferred(metadata: Mapping[str, Any] | None) -> bool:
    """Mark and authorize only a certified carrier's Stage-A deferral."""

    if not is_certified_inner_coordinate(metadata):
        return False
    if isinstance(metadata, dict):
        metadata["stageA_buckingham_decision"] = STAGEA_BUCKINGHAM_DEFERRED
        metadata[STAGEA_BUCKINGHAM_DEFERRED] = True
    return True


def stagea_provisional_unit_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    carrier_dim: Sequence[Any] | None,
    target_dim: Sequence[Any] | None,
) -> dict[str, Any] | None:
    """Build the marker retained by a provisional Stage-A ``NN[z]`` atom."""

    if not is_certified_inner_coordinate(metadata):
        return None
    context = context_from_metadata(
        metadata,
        carrier_dim=carrier_dim,
        target_dim=target_dim,
    )
    result = precheck_carrier_units(context)
    if result.decision is not UnitDecision.DEFERRED_UNTIL_OUTER_MAP:
        return None
    row = result.to_metadata(context=context)
    row.update(
        {
            "outer_map_pending": True,
            "stageA_buckingham_decision": STAGEA_BUCKINGHAM_DEFERRED,
        }
    )
    return row


def context_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    carrier_dim: Sequence[Any] | None,
    target_dim: Sequence[Any] | None,
) -> CarrierUnitContext:
    """Build a normalized context without inferring role from route/source text."""

    row = dict(metadata) if isinstance(metadata, Mapping) else {}
    return CarrierUnitContext(
        role=str(row.get("candidate_role", CANDIDATE_ROLE_TARGET_EXPRESSION) or ""),
        carrier_dim=_coerce_dim(carrier_dim),
        target_dim=_coerce_dim(target_dim),
        source=str(row.get("candidate_source", "") or ""),
        certified=bool(row.get("carrier_certified", False)),
    )


def precheck_carrier_units(context: CarrierUnitContext) -> CarrierUnitResult:
    """Validate the carrier itself and defer only the carrier-to-target relation."""

    if context.carrier_dim is None:
        return CarrierUnitResult(
            decision=UnitDecision.INVALID,
            diagnostic=CARRIER_INTERNAL_UNITS_INVALID,
            reason="carrier dimension is unknown or its AST is internally inconsistent",
            carrier_dim=None,
            target_dim=context.target_dim,
        )
    if not (
        context.role == CANDIDATE_ROLE_INNER_COORDINATE and bool(context.certified)
    ):
        return CarrierUnitResult(
            decision=UnitDecision.INVALID,
            diagnostic=CARRIER_INTERNAL_UNITS_INVALID,
            reason="target-unit deferral requires an explicitly certified inner coordinate",
            carrier_dim=context.carrier_dim,
            target_dim=context.target_dim,
        )
    return CarrierUnitResult(
        decision=UnitDecision.DEFERRED_UNTIL_OUTER_MAP,
        diagnostic=CARRIER_UNITS_DEFERRED,
        reason="carrier is internally valid; target compatibility awaits the outer map",
        carrier_dim=context.carrier_dim,
        target_dim=context.target_dim,
    )


def validate_outer_map_units(
    context: CarrierUnitContext,
    mapping: Mapping[str, Any] | None,
    *,
    linear_head_term_dims: Sequence[Sequence[Any] | None] | None = None,
) -> CarrierUnitResult:
    """Validate the dimensional action of a fitted outer mapping.

    Numerical mapping parameters are dimensionless unless a future caller
    supplies a declared parameter-unit policy.  Consequently, dimensional
    powers and raw polynomial monomials are supported explicitly, while an
    unknown family or an extra fitted linear head fails closed.
    """

    mapping_is_valid = mapping is None or isinstance(mapping, Mapping)
    try:
        mapping_row = dict(mapping or {}) if mapping_is_valid else {}
    except Exception:
        mapping_is_valid = False
        mapping_row = {}
    family = str(mapping_row.get("kind", "") or "identity").strip().lower()
    invalid = lambda reason, assembled=None: CarrierUnitResult(
        decision=UnitDecision.INVALID,
        diagnostic=OUTER_MAP_UNITS_INVALID,
        reason=str(reason),
        carrier_dim=context.carrier_dim,
        target_dim=context.target_dim,
        map_family=family,
        assembled_dim=assembled,
    )
    if not mapping_is_valid:
        return invalid("outer-map metadata is malformed")
    if context.carrier_dim is None:
        return invalid("carrier dimension is unresolved")
    if context.target_dim is None:
        return invalid("target dimension is unresolved")
    if len(context.carrier_dim) != len(context.target_dim):
        return invalid("carrier and target dimensions use different bases")
    if not (
        context.role == CANDIDATE_ROLE_INNER_COORDINATE and bool(context.certified)
    ):
        return invalid("outer-map validation requires an explicitly certified inner coordinate")

    def valid(reason, assembled):
        head_reason = _linear_head_units_rejection(
            mapping_row,
            target_dim=context.target_dim,
            term_dims=linear_head_term_dims,
        )
        if head_reason is not None:
            return invalid(head_reason, assembled)
        return CarrierUnitResult(
            decision=UnitDecision.VALID,
            diagnostic=OUTER_MAP_UNITS_VALID,
            reason=str(reason),
            carrier_dim=context.carrier_dim,
            target_dim=context.target_dim,
            map_family=family,
            assembled_dim=assembled,
        )

    if family in ("", "identity", "basis_state_native"):
        assembled = context.carrier_dim
        if _dims_equal(assembled, context.target_dim):
            return valid("identity map preserves the target dimension", assembled)
        return invalid("identity map does not convert the carrier to the target dimension", assembled)

    if family == "power":
        try:
            exponent = float(mapping_row["b"])
        except Exception:
            return invalid("power map has no finite exponent")
        if not math.isfinite(exponent):
            return invalid("power map has no finite exponent")
        assembled = _scale_dim(context.carrier_dim, exponent)
        if _dims_equal(assembled, context.target_dim):
            return valid(f"power map applies dimensional exponent {exponent:.12g}", assembled)
        return invalid(
            f"power exponent {exponent:.12g} does not produce the target dimension",
            assembled,
        )

    if family == "poly":
        raw_coeffs = _raw_poly_coefficients(
            mapping_row.get("coeffs"),
            mu=mapping_row.get("mu", 0.0),
            std=mapping_row.get("std", 1.0),
        )
        if raw_coeffs is None:
            return invalid("polynomial mapping parameters are malformed")
        assembled, reason = _single_polynomial_dimension(
            raw_coeffs,
            context.carrier_dim,
        )
        if assembled is None:
            return invalid(reason)
        if _dims_equal(assembled, context.target_dim):
            return valid(reason, assembled)
        return invalid(f"{reason}; assembled dimension does not match target", assembled)

    if family == "pade":
        numer = _raw_poly_coefficients(
            mapping_row.get("numer"),
            mu=mapping_row.get("mu", 0.0),
            std=mapping_row.get("std", 1.0),
        )
        denom = _raw_poly_coefficients(
            mapping_row.get("denom"),
            mu=mapping_row.get("mu", 0.0),
            std=mapping_row.get("std", 1.0),
        )
        if numer is None or denom is None:
            return invalid("Padé mapping parameters are malformed")
        numer_dim, numer_reason = _single_polynomial_dimension(numer, context.carrier_dim)
        denom_dim, denom_reason = _single_polynomial_dimension(denom, context.carrier_dim)
        if numer_dim is None:
            return invalid(f"Padé numerator is invalid: {numer_reason}")
        if denom_dim is None:
            return invalid(f"Padé denominator is invalid: {denom_reason}")
        assembled = tuple(a - b for a, b in zip(numer_dim, denom_dim))
        if _dims_equal(assembled, context.target_dim):
            return valid("Padé numerator/denominator have resolved dimensions", assembled)
        return invalid("Padé mapping does not produce the target dimension", assembled)

    if family in ("sine", "exp"):
        dimless = tuple(0.0 for _ in context.carrier_dim)
        if not _dims_equal(context.carrier_dim, dimless):
            return invalid(f"{family} mapping requires a dimensionless carrier argument")
        if not _dims_equal(context.target_dim, dimless):
            return invalid(
                f"{family} mapping has no declared unitful amplitude/offset parameters",
                dimless,
            )
        return valid(f"{family} maps a dimensionless carrier to a dimensionless target", dimless)

    return invalid(f"outer-map family {family!r} has no dimensional action contract")


def _linear_head_units_rejection(
    mapping: Mapping[str, Any],
    *,
    target_dim: tuple[float, ...],
    term_dims: Sequence[Sequence[Any] | None] | None,
) -> str | None:
    if "_lin_head" not in mapping:
        return None
    head = mapping["_lin_head"]
    if not isinstance(head, Mapping):
        return "auxiliary linear-head parameters are malformed"

    try:
        coeffs = [float(value) for value in list(head.get("coeffs") or ())]
        terms = list(head.get("terms") or ())
    except Exception:
        return "auxiliary linear-head parameters are malformed"
    if not all(math.isfinite(value) for value in coeffs):
        return "auxiliary linear-head parameters are nonfinite"
    if len(coeffs) != len(terms) + 1:
        return "auxiliary linear-head parameters are malformed"

    tolerance = 1.0e-12
    dimless = tuple(0.0 for _ in target_dim)
    if abs(coeffs[0]) > tolerance and not _dims_equal(target_dim, dimless):
        return "auxiliary linear-head bias would require an undeclared unitful parameter"

    normalized_term_dims = list(term_dims or ())
    if len(normalized_term_dims) != len(terms):
        if any(abs(value) > tolerance for value in coeffs[1:]):
            return "auxiliary linear-head term dimensions are unresolved"
        return None
    for coeff, raw_dim in zip(coeffs[1:], normalized_term_dims):
        if abs(coeff) <= tolerance:
            continue
        dim = _coerce_dim(raw_dim)
        if dim is None:
            return "auxiliary linear-head term dimension is unresolved"
        if not _dims_equal(dim, target_dim):
            return "auxiliary linear head adds a term with non-target dimension"
    return None


def _single_polynomial_dimension(
    raw_coeffs: Sequence[float],
    carrier_dim: tuple[float, ...],
) -> tuple[tuple[float, ...] | None, str]:
    significant = _significant_degrees(raw_coeffs)
    if not significant:
        return None, "zero polynomial has no resolved output dimension"
    dims = [_scale_dim(carrier_dim, float(degree)) for degree in significant]
    first = dims[0]
    if any(not _dims_equal(first, other) for other in dims[1:]):
        return None, "polynomial adds terms with incompatible dimensions"
    degrees = ",".join(str(v) for v in significant)
    return first, f"nonzero raw polynomial degree(s) {degrees} share one dimension"


def _raw_poly_coefficients(
    coeffs: Any,
    *,
    mu: Any,
    std: Any,
) -> tuple[float, ...] | None:
    try:
        if hasattr(coeffs, "detach"):
            coeffs = coeffs.detach().cpu().tolist()
        elif hasattr(coeffs, "tolist"):
            coeffs = coeffs.tolist()
        values = [float(value) for value in list(coeffs or ())]
        mu_f = float(mu)
        std_f = float(std)
    except Exception:
        return None
    if (
        not values
        or not all(math.isfinite(value) for value in values)
        or not math.isfinite(mu_f)
        or not math.isfinite(std_f)
        or abs(std_f) <= 1.0e-30
    ):
        return None
    raw = [0.0] * len(values)
    for degree, coeff in enumerate(values):
        denom = std_f ** degree
        if not math.isfinite(denom) or abs(denom) <= 1.0e-300:
            return None
        for raw_degree in range(degree + 1):
            raw[raw_degree] += (
                coeff
                * float(comb(degree, raw_degree))
                * ((-mu_f) ** (degree - raw_degree))
                / denom
            )
    return tuple(float(value) for value in raw)


def _significant_degrees(coeffs: Sequence[float]) -> tuple[int, ...]:
    scale = max((abs(float(value)) for value in coeffs), default=0.0)
    tolerance = max(1.0e-12, scale * 1.0e-9)
    return tuple(
        int(degree)
        for degree, value in enumerate(coeffs)
        if abs(float(value)) > tolerance
    )


def _coerce_dim(dim: Sequence[Any] | None) -> tuple[float, ...] | None:
    if dim is None:
        return None
    try:
        values = tuple(float(value) for value in dim)
    except Exception:
        return None
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return values


def _scale_dim(dim: Sequence[float], exponent: float) -> tuple[float, ...]:
    return tuple(float(value) * float(exponent) for value in dim)


def _dims_equal(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    *,
    tol: float = 1.0e-8,
) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(left, right))


def _dim_payload(dim: Sequence[float] | None) -> list[float] | None:
    if dim is None:
        return None
    return [float(value) for value in dim]


__all__ = [
    "CANDIDATE_ROLE_INNER_COORDINATE",
    "CANDIDATE_ROLE_TARGET_EXPRESSION",
    "CARRIER_INTERNAL_UNITS_INVALID",
    "CARRIER_UNITS_DEFERRED",
    "OUTER_MAP_UNITS_INVALID",
    "OUTER_MAP_UNITS_VALID",
    "STAGEA_BUCKINGHAM_DEFERRED",
    "CarrierUnitContext",
    "CarrierUnitResult",
    "UnitDecision",
    "context_from_metadata",
    "is_certified_inner_coordinate",
    "mark_inner_coordinate_metadata",
    "mark_stagea_buckingham_deferred",
    "precheck_carrier_units",
    "stagea_provisional_unit_metadata",
    "validate_outer_map_units",
]
