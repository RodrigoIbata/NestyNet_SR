# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


SHARED_OPPORTUNITY_SCHEMA_VERSION = 2

SHARED_OPPORTUNITY_EVIDENCE_LEVELS: tuple[str, ...] = (
    "preview_only",
    "preview_support",
    "exact_known",
)

SHARED_OPPORTUNITY_MASK_FIELD_NAMES: tuple[str, ...] = (
    "opportunity_has_path",
    "opportunity_has_path_source",
    "opportunity_has_target_mode",
    "opportunity_has_current_best_child",
    "opportunity_route_valid_repair",
    "opportunity_route_valid_build",
    "opportunity_route_valid_hole",
    "opportunity_evidence_preview_only",
    "opportunity_evidence_preview_support",
    "opportunity_evidence_exact_known",
)

SHARED_OPPORTUNITY_WITNESS_FIELD_NAMES: tuple[str, ...] = (
    "witness_value_loss",
    "witness_grad_loss",
    "witness_d2_loss",
    "witness_diag_loss",
    "witness_physics_loss",
    "witness_energy_total",
    "witness_energy_delta_estimate",
)

SHARED_OPPORTUNITY_REALIZED_WITNESS_FIELD_NAMES: tuple[str, ...] = (
    "realized_witness_value_loss_before",
    "realized_witness_grad_loss_before",
    "realized_witness_d2_loss_before",
    "realized_witness_diag_loss_before",
    "realized_witness_physics_loss_before",
    "realized_witness_energy_total_before",
    "realized_witness_value_loss_after",
    "realized_witness_grad_loss_after",
    "realized_witness_d2_loss_after",
    "realized_witness_diag_loss_after",
    "realized_witness_physics_loss_after",
    "realized_witness_energy_total_after",
    "realized_witness_value_delta",
    "realized_witness_grad_delta",
    "realized_witness_d2_delta",
    "realized_witness_diag_delta",
    "realized_witness_physics_delta",
    "realized_witness_energy_delta",
)

_WITNESS_COMPONENT_KEYS: tuple[str, ...] = (
    "value_loss",
    "grad_loss",
    "d2_loss",
    "diag_loss",
    "physics_loss",
)


def _bool01(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _normalize_route_source(value: Any) -> str:
    route = str(value or "").strip().lower().replace("-", "_")
    if route in {"repair", "build", "hole"}:
        return route
    return ""


def _normalize_mode_name(value: Any) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode in ("id",):
        return "identity"
    if mode in ("fitbest", "fitted", "legacy"):
        return "full"
    return mode


def _normalize_path_source(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("critic"):
        return "critic"
    if token.startswith("inverse"):
        return "inverse"
    if token == "random":
        return "random"
    return token


def _normalize_evidence_level(value: Any) -> str:
    level = str(value or "").strip().lower().replace("-", "_")
    if level in SHARED_OPPORTUNITY_EVIDENCE_LEVELS:
        return level
    return ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _first_finite_float(*values: Any) -> float | None:
    for value in values:
        out = _safe_float_or_none(value)
        if out is not None:
            return float(out)
    return None


def _normalize_witness_energy_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    value_loss = _first_finite_float(
        row.get("witness_value_loss", None),
        row.get("value_local_probe_mse", None),
        row.get("best_preview_probe_mse", None),
        row.get("current_best_child_eff_mse", None),
        row.get("hole_best_shortlist_eff_mse", None),
        row.get("hole_best_exact_eff_mse", None),
        row.get("preview_solvability", None),
    )
    grad_loss = _first_finite_float(
        row.get("witness_grad_loss", None),
    )
    d2_loss = _first_finite_float(
        row.get("witness_d2_loss", None),
    )
    diag_loss = _first_finite_float(
        row.get("witness_diag_loss", None),
    )
    physics_loss = _first_finite_float(
        row.get("witness_physics_loss", None),
    )
    total = _first_finite_float(
        row.get("witness_energy_total", None),
    )
    if total is None:
        pieces = [value_loss, grad_loss, d2_loss, diag_loss, physics_loss]
        finite_pieces = [float(v) for v in pieces if v is not None]
        if finite_pieces:
            total = float(sum(finite_pieces))
    delta_estimate = _first_finite_float(
        row.get("witness_energy_delta_estimate", None),
        row.get("best_tuple_allocation_estimate", None),
        row.get("best_tuple_utility_estimate", None),
        row.get("predicted_value", None),
    )
    return {
        "witness_value_loss": value_loss,
        "witness_grad_loss": grad_loss,
        "witness_d2_loss": d2_loss,
        "witness_diag_loss": diag_loss,
        "witness_physics_loss": physics_loss,
        "witness_energy_total": total,
        "witness_energy_delta_estimate": delta_estimate,
    }


def _normalize_realized_witness_energy_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    before_fields = {
        key: _first_finite_float(row.get(f"realized_witness_{key}_before", None))
        for key in _WITNESS_COMPONENT_KEYS
    }
    after_fields = {
        key: _first_finite_float(
            row.get(f"realized_witness_{key}_after", None),
            row.get(f"realized_witness_{key}", None),
        )
        for key in _WITNESS_COMPONENT_KEYS
    }
    before_total = _first_finite_float(row.get("realized_witness_energy_total_before", None))
    if before_total is None:
        finite_before = [float(value) for value in before_fields.values() if value is not None]
        if finite_before:
            before_total = float(sum(finite_before))
    after_total = _first_finite_float(
        row.get("realized_witness_energy_total_after", None),
        row.get("realized_witness_energy_total", None),
    )
    if after_total is None:
        finite_after = [float(value) for value in after_fields.values() if value is not None]
        if finite_after:
            after_total = float(sum(finite_after))
    delta_fields: dict[str, float | None] = {}
    for key in _WITNESS_COMPONENT_KEYS:
        delta_value = _first_finite_float(row.get(f"realized_witness_{key}_delta", None))
        if delta_value is None and before_fields.get(key, None) is not None and after_fields.get(key, None) is not None:
            delta_value = float(before_fields[key]) - float(after_fields[key])
        delta_fields[key] = delta_value
    total_delta = _first_finite_float(row.get("realized_witness_energy_delta", None))
    if total_delta is None and before_total is not None and after_total is not None:
        total_delta = float(before_total) - float(after_total)
    return {
        "realized_witness_value_loss_before": before_fields["value_loss"],
        "realized_witness_grad_loss_before": before_fields["grad_loss"],
        "realized_witness_d2_loss_before": before_fields["d2_loss"],
        "realized_witness_diag_loss_before": before_fields["diag_loss"],
        "realized_witness_physics_loss_before": before_fields["physics_loss"],
        "realized_witness_energy_total_before": before_total,
        "realized_witness_value_loss_after": after_fields["value_loss"],
        "realized_witness_grad_loss_after": after_fields["grad_loss"],
        "realized_witness_d2_loss_after": after_fields["d2_loss"],
        "realized_witness_diag_loss_after": after_fields["diag_loss"],
        "realized_witness_physics_loss_after": after_fields["physics_loss"],
        "realized_witness_energy_total_after": after_total,
        "realized_witness_value_delta": delta_fields["value_loss"],
        "realized_witness_grad_delta": delta_fields["grad_loss"],
        "realized_witness_d2_delta": delta_fields["d2_loss"],
        "realized_witness_diag_delta": delta_fields["diag_loss"],
        "realized_witness_physics_delta": delta_fields["physics_loss"],
        "realized_witness_energy_delta": total_delta,
    }


def normalize_witness_energy_fields(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row_map = row if isinstance(row, Mapping) else {}
    return _normalize_witness_energy_fields(row_map)


def normalize_realized_witness_energy_fields(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row_map = row if isinstance(row, Mapping) else {}
    return _normalize_realized_witness_energy_fields(row_map)


def _stable_id(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def _derive_evidence_level(row: Mapping[str, Any]) -> str:
    explicit = _normalize_evidence_level(row.get("evidence_level", ""))
    if explicit:
        return explicit
    exact_observed = _safe_int(row.get("exact_child_observed_count", row.get("budget_exact_spent", 0)), 0)
    if exact_observed > 0 or bool(row.get("exact_child_score_observed", False)):
        return "exact_known"
    candidate_count = _safe_int(row.get("candidate_count_observed", 0), 0)
    if candidate_count > 1:
        return "preview_support"
    return "preview_only"


@dataclass(frozen=True)
class SharedOpportunityRecord:
    route_source: str
    opportunity_type: str
    opportunity_id: str
    decision_id: str
    decision_context_id: str
    beam_id: str
    parent_key: str
    parent_expr: str
    action: str = ""
    path: tuple[int, ...] = ()
    path_source: str = ""
    target_mode: str = ""
    method_name: str = ""
    subroute: str = ""
    evidence_level: str = "preview_only"
    budget_exact_spent: int = 0
    budget_remaining: int = 0
    budget_widen_spent: int = 0
    budget_micro_spent: int = 0
    current_best_child_expr: str = ""
    current_best_child_eff_mse: float | None = None
    global_best_eff_mse: float | None = None
    best_alt_route_eff_mse: float | None = None
    candidate_count_observed: int = 0
    candidate_count_unique: int = 0
    observed_wall_seconds: float | None = None
    observed_exact_evals: int = 0
    observed_preview_evals: int = 0
    observed_micro_tokens: int = 0
    observed_widen_tokens: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SHARED_OPPORTUNITY_SCHEMA_VERSION

    @property
    def masks(self) -> dict[str, float]:
        evidence = _normalize_evidence_level(self.evidence_level) or "preview_only"
        return {
            "opportunity_has_path": _bool01(bool(self.path)),
            "opportunity_has_path_source": _bool01(bool(self.path_source)),
            "opportunity_has_target_mode": _bool01(bool(self.target_mode)),
            "opportunity_has_current_best_child": _bool01(bool(self.current_best_child_expr)),
            "opportunity_route_valid_repair": _bool01(self.route_source == "repair"),
            "opportunity_route_valid_build": _bool01(self.route_source == "build"),
            "opportunity_route_valid_hole": _bool01(self.route_source == "hole"),
            "opportunity_evidence_preview_only": _bool01(evidence == "preview_only"),
            "opportunity_evidence_preview_support": _bool01(evidence == "preview_support"),
            "opportunity_evidence_exact_known": _bool01(evidence == "exact_known"),
        }

    def to_row(self) -> dict[str, Any]:
        row = dict(self.payload)
        row.update({
            "shared_opportunity_schema_version": int(self.schema_version),
            "route_source": str(self.route_source),
            "opportunity_type": str(self.opportunity_type),
            "opportunity_id": str(self.opportunity_id),
            "decision_id": str(self.decision_id),
            "decision_context_id": str(self.decision_context_id),
            "beam_id": str(self.beam_id),
            "parent_key": str(self.parent_key),
            "parent_expr": str(self.parent_expr),
            "action": str(self.action),
            "path": [int(v) for v in self.path],
            "path_source": str(self.path_source),
            "target_mode": str(self.target_mode),
            "method_name": str(self.method_name),
            "subroute": str(self.subroute),
            "evidence_level": str(self.evidence_level),
            "budget_exact_spent": int(self.budget_exact_spent),
            "budget_remaining": int(self.budget_remaining),
            "budget_widen_spent": int(self.budget_widen_spent),
            "budget_micro_spent": int(self.budget_micro_spent),
            "current_best_child_expr": str(self.current_best_child_expr),
            "current_best_child_eff_mse": (
                None if self.current_best_child_eff_mse is None else float(self.current_best_child_eff_mse)
            ),
            "global_best_eff_mse": (
                None if self.global_best_eff_mse is None else float(self.global_best_eff_mse)
            ),
            "best_alt_route_eff_mse": (
                None if self.best_alt_route_eff_mse is None else float(self.best_alt_route_eff_mse)
            ),
            "candidate_count_observed": int(self.candidate_count_observed),
            "candidate_count_unique": int(self.candidate_count_unique),
            "observed_wall_seconds": (
                None if self.observed_wall_seconds is None else float(self.observed_wall_seconds)
            ),
            "observed_exact_evals": int(self.observed_exact_evals),
            "observed_preview_evals": int(self.observed_preview_evals),
            "observed_micro_tokens": int(self.observed_micro_tokens),
            "observed_widen_tokens": int(self.observed_widen_tokens),
        })
        row.update(self.masks)
        return row


def coerce_shared_opportunity_record(
    row: Mapping[str, Any] | SharedOpportunityRecord | None,
    *,
    route_source: str = "",
) -> SharedOpportunityRecord:
    if isinstance(row, SharedOpportunityRecord):
        return row
    row_map = row if isinstance(row, Mapping) else {}
    route_name = _normalize_route_source(route_source or row_map.get("route_source", ""))
    action_name = str(row_map.get("action", "") or "").strip().lower().replace("-", "_")
    try:
        path = tuple(int(v) for v in (row_map.get("path", ()) or ()))
    except Exception:
        path = ()
    target_mode = _normalize_mode_name(row_map.get("target_mode", ""))
    path_source = _normalize_path_source(row_map.get("path_source", ""))
    decision_id = str(row_map.get("decision_id", "") or row_map.get("slate_id", "") or "").strip()
    decision_context_id = str(row_map.get("decision_context_id", "") or decision_id).strip()
    beam_id = str(row_map.get("beam_id", "") or "").strip()
    if not beam_id:
        beam_id = _stable_id(
            route_name,
            decision_id,
            action_name,
            path,
            target_mode,
            row_map.get("beam_rank", ""),
            row_map.get("opportunity_type", ""),
        )
    opportunity_id = str(row_map.get("opportunity_id", "") or "").strip()
    if not opportunity_id:
        opportunity_id = _stable_id(route_name, decision_id, beam_id, action_name, path, target_mode)
    parent_expr = str(row_map.get("parent_expr", "") or "").strip()
    current_best_child_expr = str(
        row_map.get("current_best_child_expr", "")
        or row_map.get("best_preview_child_expr", "")
        or row_map.get("child_expr", "")
        or ""
    ).strip()
    payload = dict(row_map)
    payload.update(_normalize_witness_energy_fields(row_map))
    payload.update(_normalize_realized_witness_energy_fields(row_map))
    return SharedOpportunityRecord(
        route_source=route_name,
        opportunity_type=str(row_map.get("opportunity_type", "") or "").strip().lower().replace("-", "_"),
        opportunity_id=opportunity_id,
        decision_id=decision_id,
        decision_context_id=decision_context_id,
        beam_id=beam_id,
        parent_key=str(row_map.get("parent_key", "") or "").strip(),
        parent_expr=parent_expr,
        action=action_name,
        path=path,
        path_source=path_source,
        target_mode=target_mode if route_name in {"repair", "hole"} else "",
        method_name=str(row_map.get("method_name", "") or "").strip().lower().replace("-", "_"),
        subroute=str(row_map.get("subroute", "") or row_map.get("spec_kind", "") or "").strip().lower().replace("-", "_"),
        evidence_level=_derive_evidence_level(row_map),
        budget_exact_spent=_safe_int(row_map.get("budget_exact_spent", 0), 0),
        budget_remaining=_safe_int(row_map.get("budget_remaining", 0), 0),
        budget_widen_spent=_safe_int(row_map.get("budget_widen_spent", 0), 0),
        budget_micro_spent=_safe_int(row_map.get("budget_micro_spent", 0), 0),
        current_best_child_expr=current_best_child_expr,
        current_best_child_eff_mse=_safe_float_or_none(row_map.get("current_best_child_eff_mse", None)),
        global_best_eff_mse=_safe_float_or_none(row_map.get("global_best_eff_mse", None)),
        best_alt_route_eff_mse=_safe_float_or_none(row_map.get("best_alt_route_eff_mse", None)),
        candidate_count_observed=_safe_int(row_map.get("candidate_count_observed", 0), 0),
        candidate_count_unique=_safe_int(row_map.get("candidate_count_unique", 0), 0),
        observed_wall_seconds=_safe_float_or_none(row_map.get("observed_wall_seconds", None)),
        observed_exact_evals=_safe_int(row_map.get("observed_exact_evals", 0), 0),
        observed_preview_evals=_safe_int(row_map.get("observed_preview_evals", 0), 0),
        observed_micro_tokens=_safe_int(row_map.get("observed_micro_tokens", 0), 0),
        observed_widen_tokens=_safe_int(row_map.get("observed_widen_tokens", 0), 0),
        payload=payload,
        schema_version=_safe_int(
            row_map.get("shared_opportunity_schema_version", SHARED_OPPORTUNITY_SCHEMA_VERSION),
            SHARED_OPPORTUNITY_SCHEMA_VERSION,
        ),
    )


def shared_opportunity_row_dict(
    row: Mapping[str, Any] | SharedOpportunityRecord | None,
    *,
    route_source: str = "",
) -> dict[str, Any]:
    return coerce_shared_opportunity_record(row, route_source=route_source).to_row()


__all__ = [
    "SHARED_OPPORTUNITY_SCHEMA_VERSION",
    "SHARED_OPPORTUNITY_EVIDENCE_LEVELS",
    "SHARED_OPPORTUNITY_MASK_FIELD_NAMES",
    "SHARED_OPPORTUNITY_WITNESS_FIELD_NAMES",
    "SHARED_OPPORTUNITY_REALIZED_WITNESS_FIELD_NAMES",
    "SharedOpportunityRecord",
    "coerce_shared_opportunity_record",
    "normalize_realized_witness_energy_fields",
    "normalize_witness_energy_fields",
    "shared_opportunity_row_dict",
]
