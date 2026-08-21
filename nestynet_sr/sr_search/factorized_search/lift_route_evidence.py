# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from typing import Any, Mapping

from .subproblem_active_vars import normalize_active_vars
from .subproblem_spec import (
    deserialize_subproblem_spec,
    extract_family_regime_metadata,
    serialize_family_evidence,
)
from .subproblem_tests import build_expanded_family_evidence_bundle


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _evidence_status(payload: Mapping[str, Any] | None) -> str:
    data = dict(payload or {})
    meta = dict(data.get("metadata", {}) or {})
    hard = dict(data.get("hard_constraints", {}) or {})
    status = str(meta.get("status", hard.get("status", "")) or "")
    return str(status)


def _family_score(payload: Mapping[str, Any] | None, family_name: str) -> float:
    data = dict(payload or {})
    scores = dict(data.get("family_scores", {}) or {})
    value = _safe_float(scores.get(str(family_name), 0.0))
    return 0.0 if value is None else float(max(0.0, value))


def _has_explicit_constant_lift_task(metadata: Mapping[str, Any] | None) -> bool:
    merged = dict(metadata or {})
    task = dict(merged.get("constant_lift_task", {}) or {})
    if isinstance(task.get("values_by_regime", None), Mapping) and dict(task.get("values_by_regime", {}) or {}):
        return True
    values = merged.get("constant_lift_values_by_regime", None)
    return isinstance(values, Mapping) and bool(dict(values or {}))


def build_local_lift_route_context(spec_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    spec = deserialize_subproblem_spec(spec_payload)
    if spec is None or spec.witness is None:
        return {}
    witness = spec.witness
    if witness.x_fit is None or witness.t_fit is None or witness.x_probe is None or witness.t_probe is None:
        return {}
    nvars = int(witness.x_fit.shape[1]) if getattr(witness.x_fit, "ndim", 0) >= 2 else 1
    active_vars = normalize_active_vars(tuple(spec.active_vars or ()), nvars=max(1, int(nvars)))
    if not active_vars and int(nvars) > 0:
        active_vars = tuple(range(int(nvars)))
    regime_metadata = extract_family_regime_metadata(
        dict(spec.metadata or {}),
        dict(witness.diagnostics or {}),
    )
    evidence_bundle = build_expanded_family_evidence_bundle(
        x_fit=witness.x_fit,
        t_fit=witness.t_fit,
        x_probe=witness.x_probe,
        t_probe=witness.t_probe,
        grad_fit=witness.grad_fit,
        grad_probe=witness.grad_probe,
        d2_fit=witness.d2_fit,
        d2_probe=witness.d2_probe,
        target_dim=spec.target_dim,
        active_vars=active_vars,
        wrappers_left=int(spec.wrappers_left),
        recursion_level=int(spec.recursion_level),
        direction=str(spec.direction or ""),
        target_mode=str(spec.target_mode or ""),
        target_mapping_kind=str(spec.target_mapping_kind or ""),
        regime_metadata=regime_metadata,
    )
    serialized = {
        str(name): serialize_family_evidence(evidence)
        for name, evidence in sorted(dict(evidence_bundle or {}).items())
        if serialize_family_evidence(evidence) is not None
    }

    coordinate_payload = dict(serialized.get("coordinate_invariant", {}) or {})
    coordinate_hard = dict(coordinate_payload.get("hard_constraints", {}) or {})
    low_rank_payload = dict(serialized.get("low_rank_dependence", {}) or {})
    low_rank_hard = dict(low_rank_payload.get("hard_constraints", {}) or {})
    regime_payload = dict(serialized.get("regime_lift", {}) or {})
    regime_hard = dict(regime_payload.get("hard_constraints", {}) or {})

    coordinate_score = max(
        _family_score(coordinate_payload, "coordinate_invariant"),
        0.75 * _family_score(low_rank_payload, "low_rank_dependence"),
    )
    coordinate_status = _evidence_status(coordinate_payload) or _evidence_status(low_rank_payload)
    coordinate_seed_nodes = list(coordinate_payload.get("seed_nodes", []) or [])
    coordinate_preferred = bool(
        coordinate_seed_nodes
        or str(coordinate_status) == "single_index_like"
        or str(low_rank_hard.get("status", "")) == "strong_single_index"
    )
    coordinate_reason_family = "coordinate_invariant" if coordinate_seed_nodes else (
        "coordinate_invariant" if str(coordinate_status) == "single_index_like" else (
            "low_rank_dependence" if float(coordinate_score) > 0.0 else ""
        )
    )

    regime_score = _family_score(regime_payload, "regime_lift")
    regime_status = _evidence_status(regime_payload)
    top_constant_cv = _safe_float(regime_hard.get("top_constant_cv", None))
    trigger_mean_cv = _safe_float(regime_hard.get("trigger_mean_cv", None))
    if (
        str(regime_status) != "drifting_constants"
        and top_constant_cv is not None
        and trigger_mean_cv is not None
        and float(top_constant_cv) > float(trigger_mean_cv)
    ):
        regime_status = "drifting_constants"
    explicit_constant_task = _has_explicit_constant_lift_task(dict(spec.metadata or {}))
    regime_preferred = bool(explicit_constant_task or str(regime_status) == "drifting_constants")

    return {
        "problem_id": str(spec.problem_id or ""),
        "direction": str(spec.direction or ""),
        "active_vars": [int(v) for v in tuple(active_vars or ())],
        "expanded_family_evidence": serialized,
        "coordinate_lift": {
            "preferred": bool(coordinate_preferred),
            "score": float(coordinate_score),
            "status": str(coordinate_status or ""),
            "reason_family": str(coordinate_reason_family or ""),
            "seed_node_count": int(len(coordinate_seed_nodes)),
            "coordinate_vars": list(coordinate_hard.get("coordinate_vars", []) or []),
            "dominant_var_frac": _safe_float(low_rank_hard.get("dominant_var_frac", None)),
            "top_var": low_rank_hard.get("top_var", None),
        },
        "constant_lift": {
            "preferred": bool(regime_preferred),
            "score": float(regime_score),
            "status": str(regime_status or ""),
            "reason_family": "regime_lift",
            "has_explicit_task": bool(explicit_constant_task),
            "top_constant_name": str(regime_hard.get("top_constant_name", "") or ""),
            "top_constant_cv": top_constant_cv,
            "mean_cv": _safe_float(regime_hard.get("mean_cv", None)),
        },
    }


__all__ = ["build_local_lift_route_context"]
