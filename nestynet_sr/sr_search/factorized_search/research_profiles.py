# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Any, Mapping


RESEARCH_PROFILE_NAMES: tuple[str, ...] = (
    "legacy",
    "teacher_witness",
    "teacher_witness_full",
    "teacher_witness_exact",
)

_PROFILE_ALIASES: dict[str, str] = {
    "": "legacy",
    "current": "legacy",
    "default": "legacy",
    "legacy": "legacy",
    "teacher_witness": "teacher_witness",
    "teacher-witness": "teacher_witness",
    "teacher_witness_full": "teacher_witness_full",
    "teacher-witness-full": "teacher_witness_full",
    "teacher_witness_exact": "teacher_witness_exact",
    "teacher-witness-exact": "teacher_witness_exact",
}


DISCOVERY_RESEARCH_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "legacy": {},
    "teacher_witness": {
        "discovery_constant_lift_enable": True,
        "discovery_constant_lift_apply_enable": True,
        "discovery_constant_lift_apply_topk": 1,
        "witness_capture_enable": True,
        "witness_hessian_diag_enable": False,
        "diagnostic_set": "physics",
        "beta": 1.0,
        "gamma": 0.25,
        "disagreement_mode": "witness",
        "experiment_optimize_enable": False,
        "theory_benchmark_enable": False,
    },
    "teacher_witness_full": {
        "discovery_constant_lift_enable": True,
        "discovery_constant_lift_apply_enable": True,
        "discovery_constant_lift_apply_topk": 2,
        "witness_capture_enable": True,
        "witness_hessian_diag_enable": True,
        "diagnostic_set": "physics",
        "beta": 1.0,
        "gamma": 0.5,
        "disagreement_mode": "witness",
        "experiment_optimize_enable": True,
        "theory_benchmark_enable": True,
    },
    "teacher_witness_exact": {
        "discovery_constant_lift_enable": True,
        "discovery_constant_lift_apply_enable": True,
        "discovery_constant_lift_apply_topk": 2,
        "witness_capture_enable": True,
        "witness_hessian_diag_enable": True,
        "diagnostic_set": "physics",
        "beta": 1.0,
        "gamma": 0.5,
        "disagreement_mode": "witness",
        "experiment_optimize_enable": True,
        "theory_benchmark_enable": True,
    },
}


ENGINE_RESEARCH_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "legacy": {},
    "teacher_witness": {
        "inverse_steering_enable": True,
        "inverse_spec_enable": True,
        "hole_search_enable": True,
        "hole_search_solver_market_enable": True,
        "inverse_spec_recursive_sr_enable": True,
        "inverse_spec_constant_lift_route_enable": True,
        "inverse_spec_coordinate_lift_enable": True,
        "inverse_spec_tangent_edit_enable": True,
        "inverse_spec_soft_edit_enable": True,
        "inverse_spec_witness_jets_enable": True,
        "inverse_spec_witness_loss_enable": True,
        "inverse_spec_witness_grad_weight": 1.0,
        "inverse_spec_witness_d2_weight": 0.25,
        "inverse_spec_witness_diag_weight": 0.10,
        "inverse_spec_witness_physics_weight": 0.05,
        "inverse_spec_active_var_screen_enable": True,
        "inverse_spec_directional_market_enable": True,
    },
    "teacher_witness_full": {
        "inverse_steering_enable": True,
        "inverse_spec_enable": True,
        "hole_search_enable": True,
        "hole_search_solver_market_enable": True,
        "hole_search_solver_market_proposal_objects_enable": True,
        "inverse_spec_recursive_sr_enable": True,
        "inverse_spec_constant_lift_route_enable": True,
        "inverse_spec_coordinate_lift_enable": True,
        "inverse_spec_tangent_edit_enable": True,
        "inverse_spec_soft_edit_enable": True,
        "inverse_spec_witness_jets_enable": True,
        "inverse_spec_witness_d2_enable": True,
        "inverse_spec_witness_loss_enable": True,
        "inverse_spec_witness_grad_weight": 1.0,
        "inverse_spec_witness_d2_weight": 0.25,
        "inverse_spec_witness_diag_weight": 0.25,
        "inverse_spec_witness_physics_weight": 0.10,
        "inverse_spec_active_var_screen_enable": True,
        "inverse_spec_directional_market_enable": True,
        "inverse_spec_family_battery_enable": True,
        "inverse_spec_family_battery_mode": "expanded",
        "scheduler_witness_energy_enable": True,
        "scheduler_objective_mode": "witness",
    },
    "teacher_witness_exact": {
        "inverse_steering_enable": True,
        "inverse_spec_enable": True,
        "hole_search_enable": True,
        "hole_search_solver_market_enable": True,
        "hole_search_solver_market_proposal_objects_enable": True,
        "inverse_spec_recursive_sr_enable": True,
        "inverse_spec_constant_lift_route_enable": True,
        "inverse_spec_coordinate_lift_enable": True,
        "inverse_spec_tangent_edit_enable": True,
        "inverse_spec_soft_edit_enable": True,
        "inverse_spec_witness_jets_enable": True,
        "inverse_spec_witness_d2_enable": True,
        "inverse_spec_witness_loss_enable": True,
        "inverse_spec_witness_grad_weight": 1.0,
        "inverse_spec_witness_d2_weight": 0.25,
        "inverse_spec_witness_diag_weight": 0.25,
        "inverse_spec_witness_physics_weight": 0.10,
        "inverse_spec_active_var_screen_enable": True,
        "inverse_spec_directional_market_enable": True,
        "inverse_spec_family_battery_enable": True,
        "inverse_spec_family_battery_mode": "expanded",
        "scheduler_witness_energy_enable": True,
        "scheduler_objective_mode": "witness",
    },
}


def normalize_research_profile_name(name: str | None) -> str:
    token = str(name or "").strip().lower().replace("-", "_")
    try:
        return str(_PROFILE_ALIASES[token])
    except KeyError as exc:
        allowed = ", ".join(RESEARCH_PROFILE_NAMES)
        raise ValueError(f"unknown research profile {name!r}; expected one of {allowed}") from exc


def resolve_discovery_research_profile(name: str | None) -> tuple[str, dict[str, Any]]:
    canonical = normalize_research_profile_name(name)
    return canonical, dict(DISCOVERY_RESEARCH_PROFILE_OVERRIDES.get(canonical, {}))


def resolve_engine_research_profile(name: str | None) -> tuple[str, dict[str, Any]]:
    canonical = normalize_research_profile_name(name)
    return canonical, dict(ENGINE_RESEARCH_PROFILE_OVERRIDES.get(canonical, {}))


def apply_research_profile_overrides(
    values: Mapping[str, Any] | None,
    *,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = dict(values or {})
    out.update(dict(overrides or {}))
    return out


__all__ = [
    "DISCOVERY_RESEARCH_PROFILE_OVERRIDES",
    "ENGINE_RESEARCH_PROFILE_OVERRIDES",
    "RESEARCH_PROFILE_NAMES",
    "apply_research_profile_overrides",
    "normalize_research_profile_name",
    "resolve_discovery_research_profile",
    "resolve_engine_research_profile",
]
