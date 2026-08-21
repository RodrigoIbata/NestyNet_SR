# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.sr_search.factorized_search.research_profiles import (
    apply_research_profile_overrides,
    resolve_discovery_research_profile,
    resolve_engine_research_profile,
)


def test_discovery_research_profile_alias_resolves_to_teacher_witness_full():
    name, overrides = resolve_discovery_research_profile("teacher-witness-full")

    assert name == "teacher_witness_full"
    assert overrides["witness_capture_enable"] is True
    assert overrides["witness_hessian_diag_enable"] is True
    assert overrides["diagnostic_set"] == "physics"
    assert overrides["disagreement_mode"] == "witness"
    assert overrides["experiment_optimize_enable"] is True
    assert overrides["theory_benchmark_enable"] is True


def test_engine_research_profile_teacher_witness_full_enables_local_market_stack():
    name, overrides = resolve_engine_research_profile("teacher_witness_full")

    assert name == "teacher_witness_full"
    assert overrides["inverse_steering_enable"] is True
    assert overrides["inverse_spec_enable"] is True
    assert overrides["hole_search_enable"] is True
    assert overrides["hole_search_solver_market_enable"] is True
    assert overrides["inverse_spec_recursive_sr_enable"] is True
    assert overrides["inverse_spec_constant_lift_route_enable"] is True
    assert overrides["inverse_spec_coordinate_lift_enable"] is True
    assert overrides["inverse_spec_tangent_edit_enable"] is True
    assert overrides["inverse_spec_soft_edit_enable"] is True
    assert overrides["inverse_spec_witness_jets_enable"] is True
    assert overrides["inverse_spec_witness_d2_enable"] is True
    assert overrides["inverse_spec_witness_loss_enable"] is True
    assert overrides["inverse_spec_witness_grad_weight"] == 1.0
    assert overrides["inverse_spec_witness_d2_weight"] == 0.25
    assert overrides["inverse_spec_witness_diag_weight"] == 0.25
    assert overrides["inverse_spec_witness_physics_weight"] == 0.10
    assert overrides["inverse_spec_directional_market_enable"] is True
    assert overrides["scheduler_witness_energy_enable"] is True
    assert overrides["scheduler_objective_mode"] == "witness"


def test_research_profile_teacher_witness_exact_resolves_and_enables_exact_teacher_audit_stack():
    disc_name, disc_overrides = resolve_discovery_research_profile("teacher-witness-exact")
    eng_name, eng_overrides = resolve_engine_research_profile("teacher_witness_exact")

    assert disc_name == "teacher_witness_exact"
    assert disc_overrides["witness_capture_enable"] is True
    assert disc_overrides["witness_hessian_diag_enable"] is True
    assert disc_overrides["diagnostic_set"] == "physics"
    assert disc_overrides["disagreement_mode"] == "witness"
    assert disc_overrides["experiment_optimize_enable"] is True
    assert disc_overrides["theory_benchmark_enable"] is True
    assert eng_name == "teacher_witness_exact"
    assert eng_overrides["inverse_spec_witness_jets_enable"] is True
    assert eng_overrides["inverse_spec_witness_d2_enable"] is True
    assert eng_overrides["inverse_spec_witness_loss_enable"] is True
    assert eng_overrides["inverse_spec_witness_diag_weight"] == 0.25
    assert eng_overrides["inverse_spec_witness_physics_weight"] == 0.10
    assert eng_overrides["inverse_spec_family_battery_mode"] == "expanded"
    assert eng_overrides["scheduler_witness_energy_enable"] is True
    assert eng_overrides["scheduler_objective_mode"] == "witness"


def test_engine_research_profile_teacher_witness_turns_on_diag_and_physics_channels():
    name, overrides = resolve_engine_research_profile("teacher_witness")

    assert name == "teacher_witness"
    assert overrides["inverse_spec_witness_loss_enable"] is True
    assert overrides["inverse_spec_witness_diag_weight"] == 0.10
    assert overrides["inverse_spec_witness_physics_weight"] == 0.05


def test_apply_research_profile_overrides_prefers_profile_values():
    values = {"beta": 0.0, "gamma": 0.0, "disagreement_mode": "auto"}
    out = apply_research_profile_overrides(
        values,
        overrides={"beta": 1.0, "disagreement_mode": "witness"},
    )

    assert out == {"beta": 1.0, "gamma": 0.0, "disagreement_mode": "witness"}
