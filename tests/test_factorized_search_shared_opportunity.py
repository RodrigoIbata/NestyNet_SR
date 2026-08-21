# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import pytest

from nestynet_sr.sr_search.factorized_search.shared_opportunity import shared_opportunity_row_dict


def test_shared_opportunity_row_dict_supports_hole_route_and_v2_fields():
    row = shared_opportunity_row_dict(
        {
            "route_source": "hole",
            "opportunity_type": "hole_opportunity",
            "opportunity_id": "hole_1",
            "decision_id": "decision_hole",
            "decision_context_id": "ctx_hole",
            "beam_id": "beam_hole",
            "parent_key": "parent_hole",
            "parent_expr": "x+y",
            "action": "hole_search",
            "path": [1, 2],
            "path_source": "hole_frontier",
            "target_mode": "identity",
            "method_name": "solver_market",
            "subroute": "path_hole",
            "current_best_child_eff_mse": 0.125,
            "global_best_eff_mse": 0.100,
            "best_alt_route_eff_mse": 0.140,
            "observed_wall_seconds": 0.5,
            "observed_exact_evals": 1,
            "observed_preview_evals": 3,
            "hole_search_solver_market_route_count": 2,
            "hole_search_solver_market_selected_route": "inverse_spec_path",
        },
        route_source="hole",
    )

    assert row["shared_opportunity_schema_version"] == 2
    assert row["route_source"] == "hole"
    assert row["target_mode"] == "identity"
    assert row["method_name"] == "solver_market"
    assert row["subroute"] == "path_hole"
    assert row["decision_context_id"] == "ctx_hole"
    assert row["parent_key"] == "parent_hole"
    assert row["opportunity_route_valid_hole"] == 1.0
    assert row["opportunity_route_valid_build"] == 0.0
    assert row["observed_wall_seconds"] == 0.5
    assert row["observed_exact_evals"] == 1
    assert row["observed_preview_evals"] == 3
    assert row["hole_search_solver_market_route_count"] == 2
    assert row["hole_search_solver_market_selected_route"] == "inverse_spec_path"
    assert row["witness_value_loss"] == 0.125
    assert row["witness_energy_total"] == 0.125


def test_shared_opportunity_row_dict_normalizes_witness_energy_fields():
    row = shared_opportunity_row_dict(
        {
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "opportunity_id": "repair_1",
            "decision_id": "decision_repair",
            "beam_id": "beam_repair",
            "parent_expr": "x",
            "action": "inv_steer",
            "best_preview_probe_mse": 0.2,
            "best_tuple_utility_estimate": 0.4,
        },
        route_source="repair",
    )

    assert row["witness_value_loss"] == 0.2
    assert row["witness_energy_total"] == 0.2
    assert row["witness_energy_delta_estimate"] == 0.4


def test_shared_opportunity_row_dict_preserves_explicit_witness_fields():
    row = shared_opportunity_row_dict(
        {
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "opportunity_id": "repair_explicit",
            "decision_id": "decision_repair_explicit",
            "beam_id": "beam_repair_explicit",
            "parent_expr": "x",
            "action": "inv_steer",
            "best_preview_probe_mse": 0.2,
            "witness_value_loss": 0.05,
            "witness_grad_loss": 0.03,
            "witness_energy_total": 0.08,
        },
        route_source="repair",
    )

    assert row["witness_value_loss"] == 0.05
    assert row["witness_grad_loss"] == 0.03
    assert row["witness_energy_total"] == 0.08


def test_shared_opportunity_row_dict_preserves_witness_jet_provenance_fields():
    row = shared_opportunity_row_dict(
        {
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "opportunity_id": "repair_oracle",
            "decision_id": "decision_repair_oracle",
            "beam_id": "beam_repair_oracle",
            "parent_expr": "x",
            "action": "inv_steer",
            "witness_fit_jet_source": "oracle",
            "witness_probe_jet_source": "oracle",
            "witness_numeric_jet_fallback_used": False,
            "witness_exact_jet_used": True,
        },
        route_source="repair",
    )

    assert row["witness_fit_jet_source"] == "oracle"
    assert row["witness_probe_jet_source"] == "oracle"
    assert row["witness_numeric_jet_fallback_used"] is False
    assert row["witness_exact_jet_used"] is True


def test_shared_opportunity_row_dict_normalizes_realized_witness_transition_fields():
    row = shared_opportunity_row_dict(
        {
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "opportunity_id": "repair_realized_witness",
            "decision_id": "decision_repair_realized_witness",
            "beam_id": "beam_repair_realized_witness",
            "parent_expr": "x",
            "action": "inv_steer",
            "realized_witness_value_loss_before": 0.7,
            "realized_witness_grad_loss_before": 0.2,
            "realized_witness_value_loss_after": 0.25,
            "realized_witness_grad_loss_after": 0.05,
        },
        route_source="repair",
    )

    assert row["realized_witness_energy_total_before"] == pytest.approx(0.9)
    assert row["realized_witness_energy_total_after"] == pytest.approx(0.3)
    assert row["realized_witness_value_delta"] == pytest.approx(0.45)
    assert row["realized_witness_grad_delta"] == pytest.approx(0.15)
    assert row["realized_witness_energy_delta"] == pytest.approx(0.6)
