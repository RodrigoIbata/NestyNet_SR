# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.factorized_search.solver_market import (
    SolverMarketRouteCall,
    run_preview_solver_market,
)


def test_solver_market_dedups_rows_and_selects_best_route():
    def _route_a(**kwargs):
        return {
            "rows": [
                {
                    "expr": ("var", 0),
                    "child_key": "x0",
                    "local_probe_mse": 0.15,
                    "local_fit_mse": 0.3,
                    "witness_value_loss": 0.15,
                    "witness_energy_total": 0.15,
                    "witness_fit_jet_source": "numeric_local_quadratic",
                    "witness_exact_jet_used": False,
                },
                {"expr": ("var", 1), "child_key": "x1", "local_probe_mse": 0.6, "local_fit_mse": 0.5},
            ],
            "solver_meta": {"status": "ok", "child_spec_states": [{"spec_kind": "local_problem", "path": [1]}]},
        }

    def _route_b(**kwargs):
        return {
            "rows": [
                {
                    "expr": ("var", 0),
                    "child_key": "x0",
                    "local_probe_mse": 0.2,
                    "local_fit_mse": 0.1,
                    "witness_value_loss": 0.05,
                    "witness_grad_loss": 0.02,
                    "witness_energy_total": 0.07,
                    "witness_fit_jet_source": "oracle",
                    "witness_probe_jet_source": "oracle",
                    "witness_exact_jet_used": True,
                },
                {"expr": ("var", 2), "child_key": "x2", "local_probe_mse": 0.5, "local_fit_mse": 0.4},
            ],
            "solver_meta": {"status": "ok", "child_spec_states": [{"spec_kind": "local_problem", "path": [1]}]},
        }

    result = run_preview_solver_market(
        [
            SolverMarketRouteCall("route_a", "solver_a", "path_hole", _route_a, {}),
            SolverMarketRouteCall("route_b", "solver_b", "path_hole", _route_b, {}),
        ],
        preview_topk=3,
        exact_topk=2,
    )

    rows = result["rows"]
    meta = result["solver_meta"]

    assert [row["child_key"] for row in rows] == ["x0", "x2", "x1"]
    assert rows[0]["solver_market_route"] == "route_b"
    assert rows[0]["witness_value_loss"] == 0.05
    assert rows[0]["witness_grad_loss"] == 0.02
    assert rows[0]["witness_energy_total"] == 0.07
    assert meta["solver_market_route_count"] == 2
    assert meta["solver_market_candidate_count_raw"] == 4
    assert meta["solver_market_candidate_count_unique"] == 3
    assert meta["solver_market_selected_route"] == "route_b"
    assert meta["solver_market_selected_method_name"] == "solver_b"
    assert meta["solver_market_selected_fit_jet_source"] == "oracle"
    assert meta["solver_market_selected_exact_jet_used"] is True
    assert meta["solver_market_routes"][1]["preview_best_exact_jet_used"] is True
    assert meta["child_spec_state_count"] == 1
