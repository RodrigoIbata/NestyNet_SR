# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.factorized_search.solver_market import (
    SolverMarketRouteCall,
    run_preview_solver_market,
)
from nestynet_sr.sr_search.factorized_search.subproblem_spec import deserialize_solver_proposal


def test_solver_market_uses_solver_proposals_internally_when_enabled():
    def _route_a(**kwargs):
        return {
            "rows": [
                {
                    "expr": ("var", 0),
                    "child_key": "x0",
                    "local_probe_mse": 0.4,
                    "local_fit_mse": 0.3,
                    "generation_source": "route_a_gen",
                    "proposal_family": "local_problem",
                },
            ],
            "solver_meta": {"status": "ok"},
        }

    def _route_b(**kwargs):
        return {
            "rows": [
                {
                    "expr": ("var", 0),
                    "child_key": "x0",
                    "local_probe_mse": 0.2,
                    "local_fit_mse": 0.1,
                    "generation_source": "route_b_gen",
                    "proposal_family": "local_problem",
                },
                {
                    "expr": ("var", 1),
                    "child_key": "x1",
                    "local_probe_mse": 0.5,
                    "local_fit_mse": 0.4,
                    "generation_source": "route_b_gen",
                    "proposal_family": "local_problem",
                },
            ],
            "solver_meta": {"status": "ok"},
        }

    result = run_preview_solver_market(
        [
            SolverMarketRouteCall("route_a", "solver_a", "path_hole", _route_a, {}),
            SolverMarketRouteCall("route_b", "solver_b", "path_hole", _route_b, {}),
        ],
        preview_topk=2,
        exact_topk=1,
        proposal_objects_enable=True,
    )

    rows = result["rows"]
    meta = result["solver_meta"]

    assert [row["child_key"] for row in rows] == ["x0", "x1"]
    assert rows[0]["solver_market_route"] == "route_b"
    proposal = deserialize_solver_proposal(rows[0]["solver_proposal"])
    assert proposal is not None
    assert proposal.preview_loss == 0.2
    assert proposal.source == "route_b_gen"
    assert proposal.family == "local_problem"
    assert meta["solver_market_proposal_objects_enable"] is True
    assert meta["solver_market_candidate_count_unique"] == 2
    assert meta["solver_market_selected_proposal"]["preview_loss"] == 0.2
