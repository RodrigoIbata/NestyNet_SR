# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.discovery.active_design import ExperimentCandidate
from nestynet_sr.discovery.closed_loop import run_closed_loop_iteration
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec


def test_run_closed_loop_iteration_builds_committee_and_selects_experiment():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim((1, 0)), us.dim((0, 1))),
        y_dim=us.dim((1, 1)),
    )
    candidate_laws = [
        {
            "member_id": "m0",
            "expr": ("mul", ("var", 0), ("var", 1)),
            "validation_error": 0.05,
            "simplicity_score": 1.0,
        },
        {
            "member_id": "m1",
            "expr": ("mul", ("const", 3.0), ("mul", ("var", 0), ("var", 1))),
            "validation_error": 0.07,
            "simplicity_score": 0.9,
        },
        {
            "member_id": "m2",
            "expr": ("add", ("var", 0), ("const", 1.0)),
            "validation_error": 0.15,
            "simplicity_score": 0.8,
        },
    ]
    design_candidates = [
        ExperimentCandidate(
            experiment_id="near_agreement",
            observable_predictions={"m0": 1.0, "m2": 1.1},
            cost=0.1,
        ),
        ExperimentCandidate(
            experiment_id="high_disagreement",
            observable_predictions={"m0": 0.0, "m2": 2.0},
            cost=0.1,
        ),
    ]

    result = run_closed_loop_iteration(
        candidate_laws,
        design_candidates,
        units_spec=spec,
        lambda_cost=0.1,
        lambda_noise=0.1,
        lambda_feasibility=0.1,
    )

    assert len(result.committee_state.members) == 2
    assert result.selected_experiment is not None
    assert result.selected_experiment["experiment_id"] == "high_disagreement"
    assert "m0" in result.physics_reports


def test_run_closed_loop_iteration_witness_mode_prefers_shape_split():
    candidate_laws = [
        {
            "member_id": "m0",
            "expr": ("var", 0),
            "validation_error": 0.05,
            "simplicity_score": 1.0,
        },
        {
            "member_id": "m1",
            "expr": ("sub", ("const", 2.0), ("var", 0)),
            "validation_error": 0.05,
            "simplicity_score": 1.0,
        },
    ]
    design_candidates = [
        ExperimentCandidate(
            experiment_id="shape_agree",
            observable_predictions={"m0": [0.0, 2.0], "m1": [0.0, 2.0]},
        ),
        ExperimentCandidate(
            experiment_id="shape_split",
            observable_predictions={"m0": [0.0, 2.0], "m1": [2.0, 0.0]},
        ),
    ]

    result = run_closed_loop_iteration(
        candidate_laws,
        design_candidates,
        disagreement_mode="witness",
    )

    assert result.selected_experiment is not None
    assert result.selected_experiment["experiment_id"] == "shape_split"
    assert result.ranked_experiments[0]["disagreement_mode"] == "witness"
