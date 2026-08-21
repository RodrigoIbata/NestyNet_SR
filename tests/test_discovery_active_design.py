# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.discovery.active_design import (
    ExperimentCandidate,
    committee_disagreement_with_mode,
    select_next_experiment,
)
from nestynet_sr.discovery.committee import CommitteeMember, CommitteeState


def _committee() -> CommitteeState:
    return CommitteeState(
        members=(
            CommitteeMember(member_id="m0", symbolic_structure="x0", committee_weight=0.6),
            CommitteeMember(member_id="m1", symbolic_structure="x1", committee_weight=0.4),
        ),
        canonical_member_count=2,
    )


def test_select_next_experiment_prefers_high_disagreement_under_penalties():
    committee = _committee()
    candidates = [
        ExperimentCandidate(
            experiment_id="low_var",
            observable_predictions={"m0": 1.0, "m1": 1.1},
            cost=0.1,
            noise_risk=0.1,
            feasibility_penalty=0.0,
        ),
        ExperimentCandidate(
            experiment_id="high_var",
            observable_predictions={"m0": 0.0, "m1": 2.0},
            cost=0.2,
            noise_risk=0.1,
            feasibility_penalty=0.0,
        ),
    ]

    result = select_next_experiment(
        committee,
        candidates,
        lambda_cost=0.2,
        lambda_noise=0.1,
        lambda_feasibility=0.1,
    )

    assert result["selected"]["experiment_id"] == "high_var"
    assert result["ranking"][0]["score"] > result["ranking"][1]["score"]


def test_select_next_experiment_retired_scalar_alias_uses_witness_shape_ranking():
    committee = _committee()
    candidates = [
        ExperimentCandidate(
            experiment_id="shape_agree",
            observable_predictions={"m0": [0.0, 2.0], "m1": [0.0, 2.0]},
        ),
        ExperimentCandidate(
            experiment_id="shape_split",
            observable_predictions={"m0": [0.0, 2.0], "m1": [2.0, 0.0]},
        ),
    ]

    result = select_next_experiment(
        committee,
        candidates,
        disagreement_mode="scalar",
    )
    assert result["disagreement_mode"] == "witness"
    assert result["selected"]["experiment_id"] == "shape_split"
    assert result["ranking"][0]["score"] > result["ranking"][1]["score"]


def test_select_next_experiment_defaults_to_witness_mode():
    committee = _committee()
    candidates = [
        ExperimentCandidate(
            experiment_id="shape_agree",
            observable_predictions={"m0": [0.0, 2.0], "m1": [0.0, 2.0]},
        ),
        ExperimentCandidate(
            experiment_id="shape_split",
            observable_predictions={"m0": [0.0, 2.0], "m1": [2.0, 0.0]},
        ),
    ]

    result = select_next_experiment(committee, candidates)

    assert result["disagreement_mode"] == "witness"
    assert result["selected"]["experiment_id"] == "shape_split"


def test_select_next_experiment_witness_mode_can_rank_diagnostic_structure():
    committee = _committee()
    candidates = [
        ExperimentCandidate(
            experiment_id="diag_agree",
            observable_predictions={"m0": [1.0, 1.0], "m1": [1.0, 1.0]},
            derivative_predictions={"m0": [[0.0], [0.0]], "m1": [[0.0], [0.0]]},
            diagnostic_predictions={
                "m0": {"mirror_even_residual": 0.0, "zero_crossing_count": 0.0},
                "m1": {"mirror_even_residual": 0.0, "zero_crossing_count": 0.0},
            },
        ),
        ExperimentCandidate(
            experiment_id="diag_split",
            observable_predictions={"m0": [1.0, 1.0], "m1": [1.0, 1.0]},
            derivative_predictions={"m0": [[0.0], [0.0]], "m1": [[0.0], [0.0]]},
            diagnostic_predictions={
                "m0": {"mirror_even_residual": 0.0, "zero_crossing_count": 0.0},
                "m1": {"mirror_even_residual": 1.0, "zero_crossing_count": 2.0},
            },
        ),
    ]

    result = select_next_experiment(
        committee,
        candidates,
        beta=0.0,
        gamma=1.0,
        disagreement_mode="witness",
    )

    assert result["selected"]["experiment_id"] == "diag_split"
    assert result["ranking"][0]["disagreement"]["diagnostic_component"] > result["ranking"][1]["disagreement"]["diagnostic_component"]


def test_committee_disagreement_witness_mode_exposes_distance_aliases():
    committee = _committee()
    candidate = ExperimentCandidate(
        experiment_id="shape_split",
        observable_predictions={"m0": [0.0, 2.0], "m1": [2.0, 0.0]},
        derivative_predictions={"m0": [[0.0], [1.0]], "m1": [[1.0], [0.0]]},
        diagnostic_predictions={
            "m0": {"mirror_even_residual": 0.0},
            "m1": {"mirror_even_residual": 1.0},
        },
    )

    disagreement = committee_disagreement_with_mode(
        committee,
        candidate,
        disagreement_mode="witness",
    )

    assert disagreement["mode"] == "witness"
    assert disagreement["observable_distance"] == disagreement["observable_component"]
    assert disagreement["derivative_distance"] == disagreement["derivative_component"]
    assert disagreement["diagnostic_distance"] == disagreement["diagnostic_component"]
    assert disagreement["observable_variance"] == disagreement["observable_component"]
    assert disagreement["derivative_variance"] == disagreement["derivative_component"]
    assert disagreement["diagnostic_variance"] == disagreement["diagnostic_component"]
