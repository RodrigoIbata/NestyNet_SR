# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.discovery.active_design import ExperimentCandidate, select_next_experiment
from nestynet_sr.discovery.committee import CommitteeMember, CommitteeState
from nestynet_sr.discovery.experiment_opt import optimize_continuous_experiment_candidates


def _committee() -> CommitteeState:
    return CommitteeState(
        members=(
            CommitteeMember(member_id="m0", symbolic_structure="x0", committee_weight=0.5),
            CommitteeMember(member_id="m1", symbolic_structure="2-x0", committee_weight=0.5),
        ),
        canonical_member_count=2,
    )


def test_select_next_experiment_can_optimize_continuous_points():
    committee = _committee()
    candidates = [
        ExperimentCandidate(
            experiment_id="mid_box",
            conditions={"type": "points", "n_points": 2, "shape": [2, 1]},
            metadata={
                "continuous_optimizer": {
                    "enabled": True,
                    "source_type": "points",
                    "points": [[0.8], [1.2]],
                    "bounds": [[0.0, 2.0]],
                }
            },
        )
    ]
    forward_fns = {
        "m0": lambda x: x[:, :1],
        "m1": lambda x: 2.0 - x[:, :1],
    }

    result = select_next_experiment(
        committee,
        candidates,
        disagreement_mode="witness",
        optimize_continuous=True,
        experiment_optimizer=lambda state, rows, **kwargs: optimize_continuous_experiment_candidates(
            state,
            rows,
            forward_fns_by_member_id=forward_fns,
            beta=float(kwargs.get("beta", 0.0)),
            gamma=float(kwargs.get("gamma", 0.0)),
            disagreement_mode=str(kwargs.get("disagreement_mode", "witness")),
            lambda_cost=float(kwargs.get("lambda_cost", 1.0)),
            lambda_noise=float(kwargs.get("lambda_noise", 1.0)),
            lambda_feasibility=float(kwargs.get("lambda_feasibility", 1.0)),
            opt_steps=24,
            opt_lr=0.1,
            project_mode="nearest_box",
            include_gradients=True,
            include_diagnostics=False,
        ),
    )

    assert result["optimization"]["enabled"] is True
    assert result["optimization"]["optimized_candidate_count"] == 1
    assert result["selected"]["conditions"]["optimized"] is True
    assert result["selected"]["score"] > 0.0


def test_optimize_continuous_experiment_candidates_emits_physics_diagnostics():
    committee = _committee()
    candidates = [
        ExperimentCandidate(
            experiment_id="diag_box",
            conditions={"type": "points", "n_points": 3, "shape": [3, 1]},
            metadata={
                "continuous_optimizer": {
                    "enabled": True,
                    "source_type": "points",
                    "points": [[-1.0], [0.0], [1.0]],
                    "bounds": [[-2.0, 2.0]],
                }
            },
        )
    ]
    forward_fns = {
        "m0": lambda x: x[:, :1] * x[:, :1],
        "m1": lambda x: x[:, :1] * x[:, :1] * x[:, :1],
    }

    result = optimize_continuous_experiment_candidates(
        committee,
        candidates,
        forward_fns_by_member_id=forward_fns,
        beta=0.0,
        gamma=1.0,
        disagreement_mode="witness",
        lambda_cost=1.0,
        lambda_noise=1.0,
        lambda_feasibility=1.0,
        opt_steps=8,
        opt_lr=0.05,
        project_mode="nearest_box",
        include_gradients=True,
        include_diagnostics=True,
    )

    diag = result["candidates"][0].diagnostic_predictions
    assert "mirror_even_residual" in diag["m0"]
    assert "zero_crossing_soft_count" in diag["m0"]
    assert "tail_slope_gap_abs" in diag["m1"]


def test_optimize_continuous_experiment_candidates_retired_scalar_alias_uses_witness_surface():
    committee = _committee()
    candidates = [
        ExperimentCandidate(
            experiment_id="mid_box",
            conditions={"type": "points", "n_points": 2, "shape": [2, 1]},
            metadata={
                "continuous_optimizer": {
                    "enabled": True,
                    "source_type": "points",
                    "points": [[0.8], [1.2]],
                    "bounds": [[0.0, 2.0]],
                }
            },
        )
    ]
    forward_fns = {
        "m0": lambda x: x[:, :1],
        "m1": lambda x: 2.0 - x[:, :1],
    }

    result = optimize_continuous_experiment_candidates(
        committee,
        candidates,
        forward_fns_by_member_id=forward_fns,
        beta=0.0,
        gamma=0.0,
        disagreement_mode="scalar",
        lambda_cost=1.0,
        lambda_noise=1.0,
        lambda_feasibility=1.0,
        opt_steps=16,
        opt_lr=0.1,
        project_mode="nearest_box",
        include_gradients=True,
        include_diagnostics=False,
    )

    summary = result["summary"]
    assert summary["optimized_candidate_count"] == 1
    assert summary["total_score_improvement"] > 0.0
