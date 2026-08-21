# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import pytest

from nestynet_sr.sr_search.factorized_search.scheduler import (
    build_plan_candidates,
    build_plan_candidates_from_rows,
    choose_plan,
)


def test_build_plan_candidates_unions_routes_and_expands_budgets():
    candidates = build_plan_candidates(
        parent_key="parent_0",
        decision_context_id="decision_0",
        current_best_route_eff_mse=1.0,
        build_opportunity_rows=[
            {
                "opportunity_id": "build_0",
                "action": "replace",
                "path": [1],
                "budget_remaining": 1,
            }
        ],
        repair_opportunity_rows=[
            {
                "opportunity_id": "repair_0",
                "path": [2],
                "target_mode": "identity",
                "budget_remaining": 8,
            }
        ],
        hole_opportunity_rows=[
            {
                "opportunity_id": "hole_0",
                "path": [3],
                "budget_remaining": 8,
            }
        ],
        exact_budget_ladder=(1, 2, 4, 8),
        hole_exact_budget_cap=2,
    )

    build_budgets = sorted(
        candidate.exact_budget for candidate in candidates if candidate.route == "build"
    )
    repair_budgets = sorted(
        candidate.exact_budget for candidate in candidates if candidate.route == "repair"
    )
    hole_budgets = sorted(
        candidate.exact_budget for candidate in candidates if candidate.route == "hole"
    )

    assert build_budgets == [1]
    assert repair_budgets == [1, 2, 4, 8]
    assert hole_budgets == [1, 2]


def test_build_plan_candidates_force_shared_scheduler_context_and_preserve_route_provenance():
    candidates = build_plan_candidates(
        parent_key="parent_0",
        decision_context_id="scheduler_decision_0",
        build_opportunity_rows=[
            {
                "opportunity_id": "build_0",
                "decision_id": "build_decision_0",
                "decision_context_id": "build_context_0",
                "action": "replace",
            }
        ],
        repair_opportunity_rows=[
            {
                "opportunity_id": "repair_0",
                "decision_id": "repair_decision_0",
                "decision_context_id": "repair_context_0",
                "path": [1],
                "target_mode": "identity",
                "budget_remaining": 1,
            }
        ],
        hole_opportunity_rows=[
            {
                "opportunity_id": "hole_0",
                "decision_id": "hole_decision_0",
                "decision_context_id": "hole_context_0",
                "path": [2],
                "budget_remaining": 1,
            }
        ],
        exact_budget_ladder=(1,),
    )

    assert {candidate.decision_id for candidate in candidates} == {"scheduler_decision_0"}
    assert {
        str(candidate.features.get("decision_context_id", "") or "")
        for candidate in candidates
    } == {"scheduler_decision_0"}
    assert {
        str(candidate.features.get("scheduler_decision_context_id", "") or "")
        for candidate in candidates
    } == {"scheduler_decision_0"}
    route_provenance = {
        candidate.route: (
            str(candidate.features.get("route_decision_id", "") or ""),
            str(candidate.features.get("route_decision_context_id", "") or ""),
        )
        for candidate in candidates
    }
    assert route_provenance == {
        "build": ("build_decision_0", "build_context_0"),
        "repair": ("repair_decision_0", "repair_context_0"),
        "hole": ("hole_decision_0", "hole_context_0"),
    }


def test_build_plan_candidates_from_rows_prefers_shared_scheduler_context():
    candidates = build_plan_candidates_from_rows(
        [
            {
                "route_source": "build",
                "opportunity_id": "build_0",
                "decision_id": "build_decision_0",
                "decision_context_id": "build_context_0",
                "scheduler_decision_context_id": "scheduler_decision_0",
                "action": "replace",
            },
            {
                "route_source": "repair",
                "opportunity_id": "repair_0",
                "decision_id": "repair_decision_0",
                "decision_context_id": "repair_context_0",
                "scheduler_decision_context_id": "scheduler_decision_0",
                "path": [1],
                "target_mode": "identity",
                "budget_remaining": 1,
            },
        ],
        exact_budget_ladder=(1,),
    )

    assert {candidate.decision_id for candidate in candidates} == {"scheduler_decision_0"}
    assert {
        str(candidate.features.get("decision_context_id", "") or "")
        for candidate in candidates
    } == {"scheduler_decision_0"}


def test_choose_plan_uses_uncertainty_bonus_and_gap_based_fallback(monkeypatch):
    candidates = build_plan_candidates(
        parent_key="parent_0",
        build_opportunity_rows=[{"opportunity_id": "build_0", "action": "replace"}],
        repair_opportunity_rows=[{"opportunity_id": "repair_0", "path": [1], "budget_remaining": 1}],
        exact_budget_ladder=(1,),
    )

    def fake_predict_scheduler_plan_slate(_bundle, rows, *, acquisition_threshold=0.25):
        assert acquisition_threshold == 0.25
        assert len(rows) == 2
        return {
            "trained": True,
            "rows": [
                {
                    "route_source": "build",
                    "opportunity_id": "build_0",
                    "break_prob_0p25_at_budget_1": 0.91,
                    "acquisition_estimate_at_budget_1": 0.40,
                    "acquisition_sigma_at_budget_1": 0.10,
                },
                {
                    "route_source": "repair",
                    "opportunity_id": "repair_0",
                    "break_prob_0p25_at_budget_1": 0.95,
                    "acquisition_estimate_at_budget_1": 0.401,
                    "acquisition_sigma_at_budget_1": 0.10,
                },
            ],
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler.predict_scheduler_plan_slate",
        fake_predict_scheduler_plan_slate,
    )

    decision = choose_plan(
        {"scheduler_critic_trained": True},
        candidates,
        advisory_only=False,
        fallback_min_confidence=0.20,
        acquisition_threshold=0.25,
        uncertainty_bonus=0.05,
    )

    assert decision.trained is True
    assert decision.chosen_candidate is not None
    assert decision.chosen_candidate.route == "repair"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "low_dominance_confidence"
    assert decision.confidence < 0.20
    assert decision.dominance_prob < 0.60
    assert decision.rows[0]["plan_prediction_components"]["contributions"]["uncertainty"] == pytest.approx(0.005)


def test_choose_plan_does_not_fallback_for_dominant_acquisition_even_with_low_breakthrough_prob(monkeypatch):
    candidates = build_plan_candidates(
        parent_key="parent_0",
        build_opportunity_rows=[{"opportunity_id": "build_0", "action": "replace"}],
        repair_opportunity_rows=[{"opportunity_id": "repair_0", "path": [1], "budget_remaining": 1}],
        exact_budget_ladder=(1,),
    )

    def fake_predict_scheduler_plan_slate(_bundle, rows, *, acquisition_threshold=0.25):
        assert acquisition_threshold == 0.25
        assert len(rows) == 2
        return {
            "trained": True,
            "rows": [
                {
                    "route_source": "build",
                    "opportunity_id": "build_0",
                    "break_prob_0p25_at_budget_1": 0.92,
                    "acquisition_estimate_at_budget_1": 0.15,
                    "acquisition_sigma_at_budget_1": 0.10,
                },
                {
                    "route_source": "repair",
                    "opportunity_id": "repair_0",
                    "break_prob_0p25_at_budget_1": 0.12,
                    "acquisition_estimate_at_budget_1": 0.45,
                    "acquisition_sigma_at_budget_1": 0.10,
                },
            ],
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler.predict_scheduler_plan_slate",
        fake_predict_scheduler_plan_slate,
    )

    decision = choose_plan(
        {"scheduler_critic_trained": True},
        candidates,
        advisory_only=False,
        fallback_min_confidence=0.20,
        acquisition_threshold=0.25,
        uncertainty_bonus=0.05,
    )

    assert decision.trained is True
    assert decision.chosen_candidate is not None
    assert decision.chosen_candidate.route == "repair"
    assert decision.fallback_used is False
    assert decision.confidence > 0.20
    assert decision.dominance_prob > 0.60
    components = decision.rows[0]["plan_prediction_components"]
    assert components["raw"]["break_prob"] == pytest.approx(0.12)
    assert components["raw"]["tail_gain"] == pytest.approx(0.0)
    assert components["contributions"]["break"] == pytest.approx(0.12)


def test_choose_plan_uses_objective_score_when_bundle_requests_witness_mode(monkeypatch):
    candidates = build_plan_candidates(
        parent_key="parent_0",
        build_opportunity_rows=[{"opportunity_id": "build_0", "action": "replace"}],
        repair_opportunity_rows=[{"opportunity_id": "repair_0", "path": [1], "budget_remaining": 1}],
        exact_budget_ladder=(1,),
    )

    def fake_predict_scheduler_plan_slate(_bundle, rows, *, acquisition_threshold=0.25, acquisition_weights=None):
        assert acquisition_threshold == 0.25
        assert acquisition_weights is None
        assert len(rows) == 2
        return {
            "trained": True,
            "objective_mode": "witness",
            "rows": [
                {
                    "route_source": "build",
                    "opportunity_id": "build_0",
                    "break_prob_0p25_at_budget_1": 0.9,
                    "acquisition_estimate_at_budget_1": 0.8,
                    "acquisition_sigma_at_budget_1": 0.01,
                    "objective_estimate_at_budget_1": 0.1,
                    "objective_sigma_at_budget_1": 0.01,
                    "witness_delta_pred_at_budget_1": 0.2,
                    "witness_rate_pred_at_budget_1": 0.1,
                    "cost_total_pred_at_budget_1": 2.0,
                },
                {
                    "route_source": "repair",
                    "opportunity_id": "repair_0",
                    "break_prob_0p25_at_budget_1": 0.1,
                    "acquisition_estimate_at_budget_1": 0.2,
                    "acquisition_sigma_at_budget_1": 0.01,
                    "objective_estimate_at_budget_1": 0.6,
                    "objective_sigma_at_budget_1": 0.01,
                    "witness_delta_pred_at_budget_1": 0.9,
                    "witness_rate_pred_at_budget_1": 0.6,
                    "cost_total_pred_at_budget_1": 1.5,
                },
            ],
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler.predict_scheduler_plan_slate",
        fake_predict_scheduler_plan_slate,
    )

    decision = choose_plan(
        {"scheduler_critic_trained": True, "objective_mode": "witness"},
        candidates,
        advisory_only=False,
        acquisition_threshold=0.25,
        uncertainty_bonus=0.05,
    )

    assert decision.trained is True
    assert decision.objective_mode == "witness"
    assert decision.chosen_candidate is not None
    assert decision.chosen_candidate.route == "repair"
    assert decision.rows[0]["plan_objective_mode"] == "witness"
    assert decision.rows[0]["plan_objective_estimate"] == pytest.approx(0.6)
    assert decision.rows[0]["plan_witness_rate_pred"] == pytest.approx(0.6)
