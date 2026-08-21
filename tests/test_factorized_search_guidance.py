# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.factorized_search.policy.guidance import (
    _credible_route_compare_decision,
    _credible_route_preview_repair_opportunity_rows,
    _repair_route_compare_decision,
)


def _route_pred(
    *,
    repair_eff: float,
    build_eff: float,
    best_route: str,
    repair_prob: float,
    margin_estimate: float,
    exact_margin: float,
):
    return {
        "trained": True,
        "best_route": str(best_route),
        "repair_prob": float(repair_prob),
        "build_prob": float(1.0 - repair_prob),
        "margin_estimate": float(margin_estimate),
        "exact_margin": float(exact_margin),
        "repair_summary": {
            "rows": [
                {"child_eff_mse": float(repair_eff)},
            ],
        },
        "build_summary": {
            "rows": [
                {"child_eff_mse": float(build_eff)},
            ],
        },
    }


def test_credible_route_compare_promotes_repair_when_hidden_upside_flips_margin():
    route_pred = _route_pred(
        repair_eff=0.40,
        build_eff=0.25,
        best_route="build",
        repair_prob=0.10,
        margin_estimate=-0.20,
        exact_margin=-0.20,
    )
    repair_unseen_pred = {
        "trained": True,
        "rows": [
            {
                "route_source": "repair",
                "beam_id": "beam_a",
                "opportunity_id": "opp_a",
                "budget_remaining": 2,
                "acquisition_estimate": 0.20,
                "expected_gain_next_under_executor": 0.24,
                "fragility_prob": 0.05,
                "cost_estimate": 0.02,
            },
            {
                "route_source": "repair",
                "beam_id": "beam_b",
                "opportunity_id": "opp_b",
                "budget_remaining": 1,
                "acquisition_estimate": 0.05,
                "expected_gain_next_under_executor": 0.08,
                "fragility_prob": 0.01,
                "cost_estimate": 0.01,
            },
        ],
    }

    out = _credible_route_compare_decision(
        route_pred,
        repair_unseen_pred,
        credible_route_enable=True,
        macro_enabled=False,
        max_repair_prob=0.35,
        min_build_margin=0.05,
    )

    assert out["trained"] is True
    assert out["credible_route_enable"] is True
    assert out["credible_route_used"] is True
    assert out["legacy_best_route"] == "build"
    assert out["best_route"] == "repair"
    assert out["source"] == "credible_repair_preferred"
    assert out["repair_unseen_trained"] is True
    assert out["repair_unseen_upside"] == 0.20
    assert out["repair_unseen_best_opportunity_id"] == "opp_a"
    assert out["repair_observed_score"] == -0.40
    assert out["build_observed_score"] == -0.25
    assert out["repair_credible_score"] == -0.20
    assert out["build_credible_score"] == -0.25
    assert out["margin_estimate"] > 0.0
    assert out["repair_prob"] > 0.5
    assert out["veto_repair"] is False
    assert out["exact_margin"] == -0.20


def test_credible_route_compare_keeps_build_veto_when_unseen_upside_is_too_small():
    route_pred = _route_pred(
        repair_eff=0.35,
        build_eff=0.20,
        best_route="build",
        repair_prob=0.12,
        margin_estimate=-0.15,
        exact_margin=-0.15,
    )
    repair_unseen_pred = {
        "trained": True,
        "rows": [
            {
                "route_source": "repair",
                "beam_id": "beam_a",
                "opportunity_id": "opp_a",
                "budget_remaining": 2,
                "acquisition_estimate": 0.05,
                "expected_gain_next_under_executor": 0.08,
                "fragility_prob": 0.02,
                "cost_estimate": 0.01,
            }
        ],
    }

    out = _credible_route_compare_decision(
        route_pred,
        repair_unseen_pred,
        credible_route_enable=True,
        macro_enabled=False,
        max_repair_prob=0.35,
        min_build_margin=0.05,
    )

    assert out["credible_route_used"] is True
    assert out["best_route"] == "build"
    assert out["source"] == "credible_build_veto"
    assert out["repair_unseen_upside"] == 0.05
    assert out["margin_estimate"] < -0.05
    assert out["repair_prob"] <= 0.35
    assert out["veto_repair"] is True


def test_credible_route_compare_falls_back_to_legacy_when_unseen_prediction_is_missing():
    route_pred = _route_pred(
        repair_eff=0.30,
        build_eff=0.20,
        best_route="build",
        repair_prob=0.20,
        margin_estimate=-0.08,
        exact_margin=-0.08,
    )

    legacy = _repair_route_compare_decision(
        route_pred,
        macro_enabled=False,
        max_repair_prob=0.35,
        min_build_margin=0.05,
    )
    out = _credible_route_compare_decision(
        route_pred,
        {"trained": False, "rows": []},
        credible_route_enable=True,
        macro_enabled=False,
        max_repair_prob=0.35,
        min_build_margin=0.05,
    )

    assert out["credible_route_enable"] is True
    assert out["credible_route_used"] is False
    assert out["trained"] == legacy["trained"]
    assert out["best_route"] == legacy["best_route"]
    assert out["repair_prob"] == legacy["repair_prob"]
    assert out["build_prob"] == legacy["build_prob"]
    assert out["margin_estimate"] == legacy["margin_estimate"]
    assert out["veto_repair"] == legacy["veto_repair"]
    assert out["source"] == legacy["source"]


def test_credible_route_preview_rows_ignore_post_allocation_final_slate():
    candidate_meta = {
        "repair_opportunity_slate": [
            {"opportunity_id": "preview_1", "route_source": "repair", "budget_remaining": 2},
            {"opportunity_id": "preview_2", "route_source": "repair", "budget_remaining": 1},
        ],
        "repair_opportunity_slate_final": [
            {"opportunity_id": "final_1", "route_source": "repair", "budget_remaining": 0},
        ],
    }

    rows = _credible_route_preview_repair_opportunity_rows(candidate_meta)

    assert [row["opportunity_id"] for row in rows] == ["preview_1", "preview_2"]
