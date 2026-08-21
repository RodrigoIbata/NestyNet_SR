# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_search.factorized_search.controller_diagnostics import (
    analyze_oracle_opportunity_funnel,
    analyze_oracle_path_diagnostics,
    analyze_search_opportunity_funnel,
    analyze_search_path_diagnostics,
    analyze_search_route_diagnostics,
)


def _fake_predictor(_bundle, row):
    path_rows = list(row.get("path_summaries", []) or [])
    rows = []
    for item in path_rows:
        path = list(item.get("path", []) or [])
        weight = 0.0
        if path == [2]:
            weight = 0.9
        elif path == [1]:
            weight = 0.8
        rows.append({
            "path": path,
            "policy_weight": weight,
        })
    return {
        "path": {
            "trained": True,
            "best_path": [1],
            "rows": rows,
        }
    }


def test_analyze_oracle_path_diagnostics_smoke():
    payload = {
        "rows": [
            {
                "truth_depth": 3,
                "target_path": [1],
                "selected_path": [1],
                "controller_row": {
                    "path_summaries": [
                        {"path": [1], "weighted_rel_gain": 0.9},
                        {"path": [2], "weighted_rel_gain": 0.8},
                    ],
                },
            },
            {
                "truth_depth": 4,
                "target_path": [2],
                "selected_path": [2],
                "controller_row": {
                    "path_summaries": [
                        {"path": [1], "weighted_rel_gain": 0.9},
                        {"path": [2], "weighted_rel_gain": 0.8},
                    ],
                },
            },
        ],
    }

    report = analyze_oracle_path_diagnostics(
        payload,
        bundle={"stub": True},
        predictor=_fake_predictor,
        recall_ks=(1, 2),
    )

    assert report["n_rows"] == 2
    assert report["by_depth"]["3"]["proposal_recall_at_1"] == 1.0
    assert report["by_depth"]["4"]["proposal_recall_at_1"] == 0.0
    assert report["by_depth"]["4"]["proposal_recall_at_2"] == 1.0
    assert report["oracle_target_accuracy"]["path_head_top1"]["n"] == 2
    assert report["oracle_target_accuracy"]["policy_attn_top1"]["n"] == 2


def test_analyze_search_path_diagnostics_uses_executed_path():
    rows = [
        {
            "parent_depth": 3,
            "controller_policy_action": "replace",
            "selected_path": [1],
            "controller_action_path": [2],
            "path_summaries": [
                {"path": [1], "weighted_rel_gain": 0.9},
                {"path": [2], "weighted_rel_gain": 0.8},
            ],
        },
        {
            "parent_depth": 3,
            "controller_policy_action": "replace",
            "controller_action_path": [1],
            "path_summaries": [
                {"path": [1], "weighted_rel_gain": 0.9},
                {"path": [2], "weighted_rel_gain": 0.8},
            ],
        },
    ]

    report = analyze_search_path_diagnostics(
        rows,
        bundle={"stub": True},
        predictor=_fake_predictor,
    )

    assert report["n_rows"] == 2
    assert report["n_rows_with_path_summaries"] == 2
    assert report["executed_path_accuracy"]["path_head_top1"]["n"] == 2
    assert report["executed_path_accuracy"]["policy_attn_top1"]["n"] == 2
    assert report["authority_agreement"]["available"]["executed_path"] == 2


def test_analyze_oracle_opportunity_funnel_prefers_explicit_opportunity_rows():
    payload = {
        "rows": [
            {
                "truth_depth": 6,
                "oracle_truth_path": [1],
                "path_summaries": [
                    {"path": [1], "weighted_rel_gain": 0.9},
                    {"path": [2], "weighted_rel_gain": 0.7},
                ],
                "repair_opportunity_slate": [
                    {"path": [1], "candidate_count_observed": 2, "candidate_count_unique": 2},
                    {"path": [2], "candidate_count_observed": 1, "candidate_count_unique": 1},
                ],
                "inverse_repair_slate": [
                    {"oracle_is_truth_candidate": True, "exact_child_score_observed": True},
                    {"oracle_is_truth_candidate": False, "exact_child_score_observed": False},
                ],
                "build_opportunity_slate": [
                    {"action": "replace", "candidate_count_observed": 1},
                    {"action": "wrap_un", "candidate_count_observed": 1},
                ],
                "controller_build_slate": [
                    {"oracle_is_truth_candidate": False, "exact_child_score_observed": True},
                    {"oracle_is_truth_candidate": True, "exact_child_score_observed": True},
                ],
            }
        ],
    }

    report = analyze_oracle_opportunity_funnel(payload)

    assert report["mode"] == "oracle_opportunity_funnel"
    assert report["n_rows"] == 1
    depth_row = report["by_depth"]["6"]
    assert depth_row["repair_truth_path_in_paths_rate"] == 1.0
    assert depth_row["repair_truth_path_in_opportunities_rate"] == 1.0
    assert depth_row["repair_truth_candidate_in_slate_rate"] == 1.0
    assert depth_row["build_truth_candidate_in_slate_rate"] == 1.0
    assert depth_row["repair_mean_opportunity_count"] == 2.0
    assert depth_row["build_mean_opportunity_count"] == 2.0
    assert depth_row["repair_mean_exact_observed_count"] == 1.0
    assert depth_row["build_mean_exact_observed_count"] == 2.0


def test_analyze_search_opportunity_funnel_uses_logged_opportunity_rows():
    rows = [
        {
            "parent_depth": 5,
            "repair_opportunity_slate": [
                {"path": [1], "candidate_count_observed": 2},
                {"path": [2], "candidate_count_observed": 1},
            ],
            "build_opportunity_slate": [
                {"action": "replace", "candidate_count_observed": 1},
            ],
            "inverse_repair_slate": [
                {"exact_child_score_observed": True},
                {"exact_child_score_observed": False},
            ],
            "controller_build_slate": [
                {"exact_child_score_observed": True},
            ],
        }
    ]

    report = analyze_search_opportunity_funnel(rows)

    assert report["mode"] == "search_opportunity_funnel"
    assert report["n_rows"] == 1
    depth_row = report["by_depth"]["5"]
    assert depth_row["repair_mean_opportunity_count"] == 2.0
    assert depth_row["build_mean_opportunity_count"] == 1.0
    assert depth_row["repair_mean_exact_observed_count"] == 1.0
    assert depth_row["build_mean_exact_observed_count"] == 1.0


def test_analyze_search_route_diagnostics_summarizes_preview_and_reward():
    rows = [
        {
            "parent_depth": 3,
            "status": "scored",
            "route_scheduler_selected_route": "opportunity_expand",
            "route_scheduler_selection_source": "ucb",
            "route_scheduler_best_available_route": "opportunity_expand",
            "route_scheduler_selected_best_preview_route": True,
            "route_scheduler_chosen_route_score": 0.8,
            "route_scheduler_best_available_route_score": 0.8,
            "route_scheduler_preview_gap": 0.0,
            "route_scheduler_realized_raw_reward": 0.4,
            "route_scheduler_realized_adjusted_reward": 0.2,
            "route_scheduler_adjusted_regret_proxy": 0.6,
            "route_scheduler_wall_s": 1.0,
            "route_scheduler_route_rows": [
                {"route": "expression_expand", "route_score": 0.0, "preview_score": 0.0, "learned_bonus": None, "selected": False},
                {"route": "opportunity_expand", "route_score": 0.8, "preview_score": 0.6, "learned_bonus": 0.3, "selected": True},
            ],
        },
        {
            "parent_depth": 4,
            "status": "score_none",
            "route_scheduler_selected_route": "expression_expand",
            "route_scheduler_selection_source": "epsilon",
            "route_scheduler_best_available_route": "opportunity_expand",
            "route_scheduler_selected_best_preview_route": False,
            "route_scheduler_chosen_route_score": 0.0,
            "route_scheduler_best_available_route_score": 0.5,
            "route_scheduler_preview_gap": 0.5,
            "route_scheduler_realized_raw_reward": 0.0,
            "route_scheduler_realized_adjusted_reward": 0.0,
            "route_scheduler_adjusted_regret_proxy": 0.5,
            "route_scheduler_wall_s": 2.0,
            "route_scheduler_route_rows": [
                {"route": "expression_expand", "route_score": 0.0, "preview_score": 0.0, "learned_bonus": None, "selected": True},
                {"route": "opportunity_expand", "route_score": 0.5, "preview_score": 0.4, "learned_bonus": 0.1, "selected": False},
            ],
        },
    ]

    report = analyze_search_route_diagnostics(rows)

    assert report["mode"] == "search_route_diagnostics"
    assert report["n_rows"] == 2
    assert report["n_rows_with_route_diagnostics"] == 2
    assert report["chosen_route_counts"]["expression_expand"] == 1
    assert report["chosen_route_counts"]["opportunity_expand"] == 1
    assert report["best_available_route_counts"]["opportunity_expand"] == 2
    assert report["selected_best_preview_rate"] == 0.5
    assert report["mean_adjusted_regret_proxy"] == 0.55
    assert report["by_route"]["opportunity_expand"]["mean_realized_adjusted_reward"] == 0.2
    assert report["by_depth"]["4"]["mean_preview_gap"] == 0.5
