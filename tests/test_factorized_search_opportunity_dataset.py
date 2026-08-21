# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import pytest

from nestynet_sr.sr_search.factorized_search.opportunity_dataset import build_opportunity_shadow_dataset
from nestynet_sr.sr_search.factorized_search.opportunity_shadow_eval import (
    OpportunityShadowEvalConfig,
    build_sampled_opportunity_shadow_dataset,
    shadow_sample_probability_for_depth,
)
from nestynet_sr.sr_search.factorized_search.shared_opportunity import shared_opportunity_row_dict


def _shadow_source_row() -> dict:
    return {
        "spec_id": "shadow_demo",
        "truth_depth": 7,
        "parent_expr": "parent_demo",
        "estimated_parent_eff_mse": 1.0,
        "inverse_repair_slate_id": "repair_slate_demo",
        "inverse_repair_slate": [
            {
                "beam_rank": 0,
                "path": [1],
                "target_mode": "identity",
                "local_rank": 0,
                "tuple_allocation_estimate": 0.90,
                "local_probe_mse": 0.55,
                "local_fit_mse": 0.50,
                "child_key": "repair_bad",
                "child_expr": "repair_bad",
                "child_eff_mse": 0.55,
                "child_raw_mse": 0.56,
                "dedup_kept": True,
                "exact_child_score_observed": True,
                "oracle_mapping_fragile": True,
            },
            {
                "beam_rank": 0,
                "path": [1],
                "target_mode": "identity",
                "local_rank": 1,
                "tuple_allocation_estimate": 0.40,
                "local_probe_mse": 0.20,
                "local_fit_mse": 0.18,
                "child_key": "repair_good",
                "child_expr": "repair_good",
                "child_eff_mse": 0.25,
                "child_raw_mse": 0.26,
                "dedup_kept": True,
                "exact_child_score_observed": True,
                "oracle_mapping_stable": True,
            },
            {
                "beam_rank": 1,
                "path": [2],
                "target_mode": "full",
                "local_rank": 0,
                "local_probe_mse": 0.40,
                "local_fit_mse": 0.38,
                "child_key": "repair_other",
                "child_expr": "repair_other",
                "child_eff_mse": 0.40,
                "child_raw_mse": 0.41,
                "dedup_kept": True,
                "exact_child_score_observed": True,
            },
        ],
        "controller_build_slate_id": "build_slate_demo",
        "controller_build_slate": [
            {
                "action": "replace",
                "path": [0],
                "path_source": "critic_path_head",
                "child_key": "build_good",
                "child_expr": "build_good",
                "child_eff_mse": 0.30,
                "child_raw_mse": 0.31,
                "exact_child_score_observed": True,
                "status": "scored",
            }
        ],
        "hole_opportunity_slate_id": "hole_slate_demo",
        "hole_opportunity_slate": [
            shared_opportunity_row_dict(
                {
                    "route_source": "hole",
                    "opportunity_type": "hole_opportunity",
                    "opportunity_id": "hole_demo_exact",
                    "decision_id": "hole_decision_demo",
                    "decision_context_id": "hole_ctx_demo",
                    "beam_id": "hole_decision_demo:0",
                    "parent_key": "parent_demo",
                    "parent_expr": "parent_demo",
                    "action": "hole_search",
                    "path": [2],
                    "path_source": "hole_frontier",
                    "target_mode": "identity",
                    "method_name": "archive_mine",
                    "subroute": "path_hole",
                    "evidence_level": "exact_known",
                    "current_best_child_eff_mse": 0.45,
                    "hole_best_exact_eff_mse": 0.45,
                    "parent_eff_mse": 1.0,
                    "budget_exact_spent": 0,
                    "budget_remaining": 3,
                    "candidate_count_observed": 1,
                    "candidate_count_unique": 1,
                    "preview_candidate_count_total": 3,
                    "preview_candidate_count_unique_total": 3,
                    "observed_exact_evals": 1,
                    "observed_preview_evals": 3,
                },
                route_source="hole",
            ),
            shared_opportunity_row_dict(
                {
                    "route_source": "hole",
                    "opportunity_type": "hole_opportunity",
                    "opportunity_id": "hole_demo_preview",
                    "decision_id": "hole_decision_demo",
                    "decision_context_id": "hole_ctx_demo",
                    "beam_id": "hole_decision_demo:1",
                    "parent_key": "parent_demo",
                    "parent_expr": "parent_demo",
                    "action": "hole_search",
                    "path": [3],
                    "path_source": "hole_frontier",
                    "target_mode": "identity",
                    "method_name": "archive_mine",
                    "subroute": "path_hole",
                    "evidence_level": "preview_support",
                    "current_best_child_eff_mse": 0.80,
                    "hole_best_shortlist_eff_mse": 0.80,
                    "parent_eff_mse": 1.0,
                    "budget_exact_spent": 0,
                    "budget_remaining": 2,
                    "candidate_count_observed": 0,
                    "candidate_count_unique": 0,
                    "preview_candidate_count_total": 2,
                    "preview_candidate_count_unique_total": 2,
                    "observed_exact_evals": 0,
                    "observed_preview_evals": 2,
                },
                route_source="hole",
            ),
        ],
    }


def test_build_opportunity_shadow_dataset_emits_sequential_prefix_rows():
    payload = build_opportunity_shadow_dataset(
        [_shadow_source_row()],
        budget_ladder=(0, 1, 2),
        include_repair=True,
        include_build=True,
    )

    assert payload["mode"] == "opportunity_shadow_dataset"
    assert payload["n_source_rows"] == 1
    assert payload["n_rows"] >= 7

    rows = payload["rows"]
    repair_prefix0 = next(
        row
        for row in rows
        if row["route_source"] == "repair"
        and row["path"] == [1]
        and int(row["shadow_prefix_index"]) == 0
    )
    assert repair_prefix0["label_budget_origin"] == "additional_exact_tokens"
    assert repair_prefix0["shadow_source_row_id"] == "repair_slate_demo"
    assert repair_prefix0["budget_exact_spent"] == 0
    assert repair_prefix0["budget_remaining"] == 2
    assert repair_prefix0["expected_gain_next_under_executor"] == pytest.approx(0.45)
    assert repair_prefix0["expected_gain_next_under_oracle_executor"] == pytest.approx(0.75)
    assert repair_prefix0["expected_gain_at_budget_2_under_executor"] == pytest.approx(0.75)
    assert repair_prefix0["coverage_at_budget_0"] == 0.0
    assert repair_prefix0["coverage_at_budget_1"] == 1.0
    assert repair_prefix0["route_flip_at_budget_1"] == 0.0
    assert repair_prefix0["route_flip_at_budget_2"] == 1.0
    assert repair_prefix0["fragility_at_budget_1"] == 1.0
    assert repair_prefix0["stability_at_budget_2"] == 1.0

    repair_prefix1 = next(
        row
        for row in rows
        if row["route_source"] == "repair"
        and row["path"] == [1]
        and int(row["shadow_prefix_index"]) == 1
    )
    assert repair_prefix1["budget_exact_spent"] == 1
    assert repair_prefix1["budget_remaining"] == 1
    assert repair_prefix1["current_best_child_expr"] == "repair_bad"
    assert repair_prefix1["current_best_child_eff_mse"] == 0.55
    assert repair_prefix1["expected_gain_next_under_executor"] == pytest.approx(0.30)
    assert repair_prefix1["expected_gain_next_under_oracle_executor"] == pytest.approx(0.30)

    build_prefix0 = next(
        row
        for row in rows
        if row["route_source"] == "build"
        and row["action"] == "replace"
        and int(row["shadow_prefix_index"]) == 0
    )
    assert build_prefix0["budget_exact_spent"] == 0
    assert build_prefix0["budget_remaining"] == 1
    assert build_prefix0["expected_gain_next_under_executor"] == pytest.approx(0.70)
    assert build_prefix0["coverage_at_budget_1"] == 1.0

    build_prefix1 = next(
        row
        for row in rows
        if row["route_source"] == "build"
        and row["action"] == "replace"
        and int(row["shadow_prefix_index"]) == 1
    )
    assert build_prefix1["budget_exact_spent"] == 1
    assert build_prefix1["budget_remaining"] == 0
    assert build_prefix1["expected_gain_next_under_executor"] == 0.0

    hole_prefix0 = next(
        row
        for row in rows
        if row["route_source"] == "hole"
        and row["opportunity_id"] == "hole_demo_exact"
        and int(row["shadow_prefix_index"]) == 0
    )
    assert hole_prefix0["shadow_source_row_id"] == "repair_slate_demo"
    assert hole_prefix0["expected_gain_next_under_executor"] == pytest.approx(0.55)
    assert hole_prefix0["coverage_at_budget_1"] == 1.0

    hole_prefix1 = next(
        row
        for row in rows
        if row["route_source"] == "hole"
        and row["opportunity_id"] == "hole_demo_exact"
        and int(row["shadow_prefix_index"]) == 1
    )
    assert hole_prefix1["budget_exact_spent"] == 1
    assert hole_prefix1["expected_gain_next_under_executor"] == 0.0

    hole_preview_prefix0 = next(
        row
        for row in rows
        if row["route_source"] == "hole"
        and row["opportunity_id"] == "hole_demo_preview"
        and int(row["shadow_prefix_index"]) == 0
    )
    assert hole_preview_prefix0["evidence_level"] == "preview_support"
    assert hole_preview_prefix0["coverage_at_budget_1"] == 0.0
    assert hole_preview_prefix0["expected_gain_next_under_executor"] == 0.0


def test_shadow_sampling_config_supports_depth_bias_and_full_sampling():
    shallow = dict(_shadow_source_row())
    shallow["truth_depth"] = 3
    shallow["inverse_repair_slate_id"] = "repair_slate_shallow"
    deep = dict(_shadow_source_row())
    deep["truth_depth"] = 8
    deep["inverse_repair_slate_id"] = "repair_slate_deep"

    assert shadow_sample_probability_for_depth(
        8,
        shadow_sample_rate=0.10,
        depth_oversample_min=6,
        depth_oversample_multiplier=2.0,
    ) > shadow_sample_probability_for_depth(
        3,
        shadow_sample_rate=0.10,
        depth_oversample_min=6,
        depth_oversample_multiplier=2.0,
    )

    payload = build_sampled_opportunity_shadow_dataset(
        [shallow, deep],
        config=OpportunityShadowEvalConfig(
            shadow_sample_rate=1.0,
            budget_ladder=(0, 1, 2),
            include_repair=True,
            include_build=False,
            include_hole=True,
        ),
    )
    assert payload["n_source_rows_total"] == 2
    assert payload["n_source_rows_sampled"] == 2
    assert payload["sampling"]["config"]["shadow_sample_rate"] == 1.0
    assert payload["sampling"]["config"]["include_hole"] is True
    assert payload["rows"]
