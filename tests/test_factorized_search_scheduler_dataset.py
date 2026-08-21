# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import json
import math

import pytest

from nestynet_sr.sr_search.factorized_search.opportunity_dataset import build_opportunity_shadow_dataset
from nestynet_sr.sr_search.factorized_search.scheduler_dataset import (
    augment_scheduler_shadow_rows,
    build_scheduler_shadow_dataset,
    load_scheduler_dataset_rows,
    summarize_scheduler_group_integrity,
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
                    "opportunity_id": "hole_demo",
                    "decision_id": "hole_demo",
                    "decision_context_id": "decision_demo",
                    "beam_id": "hole_demo:0",
                    "parent_expr": "parent_demo",
                    "parent_key": "parent_demo",
                    "action": "hole_search",
                    "path": [2],
                    "path_source": "hole_frontier",
                    "target_mode": "identity",
                    "method_name": "hole_frontier",
                    "subroute": "frontier",
                    "evidence_level": "exact_known",
                    "parent_depth": 7,
                    "parent_eff_mse": 1.0,
                    "current_best_route_eff_mse": 1.0,
                    "current_best_child_eff_mse": 0.90,
                    "hole_best_exact_eff_mse": 0.90,
                    "budget_exact_spent": 0,
                    "budget_remaining": 2,
                    "candidate_count_observed": 1,
                    "candidate_count_unique": 1,
                    "preview_candidate_count_total": 2,
                    "preview_candidate_count_unique_total": 2,
                    "predicted_value": 0.25,
                    "predicted_cost": 0.10,
                    "preview_solvability": 0.60,
                    "observed_wall_seconds": 0.3,
                    "observed_exact_evals": 1,
                    "observed_preview_evals": 2,
                },
                route_source="hole",
            ),
        ],
    }


def test_build_scheduler_shadow_dataset_adds_budgeted_log_labels_and_group_metadata():
    base = build_opportunity_shadow_dataset(
        [_shadow_source_row()],
        budget_ladder=(0, 1, 2),
        include_repair=True,
        include_build=True,
        include_hole=True,
    )
    rows = [dict(row) for row in base["rows"] if int(row.get("shadow_prefix_index", -1)) == 0]
    for row in rows:
        row["decision_context_id"] = "decision_demo"
        row["current_best_route_eff_mse"] = 1.0

    payload = build_scheduler_shadow_dataset(
        rows,
        budget_ladder=(1, 2),
        threshold_ladder=(0.25, 0.5),
    )

    assert payload["mode"] == "scheduler_shadow_dataset"
    assert payload["n_rows"] == len(payload["rows"])
    assert payload["budget_ladder"] == [1, 2]
    assert payload["threshold_ladder"] == [0.25, 0.5]

    build_row = next(row for row in payload["rows"] if row["route_source"] == "build")
    repair_row = next(
        row for row in payload["rows"] if row["route_source"] == "repair" and row["path"] == [1]
    )
    hole_row = next(row for row in payload["rows"] if row["route_source"] == "hole")

    assert build_row["decision_group_id"] == "decision_demo"
    assert repair_row["source_row_family_id"]
    assert build_row["future_best_route_eff_mse_at_budget_1_under_executor"] == pytest.approx(0.30)
    assert build_row["delta_log_eff_at_budget_1"] == pytest.approx(-math.log(0.30), rel=1.0e-5)
    assert build_row["improve_ge_0p25_at_budget_1"] == 1.0
    assert build_row["tail_gain_0p25_at_budget_1"] == pytest.approx(build_row["delta_log_eff_at_budget_1"] - 0.25)
    assert build_row["cost_exact_at_budget_2"] == pytest.approx(2.0)
    assert build_row["cost_exact_label_source_at_budget_2"] == "budget_exact_heuristic"
    assert build_row["cost_exact_observed_mask_at_budget_2"] == pytest.approx(0.0)
    assert hole_row["cost_wall_label_source_at_budget_2"] == "observed_scaled"
    assert hole_row["cost_wall_observed_mask_at_budget_2"] == pytest.approx(1.0)
    assert hole_row["cost_total_at_budget_2"] == pytest.approx(4.0)

    assert build_row["route_win_at_budget_1"] == 1.0
    assert repair_row["route_win_at_budget_1"] == 0.0
    assert hole_row["route_win_at_budget_1"] == 0.0
    assert build_row["best_alt_route_eff_mse"] == pytest.approx(0.55)


def test_build_scheduler_shadow_dataset_stock_path_emits_hole_rows():
    payload = build_scheduler_shadow_dataset(
        [_shadow_source_row()],
        budget_ladder=(1, 2),
        threshold_ladder=(0.25, 0.5),
        include_repair=True,
        include_build=True,
        include_hole=True,
    )

    hole_rows = [row for row in payload["rows"] if row["route_source"] == "hole"]
    assert hole_rows
    assert any(row["decision_group_id"] for row in hole_rows)
    assert any("future_best_route_eff_mse_at_budget_1_under_executor" in row for row in hole_rows)


def test_scheduler_shadow_rows_mark_heuristic_wall_cost_as_unobserved():
    rows = augment_scheduler_shadow_rows(
        [
            shared_opportunity_row_dict(
                {
                    "route_source": "build",
                    "opportunity_type": "build_action",
                    "opportunity_id": "build_heuristic_cost",
                    "decision_id": "decision_heuristic_cost",
                    "decision_context_id": "decision_heuristic_cost",
                    "beam_id": "build_heuristic_cost:0",
                    "parent_key": "parent_heuristic_cost",
                    "parent_expr": "parent_heuristic_cost",
                    "action": "replace",
                    "path": [0],
                    "method_name": "build",
                    "subroute": "tuple",
                    "current_best_route_eff_mse": 1.0,
                    "parent_eff_mse": 1.1,
                    "budget_exact_spent": 1,
                    "budget_remaining": 2,
                    "preview_candidate_count_total": 2,
                    "preview_candidate_count_unique_total": 2,
                    "observed_exact_evals": 0,
                    "observed_preview_evals": 2,
                    "cost_estimate": 0.75,
                    "expected_gain_at_budget_1_under_executor": 0.40,
                    "expected_gain_at_budget_1_under_oracle_executor": 0.42,
                    "expected_gain_at_budget_2_under_executor": 0.50,
                    "expected_gain_at_budget_2_under_oracle_executor": 0.52,
                },
                route_source="build",
            )
        ],
        budget_ladder=(1, 2),
        threshold_ladder=(0.25,),
    )

    row = rows[0]
    assert row["cost_wall_at_budget_1"] == pytest.approx(0.75)
    assert row["cost_wall_label_source_at_budget_1"] == "heuristic_estimate_scaled"
    assert row["cost_wall_observed_mask_at_budget_1"] == pytest.approx(0.0)
    assert row["cost_exact_at_budget_1"] == pytest.approx(1.0)
    assert row["cost_exact_label_source_at_budget_1"] == "budget_exact_heuristic"
    assert row["cost_exact_observed_mask_at_budget_1"] == pytest.approx(0.0)


def test_scheduler_shadow_rows_group_by_shared_scheduler_context_when_route_contexts_differ():
    rows = augment_scheduler_shadow_rows(
        [
            shared_opportunity_row_dict(
                {
                    "route_source": "build",
                    "opportunity_type": "build_action",
                    "opportunity_id": "build_scheduler_group",
                    "decision_id": "build_decision_group",
                    "decision_context_id": "build_context_group",
                    "scheduler_decision_context_id": "scheduler_context_group",
                    "beam_id": "build_scheduler_group:0",
                    "parent_key": "parent_group",
                    "parent_expr": "parent_group",
                    "route_decision_id": "build_decision_group",
                    "route_decision_context_id": "build_context_group",
                    "action": "replace",
                    "path": [0],
                    "method_name": "build",
                    "subroute": "tuple",
                    "current_best_route_eff_mse": 1.0,
                    "parent_eff_mse": 1.0,
                    "budget_exact_spent": 0,
                    "budget_remaining": 1,
                    "expected_gain_at_budget_1_under_executor": 0.70,
                    "expected_gain_at_budget_1_under_oracle_executor": 0.70,
                },
                route_source="build",
            ),
            shared_opportunity_row_dict(
                {
                    "route_source": "repair",
                    "opportunity_type": "repair_path",
                    "opportunity_id": "repair_scheduler_group",
                    "decision_id": "repair_decision_group",
                    "decision_context_id": "repair_context_group",
                    "scheduler_decision_context_id": "scheduler_context_group",
                    "beam_id": "repair_scheduler_group:0",
                    "parent_key": "parent_group",
                    "parent_expr": "parent_group",
                    "route_decision_id": "repair_decision_group",
                    "route_decision_context_id": "repair_context_group",
                    "path": [1],
                    "target_mode": "identity",
                    "method_name": "repair",
                    "subroute": "inverse",
                    "current_best_route_eff_mse": 1.0,
                    "parent_eff_mse": 1.0,
                    "budget_exact_spent": 0,
                    "budget_remaining": 1,
                    "expected_gain_at_budget_1_under_executor": 0.20,
                    "expected_gain_at_budget_1_under_oracle_executor": 0.20,
                },
                route_source="repair",
            ),
        ],
        budget_ladder=(1,),
        threshold_ladder=(0.25,),
    )

    assert {row["decision_group_id"] for row in rows} == {"scheduler_context_group"}
    assert {row["decision_context_id"] for row in rows} == {"scheduler_context_group"}
    assert {row["route_decision_context_id"] for row in rows} == {
        "build_context_group",
        "repair_context_group",
    }
    build_row = next(row for row in rows if row["route_source"] == "build")
    repair_row = next(row for row in rows if row["route_source"] == "repair")
    assert build_row["route_win_at_budget_1"] == 1.0
    assert repair_row["route_win_at_budget_1"] == 0.0
    assert all(not bool(row["decision_group_degraded"]) for row in rows)
    assert {row["decision_grouping_source"] for row in rows} == {"scheduler_context"}


def test_scheduler_shadow_rows_mark_route_local_fallback_groups_as_degraded():
    rows = augment_scheduler_shadow_rows(
        [
            shared_opportunity_row_dict(
                {
                    "route_source": "build",
                    "opportunity_type": "build_action",
                    "opportunity_id": "build_fallback_group",
                    "decision_id": "build_decision_fallback",
                    "decision_context_id": "build_context_fallback",
                    "beam_id": "build_fallback_group:0",
                    "parent_key": "parent_fallback",
                    "parent_expr": "parent_fallback",
                    "action": "replace",
                    "path": [0],
                    "method_name": "build",
                    "subroute": "tuple",
                    "current_best_route_eff_mse": 1.0,
                    "parent_eff_mse": 1.0,
                    "budget_exact_spent": 0,
                    "budget_remaining": 1,
                    "expected_gain_at_budget_1_under_executor": 0.20,
                    "expected_gain_at_budget_1_under_oracle_executor": 0.20,
                },
                route_source="build",
            )
        ],
        budget_ladder=(1,),
        threshold_ladder=(0.25,),
    )

    row = rows[0]
    assert row["decision_group_degraded"] is True
    assert row["decision_grouping_source"] == "route_local_fallback"
    summary = summarize_scheduler_group_integrity(rows)
    assert summary["n_degraded_decision_groups"] == 1
    assert summary["decision_grouping_source_counts"]["route_local_fallback"] == 1


def test_scheduler_shadow_dataset_backfills_observed_plan_costs_from_outcome_log():
    base = build_opportunity_shadow_dataset(
        [_shadow_source_row()],
        budget_ladder=(0, 1, 2),
        include_repair=True,
        include_build=True,
        include_hole=True,
    )
    build_prefix0 = next(
        row
        for row in base["rows"]
        if row["route_source"] == "build" and int(row.get("shadow_prefix_index", -1)) == 0
    )
    payload = build_scheduler_shadow_dataset(
        {
            "mode": str(base["mode"]),
            "rows": list(base["rows"]),
            "scheduler_outcome_log": [
                {
                    "decision_context_id": build_prefix0["decision_context_id"],
                    "route": build_prefix0["route_source"],
                    "opportunity_id": build_prefix0["opportunity_id"],
                    "budget_exact_spent": build_prefix0["budget_exact_spent"],
                    "exact_budget": 2,
                    "executed": True,
                    "scheduler_applied": True,
                    "realized_wall_seconds": 3.5,
                    "realized_exact_evals": 5,
                    "realized_preview_evals": 7,
                    "realized_micro_tokens": 11,
                    "realized_widen_tokens": 13,
                    "realized_witness_energy_total_before": 0.8,
                    "realized_witness_energy_total_after": 0.3,
                    "realized_witness_energy_delta": 0.5,
                }
            ],
        },
        budget_ladder=(1, 2),
        threshold_ladder=(0.25, 0.5),
    )

    build_row = next(
        row
        for row in payload["rows"]
        if row["route_source"] == "build"
        and row["opportunity_id"] == build_prefix0["opportunity_id"]
        and int(row.get("shadow_prefix_index", -1)) == 0
    )

    assert build_row["cost_wall_at_budget_2"] == pytest.approx(3.5)
    assert build_row["cost_wall_label_source_at_budget_2"] == "observed_outcome"
    assert build_row["cost_wall_observed_mask_at_budget_2"] == pytest.approx(1.0)
    assert build_row["cost_exact_at_budget_2"] == pytest.approx(5.0)
    assert build_row["cost_exact_label_source_at_budget_2"] == "observed_outcome"
    assert build_row["cost_exact_observed_mask_at_budget_2"] == pytest.approx(1.0)
    assert build_row["cost_total_at_budget_2"] == pytest.approx(36.0)
    assert build_row["cost_total_label_source_at_budget_2"] == "observed_outcome"
    assert build_row["witness_energy_total_before_at_budget_2"] == pytest.approx(0.8)
    assert build_row["witness_energy_total_after_at_budget_2"] == pytest.approx(0.3)
    assert build_row["witness_energy_delta_at_budget_2"] == pytest.approx(0.5)
    assert build_row["witness_energy_label_source_at_budget_2"] == "observed_outcome"
    assert build_row["witness_energy_observed_mask_at_budget_2"] == pytest.approx(1.0)
    assert payload["source_meta"]["n_rows_with_observed_cost_backfill"] >= 1
    assert payload["source_meta"]["n_budget_backfills"] >= 1
    assert payload["source_meta"]["n_rows_with_observed_witness_backfill"] >= 1
    assert payload["source_meta"]["n_budget_witness_backfills"] >= 1


def test_load_scheduler_dataset_rows_preserves_payload_level_observed_cost_backfill(tmp_path):
    base = build_opportunity_shadow_dataset(
        [_shadow_source_row()],
        budget_ladder=(0, 1),
        include_repair=True,
        include_build=True,
        include_hole=True,
    )
    hole_prefix0 = next(
        row
        for row in base["rows"]
        if row["route_source"] == "hole" and int(row.get("shadow_prefix_index", -1)) == 0
    )
    dataset_path = tmp_path / "scheduler_source.json"
    dataset_path.write_text(
        json.dumps(
            {
                "mode": str(base["mode"]),
                "budget_ladder": [1, 2],
                "threshold_ladder": [0.25],
                "rows": list(base["rows"]),
                "scheduler_outcome_log": [
                    {
                        "decision_context_id": hole_prefix0["decision_context_id"],
                        "route": hole_prefix0["route_source"],
                        "opportunity_id": hole_prefix0["opportunity_id"],
                        "budget_exact_spent": hole_prefix0["budget_exact_spent"],
                        "exact_budget": 2,
                        "executed": True,
                        "scheduler_applied": True,
                        "realized_wall_seconds": 1.25,
                        "realized_exact_evals": 2,
                        "realized_preview_evals": 3,
                        "realized_micro_tokens": 5,
                        "realized_widen_tokens": 7,
                        "realized_witness_energy_total_before": 0.6,
                        "realized_witness_energy_total_after": 0.25,
                        "realized_witness_energy_delta": 0.35,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = load_scheduler_dataset_rows([dataset_path])
    hole_row = next(
        row
        for row in rows
        if row["route_source"] == "hole"
        and row["opportunity_id"] == hole_prefix0["opportunity_id"]
        and int(row.get("shadow_prefix_index", -1)) == 0
    )
    assert hole_row["cost_wall_at_budget_2"] == pytest.approx(1.25)
    assert hole_row["cost_exact_at_budget_2"] == pytest.approx(2.0)
    assert hole_row["cost_total_at_budget_2"] == pytest.approx(17.0)
    assert hole_row["witness_energy_total_before_at_budget_2"] == pytest.approx(0.6)
    assert hole_row["witness_energy_total_after_at_budget_2"] == pytest.approx(0.25)
    assert hole_row["witness_energy_delta_at_budget_2"] == pytest.approx(0.35)
