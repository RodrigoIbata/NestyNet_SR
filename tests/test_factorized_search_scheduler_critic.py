# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import json

import pytest

from nestynet_sr.sr_search.factorized_search.scheduler_critic import (
    SCHEDULER_FEATURE_NAMES,
    SCHEDULER_WITNESS_FEATURE_NAMES,
    _build_feature_tensors,
    evaluate_scheduler_critic,
    load_scheduler_bundle,
    predict_scheduler_plan_slate,
    save_scheduler_bundle,
    scheduler_feature_names,
    scheduler_row_group_id,
    scheduler_feature_vector,
    split_scheduler_rows_grouped,
    train_scheduler_critic,
)
from nestynet_sr.sr_search.factorized_search.scheduler_dataset import augment_scheduler_shadow_rows
from nestynet_sr.sr_search.factorized_search.scheduler_train import run_scheduler_training
from nestynet_sr.sr_search.factorized_search.shared_opportunity import shared_opportunity_row_dict


def _route_template(route: str) -> tuple[str, str, list[int], str, str, str]:
    if route == "repair":
        return ("repair_beam", "inv_steer", [1], "inverse_beam", "inverse", "beam")
    if route == "build":
        return ("build_action", "replace", [0], "critic_path_head", "build", "tuple")
    return ("hole_frontier", "hole_search", [2], "hole_frontier", "hole_frontier", "frontier")


def _synthetic_scheduler_rows(n_groups: int = 18) -> list[dict]:
    rows: list[dict] = []
    route_names = ("repair", "build", "hole")
    for group_idx in range(int(n_groups)):
        dominant = int(group_idx % len(route_names))
        current_eff = 1.0 + (0.05 * float(group_idx % 4))
        for route_idx, route in enumerate(route_names):
            opportunity_type, action, path, path_source, method_name, subroute = _route_template(route)
            preference_bonus = 0.26 if route_idx == dominant else (0.10 if route_idx == ((dominant + 1) % 3) else 0.03)
            base_gain = 0.10 + (0.02 * float(group_idx % 5))
            gain1 = min(0.85 * current_eff, base_gain + (0.40 * preference_bonus))
            gain2 = min(0.90 * current_eff, gain1 + 0.08 + (0.10 * preference_bonus))
            gain4 = min(0.95 * current_eff, gain2 + 0.10 + (0.10 * preference_bonus))
            confidence = 0.25 + (0.55 * preference_bonus) + (0.05 * float(group_idx % 3))
            row = shared_opportunity_row_dict(
                {
                    "route_source": route,
                    "opportunity_type": opportunity_type,
                    "opportunity_id": f"{route}_{group_idx}",
                    "decision_id": f"decision_{group_idx}",
                    "decision_context_id": f"decision_{group_idx}",
                    "beam_id": f"{route}_{group_idx}:0",
                    "parent_expr": f"parent_{group_idx}",
                    "parent_key": f"parent_{group_idx}",
                    "action": action,
                    "path": path,
                    "path_source": path_source,
                    "target_mode": "identity" if route != "build" else "",
                    "method_name": method_name,
                    "subroute": subroute,
                    "evidence_level": "preview_support" if group_idx % 2 else "exact_known",
                    "parent_depth": 4 + (group_idx % 5),
                    "parent_eff_mse": current_eff + 0.15,
                    "current_best_route_eff_mse": current_eff,
                    "current_best_child_eff_mse": max(0.0, current_eff - gain1),
                    "budget_exact_spent": 0,
                    "budget_remaining": 4,
                    "candidate_count_observed": 0,
                    "candidate_count_unique": 0,
                    "preview_candidate_count_total": 3,
                    "preview_candidate_count_unique_total": 3,
                    "shadow_total_exact_available": 4,
                    "shadow_total_preview_available": 4,
                    "best_preview_probe_mse": max(0.0, current_eff - (0.6 * gain1)),
                    "best_preview_fit_mse": max(0.0, current_eff - (0.7 * gain1)),
                    "best_tuple_utility_estimate": gain2 if route == "build" else gain1,
                    "best_tuple_allocation_estimate": gain1 if route != "hole" else 0.5 * gain1,
                    "path_gain": gain2,
                    "path_gain_pre_cut": gain4,
                    "rel_gain": gain1 / current_eff,
                    "transport_rel": 0.10 + (0.20 * preference_bonus),
                    "lin_rel": 0.05 + (0.10 * preference_bonus),
                    "valid_frac": 0.35 + (0.40 * preference_bonus),
                    "confidence": confidence,
                    "effective_n": 8 + group_idx,
                    "branch_factor": 0.8 + (0.1 * preference_bonus),
                    "cut_factor": 0.6 + (0.2 * preference_bonus),
                    "branch_support": 0.15 + (0.60 * preference_bonus),
                    "family_scale": 1.0 + (0.1 * float(group_idx % 4)),
                    "observed_wall_seconds": 0.2 + (0.05 * route_idx) + (0.01 * group_idx),
                    "observed_exact_evals": 1 + route_idx,
                    "observed_preview_evals": 2 + (group_idx % 3),
                    "predicted_value": gain4 if route == "hole" else 0.0,
                    "predicted_cost": 0.12 + (0.04 * route_idx),
                    "preview_solvability": 0.20 + (0.60 * preference_bonus if route == "hole" else 0.10),
                    "preview_recursive_depth": 1 + (group_idx % 3) if route == "hole" else 0,
                    "expected_gain_at_budget_1_under_executor": gain1,
                    "expected_gain_at_budget_1_under_oracle_executor": min(current_eff, gain1 + 0.03),
                    "expected_gain_at_budget_2_under_executor": gain2,
                    "expected_gain_at_budget_2_under_oracle_executor": min(current_eff, gain2 + 0.04),
                    "expected_gain_at_budget_4_under_executor": gain4,
                    "expected_gain_at_budget_4_under_oracle_executor": min(current_eff, gain4 + 0.05),
                    "new_residual_basin_at_budget_1": 1.0 if route == "hole" and route_idx == dominant else 0.0,
                    "new_residual_basin_at_budget_2": 1.0 if route == "hole" and route_idx == dominant else 0.0,
                    "new_residual_basin_at_budget_4": 1.0 if route == "hole" and route_idx == dominant else 0.0,
                    "fragility_at_budget_1": 1.0 if route == "build" and route_idx != dominant else 0.0,
                    "fragility_at_budget_2": 1.0 if route == "build" and route_idx != dominant else 0.0,
                    "fragility_at_budget_4": 1.0 if route == "build" and route_idx != dominant else 0.0,
                    "stability_at_budget_1": 1.0 if route != "build" and route_idx == dominant else 0.0,
                    "stability_at_budget_2": 1.0 if route != "build" and route_idx == dominant else 0.0,
                    "stability_at_budget_4": 1.0 if route != "build" and route_idx == dominant else 0.0,
                },
                route_source=route,
            )
            rows.append(row)
    return augment_scheduler_shadow_rows(
        rows,
        budget_ladder=(1, 2, 4),
        threshold_ladder=(0.1, 0.25, 0.5),
    )


def _attach_witness_labels(rows: list[dict]) -> list[dict]:
    for row in rows:
        route = str(row.get("route_source", "") or "")
        route_scale = 1.0 if route == "hole" else (0.7 if route == "repair" else 0.35)
        for budget in (1, 2, 4):
            before = 0.9
            delta = route_scale * (0.08 + (0.04 * float(budget)))
            row[f"witness_energy_total_before_at_budget_{int(budget)}"] = before
            row[f"witness_energy_total_after_at_budget_{int(budget)}"] = before - delta
            row[f"witness_energy_delta_at_budget_{int(budget)}"] = delta
            row[f"witness_energy_label_source_at_budget_{int(budget)}"] = "observed_outcome"
            row[f"witness_energy_observed_mask_at_budget_{int(budget)}"] = 1.0
    return rows


def test_split_scheduler_rows_grouped_keeps_decision_groups_together():
    rows = _synthetic_scheduler_rows(12)
    train_rows, val_rows = split_scheduler_rows_grouped(rows, val_fraction=0.25, seed=7)

    assert train_rows
    assert val_rows
    train_groups = {scheduler_row_group_id(row) for row in train_rows}
    val_groups = {scheduler_row_group_id(row) for row in val_rows}
    assert train_groups.isdisjoint(val_groups)


def test_scheduler_row_group_id_prefers_shared_scheduler_context():
    rows = _synthetic_scheduler_rows(1)
    seen = set()
    for row in rows:
        route = str(row.get("route_source", "") or "")
        row["decision_id"] = f"{route}_decision_local"
        row["decision_context_id"] = f"{route}_context_local"
        row["route_decision_id"] = str(row["decision_id"])
        row["route_decision_context_id"] = str(row["decision_context_id"])
        row["scheduler_decision_context_id"] = "scheduler_context_critic"
        row["decision_group_id"] = "scheduler_context_critic"
        seen.add(scheduler_row_group_id(row))
    assert seen == {"scheduler_context_critic"}


def test_train_predict_and_save_scheduler_critic_round_trip(tmp_path):
    rows = _synthetic_scheduler_rows()
    bundle = train_scheduler_critic(
        rows,
        hidden_dim=32,
        epochs=60,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=11,
        ensemble_size=2,
    )

    assert bundle["scheduler_critic_trained"] is True
    assert bundle["model_kind"] == "opportunity_scheduler_v2"
    assert bundle["feature_schema_version"] == 1
    assert bundle["witness_energy_feature_enable"] is False
    assert bundle["ensemble_size"] == 2
    assert bundle["metrics"]["train"]["n_rows"] > 0

    out_path = tmp_path / "scheduler_bundle.pt"
    save_scheduler_bundle(bundle, out_path)
    loaded = load_scheduler_bundle(out_path)
    pred = predict_scheduler_plan_slate(loaded, rows[:9])

    assert pred["trained"] is True
    assert pred["feature_schema_version"] == 1
    assert pred["witness_energy_feature_enable"] is False
    assert pred["best_route"] in {"repair", "build", "hole"}
    assert pred["best_budget"] in {1, 2, 4}
    assert pred["rows"]

    best_row = pred["rows"][0]
    assert "break_prob_0p25_at_budget_1" in best_row
    assert "tail_gain_0p25_pred_at_budget_1" in best_row
    assert "route_win_prob_at_budget_4" in best_row
    assert "cost_exact_pred_at_budget_4" in best_row
    assert "acquisition_sigma_at_budget_4" in best_row
    assert best_row["break_prob_0p25_at_budget_1"] <= best_row["break_prob_0p25_at_budget_2"] <= best_row["break_prob_0p25_at_budget_4"]
    assert best_row["cost_exact_pred_at_budget_1"] <= best_row["cost_exact_pred_at_budget_2"] <= best_row["cost_exact_pred_at_budget_4"]

    metrics = evaluate_scheduler_critic(loaded, rows)
    assert metrics["n_rows"] == len(rows)
    assert metrics["break_brier"]["1"] is not None
    assert metrics["tail_mae"]["1"] is not None
    assert metrics["cost_exact_mae"]["1"] is not None
    assert metrics["top1_route_win_accuracy"] is not None
    assert metrics["break_monotonic_violation_rate"] == pytest.approx(0.0)
    assert metrics["tail_monotonic_violation_rate"] == pytest.approx(0.0)
    assert metrics["cost_exact_monotonic_violation_rate"] == pytest.approx(0.0)
    assert metrics["route_budget_metrics"]["hole"]["1"]["count"] > 0
    assert metrics["route_budget_metrics"]["build"]["1"]["break_brier"] is not None


def test_scheduler_feature_vector_adds_witness_energy_features_when_enabled():
    row = shared_opportunity_row_dict(
        {
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "opportunity_id": "repair_witness_feat",
            "decision_id": "decision_witness_feat",
            "decision_context_id": "decision_witness_feat",
            "beam_id": "repair_witness_feat:0",
            "parent_expr": "parent_witness_feat",
            "parent_key": "parent_witness_feat",
            "action": "inv_steer",
            "path": [1],
            "path_source": "inverse_beam",
            "target_mode": "identity",
            "method_name": "inverse",
            "subroute": "beam",
            "witness_value_loss": 0.6,
            "witness_grad_loss": 0.2,
            "witness_energy_total": 0.8,
            "witness_energy_delta_estimate": 0.3,
        },
        route_source="repair",
    )
    feature_names = scheduler_feature_names(witness_energy_feature_enable=True)
    values = scheduler_feature_vector(row, feature_names=feature_names)
    feature_map = dict(zip(feature_names, values))

    assert set(SCHEDULER_WITNESS_FEATURE_NAMES).issubset(feature_map.keys())
    assert feature_map["witness_value_log_loss"] > 0.0
    assert feature_map["witness_grad_log_loss"] > 0.0
    assert feature_map["witness_energy_total_log"] > 0.0
    assert feature_map["witness_energy_delta_estimate"] == pytest.approx(0.3)
    assert feature_map["witness_grad_present"] == pytest.approx(1.0)


def test_scheduler_critic_round_trip_with_witness_energy_feature_schema(tmp_path):
    rows = _attach_witness_labels(_synthetic_scheduler_rows(12))
    bundle = train_scheduler_critic(
        rows,
        hidden_dim=24,
        epochs=20,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=17,
        ensemble_size=1,
        witness_energy_feature_enable=True,
        objective_mode="witness",
    )

    assert bundle["witness_energy_feature_enable"] is True
    assert bundle["feature_schema_version"] == 2
    assert bundle["objective_mode"] == "witness"
    assert set(SCHEDULER_WITNESS_FEATURE_NAMES).issubset(set(bundle["feature_names"]))

    out_path = tmp_path / "scheduler_bundle_witness.pt"
    save_scheduler_bundle(bundle, out_path)
    loaded = load_scheduler_bundle(out_path)
    pred = predict_scheduler_plan_slate(loaded, rows[:6])

    assert loaded["witness_energy_feature_enable"] is True
    assert loaded["feature_schema_version"] == 2
    assert loaded["objective_mode"] == "witness"
    assert pred["trained"] is True
    assert pred["witness_energy_feature_enable"] is True
    assert pred["feature_schema_version"] == 2
    assert pred["objective_mode"] == "witness"
    assert "witness_delta_pred_at_budget_1" in pred["rows"][0]
    assert "cost_total_pred_at_budget_1" in pred["rows"][0]
    assert pred["rows"][0]["objective_estimate_at_budget_1"] == pytest.approx(
        pred["rows"][0]["witness_rate_pred_at_budget_1"]
    )


def test_run_scheduler_training_writes_bundle_and_summary(tmp_path):
    rows = _synthetic_scheduler_rows(12)
    dataset_path = tmp_path / "scheduler_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "mode": "scheduler_shadow_dataset",
                "budget_ladder": [1, 2, 4],
                "threshold_ladder": [0.1, 0.25, 0.5],
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "scheduler_bundle.pt"

    summary = run_scheduler_training(
        dataset_paths=[str(dataset_path)],
        output_path=str(output_path),
        hidden_dim=32,
        epochs=40,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=3,
        ensemble_size=2,
    )

    assert output_path.exists()
    assert output_path.with_suffix(output_path.suffix + ".json").exists()
    assert summary["witness_energy_feature_enable"] is False
    assert summary["objective_mode"] == "acquisition"
    assert summary["metrics"]["train"]["n_rows"] > 0
    assert "training_rebalance" in summary
    assert summary["full_eval"]["break_brier"]["1"] is not None
    assert summary["sample_prediction"]["trained"] is True


def test_run_scheduler_training_persists_route_aliases(tmp_path):
    rows = _synthetic_scheduler_rows(8)
    dataset_path = tmp_path / "scheduler_dataset_alias.json"
    dataset_path.write_text(
        json.dumps(
            {
                "mode": "scheduler_shadow_dataset",
                "budget_ladder": [1, 2, 4],
                "threshold_ladder": [0.1, 0.25, 0.5],
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "scheduler_bundle_alias.pt"

    summary = run_scheduler_training(
        dataset_paths=[str(dataset_path)],
        output_path=str(output_path),
        budget_ladder=[1, 2, 4],
        threshold_ladder=[0.1, 0.25, 0.5],
        route_aliases={"hole": "repair"},
        hidden_dim=24,
        epochs=20,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=5,
        ensemble_size=1,
    )

    loaded = load_scheduler_bundle(output_path)
    pred = predict_scheduler_plan_slate(loaded, rows[:6])

    assert summary["route_aliases"] == {"hole": "repair"}
    assert loaded["route_aliases"] == {"hole": "repair"}
    assert pred["trained"] is True


def test_scheduler_training_rebalancing_records_effective_counts_and_weights():
    rows = _synthetic_scheduler_rows(12)
    bundle = train_scheduler_critic(
        rows,
        hidden_dim=24,
        epochs=20,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=13,
        ensemble_size=1,
        route_weights={"hole": 2.5, "repair": 1.2},
        budget_weight_map={1: 1.0, 2: 1.5, 4: 2.0},
        deep_repair_min_depth=5,
        deep_repair_weight=3.0,
        hole_oversample_repeat=2,
        deep_repair_oversample_repeat=2,
    )

    train_metrics = bundle["metrics"]["train"]
    rebalance = bundle["training_rebalance"]
    assert train_metrics["effective_n_rows"] >= train_metrics["n_rows"]
    assert rebalance["route_weights"]["hole"] == pytest.approx(2.5)
    assert rebalance["budget_weights"]["4"] == pytest.approx(2.0)
    assert rebalance["deep_repair_weight"] == pytest.approx(3.0)
    assert rebalance["oversample"]["output_rows"] >= rebalance["oversample"]["input_rows"]
    assert rebalance["mean_row_weight_by_route"]["hole"] >= rebalance["mean_row_weight_by_route"]["build"]


def test_scheduler_critic_cost_masks_ignore_heuristic_wall_labels():
    observed_row = shared_opportunity_row_dict(
        {
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "opportunity_id": "repair_observed_cost",
            "decision_id": "decision_cost_mask",
            "decision_context_id": "decision_cost_mask",
            "beam_id": "repair_observed_cost:0",
            "parent_expr": "parent_cost_mask",
            "parent_key": "parent_cost_mask",
            "action": "inv_steer",
            "path": [1],
            "target_mode": "identity",
            "method_name": "inverse",
            "subroute": "beam",
            "parent_depth": 4,
            "parent_eff_mse": 1.1,
            "current_best_route_eff_mse": 1.0,
            "budget_exact_spent": 0,
            "budget_remaining": 2,
            "preview_candidate_count_total": 2,
            "preview_candidate_count_unique_total": 2,
            "observed_wall_seconds": 0.3,
            "observed_exact_evals": 1,
            "observed_preview_evals": 2,
            "expected_gain_at_budget_1_under_executor": 0.30,
            "expected_gain_at_budget_1_under_oracle_executor": 0.32,
            "expected_gain_at_budget_2_under_executor": 0.40,
            "expected_gain_at_budget_2_under_oracle_executor": 0.42,
        },
        route_source="repair",
    )
    heuristic_row = shared_opportunity_row_dict(
        {
            "route_source": "build",
            "opportunity_type": "build_action",
            "opportunity_id": "build_heuristic_cost",
            "decision_id": "decision_cost_mask",
            "decision_context_id": "decision_cost_mask",
            "beam_id": "build_heuristic_cost:0",
            "parent_expr": "parent_cost_mask",
            "parent_key": "parent_cost_mask",
            "action": "replace",
            "path": [0],
            "method_name": "build",
            "subroute": "tuple",
            "parent_depth": 4,
            "parent_eff_mse": 1.1,
            "current_best_route_eff_mse": 1.0,
            "budget_exact_spent": 1,
            "budget_remaining": 2,
            "preview_candidate_count_total": 2,
            "preview_candidate_count_unique_total": 2,
            "observed_exact_evals": 0,
            "observed_preview_evals": 2,
            "cost_estimate": 0.8,
            "expected_gain_at_budget_1_under_executor": 0.45,
            "expected_gain_at_budget_1_under_oracle_executor": 0.46,
            "expected_gain_at_budget_2_under_executor": 0.50,
            "expected_gain_at_budget_2_under_oracle_executor": 0.52,
        },
        route_source="build",
    )

    rows = augment_scheduler_shadow_rows(
        [observed_row, heuristic_row],
        budget_ladder=(1, 2),
        threshold_ladder=(0.25,),
    )
    tensors = _build_feature_tensors(
        rows,
        feature_names=SCHEDULER_FEATURE_NAMES,
        budget_ladder=(1, 2),
        threshold_ladder=(0.25,),
    )

    assert tensors["cost_wall_masks"][:, 0].tolist() == pytest.approx([1.0, 0.0])
    assert tensors["cost_exact_masks"][:, 0].tolist() == pytest.approx([1.0, 0.0])
