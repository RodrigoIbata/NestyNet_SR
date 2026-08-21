# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import json

import pytest

from nestynet_sr.sr_search.factorized_search.opportunity_critic import (
    evaluate_opportunity_controller,
    load_opportunity_bundle,
    OPPORTUNITY_WITNESS_FEATURE_NAMES,
    opportunity_feature_names,
    opportunity_feature_vector,
    opportunity_row_group_id,
    predict_opportunity_slate,
    save_opportunity_bundle,
    split_opportunity_rows_grouped,
    train_opportunity_controller,
)
from nestynet_sr.sr_search.factorized_search.opportunity_train import run_opportunity_training
from nestynet_sr.sr_search.factorized_search.shared_opportunity import shared_opportunity_row_dict


def _synthetic_opportunity_rows(n: int = 48) -> list[dict]:
    rows: list[dict] = []
    for idx in range(int(n)):
        route_source = "repair" if idx % 2 == 0 else "build"
        opportunity_type = "repair_beam" if route_source == "repair" else "build_action"
        action = "inv_steer" if route_source == "repair" else ("replace" if idx % 4 else "wrap_un")
        target_mode = "identity" if route_source == "repair" and idx % 3 == 0 else ("full" if route_source == "repair" else "")
        base = float((idx % 12) / 11.0)
        confidence = 0.20 + (0.72 * base if route_source == "repair" else 0.45 * base)
        path_gain = 0.10 + 0.85 * base
        rel_gain = 0.08 + 0.70 * base
        repair_bonus = 0.12 if route_source == "repair" else -0.03
        gain_next = max(0.0, min(0.95, (0.60 * path_gain) + (0.45 * confidence) + repair_bonus - 0.35))
        fragility = 1.0 if confidence < 0.45 else 0.0
        row = shared_opportunity_row_dict(
            {
                "route_source": route_source,
                "opportunity_type": opportunity_type,
                "opportunity_id": f"opp_{idx}",
                "decision_id": f"decision_{idx // 4}",
                "beam_id": f"beam_{idx}",
                "parent_expr": f"parent_{idx // 3}",
                "action": action,
                "path": [1] if route_source == "repair" else [],
                "path_source": "inverse_beam" if route_source == "repair" else "critic_path_head",
                "target_mode": target_mode,
                "evidence_level": "preview_support" if idx % 3 else "exact_known",
                "parent_depth": 4 + (idx % 5),
                "parent_eff_mse": 0.50 + (0.4 * (1.0 - base)),
                "budget_exact_spent": idx % 2,
                "budget_remaining": 4 - (idx % 2),
                "candidate_count_observed": 1 + (idx % 3),
                "candidate_count_unique": 1 + (idx % 2),
                "preview_candidate_count_total": 2 + (idx % 4),
                "preview_candidate_count_unique_total": 2 + (idx % 3),
                "shadow_total_exact_available": 4,
                "shadow_total_preview_available": 5,
                "shadow_executor_reveals_observed": idx % 2,
                "current_best_child_expr": f"child_{idx}",
                "current_best_child_eff_mse": 0.85 - (0.55 * base),
                "current_best_route_eff_mse": 0.90 - (0.45 * base),
                "best_preview_probe_mse": 0.80 - (0.40 * base),
                "best_preview_fit_mse": 0.75 - (0.35 * base),
                "best_tuple_utility_estimate": 0.10 + (0.80 * base),
                "best_tuple_allocation_estimate": 0.08 + (0.72 * base),
                "path_gain": path_gain,
                "path_gain_pre_cut": path_gain + 0.05,
                "rel_gain": rel_gain,
                "transport_rel": 0.05 + (0.60 * base),
                "lin_rel": 0.02 + (0.40 * base),
                "valid_frac": 0.30 + (0.65 * base),
                "confidence": confidence,
                "effective_n": 3 + idx,
                "branch_factor": 0.8 + (0.2 * base),
                "cut_factor": 0.6 + (0.35 * base),
                "branch_support": 0.1 + (0.8 * base),
                "family_scale": 1.0 + (0.2 * base),
                "expected_gain_next_under_executor": gain_next,
                "cost_estimate": 0.05,
                "coverage_at_budget_0": 0.0,
                "coverage_at_budget_1": 1.0 if gain_next > 0.14 else 0.0,
                "coverage_at_budget_2": 1.0 if gain_next > 0.07 else 0.0,
                "coverage_at_budget_4": 1.0 if gain_next > 0.02 else 0.0,
                "cond_gain_at_budget_1_if_covered_under_executor": gain_next if gain_next > 0.14 else 0.0,
                "cond_gain_at_budget_2_if_covered_under_executor": min(1.0, gain_next * 1.15 + 0.02) if gain_next > 0.07 else 0.0,
                "cond_gain_at_budget_4_if_covered_under_executor": min(1.0, gain_next * 1.30 + 0.04) if gain_next > 0.02 else 0.0,
                "fragility_at_budget_1": fragility,
                "route_flip_at_budget_1": 1.0 if route_source == "repair" and gain_next > 0.26 else 0.0,
                "new_residual_basin_at_budget_1": 1.0 if route_source == "repair" and path_gain > 0.70 else 0.0,
            },
            route_source=route_source,
        )
        rows.append(row)
    return rows


def test_train_and_predict_opportunity_controller_round_trip(tmp_path):
    rows = _synthetic_opportunity_rows()
    bundle = train_opportunity_controller(
        rows,
        hidden_dim=32,
        epochs=80,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=7,
    )

    assert bundle["opportunity_controller_trained"] is True
    assert bundle["model_kind"] == "opportunity_controller_v1"
    assert bundle["feature_schema_version"] == 1
    assert bundle["witness_energy_feature_enable"] is False
    assert "1" in bundle["coverage_temperature"]
    assert "2" in bundle["coverage_temperature"]
    assert "4" in bundle["coverage_temperature"]
    assert "calibrated_monotonic_violation_rate" in bundle["calibration"]

    out_path = tmp_path / "opportunity_bundle.pt"
    save_opportunity_bundle(bundle, out_path)
    loaded = load_opportunity_bundle(out_path)
    pred = predict_opportunity_slate(loaded, rows[:8])

    assert pred["trained"] is True
    assert pred["feature_schema_version"] == 1
    assert pred["witness_energy_feature_enable"] is False
    assert pred["rows"]
    assert pred["best_opportunity_id"]
    assert pred["best_route"] in {"repair", "build"}
    best_row = pred["rows"][0]
    assert "expected_gain_next_under_executor" in best_row
    assert "cover_prob_at_1" in best_row
    assert "cover_prob_at_2" in best_row
    assert "cover_prob_at_4" in best_row
    assert best_row["cover_prob_at_0"] == 0.0
    assert best_row["cover_prob_at_1"] <= best_row["cover_prob_at_2"] <= best_row["cover_prob_at_4"]


def test_evaluate_opportunity_controller_reports_metrics_and_monotonicity():
    rows = _synthetic_opportunity_rows()
    bundle = train_opportunity_controller(
        rows,
        hidden_dim=32,
        epochs=80,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=11,
    )
    metrics = evaluate_opportunity_controller(bundle, rows)

    assert metrics["n_rows"] == len(rows)
    assert metrics["gain_mae"] is not None
    assert metrics["gain_rmse"] is not None
    assert metrics["gain_mae"] < 0.25
    assert metrics["cover_brier"]["1"] is not None
    assert metrics["cover_ece"]["1"] is not None
    assert metrics["cover_monotonic_violation_rate"] == pytest.approx(0.0)


def test_run_opportunity_training_writes_bundle_and_summary(tmp_path):
    rows = _synthetic_opportunity_rows()
    dataset_path = tmp_path / "opportunity_dataset.json"
    dataset_path.write_text(json.dumps({"mode": "opportunity_shadow_dataset", "rows": rows}, indent=2), encoding="utf-8")
    output_path = tmp_path / "opportunity_bundle.pt"

    summary = run_opportunity_training(
        dataset_paths=[str(dataset_path)],
        output_path=str(output_path),
        hidden_dim=32,
        epochs=60,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=3,
    )

    assert output_path.exists()
    assert output_path.with_suffix(output_path.suffix + ".json").exists()
    assert summary["witness_energy_feature_enable"] is False
    assert summary["metrics"]["train"]["n_rows"] > 0
    assert summary["full_eval"]["gain_mae"] is not None
    assert summary["sample_prediction"]["trained"] is True


def test_split_opportunity_rows_grouped_keeps_shared_decision_groups_together():
    rows = _synthetic_opportunity_rows(24)
    for idx, row in enumerate(rows):
        row["decision_id"] = f"decision_{idx // 3}"
        row["shadow_source_row_id"] = f"source_{idx // 6}"
    train_rows, val_rows = split_opportunity_rows_grouped(rows, val_fraction=0.25, seed=5)

    assert train_rows
    assert val_rows
    train_groups = {opportunity_row_group_id(row) for row in train_rows}
    val_groups = {opportunity_row_group_id(row) for row in val_rows}
    assert train_groups.isdisjoint(val_groups)


def test_evaluate_opportunity_controller_leaves_cost_metric_empty_without_targets():
    rows = _synthetic_opportunity_rows()
    for row in rows:
        row.pop("cost_estimate", None)
        row.pop("observed_wall_seconds", None)
        row.pop("observed_exact_evals", None)
    bundle = train_opportunity_controller(
        rows,
        hidden_dim=32,
        epochs=40,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=13,
    )
    metrics = evaluate_opportunity_controller(bundle, rows)

    assert metrics["cost_mae"] is None


def test_opportunity_feature_vector_adds_witness_energy_features_when_enabled():
    row = shared_opportunity_row_dict(
        {
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "opportunity_id": "opp_witness_features",
            "decision_id": "decision_witness_features",
            "beam_id": "beam_witness_features",
            "parent_expr": "parent_witness_features",
            "action": "inv_steer",
            "path": [1],
            "path_source": "inverse_beam",
            "target_mode": "identity",
            "witness_value_loss": 0.6,
            "witness_grad_loss": 0.2,
            "witness_energy_total": 0.8,
            "witness_energy_delta_estimate": 0.3,
            "expected_gain_next_under_executor": 0.4,
        },
        route_source="repair",
    )
    feature_names = opportunity_feature_names(witness_energy_feature_enable=True)
    values = opportunity_feature_vector(row, feature_names=feature_names)
    feature_map = dict(zip(feature_names, values))

    assert set(OPPORTUNITY_WITNESS_FEATURE_NAMES).issubset(feature_map.keys())
    assert feature_map["witness_value_log_loss"] > 0.0
    assert feature_map["witness_grad_log_loss"] > 0.0
    assert feature_map["witness_energy_total_log"] > 0.0
    assert feature_map["witness_energy_delta_estimate"] == pytest.approx(0.3)
    assert feature_map["witness_grad_present"] == pytest.approx(1.0)


def test_opportunity_controller_round_trip_with_witness_energy_feature_schema(tmp_path):
    rows = _synthetic_opportunity_rows(24)
    bundle = train_opportunity_controller(
        rows,
        hidden_dim=24,
        epochs=30,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=19,
        witness_energy_feature_enable=True,
    )

    assert bundle["witness_energy_feature_enable"] is True
    assert bundle["feature_schema_version"] == 2
    assert set(OPPORTUNITY_WITNESS_FEATURE_NAMES).issubset(set(bundle["feature_names"]))

    out_path = tmp_path / "opportunity_bundle_witness.pt"
    save_opportunity_bundle(bundle, out_path)
    loaded = load_opportunity_bundle(out_path)
    pred = predict_opportunity_slate(loaded, rows[:6])

    assert loaded["witness_energy_feature_enable"] is True
    assert loaded["feature_schema_version"] == 2
    assert pred["trained"] is True
    assert pred["witness_energy_feature_enable"] is True
    assert pred["feature_schema_version"] == 2
