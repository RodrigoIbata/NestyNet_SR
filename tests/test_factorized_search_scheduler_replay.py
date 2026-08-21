# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from types import SimpleNamespace

from nestynet_sr.sr_search.factorized_search.scheduler_critic import train_scheduler_critic
from nestynet_sr.sr_search.factorized_search.scheduler_dataset import augment_scheduler_shadow_rows
from nestynet_sr.sr_search.factorized_search.scheduler_replay import replay_scheduler_decisions
from nestynet_sr.sr_search.factorized_search.shared_opportunity import shared_opportunity_row_dict


def _route_spec(route: str) -> tuple[str, str, list[int], str, str]:
    if route == "repair":
        return ("repair_beam", "inv_steer", [1], "identity", "inverse")
    if route == "build":
        return ("build_action", "replace", [0], "", "build")
    return ("hole_frontier", "hole_search", [2], "", "hole")


def _replay_rows(n_groups: int = 9) -> list[dict]:
    rows: list[dict] = []
    routes = ("repair", "build", "hole")
    for group_idx in range(int(n_groups)):
        dominant = int(group_idx % len(routes))
        actual_choice = int((group_idx + 1) % len(routes))
        current_eff = 1.0 + (0.05 * float(group_idx % 3))
        for route_idx, route in enumerate(routes):
            opportunity_type, action, path, target_mode, method_name = _route_spec(route)
            route_bonus = 0.36 if route_idx == dominant else (0.18 if route_idx == actual_choice else 0.06)
            gain1 = min(0.80 * current_eff, 0.08 + route_bonus)
            gain2 = min(0.90 * current_eff, gain1 + 0.08 + (0.04 * route_bonus))
            gain4 = min(0.96 * current_eff, gain2 + 0.10 + (0.06 * route_bonus))
            rows.append(
                shared_opportunity_row_dict(
                    {
                        "route_source": route,
                        "opportunity_type": opportunity_type,
                        "opportunity_id": f"{route}_{group_idx}",
                        "decision_id": f"decision_{group_idx}",
                        "decision_context_id": f"decision_{group_idx}",
                        "parent_key": f"parent_{group_idx}",
                        "beam_id": f"{route}_{group_idx}:0",
                        "action": action,
                        "path": path,
                        "target_mode": target_mode,
                        "method_name": method_name,
                        "subroute": "frontier" if route == "hole" else ("tuple" if route == "build" else "beam"),
                        "evidence_level": "preview_support",
                        "parent_depth": 3 + (group_idx % 4),
                        "current_best_route_eff_mse": current_eff,
                        "parent_eff_mse": current_eff + 0.1,
                        "budget_exact_spent": 1 if route_idx == actual_choice else 0,
                        "budget_remaining": 4,
                        "candidate_count_observed": 0,
                        "candidate_count_unique": 0,
                        "preview_candidate_count_total": 3,
                        "preview_candidate_count_unique_total": 3,
                        "shadow_total_exact_available": 4,
                        "shadow_total_preview_available": 4,
                        "observed_wall_seconds": 0.10 + (0.02 * route_idx),
                        "observed_exact_evals": 1 if route_idx == actual_choice else 0,
                        "observed_preview_evals": 2,
                        "observed_micro_tokens": 0,
                        "observed_widen_tokens": 0,
                        "predicted_value": gain4 if route == "hole" else 0.0,
                        "predicted_cost": 0.10 + (0.03 * route_idx),
                        "preview_solvability": 0.20 + route_bonus,
                        "expected_gain_at_budget_1_under_executor": gain1,
                        "expected_gain_at_budget_1_under_oracle_executor": min(current_eff, gain1 + 0.02),
                        "expected_gain_at_budget_2_under_executor": gain2,
                        "expected_gain_at_budget_2_under_oracle_executor": min(current_eff, gain2 + 0.03),
                        "expected_gain_at_budget_4_under_executor": gain4,
                        "expected_gain_at_budget_4_under_oracle_executor": min(current_eff, gain4 + 0.04),
                        "new_residual_basin_at_budget_1": 1.0 if route == "hole" and route_idx == dominant else 0.0,
                        "new_residual_basin_at_budget_2": 1.0 if route == "hole" and route_idx == dominant else 0.0,
                        "new_residual_basin_at_budget_4": 1.0 if route == "hole" and route_idx == dominant else 0.0,
                        "fragility_at_budget_1": 1.0 if route == "build" and route_idx != dominant else 0.0,
                        "fragility_at_budget_2": 1.0 if route == "build" and route_idx != dominant else 0.0,
                        "fragility_at_budget_4": 1.0 if route == "build" and route_idx != dominant else 0.0,
                        "stability_at_budget_1": 1.0 if route_idx == dominant else 0.0,
                        "stability_at_budget_2": 1.0 if route_idx == dominant else 0.0,
                        "stability_at_budget_4": 1.0 if route_idx == dominant else 0.0,
                        "actual_selected": bool(route_idx == actual_choice),
                    },
                    route_source=route,
                )
            )
    return augment_scheduler_shadow_rows(
        rows,
        budget_ladder=(1, 2, 4),
        threshold_ladder=(0.1, 0.25, 0.5),
    )


def _attach_witness_labels(rows: list[dict]) -> list[dict]:
    for row in rows:
        route = str(row.get("route_source", "") or "")
        route_scale = 1.0 if route == "hole" else (0.75 if route == "repair" else 0.4)
        for budget in (1, 2, 4):
            before = 0.9
            delta = route_scale * (0.08 + (0.05 * float(budget)))
            row[f"witness_energy_total_before_at_budget_{int(budget)}"] = before
            row[f"witness_energy_total_after_at_budget_{int(budget)}"] = before - delta
            row[f"witness_energy_delta_at_budget_{int(budget)}"] = delta
            row[f"witness_energy_label_source_at_budget_{int(budget)}"] = "observed_outcome"
            row[f"witness_energy_observed_mask_at_budget_{int(budget)}"] = 1.0
    return rows


def test_replay_scheduler_decisions_reports_regret_and_calibration():
    rows = _replay_rows()
    bundle = train_scheduler_critic(
        rows,
        hidden_dim=32,
        epochs=40,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=5,
        ensemble_size=2,
        budget_ladder=(1, 2, 4),
        threshold_ladder=(0.1, 0.25, 0.5),
    )

    summary = replay_scheduler_decisions(rows, bundle, acquisition_threshold=0.25)

    assert summary["trained"] is True
    assert summary["n_groups"] == 9
    assert summary["groups_replayed"] == 9
    assert summary["groups_with_actual_choice"] == 9
    assert summary["top1_hit_rate"] is not None
    assert summary["mean_regret"] is not None
    assert summary["mean_normalized_regret"] is not None
    assert summary["actual_mean_regret"] is not None
    assert summary["actual_mean_wasted_budget"] is not None
    assert "build" in summary["calibration_by_route"]
    assert "repair" in summary["calibration_by_route"]
    assert "hole" in summary["calibration_by_route"]
    assert "3-4" in summary["calibration_by_depth"] or "5-6" in summary["calibration_by_depth"]
    assert "1" in summary["calibration_by_budget"]
    assert len(summary["decision_rows"]) == 9


def test_replay_scheduler_decisions_resolves_fallback_with_actual_choice(monkeypatch):
    rows = _replay_rows(1)
    scored_row = dict(rows[0])
    scored_row.update({
        "plan_route": str(rows[0].get("route_source", "") or ""),
        "plan_exact_budget": 1,
    })

    def fake_choose_plan(*_args, **_kwargs):
        return SimpleNamespace(
            trained=True,
            advisory_only=False,
            candidate_count=3,
            fallback_used=True,
            fallback_reason="low_confidence",
            confidence=0.1,
            chosen_candidate=None,
            chosen_route="",
            chosen_opportunity_id="",
            chosen_exact_budget=0,
            route_scores={},
            rows=(scored_row,),
        )

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_replay.choose_plan",
        fake_choose_plan,
    )

    summary = replay_scheduler_decisions(
        rows,
        {"scheduler_critic_trained": True, "budget_ladder": [1, 2, 4], "threshold_ladder": [0.1, 0.25, 0.5]},
        acquisition_threshold=0.25,
    )

    assert summary["trained"] is True
    assert summary["groups_replayed"] == 1
    assert summary["groups_with_scheduler_fallback"] == 1
    assert summary["groups_with_resolved_fallback"] == 1
    assert summary["groups_with_unresolved_fallback"] == 0
    assert summary["decision_rows"][0]["predicted_source"] == "fallback_actual_choice"
    assert summary["decision_rows"][0]["scheduler_fallback_used"] is True


def test_replay_scheduler_decisions_uses_shared_scheduler_context_and_flags_degraded_fallback_groups():
    unified_rows = _replay_rows(1)
    for row in unified_rows:
        route = str(row.get("route_source", "") or "")
        row["decision_id"] = f"{route}_decision_local"
        row["decision_context_id"] = f"{route}_context_local"
        row["route_decision_id"] = str(row["decision_id"])
        row["route_decision_context_id"] = str(row["decision_context_id"])
        row["scheduler_decision_context_id"] = "scheduler_context_demo"
        row["decision_group_id"] = "scheduler_context_demo"
    bundle = train_scheduler_critic(
        unified_rows,
        hidden_dim=16,
        epochs=10,
        lr=5.0e-3,
        val_fraction=0.34,
        seed=3,
        ensemble_size=1,
        budget_ladder=(1, 2, 4),
        threshold_ladder=(0.1, 0.25, 0.5),
    )
    summary = replay_scheduler_decisions(unified_rows, bundle, acquisition_threshold=0.25)
    assert summary["n_groups"] == 1
    assert summary["groups_with_degraded_context"] == 0
    assert summary["grouping_source_counts"]["scheduler_context"] == 1
    assert summary["decision_rows"][0]["group_id"] == "scheduler_context_demo"
    assert summary["decision_rows"][0]["grouping_degraded"] is False

    degraded_rows = _replay_rows(1)
    for row in degraded_rows:
        route = str(row.get("route_source", "") or "")
        row["decision_id"] = f"{route}_decision_local"
        row["decision_context_id"] = f"{route}_context_local"
        row.pop("scheduler_decision_context_id", None)
        row.pop("decision_group_id", None)
    degraded_summary = replay_scheduler_decisions(
        degraded_rows,
        bundle,
        acquisition_threshold=0.25,
    )
    assert degraded_summary["n_groups"] == 3
    assert degraded_summary["groups_with_degraded_context"] == 3
    assert degraded_summary["grouping_source_counts"]["route_local_fallback"] == 3
    assert all(row["grouping_degraded"] for row in degraded_summary["decision_rows"])


def test_replay_scheduler_decisions_reports_witness_regret_when_labels_are_available():
    rows = _attach_witness_labels(_replay_rows())
    bundle = train_scheduler_critic(
        rows,
        hidden_dim=32,
        epochs=20,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=7,
        ensemble_size=2,
        budget_ladder=(1, 2, 4),
        threshold_ladder=(0.1, 0.25, 0.5),
    )

    summary = replay_scheduler_decisions(rows, bundle, acquisition_threshold=0.25)

    assert summary["groups_with_witness_labels"] == 9
    assert summary["mean_witness_regret"] is not None
    assert summary["mean_normalized_witness_regret"] is not None
    assert "predicted_witness_delta" in summary["decision_rows"][0]
    assert "witness_regret" in summary["decision_rows"][0]


def test_witness_objective_replay_improves_witness_regret_over_acquisition_bundle():
    rows = _attach_witness_labels(_replay_rows())
    acquisition_bundle = train_scheduler_critic(
        rows,
        hidden_dim=32,
        epochs=30,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=9,
        ensemble_size=1,
        objective_mode="acquisition",
    )
    witness_bundle = train_scheduler_critic(
        rows,
        hidden_dim=32,
        epochs=30,
        lr=5.0e-3,
        val_fraction=0.25,
        seed=9,
        ensemble_size=1,
        objective_mode="witness",
    )

    acquisition_summary = replay_scheduler_decisions(rows, acquisition_bundle, acquisition_threshold=0.25)
    witness_summary = replay_scheduler_decisions(rows, witness_bundle, acquisition_threshold=0.25)

    assert acquisition_summary["mean_witness_regret"] is not None
    assert witness_summary["mean_witness_regret"] is not None
    assert witness_summary["mean_witness_regret"] <= acquisition_summary["mean_witness_regret"]
