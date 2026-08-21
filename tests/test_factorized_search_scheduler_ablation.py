# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.sr_search.factorized_search.controller_harness import (
    RunSummary,
    _summarize_pairwise_comparisons,
)
from nestynet_sr.sr_search.factorized_search.scheduler_ablation import run_scheduler_ablations
from nestynet_sr.sr_search.factorized_search.stage1_benchmark_harness import _arm_configs


def _run_summary(
    label: str,
    *,
    target: str = "toy",
    seed: int = 0,
    best_eff_mse: float = 1.0,
    elapsed_s: float = 1.0,
    exact_eval_count: int = 1,
    solve_hit: bool = False,
) -> RunSummary:
    return RunSummary(
        label=str(label),
        target=str(target),
        seed=int(seed),
        profile="default",
        n_iter=8,
        max_depth=4,
        best_eff_mse=float(best_eff_mse),
        best_raw_mse=float(best_eff_mse),
        best_expr="x0",
        residual_basins=1,
        n_eval=1,
        elapsed_s=float(elapsed_s),
        solve_hit=bool(solve_hit),
        exact_eval_count=int(exact_eval_count),
    )


def test_matched_exact_comparison_discounts_more_expensive_candidate():
    baseline = _run_summary("baseline", best_eff_mse=1.0e-3, exact_eval_count=10)
    candidate = _run_summary("scheduler_control", best_eff_mse=1.0e-4, exact_eval_count=20)
    summary = _summarize_pairwise_comparisons([baseline], [candidate], comparison_mode="matched_exact")

    assert summary["comparison_mode"] == "matched_exact"
    assert summary["n_pairs"] == 1
    assert summary["mean_delta_log_eff"] > 0.0
    assert summary["mean_matched_delta_log_eff"] > 0.0
    assert summary["mean_matched_delta_log_eff"] < summary["mean_delta_log_eff"]
    assert summary["mean_candidate_cost_ratio"] == 2.0


def test_stage1_scheduler_arm_configs_forward_ablation_overrides():
    arms = dict(
        _arm_configs(
            critic_path="",
            scheduler_bundle_path="/tmp/stub_scheduler_bundle.pt",
            scheduler_budget_ladder=(1, 2, 4),
            scheduler_arm_overrides={
                "scheduler_uncertainty_bonus": 0.0,
                "scheduler_acquisition_weights": {"cost_exact": 0.0, "cost_wall": 0.0},
            },
            blends=(),
            refine_enable=False,
            arm_modes=("scheduler_advisory", "scheduler_control"),
            macro_profile="repair_probe",
        )
    )
    control = arms["stage1_scheduler_control_repair_probe"]
    advisory = arms["stage1_scheduler_advisory_repair_probe"]

    assert control["scheduler_uncertainty_bonus"] == 0.0
    assert advisory["scheduler_uncertainty_bonus"] == 0.0
    assert control["scheduler_acquisition_weights"]["cost_exact"] == 0.0
    assert advisory["scheduler_acquisition_weights"]["cost_wall"] == 0.0


def test_scheduler_ablation_runner_applies_named_overrides(monkeypatch):
    captured: list[dict[str, object]] = []

    def _fake_controller_benchmark(**kwargs):
        captured.append(dict(kwargs))
        return {
            "comparisons": {
                "scheduler_control": {
                    "comparison_mode": kwargs.get("comparison_mode"),
                    "matched_win_rate": 1.0,
                    "mean_matched_delta_log_eff": 0.5,
                    "mean_candidate_cost_ratio": 1.1,
                }
            }
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_ablation.run_controller_benchmark",
        _fake_controller_benchmark,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_ablation.load_scheduler_bundle",
        lambda _path: {"budget_ladder": [1, 2, 4], "threshold_ladder": [0.1, 0.25, 0.5]},
    )

    report = run_scheduler_ablations(
        benchmark_kind="controller",
        ablations=("no_cost_term", "no_uncertainty_bonus", "one_step_budget_only", "exact_scored_build"),
        targets=("toy",),
        seeds=(0,),
        scheduler_bundle_path="/tmp/stub_scheduler_bundle.pt",
        comparison_mode="matched_wall",
    )

    assert report["ablation_order"] == [
        "no_cost_term",
        "no_uncertainty_bonus",
        "one_step_budget_only",
        "exact_scored_build",
    ]
    assert captured[0]["scheduler_arm_overrides"]["scheduler_acquisition_weights"]["cost_exact"] == 0.0
    assert captured[1]["scheduler_arm_overrides"]["scheduler_uncertainty_bonus"] == 0.0
    assert captured[2]["scheduler_budget_ladder"] == [1]
    assert captured[3]["scheduler_arm_overrides"]["scheduler_build_preview_only"] is False
    assert report["summaries"]["no_cost_term"]["primary_comparison_key"] == "scheduler_control"
    assert report["summaries"]["no_uncertainty_bonus"]["primary_comparison"]["comparison_mode"] == "matched_wall"


def test_scheduler_ablation_runner_trains_derived_bundles_for_training_variants(monkeypatch):
    captured_train: list[dict[str, object]] = []
    captured_stage1: list[dict[str, object]] = []

    def _fake_train(**kwargs):
        captured_train.append(dict(kwargs))
        return {
            "output_path": kwargs.get("output_path"),
            "metrics": {"train": {"n_rows": 4}},
            "full_eval": {"n_rows": 4},
            "sample_prediction": {"trained": True},
        }

    def _fake_stage1_benchmark(**kwargs):
        captured_stage1.append(dict(kwargs))
        return {
            "comparisons": {
                "scheduler_control": {
                    "comparison_mode": kwargs.get("comparison_mode"),
                    "matched_win_rate": 0.75,
                    "mean_matched_delta_log_eff": 0.2,
                    "mean_candidate_cost_ratio": 1.0,
                }
            }
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_ablation.run_scheduler_training",
        _fake_train,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_ablation.run_stage1_benchmark",
        _fake_stage1_benchmark,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_ablation.load_scheduler_bundle",
        lambda _path: {
            "hidden_dim": 32,
            "ensemble_size": 2,
            "budget_ladder": [1, 2, 4],
            "threshold_ladder": [0.1, 0.25, 0.5],
        },
    )

    report = run_scheduler_ablations(
        benchmark_kind="stage1",
        ablations=("merged_repair_hole_route_families", "oracle_refine_live_advisory_finetune"),
        targets=("toy",),
        seeds=(0,),
        scheduler_bundle_path="/tmp/stub_scheduler_bundle.pt",
        scheduler_dataset_paths=("rows_a.json",),
        comparison_mode="matched_exact",
    )

    assert len(captured_train) == 2
    assert captured_train[0]["route_aliases"] == {"hole": "repair"}
    assert captured_train[0]["init_bundle_path"] == "/tmp/stub_scheduler_bundle.pt"
    assert captured_train[1]["route_aliases"] == {}
    assert captured_stage1[0]["scheduler_bundle_path"] != "/tmp/stub_scheduler_bundle.pt"
    assert report["summaries"]["merged_repair_hole_route_families"]["bundle_source"] == "derived_bundle"
    assert report["summaries"]["oracle_refine_live_advisory_finetune"]["training_summary"]["sample_prediction"]["trained"] is True
