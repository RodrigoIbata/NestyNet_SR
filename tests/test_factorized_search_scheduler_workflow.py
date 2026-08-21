# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import json

from nestynet_sr.sr_search.factorized_search.scheduler_workflow import run_scheduler_workflow


def test_scheduler_workflow_uses_existing_bundle_and_writes_packet(tmp_path, monkeypatch):
    captured_controller: list[dict] = []
    captured_stage1: list[dict] = []
    captured_promotions: list[dict] = []

    def _fake_controller_benchmark(**kwargs):
        captured_controller.append(dict(kwargs))
        return {
            "arm_overall": {
                "macro": {"solve_rate": 0.4, "mean_wall_s": 1.0, "mean_exact_eval_count": 10.0},
                "scheduler_control": {"solve_rate": 0.5, "mean_wall_s": 1.02, "mean_exact_eval_count": 11.0},
            },
            "comparisons": {"scheduler_control": {"comparison_mode": kwargs.get("comparison_mode")}},
        }

    def _fake_stage1_benchmark(**kwargs):
        captured_stage1.append(dict(kwargs))
        return {
            "overall": {
                "stage0_selective": {"solve_rate": 0.4, "mean_wall_s": 1.0, "mean_exact_eval_count": 10.0},
                "stage1_scheduler_control": {"solve_rate": 0.5, "mean_wall_s": 1.02, "mean_exact_eval_count": 11.0},
            },
            "comparisons": {"stage1_scheduler_control": {"comparison_mode": kwargs.get("comparison_mode")}},
        }

    def _fake_load_bundle(_path):
        return {"scheduler_critic_trained": True, "budget_ladder": [1, 2, 4], "threshold_ladder": [0.25]}

    def _fake_load_rows(paths):
        return [{"decision_id": "d0", "route_source": "build"}] if paths else []

    def _fake_replay(rows, bundle, **kwargs):
        return {
            "trained": True,
            "groups_replayed": 1,
            "decision_rows": [],
            "top1_hit_rate": 1.0,
            "calibration_by_route": {},
            "calibration_by_depth": {},
            "calibration_by_budget": {},
        }

    def _fake_promotion(**kwargs):
        captured_promotions.append(dict(kwargs))
        return {"mode": "scheduler_promotion", "decision": "promote", "meets_promotion_bar": True}

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.run_controller_benchmark",
        _fake_controller_benchmark,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.run_stage1_benchmark",
        _fake_stage1_benchmark,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.load_scheduler_bundle",
        _fake_load_bundle,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.load_scheduler_dataset_rows",
        _fake_load_rows,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.replay_scheduler_decisions",
        _fake_replay,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.recommend_scheduler_promotion",
        _fake_promotion,
    )

    report = run_scheduler_workflow(
        scheduler_bundle_path="/tmp/base_scheduler.pt",
        scheduler_dataset_paths=("rows.json",),
        output_dir=str(tmp_path),
        comparison_modes=("matched_exact", "matched_wall"),
        controller_targets=("toy",),
        controller_seeds=(0,),
        stage1_targets=("toy",),
        stage1_seeds=(0,),
    )

    assert report["config"]["effective_scheduler_bundle_path"] == "/tmp/base_scheduler.pt"
    assert report["training_summary"] is None
    assert report["promotion"]["decision"] == "promote"
    assert sorted(report["promotion_by_mode"].keys()) == ["matched_exact", "matched_wall"]
    assert [call["comparison_mode"] for call in captured_controller] == ["matched_exact", "matched_wall"]
    assert [call["comparison_mode"] for call in captured_stage1] == ["matched_exact", "matched_wall"]
    assert captured_stage1[0]["scheduler_dataset_paths"] == ()
    assert len(captured_promotions) == 2
    assert (tmp_path / "scheduler_workflow_packet.json").exists()
    assert (tmp_path / "controller_report_matched_exact.json").exists()
    assert (tmp_path / "stage1_report_matched_wall.json").exists()
    assert (tmp_path / "scheduler_replay.json").exists()


def test_scheduler_workflow_trains_derived_bundle_before_benchmarking(tmp_path, monkeypatch):
    captured_training: list[dict] = []
    captured_controller: list[dict] = []

    def _fake_training(**kwargs):
        captured_training.append(dict(kwargs))
        output_path = kwargs["output_path"]
        return {
            "output_path": output_path,
            "metrics": {"train": {"n_rows": 8}},
            "full_eval": {"n_rows": 8},
            "sample_prediction": {"trained": True},
        }

    def _fake_controller_benchmark(**kwargs):
        captured_controller.append(dict(kwargs))
        return {
            "arm_overall": {
                "macro": {"solve_rate": 0.4, "mean_wall_s": 1.0, "mean_exact_eval_count": 10.0},
                "scheduler_control": {"solve_rate": 0.5, "mean_wall_s": 1.02, "mean_exact_eval_count": 11.0},
            },
            "comparisons": {"scheduler_control": {"comparison_mode": kwargs.get("comparison_mode")}},
        }

    def _fake_stage1_benchmark(**kwargs):
        return {
            "overall": {
                "stage0_selective": {"solve_rate": 0.4, "mean_wall_s": 1.0, "mean_exact_eval_count": 10.0},
                "stage1_scheduler_control": {"solve_rate": 0.5, "mean_wall_s": 1.02, "mean_exact_eval_count": 11.0},
            },
            "comparisons": {"stage1_scheduler_control": {"comparison_mode": kwargs.get("comparison_mode")}},
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.run_scheduler_training",
        _fake_training,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.run_controller_benchmark",
        _fake_controller_benchmark,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.run_stage1_benchmark",
        _fake_stage1_benchmark,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.load_scheduler_bundle",
        lambda _path: {"scheduler_critic_trained": True, "budget_ladder": [1, 2, 4], "threshold_ladder": [0.25]},
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.load_scheduler_dataset_rows",
        lambda _paths: [{"decision_id": "d0", "route_source": "build"}],
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.replay_scheduler_decisions",
        lambda *_args, **_kwargs: {"trained": True, "groups_replayed": 1, "decision_rows": []},
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler_workflow.recommend_scheduler_promotion",
        lambda **_kwargs: {"mode": "scheduler_promotion", "decision": "hold", "meets_promotion_bar": False},
    )

    report = run_scheduler_workflow(
        scheduler_bundle_path="/tmp/base_scheduler.pt",
        scheduler_dataset_paths=("rows_a.json", "rows_b.json"),
        output_dir=str(tmp_path),
        comparison_modes=("matched_exact",),
        train_scheduler_bundle=True,
        route_aliases={"hole": "repair"},
        objective_mode="witness",
        objective_hybrid_mix=0.7,
        controller_targets=("toy",),
        controller_seeds=(0,),
        stage1_targets=("toy",),
        stage1_seeds=(0,),
    )

    assert len(captured_training) == 1
    assert captured_training[0]["dataset_paths"] == ["rows_a.json", "rows_b.json"]
    assert captured_training[0]["init_bundle_path"] == "/tmp/base_scheduler.pt"
    assert captured_training[0]["route_aliases"] == {"hole": "repair"}
    assert captured_training[0]["objective_mode"] == "witness"
    assert captured_training[0]["objective_hybrid_mix"] == 0.7
    derived_path = captured_training[0]["output_path"]
    assert derived_path.endswith("scheduler_bundle.pt")
    assert report["config"]["effective_scheduler_bundle_path"] == derived_path
    assert captured_controller[0]["scheduler_bundle_path"] == derived_path
    assert report["promotion"]["decision"] == "hold"
    summary_path = tmp_path / "scheduler_training_summary.json"
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["sample_prediction"]["trained"] is True
