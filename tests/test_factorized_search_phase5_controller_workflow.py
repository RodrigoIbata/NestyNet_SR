# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nestynet_sr.sr_search.factorized_search.controller_harness import _summarize_pairwise_comparisons
import nestynet_sr.sr_search.factorized_search.phase5_controller_workflow as phase5_mod
from nestynet_sr.sr_search.factorized_search.stage1_benchmark_harness import Stage1RunSummary


def _three_arm_report(*, refine_enable: bool = True) -> dict:
    suffix = "_plus" if refine_enable else ""
    baseline = f"stage0_selective{suffix}"
    hybrid = f"stage1_hybrid_b050{suffix}"
    macro = f"stage1_macro_b050{suffix}"
    return {
        "config": {
            "arms": [baseline, hybrid, macro],
            "solve_mse": 1.0e-8,
        },
        "identical_vs_stage0": {
            hybrid: {"identical_pairs": 0, "total_pairs": 1},
            macro: {"identical_pairs": 0, "total_pairs": 1},
        },
        "runs": [
            {
                "arm": baseline,
                "best_mse": 1.0,
                "elapsed_s": 1.0,
                "repair_selected": 0,
                "macro_selected": 0,
            },
            {
                "arm": hybrid,
                "best_mse": 0.0,
                "elapsed_s": 1.2,
                "repair_selected": 2,
                "macro_selected": 0,
            },
            {
                "arm": macro,
                "best_mse": 0.0,
                "elapsed_s": 1.4,
                "repair_selected": 1,
                "macro_selected": 3,
            },
        ],
    }


def test_run_phase5_controller_workflow_trains_then_benchmarks(monkeypatch, tmp_path: Path):
    spec_path = tmp_path / "toy_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    critic_path = tmp_path / "oracle_bundle.pt"
    critic_path.write_bytes(b"stub")
    calls: dict[str, dict] = {}

    def _fake_pretrain(spec_paths, **kwargs):
        calls["pretrain"] = {
            "spec_paths": [str(path) for path in spec_paths],
            **kwargs,
        }
        return {
            "final_bundle_path": str(critic_path),
            "n_curriculum_rows": 7,
        }

    def _fake_benchmark(**kwargs):
        calls["benchmark"] = dict(kwargs)
        return _three_arm_report(refine_enable=True)

    monkeypatch.setattr(phase5_mod, "run_oracle_pretrain_pipeline", _fake_pretrain)
    monkeypatch.setattr(phase5_mod, "run_stage1_benchmark", _fake_benchmark)

    summary = phase5_mod.run_phase5_controller_workflow(
        output_dir=tmp_path / "phase5",
        specs=[spec_path],
        oracle_seeds=(3, 4),
        benchmark_targets=("addsum",),
        benchmark_seeds=(10,),
        benchmark_blend=0.50,
        benchmark_macro_profile="repair_probe",
        benchmark_refine_enable=True,
    )

    assert summary["used_existing_critic_path"] is False
    assert summary["critic_path"] == str(critic_path)
    assert calls["pretrain"]["spec_paths"] == [str(spec_path)]
    assert tuple(calls["pretrain"]["seeds"]) == (3, 4)
    assert calls["benchmark"]["critic_path"] == str(critic_path)
    assert list(calls["benchmark"]["blends"]) == [0.50]
    assert tuple(calls["benchmark"]["arm_modes"]) == ("priority", "macro")
    assert tuple(calls["benchmark"]["targets"]) == ("addsum",)
    assert tuple(calls["benchmark"]["seeds"]) == (10,)
    assert calls["benchmark"]["macro_profile"] == "repair_probe"

    report_path = tmp_path / "phase5" / "stage1_three_arm_report.json"
    summary_path = tmp_path / "phase5" / "phase5_controller_summary.json"
    assert report_path.is_file()
    assert summary_path.is_file()

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["benchmark_overview"]["baseline_arm"] == "stage0_selective_plus"
    assert saved_summary["benchmark_overview"]["arm_summary"]["stage1_hybrid_b050_plus"]["solved"] == 1


def test_run_phase5_controller_workflow_reuses_existing_critic(monkeypatch, tmp_path: Path):
    critic_path = tmp_path / "existing_bundle.pt"
    critic_path.write_bytes(b"bundle")
    calls: dict[str, dict] = {}

    def _fail_pretrain(*_args, **_kwargs):
        raise AssertionError("pretrain should not run when critic_path is provided")

    def _fake_benchmark(**kwargs):
        calls["benchmark"] = dict(kwargs)
        return _three_arm_report(refine_enable=False)

    monkeypatch.setattr(phase5_mod, "run_oracle_pretrain_pipeline", _fail_pretrain)
    monkeypatch.setattr(phase5_mod, "run_stage1_benchmark", _fake_benchmark)

    summary = phase5_mod.run_phase5_controller_workflow(
        output_dir=tmp_path / "phase5_reuse",
        critic_path=critic_path,
        benchmark_refine_enable=False,
    )

    assert summary["used_existing_critic_path"] is True
    assert summary["pretrain_summary"] is None
    assert calls["benchmark"]["critic_path"] == str(critic_path)
    assert tuple(calls["benchmark"]["arm_modes"]) == ("priority", "macro")
    assert summary["benchmark_overview"]["baseline_arm"] == "stage0_selective"


def test_phase5_benchmark_overview_rejects_non_three_arm_report():
    with pytest.raises(ValueError, match="exactly 3 scheduler arms"):
        phase5_mod._phase5_benchmark_overview(
            {
                "config": {
                    "arms": [
                        "stage0_selective",
                        "stage1_hybrid_b050",
                        "stage1_gate_b050",
                        "stage1_macro_b050",
                    ]
                },
                "runs": [],
            }
        )


def test_stage1_run_summary_matches_pairwise_comparison_surface():
    baseline = Stage1RunSummary(
        arm="stage0_selective_plus",
        target="addsum",
        seed=10,
        n_iter=12,
        max_depth=4,
        refine_enable=True,
        critic_blend=None,
        critic_mode=None,
        macro_enabled=False,
        best_mse=1.0,
        best_expr="x0",
        residual_basins=1,
        n_eval=10,
        elapsed_s=1.0,
        solve_hit=False,
        exact_eval_count=10,
    )
    candidate = Stage1RunSummary(
        arm="stage1_macro_b050_plus",
        target="addsum",
        seed=10,
        n_iter=12,
        max_depth=4,
        refine_enable=True,
        critic_blend=0.50,
        critic_mode="priority",
        macro_enabled=True,
        best_mse=0.25,
        best_expr="(x0+x1)",
        residual_basins=2,
        n_eval=12,
        elapsed_s=1.5,
        solve_hit=False,
        exact_eval_count=12,
    )

    summary = _summarize_pairwise_comparisons([baseline], [candidate], comparison_mode="matched_exact")

    assert summary["n_pairs"] == 1
    assert summary["mean_delta_log_eff"] is not None
    assert summary["mean_delta_log_eff"] > 0.0
