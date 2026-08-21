# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json

import nestynet_sr.sr_search.factorized_search.scheduler_promotion as promotion_mod


def _controller_report(*, macro_solve: float, sched_solve: float, macro_wall: float, sched_wall: float) -> dict:
    return {
        "arm_overall": {
            "macro": {
                "solve_rate": float(macro_solve),
                "mean_wall_s": float(macro_wall),
                "median_eff_mse": 1.0e-6,
                "mean_exact_eval_count": 20.0,
                "route_usage": {"build": 10, "inverse": 3, "repair": 4, "hole": 0},
            },
            "scheduler_control": {
                "solve_rate": float(sched_solve),
                "mean_wall_s": float(sched_wall),
                "median_eff_mse": 8.0e-7,
                "mean_exact_eval_count": 21.0,
                "route_usage": {"build": 8, "inverse": 5, "repair": 5, "hole": 2},
            },
        }
    }


def _stage1_report(*, bad_calibration: bool = False, high_ineligible_budget: bool = False, negative_eligible_lift: bool = False) -> dict:
    cal_gap = 0.35 if bad_calibration else 0.04
    ineligible_budget = 3 if high_ineligible_budget else 1
    eligible_actual_0 = 0.80 if negative_eligible_lift else 0.50
    eligible_actual_1 = 0.50 if negative_eligible_lift else 0.30
    return {
        "scheduler_replay": {
            "trained": True,
            "groups_replayed": 4,
            "groups_with_actual_choice": 4,
            "top1_hit_rate": 0.75,
            "mean_regret": 0.10,
            "actual_mean_regret": 0.24,
            "mean_wasted_budget": 0.25,
            "actual_mean_wasted_budget": 0.50,
            "decision_rows": [
                {
                    "group_id": "g0",
                    "oracle_utility": 0.90,
                    "predicted_utility": 0.72,
                    "actual_utility": eligible_actual_0,
                    "predicted_budget": 2,
                    "actual_budget": 2,
                },
                {
                    "group_id": "g1",
                    "oracle_utility": 0.55,
                    "predicted_utility": 0.43,
                    "actual_utility": eligible_actual_1,
                    "predicted_budget": 1,
                    "actual_budget": 1,
                },
                {
                    "group_id": "g2",
                    "oracle_utility": 0.0,
                    "predicted_utility": -0.05,
                    "actual_utility": -0.10,
                    "predicted_budget": ineligible_budget,
                    "actual_budget": 1,
                },
                {
                    "group_id": "g3",
                    "oracle_utility": -0.02,
                    "predicted_utility": -0.02,
                    "actual_utility": -0.04,
                    "predicted_budget": 1,
                    "actual_budget": 1,
                },
            ],
            "calibration_by_route": {
                "build": {
                    "count": 10,
                    "mean_prob": 0.45,
                    "empirical_rate": 0.45 + cal_gap,
                    "brier": 0.16,
                },
                "repair": {
                    "count": 8,
                    "mean_prob": 0.30,
                    "empirical_rate": 0.32,
                    "brier": 0.18,
                },
            },
            "calibration_by_depth": {
                "0-2": {
                    "count": 18,
                    "mean_prob": 0.38,
                    "empirical_rate": 0.40,
                    "brier": 0.17,
                }
            },
            "calibration_by_budget": {
                "1": {
                    "count": 10,
                    "mean_prob": 0.34,
                    "empirical_rate": 0.36,
                    "brier": 0.18,
                },
                "2": {
                    "count": 8,
                    "mean_prob": 0.49,
                    "empirical_rate": 0.51,
                    "brier": 0.16,
                },
            },
        }
    }


def test_recommend_scheduler_promotion_promotes_when_all_gates_pass():
    report = promotion_mod.recommend_scheduler_promotion(
        controller_report=_controller_report(macro_solve=0.56, sched_solve=0.58, macro_wall=10.0, sched_wall=10.4),
        stage1_report=_stage1_report(),
    )

    assert report["decision"] == "promote"
    assert report["meets_promotion_bar"] is True
    assert report["gates"]["eligible_utility"]["passed"] is True
    assert report["gates"]["ineligible_tax"]["passed"] is True
    assert report["gates"]["calibration"]["passed"] is True
    assert report["gates"]["online_noninferior"]["passed"] is True


def test_recommend_scheduler_promotion_holds_when_gates_fail():
    report = promotion_mod.recommend_scheduler_promotion(
        controller_report=_controller_report(macro_solve=0.60, sched_solve=0.52, macro_wall=10.0, sched_wall=12.5),
        stage1_report=_stage1_report(
            bad_calibration=True,
            high_ineligible_budget=True,
            negative_eligible_lift=True,
        ),
    )

    assert report["decision"] == "hold"
    assert report["meets_promotion_bar"] is False
    assert report["gates"]["eligible_utility"]["passed"] is False
    assert report["gates"]["ineligible_tax"]["passed"] is False
    assert report["gates"]["calibration"]["passed"] is False
    assert report["gates"]["online_noninferior"]["passed"] is False


def test_scheduler_promotion_main_writes_output(tmp_path):
    controller_path = tmp_path / "controller_report.json"
    stage1_path = tmp_path / "stage1_report.json"
    out_path = tmp_path / "scheduler_promotion.json"
    controller_path.write_text(
        json.dumps(_controller_report(macro_solve=0.56, sched_solve=0.58, macro_wall=10.0, sched_wall=10.4)),
        encoding="utf-8",
    )
    stage1_path.write_text(json.dumps(_stage1_report()), encoding="utf-8")

    rc = promotion_mod.main(
        [
            "--controller_report",
            str(controller_path),
            "--stage1_report",
            str(stage1_path),
            "--output",
            str(out_path),
        ]
    )

    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["mode"] == "scheduler_promotion"
    assert report["decision"] == "promote"
