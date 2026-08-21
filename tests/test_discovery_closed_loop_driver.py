# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json

from nestynet_sr.discovery.closed_loop_driver import run_closed_loop_driver
from nestynet_sr.run_SR import write_json_report


def test_closed_loop_driver_resolves_discovery_report_from_main_report(tmp_path):
    discovery_path = tmp_path / "toy.discovery.json"
    discovery_path.write_text(
        json.dumps(
            {
                "mode": "sr_discovery_integration",
                "dataset": "toy.csv",
                "datasets": ["toy.csv"],
                "report_path": str(tmp_path / "toy.report.json"),
                "committee_members": [
                    {
                        "member_id": "m0",
                        "symbolic_structure": ["var", 0],
                        "validation_error": 0.05,
                        "simplicity_score": 1.0,
                    },
                    {
                        "member_id": "m1",
                        "symbolic_structure": ["neg", ["var", 0]],
                        "validation_error": 0.06,
                        "simplicity_score": 0.9,
                    },
                ],
                "experiment_candidates_full": [
                    {
                        "experiment_id": "flat",
                        "observable_predictions": {"m0": 0.0, "m1": 0.0},
                        "cost": 0.1,
                    },
                    {
                        "experiment_id": "spread",
                        "observable_predictions": {"m0": 1.0, "m1": -1.0},
                        "cost": 0.1,
                    },
                ],
                "config": {
                    "beta": 0.0,
                    "gamma": 0.0,
                    "lambda_cost": 0.1,
                    "lambda_noise": 0.1,
                    "lambda_feasibility": 0.1,
                    "y_transform_name": "identity",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "toy.report.json"
    write_json_report(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(report_path),
        device="cpu",
        dtype="float64",
        seed=0,
        walltime=0.0,
        stageA_data=None,
        stageB_data=None,
        de_data=None,
        discovery_summary={
            "enabled": True,
            "results_path": str(discovery_path),
            "committee_member_count": 2,
        },
        enable_truth_eval=False,
    )

    output_path = tmp_path / "toy.closed_loop.json"
    result = run_closed_loop_driver(
        report_path=str(report_path),
        output_path=str(output_path),
    )

    assert result["selected_experiment"]["experiment_id"] == "spread"
    assert output_path.is_file()
