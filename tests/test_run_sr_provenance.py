# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import json

import torch

from nestynet_sr.run_sr_reports import _build_run_provenance, write_json_report


def test_run_provenance_records_source_dependency_backend_and_rng():
    provenance = _build_run_provenance(
        seed=1234,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert provenance["schema_version"] == 1
    assert provenance["source"]["nestynet_sr"]["module_file"]
    assert provenance["source"]["nestynet"]["module_file"]
    assert provenance["dependencies"]["python"]
    assert provenance["dependencies"]["torch"] == str(torch.__version__)
    assert provenance["runtime"]["device"] == "cpu"
    assert provenance["runtime"]["dtype"] == "torch.float64"
    assert isinstance(provenance["backend"]["deterministic_algorithms"], bool)
    assert provenance["rng"]["reported_seed"] == 1234
    assert provenance["rng"]["repeatable_seed_enabled"] is True
    assert provenance["rng"]["seeded_streams"] == [
        "python_random",
        "numpy_legacy",
        "numpy_generator",
        "torch",
    ]
    assert "candidate_ast" in provenance["rng"]["stageB_candidate_seed_policy"]
    json.dumps(provenance)


def test_json_report_embeds_run_provenance(tmp_path):
    report_path = tmp_path / "toy.report.json"

    write_json_report(
        filepath="toy.csv",
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=77,
        walltime=0.0,
        enable_truth_eval=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    provenance = metadata["provenance"]
    assert metadata["seed"] == 77
    assert metadata["git_hash"] == provenance["source"]["nestynet_sr"]["git_hash"]
    assert provenance["rng"]["reported_seed"] == 77
    assert "nestynet" in provenance["source"]


def test_json_report_persists_stageb_candidate_metrics(tmp_path):
    report_path = tmp_path / "toy.report.json"
    candidate_metrics = {
        "full_rewrite": True,
        "generic_approximant": True,
        "portfolio_val_loss": 1.0e-12,
        "portfolio_noise_floor_raw": 0.0,
        "loss_good_enough_eff": 1.0e-9,
    }

    write_json_report(
        filepath="toy.csv",
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=77,
        walltime=0.0,
        stageB_data={
            "ast": None,
            "phi_expr_str": "x0",
            "y_expr_str": "x0",
            "sympy_meta": {"accepted": True, "parse_success": True},
            "candidate_metrics": candidate_metrics,
        },
        enable_truth_eval=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stageB"]["candidate_metrics"] == candidate_metrics
