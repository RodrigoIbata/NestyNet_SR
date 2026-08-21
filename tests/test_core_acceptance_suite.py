# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
import pickle
from pathlib import Path

import nestynet_sr.run_core_acceptance_suite as acceptance_mod


def test_default_manifest_resolves_cases():
    manifest_path, payload = acceptance_mod.load_acceptance_suite()
    cases = acceptance_mod.resolve_suite_cases(payload, manifest_path=manifest_path)
    assert payload["suite_id"] == "frozen_core_fast"
    assert len(cases) == 7
    assert {
        "pytest_compound_detection",
        "pytest_outer_peel_square_decision",
        "pytest_nonlinear_substitution_screen",
        "pytest_model_selection_complexity",
        "controller_harness_smoke",
        "sr_pb010_no_factorized_search",
        "de_dho_firstclass",
    } == {case["case_id"] for case in cases}


def test_evaluate_sr_report_accepts_exact_case():
    report = {
        "stageA": {"val_loss": 1.0e-12},
        "stageB": {"val_loss": 1.0e-30},
        "stageC": {"y_expr_str": "x0*x1"},
        "truth_eval": {"success": True, "rmse_rel": 0.0, "frac_valid": 1.0},
    }
    expect = {
        "stageA_required": True,
        "stageB_required": True,
        "stageC_required": True,
        "truth_eval_required": True,
        "truth_success": True,
        "truth_rmse_rel_max": 1.0e-12,
        "truth_frac_valid_min": 0.99,
        "stageB_val_loss_max": 1.0e-20,
        "stageC_equivalent_expr": "x1*x0",
    }

    result = acceptance_mod.evaluate_sr_report(report, expect)
    assert result["success"] is True
    assert result["reasons"] == []
    assert result["metrics"]["stageC_expr_equivalent"] is True


def test_evaluate_sr_report_flags_missing_truth_eval():
    report = {
        "stageA": {"val_loss": 1.0e-12},
        "stageB": {"val_loss": 1.0e-8},
        "stageC": {"y_expr_str": "x0*x1"},
    }
    expect = {"truth_eval_required": True, "truth_success": True}

    result = acceptance_mod.evaluate_sr_report(report, expect)
    assert result["success"] is False
    assert any("truth_eval" in reason for reason in result["reasons"])


def test_evaluate_sr_report_rejects_ineligible_diagnostic_incumbent():
    report = {
        "stageC": {"y_expr_str": "x0 + x0/x1"},
        "truth_eval": {"success": True, "rmse_rel": 0.0, "frac_valid": 1.0},
        "final_polish": {"status": "no_safe_unit_valid_replacement"},
        "final_selection": {
            "source": "stageB",
            "applied": False,
            "eligible_for_success": False,
            "reason": "no safe unit-valid replacement",
            "expr": "x0 + x0/x1",
        },
    }

    result = acceptance_mod.evaluate_sr_report(
        report,
        {"truth_eval_required": True, "truth_success": True},
    )

    assert result["success"] is False
    assert result["metrics"]["raw_truth_success"] is True
    assert result["metrics"]["truth_success"] is False
    assert result["metrics"]["final_selection_eligible"] is False
    assert any("not eligible" in reason for reason in result["reasons"])


def test_evaluate_sr_report_accepts_later_eligible_coe_selection():
    report = {
        "stageC": {"y_expr_str": "x0 + x0/x1"},
        "truth_eval": {"success": True, "rmse_rel": 0.0, "frac_valid": 1.0},
        "final_polish": {"status": "no_safe_unit_valid_replacement"},
        "final_selection": {
            "source": "coe_committee",
            "applied": True,
            "eligible_for_success": True,
            "expr": "x0",
            "unit_admissibility": {"checked": True, "valid": True},
        },
    }

    result = acceptance_mod.evaluate_sr_report(
        report,
        {"truth_eval_required": True, "truth_success": True},
    )

    assert result["success"] is True
    assert result["metrics"]["final_selection_eligible"] is True


def test_evaluate_de_report_checks_coefficients_from_sr_artifact(tmp_path: Path):
    pkl_path = tmp_path / "dho.pkl"
    payload = {
        "term_asts": ["u", "u_x0", "u ** 2"],
        "coeffs": [1.0, 1.6, 0.005],
    }
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    report = {
        "de": {
            "enabled": True,
            "order": 2,
            "rms_val": [0.01],
            "stageB": {"val_loss": 1.0e-4},
            "artifacts": {"pkl": str(pkl_path)},
        }
    }
    expect = {
        "de_enabled": True,
        "de_order": 2,
        "de_rms_val_max": 0.1,
        "de_stageb_val_loss_max": 0.01,
        "de_other_terms_abs_max": 0.01,
        "de_expected_coefficients": {
            "u": {"value": 1.0, "abs_tol": 0.05},
            "u_x0": {"value": 1.6, "abs_tol": 0.05},
        },
    }

    result = acceptance_mod.evaluate_de_report("sr_firstclass_de", report, expect)
    assert result["success"] is True
    assert result["metrics"]["de_order"] == 2
    assert result["metrics"]["de_coeff_u"] == 1.0
    assert result["metrics"]["de_coeff_u_x0"] == 1.6


def test_compare_case_summaries_flags_regression():
    baseline = [
        {
            "case_id": "toy",
            "success": True,
            "wall_seconds": 1.0,
            "truth_rmse_rel": 1.0e-8,
        }
    ]
    current = [
        {
            "case_id": "toy",
            "success": False,
            "wall_seconds": 3.0,
            "truth_rmse_rel": 1.0e-4,
        }
    ]

    regressions = acceptance_mod.compare_case_summaries(
        current,
        baseline,
        wall_time_factor=2.0,
        metric_factor=10.0,
    )
    assert len(regressions) == 1
    reason_text = " | ".join(regressions[0]["reasons"])
    assert "baseline passed" in reason_text
    assert "wall_seconds" in reason_text or "truth_rmse_rel" in reason_text


def test_evaluate_command_case_checks_output_token():
    result = acceptance_mod.evaluate_command_case(
        case={"expect": {"stdout_must_contain": ["All tests passed!"]}},
        returncode=0,
        stdout="All tests passed!\n",
        stderr="",
    )
    assert result["success"] is True
    assert result["reasons"] == []


def test_resolve_suite_cases_handles_python_and_pytest(tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    test_file = tmp_path / "test_demo.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_id": "toy",
                "cases": [
                    {"case_id": "script", "kind": "python", "script": str(script)},
                    {"case_id": "pytest", "kind": "pytest", "paths": [str(test_file)]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _, payload = acceptance_mod.load_acceptance_suite(manifest_path)
    cases = acceptance_mod.resolve_suite_cases(payload, manifest_path=manifest_path)
    assert cases[0]["kind"] == "python"
    assert cases[1]["kind"] == "pytest"
    assert Path(cases[0]["script"]).is_file()
    assert Path(cases[1]["paths"][0]).is_file()


def test_main_writes_outputs(monkeypatch, tmp_path: Path):
    manifest = {
        "suite_id": "toy_suite",
        "cases": [
            {
                "case_id": "toy_case",
                "kind": "sr",
                "filepath": "data/pb010_I_12_5_data.csv",
                "expect": {},
            }
        ],
    }
    manifest_path = tmp_path / "toy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _fake_run_case(case, *, output_dir, defaults=None, python_executable=None):
        return {
            "case_id": str(case["case_id"]),
            "kind": str(case["kind"]),
            "success": True,
            "returncode": 0,
            "timed_out": False,
            "wall_seconds": 0.25,
            "report_path": str(output_dir / "toy.report.json"),
            "log_path": str(output_dir / "toy.log"),
            "command": ["python", "fake.py"],
            "reasons": [],
            "truth_rmse_rel": 0.0,
        }

    monkeypatch.setattr(acceptance_mod, "run_case", _fake_run_case)

    rc = acceptance_mod.main(
        [
            "--suite_manifest",
            str(manifest_path),
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    results_json = tmp_path / "out" / "core_acceptance_results.json"
    summary_csv = tmp_path / "out" / "core_acceptance_case_summary.csv"
    assert results_json.is_file()
    assert summary_csv.is_file()
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "toy_suite"
    assert payload["n_pass"] == 1


def test_main_can_bless_baseline(monkeypatch, tmp_path: Path):
    manifest = {
        "suite_id": "toy_suite",
        "cases": [
            {
                "case_id": "toy_case",
                "kind": "sr",
                "filepath": "data/pb010_I_12_5_data.csv",
                "expect": {},
            }
        ],
    }
    manifest_path = tmp_path / "toy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _fake_run_case(case, *, output_dir, defaults=None, python_executable=None):
        return {
            "case_id": str(case["case_id"]),
            "kind": str(case["kind"]),
            "success": True,
            "returncode": 0,
            "timed_out": False,
            "wall_seconds": 0.5,
            "report_path": str(output_dir / "toy.report.json"),
            "log_path": str(output_dir / "toy.log"),
            "command": ["python", "fake.py"],
            "reasons": [],
        }

    monkeypatch.setattr(acceptance_mod, "run_case", _fake_run_case)

    rc = acceptance_mod.main(
        [
            "--suite_manifest",
            str(manifest_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--bless_baseline",
            str(tmp_path / "baselines"),
        ]
    )
    assert rc == 0
    baseline_json = tmp_path / "baselines" / "toy_suite.baseline.json"
    assert baseline_json.is_file()
    payload = json.loads(baseline_json.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "toy_suite"
    assert payload["blessed_baseline"]["source_output_dir"] == str(tmp_path / "out")
