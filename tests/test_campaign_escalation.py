import copy
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nestynet_sr.campaign_escalation import (
    ACTION_PENDING,
    ACTION_RETRY_CHEAP,
    ACTION_RETRY_COE,
    ACTION_RUN_COE,
    ACTION_SKIP,
    ACTION_TERMINAL_FAILURE,
    build_escalation_manifest,
    classify_report_selection,
    manifest_bytes,
    report_campaign_outcome,
)
from nestynet_sr.run_sr_reports import _update_report_with_campaign_outcome


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "build_coe_escalation_manifest.py"
WRAPPER = PROJECT_ROOT / "scripts" / "run_allstages_escalating.sh"


def _report_path(results_dir: Path, problem_id: str) -> Path:
    return results_dir / f"{problem_id}_toy_data.report.json"


def _write_phase(
    results_dir: Path,
    problem_id: str,
    *,
    report: dict | None,
    success: bool = True,
    error: str | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = results_dir.parent / "data" / f"{problem_id}_toy_data.csv"
    summary = {
        "total_problems": 1,
        "successful": int(success),
        "failed": int(not success),
        "results": [
            {
                "stem": problem_id,
                "filepath": str(dataset),
                "success": success,
                "error": error,
            }
        ],
    }
    (results_dir / f"allstages_suite_summary_{problem_id}.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    if report is not None:
        _report_path(results_dir, problem_id).write_text(
            json.dumps(report),
            encoding="utf-8",
        )


def _eligible_report(*, coe: bool = False) -> dict:
    report = {
        "stageC": {"certified": True, "y_expr_str": "x0"},
        "final_selection": {
            "source": "coe_committee" if coe else "final_polish",
            "applied": True,
            "eligible_for_success": True,
            "expr": "x0",
            "unit_admissibility": {"checked": True, "valid": True},
        },
    }
    if coe:
        report["coe_committee"] = {"enabled": True, "status": "success"}
    return report


def _no_safe_report() -> dict:
    return {
        "stageC": {
            "certified": False,
            "y_expr_str": "x0 + x0/x1",
            "unit_admissibility": {"checked": True, "valid": False},
        },
        "final_polish": {
            "status": "no_safe_unit_valid_replacement",
            "needs_escalation": True,
            "escalation_reason": "final_polish_no_safe_unit_valid_replacement",
        },
        "final_selection": {
            "source": "stageB",
            "applied": False,
            "eligible_for_success": False,
            "expr": "x0 + x0/x1",
        },
    }


def _noiseless_generic_report(
    *,
    val_mse: float = 1.0e-12,
    good_enough_mse: float = 1.0e-9,
    noise_floor: float = 0.0,
    coe: bool = False,
) -> dict:
    report = _eligible_report(coe=coe)
    expr = "(x0**6 + 0.123*x0 + 0.456)/(x0**5 + 0.789)"
    report["final_selection"]["expr"] = expr
    report["stageB"] = {
        "candidate_metrics": {
            "full_rewrite": True,
            "generic_approximant": True,
            "num_nn": 0,
            "has_original_y_validation": False,
            "portfolio_val_loss": float(val_mse),
            "portfolio_noise_floor_raw": float(noise_floor),
            "loss_good_enough_eff": float(good_enough_mse),
        }
    }
    report["final_polish"] = {
        "status": "success",
        "recommended": {
            "expr": expr,
            "val_mse": float(val_mse),
        },
    }
    return report


def _problem(manifest: dict, problem_id: str) -> dict:
    return next(row for row in manifest["problems"] if row["id"] == problem_id)


def test_report_classifier_is_structural_and_validates_coefficient_metadata():
    assert classify_report_selection(_eligible_report()).eligible is True
    assert classify_report_selection({}).reason_code == "no_symbolic_selection"
    assert (
        classify_report_selection(
            {"stageC": {"certified": False, "y_expr_str": "x0"}}
        ).reason_code
        == "stagec_uncertified"
    )
    assert (
        classify_report_selection(_no_safe_report()).reason_code
        == "final_polish_no_safe_unit_valid_replacement"
    )
    invalid_metadata = _eligible_report()
    invalid_metadata["final_selection"]["expr"] = "c*x0"
    invalid_metadata["campaign_outcome"] = {
        "schema_version": 1,
        "policy": "truth_blind_cheap_then_coe_v1",
        "truth_blind": True,
        "selection_eligible": True,
        "reason_code": "selection_eligible",
    }
    assert (
        classify_report_selection(invalid_metadata).reason_code
        == "final_selection_invalid_coefficient_metadata"
    )
    assert (
        classify_report_selection(
            {"stageC": {"certified": True, "y_expr_str": "c*x0"}}
        ).reason_code
        == "stagec_invalid_coefficient_metadata"
    )


def test_manifest_process_failures_retry_same_phase(tmp_path):
    cheap = tmp_path / "cheap"
    _write_phase(
        cheap,
        "pb001",
        report=None,
        success=False,
        error="Exit code 17",
    )
    _write_phase(cheap, "pb002", report=None, success=True)
    (cheap / "allstages_suite_summary_pb003.json").write_text("not JSON")

    manifest = build_escalation_manifest(
        cheap_results=cheap,
        coe_results=tmp_path / "coe",
        problem_ids=["pb000", "pb001", "pb002", "pb003"],
    )

    assert _problem(manifest, "pb000")["action"] == ACTION_PENDING
    assert _problem(manifest, "pb001")["action"] == ACTION_RETRY_CHEAP
    assert _problem(manifest, "pb002")["action"] == ACTION_RETRY_CHEAP
    assert _problem(manifest, "pb003")["action"] == ACTION_RETRY_CHEAP
    assert manifest["run_coe_ids"] == []


def test_manifest_rejects_executable_report_expression_without_running_it(
    monkeypatch,
    tmp_path,
):
    executed = []
    monkeypatch.setattr(os, "getcwd", lambda: executed.append(True) or "/executed")
    report = _eligible_report()
    report["final_selection"]["expr"] = "__import__('os').getcwd()"
    cheap = tmp_path / "cheap"
    _write_phase(cheap, "pb004", report=report)

    manifest = build_escalation_manifest(
        cheap_results=cheap,
        coe_results=tmp_path / "coe",
        problem_ids=["pb004"],
    )

    assert executed == []
    row = _problem(manifest, "pb004")
    assert row["action"] == ACTION_RUN_COE
    assert row["reason_code"] == (
        "cheap_final_selection_invalid_coefficient_metadata"
    )


@pytest.mark.parametrize(
    ("report", "reason_code"),
    [
        (_no_safe_report(), "cheap_final_polish_no_safe_unit_valid_replacement"),
        (
            {"stageC": {"certified": False, "y_expr_str": "x0"}},
            "cheap_stagec_uncertified",
        ),
        ({}, "cheap_no_symbolic_selection"),
    ],
)
def test_completed_cheap_internal_failure_enters_coe_queue(
    tmp_path,
    report,
    reason_code,
):
    cheap = tmp_path / "cheap"
    _write_phase(cheap, "pb007", report=report)

    manifest = build_escalation_manifest(
        cheap_results=cheap,
        coe_results=tmp_path / "coe",
        problem_ids=["pb007"],
    )

    row = _problem(manifest, "pb007")
    assert row["action"] == ACTION_RUN_COE
    assert row["reason_code"] == reason_code


def test_noiseless_generic_cheap_selection_enters_coe_queue(tmp_path):
    report = _noiseless_generic_report(
        val_mse=1.0e-12,
        good_enough_mse=1.0e-9,
    )
    decision = classify_report_selection(report)
    assert decision.eligible is False
    assert decision.reason_code == "noiseless_generic_approximant"

    cheap = tmp_path / "cheap"
    _write_phase(cheap, "pb001", report=report)
    manifest = build_escalation_manifest(
        cheap_results=cheap,
        coe_results=tmp_path / "coe",
        problem_ids=["pb001"],
    )

    row = _problem(manifest, "pb001")
    assert row["action"] == ACTION_RUN_COE
    assert row["reason_code"] == "cheap_noiseless_generic_approximant"


def test_generic_guard_allows_numerically_exact_or_noisy_selection():
    numerically_exact = _noiseless_generic_report(
        val_mse=1.0e-30,
        good_enough_mse=1.0e-9,
    )
    noisy = _noiseless_generic_report(
        val_mse=1.1e-6,
        good_enough_mse=1.0e-8,
        noise_floor=1.0e-6,
    )

    assert classify_report_selection(numerically_exact).eligible is True
    assert classify_report_selection(noisy).eligible is True


def test_generic_guard_does_not_override_completed_coe_adjudication():
    report = _noiseless_generic_report(
        val_mse=1.0e-12,
        good_enough_mse=1.0e-9,
        coe=True,
    )

    decision = classify_report_selection(report)

    assert decision.eligible is True
    assert report_campaign_outcome(report)["action"] == "complete"


def test_generic_guard_also_covers_stagec_only_cheap_report():
    report = _noiseless_generic_report(
        val_mse=1.0e-12,
        good_enough_mse=1.0e-9,
    )
    expr = report["final_selection"]["expr"]
    report.pop("final_selection")
    report.pop("final_polish")
    report["stageC"] = {
        "certified": True,
        "y_expr_str": expr,
    }

    decision = classify_report_selection(report)

    assert decision.eligible is False
    assert decision.reason_code == "noiseless_generic_approximant"


def test_completed_coe_is_resumable_or_terminal(tmp_path):
    cheap = tmp_path / "cheap"
    coe = tmp_path / "coe"
    for problem_id in ("pb010", "pb011", "pb012"):
        _write_phase(cheap, problem_id, report=_no_safe_report())
    _write_phase(coe, "pb010", report=_eligible_report(coe=True))
    _write_phase(
        coe,
        "pb011",
        report=None,
        success=False,
        error="Exit code 9",
    )
    terminal = _no_safe_report()
    terminal["coe_committee"] = {"enabled": True, "status": "no_selection"}
    _write_phase(coe, "pb012", report=terminal)

    manifest = build_escalation_manifest(
        cheap_results=cheap,
        coe_results=coe,
        problem_ids=["pb010", "pb011", "pb012"],
    )

    assert _problem(manifest, "pb010")["action"] == ACTION_SKIP
    assert _problem(manifest, "pb011")["action"] == ACTION_RETRY_COE
    assert _problem(manifest, "pb012")["action"] == ACTION_TERMINAL_FAILURE
    assert manifest["run_coe_ids"] == []


def _replace_truth_fields(value):
    if isinstance(value, dict):
        return {
            key: (
                {"success": not bool(nested.get("success", False)), "poison": 99}
                if "truth" in key.lower() and isinstance(nested, dict)
                else _replace_truth_fields(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_replace_truth_fields(item) for item in value]
    return value


def test_manifest_bytes_ignore_truth_and_file_creation_order(tmp_path):
    reports = {
        "pb020": {
            **_eligible_report(),
            "truth_eval": {"success": True, "rmse_rel": 0.0},
            "stageC": {
                "certified": True,
                "y_expr_str": "x0",
                "truth_canary": {"success": False},
            },
        },
        "pb021": {
            **_no_safe_report(),
            "truth_eval": {"success": False, "rmse_rel": 10.0},
        },
    }
    first = tmp_path / "first" / "results"
    second = tmp_path / "second" / "results"
    for problem_id in reversed(list(reports)):
        _write_phase(first, problem_id, report=reports[problem_id])
    for problem_id in reports:
        _write_phase(
            second,
            problem_id,
            report=_replace_truth_fields(copy.deepcopy(reports[problem_id])),
        )

    first_manifest = build_escalation_manifest(
        cheap_results=first,
        coe_results=tmp_path / "first_coe",
        problem_ids=["pb021", "pb020"],
    )
    second_manifest = build_escalation_manifest(
        cheap_results=second,
        coe_results=tmp_path / "second_coe",
        problem_ids=["pb020", "pb021"],
    )

    assert manifest_bytes(first_manifest) == manifest_bytes(second_manifest)


def test_campaign_outcome_records_settled_phase_without_changing_truth(tmp_path):
    cheap = _no_safe_report()
    cheap["truth_eval"] = {"success": True, "rmse_rel": 0.0}
    cheap_outcome = report_campaign_outcome(cheap)
    assert cheap_outcome == {
        "schema_version": 1,
        "policy": "truth_blind_cheap_then_coe_v1",
        "truth_blind": True,
        "phase": "cheap",
        "selection_eligible": False,
        "action": ACTION_RUN_COE,
        "reason_code": "final_polish_no_safe_unit_valid_replacement",
        "reason": "final polish found no safe unit-valid replacement",
    }

    report_path = tmp_path / "pb001.report.json"
    coe = _eligible_report(coe=True)
    coe["truth_eval"] = {"success": False, "rmse_rel": 99.0}
    report_path.write_text(json.dumps(coe), encoding="utf-8")
    original_truth = copy.deepcopy(coe["truth_eval"])

    outcome = _update_report_with_campaign_outcome(str(report_path))

    assert outcome["phase"] == "coe"
    assert outcome["action"] == "complete"
    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert updated["truth_eval"] == original_truth
    assert updated["campaign_outcome"] == outcome


def test_manifest_cli_build_and_list(tmp_path):
    cheap = tmp_path / "cheap"
    _write_phase(cheap, "pb001", report=_eligible_report())
    _write_phase(cheap, "pb002", report=_no_safe_report())
    output = tmp_path / "manifest.json"

    built = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "build",
            "--cheap-results",
            str(cheap),
            "--coe-results",
            str(tmp_path / "coe"),
            "--output",
            str(output),
            "--ids",
            "pb002,pb001",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    listed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "list",
            "--manifest",
            str(output),
            "--action",
            ACTION_RUN_COE,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert built.returncode == 0, built.stderr
    assert listed.returncode == 0, listed.stderr
    assert listed.stdout.strip() == "pb002"


def _write_fake_runner(campaign: Path) -> tuple[Path, Path]:
    runner = campaign / "fake_runner.py"
    calls = campaign / "calls.jsonl"
    runner.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import os
            from pathlib import Path

            campaign = Path({str(campaign)!r})
            results = Path(os.environ["RESULTS_DIR"])
            results.mkdir(parents=True, exist_ok=True)
            mode = os.environ.get("COE_MODE", "off")
            ids = os.environ.get("IDS", "").replace(",", " ").split()
            with (campaign / "calls.jsonl").open("a") as stream:
                stream.write(json.dumps({{"mode": mode, "ids": ids}}) + "\\n")
            for problem_id in ids:
                failed = mode == "off" and problem_id == "pb003"
                dataset = campaign / "data" / f"{{problem_id}}_toy_data.csv"
                summary = {{
                    "total_problems": 1,
                    "successful": int(not failed),
                    "failed": int(failed),
                    "results": [{{
                        "stem": problem_id,
                        "filepath": str(dataset),
                        "success": not failed,
                        "error": "Exit code 23" if failed else None,
                    }}],
                }}
                (results / f"allstages_suite_summary_{{problem_id}}.json").write_text(
                    json.dumps(summary)
                )
                if failed:
                    continue
                if mode != "off":
                    report = {{
                        "coe_committee": {{"enabled": True, "status": "success"}},
                        "final_selection": {{
                            "source": "coe_committee",
                            "applied": True,
                            "eligible_for_success": True,
                            "expr": "x0",
                            "unit_admissibility": {{"checked": True, "valid": True}},
                        }},
                    }}
                elif problem_id == "pb002":
                    report = {{
                        "final_polish": {{
                            "status": "no_safe_unit_valid_replacement",
                            "needs_escalation": True,
                            "escalation_reason": "final_polish_no_safe_unit_valid_replacement",
                        }},
                        "final_selection": {{
                            "source": "stageB",
                            "applied": False,
                            "eligible_for_success": False,
                            "expr": "x0 + x0/x1",
                        }},
                    }}
                else:
                    report = {{
                        "final_selection": {{
                            "source": "final_polish",
                            "applied": True,
                            "eligible_for_success": True,
                            "expr": "x0",
                            "unit_admissibility": {{"checked": True, "valid": True}},
                        }},
                    }}
                (results / f"{{problem_id}}_toy_data.report.json").write_text(
                    json.dumps(report)
                )
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner, calls


def _wrapper_env(campaign: Path, runner: Path, ids: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CAMPAIGN_ROOT": str(campaign),
            "CAMPAIGN_RUNNER": str(runner),
            "IDS": ids,
            "PYTHON_BIN": sys.executable,
            "RUN_CHEAP": "1",
            "RUN_COE": "1",
        }
    )
    return env


def test_wrapper_runs_only_failed_cheap_case_through_coe_and_resumes(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "nestynet_sr").symlink_to(
        PROJECT_ROOT / "nestynet_sr",
        target_is_directory=True,
    )
    copied_wrapper = campaign / "scripts" / WRAPPER.name
    copied_wrapper.parent.mkdir()
    shutil.copy2(WRAPPER, copied_wrapper)
    runner, calls_path = _write_fake_runner(campaign)
    env = _wrapper_env(campaign, runner, "pb001 pb002")

    first = subprocess.run(
        ["bash", str(copied_wrapper)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls_after_first = [
        json.loads(line)
        for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    second = subprocess.run(
        ["bash", str(copied_wrapper)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls_after_second = [
        json.loads(line)
        for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert calls_after_first == [
        {"mode": "off", "ids": ["pb001", "pb002"]},
        {"mode": "reservoir_discovery", "ids": ["pb002"]},
    ]
    assert calls_after_second == calls_after_first
    manifest = json.loads(
        (campaign / "coe_escalation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"][ACTION_SKIP] == 2
    assert manifest["run_coe_ids"] == []


def test_wrapper_does_not_escalate_cheap_process_failure(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    runner, calls_path = _write_fake_runner(campaign)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=PROJECT_ROOT,
        env=_wrapper_env(campaign, runner, "pb003"),
        capture_output=True,
        text=True,
        check=False,
    )

    calls = [
        json.loads(line)
        for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(
        (campaign / "coe_escalation_manifest.json").read_text(encoding="utf-8")
    )
    assert result.returncode == 1
    assert calls == [{"mode": "off", "ids": ["pb003"]}]
    assert manifest["retry_cheap_ids"] == ["pb003"]
    assert manifest["run_coe_ids"] == []
