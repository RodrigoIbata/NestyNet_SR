import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from nestynet_sr.coe_promotion import (
    REASON_RESIDUAL_ABOVE_NOISE,
    build_coe_promotion_manifest,
    promotion_manifest_bytes,
    residual_promotion_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "list_coe_promotions.py"


def _report_path(results_dir: Path, problem_id: str) -> Path:
    return results_dir / f"{problem_id}_toy_data.report.json"


def _write_phase(
    results_dir: Path,
    problem_id: str,
    *,
    report: dict | None,
    success: bool = True,
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
                "error": None if success else "Exit code 17",
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


def _eligible_report(*, mse_ratio: float, n_full: int = 100_000) -> dict:
    declared_noise_mse = 2.0e-4
    noise_mse_se = declared_noise_mse * math.sqrt(2.0 / n_full)
    expr = "0.999*x0"
    return {
        "stageC": {"certified": True, "y_expr_str": expr},
        "final_polish": {
            "status": "success",
            "noise_loss_equiv_abs_floor": declared_noise_mse
            * math.sqrt(2.0 / 2_000),
            "recommended": {
                "expr": expr,
                "full_dataset_mse": declared_noise_mse * mse_ratio,
            },
            "full_dataset_snap": {
                "enabled": True,
                "status": "selected",
                "selected_expr": expr,
                "selected_full_mse": declared_noise_mse * mse_ratio,
                "n_full": n_full,
                "loss_equiv_abs_floor": noise_mse_se,
            },
        },
        "final_selection": {
            "source": "final_polish",
            "applied": True,
            "eligible_for_success": True,
            "expr": expr,
            "unit_admissibility": {"checked": True, "valid": True},
        },
    }


def _hard_failure_report() -> dict:
    return {
        "stageC": {
            "certified": False,
            "symbolic_status": "unresolved",
        },
        "final_polish": {"status": "skipped"},
    }


def _problem(manifest: dict, problem_id: str) -> dict:
    return next(row for row in manifest["problems"] if row["id"] == problem_id)


def test_pb111_shaped_residual_is_promoted_without_truth_fields():
    report = _eligible_report(mse_ratio=1.69245)

    evidence = residual_promotion_evidence(report)

    assert evidence["assessable"] is True
    assert evidence["promote"] is True
    assert evidence["reason_code"] == REASON_RESIDUAL_ABOVE_NOISE
    assert evidence["mse_ratio"] == pytest.approx(1.69245)
    assert evidence["excess_z"] > 100.0


def test_residual_requires_configured_ratio_and_significance():
    modest = residual_promotion_evidence(
        _eligible_report(mse_ratio=1.10),
        min_mse_ratio=1.50,
    )
    too_few_rows = residual_promotion_evidence(
        _eligible_report(mse_ratio=1.60, n_full=4),
    )

    assert modest["assessable"] is True
    assert modest["promote"] is False
    assert too_few_rows["mse_ratio"] == pytest.approx(1.60)
    assert too_few_rows["excess_z"] < 5.0
    assert too_few_rows["promote"] is False


def test_default_ratio_threshold_defers_to_significance_gate():
    pb110_shaped = residual_promotion_evidence(
        _eligible_report(mse_ratio=1.0455),
    )
    pb032_shaped = residual_promotion_evidence(
        _eligible_report(mse_ratio=1.0146),
    )

    assert pb110_shaped["min_mse_ratio"] == 1.0
    assert pb110_shaped["excess_z"] > 5.0
    assert pb110_shaped["promote"] is True
    assert pb032_shaped["excess_z"] < 5.0
    assert pb032_shaped["promote"] is False


def test_missing_declared_noise_evidence_does_not_promote():
    report = _eligible_report(mse_ratio=100.0)
    del report["final_polish"]["noise_loss_equiv_abs_floor"]

    evidence = residual_promotion_evidence(report)

    assert evidence["assessable"] is False
    assert evidence["promote"] is False


def test_manifest_combines_hard_failures_and_residual_opportunities(tmp_path):
    results = tmp_path / "results"
    _write_phase(results, "pb001", report=_eligible_report(mse_ratio=1.01))
    _write_phase(results, "pb002", report=_hard_failure_report())
    _write_phase(results, "pb003", report=_eligible_report(mse_ratio=1.75))

    manifest = build_coe_promotion_manifest(
        cheap_results=results,
        coe_results=tmp_path / "results_CoE",
        problem_ids=["pb003", "pb001", "pb002"],
    )

    assert manifest["promote_ids"] == ["pb002", "pb003"]
    assert manifest["counts"] == {
        "promote": 2,
        "hard_failure": 1,
        "retry_coe": 0,
        "eligible_residual_above_noise": 1,
        "not_promoted": 1,
    }
    assert _problem(manifest, "pb002")["promotion_class"] == "hard_failure"
    assert _problem(manifest, "pb003")["promotion_class"] == (
        "eligible_residual_above_noise"
    )


def test_residual_promotions_honor_completed_and_retryable_coe_artifacts(tmp_path):
    results = tmp_path / "results"
    coe = tmp_path / "results_CoE"
    for problem_id in ("pb020", "pb021"):
        _write_phase(results, problem_id, report=_eligible_report(mse_ratio=1.75))
    settled = _eligible_report(mse_ratio=1.05)
    settled["final_selection"]["source"] = "coe_committee"
    settled["coe_committee"] = {"enabled": True, "status": "success"}
    _write_phase(coe, "pb020", report=settled)
    _write_phase(coe, "pb021", report=None, success=False)

    manifest = build_coe_promotion_manifest(
        cheap_results=results,
        coe_results=coe,
        problem_ids=["pb020", "pb021"],
    )

    assert manifest["promote_ids"] == ["pb021"]
    assert _problem(manifest, "pb020")["reason_code"] == (
        "coe_or_campaign_already_settled"
    )
    assert _problem(manifest, "pb021")["promotion_class"] == "retry_coe"


def test_completed_ineligible_coe_is_settled_not_pending(tmp_path):
    results = tmp_path / "results"
    coe = tmp_path / "results_CoE"
    _write_phase(results, "pb022", report=_hard_failure_report())
    _write_phase(coe, "pb022", report=_hard_failure_report())

    manifest = build_coe_promotion_manifest(
        cheap_results=results,
        coe_results=coe,
        problem_ids=["pb022"],
    )

    assert manifest["promote_ids"] == []
    assert _problem(manifest, "pb022")["base_action"] == "terminal_failure"
    assert _problem(manifest, "pb022")["reason_code"] == (
        "coe_or_campaign_already_settled"
    )


def test_promotion_manifest_is_truth_blind_and_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = _eligible_report(mse_ratio=1.75)
    report["truth_eval"] = {"success": True, "rmse_rel": 0.0}
    poisoned = copy.deepcopy(report)
    poisoned["truth_eval"] = {"success": False, "rmse_rel": 999.0}
    poisoned["stageC"]["truth_canary"] = {"success": False}
    _write_phase(first, "pb010", report=report)
    _write_phase(second, "pb010", report=poisoned)

    first_manifest = build_coe_promotion_manifest(
        cheap_results=first,
        coe_results=None,
        problem_ids=["pb010"],
    )
    second_manifest = build_coe_promotion_manifest(
        cheap_results=second,
        coe_results=None,
        problem_ids=["pb010"],
    )

    assert promotion_manifest_bytes(first_manifest) == promotion_manifest_bytes(
        second_manifest
    )


def test_cli_prints_sorted_list_and_writes_evidence_manifest(tmp_path):
    results = tmp_path / "results"
    _write_phase(results, "pb001", report=_eligible_report(mse_ratio=1.01))
    _write_phase(results, "pb002", report=_hard_failure_report())
    _write_phase(results, "pb003", report=_eligible_report(mse_ratio=1.75))
    json_output = tmp_path / "promotion.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--results",
            str(results),
            "--coe-results",
            str(tmp_path / "results_CoE"),
            "--ids",
            "3,1,2",
            "--json-output",
            str(json_output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "pb002 pb003\n"
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["promote_ids"] == ["pb002", "pb003"]
