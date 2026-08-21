# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "scripts" / "run_feynman_de_coe_control_suite.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_feynman_de_coe_control import summarize  # noqa: E402


def _write_case_summary(
    case_dir: Path,
    *,
    pid: str,
    status: str,
    selected_engine: str,
    internal_selected_engine: str,
    traj_scores: list[float],
) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "summary.json"
    path.write_text(
        json.dumps(
            {
                "engine": "factorized_de",
                "problems": [
                    {
                        "id": str(pid),
                        "description": f"case {pid}",
                        "status": str(status),
                        "message": "synthetic summary",
                        "selected_engine": str(selected_engine),
                        "internal_selected_engine": str(internal_selected_engine),
                        "first_line_status": "NONE",
                        "rescued_additional": selected_engine != "stlsq",
                        "n_traj": 4,
                        "n_fit_traj": 3,
                        "n_probe_traj": 1,
                        "holdout_last_k": 1,
                        "traj_scores": [
                            {"traj_id": f"ic{i}", "nrmse": float(value)}
                            for i, value in enumerate(traj_scores)
                        ],
                        "canonical_equation": f"eq {pid}",
                        "json_path": str(case_dir / "de.json"),
                        "engines": {
                            "factorized_de": {
                                "selected_engine": str(selected_engine),
                                "internal_selected_engine": str(internal_selected_engine),
                                "internal_selected_engine_mismatch": selected_engine != internal_selected_engine,
                                "factorized_shortlist_size": 5,
                                "factorized_validated_candidates": 2,
                                "factorized_search_shortlist_size": 4,
                                "factorized_search_validated_candidates": 3,
                                "selected_lane": "factorized",
                                "typed_selected_lane": "x_coeff_on_u",
                                "whole_rhs_attempted": selected_engine == "factorized_search",
                                "whole_rhs_attempts_run": 1 if selected_engine == "factorized_search" else 0,
                                "family_gate_skips": 2,
                                "typed_explorer_launches": 1,
                            }
                        },
                    }
                ],
                "counts": {str(status): 1},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_summarize_de_coe_control_extracts_rollout_overrides(tmp_path: Path):
    root = tmp_path / "results"
    _write_case_summary(
        root / "de002",
        pid="002",
        status="PASS",
        selected_engine="factorized",
        internal_selected_engine="factorized_search",
        traj_scores=[0.02, 0.01, 0.03],
    )
    _write_case_summary(
        root / "de010",
        pid="010",
        status="FAIL",
        selected_engine="factorized_search",
        internal_selected_engine="factorized_search",
        traj_scores=[0.2],
    )

    payload = summarize([root])

    assert payload["n_reports"] == 2
    assert payload["n_rows"] == 2
    assert payload["status_counts"] == {"FAIL": 1, "PASS": 1}
    assert payload["selected_engine_counts"] == {"factorized": 1, "factorized_search": 1}
    assert payload["internal_selected_engine_counts"] == {"factorized_search": 2}
    assert payload["selected_lane_counts"] == {"factorized": 2}
    assert payload["typed_selected_lane_counts"] == {"x_coeff_on_u": 2}
    assert payload["rollout_override_count"] == 1
    assert payload["rollout_override_ids"] == ["002"]
    assert payload["validated_candidates_total"] == 10
    assert payload["whole_rhs_attempted"] == 1
    assert payload["whole_rhs_attempts_run"] == 1
    assert payload["family_gate_skips"] == 4
    assert payload["typed_explorer_launches"] == 2

    row = next(row for row in payload["rows"] if row["problem_id"] == "002")
    assert row["rollout_override"] is True
    assert row["selected_lane"] == "factorized"
    assert row["typed_selected_lane"] == "x_coeff_on_u"
    assert row["whole_rhs_attempted"] is False
    assert row["family_gate_skips"] == 2
    assert row["typed_explorer_launches"] == 1
    assert row["worst_traj_nrmse"] == 0.03
    assert row["median_traj_nrmse"] == 0.02
    assert row["factorized_validated_candidates"] == 2
    assert row["factorized_search_validated_candidates"] == 3
    assert row["validated_candidates_total"] == 5


def test_de_coe_control_launcher_dry_run(tmp_path: Path):
    results_root = tmp_path / "results"
    data_dir = tmp_path / "data"
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--dry_run",
            "--ids",
            "002,010",
            "--jobs",
            "2",
            "--results_root",
            str(results_root),
            "--data_dir",
            str(data_dir),
            "--de-coe-mode",
            "adjudicate",
            "--de-coe-csr-on-ties",
            "--de-coe-reservoir-scouts",
            "2",
            "--factorized-de-whole-rhs",
            "auto",
            "--factorized-search-de-refine-mode",
            "rare_final_polish",
            "--factorized-search-max-attempts",
            "1",
        ],
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )

    assert proc.returncode == 0, proc.stderr
    assert '"engine": "factorized_de"' in proc.stdout
    assert '"sim_validate_max_candidates": 3' in proc.stdout
    assert "--only 002" in proc.stdout
    assert "--only 010" in proc.stdout
    assert "--engine factorized_de" in proc.stdout
    assert "--n_points 1500" in proc.stdout
    assert "--sim_validate_max_candidates 3" in proc.stdout
    assert "--de-coe-mode adjudicate" in proc.stdout
    assert "--de-coe-csr-on-ties" in proc.stdout
    assert "--de-coe-reservoir-scouts 2" in proc.stdout
    assert "--factorized-de-whole-rhs auto" in proc.stdout
    assert "--factorized-search-de-refine-mode rare_final_polish" in proc.stdout
    assert "--factorized-search-max-attempts 1" in proc.stdout
    assert "--fast" in proc.stdout
    assert (results_root / "launch_manifest.json").exists()


def test_de_coe_control_launcher_resume_aggregates_existing_summaries(tmp_path: Path):
    results_root = tmp_path / "results"
    data_dir = tmp_path / "data"
    _write_case_summary(
        results_root / "de002",
        pid="002",
        status="PASS",
        selected_engine="factorized",
        internal_selected_engine="factorized_search",
        traj_scores=[0.01],
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--resume",
            "--ids",
            "002",
            "--results_root",
            str(results_root),
            "--data_dir",
            str(data_dir),
        ],
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )

    assert proc.returncode == 0, proc.stderr
    out_path = results_root / "de_coe_control_summary.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["control_summary"]["n_rows"] == 1
    assert payload["control_summary"]["pass_count"] == 1
    assert payload["control_summary"]["rollout_override_count"] == 1
    assert payload["cases"][0]["skipped_existing"] is True
