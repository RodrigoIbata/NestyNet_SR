# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import argparse
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("sympy")

from nestynet_sr.sr_search.factorized_search.oracle_suite import run_oracle_suite


def _mk_spec(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _simple_payload(spec_id: str, *, expr: str) -> dict:
    return {
        "id": spec_id,
        "basis": ["L"],
        "variables": [{"name": "x", "bounds": [0.3, 3.2], "dim": [1]}],
        "constants": [],
        "target": {"expr": expr, "dim": [1]},
    }


def _override_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        max_depth=2,
        poly_degree=3,
        return_topk=3,
        n_fit=48,
        n_probe=64,
        brute_depth=1,
        no_brute_force=False,
        n_seeds=1,
        split_iter_across_seeds=True,
        refine_lbfgs_steps=None,
        refine_num_restarts=None,
        refine_max_variants=None,
        refine_max_params=None,
        refine_linear_combo_enable=None,
        refine_gate_best_factor=None,
        refine_max_trials=None,
    )


def test_run_oracle_suite_smoke(tmp_path: Path):
    s1 = tmp_path / "spec1.json"
    s2 = tmp_path / "spec2.json"
    _mk_spec(s1, _simple_payload("s1", expr="x"))
    _mk_spec(s2, _simple_payload("s2", expr="x + 0.1*x"))

    out_dir = tmp_path / "out"
    payload = run_oracle_suite(
        [s1, s2],
        budgets=[30],
        modes=["refine_off"],
        n_repeats=1,
        seed=3,
        dtype=torch.float64,
        enforce_dims=True,
        success_mse_threshold=1.0,
        verbose=False,
        hp_overrides=_override_namespace(),
        output_dir=out_dir,
        save_individual_reports=False,
    )

    assert payload["n_specs"] == 2
    assert len(payload["rows"]) == 2
    assert len(payload["summary"]) == 1

    rows_csv = out_dir / "oracle_suite_rows.csv"
    summary_csv = out_dir / "oracle_suite_summary.csv"
    results_json = out_dir / "oracle_suite_results.json"
    assert rows_csv.exists()
    assert summary_csv.exists()
    assert results_json.exists()


def test_run_oracle_suite_saves_report_paths_when_individual_reports_enabled(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("s1", expr="x"))

    out_dir = tmp_path / "out_reports"
    payload = run_oracle_suite(
        [spec_path],
        budgets=[20],
        modes=["refine_off"],
        n_repeats=1,
        seed=1,
        dtype=torch.float64,
        enforce_dims=True,
        success_mse_threshold=1.0,
        verbose=False,
        hp_overrides=_override_namespace(),
        output_dir=out_dir,
        save_individual_reports=True,
        jobs=1,
    )

    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert "report_path" in row
    assert Path(row["report_path"]).is_file()
