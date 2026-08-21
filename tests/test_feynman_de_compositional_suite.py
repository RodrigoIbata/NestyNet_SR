# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FEYNMAN_DE_DIR = REPO_ROOT / "examples" / "feynman_de"
if str(FEYNMAN_DE_DIR) in sys.path:
    sys.path.remove(str(FEYNMAN_DE_DIR))
sys.path.insert(0, str(FEYNMAN_DE_DIR))
_problem_defs_mod = sys.modules.get("problem_defs")
if _problem_defs_mod is not None:
    _problem_defs_path = Path(getattr(_problem_defs_mod, "__file__", "") or "").resolve()
    if _problem_defs_path.parent != FEYNMAN_DE_DIR:
        del sys.modules["problem_defs"]

from problem_defs import default_param_values, load_problems, resolve_rhs  # noqa: E402


BENCHMARK_FILE = REPO_ROOT / "data" / "feynman_de_compositional.txt"
LAUNCHER = REPO_ROOT / "scripts" / "run_feynman_de_compositional_suite.py"


def test_compositional_benchmark_parses_and_rhs_compiles():
    problems = load_problems(BENCHMARK_FILE)
    assert sorted(problems.keys()) == ["890", "891", "892", "893", "900", "901", "902", "903", "904", "905", "906"]

    for pid, problem in problems.items():
        params = default_param_values(problem)
        rhs_fn, rhs_source = resolve_rhs(problem, prefer_manual=False)
        assert rhs_source == "compiled"

        if int(problem.order) == 1:
            out = rhs_fn(0.25, [1.0], params)
            assert isinstance(out, list)
            assert len(out) == 1
            assert np.isfinite(float(out[0]))
        else:
            out = rhs_fn(0.25, [1.0, 0.0], params)
            assert isinstance(out, list)
            assert len(out) == 2
            assert np.isfinite(float(out[0]))
            assert np.isfinite(float(out[1]))


def test_compositional_launcher_dry_run(tmp_path: Path):
    results_root = tmp_path / "results"
    data_dir = tmp_path / "data"
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--dry_run",
            "--fast",
            "--jobs",
            "2",
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
    assert '"ids": [' in proc.stdout
    assert '"engine": "factorized_search_only"' in proc.stdout
    assert '"n_traj": 6' in proc.stdout
    assert '"holdout_last_k": 2' in proc.stdout
    assert '"sim_validate_traj_time_budget_s": 120.0' in proc.stdout
    assert "--benchmark_file" in proc.stdout
    assert "--only 890" in proc.stdout
    assert "--only 906" in proc.stdout
    assert "--engine factorized_search_only" in proc.stdout
    assert (results_root / "launch_manifest.json").exists()


def test_compositional_launcher_skip_generate_falls_back_when_cache_incomplete(tmp_path: Path):
    results_root = tmp_path / "results"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--dry_run",
            "--fast",
            "--skip_generate",
            "--jobs",
            "1",
            "--ids",
            "890",
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
    assert "--only 890" in proc.stdout
    assert "--skip_generate" not in proc.stdout


def test_compositional_launcher_forwards_factorized_search_preset(tmp_path: Path):
    results_root = tmp_path / "results"
    data_dir = tmp_path / "data"
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--dry_run",
            "--ids",
            "890",
            "--engine",
            "factorized_search_only",
            "--factorized-search-preset",
            "compositional",
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
    assert "--factorized-search-preset compositional" in proc.stdout


def test_compositional_launcher_forwards_compositional_fast_preset(tmp_path: Path):
    results_root = tmp_path / "results"
    data_dir = tmp_path / "data"
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--dry_run",
            "--ids",
            "890",
            "--engine",
            "factorized_search_only",
            "--factorized-search-preset",
            "compositional_fast",
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
    assert "--factorized-search-preset compositional_fast" in proc.stdout
