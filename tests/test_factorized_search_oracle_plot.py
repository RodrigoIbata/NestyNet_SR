# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import csv
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from nestynet_sr.sr_search.factorized_search.oracle_plot import load_summary_csv, plot_suite_summary


def test_plot_suite_summary_from_csv(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    rows = [
        {
            "mode": "refine_off",
            "budget": 100,
            "n_runs": 4,
            "solve_rate": 0.25,
            "best_mse_median": 1.0e-2,
            "best_mse_mean": 2.0e-2,
            "wall_seconds_mean": 0.5,
        },
        {
            "mode": "refine_on",
            "budget": 100,
            "n_runs": 4,
            "solve_rate": 0.5,
            "best_mse_median": 2.0e-3,
            "best_mse_mean": 4.0e-3,
            "wall_seconds_mean": 0.7,
        },
        {
            "mode": "refine_off",
            "budget": 1000,
            "n_runs": 4,
            "solve_rate": 0.75,
            "best_mse_median": 3.0e-4,
            "best_mse_mean": 5.0e-4,
            "wall_seconds_mean": 1.2,
        },
        {
            "mode": "refine_on",
            "budget": 1000,
            "n_runs": 4,
            "solve_rate": 1.0,
            "best_mse_median": 8.0e-5,
            "best_mse_mean": 1.1e-4,
            "wall_seconds_mean": 1.6,
        },
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    loaded = load_summary_csv(summary_csv)
    out = plot_suite_summary(loaded, output_dir=tmp_path / "plots", title_prefix="unit-test")

    for path in out.values():
        p = Path(path)
        assert p.exists()
        assert p.stat().st_size > 0
