# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Surrogate windowing for oscillation-dense datasets (de122 regression).

A fixed-segment surrogate cannot resolve the harmonic content of long
oscillatory records (Duffing third harmonic at 48 segments / 10 periods),
and LM cannot train enough segments for the full span. Few-period windows
restore the segments-per-wavelength ratio at constant model size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from nestynet_sr.run_de import (
    _estimate_oscillation_periods,
    _plan_surrogate_windows,
    _window_data_hp,
)
from nestynet_sr.sr_search.config import DataHyperparams


def test_estimate_oscillation_periods_counts_sine_periods():
    x = np.linspace(0.0, 31.4, 5000)
    y = np.cos(2.0 * x)  # period pi -> ~10 periods
    p = _estimate_oscillation_periods(x, y)
    assert 9.0 <= p <= 11.0


def test_estimate_oscillation_periods_zero_for_monotonic_and_noise():
    x = np.linspace(0.0, 10.0, 500)
    assert _estimate_oscillation_periods(x, np.exp(-x)) == 0.0
    # small noise riding on a monotonic trend stays below the hysteresis
    rng = np.random.default_rng(0)
    y = np.exp(-x) + 1.0e-3 * rng.standard_normal(x.size)
    assert _estimate_oscillation_periods(x, y) <= 1.0


def _write_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("y,x0\n")
        for yi, xi in zip(y, x):
            f.write(f"{yi:.18e},{xi:.18e}\n")


def test_plan_surrogate_windows_splits_oscillatory_dataset(tmp_path: Path):
    x = np.linspace(0.0, 31.4, 3000)
    csv = tmp_path / "osc.csv"
    _write_csv(csv, x, np.cos(2.0 * x))

    out = _plan_surrogate_windows(
        str(csv), x_axis=0, max_periods=4.0, output_dir=str(tmp_path / "run")
    )
    assert len(out) == 3
    total_rows = 0
    last_x_max = -np.inf
    for p in out:
        a = np.genfromtxt(p, delimiter=",", names=True)
        total_rows += int(a.shape[0])
        assert float(a["x0"].min()) >= last_x_max  # contiguous, ordered windows
        last_x_max = float(a["x0"].max())
    assert total_rows == 3000


def test_plan_surrogate_windows_keeps_nonoscillatory_dataset_whole(tmp_path: Path):
    x = np.linspace(0.0, 10.0, 1000)
    csv = tmp_path / "decay.csv"
    _write_csv(csv, x, np.exp(-x))

    out = _plan_surrogate_windows(
        str(csv), x_axis=0, max_periods=4.0, output_dir=str(tmp_path / "run")
    )
    assert out == [str(csv)]


def test_plan_surrogate_windows_disabled_when_max_periods_zero(tmp_path: Path):
    x = np.linspace(0.0, 31.4, 3000)
    csv = tmp_path / "osc.csv"
    _write_csv(csv, x, np.cos(2.0 * x))

    out = _plan_surrogate_windows(
        str(csv), x_axis=0, max_periods=0.0, output_dir=str(tmp_path / "run")
    )
    assert out == [str(csv)]


def test_window_data_hp_scales_to_window_rows():
    base = DataHyperparams(batch_size=2000, ndata_select=2000, ndata_select_val=2000)
    base.data_split_strategy = "interleaved"

    hp = _window_data_hp(base, n_rows=1666)
    assert hp.ndata_select <= 1666 * 0.45 + 1
    assert hp.ndata_select + hp.ndata_select_val <= 1666
    assert hp.batch_size <= hp.ndata_select
    assert hp.data_split_strategy == "interleaved"

    # large windows keep the configured sizes
    hp_big = _window_data_hp(base, n_rows=10000)
    assert hp_big.ndata_select == 2000
    assert hp_big.batch_size == 2000
