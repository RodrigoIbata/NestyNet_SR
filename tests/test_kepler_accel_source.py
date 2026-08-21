# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""The weathered-profile acceleration seam: FD default vs chart-surrogate option.

Pins three contracts: the default ``accel_source="gradient"`` reproduces the
historical ``np.gradient`` accelerations exactly (no provenance recorded); the
``"surrogate"`` path returns analytic chart-derivative accelerations with a
provenance record (channels, omegas, FD comparison, optional certificate); and
the clean profile refuses ``accel_source`` overrides.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "kepler_ephemeris_real"
_MODULE_PATH = _EXAMPLE_DIR / "kepler_demo_utils.py"
_SPEC = importlib.util.spec_from_file_location("_kepler_accel_source_test_utils", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"unable to load {_MODULE_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


def _circular_orbit_states(n: int = 900, period_days: float = 100.0):
    """A planar circular orbit given as 3D state vectors (z = 0)."""
    t_days = np.arange(n, dtype=np.float64)
    omega = 2.0 * math.pi / period_days
    a = 1.3
    x = a * np.cos(omega * t_days)
    y = a * np.sin(omega * t_days)
    vx = -a * omega * np.sin(omega * t_days)
    vy = a * omega * np.cos(omega * t_days)
    zeros = np.zeros_like(t_days)
    positions = np.column_stack([x, y, zeros])
    velocities = np.column_stack([vx, vy, zeros])
    return t_days, positions, velocities, omega, a


def test_gradient_default_matches_historical_fd_exactly():
    t_days, positions, velocities, _, _ = _circular_orbit_states()
    ds = _MOD._state_series_to_weathered_dataset(
        orbit_id="circ",
        split="train",
        mu=1.0e-4,
        t_days=t_days,
        positions_xyz=positions,
        velocities_xyz=velocities,
    )
    x, y, vx, vy = _MOD._project_series_to_plane(positions, velocities)
    assert np.array_equal(ds.ax, np.gradient(vx, t_days, edge_order=2))
    assert np.array_equal(ds.ay, np.gradient(vy, t_days, edge_order=2))
    assert ds.accel_provenance is None


def test_clean_profile_refuses_surrogate_accel_source():
    spec = _MOD.EphemerisBodySpec(orbit_id="venus", body_name="venus", split="train")
    with pytest.raises(ValueError, match="weathered"):
        _MOD.generate_kepler_dataset(spec, profile="clean", accel_source="surrogate")


def test_unknown_accel_source_rejected():
    t_days, positions, velocities, _, _ = _circular_orbit_states(n=64)
    with pytest.raises(ValueError, match="accel_source"):
        _MOD._state_series_to_weathered_dataset(
            orbit_id="circ",
            split="train",
            mu=1.0e-4,
            t_days=t_days,
            positions_xyz=positions,
            velocities_xyz=velocities,
            accel_source="autograd",
        )


def test_surrogate_accelerations_analytic_and_documented():
    from nestynet.charts import FitConfig

    t_days, positions, velocities, omega, a = _circular_orbit_states()
    ds = _MOD._state_series_to_weathered_dataset(
        orbit_id="circ",
        split="train",
        mu=1.0e-4,
        t_days=t_days,
        positions_xyz=positions,
        velocities_xyz=velocities,
        accel_source="surrogate",
        accel_certificate=True,
        cheap_fit_cfg=FitConfig(segments=8, epochs=120, restarts=1),
        deep_fit_cfg=FitConfig(segments=12, epochs=300, restarts=1),
    )
    prov = ds.accel_provenance
    assert prov is not None and prov["accel_source"] == "chart_surrogate"
    for channel in ("vx", "vy"):
        diag = prov["channels"][channel]
        assert math.isfinite(diag["val_rel_rmse"])
        assert diag["omegas"], "expected the cylinder's circle frequency to be recovered"
        assert abs(diag["omegas"][0] - omega) / omega < 1.0e-3
        assert "identity" in diag["chart"] and "circle" in diag["chart"]
    # analytic accelerations must beat the coarse-cadence FD near the truth
    trim = slice(_MOD._ACCEL_EDGE_TRIM, len(t_days) - _MOD._ACCEL_EDGE_TRIM)
    x, y, vx, vy = _MOD._project_series_to_plane(positions, velocities)
    ax_true = -a * omega * omega * np.cos(omega * t_days)
    rel = np.sqrt(np.mean(np.square(ds.ax[trim] - ax_true[trim]))) / np.sqrt(
        np.mean(np.square(ax_true[trim]))
    )
    assert rel < 1.0e-3
    certificate = prov["certificate"]
    assert math.isfinite(certificate["measured_derivative_rel_rmse"]["x"])
    assert math.isfinite(certificate["gap_ratio"]["y"])


def test_surrogate_accel_cache_roundtrip(tmp_path):
    from nestynet.charts import FitConfig

    t_days, positions, velocities, _, _ = _circular_orbit_states(n=600)
    kwargs = dict(
        orbit_id="circ",
        split="train",
        mu=1.0e-4,
        t_days=t_days,
        positions_xyz=positions,
        velocities_xyz=velocities,
        accel_source="surrogate",
        accel_cache_dir=tmp_path,
        cheap_fit_cfg=FitConfig(segments=8, epochs=80, restarts=1),
        deep_fit_cfg=FitConfig(segments=10, epochs=150, restarts=1),
    )
    first = _MOD._state_series_to_weathered_dataset(**kwargs)
    assert first.accel_provenance["accel_cache"]["hit"] is False
    assert (tmp_path / "circ.npz").exists()

    second = _MOD._state_series_to_weathered_dataset(**kwargs)
    assert second.accel_provenance["accel_cache"]["hit"] is True
    assert np.array_equal(first.ax, second.ax)
    assert np.array_equal(first.ay, second.ay)

    # a changed input must invalidate the entry, not silently reuse it
    perturbed = velocities.copy()
    perturbed[0, 0] += 1.0e-9
    third = _MOD._state_series_to_weathered_dataset(**{**kwargs, "velocities_xyz": perturbed})
    assert third.accel_provenance["accel_cache"]["hit"] is False


def test_cache_readonly_loader_and_rddot(tmp_path):
    from nestynet.charts import FitConfig

    t_days, positions, velocities, _, _ = _circular_orbit_states(n=600)
    _MOD._state_series_to_weathered_dataset(
        orbit_id="circ",
        split="train",
        mu=1.0e-4,
        t_days=t_days,
        positions_xyz=positions,
        velocities_xyz=velocities,
        accel_source="surrogate",
        accel_cache_dir=tmp_path,
        cheap_fit_cfg=FitConfig(segments=8, epochs=80, restarts=1),
        deep_fit_cfg=FitConfig(segments=10, epochs=150, restarts=1),
    )

    loaded = _MOD.load_cached_surrogate_accels(
        tmp_path, "circ",
        t_days=t_days, positions_xyz=positions, velocities_xyz=velocities,
        certificate=False, harmonic=False,
    )
    assert loaded is not None
    ax, ay, provenance = loaded
    assert ax.shape == t_days.shape and provenance["accel_source"] == "chart_surrogate"

    # flag mismatch -> None, never a wrong entry
    assert _MOD.load_cached_surrogate_accels(
        tmp_path, "circ",
        t_days=t_days, positions_xyz=positions, velocities_xyz=velocities,
        certificate=True, harmonic=False,
    ) is None
    assert _MOD.load_cached_surrogate_accels(
        tmp_path, "missing",
        t_days=t_days, positions_xyz=positions, velocities_xyz=velocities,
        certificate=False, harmonic=False,
    ) is None

    rddot = _MOD.cached_surrogate_rddot(
        tmp_path, "circ",
        t_days=t_days, positions_xyz=positions, velocities_xyz=velocities,
        certificate=False, harmonic=False,
    )
    assert rddot is not None and rddot.shape == t_days.shape
    # circular orbit: radial acceleration vanishes (rddot = a_r + r*omega^2 = 0)
    x, y, vx, vy = _MOD._project_series_to_plane(positions, velocities)
    scale = float(np.sqrt(np.mean((vx**2 + vy**2))))
    trim = slice(32, len(t_days) - 32)
    assert float(np.max(np.abs(rddot[trim]))) < 1.0e-3 * scale
