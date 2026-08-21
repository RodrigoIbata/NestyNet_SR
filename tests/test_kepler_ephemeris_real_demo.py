# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "kepler_ephemeris_real"
_MODULE_PATH = _EXAMPLE_DIR / "kepler_demo_utils.py"
_SPEC = importlib.util.spec_from_file_location("_kepler_ephemeris_real_test_utils", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"unable to load {_MODULE_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

_FETCHER_PATH = _EXAMPLE_DIR / "fetch_horizons_vectors.py"
_FETCHER_SPEC = importlib.util.spec_from_file_location(
    "_kepler_ephemeris_real_fetcher_test_utils", _FETCHER_PATH
)
if _FETCHER_SPEC is None or _FETCHER_SPEC.loader is None:
    raise ImportError(f"unable to load {_FETCHER_PATH}")
_FETCHER = importlib.util.module_from_spec(_FETCHER_SPEC)
sys.modules[_FETCHER_SPEC.name] = _FETCHER
_FETCHER_SPEC.loader.exec_module(_FETCHER)

DEFAULT_PROFILE = _MOD.DEFAULT_PROFILE
DEFAULT_PROVIDER = _MOD.DEFAULT_PROVIDER
DEFAULT_RAW_MANIFEST_PATH = _MOD.DEFAULT_RAW_MANIFEST_PATH
DEFAULT_SOLAR_MU_AU_DAY = _MOD.DEFAULT_SOLAR_MU_AU_DAY
analyze_kepler_reduced_family = _MOD.analyze_kepler_reduced_family
build_default_kepler_datasets = _MOD.build_default_kepler_datasets
build_default_orbit_specs = _MOD.build_default_orbit_specs
build_generation_provenance = _MOD.build_generation_provenance
load_generation_provenance = _MOD.load_generation_provenance
load_kepler_datasets_from_manifest = _MOD.load_kepler_datasets_from_manifest
parse_horizons_vectors_text = _FETCHER.parse_horizons_vectors_text


def _scan_row(summary: dict, exponent: float) -> dict:
    rows = list(summary["power_scan"]["rows"])
    return min(rows, key=lambda row: abs(float(row["exponent"]) - float(exponent)))


def test_kepler_ephemeris_astropy_builtin_clean_profile_recovers_two_body_family():
    datasets = build_default_kepler_datasets(
        mu=float(DEFAULT_SOLAR_MU_AU_DAY),
        provider="astropy_builtin",
        profile="clean",
        years=5.0,
        cadence_days=2.0,
    )
    summary = analyze_kepler_reduced_family(
        datasets,
        power_exponents=np.linspace(1.8, 2.2, 41, dtype=np.float64),
    )

    assert len(summary["orbit_registry"]) == 6
    assert summary["stage_a"]["max_rel_error"] < 1.0e-12
    assert summary["stage_b_all"]["mu_abs_error"] < 1.0e-12
    assert summary["stage_b_all"]["max_k_abs_error"] < 1.0e-12
    assert summary["energy"]["coeff_max_abs_error"] < 1.0e-12
    assert summary["power_scan"]["best_holdout_exponent"] == 2.0
    assert _scan_row(summary, 2.0)["holdout_mean_rmse"] < 1.0e-12


def test_parse_horizons_vectors_text_normalizes_units():
    text = """
API VERSION: 1.2
Target body name: 4 Vesta (A807 FA)               {source: JPL#36}
Center body name: Sun (10)                        {source: DE441}
$$SOE
2451544.500000000, A.D. 2000-Jan-01 00:00:00.0000,  1.495978707000000E+08,  0.000000000000000E+00,  0.000000000000000E+00,  0.000000000000000E+00,  1.731456836805556E+01,  0.000000000000000E+00,
2451545.500000000, A.D. 2000-Jan-02 00:00:00.0000,  1.495978707000000E+08,  1.495978707000000E+06,  0.000000000000000E+00,  0.000000000000000E+00,  1.731456836805556E+01,  0.000000000000000E+00,
$$EOE
"""
    parsed = parse_horizons_vectors_text(text)
    rows = parsed["rows"]

    assert parsed["target_name"].startswith("4 Vesta")
    assert parsed["center_name"].startswith("Sun")
    assert len(rows) == 2
    assert abs(float(rows[0]["x_au"]) - 1.0) < 1.0e-12
    assert abs(float(rows[0]["vy_au_per_d"]) - 0.01) < 1.0e-12
    assert float(rows[0]["t_day"]) == 0.0
    assert float(rows[1]["t_day"]) == 1.0


def test_kepler_ephemeris_real_manifest_resolves_to_local_copy():
    specs = build_default_orbit_specs(
        provider="raw_csv",
        raw_manifest=DEFAULT_RAW_MANIFEST_PATH,
    )

    assert len(specs) == 10
    for spec in specs:
        assert spec.csv_path is not None
        path = Path(spec.csv_path)
        assert path.exists()
        assert "examples/kepler_ephemeris_real/data/raw" in str(path)

    provenance = build_generation_provenance(
        provider=DEFAULT_PROVIDER,
        profile=DEFAULT_PROFILE,
        mu=float(DEFAULT_SOLAR_MU_AU_DAY),
        start_date="1980-01-01",
        years=30.0,
        cadence_days=1.0,
        raw_manifest=DEFAULT_RAW_MANIFEST_PATH,
    )
    assert provenance["dataset_family"] == "kepler_ephemeris_real"
    raw_rows = list(provenance["raw_manifest_rows"])
    assert len(raw_rows) == 10
    assert all("examples/kepler_ephemeris_real/data/raw" in str(row["csv_path"]) for row in raw_rows)


def test_kepler_ephemeris_real_weathered_profile_stays_close_to_kepler():
    datasets = build_default_kepler_datasets(
        mu=float(DEFAULT_SOLAR_MU_AU_DAY),
        provider=DEFAULT_PROVIDER,
        profile=DEFAULT_PROFILE,
        raw_manifest=DEFAULT_RAW_MANIFEST_PATH,
    )
    summary = analyze_kepler_reduced_family(
        datasets,
        power_exponents=np.linspace(1.8, 2.2, 41, dtype=np.float64),
    )

    assert len(summary["orbit_registry"]) == 10
    assert summary["stage_a"]["max_rel_error"] < 5.0e-4
    assert summary["stage_b_all"]["mu_abs_error"] < 2.0e-5
    assert summary["stage_b_all"]["max_k_abs_error"] < 5.0e-5
    assert summary["energy"]["coeff_max_abs_error"] < 6.0e-2
    assert abs(float(summary["power_scan"]["best_holdout_exponent"]) - 2.0) <= 0.03
    assert _scan_row(summary, 2.0)["holdout_mean_rmse"] < _scan_row(summary, 1.8)["holdout_mean_rmse"]
    assert _scan_row(summary, 2.0)["holdout_mean_rmse"] < _scan_row(summary, 2.2)["holdout_mean_rmse"]


def test_kepler_ephemeris_real_generate_cli_defaults_write_weathered_manifest(tmp_path: Path):
    script = _EXAMPLE_DIR / "generate_kepler_data.py"
    subprocess.run(
        [sys.executable, str(script), "--output_root", str(tmp_path)],
        check=True,
    )

    loaded = load_generation_provenance(tmp_path / "data")
    assert loaded is not None
    assert loaded["provider"] == "raw_csv"
    assert loaded["profile"] == "weathered"
    assert loaded["dataset_family"] == "kepler_ephemeris_real"

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    rows = list(manifest["orbits"])
    assert len(rows) == 10
    assert Path(rows[0]["combined_csv"]).exists()

    datasets = load_kepler_datasets_from_manifest(tmp_path / "data")
    assert len(datasets) == 10
