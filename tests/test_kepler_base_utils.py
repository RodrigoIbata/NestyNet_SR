# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "kepler_ephemeris_real"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import make_paper_figures as kepler_figures
import run_class_sr_discovery as kepler_runner
import kepler_demo_utils as _real_utils
from kepler_demo_utils import (
    _jsonable,
    analyze_classsr_probe_stability,
    analyze_kepler_reduced_family,
    analyze_symbolic_kepler_family,
    build_default_kepler_datasets,
    evaluate_symbolic_holdout_generalization,
    extract_inverse_power_coefficients_from_predictor,
    fit_areal_law,
    fit_radial_family,
    merge_symbolic_kepler_tables,
    suggest_symbolic_readout_points_from_r_values,
    target_filepaths,
    write_generated_artifacts,
)


_SYNTHETIC_KW = dict(provider="astropy_builtin", profile="clean")


def _scan_row(summary: dict, exponent: float) -> dict:
    rows = list(summary["power_scan"]["rows"])
    return min(rows, key=lambda row: abs(float(row["exponent"]) - float(exponent)))


def test_kepler_reduced_direct_fit_recovers_family_and_invariant():
    datasets = build_default_kepler_datasets(
        seed=321,
        train_samples=256,
        validation_samples=256,
        holdout_samples=512,
        **_SYNTHETIC_KW,
    )
    summary = analyze_kepler_reduced_family(
        datasets,
        power_exponents=np.linspace(1.7, 2.3, 61, dtype=np.float64),
    )

    assert summary["stage_a"]["max_rel_error"] < 1.0e-10
    assert summary["stage_b_all"]["mu_abs_error"] < 1.0e-10
    assert summary["stage_b_all"]["max_k_abs_error"] < 1.0e-10
    assert abs(float(summary["coefficient_lift"]["intercept"])) < 1.0e-10
    assert abs(float(summary["coefficient_lift"]["slope"]) - 1.0) < 1.0e-10
    assert summary["energy"]["coeff_max_abs_error"] < 1.0e-10
    assert summary["power_scan"]["best_holdout_exponent"] == 2.0
    assert summary["hamiltonian"]["consistency"]["max_theta_rmse"] < 1.0e-10
    assert summary["hamiltonian"]["consistency"]["max_radial_rmse"] < 1.0e-10
    assert summary["hamiltonian"]["consistency"]["max_natural_reduced_energy_abs_error"] < 1.0e-10
    assert summary["hamiltonian"]["consistency"]["max_natural_cartesian_energy_abs_error"] < 1.0e-10
    assert "0.5 * p_r^2" in summary["hamiltonian"]["recovered_formulas"]["natural_reduced_plain"]

    exact_row = _scan_row(summary, 2.0)
    wrong_row = _scan_row(summary, 1.8)
    assert exact_row["holdout_mean_rmse"] < wrong_row["holdout_mean_rmse"]


def test_kepler_reduced_artifact_writer_creates_manifest_and_targets(tmp_path: Path):
    datasets = build_default_kepler_datasets(
        seed=222,
        train_samples=64,
        validation_samples=64,
        holdout_samples=96,
        **_SYNTHETIC_KW,
    )
    result = write_generated_artifacts(tmp_path, datasets)

    manifest_path = Path(result["manifest_path"])
    metadata_path = Path(result["metadata_path"])
    assert manifest_path.exists()
    assert metadata_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(manifest["orbits"]) == 6
    first_row = manifest["orbits"][0]
    assert Path(first_row["combined_csv"]).exists()
    assert Path(first_row["omega_csv"]).exists()
    assert Path(first_row["rddot_csv"]).exists()
    assert set(metadata_rows[0]) == {"orbit_index", "split_index"}


def test_extract_inverse_power_coefficients_from_predictor_recovers_kepler_like_form():
    result = extract_inverse_power_coefficients_from_predictor(
        lambda r: 1.75 / (r ** 3) - 0.9 / (r ** 2),
        exponents=[3.0, 2.0],
        probe_points=np.asarray([0.6, 0.8, 1.1, 1.5, 2.1, 3.0], dtype=np.float64),
    )

    assert abs(float(result["intercept"])) < 1.0e-12
    assert np.allclose(result["coeffs"], np.asarray([1.75, -0.9]), atol=1.0e-12)
    assert float(result["probe_rmse"]) < 1.0e-12


def test_extract_inverse_power_coefficients_filters_nonfinite_outputs():
    sample_points = suggest_symbolic_readout_points_from_r_values(
        np.linspace(0.5, 3.0, 200, dtype=np.float64),
        n_points=21,
    )
    result = extract_inverse_power_coefficients_from_predictor(
        lambda r: float("inf") if abs(r - sample_points[5]) < 1.0e-12 else 1.75 / (r ** 3) - 0.9 / (r ** 2),
        exponents=[3.0, 2.0],
        probe_points=sample_points,
    )

    assert np.allclose(result["coeffs"], np.asarray([1.75, -0.9]), atol=1.0e-12)
    assert int(result["power_fit_stats"]["n_requested_samples"]) == int(sample_points.size)
    assert int(result["power_fit_stats"]["n_samples"]) == int(sample_points.size - 1)


def test_analyze_symbolic_kepler_family_jointly_refits_shared_mu_from_symbolic_samples():
    datasets = build_default_kepler_datasets(
        seed=246,
        train_samples=96,
        validation_samples=96,
        holdout_samples=96,
        **_SYNTHETIC_KW,
    )[:3]

    merged_rows = []
    for idx, dataset in enumerate(datasets):
        extracted = extract_inverse_power_coefficients_from_predictor(
            lambda r, h=dataset.h, mu=dataset.mu: (h * h) / (r ** 3) - mu / (r ** 2),
            exponents=[3.0, 2.0],
            probe_points=np.asarray(dataset.r, dtype=np.float64),
        )
        merged_rows.append(
            {
                "dataset_id": dataset.orbit_id,
                "ell": float(dataset.h),
                "k": float(dataset.h * dataset.h + 1.0e-2 * (idx + 1)),
                "minus_mu": float(-(dataset.mu + 2.0e-3 * (idx + 1))),
                "omega_intercept": 0.0,
                "rddot_intercept": 0.0,
                "omega_probe_rmse": 0.0,
                "rddot_probe_rmse": 0.0,
                "omega_extraction_mode": "analytic_fit",
                "rddot_extraction_mode": "analytic_fit",
                "rddot_power_fit_stats": extracted["power_fit_stats"],
            }
        )

    summary = analyze_symbolic_kepler_family(merged_rows, datasets=datasets)

    assert summary["stage_b"]["mu_refit_method"] == "joint_symbolic_rddot_fit"
    assert abs(float(summary["stage_b"]["mu_mean"]) - float(datasets[0].mu)) < 1.0e-10
    assert summary["stage_b"]["max_k_abs_error"] < 1.0e-10
    assert abs(float(summary["coefficient_lift"]["intercept"])) < 1.0e-10
    assert abs(float(summary["coefficient_lift"]["slope"]) - 1.0) < 1.0e-10
    assert summary["energy"]["coeff_max_abs_error"] < 1.0e-10


def test_run_class_sr_discovery_uses_multi_dataset_result_stem():
    filepaths = [
        Path("examples/kepler_ephemeris_real/data/omega/omega_orbit_train_01.csv"),
        Path("examples/kepler_ephemeris_real/data/omega/omega_orbit_validation_01.csv"),
        Path("examples/kepler_ephemeris_real/data/omega/omega_orbit_holdout_01.csv"),
    ]

    stem = kepler_runner._derive_result_base_filename(filepaths)

    assert stem == "omega_orbit_multi3"


def test_build_run_sr_cmd_omits_param_metadata_by_default():
    cmd = kepler_runner._build_run_sr_cmd(
        target_name="omega",
        filepaths=[Path("examples/kepler_ephemeris_real/data/omega/omega_orbit_train_01.csv")],
        metadata_json=None,
        fast=True,
        class_cv_threshold=0.15,
        ndata_train=64,
        ndata_val=32,
        batch_size=32,
        class_sr_max_points=64,
    )

    assert "--class_param_sr_metadata" not in cmd


def test_build_run_sr_cmd_accepts_innocuous_param_metadata():
    cmd = kepler_runner._build_run_sr_cmd(
        target_name="rddot",
        filepaths=[Path("examples/kepler_ephemeris_real/data/rddot/rddot_orbit_train_01.csv")],
        metadata_json='[{"orbit_index": 0.0, "split_index": 0.0}]',
        fast=False,
        class_cv_threshold=0.15,
        ndata_train=64,
        ndata_val=32,
        batch_size=32,
        class_sr_max_points=64,
    )

    assert "--class_param_sr_metadata" in cmd


def test_kepler_target_filepaths_follow_manifest(tmp_path: Path):
    data_dir = tmp_path / "data"
    omega_dir = data_dir / "omega"
    omega_dir.mkdir(parents=True)

    keep_a = omega_dir / "omega_orbit_train_01.csv"
    keep_b = omega_dir / "omega_orbit_holdout_01.csv"
    stale = omega_dir / "omega_orbit_validation_99.csv"
    for path in (keep_a, keep_b, stale):
        np.savetxt(
            path,
            np.column_stack(
                [
                    np.linspace(0.0, 1.0, 8),
                    np.linspace(1.0, 2.0, 8),
                ]
            ),
            delimiter=",",
            header="y,x0",
            comments="",
        )

    manifest = {
        "orbits": [
            {"orbit_id": "orbit_train_01", "split": "train", "omega_csv": str(keep_a)},
            {"orbit_id": "orbit_holdout_01", "split": "holdout", "omega_csv": str(keep_b)},
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    filepaths = target_filepaths(data_dir, "omega")
    holdout_only = target_filepaths(data_dir, "omega", splits=["holdout"])

    assert filepaths == [keep_a, keep_b]
    assert holdout_only == [keep_b]


def test_make_kepler_paper_figures_writes_png_and_pdf(tmp_path: Path, monkeypatch):
    datasets = build_default_kepler_datasets(
        seed=444,
        train_samples=64,
        validation_samples=64,
        holdout_samples=96,
        **_SYNTHETIC_KW,
    )
    write_generated_artifacts(tmp_path, datasets)

    direct_summary = analyze_kepler_reduced_family(
        datasets,
        power_exponents=np.linspace(1.75, 2.25, 21, dtype=np.float64),
    )
    stage_a = fit_areal_law(datasets)
    stage_b = fit_radial_family(datasets, exponent=2.0)

    omega_table = {
        "target": "omega",
        "rows": [
            {
                "dataset_id": row["orbit_id"],
                "coeffs": [row["ell_fit"]],
                "intercept": 0.0,
                "probe_rmse": row["rmse"],
                "extraction_mode": "analytic_fit",
            }
            for row in stage_a["per_dataset"]
        ],
    }
    rddot_table = {
        "target": "rddot",
        "rows": [
            {
                "dataset_id": row["orbit_id"],
                "coeffs": [row["k_fit"], -float(stage_b["mu"])],
                "intercept": 0.0,
                "probe_rmse": row["rmse"],
                "extraction_mode": "analytic_fit",
            }
            for row in stage_b["per_dataset"]
        ],
    }
    datasets_by_id = {dataset.orbit_id: dataset for dataset in datasets}
    merged_rows = merge_symbolic_kepler_tables(
        omega_table,
        rddot_table,
        datasets_by_id=datasets_by_id,
    )
    symbolic_summary = {
        "status": "extractable",
        "omega": omega_table,
        "rddot": rddot_table,
        "merged_rows": merged_rows,
        "symbolic_summary": analyze_symbolic_kepler_family(merged_rows, datasets=datasets),
        "max_intercept": 0.0,
        "max_probe_rmse": max(
            max(float(row["rmse"]) for row in stage_a["per_dataset"]),
            max(float(row["rmse"]) for row in stage_b["per_dataset"]),
        ),
    }

    direct_path = tmp_path / "direct_summary.json"
    symbolic_path = tmp_path / "symbolic_summary.json"
    output_dir = tmp_path / "figures"
    direct_path.write_text(json.dumps(_jsonable(direct_summary), indent=2), encoding="utf-8")
    symbolic_path.write_text(json.dumps(_jsonable(symbolic_summary), indent=2), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_paper_figures.py",
            "--symbolic_summary",
            str(symbolic_path),
            "--direct_summary",
            str(direct_path),
            "--data_dir",
            str(tmp_path / "data"),
            "--output_dir",
            str(output_dir),
        ],
    )
    assert kepler_figures.main() == 0

    for stem in (
        "areal_law_family",
        "radial_family_selection",
        "coefficient_manifold",
        "energy_invariant",
    ):
        assert (output_dir / f"{stem}.png").exists()
        assert (output_dir / f"{stem}.pdf").exists()


def test_symbolic_holdout_generalization_recovers_exact_holdout_family():
    datasets = build_default_kepler_datasets(
        seed=555,
        train_samples=128,
        validation_samples=128,
        holdout_samples=192,
        **_SYNTHETIC_KW,
    )
    train_datasets = [dataset for dataset in datasets if dataset.split != "holdout"]
    holdout_datasets = [dataset for dataset in datasets if dataset.split == "holdout"]

    stage_a = fit_areal_law(train_datasets)
    stage_b = fit_radial_family(train_datasets, exponent=2.0)
    omega_table = {
        "target": "omega",
        "rows": [
            {
                "dataset_id": row["orbit_id"],
                "coeffs": [row["ell_fit"]],
                "intercept": 0.0,
                "probe_rmse": row["rmse"],
                "extraction_mode": "analytic_fit",
            }
            for row in stage_a["per_dataset"]
        ],
    }
    rddot_table = {
        "target": "rddot",
        "rows": [
            {
                "dataset_id": row["orbit_id"],
                "coeffs": [row["k_fit"], -float(stage_b["mu"])],
                "intercept": 0.0,
                "probe_rmse": row["rmse"],
                "extraction_mode": "analytic_fit",
            }
            for row in stage_b["per_dataset"]
        ],
    }
    merged_rows = merge_symbolic_kepler_tables(
        omega_table,
        rddot_table,
        datasets_by_id={dataset.orbit_id: dataset for dataset in train_datasets},
    )
    symbolic_summary = analyze_symbolic_kepler_family(merged_rows, datasets=train_datasets)
    full_summary = {
        "status": "extractable",
        "merged_rows": merged_rows,
        "symbolic_summary": symbolic_summary,
    }

    holdout = evaluate_symbolic_holdout_generalization(
        full_summary,
        holdout_datasets=holdout_datasets,
    )

    assert holdout["aggregate"]["max_ell_rel_error"] < 1.0e-10
    assert holdout["aggregate"]["max_k_abs_error"] < 1.0e-10
    assert holdout["aggregate"]["max_radial_rmse"] < 1.0e-10
    assert holdout["aggregate"]["max_lift_penalty_vs_oracle_refit"] < 1.0e-10


def test_probe_stability_canonicalizes_prefixed_dataset_ids(monkeypatch):
    datasets = build_default_kepler_datasets(
        seed=777,
        train_samples=32,
        validation_samples=32,
        holdout_samples=32,
        **_SYNTHETIC_KW,
    )[:2]

    def _fake_extract(**kwargs):
        _ = kwargs
        rows = []
        for dataset in datasets:
            rows.append(
                {
                    "dataset_id": f"omega_{dataset.orbit_id}",
                    "coeffs": np.asarray([float(dataset.h)], dtype=np.float64),
                    "intercept": 0.0,
                    "probe_rmse": 0.0,
                }
            )
        return {"rows": rows}

    monkeypatch.setattr(
        _real_utils._BASE,
        "extract_classsr_inverse_power_rows",
        _fake_extract,
    )

    summary = analyze_classsr_probe_stability(
        class_sr_json_path="dummy.json",
        stageb_pkl_path="dummy.pkl",
        datasets=datasets,
        exponents=[2.0],
        n_clouds=3,
        n_points=5,
        seed=123,
    )

    assert len(summary["per_dataset"]) == len(datasets)
    assert summary["aggregate"]["max_coeff_rel_std_by_exponent"]["r^-2"] == 0.0


def test_leverage_round_robin_split_matches_published_rule():
    import dataclasses

    from kepler_demo_utils import (
        LEVERAGE_ROUND_ROBIN_STRATEGY,
        assign_leverage_round_robin_splits,
    )

    template = None
    datasets = []
    rng = np.random.default_rng(7)
    leverages = rng.permutation(np.linspace(1.05, 3.0, 20))
    for index, leverage in enumerate(leverages):
        t = np.linspace(0.0, 10.0, 16)
        zeros = np.zeros_like(t)
        ds = _real_utils.KeplerReducedDataset(
            orbit_id=f"body{index:02d}",
            split="candidate",
            mu=1.0, a=1.0, e=0.1, period=1.0, mean_motion=1.0, h=1.0,
            energy=-1.0, dynamic_range=float(leverage),
            t=t, mean_anomaly=zeros, eccentric_anomaly=zeros,
            x=zeros, y=zeros, vx=zeros, vy=zeros, ax=zeros, ay=zeros,
            r=zeros + 1.0, theta=zeros, rdot=zeros, omega=zeros, rddot=zeros,
        )
        datasets.append(ds)
        template = template or ds

    assigned = assign_leverage_round_robin_splits(datasets)
    by_split = {}
    for ds in assigned:
        by_split.setdefault(ds.split, []).append(ds)
    assert sorted(by_split) == ["holdout", "train", "validation"]
    assert len(by_split["holdout"]) == 2 and len(by_split["validation"]) == 2
    assert len(by_split["train"]) == 16

    # the rule: ascending leverage, index 0 and 10 -> holdout, 1 and 11 -> validation
    ordered = sorted(assigned, key=lambda ds: (float(ds.dynamic_range), ds.orbit_id))
    for index, ds in enumerate(ordered):
        expected = "holdout" if index % 10 == 0 else "validation" if index % 10 == 1 else "train"
        assert ds.split == expected
    assert "round-robin modulo 10" in LEVERAGE_ROUND_ROBIN_STRATEGY

    # matches the figure script's canonical reconstruction exactly
    import make_direct_paper_figures as figure_mod

    reconstructed = figure_mod._round_robin_direct_holdout_splits(datasets)
    assert [ds.orbit_id for ds in sorted(by_split["holdout"], key=lambda d: d.orbit_id)] == sorted(
        ds.orbit_id for ds in reconstructed["holdout"]
    )
