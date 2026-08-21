# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "special_relativity"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

_SR_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "special_relativity_run_class_sr_discovery",
    _EXAMPLE_DIR / "run_class_sr_discovery.py",
)
assert _SR_RUNNER_SPEC is not None and _SR_RUNNER_SPEC.loader is not None
sr_runner = importlib.util.module_from_spec(_SR_RUNNER_SPEC)
_SR_RUNNER_SPEC.loader.exec_module(sr_runner)
from sr_demo_utils import (
    analyze_operational_boost_family,
    extract_affine_coefficients_from_ast,
    extract_affine_coefficients_from_model,
    generate_operational_interval_dataset,
)
from nestynet_sr.sr_core.bridges import AddNode, MulNode, Scale, Var


def test_special_relativity_affine_lift_and_metric_recovery_noiseless():
    betas = (-0.8, -0.55, -0.25, 0.25, 0.55, 0.8)
    datasets = [
        generate_operational_interval_dataset(beta, n_samples=1024, seed=120 + idx)
        for idx, beta in enumerate(betas)
    ]

    summary = analyze_operational_boost_family(datasets)
    coeff = summary["coefficient_laws"]
    metric = summary["metric"]

    assert coeff["max_beta_residual"] < 1.0e-10
    assert coeff["max_z_residual"] < 1.0e-10
    assert coeff["gamma_max_abs_error"] < 1.0e-10
    assert coeff["symmetry_max_abs_error"] < 1.0e-10

    recovered_metric = metric["metric"]
    assert metric["is_indefinite"] is True
    assert metric["max_preservation_error"] < 1.0e-10
    assert np.allclose(recovered_metric, np.asarray([[1.0, 0.0], [0.0, -1.0]]), atol=1.0e-10)


def test_special_relativity_affine_lift_remains_stable_with_small_noise():
    betas = (-0.75, -0.45, -0.15, 0.15, 0.45, 0.75)
    datasets = [
        generate_operational_interval_dataset(
            beta,
            n_samples=2048,
            seed=250 + idx,
            noise_std=0.01,
            near_null_width=0.02,
        )
        for idx, beta in enumerate(betas)
    ]

    summary = analyze_operational_boost_family(datasets)
    coeff = summary["coefficient_laws"]
    metric = summary["metric"]

    assert coeff["max_beta_residual"] < 3.0e-3
    assert coeff["max_z_residual"] < 4.0e-3
    assert coeff["gamma_max_abs_error"] < 3.0e-3
    assert coeff["symmetry_max_abs_error"] < 3.0e-3

    recovered_metric = metric["metric"]
    assert metric["is_indefinite"] is True
    assert metric["max_preservation_error"] < 2.0e-2
    assert abs(float(recovered_metric[0, 0]) - 1.0) < 2.0e-2
    assert abs(float(recovered_metric[0, 1])) < 2.0e-2
    assert abs(float(recovered_metric[1, 1]) + 1.0) < 2.0e-2


def test_extract_affine_coefficients_from_ast_recovers_linear_map():
    root = AddNode(
        MulNode(Scale("a", tag="a", init=1.0), Var(0)),
        MulNode(Scale("b", tag="b", init=1.0), Var(1)),
    )

    result = extract_affine_coefficients_from_ast(
        root,
        param_values={"a": 1.25, "b": -0.75},
        num_vars=2,
    )

    assert abs(float(result["intercept"])) < 1.0e-12
    assert np.allclose(result["coeffs"], np.asarray([1.25, -0.75]), atol=1.0e-12)
    assert float(result["probe_rmse"]) < 1.0e-12


def test_extract_affine_coefficients_from_model_recovers_linear_map():
    class LinearMap(torch.nn.Module):
        def forward(self, x):
            return 1.25 * x[:, :1] - 0.75 * x[:, 1:2]

    result = extract_affine_coefficients_from_model(
        LinearMap(),
        num_vars=2,
    )

    assert abs(float(result["intercept"])) < 1.0e-12
    assert np.allclose(result["coeffs"], np.asarray([1.25, -0.75]), atol=1.0e-12)
    assert float(result["probe_rmse"]) < 1.0e-12


def test_run_class_sr_discovery_uses_multi_dataset_result_stem():
    filepaths = [
        Path("examples/special_relativity/data/uprime/uprime_beta_m0600.csv"),
        Path("examples/special_relativity/data/uprime/uprime_beta_p0000.csv"),
        Path("examples/special_relativity/data/uprime/uprime_beta_p0600.csv"),
    ]

    stem = sr_runner._derive_result_base_filename(filepaths)

    assert stem == "uprime_beta_multi3"


def test_run_class_sr_discovery_prefers_manifest_beta_mapping(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = {
        "regimes": [
            {"regime_id": "beta_m0600", "beta": -0.6},
            {"regime_id": "beta_p0000", "beta": 0.0},
            {"regime_id": "beta_p0600", "beta": 0.6},
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    resolved = sr_runner._resolve_beta_by_dataset(
        data_dir=data_dir,
        fallback_betas=[-0.8, -0.6, -0.3, 0.3, 0.6, 0.8],
    )

    assert resolved == {
        "beta_m0600": -0.6,
        "beta_p0000": 0.0,
        "beta_p0600": 0.6,
    }


def test_run_class_sr_discovery_uses_actual_dataset_size_for_defaults(tmp_path: Path):
    target_dir = tmp_path / "uprime"
    target_dir.mkdir()
    csv_path = target_dir / "uprime_beta_p0600.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
            [
                np.linspace(0.0, 1.0, 128),
                np.linspace(1.0, 2.0, 128),
                np.linspace(2.0, 3.0, 128),
            ]
        ),
        delimiter=",",
        header="y,x0,x1",
        comments="",
    )

    n_effective = sr_runner._infer_effective_n_samples([csv_path])
    n_train, n_val, batch_size, class_points = sr_runner._resolve_run_dimensions(
        actual_n_samples=n_effective,
        ndata_train=None,
        ndata_val=None,
        batch_size=None,
        class_sr_max_points=None,
    )

    assert n_effective == 128
    assert (n_train, n_val, batch_size, class_points) == (64, 64, 64, 64)


def test_run_class_sr_discovery_target_filepaths_follow_manifest(tmp_path: Path):
    data_dir = tmp_path / "data"
    uprime_dir = data_dir / "uprime"
    uprime_dir.mkdir(parents=True)

    keep_a = uprime_dir / "uprime_beta_m0600.csv"
    keep_b = uprime_dir / "uprime_beta_p0600.csv"
    stale = uprime_dir / "uprime_beta_p0000.csv"
    for path in (keep_a, keep_b, stale):
        np.savetxt(
            path,
            np.column_stack(
                [
                    np.linspace(0.0, 1.0, 8),
                    np.linspace(1.0, 2.0, 8),
                    np.linspace(2.0, 3.0, 8),
                ]
            ),
            delimiter=",",
            header="y,x0,x1",
            comments="",
        )

    manifest = {
        "regimes": [
            {"regime_id": "beta_m0600", "uprime_csv": str(keep_a)},
            {"regime_id": "beta_p0600", "uprime_csv": str(keep_b)},
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    filepaths = sr_runner._target_filepaths(data_dir, "uprime")

    assert filepaths == [keep_a, keep_b]
