# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AddNode,
    ArgNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    PowNode,
    RealNode,
    SinNode,
    ast_from_composite,
    build_composite_from_ast,
    collect_all_atoms,
    make_reuse_only_nn_factory,
    make_stage_a_nn_factory,
    sync_ast_num_segments_from_state_dict,
)
from nestynet_sr.sr_search.config import ModelHyperparams
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast
from nestynet_sr.sr_search.xcoord import XCoordSystem


@dataclass(frozen=True)
class OrbitSpec:
    orbit_id: str
    split: str
    a: float
    e: float
    n_samples: int
    mean_anomaly0: float = 0.0


@dataclass(frozen=True)
class KeplerReducedDataset:
    orbit_id: str
    split: str
    mu: float
    a: float
    e: float
    period: float
    mean_motion: float
    h: float
    energy: float
    dynamic_range: float
    t: np.ndarray
    mean_anomaly: np.ndarray
    eccentric_anomaly: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    rdot: np.ndarray
    omega: np.ndarray
    rddot: np.ndarray
    # How ax/ay were obtained (surrogate diagnostics + certificate); None for
    # the finite-difference path and the clean two-body propagation.
    accel_provenance: dict[str, Any] | None = None


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _split_sort_key(name: str) -> tuple[int, str]:
    order = {"train": 0, "validation": 1, "holdout": 2}
    return int(order.get(str(name), 99)), str(name)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def _solve_kepler_equation(mean_anomaly: np.ndarray, e: float) -> np.ndarray:
    m = np.asarray(mean_anomaly, dtype=np.float64)
    two_pi = 2.0 * math.pi
    wrapped = np.mod(m, two_pi)
    if float(e) < 0.8:
        ecc = wrapped.copy()
    else:
        ecc = np.full_like(wrapped, math.pi)

    for _ in range(40):
        f = ecc - float(e) * np.sin(ecc) - wrapped
        fp = 1.0 - float(e) * np.cos(ecc)
        delta = f / fp
        ecc = ecc - delta
        if float(np.max(np.abs(delta))) < 1.0e-13:
            break

    out = ecc + (m - wrapped)
    return np.asarray(out, dtype=np.float64)


def build_default_orbit_specs(
    *,
    seed: int = 123,
    train_samples: int = 1024,
    validation_samples: int = 1024,
    holdout_samples: int = 2048,
) -> list[OrbitSpec]:
    rng = np.random.default_rng(int(seed))
    specs: list[OrbitSpec] = []
    split_defs = (
        ("train", np.linspace(0.02, 0.25, 8), int(train_samples)),
        ("validation", np.linspace(0.35, 0.50, 2), int(validation_samples)),
        ("holdout", np.linspace(0.70, 0.85, 2), int(holdout_samples)),
    )
    for split, eccentricities, n_samples in split_defs:
        a_values = rng.uniform(0.8, 2.5, size=len(eccentricities))
        phases = rng.uniform(0.0, 2.0 * math.pi, size=len(eccentricities))
        for idx, (ecc, a_value, phase) in enumerate(zip(eccentricities, a_values, phases), start=1):
            specs.append(
                OrbitSpec(
                    orbit_id=f"orbit_{split}_{idx:02d}",
                    split=str(split),
                    a=float(a_value),
                    e=float(ecc),
                    n_samples=int(n_samples),
                    mean_anomaly0=float(phase),
                )
            )
    specs.sort(key=lambda spec: (_split_sort_key(spec.split), spec.orbit_id))
    return specs


def generate_kepler_dataset(spec: OrbitSpec, *, mu: float = 1.0) -> KeplerReducedDataset:
    mu_f = float(mu)
    if mu_f <= 0.0:
        raise ValueError("mu must be positive")
    if float(spec.a) <= 0.0:
        raise ValueError("semi-major axis must be positive")
    if not (0.0 <= float(spec.e) < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if int(spec.n_samples) < 8:
        raise ValueError("need at least 8 samples per orbit")

    a = float(spec.a)
    e = float(spec.e)
    n_samples = int(spec.n_samples)
    mean_motion = math.sqrt(mu_f / (a ** 3))
    period = 2.0 * math.pi / mean_motion
    t = np.linspace(0.0, period, n_samples, endpoint=False, dtype=np.float64)
    mean_anomaly = float(spec.mean_anomaly0) + mean_motion * t
    eccentric_anomaly = _solve_kepler_equation(mean_anomaly, e)

    cos_e = np.cos(eccentric_anomaly)
    sin_e = np.sin(eccentric_anomaly)
    sqrt_one_minus_e2 = math.sqrt(max(1.0 - e * e, 0.0))
    radius = a * (1.0 - e * cos_e)

    x = a * (cos_e - e)
    y = a * sqrt_one_minus_e2 * sin_e
    theta = np.unwrap(np.arctan2(y, x))

    edot = mean_motion / (1.0 - e * cos_e)
    vx = -a * sin_e * edot
    vy = a * sqrt_one_minus_e2 * cos_e * edot

    h = math.sqrt(mu_f * a * (1.0 - e * e))
    omega = h / np.square(radius)
    rdot = a * e * sin_e * edot
    rddot = (h * h) / np.power(radius, 3) - mu_f / np.square(radius)

    ax = -mu_f * x / np.power(radius, 3)
    ay = -mu_f * y / np.power(radius, 3)
    energy = -mu_f / (2.0 * a)
    dynamic_range = (1.0 + e) / max(1.0 - e, 1.0e-12)

    return KeplerReducedDataset(
        orbit_id=str(spec.orbit_id),
        split=str(spec.split),
        mu=mu_f,
        a=a,
        e=e,
        period=float(period),
        mean_motion=float(mean_motion),
        h=float(h),
        energy=float(energy),
        dynamic_range=float(dynamic_range),
        t=np.asarray(t, dtype=np.float64),
        mean_anomaly=np.asarray(mean_anomaly, dtype=np.float64),
        eccentric_anomaly=np.asarray(eccentric_anomaly, dtype=np.float64),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        vx=np.asarray(vx, dtype=np.float64),
        vy=np.asarray(vy, dtype=np.float64),
        ax=np.asarray(ax, dtype=np.float64),
        ay=np.asarray(ay, dtype=np.float64),
        r=np.asarray(radius, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        rdot=np.asarray(rdot, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        rddot=np.asarray(rddot, dtype=np.float64),
    )


def build_default_kepler_datasets(
    *,
    mu: float = 1.0,
    seed: int = 123,
    train_samples: int = 1024,
    validation_samples: int = 1024,
    holdout_samples: int = 2048,
) -> list[KeplerReducedDataset]:
    specs = build_default_orbit_specs(
        seed=int(seed),
        train_samples=int(train_samples),
        validation_samples=int(validation_samples),
        holdout_samples=int(holdout_samples),
    )
    return [generate_kepler_dataset(spec, mu=float(mu)) for spec in specs]


LEVERAGE_ROUND_ROBIN_STRATEGY = (
    "sorted-by-dynamic-range round-robin modulo 10: "
    "holdout=i%10==0, validation=i%10==1, train=otherwise"
)


def assign_leverage_round_robin_splits(
    datasets: Sequence[KeplerReducedDataset],
) -> list[KeplerReducedDataset]:
    """Deterministic radial-leverage split used by the paper-4 Kepler showcase.

    Sort ascending by ``(dynamic_range, orbit_id)`` and assign round-robin
    modulo 10: index 0 -> holdout, 1 -> validation, else train.  Mirrors
    ``_round_robin_direct_holdout_splits`` in ``make_direct_paper_figures.py``
    (the recorded ``split_strategy`` of the published 308-body summary).
    """
    import dataclasses

    ordered = sorted(list(datasets), key=lambda ds: (float(ds.dynamic_range), ds.orbit_id))
    out = []
    for index, dataset in enumerate(ordered):
        if index % 10 == 0:
            split = "holdout"
        elif index % 10 == 1:
            split = "validation"
        else:
            split = "train"
        out.append(dataclasses.replace(dataset, split=split))
    out.sort(key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    return out


def split_datasets(
    datasets: Sequence[KeplerReducedDataset],
) -> dict[str, list[KeplerReducedDataset]]:
    out: dict[str, list[KeplerReducedDataset]] = {}
    for dataset in list(datasets):
        out.setdefault(str(dataset.split), []).append(dataset)
    for key in out:
        out[key] = sorted(out[key], key=lambda ds: ds.orbit_id)
    return out


def _write_xy_csv(path: Path, y: np.ndarray, x0: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([y, x0]),
        delimiter=",",
        header="y,x0",
        comments="",
    )


def _coverage_permutation(n: int, *, phase: float = 0.0) -> np.ndarray:
    if int(n) <= 0:
        raise ValueError("coverage permutation requires a positive length")
    idx = np.arange(int(n), dtype=np.int64)
    frac = np.mod(idx * ((math.sqrt(5.0) - 1.0) / 2.0) + float(phase), 1.0)
    return np.argsort(frac, kind="mergesort")


def write_generated_artifacts(output_root: str | Path, datasets: Sequence[KeplerReducedDataset]) -> dict[str, Any]:
    root = Path(output_root)
    data_dir = root / "data"
    omega_dir = data_dir / "omega"
    rddot_dir = data_dir / "rddot"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    split_to_index = {"train": 0.0, "validation": 1.0, "holdout": 2.0}

    for idx, dataset in enumerate(list(datasets)):
        combined_path = data_dir / f"orbits_{dataset.orbit_id}.csv"
        np.savetxt(
            combined_path,
            np.column_stack(
                [
                    dataset.t,
                    dataset.x,
                    dataset.y,
                    dataset.vx,
                    dataset.vy,
                    dataset.ax,
                    dataset.ay,
                    dataset.r,
                    dataset.theta,
                    dataset.rdot,
                    dataset.omega,
                    dataset.rddot,
                    dataset.mean_anomaly,
                    dataset.eccentric_anomaly,
                ]
            ),
            delimiter=",",
            header="t,x,y,vx,vy,ax,ay,r,theta,rdot,omega,rddot,mean_anomaly,eccentric_anomaly",
            comments="",
        )
        omega_path = omega_dir / f"omega_{dataset.orbit_id}.csv"
        rddot_path = rddot_dir / f"rddot_{dataset.orbit_id}.csv"
        omega_perm = _coverage_permutation(dataset.r.shape[0], phase=0.0)
        rddot_perm = _coverage_permutation(dataset.r.shape[0], phase=0.5)
        _write_xy_csv(omega_path, dataset.omega[omega_perm], dataset.r[omega_perm])
        _write_xy_csv(rddot_path, dataset.rddot[rddot_perm], dataset.r[rddot_perm])

        manifest_rows.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "mu": float(dataset.mu),
                "a": float(dataset.a),
                "e": float(dataset.e),
                "h": float(dataset.h),
                "energy": float(dataset.energy),
                "dynamic_range": float(dataset.dynamic_range),
                "period": float(dataset.period),
                "n_samples": int(dataset.t.shape[0]),
                "combined_csv": str(combined_path),
                "omega_csv": str(omega_path),
                "rddot_csv": str(rddot_path),
            }
        )
        if dataset.accel_provenance is not None:
            manifest_rows[-1]["accel_provenance"] = _jsonable(dataset.accel_provenance)
        metadata_rows.append(
            {
                "orbit_index": float(idx),
                "split_index": float(split_to_index.get(dataset.split, -1.0)),
            }
        )

    manifest = {
        "mu": float(datasets[0].mu) if datasets else None,
        "orbits": manifest_rows,
        "param_sr_metadata_rows": metadata_rows,
    }
    manifest_path = data_dir / "manifest.json"
    metadata_path = data_dir / "param_sr_metadata_rows.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(_jsonable(metadata_rows), indent=2), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "n_orbits": len(manifest_rows),
    }


def fit_areal_law(datasets: Sequence[KeplerReducedDataset]) -> dict[str, Any]:
    per_dataset: list[dict[str, Any]] = []
    ell_by_dataset: dict[str, float] = {}
    max_abs_error = 0.0
    max_rel_error = 0.0

    for dataset in list(datasets):
        feature = 1.0 / np.square(dataset.r)
        target = dataset.omega
        ell = float(np.dot(feature, target) / np.dot(feature, feature))
        pred = ell * feature
        rmse = _rmse(target, pred)
        abs_error = abs(ell - float(dataset.h))
        rel_error = abs_error / max(abs(float(dataset.h)), 1.0e-12)
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        ell_by_dataset[dataset.orbit_id] = ell
        per_dataset.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "ell_fit": ell,
                "h_true": float(dataset.h),
                "rmse": rmse,
                "max_abs_residual": float(np.max(np.abs(target - pred))),
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
        )

    return {
        "dataset_ids": [row["orbit_id"] for row in per_dataset],
        "ell_by_dataset": ell_by_dataset,
        "per_dataset": per_dataset,
        "max_abs_error": float(max_abs_error),
        "max_rel_error": float(max_rel_error),
    }


def fit_radial_family(
    datasets: Sequence[KeplerReducedDataset],
    *,
    exponent: float = 2.0,
) -> dict[str, Any]:
    dataset_list = list(datasets)
    if not dataset_list:
        raise ValueError("fit_radial_family requires at least one dataset")

    rows = []
    targets = []
    n_datasets = len(dataset_list)
    for d_idx, dataset in enumerate(dataset_list):
        inv_r3 = 1.0 / np.power(dataset.r, 3)
        inv_rp = 1.0 / np.power(dataset.r, float(exponent))
        block = np.zeros((dataset.r.shape[0], n_datasets + 1), dtype=np.float64)
        block[:, d_idx] = inv_r3
        block[:, -1] = -inv_rp
        rows.append(block)
        targets.append(np.asarray(dataset.rddot, dtype=np.float64))

    design = np.vstack(rows)
    y = np.concatenate(targets)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    k_coeffs = coeffs[:-1]
    mu_fit = float(coeffs[-1])
    per_dataset: list[dict[str, Any]] = []
    k_by_dataset: dict[str, float] = {}
    rmse_values = []
    max_k_abs_error = 0.0
    for d_idx, dataset in enumerate(dataset_list):
        k_fit = float(k_coeffs[d_idx])
        pred = k_fit / np.power(dataset.r, 3) - mu_fit / np.power(dataset.r, float(exponent))
        rmse = _rmse(dataset.rddot, pred)
        rmse_values.append(rmse)
        k_true = float(dataset.h * dataset.h)
        abs_error = abs(k_fit - k_true)
        max_k_abs_error = max(max_k_abs_error, abs_error)
        k_by_dataset[dataset.orbit_id] = k_fit
        per_dataset.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "k_fit": k_fit,
                "k_true": k_true,
                "rmse": rmse,
                "max_abs_residual": float(np.max(np.abs(dataset.rddot - pred))),
                "abs_error": abs_error,
            }
        )

    return {
        "exponent": float(exponent),
        "mu": mu_fit,
        "mu_true": float(dataset_list[0].mu),
        "dataset_ids": [row["orbit_id"] for row in per_dataset],
        "k_by_dataset": k_by_dataset,
        "per_dataset": per_dataset,
        "mean_rmse": _mean(rmse_values),
        "max_rmse": float(max(rmse_values) if rmse_values else float("nan")),
        "max_k_abs_error": float(max_k_abs_error),
        "mu_abs_error": float(abs(mu_fit - float(dataset_list[0].mu))),
    }


def evaluate_radial_family_with_fixed_mu(
    datasets: Sequence[KeplerReducedDataset],
    *,
    mu: float,
    exponent: float,
) -> dict[str, Any]:
    per_dataset: list[dict[str, Any]] = []
    k_by_dataset: dict[str, float] = {}
    rmse_values = []
    for dataset in list(datasets):
        inv_r3 = 1.0 / np.power(dataset.r, 3)
        rhs = dataset.rddot + float(mu) / np.power(dataset.r, float(exponent))
        k_fit = float(np.dot(inv_r3, rhs) / np.dot(inv_r3, inv_r3))
        pred = k_fit / np.power(dataset.r, 3) - float(mu) / np.power(dataset.r, float(exponent))
        rmse = _rmse(dataset.rddot, pred)
        rmse_values.append(rmse)
        k_by_dataset[dataset.orbit_id] = k_fit
        per_dataset.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "k_fit": k_fit,
                "rmse": rmse,
                "max_abs_residual": float(np.max(np.abs(dataset.rddot - pred))),
            }
        )
    return {
        "mu": float(mu),
        "exponent": float(exponent),
        "k_by_dataset": k_by_dataset,
        "per_dataset": per_dataset,
        "mean_rmse": _mean(rmse_values),
        "max_rmse": float(max(rmse_values) if rmse_values else float("nan")),
    }


def scan_inverse_power_family(
    *,
    train_datasets: Sequence[KeplerReducedDataset],
    validation_datasets: Sequence[KeplerReducedDataset],
    holdout_datasets: Sequence[KeplerReducedDataset],
    exponents: Iterable[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exponent_values = [float(v) for v in exponents]
    if not exponent_values:
        raise ValueError("scan_inverse_power_family requires at least one exponent")

    for exponent in exponent_values:
        fit = fit_radial_family(train_datasets, exponent=float(exponent))
        val_eval = evaluate_radial_family_with_fixed_mu(
            validation_datasets,
            mu=float(fit["mu"]),
            exponent=float(exponent),
        )
        holdout_eval = evaluate_radial_family_with_fixed_mu(
            holdout_datasets,
            mu=float(fit["mu"]),
            exponent=float(exponent),
        )
        rows.append(
            {
                "exponent": float(exponent),
                "mu_fit": float(fit["mu"]),
                "train_mean_rmse": float(fit["mean_rmse"]),
                "validation_mean_rmse": float(val_eval["mean_rmse"]),
                "holdout_mean_rmse": float(holdout_eval["mean_rmse"]),
                "train_max_rmse": float(fit["max_rmse"]),
                "validation_max_rmse": float(val_eval["max_rmse"]),
                "holdout_max_rmse": float(holdout_eval["max_rmse"]),
            }
        )

    best_train = min(rows, key=lambda row: (row["train_mean_rmse"], abs(row["exponent"] - 2.0)))
    best_holdout = min(rows, key=lambda row: (row["holdout_mean_rmse"], abs(row["exponent"] - 2.0)))
    exact_row = min(rows, key=lambda row: abs(row["exponent"] - 2.0))
    sorted_holdout = sorted(rows, key=lambda row: row["holdout_mean_rmse"])
    holdout_margin = float("nan")
    if len(sorted_holdout) >= 2:
        holdout_margin = float(sorted_holdout[1]["holdout_mean_rmse"] - sorted_holdout[0]["holdout_mean_rmse"])

    return {
        "rows": rows,
        "best_train_exponent": float(best_train["exponent"]),
        "best_holdout_exponent": float(best_holdout["exponent"]),
        "exact_exponent_row": exact_row,
        "holdout_margin_to_second": holdout_margin,
    }


def lift_coefficient_relation(
    *,
    datasets: Sequence[KeplerReducedDataset],
    ell_by_dataset: Dict[str, float],
    k_by_dataset: Dict[str, float],
) -> dict[str, Any]:
    dataset_list = sorted(list(datasets), key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    ell = np.asarray([float(ell_by_dataset[dataset.orbit_id]) for dataset in dataset_list], dtype=np.float64)
    k = np.asarray([float(k_by_dataset[dataset.orbit_id]) for dataset in dataset_list], dtype=np.float64)
    ell_sq = np.square(ell)

    design_sq = np.column_stack([np.ones_like(ell_sq), ell_sq])
    coeff_sq, _, _, _ = np.linalg.lstsq(design_sq, k, rcond=None)
    pred_sq = design_sq @ coeff_sq

    design_lin = np.column_stack([np.ones_like(ell), ell])
    coeff_lin, _, _, _ = np.linalg.lstsq(design_lin, k, rcond=None)
    pred_lin = design_lin @ coeff_lin

    per_dataset = []
    for dataset, ell_val, k_val, pred_val in zip(dataset_list, ell, k, pred_sq):
        per_dataset.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "ell": float(ell_val),
                "ell_sq": float(ell_val * ell_val),
                "k": float(k_val),
                "sq_model_residual": float(k_val - pred_val),
            }
        )

    return {
        "intercept": float(coeff_sq[0]),
        "slope": float(coeff_sq[1]),
        "quadratic_rmse": _rmse(k, pred_sq),
        "linear_rmse": _rmse(k, pred_lin),
        "max_abs_residual": float(np.max(np.abs(k - pred_sq))),
        "dataset_ids": [row["orbit_id"] for row in per_dataset],
        "per_dataset": per_dataset,
    }


def recover_energy_integral(
    *,
    datasets: Sequence[KeplerReducedDataset],
    ell_by_dataset: Dict[str, float],
    mu: float,
) -> dict[str, Any]:
    centered_rows = []
    basis_map: dict[str, np.ndarray] = {}
    for dataset in list(datasets):
        ell = float(ell_by_dataset[dataset.orbit_id])
        basis = np.column_stack(
            [
                np.square(dataset.rdot),
                (ell * ell) / np.square(dataset.r),
                float(mu) / dataset.r,
            ]
        ).astype(np.float64, copy=False)
        basis_map[dataset.orbit_id] = basis
        centered_rows.append(basis - np.mean(basis, axis=0, keepdims=True))

    stacked = np.vstack(centered_rows)
    _, singular_values, vh = np.linalg.svd(stacked, full_matrices=False)
    coeffs = np.asarray(vh[-1], dtype=np.float64)
    if float(coeffs[0]) < 0.0:
        coeffs = -coeffs
    if abs(float(coeffs[0])) <= 1.0e-14:
        raise ValueError("failed to recover an energy coefficient with nonzero v^2 weight")
    coeffs = coeffs * (0.5 / float(coeffs[0]))

    expected = np.asarray([0.5, 0.5, -1.0], dtype=np.float64)
    per_dataset = []
    residuals = []
    energy_by_dataset: dict[str, float] = {}
    for dataset in list(datasets):
        basis = basis_map[dataset.orbit_id]
        energy_series = (
            float(coeffs[0]) * basis[:, 0]
            + float(coeffs[1]) * basis[:, 1]
            + float(coeffs[2]) * basis[:, 2]
        )
        energy_level = float(np.mean(energy_series))
        residual = energy_series - energy_level
        residuals.append(float(np.max(np.abs(residual))))
        energy_by_dataset[dataset.orbit_id] = energy_level
        per_dataset.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "energy_fit": energy_level,
                "energy_true": float(dataset.energy),
                "energy_abs_error": float(abs(energy_level - float(dataset.energy))),
                "max_centered_residual": float(np.max(np.abs(residual))),
            }
        )

    return {
        "coeffs": {
            "rdot_sq": float(coeffs[0]),
            "ell_sq_over_r_sq": float(coeffs[1]),
            "mu_over_r": float(coeffs[2]),
        },
        "expected_coeffs": {
            "rdot_sq": 0.5,
            "ell_sq_over_r_sq": 0.5,
            "mu_over_r": -1.0,
        },
        "coeff_max_abs_error": float(np.max(np.abs(coeffs - expected))),
        "energy_by_dataset": energy_by_dataset,
        "per_dataset": per_dataset,
        "max_centered_residual": float(max(residuals) if residuals else 0.0),
        "singular_values": np.asarray(singular_values, dtype=np.float64),
    }


def assemble_kepler_hamiltonian(
    *,
    datasets: Sequence[KeplerReducedDataset],
    ell_by_dataset: Dict[str, float],
    mu: float,
    coefficient_lift: dict[str, Any],
    energy: dict[str, Any],
) -> dict[str, Any]:
    dataset_list = sorted(list(datasets), key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    if not dataset_list:
        raise ValueError("assemble_kepler_hamiltonian requires at least one dataset")

    mu_f = float(mu)
    coeffs = dict(energy["coeffs"])
    recovered_pr_sq = float(coeffs["rdot_sq"])
    recovered_ptheta_sq_over_r_sq = float(coeffs["ell_sq_over_r_sq"])
    recovered_mu_over_r = float(coeffs["mu_over_r"])

    per_dataset = []
    theta_rmse = []
    radial_rmse = []
    ptheta_state_abs_error = []
    natural_reduced_energy_abs_error = []
    natural_cartesian_energy_abs_error = []
    reduced_cartesian_gap = []
    natural_reduced_centered_residual = []
    natural_cartesian_centered_residual = []

    for dataset in dataset_list:
        ell = float(ell_by_dataset[dataset.orbit_id])
        ptheta_series = dataset.x * dataset.vy - dataset.y * dataset.vx

        recovered_reduced_series = (
            recovered_pr_sq * np.square(dataset.rdot)
            + recovered_ptheta_sq_over_r_sq * ((ell * ell) / np.square(dataset.r))
            + recovered_mu_over_r * (mu_f / dataset.r)
        )
        recovered_reduced_level = float(np.mean(recovered_reduced_series))

        natural_reduced_series = (
            0.5 * np.square(dataset.rdot)
            + 0.5 * ((ell * ell) / np.square(dataset.r))
            - mu_f / dataset.r
        )
        natural_reduced_level = float(np.mean(natural_reduced_series))

        natural_cartesian_series = 0.5 * (np.square(dataset.vx) + np.square(dataset.vy)) - mu_f / dataset.r
        natural_cartesian_level = float(np.mean(natural_cartesian_series))

        theta_pred = ell / np.square(dataset.r)
        radial_pred = (ell * ell) / np.power(dataset.r, 3) - mu_f / np.square(dataset.r)

        theta_fit_rmse = _rmse(dataset.omega, theta_pred)
        radial_fit_rmse = _rmse(dataset.rddot, radial_pred)
        ptheta_err = float(np.max(np.abs(ptheta_series - ell)))
        reduced_energy_err = float(abs(natural_reduced_level - float(dataset.energy)))
        cartesian_energy_err = float(abs(natural_cartesian_level - float(dataset.energy)))
        level_gap = float(abs(natural_reduced_level - natural_cartesian_level))
        reduced_centered = natural_reduced_series - natural_reduced_level
        cartesian_centered = natural_cartesian_series - natural_cartesian_level

        theta_rmse.append(theta_fit_rmse)
        radial_rmse.append(radial_fit_rmse)
        ptheta_state_abs_error.append(ptheta_err)
        natural_reduced_energy_abs_error.append(reduced_energy_err)
        natural_cartesian_energy_abs_error.append(cartesian_energy_err)
        reduced_cartesian_gap.append(level_gap)
        natural_reduced_centered_residual.append(float(np.max(np.abs(reduced_centered))))
        natural_cartesian_centered_residual.append(float(np.max(np.abs(cartesian_centered))))

        per_dataset.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "p_theta_fit": ell,
                "p_theta_state_mean": float(np.mean(ptheta_series)),
                "p_theta_state_std": float(np.std(ptheta_series)),
                "p_theta_state_max_abs_error": ptheta_err,
                "theta_rmse": theta_fit_rmse,
                "radial_rmse": radial_fit_rmse,
                "recovered_reduced_energy": recovered_reduced_level,
                "natural_reduced_energy": natural_reduced_level,
                "natural_cartesian_energy": natural_cartesian_level,
                "true_energy": float(dataset.energy),
                "natural_reduced_energy_abs_error": reduced_energy_err,
                "natural_cartesian_energy_abs_error": cartesian_energy_err,
                "reduced_vs_cartesian_energy_gap": level_gap,
                "natural_reduced_max_centered_residual": float(np.max(np.abs(reduced_centered))),
                "natural_cartesian_max_centered_residual": float(np.max(np.abs(cartesian_centered))),
            }
        )

    reduced_formula_plain = (
        f"H(r, theta, p_r, p_theta) = "
        f"{recovered_pr_sq:.9g} * p_r^2 + "
        f"{recovered_ptheta_sq_over_r_sq:.9g} * p_theta^2 / r^2 "
        f"{recovered_mu_over_r:+.9g} * mu / r"
    )
    natural_formula_plain = "H(r, theta, p_r, p_theta) = 0.5 * p_r^2 + 0.5 * p_theta^2 / r^2 - mu / r"
    cartesian_formula_plain = "H(x, y, p_x, p_y) = 0.5 * (p_x^2 + p_y^2) - mu / sqrt(x^2 + y^2)"

    return {
        "assumptions": {
            "unit_test_mass": True,
            "euclidean_configuration_space": True,
            "canonical_identification": {
                "p_r": "dot(r)",
                "p_theta": "ell_d",
            },
        },
        "shared_mu": mu_f,
        "recovered_reduced_coeffs": {
            "p_r_sq": recovered_pr_sq,
            "p_theta_sq_over_r_sq": recovered_ptheta_sq_over_r_sq,
            "mu_over_r": recovered_mu_over_r,
        },
        "natural_reduced_coeffs": {
            "p_r_sq": 0.5,
            "p_theta_sq_over_r_sq": 0.5,
            "mu_over_r": -1.0,
        },
        "recovered_formulas": {
            "reduced_plain": reduced_formula_plain,
            "reduced_latex": (
                r"H(r,\theta,p_r,p_\theta)="
                rf"{recovered_pr_sq:.9g}\,p_r^2+"
                rf"{recovered_ptheta_sq_over_r_sq:.9g}\,\frac{{p_\theta^2}}{{r^2}}"
                rf"{recovered_mu_over_r:+.9g}\,\frac{{\mu}}{{r}}"
            ),
            "natural_reduced_plain": natural_formula_plain,
            "natural_reduced_latex": (
                r"H(r,\theta,p_r,p_\theta)=\frac{1}{2}p_r^2+\frac{p_\theta^2}{2r^2}-\frac{\mu}{r}"
            ),
            "cartesian_plain": cartesian_formula_plain,
            "cartesian_latex": (
                r"H(x,y,p_x,p_y)=\frac{1}{2}(p_x^2+p_y^2)-\frac{\mu}{\sqrt{x^2+y^2}}"
            ),
        },
        "consistency": {
            "lift_intercept": float(coefficient_lift["intercept"]),
            "lift_slope": float(coefficient_lift["slope"]),
            "lift_max_abs_residual": float(coefficient_lift["max_abs_residual"]),
            "energy_coeff_max_abs_error": float(energy["coeff_max_abs_error"]),
            "max_theta_rmse": float(max(theta_rmse) if theta_rmse else 0.0),
            "max_radial_rmse": float(max(radial_rmse) if radial_rmse else 0.0),
            "max_p_theta_state_abs_error": float(max(ptheta_state_abs_error) if ptheta_state_abs_error else 0.0),
            "max_natural_reduced_energy_abs_error": float(
                max(natural_reduced_energy_abs_error) if natural_reduced_energy_abs_error else 0.0
            ),
            "max_natural_cartesian_energy_abs_error": float(
                max(natural_cartesian_energy_abs_error) if natural_cartesian_energy_abs_error else 0.0
            ),
            "max_reduced_vs_cartesian_energy_gap": float(max(reduced_cartesian_gap) if reduced_cartesian_gap else 0.0),
            "max_natural_reduced_centered_residual": float(
                max(natural_reduced_centered_residual) if natural_reduced_centered_residual else 0.0
            ),
            "max_natural_cartesian_centered_residual": float(
                max(natural_cartesian_centered_residual) if natural_cartesian_centered_residual else 0.0
            ),
        },
        "per_dataset": per_dataset,
    }


def analyze_kepler_reduced_family(
    datasets: Sequence[KeplerReducedDataset],
    *,
    power_exponents: Sequence[float] | None = None,
) -> dict[str, Any]:
    dataset_list = sorted(list(datasets), key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    if not dataset_list:
        raise ValueError("analyze_kepler_reduced_family requires at least one dataset")

    by_split = split_datasets(dataset_list)
    if power_exponents is None:
        power_exponents = np.linspace(1.6, 2.4, 81, dtype=np.float64)

    stage_a = fit_areal_law(dataset_list)
    stage_b_all = fit_radial_family(dataset_list, exponent=2.0)
    power_scan = scan_inverse_power_family(
        train_datasets=by_split.get("train", []),
        validation_datasets=by_split.get("validation", []),
        holdout_datasets=by_split.get("holdout", []),
        exponents=power_exponents,
    )
    coefficient_lift = lift_coefficient_relation(
        datasets=dataset_list,
        ell_by_dataset=stage_a["ell_by_dataset"],
        k_by_dataset=stage_b_all["k_by_dataset"],
    )
    energy = recover_energy_integral(
        datasets=dataset_list,
        ell_by_dataset=stage_a["ell_by_dataset"],
        mu=float(stage_b_all["mu"]),
    )
    hamiltonian = assemble_kepler_hamiltonian(
        datasets=dataset_list,
        ell_by_dataset=stage_a["ell_by_dataset"],
        mu=float(stage_b_all["mu"]),
        coefficient_lift=coefficient_lift,
        energy=energy,
    )

    orbit_registry = [
        {
            "orbit_id": dataset.orbit_id,
            "split": dataset.split,
            "a": float(dataset.a),
            "e": float(dataset.e),
            "h": float(dataset.h),
            "energy": float(dataset.energy),
            "dynamic_range": float(dataset.dynamic_range),
            "period": float(dataset.period),
            "n_samples": int(dataset.t.shape[0]),
        }
        for dataset in dataset_list
    ]

    return {
        "mu_true": float(dataset_list[0].mu),
        "orbit_registry": orbit_registry,
        "stage_a": stage_a,
        "stage_b_all": stage_b_all,
        "power_scan": power_scan,
        "coefficient_lift": coefficient_lift,
        "energy": energy,
        "hamiltonian": hamiltonian,
    }


def load_classsr_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stageb_payload(path: str | Path) -> dict[str, Any]:
    return pickle.loads(Path(path).read_bytes())


def load_kepler_manifest(data_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(data_dir) / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_kepler_datasets_from_manifest(data_dir: str | Path) -> list[KeplerReducedDataset]:
    data_root = Path(data_dir)
    manifest = load_kepler_manifest(data_root)
    rows = list(manifest.get("orbits", []) or [])
    datasets: list[KeplerReducedDataset] = []
    for row in rows:
        combined_path = Path(str(row["combined_csv"]))
        arr = np.genfromtxt(combined_path, delimiter=",", names=True, dtype=np.float64)
        if arr.shape == ():
            arr = np.asarray([tuple(arr.tolist())], dtype=arr.dtype)
        datasets.append(
            KeplerReducedDataset(
                orbit_id=str(row["orbit_id"]),
                split=str(row["split"]),
                mu=float(row["mu"]),
                a=float(row["a"]),
                e=float(row["e"]),
                period=float(row["period"]),
                mean_motion=float(2.0 * math.pi / float(row["period"])),
                h=float(row["h"]),
                energy=float(row["energy"]),
                dynamic_range=float(row["dynamic_range"]),
                t=np.asarray(arr["t"], dtype=np.float64),
                mean_anomaly=np.asarray(arr["mean_anomaly"], dtype=np.float64),
                eccentric_anomaly=np.asarray(arr["eccentric_anomaly"], dtype=np.float64),
                x=np.asarray(arr["x"], dtype=np.float64),
                y=np.asarray(arr["y"], dtype=np.float64),
                vx=np.asarray(arr["vx"], dtype=np.float64),
                vy=np.asarray(arr["vy"], dtype=np.float64),
                ax=np.asarray(arr["ax"], dtype=np.float64),
                ay=np.asarray(arr["ay"], dtype=np.float64),
                r=np.asarray(arr["r"], dtype=np.float64),
                theta=np.asarray(arr["theta"], dtype=np.float64),
                rdot=np.asarray(arr["rdot"], dtype=np.float64),
                omega=np.asarray(arr["omega"], dtype=np.float64),
                rddot=np.asarray(arr["rddot"], dtype=np.float64),
            )
        )
    datasets.sort(key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    return datasets


def target_filepaths(
    data_dir: str | Path,
    target: str,
    *,
    splits: Sequence[str] | None = None,
) -> list[Path]:
    target_name = str(target)
    data_root = Path(data_dir)
    target_dir = data_root / target_name
    if not target_dir.exists():
        raise FileNotFoundError(f"missing generated target directory: {target_dir}")
    split_filter = None if splits is None else {str(item) for item in splits}
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_paths = []
        key = f"{target_name}_csv"
        for row in list(manifest.get("orbits", []) or []):
            if not isinstance(row, dict):
                continue
            if split_filter is not None and str(row.get("split", "")) not in split_filter:
                continue
            csv_path = row.get(key, None)
            if csv_path is None:
                continue
            selected_paths.append(Path(str(csv_path)))
        if split_filter is not None or selected_paths:
            return selected_paths
    return sorted(target_dir.glob("*.csv"))


def infer_effective_n_samples(*file_groups: list[Path]) -> int:
    counts = []
    for group in file_groups:
        for path in group:
            with Path(path).open("r", encoding="utf-8") as handle:
                line_count = sum(1 for _ in handle)
            counts.append(max(0, int(line_count) - 1))
    if not counts:
        raise ValueError("cannot infer sample count from an empty file set")
    n_effective = min(int(count) for count in counts)
    if n_effective <= 1:
        raise ValueError(f"need at least 2 rows per dataset, got counts={counts!r}")
    return int(n_effective)


def resolve_run_dimensions(
    *,
    actual_n_samples: int,
    ndata_train: int | None,
    ndata_val: int | None,
    batch_size: int | None,
    class_sr_max_points: int | None,
) -> tuple[int, int, int, int]:
    n_total = int(actual_n_samples)
    if n_total <= 1:
        raise ValueError(f"need at least 2 samples per dataset, got {n_total}")

    if ndata_train is not None:
        n_train = int(ndata_train)
    else:
        n_train = min(2000, max(16, n_total // 2))
    n_train = min(max(1, n_train), max(1, n_total - 1))

    if ndata_val is not None:
        n_val = int(ndata_val)
    else:
        n_val = min(2000, max(1, n_total - n_train))
    n_val = min(max(1, n_val), max(1, n_total - n_train))

    if batch_size is not None:
        resolved_batch_size = int(batch_size)
    else:
        resolved_batch_size = min(256, max(8, min(n_train, n_val)))
    resolved_batch_size = min(max(1, resolved_batch_size), max(1, min(n_train, n_val)))

    if class_sr_max_points is not None:
        resolved_class_points = int(class_sr_max_points)
    else:
        resolved_class_points = min(n_train, n_val)
    resolved_class_points = min(max(1, resolved_class_points), max(1, min(n_train, n_val)))

    return int(n_train), int(n_val), int(resolved_batch_size), int(resolved_class_points)


def _scalar_param_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            return None
        raw = raw[0]
    try:
        out = float(raw)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def build_classsr_param_map(
    class_sr_payload: dict[str, Any],
    *,
    dataset_index: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for tag, raw in dict(class_sr_payload.get("class_params", {}) or {}).items():
        value = _scalar_param_value(raw)
        if value is not None:
            out[str(tag)] = float(value)
    experiment_params = list(class_sr_payload.get("experiment_params", []) or [])
    if int(dataset_index) < len(experiment_params):
        for tag, raw in dict(experiment_params[int(dataset_index)] or {}).items():
            value = _scalar_param_value(raw)
            if value is not None:
                out[str(tag)] = float(value)
    return out


def evaluate_ast_numeric(
    node: Any,
    *,
    x_values: Sequence[float],
    param_values: dict[str, float] | None = None,
) -> float:
    params = dict(param_values or {})

    if isinstance(node, ConstNode):
        return float(node.value)
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "") or "").lower()
        tag = getattr(node, "tag", None)
        kwargs = dict(getattr(node, "kwargs", {}) or {})
        if kind == "var":
            var_idxs = tuple(getattr(node, "var_idxs", ()) or ())
            if len(var_idxs) != 1:
                raise ValueError(f"expected scalar var leaf, got var_idxs={var_idxs!r}")
            idx = int(var_idxs[0])
            return float(x_values[idx])
        if kind in {"scale", "free_const"}:
            if tag is not None and str(tag) in params:
                return float(params[str(tag)])
            fallback = kwargs.get("value", kwargs.get("init", 1.0))
            return float(fallback)
        if kind == "fixed_const":
            if tag is not None and str(tag) in params:
                return float(params[str(tag)])
            return float(kwargs.get("value", 1.0))
        raise ValueError(f"unsupported AtomNode kind {kind!r} in Kepler demo evaluator")
    if isinstance(node, AddNode):
        return evaluate_ast_numeric(node.left, x_values=x_values, param_values=params) + evaluate_ast_numeric(
            node.right,
            x_values=x_values,
            param_values=params,
        )
    if isinstance(node, MulNode):
        return evaluate_ast_numeric(node.left, x_values=x_values, param_values=params) * evaluate_ast_numeric(
            node.right,
            x_values=x_values,
            param_values=params,
        )
    if isinstance(node, PowNode):
        base = evaluate_ast_numeric(node.base, x_values=x_values, param_values=params)
        return float(base ** float(node.exponent))
    if isinstance(node, SinNode):
        return float(math.sin(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, CosNode):
        return float(math.cos(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ExpNode):
        return float(math.exp(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, LogNode):
        return float(math.log(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, AbsNode):
        return float(abs(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, RealNode):
        return float(np.real(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ImagNode):
        return float(np.imag(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ArgNode):
        return float(np.angle(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ConjNode):
        value = evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)
        return float(np.conj(value).real)
    raise ValueError(f"unsupported AST node type {type(node)!r} in Kepler demo evaluator")


def _tag_to_leafidx(root: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, atom in enumerate(collect_all_atoms(root)):
        tag = str(getattr(atom, "tag", None) or f"leaf{idx}")
        out.setdefault(tag, idx)
    return out


def _set_leaf_params_from_flat(leaf: torch.nn.Module, raw_values: Sequence[float]) -> None:
    flat = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    offset = 0
    with torch.no_grad():
        for param in leaf.parameters():
            n = int(param.numel())
            if offset + n > int(flat.size):
                raise ValueError(
                    f"parameter vector too short for leaf {type(leaf).__name__}: "
                    f"needed at least {offset + n}, got {flat.size}"
                )
            view = flat[offset:offset + n].reshape(tuple(param.shape))
            tensor = torch.as_tensor(view, dtype=param.dtype, device=param.device)
            param.copy_(tensor)
            offset += n
    if offset != int(flat.size):
        raise ValueError(
            f"parameter vector size mismatch for leaf {type(leaf).__name__}: "
            f"consumed {offset}, got {flat.size}"
        )


def _candidate_model_ckpt_paths(class_sr_json_path: str | Path) -> list[Path]:
    path = Path(class_sr_json_path).resolve()
    stem = path.name
    if stem.endswith("_classSR.json"):
        stem = stem[: -len("_classSR.json")]
    candidates = [
        path.parent / f"{stem}.identity.mod",
        path.parent / f"{stem}.mod",
    ]
    try:
        repo_root = path.parents[3]
        candidates.extend(
            [
                repo_root / "models" / f"{stem}.identity.mod",
                repo_root / "models" / f"{stem}.mod",
            ]
        )
    except IndexError:
        pass
    out = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def find_model_ckpt_path(class_sr_json_path: str | Path) -> Path | None:
    for candidate in _candidate_model_ckpt_paths(class_sr_json_path):
        if candidate.exists():
            return candidate
    return None


def load_stagea_model_template(
    model_ckpt_path: str | Path,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    device_obj = torch.device("cpu") if device is None else torch.device(device)
    payload = torch.load(Path(model_ckpt_path), map_location=device_obj, weights_only=False)
    ast_mod = payload.get("ast", None)
    state_dict = payload.get("model_state_dict", None)
    if ast_mod is None or state_dict is None:
        raise ValueError(f"model checkpoint missing ast/model_state_dict: {model_ckpt_path}")

    sync_ast_num_segments_from_state_dict(ast_mod, state_dict)
    model_hp = ModelHyperparams()
    leaf_builder = LeafBuilder(model_hp, device_obj, dtype)
    model, _, _ = build_composite_ast(
        ast_mod,
        num_segments=None,
        dual_layer=None,
        leaf_builder=leaf_builder,
        device=device_obj,
        dtype=dtype,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    tagged_ast, reuse = ast_from_composite(model)
    return {
        "model": model,
        "tagged_ast": tagged_ast,
        "reuse": reuse,
        "leaf_builder": leaf_builder,
        "fresh_nn_factory": make_stage_a_nn_factory(leaf_builder),
        "device": device_obj,
        "dtype": dtype,
    }


def build_stageb_model_from_template(
    stageb_root: Any,
    *,
    template: dict[str, Any],
) -> torch.nn.Module:
    device_obj = template["device"]
    dtype = template["dtype"]
    reuse_build = {
        str(tag): copy.deepcopy(leaf).to(device=device_obj, dtype=dtype)
        for tag, leaf in dict(template["reuse"]).items()
    }
    nn_factory = make_reuse_only_nn_factory(
        device=device_obj,
        dtype=dtype,
        fresh_nn_factory=template["fresh_nn_factory"],
    )
    model = build_composite_from_ast(
        stageb_root,
        dtype=dtype,
        device=device_obj,
        nn_factory=nn_factory,
        reuse=reuse_build,
    )
    model.eval()
    return model


def apply_classsr_params_to_model(
    model: torch.nn.Module,
    *,
    root: Any,
    class_sr_payload: dict[str, Any],
    dataset_index: int,
) -> None:
    tag_to_leafidx = _tag_to_leafidx(root)
    class_params = dict(class_sr_payload.get("class_params", {}) or {})
    experiment_params = list(class_sr_payload.get("experiment_params", []) or [])
    params_by_tag: dict[str, Sequence[float]] = {}
    for tag, values in class_params.items():
        params_by_tag[str(tag)] = values
    if int(dataset_index) < len(experiment_params):
        for tag, values in dict(experiment_params[int(dataset_index)] or {}).items():
            params_by_tag[str(tag)] = values
    for tag, values in params_by_tag.items():
        leaf_idx = tag_to_leafidx.get(str(tag), None)
        if leaf_idx is None or leaf_idx >= len(model.leaf):
            continue
        _set_leaf_params_from_flat(model.leaf[leaf_idx], values)


def suggest_probe_points_from_r_values(
    r_values: Sequence[float],
    *,
    n_points: int = 9,
) -> np.ndarray:
    arr = np.asarray(r_values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        raise ValueError("suggest_probe_points_from_r_values requires positive finite radii")
    quantiles = np.linspace(0.05, 0.95, int(n_points))
    points = np.quantile(arr, quantiles)
    points = np.asarray(points, dtype=np.float64)
    points = np.maximum(points, 1.0e-6)
    return points


def suggest_symbolic_readout_points_from_r_values(
    r_values: Sequence[float],
    *,
    n_points: int = 129,
    qmin: float = 0.01,
    qmax: float = 0.99,
) -> np.ndarray:
    if int(n_points) < 3:
        raise ValueError("suggest_symbolic_readout_points_from_r_values requires at least 3 points")
    arr = np.asarray(r_values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        raise ValueError("suggest_symbolic_readout_points_from_r_values requires positive finite radii")
    lo = float(np.clip(qmin, 0.0, 1.0))
    hi = float(np.clip(qmax, lo, 1.0))
    quantiles = np.linspace(lo, hi, int(n_points), dtype=np.float64)
    points = np.asarray(np.quantile(arr, quantiles), dtype=np.float64)
    points = np.maximum(points, 1.0e-6)
    return np.unique(points)


def make_probe_clouds_from_r_values(
    r_values: Sequence[float],
    *,
    n_clouds: int = 8,
    n_points: int = 9,
    seed: int = 123,
) -> list[np.ndarray]:
    if int(n_clouds) <= 0:
        raise ValueError("n_clouds must be positive")
    if int(n_points) < 3:
        raise ValueError("n_points must be at least 3")

    arr = np.asarray(r_values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        raise ValueError("make_probe_clouds_from_r_values requires positive finite radii")

    base_quantiles = np.linspace(0.05, 0.95, int(n_points), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    clouds = [np.asarray(np.quantile(arr, base_quantiles), dtype=np.float64)]
    for _ in range(1, int(n_clouds)):
        jitter = rng.normal(loc=0.0, scale=0.03, size=int(n_points))
        q = np.clip(base_quantiles + jitter, 0.02, 0.98)
        q.sort()
        clouds.append(np.asarray(np.quantile(arr, q), dtype=np.float64))
    return [np.maximum(cloud, 1.0e-6) for cloud in clouds]


def extract_inverse_power_coefficients_from_predictor(
    predictor,
    *,
    exponents: Sequence[float],
    probe_points: Sequence[float],
    fit_intercept: bool = True,
) -> dict[str, Any]:
    x = np.asarray(probe_points, dtype=np.float64).reshape(-1)
    exponents_list = [float(exp) for exp in exponents]
    min_points = len(exponents_list) + (1 if bool(fit_intercept) else 0)
    if x.size < min_points:
        raise ValueError("need at least as many probe points as fitted coefficients")
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("probe points must be positive finite radii")

    y = np.asarray([float(predictor(float(v))) for v in x], dtype=np.float64)
    finite_mask = np.isfinite(y)
    if not np.all(finite_mask):
        x = x[finite_mask]
        y = y[finite_mask]
    if x.size < min_points:
        raise ValueError(
            "insufficient finite predictor outputs for inverse-power extraction "
            f"(retained {int(x.size)} of {int(len(finite_mask))} points)"
        )
    cols = []
    if bool(fit_intercept):
        cols.append(np.ones_like(x))
    for exponent in exponents_list:
        cols.append(np.power(x, -float(exponent)))
    design = np.column_stack(cols)
    coeff_vec, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeff_vec
    residuals = y - pred
    power_design = np.column_stack([np.power(x, -float(exponent)) for exponent in exponents_list])
    power_gram = power_design.T @ power_design
    power_target = power_design.T @ y

    offset = 1 if bool(fit_intercept) else 0
    coeffs = np.asarray(coeff_vec[offset:], dtype=np.float64)
    coeff_by_exponent = {
        f"r^-{float(exp):g}": float(coeff)
        for exp, coeff in zip(exponents, coeffs)
    }
    return {
        "intercept": float(coeff_vec[0]) if bool(fit_intercept) else 0.0,
        "coeffs": coeffs,
        "coeff_by_exponent": coeff_by_exponent,
        "exponents": list(exponents_list),
        "probe_points": x,
        "probe_values": y,
        "probe_residuals": residuals,
        "probe_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_probe_abs_residual": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        "power_fit_stats": {
            "exponents": list(exponents_list),
            "gram": np.asarray(power_gram, dtype=np.float64),
            "target": np.asarray(power_target, dtype=np.float64),
            "y_sq_sum": float(np.dot(y, y)),
            "n_samples": int(x.size),
            "n_requested_samples": int(len(finite_mask)),
        },
    }


def extract_inverse_power_coefficients_from_ast(
    root: Any,
    *,
    param_values: dict[str, float] | None = None,
    exponents: Sequence[float],
    probe_points: Sequence[float],
    num_vars: int = 1,
) -> dict[str, Any]:
    if int(num_vars) != 1:
        raise ValueError("Kepler inverse-power extraction expects exactly one raw variable")
    return extract_inverse_power_coefficients_from_predictor(
        lambda value: evaluate_ast_numeric(root, x_values=[value], param_values=param_values),
        exponents=exponents,
        probe_points=probe_points,
        fit_intercept=True,
    )


def extract_inverse_power_coefficients_from_model(
    model: torch.nn.Module,
    *,
    xcoord_system: XCoordSystem | None = None,
    exponents: Sequence[float],
    probe_points: Sequence[float],
    num_vars: int = 1,
) -> dict[str, Any]:
    if int(num_vars) != 1:
        raise ValueError("Kepler inverse-power extraction expects exactly one raw variable")
    xcoords = xcoord_system

    def _predict(value: float) -> float:
        x_arr = np.asarray([[float(value)]], dtype=np.float64)
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float64)
        if xcoords is not None and not xcoords.is_identity():
            x_tensor = xcoords.apply_torch(x_tensor)
        with torch.no_grad():
            y = model(x_tensor)
        y_arr = np.asarray(y.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        if y_arr.size != 1:
            raise ValueError(f"expected scalar model output, got shape {tuple(y.shape)!r}")
        return float(y_arr[0])

    return extract_inverse_power_coefficients_from_predictor(
        _predict,
        exponents=exponents,
        probe_points=probe_points,
        fit_intercept=True,
    )


def extract_classsr_inverse_power_rows(
    *,
    class_sr_json_path: str | Path,
    stageb_pkl_path: str | Path,
    exponents: Sequence[float],
    dataset_probe_points: dict[str, Sequence[float]] | None = None,
    dataset_sample_points: dict[str, Sequence[float]] | None = None,
    model_ckpt_path: str | Path | None = None,
    num_vars: int = 1,
) -> dict[str, Any]:
    class_payload = load_classsr_payload(class_sr_json_path)
    stageb_payload = load_stageb_payload(stageb_pkl_path)
    root = stageb_payload["stageB_ast"]
    dataset_ids = list(stageb_payload.get("stageB_dataset_ids", []) or [])
    experiment_params = list(class_payload.get("experiment_params", []) or [])
    if not dataset_ids:
        dataset_ids = [f"dataset_{i}" for i in range(len(experiment_params))]

    xcoord_system = XCoordSystem.from_map(
        stageb_payload.get("x_transform_map", None),
        Nx_raw=int(num_vars),
    )
    use_model_bridge = False
    template = None
    resolved_model_ckpt_path = None if model_ckpt_path is None else Path(model_ckpt_path)
    if resolved_model_ckpt_path is None:
        resolved_model_ckpt_path = find_model_ckpt_path(class_sr_json_path)
    if resolved_model_ckpt_path is not None:
        try:
            template = load_stagea_model_template(resolved_model_ckpt_path)
            use_model_bridge = True
        except Exception:
            template = None
            use_model_bridge = False

    rows = []
    for idx, dataset_id in enumerate(dataset_ids):
        sample_points = None
        fit_point_source = "probe_points"
        if dataset_sample_points is not None:
            sample_points = dataset_sample_points.get(str(dataset_id), None)
            if sample_points is None:
                sample_points = dataset_sample_points.get(_canonical_symbolic_dataset_id(str(dataset_id)), None)
            if sample_points is not None:
                fit_point_source = "dataset_samples"
        if dataset_probe_points is not None:
            if sample_points is None:
                sample_points = dataset_probe_points.get(str(dataset_id), None)
                if sample_points is None:
                    sample_points = dataset_probe_points.get(_canonical_symbolic_dataset_id(str(dataset_id)), None)
        if sample_points is None:
            raise ValueError(f"missing fit points for dataset_id={dataset_id!r}")
        param_values = build_classsr_param_map(class_payload, dataset_index=idx)
        if use_model_bridge:
            model = build_stageb_model_from_template(root, template=template)
            apply_classsr_params_to_model(
                model,
                root=root,
                class_sr_payload=class_payload,
                dataset_index=idx,
            )
            fitted = extract_inverse_power_coefficients_from_model(
                model,
                xcoord_system=xcoord_system,
                exponents=exponents,
                probe_points=sample_points,
                num_vars=int(num_vars),
            )
            extraction_mode = "model"
        else:
            fitted = extract_inverse_power_coefficients_from_ast(
                root,
                param_values=param_values,
                exponents=exponents,
                probe_points=sample_points,
                num_vars=int(num_vars),
            )
            extraction_mode = "ast"
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "param_values": dict(param_values),
                "extraction_mode": extraction_mode,
                "intercept": float(fitted["intercept"]),
                "coeffs": np.asarray(fitted["coeffs"], dtype=np.float64),
                "coeff_by_exponent": dict(fitted["coeff_by_exponent"]),
                "probe_rmse": float(fitted["probe_rmse"]),
                "max_probe_abs_residual": float(fitted["max_probe_abs_residual"]),
                "power_fit_stats": dict(fitted["power_fit_stats"]),
                "fit_point_source": str(fit_point_source),
            }
        )

    return {
        "class_sr_json_path": str(class_sr_json_path),
        "stageb_pkl_path": str(stageb_pkl_path),
        "model_ckpt_path": None if resolved_model_ckpt_path is None else str(resolved_model_ckpt_path),
        "dataset_ids": [str(item) for item in dataset_ids],
        "y_expr_str": stageb_payload.get("y_expr_str", None),
        "phi_expr_strs": stageb_payload.get("phi_expr_strs", None),
        "derived_invariants": list(class_payload.get("derived_invariants", []) or []),
        "exponents": [float(exp) for exp in exponents],
        "rows": rows,
    }


def _refit_symbolic_shared_mu_from_rows(
    merged_rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not merged_rows:
        return None
    n_rows = len(merged_rows)
    normal = np.zeros((n_rows + 1, n_rows + 1), dtype=np.float64)
    rhs = np.zeros(n_rows + 1, dtype=np.float64)
    per_dataset = []
    for row_idx, row in enumerate(merged_rows):
        stats = row.get("rddot_power_fit_stats", None)
        if not isinstance(stats, dict):
            return None
        exponents = [float(item) for item in list(stats.get("exponents", []) or [])]
        if exponents != [3.0, 2.0]:
            return None
        gram = np.asarray(stats.get("gram", None), dtype=np.float64)
        target = np.asarray(stats.get("target", None), dtype=np.float64).reshape(-1)
        y_sq_sum = float(stats.get("y_sq_sum", float("nan")))
        n_samples = int(stats.get("n_samples", 0))
        if gram.shape != (2, 2) or target.shape != (2,) or not np.isfinite(y_sq_sum) or n_samples <= 0:
            return None

        normal[row_idx, row_idx] += float(gram[0, 0])
        normal[row_idx, -1] += float(gram[0, 1])
        normal[-1, row_idx] += float(gram[1, 0])
        normal[-1, -1] += float(gram[1, 1])
        rhs[row_idx] += float(target[0])
        rhs[-1] += float(target[1])
        per_dataset.append(
            {
                "dataset_id": str(row["dataset_id"]),
                "gram": gram,
                "target": target,
                "y_sq_sum": y_sq_sum,
                "n_samples": n_samples,
            }
        )

    coeffs, _, _, _ = np.linalg.lstsq(normal, rhs, rcond=None)
    beta_shared = float(coeffs[-1])
    mu_fit = float(-beta_shared)
    k_by_dataset: dict[str, float] = {}
    rmse_values = []
    for row_idx, dataset_stats in enumerate(per_dataset):
        k_fit = float(coeffs[row_idx])
        dataset_id = str(dataset_stats["dataset_id"])
        gram = np.asarray(dataset_stats["gram"], dtype=np.float64)
        target = np.asarray(dataset_stats["target"], dtype=np.float64)
        y_sq_sum = float(dataset_stats["y_sq_sum"])
        n_samples = int(dataset_stats["n_samples"])
        local_coeffs = np.asarray([k_fit, beta_shared], dtype=np.float64)
        sse = float(y_sq_sum - 2.0 * np.dot(local_coeffs, target) + local_coeffs @ gram @ local_coeffs)
        sse = max(sse, 0.0)
        rmse = float(np.sqrt(sse / max(n_samples, 1)))
        k_by_dataset[dataset_id] = k_fit
        rmse_values.append(rmse)

    return {
        "method": "joint_symbolic_rddot_fit",
        "shared_r2_coefficient": beta_shared,
        "mu": mu_fit,
        "k_by_dataset": k_by_dataset,
        "mean_rmse": _mean(rmse_values),
        "max_rmse": float(max(rmse_values) if rmse_values else 0.0),
    }


def analyze_classsr_probe_stability(
    *,
    class_sr_json_path: str | Path,
    stageb_pkl_path: str | Path,
    datasets: Sequence[KeplerReducedDataset],
    exponents: Sequence[float],
    n_clouds: int = 8,
    n_points: int = 9,
    seed: int = 123,
    model_ckpt_path: str | Path | None = None,
    num_vars: int = 1,
) -> dict[str, Any]:
    dataset_list = sorted(list(datasets), key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    if not dataset_list:
        raise ValueError("analyze_classsr_probe_stability requires at least one dataset")

    dataset_clouds: dict[str, list[np.ndarray]] = {}
    for idx, dataset in enumerate(dataset_list):
        dataset_clouds[dataset.orbit_id] = make_probe_clouds_from_r_values(
            dataset.r,
            n_clouds=int(n_clouds),
            n_points=int(n_points),
            seed=int(seed) + 1009 * idx,
        )

    tables = []
    for cloud_idx in range(int(n_clouds)):
        table = extract_classsr_inverse_power_rows(
            class_sr_json_path=class_sr_json_path,
            stageb_pkl_path=stageb_pkl_path,
            exponents=exponents,
            dataset_probe_points={
                orbit_id: clouds[cloud_idx]
                for orbit_id, clouds in dataset_clouds.items()
            },
            model_ckpt_path=model_ckpt_path,
            num_vars=int(num_vars),
        )
        tables.append(table)

    per_dataset = []
    max_intercept_std = 0.0
    max_probe_rmse_mean = 0.0
    max_probe_rmse_max = 0.0
    max_coeff_std_by_exponent = {f"r^-{float(exp):g}": 0.0 for exp in exponents}
    max_coeff_rel_std_by_exponent = {f"r^-{float(exp):g}": 0.0 for exp in exponents}

    rows_by_dataset = {}
    for table in tables:
        for row in list(table.get("rows", []) or []):
            dataset_id = _canonical_symbolic_dataset_id(str(row["dataset_id"]))
            rows_by_dataset.setdefault(dataset_id, []).append(row)

    for dataset in dataset_list:
        rows = rows_by_dataset.get(dataset.orbit_id, None)
        if not rows:
            continue
        coeff_matrix = np.asarray([row["coeffs"] for row in rows], dtype=np.float64)
        intercepts = np.asarray([float(row["intercept"]) for row in rows], dtype=np.float64)
        probe_rmses = np.asarray([float(row["probe_rmse"]) for row in rows], dtype=np.float64)
        coeff_mean = np.mean(coeff_matrix, axis=0)
        coeff_std = np.std(coeff_matrix, axis=0)
        coeff_rel_std = coeff_std / np.maximum(np.abs(coeff_mean), 1.0e-12)

        coeff_mean_by_exponent = {}
        coeff_std_by_exponent = {}
        coeff_rel_std_by_exponent = {}
        for exp, mean_val, std_val, rel_std in zip(exponents, coeff_mean, coeff_std, coeff_rel_std):
            key = f"r^-{float(exp):g}"
            coeff_mean_by_exponent[key] = float(mean_val)
            coeff_std_by_exponent[key] = float(std_val)
            coeff_rel_std_by_exponent[key] = float(rel_std)
            max_coeff_std_by_exponent[key] = max(max_coeff_std_by_exponent[key], float(std_val))
            max_coeff_rel_std_by_exponent[key] = max(max_coeff_rel_std_by_exponent[key], float(rel_std))

        intercept_std = float(np.std(intercepts))
        probe_rmse_mean = float(np.mean(probe_rmses))
        probe_rmse_max = float(np.max(probe_rmses))
        max_intercept_std = max(max_intercept_std, intercept_std)
        max_probe_rmse_mean = max(max_probe_rmse_mean, probe_rmse_mean)
        max_probe_rmse_max = max(max_probe_rmse_max, probe_rmse_max)

        per_dataset.append(
            {
                "dataset_id": dataset.orbit_id,
                "split": dataset.split,
                "n_clouds": int(len(rows)),
                "intercept_mean": float(np.mean(intercepts)),
                "intercept_std": intercept_std,
                "coeff_mean_by_exponent": coeff_mean_by_exponent,
                "coeff_std_by_exponent": coeff_std_by_exponent,
                "coeff_rel_std_by_exponent": coeff_rel_std_by_exponent,
                "probe_rmse_mean": probe_rmse_mean,
                "probe_rmse_max": probe_rmse_max,
            }
        )

    per_dataset.sort(key=lambda row: (_split_sort_key(str(row["split"])), str(row["dataset_id"])))
    return {
        "class_sr_json_path": str(class_sr_json_path),
        "stageb_pkl_path": str(stageb_pkl_path),
        "n_clouds": int(n_clouds),
        "n_points_per_cloud": int(n_points),
        "exponents": [float(exp) for exp in exponents],
        "aggregate": {
            "max_intercept_std": float(max_intercept_std),
            "max_coeff_std_by_exponent": dict(max_coeff_std_by_exponent),
            "max_coeff_rel_std_by_exponent": dict(max_coeff_rel_std_by_exponent),
            "max_probe_rmse_mean": float(max_probe_rmse_mean),
            "max_probe_rmse_max": float(max_probe_rmse_max),
        },
        "per_dataset": per_dataset,
    }


def evaluate_symbolic_holdout_generalization(
    symbolic_run_summary: dict[str, Any],
    *,
    holdout_datasets: Sequence[KeplerReducedDataset],
) -> dict[str, Any]:
    if str(symbolic_run_summary.get("status")) != "extractable":
        raise ValueError("symbolic_run_summary must have status='extractable'")
    holdout_list = sorted(list(holdout_datasets), key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    if not holdout_list:
        raise ValueError("evaluate_symbolic_holdout_generalization requires at least one holdout dataset")

    symbolic_summary = dict(symbolic_run_summary["symbolic_summary"])
    stage_b = dict(symbolic_summary["stage_b"])
    lift = dict(symbolic_summary["coefficient_lift"])
    mu_train = float(stage_b["mu_mean"])
    lift_intercept = float(lift["intercept"])
    lift_slope = float(lift["slope"])

    per_dataset = []
    max_ell_rel_error = 0.0
    max_k_abs_error = 0.0
    radial_rmses = []
    oracle_radial_rmses = []
    lift_penalties = []
    for dataset in holdout_list:
        inv_r2 = 1.0 / np.square(dataset.r)
        ell_fit = float(np.dot(inv_r2, dataset.omega) / np.dot(inv_r2, inv_r2))
        omega_pred = ell_fit * inv_r2
        omega_rmse = _rmse(dataset.omega, omega_pred)

        k_pred = lift_intercept + lift_slope * (ell_fit * ell_fit)
        radial_pred = k_pred / np.power(dataset.r, 3) - mu_train / np.square(dataset.r)
        radial_rmse = _rmse(dataset.rddot, radial_pred)

        inv_r3 = 1.0 / np.power(dataset.r, 3)
        rhs = dataset.rddot + mu_train / np.square(dataset.r)
        oracle_k = float(np.dot(inv_r3, rhs) / np.dot(inv_r3, inv_r3))
        oracle_radial_pred = oracle_k / np.power(dataset.r, 3) - mu_train / np.square(dataset.r)
        oracle_radial_rmse = _rmse(dataset.rddot, oracle_radial_pred)
        lift_penalty = float(radial_rmse - oracle_radial_rmse)

        ell_rel_error = abs(ell_fit - float(dataset.h)) / max(abs(float(dataset.h)), 1.0e-12)
        k_abs_error = abs(k_pred - float(dataset.h * dataset.h))
        max_ell_rel_error = max(max_ell_rel_error, ell_rel_error)
        max_k_abs_error = max(max_k_abs_error, k_abs_error)
        radial_rmses.append(radial_rmse)
        oracle_radial_rmses.append(oracle_radial_rmse)
        lift_penalties.append(lift_penalty)

        per_dataset.append(
            {
                "orbit_id": dataset.orbit_id,
                "split": dataset.split,
                "ell_fit_from_areal_law": ell_fit,
                "ell_true": float(dataset.h),
                "ell_rel_error": float(ell_rel_error),
                "omega_rmse": float(omega_rmse),
                "k_pred_from_lift": float(k_pred),
                "k_true": float(dataset.h * dataset.h),
                "k_abs_error": float(k_abs_error),
                "mu_train": float(mu_train),
                "radial_rmse": float(radial_rmse),
                "oracle_k_if_refit_from_rddot": float(oracle_k),
                "oracle_radial_rmse_if_refit": float(oracle_radial_rmse),
                "lift_penalty_vs_oracle_refit": float(lift_penalty),
            }
        )

    return {
        "train_mu_mean": float(mu_train),
        "lift_intercept": float(lift_intercept),
        "lift_slope": float(lift_slope),
        "n_holdout_orbits": int(len(holdout_list)),
        "aggregate": {
            "max_ell_rel_error": float(max_ell_rel_error),
            "max_k_abs_error": float(max_k_abs_error),
            "mean_radial_rmse": _mean(radial_rmses),
            "max_radial_rmse": float(max(radial_rmses) if radial_rmses else 0.0),
            "mean_oracle_radial_rmse_if_refit": _mean(oracle_radial_rmses),
            "max_lift_penalty_vs_oracle_refit": float(max(lift_penalties) if lift_penalties else 0.0),
        },
        "per_dataset": per_dataset,
    }


def _canonical_symbolic_dataset_id(dataset_id: str) -> str:
    text = str(dataset_id)
    for prefix in ("omega_", "rddot_"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def merge_symbolic_kepler_tables(
    omega_table: dict[str, Any],
    rddot_table: dict[str, Any],
    *,
    datasets_by_id: dict[str, KeplerReducedDataset] | None = None,
) -> list[dict[str, Any]]:
    r_rows_by_id = {
        _canonical_symbolic_dataset_id(str(row["dataset_id"])): row
        for row in list(rddot_table.get("rows", []) or [])
    }
    merged = []
    for omega_row in list(omega_table.get("rows", []) or []):
        source_dataset_id = str(omega_row["dataset_id"])
        dataset_id = _canonical_symbolic_dataset_id(source_dataset_id)
        r_row = r_rows_by_id.get(dataset_id, None)
        if r_row is None:
            continue
        omega_coeffs = np.asarray(omega_row["coeffs"], dtype=np.float64)
        rddot_coeffs = np.asarray(r_row["coeffs"], dtype=np.float64)
        dataset = None if datasets_by_id is None else datasets_by_id.get(dataset_id, None)
        merged.append(
            {
                "dataset_id": dataset_id,
                "omega_dataset_id": source_dataset_id,
                "rddot_dataset_id": str(r_row["dataset_id"]),
                "split": None if dataset is None else str(dataset.split),
                "ell": float(omega_coeffs[0]),
                "k": float(rddot_coeffs[0]),
                "minus_mu": float(rddot_coeffs[1]),
                "omega_intercept": float(omega_row["intercept"]),
                "rddot_intercept": float(r_row["intercept"]),
                "omega_probe_rmse": float(omega_row["probe_rmse"]),
                "rddot_probe_rmse": float(r_row["probe_rmse"]),
                "omega_extraction_mode": str(omega_row["extraction_mode"]),
                "rddot_extraction_mode": str(r_row["extraction_mode"]),
                "omega_fit_point_source": str(omega_row.get("fit_point_source", "probe_points")),
                "rddot_fit_point_source": str(r_row.get("fit_point_source", "probe_points")),
                "omega_power_fit_stats": copy.deepcopy(omega_row.get("power_fit_stats", None)),
                "rddot_power_fit_stats": copy.deepcopy(r_row.get("power_fit_stats", None)),
                "h_true": None if dataset is None else float(dataset.h),
                "k_true": None if dataset is None else float(dataset.h * dataset.h),
                "energy_true": None if dataset is None else float(dataset.energy),
            }
        )
    merged.sort(key=lambda row: (_split_sort_key(str(row.get("split", ""))), str(row["dataset_id"])))
    return merged


def analyze_symbolic_kepler_family(
    merged_rows: Sequence[dict[str, Any]],
    *,
    datasets: Sequence[KeplerReducedDataset],
) -> dict[str, Any]:
    dataset_list = sorted(list(datasets), key=lambda ds: (_split_sort_key(ds.split), ds.orbit_id))
    if not merged_rows:
        raise ValueError("analyze_symbolic_kepler_family requires at least one merged row")
    datasets_by_id = {dataset.orbit_id: dataset for dataset in dataset_list}
    ell_by_dataset = {str(row["dataset_id"]): float(row["ell"]) for row in merged_rows}
    raw_k_by_dataset = {str(row["dataset_id"]): float(row["k"]) for row in merged_rows}
    k_by_dataset = dict(raw_k_by_dataset)
    mu_values = np.asarray([-float(row["minus_mu"]) for row in merged_rows], dtype=np.float64)
    mu_raw_mean = float(np.mean(mu_values))
    mu_std = float(np.std(mu_values))
    mu_refit = _refit_symbolic_shared_mu_from_rows(merged_rows)
    mu_mean = float(mu_raw_mean)
    stage_b_fit_rmse = float("nan")
    mu_refit_method = "per_dataset_mean"
    if mu_refit is not None:
        mu_mean = float(mu_refit["mu"])
        k_by_dataset = dict(mu_refit["k_by_dataset"])
        stage_b_fit_rmse = float(mu_refit["max_rmse"])
        mu_refit_method = str(mu_refit["method"])

    max_h_rel_error = 0.0
    max_k_abs_error = 0.0
    for row in merged_rows:
        dataset_id = str(row["dataset_id"])
        dataset = datasets_by_id[dataset_id]
        max_h_rel_error = max(
            max_h_rel_error,
            abs(float(row["ell"]) - float(dataset.h)) / max(abs(float(dataset.h)), 1.0e-12),
        )
        max_k_abs_error = max(
            max_k_abs_error,
            abs(float(k_by_dataset[dataset_id]) - float(dataset.h * dataset.h)),
        )

    coefficient_lift = lift_coefficient_relation(
        datasets=dataset_list,
        ell_by_dataset=ell_by_dataset,
        k_by_dataset=k_by_dataset,
    )
    energy = recover_energy_integral(
        datasets=dataset_list,
        ell_by_dataset=ell_by_dataset,
        mu=mu_mean,
    )
    hamiltonian = assemble_kepler_hamiltonian(
        datasets=dataset_list,
        ell_by_dataset=ell_by_dataset,
        mu=mu_mean,
        coefficient_lift=coefficient_lift,
        energy=energy,
    )

    return {
        "stage_a": {
            "max_h_rel_error": float(max_h_rel_error),
        },
        "stage_b": {
            "mu_values": mu_values,
            "mu_raw_mean": float(mu_raw_mean),
            "mu_mean": float(mu_mean),
            "mu_std": float(mu_std),
            "mu_abs_error": float(abs(mu_mean - float(dataset_list[0].mu))),
            "max_k_abs_error": float(max_k_abs_error),
            "fit_rmse": float(stage_b_fit_rmse),
            "mu_refit_method": str(mu_refit_method),
            "mu_refit": None if mu_refit is None else dict(mu_refit),
        },
        "coefficient_lift": coefficient_lift,
        "energy": energy,
        "hamiltonian": hamiltonian,
    }


__all__ = [
    "KeplerReducedDataset",
    "OrbitSpec",
    "_jsonable",
    "analyze_symbolic_kepler_family",
    "analyze_kepler_reduced_family",
    "assemble_kepler_hamiltonian",
    "apply_classsr_params_to_model",
    "build_default_kepler_datasets",
    "build_default_orbit_specs",
    "build_classsr_param_map",
    "build_stageb_model_from_template",
    "evaluate_radial_family_with_fixed_mu",
    "evaluate_symbolic_holdout_generalization",
    "analyze_classsr_probe_stability",
    "extract_classsr_inverse_power_rows",
    "extract_inverse_power_coefficients_from_ast",
    "extract_inverse_power_coefficients_from_model",
    "extract_inverse_power_coefficients_from_predictor",
    "find_model_ckpt_path",
    "fit_areal_law",
    "fit_radial_family",
    "generate_kepler_dataset",
    "infer_effective_n_samples",
    "lift_coefficient_relation",
    "load_classsr_payload",
    "load_kepler_datasets_from_manifest",
    "load_kepler_manifest",
    "load_stagea_model_template",
    "load_stageb_payload",
    "merge_symbolic_kepler_tables",
    "make_probe_clouds_from_r_values",
    "recover_energy_integral",
    "resolve_run_dimensions",
    "scan_inverse_power_family",
    "split_datasets",
    "assign_leverage_round_robin_splits",
    "LEVERAGE_ROUND_ROBIN_STRATEGY",
    "suggest_probe_points_from_r_values",
    "suggest_symbolic_readout_points_from_r_values",
    "target_filepaths",
    "write_generated_artifacts",
]
