#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Problem definitions and data generation utilities for a MOND PDE benchmark.

We use the AQUAL MOND equation:

    div(mu(|grad(phi)| / a0) * grad(phi)) = 4*pi*G*rho

To keep the benchmark reproducible and fast, the dataset is "manufactured":
1. Build a smooth potential phi(x, y) from Gaussian wells.
2. Compute rho from the MOND operator on that phi.
3. Add optional noise to observed phi.
4. Build a regression library from derivatives of noisy phi.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_INTERIOR_PAD = 2


@dataclass
class MONDProblem:
    id: str
    description: str
    mu_mode: str
    sources: list[tuple[float, float, float, float]] = field(default_factory=list)
    background_grad: tuple[float, float] = (0.0, 0.0)
    grid_size: int = 96
    domain_min: float = -1.0
    domain_max: float = 1.0
    a0: float = 1.0
    big_g: float = 1.0
    default_noise: float = 0.0
    deriv_eps: float = 1e-6


@dataclass
class GroundTruth:
    expected_terms: dict[str, float]
    coeff_rtol: float
    coeff_atol: float
    decoy_atol: float
    rms_tol: float


@dataclass
class MondDataset:
    feature_names: list[str]
    theta: np.ndarray
    target: np.ndarray
    x: np.ndarray
    y: np.ndarray
    phi_observed: np.ndarray
    rho: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    phi_true_grid: np.ndarray
    phi_observed_grid: np.ndarray
    rho_grid: np.ndarray
    mask_grid: np.ndarray
    metadata: dict[str, float | int | str | list]

    def save(self, npz_path: Path, csv_path: Path, meta_path: Path) -> None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            npz_path,
            feature_names=np.asarray(self.feature_names, dtype="<U64"),
            theta=self.theta,
            target=self.target,
            x=self.x,
            y=self.y,
            phi_observed=self.phi_observed,
            rho=self.rho,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            phi_true_grid=self.phi_true_grid,
            phi_observed_grid=self.phi_observed_grid,
            rho_grid=self.rho_grid,
            mask_grid=self.mask_grid.astype(np.uint8),
        )

        table = np.column_stack([self.x, self.y, self.phi_observed, self.rho, self.theta])
        header = "x,y,phi_observed,rho," + ",".join(self.feature_names)
        np.savetxt(csv_path, table, delimiter=",", header=header, comments="")
        meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, npz_path: Path, meta_path: Path | None = None) -> "MondDataset":
        blob = np.load(npz_path, allow_pickle=False)
        feature_names = [str(v) for v in blob["feature_names"].tolist()]
        metadata: dict[str, float | int | str | list] = {}
        if meta_path is not None and meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        return cls(
            feature_names=feature_names,
            theta=np.asarray(blob["theta"], dtype=np.float64),
            target=np.asarray(blob["target"], dtype=np.float64),
            x=np.asarray(blob["x"], dtype=np.float64),
            y=np.asarray(blob["y"], dtype=np.float64),
            phi_observed=np.asarray(blob["phi_observed"], dtype=np.float64),
            rho=np.asarray(blob["rho"], dtype=np.float64),
            x_grid=np.asarray(blob["x_grid"], dtype=np.float64),
            y_grid=np.asarray(blob["y_grid"], dtype=np.float64),
            phi_true_grid=np.asarray(blob["phi_true_grid"], dtype=np.float64),
            phi_observed_grid=np.asarray(blob["phi_observed_grid"], dtype=np.float64),
            rho_grid=np.asarray(blob["rho_grid"], dtype=np.float64),
            mask_grid=np.asarray(blob["mask_grid"], dtype=np.uint8).astype(bool),
            metadata=metadata,
        )


PROBLEMS: dict[str, MONDProblem] = {
    "000": MONDProblem(
        id="000",
        description="Deep-MOND with two smooth clumps",
        mu_mode="deep",
        sources=[
            (1.35, -0.35, 0.10, 0.24),
            (0.90, 0.28, -0.20, 0.18),
        ],
        background_grad=(0.10, -0.06),
        default_noise=0.0,
    ),
    "001": MONDProblem(
        id="001",
        description="Deep-MOND with three asymmetric clumps",
        mu_mode="deep",
        sources=[
            (1.10, -0.45, -0.30, 0.20),
            (0.80, 0.25, 0.25, 0.16),
            (0.55, 0.05, -0.05, 0.12),
        ],
        background_grad=(-0.08, 0.11),
        default_noise=0.0,
    ),
    "100": MONDProblem(
        id="100",
        description="Simple-interpolation MOND transition regime",
        mu_mode="simple",
        sources=[
            (1.20, -0.30, -0.12, 0.22),
            (0.95, 0.30, 0.24, 0.20),
            (0.45, 0.00, -0.42, 0.14),
        ],
        background_grad=(0.06, 0.05),
        default_noise=0.0,
    ),
}


def load_problems() -> dict[str, MONDProblem]:
    return dict(PROBLEMS)


def _build_grid(problem: MONDProblem, grid_size: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    n = int(problem.grid_size if grid_size is None else grid_size)
    if n < 8:
        raise ValueError(f"grid_size must be >= 8, got {n}")

    x = np.linspace(problem.domain_min, problem.domain_max, n, dtype=np.float64)
    y = np.linspace(problem.domain_min, problem.domain_max, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return x, y, xx, yy, dx, dy


def _build_potential(problem: MONDProblem, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    gx, gy = problem.background_grad
    phi = gx * xx + gy * yy
    for mass, x0, y0, sigma in problem.sources:
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        phi -= mass * np.exp(-0.5 * r2 / (sigma**2))
    return phi


def _compute_fields(phi: np.ndarray, dx: float, dy: float, eps: float) -> dict[str, np.ndarray]:
    dphi_dy, dphi_dx = np.gradient(phi, dy, dx, edge_order=2)
    d2phi_dx2 = np.gradient(dphi_dx, dx, axis=1, edge_order=2)
    d2phi_dy2 = np.gradient(dphi_dy, dy, axis=0, edge_order=2)
    lap_phi = d2phi_dx2 + d2phi_dy2

    grad_phi_sq = dphi_dx**2 + dphi_dy**2
    grad_norm = np.sqrt(grad_phi_sq + eps**2)
    dgrad_dy, dgrad_dx = np.gradient(grad_norm, dy, dx, edge_order=2)
    grad_g_dot_grad_phi = dgrad_dx * dphi_dx + dgrad_dy * dphi_dy

    return {
        "dphi_dx": dphi_dx,
        "dphi_dy": dphi_dy,
        "lap_phi": lap_phi,
        "grad_phi_sq": grad_phi_sq,
        "grad_norm": grad_norm,
        "grad_g_dot_grad_phi": grad_g_dot_grad_phi,
    }


def _mu_and_mup_over_a0(
    grad_norm: np.ndarray,
    *,
    a0: float,
    mu_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    a0_safe = max(float(a0), 1e-12)
    s = grad_norm / a0_safe

    if mu_mode == "deep":
        mu = s
        mup_over_a0 = np.full_like(s, 1.0 / a0_safe)
        return mu, mup_over_a0

    if mu_mode == "simple":
        denom = 1.0 + s
        mu = s / denom
        dmu_ds = 1.0 / (denom**2)
        mup_over_a0 = dmu_ds / a0_safe
        return mu, mup_over_a0

    raise ValueError(f"Unknown mu_mode={mu_mode!r}. Expected 'deep' or 'simple'.")


def _interior_mask(shape: tuple[int, int], pad: int) -> np.ndarray:
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")
    mask = np.ones(shape, dtype=bool)
    if pad > 0:
        mask[:pad, :] = False
        mask[-pad:, :] = False
        mask[:, :pad] = False
        mask[:, -pad:] = False
    return mask


def generate_dataset(
    problem: MONDProblem,
    *,
    noise_std: float | None = None,
    seed: int = 0,
    grid_size: int | None = None,
    interior_pad: int = DEFAULT_INTERIOR_PAD,
) -> MondDataset:
    noise = float(problem.default_noise if noise_std is None else noise_std)
    x_axis, y_axis, xx, yy, dx, dy = _build_grid(problem, grid_size)
    phi_true = _build_potential(problem, xx, yy)
    fields_true = _compute_fields(phi_true, dx, dy, eps=problem.deriv_eps)

    mu_true, mup_over_a0_true = _mu_and_mup_over_a0(
        fields_true["grad_norm"],
        a0=problem.a0,
        mu_mode=problem.mu_mode,
    )

    mond_operator_true = (
        mu_true * fields_true["lap_phi"]
        + mup_over_a0_true * fields_true["grad_g_dot_grad_phi"]
    )
    kappa = 4.0 * math.pi * problem.big_g * problem.a0
    rho_grid = mond_operator_true / kappa

    rng = np.random.default_rng(seed)
    phi_observed = phi_true.copy()
    if noise > 0.0:
        phi_observed += rng.normal(0.0, noise * float(np.std(phi_true)), size=phi_true.shape)

    fields_obs = _compute_fields(phi_observed, dx, dy, eps=problem.deriv_eps)
    mu_obs, mup_over_a0_obs = _mu_and_mup_over_a0(
        fields_obs["grad_norm"],
        a0=problem.a0,
        mu_mode=problem.mu_mode,
    )

    if problem.mu_mode == "deep":
        feature_arrays = {
            "g_lap_phi": fields_obs["grad_norm"] * fields_obs["lap_phi"],
            "grad_g_dot_grad_phi": fields_obs["grad_g_dot_grad_phi"],
            "lap_phi": fields_obs["lap_phi"],
            "grad_phi_sq": fields_obs["grad_phi_sq"],
            "g": fields_obs["grad_norm"],
            "1": np.ones_like(phi_observed),
        }
    else:
        feature_arrays = {
            "mu_lap_phi": mu_obs * fields_obs["lap_phi"],
            "muprime_grad_g_dot_grad_phi": mup_over_a0_obs * fields_obs["grad_g_dot_grad_phi"],
            "g_lap_phi": fields_obs["grad_norm"] * fields_obs["lap_phi"],
            "lap_phi": fields_obs["lap_phi"],
            "grad_phi_sq": fields_obs["grad_phi_sq"],
            "mu": mu_obs,
            "1": np.ones_like(phi_observed),
        }

    mask = _interior_mask(phi_observed.shape, int(interior_pad))
    feature_names = list(feature_arrays.keys())
    theta = np.column_stack([feature_arrays[name][mask].reshape(-1) for name in feature_names])
    target = rho_grid[mask].reshape(-1)
    x = xx[mask].reshape(-1)
    y = yy[mask].reshape(-1)
    phi_obs_vec = phi_observed[mask].reshape(-1)
    rho_vec = rho_grid[mask].reshape(-1)

    metadata: dict[str, float | int | str | list] = {
        "problem_id": problem.id,
        "description": problem.description,
        "mu_mode": problem.mu_mode,
        "grid_size": int(phi_observed.shape[0]),
        "domain_min": float(problem.domain_min),
        "domain_max": float(problem.domain_max),
        "a0": float(problem.a0),
        "G": float(problem.big_g),
        "kappa_4piGa0": float(kappa),
        "noise_std_rel_phi": float(noise),
        "seed": int(seed),
        "interior_pad": int(interior_pad),
        "n_samples": int(target.shape[0]),
        "sources_mass_x0_y0_sigma": [list(v) for v in problem.sources],
        "background_grad_xy": [float(problem.background_grad[0]), float(problem.background_grad[1])],
        "x_min": float(x_axis[0]),
        "x_max": float(x_axis[-1]),
        "y_min": float(y_axis[0]),
        "y_max": float(y_axis[-1]),
    }

    return MondDataset(
        feature_names=feature_names,
        theta=theta.astype(np.float64, copy=False),
        target=target.astype(np.float64, copy=False),
        x=x.astype(np.float64, copy=False),
        y=y.astype(np.float64, copy=False),
        phi_observed=phi_obs_vec.astype(np.float64, copy=False),
        rho=rho_vec.astype(np.float64, copy=False),
        x_grid=xx.astype(np.float64, copy=False),
        y_grid=yy.astype(np.float64, copy=False),
        phi_true_grid=phi_true.astype(np.float64, copy=False),
        phi_observed_grid=phi_observed.astype(np.float64, copy=False),
        rho_grid=rho_grid.astype(np.float64, copy=False),
        mask_grid=mask,
        metadata=metadata,
    )


def ground_truth_for_problem(problem: MONDProblem) -> GroundTruth:
    kappa = 4.0 * math.pi * problem.big_g * problem.a0
    inv_kappa = 1.0 / kappa

    if problem.mu_mode == "deep":
        return GroundTruth(
            expected_terms={
                "g_lap_phi": inv_kappa,
                "grad_g_dot_grad_phi": inv_kappa,
            },
            coeff_rtol=0.20,
            coeff_atol=0.015,
            decoy_atol=0.030,
            rms_tol=0.010,
        )

    return GroundTruth(
        expected_terms={
            "mu_lap_phi": inv_kappa,
            "muprime_grad_g_dot_grad_phi": inv_kappa,
        },
        coeff_rtol=0.25,
        coeff_atol=0.020,
        decoy_atol=0.035,
        rms_tol=0.015,
    )


__all__ = [
    "DEFAULT_INTERIOR_PAD",
    "GroundTruth",
    "MONDProblem",
    "MondDataset",
    "PROBLEMS",
    "generate_dataset",
    "ground_truth_for_problem",
    "load_problems",
]
