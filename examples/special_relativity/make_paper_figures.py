#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "xdg-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from sr_demo_utils import (
    generate_operational_interval_dataset,
    lorentz_boost_matrix,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_summary_path() -> Path:
    return _repo_root() / "results" / "special_relativity_classsr" / "symbolic_interval_summary.json"


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "manifest.json"


def _default_output_dir() -> Path:
    return _repo_root() / "results" / "special_relativity_classsr" / "figures"


def _load_summary(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_manifest(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, output_stem: Path, *, show: bool) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _pick_reference_row(summary: dict[str, object]) -> dict[str, object]:
    rows = list(summary["merged_rows"])
    return max(rows, key=lambda row: abs(float(row["beta"])))


def _load_interval_points(manifest: dict[str, object] | None, regime_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    if manifest is None:
        return None
    for row in list(manifest.get("regimes", []) or []):
        if str(row.get("regime_id", "")) != str(regime_id):
            continue
        combined_csv = Path(str(row["combined_csv"]))
        if not combined_csv.exists():
            continue
        df = pd.read_csv(combined_csv)
        unprimed = df[["u", "x"]].to_numpy(dtype=np.float64, copy=True)
        primed = df[["u_prime", "x_prime"]].to_numpy(dtype=np.float64, copy=True)
        return unprimed, primed
    return None


def _sample_metric_level_set(
    metric: np.ndarray,
    level: float,
    *,
    x_extent: float,
    n_points: int = 900,
) -> list[np.ndarray]:
    g00 = float(metric[0, 0])
    g01 = float(metric[0, 1])
    g11 = float(metric[1, 1])
    if abs(g00) < 1.0e-12:
        raise ValueError("metric has degenerate g00; cannot parametrize level set")

    x_vals = np.linspace(-float(x_extent), float(x_extent), int(n_points), dtype=np.float64)
    aa = g00 * np.ones_like(x_vals)
    bb = 2.0 * g01 * x_vals
    cc = g11 * np.square(x_vals) - float(level)
    disc = np.square(bb) - 4.0 * aa * cc
    mask = disc >= 0.0
    if not np.any(mask):
        return []

    x_ok = x_vals[mask]
    sqrt_disc = np.sqrt(np.maximum(disc[mask], 0.0))
    denom = 2.0 * aa[mask]
    branches = []
    for sign in (+1.0, -1.0):
        u_vals = (-bb[mask] + sign * sqrt_disc) / denom
        pts = np.column_stack([u_vals, x_ok])
        order = np.argsort(pts[:, 1])
        branches.append(pts[order])
    return branches


def _plot_coefficient_manifold(summary: dict[str, object], output_dir: Path, *, show: bool) -> None:
    rows = list(summary["merged_rows"])
    betas = np.asarray([float(row["beta"]) for row in rows], dtype=np.float64)
    a = np.asarray([float(row["a"]) for row in rows], dtype=np.float64)
    b = np.asarray([float(row["b"]) for row in rows], dtype=np.float64)
    c = np.asarray([float(row["c"]) for row in rows], dtype=np.float64)
    d = np.asarray([float(row["d"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 7.1))
    beta_curve = np.linspace(-0.92, 0.92, 1000, dtype=np.float64)
    gamma_curve = 1.0 / np.sqrt(1.0 - np.square(beta_curve))
    branch_curve = -beta_curve * gamma_curve

    ax.plot(gamma_curve, branch_curve, color="#222222", lw=2.2, label=r"Lorentz family $a^2-b^2=1$")
    ax.plot(gamma_curve, -branch_curve, color="#bbbbbb", lw=1.5, ls="--")

    norm = plt.Normalize(vmin=float(np.min(betas)), vmax=float(np.max(betas)))
    cmap = plt.get_cmap("coolwarm")
    scatter_u = ax.scatter(
        a,
        b,
        c=betas,
        cmap=cmap,
        norm=norm,
        s=88,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        label=r"Recovered $(a,b)$ from $u'$",
        zorder=3,
    )
    ax.scatter(
        d,
        c,
        c=betas,
        cmap=cmap,
        norm=norm,
        s=88,
        marker="s",
        edgecolors="black",
        linewidths=0.5,
        label=r"Recovered $(d,c)$ from $x'$",
        zorder=3,
    )
    ax.scatter([1.0], [0.0], s=150, marker="x", color="#222222", linewidths=2.5, label="Galilean anchor")

    for row in rows:
        beta = float(row["beta"])
        label = f"{beta:+.1f}"
        ax.annotate(
            label,
            (float(row["a"]), float(row["b"])),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color="#1f1f1f",
        )

    ax.set_xlabel(r"$a$ or $d$")
    ax.set_ylabel(r"$b$ or $c$")
    ax.set_title("Recovered Boost Coefficients on the Lorentz Hyperbola")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", frameon=True)
    cbar = fig.colorbar(scatter_u, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$\beta$")

    x_max = max(1.35, float(np.max(np.abs(np.concatenate([a, d])))) * 1.15)
    y_max = max(0.9, float(np.max(np.abs(np.concatenate([b, c])))) * 1.2)
    ax.set_xlim(0.85, x_max)
    ax.set_ylim(-y_max, y_max)

    _save_figure(fig, output_dir / "coefficient_manifold", show=show)


def _plot_interval_geometry(
    summary: dict[str, object],
    manifest: dict[str, object] | None,
    output_dir: Path,
    *,
    show: bool,
) -> None:
    row = _pick_reference_row(summary)
    beta = float(row["beta"])
    regime_id = str(row["dataset_id"])
    matrix = np.asarray(row["matrix"], dtype=np.float64)
    metric = np.asarray(summary["symbolic_summary"]["metric"]["metric"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), sharex=True, sharey=True)
    colors = {
        "null": "#e0a100",
        "timelike": "#1f77b4",
        "spacelike": "#c23b22",
    }

    x_extent = 10.5
    null_branches = _sample_metric_level_set(metric, 0.0, x_extent=x_extent)
    timelike_levels = [16.0, 49.0]
    spacelike_levels = [-16.0, -49.0]
    clouds = _load_interval_points(manifest, regime_id)

    for ax_idx, ax in enumerate(axes):
        transform = np.eye(2, dtype=np.float64) if ax_idx == 0 else matrix
        if clouds is not None:
            points = clouds[0] if ax_idx == 0 else clouds[1]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=5,
                color="#888888",
                alpha=0.18,
                edgecolors="none",
                zorder=1,
            )
        for pts in null_branches:
            pts_t = (transform @ pts.T).T
            ax.plot(pts_t[:, 0], pts_t[:, 1], color=colors["null"], lw=2.2, zorder=2)
        for level in timelike_levels:
            for pts in _sample_metric_level_set(metric, level, x_extent=x_extent):
                pts_t = (transform @ pts.T).T
                ax.plot(pts_t[:, 0], pts_t[:, 1], color=colors["timelike"], lw=1.8, zorder=2)
        for level in spacelike_levels:
            for pts in _sample_metric_level_set(metric, level, x_extent=x_extent):
                pts_t = (transform @ pts.T).T
                ax.plot(pts_t[:, 0], pts_t[:, 1], color=colors["spacelike"], lw=1.8, ls="--", zorder=2)

        ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.5)
        ax.axvline(0.0, color="#333333", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_aspect("equal", adjustable="box")
        if ax_idx == 0:
            ax.set_title(r"Frame $S$")
            ax.set_xlabel(r"$u = c\,\Delta t$")
            ax.set_ylabel(r"$x = \Delta x$")
        else:
            ax.set_title(rf"Frame $S'$ at $\beta={beta:+.2f}$")
            ax.set_xlabel(r"$u'$")

    coeffs = summary["symbolic_summary"]["metric"]["quadratic_coeffs"]
    fig.suptitle(
        "Recovered Interval Geometry\n"
        rf"$Q \approx {coeffs['u2']:.3f}\,u^2 + {coeffs['ux']:.3f}\,ux + {coeffs['x2']:.3f}\,x^2$",
        y=0.98,
    )
    axes[0].set_xlim(-12.0, 12.0)
    axes[0].set_ylim(-12.0, 12.0)

    _save_figure(fig, output_dir / "interval_geometry", show=show)


def _family_prediction_mse(dataset, family: str) -> float:
    beta = float(dataset.beta)
    u = np.asarray(dataset.u, dtype=np.float64)
    x = np.asarray(dataset.x, dtype=np.float64)
    up = np.asarray(dataset.u_prime, dtype=np.float64)
    xp = np.asarray(dataset.x_prime, dtype=np.float64)

    if family == "lorentz":
        mat = lorentz_boost_matrix(beta)
    elif family == "galilean":
        mat = np.asarray([[1.0, 0.0], [-beta, 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"unknown family {family!r}")

    up_pred = float(mat[0, 0]) * u + float(mat[0, 1]) * x
    xp_pred = float(mat[1, 0]) * u + float(mat[1, 1]) * x
    return float(np.mean(np.square(up - up_pred) + np.square(xp - xp_pred)))


def _compute_phase_diagram(
    *,
    beta_max_values: np.ndarray,
    noise_values: np.ndarray,
    n_samples: int,
    repeats: int,
    simplicity_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    preference = np.zeros((len(noise_values), len(beta_max_values)), dtype=np.float64)
    ratio = np.zeros_like(preference)
    eps = 1.0e-18

    for j, noise_std in enumerate(noise_values):
        for i, beta_max in enumerate(beta_max_values):
            gal_errs = []
            lor_errs = []
            for rep in range(int(repeats)):
                betas = (-float(beta_max), 0.0, float(beta_max))
                for k, beta in enumerate(betas):
                    dataset = generate_operational_interval_dataset(
                        beta,
                        n_samples=int(n_samples),
                        seed=1000 + 97 * rep + 11 * i + 3 * j + k,
                        noise_std=float(noise_std),
                    )
                    gal_errs.append(_family_prediction_mse(dataset, "galilean"))
                    lor_errs.append(_family_prediction_mse(dataset, "lorentz"))
            gal_mean = float(np.mean(gal_errs))
            lor_mean = float(np.mean(lor_errs))
            ratio[j, i] = gal_mean / max(lor_mean, eps)
            preference[j, i] = math.log10((gal_mean + eps) / (lor_mean + eps)) - math.log10(float(simplicity_prior))
    return preference, ratio


def _log_edges(values: np.ndarray) -> np.ndarray:
    log_values = np.log10(np.asarray(values, dtype=np.float64))
    midpoints = 0.5 * (log_values[:-1] + log_values[1:])
    edges = np.empty(len(values) + 1, dtype=np.float64)
    edges[1:-1] = midpoints
    edges[0] = log_values[0] - 0.5 * (log_values[1] - log_values[0])
    edges[-1] = log_values[-1] + 0.5 * (log_values[-1] - log_values[-2])
    return np.power(10.0, edges)


def _plot_theory_phase_diagram(output_dir: Path, *, show: bool) -> None:
    beta_max_values = np.geomspace(1.0e-2, 0.50, 12, dtype=np.float64)
    noise_values = np.geomspace(1.0e-3, 1.0, 12, dtype=np.float64)
    simplicity_prior = 10.0
    preference, _ = _compute_phase_diagram(
        beta_max_values=beta_max_values,
        noise_values=noise_values,
        n_samples=192,
        repeats=4,
        simplicity_prior=float(simplicity_prior),
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.7))
    color_limit = 3
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    im = ax.pcolormesh(
        _log_edges(beta_max_values),
        _log_edges(noise_values),
        preference,
        cmap="coolwarm_r",
        norm=norm,
        shading="auto",
    )
    ax.contour(
        beta_max_values,
        noise_values,
        preference,
        levels=[0.0],
        colors=["black"],
        linewidths=1.8,
    )
    ax.set_xlim(float(beta_max_values[0]), float(beta_max_values[-1]))
    ax.set_ylim(float(noise_values[0]), float(noise_values[-1]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Max sampled $|\beta|$")
    ax.set_ylabel("Observation noise std")
    ax.set_title("Lorentz-vs-Galilean Preference\n(10x simplicity prior for Galilean)")

    ax.text(0.25, 2.0e-3, "Lorentz preferred", color="white", fontsize=15, ha="center", va="center")
    ax.text(0.025, 2.5e-1, "Galilean\npreferred", color="white", fontsize=15, ha="center", va="center")

    cbar_ticks = np.arange(-color_limit, color_limit + 1, 1, dtype=int)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, ticks=cbar_ticks)
    cbar.set_label(r"$\log_{10}(\mathrm{err}_{\mathrm{Gal}}/\mathrm{err}_{\mathrm{Lor}}) - \log_{10}(10)$")

    _save_figure(fig, output_dir / "theory_phase_diagram", show=show)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing figures for the special-relativity discovery demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(_default_summary_path()),
        help="Path to symbolic_interval_summary.json",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(_default_manifest_path()),
        help="Path to the interval manifest.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_default_output_dir()),
        help="Directory for generated figures",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    args = parser.parse_args()

    summary = _load_summary(args.summary)
    manifest = _load_manifest(args.manifest)
    output_dir = Path(args.output_dir).resolve()

    _plot_coefficient_manifold(summary, output_dir, show=bool(args.show))
    _plot_interval_geometry(summary, manifest, output_dir, show=bool(args.show))
    _plot_theory_phase_diagram(output_dir, show=bool(args.show))

    print(f"Wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
