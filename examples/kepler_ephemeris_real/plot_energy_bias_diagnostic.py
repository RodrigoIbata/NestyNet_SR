#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import json
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
from matplotlib.colors import Normalize
import numpy as np

from kepler_demo_utils import build_default_kepler_datasets


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_summary_path() -> Path:
    return (
        _repo_root()
        / "results"
        / "kepler_ephemeris_real_weathered_308_joint_1d"
        / "kepler_ephemeris_weathered_308_joint_summary.json"
    )


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset_map(raw_manifest: str) -> dict[str, object]:
    datasets = build_default_kepler_datasets(
        provider="raw_csv",
        profile="weathered",
        raw_manifest=raw_manifest,
    )
    return {str(dataset.orbit_id): dataset for dataset in datasets}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the reduced-energy bias collapse diagnostic for the 308-body weathered Kepler ensemble.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(_default_summary_path()),
        help="Path to the direct-fit Kepler summary JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path stem for the figure; defaults next to the summary JSON",
    )
    args = parser.parse_args()

    summary = _load_json(args.summary)
    raw_manifest = str(summary["raw_manifest"])
    datasets_by_id = _dataset_map(raw_manifest)

    energy = dict(summary["energy"])
    coeffs = dict(energy["coeffs"])
    expected = dict(energy["expected_coeffs"])
    rows = list(energy["per_dataset"])

    orbit_ids = [str(row["orbit_id"]) for row in rows]
    e = np.asarray([float(datasets_by_id[orbit_id].e) for orbit_id in orbit_ids], dtype=np.float64)
    a = np.asarray([float(datasets_by_id[orbit_id].a) for orbit_id in orbit_ids], dtype=np.float64)
    x = np.sqrt(np.clip(1.0 - e**2, 0.0, 1.0))

    energy_true = np.asarray([float(row["energy_true"]) for row in rows], dtype=np.float64)
    energy_fit = np.asarray([float(row["energy_fit"]) for row in rows], dtype=np.float64)
    delta_e = energy_fit - energy_true
    y = delta_e / (-2.0 * energy_true)

    delta_rdot = float(coeffs["rdot_sq"]) - float(expected["rdot_sq"])
    delta_ell = float(coeffs["ell_sq_over_r_sq"]) - float(expected["ell_sq_over_r_sq"])
    delta_mu = float(coeffs["mu_over_r"]) - float(expected["mu_over_r"])
    slope, intercept = np.polyfit(x, y, deg=1)
    y_model = slope * x + intercept
    y_resid = y - y_model
    corr_xy = float(np.corrcoef(x, y)[0, 1])
    outer_mask = a > 2.8
    inner_mask = a < 2.5
    inner_resid = float(np.std(y_resid[inner_mask])) if np.any(inner_mask) else float("nan")
    outer_resid = float(np.std(y_resid[outer_mask])) if np.any(outer_mask) else float("nan")

    order = np.argsort(x)
    x_line = x[order]
    y_line = y_model[order]

    norm = Normalize(vmin=float(np.min(a)), vmax=float(np.max(a)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    scatter = ax.scatter(
        x,
        y,
        c=a,
        cmap=cmap,
        norm=norm,
        s=28,
        alpha=0.82,
        edgecolors="none",
        label="308 weathered asteroids",
        zorder=2,
    )
    ax.plot(
        x_line,
        y_line,
        color="#1f1f1f",
        lw=2.0,
        label="Empirical linear ridge",
        zorder=3,
    )
    ax.set_xlabel(r"$\sqrt{1-e^2}$")
    ax.set_ylabel(r"$\Delta E/(-2E_{\mathrm{true}})$")
    ax.set_title("Energy-Bias Collapse Diagnostic")
    ax.grid(alpha=0.22, lw=0.6)
    ax.legend(loc="best", frameon=True, fontsize=8.2)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Semi-major axis $a$ [AU]")

    ax.text(
        0.02,
        0.03,
        (
            rf"$\delta c_{{\dot r^2}}={delta_rdot:.2e}$"
            "\n"
            rf"$\delta c_{{\ell^2/r^2}}={delta_ell:.2e}$"
            "\n"
            rf"$\delta c_{{\mu/r}}={delta_mu:.2e}$"
            "\n"
            rf"$m={slope:.2e},\, b={intercept:.2e}$"
            "\n"
            rf"$\sigma_{{\rm resid}}={np.std(y_resid):.2e}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.2},
    )

    if args.output is None:
        output_stem = Path(args.summary).resolve().with_name("energy_bias_collapse_diagnostic")
    else:
        output_stem = Path(args.output).resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    e_norm = Normalize(vmin=float(np.min(e)), vmax=float(np.max(e)))
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    scatter2 = ax2.scatter(
        a,
        y_resid,
        c=e,
        cmap=cmap,
        norm=e_norm,
        s=28,
        alpha=0.82,
        edgecolors="none",
        label="Residual about linear ridge",
        zorder=2,
    )
    ax2.axhline(0.0, color="#222222", lw=1.5, zorder=1)

    bin_edges = np.linspace(float(np.min(a)), float(np.max(a)), 10, dtype=np.float64)
    bin_centers: list[float] = []
    bin_means: list[float] = []
    bin_stds: list[float] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi < float(np.max(a)):
            mask = (a >= lo) & (a < hi)
        else:
            mask = (a >= lo) & (a <= hi)
        if int(np.count_nonzero(mask)) < 6:
            continue
        values = y_resid[mask]
        bin_centers.append(0.5 * (lo + hi))
        bin_means.append(float(np.mean(values)))
        bin_stds.append(float(np.std(values)))

    centers = np.asarray(bin_centers, dtype=np.float64)
    means = np.asarray(bin_means, dtype=np.float64)
    stds = np.asarray(bin_stds, dtype=np.float64)
    ax2.plot(centers, means, color="#111111", lw=2.0, label="Binned mean residual", zorder=3)
    ax2.fill_between(
        centers,
        means - stds,
        means + stds,
        color="#111111",
        alpha=0.14,
        label=r"$\pm 1\sigma$ by $a$ bin",
        zorder=1,
    )

    ax2.set_xlabel(r"Semi-major axis $a$ [AU]")
    ax2.set_ylabel(r"Residual in $\Delta E/(-2E_{\mathrm{true}})$")
    ax2.set_title("Residual About Energy-Bias Ridge vs Semi-Major Axis")
    ax2.grid(alpha=0.22, lw=0.6)
    ax2.legend(loc="best", frameon=True, fontsize=8.2)

    cbar2 = fig2.colorbar(scatter2, ax=ax2, pad=0.02)
    cbar2.set_label(r"Eccentricity $e$")

    ax2.text(
        0.02,
        0.03,
        (
            rf"$\sigma_{{\rm inner}}={inner_resid:.2e}$"
            "\n"
            rf"$\sigma_{{\rm outer}}={outer_resid:.2e}$"
            "\n"
            rf"$\sigma_{{\rm outer}}/\sigma_{{\rm inner}}={outer_resid / inner_resid:.2f}$"
        ),
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.2},
    )

    residual_output = output_stem.with_name(f"{output_stem.name}_residual_vs_a")
    fig2.savefig(residual_output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig2.savefig(residual_output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)

    print(f"saved {output_stem.with_suffix('.png')}")
    print(f"saved {residual_output.with_suffix('.png')}")
    print(f"corr(x, y)     = {corr_xy:.6f}")
    print(f"residual std all   = {np.std(y_resid):.6e}")
    print(f"residual std inner = {inner_resid:.6e}")
    print(f"residual std outer = {outer_resid:.6e}")


if __name__ == "__main__":
    main()
