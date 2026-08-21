#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

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

from kepler_demo_utils import load_kepler_datasets_from_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_symbolic_summary_path() -> Path:
    return _repo_root() / "results" / "kepler_ephemeris_real_weathered_classsr" / "symbolic_kepler_summary.json"


def _default_direct_summary_path() -> Path:
    return _repo_root() / "results" / "kepler_ephemeris_real_weathered" / "kepler_ephemeris_weathered_summary.json"


def _default_manifest_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _default_output_dir() -> Path:
    return _repo_root() / "results" / "kepler_ephemeris_real_weathered_classsr" / "figures"


def _load_json(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, output_stem: Path, *, show: bool) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _dataset_maps(data_dir: Path):
    datasets = load_kepler_datasets_from_manifest(data_dir)
    by_id = {dataset.orbit_id: dataset for dataset in datasets}
    return datasets, by_id


def _split_style(split: str) -> dict[str, object]:
    styles = {
        "train": {"linestyle": "-", "marker": "o"},
        "validation": {"linestyle": "--", "marker": "s"},
        "holdout": {"linestyle": "-.", "marker": "^"},
    }
    return dict(styles.get(str(split), {"linestyle": "-", "marker": "o"}))


def _eccentricity_color_meta(datasets) -> tuple[Normalize, object]:
    ecc = np.asarray([float(dataset.e) for dataset in datasets], dtype=np.float64)
    norm = Normalize(vmin=float(np.min(ecc)), vmax=float(np.max(ecc)))
    cmap = plt.get_cmap("cividis")
    return norm, cmap


def _plot_areal_law_family(
    symbolic_summary: dict[str, object],
    datasets_by_id: dict[str, object],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    rows = list(symbolic_summary["merged_rows"])
    datasets = [datasets_by_id[str(row["dataset_id"])] for row in rows]
    norm, cmap = _eccentricity_color_meta(datasets)

    fig, ax = plt.subplots(figsize=(9.1, 6.6))
    for row in rows:
        dataset = datasets_by_id[str(row["dataset_id"])]
        color = cmap(norm(float(dataset.e)))
        style = _split_style(dataset.split)
        x = 1.0 / np.square(np.asarray(dataset.r, dtype=np.float64))
        y = np.asarray(dataset.omega, dtype=np.float64)
        order = np.argsort(x)
        x_sorted = x[order]
        y_fit = float(row["omega_intercept"]) + float(row["ell"]) * x_sorted
        ax.scatter(
            x,
            y,
            s=10,
            color=color,
            alpha=0.18 if dataset.split == "train" else 0.28,
            edgecolors="none",
            zorder=1,
        )
        ax.plot(
            x_sorted,
            y_fit,
            color=color,
            lw=2.6 if dataset.split == "holdout" else 1.4,
            ls=str(style["linestyle"]),
            zorder=2,
        )
        if dataset.split == "holdout":
            ax.annotate(
                rf"$e={dataset.e:.2f}$",
                (float(x_sorted[-1]), float(y_fit[-1])),
                xytext=(6, 2),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )

    ax.set_xlabel(r"$r^{-2}$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_title("Recovered Areal-Law Family")
    ax.grid(alpha=0.25, lw=0.6)

    handles = []
    for split in ("train", "validation", "holdout"):
        style = _split_style(split)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#333333",
                lw=2.0,
                ls=str(style["linestyle"]),
                marker=str(style["marker"]),
                markersize=6,
                label=split,
            )
        )
    ax.legend(handles=handles, loc="upper left", frameon=True, title="Orbit split")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Eccentricity $e$")

    _save_figure(fig, output_dir / "areal_law_family", show=show)


def _plot_radial_family_and_selection(
    symbolic_summary: dict[str, object],
    direct_summary: dict[str, object] | None,
    datasets_by_id: dict[str, object],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    rows = list(symbolic_summary["merged_rows"])
    datasets = [datasets_by_id[str(row["dataset_id"])] for row in rows]
    norm, cmap = _eccentricity_color_meta(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))

    holdout_rows = [row for row in rows if str(row["split"]) == "holdout"]
    for row in holdout_rows:
        dataset = datasets_by_id[str(row["dataset_id"])]
        color = cmap(norm(float(dataset.e)))
        r = np.asarray(dataset.r, dtype=np.float64)
        y = np.asarray(dataset.rddot, dtype=np.float64)
        order = np.argsort(r)
        r_sorted = r[order]
        y_fit = (
            float(row["rddot_intercept"])
            + float(row["k"]) / np.power(r_sorted, 3)
            + float(row["minus_mu"]) / np.power(r_sorted, 2)
        )
        axes[0].scatter(
            r,
            y,
            s=12,
            color=color,
            alpha=0.24,
            edgecolors="none",
            zorder=1,
        )
        axes[0].plot(
            r_sorted,
            y_fit,
            color=color,
            lw=2.5,
            label=rf"$e={dataset.e:.2f}$, $\Lambda={dataset.dynamic_range:.1f}$",
            zorder=2,
        )

    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$\ddot{r}$")
    axes[0].set_title("High-E Hold-Outs in the Recovered Radial Family")
    axes[0].grid(alpha=0.25, lw=0.6)
    if holdout_rows:
        axes[0].legend(loc="best", frameon=True)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No hold-out rows\navailable in symbolic summary",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[0].transAxes,
        )

    if direct_summary is not None:
        scan_rows = sorted(list(direct_summary["power_scan"]["rows"]), key=lambda row: float(row["exponent"]))
        exponents = np.asarray([float(row["exponent"]) for row in scan_rows], dtype=np.float64)
        train_rmse = np.asarray([float(row["train_mean_rmse"]) for row in scan_rows], dtype=np.float64)
        holdout_rmse = np.asarray([float(row["holdout_mean_rmse"]) for row in scan_rows], dtype=np.float64)
        axes[1].semilogy(exponents, train_rmse, color="#4c78a8", lw=2.0, label="Low-e training")
        axes[1].semilogy(exponents, holdout_rmse, color="#c23b22", lw=2.0, label="High-e hold-out")
        axes[1].axvline(2.0, color="#222222", lw=1.5, ls="--")
        axes[1].annotate(
            r"$p=2$",
            (2.0, float(np.min(holdout_rmse))),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=10,
            color="#222222",
        )
        axes[1].set_xlabel(r"Candidate force exponent $p$")
        axes[1].set_ylabel("Mean RMSE")
        axes[1].set_title("High-E Hold-Outs Lock In the Inverse-Square Law")
        axes[1].grid(alpha=0.25, lw=0.6)
        axes[1].legend(loc="best", frameon=True)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Direct-fit summary\nnot available",
            ha="center",
            va="center",
            fontsize=13,
            transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()

    _save_figure(fig, output_dir / "radial_family_selection", show=show)


def _plot_coefficient_manifold(
    symbolic_summary: dict[str, object],
    datasets_by_id: dict[str, object],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    rows = list(symbolic_summary["merged_rows"])
    datasets = [datasets_by_id[str(row["dataset_id"])] for row in rows]
    norm, cmap = _eccentricity_color_meta(datasets)
    lift = dict(symbolic_summary["symbolic_summary"]["coefficient_lift"])

    ell = np.asarray([float(row["ell"]) for row in rows], dtype=np.float64)
    ell_sq = np.square(ell)

    fig, ax = plt.subplots(figsize=(8.6, 6.7))
    x_line = np.linspace(0.0, float(np.max(ell_sq)) * 1.08, 400, dtype=np.float64)
    ax.plot(x_line, x_line, color="#222222", lw=2.2, label=r"$k = \ell^2$")
    ax.plot(
        x_line,
        float(lift["intercept"]) + float(lift["slope"]) * x_line,
        color="#8c564b",
        lw=2.0,
        ls="--",
        label=rf"Recovered fit $k \approx {lift['intercept']:+.2e} + {lift['slope']:.4f}\,\ell^2$",
    )

    for row in rows:
        dataset = datasets_by_id[str(row["dataset_id"])]
        style = _split_style(dataset.split)
        ax.scatter(
            float(row["ell"]) ** 2,
            float(row["k"]),
            s=86,
            marker=str(style["marker"]),
            color=cmap(norm(float(dataset.e))),
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
        )
        if dataset.split == "holdout":
            ax.annotate(
                rf"$e={dataset.e:.2f}$",
                (float(row["ell"]) ** 2, float(row["k"])),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=9,
                color="#1f1f1f",
            )

    ax.set_xlabel(r"$\ell^2$")
    ax.set_ylabel(r"$k$")
    ax.set_title("Recovered Coefficient Manifold")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", frameon=True)

    handles = []
    for split in ("train", "validation", "holdout"):
        style = _split_style(split)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#333333",
                marker=str(style["marker"]),
                linestyle="None",
                markersize=7,
                label=split,
            )
        )
    leg2 = ax.legend(handles=handles, loc="lower right", frameon=True, title="Orbit split")
    ax.add_artist(leg2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Eccentricity $e$")

    _save_figure(fig, output_dir / "coefficient_manifold", show=show)


def _plot_energy_invariant(
    symbolic_summary: dict[str, object],
    datasets_by_id: dict[str, object],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    energy = dict(symbolic_summary["symbolic_summary"]["energy"])
    coeffs = dict(energy["coeffs"])
    expected = dict(energy["expected_coeffs"])
    per_dataset = list(energy["per_dataset"])
    datasets = [datasets_by_id[str(row["orbit_id"])] for row in per_dataset]
    norm, cmap = _eccentricity_color_meta(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7))

    labels = [r"$\dot{r}^2$", r"$\ell^2/r^2$", r"$\mu/r$"]
    recovered_vals = [
        float(coeffs["rdot_sq"]),
        float(coeffs["ell_sq_over_r_sq"]),
        float(coeffs["mu_over_r"]),
    ]
    expected_vals = [
        float(expected["rdot_sq"]),
        float(expected["ell_sq_over_r_sq"]),
        float(expected["mu_over_r"]),
    ]
    xpos = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    axes[0].bar(xpos - width / 2.0, expected_vals, width=width, color="#c9c9c9", label="Expected")
    axes[0].bar(xpos + width / 2.0, recovered_vals, width=width, color="#4c78a8", label="Recovered")
    axes[0].set_xticks(xpos, labels)
    axes[0].axhline(0.0, color="#333333", lw=0.8, alpha=0.5)
    axes[0].set_title("Recovered Energy Coefficients")
    axes[0].grid(alpha=0.2, lw=0.6, axis="y")
    axes[0].legend(loc="best", frameon=True)

    true_energy = np.asarray([float(row["energy_true"]) for row in per_dataset], dtype=np.float64)
    fit_energy = np.asarray([float(row["energy_fit"]) for row in per_dataset], dtype=np.float64)
    lo = float(min(np.min(true_energy), np.min(fit_energy))) * 1.05
    hi = float(max(np.max(true_energy), np.max(fit_energy))) * 0.95
    diag = np.linspace(lo, hi, 200, dtype=np.float64)
    axes[1].plot(diag, diag, color="#222222", lw=2.0, label=r"$E_{\mathrm{fit}} = E_{\mathrm{true}}$")
    for row in per_dataset:
        dataset = datasets_by_id[str(row["orbit_id"])]
        style = _split_style(dataset.split)
        axes[1].scatter(
            float(row["energy_true"]),
            float(row["energy_fit"]),
            s=84,
            marker=str(style["marker"]),
            color=cmap(norm(float(dataset.e))),
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
        )
    axes[1].set_xlabel(r"True $E_d$")
    axes[1].set_ylabel(r"Recovered $E_d$")
    axes[1].set_title("Recovered Specific-Energy Levels")
    axes[1].grid(alpha=0.25, lw=0.6)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], pad=0.02)
    cbar.set_label(r"Eccentricity $e$")

    fig.suptitle(
        "Reduced-Energy Post-Pass\n"
        rf"max coefficient error = {float(energy['coeff_max_abs_error']):.2e}",
        y=0.98,
    )

    _save_figure(fig, output_dir / "energy_invariant", show=show)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing figures for the ephemeris-backed Kepler showcase",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbolic_summary",
        type=str,
        default=str(_default_symbolic_summary_path()),
        help="Path to the symbolic Kepler summary JSON",
    )
    parser.add_argument(
        "--direct_summary",
        type=str,
        default=str(_default_direct_summary_path()),
        help="Path to the direct-fit Kepler summary JSON",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(_default_manifest_dir()),
        help="Directory containing the Kepler data manifest",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_default_output_dir()),
        help="Directory where the figures will be written",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively as well as saving them")
    args = parser.parse_args()

    symbolic_summary = _load_json(args.symbolic_summary)
    if symbolic_summary is None:
        raise FileNotFoundError(f"missing symbolic summary: {args.symbolic_summary}")
    if str(symbolic_summary.get("status")) != "extractable":
        raise ValueError(
            f"symbolic summary is not extractable: status={symbolic_summary.get('status')!r}"
        )
    direct_summary = _load_json(args.direct_summary)

    datasets, datasets_by_id = _dataset_maps(Path(args.data_dir))
    output_dir = Path(args.output_dir).resolve()

    _plot_areal_law_family(symbolic_summary, datasets_by_id, output_dir, show=bool(args.show))
    _plot_radial_family_and_selection(symbolic_summary, direct_summary, datasets_by_id, output_dir, show=bool(args.show))
    _plot_coefficient_manifold(symbolic_summary, datasets_by_id, output_dir, show=bool(args.show))
    _plot_energy_invariant(symbolic_summary, datasets_by_id, output_dir, show=bool(args.show))

    print(f"Saved figures to {output_dir}")
    for stem in (
        "areal_law_family",
        "radial_family_selection",
        "coefficient_manifold",
        "energy_invariant",
    ):
        print(f"  - {output_dir / (stem + '.png')}")
        print(f"  - {output_dir / (stem + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
