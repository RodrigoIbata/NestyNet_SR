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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from kepler_demo_utils import (
    _jsonable,
    build_default_kepler_datasets,
    evaluate_radial_family_with_fixed_mu,
    fit_radial_family,
    lift_coefficient_relation,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_direct_summary_path() -> Path:
    return _repo_root() / "results" / "kepler_ephemeris_real_weathered_308_joint_1d" / "kepler_ephemeris_weathered_308_joint_summary.json"


def _default_holdout_summary_path() -> Path:
    return _repo_root() / "results" / "kepler_ephemeris_real_weathered_holdout_full" / "holdout_generalization_summary.json"


def _default_selection_summary_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "selection_jpl_ssodnet_mass_gt_1e17_arc15000_summary.json"


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, output_stem: Path, *, show: bool) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _infer_raw_manifest(direct_summary: dict[str, object], raw_manifest: str | None) -> Path:
    if raw_manifest is not None:
        return Path(raw_manifest).resolve()
    summary_raw = direct_summary.get("raw_manifest", None)
    if summary_raw is None:
        raise ValueError("raw manifest was not provided and could not be inferred from the direct summary")
    return Path(str(summary_raw)).resolve()


def _load_body_names(raw_manifest_path: Path) -> dict[str, str]:
    rows = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    return {
        str(row["orbit_id"]): str(row.get("body_name", row["orbit_id"]))
        for row in list(rows)
        if isinstance(row, dict) and "orbit_id" in row
    }


def _display_body_name(name: str) -> str:
    text = str(name).strip()
    parts = text.split(maxsplit=1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return text


def _latex_sci(value: float, precision: int = 1) -> str:
    mantissa, exponent = f"{float(value):.{int(precision)}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _dataset_bundle(
    raw_manifest_path: Path,
    *,
    accel_source: str = "gradient",
    accel_cache_dir: str | None = None,
):
    datasets = build_default_kepler_datasets(
        provider="raw_csv",
        profile="weathered",
        raw_manifest=str(raw_manifest_path),
        accel_source=str(accel_source),
        accel_certificate=accel_cache_dir is not None,
        accel_cache_dir=accel_cache_dir,
    )
    datasets = sorted(datasets, key=lambda ds: (float(ds.dynamic_range), ds.orbit_id))
    by_id = {dataset.orbit_id: dataset for dataset in datasets}
    return datasets, by_id


def _round_robin_direct_holdout_splits(datasets):
    ordered = sorted(list(datasets), key=lambda ds: (float(ds.dynamic_range), ds.orbit_id))
    holdout = [dataset for idx, dataset in enumerate(ordered) if idx % 10 == 0]
    validation = [dataset for idx, dataset in enumerate(ordered) if idx % 10 == 1]
    train = [dataset for idx, dataset in enumerate(ordered) if idx % 10 not in (0, 1)]
    trainval = [dataset for idx, dataset in enumerate(ordered) if idx % 10 != 0]
    return {
        "ordered": ordered,
        "train": train,
        "validation": validation,
        "trainval": trainval,
        "holdout": holdout,
    }


def _eccentricity_color_meta(datasets) -> tuple[Normalize, object]:
    ecc = np.asarray([float(dataset.e) for dataset in datasets], dtype=np.float64)
    norm = Normalize(vmin=float(np.min(ecc)), vmax=float(np.max(ecc)))
    cmap = plt.get_cmap("viridis")
    return norm, cmap


def _dynamic_range_color_meta(datasets) -> tuple[Normalize, object]:
    leverage = np.asarray([float(dataset.dynamic_range) for dataset in datasets], dtype=np.float64)
    norm = Normalize(vmin=float(np.min(leverage)), vmax=float(np.max(leverage)))
    cmap = plt.get_cmap("viridis")
    return norm, cmap


def _representative_ids_by_quantile(datasets, *, quantiles: list[float]) -> list[str]:
    if not datasets:
        return []
    ordered = sorted(datasets, key=lambda ds: (float(ds.dynamic_range), ds.orbit_id))
    out: list[str] = []
    seen: set[str] = set()
    for quantile in quantiles:
        idx = int(round(float(quantile) * float(len(ordered) - 1)))
        orbit_id = str(ordered[idx].orbit_id)
        if orbit_id in seen:
            continue
        seen.add(orbit_id)
        out.append(orbit_id)
    return out


def _top_dynamic_range_ids(datasets, *, count: int) -> list[str]:
    ordered = sorted(datasets, key=lambda ds: (-float(ds.dynamic_range), ds.orbit_id))
    return [str(dataset.orbit_id) for dataset in ordered[: int(count)]]


def _sample_stride(values: np.ndarray, target_points: int) -> int:
    if int(values.shape[0]) <= int(target_points):
        return 1
    return max(1, int(values.shape[0]) // int(target_points))


def _stage_row_map(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["orbit_id"]): dict(row) for row in list(rows)}


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        str(label),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.6},
        zorder=10,
    )


def _per_object_cleanliness(
    datasets,
    direct_summary: dict[str, object],
) -> list[dict[str, float | str]]:
    by_id = {dataset.orbit_id: dataset for dataset in datasets}
    rows = []
    for row in list(direct_summary["stage_b_all"]["per_dataset"]):
        orbit_id = str(row["orbit_id"])
        dataset = by_id[orbit_id]
        radial_rmse = float(row["rmse"])
        radial_scale = float(np.sqrt(np.mean(np.square(np.asarray(dataset.rddot, dtype=np.float64)))))
        radial_rel = float(radial_rmse / max(radial_scale, 1.0e-15))
        rows.append(
            {
                "orbit_id": orbit_id,
                "dynamic_range": float(dataset.dynamic_range),
                "eccentricity": float(dataset.e),
                "radial_rel_rmse": radial_rel,
            }
        )
    rows.sort(key=lambda item: (float(item["radial_rel_rmse"]), -float(item["dynamic_range"]), str(item["orbit_id"])))
    return rows


def _plot_areal_law_family(
    direct_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    stage_a_rows = _stage_row_map(list(direct_summary["stage_a"]["per_dataset"]))
    reps = _top_dynamic_range_ids(datasets, count=3)
    norm, cmap = _eccentricity_color_meta(datasets)

    fig = plt.figure(figsize=(8.8, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.075])
    ax = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])
    ax.set_box_aspect(1.0)
    ax.tick_params(labelsize=8.5)
    for dataset in datasets:
        x = 1.0 / np.square(np.asarray(dataset.r, dtype=np.float64))
        y = np.asarray(dataset.omega, dtype=np.float64)
        stride = _sample_stride(x, 220)
        ax.scatter(
            x[::stride],
            y[::stride],
            s=4,
            color=cmap(norm(float(dataset.e))),
            alpha=0.045,
            edgecolors="none",
            zorder=1,
        )
    for orbit_id in reps:
        dataset = datasets_by_id[orbit_id]
        row = stage_a_rows[orbit_id]
        color = cmap(norm(float(dataset.e)))
        x = 1.0 / np.square(np.asarray(dataset.r, dtype=np.float64))
        y = np.asarray(dataset.omega, dtype=np.float64)
        stride = _sample_stride(x, 260)
        order = np.argsort(x)
        x_sorted = x[order]
        ax.scatter(
            x[::stride],
            y[::stride],
            s=14,
            color=color,
            alpha=0.24,
            edgecolors="none",
            zorder=1,
        )
        x_line = np.linspace(float(np.min(x_sorted)), float(np.max(x_sorted)), 300, dtype=np.float64)
        y_fit = float(row["ell_fit"]) * x_line
        ax.plot(x_line, y_fit, color=color, lw=2.4, zorder=3)
        ax.annotate(
            _display_body_name(body_name_by_id.get(orbit_id, orbit_id)),
            (float(x_line[-1]), float(y_fit[-1])),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.0,
            color="#111111",
            zorder=9,
        )

    ax.set_xlabel(r"$r^{-2}$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_title("Areal-law family", fontsize=11)
    ax.grid(alpha=0.22, lw=0.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    ax_pos = ax.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax_pos.y0, cax_pos.width, ax_pos.height])
    cbar.set_label(r"Eccentricity $e$")
    cbar.ax.tick_params(labelsize=8.0)

    _save_figure(fig, output_dir / "areal_law_family", show=show)


def _plot_radial_family_and_selection(
    direct_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    stage_b_rows = _stage_row_map(list(direct_summary["stage_b_all"]["per_dataset"]))
    mu_fit = float(direct_summary["stage_b_all"]["mu"])
    reps = _top_dynamic_range_ids(datasets, count=6)
    rep_cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.5))

    for idx, orbit_id in enumerate(reps):
        dataset = datasets_by_id[orbit_id]
        row = stage_b_rows[orbit_id]
        color = rep_cmap(idx % 10)
        r = np.asarray(dataset.r, dtype=np.float64)
        y = np.asarray(dataset.rddot, dtype=np.float64)
        stride = _sample_stride(r, 280)
        order = np.argsort(r)
        r_sorted = r[order]
        y_fit = float(row["k_fit"]) / np.power(r_sorted, 3) - mu_fit / np.power(r_sorted, 2)
        axes[0].scatter(
            r[::stride],
            y[::stride],
            s=10,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )
        axes[0].plot(
            r_sorted,
            y_fit,
            color=color,
            lw=2.2,
            label=rf"{body_name_by_id.get(orbit_id, orbit_id)} ($\Lambda={dataset.dynamic_range:.2f}$)",
            zorder=2,
        )

    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$\ddot{r}$")
    axes[0].set_title("Radial family")
    axes[0].grid(alpha=0.22, lw=0.6)
    axes[0].legend(loc="best", frameon=True, fontsize=8)

    scan_rows = sorted(list(direct_summary["power_scan"]["rows"]), key=lambda row: float(row["exponent"]))
    exponents = np.asarray([float(row["exponent"]) for row in scan_rows], dtype=np.float64)
    train_rmse = np.asarray([float(row["train_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    holdout_rmse = np.asarray([float(row["holdout_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    axes[1].semilogy(exponents, train_rmse, color="#4c78a8", lw=2.0, label="Round-robin train")
    axes[1].semilogy(exponents, holdout_rmse, color="#c23b22", lw=2.0, label="Round-robin holdout")
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
    axes[1].set_ylabel(r"Mean RMSE of $\ddot{r}$ [$\mathrm{AU}\,\mathrm{day}^{-2}$]")
    axes[1].set_title("Exponent scan")
    axes[1].grid(alpha=0.22, lw=0.6)
    axes[1].legend(loc="best", frameon=True)

    _save_figure(fig, output_dir / "radial_family_selection", show=show)


def _plot_coefficient_manifold(
    direct_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    lift = dict(direct_summary["coefficient_lift"])
    rows = list(lift["per_dataset"])
    norm, cmap = _eccentricity_color_meta(datasets)
    reps = _top_dynamic_range_ids(datasets, count=5)

    fig, ax = plt.subplots(figsize=(8.7, 6.6))
    ell_sq = np.asarray([float(row["ell_sq"]) for row in rows], dtype=np.float64)

    x_line = np.linspace(0.0, float(np.max(ell_sq)) * 1.08, 400, dtype=np.float64)
    ax.plot(x_line, x_line, color="#222222", lw=2.0, label=r"$k_d = \ell_d^2$")
    ax.plot(
        x_line,
        float(lift["intercept"]) + float(lift["slope"]) * x_line,
        color="#8c564b",
        lw=2.0,
        ls="--",
        label=rf"Recovered fit $k_d \approx {_latex_sci(float(lift['intercept']), precision=2)} + {lift['slope']:.4f}\,\ell_d^2$",
    )

    for row in rows:
        dataset = datasets_by_id[str(row["orbit_id"])]
        ax.scatter(
            float(row["ell_sq"]),
            float(row["k"]),
            s=36,
            color=cmap(norm(float(dataset.e))),
            alpha=0.86,
            edgecolors="none",
            zorder=2,
        )

    for orbit_id in reps:
        row = next(item for item in rows if str(item["orbit_id"]) == orbit_id)
        dataset = datasets_by_id[orbit_id]
        ax.scatter(
            float(row["ell_sq"]),
            float(row["k"]),
            s=80,
            color=cmap(norm(float(dataset.e))),
            edgecolors="black",
            linewidths=0.55,
            zorder=3,
        )
        ax.annotate(
            str(body_name_by_id.get(orbit_id, orbit_id)),
            (float(row["ell_sq"]), float(row["k"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#1f1f1f",
        )

    ax.set_xlabel(r"$\ell_d^2$")
    ax.set_ylabel(r"$k_d$")
    ax.set_title("Recovered coefficient relation")
    ax.grid(alpha=0.22, lw=0.6)
    ax.legend(loc="upper left", frameon=True)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Eccentricity $e$")

    _save_figure(fig, output_dir / "coefficient_manifold", show=show)


def _plot_radial_family_triptych(
    direct_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    stage_b_rows = _stage_row_map(list(direct_summary["stage_b_all"]["per_dataset"]))
    mu_fit = float(direct_summary["stage_b_all"]["mu"])
    reps = _top_dynamic_range_ids(datasets, count=3)
    rep_cmap = plt.get_cmap("tab10")

    lift = dict(direct_summary["coefficient_lift"])
    coeff_rows = list(lift["per_dataset"])
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10.6, 3.9), constrained_layout=True)
    for ax in (ax0, ax1, ax2):
        ax.set_box_aspect(1.0)
        ax.tick_params(labelsize=8.5)

    for idx, orbit_id in enumerate(reps):
        dataset = datasets_by_id[orbit_id]
        row = stage_b_rows[orbit_id]
        color = rep_cmap(idx % 10)
        r = np.asarray(dataset.r, dtype=np.float64)
        y = np.asarray(dataset.rddot, dtype=np.float64)
        stride = _sample_stride(r, 260)
        order = np.argsort(r)
        r_sorted = r[order]
        y_fit = float(row["k_fit"]) / np.power(r_sorted, 3) - mu_fit / np.power(r_sorted, 2)
        ax0.scatter(
            r[::stride],
            y[::stride],
            s=9,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )
        ax0.plot(
            r_sorted,
            y_fit,
            color=color,
            lw=2.1,
            label=str(body_name_by_id.get(orbit_id, orbit_id)),
            zorder=2,
        )

    ax0.set_xlabel(r"$r\, \, \, [\mathrm{AU}]$")
    ax0.set_ylabel(r"$\ddot{r}\, \, \, [\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax0.set_title("Radial family", fontsize=11)
    ax0.grid(alpha=0.22, lw=0.6)
    ax0.legend(loc="upper right", frameon=True, fontsize=7.0)
    _add_panel_label(ax0, "(a)")

    scan_rows = sorted(list(direct_summary["power_scan"]["rows"]), key=lambda row: float(row["exponent"]))
    exponents = np.asarray([float(row["exponent"]) for row in scan_rows], dtype=np.float64)
    train_rmse = np.asarray([float(row["train_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    holdout_rmse = np.asarray([float(row["holdout_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    ax1.plot(exponents, train_rmse, color="#4c78a8", lw=2.0, label="Train (246)")
    ax1.plot(exponents, holdout_rmse, color="#c23b22", lw=2.0, ls="--", label="Holdout (31)")
    ax1.axvline(2.0, color="#222222", lw=1.5, ls="--")
    ax1.annotate(
        r"$p=2$",
        (2.0, float(np.min(holdout_rmse))),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
        color="#222222",
    )
    ax1.set_xlabel(r"Candidate force exponent $p$")
    ax1.set_ylabel(r"Mean RMSE of $\ddot{r}$ [$\mathrm{AU}\,\mathrm{day}^{-2}$]")
    ax1.set_title("Exponent scan", fontsize=11)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-8, -8))
    ax1.grid(alpha=0.22, lw=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.5)
    ax1.text(
        0.04,
        0.05,
        "same 308 bodies:\n246 train / 31 val / 31 holdout",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
    )
    _add_panel_label(ax1, "(b)")

    ell_sq = np.asarray([float(row["ell_sq"]) for row in coeff_rows], dtype=np.float64)
    x_line = np.linspace(0.0, float(np.max(ell_sq)) * 1.08, 300, dtype=np.float64)
    ax2.plot(x_line, x_line, color="#222222", lw=1.8, label=r"$k_d=\ell_d^2$", zorder=1)
    ax2.plot(
        x_line,
        float(lift["intercept"]) + float(lift["slope"]) * x_line,
        color="#8c564b",
        lw=1.8,
        ls="--",
        label=rf"Recovered fit $k_d \approx {_latex_sci(float(lift['intercept']), precision=1)} + {lift['slope']:.4f}\,\ell_d^2$",
        zorder=1,
    )
    for row in coeff_rows:
        ax2.scatter(
            float(row["ell_sq"]),
            float(row["k"]),
            s=16,
            color="#6f6f6f",
            alpha=0.62,
            edgecolors="none",
            zorder=2,
        )

    for idx, orbit_id in enumerate(reps):
        row = next(item for item in coeff_rows if str(item["orbit_id"]) == orbit_id)
        dataset = datasets_by_id[orbit_id]
        color = rep_cmap(idx % 10)
        ax2.scatter(
            float(row["ell_sq"]),
            float(row["k"]),
            s=40,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax2.annotate(
            str(body_name_by_id.get(orbit_id, orbit_id)),
            (float(row["ell_sq"]), float(row["k"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="#1f1f1f",
        )

    ax2.set_xlabel(r"$\ell_d^2\, \, \, [\mathrm{AU}^4\,\mathrm{day}^{-2}]$")
    ax2.set_ylabel(r"$k_d\, \, \, [\mathrm{AU}^4\,\mathrm{day}^{-2}]$")
    ax2.set_title("Recovered coefficient relation", fontsize=11)
    ax2.grid(alpha=0.22, lw=0.6)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.0)
    _add_panel_label(ax2, "(c)")

    _save_figure(fig, output_dir / "radial_family_triptych", show=show)


def _plot_holdout_transfer_energy(
    direct_summary: dict[str, object],
    holdout_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    output_dir: Path,
    *,
    show: bool,
    accel_source: str = "gradient",
    accel_cache_dir: str | None = None,
) -> None:
    holdout_info = dict(holdout_summary["holdout_generalization"])
    holdout_rows = list(holdout_info["per_dataset"])
    holdout_provenance = dict(holdout_summary.get("data_provenance", {}) or {})
    raw_manifest_path = Path(str(holdout_provenance["raw_manifest_path"])).resolve()
    holdout_datasets, holdout_by_id = _dataset_bundle(
        raw_manifest_path, accel_source=accel_source, accel_cache_dir=accel_cache_dir
    )
    holdout_body_name_by_id = _load_body_names(raw_manifest_path)
    energy = dict(direct_summary["energy"])
    per_dataset = list(energy["per_dataset"])
    ecc_norm, ecc_cmap = _eccentricity_color_meta(datasets)

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 10.6), constrained_layout=True)
    for ax in axes:
        ax.set_box_aspect(1.0)
        ax.tick_params(labelsize=8.5)

    holdout_cmap = plt.get_cmap("tab10")
    ax0 = axes[0]
    for idx, row in enumerate(holdout_rows):
        orbit_id = str(row["orbit_id"])
        dataset = holdout_by_id[orbit_id]
        color = holdout_cmap(idx % 10)
        r = np.asarray(dataset.r, dtype=np.float64)
        y = np.asarray(dataset.rddot, dtype=np.float64)
        stride = _sample_stride(r, 260)
        order = np.argsort(r)
        r_sorted = r[order]
        y_pred = float(row["k_pred_from_lift"]) / np.power(r_sorted, 3) - float(row["mu_train"]) / np.power(r_sorted, 2)
        ax0.scatter(
            r[::stride],
            y[::stride],
            s=12,
            color=color,
            alpha=0.24,
            edgecolors="none",
            zorder=1,
        )
        ax0.plot(
            r_sorted,
            y_pred,
            color=color,
            lw=2.2,
            label=(
                f"{holdout_body_name_by_id.get(orbit_id, orbit_id)} "
                f"(RMSE={float(row['radial_rmse']):.2e})"
            ),
            zorder=2,
        )

    ax0.set_xlabel(r"$r\, \, \, [\mathrm{AU}]$")
    ax0.set_ylabel(r"$\ddot{r}\, \, \, [\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax0.set_title("Symbolic holdout transfer", fontsize=11)
    ax0.grid(alpha=0.22, lw=0.6)
    ax0.legend(loc="upper right", frameon=True, fontsize=7.2)
    ax0.text(
        0.02,
        0.04,
        "No holdout refit of $k_d$:\n$k_d$ is predicted from the learned lift $k_d=a+b\\,\\ell_d^2$.",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
    _add_panel_label(ax0, "(a)")

    ax1 = axes[1]
    ax1.axhline(0.0, color="#222222", lw=1.9, label=r"$E_{\mathrm{fit}} - E_{\mathrm{true}} = 0$")
    for row in per_dataset:
        dataset = datasets_by_id[str(row["orbit_id"])]
        ax1.scatter(
            float(row["energy_true"]),
            float(row["energy_fit"]) - float(row["energy_true"]),
            s=24,
            color=ecc_cmap(ecc_norm(float(dataset.e))),
            alpha=0.86,
            edgecolors="none",
            zorder=2,
        )
    ax1.set_xlabel(r"True $E_d$ $[\mathrm{AU}^{2}\,\mathrm{day}^{-2}]$")
    ax1.set_ylabel(r"Recovered $E_d - $ True $E_d$ $[\mathrm{AU}^{2}\,\mathrm{day}^{-2}]$")
    ax1.set_title("Energy closure", fontsize=11)
    ax1.grid(alpha=0.22, lw=0.6)
    ax1.text(
        0.02,
        0.86,
        (
            r"$H(r,\theta,p_r,p_\theta)=\frac{1}{2}p_r^2+\frac{p_\theta^2}{2r^2}-\frac{\mu}{r}$"
            "\n"
            rf"max coefficient error $= {_latex_sci(float(energy['coeff_max_abs_error']), precision=2)}$"
        ),
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
    sm = plt.cm.ScalarMappable(norm=ecc_norm, cmap=ecc_cmap)
    sm.set_array([])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="6%", pad=0.10)
    cbar = fig.colorbar(sm, cax=cax)
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    ax1_pos = ax1.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax1_pos.y0, cax_pos.width, ax1_pos.height])
    cbar.set_label(r"Eccentricity $e$")
    cbar.ax.tick_params(labelsize=8.0)
    _add_panel_label(ax1, "(b)")

    _save_figure(fig, output_dir / "holdout_transfer_energy", show=show)


def _build_direct_holdout_corroboration(
    direct_summary: dict[str, object],
    datasets,
) -> dict[str, object]:
    split_strategy = str(direct_summary.get("split_strategy", ""))
    expected = "sorted-by-dynamic-range round-robin modulo 10: holdout=i%10==0, validation=i%10==1, train=otherwise"
    if split_strategy and split_strategy != expected:
        raise ValueError(f"unexpected split strategy for direct hold-out corroboration: {split_strategy}")

    split_sets = _round_robin_direct_holdout_splits(datasets)
    trainval_datasets = list(split_sets["trainval"])
    holdout_datasets = list(split_sets["holdout"])
    if int(len(holdout_datasets)) != int(direct_summary.get("n_holdout", len(holdout_datasets))):
        raise ValueError("reconstructed hold-out split size does not match the direct summary")

    ell_by_dataset = {
        str(row["orbit_id"]): float(row["ell_fit"])
        for row in list(direct_summary["stage_a"]["per_dataset"])
    }
    train_fit = fit_radial_family(trainval_datasets, exponent=2.0)
    holdout_oracle = evaluate_radial_family_with_fixed_mu(
        holdout_datasets,
        mu=float(train_fit["mu"]),
        exponent=2.0,
    )
    oracle_rows = _stage_row_map(list(holdout_oracle["per_dataset"]))
    lift = lift_coefficient_relation(
        datasets=trainval_datasets,
        ell_by_dataset=ell_by_dataset,
        k_by_dataset=dict(train_fit["k_by_dataset"]),
    )
    lift_intercept = float(lift["intercept"])
    lift_slope = float(lift["slope"])
    mu_train = float(train_fit["mu"])

    per_dataset = []
    k_rel_errors = []
    rmse_ratios = []
    rmse_penalties = []
    for dataset in holdout_datasets:
        orbit_id = str(dataset.orbit_id)
        ell_fit = float(ell_by_dataset[orbit_id])
        k_pred = float(lift_intercept + lift_slope * ell_fit * ell_fit)
        oracle_row = oracle_rows[orbit_id]
        k_oracle = float(holdout_oracle["k_by_dataset"][orbit_id])
        radial_pred = k_pred / np.power(dataset.r, 3) - mu_train / np.square(dataset.r)
        rmse_transfer = float(np.sqrt(np.mean(np.square(np.asarray(dataset.rddot, dtype=np.float64) - radial_pred))))
        rmse_oracle = float(oracle_row["rmse"])
        k_rel_error = float(abs(k_pred - k_oracle) / max(abs(k_oracle), 1.0e-15))
        rmse_ratio = float(rmse_transfer / max(rmse_oracle, 1.0e-15))
        rmse_penalty = float(rmse_transfer - rmse_oracle)
        k_rel_errors.append(k_rel_error)
        rmse_ratios.append(rmse_ratio)
        rmse_penalties.append(rmse_penalty)
        per_dataset.append(
            {
                "orbit_id": orbit_id,
                "eccentricity": float(dataset.e),
                "dynamic_range": float(dataset.dynamic_range),
                "ell_fit": ell_fit,
                "ell_sq": float(ell_fit * ell_fit),
                "k_pred": k_pred,
                "k_oracle": k_oracle,
                "k_rel_error": k_rel_error,
                "rmse_transfer": rmse_transfer,
                "rmse_oracle": rmse_oracle,
                "rmse_ratio": rmse_ratio,
                "rmse_penalty": rmse_penalty,
            }
        )

    return {
        "n_trainval": int(len(trainval_datasets)),
        "n_holdout": int(len(holdout_datasets)),
        "mu_train": mu_train,
        "lift_intercept": lift_intercept,
        "lift_slope": lift_slope,
        "per_dataset": per_dataset,
        "aggregate": {
            "median_k_rel_error": float(np.median(np.asarray(k_rel_errors, dtype=np.float64))),
            "max_k_rel_error": float(np.max(np.asarray(k_rel_errors, dtype=np.float64))),
            "median_rmse_ratio": float(np.median(np.asarray(rmse_ratios, dtype=np.float64))),
            "max_rmse_ratio": float(np.max(np.asarray(rmse_ratios, dtype=np.float64))),
            "max_rmse_penalty": float(np.max(np.asarray(rmse_penalties, dtype=np.float64))),
        },
    }


def _plot_direct_holdout_corroboration(
    direct_summary: dict[str, object],
    datasets,
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    summary = _build_direct_holdout_corroboration(direct_summary, datasets)
    rows = list(summary["per_dataset"])
    aggregate = dict(summary["aggregate"])

    k_oracle = np.asarray([float(row["k_oracle"]) for row in rows], dtype=np.float64)
    k_pred = np.asarray([float(row["k_pred"]) for row in rows], dtype=np.float64)
    rmse_oracle = np.asarray([float(row["rmse_oracle"]) for row in rows], dtype=np.float64)
    rmse_transfer = np.asarray([float(row["rmse_transfer"]) for row in rows], dtype=np.float64)
    eccentricity = np.asarray([float(row["eccentricity"]) for row in rows], dtype=np.float64)
    ecc_norm = Normalize(vmin=float(np.min(eccentricity)), vmax=float(np.max(eccentricity)))
    ecc_cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(8.8, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.075])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])
    for ax in (ax0, ax1):
        ax.set_box_aspect(1.0)
        ax.tick_params(labelsize=8.5)

    k_lo = float(min(np.min(k_oracle), np.min(k_pred)))
    k_hi = float(max(np.max(k_oracle), np.max(k_pred)))
    k_pad = 0.03 * max(k_hi - k_lo, 1.0e-12)
    ax0.plot(
        [k_lo - k_pad, k_hi + k_pad],
        [k_lo - k_pad, k_hi + k_pad],
        color="#222222",
        lw=1.8,
        label=r"$k_d^{\mathrm{pred}} = k_d^{\mathrm{refit}}$",
        zorder=1,
    )
    ax0.scatter(
        k_oracle,
        k_pred,
        s=30,
        c=eccentricity,
        cmap=ecc_cmap,
        norm=ecc_norm,
        alpha=0.90,
        edgecolors="none",
        zorder=2,
    )
    ax0.set_xlim(k_lo - k_pad, k_hi + k_pad)
    ax0.set_ylim(k_lo - k_pad, k_hi + k_pad)
    ax0.ticklabel_format(axis="both", style="sci", scilimits=(-4, -4))
    ax0.set_xlabel(r"Hold-out refit $k_d$ $[\mathrm{AU}^{4}\,\mathrm{day}^{-2}]$")
    ax0.set_ylabel(r"Transferred $k_d$ $[\mathrm{AU}^{4}\,\mathrm{day}^{-2}]$")
    ax0.set_title("Hold-out coefficient transfer", fontsize=11)
    ax0.grid(alpha=0.22, lw=0.6)
    ax0.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        frameon=True,
        fontsize=7.2,
    )
    ax0.text(
        0.98,
        0.04,
        (
            f"fit on {int(summary['n_trainval'])} non-holdouts\n"
            f"predict on {int(summary['n_holdout'])} holdouts\n"
            f"median error = {1.0e6 * float(aggregate['median_k_rel_error']):.1f} ppm\n"
            f"max = {1.0e6 * float(aggregate['max_k_rel_error']):.1f} ppm"
        ),
        transform=ax0.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.9,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
    )
    _add_panel_label(ax0, "(a)")

    rmse_lo = float(min(np.min(rmse_oracle), np.min(rmse_transfer)))
    rmse_hi = float(max(np.max(rmse_oracle), np.max(rmse_transfer)))
    rmse_pad = 0.04 * max(rmse_hi - rmse_lo, 1.0e-15)
    ax1.plot(
        [rmse_lo - rmse_pad, rmse_hi + rmse_pad],
        [rmse_lo - rmse_pad, rmse_hi + rmse_pad],
        color="#222222",
        lw=1.8,
        label=r"$\mathrm{RMSE}_{\mathrm{transfer}} = \mathrm{RMSE}_{\mathrm{refit}}$",
        zorder=1,
    )
    ax1.scatter(
        rmse_oracle,
        rmse_transfer,
        s=30,
        c=eccentricity,
        cmap=ecc_cmap,
        norm=ecc_norm,
        alpha=0.90,
        edgecolors="none",
        zorder=2,
    )
    ax1.set_xlim(rmse_lo - rmse_pad, rmse_hi + rmse_pad)
    ax1.set_ylim(rmse_lo - rmse_pad, rmse_hi + rmse_pad)
    ax1.ticklabel_format(axis="both", style="sci", scilimits=(-7, -7))
    ax1.set_xlabel(r"Hold-out refit RMSE of $\ddot{r}$ $[\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax1.set_ylabel(r"Transferred-law RMSE of $\ddot{r}$ $[\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax1.set_title("Hold-out radial corroboration", fontsize=11)
    ax1.grid(alpha=0.22, lw=0.6)
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        frameon=True,
        fontsize=7.0,
    )
    ax1.text(
        0.98,
        0.04,
        (
            f"median RMSE inflation = {100.0 * (float(aggregate['median_rmse_ratio']) - 1.0):.3f}%\n"
            f"max = {100.0 * (float(aggregate['max_rmse_ratio']) - 1.0):.3f}%"
        ),
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.9,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
    )
    sm = plt.cm.ScalarMappable(norm=ecc_norm, cmap=ecc_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    ax1_pos = ax1.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax1_pos.y0, cax_pos.width, ax1_pos.height])
    cbar.set_label(r"Eccentricity $e$")
    cbar.ax.tick_params(labelsize=8.0)
    _add_panel_label(ax1, "(b)")

    _save_figure(fig, output_dir / "holdout_corroboration", show=show)


def _plot_kepler_showcase_quadpanel(
    direct_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    stage_b_rows = _stage_row_map(list(direct_summary["stage_b_all"]["per_dataset"]))
    mu_fit = float(direct_summary["stage_b_all"]["mu"])
    reps = _top_dynamic_range_ids(datasets, count=3)
    rep_cmap = plt.get_cmap("tab10")

    lift = dict(direct_summary["coefficient_lift"])
    coeff_rows = list(lift["per_dataset"])
    energy = dict(direct_summary["energy"])
    per_dataset = list(energy["per_dataset"])
    ecc_norm, ecc_cmap = _eccentricity_color_meta(datasets)

    fig = plt.figure(figsize=(8.8, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.075])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])
    cax = fig.add_subplot(grid[1, 2])
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_box_aspect(1.0)
        ax.tick_params(labelsize=8.5)

    for idx, orbit_id in enumerate(reps):
        dataset = datasets_by_id[orbit_id]
        row = stage_b_rows[orbit_id]
        color = rep_cmap(idx % 10)
        r = np.asarray(dataset.r, dtype=np.float64)
        y = np.asarray(dataset.rddot, dtype=np.float64)
        stride = _sample_stride(r, 260)
        order = np.argsort(r)
        r_sorted = r[order]
        y_fit = float(row["k_fit"]) / np.power(r_sorted, 3) - mu_fit / np.power(r_sorted, 2)
        ax0.scatter(
            r[::stride],
            y[::stride],
            s=8,
            color=color,
            alpha=0.18,
            edgecolors="none",
            zorder=1,
        )
        ax0.plot(
            r_sorted,
            y_fit,
            color=color,
            lw=2.0,
            label=_display_body_name(body_name_by_id.get(orbit_id, orbit_id)),
            zorder=2,
        )

    ax0.set_xlabel(r"$r\, \, \, [\mathrm{AU}]$")
    ax0.set_ylabel(r"$\ddot{r}\, \, \, [\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax0.set_title("Radial family", fontsize=11)
    ax0.grid(alpha=0.22, lw=0.6)
    ax0.legend(loc="upper right", frameon=True, fontsize=7.0)
    _add_panel_label(ax0, "(a)")

    scan_rows = sorted(list(direct_summary["power_scan"]["rows"]), key=lambda row: float(row["exponent"]))
    exponents = np.asarray([float(row["exponent"]) for row in scan_rows], dtype=np.float64)
    train_rmse = np.asarray([float(row["train_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    holdout_rmse = np.asarray([float(row["holdout_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    ax1.plot(exponents, train_rmse, color="#4c78a8", lw=2.0, label="Train (246)")
    ax1.plot(exponents, holdout_rmse, color="#c23b22", lw=2.0, ls="--", label="Holdout (31)")
    ax1.axvline(2.0, color="#222222", lw=1.5, ls="--")
    ax1.annotate(
        r"$p=2$",
        (2.0, float(np.min(holdout_rmse))),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
        color="#222222",
    )
    ax1.set_xlabel(r"Candidate force exponent $p$")
    ax1.set_ylabel(r"Mean RMSE of $\ddot{r}$ [$\mathrm{AU}\,\mathrm{day}^{-2}$]")
    ax1.set_title("Exponent scan", fontsize=11)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-8, -8))
    ax1.grid(alpha=0.22, lw=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.2)
    _add_panel_label(ax1, "(b)")

    ell_sq = np.asarray([float(row["ell_sq"]) for row in coeff_rows], dtype=np.float64)
    k_vals = np.asarray([float(row["k"]) for row in coeff_rows], dtype=np.float64)
    common_lo = float(min(np.min(ell_sq), np.min(k_vals)))
    common_hi = float(max(np.max(ell_sq), np.max(k_vals)))
    common_pad = 0.05 * max(common_hi - common_lo, 1.0e-12)
    axis_lo = common_lo - common_pad
    axis_hi = common_hi + common_pad
    x_line = np.linspace(axis_lo, axis_hi, 300, dtype=np.float64)
    ax2.plot(x_line, x_line, color="#222222", lw=1.8, label=r"$k_d=\ell_d^2$", zorder=1)
    ax2.plot(
        x_line,
        float(lift["intercept"]) + float(lift["slope"]) * x_line,
        color="#8c564b",
        lw=1.8,
        ls="--",
        label=rf"Recovered fit $k_d \approx {_latex_sci(float(lift['intercept']), precision=1)} + {lift['slope']:.4f}\,\ell_d^2$",
        zorder=1,
    )
    for row in coeff_rows:
        ax2.scatter(
            float(row["ell_sq"]),
            float(row["k"]),
            s=16,
            color="#6f6f6f",
            alpha=0.62,
            edgecolors="none",
            zorder=2,
        )

    for idx, orbit_id in enumerate(reps):
        row = next(item for item in coeff_rows if str(item["orbit_id"]) == orbit_id)
        color = rep_cmap(idx % 10)
        ax2.scatter(
            float(row["ell_sq"]),
            float(row["k"]),
            s=42,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax2.annotate(
            _display_body_name(body_name_by_id.get(orbit_id, orbit_id)),
            (float(row["ell_sq"]), float(row["k"])),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.0,
            color="#1f1f1f",
        )

    ax2.set_xlabel(r"$\ell_d^2\, \, \, [\mathrm{AU}^4\,\mathrm{day}^{-2}]$")
    ax2.set_ylabel(r"$k_d\, \, \, [\mathrm{AU}^4\,\mathrm{day}^{-2}]$")
    ax2.set_xlim(axis_lo, axis_hi)
    ax2.set_ylim(axis_lo, axis_hi)
    ax2.ticklabel_format(axis="both", style="sci", scilimits=(-4, -4))
    ax2.set_title("Recovered coefficient relation", fontsize=11)
    ax2.grid(alpha=0.22, lw=0.6)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.0)
    _add_panel_label(ax2, "(c)")

    ax3.axhline(0.0, color="#222222", lw=1.9, label=r"$E_{\mathrm{fit}} - E_{\mathrm{true}} = 0$")
    for row in per_dataset:
        dataset = datasets_by_id[str(row["orbit_id"])]
        ax3.scatter(
            float(row["energy_true"]),
            float(row["energy_fit"]) - float(row["energy_true"]),
            s=24,
            color=ecc_cmap(ecc_norm(float(dataset.e))),
            alpha=0.86,
            edgecolors="none",
            zorder=2,
        )
    ax3.set_xlabel(r"True $E_d$ $[\mathrm{AU}^{2}\,\mathrm{day}^{-2}]$")
    ax3.set_ylabel(r"Recovered $E_d - $ True $E_d$ $[\mathrm{AU}^{2}\,\mathrm{day}^{-2}]$")
    ax3.set_title("Energy closure", fontsize=11)
    ax3.grid(alpha=0.22, lw=0.6)
    ax3.text(
        0.02,
        0.86,
        (
            r"$H(r,\theta,p_r,p_\theta)=\frac{1}{2}p_r^2+\frac{p_\theta^2}{2r^2}-\frac{\mu}{r}$"
            "\n"
            rf"max coefficient error $= {_latex_sci(float(energy['coeff_max_abs_error']), precision=2)}$"
        ),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
    sm = plt.cm.ScalarMappable(norm=ecc_norm, cmap=ecc_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    ax3_pos = ax3.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax3_pos.y0, cax_pos.width, ax3_pos.height])
    cbar.set_label(r"Eccentricity $e$")
    cbar.ax.tick_params(labelsize=8.0)
    _add_panel_label(ax3, "(d)")

    _save_figure(fig, output_dir / "kepler_showcase_quadpanel", show=show)


def _plot_kepler_showcase_sixpanel(
    direct_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    """The paper's six-panel showcase: quadpanel (a)-(d) over the hold-out
    corroboration pair (e)-(f), with one shared eccentricity colorbar."""
    stage_b_rows = _stage_row_map(list(direct_summary["stage_b_all"]["per_dataset"]))
    mu_fit = float(direct_summary["stage_b_all"]["mu"])
    reps = _top_dynamic_range_ids(datasets, count=3)
    rep_cmap = plt.get_cmap("tab10")
    lift = dict(direct_summary["coefficient_lift"])
    coeff_rows = list(lift["per_dataset"])
    energy = dict(direct_summary["energy"])
    per_dataset = list(energy["per_dataset"])
    ecc_norm, ecc_cmap = _eccentricity_color_meta(datasets)

    corroboration = _build_direct_holdout_corroboration(direct_summary, datasets)
    corroboration_rows = list(corroboration["per_dataset"])
    aggregate = dict(corroboration["aggregate"])
    k_oracle = np.asarray([float(row["k_oracle"]) for row in corroboration_rows], dtype=np.float64)
    k_pred = np.asarray([float(row["k_pred"]) for row in corroboration_rows], dtype=np.float64)
    rmse_oracle = np.asarray([float(row["rmse_oracle"]) for row in corroboration_rows], dtype=np.float64)
    rmse_transfer = np.asarray([float(row["rmse_transfer"]) for row in corroboration_rows], dtype=np.float64)
    holdout_ecc = np.asarray([float(row["eccentricity"]) for row in corroboration_rows], dtype=np.float64)

    fig = plt.figure(figsize=(8.8, 12.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 0.075])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])
    ax4 = fig.add_subplot(grid[2, 0])
    ax5 = fig.add_subplot(grid[2, 1])
    cax = fig.add_subplot(grid[1, 2])
    for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
        ax.set_box_aspect(1.0)
        ax.tick_params(labelsize=8.5)

    # (a) radial family: the whole ensemble in grey underneath (each body is a short
    # arc of its own family member), with the highest-leverage orbits highlighted
    for dataset in datasets:
        r_all = np.asarray(dataset.r, dtype=np.float64)
        y_all = np.asarray(dataset.rddot, dtype=np.float64)
        st = _sample_stride(r_all, 120)
        ax0.scatter(
            r_all[::st], y_all[::st], s=3, color="0.74", alpha=0.30, edgecolors="none",
            zorder=0, rasterized=True,
        )
    for idx, orbit_id in enumerate(reps):
        dataset = datasets_by_id[orbit_id]
        row = stage_b_rows[orbit_id]
        color = rep_cmap(idx % 10)
        r = np.asarray(dataset.r, dtype=np.float64)
        y = np.asarray(dataset.rddot, dtype=np.float64)
        stride = _sample_stride(r, 260)
        order = np.argsort(r)
        r_sorted = r[order]
        y_fit = float(row["k_fit"]) / np.power(r_sorted, 3) - mu_fit / np.power(r_sorted, 2)
        ax0.scatter(r[::stride], y[::stride], s=8, color=color, alpha=0.18, edgecolors="none", zorder=1)
        ax0.plot(
            r_sorted,
            y_fit,
            color=color,
            lw=2.0,
            label=_display_body_name(body_name_by_id.get(orbit_id, orbit_id)),
            zorder=2,
        )
    ax0.set_xlabel(r"$r\, \, \, [\mathrm{AU}]$")
    ax0.set_ylabel(r"$\ddot{r}\, \, \, [\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax0.set_title("Radial family", fontsize=11)
    ax0.grid(alpha=0.22, lw=0.6)
    handles, labels = ax0.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="none", color="0.62", markersize=4))
    labels.append(f"all {len(datasets)} bodies")
    ax0.legend(handles, labels, loc="upper right", frameon=True, fontsize=7.0)
    _add_panel_label(ax0, "(a)")

    # (b) exponent scan
    scan_rows = sorted(list(direct_summary["power_scan"]["rows"]), key=lambda row: float(row["exponent"]))
    exponents = np.asarray([float(row["exponent"]) for row in scan_rows], dtype=np.float64)
    train_rmse = np.asarray([float(row["train_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    holdout_rmse = np.asarray([float(row["holdout_mean_rmse"]) for row in scan_rows], dtype=np.float64)
    ax1.plot(exponents, train_rmse, color="#4c78a8", lw=2.0, label="Train (246)")
    ax1.plot(exponents, holdout_rmse, color="#c23b22", lw=2.0, ls="--", label="Holdout (31)")
    ax1.axvline(2.0, color="#222222", lw=1.5, ls="--")
    ax1.annotate(
        r"$p=2$",
        (2.0, float(np.min(holdout_rmse))),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
        color="#222222",
    )
    ax1.set_xlabel(r"Candidate force exponent $p$")
    ax1.set_ylabel(r"Mean RMSE of $\ddot{r}$ [$\mathrm{AU}\,\mathrm{day}^{-2}$]")
    ax1.set_title("Exponent scan", fontsize=11)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-8, -8))
    ax1.grid(alpha=0.22, lw=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.2)
    _add_panel_label(ax1, "(b)")

    # (c) recovered coefficient relation
    ell_sq = np.asarray([float(row["ell_sq"]) for row in coeff_rows], dtype=np.float64)
    k_vals = np.asarray([float(row["k"]) for row in coeff_rows], dtype=np.float64)
    common_lo = float(min(np.min(ell_sq), np.min(k_vals)))
    common_hi = float(max(np.max(ell_sq), np.max(k_vals)))
    common_pad = 0.05 * max(common_hi - common_lo, 1.0e-12)
    axis_lo = common_lo - common_pad
    axis_hi = common_hi + common_pad
    x_line = np.linspace(axis_lo, axis_hi, 300, dtype=np.float64)
    ax2.plot(x_line, x_line, color="#222222", lw=1.8, label=r"$k_d=\ell_d^2$", zorder=1)
    ax2.plot(
        x_line,
        float(lift["intercept"]) + float(lift["slope"]) * x_line,
        color="#8c564b",
        lw=1.8,
        ls="--",
        label=rf"Recovered fit $k_d \approx {_latex_sci(float(lift['intercept']), precision=1)} + {lift['slope']:.4f}\,\ell_d^2$",
        zorder=1,
    )
    for row in coeff_rows:
        ax2.scatter(float(row["ell_sq"]), float(row["k"]), s=16, color="#6f6f6f", alpha=0.62, edgecolors="none", zorder=2)
    for idx, orbit_id in enumerate(reps):
        row = next(item for item in coeff_rows if str(item["orbit_id"]) == orbit_id)
        color = rep_cmap(idx % 10)
        ax2.scatter(float(row["ell_sq"]), float(row["k"]), s=42, color=color, edgecolors="black", linewidths=0.5, zorder=3)
        ax2.annotate(
            _display_body_name(body_name_by_id.get(orbit_id, orbit_id)),
            (float(row["ell_sq"]), float(row["k"])),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.0,
            color="#1f1f1f",
        )
    ax2.set_xlabel(r"$\ell_d^2\, \, \, [\mathrm{AU}^4\,\mathrm{day}^{-2}]$")
    ax2.set_ylabel(r"$k_d\, \, \, [\mathrm{AU}^4\,\mathrm{day}^{-2}]$")
    ax2.set_xlim(axis_lo, axis_hi)
    ax2.set_ylim(axis_lo, axis_hi)
    ax2.ticklabel_format(axis="both", style="sci", scilimits=(-4, -4))
    ax2.set_title("Recovered coefficient relation", fontsize=11)
    ax2.grid(alpha=0.22, lw=0.6)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.0)
    _add_panel_label(ax2, "(c)")

    # (d) energy closure
    ax3.axhline(0.0, color="#222222", lw=1.9, label=r"$E_{\mathrm{fit}} - E_{\mathrm{true}} = 0$")
    for row in per_dataset:
        dataset = datasets_by_id[str(row["orbit_id"])]
        ax3.scatter(
            float(row["energy_true"]),
            float(row["energy_fit"]) - float(row["energy_true"]),
            s=24,
            color=ecc_cmap(ecc_norm(float(dataset.e))),
            alpha=0.86,
            edgecolors="none",
            zorder=2,
        )
    ax3.set_xlabel(r"True $E_d$ $[\mathrm{AU}^{2}\,\mathrm{day}^{-2}]$")
    ax3.set_ylabel(r"Recovered $E_d - $ True $E_d$ $[\mathrm{AU}^{2}\,\mathrm{day}^{-2}]$")
    ax3.set_title("Energy closure", fontsize=11)
    ax3.grid(alpha=0.22, lw=0.6)
    ax3.text(
        0.02,
        0.86,
        (
            r"$H(r,\theta,p_r,p_\theta)=\frac{1}{2}p_r^2+\frac{p_\theta^2}{2r^2}-\frac{\mu}{r}$"
            "\n"
            rf"max coefficient error $= {_latex_sci(float(energy['coeff_max_abs_error']), precision=2)}$"
        ),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
    _add_panel_label(ax3, "(d)")

    # (e) hold-out coefficient transfer
    k_lo = float(min(np.min(k_oracle), np.min(k_pred)))
    k_hi = float(max(np.max(k_oracle), np.max(k_pred)))
    k_pad = 0.03 * max(k_hi - k_lo, 1.0e-12)
    ax4.plot(
        [k_lo - k_pad, k_hi + k_pad],
        [k_lo - k_pad, k_hi + k_pad],
        color="#222222",
        lw=1.8,
        label=r"$k_d^{\mathrm{pred}} = k_d^{\mathrm{refit}}$",
        zorder=1,
    )
    ax4.scatter(k_oracle, k_pred, s=30, c=holdout_ecc, cmap=ecc_cmap, norm=ecc_norm, alpha=0.90, edgecolors="none", zorder=2)
    ax4.set_xlim(k_lo - k_pad, k_hi + k_pad)
    ax4.set_ylim(k_lo - k_pad, k_hi + k_pad)
    ax4.ticklabel_format(axis="both", style="sci", scilimits=(-4, -4))
    ax4.set_xlabel(r"Hold-out refit $k_d$ $[\mathrm{AU}^{4}\,\mathrm{day}^{-2}]$")
    ax4.set_ylabel(r"Transferred $k_d$ $[\mathrm{AU}^{4}\,\mathrm{day}^{-2}]$")
    ax4.set_title("Hold-out coefficient transfer", fontsize=11)
    ax4.grid(alpha=0.22, lw=0.6)
    ax4.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=True, fontsize=7.2)
    ax4.text(
        0.98,
        0.04,
        (
            f"fit on {int(corroboration['n_trainval'])} non-holdouts\n"
            f"predict on {int(corroboration['n_holdout'])} holdouts\n"
            f"median error = {1.0e6 * float(aggregate['median_k_rel_error']):.1f} ppm\n"
            f"max = {1.0e6 * float(aggregate['max_k_rel_error']):.1f} ppm"
        ),
        transform=ax4.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.9,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
    )
    _add_panel_label(ax4, "(e)")

    # (f) hold-out radial corroboration
    rmse_lo = float(min(np.min(rmse_oracle), np.min(rmse_transfer)))
    rmse_hi = float(max(np.max(rmse_oracle), np.max(rmse_transfer)))
    rmse_pad = 0.04 * max(rmse_hi - rmse_lo, 1.0e-15)
    ax5.plot(
        [rmse_lo - rmse_pad, rmse_hi + rmse_pad],
        [rmse_lo - rmse_pad, rmse_hi + rmse_pad],
        color="#222222",
        lw=1.8,
        label=r"$\mathrm{RMSE}_{\mathrm{transfer}} = \mathrm{RMSE}_{\mathrm{refit}}$",
        zorder=1,
    )
    ax5.scatter(rmse_oracle, rmse_transfer, s=30, c=holdout_ecc, cmap=ecc_cmap, norm=ecc_norm, alpha=0.90, edgecolors="none", zorder=2)
    ax5.set_xlim(rmse_lo - rmse_pad, rmse_hi + rmse_pad)
    ax5.set_ylim(rmse_lo - rmse_pad, rmse_hi + rmse_pad)
    ax5.ticklabel_format(axis="both", style="sci", scilimits=(-7, -7))
    ax5.set_xlabel(r"Hold-out refit RMSE of $\ddot{r}$ $[\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax5.set_ylabel(r"Transferred-law RMSE of $\ddot{r}$ $[\mathrm{AU}\,\mathrm{day}^{-2}]$")
    ax5.set_title("Hold-out radial corroboration", fontsize=11)
    ax5.grid(alpha=0.22, lw=0.6)
    ax5.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=True, fontsize=7.0)
    ax5.text(
        0.98,
        0.04,
        (
            f"median RMSE inflation = {100.0 * (float(aggregate['median_rmse_ratio']) - 1.0):.3f}%\n"
            f"max = {100.0 * (float(aggregate['max_rmse_ratio']) - 1.0):.3f}%"
        ),
        transform=ax5.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.9,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
    )
    _add_panel_label(ax5, "(f)")

    sm = plt.cm.ScalarMappable(norm=ecc_norm, cmap=ecc_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    ax3_pos = ax3.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax3_pos.y0, cax_pos.width, ax3_pos.height])
    cbar.set_label(r"Eccentricity $e$ (panels d, e, f)")
    cbar.ax.tick_params(labelsize=8.0)

    _save_figure(fig, output_dir / "kepler_showcase_sixpanel", show=show)


def _plot_energy_invariant(
    direct_summary: dict[str, object],
    datasets,
    datasets_by_id: dict[str, object],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    energy = dict(direct_summary["energy"])
    coeffs = dict(energy["coeffs"])
    expected = dict(energy["expected_coeffs"])
    per_dataset = list(energy["per_dataset"])
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
        axes[1].scatter(
            float(row["energy_true"]),
            float(row["energy_fit"]),
            s=28,
            color=cmap(norm(float(dataset.e))),
            alpha=0.86,
            edgecolors="none",
            zorder=2,
        )
    axes[1].set_xlabel(r"True $E_d$")
    axes[1].set_ylabel(r"Recovered $E_d$")
    axes[1].set_title("Recovered Specific-Energy Levels for All 308 Bodies")
    axes[1].grid(alpha=0.22, lw=0.6)

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


def _plot_cleanliness_story(
    direct_summary: dict[str, object],
    datasets,
    body_name_by_id: dict[str, str],
    output_dir: Path,
    *,
    show: bool,
) -> dict[str, object]:
    rows = _per_object_cleanliness(datasets, direct_summary)
    by_id = {str(row["orbit_id"]): dict(row) for row in rows}
    ceres = dict(by_id["mp_1_ceres"])
    cleaner_than_ceres = [row for row in rows if float(row["radial_rel_rmse"]) < float(ceres["radial_rel_rmse"])]
    more_leverage_than_ceres = [row for row in rows if float(row["dynamic_range"]) > float(ceres["dynamic_range"])]
    both = [
        row for row in rows
        if float(row["radial_rel_rmse"]) < float(ceres["radial_rel_rmse"])
        and float(row["dynamic_range"]) > float(ceres["dynamic_range"])
    ]
    frontier = []
    best_so_far = float("inf")
    for row in sorted(rows, key=lambda item: (-float(item["dynamic_range"]), float(item["radial_rel_rmse"]))):
        if float(row["radial_rel_rmse"]) < best_so_far:
            frontier.append(dict(row))
            best_so_far = float(row["radial_rel_rmse"])

    fig, ax = plt.subplots(figsize=(8.9, 6.4))
    x = np.asarray([float(row["dynamic_range"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["radial_rel_rmse"]) for row in rows], dtype=np.float64)
    ecc = np.asarray([float(row["eccentricity"]) for row in rows], dtype=np.float64)
    norm = Normalize(vmin=float(np.min(ecc)), vmax=float(np.max(ecc)))
    cmap = plt.get_cmap("viridis")
    ax.scatter(x, y, c=ecc, cmap=cmap, norm=norm, s=28, alpha=0.85, edgecolors="none", zorder=1)

    ax.scatter(
        float(ceres["dynamic_range"]),
        float(ceres["radial_rel_rmse"]),
        s=92,
        color="#c23b22",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    ax.annotate(
        "Ceres",
        (float(ceres["dynamic_range"]), float(ceres["radial_rel_rmse"])),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9,
        color="#8a1e08",
    )

    for row in frontier[:8]:
        orbit_id = str(row["orbit_id"])
        ax.scatter(
            float(row["dynamic_range"]),
            float(row["radial_rel_rmse"]),
            s=76,
            color=cmap(norm(float(row["eccentricity"]))),
            edgecolors="black",
            linewidths=0.55,
            zorder=4,
        )
        ax.annotate(
            str(body_name_by_id.get(orbit_id, orbit_id)),
            (float(row["dynamic_range"]), float(row["radial_rel_rmse"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#1f1f1f",
        )

    ax.set_xlabel(r"Radial leverage $\Lambda = Q/q$")
    ax.set_ylabel("Relative radial Kepler residual")
    ax.set_title("Most of the 308-Body Pool Is Cleaner than Ceres")
    ax.grid(alpha=0.22, lw=0.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Eccentricity $e$")

    _save_figure(fig, output_dir / "ensemble_cleanliness", show=show)

    return {
        "ceres_orbit_id": "mp_1_ceres",
        "ceres_dynamic_range": float(ceres["dynamic_range"]),
        "ceres_radial_rel_rmse": float(ceres["radial_rel_rmse"]),
        "n_cleaner_than_ceres": int(len(cleaner_than_ceres)),
        "n_more_leverage_than_ceres": int(len(more_leverage_than_ceres)),
        "n_cleaner_and_more_leverage_than_ceres": int(len(both)),
        "frontier_orbit_ids": [str(row["orbit_id"]) for row in frontier],
    }


def _write_story_report(
    *,
    direct_summary: dict[str, object],
    selection_summary: dict[str, object] | None,
    story_stats: dict[str, object],
    output_dir: Path,
) -> Path:
    criteria = None if selection_summary is None else dict(selection_summary.get("criteria", {}))
    counts = None if selection_summary is None else dict(selection_summary.get("counts", {}))
    stage_a = dict(direct_summary["stage_a"])
    stage_b = dict(direct_summary["stage_b_all"])
    lift = dict(direct_summary["coefficient_lift"])
    energy = dict(direct_summary["energy"])
    power_scan = dict(direct_summary["power_scan"])
    exact = dict(power_scan["exact_exponent_row"])

    lines = [
        "# Kepler Showcase: 308-Body Daily Weathered Ensemble",
        "",
        "## Parent Pool",
        "",
        f"- Object count: `{int(direct_summary.get('n_objects', len(direct_summary['orbit_registry'])))}`",
        f"- Cadence: `{float(direct_summary.get('cadence_days', 1.0)):.1f}` day",
        "- Time window: `1980-01-01` to `2009-12-31`",
    ]
    if criteria is not None:
        jpl = dict(criteria.get("jpl", {}))
        ssodnet = dict(criteria.get("ssodnet", {}))
        lines.extend(
            [
                f"- JPL geometric cut: `a in [{jpl.get('a_min_au')}, {jpl.get('a_max_au')}] AU`, `q > {jpl.get('q_min_au_strict')} AU`",
                f"- JPL quality cut: `data_arc >= {jpl.get('data_arc_min_days')} d`, `n_obs_used >= {jpl.get('n_obs_min')}`",
                f"- SsODNet mass cut: `mass > {ssodnet.get('mass_min_kg'):.0e} kg`",
            ]
        )
    if counts is not None:
        lines.extend(
            [
                f"- JPL bulk candidate count before cross-match: `{int(counts.get('jpl_bulk_query_count', 0))}`",
                f"- SsODNet mass-selected count: `{int(counts.get('ssodnet_mass_selected', 0))}`",
                f"- Final selected count: `{int(counts.get('selected_count', 0))}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Joint Reduced-Kepler Fit",
            "",
            f"- `max |ell-h|/|h| = {float(stage_a['max_rel_error']):.3e}`",
            f"- `mu_true = {float(direct_summary['mu_true']):.9e}`",
            f"- `mu_fit = {float(stage_b['mu']):.9e}`",
            f"- `mu_abs_error = {float(stage_b['mu_abs_error']):.3e}`",
            f"- `max |k-h^2| = {float(stage_b['max_k_abs_error']):.3e}`",
            f"- `lift slope = {float(lift['slope']):.6f}`",
            f"- `energy coeff max abs error = {float(energy['coeff_max_abs_error']):.3e}`",
            "",
            "## Inverse-Power Scan",
            "",
            "- Holdout split used only for the scan: deterministic round-robin after sorting by radial leverage.",
            f"- `best_train_exponent = {float(power_scan['best_train_exponent']):.3f}`",
            f"- `best_holdout_exponent = {float(power_scan['best_holdout_exponent']):.3f}`",
            f"- `p=2 holdout mean RMSE = {float(exact['holdout_mean_rmse']):.3e}`",
            f"- `holdout margin to second = {float(power_scan['holdout_margin_to_second']):.3e}`",
            "",
            "## Ceres Benchmark",
            "",
            f"- `Ceres` radial leverage `Lambda = {float(story_stats['ceres_dynamic_range']):.3f}`",
            f"- `Ceres` relative radial residual `= {float(story_stats['ceres_radial_rel_rmse']):.3e}`",
            f"- Bodies cleaner than Ceres: `{int(story_stats['n_cleaner_than_ceres'])}`",
            f"- Bodies with more leverage than Ceres: `{int(story_stats['n_more_leverage_than_ceres'])}`",
            f"- Bodies with both more leverage and lower residual than Ceres: `{int(story_stats['n_cleaner_and_more_leverage_than_ceres'])}`",
            "",
            "## Interpretation",
            "",
            "- The all-308 daily HORIZONS ensemble remains strongly coherent under the shared reduced-Kepler fit.",
            "- The parent pool behaves more like a broad clean main-belt family with a dirty tail than like a generic noisy asteroid catalog.",
            "- At the direct-fit level, the data support using the full reproducible parent pool rather than a hand-picked sub-ensemble.",
        ]
    )

    output_path = output_dir / "showcase_story.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate direct-fit paper figures and a short story report for the all-308 real-data Kepler showcase",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--direct_summary",
        type=str,
        default=str(_default_direct_summary_path()),
        help="Path to the joint direct-fit summary JSON",
    )
    parser.add_argument(
        "--raw_manifest",
        type=str,
        default=None,
        help="Raw state manifest used to build the daily weathered datasets; inferred from the summary when omitted",
    )
    parser.add_argument(
        "--selection_summary",
        type=str,
        default=str(_default_selection_summary_path()),
        help="Selection-summary JSON describing the reproducible parent-pool rule",
    )
    parser.add_argument(
        "--holdout_summary",
        type=str,
        default=str(_default_holdout_summary_path()),
        help="Optional symbolic hold-out summary JSON used to build the transfer figure",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where the figures and report will be written; defaults to <direct_summary_dir>/figures",
    )
    parser.add_argument(
        "--accel_source",
        choices=("gradient", "surrogate"),
        default="gradient",
        help="Acceleration source for the rebuilt weathered datasets; must match the direct summary's provenance",
    )
    parser.add_argument(
        "--accel_cache_dir",
        type=str,
        default=None,
        help="Precomputed surrogate-acceleration cache (required with --accel_source surrogate; entries were built with certificate fits)",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively as well as saving them")
    args = parser.parse_args()

    direct_summary = _load_json(args.direct_summary)
    raw_manifest_path = _infer_raw_manifest(direct_summary, args.raw_manifest)
    selection_summary = None
    selection_path = Path(args.selection_summary)
    if selection_path.exists():
        selection_summary = _load_json(selection_path)
    holdout_summary = None
    holdout_path = Path(args.holdout_summary)
    if holdout_path.exists():
        holdout_summary = _load_json(holdout_path)

    if args.output_dir is None:
        output_dir = Path(args.direct_summary).resolve().parent / "figures"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    body_name_by_id = _load_body_names(raw_manifest_path)
    datasets, datasets_by_id = _dataset_bundle(
        raw_manifest_path,
        accel_source=str(args.accel_source),
        accel_cache_dir=args.accel_cache_dir,
    )

    _plot_areal_law_family(direct_summary, datasets, datasets_by_id, body_name_by_id, output_dir, show=bool(args.show))
    _plot_radial_family_and_selection(direct_summary, datasets, datasets_by_id, body_name_by_id, output_dir, show=bool(args.show))
    _plot_coefficient_manifold(direct_summary, datasets, datasets_by_id, body_name_by_id, output_dir, show=bool(args.show))
    _plot_radial_family_triptych(direct_summary, datasets, datasets_by_id, body_name_by_id, output_dir, show=bool(args.show))
    if holdout_summary is not None:
        _plot_holdout_transfer_energy(
            direct_summary,
            holdout_summary,
            datasets,
            datasets_by_id,
            output_dir,
            show=bool(args.show),
            accel_source=str(args.accel_source),
            accel_cache_dir=args.accel_cache_dir,
        )
    _plot_direct_holdout_corroboration(
        direct_summary,
        datasets,
        body_name_by_id,
        output_dir,
        show=bool(args.show),
    )
    _plot_kepler_showcase_sixpanel(
        direct_summary,
        datasets,
        datasets_by_id,
        body_name_by_id,
        output_dir,
        show=bool(args.show),
    )
    _plot_kepler_showcase_quadpanel(
        direct_summary,
        datasets,
        datasets_by_id,
        body_name_by_id,
        output_dir,
        show=bool(args.show),
    )
    _plot_energy_invariant(direct_summary, datasets, datasets_by_id, output_dir, show=bool(args.show))
    story_stats = _plot_cleanliness_story(direct_summary, datasets, body_name_by_id, output_dir, show=bool(args.show))
    report_path = _write_story_report(
        direct_summary=direct_summary,
        selection_summary=selection_summary,
        story_stats=story_stats,
        output_dir=output_dir,
    )

    stats_path = output_dir / "showcase_story_stats.json"
    stats_path.write_text(json.dumps(_jsonable(story_stats), indent=2), encoding="utf-8")

    print(f"Saved figures and report to {output_dir}")
    for stem in (
        "areal_law_family",
        "radial_family_selection",
        "coefficient_manifold",
        "radial_family_triptych",
        "holdout_transfer_energy",
        "holdout_corroboration",
        "kepler_showcase_quadpanel",
        "energy_invariant",
        "ensemble_cleanliness",
    ):
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        if png_path.exists():
            print(f"  - {png_path}")
        if pdf_path.exists():
            print(f"  - {pdf_path}")
    print(f"  - {report_path}")
    print(f"  - {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
