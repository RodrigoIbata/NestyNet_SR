#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Visualize logistic-growth ODE discovery results across baseline/template stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def _safe_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_time_and_state(data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load logistic data from current CSV schema."""
    df = pd.read_csv(data_file)
    cols = list(df.columns)

    if "y" not in df.columns:
        raise ValueError(
            f"Could not find dependent-variable column in {data_file}. "
            f"Expected: y. Found: {cols}"
        )
    y_col = "y"

    if "t" not in df.columns:
        raise ValueError(
            f"Could not find time column in {data_file}. "
            f"Expected: t. Found: {cols}"
        )
    t_col = "t"

    return df[t_col].to_numpy(), df[y_col].to_numpy()


def _extract_equation_line(content: str) -> str | None:
    for line in content.splitlines():
        line_s = line.strip()
        if "=" in line_s and "u_x0" in line_s:
            return line_s
    return None


def load_result(result_file: Path | None):
    """Load ODE discovery result from a .human file."""
    if result_file is None or not result_file.exists():
        return None

    content = result_file.read_text(encoding="utf-8")
    result = {"content": content, "source_file": str(result_file)}
    result["equation"] = _extract_equation_line(content)

    for line in content.splitlines():
        if "RMS (train):" in line:
            result["rms_train"] = _safe_float(line.split(":", 1)[1].strip())
        if "RMS (val):" in line:
            result["rms_val"] = _safe_float(line.split(":", 1)[1].strip())

    return result


def load_metadata(json_file: Path | None):
    """Load metadata from JSON output."""
    if json_file is None or not json_file.exists():
        return None
    return json.loads(json_file.read_text(encoding="utf-8"))


def _extract_varpro_metadata(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    de_block = payload.get("de_discovery", {})
    if isinstance(de_block, dict):
        vp = de_block.get("varpro_metadata", {})
        if isinstance(vp, dict):
            return vp
    return {}


def _extract_power_param(params: dict | None) -> float | None:
    if not isinstance(params, dict):
        return None
    if "p" in params:
        p = _safe_float(params["p"], default=float("nan"))
        return p if np.isfinite(p) else None
    for key, value in params.items():
        if str(key).startswith("p"):
            p = _safe_float(value, default=float("nan"))
            if np.isfinite(p):
                return p
    return None


def _artifact_candidates(results_dir: Path, stem: str, stage: str, suffix: str) -> list[Path]:
    if stage == "selected":
        candidates = [results_dir / f"{stem}_selected_de.{suffix}"]
        candidates.append(results_dir / f"{stem}_de.{suffix}")
        return candidates
    candidates = [results_dir / f"{stem}_{stage}_de.{suffix}"]
    # Fallback to latest active output when explicit stage snapshots are missing.
    if stage == "lm":
        candidates.append(results_dir / f"{stem}_de.{suffix}")
    return candidates


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_results_dir(results_dir: Path, stem: str) -> Path:
    baseline_here = _find_first_existing(
        _artifact_candidates(results_dir, stem, "baseline", "human")
    )
    if baseline_here is not None:
        return results_dir

    nested = results_dir / "logistic_growth"
    baseline_nested = _find_first_existing(
        _artifact_candidates(nested, stem, "baseline", "human")
    )
    if baseline_nested is not None:
        return nested
    return results_dir


def _render_stage_panel(ax, title: str, color: str, result, notes: list[str]):
    ax.axis("off")
    ax.text(
        0.5,
        0.95,
        title,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
        color=color,
    )

    if not result:
        ax.text(
            0.5,
            0.5,
            "Results not found",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
            color="gray",
            style="italic",
        )
        return

    eq = result.get("equation")
    if isinstance(eq, str):
        eq_str = eq.strip()
        if len(eq_str) > 84:
            eq_str = eq_str[:81] + "..."
        ax.text(
            0.5,
            0.76,
            eq_str,
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            family="monospace",
            wrap=True,
        )

    rms_train = result.get("rms_train", float("nan"))
    rms_val = result.get("rms_val", float("nan"))
    ax.text(
        0.1,
        0.50,
        f"RMS train: {rms_train:.3e}",
        fontsize=10,
        transform=ax.transAxes,
        family="monospace",
    )
    ax.text(
        0.1,
        0.42,
        f"RMS val:   {rms_val:.3e}",
        fontsize=10,
        transform=ax.transAxes,
        family="monospace",
    )

    y = 0.26
    for note in notes:
        ax.text(0.1, y, note, fontsize=9.5, transform=ax.transAxes)
        y -= 0.08


def plot_comprehensive_results(
    data_file: Path,
    results_dir: Path,
    output_file: Path,
    stem: str | None = None,
    show_plot: bool = False,
):
    """Create comprehensive comparison plots."""

    t, u = _load_time_and_state(data_file)
    stem = stem or data_file.stem
    results_dir = _resolve_results_dir(results_dir, stem)

    dudt_num = np.gradient(u, t)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    color_true = "red"
    color_baseline = "blue"
    color_heuristic = "green"
    color_lm = "purple"

    # Row 1: data and physical context
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, u, "k-", linewidth=1.5, label="Data", alpha=0.7)
    ax1.axhline(y=10.0, color="r", linestyle="--", alpha=0.5, label="K=10")
    ax1.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="u0=1")
    ax1.set_xlabel("Time t", fontsize=11)
    ax1.set_ylabel("Population u", fontsize=11)
    ax1.set_title("Logistic Growth Data", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    u_theory = np.linspace(0, 10.5, 200)
    dudt_theory = 0.5 * u_theory - 0.05 * u_theory**2
    ax2.plot(
        u_theory, dudt_theory, "r--", linewidth=2, label="True: 0.5u - 0.05u^2", alpha=0.7
    )
    ax2.plot(u, dudt_num, "k.", markersize=2, alpha=0.3, label="Numerical du/dt")
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax2.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
    ax2.plot(0, 0, "ro", markersize=8, label="Unstable (u=0)")
    ax2.plot(10, 0, "go", markersize=8, label="Stable (u=K)")
    ax2.set_xlabel("Population u", fontsize=11)
    ax2.set_ylabel("Growth rate du/dt", fontsize=11)
    ax2.set_title("Phase Portrait", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.text(
        0.5,
        0.9,
        "Ground Truth",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        transform=ax3.transAxes,
    )
    ax3.text(
        0.5,
        0.75,
        r"$\frac{du}{dt} = ru(1 - \frac{u}{K})$",
        ha="center",
        va="top",
        fontsize=14,
        transform=ax3.transAxes,
    )
    ax3.text(
        0.5,
        0.60,
        r"$\frac{du}{dt} = 0.5u - 0.05u^2$",
        ha="center",
        va="top",
        fontsize=13,
        transform=ax3.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax3.text(0.1, 0.40, "Parameters:", fontsize=11, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.1, 0.32, "r = 0.5   (growth rate)", fontsize=10, transform=ax3.transAxes, family="monospace")
    ax3.text(0.1, 0.24, "K = 10.0  (capacity)", fontsize=10, transform=ax3.transAxes, family="monospace")
    ax3.text(0.1, 0.12, "Expected coefficients:", fontsize=11, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.1, 0.04, "c1 = -0.500, c2 = 0.050, p = 2.0", fontsize=10, transform=ax3.transAxes, family="monospace")

    # Row 2: discovered equations
    baseline_human = _find_first_existing(_artifact_candidates(results_dir, stem, "baseline", "human"))
    heuristic_human = _find_first_existing(_artifact_candidates(results_dir, stem, "heuristic", "human"))
    selected_human = _find_first_existing(_artifact_candidates(results_dir, stem, "selected", "human"))
    lm_human = _find_first_existing(_artifact_candidates(results_dir, stem, "lm", "human"))
    final_human = selected_human or lm_human

    results_baseline = load_result(baseline_human)
    results_heuristic = load_result(heuristic_human)
    results_lm = load_result(final_human)

    ax4 = fig.add_subplot(gs[1, 0])
    _render_stage_panel(
        ax4,
        "Test 1: Baseline STLSQ",
        color_baseline,
        results_baseline,
        ["Reference sparse fit", "No nonlinear template"],
    )

    ax5 = fig.add_subplot(gs[1, 1])
    _render_stage_panel(
        ax5,
        "Test 2: Heuristic Template",
        color_heuristic,
        results_heuristic,
        ["Power template inserted", "p from heuristic initialization"],
    )

    ax6 = fig.add_subplot(gs[1, 2])
    final_title = "Test 3: Selected Final Model *" if selected_human is not None else "Test 3: LM-Optimized Template *"
    final_notes = (
        ["Selected by BIC + identifiability", "Compares linear vs nonlinear branch"]
        if selected_human is not None
        else ["Power template + LM over p", "Expected best validation RMS"]
    )
    _render_stage_panel(
        ax6,
        final_title,
        color_lm,
        results_lm,
        final_notes,
    )

    # Row 3: quantitative comparisons
    ax7 = fig.add_subplot(gs[2, 0])
    methods = []
    rms_trains = []
    rms_vals = []

    if results_baseline:
        methods.append("Baseline\nSTLSQ")
        rms_trains.append(results_baseline.get("rms_train", 0.0))
        rms_vals.append(results_baseline.get("rms_val", 0.0))
    if results_heuristic:
        methods.append("Heuristic\nTemplate")
        rms_trains.append(results_heuristic.get("rms_train", 0.0))
        rms_vals.append(results_heuristic.get("rms_val", 0.0))
    if results_lm:
        methods.append("Selected\nFinal" if selected_human is not None else "LM-Optimized\nTemplate")
        rms_trains.append(results_lm.get("rms_train", 0.0))
        rms_vals.append(results_lm.get("rms_val", 0.0))

    if methods:
        x = np.arange(len(methods))
        width = 0.35
        colors = [color_baseline, color_heuristic, color_lm][: len(methods)]
        ax7.bar(x - width / 2, rms_trains, width, label="Train RMS", color=colors, alpha=0.7)
        ax7.bar(x + width / 2, rms_vals, width, label="Val RMS", color=colors, alpha=0.4)
        ax7.set_ylabel("RMS Residual", fontsize=11)
        ax7.set_title("RMS Comparison (Lower is Better)", fontsize=12, fontweight="bold")
        ax7.set_xticks(x)
        ax7.set_xticklabels(methods, fontsize=9)
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis="y")
        ax7.set_yscale("log")
    else:
        ax7.text(0.5, 0.5, "No stage artifacts found", ha="center", va="center", transform=ax7.transAxes)
        ax7.set_axis_off()

    ax8 = fig.add_subplot(gs[2, 1])
    selected_json = _find_first_existing(_artifact_candidates(results_dir, stem, "selected", "json"))
    lm_json = _find_first_existing(_artifact_candidates(results_dir, stem, "lm", "json"))
    final_json = selected_json or lm_json
    meta_lm = load_metadata(final_json)
    vp_meta = _extract_varpro_metadata(meta_lm)
    p_init = _extract_power_param(vp_meta.get("template_params_init", {})) if vp_meta else None
    p_fit = _extract_power_param(vp_meta.get("template_params", {})) if vp_meta else None

    if p_init is not None and p_fit is not None:
        ax8.barh(
            ["p (init)", "p (LM-fit)", "p (true)"],
            [p_init, p_fit, 2.0],
            color=[color_heuristic, color_lm, color_true],
            alpha=0.7,
        )
        ax8.axvline(x=2.0, color="red", linestyle="--", linewidth=2, label="True value")
        ax8.set_xlabel("Exponent p", fontsize=11)
        ax8.set_title("Template Parameter Refinement", fontsize=12, fontweight="bold")
        ax8.grid(True, alpha=0.3, axis="x")
        ax8.legend(fontsize=9)
        ax8.text(p_init, 0, f" {p_init:.4f}", va="center", fontsize=9)
        ax8.text(p_fit, 1, f" {p_fit:.4f}", va="center", fontsize=9, fontweight="bold")
        ax8.text(2.0, 2, " 2.0000", va="center", fontsize=9)
    else:
        ax8.text(
            0.5,
            0.5,
            "Run LM stage with --save_json\nto show parameter refinement",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax8.transAxes,
            color="gray",
            style="italic",
        )

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    ax9.text(
        0.5,
        0.95,
        "Summary",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax9.transAxes,
    )

    summary_text = f"""
Data: {len(t)} points
Time: [{t.min():.1f}, {t.max():.1f}]

Ground Truth:
  du/dt = 0.5u - 0.05u^2
  c1 = -0.500
  c2 = 0.050
  p = 2.0

Expected Results:
  Test 1: baseline sparse fit
  Test 2: template, heuristic p
  Test 3: template + LM-refined p

Key Innovation:
  Phase 2 LM-over-psi optimization
  updates nonlinear template params
  while VarPro eliminates linear coeffs
"""
    ax9.text(0.05, 0.80, summary_text.strip(), va="top", fontsize=9, transform=ax9.transAxes, family="monospace")

    plt.suptitle(
        "Logistic Growth ODE Discovery - Phase 2 LM-over-psi Validation",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved comprehensive plot to: {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot logistic growth ODE discovery results")
    parser.add_argument(
        "--data",
        type=str,
        default=str(REPO_ROOT / "data" / "logistic_growth.csv"),
        help="Path to data file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(REPO_ROOT / "results" / "logistic_growth"),
        help="Results directory (or parent results directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "logistic_comparison.png"),
        help="Output plot file",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Artifact stem (defaults to --data filename stem)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot window (off by default for headless use)",
    )

    args = parser.parse_args()

    plot_comprehensive_results(
        Path(args.data),
        Path(args.results_dir),
        Path(args.output),
        stem=args.stem,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
