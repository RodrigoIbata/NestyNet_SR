#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Visualize Lane-Emden ODE discovery results across baseline/template stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def load_data(data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Lane-Emden data."""
    df = pd.read_csv(data_file)
    xi = df["x0"].values
    theta = df["y"].values
    return xi, theta


def _safe_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _extract_equation_line(content: str) -> str | None:
    for line in content.splitlines():
        line_s = line.strip()
        if "=" not in line_s:
            continue
        if "u_x0x0" in line_s or "θ_x0x0" in line_s:
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
            result["rms_train"] = _safe_float(line.split(":", 1)[1])
        if "RMS (val):" in line:
            result["rms_val"] = _safe_float(line.split(":", 1)[1])

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
        return _safe_float(params["p"], default=float("nan"))
    for key, value in params.items():
        if str(key).startswith("p"):
            val = _safe_float(value, default=float("nan"))
            if np.isfinite(val):
                return val
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
    # Support passing either ".../results/lane_emden" or ".../results".
    baseline_here = _find_first_existing(_artifact_candidates(results_dir, stem, "baseline", "human"))
    if baseline_here is not None:
        return results_dir
    nested = results_dir / "lane_emden"
    baseline_nested = _find_first_existing(_artifact_candidates(nested, stem, "baseline", "human"))
    if baseline_nested is not None:
        return nested
    return results_dir


def plot_comprehensive_results(
    data_file: Path,
    results_dir: Path,
    output_file: Path,
    n_true: float = 1.0,
    stem: str | None = None,
    show_plot: bool = False,
):
    """Create comprehensive comparison plots."""

    # Load data
    xi, theta = load_data(data_file)
    stem = stem or data_file.stem
    results_dir = _resolve_results_dir(results_dir, stem)

    # Theoretical solution (n=1)
    if n_true == 1:
        theta_theory = np.where(np.abs(xi) < 1e-10, 1.0, np.sin(xi) / xi)
    elif n_true == 0:
        theta_theory = 1.0 - xi**2 / 6.0
    elif n_true == 5:
        theta_theory = 1.0 / np.sqrt(1.0 + xi**2 / 3.0)
    else:
        theta_theory = None

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Colors
    color_true = "red"
    color_baseline = "blue"
    color_heuristic = "green"
    color_lm = "purple"

    # ============================================================
    # Row 1: Data and solutions
    # ============================================================

    # Plot 1: θ(ξ) solution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(xi, theta, 'k-', linewidth=1.5, label='Data', alpha=0.7)
    if theta_theory is not None:
        ax1.plot(xi, theta_theory, 'r--', linewidth=2,
                label=f'Theory (n={n_true})', alpha=0.7)
    ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='θ(0)=1')
    ax1.set_xlabel('Dimensionless radius ξ', fontsize=11)
    ax1.set_ylabel('Dimensionless density θ', fontsize=11)
    ax1.set_title('Lane-Emden Solution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Plot 2: Derivatives
    ax2 = fig.add_subplot(gs[0, 1])
    dtheta = np.gradient(theta, xi)
    d2theta = np.gradient(dtheta, xi)

    ax2.plot(xi, dtheta, 'b-', linewidth=1.5, label="θ'", alpha=0.7)
    ax2.plot(xi, d2theta, 'r-', linewidth=1.5, label="θ''", alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Dimensionless radius ξ', fontsize=11)
    ax2.set_ylabel('Derivatives', fontsize=11)
    ax2.set_title('Numerical Derivatives', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Plot 3: Ground truth equation
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.text(0.5, 0.9, 'Ground Truth', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax3.transAxes)

    # Equation in LaTeX
    eq_text = r'$\frac{1}{\xi^2}\frac{d}{d\xi}\left(\xi^2\frac{d\theta}{d\xi}\right) + \theta^n = 0$'
    ax3.text(0.5, 0.75, eq_text, ha='center', va='top',
             fontsize=12, transform=ax3.transAxes)

    eq_text2 = r'$\theta_{\xi\xi} + \frac{2}{\xi}\theta_\xi + \theta^n = 0$'
    ax3.text(0.5, 0.60, eq_text2, ha='center', va='top',
             fontsize=12, transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.text(0.1, 0.45, 'Parameters:', fontsize=11, fontweight='bold',
             transform=ax3.transAxes)
    ax3.text(0.1, 0.37, f'n = {n_true}  (polytropic index)', fontsize=10,
             transform=ax3.transAxes, family='monospace')

    ax3.text(0.1, 0.25, 'Expected coefficients:', fontsize=11, fontweight='bold',
             transform=ax3.transAxes)
    ax3.text(0.1, 0.17, 'c₁ = 2.0  (for ξ⁻¹θ_ξ)', fontsize=10,
             transform=ax3.transAxes, family='monospace')
    ax3.text(0.1, 0.09, 'c₂ = 1.0  (for θⁿ)', fontsize=10,
             transform=ax3.transAxes, family='monospace')
    ax3.text(0.1, 0.01, f'n  = {n_true}', fontsize=10,
             transform=ax3.transAxes, family='monospace')

    # ============================================================
    # Row 2: Results comparison
    # ============================================================

    # Try to load results
    baseline_human = _find_first_existing(_artifact_candidates(results_dir, stem, "baseline", "human"))
    heuristic_human = _find_first_existing(_artifact_candidates(results_dir, stem, "heuristic", "human"))
    selected_human = _find_first_existing(_artifact_candidates(results_dir, stem, "selected", "human"))
    lm_human = _find_first_existing(_artifact_candidates(results_dir, stem, "lm", "human"))
    final_human = selected_human or lm_human

    results_baseline = load_result(baseline_human)
    results_heuristic = load_result(heuristic_human)
    results_lm = load_result(final_human)

    # Plot 4: Baseline STLSQ
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    ax4.text(0.5, 0.95, 'Test 1: Baseline STLSQ', ha='center', va='top',
             fontsize=12, fontweight='bold', transform=ax4.transAxes,
             color=color_baseline)

    if results_baseline:
        if results_baseline.get("equation"):
            # Truncate long equations
            eq_str = results_baseline["equation"].strip()
            if len(eq_str) > 80:
                eq_str = eq_str[:77] + '...'
            ax4.text(0.5, 0.75, eq_str, ha='center', va='top',
                     fontsize=8, transform=ax4.transAxes, family='monospace',
                     wrap=True)

        rms_train = results_baseline.get('rms_train', float('nan'))
        rms_val = results_baseline.get('rms_val', float('nan'))
        ax4.text(0.1, 0.50, f'RMS train: {rms_train:.3e}', fontsize=10,
                 transform=ax4.transAxes, family='monospace')
        ax4.text(0.1, 0.42, f'RMS val:   {rms_val:.3e}', fontsize=10,
                 transform=ax4.transAxes, family='monospace')

        ax4.text(0.1, 0.25, 'Default STLSQ library', fontsize=10,
                 transform=ax4.transAxes)
        ax4.text(0.1, 0.17, 'May or may not find θⁿ', fontsize=9,
                 transform=ax4.transAxes, style='italic')
    else:
        ax4.text(0.5, 0.5, 'Results not found', ha='center', va='center',
                 fontsize=11, transform=ax4.transAxes, color='gray', style='italic')

    # Plot 5: Heuristic template
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    ax5.text(0.5, 0.95, 'Test 2: Heuristic Template', ha='center', va='top',
             fontsize=12, fontweight='bold', transform=ax5.transAxes,
             color=color_heuristic)

    if results_heuristic:
        if results_heuristic.get("equation"):
            eq_str = results_heuristic["equation"].strip()
            if len(eq_str) > 80:
                eq_str = eq_str[:77] + '...'
            ax5.text(0.5, 0.75, eq_str, ha='center', va='top',
                     fontsize=8, transform=ax5.transAxes, family='monospace',
                     wrap=True)

        rms_train = results_heuristic.get('rms_train', float('nan'))
        rms_val = results_heuristic.get('rms_val', float('nan'))
        ax5.text(0.1, 0.50, f'RMS train: {rms_train:.3e}', fontsize=10,
                 transform=ax5.transAxes, family='monospace')
        ax5.text(0.1, 0.42, f'RMS val:   {rms_val:.3e}', fontsize=10,
                 transform=ax5.transAxes, family='monospace')

        ax5.text(0.1, 0.25, '✓ Discovers θⁿ term', fontsize=10,
                 transform=ax5.transAxes, color='green', fontweight='bold')
        ax5.text(0.1, 0.17, 'n from log-log regression', fontsize=9,
                 transform=ax5.transAxes)
        ax5.text(0.1, 0.10, '(heuristic, not optimized)', fontsize=9,
                 transform=ax5.transAxes, style='italic')
    else:
        ax5.text(0.5, 0.5, 'Results not found', ha='center', va='center',
                 fontsize=11, transform=ax5.transAxes, color='gray', style='italic')

    # Plot 6: LM-optimized template
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    final_title = 'Test 3: Selected Final Model *' if selected_human is not None else 'Test 3: LM-Optimized Template *'
    ax6.text(0.5, 0.95, final_title, ha='center', va='top',
             fontsize=12, fontweight='bold', transform=ax6.transAxes,
             color=color_lm)

    if results_lm:
        if results_lm.get("equation"):
            eq_str = results_lm["equation"].strip()
            if len(eq_str) > 80:
                eq_str = eq_str[:77] + '...'
            ax6.text(0.5, 0.75, eq_str, ha='center', va='top',
                     fontsize=8, transform=ax6.transAxes, family='monospace',
                     wrap=True)

        rms_train = results_lm.get('rms_train', float('nan'))
        rms_val = results_lm.get('rms_val', float('nan'))
        ax6.text(0.1, 0.50, f'RMS train: {rms_train:.3e}', fontsize=10,
                 transform=ax6.transAxes, family='monospace')
        ax6.text(0.1, 0.42, f'RMS val:   {rms_val:.3e}', fontsize=10,
                 transform=ax6.transAxes, family='monospace')

        if selected_human is not None:
            ax6.text(0.1, 0.25, '✓ Selected by BIC + identifiability', fontsize=9.5,
                     transform=ax6.transAxes, color='purple', fontweight='bold')
            ax6.text(0.1, 0.17, 'Compares linear vs nonlinear branch', fontsize=9,
                     transform=ax6.transAxes)
            ax6.text(0.1, 0.10, 'then exports selected model', fontsize=9,
                     transform=ax6.transAxes, style='italic')
        else:
            ax6.text(0.1, 0.25, '✓✓ n optimized by LM!', fontsize=10,
                     transform=ax6.transAxes, color='purple', fontweight='bold')
            ax6.text(0.1, 0.17, 'Nonlinear params refined', fontsize=9,
                     transform=ax6.transAxes)
            ax6.text(0.1, 0.10, 'while β eliminated by VarPro', fontsize=9,
                     transform=ax6.transAxes, style='italic')
    else:
        ax6.text(0.5, 0.5, 'Results not found', ha='center', va='center',
                 fontsize=11, transform=ax6.transAxes, color='gray', style='italic')

    # ============================================================
    # Row 3: Quantitative comparison
    # ============================================================

    # Plot 7: RMS comparison
    ax7 = fig.add_subplot(gs[2, 0])

    methods = []
    rms_trains = []
    rms_vals = []

    if results_baseline:
        methods.append('Baseline\nSTLSQ')
        rms_trains.append(results_baseline.get('rms_train', 0))
        rms_vals.append(results_baseline.get('rms_val', 0))

    if results_heuristic:
        methods.append('Heuristic\nTemplate')
        rms_trains.append(results_heuristic.get('rms_train', 0))
        rms_vals.append(results_heuristic.get('rms_val', 0))

    if results_lm:
        methods.append('LM-Optimized\nTemplate')
        rms_trains.append(results_lm.get('rms_train', 0))
        rms_vals.append(results_lm.get('rms_val', 0))

    if methods:
        x = np.arange(len(methods))
        width = 0.35

        ax7.bar(x - width/2, rms_trains, width, label='Train RMS',
                color=[color_baseline, color_heuristic, color_lm][:len(methods)], alpha=0.7)
        ax7.bar(x + width/2, rms_vals, width, label='Val RMS',
                color=[color_baseline, color_heuristic, color_lm][:len(methods)], alpha=0.4)

        ax7.set_ylabel('RMS Residual', fontsize=11)
        ax7.set_title('RMS Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(methods, fontsize=9)
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_yscale('log')

    # Plot 8: Polytropic index comparison (if metadata available)
    ax8 = fig.add_subplot(gs[2, 1])

    selected_json = _find_first_existing(_artifact_candidates(results_dir, stem, "selected", "json"))
    lm_json = _find_first_existing(_artifact_candidates(results_dir, stem, "lm", "json"))
    final_json = selected_json or lm_json
    meta_lm = load_metadata(final_json)
    vp_meta = _extract_varpro_metadata(meta_lm)

    if vp_meta:

        # Extract template params
        n_init = _extract_power_param(vp_meta.get("template_params_init", {}))
        n_fit = _extract_power_param(vp_meta.get("template_params", {}))

        if n_init is not None and n_fit is not None and np.isfinite(n_init) and np.isfinite(n_fit):
            ax8.barh(['n (init)', 'n (LM-fit)', 'n (true)'],
                    [n_init, n_fit, n_true],
                    color=[color_heuristic, color_lm, color_true], alpha=0.7)
            ax8.axvline(x=n_true, color='red', linestyle='--', linewidth=2, label='True value')
            ax8.set_xlabel('Polytropic Index n', fontsize=11)
            ax8.set_title('Template Parameter Refinement', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='x')
            ax8.legend(fontsize=9)

            # Add text annotations
            ax8.text(n_init, 0, f' {n_init:.4f}', va='center', fontsize=9)
            ax8.text(n_fit, 1, f' {n_fit:.4f}', va='center', fontsize=9, fontweight='bold')
            ax8.text(n_true, 2, f' {n_true:.4f}', va='center', fontsize=9)
        else:
            ax8.text(0.5, 0.5, 'Metadata not available', ha='center', va='center',
                     fontsize=11, transform=ax8.transAxes, color='gray', style='italic')
    else:
        ax8.text(0.5, 0.5, 'Run Test 3 to see\nparameter refinement', ha='center', va='center',
                 fontsize=11, transform=ax8.transAxes, color='gray', style='italic')

    # Plot 9: Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    ax9.text(0.5, 0.95, 'Summary', ha='center', va='top',
             fontsize=12, fontweight='bold', transform=ax9.transAxes)

    summary_text = f"""
Data: {len(xi)} points
Domain: [{xi.min():.1f}, {xi.max():.1f}]

Ground Truth (n={n_true}):
  θ_ξξ + (2/ξ)θ_ξ + θ^{n_true} = 0
  c₁ = 2.0
  c₂ = 1.0
  n = {n_true}
"""

    if n_true == 1:
        summary_text += "\n  Solution: θ = sin(ξ)/ξ"

    summary_text += """

Expected Results:
  Test 1: May find θⁿ
  Test 2: n ≈ heuristic init
  Test 3: n ≈ {:.1f} (LM-refined)

Key Innovation:
  Phase 2 LM-over-ψ on
  2nd-order ODE with
  nonlinear polytropic
  index discovery
""".format(n_true)

    ax9.text(0.05, 0.80, summary_text.strip(), va='top',
             fontsize=9, transform=ax9.transAxes, family='monospace')

    # Save figure
    plt.suptitle('Lane-Emden ODE Discovery - Phase 2 LM-over-ψ Validation',
                 fontsize=16, fontweight='bold', y=0.98)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comprehensive plot to: {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description='Plot Lane-Emden ODE discovery results'
    )
    parser.add_argument('--data', type=str,
                       default=str(REPO_ROOT / "data" / "lane_emden.csv"),
                       help='Path to data file')
    parser.add_argument('--results_dir', type=str,
                       default=str(REPO_ROOT / "results" / "lane_emden"),
                       help='Results directory (or parent results directory)')
    parser.add_argument('--output', type=str,
                       default=str(SCRIPT_DIR / "lane_emden_comparison.png"),
                       help='Output plot file')
    parser.add_argument('--n', type=float, default=1.0,
                       help='Polytropic index (ground truth)')
    parser.add_argument('--stem', type=str, default=None,
                       help='Artifact stem (defaults to --data filename stem)')
    parser.add_argument('--show', action='store_true',
                       help='Display plot window (off by default for headless use)')

    args = parser.parse_args()

    plot_comprehensive_results(
        Path(args.data),
        Path(args.results_dir),
        Path(args.output),
        n_true=args.n,
        stem=args.stem,
        show_plot=args.show,
    )

if __name__ == '__main__':
    main()
