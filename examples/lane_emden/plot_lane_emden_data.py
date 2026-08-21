#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Simple visualization of Lane-Emden data and derivatives.

Helps verify data quality before running ODE discovery.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

def plot_lane_emden_data(
    filepath,
    output_path,
    show_derivatives=True,
    n_true=1.0,
    show_plot=False,
):
    """Plot Lane-Emden data and estimated derivatives."""

    df = pd.read_csv(filepath)
    xi = df['x0'].values
    theta = df['y'].values

    # Estimate derivatives numerically
    if show_derivatives:
        dtheta = np.gradient(theta, xi)
        d2theta = np.gradient(dtheta, xi)

    # Create figure
    fig, axes = plt.subplots(2 if show_derivatives else 1, 2,
                             figsize=(14, 8 if show_derivatives else 5))
    if not show_derivatives:
        axes = [axes]

    # Plot 1: θ(ξ)
    axes[0][0].plot(xi, theta, 'b-', linewidth=2, label='θ(ξ)')
    axes[0][0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5,
                      label='θ(0)=1')
    axes[0][0].axhline(y=0.0, color='k', linestyle='-', linewidth=0.5)
    axes[0][0].set_xlabel('Dimensionless radius ξ', fontsize=12)
    axes[0][0].set_ylabel('Dimensionless density θ', fontsize=12)
    axes[0][0].set_title('Lane-Emden Solution', fontsize=14, fontweight='bold')
    axes[0][0].grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=10)

    # Plot 2: Theoretical comparison
    axes[0][1].plot(xi, theta, 'b-', linewidth=2, label='Data', alpha=0.7)

    # Overlay theoretical solution if n=0,1,5
    if n_true == 0:
        theta_theory = 1.0 - xi**2 / 6.0
        axes[0][1].plot(xi, theta_theory, 'r--', linewidth=2,
                       label='Theory: 1 - ξ²/6', alpha=0.7)
    elif n_true == 1:
        theta_theory = np.where(np.abs(xi) < 1e-10, 1.0, np.sin(xi) / xi)
        axes[0][1].plot(xi, theta_theory, 'r--', linewidth=2,
                       label='Theory: sin(ξ)/ξ', alpha=0.7)
    elif n_true == 5:
        theta_theory = 1.0 / np.sqrt(1.0 + xi**2 / 3.0)
        axes[0][1].plot(xi, theta_theory, 'r--', linewidth=2,
                       label='Theory: (1+ξ²/3)^(-1/2)', alpha=0.7)

    axes[0][1].set_xlabel('Dimensionless radius ξ', fontsize=12)
    axes[0][1].set_ylabel('Dimensionless density θ', fontsize=12)
    axes[0][1].set_title(f'n={n_true} Polytrope', fontsize=14, fontweight='bold')
    axes[0][1].grid(True, alpha=0.3)
    axes[0][1].legend(fontsize=10)

    if show_derivatives:
        # Plot 3: First derivative
        axes[1][0].plot(xi, dtheta, 'g-', linewidth=2, label="θ' (numerical)")
        axes[1][0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Theoretical derivative for n=1
        if n_true == 1:
            dtheta_theory = (xi * np.cos(xi) - np.sin(xi)) / xi**2
            dtheta_theory = np.where(np.abs(xi) < 1e-10, 0.0, dtheta_theory)
            axes[1][0].plot(xi, dtheta_theory, 'r--', linewidth=2,
                           label="θ' (theory)", alpha=0.7)

        axes[1][0].set_xlabel('Dimensionless radius ξ', fontsize=12)
        axes[1][0].set_ylabel("dθ/dξ", fontsize=12)
        axes[1][0].set_title('First Derivative', fontsize=14, fontweight='bold')
        axes[1][0].grid(True, alpha=0.3)
        axes[1][0].legend(fontsize=10)

        # Plot 4: Residual check
        # Calculate: θ'' + (2/ξ)θ' + θⁿ (should be ~0)
        residual = d2theta + (2.0 / (xi + 1e-10)) * dtheta + theta**n_true
        axes[1][1].plot(xi, residual, 'purple', linewidth=2)
        axes[1][1].axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axes[1][1].set_xlabel('Dimensionless radius ξ', fontsize=12)
        axes[1][1].set_ylabel(f"θ'' + (2/ξ)θ' + θ^{n_true}", fontsize=12)
        axes[1][1].set_title('ODE Residual Check', fontsize=14, fontweight='bold')
        axes[1][1].grid(True, alpha=0.3)
        axes[1][1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # Add RMS text
        rms = np.sqrt(np.mean(residual**2))
        axes[1][1].text(0.05, 0.95, f'RMS: {rms:.2e}',
                       transform=axes[1][1].transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Lane-Emden data'
    )
    parser.add_argument('--filepath', type=str,
                       default=str(REPO_ROOT / "data" / "lane_emden.csv"),
                       help='Path to CSV data file')
    parser.add_argument('--n', type=float, default=1.0,
                       help='Polytropic index (for theoretical comparison)')
    parser.add_argument('--output', type=str,
                       default=str(SCRIPT_DIR / "lane_emden_data.png"),
                       help='Output plot file path')
    parser.add_argument('--no_derivatives', action='store_true',
                       help='Do not show derivative plots')
    parser.add_argument('--show', action='store_true',
                       help='Display plot window (off by default for headless use)')

    args = parser.parse_args()

    if not Path(args.filepath).exists():
        print(f"❌ File not found: {args.filepath}")
        print("   Run: python generate_lane_emden.py")
        return 1

    plot_lane_emden_data(args.filepath,
                        args.output,
                        show_derivatives=not args.no_derivatives,
                        n_true=args.n,
                        show_plot=args.show)

if __name__ == '__main__':
    main()
