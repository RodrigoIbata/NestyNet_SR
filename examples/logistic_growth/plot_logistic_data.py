#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Simple visualization of logistic growth data and derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def _load_time_and_state(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load time/state arrays from current or legacy logistic CSV formats."""
    df = pd.read_csv(filepath)
    cols = list(df.columns)

    if "y" in df.columns:
        y_col = "y"
    elif "u" in df.columns:
        y_col = "u"
    else:
        raise ValueError(
            f"Could not find dependent-variable column in {filepath}. Expected one of: y, u. Found: {cols}"
        )

    if "t" in df.columns:
        t_col = "t"
    elif "x0" in df.columns:
        t_col = "x0"
    else:
        raise ValueError(
            f"Could not find time column in {filepath}. Expected one of: t, x0. Found: {cols}"
        )

    return df[t_col].to_numpy(), df[y_col].to_numpy()


def plot_logistic_data(
    filepath: Path,
    output_path: Path,
    show_derivatives: bool = True,
    show_plot: bool = False,
):
    """Plot logistic growth data and estimated derivatives."""

    t, u = _load_time_and_state(filepath)

    # Estimate derivative numerically
    if show_derivatives:
        dudt = np.gradient(u, t)

    # Create figure
    fig, axes = plt.subplots(2 if show_derivatives else 1, 1,
                             figsize=(10, 8 if show_derivatives else 5))
    if not show_derivatives:
        axes = [axes]

    # Plot 1: u(t)
    axes[0].plot(t, u, 'b-', linewidth=2, label='u(t)')
    axes[0].axhline(y=10.0, color='r', linestyle='--', alpha=0.5,
                   label='Carrying capacity K=10')
    axes[0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5,
                   label='Initial u₀=1')
    axes[0].set_xlabel('Time t', fontsize=12)
    axes[0].set_ylabel('Population u', fontsize=12)
    axes[0].set_title('Logistic Growth: u(t)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    if show_derivatives:
        # Plot 2: du/dt vs u (phase portrait)
        axes[1].plot(u, dudt, 'b-', linewidth=2, label='Numerical du/dt')

        # Overlay theoretical curve: du/dt = 0.5*u - 0.05*u²
        u_theory = np.linspace(0, 10.5, 200)
        dudt_theory = 0.5 * u_theory - 0.05 * u_theory**2

        axes[1].plot(u_theory, dudt_theory, 'r--', linewidth=2,
                    label='Theory: 0.5u - 0.05u²', alpha=0.7)
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Population u', fontsize=12)
        axes[1].set_ylabel('Growth rate du/dt', fontsize=12)
        axes[1].set_title('Phase Portrait: du/dt vs u', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)

        # Add annotation for maximum growth rate
        u_max_growth = 5.0  # At u = K/2
        dudt_max = 0.5 * u_max_growth - 0.05 * u_max_growth**2
        axes[1].plot(u_max_growth, dudt_max, 'go', markersize=10,
                    label=f'Max growth at u=K/2={u_max_growth}')
        axes[1].text(u_max_growth + 0.3, dudt_max, f'Max: {dudt_max:.3f}',
                    fontsize=10, va='center')

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize logistic growth data'
    )
    parser.add_argument('--filepath', type=str,
                       default=str(REPO_ROOT / "data" / "logistic_growth.csv"),
                       help='Path to CSV data file')
    parser.add_argument('--no_derivatives', action='store_true',
                       help='Do not show derivative plot')
    parser.add_argument('--output', type=str,
                       default=str(SCRIPT_DIR / "logistic_growth_data.png"),
                       help='Output plot file path')
    parser.add_argument('--show', action='store_true',
                       help='Display plot window (off by default for headless use)')

    args = parser.parse_args()

    if not Path(args.filepath).exists():
        print(f"❌ File not found: {args.filepath}")
        print("   Run: python generate_logistic_growth.py")
        return 1

    plot_logistic_data(
        Path(args.filepath),
        Path(args.output),
        show_derivatives=not args.no_derivatives,
        show_plot=args.show,
    )

if __name__ == '__main__':
    main()
