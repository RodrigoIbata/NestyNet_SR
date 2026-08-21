#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Generate synthetic logistic growth data for ODE discovery testing.

Logistic Growth Equation:
    du/dt = r*u*(1 - u/K)

Expanded form:
    du/dt = r*u - (r/K)*u²

This is a perfect test case for PowerLawTemplate discovery of the u² term.

Ground Truth Parameters:
    r = 0.5   (intrinsic growth rate)
    K = 10.0  (carrying capacity)

Expected Discovery:
    u_t + c1*u + c2*u^p = 0
    where:
        c1 ≈ -0.5  (= -r)
        c2 ≈ 0.05  (= r/K)
        p  ≈ 2.0   (power law exponent)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

def logistic_ode(t, u, r, K):
    """Logistic growth ODE: du/dt = r*u*(1 - u/K)"""
    return [r * u[0] * (1.0 - u[0] / K)]

def generate_logistic_data(
    r: float = 0.5,
    K: float = 10.0,
    u0: float = 1.0,
    t_max: float = 20.0,
    n_points: int = 200,
    noise_level: float = 0.0,
    seed: int = 42
):
    """Generate synthetic logistic growth data.

    Parameters
    ----------
    r : float
        Intrinsic growth rate
    K : float
        Carrying capacity
    u0 : float
        Initial population
    t_max : float
        Maximum time
    n_points : int
        Number of time points
    noise_level : float
        Relative noise level (e.g., 0.01 = 1% noise)
    seed : int
        Random seed for noise

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['t', 'y']
    """
    np.random.seed(seed)

    # Solve ODE
    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(
        logistic_ode,
        [0, t_max],
        [u0],
        args=(r, K),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )

    t = sol.t
    u = sol.y[0]

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * u.std(), size=u.shape)
        u = u + noise

    # Create DataFrame with coordinate type-aware column names
    # Use 't' for time coordinate (will be auto-detected by PhysDataset)
    df = pd.DataFrame({
        't': t,      # independent variable (time) - will auto-detect as time coordinate
        'y': u       # dependent variable (population)
    })

    return df, {'r': r, 'K': K, 'u0': u0, 'c1_true': -r, 'c2_true': r/K}

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic logistic growth data for ODE discovery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Physics parameters
    parser.add_argument('--r', type=float, default=0.5,
                       help='Growth rate')
    parser.add_argument('--K', type=float, default=10.0,
                       help='Carrying capacity')
    parser.add_argument('--u0', type=float, default=1.0,
                       help='Initial population')

    # Data generation
    parser.add_argument('--t_max', type=float, default=20.0,
                       help='Maximum time')
    parser.add_argument('--n_points', type=int, default=2000,
                       help='Number of time points')
    parser.add_argument('--noise', type=float, default=0.0,
                       help='Relative noise level (0.01 = 1%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output
    parser.add_argument('--output', type=str, default=str(REPO_ROOT / "data" / "logistic_growth.csv"),
                       help='Output CSV file path')

    args = parser.parse_args()

    # Generate data
    print("=" * 70)
    print("Generating Logistic Growth Data")
    print("=" * 70)
    print("\nGround Truth Parameters:")
    print(f"  r (growth rate):     {args.r}")
    print(f"  K (carrying cap):    {args.K}")
    print(f"  u0 (initial):        {args.u0}")
    print("\nExpected Coefficients:")
    print(f"  c1 = -r:             {-args.r:.6f}")
    print(f"  c2 = r/K:            {args.r/args.K:.6f}")
    print("  p (power):           2.000000")
    print("\nData Settings:")
    print(f"  Time span:           [0, {args.t_max}]")
    print(f"  Points:              {args.n_points}")
    print(f"  Noise level:         {args.noise*100:.2f}%")
    print(f"  Random seed:         {args.seed}")

    df, params = generate_logistic_data(
        r=args.r,
        K=args.K,
        u0=args.u0,
        t_max=args.t_max,
        n_points=args.n_points,
        noise_level=args.noise,
        seed=args.seed
    )

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} points to: {output_path}")

    # Save metadata
    meta_path = output_path.with_suffix('.meta.txt')
    with open(meta_path, 'w') as f:
        f.write("Logistic Growth Data - Ground Truth\n")
        f.write("=" * 50 + "\n\n")
        f.write("Data Format:\n")
        f.write("  Columns: 't' (time coordinate), 'y' (population)\n")
        f.write("  Note: 't' column will be auto-detected as time coordinate\n\n")
        f.write("Equation: du/dt = r*u*(1 - u/K)\n")
        f.write("Expanded: du/dt = r*u - (r/K)*u²\n\n")
        f.write("Parameters:\n")
        f.write(f"  r  = {params['r']}\n")
        f.write(f"  K  = {params['K']}\n")
        f.write(f"  u0 = {params['u0']}\n\n")
        f.write("Expected Discovery (implicit form):\n")
        f.write("  u_t + c1*u + c2*u^p = 0\n\n")
        f.write("Expected Coefficients:\n")
        f.write(f"  c1 = {params['c1_true']:.6f}\n")
        f.write(f"  c2 = {params['c2_true']:.6f}\n")
        f.write("  p  = 2.000000\n")

    print(f"✓ Saved metadata to: {meta_path}")

    # Print statistics
    print("\nData Statistics:")
    print(f"  u range:             [{df['y'].min():.4f}, {df['y'].max():.4f}]")
    print(f"  u mean:              {df['y'].mean():.4f}")
    print(f"  u std:               {df['y'].std():.4f}")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("\n1. Run ODE discovery WITHOUT templates (baseline):")
    print(f"   python run_de.py --filepath {output_path} --order_candidates 1")
    print("\n   Note: x_axis will auto-detect from 't' column (no --x_axis needed)")
    print("\n2. Run with PowerLaw template (heuristic init only):")
    print(f"   python run_de.py --filepath {output_path} --order_candidates 1 \\")
    print("       --varpro --varpro_templates power")
    print("\n3. Run with PowerLaw template + LM optimization over ψ:")
    print(f"   python run_de.py --filepath {output_path} --order_candidates 1 \\")
    print("       --varpro --varpro_templates power \\")
    print("       --template_lm --template_lm_epochs 200")
    print("\n   Expected: Discover p ≈ 2.0, c1 ≈ {:.4f}, c2 ≈ {:.4f}".format(
        params['c1_true'], params['c2_true']))
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
