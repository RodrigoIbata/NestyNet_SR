#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Generate synthetic Lane-Emden equation data for ODE discovery testing.

Lane-Emden Equation (Stellar Structure):
    (1/ξ²) d/dξ(ξ² dθ/dξ) + θⁿ = 0

Expanded form (2nd-order ODE):
    θ_ξξ + (2/ξ) θ_ξ + θⁿ = 0

Boundary conditions:
    θ(0) = 1, θ'(0) = 0

This is a perfect test case for PowerLawTemplate discovery of the polytropic index n.

Analytical Solutions:
    n=0: θ(ξ) = 1 - ξ²/6
    n=1: θ(ξ) = sin(ξ)/ξ
    n=5: Complex analytical form (Emden-Chandrasekhar solution)

Expected Discovery:
    θ_ξξ + c1*(1/ξ)*θ_ξ + c2*θⁿ = 0
    where:
        c1 ≈ 2.0   (coefficient of (1/ξ)*θ_ξ term)
        c2 ≈ 1.0   (coefficient of θⁿ term)
        n = polytropic index (to be discovered)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

def lane_emden_analytical(xi, n):
    """Analytical solutions for specific polytropic indices.

    Parameters
    ----------
    xi : array
        Dimensionless radius
    n : float
        Polytropic index

    Returns
    -------
    theta : array
        Dimensionless density
    """
    if n == 0:
        # Linear: θ = 1 - ξ²/6
        theta = 1.0 - xi**2 / 6.0
    elif n == 1:
        # Isothermal: θ = sin(ξ)/ξ
        # Handle ξ=0 singularity with limit
        theta = np.where(np.abs(xi) < 1e-10, 1.0 - xi**2/6.0, np.sin(xi) / xi)
    elif n == 5:
        # Emden-Chandrasekhar solution (complex)
        # Using approximation: θ ≈ (1 + ξ²/3)^(-1/2)
        theta = 1.0 / np.sqrt(1.0 + xi**2 / 3.0)
    else:
        raise ValueError(f"Analytical solution only available for n=0,1,5. Got n={n}")

    return theta

def lane_emden_ode(xi, y, n):
    """Lane-Emden ODE for numerical integration.

    dy/dξ = [θ', (θ')']
    where (θ')' = -(2/ξ) θ' - θⁿ

    To handle singularity at ξ=0, use series expansion:
    θ(ξ) ≈ 1 - ξ²/6 + n*ξ⁴/120 + ...
    """
    theta, dtheta = y

    # Handle ξ=0 singularity using L'Hôpital's rule:
    # lim(ξ→0) (2/ξ)*θ' = lim(ξ→0) 2*θ' / ξ = lim(ξ→0) 2*θ'' = -2*θⁿ(0) = -2
    if np.abs(xi) < 1e-10:
        d2theta = -(n + 1.0) / 3.0  # From series expansion
    else:
        d2theta = -(2.0 / xi) * dtheta - theta**n

    return [dtheta, d2theta]

def generate_lane_emden_data(
    n: float = 1.0,
    xi_min: float = 0.2,
    xi_max: float = 3.0,
    n_points: int = 2000,
    use_analytical: bool = True,
    noise_level: float = 0.0,
    seed: int = 42
):
    """Generate synthetic Lane-Emden data.

    Parameters
    ----------
    n : float
        Polytropic index
    xi_min : float
        Minimum dimensionless radius (> 0)
    xi_max : float
        Maximum dimensionless radius
    n_points : int
        Number of grid points
    use_analytical : bool
        Use analytical solution if available (n=0,1,5)
    noise_level : float
        Relative noise level (e.g., 0.01 = 1% noise)
    seed : int
        Random seed for noise

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['y', 'x0']
    params : dict
        Ground truth parameters
    """
    np.random.seed(seed)

    if xi_min <= 0.0:
        raise ValueError(f"xi_min must be > 0. Got xi_min={xi_min}")
    if xi_max <= xi_min:
        raise ValueError(f"xi_max must be > xi_min. Got xi_min={xi_min}, xi_max={xi_max}")

    # Keep data away from ξ=0 singularity for stable derivative discovery.
    xi = np.linspace(xi_min, xi_max, n_points)

    # Generate solution
    if use_analytical and n in [0, 1, 5]:
        theta = lane_emden_analytical(xi, n)
    else:
        # Numerical integration
        # Initial conditions: θ(0) = 1, θ'(0) = 0
        # Use series expansion for small ξ: θ ≈ 1 - ξ²/6
        xi_start = max(1e-4, xi_min)
        theta_start = 1.0 - xi_start**2 / 6.0
        dtheta_start = -xi_start / 3.0

        sol = solve_ivp(
            lane_emden_ode,
            [xi_start, xi_max],
            [theta_start, dtheta_start],
            args=(n,),
            t_eval=xi,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )

        theta = sol.y[0]

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * theta.std(), size=theta.shape)
        theta = theta + noise

    # Create DataFrame with ODE discovery column names
    df = pd.DataFrame({
        'y': theta,     # θ (dimensionless density)
        'x0': xi        # ξ (dimensionless radius)
    })

    return df, {
        'n': n,
        'c1_true': 2.0,      # Coefficient of (1/ξ)*θ_ξ
        'c2_true': 1.0,      # Coefficient of θⁿ
        'method': 'analytical' if (use_analytical and n in [0, 1, 5]) else 'numerical'
    }

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic Lane-Emden data for ODE discovery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Physics parameters
    parser.add_argument('--n', type=float, default=1.0,
                       help='Polytropic index (0, 1, or 5 for analytical)')
    parser.add_argument('--xi_min', type=float, default=0.2,
                       help='Minimum dimensionless radius (>0, avoids ξ=0 singularity)')
    parser.add_argument('--xi_max', type=float, default=3.0,
                       help='Maximum dimensionless radius')

    # Data generation
    parser.add_argument('--n_points', type=int, default=2000,
                       help='Number of grid points')
    parser.add_argument('--numerical', action='store_true',
                       help='Force numerical integration (even for n=0,1,5)')
    parser.add_argument('--noise', type=float, default=0.0,
                       help='Relative noise level (0.01 = 1%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output
    parser.add_argument('--output', type=str, default=str(REPO_ROOT / "data" / "lane_emden.csv"),
                       help='Output CSV file path')

    args = parser.parse_args()

    # Generate data
    print("=" * 70)
    print("Generating Lane-Emden Data")
    print("=" * 70)
    print("\nPhysics Parameters:")
    print(f"  n (polytropic index): {args.n}")
    print(f"  ξ_min (min radius):   {args.xi_min}")
    print(f"  ξ_max (max radius):   {args.xi_max}")

    if args.n == 0:
        print("\nAnalytical solution:  θ(ξ) = 1 - ξ²/6")
    elif args.n == 1:
        print("\nAnalytical solution:  θ(ξ) = sin(ξ)/ξ")
    elif args.n == 5:
        print("\nAnalytical solution:  θ(ξ) ≈ (1 + ξ²/3)^(-1/2)")
    else:
        print("\nNo analytical solution (will use numerical integration)")

    print("\nExpected Coefficients:")
    print("  c1 (coeff of ξ⁻¹θ_ξ): 2.000000")
    print("  c2 (coeff of θⁿ):     1.000000")
    print(f"  n  (exponent):        {args.n:.6f}")

    print("\nData Settings:")
    print(f"  Grid points:          {args.n_points}")
    print(f"  Domain:               [{args.xi_min}, {args.xi_max}]")
    print(f"  Noise level:          {args.noise*100:.2f}%")
    print(f"  Random seed:          {args.seed}")
    print(f"  Integration:          {'Numerical' if args.numerical else 'Analytical (if available)'}")

    df, params = generate_lane_emden_data(
        n=args.n,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        n_points=args.n_points,
        use_analytical=not args.numerical,
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
        f.write("Lane-Emden Equation - Ground Truth\n")
        f.write("=" * 50 + "\n\n")
        f.write("Equation: (1/ξ²) d/dξ(ξ² dθ/dξ) + θⁿ = 0\n")
        f.write("Expanded: θ_ξξ + (2/ξ) θ_ξ + θⁿ = 0\n\n")
        f.write("Parameters:\n")
        f.write(f"  n  = {params['n']} (polytropic index)\n\n")
        f.write("Expected Discovery (implicit form):\n")
        f.write("  θ_ξξ + c1*(1/ξ)*θ_ξ + c2*θⁿ = 0\n\n")
        f.write("Expected Coefficients:\n")
        f.write(f"  c1 = {params['c1_true']:.6f}\n")
        f.write(f"  c2 = {params['c2_true']:.6f}\n")
        f.write(f"  n  = {params['n']:.6f}\n\n")
        f.write(f"Solution method: {params['method']}\n")

        if params['n'] == 0:
            f.write("\nAnalytical: θ(ξ) = 1 - ξ²/6\n")
        elif params['n'] == 1:
            f.write("\nAnalytical: θ(ξ) = sin(ξ)/ξ\n")
        elif params['n'] == 5:
            f.write("\nAnalytical: θ(ξ) ≈ (1 + ξ²/3)^(-1/2)\n")

    print(f"✓ Saved metadata to: {meta_path}")

    # Print statistics
    print("\nData Statistics:")
    print(f"  θ range:             [{df['y'].min():.4f}, {df['y'].max():.4f}]")
    print(f"  θ mean:              {df['y'].mean():.4f}")
    print(f"  θ std:               {df['y'].std():.4f}")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("\n1. Run ODE discovery WITHOUT templates (baseline):")
    print(f"   python ../../run_de.py --filepath {output_path} \\")
    print("       --order_candidates 2 --include_inv_xdu --no_const --no_x --no_xu")
    print("\n2. Run with PowerLaw template (heuristic init only):")
    print(f"   python ../../run_de.py --filepath {output_path} \\")
    print("       --order_candidates 2 --include_inv_xdu --no_const --no_x --no_xu \\")
    print("       --varpro --varpro_templates power")
    print("\n3. Run with PowerLaw template + LM optimization over ψ:")
    print(f"   python ../../run_de.py --filepath {output_path} \\")
    print("       --order_candidates 2 --include_inv_xdu --no_const --no_x --no_xu \\")
    print("       --varpro --varpro_templates power \\")
    print("       --template_lm --template_lm_epochs 200")
    print(f"\n   Expected: Discover n ≈ {args.n:.1f}, c1 ≈ 2.0, c2 ≈ 1.0")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
