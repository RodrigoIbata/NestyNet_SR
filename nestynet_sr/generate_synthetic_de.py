#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Generate synthetic DE data for Phase 2 testing.

Creates CSV files with solutions to known DEs for validating template discovery:
1. Power law: u_x = k*u^2 (logistic-like, p=2)
2. Exponential: u_x = k*exp(a*x) (exponential growth)
3. Mixed: u_x = k*x*u (linear growth with position dependence)
"""

import argparse

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def generate_power_law_de(k=1.0, p=2.0, x_range=(0.0, 2.0), n_points=2000, u0=0.5):
    """Generate data for u_x = k*u^p.

    For p=2: du/dx = k*u^2
    Solution: u(x) = u0 / (1 - k*u0*x) (for k*u0*x < 1)

    Parameters
    ----------
    k : float
        Coefficient
    p : float
        Power exponent
    x_range : tuple
        (x_min, x_max)
    n_points : int
        Number of data points
    u0 : float
        Initial condition at x=x_min

    Returns
    -------
    DataFrame with columns [x0, y0]
    """
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_points)

    if p == 2.0:
        # Analytical solution for p=2
        # Ensure we don't hit singularity
        x_sing = 1.0 / (k * u0)
        if x_max >= x_sing:
            print(f"Warning: x_max={x_max} >= singularity at {x_sing:.3f}, truncating")
            x_max = 0.9 * x_sing
            x = np.linspace(x_min, x_max, n_points)

        u = u0 / (1.0 - k * u0 * x)
    else:
        # Numerical integration for general p
        def ode_func(t, y):
            return [k * y[0] ** p]

        sol = solve_ivp(
            ode_func, [x_min, x_max], [u0], t_eval=x, method="RK45", rtol=1e-10, atol=1e-12
        )
        u = sol.y[0]

    df = pd.DataFrame({"x0": x, "y0": u})
    return df


def generate_exponential_de(k=0.5, a=1.0, x_range=(0.0, 2.0), n_points=2000, u0=1.0):
    """Generate data for u_x = k*exp(a*x).

    Solution: u(x) = u0 + (k/a) * (exp(a*x) - exp(a*x0))

    Parameters
    ----------
    k : float
        Coefficient
    a : float
        Exponential rate
    x_range : tuple
        (x_min, x_max)
    n_points : int
        Number of data points
    u0 : float
        Initial condition at x=x_min

    Returns
    -------
    DataFrame with columns [x0, y0]
    """
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_points)

    # Analytical solution
    u = u0 + (k / a) * (np.exp(a * x) - np.exp(a * x_min))

    df = pd.DataFrame({"x0": x, "y0": u})
    return df


def generate_mixed_de(k=1.0, x_range=(0.0, 2.0), n_points=2000, u0=1.0):
    """Generate data for u_x = k*x*u (product form).

    Solution: u(x) = u0 * exp(k * (x^2 - x0^2) / 2)

    Parameters
    ----------
    k : float
        Coefficient
    x_range : tuple
        (x_min, x_max)
    n_points : int
        Number of data points
    u0 : float
        Initial condition at x=x_min

    Returns
    -------
    DataFrame with columns [x0, y0]
    """
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_points)

    # Analytical solution
    u = u0 * np.exp(k * (x**2 - x_min**2) / 2)

    df = pd.DataFrame({"x0": x, "y0": u})
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DE data")
    parser.add_argument(
        "--de_type",
        type=str,
        required=True,
        choices=["power", "exp", "mixed"],
        help="Type of DE to generate",
    )
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--k", type=float, default=1.0, help="Coefficient k")
    parser.add_argument("--p", type=float, default=2.0, help="Power exponent (for power law)")
    parser.add_argument("--a", type=float, default=1.0, help="Exponential rate (for exponential)")
    parser.add_argument("--x_min", type=float, default=0.0, help="Minimum x value")
    parser.add_argument("--x_max", type=float, default=2.0, help="Maximum x value")
    parser.add_argument("--n_points", type=int, default=2000, help="Number of data points")
    parser.add_argument(
        "--u0", type=float, default=None, help="Initial condition (auto-set if not specified)"
    )

    args = parser.parse_args()

    # Set default u0 based on DE type
    if args.u0 is None:
        if args.de_type == "power":
            args.u0 = 0.5
        else:
            args.u0 = 1.0

    print(f"Generating {args.de_type} DE data:")
    print(f"  k={args.k}")

    if args.de_type == "power":
        print(f"  p={args.p}")
        print(f"  Equation: u_x = {args.k}*u^{args.p}")
        df = generate_power_law_de(
            k=args.k, p=args.p, x_range=(args.x_min, args.x_max), n_points=args.n_points, u0=args.u0
        )
    elif args.de_type == "exp":
        print(f"  a={args.a}")
        print(f"  Equation: u_x = {args.k}*exp({args.a}*x)")
        df = generate_exponential_de(
            k=args.k, a=args.a, x_range=(args.x_min, args.x_max), n_points=args.n_points, u0=args.u0
        )
    elif args.de_type == "mixed":
        print(f"  Equation: u_x = {args.k}*x*u")
        df = generate_mixed_de(
            k=args.k, x_range=(args.x_min, args.x_max), n_points=args.n_points, u0=args.u0
        )

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} points to {args.output}")
    print(f"  x range: [{df['x0'].min():.3f}, {df['x0'].max():.3f}]")
    print(f"  u range: [{df['y0'].min():.3f}, {df['y0'].max():.3f}]")


if __name__ == "__main__":
    main()
