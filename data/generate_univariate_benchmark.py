#!/usr/bin/env python
"""Generate all univariate benchmark datasets (u000-u024).

Each dataset has 4000 uniformly-sampled points in the specified x range.
Output: data/u000.csv, data/u001.csv, ..., data/u024.csv

Dimension basis: [L, M, T, I, Theta]  (only L, M, T used)
"""
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------
# Each entry: id, expression string, numpy lambda, x_range,
#             x_dim, y_dim, free_const_dims, description
#
# free_const_dims maps constant names to their [L,M,T,I,Theta] exponent
# vectors and ground-truth values.  The SR engine would receive only the
# dimension vectors; the values are for data generation and validation.
# ---------------------------------------------------------------------------

BENCHMARKS = [
    # ---- Tier 1: Monomials & Polynomials ----
    {
        "id": "u000",
        "expr": "2.5*x0",
        "fn": lambda x: 2.5 * x,
        "x_range": (0.1, 5.0),
        "x_dim": [0, 0, 1, 0, 0],       # T
        "y_dim": [1, 0, 0, 0, 0],       # L
        "consts": {"c0": {"dim": [1, 0, -1, 0, 0], "val": 2.5}},   # L/T
        "desc": "Linear: uniform motion d = v*t",
    },
    {
        "id": "u001",
        "expr": "1.5*x0**2",
        "fn": lambda x: 1.5 * x**2,
        "x_range": (0.1, 4.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {"c0": {"dim": [1, 0, -2, 0, 0], "val": 1.5}},   # L/T^2
        "desc": "Quadratic monomial: free fall s = (1/2)g*t^2",
    },
    {
        "id": "u002",
        "expr": "0.5*x0**2 + 3.0*x0",
        "fn": lambda x: 0.5 * x**2 + 3.0 * x,
        "x_range": (0.1, 5.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, -2, 0, 0], "val": 0.5},
            "c1": {"dim": [1, 0, -1, 0, 0], "val": 3.0},
        },
        "desc": "2-term polynomial: accelerated motion with initial velocity",
    },
    {
        "id": "u003",
        "expr": "0.8*x0**3",
        "fn": lambda x: 0.8 * x**3,
        "x_range": (0.2, 3.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {"c0": {"dim": [1, 0, -3, 0, 0], "val": 0.8}},
        "desc": "Cubic monomial",
    },
    {
        "id": "u004",
        "expr": "0.3*x0**3 + 2.0*x0",
        "fn": lambda x: 0.3 * x**3 + 2.0 * x,
        "x_range": (0.2, 3.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, -3, 0, 0], "val": 0.3},
            "c1": {"dim": [1, 0, -1, 0, 0], "val": 2.0},
        },
        "desc": "Odd polynomial: anharmonic restoring force",
    },
    {
        "id": "u005",
        "expr": "0.2*x0**4 - 1.5*x0**2 + 1.0",
        "fn": lambda x: 0.2 * x**4 - 1.5 * x**2 + 1.0,
        "x_range": (0.5, 3.0),
        "x_dim": [1, 0, 0, 0, 0],       # L
        "y_dim": [2, 1, -2, 0, 0],      # M*L^2/T^2 (energy)
        "consts": {
            "c0": {"dim": [-2, 1, -2, 0, 0], "val": 0.2},   # M/(L^2*T^2)
            "c1": {"dim": [0, 1, -2, 0, 0], "val": -1.5},   # M/T^2
            "c2": {"dim": [2, 1, -2, 0, 0], "val": 1.0},     # M*L^2/T^2
        },
        "desc": "Even polynomial (3 terms): double-well potential",
    },

    # ---- Tier 2: Power Laws & Rationals ----
    {
        "id": "u006",
        "expr": "5.0/x0",
        "fn": lambda x: 5.0 / x,
        "x_range": (0.5, 5.0),
        "x_dim": [1, 0, 0, 0, 0],       # L
        "y_dim": [1, 1, -2, 0, 0],      # M*L/T^2 (force)
        "consts": {"c0": {"dim": [2, 1, -2, 0, 0], "val": 5.0}},  # M*L^2/T^2
        "desc": "Inverse: gravitational/Coulomb force ~1/r",
    },
    {
        "id": "u007",
        "expr": "3.0/x0**2",
        "fn": lambda x: 3.0 / x**2,
        "x_range": (0.5, 5.0),
        "x_dim": [1, 0, 0, 0, 0],
        "y_dim": [1, 1, -2, 0, 0],
        "consts": {"c0": {"dim": [3, 1, -2, 0, 0], "val": 3.0}},  # M*L^3/T^2
        "desc": "Inverse square: Coulomb force ~1/r^2",
    },
    {
        "id": "u008",
        "expr": "2.0*sqrt(x0)",
        "fn": lambda x: 2.0 * np.sqrt(x),
        "x_range": (0.1, 10.0),
        "x_dim": [1, 0, 0, 0, 0],       # L
        "y_dim": [0, 0, 1, 0, 0],       # T
        "consts": {"c0": {"dim": [-0.5, 0, 1, 0, 0], "val": 2.0}},  # T/sqrt(L)
        "desc": "Square root: pendulum period T ~ sqrt(L)",
    },
    {
        "id": "u009",
        "expr": "1.2*x0**(3/2)",
        "fn": lambda x: 1.2 * x**1.5,
        "x_range": (0.1, 4.0),
        "x_dim": [0, 0, 1, 0, 0],       # T
        "y_dim": [1, 0, 0, 0, 0],       # L
        "consts": {"c0": {"dim": [1, 0, -1.5, 0, 0], "val": 1.2}},  # L/T^(3/2)
        "desc": "Power 3/2: Kepler's third law",
    },
    {
        "id": "u010",
        "expr": "1.5*x0 + 2.0/x0",
        "fn": lambda x: 1.5 * x + 2.0 / x,
        "x_range": (0.5, 5.0),
        "x_dim": [1, 0, 0, 0, 0],       # L
        "y_dim": [1, 0, 0, 0, 0],       # L
        "consts": {
            "c0": {"dim": [0, 0, 0, 0, 0], "val": 1.5},     # dimensionless
            "c1": {"dim": [2, 0, 0, 0, 0], "val": 2.0},     # L^2
        },
        "desc": "Polynomial + rational: effective potential analog",
    },

    # ---- Tier 3: Single Transcendental ----
    {
        "id": "u011",
        "expr": "3.0*sin(2.0*x0)",
        "fn": lambda x: 3.0 * np.sin(2.0 * x),
        "x_range": (0.1, 6.0),
        "x_dim": [0, 0, 1, 0, 0],       # T
        "y_dim": [1, 0, 0, 0, 0],       # L
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 3.0},     # L (amplitude)
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 2.0},    # 1/T (frequency)
        },
        "desc": "Sine: simple harmonic motion x = A*sin(omega*t)",
    },
    {
        "id": "u012",
        "expr": "2.5*cos(1.5*x0)",
        "fn": lambda x: 2.5 * np.cos(1.5 * x),
        "x_range": (0.1, 6.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 2.5},
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 1.5},
        },
        "desc": "Cosine: SHM (cosine phase)",
    },
    {
        "id": "u013",
        "expr": "5.0*exp(-0.5*x0)",
        "fn": lambda x: 5.0 * np.exp(-0.5 * x),
        "x_range": (0.1, 8.0),
        "x_dim": [0, 0, 1, 0, 0],       # T
        "y_dim": [0, 1, 0, 0, 0],       # M
        "consts": {
            "c0": {"dim": [0, 1, 0, 0, 0], "val": 5.0},     # M
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 0.5},    # 1/T
        },
        "desc": "Exponential decay: radioactive decay N = N0*exp(-lambda*t)",
    },
    {
        "id": "u014",
        "expr": "2.0*log(0.5*x0)",
        "fn": lambda x: 2.0 * np.log(0.5 * x),
        "x_range": (0.1, 10.0),
        "x_dim": [0, 0, 1, 0, 0],       # T
        "y_dim": [2, 1, -2, 0, 0],      # M*L^2/T^2 (energy)
        "consts": {
            "c0": {"dim": [2, 1, -2, 0, 0], "val": 2.0},    # M*L^2/T^2
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 0.5},    # 1/T
        },
        "desc": "Logarithm: entropy-like measure (argument must be dimensionless)",
    },
    {
        "id": "u015",
        "expr": "2.0*exp(-0.5*x0**2)",
        "fn": lambda x: 2.0 * np.exp(-0.5 * x**2),
        "x_range": (0.1, 4.0),
        "x_dim": [0, 0, 1, 0, 0],       # T
        "y_dim": [1, 0, 0, 0, 0],       # L
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 2.0},     # L
            "c1": {"dim": [0, 0, -2, 0, 0], "val": 0.5},    # 1/T^2
        },
        "desc": "Gaussian: unnormalized probability density",
    },

    # ---- Tier 4: Products & Sums ----
    {
        "id": "u016",
        "expr": "3.0*x0*exp(-0.8*x0)",
        "fn": lambda x: 3.0 * x * np.exp(-0.8 * x),
        "x_range": (0.1, 6.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, -1, 0, 0], "val": 3.0},    # L/T
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 0.8},    # 1/T
        },
        "desc": "Polynomial * exponential: impulse response t*exp(-t/tau)",
    },
    {
        "id": "u017",
        "expr": "2.0*sin(3.0*x0) + 0.5*x0",
        "fn": lambda x: 2.0 * np.sin(3.0 * x) + 0.5 * x,
        "x_range": (0.1, 5.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 2.0},     # L
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 3.0},    # 1/T
            "c2": {"dim": [1, 0, -1, 0, 0], "val": 0.5},    # L/T
        },
        "desc": "Additive trig + polynomial: oscillation around linear drift",
    },
    {
        "id": "u018",
        "expr": "1.5*x0*sin(2.0*x0)",
        "fn": lambda x: 1.5 * x * np.sin(2.0 * x),
        "x_range": (0.1, 5.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, -1, 0, 0], "val": 1.5},    # L/T
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 2.0},    # 1/T
        },
        "desc": "Polynomial * trig: resonance buildup",
    },
    {
        "id": "u019",
        "expr": "2.0*x0**2*exp(-0.5*x0)",
        "fn": lambda x: 2.0 * x**2 * np.exp(-0.5 * x),
        "x_range": (0.1, 8.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, -2, 0, 0], "val": 2.0},    # L/T^2
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 0.5},    # 1/T
        },
        "desc": "Quadratic * exponential: gamma-like shape",
    },
    {
        "id": "u020",
        "expr": "3.0*exp(-0.3*x0)*sin(2.0*x0)",
        "fn": lambda x: 3.0 * np.exp(-0.3 * x) * np.sin(2.0 * x),
        "x_range": (0.1, 10.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 3.0},     # L
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 0.3},    # 1/T
            "c2": {"dim": [0, 0, -1, 0, 0], "val": 2.0},    # 1/T
        },
        "desc": "Damped oscillation: exp(-gamma*t)*sin(omega*t)",
    },

    # ---- Tier 5: Nested & Composite (Hardest) ----
    {
        "id": "u021",
        "expr": "2.0*sin(x0) + sin(3.0*x0)",
        "fn": lambda x: 2.0 * np.sin(x) + np.sin(3.0 * x),
        "x_range": (0.1, 8.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 2.0},     # L
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 1.0},    # 1/T
            "c2": {"dim": [1, 0, 0, 0, 0], "val": 1.0},     # L
            "c3": {"dim": [0, 0, -1, 0, 0], "val": 3.0},    # 1/T
        },
        "desc": "Double harmonic: superposition of two modes",
    },
    {
        "id": "u022",
        "expr": "10.0/(1 + exp(-1.5*x0))",
        "fn": lambda x: 10.0 / (1.0 + np.exp(-1.5 * x)),
        "x_range": (-3.0, 3.0),
        "x_dim": [0, 0, 1, 0, 0],       # T
        "y_dim": [0, 1, 0, 0, 0],       # M
        "consts": {
            "c0": {"dim": [0, 1, 0, 0, 0], "val": 10.0},    # M
            "c1": {"dim": [0, 0, -1, 0, 0], "val": 1.5},    # 1/T
        },
        "desc": "Sigmoid/logistic: population saturation",
    },
    {
        "id": "u023",
        "expr": "2.0*sin(0.8*x0**2)",
        "fn": lambda x: 2.0 * np.sin(0.8 * x**2),
        "x_range": (0.1, 4.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 2.0},     # L
            "c1": {"dim": [0, 0, -2, 0, 0], "val": 0.8},    # 1/T^2
        },
        "desc": "Nested trig: sin(k*x^2), chirp-like signal",
    },
    {
        "id": "u024",
        "expr": "2.0*sin(0.8*x0**2)*cos(1.5*x0)",
        "fn": lambda x: 2.0 * np.sin(0.8 * x**2) * np.cos(1.5 * x),
        "x_range": (0.1, 4.0),
        "x_dim": [0, 0, 1, 0, 0],
        "y_dim": [1, 0, 0, 0, 0],
        "consts": {
            "c0": {"dim": [1, 0, 0, 0, 0], "val": 2.0},     # L
            "c1": {"dim": [0, 0, -2, 0, 0], "val": 0.8},    # 1/T^2
            "c2": {"dim": [0, 0, -1, 0, 0], "val": 1.5},    # 1/T
        },
        "desc": "Nguyen-5 analog: sin(k*x^2)*cos(omega*x) with dimensions",
    },
]


def generate_all(n_samples: int = 4000, seed: int = 42, output_dir: str = "data"):
    """Generate CSV datasets for all 25 univariate benchmark equations."""
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    for b in BENCHMARKS:
        x = np.random.uniform(b["x_range"][0], b["x_range"][1], n_samples)
        y = b["fn"](x)

        df = pd.DataFrame({"y": y, "x0": x})
        path = os.path.join(output_dir, f"{b['id']}.csv")
        df.to_csv(path, index=False)

        print(f"  {b['id']}  {b['expr']:45s}  "
              f"x=[{b['x_range'][0]:5.1f}, {b['x_range'][1]:5.1f}]  "
              f"y=[{y.min():.3g}, {y.max():.3g}]  -> {path}")

    print(f"\nGenerated {len(BENCHMARKS)} datasets ({n_samples} samples each) in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate univariate benchmark datasets (u000-u024)."
    )
    parser.add_argument("--samples", type=int, default=4000,
                        help="Number of samples per dataset (default: 4000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same directory as this script)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    generate_all(n_samples=args.samples, seed=args.seed, output_dir=args.output_dir)
