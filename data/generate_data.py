# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import argparse
import numpy as np
import pandas as pd
import sympy
import sys
import os

def generate_dataset(expression: str, output_path: str, n_samples: int = 10000,
                    x_min: float = 1.0, x_max: float = 5.0,
                    xmin_per_var=None, xmax_per_var=None, seed: int = 42):
    """
    Generates a dataset based on a mathematical expression.

    Example:
    python data/generate_data.py --expr "sin(log(x0*x1))" --output "data/my_dataset.csv"
    """
    np.random.seed(seed)

    # 1. Parse Expression using SymPy
    try:
        expr = sympy.sympify(expression)
    except sympy.SympifyError as e:
        print(f"Error parsing expression: {e}")
        sys.exit(1)

    # 2. Identify variables (x0, x1, ...)
    # detailed extraction of symbols
    symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
    var_names = [s.name for s in symbols]

    print(f"Detected variables: {var_names}")

    if not var_names:
        print("Error: No variables found in expression. Use x0, x1, etc.")
        sys.exit(1)

    # 3. Generate Random Data
    n_vars = len(var_names)
    if xmin_per_var is not None:
        if len(xmin_per_var) != n_vars:
            print(f"Error: --xmin has {len(xmin_per_var)} values but expression has {n_vars} variables")
            sys.exit(1)
    if xmax_per_var is not None:
        if len(xmax_per_var) != n_vars:
            print(f"Error: --xmax has {len(xmax_per_var)} values but expression has {n_vars} variables")
            sys.exit(1)

    data = {}
    for i, var in enumerate(var_names):
        lo = xmin_per_var[i] if xmin_per_var is not None else x_min
        hi = xmax_per_var[i] if xmax_per_var is not None else x_max
        data[var] = np.random.uniform(lo, hi, n_samples)

    df = pd.DataFrame(data)

    # 4. Evaluate Expression
    # lambdify for fast numpy evaluation
    f = sympy.lambdify(symbols, expr, "numpy")

    try:
        # Pass columns as arguments
        args = [df[v].values for v in var_names]
        y = f(*args)

        # specific check for single value result (if expression is constant-like despite symbols, shouldn't happen with free_symbols check but safety first)
        if np.isscalar(y):
            y = np.full(n_samples, y)

    except Exception as e:
        print(f"Error evaluating expression: {e}")
        sys.exit(1)

    # 5. Format Output
    # The user example was: y,x0,x1,x2
    # So we want 'y' first, then sorted variables.
    df['y'] = y

    # Reorder columns: y, then variables sorted
    cols = ['y'] + var_names
    df = df[cols]

    # 6. Save
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print("Head:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from expression.")
    parser.add_argument("--expr", type=str, required=True, help="Mathematical expression, e.g., 'sin(log(x0*x1))'")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--min", type=float, default=1.0, help="Min feature value (to avoid log(0))")
    parser.add_argument("--max", type=float, default=5.0, help="Max feature value")
    parser.add_argument("--xmin", type=str, default=None, help='Per-variable min, e.g. "[1. 4.]"')
    parser.add_argument("--xmax", type=str, default=None, help='Per-variable max, e.g. "[3. 6.]"')

    args = parser.parse_args()

    def parse_bracket_list(s):
        """Parse '[1. 4.]' or '[1.0, 4.0]' into a list of floats."""
        s = s.strip().strip("[]")
        return [float(v) for v in s.replace(",", " ").split()]

    xmin_pv = parse_bracket_list(args.xmin) if args.xmin is not None else None
    xmax_pv = parse_bracket_list(args.xmax) if args.xmax is not None else None

    generate_dataset(args.expr, args.output, args.samples, args.min, args.max,
                     xmin_per_var=xmin_pv, xmax_per_var=xmax_pv)
