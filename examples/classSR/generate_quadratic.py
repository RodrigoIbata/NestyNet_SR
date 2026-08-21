#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate CSV data for 3 quadratic datasets: y = c0 + c1*x0 + g0*x0².

Shared class constant:  g0 = 0.5  (quadratic coefficient)
Per-dataset constants:
  Dataset 1: c0 = 2.0, c1 = 1.5
  Dataset 2: c0 = 3.0, c1 = 0.8
  Dataset 3: c0 = 1.0, c1 = 2.5

Units:  y [L] = [1,0],  x0 [T] = [0,1]
  => c0 [L] = [1,0],  c1 [L/T] = [1,-1],  g0 [L/T²] = [1,-2]

Each CSV has columns: y,x0  with 5000 rows, shuffled.
"""

import pathlib

import numpy as np

# --- Ground-truth parameters ---
G0 = 0.5
DATASETS = [
    {"c0": 2.0, "c1": 1.5},
    {"c0": 3.0, "c1": 0.8},
    {"c0": 1.0, "c1": 2.5},
]
N_POINTS = 5000
X_MIN, X_MAX = 0.0, 5.0

DATA_DIR = pathlib.Path(__file__).parent / "data"


def generate_quadratic(c0: float, c1: float, g0: float,
                       n: int = N_POINTS, seed: int = 42) -> tuple:
    """Return (x0, y) arrays for a single quadratic, randomly shuffled."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(X_MIN, X_MAX, size=n)
    y = c0 + c1 * x0 + g0 * x0**2
    idx = rng.permutation(n)
    return x0[idx], y[idx]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(DATASETS, start=1):
        x0, y = generate_quadratic(params["c0"], params["c1"], G0, seed=42 + i)
        path = DATA_DIR / f"quad_{i}.csv"
        np.savetxt(path, np.column_stack([y, x0]),
                   delimiter=",", header="y,x0", comments="")
        print(f"Wrote {path}  (c0={params['c0']}, c1={params['c1']}, g0={G0}, N={len(x0)})")

    # Quick plot (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        x_plot = np.linspace(X_MIN, X_MAX, 200)
        for i, params in enumerate(DATASETS, start=1):
            y_plot = params["c0"] + params["c1"] * x_plot + G0 * x_plot**2
            ax.plot(x_plot, y_plot,
                    label=f"quad {i} (c0={params['c0']}, c1={params['c1']})")
        ax.set_xlabel("x0 [T]")
        ax.set_ylabel("y [L]")
        ax.set_title(f"Quadratics: y = c0 + c1*x0 + {G0}*x0²")
        ax.legend()
        fig.tight_layout()
        fig.savefig(DATA_DIR / "quadratics_overview.png", dpi=120)
        plt.close(fig)
        print(f"Wrote {DATA_DIR / 'quadratics_overview.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
