#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate CSV data for 5 damped springs: y = cos(omega_i * t) * exp(-k * t).

Shared class constant:  k = 0.3  (damping)
Per-spring constants:   omega = [2.0, 3.0, 5.0, 7.0, 4.5]

Each CSV has columns: y,x0  with 5000 rows (enough for NestyNet train+val splits).
"""

import pathlib

import numpy as np

# --- Ground-truth parameters ---
OMEGAS = [2.0, 3.0, 5.0, 7.0, 4.5]
K = 0.3
N_POINTS = 5000
T_MIN, T_MAX = 0.0, 10.0

DATA_DIR = pathlib.Path(__file__).parent / "data"


def generate_spring(omega: float, k: float, n: int = N_POINTS, seed: int = 42) -> tuple:
    """Return (t, y) arrays for a single damped spring, randomly shuffled.

    PhysDataset takes the first N rows for training and the next N for
    validation, so we shuffle to ensure both splits cover the full t range.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(T_MIN, T_MAX, size=n)
    t.sort()  # sort for determinism, then shuffle
    y = np.cos(omega * t) * np.exp(-k * t)
    # Shuffle rows so train/val splits cover the whole t range
    idx = rng.permutation(n)
    return t[idx], y[idx]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i, omega in enumerate(OMEGAS, start=1):
        t, y = generate_spring(omega, K, seed=42 + i)
        path = DATA_DIR / f"spring_{i}.csv"
        np.savetxt(path, np.column_stack([y, t]), delimiter=",", header="y,x0", comments="")
        print(f"Wrote {path}  (omega={omega}, k={K}, N={len(t)})")

    # Quick plot (optional; saves to data dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        t_plot = np.linspace(T_MIN, T_MAX, 1000)
        for i, omega in enumerate(OMEGAS, start=1):
            y_plot = np.cos(omega * t_plot) * np.exp(-K * t_plot)
            ax.plot(t_plot, y_plot, label=f"spring {i} (ω={omega})")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title(f"Damped springs: y = cos(ω·t)·exp(−{K}·t)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(DATA_DIR / "springs_overview.png", dpi=120)
        plt.close(fig)
        print(f"Wrote {DATA_DIR / 'springs_overview.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
