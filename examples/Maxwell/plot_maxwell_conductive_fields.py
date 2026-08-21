#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Visualize conductive-medium Maxwell toy fields."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def _sym_vmax(a: np.ndarray, floor: float = 1e-12) -> float:
    return float(max(np.max(np.abs(a)), floor))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_data = repo_root / "data" / "maxwell" / "fake_maxwell_conductive.npz"
    default_output = script_dir / "maxwell_conductive_fields.png"

    parser = argparse.ArgumentParser(
        description="Plot conductive-medium Maxwell toy fields",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=default_data, help="Input .npz data file")
    parser.add_argument("--output", type=Path, default=default_output, help="Output figure path")
    parser.add_argument("--show", action="store_true", help="Show figure window")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            f"Generate it first with: python {script_dir / 'generate_fake_maxwell_conductive_data.py'}"
        )

    blob = np.load(args.data)
    Y = np.asarray(blob["Y"], dtype=np.float64)
    t_vals = np.asarray(blob["t_vals"], dtype=np.float64)
    x_vals = np.asarray(blob["x_vals"], dtype=np.float64)
    y_vals = np.asarray(blob["y_vals"], dtype=np.float64)
    z_vals = np.asarray(blob["z_vals"], dtype=np.float64)
    sigma = float(blob["sigma"][0]) if "sigma" in blob else float("nan")

    nt = int(len(t_vals))
    nx = int(len(x_vals))
    ny = int(len(y_vals))
    nz = int(len(z_vals))
    Yg = Y.reshape(nt, nx, ny, nz, 6)

    ix = nx // 2
    iy = ny // 2
    iz = nz // 2

    ex_tz = Yg[:, ix, iy, :, 0]
    by_tz = Yg[:, ix, iy, :, 4]

    # RMS envelope over space/components.
    e_rms_t = np.sqrt(np.mean(Yg[..., 0:3] ** 2, axis=(1, 2, 3, 4)))
    b_rms_t = np.sqrt(np.mean(Yg[..., 3:6] ** 2, axis=(1, 2, 3, 4)))
    gamma = 0.5 * sigma
    env = e_rms_t[0] * np.exp(-gamma * t_vals)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    extent_tz = [float(z_vals[0]), float(z_vals[-1]), float(t_vals[0]), float(t_vals[-1])]

    vmax = _sym_vmax(ex_tz)
    im = axes[0, 0].imshow(
        ex_tz, origin="lower", aspect="auto", extent=extent_tz, cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    axes[0, 0].set_title("Ex(t,z) at x_mid,y_mid")
    axes[0, 0].set_xlabel("z")
    axes[0, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.85)

    vmax = _sym_vmax(by_tz)
    im = axes[0, 1].imshow(
        by_tz, origin="lower", aspect="auto", extent=extent_tz, cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    axes[0, 1].set_title("By(t,z) at x_mid,y_mid")
    axes[0, 1].set_xlabel("z")
    axes[0, 1].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.85)

    axes[1, 0].plot(t_vals, e_rms_t, lw=2, label="RMS(|E|)")
    axes[1, 0].plot(t_vals, b_rms_t, lw=2, ls="--", label="RMS(|B|)")
    axes[1, 0].plot(t_vals, env, lw=2, ls=":", label=r"$E_0 e^{-\sigma t / 2}$")
    axes[1, 0].set_title("Global RMS Decay")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("RMS amplitude")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(loc="best", fontsize=9)

    tidx = nt // 3
    axes[1, 1].plot(z_vals, ex_tz[tidx], lw=2, label=f"Ex(t={t_vals[tidx]:.2f}, z)")
    axes[1, 1].plot(z_vals, by_tz[tidx], lw=2, ls="--", label=f"By(t={t_vals[tidx]:.2f}, z)")
    axes[1, 1].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[1, 1].set_title("Damped Wave Slice")
    axes[1, 1].set_xlabel("z")
    axes[1, 1].set_ylabel("amplitude")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(loc="best", fontsize=9)

    fig.suptitle(
        "Conductive-Medium Maxwell Toy\n"
        f"sigma={sigma:.3f}, expected damping rate gamma=sigma/2={gamma:.3f}, "
        f"point=(x[{ix}], y[{iy}], z[{iz}])",
        fontsize=12,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"Saved figure: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
