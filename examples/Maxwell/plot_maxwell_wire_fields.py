#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Visualize wire-source Maxwell toy fields from generate_fake_maxwell_wire_data.py."""

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
    default_data = repo_root / "data" / "maxwell" / "fake_maxwell_wire_source.npz"
    default_output = script_dir / "maxwell_wire_fields.png"

    parser = argparse.ArgumentParser(
        description="Plot wire-source Maxwell toy fields",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=default_data, help="Input .npz data file")
    parser.add_argument("--output", type=Path, default=default_output, help="Output figure path")
    parser.add_argument("--time_index", type=int, default=-1, help="Time slice index (-1 means nt//4)")
    parser.add_argument("--show", action="store_true", help="Show figure window")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            f"Generate it first with: python {script_dir / 'generate_fake_maxwell_wire_data.py'}"
        )

    blob = np.load(args.data)
    X = np.asarray(blob["X"], dtype=np.float64)
    Y = np.asarray(blob["Y"], dtype=np.float64)
    t_vals = np.asarray(blob["t_vals"], dtype=np.float64)
    x_vals = np.asarray(blob["x_vals"], dtype=np.float64)
    y_vals = np.asarray(blob["y_vals"], dtype=np.float64)
    z_vals = np.asarray(blob["z_vals"], dtype=np.float64)

    nt = int(len(t_vals))
    nx = int(len(x_vals))
    ny = int(len(y_vals))
    nz = int(len(z_vals))
    if nz != 1:
        raise ValueError("This plotter currently expects nz=1 in the wire toy dataset.")
    if X.shape[0] != nt * nx * ny * nz:
        raise ValueError("Grid shape mismatch in dataset.")

    Yg = Y.reshape(nt, nx, ny, nz, 9)

    tidx = nt // 4 if int(args.time_index) < 0 else int(args.time_index)
    tidx = max(0, min(nt - 1, tidx))
    zidx = 0

    ez = Yg[tidx, :, :, zidx, 2].T
    bx = Yg[tidx, :, :, zidx, 3].T
    by = Yg[tidx, :, :, zidx, 4].T
    jz = Yg[tidx, :, :, zidx, 8].T

    xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")
    bmag = np.sqrt(bx * bx + by * by)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent = [float(x_vals[0]), float(x_vals[-1]), float(y_vals[0]), float(y_vals[-1])]

    vmax = _sym_vmax(jz)
    im = axes[0, 0].imshow(
        jz, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    axes[0, 0].set_title("Source Current Density Jz(x,y)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.85)

    stride = max(1, nx // 18)
    axes[0, 1].quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        bx[::stride, ::stride],
        by[::stride, ::stride],
        bmag[::stride, ::stride],
        cmap="viridis",
        pivot="mid",
        scale=None,
    )
    axes[0, 1].set_title("Magnetic Field Vectors (Bx, By)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal")

    vmax = _sym_vmax(ez)
    im = axes[1, 0].imshow(
        ez, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    axes[1, 0].set_title("Electric Field Ez(x,y)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.85)

    ic = ny // 2
    axes[1, 1].plot(x_vals, jz[ic, :], label="Jz(y=0, x)", lw=2)
    axes[1, 1].plot(x_vals, ez[ic, :], label="Ez(y=0, x)", lw=2, ls="--")
    axes[1, 1].plot(x_vals, bmag[ic, :], label="|B|(y=0, x)", lw=2, ls=":")
    axes[1, 1].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[1, 1].set_title("Centerline Profiles")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("amplitude")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(loc="best", fontsize=9)

    t = float(t_vals[tidx])
    a0 = float(blob["a0"][0]) if "a0" in blob else float("nan")
    r0 = float(blob["r0"][0]) if "r0" in blob else float("nan")
    omega = float(blob["omega"][0]) if "omega" in blob else float("nan")
    phase = float(blob["phase"][0]) if "phase" in blob else float("nan")
    fig.suptitle(
        "Wire-Source Maxwell Toy Fields\n"
        f"t={t:.3f}, a0={a0:.3f}, r0={r0:.3f}, omega={omega:.3f}, phase={phase:.3f}",
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
