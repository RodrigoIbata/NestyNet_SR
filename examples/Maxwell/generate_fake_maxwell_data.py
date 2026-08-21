#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate synthetic Maxwell plane-wave field data.

The generated fields satisfy vacuum Maxwell dynamics (c = 1):

    dE/dt = +curl(B)
    dB/dt = -curl(E)

using the analytic plane wave:

    E = (sin(k*z - w*t), 0, 0)
    B = (0, sin(k*z - w*t), 0),   w = c*k

Arrays saved in the output .npz:
    X : (N, 4)    coordinates [t, x, y, z]
    Y : (N, 6)    field values [Ex, Ey, Ez, Bx, By, Bz]
    G : (N, 6, 4) first derivatives dY_i / dX_j
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def build_maxwell_plane_wave_dataset(
    *,
    c: float,
    k: float,
    nt: int,
    nx: int = 1,
    ny: int = 1,
    nz: int,
    t_max: float,
    x_max: float = 0.0,
    y_max: float = 0.0,
    z_max: float,
    x0: float,
    y0: float,
    noise_std: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Build coordinates, field values, and gradients for the plane wave."""
    omega = c * k
    t_vals = np.linspace(0.0, t_max, nt, dtype=np.float64)
    if int(nx) <= 1:
        x_vals = np.asarray([float(x0)], dtype=np.float64)
    else:
        x_vals = np.linspace(float(x0) - float(x_max), float(x0) + float(x_max), int(nx), dtype=np.float64)
    if int(ny) <= 1:
        y_vals = np.asarray([float(y0)], dtype=np.float64)
    else:
        y_vals = np.linspace(float(y0) - float(y_max), float(y0) + float(y_max), int(ny), dtype=np.float64)
    # endpoint=False so [0, z_max) is exactly one spatial period (k integer):
    # this makes the FFT spectral Hessian machine-precise for the broadened
    # library's ∇² decoys.
    z_vals = np.linspace(0.0, z_max, nz, endpoint=False, dtype=np.float64)
    tt, xx, yy, zz = np.meshgrid(t_vals, x_vals, y_vals, z_vals, indexing="ij")

    phase = k * zz - omega * tt
    sin_ph = np.sin(phase)
    cos_ph = np.cos(phase)

    n = int(tt.size)
    X = np.zeros((n, 4), dtype=np.float64)
    X[:, 0] = tt.reshape(-1)  # t
    X[:, 1] = xx.reshape(-1)  # x
    X[:, 2] = yy.reshape(-1)  # y
    X[:, 3] = zz.reshape(-1)  # z

    Y = np.zeros((n, 6), dtype=np.float64)
    Y[:, 0] = sin_ph.reshape(-1)  # Ex
    Y[:, 4] = sin_ph.reshape(-1)  # By

    # G[row, output_component, coordinate_axis]
    G = np.zeros((n, 6, 4), dtype=np.float64)
    d_sin_dt = (-omega * cos_ph).reshape(-1)
    d_sin_dz = (k * cos_ph).reshape(-1)
    G[:, 0, 0] = d_sin_dt
    G[:, 0, 3] = d_sin_dz
    G[:, 4, 0] = d_sin_dt
    G[:, 4, 3] = d_sin_dz

    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        Y += rng.normal(0.0, noise_std, size=Y.shape)

    arrays = {
        "X": X,
        "Y": Y,
        "G": G,
        "t_vals": t_vals,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_vals": z_vals,
        "c": np.asarray([c], dtype=np.float64),
        "k": np.asarray([k], dtype=np.float64),
        "omega": np.asarray([omega], dtype=np.float64),
        "x0": np.asarray([x0], dtype=np.float64),
        "y0": np.asarray([y0], dtype=np.float64),
        "x_max": np.asarray([x_max], dtype=np.float64),
        "y_max": np.asarray([y_max], dtype=np.float64),
        "noise_std": np.asarray([noise_std], dtype=np.float64),
    }

    meta = {
        "c": float(c),
        "k": float(k),
        "omega": float(omega),
        "nt": int(nt),
        "nx": int(x_vals.size),
        "ny": int(y_vals.size),
        "nz": int(nz),
        "n_points": int(n),
        "x0": float(x0),
        "y0": float(y0),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "t_max": float(t_max),
        "z_max": float(z_max),
        "noise_std": float(noise_std),
    }
    return arrays, meta


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_output = repo_root / "data" / "maxwell" / "fake_maxwell_plane_wave.npz"

    parser = argparse.ArgumentParser(
        description="Generate synthetic Maxwell plane-wave data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output .npz path")
    parser.add_argument("--c", type=float, default=1.0, help="Wave speed")
    parser.add_argument("--k", type=float, default=1.0, help="Wavenumber")
    parser.add_argument("--nt", type=int, default=64, help="Number of time samples")
    parser.add_argument("--nx", type=int, default=1, help="Number of x samples")
    parser.add_argument("--ny", type=int, default=1, help="Number of y samples")
    parser.add_argument("--nz", type=int, default=64, help="Number of z samples")
    parser.add_argument("--t_max", type=float, default=2.0 * math.pi, help="Time domain max")
    parser.add_argument("--x_max", type=float, default=0.0, help="Half-width around x0")
    parser.add_argument("--y_max", type=float, default=0.0, help="Half-width around y0")
    parser.add_argument("--z_max", type=float, default=2.0 * math.pi, help="Z domain max")
    parser.add_argument("--x0", type=float, default=0.0, help="Fixed x coordinate")
    parser.add_argument("--y0", type=float, default=0.0, help="Fixed y coordinate")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Additive noise std for Y")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise")
    args = parser.parse_args()

    arrays, meta = build_maxwell_plane_wave_dataset(
        c=args.c,
        k=args.k,
        nt=args.nt,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        t_max=args.t_max,
        x_max=args.x_max,
        y_max=args.y_max,
        z_max=args.z_max,
        x0=args.x0,
        y0=args.y0,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)

    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("=" * 72)
    print("Generated synthetic Maxwell data")
    print("=" * 72)
    print(f"Output:      {args.output}")
    print(f"Metadata:    {meta_path}")
    print(f"Points:      {meta['n_points']} ({meta['nt']} x {meta['nx']} x {meta['ny']} x {meta['nz']})")
    print(f"Parameters:  c={meta['c']}, k={meta['k']}, omega={meta['omega']}")
    print(f"Noise std:   {meta['noise_std']}")
    print("Stored arrays: X (coords), Y (fields), G (gradients)")
    print("=" * 72)


if __name__ == "__main__":
    main()
