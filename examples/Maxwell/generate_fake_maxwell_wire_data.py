#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate a wire-inspired, source-driven Maxwell toy dataset.

This is a physically motivated toy for an AC current-carrying wire core:
we define a smooth z-directed vector potential

    A = (0, 0, Az),  Az(x,y,t) = A0 * exp(-(x^2+y^2)/r0^2) * sin(omega*t + phase)

and derive fields/source exactly from it:

    E = -dA/dt
    B = curl(A)
    J = curl(B) - dE/dt

so Maxwell is exactly satisfied:

    dE/dt - curl(B) + J = 0   (Ampere with source)
    dB/dt + curl(E) = 0       (Faraday)
    div(E) = div(B) = 0

Saved arrays:
    X : (N, 4)    [t, x, y, z]
    Y : (N, 9)    [Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz]
    G : (N, 9, 4) dY_i/dX_j
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def build_wire_dataset(
    *,
    a0: float,
    r0: float,
    omega: float,
    phase: float,
    nt: int,
    nx: int,
    ny: int,
    nz: int,
    t_max: float,
    x_max: float,
    y_max: float,
    z_max: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Build coordinates, fields, gradients, and diagnostics."""
    t_vals = np.linspace(0.0, t_max, nt, dtype=np.float64)
    x_vals = np.linspace(-x_max, x_max, nx, dtype=np.float64)
    y_vals = np.linspace(-y_max, y_max, ny, dtype=np.float64)
    if nz <= 1:
        z_vals = np.asarray([0.0], dtype=np.float64)
    else:
        z_vals = np.linspace(-z_max, z_max, nz, dtype=np.float64)

    tt, xx, yy, zz = np.meshgrid(t_vals, x_vals, y_vals, z_vals, indexing="ij")
    n = int(tt.size)

    inv_r2 = 1.0 / (r0 * r0)
    r2 = xx * xx + yy * yy
    psi = np.exp(-r2 * inv_r2)
    s = np.sin(omega * tt + phase)
    c = np.cos(omega * tt + phase)

    psi_x = -2.0 * xx * inv_r2 * psi
    psi_y = -2.0 * yy * inv_r2 * psi
    psi_xx = (-2.0 * inv_r2 + 4.0 * xx * xx * inv_r2 * inv_r2) * psi
    psi_yy = (-2.0 * inv_r2 + 4.0 * yy * yy * inv_r2 * inv_r2) * psi
    psi_xy = (4.0 * xx * yy * inv_r2 * inv_r2) * psi
    lap_psi = psi_xx + psi_yy

    # Fields.
    ex = np.zeros_like(psi)
    ey = np.zeros_like(psi)
    ez = -a0 * omega * psi * c

    bx = a0 * psi_y * s
    by = -a0 * psi_x * s
    bz = np.zeros_like(psi)

    jx = np.zeros_like(psi)
    jy = np.zeros_like(psi)
    jz = -a0 * (lap_psi + omega * omega * psi) * s

    X = np.stack([tt.reshape(-1), xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1).astype(np.float64)
    Y = np.stack(
        [
            ex.reshape(-1),
            ey.reshape(-1),
            ez.reshape(-1),
            bx.reshape(-1),
            by.reshape(-1),
            bz.reshape(-1),
            jx.reshape(-1),
            jy.reshape(-1),
            jz.reshape(-1),
        ],
        axis=1,
    ).astype(np.float64)

    G = np.zeros((n, 9, 4), dtype=np.float64)

    # Ez = -A0*omega*psi*cos(...)
    G[:, 2, 0] = (a0 * omega * omega * psi * s).reshape(-1)      # dEz/dt
    G[:, 2, 1] = (-a0 * omega * psi_x * c).reshape(-1)            # dEz/dx
    G[:, 2, 2] = (-a0 * omega * psi_y * c).reshape(-1)            # dEz/dy
    # dEz/dz = 0

    # Bx = A0*psi_y*sin(...)
    G[:, 3, 0] = (a0 * omega * psi_y * c).reshape(-1)             # dBx/dt
    G[:, 3, 1] = (a0 * psi_xy * s).reshape(-1)                    # dBx/dx
    G[:, 3, 2] = (a0 * psi_yy * s).reshape(-1)                    # dBx/dy

    # By = -A0*psi_x*sin(...)
    G[:, 4, 0] = (-a0 * omega * psi_x * c).reshape(-1)            # dBy/dt
    G[:, 4, 1] = (-a0 * psi_xx * s).reshape(-1)                   # dBy/dx
    G[:, 4, 2] = (-a0 * psi_xy * s).reshape(-1)                   # dBy/dy

    # Gauss diagnostics.
    div_e = G[:, 0, 1] + G[:, 1, 2] + G[:, 2, 3]
    div_b = G[:, 3, 1] + G[:, 4, 2] + G[:, 5, 3]
    div_e_rms = float(np.sqrt(np.mean(div_e * div_e)))
    div_b_rms = float(np.sqrt(np.mean(div_b * div_b)))

    # PDE diagnostics.
    dE_dt = G[:, 0:3, 0]
    dB_dt = G[:, 3:6, 0]
    J = Y[:, 6:9]

    curl_b = np.zeros((n, 3), dtype=np.float64)
    curl_b[:, 0] = G[:, 5, 2] - G[:, 4, 3]
    curl_b[:, 1] = G[:, 3, 3] - G[:, 5, 1]
    curl_b[:, 2] = G[:, 4, 1] - G[:, 3, 2]

    curl_e = np.zeros((n, 3), dtype=np.float64)
    curl_e[:, 0] = G[:, 2, 2] - G[:, 1, 3]
    curl_e[:, 1] = G[:, 0, 3] - G[:, 2, 1]
    curl_e[:, 2] = G[:, 1, 1] - G[:, 0, 2]

    amp_res = dE_dt - curl_b + J
    far_res = dB_dt + curl_e
    amp_rms = np.sqrt(np.mean(amp_res * amp_res, axis=0))
    far_rms = np.sqrt(np.mean(far_res * far_res, axis=0))

    arrays = {
        "X": X,
        "Y": Y,
        "G": G,
        "t_vals": t_vals,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_vals": z_vals,
        "a0": np.asarray([a0], dtype=np.float64),
        "r0": np.asarray([r0], dtype=np.float64),
        "omega": np.asarray([omega], dtype=np.float64),
        "phase": np.asarray([phase], dtype=np.float64),
        "div_e_rms": np.asarray([div_e_rms], dtype=np.float64),
        "div_b_rms": np.asarray([div_b_rms], dtype=np.float64),
        "amp_rms": np.asarray(amp_rms, dtype=np.float64),
        "far_rms": np.asarray(far_rms, dtype=np.float64),
    }

    meta = {
        "a0": float(a0),
        "r0": float(r0),
        "omega": float(omega),
        "phase": float(phase),
        "nt": int(nt),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz if nz > 0 else 1),
        "n_points": int(n),
        "div_e_rms": float(div_e_rms),
        "div_b_rms": float(div_b_rms),
        "ampere_rms_x": float(amp_rms[0]),
        "ampere_rms_y": float(amp_rms[1]),
        "ampere_rms_z": float(amp_rms[2]),
        "faraday_rms_x": float(far_rms[0]),
        "faraday_rms_y": float(far_rms[1]),
        "faraday_rms_z": float(far_rms[2]),
    }
    return arrays, meta


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_output = repo_root / "data" / "maxwell" / "fake_maxwell_wire_source.npz"

    parser = argparse.ArgumentParser(
        description="Generate source-driven Maxwell toy data (wire-inspired)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output .npz path")
    parser.add_argument("--a0", type=float, default=1.0, help="Vector-potential amplitude")
    parser.add_argument("--r0", type=float, default=0.7, help="Gaussian core radius")
    parser.add_argument("--omega", type=float, default=1.2, help="AC angular frequency")
    parser.add_argument("--phase", type=float, default=0.3, help="AC phase")
    parser.add_argument("--nt", type=int, default=20, help="Number of time samples")
    parser.add_argument("--nx", type=int, default=31, help="Number of x samples")
    parser.add_argument("--ny", type=int, default=31, help="Number of y samples")
    parser.add_argument("--nz", type=int, default=1, help="Number of z samples")
    parser.add_argument("--t_max", type=float, default=2.0 * math.pi, help="Time domain max")
    parser.add_argument("--x_max", type=float, default=2.5, help="Half-width x domain")
    parser.add_argument("--y_max", type=float, default=2.5, help="Half-width y domain")
    parser.add_argument("--z_max", type=float, default=0.0, help="Half-width z domain if nz > 1")
    args = parser.parse_args()

    arrays, meta = build_wire_dataset(
        a0=args.a0,
        r0=args.r0,
        omega=args.omega,
        phase=args.phase,
        nt=args.nt,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        t_max=args.t_max,
        x_max=args.x_max,
        y_max=args.y_max,
        z_max=args.z_max,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("=" * 84)
    print("Generated Maxwell wire-source toy dataset")
    print("=" * 84)
    print(f"Output:       {args.output}")
    print(f"Metadata:     {meta_path}")
    print(f"Grid:         nt={meta['nt']}, nx={meta['nx']}, ny={meta['ny']}, nz={meta['nz']}")
    print(f"N points:     {meta['n_points']}")
    print(f"Params:       a0={meta['a0']}, r0={meta['r0']}, omega={meta['omega']}, phase={meta['phase']}")
    print(f"Gauss RMS:    div(E)={meta['div_e_rms']:.3e}, div(B)={meta['div_b_rms']:.3e}")
    print(
        "Ampere RMS:   "
        f"[{meta['ampere_rms_x']:.3e}, {meta['ampere_rms_y']:.3e}, {meta['ampere_rms_z']:.3e}]"
    )
    print(
        "Faraday RMS:  "
        f"[{meta['faraday_rms_x']:.3e}, {meta['faraday_rms_y']:.3e}, {meta['faraday_rms_z']:.3e}]"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
