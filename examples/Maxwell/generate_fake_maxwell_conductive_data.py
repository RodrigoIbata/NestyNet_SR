#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate conductive-medium Maxwell data from exact damped wave modes.

Normalized source-free conductive system:
    dE/dt - curl(B) + sigma * E = 0
    dB/dt + curl(E) = 0
    div(E) = div(B) = 0

We superpose 3 orthogonal damped plane-wave modes:
  mode 1: propagates in z, contributes (Ex, By)
  mode 2: propagates in x, contributes (Ey, Bz)
  mode 3: propagates in y, contributes (Ez, Bx)

For each mode with wavenumber k, conductivity sigma:
    omega solves: omega^2 + i*sigma*omega - k^2 = 0
We pick the oscillatory branch omega = omega_r - i*sigma/2 (k > sigma/2).

Saved arrays:
    X : (N, 4)    [t, x, y, z]
    Y : (N, 6)    [Ex, Ey, Ez, Bx, By, Bz]
    G : (N, 6, 4) dY_i/dX_j
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _omega_conductive(k: float, sigma: float) -> complex:
    disc = 4.0 * k * k - sigma * sigma
    if disc <= 0.0:
        raise ValueError(
            f"Need oscillatory regime k > sigma/2. Got k={k}, sigma={sigma} (disc={disc})."
        )
    omega_r = 0.5 * math.sqrt(disc)
    return complex(omega_r, -0.5 * sigma)


def build_conductive_dataset(
    *,
    sigma: float,
    a1: float,
    a2: float,
    a3: float,
    k1: float,
    k2: float,
    k3: float,
    phi1: float,
    phi2: float,
    phi3: float,
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
    # Periodic spatial axes: endpoint=False gives spacing 2*pi/n (not 2*pi/(n-1))
    # and avoids duplicating the 0 and 2*pi samples, which otherwise wastes a
    # point and coarsens the effective grid (modified-wavenumber dispersion).
    x_vals = np.linspace(0.0, x_max, nx, endpoint=False, dtype=np.float64)
    y_vals = np.linspace(0.0, y_max, ny, endpoint=False, dtype=np.float64)
    z_vals = np.linspace(0.0, z_max, nz, endpoint=False, dtype=np.float64)
    tt, xx, yy, zz = np.meshgrid(t_vals, x_vals, y_vals, z_vals, indexing="ij")

    n = int(tt.size)

    # Complex field accumulators for exact linear superposition.
    e_c = np.zeros((3,) + tt.shape, dtype=np.complex128)
    b_c = np.zeros((3,) + tt.shape, dtype=np.complex128)
    de_dt_c = np.zeros_like(e_c)
    db_dt_c = np.zeros_like(b_c)
    de_dx_c = np.zeros((3, 4) + tt.shape, dtype=np.complex128)
    db_dx_c = np.zeros((3, 4) + tt.shape, dtype=np.complex128)

    # (k, A, phase, spatial_axis, Ecomp, Bcomp)
    # axes: t=0, x=1, y=2, z=3
    modes = [
        (k1, a1, phi1, 3, 0, 1),  # Ex / By, propagate z
        (k2, a2, phi2, 1, 1, 2),  # Ey / Bz, propagate x
        (k3, a3, phi3, 2, 2, 0),  # Ez / Bx, propagate y
    ]

    coords = {1: xx, 2: yy, 3: zz}
    omegas = []
    for k, amp, phase, ax, ecomp, bcomp in modes:
        omega = _omega_conductive(float(k), float(sigma))
        omegas.append(omega)

        theta = k * coords[ax] - omega * tt + phase
        exp_i = np.exp(1j * theta)

        e_mode = amp * exp_i
        b_mode = (k / omega) * amp * exp_i

        e_c[ecomp] += e_mode
        b_c[bcomp] += b_mode

        de_dt_c[ecomp] += (-1j * omega) * e_mode
        db_dt_c[bcomp] += (-1j * omega) * b_mode

        de_dx_c[ecomp, ax] += (1j * k) * e_mode
        db_dx_c[bcomp, ax] += (1j * k) * b_mode

    e = np.real(e_c)
    b = np.real(b_c)
    de_dt = np.real(de_dt_c)
    db_dt = np.real(db_dt_c)
    de_dx = np.real(de_dx_c)
    db_dx = np.real(db_dx_c)

    X = np.stack(
        [tt.reshape(-1), xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)],
        axis=1,
    ).astype(np.float64, copy=False)

    Y = np.stack(
        [
            e[0].reshape(-1),
            e[1].reshape(-1),
            e[2].reshape(-1),
            b[0].reshape(-1),
            b[1].reshape(-1),
            b[2].reshape(-1),
        ],
        axis=1,
    ).astype(np.float64, copy=False)

    G = np.zeros((n, 6, 4), dtype=np.float64)
    # E gradients
    for c in range(3):
        G[:, c, 0] = de_dt[c].reshape(-1)
        for ax in (1, 2, 3):
            G[:, c, ax] = de_dx[c, ax].reshape(-1)
    # B gradients
    for c in range(3):
        G[:, 3 + c, 0] = db_dt[c].reshape(-1)
        for ax in (1, 2, 3):
            G[:, 3 + c, ax] = db_dx[c, ax].reshape(-1)

    div_e = G[:, 0, 1] + G[:, 1, 2] + G[:, 2, 3]
    div_b = G[:, 3, 1] + G[:, 4, 2] + G[:, 5, 3]
    div_e_rms = float(np.sqrt(np.mean(div_e * div_e)))
    div_b_rms = float(np.sqrt(np.mean(div_b * div_b)))

    dE_dt = G[:, 0:3, 0]
    dB_dt = G[:, 3:6, 0]
    E = Y[:, 0:3]

    curl_b = np.zeros((n, 3), dtype=np.float64)
    curl_b[:, 0] = G[:, 5, 2] - G[:, 4, 3]
    curl_b[:, 1] = G[:, 3, 3] - G[:, 5, 1]
    curl_b[:, 2] = G[:, 4, 1] - G[:, 3, 2]

    curl_e = np.zeros((n, 3), dtype=np.float64)
    curl_e[:, 0] = G[:, 2, 2] - G[:, 1, 3]
    curl_e[:, 1] = G[:, 0, 3] - G[:, 2, 1]
    curl_e[:, 2] = G[:, 1, 1] - G[:, 0, 2]

    amp_res = dE_dt - curl_b + sigma * E
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
        "sigma": np.asarray([sigma], dtype=np.float64),
        "A": np.asarray([a1, a2, a3], dtype=np.float64),
        "k": np.asarray([k1, k2, k3], dtype=np.float64),
        "phi": np.asarray([phi1, phi2, phi3], dtype=np.float64),
        "omega_real": np.asarray([np.real(w) for w in omegas], dtype=np.float64),
        "omega_imag": np.asarray([np.imag(w) for w in omegas], dtype=np.float64),
        "div_e_rms": np.asarray([div_e_rms], dtype=np.float64),
        "div_b_rms": np.asarray([div_b_rms], dtype=np.float64),
        "amp_rms": np.asarray(amp_rms, dtype=np.float64),
        "far_rms": np.asarray(far_rms, dtype=np.float64),
    }

    meta = {
        "sigma": float(sigma),
        "a1": float(a1),
        "a2": float(a2),
        "a3": float(a3),
        "k1": float(k1),
        "k2": float(k2),
        "k3": float(k3),
        "phi1": float(phi1),
        "phi2": float(phi2),
        "phi3": float(phi3),
        "omega1_real": float(np.real(omegas[0])),
        "omega2_real": float(np.real(omegas[1])),
        "omega3_real": float(np.real(omegas[2])),
        "omega1_imag": float(np.imag(omegas[0])),
        "omega2_imag": float(np.imag(omegas[1])),
        "omega3_imag": float(np.imag(omegas[2])),
        "nt": int(nt),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
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
    default_output = repo_root / "data" / "maxwell" / "fake_maxwell_conductive.npz"

    parser = argparse.ArgumentParser(
        description="Generate conductive-medium Maxwell toy data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output .npz path")
    parser.add_argument("--sigma", type=float, default=0.6, help="Conductivity coefficient")
    parser.add_argument("--a1", type=float, default=1.0, help="Mode-1 amplitude")
    parser.add_argument("--a2", type=float, default=0.8, help="Mode-2 amplitude")
    parser.add_argument("--a3", type=float, default=0.7, help="Mode-3 amplitude")
    parser.add_argument("--k1", type=float, default=1.2, help="Mode-1 wavenumber")
    parser.add_argument("--k2", type=float, default=1.6, help="Mode-2 wavenumber")
    parser.add_argument("--k3", type=float, default=2.0, help="Mode-3 wavenumber")
    parser.add_argument("--phi1", type=float, default=0.1, help="Mode-1 phase")
    parser.add_argument("--phi2", type=float, default=0.7, help="Mode-2 phase")
    parser.add_argument("--phi3", type=float, default=-0.5, help="Mode-3 phase")
    parser.add_argument("--nt", type=int, default=16, help="Number of time samples")
    parser.add_argument("--nx", type=int, default=10, help="Number of x samples")
    parser.add_argument("--ny", type=int, default=10, help="Number of y samples")
    parser.add_argument("--nz", type=int, default=10, help="Number of z samples")
    parser.add_argument("--t_max", type=float, default=8.0, help="Time domain max")
    parser.add_argument("--x_max", type=float, default=2.0 * math.pi, help="X domain max")
    parser.add_argument("--y_max", type=float, default=2.0 * math.pi, help="Y domain max")
    parser.add_argument("--z_max", type=float, default=2.0 * math.pi, help="Z domain max")
    args = parser.parse_args()

    arrays, meta = build_conductive_dataset(
        sigma=args.sigma,
        a1=args.a1,
        a2=args.a2,
        a3=args.a3,
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
        phi1=args.phi1,
        phi2=args.phi2,
        phi3=args.phi3,
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
    print("Generated conductive-medium Maxwell synthetic dataset")
    print("=" * 84)
    print(f"Output:       {args.output}")
    print(f"Metadata:     {meta_path}")
    print(f"Grid:         nt={meta['nt']}, nx={meta['nx']}, ny={meta['ny']}, nz={meta['nz']}")
    print(f"N points:     {meta['n_points']}")
    print(f"Sigma:        {meta['sigma']}")
    print(f"Wavenumbers:  ({meta['k1']}, {meta['k2']}, {meta['k3']})")
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
