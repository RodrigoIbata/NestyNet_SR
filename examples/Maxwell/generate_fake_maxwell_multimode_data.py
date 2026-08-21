#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate synthetic *multi-mode* vacuum Maxwell field data.

A superposition of source-free vacuum plane waves is still an exact vacuum
Maxwell solution (Maxwell is linear).  Each mode m is transverse and uses the
vacuum dispersion (c = 1):

    phase_m = k_m . x - |k_m| t + phi_m
    E_m     = a_m  p_m            sin(phase_m),   with k_m . p_m = 0
    B_m     = a_m (k_hat_m x p_m) sin(phase_m)
    E = sum_m E_m,   B = sum_m B_m

The point of the *multi-mode* field (versus the single linearly-polarized
plane wave in ``generate_fake_maxwell_data.py``) is identifiability of the
broadened discovery library: because

    lap E = - sum_m |k_m|^2 E_m,

a field built from several distinct |k_m| makes ``lap E`` *not* a scalar
multiple of ``E``.  The single-mode wave has ``lap E = -|k|^2 E`` exactly, which
aliases the ``lap E`` and ``E`` library columns and makes the design matrix
rank-deficient.  Mixing modes across components breaks that alias.

Arrays saved in the output .npz (same layout as the single-mode generator):
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

# Default mode list: (k-vector, polarization p, amplitude, phase).
# Each p is orthogonal to its k (transverse / divergence-free).  Distinct |k|
# values (1, 2, 1) are spread across components so that, e.g., Ey carries both
# |k|=2 and |k|=1 content -> lap(E) is not proportional to E.  Integer k keeps
# the field periodic on a [0, 2*pi)^3 box (endpoint=False) for exact FFT
# spectral derivatives.
DEFAULT_MODES: tuple[tuple[tuple[float, float, float], tuple[float, float, float], float, float], ...] = (
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0, 0.0),   # along z, x-pol, |k|=1
    ((0.0, 0.0, 2.0), (0.0, 1.0, 0.0), 0.8, 0.3),   # along z, y-pol, |k|=2
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.6, -0.4),  # along x, y-pol, |k|=1
)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


def build_multimode_vacuum_dataset(
    *,
    modes=DEFAULT_MODES,
    nt: int,
    nx: int,
    ny: int,
    nz: int,
    t_max: float,
    box: float = 2.0 * math.pi,
    noise_std: float = 0.0,
    seed: int = 0,
    tol: float = 1e-9,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Build coordinates, fields, and gradients for the multi-mode vacuum field."""
    # endpoint=False -> [0, box) is exactly one spatial period for integer k.
    t_vals = np.linspace(0.0, t_max, nt, dtype=np.float64)
    x_vals = np.linspace(0.0, box, nx, endpoint=False, dtype=np.float64)
    y_vals = np.linspace(0.0, box, ny, endpoint=False, dtype=np.float64)
    z_vals = np.linspace(0.0, box, nz, endpoint=False, dtype=np.float64)
    tt, xx, yy, zz = np.meshgrid(t_vals, x_vals, y_vals, z_vals, indexing="ij")
    n = int(tt.size)

    coords = [tt, xx, yy, zz]  # axis 0 = t, axes 1..3 = x,y,z
    E = [np.zeros_like(tt) for _ in range(3)]
    B = [np.zeros_like(tt) for _ in range(3)]
    # First derivatives: dE[c][axis], dB[c][axis], axis in 0..3 (t,x,y,z)
    dE = [[np.zeros_like(tt) for _ in range(4)] for _ in range(3)]
    dB = [[np.zeros_like(tt) for _ in range(4)] for _ in range(3)]

    mode_meta = []
    for kvec, pvec, amp, phi in modes:
        k = np.asarray(kvec, dtype=np.float64)
        p = np.asarray(pvec, dtype=np.float64)
        kmag = float(np.linalg.norm(k))
        if kmag <= 0.0:
            raise ValueError("each mode needs a non-zero k vector")
        if abs(float(np.dot(k, p))) > tol * max(1.0, kmag):
            raise ValueError(f"mode polarization {pvec} is not transverse to k={kvec} (k.p != 0)")
        khat = k / kmag
        omega = kmag  # c = 1
        bvec = _cross(khat, p)

        phase = k[0] * xx + k[1] * yy + k[2] * zz - omega * tt + float(phi)
        s = np.sin(phase)
        cphase = np.cos(phase)

        # d(phase)/d x_axis : [-omega, k_x, k_y, k_z]
        dphase = [-omega, k[0], k[1], k[2]]
        for c in range(3):
            E[c] += amp * p[c] * s
            B[c] += amp * bvec[c] * s
            for ax in range(4):
                dE[c][ax] += amp * p[c] * cphase * dphase[ax]
                dB[c][ax] += amp * bvec[c] * cphase * dphase[ax]
        mode_meta.append(
            {"k": [float(v) for v in k], "p": [float(v) for v in p], "amp": float(amp),
             "phi": float(phi), "kmag": kmag, "B_dir": [float(v) for v in bvec]}
        )

    X = np.zeros((n, 4), dtype=np.float64)
    for ax in range(4):
        X[:, ax] = coords[ax].reshape(-1)

    Y = np.zeros((n, 6), dtype=np.float64)
    G = np.zeros((n, 6, 4), dtype=np.float64)
    for c in range(3):
        Y[:, c] = E[c].reshape(-1)        # Ex,Ey,Ez
        Y[:, 3 + c] = B[c].reshape(-1)    # Bx,By,Bz
        for ax in range(4):
            G[:, c, ax] = dE[c][ax].reshape(-1)
            G[:, 3 + c, ax] = dB[c][ax].reshape(-1)

    # Maxwell-consistency diagnostics (clean, before noise):
    #   dE/dt - curl(B) = 0,  dB/dt + curl(E) = 0
    curl_B = np.stack(
        [
            G[:, 5, 2] - G[:, 4, 3],  # dBz/dy - dBy/dz
            G[:, 3, 3] - G[:, 5, 1],  # dBx/dz - dBz/dx
            G[:, 4, 1] - G[:, 3, 2],  # dBy/dx - dBx/dy
        ],
        axis=1,
    )
    curl_E = np.stack(
        [
            G[:, 2, 2] - G[:, 1, 3],
            G[:, 0, 3] - G[:, 2, 1],
            G[:, 1, 1] - G[:, 0, 2],
        ],
        axis=1,
    )
    amp_res = G[:, 0:3, 0] - curl_B
    far_res = G[:, 3:6, 0] + curl_E
    div_e = G[:, 0, 1] + G[:, 1, 2] + G[:, 2, 3]
    div_b = G[:, 3, 1] + G[:, 4, 2] + G[:, 5, 3]
    amp_rms = float(np.sqrt(np.mean(amp_res * amp_res)))
    far_rms = float(np.sqrt(np.mean(far_res * far_res)))
    div_e_rms = float(np.sqrt(np.mean(div_e * div_e)))
    div_b_rms = float(np.sqrt(np.mean(div_b * div_b)))

    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        Y = Y + rng.normal(0.0, noise_std, size=Y.shape)

    arrays = {
        "X": X,
        "Y": Y,
        "G": G,
        "t_vals": t_vals,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_vals": z_vals,
        "noise_std": np.asarray([noise_std], dtype=np.float64),
    }
    meta = {
        "c": 1.0,
        "nt": int(nt),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "n_points": int(n),
        "t_max": float(t_max),
        "box": float(box),
        "noise_std": float(noise_std),
        "modes": mode_meta,
        "ampere_residual_rms": amp_rms,
        "faraday_residual_rms": far_rms,
        "div_e_rms": div_e_rms,
        "div_b_rms": div_b_rms,
    }
    return arrays, meta


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_output = repo_root / "data" / "maxwell" / "fake_maxwell_multimode.npz"

    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-mode vacuum Maxwell data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output .npz path")
    parser.add_argument("--nt", type=int, default=16, help="Number of time samples")
    parser.add_argument("--nx", type=int, default=8, help="Number of x samples")
    parser.add_argument("--ny", type=int, default=8, help="Number of y samples")
    parser.add_argument("--nz", type=int, default=8, help="Number of z samples")
    parser.add_argument("--t_max", type=float, default=2.0 * math.pi, help="Time domain max")
    parser.add_argument("--box", type=float, default=2.0 * math.pi, help="Spatial box length")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Additive noise std for Y")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise")
    args = parser.parse_args()

    arrays, meta = build_multimode_vacuum_dataset(
        nt=args.nt,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        t_max=args.t_max,
        box=args.box,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("=" * 72)
    print("Generated synthetic multi-mode vacuum Maxwell data")
    print("=" * 72)
    print(f"Output:      {args.output}")
    print(f"Points:      {meta['n_points']} ({meta['nt']} x {meta['nx']} x {meta['ny']} x {meta['nz']})")
    print(f"Modes:       {len(meta['modes'])} (|k| = {[m['kmag'] for m in meta['modes']]})")
    print(f"Ampere res:  {meta['ampere_residual_rms']:.3e}")
    print(f"Faraday res: {meta['faraday_residual_rms']:.3e}")
    print(f"div(E),div(B): {meta['div_e_rms']:.3e}, {meta['div_b_rms']:.3e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
