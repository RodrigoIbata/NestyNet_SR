#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Generate synthetic damped harmonic oscillator (DHO) data for DE discovery.

Equation:
    y'' + gamma*y' + omega^2*y = 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def _dho_rhs(_t: float, state: np.ndarray, gamma: float, omega: float) -> list[float]:
    y, v = float(state[0]), float(state[1])
    return [v, -(gamma * v) - (omega * omega * y)]


def generate_dho_data(
    *,
    gamma: float,
    omega: float,
    y0: float,
    v0: float,
    t_min: float,
    t_max: float,
    n_points: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(float(t_min), float(t_max), int(n_points), dtype=np.float64)
    sol = solve_ivp(
        _dho_rhs,
        [float(t_min), float(t_max)],
        [float(y0), float(v0)],
        args=(float(gamma), float(omega)),
        t_eval=t,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    y = np.asarray(sol.y[0], dtype=np.float64)

    if float(noise) > 0.0:
        rng = np.random.default_rng(int(seed))
        y = y + rng.normal(loc=0.0, scale=float(noise) * float(y.std()), size=y.shape)

    return t, y


def _write_csv(path: Path, y: np.ndarray, x0: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "y,x0"
    # Column order is (y, x0) to match NestyNet CSV conventions.
    np.savetxt(
        str(path),
        np.column_stack([y, x0]),
        delimiter=",",
        header=header,
        comments="",
    )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_output = repo_root / "data" / "dho.csv"

    parser = argparse.ArgumentParser(
        description="Generate damped harmonic oscillator data for DE discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gamma", type=float, default=0.4, help="Damping coefficient")
    parser.add_argument("--omega", type=float, default=2.0, help="Natural angular frequency")
    parser.add_argument("--y0", type=float, default=1.0, help="Initial displacement")
    parser.add_argument("--v0", type=float, default=0.2, help="Initial velocity")
    parser.add_argument("--t_min", type=float, default=0.0, help="Minimum x0 value")
    parser.add_argument("--t_max", type=float, default=10.0, help="Maximum x0 value")
    parser.add_argument("--n_points", type=int, default=5000, help="Number of sampled points")
    parser.add_argument("--noise", type=float, default=0.0, help="Relative Gaussian noise")
    parser.add_argument("--seed", type=int, default=42, help="Noise seed")
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="Output CSV path",
    )
    args = parser.parse_args()

    t, y = generate_dho_data(
        gamma=args.gamma,
        omega=args.omega,
        y0=args.y0,
        v0=args.v0,
        t_min=args.t_min,
        t_max=args.t_max,
        n_points=args.n_points,
        noise=args.noise,
        seed=args.seed,
    )

    out_path = Path(args.output).resolve()
    _write_csv(out_path, y, t)

    meta_path = out_path.with_suffix(".meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("Damped Harmonic Oscillator (DHO) Synthetic Data\n")
        f.write("=" * 56 + "\n\n")
        f.write("Equation:\n")
        f.write("  y'' + gamma*y' + omega^2*y = 0\n\n")
        f.write("Parameters:\n")
        f.write(f"  gamma = {float(args.gamma)}\n")
        f.write(f"  omega = {float(args.omega)}\n")
        f.write(f"  y0    = {float(args.y0)}\n")
        f.write(f"  v0    = {float(args.v0)}\n")
        f.write(f"  x0 range = [{float(args.t_min)}, {float(args.t_max)}]\n")
        f.write(f"  points   = {int(args.n_points)}\n")
        f.write(f"  noise    = {float(args.noise)}\n\n")
        f.write("Expected DE discovery (implicit):\n")
        f.write("  u_x0x0 + c_u*u + c_du*u_x0 = 0\n")
        f.write(f"  c_u  ~= omega^2 = {float(args.omega) ** 2:.6g}\n")
        f.write(f"  c_du ~= gamma   = {float(args.gamma):.6g}\n")

    print("=" * 70)
    print("Generated DHO data")
    print("=" * 70)
    print(f"Output CSV: {out_path}")
    print(f"Metadata:   {meta_path}")
    print("Header:     y,x0")
    print(f"y range:    [{float(y.min()):.6g}, {float(y.max()):.6g}]")
    print(f"x0 range:   [{float(t.min()):.6g}, {float(t.max()):.6g}]")
    print("\nNext step:")
    print(
        "  python examples/dho/smoke_dho_discovery.py "
        f"--datafile {out_path}"
    )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
