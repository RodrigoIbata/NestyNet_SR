#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sr_demo_utils import (
    beta_to_regime_id,
    generate_operational_interval_dataset,
)


def _parse_betas(spec: str) -> list[float]:
    out = []
    for chunk in str(spec).split(","):
        token = chunk.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("expected at least one beta value")
    return out


def _write_xy_csv(path: Path, y: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([y, x0, x1]),
        delimiter=",",
        header="y,x0,x1",
        comments="",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 1+1D special-relativity interval datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--betas",
        type=str,
        default="-0.8,-0.6,-0.3,0.3,0.6,0.8",
        help="Comma-separated relative velocities beta=v/c",
    )
    parser.add_argument("--n_samples", type=int, default=4096, help="Samples per regime")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    parser.add_argument("--u_max", type=float, default=10.0, help="Max |u| scale for interval sampling")
    parser.add_argument("--x_max", type=float, default=10.0, help="Max |x| scale for interval sampling")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise applied to primed observables")
    parser.add_argument("--near_null_width", type=float, default=0.03, help="Half-width around the light cone")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    uprime_dir = output_dir / "uprime"
    xprime_dir = output_dir / "xprime"
    output_dir.mkdir(parents=True, exist_ok=True)
    uprime_dir.mkdir(parents=True, exist_ok=True)
    xprime_dir.mkdir(parents=True, exist_ok=True)

    for stale_path in output_dir.glob("intervals_*.csv"):
        stale_path.unlink()
    for target_dir in (uprime_dir, xprime_dir):
        for stale_path in target_dir.glob("*.csv"):
            stale_path.unlink()

    manifest_rows = []
    metadata_rows = []
    combined_paths = []

    for idx, beta in enumerate(_parse_betas(args.betas)):
        dataset = generate_operational_interval_dataset(
            beta,
            n_samples=int(args.n_samples),
            seed=int(args.seed) + idx,
            u_max=float(args.u_max),
            x_max=float(args.x_max),
            near_null_width=float(args.near_null_width),
            noise_std=float(args.noise_std),
        )

        regime_id = beta_to_regime_id(beta)
        combined_path = output_dir / f"intervals_{regime_id}.csv"
        np.savetxt(
            combined_path,
            np.column_stack([dataset.u, dataset.x, dataset.u_prime, dataset.x_prime]),
            delimiter=",",
            header="u,x,u_prime,x_prime",
            comments="",
        )
        combined_paths.append(str(combined_path))

        uprime_path = uprime_dir / f"uprime_{regime_id}.csv"
        xprime_path = xprime_dir / f"xprime_{regime_id}.csv"
        _write_xy_csv(uprime_path, dataset.u_prime, dataset.u, dataset.x)
        _write_xy_csv(xprime_path, dataset.x_prime, dataset.u, dataset.x)

        manifest_rows.append(
            {
                "regime_id": regime_id,
                "beta": float(beta),
                "combined_csv": str(combined_path),
                "uprime_csv": str(uprime_path),
                "xprime_csv": str(xprime_path),
                "n_samples": int(args.n_samples),
            }
        )
        metadata_rows.append(
            {
                "beta": float(beta),
                "beta_sq": float(beta * beta),
                "one_minus_beta_sq": float(1.0 - beta * beta),
                "regime_index": float(idx),
            }
        )

    manifest = {
        "betas": [float(row["beta"]) for row in manifest_rows],
        "regimes": manifest_rows,
        "combined_csvs": combined_paths,
        "param_sr_metadata_rows": metadata_rows,
    }

    manifest_path = output_dir / "manifest.json"
    metadata_path = output_dir / "param_sr_metadata_rows.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote metadata rows: {metadata_path}")
    for row in manifest_rows:
        print(
            f"{row['regime_id']}: beta={row['beta']:+.3f} "
            f"combined={row['combined_csv']} "
            f"u'={row['uprime_csv']} x'={row['xprime_csv']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
