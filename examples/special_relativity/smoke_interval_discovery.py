#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sr_demo_utils import (
    analyze_operational_boost_family,
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


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_xy_csv(path: Path, y: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([y, x0, x1]),
        delimiter=",",
        header="y,x0,x1",
        comments="",
    )


def _write_generated_artifacts(script_dir: Path, datasets) -> None:
    data_dir = script_dir / "data"
    uprime_dir = data_dir / "uprime"
    xprime_dir = data_dir / "xprime"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    metadata_rows = []
    for idx, dataset in enumerate(datasets):
        regime_id = beta_to_regime_id(dataset.beta)
        combined_path = data_dir / f"intervals_{regime_id}.csv"
        np.savetxt(
            combined_path,
            np.column_stack([dataset.u, dataset.x, dataset.u_prime, dataset.x_prime]),
            delimiter=",",
            header="u,x,u_prime,x_prime",
            comments="",
        )
        uprime_path = uprime_dir / f"uprime_{regime_id}.csv"
        xprime_path = xprime_dir / f"xprime_{regime_id}.csv"
        _write_xy_csv(uprime_path, dataset.u_prime, dataset.u, dataset.x)
        _write_xy_csv(xprime_path, dataset.x_prime, dataset.u, dataset.x)
        manifest_rows.append(
            {
                "regime_id": regime_id,
                "beta": float(dataset.beta),
                "combined_csv": str(combined_path),
                "uprime_csv": str(uprime_path),
                "xprime_csv": str(xprime_path),
                "n_samples": int(dataset.metadata.get("n_samples", len(dataset.u))),
            }
        )
        metadata_rows.append(
            {
                "beta": float(dataset.beta),
                "beta_sq": float(dataset.beta * dataset.beta),
                "one_minus_beta_sq": float(1.0 - dataset.beta * dataset.beta),
                "regime_index": float(idx),
            }
        )

    manifest = {
        "betas": [float(row["beta"]) for row in manifest_rows],
        "regimes": manifest_rows,
        "param_sr_metadata_rows": metadata_rows,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (data_dir / "param_sr_metadata_rows.json").write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the special-relativity interval discovery scaffold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--betas",
        type=str,
        default="-0.8,-0.6,-0.3,0.3,0.6,0.8",
        help="Comma-separated beta values",
    )
    parser.add_argument("--n_samples", type=int, default=4096, help="Samples per regime")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Noise on primed observables")
    parser.add_argument("--near_null_width", type=float, default=0.03, help="Half-width around null intervals")
    parser.add_argument("--u_max", type=float, default=10.0, help="Max |u| scale")
    parser.add_argument("--x_max", type=float, default=10.0, help="Max |x| scale")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path("results")),
        help="Directory for the summary JSON",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also regenerate the CSV artifacts under examples/special_relativity/data",
    )
    parser.add_argument("--beta_tol", type=float, default=1.0e-8, help="Tolerance on recovered beta law")
    parser.add_argument("--z_tol", type=float, default=1.0e-8, help="Tolerance on recovered 1-beta^2 law")
    parser.add_argument("--metric_tol", type=float, default=1.0e-8, help="Tolerance on recovered metric")
    args = parser.parse_args()

    datasets = [
        generate_operational_interval_dataset(
            beta,
            n_samples=int(args.n_samples),
            seed=int(args.seed) + idx,
            noise_std=float(args.noise_std),
            near_null_width=float(args.near_null_width),
            u_max=float(args.u_max),
            x_max=float(args.x_max),
        )
        for idx, beta in enumerate(_parse_betas(args.betas))
    ]
    script_dir = Path(__file__).resolve().parent
    if args.generate:
        _write_generated_artifacts(script_dir, datasets)
    summary = analyze_operational_boost_family(datasets)

    coeff = summary["coefficient_laws"]
    metric = summary["metric"]
    regime_fits = summary["regime_fits"]

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "special_relativity_interval_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print("\nRecovered regime matrices")
    for row in regime_fits:
        print(
            f"{row['regime_id']}: beta={row['beta']:+.3f} "
            f"a={row['a']:.9f} b={row['b']:.9f} c={row['c']:.9f} d={row['d']:.9f} "
            f"rmse=({row['u_rmse']:.3e}, {row['x_rmse']:.3e})"
        )

    print("\nCoefficient lift")
    print(f"max |(-b/a) - beta|         = {coeff['max_beta_residual']:.3e}")
    print(f"max |(1/a^2) - (1-beta^2)| = {coeff['max_z_residual']:.3e}")
    print(f"max |a - gamma(beta)|       = {coeff['gamma_max_abs_error']:.3e}")
    print(
        "fit r(beta) ~= c0 + c1*beta : "
        f"c0={coeff['r_linear_coeffs'][0]:+.6e}, c1={coeff['r_linear_coeffs'][1]:+.6e}"
    )
    print(
        "fit z(beta) ~= c0 + c2*beta^2 : "
        f"c0={coeff['z_even_coeffs'][0]:+.6e}, c2={coeff['z_even_coeffs'][1]:+.6e}"
    )

    recovered_metric = np.asarray(metric["metric"], dtype=np.float64)
    print("\nRecovered invariant form")
    print(recovered_metric)
    print(f"quadratic coeffs = {metric['quadratic_coeffs']}")
    print(f"is indefinite    = {metric['is_indefinite']}")
    print(f"max preserve err = {metric['max_preservation_error']:.3e}")
    print(f"summary json     = {summary_path}")

    if coeff["max_beta_residual"] > float(args.beta_tol):
        print(
            "FAIL: recovered beta law exceeded tolerance "
            f"({coeff['max_beta_residual']:.3e} > {float(args.beta_tol):.3e})"
        )
        return 1
    if coeff["max_z_residual"] > float(args.z_tol):
        print(
            "FAIL: recovered z law exceeded tolerance "
            f"({coeff['max_z_residual']:.3e} > {float(args.z_tol):.3e})"
        )
        return 1
    if not bool(metric["is_indefinite"]):
        print("FAIL: recovered metric is not indefinite")
        return 1
    if metric["max_preservation_error"] > float(args.metric_tol):
        print(
            "FAIL: invariant-form preservation error exceeded tolerance "
            f"({metric['max_preservation_error']:.3e} > {float(args.metric_tol):.3e})"
        )
        return 1

    expected_metric = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    if not np.allclose(recovered_metric, expected_metric, atol=float(args.metric_tol)):
        print(
            "FAIL: recovered metric deviates from diag(1,-1) beyond tolerance "
            f"(tol={float(args.metric_tol):.3e})"
        )
        return 1

    print("\nPASS: operational interval data lock onto the Lorentz-family lift and Minkowski form")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
