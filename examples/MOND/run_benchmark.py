#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
MOND PDE benchmark runner.

For each configured problem:
1. Generate synthetic data from a manufactured MOND field.
2. Fit a sparse linear model (STLSQ) for rho from candidate features.
3. Validate recovered coefficients against the known MOND structure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from problem_defs import (
    MONDProblem,
    MondDataset,
    generate_dataset,
    ground_truth_for_problem,
    load_problems,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ridge_solve(theta: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    k = int(theta.shape[1])
    gram = theta.T @ theta
    gram += float(ridge) * np.eye(k, dtype=np.float64)
    rhs = theta.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(theta, y, rcond=None)[0]


def stlsq(
    theta: np.ndarray,
    y: np.ndarray,
    *,
    lam: float = 1e-2,
    ridge: float = 1e-12,
    max_iter: int = 20,
) -> np.ndarray:
    k = int(theta.shape[1])
    active = np.ones(k, dtype=bool)
    coeffs = np.zeros(k, dtype=np.float64)

    for _ in range(max_iter):
        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        c_active = _ridge_solve(theta[:, idx], y, ridge)

        coeffs_new = np.zeros(k, dtype=np.float64)
        coeffs_new[idx] = c_active
        active_new = np.abs(coeffs_new) >= float(lam)

        if np.array_equal(active_new, active):
            coeffs = coeffs_new
            break

        active = active_new
        coeffs = coeffs_new

    if np.any(active):
        idx = np.flatnonzero(active)
        c_active = _ridge_solve(theta[:, idx], y, ridge)
        coeffs = np.zeros(k, dtype=np.float64)
        coeffs[idx] = c_active

    return coeffs


def fit_baseline(
    dataset: MondDataset,
    *,
    stlsq_lambda: float,
    ridge: float,
    stlsq_max_iter: int,
) -> tuple[dict[str, float], dict]:
    theta = np.asarray(dataset.theta, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.float64)
    col_scales = np.sqrt(np.mean(theta**2, axis=0))
    col_scales = np.where(col_scales > 1e-12, col_scales, 1.0)
    theta_scaled = theta / col_scales

    coeffs_scaled = stlsq(
        theta_scaled,
        y,
        lam=stlsq_lambda,
        ridge=ridge,
        max_iter=stlsq_max_iter,
    )
    coeffs = coeffs_scaled / col_scales
    pred = np.einsum("ij,j->i", theta, coeffs)
    residual = pred - y

    rms = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    y_centered = y - float(np.mean(y))
    ss_tot = float(np.sum(y_centered**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    coeff_map = {
        name: float(coeffs[j]) for j, name in enumerate(dataset.feature_names)
    }
    metrics = {
        "rms": rms,
        "r2": float(r2),
        "n_samples": int(y.shape[0]),
    }
    return coeff_map, metrics


def format_equation(
    coeff_map: dict[str, float],
    *,
    ordered_terms: list[str],
    min_abs: float = 1e-6,
) -> str:
    parts = []
    for name in ordered_terms:
        c = float(coeff_map.get(name, 0.0))
        if abs(c) < float(min_abs):
            continue
        parts.append(f"{c:+.6g}*{name}")
    if not parts:
        return "rho = 0"
    return "rho = " + " ".join(parts)


def validate(
    problem: MONDProblem,
    coeff_map: dict[str, float],
    metrics: dict,
) -> tuple[str, str]:
    gt = ground_truth_for_problem(problem)

    issues: list[str] = []
    critical = False
    for term, expected in gt.expected_terms.items():
        got = float(coeff_map.get(term, 0.0))
        abs_err = abs(got - expected)
        rel_err = abs_err / max(abs(expected), 1e-12)

        if got * expected <= 0.0:
            critical = True
            issues.append(
                f"{term}: wrong sign (expected {expected:+.4e}, got {got:+.4e})"
            )
            continue

        if abs(got) < 0.5 * abs(expected):
            critical = True
            issues.append(
                f"{term}: too small (expected {expected:+.4e}, got {got:+.4e})"
            )
            continue

        if rel_err > gt.coeff_rtol and abs_err > gt.coeff_atol:
            issues.append(
                f"{term}: expected {expected:+.4e}, got {got:+.4e} "
                f"(rel={rel_err:.1%})"
            )

    expected_keys = set(gt.expected_terms.keys())
    decoys = [k for k in coeff_map if k not in expected_keys]
    bad_decoys = [k for k in decoys if abs(coeff_map[k]) > gt.decoy_atol]
    if bad_decoys:
        issues.append(
            "large decoys: " + ", ".join(f"{k}={coeff_map[k]:+.3e}" for k in bad_decoys)
        )

    rms = float(metrics.get("rms", math.inf))
    if rms > gt.rms_tol:
        issues.append(f"rms too high: {rms:.3e} (tol={gt.rms_tol:.3e})")

    if critical:
        return "FAIL", "; ".join(issues)
    if issues:
        return "PARTIAL", "; ".join(issues)
    return "PASS", f"rms={rms:.3e}, r2={metrics.get('r2', float('nan')):.6f}"


def run_problem(
    problem: MONDProblem,
    *,
    data_dir: Path,
    results_dir: Path,
    fast: bool,
    skip_generate: bool,
    noise_override: float | None,
    seed: int,
    stlsq_lambda: float,
    stlsq_max_iter: int,
    ridge: float,
    interior_pad: int,
    verbose: bool,
) -> dict:
    pid = problem.id
    npz_path = data_dir / f"mond{pid}.npz"
    csv_path = data_dir / f"mond{pid}.csv"
    meta_path = data_dir / f"mond{pid}.meta.json"

    result: dict = {
        "id": pid,
        "description": problem.description,
        "status": "ERROR",
        "message": "",
    }

    try:
        if (not skip_generate) or (not npz_path.exists()):
            grid_size = 64 if fast else problem.grid_size
            dataset = generate_dataset(
                problem,
                noise_std=noise_override,
                seed=seed,
                grid_size=grid_size,
                interior_pad=interior_pad,
            )
            dataset.save(npz_path=npz_path, csv_path=csv_path, meta_path=meta_path)
            if verbose:
                print(f"  Generated: {npz_path}")
        else:
            dataset = MondDataset.load(npz_path=npz_path, meta_path=meta_path)
            if verbose:
                print(f"  Loaded:    {npz_path}")
    except Exception as exc:
        result["message"] = f"Dataset generation/load failed: {exc}"
        return result

    try:
        coeff_map, metrics = fit_baseline(
            dataset,
            stlsq_lambda=stlsq_lambda,
            ridge=ridge,
            stlsq_max_iter=stlsq_max_iter,
        )
    except Exception as exc:
        result["message"] = f"Baseline fit failed: {exc}"
        return result

    status, message = validate(problem, coeff_map, metrics)
    canonical = format_equation(
        coeff_map,
        ordered_terms=dataset.feature_names,
        min_abs=max(1e-6, 0.5 * stlsq_lambda),
    )

    result.update(
        {
            "status": status,
            "message": message,
            "coeff_map": coeff_map,
            "metrics": metrics,
            "canonical_equation": canonical,
            "data_npz": str(npz_path),
            "data_csv": str(csv_path),
            "data_meta": str(meta_path),
            "noise_std_rel_phi": (
                float(dataset.metadata.get("noise_std_rel_phi", 0.0))
                if dataset.metadata
                else float(problem.default_noise)
            ),
        }
    )

    out_json = results_dir / f"mond{pid}_result.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MOND PDE benchmark with sparse baseline discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated IDs, e.g. '000,001' or 'mond000,mond001'",
    )
    parser.add_argument("--all", action="store_true", help="Run all MOND problems")
    parser.add_argument("--fast", action="store_true", help="Use a smaller grid")
    parser.add_argument("--skip_generate", action="store_true", help="Reuse existing .npz datasets")
    parser.add_argument("--noise", type=float, default=None, help="Override relative phi noise level")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for noise generation")
    parser.add_argument("--stlsq_lambda", type=float, default=1e-2, help="STLSQ threshold")
    parser.add_argument("--stlsq_max_iter", type=int, default=20, help="Max STLSQ iterations")
    parser.add_argument("--ridge", type=float, default=1e-12, help="Ridge regularization")
    parser.add_argument(
        "--interior_pad",
        type=int,
        default=2,
        help="Drop this many boundary cells for derivatives",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "mond"),
        help="Directory for generated datasets",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(REPO_ROOT / "results" / "mond"),
        help="Directory for benchmark outputs",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_problems = load_problems()
    if args.only:
        raw_ids = [s.strip().lower() for s in args.only.split(",") if s.strip()]
        ids = [rid.replace("mond", "") for rid in raw_ids]
        problems = {pid: all_problems[pid] for pid in ids if pid in all_problems}
        missing = [pid for pid in ids if pid not in all_problems]
        if missing:
            print(f"Warning: unknown problem IDs: {missing}")
    elif args.all:
        problems = all_problems
    else:
        print("Specify --only <ids> or --all")
        return 1

    if not problems:
        print("No problems to run.")
        return 1

    results: list[dict] = []
    for pid in sorted(problems.keys()):
        problem = problems[pid]
        print(f"\n{'=' * 72}")
        print(f"mond{pid}: {problem.description}")
        print(f"  mu_mode={problem.mu_mode}, grid={64 if args.fast else problem.grid_size}")
        print(f"{'=' * 72}")

        res = run_problem(
            problem,
            data_dir=data_dir,
            results_dir=results_dir,
            fast=args.fast,
            skip_generate=args.skip_generate,
            noise_override=args.noise,
            seed=args.seed,
            stlsq_lambda=args.stlsq_lambda,
            stlsq_max_iter=args.stlsq_max_iter,
            ridge=args.ridge,
            interior_pad=args.interior_pad,
            verbose=args.verbose,
        )
        results.append(res)

        marker = {
            "PASS": "OK",
            "PARTIAL": "~~",
            "FAIL": "XX",
            "ERROR": "!!",
        }.get(res["status"], "??")
        print(f"  [{marker}] {res['status']}: {res['message']}")
        if res.get("canonical_equation"):
            print(f"  Discovered: {res['canonical_equation']}")

    print("\n" + "=" * 72)
    print("MOND BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"{'ID':<8} {'Description':<40} {'Status':<8} {'Details'}")
    print("-" * 72)
    for r in results:
        pid = r["id"]
        desc = str(r["description"])[:38]
        status = r["status"]
        msg = str(r["message"]).split(";")[0][:36]
        print(f"mond{pid:<4} {desc:<40} {status:<8} {msg}")
    print("-" * 72)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    parts = [f"{k}: {v}" for k, v in sorted(counts.items())]
    print(f"Total: {len(results)} | {' | '.join(parts)}")
    print("=" * 72)

    summary_path = results_dir / "summary.json"
    summary = {"problems": results, "counts": counts}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")

    if counts.get("FAIL", 0) > 0 or counts.get("ERROR", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
