#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kepler_demo_utils import (
    DEFAULT_CADENCE_DAYS,
    DEFAULT_PROFILE,
    DEFAULT_PROVIDER,
    DEFAULT_RAW_MANIFEST_PATH,
    DEFAULT_SOLAR_MU_AU_DAY,
    DEFAULT_START_DATE,
    DEFAULT_YEARS,
    LEVERAGE_ROUND_ROBIN_STRATEGY,
    _jsonable,
    analyze_kepler_reduced_family,
    assign_leverage_round_robin_splits,
    build_default_kepler_datasets,
    write_generated_artifacts,
)


def _find_scan_row(rows, exponent: float) -> dict | None:
    best = None
    best_gap = float("inf")
    for row in rows:
        gap = abs(float(row["exponent"]) - float(exponent))
        if gap < best_gap:
            best_gap = gap
            best = row
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the HORIZONS-backed reduced-Kepler direct-fit discovery scaffold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mu", type=float, default=float(DEFAULT_SOLAR_MU_AU_DAY), help="Shared gravitational parameter in AU^3/day^2")
    parser.add_argument("--provider", choices=("astropy_builtin", "raw_csv"), default=DEFAULT_PROVIDER, help="Ephemeris source")
    parser.add_argument("--profile", choices=("clean", "weathered"), default=DEFAULT_PROFILE, help="Exact two-body propagation from real initial conditions or the raw ephemeris trajectory")
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help="Ephemeris start date")
    parser.add_argument("--years", type=float, default=float(DEFAULT_YEARS), help="Trajectory span in years")
    parser.add_argument("--cadence_days", type=float, default=float(DEFAULT_CADENCE_DAYS), help="Time step in days")
    parser.add_argument("--raw_manifest", type=str, default=str(DEFAULT_RAW_MANIFEST_PATH), help="JSON manifest for normalized external heliocentric state CSVs")
    parser.add_argument("--seed", type=int, default=123, help="Legacy no-op kept for interface compatibility")
    parser.add_argument("--train_samples", type=int, default=1024, help="Legacy no-op kept for interface compatibility")
    parser.add_argument("--validation_samples", type=int, default=1024, help="Legacy no-op kept for interface compatibility")
    parser.add_argument("--holdout_samples", type=int, default=2048, help="Legacy no-op kept for interface compatibility")
    parser.add_argument(
        "--accel_source",
        choices=("gradient", "surrogate"),
        default="gradient",
        help="Weathered-profile accelerations: second-order finite differences of the "
        "ephemeris velocities, or analytic first derivatives of blind-chart NestyNet "
        "surrogates fitted to the velocity channels",
    )
    parser.add_argument(
        "--accel_certificate",
        action="store_true",
        help="With --accel_source surrogate: also fit the position channels and score "
        "their analytic derivatives against the velocity data (the measured "
        "derivative-gap certificate for the unsupervised acceleration channel)",
    )
    parser.add_argument(
        "--accel_harmonic",
        action="store_true",
        help="With --accel_source surrogate: add a second circle at twice the refined "
        "frequency (the eccentricity harmonic) to each cylinder chart",
    )
    parser.add_argument(
        "--split_strategy",
        choices=("manifest", "leverage_round_robin"),
        default="manifest",
        help="Body splits: as declared in the manifest, or the deterministic "
        "radial-leverage round-robin (sorted by dynamic range, i%%10==0 holdout, "
        "i%%10==1 validation) used for candidate-pool manifests like the 308-body "
        "ensemble",
    )
    parser.add_argument(
        "--accel_cache_dir",
        type=str,
        default=None,
        help="With --accel_source surrogate: content-addressed per-body cache of the "
        "surrogate accelerations (see precompute_kepler_surrogate_accels.py); cached "
        "bodies load instantly, missing ones are fitted and stored",
    )
    parser.add_argument("--generate", action="store_true", help="Also regenerate the CSV artifacts under data/")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory for the summary JSON",
    )
    parser.add_argument("--stage_a_tol", type=float, default=None, help="Tolerance for the areal-law fit")
    parser.add_argument("--stage_b_tol", type=float, default=None, help="Tolerance for the radial-law fit")
    parser.add_argument("--lift_tol", type=float, default=None, help="Tolerance for the coefficient lift")
    parser.add_argument("--energy_tol", type=float, default=None, help="Tolerance for the energy post-pass")
    parser.add_argument("--enforce", action="store_true", help="Return nonzero on tolerance failures")
    parser.add_argument(
        "--power_exp_min",
        type=float,
        default=1.6,
        help="Minimum inverse-power exponent in the train/holdout scan",
    )
    parser.add_argument(
        "--power_exp_max",
        type=float,
        default=2.4,
        help="Maximum inverse-power exponent in the train/holdout scan",
    )
    parser.add_argument(
        "--power_exp_count",
        type=int,
        default=81,
        help="Number of inverse-power exponents in the train/holdout scan",
    )
    args = parser.parse_args()

    datasets = build_default_kepler_datasets(
        mu=float(args.mu),
        seed=int(args.seed),
        train_samples=int(args.train_samples),
        validation_samples=int(args.validation_samples),
        holdout_samples=int(args.holdout_samples),
        provider=str(args.provider),
        profile=str(args.profile),
        start_date=str(args.start_date),
        years=float(args.years),
        cadence_days=float(args.cadence_days),
        raw_manifest=args.raw_manifest,
        accel_source=str(args.accel_source),
        accel_certificate=bool(args.accel_certificate),
        accel_harmonic=bool(args.accel_harmonic),
        accel_cache_dir=args.accel_cache_dir,
    )
    if str(args.split_strategy) == "leverage_round_robin":
        datasets = assign_leverage_round_robin_splits(datasets)
    if args.generate:
        write_generated_artifacts(Path(__file__).resolve().parent, datasets)

    if str(args.accel_source) == "surrogate":
        print("\nSurrogate accelerations (cylinder-chart surrogate per velocity channel)")
        for ds in datasets:
            prov = ds.accel_provenance or {}
            channels = prov.get("channels", {})
            fd_diff = prov.get("fd_rel_diff", {})
            for channel in ("vx", "vy"):
                diag = channels.get(channel, {})
                print(
                    f"{ds.orbit_id}.{channel}: val relRMSE {diag.get('val_rel_rmse', float('nan')):.3e} "
                    f"omegas {['%.6e' % om for om in diag.get('omegas', [])]} "
                    f"| accel-vs-FD rel diff {fd_diff.get('a' + channel[1:], float('nan')):.3e}"
                )
            certificate = prov.get("certificate", None)
            if certificate is not None:
                measured = certificate["measured_derivative_rel_rmse"]
                ratio = certificate["gap_ratio"]
                print(
                    f"{ds.orbit_id}: certificate d1-vs-data relRMSE "
                    f"x:{measured['x']:.3e} (gap ratio {ratio['x']:.1f}) "
                    f"y:{measured['y']:.3e} (gap ratio {ratio['y']:.1f})"
                )

    exponents = np.linspace(
        float(args.power_exp_min),
        float(args.power_exp_max),
        int(args.power_exp_count),
        dtype=np.float64,
    )
    summary = analyze_kepler_reduced_family(datasets, power_exponents=exponents)
    if str(args.split_strategy) == "leverage_round_robin":
        summary["split_strategy"] = LEVERAGE_ROUND_ROBIN_STRATEGY
        for split_name in ("train", "validation", "holdout"):
            summary[f"n_{split_name}"] = sum(1 for ds in datasets if ds.split == split_name)
    summary["acceleration_provenance"] = {
        "accel_source": str(args.accel_source),
        "orbits": {
            ds.orbit_id: ds.accel_provenance
            for ds in datasets
            if ds.accel_provenance is not None
        },
    }

    if args.results_dir is None:
        args.results_dir = str(Path("results") / f"kepler_ephemeris_real_{args.profile}")
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"kepler_ephemeris_{args.profile}_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print("\nOrbit ensemble")
    for row in summary["orbit_registry"]:
        print(
            f"{row['orbit_id']}: split={row['split']:<10s} "
            f"a={row['a']:.4f} e={row['e']:.4f} h={row['h']:.6f} "
            f"E={row['energy']:.6f} Lambda={row['dynamic_range']:.3f}"
        )

    stage_a = summary["stage_a"]
    stage_b = summary["stage_b_all"]
    lift = summary["coefficient_lift"]
    energy = summary["energy"]
    hamiltonian = summary["hamiltonian"]
    power_scan = summary["power_scan"]
    exact_row = power_scan["exact_exponent_row"]
    p18_row = _find_scan_row(power_scan["rows"], 1.8)
    p22_row = _find_scan_row(power_scan["rows"], 2.2)

    print("\nStage A: areal law")
    print(f"max |ell - h| / |h| = {stage_a['max_rel_error']:.3e}")

    print("\nStage B: radial family")
    print(f"mu_true  = {summary['mu_true']:.9f}")
    print(f"mu_fit   = {stage_b['mu']:.9f}")
    print(f"max |k-h^2| = {stage_b['max_k_abs_error']:.3e}")
    print(f"mean rmse   = {stage_b['mean_rmse']:.3e}")

    print("\nInverse-power scan")
    print(f"best train exponent   = {power_scan['best_train_exponent']:.3f}")
    print(f"best holdout exponent = {power_scan['best_holdout_exponent']:.3f}")
    print(f"holdout margin to #2  = {power_scan['holdout_margin_to_second']:.3e}")
    print(
        f"p=2 holdout mean rmse = {exact_row['holdout_mean_rmse']:.3e}"
    )
    if p18_row is not None:
        print(f"p=1.8 holdout mean rmse = {p18_row['holdout_mean_rmse']:.3e}")
    if p22_row is not None:
        print(f"p=2.2 holdout mean rmse = {p22_row['holdout_mean_rmse']:.3e}")

    print("\nCoefficient lift")
    print("k ~= intercept + slope*h^2")
    print(f"intercept = {lift['intercept']:+.6e}")
    print(f"slope     = {lift['slope']:+.6e}")
    print(f"sq-rmse    = {lift['quadratic_rmse']:.3e}")
    print(f"lin-rmse   = {lift['linear_rmse']:.3e}")

    print("\nEnergy post-pass")
    print(f"coeffs = {energy['coeffs']}")
    print(f"coeff max abs error  = {energy['coeff_max_abs_error']:.3e}")
    print(f"max centered residual = {energy['max_centered_residual']:.3e}")

    print("\nHamiltonian assembly")
    print(f"H_reduced = {hamiltonian['recovered_formulas']['natural_reduced_plain']}")
    print(
        "flow max rmse = "
        f"theta:{hamiltonian['consistency']['max_theta_rmse']:.3e}, "
        f"radial:{hamiltonian['consistency']['max_radial_rmse']:.3e}"
    )
    print(
        "max energy abs error = "
        f"reduced:{hamiltonian['consistency']['max_natural_reduced_energy_abs_error']:.3e}, "
        f"cartesian:{hamiltonian['consistency']['max_natural_cartesian_energy_abs_error']:.3e}"
    )
    print(f"summary json = {summary_path}")

    if args.stage_a_tol is None:
        args.stage_a_tol = 1.0e-10 if str(args.profile) == "clean" else 5.0e-2
    if args.stage_b_tol is None:
        args.stage_b_tol = 1.0e-10 if str(args.profile) == "clean" else 5.0e-3
    if args.lift_tol is None:
        args.lift_tol = 1.0e-10 if str(args.profile) == "clean" else 5.0e-3
    if args.energy_tol is None:
        args.energy_tol = 1.0e-10 if str(args.profile) == "clean" else 5.0e-2

    failures = []
    if float(stage_a["max_rel_error"]) > float(args.stage_a_tol):
        failures.append("areal-law fit exceeded tolerance")
    if float(stage_b["mu_abs_error"]) > float(args.stage_b_tol):
        failures.append("shared-mu radial fit exceeded tolerance")
    if float(lift["max_abs_residual"]) > float(args.lift_tol):
        failures.append("coefficient lift exceeded tolerance")
    if float(energy["coeff_max_abs_error"]) > float(args.energy_tol):
        failures.append("energy post-pass exceeded tolerance")
    if abs(float(power_scan["best_holdout_exponent"]) - 2.0) > 0.051:
        failures.append("holdout scan did not select the inverse-square exponent")
    if failures and bool(args.enforce):
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    if failures:
        print("\nNOTE: the real-trajectory scaffold completed, but the weathered profile missed some strict Kepler checks")
        for failure in failures:
            print(f"- {failure}")
        return 0

    if str(args.profile) == "weathered":
        print("\nPASS: the weathered HORIZONS scaffold recovers the staged family approximately")
    else:
        print("\nPASS: the cleaned two-body control recovers the staged family exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
