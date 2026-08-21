#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from kepler_demo_utils import (
    DEFAULT_CADENCE_DAYS,
    DEFAULT_PROFILE,
    DEFAULT_PROVIDER,
    DEFAULT_RAW_MANIFEST_PATH,
    DEFAULT_SOLAR_MU_AU_DAY,
    DEFAULT_START_DATE,
    DEFAULT_YEARS,
    _jsonable,
    evaluate_symbolic_holdout_generalization,
    load_generation_provenance,
    load_kepler_datasets_from_manifest,
)


def _parse_split_csv(raw: str) -> list[str]:
    items = [part.strip() for part in str(raw).split(",")]
    out = [item for item in items if item]
    if not out:
        raise ValueError("at least one split must be selected")
    return out


def _write_report(
    *,
    report_path: Path,
    summary_path: Path,
    train_summary_path: Path,
    train_splits: list[str],
    holdout_splits: list[str],
    summary: dict[str, object],
) -> None:
    holdout = dict(summary["holdout_generalization"])
    aggregate = dict(holdout["aggregate"])
    rows = list(holdout["per_dataset"])
    probe_stability = dict(summary.get("train_probe_stability", {}) or {})
    data_provenance = dict(summary.get("data_provenance", {}) or {})
    train_extraction_note = str(summary.get("train_extraction_note", "not recorded"))

    lines = [
        "# Symbolic Hold-Out Generalization",
        "",
        "This report evaluates a true cross-orbit symbolic generalization test.",
        "",
        f"Training extraction note: {train_extraction_note}",
        "",
        "Training setup:",
        f"- symbolic training splits: `{train_splits}`",
        f"- evaluation splits: `{holdout_splits}`",
        "- Stage A and Stage B are fit only on the training splits.",
        "- On each hold-out orbit, `ell_d` is recovered from the areal-law stage only.",
        "- The radial barrier coefficient is then predicted from the discovered lift relation `k = intercept + slope * ell^2`.",
        "- The hold-out radial law uses the training `mu` and does not refit `k` from `ddot(r)`.",
    ]
    if data_provenance:
        raw_rows = list(data_provenance.get("raw_manifest_rows", []) or [])
        lines.extend(
            [
                "",
                "Data provenance:",
                f"- provider: `{data_provenance.get('provider', 'unknown')}`",
                f"- profile: `{data_provenance.get('profile', 'unknown')}`",
                f"- start date: `{data_provenance.get('start_date', 'unknown')}`",
                f"- years: `{data_provenance.get('years', 'unknown')}`",
                f"- cadence days: `{data_provenance.get('cadence_days', 'unknown')}`",
                f"- raw manifest path: `{data_provenance.get('raw_manifest_path', 'n/a')}`",
            ]
        )
        if raw_rows:
            orbit_ids = ", ".join(str(row.get("orbit_id", "")) for row in raw_rows)
            lines.append(f"- bodies: `{orbit_ids}`")
        if str(data_provenance.get("profile", "")) == "weathered":
            lines.extend(
                [
                    "- note: this is the weathered real-data profile, so the hold-out test probes approximate reduced-Kepler structure on perturbed ephemeris trajectories rather than an exact two-body closure",
                ]
            )
    lines.extend(
        [
        "",
        "Aggregate:",
        f"- training `mu_mean`: `{float(holdout['train_mu_mean']):.9f}`",
        f"- lift intercept: `{float(holdout['lift_intercept']):+.3e}`",
        f"- lift slope: `{float(holdout['lift_slope']):+.6f}`",
        f"- max hold-out `|ell-h|/|h|`: `{float(aggregate['max_ell_rel_error']):.3e}`",
        f"- max hold-out `|k-h^2|`: `{float(aggregate['max_k_abs_error']):.3e}`",
        f"- mean hold-out radial RMSE: `{float(aggregate['mean_radial_rmse']):.3e}`",
        f"- max hold-out radial RMSE: `{float(aggregate['max_radial_rmse']):.3e}`",
        f"- mean oracle radial RMSE if `k` were refit from `ddot(r)`: `{float(aggregate['mean_oracle_radial_rmse_if_refit']):.3e}`",
        f"- max lift penalty vs oracle refit: `{float(aggregate['max_lift_penalty_vs_oracle_refit']):.3e}`",
        ]
    )
    if probe_stability:
        omega = dict(probe_stability.get("omega", {}) or {})
        rddot = dict(probe_stability.get("rddot", {}) or {})
        omega_agg = dict(omega.get("aggregate", {}) or {})
        rddot_agg = dict(rddot.get("aggregate", {}) or {})
        if omega_agg or rddot_agg:
            lines.extend(
                [
                    "",
                    "Probe stability on the training symbolic run:",
                    f"- omega max relative coefficient scatter: `{float(omega_agg.get('max_coeff_rel_std_by_exponent', {}).get('r^-2', 0.0)):.3e}`",
                    f"- rddot max relative coefficient scatter on `r^-3`: `{float(rddot_agg.get('max_coeff_rel_std_by_exponent', {}).get('r^-3', 0.0)):.3e}`",
                    f"- rddot max relative coefficient scatter on `r^-2`: `{float(rddot_agg.get('max_coeff_rel_std_by_exponent', {}).get('r^-2', 0.0)):.3e}`",
                ]
            )

    lines.extend(
        [
            "",
            "Per hold-out orbit:",
            "",
            "| Orbit | ell_fit | ell_true | k_pred | k_true | radial RMSE | oracle radial RMSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['orbit_id']} | {float(row['ell_fit_from_areal_law']):.6f} | "
            f"{float(row['ell_true']):.6f} | {float(row['k_pred_from_lift']):.6f} | "
            f"{float(row['k_true']):.6f} | {float(row['radial_rmse']):.3e} | "
            f"{float(row['oracle_radial_rmse_if_refit']):.3e} |"
        )

    lines.extend(
        [
            "",
            "Artifacts:",
            f"- training symbolic summary: `{train_summary_path}`",
            f"- this report: `{report_path}`",
            f"- machine-readable summary: `{summary_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train symbolic reduced-Kepler laws on real HORIZONS trajectories and evaluate hold-out generalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generate", action="store_true", help="Regenerate the Kepler CSV inputs first")
    parser.add_argument("--fast", action="store_true", help="Use the fast Class-SR profile")
    parser.add_argument("--reuse_existing", action="store_true", help="Reuse existing symbolic artifacts when present")
    parser.add_argument("--mu", type=float, default=float(DEFAULT_SOLAR_MU_AU_DAY), help="Shared gravitational parameter")
    parser.add_argument("--provider", choices=("astropy_builtin", "raw_csv"), default=DEFAULT_PROVIDER, help="Ephemeris source used when --generate is set")
    parser.add_argument("--profile", choices=("clean", "weathered"), default=DEFAULT_PROFILE, help="Ephemeris profile used when --generate is set")
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help="Ephemeris start date used when --generate is set")
    parser.add_argument("--years", type=float, default=float(DEFAULT_YEARS), help="Trajectory span used when --generate is set")
    parser.add_argument("--cadence_days", type=float, default=float(DEFAULT_CADENCE_DAYS), help="Cadence used when --generate is set")
    parser.add_argument("--raw_manifest", type=str, default=str(DEFAULT_RAW_MANIFEST_PATH), help="Raw normalized-state manifest used when --generate with provider=raw_csv")
    parser.add_argument("--seed", type=int, default=123, help="Seed for the orbit ensemble")
    parser.add_argument("--train_samples", type=int, default=1024, help="Samples per training orbit when generating")
    parser.add_argument("--validation_samples", type=int, default=1024, help="Samples per validation orbit when generating")
    parser.add_argument("--holdout_samples", type=int, default=2048, help="Samples per holdout orbit when generating")
    parser.add_argument("--train_splits", type=str, default="train,validation", help="Splits used for symbolic training")
    parser.add_argument("--holdout_splits", type=str, default="holdout", help="Splits used only for evaluation")
    parser.add_argument("--param_metadata_mode", choices=("none", "indices"), default="none")
    parser.add_argument("--class_cv_threshold", type=float, default=0.15)
    parser.add_argument("--ndata_train", type=int, default=None)
    parser.add_argument("--ndata_val", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--class_sr_max_points", type=int, default=None)
    parser.add_argument("--basis_tol", type=float, default=1.0e-4)
    parser.add_argument("--intercept_tol", type=float, default=5.0e-4)
    parser.add_argument("--probe_stability_clouds", type=int, default=8)
    parser.add_argument("--probe_stability_points", type=int, default=9)
    parser.add_argument("--probe_stability_seed", type=int, default=123)
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path("results") / "kepler_ephemeris_real_weathered_holdout"),
        help="Directory where the hold-out evaluation summary will be written",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_splits = _parse_split_csv(args.train_splits)
    holdout_splits = _parse_split_csv(args.holdout_splits)
    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    symbolic_train_dir = results_root / "train_symbolic"

    cmd = [
        sys.executable,
        str(script_dir / "run_class_sr_discovery.py"),
        "--results_dir",
        str(symbolic_train_dir),
        "--include_splits",
        ",".join(train_splits),
        "--param_metadata_mode",
        str(args.param_metadata_mode),
        "--class_cv_threshold",
        str(float(args.class_cv_threshold)),
        "--basis_tol",
        str(float(args.basis_tol)),
        "--intercept_tol",
        str(float(args.intercept_tol)),
        "--probe_stability_clouds",
        str(int(args.probe_stability_clouds)),
        "--probe_stability_points",
        str(int(args.probe_stability_points)),
        "--probe_stability_seed",
        str(int(args.probe_stability_seed)),
        "--mu",
        str(float(args.mu)),
        "--provider",
        str(args.provider),
        "--profile",
        str(args.profile),
        "--start_date",
        str(args.start_date),
        "--years",
        str(float(args.years)),
        "--cadence_days",
        str(float(args.cadence_days)),
        "--seed",
        str(int(args.seed)),
        "--train_samples",
        str(int(args.train_samples)),
        "--validation_samples",
        str(int(args.validation_samples)),
        "--holdout_samples",
        str(int(args.holdout_samples)),
        "--strict_extract",
    ]
    if args.raw_manifest is not None:
        cmd.extend(["--raw_manifest", str(args.raw_manifest)])
    if args.ndata_train is not None:
        cmd.extend(["--ndata_train", str(int(args.ndata_train))])
    if args.ndata_val is not None:
        cmd.extend(["--ndata_val", str(int(args.ndata_val))])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(int(args.batch_size))])
    if args.class_sr_max_points is not None:
        cmd.extend(["--class_sr_max_points", str(int(args.class_sr_max_points))])
    if bool(args.generate):
        cmd.append("--generate")
    if bool(args.fast):
        cmd.append("--fast")
    if bool(args.reuse_existing):
        cmd.append("--reuse_existing")

    rc = subprocess.call(cmd)
    if rc != 0:
        return int(rc)

    train_summary_path = symbolic_train_dir / "symbolic_kepler_summary.json"
    if not train_summary_path.exists():
        raise FileNotFoundError(f"missing symbolic training summary: {train_summary_path}")
    symbolic_train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    if str(symbolic_train_summary.get("status")) != "extractable":
        raise ValueError(
            f"symbolic training summary is not extractable: status={symbolic_train_summary.get('status')!r}"
        )

    datasets = load_kepler_datasets_from_manifest(script_dir / "data")
    data_provenance = load_generation_provenance(script_dir / "data")
    holdout_datasets = [dataset for dataset in datasets if str(dataset.split) in set(holdout_splits)]
    holdout_generalization = evaluate_symbolic_holdout_generalization(
        symbolic_train_summary,
        holdout_datasets=holdout_datasets,
    )

    summary = {
        "status": "ok",
        "train_splits": train_splits,
        "holdout_splits": holdout_splits,
        "symbolic_train_summary_path": str(train_summary_path),
        "data_provenance": data_provenance,
        "train_extraction_note": symbolic_train_summary.get("extraction_note", None),
        "train_probe_stability": symbolic_train_summary.get("probe_stability", None),
        "holdout_generalization": holdout_generalization,
    }
    summary_path = results_root / "holdout_generalization_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    report_path = results_root / "holdout_generalization_report.md"
    _write_report(
        report_path=report_path,
        summary_path=summary_path,
        train_summary_path=train_summary_path,
        train_splits=train_splits,
        holdout_splits=holdout_splits,
        summary=summary,
    )

    aggregate = holdout_generalization["aggregate"]
    print("Symbolic hold-out generalization")
    print(f"train_splits              = {train_splits}")
    print(f"holdout_splits            = {holdout_splits}")
    print(f"train summary             = {train_summary_path}")
    print(f"max |ell-h| / |h|         = {aggregate['max_ell_rel_error']:.3e}")
    print(f"max |k-h^2|               = {aggregate['max_k_abs_error']:.3e}")
    print(f"mean radial rmse          = {aggregate['mean_radial_rmse']:.3e}")
    print(f"max radial rmse           = {aggregate['max_radial_rmse']:.3e}")
    print(f"max lift penalty vs refit = {aggregate['max_lift_penalty_vs_oracle_refit']:.3e}")
    print(f"summary json              = {summary_path}")
    print(f"report                    = {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
