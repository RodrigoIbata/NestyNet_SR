#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from kepler_demo_utils import (
    DEFAULT_CADENCE_DAYS,
    DEFAULT_PROFILE,
    DEFAULT_PROVIDER,
    DEFAULT_RAW_MANIFEST_PATH,
    DEFAULT_SOLAR_MU_AU_DAY,
    DEFAULT_START_DATE,
    DEFAULT_YEARS,
    _jsonable,
    analyze_classsr_probe_stability,
    analyze_symbolic_kepler_family,
    extract_classsr_inverse_power_rows,
    infer_effective_n_samples,
    load_classsr_payload,
    load_generation_provenance,
    load_kepler_datasets_from_manifest,
    load_stageb_payload,
    merge_symbolic_kepler_tables,
    resolve_run_dimensions,
    suggest_probe_points_from_r_values,
    suggest_symbolic_readout_points_from_r_values,
    target_filepaths,
)

_DEFAULT_TRAIN_SAMPLES = 1024
_DEFAULT_VALIDATION_SAMPLES = 1024
_DEFAULT_HOLDOUT_SAMPLES = 2048
_FAST_TRAIN_SAMPLES = 128
_FAST_VALIDATION_SAMPLES = 128
_FAST_HOLDOUT_SAMPLES = 256
_FAST_TRAIN = 64
_FAST_VAL = 32
_FAST_BATCH = 32
_FAST_CLASS_POINTS = 64


def _parse_split_csv(raw: str) -> list[str]:
    items = [part.strip() for part in str(raw).split(",")]
    out = [item for item in items if item]
    if not out:
        raise ValueError("at least one split must be selected")
    return out


def _run_and_stream(cmd: list[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="")
        process.wait()
    return int(process.returncode)


def _generate_if_requested(args, script_dir: Path) -> None:
    if not bool(args.generate):
        return
    cmd = [
        sys.executable,
        str(script_dir / "generate_kepler_data.py"),
        "--mu",
        str(float(args.mu)),
        "--seed",
        str(int(args.seed)),
        "--train_samples",
        str(int(args.train_samples)),
        "--validation_samples",
        str(int(args.validation_samples)),
        "--holdout_samples",
        str(int(args.holdout_samples)),
        "--output_root",
        str(script_dir),
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
    ]
    if args.raw_manifest is not None:
        cmd.extend(["--raw_manifest", str(args.raw_manifest)])
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _build_run_sr_cmd(
    *,
    target_name: str,
    filepaths: list[Path],
    metadata_json: str | None,
    fast: bool,
    class_cv_threshold: float,
    ndata_train: int,
    ndata_val: int,
    batch_size: int,
    class_sr_max_points: int,
) -> list[str]:
    target = str(target_name)
    if target == "omega":
        y_units = "[0,-1]"
    elif target == "rddot":
        y_units = "[1,-2]"
    else:
        raise ValueError(f"unsupported target {target_name!r}")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nestynet_sr.run_SR",
        "--filepaths",
        *[str(path) for path in filepaths],
        "--class_sr",
        "--class_auto_include_scales",
        "--class_cv_threshold",
        str(float(class_cv_threshold)),
        "--ndata_train",
        str(int(ndata_train)),
        "--ndata_val",
        str(int(ndata_val)),
        "--batch_size",
        str(int(batch_size)),
        "--class_sr_max_points",
        str(int(class_sr_max_points)),
        "--force_y_ops",
        "identity",
        "--no_ysearch",
        "--no_stageA_separabilities",
        "--factorized-search",
        "--y_units",
        y_units,
        "--x_units",
        "[[1,0]]",
        "--units_basis",
        "L,T",
        "--log_level",
        "INFO",
    ]
    if metadata_json is not None:
        cmd.extend(
            [
                "--class_param_sr_metadata",
                str(metadata_json),
            ]
        )
    if bool(fast):
        cmd.extend(
            [
                "--fast",
                "--single_layer",
                "--no-factorized-search",
                "--no-refine-skeleton",
                "--no_class_param_sr",
                "--disable_compound_detection",
                "--max_ab_iters",
                "1",
                "--stageB_epochs",
                "120",
                "--stageB_max_outer_iters",
                "4",
            ]
        )
    return cmd


def _derive_result_base_filename(filepaths: list[Path]) -> str:
    if len(filepaths) == 1:
        return Path(filepaths[0]).stem
    stems = [Path(path).stem for path in filepaths]
    common = os.path.commonprefix(stems).rstrip("_-.")
    if common and len(common) >= 3:
        return f"{common}_multi{len(stems)}"
    return f"multi{len(stems)}_{stems[0]}"


def _result_paths(results_dir: Path, filepaths: list[Path]) -> tuple[Path, Path]:
    stem = _derive_result_base_filename(filepaths)
    return (
        results_dir / f"{stem}_classSR.json",
        results_dir / f"{stem}_stageB.pkl",
    )


def _copy_run_sr_artifacts(
    *,
    filepaths: list[Path],
    repo_results_dir: Path,
    repo_models_dir: Path,
    target_dir: Path,
) -> None:
    stem = _derive_result_base_filename(filepaths)
    target_dir.mkdir(parents=True, exist_ok=True)
    suffixes = (
        ".pkl",
        ".human",
        ".report.json",
        ".state.pkl",
        "_stageB.pkl",
        "_stageB_model.pt",
        "_final.human",
        "_classSR.json",
        "_classSR.human",
        ".decisions.json",
    )
    for suffix in suffixes:
        src = repo_results_dir / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, target_dir / src.name)
    for suffix in (".identity.mod", ".mod"):
        src = repo_models_dir / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, target_dir / src.name)


def _extract_table_or_diagnostic(
    *,
    target_name: str,
    target_dir: Path,
    filepaths: list[Path],
    exponents: list[float],
    probe_points_by_dataset: dict[str, np.ndarray],
    sample_points_by_dataset: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    class_sr_json_path, stageb_pkl_path = _result_paths(target_dir, filepaths)
    base_filename = _derive_result_base_filename(filepaths)
    diagnostic: dict[str, object] = {
        "target": str(target_name),
        "base_filename": str(base_filename),
        "class_sr_json_path": str(class_sr_json_path),
        "stageb_pkl_path": str(stageb_pkl_path),
        "artifacts_present": {
            "class_sr_json": bool(class_sr_json_path.exists()),
            "stageb_pkl": bool(stageb_pkl_path.exists()),
        },
    }
    if not class_sr_json_path.exists() or not stageb_pkl_path.exists():
        diagnostic["status"] = "missing_artifacts"
        return diagnostic

    try:
        table = extract_classsr_inverse_power_rows(
            class_sr_json_path=class_sr_json_path,
            stageb_pkl_path=stageb_pkl_path,
            exponents=exponents,
            dataset_probe_points=probe_points_by_dataset,
            dataset_sample_points=sample_points_by_dataset,
        )
    except Exception as exc:
        diagnostic["status"] = "opaque_or_non_inverse_power"
        diagnostic["error"] = f"{type(exc).__name__}: {exc}"
        try:
            class_payload = load_classsr_payload(class_sr_json_path)
            diagnostic["class_tags"] = list(class_payload.get("class_tags", []) or [])
            diagnostic["experiment_tags"] = list(class_payload.get("experiment_tags", []) or [])
            diagnostic["cv_per_tag"] = dict(class_payload.get("cv_per_tag", {}) or {})
            diagnostic["derived_invariants"] = list(class_payload.get("derived_invariants", []) or [])
        except Exception as payload_exc:
            diagnostic["class_payload_error"] = f"{type(payload_exc).__name__}: {payload_exc}"
        try:
            stageb_payload = load_stageb_payload(stageb_pkl_path)
            root = stageb_payload.get("stageB_ast", None)
            diagnostic["dataset_ids"] = list(stageb_payload.get("stageB_dataset_ids", []) or [])
            diagnostic["phi_expr_str"] = stageb_payload.get("phi_expr_str", None)
            diagnostic["phi_expr_strs"] = stageb_payload.get("phi_expr_strs", None)
            diagnostic["y_expr_str"] = stageb_payload.get("y_expr_str", None)
            diagnostic["stageb_root_type"] = None if root is None else type(root).__name__
            diagnostic["stageb_root_repr"] = None if root is None else str(root)
        except Exception as payload_exc:
            diagnostic["stageb_payload_error"] = f"{type(payload_exc).__name__}: {payload_exc}"
        return diagnostic

    return {
        "status": "extractable",
        "target": str(target_name),
        "base_filename": str(base_filename),
        "table": table,
    }


def _write_hamiltonian_report(
    *,
    report_path: Path,
    summary_path: Path,
    hamiltonian: dict[str, object],
    data_provenance: dict[str, object] | None,
    extraction_note: str,
) -> None:
    assumptions = dict(hamiltonian["assumptions"])
    canonical = dict(assumptions["canonical_identification"])
    formulas = dict(hamiltonian["recovered_formulas"])
    consistency = dict(hamiltonian["consistency"])
    per_dataset = list(hamiltonian["per_dataset"])

    lines = [
        "# Kepler Hamiltonian Assembly",
        "",
        "This report assembles the natural Kepler Hamiltonian from the recovered reduced flow, coefficient lift, and energy post-pass.",
        "",
        f"Extraction note: {extraction_note}",
        "",
        "## Assumptions",
        "",
        f"- Unit test mass: `{assumptions['unit_test_mass']}`",
        f"- Euclidean configuration space: `{assumptions['euclidean_configuration_space']}`",
        f"- Canonical identification: `p_r = {canonical['p_r']}`, `p_theta = {canonical['p_theta']}`",
        f"- Shared recovered `mu`: `{float(hamiltonian['shared_mu']):.12g}`",
    ]
    if data_provenance:
        raw_rows = list(data_provenance.get("raw_manifest_rows", []) or [])
        split_counts: dict[str, int] = {}
        for row in raw_rows:
            split = str((row or {}).get("split", "") or "")
            if split:
                split_counts[split] = int(split_counts.get(split, 0)) + 1
        lines.extend(
            [
                "",
                "## Data Provenance",
                "",
                f"- Provider: `{data_provenance.get('provider', 'unknown')}`",
                f"- Profile: `{data_provenance.get('profile', 'unknown')}`",
                f"- Start date: `{data_provenance.get('start_date', 'unknown')}`",
                f"- Years: `{data_provenance.get('years', 'unknown')}`",
                f"- Cadence days: `{data_provenance.get('cadence_days', 'unknown')}`",
                f"- Raw manifest path: `{data_provenance.get('raw_manifest_path', 'n/a')}`",
            ]
        )
        if raw_rows:
            orbit_ids = ", ".join(str(row.get("orbit_id", "")) for row in raw_rows)
            lines.append(f"- Bodies: `{orbit_ids}`")
        if split_counts:
            counts_text = ", ".join(f"{key}:{value}" for key, value in sorted(split_counts.items()))
            lines.append(f"- Split counts: `{counts_text}`")
        if str(data_provenance.get("profile", "")) == "weathered":
            lines.extend(
                [
                    "",
                    "## Weathered-Profile Note",
                    "",
                    "The default real-data profile uses the HORIZONS trajectories themselves rather than a cleaned two-body re-propagation.",
                    "The recovered Kepler family should therefore be interpreted as an approximate reduced structure in the presence of real multi-body perturbations.",
                ]
            )

    lines.extend(
        [
            "",
            "## Recovered Forms",
            "",
            f"- Recovered reduced form: `{formulas['reduced_plain']}`",
            f"- Natural reduced form: `{formulas['natural_reduced_plain']}`",
            f"- Cartesian form: `{formulas['cartesian_plain']}`",
            "",
            "## Consistency",
            "",
            f"- Lift slope: `{float(consistency['lift_slope']):.9g}`",
            f"- Lift intercept: `{float(consistency['lift_intercept']):+.9g}`",
            f"- Energy coefficient max abs error: `{float(consistency['energy_coeff_max_abs_error']):.3e}`",
            f"- Max theta-flow RMSE under natural Hamiltonian: `{float(consistency['max_theta_rmse']):.3e}`",
            f"- Max radial-flow RMSE under natural Hamiltonian: `{float(consistency['max_radial_rmse']):.3e}`",
            f"- Max |p_theta(state) - ell|: `{float(consistency['max_p_theta_state_abs_error']):.3e}`",
            f"- Max reduced-energy abs error: `{float(consistency['max_natural_reduced_energy_abs_error']):.3e}`",
            f"- Max Cartesian-energy abs error: `{float(consistency['max_natural_cartesian_energy_abs_error']):.3e}`",
            f"- Max reduced-vs-Cartesian energy gap: `{float(consistency['max_reduced_vs_cartesian_energy_gap']):.3e}`",
            "",
            "## Per-Orbit Energy Levels",
            "",
            "| Orbit | Split | p_theta | True E | Reduced H | Cartesian H | |Reduced-True| |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_dataset:
        lines.append(
            "| "
            f"{row['orbit_id']} | {row['split']} | {float(row['p_theta_fit']):.6f} | "
            f"{float(row['true_energy']):.6f} | {float(row['natural_reduced_energy']):.6f} | "
            f"{float(row['natural_cartesian_energy']):.6f} | "
            f"{float(row['natural_reduced_energy_abs_error']):.3e} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Symbolic summary JSON: `{summary_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run symbolic Class-SR discovery for the HORIZONS-backed reduced-Kepler analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generate", action="store_true", help="Regenerate the Kepler CSV inputs first")
    parser.add_argument("--mu", type=float, default=float(DEFAULT_SOLAR_MU_AU_DAY), help="Shared gravitational parameter")
    parser.add_argument("--provider", choices=("astropy_builtin", "raw_csv"), default=DEFAULT_PROVIDER, help="Ephemeris source used when --generate is set")
    parser.add_argument("--profile", choices=("clean", "weathered"), default=DEFAULT_PROFILE, help="Ephemeris profile used when --generate is set")
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help="Ephemeris start date used when --generate is set")
    parser.add_argument("--years", type=float, default=float(DEFAULT_YEARS), help="Trajectory span used when --generate is set")
    parser.add_argument("--cadence_days", type=float, default=float(DEFAULT_CADENCE_DAYS), help="Cadence used when --generate is set")
    parser.add_argument("--raw_manifest", type=str, default=str(DEFAULT_RAW_MANIFEST_PATH), help="Raw normalized-state manifest used when --generate with provider=raw_csv")
    parser.add_argument("--seed", type=int, default=123, help="Seed for the default orbit ensemble")
    parser.add_argument("--train_samples", type=int, default=_DEFAULT_TRAIN_SAMPLES, help="Samples per training orbit when generating")
    parser.add_argument("--validation_samples", type=int, default=_DEFAULT_VALIDATION_SAMPLES, help="Samples per validation orbit when generating")
    parser.add_argument("--holdout_samples", type=int, default=_DEFAULT_HOLDOUT_SAMPLES, help="Samples per holdout orbit when generating")
    parser.add_argument("--fast", action="store_true", help="Pass --fast to run_SR.py and use compact data defaults")
    parser.add_argument(
        "--include_splits",
        type=str,
        default="train,validation,holdout",
        help="Comma-separated orbit splits to include in symbolic discovery",
    )
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Skip new run_SR.py launches and reuse existing Class-SR artifacts already present on disk",
    )
    parser.add_argument("--class_cv_threshold", type=float, default=0.15, help="Class-SR CV threshold")
    parser.add_argument(
        "--param_metadata_mode",
        choices=("none", "indices"),
        default="none",
        help="Whether to pass per-orbit metadata into Class-SR joint fitting",
    )
    parser.add_argument("--ndata_train", type=int, default=None, help="Override run_SR train sample count")
    parser.add_argument("--ndata_val", type=int, default=None, help="Override run_SR validation sample count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override run_SR batch size")
    parser.add_argument(
        "--class_sr_max_points",
        type=int,
        default=None,
        help="Override per-dataset point cap used inside Class-SR joint fitting",
    )
    parser.add_argument(
        "--basis_tol",
        type=float,
        default=1.0e-4,
        help="Max allowed inverse-power probe RMSE for extracted symbolic models",
    )
    parser.add_argument(
        "--intercept_tol",
        type=float,
        default=5.0e-4,
        help="Max allowed fitted intercept magnitude in extracted inverse-power models",
    )
    parser.add_argument("--probe_stability_clouds", type=int, default=8, help="Number of probe clouds per dataset")
    parser.add_argument("--probe_stability_points", type=int, default=9, help="Points per probe cloud")
    parser.add_argument("--probe_stability_seed", type=int, default=123, help="Base seed for probe-cloud generation")
    parser.add_argument(
        "--strict_extract",
        action="store_true",
        help="Return a nonzero exit code if the symbolic run finishes but does not expose extractable inverse-power maps",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path("results") / "kepler_ephemeris_real_weathered_classsr"),
        help="Root output directory for symbolic SR runs",
    )
    args = parser.parse_args()

    if bool(args.fast):
        if int(args.train_samples) == _DEFAULT_TRAIN_SAMPLES:
            args.train_samples = _FAST_TRAIN_SAMPLES
        if int(args.validation_samples) == _DEFAULT_VALIDATION_SAMPLES:
            args.validation_samples = _FAST_VALIDATION_SAMPLES
        if int(args.holdout_samples) == _DEFAULT_HOLDOUT_SAMPLES:
            args.holdout_samples = _FAST_HOLDOUT_SAMPLES
        if args.ndata_train is None:
            args.ndata_train = _FAST_TRAIN
        if args.ndata_val is None:
            args.ndata_val = _FAST_VAL
        if args.batch_size is None:
            args.batch_size = _FAST_BATCH
        if args.class_sr_max_points is None:
            args.class_sr_max_points = _FAST_CLASS_POINTS
        print(
            "[Kepler fast profile] using compact symbolic smoke settings: "
            f"train_samples={args.train_samples}, validation_samples={args.validation_samples}, "
            f"holdout_samples={args.holdout_samples}, ndata_train={args.ndata_train}, "
            f"ndata_val={args.ndata_val}, batch_size={args.batch_size}, "
            f"class_sr_max_points={args.class_sr_max_points}, single_layer, "
            "no_stageA_separabilities, no_factorized_search, no_factorized_search_plus, "
            "no_class_param_sr, disable_compound_detection"
        )

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    _generate_if_requested(args, script_dir)
    include_splits = _parse_split_csv(args.include_splits)

    metadata_path = data_dir / "param_sr_metadata_rows.json"
    manifest_path = data_dir / "manifest.json"
    if not metadata_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"missing generated data under {data_dir}. Run with --generate first."
        )
    omega_files = target_filepaths(data_dir, "omega", splits=include_splits)
    rddot_files = target_filepaths(data_dir, "rddot", splits=include_splits)
    datasets = [
        dataset
        for dataset in load_kepler_datasets_from_manifest(data_dir)
        if str(dataset.split) in set(include_splits)
    ]
    data_provenance = load_generation_provenance(data_dir)
    datasets_by_id = {dataset.orbit_id: dataset for dataset in datasets}
    selected_ids = [dataset.orbit_id for dataset in datasets]
    metadata_json = None
    if str(args.param_metadata_mode) == "indices":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata_rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        filtered_rows = [
            meta_row
            for row, meta_row in zip(list(manifest.get("orbits", []) or []), metadata_rows)
            if str(row.get("orbit_id", "")) in set(selected_ids)
        ]
        metadata_json = json.dumps(filtered_rows)
        print("[Kepler metadata] passing innocuous orbit-index metadata into Class-SR")
    else:
        print("[Kepler metadata] not passing per-orbit metadata into Class-SR")

    actual_n_samples = infer_effective_n_samples(omega_files, rddot_files)
    results_root = Path(args.results_dir).resolve()
    omega_out = results_root / "omega"
    rddot_out = results_root / "rddot"
    repo_results_dir = Path("results").resolve()
    repo_models_dir = Path("models").resolve()
    n_train, n_val, batch_size, class_sr_max_points = resolve_run_dimensions(
        actual_n_samples=actual_n_samples,
        ndata_train=args.ndata_train,
        ndata_val=args.ndata_val,
        batch_size=args.batch_size,
        class_sr_max_points=args.class_sr_max_points,
    )
    print(
        "[Kepler split] "
        f"splits={include_splits}, datasets={len(omega_files)}, n_samples_min={actual_n_samples}, "
        f"ndata_train={n_train}, ndata_val={n_val}, "
        f"batch_size={batch_size}, class_sr_max_points={class_sr_max_points}"
    )

    if not bool(args.reuse_existing):
        print("\n=== Class-SR symbolic discovery: omega target ===")
        rc = _run_and_stream(
            _build_run_sr_cmd(
                target_name="omega",
                filepaths=omega_files,
                metadata_json=metadata_json,
                fast=bool(args.fast),
                class_cv_threshold=float(args.class_cv_threshold),
                ndata_train=n_train,
                ndata_val=n_val,
                batch_size=batch_size,
                class_sr_max_points=class_sr_max_points,
            ),
            log_path=omega_out / "run.log",
        )
        if rc != 0:
            print(f"omega Class-SR run failed with exit code {rc}")
            return rc
    else:
        print("\n=== Reusing existing Class-SR artifacts: omega target ===")
    _copy_run_sr_artifacts(
        filepaths=omega_files,
        repo_results_dir=repo_results_dir,
        repo_models_dir=repo_models_dir,
        target_dir=omega_out,
    )

    if not bool(args.reuse_existing):
        print("\n=== Class-SR symbolic discovery: rddot target ===")
        rc = _run_and_stream(
            _build_run_sr_cmd(
                target_name="rddot",
                filepaths=rddot_files,
                metadata_json=metadata_json,
                fast=bool(args.fast),
                class_cv_threshold=float(args.class_cv_threshold),
                ndata_train=n_train,
                ndata_val=n_val,
                batch_size=batch_size,
                class_sr_max_points=class_sr_max_points,
            ),
            log_path=rddot_out / "run.log",
        )
        if rc != 0:
            print(f"rddot Class-SR run failed with exit code {rc}")
            return rc
    else:
        print("\n=== Reusing existing Class-SR artifacts: rddot target ===")
    _copy_run_sr_artifacts(
        filepaths=rddot_files,
        repo_results_dir=repo_results_dir,
        repo_models_dir=repo_models_dir,
        target_dir=rddot_out,
    )

    probe_points_by_dataset = {
        dataset.orbit_id: suggest_probe_points_from_r_values(dataset.r, n_points=9)
        for dataset in datasets
    }
    sample_points_by_dataset = {
        dataset.orbit_id: suggest_symbolic_readout_points_from_r_values(dataset.r, n_points=129)
        for dataset in datasets
    }
    omega_result = _extract_table_or_diagnostic(
        target_name="omega",
        target_dir=omega_out,
        filepaths=omega_files,
        exponents=[2.0],
        probe_points_by_dataset=probe_points_by_dataset,
        sample_points_by_dataset=sample_points_by_dataset,
    )
    rddot_result = _extract_table_or_diagnostic(
        target_name="rddot",
        target_dir=rddot_out,
        filepaths=rddot_files,
        exponents=[3.0, 2.0],
        probe_points_by_dataset=probe_points_by_dataset,
        sample_points_by_dataset=sample_points_by_dataset,
    )

    summary_path = results_root / "symbolic_kepler_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        str(omega_result.get("status")) != "extractable"
        or str(rddot_result.get("status")) != "extractable"
    ):
        extraction_note = (
            "Inverse-power extraction from the saved Class-SR artifacts, evaluated on the observed orbit radii, "
            "did not succeed on at least one target."
        )
        summary = {
            "status": "opaque_stageb_models",
            "include_splits": include_splits,
            "data_provenance": data_provenance,
            "extraction_note": extraction_note,
            "omega": omega_result,
            "rddot": rddot_result,
        }
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        print("\nSymbolic extraction summary")
        print("status                    = opaque_stageb_models")
        print(f"summary json              = {summary_path}")
        for target_name, payload in (("omega", omega_result), ("rddot", rddot_result)):
            print(
                f"{target_name:25s}= {payload.get('status')} "
                f"({payload.get('error', payload.get('stageb_root_repr', 'no detail'))})"
            )
        if bool(args.strict_extract):
            print("\nFAIL: symbolic runs completed but Stage B remained opaque to inverse-power extraction")
            return 1
        print(
            "\nNOTE: symbolic runs completed, but the current Stage B/Class-SR outputs "
            "did not expose extractable inverse-power forms yet."
        )
        return 0

    omega_table = dict(omega_result["table"])
    rddot_table = dict(rddot_result["table"])
    omega_probe_stability = analyze_classsr_probe_stability(
        class_sr_json_path=_result_paths(omega_out, omega_files)[0],
        stageb_pkl_path=_result_paths(omega_out, omega_files)[1],
        datasets=datasets,
        exponents=[2.0],
        n_clouds=int(args.probe_stability_clouds),
        n_points=int(args.probe_stability_points),
        seed=int(args.probe_stability_seed) + 17,
    )
    rddot_probe_stability = analyze_classsr_probe_stability(
        class_sr_json_path=_result_paths(rddot_out, rddot_files)[0],
        stageb_pkl_path=_result_paths(rddot_out, rddot_files)[1],
        datasets=datasets,
        exponents=[3.0, 2.0],
        n_clouds=int(args.probe_stability_clouds),
        n_points=int(args.probe_stability_points),
        seed=int(args.probe_stability_seed) + 29,
    )
    merged_rows = merge_symbolic_kepler_tables(
        omega_table,
        rddot_table,
        datasets_by_id=datasets_by_id,
    )
    symbolic_summary = analyze_symbolic_kepler_family(merged_rows, datasets=datasets)

    max_intercept = max(
        max(abs(float(row["omega_intercept"])), abs(float(row["rddot_intercept"])))
        for row in merged_rows
    ) if merged_rows else float("inf")
    max_probe_rmse = max(
        max(float(row["omega_probe_rmse"]), float(row["rddot_probe_rmse"]))
        for row in merged_rows
    ) if merged_rows else float("inf")
    extraction_note = (
        "The coefficient tables in this summary are recovered by inverse-power regression on the saved "
        "Class-SR artifacts, evaluated on each orbit's observed radii. Separate probe-cloud diagnostics are "
        "still reported to test readout stability away from the exact trajectory samples. The direct Class-SR "
        "simplification path may remain opaque (for example `NN[x0]`) even when the extracted inverse-power "
        "family is stable and accurate."
    )

    summary = {
        "status": "extractable",
        "include_splits": include_splits,
        "data_provenance": data_provenance,
        "extraction_note": extraction_note,
        "omega": omega_table,
        "rddot": rddot_table,
        "merged_rows": merged_rows,
        "symbolic_summary": symbolic_summary,
        "probe_stability": {
            "omega": omega_probe_stability,
            "rddot": rddot_probe_stability,
        },
        "max_intercept": float(max_intercept),
        "max_probe_rmse": float(max_probe_rmse),
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print("\nMerged symbolic coefficient table")
    for row in merged_rows:
        print(
            f"{row['dataset_id']}: ell={row['ell']:.6f} "
            f"k={row['k']:.6f} mu={-row['minus_mu']:.6f} "
            f"offsets=({row['omega_intercept']:.3e},{row['rddot_intercept']:.3e}) "
            f"probe_rmse=({row['omega_probe_rmse']:.3e},{row['rddot_probe_rmse']:.3e})"
        )

    stage_a = symbolic_summary["stage_a"]
    stage_b = symbolic_summary["stage_b"]
    lift = symbolic_summary["coefficient_lift"]
    energy = symbolic_summary["energy"]
    hamiltonian = symbolic_summary["hamiltonian"]
    hamiltonian_report_path = results_root / "kepler_hamiltonian_report.md"
    _write_hamiltonian_report(
        report_path=hamiltonian_report_path,
        summary_path=summary_path,
        hamiltonian=hamiltonian,
        data_provenance=data_provenance,
        extraction_note=extraction_note,
    )
    print("\nSymbolic lift summary")
    print(f"max |ell-h| / |h|         = {stage_a['max_h_rel_error']:.3e}")
    print(f"mu_mean                   = {stage_b['mu_mean']:.9f}")
    print(f"mu_std                    = {stage_b['mu_std']:.3e}")
    print(f"max |k-h^2|               = {stage_b['max_k_abs_error']:.3e}")
    print(f"k ~= intercept + slope*h^2: intercept={lift['intercept']:+.3e}, slope={lift['slope']:+.6f}")
    print(f"energy coeff max abs err  = {energy['coeff_max_abs_error']:.3e}")
    print(f"H_reduced                 = {hamiltonian['recovered_formulas']['natural_reduced_plain']}")
    print(
        "Hamiltonian flow max RMSE = "
        f"theta:{hamiltonian['consistency']['max_theta_rmse']:.3e}, "
        f"radial:{hamiltonian['consistency']['max_radial_rmse']:.3e}"
    )
    print(
        "probe stability max rel std = "
        f"omega:{omega_probe_stability['aggregate']['max_coeff_rel_std_by_exponent']['r^-2']:.3e}, "
        f"rddot(r^-3):{rddot_probe_stability['aggregate']['max_coeff_rel_std_by_exponent']['r^-3']:.3e}, "
        f"rddot(r^-2):{rddot_probe_stability['aggregate']['max_coeff_rel_std_by_exponent']['r^-2']:.3e}"
    )
    print(f"summary json              = {summary_path}")
    print(f"hamiltonian report        = {hamiltonian_report_path}")

    if max_probe_rmse > float(args.basis_tol):
        print(
            f"FAIL: symbolic expression is not inverse-power enough under probes "
            f"({max_probe_rmse:.3e} > {float(args.basis_tol):.3e})"
        )
        return 1
    if max_intercept > float(args.intercept_tol):
        print(
            f"FAIL: symbolic inverse-power intercept too large "
            f"({max_intercept:.3e} > {float(args.intercept_tol):.3e})"
        )
        return 1

    print("\nPASS: symbolic Class-SR run produced extractable ephemeris-Kepler coefficient tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
