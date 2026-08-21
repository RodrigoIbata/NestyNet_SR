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

from sr_demo_utils import (
    analyze_symbolic_boost_family,
    beta_to_regime_id,
    extract_classsr_affine_rows,
    load_classsr_payload,
    load_stageb_payload,
    merge_symbolic_affine_tables,
)

_DEFAULT_BETAS = "-0.8,-0.6,-0.3,0.3,0.6,0.8"
_FAST_BETAS = "-0.6,0.0,0.6"
_DEFAULT_SAMPLES = 4096
_FAST_SAMPLES = 128
_FAST_TRAIN = 64
_FAST_VAL = 32
_FAST_BATCH = 32
_FAST_CLASS_POINTS = 64


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


def _parse_beta_from_regime_id(regime_id: str) -> float:
    token = str(regime_id).strip()
    if not token.startswith("beta_") or len(token) < 7:
        raise ValueError(f"unrecognized regime id: {regime_id!r}")
    tail = token[5:]
    sign_token = tail[0]
    digits = tail[1:]
    if sign_token not in {"p", "m"} or not digits.isdigit():
        raise ValueError(f"unrecognized regime id: {regime_id!r}")
    value = float(int(digits)) / 1000.0
    return value if sign_token == "p" else -value


def _count_csv_rows(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, int(line_count) - 1)


def _infer_effective_n_samples(*file_groups: list[Path]) -> int:
    counts = [_count_csv_rows(path) for group in file_groups for path in group]
    if not counts:
        raise ValueError("cannot infer sample count from an empty file set")
    n_effective = min(int(count) for count in counts)
    if n_effective <= 1:
        raise ValueError(f"need at least 2 rows per dataset, got counts={counts!r}")
    return int(n_effective)


def _resolve_beta_by_dataset(*, data_dir: Path, fallback_betas: list[float]) -> dict[str, float]:
    manifest_path = Path(data_dir) / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        regimes = list(payload.get("regimes", []) or [])
        beta_map: dict[str, float] = {}
        for row in regimes:
            if not isinstance(row, dict):
                continue
            regime_id = row.get("regime_id", None)
            beta = row.get("beta", None)
            if regime_id is None or beta is None:
                continue
            beta_map[str(regime_id)] = float(beta)
        if beta_map:
            return beta_map
    return {
        beta_to_regime_id(float(beta)): float(beta)
        for beta in fallback_betas
    }


def _resolve_run_dimensions(
    *,
    actual_n_samples: int,
    ndata_train: int | None,
    ndata_val: int | None,
    batch_size: int | None,
    class_sr_max_points: int | None,
) -> tuple[int, int, int, int]:
    n_total = int(actual_n_samples)
    if n_total <= 1:
        raise ValueError(f"need at least 2 samples per dataset, got {n_total}")

    if ndata_train is not None:
        n_train = int(ndata_train)
    else:
        n_train = min(2000, max(16, n_total // 2))
    n_train = min(max(1, n_train), max(1, n_total - 1))

    if ndata_val is not None:
        n_val = int(ndata_val)
    else:
        n_val = min(2000, max(1, n_total - n_train))
    n_val = min(max(1, n_val), max(1, n_total - n_train))

    if batch_size is not None:
        resolved_batch_size = int(batch_size)
    else:
        resolved_batch_size = min(256, max(8, min(n_train, n_val)))
    resolved_batch_size = min(max(1, resolved_batch_size), max(1, min(n_train, n_val)))

    if class_sr_max_points is not None:
        resolved_class_points = int(class_sr_max_points)
    else:
        resolved_class_points = min(n_train, n_val)
    resolved_class_points = min(max(1, resolved_class_points), max(1, min(n_train, n_val)))

    return int(n_train), int(n_val), int(resolved_batch_size), int(resolved_class_points)


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
        str(script_dir / "generate_interval_data.py"),
        f"--betas={str(args.betas)}",
        "--n_samples",
        str(int(args.n_samples)),
        "--seed",
        str(int(args.seed)),
        "--u_max",
        str(float(args.u_max)),
        "--x_max",
        str(float(args.x_max)),
        "--noise_std",
        str(float(args.noise_std)),
        "--near_null_width",
        str(float(args.near_null_width)),
        "--output_dir",
        str(script_dir / "data"),
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _target_filepaths(data_dir: Path, target: str) -> list[Path]:
    target_dir = data_dir / target
    if not target_dir.exists():
        raise FileNotFoundError(f"missing generated target directory: {target_dir}")
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_paths = []
        for row in list(payload.get("regimes", []) or []):
            if not isinstance(row, dict):
                continue
            key = f"{str(target)}_csv"
            csv_path = row.get(key, None)
            if csv_path is None:
                continue
            selected_paths.append(Path(str(csv_path)))
        if selected_paths:
            return sorted(selected_paths)
    return sorted(target_dir.glob("*.csv"))


def _filepaths_to_regime_ids(filepaths: list[Path], *, target_name: str) -> list[str]:
    prefix = f"{str(target_name)}_"
    regime_ids: list[str] = []
    for path in filepaths:
        stem = Path(path).stem
        regime_ids.append(stem[len(prefix):] if stem.startswith(prefix) else stem)
    return regime_ids


def _build_run_sr_cmd(
    *,
    filepaths: list[Path],
    metadata_path: Path,
    fast: bool,
    class_cv_threshold: float,
    ndata_train: int,
    ndata_val: int,
    batch_size: int,
    class_sr_max_points: int,
) -> list[str]:
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
        "--class_param_sr_metadata",
        metadata_path.read_text(encoding="utf-8"),
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
        "[1,0]",
        "--x_units",
        "[[1,0],[1,0]]",
        "--units_basis",
        "L,T",
        "--log_level",
        "INFO",
    ]
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
    model_suffixes = (
        ".identity.mod",
        ".mod",
    )
    for suffix in model_suffixes:
        src = repo_models_dir / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, target_dir / src.name)


def _extract_table_or_diagnostic(
    *,
    target_name: str,
    target_dir: Path,
    filepaths: list[Path],
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
        table = extract_classsr_affine_rows(
            class_sr_json_path=class_sr_json_path,
            stageb_pkl_path=stageb_pkl_path,
        )
    except Exception as exc:
        diagnostic["status"] = "opaque_or_non_affine"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run symbolic Class-SR discovery for the special-relativity interval demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generate", action="store_true", help="Regenerate the interval CSV inputs first")
    parser.add_argument("--betas", type=str, default=_DEFAULT_BETAS, help="Comma-separated beta grid")
    parser.add_argument("--n_samples", type=int, default=_DEFAULT_SAMPLES, help="Samples per regime when generating")
    parser.add_argument("--seed", type=int, default=123, help="Base seed when generating")
    parser.add_argument("--u_max", type=float, default=10.0, help="Generation scale for u")
    parser.add_argument("--x_max", type=float, default=10.0, help="Generation scale for x")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Generation noise on primed observables")
    parser.add_argument("--near_null_width", type=float, default=0.03, help="Light-cone shell thickness")
    parser.add_argument("--fast", action="store_true", help="Pass --fast to run_SR.py")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Skip new run_SR.py launches and reuse existing Class-SR artifacts already present on disk",
    )
    parser.add_argument("--class_cv_threshold", type=float, default=0.15, help="Class-SR CV threshold")
    parser.add_argument("--ndata_train", type=int, default=None, help="Override run_SR train sample count")
    parser.add_argument("--ndata_val", type=int, default=None, help="Override run_SR validation sample count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override run_SR batch size")
    parser.add_argument(
        "--class_sr_max_points",
        type=int,
        default=None,
        help="Override per-dataset point cap used inside Class-SR joint fitting",
    )
    parser.add_argument("--affine_tol", type=float, default=1.0e-4, help="Max allowed probe non-affinity RMSE")
    parser.add_argument("--intercept_tol", type=float, default=1.0e-4, help="Max allowed affine intercept magnitude")
    parser.add_argument(
        "--strict_extract",
        action="store_true",
        help="Return a nonzero exit code if the symbolic run finishes but does not expose extractable affine maps",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path("results") / "special_relativity_classsr"),
        help="Root output directory for symbolic SR runs",
    )
    args = parser.parse_args()

    if bool(args.fast):
        if str(args.betas) == _DEFAULT_BETAS:
            args.betas = _FAST_BETAS
        if int(args.n_samples) == _DEFAULT_SAMPLES:
            args.n_samples = _FAST_SAMPLES
        if args.ndata_train is None:
            args.ndata_train = _FAST_TRAIN
        if args.ndata_val is None:
            args.ndata_val = _FAST_VAL
        if args.batch_size is None:
            args.batch_size = _FAST_BATCH
        if args.class_sr_max_points is None:
            args.class_sr_max_points = _FAST_CLASS_POINTS
        print(
            "[SR fast profile] using compact symbolic smoke settings: "
            f"betas={args.betas}, n_samples={args.n_samples}, "
            f"ndata_train={args.ndata_train}, ndata_val={args.ndata_val}, "
            f"batch_size={args.batch_size}, class_sr_max_points={args.class_sr_max_points}, "
            "single_layer, no_stageA_separabilities, no_factorized_search, no_factorized_search_plus, "
            "no_class_param_sr, disable_compound_detection"
        )

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    _generate_if_requested(args, script_dir)

    metadata_path = data_dir / "param_sr_metadata_rows.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"missing metadata rows: {metadata_path}. Run with --generate first."
        )

    uprime_files = _target_filepaths(data_dir, "uprime")
    xprime_files = _target_filepaths(data_dir, "xprime")
    beta_by_dataset = _resolve_beta_by_dataset(
        data_dir=data_dir,
        fallback_betas=_parse_betas(args.betas),
    )
    actual_n_samples = _infer_effective_n_samples(uprime_files, xprime_files)
    if int(actual_n_samples) != int(args.n_samples):
        print(
            "[SR note] using on-disk dataset size instead of requested n_samples: "
            f"requested={int(args.n_samples)}, actual={int(actual_n_samples)}"
        )

    results_root = Path(args.results_dir).resolve()
    uprime_out = results_root / "uprime"
    xprime_out = results_root / "xprime"
    repo_results_dir = Path("results").resolve()
    repo_models_dir = Path("models").resolve()
    n_train, n_val, batch_size, class_sr_max_points = _resolve_run_dimensions(
        actual_n_samples=actual_n_samples,
        ndata_train=args.ndata_train,
        ndata_val=args.ndata_val,
        batch_size=args.batch_size,
        class_sr_max_points=args.class_sr_max_points,
    )
    print(
        "[SR split] "
        f"datasets={len(uprime_files)}, n_samples={actual_n_samples}, "
        f"ndata_train={n_train}, ndata_val={n_val}, "
        f"batch_size={batch_size}, class_sr_max_points={class_sr_max_points}"
    )

    if not bool(args.reuse_existing):
        print("\n=== Class-SR symbolic discovery: u' target ===")
        rc = _run_and_stream(
            _build_run_sr_cmd(
                filepaths=uprime_files,
                metadata_path=metadata_path,
                fast=bool(args.fast),
                class_cv_threshold=float(args.class_cv_threshold),
                ndata_train=n_train,
                ndata_val=n_val,
                batch_size=batch_size,
                class_sr_max_points=class_sr_max_points,
            ),
            log_path=uprime_out / "run.log",
        )
        if rc != 0:
            print(f"u' Class-SR run failed with exit code {rc}")
            return rc
    else:
        print("\n=== Reusing existing Class-SR artifacts: u' target ===")
    _copy_run_sr_artifacts(
        filepaths=uprime_files,
        repo_results_dir=repo_results_dir,
        repo_models_dir=repo_models_dir,
        target_dir=uprime_out,
    )

    if not bool(args.reuse_existing):
        print("\n=== Class-SR symbolic discovery: x' target ===")
        rc = _run_and_stream(
            _build_run_sr_cmd(
                filepaths=xprime_files,
                metadata_path=metadata_path,
                fast=bool(args.fast),
                class_cv_threshold=float(args.class_cv_threshold),
                ndata_train=n_train,
                ndata_val=n_val,
                batch_size=batch_size,
                class_sr_max_points=class_sr_max_points,
            ),
            log_path=xprime_out / "run.log",
        )
        if rc != 0:
            print(f"x' Class-SR run failed with exit code {rc}")
            return rc
    else:
        print("\n=== Reusing existing Class-SR artifacts: x' target ===")
    _copy_run_sr_artifacts(
        filepaths=xprime_files,
        repo_results_dir=repo_results_dir,
        repo_models_dir=repo_models_dir,
        target_dir=xprime_out,
    )

    uprime_result = _extract_table_or_diagnostic(
        target_name="uprime",
        target_dir=uprime_out,
        filepaths=uprime_files,
    )
    xprime_result = _extract_table_or_diagnostic(
        target_name="xprime",
        target_dir=xprime_out,
        filepaths=xprime_files,
    )

    summary_path = results_root / "symbolic_interval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        str(uprime_result.get("status")) != "extractable"
        or str(xprime_result.get("status")) != "extractable"
    ):
        summary = {
            "status": "opaque_stageb_models",
            "uprime": uprime_result,
            "xprime": xprime_result,
        }
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        print("\nSymbolic extraction summary")
        print("status                    = opaque_stageb_models")
        print(f"summary json              = {summary_path}")
        for target_name, payload in (("uprime", uprime_result), ("xprime", xprime_result)):
            print(
                f"{target_name:25s}= {payload.get('status')} "
                f"({payload.get('error', payload.get('stageb_root_repr', 'no detail'))})"
            )
        if bool(args.strict_extract):
            print("\nFAIL: symbolic runs completed but Stage B remained opaque to affine extraction")
            return 1
        print(
            "\nNOTE: symbolic runs completed, but the fast-profile Stage B/Class-SR outputs "
            "did not expose extractable affine maps yet."
        )
        return 0

    uprime_table = dict(uprime_result["table"])
    xprime_table = dict(xprime_result["table"])
    extracted_ids = set(uprime_table.keys()) | set(xprime_table.keys())
    missing_beta_ids = sorted(dataset_id for dataset_id in extracted_ids if dataset_id not in beta_by_dataset)
    if missing_beta_ids:
        derived_beta_ids: dict[str, float] = {}
        for dataset_id in missing_beta_ids:
            try:
                derived_beta_ids[str(dataset_id)] = _parse_beta_from_regime_id(dataset_id)
            except ValueError:
                continue
        beta_by_dataset.update(derived_beta_ids)

    merged_rows = merge_symbolic_affine_tables(
        uprime_table,
        xprime_table,
        beta_by_dataset=beta_by_dataset,
    )
    symbolic_summary = analyze_symbolic_boost_family(merged_rows)

    max_intercept = max(
        max(abs(float(row["u_intercept"])), abs(float(row["x_intercept"])))
        for row in merged_rows
    ) if merged_rows else float("inf")
    max_probe_rmse = max(
        max(float(row["u_probe_rmse"]), float(row["x_probe_rmse"]))
        for row in merged_rows
    ) if merged_rows else float("inf")

    summary = {
        "status": "extractable",
        "uprime": uprime_table,
        "xprime": xprime_table,
        "merged_rows": merged_rows,
        "symbolic_summary": symbolic_summary,
        "max_intercept": float(max_intercept),
        "max_probe_rmse": float(max_probe_rmse),
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print("\nMerged symbolic coefficient table")
    for row in merged_rows:
        print(
            f"{row['dataset_id']}: beta={row['beta']:+.3f} "
            f"a={row['a']:.6f} b={row['b']:.6f} c={row['c']:.6f} d={row['d']:.6f} "
            f"offsets=({row['u_intercept']:.3e},{row['x_intercept']:.3e}) "
            f"probe_rmse=({row['u_probe_rmse']:.3e},{row['x_probe_rmse']:.3e})"
        )

    coeff = symbolic_summary["coefficient_laws"]
    metric = symbolic_summary["metric"]
    print("\nSymbolic lift summary")
    print(f"max |(-b/a)-beta|         = {coeff['max_beta_residual']:.3e}")
    print(f"max |(1/a^2)-(1-beta^2)| = {coeff['max_z_residual']:.3e}")
    print(f"max |a-gamma(beta)|       = {coeff['gamma_max_abs_error']:.3e}")
    print(f"max preserve err          = {metric['max_preservation_error']:.3e}")
    print(f"metric indefinite         = {metric['is_indefinite']}")
    print(f"summary json              = {summary_path}")

    if max_probe_rmse > float(args.affine_tol):
        print(
            f"FAIL: symbolic expression is not affine enough under probes "
            f"({max_probe_rmse:.3e} > {float(args.affine_tol):.3e})"
        )
        return 1
    if max_intercept > float(args.intercept_tol):
        print(
            f"FAIL: symbolic affine intercept too large "
            f"({max_intercept:.3e} > {float(args.intercept_tol):.3e})"
        )
        return 1

    print("\nPASS: symbolic Class-SR run produced extractable affine boost tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
