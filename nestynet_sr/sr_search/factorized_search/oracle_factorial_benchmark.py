# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Reduced factorial oracle benchmark for integrated factorized symbolic search utility.

This runner sweeps a reduced spec suite across the four clean binary toggles
already exposed by the oracle factorized symbolic search harness:

- ``refine_enable``
- ``inverse_steering_enable``
- ``inverse_spec_enable``
- ``hole_search_enable``

It persists one machine-readable row per ``target x seed x arm x budget``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestynet_sr.sr_search.factorized_search.oracle_lab import (
    _apply_cli_overrides,
    _parse_args as _parse_oracle_lab_args,
    default_oracle_hyperparams,
    load_equation_spec,
    run_oracle_equation,
)


REPO_ROOT = ROOT
DEFAULT_SUITE_MANIFEST = (
    REPO_ROOT / "examples" / "oracle_factorized_search" / "factorial_suites" / "reduced_quick4.json"
)
DEFAULT_TOGGLE_ORDER = (
    "refine_enable",
    "inverse_steering_enable",
    "inverse_spec_enable",
    "hole_search_enable",
)
_ARM_TOKEN = {
    "refine_enable": "plus",
    "inverse_steering_enable": "inv",
    "inverse_spec_enable": "spec",
    "hole_search_enable": "hole",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return str(value)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def load_factorial_suite(path: str | pathlib.Path | None = None) -> tuple[pathlib.Path, dict[str, Any]]:
    manifest_path = pathlib.Path(path) if path is not None else DEFAULT_SUITE_MANIFEST
    payload = _load_json(manifest_path)
    specs = list(payload.get("specs") or [])
    if not specs:
        raise ValueError(f"No specs declared in factorial suite manifest: {manifest_path}")
    return manifest_path, payload


def _resolve_suite_spec_path(raw: str, *, manifest_path: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(str(raw))
    candidates: list[pathlib.Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(manifest_path.parent / p)
        candidates.append(REPO_ROOT / p)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Could not resolve suite spec path: {raw}")


def resolve_suite_spec_paths(payload: Mapping[str, Any], *, manifest_path: pathlib.Path) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    seen: set[str] = set()
    for item in list(payload.get("specs") or []):
        raw = str(item.get("path", "") if isinstance(item, dict) else item or "")
        if raw == "":
            continue
        resolved = _resolve_suite_spec_path(raw, manifest_path=manifest_path).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    if not out:
        raise ValueError(f"Factorial suite resolved zero spec files: {manifest_path}")
    return out


def _parse_positive_int_list(raw: str | None, default: Sequence[int]) -> list[int]:
    if raw is None:
        vals = [int(v) for v in default]
    else:
        vals = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    out = sorted(set(v for v in vals if v > 0))
    if not out:
        raise ValueError("Expected at least one positive integer")
    return out


def _parse_seed_list(raw: str | None, default: Sequence[int]) -> list[int]:
    if raw is None:
        vals = [int(v) for v in default]
    else:
        vals = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    out = sorted(set(v for v in vals if v >= 0))
    if not out:
        raise ValueError("Expected at least one non-negative seed")
    return out


def _toggle_order_from_manifest(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = list(payload.get("toggle_order") or DEFAULT_TOGGLE_ORDER)
    out: list[str] = []
    seen: set[str] = set()
    for token in raw:
        key = str(token or "").strip()
        if key in DEFAULT_TOGGLE_ORDER and key not in seen:
            out.append(key)
            seen.add(key)
    if tuple(out) != DEFAULT_TOGGLE_ORDER:
        missing = [name for name in DEFAULT_TOGGLE_ORDER if name not in out]
        out.extend(missing)
    return tuple(out)


def enumerate_factorial_arms(toggle_order: Sequence[str] | None = None) -> list[dict[str, Any]]:
    order = tuple(toggle_order or DEFAULT_TOGGLE_ORDER)
    arms: list[dict[str, Any]] = []
    for bits in itertools.product((0, 1), repeat=len(order)):
        toggles = {name: bool(bit) for name, bit in zip(order, bits)}
        arm_id = "_".join(f"{_ARM_TOKEN.get(name, name)}{int(bool(toggles[name]))}" for name in order)
        arms.append({"arm_id": arm_id, "toggles": toggles})
    return arms


def _mean(values: Iterable[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _median(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return float("nan")
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def aggregate_rows_by_arm(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        if str(row.get("status", "ok")) != "ok":
            continue
        groups[(str(row.get("arm_id", "")), int(row.get("budget", 0) or 0))].append(dict(row))
    out: list[dict[str, Any]] = []
    for (arm_id, budget), group_rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        first = group_rows[0]
        out.append(
            {
                "arm_id": arm_id,
                "budget": int(budget),
                "refine_enable": bool(first.get("refine_enable", False)),
                "inverse_steering_enable": bool(first.get("inverse_steering_enable", False)),
                "inverse_spec_enable": bool(first.get("inverse_spec_enable", False)),
                "hole_search_enable": bool(first.get("hole_search_enable", False)),
                "n_runs": int(len(group_rows)),
                "solve_rate": _mean(float(row.get("success", 0.0) or 0.0) for row in group_rows),
                "best_mse_mean": _mean(float(row.get("best_mse", float("inf"))) for row in group_rows),
                "best_mse_median": _median(float(row.get("best_mse", float("inf"))) for row in group_rows),
                "wall_seconds_mean": _mean(float(row.get("wall_seconds", float("nan"))) for row in group_rows),
            }
        )
    return out


def aggregate_rows_by_target(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        if str(row.get("status", "ok")) != "ok":
            continue
        groups[(str(row.get("spec_id", "")), str(row.get("arm_id", "")), int(row.get("budget", 0) or 0))].append(dict(row))
    out: list[dict[str, Any]] = []
    for (spec_id, arm_id, budget), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
        first = group_rows[0]
        out.append(
            {
                "spec_id": spec_id,
                "arm_id": arm_id,
                "budget": int(budget),
                "refine_enable": bool(first.get("refine_enable", False)),
                "inverse_steering_enable": bool(first.get("inverse_steering_enable", False)),
                "inverse_spec_enable": bool(first.get("inverse_spec_enable", False)),
                "hole_search_enable": bool(first.get("hole_search_enable", False)),
                "n_runs": int(len(group_rows)),
                "solve_rate": _mean(float(row.get("success", 0.0) or 0.0) for row in group_rows),
                "best_mse_mean": _mean(float(row.get("best_mse", float("inf"))) for row in group_rows),
                "best_mse_median": _median(float(row.get("best_mse", float("inf"))) for row in group_rows),
                "wall_seconds_mean": _mean(float(row.get("wall_seconds", float("nan"))) for row in group_rows),
            }
        )
    return out


def _effective_toggles(raw_toggles: Mapping[str, Any]) -> dict[str, bool]:
    refine_enable = bool(raw_toggles.get("refine_enable", False))
    inverse_steering_enable = bool(raw_toggles.get("inverse_steering_enable", False))
    inverse_spec_enable = bool(raw_toggles.get("inverse_spec_enable", False)) and inverse_steering_enable
    hole_search_enable = bool(raw_toggles.get("hole_search_enable", False)) and inverse_steering_enable and inverse_spec_enable
    return {
        "refine_enable": refine_enable,
        "inverse_steering_enable": inverse_steering_enable,
        "inverse_spec_enable": inverse_spec_enable,
        "hole_search_enable": hole_search_enable,
    }


def _build_hp(overrides: Mapping[str, Any], *, spec_path: pathlib.Path, budget: int, arm_toggles: Mapping[str, Any]) -> Any:
    args = _parse_oracle_lab_args(["--spec", str(spec_path)])
    for key, value in dict(overrides or {}).items():
        setattr(args, str(key), value)
    for key, value in dict(arm_toggles or {}).items():
        setattr(args, str(key), bool(value))
    hp = default_oracle_hyperparams()
    hp = _apply_cli_overrides(hp, args)
    hp.n_iter = int(budget)
    hp.refine_enable = bool(arm_toggles.get("refine_enable", False))
    return hp


def _run_factorial_job(job: Mapping[str, Any]) -> dict[str, Any]:
    spec_path = pathlib.Path(str(job["spec_path"]))
    arm = dict(job["arm"])
    raw_toggles = dict(arm.get("toggles", {}) or {})
    effective_toggles = _effective_toggles(raw_toggles)
    budget = int(job["budget"])
    seed = int(job["seed"])
    dtype_name = str(job.get("dtype", "float64"))
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    quiet = bool(job.get("quiet", True))

    try:
        spec = load_equation_spec(spec_path)
        hp = _build_hp(dict(job.get("hp_overrides") or {}), spec_path=spec_path, budget=budget, arm_toggles=raw_toggles)
        report = run_oracle_equation(
            spec,
            factorized_search_hp=hp,
            seed=seed,
            dtype=dtype,
            enforce_dims=bool(job.get("enforce_dims", True)),
            verbose=not quiet,
        )
        best = report.get("best")
        if best is None:
            best_mse = float("inf")
            best_expr = ""
            mapping_kind = ""
        else:
            best_mse = float(best.get("mse", float("inf")))
            best_expr = str(best.get("expr", ""))
            mapping_kind = str(best.get("mapping_kind", ""))
        row = {
            "status": "ok",
            "spec_id": str(spec.id),
            "spec_path": str(spec_path),
            "budget": int(budget),
            "seed": int(seed),
            "arm_id": str(arm["arm_id"]),
            "refine_enable": bool(raw_toggles.get("refine_enable", False)),
            "inverse_steering_enable": bool(raw_toggles.get("inverse_steering_enable", False)),
            "inverse_spec_enable": bool(raw_toggles.get("inverse_spec_enable", False)),
            "hole_search_enable": bool(raw_toggles.get("hole_search_enable", False)),
            "effective_refine_enable": bool(effective_toggles["refine_enable"]),
            "effective_inverse_steering_enable": bool(effective_toggles["inverse_steering_enable"]),
            "effective_inverse_spec_enable": bool(effective_toggles["inverse_spec_enable"]),
            "effective_hole_search_enable": bool(effective_toggles["hole_search_enable"]),
            "best_mse": float(best_mse),
            "success": int(math.isfinite(best_mse) and best_mse <= float(job["success_mse_threshold"])),
            "best_expr": best_expr,
            "mapping_kind": mapping_kind,
            "wall_seconds": float(report.get("wall_seconds", float("nan"))),
        }
        return {"row": row, "report": report}
    except Exception as exc:
        row = {
            "status": "error",
            "spec_id": "",
            "spec_path": str(spec_path),
            "budget": int(budget),
            "seed": int(seed),
            "arm_id": str(arm["arm_id"]),
            "refine_enable": bool(raw_toggles.get("refine_enable", False)),
            "inverse_steering_enable": bool(raw_toggles.get("inverse_steering_enable", False)),
            "inverse_spec_enable": bool(raw_toggles.get("inverse_spec_enable", False)),
            "hole_search_enable": bool(raw_toggles.get("hole_search_enable", False)),
            "effective_refine_enable": bool(effective_toggles["refine_enable"]),
            "effective_inverse_steering_enable": bool(effective_toggles["inverse_steering_enable"]),
            "effective_inverse_spec_enable": bool(effective_toggles["inverse_spec_enable"]),
            "effective_hole_search_enable": bool(effective_toggles["hole_search_enable"]),
            "best_mse": None,
            "success": 0,
            "best_expr": "",
            "mapping_kind": "",
            "wall_seconds": None,
            "error": str(exc),
        }
        return {"row": row, "report": None}


def run_factorial_suite(
    *,
    spec_paths: Sequence[pathlib.Path],
    arms: Sequence[dict[str, Any]],
    budgets: Sequence[int],
    seeds: Sequence[int],
    output_dir: pathlib.Path,
    hp_overrides: Mapping[str, Any],
    dtype: torch.dtype,
    enforce_dims: bool,
    success_mse_threshold: float,
    quiet: bool,
    jobs: int,
    save_individual_reports: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    if save_individual_reports:
        cases_dir.mkdir(parents=True, exist_ok=True)

    job_payloads: list[dict[str, Any]] = []
    dtype_name = "float64" if dtype == torch.float64 else "float32"
    for spec_path in spec_paths:
        for budget in budgets:
            for seed in seeds:
                for arm in arms:
                    job_payloads.append(
                        {
                            "spec_path": str(spec_path),
                            "budget": int(budget),
                            "seed": int(seed),
                            "arm": dict(arm),
                            "dtype": dtype_name,
                            "enforce_dims": bool(enforce_dims),
                            "success_mse_threshold": float(success_mse_threshold),
                            "quiet": bool(quiet),
                            "hp_overrides": dict(hp_overrides or {}),
                        }
                    )

    results: list[dict[str, Any]] = []
    max_workers = max(1, int(jobs))
    if max_workers == 1:
        for job in job_payloads:
            results.append(_run_factorial_job(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            for result in ex.map(_run_factorial_job, job_payloads):
                results.append(result)

    rows: list[dict[str, Any]] = []
    for result, job in zip(results, job_payloads):
        row = dict(result.get("row", {}) or {})
        report = result.get("report", None)
        rows.append(row)
        if save_individual_reports and report is not None and str(row.get("status", "")) == "ok":
            spec_stem = pathlib.Path(str(job["spec_path"])).stem
            report_path = cases_dir / f"{spec_stem}__seed{int(job['seed']):04d}__iter{int(job['budget']):04d}__{row['arm_id']}.json"
            _write_json({"row": row, "report": report}, report_path)

    arm_summary = aggregate_rows_by_arm(rows)
    target_summary = aggregate_rows_by_target(rows)
    _write_json({"rows": rows}, output_dir / "oracle_factorial_rows.json")
    _write_csv(rows, output_dir / "oracle_factorial_rows.csv")
    _write_json({"summary": arm_summary}, output_dir / "oracle_factorial_arm_summary.json")
    _write_csv(arm_summary, output_dir / "oracle_factorial_arm_summary.csv")
    _write_json({"summary": target_summary}, output_dir / "oracle_factorial_target_summary.json")
    _write_csv(target_summary, output_dir / "oracle_factorial_target_summary.csv")
    return {"rows": rows, "arm_summary": arm_summary, "target_summary": target_summary}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reduced factorial oracle benchmark for factorized symbolic search integrated utility")
    p.add_argument("--suite_manifest", type=str, default=str(DEFAULT_SUITE_MANIFEST))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--budgets", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default=None)
    p.add_argument("--ignore_dims", action="store_true", default=None)
    p.add_argument("--success_mse", type=float, default=None)
    p.add_argument("--quiet", action="store_true", default=None)
    p.add_argument("--fast_benchmark", action="store_true", default=None)
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--save_individual_reports", action="store_true")

    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--poly_degree", type=int, default=None)
    p.add_argument("--return_topk", type=int, default=None)
    p.add_argument("--n_fit", type=int, default=None)
    p.add_argument("--n_probe", type=int, default=None)
    p.add_argument("--brute_depth", type=int, default=None)
    p.add_argument("--wall_time_limit_s", type=float, default=None)
    p.add_argument("--no_brute_force", action="store_true", default=None)
    p.add_argument("--n_seeds", type=int, default=None)

    split_g = p.add_mutually_exclusive_group()
    split_g.add_argument("--split_iter_across_seeds", dest="split_iter_across_seeds", action="store_true")
    split_g.add_argument("--no_split_iter_across_seeds", dest="split_iter_across_seeds", action="store_false")
    p.set_defaults(split_iter_across_seeds=None)

    p.add_argument("--refine_lbfgs_steps", type=int, default=None)
    p.add_argument("--refine_num_restarts", type=int, default=None)
    p.add_argument("--refine_max_variants", type=int, default=None)
    p.add_argument("--refine_max_params", type=int, default=None)
    linear_g = p.add_mutually_exclusive_group()
    linear_g.add_argument("--refine_linear_combo", dest="refine_linear_combo_enable", action="store_true")
    linear_g.add_argument("--no_refine_linear_combo", dest="refine_linear_combo_enable", action="store_false")
    p.set_defaults(refine_linear_combo_enable=None)
    p.add_argument("--refine_gate_best_factor", type=float, default=None)
    p.add_argument("--refine_max_trials", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path, manifest = load_factorial_suite(args.suite_manifest)
    spec_paths = resolve_suite_spec_paths(manifest, manifest_path=manifest_path)
    defaults = dict(manifest.get("defaults", {}) or {})
    suite_id = str(manifest.get("suite_id", "oracle_factorial_suite") or "oracle_factorial_suite")

    budgets = _parse_positive_int_list(args.budgets, defaults.get("budgets", [1000]))
    seeds = _parse_seed_list(args.seeds, defaults.get("seeds", [0]))
    dtype_name = str(args.dtype or defaults.get("dtype", "float64")).lower()
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    ignore_dims = bool(args.ignore_dims if args.ignore_dims is not None else defaults.get("ignore_dims", False))
    success_mse = float(args.success_mse if args.success_mse is not None else defaults.get("success_mse", 1.0e-6))
    quiet = bool(args.quiet if args.quiet is not None else defaults.get("quiet", True))
    fast_benchmark = bool(args.fast_benchmark if args.fast_benchmark is not None else defaults.get("fast_benchmark", False))
    jobs = int(args.jobs if args.jobs is not None else defaults.get("jobs", 1))
    if args.wall_time_limit_s is None and defaults.get("wall_time_limit_s", None) is not None:
        args.wall_time_limit_s = float(defaults.get("wall_time_limit_s"))
    if fast_benchmark:
        args.no_brute_force = True

    output_dir = pathlib.Path(args.output_dir or (REPO_ROOT / "results" / f"oracle_factorial_{suite_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    toggle_order = _toggle_order_from_manifest(manifest)
    arms = enumerate_factorial_arms(toggle_order)
    hp_overrides = dict(vars(args))
    hp_overrides.pop("suite_manifest", None)
    hp_overrides.pop("output_dir", None)
    hp_overrides.pop("budgets", None)
    hp_overrides.pop("seeds", None)
    hp_overrides.pop("dtype", None)
    hp_overrides.pop("ignore_dims", None)
    hp_overrides.pop("success_mse", None)
    hp_overrides.pop("quiet", None)
    hp_overrides.pop("fast_benchmark", None)
    hp_overrides.pop("jobs", None)
    hp_overrides.pop("save_individual_reports", None)

    payload = run_factorial_suite(
        spec_paths=spec_paths,
        arms=arms,
        budgets=budgets,
        seeds=seeds,
        output_dir=output_dir,
        hp_overrides=hp_overrides,
        dtype=dtype,
        enforce_dims=not ignore_dims,
        success_mse_threshold=success_mse,
        quiet=quiet,
        jobs=jobs,
        save_individual_reports=bool(args.save_individual_reports),
    )

    result_payload = {
        "suite_id": suite_id,
        "suite_manifest": str(manifest_path),
        "budgets": [int(v) for v in budgets],
        "seeds": [int(v) for v in seeds],
        "toggle_order": list(toggle_order),
        "arms": arms,
        "rows": payload["rows"],
        "arm_summary": payload["arm_summary"],
        "target_summary": payload["target_summary"],
    }
    _write_json(result_payload, output_dir / "oracle_factorial_results.json")

    n_errors = sum(1 for row in payload["rows"] if str(row.get("status", "")) != "ok")
    print(
        f"[oracle_factorial] suite={suite_id} specs={len(spec_paths)} budgets={budgets} "
        f"seeds={seeds} arms={len(arms)} output_dir={output_dir}"
    )
    if n_errors:
        print(f"[oracle_factorial] errors={n_errors}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
