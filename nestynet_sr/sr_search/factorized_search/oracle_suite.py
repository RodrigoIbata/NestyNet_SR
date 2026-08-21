# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Batch oracle-suite runner for factorized symbolic search/continuous skeleton refinement ablations.

This utility runs many equation specs across iteration budgets and mode toggles
(typically ``refine_off`` vs ``refine_on``), then emits machine-readable CSV/JSON
tables for downstream analysis/plotting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import math
import pathlib
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch

from .config import FactorizedSearchConfig
from .oracle_lab import (
    _apply_cli_overrides,
    _parse_args as _parse_oracle_lab_args,
    default_oracle_hyperparams,
    load_equation_spec,
    run_oracle_equation,
)


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t == "":
            continue
        vals.append(int(t))
    out = sorted(set(v for v in vals if v > 0))
    if not out:
        raise ValueError("Expected at least one positive integer budget")
    return out


def _parse_modes(raw: str) -> list[str]:
    allowed = {"refine_off", "refine_on"}
    vals = [t.strip().lower() for t in str(raw).split(",") if t.strip()]
    out = [v for v in vals if v in allowed]
    if not out:
        raise ValueError(f"Expected modes from {sorted(allowed)}")
    # preserve first appearance order, dedup
    uniq: list[str] = []
    seen: set[str] = set()
    for v in out:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def _resolve_spec_paths(specs: Sequence[str] | None, spec_glob: str | None) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []

    for s in list(specs or []):
        p = pathlib.Path(s)
        if p.is_file():
            paths.append(p)

    if spec_glob:
        for s in sorted(glob.glob(str(spec_glob))):
            p = pathlib.Path(s)
            if p.is_file():
                paths.append(p)

    uniq: list[pathlib.Path] = []
    seen: set[str] = set()
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)

    if not uniq:
        raise FileNotFoundError("No spec files resolved. Check --specs/--spec_glob")
    return uniq


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return float((ys[mid - 1] + ys[mid]) / 2.0)


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate suite rows by ``(profile, mode, budget)``."""

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (str(r.get("profile", "current")), str(r["mode"]), int(r["budget"]))
        groups[key].append(r)

    out: list[dict[str, Any]] = []
    for (profile, mode, budget), rs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        mse_vals = [float(r["best_mse"]) for r in rs if math.isfinite(float(r["best_mse"]))]
        time_vals = [float(r["wall_seconds"]) for r in rs if math.isfinite(float(r["wall_seconds"]))]
        succ = [float(r["success"]) for r in rs]
        out.append(
            {
                "profile": profile,
                "mode": mode,
                "budget": int(budget),
                "n_runs": int(len(rs)),
                "solve_rate": _mean(succ),
                "best_mse_median": _median(mse_vals),
                "best_mse_mean": _mean(mse_vals),
                "wall_seconds_mean": _mean(time_vals),
            }
        )

    return out


def write_csv(rows: list[dict[str, Any]], path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(payload: dict[str, Any], path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_hp(args: argparse.Namespace, *, budget: int, refine_enable: bool) -> FactorizedSearchConfig:
    hp = default_oracle_hyperparams()
    hp = _apply_cli_overrides(hp, args)
    hp.n_iter = int(budget)
    hp.refine_enable = bool(refine_enable)
    return hp


def _run_oracle_suite_job(job: dict[str, Any]) -> dict[str, Any]:
    spec_path = pathlib.Path(str(job["spec_path"]))
    budget = int(job["budget"])
    mode = str(job["mode"])
    profile = str(job.get("profile", "current") or "current")
    rep = int(job["repeat"])
    rep_seed = int(job["seed"])
    dtype_name = str(job["dtype"])
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    merged_overrides = dict(job.get("hp_overrides") or {})
    merged_overrides.update(dict(job.get("profile_override") or {}))
    args = _parse_oracle_lab_args(["--spec", str(spec_path)])
    for key, value in merged_overrides.items():
        setattr(args, str(key), value)
    spec = load_equation_spec(spec_path)
    hp = _make_hp(args, budget=budget, refine_enable=(mode == "refine_on"))

    report = run_oracle_equation(
        spec,
        factorized_search_hp=hp,
        seed=rep_seed,
        dtype=dtype,
        enforce_dims=bool(job["enforce_dims"]),
        verbose=bool(job["verbose"]),
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

    success = bool(math.isfinite(best_mse) and best_mse <= float(job["success_mse_threshold"]))
    row = {
        "spec_id": str(spec.id),
        "spec_path": str(spec_path),
        "profile": profile,
        "mode": str(mode),
        "budget": int(budget),
        "repeat": int(rep),
        "seed": int(rep_seed),
        "best_mse": float(best_mse),
        "success": int(success),
        "best_expr": best_expr,
        "mapping_kind": mapping_kind,
        "wall_seconds": float(report.get("wall_seconds", float("nan"))),
    }
    return {
        "job_index": int(job["job_index"]),
        "row": row,
        "report": report,
    }


def run_oracle_suite(
    spec_paths: Sequence[str | pathlib.Path],
    *,
    budgets: Sequence[int],
    modes: Sequence[str],
    profiles: Sequence[str] | None = None,
    profile_overrides: dict[str, dict[str, Any]] | None = None,
    n_repeats: int,
    seed: int,
    dtype: torch.dtype,
    enforce_dims: bool,
    success_mse_threshold: float,
    verbose: bool,
    hp_overrides: argparse.Namespace,
    output_dir: str | pathlib.Path,
    save_individual_reports: bool,
    jobs: int = 6,
) -> dict[str, Any]:
    """Run batch oracle experiments and save tabular outputs."""

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_files = [pathlib.Path(p) for p in spec_paths]
    job_payloads: list[dict[str, Any]] = []
    hp_override_map = dict(vars(hp_overrides))
    dtype_name = "float64" if dtype == torch.float64 else "float32"
    profile_names = [str(p or "current") for p in (profiles or ["current"])]
    profile_override_map = {
        str(k): dict(v or {}) for k, v in dict(profile_overrides or {}).items()
    }
    job_index = 0
    for sp in spec_files:
        for budget in budgets:
            for mode in modes:
                for profile in profile_names:
                    for rep in range(int(n_repeats)):
                        rep_seed = int(seed) + int(rep) * 1_000_003
                        job_payloads.append(
                            {
                                "job_index": int(job_index),
                                "spec_path": str(sp),
                                "budget": int(budget),
                                "mode": str(mode),
                                "profile": str(profile),
                                "profile_override": dict(profile_override_map.get(str(profile), {})),
                                "repeat": int(rep),
                                "seed": int(rep_seed),
                                "dtype": dtype_name,
                                "enforce_dims": bool(enforce_dims),
                                "verbose": bool(verbose),
                                "success_mse_threshold": float(success_mse_threshold),
                                "hp_overrides": hp_override_map,
                            }
                        )
                        job_index += 1

    rows_by_index: dict[int, dict[str, Any]] = {}
    report_by_index: dict[int, dict[str, Any]] = {}
    max_workers = max(1, int(jobs))
    if max_workers <= 1 or len(job_payloads) <= 1:
        for job in job_payloads:
            result = _run_oracle_suite_job(job)
            rows_by_index[int(result["job_index"])] = dict(result["row"])
            report_by_index[int(result["job_index"])] = dict(result["report"])
    else:
        def _run_parallel(executor_factory):
            with executor_factory(max_workers=max_workers) as ex:
                futures = [ex.submit(_run_oracle_suite_job, job) for job in job_payloads]
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    rows_by_index[int(result["job_index"])] = dict(result["row"])
                    report_by_index[int(result["job_index"])] = dict(result["report"])

        try:
            _run_parallel(concurrent.futures.ProcessPoolExecutor)
        except (PermissionError, OSError):
            _run_parallel(concurrent.futures.ThreadPoolExecutor)

    rows: list[dict[str, Any]] = []
    for job in job_payloads:
        idx = int(job["job_index"])
        row = dict(rows_by_index[idx])
        report = report_by_index[idx]
        report_path = None
        rows.append(row)
        if save_individual_reports:
            ind_dir = out_dir / "individual_reports"
            ind_dir.mkdir(parents=True, exist_ok=True)
            spec_id = str(row["spec_id"])
            ind_path = ind_dir / f"{spec_id}.{row['profile']}.{row['mode']}.n{row['budget']}.r{row['repeat']}.json"
            write_json(report, ind_path)
            report_path = ind_path
        if report_path is not None:
            row["report_path"] = str(report_path)
            rows[-1] = dict(row)

    summary = aggregate_rows(rows)

    payload = {
        "n_specs": int(len(spec_files)),
        "spec_paths": [str(p) for p in spec_files],
        "budgets": [int(x) for x in budgets],
        "profiles": [str(x) for x in profile_names],
        "modes": [str(x) for x in modes],
        "n_repeats": int(n_repeats),
        "seed": int(seed),
        "dtype": str(dtype),
        "enforce_dims": bool(enforce_dims),
        "success_mse_threshold": float(success_mse_threshold),
        "rows": rows,
        "summary": summary,
    }

    write_json(payload, out_dir / "oracle_suite_results.json")
    write_csv(rows, out_dir / "oracle_suite_rows.csv")
    write_csv(summary, out_dir / "oracle_suite_summary.csv")

    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch oracle factorized symbolic search/continuous skeleton refinement suite runner")

    p.add_argument(
        "--specs",
        nargs="*",
        default=None,
        help="Explicit spec file paths",
    )
    p.add_argument(
        "--spec_glob",
        type=str,
        default="examples/oracle_factorized_search/specs/*.json",
        help="Glob for spec files",
    )

    p.add_argument("--budgets", type=str, default="1000,5000,20000")
    p.add_argument("--modes", type=str, default="refine_off,refine_on")
    p.add_argument("--profiles", type=str, default="current")
    p.add_argument("--n_repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    p.add_argument("--ignore_dims", action="store_true")
    p.add_argument("--success_mse", type=float, default=1.0e-6)
    p.add_argument("--output_dir", type=str, default="results/oracle_suite")
    p.add_argument("--save_individual_reports", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--jobs", type=int, default=6, help="Number of parallel worker processes")

    # Core overrides
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--poly_degree", type=int, default=None)
    p.add_argument("--return_topk", type=int, default=None)
    p.add_argument("--n_fit", type=int, default=None)
    p.add_argument("--n_probe", type=int, default=None)
    p.add_argument("--brute_depth", type=int, default=None)
    p.add_argument("--wall_time_limit_s", type=float, default=None)
    p.add_argument("--no_brute_force", action="store_true")
    p.add_argument("--n_seeds", type=int, default=None)

    split_g = p.add_mutually_exclusive_group()
    split_g.add_argument(
        "--split_iter_across_seeds",
        dest="split_iter_across_seeds",
        action="store_true",
    )
    split_g.add_argument(
        "--no_split_iter_across_seeds",
        dest="split_iter_across_seeds",
        action="store_false",
    )
    p.set_defaults(split_iter_across_seeds=None)

    # Plus overrides
    p.add_argument("--refine_lbfgs_steps", type=int, default=None)
    p.add_argument("--refine_num_restarts", type=int, default=None)
    p.add_argument("--refine_max_variants", type=int, default=None)
    p.add_argument("--refine_max_params", type=int, default=None)

    linear_g = p.add_mutually_exclusive_group()
    linear_g.add_argument(
        "--refine_linear_combo",
        dest="refine_linear_combo_enable",
        action="store_true",
    )
    linear_g.add_argument(
        "--no_refine_linear_combo",
        dest="refine_linear_combo_enable",
        action="store_false",
    )
    p.set_defaults(refine_linear_combo_enable=None)

    p.add_argument("--refine_gate_best_factor", type=float, default=None)
    p.add_argument("--refine_max_trials", type=int, default=None)

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    budgets = _parse_int_list(args.budgets)
    modes = _parse_modes(args.modes)
    profiles = [tok.strip() for tok in str(args.profiles).split(",") if tok.strip()]
    spec_paths = _resolve_spec_paths(args.specs, args.spec_glob)

    dtype = torch.float64 if str(args.dtype).lower() == "float64" else torch.float32

    payload = run_oracle_suite(
        spec_paths,
        budgets=budgets,
        modes=modes,
        profiles=profiles,
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
        dtype=dtype,
        enforce_dims=not bool(args.ignore_dims),
        success_mse_threshold=float(args.success_mse),
        verbose=not bool(args.quiet),
        hp_overrides=args,
        output_dir=args.output_dir,
        save_individual_reports=bool(args.save_individual_reports),
        jobs=int(args.jobs),
    )

    print(f"[suite] specs={payload['n_specs']} rows={len(payload['rows'])}")
    for row in payload["summary"]:
        print(
            f"[suite] profile={row['profile']} mode={row['mode']} budget={row['budget']} "
            f"solve_rate={float(row['solve_rate']):.3f} "
            f"mse_median={float(row['best_mse_median']):.3e} "
            f"time_mean={float(row['wall_seconds_mean']):.3f}s"
        )

    print(f"[suite] outputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
