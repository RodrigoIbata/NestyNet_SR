# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Small fixed regression-suite runner for oracle factorized symbolic search workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch

from nestynet_sr.discovery.active_design import resolve_disagreement_mode

from .oracle_discovery_benchmark import run_oracle_discovery_benchmark
from .oracle_suite import run_oracle_suite


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_SUITE_MANIFEST = REPO_ROOT / "examples" / "oracle_factorized_search" / "regression_suites" / "quick12.json"


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_regression_suite(path: str | pathlib.Path | None = None) -> tuple[pathlib.Path, dict[str, Any]]:
    manifest_path = pathlib.Path(path) if path is not None else DEFAULT_SUITE_MANIFEST
    payload = _load_json(manifest_path)
    specs = list(payload.get("specs") or [])
    if not specs:
        raise ValueError(f"No specs declared in suite manifest: {manifest_path}")
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


def resolve_suite_spec_paths(payload: dict[str, Any], *, manifest_path: pathlib.Path) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    seen: set[str] = set()
    for item in list(payload.get("specs") or []):
        if isinstance(item, dict):
            raw = str(item.get("path", "") or "")
        else:
            raw = str(item or "")
        if not raw:
            continue
        resolved = _resolve_suite_spec_path(raw, manifest_path=manifest_path).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    if not out:
        raise ValueError(f"Suite manifest resolved zero spec files: {manifest_path}")
    return out


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


def aggregate_rows_by_spec(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("spec_id", "")),
            str(row.get("profile", "current")),
            str(row.get("mode", "")),
            int(row.get("budget", 0) or 0),
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for (spec_id, profile, mode, budget), rs in sorted(groups.items(), key=lambda item: (item[0][0], item[0][3], item[0][2], item[0][1])):
        mse_vals = [float(r.get("best_mse", float("inf"))) for r in rs if math.isfinite(float(r.get("best_mse", float("inf"))))]
        wall_vals = [float(r.get("wall_seconds", float("nan"))) for r in rs if math.isfinite(float(r.get("wall_seconds", float("nan"))))]
        succ_vals = [float(r.get("success", 0.0) or 0.0) for r in rs]
        out.append(
            {
                "spec_id": spec_id,
                "profile": profile,
                "mode": mode,
                "budget": int(budget),
                "n_runs": int(len(rs)),
                "solve_rate": _mean(succ_vals),
                "best_mse_median": _median(mse_vals),
                "best_mse_mean": _mean(mse_vals),
                "wall_seconds_mean": _mean(wall_vals),
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compare_spec_summaries(
    current_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    mse_factor: float,
    time_factor: float,
) -> list[dict[str, Any]]:
    baseline_map = {
        (
            str(r.get("spec_id", "")),
            str(r.get("profile", "current")),
            str(r.get("mode", "")),
            int(r.get("budget", 0) or 0),
        ): r
        for r in list(baseline_rows or [])
    }
    regressions: list[dict[str, Any]] = []
    for row in list(current_rows or []):
        key = (
            str(row.get("spec_id", "")),
            str(row.get("profile", "current")),
            str(row.get("mode", "")),
            int(row.get("budget", 0) or 0),
        )
        base = baseline_map.get(key)
        if base is None:
            continue
        cur_mse = float(row.get("best_mse_median", float("inf")) or float("inf"))
        base_mse = float(base.get("best_mse_median", float("inf")) or float("inf"))
        cur_time = float(row.get("wall_seconds_mean", float("nan")) or float("nan"))
        base_time = float(base.get("wall_seconds_mean", float("nan")) or float("nan"))
        cur_solve = float(row.get("solve_rate", 0.0) or 0.0)
        base_solve = float(base.get("solve_rate", 0.0) or 0.0)

        reasons: list[str] = []
        if base_solve >= 0.999 and cur_solve < base_solve:
            reasons.append(f"solve_rate {cur_solve:.3f} < baseline {base_solve:.3f}")
        if math.isfinite(base_mse) and math.isfinite(cur_mse):
            if cur_mse > max(1.0e-12, base_mse) * float(mse_factor):
                reasons.append(f"best_mse_median {cur_mse:.3e} > {float(mse_factor):.2f}x baseline {base_mse:.3e}")
        if math.isfinite(base_time) and math.isfinite(cur_time):
            if cur_time > max(1.0e-12, base_time) * float(time_factor):
                reasons.append(f"wall_seconds_mean {cur_time:.3f}s > {float(time_factor):.2f}x baseline {base_time:.3f}s")
        if reasons:
            regressions.append(
                {
                    "spec_id": key[0],
                    "profile": key[1],
                    "mode": key[2],
                    "budget": int(key[3]),
                    "reasons": reasons,
                    "current": row,
                    "baseline": base,
                }
            )
    return regressions


def _parse_int_list(raw: str | None, default: Sequence[int]) -> list[int]:
    if raw is None:
        return [int(v) for v in default]
    vals = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    out = sorted(set(v for v in vals if v > 0))
    if not out:
        raise ValueError("Expected at least one positive budget")
    return out


def _parse_modes(raw: str | None, default: Sequence[str]) -> list[str]:
    if raw is None:
        return [str(v) for v in default]
    vals = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for val in vals:
        if val in {"refine_off", "refine_on"} and val not in seen:
            out.append(val)
            seen.add(val)
    if not out:
        raise ValueError("Expected at least one mode from refine_off,refine_on")
    return out


def _parse_profiles(raw: str | None, default: Sequence[str]) -> list[str]:
    if raw is None:
        vals = [str(v) for v in default]
    else:
        vals = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for val in vals:
        if val not in seen:
            out.append(val)
            seen.add(val)
    if not out:
        raise ValueError("Expected at least one profile")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed small oracle regression suite runner")
    p.add_argument("--suite_manifest", type=str, default=str(DEFAULT_SUITE_MANIFEST))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--budgets", type=str, default=None)
    p.add_argument("--modes", type=str, default=None)
    p.add_argument("--profiles", type=str, default=None)
    p.add_argument("--n_repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default=None)
    p.add_argument("--ignore_dims", action="store_true", default=None)
    p.add_argument("--success_mse", type=float, default=None)
    p.add_argument("--quiet", action="store_true", default=None)
    p.add_argument("--fast_benchmark", action="store_true", default=None)
    p.add_argument("--save_individual_reports", action="store_true")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--regression_mse_factor", type=float, default=1.5)
    p.add_argument("--regression_time_factor", type=float, default=2.0)
    p.add_argument("--fail_on_regression", action="store_true")
    p.add_argument("--discovery_enable", action="store_true")
    p.add_argument("--discovery_committee_topk", type=int, default=8)
    p.add_argument("--discovery_max_members", type=int, default=None)
    p.add_argument("--discovery_experiment_manifest", type=str, default=None)
    p.add_argument("--discovery_witness_capture_enable", action="store_true")
    p.add_argument("--discovery_witness_hessian_diag_enable", action="store_true")
    p.add_argument(
        "--discovery_diagnostic_set",
        type=str,
        default="basic",
        choices=["basic", "extended", "physics"],
    )
    p.add_argument("--discovery_beta", type=float, default=0.0)
    p.add_argument("--discovery_gamma", type=float, default=0.0)
    p.add_argument(
        "--discovery_disagreement_mode",
        type=str,
        default="auto",
        choices=["auto", "witness"],
    )
    p.add_argument("--discovery_lambda_cost", type=float, default=1.0)
    p.add_argument("--discovery_lambda_noise", type=float, default=1.0)
    p.add_argument("--discovery_lambda_feasibility", type=float, default=1.0)

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
    manifest_path, manifest = load_regression_suite(args.suite_manifest)
    spec_paths = resolve_suite_spec_paths(manifest, manifest_path=manifest_path)
    defaults = dict(manifest.get("defaults") or {})
    profile_overrides = {
        str(k): dict(v or {}) for k, v in dict(manifest.get("profile_overrides") or {}).items()
    }

    budgets = _parse_int_list(args.budgets, defaults.get("budgets", [100]))
    modes = _parse_modes(args.modes, defaults.get("modes", ["refine_off"]))
    profiles = _parse_profiles(args.profiles, defaults.get("profiles", ["current"]))
    n_repeats = int(args.n_repeats if args.n_repeats is not None else defaults.get("n_repeats", 1))
    seed = int(args.seed if args.seed is not None else defaults.get("seed", 0))
    dtype_name = str(args.dtype or defaults.get("dtype", "float64")).lower()
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    ignore_dims = bool(args.ignore_dims if args.ignore_dims is not None else defaults.get("ignore_dims", False))
    quiet = bool(args.quiet if args.quiet is not None else defaults.get("quiet", True))
    success_mse = float(args.success_mse if args.success_mse is not None else defaults.get("success_mse", 1.0e-6))
    fast_benchmark = bool(args.fast_benchmark if args.fast_benchmark is not None else defaults.get("fast_benchmark", False))
    if args.wall_time_limit_s is None and defaults.get("wall_time_limit_s", None) is not None:
        args.wall_time_limit_s = float(defaults.get("wall_time_limit_s"))
    if fast_benchmark:
        args.no_brute_force = True
    if args.n_seeds is None and defaults.get("n_seeds", None) is not None:
        args.n_seeds = int(defaults.get("n_seeds"))
    if args.split_iter_across_seeds is None and defaults.get("split_iter_across_seeds", None) is not None:
        args.split_iter_across_seeds = bool(defaults.get("split_iter_across_seeds"))

    suite_id = str(manifest.get("suite_id", "regression_suite") or "regression_suite")
    output_dir = pathlib.Path(args.output_dir or (REPO_ROOT / "results" / f"oracle_regression_{suite_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = run_oracle_suite(
        spec_paths,
        budgets=budgets,
        modes=modes,
        profiles=profiles,
        profile_overrides=profile_overrides,
        n_repeats=n_repeats,
        seed=seed,
        dtype=dtype,
        enforce_dims=not ignore_dims,
        success_mse_threshold=success_mse,
        verbose=not quiet,
        hp_overrides=args,
        output_dir=output_dir,
        save_individual_reports=bool(args.save_individual_reports or args.discovery_enable),
        jobs=int(args.jobs),
    )

    spec_summary = aggregate_rows_by_spec(payload.get("rows", []))
    regression_payload: dict[str, Any] = {
        "suite_id": suite_id,
        "suite_manifest": str(manifest_path),
        "n_specs": int(len(spec_paths)),
        "budgets": [int(v) for v in budgets],
        "profiles": [str(v) for v in profiles],
        "modes": [str(v) for v in modes],
        "n_repeats": int(n_repeats),
        "rows": payload.get("rows", []),
        "spec_summary": spec_summary,
    }

    if bool(args.discovery_enable):
        discovery_payload = run_oracle_discovery_benchmark(
            regression_payload,
            output_dir=output_dir,
            committee_topk=max(1, int(args.discovery_committee_topk)),
            max_members=None if args.discovery_max_members is None else int(args.discovery_max_members),
            experiment_manifest_path=args.discovery_experiment_manifest,
            beta=float(args.discovery_beta),
            gamma=float(args.discovery_gamma),
            disagreement_mode=resolve_disagreement_mode(
                args.discovery_disagreement_mode,
                default_mode="witness",
            ),
            lambda_cost=float(args.discovery_lambda_cost),
            lambda_noise=float(args.discovery_lambda_noise),
            lambda_feasibility=float(args.discovery_lambda_feasibility),
            witness_capture_enable=bool(args.discovery_witness_capture_enable),
            witness_hessian_diag_enable=bool(args.discovery_witness_hessian_diag_enable),
            diagnostic_set=str(args.discovery_diagnostic_set or "basic"),
            dtype=dtype,
        )
        discovery_path = output_dir / "oracle_discovery_results.json"
        _write_json(discovery_payload, discovery_path)
        regression_payload["discovery_results_path"] = str(discovery_path)
        regression_payload["discovery_enabled"] = True
    else:
        regression_payload["discovery_enabled"] = False

    _write_json(regression_payload, output_dir / "oracle_regression_results.json")
    _write_csv(spec_summary, output_dir / "oracle_regression_spec_summary.csv")

    regressions: list[dict[str, Any]] = []
    if args.baseline:
        baseline_path = pathlib.Path(args.baseline)
        baseline_payload = _load_json(baseline_path)
        baseline_summary = list(baseline_payload.get("spec_summary") or [])
        regressions = compare_spec_summaries(
            spec_summary,
            baseline_summary,
            mse_factor=float(args.regression_mse_factor),
            time_factor=float(args.regression_time_factor),
        )
        compare_payload = {
            "baseline": str(baseline_path),
            "regression_mse_factor": float(args.regression_mse_factor),
            "regression_time_factor": float(args.regression_time_factor),
            "n_regressions": int(len(regressions)),
            "regressions": regressions,
        }
        _write_json(compare_payload, output_dir / "oracle_regression_compare.json")

    print(f"[regression] suite={suite_id} specs={len(spec_paths)} budgets={budgets} modes={modes}")
    print(f"[regression] outputs written to {output_dir}")
    if regressions:
        print(f"[regression] regressions={len(regressions)}")
        for row in regressions:
            print(
                f"[regression] {row['spec_id']} mode={row['mode']} budget={row['budget']} :: "
                + "; ".join(str(v) for v in row.get("reasons", []))
            )
    else:
        print("[regression] no regressions flagged")
    if regressions and args.fail_on_regression:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
