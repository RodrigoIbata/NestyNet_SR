# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .engine.search import run_explorer_core
from .explorer import make_engine_runtime_hooks, node_str
from .oracle_lab import build_oracle_dataset, compile_target_expression, load_equation_spec


DEFAULT_SPEC_PATHS = (
    "examples/oracle_factorized_search/specs/feynman_037.json",
    "examples/oracle_factorized_search/specs/feynman_090.json",
    "examples/oracle_factorized_search/specs/trig_constant_demo.json",
)

DEFAULT_ARM_ORDER = (
    "baseline_grouped",
    "opportunity_only",
    "credible_only",
    "full_enabled",
)

DEFAULT_ARM_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline_grouped": {
        "repair_controller_credible_route_enable": False,
        "repair_opportunity_controller_enable": False,
        "repair_opportunity_controller_path": "",
    },
    "opportunity_only": {
        "repair_controller_credible_route_enable": False,
        "repair_opportunity_controller_enable": True,
        "repair_opportunity_controller_path": "results/opportunity_benchmark/pr5_opportunity_controller.pt",
    },
    "credible_only": {
        "repair_controller_credible_route_enable": True,
        "repair_opportunity_controller_enable": False,
        "repair_opportunity_controller_path": "results/opportunity_benchmark/pr5_opportunity_controller.pt",
    },
    "full_enabled": {
        "repair_controller_credible_route_enable": True,
        "repair_opportunity_controller_enable": True,
        "repair_opportunity_controller_path": "results/opportunity_benchmark/pr5_opportunity_controller.pt",
    },
}

DEFAULT_COMMON_KWARGS: dict[str, Any] = {
    "poly_degree": 4,
    "brute_depth": 0,
    "early_stop_mse": 1.0e-10,
    "inverse_steering_enable": True,
    "inverse_experiment_log_enable": True,
    "repair_controller_enable": True,
    "repair_controller_min_score": 0.15,
    "repair_controller_critic_enable": True,
    "repair_controller_critic_path": "results/repair_tuple_phase4_grouped/repair_tuple_ranker_grouped.pt",
    "repair_controller_critic_mode": "priority",
    "repair_controller_route_compare_enable": True,
    "repair_controller_route_compare_path": "results/repair_build_route_phase4_groupedbuild/repair_route_compare_learned.pt",
    "repair_controller_route_compare_repair_tuple_path": "results/repair_tuple_phase4_grouped/repair_tuple_ranker_grouped.pt",
    "repair_controller_route_compare_build_tuple_path": "results/build_tuple_phase2_grouped/build_tuple_ranker_grouped.pt",
    "repair_controller_route_compare_max_repair_prob": 0.35,
    "repair_controller_route_compare_min_build_margin": 0.05,
    "repair_controller_max_setup_steps": 0,
    "refine_enable": False,
    "print_every": 0,
    "verbose": False,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return str(value)


def _float_list(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        try:
            out.append(float(row.get(key, 0.0) or 0.0))
        except Exception:
            continue
    return out


def _stats(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(sum(vals) / len(vals)),
        "median": float(statistics.median(vals)),
        "max": float(max(vals)),
    }


def _selected_excerpt(rows: Sequence[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    selected_rows = [row for row in rows if bool(row.get("controller_selected", False))]
    for row in selected_rows[: max(0, int(limit))]:
        out.append({
            "status": str(row.get("status", "") or ""),
            "controller_score": float(row.get("controller_score", 0.0) or 0.0),
            "route_best_route": str(row.get("controller_route_compare_best_route", "") or ""),
            "route_source": str(row.get("controller_route_compare_source", "") or ""),
            "route_preview_source": str(row.get("controller_route_compare_preview_source", "") or ""),
            "credible_used": bool(row.get("controller_route_compare_credible_used", False)),
            "repair_unseen_upside": float(row.get("controller_route_compare_repair_unseen_upside", 0.0) or 0.0),
            "repair_credible_score": _jsonable(row.get("controller_route_compare_repair_credible_score", None)),
            "build_credible_score": _jsonable(row.get("controller_route_compare_build_credible_score", None)),
            "veto_repair": bool(row.get("controller_route_compare_veto_repair", False)),
            "repair_option_status": str(row.get("repair_option_status", "") or ""),
            "repair_option_continue_source": list(row.get("repair_option_continue_source", []) or []),
        })
    return out


def _summarize_inverse_log(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    row_list = list(rows or [])
    selected_rows = [row for row in row_list if bool(row.get("controller_selected", False))]
    return {
        "n_rows": int(len(row_list)),
        "n_selected_rows": int(len(selected_rows)),
        "status_counts": dict(sorted(Counter(str(row.get("status", "") or "") for row in row_list).items())),
        "controller_score_source_counts": dict(sorted(Counter(str(row.get("controller_score_source", "") or "") for row in row_list).items())),
        "route_best_route_counts": dict(sorted(Counter(str(row.get("controller_route_compare_best_route", "") or "") for row in row_list if row.get("controller_route_compare_best_route", None) is not None).items())),
        "route_source_counts": dict(sorted(Counter(str(row.get("controller_route_compare_source", "") or "") for row in row_list if row.get("controller_route_compare_source", None) is not None).items())),
        "route_compare_preview_source_counts": dict(sorted(Counter(str(row.get("controller_route_compare_preview_source", "") or "") for row in row_list if row.get("controller_route_compare_preview_source", None) is not None).items())),
        "route_veto_count": int(sum(1 for row in row_list if bool(row.get("controller_route_compare_veto_repair", False)))),
        "credible_used_count": int(sum(1 for row in row_list if bool(row.get("controller_route_compare_credible_used", False)))),
        "repair_unseen_upside_stats": _stats(_float_list(row_list, "controller_route_compare_repair_unseen_upside")),
        "repair_unseen_acquisition_stats": _stats(_float_list(row_list, "controller_route_compare_repair_unseen_best_acquisition_estimate")),
        "selected_status_counts": dict(sorted(Counter(str(row.get("status", "") or "") for row in selected_rows).items())),
        "selected_route_best_route_counts": dict(sorted(Counter(str(row.get("controller_route_compare_best_route", "") or "") for row in selected_rows if row.get("controller_route_compare_best_route", None) is not None).items())),
    }


def _case_key(*, spec_id: str, seed: int, arm: str, n_iter: int) -> str:
    return f"{spec_id}__seed{int(seed):04d}__iter{int(n_iter):04d}__{arm}"


def _case_output_path(output_dir: Path, *, spec_id: str, seed: int, arm: str, n_iter: int) -> Path:
    return output_dir / "cases" / f"{_case_key(spec_id=spec_id, seed=seed, arm=arm, n_iter=n_iter)}.json"


def _load_existing_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    runs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        runs.append(json.loads(text))
    return runs


def _append_run(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(row), sort_keys=True))
        fh.write("\n")


def _build_common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    out = dict(DEFAULT_COMMON_KWARGS)
    out["poly_degree"] = int(args.poly_degree)
    out["brute_depth"] = int(args.brute_depth)
    out["refine_enable"] = bool(args.refine_enable)
    out["repair_controller_critic_path"] = str(args.repair_critic_path)
    out["repair_controller_route_compare_path"] = str(args.route_compare_path)
    out["repair_controller_route_compare_repair_tuple_path"] = str(args.route_compare_repair_tuple_path)
    out["repair_controller_route_compare_build_tuple_path"] = str(args.route_compare_build_tuple_path)
    out["repair_controller_route_compare_max_repair_prob"] = float(args.route_compare_max_repair_prob)
    out["repair_controller_route_compare_min_build_margin"] = float(args.route_compare_min_build_margin)
    out["repair_controller_max_setup_steps"] = int(args.repair_controller_max_setup_steps)
    return out


def run_single_case(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = Path(args.spec_path)
    spec = load_equation_spec(spec_path)
    target_fn = compile_target_expression(spec)
    ds = build_oracle_dataset(
        spec,
        target_fn,
        n_fit=int(args.n_fit),
        n_probe=int(args.n_probe),
        seed=int(args.seed),
        dtype=torch.float64,
    )
    arm = str(args.arm)
    if arm not in DEFAULT_ARM_OVERRIDES:
        raise KeyError(f"Unknown arm: {arm}")

    kwargs = _build_common_kwargs(args)
    kwargs.update(DEFAULT_ARM_OVERRIDES[arm])
    kwargs.update({
        "target_fn": target_fn,
        "nvars": int(ds["x_fit"].shape[1]),
        "n_iter": int(args.n_iter),
        "max_depth": int(args.max_depth),
        "lo": 0.0,
        "hi": 1.0,
        "seed": int(args.seed),
        "seed_search": int(args.seed),
        "dtype": torch.float64,
        "var_dims": ds["var_dims"],
        "y_dims": ds["y_dims"],
        "x_fit_data": ds["x_fit"],
        "y_fit_data": ds["y_fit"],
        "x_probe_data": ds["x_probe"],
        "y_probe_data": ds["y_probe"],
    })
    kwargs.setdefault("_runtime_hooks", make_engine_runtime_hooks())

    started = time.perf_counter()
    if bool(args.suppress_child_stdout):
        with contextlib.redirect_stdout(io.StringIO()):
            arch = run_explorer_core(**kwargs)
    else:
        arch = run_explorer_core(**kwargs)
    elapsed = float(max(0.0, time.perf_counter() - started))

    best_rows = list(arch.best(1))
    best_rec = best_rows[0] if best_rows else None
    best_mse = float(getattr(best_rec, "best_mse", float("inf")))
    best_expr = ""
    if best_rec is not None and getattr(best_rec, "best_expr", None) is not None:
        best_expr = node_str(best_rec.best_expr)

    action_dist = getattr(arch, "action_distribution", {}) or {}
    action_counts = {
        str(k): int(v)
        for k, v in dict(action_dist.get("counts", {}) or {}).items()
    }
    repair_stats = dict(getattr(arch, "repair_controller_stats", {}) or {})
    inverse_rows = list(getattr(arch, "inverse_experiment_log", []) or [])
    inverse_summary = _summarize_inverse_log(inverse_rows)

    return {
        "status": "ok",
        "spec_id": str(spec.id),
        "spec_path": str(spec_path),
        "seed": int(args.seed),
        "arm": arm,
        "n_iter": int(args.n_iter),
        "max_depth": int(args.max_depth),
        "elapsed_s": elapsed,
        "best_mse": best_mse,
        "best_expr": best_expr,
        "action_counts": action_counts,
        "repair_controller_stats": _jsonable(repair_stats),
        "inverse_log_summary": inverse_summary,
        "selected_rows_excerpt": _selected_excerpt(inverse_rows),
        "repair_selected": int(repair_stats.get("selected", 0) or 0),
        "repair_option_selected": int(repair_stats.get("option_repair_selected", 0) or 0),
        "repair_considered": int(repair_stats.get("considered", 0) or 0),
        "route_compare_vetoed": int(inverse_summary.get("route_veto_count", 0)),
        "credible_used_count": int(inverse_summary.get("credible_used_count", 0)),
        "opportunity_controller_loaded": bool(repair_stats.get("opportunity_controller_loaded", False)),
        "route_compare_loaded": bool(repair_stats.get("route_compare_loaded", False)),
    }


def _group_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        if str(row.get("status", "")) != "ok":
            continue
        key = (
            int(row.get("n_iter", 0) or 0),
            str(row.get("spec_id", "") or ""),
            str(row.get("arm", "") or ""),
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        n_iter, spec_id, arm = key
        rows = grouped[key]
        best_mse_vals = [float(row["best_mse"]) for row in rows]
        elapsed_vals = [float(row["elapsed_s"]) for row in rows]
        veto_vals = [int(row.get("route_compare_vetoed", 0) or 0) for row in rows]
        credible_vals = [int(row.get("credible_used_count", 0) or 0) for row in rows]
        out.append({
            "n_iter": int(n_iter),
            "spec_id": spec_id,
            "arm": arm,
            "count": int(len(rows)),
            "median_best_mse": float(statistics.median(best_mse_vals)),
            "mean_best_mse": float(sum(best_mse_vals) / len(best_mse_vals)),
            "mean_elapsed_s": float(sum(elapsed_vals) / len(elapsed_vals)),
            "mean_route_veto_count": float(sum(veto_vals) / len(veto_vals)),
            "mean_credible_used_count": float(sum(credible_vals) / len(credible_vals)),
        })
    return out


def _pairwise_vs_baseline(
    runs: Sequence[dict[str, Any]],
    *,
    baseline_arm: str,
    pairwise_eps: float,
) -> list[dict[str, Any]]:
    by_case: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        if str(row.get("status", "")) != "ok":
            continue
        key = (
            int(row.get("n_iter", 0) or 0),
            str(row.get("spec_id", "") or ""),
            int(row.get("seed", 0) or 0),
        )
        by_case[key][str(row.get("arm", "") or "")] = row

    tallies: dict[tuple[int, str, str], dict[str, int]] = defaultdict(lambda: {"better": 0, "worse": 0, "tie": 0, "count": 0})
    for key, arm_rows in by_case.items():
        baseline_row = arm_rows.get(str(baseline_arm))
        if baseline_row is None:
            continue
        baseline_mse = float(baseline_row.get("best_mse", float("inf")))
        n_iter, spec_id, _seed = key
        for arm, row in arm_rows.items():
            if arm == str(baseline_arm):
                continue
            mse = float(row.get("best_mse", float("inf")))
            tally = tallies[(int(n_iter), str(spec_id), str(arm))]
            tally["count"] += 1
            if mse < baseline_mse - float(pairwise_eps):
                tally["better"] += 1
            elif mse > baseline_mse + float(pairwise_eps):
                tally["worse"] += 1
            else:
                tally["tie"] += 1

    out: list[dict[str, Any]] = []
    for key in sorted(tallies.keys()):
        n_iter, spec_id, arm = key
        tally = tallies[key]
        out.append({
            "n_iter": int(n_iter),
            "spec_id": spec_id,
            "arm": arm,
            "baseline_arm": str(baseline_arm),
            "count": int(tally["count"]),
            "better": int(tally["better"]),
            "worse": int(tally["worse"]),
            "tie": int(tally["tie"]),
        })
    return out


def _write_summary(
    *,
    output_path: Path,
    config: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    baseline_arm: str,
    pairwise_eps: float,
) -> None:
    payload = {
        "config": _jsonable(config),
        "runs": _jsonable(list(runs)),
        "group_summaries": _group_runs(runs),
        "pairwise_vs_baseline": _pairwise_vs_baseline(
            runs,
            baseline_arm=baseline_arm,
            pairwise_eps=pairwise_eps,
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _case_args_for_subprocess(
    args: argparse.Namespace,
    *,
    spec_path: str,
    arm: str,
    seed: int,
    n_iter: int,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "nestynet_sr.sr_search.factorized_search.opportunity_benchmark",
        "run-case",
        "--spec-path",
        str(spec_path),
        "--arm",
        str(arm),
        "--seed",
        str(int(seed)),
        "--n-iter",
        str(int(n_iter)),
        "--max-depth",
        str(int(args.max_depth)),
        "--brute-depth",
        str(int(args.brute_depth)),
        "--poly-degree",
        str(int(args.poly_degree)),
        "--n-fit",
        str(int(args.n_fit)),
        "--n-probe",
        str(int(args.n_probe)),
        "--repair-critic-path",
        str(args.repair_critic_path),
        "--route-compare-path",
        str(args.route_compare_path),
        "--route-compare-repair-tuple-path",
        str(args.route_compare_repair_tuple_path),
        "--route-compare-build-tuple-path",
        str(args.route_compare_build_tuple_path),
        "--route-compare-max-repair-prob",
        str(float(args.route_compare_max_repair_prob)),
        "--route-compare-min-build-margin",
        str(float(args.route_compare_min_build_margin)),
        "--repair-controller-max-setup-steps",
        str(int(args.repair_controller_max_setup_steps)),
        "--output",
        str(output_path),
        "--suppress-child-stdout",
    ] + (["--plus-enable"] if bool(args.refine_enable) else [])


def run_sweep(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cases").mkdir(parents=True, exist_ok=True)

    cases_jsonl = output_dir / "runs.jsonl"
    summary_json = output_dir / "summary.json"
    config_json = output_dir / "config.json"

    spec_paths = [str(Path(p)) for p in args.spec_paths]
    arms = [str(arm) for arm in args.arms]
    n_iter_values = [int(v) for v in args.n_iter_values]
    seeds = [int(v) for v in args.seeds]

    config = {
        "spec_paths": spec_paths,
        "arms": arms,
        "seeds": seeds,
        "n_iter_values": n_iter_values,
        "max_depth": int(args.max_depth),
        "brute_depth": int(args.brute_depth),
        "poly_degree": int(args.poly_degree),
        "n_fit": int(args.n_fit),
        "n_probe": int(args.n_probe),
        "refine_enable": bool(args.refine_enable),
        "repair_critic_path": str(args.repair_critic_path),
        "route_compare_path": str(args.route_compare_path),
        "route_compare_repair_tuple_path": str(args.route_compare_repair_tuple_path),
        "route_compare_build_tuple_path": str(args.route_compare_build_tuple_path),
        "route_compare_max_repair_prob": float(args.route_compare_max_repair_prob),
        "route_compare_min_build_margin": float(args.route_compare_min_build_margin),
        "repair_controller_max_setup_steps": int(args.repair_controller_max_setup_steps),
        "baseline_arm": str(args.baseline_arm),
        "pairwise_eps": float(args.pairwise_eps),
        "isolate_processes": bool(args.isolate_processes),
        "continue_on_error": bool(args.continue_on_error),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_json.write_text(json.dumps(_jsonable(config), indent=2), encoding="utf-8")

    existing_runs = _load_existing_runs(cases_jsonl) if bool(args.resume) else []
    completed_keys = {
        _case_key(
            spec_id=str(row.get("spec_id", "") or ""),
            seed=int(row.get("seed", 0) or 0),
            arm=str(row.get("arm", "") or ""),
            n_iter=int(row.get("n_iter", 0) or 0),
        )
        for row in existing_runs
    }
    runs = list(existing_runs)

    total_cases = len(spec_paths) * len(arms) * len(seeds) * len(n_iter_values)
    completed_count = int(len(completed_keys))
    print(
        f"[opportunity_benchmark] output_dir={output_dir} total_cases={total_cases} "
        f"completed_resume={completed_count}",
        flush=True,
    )

    for n_iter in n_iter_values:
        for spec_path in spec_paths:
            spec_id = load_equation_spec(spec_path).id
            for seed in seeds:
                for arm in arms:
                    case_key = _case_key(spec_id=spec_id, seed=seed, arm=arm, n_iter=n_iter)
                    if case_key in completed_keys:
                        continue
                    case_output = _case_output_path(output_dir, spec_id=spec_id, seed=seed, arm=arm, n_iter=n_iter)
                    started = time.perf_counter()
                    print(
                        f"[opportunity_benchmark] start case={case_key} "
                        f"progress={completed_count}/{total_cases}",
                        flush=True,
                    )
                    try:
                        if bool(args.isolate_processes):
                            cmd = _case_args_for_subprocess(
                                args,
                                spec_path=spec_path,
                                arm=arm,
                                seed=seed,
                                n_iter=n_iter,
                                output_path=case_output,
                            )
                            env = dict(os.environ)
                            env.setdefault("OMP_NUM_THREADS", "1")
                            env.setdefault("MKL_NUM_THREADS", "1")
                            proc = subprocess.run(
                                cmd,
                                cwd=str(repo_root),
                                env=env,
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                            if not case_output.exists():
                                raise FileNotFoundError(str(case_output))
                            row = json.loads(case_output.read_text(encoding="utf-8"))
                            row["subprocess_stdout_tail"] = proc.stdout[-4000:]
                            row["subprocess_stderr_tail"] = proc.stderr[-4000:]
                        else:
                            ns = argparse.Namespace(**vars(args))
                            ns.spec_path = str(spec_path)
                            ns.arm = str(arm)
                            ns.seed = int(seed)
                            ns.n_iter = int(n_iter)
                            row = run_single_case(ns)
                            case_output.write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
                    except Exception as exc:
                        row = {
                            "status": "error",
                            "spec_id": str(spec_id),
                            "spec_path": str(spec_path),
                            "seed": int(seed),
                            "arm": str(arm),
                            "n_iter": int(n_iter),
                            "max_depth": int(args.max_depth),
                            "elapsed_s": float(max(0.0, time.perf_counter() - started)),
                            "error": str(exc),
                        }
                        case_output.write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
                        print(
                            f"[opportunity_benchmark] error case={case_key} error={exc}",
                            flush=True,
                        )
                        if not bool(args.continue_on_error):
                            _append_run(cases_jsonl, row)
                            runs.append(row)
                            _write_summary(
                                output_path=summary_json,
                                config=config,
                                runs=runs,
                                baseline_arm=str(args.baseline_arm),
                                pairwise_eps=float(args.pairwise_eps),
                            )
                            return 1

                    _append_run(cases_jsonl, row)
                    runs.append(row)
                    completed_keys.add(case_key)
                    completed_count += 1
                    _write_summary(
                        output_path=summary_json,
                        config=config,
                        runs=runs,
                        baseline_arm=str(args.baseline_arm),
                        pairwise_eps=float(args.pairwise_eps),
                    )
                    status = str(row.get("status", ""))
                    if status == "ok":
                        print(
                            f"[opportunity_benchmark] done case={case_key} "
                            f"best_mse={float(row.get('best_mse', float('inf'))):.12f} "
                            f"elapsed_s={float(row.get('elapsed_s', 0.0)):.2f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[opportunity_benchmark] recorded error case={case_key}",
                            flush=True,
                        )

    print(
        f"[opportunity_benchmark] finished output_dir={output_dir} total_cases={total_cases}",
        flush=True,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Opportunity-controller benchmark runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_case = argparse.ArgumentParser(add_help=False)
    common_case.add_argument("--spec-path", type=str, required=True)
    common_case.add_argument("--arm", type=str, choices=list(DEFAULT_ARM_OVERRIDES.keys()), required=True)
    common_case.add_argument("--seed", type=int, required=True)
    common_case.add_argument("--n-iter", type=int, required=True)
    common_case.add_argument("--max-depth", type=int, default=5)
    common_case.add_argument("--brute-depth", type=int, default=0)
    common_case.add_argument("--poly-degree", type=int, default=4)
    common_case.add_argument("--n-fit", type=int, default=512)
    common_case.add_argument("--n-probe", type=int, default=2048)
    common_case.add_argument("--plus-enable", action="store_true")
    common_case.add_argument("--repair-critic-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_critic_path"])
    common_case.add_argument("--route-compare-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_route_compare_path"])
    common_case.add_argument("--route-compare-repair-tuple-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_route_compare_repair_tuple_path"])
    common_case.add_argument("--route-compare-build-tuple-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_route_compare_build_tuple_path"])
    common_case.add_argument("--route-compare-max-repair-prob", type=float, default=float(DEFAULT_COMMON_KWARGS["repair_controller_route_compare_max_repair_prob"]))
    common_case.add_argument("--route-compare-min-build-margin", type=float, default=float(DEFAULT_COMMON_KWARGS["repair_controller_route_compare_min_build_margin"]))
    common_case.add_argument("--repair-controller-max-setup-steps", type=int, default=int(DEFAULT_COMMON_KWARGS["repair_controller_max_setup_steps"]))

    run_case = sub.add_parser("run-case", parents=[common_case])
    run_case.add_argument("--output", type=str, required=True)
    run_case.add_argument("--suppress-child-stdout", action="store_true")

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--spec-paths", nargs="+", default=list(DEFAULT_SPEC_PATHS))
    sweep.add_argument("--arms", nargs="+", default=list(DEFAULT_ARM_ORDER))
    sweep.add_argument("--seeds", nargs="+", type=int, required=True)
    sweep.add_argument("--n-iter-values", nargs="+", type=int, default=[120])
    sweep.add_argument("--max-depth", type=int, default=5)
    sweep.add_argument("--brute-depth", type=int, default=0)
    sweep.add_argument("--poly-degree", type=int, default=4)
    sweep.add_argument("--n-fit", type=int, default=512)
    sweep.add_argument("--n-probe", type=int, default=2048)
    sweep.add_argument("--plus-enable", action="store_true")
    sweep.add_argument("--repair-critic-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_critic_path"])
    sweep.add_argument("--route-compare-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_route_compare_path"])
    sweep.add_argument("--route-compare-repair-tuple-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_route_compare_repair_tuple_path"])
    sweep.add_argument("--route-compare-build-tuple-path", type=str, default=DEFAULT_COMMON_KWARGS["repair_controller_route_compare_build_tuple_path"])
    sweep.add_argument("--route-compare-max-repair-prob", type=float, default=float(DEFAULT_COMMON_KWARGS["repair_controller_route_compare_max_repair_prob"]))
    sweep.add_argument("--route-compare-min-build-margin", type=float, default=float(DEFAULT_COMMON_KWARGS["repair_controller_route_compare_min_build_margin"]))
    sweep.add_argument("--repair-controller-max-setup-steps", type=int, default=int(DEFAULT_COMMON_KWARGS["repair_controller_max_setup_steps"]))
    sweep.add_argument("--output-dir", type=str, required=True)
    sweep.add_argument("--baseline-arm", type=str, default="baseline_grouped")
    sweep.add_argument("--pairwise-eps", type=float, default=1.0e-12)
    sweep.add_argument("--resume", action="store_true")
    sweep.add_argument("--isolate-processes", action="store_true")
    sweep.add_argument("--continue-on-error", action="store_true")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run-case":
        row = run_single_case(args)
        Path(args.output).write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
        return 0
    if args.cmd == "sweep":
        return run_sweep(args)
    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
