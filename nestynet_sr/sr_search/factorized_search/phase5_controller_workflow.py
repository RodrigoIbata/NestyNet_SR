# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Sequence

import torch

from .config import FactorizedSearchConfig
from .oracle_lab import default_oracle_hyperparams
from .oracle_pretrain import run_oracle_pretrain_pipeline
from .oracle_suite import _resolve_spec_paths
from .stage1_benchmark_harness import (
    DEFAULT_SEEDS as DEFAULT_BENCHMARK_SEEDS,
    DEFAULT_TARGETS as DEFAULT_BENCHMARK_TARGETS,
    run_stage1_benchmark,
)


def _parse_int_csv(raw: str | None, *, default: Sequence[int]) -> list[int]:
    if raw is None:
        return [int(v) for v in default]
    out: list[int] = []
    for tok in str(raw).split(","):
        token = str(tok).strip()
        if token:
            out.append(int(token))
    return out or [int(v) for v in default]


def _parse_str_csv(raw: str | None, *, default: Sequence[str]) -> list[str]:
    if raw is None:
        return [str(v) for v in default]
    out = [str(tok).strip() for tok in str(raw).split(",") if str(tok).strip()]
    return out or [str(v) for v in default]


def _dtype_from_name(name: str) -> torch.dtype:
    token = str(name or "float64").strip().lower()
    if token in {"float32", "fp32", "f32"}:
        return torch.float32
    if token in {"float64", "fp64", "f64", "double"}:
        return torch.float64
    raise ValueError(f"unknown dtype: {name!r}")


def _write_json(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_mean(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _safe_median(values: Sequence[float]) -> float:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return float("nan")
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _safe_ratio(numer: float, denom: float) -> float | None:
    if not math.isfinite(float(numer)) or not math.isfinite(float(denom)):
        return None
    if abs(float(denom)) <= 1.0e-30:
        return None
    return float(float(numer) / float(denom))


def _make_oracle_hp(args: argparse.Namespace) -> FactorizedSearchConfig:
    hp = default_oracle_hyperparams()
    if args.oracle_max_depth is not None:
        hp.max_depth = int(args.oracle_max_depth)
    if args.oracle_poly_degree is not None:
        hp.poly_degree = int(args.oracle_poly_degree)
    if args.oracle_n_fit is not None:
        hp.n_fit = int(args.oracle_n_fit)
    if args.oracle_n_probe is not None:
        hp.n_probe = int(args.oracle_n_probe)
    if args.oracle_inverse_max_paths is not None:
        hp.inverse_max_paths = int(args.oracle_inverse_max_paths)
    if args.oracle_inverse_topk_terms is not None:
        hp.inverse_topk_terms = int(args.oracle_inverse_topk_terms)
    if args.oracle_inverse_shortlist_mult is not None:
        hp.inverse_shortlist_mult = int(args.oracle_inverse_shortlist_mult)
    return hp


def _is_phase5_baseline_arm(name: str) -> bool:
    return str(name).startswith("stage0_selective")


def _is_phase5_hybrid_arm(name: str) -> bool:
    return str(name).startswith("stage1_hybrid")


def _is_phase5_macro_arm(name: str) -> bool:
    return str(name).startswith("stage1_macro")


def _validate_phase5_arms(report: dict[str, Any]) -> list[str]:
    cfg = dict(report.get("config", {}) or {})
    arms = [str(name) for name in cfg.get("arms", [])]
    if len(arms) != 3:
        raise ValueError(f"phase-5 workflow requires exactly 3 scheduler arms, got {arms!r}")
    if sum(1 for name in arms if _is_phase5_baseline_arm(name)) != 1:
        raise ValueError(f"phase-5 workflow expected exactly one stage-0 baseline arm, got {arms!r}")
    if sum(1 for name in arms if _is_phase5_hybrid_arm(name)) != 1:
        raise ValueError(f"phase-5 workflow expected exactly one stage-1 hybrid arm, got {arms!r}")
    if sum(1 for name in arms if _is_phase5_macro_arm(name)) != 1:
        raise ValueError(f"phase-5 workflow expected exactly one stage-1 macro arm, got {arms!r}")
    return arms


def _phase5_benchmark_overview(report: dict[str, Any]) -> dict[str, Any]:
    arms = _validate_phase5_arms(report)
    cfg = dict(report.get("config", {}) or {})
    runs = [dict(row) for row in report.get("runs", []) if isinstance(row, dict)]
    solve_mse = float(cfg.get("solve_mse", 1.0e-10))

    arm_summary: dict[str, Any] = {}
    for arm in arms:
        subset = [row for row in runs if str(row.get("arm")) == arm]
        mses = [float(row.get("best_mse", float("nan"))) for row in subset]
        walls = [float(row.get("elapsed_s", float("nan"))) for row in subset]
        repairs = [float(row.get("repair_selected", float("nan"))) for row in subset]
        macros = [float(row.get("macro_selected", float("nan"))) for row in subset]
        solved = sum(1 for mse in mses if math.isfinite(mse) and mse <= solve_mse)
        arm_summary[arm] = {
            "n_runs": int(len(subset)),
            "solved": int(solved),
            "solve_rate": float((solved / len(subset)) if subset else float("nan")),
            "mean_mse": _safe_mean(mses),
            "median_mse": _safe_median(mses),
            "mean_wall_s": _safe_mean(walls),
            "mean_repair_selected": _safe_mean(repairs),
            "mean_macro_selected": _safe_mean(macros),
        }

    baseline_arm = next(arm for arm in arms if _is_phase5_baseline_arm(arm))
    baseline_stats = dict(arm_summary.get(baseline_arm, {}) or {})
    comparisons_vs_stage0: dict[str, Any] = {}
    for arm in arms:
        if arm == baseline_arm:
            continue
        cur = dict(arm_summary.get(arm, {}) or {})
        comparisons_vs_stage0[arm] = {
            "solve_rate_delta": (
                float(cur["solve_rate"] - baseline_stats["solve_rate"])
                if math.isfinite(float(cur.get("solve_rate", float("nan"))))
                and math.isfinite(float(baseline_stats.get("solve_rate", float("nan"))))
                else float("nan")
            ),
            "mean_wall_s_delta": (
                float(cur["mean_wall_s"] - baseline_stats["mean_wall_s"])
                if math.isfinite(float(cur.get("mean_wall_s", float("nan"))))
                and math.isfinite(float(baseline_stats.get("mean_wall_s", float("nan"))))
                else float("nan")
            ),
            "mean_mse_ratio": _safe_ratio(
                float(cur.get("mean_mse", float("nan"))),
                float(baseline_stats.get("mean_mse", float("nan"))),
            ),
            "identical_vs_stage0": dict(report.get("identical_vs_stage0", {}).get(arm, {}) or {}),
        }

    return {
        "arms": arms,
        "baseline_arm": baseline_arm,
        "arm_summary": arm_summary,
        "comparisons_vs_stage0": comparisons_vs_stage0,
    }


def run_phase5_controller_workflow(
    *,
    output_dir: str | pathlib.Path,
    specs: Sequence[str | pathlib.Path] | None = None,
    spec_glob: str | None = None,
    critic_path: str | pathlib.Path | None = None,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    oracle_seeds: Sequence[int] = (0,),
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    pretrain_depth_min: int = 3,
    pretrain_depth_max: int = 8,
    pretrain_compare_modes: Sequence[str] = ("identity", "full", "affine"),
    pretrain_topk: int = 8,
    pretrain_max_corrupt_paths_per_spec: int | None = None,
    pretrain_sweep_all_paths: bool = False,
    pretrain_sweep_max_paths: int | None = None,
    hidden_dim: int = 32,
    pretrain_epochs: int = 200,
    pretrain_lr: float = 1.0e-2,
    pretrain_weight_decay: float = 1.0e-4,
    pretrain_val_fraction: float = 0.2,
    pretrain_seed: int = 0,
    aux_report_paths: Sequence[str | pathlib.Path] = (),
    aux_hidden_dim: int | None = None,
    aux_epochs: int = 250,
    aux_lr: float = 1.0e-2,
    aux_weight_decay: float = 1.0e-4,
    aux_val_fraction: float = 0.2,
    aux_seed: int = 0,
    continue_on_error: bool = True,
    verbose: bool = True,
    benchmark_targets: Sequence[str] = DEFAULT_BENCHMARK_TARGETS,
    benchmark_seeds: Sequence[int] = DEFAULT_BENCHMARK_SEEDS,
    benchmark_blend: float = 0.50,
    benchmark_macro_profile: str = "default",
    benchmark_macro_controller_learned_policy_weight: float | None = None,
    benchmark_macro_controller_learned_route_weight: float | None = None,
    benchmark_macro_controller_learned_q_weight: float | None = None,
    benchmark_n_iter: int = 180,
    benchmark_max_depth: int = 5,
    benchmark_brute_depth: int = 0,
    benchmark_n_fit: int = 128,
    benchmark_n_probe: int = 512,
    benchmark_refine_enable: bool = True,
    benchmark_solve_mse: float = 1.0e-10,
    benchmark_capture_search_output: bool = True,
    benchmark_threads: int | None = 1,
) -> dict[str, Any]:
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrain_summary: dict[str, Any] | None = None
    resolved_critic_path: str
    if critic_path is None:
        spec_paths = _resolve_spec_paths(
            [str(path) for path in list(specs or [])],
            None if spec_glob is None else str(spec_glob),
        )
        pretrain_dir = out_dir / "oracle_pretrain"
        pretrain_summary = run_oracle_pretrain_pipeline(
            spec_paths,
            output_dir=pretrain_dir,
            factorized_search_hp=factorized_search_hp,
            seeds=tuple(int(seed) for seed in oracle_seeds),
            dtype=dtype,
            enforce_dims=bool(enforce_dims),
            depth_min=int(pretrain_depth_min),
            depth_max=int(pretrain_depth_max),
            compare_modes=tuple(str(mode) for mode in pretrain_compare_modes),
            topk=int(pretrain_topk),
            max_corrupt_paths_per_spec=(
                None
                if pretrain_max_corrupt_paths_per_spec is None
                else int(pretrain_max_corrupt_paths_per_spec)
            ),
            sweep_all_paths=bool(pretrain_sweep_all_paths),
            sweep_max_paths=None if pretrain_sweep_max_paths is None else int(pretrain_sweep_max_paths),
            hidden_dim=int(hidden_dim),
            pretrain_epochs=int(pretrain_epochs),
            pretrain_lr=float(pretrain_lr),
            pretrain_weight_decay=float(pretrain_weight_decay),
            pretrain_val_fraction=float(pretrain_val_fraction),
            pretrain_seed=int(pretrain_seed),
            aux_report_paths=[str(path) for path in aux_report_paths],
            aux_hidden_dim=None if aux_hidden_dim is None else int(aux_hidden_dim),
            aux_epochs=int(aux_epochs),
            aux_lr=float(aux_lr),
            aux_weight_decay=float(aux_weight_decay),
            aux_val_fraction=float(aux_val_fraction),
            aux_seed=int(aux_seed),
            continue_on_error=bool(continue_on_error),
            verbose=bool(verbose),
        )
        resolved_critic_path = str(pretrain_summary["final_bundle_path"])
    else:
        resolved_critic_path = str(critic_path)

    benchmark_report = run_stage1_benchmark(
        targets=tuple(str(target) for target in benchmark_targets),
        seeds=tuple(int(seed) for seed in benchmark_seeds),
        critic_path=resolved_critic_path,
        blends=[float(benchmark_blend)],
        arm_modes=("priority", "macro"),
        macro_profile=str(benchmark_macro_profile),
        macro_controller_learned_policy_weight=benchmark_macro_controller_learned_policy_weight,
        macro_controller_learned_route_weight=benchmark_macro_controller_learned_route_weight,
        macro_controller_learned_q_weight=benchmark_macro_controller_learned_q_weight,
        n_iter=int(benchmark_n_iter),
        max_depth=int(benchmark_max_depth),
        brute_depth=int(benchmark_brute_depth),
        n_fit=int(benchmark_n_fit),
        n_probe=int(benchmark_n_probe),
        refine_enable=bool(benchmark_refine_enable),
        solve_mse=float(benchmark_solve_mse),
        dtype=dtype,
        capture_search_output=bool(benchmark_capture_search_output),
        threads=None if benchmark_threads is None else int(benchmark_threads),
    )
    benchmark_overview = _phase5_benchmark_overview(benchmark_report)
    benchmark_report_path = out_dir / "stage1_three_arm_report.json"
    _write_json(benchmark_report, benchmark_report_path)

    summary = {
        "mode": "phase5_controller_workflow",
        "output_dir": str(out_dir),
        "critic_path": resolved_critic_path,
        "used_existing_critic_path": bool(critic_path is not None),
        "pretrain_summary": pretrain_summary,
        "benchmark_report_path": str(benchmark_report_path),
        "benchmark_overview": benchmark_overview,
    }
    _write_json(summary, out_dir / "phase5_controller_summary.json")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-5 offline controller workflow: oracle pretrain plus disciplined three-arm scheduler benchmark"
    )
    p.add_argument("--specs", nargs="*", default=None, help="Explicit oracle equation spec files")
    p.add_argument("--spec_glob", type=str, default=None, help="Glob for oracle equation specs")
    p.add_argument(
        "--critic_path",
        type=str,
        default="",
        help="Existing critic bundle path. When provided, skips oracle pretraining.",
    )
    p.add_argument("--output_dir", type=str, default="results/phase5_controller_workflow")
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    p.add_argument("--ignore_dims", action="store_true", help="Disable dimensional filtering during pretraining")
    p.add_argument("--oracle_seeds", type=str, default="0", help="Comma-separated oracle curriculum seeds")
    p.add_argument("--pretrain_depth_min", type=int, default=3)
    p.add_argument("--pretrain_depth_max", type=int, default=8)
    p.add_argument("--pretrain_compare_modes", type=str, default="identity,full,affine")
    p.add_argument("--pretrain_topk", type=int, default=8)
    p.add_argument("--pretrain_max_corrupt_paths_per_spec", type=int, default=None)
    p.add_argument("--pretrain_sweep_all_paths", action="store_true")
    p.add_argument("--pretrain_sweep_max_paths", type=int, default=None)
    p.add_argument("--oracle_max_depth", type=int, default=None)
    p.add_argument("--oracle_poly_degree", type=int, default=None)
    p.add_argument("--oracle_n_fit", type=int, default=None)
    p.add_argument("--oracle_n_probe", type=int, default=None)
    p.add_argument("--oracle_inverse_max_paths", type=int, default=None)
    p.add_argument("--oracle_inverse_topk_terms", type=int, default=None)
    p.add_argument("--oracle_inverse_shortlist_mult", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--pretrain_lr", type=float, default=1.0e-2)
    p.add_argument("--pretrain_weight_decay", type=float, default=1.0e-4)
    p.add_argument("--pretrain_val_fraction", type=float, default=0.2)
    p.add_argument("--pretrain_seed", type=int, default=0)
    p.add_argument("--aux_reports", nargs="*", default=[])
    p.add_argument("--aux_hidden_dim", type=int, default=None)
    p.add_argument("--aux_epochs", type=int, default=250)
    p.add_argument("--aux_lr", type=float, default=1.0e-2)
    p.add_argument("--aux_weight_decay", type=float, default=1.0e-4)
    p.add_argument("--aux_val_fraction", type=float, default=0.2)
    p.add_argument("--aux_seed", type=int, default=0)
    p.add_argument("--strict", action="store_true", help="Fail the pretrain stage on spec-level errors")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--benchmark_targets", nargs="*", default=list(DEFAULT_BENCHMARK_TARGETS))
    p.add_argument("--benchmark_seeds", nargs="*", type=int, default=list(DEFAULT_BENCHMARK_SEEDS))
    p.add_argument("--benchmark_blend", type=float, default=0.50)
    p.add_argument("--benchmark_macro_profile", choices=["default", "repair_probe"], default="default")
    p.add_argument("--benchmark_macro_controller_learned_policy_weight", type=float, default=None)
    p.add_argument("--benchmark_macro_controller_learned_route_weight", type=float, default=None)
    p.add_argument("--benchmark_macro_controller_learned_q_weight", type=float, default=None)
    p.add_argument("--benchmark_n_iter", type=int, default=180)
    p.add_argument("--benchmark_max_depth", type=int, default=5)
    p.add_argument("--benchmark_brute_depth", type=int, default=0)
    p.add_argument("--benchmark_n_fit", type=int, default=128)
    p.add_argument("--benchmark_n_probe", type=int, default=512)
    p.add_argument("--benchmark_solve_mse", type=float, default=1.0e-10)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no_plus", action="store_true")
    p.add_argument("--no_capture_search_output", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    critic_path = str(args.critic_path).strip() or None
    if critic_path is None and not args.specs and not args.spec_glob:
        raise SystemExit("phase5_controller_workflow requires --specs/--spec_glob unless --critic_path is supplied")

    factorized_search_hp = _make_oracle_hp(args)
    summary = run_phase5_controller_workflow(
        output_dir=str(args.output_dir),
        specs=args.specs,
        spec_glob=args.spec_glob,
        critic_path=critic_path,
        factorized_search_hp=factorized_search_hp,
        oracle_seeds=_parse_int_csv(args.oracle_seeds, default=(0,)),
        dtype=_dtype_from_name(args.dtype),
        enforce_dims=not bool(args.ignore_dims),
        pretrain_depth_min=int(args.pretrain_depth_min),
        pretrain_depth_max=int(args.pretrain_depth_max),
        pretrain_compare_modes=_parse_str_csv(
            args.pretrain_compare_modes,
            default=("identity", "full", "affine"),
        ),
        pretrain_topk=int(args.pretrain_topk),
        pretrain_max_corrupt_paths_per_spec=(
            None
            if args.pretrain_max_corrupt_paths_per_spec is None
            else int(args.pretrain_max_corrupt_paths_per_spec)
        ),
        pretrain_sweep_all_paths=bool(args.pretrain_sweep_all_paths),
        pretrain_sweep_max_paths=None if args.pretrain_sweep_max_paths is None else int(args.pretrain_sweep_max_paths),
        hidden_dim=int(args.hidden_dim),
        pretrain_epochs=int(args.pretrain_epochs),
        pretrain_lr=float(args.pretrain_lr),
        pretrain_weight_decay=float(args.pretrain_weight_decay),
        pretrain_val_fraction=float(args.pretrain_val_fraction),
        pretrain_seed=int(args.pretrain_seed),
        aux_report_paths=list(args.aux_reports or []),
        aux_hidden_dim=None if args.aux_hidden_dim is None else int(args.aux_hidden_dim),
        aux_epochs=int(args.aux_epochs),
        aux_lr=float(args.aux_lr),
        aux_weight_decay=float(args.aux_weight_decay),
        aux_val_fraction=float(args.aux_val_fraction),
        aux_seed=int(args.aux_seed),
        continue_on_error=not bool(args.strict),
        verbose=not bool(args.quiet),
        benchmark_targets=tuple(str(target) for target in args.benchmark_targets),
        benchmark_seeds=tuple(int(seed) for seed in args.benchmark_seeds),
        benchmark_blend=float(args.benchmark_blend),
        benchmark_macro_profile=str(args.benchmark_macro_profile),
        benchmark_macro_controller_learned_policy_weight=args.benchmark_macro_controller_learned_policy_weight,
        benchmark_macro_controller_learned_route_weight=args.benchmark_macro_controller_learned_route_weight,
        benchmark_macro_controller_learned_q_weight=args.benchmark_macro_controller_learned_q_weight,
        benchmark_n_iter=int(args.benchmark_n_iter),
        benchmark_max_depth=int(args.benchmark_max_depth),
        benchmark_brute_depth=int(args.benchmark_brute_depth),
        benchmark_n_fit=int(args.benchmark_n_fit),
        benchmark_n_probe=int(args.benchmark_n_probe),
        benchmark_refine_enable=not bool(args.no_plus),
        benchmark_solve_mse=float(args.benchmark_solve_mse),
        benchmark_capture_search_output=not bool(args.no_capture_search_output),
        benchmark_threads=None if args.threads is None else int(args.threads),
    )
    if bool(args.json):
        print(json.dumps(summary, indent=2))
    else:
        overview = dict(summary.get("benchmark_overview", {}) or {})
        print(
            f"[phase5] critic={summary['critic_path']} "
            f"arms={overview.get('arms', [])} "
            f"out={summary['output_dir']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
