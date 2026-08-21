# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from nestynet_sr.sr_search.factorized_search.controller_harness import (
    DEFAULT_SEEDS as CONTROLLER_DEFAULT_SEEDS,
    DEFAULT_TARGETS as CONTROLLER_DEFAULT_TARGETS,
    run_controller_benchmark,
)
from nestynet_sr.sr_search.factorized_search.scheduler_critic import load_scheduler_bundle
from nestynet_sr.sr_search.factorized_search.scheduler_dataset import load_scheduler_dataset_rows
from nestynet_sr.sr_search.factorized_search.scheduler_promotion import (
    DEFAULT_ELIGIBILITY_FLOOR,
    DEFAULT_PROMOTE_MAX_CALIBRATION_ERROR,
    DEFAULT_PROMOTE_MAX_INELIGIBLE_MEAN_BUDGET,
    DEFAULT_PROMOTE_MAX_ONLINE_WALL_RATIO,
    DEFAULT_PROMOTE_MIN_ELIGIBLE_UTILITY_LIFT,
    DEFAULT_PROMOTE_MIN_ONLINE_SOLVE_DELTA,
    recommend_scheduler_promotion,
)
from nestynet_sr.sr_search.factorized_search.scheduler_replay import replay_scheduler_decisions
from nestynet_sr.sr_search.factorized_search.scheduler_train import run_scheduler_training
from nestynet_sr.sr_search.factorized_search.stage1_benchmark_harness import (
    DEFAULT_BLENDS as STAGE1_DEFAULT_BLENDS,
    DEFAULT_SEEDS as STAGE1_DEFAULT_SEEDS,
    DEFAULT_TARGETS as STAGE1_DEFAULT_TARGETS,
    run_stage1_benchmark,
)


DEFAULT_WORKFLOW_COMPARISON_MODES: tuple[str, ...] = ("matched_exact", "matched_wall")
DEFAULT_STAGE1_WORKFLOW_ARM_MODES: tuple[str, ...] = ("scheduler_advisory", "scheduler_control")
VALID_COMPARISON_MODES = {"raw", "matched_exact", "matched_wall"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _dtype_from_name(name: str) -> torch.dtype:
    token = str(name or "float64").strip().lower()
    if token in {"float32", "f32", "fp32"}:
        return torch.float32
    if token in {"float64", "f64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"unknown dtype: {name!r}")


def _parse_weight_spec(spec: str, *, int_keys: bool = False) -> dict[Any, float]:
    token = str(spec or "").strip()
    if not token:
        return {}
    out: dict[Any, float] = {}
    for item in token.split(","):
        piece = str(item or "").strip()
        if not piece or "=" not in piece:
            continue
        key_raw, value_raw = piece.split("=", 1)
        key = int(key_raw.strip()) if int_keys else str(key_raw.strip())
        out[key] = float(value_raw.strip())
    return out


def _parse_alias_spec(spec: str) -> dict[str, str]:
    token = str(spec or "").strip()
    if not token:
        return {}
    out: dict[str, str] = {}
    for item in token.split(","):
        piece = str(item or "").strip()
        if not piece or "=" not in piece:
            continue
        key_raw, value_raw = piece.split("=", 1)
        key = str(key_raw.strip())
        value = str(value_raw.strip())
        if key and value:
            out[key] = value
    return out


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _normalize_comparison_modes(modes: Sequence[str] | None) -> tuple[str, ...]:
    if not modes:
        return DEFAULT_WORKFLOW_COMPARISON_MODES
    out: list[str] = []
    seen: set[str] = set()
    for raw in modes:
        mode = str(raw or "").strip().lower()
        if mode not in VALID_COMPARISON_MODES or mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    if not out:
        raise ValueError("no valid comparison modes were provided")
    return tuple(out)


def _normalize_strings(values: Sequence[Any] | None, fallback: Sequence[str]) -> list[str]:
    out = [str(v) for v in list(values or []) if str(v).strip()]
    return out if out else [str(v) for v in fallback]


def _normalize_ints(values: Sequence[Any] | None, fallback: Sequence[int]) -> list[int]:
    out = [int(v) for v in list(values or [])]
    return out if out else [int(v) for v in fallback]


def _resolve_output_dir(output_dir: str | None, *, train_scheduler_bundle: bool) -> Path | None:
    token = str(output_dir or "").strip()
    if token:
        out = Path(token)
        out.mkdir(parents=True, exist_ok=True)
        return out
    if not train_scheduler_bundle:
        return None
    out = Path(tempfile.mkdtemp(prefix="scheduler_workflow_"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_effective_bundle(
    *,
    scheduler_bundle_path: str,
    scheduler_dataset_paths: Sequence[str],
    train_scheduler_bundle: bool,
    output_dir: Path | None,
    init_bundle_path: str,
    budget_ladder: Sequence[int] | None,
    threshold_ladder: Sequence[float] | None,
    route_aliases: Mapping[str, str] | None,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
    ensemble_size: int,
    break_weight: float,
    tail_weight: float,
    route_win_weight: float,
    new_residual_basin_weight: float,
    fragile_weight: float,
    stable_weight: float,
    cost_weight: float,
    rank_weight: float,
    route_weights: Mapping[str, float] | None,
    budget_weight_map: Mapping[int, float] | None,
    deep_repair_min_depth: int,
    deep_repair_weight: float,
    hole_oversample_repeat: int,
    deep_repair_oversample_repeat: int,
    objective_mode: str | None,
    objective_hybrid_mix: float | None,
) -> tuple[str, dict[str, Any] | None]:
    base_bundle_path = str(scheduler_bundle_path or "").strip()
    dataset_paths = [str(path) for path in list(scheduler_dataset_paths or []) if str(path).strip()]
    if not bool(train_scheduler_bundle):
        if not base_bundle_path:
            raise ValueError("scheduler_bundle_path is required when train_scheduler_bundle is false")
        return base_bundle_path, None
    if not dataset_paths:
        raise ValueError("scheduler_dataset_paths are required when train_scheduler_bundle is true")
    if output_dir is None:
        raise ValueError("an output directory is required for training a derived scheduler bundle")
    derived_bundle_path = output_dir / "scheduler_bundle.pt"
    summary = run_scheduler_training(
        dataset_paths=list(dataset_paths),
        output_path=str(derived_bundle_path),
        init_bundle_path=str(init_bundle_path or base_bundle_path or ""),
        budget_ladder=None if budget_ladder is None else [int(v) for v in list(budget_ladder)],
        threshold_ladder=None if threshold_ladder is None else [float(v) for v in list(threshold_ladder)],
        route_aliases=dict(route_aliases or {}),
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_fraction=float(val_fraction),
        seed=int(seed),
        ensemble_size=int(ensemble_size),
        break_weight=float(break_weight),
        tail_weight=float(tail_weight),
        route_win_weight=float(route_win_weight),
        new_residual_basin_weight=float(new_residual_basin_weight),
        fragile_weight=float(fragile_weight),
        stable_weight=float(stable_weight),
        cost_weight=float(cost_weight),
        rank_weight=float(rank_weight),
        route_weights=None if route_weights is None else {str(k): float(v) for k, v in dict(route_weights).items()},
        budget_weight_map=None if budget_weight_map is None else {int(k): float(v) for k, v in dict(budget_weight_map).items()},
        deep_repair_min_depth=int(deep_repair_min_depth),
        deep_repair_weight=float(deep_repair_weight),
        hole_oversample_repeat=int(hole_oversample_repeat),
        deep_repair_oversample_repeat=int(deep_repair_oversample_repeat),
        objective_mode=objective_mode,
        objective_hybrid_mix=objective_hybrid_mix,
    )
    return str(derived_bundle_path), dict(summary or {})


def run_scheduler_workflow(
    *,
    scheduler_bundle_path: str = "",
    scheduler_dataset_paths: Sequence[str] = (),
    output_dir: str | None = None,
    comparison_modes: Sequence[str] = DEFAULT_WORKFLOW_COMPARISON_MODES,
    train_scheduler_bundle: bool = False,
    init_bundle_path: str = "",
    budget_ladder: Sequence[int] | None = None,
    threshold_ladder: Sequence[float] | None = None,
    route_aliases: Mapping[str, str] | None = None,
    hidden_dim: int = 64,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    training_seed: int = 0,
    ensemble_size: int = 4,
    break_weight: float = 1.0,
    tail_weight: float = 0.5,
    route_win_weight: float = 0.4,
    new_residual_basin_weight: float = 0.15,
    fragile_weight: float = 0.1,
    stable_weight: float = 0.1,
    cost_weight: float = 0.1,
    rank_weight: float = 0.1,
    route_weights: Mapping[str, float] | None = None,
    budget_weight_map: Mapping[int, float] | None = None,
    deep_repair_min_depth: int = 5,
    deep_repair_weight: float = 1.0,
    hole_oversample_repeat: int = 1,
    deep_repair_oversample_repeat: int = 1,
    objective_mode: str | None = None,
    objective_hybrid_mix: float | None = None,
    controller_targets: Sequence[str] | None = None,
    controller_seeds: Sequence[int] | None = None,
    controller_profile: str = "default",
    stage1_targets: Sequence[str] | None = None,
    stage1_seeds: Sequence[int] | None = None,
    stage1_critic_path: str = "",
    stage1_blends: Sequence[float] = STAGE1_DEFAULT_BLENDS,
    stage1_arm_modes: Sequence[str] = DEFAULT_STAGE1_WORKFLOW_ARM_MODES,
    stage1_macro_profile: str = "default",
    n_iter: int = 120,
    max_depth: int = 5,
    brute_depth: int = 0,
    n_fit: int = 128,
    n_probe: int = 512,
    refine_enable: bool = True,
    solve_mse: float = 1.0e-10,
    dtype: torch.dtype = torch.float64,
    threads: int | None = 1,
    capture_search_output: bool = True,
    replay_acquisition_threshold: float = 0.25,
    replay_fallback_min_confidence: float = 0.0,
    replay_uncertainty_bonus: float = 0.05,
    replay_hole_exact_budget_cap: int | None = None,
    eligibility_floor: float = DEFAULT_ELIGIBILITY_FLOOR,
    promote_min_eligible_utility_lift: float = DEFAULT_PROMOTE_MIN_ELIGIBLE_UTILITY_LIFT,
    promote_max_ineligible_mean_budget: float = DEFAULT_PROMOTE_MAX_INELIGIBLE_MEAN_BUDGET,
    promote_max_calibration_error: float = DEFAULT_PROMOTE_MAX_CALIBRATION_ERROR,
    promote_min_online_solve_delta: float = DEFAULT_PROMOTE_MIN_ONLINE_SOLVE_DELTA,
    promote_max_online_wall_ratio: float = DEFAULT_PROMOTE_MAX_ONLINE_WALL_RATIO,
) -> dict[str, Any]:
    compare_modes = _normalize_comparison_modes(comparison_modes)
    out_dir = _resolve_output_dir(output_dir, train_scheduler_bundle=bool(train_scheduler_bundle))
    effective_bundle_path, training_summary = _resolve_effective_bundle(
        scheduler_bundle_path=str(scheduler_bundle_path or ""),
        scheduler_dataset_paths=scheduler_dataset_paths,
        train_scheduler_bundle=bool(train_scheduler_bundle),
        output_dir=out_dir,
        init_bundle_path=str(init_bundle_path or ""),
        budget_ladder=budget_ladder,
        threshold_ladder=threshold_ladder,
        route_aliases=route_aliases,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_fraction=float(val_fraction),
        seed=int(training_seed),
        ensemble_size=int(ensemble_size),
        break_weight=float(break_weight),
        tail_weight=float(tail_weight),
        route_win_weight=float(route_win_weight),
        new_residual_basin_weight=float(new_residual_basin_weight),
        fragile_weight=float(fragile_weight),
        stable_weight=float(stable_weight),
        cost_weight=float(cost_weight),
        rank_weight=float(rank_weight),
        route_weights=route_weights,
        budget_weight_map=budget_weight_map,
        deep_repair_min_depth=int(deep_repair_min_depth),
        deep_repair_weight=float(deep_repair_weight),
        hole_oversample_repeat=int(hole_oversample_repeat),
        deep_repair_oversample_repeat=int(deep_repair_oversample_repeat),
        objective_mode=objective_mode,
        objective_hybrid_mix=objective_hybrid_mix,
    )

    replay_report = None
    dataset_path_list = [str(path) for path in list(scheduler_dataset_paths or []) if str(path).strip()]
    if dataset_path_list:
        bundle = load_scheduler_bundle(str(effective_bundle_path))
        rows = load_scheduler_dataset_rows(dataset_path_list)
        replay_report = replay_scheduler_decisions(
            rows,
            bundle,
            acquisition_threshold=float(replay_acquisition_threshold),
            budget_ladder=budget_ladder,
            fallback_min_confidence=float(replay_fallback_min_confidence),
            uncertainty_bonus=float(replay_uncertainty_bonus),
            hole_exact_budget_cap=replay_hole_exact_budget_cap,
        )

    controller_reports: dict[str, dict[str, Any]] = {}
    stage1_reports: dict[str, dict[str, Any]] = {}
    promotion_by_mode: dict[str, dict[str, Any]] = {}
    for mode in compare_modes:
        controller_report = run_controller_benchmark(
            targets=_normalize_strings(controller_targets, CONTROLLER_DEFAULT_TARGETS),
            seeds=_normalize_ints(controller_seeds, CONTROLLER_DEFAULT_SEEDS),
            profile=str(controller_profile),
            scheduler_bundle_path=str(effective_bundle_path),
            scheduler_budget_ladder=budget_ladder,
            comparison_mode=str(mode),
            n_iter=int(n_iter),
            max_depth=int(max_depth),
            brute_depth=int(brute_depth),
            solve_mse=float(solve_mse),
            dtype=dtype,
            capture_search_output=bool(capture_search_output),
            threads=threads,
        )
        controller_reports[str(mode)] = dict(controller_report or {})

        stage1_report = run_stage1_benchmark(
            targets=_normalize_strings(stage1_targets, STAGE1_DEFAULT_TARGETS),
            seeds=_normalize_ints(stage1_seeds, STAGE1_DEFAULT_SEEDS),
            critic_path=str(stage1_critic_path or ""),
            scheduler_bundle_path=str(effective_bundle_path),
            scheduler_dataset_paths=(),
            scheduler_budget_ladder=budget_ladder,
            comparison_mode=str(mode),
            blends=[float(v) for v in list(stage1_blends or STAGE1_DEFAULT_BLENDS)],
            arm_modes=[str(v) for v in list(stage1_arm_modes or DEFAULT_STAGE1_WORKFLOW_ARM_MODES)],
            macro_profile=str(stage1_macro_profile),
            n_iter=int(n_iter),
            max_depth=int(max_depth),
            brute_depth=int(brute_depth),
            n_fit=int(n_fit),
            n_probe=int(n_probe),
            refine_enable=bool(refine_enable),
            solve_mse=float(solve_mse),
            dtype=dtype,
            capture_search_output=bool(capture_search_output),
            threads=threads,
        )
        stage1_report = dict(stage1_report or {})
        stage1_config = dict(stage1_report.get("config", {}) or {})
        stage1_config["scheduler_dataset_paths"] = list(dataset_path_list)
        stage1_report["config"] = stage1_config
        if replay_report is not None:
            stage1_report["scheduler_replay"] = dict(replay_report)
        stage1_reports[str(mode)] = stage1_report

        promotion_by_mode[str(mode)] = recommend_scheduler_promotion(
            controller_report=controller_report,
            stage1_report=stage1_report,
            eligibility_floor=float(eligibility_floor),
            promote_min_eligible_utility_lift=float(promote_min_eligible_utility_lift),
            promote_max_ineligible_mean_budget=float(promote_max_ineligible_mean_budget),
            promote_max_calibration_error=float(promote_max_calibration_error),
            promote_min_online_solve_delta=float(promote_min_online_solve_delta),
            promote_max_online_wall_ratio=float(promote_max_online_wall_ratio),
        )

    primary_mode = "matched_wall" if "matched_wall" in compare_modes else str(compare_modes[0])
    packet = {
        "mode": "scheduler_workflow",
        "config": {
            "comparison_modes": [str(v) for v in compare_modes],
            "primary_promotion_mode": str(primary_mode),
            "scheduler_bundle_path": str(scheduler_bundle_path or ""),
            "effective_scheduler_bundle_path": str(effective_bundle_path),
            "scheduler_dataset_paths": list(dataset_path_list),
            "train_scheduler_bundle": bool(train_scheduler_bundle),
            "init_bundle_path": str(init_bundle_path or ""),
            "budget_ladder": None if budget_ladder is None else [int(v) for v in list(budget_ladder)],
            "threshold_ladder": None if threshold_ladder is None else [float(v) for v in list(threshold_ladder)],
            "route_aliases": dict(route_aliases or {}),
            "objective_mode": None if objective_mode is None else str(objective_mode),
            "objective_hybrid_mix": None if objective_hybrid_mix is None else float(objective_hybrid_mix),
            "controller_targets": _normalize_strings(controller_targets, CONTROLLER_DEFAULT_TARGETS),
            "controller_seeds": _normalize_ints(controller_seeds, CONTROLLER_DEFAULT_SEEDS),
            "controller_profile": str(controller_profile),
            "stage1_targets": _normalize_strings(stage1_targets, STAGE1_DEFAULT_TARGETS),
            "stage1_seeds": _normalize_ints(stage1_seeds, STAGE1_DEFAULT_SEEDS),
            "stage1_critic_path": str(stage1_critic_path or ""),
            "stage1_blends": [float(v) for v in list(stage1_blends or STAGE1_DEFAULT_BLENDS)],
            "stage1_arm_modes": [str(v) for v in list(stage1_arm_modes or DEFAULT_STAGE1_WORKFLOW_ARM_MODES)],
            "stage1_macro_profile": str(stage1_macro_profile),
            "n_iter": int(n_iter),
            "max_depth": int(max_depth),
            "brute_depth": int(brute_depth),
            "n_fit": int(n_fit),
            "n_probe": int(n_probe),
            "refine_enable": bool(refine_enable),
            "solve_mse": float(solve_mse),
            "replay_acquisition_threshold": float(replay_acquisition_threshold),
            "replay_fallback_min_confidence": float(replay_fallback_min_confidence),
            "replay_uncertainty_bonus": float(replay_uncertainty_bonus),
            "replay_hole_exact_budget_cap": None if replay_hole_exact_budget_cap is None else int(replay_hole_exact_budget_cap),
            "threads": None if threads is None else int(threads),
            "dtype": str(dtype),
        },
        "training_summary": training_summary,
        "replay_report": replay_report,
        "controller_reports": controller_reports,
        "stage1_reports": stage1_reports,
        "promotion_by_mode": promotion_by_mode,
        "promotion": dict(promotion_by_mode.get(primary_mode, {}) or {}),
    }

    if out_dir is not None:
        for mode, report in controller_reports.items():
            _write_json(report, out_dir / f"controller_report_{mode}.json")
        for mode, report in stage1_reports.items():
            _write_json(report, out_dir / f"stage1_report_{mode}.json")
        for mode, report in promotion_by_mode.items():
            _write_json(report, out_dir / f"scheduler_promotion_{mode}.json")
        if training_summary is not None:
            _write_json(training_summary, out_dir / "scheduler_training_summary.json")
        if replay_report is not None:
            _write_json(replay_report, out_dir / "scheduler_replay.json")
        _write_json(packet, out_dir / "scheduler_workflow_packet.json")
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the advisory->train->replay->promotion scheduler workflow")
    p.add_argument("--scheduler_bundle_path", type=str, default="")
    p.add_argument("--scheduler_dataset_paths", nargs="*", default=[])
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--comparison_modes", nargs="*", default=list(DEFAULT_WORKFLOW_COMPARISON_MODES))
    p.add_argument("--train_scheduler_bundle", action="store_true")
    p.add_argument("--init_bundle_path", type=str, default="")
    p.add_argument("--budget_ladder", type=str, default="")
    p.add_argument("--threshold_ladder", type=str, default="")
    p.add_argument("--route_aliases", type=str, default="")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--training_seed", type=int, default=0)
    p.add_argument("--ensemble_size", type=int, default=4)
    p.add_argument("--break_weight", type=float, default=1.0)
    p.add_argument("--tail_weight", type=float, default=0.5)
    p.add_argument("--route_win_weight", type=float, default=0.4)
    p.add_argument("--new_residual_basin_weight", type=float, default=0.15)
    p.add_argument("--fragile_weight", type=float, default=0.1)
    p.add_argument("--stable_weight", type=float, default=0.1)
    p.add_argument("--cost_weight", type=float, default=0.1)
    p.add_argument("--rank_weight", type=float, default=0.1)
    p.add_argument("--route_weights", type=str, default="")
    p.add_argument("--budget_weights", type=str, default="")
    p.add_argument("--deep_repair_min_depth", type=int, default=5)
    p.add_argument("--deep_repair_weight", type=float, default=1.0)
    p.add_argument("--hole_oversample_repeat", type=int, default=1)
    p.add_argument("--deep_repair_oversample_repeat", type=int, default=1)
    p.add_argument("--objective_mode", type=str, default="")
    p.add_argument("--objective_hybrid_mix", type=float, default=None)
    p.add_argument("--controller_targets", nargs="*", default=None)
    p.add_argument("--controller_seeds", nargs="*", type=int, default=None)
    p.add_argument("--controller_profile", type=str, default="default")
    p.add_argument("--stage1_targets", nargs="*", default=None)
    p.add_argument("--stage1_seeds", nargs="*", type=int, default=None)
    p.add_argument("--stage1_critic_path", type=str, default="")
    p.add_argument("--stage1_blends", nargs="*", type=float, default=list(STAGE1_DEFAULT_BLENDS))
    p.add_argument("--stage1_arm_modes", nargs="*", default=list(DEFAULT_STAGE1_WORKFLOW_ARM_MODES))
    p.add_argument("--stage1_macro_profile", type=str, default="default")
    p.add_argument("--n_iter", type=int, default=120)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--brute_depth", type=int, default=0)
    p.add_argument("--n_fit", type=int, default=128)
    p.add_argument("--n_probe", type=int, default=512)
    p.add_argument("--solve_mse", type=float, default=1.0e-10)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no_plus", action="store_true")
    p.add_argument("--no_capture_search_output", action="store_true")
    p.add_argument("--replay_acquisition_threshold", type=float, default=0.25)
    p.add_argument("--replay_fallback_min_confidence", type=float, default=0.0)
    p.add_argument("--replay_uncertainty_bonus", type=float, default=0.05)
    p.add_argument("--replay_hole_exact_budget_cap", type=int, default=None)
    p.add_argument("--eligibility_floor", type=float, default=DEFAULT_ELIGIBILITY_FLOOR)
    p.add_argument("--promote_min_eligible_utility_lift", type=float, default=DEFAULT_PROMOTE_MIN_ELIGIBLE_UTILITY_LIFT)
    p.add_argument("--promote_max_ineligible_mean_budget", type=float, default=DEFAULT_PROMOTE_MAX_INELIGIBLE_MEAN_BUDGET)
    p.add_argument("--promote_max_calibration_error", type=float, default=DEFAULT_PROMOTE_MAX_CALIBRATION_ERROR)
    p.add_argument("--promote_min_online_solve_delta", type=float, default=DEFAULT_PROMOTE_MIN_ONLINE_SOLVE_DELTA)
    p.add_argument("--promote_max_online_wall_ratio", type=float, default=DEFAULT_PROMOTE_MAX_ONLINE_WALL_RATIO)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    report = run_scheduler_workflow(
        scheduler_bundle_path=str(args.scheduler_bundle_path or ""),
        scheduler_dataset_paths=[str(path) for path in list(args.scheduler_dataset_paths or [])],
        output_dir=str(args.output_dir or ""),
        comparison_modes=[str(mode) for mode in list(args.comparison_modes or [])],
        train_scheduler_bundle=bool(args.train_scheduler_bundle),
        init_bundle_path=str(args.init_bundle_path or ""),
        budget_ladder=[int(token.strip()) for token in str(args.budget_ladder or "").split(",") if token.strip()] or None,
        threshold_ladder=[float(token.strip()) for token in str(args.threshold_ladder or "").split(",") if token.strip()] or None,
        route_aliases=_parse_alias_spec(str(args.route_aliases or "")),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        training_seed=int(args.training_seed),
        ensemble_size=int(args.ensemble_size),
        break_weight=float(args.break_weight),
        tail_weight=float(args.tail_weight),
        route_win_weight=float(args.route_win_weight),
        new_residual_basin_weight=float(args.new_residual_basin_weight),
        fragile_weight=float(args.fragile_weight),
        stable_weight=float(args.stable_weight),
        cost_weight=float(args.cost_weight),
        rank_weight=float(args.rank_weight),
        route_weights=_parse_weight_spec(str(args.route_weights or ""), int_keys=False),
        budget_weight_map=_parse_weight_spec(str(args.budget_weights or ""), int_keys=True),
        deep_repair_min_depth=int(args.deep_repair_min_depth),
        deep_repair_weight=float(args.deep_repair_weight),
        hole_oversample_repeat=int(args.hole_oversample_repeat),
        deep_repair_oversample_repeat=int(args.deep_repair_oversample_repeat),
        objective_mode=(None if not str(args.objective_mode or "").strip() else str(args.objective_mode)),
        objective_hybrid_mix=args.objective_hybrid_mix,
        controller_targets=None if args.controller_targets is None else [str(v) for v in args.controller_targets],
        controller_seeds=None if args.controller_seeds is None else [int(v) for v in args.controller_seeds],
        controller_profile=str(args.controller_profile),
        stage1_targets=None if args.stage1_targets is None else [str(v) for v in args.stage1_targets],
        stage1_seeds=None if args.stage1_seeds is None else [int(v) for v in args.stage1_seeds],
        stage1_critic_path=str(args.stage1_critic_path or ""),
        stage1_blends=[float(v) for v in list(args.stage1_blends or STAGE1_DEFAULT_BLENDS)],
        stage1_arm_modes=[str(v) for v in list(args.stage1_arm_modes or DEFAULT_STAGE1_WORKFLOW_ARM_MODES)],
        stage1_macro_profile=str(args.stage1_macro_profile),
        n_iter=int(args.n_iter),
        max_depth=int(args.max_depth),
        brute_depth=int(args.brute_depth),
        n_fit=int(args.n_fit),
        n_probe=int(args.n_probe),
        refine_enable=not bool(args.no_plus),
        solve_mse=float(args.solve_mse),
        dtype=_dtype_from_name(str(args.dtype)),
        threads=None if args.threads is None else int(args.threads),
        capture_search_output=not bool(args.no_capture_search_output),
        replay_acquisition_threshold=float(args.replay_acquisition_threshold),
        replay_fallback_min_confidence=float(args.replay_fallback_min_confidence),
        replay_uncertainty_bonus=float(args.replay_uncertainty_bonus),
        replay_hole_exact_budget_cap=None if args.replay_hole_exact_budget_cap is None else int(args.replay_hole_exact_budget_cap),
        eligibility_floor=float(args.eligibility_floor),
        promote_min_eligible_utility_lift=float(args.promote_min_eligible_utility_lift),
        promote_max_ineligible_mean_budget=float(args.promote_max_ineligible_mean_budget),
        promote_max_calibration_error=float(args.promote_max_calibration_error),
        promote_min_online_solve_delta=float(args.promote_min_online_solve_delta),
        promote_max_online_wall_ratio=float(args.promote_max_online_wall_ratio),
    )
    if args.json or not str(args.output_dir or "").strip():
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
