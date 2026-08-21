# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Held-out benchmark harness for the Stage-1 repair critic.

Examples:
  python nestynet_sr/sr_search/factorized_search/stage1_benchmark_harness.py \
    --critic_path /tmp/repair_critic_stage1_enriched_v2.pt

  python nestynet_sr/sr_search/factorized_search/stage1_benchmark_harness.py \
    --targets feynman_090 custom_rational_sqdiff eq026_nested_recip \
    --seeds 10 11 12 13 \
    --critic_path /tmp/repair_critic_stage1_enriched_v2.pt \
    --arm_modes gate macro \
    --macro_profile repair_probe \
    --n_iter 180 --max_depth 5 --json
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import io
import json
import math
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.controller_harness import (
    DEFAULT_COMPARISON_MODE,
    _observed_cost_totals,
    _normalize_comparison_mode,
    _profile_overrides,
    _route_usage_from_action_counts,
    _scheduler_log_means,
    _resolve_target_spec,
    _set_torch_threads,
    _summarize_pairwise_comparisons,
)
from nestynet_sr.sr_search.factorized_search.engine.search import run_explorer_core
from nestynet_sr.sr_search.factorized_search.scheduler_critic import load_scheduler_bundle
from nestynet_sr.sr_search.factorized_search.scheduler_dataset import load_scheduler_dataset_rows
from nestynet_sr.sr_search.factorized_search.scheduler_replay import replay_scheduler_decisions


DEFAULT_TARGETS = ("addsum", "poly", "feynman_090", "custom_rational_sqdiff", "eq026_nested_recip")
DEFAULT_SEEDS = (10, 11, 12, 13)
DEFAULT_BLENDS = (0.25, 0.50)
DEFAULT_ARM_MODES = ("priority", "gate", "macro")


@dataclass
class Stage1RunSummary:
    arm: str
    target: str
    seed: int
    n_iter: int
    max_depth: int
    refine_enable: bool
    critic_blend: float | None
    critic_mode: str | None
    macro_enabled: bool
    best_mse: float
    best_expr: str
    residual_basins: int
    n_eval: int
    elapsed_s: float
    solve_hit: bool = False
    exact_eval_count: int = 0
    preview_eval_count: int = 0
    route_usage: dict[str, int] = field(default_factory=dict)
    repair_considered: int = 0
    repair_selected: int = 0
    repair_option_selected: int = 0
    repair_no_candidate: int = 0
    repair_blocked_low_score: int = 0
    repair_blocked_low_concentration: int = 0
    macro_selected: int = 0
    macro_repair_selected: int = 0
    macro_fallback_selected: int = 0
    macro_decision_source_counts: dict[str, int] = field(default_factory=dict)
    scheduler_enabled: bool = False
    scheduler_advisory_only: bool = False
    scheduler_bundle_loaded: bool = False
    scheduler_decision_count: int = 0
    scheduler_scored: int = 0
    scheduler_control_selected: int = 0
    scheduler_fallback_selected: int = 0
    scheduler_route_counts: dict[str, int] = field(default_factory=dict)
    scheduler_mean_confidence: float | None = None
    critic_loaded: bool = False
    controller_score_source_counts: dict[str, int] = field(default_factory=dict)
    controller_status_counts: dict[str, int] = field(default_factory=dict)
    search_stdout_tail: list[str] = field(default_factory=list)

    # Share the controller_harness.RunSummary comparison surface so the
    # stage-1 harness can reuse the same pairwise aggregation helpers.
    @property
    def label(self) -> str:
        return str(self.arm)

    @property
    def best_eff_mse(self) -> float:
        return float(self.best_mse)

    @property
    def best_raw_mse(self) -> float:
        return float(self.best_mse)


def _dtype_from_name(name: str) -> torch.dtype:
    token = str(name or "float64").strip().lower()
    if token in ("float32", "fp32", "f32"):
        return torch.float32
    if token in ("float64", "fp64", "f64", "double"):
        return torch.float64
    raise ValueError(f"unknown dtype: {name!r}")


def _tail_lines(text: str, n: int = 8) -> list[str]:
    rows = [str(line) for line in str(text).splitlines() if str(line).strip()]
    if len(rows) <= n:
        return rows
    return rows[-n:]


def _safe_mean(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _safe_median(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _normalize_arm_modes(arm_modes: Sequence[str] | None) -> tuple[str, ...]:
    if not arm_modes:
        return ()
    aliases = {
        "hybrid": "priority",
        "sidecar": "priority",
        "gated": "gate",
        "controller_gate": "gate",
        "macro_controller": "macro",
        "scheduler": "scheduler_advisory",
        "advisory": "scheduler_advisory",
        "scheduler_advice": "scheduler_advisory",
        "control": "scheduler_control",
    }
    out: list[str] = []
    seen: set[str] = set()
    for mode in arm_modes:
        key = aliases.get(str(mode or "").strip().lower(), str(mode or "").strip().lower())
        if key not in {"priority", "gate", "macro", "scheduler_advisory", "scheduler_control"} or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _count_log_values(rows: Sequence[Any] | None, key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        value = str(row.get(key, "")).strip()
        if value:
            counter[value] += 1
    return dict(sorted(counter.items()))


def _arm_configs(
    *,
    critic_path: str = "",
    scheduler_bundle_path: str = "",
    scheduler_budget_ladder: Sequence[int] | None = None,
    scheduler_arm_overrides: Mapping[str, Any] | None = None,
    blends: Sequence[float],
    refine_enable: bool,
    arm_modes: Sequence[str] = DEFAULT_ARM_MODES,
    macro_profile: str = "default",
    macro_controller_learned_policy_weight: float | None = None,
    macro_controller_learned_route_weight: float | None = None,
    macro_controller_learned_q_weight: float | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    arms: list[tuple[str, dict[str, Any]]] = []
    baseline_cfg = dict(_profile_overrides("default", macro_enabled=False))
    baseline_cfg.update({
        "repair_controller_critic_enable": False,
        "refine_enable": bool(refine_enable),
    })
    arms.append(
        (
            "stage0_selective_plus" if refine_enable else "stage0_selective",
            baseline_cfg,
        )
    )
    arm_modes_norm = _normalize_arm_modes(arm_modes)
    if not arm_modes_norm:
        return arms
    critic_modes = [mode for mode in arm_modes_norm if mode in {"priority", "gate", "macro"}]
    scheduler_modes = [mode for mode in arm_modes_norm if mode in {"scheduler_advisory", "scheduler_control"}]
    critic_path = str(critic_path or "").strip()
    scheduler_bundle_path = str(scheduler_bundle_path or "").strip()
    if critic_modes and not critic_path:
        raise ValueError("--critic_path is required when Stage-1 critic arms are requested")
    if scheduler_modes and not scheduler_bundle_path:
        raise ValueError("--scheduler_bundle_path is required when scheduler arms are requested")
    for blend in blends if critic_modes else ():
        blend_f = float(blend)
        tag = int(round(100.0 * blend_f))
        critic_common = {
            "repair_controller_critic_enable": True,
            "repair_controller_critic_path": critic_path,
            "repair_controller_critic_blend": float(blend_f),
            "refine_enable": bool(refine_enable),
        }
        if "priority" in arm_modes_norm:
            label = f"stage1_hybrid_b{tag:03d}"
            if refine_enable:
                label += "_plus"
            cfg = dict(_profile_overrides("default", macro_enabled=False))
            cfg.update(critic_common)
            cfg["repair_controller_critic_mode"] = "priority"
            arms.append((label, cfg))
        if "gate" in arm_modes_norm:
            label = f"stage1_gate_b{tag:03d}"
            if refine_enable:
                label += "_plus"
            cfg = dict(_profile_overrides("default", macro_enabled=False))
            cfg.update(critic_common)
            cfg["repair_controller_critic_mode"] = "decisive"
            arms.append((label, cfg))
        if "macro" in arm_modes_norm:
            macro_name = str(macro_profile or "default").strip().lower()
            label = f"stage1_macro_b{tag:03d}"
            if macro_name not in ("", "default", "dropin"):
                label = f"stage1_macro_{macro_name}_b{tag:03d}"
            if refine_enable:
                label += "_plus"
            cfg = dict(_profile_overrides(macro_profile, macro_enabled=True))
            cfg.update(critic_common)
            cfg["repair_controller_critic_mode"] = "priority"
            if macro_controller_learned_policy_weight is not None:
                cfg["macro_controller_learned_policy_weight"] = float(macro_controller_learned_policy_weight)
            if macro_controller_learned_route_weight is not None:
                cfg["macro_controller_learned_route_weight"] = float(macro_controller_learned_route_weight)
            if macro_controller_learned_q_weight is not None:
                cfg["macro_controller_learned_q_weight"] = float(macro_controller_learned_q_weight)
            arms.append((label, cfg))
    if scheduler_modes:
        scheduler_common = dict(_profile_overrides(macro_profile, macro_enabled=True))
        scheduler_common.update({
            "repair_controller_critic_enable": False,
            "refine_enable": bool(refine_enable),
            "scheduler_enable": True,
            "scheduler_bundle_path": str(scheduler_bundle_path),
        })
        if scheduler_budget_ladder is not None:
            scheduler_common["scheduler_budget_ladder"] = [int(v) for v in list(scheduler_budget_ladder)]
        if scheduler_arm_overrides is not None:
            scheduler_common.update(dict(scheduler_arm_overrides))
        macro_name = str(macro_profile or "default").strip().lower()
        if "scheduler_advisory" in scheduler_modes:
            label = "stage1_scheduler_advisory"
            if macro_name not in ("", "default", "dropin"):
                label = f"stage1_scheduler_advisory_{macro_name}"
            if refine_enable:
                label += "_plus"
            cfg = dict(scheduler_common)
            cfg["scheduler_advisory_only"] = True
            arms.append((label, cfg))
        if "scheduler_control" in scheduler_modes:
            label = "stage1_scheduler_control"
            if macro_name not in ("", "default", "dropin"):
                label = f"stage1_scheduler_control_{macro_name}"
            if refine_enable:
                label += "_plus"
            cfg = dict(scheduler_common)
            cfg["scheduler_advisory_only"] = False
            arms.append((label, cfg))
    return arms


def _best_from_arch(arch) -> tuple[float, str]:
    if not getattr(arch, "d", None):
        return float("inf"), ""
    try:
        best = arch.best(1)[0]
    except Exception:
        return float("inf"), ""
    return (
        float(getattr(best, "best_mse", float("inf"))),
        str(explorer.node_str(getattr(best, "best_expr", None))),
    )


def run_stage1_experiment(
    target: str,
    *,
    seed: int,
    arm: str,
    arm_cfg: dict[str, Any],
    n_iter: int = 180,
    max_depth: int = 5,
    brute_depth: int = 0,
    n_fit: int = 128,
    n_probe: int = 512,
    refine_enable: bool = True,
    solve_mse: float = 1.0e-10,
    dtype: torch.dtype = torch.float64,
    capture_search_output: bool = True,
    threads: int | None = 1,
) -> Stage1RunSummary:
    _set_torch_threads(threads)
    spec = _resolve_target_spec(str(target))
    kwargs = {
        "target_fn": spec["fn"],
        "nvars": int(spec["nvars"]),
        "seed": int(seed),
        "n_iter": int(n_iter),
        "max_depth": int(max_depth),
        "brute_depth": int(brute_depth),
        "n_fit": int(n_fit),
        "n_probe": int(n_probe),
        "dtype": dtype,
        "print_every": 0,
        "lo": spec.get("lo", 1.0),
        "hi": spec.get("hi", 5.0),
        "y_dims": spec.get("y_dims", None),
        "var_dims": spec.get("var_dims", None),
        "refine_enable": bool(refine_enable),
    }
    kwargs.update(dict(arm_cfg))
    kwargs.setdefault("_runtime_hooks", explorer.make_engine_runtime_hooks())

    start = time.time()
    if capture_search_output:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            arch = run_explorer_core(**kwargs)
        stdout_text = buf.getvalue()
    else:
        arch = run_explorer_core(**kwargs)
        stdout_text = ""
    elapsed_s = time.time() - start

    best_mse, best_expr = _best_from_arch(arch)
    rcs = dict(getattr(arch, "repair_controller_stats", {}) or {})
    mcs = dict(getattr(arch, "macro_controller_stats", {}) or {})
    scs = dict(getattr(arch, "scheduler_stats", {}) or {})
    inv_log = list(getattr(arch, "inverse_experiment_log", []) or [])
    scheduler_log = list(getattr(arch, "scheduler_decision_log", []) or [])
    exact_eval_count, preview_eval_count = _observed_cost_totals(inv_log)
    scheduler_mean_confidence, _scheduler_mean_candidates = _scheduler_log_means(scheduler_log)
    action_counts = {
        str(k): int(v)
        for k, v in dict((getattr(arch, "action_distribution", {}) or {}).get("counts", {}) or {}).items()
    }
    route_usage = _route_usage_from_action_counts(action_counts)
    blend = arm_cfg.get("repair_controller_critic_blend", None)
    return Stage1RunSummary(
        arm=str(arm),
        target=str(target),
        seed=int(seed),
        n_iter=int(n_iter),
        max_depth=int(max_depth),
        refine_enable=bool(refine_enable),
        critic_blend=None if blend is None else float(blend),
        critic_mode=None if blend is None else str(arm_cfg.get("repair_controller_critic_mode", "priority")),
        macro_enabled=bool(arm_cfg.get("macro_controller_enable", False)),
        best_mse=float(best_mse),
        best_expr=str(best_expr),
        residual_basins=int(len(getattr(arch, "d", {}) or {})),
        n_eval=int(getattr(arch, "n_eval", 0)),
        elapsed_s=float(elapsed_s),
        solve_hit=bool(math.isfinite(best_mse) and float(best_mse) <= float(solve_mse)),
        exact_eval_count=int(exact_eval_count),
        preview_eval_count=int(preview_eval_count),
        route_usage=route_usage,
        repair_considered=int(rcs.get("considered", 0)),
        repair_selected=int(rcs.get("selected", 0)),
        repair_option_selected=int(rcs.get("option_repair_selected", 0)),
        repair_no_candidate=int(rcs.get("no_candidate", 0)),
        repair_blocked_low_score=int(rcs.get("blocked_low_score", 0)),
        repair_blocked_low_concentration=int(rcs.get("blocked_low_concentration", 0)),
        macro_selected=int(mcs.get("selected", 0)),
        macro_repair_selected=int(mcs.get("repair_selected", 0)),
        macro_fallback_selected=int(mcs.get("fallback_selected", 0)),
        macro_decision_source_counts=dict(mcs.get("decision_source_counts", {}) or {}),
        scheduler_enabled=bool(scs.get("enabled", False)),
        scheduler_advisory_only=bool(scs.get("advisory_only", False)),
        scheduler_bundle_loaded=bool(scs.get("bundle_loaded", False)),
        scheduler_decision_count=int(scs.get("decision_count", 0)),
        scheduler_scored=int(scs.get("scored", 0)),
        scheduler_control_selected=int(scs.get("control_selected", 0)),
        scheduler_fallback_selected=int(scs.get("fallback_selected", 0)),
        scheduler_route_counts={str(k): int(v) for k, v in dict(scs.get("route_counts", {}) or {}).items()},
        scheduler_mean_confidence=scheduler_mean_confidence,
        critic_loaded=bool(rcs.get("critic_loaded", False)),
        controller_score_source_counts=_count_log_values(inv_log, "controller_score_source"),
        controller_status_counts=_count_log_values(inv_log, "status"),
        search_stdout_tail=_tail_lines(stdout_text, n=8),
    )


def run_stage1_benchmark(
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    critic_path: str = "",
    scheduler_bundle_path: str = "",
    scheduler_dataset_paths: Sequence[str] = (),
    scheduler_budget_ladder: Sequence[int] | None = None,
    scheduler_arm_overrides: Mapping[str, Any] | None = None,
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    blends: Sequence[float] = DEFAULT_BLENDS,
    arm_modes: Sequence[str] = DEFAULT_ARM_MODES,
    macro_profile: str = "default",
    macro_controller_learned_policy_weight: float | None = None,
    macro_controller_learned_route_weight: float | None = None,
    macro_controller_learned_q_weight: float | None = None,
    n_iter: int = 180,
    max_depth: int = 5,
    brute_depth: int = 0,
    n_fit: int = 128,
    n_probe: int = 512,
    refine_enable: bool = True,
    solve_mse: float = 1.0e-10,
    dtype: torch.dtype = torch.float64,
    capture_search_output: bool = True,
    threads: int | None = 1,
) -> dict[str, Any]:
    normalized_comparison_mode = _normalize_comparison_mode(comparison_mode)
    arms = _arm_configs(
        critic_path=critic_path,
        scheduler_bundle_path=scheduler_bundle_path,
        scheduler_budget_ladder=scheduler_budget_ladder,
        scheduler_arm_overrides=scheduler_arm_overrides,
        blends=blends,
        refine_enable=bool(refine_enable),
        arm_modes=arm_modes,
        macro_profile=macro_profile,
        macro_controller_learned_policy_weight=macro_controller_learned_policy_weight,
        macro_controller_learned_route_weight=macro_controller_learned_route_weight,
        macro_controller_learned_q_weight=macro_controller_learned_q_weight,
    )
    runs: list[Stage1RunSummary] = []
    for target in targets:
        for seed in seeds:
            for arm_name, arm_cfg in arms:
                runs.append(
                    run_stage1_experiment(
                        str(target),
                        seed=int(seed),
                        arm=str(arm_name),
                        arm_cfg=arm_cfg,
                        n_iter=int(n_iter),
                        max_depth=int(max_depth),
                        brute_depth=int(brute_depth),
                        n_fit=int(n_fit),
                        n_probe=int(n_probe),
                        refine_enable=bool(refine_enable),
                        solve_mse=float(solve_mse),
                        dtype=dtype,
                        capture_search_output=capture_search_output,
                        threads=threads,
                    )
                )

    baseline_arm = arms[0][0]
    by_target: dict[str, dict[str, Any]] = {}
    for target in targets:
        by_target[str(target)] = {}
        for arm_name, _arm_cfg in arms:
            subset = [r for r in runs if r.target == str(target) and r.arm == str(arm_name)]
            mses = [r.best_mse for r in subset]
            walls = [r.elapsed_s for r in subset]
            sels = [r.repair_selected for r in subset]
            opts = [r.repair_option_selected for r in subset]
            solved = sum(1 for v in mses if math.isfinite(v) and v <= float(solve_mse))
            by_target[str(target)][str(arm_name)] = {
                "n_runs": int(len(subset)),
                "solved": int(solved),
                "median_mse": _safe_median(mses),
                "mean_mse": _safe_mean(mses),
                "mean_wall_s": _safe_mean(walls),
                "mean_repair_selected": _safe_mean(sels),
                "mean_option_repair_selected": _safe_mean(opts),
                "mean_macro_selected": _safe_mean([r.macro_selected for r in subset]),
                "mean_macro_repair_selected": _safe_mean([r.macro_repair_selected for r in subset]),
            }

    overall: dict[str, Any] = {}
    for arm_name, _arm_cfg in arms:
        subset = [r for r in runs if r.arm == str(arm_name)]
        overall[str(arm_name)] = {
            "n_runs": int(len(subset)),
            "solve_rate": 0.0 if not subset else float(sum(1 for r in subset if r.solve_hit)) / float(len(subset)),
            "median_mse": _safe_median([r.best_mse for r in subset]),
            "mean_mse": _safe_mean([r.best_mse for r in subset]),
            "mean_wall_s": _safe_mean([r.elapsed_s for r in subset]),
            "mean_exact_eval_count": _safe_mean([r.exact_eval_count for r in subset]),
            "mean_preview_eval_count": _safe_mean([r.preview_eval_count for r in subset]),
            "mean_repair_selected": _safe_mean([r.repair_selected for r in subset]),
            "mean_macro_repair_selected": _safe_mean([r.macro_repair_selected for r in subset]),
            "mean_scheduler_decision_count": _safe_mean([r.scheduler_decision_count for r in subset]),
            "mean_scheduler_control_selected": _safe_mean([r.scheduler_control_selected for r in subset]),
            "route_usage": {
                str(key): int(sum(int((r.route_usage or {}).get(key, 0)) for r in subset))
                for key in ("build", "inverse", "repair", "hole")
            },
        }

    identical_vs_stage0: dict[str, Any] = {}
    for arm_name, _arm_cfg in arms[1:]:
        same = 0
        total = 0
        for target in targets:
            for seed in seeds:
                base = next((r for r in runs if r.target == str(target) and r.seed == int(seed) and r.arm == baseline_arm), None)
                cur = next((r for r in runs if r.target == str(target) and r.seed == int(seed) and r.arm == str(arm_name)), None)
                if base is None or cur is None:
                    continue
                total += 1
                if (
                    float(base.best_mse) == float(cur.best_mse)
                    and int(base.residual_basins) == int(cur.residual_basins)
                    and int(base.n_eval) == int(cur.n_eval)
                    and int(base.repair_considered) == int(cur.repair_considered)
                    and int(base.repair_selected) == int(cur.repair_selected)
                    and int(base.repair_option_selected) == int(cur.repair_option_selected)
                ):
                    same += 1
        identical_vs_stage0[str(arm_name)] = {
            "identical_pairs": int(same),
            "total_pairs": int(total),
        }

    baseline_runs = [r for r in runs if r.arm == str(baseline_arm)]
    comparisons = {
        str(arm_name): _summarize_pairwise_comparisons(
            baseline_runs,
            [r for r in runs if r.arm == str(arm_name)],
            comparison_mode=normalized_comparison_mode,
        )
        for arm_name, _arm_cfg in arms[1:]
    }

    scheduler_replay = None
    scheduler_bundle_path_s = str(scheduler_bundle_path or "").strip()
    if scheduler_bundle_path_s and scheduler_dataset_paths:
        scheduler_bundle = load_scheduler_bundle(scheduler_bundle_path_s)
        scheduler_rows = load_scheduler_dataset_rows(list(scheduler_dataset_paths))
        scheduler_replay = replay_scheduler_decisions(
            scheduler_rows,
            scheduler_bundle,
            acquisition_threshold=0.25,
            budget_ladder=scheduler_budget_ladder,
        )

    return {
        "config": {
            "targets": [str(t) for t in targets],
            "seeds": [int(s) for s in seeds],
            "arms": [str(name) for name, _ in arms],
            "critic_path": str(critic_path),
            "scheduler_bundle_path": str(scheduler_bundle_path or ""),
            "scheduler_dataset_paths": [str(path) for path in scheduler_dataset_paths],
            "scheduler_budget_ladder": None
            if scheduler_budget_ladder is None
            else [int(v) for v in list(scheduler_budget_ladder)],
            "scheduler_arm_overrides": dict(scheduler_arm_overrides or {}),
            "comparison_mode": str(normalized_comparison_mode),
            "blends": [float(b) for b in blends],
            "arm_modes": [str(mode) for mode in _normalize_arm_modes(arm_modes)],
            "macro_profile": str(macro_profile),
            "macro_controller_learned_policy_weight": None if macro_controller_learned_policy_weight is None else float(macro_controller_learned_policy_weight),
            "macro_controller_learned_route_weight": None if macro_controller_learned_route_weight is None else float(macro_controller_learned_route_weight),
            "macro_controller_learned_q_weight": None if macro_controller_learned_q_weight is None else float(macro_controller_learned_q_weight),
            "n_iter": int(n_iter),
            "max_depth": int(max_depth),
            "brute_depth": int(brute_depth),
            "n_fit": int(n_fit),
            "n_probe": int(n_probe),
            "refine_enable": bool(refine_enable),
            "solve_mse": float(solve_mse),
            "threads": None if threads is None else int(threads),
            "dtype": str(dtype),
        },
        "identical_vs_stage0": identical_vs_stage0,
        "comparisons": comparisons,
        "overall": overall,
        "by_target": by_target,
        "scheduler_replay": scheduler_replay,
        "runs": [asdict(run) for run in runs],
    }


def _print_human_report(report: dict[str, Any]) -> None:
    cfg = dict(report.get("config", {}) or {})
    print("\n=== Stage-1 benchmark summary ===")
    print(
        f"targets={cfg.get('targets')} seeds={cfg.get('seeds')} "
        f"n_iter={cfg.get('n_iter')} max_depth={cfg.get('max_depth')} "
        f"plus={cfg.get('refine_enable')} critic={cfg.get('critic_path')} "
        f"comparison_mode={cfg.get('comparison_mode')}"
    )
    print("overall:")
    for arm, stats in dict(report.get("overall", {}) or {}).items():
        print(
            f"  {arm}: median_mse={stats.get('median_mse'):.6g} "
            f"mean_mse={stats.get('mean_mse'):.6g} "
            f"mean_wall_s={stats.get('mean_wall_s'):.3f} "
            f"solve_rate={stats.get('solve_rate'):.3f} "
            f"mean_exact={stats.get('mean_exact_eval_count'):.2f} "
            f"mean_repair={stats.get('mean_repair_selected'):.2f} "
            f"mean_macro_repair={stats.get('mean_macro_repair_selected'):.2f} "
            f"mean_sched={stats.get('mean_scheduler_decision_count'):.2f}"
        )
    ident = dict(report.get("identical_vs_stage0", {}) or {})
    if ident:
        print("identical vs stage0:")
        for arm, stats in ident.items():
            print(f"  {arm}: {stats.get('identical_pairs')}/{stats.get('total_pairs')}")
    comparisons = dict(report.get("comparisons", {}) or {})
    if comparisons:
        print("comparisons:")
        for arm, stats in comparisons.items():
            print(
                f"  {arm}: mode={stats.get('comparison_mode')} "
                f"matched_win_rate={stats.get('matched_win_rate')} "
                f"mean_matched_delta_log_eff={stats.get('mean_matched_delta_log_eff')} "
                f"mean_cost_ratio={stats.get('mean_candidate_cost_ratio')}"
            )
    replay = report.get("scheduler_replay", None)
    if isinstance(replay, dict) and bool(replay.get("trained", False)):
        print(
            "scheduler replay: "
            f"groups={replay.get('groups_replayed')} "
            f"top1_hit_rate={replay.get('top1_hit_rate')} "
            f"mean_regret={replay.get('mean_regret')} "
            f"mean_wasted_budget={replay.get('mean_wasted_budget')}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--critic_path", type=str, default="")
    p.add_argument("--scheduler_bundle_path", type=str, default="")
    p.add_argument("--scheduler_dataset_paths", nargs="*", default=[])
    p.add_argument("--scheduler_budget_ladder", type=str, default="1,2,4,8")
    p.add_argument("--comparison_mode", type=str, default=DEFAULT_COMPARISON_MODE, choices=["raw", "matched_exact", "matched_wall"])
    p.add_argument("--blends", nargs="*", type=float, default=list(DEFAULT_BLENDS))
    p.add_argument(
        "--arm_modes",
        nargs="*",
        default=list(DEFAULT_ARM_MODES),
        choices=["priority", "gate", "macro", "scheduler_advisory", "scheduler_control"],
    )
    p.add_argument("--macro_profile", type=str, default="default", choices=["default", "repair_probe"])
    p.add_argument("--macro_controller_learned_policy_weight", type=float, default=None)
    p.add_argument("--macro_controller_learned_route_weight", type=float, default=None)
    p.add_argument("--macro_controller_learned_q_weight", type=float, default=None)
    p.add_argument("--n_iter", type=int, default=180)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--brute_depth", type=int, default=0)
    p.add_argument("--n_fit", type=int, default=128)
    p.add_argument("--n_probe", type=int, default=512)
    p.add_argument("--solve_mse", type=float, default=1.0e-10)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no_plus", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    report = run_stage1_benchmark(
        targets=args.targets,
        seeds=args.seeds,
        critic_path=str(args.critic_path),
        scheduler_bundle_path=str(args.scheduler_bundle_path),
        scheduler_dataset_paths=[str(path) for path in args.scheduler_dataset_paths],
        scheduler_budget_ladder=[
            int(s.strip())
            for s in str(args.scheduler_budget_ladder).split(",")
            if s.strip()
        ],
        comparison_mode=str(args.comparison_mode),
        blends=args.blends,
        arm_modes=args.arm_modes,
        macro_profile=str(args.macro_profile),
        macro_controller_learned_policy_weight=args.macro_controller_learned_policy_weight,
        macro_controller_learned_route_weight=args.macro_controller_learned_route_weight,
        macro_controller_learned_q_weight=args.macro_controller_learned_q_weight,
        n_iter=int(args.n_iter),
        max_depth=int(args.max_depth),
        brute_depth=int(args.brute_depth),
        n_fit=int(args.n_fit),
        n_probe=int(args.n_probe),
        refine_enable=not bool(args.no_plus),
        solve_mse=float(args.solve_mse),
        dtype=_dtype_from_name(args.dtype),
        capture_search_output=True,
        threads=args.threads,
    )

    if args.output:
        out_path = pathlib.Path(str(args.output))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if bool(args.json):
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(report)


if __name__ == "__main__":
    main()
