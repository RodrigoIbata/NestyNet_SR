# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Paired benchmark harness for the factorized symbolic search macro controller.

Examples:
  python nestynet_sr/sr_search/factorized_search/controller_harness.py
  python nestynet_sr/sr_search/factorized_search/controller_harness.py --profile repair_probe --targets addsum --seeds 0 1 --n_iter 40
  python nestynet_sr/sr_search/factorized_search/controller_harness.py --targets custom_rational_sqdiff eq026_nested_recip --seeds 0 1 --profile default --n_iter 120
  python nestynet_sr/sr_search/factorized_search/controller_harness.py --json

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python nestynet_sr/sr_search/factorized_search/controller_harness.py --targets addsum --seeds 0 --profile repair_probe --n_iter 8 --max_depth 4
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import contextlib
import io
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.engine.search import run_explorer_core


DEFAULT_TARGETS = ("addsum", "poly", "feynman_090")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_COMPARISON_MODE = "matched_exact"
VALID_COMPARISON_MODES = {"raw", "matched_exact", "matched_wall"}


def custom_rational_sqdiff(x):
    """x0*x1 / (x0^2 - x1^2)^2 over a pole-avoiding box."""
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    return x0 * x1 / ((x0**2 - x1**2) ** 2)


def eq026_nested_recip(x):
    """AI-Feynman eq026: 1 / (x2/x1 + 1/x0)."""
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]
    return 1.0 / (x2 / x1 + 1.0 / x0)


HARNESS_EXTRA_TARGETS: dict[str, dict[str, Any]] = {
    "custom_rational_sqdiff": {
        "nvars": 2,
        "fn": custom_rational_sqdiff,
        "y_dims": (0.0, 0.0, 0.0, 0.0, 0.0),
        "var_dims": [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        ],
        "lo": [4.0, 1.0],
        "hi": [6.0, 3.0],
    },
    "eq026_nested_recip": {
        "nvars": 3,
        "fn": eq026_nested_recip,
        "y_dims": (1.0, 0.0, 0.0, 0.0, 0.0),
        "var_dims": [
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        ],
        "lo": [1.0, 1.0, 1.0],
        "hi": [5.0, 5.0, 5.0],
    },
}


@dataclass
class RunSummary:
    label: str
    target: str
    seed: int
    profile: str
    n_iter: int
    max_depth: int
    best_eff_mse: float
    best_raw_mse: float
    best_expr: str
    residual_basins: int
    n_eval: int
    elapsed_s: float
    solve_hit: bool = False
    action_counts: dict[str, int] = field(default_factory=dict)
    route_usage: dict[str, int] = field(default_factory=dict)
    exact_eval_count: int = 0
    preview_eval_count: int = 0
    inverse_gate_considered: int = 0
    inverse_gate_allowed: int = 0
    repair_considered: int = 0
    repair_selected: int = 0
    repair_option_selected: int = 0
    parent_repair_selected: int = 0
    repair_no_candidate: int = 0
    macro_selected: int = 0
    macro_repair_selected: int = 0
    macro_fallback_selected: int = 0
    macro_policy_counts: dict[str, int] = field(default_factory=dict)
    scheduler_enabled: bool = False
    scheduler_advisory_only: bool = False
    scheduler_bundle_loaded: bool = False
    scheduler_decision_count: int = 0
    scheduler_scored: int = 0
    scheduler_control_selected: int = 0
    scheduler_fallback_selected: int = 0
    scheduler_route_counts: dict[str, int] = field(default_factory=dict)
    scheduler_mean_confidence: float | None = None
    scheduler_mean_candidate_count: float | None = None
    inverse_status_counts: dict[str, int] = field(default_factory=dict)
    controller_policy_counts: dict[str, int] = field(default_factory=dict)
    search_stdout_tail: list[str] = field(default_factory=list)


@dataclass
class ControllerSuiteSummary:
    target: str
    seed: int
    profile: str
    runs: dict[str, RunSummary] = field(default_factory=dict)


@dataclass
class PairSummary:
    target: str
    seed: int
    profile: str
    baseline: RunSummary
    macro: RunSummary
    ratio_best_eff_mse: float
    ratio_best_raw_mse: float
    delta_best_eff_mse: float
    delta_best_raw_mse: float
    delta_residual_basins: int
    delta_macro_repair_selected: int
    delta_repair_selected: int
    scheduler_advisory: RunSummary | None = None
    scheduler_control: RunSummary | None = None



def _set_torch_threads(threads: int | None) -> None:
    if threads is None:
        return
    try:
        t = max(1, int(threads))
    except Exception:
        return
    try:
        torch.set_num_threads(t)
    except Exception:
        pass




def _resolve_target_spec(target: str) -> dict[str, Any]:
    name = str(target)
    if name in HARNESS_EXTRA_TARGETS:
        return {
            "name": name,
            **dict(HARNESS_EXTRA_TARGETS[name]),
        }
    if name in explorer.TARGET_FUNCS:
        nvars, fn, y_dims, var_dims = explorer.TARGET_FUNCS[name]
        return {
            "name": name,
            "nvars": int(nvars),
            "fn": fn,
            "y_dims": y_dims,
            "var_dims": var_dims,
            "lo": 1.0,
            "hi": 5.0,
        }
    raise KeyError(f"unknown target: {target!r}")


def _tail_lines(text: str, n: int = 8) -> list[str]:
    rows = [str(line) for line in str(text).splitlines() if str(line).strip()]
    if len(rows) <= n:
        return rows
    return rows[-n:]



def _profile_overrides(profile: str, *, macro_enabled: bool) -> dict[str, Any]:
    name = str(profile or "default").strip().lower()
    if name in ("", "default", "dropin"):
        out = {
            "inverse_steering_enable": True,
            "repair_controller_enable": True,
            "inverse_experiment_log_enable": True,
        }
        if macro_enabled:
            out["macro_controller_enable"] = True
        return out
    if name in ("repair_probe", "probe", "aggressive"):
        out = {
            "inverse_steering_enable": True,
            "repair_controller_enable": True,
            "inverse_experiment_log_enable": True,
            "repair_controller_focus_prob": 1.0,
            "repair_controller_min_score": 0.0,
            "repair_controller_min_concentration": 0.0,
            "repair_controller_adaptive": False,
        }
        if macro_enabled:
            out.update({
                "macro_controller_enable": True,
                "macro_controller_repair_bonus": 4.0,
                "macro_controller_repair_margin_scale": 0.0,
                "macro_controller_build_bias": -0.10,
                "macro_controller_inverse_bonus": 0.0,
            })
        return out
    raise ValueError(f"unknown profile: {profile!r}")


def _route_usage_from_action_counts(action_counts: Mapping[str, Any]) -> dict[str, int]:
    build_actions = {
        "replace",
        "wrap_un",
        "add_rand",
        "mul_rand",
        "residual",
        "boost",
        "prune",
        "crossover",
    }
    out = {
        "build": 0,
        "inverse": 0,
        "repair": 0,
        "hole": 0,
    }
    for key, value in dict(action_counts or {}).items():
        name = str(key or "")
        count = max(0, int(value))
        if name in build_actions:
            out["build"] += count
        elif name == "inv_steer":
            out["inverse"] += count
        elif name == "repair_option":
            out["repair"] += count
        elif name == "hole_search":
            out["hole"] += count
    return {str(k): int(v) for k, v in out.items()}


def _observed_cost_totals(rows: Sequence[Any]) -> tuple[int, int]:
    exact_total = 0
    preview_total = 0
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        exact_total += max(0, int(row.get("observed_exact_evals", 0) or 0))
        preview_total += max(0, int(row.get("observed_preview_evals", 0) or 0))
    return int(exact_total), int(preview_total)


def _scheduler_log_means(rows: Sequence[Any]) -> tuple[float | None, float | None]:
    confidences: list[float] = []
    candidate_counts: list[float] = []
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        try:
            conf = float(row.get("scheduler_confidence", float("nan")))
        except Exception:
            conf = float("nan")
        if math.isfinite(conf):
            confidences.append(float(conf))
        try:
            count = float(row.get("scheduler_candidate_count", float("nan")))
        except Exception:
            count = float("nan")
        if math.isfinite(count):
            candidate_counts.append(float(count))
    mean_conf = None if not confidences else float(sum(confidences) / len(confidences))
    mean_count = None if not candidate_counts else float(sum(candidate_counts) / len(candidate_counts))
    return mean_conf, mean_count


def _normalize_comparison_mode(mode: str | None) -> str:
    token = str(mode or DEFAULT_COMPARISON_MODE).strip().lower()
    aliases = {
        "budget": "matched_exact",
        "exact": "matched_exact",
        "matched_budget": "matched_exact",
        "time": "matched_wall",
        "wall": "matched_wall",
        "matched_time": "matched_wall",
    }
    token = aliases.get(token, token)
    if token not in VALID_COMPARISON_MODES:
        raise ValueError(f"unknown comparison_mode: {mode!r}")
    return token


def _safe_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return float(out)


def _delta_log_eff(candidate_mse: Any, baseline_mse: Any, *, eps: float = 1.0e-30) -> float | None:
    cand = _safe_positive(candidate_mse)
    base = _safe_positive(baseline_mse)
    if cand is None or base is None:
        return None
    try:
        return float(math.log(float(base) + float(eps)) - math.log(float(cand) + float(eps)))
    except Exception:
        return None


def _comparison_cost(run: RunSummary, *, comparison_mode: str) -> float | None:
    mode = _normalize_comparison_mode(comparison_mode)
    if mode == "matched_wall":
        return _safe_positive(run.elapsed_s)
    if mode == "matched_exact":
        return _safe_positive(run.exact_eval_count)
    return 1.0


def _pairwise_comparison_record(
    baseline: RunSummary,
    candidate: RunSummary,
    *,
    comparison_mode: str,
) -> dict[str, Any]:
    mode = _normalize_comparison_mode(comparison_mode)
    delta_log_eff = _delta_log_eff(candidate.best_eff_mse, baseline.best_eff_mse)
    baseline_cost = _comparison_cost(baseline, comparison_mode=mode)
    candidate_cost = _comparison_cost(candidate, comparison_mode=mode)
    if mode == "raw":
        discount = 1.0
    elif baseline_cost is None or candidate_cost is None:
        discount = None
    else:
        discount = float(min(1.0, float(baseline_cost) / max(float(candidate_cost), 1.0e-30)))
    matched_delta = None
    if delta_log_eff is not None:
        matched_delta = float(delta_log_eff if discount is None else float(delta_log_eff) * float(discount))
    return {
        "target": str(candidate.target),
        "seed": int(candidate.seed),
        "baseline_arm": str(baseline.label),
        "candidate_arm": str(candidate.label),
        "comparison_mode": str(mode),
        "baseline_cost": baseline_cost,
        "candidate_cost": candidate_cost,
        "candidate_cost_ratio": None
        if baseline_cost is None or candidate_cost is None
        else float(candidate_cost) / max(float(baseline_cost), 1.0e-30),
        "cost_discount": discount,
        "baseline_solve_hit": bool(baseline.solve_hit),
        "candidate_solve_hit": bool(candidate.solve_hit),
        "baseline_best_eff_mse": float(baseline.best_eff_mse),
        "candidate_best_eff_mse": float(candidate.best_eff_mse),
        "delta_log_eff": delta_log_eff,
        "matched_delta_log_eff": matched_delta,
        "raw_candidate_win": bool(delta_log_eff is not None and float(delta_log_eff) > 0.0),
        "matched_candidate_win": bool(matched_delta is not None and float(matched_delta) > 0.0),
    }


def _summarize_pairwise_comparisons(
    baseline_runs: Sequence[RunSummary],
    candidate_runs: Sequence[RunSummary],
    *,
    comparison_mode: str,
) -> dict[str, Any]:
    mode = _normalize_comparison_mode(comparison_mode)
    baseline_index = {
        (str(run.target), int(run.seed)): run
        for run in baseline_runs
    }
    candidate_index = {
        (str(run.target), int(run.seed)): run
        for run in candidate_runs
    }
    records = [
        _pairwise_comparison_record(
            baseline_index[key],
            candidate_index[key],
            comparison_mode=mode,
        )
        for key in sorted(set(baseline_index).intersection(candidate_index))
    ]
    delta_values = [
        float(row["delta_log_eff"])
        for row in records
        if row.get("delta_log_eff", None) is not None and math.isfinite(float(row["delta_log_eff"]))
    ]
    matched_values = [
        float(row["matched_delta_log_eff"])
        for row in records
        if row.get("matched_delta_log_eff", None) is not None and math.isfinite(float(row["matched_delta_log_eff"]))
    ]
    cost_ratios = [
        float(row["candidate_cost_ratio"])
        for row in records
        if row.get("candidate_cost_ratio", None) is not None and math.isfinite(float(row["candidate_cost_ratio"]))
    ]
    return {
        "comparison_mode": str(mode),
        "n_pairs": int(len(records)),
        "mean_delta_log_eff": None if not delta_values else float(sum(delta_values) / len(delta_values)),
        "median_delta_log_eff": None if not delta_values else _median(delta_values),
        "mean_matched_delta_log_eff": None if not matched_values else float(sum(matched_values) / len(matched_values)),
        "median_matched_delta_log_eff": None if not matched_values else _median(matched_values),
        "raw_win_rate": 0.0
        if not records
        else float(sum(1 for row in records if bool(row.get("raw_candidate_win", False)))) / float(len(records)),
        "matched_win_rate": 0.0
        if not records
        else float(sum(1 for row in records if bool(row.get("matched_candidate_win", False)))) / float(len(records)),
        "mean_candidate_cost_ratio": None if not cost_ratios else float(sum(cost_ratios) / len(cost_ratios)),
        "records": records,
    }



def _summarize_arch(
    arch,
    *,
    label: str,
    target: str,
    seed: int,
    profile: str,
    n_iter: int,
    max_depth: int,
    elapsed_s: float,
    stdout_text: str,
    solve_mse: float = 1.0e-10,
) -> RunSummary:
    if arch.d:
        best = arch.best(1)[0]
        best_eff_mse = float(getattr(best, "best_mse", float("inf")))
        best_raw_mse = float(getattr(best, "best_raw_mse", best_eff_mse))
        best_expr = explorer.node_str(getattr(best, "best_expr", None))
    else:
        best_eff_mse = float("inf")
        best_raw_mse = float("inf")
        best_expr = ""

    ad = getattr(arch, "action_distribution", {}) or {}
    action_counts = {str(k): int(v) for k, v in dict(ad.get("counts", {}) or {}).items()}
    route_usage = _route_usage_from_action_counts(action_counts)

    igs = getattr(arch, "inverse_gate_stats", {}) or {}
    rcs = getattr(arch, "repair_controller_stats", {}) or {}
    mcs = getattr(arch, "macro_controller_stats", {}) or {}
    scs = getattr(arch, "scheduler_stats", {}) or {}
    log = list(getattr(arch, "inverse_experiment_log", []) or [])
    scheduler_log = list(getattr(arch, "scheduler_decision_log", []) or [])
    exact_eval_count, preview_eval_count = _observed_cost_totals(log)
    scheduler_mean_confidence, scheduler_mean_candidate_count = _scheduler_log_means(scheduler_log)

    status_counts = Counter(str(row.get("status", "")) for row in log if str(row.get("status", "")))
    policy_counts = Counter(str(row.get("controller_policy_action", "")) for row in log if str(row.get("controller_policy_action", "")))

    return RunSummary(
        label=str(label),
        target=str(target),
        seed=int(seed),
        profile=str(profile),
        n_iter=int(n_iter),
        max_depth=int(max_depth),
        best_eff_mse=float(best_eff_mse),
        best_raw_mse=float(best_raw_mse),
        best_expr=str(best_expr),
        residual_basins=int(len(getattr(arch, "d", {}) or {})),
        n_eval=int(getattr(arch, "n_eval", 0)),
        elapsed_s=float(elapsed_s),
        solve_hit=bool(math.isfinite(best_eff_mse) and float(best_eff_mse) <= float(solve_mse)),
        action_counts=action_counts,
        route_usage=route_usage,
        exact_eval_count=int(exact_eval_count),
        preview_eval_count=int(preview_eval_count),
        inverse_gate_considered=int(igs.get("considered", 0)),
        inverse_gate_allowed=int(igs.get("allowed", 0)),
        repair_considered=int(rcs.get("considered", 0)),
        repair_selected=int(rcs.get("selected", 0)),
        repair_option_selected=int(rcs.get("option_repair_selected", 0)),
        parent_repair_selected=int(rcs.get("parent_repair_selected", 0)),
        repair_no_candidate=int(rcs.get("no_candidate", 0)),
        macro_selected=int(mcs.get("selected", 0)),
        macro_repair_selected=int(mcs.get("repair_selected", 0)),
        macro_fallback_selected=int(mcs.get("fallback_selected", 0)),
        macro_policy_counts={str(k): int(v) for k, v in dict(mcs.get("policy_counts", {}) or {}).items()},
        scheduler_enabled=bool(scs.get("enabled", False)),
        scheduler_advisory_only=bool(scs.get("advisory_only", False)),
        scheduler_bundle_loaded=bool(scs.get("bundle_loaded", False)),
        scheduler_decision_count=int(scs.get("decision_count", 0)),
        scheduler_scored=int(scs.get("scored", 0)),
        scheduler_control_selected=int(scs.get("control_selected", 0)),
        scheduler_fallback_selected=int(scs.get("fallback_selected", 0)),
        scheduler_route_counts={str(k): int(v) for k, v in dict(scs.get("route_counts", {}) or {}).items()},
        scheduler_mean_confidence=scheduler_mean_confidence,
        scheduler_mean_candidate_count=scheduler_mean_candidate_count,
        inverse_status_counts={str(k): int(v) for k, v in status_counts.items()},
        controller_policy_counts={str(k): int(v) for k, v in policy_counts.items()},
        search_stdout_tail=_tail_lines(stdout_text, n=8),
    )



def run_single_experiment(
    target: str,
    *,
    seed: int,
    label: str,
    macro_enabled: bool,
    profile: str = "default",
    arm_cfg: dict[str, Any] | None = None,
    n_iter: int = 120,
    max_depth: int = 5,
    brute_depth: int = 0,
    dtype: torch.dtype = torch.float64,
    capture_search_output: bool = True,
    threads: int | None = 1,
    solve_mse: float = 1.0e-10,
) -> RunSummary:
    _set_torch_threads(threads)
    target_spec = _resolve_target_spec(str(target))
    kwargs = {
        "target_fn": target_spec["fn"],
        "nvars": int(target_spec["nvars"]),
        "n_iter": int(n_iter),
        "max_depth": int(max_depth),
        "brute_depth": int(brute_depth),
        "seed": int(seed),
        "dtype": dtype,
        "print_every": 0,
        "lo": target_spec.get("lo", 1.0),
        "hi": target_spec.get("hi", 5.0),
        "y_dims": target_spec.get("y_dims", None),
        "var_dims": target_spec.get("var_dims", None),
    }
    merged_arm_cfg = dict(_profile_overrides(profile, macro_enabled=bool(macro_enabled)))
    merged_arm_cfg.update(dict(arm_cfg or {}))
    kwargs.update(merged_arm_cfg)
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
    return _summarize_arch(
        arch,
        label=str(label),
        target=str(target),
        seed=int(seed),
        profile=str(profile),
        n_iter=int(n_iter),
        max_depth=int(max_depth),
        elapsed_s=float(elapsed_s),
        stdout_text=stdout_text,
        solve_mse=float(solve_mse),
    )


def _controller_arm_configs(
    *,
    scheduler_bundle_path: str = "",
    scheduler_budget_ladder: Sequence[int] | None = None,
    scheduler_arm_overrides: Mapping[str, Any] | None = None,
) -> list[tuple[str, bool, dict[str, Any]]]:
    arms: list[tuple[str, bool, dict[str, Any]]] = [
        ("baseline", False, {}),
        ("macro", True, {}),
    ]
    bundle_path = str(scheduler_bundle_path or "").strip()
    if not bundle_path:
        return arms
    scheduler_common: dict[str, Any] = {
        "scheduler_enable": True,
        "scheduler_bundle_path": str(bundle_path),
    }
    if scheduler_budget_ladder is not None:
        scheduler_common["scheduler_budget_ladder"] = [int(v) for v in list(scheduler_budget_ladder)]
    if scheduler_arm_overrides is not None:
        scheduler_common.update(dict(scheduler_arm_overrides))
    arms.append(
        (
            "scheduler_advisory",
            True,
            {
                **scheduler_common,
                "scheduler_advisory_only": True,
            },
        )
    )
    arms.append(
        (
            "scheduler_control",
            True,
            {
                **scheduler_common,
                "scheduler_advisory_only": False,
            },
        )
    )
    return arms


def run_controller_suite(
    target: str,
    *,
    seed: int,
    profile: str = "default",
    scheduler_bundle_path: str = "",
    scheduler_budget_ladder: Sequence[int] | None = None,
    scheduler_arm_overrides: Mapping[str, Any] | None = None,
    n_iter: int = 120,
    max_depth: int = 5,
    brute_depth: int = 0,
    dtype: torch.dtype = torch.float64,
    capture_search_output: bool = True,
    threads: int | None = 1,
    solve_mse: float = 1.0e-10,
) -> ControllerSuiteSummary:
    runs: dict[str, RunSummary] = {}
    for label, macro_enabled, arm_cfg in _controller_arm_configs(
        scheduler_bundle_path=str(scheduler_bundle_path or ""),
        scheduler_budget_ladder=scheduler_budget_ladder,
        scheduler_arm_overrides=scheduler_arm_overrides,
    ):
        runs[str(label)] = run_single_experiment(
            target,
            seed=int(seed),
            label=str(label),
            macro_enabled=bool(macro_enabled),
            profile=profile,
            arm_cfg=arm_cfg,
            n_iter=n_iter,
            max_depth=max_depth,
            brute_depth=brute_depth,
            dtype=dtype,
            capture_search_output=capture_search_output,
            threads=threads,
            solve_mse=float(solve_mse),
        )
    return ControllerSuiteSummary(
        target=str(target),
        seed=int(seed),
        profile=str(profile),
        runs=runs,
    )



def _safe_ratio(num: float, den: float) -> float:
    den_f = float(den)
    if not math.isfinite(den_f) or den_f == 0.0:
        den_f = 1.0
    return float(num) / den_f



def run_controller_pair(
    target: str,
    *,
    seed: int,
    profile: str = "default",
    scheduler_bundle_path: str = "",
    scheduler_budget_ladder: Sequence[int] | None = None,
    scheduler_arm_overrides: Mapping[str, Any] | None = None,
    n_iter: int = 120,
    max_depth: int = 5,
    brute_depth: int = 0,
    dtype: torch.dtype = torch.float64,
    capture_search_output: bool = True,
    threads: int | None = 1,
    solve_mse: float = 1.0e-10,
) -> PairSummary:
    suite = run_controller_suite(
        target,
        seed=seed,
        profile=profile,
        scheduler_bundle_path=str(scheduler_bundle_path or ""),
        scheduler_budget_ladder=scheduler_budget_ladder,
        scheduler_arm_overrides=scheduler_arm_overrides,
        n_iter=n_iter,
        max_depth=max_depth,
        brute_depth=brute_depth,
        dtype=dtype,
        capture_search_output=capture_search_output,
        threads=threads,
        solve_mse=float(solve_mse),
    )
    baseline = suite.runs["baseline"]
    macro = suite.runs["macro"]
    return PairSummary(
        target=str(target),
        seed=int(seed),
        profile=str(profile),
        baseline=baseline,
        macro=macro,
        ratio_best_eff_mse=_safe_ratio(macro.best_eff_mse, max(1.0e-30, baseline.best_eff_mse)),
        ratio_best_raw_mse=_safe_ratio(macro.best_raw_mse, max(1.0e-30, baseline.best_raw_mse)),
        delta_best_eff_mse=float(macro.best_eff_mse - baseline.best_eff_mse),
        delta_best_raw_mse=float(macro.best_raw_mse - baseline.best_raw_mse),
        delta_residual_basins=int(macro.residual_basins - baseline.residual_basins),
        delta_macro_repair_selected=int(macro.macro_repair_selected - baseline.macro_repair_selected),
        delta_repair_selected=int(macro.repair_selected - baseline.repair_selected),
        scheduler_advisory=suite.runs.get("scheduler_advisory", None),
        scheduler_control=suite.runs.get("scheduler_control", None),
    )



def _median(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _aggregate_arm_runs(runs: Sequence[RunSummary], *, solve_mse: float) -> dict[str, Any]:
    route_totals: Counter[str] = Counter()
    scheduler_route_totals: Counter[str] = Counter()
    scheduler_confidences = [
        float(run.scheduler_mean_confidence)
        for run in runs
        if run.scheduler_mean_confidence is not None and math.isfinite(float(run.scheduler_mean_confidence))
    ]
    for run in runs:
        route_totals.update({str(k): int(v) for k, v in dict(run.route_usage or {}).items()})
        scheduler_route_totals.update({str(k): int(v) for k, v in dict(run.scheduler_route_counts or {}).items()})
    return {
        "n_runs": int(len(runs)),
        "solve_rate": 0.0 if not runs else float(sum(1 for run in runs if run.solve_hit)) / float(len(runs)),
        "median_eff_mse": _median([run.best_eff_mse for run in runs]),
        "median_raw_mse": _median([run.best_raw_mse for run in runs]),
        "mean_wall_s": 0.0 if not runs else float(sum(run.elapsed_s for run in runs) / len(runs)),
        "mean_exact_eval_count": 0.0 if not runs else float(sum(run.exact_eval_count for run in runs) / len(runs)),
        "mean_preview_eval_count": 0.0 if not runs else float(sum(run.preview_eval_count for run in runs) / len(runs)),
        "mean_repair_selected": 0.0 if not runs else float(sum(run.repair_selected for run in runs) / len(runs)),
        "mean_macro_selected": 0.0 if not runs else float(sum(run.macro_selected for run in runs) / len(runs)),
        "mean_scheduler_decision_count": 0.0 if not runs else float(sum(run.scheduler_decision_count for run in runs) / len(runs)),
        "mean_scheduler_control_selected": 0.0 if not runs else float(sum(run.scheduler_control_selected for run in runs) / len(runs)),
        "mean_scheduler_confidence": None
        if not scheduler_confidences
        else float(sum(scheduler_confidences) / len(scheduler_confidences)),
        "route_usage": {str(k): int(v) for k, v in sorted(route_totals.items())},
        "scheduler_route_usage": {str(k): int(v) for k, v in sorted(scheduler_route_totals.items())},
        "solve_mse": float(solve_mse),
    }



def run_controller_benchmark(
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    profile: str = "default",
    scheduler_bundle_path: str = "",
    scheduler_budget_ladder: Sequence[int] | None = None,
    scheduler_arm_overrides: Mapping[str, Any] | None = None,
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    n_iter: int = 120,
    max_depth: int = 5,
    brute_depth: int = 0,
    solve_mse: float = 1.0e-10,
    dtype: torch.dtype = torch.float64,
    capture_search_output: bool = True,
    threads: int | None = 1,
) -> dict[str, Any]:
    normalized_comparison_mode = _normalize_comparison_mode(comparison_mode)
    suites: list[ControllerSuiteSummary] = []
    pairs: list[PairSummary] = []
    for target in targets:
        for seed in seeds:
            suite = run_controller_suite(
                str(target),
                seed=int(seed),
                profile=profile,
                scheduler_bundle_path=str(scheduler_bundle_path or ""),
                scheduler_budget_ladder=scheduler_budget_ladder,
                scheduler_arm_overrides=scheduler_arm_overrides,
                n_iter=n_iter,
                max_depth=max_depth,
                brute_depth=brute_depth,
                dtype=dtype,
                capture_search_output=capture_search_output,
                threads=threads,
                solve_mse=float(solve_mse),
            )
            suites.append(suite)
            baseline = suite.runs["baseline"]
            macro = suite.runs["macro"]
            pairs.append(
                PairSummary(
                    target=str(target),
                    seed=int(seed),
                    profile=str(profile),
                    baseline=baseline,
                    macro=macro,
                    ratio_best_eff_mse=_safe_ratio(macro.best_eff_mse, max(1.0e-30, baseline.best_eff_mse)),
                    ratio_best_raw_mse=_safe_ratio(macro.best_raw_mse, max(1.0e-30, baseline.best_raw_mse)),
                    delta_best_eff_mse=float(macro.best_eff_mse - baseline.best_eff_mse),
                    delta_best_raw_mse=float(macro.best_raw_mse - baseline.best_raw_mse),
                    delta_residual_basins=int(macro.residual_basins - baseline.residual_basins),
                    delta_macro_repair_selected=int(macro.macro_repair_selected - baseline.macro_repair_selected),
                    delta_repair_selected=int(macro.repair_selected - baseline.repair_selected),
                    scheduler_advisory=suite.runs.get("scheduler_advisory", None),
                    scheduler_control=suite.runs.get("scheduler_control", None),
                )
            )

    eff_ratios = [p.ratio_best_eff_mse for p in pairs]
    raw_ratios = [p.ratio_best_raw_mse for p in pairs]
    delta_residual_basins = [float(p.delta_residual_basins) for p in pairs]
    macro_repair = [int(p.macro.macro_repair_selected) for p in pairs]
    baseline_repair = [int(p.baseline.repair_option_selected) for p in pairs]

    aggregate = {
        "n_pairs": int(len(pairs)),
        "profile": str(profile),
        "median_eff_mse_ratio_macro_over_baseline": _median(eff_ratios),
        "median_raw_mse_ratio_macro_over_baseline": _median(raw_ratios),
        "macro_eff_mse_wins": int(sum(1 for v in eff_ratios if math.isfinite(v) and v < 1.0)),
        "macro_raw_mse_wins": int(sum(1 for v in raw_ratios if math.isfinite(v) and v < 1.0)),
        "median_delta_residual_basins": _median(delta_residual_basins),
        "macro_runs_with_repair_option": int(sum(1 for v in macro_repair if int(v) > 0)),
        "baseline_runs_with_repair_option": int(sum(1 for v in baseline_repair if int(v) > 0)),
        "total_macro_repair_selected": int(sum(macro_repair)),
        "total_baseline_repair_option_selected": int(sum(baseline_repair)),
    }
    arm_labels = [
        label
        for label, _macro_enabled, _arm_cfg in _controller_arm_configs(
            scheduler_bundle_path=str(scheduler_bundle_path or ""),
            scheduler_budget_ladder=scheduler_budget_ladder,
            scheduler_arm_overrides=scheduler_arm_overrides,
        )
    ]
    arm_overall = {
        str(label): _aggregate_arm_runs(
            [suite.runs[str(label)] for suite in suites if str(label) in suite.runs],
            solve_mse=float(solve_mse),
        )
        for label in arm_labels
    }
    baseline_runs = [suite.runs["baseline"] for suite in suites if "baseline" in suite.runs]
    comparisons = {
        str(label): _summarize_pairwise_comparisons(
            baseline_runs,
            [suite.runs[str(label)] for suite in suites if str(label) in suite.runs],
            comparison_mode=normalized_comparison_mode,
        )
        for label in arm_labels
        if str(label) != "baseline"
    }
    return {
        "config": {
            "targets": [str(t) for t in targets],
            "seeds": [int(s) for s in seeds],
            "profile": str(profile),
            "scheduler_bundle_path": str(scheduler_bundle_path or ""),
            "scheduler_budget_ladder": None
            if scheduler_budget_ladder is None
            else [int(v) for v in list(scheduler_budget_ladder)],
            "scheduler_arm_overrides": dict(scheduler_arm_overrides or {}),
            "comparison_mode": str(normalized_comparison_mode),
            "n_iter": int(n_iter),
            "max_depth": int(max_depth),
            "brute_depth": int(brute_depth),
            "solve_mse": float(solve_mse),
            "threads": None if threads is None else int(threads),
            "dtype": str(dtype),
        },
        "aggregate": aggregate,
        "arm_overall": arm_overall,
        "comparisons": comparisons,
        "pairs": [asdict(p) for p in pairs],
        "suites": [asdict(suite) for suite in suites],
    }



def _format_run_line(run: RunSummary) -> str:
    return (
        f"{run.label:8s} best_eff={run.best_eff_mse:.4g} best_raw={run.best_raw_mse:.4g} "
        f"residual_basins={run.residual_basins:3d} eval={run.n_eval:4d} t={run.elapsed_s:.2f}s "
        f"repair={run.repair_selected:2d} macro_repair={run.macro_repair_selected:2d} "
        f"exact={run.exact_eval_count:3d} sched={run.scheduler_decision_count:2d}"
    )



def _print_human_report(report: dict[str, Any]) -> None:
    print("\n=== Controller benchmark summary ===")
    agg = dict(report.get("aggregate", {}) or {})
    cfg = dict(report.get("config", {}) or {})
    print(
        f"profile={cfg.get('profile')} targets={cfg.get('targets')} seeds={cfg.get('seeds')} "
        f"n_iter={cfg.get('n_iter')} max_depth={cfg.get('max_depth')} "
        f"comparison_mode={cfg.get('comparison_mode')}"
    )
    print(
        "aggregate: "
        f"median_eff_ratio={agg.get('median_eff_mse_ratio_macro_over_baseline')} "
        f"median_raw_ratio={agg.get('median_raw_mse_ratio_macro_over_baseline')} "
        f"macro_eff_wins={agg.get('macro_eff_mse_wins')}/{agg.get('n_pairs')} "
        f"macro_raw_wins={agg.get('macro_raw_mse_wins')}/{agg.get('n_pairs')} "
        f"macro_repair_runs={agg.get('macro_runs_with_repair_option')} "
        f"baseline_repair_runs={agg.get('baseline_runs_with_repair_option')}"
    )
    arm_overall = dict(report.get("arm_overall", {}) or {})
    if arm_overall:
        print("arms:")
        for arm_name, stats in arm_overall.items():
            print(
                f"  {arm_name}: solve_rate={stats.get('solve_rate')} "
                f"median_eff={stats.get('median_eff_mse')} "
                f"mean_wall_s={stats.get('mean_wall_s')} "
                f"mean_exact={stats.get('mean_exact_eval_count')} "
                f"mean_sched={stats.get('mean_scheduler_decision_count')} "
                f"routes={stats.get('route_usage')}"
            )
    comparisons = dict(report.get("comparisons", {}) or {})
    if comparisons:
        print("comparisons:")
        for arm_name, stats in comparisons.items():
            print(
                f"  {arm_name}: mode={stats.get('comparison_mode')} "
                f"matched_win_rate={stats.get('matched_win_rate')} "
                f"mean_matched_delta_log_eff={stats.get('mean_matched_delta_log_eff')} "
                f"mean_cost_ratio={stats.get('mean_candidate_cost_ratio')}"
            )
    for pair in report.get("pairs", []):
        target = pair.get("target")
        seed = pair.get("seed")
        print(f"\n--- {target} seed={seed} ---")
        baseline = RunSummary(**pair["baseline"])
        macro = RunSummary(**pair["macro"])
        print(_format_run_line(baseline))
        print(_format_run_line(macro))
        if pair.get("scheduler_advisory", None):
            print(_format_run_line(RunSummary(**pair["scheduler_advisory"])))
        if pair.get("scheduler_control", None):
            print(_format_run_line(RunSummary(**pair["scheduler_control"])))
        print(
            f"delta_eff={pair.get('delta_best_eff_mse'):.4g} "
            f"ratio_eff={pair.get('ratio_best_eff_mse'):.4g} "
            f"delta_residual_basins={pair.get('delta_residual_basins')}"
        )
        tail_payload = pair.get("scheduler_control") or pair.get("scheduler_advisory") or pair.get("macro")
        tail_run = RunSummary(**tail_payload) if isinstance(tail_payload, dict) else macro
        if tail_run.search_stdout_tail:
            print(f"{tail_run.label} tail:")
            for line in tail_run.search_stdout_tail:
                print(f"  {line}")



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--profile", type=str, default="default", choices=["default", "repair_probe"])
    p.add_argument("--scheduler_bundle_path", type=str, default="")
    p.add_argument("--scheduler_budget_ladder", type=str, default="1,2,4,8")
    p.add_argument("--comparison_mode", type=str, default=DEFAULT_COMPARISON_MODE, choices=sorted(VALID_COMPARISON_MODES))
    p.add_argument("--n_iter", type=int, default=120)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--brute_depth", type=int, default=0)
    p.add_argument("--solve_mse", type=float, default=1.0e-10)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no_capture", action="store_true", help="let inner search print directly")
    args = p.parse_args()

    report = run_controller_benchmark(
        targets=args.targets,
        seeds=args.seeds,
        profile=args.profile,
        scheduler_bundle_path=str(args.scheduler_bundle_path),
        scheduler_budget_ladder=[
            int(s.strip())
            for s in str(args.scheduler_budget_ladder).split(",")
            if s.strip()
        ],
        comparison_mode=str(args.comparison_mode),
        n_iter=int(args.n_iter),
        max_depth=int(args.max_depth),
        brute_depth=int(args.brute_depth),
        solve_mse=float(args.solve_mse),
        capture_search_output=not bool(args.no_capture),
        threads=int(args.threads),
    )
    if bool(args.json):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report)


if __name__ == "__main__":
    main()
