# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Y-transform controller utilities for Stage A exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class YSearchState:
    """Search state keyed by committed y-transform stack."""

    y_stack: tuple[str, ...]


@dataclass(frozen=True)
class YSearchControllerConfig:
    max_depth: int = 1
    beam: int = 3
    expand_k: int = 2
    confirm_improve_ratio: float = 0.3
    eps_parent_loss: float = 1.0e-12
    # Hard cap on expensive state evaluations (0 => unbounded).
    max_state_evals: int = 0
    # Optional split-recursion controls (0 => disabled).
    max_recursive_branches: int = 0
    max_split_plans_per_state: int = 1


@dataclass
class YSearchTrial:
    name: str
    state: YSearchState
    val_loss_base: float
    split_success: bool
    strong_structure_trigger: bool
    accept_branch: bool
    payload: object


@dataclass
class YSearchResult:
    best_trial: Optional[YSearchTrial]
    accepted_trials: List[YSearchTrial]
    all_trials: List[YSearchTrial]
    frontier_trials: List[YSearchTrial]
    state_evals: int = 0
    budget_exhausted: bool = False
    recursive_calls: int = 0


@dataclass(frozen=True)
class StageAStateKey:
    """Cache key for Stage A analyses in y-search states."""

    y_stack_sig: tuple
    data_sig: tuple
    model_sig: tuple
    train_cfg_sig: tuple
    seed: int
    fast: bool


def make_stagea_state_key(
    *,
    y_stack_sig,
    data_sig,
    model_sig,
    train_cfg_sig,
    seed: int = 0,
    fast: bool = False,
) -> StageAStateKey:
    return StageAStateKey(
        y_stack_sig=tuple(y_stack_sig),
        data_sig=tuple(data_sig),
        model_sig=tuple(model_sig),
        train_cfg_sig=tuple(train_cfg_sig),
        seed=int(seed),
        fast=bool(fast),
    )


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _default_trigger(payload) -> bool:
    try:
        return bool(payload.get("split_success", False))
    except Exception:
        return False


def _confirmation_status(payload) -> str:
    if not isinstance(payload, dict):
        return "unresolved"
    status = str(
        payload.get(
            "branch_confirmation",
            payload.get("branch_confirmation_status", ""),
        )
        or ""
    )
    confirmed_statuses = {
        "outer_affine_confirmed",
        "split_confirmed",
        "stageB_confirmed",
        "analytic_rewrite_confirmed",
    }
    if status in confirmed_statuses:
        return status
    try:
        if bool(payload.get("split_success", False)):
            return "split_confirmed"
    except Exception:
        pass
    sig = payload.get("stagea_signals", {})
    if isinstance(sig, dict):
        for key, name in (
            ("outer_affine_confirmed", "outer_affine_confirmed"),
            ("stageB_confirmed", "stageB_confirmed"),
            ("analytic_rewrite_confirmed", "analytic_rewrite_confirmed"),
            ("split_success", "split_confirmed"),
        ):
            try:
                if float(sig.get(key, 0.0)) >= 0.5:
                    return name
            except Exception:
                continue
        for key in ("full_compound_compressed", "full_compound_solved"):
            try:
                if float(sig.get(key, 0.0)) >= 0.5:
                    return "provisional"
            except Exception:
                continue
    if status in {"provisional", "unresolved"}:
        return status
    return "unresolved"


def _confirmation_rank(payload) -> int:
    status = _confirmation_status(payload)
    if status in {"outer_affine_confirmed", "stageB_confirmed", "analytic_rewrite_confirmed", "split_confirmed"}:
        return 0
    if status == "provisional":
        return 1
    return 2


def _fallback_key(t: "YSearchTrial"):
    """Structure-aware key used when no trial is formally accepted, and
    also as a global ordering in split-recursion best-selection.

    When no trial is formally accepted (e.g. parent already at ~1e-11),
    raw val_loss_base is misleading — a slightly lower loss does not mean
    the transform is actually correct.  This key prefers:

    0. accept_branch   (accepted always beats rejected)
    1. branch confirmation status (certificate beats provisional hints)
    2. split_success (structural decomposition found)
    3. strong_structure_trigger (confirmed progress)
    4. inv_branch_ok  (y-data falls in the principal branch of the inverse)
    5. val_loss_base   (tie-break)
    6. name            (deterministic)
    """
    payload = t.payload if isinstance(t.payload, dict) else {}
    inv_ok = 0.0
    try:
        inv_ok = float(payload.get("inv_branch_ok", 1.0))
    except Exception:
        inv_ok = 0.0
    return (
        0 if bool(t.accept_branch) else 1,
        _confirmation_rank(payload),
        0 if bool(t.split_success) else 1,
        0 if bool(t.strong_structure_trigger) else 1,
        0 if inv_ok >= 0.5 else 1,
        float(t.val_loss_base),
        str(getattr(t, "name", getattr(t.state, "y_stack", ""))),
    )


def run_depth1_ysearch(
    *,
    parent_state: YSearchState,
    candidate_names: Iterable[str],
    evaluate_candidate: Callable[[str], Optional[dict]],
    parent_val_loss_base: Optional[float],
    cfg: YSearchControllerConfig,
    strong_structure_trigger_fn: Optional[Callable[[dict], bool]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> YSearchResult:
    """Evaluate depth-1 child transforms with reversible branch acceptance."""
    names = list(candidate_names)
    if int(cfg.max_depth) <= 0 or len(parent_state.y_stack) >= int(cfg.max_depth):
        return YSearchResult(best_trial=None, accepted_trials=[], all_trials=[], frontier_trials=[])

    k = max(1, int(cfg.expand_k))
    names = names[:k]

    trigger_fn = strong_structure_trigger_fn or _default_trigger
    parent_loss = float(parent_val_loss_base) if _finite(parent_val_loss_base) else float("inf")
    eps = max(0.0, float(cfg.eps_parent_loss))
    ratio = float(cfg.confirm_improve_ratio)

    all_trials: List[YSearchTrial] = []
    for name in names:
        payload = evaluate_candidate(str(name))
        if payload is None:
            continue

        val = payload.get("val_loss_base", float("inf"))
        val_f = float(val) if _finite(val) else float("inf")
        split_success = bool(payload.get("split_success", False))
        strong_trigger = bool(trigger_fn(payload))

        if _finite(parent_loss):
            if parent_loss <= eps:
                accept = bool(strong_trigger)
            else:
                accept = bool((val_f <= ratio * max(parent_loss, 1.0e-15)) or strong_trigger)
        else:
            accept = bool(_finite(val_f) or strong_trigger)

        st = YSearchState(y_stack=tuple(parent_state.y_stack) + (str(name),))
        trial = YSearchTrial(
            name=str(name),
            state=st,
            val_loss_base=float(val_f),
            split_success=split_success,
            strong_structure_trigger=strong_trigger,
            accept_branch=accept,
            payload=payload,
        )
        all_trials.append(trial)

        if log_fn is not None:
            sig_msg = ""
            try:
                sig = payload.get("stagea_signals", {}) if isinstance(payload, dict) else {}
                if isinstance(sig, dict):
                    sep_s = sig.get("sep_score", None)
                    split_s = sig.get("best_split_score", None)
                    trig_s = sig.get("trig_affine_conf", None)
                    reasons = (
                        payload.get("ysearch_trigger_reason_str", "")
                        if isinstance(payload, dict)
                        else ""
                    )
                    plans_n = 0
                    try:
                        plans = payload.get("split_plans", None) if isinstance(payload, dict) else None
                        plans_n = len(plans) if isinstance(plans, (list, tuple)) else 0
                    except Exception:
                        plans_n = 0
                    parts = []
                    if _finite(sep_s):
                        parts.append(f"sep={float(sep_s):.3g}")
                    if _finite(split_s):
                        parts.append(f"split_score={float(split_s):.3g}")
                    if _finite(trig_s):
                        parts.append(f"trig={float(trig_s):.3g}")
                    if plans_n > 0:
                        parts.append(f"plans={int(plans_n)}")
                    status = _confirmation_status(payload)
                    if status:
                        parts.append(f"confirm={status}")
                    if reasons:
                        parts.append(f"reasons={str(reasons)}")
                    if parts:
                        sig_msg = ", " + ", ".join(parts)
            except Exception:
                sig_msg = ""
            log_fn(
                "[ysearch] trial "
                f"name={trial.name}, val_loss_base={trial.val_loss_base:.3g}, "
                f"split={int(trial.split_success)}, trigger={int(trial.strong_structure_trigger)}, "
                f"accept={int(trial.accept_branch)}{sig_msg}"
            )

    accepted = [t for t in all_trials if t.accept_branch]
    pool = accepted if accepted else all_trials
    if not pool:
        return YSearchResult(
            best_trial=None,
            accepted_trials=accepted,
            all_trials=all_trials,
            frontier_trials=[],
        )

    # When accepted is non-empty, use plain val_loss_base ordering (the
    # acceptance gate already confirmed structural progress).  When falling
    # back to all_trials, use structure-aware key so that transforms with
    # correct invertibility and structural signals are preferred over those
    # that merely happen to have slightly lower proxy loss.
    if accepted:
        _key = lambda t: (
            _confirmation_rank(t.payload if isinstance(t.payload, dict) else {}),
            float(t.val_loss_base),
            0 if bool(t.split_success) else 1,
            str(t.name),
        )
    else:
        _key = _fallback_key

    best = min(pool, key=_key)
    ranked = sorted(pool, key=_key)
    beam = max(1, int(getattr(cfg, "beam", 3)))
    frontier = ranked[:beam]
    return YSearchResult(
        best_trial=best,
        accepted_trials=accepted,
        all_trials=all_trials,
        frontier_trials=frontier,
    )


def run_depth1_ysearch_beam(
    *,
    parent_state: YSearchState,
    candidate_names: Iterable[str],
    evaluate_candidate: Callable[[str], Optional[dict]],
    parent_val_loss_base: Optional[float],
    cfg: YSearchControllerConfig,
    strong_structure_trigger_fn: Optional[Callable[[dict], bool]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    stagea_cache: Optional[Dict[StageAStateKey, dict]] = None,
    make_key_fn: Optional[Callable[[str], Optional[StageAStateKey]]] = None,
) -> YSearchResult:
    """Depth-1 controller with beam pruning and optional Stage-A cache."""
    cache = stagea_cache if stagea_cache is not None else {}

    def _evaluate_with_cache(name: str) -> Optional[dict]:
        key = make_key_fn(name) if make_key_fn is not None else None
        if key is not None and key in cache:
            if log_fn is not None:
                log_fn(f"[ysearch] cache hit for state={key.y_stack_sig}")
            return cache[key]
        payload = evaluate_candidate(name)
        if key is not None and payload is not None:
            cache[key] = payload
        return payload

    return run_depth1_ysearch(
        parent_state=parent_state,
        candidate_names=candidate_names,
        evaluate_candidate=_evaluate_with_cache,
        parent_val_loss_base=parent_val_loss_base,
        cfg=cfg,
        strong_structure_trigger_fn=strong_structure_trigger_fn,
        log_fn=log_fn,
    )


def run_ysearch_beam(
    *,
    parent_state: YSearchState,
    candidate_names: Iterable[str],
    evaluate_state: Callable[[Tuple[str, ...]], Optional[dict]],
    parent_val_loss_base: Optional[float],
    cfg: YSearchControllerConfig,
    strong_structure_trigger_fn: Optional[Callable[[dict], bool]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    stagea_cache: Optional[Dict[StageAStateKey, dict]] = None,
    make_key_fn: Optional[Callable[[Tuple[str, ...]], Optional[StageAStateKey]]] = None,
) -> YSearchResult:
    """General beam search over y-transform stacks (default depth=1)."""
    names = list(candidate_names)
    if not names:
        return YSearchResult(
            best_trial=None,
            accepted_trials=[],
            all_trials=[],
            frontier_trials=[],
        )

    max_depth = max(0, int(cfg.max_depth))
    beam = max(1, int(cfg.beam))
    expand_k = max(1, int(cfg.expand_k))
    trigger_fn = strong_structure_trigger_fn or _default_trigger
    cache = stagea_cache if stagea_cache is not None else {}
    eval_budget = max(0, int(getattr(cfg, "max_state_evals", 0)))
    eval_count = 0
    budget_exhausted = False

    parent_loss = (
        float(parent_val_loss_base) if _finite(parent_val_loss_base) else float("inf")
    )
    eps = max(0.0, float(cfg.eps_parent_loss))
    ratio = float(cfg.confirm_improve_ratio)

    if len(parent_state.y_stack) >= max_depth:
        return YSearchResult(
            best_trial=None,
            accepted_trials=[],
            all_trials=[],
            frontier_trials=[],
        )

    def _eval_with_cache(stack: Tuple[str, ...]) -> Optional[dict]:
        nonlocal eval_count, budget_exhausted
        key = make_key_fn(stack) if make_key_fn is not None else None
        if key is not None and key in cache:
            if log_fn is not None:
                log_fn(f"[ysearch] cache hit for state={key.y_stack_sig}")
            return cache[key]
        if eval_budget > 0 and eval_count >= eval_budget:
            budget_exhausted = True
            return None
        payload = evaluate_state(stack)
        eval_count += 1
        if key is not None and payload is not None:
            cache[key] = payload
        return payload

    frontier: List[Tuple[YSearchState, float]] = [(parent_state, parent_loss)]
    all_trials: List[YSearchTrial] = []
    accepted_trials: List[YSearchTrial] = []
    frontier_trials: List[YSearchTrial] = []

    while frontier:
        if budget_exhausted:
            break
        next_trials: List[YSearchTrial] = []
        next_frontier: List[Tuple[YSearchState, float]] = []

        for state, state_parent_loss in frontier:
            if budget_exhausted:
                break
            if len(state.y_stack) >= max_depth:
                continue
            local_names = names[:expand_k]
            for name in local_names:
                if budget_exhausted:
                    break
                child_stack = tuple(state.y_stack) + (str(name),)
                payload = _eval_with_cache(child_stack)
                if payload is None:
                    continue

                val = payload.get("val_loss_base", float("inf"))
                val_f = float(val) if _finite(val) else float("inf")
                split_success = bool(payload.get("split_success", False))
                strong_trigger = bool(trigger_fn(payload))

                if _finite(state_parent_loss):
                    if state_parent_loss <= eps:
                        accept = bool(strong_trigger)
                    else:
                        accept = bool(
                            (val_f <= ratio * max(float(state_parent_loss), 1.0e-15))
                            or strong_trigger
                        )
                else:
                    accept = bool(_finite(val_f) or strong_trigger)

                trial = YSearchTrial(
                    name=str(name),
                    state=YSearchState(y_stack=child_stack),
                    val_loss_base=float(val_f),
                    split_success=split_success,
                    strong_structure_trigger=strong_trigger,
                    accept_branch=accept,
                    payload=payload,
                )
                all_trials.append(trial)
                next_trials.append(trial)
                if accept:
                    accepted_trials.append(trial)
                    next_frontier.append((trial.state, val_f))

                if log_fn is not None:
                    sig_msg = ""
                    try:
                        sig = payload.get("stagea_signals", {}) if isinstance(payload, dict) else {}
                        if isinstance(sig, dict):
                            sep_s = sig.get("sep_score", None)
                            split_s = sig.get("best_split_score", None)
                            trig_s = sig.get("trig_affine_conf", None)
                            reasons = (
                                payload.get("ysearch_trigger_reason_str", "")
                                if isinstance(payload, dict)
                                else ""
                            )
                            plans_n = 0
                            try:
                                plans = (
                                    payload.get("split_plans", None)
                                    if isinstance(payload, dict)
                                    else None
                                )
                                plans_n = len(plans) if isinstance(plans, (list, tuple)) else 0
                            except Exception:
                                plans_n = 0
                            parts = []
                            if _finite(sep_s):
                                parts.append(f"sep={float(sep_s):.3g}")
                            if _finite(split_s):
                                parts.append(f"split_score={float(split_s):.3g}")
                            if _finite(trig_s):
                                parts.append(f"trig={float(trig_s):.3g}")
                            if plans_n > 0:
                                parts.append(f"plans={int(plans_n)}")
                            status = _confirmation_status(payload)
                            if status:
                                parts.append(f"confirm={status}")
                            if reasons:
                                parts.append(f"reasons={str(reasons)}")
                            if parts:
                                sig_msg = ", " + ", ".join(parts)
                    except Exception:
                        sig_msg = ""
                    log_fn(
                        "[ysearch] trial "
                        f"stack={trial.state.y_stack}, val_loss_base={trial.val_loss_base:.3g}, "
                        f"split={int(trial.split_success)}, trigger={int(trial.strong_structure_trigger)}, "
                        f"accept={int(trial.accept_branch)}{sig_msg}"
                    )
        if budget_exhausted and log_fn is not None:
            log_fn(
                f"[ysearch] state-eval budget exhausted: {eval_count}/{eval_budget}"
            )

        if not next_frontier:
            frontier_trials = []
            break

        next_frontier.sort(
            key=lambda t: (
                float(t[1]),
                str(t[0].y_stack),
            )
        )
        frontier = next_frontier[:beam]

        next_trials.sort(
            key=lambda t: (
                0 if bool(t.accept_branch) else 1,
                float(t.val_loss_base),
                str(t.state.y_stack),
            )
        )
        frontier_trials = next_trials[:beam]

    pool = accepted_trials if accepted_trials else all_trials
    if not pool:
        return YSearchResult(
            best_trial=None,
            accepted_trials=accepted_trials,
            all_trials=all_trials,
            frontier_trials=frontier_trials,
            state_evals=int(eval_count),
            budget_exhausted=bool(budget_exhausted),
        )

    if accepted_trials:
        _key = lambda t: (
            _confirmation_rank(t.payload if isinstance(t.payload, dict) else {}),
            float(t.val_loss_base),
            0 if bool(t.split_success) else 1,
            str(t.state.y_stack),
        )
    else:
        _key = _fallback_key

    best = min(pool, key=_key)
    return YSearchResult(
        best_trial=best,
        accepted_trials=accepted_trials,
        all_trials=all_trials,
        frontier_trials=frontier_trials,
        state_evals=int(eval_count),
        budget_exhausted=bool(budget_exhausted),
    )


def run_ysearch_beam_with_split_recursion(
    *,
    parent_state: YSearchState,
    candidate_names: Iterable[str],
    evaluate_state: Callable[[Tuple[str, ...]], Optional[dict]],
    parent_val_loss_base: Optional[float],
    cfg: YSearchControllerConfig,
    strong_structure_trigger_fn: Optional[Callable[[dict], bool]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    stagea_cache: Optional[Dict[StageAStateKey, dict]] = None,
    make_key_fn: Optional[Callable[[Tuple[str, ...]], Optional[StageAStateKey]]] = None,
    split_plans_fn: Optional[Callable[[dict], Iterable[Any]]] = None,
    recurse_split_fn: Optional[
        Callable[[YSearchState, Any], Optional[YSearchResult]]
    ] = None,
) -> YSearchResult:
    """Run y-search and optionally recurse into split plans with strict caps.

    Recursion is opt-in via ``split_plans_fn`` + ``recurse_split_fn`` and bounded by:
    - ``cfg.max_recursive_branches`` (total recursive calls)
    - ``cfg.max_split_plans_per_state`` (plans consumed per accepted state)
    """
    primary = run_ysearch_beam(
        parent_state=parent_state,
        candidate_names=candidate_names,
        evaluate_state=evaluate_state,
        parent_val_loss_base=parent_val_loss_base,
        cfg=cfg,
        strong_structure_trigger_fn=strong_structure_trigger_fn,
        log_fn=log_fn,
        stagea_cache=stagea_cache,
        make_key_fn=make_key_fn,
    )

    max_recursive = max(0, int(getattr(cfg, "max_recursive_branches", 0)))
    max_plans = max(0, int(getattr(cfg, "max_split_plans_per_state", 0)))
    if (
        max_recursive <= 0
        or max_plans <= 0
        or split_plans_fn is None
        or recurse_split_fn is None
    ):
        return primary

    merged_all = list(primary.all_trials)
    merged_accepted = list(primary.accepted_trials)
    merged_frontier = list(primary.frontier_trials)
    best_trial = primary.best_trial
    total_state_evals = int(primary.state_evals)
    budget_exhausted = bool(primary.budget_exhausted)
    recursive_calls = 0
    global_eval_budget = max(0, int(getattr(cfg, "max_state_evals", 0)))
    remaining_eval_budget = (
        max(0, global_eval_budget - int(primary.state_evals))
        if global_eval_budget > 0
        else 0
    )

    # Prefer accepted branches for recursion; fall back to frontier if needed.
    seeds = list(primary.accepted_trials) if primary.accepted_trials else list(primary.frontier_trials)
    for trial in seeds:
        if global_eval_budget > 0 and remaining_eval_budget <= 0:
            budget_exhausted = True
            break
        if recursive_calls >= max_recursive:
            break
        payload = trial.payload if isinstance(trial.payload, dict) else {}
        try:
            plans = list(split_plans_fn(payload))
        except Exception:
            plans = []
        if not plans:
            continue
        for plan in plans[:max_plans]:
            if global_eval_budget > 0 and remaining_eval_budget <= 0:
                budget_exhausted = True
                break
            if recursive_calls >= max_recursive:
                break
            recursive_calls += 1
            child = recurse_split_fn(trial.state, plan)
            if child is None:
                continue

            merged_all.extend(list(child.all_trials))
            merged_accepted.extend(list(child.accepted_trials))
            merged_frontier.extend(list(child.frontier_trials))
            child_evals = int(getattr(child, "state_evals", 0) or 0)
            total_state_evals += child_evals
            if global_eval_budget > 0:
                remaining_eval_budget = max(0, remaining_eval_budget - child_evals)
            budget_exhausted = bool(
                budget_exhausted or bool(getattr(child, "budget_exhausted", False))
            )
            if child.best_trial is not None:
                if (best_trial is None) or (
                    _fallback_key(child.best_trial) < _fallback_key(best_trial)
                ):
                    best_trial = child.best_trial

    if log_fn is not None and recursive_calls >= max_recursive and max_recursive > 0:
        log_fn(
            f"[ysearch] split-recursion budget exhausted: {recursive_calls}/{max_recursive}"
        )

    return YSearchResult(
        best_trial=best_trial,
        accepted_trials=merged_accepted,
        all_trials=merged_all,
        frontier_trials=merged_frontier,
        state_evals=int(total_state_evals),
        budget_exhausted=bool(budget_exhausted),
        recursive_calls=int(recursive_calls),
    )
