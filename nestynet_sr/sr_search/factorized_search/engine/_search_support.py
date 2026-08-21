# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Factorized-search action policy, stopping, and refinement-slate helpers."""

from __future__ import annotations

import math
import logging
from types import SimpleNamespace
from typing import Any, Mapping
from ..expr_ast import BINARY_OPS, UNARY_OPS, collect_paths, get_at, node_str
from ..expr_mapping import mapping_is_structural
from ..policy.build_slate import controller_selected_action_path as _controller_selected_action_path_impl, normalize_controller_build_slate_actions as _normalize_controller_build_slate_actions_impl
from .actions import ACTION_ID_BY_NAME, ACTION_NAME, A_ADD_RAND, A_BOOST, A_CROSSOVER, A_INVSTEER, A_MUL_RAND, A_PRUNE, A_REPAIR, A_REPLACE, A_RESIDUAL, A_WRAP_UNARY
from .signals import CandidateStateFeatures, InverseSteeringPotential

_log = logging.getLogger(__name__)


_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS = ("replace", "wrap_un", "residual")


_INVERSE_CANDIDATE_META_KEYS = tuple(CandidateStateFeatures.__dataclass_fields__.keys())


_INVERSE_EXTRA_META_KEYS = (
    "inverse_path_mode_beam_count",
    "inverse_path_mode_beam",
    "inverse_exact_score_budget",
    "inverse_exact_support_floor_beams",
    "inverse_exact_support_floor_selected",
    "inverse_exact_global_allocated",
    "inverse_exact_score_observed_count",
    "inverse_repair_slate_id",
    "inverse_repair_slate_count",
    "inverse_repair_slate",
    "repair_opportunity_slate_id",
    "repair_opportunity_slate_count",
    "repair_opportunity_slate",
    "repair_opportunity_slate_final_count",
    "repair_opportunity_slate_final",
    "inverse_exact_allocator_mode",
    "inverse_opportunity_controller_requested",
    "inverse_opportunity_controller_used",
    "inverse_opportunity_controller_error",
    "inverse_exact_budget_trace_count",
    "inverse_exact_budget_trace",
    "inverse_tuple_ranker_used",
    "inverse_tuple_ranker_best_child_key",
    "inverse_tuple_ranker_row_count",
    "inverse_tuple_ranker_child_value_lambda",
    "inverse_tuple_ranker_regret_weight",
    "controller_build_slate_id",
    "controller_build_slate_count",
    "controller_build_slate_exact_observed_count",
    "controller_build_slate_preview_only",
    "controller_build_slate",
    "build_opportunity_slate_id",
    "build_opportunity_slate_count",
    "build_opportunity_slate_preview_only",
    "build_opportunity_slate",
    "observed_wall_seconds",
    "observed_exact_evals",
    "observed_preview_evals",
    "observed_micro_tokens",
    "observed_widen_tokens",
)


def _tracked_macro_actions(active_actions, *, repair_controller_enable=False):
    tracked = list(active_actions or [])
    if bool(repair_controller_enable) and A_REPAIR not in tracked:
        if A_INVSTEER in tracked:
            tracked.insert(tracked.index(A_INVSTEER) + 1, A_REPAIR)
        else:
            tracked.append(A_REPAIR)
    return tracked


def _macro_action_fields(action: int, *, source: str | None = None) -> dict[str, Any]:
    out = {
        "macro_action_id": int(action),
        "macro_action": str(ACTION_NAME.get(action, f"action_{int(action)}")),
    }
    if source is not None:
        out["macro_action_source"] = str(source)
    return out


def _macro_decision_log_fields(decision: Any) -> dict[str, Any]:
    if decision is None:
        return {}
    out: dict[str, Any] = {
        "controller_macro_policy_source": str(getattr(decision, "policy_source", "")),
        "controller_macro_selected_route": getattr(decision, "selected_route", None),
        "controller_macro_selected_path": [int(v) for v in (getattr(decision, "selected_path", ()) or ())],
        "controller_macro_learned_best_path": [int(v) for v in (getattr(decision, "learned_best_path", ()) or ())],
        "controller_macro_learned_confidence": float(getattr(decision, "learned_confidence", 0.0) or 0.0),
    }
    try:
        out["controller_macro_route_decision_scores"] = {
            str(k): float(v)
            for k, v in dict(getattr(decision, "route_decision_scores", {}) or {}).items()
        }
    except Exception:
        out["controller_macro_route_decision_scores"] = {}
    try:
        value_est = getattr(decision, "learned_value_estimate", None)
        out["controller_macro_learned_value_estimate"] = None if value_est is None else float(value_est)
    except Exception:
        out["controller_macro_learned_value_estimate"] = None
    try:
        value_norm = getattr(decision, "learned_value_normalized", None)
        out["controller_macro_learned_value_normalized"] = None if value_norm is None else float(value_norm)
    except Exception:
        out["controller_macro_learned_value_normalized"] = None
    try:
        out["controller_macro_learned_scores"] = {
            str(k): float(v)
            for k, v in dict(getattr(decision, "learned_scores", {}) or {}).items()
        }
    except Exception:
        out["controller_macro_learned_scores"] = {}
    try:
        out["controller_macro_learned_action_probs"] = {
            str(k): float(v)
            for k, v in dict(getattr(decision, "learned_action_probs", {}) or {}).items()
        }
    except Exception:
        out["controller_macro_learned_action_probs"] = {}
    try:
        out["controller_macro_learned_route_scores"] = {
            str(k): float(v)
            for k, v in dict(getattr(decision, "learned_route_scores", {}) or {}).items()
        }
    except Exception:
        out["controller_macro_learned_route_scores"] = {}
    try:
        out["controller_macro_learned_route_probs"] = {
            str(k): float(v)
            for k, v in dict(getattr(decision, "learned_route_probs", {}) or {}).items()
        }
    except Exception:
        out["controller_macro_learned_route_probs"] = {}
    try:
        out["controller_macro_learned_action_value_estimates"] = {
            str(k): float(v)
            for k, v in dict(getattr(decision, "learned_action_value_estimates", {}) or {}).items()
        }
    except Exception:
        out["controller_macro_learned_action_value_estimates"] = {}
    try:
        out["controller_macro_learned_action_value_normalized"] = {
            str(k): float(v)
            for k, v in dict(getattr(decision, "learned_action_value_normalized", {}) or {}).items()
        }
    except Exception:
        out["controller_macro_learned_action_value_normalized"] = {}
    try:
        out["controller_macro_learned_best_route"] = getattr(decision, "learned_best_route", None)
    except Exception:
        out["controller_macro_learned_best_route"] = None
    return out


def _merge_inverse_proposal_log_fields(
    entry: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    *,
    status_key: str,
    value_prefix: str = "",
) -> dict[str, Any] | None:
    if not isinstance(entry, dict) or not isinstance(meta, dict):
        return entry
    status_val = meta.get("status", None)
    if status_val is not None:
        entry[str(status_key)] = str(status_val)
    for key in _INVERSE_CANDIDATE_META_KEYS:
        if key not in meta:
            continue
        dst = f"{value_prefix}{key}" if value_prefix else str(key)
        entry[dst] = meta[key]
    for key in _INVERSE_EXTRA_META_KEYS:
        if key not in meta:
            continue
        dst = f"{value_prefix}{key}" if value_prefix else str(key)
        entry[dst] = meta[key]
    return entry


def _merge_repair_option_log_fields(
    entry: dict[str, Any] | None,
    meta: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(entry, dict) or not isinstance(meta, dict):
        return entry
    for key, value in meta.items():
        if str(key).startswith("repair_option_"):
            entry[str(key)] = value
    return _merge_inverse_proposal_log_fields(
        entry,
        meta,
        status_key="repair_option_status",
        value_prefix="repair_option_final_",
    )


def _init_crossover_policy_stats(policies=("legacy",)):
    out = {}
    for pol in policies:
        out[str(pol)] = {
            "selected": 0,
            "proposed": 0,
            "accepted": 0,
            "reward_sum": 0.0,
            "reward_count": 0,
        }
    return out


def _finalize_crossover_policy_stats(stats):
    out = {}
    ordered = [k for k in ("legacy", "local", "foreign") if k in stats]
    ordered.extend(sorted(k for k in stats if k not in ordered))
    for pol in ordered:
        st = stats.get(pol, {})
        sel = int(st.get("selected", 0))
        prop = int(st.get("proposed", 0))
        acc = int(st.get("accepted", 0))
        rc = int(st.get("reward_count", 0))
        rs = float(st.get("reward_sum", 0.0))
        out[pol] = {
            "selected": sel,
            "proposed": prop,
            "accepted": acc,
            "proposal_rate": (prop / float(sel)) if sel > 0 else 0.0,
            "accept_rate_given_selected": (acc / float(sel)) if sel > 0 else 0.0,
            "accept_rate_given_proposed": (acc / float(prop)) if prop > 0 else 0.0,
            "avg_reward": (rs / float(rc)) if rc > 0 else 0.0,
            "masked_no_partner_iters": int(st.get("masked_no_partner_iters", 0)),
        }
    return out


def _degenerate_abort_should_stop(
    *,
    n_evaluated: int,
    accepted_total: int,
    start_best: float,
    current_best: float,
    enable: bool,
    min_evals: int,
    max_accepted: int,
    stall_delta: float,
) -> bool:
    """Decide whether a mutation loop is degenerate and should abort.

    Degenerate means: the launch has run for a while, almost nothing has been
    accepted into the archive, and the best score has not improved since the
    brute phase. Finding any finite score from an empty archive counts as
    progress.
    """
    if not enable:
        return False
    if int(n_evaluated) < max(1, int(min_evals)):
        return False
    if int(accepted_total) > int(max_accepted):
        return False
    if math.isfinite(float(current_best)):
        if not math.isfinite(float(start_best)):
            return False
        rel = (float(start_best) - float(current_best)) / max(float(start_best), 1e-30)
        if rel >= float(stall_delta):
            return False
    return True


def _relative_best_improvement(previous_best: float, current_best: float) -> float:
    """Return relative improvement for nonnegative best-score traces."""
    prev = float(previous_best)
    cur = float(current_best)
    if not math.isfinite(prev):
        return float("inf") if math.isfinite(cur) else 0.0
    if not math.isfinite(cur):
        return 0.0
    return (prev - cur) / max(abs(prev), 1e-30)


def _archive_best_stall_mse(arch: Any, *, prefer_raw: bool) -> float:
    """Return the archive best score used for stall/plateau detection."""
    if arch is None or not getattr(arch, "d", None):
        return float("inf")
    if not bool(prefer_raw):
        try:
            recs = arch.best(1)
            if recs:
                val = float(getattr(recs[0], "best_mse", float("inf")))
                if math.isfinite(val):
                    return val
        except Exception:
            pass
        return float("inf")

    vals: list[float] = []
    try:
        records = list(getattr(arch, "d", {}).values())
    except Exception:
        records = []
    for rec in records:
        for attr in ("min_raw_mse", "best_raw_mse"):
            try:
                val = float(getattr(rec, attr, float("inf")))
            except Exception:
                val = float("inf")
            if math.isfinite(val):
                vals.append(val)
        try:
            elites = list(getattr(rec, "elites", []) or [])
        except Exception:
            elites = []
        for elite in elites:
            try:
                val = float(getattr(elite, "raw_mse", getattr(elite, "mse", float("inf"))))
            except Exception:
                val = float("inf")
            if math.isfinite(val):
                vals.append(val)
    if vals:
        return float(min(vals))

    try:
        recs = arch.best(1)
        if recs:
            val = float(getattr(recs[0], "best_raw_mse", getattr(recs[0], "best_mse", float("inf"))))
            if math.isfinite(val):
                return val
    except Exception:
        pass
    return float("inf")


def _plateau_stop_should_stop(
    *,
    enable: bool,
    n_evaluated: int,
    min_evals: int,
    consecutive_soft_restarts: int,
    max_soft_restarts: int,
    has_archive: bool,
) -> bool:
    """Decide whether repeated stall-triggered soft restarts should stop."""
    if not enable:
        return False
    if not has_archive:
        return False
    if int(max_soft_restarts) <= 0:
        return False
    if int(n_evaluated) < max(0, int(min_evals)):
        return False
    return int(consecutive_soft_restarts) >= int(max_soft_restarts)


def _remove_allowed_action(allowed_actions, active_actions, action):
    base = list(active_actions if allowed_actions is None else allowed_actions)
    out = [a for a in base if a != action]
    if not out:
        out = [a for a in active_actions if a != action]
    return out if out else list(active_actions)


def _finalize_action_distribution(
    action_ids,
    selected_counts,
    proposed_counts,
    reward_counts,
    accepted_counts,
    *,
    name_overrides=None,
):
    ov = name_overrides if isinstance(name_overrides, dict) else {}
    counts = {}
    proposed = {}
    rewards = {}
    accepted = {}
    total = 0
    for a in action_ids:
        nm = str(ov.get(a, ACTION_NAME.get(a, f"action_{a}")))
        c = int(selected_counts.get(a, 0))
        p = int(proposed_counts.get(a, 0))
        r = int(reward_counts.get(a, 0))
        ac = int(accepted_counts.get(a, 0))
        counts[nm] = c
        proposed[nm] = p
        rewards[nm] = r
        accepted[nm] = ac
        total += c

    if total > 0:
        fractions = {k: (float(v) / float(total)) for k, v in counts.items()}
    else:
        fractions = {k: 0.0 for k in counts}

    return {
        "counts": counts,
        "fractions": fractions,
        "proposed_counts": proposed,
        "reward_update_counts": rewards,
        "accepted_counts": accepted,
        "total_selected": int(total),
    }


def _coerce_guided_path(path_like):
    if path_like is None:
        return None
    try:
        path = tuple(int(v) for v in path_like)
    except Exception:
        return None
    if not path:
        return None
    return path


def _action_candidate_paths(node, action):
    all_paths = collect_paths(node)
    if action in (A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_CROSSOVER):
        return list(all_paths)
    if action == A_PRUNE:
        return [
            p for p in all_paths
            if get_at(node, p)[0] in UNARY_OPS or get_at(node, p)[0] in BINARY_OPS
        ]
    return []


def _select_action_path(node, action, rng, *, path=None):
    candidates = _action_candidate_paths(node, action)
    if not candidates:
        return None
    guided = _coerce_guided_path(path)
    if guided is not None and guided in set(candidates):
        return guided
    return rng.choice(candidates)


def _normalize_controller_build_slate_actions(action_names: list[Any] | tuple[Any, ...] | None) -> tuple[int, ...]:
    return _normalize_controller_build_slate_actions_impl(
        action_names,
        default_actions=_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS,
        action_id_by_name=ACTION_ID_BY_NAME,
        allowed_action_ids=(A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_RESIDUAL, A_BOOST, A_PRUNE),
    )


def _controller_selected_action_path(
    node,
    action,
    *,
    controller_policy_guidance: Mapping[str, Any] | None = None,
    macro_decision: Any = None,
    macro_state: Any = None,
    inverse_gate_diag: InverseSteeringPotential | None = None,
):
    return _controller_selected_action_path_impl(
        node,
        action,
        controller_policy_guidance=controller_policy_guidance,
        macro_decision=macro_decision,
        macro_state=macro_state,
        inverse_gate_diag=inverse_gate_diag,
        action_candidate_paths_fn=_action_candidate_paths,
        coerce_guided_path_fn=_coerce_guided_path,
    )


_LEGACY_SEARCH_HELPERS = [
    "_decorate_refine_variants",
    "_run_brute_phase",
    "_variant_has_gate_potential",
    "apply_boost_action",
    "apply_inverse_steering_action",
    "run_repair_option",
]


_OPTIONAL_RUNTIME_HOOKS = [
    "_actor_critic_reward_terms",
    "_analytic_repair_controller_score",
    "_hybrid_repair_controller_scores",
    "_normalize_repair_controller_critic_mode",
    "_repair_controller_component_gate",
    "_repair_controller_path_policy",
    "_repair_controller_stagnation_state",
    "_repair_controller_threshold",
    "_repair_controller_weights",
    "_repair_option_candidate_paths",
    "_repair_parent_preview_retry_gate",
    "_repair_parent_record_attempt",
    "_repair_parent_retry_gate",
    "_repair_preview_signature",
    "_repair_route_compare_decision",
    "apply_action",
    "apply_crossover_action",
    "apply_residual_action",
    "choose_parent_repair_aware",
    "estimate_inverse_steering_potential",
    "load_repair_critic_bundle",
    "predict_repair_build_route",
    "predict_repair_controller_heads",
    "rand_node",
]


def _normalize_refine_mode(mode: object) -> str:
    out = str(mode or "slate").strip().lower()
    aliases = {
        "": "slate",
        "none": "off",
        "false": "off",
        "disabled": "off",
        "true": "slate",
        "on": "slate",
        "legacy": "inline",
    }
    out = aliases.get(out, out)
    if out not in {"off", "inline", "slate", "final_polish"}:
        raise ValueError(f"unsupported refine_mode={mode!r}")
    return out


def _slate_float(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _archive_elite_records_for_refinement_slate(arch: Any) -> list[Any]:
    rows: list[Any] = []
    try:
        items = list(arch.items())
    except Exception:
        items = list(getattr(arch, "d", {}).items())
    for residual_basin_key, rec in items:
        elites = list(getattr(rec, "elites", []) or [])
        if not elites and getattr(rec, "best_expr", None) is not None:
            rows.append(
                SimpleNamespace(
                    best_mse=_slate_float(getattr(rec, "best_mse", float("inf"))),
                    best_expr=getattr(rec, "best_expr", None),
                    visits=int(getattr(rec, "visits", 0) or 0),
                    mapping=getattr(rec, "mapping", None),
                    z=getattr(rec, "z", None),
                    best_raw_mse=_slate_float(getattr(rec, "best_raw_mse", getattr(rec, "best_mse", float("inf")))),
                    residual_basin_key=getattr(rec, "residual_basin_key", residual_basin_key),
                    best_elite_id=str(getattr(rec, "best_elite_id", "") or ""),
                )
            )
            continue
        for elite in elites:
            rows.append(
                SimpleNamespace(
                    best_mse=_slate_float(getattr(elite, "mse", float("inf"))),
                    best_expr=getattr(elite, "expr", None),
                    visits=int(getattr(rec, "visits", 0) or 0),
                    mapping=getattr(elite, "mapping", None),
                    z=getattr(elite, "z", None),
                    best_raw_mse=_slate_float(getattr(elite, "raw_mse", getattr(elite, "mse", float("inf")))),
                    residual_basin_key=residual_basin_key,
                    best_elite_id=str(getattr(elite, "elite_id", "") or ""),
                )
            )
    return rows


def _record_refinement_slate_candidate(
    out: list[Any],
    seen_exprs: set[str],
    rec: Any,
    *,
    source: str,
    limit: int,
) -> bool:
    if len(out) >= int(limit):
        return False
    expr = getattr(rec, "best_expr", None)
    if expr is None:
        return True
    try:
        expr_key = str(node_str(expr))
    except Exception:
        expr_key = repr(expr)
    if expr_key in seen_exprs:
        return True
    seen_exprs.add(expr_key)
    out.append(
        SimpleNamespace(
            best_mse=_slate_float(getattr(rec, "best_mse", float("inf"))),
            best_expr=expr,
            visits=int(getattr(rec, "visits", 0) or 0),
            mapping=getattr(rec, "mapping", None),
            z=getattr(rec, "z", None),
            best_raw_mse=_slate_float(getattr(rec, "best_raw_mse", getattr(rec, "best_mse", float("inf")))),
            residual_basin_key=getattr(rec, "residual_basin_key", None),
            best_elite_id=str(getattr(rec, "best_elite_id", "") or ""),
            slate_source=str(source),
            expr_key=expr_key,
        )
    )
    return True


def _select_refinement_slate(arch: Any, *, top_k: int, diverse_k: int) -> list[Any]:
    try:
        top_k = max(0, int(top_k))
    except Exception:
        top_k = 0
    try:
        diverse_k = max(0, int(diverse_k))
    except Exception:
        diverse_k = 0
    limit = int(top_k + diverse_k)
    if limit <= 0 or not getattr(arch, "d", None):
        return []

    selected: list[Any] = []
    seen_exprs: set[str] = set()

    def add_many(records: Any, *, source: str, cap: int) -> None:
        if cap <= 0:
            return
        for rec in list(records or [])[: int(cap)]:
            if not _record_refinement_slate_candidate(
                selected,
                seen_exprs,
                rec,
                source=source,
                limit=limit,
            ):
                break

    try:
        add_many(arch.best(top_k, strategy="mse"), source="effective", cap=top_k)
    except Exception:
        try:
            add_many(arch.best(top_k), source="effective", cap=top_k)
        except Exception:
            pass

    elite_rows = _archive_elite_records_for_refinement_slate(arch)
    raw_rows = sorted(
        elite_rows,
        key=lambda rec: (
            _slate_float(getattr(rec, "best_raw_mse", float("inf"))),
            _slate_float(getattr(rec, "best_mse", float("inf"))),
        ),
    )
    structural_raw_rows = []
    for rec in raw_rows:
        try:
            is_structural = bool(mapping_is_structural(getattr(rec, "mapping", None)))
        except Exception:
            is_structural = False
        if is_structural:
            structural_raw_rows.append(rec)
    add_many(structural_raw_rows, source="raw_structural", cap=top_k)

    try:
        diverse_rows = arch.best(max(diverse_k, top_k), strategy="mse_decade_size")
    except Exception:
        try:
            diverse_rows = arch.best(max(diverse_k, top_k))
        except Exception:
            diverse_rows = []
    add_many(diverse_rows, source="diverse", cap=diverse_k)

    if len(selected) < limit:
        add_many(raw_rows, source="raw", cap=limit)
    return selected[:limit]

__engine_search_definitions__ = (
    "_tracked_macro_actions",
    "_macro_action_fields",
    "_macro_decision_log_fields",
    "_merge_inverse_proposal_log_fields",
    "_merge_repair_option_log_fields",
    "_init_crossover_policy_stats",
    "_finalize_crossover_policy_stats",
    "_degenerate_abort_should_stop",
    "_relative_best_improvement",
    "_archive_best_stall_mse",
    "_plateau_stop_should_stop",
    "_remove_allowed_action",
    "_finalize_action_distribution",
    "_coerce_guided_path",
    "_action_candidate_paths",
    "_select_action_path",
    "_normalize_controller_build_slate_actions",
    "_controller_selected_action_path",
    "_normalize_refine_mode",
    "_slate_float",
    "_archive_elite_records_for_refinement_slate",
    "_record_refinement_slate_candidate",
    "_select_refinement_slate",
)

__engine_search_constants__ = (
    "_log",
    "_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS",
    "_INVERSE_CANDIDATE_META_KEYS",
    "_INVERSE_EXTRA_META_KEYS",
    "_LEGACY_SEARCH_HELPERS",
    "_OPTIONAL_RUNTIME_HOOKS",
)

__engine_search_late_bindings__ = (

)
