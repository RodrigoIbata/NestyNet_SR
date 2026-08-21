# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Repair-option action orchestration."""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Mapping, Sequence

from nestynet_sr.sr_search.model_selection import mapping_cost

from ..config import (
    InverseSteeringConfig,
    coerce_inverse_steering_config,
)
from ..expr_ast import node_depth, node_dims, node_size, node_str, simplify
from .opportunity_critic import predict_opportunity_slate
from .repair_policy import _repair_option_candidate_paths
from ..shared_opportunity import shared_opportunity_row_dict


ScoreExprFn = Callable[..., Any]
InverseActionFn = Callable[..., Any]


def _repair_option_meta_template(initial_path: Sequence[int] | None) -> dict[str, Any]:
    return {
        "status": "started",
        "repair_option_anchor_path": [int(v) for v in (initial_path or ())],
        "repair_option_steps_attempted": 0,
        "repair_option_steps_accepted": 0,
        "repair_option_setup_steps_used": 0,
        "repair_option_step_statuses": [],
        "repair_option_step_paths": [],
        "repair_option_step_rel_improve": [],
        "repair_option_step_signed_rel_improve": [],
        "repair_option_step_nonmyopic_continue": [],
        "repair_option_step_continue_source": [],
        "repair_option_step_value_estimate": [],
        "repair_option_step_regret_estimate": [],
        "repair_option_step_allocation_estimate": [],
        "repair_option_setup_controller_requested": False,
        "repair_option_setup_controller_used": False,
        "repair_option_setup_controller_error": "",
        "repair_option_reveal_trace_count": 0,
        "repair_option_reveal_trace": [],
    }


def _repair_option_return(
    expr,
    option_meta: dict[str, Any],
    *,
    return_meta: bool,
    **updates,
):
    if not return_meta:
        return expr
    out = dict(option_meta)
    out.update(updates)
    return expr, out


def _filter_inverse_action_kwargs(
    inverse_action_fn: InverseActionFn,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(inverse_action_fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)
    allowed = {
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


def _normalize_repair_option_limits(
    *,
    max_steps: Any,
    ancestor_hops: Any,
    min_step_rel_improve: Any,
    max_setup_steps: Any,
    setup_step_value_min: Any,
    setup_step_regret_max: Any,
    setup_step_max_worsen: Any,
) -> dict[str, float | int]:
    try:
        max_steps_i = max(1, int(max_steps))
    except Exception:
        max_steps_i = 3
    try:
        ancestor_hops_i = max(0, int(ancestor_hops))
    except Exception:
        ancestor_hops_i = 1
    try:
        min_step_rel_improve_f = max(0.0, float(min_step_rel_improve))
    except Exception:
        min_step_rel_improve_f = 1.0e-3
    try:
        max_setup_steps_i = max(0, int(max_setup_steps))
    except Exception:
        max_setup_steps_i = 0
    try:
        setup_step_value_min_f = float(setup_step_value_min)
    except Exception:
        setup_step_value_min_f = 0.10
    try:
        setup_step_regret_max_f = max(0.0, float(setup_step_regret_max))
    except Exception:
        setup_step_regret_max_f = 0.50
    try:
        setup_step_max_worsen_f = max(0.0, float(setup_step_max_worsen))
    except Exception:
        setup_step_max_worsen_f = 0.05
    return {
        "max_steps": int(max_steps_i),
        "ancestor_hops": int(ancestor_hops_i),
        "min_step_rel_improve": float(min_step_rel_improve_f),
        "max_setup_steps": int(max_setup_steps_i),
        "setup_step_value_min": float(setup_step_value_min_f),
        "setup_step_regret_max": float(setup_step_regret_max_f),
        "setup_step_max_worsen": float(setup_step_max_worsen_f),
    }


def _score_repair_option_expr(
    node,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    proj,
    fp_mode,
    q_scale,
    q_clip,
    poly_degree,
    *,
    max_depth: int,
    var_dims=None,
    complexity_penalty: float = 0.0,
    score_expr_cfg: dict[str, Any] | None = None,
    score_expr_fn: ScoreExprFn | None = None,
) -> dict[str, Any] | None:
    if score_expr_fn is None:
        return None
    expr = simplify(node)
    while isinstance(expr, tuple) and expr and expr[0] == "neg":
        expr = expr[1]
    if isinstance(expr, tuple) and expr and expr[0] == "sub" and node_str(expr[1]) > node_str(expr[2]):
        expr = ("sub", expr[2], expr[1])
    if node_depth(expr) > int(max_depth):
        return None
    if var_dims is not None:
        try:
            d = node_dims(expr, var_dims)
        except Exception:
            d = None
        if d is None:
            return None
    try:
        sc = score_expr_fn(
            expr,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            fp_mode,
            q_scale,
            q_clip,
            poly_degree,
            refine_enable=False,
            refine_cfg=score_expr_cfg if isinstance(score_expr_cfg, dict) else {},
            return_expr=True,
        )
    except Exception:
        sc = None
    if sc is None:
        return None
    try:
        raw_mse = float(sc[0])
        mapping = sc[3]
        scored_expr = sc[4]
    except Exception:
        return None
    if mapping is None or (not math.isfinite(raw_mse)):
        return None
    return {
        "expr": scored_expr,
        "mapping": mapping,
        "raw_mse": float(raw_mse),
        "eff_mse": float(raw_mse + float(complexity_penalty) * float(node_size(expr) + mapping_cost(mapping))),
    }


def _initialize_repair_option_state(
    parent_node,
    parent_mapping,
    current_eff_mse: float | None,
    *,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    proj,
    fp_mode,
    q_scale,
    q_clip,
    poly_degree,
    max_depth: int,
    var_dims=None,
    complexity_penalty: float = 0.0,
    score_expr_cfg: dict[str, Any] | None = None,
    score_expr_fn: ScoreExprFn | None = None,
):
    current_node = parent_node
    current_mapping = parent_mapping
    try:
        current_eff = float(current_eff_mse)
    except Exception:
        current_eff = None
    if current_eff is not None and math.isfinite(current_eff):
        return current_node, current_mapping, float(current_eff)
    scored_parent = _score_repair_option_expr(
        current_node,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        fp_mode,
        q_scale,
        q_clip,
        poly_degree,
        max_depth=int(max_depth),
        var_dims=var_dims,
        complexity_penalty=float(complexity_penalty),
        score_expr_cfg=score_expr_cfg,
        score_expr_fn=score_expr_fn,
    )
    if scored_parent is not None:
        current_node = scored_parent["expr"]
        current_mapping = scored_parent["mapping"]
        current_eff = float(scored_parent["eff_mse"])
    return current_node, current_mapping, current_eff


def _repair_option_step_rel_improve(
    step_meta: dict[str, Any] | None,
    current_eff: float | None,
) -> float:
    rel_improve_f = _repair_option_step_signed_rel_improve(step_meta, current_eff)
    return float(max(0.0, rel_improve_f))


def _repair_option_step_signed_rel_improve(
    step_meta: dict[str, Any] | None,
    current_eff: float | None,
) -> float:
    step_child_eff = (step_meta or {}).get("estimated_child_eff_mse", None)
    try:
        step_child_eff_f = float(step_child_eff)
    except Exception:
        step_child_eff_f = None
    rel_improve = (step_meta or {}).get("estimated_one_hole_rel_improve_eff", None)
    try:
        rel_improve_f = float(rel_improve)
    except Exception:
        rel_improve_f = None
    if (
        rel_improve_f is None
        and step_child_eff_f is not None
        and current_eff is not None
        and math.isfinite(current_eff)
        and current_eff > 1.0e-30
    ):
        rel_improve_f = (current_eff - step_child_eff_f) / current_eff
    if rel_improve_f is None or (not math.isfinite(rel_improve_f)):
        return 0.0
    return float(rel_improve_f)


def _repair_option_step_tuple_signals(
    step_meta: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None]:
    step_meta = step_meta if isinstance(step_meta, dict) else {}
    try:
        value_est = float(step_meta.get("tuple_value_estimate", None))
    except Exception:
        value_est = None
    if value_est is not None and (not math.isfinite(value_est)):
        value_est = None
    try:
        regret_est = float(step_meta.get("tuple_regret_estimate", None))
    except Exception:
        regret_est = None
    if regret_est is not None and (not math.isfinite(regret_est)):
        regret_est = None
    try:
        allocation_est = float(
            step_meta.get(
                "tuple_allocation_estimate",
                step_meta.get(
                    "tuple_combined_estimate",
                    step_meta.get("tuple_utility_estimate", None),
                ),
            )
        )
    except Exception:
        allocation_est = None
    if allocation_est is not None and (not math.isfinite(allocation_est)):
        allocation_est = None
    return value_est, regret_est, allocation_est


def _repair_option_allow_setup_step(
    *,
    step_meta: dict[str, Any] | None,
    current_eff: float | None,
    step_idx: int,
    limits: dict[str, float | int],
    setup_steps_used: int,
    has_accepted_step: bool,
) -> bool:
    if has_accepted_step:
        return False
    if int(setup_steps_used) >= int(limits.get("max_setup_steps", 0)):
        return False
    if int(step_idx) + 1 >= int(limits.get("max_steps", 0)):
        return False
    signed_rel = _repair_option_step_signed_rel_improve(step_meta, current_eff)
    if signed_rel < -float(limits.get("setup_step_max_worsen", 0.0)):
        return False
    value_est, regret_est, allocation_est = _repair_option_step_tuple_signals(step_meta)
    if regret_est is None or regret_est > float(limits.get("setup_step_regret_max", 0.0)):
        return False
    if value_est is not None and value_est >= float(limits.get("setup_step_value_min", 0.0)):
        return True
    return allocation_est is not None and allocation_est >= float(limits.get("setup_step_value_min", 0.0))


def _repair_option_decision_id(
    current_node,
    anchor_path: Sequence[int] | None,
) -> str:
    return f"repair_option:{node_str(current_node)}:{tuple(int(v) for v in (anchor_path or ()))}"


def _serialize_repair_option_setup_opportunity_row(
    *,
    current_node,
    step_expr,
    step_meta: Mapping[str, Any] | None,
    current_eff: float | None,
    step_idx: int,
    setup_steps_used: int,
    limits: Mapping[str, Any],
    anchor_path: Sequence[int] | None,
) -> dict[str, Any]:
    step_meta = step_meta if isinstance(step_meta, Mapping) else {}
    path_like = step_meta.get("selected_path", None)
    if not path_like:
        path_like = anchor_path
    try:
        path = [int(v) for v in (path_like or ())]
    except Exception:
        path = []
    value_est, regret_est, allocation_est = _repair_option_step_tuple_signals(dict(step_meta))
    rel_gain = _repair_option_step_signed_rel_improve(dict(step_meta), current_eff)
    best_child_eff = step_meta.get("estimated_child_eff_mse", None)
    if best_child_eff is not None:
        try:
            best_child_eff = float(best_child_eff)
        except Exception:
            best_child_eff = None
    evidence_level = "exact_known" if best_child_eff is not None else "preview_only"
    current_best_child_expr = ""
    if step_expr is not None:
        try:
            current_best_child_expr = str(node_str(step_expr))
        except Exception:
            current_best_child_expr = ""
    if not current_best_child_expr:
        try:
            current_best_child_expr = str(node_str(current_node))
        except Exception:
            current_best_child_expr = ""
    return shared_opportunity_row_dict({
        "route_source": "repair",
        "opportunity_type": "repair_setup",
        "decision_id": _repair_option_decision_id(current_node, anchor_path),
        "beam_id": f"repair_setup:{int(step_idx)}",
        "parent_expr": str(node_str(current_node)),
        "action": "repair_option",
        "path": path,
        "path_source": "guided" if path else "random",
        "target_mode": str(step_meta.get("selected_target_mode", "") or ""),
        "evidence_level": str(evidence_level),
        "parent_depth": int(node_depth(current_node)),
        "parent_eff_mse": None if current_eff is None or not math.isfinite(current_eff) else float(current_eff),
        "budget_exact_spent": int(max(0, setup_steps_used)),
        "budget_remaining": int(max(0, int(limits.get("max_setup_steps", 0)) - int(setup_steps_used))),
        "budget_widen_spent": 0,
        "budget_micro_spent": 0,
        "current_best_child_expr": str(current_best_child_expr),
        "current_best_child_eff_mse": None if best_child_eff is None else float(best_child_eff),
        "current_best_route_eff_mse": None if current_eff is None or not math.isfinite(current_eff) else float(current_eff),
        "candidate_count_observed": 1,
        "candidate_count_unique": 1,
        "preview_candidate_count_total": 1,
        "preview_candidate_count_unique_total": 1,
        "shadow_total_exact_available": 1,
        "shadow_total_preview_available": 1,
        "shadow_executor_reveals_observed": int(max(0, setup_steps_used)),
        "exact_child_observed_count": 1 if best_child_eff is not None else 0,
        "best_preview_probe_mse": None if best_child_eff is None else float(best_child_eff),
        "best_preview_fit_mse": None if best_child_eff is None else float(best_child_eff),
        "best_tuple_utility_estimate": None if value_est is None else float(value_est),
        "best_tuple_allocation_estimate": None if allocation_est is None else float(allocation_est),
        "path_gain": float(max(0.0, rel_gain)),
        "path_gain_pre_cut": float(max(0.0, rel_gain)),
        "rel_gain": float(rel_gain),
        "transport_rel": 0.0,
        "lin_rel": 0.0,
        "valid_frac": 1.0 if best_child_eff is not None else 0.0,
        "confidence": 1.0 if best_child_eff is not None else 0.0,
        "effective_n": 1.0,
        "branch_factor": 1.0,
        "cut_factor": 1.0,
        "branch_support": 1.0,
        "family_scale": 1.0,
        "tuple_value_estimate": None if value_est is None else float(value_est),
        "tuple_regret_estimate": None if regret_est is None else float(regret_est),
    }, route_source="repair")


def _repair_option_allow_setup_step_with_opportunity_controller(
    *,
    current_node,
    step_expr,
    step_meta: Mapping[str, Any] | None,
    current_eff: float | None,
    step_idx: int,
    limits: Mapping[str, Any],
    setup_steps_used: int,
    has_accepted_step: bool,
    anchor_path: Sequence[int] | None,
    opportunity_bundle: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    trace = {
        "reveal_type": "continue_setup_step",
        "step_index": int(step_idx),
        "decision_source": "opportunity_controller",
        "allow_continue": False,
    }
    if has_accepted_step:
        trace["decision_source"] = "controller_block_has_accepted_step"
        return False, trace
    if int(setup_steps_used) >= int(limits.get("max_setup_steps", 0)):
        trace["decision_source"] = "controller_block_budget_exhausted"
        return False, trace
    if int(step_idx) + 1 >= int(limits.get("max_steps", 0)):
        trace["decision_source"] = "controller_block_step_limit"
        return False, trace
    signed_rel = _repair_option_step_signed_rel_improve(dict(step_meta or {}), current_eff)
    trace["signed_rel_improve"] = float(signed_rel)
    if signed_rel < -float(limits.get("setup_step_max_worsen", 0.0)):
        trace["decision_source"] = "controller_block_worsen_guard"
        return False, trace
    pred = predict_opportunity_slate(
        dict(opportunity_bundle),
        [
            _serialize_repair_option_setup_opportunity_row(
                current_node=current_node,
                step_expr=step_expr,
                step_meta=step_meta,
                current_eff=current_eff,
                step_idx=step_idx,
                setup_steps_used=setup_steps_used,
                limits=limits,
                anchor_path=anchor_path,
            )
        ],
    )
    if not bool((pred or {}).get("trained", False)):
        raise RuntimeError("repair option opportunity controller prediction unavailable")
    pred_rows = [dict(row) for row in list(pred.get("rows", []) or []) if isinstance(row, Mapping)]
    if not pred_rows:
        raise RuntimeError("repair option opportunity controller returned no opportunity rows")
    pred_row = pred_rows[0]
    acquisition = float(pred_row.get("acquisition_estimate", 0.0) or 0.0)
    allow_continue = bool(acquisition > 0.0)
    trace.update({
        "allow_continue": bool(allow_continue),
        "opportunity_id": str(pred_row.get("opportunity_id", "") or ""),
        "beam_id": str(pred_row.get("beam_id", "") or ""),
        "path": [int(v) for v in (pred_row.get("path", []) or [])],
        "target_mode": str(pred_row.get("target_mode", "") or ""),
        "expected_gain_next_under_executor": float(pred_row.get("expected_gain_next_under_executor", 0.0) or 0.0),
        "cost_estimate": float(pred_row.get("cost_estimate", 0.0) or 0.0),
        "fragility_prob": float(pred_row.get("fragility_prob", 0.0) or 0.0),
        "route_flip_prob": float(pred_row.get("route_flip_prob", 0.0) or 0.0),
        "new_residual_basin_prob": float(pred_row.get("new_residual_basin_prob", 0.0) or 0.0),
        "acquisition_estimate": float(acquisition),
        "budget_remaining_before": int(pred_row.get("budget_remaining", 0) or 0),
        "budget_exact_spent_before": int(pred_row.get("budget_exact_spent", 0) or 0),
    })
    return allow_continue, trace


def _append_repair_option_reveal_trace(
    option_meta: dict[str, Any],
    trace_entry: Mapping[str, Any] | None,
) -> None:
    if not isinstance(option_meta, dict) or not isinstance(trace_entry, Mapping):
        return
    reveal_trace = list(option_meta.get("repair_option_reveal_trace", []) or [])
    reveal_trace.append(dict(trace_entry))
    option_meta["repair_option_reveal_trace"] = reveal_trace
    option_meta["repair_option_reveal_trace_count"] = int(len(reveal_trace))


def _run_repair_option_step(
    current_node,
    current_mapping,
    *,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    pool_nodes,
    pool_phi_fit,
    pool_phi_probe,
    pool_dims,
    rng,
    max_depth,
    nvars,
    poly_degree,
    var_dims=None,
    max_paths=12,
    topk_terms=6,
    shortlist_mult=4,
    min_valid_frac=0.25,
    min_confidence=0.10,
    safe_eps=1.0e-12,
    confidence_mode="conditioning",
    confidence_target_gain=4.0,
    confidence_floor=0.05,
    branch_beam_width=1,
    micro_search_enable=False,
    micro_search_max_depth=3,
    micro_search_beam_width=24,
    micro_search_topk=16,
    micro_search_seed_terms=8,
    local_score_mode="affine",
    target_mode="robust",
    full_mapping_penalty=0.75,
    exact_simple_target_bonus=0.10,
    additive_descend_penalty=0.15,
    nonadditive_leaf_penalty=0.20,
    exact_path_eta=0.98,
    exact_transport_min_lin_rel=0.0,
    periodic_min_valid_scale=1.25,
    periodic_min_confidence_scale=1.35,
    periodic_path_penalty=0.65,
    nonperiodic_muldiv_bonus=0.10,
    nonperiodic_explogsqrt_bonus=0.05,
    branch_ambiguity_penalty=0.50,
    transport_min_lin_rel=0.02,
    transport_min_effective_n=8.0,
    complexity_penalty=0.0,
    candidate_paths=None,
    proj=None,
    fp_mode="bits",
    q_scale=2.0,
    q_clip=8.0,
    score_expr_cfg=None,
    inverse_action_fn: InverseActionFn | None = None,
    inverse_action_config: InverseSteeringConfig | Mapping[str, Any] | None = None,
):
    if inverse_action_fn is None:
        return None, None
    inverse_cfg = (
        coerce_inverse_steering_config(inverse_action_config)
        if inverse_action_config is not None
        else coerce_inverse_steering_config(locals())
    )
    inverse_action_kwargs = _filter_inverse_action_kwargs(
        inverse_action_fn,
        inverse_cfg.to_action_kwargs(),
    )
    return inverse_action_fn(
        current_node,
        current_mapping,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes,
        pool_phi_fit,
        pool_phi_probe,
        pool_dims,
        rng,
        max_depth,
        nvars,
        poly_degree,
        var_dims=var_dims,
        **inverse_action_kwargs,
        complexity_penalty=float(complexity_penalty),
        candidate_paths=(candidate_paths if candidate_paths else None),
        proj=proj,
        fp_mode=fp_mode,
        q_scale=q_scale,
        q_clip=q_clip,
        score_expr_cfg=score_expr_cfg,
        return_meta=True,
    )


def run_repair_option_action(
    parent_node,
    parent_mapping,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    pool_nodes,
    pool_phi_fit,
    pool_phi_probe,
    pool_dims,
    rng,
    max_depth,
    nvars,
    poly_degree,
    *,
    var_dims=None,
    max_steps: int = 3,
    ancestor_hops: int = 1,
    min_step_rel_improve: float = 1.0e-3,
    max_setup_steps: int = 0,
    setup_step_value_min: float = 0.10,
    setup_step_regret_max: float = 0.50,
    setup_step_max_worsen: float = 0.05,
    initial_path: Sequence[int] | None = None,
    initial_candidate_paths: Sequence[Sequence[int]] | None = None,
    first_step_expr=None,
    first_step_meta: dict[str, Any] | None = None,
    current_eff_mse: float | None = None,
    max_paths=12,
    topk_terms=6,
    shortlist_mult=4,
    min_valid_frac=0.25,
    min_confidence=0.10,
    safe_eps=1.0e-12,
    confidence_mode="conditioning",
    confidence_target_gain=4.0,
    confidence_floor=0.05,
    branch_beam_width=1,
    micro_search_enable=False,
    micro_search_max_depth=3,
    micro_search_beam_width=24,
    micro_search_topk=16,
    micro_search_seed_terms=8,
    local_score_mode="affine",
    target_mode="robust",
    full_mapping_penalty=0.75,
    exact_simple_target_bonus=0.10,
    additive_descend_penalty=0.15,
    nonadditive_leaf_penalty=0.20,
    exact_path_eta=0.98,
    exact_transport_min_lin_rel=0.0,
    periodic_min_valid_scale=1.25,
    periodic_min_confidence_scale=1.35,
    periodic_path_penalty=0.65,
    nonperiodic_muldiv_bonus=0.10,
    nonperiodic_explogsqrt_bonus=0.05,
    branch_ambiguity_penalty=0.50,
    transport_min_lin_rel=0.02,
    transport_min_effective_n=8.0,
    complexity_penalty=0.0,
    proj=None,
    fp_mode="bits",
    q_scale=2.0,
    q_clip=8.0,
    score_expr_cfg=None,
    return_meta=False,
    score_expr_fn: ScoreExprFn | None = None,
    inverse_action_fn: InverseActionFn | None = None,
    inverse_action_config: InverseSteeringConfig | Mapping[str, Any] | None = None,
    repair_opportunity_controller_enable: bool = False,
    repair_opportunity_bundle: Mapping[str, Any] | None = None,
):
    option_meta = _repair_option_meta_template(initial_path)
    option_meta["repair_option_setup_controller_requested"] = bool(repair_opportunity_controller_enable)
    limits = _normalize_repair_option_limits(
        max_steps=max_steps,
        ancestor_hops=ancestor_hops,
        min_step_rel_improve=min_step_rel_improve,
        max_setup_steps=max_setup_steps,
        setup_step_value_min=setup_step_value_min,
        setup_step_regret_max=setup_step_regret_max,
        setup_step_max_worsen=setup_step_max_worsen,
    )
    anchor_path = tuple(int(v) for v in (initial_path or ()))
    current_node, current_mapping, current_eff = _initialize_repair_option_state(
        parent_node,
        parent_mapping,
        current_eff_mse,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        proj=proj,
        fp_mode=fp_mode,
        q_scale=q_scale,
        q_clip=q_clip,
        poly_degree=poly_degree,
        max_depth=int(max_depth),
        var_dims=var_dims,
        complexity_penalty=float(complexity_penalty),
        score_expr_cfg=score_expr_cfg,
        score_expr_fn=score_expr_fn,
    )

    best_expr = None
    best_meta = None
    setup_steps_used = 0

    for step_idx in range(int(limits["max_steps"])):
        include_ancestors = step_idx > 0
        candidate_paths_step = _repair_option_candidate_paths(
            current_node,
            anchor_path,
            ancestor_hops=int(limits["ancestor_hops"]),
            include_ancestors=include_ancestors,
            fallback_paths=initial_candidate_paths if step_idx == 0 else None,
        )
        if step_idx > 0 and not candidate_paths_step:
            break
        if step_idx == 0 and (first_step_meta is not None or first_step_expr is not None):
            step_expr = first_step_expr
            step_meta = dict(first_step_meta or {})
        else:
            step_expr, step_meta = _run_repair_option_step(
                current_node,
                current_mapping,
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                pool_nodes=pool_nodes,
                pool_phi_fit=pool_phi_fit,
                pool_phi_probe=pool_phi_probe,
                pool_dims=pool_dims,
                rng=rng,
                max_depth=max_depth,
                nvars=nvars,
                poly_degree=poly_degree,
                var_dims=var_dims,
                max_paths=max_paths,
                topk_terms=topk_terms,
                shortlist_mult=shortlist_mult,
                min_valid_frac=min_valid_frac,
                min_confidence=min_confidence,
                safe_eps=safe_eps,
                confidence_mode=confidence_mode,
                confidence_target_gain=confidence_target_gain,
                confidence_floor=confidence_floor,
                branch_beam_width=branch_beam_width,
                micro_search_enable=micro_search_enable,
                micro_search_max_depth=micro_search_max_depth,
                micro_search_beam_width=micro_search_beam_width,
                micro_search_topk=micro_search_topk,
                micro_search_seed_terms=micro_search_seed_terms,
                local_score_mode=local_score_mode,
                target_mode=target_mode,
                full_mapping_penalty=full_mapping_penalty,
                exact_simple_target_bonus=exact_simple_target_bonus,
                additive_descend_penalty=additive_descend_penalty,
                nonadditive_leaf_penalty=nonadditive_leaf_penalty,
                exact_path_eta=exact_path_eta,
                exact_transport_min_lin_rel=exact_transport_min_lin_rel,
                periodic_min_valid_scale=periodic_min_valid_scale,
                periodic_min_confidence_scale=periodic_min_confidence_scale,
                periodic_path_penalty=periodic_path_penalty,
                nonperiodic_muldiv_bonus=nonperiodic_muldiv_bonus,
                nonperiodic_explogsqrt_bonus=nonperiodic_explogsqrt_bonus,
                branch_ambiguity_penalty=branch_ambiguity_penalty,
                transport_min_lin_rel=transport_min_lin_rel,
                transport_min_effective_n=transport_min_effective_n,
                complexity_penalty=complexity_penalty,
                candidate_paths=candidate_paths_step,
                proj=proj,
                fp_mode=fp_mode,
                q_scale=q_scale,
                q_clip=q_clip,
                score_expr_cfg=score_expr_cfg,
                inverse_action_fn=inverse_action_fn,
                inverse_action_config=inverse_action_config,
            )
        option_meta["repair_option_steps_attempted"] = int(option_meta.get("repair_option_steps_attempted", 0)) + 1
        option_meta["repair_option_step_statuses"].append(str((step_meta or {}).get("status", "no_meta")))
        option_meta["repair_option_step_paths"].append(list((step_meta or {}).get("selected_path", []) or []))

        if step_expr is None or not isinstance(step_meta, dict):
            if best_expr is None:
                return _repair_option_return(
                    None,
                    option_meta,
                    return_meta=return_meta,
                    status=str((step_meta or {}).get("status", "repair_option_no_candidate")),
                )
            break

        rel_improve_signed_f = _repair_option_step_signed_rel_improve(step_meta, current_eff)
        rel_improve_f = float(max(0.0, rel_improve_signed_f))
        value_est, regret_est, allocation_est = _repair_option_step_tuple_signals(step_meta)
        allow_setup_step = False
        option_meta["repair_option_step_rel_improve"].append(float(rel_improve_f))
        option_meta["repair_option_step_signed_rel_improve"].append(float(rel_improve_signed_f))
        option_meta["repair_option_step_value_estimate"].append(None if value_est is None else float(value_est))
        option_meta["repair_option_step_regret_estimate"].append(None if regret_est is None else float(regret_est))
        option_meta["repair_option_step_allocation_estimate"].append(None if allocation_est is None else float(allocation_est))
        if float(rel_improve_f) < float(limits["min_step_rel_improve"]):
            controller_used_this_step = False
            trace_entry = None
            if bool(repair_opportunity_controller_enable) and isinstance(repair_opportunity_bundle, Mapping) and bool(repair_opportunity_bundle.get("opportunity_controller_trained", False)):
                try:
                    allow_setup_step, trace_entry = _repair_option_allow_setup_step_with_opportunity_controller(
                        current_node=current_node,
                        step_expr=step_expr,
                        step_meta=step_meta,
                        current_eff=current_eff,
                        step_idx=int(step_idx),
                        limits=limits,
                        setup_steps_used=int(setup_steps_used),
                        has_accepted_step=best_expr is not None,
                        anchor_path=anchor_path,
                        opportunity_bundle=repair_opportunity_bundle,
                    )
                    controller_used_this_step = True
                    option_meta["repair_option_setup_controller_used"] = True
                except Exception as exc:
                    option_meta["repair_option_setup_controller_error"] = str(exc)
            if not controller_used_this_step:
                allow_setup_step = _repair_option_allow_setup_step(
                    step_meta=step_meta,
                    current_eff=current_eff,
                    step_idx=int(step_idx),
                    limits=limits,
                    setup_steps_used=int(setup_steps_used),
                    has_accepted_step=best_expr is not None,
                )
                trace_entry = {
                    "reveal_type": "continue_setup_step",
                    "step_index": int(step_idx),
                    "decision_source": "legacy_nonmyopic",
                    "allow_continue": bool(allow_setup_step),
                    "signed_rel_improve": float(rel_improve_signed_f),
                    "tuple_value_estimate": None if value_est is None else float(value_est),
                    "tuple_regret_estimate": None if regret_est is None else float(regret_est),
                    "tuple_allocation_estimate": None if allocation_est is None else float(allocation_est),
                }
            option_meta["repair_option_step_nonmyopic_continue"].append(bool(allow_setup_step))
            option_meta["repair_option_step_continue_source"].append(str((trace_entry or {}).get("decision_source", "")))
            _append_repair_option_reveal_trace(option_meta, trace_entry)
            if (not allow_setup_step) and best_expr is None:
                low_gain_meta = dict(step_meta)
                low_gain_meta["status"] = "repair_option_low_step_gain"
                return _repair_option_return(
                    None,
                    option_meta,
                    return_meta=return_meta,
                    **low_gain_meta,
                )
            if not allow_setup_step:
                break
        else:
            option_meta["repair_option_step_nonmyopic_continue"].append(False)
            option_meta["repair_option_step_continue_source"].append("accept_step")
            _append_repair_option_reveal_trace(option_meta, {
                "reveal_type": "accept_step",
                "step_index": int(step_idx),
                "decision_source": "rel_improve_gate",
                "allow_continue": False,
                "signed_rel_improve": float(rel_improve_signed_f),
            })

        scored_child = _score_repair_option_expr(
            step_expr,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            fp_mode,
            q_scale,
            q_clip,
            poly_degree,
            max_depth=int(max_depth),
            var_dims=var_dims,
            complexity_penalty=float(complexity_penalty),
            score_expr_cfg=score_expr_cfg,
            score_expr_fn=score_expr_fn,
        )
        if scored_child is not None:
            current_node = scored_child["expr"]
            current_mapping = scored_child["mapping"]
            current_eff = float(scored_child["eff_mse"])
        else:
            current_node = step_expr
            try:
                step_child_eff_f = float(step_meta.get("estimated_child_eff_mse", None))
            except Exception:
                step_child_eff_f = None
            if step_child_eff_f is not None and math.isfinite(step_child_eff_f):
                current_eff = float(step_child_eff_f)

        if allow_setup_step:
            setup_steps_used += 1
            option_meta["repair_option_setup_steps_used"] = int(setup_steps_used)
            continue

        option_meta["repair_option_steps_accepted"] = int(option_meta.get("repair_option_steps_accepted", 0)) + 1
        best_expr = current_node
        best_meta = dict(step_meta)
        if scored_child is not None:
            best_meta["estimated_child_raw_mse"] = float(scored_child["raw_mse"])
            best_meta["estimated_child_eff_mse"] = float(scored_child["eff_mse"])
        if not candidate_paths_step:
            break

    if best_expr is None or best_meta is None:
        return _repair_option_return(
            None,
            option_meta,
            return_meta=return_meta,
            status="repair_option_no_accepted_steps",
        )

    out_meta = dict(best_meta)
    out_meta.update(option_meta)
    out_meta["status"] = "ok"
    return _repair_option_return(
        best_expr,
        option_meta,
        return_meta=return_meta,
        **out_meta,
    )
