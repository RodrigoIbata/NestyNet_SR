# ruff: noqa: F401, F821
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Legacy explorer actions, controller logging, and repair integration."""

import argparse, math, random, json, hashlib, time
import itertools
from typing import Any, Mapping, Sequence
import torch
from .basis_scoring import make_additive_basis_transition
from nestynet_sr.sr_search.factorized_search.config import (
    InverseSteeringConfig,
    coerce_inverse_steering_config,
)
from nestynet_sr.sr_search.factorized_search.engine.actions import (
    apply_action_impl as _apply_action_impl,
    apply_crossover_action_impl as _apply_crossover_action_impl,
    apply_residual_action_impl as _apply_residual_action_impl,
)
from nestynet_sr.sr_search.model_selection import mapping_cost
from nestynet_sr.sr_search.factorized_search.engine.archive import ResidualBasinArchive, Elite, Rec
from nestynet_sr.sr_search.factorized_search.engine.scoring import (
    _LEGACY_REFINEMENT_HELPERS as _ENGINE_REFINEMENT_HOOK_NAMES,
    _eval_node_hparam_safe as _engine_eval_node_hparam_safe,
    _harvest_pool_from_archive as _engine_harvest_pool_from_archive,
    _mapping_equiv_root as _engine_mapping_equiv_root,
    fingerprint as _engine_fingerprint,
    score_expr as _engine_score_expr,
)
from nestynet_sr.sr_search.factorized_search.engine.search import (
    Explorer as _engine_Explorer,
    _LEGACY_SEARCH_HELPERS as _ENGINE_RUNTIME_HOOK_NAMES,
    _OPTIONAL_RUNTIME_HOOKS as _ENGINE_OPTIONAL_RUNTIME_HOOK_NAMES,
    run_explorer_core as _engine_run_explorer_core,
)
from nestynet_sr.sr_search.factorized_search.engine.signals import (
    CandidateStateFeatures,
    InverseSteeringPotential,
    PathStateFeatures,
)
from nestynet_sr.sr_search.factorized_search.policy.features import (
    build_controller_state_record,
    coerce_repair_feature_row,
    RepairControllerFeatureRecord,
)
from nestynet_sr.sr_search.factorized_search.policy.build_slate import (
    collect_controller_build_slate as _collect_controller_build_slate_impl,
    controller_selected_action_path as _controller_selected_action_path_impl,
    normalize_controller_build_slate_actions as _normalize_controller_build_slate_actions_impl,
)
from nestynet_sr.sr_search.factorized_search.policy.guidance import (
    _annotate_inverse_experiment_lineage,
    _choose_repair_execution_preview,
    _credible_route_compare_decision,
    _credible_route_preview_repair_opportunity_rows,
    _controller_build_slate_id,
    _derived_controller_build_rng,
    _logged_action_path_from_row,
    _preview_child_eff_mse,
    _repair_route_compare_decision,
    _serialize_lineage_key,
)
from nestynet_sr.sr_search.factorized_search.policy.parent_selection import (
    choose_parent,
    choose_parent_repair_aware,
)
from nestynet_sr.sr_search.factorized_search.repair_critic import (
    load_repair_critic_bundle,
    predict_repair_build_route,
    predict_repair_controller_heads,
)
from nestynet_sr.sr_search.factorized_search.opportunity_critic import (
    load_opportunity_bundle,
    predict_opportunity_slate,
)
from nestynet_sr.sr_search.factorized_search.research_profiles import (
    RESEARCH_PROFILE_NAMES,
    resolve_engine_research_profile,
)
from nestynet_sr.sr_search.factorized_search.shared_candidate import shared_candidate_row_dict
from nestynet_sr.sr_search.factorized_search.controller import (
    MacroController,
    build_macro_controller_state,
)
import logging as _logging

from .expr_ast import (
    BINARY_OPS,
    UNARY_OPS,
    build_pool,
    cap_depth,
    collect_paths,
    compute_reachable,
    dim_round,
    dims_eq,
    eval_node,
    get_at,
    node_cost_physics_prior,
    node_depth,
    node_dims,
    node_size,
    node_str,
    rand_node,
    rand_node_dim,
    replace_at,
    sample_box,
    set_dim_precision,
    simplify,
)
from .expr_enum import enumerate_trees as _shared_enumerate_trees
from .expr_enum import enumerate_trees_dim as _shared_enumerate_trees_dim
from .expr_mapping import (
    _mapping_nparams,
    eval_exp_mapping,
    eval_mapping,
    eval_pade,
    eval_poly,
    eval_power,
    eval_sine,
    fit_best,
    fit_exp_mapping,
    fit_pade,
    fit_poly,
    fit_power,
    fit_sine,
    mean_squared_error_same_shape,
    mapping_is_structural,
)


def _use_affine_fast_path(poly_degree, _family_mode=None) -> bool:
    return int(poly_degree) == 1


def _fit_best_with_cfg(pred, y, poly_degree, cfg):
    family_mode = "full"
    if isinstance(cfg, Mapping):
        try:
            family_mode = str(cfg.get("score_mapping_family_mode", "full") or "full")
        except Exception:
            family_mode = "full"
    return fit_best(
        pred,
        y,
        poly_degree,
        family_mode=family_mode,
        affine_fast=_use_affine_fast_path(poly_degree, family_mode),
        diagnostics=_refine_diag(cfg),
    )
from .inverse_core import (
    InverseStep,
    InverseTarget,
    _blend_inverse_backprop_target,
    _bool_col,
    _cheap_affine_probe_stats_from_preds,
    _collect_nodes_preorder,
    _combine_inverse_confidence,
    _compute_path_influences,
    _conditioning_confidence_from_gain,
    _conditioning_point_weight_from_gain,
    _effective_sample_size,
    _ensure_col,
    _estimate_path_transport_scores,
    _eval_linear_head,
    _finite_mask,
    _fit_affine_mapping_from_pair,
    _invert_binary_context,
    _invert_shifted_sinusoid,
    _invert_shifted_sinusoid_branches,
    _invert_unary_context,
    _invert_unary_context_branches,
    _linearized_residual_gain,
    _mapping_inverse_point_weight,
    _mapping_output_derivative,
    _mask_fraction,
    _masked_point_weight,
    _normalize_inverse_local_score_mode,
    _normalize_inverse_target_mode,
    _path_transport_scalar,
    _prepare_nonnegative_weights,
    _score_inverse_local_predictions,
    _score_predictions_on_target,
    _slice_by_mask,
    _weighted_centered_mse,
    _weighted_inner_cols,
    _weighted_mse_cols,
    eval_mapping_total,
    invert_context_target,
    invert_context_target_beam,
    invert_mapping_target,
    _inverse_target_mode_rows,
)
from .inverse_search import (
    _deterministic_row_subset,
    _eval_quantized_monomial_from_pool,
    _inverse_additive_combo_candidates,
    _inverse_branch_beam_factor,
    _inverse_collect_local_repair_candidates,
    _inverse_effective_branch_beam_width,
    _inverse_effective_thresholds,
    _inverse_family_gain_scale,
    _inverse_mapping_static_weight,
    _inverse_muldiv_monomial_candidates,
    _inverse_path_cut_factor,
    _inverse_path_profile,
    _inverse_pool_shortlist,
    _inverse_rank_local_repair_candidates,
    _inverse_sqrt_quadratic_candidates,
    _inverse_static_path_score,
    _inverse_subtree_micro_search,
    _mapping_cache_signature,
    _mapping_kind_lower,
    _node_pow_small_int,
    _pool_cache_signature,
    _quantize_monomial_exponent,
    _weighted_linear_fit,
    estimate_inverse_steering_potential,
)
from .inverse_action import run_inverse_steering_action
from .repair_action import (
    _score_repair_option_expr as _score_repair_option_expr_impl,
    run_repair_option_action,
)
from .repair_policy import (
    _actor_critic_reward_terms,
    _analytic_repair_controller_score,
    _hybrid_repair_controller_scores,
    _normalize_repair_controller_critic_mode,
    _repair_controller_component_gate,
    _repair_controller_path_policy,
    _repair_controller_relation_score,
    _repair_controller_stagnation_state,
    _repair_controller_threshold,
    _repair_controller_weights,
    _repair_option_candidate_paths,
    _repair_parent_record_attempt,
    _repair_parent_preview_retry_gate,
    _repair_parent_retry_gate,
    _repair_preview_signature,
)

# --- target functions --- (nvars, function)

def pb011_function(x):
    return x[:,0:1]*(x[:,1:2]+x[:,2:3]*x[:,3:4]*torch.sin(x[:,4:5]))

def addsum_function(x):
    """Sum of simple terms that are all in the pool."""
    return x[:,0:1]*x[:,1:2] + torch.sin(x[:,2:3]) + x[:,3:4]*torch.sin(x[:,4:5])

def poly_function(x):
    """Polynomial + trig — partially in pool."""
    return x[:,0:1]**2 + x[:,1:2]*x[:,2:3] + torch.sin(x[:,3:4]*x[:,4:5])

def exp_product(x):
    """exp of a pool term — needs nonlinear mapping to unwrap."""
    return torch.exp(x[:,0:1]*x[:,1:2] / 10.0)

def square_addsum(x):
    """Square of addsum — needs degree-2 mapping to unwrap."""
    inner = x[:,0:1]*x[:,1:2] + torch.sin(x[:,2:3]) + x[:,3:4]*torch.sin(x[:,4:5])
    return inner ** 2

def feynman_012(x):
    """I.12.1  F = m*(v1² + v2² + v3²)/2"""
    return x[:,0:1]*(x[:,1:2]**2 + x[:,2:3]**2 + x[:,3:4]**2)/2

def feynman_090(x):
    """II.2.42  P = q*sqrt(E1² + E2² + E3²)"""
    return x[:,0:1]*torch.sqrt(x[:,1:2]**2 + x[:,2:3]**2 + x[:,3:4]**2)

def feynman_028(x):
    """I.34.27  d = sqrt(d1² - 2*d1*d2*cos(θ1-θ2) + d2²)"""
    return torch.sqrt(x[:,0:1]**2 - 2*x[:,0:1]*x[:,1:2]*torch.cos(x[:,2:3]-x[:,3:4]) + x[:,1:2]**2)

TARGET_FUNCS = {
    "pb011":        (5, pb011_function, None, None),
    "addsum":       (5, addsum_function, None, None),
    "poly":         (5, poly_function, None, None),
    "exp_product":  (5, exp_product, None, None),
    "square_addsum":(5, square_addsum, None, None),
    "feynman_012":  (4, feynman_012,
                     (2.0,-2.0,1.0,0.0,0.0),
                     [(0.0,0.0,1.0,0.0,0.0),(1.0,-1.0,0.0,0.0,0.0),
                      (1.0,-1.0,0.0,0.0,0.0),(1.0,-1.0,0.0,0.0,0.0)]),
    "feynman_090":  (4, feynman_090,
                     (2.0,-2.0,1.0,0.0,0.0),
                     [(4.0,-3.0,1.0,0.0,-1.0),(-2.0,1.0,0.0,0.0,1.0),
                      (-2.0,1.0,0.0,0.0,1.0),(-2.0,1.0,0.0,0.0,1.0)]),
    "feynman_028":  (4, feynman_028,
                     (1.0,0.0,0.0,0.0,0.0),
                     [(1.0,0.0,0.0,0.0,0.0),(1.0,0.0,0.0,0.0,0.0),
                      (0.0,0.0,0.0,0.0,0.0),(0.0,0.0,0.0,0.0,0.0)]),
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


def _normalize_controller_build_slate_actions(action_names: Sequence[Any] | None) -> tuple[int, ...]:
    return _normalize_controller_build_slate_actions_impl(
        action_names,
        default_actions=_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS,
        action_id_by_name=ACTION_ID_BY_NAME,
        allowed_action_ids=(A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_RESIDUAL, A_BOOST, A_PRUNE),
    )


def _collect_controller_build_slate(
    *,
    parent_key: Any,
    parent_rec: Any,
    n_evaluated: int,
    seed_search: int | None,
    active_actions: Sequence[int],
    action_names: Sequence[Any] | None,
    max_actions: int,
    controller_policy_guidance: Any,
    macro_decision: Any,
    macro_state: Any,
    inverse_gate_diag: Any,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    proj: torch.Tensor,
    fp_mode: str,
    q_scale: float,
    q_clip: float,
    poly_degree: int,
    refine_enable: bool,
    refine_cfg: dict[str, Any] | None,
    refine_state: dict[str, Any] | None,
    best_raw_mse_struct: float,
    best_raw_mse: float,
    early_stop_mse: float,
    complexity_penalty: float,
    boost_enable: bool,
    boost_pool_nodes: Sequence[Any],
    boost_pool_phi_fit: torch.Tensor,
    boost_pool_norms_fit: torch.Tensor,
    boost_pool_phi: torch.Tensor,
    boost_pool_norms: torch.Tensor,
    boost_pool_dims: Sequence[Any] | None,
    boost_selection_split: str,
    boost_ridge: float | None,
    boost_include_parent: bool,
    boost_from_scratch_prob: float,
    boost_prune_rel: float,
    boost_max_terms: int,
    boost_topk_try: int,
    boost_min_rel_improve: float,
    max_depth: int,
    nvars: int,
    var_dims: Sequence[Any] | None,
    y_dims: Any,
    reach: Any,
    preview_only: bool = False,
) -> dict[str, Any]:
    return _collect_controller_build_slate_impl(
        parent_key=parent_key,
        parent_rec=parent_rec,
        n_evaluated=n_evaluated,
        seed_search=seed_search,
        active_actions=active_actions,
        action_names=action_names,
        max_actions=max_actions,
        controller_policy_guidance=controller_policy_guidance,
        macro_decision=macro_decision,
        macro_state=macro_state,
        inverse_gate_diag=inverse_gate_diag,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        proj=proj,
        fp_mode=fp_mode,
        q_scale=q_scale,
        q_clip=q_clip,
        poly_degree=poly_degree,
        refine_enable=refine_enable,
        refine_cfg=refine_cfg,
        refine_state=refine_state,
        best_raw_mse_struct=best_raw_mse_struct,
        best_raw_mse=best_raw_mse,
        early_stop_mse=early_stop_mse,
        complexity_penalty=complexity_penalty,
        boost_enable=boost_enable,
        boost_pool_nodes=boost_pool_nodes,
        boost_pool_phi_fit=boost_pool_phi_fit,
        boost_pool_norms_fit=boost_pool_norms_fit,
        boost_pool_phi=boost_pool_phi,
        boost_pool_norms=boost_pool_norms,
        boost_pool_dims=boost_pool_dims,
        boost_selection_split=boost_selection_split,
        boost_ridge=boost_ridge,
        boost_include_parent=boost_include_parent,
        boost_from_scratch_prob=boost_from_scratch_prob,
        boost_prune_rel=boost_prune_rel,
        boost_max_terms=boost_max_terms,
        boost_topk_try=boost_topk_try,
        boost_min_rel_improve=boost_min_rel_improve,
        max_depth=max_depth,
        nvars=nvars,
        var_dims=var_dims,
        y_dims=y_dims,
        reach=reach,
        preview_only=preview_only,
        normalize_actions_fn=_normalize_controller_build_slate_actions,
        controller_selected_action_path_fn=_controller_selected_action_path,
        controller_build_slate_id_fn=_controller_build_slate_id,
        derived_controller_build_rng_fn=_derived_controller_build_rng,
        apply_residual_action_fn=apply_residual_action,
        apply_boost_action_fn=apply_boost_action,
        apply_action_fn=apply_action,
        simplify_fn=simplify,
        node_str_fn=node_str,
        node_size_fn=node_size,
        node_depth_fn=node_depth,
        node_dims_fn=node_dims,
        dims_eq_fn=dims_eq,
        score_expr_fn=score_expr,
        shared_candidate_row_dict_fn=shared_candidate_row_dict,
        mapping_cost_fn=mapping_cost,
        action_name_map=ACTION_NAME,
        path_select_action_ids=(A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_PRUNE),
        residual_action_id=A_RESIDUAL,
        boost_action_id=A_BOOST,
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
) -> dict[str, Any] | None:
    return _score_repair_option_expr_impl(
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
        max_depth=int(max_depth),
        var_dims=var_dims,
        complexity_penalty=float(complexity_penalty),
        score_expr_cfg=score_expr_cfg,
        score_expr_fn=score_expr,
    )


# --- mutation actions ---
A_REPLACE=0
A_WRAP_UNARY=1
A_ADD_RAND=2
A_MUL_RAND=3
A_RESIDUAL=4
A_PRUNE=5
A_CROSSOVER=6
A_BOOST=7
A_INVSTEER=8
A_REPAIR=9
A_HOLESEARCH=10
# Backward-compat aliases (split crossover policies were removed).
A_CROSSOVER_LOCAL=A_CROSSOVER
A_CROSSOVER_FOREIGN=A_CROSSOVER
ACTIONS=[A_REPLACE,A_WRAP_UNARY,A_ADD_RAND,A_MUL_RAND,A_RESIDUAL,A_INVSTEER,A_REPAIR,A_BOOST,A_PRUNE,A_CROSSOVER,A_HOLESEARCH]
ACTION_NAME={
    A_REPLACE:"replace",
    A_WRAP_UNARY:"wrap_un",
    A_ADD_RAND:"add_rand",
    A_MUL_RAND:"mul_rand",
    A_RESIDUAL:"residual",
    A_INVSTEER:"inv_steer",
    A_REPAIR:"repair_option",
    A_BOOST:"boost",
    A_PRUNE:"prune",
    A_CROSSOVER:"crossover",
    A_HOLESEARCH:"hole_search",
}
ACTION_ID_BY_NAME = {v: k for k, v in ACTION_NAME.items()}


def _eval_mapping_total_local(pred, mapping, x=None):
    y_hat = eval_mapping(pred, mapping)
    if isinstance(mapping, dict) and x is not None:
        head_pred = _eval_linear_head(mapping.get("_lin_head", None), x)
        if head_pred is not None and torch.isfinite(head_pred).all():
            y_hat = y_hat + head_pred
    return y_hat

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

_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS = ("replace", "wrap_un", "residual")


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


def apply_action(node, action, rng, max_depth, nvars, var_dims=None, reach=None, path=None):
    return _apply_action_impl(
        node,
        action,
        rng,
        max_depth,
        nvars,
        var_dims=var_dims,
        reach=reach,
        path=path,
        replace_action_id=A_REPLACE,
        wrap_unary_action_id=A_WRAP_UNARY,
        add_rand_action_id=A_ADD_RAND,
        mul_rand_action_id=A_MUL_RAND,
        prune_action_id=A_PRUNE,
        unary_ops=UNARY_OPS,
        select_action_path_fn=_select_action_path,
        action_candidate_paths_fn=_action_candidate_paths,
        coerce_guided_path_fn=_coerce_guided_path,
        node_dims_fn=node_dims,
        dims_eq_fn=dims_eq,
        get_at_fn=get_at,
        replace_at_fn=replace_at,
        rand_node_dim_fn=rand_node_dim,
        rand_node_fn=rand_node,
        node_depth_fn=node_depth,
    )

def apply_crossover_action(
    recipient,
    arch,
    parent_key,
    rng,
    max_depth,
    nvars,
    var_dims=None,
    exploit_frac=0.35,
    exploit_topk=50,
    path=None,
    **_unused,
):
    return _apply_crossover_action_impl(
        recipient,
        arch,
        parent_key,
        rng,
        max_depth,
        nvars,
        var_dims=var_dims,
        exploit_frac=exploit_frac,
        exploit_topk=exploit_topk,
        path=path,
        crossover_action_id=A_CROSSOVER,
        select_action_path_fn=_select_action_path,
        node_dims_fn=node_dims,
        get_at_fn=get_at,
        choose_parent_fn=choose_parent,
        collect_paths_fn=collect_paths,
        replace_at_fn=replace_at,
        node_depth_fn=node_depth,
        dims_eq_fn=dims_eq,
    )

@torch.no_grad()
def apply_residual_action(parent_node, parent_mapping,
                           x_fit, y_fit, x_probe, y_probe,
                           pool_nodes, pool_phi, pool_norms, pool_dims,
                           rng, max_depth, nvars, poly_degree,
                           var_dims=None, topk=3, complexity_penalty=0.0):
    return _apply_residual_action_impl(
        parent_node,
        parent_mapping,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes,
        pool_phi,
        pool_norms,
        pool_dims,
        rng,
        max_depth,
        nvars,
        poly_degree,
        var_dims=var_dims,
        topk=topk,
        complexity_penalty=complexity_penalty,
        eval_node_fn=eval_node,
        eval_mapping_total_fn=_eval_mapping_total_local,
        node_dims_fn=node_dims,
        dims_eq_fn=dims_eq,
        collect_paths_fn=collect_paths,
        get_at_fn=get_at,
        replace_at_fn=replace_at,
        fit_best_fn=fit_best,
        eval_mapping_fn=eval_mapping,
        node_depth_fn=node_depth,
        node_size_fn=node_size,
    )


@torch.no_grad()
def apply_inverse_steering_action(
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
    inverse_spec_enable=False,
    inverse_spec_enum_max_depth=4,
    inverse_spec_enum_max_trees=5000,
    inverse_spec_preview_topk=16,
    inverse_spec_local_score_mode="affine",
    inverse_spec_include_legacy_seed=True,
    inverse_spec_complexity_penalty=0.0,
    inverse_spec_family_battery_enable=False,
    inverse_spec_family_battery_mode="outer",
    inverse_spec_repair_quota=0.0,
    inverse_spec_recursive_enable=True,
    inverse_spec_recursive_max_depth=2,
    inverse_spec_recursive_trigger_rel_mse=0.25,
    inverse_spec_recursive_seed_cap=6,
    inverse_spec_recursive_branch_topk=4,
    inverse_spec_recursive_child_topk=2,
    inverse_spec_constant_lift_route_enable=False,
    inverse_spec_constant_lift_route_topk=2,
    inverse_spec_coordinate_lift_enable=False,
    inverse_spec_coordinate_lift_topk=4,
    inverse_spec_coordinate_lift_mode="both",
    inverse_spec_witness_jets_enable=False,
    inverse_spec_witness_d2_enable=False,
    inverse_spec_witness_max_rows=64,
    inverse_spec_witness_loss_enable=False,
    inverse_spec_witness_grad_weight=1.0,
    inverse_spec_witness_d2_weight=0.0,
    inverse_spec_witness_diag_weight=0.0,
    inverse_spec_witness_physics_weight=0.0,
    inverse_spec_active_var_screen_enable=False,
    inverse_spec_active_var_grad_tol=1.0e-3,
    inverse_spec_active_var_max_count=4,
    inverse_spec_directional_market_enable=False,
    inverse_spec_max_subtree_depth=None,
    inverse_spec_fit_cap=96,
    inverse_spec_probe_cap=192,
    inverse_spec_exact_budget=4,
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
    path_target_modes=None,
    proj=None,
    fp_mode="bits",
    q_scale=2.0,
    q_clip=8.0,
    score_expr_cfg=None,
    return_meta=False,
    repair_tuple_bundle=None,
    repair_tuple_controller_row=None,
    repair_opportunity_controller_enable=False,
    repair_opportunity_bundle=None,
    inverse_spec_regime_metadata=None,
):
    return run_inverse_steering_action(
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
        inverse_spec_enable=inverse_spec_enable,
        inverse_spec_enum_max_depth=inverse_spec_enum_max_depth,
        inverse_spec_enum_max_trees=inverse_spec_enum_max_trees,
        inverse_spec_preview_topk=inverse_spec_preview_topk,
        inverse_spec_local_score_mode=inverse_spec_local_score_mode,
        inverse_spec_include_legacy_seed=inverse_spec_include_legacy_seed,
        inverse_spec_complexity_penalty=inverse_spec_complexity_penalty,
        inverse_spec_family_battery_enable=inverse_spec_family_battery_enable,
        inverse_spec_family_battery_mode=inverse_spec_family_battery_mode,
        inverse_spec_recursive_enable=inverse_spec_recursive_enable,
        inverse_spec_recursive_max_depth=inverse_spec_recursive_max_depth,
        inverse_spec_recursive_trigger_rel_mse=inverse_spec_recursive_trigger_rel_mse,
        inverse_spec_recursive_seed_cap=inverse_spec_recursive_seed_cap,
        inverse_spec_recursive_branch_topk=inverse_spec_recursive_branch_topk,
        inverse_spec_recursive_child_topk=inverse_spec_recursive_child_topk,
        inverse_spec_witness_jets_enable=inverse_spec_witness_jets_enable,
        inverse_spec_witness_d2_enable=inverse_spec_witness_d2_enable,
        inverse_spec_witness_max_rows=inverse_spec_witness_max_rows,
        inverse_spec_witness_loss_enable=inverse_spec_witness_loss_enable,
        inverse_spec_witness_grad_weight=inverse_spec_witness_grad_weight,
        inverse_spec_witness_d2_weight=inverse_spec_witness_d2_weight,
        inverse_spec_witness_diag_weight=inverse_spec_witness_diag_weight,
        inverse_spec_witness_physics_weight=inverse_spec_witness_physics_weight,
        inverse_spec_active_var_screen_enable=inverse_spec_active_var_screen_enable,
        inverse_spec_active_var_grad_tol=inverse_spec_active_var_grad_tol,
        inverse_spec_active_var_max_count=inverse_spec_active_var_max_count,
        inverse_spec_max_subtree_depth=inverse_spec_max_subtree_depth,
        inverse_spec_fit_cap=inverse_spec_fit_cap,
        inverse_spec_probe_cap=inverse_spec_probe_cap,
        inverse_spec_exact_budget=inverse_spec_exact_budget,
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
        candidate_paths=candidate_paths,
        path_target_modes=path_target_modes,
        proj=proj,
        fp_mode=fp_mode,
        q_scale=q_scale,
        q_clip=q_clip,
        score_expr_cfg=score_expr_cfg,
        return_meta=return_meta,
        score_expr_fn=score_expr,
        repair_tuple_bundle=repair_tuple_bundle,
        repair_tuple_controller_row=repair_tuple_controller_row,
        repair_opportunity_controller_enable=repair_opportunity_controller_enable,
        repair_opportunity_bundle=repair_opportunity_bundle,
        inverse_spec_regime_metadata=inverse_spec_regime_metadata,
    )


def run_repair_option(
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
    inverse_spec_enable=False,
    inverse_spec_enum_max_depth=4,
    inverse_spec_enum_max_trees=5000,
    inverse_spec_preview_topk=16,
    inverse_spec_local_score_mode="affine",
    inverse_spec_include_legacy_seed=True,
    inverse_spec_complexity_penalty=0.0,
    inverse_spec_family_battery_enable=False,
    inverse_spec_family_battery_mode="outer",
    inverse_spec_repair_quota=0.0,
    inverse_spec_recursive_enable=True,
    inverse_spec_recursive_max_depth=2,
    inverse_spec_recursive_trigger_rel_mse=0.25,
    inverse_spec_recursive_seed_cap=6,
    inverse_spec_recursive_branch_topk=4,
    inverse_spec_recursive_child_topk=2,
    inverse_spec_constant_lift_route_enable=False,
    inverse_spec_constant_lift_route_topk=2,
    inverse_spec_coordinate_lift_enable=False,
    inverse_spec_coordinate_lift_topk=4,
    inverse_spec_coordinate_lift_mode="both",
    inverse_spec_witness_jets_enable=False,
    inverse_spec_witness_d2_enable=False,
    inverse_spec_witness_max_rows=64,
    inverse_spec_witness_loss_enable=False,
    inverse_spec_witness_grad_weight=1.0,
    inverse_spec_witness_d2_weight=0.0,
    inverse_spec_witness_diag_weight=0.0,
    inverse_spec_witness_physics_weight=0.0,
    inverse_spec_active_var_screen_enable=False,
    inverse_spec_active_var_grad_tol=1.0e-3,
    inverse_spec_active_var_max_count=4,
    inverse_spec_directional_market_enable=False,
    inverse_spec_max_subtree_depth=None,
    inverse_spec_fit_cap=96,
    inverse_spec_probe_cap=192,
    inverse_spec_exact_budget=4,
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
    inverse_action_config: InverseSteeringConfig | Mapping[str, Any] | None = None,
    repair_opportunity_controller_enable: bool = False,
    repair_opportunity_bundle: Mapping[str, Any] | None = None,
):
    inverse_action_config = (
        coerce_inverse_steering_config(inverse_action_config)
        if inverse_action_config is not None
        else coerce_inverse_steering_config(locals())
    )
    return run_repair_option_action(
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
        var_dims=var_dims,
        max_steps=max_steps,
        ancestor_hops=ancestor_hops,
        min_step_rel_improve=min_step_rel_improve,
        max_setup_steps=max_setup_steps,
        setup_step_value_min=setup_step_value_min,
        setup_step_regret_max=setup_step_regret_max,
        setup_step_max_worsen=setup_step_max_worsen,
        initial_path=initial_path,
        initial_candidate_paths=initial_candidate_paths,
        first_step_expr=first_step_expr,
        first_step_meta=first_step_meta,
        current_eff_mse=current_eff_mse,
        complexity_penalty=complexity_penalty,
        proj=proj,
        fp_mode=fp_mode,
        q_scale=q_scale,
        q_clip=q_clip,
        score_expr_cfg=score_expr_cfg,
        return_meta=return_meta,
        score_expr_fn=score_expr,
        inverse_action_fn=apply_inverse_steering_action,
        inverse_action_config=inverse_action_config,
        repair_opportunity_controller_enable=repair_opportunity_controller_enable,
        repair_opportunity_bundle=repair_opportunity_bundle,
    )
