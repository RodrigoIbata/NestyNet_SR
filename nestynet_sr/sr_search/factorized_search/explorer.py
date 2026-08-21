# ruff: noqa: F401, F811
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

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


import inspect as _inspect
import sys as _sys
import types as _types

_log = _logging.getLogger(__name__)

from . import _explorer_actions as _actions
from ._explorer_actions import (
    _use_affine_fast_path,
    _fit_best_with_cfg,
    pb011_function,
    addsum_function,
    poly_function,
    exp_product,
    square_addsum,
    feynman_012,
    feynman_090,
    feynman_028,
    TARGET_FUNCS,
    _coerce_guided_path,
    _action_candidate_paths,
    _select_action_path,
    _normalize_controller_build_slate_actions,
    _collect_controller_build_slate,
    _controller_selected_action_path,
    _score_repair_option_expr,
    A_REPLACE,
    A_WRAP_UNARY,
    A_ADD_RAND,
    A_MUL_RAND,
    A_RESIDUAL,
    A_PRUNE,
    A_CROSSOVER,
    A_BOOST,
    A_INVSTEER,
    A_REPAIR,
    A_HOLESEARCH,
    A_CROSSOVER_LOCAL,
    A_CROSSOVER_FOREIGN,
    ACTIONS,
    ACTION_NAME,
    ACTION_ID_BY_NAME,
    _eval_mapping_total_local,
    _INVERSE_CANDIDATE_META_KEYS,
    _INVERSE_EXTRA_META_KEYS,
    _CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS,
    _tracked_macro_actions,
    _macro_action_fields,
    _macro_decision_log_fields,
    _merge_inverse_proposal_log_fields,
    _merge_repair_option_log_fields,
    apply_action,
    apply_crossover_action,
    apply_residual_action,
    apply_inverse_steering_action,
    run_repair_option,
)

from . import _explorer_scoring as _scoring
from ._explorer_scoring import (
    _balanced_add_tree,
    _strip_scalar_prefix,
    _extract_scalar_core,
    _collect_linear_terms,
    _mapping_equiv_root,
    _compile_linear_combo,
    _harvest_pool_from_archive,
    apply_boost_action,
    fingerprint,
    _negate_smart,
    _pick_best_equiv_score,
    _score_expr_base,
    _score_expr_base_joint_affine,
    _score_expr_base_joint_linear_terms,
    _collect_trig_paths,
    _trig_arg_has_const_scale,
    _collect_log_paths,
    _log_arg_has_const_scale,
    _collect_exp_paths,
    _exp_arg_has_const_scale,
    _collect_sqr_shift_paths,
    _sqr_shift_already_present,
    _collect_sqrt_shift_paths,
    _sqrt_shift_already_present,
    _wrap_param_slots,
    _refine_diag,
    _diag_inc,
    _diag_inc_context,
    _diag_add_time,
    _node_var_indices,
    _refine_tensor_signature,
    _refine_cfg_signature,
    _refine_attempt_cache_key,
    _refine_cache_get,
    _refine_cache_put,
    _prune_mapping_equiv_root_slot_paths,
    _decorate_refine_variants,
    _eval_node_hparam,
    _materialize_hparams,
    _build_init_logs,
    _raw_to_hparams,
    _stable_seed_from_text,
    _select_subset_indices,
    _slice_fit_subset,
    _slice_fit_subset_multi,
    _solve_linear_coeffs,
    _solve_linearized_fit,
    _joint_dataset_weights,
    _solve_linearized_fit_multi,
    _linearized_loss_value,
    _build_single_slot_variant,
    _slot_sensitivity_score,
    _rank_paths_by_sensitivity,
    _variant_has_gate_potential,
    _build_grid_seed_logs,
    _normalize_refine_optimizer,
    _score_refine_raw_log,
    _ranked_grid_refine_seeds,
    _init_logs_from_grid_rank,
    _flatten_add_terms,
    _select_linear_basis_nodes,
    _eval_node_hparam_safe,
    _build_phi_hparam,
    _materialize_linearized_candidate,
    _refine_hparams,
    _refine_budget_left,
    score_expr,
)


# Publicly re-export the canonical engine Explorer so monkeypatches and
# contract-surface checks see one shared class identity.
Explorer = _engine_Explorer




from . import _explorer_brute as _brute
from ._explorer_brute import (
    _init_crossover_policy_stats,
    _finalize_crossover_policy_stats,
    _remove_allowed_action,
    _finalize_action_distribution,
    enumerate_trees,
    enumerate_trees_dim,
    _has_const_zero,
    _dedup_new,
    _auto_brute_depth,
    _enumerate_incremental,
    _enumerate_dim_incremental,
    _build_monomial_ast,
    _monomial_presearch,
    _lorentz_peel_presearch,
    _planck_peel_presearch,
    _hyperbolic_peel_presearch,
    _gaussian_peel_presearch,
    _invtrig_peel_presearch,
    _archive_best_mse,
    _archive_best_structural_mse,
    _promote_structural_shadow_archive,
    _run_brute_phase,
)

# Keep the engine scoring implementation canonical for public explorer entry
# points. The legacy definitions above are left in place to avoid a broad
# rewrite of this compatibility module; exported names and wrappers resolve
# through these globals.
_mapping_equiv_root = _engine_mapping_equiv_root
fingerprint = _engine_fingerprint  # noqa: F811
_harvest_pool_from_archive = _engine_harvest_pool_from_archive
_eval_node_hparam_safe = _engine_eval_node_hparam_safe


def make_engine_refinement_hooks():
    """Return explorer-owned refinement hooks required by engine.scoring."""
    hooks = {}
    for name in _ENGINE_REFINEMENT_HOOK_NAMES:
        try:
            hooks[name] = globals()[name]
        except KeyError as exc:
            raise RuntimeError(f"missing engine refinement hook {name!r}") from exc
    return hooks


def make_engine_runtime_hooks():
    """Return explorer-owned runtime hooks required by engine.search."""
    hooks = {}
    for name in _ENGINE_RUNTIME_HOOK_NAMES:
        try:
            hooks[name] = globals()[name]
        except KeyError as exc:
            raise RuntimeError(f"missing engine runtime hook {name!r}") from exc
    for name in _ENGINE_OPTIONAL_RUNTIME_HOOK_NAMES:
        if name in globals():
            hooks[name] = globals()[name]
    return hooks


def score_expr(*args, **kwargs):  # noqa: F811
    refine_cfg = kwargs.get("refine_cfg", None)
    if isinstance(refine_cfg, Mapping):
        refine_cfg = dict(refine_cfg)
    else:
        refine_cfg = {}
    refine_cfg.setdefault("_legacy_refinement_hooks", make_engine_refinement_hooks())
    kwargs["refine_cfg"] = refine_cfg
    return _engine_score_expr(*args, **kwargs)


# --- core search (reusable) ---

# The active runtime implementation lives in engine.search.
# Keep only the thin wrapper below so controller logic does not diverge
# between this module and the engine implementation.

def run_explorer_core(*args, **kwargs):
    kwargs.setdefault("_explorer_cls", Explorer)
    kwargs.setdefault("_score_expr_fn", score_expr)
    kwargs.setdefault("_harvest_pool_from_archive_fn", _harvest_pool_from_archive)
    kwargs.setdefault("_eval_node_hparam_safe_fn", _eval_node_hparam_safe)
    kwargs.setdefault("_runtime_hooks", make_engine_runtime_hooks())
    return _engine_run_explorer_core(*args, **kwargs)


# --- CLI main ---

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--target",type=str,default="pb011",choices=list(TARGET_FUNCS.keys()))
    p.add_argument("--no_residual",action="store_true",help="disable residual action for A/B comparison")
    p.add_argument("--no_crossover",action="store_true",help="disable crossover action")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--n_iter",type=int,default=20000)
    p.add_argument("--n_fit",type=int,default=512)
    p.add_argument("--n_probe",type=int,default=2048)
    p.add_argument("--max_depth",type=int,default=6)
    p.add_argument("--lo",type=float,default=1.0)
    p.add_argument("--hi",type=float,default=5.0)
    p.add_argument("--dtype",type=str,default="float32")
    p.add_argument("--emb_dim",type=int,default=16)
    p.add_argument("--fp",type=str,default="bits",choices=["bits","quant"])
    p.add_argument("--q_scale",type=float,default=2.0)
    p.add_argument("--q_clip",type=int,default=6)
    p.add_argument("--p_restart",type=float,default=0.20)
    p.add_argument("--exploit_frac",type=float,default=0.35)
    p.add_argument("--exploit_topk",type=int,default=50)
    p.add_argument("--eps_action",type=float,default=0.10)
    p.add_argument("--ucb_action",type=float,default=1.0)
    p.add_argument("--novelty_bonus",type=float,default=0.0)
    p.add_argument("--complexity_penalty",type=float,default=0.0)
    p.add_argument("--actor_critic_best_bonus", type=float, default=0.15)
    p.add_argument("--actor_critic_time_penalty", type=float, default=0.0)
    p.add_argument("--actor_critic_reward_eps", type=float, default=1.0e-30)
    p.add_argument("--actor_critic_descendant_horizon", type=int, default=8)
    p.add_argument("--macro_controller_learned_policy_weight", type=float, default=0.75)
    p.add_argument("--macro_controller_learned_route_weight", type=float, default=0.60)
    p.add_argument("--macro_controller_learned_q_weight", type=float, default=0.75)
    p.add_argument("--macro_controller_learned_value_scale", type=float, default=0.75)
    p.add_argument("--poly_degree",type=int,default=4,help="degree of polynomial mapping (1=affine)")
    p.add_argument("--print_every",type=int,default=2000)
    p.add_argument("--report_topk",type=int,default=10)
    p.add_argument("--residual_topk",type=int,default=5)
    p.add_argument("--inverse_steering_enable",action="store_true")
    p.add_argument("--inverse_max_paths",type=int,default=12)
    p.add_argument("--inverse_topk_terms",type=int,default=6)
    p.add_argument("--inverse_min_valid_frac",type=float,default=0.25)
    p.add_argument("--inverse_min_confidence",type=float,default=0.10)
    p.add_argument("--inverse_confidence_mode",type=str,default="conditioning",choices=["conditioning","heuristic"])
    p.add_argument("--inverse_confidence_target_gain",type=float,default=4.0)
    p.add_argument("--inverse_confidence_floor",type=float,default=0.05)
    p.add_argument("--inverse_branch_beam_width",type=int,default=1)
    p.add_argument("--inverse_micro_search_enable",action="store_true")
    p.add_argument("--inverse_micro_search_max_depth",type=int,default=3)
    p.add_argument("--inverse_micro_search_beam_width",type=int,default=24)
    p.add_argument("--inverse_micro_search_topk",type=int,default=16)
    p.add_argument("--inverse_micro_search_seed_terms",type=int,default=8)
    p.add_argument("--inverse_local_score_mode",type=str,default="affine",choices=["strict","affine","fitbest"])
    p.add_argument("--inverse_spec_enable",action="store_true")
    p.add_argument("--inverse_spec_enum_max_depth",type=int,default=4)
    p.add_argument("--inverse_spec_enum_max_trees",type=int,default=5000)
    p.add_argument("--inverse_spec_preview_topk",type=int,default=16)
    p.add_argument("--inverse_spec_local_score_mode",type=str,default="affine",choices=["strict","affine","fitbest"])
    p.add_argument("--inverse_spec_disable_legacy_seed",action="store_true")
    p.add_argument("--inverse_spec_complexity_penalty",type=float,default=0.0)
    p.add_argument("--inverse_spec_family_battery_enable",action="store_true")
    p.add_argument("--inverse_spec_family_battery_mode",type=str,default="outer",choices=["outer","expanded"])
    p.add_argument("--inverse_spec_repair_quota",type=float,default=0.0)
    p.add_argument("--inverse_spec_recursive_disable",action="store_true")
    p.add_argument("--inverse_spec_recursive_max_depth",type=int,default=2)
    p.add_argument("--inverse_spec_recursive_trigger_rel_mse",type=float,default=0.25)
    p.add_argument("--inverse_spec_recursive_seed_cap",type=int,default=6)
    p.add_argument("--inverse_spec_recursive_branch_topk",type=int,default=4)
    p.add_argument("--inverse_spec_recursive_child_topk",type=int,default=2)
    p.add_argument("--inverse_spec_max_subtree_depth",type=int,default=None)
    p.add_argument("--inverse_spec_fit_cap",type=int,default=96)
    p.add_argument("--inverse_spec_probe_cap",type=int,default=192)
    p.add_argument("--inverse_spec_exact_budget",type=int,default=4)
    p.add_argument("--inverse_target_mode",type=str,default="robust",choices=["robust","full","identity","affine","simple"])
    p.add_argument("--inverse_full_mapping_penalty",type=float,default=0.75)
    p.add_argument("--inverse_exact_simple_target_bonus",type=float,default=0.10)
    p.add_argument("--inverse_additive_descend_penalty",type=float,default=0.15)
    p.add_argument("--inverse_nonadditive_leaf_penalty",type=float,default=0.20)
    p.add_argument("--inverse_exact_path_eta",type=float,default=0.98)
    p.add_argument("--inverse_exact_transport_min_lin_rel",type=float,default=0.0)
    p.add_argument("--inverse_gate_disable",action="store_true")
    p.add_argument("--inverse_gate_min_depth",type=int,default=4)
    p.add_argument("--inverse_gate_min_size",type=int,default=6)
    p.add_argument("--inverse_gate_max_paths",type=int,default=6)
    p.add_argument("--inverse_gate_min_structural_score",type=float,default=0.75)
    p.add_argument("--inverse_gate_min_weighted_rel_gain",type=float,default=0.05)
    p.add_argument("--inverse_gate_structural_bias",type=float,default=0.20)
    p.add_argument("--inverse_periodic_min_valid_scale",type=float,default=1.25)
    p.add_argument("--inverse_periodic_min_confidence_scale",type=float,default=1.35)
    p.add_argument("--inverse_periodic_path_penalty",type=float,default=0.65)
    p.add_argument("--inverse_nonperiodic_muldiv_bonus",type=float,default=0.10)
    p.add_argument("--inverse_nonperiodic_explogsqrt_bonus",type=float,default=0.05)
    p.add_argument("--inverse_branch_ambiguity_penalty",type=float,default=0.50)
    p.add_argument("--inverse_transport_min_lin_rel",type=float,default=0.02)
    p.add_argument("--inverse_transport_min_effective_n",type=float,default=8.0)
    p.add_argument("--inverse_gate_best_factor",type=float,default=20.0)
    p.add_argument("--inverse_gate_warmup",type=int,default=0)
    p.add_argument("--controller_build_slate_enable", action="store_true")
    p.add_argument("--controller_build_slate_actions", type=str, default="replace,wrap_un,residual")
    p.add_argument("--controller_build_slate_max_actions", type=int, default=3)
    p.add_argument("--repair_controller_credible_route_enable", action="store_true")
    p.add_argument("--repair_opportunity_controller_enable", action="store_true")
    p.add_argument("--repair_opportunity_controller_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_enable", action="store_true")
    p.add_argument("--repair_controller_route_compare_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_repair_tuple_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_build_tuple_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_max_repair_prob", type=float, default=0.35)
    p.add_argument("--repair_controller_route_compare_min_build_margin", type=float, default=0.05)
    p.add_argument("--repair_controller_max_setup_steps", type=int, default=0)
    p.add_argument("--repair_controller_setup_step_value_min", type=float, default=0.10)
    p.add_argument("--repair_controller_setup_step_regret_max", type=float, default=0.50)
    p.add_argument("--repair_controller_setup_step_max_worsen", type=float, default=0.05)
    p.add_argument("--hole_search_enable",action="store_true",default=False)
    p.add_argument("--no_hole_search",dest="hole_search_enable",action="store_false")
    p.add_argument("--hole_search_quota",type=float,default=0.10)
    p.add_argument("--hole_search_exact_budget",type=int,default=2)
    p.add_argument("--hole_search_cooldown_iters",type=int,default=32)
    p.add_argument("--hole_search_mine_cooldown_iters",type=int,default=50)
    p.add_argument("--hole_search_max_frontier",type=int,default=128)
    p.add_argument("--hole_search_enum_max_depth",type=int,default=4)
    p.add_argument("--hole_search_enum_max_trees",type=int,default=3000)
    p.add_argument("--hole_search_preview_topk",type=int,default=8)
    p.add_argument("--hole_search_solver_market_enable",action="store_true")
    p.add_argument("--hole_search_solver_market_preview_topk",type=int,default=4)
    p.add_argument("--hole_search_solver_market_exact_topk",type=int,default=2)
    p.add_argument("--hole_search_solver_market_proposal_objects_enable",action="store_true")
    p.add_argument("--inverse_spec_recursive_sr_enable",action="store_true")
    p.add_argument("--inverse_spec_recursive_sr_preview_topk",type=int,default=4)
    p.add_argument("--inverse_spec_recursive_sr_exact_budget",type=int,default=2)
    p.add_argument("--inverse_spec_constant_lift_route_enable",action="store_true")
    p.add_argument("--inverse_spec_constant_lift_route_topk",type=int,default=2)
    p.add_argument("--inverse_spec_coordinate_lift_enable",action="store_true")
    p.add_argument("--inverse_spec_coordinate_lift_topk",type=int,default=4)
    p.add_argument("--inverse_spec_coordinate_lift_mode",type=str,default="both",choices=["single_index","invariant","both"])
    p.add_argument("--inverse_spec_tangent_edit_enable",action="store_true")
    p.add_argument("--inverse_spec_tangent_edit_topk",type=int,default=8)
    p.add_argument("--inverse_spec_soft_edit_enable",action="store_true")
    p.add_argument("--inverse_spec_soft_edit_steps",type=int,default=64)
    p.add_argument("--inverse_spec_soft_edit_l1",type=float,default=1.0e-3)
    p.add_argument("--inverse_spec_witness_jets_enable",action="store_true")
    p.add_argument("--inverse_spec_witness_d2_enable",action="store_true")
    p.add_argument("--inverse_spec_witness_max_rows",type=int,default=64)
    p.add_argument("--inverse_spec_witness_loss_enable",action="store_true")
    p.add_argument("--inverse_spec_witness_grad_weight",type=float,default=1.0)
    p.add_argument("--inverse_spec_witness_d2_weight",type=float,default=0.0)
    p.add_argument("--inverse_spec_witness_diag_weight",type=float,default=0.0)
    p.add_argument("--inverse_spec_witness_physics_weight",type=float,default=0.0)
    p.add_argument("--inverse_spec_active_var_screen_enable",action="store_true")
    p.add_argument("--inverse_spec_active_var_grad_tol",type=float,default=1.0e-3)
    p.add_argument("--inverse_spec_active_var_max_count",type=int,default=4)
    p.add_argument("--inverse_spec_directional_market_enable",action="store_true")
    p.add_argument("--hole_search_tournament_enable",type=int,default=1)
    p.add_argument("--hole_search_tournament_n",type=int,default=8)
    p.add_argument("--hole_search_tournament_elite_k",type=int,default=2)
    p.add_argument("--hole_search_tournament_preview_trees",type=int,default=64)
    p.add_argument("--scheduler_enable",action="store_true")
    p.add_argument("--scheduler_control",action="store_true")
    p.add_argument("--scheduler_bundle_path",type=str,default="")
    p.add_argument("--scheduler_budget_ladder",type=str,default="1,2,4,8")
    p.add_argument("--scheduler_build_preview_only",type=int,default=1)
    p.add_argument("--scheduler_fallback_min_confidence",type=float,default=0.0)
    p.add_argument("--scheduler_acquisition_threshold",type=float,default=0.25)
    p.add_argument("--scheduler_uncertainty_bonus",type=float,default=0.05)
    p.add_argument("--scheduler_witness_energy_enable",action="store_true")
    p.add_argument(
        "--research_profile",
        type=str,
        default="legacy",
        choices=list(RESEARCH_PROFILE_NAMES),
        help="Named factorized symbolic search research profile preset; defaults to legacy behavior.",
    )
    p.add_argument("--save_json",type=str,default="")
    a=p.parse_args()
    _resolved_research_profile, _research_profile_overrides = resolve_engine_research_profile(
        getattr(a, "research_profile", "legacy")
    )
    for _key, _value in _research_profile_overrides.items():
        setattr(a, _key, _value)

    nvars, target_fn, y_dims, var_dims = TARGET_FUNCS[a.target]
    dtype={"float32":torch.float32,"float64":torch.float64}.get(a.dtype,torch.float32)

    arch = run_explorer_core(
        target_fn, nvars,
        n_iter=a.n_iter, max_depth=a.max_depth, poly_degree=a.poly_degree,
        lo=a.lo, hi=a.hi, seed=a.seed,
        var_dims=var_dims, y_dims=y_dims,
        dtype=dtype,
        no_residual=a.no_residual, no_crossover=a.no_crossover, residual_topk=a.residual_topk,
        inverse_steering_enable=a.inverse_steering_enable,
        inverse_max_paths=a.inverse_max_paths,
        inverse_topk_terms=a.inverse_topk_terms,
        inverse_min_valid_frac=a.inverse_min_valid_frac,
        inverse_min_confidence=a.inverse_min_confidence,
        inverse_confidence_mode=a.inverse_confidence_mode,
        inverse_confidence_target_gain=a.inverse_confidence_target_gain,
        inverse_confidence_floor=a.inverse_confidence_floor,
        inverse_branch_beam_width=a.inverse_branch_beam_width,
        inverse_micro_search_enable=a.inverse_micro_search_enable,
        inverse_micro_search_max_depth=a.inverse_micro_search_max_depth,
        inverse_micro_search_beam_width=a.inverse_micro_search_beam_width,
        inverse_micro_search_topk=a.inverse_micro_search_topk,
        inverse_micro_search_seed_terms=a.inverse_micro_search_seed_terms,
        inverse_local_score_mode=a.inverse_local_score_mode,
        inverse_spec_enable=a.inverse_spec_enable,
        inverse_spec_enum_max_depth=a.inverse_spec_enum_max_depth,
        inverse_spec_enum_max_trees=a.inverse_spec_enum_max_trees,
        inverse_spec_preview_topk=a.inverse_spec_preview_topk,
        inverse_spec_local_score_mode=a.inverse_spec_local_score_mode,
        inverse_spec_include_legacy_seed=(not a.inverse_spec_disable_legacy_seed),
        inverse_spec_complexity_penalty=a.inverse_spec_complexity_penalty,
        inverse_spec_family_battery_enable=a.inverse_spec_family_battery_enable,
        inverse_spec_family_battery_mode=a.inverse_spec_family_battery_mode,
        inverse_spec_repair_quota=a.inverse_spec_repair_quota,
        inverse_spec_recursive_enable=(not a.inverse_spec_recursive_disable),
        inverse_spec_recursive_max_depth=a.inverse_spec_recursive_max_depth,
        inverse_spec_recursive_trigger_rel_mse=a.inverse_spec_recursive_trigger_rel_mse,
        inverse_spec_recursive_seed_cap=a.inverse_spec_recursive_seed_cap,
        inverse_spec_recursive_branch_topk=a.inverse_spec_recursive_branch_topk,
        inverse_spec_recursive_child_topk=a.inverse_spec_recursive_child_topk,
        inverse_spec_max_subtree_depth=a.inverse_spec_max_subtree_depth,
        inverse_spec_fit_cap=a.inverse_spec_fit_cap,
        inverse_spec_probe_cap=a.inverse_spec_probe_cap,
        inverse_spec_exact_budget=a.inverse_spec_exact_budget,
        inverse_target_mode=a.inverse_target_mode,
        inverse_full_mapping_penalty=a.inverse_full_mapping_penalty,
        inverse_exact_simple_target_bonus=a.inverse_exact_simple_target_bonus,
        inverse_additive_descend_penalty=a.inverse_additive_descend_penalty,
        inverse_nonadditive_leaf_penalty=a.inverse_nonadditive_leaf_penalty,
        inverse_exact_path_eta=a.inverse_exact_path_eta,
        inverse_exact_transport_min_lin_rel=a.inverse_exact_transport_min_lin_rel,
        inverse_gate_enable=(not a.inverse_gate_disable),
        inverse_gate_warmup=a.inverse_gate_warmup,
        inverse_gate_best_factor=a.inverse_gate_best_factor,
        inverse_gate_min_depth=a.inverse_gate_min_depth,
        inverse_gate_min_size=a.inverse_gate_min_size,
        inverse_gate_max_paths=a.inverse_gate_max_paths,
        inverse_gate_min_structural_score=a.inverse_gate_min_structural_score,
        inverse_gate_min_weighted_rel_gain=a.inverse_gate_min_weighted_rel_gain,
        inverse_gate_structural_bias=a.inverse_gate_structural_bias,
        inverse_periodic_min_valid_scale=a.inverse_periodic_min_valid_scale,
        inverse_periodic_min_confidence_scale=a.inverse_periodic_min_confidence_scale,
        inverse_periodic_path_penalty=a.inverse_periodic_path_penalty,
        inverse_nonperiodic_muldiv_bonus=a.inverse_nonperiodic_muldiv_bonus,
        inverse_nonperiodic_explogsqrt_bonus=a.inverse_nonperiodic_explogsqrt_bonus,
        inverse_branch_ambiguity_penalty=a.inverse_branch_ambiguity_penalty,
        inverse_transport_min_lin_rel=a.inverse_transport_min_lin_rel,
        inverse_transport_min_effective_n=a.inverse_transport_min_effective_n,
        controller_build_slate_enable=a.controller_build_slate_enable,
        controller_build_slate_actions=[s.strip() for s in str(a.controller_build_slate_actions).split(",") if s.strip()],
        controller_build_slate_max_actions=a.controller_build_slate_max_actions,
        repair_controller_credible_route_enable=a.repair_controller_credible_route_enable,
        repair_opportunity_controller_enable=a.repair_opportunity_controller_enable,
        repair_opportunity_controller_path=a.repair_opportunity_controller_path,
        repair_controller_route_compare_enable=a.repair_controller_route_compare_enable,
        repair_controller_route_compare_path=a.repair_controller_route_compare_path,
        repair_controller_route_compare_repair_tuple_path=a.repair_controller_route_compare_repair_tuple_path,
        repair_controller_route_compare_build_tuple_path=a.repair_controller_route_compare_build_tuple_path,
        repair_controller_route_compare_max_repair_prob=a.repair_controller_route_compare_max_repair_prob,
        repair_controller_route_compare_min_build_margin=a.repair_controller_route_compare_min_build_margin,
        repair_controller_max_setup_steps=a.repair_controller_max_setup_steps,
        repair_controller_setup_step_value_min=a.repair_controller_setup_step_value_min,
        repair_controller_setup_step_regret_max=a.repair_controller_setup_step_regret_max,
        repair_controller_setup_step_max_worsen=a.repair_controller_setup_step_max_worsen,
        hole_search_enable=a.hole_search_enable,
        hole_search_quota=a.hole_search_quota,
        hole_search_exact_budget=a.hole_search_exact_budget,
        hole_search_cooldown_iters=a.hole_search_cooldown_iters,
        hole_search_mine_cooldown_iters=a.hole_search_mine_cooldown_iters,
        hole_search_max_frontier=a.hole_search_max_frontier,
        hole_search_enum_max_depth=a.hole_search_enum_max_depth,
        hole_search_enum_max_trees=a.hole_search_enum_max_trees,
        hole_search_preview_topk=a.hole_search_preview_topk,
        hole_search_solver_market_enable=a.hole_search_solver_market_enable,
        hole_search_solver_market_preview_topk=a.hole_search_solver_market_preview_topk,
        hole_search_solver_market_exact_topk=a.hole_search_solver_market_exact_topk,
        hole_search_solver_market_proposal_objects_enable=a.hole_search_solver_market_proposal_objects_enable,
        inverse_spec_recursive_sr_enable=a.inverse_spec_recursive_sr_enable,
        inverse_spec_recursive_sr_preview_topk=a.inverse_spec_recursive_sr_preview_topk,
        inverse_spec_recursive_sr_exact_budget=a.inverse_spec_recursive_sr_exact_budget,
        inverse_spec_constant_lift_route_enable=a.inverse_spec_constant_lift_route_enable,
        inverse_spec_constant_lift_route_topk=a.inverse_spec_constant_lift_route_topk,
        inverse_spec_coordinate_lift_enable=a.inverse_spec_coordinate_lift_enable,
        inverse_spec_coordinate_lift_topk=a.inverse_spec_coordinate_lift_topk,
        inverse_spec_coordinate_lift_mode=a.inverse_spec_coordinate_lift_mode,
        inverse_spec_tangent_edit_enable=a.inverse_spec_tangent_edit_enable,
        inverse_spec_tangent_edit_topk=a.inverse_spec_tangent_edit_topk,
        inverse_spec_soft_edit_enable=a.inverse_spec_soft_edit_enable,
        inverse_spec_soft_edit_steps=a.inverse_spec_soft_edit_steps,
        inverse_spec_soft_edit_l1=a.inverse_spec_soft_edit_l1,
        inverse_spec_witness_jets_enable=a.inverse_spec_witness_jets_enable,
        inverse_spec_witness_d2_enable=a.inverse_spec_witness_d2_enable,
        inverse_spec_witness_max_rows=a.inverse_spec_witness_max_rows,
        inverse_spec_witness_loss_enable=a.inverse_spec_witness_loss_enable,
        inverse_spec_witness_grad_weight=a.inverse_spec_witness_grad_weight,
        inverse_spec_witness_d2_weight=a.inverse_spec_witness_d2_weight,
        inverse_spec_witness_diag_weight=a.inverse_spec_witness_diag_weight,
        inverse_spec_witness_physics_weight=a.inverse_spec_witness_physics_weight,
        inverse_spec_active_var_screen_enable=a.inverse_spec_active_var_screen_enable,
        inverse_spec_active_var_grad_tol=a.inverse_spec_active_var_grad_tol,
        inverse_spec_active_var_max_count=a.inverse_spec_active_var_max_count,
        inverse_spec_directional_market_enable=a.inverse_spec_directional_market_enable,
        scheduler_enable=a.scheduler_enable,
        scheduler_advisory_only=(not a.scheduler_control),
        scheduler_bundle_path=a.scheduler_bundle_path,
        scheduler_budget_ladder=[
            int(s.strip())
            for s in str(a.scheduler_budget_ladder).split(",")
            if s.strip()
        ],
        scheduler_build_preview_only=bool(a.scheduler_build_preview_only),
        scheduler_fallback_min_confidence=a.scheduler_fallback_min_confidence,
        scheduler_acquisition_threshold=a.scheduler_acquisition_threshold,
        scheduler_uncertainty_bonus=a.scheduler_uncertainty_bonus,
        scheduler_witness_energy_enable=a.scheduler_witness_energy_enable,
        n_fit=a.n_fit, n_probe=a.n_probe,
        emb_dim=a.emb_dim, fp_mode=a.fp, q_scale=a.q_scale, q_clip=a.q_clip,
        p_restart=a.p_restart, exploit_frac=a.exploit_frac, exploit_topk=a.exploit_topk,
        eps_action=a.eps_action, ucb_action=a.ucb_action,
        novelty_bonus=a.novelty_bonus, complexity_penalty=a.complexity_penalty,
        actor_critic_best_bonus=a.actor_critic_best_bonus,
        actor_critic_time_penalty=a.actor_critic_time_penalty,
        actor_critic_reward_eps=a.actor_critic_reward_eps,
        actor_critic_descendant_horizon=a.actor_critic_descendant_horizon,
        macro_controller_learned_policy_weight=a.macro_controller_learned_policy_weight,
        macro_controller_learned_route_weight=a.macro_controller_learned_route_weight,
        macro_controller_learned_q_weight=a.macro_controller_learned_q_weight,
        macro_controller_learned_value_scale=a.macro_controller_learned_value_scale,
        print_every=a.print_every, label=a.target,
    )

    print(f"\ndone eval {arch.n_eval} residual_basins {len(arch.d)} avg_vis {arch.n_eval/max(1,len(arch.d)):.2f}")
    cps = getattr(arch, "crossover_policy_stats", None)
    ad = getattr(arch, "action_distribution", None)
    if isinstance(cps, dict):
        pol_rows = []
        for pol, st in cps.items():
            sel = int(st.get("selected", 0))
            if sel <= 0:
                continue
            pol_rows.append(
                f"{pol} sel={sel} prop={int(st.get('proposed', 0))} acc={int(st.get('accepted', 0))}"
            )
        if pol_rows:
            print("crossover policies: " + " | ".join(pol_rows))
    if isinstance(ad, dict):
        cc = ad.get("counts", {})
        tot = int(ad.get("total_selected", 0))
        if isinstance(cc, dict) and tot > 0:
            top = sorted(cc.items(), key=lambda kv: kv[1], reverse=True)
            top = [f"{k}:{int(v)}" for k, v in top if int(v) > 0]
            if top:
                print(f"action distribution (selected, n={tot}): {' '.join(top)}")
    print("top best:")
    for r in arch.best(a.report_topk):
        print(f"  mse {r.best_mse:.6g}  size {node_size(r.best_expr)}  depth {node_depth(r.best_expr)}  {node_str(r.best_expr)}")

    if a.save_json:
        out={
            "research_profile": str(_resolved_research_profile),
            "n_eval":arch.n_eval,
            "n_residual_basins":len(arch.d),
            "residual_basins":[],
        }
        if isinstance(cps, dict):
            out["crossover_policy_stats"] = cps
        if isinstance(ad, dict):
            out["action_distribution"] = ad
        igs = getattr(arch, "inverse_gate_stats", None)
        if isinstance(igs, dict):
            out["inverse_gate_stats"] = igs
        for k,r in arch.d.items():
            m = r.mapping
            m_serial = {"kind": m["kind"]}
            if m["kind"] == "poly":
                m_serial["coeffs"] = [float(c) for c in m["coeffs"]]
                m_serial["mu"] = m["mu"]; m_serial["std"] = m["std"]
            elif m["kind"] == "power":
                m_serial["log_a"] = m["log_a"]; m_serial["b"] = m["b"]
            elif m["kind"] == "pade":
                m_serial["numer"] = [float(c) for c in m["numer"]]
                m_serial["denom"] = [float(c) for c in m["denom"]]
                m_serial["mu"] = m["mu"]; m_serial["std"] = m["std"]
            elif m["kind"] == "sine":
                m_serial["A"] = m["A"]; m_serial["B"] = m["B"]
                m_serial["c"] = m["c"]; m_serial["omega"] = m["omega"]
                m_serial["mu"] = m["mu"]; m_serial["std"] = m["std"]
            elif m["kind"] == "exp":
                m_serial["a"] = m["a"]; m_serial["b"] = m["b"]
                m_serial["c"] = m["c"]
                m_serial["mu"] = m["mu"]; m_serial["std"] = m["std"]
            out["residual_basins"].append({
                "key": int(k) if isinstance(k, int) else list(k),
                "visits": r.visits,
                "visits_since_improve": int(getattr(r, "visits_since_improve", r.visits)),
                "last_improve_eval": int(getattr(r, "last_improve_eval", 0)),
                "best_mse": r.best_mse,
                "best_raw_mse": float(getattr(r, "best_raw_mse", r.best_mse)),
                "mapping": m_serial,
                "expr": node_str(r.best_expr),
            })
        with open(a.save_json,"w") as f:
            json.dump(out,f,indent=2)
        print(f"wrote {a.save_json}")

if __name__=="__main__":
    main()

_IMPL_MODULES = (_actions, _scoring, _brute)
_GROUP_DEFINITION_NAMES = {"_actions":["_use_affine_fast_path","_fit_best_with_cfg","pb011_function","addsum_function","poly_function","exp_product","square_addsum","feynman_012","feynman_090","feynman_028","_coerce_guided_path","_action_candidate_paths","_select_action_path","_normalize_controller_build_slate_actions","_collect_controller_build_slate","_controller_selected_action_path","_score_repair_option_expr","_eval_mapping_total_local","_tracked_macro_actions","_macro_action_fields","_macro_decision_log_fields","_merge_inverse_proposal_log_fields","_merge_repair_option_log_fields","apply_action","apply_crossover_action","apply_residual_action","apply_inverse_steering_action","run_repair_option"],"_scoring":["_balanced_add_tree","_strip_scalar_prefix","_extract_scalar_core","_collect_linear_terms","_mapping_equiv_root","_compile_linear_combo","_harvest_pool_from_archive","apply_boost_action","fingerprint","_negate_smart","_pick_best_equiv_score","_score_expr_base","_score_expr_base_joint_affine","_score_expr_base_joint_linear_terms","_collect_trig_paths","_trig_arg_has_const_scale","_collect_log_paths","_log_arg_has_const_scale","_collect_exp_paths","_exp_arg_has_const_scale","_collect_sqr_shift_paths","_sqr_shift_already_present","_collect_sqrt_shift_paths","_sqrt_shift_already_present","_wrap_param_slots","_refine_diag","_diag_inc","_diag_inc_context","_diag_add_time","_node_var_indices","_refine_tensor_signature","_refine_cfg_signature","_refine_attempt_cache_key","_refine_cache_get","_refine_cache_put","_prune_mapping_equiv_root_slot_paths","_decorate_refine_variants","_eval_node_hparam","_materialize_hparams","_build_init_logs","_raw_to_hparams","_stable_seed_from_text","_select_subset_indices","_slice_fit_subset","_slice_fit_subset_multi","_solve_linear_coeffs","_solve_linearized_fit","_joint_dataset_weights","_solve_linearized_fit_multi","_linearized_loss_value","_build_single_slot_variant","_slot_sensitivity_score","_rank_paths_by_sensitivity","_variant_has_gate_potential","_build_grid_seed_logs","_normalize_refine_optimizer","_score_refine_raw_log","_ranked_grid_refine_seeds","_init_logs_from_grid_rank","_flatten_add_terms","_select_linear_basis_nodes","_eval_node_hparam_safe","_build_phi_hparam","_materialize_linearized_candidate","_refine_hparams","_refine_budget_left","score_expr"],"_brute":["_init_crossover_policy_stats","_finalize_crossover_policy_stats","_remove_allowed_action","_finalize_action_distribution","enumerate_trees","enumerate_trees_dim","_has_const_zero","_dedup_new","_auto_brute_depth","_enumerate_incremental","_enumerate_dim_incremental","_build_monomial_ast","_monomial_presearch","_lorentz_peel_presearch","_planck_peel_presearch","_hyperbolic_peel_presearch","_gaussian_peel_presearch","_invtrig_peel_presearch","_archive_best_mse","_archive_best_structural_mse","_promote_structural_shadow_archive","_run_brute_phase"]}
_SYNC_GLOBAL_NAMES = frozenset(["argparse","math","random","json","hashlib","time","itertools","Any","Mapping","Sequence","torch","make_additive_basis_transition","InverseSteeringConfig","coerce_inverse_steering_config","_apply_action_impl","_apply_crossover_action_impl","_apply_residual_action_impl","mapping_cost","ResidualBasinArchive","Elite","Rec","_ENGINE_REFINEMENT_HOOK_NAMES","_engine_eval_node_hparam_safe","_engine_harvest_pool_from_archive","_engine_mapping_equiv_root","_engine_fingerprint","_engine_score_expr","_engine_Explorer","_ENGINE_RUNTIME_HOOK_NAMES","_ENGINE_OPTIONAL_RUNTIME_HOOK_NAMES","_engine_run_explorer_core","CandidateStateFeatures","InverseSteeringPotential","PathStateFeatures","build_controller_state_record","coerce_repair_feature_row","RepairControllerFeatureRecord","_collect_controller_build_slate_impl","_controller_selected_action_path_impl","_normalize_controller_build_slate_actions_impl","_annotate_inverse_experiment_lineage","_choose_repair_execution_preview","_credible_route_compare_decision","_credible_route_preview_repair_opportunity_rows","_controller_build_slate_id","_derived_controller_build_rng","_logged_action_path_from_row","_preview_child_eff_mse","_repair_route_compare_decision","_serialize_lineage_key","choose_parent","choose_parent_repair_aware","load_repair_critic_bundle","predict_repair_build_route","predict_repair_controller_heads","load_opportunity_bundle","predict_opportunity_slate","RESEARCH_PROFILE_NAMES","resolve_engine_research_profile","shared_candidate_row_dict","MacroController","build_macro_controller_state","_logging","BINARY_OPS","UNARY_OPS","build_pool","cap_depth","collect_paths","compute_reachable","dim_round","dims_eq","eval_node","get_at","node_cost_physics_prior","node_depth","node_dims","node_size","node_str","rand_node","rand_node_dim","replace_at","sample_box","set_dim_precision","simplify","_shared_enumerate_trees","_shared_enumerate_trees_dim","_mapping_nparams","eval_exp_mapping","eval_mapping","eval_pade","eval_poly","eval_power","eval_sine","fit_best","fit_exp_mapping","fit_pade","fit_poly","fit_power","fit_sine","mean_squared_error_same_shape","mapping_is_structural","_use_affine_fast_path","_fit_best_with_cfg","InverseStep","InverseTarget","_blend_inverse_backprop_target","_bool_col","_cheap_affine_probe_stats_from_preds","_collect_nodes_preorder","_combine_inverse_confidence","_compute_path_influences","_conditioning_confidence_from_gain","_conditioning_point_weight_from_gain","_effective_sample_size","_ensure_col","_estimate_path_transport_scores","_eval_linear_head","_finite_mask","_fit_affine_mapping_from_pair","_invert_binary_context","_invert_shifted_sinusoid","_invert_shifted_sinusoid_branches","_invert_unary_context","_invert_unary_context_branches","_linearized_residual_gain","_mapping_inverse_point_weight","_mapping_output_derivative","_mask_fraction","_masked_point_weight","_normalize_inverse_local_score_mode","_normalize_inverse_target_mode","_path_transport_scalar","_prepare_nonnegative_weights","_score_inverse_local_predictions","_score_predictions_on_target","_slice_by_mask","_weighted_centered_mse","_weighted_inner_cols","_weighted_mse_cols","eval_mapping_total","invert_context_target","invert_context_target_beam","invert_mapping_target","_inverse_target_mode_rows","_deterministic_row_subset","_eval_quantized_monomial_from_pool","_inverse_additive_combo_candidates","_inverse_branch_beam_factor","_inverse_collect_local_repair_candidates","_inverse_effective_branch_beam_width","_inverse_effective_thresholds","_inverse_family_gain_scale","_inverse_mapping_static_weight","_inverse_muldiv_monomial_candidates","_inverse_path_cut_factor","_inverse_path_profile","_inverse_pool_shortlist","_inverse_rank_local_repair_candidates","_inverse_sqrt_quadratic_candidates","_inverse_static_path_score","_inverse_subtree_micro_search","_mapping_cache_signature","_mapping_kind_lower","_node_pow_small_int","_pool_cache_signature","_quantize_monomial_exponent","_weighted_linear_fit","estimate_inverse_steering_potential","run_inverse_steering_action","_score_repair_option_expr_impl","run_repair_option_action","_actor_critic_reward_terms","_analytic_repair_controller_score","_hybrid_repair_controller_scores","_normalize_repair_controller_critic_mode","_repair_controller_component_gate","_repair_controller_path_policy","_repair_controller_relation_score","_repair_controller_stagnation_state","_repair_controller_threshold","_repair_controller_weights","_repair_option_candidate_paths","_repair_parent_record_attempt","_repair_parent_preview_retry_gate","_repair_parent_retry_gate","_repair_preview_signature","_log","pb011_function","addsum_function","poly_function","exp_product","square_addsum","feynman_012","feynman_090","feynman_028","TARGET_FUNCS","_coerce_guided_path","_action_candidate_paths","_select_action_path","_normalize_controller_build_slate_actions","_collect_controller_build_slate","_controller_selected_action_path","_score_repair_option_expr","A_REPLACE","A_WRAP_UNARY","A_ADD_RAND","A_MUL_RAND","A_RESIDUAL","A_PRUNE","A_CROSSOVER","A_BOOST","A_INVSTEER","A_REPAIR","A_HOLESEARCH","A_CROSSOVER_LOCAL","A_CROSSOVER_FOREIGN","ACTIONS","ACTION_NAME","ACTION_ID_BY_NAME","_eval_mapping_total_local","_INVERSE_CANDIDATE_META_KEYS","_INVERSE_EXTRA_META_KEYS","_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS","_tracked_macro_actions","_macro_action_fields","_macro_decision_log_fields","_merge_inverse_proposal_log_fields","_merge_repair_option_log_fields","apply_action","apply_crossover_action","apply_residual_action","apply_inverse_steering_action","run_repair_option","_balanced_add_tree","_strip_scalar_prefix","_extract_scalar_core","_collect_linear_terms","_mapping_equiv_root","_compile_linear_combo","_harvest_pool_from_archive","apply_boost_action","fingerprint","_negate_smart","_pick_best_equiv_score","_score_expr_base","_score_expr_base_joint_affine","_score_expr_base_joint_linear_terms","_collect_trig_paths","_trig_arg_has_const_scale","_collect_log_paths","_log_arg_has_const_scale","_collect_exp_paths","_exp_arg_has_const_scale","_collect_sqr_shift_paths","_sqr_shift_already_present","_collect_sqrt_shift_paths","_sqrt_shift_already_present","_wrap_param_slots","_refine_diag","_diag_inc","_diag_inc_context","_diag_add_time","_node_var_indices","_refine_tensor_signature","_refine_cfg_signature","_refine_attempt_cache_key","_refine_cache_get","_refine_cache_put","_prune_mapping_equiv_root_slot_paths","_decorate_refine_variants","_eval_node_hparam","_materialize_hparams","_build_init_logs","_raw_to_hparams","_stable_seed_from_text","_select_subset_indices","_slice_fit_subset","_slice_fit_subset_multi","_solve_linear_coeffs","_solve_linearized_fit","_joint_dataset_weights","_solve_linearized_fit_multi","_linearized_loss_value","_build_single_slot_variant","_slot_sensitivity_score","_rank_paths_by_sensitivity","_variant_has_gate_potential","_build_grid_seed_logs","_normalize_refine_optimizer","_score_refine_raw_log","_ranked_grid_refine_seeds","_init_logs_from_grid_rank","_flatten_add_terms","_select_linear_basis_nodes","_eval_node_hparam_safe","_build_phi_hparam","_materialize_linearized_candidate","_refine_hparams","_refine_budget_left","score_expr","Explorer","_init_crossover_policy_stats","_finalize_crossover_policy_stats","_remove_allowed_action","_finalize_action_distribution","enumerate_trees","enumerate_trees_dim","_has_const_zero","_dedup_new","_auto_brute_depth","_enumerate_incremental","_enumerate_dim_incremental","_build_monomial_ast","_monomial_presearch","_lorentz_peel_presearch","_planck_peel_presearch","_hyperbolic_peel_presearch","_gaussian_peel_presearch","_invtrig_peel_presearch","_archive_best_mse","_archive_best_structural_mse","_promote_structural_shadow_archive","_run_brute_phase","make_engine_refinement_hooks","make_engine_runtime_hooks","run_explorer_core","main"])


for _module_key, _module in zip(
    ("_actions", "_scoring", "_brute"), _IMPL_MODULES
):
    for _definition_name in _GROUP_DEFINITION_NAMES[_module_key]:
        _definition = getattr(_module, _definition_name)
        if globals().get(_definition_name) is _definition:
            _unwrap_seen = set()
            _unwrap_target = _definition
            while _inspect.isfunction(_unwrap_target) and id(_unwrap_target) not in _unwrap_seen:
                _unwrap_seen.add(id(_unwrap_target))
                _unwrap_target.__module__ = __name__
                _unwrap_target = getattr(_unwrap_target, "__wrapped__", None)

# Restore the single historical global namespace seen by every extracted body.
for _global_name in _SYNC_GLOBAL_NAMES:
    if _global_name not in globals():
        continue
    _global_value = globals()[_global_name]
    for _module in _IMPL_MODULES:
        setattr(_module, _global_name, _global_value)


class _CompatibilityModule(_types.ModuleType):
    """Keep historical explorer monkeypatches visible to extracted helpers."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in _SYNC_GLOBAL_NAMES:
            for module in _IMPL_MODULES:
                setattr(module, name, value)


_sys.modules[__name__].__class__ = _CompatibilityModule
