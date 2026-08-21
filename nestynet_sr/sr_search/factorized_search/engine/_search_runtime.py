# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""The closure-heavy factorized-search driver, preserved as one state machine."""

from __future__ import annotations

# ruff: noqa: F821

import math
import random
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import torch
from nestynet_sr.sr_core.carrier_units import mark_inner_coordinate_metadata
from ..config import coerce_inverse_steering_config
from ..controller import MacroController, build_macro_controller_state
from ..expr_ast import UNARY_OPS, build_pool, collect_paths, compute_reachable, dim_round, dims_eq, eval_node, get_at, is_valid_node, node_depth, node_dims, node_size, node_str, rand_node, rand_node_dim, replace_at, sample_box, set_dim_precision, simplify
from ..expr_mapping import eval_mapping, fit_best, mapping_is_structural
from ..inverse_core import _normalize_inverse_local_score_mode, _normalize_inverse_target_mode, eval_mapping_total
from ..inverse_search import estimate_inverse_steering_potential, _mapping_cache_signature, _pool_cache_signature
from nestynet_sr.sr_search.model_selection import mapping_cost
from ..proposal_families.runner import run_closure_search_pass_impl as _run_closure_search_pass_impl
from ..opportunity_critic import load_opportunity_bundle, predict_opportunity_slate
from ..scheduler import build_plan_candidates, choose_plan
from ..scheduler_critic import load_scheduler_bundle
from nestynet_sr.sr_expr_ir.config import ExpressionIRConfig, coerce_expr_ir_config
from nestynet_sr.sr_expr_ir.reporting import expression_ir_report
from nestynet_sr.sr_expr_ir.stats import ExpressionIRStats
from ..proposal_families.gs import build_gs_fss_context, coerce_gs_fss_context, extend_pool_with_gs_atoms, gs_fss_report
from ..shared_opportunity import normalize_realized_witness_energy_fields, normalize_witness_energy_fields
from ..policy.guidance import _annotate_inverse_experiment_lineage, _choose_repair_execution_preview, _credible_route_compare_decision, _credible_route_preview_repair_opportunity_rows, _controller_build_slate_id, _derived_controller_build_rng
from ..policy.build_slate import collect_controller_build_slate as _collect_controller_build_slate_impl
from ..policy.features import RepairControllerFeatureRecord, build_controller_state_record
from ..policy.parent_selection import choose_parent, choose_parent_repair_aware
from ..repair_critic import load_repair_critic_bundle, predict_repair_build_route, predict_repair_controller_heads
from ..repair_policy import _actor_critic_reward_terms, _analytic_repair_controller_score, _hybrid_repair_controller_scores, _normalize_repair_controller_critic_mode, _repair_controller_component_gate, _repair_controller_path_policy, _repair_controller_stagnation_state, _repair_controller_threshold, _repair_controller_weights, _repair_option_candidate_paths, _repair_parent_preview_retry_gate, _repair_parent_record_attempt, _repair_parent_retry_gate, _repair_preview_signature
from ..shared_candidate import shared_candidate_row_dict
from .actions import ACTION_ID_BY_NAME, ACTION_NAME, A_ADD_RAND, A_BOOST, A_CROSSOVER, A_HOLESEARCH, A_INVSTEER, A_MUL_RAND, A_PRUNE, A_REPAIR, A_REPLACE, A_RESIDUAL, A_WRAP_UNARY, apply_action_impl as _apply_action_impl, apply_crossover_action_impl as _apply_crossover_action_impl, apply_residual_action_impl as _apply_residual_action_impl
from .archive import ResidualBasinArchive
from .proposal_execution import ProposalScoringState, merge_route_status_counts as _merge_route_status_counts_impl, record_route_status as _record_route_status_impl, run_closure_search_pass as _run_closure_search_pass_route, score_native_candidate_basis_state as _score_native_candidate_basis_state_impl, score_external_candidate_expr as _score_external_candidate_expr_impl
from .scoring import _eval_node_hparam_safe as _engine_eval_node_hparam_safe, _harvest_pool_from_archive as _engine_harvest_pool_from_archive, score_expr as _engine_score_expr
from .signals import InverseSteeringPotential

from ._search_state import (
    Explorer,
    _ExecutionContext,
    _ParentSnapshotStore,
    _RouteScheduler,
    _periodogram_frequency_hints,
)
from ._search_support import (
    _CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS,
    _LEGACY_SEARCH_HELPERS,
    _OPTIONAL_RUNTIME_HOOKS,
    _action_candidate_paths,
    _archive_best_stall_mse,
    _coerce_guided_path,
    _controller_selected_action_path,
    _degenerate_abort_should_stop,
    _finalize_action_distribution,
    _finalize_crossover_policy_stats,
    _init_crossover_policy_stats,
    _log,
    _macro_action_fields,
    _macro_decision_log_fields,
    _merge_inverse_proposal_log_fields,
    _merge_repair_option_log_fields,
    _normalize_controller_build_slate_actions,
    _normalize_refine_mode,
    _plateau_stop_should_stop,
    _relative_best_improvement,
    _remove_allowed_action,
    _select_action_path,
    _select_refinement_slate,
    _slate_float,
    _tracked_macro_actions,
)

def _collect_controller_build_slate(
    *,
    parent_key: Any,
    parent_rec: Any,
    n_evaluated: int,
    seed_search: int | None,
    active_actions: list[int] | tuple[int, ...],
    action_names: list[Any] | tuple[Any, ...] | None,
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
    boost_pool_nodes: list[Any] | tuple[Any, ...],
    boost_pool_phi_fit: torch.Tensor,
    boost_pool_norms_fit: torch.Tensor,
    boost_pool_phi: torch.Tensor,
    boost_pool_norms: torch.Tensor,
    boost_pool_dims: list[Any] | tuple[Any, ...] | None,
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
    var_dims: list[Any] | tuple[Any, ...] | None,
    y_dims: Any,
    reach: Any,
    score_expr_fn,
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
        score_expr_fn=score_expr_fn,
        shared_candidate_row_dict_fn=shared_candidate_row_dict,
        mapping_cost_fn=mapping_cost,
        action_name_map=ACTION_NAME,
        path_select_action_ids=(A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_PRUNE),
        residual_action_id=A_RESIDUAL,
        boost_action_id=A_BOOST,
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
def apply_residual_action(
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
    var_dims=None,
    topk=3,
    complexity_penalty=0.0,
):
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
        eval_mapping_total_fn=eval_mapping_total,
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


def _bind_runtime_hooks(hooks: Mapping[str, Any] | None) -> None:
    """Bind explicit compatibility hooks for explorer-owned code not extracted yet."""
    if not isinstance(hooks, Mapping):
        raise RuntimeError(
            "run_explorer_core requires explicit _runtime_hooks for unextracted "
            "legacy routes; call via factorized_search.explorer.run_explorer_core "
            "or pass hooks from explorer.make_engine_runtime_hooks()"
        )
    g = globals()
    missing = []
    for name in _LEGACY_SEARCH_HELPERS:
        if name not in hooks:
            missing.append(name)
            continue
        g[name] = hooks[name]
    if missing:
        raise RuntimeError(f"missing runtime hook(s): {', '.join(missing)}")
    for name in _OPTIONAL_RUNTIME_HOOKS:
        if name in hooks:
            g[name] = hooks[name]


def run_explorer_core(
    target_fn, nvars, *,
    n_iter=20000, max_depth=6, poly_degree=4,
    wall_time_limit_s=None,
    lo=1.0, hi=5.0, seed=0, seed_search=None,
    var_dims=None, y_dims=None,
    dtype=torch.float32,
    no_residual=False, no_crossover=False, residual_topk=5,
    periodic_seed_enable=True,
    periodic_seed_max_hints=2,
    periodic_seed_min_prominence=8.0,
    carrier_seed_exprs=(),
    inverse_steering_enable=False,
    inverse_max_paths=12,
    inverse_topk_terms=6,
    inverse_shortlist_mult=4,
    inverse_min_valid_frac=0.25,
    inverse_min_confidence=0.10,
    inverse_safe_eps=None,
    inverse_confidence_mode="conditioning",
    inverse_confidence_target_gain=4.0,
    inverse_confidence_floor=0.05,
    inverse_branch_beam_width=1,
    inverse_micro_search_enable=False,
    inverse_micro_search_max_depth=3,
    inverse_micro_search_beam_width=24,
    inverse_micro_search_topk=16,
    inverse_micro_search_seed_terms=8,
    inverse_local_score_mode="affine",
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
    repair_pass_enable=False,
    repair_pass_elite_k=8,
    repair_pass_paths_per_elite=2,
    repair_pass_rounds=2,
    closure_search_enable=False,
    closure_search_families=("periodic", "exp", "log", "rational", "power", "quadratic"),
    closure_search_max_proposals=16,
    closure_search_anchors_per_family=4,
    closure_search_preview_topk=4,
    closure_search_exact_topk=2,
    closure_search_beam_width=4,
    closure_search_seed_exact_topk=6,
    closure_search_seed_beam_width=4,
    closure_search_seed_scaffold_reserve=8,
    closure_search_seed_family_cap=2,
    closure_search_seed_exact_bound_bonus=0.25,
    closure_search_pair_normal_enable=False,
    closure_search_pair_normal_topk=3,
    closure_search_pair_normal_max_pairs=1,
    closure_search_pair_rescue_enable=True,
    closure_search_pair_rescue_topk=4,
    closure_search_pair_rescue_max_pairs=6,
    closure_search_emergent_basis_enable=False,
    closure_search_emergent_basis_max_source_rows=32,
    closure_search_emergent_basis_score_topk=8,
    closure_search_emergent_basis_max_per_round=1,
    closure_search_emergent_basis_max_total=4,
    closure_search_emergent_basis_min_probe_gain_rel=5.0e-3,
    closure_search_emergent_aux_atoms_enable=False,
    closure_search_emergent_aux_atoms_max_source_rows=48,
    closure_search_emergent_aux_atoms_max_new_per_round=5,
    closure_search_emergent_aux_atoms_max_total=8,
    closure_search_emergent_aux_atoms_max_target=4,
    closure_search_emergent_aux_atoms_max_dimensionless=3,
    closure_search_emergent_aux_atoms_max_rational_derived=2,
    closure_search_emergent_aux_atoms_max_seed_blocks=8,
    closure_search_debug_topk=0,
    closure_search_min_valid_frac=0.05,
    closure_search_min_confidence=0.02,
    closure_search_periodic_min_valid_scale=1.0,
    closure_search_periodic_min_confidence_scale=1.0,
    closure_search_transport_min_lin_rel=0.0,
    closure_search_anchor_head_compare_enable=False,
    hole_search_enable=False,
    hole_search_quota=0.10,
    hole_search_exact_budget=2,
    hole_search_cooldown_iters=32,
    hole_search_mine_cooldown_iters=50,
    hole_search_max_frontier=128,
    hole_search_first_class_scheduler_enable=True,
    hole_search_route_scheduler_enable=True,
    hole_search_route_ucb_c=0.25,
    hole_search_route_eps=0.05,
    hole_search_route_acquisition_weight=0.25,
    hole_search_route_reward_mode="penalized",
    hole_search_route_time_penalty=0.01,
    hole_search_route_time_floor=1.0,
    hole_search_abstraction_enable=True,
    hole_search_abstraction_on_improve=True,
    hole_search_abstraction_on_stall=True,
    hole_search_abstraction_cooldown_iters=25,
    hole_search_abstraction_max_parents=2,
    hole_search_abstraction_max_paths_per_parent=3,
    hole_search_abstraction_improve_min_delta_log_mse=0.15,
    hole_search_abstraction_stage_enable=True,
    hole_search_abstraction_stage_max_entries=64,
    hole_search_abstraction_promote_topk=2,
    hole_search_abstraction_promote_frontier_floor=3,
    hole_search_enum_max_depth=4,
    hole_search_enum_max_trees=3000,
    hole_search_preview_topk=8,
    hole_search_solver_market_enable=False,
    hole_search_solver_market_preview_topk=4,
    hole_search_solver_market_exact_topk=2,
    hole_search_solver_market_proposal_objects_enable=False,
    inverse_spec_recursive_sr_enable=False,
    inverse_spec_recursive_sr_preview_topk=4,
    inverse_spec_recursive_sr_exact_budget=2,
    inverse_spec_constant_lift_route_enable=False,
    inverse_spec_constant_lift_route_topk=2,
    inverse_spec_coordinate_lift_enable=False,
    inverse_spec_coordinate_lift_topk=4,
    inverse_spec_coordinate_lift_mode="both",
    inverse_spec_tangent_edit_enable=False,
    inverse_spec_tangent_edit_topk=8,
    inverse_spec_soft_edit_enable=False,
    inverse_spec_soft_edit_steps=64,
    inverse_spec_soft_edit_l1=1.0e-3,
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
    hole_search_tournament_enable=True,
    hole_search_tournament_n=8,
    hole_search_tournament_elite_k=2,
    hole_search_tournament_preview_trees=64,
    inverse_spec_recursive_enable=True,
    inverse_spec_recursive_max_depth=2,
    inverse_spec_recursive_trigger_rel_mse=0.25,
    inverse_spec_recursive_seed_cap=6,
    inverse_spec_recursive_branch_topk=4,
    inverse_spec_recursive_child_topk=2,
    inverse_spec_max_subtree_depth=None,
    inverse_spec_fit_cap=96,
    inverse_spec_probe_cap=192,
    inverse_spec_exact_budget=4,
    inverse_target_mode="robust",
    inverse_full_mapping_penalty=0.75,
    inverse_exact_simple_target_bonus=0.10,
    inverse_additive_descend_penalty=0.15,
    inverse_nonadditive_leaf_penalty=0.20,
    inverse_exact_path_eta=0.98,
    inverse_exact_transport_min_lin_rel=0.0,
    inverse_gate_enable=True,
    inverse_gate_warmup=0,
    inverse_gate_best_factor=20.0,
    inverse_gate_min_residual_basins=0,
    inverse_gate_min_depth=4,
    inverse_gate_min_size=6,
    inverse_gate_max_paths=6,
    inverse_gate_min_structural_score=0.75,
    inverse_gate_min_weighted_rel_gain=0.05,
    inverse_gate_structural_bias=0.20,
    inverse_periodic_min_valid_scale=1.25,
    inverse_periodic_min_confidence_scale=1.35,
    inverse_periodic_path_penalty=0.65,
    inverse_nonperiodic_muldiv_bonus=0.10,
    inverse_nonperiodic_explogsqrt_bonus=0.05,
    inverse_branch_ambiguity_penalty=0.50,
    inverse_transport_min_lin_rel=0.02,
    inverse_transport_min_effective_n=8.0,
    inverse_experiment_log_enable=False,
    controller_build_slate_enable=False,
    controller_build_slate_actions=_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS,
    controller_build_slate_max_actions=3,
    repair_controller_credible_route_enable=False,
    repair_opportunity_controller_enable=False,
    repair_opportunity_controller_path="",
    inverse_spec_regime_metadata: Mapping[str, Any] | None = None,
    repair_controller_enable=False,
    repair_controller_min_score=0.15,
    repair_controller_steps=3,
    repair_controller_ancestor_hops=1,
    repair_controller_min_step_rel_improve=1.0e-3,
    repair_controller_max_setup_steps=0,
    repair_controller_setup_step_value_min=0.10,
    repair_controller_setup_step_regret_max=0.50,
    repair_controller_setup_step_max_worsen=0.05,
    repair_controller_adaptive=True,
    repair_controller_adapt_quantile=0.75,
    repair_controller_adapt_window=128,
    repair_controller_adapt_min_samples=16,
    repair_controller_min_concentration=0.30,
    repair_controller_potential_weight=1.00,
    repair_controller_concentration_weight=0.35,
    repair_controller_contrast_weight=0.20,
    repair_controller_cost_weight=0.10,
    repair_controller_stagnation_weight=0.15,
    repair_controller_frontier_topk=24,
    repair_controller_stagnation_visits=8,
    repair_controller_focus_prob=0.50,
    repair_controller_parent_max_repeats=2,
    repair_controller_parent_min_eval_gap=32,
    repair_controller_parent_reset_rel_improve=0.05,
    repair_controller_parent_preview_max_repeats=1,
    repair_controller_policy_priority_weight=0.15,
    repair_controller_policy_priority_cap=0.25,
    repair_controller_critic_enable=False,
    repair_controller_critic_path="",
    repair_controller_critic_blend=1.0,
    repair_controller_critic_mode="priority",
    repair_controller_route_compare_enable=False,
    repair_controller_route_compare_path="",
    repair_controller_route_compare_repair_tuple_path="",
    repair_controller_route_compare_build_tuple_path="",
    repair_controller_route_compare_max_repair_prob=0.35,
    repair_controller_route_compare_min_build_margin=0.05,
    macro_controller_enable=False,
    macro_controller_repair_bonus=0.50,
    macro_controller_repair_margin_scale=0.75,
    macro_controller_build_bias=0.05,
    macro_controller_inverse_bonus=0.10,
    macro_controller_learned_policy_weight=0.75,
    macro_controller_learned_route_weight=0.60,
    macro_controller_learned_q_weight=0.75,
    macro_controller_learned_value_scale=0.75,
    scheduler_enable=False,
    scheduler_advisory_only=True,
    scheduler_bundle_path="",
    scheduler_budget_ladder=(1, 2, 4, 8),
    scheduler_build_preview_only=True,
    scheduler_fallback_min_confidence=0.0,
    scheduler_acquisition_threshold=0.25,
    scheduler_uncertainty_bonus=0.05,
    scheduler_acquisition_weights=None,
    scheduler_witness_energy_enable=False,
    boost_enable=False,
    boost_max_terms=6,
    boost_topk_try=15,
    boost_min_rel_improve=1.0e-3,
    boost_selection_split="fit",
    boost_ridge=None,
    boost_include_parent=True,
    boost_from_scratch_prob=0.25,
    boost_prune_rel=1.0e-10,
    boost_safe_eval=True,
    boost_harvest_enable=False,
    boost_harvest_every=500,
    boost_harvest_topk_residual_basins=50,
    boost_harvest_elites_per_residual_basin=2,
    boost_pool_extra_max=256,
    boost_subtree_depth_max=3,
    boost_subtree_size_max=12,
    boost_gate_enable=True,
    boost_gate_warmup=200,
    boost_gate_best_factor=30.0,
    boost_gate_gain_frac=1.0e-2,
    boost_gate_peak_ratio=5.0,
    boost_gate_min_valid=8,
    boost_gate_min_residual_basins=10,
    boost_gate_adaptive=True,
    boost_gate_adapt_quantile=0.75,
    boost_gate_adapt_window=256,
    boost_gate_adapt_min_samples=32,
    boost_gate_adapt_mix=1.0,
    boost_gate_gain_frac_floor=1.0e-4,
    boost_gate_gain_frac_cap=0.25,
    n_fit=512, n_probe=2048,
    x_fit_data=None, y_fit_data=None,
    x_probe_data=None, y_probe_data=None,
    emb_dim=16, fp_mode="bits", q_scale=2.0, q_clip=6,
    p_restart=0.20, exploit_frac=0.35, exploit_topk=50,
    eps_action=0.10, ucb_action=1.0,
    novelty_bonus=0.0, complexity_penalty=0.0,
    actor_critic_best_bonus=0.15,
    actor_critic_time_penalty=0.0,
    actor_critic_reward_eps=1.0e-30,
    actor_critic_descendant_horizon=8,
    crossover_village_topk=12,  # legacy compatibility (unused)
    crossover_foreign_topk=12,  # legacy compatibility (unused)
    crossover_expr_weight=0.15,  # legacy compatibility (unused)
    crossover_mode="legacy",  # legacy compatibility (legacy-only behavior)
    print_every=0, label="",
    brute_depth=None,
    early_stop_mse=1e-10,
    brute_max_expressions=50_000,
    stall_window=500, stall_patience=3, stall_delta=1e-4,
    plateau_stop_enable=False,
    plateau_stop_max_soft_restarts=0,
    plateau_stop_min_evals=0,
    degenerate_abort_enable=True,
    degenerate_abort_min_evals=1000,
    degenerate_abort_max_accepted=8,
    refine_enable=False,
    refine_profile=None,
    refine_mode="slate",
    refine_during_brute=True,
    refine_during_mutation=True,
    refine_during_controller_slate=False,
    refine_during_slate=False,
    refine_slate_after_brute=True,
    refine_slate_period=0,
    refine_final_polish=True,
    refine_slate_k=16,
    refine_slate_diverse_k=8,
    refine_slate_budget=32,
    refine_optimizer="lbfgs",
    refine_lbfgs_escalate_improve_factor=2.0,
    refine_lbfgs_steps=8,
    refine_fit_subset=256,
    refine_fit_subset_mode="hash_random",
    # Optional multi-dataset continuous skeleton refinement:
    # provide a list of (x_fit, y_fit) (or (id, x_fit, y_fit)) so that continuous skeleton refinement
    # can optimize shared nonlinear hparams while solving linear coefficients
    # per dataset.
    refine_joint_fit_data=None,
    # Optional joint probe data (same format as refine_joint_fit_data).
    refine_joint_probe_data=None,
    # If enabled, scoring (not just refinement) will fit per-dataset affine output maps
    # (degree-1 poly) and aggregate the probe loss across datasets.
    refine_joint_score_enable=True,
    refine_joint_terms_enable=False,
    refine_joint_weight_mode="points",  # points | datasets
    refine_joint_enable=True,
    refine_num_restarts=2,
    refine_max_variants=4,
    refine_max_params=2,
    refine_linear_combo_enable=True,
    refine_linear_terms_max=6,
    refine_linear_prune_rel=1.0e-10,
    refine_linear_ridge=1.0e-8,
    refine_slot_sensitivity_enable=True,
    refine_slot_sensitivity_subset=64,
    refine_slot_sensitivity_delta=0.1,
    refine_slot_sensitivity_max_paths=24,
    refine_prune_mapping_equiv_root_slots=True,
    refine_attempt_cache_enable=True,
    refine_attempt_cache_max_entries=4096,
    refine_gate_best_factor=10.0,
    refine_gate_potential_enable=True,
    refine_gate_potential_subset=64,
    refine_gate_potential_improve_factor=5.0,
    refine_gate_log_min=math.log(0.5),
    refine_gate_log_max=math.log(4.0),
    refine_gate_grid_size=4,
    refine_gate_max_evals=64,
    refine_max_trials=1500,
    refine_trials_per_brute_depth=64,
    refine_trials_per_mutation_window=64,
    refine_mutation_window=500,
    refine_safe_eps=1.0e-6,
    refine_safe_penalty_weight=1.0e-2,
    refine_safe_exp_clip=30.0,
    refine_theta_l2=1.0e-4,
    refine_init_log_min=-1.5,
    refine_init_log_max=1.5,
    refine_grid_enable=True,
    refine_grid_size=33,
    refine_grid_size_2d=11,
    refine_grid_passes=2,
    refine_grid_topk=2,
    refine_grid_max_evals=256,
    refine_stall_gate_relax_factor=3.0,
    refine_stall_gate_relax_max=100.0,

    # Scoring augmentation: multi-term linear head on residual
    score_head_enable=True,
    score_head_vars_enable=True,
    score_head_omp_enable=False,
    score_head_omp_max_terms=2,
    score_head_omp_topk_try=15,
    score_head_ridge=None,
    score_head_min_rel_improve=0.0,
    score_head_untyped_enable=False,
    score_mapping_family_mode="full",
    brute_score_mapping_family_mode="gated",
    score_pade_structural_enable=False,
    score_pade_structural_max_degree=2,
    score_pade_structural_max_total_degree=3,
    score_pade_structural_max_depth=8,
    score_pade_structural_max_size=64,
    score_pade_structural_coeff_tol=1.0e-10,
    score_pade_structural_mse_rel_tol=1.0e-6,
    score_mapping_expensive_gate_best_factor=5.0,
    score_mapping_expensive_rel_y=0.10,
    score_prescreen_enable=True,
    score_prescreen_family_mode="cheap",
    score_prescreen_residual_family_mode="gated",
    score_prescreen_residual_allow_hint=False,
    score_prescreen_residual_use_global_best=False,
    score_prescreen_parent_best_factor=1.5,
    score_prescreen_global_best_factor=3.0,
    score_prescreen_residual_parent_best_factor=1.1,
    score_prescreen_residual_global_best_factor=1.5,
    score_finite_mask_enable=False,
    score_finite_mask_min_fit_frac=0.98,
    score_finite_mask_min_probe_frac=0.98,
    score_finite_mask_min_dataset_frac=0.95,
    score_finite_mask_min_points=8,
    score_domain_projection_enable=False,
    score_domain_projection_abs_tol=1.0e-8,
    score_domain_projection_rel_tol=1.0e-8,
    score_domain_projection_max_frac=1.0,
    score_domain_projection_positive_floor=1.0e-12,

    verbose=True,
    stop_event=None,
    **_engine_hooks,
):
    """Run the residual-basin archive explorer and return the archive.

    Parameters
    ----------
    target_fn : callable
        Maps (N, nvars) tensor -> (N, 1) tensor.
    nvars : int
        Number of input variables.
    var_dims, y_dims : optional sequences of floats
        Dimensional exponents for variables and target.
    print_every : int
        Print progress every N iterations (0 = silent).

    Returns
    -------
    ResidualBasinArchive
    """
    explorer_cls = _engine_hooks.pop("_explorer_cls", Explorer)
    score_expr_fn = _engine_hooks.pop("_score_expr_fn", _engine_score_expr)
    harvest_pool_fn = _engine_hooks.pop("_harvest_pool_from_archive_fn", _engine_harvest_pool_from_archive)
    eval_node_hparam_safe_fn = _engine_hooks.pop("_eval_node_hparam_safe_fn", _engine_eval_node_hparam_safe)
    runtime_hooks = _engine_hooks.pop("_runtime_hooks", None)
    gs_fss_context = _engine_hooks.pop("gs_fss_context", None)
    _bind_runtime_hooks(runtime_hooks)
    expr_ir_hook_values: dict[str, Any] = {}
    for _field_name in ExpressionIRConfig.__dataclass_fields__:
        if _field_name == "expr_ir":
            _prefixed = "expr_ir"
        elif _field_name == "canonicalize":
            _prefixed = "expr_canonicalize"
        elif _field_name == "domain_mode":
            _prefixed = "expr_domain_mode"
        else:
            _prefixed = f"expr_{_field_name}"
        for _key in (_field_name, _prefixed):
            if _key in _engine_hooks:
                expr_ir_hook_values[_key] = _engine_hooks.pop(_key)
    expr_ir_runtime_cfg = coerce_expr_ir_config(expr_ir_hook_values)
    expr_ir_stats = ExpressionIRStats()
    if _engine_hooks:
        keys = ", ".join(sorted(str(k) for k in _engine_hooks))
        raise TypeError(f"Unexpected internal engine hooks: {keys}")

    refine_mode_norm = _normalize_refine_mode(refine_mode)
    refine_enable_requested = bool(refine_enable)
    refine_active = bool(refine_enable_requested and refine_mode_norm != "off")
    refine_inline_enable = bool(refine_active and refine_mode_norm == "inline")
    brute_refine_enable = bool(refine_inline_enable and bool(refine_during_brute))
    mutation_refine_enable = bool(refine_inline_enable and bool(refine_during_mutation))
    controller_slate_refine_enable = bool(
        refine_inline_enable and bool(refine_during_controller_slate)
    )
    try:
        refine_slate_period_i = max(0, int(refine_slate_period))
    except Exception:
        refine_slate_period_i = 0
    try:
        refine_slate_k_i = max(0, int(refine_slate_k))
    except Exception:
        refine_slate_k_i = 0
    try:
        refine_slate_diverse_k_i = max(0, int(refine_slate_diverse_k))
    except Exception:
        refine_slate_diverse_k_i = 0
    try:
        refine_slate_budget_i = max(0, int(refine_slate_budget))
    except Exception:
        refine_slate_budget_i = 0
    scheduled_slate_refine_enable = bool(
        refine_active
        and (
            refine_mode_norm in {"slate", "final_polish"}
            or bool(refine_during_slate)
        )
    )
    after_brute_slate_refine_enable = bool(
        scheduled_slate_refine_enable
        and refine_mode_norm != "final_polish"
        and bool(refine_slate_after_brute)
    )
    periodic_slate_refine_enable = bool(
        scheduled_slate_refine_enable
        and refine_mode_norm != "final_polish"
        and refine_slate_period_i > 0
    )
    final_polish_refine_enable = bool(
        scheduled_slate_refine_enable
        and bool(refine_final_polish)
    )
    refine_slate_stats: dict[str, Any] = {
        "enabled": bool(scheduled_slate_refine_enable),
        "after_brute_enable": bool(after_brute_slate_refine_enable),
        "periodic_enable": bool(periodic_slate_refine_enable),
        "final_polish_enable": bool(final_polish_refine_enable),
        "period": int(refine_slate_period_i),
        "slate_k": int(refine_slate_k_i),
        "diverse_k": int(refine_slate_diverse_k_i),
        "budget": int(refine_slate_budget_i),
        "passes": [],
        "total_passes": 0,
        "total_selected": 0,
        "total_scored": 0,
        "total_accepted": 0,
        "total_trials_used": 0,
    }
    refine_diagnostics: dict[str, Any] = {}
    refine_attempt_cache: dict[Any, Any] = {}

    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = random.Random(seed_search if seed_search is not None else seed)
    try:
        wall_time_limit_s = None if wall_time_limit_s is None else float(wall_time_limit_s)
        if wall_time_limit_s is not None and (not math.isfinite(wall_time_limit_s) or wall_time_limit_s <= 0.0):
            wall_time_limit_s = None
    except Exception:
        wall_time_limit_s = None
    try:
        plateau_stop_max_soft_restarts_i = max(0, int(plateau_stop_max_soft_restarts))
    except Exception:
        plateau_stop_max_soft_restarts_i = 0
    try:
        plateau_stop_min_evals_i = max(0, int(plateau_stop_min_evals))
    except Exception:
        plateau_stop_min_evals_i = 0
    plateau_stop_active = bool(plateau_stop_enable) and plateau_stop_max_soft_restarts_i > 0
    plateau_stop_requested = False
    plateau_stop_eval = 0
    plateau_stop_best_mse = float("inf")
    plateau_stop_consecutive_soft_restarts = 0
    n_soft_restarts = 0
    if plateau_stop_active:
        refine_diagnostics["plateau_stop_enable"] = True
        refine_diagnostics["plateau_stop_max_soft_restarts"] = int(plateau_stop_max_soft_restarts_i)
        refine_diagnostics["plateau_stop_min_evals"] = int(plateau_stop_min_evals_i)
        refine_diagnostics["plateau_stop_stall_metric"] = "raw_mse"
    search_started = time.perf_counter()
    setup_started = search_started
    phase_timing: dict[str, Any] = {
        "setup_wall_s": 0.0,
        "pool_eval_wall_s": 0.0,
        "brute_wall_s": 0.0,
        "brute_scored": 0,
        "mutation_wall_s": 0.0,
    }
    wall_time_deadline = None if wall_time_limit_s is None else (search_started + float(wall_time_limit_s))
    repair_controller_enable = bool(repair_controller_enable) and bool(inverse_steering_enable)
    macro_controller_enable = bool(macro_controller_enable)
    scheduler_enable = bool(scheduler_enable)
    scheduler_advisory_only = bool(scheduler_advisory_only)
    scheduler_witness_energy_enable = bool(scheduler_witness_energy_enable)
    try:
        scheduler_fallback_min_confidence = max(0.0, float(scheduler_fallback_min_confidence))
    except Exception:
        scheduler_fallback_min_confidence = 0.0
    try:
        scheduler_acquisition_threshold = max(0.0, float(scheduler_acquisition_threshold))
    except Exception:
        scheduler_acquisition_threshold = 0.25
    try:
        scheduler_uncertainty_bonus = max(0.0, float(scheduler_uncertainty_bonus))
    except Exception:
        scheduler_uncertainty_bonus = 0.05
    if isinstance(scheduler_acquisition_weights, Mapping):
        scheduler_acquisition_weights = {
            str(key): float(value)
            for key, value in dict(scheduler_acquisition_weights).items()
        }
    else:
        scheduler_acquisition_weights = None
    repair_controller_critic_mode = _normalize_repair_controller_critic_mode(
        repair_controller_critic_mode,
        default="priority",
    )

    base_actions = [A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_RESIDUAL, A_PRUNE]
    if bool(inverse_steering_enable):
        base_actions.insert(base_actions.index(A_RESIDUAL) + 1, A_INVSTEER)
    if bool(boost_enable):
        insert_after = A_INVSTEER if bool(inverse_steering_enable) else A_RESIDUAL
        base_actions.insert(base_actions.index(insert_after) + 1, A_BOOST)
    active_actions = [act for act in base_actions if not (act == A_RESIDUAL and no_residual)]
    if not no_crossover:
        active_actions.append(A_CROSSOVER)
    if (
        bool(hole_search_enable)
        and bool(inverse_steering_enable)
        and bool(inverse_spec_enable)
        and ((not bool(hole_search_first_class_scheduler_enable)) or bool(scheduler_enable))
    ):
        active_actions.append(A_HOLESEARCH)
    prescreen_actions = {
        A_REPLACE,
        A_WRAP_UNARY,
        A_ADD_RAND,
        A_MUL_RAND,
        A_RESIDUAL,
        A_PRUNE,
        A_CROSSOVER,
    }
    full_score_actions = {
        A_INVSTEER,
        A_REPAIR,
        A_HOLESEARCH,
        A_BOOST,
    }
    tracked_actions = _tracked_macro_actions(
        active_actions,
        repair_controller_enable=repair_controller_enable,
    )

    def _wall_time_exceeded() -> bool:
        if wall_time_deadline is None:
            return False
        try:
            return bool(time.perf_counter() >= float(wall_time_deadline))
        except Exception:
            return False

    def _finalize_search_state(reason: str | None) -> None:
        elapsed = float(max(0.0, time.perf_counter() - search_started))
        arch.search_stop_reason = str(reason or "")
        arch.search_wall_time_limit_s = None if wall_time_limit_s is None else float(wall_time_limit_s)
        arch.search_wall_time_elapsed_s = float(elapsed)
        arch.search_wall_time_limit_hit = bool(str(reason or "") == "wall_time_limit")
        arch.refine_runtime_config = {
            "refine_enable_requested": bool(refine_enable_requested),
            "refine_profile": str(refine_profile or ""),
            "refine_mode": str(refine_mode_norm),
            "refine_active": bool(refine_active),
            "refine_inline_enable": bool(refine_inline_enable),
            "refine_during_brute": bool(refine_during_brute),
            "refine_during_mutation": bool(refine_during_mutation),
            "refine_during_controller_slate": bool(refine_during_controller_slate),
            "refine_during_slate": bool(refine_during_slate),
            "brute_refine_enable": bool(brute_refine_enable),
            "mutation_refine_enable": bool(mutation_refine_enable),
            "controller_slate_refine_enable": bool(controller_slate_refine_enable),
            "scheduled_slate_refine_enable": bool(scheduled_slate_refine_enable),
            "after_brute_slate_refine_enable": bool(after_brute_slate_refine_enable),
            "periodic_slate_refine_enable": bool(periodic_slate_refine_enable),
            "final_polish_refine_enable": bool(final_polish_refine_enable),
            "refine_slate_after_brute": bool(refine_slate_after_brute),
            "refine_slate_period": int(refine_slate_period_i),
            "refine_final_polish": bool(refine_final_polish),
            "refine_slate_k": int(refine_slate_k_i),
            "refine_slate_diverse_k": int(refine_slate_diverse_k_i),
            "refine_slate_budget": int(refine_slate_budget_i),
        }
        arch.refine_slate_stats = refine_slate_stats
        diag = dict(refine_diagnostics)
        diag["attempt_cache_size"] = int(len(refine_attempt_cache))
        diag["stall_checks"] = int(hole_search_stats.get("stall_checks", 0) or 0)
        diag["stall_triggered"] = int(hole_search_stats.get("stall_triggered", 0) or 0)
        diag["stall_metric"] = str(
            hole_search_stats.get(
                "stall_metric",
                "raw_mse" if plateau_stop_active else "effective_mse",
            )
        )
        last_best = hole_search_stats.get("stall_last_best_mse", None)
        diag["stall_last_best_mse"] = (
            float(last_best)
            if last_best is not None and math.isfinite(float(last_best))
            else None
        )
        last_rel = hole_search_stats.get("stall_last_rel_improve", None)
        diag["stall_last_rel_improve"] = (
            float(last_rel)
            if last_rel is not None and math.isfinite(float(last_rel))
            else None
        )
        diag["soft_restarts"] = int(n_soft_restarts)
        if plateau_stop_active:
            diag["plateau_stop_enable"] = True
            diag["plateau_stop_requested"] = bool(plateau_stop_requested)
            diag["plateau_stop_eval"] = int(plateau_stop_eval)
            diag["plateau_stop_best_mse"] = (
                float(plateau_stop_best_mse) if math.isfinite(float(plateau_stop_best_mse)) else None
            )
            diag["plateau_stop_soft_restarts"] = int(plateau_stop_consecutive_soft_restarts)
            diag["plateau_stop_max_soft_restarts"] = int(plateau_stop_max_soft_restarts_i)
            diag["plateau_stop_min_evals"] = int(plateau_stop_min_evals_i)
        phase_diag = {
            "search_stop_reason": str(reason or ""),
            "setup_wall_s": float(phase_timing.get("setup_wall_s", 0.0) or 0.0),
            "pool_eval_wall_s": float(phase_timing.get("pool_eval_wall_s", 0.0) or 0.0),
            "brute_wall_s": float(phase_timing.get("brute_wall_s", 0.0) or 0.0),
            "brute_scored": int(phase_timing.get("brute_scored", 0) or 0),
            "mutation_wall_s": float(phase_timing.get("mutation_wall_s", 0.0) or 0.0),
        }
        if plateau_stop_active:
            phase_diag["plateau_stop_requested"] = bool(plateau_stop_requested)
            phase_diag["plateau_stop_eval"] = int(plateau_stop_eval)
        diag.update(phase_diag)
        if isinstance(score_prescreen_stats, dict):
            diag["prescore_calls"] = int(score_prescreen_stats.get("prescore_calls", 0) or 0)
            diag["prescore_promoted"] = int(score_prescreen_stats.get("prescore_promoted", 0) or 0)
            diag["prescore_dropped"] = int(score_prescreen_stats.get("prescore_dropped", 0) or 0)
            diag["full_score_calls"] = int(score_prescreen_stats.get("full_score_calls", 0) or 0)
            by_action = score_prescreen_stats.get("full_score_calls_by_action", {})
            if isinstance(by_action, Mapping):
                diag["full_score_calls_by_action"] = {
                    str(k): int(v) for k, v in dict(by_action).items()
                }
        arch.search_phase_diagnostics = dict(phase_diag)
        arch.score_prescreen_stats = score_prescreen_stats
        arch.refine_diagnostics = diag

    from ..hole_search import HoleFrontier
    hole_frontier = HoleFrontier(
        cooldown_iters=int(hole_search_cooldown_iters),
        max_entries=int(hole_search_max_frontier),
    ) if bool(hole_search_enable) else None
    abstraction_stage = HoleFrontier(
        cooldown_iters=int(hole_search_cooldown_iters),
        max_entries=int(hole_search_abstraction_stage_max_entries),
        staleness_window=max(256, 8 * max(1, int(hole_search_cooldown_iters))),
    ) if (bool(hole_search_enable) and bool(hole_search_abstraction_enable) and bool(hole_search_abstraction_stage_enable)) else None
    hole_search_stats = {
        "prepare_calls": 0,
        "prepared_executable_checks": 0,
        "prepared_resolution_live_archive": 0,
        "prepared_resolution_snapshot": 0,
        "prepared_resolution_missing": 0,
        "prepare_prune_wall_seconds": 0.0,
        "prepare_mine_wall_seconds": 0.0,
        "prepare_select_wall_seconds": 0.0,
        "prepare_wall_seconds": 0.0,
        "prepared_with_any_frontier_entries": 0,
        "prepared_with_nonempty_frontier": 0,
        "prepared_executable_available": 0,
        "prepared_executable_available_inverse_slate": 0,
        "prepared_executable_available_archive_mine": 0,
        "prepared_executable_available_abstraction": 0,
        "prepared_executable_available_other": 0,
        "prepared_dead_pruned": 0,
        "prepared_dead_pruned_inverse_slate": 0,
        "prepared_dead_pruned_archive_mine": 0,
        "prepared_dead_pruned_abstraction": 0,
        "prepared_dead_pruned_other": 0,
        "selected": 0,
        "fired": 0,
        "first_class_scheduler_selected": 0,
        "first_class_scheduler_available": 0,
        "exact_outcome_updates": 0,
        "followup_spec_states_emitted": 0,
        "followup_spec_states_ingested": 0,
        "frontier_size": 0,
        "ingested": 0,
        "mined": 0,
        "selected_with_any_frontier_entries": 0,
        "selected_with_nonempty_frontier": 0,
        "selected_with_opportunity": 0,
        "selected_with_opportunity_inverse_slate": 0,
        "selected_with_opportunity_archive_mine": 0,
        "selected_with_opportunity_abstraction": 0,
        "selected_with_opportunity_other": 0,
        "selected_with_resolved_parent": 0,
        "selected_with_snapshot_parent": 0,
        "selected_with_snapshot_parent_inverse_slate": 0,
        "selected_with_snapshot_parent_archive_mine": 0,
        "selected_with_snapshot_parent_abstraction": 0,
        "selected_with_snapshot_parent_other": 0,
        "invalidated_parent": 0,
        "invalidated_parent_inverse_slate": 0,
        "invalidated_parent_archive_mine": 0,
        "invalidated_parent_abstraction": 0,
        "invalidated_parent_other": 0,
        "snapshot_parent_missing": 0,
        "snapshot_parent_missing_inverse_slate": 0,
        "snapshot_parent_missing_archive_mine": 0,
        "snapshot_parent_missing_abstraction": 0,
        "snapshot_parent_missing_other": 0,
        "run_hole_search_action_called": 0,
        "child_expr_none": 0,
        "last_mined_iter": None,
        "last_abstracted_iter": None,
        "abstract_events": 0,
        "abstract_events_on_improve": 0,
        "abstract_events_on_stall": 0,
        "abstracted": 0,
        "abstracted_on_improve": 0,
        "abstracted_on_stall": 0,
        "abstraction_stage_size": 0,
        "abstraction_staged": 0,
        "abstraction_promoted": 0,
        "abstraction_promoted_on_prepare": 0,
        "abstraction_stage_dead_pruned": 0,
        "abstraction_attempts": 0,
        "abstraction_attempts_on_improve": 0,
        "abstraction_attempts_on_stall": 0,
        "abstraction_blocked_disabled_or_missing_input": 0,
        "abstraction_blocked_small_expr": 0,
        "abstraction_blocked_cooldown": 0,
        "abstraction_added_zero": 0,
        "stall_checks": 0,
        "stall_triggered": 0,
        "stall_score_none_skips": 0,
        "stall_last_rel_improve": None,
        "stall_abstraction_attempts": 0,
        "stall_abstraction_parent_candidates": 0,
        "stall_abstraction_parent_selected": 0,
        "stall_abstraction_parent_selected_best": 0,
        "best_eff_mse": None,
    }
    score_prescreen_stats = {
        "prescore_calls": 0,
        "prescore_promoted": 0,
        "prescore_dropped": 0,
        "prescore_promoted_by_hint": 0,
        "prescore_promoted_by_parent_threshold": 0,
        "prescore_promoted_by_global_best_threshold": 0,
        "full_score_calls": 0,
        "full_score_calls_by_action": {},
    }
    route_scheduler_enabled = bool(hole_search_enable) and (
    bool(hole_search_route_scheduler_enable)
    or (
        bool(hole_search_first_class_scheduler_enable)
        and bool(inverse_steering_enable)
        and bool(inverse_spec_enable)
    )
)
    hole_search_route_reward_mode = str(hole_search_route_reward_mode or "penalized").strip().lower()
    if hole_search_route_reward_mode not in {"raw", "per_second", "penalized"}:
        hole_search_route_reward_mode = "penalized"
    try:
        hole_search_route_time_penalty = max(0.0, float(hole_search_route_time_penalty))
    except Exception:
        hole_search_route_time_penalty = 0.01
    try:
        hole_search_route_time_floor = max(1.0e-6, float(hole_search_route_time_floor))
    except Exception:
        hole_search_route_time_floor = 1.0
    route_scheduler = _RouteScheduler(
        ("expression_expand", "opportunity_expand"),
        ucb_c=float(hole_search_route_ucb_c),
        eps=float(hole_search_route_eps),
    ) if route_scheduler_enabled else None
    route_scheduler_stats = {
        "enabled": bool(route_scheduler_enabled),
        "mode": "first_class_agenda" if bool(hole_search_first_class_scheduler_enable) and bool(inverse_steering_enable) and bool(inverse_spec_enable) else "legacy_route",
        "first_class_enabled": bool(hole_search_first_class_scheduler_enable) and bool(inverse_steering_enable) and bool(inverse_spec_enable),
        "considered": 0,
        "opportunity_available": 0,
        "selected_expression_expand": 0,
        "selected_opportunity_expand": 0,
        "selection_forced": 0,
        "selection_epsilon": 0,
        "selection_ucb": 0,
        "model_scored": 0,
        "model_trained": 0,
        "reward_count": 0,
        "reward_sum": 0.0,
        "reward_sum_raw": 0.0,
        "reward_sum_adjusted": 0.0,
        "wall_seconds_sum": 0.0,
        "reward_mode": str(hole_search_route_reward_mode),
        "time_penalty": float(hole_search_route_time_penalty),
        "time_floor": float(hole_search_route_time_floor),
        "route_summary": {},
    }

    dm = var_dims is not None
    if dm:
        set_dim_precision(max_depth)
        var_dims = [dim_round(tuple(d)) for d in var_dims]
        y_dims = dim_round(tuple(y_dims)) if y_dims is not None else None
        reach = compute_reachable(var_dims, max_depth, target_dim=y_dims)
    else:
        reach = None

    gs_fss_context_runtime = coerce_gs_fss_context(gs_fss_context)
    if gs_fss_context_runtime is None and (
        bool(getattr(expr_ir_runtime_cfg, "gs_fss_aux_generator", False))
        or bool(getattr(expr_ir_runtime_cfg, "gs_fss_score", False))
    ):
        gs_fss_context_runtime = build_gs_fss_context(
            nvars=int(nvars),
            var_dims=var_dims,
            y_dims=y_dims,
            cfg=expr_ir_runtime_cfg,
            enabled=True,
            max_depth=int(max_depth),
        )

    if (x_fit_data is not None and y_fit_data is not None
            and x_probe_data is not None and y_probe_data is not None):
        # Use pre-built data directly
        x_fit = x_fit_data.to(dtype=dtype)
        y_fit = y_fit_data.to(dtype=dtype)
        if y_fit.dim() == 1:
            y_fit = y_fit.unsqueeze(-1)
        x_probe = x_probe_data.to(dtype=dtype)
        y_probe = y_probe_data.to(dtype=dtype)
        if y_probe.dim() == 1:
            y_probe = y_probe.unsqueeze(-1)
        n_probe = x_probe.shape[0]
    else:
        x_fit = sample_box(n_fit, nvars, lo, hi, dtype=dtype, g=g)
        x_probe = sample_box(n_probe, nvars, lo, hi, dtype=dtype, g=g)
        y_fit = target_fn(x_fit)
        y_probe = target_fn(x_probe)
    proj = torch.randn((n_probe, emb_dim), generator=g, dtype=dtype).to(x_probe.device)

    pool_eval_started = time.perf_counter()
    expr_signature_context = {"var_dims": var_dims, "y_dims": y_dims, "nvars": int(nvars)}
    pool_nodes = build_pool(
        nvars,
        ir_cfg=expr_ir_runtime_cfg,
        ir_stats=expr_ir_stats,
        signature_context=expr_signature_context,
    )
    pool_before_gs = int(len(pool_nodes))
    pool_nodes = extend_pool_with_gs_atoms(
        pool_nodes,
        gs_fss_context_runtime,
        ir_cfg=expr_ir_runtime_cfg,
        ir_stats=expr_ir_stats,
        signature_context=expr_signature_context,
        max_depth=int(max_depth),
    )
    pool_gs_added = max(0, int(len(pool_nodes)) - pool_before_gs)
    if pool_gs_added > 0:
        expr_ir_stats.gs_fss_aux_generator_count += int(pool_gs_added)
    pool_dims = [node_dims(n, var_dims) for n in pool_nodes] if dm else [None]*len(pool_nodes)
    pool_phi_list = []
    for n in pool_nodes:
        v = eval_node(n, x_probe).squeeze(-1)
        pool_phi_list.append(v if torch.isfinite(v).all() else torch.zeros_like(v))
    pool_phi = torch.stack(pool_phi_list, dim=1)
    pool_norms = (pool_phi * pool_phi).sum(dim=0)

    # Also evaluate the pool on the fit split for boosting / OMP-style actions.
    pool_phi_fit_list = []
    for n in pool_nodes:
        v = eval_node(n, x_fit).squeeze(-1)
        pool_phi_fit_list.append(v if torch.isfinite(v).all() else torch.zeros_like(v))
    pool_phi_fit = torch.stack(pool_phi_fit_list, dim=1)
    pool_norms_fit = (pool_phi_fit * pool_phi_fit).sum(dim=0)
    phase_timing["pool_eval_wall_s"] = float(time.perf_counter() - pool_eval_started)

    # Optional: safe-eval variants for stiff / singular terms (used by A_BOOST).
    pool_phi_safe = pool_phi
    pool_norms_safe = pool_norms
    pool_phi_fit_safe = pool_phi_fit
    pool_norms_fit_safe = pool_norms_fit
    safe_cfg = {"safe_eps": float(refine_safe_eps), "safe_exp_clip": float(refine_safe_exp_clip)}
    if inverse_safe_eps is None:
        inverse_safe_eps = float(refine_safe_eps)
    inverse_confidence_mode = str(inverse_confidence_mode).strip().lower()
    if inverse_confidence_mode == "":
        inverse_confidence_mode = "conditioning"
    try:
        inverse_confidence_target_gain = max(1.0e-12, float(inverse_confidence_target_gain))
    except Exception:
        inverse_confidence_target_gain = 4.0
    try:
        inverse_confidence_floor = float(inverse_confidence_floor)
    except Exception:
        inverse_confidence_floor = 0.05
    inverse_confidence_floor = min(1.0, max(0.0, inverse_confidence_floor))
    try:
        inverse_branch_beam_width = max(1, int(inverse_branch_beam_width))
    except Exception:
        inverse_branch_beam_width = 1
    try:
        inverse_micro_search_max_depth = max(1, int(inverse_micro_search_max_depth))
    except Exception:
        inverse_micro_search_max_depth = 3
    try:
        inverse_micro_search_beam_width = max(1, int(inverse_micro_search_beam_width))
    except Exception:
        inverse_micro_search_beam_width = 24
    try:
        inverse_micro_search_topk = max(1, int(inverse_micro_search_topk))
    except Exception:
        inverse_micro_search_topk = 16
    try:
        inverse_micro_search_seed_terms = max(1, int(inverse_micro_search_seed_terms))
    except Exception:
        inverse_micro_search_seed_terms = 8
    inverse_local_score_mode = _normalize_inverse_local_score_mode(inverse_local_score_mode, default="affine")
    try:
        inverse_spec_enum_max_depth = max(1, int(inverse_spec_enum_max_depth))
    except Exception:
        inverse_spec_enum_max_depth = 4
    try:
        inverse_spec_enum_max_trees = max(1, int(inverse_spec_enum_max_trees))
    except Exception:
        inverse_spec_enum_max_trees = 5000
    try:
        inverse_spec_preview_topk = max(1, int(inverse_spec_preview_topk))
    except Exception:
        inverse_spec_preview_topk = 16
    inverse_spec_local_score_mode = _normalize_inverse_local_score_mode(
        inverse_spec_local_score_mode,
        default="affine",
    )
    try:
        inverse_spec_complexity_penalty = max(0.0, float(inverse_spec_complexity_penalty))
    except Exception:
        inverse_spec_complexity_penalty = 0.0
    inverse_spec_family_battery_enable = bool(inverse_spec_family_battery_enable)
    inverse_spec_family_battery_mode = str(inverse_spec_family_battery_mode or "outer").strip().lower()
    if inverse_spec_family_battery_mode != "expanded":
        inverse_spec_family_battery_mode = "outer"
    try:
        inverse_spec_repair_quota = max(0.0, float(inverse_spec_repair_quota))
    except Exception:
        inverse_spec_repair_quota = 0.0
    hole_search_enable = bool(hole_search_enable)
    try:
        hole_search_quota = max(0.0, float(hole_search_quota))
    except Exception:
        hole_search_quota = 0.10
    try:
        hole_search_exact_budget = max(0, int(hole_search_exact_budget))
    except Exception:
        hole_search_exact_budget = 2
    try:
        hole_search_cooldown_iters = max(0, int(hole_search_cooldown_iters))
    except Exception:
        hole_search_cooldown_iters = 32
    try:
        hole_search_mine_cooldown_iters = max(0, int(hole_search_mine_cooldown_iters))
    except Exception:
        hole_search_mine_cooldown_iters = 50
    try:
        hole_search_max_frontier = max(1, int(hole_search_max_frontier))
    except Exception:
        hole_search_max_frontier = 128
    hole_search_first_class_scheduler_enable = bool(hole_search_first_class_scheduler_enable)
    hole_search_first_class_scheduler_enable = bool(
        hole_search_first_class_scheduler_enable
        and hole_search_enable
        and bool(inverse_steering_enable)
        and bool(inverse_spec_enable)
    )
    hole_search_abstraction_enable = bool(hole_search_abstraction_enable)
    hole_search_abstraction_on_improve = bool(hole_search_abstraction_on_improve)
    hole_search_abstraction_on_stall = bool(hole_search_abstraction_on_stall)
    try:
        hole_search_abstraction_cooldown_iters = max(0, int(hole_search_abstraction_cooldown_iters))
    except Exception:
        hole_search_abstraction_cooldown_iters = 25
    try:
        hole_search_abstraction_max_parents = max(1, int(hole_search_abstraction_max_parents))
    except Exception:
        hole_search_abstraction_max_parents = 2
    try:
        hole_search_abstraction_max_paths_per_parent = max(1, int(hole_search_abstraction_max_paths_per_parent))
    except Exception:
        hole_search_abstraction_max_paths_per_parent = 3
    try:
        hole_search_abstraction_improve_min_delta_log_mse = max(
            0.0,
            float(hole_search_abstraction_improve_min_delta_log_mse),
        )
    except Exception:
        hole_search_abstraction_improve_min_delta_log_mse = 0.15
    hole_search_abstraction_stage_enable = bool(hole_search_abstraction_stage_enable)
    try:
        hole_search_abstraction_stage_max_entries = max(8, int(hole_search_abstraction_stage_max_entries))
    except Exception:
        hole_search_abstraction_stage_max_entries = 64
    try:
        hole_search_abstraction_promote_topk = max(1, int(hole_search_abstraction_promote_topk))
    except Exception:
        hole_search_abstraction_promote_topk = 2
    try:
        hole_search_abstraction_promote_frontier_floor = max(0, int(hole_search_abstraction_promote_frontier_floor))
    except Exception:
        hole_search_abstraction_promote_frontier_floor = 3
    try:
        hole_search_enum_max_depth = max(1, int(hole_search_enum_max_depth))
    except Exception:
        hole_search_enum_max_depth = 4
    try:
        hole_search_enum_max_trees = max(1, int(hole_search_enum_max_trees))
    except Exception:
        hole_search_enum_max_trees = 3000
    try:
        hole_search_preview_topk = max(1, int(hole_search_preview_topk))
    except Exception:
        hole_search_preview_topk = 8
    hole_search_solver_market_enable = bool(hole_search_solver_market_enable)
    try:
        hole_search_solver_market_preview_topk = max(1, int(hole_search_solver_market_preview_topk))
    except Exception:
        hole_search_solver_market_preview_topk = 4
    try:
        hole_search_solver_market_exact_topk = max(1, int(hole_search_solver_market_exact_topk))
    except Exception:
        hole_search_solver_market_exact_topk = 2
    hole_search_solver_market_proposal_objects_enable = bool(hole_search_solver_market_proposal_objects_enable)
    inverse_spec_recursive_sr_enable = bool(inverse_spec_recursive_sr_enable)
    try:
        inverse_spec_recursive_sr_preview_topk = max(1, int(inverse_spec_recursive_sr_preview_topk))
    except Exception:
        inverse_spec_recursive_sr_preview_topk = 4
    try:
        inverse_spec_recursive_sr_exact_budget = max(1, int(inverse_spec_recursive_sr_exact_budget))
    except Exception:
        inverse_spec_recursive_sr_exact_budget = 2
    inverse_spec_constant_lift_route_enable = bool(inverse_spec_constant_lift_route_enable)
    try:
        inverse_spec_constant_lift_route_topk = max(1, int(inverse_spec_constant_lift_route_topk))
    except Exception:
        inverse_spec_constant_lift_route_topk = 2
    inverse_spec_coordinate_lift_enable = bool(inverse_spec_coordinate_lift_enable)
    try:
        inverse_spec_coordinate_lift_topk = max(1, int(inverse_spec_coordinate_lift_topk))
    except Exception:
        inverse_spec_coordinate_lift_topk = 4
    inverse_spec_coordinate_lift_mode = str(inverse_spec_coordinate_lift_mode or "both").strip().lower()
    if inverse_spec_coordinate_lift_mode not in {"single_index", "invariant", "both"}:
        inverse_spec_coordinate_lift_mode = "both"
    inverse_spec_tangent_edit_enable = bool(inverse_spec_tangent_edit_enable)
    try:
        inverse_spec_tangent_edit_topk = max(1, int(inverse_spec_tangent_edit_topk))
    except Exception:
        inverse_spec_tangent_edit_topk = 8
    inverse_spec_soft_edit_enable = bool(inverse_spec_soft_edit_enable)
    try:
        inverse_spec_soft_edit_steps = max(1, int(inverse_spec_soft_edit_steps))
    except Exception:
        inverse_spec_soft_edit_steps = 64
    try:
        inverse_spec_soft_edit_l1 = max(0.0, float(inverse_spec_soft_edit_l1))
    except Exception:
        inverse_spec_soft_edit_l1 = 1.0e-3
    inverse_spec_witness_jets_enable = bool(inverse_spec_witness_jets_enable)
    inverse_spec_witness_d2_enable = bool(inverse_spec_witness_d2_enable)
    try:
        inverse_spec_witness_max_rows = max(4, int(inverse_spec_witness_max_rows))
    except Exception:
        inverse_spec_witness_max_rows = 64
    inverse_spec_witness_loss_enable = bool(inverse_spec_witness_loss_enable)
    try:
        inverse_spec_witness_grad_weight = max(0.0, float(inverse_spec_witness_grad_weight))
    except Exception:
        inverse_spec_witness_grad_weight = 1.0
    try:
        inverse_spec_witness_d2_weight = max(0.0, float(inverse_spec_witness_d2_weight))
    except Exception:
        inverse_spec_witness_d2_weight = 0.0
    try:
        inverse_spec_witness_diag_weight = max(0.0, float(inverse_spec_witness_diag_weight))
    except Exception:
        inverse_spec_witness_diag_weight = 0.0
    try:
        inverse_spec_witness_physics_weight = max(0.0, float(inverse_spec_witness_physics_weight))
    except Exception:
        inverse_spec_witness_physics_weight = 0.0
    inverse_spec_active_var_screen_enable = bool(inverse_spec_active_var_screen_enable)
    try:
        inverse_spec_active_var_grad_tol = max(0.0, float(inverse_spec_active_var_grad_tol))
    except Exception:
        inverse_spec_active_var_grad_tol = 1.0e-3
    try:
        inverse_spec_active_var_max_count = max(1, int(inverse_spec_active_var_max_count))
    except Exception:
        inverse_spec_active_var_max_count = 4
    inverse_spec_directional_market_enable = bool(inverse_spec_directional_market_enable)
    hole_search_tournament_enable = bool(hole_search_tournament_enable)
    try:
        hole_search_tournament_n = max(2, int(hole_search_tournament_n))
    except Exception:
        hole_search_tournament_n = 8
    try:
        hole_search_tournament_elite_k = max(1, int(hole_search_tournament_elite_k))
    except Exception:
        hole_search_tournament_elite_k = 2
    try:
        hole_search_tournament_preview_trees = max(8, int(hole_search_tournament_preview_trees))
    except Exception:
        hole_search_tournament_preview_trees = 64
    try:
        inverse_spec_recursive_max_depth = max(0, int(inverse_spec_recursive_max_depth))
    except Exception:
        inverse_spec_recursive_max_depth = 2
    try:
        inverse_spec_recursive_trigger_rel_mse = max(0.0, float(inverse_spec_recursive_trigger_rel_mse))
    except Exception:
        inverse_spec_recursive_trigger_rel_mse = 0.25
    try:
        inverse_spec_recursive_seed_cap = max(1, int(inverse_spec_recursive_seed_cap))
    except Exception:
        inverse_spec_recursive_seed_cap = 6
    try:
        inverse_spec_recursive_branch_topk = max(1, int(inverse_spec_recursive_branch_topk))
    except Exception:
        inverse_spec_recursive_branch_topk = 4
    try:
        inverse_spec_recursive_child_topk = max(1, int(inverse_spec_recursive_child_topk))
    except Exception:
        inverse_spec_recursive_child_topk = 2
    try:
        if inverse_spec_max_subtree_depth is not None:
            inverse_spec_max_subtree_depth = max(1, int(inverse_spec_max_subtree_depth))
    except Exception:
        inverse_spec_max_subtree_depth = None
    try:
        inverse_spec_fit_cap = max(32, int(inverse_spec_fit_cap))
    except Exception:
        inverse_spec_fit_cap = 96
    try:
        inverse_spec_probe_cap = max(64, int(inverse_spec_probe_cap))
    except Exception:
        inverse_spec_probe_cap = 192
    try:
        inverse_spec_exact_budget = max(0, int(inverse_spec_exact_budget))
    except Exception:
        inverse_spec_exact_budget = 4
    inverse_target_mode = _normalize_inverse_target_mode(inverse_target_mode, default="robust")
    try:
        inverse_full_mapping_penalty = max(0.0, float(inverse_full_mapping_penalty))
    except Exception:
        inverse_full_mapping_penalty = 0.75
    try:
        inverse_exact_simple_target_bonus = max(0.0, float(inverse_exact_simple_target_bonus))
    except Exception:
        inverse_exact_simple_target_bonus = 0.10
    try:
        inverse_additive_descend_penalty = min(0.95, max(0.0, float(inverse_additive_descend_penalty)))
    except Exception:
        inverse_additive_descend_penalty = 0.15
    try:
        inverse_nonadditive_leaf_penalty = min(0.95, max(0.0, float(inverse_nonadditive_leaf_penalty)))
    except Exception:
        inverse_nonadditive_leaf_penalty = 0.20
    try:
        inverse_exact_path_eta = min(0.999, max(0.0, float(inverse_exact_path_eta)))
    except Exception:
        inverse_exact_path_eta = 0.98
    try:
        inverse_exact_transport_min_lin_rel = float(inverse_exact_transport_min_lin_rel)
    except Exception:
        inverse_exact_transport_min_lin_rel = 0.0
    try:
        inverse_periodic_min_valid_scale = max(0.0, float(inverse_periodic_min_valid_scale))
    except Exception:
        inverse_periodic_min_valid_scale = 1.25
    try:
        inverse_periodic_min_confidence_scale = max(0.0, float(inverse_periodic_min_confidence_scale))
    except Exception:
        inverse_periodic_min_confidence_scale = 1.35
    try:
        inverse_periodic_path_penalty = max(0.0, float(inverse_periodic_path_penalty))
    except Exception:
        inverse_periodic_path_penalty = 0.65
    try:
        inverse_nonperiodic_muldiv_bonus = max(0.0, float(inverse_nonperiodic_muldiv_bonus))
    except Exception:
        inverse_nonperiodic_muldiv_bonus = 0.10
    try:
        inverse_nonperiodic_explogsqrt_bonus = max(0.0, float(inverse_nonperiodic_explogsqrt_bonus))
    except Exception:
        inverse_nonperiodic_explogsqrt_bonus = 0.05
    try:
        inverse_branch_ambiguity_penalty = min(1.0, max(0.0, float(inverse_branch_ambiguity_penalty)))
    except Exception:
        inverse_branch_ambiguity_penalty = 0.50
    try:
        inverse_transport_min_lin_rel = max(0.0, float(inverse_transport_min_lin_rel))
    except Exception:
        inverse_transport_min_lin_rel = 0.02
    try:
        inverse_transport_min_effective_n = max(0.0, float(inverse_transport_min_effective_n))
    except Exception:
        inverse_transport_min_effective_n = 8.0
    try:
        closure_search_min_valid_frac = max(0.0, float(closure_search_min_valid_frac))
    except Exception:
        closure_search_min_valid_frac = 0.05
    try:
        closure_search_min_confidence = max(0.0, float(closure_search_min_confidence))
    except Exception:
        closure_search_min_confidence = 0.02
    try:
        closure_search_periodic_min_valid_scale = max(
            0.0, float(closure_search_periodic_min_valid_scale)
        )
    except Exception:
        closure_search_periodic_min_valid_scale = 1.0
    try:
        closure_search_periodic_min_confidence_scale = max(
            0.0, float(closure_search_periodic_min_confidence_scale)
        )
    except Exception:
        closure_search_periodic_min_confidence_scale = 1.0
    try:
        closure_search_transport_min_lin_rel = max(
            0.0, float(closure_search_transport_min_lin_rel)
        )
    except Exception:
        closure_search_transport_min_lin_rel = 0.0
    closure_search_anchor_head_compare_enable = bool(closure_search_anchor_head_compare_enable)
    repair_inverse_action_config = coerce_inverse_steering_config(locals())
    if bool(boost_enable) and bool(boost_safe_eval):
        _tmp = []
        for n in pool_nodes:
            v, _p = eval_node_hparam_safe_fn(n, x_probe, {}, safe_cfg)
            v = v.squeeze(-1)
            _tmp.append(v if torch.isfinite(v).all() else torch.zeros_like(v))
        pool_phi_safe = torch.stack(_tmp, dim=1)
        pool_norms_safe = (pool_phi_safe * pool_phi_safe).sum(dim=0)

        _tmp = []
        for n in pool_nodes:
            v, _p = eval_node_hparam_safe_fn(n, x_fit, {}, safe_cfg)
            v = v.squeeze(-1)
            _tmp.append(v if torch.isfinite(v).all() else torch.zeros_like(v))
        pool_phi_fit_safe = torch.stack(_tmp, dim=1)
        pool_norms_fit_safe = (pool_phi_fit_safe * pool_phi_fit_safe).sum(dim=0)

    # The boost pool can optionally be expanded with harvested archive subtrees.
    boost_pool_nodes = pool_nodes
    boost_pool_dims = pool_dims
    boost_pool_phi = pool_phi_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_phi
    boost_pool_norms = pool_norms_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_norms
    boost_pool_phi_fit = pool_phi_fit_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_phi_fit
    boost_pool_norms_fit = pool_norms_fit_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_norms_fit
    boost_pool_seen = set(node_str(n) for n in boost_pool_nodes)
    boost_dyn_nodes = []
    boost_dyn_dims = []
    boost_dyn_phi = None
    boost_dyn_norms = None
    boost_dyn_phi_fit = None
    boost_dyn_norms_fit = None
    last_boost_harvest_eval = -1
    hole_search_last_mine_iter = -max(1, int(hole_search_mine_cooldown_iters))
    hole_search_last_abstraction_iter = -max(1, int(hole_search_abstraction_cooldown_iters))
    parent_snapshot_store = _ParentSnapshotStore(
        max_entries=max(64, 2 * int(hole_search_max_frontier)),
        staleness_window=max(512, 8 * max(1, int(hole_search_cooldown_iters))),
    ) if bool(hole_search_enable) else None

    def _hole_search_source_bucket(source: object) -> str:
        source_key = str(source or "other")
        if source_key in ("inverse_slate", "archive_mine"):
            return source_key
        if source_key.startswith("abstraction"):
            return "abstraction"
        return "other"

    def _capture_parent_snapshot(
        *,
        residual_basin_key: Any,
        elite_id: Any,
        expr: Any,
        mapping: dict,
        eff_mse: float | None,
        raw_mse: float | None,
        current_iter: int,
        expr_str: str,
    ) -> str:
        if parent_snapshot_store is None:
            return ""
        return str(parent_snapshot_store.capture(
            residual_basin_key=residual_basin_key,
            elite_id=elite_id,
            expr=expr,
            mapping=mapping,
            eff_mse=eff_mse,
            raw_mse=raw_mse,
            current_iter=int(current_iter),
            expr_str=str(expr_str or ""),
        ) or "")

    def _prune_parent_snapshots(current_iter: int) -> None:
        if parent_snapshot_store is None:
            return
        protected_ids = set()
        if hole_frontier is not None:
            protected_ids.update(hole_frontier.active_snapshot_ids())
        if abstraction_stage is not None:
            protected_ids.update(abstraction_stage.active_snapshot_ids())
        parent_snapshot_store.prune(
            current_iter=int(current_iter),
            protected_ids=protected_ids,
        )

    def _stall_abstraction_parent_score(rec) -> tuple[float, float, float]:
        expr = getattr(rec, "best_expr", None)
        try:
            depth = float(node_depth(expr)) if expr is not None else 0.0
        except Exception:
            depth = 0.0
        try:
            size = float(node_size(expr)) if expr is not None else 0.0
        except Exception:
            size = 0.0
        try:
            visits_since_improve = float(getattr(rec, "visits_since_improve", 0.0) or 0.0)
        except Exception:
            visits_since_improve = 0.0
        try:
            mse = float(getattr(rec, "best_mse", float("inf")))
        except Exception:
            mse = float("inf")
        stagnation_score = math.log1p(max(0.0, visits_since_improve))
        structure_score = 0.35 * max(0.0, depth) + 0.08 * max(0.0, size)
        mse_score = -float(mse) if math.isfinite(mse) else -1.0e30
        return (stagnation_score + structure_score, structure_score, mse_score)

    def _emit_hole_abstraction(
        *,
        parent_key: Any,
        parent_elite_id: Any,
        parent_expr: Any,
        parent_mapping: dict,
        parent_eff_mse: float | None,
        current_iter: int,
        source: str,
    ) -> int:
        nonlocal hole_search_last_abstraction_iter
        source_key = str(source or "abstraction")
        target_frontier = abstraction_stage if abstraction_stage is not None else hole_frontier
        hole_search_stats["abstraction_attempts"] = int(hole_search_stats.get("abstraction_attempts", 0)) + 1
        if source_key == "abstraction_improve":
            hole_search_stats["abstraction_attempts_on_improve"] = int(
                hole_search_stats.get("abstraction_attempts_on_improve", 0)
            ) + 1
        elif source_key == "abstraction_stall":
            hole_search_stats["abstraction_attempts_on_stall"] = int(
                hole_search_stats.get("abstraction_attempts_on_stall", 0)
            ) + 1
            hole_search_stats["stall_abstraction_attempts"] = int(
                hole_search_stats.get("stall_abstraction_attempts", 0)
            ) + 1
        if (
            target_frontier is None
            or not bool(hole_search_enable)
            or not bool(hole_search_abstraction_enable)
            or parent_expr is None
            or parent_mapping is None
        ):
            hole_search_stats["abstraction_blocked_disabled_or_missing_input"] = int(
                hole_search_stats.get("abstraction_blocked_disabled_or_missing_input", 0)
            ) + 1
            return 0
        try:
            parent_eff_mse_f = float(parent_eff_mse)
        except Exception:
            parent_eff_mse_f = float("inf")
        if not math.isfinite(parent_eff_mse_f):
            parent_eff_mse_f = float("inf")
        try:
            if node_depth(parent_expr) < 2 or node_size(parent_expr) < 4:
                hole_search_stats["abstraction_blocked_small_expr"] = int(
                    hole_search_stats.get("abstraction_blocked_small_expr", 0)
                ) + 1
                return 0
        except Exception:
            hole_search_stats["abstraction_blocked_small_expr"] = int(
                hole_search_stats.get("abstraction_blocked_small_expr", 0)
            ) + 1
            return 0
        if (int(current_iter) - int(hole_search_last_abstraction_iter)) < int(hole_search_abstraction_cooldown_iters):
            hole_search_stats["abstraction_blocked_cooldown"] = int(
                hole_search_stats.get("abstraction_blocked_cooldown", 0)
            ) + 1
            return 0
        from ..hole_search import abstract_frontier_from_parent
        if source_key == "abstraction_improve":
            abstraction_max_paths = 1
            abstraction_preview_topk = min(int(hole_search_preview_topk), 2)
            abstraction_preview_max_trees = min(int(hole_search_enum_max_trees), 32)
        else:
            abstraction_max_paths = max(1, int(hole_search_abstraction_max_paths_per_parent))
            abstraction_preview_topk = min(int(hole_search_preview_topk), 4)
            abstraction_preview_max_trees = min(int(hole_search_enum_max_trees), 64)
        abstraction_debug = {}
        n_added = int(abstract_frontier_from_parent(
            target_frontier,
            parent_key=str(parent_key),
            parent_elite_id=str(parent_elite_id or ""),
            parent_expr=parent_expr,
            parent_mapping=parent_mapping,
            parent_eff_mse=parent_eff_mse_f,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=boost_pool_nodes,
            pool_phi_fit=boost_pool_phi_fit,
            pool_phi_probe=boost_pool_phi,
            pool_dims=boost_pool_dims,
            max_depth=int(max_depth),
            nvars=int(nvars),
            poly_degree=int(poly_degree),
            var_dims=var_dims,
            current_iter=int(current_iter),
            max_paths_per_parent=int(abstraction_max_paths),
            preview_enum_max_depth=int(min(hole_search_enum_max_depth, 2)),
            preview_enum_max_trees=int(abstraction_preview_max_trees),
            preview_topk=int(abstraction_preview_topk),
            safe_eps=float(inverse_safe_eps),
            confidence_mode=str(inverse_confidence_mode),
            confidence_target_gain=float(inverse_confidence_target_gain),
            confidence_floor=float(inverse_confidence_floor),
            source=source_key,
            recursive_enable=False,
            recursive_max_depth=0,
            recursive_trigger_rel_mse=1.0,
            regime_metadata=inverse_spec_regime_metadata,
            candidate_path_cap=max(6, 6 * int(abstraction_max_paths)),
            snapshot_parent_fn=lambda **kwargs: _capture_parent_snapshot(
                residual_basin_key=kwargs.get("parent_key", ""),
                elite_id=kwargs.get("parent_elite_id", ""),
                expr=kwargs.get("parent_expr", None),
                mapping=kwargs.get("parent_mapping", {}),
                eff_mse=kwargs.get("parent_eff_mse", None),
                raw_mse=kwargs.get("parent_eff_mse", None),
                current_iter=int(kwargs.get("current_iter", current_iter)),
                expr_str=str(kwargs.get("parent_expr_str", "") or ""),
            ),
            debug_stats=abstraction_debug,
        ))
        for key, value in abstraction_debug.items():
            stat_key = f"abstraction_{key}"
            hole_search_stats[stat_key] = int(hole_search_stats.get(stat_key, 0) or 0) + int(value)
            if source_key == "abstraction_stall":
                stall_key = f"stall_abstraction_{key}"
                hole_search_stats[stall_key] = int(hole_search_stats.get(stall_key, 0) or 0) + int(value)
        if n_added > 0:
            hole_search_last_abstraction_iter = int(current_iter)
            hole_search_stats["abstract_events"] = int(hole_search_stats.get("abstract_events", 0)) + 1
            hole_search_stats["abstracted"] = int(hole_search_stats.get("abstracted", 0)) + int(n_added)
            if abstraction_stage is not None:
                hole_search_stats["abstraction_staged"] = int(
                    hole_search_stats.get("abstraction_staged", 0)
                ) + int(n_added)
            if str(source) == "abstraction_improve":
                hole_search_stats["abstract_events_on_improve"] = int(
                    hole_search_stats.get("abstract_events_on_improve", 0)
                ) + 1
                hole_search_stats["abstracted_on_improve"] = int(
                    hole_search_stats.get("abstracted_on_improve", 0)
                ) + int(n_added)
            elif str(source) == "abstraction_stall":
                hole_search_stats["abstract_events_on_stall"] = int(
                    hole_search_stats.get("abstract_events_on_stall", 0)
                ) + 1
                hole_search_stats["abstracted_on_stall"] = int(
                    hole_search_stats.get("abstracted_on_stall", 0)
                ) + int(n_added)
            hole_search_stats["frontier_size"] = int(len(hole_frontier)) if hole_frontier is not None else 0
            hole_search_stats["abstraction_stage_size"] = int(len(abstraction_stage)) if abstraction_stage is not None else 0
            hole_search_stats["last_abstracted_iter"] = int(current_iter)
            _prune_parent_snapshots(int(current_iter))
        else:
            hole_search_stats["abstraction_added_zero"] = int(
                hole_search_stats.get("abstraction_added_zero", 0)
            ) + 1
            hole_search_stats["abstraction_stage_size"] = int(len(abstraction_stage)) if abstraction_stage is not None else 0
        return int(n_added)

    def _maybe_run_stall_check(current_eval: int) -> bool:
        nonlocal stall_best, stall_count, n_soft_restarts
        nonlocal plateau_stop_requested, plateau_stop_eval, plateau_stop_best_mse
        nonlocal plateau_stop_consecutive_soft_restarts
        if stall_window <= 0 or current_eval <= 0 or (current_eval % stall_window) != 0:
            return False
        hole_search_stats["stall_checks"] = int(hole_search_stats.get("stall_checks", 0)) + 1
        stall_metric = "raw_mse" if plateau_stop_active else "effective_mse"
        hole_search_stats["stall_metric"] = stall_metric
        current_best = _archive_best_stall_mse(arch, prefer_raw=bool(plateau_stop_active))
        hole_search_stats["stall_last_best_mse"] = (
            float(current_best) if math.isfinite(float(current_best)) else None
        )
        rel_improve = _relative_best_improvement(stall_best, current_best)
        hole_search_stats["stall_last_rel_improve"] = (
            float(rel_improve) if math.isfinite(float(rel_improve)) else None
        )
        if rel_improve < stall_delta:
            hole_search_stats["stall_triggered"] = int(hole_search_stats.get("stall_triggered", 0)) + 1
            stall_count += 1
            if (
                bool(hole_search_abstraction_enable)
                and bool(hole_search_abstraction_on_stall)
                and hole_frontier is not None
                and arch.d
            ):
                stall_parent_limit = max(1, int(hole_search_abstraction_max_parents))
                try:
                    stall_candidates = arch.best(
                        max(stall_parent_limit * 3, stall_parent_limit + 1),
                        strategy="mse_decade_size",
                    )
                except Exception:
                    stall_candidates = arch.best(max(stall_parent_limit * 3, stall_parent_limit + 1))
                hole_search_stats["stall_abstraction_parent_candidates"] = int(
                    hole_search_stats.get("stall_abstraction_parent_candidates", 0)
                ) + int(len(stall_candidates or []))
                selected_stall_parents = []
                seen_stall_keys = set()
                try:
                    best_parent = arch.best(1)[0]
                    best_sig = (
                        str(getattr(best_parent, "residual_basin_key", "") or node_str(getattr(best_parent, "best_expr", None))),
                        str(getattr(best_parent, "best_elite_id", "") or ""),
                    )
                    seen_stall_keys.add(best_sig)
                    selected_stall_parents.append(best_parent)
                    hole_search_stats["stall_abstraction_parent_selected_best"] = int(
                        hole_search_stats.get("stall_abstraction_parent_selected_best", 0)
                    ) + 1
                except Exception:
                    pass
                ranked_candidates = list(stall_candidates or [])
                ranked_candidates.sort(key=_stall_abstraction_parent_score, reverse=True)
                for stall_parent in ranked_candidates:
                    sig = (
                        str(getattr(stall_parent, "residual_basin_key", "") or node_str(getattr(stall_parent, "best_expr", None))),
                        str(getattr(stall_parent, "best_elite_id", "") or ""),
                    )
                    if sig in seen_stall_keys:
                        continue
                    seen_stall_keys.add(sig)
                    selected_stall_parents.append(stall_parent)
                    if len(selected_stall_parents) >= stall_parent_limit:
                        break
                hole_search_stats["stall_abstraction_parent_selected"] = int(
                    hole_search_stats.get("stall_abstraction_parent_selected", 0)
                ) + int(len(selected_stall_parents[:stall_parent_limit]))
                for stall_parent in selected_stall_parents[:stall_parent_limit]:
                    _emit_hole_abstraction(
                        parent_key=str(getattr(stall_parent, "residual_basin_key", "") or node_str(getattr(stall_parent, "best_expr", None))),
                        parent_elite_id=str(getattr(stall_parent, "best_elite_id", "") or ""),
                        parent_expr=getattr(stall_parent, "best_expr", None),
                        parent_mapping=getattr(stall_parent, "mapping", None),
                        parent_eff_mse=getattr(stall_parent, "best_mse", None),
                        current_iter=int(current_eval),
                        source="abstraction_stall",
                    )
        else:
            stall_count = 0
            plateau_stop_consecutive_soft_restarts = 0
        stall_best = current_best

        if mutation_refine_enable and refine_state is not None:
            relax_base = max(1.0, float(refine_cfg.get("stall_gate_relax_factor", 3.0)))
            relax_cap = max(1.0, float(refine_cfg.get("stall_gate_relax_max", 100.0)))
            if stall_count > 0:
                refine_state["gate_relax_factor"] = min(relax_cap, relax_base ** stall_count)
            else:
                refine_state["gate_relax_factor"] = 1.0
            refine_state["stall_count"] = int(stall_count)

        if stall_count >= stall_patience:
            n_soft_restarts += 1
            plateau_stop_consecutive_soft_restarts += 1
            rng.seed(seed + n_soft_restarts * 1_000_003)
            stall_count = 0
            if mutation_refine_enable and refine_state is not None:
                refine_state["gate_relax_factor"] = 1.0
                refine_state["stall_count"] = 0
            if plateau_stop_active:
                refine_diagnostics["plateau_stop_soft_restarts"] = int(plateau_stop_consecutive_soft_restarts)
                refine_diagnostics["plateau_stop_last_eval"] = int(current_eval)
                if _plateau_stop_should_stop(
                    enable=True,
                    n_evaluated=int(current_eval),
                    min_evals=int(plateau_stop_min_evals_i),
                    consecutive_soft_restarts=int(plateau_stop_consecutive_soft_restarts),
                    max_soft_restarts=int(plateau_stop_max_soft_restarts_i),
                    has_archive=bool(arch.d),
                ):
                    plateau_stop_requested = True
                    plateau_stop_eval = int(current_eval)
                    plateau_stop_best_mse = float(current_best)
                    refine_diagnostics["plateau_stop_requested"] = True
                    refine_diagnostics["plateau_stop_eval"] = int(current_eval)
                    refine_diagnostics["plateau_stop_best_mse"] = (
                        float(current_best) if math.isfinite(float(current_best)) else None
                    )
                    if verbose:
                        print(
                            f"[mutate] plateau stop requested at iter {current_eval}: "
                            f"soft_restarts={plateau_stop_consecutive_soft_restarts}, "
                            f"best_mse={current_best:.3e}"
                        )
            if verbose and print_every > 0:
                print(f"[mutate] soft restart #{n_soft_restarts} at iter {current_eval}, "
                      f"best_mse={current_best:.3e}")
        return bool(plateau_stop_requested)

    def _execute_hole_search(
        *,
        opportunity,
        parent_node: Any,
        parent_mapping: dict,
        resolution_source: str,
        resolved_parent_key: Any,
        resolved_parent_rec: Any,
        resolved_parent_elite_id: Any,
        resolved_parent_eff_mse: float | None,
        current_iter: int,
        exec_ctx: _ExecutionContext,
        exact_budget_override: int | None = None,
    ) -> tuple[Any, dict]:
        from ..hole_search import run_hole_search_action

        exec_ctx.executed_parent_key = resolved_parent_key
        exec_ctx.executed_parent_rec = resolved_parent_rec
        exec_ctx.executed_parent_elite_id = str(resolved_parent_elite_id or "")
        try:
            exec_ctx.executed_parent_eff_mse = (
                None if resolved_parent_eff_mse is None else float(resolved_parent_eff_mse)
            )
        except Exception:
            exec_ctx.executed_parent_eff_mse = None

        hole_search_stats["run_hole_search_action_called"] = int(
            hole_search_stats.get("run_hole_search_action_called", 0)
        ) + 1
        hole_ret = run_hole_search_action(
            opportunity,
            parent_node=parent_node,
            parent_mapping=parent_mapping,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=boost_pool_nodes,
            pool_phi_fit=boost_pool_phi_fit,
            pool_phi_probe=boost_pool_phi,
            pool_dims=boost_pool_dims,
            rng=rng,
            max_depth=int(max_depth),
            nvars=int(nvars),
            poly_degree=int(poly_degree),
            var_dims=var_dims,
            enum_max_depth=int(hole_search_enum_max_depth),
            enum_max_trees=int(hole_search_enum_max_trees),
            preview_topk=int(hole_search_preview_topk),
            exact_budget=max(
                1,
                int(exact_budget_override if exact_budget_override is not None else hole_search_exact_budget),
            ),
            solver_market_enable=bool(hole_search_solver_market_enable),
            solver_market_preview_topk=int(hole_search_solver_market_preview_topk),
            solver_market_exact_topk=int(hole_search_solver_market_exact_topk),
            solver_market_proposal_objects_enable=bool(hole_search_solver_market_proposal_objects_enable),
            inverse_spec_recursive_sr_enable=bool(inverse_spec_recursive_sr_enable),
            inverse_spec_recursive_sr_preview_topk=int(inverse_spec_recursive_sr_preview_topk),
            inverse_spec_recursive_sr_exact_budget=int(inverse_spec_recursive_sr_exact_budget),
            inverse_spec_constant_lift_route_enable=bool(inverse_spec_constant_lift_route_enable),
            inverse_spec_constant_lift_route_topk=int(inverse_spec_constant_lift_route_topk),
            inverse_spec_coordinate_lift_enable=bool(inverse_spec_coordinate_lift_enable),
            inverse_spec_coordinate_lift_topk=int(inverse_spec_coordinate_lift_topk),
            inverse_spec_coordinate_lift_mode=str(inverse_spec_coordinate_lift_mode or "both"),
            inverse_spec_tangent_edit_enable=bool(inverse_spec_tangent_edit_enable),
            inverse_spec_tangent_edit_topk=int(inverse_spec_tangent_edit_topk),
            inverse_spec_soft_edit_enable=bool(inverse_spec_soft_edit_enable),
            inverse_spec_soft_edit_steps=int(inverse_spec_soft_edit_steps),
            inverse_spec_soft_edit_l1=float(inverse_spec_soft_edit_l1),
            inverse_spec_witness_jets_enable=bool(inverse_spec_witness_jets_enable),
            inverse_spec_witness_d2_enable=bool(inverse_spec_witness_d2_enable),
            inverse_spec_witness_max_rows=int(inverse_spec_witness_max_rows),
            inverse_spec_witness_loss_enable=bool(inverse_spec_witness_loss_enable),
            inverse_spec_witness_grad_weight=float(inverse_spec_witness_grad_weight),
            inverse_spec_witness_d2_weight=float(inverse_spec_witness_d2_weight),
            inverse_spec_witness_diag_weight=float(inverse_spec_witness_diag_weight),
            inverse_spec_witness_physics_weight=float(inverse_spec_witness_physics_weight),
            inverse_spec_active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
            inverse_spec_active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
            inverse_spec_active_var_max_count=int(inverse_spec_active_var_max_count),
            inverse_spec_directional_market_enable=bool(inverse_spec_directional_market_enable),
            max_subtree_depth=(
                int(inverse_spec_max_subtree_depth)
                if inverse_spec_max_subtree_depth is not None
                else None
            ),
            complexity_penalty=float(complexity_penalty),
            family_battery_enable=bool(inverse_spec_family_battery_enable),
            family_battery_mode=str(inverse_spec_family_battery_mode or "outer"),
            recursive_enable=bool(inverse_spec_recursive_enable),
            recursive_max_depth=int(inverse_spec_recursive_max_depth),
            recursive_trigger_rel_mse=float(inverse_spec_recursive_trigger_rel_mse),
            recursive_seed_cap=int(inverse_spec_recursive_seed_cap),
            recursive_branch_topk=int(inverse_spec_recursive_branch_topk),
            recursive_child_topk=int(inverse_spec_recursive_child_topk),
            safe_eps=float(inverse_safe_eps),
            confidence_mode=str(inverse_confidence_mode),
            confidence_target_gain=float(inverse_confidence_target_gain),
            confidence_floor=float(inverse_confidence_floor),
            branch_beam_width=int(inverse_branch_beam_width),
            proj=proj,
            fp_mode=str(fp_mode),
            q_scale=float(q_scale),
            q_clip=float(q_clip),
            score_expr_cfg=refine_cfg,
            score_expr_fn=score_expr_fn,
            return_meta=True,
            inverse_spec_regime_metadata=inverse_spec_regime_metadata,
        )
        if isinstance(hole_ret, tuple):
            expr, hole_meta = hole_ret
        else:
            expr, hole_meta = hole_ret, {}
        if expr is None:
            hole_search_stats["child_expr_none"] = int(
                hole_search_stats.get("child_expr_none", 0)
            ) + 1
        hole_frontier.record_spec_attempt(
            current_iter=int(current_iter),
            child_eff_mse=hole_meta.get("hole_search_best_eff_mse") if isinstance(hole_meta, dict) else None,
            opportunity=opportunity,
        )
        hole_search_stats["fired"] = int(hole_search_stats.get("fired", 0)) + 1
        if isinstance(hole_meta, dict):
            hole_meta["hole_search_selected_exact_budget"] = max(
                1,
                int(exact_budget_override if exact_budget_override is not None else hole_search_exact_budget),
            )
            hole_meta["hole_search_executed_parent_key"] = (
                None if exec_ctx.executed_parent_key is None else str(exec_ctx.executed_parent_key)
            )
            hole_meta["hole_search_executed_parent_elite_id"] = str(exec_ctx.executed_parent_elite_id)
            hole_meta["hole_search_selected_parent_key"] = (
                None if exec_ctx.selected_parent_key is None else str(exec_ctx.selected_parent_key)
            )
            hole_meta["hole_search_selected_parent_elite_id"] = str(exec_ctx.selected_parent_elite_id)
            hole_meta["hole_search_parent_resolution_source"] = str(resolution_source or "")
            hole_meta["hole_search_parent_snapshot_id"] = str(
                getattr(opportunity, "parent_snapshot_id", "") or ""
            )
        return expr, hole_meta

    def _execute_prepared_spec_expand(
        *,
        opportunity,
        resolution: Mapping[str, Any] | None,
        current_iter: int,
        exec_ctx: _ExecutionContext,
        exact_budget_override: int | None = None,
    ) -> tuple[Any, dict[str, Any], Any, str | None, float | None, float | None]:
        expr = None
        hole_meta: dict[str, Any] = {}
        executed_opp = None
        executed_status = None
        executed_wall_s = None
        executed_shortlist_eff_mse = None
        resolved_parent_node = None
        resolved_parent_mapping = None
        resolved_parent_key = str(getattr(opportunity, "parent_key", "") or "") if opportunity is not None else ""
        resolved_parent_elite_id = str(getattr(opportunity, "parent_elite_id", "") or "") if opportunity is not None else ""
        resolved_parent_eff_mse = None
        if hole_frontier is None or opportunity is None or resolution is None:
            return expr, hole_meta, executed_opp, executed_status, executed_wall_s, executed_shortlist_eff_mse
        if len(hole_frontier) > 0:
            hole_search_stats["selected_with_any_frontier_entries"] = int(
                hole_search_stats.get("selected_with_any_frontier_entries", 0)
            ) + 1
        if hole_frontier.nonempty(int(current_iter)):
            hole_search_stats["selected_with_nonempty_frontier"] = int(
                hole_search_stats.get("selected_with_nonempty_frontier", 0)
            ) + 1
        executed_opp = opportunity
        hole_search_stats["selected_with_opportunity"] = int(
            hole_search_stats.get("selected_with_opportunity", 0)
        ) + 1
        opp_source_bucket = _hole_search_source_bucket(getattr(opportunity, "source", "other"))
        hole_search_stats[f"selected_with_opportunity_{opp_source_bucket}"] = int(
            hole_search_stats.get(f"selected_with_opportunity_{opp_source_bucket}", 0)
        ) + 1
        resolution_source = str(resolution.get("resolution_source", "") or "")
        opp_rec = resolution.get("rec", None)
        opp_elite = resolution.get("elite", None)
        opp_snapshot = resolution.get("snapshot", None)
        if resolution_source == "live_archive" and opp_rec is not None and opp_elite is not None:
            hole_search_stats["selected_with_resolved_parent"] = int(
                hole_search_stats.get("selected_with_resolved_parent", 0)
            ) + 1
            resolved_parent_node = getattr(opp_elite, "expr", None)
            resolved_parent_mapping = getattr(opp_elite, "mapping", None)
            resolved_parent_key = str(getattr(opportunity, "parent_key", "") or "")
            resolved_parent_elite_id = str(
                getattr(opp_elite, "elite_id", getattr(opportunity, "parent_elite_id", "")) or ""
            )
            resolved_parent_eff_mse = getattr(opp_elite, "mse", float("inf"))
            expr, hole_meta = _execute_hole_search(
                opportunity=opportunity,
                parent_node=resolved_parent_node,
                parent_mapping=resolved_parent_mapping,
                resolution_source="live_archive",
                resolved_parent_key=resolved_parent_key,
                resolved_parent_rec=opp_rec,
                resolved_parent_elite_id=resolved_parent_elite_id,
                resolved_parent_eff_mse=resolved_parent_eff_mse,
                current_iter=int(current_iter),
                exec_ctx=exec_ctx,
                exact_budget_override=exact_budget_override,
            )
        elif resolution_source == "snapshot" and opp_snapshot is not None:
            hole_search_stats["selected_with_snapshot_parent"] = int(
                hole_search_stats.get("selected_with_snapshot_parent", 0)
            ) + 1
            hole_search_stats[f"selected_with_snapshot_parent_{opp_source_bucket}"] = int(
                hole_search_stats.get(f"selected_with_snapshot_parent_{opp_source_bucket}", 0)
            ) + 1
            resolved_parent_node = getattr(opp_snapshot, "expr", None)
            resolved_parent_mapping = getattr(opp_snapshot, "mapping", None)
            resolved_parent_key = str(getattr(opp_snapshot, "residual_basin_key", opportunity.parent_key) or opportunity.parent_key)
            resolved_parent_elite_id = str(
                getattr(opp_snapshot, "elite_id", getattr(opportunity, "parent_elite_id", "")) or ""
            )
            resolved_parent_eff_mse = getattr(opp_snapshot, "eff_mse", float("inf"))
            expr, hole_meta = _execute_hole_search(
                opportunity=opportunity,
                parent_node=resolved_parent_node,
                parent_mapping=resolved_parent_mapping,
                resolution_source="snapshot",
                resolved_parent_key=resolved_parent_key,
                resolved_parent_rec=None,
                resolved_parent_elite_id=resolved_parent_elite_id,
                resolved_parent_eff_mse=resolved_parent_eff_mse,
                current_iter=int(current_iter),
                exec_ctx=exec_ctx,
                exact_budget_override=exact_budget_override,
            )
        if (
            isinstance(hole_meta, Mapping)
            and hole_frontier is not None
            and resolved_parent_node is not None
            and resolved_parent_mapping is not None
        ):
            followup_rows = [
                row
                for row in list(hole_meta.get("hole_search_followup_spec_states", []) or [])
                if isinstance(row, Mapping)
            ]
            if followup_rows:
                parent_snapshot_id = str(resolution.get("bound_snapshot_id", "") or getattr(opportunity, "parent_snapshot_id", "") or "")
                if not parent_snapshot_id:
                    parent_snapshot_id = _capture_parent_snapshot(
                        residual_basin_key=str(resolved_parent_key),
                        elite_id=str(resolved_parent_elite_id),
                        expr=resolved_parent_node,
                        mapping=resolved_parent_mapping,
                        eff_mse=float(resolved_parent_eff_mse) if resolved_parent_eff_mse is not None else None,
                        raw_mse=float(resolved_parent_eff_mse) if resolved_parent_eff_mse is not None else None,
                        current_iter=int(current_iter),
                        expr_str=node_str(resolved_parent_node),
                    )
                if parent_snapshot_id:
                    emitted_count = int(len(followup_rows))
                    ingested_count = int(
                        hole_frontier.ingest_opportunity_slate(
                            str(resolved_parent_key),
                            node_str(resolved_parent_node),
                            followup_rows,
                            int(current_iter),
                            parent_elite_id=str(resolved_parent_elite_id),
                            parent_snapshot_id=str(parent_snapshot_id),
                            source="hole_followup",
                        )
                    )
                    hole_search_stats["followup_spec_states_emitted"] = int(
                        hole_search_stats.get("followup_spec_states_emitted", 0)
                    ) + emitted_count
                    hole_search_stats["followup_spec_states_ingested"] = int(
                        hole_search_stats.get("followup_spec_states_ingested", 0)
                    ) + ingested_count
                    hole_search_stats["frontier_size"] = int(len(hole_frontier))
        if isinstance(hole_meta, Mapping):
            executed_status = str(hole_meta.get("status", "ok") or "ok")
            try:
                executed_wall_s = float(hole_meta.get("hole_search_wall_seconds", 0.0))
            except Exception:
                executed_wall_s = None
            try:
                shortlist_mse = hole_meta.get("hole_search_best_eff_mse", None)
                executed_shortlist_eff_mse = None if shortlist_mse is None else float(shortlist_mse)
            except Exception:
                executed_shortlist_eff_mse = None
        hole_search_stats["selected"] = int(hole_search_stats.get("selected", 0)) + 1
        hole_search_stats["frontier_size"] = int(len(hole_frontier))
        return (
            expr,
            hole_meta,
            executed_opp,
            executed_status,
            executed_wall_s,
            executed_shortlist_eff_mse,
        )

    def _ingest_hole_search_slate_from_meta(
        *,
        meta_dict: Any,
        source: str,
        current_iter: int,
        parent_key: Any,
        parent_rec: Any,
    ) -> int:
        if hole_frontier is None or not isinstance(meta_dict, Mapping):
            return 0
        slate = meta_dict.get("repair_opportunity_slate_final", meta_dict.get("repair_opportunity_slate", []))
        if not slate:
            return 0
        parent_expr = getattr(parent_rec, "best_expr", None)
        parent_mapping = getattr(parent_rec, "mapping", None)
        if parent_expr is None or parent_mapping is None:
            return 0
        parent_snapshot_id = _capture_parent_snapshot(
            residual_basin_key=str(parent_key),
            elite_id=str(getattr(parent_rec, "best_elite_id", "") or ""),
            expr=parent_expr,
            mapping=parent_mapping,
            eff_mse=float(getattr(parent_rec, "best_mse", float("inf"))),
            raw_mse=float(getattr(parent_rec, "best_raw_mse", getattr(parent_rec, "best_mse", float("inf")))),
            current_iter=int(current_iter),
            expr_str=node_str(parent_expr),
        )
        n_ingested = hole_frontier.ingest_opportunity_slate(
            str(parent_key),
            node_str(parent_expr),
            list(slate),
            int(current_iter),
            parent_elite_id=str(getattr(parent_rec, "best_elite_id", "") or ""),
            parent_snapshot_id=str(parent_snapshot_id),
            source=str(source or "other"),
        )
        hole_search_stats["ingested"] = int(hole_search_stats.get("ingested", 0)) + int(n_ingested)
        hole_search_stats["frontier_size"] = int(len(hole_frontier))
        _prune_parent_snapshots(int(current_iter))
        return int(n_ingested)

    def _resolve_hole_search_parent(
        opp,
        *,
        current_iter: int,
        touch_snapshot: bool,
    ) -> dict[str, Any] | None:
        snapshot = None
        snapshot_id = str(getattr(opp, "parent_snapshot_id", "") or "")
        if bool(hole_search_first_class_scheduler_enable) and not snapshot_id:
            source_bucket = _hole_search_source_bucket(getattr(opp, "source", "other"))
            hole_search_stats["snapshot_parent_missing"] = int(
                hole_search_stats.get("snapshot_parent_missing", 0)
            ) + 1
            hole_search_stats[f"snapshot_parent_missing_{source_bucket}"] = int(
                hole_search_stats.get(f"snapshot_parent_missing_{source_bucket}", 0)
            ) + 1
            return None
        if parent_snapshot_store is not None and snapshot_id:
            snapshot = parent_snapshot_store.get(
                snapshot_id,
                current_iter=int(current_iter) if bool(touch_snapshot) else None,
            )
            if snapshot is not None:
                try:
                    snapshot_expr_str = str(getattr(snapshot, "expr_str", "") or node_str(getattr(snapshot, "expr", None)))
                except Exception:
                    snapshot_expr_str = str(getattr(opp, "parent_expr_str", "") or "")
                live_rec, live_elite = arch.resolve_elite(
                    getattr(snapshot, "residual_basin_key", getattr(opp, "parent_key", "")),
                    elite_id=str(getattr(snapshot, "elite_id", getattr(opp, "parent_elite_id", "")) or getattr(opp, "parent_elite_id", "")),
                    expr_str=snapshot_expr_str,
                )
                if live_rec is not None and live_elite is not None:
                    try:
                        live_expr_str = node_str(getattr(live_elite, "expr", None))
                    except Exception:
                        live_expr_str = ""
                    same_expr = bool(live_expr_str) and live_expr_str == snapshot_expr_str
                    same_elite = str(getattr(live_elite, "elite_id", "") or "") == str(getattr(snapshot, "elite_id", "") or getattr(opp, "parent_elite_id", "") or "")
                    if same_expr and (same_elite or (not bool(hole_search_first_class_scheduler_enable))):
                        return {
                            "resolution_source": "live_archive",
                            "rec": live_rec,
                            "elite": live_elite,
                            "snapshot": snapshot,
                            "bound_snapshot_id": snapshot_id,
                        }
                if bool(hole_search_first_class_scheduler_enable):
                    return {
                        "resolution_source": "snapshot",
                        "rec": None,
                        "elite": None,
                        "snapshot": snapshot,
                        "bound_snapshot_id": snapshot_id,
                    }
        opp_rec, opp_elite = arch.resolve_elite(
            opp.parent_key,
            elite_id=getattr(opp, "parent_elite_id", ""),
            expr_str=opp.parent_expr_str,
        )
        if opp_rec is not None and opp_elite is not None:
            return {
                "resolution_source": "live_archive",
                "rec": opp_rec,
                "elite": opp_elite,
                "snapshot": snapshot,
                "bound_snapshot_id": snapshot_id,
            }
        if snapshot is not None:
            return {
                "resolution_source": "snapshot",
                "rec": None,
                "elite": None,
                "snapshot": snapshot,
                "bound_snapshot_id": snapshot_id,
            }
        return None

    def _promote_abstraction_stage(current_iter: int) -> int:
        if abstraction_stage is None or hole_frontier is None:
            return 0
        try:
            abstraction_stage.prune(int(current_iter))
        except Exception:
            pass
        eligible_stage = abstraction_stage.eligible(int(current_iter))
        if not eligible_stage:
            hole_search_stats["abstraction_stage_size"] = int(len(abstraction_stage))
            return 0

        frontier_eligible = hole_frontier.eligible(int(current_iter))
        frontier_scores = sorted(float(hole_frontier._score(opp)) for opp in frontier_eligible)
        frontier_median = None
        if frontier_scores:
            frontier_median = float(frontier_scores[len(frontier_scores) // 2])
        frontier_sparse = len(frontier_eligible) < int(hole_search_abstraction_promote_frontier_floor)

        promoted = 0
        dead_stage_keys: list[tuple[str, str, tuple[int, ...], str]] = []
        eligible_stage.sort(key=abstraction_stage._score, reverse=True)
        for opp in eligible_stage:
            if promoted >= int(hole_search_abstraction_promote_topk):
                break
            resolution = _resolve_hole_search_parent(
                opp,
                current_iter=int(current_iter),
                touch_snapshot=False,
            )
            if resolution is None:
                dead_stage_keys.append(opp.frontier_key)
                continue
            stage_score = float(abstraction_stage._score(opp))
            if (not frontier_sparse) and frontier_median is not None and stage_score < frontier_median:
                continue
            existing = hole_frontier._entries.get(opp.frontier_key)
            existing_score = float(hole_frontier._score(existing)) if existing is not None else -float("inf")
            if existing is not None and stage_score < existing_score:
                abstraction_stage.drop_spec_state(frontier_key=opp.frontier_key)
                continue
            hole_frontier.enqueue_spec_state(
                opp,
                current_iter=int(current_iter),
                preserve_existing_lifecycle=True,
            )
            abstraction_stage.drop_spec_state(frontier_key=opp.frontier_key)
            promoted += 1

        for dead_key in dead_stage_keys:
            abstraction_stage.drop_spec_state(frontier_key=dead_key)
        if dead_stage_keys:
            hole_search_stats["abstraction_stage_dead_pruned"] = int(
                hole_search_stats.get("abstraction_stage_dead_pruned", 0)
            ) + int(len(dead_stage_keys))
        if promoted > 0:
            hole_search_stats["abstraction_promoted"] = int(
                hole_search_stats.get("abstraction_promoted", 0)
            ) + int(promoted)
            hole_search_stats["abstraction_promoted_on_prepare"] = int(
                hole_search_stats.get("abstraction_promoted_on_prepare", 0)
            ) + int(promoted)
        hole_search_stats["frontier_size"] = int(len(hole_frontier))
        hole_search_stats["abstraction_stage_size"] = int(len(abstraction_stage))
        return int(promoted)

    def _prepare_hole_search_opportunity(current_iter: int):
        nonlocal hole_search_last_mine_iter
        if hole_frontier is None or not (bool(hole_search_enable) and bool(inverse_steering_enable) and bool(inverse_spec_enable)):
            return None, None
        hole_search_stats["prepare_calls"] = int(hole_search_stats.get("prepare_calls", 0)) + 1
        prepare_t0 = time.perf_counter()
        try:
            prune_t0 = time.perf_counter()
            try:
                hole_frontier.prune(int(current_iter))
            except Exception:
                pass
            if abstraction_stage is not None:
                try:
                    abstraction_stage.prune(int(current_iter))
                except Exception:
                    pass
            _prune_parent_snapshots(int(current_iter))
            hole_search_stats["prepare_prune_wall_seconds"] = float(
                hole_search_stats.get("prepare_prune_wall_seconds", 0.0)
            ) + max(0.0, time.perf_counter() - prune_t0)
            _promote_abstraction_stage(int(current_iter))

            if (
                len(hole_frontier) < 3
                and arch.d
                and (int(current_iter) - int(hole_search_last_mine_iter))
                >= int(hole_search_mine_cooldown_iters)
            ):
                from ..hole_search import mine_frontier_from_archive
                mine_t0 = time.perf_counter()
                try:
                    archive_recs = list(arch.d.values())
                    mine_frontier_from_archive(
                        hole_frontier, archive_recs,
                        x_fit=x_fit, y_fit=y_fit,
                        x_probe=x_probe, y_probe=y_probe,
                        pool_nodes=boost_pool_nodes,
                        pool_phi_fit=boost_pool_phi_fit,
                        pool_phi_probe=boost_pool_phi,
                        pool_dims=boost_pool_dims,
                        max_depth=int(max_depth),
                        nvars=int(nvars),
                        poly_degree=int(poly_degree),
                        var_dims=var_dims,
                        current_iter=int(current_iter),
                        max_parents=5,
                        max_paths_per_parent=3,
                        preview_enum_max_depth=2,
                        preview_enum_max_trees=32,
                        preview_topk=4,
                        safe_eps=float(inverse_safe_eps),
                        confidence_mode=str(inverse_confidence_mode),
                        confidence_target_gain=float(inverse_confidence_target_gain),
                        confidence_floor=float(inverse_confidence_floor),
                        snapshot_parent_fn=lambda **kwargs: _capture_parent_snapshot(
                            residual_basin_key=kwargs.get("parent_key"),
                            elite_id=kwargs.get("parent_elite_id"),
                            expr=kwargs.get("parent_expr"),
                            mapping=kwargs.get("parent_mapping"),
                            eff_mse=kwargs.get("parent_eff_mse"),
                            raw_mse=kwargs.get("parent_eff_mse"),
                            current_iter=int(kwargs.get("current_iter", current_iter)),
                            expr_str=str(kwargs.get("parent_expr_str", "")),
                        ),
                    )
                    hole_search_stats["mined"] = int(hole_search_stats.get("mined", 0)) + 1
                    hole_search_stats["frontier_size"] = int(len(hole_frontier))
                    hole_search_stats["abstraction_stage_size"] = int(len(abstraction_stage)) if abstraction_stage is not None else 0
                    hole_search_last_mine_iter = int(current_iter)
                    hole_search_stats["last_mined_iter"] = int(current_iter)
                except Exception:
                    pass
                finally:
                    hole_search_stats["prepare_mine_wall_seconds"] = float(
                        hole_search_stats.get("prepare_mine_wall_seconds", 0.0)
                    ) + max(0.0, time.perf_counter() - mine_t0)

            if len(hole_frontier) > 0:
                hole_search_stats["prepared_with_any_frontier_entries"] = int(
                    hole_search_stats.get("prepared_with_any_frontier_entries", 0)
                ) + 1
            if hole_frontier.nonempty(int(current_iter)):
                hole_search_stats["prepared_with_nonempty_frontier"] = int(
                    hole_search_stats.get("prepared_with_nonempty_frontier", 0)
                ) + 1

            resolution_cache = {}
            dead_entries = {}

            def _is_executable(opp) -> bool:
                hole_search_stats["prepared_executable_checks"] = int(
                    hole_search_stats.get("prepared_executable_checks", 0)
                ) + 1
                resolution = _resolve_hole_search_parent(
                    opp,
                    current_iter=int(current_iter),
                    touch_snapshot=False,
                )
                if resolution is None:
                    hole_search_stats["prepared_resolution_missing"] = int(
                        hole_search_stats.get("prepared_resolution_missing", 0)
                    ) + 1
                    dead_entries[tuple(opp.frontier_key)] = _hole_search_source_bucket(
                        getattr(opp, "source", "other")
                    )
                    return False
                resolution_source = str(resolution.get("resolution_source", "") or "")
                if resolution_source == "live_archive":
                    hole_search_stats["prepared_resolution_live_archive"] = int(
                        hole_search_stats.get("prepared_resolution_live_archive", 0)
                    ) + 1
                elif resolution_source == "snapshot":
                    hole_search_stats["prepared_resolution_snapshot"] = int(
                        hole_search_stats.get("prepared_resolution_snapshot", 0)
                    ) + 1
                resolution_cache[opp.frontier_key] = resolution
                return True

            select_t0 = time.perf_counter()
            # ---- Risk-seeking tournament: cheap-preview N, full-execute top-k ----
            if bool(hole_search_tournament_enable):
                tournament_candidates = hole_frontier.select_n_spec_states(
                    int(current_iter), rng, _is_executable,
                    n=int(hole_search_tournament_n),
                )
                hole_search_stats["tournament_candidates"] = int(
                    hole_search_stats.get("tournament_candidates", 0)
                ) + int(len(tournament_candidates))
                if len(tournament_candidates) >= 2:
                    from ..hole_search import run_hole_tournament

                    def _parent_resolver_for_tournament(opp):
                        res = resolution_cache.get(opp.frontier_key)
                        if res is None:
                            return None
                        src = str(res.get("resolution_source", "") or "")
                        if src == "live_archive":
                            elite = res.get("elite")
                            if elite is None:
                                return None
                            return {"parent_node": elite.expr, "parent_mapping": elite.mapping}
                        if src == "snapshot":
                            snap = res.get("snapshot")
                            if snap is None:
                                return None
                            return {"parent_node": snap.expr, "parent_mapping": snap.mapping}
                        return None

                    tournament_t0 = time.perf_counter()
                    elites = run_hole_tournament(
                        tournament_candidates,
                        parent_resolver=_parent_resolver_for_tournament,
                        x_fit=x_fit,
                        y_fit=y_fit,
                        x_probe=x_probe,
                        y_probe=y_probe,
                        pool_nodes=boost_pool_nodes,
                        pool_phi_fit=boost_pool_phi_fit,
                        pool_phi_probe=boost_pool_phi,
                        pool_dims=boost_pool_dims,
                        max_depth=int(max_depth),
                        nvars=int(nvars),
                        poly_degree=int(poly_degree),
                        var_dims=var_dims,
                        preview_budget=int(hole_search_tournament_preview_trees),
                        preview_topk=min(4, int(hole_search_preview_topk)),
                        elite_k=int(hole_search_tournament_elite_k),
                        solver_market_enable=bool(hole_search_solver_market_enable),
                        solver_market_preview_topk=int(hole_search_solver_market_preview_topk),
                        solver_market_exact_topk=int(hole_search_solver_market_exact_topk),
                        solver_market_proposal_objects_enable=bool(
                            hole_search_solver_market_proposal_objects_enable
                        ),
                        inverse_spec_recursive_sr_enable=bool(inverse_spec_recursive_sr_enable),
                        inverse_spec_recursive_sr_preview_topk=int(inverse_spec_recursive_sr_preview_topk),
                        inverse_spec_recursive_sr_exact_budget=int(inverse_spec_recursive_sr_exact_budget),
                        inverse_spec_constant_lift_route_enable=bool(inverse_spec_constant_lift_route_enable),
                        inverse_spec_constant_lift_route_topk=int(inverse_spec_constant_lift_route_topk),
                        inverse_spec_coordinate_lift_enable=bool(inverse_spec_coordinate_lift_enable),
                        inverse_spec_coordinate_lift_topk=int(inverse_spec_coordinate_lift_topk),
                        inverse_spec_coordinate_lift_mode=str(inverse_spec_coordinate_lift_mode or "both"),
                        inverse_spec_tangent_edit_enable=bool(inverse_spec_tangent_edit_enable),
                        inverse_spec_tangent_edit_topk=int(inverse_spec_tangent_edit_topk),
                        inverse_spec_soft_edit_enable=bool(inverse_spec_soft_edit_enable),
                        inverse_spec_soft_edit_steps=int(inverse_spec_soft_edit_steps),
                        inverse_spec_soft_edit_l1=float(inverse_spec_soft_edit_l1),
                        inverse_spec_witness_jets_enable=bool(inverse_spec_witness_jets_enable),
                        inverse_spec_witness_d2_enable=bool(inverse_spec_witness_d2_enable),
                        inverse_spec_witness_max_rows=int(inverse_spec_witness_max_rows),
                        inverse_spec_witness_loss_enable=bool(inverse_spec_witness_loss_enable),
                        inverse_spec_witness_grad_weight=float(inverse_spec_witness_grad_weight),
                        inverse_spec_witness_d2_weight=float(inverse_spec_witness_d2_weight),
                        inverse_spec_witness_diag_weight=float(inverse_spec_witness_diag_weight),
                        inverse_spec_witness_physics_weight=float(inverse_spec_witness_physics_weight),
                        inverse_spec_active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
                        inverse_spec_active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
                        inverse_spec_active_var_max_count=int(inverse_spec_active_var_max_count),
                        inverse_spec_directional_market_enable=bool(inverse_spec_directional_market_enable),
                        family_battery_enable=bool(inverse_spec_family_battery_enable),
                        family_battery_mode=str(inverse_spec_family_battery_mode or "outer"),
                        safe_eps=float(inverse_safe_eps),
                        confidence_mode=str(inverse_confidence_mode),
                        confidence_target_gain=float(inverse_confidence_target_gain),
                        confidence_floor=float(inverse_confidence_floor),
                    )
                    hole_search_stats["tournament_wall_seconds"] = float(
                        hole_search_stats.get("tournament_wall_seconds", 0.0)
                    ) + max(0.0, time.perf_counter() - tournament_t0)
                    hole_search_stats["tournament_elites"] = int(
                        hole_search_stats.get("tournament_elites", 0)
                    ) + int(len(elites))
                    hole_search_stats["tournament_rounds"] = int(
                        hole_search_stats.get("tournament_rounds", 0)
                    ) + 1
                    if elites:
                        prepared_opp = elites[0][0]
                    else:
                        prepared_opp = tournament_candidates[0]
                elif tournament_candidates:
                    prepared_opp = tournament_candidates[0]
                else:
                    prepared_opp = None
            else:
                prepared_opp = hole_frontier.select_spec_state(int(current_iter), rng, _is_executable)
            hole_search_stats["prepare_select_wall_seconds"] = float(
                hole_search_stats.get("prepare_select_wall_seconds", 0.0)
            ) + max(0.0, time.perf_counter() - select_t0)

            if dead_entries:
                for dead_frontier_key, source_bucket in dead_entries.items():
                    hole_search_stats["prepared_dead_pruned"] = int(
                        hole_search_stats.get("prepared_dead_pruned", 0)
                    ) + 1
                    hole_search_stats[f"prepared_dead_pruned_{source_bucket}"] = int(
                        hole_search_stats.get(f"prepared_dead_pruned_{source_bucket}", 0)
                    ) + 1
                    hole_frontier.drop_spec_state(frontier_key=dead_frontier_key)
                    if abstraction_stage is not None:
                        abstraction_stage.drop_spec_state(frontier_key=dead_frontier_key)
                hole_search_stats["frontier_size"] = int(len(hole_frontier))
                hole_search_stats["abstraction_stage_size"] = int(len(abstraction_stage)) if abstraction_stage is not None else 0
                _prune_parent_snapshots(int(current_iter))
            if prepared_opp is None:
                return None, None
            source_bucket = _hole_search_source_bucket(getattr(prepared_opp, "source", "other"))
            hole_search_stats["prepared_executable_available"] = int(
                hole_search_stats.get("prepared_executable_available", 0)
            ) + 1
            hole_search_stats[f"prepared_executable_available_{source_bucket}"] = int(
                hole_search_stats.get(f"prepared_executable_available_{source_bucket}", 0)
            ) + 1
            return prepared_opp, resolution_cache.get(prepared_opp.frontier_key)
        finally:
            hole_search_stats["prepare_wall_seconds"] = float(
                hole_search_stats.get("prepare_wall_seconds", 0.0)
            ) + max(0.0, time.perf_counter() - prepare_t0)

    def _hole_route_source(source: Any) -> tuple[str, str, str]:
        token = str(source or "").strip().lower()
        if token in ("archive_mine", "abstraction_improve", "abstraction_stall", "abstraction"):
            return "build", "build_action", "guided"
        if token == "repair_option":
            return "repair", "repair_beam", "guided"
        return "repair", "repair_beam", "inverse_beam"

    def _route_scheduler_adjust_reward(raw_reward: float, wall_s: float | None) -> float:
        try:
            reward_f = float(raw_reward)
        except Exception:
            reward_f = 0.0
        try:
            wall_f = max(float(hole_search_route_time_floor), float(wall_s or 0.0))
        except Exception:
            wall_f = float(hole_search_route_time_floor)
        if hole_search_route_reward_mode == "raw":
            return reward_f
        if hole_search_route_reward_mode == "per_second":
            return reward_f / wall_f
        return reward_f - float(hole_search_route_time_penalty) * wall_f

    def _record_route_scheduler_reward(route_name: str, raw_reward: float, wall_s: float | None) -> float | None:
        if route_scheduler is None or str(route_name or "") not in ("expression_expand", "opportunity_expand"):
            return None
        adjusted_reward = float(_route_scheduler_adjust_reward(raw_reward, wall_s))
        route_scheduler.update(
            route_name,
            adjusted_reward,
            raw_reward=float(raw_reward),
            wall_s=wall_s,
        )
        route_scheduler_stats["reward_count"] = int(route_scheduler_stats.get("reward_count", 0)) + 1
        route_scheduler_stats["reward_sum"] = float(route_scheduler_stats.get("reward_sum", 0.0)) + float(raw_reward)
        route_scheduler_stats["reward_sum_raw"] = float(
            route_scheduler_stats.get("reward_sum_raw", 0.0)
        ) + float(raw_reward)
        route_scheduler_stats["reward_sum_adjusted"] = float(
            route_scheduler_stats.get("reward_sum_adjusted", 0.0)
        ) + float(adjusted_reward)
        route_scheduler_stats["wall_seconds_sum"] = float(
            route_scheduler_stats.get("wall_seconds_sum", 0.0)
        ) + max(0.0, float(wall_s or 0.0))
        return adjusted_reward

    def _build_route_scheduler_diagnostic_state(
        *,
        available_routes: Sequence[str],
        route_scores: Mapping[str, Any] | None,
        route_score_details: Mapping[str, Mapping[str, Any]] | None,
        selected_route: str,
        selected_source: str,
    ) -> dict[str, Any]:
        score_map = {str(k): float(v) for k, v in dict(route_scores or {}).items()}
        detail_map = {str(k): dict(v or {}) for k, v in dict(route_score_details or {}).items()}
        route_rows: list[dict[str, Any]] = []
        best_route = ""
        best_score = float("-inf")
        chosen_score = None
        for idx, route_name in enumerate(str(route) for route in list(available_routes or []) if str(route)):
            detail = dict(detail_map.get(route_name, {}) or {})
            route_score = float(detail.get("route_score", score_map.get(route_name, 0.0)) or 0.0)
            preview_score = detail.get("preview_score", None)
            learned_bonus = detail.get("learned_bonus", None)
            preview_score_out = None
            learned_bonus_out = None
            try:
                if preview_score is not None and math.isfinite(float(preview_score)):
                    preview_score_out = float(preview_score)
            except Exception:
                preview_score_out = None
            try:
                if learned_bonus is not None and math.isfinite(float(learned_bonus)):
                    learned_bonus_out = float(learned_bonus)
            except Exception:
                learned_bonus_out = None
            route_row = {
                "rank": int(idx + 1),
                "route": str(route_name),
                "route_score": float(route_score),
                "preview_score": preview_score_out,
                "learned_bonus": learned_bonus_out,
                "selected": bool(route_name == str(selected_route or "")),
            }
            route_rows.append(route_row)
            if route_score > best_score:
                best_score = float(route_score)
                best_route = str(route_name)
            if route_name == str(selected_route or ""):
                chosen_score = float(route_score)
        return {
            "available_routes": [str(route) for route in list(available_routes or []) if str(route)],
            "route_rows": route_rows,
            "selected_route": str(selected_route or ""),
            "selection_source": str(selected_source or ""),
            "reward_mode": str(hole_search_route_reward_mode or ""),
            "time_penalty": float(hole_search_route_time_penalty),
            "time_floor": float(hole_search_route_time_floor),
            "best_available_route": str(best_route or ""),
            "best_available_route_score": None if not math.isfinite(best_score) else float(best_score),
            "chosen_route_score": chosen_score,
        }

    def _finalize_route_scheduler_diagnostics(
        entry,
        *,
        route_diag_state: Mapping[str, Any] | None,
        raw_reward: float,
        wall_s: float | None,
    ):
        if not bool(route_selected_this_iter) or not isinstance(route_diag_state, Mapping):
            return entry, None
        adjusted_reward = _record_route_scheduler_reward(
            str(route_diag_state.get("selected_route", "") or ""),
            float(raw_reward),
            wall_s,
        )
        row = dict(entry) if isinstance(entry, Mapping) else {}
        best_score = route_diag_state.get("best_available_route_score", None)
        chosen_score = route_diag_state.get("chosen_route_score", None)
        try:
            best_score_f = None if best_score is None else float(best_score)
        except Exception:
            best_score_f = None
        try:
            chosen_score_f = None if chosen_score is None else float(chosen_score)
        except Exception:
            chosen_score_f = None
        score_gap = None
        if best_score_f is not None and chosen_score_f is not None:
            score_gap = float(best_score_f - chosen_score_f)
        adjusted_regret_proxy = None
        if best_score_f is not None and adjusted_reward is not None:
            adjusted_regret_proxy = float(best_score_f - float(adjusted_reward))
        wall_out = None
        try:
            wall_out = None if wall_s is None else float(max(0.0, wall_s))
        except Exception:
            wall_out = None
        row.update({
            "route_scheduler_available_routes": list(route_diag_state.get("available_routes", []) or []),
            "route_scheduler_available_route_count": int(len(list(route_diag_state.get("available_routes", []) or []))),
            "route_scheduler_route_rows": [dict(item) for item in list(route_diag_state.get("route_rows", []) or []) if isinstance(item, Mapping)],
            "route_scheduler_selected_route": str(route_diag_state.get("selected_route", "") or ""),
            "route_scheduler_selection_source": str(route_diag_state.get("selection_source", "") or ""),
            "route_scheduler_reward_mode": str(route_diag_state.get("reward_mode", "") or ""),
            "route_scheduler_time_penalty": float(route_diag_state.get("time_penalty", 0.0) or 0.0),
            "route_scheduler_time_floor": float(route_diag_state.get("time_floor", 0.0) or 0.0),
            "route_scheduler_best_available_route": str(route_diag_state.get("best_available_route", "") or ""),
            "route_scheduler_best_available_route_score": best_score_f,
            "route_scheduler_chosen_route_score": chosen_score_f,
            "route_scheduler_selected_best_preview_route": bool(
                str(route_diag_state.get("selected_route", "") or "") == str(route_diag_state.get("best_available_route", "") or "")
            ),
            "route_scheduler_preview_gap": score_gap,
            "route_scheduler_realized_raw_reward": float(raw_reward),
            "route_scheduler_realized_adjusted_reward": None if adjusted_reward is None else float(adjusted_reward),
            "route_scheduler_wall_s": wall_out,
            "route_scheduler_adjusted_regret_proxy": adjusted_regret_proxy,
        })
        route_scheduler_stats["diagnostic_count"] = int(route_scheduler_stats.get("diagnostic_count", 0)) + 1
        route_scheduler_stats["selected_best_preview_count"] = int(
            route_scheduler_stats.get("selected_best_preview_count", 0)
        ) + int(bool(row.get("route_scheduler_selected_best_preview_route", False)))
        if score_gap is not None:
            route_scheduler_stats["preview_gap_sum"] = float(route_scheduler_stats.get("preview_gap_sum", 0.0)) + float(score_gap)
        if adjusted_regret_proxy is not None:
            route_scheduler_stats["adjusted_regret_proxy_sum"] = float(
                route_scheduler_stats.get("adjusted_regret_proxy_sum", 0.0)
            ) + float(adjusted_regret_proxy)
        return row, adjusted_reward

    def _record_hole_search_frontier_outcome(
        opp,
        *,
        current_iter: int,
        exact_eff_mse: float | None = None,
        shortlist_eff_mse: float | None = None,
        reward: float | None = None,
        wall_s: float | None = None,
        parent_eff_mse: float | None = None,
        accepted: bool | None = None,
        status: str = "ok",
    ) -> None:
        if hole_frontier is None or opp is None:
            return
        try:
            hole_frontier.record_spec_outcome(
                opp,
                current_iter=int(current_iter),
                exact_eff_mse=exact_eff_mse,
                shortlist_eff_mse=shortlist_eff_mse,
                reward=reward,
                wall_s=wall_s,
                parent_eff_mse=parent_eff_mse,
                accepted=accepted,
                status=str(status or "ok"),
            )
            hole_search_stats["exact_outcome_updates"] = int(
                hole_search_stats.get("exact_outcome_updates", 0)
            ) + 1
            hole_search_stats["frontier_size"] = int(len(hole_frontier))
        except Exception:
            pass

    def _hole_route_opportunity_score(opp, resolution) -> dict[str, Any]:
        if opp is None:
            return {
                "route_score": 0.0,
                "preview_score": 0.0,
                "learned_bonus": None,
            }
        try:
            preview_score = float(hole_frontier._score(opp)) if hole_frontier is not None else 0.0
        except Exception:
            preview_score = 0.0

        learned_bonus = None
        if isinstance(repair_opportunity_bundle, Mapping):
            route_source, opportunity_type, path_source = _hole_route_source(getattr(opp, "source", ""))
            parent_node = None
            parent_eff_mse = None
            resolution_source = str((resolution or {}).get("resolution_source", "") or "")
            if resolution_source == "live_archive":
                elite = (resolution or {}).get("elite", None)
                if elite is not None:
                    parent_node = getattr(elite, "expr", None)
                    try:
                        parent_eff_mse = float(getattr(elite, "mse", None))
                    except Exception:
                        parent_eff_mse = None
            elif resolution_source == "snapshot":
                snap = (resolution or {}).get("snapshot", None)
                if snap is not None:
                    parent_node = getattr(snap, "expr", None)
                    try:
                        parent_eff_mse = float(getattr(snap, "eff_mse", None))
                    except Exception:
                        parent_eff_mse = None
            try:
                route_row = {
                    "route_source": route_source,
                    "opportunity_type": opportunity_type,
                    "action": "repair_option" if route_source == "repair" else "replace",
                    "path": [int(v) for v in tuple(getattr(opp, "path", ()) or ())],
                    "path_source": path_source,
                    "target_mode": str(getattr(opp, "target_mode", "") or ""),
                    "parent_expr": str(getattr(opp, "parent_expr_str", "") or ""),
                    "parent_depth": int(node_depth(parent_node)) if parent_node is not None else 0,
                    "parent_eff_mse": parent_eff_mse,
                    "estimated_parent_eff_mse": parent_eff_mse,
                    "candidate_count_observed": int(getattr(opp, "candidate_count", 0) or 0),
                    "candidate_count_unique": int(max(
                        int(getattr(opp, "candidate_count", 0) or 0),
                        int(getattr(opp, "preview_candidate_count", 0) or 0),
                    )),
                    "preview_candidate_count_total": int(getattr(opp, "preview_candidate_count", 0) or 0),
                    "current_best_child_eff_mse": getattr(opp, "best_child_eff_mse", None),
                    "current_best_route_eff_mse": getattr(opp, "best_child_eff_mse", None),
                    "best_preview_probe_mse": (
                        getattr(opp, "preview_solvability", None)
                        if getattr(opp, "preview_solvability", None) is not None
                        else getattr(opp, "best_preview_probe_mse", None)
                    ),
                    "path_gain": float(getattr(opp, "path_gain", 0.0) or 0.0),
                    "transport_rel": float(getattr(opp, "transport_rel", 0.0) or 0.0),
                    "valid_frac": float(getattr(opp, "valid_frac", 0.0) or 0.0),
                    "confidence": float(getattr(opp, "confidence", 0.0) or 0.0),
                    "effective_n": float(getattr(opp, "effective_n", 0.0) or 0.0),
                    "budget_exact_spent": int(getattr(opp, "attempts", 0) or 0),
                    "budget_remaining": max(0, int(hole_search_exact_budget) - int(getattr(opp, "attempts", 0) or 0)),
                    "decision_id": f"hole_route::{str(getattr(opp, 'parent_key', ''))}",
                    "beam_id": f"hole::{str(getattr(opp, 'parent_key', ''))}::{str(getattr(opp, 'parent_elite_id', ''))}::{tuple(getattr(opp, 'path', ()) or ())}",
                    "opportunity_id": f"hole::{str(getattr(opp, 'parent_key', ''))}::{str(getattr(opp, 'parent_elite_id', ''))}::{tuple(getattr(opp, 'path', ()) or ())}::{str(getattr(opp, 'target_mode', '') or '')}",
                }
                route_pred = predict_opportunity_slate(repair_opportunity_bundle, route_row)
                route_scheduler_stats["model_scored"] = int(route_scheduler_stats.get("model_scored", 0)) + 1
                if bool(route_pred.get("trained", False)):
                    route_scheduler_stats["model_trained"] = int(route_scheduler_stats.get("model_trained", 0)) + 1
                    rows = list(route_pred.get("rows", []) or [])
                    if rows:
                        learned_bonus = float(rows[0].get("acquisition_estimate", 0.0) or 0.0)
            except Exception:
                learned_bonus = None

        route_score = float(preview_score)
        if learned_bonus is not None and math.isfinite(float(learned_bonus)):
            route_score += float(hole_search_route_acquisition_weight) * math.tanh(float(learned_bonus))
        return {
            "route_score": float(route_score),
            "preview_score": float(preview_score),
            "learned_bonus": None if learned_bonus is None else float(learned_bonus),
        }

    arch = ResidualBasinArchive()
    explorer = explorer_cls(active_actions, ucb_c=ucb_action, eps=eps_action)
    macro_controller = None
    if bool(macro_controller_enable):
        macro_controller = MacroController(
            [ACTION_NAME[a] for a in tracked_actions],
            ucb_c=ucb_action,
            eps=eps_action,
            repair_bonus=macro_controller_repair_bonus,
            repair_margin_scale=macro_controller_repair_margin_scale,
            build_bias=macro_controller_build_bias,
            inverse_bonus=macro_controller_inverse_bonus,
            learned_policy_weight=macro_controller_learned_policy_weight,
            learned_route_weight=macro_controller_learned_route_weight,
            learned_q_weight=macro_controller_learned_q_weight,
            learned_value_scale=macro_controller_learned_value_scale,
        )
    macro_controller_stats = {
        "enabled": bool(macro_controller_enable),
        "selected": 0,
        "repair_selected": 0,
        "fallback_selected": 0,
        "policy_counts": {},
        "decision_source_counts": {},
    }
    crossover_policies = ("legacy",)
    crossover_policy_stats = _init_crossover_policy_stats(crossover_policies)
    action_selected_counts = {a: 0 for a in tracked_actions}
    action_proposed_counts = {a: 0 for a in tracked_actions}
    action_reward_counts = {a: 0 for a in tracked_actions}
    action_accepted_counts = {a: 0 for a in tracked_actions}
    boost_gate_stats = {"considered": 0, "allowed": 0, "blocked_quality": 0, "blocked_sharp": 0,
                       "gain_frac_hist": [], "gain_frac_thr": None}
    inverse_gate_stats = {
        "considered": 0,
        "allowed": 0,
        "blocked_quality": 0,
        "blocked_structure": 0,
        "blocked_gain": 0,
        "best_rel_gain_hist": [],
        "best_weighted_rel_gain_hist": [],
        "min_weighted_rel_gain": float(inverse_gate_min_weighted_rel_gain),
        "confidence_mode": str(inverse_confidence_mode),
        "confidence_target_gain": float(inverse_confidence_target_gain),
        "confidence_floor": float(inverse_confidence_floor),
        "branch_beam_width": int(inverse_branch_beam_width),
        "micro_search_enable": bool(inverse_micro_search_enable),
        "local_score_mode": str(inverse_local_score_mode),
        "spec_enable": bool(inverse_spec_enable),
        "spec_enum_max_depth": int(inverse_spec_enum_max_depth),
        "spec_enum_max_trees": int(inverse_spec_enum_max_trees),
        "spec_preview_topk": int(inverse_spec_preview_topk),
        "spec_local_score_mode": str(inverse_spec_local_score_mode),
        "spec_include_legacy_seed": bool(inverse_spec_include_legacy_seed),
        "spec_complexity_penalty": float(inverse_spec_complexity_penalty),
        "spec_family_battery_enable": bool(inverse_spec_family_battery_enable),
        "spec_family_battery_mode": str(inverse_spec_family_battery_mode or "outer"),
        "spec_repair_quota": float(inverse_spec_repair_quota),
        "hole_search_enable": bool(hole_search_enable),
        "hole_search_first_class_scheduler_enable": bool(hole_search_first_class_scheduler_enable),
        "hole_search_quota": float(hole_search_quota),
        "hole_search_exact_budget": int(hole_search_exact_budget),
        "hole_search_cooldown_iters": int(hole_search_cooldown_iters),
        "hole_search_mine_cooldown_iters": int(hole_search_mine_cooldown_iters),
        "hole_search_max_frontier": int(hole_search_max_frontier),
        "hole_search_enum_max_depth": int(hole_search_enum_max_depth),
        "hole_search_enum_max_trees": int(hole_search_enum_max_trees),
        "hole_search_preview_topk": int(hole_search_preview_topk),
        "hole_search_solver_market_enable": bool(hole_search_solver_market_enable),
        "hole_search_solver_market_preview_topk": int(hole_search_solver_market_preview_topk),
        "hole_search_solver_market_exact_topk": int(hole_search_solver_market_exact_topk),
        "hole_search_solver_market_proposal_objects_enable": bool(
            hole_search_solver_market_proposal_objects_enable
        ),
        "inverse_spec_recursive_sr_enable": bool(inverse_spec_recursive_sr_enable),
        "inverse_spec_recursive_sr_preview_topk": int(inverse_spec_recursive_sr_preview_topk),
        "inverse_spec_recursive_sr_exact_budget": int(inverse_spec_recursive_sr_exact_budget),
        "inverse_spec_constant_lift_route_enable": bool(inverse_spec_constant_lift_route_enable),
        "inverse_spec_constant_lift_route_topk": int(inverse_spec_constant_lift_route_topk),
        "inverse_spec_coordinate_lift_enable": bool(inverse_spec_coordinate_lift_enable),
        "inverse_spec_coordinate_lift_topk": int(inverse_spec_coordinate_lift_topk),
        "inverse_spec_coordinate_lift_mode": str(inverse_spec_coordinate_lift_mode or "both"),
        "inverse_spec_tangent_edit_enable": bool(inverse_spec_tangent_edit_enable),
        "inverse_spec_tangent_edit_topk": int(inverse_spec_tangent_edit_topk),
        "inverse_spec_soft_edit_enable": bool(inverse_spec_soft_edit_enable),
        "inverse_spec_soft_edit_steps": int(inverse_spec_soft_edit_steps),
        "inverse_spec_soft_edit_l1": float(inverse_spec_soft_edit_l1),
        "inverse_spec_witness_jets_enable": bool(inverse_spec_witness_jets_enable),
        "inverse_spec_witness_d2_enable": bool(inverse_spec_witness_d2_enable),
        "inverse_spec_witness_max_rows": int(inverse_spec_witness_max_rows),
        "inverse_spec_witness_loss_enable": bool(inverse_spec_witness_loss_enable),
        "inverse_spec_witness_grad_weight": float(inverse_spec_witness_grad_weight),
        "inverse_spec_witness_d2_weight": float(inverse_spec_witness_d2_weight),
        "inverse_spec_witness_diag_weight": float(inverse_spec_witness_diag_weight),
        "inverse_spec_witness_physics_weight": float(inverse_spec_witness_physics_weight),
        "inverse_spec_active_var_screen_enable": bool(inverse_spec_active_var_screen_enable),
        "inverse_spec_active_var_grad_tol": float(inverse_spec_active_var_grad_tol),
        "inverse_spec_active_var_max_count": int(inverse_spec_active_var_max_count),
        "inverse_spec_directional_market_enable": bool(inverse_spec_directional_market_enable),
        "hole_search_tournament_enable": bool(hole_search_tournament_enable),
        "hole_search_tournament_n": int(hole_search_tournament_n),
        "hole_search_tournament_elite_k": int(hole_search_tournament_elite_k),
        "hole_search_tournament_preview_trees": int(hole_search_tournament_preview_trees),
        "spec_recursive_enable": bool(inverse_spec_recursive_enable),
        "spec_recursive_max_depth": int(inverse_spec_recursive_max_depth),
        "spec_recursive_trigger_rel_mse": float(inverse_spec_recursive_trigger_rel_mse),
        "spec_recursive_seed_cap": int(inverse_spec_recursive_seed_cap),
        "spec_recursive_branch_topk": int(inverse_spec_recursive_branch_topk),
        "spec_recursive_child_topk": int(inverse_spec_recursive_child_topk),
        "spec_max_subtree_depth": inverse_spec_max_subtree_depth,
        "spec_fit_cap": int(inverse_spec_fit_cap),
        "spec_probe_cap": int(inverse_spec_probe_cap),
        "spec_exact_budget": int(inverse_spec_exact_budget),
        "target_mode": str(inverse_target_mode),
        "full_mapping_penalty": float(inverse_full_mapping_penalty),
        "exact_simple_target_bonus": float(inverse_exact_simple_target_bonus),
        "additive_descend_penalty": float(inverse_additive_descend_penalty),
        "nonadditive_leaf_penalty": float(inverse_nonadditive_leaf_penalty),
        "exact_path_eta": float(inverse_exact_path_eta),
        "exact_transport_min_lin_rel": float(inverse_exact_transport_min_lin_rel),
        "periodic_min_valid_scale": float(inverse_periodic_min_valid_scale),
        "periodic_min_confidence_scale": float(inverse_periodic_min_confidence_scale),
        "periodic_path_penalty": float(inverse_periodic_path_penalty),
        "nonperiodic_muldiv_bonus": float(inverse_nonperiodic_muldiv_bonus),
        "nonperiodic_explogsqrt_bonus": float(inverse_nonperiodic_explogsqrt_bonus),
        "branch_ambiguity_penalty": float(inverse_branch_ambiguity_penalty),
        "transport_min_lin_rel": float(inverse_transport_min_lin_rel),
        "transport_min_effective_n": float(inverse_transport_min_effective_n),
    }
    repair_pass_stats = {
        "enabled": bool(repair_pass_enable),
        "selection_strategy": "mse_decade_size",
        "elite_k": max(0, int(repair_pass_elite_k)),
        "paths_per_elite": max(1, int(repair_pass_paths_per_elite)),
        "rounds": max(1, int(repair_pass_rounds)),
        "elites_selected": 0,
        "elites_considered": 0,
        "elites_improved": 0,
        "potential_calls": 0,
        "potential_allowed": 0,
        "rounds_attempted": 0,
        "paths_ranked": 0,
        "solver_calls": 0,
        "solver_ok": 0,
        "scored": 0,
        "accepted_repairs": 0,
        "new_residual_basins": 0,
        "global_best_updates": 0,
        "evals_used": 0,
        "skipped_empty_archive": 0,
        "skipped_no_paths": 0,
        "skipped_invalid_expr": 0,
        "skipped_score_none": 0,
        "stopped_wall_time": False,
        "status_counts": {},
    }
    closure_search_stats = {
        "enabled": bool(closure_search_enable),
        "families": [str(v) for v in list(closure_search_families or ()) if str(v or "").strip()],
        "max_scaffolds": max(0, int(closure_search_max_proposals)),
        "anchors_per_family": max(0, int(closure_search_anchors_per_family)),
        "preview_topk": max(1, int(closure_search_preview_topk)),
        "exact_topk": max(0, int(closure_search_exact_topk)),
        "beam_width": max(1, int(closure_search_beam_width)),
        "seed_exact_topk": max(0, int(closure_search_seed_exact_topk)),
        "seed_beam_width": max(1, int(closure_search_seed_beam_width)),
        "seed_scaffold_reserve": max(0, int(closure_search_seed_scaffold_reserve)),
        "seed_family_cap": max(0, int(closure_search_seed_family_cap)),
        "seed_exact_bound_bonus": float(closure_search_seed_exact_bound_bonus),
        "pair_normal_enable": bool(closure_search_pair_normal_enable),
        "pair_normal_topk": max(0, int(closure_search_pair_normal_topk)),
        "pair_normal_max_pairs": max(0, int(closure_search_pair_normal_max_pairs)),
        "pair_rescue_enable": bool(closure_search_pair_rescue_enable),
        "pair_rescue_topk": max(0, int(closure_search_pair_rescue_topk)),
        "pair_rescue_max_pairs": max(0, int(closure_search_pair_rescue_max_pairs)),
        "emergent_basis_enable": bool(closure_search_emergent_basis_enable),
        "emergent_basis_max_source_rows": max(0, int(closure_search_emergent_basis_max_source_rows)),
        "emergent_basis_score_topk": max(0, int(closure_search_emergent_basis_score_topk)),
        "emergent_basis_max_per_round": max(0, int(closure_search_emergent_basis_max_per_round)),
        "emergent_basis_max_total": max(0, int(closure_search_emergent_basis_max_total)),
        "emergent_basis_min_probe_gain_rel": float(closure_search_emergent_basis_min_probe_gain_rel),
        "emergent_aux_atoms_enable": bool(closure_search_emergent_aux_atoms_enable),
        "emergent_aux_atoms_max_source_rows": max(0, int(closure_search_emergent_aux_atoms_max_source_rows)),
        "emergent_aux_atoms_max_new_per_round": max(0, int(closure_search_emergent_aux_atoms_max_new_per_round)),
        "emergent_aux_atoms_max_total": max(0, int(closure_search_emergent_aux_atoms_max_total)),
        "emergent_aux_atoms_max_target": max(0, int(closure_search_emergent_aux_atoms_max_target)),
        "emergent_aux_atoms_max_dimensionless": max(0, int(closure_search_emergent_aux_atoms_max_dimensionless)),
        "emergent_aux_atoms_max_rational_derived": max(0, int(closure_search_emergent_aux_atoms_max_rational_derived)),
        "emergent_aux_atoms_max_seed_blocks": max(0, int(closure_search_emergent_aux_atoms_max_seed_blocks)),
        "debug_topk": max(0, int(closure_search_debug_topk)),
        "beam_min_valid_frac": float(closure_search_min_valid_frac),
        "beam_min_confidence": float(closure_search_min_confidence),
        "beam_periodic_min_valid_scale": float(closure_search_periodic_min_valid_scale),
        "beam_periodic_min_confidence_scale": float(closure_search_periodic_min_confidence_scale),
        "beam_transport_min_lin_rel": float(closure_search_transport_min_lin_rel),
        "anchor_head_compare_enable": bool(closure_search_anchor_head_compare_enable),
        "families_considered": 0,
        "scaffolds_enumerated": 0,
        "scaffolds_considered": 0,
        "preview_calls": 0,
        "preview_candidates": 0,
        "direct_calls": 0,
        "direct_candidates": 0,
        "direct_anchor_lift_attempts": 0,
        "direct_anchor_lift_applied": 0,
        "wall_time_budget_s": None,
        "wall_time_budget_fraction": None,
        "deadline_exceeded": False,
        "scored": 0,
        "new_residual_basins": 0,
        "global_best_updates": 0,
        "anchor_head_attempts": 0,
        "anchor_head_compare_attempts": 0,
        "anchor_head_compare_improved": 0,
        "anchor_head_compare_worsened": 0,
        "anchor_head_compare_neutral": 0,
        "anchor_head_compare_delta_sum": 0.0,
        "anchor_head_compare_examples": [],
        "evals_used": 0,
        "skipped_invalid_expr": 0,
        "skipped_score_none": 0,
        "status_counts": {},
        "failure_examples": [],
    }
    repair_controller_stats = {
        "enabled": bool(repair_controller_enable),
        "min_score": float(repair_controller_min_score),
        "adaptive_enable": bool(repair_controller_adaptive),
        "adapt_quantile": float(repair_controller_adapt_quantile),
        "adapt_window": int(repair_controller_adapt_window),
        "adapt_min_samples": int(repair_controller_adapt_min_samples),
        "min_concentration": float(repair_controller_min_concentration),
        "potential_weight": float(repair_controller_potential_weight),
        "concentration_weight": float(repair_controller_concentration_weight),
        "contrast_weight": float(repair_controller_contrast_weight),
        "cost_weight": float(repair_controller_cost_weight),
        "stagnation_weight": float(repair_controller_stagnation_weight),
        "steps": int(repair_controller_steps),
        "ancestor_hops": int(repair_controller_ancestor_hops),
        "min_step_rel_improve": float(repair_controller_min_step_rel_improve),
        "max_setup_steps": int(repair_controller_max_setup_steps),
        "setup_step_value_min": float(repair_controller_setup_step_value_min),
        "setup_step_regret_max": float(repair_controller_setup_step_regret_max),
        "setup_step_max_worsen": float(repair_controller_setup_step_max_worsen),
        "frontier_topk": int(repair_controller_frontier_topk),
        "stagnation_visits": int(repair_controller_stagnation_visits),
        "focus_prob": float(repair_controller_focus_prob),
        "parent_max_repeats": int(repair_controller_parent_max_repeats),
        "parent_min_eval_gap": int(repair_controller_parent_min_eval_gap),
        "parent_reset_rel_improve": float(repair_controller_parent_reset_rel_improve),
        "parent_preview_max_repeats": int(repair_controller_parent_preview_max_repeats),
        "policy_priority_weight": float(repair_controller_policy_priority_weight),
        "policy_priority_cap": float(repair_controller_policy_priority_cap),
        "critic_enable": bool(repair_controller_critic_enable),
        "critic_path": str(repair_controller_critic_path or ""),
        "critic_blend": float(repair_controller_critic_blend),
        "critic_mode": str(repair_controller_critic_mode),
        "route_compare_enable": bool(repair_controller_route_compare_enable),
        "route_compare_path": str(repair_controller_route_compare_path or ""),
        "route_compare_repair_tuple_path": str(repair_controller_route_compare_repair_tuple_path or ""),
        "route_compare_build_tuple_path": str(repair_controller_route_compare_build_tuple_path or ""),
        "route_compare_max_repair_prob": float(repair_controller_route_compare_max_repair_prob),
        "route_compare_min_build_margin": float(repair_controller_route_compare_min_build_margin),
        "credible_route_enable": bool(repair_controller_credible_route_enable),
        "opportunity_controller_enable": bool(repair_opportunity_controller_enable),
        "opportunity_controller_path": str(repair_opportunity_controller_path or ""),
        "opportunity_controller_witness_energy_feature_enable": False,
        "opportunity_controller_feature_schema_version": 1,
        "considered": 0,
        "selected": 0,
        "option_repair_selected": 0,
        "blocked_low_score": 0,
        "blocked_low_concentration": 0,
        "no_candidate": 0,
        "blocked_retry_cooldown": 0,
        "blocked_retry_repeat_budget": 0,
        "blocked_retry_repeat_signature": 0,
        "parent_considered": 0,
        "parent_fallback": 0,
        "parent_frontier_hits": 0,
        "parent_repair_selected": 0,
        "parent_retry_cooldown": 0,
        "parent_retry_repeat_budget": 0,
        "parent_retry_repeat_signature": 0,
        "parent_frontier_candidate_hist": [],
        "score_hist": [],
    }
    repair_critic_bundle = None
    repair_route_compare_bundle = None
    route_compare_repair_tuple_bundle = None
    build_tuple_bundle = None
    repair_opportunity_bundle = None
    scheduler_bundle = None
    scheduler_stats = {
        "enabled": bool(scheduler_enable),
        "advisory_only": bool(scheduler_advisory_only),
        "witness_energy_enable": bool(scheduler_witness_energy_enable),
        "witness_energy_feature_enable": False,
        "feature_schema_version": 1,
        "bundle_path": str(scheduler_bundle_path or ""),
        "bundle_loaded": False,
        "bundle_error": "",
        "decision_count": 0,
        "scored": 0,
        "control_selected": 0,
        "fallback_selected": 0,
        "candidate_count_sum": 0,
        "route_counts": {},
    }
    scheduler_decision_log: list[dict[str, Any]] = []
    scheduler_outcome_log: list[dict[str, Any]] = []
    if bool(repair_controller_enable) and bool(repair_controller_critic_enable):
        critic_path = str(repair_controller_critic_path or "").strip()
        if critic_path:
            try:
                repair_critic_bundle = load_repair_critic_bundle(critic_path)
                repair_controller_stats["critic_loaded"] = True
            except Exception as exc:
                repair_controller_stats["critic_loaded"] = False
                repair_controller_stats["critic_error"] = str(exc)
                _log.warning("Failed to load repair critic from %s: %s", critic_path, exc)
        else:
            repair_controller_stats["critic_loaded"] = False
    if bool(repair_controller_enable) and bool(repair_controller_route_compare_enable):
        route_compare_path = str(repair_controller_route_compare_path or "").strip()
        if route_compare_path:
            try:
                repair_route_compare_bundle = load_repair_critic_bundle(route_compare_path)
                repair_controller_stats["route_compare_loaded"] = True
            except Exception as exc:
                repair_controller_stats["route_compare_loaded"] = False
                repair_controller_stats["route_compare_error"] = str(exc)
                _log.warning("Failed to load repair route comparator from %s: %s", route_compare_path, exc)
        else:
            repair_controller_stats["route_compare_loaded"] = False
        route_compare_repair_tuple_path = str(repair_controller_route_compare_repair_tuple_path or "").strip()
        if route_compare_repair_tuple_path:
            try:
                route_compare_repair_tuple_bundle = load_repair_critic_bundle(route_compare_repair_tuple_path)
                repair_controller_stats["route_compare_repair_tuple_loaded"] = True
            except Exception as exc:
                repair_controller_stats["route_compare_repair_tuple_loaded"] = False
                repair_controller_stats["route_compare_repair_tuple_error"] = str(exc)
                _log.warning("Failed to load route-comparison repair tuple scorer from %s: %s", route_compare_repair_tuple_path, exc)
        else:
            repair_controller_stats["route_compare_repair_tuple_loaded"] = False
        build_tuple_path = str(repair_controller_route_compare_build_tuple_path or "").strip()
        if build_tuple_path:
            try:
                build_tuple_bundle = load_repair_critic_bundle(build_tuple_path)
                repair_controller_stats["route_compare_build_tuple_loaded"] = True
            except Exception as exc:
                repair_controller_stats["route_compare_build_tuple_loaded"] = False
                repair_controller_stats["route_compare_build_tuple_error"] = str(exc)
                _log.warning("Failed to load build tuple scorer from %s: %s", build_tuple_path, exc)
        else:
            repair_controller_stats["route_compare_build_tuple_loaded"] = False
    if (bool(inverse_steering_enable) or bool(repair_controller_enable)) and (bool(repair_opportunity_controller_enable) or bool(repair_controller_credible_route_enable)):
        opportunity_path = str(repair_opportunity_controller_path or "").strip()
        if opportunity_path:
            try:
                repair_opportunity_bundle = load_opportunity_bundle(opportunity_path)
                repair_controller_stats["opportunity_controller_loaded"] = True
                repair_controller_stats["opportunity_controller_witness_energy_feature_enable"] = bool(
                    repair_opportunity_bundle.get("witness_energy_feature_enable", False)
                )
                repair_controller_stats["opportunity_controller_feature_schema_version"] = int(
                    repair_opportunity_bundle.get("feature_schema_version", 1) or 1
                )
            except Exception as exc:
                repair_controller_stats["opportunity_controller_loaded"] = False
                repair_controller_stats["opportunity_controller_error"] = str(exc)
                _log.warning("Failed to load repair opportunity controller from %s: %s", opportunity_path, exc)
        else:
            repair_controller_stats["opportunity_controller_loaded"] = False
    if bool(scheduler_enable):
        bundle_path = str(scheduler_bundle_path or "").strip()
        if bundle_path:
            try:
                scheduler_bundle = load_scheduler_bundle(bundle_path)
                scheduler_stats["bundle_loaded"] = True
                scheduler_stats["witness_energy_feature_enable"] = bool(
                    scheduler_bundle.get("witness_energy_feature_enable", False)
                )
                scheduler_stats["feature_schema_version"] = int(
                    scheduler_bundle.get("feature_schema_version", 1) or 1
                )
            except Exception as exc:
                scheduler_bundle = None
                scheduler_stats["bundle_loaded"] = False
                scheduler_stats["bundle_error"] = str(exc)
                _log.warning("Failed to load scheduler bundle from %s: %s", bundle_path, exc)
        else:
            scheduler_stats["bundle_error"] = "bundle_path_missing"
    scheduler_control_enabled = bool(scheduler_enable) and (not bool(scheduler_advisory_only)) and isinstance(scheduler_bundle, Mapping)
    scheduler_middle_loop_control_enabled = bool(scheduler_control_enabled)
    legacy_middle_loop_fallback_only = bool(scheduler_middle_loop_control_enabled)
    if bool(legacy_middle_loop_fallback_only):
        repair_route_compare_bundle = None
        route_compare_repair_tuple_bundle = None
        build_tuple_bundle = None
        repair_controller_stats["route_compare_runtime_disabled_by_scheduler"] = True
    else:
        repair_controller_stats["route_compare_runtime_disabled_by_scheduler"] = False
    hole_search_first_class_runtime_enable = bool(hole_search_first_class_scheduler_enable) and (not bool(scheduler_control_enabled))
    inverse_gate_cache = {}
    controller_refine_cache: dict[str, dict[str, Any]] = {}
    repair_parent_cache: dict[Any, dict[str, Any]] = {}
    repair_parent_state: dict[Any, dict[str, Any]] = {}
    inverse_experiment_log = [] if bool(inverse_experiment_log_enable) else None
    lineage_events = [] if bool(inverse_experiment_log_enable) else None
    def _append_inverse_experiment(entry, wall_t0=None, **updates):
        if inverse_experiment_log is None or entry is None:
            return None
        row = dict(entry)
        row.update(updates)
        if wall_t0 is not None and "wall_s" not in row:
            try:
                row["wall_s"] = float(max(0.0, time.perf_counter() - wall_t0))
            except Exception:
                row["wall_s"] = None
        if row.get("observed_wall_seconds", None) is None:
            wall_value = row.get("wall_s", None)
            try:
                row["observed_wall_seconds"] = None if wall_value is None else float(max(0.0, wall_value))
            except Exception:
                row["observed_wall_seconds"] = None
        action_name = str(row.get("macro_action", "") or "")
        scored_status = str(row.get("status", "") or "")
        build_actions = {"replace", "wrap_un", "add_rand", "mul_rand", "residual", "boost", "prune", "crossover"}
        if "observed_exact_evals" not in row:
            if action_name in build_actions and scored_status in {"scored", "scored_no_reward"}:
                row["observed_exact_evals"] = 1
            elif action_name == "hole_search":
                row["observed_exact_evals"] = int(row.get("hole_search_exact_scored", 0) or 0)
            else:
                row["observed_exact_evals"] = int(row.get("inverse_exact_score_observed_count", 0) or 0)
        if "observed_preview_evals" not in row:
            if action_name in build_actions:
                row["observed_preview_evals"] = int(row.get("controller_build_slate_count", 0) or 0)
            elif action_name == "hole_search":
                row["observed_preview_evals"] = int(row.get("hole_search_preview_count", 0) or 0)
            else:
                row["observed_preview_evals"] = int(row.get("inverse_repair_slate_count", 0) or 0)
        row.setdefault("observed_micro_tokens", 0)
        row.setdefault("observed_widen_tokens", 0)
        row_idx = int(len(inverse_experiment_log))
        inverse_experiment_log.append(row)
        return row_idx
    def _make_inverse_experiment_record(
        parent_rec,
        gate_diag: InverseSteeringPotential | None,
        *,
        stagnation_state: dict[str, float] | None = None,
        candidate_meta: dict[str, Any] | None = None,
        proxy_potential: float | None = None,
    ) -> RepairControllerFeatureRecord:
        stag = stagnation_state if isinstance(stagnation_state, dict) else {}
        expr = parent_rec.best_expr
        return build_controller_state_record(
            parent_expr=node_str(expr),
            parent_root_op=(str(expr[0]) if isinstance(expr, tuple) and expr else ""),
            parent_depth=int(node_depth(expr)),
            parent_size=int(node_size(expr)),
            parent_best_eff_mse=float(getattr(parent_rec, "best_mse", float("inf"))),
            parent_best_raw_mse=float(getattr(parent_rec, "best_raw_mse", getattr(parent_rec, "best_mse", float("inf")))),
            parent_visits=float(stag.get("visits", 0.0)),
            parent_visits_since_improve=float(stag.get("visits_since_improve", 0.0)),
            parent_stagnation_score=float(stag.get("stagnation_score", 0.0)),
            parent_stagnation_ratio=float(stag.get("stagnation_ratio", 0.0)),
            gate_diag=gate_diag,
            candidate_meta=candidate_meta,
            proxy_potential=proxy_potential,
            refine_features=_controller_refine_features(expr),
        )
    refine_cfg = {
        "refine_profile": str(refine_profile or ""),
        "refine_mode": str(refine_mode_norm),
        "refine_enable_requested": bool(refine_enable_requested),
        "refine_inline_enable": bool(refine_inline_enable),
        "refine_during_brute": bool(refine_during_brute),
        "refine_during_mutation": bool(refine_during_mutation),
        "refine_during_controller_slate": bool(refine_during_controller_slate),
        "refine_during_slate": bool(refine_during_slate),
        "scheduled_slate_refine_enable": bool(scheduled_slate_refine_enable),
        "refine_slate_after_brute": bool(refine_slate_after_brute),
        "refine_slate_period": int(refine_slate_period_i),
        "refine_final_polish": bool(refine_final_polish),
        "refine_slate_k": int(refine_slate_k_i),
        "refine_slate_diverse_k": int(refine_slate_diverse_k_i),
        "refine_slate_budget": int(refine_slate_budget_i),
        "optimizer": str(refine_optimizer),
        "lbfgs_escalate_improve_factor": float(refine_lbfgs_escalate_improve_factor),
        "lbfgs_steps": int(refine_lbfgs_steps),
        "fit_subset": int(refine_fit_subset),
        "fit_subset_mode": str(refine_fit_subset_mode),
        "num_restarts": int(refine_num_restarts),
        "max_variants": int(refine_max_variants),
        "max_params": int(refine_max_params),
        "linear_combo_enable": bool(refine_linear_combo_enable),
        "linear_terms_max": int(refine_linear_terms_max),
        "linear_prune_rel": float(refine_linear_prune_rel),
        "linear_ridge": float(refine_linear_ridge),
        "slot_sensitivity_enable": bool(refine_slot_sensitivity_enable),
        "slot_sensitivity_subset": int(refine_slot_sensitivity_subset),
        "slot_sensitivity_delta": float(refine_slot_sensitivity_delta),
        "slot_sensitivity_max_paths": int(refine_slot_sensitivity_max_paths),
        "prune_mapping_equiv_root_slots": bool(refine_prune_mapping_equiv_root_slots),
        "attempt_cache_enable": bool(refine_attempt_cache_enable),
        "attempt_cache_max_entries": int(refine_attempt_cache_max_entries),
        "diagnostics": refine_diagnostics,
        "attempt_cache": refine_attempt_cache,
        "gate_best_factor": float(refine_gate_best_factor),
        "gate_potential_enable": bool(refine_gate_potential_enable),
        "gate_potential_subset": int(refine_gate_potential_subset),
        "gate_potential_improve_factor": float(refine_gate_potential_improve_factor),
        "gate_log_min": float(refine_gate_log_min),
        "gate_log_max": float(refine_gate_log_max),
        "gate_grid_size": int(refine_gate_grid_size),
        "gate_max_evals": int(refine_gate_max_evals),
        "max_refines": int(refine_max_trials),
        "trials_per_brute_depth": int(refine_trials_per_brute_depth),
        "trials_per_mutation_window": int(refine_trials_per_mutation_window),
        "mutation_window": int(refine_mutation_window),
        "safe_eps": float(refine_safe_eps),
        "safe_penalty_weight": float(refine_safe_penalty_weight),
        "safe_exp_clip": float(refine_safe_exp_clip),
        "theta_l2": float(refine_theta_l2),
        "init_log_min": float(refine_init_log_min),
        "init_log_max": float(refine_init_log_max),
        "refine_grid_enable": bool(refine_grid_enable),
        "refine_grid_size": int(refine_grid_size),
        "refine_grid_size_2d": int(refine_grid_size_2d),
        "refine_grid_passes": int(refine_grid_passes),
        "refine_grid_topk": int(refine_grid_topk),
        "refine_grid_max_evals": int(refine_grid_max_evals),
        "stall_gate_relax_factor": float(refine_stall_gate_relax_factor),
        "stall_gate_relax_max": float(refine_stall_gate_relax_max),
        "var_dims": var_dims,
        "verbose": bool(verbose),
        "_legacy_refinement_hooks": runtime_hooks,
    }

    # Scoring augmentation config: multi-term linear head on residual.
    # Normally this is unit-gated. DE first-line operator-factorized DE runs may opt into
    # untyped heads for synthetic suites without dimensional metadata.
    score_head_enable_eff = bool(score_head_enable) and (
        (bool(dm) and (y_dims is not None)) or bool(score_head_untyped_enable)
    )
    refine_cfg.update({
        "max_depth": int(max_depth),
        "score_head_enable": bool(score_head_enable_eff),
        "score_head_vars_enable": bool(score_head_vars_enable),
        "score_head_omp_enable": bool(score_head_omp_enable),
        "score_head_omp_max_terms": int(score_head_omp_max_terms),
        "score_head_omp_topk_try": int(score_head_omp_topk_try),
        "score_head_ridge": score_head_ridge,
        "score_head_min_rel_improve": float(score_head_min_rel_improve),
        "score_head_untyped_enable": bool(score_head_untyped_enable),
        "score_mapping_family_mode": str(score_mapping_family_mode),
        "brute_score_mapping_family_mode": str(brute_score_mapping_family_mode),
        "score_pade_structural_enable": bool(score_pade_structural_enable),
        "score_pade_structural_max_degree": int(score_pade_structural_max_degree),
        "score_pade_structural_max_total_degree": int(score_pade_structural_max_total_degree),
        "score_pade_structural_max_depth": int(score_pade_structural_max_depth),
        "score_pade_structural_max_size": int(score_pade_structural_max_size),
        "score_pade_structural_coeff_tol": float(score_pade_structural_coeff_tol),
        "score_pade_structural_mse_rel_tol": float(score_pade_structural_mse_rel_tol),
        "score_mapping_expensive_gate_best_factor": float(score_mapping_expensive_gate_best_factor),
        "score_mapping_expensive_rel_y": float(score_mapping_expensive_rel_y),
        "score_prescreen_enable": bool(score_prescreen_enable),
        "score_prescreen_family_mode": str(score_prescreen_family_mode),
        "score_prescreen_residual_family_mode": str(score_prescreen_residual_family_mode),
        "score_prescreen_residual_allow_hint": bool(score_prescreen_residual_allow_hint),
        "score_prescreen_residual_use_global_best": bool(score_prescreen_residual_use_global_best),
        "score_prescreen_parent_best_factor": float(score_prescreen_parent_best_factor),
        "score_prescreen_global_best_factor": float(score_prescreen_global_best_factor),
        "score_prescreen_residual_parent_best_factor": float(score_prescreen_residual_parent_best_factor),
        "score_prescreen_residual_global_best_factor": float(score_prescreen_residual_global_best_factor),
        "score_finite_mask_enable": bool(score_finite_mask_enable),
        "score_finite_mask_min_fit_frac": float(score_finite_mask_min_fit_frac),
        "score_finite_mask_min_probe_frac": float(score_finite_mask_min_probe_frac),
        "score_finite_mask_min_dataset_frac": float(score_finite_mask_min_dataset_frac),
        "score_finite_mask_min_points": int(score_finite_mask_min_points),
        "score_domain_projection_enable": bool(score_domain_projection_enable),
        "score_domain_projection_abs_tol": float(score_domain_projection_abs_tol),
        "score_domain_projection_rel_tol": float(score_domain_projection_rel_tol),
        "score_domain_projection_max_frac": float(score_domain_projection_max_frac),
        "score_domain_projection_positive_floor": float(score_domain_projection_positive_floor),
    })

    # B1 baseline: raw variables that match dim(y), or all raw variables
    # when untyped DE heads are explicitly enabled.
    head_var_terms = []
    if score_head_enable_eff and bool(score_head_vars_enable):
        if bool(dm) and (y_dims is not None):
            for j, dj in enumerate(var_dims):
                if dims_eq(dj, y_dims):
                    head_var_terms.append(("var", int(j)))
        elif bool(score_head_untyped_enable):
            head_var_terms.extend(("var", int(j)) for j in range(int(nvars)))
    refine_cfg["score_head_var_terms"] = head_var_terms

    # B2 optional: pool-backed OMP (pre-filter by dim + non-degenerate norms;
    # untyped mode accepts all non-constant, finite-norm pool atoms).
    if score_head_enable_eff and bool(score_head_omp_enable):
        valid_mask = torch.zeros(len(pool_nodes), dtype=torch.bool, device=pool_norms_fit.device)
        for k, dk in enumerate(pool_dims):
            # Drop pure constants; bias is handled separately.
            if isinstance(pool_nodes[k], tuple) and len(pool_nodes[k]) >= 1 and pool_nodes[k][0] == "const":
                continue
            dim_ok = bool(score_head_untyped_enable) and not (bool(dm) and (y_dims is not None))
            if bool(dm) and (y_dims is not None) and (dk is not None):
                dim_ok = dims_eq(dk, y_dims)
            if dim_ok and float(pool_norms_fit[k]) > 1e-30:
                valid_mask[k] = True
        refine_cfg["score_head_pool_nodes"] = pool_nodes
        refine_cfg["score_head_pool_phi_fit"] = pool_phi_fit
        refine_cfg["score_head_pool_phi_probe"] = pool_phi
        refine_cfg["score_head_pool_norms_fit"] = pool_norms_fit
        refine_cfg["score_head_pool_valid_mask"] = valid_mask
        refine_cfg["score_head_pool_node_to_idx"] = {n: i for i, n in enumerate(pool_nodes)}

    # Attach optional joint multi-dataset data. Each entry may be (x, y) or (id, x, y).
    def _parse_joint_rows(rows):
        out_rows = []
        if rows is None:
            return out_rows
        for i, row in enumerate(rows):
            if row is None:
                continue
            did = None
            if isinstance(row, (tuple, list)) and len(row) == 3:
                did, xj, yj = row[0], row[1], row[2]
                if did is None:
                    did = str(i)
            elif isinstance(row, (tuple, list)) and len(row) == 2:
                did, xj, yj = str(i), row[0], row[1]
            else:
                continue
            if torch.is_tensor(xj) and torch.is_tensor(yj):
                xj = xj.to(dtype=dtype)
                yj = yj.to(dtype=dtype)
                if yj.dim() == 1:
                    yj = yj.unsqueeze(-1)
                out_rows.append((str(did), xj, yj))
        return out_rows

    try:
        jd = _parse_joint_rows(refine_joint_fit_data)
        pd = _parse_joint_rows(refine_joint_probe_data)
        if len(jd) >= 2:
            refine_cfg["joint_fit_data"] = jd
            refine_cfg["joint_weight_mode"] = str(refine_joint_weight_mode)
            refine_cfg["joint_refine_enable"] = bool(refine_joint_enable)
        if len(pd) >= 2:
            refine_cfg["joint_probe_data"] = pd
        # Only enable joint scoring if we have both fit+probe datasets.
        if len(jd) >= 2 and len(pd) >= 2:
            refine_cfg["joint_score_enable"] = bool(refine_joint_score_enable)
            refine_cfg["joint_terms_enable"] = bool(refine_joint_terms_enable)
    except Exception:
        # Never break the core search loop if joint data is malformed.
        pass
    refine_state = {"trials_done": 0, "gate_relax_factor": 1.0, "stall_count": 0}

    def _controller_refine_features(node) -> dict[str, Any]:
        expr_key = node_str(node)
        cached = controller_refine_cache.get(expr_key, None)
        if isinstance(cached, dict):
            return dict(cached)
        out: dict[str, Any] = {
            "refine_slot_count": 0,
            "refine_gate_potential": 0.0,
            "refine_variant_count": 0,
        }
        want_refine_signals = bool(refine_active) and (
            bool(inverse_experiment_log_enable)
            or bool(repair_controller_enable)
            or bool(macro_controller_enable)
            or repair_critic_bundle is not None
        )
        if not want_refine_signals or not torch.is_tensor(x_fit) or not torch.is_tensor(y_fit):
            controller_refine_cache[expr_key] = dict(out)
            return dict(out)
        try:
            refine_ctl_cfg = dict(refine_cfg)
            refine_ctl_cfg["slot_sensitivity_enable"] = False
            refine_ctl_cfg["joint_refine_enable"] = False
            refine_ctl_cfg["gate_potential_subset"] = max(8, min(int(refine_ctl_cfg.get("gate_potential_subset", 64)), 32))
            refine_ctl_cfg["gate_max_evals"] = max(4, min(int(refine_ctl_cfg.get("gate_max_evals", 64)), 16))
            max_variants = max(1, min(int(refine_ctl_cfg.get("max_variants", 4)), 4))
            max_params = max(1, min(int(refine_ctl_cfg.get("max_params", 2)), 2))
            variants = _decorate_refine_variants(
                node,
                max_variants,
                max_params,
                x_fit=x_fit,
                y_fit=y_fit,
                cfg=refine_ctl_cfg,
            )
            out["refine_variant_count"] = int(len(variants))
            if variants:
                out["refine_slot_count"] = int(max(int(n_params) for _, n_params, _ in variants))
                gate_hits = 0
                for var_h, n_params, _shift_slots in variants:
                    if int(n_params) <= 0:
                        gate_hits += 1
                    elif _variant_has_gate_potential(var_h, n_params, x_fit, y_fit, refine_ctl_cfg):
                        gate_hits += 1
                out["refine_gate_potential"] = float(gate_hits / max(1, len(variants)))
        except Exception as exc:
            out["refine_error"] = str(exc)
        controller_refine_cache[expr_key] = dict(out)
        if len(controller_refine_cache) > 2048:
            try:
                controller_refine_cache.pop(next(iter(controller_refine_cache)))
            except Exception:
                controller_refine_cache.clear()
        return dict(out)

    def _make_inverse_experiment_row(
        parent_rec,
        gate_diag: InverseSteeringPotential | None,
        *,
        stagnation_state: dict[str, float] | None = None,
        candidate_meta: dict[str, Any] | None = None,
        proxy_potential: float | None = None,
    ) -> dict[str, Any]:
        return _make_inverse_experiment_record(
            parent_rec,
            gate_diag,
            stagnation_state=stagnation_state,
            candidate_meta=candidate_meta,
            proxy_potential=proxy_potential,
        ).to_flat_dict()

    def _ensure_action_experiment_entry(
        entry,
        wall_t0,
        parent_rec,
        gate_diag: InverseSteeringPotential | None,
        *,
        action: int,
        source: str | None = None,
        macro_decision: Any = None,
        controller_record: RepairControllerFeatureRecord | Mapping[str, Any] | None = None,
        stagnation_state: dict[str, float] | None = None,
        candidate_meta: dict[str, Any] | None = None,
        proxy_potential: float | None = None,
    ) -> tuple[dict[str, Any] | None, float | None]:
        if inverse_experiment_log is None:
            return entry, wall_t0
        row = dict(entry) if isinstance(entry, Mapping) else None
        if row is None:
            if isinstance(controller_record, RepairControllerFeatureRecord):
                row = controller_record.to_flat_dict()
            elif isinstance(controller_record, Mapping):
                row = dict(controller_record)
            else:
                row = _make_inverse_experiment_row(
                    parent_rec,
                    gate_diag,
                    stagnation_state=stagnation_state,
                    candidate_meta=candidate_meta,
                    proxy_potential=proxy_potential,
                )
        if wall_t0 is None:
            try:
                wall_t0 = time.perf_counter()
            except Exception:
                wall_t0 = None
        action_id = int(action)
        action_name = str(ACTION_NAME.get(action_id, f"action_{action_id}"))
        action_source = str(source).strip() if source is not None else str(row.get("macro_action_source", "") or "").strip()
        if action_source:
            row.update(_macro_action_fields(action_id, source=action_source))
        else:
            row.update(_macro_action_fields(action_id))
        row["controller_policy_action"] = action_name
        if macro_decision is not None:
            row.update(_macro_decision_log_fields(macro_decision))
        return row, wall_t0

    def _scheduler_decision_fields(decision) -> dict[str, Any]:
        if decision is None:
            return {}
        chosen = getattr(decision, "chosen_candidate", None)
        rows = list(getattr(decision, "rows", ()) or [])
        top_row = rows[0] if rows and isinstance(rows[0], Mapping) else {}
        decision_context_id = (
            str(getattr(chosen, "decision_id", "") or "")
            or str(top_row.get("decision_context_id", "") or top_row.get("decision_id", "") or "")
        )
        return {
            "scheduler_enabled": bool(scheduler_stats.get("enabled", False)),
            "scheduler_advisory_only": bool(getattr(decision, "advisory_only", False)),
            "scheduler_candidate_count": int(getattr(decision, "candidate_count", 0) or 0),
            "scheduler_trained": bool(getattr(decision, "trained", False)),
            "scheduler_fallback_used": bool(getattr(decision, "fallback_used", False)),
            "scheduler_fallback_reason": str(getattr(decision, "fallback_reason", "") or ""),
            "scheduler_confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
            "scheduler_confidence_kind": str(getattr(decision, "confidence_kind", "dominance_gap") or "dominance_gap"),
            "scheduler_dominance_prob": float(getattr(decision, "dominance_prob", 0.0) or 0.0),
            "scheduler_acquisition_gap": float(getattr(decision, "acquisition_gap", 0.0) or 0.0),
            "scheduler_acquisition_gap_sigma": float(getattr(decision, "acquisition_gap_sigma", 0.0) or 0.0),
            "scheduler_decision_context_id": str(decision_context_id),
            "scheduler_chosen_route": str(getattr(decision, "chosen_route", "") or ""),
            "scheduler_chosen_opportunity_id": str(getattr(decision, "chosen_opportunity_id", "") or ""),
            "scheduler_chosen_exact_budget": int(getattr(decision, "chosen_exact_budget", 0) or 0),
            "scheduler_runner_up_route": str(getattr(decision, "runner_up_route", "") or ""),
            "scheduler_runner_up_opportunity_id": str(getattr(decision, "runner_up_opportunity_id", "") or ""),
            "scheduler_runner_up_exact_budget": int(getattr(decision, "runner_up_exact_budget", 0) or 0),
            "scheduler_acquisition_threshold": float(getattr(decision, "acquisition_threshold", 0.25) or 0.25),
            "scheduler_uncertainty_bonus": float(getattr(decision, "uncertainty_bonus", 0.05) or 0.05),
            "scheduler_chosen_action": "" if chosen is None else str(getattr(chosen, "action", "") or ""),
            "scheduler_chosen_path": [] if chosen is None else [int(v) for v in tuple(getattr(chosen, "path", ()) or ())],
            "scheduler_chosen_target_mode": "" if chosen is None else str(getattr(chosen, "target_mode", "") or ""),
            "scheduler_route_scores": {
                str(k): float(v)
                for k, v in dict(getattr(decision, "route_scores", {}) or {}).items()
            },
        }

    def _scheduler_candidate_log_row(row: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            return {}
        out = {
            "decision_context_id": str(
                row.get("scheduler_decision_context_id", "")
                or row.get("decision_context_id", "")
                or row.get("decision_id", "")
                or ""
            ),
            "scheduler_decision_context_id": str(
                row.get("scheduler_decision_context_id", "")
                or row.get("decision_context_id", "")
                or row.get("decision_id", "")
                or ""
            ),
            "route_decision_id": str(row.get("route_decision_id", row.get("decision_id", "")) or ""),
            "route_decision_context_id": str(
                row.get("route_decision_context_id", row.get("decision_context_id", row.get("decision_id", ""))) or ""
            ),
            "route": str(row.get("plan_route", row.get("route_source", "")) or ""),
            "opportunity_id": str(row.get("opportunity_id", "") or ""),
            "exact_budget": int(row.get("plan_exact_budget", 0) or 0),
            "budget_exact_spent": int(row.get("budget_exact_spent", 0) or 0),
            "budget_remaining": int(row.get("budget_remaining", 0) or 0),
            "action": str(row.get("plan_action", row.get("action", "")) or ""),
            "path": [int(v) for v in list(row.get("plan_path", row.get("path", [])) or [])],
            "target_mode": str(row.get("plan_target_mode", row.get("target_mode", "")) or ""),
            "acquisition": float(row.get("plan_acquisition_estimate", 0.0) or 0.0),
            "acquisition_with_uncertainty": float(row.get("plan_acquisition_with_uncertainty", 0.0) or 0.0),
            "sigma": float(row.get("plan_acquisition_sigma", 0.0) or 0.0),
            "confidence": float(row.get("plan_confidence", 0.0) or 0.0),
            "breakthrough_prob": float(row.get("plan_breakthrough_prob", row.get("plan_confidence", 0.0)) or 0.0),
            "prediction_components": dict(row.get("plan_prediction_components", {}) or {}),
        }
        if bool(scheduler_witness_energy_enable):
            witness = (
                dict(out["prediction_components"].get("witness_energy", {}) or {})
                if isinstance(out.get("prediction_components", None), Mapping)
                else {}
            )
            out.update({
                "witness_value_loss": row.get("plan_witness_value_loss", witness.get("value_loss", None)),
                "witness_grad_loss": row.get("plan_witness_grad_loss", witness.get("grad_loss", None)),
                "witness_d2_loss": row.get("plan_witness_d2_loss", witness.get("d2_loss", None)),
                "witness_diag_loss": row.get("plan_witness_diag_loss", witness.get("diag_loss", None)),
                "witness_physics_loss": row.get("plan_witness_physics_loss", witness.get("physics_loss", None)),
                "witness_energy_total": row.get("plan_witness_energy_total", witness.get("total", None)),
                "witness_energy_delta_estimate": row.get(
                    "plan_witness_energy_delta_estimate",
                    witness.get("delta_estimate", None),
                ),
            })
        return out

    def _selected_scheduler_prediction_row(decision) -> dict[str, Any]:
        rows = [row for row in list(getattr(decision, "rows", ()) or []) if isinstance(row, Mapping)]
        chosen = getattr(decision, "chosen_candidate", None)
        if chosen is not None:
            chosen_route = str(getattr(chosen, "route", "") or "")
            chosen_opportunity_id = str(getattr(chosen, "opportunity_id", "") or "")
            chosen_budget = int(getattr(chosen, "exact_budget", 0) or 0)
            for row in rows:
                route = str(row.get("plan_route", row.get("route_source", "")) or "")
                opportunity_id = str(row.get("opportunity_id", "") or "")
                exact_budget = int(row.get("plan_exact_budget", row.get("exact_budget", 0)) or 0)
                if (
                    route == chosen_route
                    and opportunity_id == chosen_opportunity_id
                    and exact_budget == chosen_budget
                ):
                    return dict(row)
        return dict(rows[0]) if rows else {}

    def _safe_bool_or_none(value: Any) -> bool | None:
        if value is None:
            return None
        return bool(value)

    def _first_present_value(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _realized_log_gain(before_value: Any, after_value: Any, *, executed: bool) -> float | None:
        if not bool(executed):
            return None
        try:
            before_f = float(before_value)
            after_f = float(after_value)
        except Exception:
            return 0.0
        if (not math.isfinite(before_f)) or (not math.isfinite(after_f)):
            return 0.0
        try:
            return float(max(0.0, math.log(before_f + 1.0e-30) - math.log(after_f + 1.0e-30)))
        except Exception:
            return 0.0

    def _realized_witness_transition_fields(
        before_row: Mapping[str, Any] | None,
        after_row: Mapping[str, Any] | None,
        *,
        executed: bool,
    ) -> dict[str, Any]:
        before = normalize_witness_energy_fields(before_row if isinstance(before_row, Mapping) else {})
        after = normalize_witness_energy_fields(after_row if bool(executed) and isinstance(after_row, Mapping) else {})
        out = {
            "realized_witness_value_loss_before": before.get("witness_value_loss", None),
            "realized_witness_grad_loss_before": before.get("witness_grad_loss", None),
            "realized_witness_d2_loss_before": before.get("witness_d2_loss", None),
            "realized_witness_diag_loss_before": before.get("witness_diag_loss", None),
            "realized_witness_physics_loss_before": before.get("witness_physics_loss", None),
            "realized_witness_energy_total_before": before.get("witness_energy_total", None),
            "realized_witness_value_loss_after": after.get("witness_value_loss", None),
            "realized_witness_grad_loss_after": after.get("witness_grad_loss", None),
            "realized_witness_d2_loss_after": after.get("witness_d2_loss", None),
            "realized_witness_diag_loss_after": after.get("witness_diag_loss", None),
            "realized_witness_physics_loss_after": after.get("witness_physics_loss", None),
            "realized_witness_energy_total_after": after.get("witness_energy_total", None),
        }
        return normalize_realized_witness_energy_fields(out)

    def _action_route_name(action_id: int | None) -> str:
        if action_id is None:
            return ""
        build_actions = {A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_RESIDUAL, A_BOOST, A_PRUNE, A_CROSSOVER}
        if int(action_id) in build_actions:
            return "build"
        if int(action_id) == int(A_REPAIR):
            return "repair"
        if int(action_id) == int(A_HOLESEARCH):
            return "hole"
        return ""

    def _record_scheduler_outcome(
        *,
        decision,
        decision_log_index: int | None,
        current_iter: int,
        applied: bool,
        selected_action: int | None,
        inverse_row_index: int | None = None,
        inverse_row: Mapping[str, Any] | None = None,
        local_parent_eff_mse: float | None = None,
        child_eff_mse: float | None = None,
        global_best_eff_mse_before: float | None = None,
        status: str = "",
    ) -> int | None:
        if decision is None:
            return None
        chosen_row = _selected_scheduler_prediction_row(decision)
        chosen = getattr(decision, "chosen_candidate", None)
        realized_row = dict(inverse_row) if isinstance(inverse_row, Mapping) else {}
        executed = bool(applied) and chosen is not None
        fragility_value = _first_present_value(
            realized_row.get("oracle_mapping_fragile", None),
            realized_row.get("hole_mapping_fragile", None),
            chosen_row.get("oracle_mapping_fragile", None),
            chosen_row.get("hole_mapping_fragile", None),
        )
        stability_value = _first_present_value(
            realized_row.get("oracle_mapping_stable", None),
            realized_row.get("hole_mapping_stable", None),
            chosen_row.get("oracle_mapping_stable", None),
            chosen_row.get("hole_mapping_stable", None),
        )
        outcome = {
            "iter": int(current_iter),
            "decision_log_index": None if decision_log_index is None else int(decision_log_index),
            "decision_context_id": str(
                ("" if chosen is None else str(getattr(chosen, "decision_id", "") or ""))
                or chosen_row.get("decision_context_id", "")
                or chosen_row.get("decision_id", "")
                or ""
            ),
            "route": str(getattr(decision, "chosen_route", "") or ""),
            "opportunity_id": str(getattr(decision, "chosen_opportunity_id", "") or ""),
            "exact_budget": int(getattr(decision, "chosen_exact_budget", 0) or 0),
            "budget_exact_spent": int(chosen_row.get("budget_exact_spent", 0) or 0),
            "budget_remaining": int(chosen_row.get("budget_remaining", 0) or 0),
            "scheduler_applied": bool(applied),
            "executed": bool(executed),
            "advisory_only": bool(getattr(decision, "advisory_only", False)),
            "fallback_used": bool(getattr(decision, "fallback_used", False)),
            "status": str(status or realized_row.get("status", "") or ("not_applied" if not applied else "")),
            "selected_action": None if selected_action is None else str(ACTION_NAME.get(int(selected_action), f"action_{int(selected_action)}")),
            "selected_route": str(_action_route_name(selected_action)),
            "inverse_experiment_row_index": None if inverse_row_index is None else int(inverse_row_index),
            "realized_local_delta_log_eff": _realized_log_gain(local_parent_eff_mse, child_eff_mse, executed=bool(executed)),
            "realized_global_delta_log_eff": _realized_log_gain(global_best_eff_mse_before, child_eff_mse, executed=bool(executed)),
            "realized_wall_seconds": _first_present_value(
                realized_row.get("observed_wall_seconds", None),
                realized_row.get("wall_s", None),
            ),
            "realized_exact_evals": None
            if _first_present_value(realized_row.get("observed_exact_evals", None), None) is None
            else int(realized_row.get("observed_exact_evals", 0) or 0),
            "realized_preview_evals": None
            if _first_present_value(realized_row.get("observed_preview_evals", None), None) is None
            else int(realized_row.get("observed_preview_evals", 0) or 0),
            "realized_micro_tokens": None
            if _first_present_value(realized_row.get("observed_micro_tokens", None), None) is None
            else int(realized_row.get("observed_micro_tokens", 0) or 0),
            "realized_widen_tokens": None
            if _first_present_value(realized_row.get("observed_widen_tokens", None), None) is None
            else int(realized_row.get("observed_widen_tokens", 0) or 0),
            "realized_new_residual_basin": _safe_bool_or_none(
                _first_present_value(
                    realized_row.get("created_new_residual_basin", None),
                    realized_row.get("new_residual_basin", None),
                    realized_row.get("oracle_new_residual_basin", None),
                )
            ),
            "realized_fragility": _safe_bool_or_none(fragility_value),
            "realized_stability": _safe_bool_or_none(stability_value),
        }
        outcome.update(
            _realized_witness_transition_fields(
                chosen_row,
                realized_row,
                executed=bool(executed),
            )
        )
        outcome_index = int(len(scheduler_outcome_log))
        scheduler_outcome_log.append(outcome)
        if decision_log_index is not None and 0 <= int(decision_log_index) < len(scheduler_decision_log):
            scheduler_decision_log[int(decision_log_index)]["realized_outcome"] = dict(outcome)
            scheduler_decision_log[int(decision_log_index)]["scheduler_outcome_index"] = int(outcome_index)
        return outcome_index

    def _record_scheduler_decision(
        *,
        decision,
        parent_key: Any,
        parent_rec: Any,
        current_iter: int,
        applied: bool,
        selected_action: int | None = None,
    ) -> int | None:
        if not bool(scheduler_stats.get("enabled", False)) or decision is None:
            return None
        scheduler_stats["decision_count"] = int(scheduler_stats.get("decision_count", 0)) + 1
        scheduler_stats["candidate_count_sum"] = int(scheduler_stats.get("candidate_count_sum", 0)) + int(
            getattr(decision, "candidate_count", 0) or 0
        )
        if bool(getattr(decision, "trained", False)):
            scheduler_stats["scored"] = int(scheduler_stats.get("scored", 0)) + 1
        if bool(getattr(decision, "fallback_used", False)):
            scheduler_stats["fallback_selected"] = int(scheduler_stats.get("fallback_selected", 0)) + 1
        if bool(applied) and not bool(getattr(decision, "advisory_only", False)):
            scheduler_stats["control_selected"] = int(scheduler_stats.get("control_selected", 0)) + 1
        chosen_route = str(getattr(decision, "chosen_route", "") or "")
        if chosen_route:
            route_counts = scheduler_stats.get("route_counts", None)
            if isinstance(route_counts, dict):
                route_counts[chosen_route] = int(route_counts.get(chosen_route, 0)) + 1
        top_rows = [
            _scheduler_candidate_log_row(row)
            for row in list(getattr(decision, "rows", ()) or [])[:8]
            if isinstance(row, Mapping)
        ]
        log_row = {
            "iter": int(current_iter),
            "parent_key": str(parent_key or ""),
            "parent_expr": "" if parent_rec is None else str(node_str(getattr(parent_rec, "best_expr", None))),
            "selected_action": None if selected_action is None else str(ACTION_NAME.get(int(selected_action), f"action_{int(selected_action)}")),
            "applied": bool(applied),
            **_scheduler_decision_fields(decision),
            "chosen_candidate_prediction": _scheduler_candidate_log_row(_selected_scheduler_prediction_row(decision)),
            "runner_up_prediction": _scheduler_candidate_log_row(
                next(
                    (
                        row
                        for row in list(getattr(decision, "rows", ()) or [])[1:]
                        if isinstance(row, Mapping)
                    ),
                    None,
                )
            ),
            "acquisition_weights": {
                str(k): float(v)
                for k, v in dict(getattr(decision, "acquisition_weights", {}) or {}).items()
            },
            "top_candidates": top_rows,
        }
        scheduler_decision_log.append(log_row)
        return int(len(scheduler_decision_log) - 1)

    def _selected_scheduler_repair_candidate():
        candidate = scheduler_selected_candidate
        if candidate is None:
            return None
        if str(getattr(candidate, "route", "") or "") != "repair":
            return None
        return candidate

    def _repair_option_runtime_inputs():
        repair_initial_path = anchor_path
        repair_candidate_paths = preview_paths
        repair_first_step_expr = repair_preview_expr
        repair_first_step_meta = repair_preview_meta if isinstance(repair_preview_meta, dict) else None
        repair_target_mode = str(inverse_target_mode)
        repair_action_config: Any = repair_inverse_action_config
        scheduler_repair_candidate = _selected_scheduler_repair_candidate()
        if scheduler_repair_candidate is not None:
            selected_path = tuple(int(v) for v in (getattr(scheduler_repair_candidate, "path", ()) or ()))
            if selected_path:
                repair_initial_path = selected_path
                repair_candidate_paths = [selected_path]
            repair_first_step_expr = None
            repair_first_step_meta = None
            repair_target_mode = str(
                getattr(scheduler_repair_candidate, "target_mode", "") or inverse_target_mode
            )
            repair_action_kwargs = dict(repair_inverse_action_config.to_action_kwargs())
            repair_action_kwargs["target_mode"] = str(repair_target_mode)
            repair_action_kwargs["inverse_spec_exact_budget"] = int(
                getattr(
                    scheduler_repair_candidate,
                    "exact_budget",
                    inverse_spec_exact_budget,
                )
                or inverse_spec_exact_budget
            )
            repair_action_config = coerce_inverse_steering_config(repair_action_kwargs)
        return (
            repair_initial_path,
            repair_candidate_paths,
            repair_first_step_expr,
            repair_first_step_meta,
            repair_target_mode,
            repair_action_config,
        )

    def _scheduler_control_route_applied(selected_action: int | None) -> bool:
        if not bool(scheduler_control_applied) or selected_action is None:
            return False
        candidate = scheduler_selected_candidate
        if candidate is None:
            return False
        chosen_route = str(getattr(candidate, "route", "") or "")
        if chosen_route == "build":
            expected_action = ACTION_ID_BY_NAME.get(str(getattr(candidate, "action", "") or ""), None)
            return expected_action is not None and int(selected_action) == int(expected_action)
        if chosen_route == "repair":
            return bool(repair_controller_selected) and int(selected_action) == int(A_REPAIR)
        if chosen_route == "hole":
            return int(selected_action) == int(A_HOLESEARCH)
        return False

    if verbose and print_every > 0:
        dims_tag = " DIMS" if dm else ""
        act_names = '+'.join(ACTION_NAME[act] for act in tracked_actions)
        print(f"config: {label}({nvars}v) poly_deg={poly_degree} [{lo},{hi}] actions={act_names}{dims_tag} seed={seed} n_iter={n_iter} pool={len(pool_nodes)}")

    early_stop_mse = float(early_stop_mse)

    best_raw_mse = float("inf")
    best_raw_mse_struct = float("inf")
    best_mse = 1e100
    n_evaluated = 0
    n_attempts = 0
    max_attempts = n_iter * 20  # safety cap to avoid infinite loop
    search_stop_reason = None
    last_refine_slate_eval = -1

    def _run_refinement_slate_pass(source: str) -> bool:
        nonlocal best_raw_mse, best_raw_mse_struct, best_mse
        if not bool(scheduled_slate_refine_enable):
            return False

        stats: dict[str, Any] = {
            "source": str(source),
            "iter": int(n_evaluated),
            "budget": int(refine_slate_budget_i),
            "selected": 0,
            "scored": 0,
            "accepted": 0,
            "trials_used": 0,
            "score_none": 0,
            "not_improved": 0,
            "new_residual_basins": 0,
            "global_best_updates": 0,
            "stopped_budget": False,
            "stopped_wall_time": False,
        }
        passes = refine_slate_stats.get("passes", None)
        if isinstance(passes, list):
            passes.append(stats)
        refine_slate_stats["total_passes"] = int(refine_slate_stats.get("total_passes", 0)) + 1

        if refine_slate_budget_i <= 0:
            stats["skipped"] = "budget"
            return False
        if _wall_time_exceeded():
            stats["stopped_wall_time"] = True
            return False
        if stop_event is not None and stop_event.is_set():
            stats["skipped"] = "stop_event"
            return False
        if not arch.d:
            stats["skipped"] = "empty_archive"
            return False

        try:
            slate = _select_refinement_slate(
                arch,
                top_k=refine_slate_k_i,
                diverse_k=refine_slate_diverse_k_i,
            )
        except Exception as exc:
            stats["skipped"] = "selection_exception"
            stats["exception"] = str(exc)
            return False
        stats["selected"] = int(len(slate))
        refine_slate_stats["total_selected"] = int(refine_slate_stats.get("total_selected", 0)) + int(len(slate))
        if not slate:
            stats["skipped"] = "empty_slate"
            return False

        best_before = arch.best(1)[0].best_mse if arch.d else float("inf")
        stats["best_eff_before"] = _slate_float(best_before)
        if math.isfinite(float(best_before)) and float(best_before) < float(best_mse):
            best_mse = float(best_before)

        trials_used = 0
        any_solved = False
        sentinel = object()
        old_refine_limits = {
            "depth_trials_left": refine_state.get("depth_trials_left", sentinel),
            "window_trials_left": refine_state.get("window_trials_left", sentinel),
            "window_idx": refine_state.get("window_idx", sentinel),
        }
        try:
            for rec in slate:
                if _wall_time_exceeded():
                    stats["stopped_wall_time"] = True
                    if stop_event is not None:
                        stop_event.set()
                    break
                if stop_event is not None and stop_event.is_set():
                    stats["skipped"] = "stop_event"
                    break
                remaining = int(refine_slate_budget_i) - int(trials_used)
                if remaining <= 0:
                    stats["stopped_budget"] = True
                    break

                expr = getattr(rec, "best_expr", None)
                if expr is None:
                    continue
                parent_eff = _slate_float(getattr(rec, "best_mse", float("inf")))
                parent_raw = _slate_float(getattr(rec, "best_raw_mse", parent_eff), default=parent_eff)
                if math.isfinite(best_raw_mse_struct):
                    best_for_gate = float(best_raw_mse_struct)
                elif math.isfinite(best_raw_mse):
                    best_for_gate = float(max(best_raw_mse, float(early_stop_mse)))
                else:
                    best_for_gate = float("inf")

                score_cfg = dict(refine_cfg)
                score_cfg["score_prescreen_enable"] = False
                score_cfg["refine_context"] = "slate"
                refine_state["depth_trials_left"] = None
                refine_state["window_trials_left"] = int(remaining)
                before_trials = int(refine_state.get("trials_done", 0))
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
                    refine_enable=True,
                    refine_cfg=score_cfg,
                    refine_best_mse=best_for_gate,
                    refine_state=refine_state,
                    return_expr=True,
                )
                after_trials = int(refine_state.get("trials_done", 0))
                used_now = max(0, int(after_trials - before_trials))
                trials_used += int(used_now)
                stats["trials_used"] = int(trials_used)
                stats["scored"] = int(stats.get("scored", 0)) + 1
                if sc is None:
                    stats["score_none"] = int(stats.get("score_none", 0)) + 1
                    continue

                mse, key, z, mapping, scored_expr = sc
                raw_mse = _slate_float(mse)
                try:
                    is_structural = bool(mapping_is_structural(mapping))
                except Exception:
                    is_structural = False
                try:
                    size_cost = float(node_size(scored_expr))
                except Exception:
                    size_cost = float(node_size(expr))
                try:
                    map_cost = float(mapping_cost(mapping))
                except Exception:
                    map_cost = 0.0
                mse_eff = float(raw_mse) + float(complexity_penalty) * (size_cost + map_cost)

                improves_parent = (
                    math.isfinite(mse_eff)
                    and (
                        (math.isfinite(parent_eff) and mse_eff < parent_eff * (1.0 - 1.0e-12))
                        or (math.isfinite(parent_raw) and raw_mse < parent_raw * (1.0 - 1.0e-12))
                        or (not math.isfinite(parent_eff) and not math.isfinite(parent_raw))
                    )
                )
                improves_global = math.isfinite(mse_eff) and mse_eff < float(best_mse) * (1.0 - 1.0e-12)
                improves_structural = (
                    bool(is_structural)
                    and math.isfinite(raw_mse)
                    and (
                        (math.isfinite(best_raw_mse_struct) and raw_mse < best_raw_mse_struct * (1.0 - 1.0e-12))
                        or not math.isfinite(best_raw_mse_struct)
                    )
                )
                if not (improves_parent or improves_global or improves_structural):
                    stats["not_improved"] = int(stats.get("not_improved", 0)) + 1
                    continue

                is_new = arch.update(key, mse_eff, scored_expr, z, mapping, raw_mse=raw_mse)
                stats["accepted"] = int(stats.get("accepted", 0)) + 1
                if is_new:
                    stats["new_residual_basins"] = int(stats.get("new_residual_basins", 0)) + 1
                if math.isfinite(raw_mse) and raw_mse < best_raw_mse:
                    best_raw_mse = float(raw_mse)
                if bool(is_structural) and math.isfinite(raw_mse) and raw_mse < best_raw_mse_struct:
                    best_raw_mse_struct = float(raw_mse)
                if math.isfinite(mse_eff) and mse_eff < best_mse:
                    best_mse = float(mse_eff)
                    stats["global_best_updates"] = int(stats.get("global_best_updates", 0)) + 1
                if best_raw_mse_struct < early_stop_mse:
                    any_solved = True
                    if stop_event is not None:
                        stop_event.set()
                    break
        finally:
            for key, value in old_refine_limits.items():
                if value is sentinel:
                    refine_state.pop(key, None)
                else:
                    refine_state[key] = value

        stats["best_eff_after"] = _slate_float(arch.best(1)[0].best_mse if arch.d else float("inf"))
        refine_slate_stats["total_scored"] = int(refine_slate_stats.get("total_scored", 0)) + int(stats.get("scored", 0))
        refine_slate_stats["total_accepted"] = int(refine_slate_stats.get("total_accepted", 0)) + int(stats.get("accepted", 0))
        refine_slate_stats["total_trials_used"] = int(refine_slate_stats.get("total_trials_used", 0)) + int(stats.get("trials_used", 0))
        if verbose and int(stats.get("scored", 0)) > 0:
            print(
                f"[refine-slate] {source}: selected={int(stats['selected'])} "
                f"scored={int(stats['scored'])} accepted={int(stats['accepted'])} "
                f"trials={int(stats['trials_used'])} best_mse={float(stats['best_eff_after']):.3e}"
            )
        return bool(any_solved)

    phase_timing["setup_wall_s"] = float(time.perf_counter() - setup_started)

    # --- Brute-force enumeration phase ---
    if brute_depth is None or brute_depth > 0:
        brute_refine_cfg = dict(refine_cfg)
        brute_refine_cfg["refine_context"] = "brute"
        brute_score_calls_before = int(refine_diagnostics.get("score_calls", 0) or 0)
        brute_started = time.perf_counter()
        brute_solved = _run_brute_phase(
            arch, nvars,
            x_fit, y_fit, x_probe, y_probe, proj,
            fp_mode, q_scale, q_clip, poly_degree,
            var_dims=var_dims, y_dims=y_dims,
            brute_depth=brute_depth,
            early_stop_mse=early_stop_mse,
            max_expressions=brute_max_expressions,
            refine_enable=brute_refine_enable,
            refine_cfg=brute_refine_cfg,
            refine_state=refine_state,
            label=label,
            shuffle_seed=rng.randrange(2**31),
            verbose=verbose,
            stop_event=stop_event,
            wall_time_deadline=wall_time_deadline,
        )
        phase_timing["brute_wall_s"] = float(phase_timing.get("brute_wall_s", 0.0) or 0.0) + float(
            time.perf_counter() - brute_started
        )
        brute_score_calls_after = int(refine_diagnostics.get("score_calls", 0) or 0)
        phase_timing["brute_scored"] = int(phase_timing.get("brute_scored", 0) or 0) + max(
            0, int(brute_score_calls_after - brute_score_calls_before)
        )
        if arch.d:
            b = arch.best(1)[0]
            if b.best_mse < best_mse:
                best_mse = b.best_mse
            if b.best_raw_mse < best_raw_mse:
                best_raw_mse = b.best_raw_mse
        if bool(after_brute_slate_refine_enable) and not bool(brute_solved):
            if _run_refinement_slate_pass("after_brute"):
                brute_solved = True
        if brute_solved:
            if verbose:
                print("[brute]  SOLVED — skipping mutation search")
            _finalize_search_state("early_stop_mse")
            arch.x_fit = x_fit
            arch.y_fit = y_fit
            arch.x_probe = x_probe
            arch.y_probe = y_probe
            arch.crossover_policy_stats = _finalize_crossover_policy_stats(crossover_policy_stats)
            arch.action_distribution = _finalize_action_distribution(
                tracked_actions,
                action_selected_counts,
                action_proposed_counts,
                action_reward_counts,
                action_accepted_counts,
            )
            arch.boost_gate_stats = boost_gate_stats
            arch.inverse_gate_stats = inverse_gate_stats
            arch.repair_pass_stats = repair_pass_stats
            arch.closure_search_stats = closure_search_stats
            arch.repair_controller_stats = repair_controller_stats
            arch.macro_controller_stats = macro_controller_stats
            route_scheduler_stats["route_summary"] = route_scheduler.summary() if route_scheduler is not None else {}
            arch.route_scheduler_stats = route_scheduler_stats
            if macro_controller is not None:
                arch.macro_controller_summary = macro_controller.summary(topk=len(tracked_actions))
            return arch
        if _wall_time_exceeded():
            if stop_event is not None:
                stop_event.set()
            if verbose:
                print(
                    f"[search] wall-time limit hit after brute phase at "
                    f"{float(max(0.0, time.perf_counter() - search_started)):.2f}s"
                )
            _finalize_search_state("wall_time_limit")
            arch.x_fit = x_fit
            arch.y_fit = y_fit
            arch.x_probe = x_probe
            arch.y_probe = y_probe
            arch.crossover_policy_stats = _finalize_crossover_policy_stats(crossover_policy_stats)
            arch.action_distribution = _finalize_action_distribution(
                tracked_actions,
                action_selected_counts,
                action_proposed_counts,
                action_reward_counts,
                action_accepted_counts,
            )
            arch.boost_gate_stats = boost_gate_stats
            arch.inverse_gate_stats = inverse_gate_stats
            arch.repair_pass_stats = repair_pass_stats
            arch.closure_search_stats = closure_search_stats
            arch.repair_controller_stats = repair_controller_stats
            arch.macro_controller_stats = macro_controller_stats
            route_scheduler_stats["route_summary"] = route_scheduler.summary() if route_scheduler is not None else {}
            arch.route_scheduler_stats = route_scheduler_stats
            if macro_controller is not None:
                arch.macro_controller_summary = macro_controller.summary(topk=len(tracked_actions))
            return arch

    if refine_state is not None:
        refine_state["depth_trials_left"] = None

    # Early exit: if dimensional filtering is active and the target dimension
    # is not in the reachable set, the mutation loop would spin futilely
    # (every rand_node_dim call returns None).  Skip it.
    if dm and y_dims is not None and reach is not None and not arch.d:
        y_key = dim_round(tuple(y_dims))
        max_reach = reach[min(max_depth, len(reach) - 1)]
        if y_key not in max_reach:
            if verbose:
                print(f"[mutate]  target dim {y_key} unreachable from var dims — skipping mutation search")
            _finalize_search_state("unreachable_dims")
            arch.x_fit = x_fit
            arch.y_fit = y_fit
            arch.x_probe = x_probe
            arch.y_probe = y_probe
            arch.crossover_policy_stats = _finalize_crossover_policy_stats(crossover_policy_stats)
            arch.action_distribution = _finalize_action_distribution(
                tracked_actions,
                action_selected_counts,
                action_proposed_counts,
                action_reward_counts,
                action_accepted_counts,
            )
            arch.boost_gate_stats = boost_gate_stats
            arch.inverse_gate_stats = inverse_gate_stats
            arch.repair_pass_stats = repair_pass_stats
            arch.closure_search_stats = closure_search_stats
            arch.repair_controller_stats = repair_controller_stats
            arch.macro_controller_stats = macro_controller_stats
            route_scheduler_stats["route_summary"] = route_scheduler.summary() if route_scheduler is not None else {}
            arch.route_scheduler_stats = route_scheduler_stats
            if macro_controller is not None:
                arch.macro_controller_summary = macro_controller.summary(topk=len(tracked_actions))
            return arch

    last_progress_eval = 0
    last_progress_residual_basins = int(len(arch.d))

    # Stall detection state
    stall_best = float('inf')
    stall_count = 0
    mutate_start_best = arch.best(1)[0].best_mse if arch.d else float("inf")
    mut_window = max(1, int(refine_cfg.get("mutation_window", 500)))
    mut_refines = int(refine_cfg.get("trials_per_mutation_window", 0))
    if mutation_refine_enable and refine_state is not None and mut_refines > 0:
        refine_state["window_trials_left"] = mut_refines
        refine_state["window_idx"] = -1

    mutation_started = time.perf_counter()

    proposal_scoring_state = ProposalScoringState(
        n_evaluated=int(n_evaluated),
        best_raw_mse=float(best_raw_mse),
        best_raw_mse_struct=float(best_raw_mse_struct),
        best_mse=float(best_mse),
    )

    def _record_route_status(stats: dict[str, Any], status: object) -> None:
        _record_route_status_impl(stats, status)

    def _merge_route_status_counts(stats: dict[str, Any], counts: Mapping[str, Any] | None) -> None:
        _merge_route_status_counts_impl(stats, counts)

    def _score_external_candidate_expr(
        expr,
        *,
        parent_raw_mse: float | None,
        stats: dict[str, Any],
        route_name: str,
        candidate_meta: Mapping[str, Any] | None = None,
    ):
        nonlocal n_evaluated, best_raw_mse, best_raw_mse_struct, best_mse
        proposal_scoring_state.n_evaluated = int(n_evaluated)
        proposal_scoring_state.best_raw_mse = float(best_raw_mse)
        proposal_scoring_state.best_raw_mse_struct = float(best_raw_mse_struct)
        proposal_scoring_state.best_mse = float(best_mse)
        external_refine_cfg = dict(refine_cfg)
        route_key = str(route_name or "external")
        if "controller" in route_key:
            external_refine_cfg["refine_context"] = "controller_slate"
        elif "slate" in route_key:
            external_refine_cfg["refine_context"] = "slate"
        else:
            external_refine_cfg["refine_context"] = "external"
        ret = _score_external_candidate_expr_impl(
            expr,
            parent_raw_mse=parent_raw_mse,
            stats=stats,
            route_name=route_name,
            candidate_meta=candidate_meta,
            state=proposal_scoring_state,
            dm=bool(dm),
            var_dims=var_dims,
            y_dims=y_dims,
            refine_cfg=external_refine_cfg,
            score_prescreen_stats=score_prescreen_stats,
            closure_search_anchor_head_compare_enable=bool(closure_search_anchor_head_compare_enable),
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            proj=proj,
            fp_mode=fp_mode,
            q_scale=q_scale,
            q_clip=q_clip,
            poly_degree=int(poly_degree),
            refine_enable=bool(mutation_refine_enable),
            refine_state=refine_state,
            early_stop_mse=float(early_stop_mse),
            complexity_penalty=float(complexity_penalty),
            score_expr_fn=score_expr_fn,
            simplify_fn=simplify,
            is_valid_node_fn=is_valid_node,
            node_str_fn=node_str,
            node_dims_fn=node_dims,
            dims_eq_fn=dims_eq,
            node_size_fn=node_size,
            mapping_cost_fn=mapping_cost,
            mapping_is_structural_fn=mapping_is_structural,
            arch=arch,
        )
        n_evaluated = int(proposal_scoring_state.n_evaluated)
        best_raw_mse = float(proposal_scoring_state.best_raw_mse)
        best_raw_mse_struct = float(proposal_scoring_state.best_raw_mse_struct)
        best_mse = float(proposal_scoring_state.best_mse)
        return ret

    def _score_native_candidate_basis_state(
        *,
        candidate_meta: Mapping[str, Any] | None,
        stats: dict[str, Any],
        route_name: str,
    ):
        if score_expr_fn is not _engine_score_expr:
            return None
        nonlocal n_evaluated, best_raw_mse, best_raw_mse_struct, best_mse
        proposal_scoring_state.n_evaluated = int(n_evaluated)
        proposal_scoring_state.best_raw_mse = float(best_raw_mse)
        proposal_scoring_state.best_raw_mse_struct = float(best_raw_mse_struct)
        proposal_scoring_state.best_mse = float(best_mse)
        ret = _score_native_candidate_basis_state_impl(
            candidate_meta=candidate_meta,
            stats=stats,
            route_name=route_name,
            state=proposal_scoring_state,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            complexity_penalty=float(complexity_penalty),
            node_str_fn=node_str,
            arch=arch,
        )
        n_evaluated = int(proposal_scoring_state.n_evaluated)
        best_raw_mse = float(proposal_scoring_state.best_raw_mse)
        best_raw_mse_struct = float(proposal_scoring_state.best_raw_mse_struct)
        best_mse = float(proposal_scoring_state.best_mse)
        return ret

    def _run_archive_repair_pass() -> None:
        if not bool(repair_pass_enable):
            return

        elite_k = max(0, int(repair_pass_elite_k))
        round_cap = max(0, int(repair_pass_rounds))
        paths_per_elite = max(1, int(repair_pass_paths_per_elite))
        if elite_k <= 0 or round_cap <= 0:
            return
        if not arch.d:
            repair_pass_stats["skipped_empty_archive"] = int(
                repair_pass_stats.get("skipped_empty_archive", 0)
            ) + 1
            return

        try:
            elite_recs = arch.best(elite_k, strategy="mse_decade_size")
        except Exception:
            elite_recs = arch.best(elite_k)
        repair_pass_stats["elites_selected"] = int(len(elite_recs or []))

        stop_repair = False
        for elite_rec in list(elite_recs or []):
            if _wall_time_exceeded():
                repair_pass_stats["stopped_wall_time"] = True
                break

            current_expr = getattr(elite_rec, "best_expr", None)
            current_mapping = getattr(elite_rec, "mapping", None)
            if current_expr is None or current_mapping is None:
                continue
            try:
                current_eff = float(getattr(elite_rec, "best_mse", float("inf")))
            except Exception:
                current_eff = float("inf")
            try:
                current_raw = float(getattr(elite_rec, "best_raw_mse", current_eff))
            except Exception:
                current_raw = current_eff
            if not math.isfinite(current_eff):
                continue

            repair_pass_stats["elites_considered"] = int(
                repair_pass_stats.get("elites_considered", 0)
            ) + 1
            elite_improved = False

            for _round_idx in range(int(round_cap)):
                if _wall_time_exceeded():
                    repair_pass_stats["stopped_wall_time"] = True
                    stop_repair = True
                    break

                repair_pass_stats["rounds_attempted"] = int(
                    repair_pass_stats.get("rounds_attempted", 0)
                ) + 1
                try:
                    diag = estimate_inverse_steering_potential(
                        current_expr,
                        current_mapping,
                        x_fit,
                        y_fit,
                        x_probe,
                        y_probe,
                        boost_pool_phi_fit,
                        boost_pool_phi,
                        boost_pool_dims,
                        pool_nodes=boost_pool_nodes,
                        var_dims=var_dims,
                        max_paths=inverse_gate_max_paths,
                        topk_terms=max(1, min(int(inverse_topk_terms), 4)),
                        shortlist_mult=max(1, min(int(inverse_shortlist_mult), 2)),
                        min_valid_frac=inverse_min_valid_frac,
                        min_confidence=inverse_min_confidence,
                        min_structural_score=inverse_gate_min_structural_score,
                        min_weighted_rel_gain=inverse_gate_min_weighted_rel_gain,
                        structural_bias=inverse_gate_structural_bias,
                        safe_eps=float(inverse_safe_eps),
                        confidence_mode=str(inverse_confidence_mode),
                        confidence_target_gain=float(inverse_confidence_target_gain),
                        confidence_floor=float(inverse_confidence_floor),
                        branch_beam_width=int(inverse_branch_beam_width),
                        local_score_mode=str(inverse_local_score_mode),
                        target_mode=str(inverse_target_mode),
                        full_mapping_penalty=float(inverse_full_mapping_penalty),
                        exact_simple_target_bonus=float(inverse_exact_simple_target_bonus),
                        additive_descend_penalty=float(inverse_additive_descend_penalty),
                        nonadditive_leaf_penalty=float(inverse_nonadditive_leaf_penalty),
                        periodic_min_valid_scale=float(inverse_periodic_min_valid_scale),
                        periodic_min_confidence_scale=float(inverse_periodic_min_confidence_scale),
                        periodic_path_penalty=float(inverse_periodic_path_penalty),
                        nonperiodic_muldiv_bonus=float(inverse_nonperiodic_muldiv_bonus),
                        nonperiodic_explogsqrt_bonus=float(inverse_nonperiodic_explogsqrt_bonus),
                        branch_ambiguity_penalty=float(inverse_branch_ambiguity_penalty),
                    )
                except Exception:
                    _record_route_status(repair_pass_stats, "potential_exception")
                    continue

                repair_pass_stats["potential_calls"] = int(
                    repair_pass_stats.get("potential_calls", 0)
                ) + 1
                if bool(getattr(diag, "allowed", False)):
                    repair_pass_stats["potential_allowed"] = int(
                        repair_pass_stats.get("potential_allowed", 0)
                    ) + 1

                candidate_paths = []
                seen_paths = set()
                best_path = tuple(int(v) for v in (getattr(diag, "best_path", ()) or ()))
                if best_path:
                    seen_paths.add(best_path)
                    candidate_paths.append(best_path)
                for raw_path in list(getattr(diag, "candidate_paths", ()) or ()):
                    path = tuple(int(v) for v in (raw_path or ()))
                    if not path or path in seen_paths:
                        continue
                    seen_paths.add(path)
                    candidate_paths.append(path)

                repair_pass_stats["paths_ranked"] = int(
                    repair_pass_stats.get("paths_ranked", 0)
                ) + int(len(candidate_paths))
                if not candidate_paths:
                    repair_pass_stats["skipped_no_paths"] = int(
                        repair_pass_stats.get("skipped_no_paths", 0)
                    ) + 1
                    break

                improved_this_round = False
                for path in candidate_paths[:paths_per_elite]:
                    if _wall_time_exceeded():
                        repair_pass_stats["stopped_wall_time"] = True
                        stop_repair = True
                        break

                    repair_pass_stats["solver_calls"] = int(
                        repair_pass_stats.get("solver_calls", 0)
                    ) + 1
                    try:
                        inv_ret = apply_inverse_steering_action(
                            current_expr,
                            current_mapping,
                            x_fit,
                            y_fit,
                            x_probe,
                            y_probe,
                            boost_pool_nodes,
                            boost_pool_phi_fit,
                            boost_pool_phi,
                            boost_pool_dims,
                            rng,
                            max_depth,
                            nvars,
                            poly_degree,
                            var_dims=var_dims,
                            max_paths=1,
                            topk_terms=inverse_topk_terms,
                            shortlist_mult=inverse_shortlist_mult,
                            min_valid_frac=inverse_min_valid_frac,
                            min_confidence=inverse_min_confidence,
                            safe_eps=float(inverse_safe_eps),
                            confidence_mode=str(inverse_confidence_mode),
                            confidence_target_gain=float(inverse_confidence_target_gain),
                            confidence_floor=float(inverse_confidence_floor),
                            branch_beam_width=int(inverse_branch_beam_width),
                            micro_search_enable=bool(inverse_micro_search_enable),
                            micro_search_max_depth=int(inverse_micro_search_max_depth),
                            micro_search_beam_width=int(inverse_micro_search_beam_width),
                            micro_search_topk=int(inverse_micro_search_topk),
                            micro_search_seed_terms=int(inverse_micro_search_seed_terms),
                            local_score_mode=str(inverse_local_score_mode),
                            inverse_spec_enable=bool(inverse_spec_enable),
                            inverse_spec_enum_max_depth=int(inverse_spec_enum_max_depth),
                            inverse_spec_enum_max_trees=int(inverse_spec_enum_max_trees),
                            inverse_spec_preview_topk=int(inverse_spec_preview_topk),
                            inverse_spec_local_score_mode=str(inverse_spec_local_score_mode),
                            inverse_spec_include_legacy_seed=bool(inverse_spec_include_legacy_seed),
                            inverse_spec_complexity_penalty=float(inverse_spec_complexity_penalty),
                            inverse_spec_family_battery_enable=bool(inverse_spec_family_battery_enable),
                            inverse_spec_family_battery_mode=str(inverse_spec_family_battery_mode or "outer"),
                            inverse_spec_repair_quota=float(inverse_spec_repair_quota),
                            inverse_spec_recursive_enable=bool(inverse_spec_recursive_enable),
                            inverse_spec_recursive_max_depth=int(inverse_spec_recursive_max_depth),
                            inverse_spec_recursive_trigger_rel_mse=float(inverse_spec_recursive_trigger_rel_mse),
                            inverse_spec_recursive_seed_cap=int(inverse_spec_recursive_seed_cap),
                            inverse_spec_recursive_branch_topk=int(inverse_spec_recursive_branch_topk),
                            inverse_spec_recursive_child_topk=int(inverse_spec_recursive_child_topk),
                            inverse_spec_witness_jets_enable=bool(inverse_spec_witness_jets_enable),
                            inverse_spec_witness_d2_enable=bool(inverse_spec_witness_d2_enable),
                            inverse_spec_witness_max_rows=int(inverse_spec_witness_max_rows),
                            inverse_spec_witness_loss_enable=bool(inverse_spec_witness_loss_enable),
                            inverse_spec_witness_grad_weight=float(inverse_spec_witness_grad_weight),
                            inverse_spec_witness_d2_weight=float(inverse_spec_witness_d2_weight),
                            inverse_spec_witness_diag_weight=float(inverse_spec_witness_diag_weight),
                            inverse_spec_witness_physics_weight=float(inverse_spec_witness_physics_weight),
                            inverse_spec_active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
                            inverse_spec_active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
                            inverse_spec_active_var_max_count=int(inverse_spec_active_var_max_count),
                            inverse_spec_directional_market_enable=bool(inverse_spec_directional_market_enable),
                            inverse_spec_max_subtree_depth=inverse_spec_max_subtree_depth,
                            inverse_spec_fit_cap=int(inverse_spec_fit_cap),
                            inverse_spec_probe_cap=int(inverse_spec_probe_cap),
                            inverse_spec_exact_budget=int(inverse_spec_exact_budget),
                            target_mode=str(inverse_target_mode),
                            full_mapping_penalty=float(inverse_full_mapping_penalty),
                            exact_simple_target_bonus=float(inverse_exact_simple_target_bonus),
                            additive_descend_penalty=float(inverse_additive_descend_penalty),
                            nonadditive_leaf_penalty=float(inverse_nonadditive_leaf_penalty),
                            exact_path_eta=float(inverse_exact_path_eta),
                            exact_transport_min_lin_rel=float(inverse_exact_transport_min_lin_rel),
                            periodic_min_valid_scale=float(inverse_periodic_min_valid_scale),
                            periodic_min_confidence_scale=float(inverse_periodic_min_confidence_scale),
                            periodic_path_penalty=float(inverse_periodic_path_penalty),
                            nonperiodic_muldiv_bonus=float(inverse_nonperiodic_muldiv_bonus),
                            nonperiodic_explogsqrt_bonus=float(inverse_nonperiodic_explogsqrt_bonus),
                            branch_ambiguity_penalty=float(inverse_branch_ambiguity_penalty),
                            transport_min_lin_rel=float(inverse_transport_min_lin_rel),
                            transport_min_effective_n=float(inverse_transport_min_effective_n),
                            complexity_penalty=complexity_penalty,
                            candidate_paths=[path],
                            proj=proj,
                            fp_mode=fp_mode,
                            q_scale=q_scale,
                            q_clip=q_clip,
                            score_expr_cfg=refine_cfg,
                            return_meta=True,
                            repair_opportunity_controller_enable=False,
                            inverse_spec_regime_metadata=inverse_spec_regime_metadata,
                        )
                    except Exception:
                        _record_route_status(repair_pass_stats, "solver_exception")
                        continue

                    if isinstance(inv_ret, tuple) and len(inv_ret) == 2:
                        repaired_expr, repair_meta = inv_ret
                    else:
                        repaired_expr, repair_meta = inv_ret, {}
                    status = str((repair_meta or {}).get("status", "")) if isinstance(repair_meta, Mapping) else ""
                    _record_route_status(
                        repair_pass_stats,
                        status or ("ok" if repaired_expr is not None else "solver_none"),
                    )
                    if repaired_expr is None:
                        continue
                    if status in ("", "ok"):
                        repair_pass_stats["solver_ok"] = int(
                            repair_pass_stats.get("solver_ok", 0)
                        ) + 1

                    scored_child = _score_external_candidate_expr(
                        repaired_expr,
                        parent_raw_mse=current_raw,
                        stats=repair_pass_stats,
                        route_name="repair_pass",
                    )
                    if scored_child is None:
                        continue

                    if float(scored_child["eff_mse"]) < float(current_eff) * (1.0 - 1.0e-12):
                        repair_pass_stats["accepted_repairs"] = int(
                            repair_pass_stats.get("accepted_repairs", 0)
                        ) + 1
                        current_expr = scored_child["expr"]
                        current_mapping = scored_child["mapping"]
                        current_eff = float(scored_child["eff_mse"])
                        current_raw = float(scored_child["raw_mse"])
                        improved_this_round = True
                        elite_improved = True
                        break

                if stop_repair:
                    break
                if not improved_this_round:
                    break

            if elite_improved:
                repair_pass_stats["elites_improved"] = int(
                    repair_pass_stats.get("elites_improved", 0)
                ) + 1
            if stop_repair:
                break

    def _run_closure_search_pass() -> None:
        _run_closure_search_pass_route(
            closure_search_enable=bool(closure_search_enable),
            closure_search_stats=closure_search_stats,
            closure_search_families=closure_search_families,
            closure_search_max_proposals=int(closure_search_max_proposals),
            closure_search_anchors_per_family=int(closure_search_anchors_per_family),
            closure_search_preview_topk=int(closure_search_preview_topk),
            closure_search_exact_topk=int(closure_search_exact_topk),
            closure_search_beam_width=int(closure_search_beam_width),
            closure_search_seed_exact_topk=int(closure_search_seed_exact_topk),
            closure_search_seed_beam_width=int(closure_search_seed_beam_width),
            closure_search_seed_scaffold_reserve=int(closure_search_seed_scaffold_reserve),
            closure_search_seed_family_cap=int(closure_search_seed_family_cap),
            closure_search_seed_exact_bound_bonus=float(closure_search_seed_exact_bound_bonus),
            closure_search_pair_normal_enable=bool(closure_search_pair_normal_enable),
            closure_search_pair_normal_topk=int(closure_search_pair_normal_topk),
            closure_search_pair_normal_max_pairs=int(closure_search_pair_normal_max_pairs),
            closure_search_pair_rescue_enable=bool(closure_search_pair_rescue_enable),
            closure_search_pair_rescue_topk=int(closure_search_pair_rescue_topk),
            closure_search_pair_rescue_max_pairs=int(closure_search_pair_rescue_max_pairs),
            closure_search_emergent_basis_enable=bool(closure_search_emergent_basis_enable),
            closure_search_emergent_basis_max_source_rows=int(closure_search_emergent_basis_max_source_rows),
            closure_search_emergent_basis_score_topk=int(closure_search_emergent_basis_score_topk),
            closure_search_emergent_basis_max_per_round=int(closure_search_emergent_basis_max_per_round),
            closure_search_emergent_basis_max_total=int(closure_search_emergent_basis_max_total),
            closure_search_emergent_basis_min_probe_gain_rel=float(
                closure_search_emergent_basis_min_probe_gain_rel
            ),
            closure_search_emergent_aux_atoms_enable=bool(closure_search_emergent_aux_atoms_enable),
            closure_search_emergent_aux_atoms_max_source_rows=int(
                closure_search_emergent_aux_atoms_max_source_rows
            ),
            closure_search_emergent_aux_atoms_max_new_per_round=int(
                closure_search_emergent_aux_atoms_max_new_per_round
            ),
            closure_search_emergent_aux_atoms_max_total=int(closure_search_emergent_aux_atoms_max_total),
            closure_search_emergent_aux_atoms_max_target=int(closure_search_emergent_aux_atoms_max_target),
            closure_search_emergent_aux_atoms_max_dimensionless=int(
                closure_search_emergent_aux_atoms_max_dimensionless
            ),
            closure_search_emergent_aux_atoms_max_rational_derived=int(
                closure_search_emergent_aux_atoms_max_rational_derived
            ),
            closure_search_emergent_aux_atoms_max_seed_blocks=int(
                closure_search_emergent_aux_atoms_max_seed_blocks
            ),
            closure_search_debug_topk=int(closure_search_debug_topk),
            closure_search_min_valid_frac=float(closure_search_min_valid_frac),
            closure_search_min_confidence=float(closure_search_min_confidence),
            closure_search_periodic_min_valid_scale=float(closure_search_periodic_min_valid_scale),
            closure_search_periodic_min_confidence_scale=float(closure_search_periodic_min_confidence_scale),
            closure_search_transport_min_lin_rel=float(closure_search_transport_min_lin_rel),
            inverse_periodic_path_penalty=float(inverse_periodic_path_penalty),
            inverse_nonperiodic_muldiv_bonus=float(inverse_nonperiodic_muldiv_bonus),
            inverse_nonperiodic_explogsqrt_bonus=float(inverse_nonperiodic_explogsqrt_bonus),
            inverse_branch_beam_width=int(inverse_branch_beam_width),
            inverse_topk_terms=int(inverse_topk_terms),
            inverse_shortlist_mult=int(inverse_shortlist_mult),
            inverse_local_score_mode=str(inverse_local_score_mode),
            inverse_micro_search_enable=bool(inverse_micro_search_enable),
            inverse_micro_search_max_depth=int(inverse_micro_search_max_depth),
            inverse_micro_search_beam_width=int(inverse_micro_search_beam_width),
            inverse_micro_search_topk=int(inverse_micro_search_topk),
            inverse_micro_search_seed_terms=int(inverse_micro_search_seed_terms),
            inverse_target_mode=str(inverse_target_mode),
            inverse_safe_eps=float(inverse_safe_eps),
            inverse_confidence_mode=str(inverse_confidence_mode),
            inverse_confidence_target_gain=float(inverse_confidence_target_gain),
            inverse_confidence_floor=float(inverse_confidence_floor),
            inverse_full_mapping_penalty=float(inverse_full_mapping_penalty),
            inverse_exact_simple_target_bonus=float(inverse_exact_simple_target_bonus),
            inverse_additive_descend_penalty=float(inverse_additive_descend_penalty),
            inverse_nonadditive_leaf_penalty=float(inverse_nonadditive_leaf_penalty),
            inverse_exact_path_eta=float(inverse_exact_path_eta),
            inverse_branch_ambiguity_penalty=float(inverse_branch_ambiguity_penalty),
            inverse_transport_min_effective_n=float(inverse_transport_min_effective_n),
            inverse_spec_regime_metadata=inverse_spec_regime_metadata,
            inverse_spec_local_score_mode=str(inverse_spec_local_score_mode),
            inverse_spec_enum_max_depth=int(inverse_spec_enum_max_depth),
            inverse_spec_enum_max_trees=int(inverse_spec_enum_max_trees),
            inverse_spec_max_subtree_depth=inverse_spec_max_subtree_depth,
            inverse_spec_complexity_penalty=float(inverse_spec_complexity_penalty),
            inverse_spec_family_battery_enable=bool(inverse_spec_family_battery_enable),
            inverse_spec_family_battery_mode=str(inverse_spec_family_battery_mode or "outer"),
            inverse_spec_recursive_enable=bool(inverse_spec_recursive_enable),
            inverse_spec_recursive_max_depth=int(inverse_spec_recursive_max_depth),
            inverse_spec_recursive_trigger_rel_mse=float(inverse_spec_recursive_trigger_rel_mse),
            inverse_spec_recursive_seed_cap=int(inverse_spec_recursive_seed_cap),
            inverse_spec_recursive_branch_topk=int(inverse_spec_recursive_branch_topk),
            inverse_spec_recursive_child_topk=int(inverse_spec_recursive_child_topk),
            inverse_spec_witness_jets_enable=bool(inverse_spec_witness_jets_enable),
            inverse_spec_witness_d2_enable=bool(inverse_spec_witness_d2_enable),
            inverse_spec_witness_max_rows=int(inverse_spec_witness_max_rows),
            inverse_spec_witness_loss_enable=bool(inverse_spec_witness_loss_enable),
            inverse_spec_witness_grad_weight=float(inverse_spec_witness_grad_weight),
            inverse_spec_witness_d2_weight=float(inverse_spec_witness_d2_weight),
            inverse_spec_witness_diag_weight=float(inverse_spec_witness_diag_weight),
            inverse_spec_witness_physics_weight=float(inverse_spec_witness_physics_weight),
            inverse_spec_active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
            inverse_spec_active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
            inverse_spec_active_var_max_count=int(inverse_spec_active_var_max_count),
            wall_time_deadline=wall_time_deadline,
            wall_time_limit_s=wall_time_limit_s,
            max_depth=int(max_depth),
            poly_degree=int(poly_degree),
            nvars=int(nvars),
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            var_dims=var_dims,
            y_dims=y_dims,
            boost_pool_nodes=boost_pool_nodes,
            boost_pool_phi_fit=boost_pool_phi_fit,
            boost_pool_phi=boost_pool_phi,
            boost_pool_dims=boost_pool_dims,
            dm=bool(dm),
            wall_time_exceeded_fn=_wall_time_exceeded,
            run_closure_search_pass_impl=_run_closure_search_pass_impl,
            score_external_candidate_expr_fn=_score_external_candidate_expr,
            score_native_candidate_basis_state_fn=_score_native_candidate_basis_state,
            node_str_fn=node_str,
        )

    def _base_search_is_active() -> bool:
        """Apply terminal search gates shared by independently budgeted phases."""
        nonlocal search_stop_reason
        if search_stop_reason is not None:
            return False
        if best_raw_mse_struct < early_stop_mse:
            search_stop_reason = "early_stop_mse"
            if stop_event is not None:
                stop_event.set()
            return False
        if _wall_time_exceeded():
            search_stop_reason = "wall_time_limit"
            if stop_event is not None:
                stop_event.set()
            return False
        if stop_event is not None and stop_event.is_set():
            search_stop_reason = "stop_event"
            return False
        return True

    def _base_search_can_score() -> bool:
        """Return whether seed or mutation scoring still has base budget."""
        if not _base_search_is_active():
            return False
        # Closure, archive repair, and final polish have independent budgets;
        # exhausting n_iter stops only seed and mutation scoring.
        return n_evaluated < n_iter

    # --- Periodic seeding phase ---
    # Seed sin/cos(omega*x_j) candidates at periodogram-detected frequencies.
    # They score well immediately (unlike canonical-frequency skeletons), so
    # the ordinary mutation / skeleton-refinement / additive-combo machinery
    # can take over frequency polishing and composition.
    if bool(periodic_seed_enable) and _base_search_can_score():
        periodic_hints = _periodogram_frequency_hints(
            x_fit,
            y_fit,
            max_hints=int(periodic_seed_max_hints),
            min_prominence=float(periodic_seed_min_prominence),
        )
        if periodic_hints:
            periodic_seed_stats: dict[str, Any] = {}
            periodic_candidates = (
                (seed_var, seed_omega, seed_fn)
                for seed_var, seed_omega in periodic_hints
                for seed_fn in ("sin", "cos")
            )
            for seed_var, seed_omega, seed_fn in periodic_candidates:
                if not _base_search_can_score():
                    break
                seed_expr = (seed_fn, ("mul", ("const", float(seed_omega)), ("var", int(seed_var))))
                try:
                    _score_external_candidate_expr(
                        seed_expr,
                        parent_raw_mse=None,
                        stats=periodic_seed_stats,
                        route_name="periodic_seed",
                    )
                except Exception:
                    continue
                if not _base_search_can_score():
                    break
            if verbose:
                print(
                    "[periodic-seed] "
                    + ", ".join(f"x{j}: omega~{w:.4g}" for j, w in periodic_hints)
                )

    # --- Generalized-symmetry carrier-seed phase ---
    # Coordinates discovered by the GS layer (charts / composition / warp) are
    # scored here as carriers, so the outer-map battery fits g(z) directly for
    # coordinates the structural search would otherwise struggle to assemble.
    # Default-empty (no GS seeds) => no-op, zero behaviour change.
    gs_carrier_seed_stats: dict[str, Any] = {}
    if carrier_seed_exprs and _base_search_can_score():
        for seed_entry in carrier_seed_exprs:
            if not _base_search_can_score():
                break
            if isinstance(seed_entry, Mapping) and "expr" in seed_entry:
                seed_expr = seed_entry.get("expr")
                seed_meta = dict(seed_entry.get("metadata") or {})
            else:
                seed_expr = seed_entry
                seed_meta = mark_inner_coordinate_metadata(
                    {},
                    source="generalized_symmetry",
                    certified=True,
                )
            try:
                _score_external_candidate_expr(
                    seed_expr,
                    parent_raw_mse=None,
                    stats=gs_carrier_seed_stats,
                    route_name="gs_carrier_seed",
                    candidate_meta=seed_meta,
                )
            except Exception:
                continue
            if not _base_search_can_score():
                break
        if verbose:
            print(f"[gs-carrier-seed] scored {len(tuple(carrier_seed_exprs))} GS coordinate seed(s)")

    if _base_search_is_active():
        _run_closure_search_pass()
        _base_search_is_active()

    while search_stop_reason is None and n_evaluated < n_iter and n_attempts < max_attempts:
        if _wall_time_exceeded():
            search_stop_reason = "wall_time_limit"
            if stop_event is not None:
                stop_event.set()
            if verbose:
                print(
                    f"[mutate] wall-time limit hit at iter {int(n_evaluated)} "
                    f"elapsed={float(max(0.0, time.perf_counter() - search_started)):.2f}s"
                )
            break
        # Periodically harvest archive subtrees to expand the boost pool.
        if bool(boost_enable) and bool(boost_harvest_enable) and int(boost_pool_extra_max) > 0:
            try:
                every = int(boost_harvest_every)
            except Exception:
                every = 0
            if every > 0 and n_evaluated > 0 and (n_evaluated % every == 0) and int(last_boost_harvest_eval) != int(n_evaluated):
                last_boost_harvest_eval = int(n_evaluated)
                base_seen = set(node_str(n) for n in pool_nodes)
                dyn_nodes = harvest_pool_fn(
                    arch,
                    rng,
                    max_nodes=int(boost_pool_extra_max),
                    topk_residual_basins=int(boost_harvest_topk_residual_basins),
                    elites_per_residual_basin=int(boost_harvest_elites_per_residual_basin),
                    subtree_depth_max=int(boost_subtree_depth_max),
                    subtree_size_max=int(boost_subtree_size_max),
                    base_seen=base_seen,
                    var_dims=var_dims if dm else None,
                    target_dim=y_dims if (dm and y_dims is not None) else None,
                )

                kept_nodes = []
                dyn_phi_list = []
                dyn_phi_fit_list = []
                dyn_dims = []
                for n in dyn_nodes:
                    d = node_dims(n, var_dims) if dm else None
                    try:
                        if bool(boost_safe_eval):
                            vp, _p = eval_node_hparam_safe_fn(n, x_probe, {}, safe_cfg)
                            vf, _p2 = eval_node_hparam_safe_fn(n, x_fit, {}, safe_cfg)
                        else:
                            vp = eval_node(n, x_probe)
                            vf = eval_node(n, x_fit)
                        vp = vp.squeeze(-1)
                        vf = vf.squeeze(-1)
                    except Exception:
                        continue
                    if (not torch.isfinite(vp).all()) or (not torch.isfinite(vf).all()):
                        continue
                    kept_nodes.append(n)
                    dyn_phi_list.append(vp)
                    dyn_phi_fit_list.append(vf)
                    dyn_dims.append(d)

                if dyn_phi_list:
                    boost_dyn_nodes = kept_nodes
                    boost_dyn_dims = dyn_dims
                    boost_dyn_phi = torch.stack(dyn_phi_list, dim=1)
                    boost_dyn_norms = (boost_dyn_phi * boost_dyn_phi).sum(dim=0)
                    boost_dyn_phi_fit = torch.stack(dyn_phi_fit_list, dim=1)
                    boost_dyn_norms_fit = (boost_dyn_phi_fit * boost_dyn_phi_fit).sum(dim=0)

                    boost_pool_nodes = pool_nodes + boost_dyn_nodes
                    boost_pool_dims = pool_dims + boost_dyn_dims

                    base_phi_probe = pool_phi_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_phi
                    base_norms_probe = pool_norms_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_norms
                    base_phi_fit = pool_phi_fit_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_phi_fit
                    base_norms_fit = pool_norms_fit_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_norms_fit

                    boost_pool_phi = torch.cat([base_phi_probe, boost_dyn_phi], dim=1)
                    boost_pool_norms = torch.cat([base_norms_probe, boost_dyn_norms], dim=0)
                    boost_pool_phi_fit = torch.cat([base_phi_fit, boost_dyn_phi_fit], dim=1)
                    boost_pool_norms_fit = torch.cat([base_norms_fit, boost_dyn_norms_fit], dim=0)
                else:
                    boost_dyn_nodes = []
                    boost_dyn_dims = []
                    boost_dyn_phi = None
                    boost_dyn_norms = None
                    boost_dyn_phi_fit = None
                    boost_dyn_norms_fit = None
                    boost_pool_nodes = pool_nodes
                    boost_pool_dims = pool_dims
                    boost_pool_phi = pool_phi_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_phi
                    boost_pool_norms = pool_norms_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_norms
                    boost_pool_phi_fit = pool_phi_fit_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_phi_fit
                    boost_pool_norms_fit = pool_norms_fit_safe if (bool(boost_enable) and bool(boost_safe_eval)) else pool_norms_fit

        # Check if another thread already found a good-enough solution
        if stop_event is not None and stop_event.is_set():
            if search_stop_reason is None:
                search_stop_reason = "stop_event"
            if verbose:
                print(f"[mutate] STOPPED by another thread at iter {n_evaluated}")
            break
        n_attempts += 1
        inverse_exp_entry = None
        inverse_exp_t0 = None

        if mutation_refine_enable and refine_state is not None and mut_refines > 0:
            win_idx = int(n_evaluated // mut_window)
            if int(refine_state.get("window_idx", -1)) != win_idx:
                refine_state["window_idx"] = win_idx
                refine_state["window_trials_left"] = mut_refines

        exec_ctx = _ExecutionContext()
        prepared_hole_opp = None
        prepared_hole_resolution = None
        selected_route_name = "expression_expand"
        selected_route_source = ""
        route_diag_state = None
        route_selected_this_iter = False
        route_eval_t0 = None
        parent_key = None
        parent_rec = None
        action = None
        proposal_mode_name = None
        crossover_policy = None
        allowed_actions = None
        inverse_gate_paths = None
        inverse_gate_diag_selected = None
        repair_preview_expr = None
        repair_preview_meta = None
        repair_option_meta = None
        repair_controller_selected = False
        controller_row = None
        controller_score = 0.0
        controller_gate_score = 0.0
        controller_threshold = 0.0
        controller_components = None
        component_ok = False
        component_reasons = []
        preview_rng = None
        preview_paths = None
        preview_path_target_modes = None
        anchor_path = ()
        controller_policy_guidance = None
        critic_full = None
        repair_macro_ready = False
        selected_macro_state = None
        selected_macro_action_name = None
        macro_decision = None
        macro_stagnation_state = None
        scheduler_plan_decision = None
        scheduler_decision_log_idx = None
        scheduler_selected_candidate = None
        scheduler_selected_hole_resolution = None
        scheduler_selected_hole_budget = None
        scheduler_forced_action_path = None
        scheduler_forced_action_source = ""
        scheduler_control_applied = False
        executed_hole_opp = None
        executed_hole_wall_s = None
        executed_hole_shortlist_eff_mse = None
        executed_hole_status = None
        restart_now = (not arch.d) or (rng.random() < p_restart)
        spec_expand_selected = False

        if (not restart_now) and bool(hole_search_first_class_runtime_enable) and route_scheduler is not None:
            prepared_hole_opp, prepared_hole_resolution = _prepare_hole_search_opportunity(int(n_evaluated))
            if prepared_hole_opp is not None and prepared_hole_resolution is not None:
                hole_search_stats["first_class_scheduler_available"] = int(
                    hole_search_stats.get("first_class_scheduler_available", 0)
                ) + 1
            route_scheduler_stats["considered"] = int(route_scheduler_stats.get("considered", 0)) + 1
            available_routes = ["expression_expand"]
            route_scores = {"expression_expand": 0.0}
            route_score_details = {
                "expression_expand": {
                    "route_score": 0.0,
                    "preview_score": 0.0,
                    "learned_bonus": None,
                }
            }
            if prepared_hole_opp is not None and prepared_hole_resolution is not None:
                route_scheduler_stats["opportunity_available"] = int(
                    route_scheduler_stats.get("opportunity_available", 0)
                ) + 1
                available_routes.append("opportunity_expand")
                hole_route_eval = _hole_route_opportunity_score(
                    prepared_hole_opp,
                    prepared_hole_resolution,
                )
                route_scores["opportunity_expand"] = float(hole_route_eval.get("route_score", 0.0) or 0.0)
                route_score_details["opportunity_expand"] = dict(hole_route_eval)
            selected_route_name, selected_route_source = route_scheduler.select(
                rng,
                available_routes,
                route_scores=route_scores,
            )
            route_diag_state = _build_route_scheduler_diagnostic_state(
                available_routes=available_routes,
                route_scores=route_scores,
                route_score_details=route_score_details,
                selected_route=selected_route_name,
                selected_source=selected_route_source,
            )
            route_scheduler.record_selection(selected_route_name)
            route_selected_this_iter = True
            route_eval_t0 = time.perf_counter()
            route_scheduler_stats[f"selected_{selected_route_name}"] = int(
                route_scheduler_stats.get(f"selected_{selected_route_name}", 0)
            ) + 1
            route_scheduler_stats[f"selection_{selected_route_source}"] = int(
                route_scheduler_stats.get(f"selection_{selected_route_source}", 0)
            ) + 1
            if selected_route_name == "opportunity_expand" and prepared_hole_opp is not None and prepared_hole_resolution is not None:
                spec_expand_selected = True
                hole_search_stats["first_class_scheduler_selected"] = int(
                    hole_search_stats.get("first_class_scheduler_selected", 0)
                ) + 1
                proposal_mode_name = "spec_expand"
                resolution_source = str(prepared_hole_resolution.get("resolution_source", "") or "")
                if resolution_source == "live_archive":
                    parent_key = str(getattr(prepared_hole_opp, "parent_key", "") or "")
                    parent_rec = prepared_hole_resolution.get("rec", None)
                elif resolution_source == "snapshot":
                    opp_snapshot = prepared_hole_resolution.get("snapshot", None)
                    parent_key = str(
                        getattr(opp_snapshot, "residual_basin_key", getattr(prepared_hole_opp, "parent_key", ""))
                        or getattr(prepared_hole_opp, "parent_key", "")
                    )
                    if opp_snapshot is not None:
                        try:
                            parent_rec = SimpleNamespace(
                                best_expr=getattr(opp_snapshot, "expr", None),
                                mapping=getattr(opp_snapshot, "mapping", {}),
                                best_mse=float(getattr(opp_snapshot, "eff_mse", float("inf"))),
                                best_raw_mse=float(
                                    getattr(
                                        opp_snapshot,
                                        "raw_mse",
                                        getattr(opp_snapshot, "eff_mse", float("inf")),
                                    )
                                ),
                                best_elite_id=str(
                                    getattr(opp_snapshot, "elite_id", getattr(prepared_hole_opp, "parent_elite_id", ""))
                                    or getattr(prepared_hole_opp, "parent_elite_id", "")
                                ),
                            )
                        except Exception:
                            parent_rec = None
                else:
                    parent_key = str(getattr(prepared_hole_opp, "parent_key", "") or "")
                    parent_rec = prepared_hole_resolution.get("rec", None)
                exec_ctx.selected_parent_key = parent_key
                exec_ctx.selected_parent_rec = parent_rec
                exec_ctx.selected_parent_elite_id = str(
                    getattr(parent_rec, "best_elite_id", getattr(prepared_hole_opp, "parent_elite_id", "")) or
                    getattr(prepared_hole_opp, "parent_elite_id", "")
                )

        if restart_now:
            if dm and y_dims is not None:
                expr = rand_node_dim(rng, max_depth, var_dims, y_dims, reach)
                if expr is None: continue
            elif dm:
                # var_dims known but y_dims unknown: generate random
                # dimensionally-valid tree with an arbitrary reachable target.
                _reach_set = list(reach[min(max_depth, len(reach) - 1)]) if reach else []
                if _reach_set:
                    _rtgt = rng.choice(_reach_set)
                    expr = rand_node_dim(rng, max_depth, var_dims, _rtgt, reach)
                    if expr is None: continue
                else:
                    expr = rand_node(rng, max_depth, nvars)
            else:
                expr = rand_node(rng, max_depth, nvars)
            parent_key = None; parent_rec = None; action = None
            crossover_policy = None
        elif spec_expand_selected:
            if bool(inverse_experiment_log_enable) and parent_rec is not None:
                inverse_exp_t0 = time.perf_counter()
                inverse_exp_entry = _make_inverse_experiment_row(parent_rec, None)
                inverse_exp_entry.update(_macro_action_fields(A_HOLESEARCH, source="route_scheduler"))
                inverse_exp_entry["controller_policy_action"] = "spec_expand"
            (
                expr,
                hole_meta,
                executed_hole_opp,
                executed_hole_status,
                executed_hole_wall_s,
                executed_hole_shortlist_eff_mse,
            ) = _execute_prepared_spec_expand(
                opportunity=prepared_hole_opp,
                resolution=prepared_hole_resolution,
                current_iter=int(n_evaluated),
                exec_ctx=exec_ctx,
            )
        else:
            if bool(repair_controller_enable):
                parent_key, parent_rec = choose_parent_repair_aware(
                    arch,
                    rng,
                    exploit_frac,
                    exploit_topk,
                    n_evaluated,
                    repair_parent_cache,
                    repair_parent_state,
                    repair_controller_stats,
                )
            else:
                parent_key, parent_rec = choose_parent(arch, rng, exploit_frac, exploit_topk)
            exec_ctx.selected_parent_key = parent_key
            exec_ctx.selected_parent_rec = parent_rec
            exec_ctx.selected_parent_elite_id = str(getattr(parent_rec, "best_elite_id", "") or "")
            exec_ctx.executed_parent_key = parent_key
            exec_ctx.executed_parent_rec = parent_rec
            exec_ctx.executed_parent_elite_id = exec_ctx.selected_parent_elite_id
            try:
                exec_ctx.executed_parent_eff_mse = float(getattr(parent_rec, "best_mse", float("inf")))
            except Exception:
                exec_ctx.executed_parent_eff_mse = None

            # --- automatic gating for expensive structured proposal operators ---
            allowed_actions = None
            # Crossover needs at least two residual basins to recombine; with a
            # single-basin archive it can only spin without proposing.
            crossover_no_partner = (A_CROSSOVER in active_actions) and len(arch.d) < 2
            if crossover_no_partner:
                allowed_actions = _remove_allowed_action(allowed_actions, active_actions, A_CROSSOVER)
                cps_mask = crossover_policy_stats.get("legacy", None)
                if isinstance(cps_mask, dict):
                    cps_mask["masked_no_partner_iters"] = int(cps_mask.get("masked_no_partner_iters", 0)) + 1
            inverse_gate_paths = None
            inverse_gate_diag_selected = None
            if bool(inverse_steering_enable) and (
                (bool(inverse_gate_enable) and (A_INVSTEER in active_actions))
                or bool(repair_controller_enable)
            ):
                inverse_gate_stats["considered"] = int(inverse_gate_stats.get("considered", 0)) + 1
                ok_quality = True
                try:
                    expr_depth = int(node_depth(parent_rec.best_expr))
                    expr_size = int(node_size(parent_rec.best_expr))
                    if expr_depth < int(inverse_gate_min_depth):
                        ok_quality = False
                    if expr_size < int(inverse_gate_min_size):
                        ok_quality = False
                    if int(n_evaluated) < int(inverse_gate_warmup):
                        ok_quality = False
                    if int(len(arch.d)) < int(inverse_gate_min_residual_basins):
                        ok_quality = False
                    bf = max(1.0, float(inverse_gate_best_factor))
                    if math.isfinite(best_mse) and math.isfinite(parent_rec.best_mse):
                        if parent_rec.best_mse > best_mse * bf:
                            ok_quality = False
                except Exception:
                    ok_quality = False

                if not ok_quality:
                    inverse_gate_stats["blocked_quality"] = int(inverse_gate_stats.get("blocked_quality", 0)) + 1
                    allowed_actions = _remove_allowed_action(allowed_actions, active_actions, A_INVSTEER)
                else:
                    pool_sig = _pool_cache_signature(boost_pool_nodes)
                    cache_key = (
                        str(parent_key),
                        node_str(parent_rec.best_expr),
                        _mapping_cache_signature(parent_rec.mapping),
                        pool_sig,
                    )
                    diag = inverse_gate_cache.get(cache_key, None)
                    if diag is None:
                        diag = estimate_inverse_steering_potential(
                            parent_rec.best_expr,
                            parent_rec.mapping,
                            x_fit,
                            y_fit,
                            x_probe,
                            y_probe,
                            boost_pool_phi_fit,
                            boost_pool_phi,
                            boost_pool_dims,
                            pool_nodes=boost_pool_nodes,
                            var_dims=var_dims,
                            max_paths=inverse_gate_max_paths,
                            topk_terms=max(1, min(int(inverse_topk_terms), 4)),
                            shortlist_mult=max(1, min(int(inverse_shortlist_mult), 2)),
                            min_valid_frac=inverse_min_valid_frac,
                            min_confidence=inverse_min_confidence,
                            min_structural_score=inverse_gate_min_structural_score,
                            min_weighted_rel_gain=inverse_gate_min_weighted_rel_gain,
                            structural_bias=inverse_gate_structural_bias,
                            safe_eps=float(inverse_safe_eps),
                            confidence_mode=str(inverse_confidence_mode),
                            confidence_target_gain=float(inverse_confidence_target_gain),
                            confidence_floor=float(inverse_confidence_floor),
                            branch_beam_width=int(inverse_branch_beam_width),
                            local_score_mode=str(inverse_local_score_mode),
                            target_mode=str(inverse_target_mode),
                            full_mapping_penalty=float(inverse_full_mapping_penalty),
                            exact_simple_target_bonus=float(inverse_exact_simple_target_bonus),
                            additive_descend_penalty=float(inverse_additive_descend_penalty),
                            nonadditive_leaf_penalty=float(inverse_nonadditive_leaf_penalty),
                            periodic_min_valid_scale=float(inverse_periodic_min_valid_scale),
                            periodic_min_confidence_scale=float(inverse_periodic_min_confidence_scale),
                            periodic_path_penalty=float(inverse_periodic_path_penalty),
                            nonperiodic_muldiv_bonus=float(inverse_nonperiodic_muldiv_bonus),
                            nonperiodic_explogsqrt_bonus=float(inverse_nonperiodic_explogsqrt_bonus),
                            branch_ambiguity_penalty=float(inverse_branch_ambiguity_penalty),
                        )
                        inverse_gate_cache[cache_key] = diag
                        if len(inverse_gate_cache) > 2048:
                            try:
                                inverse_gate_cache.pop(next(iter(inverse_gate_cache)))
                            except Exception:
                                inverse_gate_cache.clear()
                    rel_hist = inverse_gate_stats.get("best_rel_gain_hist", None)
                    if isinstance(rel_hist, list):
                        rel_hist.append(float(diag.best_rel_gain))
                        if len(rel_hist) > 256:
                            del rel_hist[:-256]
                    wrg_hist = inverse_gate_stats.get("best_weighted_rel_gain_hist", None)
                    if isinstance(wrg_hist, list):
                        wrg_hist.append(float(diag.best_weighted_rel_gain))
                        if len(wrg_hist) > 256:
                            del wrg_hist[:-256]

                    if bool(repair_controller_enable):
                        inverse_gate_diag_selected = diag
                        best_path_diag = tuple(diag.best_path or ())
                        if best_path_diag:
                            inverse_gate_paths = [best_path_diag]
                        elif inverse_gate_paths is None:
                            inverse_gate_paths = list(diag.candidate_paths)

                    if bool(diag.allowed):
                        inverse_gate_stats["allowed"] = int(inverse_gate_stats.get("allowed", 0)) + 1
                        if inverse_gate_paths is None:
                            inverse_gate_paths = list(diag.candidate_paths)
                        if inverse_gate_diag_selected is None:
                            inverse_gate_diag_selected = diag
                    else:
                        reason = str(diag.reason)
                        if reason in ("no_structural_paths", "no_viable_paths"):
                            inverse_gate_stats["blocked_structure"] = int(inverse_gate_stats.get("blocked_structure", 0)) + 1
                        else:
                            inverse_gate_stats["blocked_gain"] = int(inverse_gate_stats.get("blocked_gain", 0)) + 1
                        if bool(inverse_gate_enable):
                            allowed_actions = _remove_allowed_action(allowed_actions, active_actions, A_INVSTEER)

            if bool(boost_enable) and bool(boost_gate_enable) and (A_BOOST in active_actions):
                boost_gate_stats["considered"] = int(boost_gate_stats.get("considered", 0)) + 1
                ok_quality = True
                try:
                    if int(n_evaluated) < int(boost_gate_warmup):
                        ok_quality = False
                    if int(len(arch.d)) < int(boost_gate_min_residual_basins):
                        ok_quality = False
                    bf = max(1.0, float(boost_gate_best_factor))
                    if math.isfinite(best_mse) and math.isfinite(parent_rec.best_mse):
                        if parent_rec.best_mse > best_mse * bf:
                            ok_quality = False
                except Exception:
                    ok_quality = False

                if not ok_quality:
                    boost_gate_stats["blocked_quality"] = int(boost_gate_stats.get("blocked_quality", 0)) + 1
                    allowed_actions = _remove_allowed_action(allowed_actions, active_actions, A_BOOST)

            if bool(hole_search_first_class_runtime_enable):
                allowed_actions = _remove_allowed_action(allowed_actions, active_actions, A_HOLESEARCH)
            else:
                prepared_hole_opp, prepared_hole_resolution = _prepare_hole_search_opportunity(int(n_evaluated))
                if prepared_hole_opp is None:
                    allowed_actions = _remove_allowed_action(allowed_actions, active_actions, A_HOLESEARCH)
                if route_scheduler is not None and (not bool(scheduler_control_enabled)):
                    route_scheduler_stats["considered"] = int(route_scheduler_stats.get("considered", 0)) + 1
                    available_routes = ["expression_expand"]
                    route_scores = {"expression_expand": 0.0}
                    route_score_details = {
                        "expression_expand": {
                            "route_score": 0.0,
                            "preview_score": 0.0,
                            "learned_bonus": None,
                        }
                    }
                    if prepared_hole_opp is not None and prepared_hole_resolution is not None:
                        route_scheduler_stats["opportunity_available"] = int(
                            route_scheduler_stats.get("opportunity_available", 0)
                        ) + 1
                        available_routes.append("opportunity_expand")
                        hole_route_eval = _hole_route_opportunity_score(
                            prepared_hole_opp,
                            prepared_hole_resolution,
                        )
                        route_scores["opportunity_expand"] = float(hole_route_eval.get("route_score", 0.0) or 0.0)
                        route_score_details["opportunity_expand"] = dict(hole_route_eval)
                    selected_route_name, selected_route_source = route_scheduler.select(
                        rng,
                        available_routes,
                        route_scores=route_scores,
                    )
                    route_diag_state = _build_route_scheduler_diagnostic_state(
                        available_routes=available_routes,
                        route_scores=route_scores,
                        route_score_details=route_score_details,
                        selected_route=selected_route_name,
                        selected_source=selected_route_source,
                    )
                    route_scheduler.record_selection(selected_route_name)
                    route_selected_this_iter = True
                    route_eval_t0 = time.perf_counter()
                    route_scheduler_stats[f"selected_{selected_route_name}"] = int(
                        route_scheduler_stats.get(f"selected_{selected_route_name}", 0)
                    ) + 1
                    route_scheduler_stats[f"selection_{selected_route_source}"] = int(
                        route_scheduler_stats.get(f"selection_{selected_route_source}", 0)
                    ) + 1
                    if selected_route_name == "expression_expand":
                        allowed_actions = _remove_allowed_action(allowed_actions, active_actions, A_HOLESEARCH)
            if (selected_route_name != "opportunity_expand") and bool(repair_controller_enable):
                retry_ok = True
                retry_reason = "ok"
                if inverse_gate_diag_selected is not None:
                    retry_ok, retry_reason = _repair_parent_retry_gate(
                        parent_key,
                        parent_rec,
                        n_evaluated,
                        repair_parent_state,
                        repair_controller_stats,
                    )
                    if not retry_ok:
                        repair_controller_stats[f"blocked_retry_{retry_reason}"] = int(repair_controller_stats.get(f"blocked_retry_{retry_reason}", 0)) + 1
                if inverse_gate_diag_selected is not None and retry_ok:
                    repair_controller_stats["considered"] = int(repair_controller_stats.get("considered", 0)) + 1
                    if bool(inverse_experiment_log_enable):
                        inverse_exp_t0 = time.perf_counter()
                        inverse_exp_entry = _make_inverse_experiment_row(parent_rec, inverse_gate_diag_selected)
                        inverse_exp_entry.update(_macro_action_fields(A_REPAIR, source="repair_controller"))
                        inverse_exp_entry["proposal_generator_action_id"] = int(A_INVSTEER)
                        inverse_exp_entry["proposal_generator_action"] = str(ACTION_NAME.get(A_INVSTEER, "inv_steer"))
                    stagnation_state = _repair_controller_stagnation_state(parent_rec, repair_controller_stats)
                    if isinstance(inverse_exp_entry, dict):
                        inverse_exp_entry.update({
                            "parent_visits": float(stagnation_state.get("visits", 0.0)),
                            "parent_visits_since_improve": float(stagnation_state.get("visits_since_improve", 0.0)),
                            "parent_stagnation_score": float(stagnation_state.get("stagnation_score", 0.0)),
                            "parent_stagnation_ratio": float(stagnation_state.get("stagnation_ratio", 0.0)),
                        })
                    try:
                        anchor_path = tuple(inverse_gate_diag_selected.best_path or ())
                    except Exception:
                        anchor_path = ()
                    analytic_anchor_path = tuple(anchor_path)
                    preview_paths_fallback = _repair_option_candidate_paths(
                        parent_rec.best_expr,
                        anchor_path,
                        ancestor_hops=int(repair_controller_ancestor_hops),
                        include_ancestors=False,
                        fallback_paths=inverse_gate_paths,
                    )
                    preview_paths = list(preview_paths_fallback)
                    preview_path_target_modes = None
                    preview_guidance_source = "analytic"
                    preview_guidance_target_mode = ""
                    preview_guidance_relation = ""
                    preview_guidance_improve = 0.0
                    preview_guidance_path_prob = 0.0
                    controller_policy_row = None
                    if repair_critic_bundle is not None:
                        try:
                            controller_policy_row = _make_inverse_experiment_record(
                                parent_rec,
                                inverse_gate_diag_selected,
                                stagnation_state=stagnation_state,
                            )
                            controller_policy_guidance = _repair_controller_path_policy(
                                predict_repair_controller_heads(repair_critic_bundle, controller_policy_row),
                                fallback_path=anchor_path,
                                fallback_paths=preview_paths_fallback,
                                max_paths=max(1, min(int(inverse_max_paths), 3)),
                            )
                        except Exception as exc:
                            repair_controller_stats["critic_policy_predict_error"] = str(exc)
                            controller_policy_guidance = None
                    learned_anchor_path = tuple(anchor_path)
                    learned_preview_paths = list(preview_paths_fallback)
                    learned_preview_path_target_modes = None
                    learned_guidance_source = "analytic"
                    learned_guidance_target_mode = ""
                    learned_guidance_relation = ""
                    learned_guidance_improve = 0.0
                    learned_guidance_path_prob = 0.0
                    if isinstance(controller_policy_guidance, dict) and bool(controller_policy_guidance.get("trained", False)):
                        try:
                            policy_path = tuple(int(v) for v in (controller_policy_guidance.get("selected_path", None) or ()))
                        except Exception:
                            policy_path = ()
                        if policy_path:
                            learned_anchor_path = policy_path
                        learned_preview_paths = [
                            tuple(int(v) for v in path)
                            for path in list(controller_policy_guidance.get("candidate_paths", []) or [])
                            if path
                        ]
                        learned_preview_path_target_modes = {
                            tuple(int(v) for v in path): str(mode)
                            for path, mode in dict(controller_policy_guidance.get("path_target_modes", {}) or {}).items()
                            if path and str(mode)
                        }
                        if learned_preview_paths:
                            inverse_gate_paths = list(learned_preview_paths)
                        learned_guidance_source = str(controller_policy_guidance.get("source", "critic_path_head") or "critic_path_head")
                        learned_guidance_target_mode = str(controller_policy_guidance.get("best_target_mode", "") or "")
                        learned_guidance_relation = str(controller_policy_guidance.get("best_relation", "") or "")
                        try:
                            learned_guidance_improve = float(controller_policy_guidance.get("best_improvement_estimate", 0.0))
                        except Exception:
                            learned_guidance_improve = 0.0
                        try:
                            policy_rows = list(controller_policy_guidance.get("rows", []) or [])
                            learned_guidance_path_prob = float(policy_rows[0].get("prob", 0.0)) if policy_rows else 0.0
                        except Exception:
                            learned_guidance_path_prob = 0.0
                    policy_priority_bonus = 0.0
                    if isinstance(controller_policy_guidance, dict) and bool(controller_policy_guidance.get("trained", False)):
                        try:
                            priority_weight = max(0.0, float(repair_controller_stats.get("policy_priority_weight", 0.0)))
                        except Exception:
                            priority_weight = 0.0
                        try:
                            priority_cap = max(0.0, float(repair_controller_stats.get("policy_priority_cap", 0.0)))
                        except Exception:
                            priority_cap = 0.0
                        try:
                            policy_rows = list(controller_policy_guidance.get("rows", []) or [])
                            best_policy_score = float(policy_rows[0].get("policy_score", 0.0)) if policy_rows else 0.0
                        except Exception:
                            best_policy_score = 0.0
                        if priority_weight > 0.0 and best_policy_score > 0.0:
                            policy_priority_bonus = priority_weight * best_policy_score
                            if priority_cap > 0.0:
                                policy_priority_bonus = min(priority_cap, policy_priority_bonus)

                    def _run_controller_preview(
                        candidate_paths_local,
                        path_target_modes_local,
                        *,
                        repair_opportunity_controller_enable_local=None,
                    ):
                        use_opportunity_controller = (
                            bool(repair_opportunity_controller_enable)
                            if repair_opportunity_controller_enable_local is None
                            else bool(repair_opportunity_controller_enable_local)
                        )
                        local_rng = random.Random()
                        local_rng.setstate(rng.getstate())
                        preview_ret = apply_inverse_steering_action(
                            parent_rec.best_expr,
                            parent_rec.mapping,
                            x_fit,
                            y_fit,
                            x_probe,
                            y_probe,
                            boost_pool_nodes,
                            boost_pool_phi_fit,
                            boost_pool_phi,
                            boost_pool_dims,
                            local_rng,
                            max_depth,
                            nvars,
                            poly_degree,
                            var_dims=var_dims,
                            max_paths=inverse_max_paths,
                            topk_terms=inverse_topk_terms,
                            shortlist_mult=inverse_shortlist_mult,
                            min_valid_frac=inverse_min_valid_frac,
                            min_confidence=inverse_min_confidence,
                            safe_eps=float(inverse_safe_eps),
                            confidence_mode=str(inverse_confidence_mode),
                            confidence_target_gain=float(inverse_confidence_target_gain),
                            confidence_floor=float(inverse_confidence_floor),
                            branch_beam_width=int(inverse_branch_beam_width),
                            micro_search_enable=bool(inverse_micro_search_enable),
                            micro_search_max_depth=int(inverse_micro_search_max_depth),
                            micro_search_beam_width=int(inverse_micro_search_beam_width),
                            micro_search_topk=int(inverse_micro_search_topk),
                            micro_search_seed_terms=int(inverse_micro_search_seed_terms),
                            local_score_mode=str(inverse_local_score_mode),
                            inverse_spec_enable=bool(inverse_spec_enable),
                            inverse_spec_enum_max_depth=int(inverse_spec_enum_max_depth),
                            inverse_spec_enum_max_trees=int(inverse_spec_enum_max_trees),
                            inverse_spec_preview_topk=int(inverse_spec_preview_topk),
                            inverse_spec_local_score_mode=str(inverse_spec_local_score_mode),
                            inverse_spec_include_legacy_seed=bool(inverse_spec_include_legacy_seed),
                            inverse_spec_complexity_penalty=float(inverse_spec_complexity_penalty),
                            inverse_spec_family_battery_enable=bool(inverse_spec_family_battery_enable),
                            inverse_spec_family_battery_mode=str(inverse_spec_family_battery_mode or "outer"),
                            inverse_spec_repair_quota=float(inverse_spec_repair_quota),
                            inverse_spec_recursive_enable=bool(inverse_spec_recursive_enable),
                            inverse_spec_recursive_max_depth=int(inverse_spec_recursive_max_depth),
                            inverse_spec_recursive_trigger_rel_mse=float(inverse_spec_recursive_trigger_rel_mse),
                            inverse_spec_recursive_seed_cap=int(inverse_spec_recursive_seed_cap),
                            inverse_spec_recursive_branch_topk=int(inverse_spec_recursive_branch_topk),
                            inverse_spec_recursive_child_topk=int(inverse_spec_recursive_child_topk),
                            inverse_spec_witness_jets_enable=bool(inverse_spec_witness_jets_enable),
                            inverse_spec_witness_d2_enable=bool(inverse_spec_witness_d2_enable),
                            inverse_spec_witness_max_rows=int(inverse_spec_witness_max_rows),
                            inverse_spec_witness_loss_enable=bool(inverse_spec_witness_loss_enable),
                            inverse_spec_witness_grad_weight=float(inverse_spec_witness_grad_weight),
                            inverse_spec_witness_d2_weight=float(inverse_spec_witness_d2_weight),
                            inverse_spec_witness_diag_weight=float(inverse_spec_witness_diag_weight),
                            inverse_spec_witness_physics_weight=float(inverse_spec_witness_physics_weight),
                            inverse_spec_active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
                            inverse_spec_active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
                            inverse_spec_active_var_max_count=int(inverse_spec_active_var_max_count),
                            inverse_spec_directional_market_enable=bool(inverse_spec_directional_market_enable),
                            inverse_spec_max_subtree_depth=inverse_spec_max_subtree_depth,
                            inverse_spec_fit_cap=int(inverse_spec_fit_cap),
                            inverse_spec_probe_cap=int(inverse_spec_probe_cap),
                            inverse_spec_exact_budget=int(inverse_spec_exact_budget),
                            target_mode=str(inverse_target_mode),
                            full_mapping_penalty=float(inverse_full_mapping_penalty),
                            exact_simple_target_bonus=float(inverse_exact_simple_target_bonus),
                            additive_descend_penalty=float(inverse_additive_descend_penalty),
                            nonadditive_leaf_penalty=float(inverse_nonadditive_leaf_penalty),
                            exact_path_eta=float(inverse_exact_path_eta),
                            exact_transport_min_lin_rel=float(inverse_exact_transport_min_lin_rel),
                            periodic_min_valid_scale=float(inverse_periodic_min_valid_scale),
                            periodic_min_confidence_scale=float(inverse_periodic_min_confidence_scale),
                            periodic_path_penalty=float(inverse_periodic_path_penalty),
                            nonperiodic_muldiv_bonus=float(inverse_nonperiodic_muldiv_bonus),
                            nonperiodic_explogsqrt_bonus=float(inverse_nonperiodic_explogsqrt_bonus),
                            branch_ambiguity_penalty=float(inverse_branch_ambiguity_penalty),
                            transport_min_lin_rel=float(inverse_transport_min_lin_rel),
                            transport_min_effective_n=float(inverse_transport_min_effective_n),
                            complexity_penalty=complexity_penalty,
                            candidate_paths=(candidate_paths_local if candidate_paths_local else None),
                            path_target_modes=path_target_modes_local,
                            proj=proj,
                            fp_mode=fp_mode,
                            q_scale=q_scale,
                            q_clip=q_clip,
                            score_expr_cfg=refine_cfg,
                            return_meta=True,
                            repair_tuple_bundle=repair_critic_bundle,
                            repair_tuple_controller_row=controller_policy_row,
                            repair_opportunity_controller_enable=bool(use_opportunity_controller),
                            repair_opportunity_bundle=repair_opportunity_bundle,
                            inverse_spec_regime_metadata=inverse_spec_regime_metadata,
                        )
                        return local_rng, preview_ret

                    analytic_preview_rng, analytic_preview_ret = _run_controller_preview(
                        preview_paths_fallback,
                        None,
                    )
                    analytic_preview_expr, analytic_preview_meta = analytic_preview_ret
                    learned_preview_rng = None
                    learned_preview_expr = None
                    learned_preview_meta = None
                    if learned_guidance_source != "analytic":
                        learned_preview_rng, learned_preview_ret = _run_controller_preview(
                            learned_preview_paths,
                            learned_preview_path_target_modes,
                        )
                        learned_preview_expr, learned_preview_meta = learned_preview_ret

                    use_repair_slate_ranker = bool(
                        isinstance(repair_critic_bundle, Mapping)
                        and repair_critic_bundle.get("repair_slate_ranker_trained", False)
                    )
                    gate_preview_expr = analytic_preview_expr
                    gate_preview_meta = analytic_preview_meta
                    gate_preview_source = "analytic"
                    repair_preview_choice = _choose_repair_execution_preview(
                        analytic_preview_expr=analytic_preview_expr,
                        analytic_preview_meta=analytic_preview_meta,
                        analytic_preview_rng=analytic_preview_rng,
                        analytic_anchor_path=analytic_anchor_path,
                        analytic_preview_paths=preview_paths_fallback,
                        analytic_preview_path_target_modes=None,
                        learned_preview_expr=learned_preview_expr,
                        learned_preview_meta=learned_preview_meta,
                        learned_preview_rng=learned_preview_rng,
                        learned_anchor_path=learned_anchor_path,
                        learned_preview_paths=learned_preview_paths,
                        learned_preview_path_target_modes=learned_preview_path_target_modes,
                        learned_preview_source=learned_guidance_source,
                    )
                    if not use_repair_slate_ranker and repair_preview_choice.get("source") not in ("", "analytic"):
                        gate_preview_expr = repair_preview_choice.get("expr", analytic_preview_expr)
                        gate_preview_meta = repair_preview_choice.get("meta", analytic_preview_meta)
                        gate_preview_source = str(repair_preview_choice.get("source", "analytic"))
                    route_compare_preview_expr = gate_preview_expr
                    route_compare_preview_meta = gate_preview_meta
                    route_compare_preview_source = gate_preview_source
                    if (
                        repair_route_compare_bundle is not None
                        and bool(repair_opportunity_controller_enable)
                        and (not bool(scheduler_middle_loop_control_enabled))
                    ):
                        route_compare_analytic_preview_rng, route_compare_analytic_preview_ret = _run_controller_preview(
                            preview_paths_fallback,
                            None,
                            repair_opportunity_controller_enable_local=False,
                        )
                        route_compare_analytic_preview_expr, route_compare_analytic_preview_meta = route_compare_analytic_preview_ret
                        route_compare_learned_preview_rng = None
                        route_compare_learned_preview_expr = None
                        route_compare_learned_preview_meta = None
                        if learned_guidance_source != "analytic":
                            route_compare_learned_preview_rng, route_compare_learned_preview_ret = _run_controller_preview(
                                learned_preview_paths,
                                learned_preview_path_target_modes,
                                repair_opportunity_controller_enable_local=False,
                            )
                            route_compare_learned_preview_expr, route_compare_learned_preview_meta = route_compare_learned_preview_ret
                        route_compare_preview_choice = _choose_repair_execution_preview(
                            analytic_preview_expr=route_compare_analytic_preview_expr,
                            analytic_preview_meta=route_compare_analytic_preview_meta,
                            analytic_preview_rng=route_compare_analytic_preview_rng,
                            analytic_anchor_path=analytic_anchor_path,
                            analytic_preview_paths=preview_paths_fallback,
                            analytic_preview_path_target_modes=None,
                            learned_preview_expr=route_compare_learned_preview_expr,
                            learned_preview_meta=route_compare_learned_preview_meta,
                            learned_preview_rng=route_compare_learned_preview_rng,
                            learned_anchor_path=learned_anchor_path,
                            learned_preview_paths=learned_preview_paths,
                            learned_preview_path_target_modes=learned_preview_path_target_modes,
                            learned_preview_source=learned_guidance_source,
                        )
                        route_compare_preview_expr = route_compare_preview_choice.get("expr", route_compare_analytic_preview_expr)
                        route_compare_preview_meta = route_compare_preview_choice.get("meta", route_compare_analytic_preview_meta)
                        route_compare_preview_source = str(route_compare_preview_choice.get("source", "analytic") or "analytic")
                    repair_preview_expr = repair_preview_choice.get("expr", analytic_preview_expr)
                    repair_preview_meta = repair_preview_choice.get("meta", analytic_preview_meta)
                    preview_rng = repair_preview_choice.get("rng", analytic_preview_rng)
                    anchor_path = tuple(int(v) for v in (repair_preview_choice.get("anchor_path", analytic_anchor_path) or ()))
                    preview_paths = [
                        tuple(int(v) for v in path)
                        for path in list(repair_preview_choice.get("preview_paths", preview_paths_fallback) or [])
                        if path
                    ]
                    preview_path_target_modes = dict(repair_preview_choice.get("path_target_modes", {}) or {}) or None
                    preview_guidance_source = str(repair_preview_choice.get("source", "analytic") or "analytic")
                    preview_guidance_target_mode = (
                        learned_guidance_target_mode if preview_guidance_source == learned_guidance_source else ""
                    )
                    preview_guidance_relation = (
                        learned_guidance_relation if preview_guidance_source == learned_guidance_source else ""
                    )
                    preview_guidance_improve = (
                        float(learned_guidance_improve) if preview_guidance_source == learned_guidance_source else 0.0
                    )
                    preview_guidance_path_prob = (
                        float(learned_guidance_path_prob) if preview_guidance_source == learned_guidance_source else 0.0
                    )
                    need_controller_build_slate = bool(controller_build_slate_enable) or (
                        bool(repair_controller_route_compare_enable)
                        and (not bool(scheduler_middle_loop_control_enabled))
                    )
                    controller_build_slate_payload = {}
                    if bool(need_controller_build_slate):
                        controller_build_slate_payload = _collect_controller_build_slate(
                            parent_key=parent_key,
                            parent_rec=parent_rec,
                            n_evaluated=int(n_evaluated),
                            seed_search=seed_search,
                            active_actions=active_actions,
                            action_names=controller_build_slate_actions,
                            max_actions=int(controller_build_slate_max_actions),
                            controller_policy_guidance=controller_policy_guidance,
                            macro_decision=None,
                            macro_state=None,
                            inverse_gate_diag=inverse_gate_diag_selected,
                            x_fit=x_fit,
                            y_fit=y_fit,
                            x_probe=x_probe,
                            y_probe=y_probe,
                            proj=proj,
                            fp_mode=fp_mode,
                            q_scale=q_scale,
                            q_clip=q_clip,
                            poly_degree=poly_degree,
                            refine_enable=controller_slate_refine_enable,
                            refine_cfg=refine_cfg,
                            refine_state=refine_state,
                            best_raw_mse_struct=float(best_raw_mse_struct),
                            best_raw_mse=float(best_raw_mse),
                            early_stop_mse=float(early_stop_mse),
                            complexity_penalty=float(complexity_penalty),
                            boost_enable=bool(boost_enable),
                            boost_pool_nodes=boost_pool_nodes,
                            boost_pool_phi_fit=boost_pool_phi_fit,
                            boost_pool_norms_fit=boost_pool_norms_fit,
                            boost_pool_phi=boost_pool_phi,
                            boost_pool_norms=boost_pool_norms,
                            boost_pool_dims=boost_pool_dims,
                            boost_selection_split=str(boost_selection_split),
                            boost_ridge=float(boost_ridge) if boost_ridge is not None else None,
                            boost_include_parent=bool(boost_include_parent),
                            boost_from_scratch_prob=float(boost_from_scratch_prob),
                            boost_prune_rel=float(boost_prune_rel),
                            boost_max_terms=int(boost_max_terms),
                            boost_topk_try=int(boost_topk_try),
                            boost_min_rel_improve=float(boost_min_rel_improve),
                            max_depth=int(max_depth),
                            nvars=int(nvars),
                            var_dims=var_dims,
                            y_dims=y_dims,
                            reach=reach,
                            score_expr_fn=score_expr_fn,
                            preview_only=bool(scheduler_build_preview_only),
                        )

                    if isinstance(inverse_exp_entry, dict) and isinstance(gate_preview_meta, dict):
                        _merge_inverse_proposal_log_fields(
                            inverse_exp_entry,
                            gate_preview_meta,
                            status_key="controller_preview_status",
                        )
                    parent_eff = float(getattr(parent_rec, "best_mse", float("inf")))
                    child_eff = None
                    if isinstance(gate_preview_meta, dict):
                        child_eff = gate_preview_meta.get("estimated_child_eff_mse", None)
                    try:
                        child_eff_f = float(child_eff)
                    except Exception:
                        child_eff_f = None
                    if child_eff_f is not None and math.isfinite(parent_eff) and parent_eff > 1.0e-30 and math.isfinite(child_eff_f):
                        proxy_potential = max(0.0, parent_eff - child_eff_f) / parent_eff
                    else:
                        proxy_potential = None
                    controller_row = _make_inverse_experiment_record(
                        parent_rec,
                        inverse_gate_diag_selected,
                        stagnation_state=stagnation_state,
                        candidate_meta=gate_preview_meta,
                        proxy_potential=proxy_potential,
                    )
                    if isinstance(inverse_exp_entry, dict):
                        inverse_exp_entry.update(controller_row.to_flat_dict())
                        inverse_exp_entry.update({
                            "controller_policy_path_trained": bool((controller_policy_guidance or {}).get("trained", False)),
                            "controller_policy_gate_preview_source": str(gate_preview_source),
                            "controller_policy_preview_source": str(preview_guidance_source),
                            "controller_policy_best_path": [int(v) for v in (controller_policy_guidance or {}).get("selected_path", []) or []],
                            "controller_policy_best_target_mode": str(preview_guidance_target_mode),
                            "controller_policy_best_relation": str(preview_guidance_relation),
                            "controller_policy_best_improvement_estimate": float(preview_guidance_improve),
                            "controller_policy_best_path_prob": float(preview_guidance_path_prob),
                            "controller_policy_priority_bonus": float(policy_priority_bonus),
                            "controller_policy_candidate_paths": [
                                [int(v) for v in path]
                                for path in list(preview_paths or [])
                            ],
                            "controller_policy_path_target_modes": [
                                {
                                    "path": [int(v) for v in path],
                                    "target_mode": str(mode),
                                }
                                for path, mode in sorted(
                                    dict(preview_path_target_modes or {}).items(),
                                    key=lambda item: tuple(item[0]),
                                )
                                if path and str(mode)
                            ],
                            "controller_policy_exec_child_eff_mse": float(repair_preview_choice.get("child_eff_mse", float("inf"))),
                            "controller_policy_exec_vs_analytic_rel_gain": float(repair_preview_choice.get("relative_gain_vs_analytic", 0.0)),
                        })
                        if controller_build_slate_payload:
                            inverse_exp_entry.update(controller_build_slate_payload)
                    analytic_controller_score, controller_components = _analytic_repair_controller_score(
                        controller_row,
                        repair_controller_stats,
                    )
                    critic_preds = None
                    critic_full = None
                    critic_score = None
                    critic_blend = 0.0
                    if repair_critic_bundle is not None:
                        try:
                            critic_full = predict_repair_controller_heads(repair_critic_bundle, controller_row)
                            critic_preds = dict((critic_full or {}).get("auxiliary", {}) or {})
                            critic_score = float(critic_preds.get("utility_score", float("nan")))
                        except Exception as exc:
                            repair_controller_stats["critic_predict_error"] = str(exc)
                            critic_full = None
                            critic_preds = None
                            critic_score = None
                    if critic_score is not None and math.isfinite(float(critic_score)):
                        try:
                            critic_blend = min(1.0, max(0.0, float(repair_controller_critic_blend)))
                        except Exception:
                            critic_blend = 1.0
                    else:
                        critic_blend = 0.0
                    hybrid_controller = _hybrid_repair_controller_scores(
                        float(analytic_controller_score),
                        critic_preds,
                        float(critic_blend),
                        str(repair_controller_critic_mode),
                    )
                    controller_gate_score = float(hybrid_controller.get("gate_score", analytic_controller_score))
                    controller_score = float(hybrid_controller.get("priority_score", controller_gate_score))
                    controller_critic_bonus = float(hybrid_controller.get("critic_bonus", 0.0))
                    controller_threshold_base = _repair_controller_threshold(repair_controller_stats)
                    controller_threshold = max(
                        0.0,
                        float(controller_threshold_base) + float(hybrid_controller.get("threshold_shift", 0.0)),
                    )
                    inverse_exp_entry = dict(inverse_exp_entry or {})
                    weights_cfg = _repair_controller_weights(repair_controller_stats)
                    inverse_exp_entry.update({
                        "controller_enabled": True,
                        "controller_selected": False,
                        "controller_score": float(controller_score),
                        "controller_gate_score": float(controller_gate_score),
                        "controller_analytic_score": float(analytic_controller_score),
                        "controller_threshold_base": float(controller_threshold_base),
                        "controller_threshold": float(controller_threshold),
                        "controller_potential": float(controller_components.get("potential", 0.0)),
                        "controller_concentration": float(controller_components.get("concentration", 0.0)),
                        "controller_contrast": float(controller_components.get("contrast", 0.0)),
                        "controller_cost": float(controller_components.get("cost", 0.0)),
                        "controller_stagnation": float(controller_components.get("stagnation", 0.0)),
                        "controller_critic_bonus": float(controller_critic_bonus),
                        "controller_critic_signal": float(hybrid_controller.get("critic_signal", 0.0)),
                        "controller_critic_signed_signal": float(hybrid_controller.get("critic_signed_signal", 0.0)),
                        "controller_critic_gate_delta": float(hybrid_controller.get("critic_gate_delta", 0.0)),
                        "controller_critic_priority_delta": float(hybrid_controller.get("critic_priority_delta", 0.0)),
                        "controller_critic_threshold_shift": float(hybrid_controller.get("threshold_shift", 0.0)),
                        "controller_critic_mode": str(hybrid_controller.get("mode", repair_controller_critic_mode)),
                        "controller_score_source": str(hybrid_controller.get("source", "analytic")),
                        "controller_critic_blend": float(critic_blend),
                    })
                    if isinstance(critic_preds, dict):
                        inverse_exp_entry.update({
                            "controller_critic_score": float(critic_score if critic_score is not None else 0.0),
                            "controller_critic_accept_prob": float(critic_preds.get("accept_prob", 0.0)),
                            "controller_critic_positive_reward_prob": float(critic_preds.get("positive_reward_prob", 0.0)),
                            "controller_critic_new_residual_basin_prob": float(critic_preds.get("new_residual_basin_prob", 0.0)),
                            "controller_critic_new_best_prob": float(critic_preds.get("new_best_prob", 0.0)),
                            "controller_critic_reward_per_s_score": float(critic_preds.get("reward_per_s_score", 0.0)),
                        })
                    if isinstance(critic_full, dict):
                        path_policy_post = dict((critic_full or {}).get("path", {}) or {})
                        inverse_exp_entry.update({
                            "controller_policy_path_head_trained": bool(path_policy_post.get("trained", False)),
                            "controller_policy_post_best_path": [int(v) for v in (path_policy_post.get("best_path", []) or [])],
                            "controller_policy_post_best_target_mode": str(path_policy_post.get("best_target_mode", "") or ""),
                        })
                    route_compare_pred = None
                    route_compare_decision = None
                    if repair_route_compare_bundle is not None and (not bool(scheduler_middle_loop_control_enabled)):
                        route_compare_row = dict(controller_row.to_flat_dict())
                        if isinstance(route_compare_preview_meta, dict):
                            _merge_inverse_proposal_log_fields(
                                route_compare_row,
                                route_compare_preview_meta,
                                status_key="controller_preview_status",
                            )
                        if controller_build_slate_payload:
                            route_compare_row.update(controller_build_slate_payload)
                        try:
                            route_compare_pred = predict_repair_build_route(
                                repair_route_compare_bundle,
                                route_compare_row,
                                repair_tuple_bundle=route_compare_repair_tuple_bundle if isinstance(route_compare_repair_tuple_bundle, dict) and bool(route_compare_repair_tuple_bundle.get("repair_tuple_ranker_trained", False)) else (repair_critic_bundle if isinstance(repair_critic_bundle, dict) and bool(repair_critic_bundle.get("repair_tuple_ranker_trained", False)) else None),
                                build_tuple_bundle=build_tuple_bundle if isinstance(build_tuple_bundle, dict) and bool(build_tuple_bundle.get("build_tuple_ranker_trained", False)) else None,
                            )
                            repair_unseen_pred = None
                            if bool(repair_controller_credible_route_enable) and isinstance(repair_opportunity_bundle, dict) and isinstance(route_compare_preview_meta, dict):
                                try:
                                    repair_opportunity_rows = _credible_route_preview_repair_opportunity_rows(route_compare_preview_meta)
                                    if repair_opportunity_rows:
                                        repair_unseen_pred = predict_opportunity_slate(
                                            repair_opportunity_bundle,
                                            repair_opportunity_rows,
                                        )
                                except Exception as exc:
                                    repair_controller_stats["credible_route_predict_error"] = str(exc)
                                    repair_unseen_pred = None
                            route_compare_decision = _credible_route_compare_decision(
                                route_compare_pred,
                                repair_unseen_pred,
                                credible_route_enable=bool(repair_controller_credible_route_enable),
                                macro_enabled=bool(macro_controller_enable),
                                max_repair_prob=float(repair_controller_route_compare_max_repair_prob),
                                min_build_margin=float(repair_controller_route_compare_min_build_margin),
                            )
                        except Exception as exc:
                            repair_controller_stats["route_compare_predict_error"] = str(exc)
                            route_compare_pred = None
                            route_compare_decision = None
                    if isinstance(inverse_exp_entry, dict) and isinstance(route_compare_decision, dict):
                        inverse_exp_entry.update({
                            "controller_route_compare_trained": bool(route_compare_decision.get("trained", False)),
                            "controller_route_compare_preview_source": str(route_compare_preview_source or ""),
                            "controller_route_compare_best_route": str(route_compare_decision.get("best_route", "") or ""),
                            "controller_route_compare_repair_prob": float(route_compare_decision.get("repair_prob", 0.0) or 0.0),
                            "controller_route_compare_build_prob": float(route_compare_decision.get("build_prob", 0.0) or 0.0),
                            "controller_route_compare_margin_estimate": float(route_compare_decision.get("margin_estimate", 0.0) or 0.0),
                            "controller_route_compare_exact_margin": route_compare_decision.get("exact_margin", None),
                            "controller_route_compare_veto_repair": bool(route_compare_decision.get("veto_repair", False)),
                            "controller_route_compare_source": str(route_compare_decision.get("source", "") or ""),
                            "controller_route_compare_credible_enabled": bool(route_compare_decision.get("credible_route_enable", False)),
                            "controller_route_compare_credible_used": bool(route_compare_decision.get("credible_route_used", False)),
                            "controller_route_compare_legacy_best_route": str(route_compare_decision.get("legacy_best_route", "") or ""),
                            "controller_route_compare_legacy_repair_prob": float(route_compare_decision.get("legacy_repair_prob", 0.0) or 0.0),
                            "controller_route_compare_legacy_build_prob": float(route_compare_decision.get("legacy_build_prob", 0.0) or 0.0),
                            "controller_route_compare_legacy_margin_estimate": float(route_compare_decision.get("legacy_margin_estimate", 0.0) or 0.0),
                            "controller_route_compare_repair_observed_score": route_compare_decision.get("repair_observed_score", None),
                            "controller_route_compare_build_observed_score": route_compare_decision.get("build_observed_score", None),
                            "controller_route_compare_repair_unseen_trained": bool(route_compare_decision.get("repair_unseen_trained", False)),
                            "controller_route_compare_repair_unseen_upside": float(route_compare_decision.get("repair_unseen_upside", 0.0) or 0.0),
                            "controller_route_compare_repair_unseen_best_opportunity_id": str(route_compare_decision.get("repair_unseen_best_opportunity_id", "") or ""),
                            "controller_route_compare_repair_unseen_best_beam_id": str(route_compare_decision.get("repair_unseen_best_beam_id", "") or ""),
                            "controller_route_compare_repair_unseen_best_acquisition_estimate": float(route_compare_decision.get("repair_unseen_best_acquisition_estimate", 0.0) or 0.0),
                            "controller_route_compare_repair_unseen_best_expected_gain_next_under_executor": float(route_compare_decision.get("repair_unseen_best_expected_gain_next_under_executor", 0.0) or 0.0),
                            "controller_route_compare_repair_unseen_best_fragility_prob": float(route_compare_decision.get("repair_unseen_best_fragility_prob", 0.0) or 0.0),
                            "controller_route_compare_repair_unseen_best_cost_estimate": float(route_compare_decision.get("repair_unseen_best_cost_estimate", 0.0) or 0.0),
                            "controller_route_compare_repair_credible_score": route_compare_decision.get("repair_credible_score", None),
                            "controller_route_compare_build_credible_score": route_compare_decision.get("build_credible_score", None),
                        })
                    component_ok, component_reasons = _repair_controller_component_gate(
                        controller_row,
                        controller_components,
                        repair_controller_stats,
                    )
                    inverse_exp_entry["controller_component_gate_ok"] = bool(component_ok)
                    inverse_exp_entry["controller_component_gate_reasons"] = list(component_reasons)
                    repair_candidate_rows: list[dict[str, Any]] = []
                    for meta_source in (repair_preview_meta, route_compare_preview_meta):
                        if not isinstance(meta_source, Mapping):
                            continue
                        raw_rows = meta_source.get(
                            "repair_opportunity_slate_final",
                            meta_source.get("repair_opportunity_slate", []),
                        )
                        for raw_row in list(raw_rows or []):
                            if not isinstance(raw_row, Mapping):
                                continue
                            repair_candidate_rows.append(dict(raw_row))
                    build_candidate_rows = [
                        dict(row)
                        for row in list((controller_build_slate_payload or {}).get("build_opportunity_slate", []) or [])
                        if isinstance(row, Mapping)
                    ]
                    hole_candidate_rows: list[dict[str, Any]] = []
                    hole_runtime_by_opportunity_id: dict[str, Any] = {}
                    scheduler_decision_context_id = f"scheduler:{str(parent_key or '')}:{int(n_evaluated)}"
                    if hole_frontier is not None and A_HOLESEARCH in active_actions:
                        try:
                            from ..hole_search import export_hole_opportunity_rows

                            eligible_hole_opps = list(hole_frontier.eligible(int(n_evaluated)))
                            eligible_hole_opps = sorted(
                                eligible_hole_opps,
                                key=lambda opp: (
                                    -float(getattr(opp, "predicted_value", 0.0) or 0.0),
                                    float(getattr(opp, "predicted_cost", 1.0) or 1.0),
                                    str(getattr(opp, "parent_key", "") or ""),
                                    tuple(int(v) for v in (getattr(opp, "path", ()) or ())),
                                ),
                            )
                            hole_candidate_rows = export_hole_opportunity_rows(
                                eligible_hole_opps,
                                current_iter=int(n_evaluated),
                                decision_id=f"hole_frontier:{str(parent_key or '')}:{int(n_evaluated)}",
                                decision_context_id=scheduler_decision_context_id,
                            )
                            for opp, row in zip(eligible_hole_opps, hole_candidate_rows):
                                if isinstance(row, Mapping):
                                    hole_runtime_by_opportunity_id[str(row.get("opportunity_id", "") or "")] = opp
                        except Exception as exc:
                            scheduler_stats["bundle_error"] = str(exc)
                    if isinstance(inverse_exp_entry, dict):
                        inverse_exp_entry.update({
                            "hole_opportunity_slate_id": f"hole_frontier:{str(parent_key or '')}:{int(n_evaluated)}",
                            "hole_opportunity_slate_count": int(len(hole_candidate_rows)),
                            "hole_opportunity_slate": [dict(row) for row in hole_candidate_rows],
                        })
                    if bool(scheduler_enable) and isinstance(scheduler_bundle, Mapping):
                        plan_candidates = build_plan_candidates(
                            parent_key=str(parent_key or ""),
                            decision_context_id=scheduler_decision_context_id,
                            current_best_route_eff_mse=float(getattr(parent_rec, "best_mse", float("inf"))),
                            build_opportunity_rows=build_candidate_rows,
                            repair_opportunity_rows=repair_candidate_rows,
                            hole_opportunity_rows=hole_candidate_rows,
                            exact_budget_ladder=scheduler_budget_ladder,
                            hole_exact_budget_cap=int(hole_search_exact_budget),
                        )
                        for idx, candidate in enumerate(plan_candidates):
                            if candidate.route == "hole":
                                runtime_row = dict(candidate.payload.get("row", {}) or {})
                                runtime_row["hole_runtime_opportunity"] = hole_runtime_by_opportunity_id.get(
                                    str(candidate.opportunity_id),
                                    None,
                                )
                                plan_candidates[idx] = type(candidate)(
                                    route=candidate.route,
                                    method=candidate.method,
                                    decision_id=candidate.decision_id,
                                    opportunity_id=candidate.opportunity_id,
                                    parent_key=candidate.parent_key,
                                    action=candidate.action,
                                    path=candidate.path,
                                    target_mode=candidate.target_mode,
                                    exact_budget=candidate.exact_budget,
                                    widen_budget=candidate.widen_budget,
                                    micro_budget=candidate.micro_budget,
                                    features=dict(candidate.features),
                                    payload={"row": runtime_row},
                                )
                        scheduler_plan_decision = choose_plan(
                            scheduler_bundle,
                            plan_candidates,
                            advisory_only=bool(scheduler_advisory_only),
                            fallback_min_confidence=float(scheduler_fallback_min_confidence),
                            acquisition_threshold=float(scheduler_acquisition_threshold),
                            uncertainty_bonus=float(scheduler_uncertainty_bonus),
                            acquisition_weights=scheduler_acquisition_weights,
                        )
                        if isinstance(inverse_exp_entry, dict):
                            inverse_exp_entry.update(_scheduler_decision_fields(scheduler_plan_decision))
                        if (
                            bool(scheduler_control_enabled)
                            and scheduler_plan_decision is not None
                            and bool(getattr(scheduler_plan_decision, "trained", False))
                            and (not bool(getattr(scheduler_plan_decision, "fallback_used", False)))
                        ):
                            scheduler_selected_candidate = getattr(scheduler_plan_decision, "chosen_candidate", None)
                            if scheduler_selected_candidate is not None:
                                chosen_route = str(getattr(scheduler_selected_candidate, "route", "") or "")
                                if chosen_route == "build":
                                    action = ACTION_ID_BY_NAME.get(str(getattr(scheduler_selected_candidate, "action", "") or ""), None)
                                    scheduler_forced_action_path = tuple(int(v) for v in (getattr(scheduler_selected_candidate, "path", ()) or ()))
                                    scheduler_forced_action_source = "scheduler_plan"
                                    scheduler_control_applied = action is not None
                                    selected_route_name = "expression_expand"
                                elif chosen_route == "hole":
                                    hole_runtime_row = dict((scheduler_selected_candidate.payload or {}).get("row", {}) or {})
                                    prepared_hole_opp = hole_runtime_row.get("hole_runtime_opportunity", None)
                                    prepared_hole_resolution = (
                                        _resolve_hole_search_parent(
                                            prepared_hole_opp,
                                            current_iter=int(n_evaluated),
                                            touch_snapshot=False,
                                        )
                                        if prepared_hole_opp is not None
                                        else None
                                    )
                                    scheduler_selected_hole_resolution = prepared_hole_resolution
                                    scheduler_selected_hole_budget = int(getattr(scheduler_selected_candidate, "exact_budget", hole_search_exact_budget) or hole_search_exact_budget)
                                    if prepared_hole_opp is not None and prepared_hole_resolution is not None:
                                        action = A_HOLESEARCH
                                        selected_route_name = "opportunity_expand"
                                        scheduler_control_applied = True
                                elif chosen_route == "repair":
                                    scheduler_control_applied = True
                        elif bool(scheduler_control_enabled):
                            scheduler_selected_candidate = None
                    repair_parent_cache[parent_key] = {
                        "expr": node_str(parent_rec.best_expr),
                        "score": float(controller_score + policy_priority_bonus),
                        "score_base": float(
                            controller_score
                            - weights_cfg["stagnation"] * float(controller_components.get("stagnation", 0.0))
                            + policy_priority_bonus
                        ),
                        "gate_score": float(controller_gate_score),
                        "gate_score_base": float(
                            controller_gate_score
                            - weights_cfg["stagnation"] * float(controller_components.get("stagnation", 0.0))
                        ),
                        "threshold": float(controller_threshold),
                        "policy_priority_bonus": float(policy_priority_bonus),
                        "iter": int(n_evaluated),
                        "stagnation": float(controller_components.get("stagnation", 0.0)),
                        "stagnation_adjustable": True,
                    }
                    if len(repair_parent_cache) > 2048:
                        try:
                            repair_parent_cache.pop(next(iter(repair_parent_cache)))
                        except Exception:
                            repair_parent_cache.clear()
                    score_hist = repair_controller_stats.get("score_hist", None)
                    if isinstance(score_hist, list):
                        score_hist.append(float(controller_gate_score))
                        if len(score_hist) > 256:
                            del score_hist[:-256]
                    if repair_preview_expr is None:
                        _repair_parent_record_attempt(
                            parent_key,
                            parent_rec,
                            n_evaluated,
                            repair_parent_state,
                            repair_controller_stats,
                            count_attempt=False,
                        )
                        repair_controller_stats["no_candidate"] = int(repair_controller_stats.get("no_candidate", 0)) + 1
                        _append_inverse_experiment(
                            inverse_exp_entry,
                            inverse_exp_t0,
                            controller_selected=False,
                            status=str((repair_preview_meta or {}).get("status", "controller_no_candidate")),
                        )
                        inverse_exp_entry = None
                        inverse_exp_t0 = None
                    elif not component_ok:
                        _repair_parent_record_attempt(
                            parent_key,
                            parent_rec,
                            n_evaluated,
                            repair_parent_state,
                            repair_controller_stats,
                            count_attempt=False,
                        )
                        for reason in component_reasons:
                            repair_controller_stats[f"blocked_low_{reason}"] = int(repair_controller_stats.get(f"blocked_low_{reason}", 0)) + 1
                        status = "controller_blocked_" + ("_".join(component_reasons) if component_reasons else "component")
                        _append_inverse_experiment(
                            inverse_exp_entry,
                            inverse_exp_t0,
                            controller_selected=False,
                            status=status,
                        )
                        inverse_exp_entry = None
                        inverse_exp_t0 = None
                    elif bool((route_compare_decision or {}).get("veto_repair", False)) and not bool(macro_controller_enable) and not (
                        scheduler_selected_candidate is not None and str(getattr(scheduler_selected_candidate, "route", "") or "") == "repair"
                    ):
                        _repair_parent_record_attempt(
                            parent_key,
                            parent_rec,
                            n_evaluated,
                            repair_parent_state,
                            repair_controller_stats,
                            count_attempt=False,
                        )
                        repair_controller_stats["route_compare_vetoed"] = int(repair_controller_stats.get("route_compare_vetoed", 0)) + 1
                        if isinstance(inverse_exp_entry, dict):
                            inverse_exp_entry["controller_policy_status"] = "controller_prefers_build_route_compare"
                        _append_inverse_experiment(
                            inverse_exp_entry,
                            inverse_exp_t0,
                            controller_selected=False,
                            status="controller_prefers_build_route_compare",
                        )
                        inverse_exp_entry = None
                        inverse_exp_t0 = None
                    elif bool(scheduler_control_applied) and scheduler_selected_candidate is not None and str(getattr(scheduler_selected_candidate, "route", "") or "") != "repair":
                        if isinstance(inverse_exp_entry, dict):
                            inverse_exp_entry["controller_policy_status"] = "scheduler_prefers_" + str(getattr(scheduler_selected_candidate, "route", "") or "")
                    elif bool(macro_controller_enable) and _selected_scheduler_repair_candidate() is None:
                        repair_macro_ready = True
                    elif controller_gate_score >= float(controller_threshold) or (
                        scheduler_selected_candidate is not None and str(getattr(scheduler_selected_candidate, "route", "") or "") == "repair"
                    ):
                        preview_signature = _repair_preview_signature(repair_preview_expr, repair_preview_meta if isinstance(repair_preview_meta, Mapping) else None)
                        if scheduler_selected_candidate is not None and str(getattr(scheduler_selected_candidate, "route", "") or "") == "repair":
                            preview_retry_ok, preview_retry_reason = True, "scheduler_override"
                        else:
                            preview_retry_ok, preview_retry_reason = _repair_parent_preview_retry_gate(
                                parent_key,
                                parent_rec,
                                repair_preview_expr,
                                repair_preview_meta if isinstance(repair_preview_meta, Mapping) else None,
                                repair_parent_state,
                                repair_controller_stats,
                            )
                        if not preview_retry_ok:
                            repair_controller_stats[f"blocked_retry_{preview_retry_reason}"] = int(
                                repair_controller_stats.get(f"blocked_retry_{preview_retry_reason}", 0)
                            ) + 1
                            _repair_parent_record_attempt(
                                parent_key,
                                parent_rec,
                                n_evaluated,
                                repair_parent_state,
                                repair_controller_stats,
                                count_attempt=False,
                            )
                            _append_inverse_experiment(
                                inverse_exp_entry,
                                inverse_exp_t0,
                                controller_selected=False,
                                status="controller_blocked_" + str(preview_retry_reason),
                            )
                            inverse_exp_entry = None
                            inverse_exp_t0 = None
                        else:
                            _repair_parent_record_attempt(
                                parent_key,
                                parent_rec,
                                n_evaluated,
                                repair_parent_state,
                                repair_controller_stats,
                                count_attempt=True,
                                preview_signature=preview_signature,
                            )
                            (
                                repair_initial_path,
                                repair_candidate_paths,
                                repair_first_step_expr,
                                repair_first_step_meta,
                                repair_target_mode,
                                repair_action_config,
                            ) = _repair_option_runtime_inputs()
                            option_ret = run_repair_option(
                                    parent_rec.best_expr,
                                    parent_rec.mapping,
                                    x_fit,
                                    y_fit,
                                    x_probe,
                                    y_probe,
                                    boost_pool_nodes,
                                    boost_pool_phi_fit,
                                    boost_pool_phi,
                                    boost_pool_dims,
                                    preview_rng,
                                    max_depth,
                                    nvars,
                                    poly_degree,
                                    var_dims=var_dims,
                                    max_steps=int(repair_controller_steps),
                                    ancestor_hops=int(repair_controller_ancestor_hops),
                                    min_step_rel_improve=float(repair_controller_min_step_rel_improve),
                                    max_setup_steps=int(repair_controller_max_setup_steps),
                                    setup_step_value_min=float(repair_controller_setup_step_value_min),
                                    setup_step_regret_max=float(repair_controller_setup_step_regret_max),
                                    setup_step_max_worsen=float(repair_controller_setup_step_max_worsen),
                                    initial_path=repair_initial_path,
                                    initial_candidate_paths=repair_candidate_paths,
                                    first_step_expr=repair_first_step_expr,
                                    first_step_meta=repair_first_step_meta,
                                    current_eff_mse=float(getattr(parent_rec, "best_mse", float("inf"))),
                                    target_mode=repair_target_mode,
                                    inverse_action_config=repair_action_config,
                                    complexity_penalty=float(complexity_penalty),
                                    proj=proj,
                                    fp_mode=fp_mode,
                                    q_scale=q_scale,
                                    q_clip=q_clip,
                                    score_expr_cfg=refine_cfg,
                                    return_meta=True,
                                    repair_opportunity_controller_enable=bool(repair_opportunity_controller_enable),
                                    repair_opportunity_bundle=repair_opportunity_bundle,
                                )
                            expr, repair_option_meta = option_ret
                            if isinstance(inverse_exp_entry, dict) and isinstance(repair_option_meta, dict):
                                _merge_repair_option_log_fields(inverse_exp_entry, repair_option_meta)
                            if expr is not None:
                                repair_controller_stats["selected"] = int(repair_controller_stats.get("selected", 0)) + 1
                                repair_controller_selected = True
                                action = A_REPAIR
                                rng.setstate(preview_rng.getstate())
                                inverse_exp_entry["controller_selected"] = True
                                scheduler_control_applied = bool(
                                    scheduler_selected_candidate is not None
                                    and str(getattr(scheduler_selected_candidate, "route", "") or "") == "repair"
                                )
                            else:
                                if _selected_scheduler_repair_candidate() is not None:
                                    scheduler_control_applied = False
                                    scheduler_selected_candidate = None
                                repair_controller_stats["no_candidate"] = int(repair_controller_stats.get("no_candidate", 0)) + 1
                                _append_inverse_experiment(
                                    inverse_exp_entry,
                                    inverse_exp_t0,
                                    controller_selected=False,
                                    status=str((repair_option_meta or {}).get("status", "repair_option_none")),
                                )
                                inverse_exp_entry = None
                                inverse_exp_t0 = None
                    else:
                        _repair_parent_record_attempt(
                            parent_key,
                            parent_rec,
                            n_evaluated,
                            repair_parent_state,
                            repair_controller_stats,
                            count_attempt=False,
                        )
                        repair_controller_stats["blocked_low_score"] = int(repair_controller_stats.get("blocked_low_score", 0)) + 1
                        _append_inverse_experiment(
                            inverse_exp_entry,
                            inverse_exp_t0,
                            controller_selected=False,
                            status="controller_blocked_low_score",
                        )
                        inverse_exp_entry = None
                        inverse_exp_t0 = None

                if repair_controller_selected:
                    repair_controller_stats["option_repair_selected"] = int(repair_controller_stats.get("option_repair_selected", 0)) + 1

            if (
                (selected_route_name != "opportunity_expand")
                and (not repair_controller_selected)
                and (not scheduler_control_applied)
                and (macro_controller is not None)
            ):
                macro_refine_features = _controller_refine_features(parent_rec.best_expr)
                macro_stagnation_state = _repair_controller_stagnation_state(parent_rec, repair_controller_stats)
                selected_macro_state = build_macro_controller_state(
                    parent_key=parent_key,
                    parent_expr=node_str(parent_rec.best_expr),
                    parent_root_op=(str(parent_rec.best_expr[0]) if isinstance(parent_rec.best_expr, tuple) and parent_rec.best_expr else ""),
                    parent_depth=int(node_depth(parent_rec.best_expr)),
                    parent_size=int(node_size(parent_rec.best_expr)),
                    parent_best_eff_mse=float(getattr(parent_rec, "best_mse", float("inf"))),
                    parent_best_raw_mse=float(getattr(parent_rec, "best_raw_mse", getattr(parent_rec, "best_mse", float("inf")))),
                    parent_visits=float(getattr(parent_rec, "visits", 0)),
                    parent_visits_since_improve=float(getattr(parent_rec, "visits_since_improve", getattr(parent_rec, "visits", 0))),
                    parent_stagnation_score=float(macro_stagnation_state.get("stagnation_score", 0.0)),
                    parent_stagnation_ratio=float(macro_stagnation_state.get("stagnation_ratio", 0.0)),
                    allowed_actions=(allowed_actions if allowed_actions is not None else active_actions),
                    action_name_map=ACTION_NAME,
                    gate_diag=inverse_gate_diag_selected,
                    controller_row=controller_row,
                    repair_priority_score=controller_score,
                    repair_gate_score=controller_gate_score,
                    repair_threshold=controller_threshold,
                    repair_ready=bool(repair_macro_ready),
                    repair_preview_available=bool(repair_preview_expr is not None),
                    repair_component_ok=bool(component_ok),
                    refine_slot_count=int(macro_refine_features.get("refine_slot_count", 0)),
                    refine_gate_potential=float(macro_refine_features.get("refine_gate_potential", 0.0)),
                    source="macro_controller",
                )
                macro_policy_guidance = None
                if isinstance(critic_full, dict) and (
                    bool(((critic_full.get("macro_action", {}) or {}).get("trained", False)))
                    or bool(((critic_full.get("value", {}) or {}).get("trained", False)))
                ):
                    macro_policy_guidance = critic_full
                elif repair_critic_bundle is not None:
                    try:
                        macro_policy_row = controller_row
                        if macro_policy_row is None:
                            macro_policy_row = _make_inverse_experiment_record(
                                parent_rec,
                                inverse_gate_diag_selected,
                                stagnation_state=macro_stagnation_state,
                                candidate_meta=repair_preview_meta if isinstance(repair_preview_meta, dict) else None,
                            )
                        macro_policy_guidance = predict_repair_controller_heads(repair_critic_bundle, macro_policy_row)
                    except Exception as exc:
                        macro_controller_stats["policy_predict_error"] = str(exc)
                        macro_policy_guidance = None
                macro_decision = macro_controller.select_action(
                    selected_macro_state,
                    rng,
                    policy_guidance=macro_policy_guidance,
                )
                selected_macro_action_name = str(macro_decision.action_name)
                macro_controller_stats["selected"] = int(macro_controller_stats.get("selected", 0)) + 1
                pol_counts = macro_controller_stats.get("policy_counts", None)
                if isinstance(pol_counts, dict):
                    pol_counts[selected_macro_action_name] = int(pol_counts.get(selected_macro_action_name, 0)) + 1
                src_counts = macro_controller_stats.get("decision_source_counts", None)
                if isinstance(src_counts, dict):
                    src = str(getattr(macro_decision, "policy_source", "") or "")
                    src_counts[src] = int(src_counts.get(src, 0)) + 1
                action = ACTION_ID_BY_NAME.get(selected_macro_action_name, None)
                if action == A_CROSSOVER and crossover_no_partner:
                    # Crossover cannot propose with <2 basins; fall back below.
                    action = None
                if action is None:
                    action = explorer.select_action(parent_key, rng, allowed_actions=allowed_actions)
                    selected_macro_state = None
                    selected_macro_action_name = None
                    macro_controller_stats["fallback_selected"] = int(macro_controller_stats.get("fallback_selected", 0)) + 1
                    if bool(inverse_experiment_log_enable) and action == A_INVSTEER and inverse_exp_entry is None:
                        inverse_exp_t0 = time.perf_counter()
                        inverse_exp_entry = _make_inverse_experiment_row(parent_rec, inverse_gate_diag_selected)
                        inverse_exp_entry.update(_macro_action_fields(A_INVSTEER, source="policy"))
                        inverse_exp_entry.update(_macro_decision_log_fields(macro_decision))
                elif action == A_REPAIR:
                    _repair_parent_record_attempt(
                        parent_key,
                        parent_rec,
                        n_evaluated,
                        repair_parent_state,
                        repair_controller_stats,
                        count_attempt=True,
                    )
                    (
                        repair_initial_path,
                        repair_candidate_paths,
                        repair_first_step_expr,
                        repair_first_step_meta,
                        repair_target_mode,
                        repair_action_config,
                    ) = _repair_option_runtime_inputs()
                    option_ret = run_repair_option(
                        parent_rec.best_expr,
                        parent_rec.mapping,
                        x_fit,
                        y_fit,
                        x_probe,
                        y_probe,
                        boost_pool_nodes,
                        boost_pool_phi_fit,
                        boost_pool_phi,
                        boost_pool_dims,
                        preview_rng,
                        max_depth,
                        nvars,
                        poly_degree,
                        var_dims=var_dims,
                        max_steps=int(repair_controller_steps),
                        ancestor_hops=int(repair_controller_ancestor_hops),
                        min_step_rel_improve=float(repair_controller_min_step_rel_improve),
                        max_setup_steps=int(repair_controller_max_setup_steps),
                        setup_step_value_min=float(repair_controller_setup_step_value_min),
                        setup_step_regret_max=float(repair_controller_setup_step_regret_max),
                        setup_step_max_worsen=float(repair_controller_setup_step_max_worsen),
                        initial_path=repair_initial_path,
                        initial_candidate_paths=repair_candidate_paths,
                        first_step_expr=repair_first_step_expr,
                        first_step_meta=repair_first_step_meta,
                        current_eff_mse=float(getattr(parent_rec, "best_mse", float("inf"))),
                        target_mode=repair_target_mode,
                        inverse_action_config=repair_action_config,
                        complexity_penalty=float(complexity_penalty),
                        proj=proj,
                        fp_mode=fp_mode,
                        q_scale=q_scale,
                        q_clip=q_clip,
                        score_expr_cfg=refine_cfg,
                        return_meta=True,
                        repair_opportunity_controller_enable=bool(repair_opportunity_controller_enable),
                        repair_opportunity_bundle=repair_opportunity_bundle,
                    )
                    expr, repair_option_meta = option_ret
                    if isinstance(inverse_exp_entry, dict) and isinstance(repair_option_meta, dict):
                        _merge_repair_option_log_fields(inverse_exp_entry, repair_option_meta)
                    if expr is not None:
                        macro_controller_stats["repair_selected"] = int(macro_controller_stats.get("repair_selected", 0)) + 1
                        repair_controller_stats["selected"] = int(repair_controller_stats.get("selected", 0)) + 1
                        repair_controller_selected = True
                        action = A_REPAIR
                        rng.setstate(preview_rng.getstate())
                        if isinstance(inverse_exp_entry, dict):
                            inverse_exp_entry["controller_selected"] = True
                            inverse_exp_entry.update(_macro_decision_log_fields(macro_decision))
                    else:
                        repair_controller_stats["no_candidate"] = int(repair_controller_stats.get("no_candidate", 0)) + 1
                        _append_inverse_experiment(
                            inverse_exp_entry,
                            inverse_exp_t0,
                            controller_selected=False,
                            status=str((repair_option_meta or {}).get("status", "repair_option_none")),
                        )
                        inverse_exp_entry = None
                        inverse_exp_t0 = None
                        action = explorer.select_action(parent_key, rng, allowed_actions=allowed_actions)
                        selected_macro_state = None
                        selected_macro_action_name = None
                        macro_controller_stats["fallback_selected"] = int(macro_controller_stats.get("fallback_selected", 0)) + 1
                        if bool(inverse_experiment_log_enable) and action == A_INVSTEER and inverse_exp_entry is None:
                            inverse_exp_t0 = time.perf_counter()
                            inverse_exp_entry = _make_inverse_experiment_row(parent_rec, inverse_gate_diag_selected)
                            inverse_exp_entry.update(_macro_action_fields(A_INVSTEER, source="policy"))
                            inverse_exp_entry.update(_macro_decision_log_fields(macro_decision))
                else:
                    if bool(repair_macro_ready):
                        _repair_parent_record_attempt(
                            parent_key,
                            parent_rec,
                            n_evaluated,
                            repair_parent_state,
                            repair_controller_stats,
                            count_attempt=False,
                        )
                    if isinstance(inverse_exp_entry, dict):
                        inverse_exp_entry["controller_policy_status"] = "controller_prefers_build"
                        inverse_exp_entry.update(_macro_decision_log_fields(macro_decision))
                    if bool(inverse_experiment_log_enable) and action == A_INVSTEER and inverse_exp_entry is None:
                        inverse_exp_t0 = time.perf_counter()
                        inverse_exp_entry = _make_inverse_experiment_row(parent_rec, inverse_gate_diag_selected)
                        inverse_exp_entry.update(_macro_action_fields(A_INVSTEER, source="macro_controller"))
                        inverse_exp_entry.update(_macro_decision_log_fields(macro_decision))

            if (selected_route_name == "opportunity_expand") and prepared_hole_opp is not None and prepared_hole_resolution is not None:
                action = A_HOLESEARCH
            elif (not repair_controller_selected) and (macro_controller is None) and (not scheduler_control_applied):
                action = explorer.select_action(parent_key, rng, allowed_actions=allowed_actions)
                if bool(inverse_experiment_log_enable) and action == A_INVSTEER and inverse_exp_entry is None:
                    inverse_exp_t0 = time.perf_counter()
                    inverse_exp_entry = _make_inverse_experiment_row(parent_rec, inverse_gate_diag_selected)
                    inverse_exp_entry.update(_macro_action_fields(A_INVSTEER, source="policy"))

            # expensive boost gate: only after the action is actually selected
            if action == A_BOOST and bool(boost_enable) and bool(boost_gate_enable):
                ok_sharp = False
                try:
                    use_probe = str(boost_selection_split).lower().startswith("p")
                    x_sel = x_probe if use_probe else x_fit
                    y_sel = y_probe if use_probe else y_fit
                    phi_sel = boost_pool_phi if use_probe else boost_pool_phi_fit
                    norms_sel = boost_pool_norms if use_probe else boost_pool_norms_fit

                    p_sel = eval_node(parent_rec.best_expr, x_sel)
                    y_hat_sel = eval_mapping_total(p_sel, parent_rec.mapping, x_sel)
                    r_sel = (y_sel - y_hat_sel).squeeze(-1)

                    if torch.isfinite(r_sel).all():
                        dots = r_sel @ phi_sel
                        scores = dots * dots / (norms_sel + 1.0e-12)
                        valid = torch.isfinite(scores) & torch.isfinite(norms_sel) & (norms_sel > 1.0e-12)

                        if dm:
                            tgt = y_dims if y_dims is not None else node_dims(parent_rec.best_expr, var_dims)
                            if tgt is not None:
                                m = torch.tensor([
                                    boost_pool_dims[i] is not None and dims_eq(boost_pool_dims[i], tgt)
                                    for i in range(len(boost_pool_dims))
                                ], device=valid.device)
                                if m.any():
                                    valid = valid & m

                        if int(valid.sum().item()) >= int(boost_gate_min_valid):
                            s = scores[valid]
                            peak = float(s.max().item())
                            med = float(s.median().item())
                            sse = float((r_sel * r_sel).sum().item())
                            gain_frac = peak / max(1.0e-12, sse)
                            peak_ratio = peak / (med + 1.0e-12)

                            # --- adaptive gain-fraction threshold (running quantile) ---
                            gf_thr = float(boost_gate_gain_frac)
                            if bool(boost_gate_adaptive) and math.isfinite(gain_frac):
                                hist = boost_gate_stats.get("gain_frac_hist", None)
                                if not isinstance(hist, list):
                                    hist = []
                                    boost_gate_stats["gain_frac_hist"] = hist
                                hist.append(float(gain_frac))
                                try:
                                    w = int(boost_gate_adapt_window)
                                except Exception:
                                    w = 0
                                if w > 0 and len(hist) > w:
                                    del hist[:-w]
                                try:
                                    min_n = int(boost_gate_adapt_min_samples)
                                except Exception:
                                    min_n = 0
                                if min_n > 0 and len(hist) >= min_n:
                                    try:
                                        q = float(boost_gate_adapt_quantile)
                                    except Exception:
                                        q = 0.75
                                    q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
                                    xs = sorted(hist)
                                    idx = int(q * (len(xs) - 1))
                                    idx = 0 if idx < 0 else (len(xs) - 1 if idx >= len(xs) else idx)
                                    qv = float(xs[idx])
                                    try:
                                        mix = float(boost_gate_adapt_mix)
                                    except Exception:
                                        mix = 1.0
                                    mix = 0.0 if mix < 0.0 else (1.0 if mix > 1.0 else mix)
                                    gf_thr = (1.0 - mix) * float(boost_gate_gain_frac) + mix * qv
                                    try:
                                        gf_floor = float(boost_gate_gain_frac_floor)
                                    except Exception:
                                        gf_floor = 0.0
                                    try:
                                        gf_cap = float(boost_gate_gain_frac_cap)
                                    except Exception:
                                        gf_cap = 1.0
                                    if gf_cap <= 0.0:
                                        gf_cap = 1.0
                                    gf_thr = max(gf_floor, min(gf_cap, gf_thr))
                            boost_gate_stats["gain_frac_thr"] = float(gf_thr)

                            pr_thr = float(boost_gate_peak_ratio)
                            ok_sharp = (gain_frac >= gf_thr) and ((pr_thr <= 0.0) or (peak_ratio >= pr_thr))
                except Exception:
                    ok_sharp = False

                if not ok_sharp:
                    boost_gate_stats["blocked_sharp"] = int(boost_gate_stats.get("blocked_sharp", 0)) + 1
                    action = explorer.select_action(parent_key, rng, allowed_actions=_remove_allowed_action(allowed_actions, active_actions, A_BOOST))
                    selected_macro_state = None
                    selected_macro_action_name = None
                    if bool(inverse_experiment_log_enable) and action == A_INVSTEER and inverse_exp_entry is None:
                        inverse_exp_t0 = time.perf_counter()
                        inverse_exp_entry = _make_inverse_experiment_row(parent_rec, inverse_gate_diag_selected)
                        inverse_exp_entry.update(_macro_action_fields(A_INVSTEER, source="policy"))
                else:
                    boost_gate_stats["allowed"] = int(boost_gate_stats.get("allowed", 0)) + 1

            # Quota-forced hole search: override the selected action with
            # A_HOLESEARCH when the hole frontier has eligible opportunities.
            if (
                route_scheduler is None
                and
                prepared_hole_opp is not None
                and action is not None
                and (not scheduler_control_applied)
                and action not in (A_HOLESEARCH, A_INVSTEER, A_REPAIR)
                and A_HOLESEARCH in (allowed_actions if allowed_actions is not None else active_actions)
                and rng.random() < float(hole_search_quota)
            ):
                action = A_HOLESEARCH
                selected_macro_state = None
                selected_macro_action_name = None

            if bool(inverse_experiment_log_enable) and action is not None:
                if _scheduler_control_route_applied(action):
                    action_source = "scheduler"
                elif selected_macro_action_name is not None:
                    action_source = "macro_controller"
                elif repair_controller_selected:
                    action_source = None
                elif selected_route_name == "opportunity_expand" and action == A_HOLESEARCH:
                    action_source = "route_scheduler"
                else:
                    action_source = "policy"
                if not (action == A_HOLESEARCH and parent_rec is None):
                    inverse_exp_entry, inverse_exp_t0 = _ensure_action_experiment_entry(
                        inverse_exp_entry,
                        inverse_exp_t0,
                        parent_rec,
                        inverse_gate_diag_selected,
                        action=action,
                        source=action_source,
                        macro_decision=macro_decision,
                        controller_record=controller_row,
                        stagnation_state=macro_stagnation_state,
                        candidate_meta=repair_preview_meta if isinstance(repair_preview_meta, dict) else None,
                    )
            if scheduler_plan_decision is not None:
                scheduler_applied = _scheduler_control_route_applied(action)
                if isinstance(inverse_exp_entry, dict):
                    inverse_exp_entry["scheduler_applied"] = bool(scheduler_applied)
                scheduler_decision_log_idx = _record_scheduler_decision(
                    decision=scheduler_plan_decision,
                    parent_key=parent_key,
                    parent_rec=parent_rec,
                    current_iter=int(n_evaluated),
                    applied=bool(scheduler_applied),
                    selected_action=action,
                )

            action_path = None
            action_path_source = ""
            if action in (A_REPLACE, A_WRAP_UNARY, A_ADD_RAND, A_MUL_RAND, A_PRUNE, A_CROSSOVER):
                scheduler_build_action = None
                if scheduler_selected_candidate is not None and str(getattr(scheduler_selected_candidate, "route", "") or "") == "build":
                    scheduler_build_action = ACTION_ID_BY_NAME.get(
                        str(getattr(scheduler_selected_candidate, "action", "") or ""),
                        None,
                    )
                if (
                    scheduler_forced_action_path is not None
                    and bool(scheduler_control_applied)
                    and scheduler_build_action is not None
                    and int(action) == int(scheduler_build_action)
                ):
                    action_path = tuple(int(v) for v in tuple(scheduler_forced_action_path or ()))
                    action_path_source = str(scheduler_forced_action_source or "scheduler_plan")
                else:
                    action_path, action_path_source = _controller_selected_action_path(
                        parent_rec.best_expr,
                        action,
                        controller_policy_guidance=controller_policy_guidance,
                        macro_decision=macro_decision,
                        macro_state=selected_macro_state,
                        inverse_gate_diag=inverse_gate_diag_selected,
                    )
                if isinstance(inverse_exp_entry, dict):
                    inverse_exp_entry["controller_action_path"] = [] if action_path is None else [int(v) for v in action_path]
                    inverse_exp_entry["controller_action_path_source"] = str(action_path_source or "random")
                    inverse_exp_entry["controller_action_path_guided"] = bool(action_path_source)

            if action in action_selected_counts:
                action_selected_counts[action] = int(action_selected_counts.get(action, 0)) + 1
            crossover_policy = None
            if action == A_RESIDUAL:
                expr = apply_residual_action(
                    parent_rec.best_expr, parent_rec.mapping,
                    x_fit, y_fit, x_probe, y_probe,
                    pool_nodes, pool_phi, pool_norms, pool_dims,
                    rng, max_depth, nvars, poly_degree,
                    var_dims=var_dims, topk=residual_topk,
                    complexity_penalty=complexity_penalty,
                )
            elif action == A_INVSTEER:
                if repair_preview_expr is not None and isinstance(repair_preview_meta, dict):
                    expr = repair_preview_expr
                    inv_meta = repair_preview_meta
                    try:
                        rng.setstate(preview_rng.getstate())
                    except Exception:
                        pass
                    if isinstance(inverse_exp_entry, dict):
                        inverse_exp_entry["controller_preview_reused"] = True
                else:
                    inv_ret = apply_inverse_steering_action(
                        parent_rec.best_expr,
                        parent_rec.mapping,
                        x_fit,
                        y_fit,
                        x_probe,
                        y_probe,
                        boost_pool_nodes,
                        boost_pool_phi_fit,
                        boost_pool_phi,
                        boost_pool_dims,
                        rng,
                        max_depth,
                        nvars,
                        poly_degree,
                        var_dims=var_dims,
                        max_paths=inverse_max_paths,
                        topk_terms=inverse_topk_terms,
                        shortlist_mult=inverse_shortlist_mult,
                        min_valid_frac=inverse_min_valid_frac,
                        min_confidence=inverse_min_confidence,
                        safe_eps=float(inverse_safe_eps),
                        confidence_mode=str(inverse_confidence_mode),
                        confidence_target_gain=float(inverse_confidence_target_gain),
                        confidence_floor=float(inverse_confidence_floor),
                        branch_beam_width=int(inverse_branch_beam_width),
                        micro_search_enable=bool(inverse_micro_search_enable),
                        micro_search_max_depth=int(inverse_micro_search_max_depth),
                        micro_search_beam_width=int(inverse_micro_search_beam_width),
                        micro_search_topk=int(inverse_micro_search_topk),
                        micro_search_seed_terms=int(inverse_micro_search_seed_terms),
                        local_score_mode=str(inverse_local_score_mode),
                        inverse_spec_enable=bool(inverse_spec_enable),
                        inverse_spec_enum_max_depth=int(inverse_spec_enum_max_depth),
                        inverse_spec_enum_max_trees=int(inverse_spec_enum_max_trees),
                        inverse_spec_preview_topk=int(inverse_spec_preview_topk),
                        inverse_spec_local_score_mode=str(inverse_spec_local_score_mode),
                        inverse_spec_include_legacy_seed=bool(inverse_spec_include_legacy_seed),
                        inverse_spec_complexity_penalty=float(inverse_spec_complexity_penalty),
                        inverse_spec_family_battery_enable=bool(inverse_spec_family_battery_enable),
                        inverse_spec_family_battery_mode=str(inverse_spec_family_battery_mode or "outer"),
                        inverse_spec_repair_quota=float(inverse_spec_repair_quota),
                        inverse_spec_recursive_enable=bool(inverse_spec_recursive_enable),
                        inverse_spec_recursive_max_depth=int(inverse_spec_recursive_max_depth),
                        inverse_spec_recursive_trigger_rel_mse=float(inverse_spec_recursive_trigger_rel_mse),
                        inverse_spec_recursive_seed_cap=int(inverse_spec_recursive_seed_cap),
                        inverse_spec_recursive_branch_topk=int(inverse_spec_recursive_branch_topk),
                        inverse_spec_recursive_child_topk=int(inverse_spec_recursive_child_topk),
                        inverse_spec_witness_jets_enable=bool(inverse_spec_witness_jets_enable),
                        inverse_spec_witness_d2_enable=bool(inverse_spec_witness_d2_enable),
                        inverse_spec_witness_max_rows=int(inverse_spec_witness_max_rows),
                        inverse_spec_witness_loss_enable=bool(inverse_spec_witness_loss_enable),
                        inverse_spec_witness_grad_weight=float(inverse_spec_witness_grad_weight),
                        inverse_spec_witness_d2_weight=float(inverse_spec_witness_d2_weight),
                        inverse_spec_witness_diag_weight=float(inverse_spec_witness_diag_weight),
                        inverse_spec_witness_physics_weight=float(inverse_spec_witness_physics_weight),
                        inverse_spec_active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
                        inverse_spec_active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
                        inverse_spec_active_var_max_count=int(inverse_spec_active_var_max_count),
                        inverse_spec_directional_market_enable=bool(inverse_spec_directional_market_enable),
                        inverse_spec_max_subtree_depth=inverse_spec_max_subtree_depth,
                        inverse_spec_fit_cap=int(inverse_spec_fit_cap),
                        inverse_spec_probe_cap=int(inverse_spec_probe_cap),
                        inverse_spec_exact_budget=int(inverse_spec_exact_budget),
                        target_mode=str(inverse_target_mode),
                        full_mapping_penalty=float(inverse_full_mapping_penalty),
                        exact_simple_target_bonus=float(inverse_exact_simple_target_bonus),
                        additive_descend_penalty=float(inverse_additive_descend_penalty),
                        nonadditive_leaf_penalty=float(inverse_nonadditive_leaf_penalty),
                        exact_path_eta=float(inverse_exact_path_eta),
                        exact_transport_min_lin_rel=float(inverse_exact_transport_min_lin_rel),
                        periodic_min_valid_scale=float(inverse_periodic_min_valid_scale),
                        periodic_min_confidence_scale=float(inverse_periodic_min_confidence_scale),
                        periodic_path_penalty=float(inverse_periodic_path_penalty),
                        nonperiodic_muldiv_bonus=float(inverse_nonperiodic_muldiv_bonus),
                        nonperiodic_explogsqrt_bonus=float(inverse_nonperiodic_explogsqrt_bonus),
                        branch_ambiguity_penalty=float(inverse_branch_ambiguity_penalty),
                        transport_min_lin_rel=float(inverse_transport_min_lin_rel),
                        transport_min_effective_n=float(inverse_transport_min_effective_n),
                        complexity_penalty=complexity_penalty,
                        candidate_paths=inverse_gate_paths,
                        proj=proj,
                        fp_mode=fp_mode,
                        q_scale=q_scale,
                        q_clip=q_clip,
                        score_expr_cfg=refine_cfg,
                        return_meta=bool(inverse_experiment_log_enable or (hole_frontier is not None)),
                        repair_tuple_bundle=repair_critic_bundle,
                        repair_tuple_controller_row=inverse_exp_entry if isinstance(inverse_exp_entry, Mapping) else None,
                        repair_opportunity_controller_enable=bool(repair_opportunity_controller_enable),
                        repair_opportunity_bundle=repair_opportunity_bundle,
                        inverse_spec_regime_metadata=inverse_spec_regime_metadata,
                    )
                    if bool(inverse_experiment_log_enable) or (hole_frontier is not None):
                        expr, inv_meta = inv_ret
                    else:
                        expr = inv_ret
                        inv_meta = None
                if bool(inverse_experiment_log_enable) and isinstance(inverse_exp_entry, dict) and isinstance(inv_meta, dict):
                    _merge_inverse_proposal_log_fields(
                        inverse_exp_entry,
                        inv_meta,
                        status_key="proposal_status",
                    )
                _ingest_hole_search_slate_from_meta(
                    meta_dict=inv_meta,
                    source="inverse_slate",
                    current_iter=int(n_evaluated),
                    parent_key=parent_key,
                    parent_rec=parent_rec,
                )
            elif action == A_REPAIR:
                if not repair_controller_selected:
                    expr = None
                _ingest_hole_search_slate_from_meta(
                    meta_dict=repair_option_meta,
                    source="repair_option",
                    current_iter=int(n_evaluated),
                    parent_key=parent_key,
                    parent_rec=parent_rec,
                )
            elif action == A_HOLESEARCH:
                (
                    expr,
                    hole_meta,
                    executed_hole_opp,
                    executed_hole_status,
                    executed_hole_wall_s,
                    executed_hole_shortlist_eff_mse,
                ) = _execute_prepared_spec_expand(
                    opportunity=prepared_hole_opp,
                    resolution=prepared_hole_resolution,
                    current_iter=int(n_evaluated),
                    exec_ctx=exec_ctx,
                    exact_budget_override=(
                        scheduler_selected_hole_budget
                        if (
                            bool(scheduler_control_applied)
                            and scheduler_selected_candidate is not None
                            and str(getattr(scheduler_selected_candidate, "route", "") or "") == "hole"
                        )
                        else None
                    ),
                )
                if bool(inverse_experiment_log_enable) and isinstance(inverse_exp_entry, dict) and isinstance(hole_meta, Mapping):
                    inverse_exp_entry.update({
                        str(key): value
                        for key, value in hole_meta.items()
                        if str(key).startswith("hole_search_") or str(key).startswith("observed_")
                    })
            elif action == A_BOOST:
                expr = apply_boost_action(
                    parent_rec.best_expr,
                    x_fit, y_fit, x_probe, y_probe,
                    boost_pool_nodes,
                    boost_pool_phi_fit, boost_pool_norms_fit,
                    boost_pool_phi, boost_pool_norms,
                    boost_pool_dims,
                    rng, max_depth, nvars, poly_degree,
                    var_dims=var_dims, y_dims=y_dims,
                    max_terms=boost_max_terms,
                    topk_try=boost_topk_try,
                    min_rel_improve=boost_min_rel_improve,
                    selection_split=boost_selection_split,
                    ridge=float(boost_ridge) if boost_ridge is not None else float(refine_linear_ridge),
                    include_parent=boost_include_parent,
                    from_scratch_prob=boost_from_scratch_prob,
                    prune_rel=boost_prune_rel,
                    complexity_penalty=complexity_penalty,
                )
            elif action == A_CROSSOVER:
                crossover_policy = "legacy"
                if crossover_policy in crossover_policy_stats:
                    crossover_policy_stats[crossover_policy]["selected"] += 1
                expr = apply_crossover_action(
                    parent_rec.best_expr, arch, parent_key, rng,
                    max_depth, nvars, var_dims=var_dims,
                    exploit_frac=exploit_frac, exploit_topk=exploit_topk,
                    path=action_path,
                )
                if expr is not None:
                    if crossover_policy in crossover_policy_stats:
                        crossover_policy_stats[crossover_policy]["proposed"] += 1
            else:
                expr = apply_action(parent_rec.best_expr, action, rng, max_depth, nvars,
                                    var_dims=var_dims, reach=reach, path=action_path)
            if expr is None:
                route_wall_now = None if route_eval_t0 is None else max(0.0, time.perf_counter() - route_eval_t0)
                inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                    inverse_exp_entry,
                    route_diag_state=route_diag_state,
                    raw_reward=0.0,
                    wall_s=route_wall_now,
                )
                if executed_hole_opp is not None:
                    _record_hole_search_frontier_outcome(
                        executed_hole_opp,
                        current_iter=int(n_evaluated),
                        exact_eff_mse=None,
                        shortlist_eff_mse=executed_hole_shortlist_eff_mse,
                        reward=0.0,
                        wall_s=executed_hole_wall_s if executed_hole_wall_s is not None else route_wall_now,
                        parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                        accepted=False,
                        status=str(executed_hole_status or "proposal_none"),
                    )
                row_idx = _append_inverse_experiment(
                    inverse_exp_entry,
                    inverse_exp_t0,
                    status=str((inverse_exp_entry or {}).get("status", "proposal_none")),
                )
                if scheduler_plan_decision is not None:
                    realized_row = (
                        inverse_experiment_log[int(row_idx)]
                        if inverse_experiment_log is not None and row_idx is not None and 0 <= int(row_idx) < len(inverse_experiment_log)
                        else None
                    )
                    _record_scheduler_outcome(
                        decision=scheduler_plan_decision,
                        decision_log_index=scheduler_decision_log_idx,
                        current_iter=int(n_evaluated),
                        applied=bool(_scheduler_control_route_applied(action)),
                        selected_action=action,
                        inverse_row_index=row_idx,
                        inverse_row=realized_row,
                        local_parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                        child_eff_mse=None,
                        global_best_eff_mse_before=float(best_mse),
                        status=str((inverse_exp_entry or {}).get("status", "proposal_none")),
                    )
                continue
            if action in action_proposed_counts:
                action_proposed_counts[action] = int(action_proposed_counts.get(action, 0)) + 1

        if not is_valid_node(expr):
            route_wall_now = None if route_eval_t0 is None else max(0.0, time.perf_counter() - route_eval_t0)
            inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                inverse_exp_entry,
                route_diag_state=route_diag_state,
                raw_reward=0.0,
                wall_s=route_wall_now,
            )
            if executed_hole_opp is not None:
                _record_hole_search_frontier_outcome(
                    executed_hole_opp,
                    current_iter=int(n_evaluated),
                    exact_eff_mse=None,
                    shortlist_eff_mse=executed_hole_shortlist_eff_mse,
                    reward=0.0,
                    wall_s=executed_hole_wall_s if executed_hole_wall_s is not None else route_wall_now,
                    parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                    accepted=False,
                    status="proposal_malformed",
                )
            row_idx = _append_inverse_experiment(inverse_exp_entry, inverse_exp_t0, status="proposal_malformed")
            if scheduler_plan_decision is not None:
                realized_row = (
                    inverse_experiment_log[int(row_idx)]
                    if inverse_experiment_log is not None and row_idx is not None and 0 <= int(row_idx) < len(inverse_experiment_log)
                    else None
                )
                _record_scheduler_outcome(
                    decision=scheduler_plan_decision,
                    decision_log_index=scheduler_decision_log_idx,
                    current_iter=int(n_evaluated),
                    applied=bool(_scheduler_control_route_applied(action)),
                    selected_action=action,
                    inverse_row_index=row_idx,
                    inverse_row=realized_row,
                    local_parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                    child_eff_mse=None,
                    global_best_eff_mse_before=float(best_mse),
                    status="proposal_malformed",
                )
            continue

        # Simplify before scoring / dimension check
        expr = simplify(expr)
        if not is_valid_node(expr):
            route_wall_now = None if route_eval_t0 is None else max(0.0, time.perf_counter() - route_eval_t0)
            inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                inverse_exp_entry,
                route_diag_state=route_diag_state,
                raw_reward=0.0,
                wall_s=route_wall_now,
            )
            if executed_hole_opp is not None:
                _record_hole_search_frontier_outcome(
                    executed_hole_opp,
                    current_iter=int(n_evaluated),
                    exact_eff_mse=None,
                    shortlist_eff_mse=executed_hole_shortlist_eff_mse,
                    reward=0.0,
                    wall_s=executed_hole_wall_s if executed_hole_wall_s is not None else route_wall_now,
                    parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                    accepted=False,
                    status="proposal_malformed_after_simplify",
                )
            row_idx = _append_inverse_experiment(inverse_exp_entry, inverse_exp_t0, status="proposal_malformed_after_simplify")
            if scheduler_plan_decision is not None:
                realized_row = (
                    inverse_experiment_log[int(row_idx)]
                    if inverse_experiment_log is not None and row_idx is not None and 0 <= int(row_idx) < len(inverse_experiment_log)
                    else None
                )
                _record_scheduler_outcome(
                    decision=scheduler_plan_decision,
                    decision_log_index=scheduler_decision_log_idx,
                    current_iter=int(n_evaluated),
                    applied=bool(_scheduler_control_route_applied(action)),
                    selected_action=action,
                    inverse_row_index=row_idx,
                    inverse_row=realized_row,
                    local_parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                    child_eff_mse=None,
                    global_best_eff_mse_before=float(best_mse),
                    status="proposal_malformed_after_simplify",
                )
            continue
        # Strip top-level neg (mapping absorbs sign)
        while expr[0] == "neg":
            expr = expr[1]
        # Canonicalise top-level sub: sub(B,A) → sub(A,B) when A<B
        if expr[0] == "sub" and node_str(expr[1]) > node_str(expr[2]):
            expr = ("sub", expr[2], expr[1])

        # Enforce dimensional consistency: only score expressions with correct
        # output dimensions.  Rejected expressions do not count toward n_iter.
        if dm:
            expr_dim = node_dims(expr, var_dims)
            if expr_dim is None:
                route_wall_now = None if route_eval_t0 is None else max(0.0, time.perf_counter() - route_eval_t0)
                inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                    inverse_exp_entry,
                    route_diag_state=route_diag_state,
                    raw_reward=0.0,
                    wall_s=route_wall_now,
                )
                if executed_hole_opp is not None:
                    _record_hole_search_frontier_outcome(
                        executed_hole_opp,
                        current_iter=int(n_evaluated),
                        exact_eff_mse=None,
                        shortlist_eff_mse=executed_hole_shortlist_eff_mse,
                        reward=0.0,
                        wall_s=executed_hole_wall_s if executed_hole_wall_s is not None else route_wall_now,
                        parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                        accepted=False,
                        status="dim_invalid",
                    )
                row_idx = _append_inverse_experiment(inverse_exp_entry, inverse_exp_t0, status="dim_invalid")
                if scheduler_plan_decision is not None:
                    realized_row = (
                        inverse_experiment_log[int(row_idx)]
                        if inverse_experiment_log is not None and row_idx is not None and 0 <= int(row_idx) < len(inverse_experiment_log)
                        else None
                    )
                    _record_scheduler_outcome(
                        decision=scheduler_plan_decision,
                        decision_log_index=scheduler_decision_log_idx,
                        current_iter=int(n_evaluated),
                        applied=bool(_scheduler_control_route_applied(action)),
                        selected_action=action,
                        inverse_row_index=row_idx,
                        inverse_row=realized_row,
                        local_parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                        child_eff_mse=None,
                        global_best_eff_mse_before=float(best_mse),
                        status="dim_invalid",
                    )
                continue  # dimensionally invalid (e.g. cos of unitful arg)
            if y_dims is not None and not dims_eq(expr_dim, y_dims):
                route_wall_now = None if route_eval_t0 is None else max(0.0, time.perf_counter() - route_eval_t0)
                inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                    inverse_exp_entry,
                    route_diag_state=route_diag_state,
                    raw_reward=0.0,
                    wall_s=route_wall_now,
                )
                if executed_hole_opp is not None:
                    _record_hole_search_frontier_outcome(
                        executed_hole_opp,
                        current_iter=int(n_evaluated),
                        exact_eff_mse=None,
                        shortlist_eff_mse=executed_hole_shortlist_eff_mse,
                        reward=0.0,
                        wall_s=executed_hole_wall_s if executed_hole_wall_s is not None else route_wall_now,
                        parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                        accepted=False,
                        status="dim_mismatch",
                    )
                row_idx = _append_inverse_experiment(inverse_exp_entry, inverse_exp_t0, status="dim_mismatch")
                if scheduler_plan_decision is not None:
                    realized_row = (
                        inverse_experiment_log[int(row_idx)]
                        if inverse_experiment_log is not None and row_idx is not None and 0 <= int(row_idx) < len(inverse_experiment_log)
                        else None
                    )
                    _record_scheduler_outcome(
                        decision=scheduler_plan_decision,
                        decision_log_index=scheduler_decision_log_idx,
                        current_iter=int(n_evaluated),
                        applied=bool(_scheduler_control_route_applied(action)),
                        selected_action=action,
                        inverse_row_index=row_idx,
                        inverse_row=realized_row,
                        local_parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                        child_eff_mse=None,
                        global_best_eff_mse_before=float(best_mse),
                        status="dim_mismatch",
                    )
                continue  # wrong output dimension

        best_mse_before_update = float(best_mse)
        if action is None:
            action_name = str(proposal_mode_name or "unknown")
        else:
            action_name = str(ACTION_NAME.get(action, f"action_{int(action)}"))
        parent_raw_mse_for_scoring = None
        if parent_rec is not None:
            try:
                parent_raw_mse_for_scoring = float(
                    getattr(parent_rec, "best_raw_mse", getattr(parent_rec, "best_mse", float("inf")))
                )
            except Exception:
                parent_raw_mse_for_scoring = None
        global_best_raw_mse_for_scoring = None
        if math.isfinite(best_raw_mse_struct):
            global_best_raw_mse_for_scoring = float(best_raw_mse_struct)
        elif math.isfinite(best_raw_mse):
            global_best_raw_mse_for_scoring = float(best_raw_mse)
        score_cfg = dict(refine_cfg)
        score_cfg["refine_context"] = "mutation"
        score_cfg["score_prescreen_enable"] = bool(
            refine_cfg.get("score_prescreen_enable", False)
        ) and (action in prescreen_actions)
        score_cfg["score_prescreen_force_full"] = bool(action in full_score_actions)
        score_cfg["score_prescreen_action_name"] = action_name
        if action == A_RESIDUAL:
            score_cfg["score_prescreen_family_mode"] = str(
                refine_cfg.get(
                    "score_prescreen_residual_family_mode",
                    refine_cfg.get("score_prescreen_family_mode", "gated"),
                )
                or "gated"
            )
            score_cfg["score_prescreen_allow_hint"] = bool(
                refine_cfg.get("score_prescreen_residual_allow_hint", False)
            )
            score_cfg["score_prescreen_use_global_best"] = bool(
                refine_cfg.get("score_prescreen_residual_use_global_best", False)
            )
            score_cfg["score_prescreen_parent_best_factor"] = float(
                refine_cfg.get(
                    "score_prescreen_residual_parent_best_factor",
                    refine_cfg.get("score_prescreen_parent_best_factor", 1.1),
                )
                or 1.1
            )
            score_cfg["score_prescreen_global_best_factor"] = float(
                refine_cfg.get(
                    "score_prescreen_residual_global_best_factor",
                    refine_cfg.get("score_prescreen_global_best_factor", 1.5),
                )
                or 1.5
            )
        score_cfg["score_prescreen_parent_mse"] = parent_raw_mse_for_scoring
        score_cfg["score_prescreen_global_best_mse"] = global_best_raw_mse_for_scoring
        score_cfg["score_prescreen_stats"] = score_prescreen_stats
        sc = score_expr_fn(
            expr, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=mutation_refine_enable,
            refine_cfg=score_cfg,
            refine_best_mse=(
                float(best_raw_mse_struct)
                if math.isfinite(best_raw_mse_struct)
                else float(max(best_raw_mse, float(early_stop_mse)))
            ),
            refine_state=refine_state,
            return_expr=True,
        )
        n_evaluated += 1
        if sc is None:
            if stall_window > 0 and n_evaluated % stall_window == 0 and n_evaluated > 0:
                hole_search_stats["stall_score_none_skips"] = int(
                    hole_search_stats.get("stall_score_none_skips", 0)
                ) + 1
            route_wall_now = None if route_eval_t0 is None else max(0.0, time.perf_counter() - route_eval_t0)
            inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                inverse_exp_entry,
                route_diag_state=route_diag_state,
                raw_reward=0.0,
                wall_s=route_wall_now,
            )
            if executed_hole_opp is not None:
                _record_hole_search_frontier_outcome(
                    executed_hole_opp,
                    current_iter=int(n_evaluated),
                    exact_eff_mse=None,
                    shortlist_eff_mse=executed_hole_shortlist_eff_mse,
                    reward=0.0,
                    wall_s=executed_hole_wall_s if executed_hole_wall_s is not None else route_wall_now,
                    parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                    accepted=False,
                    status="score_none",
                )
            plateau_after_score_none = _maybe_run_stall_check(int(n_evaluated))
            row_idx = _append_inverse_experiment(inverse_exp_entry, inverse_exp_t0, status="score_none")
            if scheduler_plan_decision is not None:
                realized_row = (
                    inverse_experiment_log[int(row_idx)]
                    if inverse_experiment_log is not None and row_idx is not None and 0 <= int(row_idx) < len(inverse_experiment_log)
                    else None
                )
                _record_scheduler_outcome(
                    decision=scheduler_plan_decision,
                    decision_log_index=scheduler_decision_log_idx,
                    current_iter=int(n_evaluated),
                    applied=bool(_scheduler_control_route_applied(action)),
                    selected_action=action,
                    inverse_row_index=row_idx,
                    inverse_row=realized_row,
                    local_parent_eff_mse=exec_ctx.executed_parent_eff_mse,
                    child_eff_mse=None,
                    global_best_eff_mse_before=float(best_mse_before_update),
                    status="score_none",
                )
            if plateau_after_score_none:
                break
            continue
        mse, key, z, mapping, scored_expr = sc

        if mse < best_raw_mse:
            best_raw_mse = mse
        if mapping_is_structural(mapping) and mse < best_raw_mse_struct:
            best_raw_mse_struct = mse

        mse_eff = mse + complexity_penalty * (
            float(node_size(expr)) + float(mapping_cost(mapping))
        )
        executed_parent_eff_mse_before = None
        if exec_ctx.executed_parent_key is not None and exec_ctx.executed_parent_eff_mse is not None:
            try:
                executed_parent_eff_mse_before = float(exec_ctx.executed_parent_eff_mse)
            except Exception:
                executed_parent_eff_mse_before = None
        is_new = arch.update(key, mse_eff, scored_expr, z, mapping, raw_mse=mse)
        became_global_best = bool(mse_eff < best_mse_before_update)
        accepted_from_executed_parent = False
        route_wall_s = None if route_eval_t0 is None else float(max(0.0, time.perf_counter() - route_eval_t0))
        try:
            inverse_wall_s = (
                float(max(0.0, time.perf_counter() - inverse_exp_t0))
                if inverse_exp_t0 is not None
                else route_wall_s
            )
        except Exception:
            inverse_wall_s = route_wall_s
        actor_reward_terms = _actor_critic_reward_terms(
            executed_parent_eff_mse_before if exec_ctx.executed_parent_key is not None else None,
            mse_eff,
            created_new_residual_basin=bool(is_new),
            became_global_best=bool(became_global_best),
            wall_s=inverse_wall_s,
            novelty_bonus=float(novelty_bonus),
            best_bonus=float(actor_critic_best_bonus),
            time_penalty=float(actor_critic_time_penalty),
            eps=float(actor_critic_reward_eps),
        )
        hole_outcome_reward = None
        hole_outcome_accepted = None
        hole_outcome_status = "scored_no_reward"

        if (
            exec_ctx.executed_parent_key is not None
            and executed_parent_eff_mse_before is not None
            and math.isfinite(executed_parent_eff_mse_before)
            and math.isfinite(mse_eff)
        ):
            r = math.log(executed_parent_eff_mse_before + 1e-30) - math.log(mse_eff + 1e-30)
            if is_new: r += novelty_bonus
            if action in active_actions:
                explorer.update(exec_ctx.executed_parent_key, action, r)
            inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                inverse_exp_entry,
                route_diag_state=route_diag_state,
                raw_reward=float(r),
                wall_s=route_wall_s,
            )
            if (macro_controller is not None) and (selected_macro_state is not None) and (selected_macro_action_name is not None):
                macro_controller.update(selected_macro_state, selected_macro_action_name, r)
            if action in action_reward_counts:
                action_reward_counts[action] = int(action_reward_counts.get(action, 0)) + 1
            accepted_from_executed_parent = bool(mse_eff < executed_parent_eff_mse_before)
            hole_outcome_reward = float(r)
            hole_outcome_accepted = bool(accepted_from_executed_parent)
            hole_outcome_status = "accepted" if accepted_from_executed_parent else "scored"
            if accepted_from_executed_parent:
                if action in action_accepted_counts:
                    action_accepted_counts[action] = int(action_accepted_counts.get(action, 0)) + 1
            if crossover_policy is not None:
                if crossover_policy in crossover_policy_stats:
                    crossover_policy_stats[crossover_policy]["reward_sum"] += float(r)
                    crossover_policy_stats[crossover_policy]["reward_count"] += 1
                    if accepted_from_executed_parent:
                        crossover_policy_stats[crossover_policy]["accepted"] += 1
            row_idx = _append_inverse_experiment(
                inverse_exp_entry,
                inverse_exp_t0,
                status="scored",
                child_expr=node_str(scored_expr),
                child_raw_mse=float(mse),
                child_eff_mse=float(mse_eff),
                parent_eff_mse=float(executed_parent_eff_mse_before),
                created_new_residual_basin=bool(is_new),
                accepted=accepted_from_executed_parent,
                became_global_best=bool(became_global_best),
                reward=float(r),
                reward_per_s=(float(r) / max(1.0e-12, float(inverse_wall_s))) if inverse_wall_s is not None else None,
                selected_parent_key_raw=None if exec_ctx.selected_parent_key is None else str(exec_ctx.selected_parent_key),
                selected_parent_elite_id=str(exec_ctx.selected_parent_elite_id),
                executed_parent_key_raw=str(exec_ctx.executed_parent_key),
                executed_parent_elite_id=str(exec_ctx.executed_parent_elite_id),
                **actor_reward_terms,
            )
            if isinstance(lineage_events, list):
                lineage_events.append({
                    "row_index": row_idx,
                    "parent_key_raw": exec_ctx.executed_parent_key,
                    "selected_parent_key_raw": None if exec_ctx.selected_parent_key is None else str(exec_ctx.selected_parent_key),
                    "child_key_raw": key,
                    "parent_eff_mse": float(executed_parent_eff_mse_before),
                    "child_eff_mse": float(mse_eff),
                    "child_raw_mse": float(mse),
                })
        else:
            hole_outcome_status = "scored_no_reward"
            inverse_exp_entry, _route_adjusted_reward = _finalize_route_scheduler_diagnostics(
                inverse_exp_entry,
                route_diag_state=route_diag_state,
                raw_reward=0.0,
                wall_s=route_wall_s,
            )
            row_idx = _append_inverse_experiment(
                inverse_exp_entry,
                inverse_exp_t0,
                status="scored_no_reward",
                child_expr=node_str(scored_expr),
                child_raw_mse=float(mse),
                child_eff_mse=float(mse_eff),
                parent_eff_mse=executed_parent_eff_mse_before,
                created_new_residual_basin=bool(is_new),
                accepted=None,
                became_global_best=bool(became_global_best),
                reward=None,
                reward_per_s=None,
                selected_parent_key_raw=None if exec_ctx.selected_parent_key is None else str(exec_ctx.selected_parent_key),
                selected_parent_elite_id=str(exec_ctx.selected_parent_elite_id),
                executed_parent_key_raw=None if exec_ctx.executed_parent_key is None else str(exec_ctx.executed_parent_key),
                executed_parent_elite_id=str(exec_ctx.executed_parent_elite_id),
                **actor_reward_terms,
            )
            if isinstance(lineage_events, list) and exec_ctx.executed_parent_key is not None:
                lineage_events.append({
                    "row_index": row_idx,
                    "parent_key_raw": exec_ctx.executed_parent_key,
                    "selected_parent_key_raw": None if exec_ctx.selected_parent_key is None else str(exec_ctx.selected_parent_key),
                    "child_key_raw": key,
                    "parent_eff_mse": executed_parent_eff_mse_before,
                    "child_eff_mse": float(mse_eff),
                    "child_raw_mse": float(mse),
                })

        if scheduler_plan_decision is not None:
            realized_row = (
                inverse_experiment_log[int(row_idx)]
                if inverse_experiment_log is not None and row_idx is not None and 0 <= int(row_idx) < len(inverse_experiment_log)
                else None
            )
            _record_scheduler_outcome(
                decision=scheduler_plan_decision,
                decision_log_index=scheduler_decision_log_idx,
                current_iter=int(n_evaluated),
                applied=bool(_scheduler_control_route_applied(action)),
                selected_action=action,
                inverse_row_index=row_idx,
                inverse_row=realized_row,
                local_parent_eff_mse=executed_parent_eff_mse_before,
                child_eff_mse=float(mse_eff),
                global_best_eff_mse_before=float(best_mse_before_update),
                status="" if realized_row is None else str(realized_row.get("status", "") or ""),
            )

        if executed_hole_opp is not None:
            _record_hole_search_frontier_outcome(
                executed_hole_opp,
                current_iter=int(n_evaluated),
                exact_eff_mse=float(mse_eff),
                shortlist_eff_mse=executed_hole_shortlist_eff_mse,
                reward=hole_outcome_reward,
                wall_s=executed_hole_wall_s if executed_hole_wall_s is not None else route_wall_s,
                parent_eff_mse=executed_parent_eff_mse_before,
                accepted=hole_outcome_accepted,
                status=str(hole_outcome_status or "scored"),
            )

        if mse_eff < best_mse:
            best_mse = mse_eff

        child_rec_after_update = arch.d.get(key, None)
        delta_log_improve = 0.0
        try:
            if (
                executed_parent_eff_mse_before is not None
                and math.isfinite(float(executed_parent_eff_mse_before))
                and math.isfinite(float(mse_eff))
            ):
                delta_log_improve = float(
                    math.log(float(executed_parent_eff_mse_before) + 1.0e-30)
                    - math.log(float(mse_eff) + 1.0e-30)
                )
            elif math.isfinite(float(best_mse_before_update)) and math.isfinite(float(mse_eff)):
                delta_log_improve = float(
                    math.log(float(best_mse_before_update) + 1.0e-30)
                    - math.log(float(mse_eff) + 1.0e-30)
                )
        except Exception:
            delta_log_improve = 0.0
        residual_basin_best_after = False
        if child_rec_after_update is not None:
            try:
                residual_basin_best_after = (
                    node_str(getattr(child_rec_after_update, "best_expr", None)) == node_str(scored_expr)
                    and abs(float(getattr(child_rec_after_update, "best_mse", float("inf"))) - float(mse_eff))
                    <= max(1.0e-12, 1.0e-9 * max(abs(float(mse_eff)), abs(float(getattr(child_rec_after_update, "best_mse", mse_eff))), 1.0))
                )
            except Exception:
                residual_basin_best_after = False
        should_abstract_improve = (
            bool(hole_search_abstraction_enable)
            and bool(hole_search_abstraction_on_improve)
            and (bool(became_global_best) or bool(residual_basin_best_after))
            and float(delta_log_improve) >= float(hole_search_abstraction_improve_min_delta_log_mse)
        )
        if should_abstract_improve and hole_frontier is not None:
            child_rec = child_rec_after_update
            child_expr = getattr(child_rec, "best_expr", scored_expr) if child_rec is not None else scored_expr
            child_mapping = getattr(child_rec, "mapping", mapping) if child_rec is not None else mapping
            child_elite_id = str(getattr(child_rec, "best_elite_id", "") or "") if child_rec is not None else ""
            child_eff_mse = None
            if child_rec is not None:
                try:
                    child_eff_mse = float(getattr(child_rec, "best_mse", mse_eff))
                except Exception:
                    child_eff_mse = float(mse_eff)
            else:
                child_eff_mse = float(mse_eff)
            _emit_hole_abstraction(
                parent_key=key,
                parent_elite_id=child_elite_id,
                parent_expr=child_expr,
                parent_mapping=child_mapping,
                parent_eff_mse=child_eff_mse,
                current_iter=int(n_evaluated),
                source="abstraction_improve",
            )

        # Global solved criterion: apply the same threshold in mutation.
        if best_raw_mse_struct < early_stop_mse:
            if stop_event is not None:
                stop_event.set()  # signal other threads immediately
            search_stop_reason = "early_stop_mse"
            if verbose:
                print(
                    f"[mutate] early-stop: best_struct_mse={best_raw_mse_struct:.3e} "
                    f"< early_stop_mse={early_stop_mse:.3e}"
                )
            break

        pe = print_every if print_every > 0 else 2000
        if verbose and n_evaluated % pe == 0:
            if arch.d:
                b = arch.best(1)[0]
                if print_every > 0:
                    if macro_controller is not None:
                        na = len(tracked_actions)
                        act_summ = " ".join([f"{a}:{mu:+.3g}" for mu, a, _ in macro_controller.summary(topk=na)])
                    else:
                        na = len(active_actions)
                        act_summ = " ".join([f"{ACTION_NAME[a]}:{mu:+.3g}" for mu, a, _ in explorer.summary(topk=na)])
                    print(f"iter {n_evaluated} eval {arch.n_eval} residual_basins {len(arch.d)} best_mse {b.best_mse:.6g} best {node_str(b.best_expr)} | {act_summ}")
                else:
                    print(f"[mutate] {n_evaluated}/{n_iter} evals, residual_basins={len(arch.d)}, best_mse={b.best_mse:.3e}")
            else:
                print(f"[mutate] {n_evaluated}/{n_iter} evals, residual_basins=0, best_mse=inf")

            eval_win = max(1, int(n_evaluated - last_progress_eval))
            residual_basin_win = max(0, int(len(arch.d) - last_progress_residual_basins))
            new_residual_basin_rate = float(residual_basin_win) / float(eval_win)
            proposed_total = int(sum(int(v) for v in action_proposed_counts.values()))
            accepted_total = int(sum(int(v) for v in action_accepted_counts.values()))
            seed_tag = int(seed_search if seed_search is not None else seed)
            print(
                f"[metrics] iter={int(n_evaluated)} seed_search={seed_tag} residual_basins={int(len(arch.d))} "
                f"new_residual_basin_rate={float(new_residual_basin_rate):.6g} proposed={proposed_total} accepted={accepted_total}"
            )
            last_progress_eval = int(n_evaluated)
            last_progress_residual_basins = int(len(arch.d))

        if _maybe_run_stall_check(int(n_evaluated)):
            break
        if (
            stall_window > 0
            and int(n_evaluated) > 0
            and (int(n_evaluated) % int(stall_window)) == 0
            and _degenerate_abort_should_stop(
                n_evaluated=int(n_evaluated),
                accepted_total=int(sum(int(v) for v in action_accepted_counts.values())),
                start_best=float(mutate_start_best),
                current_best=(arch.best(1)[0].best_mse if arch.d else float("inf")),
                enable=bool(degenerate_abort_enable),
                min_evals=int(degenerate_abort_min_evals),
                max_accepted=int(degenerate_abort_max_accepted),
                stall_delta=float(stall_delta),
            )
        ):
            search_stop_reason = "degenerate_abort"
            if verbose:
                accepted_da = int(sum(int(v) for v in action_accepted_counts.values()))
                best_da = arch.best(1)[0].best_mse if arch.d else float("inf")
                print(
                    f"[mutate] degenerate abort at iter {int(n_evaluated)}: "
                    f"accepted={accepted_da}, best_mse={best_da:.3e} unchanged since brute phase"
                )
            break
        if (
            search_stop_reason is None
            and bool(periodic_slate_refine_enable)
            and int(n_evaluated) > 0
            and (int(n_evaluated) % int(refine_slate_period_i)) == 0
            and int(last_refine_slate_eval) != int(n_evaluated)
        ):
            last_refine_slate_eval = int(n_evaluated)
            if _run_refinement_slate_pass("periodic"):
                search_stop_reason = "early_stop_mse"
                break

    if search_stop_reason is None:
        _run_archive_repair_pass()

    if search_stop_reason is None and bool(final_polish_refine_enable):
        if _run_refinement_slate_pass("final_polish"):
            search_stop_reason = "early_stop_mse"

    phase_timing["mutation_wall_s"] = float(phase_timing.get("mutation_wall_s", 0.0) or 0.0) + float(
        time.perf_counter() - mutation_started
    )

    if search_stop_reason is None:
        if plateau_stop_requested:
            search_stop_reason = "plateau"
        elif n_evaluated >= n_iter:
            search_stop_reason = "n_iter"
        elif n_attempts >= max_attempts:
            search_stop_reason = "max_attempts"
        elif _wall_time_exceeded():
            search_stop_reason = "wall_time_limit"
        elif stop_event is not None and stop_event.is_set():
            search_stop_reason = "stop_event"
        else:
            search_stop_reason = "completed"
    _finalize_search_state(search_stop_reason)

    if verbose:
        if arch.d:
            b = arch.best(1)[0]
            print(f"[mutate] done: {n_evaluated} evals, residual_basins={len(arch.d)}, best_mse={b.best_mse:.3e}")
        seed_tag = int(seed_search if seed_search is not None else seed)
        proposed_total = int(sum(int(v) for v in action_proposed_counts.values()))
        accepted_total = int(sum(int(v) for v in action_accepted_counts.values()))
        eval_win = max(1, int(n_evaluated - last_progress_eval))
        residual_basin_win = max(0, int(len(arch.d) - last_progress_residual_basins))
        new_residual_basin_rate = float(residual_basin_win) / float(eval_win)
        print(
            f"[metrics] iter={int(n_evaluated)} seed_search={seed_tag} residual_basins={int(len(arch.d))} "
            f"new_residual_basin_rate={float(new_residual_basin_rate):.6g} proposed={proposed_total} accepted={accepted_total}"
        )

    # Expose fit/probe data so callers (e.g. bridge.py) can refit mappings
    arch.x_fit = x_fit
    arch.y_fit = y_fit
    arch.x_probe = x_probe
    arch.y_probe = y_probe
    arch.crossover_policy_stats = _finalize_crossover_policy_stats(crossover_policy_stats)
    arch.action_distribution = _finalize_action_distribution(
        tracked_actions,
        action_selected_counts,
        action_proposed_counts,
        action_reward_counts,
        action_accepted_counts,
    )

    if verbose:
        cps = arch.crossover_policy_stats
        if isinstance(cps, dict):
            pol_rows = []
            for pol, st in cps.items():
                try:
                    sel = int(st.get("selected", 0))
                except Exception:
                    sel = 0
                if sel <= 0:
                    continue
                pol_rows.append(
                    f"{pol} sel={sel} prop={int(st.get('proposed', 0))} "
                    f"acc={int(st.get('accepted', 0))} avg_r={float(st.get('avg_reward', 0.0)):+.3g}"
                )
            if pol_rows:
                print("[mutate] crossover policy stats: " + " | ".join(pol_rows))
        ad = arch.action_distribution
        if isinstance(ad, dict):
            cc = ad.get("counts", {})
            tot = int(ad.get("total_selected", 0))
            if isinstance(cc, dict) and tot > 0:
                top = sorted(cc.items(), key=lambda kv: kv[1], reverse=True)
                top = [f"{k}:{int(v)}" for k, v in top if int(v) > 0]
                if top:
                    print(f"[mutate] action distribution (selected, n={tot}): {' '.join(top)}")
        igs = inverse_gate_stats
        if isinstance(igs, dict):
            considered = int(igs.get("considered", 0))
            if considered > 0:
                print(
                    "[mutate] inverse gate: "
                    f"considered={considered} allowed={int(igs.get('allowed', 0))} "
                    f"blocked_quality={int(igs.get('blocked_quality', 0))} "
                    f"blocked_structure={int(igs.get('blocked_structure', 0))} "
                    f"blocked_gain={int(igs.get('blocked_gain', 0))}"
                )
        oss = closure_search_stats
        if isinstance(oss, dict) and bool(oss.get("enabled", False)):
            scaffolds = int(oss.get("scaffolds_considered", 0))
            preview_calls = int(oss.get("preview_calls", 0))
            if scaffolds > 0 or preview_calls > 0:
                print(
                    "[closure-search] "
                    f"scaffolds={scaffolds} "
                    f"preview_calls={preview_calls} "
                    f"preview_candidates={int(oss.get('preview_candidates', 0))} "
                    f"direct_calls={int(oss.get('direct_calls', 0))} "
                    f"anchor_lift={int(oss.get('direct_anchor_lift_applied', 0))}/{int(oss.get('direct_anchor_lift_attempts', 0))} "
                    f"scored={int(oss.get('scored', 0))} "
                    f"emergent_rows={int(oss.get('emergent_basis_rows', 0))} "
                    f"emergent_scored={int(oss.get('emergent_basis_scored', 0))} "
                    f"aux_atoms={int(oss.get('emergent_aux_atom_registry_size', 0))} "
                    f"aux_seeds={int(oss.get('emergent_aux_atom_seed_blocks', 0))} "
                    f"aux_seen={int(oss.get('emergent_aux_atom_observation_pool_size', 0))} "
                    f"aux_reserved={int(oss.get('emergent_aux_atom_followup_reserved', 0))} "
                    f"atom_policy={int(oss.get('atom_policy_library_records', 0))}/"
                    f"{int(oss.get('atom_policy_library_relations', 0))}"
                    f"@{int(oss.get('atom_policy_source_atoms', 0))} "
                    f"atomized={int(oss.get('atomized_linear_span_rows', 0))}"
                    f"/{int(oss.get('atomized_linear_span_scored', 0))} "
                    f"atomized_cov={int(oss.get('atomized_linear_span_coverage_candidates', 0))} "
                    f"aux_scaffolds={int(oss.get('aux_scaffolds_enumerated', 0))} "
                    f"protected_aux={int(oss.get('protected_aux_scaffolds_enumerated', 0))} "
                    f"new_residual_basins={int(oss.get('new_residual_basins', 0))} "
                    f"global_best_updates={int(oss.get('global_best_updates', 0))}"
                )
        rps = repair_pass_stats
        if isinstance(rps, dict) and bool(rps.get("enabled", False)):
            considered = int(rps.get("elites_considered", 0))
            solver_calls = int(rps.get("solver_calls", 0))
            if considered > 0 or solver_calls > 0:
                print(
                    "[repair-pass] "
                    f"elites={considered} "
                    f"improved={int(rps.get('elites_improved', 0))} "
                    f"solver_calls={solver_calls} "
                    f"accepted={int(rps.get('accepted_repairs', 0))} "
                    f"evals={int(rps.get('evals_used', 0))} "
                    f"new_residual_basins={int(rps.get('new_residual_basins', 0))} "
                    f"global_best_updates={int(rps.get('global_best_updates', 0))}"
                )
        rcs = repair_controller_stats
        if isinstance(rcs, dict) and bool(rcs.get("enabled", False)):
            considered = int(rcs.get("considered", 0))
            if considered > 0:
                score_hist = list(rcs.get("score_hist", []) or [])
                mean_score = (sum(float(v) for v in score_hist) / len(score_hist)) if score_hist else 0.0
                cur_thr = _repair_controller_threshold(rcs)
                print(
                    "[mutate] repair controller: "
                    f"considered={considered} selected={int(rcs.get('selected', 0))} "
                    f"option_repair_selected={int(rcs.get('option_repair_selected', 0))} "
                    f"blocked_low_score={int(rcs.get('blocked_low_score', 0))} "
                    f"blocked_low_concentration={int(rcs.get('blocked_low_concentration', 0))} "
                    f"blocked_retry_cooldown={int(rcs.get('blocked_retry_cooldown', 0))} "
                    f"blocked_retry_repeat_budget={int(rcs.get('blocked_retry_repeat_budget', 0))} "
                    f"blocked_retry_repeat_signature={int(rcs.get('blocked_retry_repeat_signature', 0))} "
                    f"no_candidate={int(rcs.get('no_candidate', 0))} "
                    f"mean_score={float(mean_score):.3f} "
                    f"threshold={float(cur_thr):.3f} "
                    f"critic_mode={str(rcs.get('critic_mode', 'priority'))} "
                    f"parent_repair_selected={int(rcs.get('parent_repair_selected', 0))}"
                )
        mcs = macro_controller_stats
        if isinstance(mcs, dict) and bool(mcs.get("enabled", False)):
            policy_counts = mcs.get("policy_counts", {})
            if isinstance(policy_counts, dict) and policy_counts:
                rows = [f"{k}:{int(v)}" for k, v in sorted(policy_counts.items(), key=lambda kv: kv[1], reverse=True) if int(v) > 0]
                if rows:
                    print(
                        "[mutate] macro controller: "
                        f"selected={int(mcs.get('selected', 0))} "
                        f"repair_selected={int(mcs.get('repair_selected', 0))} "
                        f"fallback_selected={int(mcs.get('fallback_selected', 0))} "
                        f"policy_counts={' '.join(rows)}"
                    )

    arch.boost_gate_stats = boost_gate_stats
    arch.inverse_gate_stats = inverse_gate_stats
    arch.repair_pass_stats = repair_pass_stats
    arch.closure_search_stats = closure_search_stats
    arch.hole_search_stats = hole_search_stats
    arch.score_prescreen_stats = score_prescreen_stats
    arch.gs_carrier_unit_stats = gs_carrier_seed_stats
    gs_fss_report_payload = gs_fss_report(gs_fss_context_runtime)
    arch.gs_fss_report = gs_fss_report_payload
    arch.expr_ir_report = expression_ir_report(
        expr_ir_runtime_cfg,
        expr_ir_stats,
        extra={"pool_terms": int(len(pool_nodes)), "gs_fss": gs_fss_report_payload},
    )
    arch.repair_controller_stats = repair_controller_stats
    arch.macro_controller_stats = macro_controller_stats
    arch.scheduler_stats = scheduler_stats
    arch.scheduler_decision_log = scheduler_decision_log
    arch.scheduler_outcome_log = scheduler_outcome_log
    route_scheduler_stats["route_summary"] = route_scheduler.summary() if route_scheduler is not None else {}
    arch.route_scheduler_stats = route_scheduler_stats
    if macro_controller is not None:
        arch.macro_controller_summary = macro_controller.summary(topk=len(tracked_actions))
    if inverse_experiment_log is not None:
        _annotate_inverse_experiment_lineage(
            inverse_experiment_log,
            lineage_events,
            horizon=int(actor_critic_descendant_horizon),
            eps=float(actor_critic_reward_eps),
        )
        arch.inverse_experiment_log = inverse_experiment_log
    return arch

__engine_search_definitions__ = (
    "_collect_controller_build_slate",
    "apply_action",
    "apply_crossover_action",
    "apply_residual_action",
    "_bind_runtime_hooks",
    "run_explorer_core",
)

__engine_search_constants__ = (

)

__engine_search_late_bindings__ = (

)
