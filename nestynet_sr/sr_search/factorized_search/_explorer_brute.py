# ruff: noqa: F401, F821, F841
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Legacy enumeration, structural presearch, and brute-phase helpers."""

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
        }
    return out


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

# --- brute-force enumeration ---

def enumerate_trees(max_depth, nvars, max_trees=None):
    return _shared_enumerate_trees(max_depth, nvars, max_trees=max_trees)


def enumerate_trees_dim(max_depth, nvars, var_dims, y_dims, max_trees=None):
    return _shared_enumerate_trees_dim(max_depth, nvars, var_dims, y_dims, max_trees=max_trees)


def _has_const_zero(node):
    """Return True if the tree contains a ("const", 0.0) node anywhere."""
    op = node[0]
    if op == "const":
        return node[1] == 0.0
    if op == "var":
        return False
    if op in UNARY_OPS:
        return _has_const_zero(node[1])
    if op in BINARY_OPS:
        return _has_const_zero(node[1]) or _has_const_zero(node[2])
    return False


def _dedup_new(raw_trees, seen):
    """Simplify, deduplicate against *seen*, return new unique trees.

    Updates *seen* in place so subsequent calls skip already-seen
    canonical forms.
    """
    new_unique = []
    for t in raw_trees:
        s = simplify(t)
        canon = _mapping_equiv_root(s, assume_simplified=True)
        if _has_const_zero(canon):
            continue
        key = node_str(canon)
        if key not in seen:
            seen.add(key)
            new_unique.append(canon)
    return new_unique


def _auto_brute_depth(nvars):
    """Ceiling for brute enumeration depth.

    The adaptive budget check inside ``enumerate_trees`` /
    ``enumerate_trees_dim`` will stop earlier when the projected tree
    count exceeds ``brute_max_expressions``.  This ceiling is just a
    hard upper bound.

    Binary ops produce O(N²) candidates per depth, so 2+ variables
    explode quickly — cap at depth 3 to keep enumeration tractable.
    """
    if nvars >= 2:
        return 3
    return 10


def _enumerate_incremental(max_depth, nvars, max_trees=None, *, verbose=True):
    """Yield ``(depth, new_unique_trees)`` for each depth.

    Trees are simplified and deduplicated across all depths so no
    expression is yielded twice.  Stops before any depth whose projected
    raw count would exceed *max_trees*.
    """
    n_un = len(UNARY_OPS)
    n_bin = len(BINARY_OPS)
    up_to = [("var", i) for i in range(nvars)]
    seen = set()

    yield 1, _dedup_new(up_to, seen)

    for _depth in range(2, max_depth + 1):
        N = len(up_to)
        projected = N + n_un * N + n_bin * N * N
        if max_trees is not None and projected > max_trees:
            if verbose:
                print(f"[brute]  adaptive: depth {_depth} would produce ~{projected:,} "
                      f"raw trees (budget {max_trees:,}), stopping at depth {_depth - 1}")
            return
        new_raw = []
        for op in UNARY_OPS:
            for t in up_to:
                new_raw.append((op, t))
        for op in BINARY_OPS:
            for l in up_to:
                for r in up_to:
                    new_raw.append((op, l, r))
        yield _depth, _dedup_new(new_raw, seen)
        up_to = up_to + new_raw


def _enumerate_dim_incremental(
    max_depth,
    nvars,
    var_dims,
    y_dims,
    max_trees=None,
    *,
    verbose=True,
):
    """Yield ``(depth, new_y_dim_trees)`` for each depth (dim-aware).

    Builds only dimensionally valid trees by construction (same logic as
    ``enumerate_trees_dim``).  At each depth, only trees matching
    *y_dims* that are new since the last depth are yielded.

    When *y_dims* is ``None``, all dimensionally valid trees are yielded
    (input-side filtering only — no transcendental functions of unitful args).
    """
    ndim = len(var_dims[0])
    dim0 = (0.0,) * ndim
    by_dim: dict[tuple, dict[str, tuple]] = {}

    # When y_dims is None we yield trees from ALL dimension buckets.
    have_y = y_dims is not None
    y_key = dim_round(tuple(y_dims)) if have_y else None

    def _add(tree, dim):
        s = simplify(tree)
        if _has_const_zero(s):
            return
        key = node_str(s)
        bucket = by_dim.setdefault(dim, {})
        if key not in bucket:
            bucket[key] = s

    def _total():
        return sum(len(b) for b in by_dim.values())

    def _canon_key_tree(tree):
        canon = _mapping_equiv_root(tree, assume_simplified=True)
        return node_str(canon), canon

    def _collect_new(prev_keys_by_dim):
        """Collect new trees, optionally filtered to y_key bucket.

        Deduplicates by mapping-equivalent canonical form (strips
        top-level neg, canonicalises top-level sub order).
        """
        raw = []
        if have_y:
            bucket = by_dim.get(y_key, {})
            prev = prev_keys_by_dim.get(y_key, set())
            raw = [tree for key, tree in bucket.items() if key not in prev]
            prev_keys_by_dim[y_key] = set(bucket.keys())
        else:
            for dim, bucket in by_dim.items():
                prev = prev_keys_by_dim.get(dim, set())
                for key, tree in bucket.items():
                    if key not in prev:
                        raw.append(tree)
                prev_keys_by_dim[dim] = set(bucket.keys())
        # Deduplicate by mapping-equivalent canonical root
        seen_canon = set()
        out = []
        for tree in raw:
            ck, canon = _canon_key_tree(tree)
            if ck not in seen_canon:
                seen_canon.add(ck)
                if not _has_const_zero(canon):
                    out.append(canon)
        return out

    # Depth 1: leaves
    for i in range(nvars):
        _add(("var", i), dim_round(var_dims[i]))

    prev_keys: dict[tuple, set] = {}
    new_trees = _collect_new(prev_keys)
    yield 1, new_trees

    prev_total = _total()

    for _depth in range(2, max_depth + 1):
        cur_total = _total()
        if max_trees is not None and _depth > 2:
            ratio = cur_total / max(prev_total, 1)
            projected = int(cur_total * ratio)
            if projected > max_trees:
                if verbose:
                    print(f"[brute]  adaptive: depth {_depth} would produce ~{projected:,} "
                          f"raw trees (budget {max_trees:,}), "
                          f"stopping at depth {_depth - 1}")
                return
        prev_total = cur_total

        # Expand one depth
        all_trees = [(dim, tree) for dim, bucket in by_dim.items()
                     for tree in bucket.values()]
        new_entries = []
        for dim, tree in all_trees:
            new_entries.append((dim, ("neg", tree)))
            if dim == dim0:
                for op in ("sin", "cos", "exp", "log"):
                    new_entries.append((dim0, (op, tree)))
            new_entries.append((dim_round(tuple(x / 2 for x in dim)), ("sqrt", tree)))
            new_entries.append((dim_round(tuple(x * 2 for x in dim)), ("sqr", tree)))
        # Binary ops: interleave add/sub/mul/div for each (L,R) pair so
        # that all four operation types appear in balanced order when
        # the budget truncates the tree list.
        dim_list = list(by_dim.keys())
        for d1 in dim_list:
            bucket1 = list(by_dim[d1].values())
            for d2 in dim_list:
                bucket2 = list(by_dim[d2].values())
                same_dim = (d1 == d2)
                d_mul = dim_round(tuple(a + b for a, b in zip(d1, d2)))
                d_div = dim_round(tuple(a - b for a, b in zip(d1, d2)))
                for L in bucket1:
                    for R in bucket2:
                        if same_dim:
                            new_entries.append((d1, ("add", L, R)))
                            new_entries.append((d1, ("sub", L, R)))
                        new_entries.append((d_mul, ("mul", L, R)))
                        new_entries.append((d_div, ("div", L, R)))
        for dim, tree in new_entries:
            _add(tree, dim)

        new_trees = _collect_new(prev_keys)
        yield _depth, new_trees


def _build_monomial_ast(exponents):
    """Build a tuple-AST for the monomial ``∏ x_i^{a_i}``."""
    def _power_node(var_idx, exp):
        v = ("var", var_idx)
        aexp = abs(exp)
        if aexp == 1:
            node = v
        elif aexp == 2:
            node = ("sqr", v)
        elif aexp == 3:
            node = ("mul", v, ("sqr", v))
        else:
            # Fallback for |exp| > 3: chain of multiplications.
            node = v
            for _ in range(aexp - 1):
                node = ("mul", node, v)
        if exp < 0:
            node = ("div", ("const", 1.0), node)
        return node

    factors = []
    for i, a in enumerate(exponents):
        if a != 0:
            factors.append(_power_node(i, a))
    if not factors:
        return ("const", 1.0)
    result = factors[0]
    for f in factors[1:]:
        result = ("mul", result, f)
    return simplify(result)


def _monomial_presearch(
    arch, nvars, var_dims, y_dims,
    x_fit, y_fit, x_probe, y_probe,
    proj, fp_mode, q_scale, q_clip, poly_degree,
    *,
    max_exp=3,
    max_complexity=8,
    refine_enable=False,
    refine_cfg=None,
    refine_state=None,
    early_stop_mse=1e-10,
    verbose=True,
):
    """Enumerate monomial expressions ``c · ∏ x_i^{a_i}`` that satisfy
    dimensional constraints.  Much faster than full tree enumeration for
    products of variables — O(k^n) exponent vectors with cheap dim filter.
    """
    import numpy as np

    dim_arr = np.array([list(d) for d in var_dims], dtype=np.float64)
    y_arr = np.array(list(y_dims), dtype=np.float64)

    # Enumerate exponent vectors and filter by dimensional consistency.
    candidates = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, y_arr, atol=1e-10):
            candidates.append(exps)

    if not candidates:
        return False

    # Cheap pre-screen: evaluate each monomial, fit y ≈ c*mono, keep top-N.
    # This avoids expensive score_expr calls for thousands of candidates.
    _MAX_FULL_SCORE = 50
    if len(candidates) > _MAX_FULL_SCORE:
        cheap_scores = []
        for exps in candidates:
            mono = torch.ones(x_fit.shape[0], dtype=x_fit.dtype, device=x_fit.device)
            for i, a in enumerate(exps):
                if a != 0:
                    mono = mono * x_fit[:, i].pow(a)
            if not torch.isfinite(mono).all() or float(mono.abs().max()) < 1e-30:
                cheap_scores.append((float("inf"), exps))
                continue
            y_sq = y_fit.squeeze(-1)
            c = (y_sq * mono).sum() / ((mono * mono).sum() + 1e-30)
            mse_cheap = float(((y_sq - c * mono) ** 2).mean())
            cheap_scores.append((mse_cheap, exps))
        cheap_scores.sort(key=lambda t: t[0])
        candidates = [exps for _, exps in cheap_scores[:_MAX_FULL_SCORE]]

    # Build ASTs, score through the standard pipeline.
    best_mse = float("inf")
    best_mse_struct = float("inf")
    n_scored = 0
    brute_refine_cfg = dict(refine_cfg or {})
    brute_refine_cfg["score_mapping_family_mode"] = str(
        brute_refine_cfg.get(
            "brute_score_mapping_family_mode",
            brute_refine_cfg.get("score_mapping_family_mode", "full"),
        ) or "full"
    )
    for exps in candidates:
        tree = _build_monomial_ast(exps)
        best_for_gate = (
            float(best_mse_struct)
            if math.isfinite(best_mse_struct)
            else float(max(best_mse, float(early_stop_mse)))
        )
        sc = score_expr(
            tree, x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=brute_refine_cfg,
            refine_best_mse=best_for_gate, refine_state=refine_state,
            return_expr=True,
        )
        if sc is None:
            continue
        mse, key, z, mapping, scored_tree = sc
        arch.update(key, mse, scored_tree, z, mapping, raw_mse=mse)
        n_scored += 1
        if mse < best_mse:
            best_mse = mse
        if mapping_is_structural(mapping) and mse < best_mse_struct:
            best_mse_struct = mse

    solved = best_mse_struct < early_stop_mse
    if verbose and n_scored > 0:
        tag = " *** STRUCT-SOLVED ***" if solved else ""
        print(f"[brute]  monomial presearch: {len(candidates)} dim-ok, "
              f"scored {n_scored}, best_mse={best_mse:.3e}{tag}")
    return solved


def _lorentz_peel_presearch(
    arch, nvars, var_dims, y_dims,
    x_fit, y_fit, x_probe, y_probe,
    proj, fp_mode, q_scale, q_clip, poly_degree,
    *,
    max_exp=3,
    max_complexity=8,
    safe_margin_eps=0.01,
    max_affine_terms=3,
    refine_enable=False,
    refine_cfg=None,
    refine_state=None,
    early_stop_mse=1e-10,
    verbose=True,
):
    """Targeted fast path for invsqrt_mul power family (Lorentz factor).

    For each dimensionless ratio pair u = x_i/x_j with safe margin,
    peel gamma = 1/sqrt(1 - u²) from y and search for the numerator
    via monomial + affine fits.  Builds full AST and scores via the
    standard pipeline.
    """
    import numpy as np

    dim0 = tuple(0.0 for _ in var_dims[0])
    y_sq_fit = y_fit.squeeze(-1)
    y_sq_probe = y_probe.squeeze(-1)

    # Identify dimensionless ratio pairs with safe margin.
    ratio_pairs = []
    for i in range(nvars):
        for j in range(nvars):
            if i == j:
                continue
            # Guard: carrier must be dimensionless.
            pair_dim = dim_round(tuple(a - b for a, b in zip(var_dims[i], var_dims[j])))
            if not dims_eq(pair_dim, dim0):
                continue
            u_fit = x_fit[:, i] / x_fit[:, j]
            u_probe = x_probe[:, i] / x_probe[:, j]
            om_fit = 1.0 - u_fit ** 2
            om_probe = 1.0 - u_probe ** 2
            # Guard: safe margin — q05(1 - u²) > eps on both splits.
            q05_fit = float(torch.quantile(om_fit, 0.05))
            q05_probe = float(torch.quantile(om_probe, 0.05))
            if q05_fit <= safe_margin_eps or q05_probe <= safe_margin_eps:
                continue
            gamma_fit = 1.0 / torch.sqrt(om_fit)
            gamma_probe = 1.0 / torch.sqrt(om_probe)
            if not (torch.isfinite(gamma_fit).all() and torch.isfinite(gamma_probe).all()):
                continue
            ratio_pairs.append((i, j, gamma_fit, gamma_probe))

    if not ratio_pairs:
        return False

    # Dimensional metadata for monomial search on peeled target.
    # Guard: peeled target preserves y dimension (gamma is dimensionless).
    dim_arr = np.array([list(d) for d in var_dims], dtype=np.float64)
    y_arr = np.array(list(y_dims), dtype=np.float64)

    brute_refine_cfg = dict(refine_cfg or {})
    brute_refine_cfg["score_mapping_family_mode"] = str(
        brute_refine_cfg.get(
            "brute_score_mapping_family_mode",
            brute_refine_cfg.get("score_mapping_family_mode", "full"),
        ) or "full"
    )

    # Pre-enumerate monomial exponent vectors (shared across all ratios).
    mono_candidates = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, y_arr, atol=1e-10):
            mono_candidates.append(exps)

    best_mse = float("inf")
    best_mse_struct = float("inf")
    n_scored = 0
    n_pairs_tried = len(ratio_pairs)

    def _score_tree(tree):
        nonlocal best_mse, best_mse_struct, n_scored
        best_for_gate = (
            float(best_mse_struct)
            if math.isfinite(best_mse_struct)
            else float(max(best_mse, float(early_stop_mse)))
        )
        sc = score_expr(
            tree, x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=brute_refine_cfg,
            refine_best_mse=best_for_gate, refine_state=refine_state,
            return_expr=True,
        )
        if sc is None:
            return
        mse, key, z, mapping, scored_tree = sc
        arch.update(key, mse, scored_tree, z, mapping, raw_mse=mse)
        n_scored += 1
        if mse < best_mse:
            best_mse = mse
        if mapping_is_structural(mapping) and mse < best_mse_struct:
            best_mse_struct = mse

    def _lorentz_denom(vi, vj):
        return ("sqrt", ("sub", ("const", 1.0), ("sqr", ("div", ("var", vi), ("var", vj)))))

    for vi, vj, gamma_fit, gamma_probe in ratio_pairs:
        denom_ast = _lorentz_denom(vi, vj)

        # Peel: y_adj = y * sqrt(1 - u²) = y / gamma.
        y_adj_fit = y_sq_fit / gamma_fit
        y_adj_probe = y_sq_probe / gamma_probe

        # (a) Monomial numerator search on peeled target.
        if mono_candidates:
            cheap = []
            for exps in mono_candidates:
                mono = torch.ones_like(y_adj_fit)
                for k, a in enumerate(exps):
                    if a != 0:
                        mono = mono * x_fit[:, k].pow(a)
                if not torch.isfinite(mono).all() or float(mono.abs().max()) < 1e-30:
                    continue
                c = float((y_adj_fit * mono).sum() / ((mono * mono).sum() + 1e-30))
                mse_cheap = float(((y_adj_fit - c * mono) ** 2).mean())
                cheap.append((mse_cheap, exps))
            cheap.sort(key=lambda t: t[0])
            for mse_c, exps in cheap[:5]:
                numer_ast = _build_monomial_ast(exps)
                tree = simplify(("div", numer_ast, denom_ast))
                _score_tree(tree)

        # (b) Affine numerator: y_adj ≈ c0 + sum(c_k * x_k) using dim-ok vars.
        affine_cols_fit = []
        affine_cols_probe = []
        affine_terms = []
        for k in range(nvars):
            if dims_eq(dim_round(var_dims[k]), dim_round(tuple(y_dims))):
                affine_cols_fit.append(x_fit[:, k])
                affine_cols_probe.append(x_probe[:, k])
                affine_terms.append(("var", k))
        # Also try pairwise products that match y_dims.
        for k1 in range(nvars):
            for k2 in range(k1, nvars):
                pd = dim_round(tuple(a + b for a, b in zip(var_dims[k1], var_dims[k2])))
                if dims_eq(pd, dim_round(tuple(y_dims))):
                    affine_cols_fit.append(x_fit[:, k1] * x_fit[:, k2])
                    affine_cols_probe.append(x_probe[:, k1] * x_probe[:, k2])
                    affine_terms.append(("mul", ("var", k1), ("var", k2)))
        if len(affine_terms) >= 1 and len(affine_terms) <= 20:
            ones_fit = torch.ones_like(y_adj_fit)
            Phi_fit = torch.stack([ones_fit] + affine_cols_fit, dim=1)
            Phi_probe = torch.stack([torch.ones_like(y_adj_probe)] + affine_cols_probe, dim=1)
            try:
                sol = torch.linalg.lstsq(Phi_fit, y_adj_fit.unsqueeze(-1)).solution.squeeze(-1)
            except Exception:
                sol = None
            if sol is not None and torch.isfinite(sol).all():
                y_hat_probe = (Phi_probe @ sol.unsqueeze(-1)).squeeze(-1)
                mse_affine = float(((y_adj_probe - y_hat_probe) ** 2).mean())
                if math.isfinite(mse_affine) and mse_affine < best_mse * 10:
                    # Build numerator AST from significant terms.
                    coeffs = sol.tolist()
                    bias = coeffs[0]
                    parts = []
                    for idx_t, (c_val, term) in enumerate(zip(coeffs[1:], affine_terms)):
                        if abs(c_val) < 1e-10 * max(abs(v) for v in coeffs):
                            continue
                        parts.append((c_val, term))
                    if parts:
                        # Assemble: (c1*t1 + c2*t2 + ...) / sqrt(1 - sqr(u))
                        # Let score_expr handle the mapping (absorbs coefficients).
                        numer = parts[0][1]
                        for _, term in parts[1:]:
                            numer = ("add", numer, term)
                        if abs(bias) > 1e-6 * max(abs(v) for v in coeffs):
                            numer = ("add", ("const", 1.0), numer)
                        tree = simplify(("div", numer, denom_ast))
                        _score_tree(tree)

        if best_mse_struct < early_stop_mse:
            break

    solved = best_mse_struct < early_stop_mse
    if verbose and n_scored > 0:
        tag = " *** STRUCT-SOLVED ***" if solved else ""
        print(f"[brute]  lorentz peel: {n_pairs_tried} ratio pairs, "
              f"scored {n_scored}, best_mse={best_mse:.3e}{tag}")
    return solved


def _planck_peel_presearch(
    arch, nvars, var_dims, y_dims,
    x_fit, y_fit, x_probe, y_probe,
    proj, fp_mode, q_scale, q_clip, poly_degree,
    *,
    max_exp=3,
    max_complexity=6,
    max_carriers=20,
    safe_exp_max=500.0,
    safe_carrier_min=1e-6,
    max_scored=200,
    refine_enable=False,
    refine_cfg=None,
    refine_state=None,
    early_stop_mse=1e-10,
    verbose=True,
):
    """Targeted fast path for Bose-Einstein / Planck functions.

    Enumerates dimensionless monomials u, then peels ``1/(exp(u)-1)``
    and ``u/(exp(u)-1)`` and ``(exp(u)-1)`` from y, searching for
    the numerator via monomial + affine fits.
    """
    import numpy as np

    dim0 = tuple(0.0 for _ in var_dims[0])
    dim_arr = np.array([list(d) for d in var_dims], dtype=np.float64)
    y_arr = np.array(list(y_dims), dtype=np.float64)
    y_sq_fit = y_fit.squeeze(-1)
    y_sq_probe = y_probe.squeeze(-1)

    # Find dimensionless monomials as carrier candidates.
    carrier_exps = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, np.zeros(len(dim0)), atol=1e-10):
            carrier_exps.append(exps)

    if not carrier_exps:
        return False

    # Pre-enumerate y-dim monomials for numerator search.
    numer_exps = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, y_arr, atol=1e-10):
            numer_exps.append(exps)

    brute_refine_cfg = dict(refine_cfg or {})
    brute_refine_cfg["score_mapping_family_mode"] = str(
        brute_refine_cfg.get(
            "brute_score_mapping_family_mode",
            brute_refine_cfg.get("score_mapping_family_mode", "full"),
        ) or "full"
    )

    best_mse = float("inf")
    best_mse_struct = float("inf")
    n_scored = 0
    n_carriers = len(carrier_exps)

    def _score_tree(tree):
        nonlocal best_mse, best_mse_struct, n_scored
        best_for_gate = (
            float(best_mse_struct)
            if math.isfinite(best_mse_struct)
            else float(max(best_mse, float(early_stop_mse)))
        )
        sc = score_expr(
            tree, x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=brute_refine_cfg,
            refine_best_mse=best_for_gate, refine_state=refine_state,
            return_expr=True,
        )
        if sc is None:
            return
        mse, key, z, mapping, scored_tree = sc
        arch.update(key, mse, scored_tree, z, mapping, raw_mse=mse)
        n_scored += 1
        if mse < best_mse:
            best_mse = mse
        if mapping_is_structural(mapping) and mse < best_mse_struct:
            best_mse_struct = mse

    def _eval_mono(exps, X):
        mono = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
        for k, a in enumerate(exps):
            if a != 0:
                mono = mono * X[:, k].pow(a)
        return mono

    def _planck_denom(u_ast):
        return ("sub", ("exp", u_ast), ("const", 1.0))

    def _search_numerator(y_adj_fit, y_adj_probe, factor_ast):
        """Try monomial numerators and score full expression."""
        if not numer_exps:
            return
        cheap = []
        for exps in numer_exps:
            mono = _eval_mono(exps, x_fit)
            if not torch.isfinite(mono).all() or float(mono.abs().max()) < 1e-30:
                continue
            c = float((y_adj_fit * mono).sum() / ((mono * mono).sum() + 1e-30))
            mse_c = float(((y_adj_fit - c * mono) ** 2).mean())
            cheap.append((mse_c, exps))
        cheap.sort(key=lambda t: t[0])
        for mse_c, exps in cheap[:5]:
            numer_ast = _build_monomial_ast(exps)
            tree = simplify(("mul", numer_ast, factor_ast))
            _score_tree(tree)

    # Trial scalings for the carrier: the Planck argument often has a
    # hidden constant (e.g. 1/(2π)) that the monomial can't represent.
    # Try a small grid of scalings to cover common physical constants.
    _TRIAL_SCALES = [1.0, 0.5, 0.2, 1.0 / (2 * math.pi), 0.1,
                     2.0, 5.0, 2 * math.pi, 10.0]

    for c_exps in carrier_exps[:max_carriers]:
        u_fit = _eval_mono(c_exps, x_fit)
        u_probe = _eval_mono(c_exps, x_probe)
        if not (torch.isfinite(u_fit).all() and torch.isfinite(u_probe).all()):
            continue
        # Try both signs of the carrier.
        for sign, s_exps in [(1, c_exps), (-1, tuple(-a for a in c_exps))]:
            su_fit_raw = u_fit * sign
            su_probe_raw = u_probe * sign
            if float(su_fit_raw.abs().min()) < safe_carrier_min:
                continue
            if float(su_probe_raw.abs().min()) < safe_carrier_min:
                continue

            base_ast = _build_monomial_ast(s_exps)

            for scale in _TRIAL_SCALES:
                su_fit = su_fit_raw * scale
                su_probe = su_probe_raw * scale
                # Clamp and compute expm1 stably.
                su_fit_c = torch.clamp(su_fit, max=safe_exp_max)
                su_probe_c = torch.clamp(su_probe, max=safe_exp_max)
                denom_fit = torch.expm1(su_fit_c)
                denom_probe = torch.expm1(su_probe_c)
                if not (torch.isfinite(denom_fit).all() and torch.isfinite(denom_probe).all()):
                    continue
                if float(denom_fit.abs().min()) < 1e-12:
                    continue
                if float(denom_probe.abs().min()) < 1e-12:
                    continue

                # Build AST: scale*u inside exp.  For scale=1, use raw monomial.
                if abs(scale - 1.0) < 1e-12:
                    u_ast = base_ast
                else:
                    u_ast = simplify(("mul", ("const", scale), base_ast))
                planck_denom = _planck_denom(u_ast)

                # Template 1: y = numerator / (exp(s*u) - 1)
                y_adj_fit = y_sq_fit * denom_fit
                y_adj_probe = y_sq_probe * denom_probe
                if torch.isfinite(y_adj_fit).all() and torch.isfinite(y_adj_probe).all():
                    factor_ast = ("div", ("const", 1.0), planck_denom)
                    _search_numerator(y_adj_fit, y_adj_probe, factor_ast)

                # Template 2: y = numerator * u / (exp(s*u) - 1)
                y_adj_fit2 = y_sq_fit * denom_fit / su_fit
                y_adj_probe2 = y_sq_probe * denom_probe / su_probe
                if torch.isfinite(y_adj_fit2).all() and torch.isfinite(y_adj_probe2).all():
                    factor_ast = ("div", u_ast, planck_denom)
                    _search_numerator(y_adj_fit2, y_adj_probe2, factor_ast)

                # Template 3: y = numerator * (exp(s*u) - 1)
                y_adj_fit3 = y_sq_fit / denom_fit
                y_adj_probe3 = y_sq_probe / denom_probe
                if torch.isfinite(y_adj_fit3).all() and torch.isfinite(y_adj_probe3).all():
                    factor_ast = planck_denom
                    _search_numerator(y_adj_fit3, y_adj_probe3, factor_ast)

                if best_mse_struct < early_stop_mse or n_scored >= max_scored:
                    break
            if best_mse_struct < early_stop_mse or n_scored >= max_scored:
                break
        if best_mse_struct < early_stop_mse or n_scored >= max_scored:
            break

    solved = best_mse_struct < early_stop_mse
    if verbose and n_scored > 0:
        tag = " *** STRUCT-SOLVED ***" if solved else ""
        print(f"[brute]  planck peel: {n_carriers} carriers, "
              f"scored {n_scored}, best_mse={best_mse:.3e}{tag}")
    return solved


def _hyperbolic_peel_presearch(
    arch, nvars, var_dims, y_dims,
    x_fit, y_fit, x_probe, y_probe,
    proj, fp_mode, q_scale, q_clip, poly_degree,
    *,
    max_exp=3,
    max_complexity=6,
    max_carriers=20,
    safe_exp_max=500.0,
    safe_tanh_min=0.01,
    max_scored=200,
    refine_enable=False,
    refine_cfg=None,
    refine_state=None,
    early_stop_mse=1e-10,
    verbose=True,
):
    """Peel hyperbolic functions: sech(s*u) and tanh(s*u).

    Numerically peels using torch.cosh/tanh, then builds ASTs via
    exp decomposition: sech(v)=2/(exp(v)+exp(-v)),
    tanh(v)=(exp(v)-exp(-v))/(exp(v)+exp(-v)).
    """
    import numpy as np

    dim0 = tuple(0.0 for _ in var_dims[0])
    dim_arr = np.array([list(d) for d in var_dims], dtype=np.float64)
    y_arr = np.array(list(y_dims), dtype=np.float64)
    y_sq_fit = y_fit.squeeze(-1)
    y_sq_probe = y_probe.squeeze(-1)

    # Find dimensionless monomial carriers.
    carrier_exps = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, np.zeros(len(dim0)), atol=1e-10):
            carrier_exps.append(exps)

    if not carrier_exps:
        return False

    # Pre-enumerate y-dim monomials for numerator search.
    numer_exps = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, y_arr, atol=1e-10):
            numer_exps.append(exps)

    brute_refine_cfg = dict(refine_cfg or {})
    brute_refine_cfg["score_mapping_family_mode"] = str(
        brute_refine_cfg.get(
            "brute_score_mapping_family_mode",
            brute_refine_cfg.get("score_mapping_family_mode", "full"),
        ) or "full"
    )

    best_mse = float("inf")
    best_mse_struct = float("inf")
    n_scored = 0
    n_carriers = len(carrier_exps)

    def _score_tree(tree):
        nonlocal best_mse, best_mse_struct, n_scored
        best_for_gate = (
            float(best_mse_struct)
            if math.isfinite(best_mse_struct)
            else float(max(best_mse, float(early_stop_mse)))
        )
        sc = score_expr(
            tree, x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=brute_refine_cfg,
            refine_best_mse=best_for_gate, refine_state=refine_state,
            return_expr=True,
        )
        if sc is None:
            return
        mse, key, z, mapping, scored_tree = sc
        arch.update(key, mse, scored_tree, z, mapping, raw_mse=mse)
        n_scored += 1
        if mse < best_mse:
            best_mse = mse
        if mapping_is_structural(mapping) and mse < best_mse_struct:
            best_mse_struct = mse

    def _eval_mono(exps, X):
        mono = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
        for k, a in enumerate(exps):
            if a != 0:
                mono = mono * X[:, k].pow(a)
        return mono

    def _sech_ast(v_ast):
        """2 / (exp(v) + exp(-v))"""
        return ("div", ("const", 2.0),
                ("add", ("exp", v_ast), ("exp", ("neg", v_ast))))

    def _tanh_ast(v_ast):
        """(exp(v) - exp(-v)) / (exp(v) + exp(-v))"""
        return ("div",
                ("sub", ("exp", v_ast), ("exp", ("neg", v_ast))),
                ("add", ("exp", v_ast), ("exp", ("neg", v_ast))))

    def _search_numerator(y_adj_fit, y_adj_probe, factor_ast):
        if not numer_exps:
            return
        cheap = []
        for exps in numer_exps:
            mono = _eval_mono(exps, x_fit)
            if not torch.isfinite(mono).all() or float(mono.abs().max()) < 1e-30:
                continue
            c = float((y_adj_fit * mono).sum() / ((mono * mono).sum() + 1e-30))
            mse_c = float(((y_adj_fit - c * mono) ** 2).mean())
            cheap.append((mse_c, exps))
        cheap.sort(key=lambda t: t[0])
        for mse_c, exps in cheap[:5]:
            numer_ast = _build_monomial_ast(exps)
            tree = simplify(("mul", numer_ast, factor_ast))
            _score_tree(tree)

    _TRIAL_SCALES = [1.0, 0.5, 0.2, 1.0 / (2 * math.pi), 0.1,
                     2.0, 5.0, 2 * math.pi, 10.0]

    for c_exps in carrier_exps[:max_carriers]:
        u_fit = _eval_mono(c_exps, x_fit)
        u_probe = _eval_mono(c_exps, x_probe)
        if not (torch.isfinite(u_fit).all() and torch.isfinite(u_probe).all()):
            continue

        for sign, s_exps in [(1, c_exps), (-1, tuple(-a for a in c_exps))]:
            su_fit_raw = u_fit * sign
            su_probe_raw = u_probe * sign

            base_ast = _build_monomial_ast(s_exps)

            for scale in _TRIAL_SCALES:
                v_fit = su_fit_raw * scale
                v_probe = su_probe_raw * scale
                if float(v_fit.abs().max()) > safe_exp_max:
                    continue

                # --- sech peel: y = numer * sech(v) → y_adj = y * cosh(v) ---
                cosh_fit = torch.cosh(v_fit)
                cosh_probe = torch.cosh(v_probe)
                if torch.isfinite(cosh_fit).all() and torch.isfinite(cosh_probe).all():
                    y_adj_fit = y_sq_fit * cosh_fit
                    y_adj_probe = y_sq_probe * cosh_probe
                    if torch.isfinite(y_adj_fit).all() and torch.isfinite(y_adj_probe).all():
                        if abs(scale - 1.0) < 1e-12:
                            v_ast = base_ast
                        else:
                            v_ast = simplify(("mul", ("const", scale), base_ast))
                        _search_numerator(y_adj_fit, y_adj_probe, _sech_ast(v_ast))

                # --- tanh peel: y = numer * tanh(v) → y_adj = y / tanh(v) ---
                tanh_fit = torch.tanh(v_fit)
                tanh_probe = torch.tanh(v_probe)
                if (tanh_fit.abs() > safe_tanh_min).all() and (tanh_probe.abs() > safe_tanh_min).all():
                    y_adj_fit = y_sq_fit / tanh_fit
                    y_adj_probe = y_sq_probe / tanh_probe
                    if torch.isfinite(y_adj_fit).all() and torch.isfinite(y_adj_probe).all():
                        if abs(scale - 1.0) < 1e-12:
                            v_ast = base_ast
                        else:
                            v_ast = simplify(("mul", ("const", scale), base_ast))
                        _search_numerator(y_adj_fit, y_adj_probe, _tanh_ast(v_ast))

                if best_mse_struct < early_stop_mse or n_scored >= max_scored:
                    break
            if best_mse_struct < early_stop_mse or n_scored >= max_scored:
                break
        if best_mse_struct < early_stop_mse or n_scored >= max_scored:
            break

    solved = best_mse_struct < early_stop_mse
    if verbose and n_scored > 0:
        tag = " *** STRUCT-SOLVED ***" if solved else ""
        print(f"[brute]  hyperbolic peel: {n_carriers} carriers, "
              f"scored {n_scored}, best_mse={best_mse:.3e}{tag}")
    return solved


def _gaussian_peel_presearch(
    arch, nvars, var_dims, y_dims,
    x_fit, y_fit, x_probe, y_probe,
    proj, fp_mode, q_scale, q_clip, poly_degree,
    *,
    max_exp=3,
    max_complexity=6,
    max_carriers=20,
    safe_exp_max=500.0,
    refine_enable=False,
    refine_cfg=None,
    refine_state=None,
    early_stop_mse=1e-10,
    carrier_nodes=None,
    verbose=True,
):
    """Peel Gaussian-type expressions: y = numerator * exp(-s * u²).

    Enumerates dimensionless monomials u, tries scaling grid for s,
    peels exp(-s*u²) from y, and searches for the numerator.
    """
    import numpy as np

    dim0 = tuple(0.0 for _ in var_dims[0])
    dim_arr = np.array([list(d) for d in var_dims], dtype=np.float64)
    y_arr = np.array(list(y_dims), dtype=np.float64)
    y_sq_fit = y_fit.squeeze(-1)
    y_sq_probe = y_probe.squeeze(-1)

    dm = var_dims is not None

    # Guard: y must be strictly positive (Gaussian envelope is always > 0).
    if float(y_sq_fit.min()) <= 0 or float(y_sq_probe.min()) <= 0:
        return False

    # Find dimensionless monomials as carrier candidates.
    carrier_exps = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, np.zeros(len(dim0)), atol=1e-10):
            carrier_exps.append(exps)

    extra_carrier_nodes = []
    seen_carrier_keys = set()
    for raw_node in list(carrier_nodes or ()):
        node = simplify(raw_node)
        if not (isinstance(node, tuple) and node):
            continue
        if str(node[0]) == "const":
            continue
        key = str(node_str(node))
        if key in seen_carrier_keys:
            continue
        if dm:
            try:
                node_dim = node_dims(node, var_dims)
            except Exception:
                node_dim = None
            if node_dim is None or not dims_eq(node_dim, dim0):
                continue
        seen_carrier_keys.add(key)
        extra_carrier_nodes.append(node)

    if not carrier_exps and not extra_carrier_nodes:
        return False

    # Pre-enumerate y-dim monomials for numerator search.
    numer_exps = []
    for exps in itertools.product(range(-max_exp, max_exp + 1), repeat=nvars):
        if all(a == 0 for a in exps):
            continue
        if sum(abs(a) for a in exps) > max_complexity:
            continue
        dim_cand = np.array(exps, dtype=np.float64) @ dim_arr
        if np.allclose(dim_cand, y_arr, atol=1e-10):
            numer_exps.append(exps)

    brute_refine_cfg = dict(refine_cfg or {})
    brute_refine_cfg["score_mapping_family_mode"] = str(
        brute_refine_cfg.get(
            "brute_score_mapping_family_mode",
            brute_refine_cfg.get("score_mapping_family_mode", "full"),
        ) or "full"
    )

    best_mse = float("inf")
    best_mse_struct = float("inf")
    n_scored = 0

    def _eval_mono(exps, X):
        mono = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
        for k, a in enumerate(exps):
            if a != 0:
                mono = mono * X[:, k].pow(a)
        return mono

    carrier_rows: list[tuple[tuple, torch.Tensor, torch.Tensor]] = []
    for node in extra_carrier_nodes:
        try:
            u_fit = eval_node(node, x_fit)
            u_probe = eval_node(node, x_probe)
        except Exception:
            continue
        if (not torch.is_tensor(u_fit)) or (not torch.is_tensor(u_probe)):
            continue
        u_fit = u_fit.squeeze(-1)
        u_probe = u_probe.squeeze(-1)
        if (not torch.isfinite(u_fit).all()) or (not torch.isfinite(u_probe).all()):
            continue
        carrier_rows.append((node, u_fit, u_probe))

    for c_exps in carrier_exps[:max_carriers]:
        u_fit = _eval_mono(c_exps, x_fit)
        u_probe = _eval_mono(c_exps, x_probe)
        if not (torch.isfinite(u_fit).all() and torch.isfinite(u_probe).all()):
            continue
        carrier_rows.append((_build_monomial_ast(c_exps), u_fit, u_probe))

    if not carrier_rows:
        return False

    n_carriers = len(carrier_rows)

    def _score_tree(tree):
        nonlocal best_mse, best_mse_struct, n_scored
        best_for_gate = (
            float(best_mse_struct)
            if math.isfinite(best_mse_struct)
            else float(max(best_mse, float(early_stop_mse)))
        )
        sc = score_expr(
            tree, x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=brute_refine_cfg,
            refine_best_mse=best_for_gate, refine_state=refine_state,
            return_expr=True,
        )
        if sc is None:
            return
        mse, key, z, mapping, scored_tree = sc
        arch.update(key, mse, scored_tree, z, mapping, raw_mse=mse)
        n_scored += 1
        if mse < best_mse:
            best_mse = mse
        if mapping_is_structural(mapping) and mse < best_mse_struct:
            best_mse_struct = mse

    # Scaling grid for the Gaussian exponent.
    _TRIAL_SCALES = [0.5, 1.0, 0.1, 0.2, 1.0 / (2 * math.pi), 2.0,
                     5.0, 0.05, 10.0, math.pi]

    for u_ast, u_fit, u_probe in carrier_rows:
        u2_fit = u_fit ** 2
        u2_probe = u_probe ** 2

        for scale in _TRIAL_SCALES:
            # Gaussian factor: exp(-s * u²)
            arg_fit = scale * u2_fit
            arg_probe = scale * u2_probe
            if float(arg_fit.max()) > safe_exp_max or float(arg_probe.max()) > safe_exp_max:
                continue
            gauss_fit = torch.exp(-arg_fit)
            gauss_probe = torch.exp(-arg_probe)
            if not (torch.isfinite(gauss_fit).all() and torch.isfinite(gauss_probe).all()):
                continue
            # Guard: Gaussian factor bounded away from zero.
            if float(gauss_fit.min()) < 1e-30 or float(gauss_probe.min()) < 1e-30:
                continue

            # Peel: y_adj = y / exp(-s*u²) = y * exp(s*u²)
            y_adj_fit = y_sq_fit / gauss_fit
            y_adj_probe = y_sq_probe / gauss_probe
            if not (torch.isfinite(y_adj_fit).all() and torch.isfinite(y_adj_probe).all()):
                continue

            # Build Gaussian AST: exp(-scale * sqr(u))
            if abs(scale - 1.0) < 1e-12:
                gauss_ast = ("exp", ("neg", ("sqr", u_ast)))
            else:
                gauss_ast = ("exp", ("neg", ("mul", ("const", scale), ("sqr", u_ast))))

            # Search monomial numerators on peeled target.
            if numer_exps:
                cheap = []
                for exps in numer_exps:
                    mono = _eval_mono(exps, x_fit)
                    if not torch.isfinite(mono).all() or float(mono.abs().max()) < 1e-30:
                        continue
                    c = float((y_adj_fit * mono).sum() / ((mono * mono).sum() + 1e-30))
                    mse_c = float(((y_adj_fit - c * mono) ** 2).mean())
                    cheap.append((mse_c, exps))
                cheap.sort(key=lambda t: t[0])
                for mse_c, exps in cheap[:5]:
                    numer_ast = _build_monomial_ast(exps)
                    tree = simplify(("mul", numer_ast, gauss_ast))
                    _score_tree(tree)

            if best_mse_struct < early_stop_mse:
                break
        if best_mse_struct < early_stop_mse:
            break

    solved = best_mse_struct < early_stop_mse
    if verbose and n_scored > 0:
        tag = " *** STRUCT-SOLVED ***" if solved else ""
        print(f"[brute]  gaussian peel: {n_carriers} carriers, "
              f"scored {n_scored}, best_mse={best_mse:.3e}{tag}")
    return solved


def _invtrig_peel_presearch(
    arch, nvars, var_dims, y_dims,
    x_fit, y_fit, x_probe, y_probe,
    proj, fp_mode, q_scale, q_clip, poly_degree,
    *,
    brute_depth=3,
    brute_budget=5_000,
    carrier_nodes=None,
    refine_enable=False,
    refine_cfg=None,
    refine_state=None,
    early_stop_mse=1e-10,
    verbose=True,
):
    """Peel inverse-trig outer transforms: try sin(y) and cos(y) as
    targets, enumerate inner expressions, wrap with arcsin/arccos.

    Both y and the peeled target must be dimensionless for inverse-trig
    to be valid.
    """
    dim0 = tuple(0.0 for _ in var_dims[0]) if var_dims else None
    # Guard: inverse trig only applies to dimensionless targets.
    if y_dims is not None and dim0 is not None and not dims_eq(dim_round(tuple(y_dims)), dim0):
        return False

    y_sq_fit = y_fit.squeeze(-1)
    y_sq_probe = y_probe.squeeze(-1)

    brute_refine_cfg = dict(refine_cfg or {})
    brute_refine_cfg["score_mapping_family_mode"] = str(
        brute_refine_cfg.get(
            "brute_score_mapping_family_mode",
            brute_refine_cfg.get("score_mapping_family_mode", "full"),
        ) or "full"
    )

    best_mse = float("inf")
    best_mse_struct = float("inf")
    n_scored = 0

    def _score_tree(tree):
        nonlocal best_mse, best_mse_struct, n_scored
        best_for_gate = (
            float(best_mse_struct)
            if math.isfinite(best_mse_struct)
            else float(max(best_mse, float(early_stop_mse)))
        )
        sc = score_expr(
            tree, x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=brute_refine_cfg,
            refine_best_mse=best_for_gate, refine_state=refine_state,
            return_expr=True,
        )
        if sc is None:
            return
        mse, key, z, mapping, scored_tree = sc
        arch.update(key, mse, scored_tree, z, mapping, raw_mse=mse)
        n_scored += 1
        if mse < best_mse:
            best_mse = mse
        if mapping_is_structural(mapping) and mse < best_mse_struct:
            best_mse_struct = mse

    dm = var_dims is not None
    peels = []
    extra_carrier_nodes = []
    seen_carrier_keys = set()
    for raw_node in list(carrier_nodes or ()):
        node = simplify(raw_node)
        if not (isinstance(node, tuple) and node):
            continue
        if str(node[0]) == "const":
            continue
        key = str(node_str(node))
        if key in seen_carrier_keys:
            continue
        if dm and dim0 is not None:
            try:
                node_dim = node_dims(node, var_dims)
            except Exception:
                node_dim = None
            if node_dim is None or not dims_eq(node_dim, dim0):
                continue
        seen_carrier_keys.add(key)
        extra_carrier_nodes.append(node)

    # sin(y) peel → wrap with arcsin
    y_sin_fit = torch.sin(y_sq_fit)
    y_sin_probe = torch.sin(y_sq_probe)
    if (y_sin_fit.abs() <= 1.0).all() and (y_sin_probe.abs() <= 1.0).all():
        if torch.isfinite(y_sin_fit).all() and torch.isfinite(y_sin_probe).all():
            peels.append(("asin", y_sin_fit, y_sin_probe))

    # cos(y) peel → wrap with arccos
    y_cos_fit = torch.cos(y_sq_fit)
    y_cos_probe = torch.cos(y_sq_probe)
    if (y_cos_fit.abs() <= 1.0).all() and (y_cos_probe.abs() <= 1.0).all():
        if torch.isfinite(y_cos_fit).all() and torch.isfinite(y_cos_probe).all():
            peels.append(("acos", y_cos_fit, y_cos_probe))

    if not peels:
        return False

    for wrap_op, yt_fit, yt_probe in peels:
        yt_fit_2d = yt_fit.unsqueeze(-1)
        yt_probe_2d = yt_probe.unsqueeze(-1)

        for node in extra_carrier_nodes:
            try:
                pf = eval_node(node, x_fit)
                pp = eval_node(node, x_probe)
            except Exception:
                continue
            if (not torch.is_tensor(pf)) or (not torch.is_tensor(pp)):
                continue
            pf = pf.squeeze(-1)
            pp = pp.squeeze(-1)
            if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
                continue
            if ((pf.abs() > 1.0).any()) or ((pp.abs() > 1.0).any()):
                continue
            wrapped = simplify((wrap_op, node))
            _score_tree(wrapped)
            if best_mse_struct < early_stop_mse:
                break
        if best_mse_struct < early_stop_mse:
            break

        # Enumerate small trees against the peeled target and score
        # the wrapped version against the original target.
        if dm:
            gen = _enumerate_dim_incremental(brute_depth, nvars, var_dims, y_dims, brute_budget, verbose=verbose)
        else:
            gen = _enumerate_incremental(brute_depth, nvars, brute_budget, verbose=verbose)

        for depth, new_trees in gen:
            if not new_trees:
                continue
            # Cheap pre-screen: score inner expression against peeled target,
            # keep only promising ones for full wrapped scoring.
            cheap = []
            for tree in new_trees:
                try:
                    pf = eval_node(tree, x_fit)
                    pp = eval_node(tree, x_probe)
                except Exception:
                    continue
                if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
                    continue
                if ((pf.abs() > 1.0).any()) or ((pp.abs() > 1.0).any()):
                    continue
                fb = _fit_best_with_cfg(pf, yt_fit_2d, poly_degree, brute_refine_cfg)
                if fb is None:
                    continue
                cheap.append((fb[0], tree))
            cheap.sort(key=lambda t: t[0])
            # Score top candidates wrapped with arcsin/arccos.
            for mse_inner, inner_tree in cheap[:10]:
                wrapped = simplify((wrap_op, inner_tree))
                _score_tree(wrapped)
                if best_mse_struct < early_stop_mse:
                    break
            if best_mse_struct < early_stop_mse:
                break
        if best_mse_struct < early_stop_mse:
            break

    solved = best_mse_struct < early_stop_mse
    if verbose and n_scored > 0:
        tag = " *** STRUCT-SOLVED ***" if solved else ""
        print(f"[brute]  invtrig peel: {len(peels)} transforms, "
              f"scored {n_scored}, best_mse={best_mse:.3e}{tag}")
    return solved


def _archive_best_mse(arch) -> float:
    best_mse = float("inf")
    for rec in getattr(arch, "d", {}).values():
        try:
            rec_best = float(getattr(rec, "best_mse", float("inf")))
        except Exception:
            rec_best = float("inf")
        if rec_best < best_mse:
            best_mse = rec_best
    return best_mse


def _archive_best_structural_mse(arch) -> float:
    best_mse = float("inf")
    for rec in getattr(arch, "d", {}).values():
        for elite in list(getattr(rec, "elites", []) or ()):
            if not mapping_is_structural(getattr(elite, "mapping", None)):
                continue
            try:
                elite_mse = float(getattr(elite, "mse", float("inf")))
            except Exception:
                elite_mse = float("inf")
            if elite_mse < best_mse:
                best_mse = elite_mse
    return best_mse


def _promote_structural_shadow_archive(
    arch,
    shadow,
    *,
    pre_best_mse: float,
    early_stop_mse: float,
    elite_rel_gate: float = 0.25,
) -> bool:
    """Merge structural elites from a shadow archive into the main archive.

    Returns ``True`` iff a structural solve was actually promoted.
    """
    merged_struct_solve = False
    for key, rec in getattr(shadow, "d", {}).items():
        elites = list(getattr(rec, "elites", []) or ())
        if not elites:
            continue
        struct_elites = [elite for elite in elites if mapping_is_structural(getattr(elite, "mapping", None))]
        if not struct_elites:
            continue
        best_struct = min(
            struct_elites,
            key=lambda elite: float(getattr(elite, "mse", float("inf"))),
        )
        best_struct_mse = float(getattr(best_struct, "mse", float("inf")))
        is_struct_solve = best_struct_mse < float(early_stop_mse)
        is_elite = math.isfinite(float(pre_best_mse)) and best_struct_mse < float(elite_rel_gate) * float(pre_best_mse)
        if not (is_struct_solve or is_elite):
            continue
        arch.update(
            key,
            best_struct_mse,
            best_struct.expr,
            best_struct.z,
            best_struct.mapping,
            raw_mse=float(getattr(best_struct, "raw_mse", best_struct_mse)),
        )
        if is_struct_solve:
            merged_struct_solve = True
    return merged_struct_solve


@torch.no_grad()
def _run_brute_phase(
    arch, nvars,
    x_fit, y_fit, x_probe, y_probe, proj,
    fp_mode, q_scale, q_clip, poly_degree,
    var_dims=None, y_dims=None,
    brute_depth=None,
    early_stop_mse=1e-10,
    max_expressions=50_000,
    refine_enable=False,
    refine_cfg=None,
    refine_state=None,
    label="",
    shuffle_seed=0,
    verbose=True,
    stop_event=None,
    wall_time_deadline=None,
):
    """Incrementally enumerate and score trees depth by depth.

    At each depth only newly-generated trees are scored.  Stops early
    if the best *structural-mapping* MSE drops below *early_stop_mse*
    or if the next depth would exceed the *max_expressions* budget.

    Returns ``True`` if the best structural-mapping MSE is below
    *early_stop_mse* (solved).
    """
    if brute_depth is None:
        brute_depth = _auto_brute_depth(nvars)

    # Fast monomial pre-search: enumerate products of variables raised to
    # integer powers, filtered by dimensional consistency.  This catches
    # pure-monomial targets that tree enumeration misses due to depth caps.
    dm = var_dims is not None
    if dm and y_dims is not None:
        mono_solved = _monomial_presearch(
            arch, nvars, var_dims, y_dims,
            x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
            early_stop_mse=early_stop_mse, verbose=verbose,
        )
        if mono_solved:
            if stop_event is not None:
                stop_event.set()
            return True

        lorentz_solved = _lorentz_peel_presearch(
            arch, nvars, var_dims, y_dims,
            x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
            early_stop_mse=early_stop_mse, verbose=verbose,
        )
        if lorentz_solved:
            if stop_event is not None:
                stop_event.set()
            return True

        # Run Planck and hyperbolic peels into a sidecar archive so that
        # non-matching candidates don't pollute mutation parent selection.
        # Merge into the main archive only on solve or genuine elite.
        pre_best_mse = _archive_best_mse(arch)
        shadow = ResidualBasinArchive()
        planck_solved = _planck_peel_presearch(
            shadow, nvars, var_dims, y_dims,
            x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
            early_stop_mse=early_stop_mse, verbose=verbose,
        )
        # Skip hyperbolic peel if Planck already solved.
        if not planck_solved:
            hyp_solved = _hyperbolic_peel_presearch(
                shadow, nvars, var_dims, y_dims,
                x_fit, y_fit, x_probe, y_probe,
                proj, fp_mode, q_scale, q_clip, poly_degree,
                refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
                early_stop_mse=early_stop_mse, verbose=verbose,
            )
        else:
            hyp_solved = False

        # Promotion: merge shadow → main archive.
        # Scan each residual_basin for the best *structural* elite (not just the
        # overall best, which might be non-structural).
        if _promote_structural_shadow_archive(
            arch,
            shadow,
            pre_best_mse=pre_best_mse,
            early_stop_mse=early_stop_mse,
        ):
            if stop_event is not None:
                stop_event.set()
            return True

        # Run Gaussian peel into its own shadow archive as well. Its
        # candidates are often plentiful on positive targets and can crowd out
        # mutation parents on non-Gaussian problems.
        gaussian_pre_best_mse = _archive_best_mse(arch)
        gaussian_shadow = ResidualBasinArchive()
        _gaussian_peel_presearch(
            gaussian_shadow, nvars, var_dims, y_dims,
            x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
            early_stop_mse=early_stop_mse, verbose=verbose,
        )
        if _promote_structural_shadow_archive(
            arch,
            gaussian_shadow,
            pre_best_mse=gaussian_pre_best_mse,
            early_stop_mse=early_stop_mse,
        ):
            if stop_event is not None:
                stop_event.set()
            return True

        invtrig_solved = _invtrig_peel_presearch(
            arch, nvars, var_dims, y_dims,
            x_fit, y_fit, x_probe, y_probe,
            proj, fp_mode, q_scale, q_clip, poly_degree,
            refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
            early_stop_mse=early_stop_mse, verbose=verbose,
        )
        if invtrig_solved:
            if stop_event is not None:
                stop_event.set()
            return True

    # Enumeration budget is separate from scoring budget: enumerate
    # generously (up to 500k trees), then shuffle + cap to max_expressions
    # for scoring.
    _ENUM_BUDGET = 500_000
    if dm:
        gen = _enumerate_dim_incremental(brute_depth, nvars, var_dims, y_dims, _ENUM_BUDGET, verbose=verbose)
        kind = "dim-ok"
    else:
        gen = _enumerate_incremental(brute_depth, nvars, _ENUM_BUDGET, verbose=verbose)
        kind = "unique"

    best_mse = float('inf')
    best_mse_struct = float('inf')
    n_scored_total = 0
    depth_reached = 0
    brute_refine_cfg = dict(refine_cfg or {})
    brute_refine_cfg["score_mapping_family_mode"] = str(
        brute_refine_cfg.get(
            "brute_score_mapping_family_mode",
            brute_refine_cfg.get("score_mapping_family_mode", "full"),
        ) or "full"
    )

    def _wall_time_exceeded() -> bool:
        if wall_time_deadline is None:
            return False
        try:
            return bool(time.perf_counter() >= float(wall_time_deadline))
        except Exception:
            return False

    for depth, new_trees in gen:
        depth_reached = depth
        if _wall_time_exceeded():
            if stop_event is not None:
                stop_event.set()
            if verbose:
                print(f"[brute]  wall-time limit hit before depth {depth}")
            break
        if stop_event is not None and stop_event.is_set():
            if verbose:
                print(f"[brute]  STOPPED by another thread before depth {depth}")
            break
        if refine_enable and refine_state is not None and refine_cfg is not None:
            lim = int(refine_cfg.get("trials_per_brute_depth", 0))
            refine_state["depth_trials_left"] = None if lim < 0 else lim

        if not new_trees:
            if verbose:
                print(f"[brute]  depth {depth}: 0 new {kind}")
            continue

        # Shuffle and cap at remaining budget so every operation type
        # (add/sub/mul/div) has a fair chance of being sampled.
        budget_left = max_expressions - n_scored_total
        if budget_left <= 0:
            if verbose:
                print(f"[brute]  scoring budget exhausted at depth {depth}")
            break
        if len(new_trees) > budget_left:
            import random as _rng_mod
            _rng_mod.Random(shuffle_seed * 1000 + depth).shuffle(new_trees)
            if verbose:
                print(f"[brute]  depth {depth}: sampling {budget_left} "
                      f"of {len(new_trees)} {kind} (seed={shuffle_seed})")
            new_trees = new_trees[:budget_left]

        # Score new trees at this depth
        n_scored = 0
        stopped = False
        for i, tree in enumerate(new_trees):
            if _wall_time_exceeded():
                if stop_event is not None:
                    stop_event.set()
                stopped = True
                if verbose:
                    print(f"[brute]  wall-time limit hit during depth {depth}")
                break
            # Check if another thread already found a good-enough solution
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if math.isfinite(best_mse_struct):
                best_for_gate = float(best_mse_struct)
            else:
                best_for_gate = float(max(best_mse, float(early_stop_mse)))
            sc = score_expr(tree, x_fit, y_fit, x_probe, y_probe,
                            proj, fp_mode, q_scale, q_clip, poly_degree,
                            refine_enable=refine_enable, refine_cfg=brute_refine_cfg,
                            refine_best_mse=best_for_gate, refine_state=refine_state,
                            return_expr=True)
            if sc is None:
                continue
            mse, key, z, mapping, scored_tree = sc
            arch.update(key, mse, scored_tree, z, mapping, raw_mse=mse)
            n_scored += 1
            if mse < best_mse:
                best_mse = mse
            if mapping_is_structural(mapping) and mse < best_mse_struct:
                best_mse_struct = mse
                if best_mse_struct < early_stop_mse:
                    if stop_event is not None:
                        stop_event.set()  # signal other threads immediately
                    break
            if verbose and (i + 1) % 2000 == 0:
                print(f"[brute]    scored {i+1}/{len(new_trees)}, "
                      f"best_mse={best_mse:.3e}, best_struct_mse={best_mse_struct:.3e}, "
                      f"residual_basins={len(arch.d)}")

        n_scored_total += n_scored
        solved = best_mse_struct < early_stop_mse
        tag = " *** STRUCT-SOLVED ***" if solved else ""
        if stopped:
            if verbose:
                print(f"[brute]  depth {depth}: STOPPED by another thread")
            break
        if verbose:
            print(f"[brute]  depth {depth}: {len(new_trees)} {kind}, "
                  f"scored {n_scored}, best_mse={best_mse:.3e}, "
                  f"best_struct_mse={best_mse_struct:.3e}{tag}")
        if solved:
            break

    solved = best_mse_struct < early_stop_mse
    tag = " *** STRUCT-SOLVED ***" if solved else ""
    if verbose:
        print(f"[brute]  done: depth={depth_reached}, scored={n_scored_total}, "
              f"residual_basins={len(arch.d)}, best_mse={best_mse:.3e}, "
              f"best_struct_mse={best_mse_struct:.3e}{tag}")

    if dm and y_dims is not None and (not solved):
        dim0_target = tuple(0.0 for _ in var_dims[0])
        late_gaussian_carriers = _harvest_pool_from_archive(
            arch,
            random.Random(int(shuffle_seed) + 9173),
            max_nodes=24,
            topk_residual_basins=24,
            elites_per_residual_basin=2,
            subtree_depth_max=3,
            subtree_size_max=10,
            var_dims=var_dims,
            target_dim=dim0_target,
        )
        if late_gaussian_carriers:
            late_gaussian_pre_best_mse = _archive_best_mse(arch)
            late_gaussian_shadow = ResidualBasinArchive()
            if verbose:
                print(f"[brute]  gaussian retry: {len(late_gaussian_carriers)} archive carriers")
            _gaussian_peel_presearch(
                late_gaussian_shadow, nvars, var_dims, y_dims,
                x_fit, y_fit, x_probe, y_probe,
                proj, fp_mode, q_scale, q_clip, poly_degree,
                refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
                early_stop_mse=early_stop_mse,
                carrier_nodes=late_gaussian_carriers,
                verbose=verbose,
            )
            if _promote_structural_shadow_archive(
                arch,
                late_gaussian_shadow,
                pre_best_mse=late_gaussian_pre_best_mse,
                early_stop_mse=early_stop_mse,
            ):
                if stop_event is not None:
                    stop_event.set()
            best_mse = min(best_mse, _archive_best_mse(arch))
            best_mse_struct = min(best_mse_struct, _archive_best_structural_mse(arch))
            solved = best_mse_struct < early_stop_mse
            tag = " *** STRUCT-SOLVED ***" if solved else ""
            if verbose:
                print(f"[brute]  after gaussian retry: best_mse={best_mse:.3e}, "
                      f"best_struct_mse={best_mse_struct:.3e}{tag}")

    if dm and (not solved):
        dim0_target = tuple(0.0 for _ in var_dims[0])
        late_invtrig_carriers = _harvest_pool_from_archive(
            arch,
            random.Random(int(shuffle_seed) + 11237),
            max_nodes=24,
            topk_residual_basins=24,
            elites_per_residual_basin=2,
            subtree_depth_max=6,
            subtree_size_max=20,
            var_dims=var_dims,
            target_dim=dim0_target,
        )
        if late_invtrig_carriers:
            late_invtrig_pre_best_mse = _archive_best_mse(arch)
            late_invtrig_shadow = ResidualBasinArchive()
            if verbose:
                print(f"[brute]  invtrig retry: {len(late_invtrig_carriers)} archive carriers")
            _invtrig_peel_presearch(
                late_invtrig_shadow, nvars, var_dims, y_dims,
                x_fit, y_fit, x_probe, y_probe,
                proj, fp_mode, q_scale, q_clip, poly_degree,
                refine_enable=refine_enable, refine_cfg=refine_cfg, refine_state=refine_state,
                early_stop_mse=early_stop_mse,
                carrier_nodes=late_invtrig_carriers,
                verbose=verbose,
            )
            if _promote_structural_shadow_archive(
                arch,
                late_invtrig_shadow,
                pre_best_mse=late_invtrig_pre_best_mse,
                early_stop_mse=early_stop_mse,
            ):
                if stop_event is not None:
                    stop_event.set()
            best_mse = min(best_mse, _archive_best_mse(arch))
            best_mse_struct = min(best_mse_struct, _archive_best_structural_mse(arch))
            solved = best_mse_struct < early_stop_mse
            tag = " *** STRUCT-SOLVED ***" if solved else ""
            if verbose:
                print(f"[brute]  after invtrig retry: best_mse={best_mse:.3e}, "
                      f"best_struct_mse={best_mse_struct:.3e}{tag}")

    return solved
