# ruff: noqa: F401, F821, F841
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Legacy explorer scoring, refinement, and parameter-search hooks."""

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


# --- residual-guided continuous search (greedy boosting / OMP) ---

def _balanced_add_tree(terms):
    """Build a roughly balanced binary add tree from a list of term nodes."""
    if not terms:
        return None
    nodes = list(terms)
    while len(nodes) > 1:
        nxt = []
        i = 0
        n = len(nodes)
        while i < n:
            if i + 1 < n:
                nxt.append(("add", nodes[i], nodes[i + 1]))
                i += 2
            else:
                nxt.append(nodes[i])
                i += 1
        nodes = nxt
    return nodes[0]


def _strip_scalar_prefix(node):
    """Strip leading neg / mul-by-const wrappers for pool dedup."""
    out = node
    for _ in range(4):
        if not (isinstance(out, tuple) and out):
            break
        op = out[0]
        if op == "neg":
            out = out[1]
            continue
        if op == "mul":
            a, b = out[1], out[2]
            if isinstance(a, tuple) and a and a[0] == "const":
                out = b
                continue
            if isinstance(b, tuple) and b and b[0] == "const":
                out = a
                continue
        break
    return out


def _extract_scalar_core(node):
    """Split a node into (scalar, core) for simple linear-root canonicalization."""
    coeff = 1.0
    cur = node
    for _ in range(8):
        if not (isinstance(cur, tuple) and cur):
            break
        op = cur[0]
        if op == "neg":
            coeff = -coeff
            cur = cur[1]
            continue
        if op == "mul":
            a, b = cur[1], cur[2]
            if isinstance(a, tuple) and a and a[0] == "const":
                try:
                    coeff *= float(a[1])
                except Exception:
                    pass
                cur = b
                continue
            if isinstance(b, tuple) and b and b[0] == "const":
                try:
                    coeff *= float(b[1])
                except Exception:
                    pass
                cur = a
                continue
        break
    return float(coeff), cur


def _collect_linear_terms(node, sign=1.0, out=None):
    """Collect signed additive terms from a root expression."""
    if out is None:
        out = []
    if not (isinstance(node, tuple) and node):
        out.append((float(sign), node))
        return out
    op = node[0]
    if op == "add":
        _collect_linear_terms(node[1], sign, out)
        _collect_linear_terms(node[2], sign, out)
        return out
    if op == "sub":
        _collect_linear_terms(node[1], sign, out)
        _collect_linear_terms(node[2], -sign, out)
        return out
    if op == "neg":
        _collect_linear_terms(node[1], -sign, out)
        return out
    out.append((float(sign), node))
    return out


def _mapping_equiv_root(node, *, assume_simplified=False):
    """Canonicalize only top-level mapping-equivalent scalar/sign variants."""
    t = node if assume_simplified else simplify(node)

    terms = _collect_linear_terms(t, 1.0, [])
    if len(terms) >= 2:
        parsed = []
        for sgn, term in terms:
            c, core = _extract_scalar_core(term)
            parsed.append((float(sgn) * float(c), core))
        if parsed:
            core0 = parsed[0][1]
            key0 = node_str(core0)
            if all(node_str(core) == key0 for _, core in parsed):
                total = sum(float(c) for c, _ in parsed)
                if abs(total) <= 1.0e-14:
                    t = ("const", 0.0)
                else:
                    t = core0

    t = _strip_scalar_prefix(t)
    if isinstance(t, tuple) and t and t[0] == "sub" and node_str(t[1]) > node_str(t[2]):
        t = ("sub", t[2], t[1])
    return t


def _compile_linear_combo(term_nodes, coeffs, Phi_fit, prune_rel, max_depth):
    """Compile Σ c_i * term_i into a single AST, pruning tiny contributors and enforcing max_depth."""
    if term_nodes is None or coeffs is None or Phi_fit is None:
        return None
    try:
        K = min(int(len(term_nodes)), int(coeffs.shape[0]), int(Phi_fit.shape[1]))
    except Exception:
        return None
    if K <= 0:
        return None

    rel = max(0.0, float(prune_rel))

    contrib = []
    for j in range(K):
        try:
            c = float(coeffs[j])
        except Exception:
            c = float("nan")
        if (not math.isfinite(c)) or abs(c) < 1.0e-14:
            contrib.append(0.0)
            continue
        col = Phi_fit[:, j]
        rms = float(torch.sqrt((col * col).mean()))
        if not math.isfinite(rms):
            rms = 0.0
        contrib.append(abs(c) * rms)

    max_contrib = max(contrib) if contrib else 0.0

    keep = []
    keep_contrib = []
    for j in range(K):
        if contrib[j] <= 0.0:
            continue
        if max_contrib > 0.0 and contrib[j] < rel * max_contrib:
            continue
        keep.append(j)
        keep_contrib.append(contrib[j])

    if not keep:
        return None

    # Pre-build simplified scaled terms.
    scaled_terms = []
    scaled_contrib = []
    for jj, j in enumerate(keep):
        try:
            c = float(coeffs[j])
        except Exception:
            continue
        if (not math.isfinite(c)) or abs(c) < 1.0e-14:
            continue
        term = term_nodes[j]
        if abs(c - 1.0) < 1.0e-12:
            t = term
        elif abs(c + 1.0) < 1.0e-12:
            t = ("neg", term)
        else:
            t = ("mul", ("const", float(c)), term)
        t = simplify(t)
        scaled_terms.append(t)
        scaled_contrib.append(keep_contrib[jj])

    if not scaled_terms:
        return None

    # If depth is too large, drop the weakest contributors until feasible.
    active = list(range(len(scaled_terms)))
    while True:
        expr = _balanced_add_tree([scaled_terms[i] for i in active])
        if expr is None:
            return None
        expr = simplify(expr)
        if node_depth(expr) <= max_depth:
            return expr
        if len(active) <= 1:
            return None
        # Drop smallest-contribution term.
        drop_i = min(active, key=lambda ii: scaled_contrib[ii])
        active.remove(drop_i)


def _harvest_pool_from_archive(
    arch,
    rng,
    *,
    max_nodes=256,
    topk_residual_basins=50,
    elites_per_residual_basin=2,
    subtree_depth_max=3,
    subtree_size_max=12,
    base_seen=None,
    var_dims=None,
    target_dim=None,
):
    """Harvest simple subtrees from the archive to expand the residual pool."""
    if arch is None or not getattr(arch, "d", None):
        return []
    try:
        max_nodes = max(0, int(max_nodes))
    except Exception:
        max_nodes = 0
    if max_nodes <= 0:
        return []

    seen = set(base_seen) if base_seen else set()
    out = []

    try:
        recs = sorted(list(arch.d.values()), key=lambda r: float(getattr(r, "best_mse", 1e100)))
    except Exception:
        recs = list(getattr(arch, "d", {}).values())

    try:
        recs = recs[: max(1, int(topk_residual_basins))]
    except Exception:
        pass

    for r in recs:
        exprs = []
        try:
            els = list(getattr(r, "elites", []) or [])
            if els:
                els = sorted(
                    els,
                    key=lambda e: (
                        float(getattr(e, "mse", 1e100)),
                        float(getattr(e, "size", 1.0e100)),
                    ),
                )
                k = max(0, int(elites_per_residual_basin))
                if k > 0:
                    exprs.extend([el.expr for el in els[:k]])
        except Exception:
            pass
        try:
            exprs.append(r.best_expr)
        except Exception:
            pass

        for expr in exprs:
            if expr is None:
                continue
            try:
                paths = collect_paths(expr)
            except Exception:
                continue
            try:
                rng.shuffle(paths)
            except Exception:
                pass

            for p in paths:
                try:
                    sub = get_at(expr, p)
                except Exception:
                    continue
                if node_depth(sub) > subtree_depth_max:
                    continue
                if node_size(sub) > subtree_size_max:
                    continue
                sub = simplify(sub)
                sub = _strip_scalar_prefix(sub)
                if not (isinstance(sub, tuple) and sub):
                    continue
                if sub[0] == "const":
                    continue
                if var_dims is not None:
                    d = node_dims(sub, var_dims)
                    if d is None:
                        continue
                    if target_dim is not None and not dims_eq(d, target_dim):
                        continue
                key = node_str(sub)
                if key in seen:
                    continue
                seen.add(key)
                out.append(sub)
                if len(out) >= max_nodes:
                    return out

    return out


@torch.no_grad()
def apply_boost_action(
    parent_node,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    pool_nodes,
    pool_phi_fit,
    pool_norms_fit,
    pool_phi_probe,
    pool_norms_probe,
    pool_dims,
    rng,
    max_depth,
    nvars,
    poly_degree,
    *,
    var_dims=None,
    y_dims=None,
    max_terms=6,
    topk_try=15,
    min_rel_improve=1.0e-3,
    selection_split="fit",
    ridge=1.0e-8,
    include_parent=True,
    from_scratch_prob=0.25,
    prune_rel=1.0e-10,
    complexity_penalty=0.0,
):
    """Greedy residual-guided additive construction over the symbolic pool.

    This is a lightweight "functional gradient boosting / matching pursuit" step:
    iteratively pick pool terms most correlated with the current residual, refit
    coefficients, and return a compiled additive skeleton.
    """
    dm = var_dims is not None
    if dm:
        target_dim = y_dims if y_dims is not None else node_dims(parent_node, var_dims)
    else:
        target_dim = None

    # Pick which residual split to use for term selection.
    use_probe = str(selection_split).lower().startswith("p")
    phi_sel = pool_phi_probe if use_probe else pool_phi_fit
    norms_sel = pool_norms_probe if use_probe else pool_norms_fit
    y_sel = y_probe if use_probe else y_fit

    # Decide whether to include the parent term as a basis function.
    use_parent = bool(include_parent) and (rng.random() >= float(from_scratch_prob))
    parent_fit = None
    parent_probe = None
    if use_parent:
        try:
            parent_fit = eval_node(parent_node, x_fit).squeeze(-1)
            parent_probe = eval_node(parent_node, x_probe).squeeze(-1)
            if (not torch.isfinite(parent_fit).all()) or (not torch.isfinite(parent_probe).all()):
                use_parent = False
        except Exception:
            use_parent = False

    # Pre-mask pool terms by dimensions (when enabled).
    if dm and target_dim is not None:
        valid = torch.tensor([
            pool_dims[i] is not None and dims_eq(pool_dims[i], target_dim)
            for i in range(len(pool_nodes))
        ], device=norms_sel.device)
    else:
        valid = torch.ones((len(pool_nodes),), dtype=torch.bool, device=norms_sel.device)
    valid = valid & torch.isfinite(norms_sel) & (norms_sel > 1.0e-12)

    # ------------------------------------------------------------------
    # Helper: solve coeffs + fit output mapping + compile expression.
    # ------------------------------------------------------------------
    def _evaluate(sel_idx):
        cols_fit = []
        cols_probe = []
        term_nodes = []
        if use_parent:
            cols_fit.append(parent_fit)
            cols_probe.append(parent_probe)
            term_nodes.append(parent_node)
        for ii in sel_idx:
            cols_fit.append(pool_phi_fit[:, ii])
            cols_probe.append(pool_phi_probe[:, ii])
            term_nodes.append(pool_nodes[ii])

        if not cols_fit:
            return None

        Phi_fit = torch.stack(cols_fit, dim=1)
        sol = _solve_linear_coeffs(Phi_fit, y_fit, ridge)
        if sol is None or (not torch.isfinite(sol).all()):
            return None

        pred_fit = Phi_fit @ sol
        fb = _fit_best_with_cfg(pred_fit, y_fit, poly_degree, cfg)
        if fb is None:
            return None
        _, mapping = fb

        Phi_probe = torch.stack(cols_probe, dim=1)
        pred_probe = Phi_probe @ sol
        y_hat_probe = eval_mapping(pred_probe, mapping)
        if not torch.isfinite(y_hat_probe).all():
            return None
        mse = mean_squared_error_same_shape(y_probe, y_hat_probe)
        if not math.isfinite(mse):
            return None

        # residual for next greedy selection step
        if use_probe:
            y_hat_sel = eval_mapping(pred_probe, mapping)
            resid = (y_probe - y_hat_sel).squeeze(-1)
        else:
            y_hat_sel = eval_mapping(pred_fit, mapping)
            resid = (y_fit - y_hat_sel).squeeze(-1)

        expr = _compile_linear_combo(
            term_nodes,
            sol.squeeze(-1),
            Phi_fit,
            prune_rel,
            max_depth,
        )
        if expr is None:
            return None

        if dm and y_dims is not None:
            d = node_dims(expr, var_dims)
            if d is None or (not dims_eq(d, y_dims)):
                return None

        mse_eff = float(
            mse
            + float(complexity_penalty)
            * float(node_size(expr) + mapping_cost(mapping))
        )
        return mse, mse_eff, expr, resid

    # Baseline residual is just the target (model=0).
    selected = []
    current = None
    current_mse = float("inf")
    current_score = float("inf")
    current_expr = None
    resid = y_sel.squeeze(-1)

    # Optional baseline: parent-only (no pool terms).
    if use_parent:
        base = _evaluate([])
        if base is not None:
            current_mse, current_score, current_expr, resid = base

    # Greedy forward selection.
    max_terms = max(1, int(max_terms))
    topk_try = max(1, int(topk_try))
    min_rel_improve = max(0.0, float(min_rel_improve))

    for _ in range(max_terms):
        if resid is None or resid.numel() <= 0:
            break

        dots = resid @ phi_sel
        scores = dots * dots / (norms_sel + 1.0e-12)
        if valid is not None:
            scores = scores.clone()
            scores[~valid] = -float("inf")
        if selected:
            scores = scores.clone()
            scores[torch.tensor(selected, dtype=torch.long, device=scores.device)] = -float("inf")

        # Try a small shortlist of best-correlated terms.
        k = min(topk_try, int(scores.numel()))
        if k <= 0:
            break
        top_idx = scores.topk(k).indices.tolist()

        best = None
        best_mse = float("inf")
        best_score = float("inf")
        best_idx = None

        for idx in top_idx:
            if idx is None:
                continue
            if not bool(valid[idx]):
                continue
            cand = _evaluate(selected + [int(idx)])
            if cand is None:
                continue
            mse_c, score_c, expr_c, resid_c = cand
            if (score_c < best_score) or (score_c == best_score and mse_c < best_mse):
                best_mse = mse_c
                best_score = score_c
                best = (expr_c, resid_c)
                best_idx = int(idx)

        if best is None or best_idx is None:
            break

        # Require meaningful improvement over the current model.
        if math.isfinite(current_mse):
            rel = (float(current_mse) - float(best_mse)) / max(1.0e-12, abs(float(current_mse)))
            if (best_score >= current_score) or (rel < min_rel_improve):
                break

        selected.append(best_idx)
        current_mse = best_mse
        current_score = best_score
        current_expr, resid = best

    # Avoid no-op proposals (parent-only scaling); require at least one pool term.
    if not selected:
        return None
    return current_expr

# --- fitting ---

def fingerprint(r,proj,mode,scale,clip,eps=1e-12):
    r=r-r.mean(); r=r/(r.std()+eps)
    z=(r@proj)/math.sqrt(r.numel())
    if mode=="bits":
        bits=(z>0).to(torch.int8).tolist()
        k=0
        for i,b in enumerate(bits): k |= (int(b)&1)<<i
        return k,z
    q=torch.clamp((z*scale).round(), -clip, clip).to(torch.int16)
    return tuple(int(v) for v in q.tolist()), z

def _negate_smart(node):
    """Return a cheap equivalent tuple-AST for (-node), preferring forms that avoid an extra 'neg' node."""
    try:
        op = node[0]
    except Exception:
        return ("neg", node)
    if op == "const":
        return ("const", -float(node[1]))
    if op == "neg":
        return node[1]
    if op == "sub":
        # -(a-b) == (b-a) with no extra node
        return ("sub", node[2], node[1])
    if op == "mul":
        a, b = node[1], node[2]
        if isinstance(a, tuple) and a and a[0] == "const":
            return ("mul", ("const", -float(a[1])), b)
        if isinstance(b, tuple) and b and b[0] == "const":
            return ("mul", a, ("const", -float(b[1])))
    if op == "div":
        a, b = node[1], node[2]
        if isinstance(a, tuple) and a and a[0] == "const":
            return ("div", ("const", -float(a[1])), b)
    return ("neg", node)


def _pick_best_equiv_score(cands, y_var=None):
    """Pick the best candidate among equivalent-score variants.

    Preference order:
      1) Any *structural* mapping within a tolerant MSE window.
      2) Fewer mapping parameters.
      3) Simpler expression (physics prior cost, then size).
      4) Lowest MSE.
    """
    cands = [c for c in (cands or []) if c is not None]
    if not cands:
        return None
    best_mse = min(float(c[0]) for c in cands)
    if y_var is None:
        y_var = 1e-30
    y_var = max(float(y_var), 1e-30)
    mse_tol = max(best_mse * 3.0, y_var * 1e-8)
    close = [c for c in cands if float(c[0]) <= mse_tol]
    pool = close if close else cands

    def _rank(c):
        mse, _, _, mapping, expr = c
        return (
            0 if mapping_is_structural(mapping) else 1,
            _mapping_nparams(mapping),
            node_cost_physics_prior(expr),
            node_size(expr),
            float(mse),
        )

    pool.sort(key=_rank)
    return pool[0]


@torch.no_grad()
def _score_expr_base(node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg=None):
    cfg = cfg or {}
    # Ensure a stable baseline form (also helps negate_smart produce nicer trees).
    node = simplify(node)

    p_fit = eval_node(node, x_fit)
    if not torch.isfinite(p_fit).all():
        return None
    if float(p_fit.std()) < 1e-12:
        return None

    p_probe = eval_node(node, x_probe)
    if not torch.isfinite(p_probe).all():
        return None

    y_var = max(float((y_fit ** 2).mean()), 1e-30)

    # Scoring augmentation: optional linear head on the residual that can soak up
    # simple unit-consistent additive terms (e.g. raw variables).
    head_enable = bool(cfg.get("score_head_enable", False))
    head_vars_enable = bool(cfg.get("score_head_vars_enable", True))
    head_omp_enable = bool(cfg.get("score_head_omp_enable", False))
    head_omp_max_terms = int(cfg.get("score_head_omp_max_terms", 0) or 0)
    head_omp_topk_try = int(cfg.get("score_head_omp_topk_try", 15) or 15)
    head_min_rel_improve = float(cfg.get("score_head_min_rel_improve", 0.0) or 0.0)

    head_ridge = cfg.get("score_head_ridge", None)
    if head_ridge is None:
        head_ridge = cfg.get("refine_linear_ridge", 1.0e-8)
    head_ridge = float(head_ridge) if head_ridge is not None else 0.0
    head_direct_combo_enable = bool(cfg.get("score_head_direct_combo_enable", True))
    head_direct_combo_prune_rel = float(cfg.get("score_head_direct_combo_prune_rel", 1.0e-6) or 0.0)
    head_direct_combo_tol = float(cfg.get("score_head_direct_combo_tol", 1.0e-6) or 0.0)
    max_depth = int(cfg.get("max_depth", 12) or 12)

    head_var_terms = []
    if head_enable and head_vars_enable:
        head_var_terms = list(cfg.get("score_head_var_terms", []) or [])

    # Optional OMP selection from the pool (requires the pool tensors in cfg).
    pool_nodes = cfg.get("score_head_pool_nodes", None)
    pool_phi_fit = cfg.get("score_head_pool_phi_fit", None)
    pool_phi_probe = cfg.get("score_head_pool_phi_probe", None)
    pool_norms_fit = cfg.get("score_head_pool_norms_fit", None)
    pool_valid_mask = cfg.get("score_head_pool_valid_mask", None)
    pool_node_to_idx = cfg.get("score_head_pool_node_to_idx", None)

    def _eval_term(term, X):
        # Fast-path raw variables.
        if isinstance(term, tuple) and len(term) == 2 and term[0] == "var":
            j = int(term[1])
            if j < 0 or j >= int(X.shape[1]):
                return None
            return X[:, j]
        v = eval_node(term, X)
        if v is None:
            return None
        if v.dim() == 2 and v.shape[1] == 1:
            v = v[:, 0]
        return v

    def _fit_head(resid_fit0, resid_probe0):
        """Fit a (bias + linear terms) head to resid_fit0, evaluate on resid_probe0.

        Returns:
            (mse_probe, r_probe, head_dict, pred_fit, pred_probe)
        where pred_* are the head contributions (shape Nx1), and r_probe is the final
        probe residual after subtracting the head (shape N,).
        """
        # resid_*0 are (N,1) tensors
        terms = []
        cols_fit = []
        cols_probe = []

        # Baseline terms (B1): unit-matching raw variables (pre-filtered upstream).
        for t in head_var_terms:
            v_fit = _eval_term(t, x_fit)
            v_probe = _eval_term(t, x_probe)
            if v_fit is None or v_probe is None:
                continue
            if (not torch.isfinite(v_fit).all()) or (not torch.isfinite(v_probe).all()):
                continue
            if float(v_fit.std()) < 1e-12:
                continue
            terms.append(t)
            cols_fit.append(v_fit)
            cols_probe.append(v_probe)

        # OMP extra terms from pool (B2): cheap greedy selection on residual.
        selected_pool = []
        if head_enable and head_omp_enable and head_omp_max_terms > 0:
            if (
                isinstance(pool_nodes, (list, tuple))
                and torch.is_tensor(pool_phi_fit)
                and torch.is_tensor(pool_phi_probe)
                and torch.is_tensor(pool_norms_fit)
                and torch.is_tensor(pool_valid_mask)
                and int(pool_phi_fit.shape[0]) == int(resid_fit0.shape[0])
                and int(pool_phi_probe.shape[0]) == int(resid_probe0.shape[0])
                and int(pool_phi_fit.shape[1]) == int(pool_norms_fit.shape[0])
                and int(pool_phi_fit.shape[1]) == int(pool_valid_mask.shape[0])
            ):
                # Exclude any pool terms already present in the baseline list.
                exclude = set()
                if isinstance(pool_node_to_idx, dict):
                    for t in terms:
                        try:
                            idx = pool_node_to_idx.get(t, None)
                        except Exception:
                            idx = None
                        if idx is not None:
                            exclude.add(int(idx))

                resid_v = resid_fit0.squeeze(-1)
                # Greedy select up to K terms.
                for _ in range(int(head_omp_max_terms)):
                    mask = pool_valid_mask.clone()
                    if exclude:
                        ex = torch.tensor(sorted(exclude), dtype=torch.long, device=mask.device)
                        mask[ex] = False
                    if selected_pool:
                        sel = torch.tensor(selected_pool, dtype=torch.long, device=mask.device)
                        mask[sel] = False
                    n_valid = int(mask.sum().item())
                    if n_valid <= 0:
                        break

                    # Correlation scores (normalized by column norm).
                    dots = torch.mv(pool_phi_fit.t(), resid_v)
                    denom = torch.sqrt(torch.clamp(pool_norms_fit, min=1e-30))
                    score = dots.abs() / denom
                    score = score.masked_fill(~mask, float("-inf"))

                    k = min(int(head_omp_topk_try), n_valid)
                    topk = torch.topk(score, k=k, largest=True).indices.tolist()

                    best_cand = None
                    best_mse = float("inf")
                    best_pred_fit = None

                    # Base Phi (ones + baseline columns).
                    base_cols_fit = cols_fit
                    base_cols_probe = cols_probe

                    for cand in topk:
                        cand = int(cand)
                        # Build Phi for this candidate set: ones | baseline | selected_pool | cand
                        phi_fit_parts = []
                        phi_probe_parts = []

                        if base_cols_fit:
                            phi_fit_parts.append(torch.stack(base_cols_fit, dim=1))
                            phi_probe_parts.append(torch.stack(base_cols_probe, dim=1))

                        if selected_pool:
                            sel = torch.tensor(selected_pool, dtype=torch.long, device=pool_phi_fit.device)
                            phi_fit_parts.append(pool_phi_fit[:, sel])
                            phi_probe_parts.append(pool_phi_probe[:, sel])

                        # Candidate column
                        phi_fit_parts.append(pool_phi_fit[:, cand:cand + 1])
                        phi_probe_parts.append(pool_phi_probe[:, cand:cand + 1])

                        phi_fit = torch.cat(phi_fit_parts, dim=1) if phi_fit_parts else pool_phi_fit[:, cand:cand + 1]
                        phi_probe = torch.cat(phi_probe_parts, dim=1) if phi_probe_parts else pool_phi_probe[:, cand:cand + 1]

                        # Add bias column.
                        ones_fit = torch.ones((phi_fit.shape[0], 1), dtype=phi_fit.dtype, device=phi_fit.device)
                        ones_probe = torch.ones((phi_probe.shape[0], 1), dtype=phi_probe.dtype, device=phi_probe.device)
                        Phi_fit = torch.cat([ones_fit, phi_fit], dim=1)
                        Phi_probe = torch.cat([ones_probe, phi_probe], dim=1)

                        sol = _solve_linear_coeffs(Phi_fit, resid_fit0, ridge=head_ridge)
                        if sol is None:
                            continue
                        pred_fit = Phi_fit @ sol
                        r_fit = resid_fit0 - pred_fit
                        mse_fit = float((r_fit.squeeze(-1) ** 2).mean())
                        if math.isfinite(mse_fit) and mse_fit < best_mse:
                            best_mse = mse_fit
                            best_cand = cand
                            best_pred_fit = pred_fit.detach()

                    if best_cand is None:
                        break
                    selected_pool.append(int(best_cand))
                    exclude.add(int(best_cand))
                    # Update residual for next round (on fit split).
                    if best_pred_fit is not None:
                        resid_v = (resid_fit0 - best_pred_fit).squeeze(-1)

        # Materialize the final term list.
        term_nodes = list(terms)
        if selected_pool and isinstance(pool_nodes, (list, tuple)):
            term_nodes.extend([pool_nodes[int(i)] for i in selected_pool])

        if not term_nodes:
            # Nothing to fit (we intentionally don't fit a bias-only head).
            return None

        # Materialize the final design matrices.
        phi_fit_cols = []
        phi_probe_cols = []
        # Baseline columns from earlier (same order as `terms`).
        if cols_fit:
            phi_fit_cols.append(torch.stack(cols_fit, dim=1))
            phi_probe_cols.append(torch.stack(cols_probe, dim=1))
        # Pool-selected columns (same order as selected_pool).
        if selected_pool:
            sel = torch.tensor(selected_pool, dtype=torch.long, device=pool_phi_fit.device)
            phi_fit_cols.append(pool_phi_fit[:, sel])
            phi_probe_cols.append(pool_phi_probe[:, sel])

        phi_fit = torch.cat(phi_fit_cols, dim=1) if phi_fit_cols else None
        phi_probe = torch.cat(phi_probe_cols, dim=1) if phi_probe_cols else None
        if phi_fit is None or phi_probe is None:
            return None

        ones_fit = torch.ones((phi_fit.shape[0], 1), dtype=phi_fit.dtype, device=phi_fit.device)
        ones_probe = torch.ones((phi_probe.shape[0], 1), dtype=phi_probe.dtype, device=phi_probe.device)
        Phi_fit = torch.cat([ones_fit, phi_fit], dim=1)
        Phi_probe = torch.cat([ones_probe, phi_probe], dim=1)

        sol = _solve_linear_coeffs(Phi_fit, resid_fit0, ridge=head_ridge)
        if sol is None:
            return None

        pred_fit = Phi_fit @ sol
        pred_probe = Phi_probe @ sol
        r_probe = (resid_probe0 - pred_probe).squeeze(-1)
        mse_probe = float((r_probe * r_probe).mean())
        if not math.isfinite(mse_probe):
            return None

        coeffs = sol.squeeze(-1).detach().cpu().tolist()
        coeffs = [float(v) for v in coeffs]
        head = {
            "terms": term_nodes,
            "coeffs": coeffs,  # [bias, a_0, ..., a_k]
            "ridge": float(head_ridge),
        }
        if selected_pool:
            head["pool_selected"] = [int(i) for i in selected_pool]

        return (
            mse_probe,
            r_probe,
            head,
            pred_fit.detach(),
            pred_probe.detach(),
            phi_fit.detach(),
            phi_probe.detach(),
        )

    def _try_direct_combo(expr_base, pred_fit_base, pred_probe_base, term_nodes, phi_fit_terms, phi_probe_terms, mse_ref):
        if not bool(head_direct_combo_enable):
            return None
        term_nodes = list(term_nodes or [])
        if not term_nodes:
            return None
        if (not torch.is_tensor(phi_fit_terms)) or (not torch.is_tensor(phi_probe_terms)):
            return None
        if int(phi_fit_terms.shape[1]) != int(len(term_nodes)):
            return None
        if int(phi_probe_terms.shape[1]) != int(len(term_nodes)):
            return None

        base_col_fit = pred_fit_base.squeeze(-1).unsqueeze(-1)
        base_col_probe = pred_probe_base.squeeze(-1).unsqueeze(-1)
        ones_fit = torch.ones((int(y_fit.shape[0]), 1), dtype=y_fit.dtype, device=y_fit.device)
        ones_probe = torch.ones((int(y_probe.shape[0]), 1), dtype=y_probe.dtype, device=y_probe.device)
        Phi_fit = torch.cat([base_col_fit, phi_fit_terms, ones_fit], dim=1)
        Phi_probe = torch.cat([base_col_probe, phi_probe_terms, ones_probe], dim=1)
        sol = _solve_linear_coeffs(Phi_fit, y_fit, ridge=head_ridge)
        if sol is None:
            return None

        coeff_vec = sol.squeeze(-1)
        compiled = _compile_linear_combo(
            [expr_base, *term_nodes, ("const", 1.0)],
            coeff_vec,
            Phi_fit,
            head_direct_combo_prune_rel,
            max_depth,
        )
        if compiled is None:
            return None

        pred_fit_combo = eval_node(compiled, x_fit)
        pred_probe_combo = eval_node(compiled, x_probe)
        if (pred_fit_combo is None) or (pred_probe_combo is None):
            return None
        if (not torch.isfinite(pred_fit_combo).all()) or (not torch.isfinite(pred_probe_combo).all()):
            return None

        r_probe_combo = (y_probe - pred_probe_combo).squeeze(-1)
        mse_combo = float((r_probe_combo * r_probe_combo).mean())
        if not math.isfinite(mse_combo):
            return None
        tol_abs = max(1.0e-12, float(y_var) * 1.0e-12)
        tol_rel = max(0.0, float(head_direct_combo_tol))
        if mse_combo > float(mse_ref) * (1.0 + tol_rel) + tol_abs:
            return None

        key_combo, z_combo = fingerprint(r_probe_combo, proj, fp_mode, q_scale, q_clip)
        mapping_combo = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        mapping_combo["_basis_transition"] = make_additive_basis_transition(
            core_expr=expr_base,
            term_nodes=term_nodes,
            coeffs=coeff_vec.detach().cpu().tolist(),
            compiled_expr=compiled,
            ridge=float(head_ridge),
            prune_rel=float(head_direct_combo_prune_rel),
        )
        return (mse_combo, key_combo, z_combo, mapping_combo, compiled)

    cands = []

    def _try(pred_fit, pred_probe, expr):
        fb = _fit_best_with_cfg(pred_fit, y_fit, poly_degree, cfg)
        if fb is None:
            return None
        _, mapping0 = fb

        # Base (univariate) mapping prediction.
        y_hat_fit0 = eval_mapping(pred_fit, mapping0)
        y_hat_probe0 = eval_mapping(pred_probe, mapping0)
        r0 = (y_probe - y_hat_probe0).squeeze(-1)
        mse0 = float((r0 * r0).mean())
        if not math.isfinite(mse0):
            return None

        mse = mse0
        r = r0
        mapping = mapping0

        # Optional linear head on the residual.
        refit_trigger_gain = max(head_min_rel_improve, 0.05)
        if head_enable and (head_var_terms or (head_omp_enable and head_omp_max_terms > 0)):
            resid_fit0 = (y_fit - y_hat_fit0)
            resid_probe0 = (y_probe - y_hat_probe0)

            head_fit1 = _fit_head(resid_fit0, resid_probe0)
            final_head = None
            final_phi_fit = None
            final_phi_probe = None
            if head_fit1 is not None:
                mse_h1, r_h1, head1, head_pred_fit1, head_pred_probe1, phi_fit_head1, phi_probe_head1 = head_fit1
                gain1 = (mse0 - mse_h1) / max(mse0, 1e-30)
                if math.isfinite(mse_h1) and (mse_h1 < mse0) and (gain1 >= head_min_rel_improve):
                    mse = mse_h1
                    r = r_h1
                    mapping = dict(mapping0)
                    mapping["_lin_head"] = head1
                    final_head = head1
                    final_phi_fit = phi_fit_head1
                    final_phi_probe = phi_probe_head1

                    # Alternating refinement: refit mapping <-> head until convergence.
                    # One pass is insufficient when f correlates with head variables.
                    _alt_gain = gain1
                    for _alt in range(4):
                        if _alt_gain < refit_trigger_gain:
                            break
                        y_fit_adj = y_fit - head_pred_fit1
                        fb2 = _fit_best_with_cfg(pred_fit, y_fit_adj, poly_degree, cfg)
                        if fb2 is None:
                            break
                        _, mapping1 = fb2
                        y_hat_fit1 = eval_mapping(pred_fit, mapping1)
                        y_hat_probe1 = eval_mapping(pred_probe, mapping1)
                        resid_fit1 = (y_fit - y_hat_fit1)
                        resid_probe1 = (y_probe - y_hat_probe1)

                        head_fit2 = _fit_head(resid_fit1, resid_probe1)
                        if head_fit2 is None:
                            break
                        mse_h2, r_h2, head2, head_pred_fit1, head_pred_probe1, phi_fit_head2, phi_probe_head2 = head_fit2
                        _alt_gain = (mse - mse_h2) / max(mse, 1e-30)
                        if not (math.isfinite(mse_h2) and mse_h2 < mse):
                            break
                        mse = mse_h2
                        r = r_h2
                        mapping = dict(mapping1)
                        mapping["_lin_head"] = head2
                        final_head = head2
                        final_phi_fit = phi_fit_head2
                        final_phi_probe = phi_probe_head2
                        if _alt_gain < 1e-3:
                            break  # converged

            direct_term_nodes = []
            direct_fit_cols = []
            direct_probe_cols = []
            for t in head_var_terms:
                v_fit = _eval_term(t, x_fit)
                v_probe = _eval_term(t, x_probe)
                if v_fit is None or v_probe is None:
                    continue
                if (not torch.isfinite(v_fit).all()) or (not torch.isfinite(v_probe).all()):
                    continue
                if float(v_fit.std()) < 1.0e-12:
                    continue
                direct_term_nodes.append(t)
                direct_fit_cols.append(v_fit)
                direct_probe_cols.append(v_probe)
            if final_head is not None:
                for jj, t in enumerate(list(final_head.get("terms", []) or [])):
                    if t in direct_term_nodes:
                        continue
                    try:
                        v_fit = final_phi_fit[:, jj]
                        v_probe = final_phi_probe[:, jj]
                    except Exception:
                        continue
                    direct_term_nodes.append(t)
                    direct_fit_cols.append(v_fit)
                    direct_probe_cols.append(v_probe)

            if direct_term_nodes:
                phi_fit_direct = torch.stack(direct_fit_cols, dim=1)
                phi_probe_direct = torch.stack(direct_probe_cols, dim=1)
                direct_combo = _try_direct_combo(
                    expr,
                    pred_fit,
                    pred_probe,
                    direct_term_nodes,
                    phi_fit_direct,
                    phi_probe_direct,
                    mse,
                )
                if direct_combo is not None:
                    return direct_combo

        key, z = fingerprint(r, proj, fp_mode, q_scale, q_clip)
        return (mse, key, z, mapping, expr)

    # Variant 1: as-is
    sc0 = _try(p_fit, p_probe, node)
    if sc0 is not None:
        cands.append(sc0)

    # Variant 2: negated representative (covers the whole mapping-equivalence class)
    node_neg = simplify(_negate_smart(node))
    if node_str(node_neg) != node_str(node):
        sc1 = _try(-p_fit, -p_probe, node_neg)
        if sc1 is not None:
            cands.append(sc1)

    return _pick_best_equiv_score(cands, y_var=y_var)


def _score_expr_base_joint_affine(node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg):
    """Score an expression across multiple datasets using per-dataset affine maps.

    The affine maps are degree-1 poly maps fitted on each dataset's fit split and
    evaluated on each dataset's probe split. The final score is a weighted
    aggregation (points-weighted or datasets-weighted) of the per-dataset probe MSEs.
    """
    if cfg is None or (not bool(cfg.get("joint_score_enable", False))):
        return None
    joint_fit = cfg.get("joint_fit_data", None)
    joint_probe = cfg.get("joint_probe_data", None)
    if (not isinstance(joint_fit, (list, tuple))) or (not isinstance(joint_probe, (list, tuple))):
        return None
    if len(joint_fit) < 2 or len(joint_probe) < 2:
        return None

    # Build id-aligned datasets. If no explicit ids are provided, we align by index.
    fit_by_id = {}
    fit_order = []
    for i, row in enumerate(joint_fit):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        fit_by_id[did_s] = (x_d, y_d)
        fit_order.append(did_s)

    probe_by_id = {}
    for i, row in enumerate(joint_probe):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        probe_by_id[did_s] = (x_d, y_d)

    pairs = []
    for did in fit_order:
        if did not in probe_by_id:
            continue
        xf, yf = fit_by_id[did]
        xp, yp = probe_by_id[did]
        pairs.append((did, xf, yf, xp, yp))
    if len(pairs) < 2:
        return None

    # Dataset weights (default: points-weighted on the probe split).
    w = _joint_dataset_weights([(xp, yp) for (_did, _xf, _yf, xp, yp) in pairs], cfg)
    if w is None or int(w.numel()) != len(pairs):
        return None

    mse_total = torch.zeros((), dtype=w.dtype, device=w.device)
    r_parts = []
    p_fit_parts = []
    y_fit_parts = []
    per_ds = []

    for wi, (did, xf, yf, xp, yp) in zip(w, pairs):
        p_fit = eval_node(node, xf)
        if (p_fit is None) or (not torch.isfinite(p_fit).all()):
            return None
        if float(p_fit.std()) < 1.0e-12:
            return None
        fb = fit_poly(p_fit, yf, degree=1, affine_fast=True, diagnostics=_refine_diag(cfg))
        if fb is None:
            return None
        sol, mu, std = fb
        mapping_d = {"kind": "poly", "coeffs": [float(sol[0]), float(sol[1])], "mu": float(mu), "std": float(std)}
        p_probe = eval_node(node, xp)
        if (p_probe is None) or (not torch.isfinite(p_probe).all()):
            return None
        y_hat = eval_mapping(p_probe, mapping_d)
        r = (yp - y_hat).squeeze(-1)
        mse_d = (r * r).mean()
        if not torch.isfinite(mse_d):
            return None
        mse_total = mse_total + wi * mse_d
        r_parts.append(r)
        p_fit_parts.append(p_fit)
        y_fit_parts.append(yf)
        per_ds.append({
            "id": did,
            "mapping": mapping_d,
            "mse": float(mse_d.detach().cpu()),
            "n_fit": int(yf.shape[0]),
            "n_probe": int(yp.shape[0]),
        })

    if not torch.isfinite(mse_total):
        return None

    r_all = torch.cat(r_parts, dim=0) if r_parts else None
    if r_all is None or int(r_all.numel()) != int(proj.shape[0]):
        return None

    # A representative (pooled) affine map for embedding/serialization.
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    try:
        p_fit_all = torch.cat(p_fit_parts, dim=0)
        y_fit_all = torch.cat(y_fit_parts, dim=0)
        fb_all = fit_poly(p_fit_all, y_fit_all, degree=1, affine_fast=True, diagnostics=_refine_diag(cfg))
        if fb_all is not None:
            sol, mu, std = fb_all
            mapping = {"kind": "poly", "coeffs": [float(sol[0]), float(sol[1])], "mu": float(mu), "std": float(std)}
    except Exception:
        pass

    mapping["_joint_affine"] = {"weight_mode": str(cfg.get("joint_weight_mode", "points")), "datasets": per_ds}

    mse = float(mse_total.detach().cpu())
    if not math.isfinite(mse):
        return None
    key, z = fingerprint(r_all, proj, fp_mode, q_scale, q_clip)
    return mse, key, z, mapping

def _score_expr_base_joint_linear_terms(node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg):
    """Score an expression across multiple datasets using per-dataset linear term coefficients.

    Uses the expression's additive terms as a basis (when enabled via
    ``linear_combo_enable``) and fits the linear coefficients independently per
    dataset on that dataset's fit split. The fitted coefficients are then
    evaluated on the corresponding probe split.

    This generalises the joint affine mapping (degree-1 poly on f(x)) to multiple
    per-dataset parameters (one coefficient per additive term, plus an intercept).
    """
    if cfg is None or (not bool(cfg.get("joint_score_enable", False))):
        return None
    if not bool(cfg.get("joint_terms_enable", False)):
        return None
    joint_fit = cfg.get("joint_fit_data", None)
    joint_probe = cfg.get("joint_probe_data", None)
    if (not isinstance(joint_fit, (list, tuple))) or (not isinstance(joint_probe, (list, tuple))):
        return None
    if len(joint_fit) < 2 or len(joint_probe) < 2:
        return None

    # Build id-aligned datasets. If no explicit ids are provided, we align by index.
    fit_by_id = {}
    fit_order = []
    for i, row in enumerate(joint_fit):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        fit_by_id[did_s] = (x_d, y_d)
        fit_order.append(did_s)

    probe_by_id = {}
    for i, row in enumerate(joint_probe):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        probe_by_id[did_s] = (x_d, y_d)

    pairs = []
    for did in fit_order:
        if did not in probe_by_id:
            continue
        xf, yf = fit_by_id[did]
        xp, yp = probe_by_id[did]
        pairs.append((did, xf, yf, xp, yp))
    if len(pairs) < 2:
        return None

    basis_nodes = _select_linear_basis_nodes(node, cfg)
    if not isinstance(basis_nodes, (list, tuple)) or len(basis_nodes) == 0:
        return None
    term_nodes = list(basis_nodes)

    # Dataset weights (default: points-weighted on the probe split).
    w = _joint_dataset_weights([(xp, yp) for (_did, _xf, _yf, xp, yp) in pairs], cfg)
    if w is None or int(w.numel()) != len(pairs):
        return None

    ridge = float(cfg.get("linear_ridge", 1.0e-8))
    mse_total = torch.zeros((), dtype=w.dtype, device=w.device)
    r_parts = []
    per_ds = []
    p_fit_parts = []
    y_fit_parts = []

    for wi, (did, xf, yf, xp, yp) in zip(w, pairs):
        cols_fit = []
        for t in term_nodes:
            v = eval_node(t, xf)
            if (v is None) or (not torch.isfinite(v).all()):
                return None
            cols_fit.append(v.reshape(-1, 1))
        if not cols_fit:
            return None
        Phi_fit = torch.cat([torch.ones_like(cols_fit[0]), *cols_fit], dim=1)
        if int(Phi_fit.shape[1]) <= 1:
            return None
        col_std = float(Phi_fit[:, 1:].detach().std(unbiased=False))
        if (not math.isfinite(col_std)) or col_std < 1.0e-12:
            return None

        sol = _solve_linear_coeffs(Phi_fit, yf, ridge)
        if sol is None or (not torch.isfinite(sol).all()):
            return None

        cols_probe = []
        for t in term_nodes:
            v = eval_node(t, xp)
            if (v is None) or (not torch.isfinite(v).all()):
                return None
            cols_probe.append(v.reshape(-1, 1))
        if not cols_probe:
            return None
        Phi_probe = torch.cat([torch.ones_like(cols_probe[0]), *cols_probe], dim=1)

        y_hat = Phi_probe @ sol
        r = (yp - y_hat).squeeze(-1)
        mse_d = (r * r).mean()
        if not torch.isfinite(mse_d):
            return None

        mse_total = mse_total + wi * mse_d
        r_parts.append(r)

        try:
            coeffs = [float(v) for v in sol.squeeze(-1).detach().cpu().tolist()]
        except Exception:
            return None

        per_ds.append({
            "id": did,
            "coeffs": coeffs,
            "mse": float(mse_d.detach().cpu()),
            "n_fit": int(yf.shape[0]),
            "n_probe": int(yp.shape[0]),
        })

        # For a representative pooled affine mapping (used for embedding/serialization).
        try:
            p_fit_parts.append(eval_node(node, xf))
            y_fit_parts.append(yf)
        except Exception:
            pass

    if not torch.isfinite(mse_total):
        return None

    r_all = torch.cat(r_parts, dim=0) if r_parts else None
    if r_all is None or int(r_all.numel()) != int(proj.shape[0]):
        return None

    # Representative (pooled) affine mapping for embedding/serialization (same policy as joint affine).
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    try:
        if p_fit_parts and y_fit_parts:
            p_fit_all = torch.cat(p_fit_parts, dim=0)
            y_fit_all = torch.cat(y_fit_parts, dim=0)
            fb_all = fit_poly(p_fit_all, y_fit_all, degree=1, affine_fast=True, diagnostics=_refine_diag(cfg))
            if fb_all is not None:
                sol_aff, mu, std = fb_all
                mapping = {"kind": "poly", "coeffs": [float(sol_aff[0]), float(sol_aff[1])], "mu": float(mu), "std": float(std)}
    except Exception:
        pass

    mapping["_joint_linear_terms"] = {
        "weight_mode": str(cfg.get("joint_weight_mode", "points")),
        "terms": term_nodes,
        "datasets": per_ds,
    }

    mse = float(mse_total.detach().cpu())
    if not math.isfinite(mse):
        return None
    key, z = fingerprint(r_all, proj, fp_mode, q_scale, q_clip)
    return mse, key, z, mapping


def _collect_trig_paths(node, path=()):
    out = []
    op = node[0]
    if op in ("sin", "cos"):
        out.append(path)
    if op in UNARY_OPS:
        out.extend(_collect_trig_paths(node[1], path + (1,)))
    elif op in BINARY_OPS:
        out.extend(_collect_trig_paths(node[1], path + (1,)))
        out.extend(_collect_trig_paths(node[2], path + (2,)))
    return out


def _trig_arg_has_const_scale(arg):
    if arg[0] != "mul":
        return False
    l, r = arg[1], arg[2]
    return (
        (l[0] == "const" and l[1] != 0.0)
        or (r[0] == "const" and r[1] != 0.0)
    )


def _collect_log_paths(node, path=()):
    out = []
    op = node[0]
    if op == "log":
        out.append(path)
    if op in UNARY_OPS:
        out.extend(_collect_log_paths(node[1], path + (1,)))
    elif op in BINARY_OPS:
        out.extend(_collect_log_paths(node[1], path + (1,)))
        out.extend(_collect_log_paths(node[2], path + (2,)))
    return out


def _log_arg_has_const_scale(arg):
    if arg[0] != "mul":
        return False
    l, r = arg[1], arg[2]
    return (
        (l[0] == "const" and l[1] != 0.0)
        or (r[0] == "const" and r[1] != 0.0)
    )


def _collect_exp_paths(node, path=()):
    out = []
    op = node[0]
    if op == "exp":
        out.append(path)
    if op in UNARY_OPS:
        out.extend(_collect_exp_paths(node[1], path + (1,)))
    elif op in BINARY_OPS:
        out.extend(_collect_exp_paths(node[1], path + (1,)))
        out.extend(_collect_exp_paths(node[2], path + (2,)))
    return out


def _exp_arg_has_const_scale(arg):
    if arg[0] != "mul":
        return False
    l, r = arg[1], arg[2]
    return (
        (l[0] == "const" and l[1] != 0.0)
        or (r[0] == "const" and r[1] != 0.0)
    )


def _collect_sqr_shift_paths(node, var_dims=None, path=()):
    """Paths to sqr nodes eligible for additive shift (child must be dimensionless)."""
    out = []
    op = node[0]
    if op == "sqr":
        if var_dims is None:
            out.append(path)
        else:
            ndim = len(var_dims[0])
            dim0 = (0.0,) * ndim
            child_dim = node_dims(node[1], var_dims)
            if child_dim is not None and dims_eq(child_dim, dim0):
                out.append(path)
    if op in UNARY_OPS:
        out.extend(_collect_sqr_shift_paths(node[1], var_dims, path + (1,)))
    elif op in BINARY_OPS:
        out.extend(_collect_sqr_shift_paths(node[1], var_dims, path + (1,)))
        out.extend(_collect_sqr_shift_paths(node[2], var_dims, path + (2,)))
    return out


def _sqr_shift_already_present(node, sqr_path):
    """True if sqr at *sqr_path* is already inside add(..., const/hparam)."""
    if len(sqr_path) < 1:
        return False
    parent_path = sqr_path[:-1]
    child_idx = sqr_path[-1]
    try:
        parent = get_at(node, parent_path)
    except (ValueError, IndexError):
        return False
    if parent[0] != "add":
        return False
    sibling_idx = 2 if child_idx == 1 else 1
    sibling = parent[sibling_idx]
    return sibling[0] in ("const", "hparam")


def _collect_sqrt_shift_paths(node, var_dims=None, path=()):
    """Paths to sqrt nodes eligible for inner additive shift (child must be dimensionless)."""
    out = []
    op = node[0]
    if op == "sqrt":
        if var_dims is None:
            out.append(path)
        else:
            ndim = len(var_dims[0])
            dim0 = (0.0,) * ndim
            child_dim = node_dims(node[1], var_dims)
            if child_dim is not None and dims_eq(child_dim, dim0):
                out.append(path)
    if op in UNARY_OPS:
        out.extend(_collect_sqrt_shift_paths(node[1], var_dims, path + (1,)))
    elif op in BINARY_OPS:
        out.extend(_collect_sqrt_shift_paths(node[1], var_dims, path + (1,)))
        out.extend(_collect_sqrt_shift_paths(node[2], var_dims, path + (2,)))
    return out


def _sqrt_shift_already_present(node, sqrt_path):
    """True if sqrt at *sqrt_path* already has add(..., const/hparam) inside."""
    try:
        sqrt_node = get_at(node, sqrt_path)
    except (ValueError, IndexError):
        return False
    arg = sqrt_node[1]
    if arg[0] != "add":
        return False
    return arg[2][0] in ("const", "hparam") or arg[1][0] in ("const", "hparam")


def _wrap_param_slots(node, trig_slot_map, log_slot_map, path=(), exp_slot_map=None,
                      sqr_shift_slot_map=None, sqrt_shift_slot_map=None):
    if exp_slot_map is None:
        exp_slot_map = {}
    if sqr_shift_slot_map is None:
        sqr_shift_slot_map = {}
    if sqrt_shift_slot_map is None:
        sqrt_shift_slot_map = {}
    _kw = dict(exp_slot_map=exp_slot_map,
               sqr_shift_slot_map=sqr_shift_slot_map,
               sqrt_shift_slot_map=sqrt_shift_slot_map)
    op = node[0]
    if op in ("var", "const", "hparam"):
        return node
    if op in UNARY_OPS:
        child = _wrap_param_slots(node[1], trig_slot_map, log_slot_map, path + (1,), **_kw)
        if op in ("sin", "cos") and path in trig_slot_map:
            child = ("mul", ("hparam", int(trig_slot_map[path])), child)
            return (op, child)
        if op == "log" and path in log_slot_map:
            child = ("mul", ("hparam", int(log_slot_map[path])), child)
            return ("log", child)
        if op == "exp" and path in exp_slot_map:
            child = ("mul", ("hparam", int(exp_slot_map[path])), child)
            return ("exp", child)
        # sqr shift:  sqr(child) → add(sqr(child), hparam_k)
        if op == "sqr" and path in sqr_shift_slot_map:
            return ("add", ("sqr", child), ("hparam", int(sqr_shift_slot_map[path])))
        # sqrt shift: sqrt(child) → sqrt(add(child, hparam_k))
        if op == "sqrt" and path in sqrt_shift_slot_map:
            return ("sqrt", ("add", child, ("hparam", int(sqrt_shift_slot_map[path]))))
        return (op, child)
    return (
        op,
        _wrap_param_slots(node[1], trig_slot_map, log_slot_map, path + (1,), **_kw),
        _wrap_param_slots(node[2], trig_slot_map, log_slot_map, path + (2,), **_kw),
    )


def _refine_diag(cfg):
    if not isinstance(cfg, dict):
        return None
    diag = cfg.get("diagnostics", cfg.get("refine_diagnostics", None))
    return diag if isinstance(diag, dict) else None


def _diag_inc(cfg, key, amount=1):
    diag = _refine_diag(cfg)
    if diag is not None:
        diag[str(key)] = int(diag.get(str(key), 0)) + int(amount)


def _diag_inc_context(cfg, suffix, amount=1):
    if not isinstance(cfg, dict):
        return
    context = str(cfg.get("refine_context", "") or "").strip().lower()
    if not context:
        return
    context = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in context)
    context = context.strip("_")
    if not context:
        return
    _diag_inc(cfg, f"{context}_{suffix}", amount)


def _diag_add_time(cfg, key, elapsed):
    diag = _refine_diag(cfg)
    if diag is not None:
        diag[str(key)] = float(diag.get(str(key), 0.0)) + float(max(0.0, elapsed))


def _node_var_indices(node, out=None):
    if out is None:
        out = set()
    if not (isinstance(node, tuple) and node):
        return out
    if node[0] == "var":
        try:
            out.add(int(node[1]))
        except Exception:
            pass
        return out
    if node[0] in ("const", "hparam"):
        return out
    if node[0] in UNARY_OPS:
        return _node_var_indices(node[1], out)
    if node[0] in BINARY_OPS:
        _node_var_indices(node[1], out)
        _node_var_indices(node[2], out)
    return out


def _refine_tensor_signature(x_fit, y_fit):
    try:
        shape_x = tuple(int(v) for v in x_fit.shape)
        shape_y = tuple(int(v) for v in y_fit.shape)
        x_det = x_fit.detach()
        y_det = y_fit.detach()
        x_mean = float(torch.nanmean(x_det).detach().cpu())
        x_std = float(torch.sqrt(torch.nanmean((x_det - x_mean) ** 2)).detach().cpu())
        y_mean = float(torch.nanmean(y_det).detach().cpu())
        y_std = float(torch.sqrt(torch.nanmean((y_det - y_mean) ** 2)).detach().cpu())
        return (
            shape_x,
            shape_y,
            str(x_fit.dtype),
            str(y_fit.dtype),
            int(id(x_fit)),
            int(id(y_fit)),
            round(x_mean, 12),
            round(x_std, 12),
            round(y_mean, 12),
            round(y_std, 12),
        )
    except Exception:
        return None


def _refine_cfg_signature(cfg):
    keys = (
        "optimizer", "lbfgs_escalate_improve_factor", "lbfgs_steps", "fit_subset",
        "fit_subset_mode", "num_restarts", "max_params", "linear_combo_enable",
        "linear_terms_max", "linear_prune_rel", "linear_ridge", "safe_eps",
        "safe_penalty_weight", "safe_exp_clip", "theta_l2", "init_log_min",
        "init_log_max", "refine_grid_enable", "refine_grid_size", "refine_grid_size_2d",
        "refine_grid_passes", "refine_grid_topk", "refine_grid_max_evals",
        "joint_refine_enable", "joint_weight_mode",
    )
    return tuple((k, cfg.get(k, None)) for k in keys)


def _refine_attempt_cache_key(var_h, n_params, shift_slots, cfg, x_fit, y_fit):
    data_identity = (int(id(x_fit)), int(id(y_fit)))
    data_sig = cfg.get("_attempt_cache_data_signature", None)
    if data_sig is None or cfg.get("_attempt_cache_data_identity", None) != data_identity:
        data_sig = _refine_tensor_signature(x_fit, y_fit)
        cfg["_attempt_cache_data_signature"] = data_sig
        cfg["_attempt_cache_data_identity"] = data_identity
    joint = cfg.get("joint_fit_data", None)
    if isinstance(joint, (list, tuple)):
        joint_sig = tuple(
            str(row[0]) if isinstance(row, (tuple, list)) and len(row) == 3 else str(i)
            for i, row in enumerate(joint)
        )
    else:
        joint_sig = ()
    return (
        node_str(var_h),
        int(n_params),
        tuple(sorted(int(v) for v in shift_slots)),
        tuple(sorted(_node_var_indices(var_h))),
        data_sig,
        joint_sig,
        _refine_cfg_signature(cfg),
    )


def _refine_cache_get(cfg, key):
    if not bool(cfg.get("attempt_cache_enable", True)):
        return None
    cache = cfg.get("attempt_cache", None)
    if not isinstance(cache, dict):
        return None
    if key not in cache:
        _diag_inc(cfg, "attempt_cache_misses")
        return None
    _diag_inc(cfg, "attempt_cache_hits")
    return cache.get(key)


def _refine_cache_put(cfg, key, entry):
    if not bool(cfg.get("attempt_cache_enable", True)):
        return
    cache = cfg.get("attempt_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        cfg["attempt_cache"] = cache
    max_entries = max(0, int(cfg.get("attempt_cache_max_entries", 4096)))
    if max_entries <= 0:
        _diag_inc(cfg, "attempt_cache_skipped_full")
        return
    if len(cache) >= max_entries:
        try:
            cache.pop(next(iter(cache)))
            _diag_inc(cfg, "attempt_cache_evictions")
        except Exception:
            _diag_inc(cfg, "attempt_cache_skipped_full")
            return
    cache[key] = dict(entry)
    _diag_inc(cfg, "attempt_cache_stores")


def _prune_mapping_equiv_root_slot_paths(node, log_paths, cfg):
    """Drop only root-log scale slots, which an affine outer map absorbs exactly."""
    if not bool(cfg.get("prune_mapping_equiv_root_slots", True)):
        return log_paths
    if isinstance(node, tuple) and node and node[0] == "log" and () in log_paths:
        _diag_inc(cfg, "mapping_equiv_root_slots_pruned")
        return [p for p in log_paths if p != ()]
    return log_paths


def _decorate_refine_variants(node, max_variants, max_params, x_fit=None, y_fit=None, cfg=None):
    """Generate decorated variants with hparam slots.

    Returns list of ``(var_node, n_params, shift_slots)`` where *shift_slots*
    is a frozenset of hparam indices that use raw (unrestricted sign)
    parameterisation instead of ``exp(raw)`` (always positive).
    """
    cfg = cfg or {}
    trig_paths = []
    for p in _collect_trig_paths(node):
        trig = get_at(node, p)
        if not _trig_arg_has_const_scale(trig[1]):
            trig_paths.append(p)
    log_paths = []
    for p in _collect_log_paths(node):
        lg = get_at(node, p)
        if not _log_arg_has_const_scale(lg[1]):
            log_paths.append(p)
    log_paths = _prune_mapping_equiv_root_slot_paths(node, log_paths, cfg)
    exp_paths = []
    for p in _collect_exp_paths(node):
        ep = get_at(node, p)
        if not _exp_arg_has_const_scale(ep[1]):
            exp_paths.append(p)

    # sqr/sqrt shift paths — only when argument is dimensionless
    _var_dims = cfg.get("var_dims", None)
    sqr_shift_paths = []
    for p in _collect_sqr_shift_paths(node, var_dims=_var_dims):
        if not _sqr_shift_already_present(node, p):
            sqr_shift_paths.append(p)
    sqrt_shift_paths = []
    for p in _collect_sqrt_shift_paths(node, var_dims=_var_dims):
        if not _sqrt_shift_already_present(node, p):
            sqrt_shift_paths.append(p)

    use_sensitivity = (
        bool(cfg.get("slot_sensitivity_enable", True))
        and torch.is_tensor(x_fit)
        and torch.is_tensor(y_fit)
    )
    if use_sensitivity:
        if len(trig_paths) > 1:
            trig_paths = _rank_paths_by_sensitivity(node, trig_paths, "trig", x_fit, y_fit, cfg)
        else:
            trig_paths.sort(key=lambda p: (len(p), p))
        if len(log_paths) > 1:
            log_paths = _rank_paths_by_sensitivity(node, log_paths, "log", x_fit, y_fit, cfg)
        else:
            log_paths.sort(key=lambda p: (len(p), p))
        if len(exp_paths) > 1:
            exp_paths = _rank_paths_by_sensitivity(node, exp_paths, "exp", x_fit, y_fit, cfg)
        else:
            exp_paths.sort(key=lambda p: (len(p), p))
        if len(sqr_shift_paths) > 1:
            sqr_shift_paths = _rank_paths_by_sensitivity(node, sqr_shift_paths, "sqr_shift", x_fit, y_fit, cfg)
        else:
            sqr_shift_paths.sort(key=lambda p: (len(p), p))
        if len(sqrt_shift_paths) > 1:
            sqrt_shift_paths = _rank_paths_by_sensitivity(node, sqrt_shift_paths, "sqrt_shift", x_fit, y_fit, cfg)
        else:
            sqrt_shift_paths.sort(key=lambda p: (len(p), p))
    else:
        trig_paths.sort(key=lambda p: (len(p), p))
        log_paths.sort(key=lambda p: (len(p), p))
        exp_paths.sort(key=lambda p: (len(p), p))
        sqr_shift_paths.sort(key=lambda p: (len(p), p))
        sqrt_shift_paths.sort(key=lambda p: (len(p), p))

    _no_shift = frozenset()
    out = []
    seen = set()

    def _push(var_node, n_params, shift_slots=_no_shift):
        key = node_str(var_node)
        if key in seen:
            return False
        seen.add(key)
        out.append((var_node, n_params, shift_slots))
        return len(out) >= max_variants

    # Single-param variants: trig, log, exp (scale slots)
    for tp in trig_paths:
        v = _wrap_param_slots(node, {tp: 0}, {})
        if _push(v, 1):
            return out
    for lp in log_paths:
        v = _wrap_param_slots(node, {}, {lp: 0})
        if _push(v, 1):
            return out
    for ep in exp_paths:
        v = _wrap_param_slots(node, {}, {}, exp_slot_map={ep: 0})
        if _push(v, 1):
            return out
    # Single-param variants: sqr/sqrt shift (shift slots — raw parameterisation)
    for sp in sqr_shift_paths:
        v = _wrap_param_slots(node, {}, {}, sqr_shift_slot_map={sp: 0})
        if _push(v, 1, frozenset({0})):
            return out
    for sp in sqrt_shift_paths:
        v = _wrap_param_slots(node, {}, {}, sqrt_shift_slot_map={sp: 0})
        if _push(v, 1, frozenset({0})):
            return out

    # Two-param variants: all pairwise combinations (including same type)
    if int(max_params) >= 2:
        # trig + trig (different paths)
        for i, tp1 in enumerate(trig_paths):
            for tp2 in trig_paths[i + 1:]:
                v = _wrap_param_slots(node, {tp1: 0, tp2: 1}, {})
                if _push(v, 2):
                    return out
        # trig + log
        for tp in trig_paths:
            for lp in log_paths:
                v = _wrap_param_slots(node, {tp: 0}, {lp: 1})
                if _push(v, 2):
                    return out
        # trig + exp
        for tp in trig_paths:
            for ep in exp_paths:
                v = _wrap_param_slots(node, {tp: 0}, {}, exp_slot_map={ep: 1})
                if _push(v, 2):
                    return out
        # trig + sqr_shift
        for tp in trig_paths:
            for sp in sqr_shift_paths:
                v = _wrap_param_slots(node, {tp: 0}, {}, sqr_shift_slot_map={sp: 1})
                if _push(v, 2, frozenset({1})):
                    return out
        # trig + sqrt_shift
        for tp in trig_paths:
            for sp in sqrt_shift_paths:
                v = _wrap_param_slots(node, {tp: 0}, {}, sqrt_shift_slot_map={sp: 1})
                if _push(v, 2, frozenset({1})):
                    return out
        # log + log (different paths)
        for i, lp1 in enumerate(log_paths):
            for lp2 in log_paths[i + 1:]:
                v = _wrap_param_slots(node, {}, {lp1: 0, lp2: 1})
                if _push(v, 2):
                    return out
        # log + exp
        for lp in log_paths:
            for ep in exp_paths:
                v = _wrap_param_slots(node, {}, {lp: 0}, exp_slot_map={ep: 1})
                if _push(v, 2):
                    return out
        # exp + exp (different paths)
        for i, ep1 in enumerate(exp_paths):
            for ep2 in exp_paths[i + 1:]:
                v = _wrap_param_slots(node, {}, {}, exp_slot_map={ep1: 0, ep2: 1})
                if _push(v, 2):
                    return out
        # sqr_shift + sqr_shift (different paths)
        for i, sp1 in enumerate(sqr_shift_paths):
            for sp2 in sqr_shift_paths[i + 1:]:
                v = _wrap_param_slots(node, {}, {}, sqr_shift_slot_map={sp1: 0, sp2: 1})
                if _push(v, 2, frozenset({0, 1})):
                    return out
        # sqr_shift + sqrt_shift
        for sp1 in sqr_shift_paths:
            for sp2 in sqrt_shift_paths:
                v = _wrap_param_slots(node, {}, {}, sqr_shift_slot_map={sp1: 0}, sqrt_shift_slot_map={sp2: 1})
                if _push(v, 2, frozenset({0, 1})):
                    return out
        # sqrt_shift + sqrt_shift (different paths)
        for i, sp1 in enumerate(sqrt_shift_paths):
            for sp2 in sqrt_shift_paths[i + 1:]:
                v = _wrap_param_slots(node, {}, {}, sqrt_shift_slot_map={sp1: 0, sp2: 1})
                if _push(v, 2, frozenset({0, 1})):
                    return out
    return out


def _eval_node_hparam(node, x, hparams):
    op = node[0]
    if op == "hparam":
        i = int(node[1])
        hp = hparams[i]
        if torch.is_tensor(hp):
            return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device) * hp
        return torch.full((x.shape[0], 1), float(hp), dtype=x.dtype, device=x.device)
    if op == "var":
        i = node[1]
        return x[:, i:i+1]
    if op == "const":
        return torch.full((x.shape[0], 1), node[1], dtype=x.dtype, device=x.device)
    if op == "sin":
        return torch.sin(_eval_node_hparam(node[1], x, hparams))
    if op == "cos":
        return torch.cos(_eval_node_hparam(node[1], x, hparams))
    if op == "exp":
        return torch.exp(_eval_node_hparam(node[1], x, hparams))
    if op == "log":
        return torch.log(_eval_node_hparam(node[1], x, hparams))
    if op == "sqrt":
        return torch.sqrt(_eval_node_hparam(node[1], x, hparams))
    if op == "sqr":
        c = _eval_node_hparam(node[1], x, hparams)
        return c * c
    if op == "neg":
        return -_eval_node_hparam(node[1], x, hparams)
    if op == "add":
        return _eval_node_hparam(node[1], x, hparams) + _eval_node_hparam(node[2], x, hparams)
    if op == "sub":
        return _eval_node_hparam(node[1], x, hparams) - _eval_node_hparam(node[2], x, hparams)
    if op == "mul":
        return _eval_node_hparam(node[1], x, hparams) * _eval_node_hparam(node[2], x, hparams)
    if op == "div":
        a = _eval_node_hparam(node[1], x, hparams)
        b = _eval_node_hparam(node[2], x, hparams)
        return a / b
    raise ValueError(op)


def _materialize_hparams(node, hparams):
    op = node[0]
    if op == "hparam":
        i = int(node[1])
        return ("const", float(hparams[i]))
    if op in ("var", "const"):
        return node
    if op in UNARY_OPS:
        return (op, _materialize_hparams(node[1], hparams))
    return (op, _materialize_hparams(node[1], hparams), _materialize_hparams(node[2], hparams))


def _build_init_logs(n_params, restarts, log_min, log_max, dtype, device):
    if restarts <= 1:
        return torch.zeros((1, n_params), dtype=dtype, device=device)
    out = torch.empty((restarts, n_params), dtype=dtype, device=device)
    span = float(log_max - log_min)
    for r in range(restarts):
        for j in range(n_params):
            t = ((r + 0.61803398875 * (j + 1)) % restarts) / float(restarts)
            t = min(1.0, max(0.0, float(t)))
            out[r, j] = float(log_min + span * t)
    neutral = float(min(log_max, max(log_min, 0.0)))
    out[0, :] = neutral
    return out


def _raw_to_hparams(raw, shift_slots):
    """Convert raw optimisation variables to hparams.

    Scale slots (trig/log/exp): ``exp(raw)`` — always positive.
    Shift slots (sqr_shift/sqrt_shift): ``raw`` directly — unrestricted sign.
    """
    if not shift_slots:
        return torch.exp(raw)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    for k in shift_slots:
        if k < raw.shape[0]:
            mask[k] = True
    return torch.where(mask, raw, torch.exp(raw))


def _stable_seed_from_text(text):
    h = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


def _select_subset_indices(n, n_sub, mode, seed, x_ref=None):
    n = int(n)
    n_sub = int(n_sub)
    if n_sub <= 0 or n_sub >= n:
        return torch.arange(n, dtype=torch.long, device=x_ref.device if torch.is_tensor(x_ref) else "cpu")

    mode = str(mode or "hash_random").strip().lower()
    if mode == "stride":
        step = max(1, n // n_sub)
        dev = x_ref.device if torch.is_tensor(x_ref) else "cpu"
        return torch.arange(0, n, step, device=dev, dtype=torch.long)[:n_sub]

    if mode == "stratified" and torch.is_tensor(x_ref) and x_ref.ndim == 2 and x_ref.shape[1] > 0:
        axis = 0
        if x_ref.shape[1] > 1:
            col_var = ((x_ref - x_ref.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
            axis = int(torch.argmax(col_var).item())
        order = torch.argsort(x_ref[:, axis])
        edges = torch.linspace(0, n, steps=n_sub + 1, dtype=torch.float64, device=order.device)
        pos = torch.clamp(((edges[:-1] + edges[1:]) * 0.5).floor().to(torch.long), min=0, max=n - 1)
        return order.index_select(0, pos)

    # hash-random default: deterministic per expression hash
    g = torch.Generator(device="cpu").manual_seed(int(seed % (2**63 - 1)))
    idx = torch.randperm(n, generator=g, dtype=torch.long)[:n_sub]
    if torch.is_tensor(x_ref):
        idx = idx.to(device=x_ref.device)
    return idx


def _slice_fit_subset(x_fit, y_fit, n_sub, mode, seed):
    n = int(x_fit.shape[0])
    if not (0 < int(n_sub) < n):
        return x_fit, y_fit
    idx = _select_subset_indices(
        n=n,
        n_sub=int(n_sub),
        mode=mode,
        seed=seed,
        x_ref=x_fit,
    )
    return x_fit.index_select(0, idx), y_fit.index_select(0, idx)


def _slice_fit_subset_multi(joint_data, n_sub_total, mode, seed, *, min_per_dataset=8):
    """Subsample each dataset in *joint_data*.

    Parameters
    ----------
    joint_data : list
        List entries may be ``(x, y)`` or ``(id, x, y)``.  The return value
        preserves that structure.
    n_sub_total : int
        Total subsample budget across all datasets. We split this budget
        roughly evenly across datasets so runtime stays comparable to the
        single-dataset path.
    mode : str
        Subsampling mode (see _select_subset_indices).
    seed : int
        Base seed. Each dataset gets a deterministic offset.
    min_per_dataset : int
        Minimum samples per dataset when subsampling is active.
    """
    if not isinstance(joint_data, (list, tuple)):
        return joint_data

    recs = []
    for i, row in enumerate(joint_data):
        if row is None:
            continue
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x, y = row[0], row[1], row[2]
            kind = 3
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x, y = None, row[0], row[1]
            kind = 2
        else:
            continue
        if torch.is_tensor(x) and torch.is_tensor(y):
            recs.append((kind, did, x, y))

    D = len(recs)
    if D <= 0:
        return []

    n_sub_total = int(n_sub_total)
    if n_sub_total <= 0:
        out = []
        for kind, did, x_d, y_d in recs:
            out.append((did, x_d, y_d) if kind == 3 else (x_d, y_d))
        return out

    # Split the total budget across datasets (roughly evenly).
    per = int(n_sub_total)
    if D > 1:
        per = max(int(min_per_dataset), int(per // D))
    per = max(1, int(per))

    out = []
    for di, (kind, did, x_d, y_d) in enumerate(recs):
        n_d = int(x_d.shape[0])
        take = min(n_d, per)
        if not (0 < take < n_d):
            out.append((did, x_d, y_d) if kind == 3 else (x_d, y_d))
            continue
        idx = _select_subset_indices(
            n=n_d,
            n_sub=take,
            mode=mode,
            seed=int(seed + (di + 1) * 1_000_003),
            x_ref=x_d,
        )
        xs = x_d.index_select(0, idx)
        ys = y_d.index_select(0, idx)
        out.append((did, xs, ys) if kind == 3 else (xs, ys))

    return out


def _solve_linear_coeffs(Phi, y, ridge):
    ridge = max(0.0, float(ridge))
    if Phi is None or y is None or Phi.ndim != 2 or y.ndim != 2:
        return None
    if int(Phi.shape[0]) <= 0 or int(Phi.shape[1]) <= 0:
        return None
    if ridge > 0.0:
        try:
            k = int(Phi.shape[1])
            eye = torch.eye(k, dtype=Phi.dtype, device=Phi.device)
            gram = Phi.transpose(0, 1) @ Phi + ridge * eye
            rhs = Phi.transpose(0, 1) @ y
            return torch.linalg.solve(gram, rhs)
        except Exception:
            pass
    try:
        return torch.linalg.lstsq(Phi, y).solution
    except Exception:
        return None


def _solve_linearized_fit(expr_h, basis_nodes, x_fit, y_fit, hparams, cfg, *, safe=True, ridge_override=None):
    _diag_inc(cfg, "linear_solves")
    Phi, pen = _build_phi_hparam(basis_nodes, x_fit, hparams, cfg=cfg, safe=safe)
    if Phi is None or int(Phi.shape[1]) <= 1:
        return None
    col_std = float(Phi[:, 1:].detach().std(unbiased=False))
    if (not math.isfinite(col_std)) or col_std < 1.0e-12:
        return None
    ridge = float(cfg.get("linear_ridge", 1.0e-8)) if ridge_override is None else float(ridge_override)
    sol = _solve_linear_coeffs(Phi, y_fit, ridge)
    if sol is None or (not torch.isfinite(sol).all()):
        return None
    y_hat = Phi @ sol
    mse = ((y_hat - y_fit) ** 2).mean()
    if not torch.isfinite(mse):
        return None
    return mse, pen, Phi, sol, y_hat


def _joint_dataset_weights(joint_data, cfg):
    """Compute weights for joint multi-dataset refinement/scoring.

    Supports joint entries shaped as ``(x, y)`` or ``(id, x, y)``.
    """
    if not isinstance(joint_data, (list, tuple)):
        return None

    # Find the first tensor to set dtype/device.
    x0 = None
    for row in joint_data:
        if row is None:
            continue
        if isinstance(row, (tuple, list)) and len(row) == 3:
            x = row[1]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            x = row[0]
        else:
            continue
        if torch.is_tensor(x):
            x0 = x
            break

    if x0 is None:
        return None

    dtype = x0.dtype
    device = x0.device

    # Collect sizes for valid datasets.
    sizes = []
    for row in joint_data:
        if row is None:
            continue
        if isinstance(row, (tuple, list)) and len(row) == 3:
            x_d = row[1]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            x_d = row[0]
        else:
            continue
        if not torch.is_tensor(x_d):
            continue
        try:
            sizes.append(float(int(x_d.shape[0])))
        except Exception:
            sizes.append(0.0)

    D = len(sizes)
    if D <= 0:
        return None

    mode = str(cfg.get("joint_weight_mode", "points")).strip().lower()
    if mode in ("datasets", "dataset", "equal", "uniform"):
        return torch.full((D,), 1.0 / float(D), dtype=dtype, device=device)

    s = torch.as_tensor(sizes, dtype=dtype, device=device)
    tot = float(s.sum().detach().cpu())
    if not math.isfinite(tot) or tot <= 0.0:
        return torch.full((D,), 1.0 / float(D), dtype=dtype, device=device)
    return s / s.sum()


def _solve_linearized_fit_multi(expr_h, basis_nodes, joint_data, hparams, cfg, *, safe=True, ridge_override=None):
    """Linearized fit for multiple datasets with shared hparams.

    For a fixed *hparams*, solves the linear coefficients independently per
    dataset and aggregates the resulting MSEs (and safety penalties).
    """
    if not isinstance(joint_data, (list, tuple)) or len(joint_data) == 0:
        return None
    _diag_inc(cfg, "linear_solves_multi")
    joint = []
    for row in joint_data:
        if row is None:
            continue
        if isinstance(row, (tuple, list)) and len(row) == 3:
            x, y = row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            x, y = row[0], row[1]
        else:
            continue
        if torch.is_tensor(x) and torch.is_tensor(y):
            joint.append((x, y))
    if len(joint) == 0:
        return None

    w = _joint_dataset_weights(joint, cfg)
    if w is None or int(w.numel()) != len(joint):
        return None

    # Weighted aggregation so the objective scale is similar to the single-dataset path.
    mse_total = torch.zeros((), dtype=joint[0][0].dtype, device=joint[0][0].device)
    pen_total = torch.zeros((), dtype=joint[0][0].dtype, device=joint[0][0].device)
    Phi_list = []
    sol_list = []
    yhat_list = []

    for wi, (x_d, y_d) in zip(w, joint):
        solved = _solve_linearized_fit(
            expr_h,
            basis_nodes,
            x_d,
            y_d,
            hparams,
            cfg,
            safe=safe,
            ridge_override=ridge_override,
        )
        if solved is None:
            return None
        mse_d, pen_d, Phi_d, sol_d, yhat_d = solved
        mse_total = mse_total + wi * mse_d
        pen_total = pen_total + wi * pen_d
        Phi_list.append(Phi_d)
        sol_list.append(sol_d)
        yhat_list.append(yhat_d)

    if not torch.isfinite(mse_total):
        return None
    return mse_total, pen_total, Phi_list, sol_list, yhat_list


def _linearized_loss_value(expr_h, basis_nodes, x_fit, y_fit, hparams, cfg, *, joint_data=None):
    if joint_data is not None:
        solved = _solve_linearized_fit_multi(expr_h, basis_nodes, joint_data, hparams, cfg, safe=True)
    else:
        solved = _solve_linearized_fit(expr_h, basis_nodes, x_fit, y_fit, hparams, cfg, safe=True)
    if solved is None:
        return None
    mse, pen, _, _, _ = solved
    safe_penalty_weight = float(cfg.get("safe_penalty_weight", 1.0e-2))
    loss = mse + safe_penalty_weight * pen
    out = float(loss.detach().cpu())
    if not math.isfinite(out):
        return None
    return out


def _build_single_slot_variant(node, slot_kind, path):
    if slot_kind == "trig":
        return _wrap_param_slots(node, {path: 0}, {})
    if slot_kind == "log":
        return _wrap_param_slots(node, {}, {path: 0})
    if slot_kind == "exp":
        return _wrap_param_slots(node, {}, {}, exp_slot_map={path: 0})
    if slot_kind == "sqr_shift":
        return _wrap_param_slots(node, {}, {}, sqr_shift_slot_map={path: 0})
    if slot_kind == "sqrt_shift":
        return _wrap_param_slots(node, {}, {}, sqrt_shift_slot_map={path: 0})
    return None


def _slot_sensitivity_score(node, slot_kind, path, x_fit, y_fit, cfg):
    var_h = _build_single_slot_variant(node, slot_kind, path)
    if var_h is None:
        return float("-inf")
    subset_n = int(cfg.get("slot_sensitivity_subset", 64))
    subset_mode = str(cfg.get("fit_subset_mode", "hash_random"))
    seed = _stable_seed_from_text(f"sens|{slot_kind}|{path}|{node_str(node)}")
    joint = cfg.get("joint_fit_data", None)
    joint_sub = None
    if bool(cfg.get("joint_refine_enable", True)) and isinstance(joint, (list, tuple)) and len(joint) >= 2:
        joint_sub = _slice_fit_subset_multi(joint, subset_n, subset_mode, seed)
        xf, yf = x_fit, y_fit
    else:
        xf, yf = _slice_fit_subset(x_fit, y_fit, subset_n, subset_mode, seed)
    basis_nodes = _select_linear_basis_nodes(var_h, cfg)
    loss0 = _linearized_loss_value(var_h, basis_nodes, xf, yf, [1.0], cfg, joint_data=joint_sub)
    delta = max(1.0e-3, float(cfg.get("slot_sensitivity_delta", 0.1)))
    loss1 = _linearized_loss_value(var_h, basis_nodes, xf, yf, [1.0 + delta], cfg, joint_data=joint_sub)
    if loss0 is None or loss1 is None:
        return float("-inf")
    denom = max(abs(loss0), 1.0e-12)
    return abs(loss1 - loss0) / denom


def _rank_paths_by_sensitivity(node, paths, slot_kind, x_fit, y_fit, cfg):
    if len(paths) <= 1:
        return paths
    max_paths = max(1, int(cfg.get("slot_sensitivity_max_paths", 24)))
    base_sorted = sorted(paths, key=lambda p: (len(p), p))
    head = base_sorted[:max_paths]
    tail = base_sorted[max_paths:]
    scored = []
    for p in head:
        sc = _slot_sensitivity_score(node, slot_kind, p, x_fit, y_fit, cfg)
        if not math.isfinite(sc):
            sc = -1.0e30
        scored.append((float(sc), p))
    scored.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    ranked = [p for _, p in scored]
    ranked.extend(tail)
    return ranked


def _variant_has_gate_potential(expr_h, n_params, x_fit, y_fit, cfg):
    if int(n_params) <= 0:
        return True
    if not bool(cfg.get("gate_potential_enable", True)):
        return False
    _diag_inc(cfg, "gate_potential_checks")

    subset_n = int(cfg.get("gate_potential_subset", 64))
    subset_mode = str(cfg.get("fit_subset_mode", "hash_random"))
    seed = _stable_seed_from_text(f"gate|{node_str(expr_h)}")
    joint = cfg.get("joint_fit_data", None)
    joint_sub = None
    if bool(cfg.get("joint_refine_enable", True)) and isinstance(joint, (list, tuple)) and len(joint) >= 2:
        joint_sub = _slice_fit_subset_multi(joint, subset_n, subset_mode, seed)
        xf, yf = x_fit, y_fit
    else:
        xf, yf = _slice_fit_subset(x_fit, y_fit, subset_n, subset_mode, seed)
    basis_nodes = _select_linear_basis_nodes(expr_h, cfg)

    base_h = [1.0] * int(n_params)
    base_loss = _linearized_loss_value(expr_h, basis_nodes, xf, yf, base_h, cfg, joint_data=joint_sub)
    if base_loss is None or (not math.isfinite(base_loss)):
        return True
    if base_loss <= 1.0e-30:
        return False

    log_min = float(cfg.get("gate_log_min", math.log(0.5)))
    log_max = float(cfg.get("gate_log_max", math.log(4.0)))
    grid_n = max(2, int(cfg.get("gate_grid_size", 4)))
    axis = torch.linspace(log_min, log_max, grid_n, dtype=xf.dtype, device=xf.device)
    vals = [1.0]
    vals.extend(float(v) for v in torch.exp(axis).detach().cpu().tolist())
    vals = sorted(set(vals))

    combos = list(itertools.product(vals, repeat=int(n_params)))
    max_evals = max(1, int(cfg.get("gate_max_evals", 64)))
    if len(combos) > max_evals:
        step = max(1, len(combos) // max_evals)
        combos = combos[::step][:max_evals]

    best = float(base_loss)
    for hp in combos:
        _diag_inc(cfg, "gate_potential_evals")
        loss = _linearized_loss_value(expr_h, basis_nodes, xf, yf, hp, cfg, joint_data=joint_sub)
        if loss is None or (not math.isfinite(loss)):
            continue
        if loss < best:
            best = loss

    improve_factor = max(1.0, float(cfg.get("gate_potential_improve_factor", 5.0)))
    return best <= float(base_loss) / improve_factor


def _build_grid_seed_logs(expr_h, n_params, xf, yf, cfg, restarts, basis_nodes, *,
                          joint_data=None, shift_slots=frozenset()):
    if int(n_params) <= 0 or not bool(cfg.get("refine_grid_enable", True)):
        return None

    log_min = float(cfg.get("init_log_min", -1.5))
    log_max = float(cfg.get("init_log_max", 1.5))
    if log_max <= log_min:
        return None

    n_params = int(n_params)
    restarts = max(1, int(restarts))
    safe_penalty_weight = float(cfg.get("safe_penalty_weight", 1.0e-2))
    coarse_n_1d = max(5, int(cfg.get("refine_grid_size", 33)))
    coarse_n_2d = max(5, int(cfg.get("refine_grid_size_2d", 11)))
    refine_passes = max(0, int(cfg.get("refine_grid_passes", 2)))
    max_evals = max(1, int(cfg.get("refine_grid_max_evals", 256)))
    topk = max(1, min(restarts, int(cfg.get("refine_grid_topk", restarts))))

    if joint_data is not None and isinstance(joint_data, (list, tuple)) and len(joint_data) > 0:
        j0 = joint_data[0]
        x0 = j0[1] if (isinstance(j0, (tuple, list)) and len(j0) == 3) else j0[0]
        _dtype = x0.dtype
        _device = x0.device
    else:
        _dtype = xf.dtype
        _device = xf.device

    def _score_log(log_vals):
        _diag_inc(cfg, "grid_evals")
        lv = torch.as_tensor(log_vals, dtype=_dtype, device=_device)
        hparams = _raw_to_hparams(lv, shift_slots)
        if joint_data is not None:
            solved = _solve_linearized_fit_multi(expr_h, basis_nodes, joint_data, hparams, cfg, safe=True)
        else:
            solved = _solve_linearized_fit(expr_h, basis_nodes, xf, yf, hparams, cfg, safe=True)
        if solved is None:
            return float("inf"), lv
        mse, pen, _, _, _ = solved
        loss = float((mse + safe_penalty_weight * pen).detach().cpu())
        if not math.isfinite(loss):
            return float("inf"), lv
        return loss, lv

    def _grid_points(center=None, span=None):
        if center is None:
            if n_params == 1:
                axis = torch.linspace(log_min, log_max, coarse_n_1d, dtype=_dtype, device=_device)
                return [(float(v),) for v in axis.detach().cpu().tolist()]
            axis = torch.linspace(log_min, log_max, coarse_n_2d, dtype=_dtype, device=_device)
            vals = [float(v) for v in axis.detach().cpu().tolist()]
            points = list(itertools.product(vals, repeat=n_params))
            if len(points) > max_evals:
                step = max(1, len(points) // max_evals)
                points = points[::step][:max_evals]
            return points

        axes = []
        for j in range(n_params):
            lo = max(log_min, float(center[j]) - span)
            hi = min(log_max, float(center[j]) + span)
            g_n = coarse_n_1d if n_params == 1 else coarse_n_2d
            axis = torch.linspace(lo, hi, g_n, dtype=_dtype, device=_device)
            axes.append([float(v) for v in axis.detach().cpu().tolist()])
        points = list(itertools.product(*axes))
        if len(points) > max_evals:
            step = max(1, len(points) // max_evals)
            points = points[::step][:max_evals]
        return points

    scored = [_score_log(p) for p in _grid_points()]
    scored = [t for t in scored if math.isfinite(t[0])]
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    best_lv = scored[0][1]
    span = max(0.1, 0.5 * (log_max - log_min))
    for _ in range(refine_passes):
        ref = [_score_log(p) for p in _grid_points(center=best_lv, span=span)]
        ref = [t for t in ref if math.isfinite(t[0])]
        if not ref:
            break
        ref.sort(key=lambda t: t[0])
        scored = ref
        best_lv = scored[0][1]
        span *= 0.35

    base = _build_init_logs(n_params, restarts, log_min, log_max, _dtype, _device)
    seed_logs = [torch.clamp(torch.zeros((n_params,), dtype=_dtype, device=_device), min=log_min, max=log_max)]
    for _, lv in scored[:topk]:
        seed_logs.append(torch.clamp(lv, min=log_min, max=log_max))

    seen = set()
    uniq = []
    for lv in seed_logs:
        key = tuple(round(float(v), 7) for v in lv.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(lv)

    for i, lv in enumerate(uniq[:restarts]):
        base[i, :] = lv
    return base


def _normalize_refine_optimizer(value):
    token = str(value or "lbfgs").strip().lower().replace("-", "_")
    aliases = {
        "": "lbfgs",
        "l_bfgs": "lbfgs",
        "lbfgs": "lbfgs",
        "grid": "grid",
        "grid_only": "grid",
        "grid_first": "grid_then_lbfgs",
        "grid_then_l_bfgs": "grid_then_lbfgs",
        "grid_then_lbfgs": "grid_then_lbfgs",
        "grid_lbfgs": "grid_then_lbfgs",
    }
    return aliases.get(token, "lbfgs")


def _score_refine_raw_log(expr_h, basis_nodes, raw_log, cfg, *, joint_data=None, x_fit=None, y_fit=None,
                          shift_slots=frozenset()):
    with torch.no_grad():
        raw = raw_log.detach()
        hparams = _raw_to_hparams(raw, shift_slots)
        if joint_data is not None:
            solved = _solve_linearized_fit_multi(expr_h, basis_nodes, joint_data, hparams, cfg, safe=True)
        else:
            solved = _solve_linearized_fit(expr_h, basis_nodes, x_fit, y_fit, hparams, cfg, safe=True)
        if solved is None:
            return None
        mse, pen, _, _, _ = solved
        safe_penalty_weight = float(cfg.get("safe_penalty_weight", 1.0e-2))
        loss = float((mse + safe_penalty_weight * pen).detach().cpu())
        if not math.isfinite(loss):
            return None
        hp = [float(v) for v in hparams.detach().cpu().tolist()]
        return loss, hp, raw.clone().detach()


def _ranked_grid_refine_seeds(expr_h, n_params, xf, yf, cfg, restarts, basis_nodes, *,
                              joint_data=None, shift_slots=frozenset()):
    if int(n_params) <= 0 or not bool(cfg.get("refine_grid_enable", True)):
        return []

    # The legacy L-BFGS seeder keeps the neutral point first.  Grid-first modes
    # need the best grid point even when the caller requested a single restart.
    grid_restarts = max(2, int(restarts))
    raw_logs = _build_grid_seed_logs(
        expr_h,
        n_params,
        xf,
        yf,
        cfg,
        grid_restarts,
        basis_nodes,
        joint_data=joint_data,
        shift_slots=shift_slots,
    )
    if raw_logs is None:
        return []

    scored = []
    seen = set()
    for raw in raw_logs:
        key = tuple(round(float(v), 7) for v in raw.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        item = _score_refine_raw_log(
            expr_h,
            basis_nodes,
            raw,
            cfg,
            joint_data=joint_data,
            x_fit=xf,
            y_fit=yf,
            shift_slots=shift_slots,
        )
        if item is None:
            continue
        scored.append(item)
    scored.sort(key=lambda t: t[0])
    return scored


def _init_logs_from_grid_rank(scored, restarts, n_params, log_min, log_max, dtype, device):
    base = _build_init_logs(n_params, restarts, log_min, log_max, dtype, device)
    if not scored:
        return base
    logs = []
    seen = set()
    for _, _, raw in scored:
        key = tuple(round(float(v), 7) for v in raw.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        logs.append(torch.clamp(raw, min=log_min, max=log_max))
    for raw in base:
        key = tuple(round(float(v), 7) for v in raw.detach().cpu().tolist())
        if key in seen:
            continue
        seen.add(key)
        logs.append(raw)
    for i, raw in enumerate(logs[:restarts]):
        base[i, :] = raw
    return base


def _flatten_add_terms(node):
    op = node[0]
    if op == "add":
        return _flatten_add_terms(node[1]) + _flatten_add_terms(node[2])
    if op == "sub":
        return _flatten_add_terms(node[1]) + [("neg", t) for t in _flatten_add_terms(node[2])]
    return [node]


def _select_linear_basis_nodes(expr_h, cfg):
    if not bool(cfg.get("linear_combo_enable", True)):
        return [expr_h]
    terms = [t for t in _flatten_add_terms(expr_h) if t[0] != "const"]
    max_terms = max(1, int(cfg.get("linear_terms_max", 6)))
    if len(terms) < 2 or len(terms) > max_terms:
        return [expr_h]
    return terms


def _eval_node_hparam_safe(node, x, hparams, cfg):
    eps = max(float(cfg.get("safe_eps", 1.0e-6)), 1.0e-12)
    exp_clip = max(float(cfg.get("safe_exp_clip", 30.0)), 1.0)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)

    op = node[0]
    if op in ("var", "const", "hparam"):
        return _eval_node_hparam(node, x, hparams), zero

    if op in UNARY_OPS:
        a, pa = _eval_node_hparam_safe(node[1], x, hparams, cfg)
        if op == "sin":
            return torch.sin(a), pa
        if op == "cos":
            return torch.cos(a), pa
        if op == "neg":
            return -a, pa
        if op == "sqr":
            return a * a, pa
        if op == "exp":
            over = torch.nn.functional.softplus(a.abs() - exp_clip)
            return torch.exp(torch.clamp(a, min=-exp_clip, max=exp_clip)), pa + over.mean()
        if op == "log":
            corr = torch.nn.functional.softplus(eps - a)
            a_safe = a + corr
            return torch.log(a_safe), pa + corr.mean()
        if op == "sqrt":
            corr = torch.nn.functional.softplus(eps - a)
            a_safe = a + corr
            return torch.sqrt(a_safe), pa + corr.mean()

    if op in BINARY_OPS:
        l, pl = _eval_node_hparam_safe(node[1], x, hparams, cfg)
        r, pr = _eval_node_hparam_safe(node[2], x, hparams, cfg)
        if op == "add":
            return l + r, pl + pr
        if op == "sub":
            return l - r, pl + pr
        if op == "mul":
            return l * r, pl + pr
        if op == "div":
            abs_r = r.abs()
            signed_floor = eps * torch.where(r >= 0, torch.ones_like(r), -torch.ones_like(r))
            denom = r + signed_floor
            out = l / denom
            near_zero = eps / (abs_r + eps)
            penalty = near_zero * near_zero
            return out, pl + pr + penalty.mean()

    # Fallback to strict op if we hit an unknown token; keep finite via penalty.
    try:
        out = _eval_node_hparam(node, x, hparams)
        if torch.isfinite(out).all():
            return out, zero
    except Exception:
        pass
    return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device), torch.tensor(1.0e6, dtype=x.dtype, device=x.device)


def _build_phi_hparam(basis_nodes, x, hparams, cfg=None, safe=False):
    cols = []
    penalty = torch.zeros((), dtype=x.dtype, device=x.device)
    for node in basis_nodes:
        if safe:
            v, p = _eval_node_hparam_safe(node, x, hparams, cfg or {})
            penalty = penalty + p
        else:
            v = _eval_node_hparam(node, x, hparams)
        if not torch.isfinite(v).all():
            return None, None
        cols.append(v)
    if not cols:
        return None, None
    return torch.cat([torch.ones_like(cols[0]), *cols], dim=1), penalty


def _materialize_linearized_candidate(expr_h, hparams, x_fit, y_fit, cfg):
    base = simplify(_materialize_hparams(expr_h, hparams))
    basis_nodes = _select_linear_basis_nodes(expr_h, cfg)
    if len(basis_nodes) < 2:
        return base

    solved = _solve_linearized_fit(expr_h, basis_nodes, x_fit, y_fit, hparams, cfg, safe=True)
    if solved is None:
        return base
    _, _, Phi, sol, _ = solved
    sol = sol.squeeze(-1)
    if not torch.isfinite(sol).all():
        return base

    coeffs = [float(v) for v in sol.detach().cpu().tolist()]
    term_coeffs = coeffs[1:]
    mat_terms = [simplify(_materialize_hparams(n, hparams)) for n in basis_nodes]

    rel = float(cfg.get("linear_prune_rel", 1.0e-10))
    rel = max(0.0, rel)
    contrib = []
    for j, c in enumerate(term_coeffs):
        col = Phi[:, j + 1]
        rms = float(torch.sqrt((col * col).mean()))
        contrib.append(abs(float(c)) * rms)
    max_contrib = max(contrib) if contrib else 0.0

    kept = []
    for j, c in enumerate(term_coeffs):
        if not math.isfinite(c):
            continue
        if abs(c) < 1.0e-14:
            continue
        if max_contrib > 0.0 and contrib[j] < rel * max_contrib:
            continue
        term = mat_terms[j]
        if abs(c - 1.0) < 1.0e-12:
            kept.append(term)
        elif abs(c + 1.0) < 1.0e-12:
            kept.append(("neg", term))
        else:
            kept.append(("mul", ("const", float(c)), term))

    out = None
    for term in kept:
        out = term if out is None else ("add", out, term)
    if out is None:
        return base
    return simplify(out)


def _refine_hparams(expr_h, n_params, x_fit, y_fit, cfg, shift_slots=frozenset()):
    t_refine = time.perf_counter()
    _diag_inc(cfg, "hparam_optimizations")
    n_params = int(n_params)
    if n_params <= 0:
        _diag_add_time(cfg, "hparam_optimization_s", time.perf_counter() - t_refine)
        return []

    subset_mode = str(cfg.get("fit_subset_mode", "hash_random"))
    subset_seed = _stable_seed_from_text(f"fit|{node_str(expr_h)}")

    # Optional joint multi-dataset refinement: optimize shared nonlinear hparams
    # while solving linear coefficients independently per dataset.
    joint = cfg.get("joint_fit_data", None)
    joint_sub = None
    if bool(cfg.get("joint_refine_enable", True)) and isinstance(joint, (list, tuple)) and len(joint) >= 2:
        n_sub_total = int(cfg.get("fit_subset", 0))
        joint_sub = _slice_fit_subset_multi(joint, n_sub_total, subset_mode, subset_seed)
        xf, yf = x_fit, y_fit
        if joint_sub:
            j0 = joint_sub[0]
            x0 = j0[1] if (isinstance(j0, (tuple, list)) and len(j0) == 3) else j0[0]
            _dtype = x0.dtype
            _device = x0.device
        else:
            _dtype = x_fit.dtype
            _device = x_fit.device
    else:
        n = int(x_fit.shape[0])
        n_sub = int(cfg.get("fit_subset", n))
        xf, yf = _slice_fit_subset(x_fit, y_fit, n_sub, subset_mode, subset_seed)
        _dtype = xf.dtype
        _device = xf.device

    restarts = max(1, int(cfg.get("num_restarts", 1)))
    log_min = float(cfg.get("init_log_min", -1.5))
    log_max = float(cfg.get("init_log_max", 1.5))
    basis_nodes = _select_linear_basis_nodes(expr_h, cfg)
    optimizer = _normalize_refine_optimizer(cfg.get("optimizer", cfg.get("refine_optimizer", "lbfgs")))
    grid_ranked = []
    if optimizer in ("grid", "grid_then_lbfgs"):
        grid_ranked = _ranked_grid_refine_seeds(
            expr_h,
            n_params,
            xf,
            yf,
            cfg,
            restarts,
            basis_nodes,
            joint_data=joint_sub,
            shift_slots=shift_slots,
        )
        if optimizer == "grid":
            _diag_inc(cfg, "grid_only_returns")
            _diag_add_time(cfg, "hparam_optimization_s", time.perf_counter() - t_refine)
            return list(grid_ranked[0][1]) if grid_ranked else None

        if grid_ranked:
            neutral_raw = torch.zeros((n_params,), dtype=_dtype, device=_device)
            neutral_item = _score_refine_raw_log(
                expr_h,
                basis_nodes,
                neutral_raw,
                cfg,
                joint_data=joint_sub,
                x_fit=xf,
                y_fit=yf,
                shift_slots=shift_slots,
            )
            base_loss = neutral_item[0] if neutral_item is not None else None
            improve_factor = max(1.0, float(cfg.get("lbfgs_escalate_improve_factor", 2.0)))
            grid_loss = float(grid_ranked[0][0])
            if base_loss is not None and float(base_loss) <= 1.0e-30:
                should_escalate = False
            else:
                should_escalate = (
                    base_loss is None
                    or (math.isfinite(base_loss) and grid_loss <= float(base_loss) / improve_factor)
                )
            if not should_escalate:
                _diag_inc(cfg, "grid_then_lbfgs_skips")
                _diag_add_time(cfg, "hparam_optimization_s", time.perf_counter() - t_refine)
                return list(grid_ranked[0][1])
            _diag_inc(cfg, "grid_then_lbfgs_escalations")
            init_logs = _init_logs_from_grid_rank(
                grid_ranked,
                restarts,
                n_params,
                log_min,
                log_max,
                _dtype,
                _device,
            )
        else:
            _diag_add_time(cfg, "hparam_optimization_s", time.perf_counter() - t_refine)
            return None
    else:
        init_logs = _build_grid_seed_logs(expr_h, n_params, xf, yf, cfg, restarts, basis_nodes,
                                          joint_data=joint_sub, shift_slots=shift_slots)
        if init_logs is None:
            init_logs = _build_init_logs(n_params, restarts, log_min, log_max, _dtype, _device)

    best_hparams = None
    best_loss = float("inf")
    steps = max(1, int(cfg.get("lbfgs_steps", 8)))
    theta_l2 = float(cfg.get("theta_l2", 1e-4))
    safe_penalty_weight = float(cfg.get("safe_penalty_weight", 1.0e-2))

    for init_log in init_logs:
        raw = init_log.clone().detach().requires_grad_(True)
        _diag_inc(cfg, "lbfgs_runs")
        opt = torch.optim.LBFGS(
            [raw],
            lr=0.8,
            max_iter=steps,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            _diag_inc(cfg, "lbfgs_closures")
            opt.zero_grad()
            hparams = _raw_to_hparams(raw, shift_slots)
            if joint_sub is not None:
                solved = _solve_linearized_fit_multi(expr_h, basis_nodes, joint_sub, hparams, cfg, safe=True)
            else:
                solved = _solve_linearized_fit(expr_h, basis_nodes, xf, yf, hparams, cfg, safe=True)
            if solved is None:
                bad = 1.0e6 + (raw * raw).sum()
                bad.backward()
                return bad
            mse, pen, _, _, _ = solved
            loss = (
                mse
                + theta_l2 * (raw * raw).sum()
                + safe_penalty_weight * pen
            )
            loss.backward()
            return loss

        try:
            with torch.enable_grad():
                opt.step(closure)
        except Exception:
            continue

        with torch.no_grad():
            hparams = _raw_to_hparams(raw, shift_slots)
            if joint_sub is not None:
                solved = _solve_linearized_fit_multi(expr_h, basis_nodes, joint_sub, hparams, cfg, safe=True)
            else:
                solved = _solve_linearized_fit(expr_h, basis_nodes, xf, yf, hparams, cfg, safe=True)
            if solved is None:
                continue
            mse, pen, _, _, _ = solved
            loss = float(mse + safe_penalty_weight * pen)
            if math.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_hparams = [float(v) for v in hparams.detach().cpu().tolist()]

    _diag_add_time(cfg, "hparam_optimization_s", time.perf_counter() - t_refine)
    return best_hparams


def _refine_budget_left(refine_state, max_refines: int) -> bool:
    if refine_state is None:
        return True
    if int(max_refines) > 0 and int(refine_state.get("trials_done", 0)) >= int(max_refines):
        return False
    depth_left = refine_state.get("depth_trials_left", None)
    if depth_left is not None and int(depth_left) <= 0:
        return False
    window_left = refine_state.get("window_trials_left", None)
    if window_left is not None and int(window_left) <= 0:
        return False
    return True


@torch.no_grad()
def score_expr(
    node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree,
    refine_enable=False, refine_cfg=None, refine_best_mse=float("inf"), refine_state=None,
    return_expr=False,
):
    cfg = refine_cfg or {}
    use_joint = bool(cfg.get("joint_score_enable", False)) and isinstance(cfg.get("joint_fit_data", None), (list, tuple)) and isinstance(cfg.get("joint_probe_data", None), (list, tuple))
    _diag_inc(cfg, "score_calls")

    def _do_score(expr):
        if use_joint:
            if bool(cfg.get("joint_terms_enable", False)):
                sc = _score_expr_base_joint_linear_terms(expr, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg)
                if sc is not None:
                    return sc
            sc = _score_expr_base_joint_affine(expr, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg)
            if sc is not None:
                return sc
        return _score_expr_base(expr, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg)

    t_base = time.perf_counter()
    base = _do_score(node)
    _diag_add_time(cfg, "base_score_s", time.perf_counter() - t_base)
    if base is None:
        return None
    # Normalize to (mse, key, z, mapping, expr)
    if len(base) == 4:
        base = (base[0], base[1], base[2], base[3], simplify(node))
    node = base[4]

    if use_joint:
        # Joint scoring paths don't run the fast sign-sweep inside _score_expr_base.
        node_neg = simplify(_negate_smart(node))
        if node_str(node_neg) != node_str(node):
            sc_neg = _do_score(node_neg)
            if sc_neg is not None:
                if len(sc_neg) == 4:
                    sc_neg = (sc_neg[0], sc_neg[1], sc_neg[2], sc_neg[3], node_neg)
                base = _pick_best_equiv_score([base, sc_neg], y_var=None)
                node = base[4]

    if not refine_enable:
        if return_expr:
            return base[0], base[1], base[2], base[3], node
        return base[0], base[1], base[2], base[3]
    _diag_inc(cfg, "refine_score_calls")
    _diag_inc_context(cfg, "refine_score_calls")

    if bool(cfg.get("score_head_only", False)):
        if return_expr:
            return base[0], base[1], base[2], base[3], node
        return base[0], base[1], base[2], base[3]

    gate_factor = float(cfg.get("gate_best_factor", 10.0))
    gate_relax = 1.0
    if refine_state is not None:
        gate_relax = max(1.0, float(refine_state.get("gate_relax_factor", 1.0)))
    gate_factor = gate_factor * gate_relax
    base_mse = float(base[0])
    gate_triggered = False
    if math.isfinite(refine_best_mse) and math.isfinite(base_mse):
        gate_triggered = base_mse > refine_best_mse * max(gate_factor, 1.0)

    max_refines = int(cfg.get("max_refines", 0))
    if not _refine_budget_left(refine_state, max_refines):
        if return_expr:
            return base[0], base[1], base[2], base[3], node
        return base[0], base[1], base[2], base[3]

    _no_shift = frozenset()
    variants = _decorate_refine_variants(
        node,
        max(1, int(cfg.get("max_variants", 4))),
        max(1, int(cfg.get("max_params", 2))),
        x_fit=x_fit,
        y_fit=y_fit,
        cfg=cfg,
    )
    if bool(cfg.get("linear_combo_enable", True)) and len(_select_linear_basis_nodes(node, cfg)) >= 2:
        variants = [(node, 0, _no_shift), *variants]
    if variants:
        seen = set()
        uniq = []
        for var_h, n_params, ss in variants:
            k = (node_str(var_h), int(n_params))
            if k in seen:
                continue
            seen.add(k)
            uniq.append((var_h, int(n_params), ss))
        variants = uniq
    _diag_inc(cfg, "variants_generated", len(variants))

    if gate_triggered:
        _diag_inc(cfg, "gate_triggered_score_calls")
        unlocked = []
        for var_h, n_params, ss in variants:
            if n_params <= 0:
                unlocked.append((var_h, n_params, ss))
                continue
            if _variant_has_gate_potential(var_h, n_params, x_fit, y_fit, cfg):
                unlocked.append((var_h, n_params, ss))
        variants = unlocked
        _diag_inc(cfg, "variants_after_gate", len(variants))

    if not variants:
        if return_expr:
            return base[0], base[1], base[2], base[3], node
        return base[0], base[1], base[2], base[3]

    best = base
    best_expr = node
    for var_h, n_params, shift_slots in variants:
        cache_key = _refine_attempt_cache_key(var_h, n_params, shift_slots, cfg, x_fit, y_fit)
        cached = _refine_cache_get(cfg, cache_key)
        h_star = None
        cand = None
        if isinstance(cached, dict):
            if cached.get("status") != "ok":
                continue
            h_star = cached.get("hparams")
            cand = cached.get("candidate")
        if cached is None and refine_state is not None:
            if not _refine_budget_left(refine_state, max_refines):
                break
            done = int(refine_state.get("trials_done", 0))
            refine_state["trials_done"] = done + 1
            if refine_state.get("depth_trials_left", None) is not None:
                refine_state["depth_trials_left"] = int(refine_state["depth_trials_left"]) - 1
            if refine_state.get("window_trials_left", None) is not None:
                refine_state["window_trials_left"] = int(refine_state["window_trials_left"]) - 1

        if cached is None:
            _diag_inc(cfg, "refinement_attempts")
            _diag_inc_context(cfg, "refinement_attempts")
            h_star = _refine_hparams(var_h, n_params, x_fit, y_fit, cfg, shift_slots=shift_slots)
        if h_star is None:
            if cached is None:
                _refine_cache_put(cfg, cache_key, {"status": "no_hparams"})
            continue
        # Scale slots must be positive; shift slots may be any finite value
        if any(
            (not math.isfinite(v)) or (i not in shift_slots and v <= 0.0)
            for i, v in enumerate(h_star)
        ):
            if cached is None:
                _refine_cache_put(cfg, cache_key, {"status": "invalid_hparams", "hparams": list(h_star)})
            continue
        if cand is None:
            cand = _materialize_linearized_candidate(var_h, h_star, x_fit, y_fit, cfg)
            if cached is None:
                _refine_cache_put(
                    cfg,
                    cache_key,
                    {"status": "ok", "hparams": list(h_star), "candidate": cand},
                )
        _diag_inc(cfg, "materialized_rescores")
        _diag_inc_context(cfg, "materialized_rescores")
        sc = _do_score(cand)
        if sc is None:
            continue
        if len(sc) == 4:
            sc = (sc[0], sc[1], sc[2], sc[3], cand)
        cand_best = sc[4]
        if sc[0] < best[0]:
            # Only print when beating the global best, not just this skeleton's base
            if bool(cfg.get("verbose", True)) and sc[0] < refine_best_mse:
                src_raw = node_str(node)
                dst_raw = node_str(cand_best)
                src_can = node_str(_mapping_equiv_root(node))
                dst_can = node_str(_mapping_equiv_root(cand_best))
                if src_can != src_raw or dst_can != dst_raw:
                    print(
                        f"  [skeleton-refine] NEW BEST {src_can} -> {dst_can}  "
                        f"[raw {src_raw} -> {dst_raw}]  "
                        f"(mse {refine_best_mse:.6g} -> {sc[0]:.6g}, "
                        f"hparams={[f'{v:.4g}' for v in h_star]})"
                    )
                else:
                    print(
                        f"  [skeleton-refine] NEW BEST {src_raw} -> {dst_raw}  "
                        f"(mse {refine_best_mse:.6g} -> {sc[0]:.6g}, "
                        f"hparams={[f'{v:.4g}' for v in h_star]})"
                    )
            best = sc
            best_expr = cand_best
            _diag_inc(cfg, "accepted_refinements")
            _diag_inc_context(cfg, "accepted_refinements")

    if return_expr:
        return best[0], best[1], best[2], best[3], best_expr
    return best[0], best[1], best[2], best[3]
