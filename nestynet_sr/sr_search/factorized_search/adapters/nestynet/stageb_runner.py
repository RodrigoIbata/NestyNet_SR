# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-B-facing factorized symbolic search runner helpers for the NestyNet adapter layer."""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    MulNode,
    PowNode,
    Scale,
    clone_ast,
    effective_arity,
    replace_atom_in_ast,
)
from nestynet_sr.sr_search.model_selection import mapping_cost as _mapping_cost
from nestynet_sr.sr_search.stageB.engine import Candidate

from ...explorer import mapping_is_structural
from .api import (
    embed_mapping_in_ast,
    promote_argument_const_scales,
    promote_const_to_scale,
    remap_var_to_exprs,
    run_explorer,
)
from .stageb_prep import _append_declared_constant_columns
from .wrapper_utils import (
    _normalize_outer_wrapper_name,
    _outer_wrapper_forward,
    _outer_wrapper_inverse_ast,
    _outer_wrapper_transformed_y_dims,
)


@dataclass(frozen=True)
class StageBEmbedContext:
    """NestyNet adapter state for embedding factorized symbolic search results back into Stage B."""

    units_mode: str = "raw"
    scale_name: str | None = None
    scale_kind: str = "fixed"
    tag_prefix: str | None = None


def _safe_name(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_")
    return "".join(out) or "x"


def _node_has_free_const(node) -> bool:
    """Return True if *node* or any descendant is a FreeConst atom."""
    if isinstance(node, AtomNode) and getattr(node, "kind", "") == "free_const":
        return True
    if isinstance(node, (AddNode, MulNode)):
        return _node_has_free_const(node.left) or _node_has_free_const(node.right)
    if isinstance(node, PowNode):
        return _node_has_free_const(node.base)
    if hasattr(node, "arg"):
        return _node_has_free_const(node.arg)
    return False


def _estimate_monomial_gain(mapping: dict[str, Any]) -> float:
    """Estimate a single multiplicative gain from a factorized symbolic search mapping."""
    kind = mapping.get("kind")
    std = float(mapping.get("std", 1.0))
    if abs(std) < 1.0e-30:
        return 1.0
    if kind == "poly":
        coeffs = mapping.get("coeffs", [0, 1])
        if len(coeffs) >= 2:
            return float(coeffs[1]) / std
    elif kind == "power":
        log_a = float(mapping.get("log_a", 0.0))
        p = float(mapping.get("b", 1.0))
        if abs(p - 1.0) < 0.3:
            return math.exp(log_a) / std
    return 1.0


def _gather_stageb_atom_teacher_data(**kwargs):
    from nestynet_sr.sr_search.candidate_builders import _gather_atom_teacher_data

    return _gather_atom_teacher_data(**kwargs)


def _build_atom_to_leaf_map(root, model):
    from nestynet_sr.sr_search.stageB.atom_mapping import build_atom_to_leaf_map

    return build_atom_to_leaf_map(root, model)


def build_stageb_probe_jobs(
    *,
    ctx: Any,
    target: Any,
    log_fn: Callable[[str], None] | None = None,
) -> list[tuple[int, str, Any, Any]]:
    """Build Stage-B probe jobs `(dataset_idx, dataset_name, loader, teacher_leaf)`."""

    def _log(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    st = ctx.state
    probe_jobs: list[tuple[int, str, Any, Any]] = []
    ds_ids = list(getattr(ctx, "dataset_ids", None) or [])
    loaders = list(getattr(ctx, "train_loader_probes", None) or [ctx.train_loader_probe])
    models = list(getattr(st, "models", None) or [])

    if models and len(models) > 1 and len(loaders) > 1:
        n_jobs = min(len(models), len(loaders))
        for di in range(n_jobs):
            try:
                atom_to_leaf_i = _build_atom_to_leaf_map(st.root, models[di])
            except Exception:
                continue
            teacher_i = atom_to_leaf_i.get(id(target))
            if teacher_i is None:
                continue
            dname = ds_ids[di] if di < len(ds_ids) else f"ds{di}"
            probe_jobs.append((di, str(dname), loaders[di], teacher_i))
    else:
        try:
            atom_to_leaf = _build_atom_to_leaf_map(st.root, st.model)
        except Exception:
            atom_to_leaf = None
        teacher = atom_to_leaf.get(id(target)) if atom_to_leaf is not None else None
        if teacher is None:
            _log(f"[Stage B]  factorized_search skip NN vars={target.var_idxs}: leaf not found")
            return []
        probe_jobs.append((0, ds_ids[0] if ds_ids else "ds0", ctx.train_loader_probe, teacher))

    if not probe_jobs:
        _log(f"[Stage B]  factorized_search skip NN vars={target.var_idxs}: no probe jobs")
    return probe_jobs


def row_raw_mse(row) -> float:
    """Return the raw MSE field used for solved checks and ranking."""
    try:
        return float(row.get("mse_raw", row.get("mse", 1e100)))
    except Exception:
        return 1e100


def has_structural_solved_result(
    rows,
    *,
    early_stop_mse: float,
) -> bool:
    """Return True if a structurally solved result is present in the rows."""
    for row in rows or []:
        mse = row_raw_mse(row)
        if mse >= float(early_stop_mse):
            continue
        if mapping_is_structural(row.get("mapping")):
            return True
    return False


def pool_stageb_results(
    results_raw,
    *,
    return_topk: int,
):
    """Pool duplicate explorer hits by `(expr, mapping-kind)` and keep the best."""
    pooled = {}
    for row in results_raw:
        key = (
            str(row.get("expr", "")),
            str((row.get("mapping") or {}).get("kind", "")),
        )
        prev = pooled.get(key)
        if prev is None or row_raw_mse(row) < row_raw_mse(prev):
            pooled[key] = row
    rows = sorted(pooled.values(), key=row_raw_mse)
    return rows[: int(return_topk)]


def _expr_ir_kwargs_from_rule(rule: Any) -> dict[str, Any]:
    from dataclasses import fields

    from nestynet_sr.sr_expr_ir.config import ExpressionIRConfig

    base = ExpressionIRConfig()
    out: dict[str, Any] = {}
    for field in fields(ExpressionIRConfig):
        if field.name == "expr_ir":
            key = "expr_ir"
        elif field.name == "canonicalize":
            key = "expr_canonicalize"
        elif field.name == "domain_mode":
            key = "expr_domain_mode"
        else:
            key = f"expr_{field.name}"
        out[key] = getattr(rule, key, getattr(base, field.name))
    return out


def _build_stageb_explorer_kwargs_from_rule(
    rule: Any,
    *,
    var_dims,
    y_dims,
    n_iter: int,
) -> dict[str, Any]:
    return {
        "n_iter": int(n_iter),
        "max_depth": int(rule.max_depth),
        "poly_degree": int(rule.poly_degree),
        "seed": int(rule.seed),
        "return_topk": int(rule.return_topk),
        "dtype": torch.float64,
        "var_dims": var_dims,
        "y_dims": y_dims,
        "brute_depth": getattr(rule, "brute_depth", None),
        "early_stop_mse": float(rule.early_stop_mse),
        "brute_max_expressions": int(rule.brute_max_expressions),
        "refine_enable": bool(rule.refine_enable),
        "refine_profile": str(getattr(rule, "refine_profile", "default")),
        "refine_mode": str(getattr(rule, "refine_mode", "slate")),
        "refine_during_brute": bool(getattr(rule, "refine_during_brute", False)),
        "refine_during_mutation": bool(getattr(rule, "refine_during_mutation", False)),
        "refine_during_controller_slate": bool(
            getattr(rule, "refine_during_controller_slate", False)
        ),
        "refine_during_slate": bool(getattr(rule, "refine_during_slate", True)),
        "refine_slate_after_brute": bool(getattr(rule, "refine_slate_after_brute", True)),
        "refine_slate_period": int(getattr(rule, "refine_slate_period", 0)),
        "refine_final_polish": bool(getattr(rule, "refine_final_polish", True)),
        "refine_slate_k": int(getattr(rule, "refine_slate_k", 16)),
        "refine_slate_diverse_k": int(getattr(rule, "refine_slate_diverse_k", 8)),
        "refine_slate_budget": int(getattr(rule, "refine_slate_budget", 32)),
        "refine_optimizer": str(getattr(rule, "refine_optimizer", "lbfgs")),
        "refine_lbfgs_escalate_improve_factor": float(
            getattr(rule, "refine_lbfgs_escalate_improve_factor", 2.0)
        ),
        "refine_lbfgs_steps": int(rule.refine_lbfgs_steps),
        "refine_fit_subset": int(rule.refine_fit_subset),
        "refine_fit_subset_mode": str(rule.refine_fit_subset_mode),
        "refine_num_restarts": int(rule.refine_num_restarts),
        "refine_max_variants": int(rule.refine_max_variants),
        "refine_max_params": int(rule.refine_max_params),
        "refine_slot_sensitivity_enable": bool(rule.refine_slot_sensitivity_enable),
        "refine_slot_sensitivity_subset": int(rule.refine_slot_sensitivity_subset),
        "refine_slot_sensitivity_delta": float(rule.refine_slot_sensitivity_delta),
        "refine_slot_sensitivity_max_paths": int(rule.refine_slot_sensitivity_max_paths),
        "refine_prune_mapping_equiv_root_slots": bool(
            getattr(rule, "refine_prune_mapping_equiv_root_slots", True)
        ),
        "refine_attempt_cache_enable": bool(getattr(rule, "refine_attempt_cache_enable", True)),
        "refine_attempt_cache_max_entries": int(
            getattr(rule, "refine_attempt_cache_max_entries", 4096)
        ),
        "refine_linear_combo_enable": bool(rule.refine_linear_combo_enable),
        "refine_linear_terms_max": int(rule.refine_linear_terms_max),
        "refine_linear_prune_rel": float(rule.refine_linear_prune_rel),
        "refine_linear_ridge": float(rule.refine_linear_ridge),
        "refine_gate_best_factor": float(rule.refine_gate_best_factor),
        "refine_gate_potential_enable": bool(rule.refine_gate_potential_enable),
        "refine_gate_potential_subset": int(rule.refine_gate_potential_subset),
        "refine_gate_potential_improve_factor": float(rule.refine_gate_potential_improve_factor),
        "refine_gate_log_min": float(rule.refine_gate_log_min),
        "refine_gate_log_max": float(rule.refine_gate_log_max),
        "refine_gate_grid_size": int(rule.refine_gate_grid_size),
        "refine_gate_max_evals": int(rule.refine_gate_max_evals),
        "refine_max_trials": int(rule.refine_max_trials),
        "refine_trials_per_brute_depth": int(rule.refine_trials_per_brute_depth),
        "refine_trials_per_mutation_window": int(rule.refine_trials_per_mutation_window),
        "refine_mutation_window": int(rule.refine_mutation_window),
        "refine_safe_eps": float(rule.refine_safe_eps),
        "refine_safe_penalty_weight": float(rule.refine_safe_penalty_weight),
        "refine_safe_exp_clip": float(rule.refine_safe_exp_clip),
        "refine_theta_l2": float(rule.refine_theta_l2),
        "refine_init_log_min": float(rule.refine_init_log_min),
        "refine_init_log_max": float(rule.refine_init_log_max),
        "refine_grid_enable": bool(rule.refine_grid_enable),
        "refine_grid_size": int(rule.refine_grid_size),
        "refine_grid_size_2d": int(rule.refine_grid_size_2d),
        "refine_grid_passes": int(rule.refine_grid_passes),
        "refine_grid_topk": int(rule.refine_grid_topk),
        "refine_grid_max_evals": int(rule.refine_grid_max_evals),
        "refine_stall_gate_relax_factor": float(rule.refine_stall_gate_relax_factor),
        "refine_stall_gate_relax_max": float(rule.refine_stall_gate_relax_max),
        "inverse_steering_enable": bool(rule.inverse_steering_enable),
        "inverse_max_paths": int(rule.inverse_max_paths),
        "inverse_topk_terms": int(rule.inverse_topk_terms),
        "inverse_shortlist_mult": int(rule.inverse_shortlist_mult),
        "inverse_min_valid_frac": float(rule.inverse_min_valid_frac),
        "inverse_min_confidence": float(rule.inverse_min_confidence),
        "inverse_safe_eps": getattr(rule, "inverse_safe_eps", None),
        "inverse_confidence_mode": str(rule.inverse_confidence_mode),
        "inverse_confidence_target_gain": float(rule.inverse_confidence_target_gain),
        "inverse_confidence_floor": float(rule.inverse_confidence_floor),
        "inverse_branch_beam_width": int(rule.inverse_branch_beam_width),
        "inverse_micro_search_enable": bool(rule.inverse_micro_search_enable),
        "inverse_micro_search_max_depth": int(rule.inverse_micro_search_max_depth),
        "inverse_micro_search_beam_width": int(rule.inverse_micro_search_beam_width),
        "inverse_micro_search_topk": int(rule.inverse_micro_search_topk),
        "inverse_micro_search_seed_terms": int(rule.inverse_micro_search_seed_terms),
        "inverse_local_score_mode": str(rule.inverse_local_score_mode),
        "inverse_spec_enable": bool(rule.inverse_spec_enable),
        "inverse_spec_enum_max_depth": int(rule.inverse_spec_enum_max_depth),
        "inverse_spec_enum_max_trees": int(rule.inverse_spec_enum_max_trees),
        "inverse_spec_preview_topk": int(rule.inverse_spec_preview_topk),
        "inverse_spec_local_score_mode": str(rule.inverse_spec_local_score_mode),
        "inverse_spec_include_legacy_seed": bool(rule.inverse_spec_include_legacy_seed),
        "inverse_spec_complexity_penalty": float(rule.inverse_spec_complexity_penalty),
        "inverse_spec_recursive_enable": bool(rule.inverse_spec_recursive_enable),
        "inverse_spec_recursive_max_depth": int(rule.inverse_spec_recursive_max_depth),
        "inverse_spec_recursive_trigger_rel_mse": float(rule.inverse_spec_recursive_trigger_rel_mse),
        "inverse_spec_recursive_seed_cap": int(rule.inverse_spec_recursive_seed_cap),
        "inverse_spec_recursive_branch_topk": int(rule.inverse_spec_recursive_branch_topk),
        "inverse_spec_recursive_child_topk": int(rule.inverse_spec_recursive_child_topk),
        "inverse_spec_max_subtree_depth": getattr(rule, "inverse_spec_max_subtree_depth", None),
        "inverse_spec_fit_cap": int(getattr(rule, "inverse_spec_fit_cap", 96)),
        "inverse_spec_probe_cap": int(getattr(rule, "inverse_spec_probe_cap", 192)),
        "inverse_spec_exact_budget": int(getattr(rule, "inverse_spec_exact_budget", 4)),
        "inverse_target_mode": str(rule.inverse_target_mode),
        "inverse_full_mapping_penalty": float(rule.inverse_full_mapping_penalty),
        "inverse_exact_simple_target_bonus": float(rule.inverse_exact_simple_target_bonus),
        "inverse_additive_descend_penalty": float(rule.inverse_additive_descend_penalty),
        "inverse_nonadditive_leaf_penalty": float(rule.inverse_nonadditive_leaf_penalty),
        "inverse_exact_path_eta": float(rule.inverse_exact_path_eta),
        "inverse_exact_transport_min_lin_rel": float(rule.inverse_exact_transport_min_lin_rel),
        "inverse_gate_enable": bool(rule.inverse_gate_enable),
        "inverse_gate_warmup": int(rule.inverse_gate_warmup),
        "inverse_gate_best_factor": float(rule.inverse_gate_best_factor),
        "inverse_gate_min_residual_basins": int(rule.inverse_gate_min_residual_basins),
        "inverse_gate_min_depth": int(rule.inverse_gate_min_depth),
        "inverse_gate_min_size": int(rule.inverse_gate_min_size),
        "inverse_gate_max_paths": int(rule.inverse_gate_max_paths),
        "inverse_gate_min_structural_score": float(rule.inverse_gate_min_structural_score),
        "inverse_gate_min_weighted_rel_gain": float(rule.inverse_gate_min_weighted_rel_gain),
        "inverse_gate_structural_bias": float(rule.inverse_gate_structural_bias),
        "inverse_periodic_min_valid_scale": float(rule.inverse_periodic_min_valid_scale),
        "inverse_periodic_min_confidence_scale": float(rule.inverse_periodic_min_confidence_scale),
        "inverse_periodic_path_penalty": float(rule.inverse_periodic_path_penalty),
        "inverse_nonperiodic_muldiv_bonus": float(rule.inverse_nonperiodic_muldiv_bonus),
        "inverse_nonperiodic_explogsqrt_bonus": float(rule.inverse_nonperiodic_explogsqrt_bonus),
        "inverse_branch_ambiguity_penalty": float(rule.inverse_branch_ambiguity_penalty),
        "inverse_transport_min_lin_rel": float(rule.inverse_transport_min_lin_rel),
        "inverse_transport_min_effective_n": float(rule.inverse_transport_min_effective_n),
        "repair_controller_enable": bool(rule.repair_controller_enable),
        "repair_controller_min_score": float(rule.repair_controller_min_score),
        "repair_controller_steps": int(rule.repair_controller_steps),
        "repair_controller_ancestor_hops": int(rule.repair_controller_ancestor_hops),
        "repair_controller_min_step_rel_improve": float(rule.repair_controller_min_step_rel_improve),
        "repair_controller_adaptive": bool(rule.repair_controller_adaptive),
        "repair_controller_adapt_quantile": float(rule.repair_controller_adapt_quantile),
        "repair_controller_adapt_window": int(rule.repair_controller_adapt_window),
        "repair_controller_adapt_min_samples": int(rule.repair_controller_adapt_min_samples),
        "repair_controller_min_concentration": float(rule.repair_controller_min_concentration),
        "repair_controller_potential_weight": float(rule.repair_controller_potential_weight),
        "repair_controller_concentration_weight": float(rule.repair_controller_concentration_weight),
        "repair_controller_contrast_weight": float(rule.repair_controller_contrast_weight),
        "repair_controller_cost_weight": float(rule.repair_controller_cost_weight),
        "repair_controller_stagnation_weight": float(rule.repair_controller_stagnation_weight),
        "repair_controller_frontier_topk": int(rule.repair_controller_frontier_topk),
        "repair_controller_stagnation_visits": int(rule.repair_controller_stagnation_visits),
        "repair_controller_focus_prob": float(rule.repair_controller_focus_prob),
        "repair_controller_parent_max_repeats": int(rule.repair_controller_parent_max_repeats),
        "repair_controller_parent_min_eval_gap": int(rule.repair_controller_parent_min_eval_gap),
        "repair_controller_parent_reset_rel_improve": float(rule.repair_controller_parent_reset_rel_improve),
        "repair_controller_critic_enable": bool(rule.repair_controller_critic_enable),
        "repair_controller_critic_path": str(rule.repair_controller_critic_path),
        "repair_controller_critic_blend": float(rule.repair_controller_critic_blend),
        "repair_controller_critic_mode": str(rule.repair_controller_critic_mode),
        "repair_opportunity_controller_enable": bool(rule.repair_opportunity_controller_enable),
        "repair_opportunity_controller_path": str(rule.repair_opportunity_controller_path),
        "boost_enable": bool(rule.boost_enable),
        "boost_max_terms": int(rule.boost_max_terms),
        "boost_topk_try": int(rule.boost_topk_try),
        "boost_min_rel_improve": float(rule.boost_min_rel_improve),
        "boost_selection_split": str(rule.boost_selection_split),
        "boost_ridge": getattr(rule, "boost_ridge", None),
        "boost_include_parent": bool(rule.boost_include_parent),
        "boost_from_scratch_prob": float(rule.boost_from_scratch_prob),
        "boost_prune_rel": float(rule.boost_prune_rel),
        "boost_safe_eval": bool(rule.boost_safe_eval),
        "boost_harvest_enable": bool(rule.boost_harvest_enable),
        "boost_harvest_every": int(rule.boost_harvest_every),
        "boost_harvest_topk_residual_basins": int(rule.boost_harvest_topk_residual_basins),
        "boost_harvest_elites_per_residual_basin": int(rule.boost_harvest_elites_per_residual_basin),
        "boost_pool_extra_max": int(rule.boost_pool_extra_max),
        "boost_subtree_depth_max": int(rule.boost_subtree_depth_max),
        "boost_subtree_size_max": int(rule.boost_subtree_size_max),
        "boost_gate_enable": bool(rule.boost_gate_enable),
        "boost_gate_warmup": int(rule.boost_gate_warmup),
        "boost_gate_best_factor": float(rule.boost_gate_best_factor),
        "boost_gate_gain_frac": float(rule.boost_gate_gain_frac),
        "boost_gate_peak_ratio": float(rule.boost_gate_peak_ratio),
        "boost_gate_min_valid": int(rule.boost_gate_min_valid),
        "boost_gate_min_residual_basins": int(rule.boost_gate_min_residual_basins),
        "boost_gate_adaptive": bool(rule.boost_gate_adaptive),
        "boost_gate_adapt_quantile": float(rule.boost_gate_adapt_quantile),
        "boost_gate_adapt_window": int(rule.boost_gate_adapt_window),
        "boost_gate_adapt_min_samples": int(rule.boost_gate_adapt_min_samples),
        "boost_gate_adapt_mix": float(rule.boost_gate_adapt_mix),
        "boost_gate_gain_frac_floor": float(rule.boost_gate_gain_frac_floor),
        "boost_gate_gain_frac_cap": float(rule.boost_gate_gain_frac_cap),
        "score_head_enable": bool(rule.score_head_enable),
        "score_head_vars_enable": bool(rule.score_head_vars_enable),
        "score_head_omp_enable": bool(rule.score_head_omp_enable),
        "score_head_omp_max_terms": int(rule.score_head_omp_max_terms),
        "score_head_omp_topk_try": int(rule.score_head_omp_topk_try),
        "score_head_ridge": getattr(rule, "score_head_ridge", None),
        "score_head_min_rel_improve": float(rule.score_head_min_rel_improve),
        **_expr_ir_kwargs_from_rule(rule),
    }


def prepare_stageb_embed_context(
    *,
    root,
    target,
    units_spec,
    enforce_units: bool,
) -> tuple[StageBEmbedContext, Any]:
    """Build the embedding context used when mapping factorized symbolic search rows back to Stage B."""

    embed_ctx = StageBEmbedContext()
    updated_units_spec = units_spec

    if bool(enforce_units) and units_spec is not None:
        try:
            from nestynet_sr.sr_core.units import compute_node_domains, infer_atom_output_dim, is_dimless

            target_dim = infer_atom_output_dim(root, target, units_spec)
            if target_dim is None:
                span_spec = dc_replace(units_spec, nn_semantics="span")
                domains = compute_node_domains(root, span_spec)
                if domains is not None:
                    dom = domains.get(id(target))
                    if dom is not None and dom.is_pinned():
                        target_dim = dom.offset

            if target_dim is not None and (not is_dimless(target_dim)):
                base = getattr(target, "tag", None)
                if base is None:
                    base = f"nn_{'_'.join(str(int(i)) for i in getattr(target, 'var_idxs', ()))}"
                safe = _safe_name(str(base))
                scale_name = f"factorized_search_S__{safe}"
                new_fixed = dict(getattr(units_spec, "fixed_const_dims", {}) or {})
                new_fixed[scale_name] = target_dim
                new_fixed[f"{scale_name}__floor"] = target_dim
                updated_units_spec = dc_replace(
                    units_spec,
                    fixed_const_dims=new_fixed,
                )
                embed_ctx = StageBEmbedContext(
                    units_mode="scaled",
                    scale_name=scale_name,
                    scale_kind="fixed",
                    tag_prefix=scale_name,
                )
                return embed_ctx, updated_units_spec
        except Exception:
            pass
    else:
        base = str(getattr(target, "tag", None) or "nn")
        embed_ctx = StageBEmbedContext(
            units_mode="raw",
            scale_name=None,
            scale_kind="fixed",
            tag_prefix=f"factorized_search__{_safe_name(base)}",
        )
        return embed_ctx, updated_units_spec

    return embed_ctx, updated_units_spec


def build_stageb_main_candidates(
    *,
    root,
    target,
    results: Sequence[dict[str, Any]],
    input_exprs: Sequence[Any],
    embed_ctx: StageBEmbedContext,
    refine_enable: bool,
    refine_stageb_promote_consts: bool,
    log_fn: Callable[[str], None] | None = None,
) -> list[Candidate]:
    """Convert pooled explorer rows into standard Stage B candidates."""

    def _log(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    candidates: list[Candidate] = []
    for ri, row in enumerate(results):
        nn_ast = row.get("nestynet_ast")
        if nn_ast is None:
            continue

        mapping = row.get("mapping")
        if mapping is None:
            continue

        base_tag = embed_ctx.tag_prefix
        if base_tag is None:
            base = str(getattr(target, "tag", None) or "nn")
            base_tag = f"factorized_search__{_safe_name(base)}"
        promote_prefix = f"{base_tag}__refine_{ri}"

        try:
            mono_tag = (
                f"{embed_ctx.tag_prefix}__mono_{ri}"
                if embed_ctx.tag_prefix
                else f"factorized_search__mono_{ri}"
            )
            mono_nn_ast = promote_const_to_scale(
                clone_ast(nn_ast),
                tag_prefix=mono_tag,
            )
            remapped = remap_var_to_exprs(mono_nn_ast, list(input_exprs))
            gain = _estimate_monomial_gain(mapping)
            mono_repl = MulNode(
                Scale(mono_tag, tag=mono_tag, init=gain),
                clone_ast(remapped),
            )
            if bool(refine_enable):
                mono_repl = promote_argument_const_scales(
                    mono_repl,
                    tag_prefix=f"{promote_prefix}__mono",
                )
            mono_root = replace_atom_in_ast(root, target, mono_repl)
            candidates.append(
                Candidate(
                    label=f"factorized_search_mono({row['expr']})",
                    root=mono_root,
                    meta={
                        "factorized_mse": row_raw_mse(row),
                        "factorized_mse_raw": row_raw_mse(row),
                        "factorized_mse_eff": float(row.get("mse_eff", row.get("mse", 1e100))),
                        "factorized_expr": row["expr"],
                        "factorized_mapping_kind": "monomial",
                        "factorized_mapping": {"kind": "monomial"},
                        "factorized_search_source_mapping": mapping,
                        "factorized_mapping_cost": 0.5,
                        "mapping_cost": 0.5,
                        "factorized_probe_dataset": row.get("_probe_dataset"),
                    },
                )
            )
        except Exception:
            pass

        try:
            embed_kwargs = dict(
                units_mode=embed_ctx.units_mode,
                scale_name=embed_ctx.scale_name,
                scale_kind=embed_ctx.scale_kind,
                trainable_dimless=True,
                tag_prefix=embed_ctx.tag_prefix,
                z_affine=True,
                z_alpha_init=1.0,
                z_beta_init=None,
                sin_arg_mode="wu",
            )
            replacement = embed_mapping_in_ast(
                nn_ast,
                mapping,
                list(input_exprs),
                z_train_alpha=True,
                **embed_kwargs,
            )
            if replacement is not None and _node_has_free_const(replacement):
                replacement = embed_mapping_in_ast(
                    nn_ast,
                    mapping,
                    list(input_exprs),
                    z_train_alpha=False,
                    **embed_kwargs,
                )
            if replacement is not None and bool(refine_enable):
                replacement = promote_argument_const_scales(
                    replacement,
                    tag_prefix=f"{promote_prefix}__full",
                )
                if bool(refine_stageb_promote_consts):
                    replacement = promote_const_to_scale(
                        replacement,
                        tag_prefix=f"{promote_prefix}__full_const",
                    )
        except Exception as exc:
            _log(f"[Stage B]  factorized_search embed failed: {exc}")
            continue
        if replacement is None:
            _log("[Stage B]  factorized_search embed returned None")
            continue

        new_root = replace_atom_in_ast(root, target, replacement)
        candidates.append(
            Candidate(
                label=f"factorized_search({row['expr']})",
                root=new_root,
                meta={
                    "factorized_mse": row_raw_mse(row),
                    "factorized_mse_raw": row_raw_mse(row),
                    "factorized_mse_eff": float(row.get("mse_eff", row.get("mse", 1e100))),
                    "factorized_expr": row["expr"],
                    "factorized_mapping_kind": mapping.get("kind", None),
                    "factorized_mapping": mapping,
                    "factorized_mapping_cost": float(_mapping_cost(mapping)),
                    "mapping_cost": float(_mapping_cost(mapping)),
                    "factorized_probe_dataset": row.get("_probe_dataset"),
                },
            )
        )

    return candidates


def run_stageb_wrapper_pass(
    *,
    rule: Any,
    root,
    target,
    probe_jobs: Sequence[tuple[int, str, Any, Any]],
    declared_consts: Sequence[dict[str, Any]],
    var_dims,
    y_dims,
    input_exprs: Sequence[Any],
    embed_ctx: StageBEmbedContext,
    main_structurally_solved: bool,
    enforce_units: bool,
    device: torch.device,
    dtype: torch.dtype,
    log_fn: Callable[[str], None] | None = None,
) -> list[Candidate]:
    """Run the reduced-budget outer-wrapper factorized symbolic search pass for a Stage-B target."""

    def _log(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    if (not bool(rule.outer_wrapper_enable)) or bool(main_structurally_solved):
        return []

    try:
        if int(effective_arity(target)) > int(rule.outer_wrapper_max_arity):
            _log(
                f"[Stage B]  factorized_search wrapper skip NN vars={target.var_idxs}: "
                f"arity>{rule.outer_wrapper_max_arity}"
            )
            return []

        if len(probe_jobs) != 1:
            _log(
                f"[Stage B]  factorized_search wrapper skip NN vars={target.var_idxs}: "
                f"requires single probe dataset (got {len(probe_jobs)})"
            )
            return []

        y_dims_base = tuple(float(v) for v in y_dims) if y_dims is not None else None
        if bool(enforce_units) and (y_dims_base is None):
            _log(
                f"[Stage B]  factorized_search wrapper skip NN vars={target.var_idxs}: "
                "target dims unavailable under unit enforcement"
            )
            return []

        from nestynet_sr.sr_search.fitting_utils import (
            _nonlinear_substitution_screen,
            _rational_probe_nd,
        )

        _, dname, loader_i, teacher_i = probe_jobs[0]
        data_w = _gather_stageb_atom_teacher_data(
            train_loader=loader_i,
            atom=target,
            teacher=teacher_i,
            device=device,
            dtype=dtype,
            max_points=max(
                int(rule.outer_wrapper_probe_max_points),
                int(rule.outer_wrapper_min_points),
            ),
        )
        if data_w is None:
            _log(
                f"[Stage B]  factorized_search wrapper skip NN vars={target.var_idxs}: "
                "teacher data gather failed"
            )
            return []

        Xw_all, Yw_all = data_w
        transform_specs: list[dict[str, Any]] = []
        seen_t: set[str] = set()
        for raw_t in rule.outer_wrapper_transforms:
            tname = _normalize_outer_wrapper_name(raw_t)
            if tname is None or tname in seen_t:
                continue
            seen_t.add(tname)

            y_dims_w = y_dims
            if bool(enforce_units):
                ok_dims, y_dims_w, dim_reason = _outer_wrapper_transformed_y_dims(
                    y_dims_base,
                    tname,
                )
                if not ok_dims:
                    _log(
                        f"[Stage B]  factorized_search wrapper skip {tname} on NN vars={target.var_idxs}: "
                        f"unit gate ({dim_reason})"
                    )
                    continue

            mask, t_all, sign_hint, _ = _outer_wrapper_forward(
                Yw_all,
                tname,
                eps=1.0e-12,
                exp_abs_cap=20.0,
                square_sign_consistency=0.98,
            )
            dom_frac = float(mask.double().mean().item()) if mask.numel() else 0.0
            n_ok = int(mask.sum().item())
            if (
                dom_frac < float(rule.outer_wrapper_min_domain_frac)
                or n_ok < int(rule.outer_wrapper_min_points)
            ):
                continue

            Xm = Xw_all[mask]
            tm = t_all[mask].view(-1)
            finite = torch.isfinite(tm)
            if not bool(finite.any()):
                continue
            Xm = Xm[finite]
            tm = tm[finite]
            if int(tm.numel()) < int(rule.outer_wrapper_min_points):
                continue

            rat_err = float(
                _rational_probe_nd(
                    Xm,
                    tm,
                    deg_num=2,
                    deg_den=2,
                    min_points=max(128, int(rule.outer_wrapper_min_points) // 2),
                    max_points=int(rule.outer_wrapper_probe_max_points),
                    dtype=torch.float64,
                    filter_outliers=True,
                    error_metric="median_rel",
                )
            )
            nls_target_dims = None
            nls_input_dims = None
            if bool(enforce_units):
                try:
                    nls_target_dims = y_dims_w
                    nls_input_dims = tuple(var_dims[: int(Xm.shape[1])])
                    if len(nls_input_dims) != int(Xm.shape[1]):
                        nls_input_dims = None
                except Exception:
                    nls_target_dims = None
                    nls_input_dims = None
            if bool(enforce_units) and (
                nls_target_dims is None or nls_input_dims is None
            ):
                nls_hits = []
            else:
                nls_hits = _nonlinear_substitution_screen(
                    Xm,
                    tm,
                    teacher=None,
                    threshold=float(rule.outer_wrapper_screen_nls_err_max),
                    max_points=int(rule.outer_wrapper_probe_max_points),
                    min_points=max(128, int(rule.outer_wrapper_min_points) // 2),
                    target_dim=nls_target_dims,
                    input_dims=nls_input_dims,
                )
            nls_err = float(nls_hits[0]["error"]) if nls_hits else float("inf")
            cheap_solved = (
                (math.isfinite(rat_err) and rat_err <= float(rule.outer_wrapper_screen_rational_err_max))
                or (math.isfinite(nls_err) and nls_err <= float(rule.outer_wrapper_screen_nls_err_max))
            )
            best_screen = min(
                rat_err if math.isfinite(rat_err) else float("inf"),
                nls_err if math.isfinite(nls_err) else float("inf"),
            )
            if not math.isfinite(best_screen):
                best_screen = 1e6
            score = dom_frac * max(0.0, -math.log10(max(best_screen, 1.0e-12)))

            if cheap_solved:
                _log(
                    f"[Stage B]  factorized_search wrapper prioritize {tname} on NN vars={target.var_idxs}: "
                    f"cheap screen strong (rat≈{rat_err:.3g}, nls≈{nls_err:.3g})"
                )
                score += 1.0

            transform_specs.append(
                {
                    "name": tname,
                    "X": Xm,
                    "Y": tm,
                    "sign_hint": float(sign_hint),
                    "domain_ok_frac": dom_frac,
                    "rat_err": rat_err,
                    "nls_err": nls_err,
                    "cheap_solved": bool(cheap_solved),
                    "y_dims": y_dims_w,
                    "score": float(score),
                }
            )

        transform_specs.sort(
            key=lambda d: (
                1 if bool(d.get("cheap_solved", False)) else 0,
                float(d.get("score", 0.0)),
            ),
            reverse=True,
        )
        transform_specs = transform_specs[: max(0, int(rule.outer_wrapper_topk))]

        wrapper_candidates: list[Candidate] = []
        n_consts = len(tuple(declared_consts or ()))
        for wi, spec_w in enumerate(transform_specs):
            Xw = spec_w["X"]
            Yw = spec_w["Y"]
            Nw = int(Yw.shape[0])
            if Nw < int(rule.outer_wrapper_min_points):
                continue

            g_split = torch.Generator(device="cpu").manual_seed(
                int(rule.seed) + 12007 * int(wi)
            )
            perm = torch.randperm(Nw, generator=g_split)
            n_fit_w = min(int(rule.n_fit), Nw)
            n_probe_w = min(int(rule.n_probe), Nw)
            idx_fit_w = perm[:n_fit_w]
            idx_probe_w = perm[n_fit_w:n_fit_w + n_probe_w]
            if idx_probe_w.numel() < n_probe_w:
                idx_probe_w = perm[:n_probe_w]

            x_fit_w = Xw[idx_fit_w]
            y_fit_w = Yw[idx_fit_w]
            x_probe_w = Xw[idx_probe_w]
            y_probe_w = Yw[idx_probe_w]

            if n_consts > 0:
                x_fit_w = _append_declared_constant_columns(x_fit_w, list(declared_consts))
                x_probe_w = _append_declared_constant_columns(x_probe_w, list(declared_consts))

            n_iter_base_w = max(
                200,
                int(float(rule.n_iter) * float(rule.outer_wrapper_iter_scale)),
            )
            if bool(spec_w.get("cheap_solved", False)):
                n_iter_w = max(100, min(200, n_iter_base_w))
            else:
                n_iter_w = n_iter_base_w
            n_seeds_w = max(1, int(rule.outer_wrapper_n_seeds))
            rows_w_raw: list[dict[str, Any]] = []

            for si in range(n_seeds_w):
                seed_search_w = int(rule.seed) + 100003 * int(wi) + int(si)
                try:
                    res_w = run_explorer(
                        x_fit_data=x_fit_w,
                        y_fit_data=y_fit_w,
                        x_probe_data=x_probe_w,
                        y_probe_data=y_probe_w,
                        n_iter=n_iter_w,
                        max_depth=rule.max_depth,
                        poly_degree=rule.poly_degree,
                        seed=int(rule.seed),
                        return_topk=max(1, int(rule.outer_wrapper_return_topk)),
                        dtype=torch.float64,
                        var_dims=var_dims,
                        y_dims=spec_w.get("y_dims", y_dims),
                        brute_depth=rule.brute_depth,
                        early_stop_mse=rule.early_stop_mse,
                        brute_max_expressions=rule.brute_max_expressions,
                        refine_enable=False,
                        seed_search=seed_search_w,
                        verbose=(si == 0 and wi == 0),
                        simplify_skeletons=False,
                    )
                except Exception as exc:
                    _log(
                        f"[Stage B]  factorized_search wrapper error on NN vars={target.var_idxs} "
                        f"T={spec_w['name']} seed={seed_search_w}: {exc}"
                    )
                    continue
                if res_w:
                    rows_w_raw.extend(res_w)

            if not rows_w_raw:
                continue

            rows_w = pool_stageb_results(
                rows_w_raw,
                return_topk=max(1, int(rule.outer_wrapper_return_topk)),
            )

            for ri_w, rw in enumerate(rows_w):
                nn_ast_w = rw.get("nestynet_ast")
                mapping_w = rw.get("mapping")
                if nn_ast_w is None or mapping_w is None:
                    continue

                base_tag_w = embed_ctx.tag_prefix
                if base_tag_w is None:
                    base = str(getattr(target, "tag", None) or "nn")
                    base_tag_w = f"factorized_search__{_safe_name(base)}"
                promote_prefix_w = f"{base_tag_w}__wrap_{spec_w['name']}_{wi}_{ri_w}"

                try:
                    embed_kwargs_w = dict(
                        units_mode=embed_ctx.units_mode,
                        scale_name=embed_ctx.scale_name,
                        scale_kind=embed_ctx.scale_kind,
                        trainable_dimless=True,
                        tag_prefix=f"{base_tag_w}__wrap_{spec_w['name']}",
                        z_affine=True,
                        z_alpha_init=1.0,
                        z_beta_init=None,
                        sin_arg_mode="wu",
                    )
                    replacement_w = embed_mapping_in_ast(
                        nn_ast_w,
                        mapping_w,
                        list(input_exprs),
                        z_train_alpha=True,
                        **embed_kwargs_w,
                    )
                    if replacement_w is not None and _node_has_free_const(replacement_w):
                        replacement_w = embed_mapping_in_ast(
                            nn_ast_w,
                            mapping_w,
                            list(input_exprs),
                            z_train_alpha=False,
                            **embed_kwargs_w,
                        )
                    if replacement_w is not None and bool(rule.refine_enable):
                        replacement_w = promote_argument_const_scales(
                            replacement_w,
                            tag_prefix=f"{promote_prefix_w}__full",
                        )
                        if bool(rule.refine_stageb_promote_consts):
                            replacement_w = promote_const_to_scale(
                                replacement_w,
                                tag_prefix=f"{promote_prefix_w}__const",
                            )
                except Exception as exc:
                    _log(
                        f"[Stage B]  factorized_search wrapper embed failed "
                        f"(T={spec_w['name']}): {exc}"
                    )
                    continue
                if replacement_w is None:
                    continue

                wrapped_sub = _outer_wrapper_inverse_ast(
                    replacement_w,
                    spec_w["name"],
                    sign_hint=float(spec_w.get("sign_hint", 1.0)),
                )
                if wrapped_sub is None:
                    continue

                new_root_w = replace_atom_in_ast(root, target, wrapped_sub)
                if new_root_w is None:
                    continue

                map_cost_w = float(_mapping_cost(mapping_w)) + 1.0
                wrapper_candidates.append(
                    Candidate(
                        label=f"factorized_wrap_{spec_w['name']}({rw['expr']})",
                        root=new_root_w,
                        meta={
                            "factorized_mse": row_raw_mse(rw),
                            "factorized_mse_raw": row_raw_mse(rw),
                            "factorized_mse_eff": float(rw.get("mse_eff", rw.get("mse", 1e100))),
                            "factorized_expr": rw["expr"],
                            "factorized_mapping_kind": mapping_w.get("kind", None),
                            "factorized_mapping": mapping_w,
                            "factorized_mapping_cost": map_cost_w,
                            "mapping_cost": map_cost_w,
                            "factorized_probe_dataset": str(dname),
                            "factorized_search_wrapper_transform": spec_w["name"],
                            "factorized_search_wrapper_domain_ok_frac": float(
                                spec_w.get("domain_ok_frac", 0.0)
                            ),
                            "factorized_search_wrapper_screen_cheap_solved": bool(
                                spec_w.get("cheap_solved", False)
                            ),
                            "factorized_search_wrapper_screen_rat_err": float(
                                spec_w.get("rat_err", float("inf"))
                            ),
                            "factorized_search_wrapper_screen_nls_err": float(
                                spec_w.get("nls_err", float("inf"))
                            ),
                        },
                    )
                )

        if wrapper_candidates:
            _log(
                f"[Stage B]  factorized_search wrapper on NN vars={target.var_idxs}: "
                f"{len(wrapper_candidates)} candidate(s) from "
                f"{len(transform_specs)} transform(s)"
            )

        return wrapper_candidates
    except Exception as exc:
        _log(
            f"[Stage B]  factorized_search wrapper pass failed on NN vars={target.var_idxs}: {exc}"
        )
        return []


def run_stageb_explorer_jobs(
    *,
    rule: Any,
    target: Any,
    probe_jobs: Sequence[tuple[int, str, Any, Any]],
    declared_consts: Sequence[dict[str, Any]],
    var_dims,
    y_dims,
    device: torch.device,
    dtype: torch.dtype,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run factorized symbolic search on one or more Stage-B probe jobs and return raw rows."""

    def _log(message: str) -> None:
        if callable(log_fn):
            log_fn(message)

    n_seeds = max(1, int(rule.n_seeds))
    if bool(rule.split_iter_across_seeds) and n_seeds > 1:
        iters_each = max(1, int(rule.n_iter) // n_seeds)
    else:
        iters_each = int(rule.n_iter)

    base_kwargs = _build_stageb_explorer_kwargs_from_rule(
        rule,
        var_dims=var_dims,
        y_dims=y_dims,
        n_iter=iters_each,
    )

    results_raw: list[dict[str, Any]] = []
    n_consts = len(tuple(declared_consts or ()))
    target_var_idxs = getattr(target, "var_idxs", ())
    joint_mode = bool(getattr(rule, "refine_joint_score_enable", False)) and (len(probe_jobs) > 1)

    if joint_mode:
        joint_fit = []
        joint_probe = []
        x_fit_list = []
        y_fit_list = []
        x_probe_list = []
        y_probe_list = []
        for di, dname, loader_i, teacher_i in probe_jobs:
            data = _gather_stageb_atom_teacher_data(
                train_loader=loader_i,
                atom=target,
                teacher=teacher_i,
                device=device,
                dtype=dtype,
                max_points=5000,
            )
            if data is None:
                continue

            x_atom, f_atom = data
            try:
                n_rows = int(x_atom.shape[0])
                g_split = torch.Generator(device="cpu").manual_seed(int(rule.seed) + 10007 * int(di))
                perm = torch.randperm(n_rows, generator=g_split)
                n_fit = min(int(rule.n_fit), n_rows)
                n_probe = min(int(rule.n_probe), n_rows)
                idx_fit = perm[:n_fit]
                idx_probe = perm[n_fit:n_fit + n_probe]
                if idx_probe.numel() < n_probe:
                    idx_probe = perm[:n_probe]
                x_fit_data = x_atom[idx_fit]
                y_fit_data = f_atom[idx_fit]
                x_probe_data = x_atom[idx_probe]
                y_probe_data = f_atom[idx_probe]
            except Exception:
                x_fit_data, y_fit_data = x_atom, f_atom
                x_probe_data, y_probe_data = x_atom, f_atom

            if n_consts > 0:
                x_fit_data = _append_declared_constant_columns(x_fit_data, list(declared_consts))
                x_probe_data = _append_declared_constant_columns(x_probe_data, list(declared_consts))

            joint_fit.append((str(dname), x_fit_data, y_fit_data))
            joint_probe.append((str(dname), x_probe_data, y_probe_data))
            x_fit_list.append(x_fit_data)
            y_fit_list.append(y_fit_data)
            x_probe_list.append(x_probe_data)
            y_probe_list.append(y_probe_data)

        if len(joint_fit) >= 2:
            joint_kwargs = dict(
                base_kwargs,
                x_fit_data=torch.cat(x_fit_list, dim=0),
                y_fit_data=torch.cat(y_fit_list, dim=0),
                x_probe_data=torch.cat(x_probe_list, dim=0),
                y_probe_data=torch.cat(y_probe_list, dim=0),
                refine_joint_fit_data=joint_fit,
                refine_joint_probe_data=joint_probe,
                refine_joint_weight_mode=str(rule.refine_joint_weight_mode),
                refine_joint_enable=bool(rule.refine_joint_enable),
                refine_joint_score_enable=True,
                refine_joint_terms_enable=bool(rule.refine_joint_terms_enable),
                simplify_skeletons=False,
            )

            stop_event = threading.Event()

            def _run_joint_seed(si: int):
                seed_search = int(rule.seed) + si
                return si, seed_search, run_explorer(
                    seed_search=seed_search,
                    verbose=(si == 0),
                    stop_event=stop_event,
                    **joint_kwargs,
                )

            with ThreadPoolExecutor(max_workers=n_seeds) as pool:
                futures = {pool.submit(_run_joint_seed, si): si for si in range(n_seeds)}
                for fut in as_completed(futures):
                    si = futures[fut]
                    try:
                        _, _, rows = fut.result()
                    except Exception as exc:
                        _log(
                            f"[Stage B]  factorized_search error on NN vars={target_var_idxs} "
                            f"dataset=JOINT seed={int(rule.seed) + si}: {exc}"
                        )
                        continue
                    if rows:
                        for row in rows:
                            out = dict(row)
                            out["_probe_dataset_idx"] = -1
                            out["_probe_dataset"] = "JOINT"
                            results_raw.append(out)
                        if has_structural_solved_result(rows, early_stop_mse=float(rule.early_stop_mse)):
                            stop_event.set()
        else:
            joint_mode = False

    if not joint_mode:
        for job_idx, (di, dname, loader_i, teacher_i) in enumerate(probe_jobs):
            data = _gather_stageb_atom_teacher_data(
                train_loader=loader_i,
                atom=target,
                teacher=teacher_i,
                device=device,
                dtype=dtype,
                max_points=5000,
            )
            if data is None:
                continue

            x_atom, f_atom = data
            try:
                n_rows = int(x_atom.shape[0])
                g_split = torch.Generator(device="cpu").manual_seed(int(rule.seed) + 10007 * int(di))
                perm = torch.randperm(n_rows, generator=g_split)
                n_fit = min(int(rule.n_fit), n_rows)
                n_probe = min(int(rule.n_probe), n_rows)
                idx_fit = perm[:n_fit]
                idx_probe = perm[n_fit:n_fit + n_probe]
                if idx_probe.numel() < n_probe:
                    idx_probe = perm[:n_probe]
                x_fit_data = x_atom[idx_fit]
                y_fit_data = f_atom[idx_fit]
                x_probe_data = x_atom[idx_probe]
                y_probe_data = f_atom[idx_probe]
            except Exception:
                x_fit_data, y_fit_data = x_atom, f_atom
                x_probe_data, y_probe_data = x_atom, f_atom

            if n_consts > 0:
                x_fit_data = _append_declared_constant_columns(x_fit_data, list(declared_consts))
                x_probe_data = _append_declared_constant_columns(x_probe_data, list(declared_consts))

            single_kwargs = dict(
                base_kwargs,
                x_fit_data=x_fit_data,
                y_fit_data=y_fit_data,
                x_probe_data=x_probe_data,
                y_probe_data=y_probe_data,
            )

            stop_event = threading.Event()

            def _run_single_seed(si: int, *, _job_idx: int = job_idx):
                seed_search = int(rule.seed) + si + 1009 * _job_idx
                return si, seed_search, run_explorer(
                    seed_search=seed_search,
                    verbose=(si == 0),
                    stop_event=stop_event,
                    **single_kwargs,
                )

            with ThreadPoolExecutor(max_workers=n_seeds) as pool:
                futures = {pool.submit(_run_single_seed, si): si for si in range(n_seeds)}
                for fut in as_completed(futures):
                    si = futures[fut]
                    try:
                        _, _, rows = fut.result()
                    except Exception as exc:
                        _log(
                            f"[Stage B]  factorized_search error on NN vars={target_var_idxs} "
                            f"dataset={dname} seed={int(rule.seed) + si + 1009 * job_idx}: {exc}"
                        )
                        continue
                    if rows:
                        for row in rows:
                            out = dict(row)
                            out["_probe_dataset_idx"] = int(di)
                            out["_probe_dataset"] = str(dname)
                            results_raw.append(out)
                        if has_structural_solved_result(rows, early_stop_mse=float(rule.early_stop_mse)):
                            stop_event.set()

    return results_raw


__all__ = [
    "StageBEmbedContext",
    "_expr_ir_kwargs_from_rule",
    "build_stageb_probe_jobs",
    "build_stageb_main_candidates",
    "has_structural_solved_result",
    "pool_stageb_results",
    "prepare_stageb_embed_context",
    "row_raw_mse",
    "run_stageb_explorer_jobs",
    "run_stageb_wrapper_pass",
]
