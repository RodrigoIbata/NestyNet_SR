# ruff: noqa: F401
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Iterative rule execution and checkpoint restoration for Stage B."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode, AtomNode, MulNode, PowNode,
    LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode,
    ConjNode, RealNode, ImagNode, AbsNode, ArgNode, Node, collect_all_atoms, collect_nn_atoms,
    _collect_var_idxs_from_node,
    atom_problem_label, count_atom_params, effective_arity, eval_inputs, get_input_exprs,
    clone_ast, clone_inputs,
    ast_to_human_readable,
)

# Optional units precheck (PhySO-like dimensional straightjacket).
# This is imported defensively so Stage B remains usable even if the
# units module is not present in some minimal deployments.
try:
    from nestynet_sr.sr_core.units import (
        _dim_in_rational_span,
        UnitsSpec,
        check_units_ast,
        compute_node_domains,
        eval_analytic_expr_dim,
        is_dimless,
        scale_dim,
        infer_atom_output_dim,
    )
except Exception:  # pragma: no cover
    _dim_in_rational_span = None  # type: ignore
    UnitsSpec = None  # type: ignore
    check_units_ast = None  # type: ignore
    compute_node_domains = None  # type: ignore
    eval_analytic_expr_dim = None  # type: ignore
    is_dimless = None  # type: ignore
    scale_dim = None  # type: ignore
    infer_atom_output_dim = None  # type: ignore

# Import hyperparameters and feature specs from sibling modules
# These will be resolved at runtime
if False:  # TYPE_CHECKING
    pass

# Import shared AST utilities from parent module
from ..ast_utils import (
    check_ast_is_tree as _check_ast_is_tree,
)
from ..ast_utils import (
    compact_expression_repr as _compact_expression_repr,
)
from ..coe_witness import (
    CoEWitnessExecutor,
    coe_stageB_refit_ast_to_payload,
    coe_witness_execution_metadata,
    coe_witness_jobs_from_specs,
    run_fixed_expression_pair_witnesses,
    run_stageB_refit_pair_witnesses,
    run_stageB_refit_pair_witness_preflight,
    summarize_witness_errors,
)
from ..model_selection import (
    ast_cost_physics_prior as _ast_cost_physics_prior,
    complexity_key as _complexity_key,
)
from ..model_selection import (
    mapping_cost as _mapping_cost,
)
from ..model_selection import (
    pareto_front_indices_2d as _pareto_front_indices_2d,
)
from ..model_selection import (
    compute_accept_threshold as _compute_accept_threshold,
    loss_within_floor_or_noise_equivalent as _loss_within_floor_or_noise_equivalent,
    noise_equivalent as _noise_equivalent,
    resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw,
)

# Shared model-selection policy (used by both Stage A & Stage B).
from ..model_selection import (
    nn_multivar_complexity as _shared_nn_multivar_complexity,
)
from ..model_selection import (
    nn_structural_score as _nn_structural_score,
)
from ..model_selection import (
    simplification_budget_decades as _simplification_budget_decades,
)
from ..monomial_screen import candidate_monomial_exponent
from .additive_gauge_scope import AdditiveGaugeGlobalScore, AdditiveGaugeScopeIndex, additive_gauge_global_score
from .homogeneous_gauge_scope import (
    HomogeneousGaugeGlobalScore,
    HomogeneousGaugeScopeIndex,
    homogeneous_gauge_global_score,
)


from ._engine_support import (
    GREEN,
    PURPLE,
    RED,
    RESET,
    _snapshot_rng_state,
    _restore_rng_state,
    GAUGE_SCOPE_RULES,
    GAUGE_TERMINALISH_RULES,
    GAUGE_SENSITIVE_RULES,
    _safe_ast_cost,
    _clamp_nonnegative_finite,
    _loss_excess_above_floor,
    _effective_loss_floor,
    _best_seen_restore_decision,
    _below_floor_regression_cap,
    _below_floor_regression_rejected,
    _candidate_mapping_cost,
    _candidate_is_unpromoted_generic,
    _mapping_descriptor,
    _candidate_mapping_descriptor,
    _candidate_has_mapping,
    _candidate_is_structural_accept,
    _phase2_trigger_flags,
    _target_uid,
    _eval_yspace_mse,
    _asinh_yspace_scale_from_loader,
    _loss_str,
    _format_dim_for_problem,
    _target_dim_for_root,
    _input_basis_dims_for_atom,
    _find_nonsense_units_leaves,
    _annotate_nonsense_units_leaves,
    _problem_candidate_desc,
    STRUCTURAL_LABEL_PREFIXES,
    STRUCTURAL_LABELS,
    candidate_pattern_name,
    SEPARABILITY_LABELS,
    _count_ast_params,
    _candidate_min_free_params,
    _cand_sort_key,
    _candidate_can_beat_floor_locked_state,
    _is_exact_final_leaf_monomial_accept,
    _stageB_state_num_params,
    _stageB_state_num_nn_atoms,
    _stageB_completion_loss_floor,
    _min_following_candidate_free_params,
    _are_we_done_yet,
    _are_we_done_yet_reason,
    _skip_post_accept_polish_for_terminal_state,
    _count_effective_params,
    _leaf_z_data,
    _effective_ratpoly_params,
    _effective_poly_params,
    _unwrap_leaf_core,
    _filter_reuse_map,
    _find_ratpoly_scale_pair,
    _ratpoly_degree_bands,
    _ratpoly_support_degrees,
    _format_ratpoly_support,
    _ratpoly_den_pivot_degree,
    _is_ratpoly_candidate,
    _ratpoly_exps_key,
    _ratpoly_support_signature_exact,
    _ratpoly_num_pivot_degree,
    _lookup_rratpoly_trim_target,
    _lookup_ratpoly_trim_target,
    _build_rratpoly_degree_trim_candidate,
    _ast_node_to_tuple,
    _target_arity,
    atom_content_hash,
    _is_structural_candidate,
    _is_separability_candidate,
    _nn_multivar_complexity,
    _compute_nn_metrics,
)

from ._engine_state import (
    StageBRule,
    StageBState,
    _Checkpoint,
    _materialized_fit_state_for_checkpoint,
    _checkpoint_state_dict_cpu,
    _TRANSIENT_FIT_STATE_SUFFIXES,
    _is_transient_fit_state_key,
    _state_value_clone,
    _load_checkpoint_state_dict,
    Candidate,
    PrecheckResult,
    StageBContext,
)

def _find_worst_accept(ctx: StageBContext) -> Optional[int]:
    """Index into ctx._checkpoints of the worst-performing accept.

    Returns the checkpoint with the highest regression score
    (cand_loss / base_loss).  Skips checkpoints whose (rule, label, target)
    is already red or amber.  Returns ``None`` if no eligible checkpoints.
    """
    best_idx: Optional[int] = None
    best_score: float = -1.0
    _floor = max(float(ctx.loss_floor or 0), float(ctx.worsening_floor or 0), 1e-30)
    for i, ckpt in enumerate(ctx._checkpoints):
        key = (ckpt.accept_rule, ckpt.accept_label, ckpt.accept_target_uid)
        if key in ctx._red_set or key == ctx._last_amber_key:
            continue
        # Read the accept record from the decision log
        rec_idx = ckpt.decision_log_len
        if rec_idx < len(ctx.decision_log):
            rec = ctx.decision_log[rec_idx]
            base_loss = float(rec.get("base_loss", 0) or 0)
            cand_loss = float(rec.get("cand_loss", 0) or 0)
        else:
            base_loss = ckpt.best_val_loss
            cand_loss = base_loss
        score = cand_loss / max(base_loss, _floor)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _pick_atom_factory(atom_factory, i: int):
    """Return per-dataset atom factory for index ``i``."""
    if atom_factory is None or callable(atom_factory):
        return atom_factory
    try:
        return atom_factory[i]
    except Exception:
        return None


def _restore_from_checkpoint(ctx: StageBContext, ckpt: _Checkpoint) -> StageBState:
    """Rebuild model from checkpoint AST + saved state_dict (no LM fit)."""
    from nestynet_sr.sr_core.bridges import (
        build_composite_from_ast,
        make_reuse_only_nn_factory,
        sync_ast_num_segments_from_state_dict,
    )

    from .atom_mapping import _refresh_reuse_from_state

    root = copy.deepcopy(ckpt.root)
    sync_ast_num_segments_from_state_dict(root, ckpt.model_state_dict)
    nn_factory = make_reuse_only_nn_factory(
        device=ctx.device, dtype=ctx.dtype, fresh_nn_factory=ctx.fresh_nn_factory,
    )
    # Rebuild model from AST
    model, _ = build_composite_from_ast(
        root, dtype=ctx.dtype, device=ctx.device,
        nn_factory=nn_factory,
        atom_factory=_pick_atom_factory(getattr(ctx, "atom_factory", None), 0),
        reuse={}, return_atom_map=True,
    )
    # Restore fit-link attrs (set by _fit_candidate_root)
    setattr(model, "fit_y_link", getattr(ctx.lm_hp, "fit_y_link", None))
    setattr(model, "fit_y_link_scale", getattr(ctx.lm_hp, "fit_y_link_scale", 1.0))
    # Load saved weights
    _load_checkpoint_state_dict(
        model,
        ckpt.model_state_dict,
        log_fn=getattr(ctx, "log", None),
        label="best-seen checkpoint",
    )
    model = model.to(ctx.device)
    reuse = _refresh_reuse_from_state(root, model)

    state = StageBState(
        root=root,
        model=model,
        reuse=reuse,
        val_loss=ckpt.val_loss,
        acceptance_noise_floor_raw=float(
            getattr(ckpt, "acceptance_noise_floor_raw", 0.0) or 0.0
        ),
        acceptance_noise_n_eff=getattr(ckpt, "acceptance_noise_n_eff", None),
        complexity_mapping_cost=float(getattr(ckpt, "complexity_mapping_cost", 0.0) or 0.0),
        simplification_path=copy.deepcopy(getattr(ckpt, "simplification_path", []) or []),
    )
    state.generic_approximant_unpromoted = bool(
        getattr(ckpt, "generic_approximant_unpromoted", False)
    )

    # Multi-dataset support
    if ckpt.reuse_state_dicts is not None:
        models = []
        reuses = []
        for i, sd in enumerate(ckpt.reuse_state_dicts):
            root_i = copy.deepcopy(ckpt.root)
            sync_ast_num_segments_from_state_dict(root_i, sd)
            m, _ = build_composite_from_ast(
                root_i, dtype=ctx.dtype, device=ctx.device,
                nn_factory=nn_factory,
                atom_factory=_pick_atom_factory(getattr(ctx, "atom_factory", None), i),
                reuse={}, return_atom_map=True,
            )
            _load_checkpoint_state_dict(
                m,
                sd,
                log_fn=getattr(ctx, "log", None),
                label=f"dataset-{i} checkpoint",
            )
            m = m.to(ctx.device)
            models.append(m)
            reuses.append(_refresh_reuse_from_state(root_i, m))
        state.models = models
        state.reuses = reuses
        state.model = models[0] if models else model
        state.reuse = reuses[0] if reuses else reuse

    return state


class StageBEngine:
    """
    Main execution engine for Stage B refinement.

    Applies a pipeline of rewrite rules iteratively until no improvement is found
    or max iterations reached.

    Attributes:
        rules: List of StageBRule instances to apply
    """

    def __init__(self, rules: List[StageBRule]):
        """
        Initialize engine with a list of rewrite rules.

        Args:
            rules: List of StageBRule instances (applied in order)
        """
        self.rules = list(rules)

    def _build_probe_views(self, ctx: StageBContext) -> List[Tuple[int, str, StageBContext]]:
        """Build dataset-specific probe views for conjoint proposal generation.

        In single-dataset mode, this returns one view (the original context).
        In multi-dataset mode, this returns one lightweight context per dataset,
        each with dataset-specific ``state.model/state.reuse`` and probe loaders.
        """
        if not isinstance(ctx.train_loader, (list, tuple)):
            ds_name = (
                str(ctx.dataset_ids[0])
                if isinstance(ctx.dataset_ids, (list, tuple)) and len(ctx.dataset_ids) > 0
                else "ds0"
            )
            return [(0, ds_name, ctx)]

        train_loaders = list(ctx.train_loader)
        val_loaders = (
            list(ctx.val_loader)
            if isinstance(ctx.val_loader, (list, tuple))
            else [ctx.val_loader for _ in range(len(train_loaders))]
        )
        n_ds = len(train_loaders)
        if len(val_loaders) < n_ds and len(val_loaders) > 0:
            val_loaders = val_loaders + [val_loaders[-1] for _ in range(n_ds - len(val_loaders))]

        models = list(getattr(ctx.state, "models", None) or [])
        if len(models) < n_ds:
            models = models + [ctx.state.model for _ in range(n_ds - len(models))]
        reuses = list(getattr(ctx.state, "reuses", None) or [])
        if len(reuses) < n_ds:
            reuses = reuses + [ctx.state.reuse for _ in range(n_ds - len(reuses))]

        loss_scales = list(ctx.loss_scales) if isinstance(ctx.loss_scales, (list, tuple)) else None
        if loss_scales is None or len(loss_scales) < n_ds:
            loss_scales = [ctx.loss_scale for _ in range(n_ds)]

        dataset_names = (
            [str(x) for x in ctx.dataset_ids]
            if isinstance(ctx.dataset_ids, (list, tuple)) and len(ctx.dataset_ids) == n_ds
            else [f"ds{i}" for i in range(n_ds)]
        )

        views: List[Tuple[int, str, StageBContext]] = []
        for i in range(n_ds):
            st_i = copy.copy(ctx.state)
            st_i.model = models[i]
            st_i.reuse = reuses[i]
            # Probe rules should use this dataset view directly.
            st_i.models = None
            st_i.reuses = None

            view = copy.copy(ctx)
            view.state = st_i
            view.train_loader = train_loaders[i]
            view.val_loader = val_loaders[i]
            view.loss_scale = float(loss_scales[i])
            view.loss_scales = None
            view.dataset_ids = [dataset_names[i]]
            view.agg_mode = "mean"
            view.agg_weights = None
            # Keep probe caches dataset-local to avoid cross-dataset contamination.
            view._cache = {}
            view._dim_cache = {}
            view._dim_cache_root_id = None
            views.append((i, dataset_names[i], view))

        return views

    def _candidate_pool_key(
        self,
        ctx: StageBContext,
        rule_name: str,
        cand: Candidate,
    ) -> Tuple[str, Any, Any]:
        """Best-effort stable key for deduplicating conjoint proposals."""
        sig = ctx.candidate_signature(cand)
        if sig is not None:
            return ("sig", rule_name, sig)

        # Fallback: pool by rule+label so dataset-specific initial values
        # do not duplicate structurally equivalent candidates.
        return ("label", rule_name, str(cand.label))

    def _mark_gauge_tainted_candidates(
        self,
        ctx: StageBContext,
        rule_name: str,
        target: Node,
        cands: List[Candidate],
    ) -> List[Candidate]:
        """Attach unresolved-gauge acceptance metadata to local candidates."""
        if rule_name in GAUGE_SCOPE_RULES:
            return cands
        if rule_name not in (GAUGE_SENSITIVE_RULES | GAUGE_TERMINALISH_RULES):
            return cands
        additive_scope = None
        try:
            additive_scope = ctx.additive_gauge_index().scope_for_target(target)
        except Exception:
            additive_scope = None
        homogeneous_scope = None
        try:
            homogeneous_scope = ctx.homogeneous_gauge_index().scope_for_target(target)
        except Exception:
            homogeneous_scope = None
        if (
            (additive_scope is None or not getattr(additive_scope, "unresolved", False))
            and (homogeneous_scope is None or not getattr(homogeneous_scope, "unresolved", False))
        ):
            return cands
        additive_before = None
        homogeneous_before = None
        if additive_scope is not None and getattr(additive_scope, "unresolved", False):
            try:
                additive_before = ctx.additive_gauge_global_score()
            except Exception:
                additive_before = None
        if homogeneous_scope is not None and getattr(homogeneous_scope, "unresolved", False):
            try:
                homogeneous_before = ctx.homogeneous_gauge_global_score()
            except Exception:
                homogeneous_before = None
        for cand in cands or []:
            if cand is None:
                continue
            if not isinstance(getattr(cand, "meta", None), dict):
                cand.meta = {}
            if additive_scope is not None and getattr(additive_scope, "unresolved", False):
                cand.meta.setdefault("additive_gauge_sensitive", rule_name in GAUGE_SENSITIVE_RULES)
                cand.meta.setdefault("additive_gauge_scope_uid", getattr(additive_scope, "uid", ""))
                cand.meta.setdefault("additive_gauge_requires_scope_improvement", True)
                cand.meta.setdefault("additive_gauge_rule_name", rule_name)
                if additive_before is not None:
                    cand.meta.setdefault("additive_gauge_score_before", additive_before)
            if homogeneous_scope is not None and getattr(homogeneous_scope, "unresolved", False):
                cand.meta.setdefault("homogeneous_gauge_sensitive", rule_name in GAUGE_SENSITIVE_RULES)
                cand.meta.setdefault("homogeneous_gauge_scope_uid", getattr(homogeneous_scope, "uid", ""))
                cand.meta.setdefault("homogeneous_gauge_requires_scope_improvement", True)
                cand.meta.setdefault("homogeneous_gauge_rule_name", rule_name)
                if homogeneous_before is not None:
                    cand.meta.setdefault("homogeneous_gauge_score_before", homogeneous_before)
        return cands

    def _propose_candidates_conjoint(
        self,
        ctx: StageBContext,
        rule: StageBRule,
        rule_name: str,
        target: Node,
    ) -> List[Candidate]:
        """Generate candidates from pooled multi-dataset probe statistics.

        We probe each dataset-specific model/reuse view and merge/deduplicate
        candidates into a single list. This treats the proposal stage as one
        combined experiment while keeping acceptance logic unchanged.
        """
        if bool(getattr(rule, "multi_probe_native", False)):
            cands = rule.propose(ctx, target) or []
            return self._mark_gauge_tainted_candidates(ctx, rule_name, target, cands)

        views = self._build_probe_views(ctx)
        if len(views) <= 1:
            cands = rule.propose(ctx, target) or []
            return self._mark_gauge_tainted_candidates(ctx, rule_name, target, cands)

        pooled: Dict[Tuple[str, Any, Any], Candidate] = {}
        order: List[Tuple[str, Any, Any]] = []
        raw_count = 0

        for ds_idx, ds_name, view in views:
            try:
                cands_i = rule.propose(view, target) or []
            except Exception as e:
                ctx.log(f"[Stage B]  conjoint probe warning ({rule_name}, {ds_name}): {e}")
                continue
            raw_count += len(cands_i)
            for cand in cands_i:
                if cand is None:
                    continue
                key = self._candidate_pool_key(ctx, rule_name, cand)
                if key not in pooled:
                    if not isinstance(getattr(cand, "meta", None), dict):
                        cand.meta = {}
                    cand.meta.setdefault("probe_dataset_idxs", [])
                    cand.meta.setdefault("probe_datasets", [])
                    pooled[key] = cand
                    order.append(key)

                meta = pooled[key].meta if isinstance(pooled[key].meta, dict) else {}
                meta.setdefault("probe_dataset_idxs", []).append(int(ds_idx))
                meta.setdefault("probe_datasets", []).append(str(ds_name))
                meta["probe_dataset_count"] = len(set(meta.get("probe_dataset_idxs", [])))
                pooled[key].meta = meta

        merged: List[Candidate] = [pooled[k] for k in order]
        for cand in merged:
            if not isinstance(getattr(cand, "meta", None), dict):
                cand.meta = {}
            idxs = sorted(set(int(i) for i in cand.meta.get("probe_dataset_idxs", [])))
            names = sorted(set(str(n) for n in cand.meta.get("probe_datasets", [])))
            cand.meta["probe_dataset_idxs"] = idxs
            cand.meta["probe_datasets"] = names
            cand.meta["probe_dataset_count"] = len(idxs)

        ctx.log(
            f"[Stage B]  conjoint probe {rule_name}: "
            f"{raw_count} raw -> {len(merged)} pooled candidates over {len(views)} datasets"
        )
        return self._mark_gauge_tainted_candidates(ctx, rule_name, target, merged)

    def _try_candidates_for_target(
        self,
        ctx: StageBContext,
        rule: StageBRule,
        rule_name: str,
        target: Node,
        cands: List[Candidate],
        exhaustive: bool = False,
    ) -> bool:
        """Try candidates for a single (rule, target) using an optional
        frugal-vs-greedy screening policy.

        Returns True if a candidate is accepted (ctx.state updated).

        This is intentionally engine-level so rule implementations remain pure
        proposal generators.
        """

        # Filter invalid candidates (keep lazy candidates with a builder)
        valid: List[Candidate] = [
            c for c in (cands or [])
            if c is not None and (getattr(c, "root", None) is not None or getattr(c, "builder", None) is not None)
        ]
        if not valid:
            return False

        # Candidate-iteration policy (LMHyperparams)
        policy = str(getattr(ctx.lm_hp, "stageB_candidate_policy", "sequential")).lower().strip()
        screen_enable = bool(getattr(ctx.lm_hp, "stageB_screen_enable", False))
        screen_topk = int(getattr(ctx.lm_hp, "stageB_screen_topk", 6) or 0)
        screen_epochs = int(getattr(ctx.lm_hp, "stageB_screen_epochs", 0) or 0)
        dom_dec = float(getattr(ctx.lm_hp, "stageB_screen_dominance_decades", 0.50) or 0.0)
        greedy_fullfit_max = int(getattr(ctx.lm_hp, "stageB_greedy_fullfit_max", 0) or 0)

        if policy not in ("sequential", "frugal", "greedy", "dynamic"):
            policy = "sequential"

        try:
            _target_desc = rule.describe_target(target)
        except Exception:
            _target_desc = "<target>"

        _target_id = _target_uid(ctx.state.root, target)

        def _rule_candidate_min_free_params(cand: Candidate) -> int:
            try:
                return max(0, int(rule.candidate_min_free_params(cand)))
            except Exception:
                return _candidate_min_free_params(cand)

        if _are_we_done_yet(
            ctx,
            following_candidates=valid,
            candidate_min_free_params_fn=_rule_candidate_min_free_params,
        ):
            return False

        def _log_precheck_reject(pre, cand: Candidate):
            sig_str = f" sig={pre.signature}" if getattr(pre, "signature", None) is not None else ""
            n_attempted = len(ctx.attempted_transformations.get(rule_name, set()))
            ctx.log(
                f"[Stage B]  Precheck reject ({pre.reason}) "
                f"rule={rule_name} cand={cand.label} target={_target_desc}"
                f"{sig_str} attempted={n_attempted}"
            )
            ctx._record_decision(
                outcome="precheck_reject",
                rule=rule_name,
                label=cand.label,
                reason=pre.reason or "precheck",
                target=_target_desc,
                target_uid=_target_id,
                base_loss=float(ctx.state.val_loss),
                cand=cand,
            )

        def _log_reject(cand: Candidate, cand_state: StageBState, reason: str, n_params_b: int):
            base_loss = float(ctx.state.val_loss)
            cand_loss = float(cand_state.val_loss)
            base_cx = _nn_multivar_complexity(ctx.state.root)
            cand_cx = _nn_multivar_complexity(cand_state.root)
            try:
                n_params_c = int(cand_state.model.num_parameters())
            except Exception:
                n_params_c = -1
            ctx.log(
                f"[Stage B]    Reject ({cand.label}): {reason} | "
                f"loss {base_loss:.3e}->{cand_loss:.3e} (Δ{cand_loss - base_loss:+.2e}) | "
                f"params {n_params_b}->{n_params_c} | "
                f"cx {base_cx}->{cand_cx}"
            )
            if cand.label == "nonsense_units_zero_prune":
                ctx.log(
                    f"[Stage B]    Keeping {_problem_candidate_desc(cand)}; "
                    f"zero-prune did not meet acceptance criterion ({reason})"
                )
            ctx._record_decision(
                outcome="reject",
                rule=rule_name,
                label=cand.label,
                reason=reason or "reject",
                target=_target_desc,
                target_uid=_target_id,
                base_loss=base_loss,
                cand_loss=cand_loss,
                n_params_base=n_params_b,
                n_params_cand=n_params_c,
                base_complexity=list(base_cx),
                cand_complexity=list(cand_cx),
                cand=cand,
                base_root=ctx.state.root,
                cand_root=cand_state.root,
                base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                cand_mapping_cost=_candidate_mapping_cost(cand),
            )

        def _attempt_full_fit(cand: Candidate) -> bool:
            # Lazy candidate: evaluate deferred builder on demand.
            if cand.root is None:
                if not cand.materialise():
                    return False

            # Fast dedup: skip candidates previously rejected on the same target
            _rkey = (rule_name, cand.label, _target_id)
            if _rkey in ctx._rejected_keys:
                ctx.log(
                    f"[Stage B]  Skipping {cand.label} (previously rejected on same target)"
                )
                ctx._record_decision(
                    outcome="dedup_skip",
                    rule=rule_name,
                    label=cand.label,
                    reason="previously-rejected-on-same-target",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=float(ctx.state.val_loss),
                    cand=cand,
                )
                return False

            # Layer 2: skip candidates banned by backtracking (red=permanent, amber=tentative)
            _skip_key = (rule_name, cand.label, _target_id)
            if _skip_key in ctx._red_set or _skip_key == ctx._last_amber_key:
                _skip_kind = "red_skip" if _skip_key in ctx._red_set else "amber_skip"
                ctx.log(f"[Stage B]  {RED}Skipped{RESET} ({cand.label}): {_skip_kind}")
                ctx._record_decision(
                    outcome=_skip_kind, rule=rule_name, label=cand.label,
                    reason=_skip_kind, target=_target_desc,
                    target_uid=_target_id,
                    base_loss=float(ctx.state.val_loss), cand=cand,
                )
                return False

            # Precheck + record attempt
            pre = ctx.precheck_candidate(rule_name, cand, record_attempt=True)
            if not pre.ok:
                _log_precheck_reject(pre, cand)
                return False

            msg = cand.meta.get("log", None) if isinstance(getattr(cand, "meta", None), dict) else None
            if msg is None:
                msg = f"[Stage B]  Trying {cand.label} on {rule.describe_target(target)}"
            ctx.log(msg)

            # Capture baseline metrics BEFORE fitting (accept() overwrites state)
            _pre_base_loss = float(ctx.state.val_loss)
            _pre_base_cx = list(_nn_multivar_complexity(ctx.state.root))
            _pre_n_params_b = int(ctx.state.model.num_parameters())

            try:
                cand_state = ctx.fit_candidate(cand)
            except Exception as exc:
                ctx.log(
                    f"[Stage B]    Candidate ({cand.label}) fit failed: "
                    f"{type(exc).__name__}: {exc}; trying next."
                )
                ctx._rejected_keys.add(_rkey)
                ctx._record_decision(
                    outcome="fit_exception_reject",
                    rule=rule_name,
                    label=cand.label,
                    reason=f"fit-exception:{type(exc).__name__}",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=_pre_base_loss,
                    cand_loss=float("inf"),
                    n_params_base=_pre_n_params_b,
                    base_complexity=_pre_base_cx,
                    cand=cand,
                    base_root=ctx.state.root,
                    cand_root=cand.root,
                    base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                    cand_mapping_cost=_candidate_mapping_cost(cand),
                )
                return False
            if not math.isfinite(cand_state.val_loss):
                ctx.log(
                    f"[Stage B]    Candidate ({cand.label}) rejected at init (non-finite loss); trying next."
                )
                ctx._record_decision(
                    outcome="nonfinite_reject",
                    rule=rule_name,
                    label=cand.label,
                    reason="non-finite-loss-after-fit",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=_pre_base_loss,
                    cand_loss=float(cand_state.val_loss),
                    n_params_base=_pre_n_params_b,
                    base_complexity=_pre_base_cx,
                    cand=cand,
                    base_root=ctx.state.root,
                    cand_root=cand_state.root,
                    base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                    cand_mapping_cost=_candidate_mapping_cost(cand),
                )
                return False

            _annotate_nonsense_units_leaves(
                cand_state,
                units_spec=getattr(ctx, "units_spec", None),
                enforce_units=bool(getattr(ctx, "enforce_units", False)),
                log_fn=ctx.log,
                mutate=False,
            )

            n_params_b = _pre_n_params_b
            n_params_c = int(cand_state.model.num_parameters())
            ctx.log(
                f"[Stage B]    Candidate ({cand.label}): params={n_params_c}, val-loss={_loss_str(cand_state.val_loss, ctx.lm_hp)}"
            )

            if _is_ratpoly_candidate(cand):
                ok, cand, cand_state, reason = ctx._select_ratpoly_candidate(cand, cand_state)
                n_params_c = int(cand_state.model.num_parameters())
            else:
                ok, reason = ctx.should_accept(cand, cand_state)
            if ok:
                ok, reason = ctx.gauge_acceptance_gate(cand, cand_state, reason)
            if ok:
                # Record accept BEFORE ctx.accept() overwrites state
                try:
                    _snap = _compact_expression_repr(cand_state.root, max_length=400)
                except Exception:
                    _snap = str(cand_state.root)
                cand_cx = list(_nn_multivar_complexity(cand_state.root))
                _coe_ok, _coe_reason = ctx.coe_stageB_committee_gate(
                    rule=rule_name,
                    label=cand.label,
                    reason=reason or "accepted",
                    target=_target_desc,
                    target_uid=_target_id,
                    cand=cand,
                    cand_state=cand_state,
                    n_params_base=n_params_b,
                    n_params_cand=n_params_c,
                )
                if not _coe_ok:
                    _log_reject(cand, cand_state, _coe_reason or "reject-coe-stageB-gate", n_params_b)
                    ctx._rejected_keys.add(_rkey)
                    return False
                if _coe_reason and "accepted" in str(_coe_reason):
                    reason = f"{reason or 'accepted'}; {_coe_reason}"
                ctx._record_decision(
                    outcome="accept",
                    rule=rule_name,
                    label=cand.label,
                    reason=reason or "accepted",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=_pre_base_loss,
                    cand_loss=float(cand_state.val_loss),
                    n_params_base=n_params_b,
                    n_params_cand=n_params_c,
                    base_complexity=_pre_base_cx,
                    cand_complexity=cand_cx,
                    cand=cand,
                    ast_snapshot=_snap,
                    base_root=ctx.state.root,
                    cand_root=cand_state.root,
                    base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                    cand_mapping_cost=_candidate_mapping_cost(cand),
                )
                _previous_root = ctx.state.root
                ctx.accept(cand, cand_state, reason or "accepted")
                if not _skip_post_accept_polish_for_terminal_state(ctx):
                    ctx.maybe_shadow_polish_subtrees_after_accept(
                        previous_root=_previous_root,
                        accepted_label=cand.label,
                    )
                    ctx.maybe_polish_after_accept()
                return True

            _log_reject(cand, cand_state, reason or "reject", n_params_b)
            ctx._rejected_keys.add(_rkey)
            return False

        def _evaluate_candidate(cand: Candidate):
            """Fit and evaluate a candidate without accepting.

            Returns ``(cand, cand_state, reason)`` if acceptable, else ``None``.
            """
            # Fast dedup: skip candidates previously rejected on the same target
            _rkey = (rule_name, cand.label, _target_id)
            if _rkey in ctx._rejected_keys:
                ctx.log(
                    f"[Stage B]  Skipping {cand.label} (previously rejected on same target)"
                )
                ctx._record_decision(
                    outcome="dedup_skip",
                    rule=rule_name,
                    label=cand.label,
                    reason="previously-rejected-on-same-target",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=float(ctx.state.val_loss),
                    cand=cand,
                )
                return None

            # Layer 2: skip candidates banned by backtracking (red=permanent, amber=tentative)
            _skip_key = (rule_name, cand.label, _target_id)
            if _skip_key in ctx._red_set or _skip_key == ctx._last_amber_key:
                _skip_kind = "red_skip" if _skip_key in ctx._red_set else "amber_skip"
                ctx.log(f"[Stage B]  {RED}Skipped{RESET} ({cand.label}): {_skip_kind}")
                ctx._record_decision(
                    outcome=_skip_kind, rule=rule_name, label=cand.label,
                    reason=_skip_kind, target=_target_desc,
                    target_uid=_target_id,
                    base_loss=float(ctx.state.val_loss), cand=cand,
                )
                return None

            pre = ctx.precheck_candidate(rule_name, cand, record_attempt=True)
            if not pre.ok:
                _log_precheck_reject(pre, cand)
                return None

            msg = (
                cand.meta.get("log", None)
                if isinstance(getattr(cand, "meta", None), dict)
                else None
            )
            if msg is None:
                msg = f"[Stage B]  Trying {cand.label} on {rule.describe_target(target)}"
            ctx.log(msg)

            _pre_base_loss = float(ctx.state.val_loss)
            _pre_base_cx = list(_nn_multivar_complexity(ctx.state.root))
            _pre_n_params_b = int(ctx.state.model.num_parameters())
            try:
                cand_state = ctx.fit_candidate(cand)
            except Exception as exc:
                ctx.log(
                    f"[Stage B]    Candidate ({cand.label}) fit failed: "
                    f"{type(exc).__name__}: {exc}; trying next."
                )
                ctx._rejected_keys.add(_rkey)
                ctx._record_decision(
                    outcome="fit_exception_reject",
                    rule=rule_name,
                    label=cand.label,
                    reason=f"fit-exception:{type(exc).__name__}",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=_pre_base_loss,
                    cand_loss=float("inf"),
                    n_params_base=_pre_n_params_b,
                    base_complexity=_pre_base_cx,
                    cand=cand,
                    base_root=ctx.state.root,
                    cand_root=cand.root,
                    base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                    cand_mapping_cost=_candidate_mapping_cost(cand),
                )
                return None
            if not math.isfinite(cand_state.val_loss):
                ctx.log(
                    f"[Stage B]    Candidate ({cand.label}) rejected at init "
                    f"(non-finite loss); trying next."
                )
                ctx._record_decision(
                    outcome="nonfinite_reject",
                    rule=rule_name,
                    label=cand.label,
                    reason="non-finite-loss-after-fit",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=float(ctx.state.val_loss),
                    cand_loss=float(cand_state.val_loss),
                    cand=cand,
                    base_root=ctx.state.root,
                    cand_root=cand_state.root,
                    base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                    cand_mapping_cost=_candidate_mapping_cost(cand),
                )
                return None

            _annotate_nonsense_units_leaves(
                cand_state,
                units_spec=getattr(ctx, "units_spec", None),
                enforce_units=bool(getattr(ctx, "enforce_units", False)),
                log_fn=ctx.log,
                mutate=False,
            )

            n_params_b = int(ctx.state.model.num_parameters())
            n_params_c = int(cand_state.model.num_parameters())
            ctx.log(
                f"[Stage B]    Candidate ({cand.label}): params={n_params_c}, "
                f"val-loss={_loss_str(cand_state.val_loss, ctx.lm_hp)}"
            )

            if _is_ratpoly_candidate(cand):
                ok, cand, cand_state, reason = ctx._select_ratpoly_candidate(cand, cand_state)
            else:
                ok, reason = ctx.should_accept(cand, cand_state)
            if ok:
                ok, reason = ctx.gauge_acceptance_gate(cand, cand_state, reason)
            if not ok:
                _log_reject(cand, cand_state, reason or "reject", n_params_b)
                ctx._rejected_keys.add(_rkey)
                return None

            return cand, cand_state, reason

        def _accept_evaluated_candidate(
            cand: Candidate,
            cand_state: StageBState,
            reason: Optional[str],
        ) -> bool:
            # Record accept before ctx.accept() overwrites state.
            try:
                _snap = _compact_expression_repr(cand_state.root, max_length=400)
            except Exception:
                _snap = str(cand_state.root)
            _pre_bl = float(ctx.state.val_loss)
            _pre_bpx = list(_nn_multivar_complexity(ctx.state.root))
            try:
                _pre_bp = int(ctx.state.model.num_parameters())
            except Exception:
                _pre_bp = -1
            try:
                _pre_cp = int(cand_state.model.num_parameters())
            except Exception:
                _pre_cp = -1
            _coe_ok, _coe_reason = ctx.coe_stageB_committee_gate(
                rule=rule_name,
                label=cand.label,
                reason=reason or "accepted",
                target=_target_desc,
                target_uid=_target_id,
                cand=cand,
                cand_state=cand_state,
                n_params_base=_pre_bp,
                n_params_cand=_pre_cp,
            )
            if not _coe_ok:
                _log_reject(
                    cand,
                    cand_state,
                    _coe_reason or "reject-coe-stageB-gate",
                    _pre_bp,
                )
                ctx._rejected_keys.add((rule_name, cand.label, _target_id))
                return False
            if _coe_reason and "accepted" in str(_coe_reason):
                reason = f"{reason or 'accepted'}; {_coe_reason}"
            ctx._record_decision(
                outcome="accept",
                rule=rule_name,
                label=cand.label,
                reason=reason or "accepted",
                target=_target_desc,
                target_uid=_target_id,
                base_loss=_pre_bl,
                cand_loss=float(cand_state.val_loss),
                n_params_base=_pre_bp,
                n_params_cand=_pre_cp,
                base_complexity=_pre_bpx,
                cand_complexity=list(_nn_multivar_complexity(cand_state.root)),
                cand=cand,
                ast_snapshot=_snap,
                base_root=ctx.state.root,
                cand_root=cand_state.root,
                base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                cand_mapping_cost=_candidate_mapping_cost(cand),
            )
            _previous_root = ctx.state.root
            ctx.accept(cand, cand_state, reason or "accepted")
            if not _skip_post_accept_polish_for_terminal_state(ctx):
                ctx.maybe_shadow_polish_subtrees_after_accept(
                    previous_root=_previous_root,
                    accepted_label=cand.label,
                )
                ctx.maybe_polish_after_accept()
            return True

        def _macro_noisy_parsimony_enabled() -> bool:
            if str(rule_name) != "compound_fn_macros":
                return False
            if not bool(getattr(ctx.lm_hp, "macro_noisy_parsimony_enable", True)):
                return False
            try:
                nf = float(_resolve_acceptance_noise_floor_raw(ctx.lm_hp, ctx.loss_scale))
            except Exception:
                nf = 0.0
            if not (math.isfinite(nf) and nf > 0.0):
                return False

            def _meta(c: Candidate) -> dict:
                return c.meta if isinstance(getattr(c, "meta", None), dict) else {}

            has_clean = any(bool(_meta(c).get("macro_clean_singleton", False)) for c in valid)
            has_rich = any(
                bool(_meta(c).get("macro_combo", False))
                or str(c.label) in {"cf_combo", "cf_sharedpref", "cf_sharedpref_resid"}
                for c in valid
            )
            return bool(has_clean and has_rich)

        def _candidate_macro_meta(c: Candidate) -> dict:
            return c.meta if isinstance(getattr(c, "meta", None), dict) else {}

        def _is_clean_macro_candidate(c: Candidate) -> bool:
            return bool(_candidate_macro_meta(c).get("macro_clean_singleton", False))

        def _is_rich_macro_candidate(c: Candidate) -> bool:
            return bool(_candidate_macro_meta(c).get("macro_combo", False)) or str(c.label) in {
                "cf_combo",
                "cf_sharedpref",
                "cf_sharedpref_resid",
            }

        def _stageB_noise_n_eff() -> Optional[float]:
            for src in (ctx, getattr(ctx, "state", None), ctx.lm_hp):
                try:
                    v = float(getattr(src, "acceptance_noise_n_eff", None))
                    if math.isfinite(v) and v > 0.0:
                        return v
                except Exception:
                    continue
            return None

        def _candidate_meta(c: Candidate) -> dict:
            return c.meta if isinstance(getattr(c, "meta", None), dict) else {}

        def _meta_int(meta: dict, key: str, default: int = 99) -> int:
            try:
                return int(meta.get(key, default))
            except Exception:
                return int(default)

        def _is_sparse_ratpoly_1d_candidate(c: Candidate) -> bool:
            meta = _candidate_meta(c)
            if bool(meta.get("terminal_protected", False)) and str(
                meta.get("terminal_family", "")
            ) == "ratpoly_1d":
                return True
            family = candidate_pattern_name(c)
            if family not in {"ratpoly_1d", "last_ratpoly_1d"}:
                return False
            deg_num = _meta_int(meta, "deg_num")
            deg_den = _meta_int(meta, "deg_den")
            n_num = _meta_int(meta, "n_terms_num")
            n_den = _meta_int(meta, "n_terms_den")
            return bool(deg_num <= 1 and deg_den <= 1 and n_num <= 2 and n_den <= 2)

        def _is_flexible_terminal_candidate(c: Candidate) -> bool:
            meta = _candidate_meta(c)
            label = str(getattr(c, "label", "") or "")
            return bool(
                meta.get("terminal_flexible_approximant", False)
                or label.startswith("leaftr_")
                or meta.get("macro_combo", False)
            )

        def _is_terminal_protected_candidate(c: Candidate) -> bool:
            meta = _candidate_meta(c)
            return bool(
                meta.get("macro_clean_singleton", False)
                or _is_sparse_ratpoly_1d_candidate(c)
            )

        def _terminal_priority_key(
            row: Tuple[Candidate, StageBState, Optional[str], float],
        ) -> Tuple[Any, ...]:
            cand, cand_state, _reason, loss = row
            meta = _candidate_meta(cand)
            family = candidate_pattern_name(cand)
            if bool(meta.get("macro_clean_singleton", False)):
                family_rank = 0
            elif _is_sparse_ratpoly_1d_candidate(cand):
                family_rank = 1
            elif family in {"inv_poly", "sparse_factor_1d"}:
                family_rank = 2
            elif _is_flexible_terminal_candidate(cand):
                family_rank = 9
            else:
                family_rank = 5
            try:
                ast_cost = float(_ast_cost_physics_prior(cand_state.root))
            except Exception:
                ast_cost = float("inf")
            n_terms = meta.get("terminal_n_terms", meta.get("n_terms", None))
            if n_terms is None:
                n_terms = _meta_int(meta, "n_terms_num", 0) + _meta_int(meta, "n_terms_den", 0)
                if int(n_terms) <= 0:
                    n_terms = 99
            try:
                n_terms_i = int(n_terms)
            except Exception:
                n_terms_i = 99
            return (
                int(family_rank),
                _rule_candidate_min_free_params(cand),
                int(n_terms_i),
                float(ast_cost),
                float(loss),
            )

        def _rank_noisy_terminal_results(
            rows: List[Tuple[Candidate, StageBState, Optional[str], float]],
        ) -> List[Tuple[Candidate, StageBState, Optional[str], float]]:
            by_loss = sorted(rows, key=lambda row: float(row[3]))
            if len(by_loss) <= 1:
                return by_loss
            try:
                nf = float(_resolve_acceptance_noise_floor_raw(ctx.lm_hp, ctx.loss_scale))
            except Exception:
                nf = 0.0
            if not (math.isfinite(nf) and nf > 0.0):
                return by_loss
            best_loss = float(by_loss[0][3])
            n_eff = _stageB_noise_n_eff()
            noise_mult = float(getattr(ctx.lm_hp, "stageB_noisy_terminal_noise_mult", 1.0) or 1.0)
            rel_tol = float(getattr(ctx.lm_hp, "stageB_noisy_terminal_rel_tol", 1.0e-3) or 1.0e-3)
            tied = [
                row
                for row in by_loss
                if _noise_equivalent(
                    float(row[3]),
                    best_loss,
                    noise_floor=nf,
                    n_eff=n_eff,
                    noise_mult=noise_mult,
                    rel_tol=rel_tol,
                )
            ]
            if not tied:
                return by_loss
            best_row = by_loss[0]
            if not (
                any(_is_terminal_protected_candidate(row[0]) for row in tied)
                and _is_flexible_terminal_candidate(best_row[0])
            ):
                return by_loss
            tied_sorted = sorted(tied, key=_terminal_priority_key)
            if len(tied_sorted) > 1:
                ctx.log(
                    "[Stage B]  Noisy terminal tournament tied set: "
                    + ", ".join(
                        f"{row[0].label}(loss={_loss_str(row[3], ctx.lm_hp)}, "
                        f"protected={_is_terminal_protected_candidate(row[0])}, "
                        f"flexible={_is_flexible_terminal_candidate(row[0])})"
                        for row in tied_sorted
                    )
                )
            tied_ids = {id(row[0]) for row in tied_sorted}
            return tied_sorted + [row for row in by_loss if id(row[0]) not in tied_ids]

        def _try_noisy_macro_tournament() -> bool:
            clean_cap = max(1, int(getattr(ctx.lm_hp, "macro_noisy_parsimony_clean_cap", 3) or 3))
            rich_cap = max(1, int(getattr(ctx.lm_hp, "macro_noisy_parsimony_rich_cap", 4) or 4))
            other_cap = max(0, int(getattr(ctx.lm_hp, "macro_noisy_parsimony_other_cap", 1) or 0))
            protected = [c for c in valid if _is_clean_macro_candidate(c)]
            rich = [c for c in valid if (not _is_clean_macro_candidate(c)) and _is_rich_macro_candidate(c)]
            other = [
                c
                for c in valid
                if (not _is_clean_macro_candidate(c)) and (not _is_rich_macro_candidate(c))
            ]
            slate: List[Candidate] = []
            seen_ids = set()
            for cand in protected[:clean_cap] + rich[:rich_cap] + other[:other_cap]:
                if id(cand) in seen_ids:
                    continue
                seen_ids.add(id(cand))
                slate.append(cand)
            if not slate:
                return False
            ctx.log(
                "[Stage B]  Noisy macro parsimony tournament: "
                + ", ".join(
                    f"{c.label}({_rule_candidate_min_free_params(c)}p"
                    f", terms={_candidate_macro_meta(c).get('n_terms', '?')})"
                    for c in slate
                )
            )

            results: List[Tuple[Candidate, StageBState, Optional[str], float]] = []
            for cand in slate:
                result = _evaluate_candidate(cand)
                if result is None:
                    continue
                cand_eval, cand_state, reason = result
                try:
                    cand_loss = float(cand_state.val_loss)
                except Exception:
                    cand_loss = float("inf")
                if math.isfinite(cand_loss):
                    results.append((cand_eval, cand_state, reason, cand_loss))
            if not results:
                return False

            best_loss = min(float(row[3]) for row in results)
            try:
                nf = float(_resolve_acceptance_noise_floor_raw(ctx.lm_hp, ctx.loss_scale))
            except Exception:
                nf = 0.0
            n_eff = _stageB_noise_n_eff()
            noise_mult = float(getattr(ctx.lm_hp, "macro_noisy_parsimony_noise_mult", 2.0) or 2.0)
            rel_tol = float(getattr(ctx.lm_hp, "macro_noisy_parsimony_rel_tol", 1.0e-3) or 1.0e-3)
            tied = [
                row
                for row in results
                if _noise_equivalent(
                    float(row[3]),
                    best_loss,
                    noise_floor=nf,
                    n_eff=n_eff,
                    noise_mult=noise_mult,
                    rel_tol=rel_tol,
                )
            ]
            if not tied:
                tied = [min(results, key=lambda row: float(row[3]))]

            def _parsimony_key(row: Tuple[Candidate, StageBState, Optional[str], float]) -> Tuple[Any, ...]:
                cand, cand_state, _reason, loss = row
                meta = _candidate_macro_meta(cand)
                clean = bool(meta.get("macro_clean_singleton", False))
                rich_c = _is_rich_macro_candidate(cand)
                try:
                    ast_cost = float(_ast_cost_physics_prior(cand_state.root))
                except Exception:
                    ast_cost = float("inf")
                return (
                    0 if clean else 1,
                    1 if rich_c else 0,
                    _rule_candidate_min_free_params(cand),
                    int(meta.get("n_terms", 99)),
                    ast_cost,
                    float(loss),
                )

            tied_sorted = sorted(tied, key=_parsimony_key)
            if len(tied_sorted) > 1:
                ctx.log(
                    "[Stage B]  Noisy macro tournament tied set: "
                    + ", ".join(
                        f"{row[0].label}(loss={_loss_str(row[3], ctx.lm_hp)}, "
                        f"clean={bool(_candidate_macro_meta(row[0]).get('macro_clean_singleton', False))}, "
                        f"terms={_candidate_macro_meta(row[0]).get('n_terms', '?')})"
                        for row in tied_sorted
                    )
                )
            for cand, cand_state, reason, _loss in tied_sorted:
                if _accept_evaluated_candidate(cand, cand_state, reason):
                    return True

            # If CoE vetoed the parsimonious tied candidate, allow the best
            # remaining evaluated macro to be considered by ordinary loss order.
            tied_cand_ids = {id(row[0]) for row in tied_sorted}
            for cand, cand_state, reason, _loss in sorted(results, key=lambda row: float(row[3])):
                if id(cand) in tied_cand_ids:
                    continue
                if _accept_evaluated_candidate(cand, cand_state, reason):
                    return True
            return False

        if _macro_noisy_parsimony_enabled() and _try_noisy_macro_tournament():
            return True

        # Sequential mode: try candidates in the order proposed by the rule.
        do_screen = (
            (not bool(exhaustive))
            and bool(screen_enable)
            and (screen_epochs > 0)
            and (len(valid) > 1)
            and (screen_epochs < int(getattr(ctx, "epochs_stageB", screen_epochs + 1)))
        )
        if (policy == "sequential") or (not do_screen):
            if not exhaustive:
                # Greedy with semi-exhaustive: stop at first accepted
                # candidate, but if that candidate is NOT a separability
                # rewrite (i.e. it's a terminal fit like sqrt_ratpoly),
                # also try remaining separability candidates — they may
                # beat it via the separability bonus in should_accept().
                def _cand_summary(c: Candidate) -> str:
                    if c.root is not None:
                        return f"{c.label}({_rule_candidate_min_free_params(c)}p)"
                    return f"{c.label}(lazy)"

                ctx.log(
                    f"[Stage B]  Greedy order ({len(valid)} cands): "
                    + ", ".join(_cand_summary(c) for c in valid)
                )
                for i, cand in enumerate(valid):
                    if _attempt_full_fit(cand):
                        _following = list(valid[i + 1 :])
                        if _are_we_done_yet(
                            ctx,
                            following_candidates=_following,
                            candidate_min_free_params_fn=_rule_candidate_min_free_params,
                        ):
                            ctx.log(
                                "[Stage B] Terminal state reached; "
                                "stopping candidate loop "
                                f"({_are_we_done_yet_reason(ctx, following_candidates=_following, candidate_min_free_params_fn=_rule_candidate_min_free_params)})."
                            )
                            return True
                        if not _is_separability_candidate(cand):
                            for later in valid[i + 1 :]:
                                if _is_separability_candidate(later):
                                    _attempt_full_fit(later)
                        return True
                return False
            # Exhaustive: try all candidates, keep the best.
            # Sort by complexity tier (ascending, stable) so simpler
            # forms are tried first.  This makes the below-floor
            # early-exit safe: once a candidate is accepted with loss
            # below floor, only separability rewrites could beat it
            # (all simpler terminal forms have already been tried).
            # Materialise lazy candidates before sorting by parameter count.
            valid_mat = [c for c in valid if c.root is not None or c.materialise()]
            valid_sorted = sorted(
                valid_mat,
                key=_rule_candidate_min_free_params,
            )
            ctx.log(
                f"[Stage B]  Exhaustive order ({len(valid_sorted)} cands): "
                + ", ".join(f"{c.label}({_rule_candidate_min_free_params(c)}p)" for c in valid_sorted)
            )

            def _accept_exhaustive_candidate(
                cand: Candidate,
                cand_state: StageBState,
                reason: Optional[str],
            ) -> bool:
                # Record accept before ctx.accept() overwrites state
                try:
                    _snap = _compact_expression_repr(cand_state.root, max_length=400)
                except Exception:
                    _snap = str(cand_state.root)
                _pre_bl = float(ctx.state.val_loss)
                _pre_bpx = list(_nn_multivar_complexity(ctx.state.root))
                try:
                    _pre_bp = int(ctx.state.model.num_parameters())
                except Exception:
                    _pre_bp = -1
                try:
                    _pre_cp = int(cand_state.model.num_parameters())
                except Exception:
                    _pre_cp = -1
                _coe_ok, _coe_reason = ctx.coe_stageB_committee_gate(
                    rule=rule_name,
                    label=cand.label,
                    reason=reason or "accepted",
                    target=_target_desc,
                    target_uid=_target_id,
                    cand=cand,
                    cand_state=cand_state,
                    n_params_base=_pre_bp,
                    n_params_cand=_pre_cp,
                )
                if not _coe_ok:
                    _log_reject(
                        cand,
                        cand_state,
                        _coe_reason or "reject-coe-stageB-gate",
                        _pre_bp,
                    )
                    ctx._rejected_keys.add((rule_name, cand.label, _target_id))
                    return False
                if _coe_reason and "accepted" in str(_coe_reason):
                    reason = f"{reason or 'accepted'}; {_coe_reason}"
                ctx._record_decision(
                    outcome="accept",
                    rule=rule_name,
                    label=cand.label,
                    reason=reason or "accepted",
                    target=_target_desc,
                    target_uid=_target_id,
                    base_loss=_pre_bl,
                    cand_loss=float(cand_state.val_loss),
                    n_params_base=_pre_bp,
                    n_params_cand=_pre_cp,
                    base_complexity=_pre_bpx,
                    cand_complexity=list(_nn_multivar_complexity(cand_state.root)),
                    cand=cand,
                    ast_snapshot=_snap,
                    base_root=ctx.state.root,
                    cand_root=cand_state.root,
                    base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                    cand_mapping_cost=_candidate_mapping_cost(cand),
                )
                _previous_root = ctx.state.root
                ctx.accept(cand, cand_state, reason or "accepted")
                if not _skip_post_accept_polish_for_terminal_state(ctx):
                    ctx.maybe_shadow_polish_subtrees_after_accept(
                        previous_root=_previous_root,
                        accepted_label=cand.label,
                    )
                    ctx.maybe_polish_after_accept()
                return True

            any_accepted = False
            below_floor_locked = False
            tiers = [
                (_tier_key, list(tier_iter))
                for _tier_key, tier_iter in groupby(
                    valid_sorted, key=_rule_candidate_min_free_params
                )
            ]
            for _tier_idx, (_tier_key, tier_cands) in enumerate(tiers):
                remaining_after_tier = [
                    c for _, later_tier in tiers[_tier_idx + 1 :] for c in later_tier
                ]

                # Evaluate all candidates in this tier against the same baseline
                tier_results = []  # List[(cand, cand_state, reason, loss)]
                for cand in tier_cands:
                    if below_floor_locked and not _is_separability_candidate(cand):
                        # Bypass the floor lock for candidates whose precheck
                        # shows a near-perfect fit *and* whose minimum visible
                        # complexity can still beat the current below-floor
                        # incumbent.  A generic ratpoly with a higher
                        # irreducible parameter count cannot win by marginally
                        # improving an already-equivalent loss.
                        _precheck_rms = cand.meta.get("precheck_rel_rms")
                        if (
                            _precheck_rms is not None
                            and _precheck_rms < 1e-3
                            and _candidate_can_beat_floor_locked_state(
                                ctx,
                                cand,
                                candidate_min_free_params_fn=_rule_candidate_min_free_params,
                            )
                        ):
                            ctx.log(
                                f"[Stage B]  Overriding floor-lock for {cand.label} "
                                f"(precheck rel_rms={_precheck_rms:.2e} < 1e-3)"
                            )
                        else:
                            _why = "baseline below floor; only simpler/separability rewrites remain"
                            if _precheck_rms is not None and _precheck_rms < 1e-3:
                                _why = (
                                    "baseline below floor; candidate minimum complexity "
                                    "cannot beat incumbent"
                                )
                            ctx.log(
                                f"[Stage B]  Skipping {cand.label} "
                                f"({_why})"
                            )
                            continue

                    result = _evaluate_candidate(cand)
                    if result is not None:
                        cand_eval, cand_state, reason = result
                        if _is_exact_final_leaf_monomial_accept(ctx, cand_eval, cand_state):
                            ctx.log(
                                f"[Stage B]    Final-leaf monomial candidate "
                                f"({cand_eval.label}) is below loss floor; checking acceptance gates."
                            )
                            if not _accept_exhaustive_candidate(cand_eval, cand_state, reason):
                                continue
                            if _are_we_done_yet(
                                ctx,
                                following_candidates=remaining_after_tier,
                                candidate_min_free_params_fn=_rule_candidate_min_free_params,
                            ):
                                ctx.log(
                                    "[Stage B] Terminal state reached; "
                                    "stopping exhaustive search "
                                    f"({_are_we_done_yet_reason(ctx, following_candidates=remaining_after_tier, candidate_min_free_params_fn=_rule_candidate_min_free_params)})."
                                )
                            return True
                        cand_loss = float(cand_state.val_loss)
                        tier_results.append((cand_eval, cand_state, reason, cand_loss))

                # Accept the best candidate from this tier
                if tier_results:
                    for cand, cand_state, reason, _loss in _rank_noisy_terminal_results(tier_results):
                        if not _accept_exhaustive_candidate(cand, cand_state, reason):
                            continue
                        any_accepted = True
                        if _are_we_done_yet(
                            ctx,
                            following_candidates=remaining_after_tier,
                            candidate_min_free_params_fn=_rule_candidate_min_free_params,
                        ):
                            ctx.log(
                                "[Stage B] Terminal state reached; "
                                "stopping exhaustive search "
                                f"({_are_we_done_yet_reason(ctx, following_candidates=remaining_after_tier, candidate_min_free_params_fn=_rule_candidate_min_free_params)})."
                            )
                            return True
                        if float(ctx.state.val_loss) <= float(ctx.loss_floor or 0.0):
                            below_floor_locked = True
                        break

            return any_accepted

        # Screening tier: cheaply rank top-K candidates.
        if screen_topk <= 0:
            screen_pool = list(valid)
        else:
            screen_pool = list(valid[: min(len(valid), screen_topk)])

        screened: List[Tuple[Candidate, float]] = []
        for cand in screen_pool:
            if cand.root is None and not cand.materialise():
                continue
            pre = ctx.precheck_candidate(rule_name, cand, record_attempt=False)
            if not pre.ok:
                _log_precheck_reject(pre, cand)
                continue
            # Skip banned candidates during screening
            _scr_skip_key = (rule_name, cand.label, _target_id)
            if _scr_skip_key in ctx._red_set or _scr_skip_key == ctx._last_amber_key:
                _skip_kind = "red_skip" if _scr_skip_key in ctx._red_set else "amber_skip"
                ctx.log(f"[Stage B]  {RED}Skipped screening{RESET} ({cand.label}): {_skip_kind}")
                ctx._record_decision(
                    outcome=_skip_kind, rule=rule_name, label=cand.label,
                    reason=f"screening_{_skip_kind}", target=_target_desc,
                    target_uid=_target_id,
                    base_loss=float(ctx.state.val_loss), cand=cand,
                )
                continue
            try:
                target_desc = rule.describe_target(target)
            except Exception:
                target_desc = "<target>"
            ctx.log(
                f"[Stage B]  Screening {cand.label} on {target_desc} "
                f"(epochs={int(screen_epochs)})"
            )
            try:
                cand_state = ctx.fit_candidate(cand, epochs_override=int(screen_epochs))
            except Exception as exc:
                ctx.log(
                    f"[Stage B]    Screen ({cand.label}) fit failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                ctx._record_decision(
                    outcome="screen_fit_exception",
                    rule=rule_name,
                    label=cand.label,
                    reason=f"screen-fit-exception:{type(exc).__name__}",
                    target=target_desc,
                    target_uid=_target_id,
                    base_loss=float(ctx.state.val_loss),
                    cand_loss=float("inf"),
                    cand=cand,
                    base_root=ctx.state.root,
                    cand_root=cand.root,
                    base_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                    cand_mapping_cost=_candidate_mapping_cost(cand),
                )
                screened.append((cand, float("inf")))
                continue
            screen_loss = float(getattr(cand_state, "val_loss", float("inf")))
            if not math.isfinite(screen_loss):
                screen_loss = float("inf")
            ctx.log(
                f"[Stage B]    Screen ({cand.label}): val-loss={_loss_str(screen_loss, ctx.lm_hp)}"
            )
            screened.append((cand, screen_loss))

        if not screened:
            # Nothing survived screening prechecks; fall back to legacy evaluation.
            for cand in valid:
                if _attempt_full_fit(cand):
                    return True
            return False

        screened.sort(key=lambda t: float(t[1]))
        best_cand, best_loss = screened[0]

        # Decide frugal vs greedy based on dominance of screened losses.
        mode = policy
        if policy == "dynamic":
            if len(screened) >= 2:
                second_loss = float(screened[1][1])
                gap_dec = 0.0
                try:
                    a = max(float(best_loss), 1e-300)
                    b = max(float(second_loss), 1e-300)
                    if math.isfinite(a) and math.isfinite(b) and (b >= a):
                        gap_dec = float(math.log10(b / a))
                except Exception:
                    gap_dec = 0.0

                if (not math.isfinite(second_loss)) and math.isfinite(best_loss):
                    gap_dec = float("inf")

                if gap_dec >= dom_dec:
                    mode = "frugal"
                    ctx.log(
                        f"[Stage B]  Screening decisive: gap={gap_dec:.2f} decades >= {dom_dec:.2f} → frugal"
                    )
                else:
                    mode = "greedy"
                    ctx.log(
                        f"[Stage B]  Screening ambiguous: gap={gap_dec:.2f} decades < {dom_dec:.2f} → greedy"
                    )
            else:
                mode = "frugal"

        if mode == "frugal":
            # Full-fit screened candidates in order; stop after first real rejection.
            for cand, _scr_loss in screened:
                result = _attempt_full_fit(cand)
                if result:
                    return True
                # If banned (not a real rejection), try next; else stop (frugal semantics)
                _fkey = (rule_name, cand.label, _target_id)
                if _fkey in ctx._red_set or _fkey == ctx._last_amber_key:
                    continue
                break  # real rejection after fit -> stop
            return False

        # Greedy: full-fit screened candidates first (best screen loss first),
        # then fall back to the rest in rule order.
        screened_ids = set(id(c) for c, _ in screened)
        full_list: List[Candidate] = [c for c, _ in screened] + [c for c in valid if id(c) not in screened_ids]

        n_attempts = 0
        for cand in full_list:
            if greedy_fullfit_max > 0 and n_attempts >= greedy_fullfit_max:
                ctx.log(
                    f"[Stage B]  Greedy full-fit cap reached ({greedy_fullfit_max}); moving on."
                )
                break
            n_attempts += 1
            if _attempt_full_fit(cand):
                return True

        return False

    def run(self, ctx: StageBContext, *, max_outer_iters: int, max_backtracks: int = 0, max_checkpoints: int = 15) -> StageBState:
        """
        Run the refinement engine: iteratively apply rules until convergence.

        Algorithm (Two-Phase):
        1. For each outer iteration (up to max_outer_iters):
           a. Phase 1: Exhaust separability rules
              - Run separability rules (additive, multiplicative) in loop
              - Restart phase when any separability rule fires
              - Continue until no separability rules fire
           b. Phase 2: Try specialized pattern rules
              - Run specialized rules once (after separability exhausted)
              - If any fires, restart from Phase 1
           c. If no rule improved, try backtracking (if budget allows), else stop
        2. Return final state

        This two-phase approach ensures we prefer simpler separable decompositions
        over complex specialized patterns (Occam's razor).

        Args:
            ctx: Execution context with state, data, hyperparameters
            max_outer_iters: Maximum number of accepted rewrites
            max_backtracks: Maximum backtrack attempts (0 to disable)
            max_checkpoints: Maximum stored checkpoints for backtracking

        Returns:
            Final StageBState after all refinements
        """
        # Initialize best_val_loss tracker from initial state
        ctx.best_val_loss = ctx.state.val_loss
        _annotate_nonsense_units_leaves(
            ctx.state,
            units_spec=getattr(ctx, "units_spec", None),
            enforce_units=bool(getattr(ctx, "enforce_units", False)),
            log_fn=ctx.log,
        )
        ctx._max_checkpoints = max(1, int(max_checkpoints))
        backtracks_remaining = max(0, int(max_backtracks))

        # Partition rules into separability (phase 1) vs specialized (phase 2)
        separability_rules = []
        specialized_rules = []

        for rule in self.rules:
            rule_name = getattr(rule, "name", type(rule).__name__)
            # Separability rules: prioritize additive and multiplicative separability
            # nn_leaf_separability: detects additive/multiplicative separability in NN atoms
            # counterterm_mul_split: detects multiplicative separability via counterterm
            # univariate_nn: converts univariate NN atoms to polynomials (structural simplification)
            # subtree_separability: checks separability of subtrees
            # log_ratio: detects additive pairs of univariate NN atoms forming log patterns
            # Policy: if an NN leaf sits inside an additive expression with
            # shared variables, overlap/addition-aware rules must get the first
            # chance to resolve the gauge before any leaf-local coordinate
            # rewrite commits.  homogeneity_peel is a useful coordinate
            # compression, but it is leaf-local and can preempt shared-structure
            # discovery in cases like NN[x0,x1,x2] + NN[x1,x2,x3].  Keep it out
            # of Phase 1: first resolve common_prefactor / overlap peels /
            # counterterms, then let Phase 2 try exact compound-function
            # closures before homogeneity becomes a fallback.
            if rule_name in (
                "nonsense_units_zero_prune",
                "monomial_peel_priority",
                "univariate_mono",
                # "product_homogeneity",    # redundant with Stage A (A<->B loop)
                "nn_leaf_separability",
                "power_product",
                "joint_product_monomial_closure",
                "common_prefactor",
                "overlap_counterterm_peel",
                "overlap_prefactor_peel",
                "additive_gauge_transfer",
                "multiplicative_homogeneity_transfer",
                "counterterm_mul_split",
                # These bounded structural closures must run before
                # univariate_nn can consume the final NN with a generic
                # rational approximant.
                "last_hard_trig_square",
                "last_hard_trig_power",
                "univariate_nn",
                "subtree_separability",
                "counterfactor_add_split",
                "log_ratio",
                # "ratio_invariance",       # redundant with Stage A (A<->B loop)
            ):
                separability_rules.append(rule)
            else:
                specialized_rules.append(rule)

        # Re-rank specialized rules: try physics-shaped closed forms before
        # more flexible/generic approximators.
        specialized_priority = {
            "compound_fn_macros": 0,
            "inverse_trig_outer_rational_closure": 0,
            "monomial_prefactor_compound": 1,
            "compound_planck": 2,
            "affine_decomp": 3,
            "nonlinear_substitution": 4,
            "univariate_oracle_invariants": 5,
            "homogeneity_peel": 6,
            # Keep flexible generalized approximators later.
            "multid_nn": 20,
            "poly_split": 21,
            "factorized_search": 30,
            "preconditioner_fallback_nn": 40,
            # True final rescue: only after all normal specialized/fallback
            # rules fail on the single remaining low-arity NN atom.
            "last_hard_trig_square": 48,
            "last_hard_trig_power": 49,
            "last_hard_atom_rescue": 50,
        }
        specialized_rules = sorted(
            specialized_rules,
            key=lambda r: specialized_priority.get(
                getattr(r, "name", type(r).__name__),
                10,
            ),
        )

        # Debug: show rule partitioning
        ctx.log(
            f"[Stage B]   Separability rules: {[getattr(r, 'name', type(r).__name__) for r in separability_rules]}"
        )
        ctx.log(
            f"[Stage B]   Specialized rules: {[getattr(r, 'name', type(r).__name__) for r in specialized_rules]}"
        )

        n_accepts = 0
        while n_accepts < max_outer_iters:
            if _are_we_done_yet(ctx):
                ctx.log(
                    "[Stage B] Terminal state reached before next pass; "
                    f"done ({_are_we_done_yet_reason(ctx)})."
                )
                break

            improved = False
            phase1_accept_count = 0
            phase1_structural_accept_count = 0
            phase1_mapping_accept_count = 0
            phase1_mapping_structural_accept_count = 0

            # Phase 1: Exhaust separability (loop until no separability rule fires)
            separability_changed = True
            while separability_changed:
                ctx.log(
                    "[Stage B] === Phase 1: Searching for separabilities (additive, multiplicative) ==="
                )
                separability_changed = False

                for rule in separability_rules:
                    rule_name = getattr(rule, "name", type(rule).__name__)
                    # Identify target nodes for this rule
                    targets_list = sorted(rule.iter_targets(ctx), key=_target_arity)
                    ctx.log(f"[Stage B] Rule {rule_name} found {len(targets_list)} targets")
                    if targets_list:
                        order_str = ", ".join(
                            f"vars={getattr(t, 'var_idxs', '?')}(arity={_target_arity(t)})"
                            for t in targets_list
                        )
                        ctx.log(f"[Stage B]   target order (arity-asc): [{order_str}]")
                    if bool(getattr(rule, "global_candidate_priority", False)) and hasattr(rule, "propose_global_candidates"):
                        try:
                            global_entries = list(rule.propose_global_candidates(ctx, targets_list) or [])
                        except Exception as exc:
                            ctx.log(f"[Stage B] Rule {rule_name} global priority failed: {exc}")
                            global_entries = []
                        if global_entries:
                            ctx.log(
                                f"[Stage B] Rule {rule_name} trying {len(global_entries)} globally ranked candidates"
                            )
                        for target, cand in global_entries:
                            if self._try_candidates_for_target(ctx, rule, rule_name, target, [cand], exhaustive=False):
                                separability_changed = True  # Restart separability phase
                                improved = True
                                phase1_accept_count += 1
                                if bool(getattr(ctx, "_last_accept_has_mapping", False)):
                                    phase1_mapping_accept_count += 1
                                    if bool(getattr(ctx, "_last_accept_mapping_structural", False)):
                                        phase1_mapping_structural_accept_count += 1
                                if bool(getattr(ctx, "_last_accept_structural", False)):
                                    phase1_structural_accept_count += 1
                                n_accepts += 1
                                break
                        if separability_changed:
                            break
                        continue
                    for target in targets_list:
                        # Generate candidate rewrites for this target
                        cands = self._propose_candidates_conjoint(ctx, rule, rule_name, target)

                        use_exhaustive = getattr(rule, "exhaustive", False) and len(targets_list) == 1
                        if self._try_candidates_for_target(ctx, rule, rule_name, target, cands, exhaustive=use_exhaustive):
                            separability_changed = True  # Restart separability phase
                            improved = True
                            phase1_accept_count += 1
                            if bool(getattr(ctx, "_last_accept_has_mapping", False)):
                                phase1_mapping_accept_count += 1
                                if bool(getattr(ctx, "_last_accept_mapping_structural", False)):
                                    phase1_mapping_structural_accept_count += 1
                            if bool(getattr(ctx, "_last_accept_structural", False)):
                                phase1_structural_accept_count += 1
                            break  # Restart separability loop

                    if separability_changed:
                        break  # Restart separability loop

            (
                run_phase2,
                phase1_only_nonstruct_accepts,
                phase1_only_nonstruct_mapping_accepts,
            ) = _phase2_trigger_flags(
                improved=improved,
                phase1_accept_count=phase1_accept_count,
                phase1_structural_accept_count=phase1_structural_accept_count,
                phase1_mapping_accept_count=phase1_mapping_accept_count,
                phase1_mapping_structural_accept_count=phase1_mapping_structural_accept_count,
            )

            # Phase 2:
            # - Standard path: run when Phase 1 made no progress.
            # - Also run when Phase 1 accepted only non-structural rewrites
            #   (e.g., approximative mapping wins), so closed-form specialized
            #   rules still get a chance.
            if run_phase2:
                if phase1_only_nonstruct_mapping_accepts:
                    ctx.log(
                        "[Stage B] === Phase 1 had only non-structural mapping-backed accepts; "
                        "running Phase 2 closed-form/specialized rules ==="
                    )
                elif phase1_only_nonstruct_accepts:
                    ctx.log(
                        "[Stage B] === Phase 1 made only non-structural accepts; "
                        "running Phase 2 closed-form/specialized rules ==="
                    )
                else:
                    ctx.log("[Stage B] === Phase 1 complete: No more separabilities found ===")
                ctx.log(
                    "[Stage B] === Phase 2: Trying specialized patterns (after exhausting separability) ==="
                )
                for rule in specialized_rules:
                    rule_name = getattr(rule, "name", type(rule).__name__)
                    # Identify target nodes for this rule
                    targets_list_p2 = sorted(rule.iter_targets(ctx), key=_target_arity)
                    if targets_list_p2:
                        order_str = ", ".join(
                            f"vars={getattr(t, 'var_idxs', '?')}(arity={_target_arity(t)})"
                            for t in targets_list_p2
                        )
                        ctx.log(f"[Stage B]   target order (arity-asc): [{order_str}]")
                    for target in targets_list_p2:
                        # Generate candidate rewrites for this target
                        cands = self._propose_candidates_conjoint(ctx, rule, rule_name, target)

                        use_exhaustive = getattr(rule, "exhaustive", False) and len(targets_list_p2) == 1
                        if self._try_candidates_for_target(ctx, rule, rule_name, target, cands, exhaustive=use_exhaustive):
                            improved = True
                            break  # Go back to Phase 1 (outer loop restart)

                    if improved:
                        break  # Go back to Phase 1 (outer loop restart)

            # Count accepted rewrites (backtracks don't consume the budget)
            if improved:
                n_accepts += 1
                if _are_we_done_yet(ctx):
                    ctx.log(
                        "[Stage B] Terminal state reached after accept; "
                        f"done ({_are_we_done_yet_reason(ctx)})."
                    )
                    break
                # Fallback early exit: no NN atoms remain, but the expression
                # still has parameters or is above the terminal floor.  There
                # are no Stage-B NN targets left, so hand off to final polish /
                # reporting rather than spinning.
                if ctx.state.num_nn_atoms is not None and ctx.state.num_nn_atoms == 0:
                    ctx.log(
                        "[Stage B] All NN atoms rewritten; expression is fully analytical. "
                        "Stopping Stage-B rewrites."
                    )
                    break
                continue

            # If no rule improved in either phase, try backtracking or stop
            if not improved:
                if backtracks_remaining > 0 and ctx._checkpoints:
                    # Amber → Red promotion: if no accepts happened since last
                    # backtrack, the amber key is confirmed dead.
                    if ctx._last_amber_key is not None and ctx._accepts_since_backtrack == 0:
                        ctx._red_set.add(ctx._last_amber_key)
                        ctx.log(f"[Stage B]   Amber → Red: {ctx._last_amber_key!r}")
                        ctx._last_amber_key = None

                    worst_idx = _find_worst_accept(ctx)
                    if worst_idx is not None:
                        ckpt = ctx._checkpoints[worst_idx]
                        backtracks_remaining -= 1

                        # Set amber key (tentatively skipped; forgiven on next accept)
                        _amber_key = (ckpt.accept_rule, ckpt.accept_label, ckpt.accept_target_uid)
                        ctx._last_amber_key = _amber_key
                        ctx._accepts_since_backtrack = 0

                        # Restore pre-accept state from lightweight checkpoint
                        ctx.state = _restore_from_checkpoint(ctx, ckpt)
                        ctx.enabled_patterns = list(ckpt.enabled_patterns)
                        ctx.best_val_loss = ckpt.best_val_loss
                        ctx.has_structural = ckpt.has_structural
                        # Note: _decision_step is NOT reset (Issue 5: keep monotonic)
                        # Note: attempted_transformations is NOT rolled back (Issue 4: global dedup)
                        ctx._cache.clear()
                        ctx._rejected_keys.clear()
                        ctx._dim_cache = {}
                        ctx._dim_cache_root_id = None

                        # Discard checkpoints from reverted accept onward
                        ctx._checkpoints = ctx._checkpoints[:worst_idx]

                        # Record backtrack event with evidence from reverted accept
                        rec_idx = ckpt.decision_log_len
                        if rec_idx < len(ctx.decision_log):
                            reverted_rec = ctx.decision_log[rec_idx]
                            reverted_base = reverted_rec.get("base_loss")
                            reverted_cand = reverted_rec.get("cand_loss")
                        else:
                            reverted_base = None
                            reverted_cand = None

                        ctx._record_decision(
                            outcome="backtrack",
                            rule=ckpt.accept_rule,
                            label=ckpt.accept_label,
                            reason=f"backtrack(amber={_amber_key!r})",
                            target=ckpt.accept_target,
                            target_uid=ckpt.accept_target_uid,
                            base_loss=float(reverted_base) if reverted_base is not None else float(ctx.state.val_loss),
                            cand_loss=float(reverted_cand) if reverted_cand is not None else float("nan"),
                        )

                        ctx.log(
                            f"[Stage B]   Backtrack: reverted accept ({ckpt.accept_label}), "
                            f"restored to loss={ctx.state.val_loss:.3e} "
                            f"(budget={backtracks_remaining} remaining)"
                        )
                        continue  # restart while-loop (no accept counted)

                ctx.log("[Stage B] No improving rewrites found; stopping.")
                break

        # Restore best-seen state if it's better than what we ended with.
        #
        # Subtlety: when both states are below the meaningful loss floor, loss
        # differences are mostly noise. In that regime we prefer the *simpler*
        # state (fewer parameters / less NN structure), matching should_accept().
        if ctx._best_seen is not None and ctx._best_seen.val_loss < ctx.state.val_loss:
            cur_loss = float(ctx.state.val_loss)
            best_loss = float(ctx._best_seen.val_loss)
            try:
                loss_floor = float(ctx.loss_floor) if ctx.loss_floor is not None else float(ctx.loss_good_enough_raw)
            except Exception:
                loss_floor = 0.0
            try:
                floor_guard_dec = float(getattr(ctx.lm_hp, "select_floor_guard_decades", 2.0))
            except Exception:
                floor_guard_dec = 2.0
            acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(ctx.lm_hp, ctx.loss_scale)
            cur_loss_cmp = _loss_excess_above_floor(cur_loss, acceptance_noise_floor_raw)
            best_loss_cmp = _loss_excess_above_floor(best_loss, acceptance_noise_floor_raw)
            try:
                best_ref = float(min(cur_loss_cmp, best_loss_cmp))
            except Exception:
                best_ref = float(cur_loss_cmp)
            loss_floor_eff = _effective_loss_floor(loss_floor, best_ref, floor_guard_dec)

            try:
                count_weight = float(getattr(ctx.lm_hp, "select_count_weight", 1.0))
            except Exception:
                count_weight = 1.0

            try:
                n_params_cur = int(ctx.state.model.num_parameters())
            except Exception:
                try:
                    n_params_cur = int(
                        sum(int(p.numel()) for p in ctx.state.model.parameters())
                    )
                except Exception:
                    n_params_cur = int(1e18)

            n_params_best = getattr(ctx._best_seen, "n_params", None)
            try:
                n_params_best = int(n_params_best) if n_params_best is not None else None
            except Exception:
                n_params_best = None
            if n_params_best is None:
                # Conservative fallback: count tensors in the saved state_dict.
                try:
                    n_params_best = int(
                        sum(
                            int(v.numel())
                            for v in ctx._best_seen.model_state_dict.values()
                            if torch.is_tensor(v)
                        )
                    )
                except Exception:
                    n_params_best = int(1e18)

            restore, reason = _best_seen_restore_decision(
                cur_loss=cur_loss_cmp,
                best_loss=best_loss_cmp,
                cur_root=ctx.state.root,
                best_root=ctx._best_seen.root,
                n_params_cur=n_params_cur,
                n_params_best=n_params_best,
                loss_floor=loss_floor,
                loss_floor_eff=loss_floor_eff,
                count_weight=float(count_weight),
                cur_mapping_cost=getattr(ctx.state, "complexity_mapping_cost", 0.0),
                best_mapping_cost=getattr(ctx._best_seen, "complexity_mapping_cost", 0.0),
                losses_noise_equivalent=bool(
                    acceptance_noise_floor_raw > 0.0
                    and _noise_equivalent(
                        cur_loss,
                        best_loss,
                        noise_floor=acceptance_noise_floor_raw,
                        n_eff=getattr(ctx.state, "acceptance_noise_n_eff", None)
                        or getattr(ctx, "acceptance_noise_n_eff", None)
                        or getattr(ctx._best_seen, "acceptance_noise_n_eff", None),
                    )
                ),
            )

            if restore:
                ctx.log(f"[Stage B]   Restoring best-seen state ({reason})")
                try:
                    ctx.state = _restore_from_checkpoint(ctx, ctx._best_seen)
                    ctx.enabled_patterns = list(ctx._best_seen.enabled_patterns)
                except Exception as exc:
                    ctx.log(
                        "[Stage B]   Best-seen restore failed; keeping current "
                        f"state ({type(exc).__name__}: {exc})"
                    )
            else:
                ctx.log(f"[Stage B]   Keeping current state ({reason})")

        if ctx.verbose:
            try:
                pareto_topk = int(getattr(ctx.lm_hp, "stageB_pareto_log_topk", 8))
            except Exception:
                pareto_topk = 8
            try:
                ctx.log_pareto_summary(max_records=max(1, pareto_topk))
            except Exception as e:
                ctx.log(f"[Stage B] Pareto summary failed: {e}")

        return ctx.state
