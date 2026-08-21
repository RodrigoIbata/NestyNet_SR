# ruff: noqa: F401
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""State containers, candidates, and execution context for Stage B."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from nestynet_sr.sr_core.coefficient_metadata import collect_coefficient_metadata
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

class StageBRule:
    """
    Base class for Stage B rewrite rules.

    Each rule must implement:
    - iter_targets(ctx): Return an iterable of nodes that are candidates for this rule
    - propose(ctx, target): Generate a list of Candidate rewrites for a target node
    - describe_target(target): Return a human-readable description of the target (optional)

    Attributes:
        name: Human-readable name for the rule (class attribute)
    """

    name: str = "<unnamed>"
    exhaustive: bool = False
    # If True, rule.propose() already handles multi-dataset probing internally
    # and should be called once on the full context (no per-dataset probe views).
    multi_probe_native: bool = False

    def iter_targets(self, ctx: Any) -> Any:  # (StageBContext) -> Iterable[Node]
        """
        Identify target nodes in the AST that this rule can rewrite.

        Args:
            ctx: Stage B context with current state

        Returns:
            Iterable of Node objects (typically AtomNode instances)
        """
        raise NotImplementedError

    def propose(
        self, ctx: Any, target: Node
    ) -> List[Any]:  # (StageBContext, Node) -> List[Candidate]
        """
        Generate rewrite candidates for a target node.

        Args:
            ctx: Stage B context with state, data, hyperparameters
            target: Node to rewrite

        Returns:
            List of Candidate rewrites (may be empty if no valid rewrites)
        """
        raise NotImplementedError

    def describe_target(self, target: Node) -> str:
        """
        Return a human-readable description of the target node.

        Args:
            target: Node to describe

        Returns:
            String description (e.g., "nn vars=(0, 1)")
        """
        if isinstance(target, AtomNode):
            tag = getattr(target, "tag", None)
            tag_s = f"#{tag}" if tag else ""
            return f"{target.kind}{tag_s} vars={tuple(int(j) for j in target.var_idxs)}"
        return type(target).__name__

    def candidate_min_free_params(self, cand: "Candidate") -> int:
        """Published lower bound on candidate fitted free parameters.

        Rule implementations may override this when the minimum degrees of
        freedom are not faithfully represented by the candidate AST.  The
        default uses candidate metadata (``min_free_params`` / legacy
        ``n_free_params``), then falls back to the AST atom parameter count.
        """

        return _candidate_min_free_params(cand)


@dataclass
class StageBState:
    """
    Represents the state of Stage B refinement at a given point.

    Attributes:
        root: AST root node representing the expression structure
        model: Fitted PyTorch model corresponding to the AST
        reuse: Map from tag to reusable modules (e.g., trained NN leaves)
        val_loss: Validation loss of the current model
        phi_expr_str: Human-readable expression string (optional)
        y_expr_str: Expression in original y-space if transform applied (optional)
        sympy_meta: Metadata from SymPy simplification (optional)
        coefficient_metadata: Versioned scalar coefficient identity/value/unit records
        coefficient_metadata_by_dataset: Per-dataset records for joint fits
        enabled_patterns: List of rewrite patterns that were accepted (optional)
        num_nn_atoms: Number of NN atoms remaining in the AST (optional)
        num_multivar_nn_atoms: Number of multivariate NN atoms (len(var_idxs) > 1) (optional)
        max_nn_arity: Maximum arity across all NN atoms (optional)
    """

    root: Node
    model: torch.nn.Module
    reuse: Dict[str, torch.nn.Module]
    val_loss: float
    # Multi-dataset support (optional)
    models: Optional[List[torch.nn.Module]] = None
    reuses: Optional[List[Dict[str, torch.nn.Module]]] = None
    val_losses: Optional[List[float]] = None
    dataset_ids: Optional[List[str]] = None
    agg_mode: Optional[str] = None
    agg_weights: Optional[List[float]] = None
    phi_expr_str: Optional[str] = None
    y_expr_str: Optional[str] = None
    sympy_meta: Optional[dict] = None
    coefficient_metadata: Optional[dict] = None
    coefficient_metadata_by_dataset: Optional[List[dict]] = None
    enabled_patterns: Optional[list] = None
    num_nn_atoms: Optional[int] = None
    num_multivar_nn_atoms: Optional[int] = None
    max_nn_arity: Optional[int] = None
    problem_leaves: Optional[List[dict]] = None
    loss_scale: Optional[float] = None
    loss_good_enough_eff: Optional[float] = None
    loss_acceptable_eff: Optional[float] = None
    acceptance_noise_floor_raw: Optional[float] = None
    acceptance_noise_n_eff: Optional[float] = None
    original_y_val_loss: Optional[float] = None
    original_y_loss_good_enough_eff: Optional[float] = None
    original_y_loss_acceptable_eff: Optional[float] = None
    coe_stageB_dry_run_log: Optional[List[dict]] = None
    coe_stageB_gate_log: Optional[List[dict]] = None

    # Persistent decision log (Layer 1 of backtracking infrastructure)
    decision_log: Optional[List[dict]] = None

    # Optional x-coordinate transform metadata (Stage A -> Stage B)
    x_transform_map: Optional[dict] = None
    phi_expr_raw_str: Optional[str] = None
    y_expr_raw_str: Optional[str] = None
    # Optional explicit complexity contribution for non-AST artifacts such as
    # factorized symbolic search output mappings (poly/Padé/etc.).
    complexity_mapping_cost: float = 0.0

    # Curated user-facing timeline of successful transformation steps
    simplification_path: list = field(default_factory=list)  # List[dict]


@dataclass
class _Checkpoint:
    """Lightweight snapshot for backtracking (no full model copies).

    Stores the AST (deep-copied) and model state_dict(s) on CPU instead of
    ``copy.deepcopy(StageBState)`` to avoid cloning entire ``nn.Module`` graphs.
    """
    root: Node                      # deep-copied AST
    val_loss: float
    model_state_dict: Dict[str, Any]   # CPU state_dict
    reuse_state_dicts: Optional[List[Dict[str, Any]]]  # multi-dataset: per-model CPU state_dicts
    enabled_patterns: List[str]
    best_val_loss: float
    has_structural: bool
    decision_log_len: int           # index of the accept record in decision_log
    decision_step: int              # _decision_step before accept
    attempted_transformations: Dict[str, Set[Tuple[int, ...]]]
    # Identity of the accept being checkpointed
    accept_step: int
    accept_rule: str
    accept_label: str
    accept_target: str
    accept_target_uid: str = ""   # path-based UID for skip-key matching
    n_params: Optional[int] = None    # cached trainable parameter count
    complexity_mapping_cost: float = 0.0
    simplification_path: List[dict] = field(default_factory=list)
    acceptance_noise_floor_raw: float = 0.0
    acceptance_noise_n_eff: Optional[float] = None
    generic_approximant_unpromoted: bool = False


def _materialized_fit_state_for_checkpoint(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Build full segmented fit views without mutating ``model``.

    NestyNet's per-segment fixed pieces are ordinary tensors and therefore are
    absent from ``state_dict()``.  During block fitting, only a subset of a
    parameter group may be registered as ``*_fit``; those fitted coordinates
    are authoritative while the other coordinates still live in the fixed
    pieces.  A restorable checkpoint consequently needs the *effective* full
    group, not either store in isolation.

    Fresh NestyNet leaves expose full ``*_fit`` Parameters, so materialize the
    effective values under those state-dict keys.  ``get_parameters()`` is the
    read-only NestyNet operation that overlays fitted coordinates on fixed-only
    coordinates.  ``remove_duplicate=False`` also covers registered aliases
    such as ``base_model`` and ``G_Model`` that both appear in ``state_dict()``.
    """

    materialized: Dict[str, torch.Tensor] = {}
    effective_by_module: Dict[int, Tuple[Optional[torch.Tensor], ...]] = {}
    try:
        named_modules = model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        named_modules = model.named_modules()

    with torch.no_grad():
        for prefix, module in named_modules:
            groups = tuple(getattr(module, "_parameter_group_names", ()))
            get_parameters = getattr(module, "get_parameters", None)
            if groups != ("a", "b", "c", "K") or not callable(get_parameters):
                continue

            module_id = id(module)
            full_groups = effective_by_module.get(module_id)
            if full_groups is None:
                effective_groups = tuple(get_parameters())
                if len(effective_groups) != len(groups):
                    raise RuntimeError(
                        f"Segmented module {prefix or '<root>'!r} returned "
                        f"{len(effective_groups)} parameter groups; expected {len(groups)}"
                    )
                full_group_values: List[Optional[torch.Tensor]] = []
                for pieces in effective_groups:
                    group_pieces = tuple(pieces)
                    full_group_values.append(
                        torch.cat([piece.reshape(-1) for piece in group_pieces])
                        if group_pieces
                        else None
                    )
                full_groups = tuple(full_group_values)
                effective_by_module[module_id] = full_groups

            for group, full in zip(groups, full_groups):
                if full is None:
                    continue
                key = f"{prefix}.{group}_fit" if prefix else f"{group}_fit"
                materialized[key] = full.detach().cpu().clone()

    return materialized


def _checkpoint_state_dict_cpu(model: torch.nn.Module) -> Dict[str, Any]:
    """Return a self-contained CPU checkpoint without changing ``model``."""

    state_dict = {
        k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
        for k, v in model.state_dict().items()
    }
    state_dict.update(_materialized_fit_state_for_checkpoint(model))
    return state_dict


_TRANSIENT_FIT_STATE_SUFFIXES = (".a_fit", ".b_fit", ".c_fit", ".K_fit")


def _is_transient_fit_state_key(key: str) -> bool:
    key_s = str(key)
    return key_s in {"a_fit", "b_fit", "c_fit", "K_fit"} or key_s.endswith(
        _TRANSIENT_FIT_STATE_SUFFIXES
    )


def _state_value_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def _load_checkpoint_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    label: str = "checkpoint",
) -> None:
    """Load a Stage-B checkpoint while tolerating transient fit-key drift.

    Current snapshots store complete effective segmented groups under ``*_fit``
    keys.  Older checkpoints, however, can differ from the rebuilt model's lazy
    fit-key set and may therefore miss keys such as
    ``leaf.3._stage0.model.c_fit`` even though the rebuilt model now exposes
    them.  Retain compatibility with those legacy snapshots by filling missing
    transient keys from the rebuilt model and removing transient extras; keep
    all non-transient mismatches strict.  New checkpoint creation must not rely
    on this fallback because fitted coordinates can be authoritative.
    """

    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        current = model.state_dict()
        missing = sorted(k for k in current.keys() if k not in state_dict)
        unexpected = sorted(k for k in state_dict.keys() if k not in current)
        if not missing and not unexpected:
            raise
        bad_missing = [k for k in missing if not _is_transient_fit_state_key(k)]
        bad_unexpected = [k for k in unexpected if not _is_transient_fit_state_key(k)]
        if bad_missing or bad_unexpected:
            raise

        patched: Dict[str, Any] = {
            k: _state_value_clone(v)
            for k, v in state_dict.items()
            if k in current
        }
        for key in missing:
            patched[key] = _state_value_clone(current[key])
        model.load_state_dict(patched, strict=True)
        if callable(log_fn):
            log_fn(
                f"[Stage B]   Loaded {label} with transient fit-key repair "
                f"(filled={len(missing)}, dropped={len(unexpected)})"
            )


@dataclass
class Candidate:
    """
    Represents a candidate rewrite proposal.

    Attributes:
        label: Human-readable name for the rewrite pattern
        root: New AST root after applying the rewrite
        init_fn: Optional custom initialization function for the new model
        meta: Additional metadata (e.g., logging messages)
        signature: Optional deduplication signature for this candidate.
            If provided, StageBEngine will use it (together with the producing rule name)
            to avoid re-attempting identical transformations across restarts.
        builder: Optional deferred builder callable.  When provided, root
            may be None; the builder is invoked lazily the first time the
            candidate is evaluated by the engine (see ``materialise``).
    """

    label: str
    root: Optional[Node] = None
    init_fn: Optional[Callable] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[Tuple[int, ...]] = None
    builder: Optional[Callable] = None

    def materialise(self) -> bool:
        """Evaluate a deferred builder, populating root/init_fn/meta.

        Returns True if the builder produced a valid candidate, False otherwise.
        No-op if root is already set or no builder is present.
        """
        if self.root is not None or self.builder is None:
            return self.root is not None
        res = self.builder()
        self.builder = None  # run at most once
        if res is None:
            return False
        if isinstance(res, tuple) and len(res) == 3:
            self.root, self.init_fn, extra_meta = res
        elif isinstance(res, tuple) and len(res) == 2:
            self.root, self.init_fn = res
            extra_meta = {}
        else:
            return False
        if self.root is None:
            return False
        if isinstance(extra_meta, dict):
            if "_label" in extra_meta:
                self.label = extra_meta.pop("_label")
            self.meta.update(extra_meta)
        return True


@dataclass(frozen=True)
class PrecheckResult:
    """Result of a cheap, static candidate precheck.

    This is the intended extension point for future constraints (e.g. unit
    consistency, domain constraints, etc.) so that expensive LM fits are only
    run on candidates that pass fast filters.
    """

    ok: bool
    reason: Optional[str] = None
    signature: Optional[Tuple[int, ...]] = None


@dataclass
class StageBContext:
    """
    Execution context for Stage B refinement engine.

    Holds all state and hyperparameters needed to evaluate and accept/reject
    rewrite candidates. Provides helper methods for fitting, acceptance logic,
    and logging.

    Attributes:
        state: Current Stage B state (AST, model, loss)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        lm_hp: Levenberg-Marquardt hyperparameters
        device: torch device (cuda/cpu)
        dtype: torch dtype (float64)
        epochs_stageB: Max LM epochs for each candidate fit
        loss_scale: Scaling factor for loss (e.g., MAD-based)
        loss_good_enough_raw: Target loss threshold (unscaled)
        score_tol: Minimum improvement required to accept strict-better rewrites
        scale_specs: Discovered scaling features from Stage A
        scaling_by_axis: Map from axis index to ScaleSpec list
        trig_by_axis: Map from axis index to TrigAxisSpec
        phase_hints: Stage-0 phase-coordinate hints; proposal evidence only
        phase_context_hints: Stage-0 contextual phase hints; proposal evidence only
        outer_link_hints: inside-out inverse-link hints; proposal evidence only
        stageA_x_transforms: Optional Stage-A x-preprocessing map (axis -> spec dict)
        trig_structure_by_axis: Second-stage trig hints (product/difference structure)
        verbose: Enable detailed logging
        fresh_nn_factory: Factory for creating new NN leaves
        atom_factory: Optional factory (or list of factories) for non-NN atom leaves
        disabled_patterns: Set of pattern labels to skip
        enabled_patterns: List of pattern labels that were accepted
        _cache: Internal cache for expensive computations
    """

    state: StageBState
    train_loader: Any
    val_loader: Any
    lm_hp: Any  # LMHyperparams
    device: torch.device
    dtype: torch.dtype
    epochs_stageB: int
    loss_scale: float
    loss_good_enough_raw: float
    score_tol: float
    scale_specs: List[Any]  # List[ScaleSpec]
    scaling_by_axis: Dict[int, List[Any]]  # Dict[int, List[ScaleSpec]]
    trig_by_axis: Dict[int, Any]  # Dict[int, TrigAxisSpec]
    phase_hints: List[Any] = field(default_factory=list)
    phase_context_hints: List[Any] = field(default_factory=list)
    outer_link_hints: List[Any] = field(default_factory=list)
    stageA_x_transforms: Dict[int, Any] = field(default_factory=dict)
    # If stageA_x_transforms are applied directly to the dataloaders, this
    # holds the resulting XCoordSystem and flips xcoords_applied=True.
    xcoords: Any = None
    xcoords_applied: bool = False
    loss_scales: Optional[List[float]] = None
    dataset_ids: Optional[List[str]] = None
    agg_mode: str = "mean"
    agg_weights: Optional[List[float]] = None
    trig_structure_by_axis: Dict[int, Any] = field(default_factory=dict)
    verbose: bool = True
    fresh_nn_factory: Any = None
    atom_factory: Any = None
    disabled_patterns: Set[str] = field(default_factory=set)
    enabled_patterns: List[str] = field(default_factory=list)
    _cache: Dict[Tuple[Any, ...], Any] = field(default_factory=dict)
    best_val_loss: float = field(default=float("inf"))
    has_structural: bool = field(default=False)
    acceptance_noise_n_eff: Optional[float] = None
    coe_stageB_dry_run: bool = False
    coe_stageB_dry_run_log: List[dict] = field(default_factory=list)
    coe_mode: str = "off"
    coe_filepath: Optional[str] = None
    coe_num_slices: int = 25
    coe_start_slice: int = 0
    coe_reference_slice: Optional[int] = None
    coe_stageB_initial_gate_slices: int = 3
    coe_stageB_gate_slices: int = 5
    coe_ndata_train: int = 2000
    coe_ndata_val: int = 2000
    coe_noise_floor_raw: float = 0.0
    coe_noise_mult: float = 3.0
    coe_rel_tol: float = 1.0e-3
    coe_inference: str = "legacy"
    coe_maxt_seed: int = 0
    coe_min_valid_fraction: float = 0.80
    coe_witness_parallelism: int = 1
    coe_stageB_refit_gate: bool = True
    coe_stageB_refit_epochs: int = 200
    coe_stageB_refit_escalate_epochs: int = 0
    coe_stageB_gate_log: List[dict] = field(default_factory=list)
    coe_eval_cache: Any = None
    # Selection floors/caps.
    #
    # loss_floor: below this, loss differences are treated as noise; prefer simpler.
    # worsening_floor: used to avoid over-tightening when the reference loss is tiny.
    # loss_cap: hard ceiling on loss for accepted candidates in Stage B.
    worsening_floor: Optional[float] = None
    loss_floor: Optional[float] = None
    loss_cap: Optional[float] = None
    y_op: Any = None  # Optional y-transform for branch-space training data.
    y_op_inv: Any = None  # Optional inverse y-transform for display (e.g., sqrt)
    y_transform_name: str = "identity"

    # Optional dimensional-analysis straightjacket (PhySO-like).
    # If provided and enforce_units=True, Stage B will reject candidates that
    # are dimensionally inconsistent *before* launching an LM fit.
    units_spec: Any = None  # sr_core.units.UnitsSpec | None
    enforce_units: bool = False
    verbose_separabilities: bool = False

    # Deduplication registry: prevents re-attempting identical transformations
    attempted_transformations: Dict[str, Set[Tuple[int, ...]]] = field(default_factory=dict)

    # Lightweight rejected-candidate dedup: prevents re-fitting candidates
    # that were already rejected on the same (rule, label, target_var_idxs)
    # triple within the current AST.  Cleared on accept() since the AST
    # change may make previously-rejected rewrites viable.
    _rejected_keys: Set[tuple] = field(default_factory=set)

    # Persistent decision log: records every accept/reject/precheck decision
    decision_log: List[dict] = field(default_factory=list)
    _decision_step: int = field(default=0)

    # Layer 2: Checkpoint + backtrack
    _checkpoints: List[Any] = field(default_factory=list)
    _max_checkpoints: int = field(default=15)
    _red_set: Set[Tuple[str, str, str]] = field(default_factory=set)     # (rule, label, target) — permanently banned
    _last_amber_key: Optional[Tuple[str, str, str]] = field(default=None)  # single most-recent backtracked key (skipped until next accept)
    _accepts_since_backtrack: int = field(default=0)                        # accepts since last backtrack; 0 → promote amber to red
    _best_seen: Optional[Any] = field(default=None)  # lightweight snapshot of best-ever state
    _last_accept_structural: bool = field(default=False)
    _last_accept_has_mapping: bool = field(default=False)
    _last_accept_mapping_structural: bool = field(default=False)

    """
    Registry of attempted transformation signatures to prevent duplicates.

    Key: rule name (e.g., "counterterm_mul_split")
    Value: Set of signatures for this rule

    Signature format (varies by rule):
      counterterm_mul_split: (atom_hash, partition_hash, degA, degB, variant_hash)
      counterfactor_add_split: (atom_hash, partition_hash, degA, degB)

    Persists across entire Stage B session to prevent infinite restart loops.
    """

    def __post_init__(self):
        """Derive selection floors/caps from the loss scale + LM hyperparams."""
        self.coe_mode = str(getattr(self.lm_hp, "coe_mode", self.coe_mode) or "off")
        if bool(getattr(self.lm_hp, "coe_stageB_dry_run", False)):
            self.coe_stageB_dry_run = True
        if self.coe_mode in {"committee_gated", "reservoir_discovery"}:
            self.coe_stageB_dry_run = True
        self.coe_filepath = getattr(self.lm_hp, "coe_filepath", self.coe_filepath)
        self.coe_num_slices = max(
            0, int(getattr(self.lm_hp, "coe_num_slices", self.coe_num_slices) or 0)
        )
        self.coe_start_slice = max(
            0, int(getattr(self.lm_hp, "coe_start_slice", self.coe_start_slice) or 0)
        )
        _coe_ref = getattr(self.lm_hp, "coe_reference_slice", self.coe_reference_slice)
        if _coe_ref is None:
            self.coe_reference_slice = None
        else:
            try:
                self.coe_reference_slice = max(0, int(_coe_ref))
            except Exception:
                self.coe_reference_slice = None
        self.coe_stageB_gate_slices = max(
            0,
            int(
                getattr(
                    self.lm_hp,
                    "coe_stageB_gate_slices",
                    self.coe_stageB_gate_slices,
                )
                or 0
            ),
        )
        self.coe_stageB_initial_gate_slices = max(
            1,
            int(
                getattr(
                    self.lm_hp,
                    "coe_stageB_initial_gate_slices",
                    self.coe_stageB_initial_gate_slices,
                )
                or 1
            ),
        )
        self.coe_ndata_train = max(
            1, int(getattr(self.lm_hp, "coe_ndata_train", self.coe_ndata_train) or 1)
        )
        self.coe_ndata_val = max(
            1, int(getattr(self.lm_hp, "coe_ndata_val", self.coe_ndata_val) or 1)
        )
        self.coe_noise_floor_raw = _clamp_nonnegative_finite(
            getattr(self.lm_hp, "coe_noise_floor_raw", self.coe_noise_floor_raw),
            default=0.0,
        )
        self.coe_noise_mult = max(
            0.0, float(getattr(self.lm_hp, "coe_noise_mult", self.coe_noise_mult) or 0.0)
        )
        self.coe_rel_tol = max(
            0.0, float(getattr(self.lm_hp, "coe_rel_tol", self.coe_rel_tol) or 0.0)
        )
        self.coe_inference = str(
            getattr(self.lm_hp, "coe_inference", self.coe_inference) or "legacy"
        )
        self.coe_maxt_seed = int(getattr(self.lm_hp, "coe_maxt_seed", self.coe_maxt_seed) or 0)
        self.coe_min_valid_fraction = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.lm_hp,
                        "coe_min_valid_fraction",
                        self.coe_min_valid_fraction,
                    )
                    or 0.0
                ),
            ),
        )
        self.coe_witness_parallelism = max(
            1,
            int(getattr(self.lm_hp, "coe_witness_parallelism", self.coe_witness_parallelism) or 1),
        )
        self.coe_stageB_refit_gate = bool(
            getattr(self.lm_hp, "coe_stageB_refit_gate", self.coe_stageB_refit_gate)
        )
        self.coe_stageB_refit_epochs = max(
            1,
            int(
                getattr(
                    self.lm_hp,
                    "coe_stageB_refit_epochs",
                    self.coe_stageB_refit_epochs,
                )
                or 1
            ),
        )
        self.coe_stageB_refit_escalate_epochs = max(
            0,
            int(
                getattr(
                    self.lm_hp,
                    "coe_stageB_refit_escalate_epochs",
                    self.coe_stageB_refit_escalate_epochs,
                )
                or 0
            ),
        )
        # A meaningful loss floor is the Stage-A/B target (in raw loss units).
        if self.loss_floor is None:
            try:
                self.loss_floor = float(self.loss_good_enough_raw)
            except Exception:
                self.loss_floor = 0.0

        if (self.loss_floor is None) or (not math.isfinite(float(self.loss_floor))) or float(self.loss_floor) < 0:
            self.loss_floor = 0.0

        # Below the floor, ratios become meaningless; treat the baseline as "excellent".
        if self.worsening_floor is None:
            self.worsening_floor = float(self.loss_floor)

        if (self.worsening_floor is None) or (not math.isfinite(float(self.worsening_floor))) or float(self.worsening_floor) < 0:
            self.worsening_floor = float(self.loss_floor)

        # Stage-B hard ceiling: never accept candidates that regress by more than
        # a bounded number of decades above the meaningful floor.
        if self.loss_cap is None:
            try:
                max_dec = float(getattr(self.lm_hp, "select_stageB_max_decades_over_floor", 1.0))
                max_dec = max(0.0, max_dec)
                cap_from_floor = float(self.loss_floor) * (10.0 ** max_dec)
            except Exception:
                cap_from_floor = float("inf")

            try:
                loss_acceptable_raw = float(getattr(self.lm_hp, "loss_acceptable", 1.0e-3)) * float(self.loss_scale)
            except Exception:
                loss_acceptable_raw = float("inf")

            self.loss_cap = float(min(loss_acceptable_raw, cap_from_floor))

    @property
    def train_loader_probe(self):
        """Return a single DataLoader suitable for data-gathering / probing.

        In multi-dataset mode ``train_loader`` is a list of DataLoaders;
        probing rules only need one representative dataset (the first).
        """
        tl = self.train_loader
        if isinstance(tl, (list, tuple)):
            return tl[0]
        return tl

    @property
    def train_loader_probes(self):
        """Return the list of DataLoaders available for probing.

        In single-dataset mode this returns a one-element list.
        In multi-dataset mode this returns all dataset loaders.
        """
        tl = self.train_loader
        if isinstance(tl, (list, tuple)):
            return list(tl)
        return [tl]

    def cached(self, key: Tuple[Any, ...], compute_fn):
        """Simple memoization for expensive probes (key should include node identity)."""
        if key in self._cache:
            return self._cache[key]
        v = compute_fn()
        self._cache[key] = v
        return v

    def additive_gauge_index(self, root: Optional[Node] = None) -> AdditiveGaugeScopeIndex:
        """Return the transient additive-gauge sidecar for *root*."""
        root_eff = self.state.root if root is None else root
        if root is self.state.root or root is None:
            return self.cached(
                ("additive_gauge_scope_index", id(root_eff)),
                lambda: AdditiveGaugeScopeIndex(root_eff),
            )
        return AdditiveGaugeScopeIndex(root_eff)

    def additive_gauge_global_score(self, root: Optional[Node] = None) -> AdditiveGaugeGlobalScore:
        """Return a lexicographic unresolved-gauge score for *root*."""
        root_eff = self.state.root if root is None else root
        if root is self.state.root or root is None:
            idx = self.additive_gauge_index(root_eff)
            return self.cached(
                ("additive_gauge_global_score", id(root_eff)),
                lambda: idx.global_score(),
            )
        return additive_gauge_global_score(root_eff)

    def homogeneous_gauge_index(self, root: Optional[Node] = None) -> HomogeneousGaugeScopeIndex:
        """Return the transient homogeneous-gauge sidecar for *root*."""
        root_eff = self.state.root if root is None else root
        if root is self.state.root or root is None:
            return self.cached(
                ("homogeneous_gauge_scope_index", id(root_eff)),
                lambda: HomogeneousGaugeScopeIndex(root_eff),
            )
        return HomogeneousGaugeScopeIndex(root_eff)

    def homogeneous_gauge_global_score(self, root: Optional[Node] = None) -> HomogeneousGaugeGlobalScore:
        """Return a lexicographic unresolved homogeneous-gauge score for *root*."""
        root_eff = self.state.root if root is None else root
        if root is self.state.root or root is None:
            idx = self.homogeneous_gauge_index(root_eff)
            return self.cached(
                ("homogeneous_gauge_global_score", id(root_eff)),
                lambda: idx.global_score(),
            )
        return homogeneous_gauge_global_score(root_eff)

    def log(self, s: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(s)

    def _record_decision(
        self,
        *,
        outcome: str,
        rule: str,
        label: str,
        reason: str,
        target: str,
        target_uid: str = "",
        base_loss: float,
        cand_loss: float = float("nan"),
        n_params_base: int = -1,
        n_params_cand: int = -1,
        base_complexity: Optional[List[int]] = None,
        cand_complexity: Optional[List[int]] = None,
        cand: Optional[Candidate] = None,
        ast_snapshot: Optional[str] = None,
        base_root: Optional[Node] = None,
        cand_root: Optional[Node] = None,
        base_mapping_cost: Optional[float] = None,
        cand_mapping_cost: Optional[float] = None,
    ) -> dict:
        """Record a decision to the persistent log.

        Sanitises non-finite floats to ``None`` for JSON safety.
        """
        def _safe(v):
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return v

        def _safe_float(v):
            try:
                f = float(v)
            except Exception:
                return None
            if not math.isfinite(f):
                return None
            return float(f)

        try:
            count_weight = float(getattr(self.lm_hp, "select_count_weight", 1.0))
        except Exception:
            count_weight = 1.0

        if base_root is None:
            try:
                base_root = self.state.root
            except Exception:
                base_root = None
        if cand_root is None and cand is not None:
            cand_root = getattr(cand, "root", None)

        if base_mapping_cost is None:
            try:
                base_mapping_cost = float(getattr(self.state, "complexity_mapping_cost", 0.0))
            except Exception:
                base_mapping_cost = 0.0
        base_map_cost = _clamp_nonnegative_finite(base_mapping_cost, default=0.0)

        cand_map_desc = _candidate_mapping_descriptor(cand) if cand is not None else _mapping_descriptor(None)
        if cand_mapping_cost is not None:
            cand_map_cost = _clamp_nonnegative_finite(cand_mapping_cost, default=0.0)
        else:
            cand_map_cost = _clamp_nonnegative_finite(cand_map_desc.get("cost", 0.0), default=0.0)
        cand_map_desc["cost"] = float(cand_map_cost)

        base_ast_cost = _safe_float(_ast_cost_physics_prior(base_root)) if base_root is not None else None
        cand_ast_cost = _safe_float(_ast_cost_physics_prior(cand_root)) if cand_root is not None else None

        base_nn_score = (
            _safe_float(_nn_structural_score(base_root, count_weight=count_weight))
            if base_root is not None
            else None
        )
        cand_nn_score = (
            _safe_float(_nn_structural_score(cand_root, count_weight=count_weight))
            if cand_root is not None
            else None
        )

        base_core_complexity = (
            _safe_float(_complexity_key(base_root, n_params_base if n_params_base >= 0 else None, count_weight=count_weight)[0])
            if base_root is not None
            else None
        )
        cand_core_complexity = (
            _safe_float(_complexity_key(cand_root, n_params_cand if n_params_cand >= 0 else None, count_weight=count_weight)[0])
            if cand_root is not None
            else None
        )

        base_total_complexity = (
            _safe_float(float(base_core_complexity) + float(base_map_cost))
            if base_core_complexity is not None
            else None
        )
        cand_total_complexity = (
            _safe_float(float(cand_core_complexity) + float(cand_map_cost))
            if cand_core_complexity is not None
            else None
        )

        self._decision_step += 1
        rec: dict = {
            "step": self._decision_step,
            "time": time.time(),
            "outcome": outcome,
            "rule": rule,
            "label": label,
            "reason": reason,
            "target": target,
            "target_uid": target_uid,
            "base_loss": _safe(base_loss),
            "cand_loss": _safe(cand_loss),
            "n_params_base": n_params_base,
            "n_params_cand": n_params_cand,
            "base_complexity": base_complexity,
            "cand_complexity": cand_complexity,
            "base_mapping_cost": _safe(base_map_cost),
            "cand_mapping_cost": _safe(cand_map_cost),
            "cand_mapping_kind": cand_map_desc.get("kind"),
            "cand_mapping_degree": cand_map_desc.get("degree"),
            "cand_mapping_class": cand_map_desc.get("class"),
            "cand_mapping_is_structural": bool(cand_map_desc.get("is_structural", False)),
            "base_ast_cost": _safe(base_ast_cost),
            "cand_ast_cost": _safe(cand_ast_cost),
            "base_nn_score": _safe(base_nn_score),
            "cand_nn_score": _safe(cand_nn_score),
            "base_complexity_score": _safe(base_core_complexity),
            "cand_complexity_score": _safe(cand_core_complexity),
            "base_complexity_total": _safe(base_total_complexity),
            "cand_complexity_total": _safe(cand_total_complexity),
        }
        # Optional meta flags from the candidate
        if cand is not None and isinstance(getattr(cand, "meta", None), dict):
            for key in ("structural", "weak_probe", "partial_sep"):
                if key in cand.meta:
                    rec[key] = bool(cand.meta[key])
            for key in (
                "coordinate_variant",
                "coordinate_variant_display",
                "pattern_family",
            ):
                if key in cand.meta:
                    rec[key] = str(cand.meta[key])
            if "factorized_mapping" in cand.meta and rec.get("cand_mapping_kind") is None:
                try:
                    rec["cand_mapping_kind"] = str((cand.meta.get("factorized_mapping") or {}).get("kind", "")).lower() or None
                except Exception:
                    pass
        if cand is not None:
            try:
                rec["cand_structural_label"] = bool(_is_structural_candidate(cand))
            except Exception:
                rec["cand_structural_label"] = False
            try:
                rec["cand_separability_label"] = bool(_is_separability_candidate(cand))
            except Exception:
                rec["cand_separability_label"] = False
        rec["pareto_trackable"] = bool(
            (rec.get("cand_loss") is not None)
            and (rec.get("cand_complexity_total") is not None)
        )
        if ast_snapshot is None and cand_root is not None:
            try:
                ast_snapshot = _compact_expression_repr(cand_root, max_length=20000)
            except Exception:
                ast_snapshot = None
        if ast_snapshot is not None:
            rec["ast_snapshot"] = ast_snapshot
        self.decision_log.append(rec)
        return rec

    def _coe_stageB_risk_tags(
        self,
        cand: Candidate,
        cand_state: StageBState,
        reason: Optional[str],
    ) -> List[str]:
        if not bool(getattr(self, "coe_stageB_dry_run", False)):
            return []
        tags: Set[str] = set()
        meta = cand.meta if isinstance(getattr(cand, "meta", None), dict) else {}
        label = str(getattr(cand, "label", "") or "").lower()
        reason_s = str(reason or "").lower()
        if _candidate_is_unpromoted_generic(cand):
            tags.add("generic_approximant")
        if getattr(self.lm_hp, "fit_y_link", None):
            tags.add("transformed_link")
        if (
            bool(meta.get("additive_gauge_sensitive", False))
            or bool(meta.get("homogeneous_gauge_sensitive", False))
            or bool(meta.get("gauge_sensitive", False))
            or bool(meta.get("additive_gauge_requires_scope_improvement", False))
            or bool(meta.get("homogeneous_gauge_requires_scope_improvement", False))
            or "gauge" in label
            or "gauge" in reason_s
        ):
            tags.add("gauge_sensitive")
        if (
            "variable_prune" in label
            or "axis" in label and "prune" in label
            or bool(meta.get("axis_deletion", False))
            or bool(meta.get("projection_prune", False))
        ):
            tags.add("axis_deletion")
        if "below-floor" in reason_s or "near-floor" in reason_s:
            tags.add("near_floor")
        try:
            floor = _resolve_acceptance_noise_floor_raw(self.lm_hp, self.loss_scale)
            cand_loss = float(cand_state.val_loss)
            base_loss = float(self.state.val_loss)
            if floor > 0.0 and math.isfinite(cand_loss) and math.isfinite(base_loss):
                if min(cand_loss, base_loss) <= 10.0 * floor:
                    tags.add("near_noise_floor")
        except Exception:
            pass
        return sorted(tags)

    def record_coe_stageB_dry_run(
        self,
        *,
        rule: str,
        label: str,
        reason: Optional[str],
        target: str,
        target_uid: str = "",
        cand: Candidate,
        cand_state: StageBState,
        n_params_base: int,
        n_params_cand: int,
    ) -> Optional[dict]:
        tags = self._coe_stageB_risk_tags(cand, cand_state, reason)
        if not tags:
            return None
        try:
            base_snapshot = _compact_expression_repr(self.state.root, max_length=240)
        except Exception:
            base_snapshot = str(self.state.root)
        try:
            cand_snapshot = _compact_expression_repr(cand_state.root, max_length=240)
        except Exception:
            cand_snapshot = str(cand_state.root)
        rec = {
            "time": time.time(),
            "mode": "dry_run",
            "outcome": "observe_accept",
            "rule": str(rule),
            "label": str(label),
            "reason": str(reason or "accepted"),
            "target": str(target),
            "target_uid": str(target_uid or ""),
            "risk_tags": tags,
            "base_loss": float(self.state.val_loss),
            "cand_loss": float(cand_state.val_loss),
            "n_params_base": int(n_params_base),
            "n_params_cand": int(n_params_cand),
            "base_snapshot": base_snapshot,
            "cand_snapshot": cand_snapshot,
            "candidate_meta": {
                k: v
                for k, v in (cand.meta if isinstance(getattr(cand, "meta", None), dict) else {}).items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
            "would_change_decision": False,
        }
        self.coe_stageB_dry_run_log.append(rec)
        self.log(
            "[CoE StageB dry-run] observed risky accept "
            f"({cand.label}; tags={','.join(tags)}); no decision changed"
        )
        return rec

    def coe_stageB_committee_gate(
        self,
        *,
        rule: str,
        label: str,
        reason: Optional[str],
        target: str,
        target_uid: str = "",
        cand: Candidate,
        cand_state: StageBState,
        n_params_base: int,
        n_params_cand: int,
    ) -> Tuple[bool, Optional[str]]:
        """Optional CoE Stage-B gate for risky accepted rewrites.

        Wave-2 starts deliberately narrow: visible analytic candidates are
        compared as raw-y fixed expressions, and NN-containing candidates are
        compared through same-history short refits.  Fit-links such as asinh
        remain fitting conditioners only; when they are active, the committee
        vote uses original-y MSE rather than fit-space loss.  Non-identity
        y-transform branches are also compared in original-y space: refits use
        the branch φ(y) data, while votes invert predictions and targets with
        y_op_inv.  Active x-coordinate transforms are replayed for refit gates
        and raw-rewritten before fixed-expression committee scoring. Expressions
        that still contain local fitted leaf wrappers are logged as unsupported
        and keep legacy behavior.
        """
        dry_rec = self.record_coe_stageB_dry_run(
            rule=rule,
            label=label,
            reason=reason,
            target=target,
            target_uid=target_uid,
            cand=cand,
            cand_state=cand_state,
            n_params_base=n_params_base,
            n_params_cand=n_params_cand,
        )
        risk_tags = list((dry_rec or {}).get("risk_tags") or [])
        if not risk_tags:
            return True, None
        if str(getattr(self, "coe_mode", "off") or "off") not in {"committee_gated", "reservoir_discovery"}:
            return True, None

        def _gate_record(
            *,
            outcome: str,
            gate_status: str,
            decision_reason: str,
            incumbent_expr: Optional[str] = None,
            candidate_expr: Optional[str] = None,
            results: Optional[List[dict]] = None,
            summary: Optional[dict] = None,
            warnings: Optional[List[str]] = None,
        ) -> dict:
            rec = {
                "time": time.time(),
                "mode": str(getattr(self, "coe_mode", "committee_gated") or "committee_gated"),
                "outcome": str(outcome),
                "gate_status": str(gate_status),
                "decision_reason": str(decision_reason),
                "rule": str(rule),
                "label": str(label),
                "reason": str(reason or "accepted"),
                "target": str(target),
                "target_uid": str(target_uid or ""),
                "risk_tags": list(risk_tags),
                "base_loss": float(self.state.val_loss),
                "cand_loss": float(cand_state.val_loss),
                "n_params_base": int(n_params_base),
                "n_params_cand": int(n_params_cand),
                "incumbent_expr": incumbent_expr,
                "candidate_expr": candidate_expr,
                "summary": dict(summary or {}),
                "results": list(results or []),
                "warnings": list(warnings or []),
            }
            self.coe_stageB_gate_log.append(rec)
            return rec

        def _unsupported(msg: str) -> Tuple[bool, Optional[str]]:
            _gate_record(
                outcome="legacy_allow",
                gate_status="unsupported",
                decision_reason=msg,
            )
            self.log(f"[CoE StageB gate] unsupported for {label}: {msg}; legacy accept")
            return True, "coe-gate-unsupported"

        filepath = getattr(self, "coe_filepath", None)
        if not filepath:
            return _unsupported("no single-dataset filepath configured")
        xcoords_obj = getattr(self, "xcoords", None)
        xcoords_active = bool(getattr(self, "xcoords_applied", False))
        try:
            if xcoords_obj is not None and hasattr(xcoords_obj, "is_identity"):
                xcoords_active = xcoords_active or (not bool(xcoords_obj.is_identity()))
        except Exception:
            xcoords_active = xcoords_active or bool(getattr(self, "stageA_x_transforms", None))
        if xcoords_active and xcoords_obj is None:
            return _unsupported("active x-coordinate transform has no coordinate replay object")
        if bool(getattr(self, "stageA_x_transforms", None)) and xcoords_obj is None:
            return _unsupported("Stage-A x-transform metadata has no committee replay object")
        fit_link_name = getattr(self.lm_hp, "fit_y_link", None)
        y_op_inv = getattr(self, "y_op_inv", None)
        y_transform_active = y_op_inv is not None
        try:
            incumbent_has_nn = bool(collect_nn_atoms(self.state.root))
            candidate_has_nn = bool(collect_nn_atoms(cand_state.root))
        except Exception as exc:
            return _unsupported(f"NN-leaf inspection failed: {type(exc).__name__}: {exc}")

        def _snapshot(root: Node) -> str:
            try:
                return _compact_expression_repr(root, max_length=360)
            except Exception:
                return str(root)

        if incumbent_has_nn or candidate_has_nn:
            if not bool(getattr(self, "coe_stageB_refit_gate", True)):
                return _unsupported("incumbent/candidate contains NN leaves and refit gate disabled")
            refit_result = self._coe_stageB_refit_committee_gate(
                rule=rule,
                label=label,
                reason=reason,
                target=target,
                target_uid=target_uid,
                cand=cand,
                cand_state=cand_state,
                n_params_base=n_params_base,
                n_params_cand=n_params_cand,
                risk_tags=risk_tags,
                gate_record_fn=_gate_record,
                incumbent_snapshot=_snapshot(self.state.root),
                candidate_snapshot=_snapshot(cand_state.root),
            )
            if refit_result is not None:
                return refit_result
            return _unsupported("incumbent/candidate contains NN leaves and refit gate unavailable")

        try:
            incumbent_expr = ast_to_human_readable(self.state.root)
            candidate_expr = ast_to_human_readable(cand_state.root)
            incumbent_coefficient_metadata = collect_coefficient_metadata(
                self.state.root,
                self.state.model,
                getattr(self, "units_spec", None),
            )
            candidate_coefficient_metadata = collect_coefficient_metadata(
                cand_state.root,
                cand_state.model,
                getattr(self, "units_spec", None),
            )
            if xcoords_active:
                try:
                    import sympy as _sp
                except Exception as exc:
                    return _unsupported(
                        f"raw-x rewrite requires SymPy: {type(exc).__name__}: {exc}"
                    )

                def _rewrite_internal_expr_to_raw(expr_text: Any) -> str:
                    expr_s = str(expr_text).replace("^", "**")
                    phi_sym = _sp.sympify(expr_s, locals={"pi": _sp.pi, "E": _sp.E})
                    raw_sym = xcoords_obj.sympy_rewrite_internal_expr_to_raw(
                        phi_sym,
                        const_mode="number",
                    )
                    return str(raw_sym)

                incumbent_expr = _rewrite_internal_expr_to_raw(incumbent_expr)
                candidate_expr = _rewrite_internal_expr_to_raw(candidate_expr)
            if y_transform_active:
                from nestynet_sr.sr_search.transform_render import wrap_phi_expr_str

                incumbent_expr = wrap_phi_expr_str(
                    str(incumbent_expr),
                    y_op_inv,
                    simplify=False,
                )
                candidate_expr = wrap_phi_expr_str(
                    str(candidate_expr),
                    y_op_inv,
                    simplify=False,
                )
                if not incumbent_expr or not candidate_expr:
                    return _unsupported("failed to render y-transform branch as raw-y expression")
        except Exception as exc:
            return _unsupported(f"expression rendering failed: {type(exc).__name__}: {exc}")

        n_slices = min(
            int(getattr(self, "coe_num_slices", 0) or 0),
            int(getattr(self, "coe_stageB_gate_slices", 0) or 0),
        )
        if n_slices <= 0:
            return _unsupported("no Stage-B gate slices configured")

        try:
            from nestynet_sr.sr_search.coe_committee import (
                CandidateArtifact,
                CommitteeEvalCache,
                _committee_tolerance,
                _load_dataset_arrays,
                build_slice_specs,
                evaluate_candidate_on_slice_cached,
            )
        except Exception as exc:
            return _unsupported(f"CoE committee evaluator unavailable: {type(exc).__name__}: {exc}")
        if getattr(self, "coe_eval_cache", None) is None:
            self.coe_eval_cache = CommitteeEvalCache(enabled=True)
        fixed_gate_max_rows: Optional[int] = None
        try:
            _X_all_fixed, _y_all_fixed, _cols_fixed = _load_dataset_arrays(str(filepath))
            fixed_gate_max_rows = int(_y_all_fixed.shape[0])
        except Exception:
            fixed_gate_max_rows = None

        inc_art = CandidateArtifact(
            candidate_id="incumbent",
            expr=str(incumbent_expr),
            source="stageB:incumbent",
            label="incumbent",
            n_free_params=int(max(0, n_params_base)),
            metadata={
                "coefficient_metadata": incumbent_coefficient_metadata,
            },
        )
        cand_art = CandidateArtifact(
            candidate_id="candidate",
            expr=str(candidate_expr),
            source=f"stageB:{rule}",
            label=str(label),
            n_free_params=int(max(0, n_params_cand)),
            metadata={
                "risk_tags": list(risk_tags),
                "coefficient_metadata": candidate_coefficient_metadata,
            },
        )
        specs = build_slice_specs(
            n_slices=n_slices,
            ndata_train=int(getattr(self, "coe_ndata_train", 2000) or 2000),
            ndata_val=int(getattr(self, "coe_ndata_val", 2000) or 2000),
            start_slice=int(getattr(self, "coe_start_slice", 0) or 0),
            skip_slice_ids=(
                ()
                if getattr(self, "coe_reference_slice", None) is None
                else (int(getattr(self, "coe_reference_slice")),)
            ),
            max_rows=fixed_gate_max_rows,
        )
        initial_slices = min(
            len(specs),
            max(1, int(getattr(self, "coe_stageB_initial_gate_slices", 3) or 3)),
        )
        executor = CoEWitnessExecutor.from_config(self)

        def _flat_results(rows_i: List[dict]) -> List[dict]:
            out: List[dict] = []
            for row_i in rows_i:
                vals = row_i.get("results")
                if isinstance(vals, list):
                    out.extend([dict(v) for v in vals if isinstance(v, dict)])
            return out

        def _paired_rows(rows_i: List[dict]) -> List[dict]:
            return [row_i for row_i in rows_i if row_i.get("status") == "success"]

        def _paired_summary(rows_i: List[dict]) -> dict:
            paired_i = _paired_rows(rows_i)
            wins = losses = ties = 0
            inc_losses: List[float] = []
            cand_losses: List[float] = []
            for pair_i in paired_i:
                inc_res_i = dict(pair_i.get("incumbent_result") or {})
                cand_res_i = dict(pair_i.get("candidate_result") or {})
                inc_losses.append(float(inc_res_i["val_mse"]))
                cand_losses.append(float(cand_res_i["val_mse"]))
                tol_i = _committee_tolerance(
                    loss_a=float(inc_res_i["val_mse"]),
                    loss_b=float(cand_res_i["val_mse"]),
                    noise_floor_raw=float(getattr(self, "coe_noise_floor_raw", 0.0) or 0.0),
                    n_eff=max(
                        1,
                        int(cand_res_i.get("n_val") or inc_res_i.get("n_val") or 1),
                    ),
                    noise_mult=float(getattr(self, "coe_noise_mult", 3.0) or 3.0),
                    rel_tol=float(getattr(self, "coe_rel_tol", 1.0e-3) or 1.0e-3),
                )
                delta_i = float(cand_res_i["val_mse"]) - float(inc_res_i["val_mse"])
                if delta_i < -tol_i:
                    wins += 1
                elif delta_i > tol_i:
                    losses += 1
                else:
                    ties += 1
            try:
                import numpy as _np

                inc_med_i = float(_np.median(_np.asarray(inc_losses, dtype=float)))
                cand_med_i = float(_np.median(_np.asarray(cand_losses, dtype=float)))
            except Exception:
                inc_med_i = float("inf")
                cand_med_i = float("inf")
            med_tol_i = _committee_tolerance(
                loss_a=inc_med_i,
                loss_b=cand_med_i,
                noise_floor_raw=float(getattr(self, "coe_noise_floor_raw", 0.0) or 0.0),
                n_eff=max(
                    1,
                    len(paired_i) * int(getattr(self, "coe_ndata_val", 2000) or 2000),
                ),
                noise_mult=float(getattr(self, "coe_noise_mult", 3.0) or 3.0),
                rel_tol=float(getattr(self, "coe_rel_tol", 1.0e-3) or 1.0e-3),
            )
            return {
                "n_slices": int(n_slices),
                "reference_slice": getattr(self, "coe_reference_slice", None),
                "excluded_slice_ids": (
                    []
                    if getattr(self, "coe_reference_slice", None) is None
                    else [int(getattr(self, "coe_reference_slice"))]
                ),
                "initial_slices": int(initial_slices),
                "evaluated_slices": int(len(rows_i)),
                "adaptive_expanded": bool(len(rows_i) > initial_slices),
                "n_paired_success": int(len(paired_i)),
                "wins": int(wins),
                "ties": int(ties),
                "losses": int(losses),
                "incumbent_median_mse": float(inc_med_i),
                "candidate_median_mse": float(cand_med_i),
                "median_delta": float(cand_med_i - inc_med_i),
                "median_tolerance": float(med_tol_i),
                "comparison_space": (
                    "raw_y"
                    if (fit_link_name or y_transform_active)
                    else "model_output"
                ),
                "x_coordinate_space": "internal_x" if xcoords_active else "raw_x",
                "x_transform_active": bool(xcoords_active),
                "fit_y_link": str(fit_link_name) if fit_link_name else None,
                "y_transform_active": bool(y_transform_active),
                "cache_stats": (
                    self.coe_eval_cache.stats()
                    if getattr(self, "coe_eval_cache", None) is not None
                    else {}
                ),
                "witness_executor": coe_witness_execution_metadata(executor, rows_i),
            }

        def _gate_decision(summary_i: dict, *, final: bool) -> Optional[str]:
            wins_i = int(summary_i.get("wins", 0) or 0)
            losses_i = int(summary_i.get("losses", 0) or 0)
            delta_i = float(summary_i.get("median_delta", float("inf")))
            tol_i = float(summary_i.get("median_tolerance", 0.0) or 0.0)
            if losses_i > wins_i and delta_i > tol_i:
                return "veto"
            # A clearly better candidate can pass immediately.  For ties, wait
            # until the configured max slice budget has spoken.
            if wins_i > losses_i and delta_i <= tol_i:
                return "allow"
            if final:
                return "allow"
            return None

        maxt_observe = str(getattr(self, "coe_inference", "legacy") or "legacy") == "maxt_observe"

        def _worker(job) -> dict:
            spec = job.payload
            inc_res = evaluate_candidate_on_slice_cached(
                inc_art,
                filepath=str(filepath),
                spec=spec,
                min_valid_fraction=float(getattr(self, "coe_min_valid_fraction", 0.80) or 0.80),
                cache=self.coe_eval_cache,
                return_row_losses=maxt_observe,
            )
            cand_res = evaluate_candidate_on_slice_cached(
                cand_art,
                filepath=str(filepath),
                spec=spec,
                min_valid_fraction=float(getattr(self, "coe_min_valid_fraction", 0.80) or 0.80),
                cache=self.coe_eval_cache,
                return_row_losses=maxt_observe,
            )
            inc_row = inc_res.to_dict()
            cand_row = cand_res.to_dict()
            return {
                "method": "fixed_expression_compare",
                "slice_id": int(spec.slice_id),
                "train_rows": [int(spec.train_start), int(spec.train_stop)],
                "val_rows": [int(spec.val_start), int(spec.val_stop)],
                "status": (
                    "success"
                    if inc_res.status == "success" and cand_res.status == "success"
                    else "error"
                ),
                "incumbent_result": inc_row,
                "candidate_result": cand_row,
                "results": [inc_row, cand_row],
            }

        def _stop_after(rows_i: List[dict]) -> bool:
            if len(rows_i) < initial_slices:
                return False
            summary_i = _paired_summary(rows_i)
            decision_i = _gate_decision(summary_i, final=len(rows_i) >= len(specs))
            return decision_i is not None

        if int(getattr(executor, "parallelism", 1) or 1) > 1:
            witness_rows = run_fixed_expression_pair_witnesses(
                specs=specs,
                incumbent=inc_art,
                candidate=cand_art,
                filepath=str(filepath),
                min_valid_fraction=float(
                    getattr(self, "coe_min_valid_fraction", 0.80) or 0.80
                ),
                executor=executor,
                prefix="stageB_fixed_expr",
                stop_after=_stop_after,
                return_row_losses=maxt_observe,
            )
        else:
            witness_rows = executor.run(
                coe_witness_jobs_from_specs(specs, prefix="stageB_fixed_expr"),
                _worker,
                stop_after=_stop_after,
            )
        results = _flat_results(witness_rows)
        paired = _paired_rows(witness_rows)
        summary = {}
        decision = None
        if len(witness_rows) >= initial_slices:
            summary = _paired_summary(witness_rows)
            decision = _gate_decision(summary, final=len(witness_rows) >= len(specs))

        if not paired:
            _gate_record(
                outcome="legacy_allow",
                gate_status="unsupported",
                decision_reason="no paired successful fixed-expression slice evaluations",
                incumbent_expr=str(incumbent_expr),
                candidate_expr=str(candidate_expr),
                results=results,
            )
            self.log(
                f"[CoE StageB gate] unsupported for {label}: no paired successful "
                "fixed-expression evaluations; legacy accept"
            )
            return True, "coe-gate-unsupported"

        if not summary:
            summary = _paired_summary(witness_rows)
        if maxt_observe:
            # Observe-only: record the calibrated paired max-T verdict next to
            # the legacy vote.  Never changes the decision, never raises.
            try:
                from nestynet_sr.stat_selection.committee_inference import (
                    maxt_decision_from_slice_rows,
                )

                baseline_rows_map: dict = {}
                candidate_rows_map: dict = {}
                for pair_row in paired:
                    inc_res_row = dict(pair_row.get("incumbent_result") or {})
                    cand_res_row = dict(pair_row.get("candidate_result") or {})
                    inc_rows = inc_res_row.get("row_losses")
                    cand_rows = cand_res_row.get("row_losses")
                    val_rows = pair_row.get("val_rows") or [None, None]
                    if inc_rows is None or cand_rows is None or val_rows[0] is None:
                        continue
                    sid = int(pair_row["slice_id"])
                    baseline_rows_map[sid] = (int(val_rows[0]), inc_rows)
                    candidate_rows_map[sid] = (int(val_rows[0]), cand_rows)
                if baseline_rows_map:
                    maxt = maxt_decision_from_slice_rows(
                        baseline_rows=baseline_rows_map,
                        member_rows={"candidate": candidate_rows_map},
                        seed=int(getattr(self, "coe_maxt_seed", 0) or 0),
                    )
                    maxt_gate = (
                        "veto" if maxt.verdict_for("candidate") == "worse" else "allow"
                    )
                    legacy_gate = "veto" if decision == "veto" else "allow"
                    summary["maxt_observe"] = {
                        **maxt.to_dict(),
                        "maxt_gate_equivalent": maxt_gate,
                        "legacy_gate": legacy_gate,
                        "agrees_with_legacy": bool(maxt_gate == legacy_gate),
                    }
                    if maxt_gate != legacy_gate:
                        self.log(
                            f"[CoE maxt-observe] DISAGREES with legacy gate for {label}: "
                            f"legacy={legacy_gate}, maxt={maxt_gate} "
                            f"(delta={maxt.member_verdicts[0].mean_delta:.3e}, "
                            f"CI=[{maxt.member_verdicts[0].ci_lower:.3e}, "
                            f"{maxt.member_verdicts[0].ci_upper:.3e}], "
                            f"G={maxt.n_units})"
                        )
                else:
                    summary["maxt_observe"] = {
                        "status": "unavailable",
                        "reason": "no paired per-row losses",
                    }
            except Exception as exc:
                summary["maxt_observe"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        wins = int(summary.get("wins", 0) or 0)
        ties = int(summary.get("ties", 0) or 0)
        losses = int(summary.get("losses", 0) or 0)
        med_delta = float(summary.get("median_delta", float("inf")))
        med_tol = float(summary.get("median_tolerance", 0.0) or 0.0)
        if decision == "veto":
            _gate_record(
                outcome="veto",
                gate_status="veto",
                decision_reason="candidate loses committee fixed-expression comparison",
                incumbent_expr=str(incumbent_expr),
                candidate_expr=str(candidate_expr),
                results=results,
                summary=summary,
            )
            self.log(
                f"[CoE StageB gate] veto {label}: "
                f"wins/ties/losses={wins}/{ties}/{losses}, "
                f"median Δ={med_delta:.3e} > tol={med_tol:.3e}, "
                f"slices={summary.get('evaluated_slices')}/{summary.get('n_slices')}"
            )
            return False, (
                "reject-coe-stageB-gate("
                f"wins/ties/losses={wins}/{ties}/{losses}, "
                f"median_delta={med_delta:.3e})"
            )

        _gate_record(
            outcome="allow",
            gate_status="accepted",
            decision_reason="candidate passes committee fixed-expression comparison",
            incumbent_expr=str(incumbent_expr),
            candidate_expr=str(candidate_expr),
            results=results,
            summary=summary,
        )
        self.log(
            f"[CoE StageB gate] allow {label}: "
            f"wins/ties/losses={wins}/{ties}/{losses}, "
            f"median Δ={med_delta:.3e}, tol={med_tol:.3e}, "
            f"slices={summary.get('evaluated_slices')}/{summary.get('n_slices')}"
        )
        return True, (
            "coe-stageB-gate-accepted("
            f"wins/ties/losses={wins}/{ties}/{losses})"
        )

    def _coe_stageB_refit_committee_gate(
        self,
        *,
        rule: str,
        label: str,
        reason: Optional[str],
        target: str,
        target_uid: str,
        cand: Candidate,
        cand_state: StageBState,
        n_params_base: int,
        n_params_cand: int,
        risk_tags: List[str],
        gate_record_fn: Callable[..., dict],
        incumbent_snapshot: str,
        candidate_snapshot: str,
    ) -> Optional[Tuple[bool, Optional[str]]]:
        """Compare the same incumbent/candidate ASTs after short slice refits.

        This is the in-tree CoE path: slice 0/reference owns the search tree;
        other slices only act as critics by briefly refitting cloned versions of
        the two already-built ASTs.  No committee slice proposes or accepts a
        different rewrite.
        """

        def _legacy_allow(msg: str, *, results=None, summary=None, warnings=None):
            gate_record_fn(
                outcome="legacy_allow",
                gate_status="refit_unsupported",
                decision_reason=msg,
                incumbent_expr=incumbent_snapshot,
                candidate_expr=candidate_snapshot,
                results=results,
                summary=summary,
                warnings=warnings,
            )
            self.log(f"[CoE StageB refit] unsupported for {label}: {msg}; legacy accept")
            return True, "coe-refit-gate-unsupported"

        filepath = getattr(self, "coe_filepath", None)
        if not filepath:
            return _legacy_allow("no single-dataset filepath configured")
        n_slices = min(
            int(getattr(self, "coe_num_slices", 0) or 0),
            int(getattr(self, "coe_stageB_gate_slices", 0) or 0),
        )
        if n_slices <= 0:
            return _legacy_allow("no Stage-B refit gate slices configured")

        try:
            import numpy as _np
            from torch.utils.data import DataLoader, TensorDataset

            from nestynet_sr.sr_search.coe_committee import (
                _committee_tolerance,
                _load_dataset_arrays,
                build_slice_specs,
            )
        except Exception as exc:
            return _legacy_allow(f"CoE refit helpers unavailable: {type(exc).__name__}: {exc}")

        try:
            X_all, y_all, _cols = _load_dataset_arrays(str(filepath))
        except Exception as exc:
            return _legacy_allow(f"slice data load failed: {type(exc).__name__}: {exc}")

        specs = build_slice_specs(
            n_slices=n_slices,
            ndata_train=int(getattr(self, "coe_ndata_train", 2000) or 2000),
            ndata_val=int(getattr(self, "coe_ndata_val", 2000) or 2000),
            start_slice=int(getattr(self, "coe_start_slice", 0) or 0),
            skip_slice_ids=(
                ()
                if getattr(self, "coe_reference_slice", None) is None
                else (int(getattr(self, "coe_reference_slice")),)
            ),
            max_rows=int(y_all.shape[0]),
        )
        initial_slices = min(
            len(specs),
            max(1, int(getattr(self, "coe_stageB_initial_gate_slices", 3) or 3)),
        )
        epochs = max(1, int(getattr(self, "coe_stageB_refit_epochs", 200) or 200))
        escalate_epochs = max(
            0,
            int(getattr(self, "coe_stageB_refit_escalate_epochs", 0) or 0),
        )
        fit_link_name = getattr(self.lm_hp, "fit_y_link", None)
        fit_link_active = bool(fit_link_name)
        xcoords_obj = getattr(self, "xcoords", None)
        xcoords_active = bool(getattr(self, "xcoords_applied", False))
        try:
            if xcoords_obj is not None and hasattr(xcoords_obj, "is_identity"):
                xcoords_active = xcoords_active or (not bool(xcoords_obj.is_identity()))
        except Exception:
            xcoords_active = xcoords_active or bool(getattr(self, "stageA_x_transforms", None))
        if xcoords_active and xcoords_obj is None:
            return _legacy_allow("active x-coordinate transform has no coordinate replay object")
        if bool(getattr(self, "stageA_x_transforms", None)) and xcoords_obj is None:
            return _legacy_allow("Stage-A x-transform metadata has no committee replay object")
        y_op = getattr(self, "y_op", None)
        y_op_inv = getattr(self, "y_op_inv", None)
        y_transform_active = y_op_inv is not None
        compare_original_y = bool(fit_link_active or y_transform_active)
        if y_transform_active and y_op is None:
            return _legacy_allow("active y-transform has no forward y_op for committee slice replay")

        def _slice_loader(start: int, stop: int):
            if start < 0 or stop <= start or stop > int(y_all.shape[0]):
                raise ValueError(
                    f"slice rows [{start}, {stop}) outside dataset with {int(y_all.shape[0])} rows"
                )
            x_slice = _np.array(X_all[start:stop], dtype=_np.float64, copy=True)
            if xcoords_active:
                try:
                    x_slice = _np.array(
                        xcoords_obj.apply_np(x_slice),
                        dtype=_np.float64,
                        copy=True,
                    )
                except Exception as exc:
                    raise ValueError(
                        f"failed to apply x-coordinate transform to CoE slice rows [{start}, {stop})"
                    ) from exc
                if not _np.all(_np.isfinite(x_slice)):
                    raise ValueError(
                        f"non-finite x-coordinate transform values on CoE slice rows [{start}, {stop})"
                    )
            xb = torch.as_tensor(x_slice, dtype=self.dtype)
            y_slice = _np.array(y_all[start:stop], dtype=_np.float64, copy=True).reshape(-1, 1)
            if y_op is not None:
                try:
                    y_slice = _np.array(y_op(y_slice), dtype=_np.float64, copy=True).reshape(-1, 1)
                except Exception as exc:
                    raise ValueError(
                        f"failed to apply y-transform to CoE slice rows [{start}, {stop})"
                    ) from exc
                if not _np.all(_np.isfinite(y_slice)):
                    raise ValueError(
                        f"non-finite y-transform target values on CoE slice rows [{start}, {stop})"
                    )
            yb = torch.as_tensor(y_slice, dtype=self.dtype)
            batch_size = int(getattr(self.train_loader_probe, "batch_size", 0) or xb.shape[0])
            batch_size = max(1, min(int(batch_size), int(xb.shape[0])))
            return DataLoader(
                TensorDataset(xb, yb),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        def _summary(
            pass_results: List[dict],
            pass_paired: List[dict],
            *,
            pass_epochs: int,
            refit_tier: str,
        ) -> dict:
            wins = losses = ties = 0
            inc_losses: List[float] = []
            cand_losses: List[float] = []
            inc_fit_losses: List[float] = []
            cand_fit_losses: List[float] = []
            inc_raw_losses: List[float] = []
            cand_raw_losses: List[float] = []
            for row in pass_paired:
                inc_v = float(row.get("incumbent_compare_loss", row["incumbent_val_loss"]))
                cand_v = float(row.get("candidate_compare_loss", row["candidate_val_loss"]))
                inc_losses.append(inc_v)
                cand_losses.append(cand_v)
                try:
                    inc_fit_losses.append(float(row["incumbent_val_loss"]))
                    cand_fit_losses.append(float(row["candidate_val_loss"]))
                except Exception:
                    pass
                if "incumbent_raw_y_mse" in row and "candidate_raw_y_mse" in row:
                    try:
                        inc_raw_losses.append(float(row["incumbent_raw_y_mse"]))
                        cand_raw_losses.append(float(row["candidate_raw_y_mse"]))
                    except Exception:
                        pass
                tol_i = _committee_tolerance(
                    loss_a=inc_v,
                    loss_b=cand_v,
                    noise_floor_raw=float(getattr(self, "coe_noise_floor_raw", 0.0) or 0.0),
                    n_eff=max(1, int(row.get("n_val", 1) or 1)),
                    noise_mult=float(getattr(self, "coe_noise_mult", 3.0) or 3.0),
                    rel_tol=float(getattr(self, "coe_rel_tol", 1.0e-3) or 1.0e-3),
                )
                delta_i = cand_v - inc_v
                row["delta"] = float(delta_i)
                row["tolerance"] = float(tol_i)
                if delta_i < -tol_i:
                    wins += 1
                    row["vote"] = "win"
                elif delta_i > tol_i:
                    losses += 1
                    row["vote"] = "loss"
                else:
                    ties += 1
                    row["vote"] = "tie"
            inc_med = float(_np.median(_np.asarray(inc_losses, dtype=float))) if inc_losses else float("inf")
            cand_med = float(_np.median(_np.asarray(cand_losses, dtype=float))) if cand_losses else float("inf")
            inc_fit_med = (
                float(_np.median(_np.asarray(inc_fit_losses, dtype=float)))
                if inc_fit_losses
                else float("inf")
            )
            cand_fit_med = (
                float(_np.median(_np.asarray(cand_fit_losses, dtype=float)))
                if cand_fit_losses
                else float("inf")
            )
            inc_raw_med = (
                float(_np.median(_np.asarray(inc_raw_losses, dtype=float)))
                if inc_raw_losses
                else float("nan")
            )
            cand_raw_med = (
                float(_np.median(_np.asarray(cand_raw_losses, dtype=float)))
                if cand_raw_losses
                else float("nan")
            )
            med_tol = _committee_tolerance(
                loss_a=inc_med,
                loss_b=cand_med,
                noise_floor_raw=float(getattr(self, "coe_noise_floor_raw", 0.0) or 0.0),
                n_eff=max(
                    1,
                    len(pass_paired) * int(getattr(self, "coe_ndata_val", 2000) or 2000),
                ),
                noise_mult=float(getattr(self, "coe_noise_mult", 3.0) or 3.0),
                rel_tol=float(getattr(self, "coe_rel_tol", 1.0e-3) or 1.0e-3),
            )
            return {
                "gate_kind": "refit_compare",
                "n_slices": int(n_slices),
                "reference_slice": getattr(self, "coe_reference_slice", None),
                "excluded_slice_ids": (
                    []
                    if getattr(self, "coe_reference_slice", None) is None
                    else [int(getattr(self, "coe_reference_slice"))]
                ),
                "initial_slices": int(initial_slices),
                "evaluated_slices": int(len(pass_results)),
                "adaptive_expanded": bool(len(pass_results) > initial_slices),
                "n_paired_success": int(len(pass_paired)),
                "wins": int(wins),
                "ties": int(ties),
                "losses": int(losses),
                "incumbent_median_mse": float(inc_med),
                "candidate_median_mse": float(cand_med),
                "median_delta": float(cand_med - inc_med),
                "median_tolerance": float(med_tol),
                "comparison_space": "raw_y" if compare_original_y else "fit_space",
                "x_coordinate_space": "internal_x" if xcoords_active else "raw_x",
                "x_transform_active": bool(xcoords_active),
                "fit_y_link": str(fit_link_name) if fit_link_active else None,
                "y_transform_active": bool(y_transform_active),
                "incumbent_median_fit_loss": float(inc_fit_med),
                "candidate_median_fit_loss": float(cand_fit_med),
                "fit_space_median_delta": float(cand_fit_med - inc_fit_med),
                "incumbent_median_raw_y_mse": float(inc_raw_med),
                "candidate_median_raw_y_mse": float(cand_raw_med),
                "raw_y_median_delta": float(cand_raw_med - inc_raw_med)
                if math.isfinite(inc_raw_med) and math.isfinite(cand_raw_med)
                else float("nan"),
                "epochs": int(pass_epochs),
                "refit_tier": str(refit_tier),
                "rng_restored": True,
                "risk_tags": list(risk_tags),
            }

        def _decision(summary_i: dict, *, final: bool) -> Optional[str]:
            wins_i = int(summary_i.get("wins", 0) or 0)
            losses_i = int(summary_i.get("losses", 0) or 0)
            delta_i = float(summary_i.get("median_delta", float("inf")))
            tol_i = float(summary_i.get("median_tolerance", 0.0) or 0.0)
            if losses_i > wins_i and delta_i > tol_i:
                return "veto"
            if wins_i > losses_i and delta_i <= tol_i:
                return "allow"
            if final:
                return "allow"
            return None

        saved = {
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "dataset_ids": self.dataset_ids,
            "loss_scales": self.loss_scales,
            "agg_mode": self.agg_mode,
            "agg_weights": self.agg_weights,
            "_cache": self._cache,
        }

        reference_rng_state = _snapshot_rng_state()

        def _restore_context() -> None:
            self.train_loader = saved["train_loader"]
            self.val_loader = saved["val_loader"]
            self.dataset_ids = saved["dataset_ids"]
            self.loss_scales = saved["loss_scales"]
            self.agg_mode = saved["agg_mode"]
            self.agg_weights = saved["agg_weights"]
            self._cache = saved["_cache"]
            _restore_rng_state(reference_rng_state)

        def _process_refit_disabled_reason() -> Optional[str]:
            """Return why this refit must use the live serial context, if any."""

            try:
                if getattr(getattr(self, "fit_candidate", None), "__func__", None) is not StageBContext.fit_candidate:
                    return "stageB_refit_custom_fit_candidate_live_context"
            except Exception:
                return "stageB_refit_custom_fit_candidate_live_context"
            if isinstance(getattr(self, "train_loader", None), (list, tuple)):
                return "stageB_refit_multi_dataset_process_payload_unsupported"
            if getattr(self, "atom_factory", None) is not None:
                return "stageB_refit_atom_factory_process_payload_unsupported"
            if xcoords_active:
                return "stageB_refit_xcoords_process_payload_unsupported"
            y_name = str(getattr(self, "y_transform_name", "identity") or "identity")
            if y_transform_active and y_name == "identity":
                return "stageB_refit_y_transform_name_unavailable"

            def _needs_live_fresh_nn(root: Node, reuse: dict) -> bool:
                try:
                    reuse_keys = {str(k) for k in (reuse or {}).keys()}
                    for atom in collect_nn_atoms(root):
                        if str(getattr(atom, "kind", "")).lower() != "nn":
                            continue
                        tag = getattr(atom, "tag", None)
                        if tag is None or str(tag) not in reuse_keys:
                            return True
                except Exception:
                    return True
                return False

            if _needs_live_fresh_nn(self.state.root, self.state.reuse or {}):
                return "stageB_refit_incumbent_requires_live_fresh_nn_factory"
            if _needs_live_fresh_nn(cand_state.root, cand_state.reuse or {}):
                return "stageB_refit_candidate_requires_live_fresh_nn_factory"
            return None

        def _portable_reuse_map(reuse: Optional[dict]) -> dict:
            out: dict = {}
            for key, module in (reuse or {}).items():
                copied = copy.deepcopy(module)
                try:
                    copied.to(device=torch.device("cpu"), dtype=self.dtype)
                except TypeError:
                    try:
                        copied.to(torch.device("cpu"))
                    except Exception:
                        pass
                out[str(key)] = copied
            return out

        def _spec_to_payload(spec: Any) -> dict:
            if hasattr(spec, "to_dict"):
                row = dict(spec.to_dict())
            else:
                row = {
                    "slice_id": getattr(spec, "slice_id"),
                    "train_start": getattr(spec, "train_start"),
                    "train_stop": getattr(spec, "train_stop"),
                    "val_start": getattr(spec, "val_start"),
                    "val_stop": getattr(spec, "val_stop"),
                }
            return {
                "slice_id": int(row["slice_id"]),
                "train_start": int(row["train_start"]),
                "train_stop": int(row["train_stop"]),
                "val_start": int(row["val_start"]),
                "val_stop": int(row["val_stop"]),
            }

        def _build_portable_refit_payloads(pass_epochs: int, refit_tier: str) -> List[dict]:
            y_name = str(getattr(self, "y_transform_name", "identity") or "identity")
            if not y_transform_active:
                y_name = "identity"
            try:
                batch_size = int(getattr(self.train_loader_probe, "batch_size", 0) or 0)
            except Exception:
                batch_size = 0
            if batch_size <= 0:
                batch_size = int(getattr(self, "coe_ndata_train", 2000) or 2000)
            incumbent_reuse = _portable_reuse_map(self.state.reuse or {})
            candidate_reuse = _portable_reuse_map(cand_state.reuse or {})
            base_payload = {
                "schema": "coe_stageB_refit_witness_v1",
                "filepath": str(filepath),
                "incumbent_root": coe_stageB_refit_ast_to_payload(self.state.root),
                "candidate_root": coe_stageB_refit_ast_to_payload(cand_state.root),
                "incumbent_reuse": incumbent_reuse,
                "candidate_reuse": candidate_reuse,
                "lm_hp": copy.deepcopy(self.lm_hp),
                "dtype": str(self.dtype).replace("torch.", ""),
                "device": "cpu",
                "force_cpu": True,
                "epochs": int(pass_epochs),
                "loss_scale": float(self.loss_scale),
                "batch_size": int(batch_size),
                "y_transform_name": y_name,
                "xcoords_active": False,
                "xcoords": None,
                "trig_by_axis": copy.deepcopy(self.trig_by_axis),
                "refit_tier": str(refit_tier),
                "atom_factory": None,
                "return_row_losses": (
                    str(getattr(self, "coe_inference", "legacy") or "legacy")
                    == "maxt_observe"
                ),
            }
            seed_base = int(torch.initial_seed() % (2**31 - 1))
            if str(refit_tier) != "tier0":
                seed_base += 1000003
            payloads: List[dict] = []
            for spec in specs:
                spec_payload = _spec_to_payload(spec)
                payload_i = dict(base_payload)
                payload_i["spec"] = spec_payload
                payload_i["seed"] = int((seed_base + int(spec_payload["slice_id"])) % (2**31 - 1))
                payloads.append(payload_i)
            return payloads

        def _run_refit_pass(pass_epochs: int, refit_tier: str):
            executor = CoEWitnessExecutor.from_config(self)
            parallel_disabled_reason = _process_refit_disabled_reason()

            def _worker(job) -> dict:
                spec = job.payload
                row = {
                    "method": "refit_compare",
                    "refit_tier": str(refit_tier),
                    "slice_id": int(spec.slice_id),
                    "train_rows": [int(spec.train_start), int(spec.train_stop)],
                    "val_rows": [int(spec.val_start), int(spec.val_stop)],
                    "epochs": int(pass_epochs),
                    "status": "error",
                }
                try:
                    train_loader = _slice_loader(int(spec.train_start), int(spec.train_stop))
                    val_loader = _slice_loader(int(spec.val_start), int(spec.val_stop))
                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    self.dataset_ids = [f"coe_slice_{int(spec.slice_id)}"]
                    self.loss_scales = None
                    self.agg_mode = "mean"
                    self.agg_weights = None
                    self._cache = {}

                    _restore_rng_state(reference_rng_state)
                    slice_rng_state = _snapshot_rng_state()
                    inc_fit = self.fit_candidate(
                        Candidate(
                            label=f"coe_refit_incumbent_{refit_tier}",
                            root=clone_ast(self.state.root),
                            meta={"_reuse_override": dict(self.state.reuse or {})},
                        ),
                        epochs_override=pass_epochs,
                    )
                    _restore_rng_state(slice_rng_state)
                    cand_fit = self.fit_candidate(
                        Candidate(
                            label=f"coe_refit_candidate_{refit_tier}",
                            root=clone_ast(cand_state.root),
                            meta={"_reuse_override": dict(cand_state.reuse or {})},
                        ),
                        epochs_override=pass_epochs,
                    )
                    inc_compare_loss = float(inc_fit.val_loss)
                    cand_compare_loss = float(cand_fit.val_loss)
                    inc_raw_y_mse = float("nan")
                    cand_raw_y_mse = float("nan")
                    comparison_space = "fit_space"
                    if compare_original_y:
                        if y_transform_active:
                            from nestynet_sr.sr_search.stageB.evaluation import (
                                _eval_original_y_mse_with_inverse,
                            )

                            inc_raw_y_mse = _eval_original_y_mse_with_inverse(
                                inc_fit.model,
                                val_loader,
                                self.device,
                                y_op_inv,
                            )
                            cand_raw_y_mse = _eval_original_y_mse_with_inverse(
                                cand_fit.model,
                                val_loader,
                                self.device,
                                y_op_inv,
                            )
                        else:
                            inc_raw_y_mse = _eval_yspace_mse(
                                inc_fit.model,
                                val_loader,
                                self.device,
                            )
                            cand_raw_y_mse = _eval_yspace_mse(
                                cand_fit.model,
                                val_loader,
                                self.device,
                            )
                        if not (
                            math.isfinite(float(inc_raw_y_mse))
                            and math.isfinite(float(cand_raw_y_mse))
                        ):
                            raise ValueError(
                                "non-finite original-y MSE under transformed committee refit"
                            )
                        inc_compare_loss = float(inc_raw_y_mse)
                        cand_compare_loss = float(cand_raw_y_mse)
                        comparison_space = "raw_y"
                    row.update(
                        {
                            "status": "success",
                            "n_train": int(spec.train_stop - spec.train_start),
                            "n_val": int(spec.val_stop - spec.val_start),
                            "incumbent_val_loss": float(inc_fit.val_loss),
                            "candidate_val_loss": float(cand_fit.val_loss),
                            "incumbent_compare_loss": float(inc_compare_loss),
                            "candidate_compare_loss": float(cand_compare_loss),
                            "comparison_space": str(comparison_space),
                            "x_coordinate_space": "internal_x" if xcoords_active else "raw_x",
                            "x_transform_active": bool(xcoords_active),
                            "fit_y_link": str(fit_link_name) if fit_link_active else None,
                            "y_transform_active": bool(y_transform_active),
                        }
                    )
                    if compare_original_y:
                        row.update(
                            {
                                "incumbent_raw_y_mse": float(inc_raw_y_mse),
                                "candidate_raw_y_mse": float(cand_raw_y_mse),
                                "incumbent_fit_loss": float(inc_fit.val_loss),
                                "candidate_fit_loss": float(cand_fit.val_loss),
                            }
                        )
                    if (
                        str(getattr(self, "coe_inference", "legacy") or "legacy")
                        == "maxt_observe"
                    ):
                        from nestynet_sr.sr_search.coe_witness import (
                            stageB_refit_row_losses,
                        )

                        row_inv = y_op_inv if y_transform_active else None
                        row["incumbent_row_losses"] = stageB_refit_row_losses(
                            inc_fit.model, val_loader, self.device, y_op_inv=row_inv
                        )
                        row["candidate_row_losses"] = stageB_refit_row_losses(
                            cand_fit.model, val_loader, self.device, y_op_inv=row_inv
                        )
                except Exception as exc:
                    msg = (
                        f"slice {int(spec.slice_id)} {refit_tier} refit failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    row["error"] = msg
                finally:
                    _restore_context()
                return row

            def _paired_from_rows(rows_i: List[dict]) -> List[dict]:
                return [row for row in rows_i if row.get("status") == "success"]

            def _stop_after(rows_i: List[dict]) -> bool:
                if len(rows_i) < initial_slices:
                    return False
                summary_i = _summary(
                    rows_i,
                    _paired_from_rows(rows_i),
                    pass_epochs=pass_epochs,
                    refit_tier=refit_tier,
                )
                decision_i = _decision(summary_i, final=len(rows_i) >= len(specs))
                return decision_i is not None

            jobs = coe_witness_jobs_from_specs(specs, prefix=f"stageB_refit_{refit_tier}")
            attempted_process = False
            if int(getattr(executor, "parallelism", 1) or 1) > 1 and parallel_disabled_reason is None:
                attempted_process = True
                try:
                    portable_payloads = _build_portable_refit_payloads(pass_epochs, refit_tier)
                    if not portable_payloads:
                        pass_results = []
                        parallel_disabled_reason = "stageB_refit_process_payload_empty"
                    else:
                        preflight_row = run_stageB_refit_pair_witness_preflight(
                            payload=portable_payloads[0],
                            prefix=f"stageB_refit_{refit_tier}",
                        )
                        if preflight_row.get("status") != "success":
                            preflight_error = summarize_witness_errors([preflight_row])
                            parallel_disabled_reason = (
                                "stageB_refit_portable_preflight_failed: "
                                f"{preflight_error}"
                            )
                            self.log(
                                f"[CoE StageB refit] portable process preflight failed for "
                                f"{label}; using live serial refit; {preflight_error}"
                            )
                            pass_results = []
                        else:
                            pass_results = run_stageB_refit_pair_witnesses(
                                payloads=portable_payloads,
                                executor=executor,
                                prefix=f"stageB_refit_{refit_tier}",
                                stop_after=_stop_after,
                            )
                except Exception as exc:
                    pass_results = []
                    parallel_disabled_reason = (
                        f"stageB_refit_process_payload_failed: {type(exc).__name__}: {exc}"
                    )
                if parallel_disabled_reason is not None:
                    fallback_reason = parallel_disabled_reason
                    pass_results = executor.run(jobs, _worker, stop_after=_stop_after)
                    for row in pass_results:
                        row["executor_fallback_reason"] = fallback_reason
                elif not pass_results:
                    fallback_reason = "stageB_refit_process_payload_empty"
                    pass_results = executor.run(jobs, _worker, stop_after=_stop_after)
                    for row in pass_results:
                        row["executor_fallback_reason"] = fallback_reason
                    parallel_disabled_reason = fallback_reason
                elif not _paired_from_rows(pass_results):
                    fallback_reason = "stageB_refit_process_payload_no_successful_pairs"
                    process_errors = summarize_witness_errors(pass_results)
                    self.log(
                        f"[CoE StageB refit] process refit produced no successful pairs for "
                        f"{label}; first process errors: {process_errors}; "
                        "retrying live serial refit"
                    )
                    pass_results = executor.run(jobs, _worker, stop_after=_stop_after)
                    for row in pass_results:
                        row["executor_fallback_reason"] = fallback_reason
                    parallel_disabled_reason = fallback_reason
            else:
                pass_results = executor.run(jobs, _worker, stop_after=_stop_after)
            pass_paired = _paired_from_rows(pass_results)
            pass_warnings = [
                str(row.get("error"))
                for row in pass_results
                if row.get("status") == "error" and row.get("error")
            ]
            pass_summary: dict = {}
            pass_decision: Optional[str] = None
            if len(pass_results) >= initial_slices:
                pass_summary = _summary(
                    pass_results,
                    pass_paired,
                    pass_epochs=pass_epochs,
                    refit_tier=refit_tier,
                )
                pass_summary["witness_executor"] = coe_witness_execution_metadata(
                    executor,
                    pass_results,
                    parallel_disabled_reason=parallel_disabled_reason,
                )
                fallback_reasons = [
                    str(row.get("executor_fallback_reason"))
                    for row in pass_results
                    if row.get("executor_fallback_reason")
                ]
                if fallback_reasons:
                    pass_summary["witness_executor"]["parallel_fallback_reason"] = fallback_reasons[0]
                if attempted_process:
                    pass_summary["witness_executor"]["process_payload_attempted"] = True
                pass_decision = _decision(
                    pass_summary,
                    final=len(pass_results) >= len(specs),
                )
            return pass_results, pass_paired, pass_summary, pass_decision, pass_warnings

        results, paired, summary, decision, warnings = _run_refit_pass(epochs, "tier0")
        if decision == "veto" and escalate_epochs > epochs:
            tier0_summary = dict(summary or {})
            self.log(
                f"[CoE StageB refit] escalating {label}: tier0 would veto; "
                f"retrying with epochs={escalate_epochs}"
            )
            (
                tier1_results,
                tier1_paired,
                tier1_summary,
                tier1_decision,
                tier1_warnings,
            ) = _run_refit_pass(escalate_epochs, "tier1")
            if tier1_paired:
                results = list(results) + list(tier1_results)
                paired = tier1_paired
                summary = dict(tier1_summary or {})
                summary["tier0_summary"] = tier0_summary
                summary["tier1_escalated"] = True
                decision = tier1_decision
                warnings = list(warnings) + list(tier1_warnings)
            else:
                return _legacy_allow(
                    "Tier-1 refit escalation produced no paired successful evaluations",
                    results=list(results) + list(tier1_results),
                    summary={"tier0_summary": tier0_summary, "tier1_escalated": True},
                    warnings=list(warnings) + list(tier1_warnings),
                )

        if not paired:
            return _legacy_allow(
                "no paired successful short-refit slice evaluations",
                results=results,
                warnings=warnings,
            )

        if not summary:
            summary = _summary(
                results,
                paired,
                pass_epochs=epochs,
                refit_tier="tier0",
            )
        if str(getattr(self, "coe_inference", "legacy") or "legacy") == "maxt_observe":
            # Observe-only: calibrated paired max-T over the refit rows with
            # CLUSTER keying by slice, because all rows of one slice share
            # that slice's short-refit fit noise.  Never changes the decision.
            try:
                from nestynet_sr.stat_selection.committee_inference import (
                    maxt_decision_from_slice_rows,
                )

                baseline_rows_map: dict = {}
                candidate_rows_map: dict = {}
                for pair_row in paired:
                    inc_rows = pair_row.get("incumbent_row_losses")
                    cand_rows = pair_row.get("candidate_row_losses")
                    val_rows = pair_row.get("val_rows") or [None, None]
                    if inc_rows is None or cand_rows is None or val_rows[0] is None:
                        continue
                    sid = int(pair_row["slice_id"])
                    baseline_rows_map[sid] = (int(val_rows[0]), inc_rows)
                    candidate_rows_map[sid] = (int(val_rows[0]), cand_rows)
                if baseline_rows_map:
                    maxt = maxt_decision_from_slice_rows(
                        baseline_rows=baseline_rows_map,
                        member_rows={"candidate": candidate_rows_map},
                        seed=int(getattr(self, "coe_maxt_seed", 0) or 0),
                        cluster_by_slice=True,
                    )
                    maxt_gate = (
                        "veto" if maxt.verdict_for("candidate") == "worse" else "allow"
                    )
                    legacy_gate = "veto" if decision == "veto" else "allow"
                    summary["maxt_observe"] = {
                        **maxt.to_dict(),
                        "cluster_by_slice": True,
                        "n_clusters": len(baseline_rows_map),
                        "maxt_gate_equivalent": maxt_gate,
                        "legacy_gate": legacy_gate,
                        "agrees_with_legacy": bool(maxt_gate == legacy_gate),
                    }
                    if maxt_gate != legacy_gate:
                        self.log(
                            f"[CoE maxt-observe] DISAGREES with legacy refit gate "
                            f"for {label}: legacy={legacy_gate}, maxt={maxt_gate} "
                            f"(delta={maxt.member_verdicts[0].mean_delta:.3e}, "
                            f"CI=[{maxt.member_verdicts[0].ci_lower:.3e}, "
                            f"{maxt.member_verdicts[0].ci_upper:.3e}], "
                            f"G={maxt.n_units}, clusters={len(baseline_rows_map)})"
                        )
                else:
                    summary["maxt_observe"] = {
                        "status": "unavailable",
                        "reason": "no paired per-row refit losses",
                    }
            except Exception as exc:
                summary["maxt_observe"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        wins = int(summary.get("wins", 0) or 0)
        ties = int(summary.get("ties", 0) or 0)
        losses = int(summary.get("losses", 0) or 0)
        med_delta = float(summary.get("median_delta", float("inf")))
        med_tol = float(summary.get("median_tolerance", 0.0) or 0.0)
        if decision == "veto":
            gate_record_fn(
                outcome="veto",
                gate_status="refit_veto",
                decision_reason="candidate loses committee short-refit comparison",
                incumbent_expr=incumbent_snapshot,
                candidate_expr=candidate_snapshot,
                results=results,
                summary=summary,
                warnings=warnings,
            )
            self.log(
                f"[CoE StageB refit] veto {label}: "
                f"wins/ties/losses={wins}/{ties}/{losses}, "
                f"median Δ={med_delta:.3e} > tol={med_tol:.3e}, "
                f"slices={summary.get('evaluated_slices')}/{summary.get('n_slices')}, "
                f"epochs={epochs}"
            )
            return False, (
                "reject-coe-stageB-refit-gate("
                f"wins/ties/losses={wins}/{ties}/{losses}, "
                f"median_delta={med_delta:.3e})"
            )

        gate_record_fn(
            outcome="allow",
            gate_status="refit_accepted",
            decision_reason="candidate passes committee short-refit comparison",
            incumbent_expr=incumbent_snapshot,
            candidate_expr=candidate_snapshot,
            results=results,
            summary=summary,
            warnings=warnings,
        )
        self.log(
            f"[CoE StageB refit] allow {label}: "
            f"wins/ties/losses={wins}/{ties}/{losses}, "
            f"median Δ={med_delta:.3e}, tol={med_tol:.3e}, "
            f"slices={summary.get('evaluated_slices')}/{summary.get('n_slices')}, "
            f"epochs={epochs}"
        )
        return True, (
            "coe-stageB-refit-gate-accepted("
            f"wins/ties/losses={wins}/{ties}/{losses})"
        )

    def pareto_front_records(
        self,
        *,
        outcomes: Optional[Set[str]] = None,
        max_records: Optional[int] = None,
    ) -> List[dict]:
        """Return non-dominated decision-log records by (loss, complexity)."""
        allowed = set(outcomes) if outcomes is not None else {"accept", "reject"}
        rows: List[dict] = []
        points: List[Tuple[float, float]] = []
        for rec in self.decision_log:
            if rec.get("outcome") not in allowed:
                continue
            if not bool(rec.get("pareto_trackable", False)):
                continue
            try:
                loss = float(rec.get("cand_loss"))
                cx = float(rec.get("cand_complexity_total"))
            except Exception:
                continue
            if not (math.isfinite(loss) and math.isfinite(cx)):
                continue
            rows.append(rec)
            points.append((loss, cx))

        if not rows:
            return []

        keep_idx = _pareto_front_indices_2d(points)
        front = [rows[i] for i in keep_idx]
        front.sort(
            key=lambda r: (
                float(r.get("cand_loss", float("inf"))),
                float(r.get("cand_complexity_total", float("inf"))),
                int(r.get("n_params_cand", int(1e18))),
                int(r.get("step", int(1e18))),
            )
        )
        if max_records is not None:
            try:
                k = max(0, int(max_records))
            except Exception:
                k = 0
            if k > 0:
                front = front[:k]
        return front

    def log_pareto_summary(
        self,
        *,
        outcomes: Optional[Set[str]] = None,
        max_records: int = 8,
    ) -> None:
        """Log a compact Pareto summary for auditability."""
        allowed = set(outcomes) if outcomes is not None else {"accept", "reject"}
        trackable = [
            rec
            for rec in self.decision_log
            if rec.get("outcome") in allowed and bool(rec.get("pareto_trackable", False))
        ]
        if not trackable:
            return

        front = self.pareto_front_records(outcomes=allowed, max_records=max_records)
        if not front:
            return

        self.log(
            f"[Stage B] Pareto(loss,complexity): front={len(front)} "
            f"trackable={len(trackable)} outcomes={sorted(allowed)}"
        )
        for rec in front:
            try:
                loss = float(rec.get("cand_loss"))
                cx = float(rec.get("cand_complexity_total"))
                ast = float(rec.get("cand_ast_cost"))
                mcost = float(rec.get("cand_mapping_cost"))
            except Exception:
                continue
            mk = rec.get("cand_mapping_kind") or "-"
            mc = rec.get("cand_mapping_class") or "-"
            np = rec.get("n_params_cand")
            self.log(
                f"[Stage B]   Pareto step={rec.get('step')} {rec.get('outcome')} "
                f"{rec.get('label')}: loss={loss:.3e} cx={cx:.3f} "
                f"(ast={ast:.3f}, map={mcost:.3f}:{mk}/{mc}) params={np}"
            )

    def is_pattern_disabled(self, pattern_name: Any) -> bool:
        """Check if a pattern or candidate is disabled.

        Matches both exact candidate labels and their family label
        (e.g. disabling ``ratpoly`` also disables ``ratpoly[1]``).
        """
        raw = getattr(pattern_name, "label", pattern_name)
        raw_s = "" if raw is None else str(raw)
        family = candidate_pattern_name(pattern_name)
        return raw_s in self.disabled_patterns or family in self.disabled_patterns

    def _x_display_labels(self) -> Optional[Dict[int, str]]:
        """Build axis -> display_label map from stageA_x_transforms, if any.

        Used to show e.g. ``cos(x2)`` instead of bare ``x2`` in compact
        expression displays when Stage A detected an x-transform.
        """
        xm = getattr(self, "stageA_x_transforms", None)
        if not xm:
            return None
        labels: Dict[int, str] = {}
        for axis, spec in xm.items():
            pipe = spec.get("pipeline", []) if isinstance(spec, dict) else []
            scales: list = []
            fns: list = []
            for step in pipe:
                kind = str(step.get("kind", "")).lower().strip()
                if kind == "scale":
                    s = step.get("scale", 1.0)
                    if s != 1.0:
                        scales.append(f"{s:.4g}*")
                elif kind:
                    fns.append(f"{kind}(")
            if scales or fns:
                inner = f"x{int(axis)}"
                if scales:
                    inner = "".join(scales) + inner
                for fn in reversed(fns):
                    inner = fn + inner + ")"
                labels[int(axis)] = inner
        return labels or None

    def was_attempted(self, rule_name: str, signature: Tuple[int, ...]) -> bool:
        """Check if this transformation was already attempted."""
        return signature in self.attempted_transformations.get(rule_name, set())

    def record_attempt(self, rule_name: str, signature: Tuple[int, ...]):
        """Record that this transformation was attempted."""
        if rule_name not in self.attempted_transformations:
            self.attempted_transformations[rule_name] = set()
        self.attempted_transformations[rule_name].add(signature)

    def _normalise_signature(self, sig: Any) -> Optional[Tuple[int, ...]]:
        """Normalise a candidate signature into a hashable Tuple[int, ...].

        Rules may provide signatures as tuples or lists. We also accept any
        iterable of ints and cast elements to int.
        """
        if sig is None:
            return None
        # Common case: already a tuple of ints
        if isinstance(sig, tuple):
            try:
                return tuple(int(x) for x in sig)
            except Exception:
                return None
        if isinstance(sig, list):
            try:
                return tuple(int(x) for x in sig)
            except Exception:
                return None
        # Fallback: try to iterate
        try:
            return tuple(int(x) for x in sig)
        except Exception:
            return None

    def candidate_signature(self, cand: Candidate) -> Optional[Tuple[int, ...]]:
        """Extract a deduplication signature from a candidate.

        Prefer the explicit Candidate.signature field (Option-B rules). Also
        checks meta["signature"] if present.
        """
        sig = getattr(cand, "signature", None)
        if sig is None and isinstance(getattr(cand, "meta", None), dict):
            sig = cand.meta.get("signature", None)
        return self._normalise_signature(sig)

    def precheck_candidate(self, rule_name: str, cand: Candidate, *, record_attempt: bool = True) -> PrecheckResult:
        """Cheap checks before running an expensive candidate fit.

        Current checks (Step 0):
          - disabled pattern filter
          - deduplication via attempted_transformations (if signature provided)

        Side effect: records the attempt when a signature is provided and the
        candidate is not a duplicate.
        """
        if self.is_pattern_disabled(cand):
            return PrecheckResult(False, reason="disabled-pattern")

        sig = self.candidate_signature(cand)
        if sig is not None and self.was_attempted(rule_name, sig):
            return PrecheckResult(False, reason="duplicate-signature", signature=sig)

        # AST sanity: candidates must be strict trees (no shared Node objects).
        # Shared subtrees create a DAG and can break leaf ordering / chain
        # evaluation (e.g. in linear_refinement), causing hard crashes.
        try:
            ok_tree, why_tree = _check_ast_is_tree(cand.root)
        except Exception as e:
            return PrecheckResult(False, reason=f"ast-treecheck-error: {e}", signature=sig)
        if not ok_tree:
            # Do NOT record the signature attempt: this is a builder bug and
            # may be fixed in code; we want the candidate to be retried after
            # the fix.
            why = why_tree or "ast-not-tree"
            return PrecheckResult(False, reason=f"ast-not-tree: {why}", signature=sig)

        # Optional units straightjacket
        if (
            getattr(self, "enforce_units", False)
            and getattr(self, "units_spec", None) is not None
            and check_units_ast is not None
        ):
            try:
                res = check_units_ast(cand.root, self.units_spec)
            except Exception as e:
                return PrecheckResult(False, reason=f"units-error: {e}", signature=sig)
            if not getattr(res, "ok", False):
                why = getattr(res, "reason", "units-reject") or "units-reject"
                return PrecheckResult(False, reason=f"units: {why}", signature=sig)
            problems = _find_nonsense_units_leaves(
                cand.root,
                units_spec=self.units_spec,
                enforce_units=True,
                mutate=False,
            )
            if problems:
                first = problems[0]
                tag = first.get("tag")
                tag_s = f" {tag}" if tag else ""
                inputs = first.get("inputs") or []
                if isinstance(inputs, (list, tuple)):
                    inputs_s = ",".join(str(x) for x in inputs)
                else:
                    inputs_s = str(inputs)
                basis = first.get("basis_dims") or ["dimless-only"]
                return PrecheckResult(
                    False,
                    reason=(
                        "units: nn-output-unreachable"
                        f"{tag_s} inputs=({inputs_s})"
                        f" target={first.get('target_dim')}"
                        f" basis={basis}"
                    ),
                    signature=sig,
                )

        # Record now: "attempt" means "we are about to spend work fitting it".
        # Some callers (e.g. screening) may want to run the static checks
        # without committing the attempt to the dedup registry.
        if (sig is not None) and bool(record_attempt):
            self.record_attempt(rule_name, sig)

        return PrecheckResult(True, signature=sig)

    def infer_target_dim(self, target):
        """Infer required output dimension of a target atom, or None if underdetermined.

        Results are cached per (root identity, target identity/tag) and
        automatically invalidated when the AST root changes.
        """
        if not getattr(self, "enforce_units", False):
            return None
        spec = getattr(self, "units_spec", None)
        if spec is None or infer_atom_output_dim is None:
            return None
        # Cache management: invalidate when AST root changes.
        root_id = id(self.state.root)
        if getattr(self, "_dim_cache_root_id", None) != root_id:
            self._dim_cache: Dict = {}
            self._dim_cache_root_id = root_id
        cache_key = (id(target), getattr(target, "tag", None))
        if cache_key in self._dim_cache:
            return self._dim_cache[cache_key]
        try:
            result = infer_atom_output_dim(self.state.root, target, spec)
        except Exception:
            result = None
        # Level 2 fallback: use Buckingham-Sudoku constraint solver
        if result is None and compute_node_domains is not None:
            try:
                from dataclasses import replace as _dc_replace
                span_spec = _dc_replace(spec, nn_semantics="span")
                domains = compute_node_domains(self.state.root, span_spec)
                if domains is not None:
                    dom = domains.get(id(target))
                    if dom is not None and dom.is_pinned():
                        result = dom.offset
            except Exception:
                pass
        self._dim_cache[cache_key] = result
        return result

    def fit_candidate(self, cand: Candidate, *, epochs_override: Optional[int] = None) -> StageBState:
        """
        Fit a candidate rewrite by rebuilding the model and running LM optimization.

        Args:
            cand: Candidate with new AST root and optional init function

        Returns:
            New StageBState with fitted model and validation loss
        """
        # Import here to avoid circular dependency
        from .fitting import _fit_candidate_root, _fit_candidate_root_multi

        epochs_stageB_eff = self.epochs_stageB
        if epochs_override is not None:
            try:
                epochs_stageB_eff = max(1, int(epochs_override))
            except Exception:
                epochs_stageB_eff = self.epochs_stageB

        blocked_tags: Set[str] = set()
        reuse_override = None
        reuses_override = None
        if isinstance(getattr(cand, "meta", None), dict):
            blocked_tags = {
                str(t) for t in cand.meta.get("reuse_blacklist_tags", [])
                if t is not None
            }
            reuse_override = cand.meta.get("_reuse_override", None)
            reuses_override = cand.meta.get("_reuses_override", None)

        is_multi = isinstance(self.train_loader, (list, tuple))
        if is_multi:
            reuse_list = (
                reuses_override
                if isinstance(reuses_override, (list, tuple))
                else (
                    self.state.reuses
                    if (getattr(self.state, "reuses", None) is not None)
                    else None
                )
            )
            if reuse_list is None:
                base_reuse = reuse_override if isinstance(reuse_override, dict) else self.state.reuse
                reuse_list = [base_reuse for _ in range(len(self.train_loader))]
            reuse_list = [
                _filter_reuse_map(r, blocked_tags)
                for r in reuse_list
            ]
            loss_scales = (
                self.loss_scales
                if (self.loss_scales is not None)
                else [self.loss_scale for _ in range(len(self.train_loader))]
            )
            return _fit_candidate_root_multi(
                root=cand.root,
                reuses=reuse_list,
                train_loaders=list(self.train_loader),
                val_loaders=list(self.val_loader),
                lm_hp=self.lm_hp,
                device=self.device,
                dtype=self.dtype,
                epochs_stageB=epochs_stageB_eff,
                loss_scales=loss_scales,
                trig_by_axis=self.trig_by_axis,
                custom_init_fn=cand.init_fn,
                fresh_nn_factory=self.fresh_nn_factory,
                dataset_ids=self.dataset_ids,
                agg_mode=self.agg_mode,
                agg_weights=self.agg_weights,
                atom_factory=self.atom_factory,
            )

        return _fit_candidate_root(
            root=cand.root,
            reuse=_filter_reuse_map(
                reuse_override if isinstance(reuse_override, dict) else self.state.reuse,
                blocked_tags,
            ),
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            lm_hp=self.lm_hp,
            device=self.device,
            dtype=self.dtype,
            epochs_stageB=epochs_stageB_eff,
            loss_scale=self.loss_scale,
            trig_by_axis=self.trig_by_axis,
            custom_init_fn=cand.init_fn,
            fresh_nn_factory=self.fresh_nn_factory,
            atom_factory=self.atom_factory,
        )

    def gauge_acceptance_gate(
        self,
        cand: Candidate,
        cand_state: StageBState,
        reason: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Reject unresolved-gauge local rewrites that do not improve the scope.

        This is deliberately fitted-candidate-aware: it sees the final candidate
        AST after initialization/fitting/trimming, not merely the proposal.
        """
        meta = cand.meta if isinstance(getattr(cand, "meta", None), dict) else {}
        if bool(meta.get("hidden_gauge_only", False)):
            return False, "reject-hidden-gauge-only"
        additive_confirmed = bool(
            meta.get("additive_gauge_confirmed", meta.get("gauge_confirmed", False))
        )
        if str(meta.get("pattern", "")) == "additive_gauge_transfer" and not additive_confirmed:
            return False, "reject-unconfirmed-additive-gauge-transfer"
        homogeneous_confirmed = bool(meta.get("homogeneous_gauge_confirmed", False))
        if str(meta.get("pattern", "")) == "multiplicative_homogeneity_transfer" and not homogeneous_confirmed:
            return False, "reject-unconfirmed-multiplicative-homogeneity-transfer"

        additive_requires_scope_improvement = bool(
            meta.get(
                "additive_gauge_requires_scope_improvement",
                meta.get("gauge_requires_scope_improvement", False),
            )
        )
        homogeneous_requires_scope_improvement = bool(
            meta.get("homogeneous_gauge_requires_scope_improvement", False)
        )
        if additive_requires_scope_improvement and not homogeneous_requires_scope_improvement:
            try:
                if not collect_nn_atoms(cand_state.root):
                    return True, reason
            except Exception:
                pass
        if additive_requires_scope_improvement:
            before = meta.get("additive_gauge_score_before", meta.get("gauge_score_before"))
            if not isinstance(before, AdditiveGaugeGlobalScore):
                try:
                    before = self.additive_gauge_global_score(self.state.root)
                except Exception:
                    before = None
            try:
                after = self.additive_gauge_global_score(cand_state.root)
            except Exception:
                after = None

            if before is not None and after is not None and after < before:
                reason = f"{reason}; additive-gauge-scope-improved {before}->{after}"
            elif additive_confirmed and bool(
                meta.get("additive_gauge_scope_simplified", meta.get("gauge_scope_simplified", False))
            ):
                pass
            else:
                return False, "reject-unresolved-additive-gauge-local-compression"

        if homogeneous_requires_scope_improvement:
            before_h = meta.get("homogeneous_gauge_score_before")
            if not isinstance(before_h, HomogeneousGaugeGlobalScore):
                try:
                    before_h = self.homogeneous_gauge_global_score(self.state.root)
                except Exception:
                    before_h = None
            try:
                after_h = self.homogeneous_gauge_global_score(cand_state.root)
            except Exception:
                after_h = None

            if before_h is not None and after_h is not None and after_h < before_h:
                return True, f"{reason}; homogeneous-gauge-scope-improved {before_h}->{after_h}"
            if homogeneous_confirmed and bool(meta.get("homogeneous_gauge_scope_simplified", False)):
                return True, reason
            return False, "reject-unresolved-homogeneous-gauge-local-compression"

        return True, reason

    def should_accept(self, cand: Candidate, cand_state: StageBState) -> Tuple[bool, Optional[str]]:
        """Decide whether to accept a fitted candidate state.

        Returns:
            (ok, reason)

        The reason is intended for *verbose* debugging and is provided for both
        accept and reject outcomes.
        """
        if not math.isfinite(cand_state.val_loss):
            return False, "reject-nonfinite-loss"

        # Check fidelity in y-space when the asinh fit link is active.
        # The asinh transformation compresses large deviations, potentially making
        # terrible fits look acceptable in asinh-space. We verify the candidate
        # is also reasonable in original y-space.
        #
        # The correct scaling uses the asinh Jacobian: d(asinh(y/s))/dy = 1/sqrt(s² + y²).
        # To convert asinh-space loss to y-space units, we multiply by D = s² + y².
        # We use two strategies and accept if either passes:
        #   Strategy A: y_mse_allowed = α * asinh_loss * D_ref  (correct Jacobian scaling)
        #   Strategy B: y_mse_allowed = β * base_y_mse         (baseline-relative guard)
        is_asinh = getattr(self.lm_hp, "fit_y_link", None) == "asinh"
        if is_asinh:
            try:
                val_loader_probe = (
                    self.val_loader[0] if isinstance(self.val_loader, (list, tuple)) else self.val_loader
                )
                y_mse = _eval_yspace_mse(cand_state.model, val_loader_probe, self.device)
                asinh_loss = float(cand_state.val_loss)

                # Get asinh scale and hyperparameters
                s = float(getattr(self.lm_hp, "fit_y_link_scale", 1.0))
                q = float(getattr(self.lm_hp, "asinh_yspace_sanity_quantile", 0.90))
                alpha = float(getattr(self.lm_hp, "asinh_yspace_sanity_factor", 20.0))
                beta = float(getattr(self.lm_hp, "asinh_yspace_regress_factor", 5.0))

                # Strategy A: Correct scaling via D_ref = quantile(s² + y²)
                D_ref = self.cached(
                    ("asinh_yspace_Dref", q, s),
                    lambda: _asinh_yspace_scale_from_loader(val_loader_probe, self.device, s, q)
                )
                y_mse_allowed_A = alpha * max(asinh_loss, 1e-30) * max(D_ref, 1e-30)

                # Strategy B: Baseline-relative guard (don't regress too far from current)
                base_y_mse = self.cached(
                    ("base_y_mse",),
                    lambda: _eval_yspace_mse(self.state.model, val_loader_probe, self.device)
                )
                y_mse_allowed_B = beta * max(base_y_mse, 1e-30)

                # Combined: pass if either strategy allows it
                y_mse_allowed = max(y_mse_allowed_A, y_mse_allowed_B)

                if math.isfinite(y_mse) and math.isfinite(y_mse_allowed) and y_mse > y_mse_allowed:
                    self.log(
                        f"{RED}[Stage B] asinh y-space sanity failed: "
                        f"y-MSE={y_mse:.3e} > allowed={y_mse_allowed:.3e} "
                        f"(asinh={asinh_loss:.3e}, D_ref={D_ref:.3e}, base_y_mse={base_y_mse:.3e}){RESET}"
                    )
                    return False, f"reject-asinh-yspace-sanity(y-MSE={y_mse:.3e})"
            except Exception as e:
                # Don't block acceptance if the sanity check itself fails
                if self.verbose:
                    self.log(f"[Stage B] asinh y-space sanity check error: {e}")

        # Track global best (used as the reference for budgets to avoid drift).
        self.best_val_loss = min(self.best_val_loss, self.state.val_loss)

        base_loss = float(self.state.val_loss)
        cand_loss = float(cand_state.val_loss)
        acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(self.lm_hp, self.loss_scale)
        acceptance_noise_n_eff = getattr(self, "acceptance_noise_n_eff", None)
        if acceptance_noise_n_eff is None:
            acceptance_noise_n_eff = getattr(self.state, "acceptance_noise_n_eff", None)
        if acceptance_noise_n_eff is None:
            acceptance_noise_n_eff = getattr(self.lm_hp, "acceptance_noise_n_eff", None)
        try:
            acceptance_noise_n_eff = float(acceptance_noise_n_eff)
            if (not math.isfinite(acceptance_noise_n_eff)) or acceptance_noise_n_eff <= 0.0:
                acceptance_noise_n_eff = None
        except Exception:
            acceptance_noise_n_eff = None
        base_loss_cmp = _loss_excess_above_floor(base_loss, acceptance_noise_floor_raw)
        cand_loss_cmp = _loss_excess_above_floor(cand_loss, acceptance_noise_floor_raw)
        best_loss_cmp = _loss_excess_above_floor(self.best_val_loss, acceptance_noise_floor_raw)

        # Shared selection hyperparams (defaults live in LMHyperparams).
        count_weight = float(getattr(self.lm_hp, "select_count_weight", 1.0))
        struct_gamma = float(getattr(self.lm_hp, "select_struct_gamma", 0.05))
        param_gamma = float(getattr(self.lm_hp, "select_param_gamma", 0.30))
        sep_bonus = float(getattr(self.lm_hp, "select_sep_bonus_decades", 0.05))
        partial_sep_bonus = float(getattr(self.lm_hp, "select_partial_sep_bonus_decades", 0.02))
        base_bonus = float(getattr(self.lm_hp, "select_base_bonus_decades", 0.0))

        loss_floor = float(self.loss_floor) if self.loss_floor is not None else float(self.loss_good_enough_raw)
        loss_cap = float(self.loss_cap) if self.loss_cap is not None else float("inf")
        floor_guard_dec = float(getattr(self.lm_hp, "select_floor_guard_decades", 2.0))
        max_regress_dec = float(getattr(self.lm_hp, "select_below_floor_max_regress_decades", 1.0))
        try:
            best_ref = float(min(base_loss_cmp, best_loss_cmp))
        except Exception:
            best_ref = float(base_loss_cmp)
        # Optional noisy-data fallback: if explicitly enabled and no external
        # acceptance floor was supplied, don't let the hard cap block
        # near-loss-neutral simplifications while the current best is still
        # above that cap.
        if (
            bool(getattr(self.lm_hp, "stageB_overcap_fallback", False))
            and acceptance_noise_floor_raw <= 0.0
            and math.isfinite(loss_cap)
            and best_ref > loss_cap
        ):
            loss_cap = float("inf")
        loss_floor_eff = _effective_loss_floor(loss_floor, best_ref, floor_guard_dec)
        below_floor_regress_cap = _below_floor_regression_cap(base_loss_cmp, max_regress_dec)

        # Complexity estimates — use effective (post-fit) parameter counts
        # so that de-Padeified ratpolys with pruned coefficients get credit
        # for being simpler than their nominal degree implies.
        n_params_base_nominal = int(self.state.model.num_parameters())
        n_params_cand_nominal = int(cand_state.model.num_parameters())
        try:
            _loader = self.train_loader
            if isinstance(_loader, (list, tuple)):
                _loader = _loader[0]
            _batch = next(iter(_loader))
            if isinstance(_batch, (list, tuple)):
                _x_eff = _batch[0].to(self.device)
            else:
                _x_eff = _batch.to(self.device)
        except Exception:
            _x_eff = torch.empty(0, 0, device=self.device, dtype=self.dtype)
        n_params_base = min(
            n_params_base_nominal,
            _count_effective_params(self.state.model, self.state.root, _x_eff),
        )
        n_params_cand = min(
            n_params_cand_nominal,
            _count_effective_params(cand_state.model, cand_state.root, _x_eff),
        )

        base_key = _complexity_key(self.state.root, n_params_base, count_weight=count_weight)
        cand_key = _complexity_key(cand_state.root, n_params_cand, count_weight=count_weight)
        base_map_cost = _clamp_nonnegative_finite(getattr(self.state, "complexity_mapping_cost", 0.0), default=0.0)
        cand_map_cost = _candidate_mapping_cost(cand)
        base_key = (float(base_key[0]) + float(base_map_cost), int(base_key[1]))
        cand_key = (float(cand_key[0]) + float(cand_map_cost), int(cand_key[1]))

        # Identify separability-like rewrites early (needed for below-floor decision).
        # These may unlock later splits even when they don't immediately reduce
        # the structural proxy.
        is_separability_rewrite = _is_separability_candidate(cand)
        is_problem_prune = cand.label == "nonsense_units_zero_prune"
        is_noise_equiv = bool(
            acceptance_noise_floor_raw > 0.0
            and _noise_equivalent(
                base_loss,
                cand_loss,
                noise_floor=acceptance_noise_floor_raw,
                n_eff=acceptance_noise_n_eff,
            )
        )
        is_complexity_simpler = bool(cand_key < base_key)
        meta = cand.meta if isinstance(getattr(cand, "meta", None), dict) else {}
        noisy_gauge_requires_strict_improvement = bool(
            meta.get("noisy_gauge_requires_strict_improvement", False)
            and acceptance_noise_floor_raw > 0.0
        )
        strictly_improves_loss = bool(cand_loss_cmp + self.score_tol < base_loss_cmp)
        if noisy_gauge_requires_strict_improvement and not strictly_improves_loss:
            return (
                False,
                "reject-noisy-gauge-sideways"
                f"(loss {base_loss:.3e}->{cand_loss:.3e}, "
                f"floor={acceptance_noise_floor_raw:.3e})",
            )

        # When both losses are below the meaningful floor, treat them as equal
        # and pick by complexity — UNLESS this is a separability rewrite that
        # decomposes the problem for further analysis.
        if (base_loss_cmp <= loss_floor_eff) and (cand_loss_cmp <= loss_floor_eff):
            if _below_floor_regression_rejected(
                cand_loss=cand_loss_cmp,
                below_floor_regress_cap=below_floor_regress_cap,
                is_separability_rewrite=is_separability_rewrite,
                relaxed_below_floor=is_problem_prune or (is_noise_equiv and is_complexity_simpler),
            ):
                ratio = cand_loss_cmp / base_loss_cmp if base_loss_cmp > 0 else float("inf")
                return False, (
                    "loss-below-floor-too-much-regression"
                    f"(ratio={ratio:.3e}, cap={below_floor_regress_cap:.3e})"
                )
            if is_complexity_simpler:
                if is_problem_prune:
                    return True, "loss-below-floor-problem-prune"
                return True, "loss-below-floor-simpler"
            if is_separability_rewrite:
                return True, "loss-below-floor-separability-pass"
            # Effective param counts are already used in the keys above
            # (computed upfront for both base and candidate).  If the keys
            # still tie, prefer the candidate with lower loss.
            if cand_key == base_key and cand_loss_cmp < base_loss_cmp:
                return True, "loss-below-floor-effective-tie-better-loss"
            return False, "loss-below-floor-not-simpler"

        # Strict improvement in loss always wins.
        if strictly_improves_loss:
            return True, "better-loss"

        if is_noise_equiv:
            if is_complexity_simpler:
                return (
                    True,
                    "noise-equivalent-simpler"
                    f"(floor={acceptance_noise_floor_raw:.3e}"
                    + (
                        f", n_eff={acceptance_noise_n_eff:.0f})"
                        if acceptance_noise_n_eff is not None
                        else ")"
                    ),
                )
            if is_separability_rewrite:
                return (
                    True,
                    "noise-equivalent-separability-pass"
                    f"(floor={acceptance_noise_floor_raw:.3e}"
                    + (
                        f", n_eff={acceptance_noise_n_eff:.0f})"
                        if acceptance_noise_n_eff is not None
                        else ")"
                    ),
                )

        # Some separability-derived rewrites are only partial (overlapping groups).
        # Stage B usually sees fewer of these than Stage A, but we keep the hook
        # for consistency.
        is_partial_sep = False
        try:
            if cand.meta:
                is_partial_sep = bool(
                    cand.meta.get("partial_sep", False)
                    or cand.meta.get("has_overlap", False)
                    or cand.meta.get("overlap", False)
                )
        except Exception:
            is_partial_sep = False

        # Basic eligibility: either we actually simplify (structurally or by params)
        # OR we are a separability rewrite (given a small bonus allowance).
        try:
            s_base = float(_nn_structural_score(self.state.root, count_weight=count_weight))
            s_cand = float(_nn_structural_score(cand_state.root, count_weight=count_weight))
        except Exception:
            s_base, s_cand = 0.0, 0.0
        simplifies = (s_cand + 1e-12 < s_base) or (n_params_cand < n_params_base)
        if (not simplifies) and (not is_separability_rewrite):
            return False, "reject-no-improvement"

        # Mild exploration allowance when the current best is not yet good-enough.
        # This mirrors the old behaviour (base_loss*1.10) but is expressed in decades.
        explore_bonus = math.log10(1.10) if float(best_loss_cmp) > float(self.loss_good_enough_raw) else 0.0

        threshold = _compute_accept_threshold(
            base_loss=base_loss,
            best_loss=float(self.best_val_loss),
            base_ast=self.state.root,
            cand_ast=cand_state.root,
            base_params=n_params_base,
            cand_params=n_params_cand,
            loss_floor=float(loss_floor_eff),
            loss_cap=float(loss_cap),
            count_weight=float(count_weight),
            struct_gamma=float(struct_gamma),
            param_gamma=float(param_gamma),
            base_bonus_decades=float(base_bonus),
            sep_bonus_decades=float(sep_bonus),
            partial_sep_bonus_decades=float(partial_sep_bonus),
            is_separability=bool(is_separability_rewrite),
            is_partial_separability=bool(is_partial_sep),
            extra_bonus_decades=float(explore_bonus),
            noise_floor=float(acceptance_noise_floor_raw),
        )

        if cand_loss <= threshold:
            # Guard: weak-probe separability candidates must not regress excessively
            # from the actual base_loss (regardless of floor-based budget).
            # When base_loss << loss_floor, the floor-based threshold can be very
            # permissive (e.g. 350x regression from a good model). For candidates
            # flagged as weak probes, apply a tighter cap relative to actual base_loss.
            is_weak_probe = bool(
                cand.meta and cand.meta.get("weak_probe", False)
            ) if isinstance(getattr(cand, "meta", None), dict) else False

            if is_weak_probe and is_separability_rewrite and base_loss > 0:
                weak_cap = base_loss * 100.0  # allow up to 100x regression
                if cand_loss > weak_cap and cand_loss > float(acceptance_noise_floor_raw + loss_floor_eff):
                    ratio = cand_loss / base_loss if base_loss > 0 else float("inf")
                    return False, f"reject-weak-probe-excessive-regression(ratio={ratio:.0f}x, cap={weak_cap:.3e})"

            bud_dec = _simplification_budget_decades(
                base_ast=self.state.root,
                cand_ast=cand_state.root,
                base_params=n_params_base,
                cand_params=n_params_cand,
                count_weight=float(count_weight),
                struct_gamma=float(struct_gamma),
                param_gamma=float(param_gamma),
                base_bonus_decades=float(base_bonus),
                sep_bonus_decades=float(sep_bonus),
                partial_sep_bonus_decades=float(partial_sep_bonus),
                is_separability=bool(is_separability_rewrite),
                is_partial_separability=bool(is_partial_sep),
                extra_bonus_decades=float(explore_bonus),
            )
            return True, f"simpler-within-budget(dec={bud_dec:.3f}, thr={threshold:.3e})"

        return False, f"simpler-over-budget(thr={threshold:.3e})"

    def _post_accept_ratpoly_trim(
        self,
        cand: Candidate,
        cand_state: StageBState,
    ) -> Tuple[Candidate, StageBState]:
        """Locally trim accepted ratpoly-family candidates without changing generic Stage B flow."""
        if not _is_ratpoly_candidate(cand):
            return cand, cand_state
        if not bool(getattr(self.lm_hp, "stageB_ratpoly_trim_enable", True)):
            return cand, cand_state

        original_cand = cand
        original_state = cand_state
        current_cand = cand
        current_state = cand_state
        base_label = str(cand.label)
        trim_epochs = getattr(self.lm_hp, "stageB_ratpoly_trim_epochs", None)
        if trim_epochs is not None:
            try:
                trim_epochs = max(1, int(trim_epochs))
            except Exception:
                trim_epochs = None

        steps: List[Dict[str, Any]] = []
        logged_lookup_miss = False

        def _fit_against_state(base_state_local: StageBState, cand_local: Candidate) -> StageBState:
            base_state_fit = replace(
                base_state_local,
                reuse=_filter_reuse_map(
                    getattr(base_state_local, "reuse", None),
                    {str(t) for t in cand_local.meta.get("reuse_blacklist_tags", [])}
                    if isinstance(getattr(cand_local, "meta", None), dict)
                    else set(),
                ),
                reuses=[
                    _filter_reuse_map(
                        r,
                        {str(t) for t in cand_local.meta.get("reuse_blacklist_tags", [])}
                        if isinstance(getattr(cand_local, "meta", None), dict)
                        else set(),
                    )
                    for r in (getattr(base_state_local, "reuses", None) or [])
                ] if getattr(base_state_local, "reuses", None) is not None else None,
            )
            ctx_fit = copy.copy(self)
            ctx_fit.state = base_state_fit
            ctx_fit._cache = dict(getattr(self, "_cache", {}))
            return ctx_fit.fit_candidate(cand_local, epochs_override=trim_epochs)

        def _accept_against_state(base_state_local: StageBState, cand_local: Candidate, cand_state_local: StageBState):
            ctx_check = copy.copy(self)
            ctx_check.state = base_state_local
            ctx_check.best_val_loss = float(base_state_local.val_loss)
            ctx_check._cache = dict(getattr(self, "_cache", {}))
            try:
                hp_local = copy.copy(self.lm_hp)
                regress_dec = float(getattr(hp_local, "select_below_floor_max_regress_decades", 1.0))
                if (not math.isfinite(regress_dec)) or regress_dec < 20.0:
                    setattr(hp_local, "select_below_floor_max_regress_decades", 20.0)
                ctx_check.lm_hp = hp_local
            except Exception:
                pass
            return ctx_check.should_accept(cand_local, cand_state_local)

        for branch in ("num", "den"):
            while True:
                meta = current_cand.meta if isinstance(getattr(current_cand, "meta", None), dict) else {}
                lookup = _lookup_ratpoly_trim_target(current_state, current_cand)
                if lookup is None:
                    if not logged_lookup_miss:
                        scale_tag = meta.get("ratpoly_scale_tag")
                        target_tag = meta.get("ratpoly_target_tag")
                        self.log(
                            f"[Stage B]    ratpoly trim skipped: no live {meta.get('leaf_kind', 'ratpoly')} leaf found "
                            f"for {scale_tag or target_tag or meta.get('ratpoly_var_idxs', '?')}"
                        )
                        logged_lookup_miss = True
                    break
                _, _, rat_core, _, leaf_kind = lookup
                if str(leaf_kind).lower() == "rratpoly":
                    exps_num = rat_core.exps_num_full.detach().cpu().to(dtype=torch.int64)
                    num_pivot_degree = _ratpoly_num_pivot_degree(exps_num, getattr(rat_core, "lead_pos_num", None))
                else:
                    exps_num = rat_core.exps_num.detach().cpu().to(dtype=torch.int64)
                    num_pivot_degree = None
                exps_den = rat_core.exps_den.detach().cpu().to(dtype=torch.int64)
                coeffs_den = rat_core.coeffs_den.detach().cpu().to(dtype=torch.float64)
                current_support = _format_ratpoly_support(exps_num, exps_den)

                if branch == "num":
                    degree_bands = _ratpoly_degree_bands(exps_num, exclude_degree=num_pivot_degree)
                else:
                    den_pivot_degree = _ratpoly_den_pivot_degree(exps_den, coeffs_den)
                    degree_bands = _ratpoly_degree_bands(exps_den, exclude_degree=den_pivot_degree)

                if not degree_bands:
                    break

                changed = False
                for degree in degree_bands:
                    if branch == "num":
                        trial_num = exps_num[(exps_num.sum(dim=1).to(dtype=torch.int64) != int(degree))]
                        trial_den = exps_den
                    else:
                        trial_num = exps_num
                        trial_den = exps_den[(exps_den.sum(dim=1).to(dtype=torch.int64) != int(degree))]
                    trial_support = _format_ratpoly_support(trial_num, trial_den)
                    trim_cand = _build_rratpoly_degree_trim_candidate(
                        current_state,
                        current_cand,
                        branch=branch,
                        degree=int(degree),
                    )
                    if trim_cand is None:
                        continue
                    trim_sig = self.candidate_signature(trim_cand)
                    if trim_sig is not None:
                        if self.was_attempted("ratpoly_trim", trim_sig):
                            self.log(
                                f"[Stage B]    ratpoly trim skip ({branch} deg={int(degree)}): "
                                f"duplicate support {trial_support}"
                            )
                            continue
                        self.record_attempt("ratpoly_trim", trim_sig)

                    trim_state = _fit_against_state(current_state, trim_cand)
                    if not math.isfinite(float(trim_state.val_loss)):
                        continue

                    local_ok, local_reason = _accept_against_state(current_state, trim_cand, trim_state)
                    anchor_ok, anchor_reason = _accept_against_state(original_state, trim_cand, trim_state)
                    if not (local_ok and anchor_ok):
                        continue

                    self.log(
                        f"[Stage B]    ratpoly trim accept ({branch}) drop degree {int(degree)}: "
                        f"{current_support} -> {trial_support}, loss {_loss_str(current_state.val_loss, self.lm_hp)}"
                        f"->{_loss_str(trim_state.val_loss, self.lm_hp)}"
                    )
                    steps.append({
                        "branch": str(branch),
                        "degree": int(degree),
                        "reason_local": str(local_reason or ""),
                        "reason_anchor": str(anchor_reason or ""),
                    })
                    current_cand = trim_cand
                    current_state = trim_state
                    changed = True
                    break

                if not changed:
                    break

        if not steps:
            return original_cand, original_state

        final_meta = dict(current_cand.meta or {})
        final_meta["ratpoly_trim_steps"] = list(steps)
        current_cand.meta = final_meta
        current_cand.label = base_label
        final_lookup = None
        final_scale_tag = final_meta.get("ratpoly_scale_tag")
        if final_scale_tag:
            final_lookup = _lookup_rratpoly_trim_target(current_state, str(final_scale_tag))
        if final_lookup is not None:
            _, _, final_rat_core, _ = final_lookup
            final_support = _format_ratpoly_support(
                final_rat_core.exps_num_full.detach().cpu().to(dtype=torch.int64),
                final_rat_core.exps_den.detach().cpu().to(dtype=torch.int64),
            )
            self.log(
                f"[Stage B]    ratpoly trim fixed-point: {len(steps)} accepted drop(s), {final_support}"
            )
        else:
            self.log(f"[Stage B]    ratpoly trim fixed-point: {len(steps)} accepted drop(s)")
        return current_cand, current_state

    def _select_ratpoly_candidate(
        self,
        cand: Candidate,
        cand_state: StageBState,
    ) -> Tuple[bool, Candidate, StageBState, Optional[str]]:
        """Compare a ratpoly candidate after giving its trimmed descendants a chance."""
        raw_ok, raw_reason = self.should_accept(cand, cand_state)
        cand_trim, cand_state_trim = self._post_accept_ratpoly_trim(cand, cand_state)
        if cand_trim is cand and cand_state_trim is cand_state:
            return raw_ok, cand, cand_state, raw_reason

        trim_ok, trim_reason = self.should_accept(cand_trim, cand_state_trim)
        if trim_ok:
            return True, cand_trim, cand_state_trim, trim_reason

        if raw_ok:
            self.log(
                f"[Stage B]    ratpoly trim fallback: keeping untrimmed candidate "
                f"(raw={raw_reason or 'accept'}, trimmed={trim_reason or 'reject'})"
            )
            return True, cand, cand_state, raw_reason

        self.log(
            f"[Stage B]    ratpoly trim fallback: both variants rejected "
            f"(raw={raw_reason or 'reject'}, trimmed={trim_reason or 'reject'})"
        )
        return False, cand, cand_state, raw_reason

    def accept(self, cand: Candidate, cand_state: StageBState, reason: str):
        """
        Accept a candidate: update state, track enabled pattern, refresh reuse map, clear cache.

        Args:
            cand: Candidate being accepted
            cand_state: New state to accept
            reason: Human-readable reason for acceptance
        """
        # Import here to avoid circular dependency
        from .atom_mapping import _refresh_reuse_from_state

        self.log(f"[Stage B]    {GREEN}Accepted{RESET} rewrite ({cand.label}, {reason})")
        if cand.label == "nonsense_units_zero_prune":
            before_n = len(getattr(self.state, "problem_leaves", []) or [])
            after_n = len(getattr(cand_state, "problem_leaves", []) or [])
            self.log(
                f"[Stage B]    Resolved {_problem_candidate_desc(cand)} by replacing it "
                f"with 0; unresolved problem leaves {before_n}->{after_n}"
            )

        # Layer 2: save lightweight checkpoint BEFORE overwriting state
        _accept_rec = self.decision_log[-1] if self.decision_log else {}
        _root_copy = copy.deepcopy(self.state.root)  # AST is small, safe to deepcopy
        _model_sd = _checkpoint_state_dict_cpu(self.state.model)
        _reuse_sds = None
        if getattr(self.state, "models", None) is not None:
            _reuse_sds = [
                _checkpoint_state_dict_cpu(m)
                for m in self.state.models
            ]
        ckpt = _Checkpoint(
            root=_root_copy,
            val_loss=float(self.state.val_loss),
            n_params=int(self.state.model.num_parameters()),
            model_state_dict=_model_sd,
            reuse_state_dicts=_reuse_sds,
            enabled_patterns=list(self.enabled_patterns),
            best_val_loss=self.best_val_loss,
            has_structural=self.has_structural,
            decision_log_len=len(self.decision_log) - 1,  # index of the accept record
            decision_step=self._decision_step - 1,         # step before the accept record
            attempted_transformations={k: set(v) for k, v in self.attempted_transformations.items()},
            accept_step=int(_accept_rec.get("step", self._decision_step)),
            accept_rule=str(_accept_rec.get("rule", "")),
            accept_label=str(_accept_rec.get("label", cand.label)),
            accept_target=str(_accept_rec.get("target", "")),
            accept_target_uid=str(_accept_rec.get("target_uid", "")),
            complexity_mapping_cost=float(getattr(self.state, "complexity_mapping_cost", 0.0) or 0.0),
            simplification_path=copy.deepcopy(getattr(self.state, "simplification_path", [])),
            acceptance_noise_floor_raw=float(
                getattr(self.state, "acceptance_noise_floor_raw", 0.0) or 0.0
            ),
            acceptance_noise_n_eff=getattr(self.state, "acceptance_noise_n_eff", None),
            generic_approximant_unpromoted=bool(
                getattr(self.state, "generic_approximant_unpromoted", False)
            ),
        )
        self._checkpoints.append(ckpt)
        if len(self._checkpoints) > self._max_checkpoints:
            self._checkpoints = self._checkpoints[-self._max_checkpoints:]

        # Progress made → forgive amber key
        self._accepts_since_backtrack += 1
        self._last_amber_key = None

        self.enabled_patterns.append(cand.label)
        cand_state.complexity_mapping_cost = float(_candidate_mapping_cost(cand))
        _map_desc = _candidate_mapping_descriptor(cand)
        _has_mapping = bool(_candidate_has_mapping(cand))
        _mapping_struct = bool(_map_desc.get("is_structural", False))
        _accept_structural = bool(_candidate_is_structural_accept(cand))
        self._last_accept_has_mapping = bool(_has_mapping)
        self._last_accept_mapping_structural = bool(_mapping_struct)
        self._last_accept_structural = bool(_accept_structural)
        # Propagate simplification path from old state to new state
        cand_state.simplification_path = list(getattr(self.state, 'simplification_path', []))
        cand_state.loss_scale = float(self.loss_scale)
        cand_state.loss_good_enough_eff = float(self.loss_good_enough_raw)
        cand_state.loss_acceptable_eff = float(self.loss_cap)
        cand_state.acceptance_noise_floor_raw = float(
            _resolve_acceptance_noise_floor_raw(self.lm_hp, self.loss_scale)
        )
        cand_state.acceptance_noise_n_eff = getattr(self, "acceptance_noise_n_eff", None)
        cand_state.coe_stageB_dry_run_log = list(getattr(self, "coe_stageB_dry_run_log", []) or [])
        cand_state.coe_stageB_gate_log = list(getattr(self, "coe_stageB_gate_log", []) or [])
        cand_state.generic_approximant_unpromoted = bool(
            _candidate_is_unpromoted_generic(cand)
        )
        self.state = cand_state
        _annotate_nonsense_units_leaves(
            self.state,
            units_spec=getattr(self, "units_spec", None),
            enforce_units=bool(getattr(self, "enforce_units", False)),
            log_fn=self.log,
            mutate=True,
        )
        # Append the new step to the simplification path
        _path_base_loss = _accept_rec.get("base_loss")
        _path_cand_loss = _accept_rec.get("cand_loss")
        _path_threshold = None
        _reason_str = _accept_rec.get("reason", "")
        if "thr=" in _reason_str:
            try:
                _path_threshold = float(_reason_str.split("thr=")[1].rstrip(")"))
            except Exception:
                pass
        _path_detail = f"rule={_accept_rec.get('rule', '')}, target={_accept_rec.get('target', '')}"
        _coord_display = _accept_rec.get("coordinate_variant_display")
        if _coord_display:
            _path_detail += f", coord={_coord_display}"
        self.state.simplification_path.append({
            "step": len(self.state.simplification_path),
            "stage": "B",
            "action": f"rewrite {cand.label}",
            "expression": ast_to_human_readable(cand_state.root),
            "val_loss": float(cand_state.val_loss),
            "mse_raw": float(_path_cand_loss) if _path_cand_loss is not None else float(cand_state.val_loss),
            "mse_eff": None,
            "complexity_total": float(_accept_rec.get("cand_complexity_total")) if _accept_rec.get("cand_complexity_total") is not None else None,
            "base_loss": float(_path_base_loss) if _path_base_loss is not None else None,
            "threshold": _path_threshold,
            "n_params": int(cand_state.model.num_parameters()),
            "ast_cost": _safe_ast_cost(cand_state.root),
            "detail": _path_detail,
        })
        # Refresh reuse map(s)
        if getattr(self.state, "models", None) is not None:
            models = list(self.state.models)
            reuses = [_refresh_reuse_from_state(self.state.root, m) for m in models]
            self.state.reuses = reuses
            self.state.reuse = reuses[0] if len(reuses) > 0 else {}
            self.state.model = models[0] if len(models) > 0 else self.state.model
        else:
            self.state.reuse = _refresh_reuse_from_state(self.state.root, self.state.model)
        self._cache.clear()
        self._rejected_keys.clear()
        # Also clear the dim inference cache (AST root changed).
        self._dim_cache = {}
        self._dim_cache_root_id = None

        # Track whether a structural rewrite has been seen.
        if _accept_structural:
            self.has_structural = True

        # Update best loss and save lightweight snapshot of best-ever state
        if cand_state.val_loss <= self.best_val_loss:
            self.best_val_loss = cand_state.val_loss
            _bs_root = copy.deepcopy(cand_state.root)
            _bs_sd = _checkpoint_state_dict_cpu(cand_state.model)
            _bs_reuse_sds = None
            if getattr(cand_state, "models", None) is not None:
                _bs_reuse_sds = [
                    _checkpoint_state_dict_cpu(m)
                    for m in cand_state.models
                ]
            self._best_seen = _Checkpoint(
                root=_bs_root,
                val_loss=float(cand_state.val_loss),
                n_params=int(cand_state.model.num_parameters()),
                model_state_dict=_bs_sd,
                reuse_state_dicts=_bs_reuse_sds,
                enabled_patterns=list(self.enabled_patterns),
                best_val_loss=float(cand_state.val_loss),
                has_structural=self.has_structural,
                decision_log_len=0, decision_step=0,  # not needed for best-seen
                attempted_transformations={},
                accept_step=0, accept_rule="", accept_label="", accept_target="", accept_target_uid="",
                complexity_mapping_cost=float(getattr(cand_state, "complexity_mapping_cost", 0.0) or 0.0),
                simplification_path=copy.deepcopy(getattr(cand_state, "simplification_path", [])),
                acceptance_noise_floor_raw=float(
                    getattr(cand_state, "acceptance_noise_floor_raw", 0.0) or 0.0
                ),
                acceptance_noise_n_eff=getattr(cand_state, "acceptance_noise_n_eff", None),
                generic_approximant_unpromoted=bool(
                    getattr(cand_state, "generic_approximant_unpromoted", False)
                ),
            )

        # Report NN metrics after acceptance
        if (
            cand_state.num_nn_atoms is not None
            and cand_state.num_multivar_nn_atoms is not None
            and cand_state.max_nn_arity is not None
        ):
            self.log(
                f"[Stage B]    NN metrics: "
                f"total={cand_state.num_nn_atoms}, "
                f"multivar={cand_state.num_multivar_nn_atoms}, "
                f"max_arity={cand_state.max_nn_arity}"
            )

        # Show current mathematical structure after acceptance
        x_labels = self._x_display_labels()
        try:
            expr_str = _compact_expression_repr(
                cand_state.root, max_length=240, y_op_inv=self.y_op_inv,
                x_labels=x_labels,
            )
            self.log(f"[Stage B]    Current: {expr_str}")
        except Exception as e:
            if self.verbose:
                self.log(f"[Stage B]    Expression display error: {e}")
            # Fallback: try without y_op_inv
            try:
                expr_str = _compact_expression_repr(
                    cand_state.root, max_length=240, y_op_inv=None,
                    x_labels=x_labels,
                )
                self.log(f"[Stage B]    Current: {expr_str}")
            except Exception:
                pass

    def maybe_polish_after_accept(self) -> bool:
        """Polish a newly fully analytic accepted state through Stage B policy.

        This hook is intentionally post-accept and fully analytic only.  It
        avoids expanding the candidate hot loop, but still lets simple algebraic
        cleanup become part of the live Stage B state before the no-NN exit.
        """
        if not bool(getattr(self.lm_hp, "stageB_polish", True)):
            return False
        if bool(getattr(self, "_stageB_polish_running", False)):
            return False
        if getattr(self.state, "num_nn_atoms", None) not in (0, None):
            return False
        try:
            if len(collect_nn_atoms(self.state.root)) > 0:
                return False
        except Exception:
            pass

        try:
            from .polish import StageBPolishConfig, build_fully_analytic_polish_candidate
        except Exception as exc:
            self.log(f"[Stage B polish] skipped: import failed ({exc})")
            return False

        config = StageBPolishConfig(
            enabled=True,
            commit=bool(getattr(self.lm_hp, "stageB_polish_commit", True)),
            max_candidates=int(getattr(self.lm_hp, "stageB_polish_max_candidates", 32) or 32),
            use_subprocess=bool(getattr(self.lm_hp, "stageB_polish_subprocess", True)),
            max_seconds=float(getattr(self.lm_hp, "stageB_polish_max_seconds", 300.0) or 300.0),
            mem_fraction=float(getattr(self.lm_hp, "stageB_polish_mem_fraction", 0.20) or 0.20),
        )
        self._stageB_polish_running = True
        try:
            result = build_fully_analytic_polish_candidate(self, config=config)
        except Exception as exc:
            if isinstance(exc, IndexError) or "index 0 is out of range" in str(exc):
                return False
            self.log(f"[Stage B polish] skipped: fully-analytic polish failed ({exc})")
            return False
        finally:
            self._stageB_polish_running = False

        if result is None:
            try:
                n_params_base = int(self.state.model.num_parameters())
            except Exception:
                n_params_base = -1
            self._record_decision(
                outcome="shadow_reject",
                rule="stageB_polish",
                label="stageB_polish",
                reason="no-acceptable-fully-analytic-polish",
                target="fully_analytic",
                base_loss=float(self.state.val_loss),
                n_params_base=n_params_base,
                base_complexity=list(_nn_multivar_complexity(self.state.root)),
                base_root=self.state.root,
            )
            return False

        cand, cand_state, polish_result, reason = result
        try:
            _snap = _compact_expression_repr(cand_state.root, max_length=400)
        except Exception:
            _snap = str(cand_state.root)
        try:
            n_params_base = int(self.state.model.num_parameters())
        except Exception:
            n_params_base = -1
        try:
            n_params_cand = int(cand_state.model.num_parameters())
        except Exception:
            n_params_cand = -1

        outcome = "accept" if bool(config.commit) else "shadow_accept"
        self._record_decision(
            outcome=outcome,
            rule="stageB_polish",
            label=cand.label,
            reason=reason or getattr(polish_result, "reason", "accepted"),
            target="fully_analytic",
            base_loss=float(self.state.val_loss),
            cand_loss=float(cand_state.val_loss),
            n_params_base=n_params_base,
            n_params_cand=n_params_cand,
            base_complexity=list(_nn_multivar_complexity(self.state.root)),
            cand_complexity=list(_nn_multivar_complexity(cand_state.root)),
            cand=cand,
            ast_snapshot=_snap,
            base_root=self.state.root,
            cand_root=cand_state.root,
            base_mapping_cost=getattr(self.state, "complexity_mapping_cost", 0.0),
            cand_mapping_cost=_candidate_mapping_cost(cand),
        )
        if not bool(config.commit):
            self.log(
                f"[Stage B polish] shadow accept {polish_result.label}: "
                f"{polish_result.expr_before} -> {polish_result.expr_after}"
            )
            return False

        self.log(
            f"[Stage B polish] accepting {polish_result.label}: "
            f"{polish_result.expr_before} -> {polish_result.expr_after}"
        )
        self.accept(cand, cand_state, reason or getattr(polish_result, "reason", "accepted"))
        return True

    def maybe_shadow_polish_subtrees_after_accept(
        self,
        *,
        previous_root: Optional[Node] = None,
        accepted_label: str = "",
    ) -> bool:
        """Shadow-polish newly analytic subtrees after an accepted rewrite.

        Subtree cleanup candidates are respliced into the full AST and scored
        through the normal Stage B acceptance policy, but the live state is not
        mutated unless subtree polish commits are enabled.
        """
        if not bool(getattr(self.lm_hp, "stageB_polish", True)):
            return False
        if not bool(getattr(self.lm_hp, "stageB_polish_subtrees", False)):
            return False
        if bool(getattr(self, "_stageB_subtree_polish_running", False)):
            return False
        try:
            if len(collect_nn_atoms(self.state.root)) == 0:
                return False
        except Exception:
            return False

        try:
            from .polish import StageBPolishConfig, shadow_polish_new_analytic_subtrees
        except Exception as exc:
            self.log(f"[Stage B polish] subtree shadow skipped: import failed ({exc})")
            return False

        config = StageBPolishConfig(
            enabled=True,
            commit=bool(getattr(self.lm_hp, "stageB_polish_subtree_commit", True)),
            max_candidates=int(getattr(self.lm_hp, "stageB_polish_max_candidates", 32) or 32),
            max_subtrees=int(getattr(self.lm_hp, "stageB_polish_max_subtrees", 8) or 8),
            use_subprocess=bool(getattr(self.lm_hp, "stageB_polish_subprocess", True)),
            max_seconds=float(getattr(self.lm_hp, "stageB_polish_max_seconds", 300.0) or 300.0),
            mem_fraction=float(getattr(self.lm_hp, "stageB_polish_mem_fraction", 0.20) or 0.20),
        )
        self._stageB_subtree_polish_running = True
        try:
            results = shadow_polish_new_analytic_subtrees(
                self,
                previous_root=previous_root,
                config=config,
            )
        except Exception as exc:
            self.log(f"[Stage B polish] subtree shadow skipped: polish failed ({exc})")
            return False
        finally:
            self._stageB_subtree_polish_running = False

        if not results:
            return False

        commit_idx = None
        if bool(config.commit):
            best_commit_key = None
            for i, res in enumerate(results):
                if (
                    getattr(res, "full_policy_ok", None) is True
                    and getattr(res, "full_cand", None) is not None
                    and getattr(res, "full_state", None) is not None
                ):
                    cand_state = getattr(res, "full_state", None)
                    try:
                        n_params = int(cand_state.model.num_parameters())
                    except Exception:
                        n_params = -1
                    try:
                        count_weight = float(getattr(self.lm_hp, "select_count_weight", 1.0))
                        cx = _complexity_key(cand_state.root, n_params, count_weight=count_weight)
                    except Exception:
                        cx = (float("inf"), int(1e18))
                    key = (float(cx[0]), int(cx[1]), float(getattr(cand_state, "val_loss", float("inf"))), i)
                    if best_commit_key is None or key < best_commit_key:
                        best_commit_key = key
                        commit_idx = i

        def _record_subtree_polish(res, *, outcome: str):
            reason = str(getattr(res, "reason", "") or "shadow")
            full_reason = getattr(res, "full_policy_reason", None)
            if full_reason and str(full_reason) not in reason:
                reason = f"{reason}; {full_reason}"
            label = str(getattr(res, "label", "stageB_polish_subtree"))
            path = str(getattr(res, "path", "subtree"))
            cand = getattr(res, "full_cand", None)
            cand_state = getattr(res, "full_state", None)
            cand_root = getattr(res, "full_root", None) or getattr(res, "cand_root", None)
            base_root = self.state.root
            cand_loss = getattr(res, "full_val_loss", None)
            if cand_loss is None:
                cand_loss = float(self.state.val_loss)
            try:
                n_params_base = int(self.state.model.num_parameters())
            except Exception:
                n_params_base = -1
            try:
                n_params_cand = int(cand_state.model.num_parameters()) if cand_state is not None else -1
            except Exception:
                n_params_cand = -1
            try:
                ast_snapshot = _compact_expression_repr(cand_root, max_length=400) if cand_root is not None else str(getattr(res, "expr_after", ""))
            except Exception:
                ast_snapshot = str(cand_root if cand_root is not None else getattr(res, "expr_after", ""))
            if outcome == "accept":
                self.log(
                    f"[Stage B polish] subtree accept {path}: "
                    f"{res.expr_before} -> {res.expr_after}; val-loss={float(cand_loss):.4e} ({reason})"
                )
            elif outcome == "shadow_accept":
                self.log(
                    f"[Stage B polish] subtree full-shadow accept {path}: "
                    f"{res.expr_before} -> {res.expr_after}; val-loss={float(cand_loss):.4e} ({reason})"
                )
            elif outcome == "shadow_reject":
                self.log(
                    f"[Stage B polish] subtree full-shadow reject {path}: "
                    f"{res.expr_before} -> {res.expr_after}; val-loss={float(cand_loss):.4e} ({reason})"
                )
            else:
                self.log(
                    f"[Stage B polish] subtree shadow skip {path}: "
                    f"{res.expr_before} ({reason})"
                )
            self._record_decision(
                outcome=outcome,
                rule="stageB_polish_subtree",
                label=label,
                reason=f"{reason}; accepted_after={accepted_label}",
                target=path,
                base_loss=float(self.state.val_loss),
                cand_loss=float(cand_loss),
                n_params_base=n_params_base,
                n_params_cand=n_params_cand,
                base_complexity=list(_nn_multivar_complexity(base_root)) if base_root is not None else None,
                cand_complexity=list(_nn_multivar_complexity(cand_root)) if cand_root is not None else None,
                cand=cand,
                ast_snapshot=ast_snapshot,
                base_root=base_root,
                cand_root=cand_root,
                base_mapping_cost=getattr(self.state, "complexity_mapping_cost", 0.0),
                cand_mapping_cost=_candidate_mapping_cost(cand) if cand is not None else 0.0,
            )
            return reason

        any_shadow_accept = False
        commit_res = None
        for i, res in enumerate(results):
            if i == commit_idx:
                commit_res = res
                continue
            policy_ok = getattr(res, "full_policy_ok", None)
            if policy_ok is True:
                any_shadow_accept = True
                outcome = "shadow_accept"
            elif policy_ok is False:
                outcome = "shadow_reject"
            else:
                outcome = "shadow_skip"
            _record_subtree_polish(res, outcome=outcome)

        if commit_res is not None:
            reason = _record_subtree_polish(commit_res, outcome="accept")
            cand = getattr(commit_res, "full_cand", None)
            cand_state = getattr(commit_res, "full_state", None)
            if cand is not None and cand_state is not None:
                self.accept(cand, cand_state, reason or getattr(commit_res, "full_policy_reason", "accepted"))
                return True

        return bool(any_shadow_accept)
