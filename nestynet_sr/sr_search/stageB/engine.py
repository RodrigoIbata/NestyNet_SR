# ruff: noqa: F401
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Stage B Engine: Core execution engine and context for Stage B refinement.

This module contains:
- StageBState: dataclass holding AST, model, and optimization state
- Candidate: dataclass representing a rewrite candidate
- StageBContext: execution context with data loaders, hyperparameters, and helper methods
- StageBEngine: main execution engine that applies rewrite rules iteratively
"""

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


import inspect as _inspect
import sys as _sys
import types as _types

from . import _engine_support as _support
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

from . import _engine_state as _state
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

from . import _engine_runtime as _runtime
from ._engine_runtime import (
    _find_worst_accept,
    _pick_atom_factory,
    _restore_from_checkpoint,
    StageBEngine,
)

# Complete postponed annotation and runtime dependencies after the acyclic
# module import.  Only Candidate is needed by an executed support body; the
# other bindings preserve historical get_type_hints() behavior.
for _state_name in (
    "StageBRule",
    "StageBState",
    "_Checkpoint",
    "Candidate",
    "PrecheckResult",
    "StageBContext",
):
    setattr(_support, _state_name, globals()[_state_name])

_HISTORICAL_DEFINITIONS = ["_snapshot_rng_state","_restore_rng_state","_safe_ast_cost","_clamp_nonnegative_finite","_loss_excess_above_floor","_effective_loss_floor","_best_seen_restore_decision","_below_floor_regression_cap","_below_floor_regression_rejected","_candidate_mapping_cost","_candidate_is_unpromoted_generic","_mapping_descriptor","_candidate_mapping_descriptor","_candidate_has_mapping","_candidate_is_structural_accept","_phase2_trigger_flags","_target_uid","_eval_yspace_mse","_asinh_yspace_scale_from_loader","_loss_str","_format_dim_for_problem","_target_dim_for_root","_input_basis_dims_for_atom","_find_nonsense_units_leaves","_annotate_nonsense_units_leaves","_problem_candidate_desc","candidate_pattern_name","_count_ast_params","_candidate_min_free_params","_cand_sort_key","_candidate_can_beat_floor_locked_state","_is_exact_final_leaf_monomial_accept","_stageB_state_num_params","_stageB_state_num_nn_atoms","_stageB_completion_loss_floor","_min_following_candidate_free_params","_are_we_done_yet","_are_we_done_yet_reason","_skip_post_accept_polish_for_terminal_state","_count_effective_params","_leaf_z_data","_effective_ratpoly_params","_effective_poly_params","_unwrap_leaf_core","_filter_reuse_map","_find_ratpoly_scale_pair","_ratpoly_degree_bands","_ratpoly_support_degrees","_format_ratpoly_support","_ratpoly_den_pivot_degree","_is_ratpoly_candidate","_ratpoly_exps_key","_ratpoly_support_signature_exact","_ratpoly_num_pivot_degree","_lookup_rratpoly_trim_target","_lookup_ratpoly_trim_target","_build_rratpoly_degree_trim_candidate","_ast_node_to_tuple","_target_arity","atom_content_hash","_is_structural_candidate","_is_separability_candidate","_nn_multivar_complexity","_compute_nn_metrics","StageBRule","StageBState","_Checkpoint","_materialized_fit_state_for_checkpoint","_checkpoint_state_dict_cpu","_is_transient_fit_state_key","_state_value_clone","_load_checkpoint_state_dict","Candidate","PrecheckResult","StageBContext","_find_worst_accept","_pick_atom_factory","_restore_from_checkpoint","StageBEngine"]
_PATCHABLE_GLOBALS = ["_lookup_rratpoly_trim_target","_lookup_ratpoly_trim_target","_build_rratpoly_degree_trim_candidate"]


def _canonicalize_definition(_obj):
    try:
        _obj.__module__ = __name__
    except (AttributeError, TypeError):
        pass
    if _inspect.isclass(_obj):
        for _member in vars(_obj).values():
            _targets = []
            if isinstance(_member, (staticmethod, classmethod)):
                _targets.append(_member.__func__)
            elif isinstance(_member, property):
                _targets.extend(
                    _target
                    for _target in (_member.fget, _member.fset, _member.fdel)
                    if _target is not None
                )
            elif _inspect.isfunction(_member):
                _targets.append(_member)
            for _target in _targets:
                try:
                    _target.__module__ = __name__
                except (AttributeError, TypeError):
                    pass


for _definition_name in _HISTORICAL_DEFINITIONS:
    _canonicalize_definition(globals()[_definition_name])


class _CompatibilityModule(_types.ModuleType):
    """Propagate patches of historical engine globals into split modules."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in _PATCHABLE_GLOBALS:
            setattr(_support, name, value)
            setattr(_state, name, value)
            setattr(_runtime, name, value)


_sys.modules[__name__].__class__ = _CompatibilityModule
