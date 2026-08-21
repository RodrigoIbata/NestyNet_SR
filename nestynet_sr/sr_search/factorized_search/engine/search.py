# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compatibility facade for the factorized-search engine."""

# ruff: noqa: F401, F822


import math
import logging
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from ..basis_scoring import (
    build_scaffold_candidate_score_cfg,
    record_anchor_head_compare,
)
from ..config import coerce_inverse_steering_config
from ..controller import MacroController, build_macro_controller_state
from ..expr_ast import (
    BINARY_OPS,
    UNARY_OPS,
    build_pool,
    collect_paths,
    compute_reachable,
    dim_round,
    dims_eq,
    eval_node,
    get_at,
    is_valid_node,
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
from ..expr_mapping import eval_mapping, fit_best, mapping_is_structural
from ..inverse_core import (
    _normalize_inverse_local_score_mode,
    _normalize_inverse_target_mode,
    eval_mapping_total,
)
from ..inverse_search import (
    estimate_inverse_steering_potential,
    _mapping_cache_signature,
    _pool_cache_signature,
)
from nestynet_sr.sr_search.model_selection import mapping_cost
from ..proposal_families.runner import run_closure_search_pass_impl as _run_closure_search_pass_impl
from ..opportunity_critic import load_opportunity_bundle, predict_opportunity_slate
from ..scheduler import build_plan_candidates, choose_plan
from ..scheduler_critic import load_scheduler_bundle
from nestynet_sr.sr_expr_ir.config import ExpressionIRConfig, coerce_expr_ir_config
from nestynet_sr.sr_expr_ir.reporting import expression_ir_report
from nestynet_sr.sr_expr_ir.stats import ExpressionIRStats
from ..proposal_families.gs import (
    build_gs_fss_context,
    coerce_gs_fss_context,
    extend_pool_with_gs_atoms,
    gs_fss_report,
)
from ..shared_opportunity import (
    normalize_realized_witness_energy_fields,
    normalize_witness_energy_fields,
)
from ..policy.guidance import (
    _annotate_inverse_experiment_lineage,
    _choose_repair_execution_preview,
    _credible_route_compare_decision,
    _credible_route_preview_repair_opportunity_rows,
    _controller_build_slate_id,
    _derived_controller_build_rng,
    _repair_route_compare_decision,
)
from ..policy.build_slate import (
    collect_controller_build_slate as _collect_controller_build_slate_impl,
    controller_selected_action_path as _controller_selected_action_path_impl,
    normalize_controller_build_slate_actions as _normalize_controller_build_slate_actions_impl,
)
from ..policy.features import RepairControllerFeatureRecord, build_controller_state_record
from ..policy.parent_selection import choose_parent, choose_parent_repair_aware
from ..repair_critic import (
    load_repair_critic_bundle,
    predict_repair_build_route,
    predict_repair_controller_heads,
)
from ..repair_policy import (
    _actor_critic_reward_terms,
    _analytic_repair_controller_score,
    _hybrid_repair_controller_scores,
    _normalize_repair_controller_critic_mode,
    _repair_controller_component_gate,
    _repair_controller_path_policy,
    _repair_controller_stagnation_state,
    _repair_controller_threshold,
    _repair_controller_weights,
    _repair_option_candidate_paths,
    _repair_parent_preview_retry_gate,
    _repair_parent_record_attempt,
    _repair_parent_retry_gate,
    _repair_preview_signature,
)
from ..shared_candidate import shared_candidate_row_dict
from .actions import (
    ACTION_ID_BY_NAME,
    ACTION_NAME,
    A_ADD_RAND,
    A_BOOST,
    A_CROSSOVER,
    A_HOLESEARCH,
    A_INVSTEER,
    A_MUL_RAND,
    A_PRUNE,
    A_REPAIR,
    A_REPLACE,
    A_RESIDUAL,
    A_WRAP_UNARY,
    apply_action_impl as _apply_action_impl,
    apply_crossover_action_impl as _apply_crossover_action_impl,
    apply_residual_action_impl as _apply_residual_action_impl,
)
from .archive import ResidualBasinArchive
from .proposal_execution import (
    ProposalScoringState,
    merge_route_status_counts as _merge_route_status_counts_impl,
    record_route_status as _record_route_status_impl,
    run_closure_search_pass as _run_closure_search_pass_route,
    score_native_candidate_basis_state as _score_native_candidate_basis_state_impl,
    score_external_candidate_expr as _score_external_candidate_expr_impl,
)
from .scoring import (
    _eval_node_hparam_safe as _engine_eval_node_hparam_safe,
    _harvest_pool_from_archive as _engine_harvest_pool_from_archive,
    score_expr as _engine_score_expr,
)
from .signals import CandidateStateFeatures, InverseSteeringPotential

import inspect as _inspect
from functools import wraps as _wraps
from types import ModuleType as _ModuleType

from . import _search_runtime as _runtime
from . import _search_state as _state
from . import _search_support as _support

_GROUP_MODULES = (_support, _state, _runtime)
_IMPLEMENTATIONS = {
    name: getattr(module, name)
    for module in _GROUP_MODULES
    for name in module.__engine_search_definitions__
}
_PATCHABLE_GLOBAL_NAMES = (
    "_periodogram_frequency_hints",
    "_run_closure_search_pass_impl",
)


def _sync_patchable_globals() -> None:
    facade_globals = globals()
    for module in _GROUP_MODULES:
        for name in _PATCHABLE_GLOBAL_NAMES:
            if hasattr(module, name):
                setattr(module, name, facade_globals[name])


def _wrapper_parameter_source(signature: _inspect.Signature) -> tuple[str, str]:
    parameters = list(signature.parameters.values())

    def render(parameter: _inspect.Parameter) -> str:
        text = parameter.name
        if parameter.default is not _inspect.Parameter.empty:
            text += "=None"
        return text

    positional_only = [p for p in parameters if p.kind is _inspect.Parameter.POSITIONAL_ONLY]
    positional_or_keyword = [p for p in parameters if p.kind is _inspect.Parameter.POSITIONAL_OR_KEYWORD]
    var_positional = next((p for p in parameters if p.kind is _inspect.Parameter.VAR_POSITIONAL), None)
    keyword_only = [p for p in parameters if p.kind is _inspect.Parameter.KEYWORD_ONLY]
    var_keyword = next((p for p in parameters if p.kind is _inspect.Parameter.VAR_KEYWORD), None)

    declaration = [render(p) for p in positional_only]
    if positional_only:
        declaration.append("/")
    declaration.extend(render(p) for p in positional_or_keyword)
    if var_positional is not None:
        declaration.append(f"*{var_positional.name}")
    elif keyword_only:
        declaration.append("*")
    declaration.extend(render(p) for p in keyword_only)
    if var_keyword is not None:
        declaration.append(f"**{var_keyword.name}")

    call = [p.name for p in positional_only]
    call.extend(p.name for p in positional_or_keyword)
    if var_positional is not None:
        call.append(f"*{var_positional.name}")
    call.extend(f"{p.name}={p.name}" for p in keyword_only)
    if var_keyword is not None:
        call.append(f"**{var_keyword.name}")
    return ", ".join(declaration), ", ".join(call)


def _forward(module: _ModuleType, name: str):
    implementation = _IMPLEMENTATIONS[name]
    raw_signature = _inspect.signature(implementation, follow_wrapped=False)
    declaration, call = _wrapper_parameter_source(raw_signature)
    namespace = {
        "_implementation": implementation,
        "_sync_patchable_globals": _sync_patchable_globals,
    }
    source = (
        f"def forwarded({declaration}):\n"
        "    _sync_patchable_globals()\n"
        f"    return _implementation({call})\n"
    )
    exec(compile(source, __file__, "exec"), namespace)
    forwarded = _wraps(implementation)(namespace["forwarded"])
    forwarded.__defaults__ = implementation.__defaults__
    forwarded.__kwdefaults__ = implementation.__kwdefaults__
    forwarded.__module__ = __name__
    forwarded.__qualname__ = name
    return forwarded


for _module in _GROUP_MODULES:
    for _name in _module.__engine_search_constants__:
        globals()[_name] = getattr(_module, _name)

# Preserve the historical logger name rather than the private support-module name.
_log = logging.getLogger(__name__)
_runtime._log = _log

for _module in _GROUP_MODULES:
    for _name in _module.__engine_search_definitions__:
        _definition = getattr(_module, _name)
        if _inspect.isclass(_definition):
            _definition.__module__ = __name__
            for _member in _definition.__dict__.values():
                if _inspect.isfunction(_member):
                    _member.__module__ = __name__
            globals()[_name] = _definition
        else:
            _wrapped_definition = _definition
            while _inspect.isfunction(_wrapped_definition):
                _wrapped_definition.__module__ = __name__
                _wrapped_definition.__qualname__ = _name
                _wrapped_definition = getattr(_wrapped_definition, "__wrapped__", None)
            globals()[_name] = _forward(_module, _name)

for _module in _GROUP_MODULES:
    for _name in _module.__engine_search_late_bindings__:
        setattr(_module, _name, globals()[_name])

_sync_patchable_globals()

__all__ = ["Explorer", "run_explorer_core"]

del _definition, _member, _module, _name, _wrapped_definition
