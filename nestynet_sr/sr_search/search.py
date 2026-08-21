# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
# ruff: noqa: F401

"""Compatibility facade for the Stage-A separability search."""

import copy
import itertools
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from nestynet_sr.sr_core import (
    Var,
    ast_to_human_readable,
    build_linear_ast,
    build_mixed_compound_ast,
    build_monomial_ast,
    build_radial_r2_ast,
    check_linear_compound,
    check_mixed_compound,
    check_monomial_compound,
    check_monomial_compound_logderiv,
    check_separability_ops,
    collect_nn_atoms,
    replace_atom_in_ast,
)
from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    Scale,
    SinNode,
    _collect_var_idxs_from_inputs,
    _collect_var_idxs_from_node,
    ast_equals,
    clone_ast,
    compound_input_expr,
    effective_arity,
    eval_input_expr,
    eval_inputs,
    extra_input_var_idxs,
    get_input_exprs,
    has_nontrivial_input,
    is_pure_1d_full_compound_ast as _shared_is_pure_1d_full_compound_ast,
    is_trivial_input,
)
from nestynet_sr.sr_core.fit_links import (
    canonical_fit_link_name,
    describe_fit_link,
    fit_link_torch,
)

from .ast_utils import compact_expression_repr as _compact_expression_repr
from .candidate_builders import _build_atom_input_tensor, _gather_atom_teacher_data
from .coe_witness import (
    CoEWitnessExecutor,
    coe_witness_execution_metadata,
    coe_witness_jobs_from_specs,
    run_threaded_witnesses,
)
from .features import (  # noqa: F401 (TrigProbeTarget, TrigScaleSpec used in docstrings)
    LeafFeatures,
    TrigAxisSpec,
    TrigProbeTarget,
    TrigScaleSpec,
    _compound_to_probe_target,
    discover_compound_features_from_data,
    discover_constant_directions,
    discover_invariance_features,
    discover_leaf_features,
    discover_model_directions,
    discover_parity_axes,
    discover_poly_in_f2,
    discover_poly_in_x,
    discover_preferred_origins,
    discover_radial_groups,
    discover_rational_poly,
    discover_saturating_axes,
    discover_scaling_features,
    discover_trig_axes,
    poisson_profile,
    probe_oracle_scaling,
    probe_trig_scaling,
    sample_line_curvature,
    trig_from_profile,
    verify_compound_null_test,
)
from .monomial_screen import (
    candidate_monomial_exponent,
    candidate_priority_from_screen,
    fit_univariate_monomial_screen,
    half_power_domain_ok,
    monomial_power_label,
    snap_to_half_integer_monomial_power,
)
from .monomial_peel_plan import (
    clean_subset_patterns,
    expand_forced_power_vector,
)
from .r1_operator_certificates import (
    R1OperatorCertificate,
    build_r1_certificate_replacement,
    r1_certificate_poly_init,
    scan_r1_operator_certificates,
)
from .shadow_coordinates import ShadowCoordinate, ShadowRegistry, shadow_parent_key
from .compound_proposals import (
    build_barycentric_compound_proposals,
    build_logexp_compound_proposals,
    build_metric_distance_compound_proposals,
    stageA_tuple_from_proposal,
)
from .model_builders import build_composite_ast, is_minimal_ast

# Shared model-selection helpers (loss floors + simplification budgets).
from .model_selection import (
    apply_noise_floor_to_acceptance_thresholds as _apply_noise_floor_to_acceptance_thresholds,
    compute_accept_threshold as _compute_accept_threshold,
    estimate_transform_noise_floor_raw as _estimate_transform_noise_floor_raw,
    noise_equivalence_tolerance as _noise_equivalence_tolerance,
    loss_excess_above_floor as _loss_excess_above_floor,
    noise_equivalent as _noise_equivalent,
    resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw,
)
from .training import train_candidate_model, train_initial_model
from .wrapper_policy import (
    build_compound_z_variants,
    compound_z_wrapper_policy,
    should_select_compound_variant,
    snap_omega,
)
from .y_transforms import precision_for_transform

import inspect as _inspect
from functools import wraps as _wraps
from types import ModuleType as _ModuleType

from . import _search_compounds as _compounds
from . import _search_detection as _detection
from . import _search_policy as _policy
from . import _search_proposals as _proposals
from . import _search_runtime as _runtime
from . import _search_shadow as _shadow
from . import _search_structure as _structure
from . import _search_training as _training

_GROUP_MODULES = (
    _shadow,
    _training,
    _detection,
    _structure,
    _proposals,
    _compounds,
    _policy,
    _runtime,
)
_IMPLEMENTATIONS = {
    name: getattr(module, name)
    for module in _GROUP_MODULES
    for name in module.__search_definitions__
}

_PATCHABLE_GLOBAL_NAMES = (
    "run_separability_for_transform",
    "_quick_separability_candidates",
    "_stageA_split_simplicity_score",
    "_detect_compound_variable_for_atom",
    "_try_compound_candidates_for_atom",
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

    positional_only = [
        parameter
        for parameter in parameters
        if parameter.kind is _inspect.Parameter.POSITIONAL_ONLY
    ]
    positional_or_keyword = [
        parameter
        for parameter in parameters
        if parameter.kind is _inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    var_positional = next(
        (
            parameter
            for parameter in parameters
            if parameter.kind is _inspect.Parameter.VAR_POSITIONAL
        ),
        None,
    )
    keyword_only = [
        parameter
        for parameter in parameters
        if parameter.kind is _inspect.Parameter.KEYWORD_ONLY
    ]
    var_keyword = next(
        (
            parameter
            for parameter in parameters
            if parameter.kind is _inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )

    declaration = [render(parameter) for parameter in positional_only]
    if positional_only:
        declaration.append("/")
    declaration.extend(render(parameter) for parameter in positional_or_keyword)
    if var_positional is not None:
        declaration.append(f"*{var_positional.name}")
    elif keyword_only:
        declaration.append("*")
    declaration.extend(render(parameter) for parameter in keyword_only)
    if var_keyword is not None:
        declaration.append(f"**{var_keyword.name}")

    call = [parameter.name for parameter in positional_only]
    call.extend(parameter.name for parameter in positional_or_keyword)
    if var_positional is not None:
        call.append(f"*{var_positional.name}")
    call.extend(f"{parameter.name}={parameter.name}" for parameter in keyword_only)
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
    for _name in _module.__search_constants__:
        globals()[_name] = getattr(_module, _name)

for _module in _GROUP_MODULES:
    for _name in _module.__search_definitions__:
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
    for _name in _module.__search_late_bindings__:
        setattr(_module, _name, globals()[_name])

_sync_patchable_globals()

del _definition, _member, _module, _name, _wrapped_definition
