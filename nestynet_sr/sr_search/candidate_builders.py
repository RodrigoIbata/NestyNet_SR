# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compatibility facade for Stage-B candidate builders.

The implementations are grouped by responsibility in private modules.  This
module retains every historical builder name and forwards the handful of
fitting helpers that tests and downstream experiments patch here.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from nestynet_sr.sr_core.bridges import AtomNode, Node

from . import _candidate_builders_common as _common
from . import _candidate_builders_multivariate as _multivariate
from . import _candidate_builders_structural as _structural
from . import _candidate_builders_univariate as _univariate
from . import fitting_utils as _fitting_utils
from .features import ScaleSpec, TrigAxisSpec

# ``functools.wraps`` preserves the original deferred annotations, which are
# evaluated in this facade's globals by ``typing.get_type_hints``.
_ANNOTATION_GLOBALS = (
    Any,
    AtomNode,
    Callable,
    Dict,
    List,
    Node,
    Optional,
    ScaleSpec,
    TrigAxisSpec,
    Tuple,
    torch,
)

# These names have historically been monkeypatched on this module.  Keep them
# as real facade globals, then copy their current values to the implementation
# modules immediately before each builder call.
_fit_planck_tail_fixed_power = _fitting_utils._fit_planck_tail_fixed_power
_fit_power_coeffs_1d = _fitting_utils._fit_power_coeffs_1d
_fit_rational_coeffs_1d = _fitting_utils._fit_rational_coeffs_1d
_fit_rational_coeffs_nd = _fitting_utils._fit_rational_coeffs_nd
_gather_teacher_data_1d = _fitting_utils._gather_teacher_data_1d
_rational_probe_nd = _fitting_utils._rational_probe_nd

_PATCHABLE_GLOBAL_NAMES = (
    "_fit_planck_tail_fixed_power",
    "_fit_power_coeffs_1d",
    "_fit_rational_coeffs_1d",
    "_fit_rational_coeffs_nd",
    "_gather_atom_teacher_data",
    "_gather_teacher_data_1d",
    "_rational_probe_nd",
)

_COMMON_FUNCTIONS = (
    "_unwrap_leaf_core",
    "_atom_inputs_match",
    "_find_matching_core",
    "_single_power_coordinate_inputs",
    "_support_is_valid",
    "_max_total_degree_from_exps",
    "_exps_override_from_tensor",
    "_exps_key",
    "_select_clear_rratpoly_pivot",
    "_move_sparse_pivot_to_end",
    "_select_sign_region",
    "_parse_pure_difference_expr",
    "_eval_input_expr_value",
    "_build_atom_input_tensor",
    "_gather_atom_teacher_data",
    "_replace_node",
)

_MULTIVARIATE_FUNCTIONS = (
    "_make_power_exp_ratpoly_rewrite",
    "_make_power_exp_poly_rewrite",
    "_build_quadratic_poly_candidate",
    "_build_trig_diff_affine_envelope_candidate",
    "_build_sqrt_ratpoly_candidate",
    "_build_log_ratpoly_candidate",
    "_build_ratpoly_candidates",
    "_build_ratpoly_candidate",
    "_dim_tuple_is_zero",
    "_stable_last_hard_ratio_sig",
    "_build_last_hard_ratio_candidates",
    "_build_nonlinear_sub_candidate",
    "_build_log_poly_candidate",
    "_build_sqrt_poly_candidate",
    "_build_inv_poly_candidates",
    "_build_inv_poly_candidate",
    "_build_poly_split_from_subtree_separability",
    "_build_additive_poly_split_candidate",
    "_build_power_exp_rat_candidate",
    "_build_pure_exp_rat_candidate",
)

_UNIVARIATE_FUNCTIONS = (
    "_build_power_exp_1d_candidate",
    "_planck_power_label",
    "_build_planck_1d_candidates",
    "_build_planck_1d_candidate",
    "_build_planck_full_1d_candidate",
    "_build_expm1_1d_candidate",
    "_build_symexp_denom_1d_candidate",
    "_make_scaling_based_rewrite",
    "_make_trig_based_rewrite",
    "_make_tanh_based_rewrite",
    "_make_affine_trig_rewrite",
    "_make_multid_trig_rewrite",
    "_make_multid_trig_pair_rewrite",
    "_build_trig_affine_envelope_candidate",
    "_make_exp_poly_rewrite",
    "_make_exp_ratpoly_rewrite",
    "_build_ratpoly_1d_candidates",
    "_build_ratpoly_1d_candidate",
    "_build_sqrt_ratpoly_1d_candidates",
    "_build_power_1d_candidate",
)

_STRUCTURAL_FUNCTIONS = (
    "_build_ratio_invariance_candidate",
    "_build_homogeneity_peel_candidate",
    "_build_product_homogeneity_candidate",
    "_build_coupled_ratio_candidate",
    "_estimate_trig_params_on_compound",
    "_estimate_univariate_trig_amplitude",
    "_build_planck_derived_feature_candidate",
    "_build_affine_decomp_candidate",
)

_IMPLEMENTATION_MODULES = (_common, _multivariate, _univariate, _structural)


def _sync_patchable_globals() -> List[Tuple[ModuleType, str, Any]]:
    """Apply facade overrides for one forwarded call and return restoration state."""
    facade_globals = globals()
    previous: List[Tuple[ModuleType, str, Any]] = []
    for module in _IMPLEMENTATION_MODULES:
        for name in _PATCHABLE_GLOBAL_NAMES:
            if hasattr(module, name):
                previous.append((module, name, getattr(module, name)))
                setattr(module, name, facade_globals[name])
    return previous


def _restore_patchable_globals(previous: List[Tuple[ModuleType, str, Any]]) -> None:
    """Undo one scoped facade synchronization, including nested calls safely."""
    for module, name, value in reversed(previous):
        setattr(module, name, value)


def _forward(module: ModuleType, name: str) -> Callable[..., Any]:
    implementation = getattr(module, name)

    @wraps(implementation)
    def forwarded(*args, **kwargs):
        previous = _sync_patchable_globals()
        try:
            return implementation(*args, **kwargs)
        finally:
            _restore_patchable_globals(previous)

    forwarded.__module__ = __name__
    forwarded.__qualname__ = name
    return forwarded


for _module, _names in (
    (_common, _COMMON_FUNCTIONS),
    (_multivariate, _MULTIVARIATE_FUNCTIONS),
    (_univariate, _UNIVARIATE_FUNCTIONS),
    (_structural, _STRUCTURAL_FUNCTIONS),
):
    for _name in _names:
        globals()[_name] = _forward(_module, _name)

del _fitting_utils, _module, _name, _names
