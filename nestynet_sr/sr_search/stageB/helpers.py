# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Stage B Helpers: Pure helper functions for Stage B refinement.

This module contains helper functions used by Stage B rules. These are pure
functions with no side effects or entry-point code.

Currently, this module re-exports functions from stageB.py to avoid circular
imports when stageB_rules.py needs to import helpers. In the future, these
functions can be moved directly into this file for complete separation.
"""

from __future__ import annotations

# Re-export all helper functions from stageB.py
# This avoids circular imports: stageB_rules -> stageB_helpers (not stageB)
#
# Future improvement: Move these function implementations from stageB.py
# into this file directly for complete separation.
# Import replace_atom_in_ast from nestynet_sr.sr_core.bridges
from nestynet_sr.sr_core.bridges import replace_atom_in_ast  # noqa: F401

# Import from parent sr_search module (candidate builders)
# All imports here are re-exports consumed by rules.py via ``from .helpers import …``
from nestynet_sr.sr_search.candidate_builders import (  # noqa: F401
    _build_additive_poly_split_candidate,
    _build_affine_decomp_candidate,
    _build_coupled_ratio_candidate,
    _build_expm1_1d_candidate,
    _build_homogeneity_peel_candidate,
    _build_inv_poly_candidate,
    _build_inv_poly_candidates,
    _build_log_poly_candidate,
    _build_log_ratpoly_candidate,
    _build_last_hard_ratio_candidates,
    _build_nonlinear_sub_candidate,
    _build_planck_1d_candidate,
    _build_planck_1d_candidates,
    _build_planck_derived_feature_candidate,
    _build_planck_full_1d_candidate,
    _build_power_1d_candidate,
    _build_power_exp_1d_candidate,
    _build_power_exp_rat_candidate,
    _build_product_homogeneity_candidate,
    _build_pure_exp_rat_candidate,
    _build_quadratic_poly_candidate,
    _build_ratio_invariance_candidate,
    _build_ratpoly_1d_candidate,
    _build_ratpoly_1d_candidates,
    _build_ratpoly_candidate,
    _build_ratpoly_candidates,
    _build_sqrt_poly_candidate,
    _build_sqrt_ratpoly_1d_candidates,
    _build_sqrt_ratpoly_candidate,
    _build_symexp_denom_1d_candidate,
    _estimate_trig_params_on_compound,
    _estimate_univariate_trig_amplitude,
    _make_affine_trig_rewrite,
    _make_exp_poly_rewrite,
    _make_exp_ratpoly_rewrite,
    _make_multid_trig_pair_rewrite,
    _make_multid_trig_rewrite,
    _make_scaling_based_rewrite,
    _make_tanh_based_rewrite,
    _make_trig_based_rewrite,
)

# Import from parent sr_search module (separability helper)
from nestynet_sr.sr_search.subtree_separability_helpers import (
    run_subtree_separability,  # noqa: F401
)

# Import from stageB package modules
from .atom_mapping import (  # noqa: F401
    _collect_all_atoms,
    _collect_multivariate_nn_atoms,
    _collect_multivariate_poly_atoms,
    _collect_univariate_nn_atoms,
    _find_nns_in_add_chain,
    _find_nns_in_mul_chain,
    build_atom_to_leaf_map,
)
from .feature_utils import (  # noqa: F401
    _best_scale_spec_for_axis,
    _is_strong_scaling_spec,
    _make_polylog_1d_rewrite,
    _make_logshifted_1d_rewrite,
    _make_poly_1d_rewrite,
    _make_power_1d_rewrite,
)
from .leaf_utils import (  # noqa: F401
    _compute_trapped_factorization,
    _copy_compatible_weights,
    _fit_poly_1d_trapped,
    _get_leaf_coefficients,
    _initialise_analytic_leaves_from_reuse,
    _leaf_coeff_param,
    _poly_like_core,
    _poly_zero_and_set,
    _set_constant_leaf_value,
)
from .models import _SubtreeModel  # noqa: F401
from .splits import (  # noqa: F401
    _build_affine_split_candidate,
    _build_counterfactor_add_split_candidate,
    _build_counterterm_mul_split_candidate,
    _build_overlap_counterterm_peel_candidates,
    _build_overlap_prefactor_peel_candidates,
)
from .subtree_utils import (  # noqa: F401
    _build_gauge_split_candidates,
    _build_subtree_separability_candidate,
    _collect_subtree_separability_targets,
    _probe_genadd_for_nn_leaf,
    _probe_trapped_for_nn_leaf,
)
from .transforms import (  # noqa: F401
    _build_nn_leaf_local_outer_transform_candidates,
    _build_subtree_separability_outer_transform_candidates,
)

__all__ = [
    # Collection functions
    "_collect_multivariate_nn_atoms",
    "_collect_multivariate_poly_atoms",
    "_collect_univariate_nn_atoms",
    "_collect_subtree_separability_targets",
    "_collect_all_atoms",
    # Mapping and model utilities
    "build_atom_to_leaf_map",
    "_SubtreeModel",
    "_find_nns_in_add_chain",
    "_find_nns_in_mul_chain",
    # AST manipulation (from nestynet_sr.sr_core.bridges)
    "replace_atom_in_ast",
    # Probe functions
    "_probe_genadd_for_nn_leaf",
    "_probe_trapped_for_nn_leaf",
    "_compute_trapped_factorization",
    # Polynomial and leaf utilities
    "_fit_poly_1d_trapped",
    "_poly_zero_and_set",
    "_get_leaf_coefficients",
    "_leaf_coeff_param",
    "_poly_like_core",
    "_copy_compatible_weights",
    "_initialise_analytic_leaves_from_reuse",
    "_set_constant_leaf_value",
    # Candidate builders
    "_build_affine_decomp_candidate",
    "_build_nonlinear_sub_candidate",
    "_build_quadratic_poly_candidate",
    "_build_sqrt_ratpoly_candidate",
    "_build_sqrt_ratpoly_1d_candidates",
    "_build_inv_poly_candidate",
    "_build_inv_poly_candidates",
    "_build_sqrt_poly_candidate",
    "_build_additive_poly_split_candidate",
    "_build_power_exp_rat_candidate",
    "_build_pure_exp_rat_candidate",
    "_build_log_ratpoly_candidate",
    "_build_log_poly_candidate",
    "_build_last_hard_ratio_candidates",
    "_build_planck_1d_candidate",
    "_build_planck_1d_candidates",
    "_build_planck_derived_feature_candidate",
    "_build_planck_full_1d_candidate",
    "_build_expm1_1d_candidate",
    "_build_symexp_denom_1d_candidate",
    "_build_power_1d_candidate",
    "_build_power_exp_1d_candidate",
    "_build_subtree_separability_candidate",
    "_build_gauge_split_candidates",
    "_build_affine_split_candidate",
    "_build_counterfactor_add_split_candidate",
    "_build_counterterm_mul_split_candidate",
    "_build_overlap_counterterm_peel_candidates",
    "_build_overlap_prefactor_peel_candidates",
    "_build_nn_leaf_local_outer_transform_candidates",
    "_build_subtree_separability_outer_transform_candidates",
    "_build_ratio_invariance_candidate",
    "_build_coupled_ratio_candidate",
    # External helper
    "run_subtree_separability",
    # Feature detection utilities
    "_best_scale_spec_for_axis",
    "_is_strong_scaling_spec",
    # Rewrite constructors
    "_make_scaling_based_rewrite",
    "_make_poly_1d_rewrite",
    "_make_polylog_1d_rewrite",
    "_make_logshifted_1d_rewrite",
    "_make_power_1d_rewrite",
    "_build_ratpoly_1d_candidate",
    "_build_ratpoly_1d_candidates",
    "_build_ratpoly_candidates",
    "_make_exp_poly_rewrite",
    "_make_exp_ratpoly_rewrite",
    "_make_tanh_based_rewrite",
    "_make_trig_based_rewrite",
    "_make_affine_trig_rewrite",
    "_estimate_trig_params_on_compound",
    "_estimate_univariate_trig_amplitude",
    "_make_multid_trig_rewrite",
    "_make_multid_trig_pair_rewrite",
]
