# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Stage B: light-weight grammar-based refinement package.

This package has been refactored from a single 4296-line file into logical modules
for improved maintainability and organization.

Package structure:
- models.py: Subtree model classes for analytic gradient computation
- atom_mapping.py: AST traversal and atom-to-leaf mapping utilities
- evaluation.py: Model evaluation and diagnostic functions
- transforms.py: Transform-based rewrite candidates
- leaf_utils.py: Leaf manipulation and weight copying
- subtree_utils.py: Subtree separability analysis
- splits.py: Counterterm and counterfactor splitting algorithms
- feature_utils.py: Feature indexing and simple rewrites
- fitting.py: Candidate fitting utilities
- main.py: Main run_stageB_from_model() orchestrator
"""

# Models
# Re-export candidate builders from parent module for backward compatibility
# These were never in stageB.py but are imported by stageB_helpers.py
from nestynet_sr.sr_search.candidate_builders import (
    _build_additive_poly_split_candidate,
    _build_affine_decomp_candidate,
    _build_log_poly_candidate,
    _build_log_ratpoly_candidate,
    _build_planck_1d_candidate,
    _build_power_exp_1d_candidate,
    _build_power_exp_rat_candidate,
    _build_pure_exp_rat_candidate,
    _build_quadratic_poly_candidate,
    _build_sqrt_poly_candidate,
    _build_sqrt_ratpoly_candidate,
    _build_symexp_denom_1d_candidate,
    _make_exp_poly_rewrite,
    _make_exp_ratpoly_rewrite,
    _make_multid_trig_pair_rewrite,
    _make_multid_trig_rewrite,
    _make_scaling_based_rewrite,
    _make_trig_based_rewrite,
)

# Re-export subtree separability helper
from nestynet_sr.sr_search.subtree_separability_helpers import run_subtree_separability

# Atom mapping
from .atom_mapping import (
    _collect_all_atoms,
    _collect_multivariate_nn_atoms,
    _collect_multivariate_poly_atoms,
    _collect_univariate_nn_atoms,
    _refresh_reuse_from_state,
    _vars_in_subtree,
    build_atom_to_leaf_map,
)

# Engine (classes and utilities for Stage B execution)
from .engine import (
    Candidate,
    StageBContext,
    StageBEngine,
    StageBRule,
    StageBState,
    _compute_nn_metrics,
    atom_content_hash,
)

# Evaluation
from .evaluation import (
    _compute_y_med_mad_from_loader,
    _eval_mse_and_rms,
    _eval_val_mse,
    _phi_pred_error_from_loader,
    _print_val_batch_stats,
    _shuffle_axis_sensitivity,
)

# Feature utils
from .feature_utils import (
    _best_scale_spec_for_axis,
    _is_strong_scaling_spec,
    _make_polylog_1d_rewrite,
    _make_logshifted_1d_rewrite,
    _make_poly_1d_rewrite,
    _scaling_index,
    _trig_index,
)

# Fitting
from .fitting import (
    _clone_reuse,
    _fit_candidate_root,
    _format_monomial_from_exponents,
    _safe_reinit_new_leaves,
    _snap_exponent_to_half_integer,
    summarize_global_power_law,
)

# Leaf utilities
from .leaf_utils import (
    _compute_trapped_factorization,
    _copy_compatible_weights,
    _debug_report_leaf_cores,
    _find_first_submodule,
    _fit_poly_1d_trapped,
    _get_leaf_coefficients,
    _initialise_analytic_leaves_from_reuse,
    _leaf_coeff_param,
    _module_path,
    _poly_like_core,
    _poly_zero_and_set,
)

# Main
from .main import run_stageB_from_model
from .models import (
    _OuterTransformedSubtreeModel,
    _SubtreeModel,
)

# Final pruning of small additive terms and per-parameter pruning
from .pruning import prune_insignificant_parameters, prune_nested_additive_terms, prune_small_additive_terms

# factorized symbolic search rewrite rule (first rule in Stage B pipeline)
from .rule_factorized_search import RuleFactorizedSearchFallback

# Rules (rewrite rule classes)
from .rules import (
    RuleAdditiveLogRatio,
    RuleAdditiveGaugeTransfer,
    RuleAffineDecomposition,
    RuleBarycentricCompound,
    RuleCommonPrefactor,
    RuleCompoundFunctionMacros,
    RuleCounterfactorAddSplitNN,
    RuleOverlapCountertermPeelNN,
    RuleCountertermMulSplitNN,
    RuleCoupledLeafRatio,
    RuleHomogeneityPeel,
    RuleJointProductMonomialClosure,
    RuleLastHardAtomRescue,
    RuleLastHardTrigSquare1D,
    RuleLastHardTrigPower1D,
    RuleLogExpCompound,
    RuleMonomialPeelPriority,
    RuleMultiplicativeHomogeneityTransfer,
    RuleMultiDNN,
    RuleNNLeafSeparability,
    RuleNonsenseUnitsZeroPrune,
    RuleOverlapPrefactorPeelNN,
    RuleOuterTransformSplitNN,
    RulePolySplit,
    RulePowerProduct,
    RulePreconditionerFallbackNN,
    RuleProductHomogeneity,
    RuleR1OperatorCertificate,
    RuleRatioInvariance,
    RuleSubtreeSeparability,
    RuleUniNN,
    RuleUnivariateMulPeel,
    RuleUnivariateOracleInvariants,
)

# Splits
from .splits import (
    _build_affine_split_candidate,
    _build_counterfactor_add_split_candidate,
    _build_counterterm_mul_split_candidate,
    _build_overlap_counterterm_peel_candidates,
    _build_overlap_prefactor_peel_candidates,
    _cross_hessian_rank1_score,
    _detect_affine_variable,
    _enumerate_unique_partitions,
    _eval_poly_design_and_grads,
    _fit_affine_split,
    _fit_counterfactor_polys_two_sided_for_add_split,
    _fit_counterterm_polys_two_sided_for_mul_split,
    _gather_nn_atom_value_grad_hess,
    _prefilter_counterterm_partitions_by_rank1_cross_hessian,
    _ridge_solve,
)

# Subtree utils
from .subtree_utils import (
    _build_subtree_separability_candidate,
    _collect_composite_subtree_separability_subtrees,
    _collect_pure_analytic_subtree_separability_subtrees,
    _collect_subtree_separability_targets,
    _infer_nn_hyperparams_from_root,
    _probe_genadd_for_nn_leaf,
    _probe_trapped_for_nn_leaf,
    _subtree_leaf_kinds,
)

# Transforms
from .transforms import (
    _build_nn_leaf_local_outer_transform_candidates,
    _build_subtree_separability_outer_transform_candidates,
    _domain_ok_frac,
    _domain_ok_frac_for_transform,
    _groups_to_global,
    _sample_subtree_values,
    _sample_u_values,
)

__all__ = [
    # Models
    "_SubtreeModel",
    "_OuterTransformedSubtreeModel",
    # Atom mapping
    "build_atom_to_leaf_map",
    "_collect_univariate_nn_atoms",
    "_collect_multivariate_nn_atoms",
    "_collect_multivariate_poly_atoms",
    "_collect_all_atoms",
    "_vars_in_subtree",
    "_refresh_reuse_from_state",
    # Evaluation
    "_compute_y_med_mad_from_loader",
    "_eval_val_mse",
    "_shuffle_axis_sensitivity",
    "_phi_pred_error_from_loader",
    "_print_val_batch_stats",
    "_eval_mse_and_rms",
    # Transforms
    "_sample_u_values",
    "_sample_subtree_values",
    "_domain_ok_frac",
    "_domain_ok_frac_for_transform",
    "_groups_to_global",
    "_build_subtree_separability_outer_transform_candidates",
    "_build_nn_leaf_local_outer_transform_candidates",
    # Leaf utilities
    "_fit_poly_1d_trapped",
    "_find_first_submodule",
    "_poly_like_core",
    "_module_path",
    "_debug_report_leaf_cores",
    "_get_leaf_coefficients",
    "_leaf_coeff_param",
    "_copy_compatible_weights",
    "_initialise_analytic_leaves_from_reuse",
    "_poly_zero_and_set",
    "_compute_trapped_factorization",
    # Subtree utils
    "_probe_genadd_for_nn_leaf",
    "_probe_trapped_for_nn_leaf",
    "_collect_subtree_separability_targets",
    "_infer_nn_hyperparams_from_root",
    "_build_subtree_separability_candidate",
    "_subtree_leaf_kinds",
    "_collect_pure_analytic_subtree_separability_subtrees",
    "_collect_composite_subtree_separability_subtrees",
    # Splits
    "_enumerate_unique_partitions",
    "_cross_hessian_rank1_score",
    "_prefilter_counterterm_partitions_by_rank1_cross_hessian",
    "_gather_nn_atom_value_grad_hess",
    "_detect_affine_variable",
    "_fit_affine_split",
    "_build_affine_split_candidate",
    "_eval_poly_design_and_grads",
    "_ridge_solve",
    "_fit_counterterm_polys_two_sided_for_mul_split",
    "_fit_counterfactor_polys_two_sided_for_add_split",
    "_build_counterfactor_add_split_candidate",
    "_build_counterterm_mul_split_candidate",
    "_build_overlap_counterterm_peel_candidates",
    "_build_overlap_prefactor_peel_candidates",
    # Feature utils
    "_scaling_index",
    "_trig_index",
    "_best_scale_spec_for_axis",
    "_is_strong_scaling_spec",
    "_make_poly_1d_rewrite",
    "_make_polylog_1d_rewrite",
    "_make_logshifted_1d_rewrite",
    "_snap_exponent_to_half_integer",
    "_format_monomial_from_exponents",
    # Fitting
    "_safe_reinit_new_leaves",
    "_clone_reuse",
    "_fit_candidate_root",
    "summarize_global_power_law",
    # Main
    "run_stageB_from_model",
    # Engine
    "StageBState",
    "Candidate",
    "StageBContext",
    "StageBEngine",
    "StageBRule",
    "atom_content_hash",
    "_compute_nn_metrics",
    # Rules
    "RuleAdditiveLogRatio",
    "RuleAdditiveGaugeTransfer",
    "RuleAffineDecomposition",
    "RuleBarycentricCompound",
    "RuleCommonPrefactor",
    "RuleCompoundFunctionMacros",
    "RuleCoupledLeafRatio",
    "RuleHomogeneityPeel",
    "RuleJointProductMonomialClosure",
    "RuleProductHomogeneity",
    "RuleLastHardAtomRescue",
    "RuleLastHardTrigSquare1D",
    "RuleLastHardTrigPower1D",
    "RuleLogExpCompound",
    "RuleMonomialPeelPriority",
    "RuleMultiplicativeHomogeneityTransfer",
    "RuleMultiDNN",
    "RuleNonsenseUnitsZeroPrune",
    "RuleOverlapCountertermPeelNN",
    "RuleOverlapPrefactorPeelNN",
    "RulePolySplit",
    "RuleRatioInvariance",
    "RuleSubtreeSeparability",
    "RuleUniNN",
    "RuleCounterfactorAddSplitNN",
    "RuleCountertermMulSplitNN",
    "RuleOuterTransformSplitNN",
    "RulePowerProduct",
    "RulePreconditionerFallbackNN",
    "RuleR1OperatorCertificate",
    "RuleNNLeafSeparability",
    "RuleUnivariateMulPeel",
    "RuleUnivariateOracleInvariants",
    "RuleFactorizedSearchFallback",
    # Pruning
    "prune_insignificant_parameters",
    "prune_nested_additive_terms",
    "prune_small_additive_terms",
    # Re-exported candidate builders (from nestynet_sr.sr_search.candidate_builders)
    "_build_affine_decomp_candidate",
    "_build_quadratic_poly_candidate",
    "_build_sqrt_ratpoly_candidate",
    "_build_sqrt_poly_candidate",
    "_build_additive_poly_split_candidate",
    "_build_power_exp_rat_candidate",
    "_build_pure_exp_rat_candidate",
    "_build_log_ratpoly_candidate",
    "_build_log_poly_candidate",
    "_build_planck_1d_candidate",
    "_build_symexp_denom_1d_candidate",
    "_build_power_exp_1d_candidate",
    "_make_scaling_based_rewrite",
    "_make_exp_poly_rewrite",
    "_make_exp_ratpoly_rewrite",
    "_make_trig_based_rewrite",
    "_make_multid_trig_rewrite",
    "_make_multid_trig_pair_rewrite",
    # Re-exported separability helper (from nestynet_sr.sr_search.subtree_separability_helpers)
    "run_subtree_separability",
]
