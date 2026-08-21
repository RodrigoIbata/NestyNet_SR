# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage B rule: factorized symbolic search rewrite for low-arity NN atoms.

Uses the factorized symbolic search explorer to propose analytic rewrites for NN atoms
with arity <= max_arity that survived all earlier, cheaper rules.
"""

from __future__ import annotations

from typing import List

from nestynet_sr.sr_core.bridges import AtomNode, Node, collect_nn_atoms, effective_arity, is_problem_atom
from nestynet_sr.sr_search.factorized_search.adapters.nestynet.stageb_prep import prepare_stageb_explorer_inputs
from .engine import Candidate, StageBContext, StageBRule


class RuleFactorizedSearchFallback(StageBRule):
    """Run the factorized symbolic search explorer on surviving NN atoms.

    This is the first rule tried in the Stage B pipeline.  For each target atom the
    explorer is launched on the atom's input/output data (sampled from the
    trained model) and the top-k results are converted to NestyNet_SR ASTs
    and proposed as candidates.

    Pattern label: ``factorized_search``
    """

    name = "factorized_search"

    def __init__(self, factorized_search_hp=None):
        from nestynet_sr.sr_search.config import FactorizedSearchConfig

        if factorized_search_hp is None:
            factorized_search_hp = FactorizedSearchConfig()

        from dataclasses import fields

        from nestynet_sr.sr_expr_ir.config import ExpressionIRConfig

        _expr_ir_base = ExpressionIRConfig()
        for _field in fields(ExpressionIRConfig):
            if _field.name == "expr_ir":
                _key = "expr_ir"
            elif _field.name == "canonicalize":
                _key = "expr_canonicalize"
            elif _field.name == "domain_mode":
                _key = "expr_domain_mode"
            else:
                _key = f"expr_{_field.name}"
            setattr(
                self,
                _key,
                getattr(factorized_search_hp, _key, getattr(_expr_ir_base, _field.name)),
            )

        self.max_arity = factorized_search_hp.max_arity
        self.n_iter = factorized_search_hp.n_iter
        self.max_depth = factorized_search_hp.max_depth
        self.poly_degree = factorized_search_hp.poly_degree
        self.return_topk = factorized_search_hp.return_topk
        self.n_fit = factorized_search_hp.n_fit
        self.n_probe = factorized_search_hp.n_probe
        self.lo = factorized_search_hp.lo
        self.hi = factorized_search_hp.hi
        self.seed = factorized_search_hp.seed
        self.n_seeds = factorized_search_hp.n_seeds
        self.split_iter_across_seeds = factorized_search_hp.split_iter_across_seeds
        self.brute_depth = factorized_search_hp.brute_depth
        self.early_stop_mse = factorized_search_hp.early_stop_mse
        self.brute_max_expressions = factorized_search_hp.brute_max_expressions
        self.refine_enable = factorized_search_hp.refine_enable
        self.refine_profile = getattr(factorized_search_hp, "refine_profile", "default")
        self.refine_mode = getattr(factorized_search_hp, "refine_mode", "slate")
        self.refine_during_brute = getattr(factorized_search_hp, "refine_during_brute", False)
        self.refine_during_mutation = getattr(factorized_search_hp, "refine_during_mutation", False)
        self.refine_during_controller_slate = getattr(
            factorized_search_hp,
            "refine_during_controller_slate",
            False,
        )
        self.refine_during_slate = getattr(factorized_search_hp, "refine_during_slate", True)
        self.refine_slate_after_brute = getattr(factorized_search_hp, "refine_slate_after_brute", True)
        self.refine_slate_period = getattr(factorized_search_hp, "refine_slate_period", 0)
        self.refine_final_polish = getattr(factorized_search_hp, "refine_final_polish", True)
        self.refine_slate_k = getattr(factorized_search_hp, "refine_slate_k", 16)
        self.refine_slate_diverse_k = getattr(factorized_search_hp, "refine_slate_diverse_k", 8)
        self.refine_slate_budget = getattr(factorized_search_hp, "refine_slate_budget", 32)
        self.refine_optimizer = getattr(factorized_search_hp, "refine_optimizer", "lbfgs")
        self.refine_lbfgs_escalate_improve_factor = getattr(
            factorized_search_hp,
            "refine_lbfgs_escalate_improve_factor",
            2.0,
        )
        self.refine_lbfgs_steps = factorized_search_hp.refine_lbfgs_steps
        self.refine_fit_subset = factorized_search_hp.refine_fit_subset
        self.refine_fit_subset_mode = factorized_search_hp.refine_fit_subset_mode
        self.refine_num_restarts = factorized_search_hp.refine_num_restarts
        self.refine_max_variants = factorized_search_hp.refine_max_variants
        self.refine_max_params = factorized_search_hp.refine_max_params
        self.refine_slot_sensitivity_enable = factorized_search_hp.refine_slot_sensitivity_enable
        self.refine_slot_sensitivity_subset = factorized_search_hp.refine_slot_sensitivity_subset
        self.refine_slot_sensitivity_delta = factorized_search_hp.refine_slot_sensitivity_delta
        self.refine_slot_sensitivity_max_paths = factorized_search_hp.refine_slot_sensitivity_max_paths
        self.refine_prune_mapping_equiv_root_slots = getattr(
            factorized_search_hp,
            "refine_prune_mapping_equiv_root_slots",
            True,
        )
        self.refine_attempt_cache_enable = getattr(
            factorized_search_hp,
            "refine_attempt_cache_enable",
            True,
        )
        self.refine_attempt_cache_max_entries = getattr(
            factorized_search_hp,
            "refine_attempt_cache_max_entries",
            4096,
        )
        self.refine_linear_combo_enable = factorized_search_hp.refine_linear_combo_enable
        self.refine_linear_terms_max = factorized_search_hp.refine_linear_terms_max
        self.refine_linear_prune_rel = factorized_search_hp.refine_linear_prune_rel
        self.refine_linear_ridge = factorized_search_hp.refine_linear_ridge
        self.refine_gate_best_factor = factorized_search_hp.refine_gate_best_factor
        self.refine_gate_potential_enable = factorized_search_hp.refine_gate_potential_enable
        self.refine_gate_potential_subset = factorized_search_hp.refine_gate_potential_subset
        self.refine_gate_potential_improve_factor = factorized_search_hp.refine_gate_potential_improve_factor
        self.refine_gate_log_min = factorized_search_hp.refine_gate_log_min
        self.refine_gate_log_max = factorized_search_hp.refine_gate_log_max
        self.refine_gate_grid_size = factorized_search_hp.refine_gate_grid_size
        self.refine_gate_max_evals = factorized_search_hp.refine_gate_max_evals
        self.refine_max_trials = factorized_search_hp.refine_max_trials
        self.refine_trials_per_brute_depth = factorized_search_hp.refine_trials_per_brute_depth
        self.refine_trials_per_mutation_window = factorized_search_hp.refine_trials_per_mutation_window
        self.refine_mutation_window = factorized_search_hp.refine_mutation_window
        self.refine_safe_eps = factorized_search_hp.refine_safe_eps
        self.refine_safe_penalty_weight = factorized_search_hp.refine_safe_penalty_weight
        self.refine_safe_exp_clip = factorized_search_hp.refine_safe_exp_clip
        self.refine_theta_l2 = factorized_search_hp.refine_theta_l2
        self.refine_init_log_min = factorized_search_hp.refine_init_log_min
        self.refine_init_log_max = factorized_search_hp.refine_init_log_max
        self.refine_grid_enable = factorized_search_hp.refine_grid_enable
        self.refine_grid_size = factorized_search_hp.refine_grid_size
        self.refine_grid_size_2d = factorized_search_hp.refine_grid_size_2d
        self.refine_grid_passes = factorized_search_hp.refine_grid_passes
        self.refine_grid_topk = factorized_search_hp.refine_grid_topk
        self.refine_grid_max_evals = factorized_search_hp.refine_grid_max_evals
        self.refine_stall_gate_relax_factor = factorized_search_hp.refine_stall_gate_relax_factor
        self.refine_stall_gate_relax_max = factorized_search_hp.refine_stall_gate_relax_max

        # Joint multi-dataset factorized symbolic search / continuous skeleton refinement:
        # when running in Class-SR / multi-dataset mode, we can
        # optionally score (and refine) using shared structure but per-dataset affine maps.
        self.refine_joint_enable = bool(getattr(factorized_search_hp, "refine_joint_enable", True))
        self.refine_joint_weight_mode = str(getattr(factorized_search_hp, "refine_joint_weight_mode", "points"))
        self.refine_joint_score_enable = bool(getattr(factorized_search_hp, "refine_joint_score_enable", True))
        self.refine_joint_terms_enable = bool(getattr(factorized_search_hp, "refine_joint_terms_enable", False))
        self.refine_stageb_promote_consts = factorized_search_hp.refine_stageb_promote_consts
        # Context-sensitive inverse steering.
        self.inverse_steering_enable = bool(getattr(factorized_search_hp, "inverse_steering_enable", False))
        self.inverse_max_paths = int(getattr(factorized_search_hp, "inverse_max_paths", 12))
        self.inverse_topk_terms = int(getattr(factorized_search_hp, "inverse_topk_terms", 6))
        self.inverse_shortlist_mult = int(getattr(factorized_search_hp, "inverse_shortlist_mult", 4))
        self.inverse_min_valid_frac = float(getattr(factorized_search_hp, "inverse_min_valid_frac", 0.25))
        self.inverse_min_confidence = float(getattr(factorized_search_hp, "inverse_min_confidence", 0.10))
        self.inverse_safe_eps = getattr(factorized_search_hp, "inverse_safe_eps", None)
        self.inverse_confidence_mode = str(getattr(factorized_search_hp, "inverse_confidence_mode", "conditioning"))
        self.inverse_confidence_target_gain = float(getattr(factorized_search_hp, "inverse_confidence_target_gain", 4.0))
        self.inverse_confidence_floor = float(getattr(factorized_search_hp, "inverse_confidence_floor", 0.05))
        self.inverse_branch_beam_width = int(getattr(factorized_search_hp, "inverse_branch_beam_width", 1))
        self.inverse_micro_search_enable = bool(getattr(factorized_search_hp, "inverse_micro_search_enable", False))
        self.inverse_micro_search_max_depth = int(getattr(factorized_search_hp, "inverse_micro_search_max_depth", 3))
        self.inverse_micro_search_beam_width = int(getattr(factorized_search_hp, "inverse_micro_search_beam_width", 24))
        self.inverse_micro_search_topk = int(getattr(factorized_search_hp, "inverse_micro_search_topk", 16))
        self.inverse_micro_search_seed_terms = int(getattr(factorized_search_hp, "inverse_micro_search_seed_terms", 8))
        self.inverse_local_score_mode = str(getattr(factorized_search_hp, "inverse_local_score_mode", "affine"))
        self.inverse_spec_enable = bool(getattr(factorized_search_hp, "inverse_spec_enable", False))
        self.inverse_spec_enum_max_depth = int(getattr(factorized_search_hp, "inverse_spec_enum_max_depth", 4))
        self.inverse_spec_enum_max_trees = int(getattr(factorized_search_hp, "inverse_spec_enum_max_trees", 5000))
        self.inverse_spec_preview_topk = int(getattr(factorized_search_hp, "inverse_spec_preview_topk", 16))
        self.inverse_spec_local_score_mode = str(getattr(factorized_search_hp, "inverse_spec_local_score_mode", "affine"))
        self.inverse_spec_include_legacy_seed = bool(getattr(factorized_search_hp, "inverse_spec_include_legacy_seed", True))
        self.inverse_spec_complexity_penalty = float(getattr(factorized_search_hp, "inverse_spec_complexity_penalty", 0.0))
        self.inverse_spec_recursive_enable = bool(getattr(factorized_search_hp, "inverse_spec_recursive_enable", True))
        self.inverse_spec_recursive_max_depth = int(getattr(factorized_search_hp, "inverse_spec_recursive_max_depth", 2))
        self.inverse_spec_recursive_trigger_rel_mse = float(getattr(factorized_search_hp, "inverse_spec_recursive_trigger_rel_mse", 0.25))
        self.inverse_spec_recursive_seed_cap = int(getattr(factorized_search_hp, "inverse_spec_recursive_seed_cap", 6))
        self.inverse_spec_recursive_branch_topk = int(getattr(factorized_search_hp, "inverse_spec_recursive_branch_topk", 4))
        self.inverse_spec_recursive_child_topk = int(getattr(factorized_search_hp, "inverse_spec_recursive_child_topk", 2))
        self.inverse_target_mode = str(getattr(factorized_search_hp, "inverse_target_mode", "robust"))
        self.inverse_full_mapping_penalty = float(getattr(factorized_search_hp, "inverse_full_mapping_penalty", 0.75))
        self.inverse_exact_simple_target_bonus = float(getattr(factorized_search_hp, "inverse_exact_simple_target_bonus", 0.10))
        self.inverse_additive_descend_penalty = float(getattr(factorized_search_hp, "inverse_additive_descend_penalty", 0.15))
        self.inverse_nonadditive_leaf_penalty = float(getattr(factorized_search_hp, "inverse_nonadditive_leaf_penalty", 0.20))
        self.inverse_exact_path_eta = float(getattr(factorized_search_hp, "inverse_exact_path_eta", 0.98))
        self.inverse_exact_transport_min_lin_rel = float(getattr(factorized_search_hp, "inverse_exact_transport_min_lin_rel", 0.0))
        self.inverse_gate_enable = bool(getattr(factorized_search_hp, "inverse_gate_enable", True))
        self.inverse_gate_warmup = int(getattr(factorized_search_hp, "inverse_gate_warmup", 0))
        self.inverse_gate_best_factor = float(getattr(factorized_search_hp, "inverse_gate_best_factor", 20.0))
        self.inverse_gate_min_residual_basins = int(getattr(factorized_search_hp, "inverse_gate_min_residual_basins", 0))
        self.inverse_gate_min_depth = int(getattr(factorized_search_hp, "inverse_gate_min_depth", 4))
        self.inverse_gate_min_size = int(getattr(factorized_search_hp, "inverse_gate_min_size", 6))
        self.inverse_gate_max_paths = int(getattr(factorized_search_hp, "inverse_gate_max_paths", 6))
        self.inverse_gate_min_structural_score = float(getattr(factorized_search_hp, "inverse_gate_min_structural_score", 0.75))
        self.inverse_gate_min_weighted_rel_gain = float(getattr(factorized_search_hp, "inverse_gate_min_weighted_rel_gain", 0.05))
        self.inverse_gate_structural_bias = float(getattr(factorized_search_hp, "inverse_gate_structural_bias", 0.20))
        self.inverse_periodic_min_valid_scale = float(getattr(factorized_search_hp, "inverse_periodic_min_valid_scale", 1.25))
        self.inverse_periodic_min_confidence_scale = float(getattr(factorized_search_hp, "inverse_periodic_min_confidence_scale", 1.35))
        self.inverse_periodic_path_penalty = float(getattr(factorized_search_hp, "inverse_periodic_path_penalty", 0.65))
        self.inverse_nonperiodic_muldiv_bonus = float(getattr(factorized_search_hp, "inverse_nonperiodic_muldiv_bonus", 0.10))
        self.inverse_nonperiodic_explogsqrt_bonus = float(getattr(factorized_search_hp, "inverse_nonperiodic_explogsqrt_bonus", 0.05))
        self.inverse_branch_ambiguity_penalty = float(getattr(factorized_search_hp, "inverse_branch_ambiguity_penalty", 0.50))
        self.inverse_transport_min_lin_rel = float(getattr(factorized_search_hp, "inverse_transport_min_lin_rel", 0.02))
        self.inverse_transport_min_effective_n = float(getattr(factorized_search_hp, "inverse_transport_min_effective_n", 8.0))
        self.repair_controller_enable = bool(getattr(factorized_search_hp, "repair_controller_enable", False))
        self.repair_controller_min_score = float(getattr(factorized_search_hp, "repair_controller_min_score", 0.15))
        self.repair_controller_steps = int(getattr(factorized_search_hp, "repair_controller_steps", 3))
        self.repair_controller_ancestor_hops = int(getattr(factorized_search_hp, "repair_controller_ancestor_hops", 1))
        self.repair_controller_min_step_rel_improve = float(getattr(factorized_search_hp, "repair_controller_min_step_rel_improve", 1.0e-3))
        self.repair_controller_adaptive = bool(getattr(factorized_search_hp, "repair_controller_adaptive", True))
        self.repair_controller_adapt_quantile = float(getattr(factorized_search_hp, "repair_controller_adapt_quantile", 0.75))
        self.repair_controller_adapt_window = int(getattr(factorized_search_hp, "repair_controller_adapt_window", 128))
        self.repair_controller_adapt_min_samples = int(getattr(factorized_search_hp, "repair_controller_adapt_min_samples", 16))
        self.repair_controller_min_concentration = float(getattr(factorized_search_hp, "repair_controller_min_concentration", 0.30))
        self.repair_controller_potential_weight = float(getattr(factorized_search_hp, "repair_controller_potential_weight", 1.00))
        self.repair_controller_concentration_weight = float(getattr(factorized_search_hp, "repair_controller_concentration_weight", 0.35))
        self.repair_controller_contrast_weight = float(getattr(factorized_search_hp, "repair_controller_contrast_weight", 0.20))
        self.repair_controller_cost_weight = float(getattr(factorized_search_hp, "repair_controller_cost_weight", 0.10))
        self.repair_controller_stagnation_weight = float(getattr(factorized_search_hp, "repair_controller_stagnation_weight", 0.15))
        self.repair_controller_frontier_topk = int(getattr(factorized_search_hp, "repair_controller_frontier_topk", 24))
        self.repair_controller_stagnation_visits = int(getattr(factorized_search_hp, "repair_controller_stagnation_visits", 8))
        self.repair_controller_focus_prob = float(getattr(factorized_search_hp, "repair_controller_focus_prob", 0.50))
        self.repair_controller_parent_max_repeats = int(getattr(factorized_search_hp, "repair_controller_parent_max_repeats", 2))
        self.repair_controller_parent_min_eval_gap = int(getattr(factorized_search_hp, "repair_controller_parent_min_eval_gap", 32))
        self.repair_controller_parent_reset_rel_improve = float(getattr(factorized_search_hp, "repair_controller_parent_reset_rel_improve", 0.05))
        self.repair_controller_critic_enable = bool(getattr(factorized_search_hp, "repair_controller_critic_enable", False))
        self.repair_controller_critic_path = str(getattr(factorized_search_hp, "repair_controller_critic_path", ""))
        self.repair_controller_critic_blend = float(getattr(factorized_search_hp, "repair_controller_critic_blend", 1.0))
        self.repair_controller_critic_mode = str(getattr(factorized_search_hp, "repair_controller_critic_mode", "priority"))
        self.repair_opportunity_controller_enable = bool(getattr(factorized_search_hp, "repair_opportunity_controller_enable", False))
        self.repair_opportunity_controller_path = str(getattr(factorized_search_hp, "repair_opportunity_controller_path", ""))

        # Residual-guided continuous search (greedy boosting / OMP)
        self.boost_enable = bool(getattr(factorized_search_hp, "boost_enable", False))
        self.boost_max_terms = int(getattr(factorized_search_hp, "boost_max_terms", 6))
        self.boost_topk_try = int(getattr(factorized_search_hp, "boost_topk_try", 15))
        self.boost_min_rel_improve = float(getattr(factorized_search_hp, "boost_min_rel_improve", 1.0e-3))
        self.boost_selection_split = str(getattr(factorized_search_hp, "boost_selection_split", "fit"))
        self.boost_ridge = getattr(factorized_search_hp, "boost_ridge", None)
        self.boost_include_parent = bool(getattr(factorized_search_hp, "boost_include_parent", True))
        self.boost_from_scratch_prob = float(getattr(factorized_search_hp, "boost_from_scratch_prob", 0.25))
        self.boost_prune_rel = float(getattr(factorized_search_hp, "boost_prune_rel", 1.0e-10))
        self.boost_safe_eval = bool(getattr(factorized_search_hp, "boost_safe_eval", True))
        self.boost_harvest_enable = bool(getattr(factorized_search_hp, "boost_harvest_enable", False))
        self.boost_harvest_every = int(getattr(factorized_search_hp, "boost_harvest_every", 500))
        self.boost_harvest_topk_residual_basins = int(getattr(factorized_search_hp, "boost_harvest_topk_residual_basins", 50))
        self.boost_harvest_elites_per_residual_basin = int(getattr(factorized_search_hp, "boost_harvest_elites_per_residual_basin", 2))
        self.boost_pool_extra_max = int(getattr(factorized_search_hp, "boost_pool_extra_max", 256))
        self.boost_subtree_depth_max = int(getattr(factorized_search_hp, "boost_subtree_depth_max", 3))
        self.boost_subtree_size_max = int(getattr(factorized_search_hp, "boost_subtree_size_max", 12))
        self.boost_gate_enable = bool(getattr(factorized_search_hp, "boost_gate_enable", True))
        self.boost_gate_warmup = int(getattr(factorized_search_hp, "boost_gate_warmup", 200))
        self.boost_gate_best_factor = float(getattr(factorized_search_hp, "boost_gate_best_factor", 30.0))
        self.boost_gate_gain_frac = float(getattr(factorized_search_hp, "boost_gate_gain_frac", 1.0e-2))
        self.boost_gate_peak_ratio = float(getattr(factorized_search_hp, "boost_gate_peak_ratio", 5.0))
        self.boost_gate_min_valid = int(getattr(factorized_search_hp, "boost_gate_min_valid", 8))
        self.boost_gate_min_residual_basins = int(getattr(factorized_search_hp, "boost_gate_min_residual_basins", 10))
        self.boost_gate_adaptive = bool(getattr(factorized_search_hp, "boost_gate_adaptive", True))
        self.boost_gate_adapt_quantile = float(getattr(factorized_search_hp, "boost_gate_adapt_quantile", 0.75))
        self.boost_gate_adapt_window = int(getattr(factorized_search_hp, "boost_gate_adapt_window", 256))
        self.boost_gate_adapt_min_samples = int(getattr(factorized_search_hp, "boost_gate_adapt_min_samples", 32))
        self.boost_gate_adapt_mix = float(getattr(factorized_search_hp, "boost_gate_adapt_mix", 1.0))
        self.boost_gate_gain_frac_floor = float(getattr(factorized_search_hp, "boost_gate_gain_frac_floor", 1.0e-4))
        self.boost_gate_gain_frac_cap = float(getattr(factorized_search_hp, "boost_gate_gain_frac_cap", 0.25))

        # Scoring augmentation: multi-term linear head on residual (cheap)
        self.score_head_enable = bool(getattr(factorized_search_hp, "score_head_enable", True))
        self.score_head_vars_enable = bool(getattr(factorized_search_hp, "score_head_vars_enable", True))
        self.score_head_omp_enable = bool(getattr(factorized_search_hp, "score_head_omp_enable", False))
        self.score_head_omp_max_terms = int(getattr(factorized_search_hp, "score_head_omp_max_terms", 2))
        self.score_head_omp_topk_try = int(getattr(factorized_search_hp, "score_head_omp_topk_try", 15))
        self.score_head_ridge = getattr(factorized_search_hp, "score_head_ridge", None)
        self.score_head_min_rel_improve = float(getattr(factorized_search_hp, "score_head_min_rel_improve", 0.0))

        # Optional outer-wrapper pass (gated, reduced-budget).
        self.outer_wrapper_enable = bool(getattr(factorized_search_hp, "outer_wrapper_enable", False))
        self.outer_wrapper_max_arity = int(getattr(factorized_search_hp, "outer_wrapper_max_arity", 2))
        self.outer_wrapper_transforms = list(
            getattr(
                factorized_search_hp,
                "outer_wrapper_transforms",
                ["log", "reciprocal", "sqrt", "square", "exp"],
            )
        )
        self.outer_wrapper_topk = int(getattr(factorized_search_hp, "outer_wrapper_topk", 2))
        self.outer_wrapper_min_domain_frac = float(
            getattr(factorized_search_hp, "outer_wrapper_min_domain_frac", 0.995)
        )
        self.outer_wrapper_min_points = int(
            getattr(factorized_search_hp, "outer_wrapper_min_points", 256)
        )
        self.outer_wrapper_probe_max_points = int(
            getattr(factorized_search_hp, "outer_wrapper_probe_max_points", 4096)
        )
        self.outer_wrapper_iter_scale = float(
            getattr(factorized_search_hp, "outer_wrapper_iter_scale", 0.20)
        )
        self.outer_wrapper_n_seeds = int(
            getattr(factorized_search_hp, "outer_wrapper_n_seeds", 1)
        )
        self.outer_wrapper_return_topk = int(
            getattr(factorized_search_hp, "outer_wrapper_return_topk", 2)
        )
        self.outer_wrapper_screen_rational_err_max = float(
            getattr(factorized_search_hp, "outer_wrapper_screen_rational_err_max", 0.02)
        )
        self.outer_wrapper_screen_nls_err_max = float(
            getattr(factorized_search_hp, "outer_wrapper_screen_nls_err_max", 0.02)
        )


    # ------------------------------------------------------------------
    def iter_targets(self, ctx: StageBContext):
        """Yield NN atoms with effective arity in [2, max_arity].

        Arity-1 atoms are handled inside RuleUniNN (after cheaper polynomial
        rewrites), so they are excluded here to avoid wasting brute+mutate
        cycles on depth-1 matches that would be tried later anyway.
        """
        for atom in collect_nn_atoms(ctx.state.root):
            if is_problem_atom(atom):
                continue
            ea = effective_arity(atom)
            if 2 <= ea <= self.max_arity:
                yield atom

    # ------------------------------------------------------------------
    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
            return []

        arity = effective_arity(target)
        if arity > self.max_arity:
            return []

        # Import from local factorized_search subpackage (inside nestynet_sr.sr_search).
        # Lazy import to avoid circular dependency at module-import time.
        try:
            from nestynet_sr.sr_search.factorized_search.adapters.nestynet.stageb_runner import (
                build_stageb_probe_jobs,
                build_stageb_main_candidates,
                has_structural_solved_result as _has_structural_solved_result_impl,
                pool_stageb_results,
                prepare_stageb_embed_context,
                run_stageb_explorer_jobs,
                run_stageb_wrapper_pass,
            )
        except ImportError as exc:
            ctx.log(f"[Stage B]  factorized_search: cannot import adapter ({exc})")
            return []

        st = ctx.state

        prep = prepare_stageb_explorer_inputs(
            root=ctx.state.root,
            target=target,
            units_spec=getattr(ctx, "units_spec", None),
        )
        declared_consts = prep.declared_consts
        var_dims = prep.var_dims
        y_dims = prep.y_dims

        probe_jobs = build_stageb_probe_jobs(ctx=ctx, target=target, log_fn=ctx.log)
        if not probe_jobs:
            return []

        results_raw = run_stageb_explorer_jobs(
            rule=self,
            target=target,
            probe_jobs=probe_jobs,
            declared_consts=declared_consts,
            var_dims=var_dims,
            y_dims=y_dims,
            device=ctx.device,
            dtype=ctx.dtype,
            log_fn=ctx.log,
        )

        if not results_raw:
            return []

        results = pool_stageb_results(results_raw, return_topk=self.return_topk)
        n_results_raw = len(results_raw)
        n_results_pooled = len(results)

        # --- convert results to Candidates ---
        # The explorer returns ASTs in *atom-local* variable indices (0..nvars-1).
        # We need to remap those to the atom's input expressions (which may be
        # compound, e.g. Mul(Var(0), Var(1)) for z = x0*x1).
        # When constants are injected, indices nvars.. map to declared FreeConst/FixedConst atoms.
        input_exprs = prep.input_exprs

        embed_ctx, updated_units_spec = prepare_stageb_embed_context(
            root=st.root,
            target=target,
            units_spec=getattr(ctx, "units_spec", None),
            enforce_units=bool(getattr(ctx, "enforce_units", False)),
        )
        ctx.units_spec = updated_units_spec

        candidates: List[Candidate] = build_stageb_main_candidates(
            root=st.root,
            target=target,
            results=results,
            input_exprs=input_exprs,
            embed_ctx=embed_ctx,
            refine_enable=self.refine_enable,
            refine_stageb_promote_consts=self.refine_stageb_promote_consts,
            log_fn=ctx.log,
        )

        wrapper_candidates = run_stageb_wrapper_pass(
            rule=self,
            root=st.root,
            target=target,
            probe_jobs=probe_jobs,
            declared_consts=declared_consts,
            var_dims=var_dims,
            y_dims=y_dims,
            input_exprs=input_exprs,
            embed_ctx=embed_ctx,
            main_structurally_solved=_has_structural_solved_result_impl(
                results_raw,
                early_stop_mse=float(self.early_stop_mse),
            ),
            enforce_units=bool(getattr(ctx, "enforce_units", False)),
            device=ctx.device,
            dtype=ctx.dtype,
            log_fn=ctx.log,
        )

        if wrapper_candidates:
            candidates.extend(wrapper_candidates)

        ctx.log(
            f"[Stage B]  factorized_search on NN vars={target.var_idxs}: "
            f"{len(candidates)} candidates from {n_results_pooled} pooled "
            f"results ({n_results_raw} raw, {len(probe_jobs)} probe dataset(s))"
        )
        return candidates
