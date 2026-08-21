# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""factorized symbolic search-owned configuration surfaces."""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping


@dataclass
class FactorizedSearchConfig:
    """Hyper-parameters for the factorized symbolic search explorer."""

    # Human-readable profile label for experiment/result provenance. This does
    # not affect the search directly; runners set it after applying their
    # benchmark/profile defaults.
    search_profile: str = "config_default"

    # Core explorer budget
    n_iter: int = 50_000
    # Optional wall-clock cap for one explorer/oracle job. This is enforced at
    # safe boundaries, so it is a soft deadline rather than a hard kill.
    wall_time_limit_s: float | None = None
    max_depth: int = 5
    poly_degree: int = 4

    # Scoring augmentation: multi-term linear head on residual
    #
    # When enabled, factorized symbolic search scoring can fit:
    #   y ~= mapping(f(x)) + (b0 + sum_i a_i * t_i(x))
    # where t_i are simple unit-consistent terms (e.g. raw variables, optionally a
    # few extra pool terms via a cheap OMP step). This is primarily a scoring
    # augmentation that helps "partial" skeletons get credit when the remaining
    # structure is an additive linear combination.
    score_head_enable: bool = True
    # Include unit-matching raw variables as candidate head terms?
    score_head_vars_enable: bool = True
    # Select up to K extra head terms from the pool via OMP?
    score_head_omp_enable: bool = False
    score_head_omp_max_terms: int = 2
    score_head_omp_topk_try: int = 15
    # Ridge regularisation for the head fit (defaults to refine_linear_ridge when None).
    score_head_ridge: float | None = None
    # Minimum relative probe-MSE improvement required to keep the head.
    score_head_min_rel_improve: float = 0.0
    # Permit score-head terms when dimensional metadata is absent. This is kept
    # opt-in because untyped heads are intentionally less conservative; the DE
    # first-line operator-factorized DE path enables it only inside trajectory-validated runs.
    score_head_untyped_enable: bool = False
    # DE-specific treatment of residual score heads.  Hidden heads are useful as
    # proposal evidence but should not become accepted differential laws unless
    # explicitly allowed or compiled into a structural expression.
    de_score_head_policy: str = "proposal_only"
    de_accept_hidden_score_head: bool = False
    de_score_head_untyped_enable: bool = False
    de_score_head_max_terms: int = 2
    # DE additive composer.  This is a factorized-search atom composer, not a
    # fixed-library sparse threshold loop: atoms must come from the FSS slate.
    de_sparse_combo_enable: bool = False
    de_sparse_combo_pre_mutation_enable: bool = True
    de_sparse_combo_pool_topk: int = 8
    de_sparse_combo_max_terms: int = 2
    de_sparse_combo_beam: int = 16
    de_sparse_combo_backward_prune: bool = True
    de_sparse_combo_ridge: float = 1.0e-8
    de_sparse_combo_prune_rel: float = 1.0e-5
    # Keep the composer as RHS = b + sum c_i atom_i by construction.  The generic
    # package default poly_degree can be larger for ordinary SR, so the composer
    # carries its own explicit affine-only guard.
    de_sparse_combo_mapping_mode: str = "affine_only"
    de_sparse_combo_corr_eps: float = 1.0e-8
    de_sparse_combo_rank_eps: float = 1.0e-10
    de_sparse_combo_max_condition: float = 1.0e10
    de_sparse_combo_cond_penalty: float = 0.0
    de_sparse_combo_coeff_stability_penalty: float = 0.0
    de_sparse_combo_coeff_spread_warn: float = 2.0
    # Mapping-family search mode for the outer scorer:
    # - "full": try all families
    # - "gated": try cheap families first, then only escalate to sine/exp on hints
    #   or when the cheap fit is competitive
    # - "cheap": only try cheap families (poly/power/pade)
    score_mapping_family_mode: str = "full"
    # Brute-force can use a stricter mapping-family gate without changing the
    # generic mutation scorer.
    brute_score_mapping_family_mode: str = "gated"
    # Optional typed rational acceptance path.  This is disabled for generic SR:
    # Padé remains a nuisance outer map unless a DE runner explicitly enables
    # compilation of low-complexity Padé maps into validated rational ASTs.
    score_pade_structural_enable: bool = False
    score_pade_structural_max_degree: int = 2
    score_pade_structural_max_total_degree: int = 3
    score_pade_structural_max_depth: int = 8
    score_pade_structural_max_size: int = 64
    score_pade_structural_coeff_tol: float = 1.0e-10
    score_pade_structural_mse_rel_tol: float = 1.0e-6
    # Escalate to sine/exp if the cheap-family fit is within this factor of the
    # current best MSE, or within the relative y-energy threshold below.
    score_mapping_expensive_gate_best_factor: float = 5.0
    score_mapping_expensive_rel_y: float = 0.10
    # Route-aware prescreen for generic mutation candidates. This runs a cheap
    # outer-map fit first and drops clearly uncompetitive candidates before the
    # canonical full scorer.
    score_prescreen_enable: bool = True
    score_prescreen_family_mode: str = "cheap"
    score_prescreen_residual_family_mode: str = "gated"
    score_prescreen_residual_allow_hint: bool = False
    score_prescreen_residual_use_global_best: bool = False
    score_prescreen_parent_best_factor: float = 1.5
    score_prescreen_global_best_factor: float = 3.0
    score_prescreen_residual_parent_best_factor: float = 1.1
    score_prescreen_residual_global_best_factor: float = 1.5
    # Optional finite-mask scoring for domain-restricted expressions.  Generic
    # SR keeps this off by default; DE-facing runners enable it so tiny surrogate
    # domain leakage does not discard expressions such as sqrt(u) before normal
    # ranking and validation.
    score_finite_mask_enable: bool = False
    score_finite_mask_min_fit_frac: float = 0.98
    score_finite_mask_min_probe_frac: float = 0.98
    score_finite_mask_min_dataset_frac: float = 0.95
    score_finite_mask_min_points: int = 8
    # Optional projection/error-tube scoring for restricted-domain operations.
    # When enabled, sqrt/log/asin/acos arguments may be projected to their
    # nearest valid boundary only if the domain violation is within the configured
    # error tube. This is meant for DE surrogate boundary leakage, not generic SR.
    score_domain_projection_enable: bool = False
    score_domain_projection_abs_tol: float = 1.0e-8
    score_domain_projection_rel_tol: float = 1.0e-8
    score_domain_projection_max_frac: float = 1.0
    score_domain_projection_positive_floor: float = 1.0e-12
    # Lightweight global complexity penalty used by oracle-style ranking harnesses.
    complexity_penalty: float = 0.0
    # Optional extra penalty on the fitted outer mapping complexity.
    mapping_complexity_penalty: float = 0.0

    # Atom filtering
    max_arity: int = 3

    # Optional gated outer-wrapper pass:
    #   1) transform target t = phi(y) with a small wrapper set
    #   2) run cheap screens in (X, t)-space
    #   3) run a reduced-budget factorized symbolic search only on top-K wrappers
    #   4) wrap discovered t-expression back with phi^{-1}
    #
    # This provides controlled "function(current factorized expression)" capability without
    # exploding Stage-B cost.
    outer_wrapper_enable: bool = True
    outer_wrapper_max_arity: int = 2
    outer_wrapper_transforms: list[str] = field(
        default_factory=lambda: ["log", "reciprocal", "sqrt", "square", "exp"]
    )
    outer_wrapper_topk: int = 2
    outer_wrapper_min_domain_frac: float = 0.995
    outer_wrapper_min_points: int = 256
    outer_wrapper_probe_max_points: int = 4096
    outer_wrapper_iter_scale: float = 0.20
    outer_wrapper_n_seeds: int = 1
    outer_wrapper_return_topk: int = 2
    # If a cheap transformed-space screen is already this good, skip factorized symbolic search for
    # that wrapper (better left to cheaper rules).
    outer_wrapper_screen_rational_err_max: float = 0.02
    outer_wrapper_screen_nls_err_max: float = 0.02

    # Result selection
    return_topk: int = 5

    # Gridding (points used per expression evaluation)
    n_fit: int = 512
    n_probe: int = 2048

    # Sampling bounds (fallback when data bounds unavailable)
    lo: float = 1.0
    hi: float = 5.0

    # Seed sweep (split iteration budget across multiple seeds)
    seed: int = 0
    n_seeds: int = 10
    split_iter_across_seeds: bool = True
    # Legacy crossover is now the only supported policy.
    crossover_mode: str = "legacy"

    # Brute-force enumeration phase (runs before mutation search)
    brute_depth: int | None = None  # None = adaptive (budget-limited, ceiling 10)
    # Global solved threshold used by BOTH brute-force and mutation phases.
    early_stop_mse: float = 1e-8
    brute_max_expressions: int = 1_000
    # Disable residual-guided build action entirely.
    no_residual: bool = False

    # Periodic seeding: a cheap periodogram per variable proposes rough
    # angular frequencies; sin/cos(omega*x_j) candidates are scored into the
    # archive before mutation so frequency identification (forcing terms,
    # parametric drives) starts from a data-driven guess instead of
    # canonical-frequency skeletons that correlation-guided search cannot see.
    periodic_seed_enable: bool = True
    periodic_seed_max_hints: int = 2
    periodic_seed_min_prominence: float = 8.0

    # Optional continuous skeleton refinement inner refinement (Phases 1-4).
    refine_enable: bool = False
    refine_profile: str = "default"
    # Placement policy for continuous skeleton refinement:
    # - "off": disable refinement even if refine_enable is set
    # - "inline": compatibility path; ordinary scoring calls may refine
    # - "slate"/"final_polish": run scheduled archive-slate refinement passes
    refine_mode: str = "slate"
    refine_during_brute: bool = False
    refine_during_mutation: bool = False
    refine_during_controller_slate: bool = False
    refine_during_slate: bool = True
    refine_slate_after_brute: bool = True
    refine_slate_period: int = 0
    refine_final_polish: bool = True
    refine_slate_k: int = 16
    refine_slate_diverse_k: int = 8
    refine_slate_budget: int = 32
    refine_optimizer: str = "lbfgs"
    refine_lbfgs_escalate_improve_factor: float = 2.0
    refine_lbfgs_steps: int = 20
    refine_fit_subset: int = 256
    refine_fit_subset_mode: str = "hash_random"
    # When continuous skeleton refinement is used with multiple datasets (e.g. DE fitting from multiple
    # trajectories/ICs), allow joint refinement of nonlinear hparams while
    # solving linear coefficients independently per dataset.
    refine_joint_enable: bool = True
    refine_joint_weight_mode: str = "points"
    refine_joint_score_enable: bool = True

    # If enabled (in joint multi-dataset scoring), fit per-dataset linear coefficients
    # for additive terms (multi-parameter per dataset) instead of only an affine
    # remap of the whole expression.
    refine_joint_terms_enable: bool = False

    # When joint multi-dataset data are provided to factorized symbolic search / continuous skeleton refinement (e.g. Class-SR / multi-trajectory),
    # optionally score candidates by fitting a per-dataset affine output map (degree-1 poly) before
    # aggregating the probe loss. This helps recover shared structure when datasets differ by linear
    # gains/offsets (e.g. different spring constants).
    refine_num_restarts: int = 2
    refine_max_variants: int = 4
    refine_max_params: int = 2
    refine_slot_sensitivity_enable: bool = True
    refine_slot_sensitivity_subset: int = 64
    refine_slot_sensitivity_delta: float = 0.1
    refine_slot_sensitivity_max_paths: int = 24
    refine_prune_mapping_equiv_root_slots: bool = True
    refine_attempt_cache_enable: bool = True
    refine_attempt_cache_max_entries: int = 4096
    refine_linear_combo_enable: bool = True
    refine_linear_terms_max: int = 6
    refine_linear_prune_rel: float = 1.0e-10
    refine_linear_ridge: float = 1.0e-8
    refine_gate_best_factor: float = 10.0
    refine_gate_potential_enable: bool = True
    refine_gate_potential_subset: int = 64
    refine_gate_potential_improve_factor: float = 5.0
    refine_gate_log_min: float = -0.6931471805599453
    refine_gate_log_max: float = 1.3862943611198906
    refine_gate_grid_size: int = 4
    refine_gate_max_evals: int = 64
    refine_max_trials: int = 1500
    refine_trials_per_brute_depth: int = 64
    refine_trials_per_mutation_window: int = 64
    refine_mutation_window: int = 500
    refine_safe_eps: float = 1.0e-6
    refine_safe_penalty_weight: float = 1.0e-2
    refine_safe_exp_clip: float = 30.0
    refine_theta_l2: float = 1.0e-4
    refine_init_log_min: float = -1.5
    refine_init_log_max: float = 1.5
    refine_grid_enable: bool = True
    refine_grid_size: int = 33
    refine_grid_size_2d: int = 11
    refine_grid_passes: int = 2
    refine_grid_topk: int = 2
    refine_grid_max_evals: int = 256
    refine_stall_gate_relax_factor: float = 3.0
    refine_stall_gate_relax_max: float = 100.0
    refine_stageb_promote_consts: bool = True

    # Stall-triggered soft restart within the mutation loop
    stall_window: int = 500
    stall_patience: int = 3
    stall_delta: float = 1e-4
    # Optional hard stop after repeated stall-triggered soft restarts without
    # global-best progress. Kept default-off for general SR; DE lanes opt in
    # where rollout validation protects correctness.
    plateau_stop_enable: bool = False
    plateau_stop_max_soft_restarts: int = 0
    plateau_stop_min_evals: int = 0

    # Degenerate-launch abort: stop the mutation loop early when almost no
    # proposals are accepted and the best score has not moved since the brute
    # phase (e.g. single-basin archives where most actions cannot propose).
    degenerate_abort_enable: bool = True
    degenerate_abort_min_evals: int = 1000
    degenerate_abort_max_accepted: int = 8

    # Context-sensitive inverse steering.
    #
    # Choose a subtree path, approximately invert the surrounding context to get
    # a pseudo-target for that subtree, and search local replacements against the
    # pseudo-target before rescoring globally. This is intended to give partial
    # credit to promising incomplete scaffolds, especially in deeper trees.
    inverse_steering_enable: bool = False
    inverse_max_paths: int = 12
    inverse_topk_terms: int = 6
    inverse_shortlist_mult: int = 4
    inverse_min_valid_frac: float = 0.25
    inverse_min_confidence: float = 0.10
    inverse_safe_eps: float | None = None
    # Confidence model for inverse target propagation:
    # - "conditioning": confidence decays with local inverse condition numbers
    # - "heuristic": keep legacy fixed per-op confidence factors
    inverse_confidence_mode: str = "conditioning"
    # In conditioning mode, gains <= 1.0 are treated as stable; larger gains are
    # softly down-weighted by this scale.
    inverse_confidence_target_gain: float = 4.0
    # Lower bound for per-step confidence (applied before path-product).
    inverse_confidence_floor: float = 0.05
    # Beam size for ambiguous inverse branches (sin/cos/sqr); 1 keeps legacy behavior.
    inverse_branch_beam_width: int = 1
    # Optional local subtree micro-search after pseudo-target construction.
    inverse_micro_search_enable: bool = False
    inverse_micro_search_max_depth: int = 3
    inverse_micro_search_beam_width: int = 24
    inverse_micro_search_topk: int = 16
    inverse_micro_search_seed_terms: int = 8
    # Local scorer used to rank inverse subtree repairs: strict | affine | fitbest.
    # Affine works better as the default nuisance model because inverse pseudo-targets
    # are often correct up to a small local scale/shift calibration error.
    inverse_local_score_mode: str = "affine"
    # Direct-spec local SR over the inverse pseudo-target. This augments the
    # existing legacy local repair candidates without changing exact scoring.
    inverse_spec_enable: bool = False
    inverse_spec_enum_max_depth: int = 4
    inverse_spec_enum_max_trees: int = 5000
    inverse_spec_preview_topk: int = 16
    inverse_spec_local_score_mode: str = "affine"
    inverse_spec_include_legacy_seed: bool = True
    inverse_spec_complexity_penalty: float = 0.0
    inverse_spec_family_battery_enable: bool = False
    inverse_spec_family_battery_mode: str = "outer"
    inverse_spec_recursive_enable: bool = True
    inverse_spec_recursive_max_depth: int = 2
    inverse_spec_recursive_trigger_rel_mse: float = 0.25
    inverse_spec_recursive_seed_cap: int = 6
    inverse_spec_recursive_branch_topk: int = 4
    inverse_spec_recursive_child_topk: int = 2
    inverse_spec_recursive_sr_enable: bool = False
    inverse_spec_recursive_sr_preview_topk: int = 4
    inverse_spec_recursive_sr_exact_budget: int = 2
    inverse_spec_constant_lift_route_enable: bool = False
    inverse_spec_constant_lift_route_topk: int = 2
    inverse_spec_coordinate_lift_enable: bool = False
    inverse_spec_coordinate_lift_topk: int = 4
    inverse_spec_coordinate_lift_mode: str = "both"
    inverse_spec_tangent_edit_enable: bool = False
    inverse_spec_tangent_edit_topk: int = 8
    inverse_spec_soft_edit_enable: bool = False
    inverse_spec_soft_edit_steps: int = 64
    inverse_spec_soft_edit_l1: float = 1.0e-3
    inverse_spec_witness_jets_enable: bool = False
    inverse_spec_witness_d2_enable: bool = False
    inverse_spec_witness_max_rows: int = 64
    inverse_spec_witness_loss_enable: bool = False
    inverse_spec_witness_grad_weight: float = 1.0
    inverse_spec_witness_d2_weight: float = 0.0
    inverse_spec_witness_diag_weight: float = 0.0
    inverse_spec_witness_physics_weight: float = 0.0
    inverse_spec_active_var_screen_enable: bool = False
    inverse_spec_active_var_grad_tol: float = 1.0e-3
    inverse_spec_active_var_max_count: int = 4
    inverse_spec_directional_market_enable: bool = False
    # Quota-forced inverse-spec repair: fraction of eligible repair opportunities
    # where A_INVSTEER is forced regardless of bandit selection.  Set > 0 to test
    # whether the full-search gap is primarily a scheduling problem.
    inverse_spec_repair_quota: float = 0.0
    # Optional end-of-phase archive repair pass. This decouples the inverse
    # solver from the online mutation loop and applies it directly to archive
    # elites after structural exploration is done.
    repair_pass_enable: bool = False
    repair_pass_elite_k: int = 8
    repair_pass_paths_per_elite: int = 2
    repair_pass_rounds: int = 2
    # One-shot outer-scaffold pass: enumerate small typed outer forms with a
    # single hole, fill the hole with inverse-spec preview, then exact-score
    # the top completed candidates before mutation starts.
    closure_search_enable: bool = False
    closure_search_families: list[str] = field(
        default_factory=lambda: ["periodic", "exp", "log", "rational", "power", "quadratic"]
    )
    closure_search_max_proposals: int = 16
    closure_search_anchors_per_family: int = 4
    closure_search_preview_topk: int = 4
    closure_search_exact_topk: int = 2
    closure_search_beam_width: int = 4
    # Empty-basis seed round: exact-score a small diverse set of candidates
    # before committing to the first basis state and restarting the loop.
    closure_search_seed_exact_topk: int = 6
    closure_search_seed_beam_width: int = 4
    closure_search_seed_scaffold_reserve: int = 8
    closure_search_seed_family_cap: int = 2
    closure_search_seed_exact_bound_bonus: float = 0.25
    closure_search_pair_normal_enable: bool = False
    closure_search_pair_normal_topk: int = 3
    closure_search_pair_normal_max_pairs: int = 1
    closure_search_pair_rescue_enable: bool = True
    closure_search_pair_rescue_topk: int = 4
    closure_search_pair_rescue_max_pairs: int = 6
    # Legacy row-promotion ablation: harvest small target-dimension
    # additive/subtractive subexpressions from good closure previews and admit
    # them as candidate rows. The auxiliary-atom path below is preferred for
    # FSS vocabulary expansion.
    closure_search_emergent_basis_enable: bool = False
    closure_search_emergent_basis_max_source_rows: int = 32
    closure_search_emergent_basis_score_topk: int = 8
    closure_search_emergent_basis_max_per_round: int = 1
    closure_search_emergent_basis_max_total: int = 4
    closure_search_emergent_basis_min_probe_gain_rel: float = 5.0e-3
    # Emergent auxiliary atom registry: harvest reusable target-dimension or
    # dimensionless motifs from round r and expose them as SeedBlocks in round
    # r+1, without promoting them as final rows.
    closure_search_emergent_aux_atoms_enable: bool = False
    closure_search_emergent_aux_atoms_max_source_rows: int = 48
    closure_search_emergent_aux_atoms_max_new_per_round: int = 5
    closure_search_emergent_aux_atoms_max_total: int = 8
    closure_search_emergent_aux_atoms_max_target: int = 4
    closure_search_emergent_aux_atoms_max_dimensionless: int = 3
    closure_search_emergent_aux_atoms_max_rational_derived: int = 2
    closure_search_emergent_aux_atoms_max_seed_blocks: int = 8
    # Optional compact controller trace export for debugging. When > 0, the
    # oracle report can include top closure preview rows, exact-scored rows,
    # rescue pools, and pair attempts up to a small capped budget.
    closure_search_debug_topk: int = 0
    # Scaffold-origin parents are intentionally weaker than live archive edits,
    # so let the outer-scaffold route use a more permissive inverse gate.
    closure_search_min_valid_frac: float = 0.05
    closure_search_min_confidence: float = 0.02
    closure_search_periodic_min_valid_scale: float = 1.0
    closure_search_periodic_min_confidence_scale: float = 1.0
    closure_search_transport_min_lin_rel: float = 0.0
    closure_search_anchor_head_compare_enable: bool = False
    # --- Hole search: opportunity-driven targeted hole filling ---
    hole_search_enable: bool = False
    hole_search_quota: float = 0.10
    hole_search_exact_budget: int = 2
    hole_search_cooldown_iters: int = 32
    # Separate cooldown for expensive archive-mining of new opportunities.
    hole_search_mine_cooldown_iters: int = 50
    hole_search_max_frontier: int = 128
    # Promote hole/spec opportunities to the native scheduler state instead of
    # treating A_HOLESEARCH as just another mutation-loop action.
    hole_search_first_class_scheduler_enable: bool = True
    hole_search_route_scheduler_enable: bool = True
    hole_search_route_ucb_c: float = 0.25
    hole_search_route_eps: float = 0.05
    hole_search_route_acquisition_weight: float = 0.25
    hole_search_route_reward_mode: str = "penalized"
    hole_search_route_time_penalty: float = 0.01
    hole_search_route_time_floor: float = 1.0
    hole_search_abstraction_enable: bool = True
    hole_search_abstraction_on_improve: bool = True
    hole_search_abstraction_on_stall: bool = True
    hole_search_abstraction_cooldown_iters: int = 25
    hole_search_abstraction_max_parents: int = 2
    hole_search_abstraction_max_paths_per_parent: int = 3
    hole_search_abstraction_improve_min_delta_log_mse: float = 0.15
    hole_search_abstraction_stage_enable: bool = True
    hole_search_abstraction_stage_max_entries: int = 64
    hole_search_abstraction_promote_topk: int = 2
    hole_search_abstraction_promote_frontier_floor: int = 3
    hole_search_enum_max_depth: int = 4
    hole_search_enum_max_trees: int = 3000
    hole_search_preview_topk: int = 8
    hole_search_solver_market_enable: bool = False
    hole_search_solver_market_preview_topk: int = 4
    hole_search_solver_market_exact_topk: int = 2
    hole_search_solver_market_proposal_objects_enable: bool = False
    scheduler_witness_energy_enable: bool = False
    # Risk-seeking tournament: cheap-preview many holes, full-execute only
    # the top elite_k.  Mirrors the PhySO risk-seeking policy (evaluate many,
    # train on top epsilon%).  Total compute is roughly neutral because the
    # cheap previews use a small enum budget.
    hole_search_tournament_enable: bool = True
    hole_search_tournament_n: int = 8
    hole_search_tournament_elite_k: int = 2
    hole_search_tournament_preview_trees: int = 64
    # Independent subtree depth cap for inverse-spec proposals (None = use max_depth).
    inverse_spec_max_subtree_depth: int | None = None
    # Larger sample caps for inverse-spec (legacy path uses 32/64 which is too
    # harsh for periodic signals and latent constants).
    inverse_spec_fit_cap: int = 96
    inverse_spec_probe_cap: int = 192
    # Separate exact-score budget for inverse-spec proposals so they don't
    # compete with legacy candidates in the same 4-slot budget.
    inverse_spec_exact_budget: int = 4
    # Robust inverse-target selection: try simpler target constructions before trusting a
    # complex fitted outer map when the peeled path is exact+monotone.
    # Supported modes: robust | full | simple | identity | affine
    inverse_target_mode: str = "robust"
    inverse_full_mapping_penalty: float = 0.75
    inverse_exact_simple_target_bonus: float = 0.10
    # Prefer symbolic cuts at the structural repair site rather than deep descendants.
    inverse_additive_descend_penalty: float = 0.15
    inverse_nonadditive_leaf_penalty: float = 0.20
    # In exact monotone contexts, keep the inverse pseudo-target almost unchanged and
    # relax the linearized transport gate so structural jumps are not vetoed too early.
    inverse_exact_path_eta: float = 0.98
    inverse_exact_transport_min_lin_rel: float = 0.0
    inverse_gate_enable: bool = True
    inverse_gate_warmup: int = 0
    inverse_gate_best_factor: float = 20.0
    inverse_gate_min_residual_basins: int = 0
    inverse_gate_min_depth: int = 4
    inverse_gate_min_size: int = 6
    inverse_gate_max_paths: int = 6
    inverse_gate_min_structural_score: float = 0.75
    inverse_gate_min_weighted_rel_gain: float = 0.05
    inverse_gate_structural_bias: float = 0.20
    # Family-aware inverse gate/action scaling:
    # periodic paths require stronger validity/confidence and get a path-level penalty,
    # while non-periodic mul/div and exp/log/sqrt contexts can get small bonuses.
    inverse_periodic_min_valid_scale: float = 1.25
    inverse_periodic_min_confidence_scale: float = 1.35
    inverse_periodic_path_penalty: float = 0.65
    inverse_nonperiodic_muldiv_bonus: float = 0.10
    inverse_nonperiodic_explogsqrt_bonus: float = 0.05
    # Branch-beam evidence factor for ambiguous inverse branches (sin/cos/sqr).
    inverse_branch_ambiguity_penalty: float = 0.50
    # Transport-safety gate for inverse action proposals.
    # Require a minimum normalized linearized gain and minimum weighted support.
    inverse_transport_min_lin_rel: float = 0.02
    inverse_transport_min_effective_n: float = 8.0
    # Optional structured logging for inverse-steering invocations. This is meant
    # for oracle / research ablations rather than normal production runs.
    inverse_experiment_log_enable: bool = False
    # Stage-0 repair controller: use inverse steering as a conditional repair
    # option driven by dynamic repairability features, instead of a generic
    # mutation that always competes in the action bandit.
    repair_controller_enable: bool = False
    repair_controller_min_score: float = 0.15
    repair_controller_steps: int = 3
    repair_controller_ancestor_hops: int = 1
    repair_controller_min_step_rel_improve: float = 1.0e-3
    repair_controller_adaptive: bool = True
    repair_controller_adapt_quantile: float = 0.75
    repair_controller_adapt_window: int = 128
    repair_controller_adapt_min_samples: int = 16
    repair_controller_min_concentration: float = 0.30
    repair_controller_potential_weight: float = 1.00
    repair_controller_concentration_weight: float = 0.35
    repair_controller_contrast_weight: float = 0.20
    repair_controller_cost_weight: float = 0.10
    repair_controller_stagnation_weight: float = 0.15
    repair_controller_frontier_topk: int = 24
    repair_controller_stagnation_visits: int = 8
    repair_controller_focus_prob: float = 0.50
    repair_controller_parent_max_repeats: int = 2
    repair_controller_parent_min_eval_gap: int = 32
    repair_controller_parent_reset_rel_improve: float = 0.05
    # Stage-1 learned repair critic: optional shared scorer trained from
    # inverse experiment logs, layered on top of the Stage-0 repair option.
    # "priority" preserves the original bounded sidecar behavior, "gate"
    # lets the critic change repair eligibility directly, and "decisive"
    # also shifts the cached per-parent threshold used by the repair frontier.
    repair_controller_critic_enable: bool = False
    repair_controller_critic_path: str = ""
    repair_controller_critic_blend: float = 1.0
    repair_controller_critic_mode: str = "priority"
    repair_opportunity_controller_enable: bool = False
    repair_opportunity_controller_path: str = ""

    # Residual-guided continuous search (greedy boosting / OMP over the pool)
    boost_enable: bool = False
    boost_max_terms: int = 6
    boost_topk_try: int = 15
    boost_min_rel_improve: float = 1.0e-3
    boost_selection_split: str = "fit"
    boost_ridge: float | None = None
    boost_include_parent: bool = True
    boost_from_scratch_prob: float = 0.25
    boost_prune_rel: float = 1.0e-10
    boost_safe_eval: bool = True

    # Automatic gating for boost action (avoid wasting compute / avoid early linear-soup fits)
    boost_gate_enable: bool = True
    boost_gate_warmup: int = 200
    boost_gate_best_factor: float = 30.0
    boost_gate_gain_frac: float = 1.0e-2
    boost_gate_peak_ratio: float = 5.0
    boost_gate_min_valid: int = 8
    boost_gate_min_residual_basins: int = 10

    # Adaptive gating: set gain_frac threshold from a running quantile
    boost_gate_adaptive: bool = True
    boost_gate_adapt_quantile: float = 0.75
    boost_gate_adapt_window: int = 256
    boost_gate_adapt_min_samples: int = 32
    boost_gate_adapt_mix: float = 1.0
    boost_gate_gain_frac_floor: float = 1.0e-4
    boost_gate_gain_frac_cap: float = 0.25

    # Optional pool expansion: harvest simple subtrees from the archive
    boost_harvest_enable: bool = False
    boost_harvest_every: int = 500
    boost_harvest_topk_residual_basins: int = 50
    boost_harvest_elites_per_residual_basin: int = 2
    boost_pool_extra_max: int = 256
    boost_subtree_depth_max: int = 3
    boost_subtree_size_max: int = 12


@dataclass(frozen=True)
class InverseSteeringConfig:
    """Shared inverse-steering execution config for repair/build call paths."""

    max_paths: int = 12
    topk_terms: int = 6
    shortlist_mult: int = 4
    min_valid_frac: float = 0.25
    min_confidence: float = 0.10
    safe_eps: float = 1.0e-12
    confidence_mode: str = "conditioning"
    confidence_target_gain: float = 4.0
    confidence_floor: float = 0.05
    branch_beam_width: int = 1
    micro_search_enable: bool = False
    micro_search_max_depth: int = 3
    micro_search_beam_width: int = 24
    micro_search_topk: int = 16
    micro_search_seed_terms: int = 8
    local_score_mode: str = "affine"
    inverse_spec_enable: bool = False
    inverse_spec_enum_max_depth: int = 4
    inverse_spec_enum_max_trees: int = 5000
    inverse_spec_preview_topk: int = 16
    inverse_spec_local_score_mode: str = "affine"
    inverse_spec_include_legacy_seed: bool = True
    inverse_spec_complexity_penalty: float = 0.0
    inverse_spec_family_battery_enable: bool = False
    inverse_spec_family_battery_mode: str = "outer"
    inverse_spec_recursive_enable: bool = True
    inverse_spec_recursive_max_depth: int = 2
    inverse_spec_recursive_trigger_rel_mse: float = 0.25
    inverse_spec_recursive_seed_cap: int = 6
    inverse_spec_recursive_branch_topk: int = 4
    inverse_spec_recursive_child_topk: int = 2
    inverse_spec_recursive_sr_enable: bool = False
    inverse_spec_recursive_sr_preview_topk: int = 4
    inverse_spec_recursive_sr_exact_budget: int = 2
    inverse_spec_constant_lift_route_enable: bool = False
    inverse_spec_constant_lift_route_topk: int = 2
    inverse_spec_coordinate_lift_enable: bool = False
    inverse_spec_coordinate_lift_topk: int = 4
    inverse_spec_coordinate_lift_mode: str = "both"
    inverse_spec_tangent_edit_enable: bool = False
    inverse_spec_tangent_edit_topk: int = 8
    inverse_spec_soft_edit_enable: bool = False
    inverse_spec_soft_edit_steps: int = 64
    inverse_spec_soft_edit_l1: float = 1.0e-3
    inverse_spec_witness_jets_enable: bool = False
    inverse_spec_witness_d2_enable: bool = False
    inverse_spec_witness_max_rows: int = 64
    inverse_spec_witness_loss_enable: bool = False
    inverse_spec_witness_grad_weight: float = 1.0
    inverse_spec_witness_d2_weight: float = 0.0
    inverse_spec_witness_diag_weight: float = 0.0
    inverse_spec_witness_physics_weight: float = 0.0
    inverse_spec_active_var_screen_enable: bool = False
    inverse_spec_active_var_grad_tol: float = 1.0e-3
    inverse_spec_active_var_max_count: int = 4
    inverse_spec_directional_market_enable: bool = False
    inverse_spec_max_subtree_depth: int | None = None
    inverse_spec_fit_cap: int = 96
    inverse_spec_probe_cap: int = 192
    inverse_spec_exact_budget: int = 4
    target_mode: str = "robust"
    full_mapping_penalty: float = 0.75
    exact_simple_target_bonus: float = 0.10
    additive_descend_penalty: float = 0.15
    nonadditive_leaf_penalty: float = 0.20
    exact_path_eta: float = 0.98
    exact_transport_min_lin_rel: float = 0.0
    periodic_min_valid_scale: float = 1.25
    periodic_min_confidence_scale: float = 1.35
    periodic_path_penalty: float = 0.65
    nonperiodic_muldiv_bonus: float = 0.10
    nonperiodic_explogsqrt_bonus: float = 0.05
    branch_ambiguity_penalty: float = 0.50
    transport_min_lin_rel: float = 0.02
    transport_min_effective_n: float = 8.0

    def to_action_kwargs(self) -> dict[str, Any]:
        return {
            spec.name: getattr(self, spec.name)
            for spec in fields(type(self))
        }


def inverse_steering_config_from_mapping(values: Mapping[str, Any] | None = None) -> InverseSteeringConfig:
    if values is None:
        return InverseSteeringConfig()
    payload: dict[str, Any] = {}
    for spec in fields(InverseSteeringConfig):
        if spec.name in values:
            payload[spec.name] = values[spec.name]
            continue
        prefixed_name = f"inverse_{spec.name}"
        if prefixed_name in values:
            payload[spec.name] = values[prefixed_name]
    return InverseSteeringConfig(**payload)


def coerce_inverse_steering_config(
    value: InverseSteeringConfig | Mapping[str, Any] | None,
) -> InverseSteeringConfig:
    if isinstance(value, InverseSteeringConfig):
        return value
    if value is None:
        return InverseSteeringConfig()
    if isinstance(value, Mapping):
        return inverse_steering_config_from_mapping(value)
    raise TypeError(f"unsupported inverse steering config type: {type(value)!r}")


DEFAULT_REFINE_PROFILE: dict[str, Any] = {
    "refine_mode": "slate",
    "refine_during_brute": False,
    "refine_during_mutation": False,
    "refine_during_controller_slate": False,
    "refine_during_slate": True,
    "refine_slate_after_brute": True,
    "refine_slate_period": 0,
    "refine_final_polish": True,
    "refine_slate_k": 16,
    "refine_slate_diverse_k": 8,
    "refine_slate_budget": 32,
}

INLINE_REFINE_PROFILE: dict[str, Any] = {
    "refine_mode": "inline",
    "refine_during_brute": True,
    "refine_during_mutation": True,
    "refine_during_controller_slate": False,
    "refine_during_slate": False,
    "refine_slate_after_brute": True,
    "refine_slate_period": 0,
    "refine_final_polish": True,
    "refine_slate_k": 16,
    "refine_slate_diverse_k": 8,
    "refine_slate_budget": 32,
}

RARE_REFINE_PROFILE: dict[str, Any] = {
    **INLINE_REFINE_PROFILE,
    "refine_max_trials": 50,
    "refine_max_variants": 1,
    "refine_max_params": 1,
    "refine_num_restarts": 1,
    "refine_lbfgs_steps": 4,
    "refine_fit_subset": 64,
    "refine_grid_size": 17,
    "refine_grid_size_2d": 5,
    "refine_grid_passes": 1,
    "refine_grid_max_evals": 32,
    "refine_gate_best_factor": 2.0,
    "refine_gate_potential_enable": True,
    "refine_gate_potential_subset": 64,
    "refine_gate_max_evals": 16,
    "refine_stall_gate_relax_factor": 1.0,
}

RARE_SLATE_REFINE_PROFILE: dict[str, Any] = {
    **RARE_REFINE_PROFILE,
    **DEFAULT_REFINE_PROFILE,
}

RARE_FINAL_POLISH_REFINE_PROFILE: dict[str, Any] = {
    **RARE_SLATE_REFINE_PROFILE,
    "refine_mode": "final_polish",
    "refine_slate_after_brute": False,
}

REFINE_PROFILE_NAMES: tuple[str, ...] = (
    "default",
    "inline",
    "rare",
    "rare_slate",
    "rare_final_polish",
)

REFINE_OPTIMIZER_NAMES: tuple[str, ...] = (
    "lbfgs",
    "grid",
    "grid_then_lbfgs",
)

_REFINE_PROFILE_ALIASES: dict[str, str] = {
    "": "default",
    "default": "default",
    "none": "default",
    "current": "default",
    "inline": "inline",
    "legacy": "inline",
    "compat_inline": "inline",
    "legacy_inline": "inline",
    "rare": "rare",
    "stingy": "rare",
    "rare_inline": "rare",
    "rare-inline": "rare",
    "rare_slate": "rare_slate",
    "rare-slate": "rare_slate",
    "slate": "rare_slate",
    "rare_final_polish": "rare_final_polish",
    "rare-final-polish": "rare_final_polish",
    "final_polish": "rare_final_polish",
    "final-polish": "rare_final_polish",
}

REFINE_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "default": dict(DEFAULT_REFINE_PROFILE),
    "inline": dict(INLINE_REFINE_PROFILE),
    "rare": dict(RARE_REFINE_PROFILE),
    "rare_slate": dict(RARE_SLATE_REFINE_PROFILE),
    "rare_final_polish": dict(RARE_FINAL_POLISH_REFINE_PROFILE),
}


def normalize_refine_profile_name(name: str | None) -> str:
    token = str(name or "").strip().lower().replace("-", "_")
    try:
        return str(_REFINE_PROFILE_ALIASES[token])
    except KeyError as exc:
        allowed = ", ".join(REFINE_PROFILE_NAMES)
        raise ValueError(f"unknown refine profile {name!r}; expected one of {allowed}") from exc


def resolve_refine_profile(name: str | None) -> tuple[str, dict[str, Any]]:
    canonical = normalize_refine_profile_name(name)
    return canonical, dict(REFINE_PROFILE_OVERRIDES.get(canonical, {}))


def apply_refine_profile(
    hp: FactorizedSearchConfig,
    profile: str | None,
) -> FactorizedSearchConfig:
    canonical, overrides = resolve_refine_profile(profile)
    hp.refine_profile = str(canonical)
    valid_fields = {spec.name for spec in fields(FactorizedSearchConfig)}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise AttributeError(f"unknown FactorizedSearchConfig field in refine profile: {key}")
        setattr(hp, key, value)
    return hp


def apply_refine_mode_placement_defaults(
    hp: FactorizedSearchConfig,
    mode: str | None,
) -> FactorizedSearchConfig:
    mode_norm = str(mode or "").strip().lower().replace("-", "_")
    aliases = {
        "": "slate",
        "none": "off",
        "false": "off",
        "disabled": "off",
        "true": "slate",
        "on": "slate",
        "legacy": "inline",
    }
    mode_norm = aliases.get(mode_norm, mode_norm)
    if mode_norm not in {"off", "inline", "slate", "final_polish"}:
        raise ValueError(f"unknown refine mode {mode!r}")
    hp.refine_mode = mode_norm
    if mode_norm == "off":
        hp.refine_during_brute = False
        hp.refine_during_mutation = False
        hp.refine_during_controller_slate = False
        hp.refine_during_slate = False
        hp.refine_slate_after_brute = False
        hp.refine_final_polish = False
    elif mode_norm == "inline":
        hp.refine_during_brute = True
        hp.refine_during_mutation = True
        hp.refine_during_controller_slate = False
        hp.refine_during_slate = False
    elif mode_norm == "slate":
        hp.refine_during_brute = False
        hp.refine_during_mutation = False
        hp.refine_during_controller_slate = False
        hp.refine_during_slate = True
        hp.refine_slate_after_brute = True
        hp.refine_final_polish = True
    elif mode_norm == "final_polish":
        hp.refine_during_brute = False
        hp.refine_during_mutation = False
        hp.refine_during_controller_slate = False
        hp.refine_during_slate = True
        hp.refine_slate_after_brute = False
        hp.refine_final_polish = True
    return hp


def _config_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_config_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_config_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _config_jsonable(v) for k, v in value.items()}
    try:
        return float(value)
    except Exception:
        return str(value)


def factorized_config_to_dict(hp: FactorizedSearchConfig) -> dict[str, Any]:
    """Return a JSON-friendly snapshot of a resolved FSS config."""

    payload = dict(asdict(hp))
    # Some DE-facing runners attach small, explicit FSS extension fields to the
    # dataclass instance rather than adding another configuration object.  Keep
    # those visible in result artifacts too; otherwise a run can depend on
    # active gates that are absent from its serialized provenance.
    for key, value in vars(hp).items():
        key_s = str(key)
        if key_s.startswith("_") or key_s in payload:
            continue
        if callable(value):
            continue
        payload[key_s] = value
    return {str(k): _config_jsonable(v) for k, v in payload.items()}


def factorized_config_diff(
    hp: FactorizedSearchConfig,
    *,
    baseline: FactorizedSearchConfig | None = None,
) -> dict[str, Any]:
    """Return JSON-friendly fields whose resolved values differ from baseline."""

    base = baseline if baseline is not None else FactorizedSearchConfig()
    cur = factorized_config_to_dict(hp)
    ref = factorized_config_to_dict(base)
    return {k: v for k, v in cur.items() if ref.get(k) != v}


def factorized_config_report(
    hp: FactorizedSearchConfig,
    *,
    baseline: FactorizedSearchConfig | None = None,
) -> dict[str, Any]:
    """Compact provenance block for benchmark outputs."""

    resolved = factorized_config_to_dict(hp)
    return {
        "schema_version": 1,
        "includes_disabled_lanes": True,
        "search_profile": str(resolved.get("search_profile", "")),
        "refine_profile": str(resolved.get("refine_profile", "")),
        "refine_mode": str(resolved.get("refine_mode", "")),
        "field_count": int(len(resolved)),
        "resolved": resolved,
        "diff_from_config_default": factorized_config_diff(hp, baseline=baseline),
    }


__all__ = [
    "FactorizedSearchConfig",
    "DEFAULT_REFINE_PROFILE",
    "INLINE_REFINE_PROFILE",
    "InverseSteeringConfig",
    "RARE_FINAL_POLISH_REFINE_PROFILE",
    "RARE_REFINE_PROFILE",
    "RARE_SLATE_REFINE_PROFILE",
    "REFINE_OPTIMIZER_NAMES",
    "REFINE_PROFILE_NAMES",
    "REFINE_PROFILE_OVERRIDES",
    "apply_refine_profile",
    "apply_refine_mode_placement_defaults",
    "coerce_inverse_steering_config",
    "factorized_config_diff",
    "factorized_config_report",
    "factorized_config_to_dict",
    "inverse_steering_config_from_mapping",
    "normalize_refine_profile_name",
    "resolve_refine_profile",
]
