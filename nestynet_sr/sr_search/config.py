# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Configuration dataclasses for the separability / symbolic regression pipeline.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


@dataclass
class LMHyperparams:
    """
    Hyper-parameters controlling the Levenberg-Marquardt optimiser.
    The actual LM implementation lives in nestynet.optimizer.
    """

    epochs: int = 2500
    epochs_min: int = 1000
    epochs_awful_check: int = 300  # Early check for hopeless fits
    awful_threshold: float = 10.0  # Abandon if val_loss > this at epochs_awful_check
    time_max: float = None
    strategy: str = "direct_solve"
    nval_patience: int = 500
    loss_target: float = 1.0e-7
    chisq_tol: float = 1.0e-10
    loss_acceptable: float = 1.0e-3
    loss_in_MAD_units: bool = True  # thresholds in units of MAD(φ(y))^2

    # ------------------------------------------------------------------
    # Fit-only output link (numeric conditioning)
    # ------------------------------------------------------------------
    # If set, the optimiser sees residuals in fit-space:
    #     r = t(y) - t(f)
    # but all analysis continues to operate in y-space.
    # Supported: asinh, square, inv_square, recip, exp, log.
    fit_y_link: str | None = None
    # Optional scale for asinh: t(y) = asinh(y / fit_y_link_scale)
    fit_y_link_scale: float = 1.0
    # Auto-enable asinh when identity y-transform fails and log-space dynamic range
    # exceeds threshold (in decades). E.g., 4.0 means 10^4 = 4 orders of magnitude.
    auto_fit_link_log_dynamic_range_threshold: float = 4.0

    # ------------------------------------------------------------------
    # Optional NestyNet canonical initialization
    # ------------------------------------------------------------------
    # Applied only to pure-NN SR fits where the target belongs directly
    # to one segmented or dual-segmented NestyNet leaf.
    canonical_init: bool = False
    canonical_init_affine_first: bool = True
    canonical_init_ridge: float = 1.0e-6
    canonical_init_bias_mode: str = "quantile"
    canonical_init_orthogonalize: bool = True

    # ------------------------------------------------------------------
    # Optional evidence-mode pass-through
    # ------------------------------------------------------------------
    # SR remains plain-loss-driven. These flags only control whether the LM
    # optimiser is constructed with an evidence controller underneath.
    evidence_enable: bool = False
    # Convenience switches for the two extra evidence pieces discussed in SR.
    evidence_disable_residual_whitening: bool = False
    evidence_disable_segment_priors: bool = False
    # Low-level evidence weights. When both the patch-whitening weights and the
    # segment-prior alpha are zero, SR collapses back to the legacy LM path.
    evidence_lambda_patch: float = 0.0
    evidence_lambda_mean: float | None = None
    evidence_lambda_slope: float | None = None
    evidence_lambda_quad: float | None = None
    evidence_segment_alpha_init: float = 1.0
    evidence_prior_rel_scale: float = 0.25
    evidence_prior_abs_scale: float = 1.0e-3
    # Segment priors are optimizer-side guidance only. By default, SR/DE use
    # the canonical benchmark schedule, then gate metrics until the prior has
    # been annealed away. The default schedule is clipped into the LM budget
    # so short smoke runs still finish with plain-metric evaluation enabled.
    evidence_prior_decay_auto: bool = True
    evidence_prior_decay_start: int | None = None
    evidence_prior_decay_interval: int = 100
    evidence_prior_decay_shape: str = "cosine"
    evidence_prior_decay_final_scale: float = 0.0
    evidence_prior_cutoff_tol: float = 1.0e-8
    evidence_gate_metrics_until_prior_decay: bool = True

    # ------------------------------------------------------------------
    # Outer-transform lift (stability for sqrt/log/reciprocal candidates)
    # ------------------------------------------------------------------
    # Some candidate families wrap a rational/polynomial core with an outer
    # transform such as sqrt(·), log(·), or reciprocal forms. These can make
    # LM numerically fragile due to singular Jacobians near the transform
    # domain boundary. When enabled, Stage B can do a short *prefit* in a
    # lifted space (e.g. square, exp, recip) before optionally refining in
    # the original space.
    outer_transform_lift_enable: bool = True
    outer_transform_lift_prefit_epochs: int = 200
    outer_transform_lift_prefit_epochs_min: int = 50
    outer_transform_lift_refine_epochs: int = 200
    outer_transform_lift_refine_epochs_min: int = 50

    # ------------------------------------------------------------------
    # asinh fit-link: y-space sanity gate (Stages A & B)
    # ------------------------------------------------------------------
    # The asinh fit-link compresses large residuals and can make very poor
    # y-space fits look deceptively good in fit-space. We therefore enforce
    # a weak sanity check in *y-space* (pre fit-link).
    #
    # We compute D_ref = quantile_q(s^2 + y^2) on the validation set, where
    # s = fit_y_link_scale, and accept only if:
    #
    #   y_MSE <= asinh_yspace_sanity_factor * asinh_MSE * D_ref
    #
    # Optionally we also allow up to asinh_yspace_regress_factor × baseline_y_MSE
    # (useful when the baseline is already excellent in y-space).
    asinh_yspace_sanity_quantile: float = 0.90
    asinh_yspace_sanity_factor: float = 20.0
    asinh_yspace_regress_factor: float = 5.0

    # ------------------------------------------------------------------
    # Shared model-selection policy (Stages A & B)
    # ------------------------------------------------------------------
    # The SR pipeline frequently compares candidate rewrites that trade off
    # validation loss and complexity. Rather than AIC/BIC-style penalties,
    # we use a *budget* approach: a candidate may worsen loss by a limited
    # multiplicative factor that grows gently with simplification.
    #
    # Structural simplification is measured via a cheap NN-only proxy
    # (sum of squared effective arities + a small count penalty).
    # Parameter simplification is measured via log10(parameter ratio).
    #
    # These coefficients convert simplification into an allowed loss
    # regression in *decades* (log10 scale).
    select_count_weight: float = 1.0
    select_struct_gamma: float = 0.05
    select_param_gamma: float = 0.30
    # Small extra allowance (in decades) for separability moves that may not
    # immediately reduce the structural proxy but often unlock later splits.
    select_sep_bonus_decades: float = 0.05
    # Tighter bonus for overlapping / partial separability moves.
    select_partial_sep_bonus_decades: float = 0.02
    # Optional global base budget (decades) applied to any simplifying move.
    select_base_bonus_decades: float = 0.0
    # Stage B safeguard: clamp any accepted rewrite to at most this many
    # decades above the meaningful loss floor (loss_target).
    select_stageB_max_decades_over_floor: float = 1.0
    # Stage B below-floor guard: when base_loss is far below loss_floor,
    # shrink the effective "equivalent-loss" regime to at most this many
    # decades above base_loss.
    select_floor_guard_decades: float = 2.0
    # Stage B below-floor safeguard: even within the below-floor regime,
    # reject candidates that regress by more than this many decades from
    # base_loss.
    select_below_floor_max_regress_decades: float = 1.0

    # Optional externally supplied irreducible loss floor used only for
    # model-selection / acceptance decisions. When provided, candidates are
    # compared in *excess-loss* space above this floor, while optimiser
    # stopping (loss_target, early stopping) remains unchanged.
    #
    # acceptance_noise_floor is expressed in the same base units as
    # loss_target / loss_acceptable (i.e. MAD(φ(y))^2 units when
    # loss_in_MAD_units=True). acceptance_noise_floor_raw bypasses that
    # scaling and is interpreted directly in raw loss units.
    acceptance_noise_floor: float | None = None
    acceptance_noise_floor_raw: float | None = None
    # Optional Stage-B-only noisy-data fallback. When enabled and no explicit
    # acceptance noise floor is supplied, ignore the Stage B hard cap while the
    # current best loss still sits above it. Default False preserves the
    # original clean-data behaviour.
    stageB_overcap_fallback: bool = False

    # ------------------------------------------------------------------
    # Overlap gauge continuation (Stage A overlap-only candidates)
    # ------------------------------------------------------------------
    # For overlapping additive/multiplicative splits we first fit the data
    # without gauge regularisation to find a good local optimum, then require the
    # candidate to survive at least one non-zero gauge stage.
    overlap_gauge_continuation_enable: bool = True
    overlap_add_gauge_weights: list[float] = field(
        default_factory=lambda: [1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    )
    overlap_mul_gauge_weights: list[float] = field(
        default_factory=lambda: [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
    )
    overlap_gauge_max_data_regress_factor: float = 10.0
    overlap_gauge_required_improve_factor: float = 0.3
    # If the ungauged gauge metric is already tiny, only require the non-zero
    # gauge stage to avoid materially regressing it.
    overlap_gauge_tiny_baseline_relax_factor: float = 1.25
    overlap_gauge_tiny_baseline_eps: float = 1.0e-10

    # ------------------------------------------------------------------
    # Stage B: candidate iteration policy (frugal vs greedy with screening)
    # ------------------------------------------------------------------
    # Stage B rules can propose many rewrite candidates for the same target.
    # Stage B rules can propose many rewrite candidates for the same target.
    # The sequential engine evaluates candidates one-by-one with full LM fits.
    # This is robust but can be wasteful when a long tail of weak candidates
    # must be tried before a good one.
    #
    # We support an optional two-tier evaluation:
    #   1) A short *screening* LM fit to cheaply rank candidates.
    #   2) A full LM fit only for the selected candidates.
    #
    # stageB_candidate_policy:
    #   - "sequential": no screening, try candidates sequentially
    #   - "frugal": screen top-K and full-fit only the best (1 attempt per target)
    #   - "greedy": screen top-K to order; then full-fit candidates (screened first)
    #   - "dynamic": if screening shows a decisive winner, use frugal else greedy
    stageB_candidate_policy: str = "dynamic"

    # Enable the screening tier. If False, Stage B uses sequential evaluation.
    stageB_screen_enable: bool = False

    # Screen at most top-K candidates per (rule, target). Candidates are taken
    # in the order proposed by the rule (rules often already apply heuristics).
    stageB_screen_topk: int = 6

    # Number of LM epochs used for screening fits.
    stageB_screen_epochs: int = 80

    # Decisiveness threshold (in log10 decades of loss ratio) used by the
    # "dynamic" policy: if the best screened loss beats the second best by at
    # least this many decades, treat it as a clear winner and be frugal.
    stageB_screen_dominance_decades: float = 0.50

    # Optional cap: in greedy modes, limit the number of full fits attempted
    # per (rule, target). 0 means "no cap" (try all candidates).
    stageB_greedy_fullfit_max: int = 0

    # Nonlinear-substitution rational proposals. ``requested_count`` counts
    # candidates that were built and passed the whole-AST unit assertion, not
    # raw screen hits. The two independent bounds make finite search
    # exhaustion distinguishable from a configured numerical/structural cap.
    stageB_nls_requested_count: int = 3
    stageB_nls_max_attempts: int = 1024
    stageB_nls_max_support_attempts: int = 2048

    # End-of-Stage-B audit log: number of Pareto-front records to print from
    # decision_log using (candidate loss, candidate total complexity).
    stageB_pareto_log_topk: int = 8

    # ------------------------------------------------------------------
    # Post-processing airbags
    # ------------------------------------------------------------------
    # Heavy symbolic post-processing is deliberately not part of Stage-B
    # model selection.  Run it in bounded child processes so SymPy/final
    # polish can fail safely without killing the accepted Stage-B result.
    stageB_polish_subprocess: bool = True
    stageB_polish_max_seconds: float = 300.0
    stageB_polish_mem_fraction: float = 0.20
    stagec_sympy_subprocess: bool = True
    stagec_sympy_max_seconds: float = 300.0
    stagec_sympy_mem_fraction: float = 0.20
    final_polish_subprocess: bool = True
    final_polish_worker_max_seconds: float = 300.0
    final_polish_worker_mem_fraction: float = 0.20

    # CoE committee witness execution. Default 1 preserves historical serial
    # behavior; higher values are consumed by the CoE witness executor.
    coe_witness_parallelism: int = 1

    # Logging configuration
    log_file: str = None  # Path to log file
    log_to_console: bool = True  # Whether to log to console
    log_level: int = None  # Logging level (logging.DEBUG, INFO, etc.)
    LM_verbose: bool = False  # Per-epoch LM optimizer progress output

    # ------------------------------------------------------------------
    # Stage B: compound-function macro proposals (template expansion)
    # ------------------------------------------------------------------
    # These are *screened* (cheap affine fit on a cached batch) and only the
    # best-scoring candidates are passed to the expensive LM fit.
    macro_enable: bool = True
    # Maximum number of input axes from a target NN leaf to consider when
    # building macro arguments (keeps the arg pool bounded for large leaves).
    macro_max_vars: int = 6
    # Maximum number of argument expressions kept in the pool (after cost sort).
    macro_max_arg_exprs: int = 64
    # Maximum number of argument expressions considered per macro argument slot.
    macro_max_pos_args: int = 20
    # Keep at most this many screened hits per macro family.
    macro_max_per_macro: int = 3
    # Total number of macro candidates returned per target leaf.
    macro_max_candidates: int = 10
    # Minimum fraction of finite values required during screening.
    macro_domain_ok_frac: float = 0.98
    # Optional: shared wrapper-policy gates for the macro argument pool.
    # If enabled, the arg pool uses wrapper_policy.macro_arg_wrapper_policy to
    # decide which cheap trig wrappers to include (e.g. sin^2 only when trig hints exist).
    macro_arg_use_wrapper_policy: bool = True
    macro_arg_trig_enable: bool = True
    macro_arg_trig_squares_enable: bool = True
    macro_arg_trig_sq_requires_hint: bool = True
    macro_arg_trig_max_bases: int = 16

    # Optional: filter the macro library to avoid redundant proposals.
    # If non-empty, only macros with names in macro_allow_names are considered.
    macro_allow_names: list[str] = field(default_factory=list)
    # Macros with names in macro_deny_names are skipped.
    macro_deny_names: list[str] = field(default_factory=list)
    # Disable macro templates that overlap with existing hint-driven trig families
    # (e.g. sinc/sin_ratio) by default. Set False to re-enable.
    macro_disable_trig_duplicates: bool = True

    # ------------------------------------------------------------------
    # Stage B: compound-function macros (tier-2: macro + simple outer algebra)
    # ------------------------------------------------------------------
    # If enabled, also try candidates of the form
    #   y ≈ a * [ factor(x) * macro(args) ] + b
    # or
    #   y ≈ a * [ macro(args) / factor(x) ] + b
    # where factor(x) is a small monomial/ratio built from the same input axes.
    macro_tier2_enable: bool = True
    macro_tier2_try_mul: bool = True
    macro_tier2_try_div: bool = True
    # Max number of base factors used to build composite factors.
    macro_tier2_max_base_factors: int = 24
    # Max number of factor expressions retained after cost sorting + dedup.
    macro_tier2_max_factor_exprs: int = 96
    # Keep at most this many tier-2 screened hits per macro family.
    macro_tier2_max_per_macro: int = 3
    # Small epsilon for division gating (avoid 1/0 blowups during screening).
    macro_tier2_div_eps: float = 1.0e-12

    # ------------------------------------------------------------------
    # Stage B: compound-function macros (tier-3: small linear combinations)
    # ------------------------------------------------------------------
    # Try small *additive* linear combinations of a few analytic features,
    # screened via least-squares on a cached batch:
    #   y ≈ a1*phi1(x) + a2*phi2(x) + b
    # This is high-ROI for Feynman-style expressions that are sums of two
    # structured terms, e.g.  p(x) + q(x)*sinc(phase)^2.
    macro_tier3_enable: bool = True
    macro_tier3_try_pairs: bool = True
    # Optional: allow 3-feature combos (more expensive). Default off.
    macro_tier3_try_triples: bool = False
    # Max number of macro-derived features used as seeds.
    macro_tier3_max_macro_features: int = 12
    # Max number of auxiliary (prefactor) features considered.
    macro_tier3_max_aux_features: int = 24
    # Keep at most this many tier-3 candidates per target leaf.
    macro_tier3_max_candidates: int = 6
    # Ridge regularisation for small least-squares problems.
    macro_tier3_ridge: float = 1.0e-8
    # Whether tier-3 combos may use two macro-features (in addition to macro+aux).
    macro_tier3_allow_macro_macro: bool = True
    # Whether tier-3 combos may use aux+aux (no macro). Usually redundant with
    # other Stage-B rules; keep off by default.
    macro_tier3_allow_aux_aux: bool = False

    # If a tier-3 combo contains numerically redundant terms, prefer the smallest
    # strict subset whose screening R^2 matches the full combo.
    macro_tier3_subset_r2_tol: float = 1.0e-12

    # Prune features whose contribution is negligible vs the largest term,
    # using the screening proxy |c| * RMS(feature).
    macro_tier3_prune_rel_contrib: float = 1.0e-10


    # ------------------------------------------------------------------
    # Stage B: compound-function macros (tier-3b: shared prefactor)
    # ------------------------------------------------------------------
    # Try candidates of the form:
    #   y ≈ p(x) * (a*macro(args) + b)
    # by screening (y/p) as an affine function of a macro core, then
    # reconstructing the multiplicative form. This is very helpful when
    # the true expression is a prefactor times a structured motif.
    macro_tier3_shared_prefactor_enable: bool = True
    macro_tier3_shared_prefactor_max_prefactors: int = 24
    macro_tier3_shared_prefactor_max_pos_args: int = 12
    macro_tier3_shared_prefactor_max_macro_exprs: int = 256
    macro_tier3_shared_prefactor_max_per_prefactor: int = 2
    macro_tier3_shared_prefactor_max_candidates: int = 6
    macro_tier3_shared_prefactor_div_eps: float = 1.0e-12

    # Optional: shared-prefactor with an additive residual constant:
    #   y ≈ p(x) * (a*macro(args) + b) + c
    # Screened as: y ≈ a*(p*m) + b*p + c.
    macro_tier3_shared_prefactor_residual_enable: bool = True
    macro_tier3_shared_prefactor_residual_max_per_prefactor: int = 2
    macro_tier3_shared_prefactor_residual_max_candidates: int = 6
    macro_tier3_shared_prefactor_residual_ridge: float = 1.0e-8

    # ------------------------------------------------------------------
    # Stage B: local leaf y-transforms (sub-expression wrappers)
    # ------------------------------------------------------------------
    # Some 1D leaves become much simpler after a small output transform.
    # Example: u(z)=1/(1+cos(az+b)) becomes v(z)=recip(u)=1+cos(az+b).
    # We can fit v with existing templates (trig, poly, ...) and then wrap
    # back with the inverse transform to model u.
    stageB_leaf_transforms_enable: bool = True
    # Names follow sr_search.features.probe_output_transforms conventions.
    stageB_leaf_transforms: list[str] = field(
        default_factory=lambda: ["identity", "log", "recip", "square", "sqrt"]
    )
    # ------------------------------------------------------------------
    # Final pruning of small additive terms (post-Stage B)
    # ------------------------------------------------------------------
    # After Stage B completes with a fully analytical expression (0 NN atoms),
    # try dropping additive terms whose RMS contribution is negligible.
    prune_final_enable: bool = True
    prune_rel_threshold: float = 1e-3   # RMS(term)/RMS(y_pred) below this -> flagged
    prune_loss_tolerance: float = 0.01  # max allowed fractional MSE increase
    prune_refit_epochs: int = 500

    # ------------------------------------------------------------------
    # Per-parameter pruning (individual coefficients in poly/ratpoly leaves)
    # ------------------------------------------------------------------
    # After Stage B completes with a fully analytical expression (0 NN atoms),
    # iteratively zero individual polynomial/rational-polynomial coefficients,
    # one at a time (least significant first), refitting after each removal.
    # Acceptance is based on AIC (n*ln(MSE) + 2*k).
    prune_param_enable: bool = True
    prune_param_aic_tolerance: float = 2.0   # accept if AIC_new <= AIC_old + this
    prune_param_refit_epochs: int = 300      # short LM refit after zeroing
    prune_param_max_pruned: int = 20         # safety cap
    prune_param_protect_denominator_constant: bool = True  # never zero den constant term
    # Optional iterative prune <-> simplify rounds.
    # This is OFF by default to preserve historical Stage-B behaviour.
    # When enabled, Stage-B can run:
    #   prune passes -> exact/noiseless SymPy simplification -> prune passes -> ...
    prune_sympy_iter_enable: bool = False
    prune_sympy_iter_max_rounds: int = 3
    # LM epochs used to rebuild/refit the simplified AST candidate.
    prune_sympy_refit_epochs: int = 200
    # "Noiseless" acceptance gate for simplified candidates.
    # Candidate is accepted only if:
    #   1) symbolic equivalence check passes exactly, and
    #   2) validation MSE does not increase beyond these tiny tolerances.
    prune_sympy_noiseless_rel_tol: float = 1.0e-12
    prune_sympy_noiseless_abs_tol: float = 1.0e-14

    # Degrees to try for polynomial inner fits in leaf-transform screening (v ≈ poly(z)).
    stageB_leaf_transforms_poly_degrees: list[int] = field(default_factory=lambda: [1, 2, 3])
    # Max number of (x,u) points used for screening fits per target leaf.
    stageB_leaf_transforms_max_points: int = 5000
    # Minimum number of valid points (after domain masking) required.
    stageB_leaf_transforms_min_points: int = 300
    # Minimum fraction of points that must be in-domain for a transform.
    stageB_leaf_transforms_min_domain_ok_frac: float = 0.98
    # Screening threshold on relative RMS error after wrapping back to u-space.
    stageB_leaf_transforms_screen_rms_rel_max: float = 0.25
    # Limit total number of leaf-transform hits (best-first) per target leaf.
    stageB_leaf_transforms_max_candidates: int = 3


@dataclass
class ModelHyperparams:
    """
    Hyper-parameters describing the neural model and segmented adaptor sizes.
    """

    model_base_name: str = "G_Model"
    block_size_target: int = None
    Gmodel_scale: float = 0.1
    num_segments_max: int = 48
    num_segments_min: int = 16
    num_segments_quickscan: int = 1
    model_size_target: int = 1000
    nparam_max: int | None = 4000
    Nout_size: int = 1
    double_precision: bool = True
    repeatable_runs: bool = True


@dataclass
class DataHyperparams:
    """
    Hyper-parameters for dataset construction and batching.
    """

    batch_size: int = 2000
    ndata_select: int = 2000
    ndata_select_val: int = 2000
    # Deterministic disjoint data block index.  data_slice=0 intentionally
    # preserves the historical "start at row 0" loader path exactly.
    data_slice: int = 0
    # How to split selected rows into train/validation subsets.
    # "contiguous" preserves historical behavior; "interleaved" spreads both
    # subsets across the selected domain while keeping them disjoint.
    data_split_strategy: str = "contiguous"


@dataclass
class SearchHyperparams:
    """
    Hyper-parameters for separability / symbolic search.
    """

    ntrial: int = 1
    acceptance_criterion: float = 10.0
    loss_acceptable: float = 1.0e-3  # Match LMHyperparams default
    precision_derivs_d2y: float = 0.001
    num_segments_map: dict = field(default_factory=lambda: {False: 32, True: 16})
    # Maximum allowed degradation factor relative to the current model's val_loss
    # when accepting ANY separability candidate.
    max_worsening_factor: float = 100.0
    # Absolute floor (in the same base units as loss_target/loss_acceptable, i.e. MAD^2)
    # on the tightened accept_threshold used by the worsening safeguard. This prevents
    # over-tightening when the current fit is extremely good, while still blocking
    # large regressions when proposing new structure.
    worsening_floor: float = 1.0e-6
    # If True, skip single-layer trial and use dual-layer from the start
    force_dual_layer: bool = False

    # ------------------------------------------------------------------
    # Overlap truth screening (Stage A overlap-only candidates)
    # ------------------------------------------------------------------
    # Overlapping separability should be judged in function space, not by
    # whether a particular gauge penalty survives LM continuation. We test
    # the split identity directly on batches from the current parent leaf.
    overlap_truth_screen_enable: bool = True
    overlap_truth_max_batches: int = 4
    overlap_truth_add_rms_factor: float = 5.0
    overlap_truth_mul_rms_factor: float = 5.0
    overlap_truth_anchor_rel_eps: float = 1.0e-8

    # Stage A x-preconditioning: optional retries in transformed x space when
    # the separability loop stalls before fully separating the expression.
    x_precondition_enable: bool = True
    # Maximum number of extra Stage A passes (0 disables; 1 enables reciprocal-only; 4+ enables reciprocal + multiple trig omega candidates).
    x_precondition_max_extra_passes: int = 4
    # Tolerance for snapping singleton scaling exponents to -1 or -2 when proposing reciprocal transforms.
    x_precondition_scaling_tol: float = 0.35

    # Optional generalized-symmetry Stage-A configuration. Kept as one
    # object so the experimental GS switchboard does not leak dozens of fields
    # into the mature search configuration.
    gs_config: object | None = None

    # ------------------------------------------------------------------
    # Compound-variable detection and prioritization
    # ------------------------------------------------------------------
    # Stage-0 phase-coordinate prescan.
    #
    # This is a hint generator, not an acceptance mechanism.  It enumerates
    # sparse unit-valid dimensionless carriers z(X), scores whether y is
    # predictable from omega*z modulo 2*pi by held-out Fourier regression, and
    # exposes frequency seeds to later Stage-B trig closures.
    phase_prescan_enabled: bool = True
    phase_prescan_sample_size: int = 4096
    phase_prescan_max_support: int = 3
    phase_prescan_max_candidates: int = 96
    phase_prescan_max_candidates_per_support: int = 32
    phase_prescan_max_exp_l1: float = 4.0
    phase_prescan_min_domain_frac: float = 0.98
    phase_prescan_log_top_k: int = 6
    phase_context_enabled: bool = True
    phase_context_max_features: int = 8
    phase_context_log_top_k: int = 6

    # If enabled, allow replacing an NN leaf f(x_S) with f(z, extras...) where
    # z = \prod_i x_i^{a_i} is a detected monomial compound. This is especially
    # useful for common physics forms with dimensionless groups.
    enable_compound_detection: bool = True
    # SVD rank-1 tolerance in the monomial test (\sigma_2/\sigma_1 <= threshold).
    # Note: smaller is stricter. The CLI flag is historically named
    # --compound_threshold.
    compound_threshold: float = 0.1
    compound_max_vars: int = 7
    compound_max_exponent: int = 5
    # Confidence gate for compound detection.
    # Confidence is computed as (1 - \sigma_2/\sigma_1) * match_score.
    compound_confidence_gate: float = 0.85
    # Maximum batches to gather when estimating gradients for compound detection.
    compound_max_batches: int = 4
    # Boost segments for 1D compound inputs (high-frequency functions need more capacity)
    compound_1d_num_segments: int = 32

    # Verbose separability diagnostics
    verbose_separabilities: bool = False

    # Deprecated no-op in unified y-search mode.
    # Kept for CLI/checkpoint backward compatibility only.
    outer_peel_autorun: bool = True

    # Virtual y-transform screening (Stage A, identity-first):
    # rank candidate transforms using chain-rule probes, then full-fit top-K.
    ysearch_enable: bool = True
    ysearch_depth: int = 1
    ysearch_beam: int = 3
    ysearch_expand_k: int = 3
    ysearch_portfolio_margin_decades: float = 0.25
    ysearch_portfolio_max_k: int = 3
    ysearch_min_valid_frac: float = 0.99
    ysearch_confirm_improve_ratio: float = 0.3
    ysearch_trigger_trig_affine_conf: float = 0.90
    ysearch_trigger_sep_min: float = 0.80
    ysearch_trigger_sep_delta: float = 0.25
    ysearch_trigger_split_score: float = 0.90
    ysearch_trigger_split_margin: float = 0.15
    ysearch_max_virtual_deriv: float = 1.0e6
    # Exact/full-compound outer certificate: φ(y) ≈ a*z + b.
    ysearch_outer_affine_confirm_rms_rel: float = 1.0e-6
    ysearch_outer_affine_min_domain_frac: float = 0.995
    ysearch_outer_affine_min_points: int = 256
    ysearch_outer_affine_max_points: int = 2048
    # Hard compute cap for Stage-A evaluations in y-search (0 => unbounded).
    ysearch_max_state_evals: int = 0
    # Optional split-recursion budget (0 => disabled).
    ysearch_max_recursive_branches: int = 0
    ysearch_max_split_plans_per_state: int = 1

    # Stage A <-> B feedback loop safety cap.
    # Loop until convergence (expression unchanged) or this cap is hit.
    max_ab_iters: int = 5
    # Stage-A loop-pass safety cap. 0 means unbounded. Scout proposer runs use
    # this to stop at the boundary where Stage A would otherwise restart.
    stageA_max_passes: int = 0

    # Additional compound families beyond monomial (Stage A)
    # ------------------------------------------------------------------
    # Linear compounds: z = \sum_i c_i x_i (detectable via rank-1 gradient test)
    compound_try_linear: bool = True
    compound_linear_max_coeff: int = 2
    # Radial compounds: z = \sum_i x_i^2 (and optionally sqrt(z))
    compound_try_radial: bool = True
    compound_radial_max_group_size: int = 3
    compound_radial_cos_threshold: float = 0.95
    compound_radial_try_sqrt: bool = True

    # Preferred-origin / translation hints as compound proposals.
    # Propose z = x_j - x0 for axes where df/dx_j is approximately linear in x_j
    # (suggesting translation structure around a preferred origin x0).
    compound_try_shift: bool = True
    compound_shift_min_r2: float = 0.85
    compound_shift_min_abs_slope: float = 1.0e-6
    compound_shift_require_in_range: bool = True
    compound_shift_max_axes_per_atom: int = 2
    # Retained-axis wrappers keep a raw input beside a compound coordinate
    # that already contains it, e.g. NN[x1/x0, x0].  They are useful
    # homogeneity hints, but must earn acceptance with an overlap-aware
    # certificate: the retained raw axis must split as a simple power/constant
    # factor, not as an arbitrary compensating NN.
    compound_try_retained_axis_wrappers: bool = True

    # Stage-A additive shared-response proposal:
    #   sum_i leaf_i(x) -> (sum_j a_j P_j(x)) * NN[pi(x)]
    # for direct additive NN siblings with a common unit-certified
    # dimensionless group pi and visible monomial prefactors P_j.  This is a
    # proposal lane only; normal validation, CoE ranking, and full-refit
    # confirmation still decide whether the transaction becomes reference
    # history.
    additive_shared_response_enable: bool = True
    additive_shared_response_max_siblings: int = 2
    additive_shared_response_max_pi_groups: int = 2
    additive_shared_response_max_prefactors_per_sibling: int = 3
    additive_shared_response_max_gauge_shift: int = 2
    additive_shared_response_max_abs_power: int = 3
    additive_shared_response_max_prefactor_support: int = 4
    additive_shared_response_screen_bins: int = 32
    additive_shared_response_min_ok_frac: float = 0.98
    additive_shared_response_min_rank_energy: float = 0.80
    additive_shared_response_min_r2_rank1: float = 0.60
    additive_shared_response_max_candidates: int = 4

    # Efficiency controls for trying detected compound proposals
    # ------------------------------------------------------------------
    # Cap how many greedy compound proposals are trained per atom. Stage A also
    # preserves tiny backup lanes for clean monomial products and same-ranked
    # effective-arity-2 proposals so useful conservative compounds cannot be
    # crowded out by high-arity variants.
    compound_max_proposals_to_try: int = 6
    # Cap how many z-wrapper variants (raw/rational/trig) are trained per proposal.
    compound_max_variants_to_try: int = 3

    # Cheap screening of 1D compound candidates before LM training.
    # Only applied when the proposed compound has no extra variables (pure 1D).
    compound_variant_screen_enable: bool = True
    compound_variant_screen_bins: int = 64
    # Skip variants whose screening score falls below this.
    compound_variant_screen_gate: float = 0.15

    # ------------------------------------------------------------------
    # Iso-z residual dependency check (Stage A, monomial compounds)
    # ------------------------------------------------------------------
    # For 1D monomial compound proposals (all vars folded into z), a common
    # failure mode is accepting a compound that is *approximately* correct on
    # the sampled domain but still has residual dependence on the raw vars.
    #
    # We probe invariance by perturbing the raw variables along iso-z manifolds
    # (holding z exactly constant) and measuring how much the (optionally
    # prefactor-peeled) NN leaf output varies.
    compound_iso_z_enable: bool = True
    # Reject if the chosen quantile of within-point std / overall std exceeds this.
    compound_iso_z_threshold: float = 0.03
    # In noisy runs, a full 1D monomial whose observed iso-z residual is above
    # the clean certificate threshold can still be retained as an uncertified
    # proposal if the excess residual is compatible with the known label noise.
    compound_iso_z_noise_mult: float = 2.0
    compound_iso_z_noise_cap: float = 0.25
    compound_iso_z_struct_margin: float = 0.01
    compound_iso_z_noisy_min_confidence: float = 0.75
    # Robust statistic to use across sampled points (0.5=median, 0.9=90th percentile).
    compound_iso_z_quantile: float = 0.90
    # Number of base points sampled from the cached training batch.
    compound_iso_z_n_sample: int = 300
    # Number of perturbations per base point (log-spaced in t).
    compound_iso_z_n_perturb: int = 10
    # Log-scale perturbation range: t in [exp(-r), exp(+r)]. (0.3 => ~[0.74, 1.35])
    compound_iso_z_log_t_range: float = 0.3
    # Minimum number of valid base points required per nullspace direction.
    compound_iso_z_min_valid: int = 64
    # Maximum number of points used for screening/pretraining.
    compound_pretrain_max_points: int | None = 5000

    # ------------------------------------------------------------------
    # Early compound detection from scaling exponents
    # ------------------------------------------------------------------
    # If all single-variable scaling exponents are close to integers with low scatter,
    # immediately propose z = Π xᵢ^kᵢ and skip expensive feature discovery.
    # Maximum rel_std for a scaling exponent to be considered "clean"
    early_compound_rel_std: float = 0.05
    # Maximum |k - round(k)| for exponent to be considered "integer"
    early_compound_k_int: float = 0.15
    # Noise-only soft monomial proposal lane.  The deterministic early/compound
    # detectors remain unchanged; when an explicit noise floor is active, this
    # lane gives a tiny number of low-complexity near-integer monomial maps a
    # ballot slot and leaves acceptance to normal validation/CoE.
    noisy_soft_monomial_rel_std: float = 0.12
    noisy_soft_monomial_group_rel_std: float = 0.12
    noisy_soft_monomial_k_int: float = 0.25
    noisy_soft_monomial_group_resid: float = 0.35
    noisy_soft_monomial_max_abs_power: int = 6
    noisy_soft_monomial_max_l1: int = 10
    noisy_soft_monomial_max_support: int = 5
    noisy_soft_monomial_max_candidates_per_atom: int = 2

    # ------------------------------------------------------------------
    # Compound z-wrapper candidates
    # ------------------------------------------------------------------
    # Whether to try trig wrappers (sin/cos) around z in compound proposals.
    compound_try_trig_wrappers: bool = True
    # Whether to try simple rational/algebraic wrappers around z in compound proposals.
    compound_try_rational_wrappers: bool = True
    # If True, allow rational wrappers for radial compounds (e.g. 1/(1+r^2)).
    # Monomial compounds can still restrict to ratio-like patterns via
    # `compound_rational_only_if_ratio_like`.
    compound_rational_allow_for_radial: bool = True
    # If True, only try rational wrappers when the compound exponents include at least
    # one positive and one negative exponent (i.e. ratio-like z).
    compound_rational_only_if_ratio_like: bool = True

    # Lightweight power/shape wrappers around z (kind-aware).
    compound_try_square_wrappers: bool = True
    compound_try_abs_wrappers: bool = True
    # Generic wrapper preference used for non-shape wrappers (e.g. rational).
    # Smaller => a wrapper must improve more before replacing raw z.
    compound_wrapper_prefer_factor: float = 0.1
    # How hard should sqrt/square/abs have to "beat" raw z when selecting the best
    # accepted wrapper variant? Smaller => stricter. (0.5 => must be 2x better.)
    compound_sqrt_wrapper_prefer_factor: float = 0.5
    compound_square_wrapper_prefer_factor: float = 0.3
    compound_abs_wrapper_prefer_factor: float = 0.3
    # Trig wrappers must beat the best non-trig variant by this factor to be selected.
    # (Default 0.01 => trig must be at least 100x better.)
    compound_trig_wrapper_prefer_factor: float = 0.01

    # (compound_*_wrapper_prefer_factor are used when comparing a wrapped z against
    # a raw z candidate; smaller => must improve more to be selected.)

    # ------------------------------------------------------------------
    # Stage A dynamic choice: disjoint separability vs long composite variables
    # ------------------------------------------------------------------
    # Controls the move order in Stage A when both separability and compound
    # structure are plausible.
    #
    # Values:
    #   - "sep_first": try separability before compounds (no candidate filtering)
    #   - "dynamic_singleton": if a *very clean* singleton disjoint split is detected cheaply,
    #                           try that (and only that) first; otherwise do compound-first.
    #   - "dynamic" / "dynamic_disjoint": if a *very clean* disjoint split (any size) is detected cheaply,
    #                                     try that (frugally) first; otherwise do compound-first.
    stageA_move_policy: str = "dynamic"

    # How strict the quickscan must be to override compound-first.
    # We require metric <= factor * precision.
    # Smaller => only override on extremely clean splits.
    stageA_singleton_sep_metric_factor: float = 0.30
    stageA_non_singleton_sep_metric_factor: float = 0.30

    # How many training batches to use for the disjoint-separability quickscan.
    # (1 is usually enough; this is a cheap heuristic, Stage B will validate.)
    stageA_disjoint_sep_quickscan_batches: int = 1

    # When Stage A decides to try disjoint separability first, how many disjoint
    # candidates to actually train per atom (1 = frugal extreme).
    stageA_disjoint_sep_max_candidates: int = 1

    # ------------------------------------------------------------------
    # Oracle-based scaling probe
    # ------------------------------------------------------------------
    # Direct evaluation of f(λ·x_j, x_rest) / f(x_j, x_rest) to confirm
    # scaling degrees discovered by the gradient-based Euler test.
    oracle_scaling_rel_std: float = 0.08


if TYPE_CHECKING:  # pragma: no cover
    from .factorized_search.config import FactorizedSearchConfig


def __getattr__(name: str):
    if name == "FactorizedSearchConfig":
        from .factorized_search.config import FactorizedSearchConfig

        return FactorizedSearchConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LMHyperparams",
    "FactorizedSearchConfig",
]
