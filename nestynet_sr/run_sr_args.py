# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Command-line argument parsing for ``nestynet_sr.run_SR``."""

from __future__ import annotations

import argparse

from nestynet_sr.sr_expr_ir.config import add_expr_ir_cli_args
from nestynet_sr.sr_search.factorized_search.config import REFINE_OPTIMIZER_NAMES, REFINE_PROFILE_NAMES
from nestynet_sr.sr_search.factorized_search.research_profiles import RESEARCH_PROFILE_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None, help="Path to input CSV file")
    parser.add_argument(
        "--filepaths",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of input CSV files (multi-dataset mode). "
        "If provided, Stage A is run on the first file and Stage B "
        "fits a shared expression structure across all files with dataset-specific constants.",
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from a previously saved checkpoint instead of re-running Stage A",
    )

    parser.add_argument(
        "--load_expressions",
        type=str,
        default=None,
        help="Path to load initial expressions from pickle file",
    )
    parser.add_argument(
        "--stageA_continuation_seed",
        action="store_true",
        help=(
            "Internal CoE scout mode: treat --load_expressions as the current "
            "reference Stage-B AST and run a bounded feedback Stage-A pass from it."
        ),
    )
    parser.add_argument(
        "--stageA_continuation_y_op",
        type=str,
        default="identity",
        help="Internal CoE scout mode: y-transform name for the continuation seed.",
    )
    parser.add_argument(
        "--stageA_continuation_fit_link",
        type=str,
        default=None,
        help=(
            "Internal CoE scout mode: pinned fit-link name for the continuation seed. "
            "Initial/reference runs should not set this."
        ),
    )
    parser.add_argument(
        "--stageA_continuation_fit_link_scale",
        type=float,
        default=1.0,
        help="Internal CoE scout mode: pinned fit-link scale for the continuation seed.",
    )
    parser.add_argument(
        "--no_stageA_auto_fit_link",
        dest="stageA_auto_fit_link",
        action="store_false",
        help="Disable opportunistic Stage-A asinh fit-link retries.",
    )
    parser.set_defaults(stageA_auto_fit_link=True)

    parser.add_argument(
        "--no_stageB", dest="stageB", action="store_false", help="disable Stage B refinement"
    )

    parser.add_argument(
        "--factorized-search", dest="use_factorized_search", action="store_true", default=None,
        help="force enable factorized symbolic search Stage B rule (default: on when units enforced, off with --ignore_units)",
    )
    parser.add_argument(
        "--no-factorized-search", dest="use_factorized_search", action="store_false",
        help="force disable factorized symbolic search Stage B rule",
    )

    parser.add_argument(
        "--no_brute_force", action="store_true",
        help="disable factorized symbolic search brute-force enumeration phase (keep mutation search only)",
    )
    parser.add_argument(
        "--refine-skeleton",
        dest="use_refine_skeleton",
        action="store_true",
        default=None,
        help="enable scheduled continuous skeleton refinement (run_SR default: on)",
    )
    parser.add_argument(
        "--no-refine-skeleton",
        dest="use_refine_skeleton",
        action="store_false",
        help="disable continuous skeleton refinement inner trig-scale refinement",
    )

    # factorized symbolic search scoring augmentation: multi-term linear head on residual
    parser.add_argument(
        "--factorized-score-head",
        dest="factorized_score_head",
        action="store_true",
        default=None,
        help="enable factorized symbolic search multi-term linear head scoring (default: on)",
    )
    parser.add_argument(
        "--no-factorized-score-head",
        dest="factorized_score_head",
        action="store_false",
        help="disable factorized symbolic search multi-term linear head scoring",
    )
    parser.add_argument(
        "--factorized-score-head-omp",
        dest="factorized_score_head_omp",
        action="store_true",
        default=None,
        help="enable OMP selection of extra head terms from the pool (default: off)",
    )
    parser.add_argument(
        "--no-factorized-score-head-omp",
        dest="factorized_score_head_omp",
        action="store_false",
        help="disable OMP selection of extra head terms from the pool",
    )
    parser.add_argument(
        "--factorized-score-head-omp-terms",
        type=int,
        default=None,
        help="max extra pool terms in scoring head OMP (default: config)",
    )
    parser.add_argument(
        "--factorized-score-head-ridge",
        type=float,
        default=None,
        help="ridge regularisation for scoring-head fit (default: config)",
    )
    parser.add_argument(
        "--factorized-score-head-min-rel-improve",
        type=float,
        default=None,
        help="min relative probe-MSE improvement to keep scoring head (default: config)",
    )

    parser.add_argument(
        "--refine-lbfgs-steps",
        type=int,
        default=None,
        help="continuous skeleton refinement LBFGS steps per inner refinement (default: config)",
    )
    parser.add_argument(
        "--refine-fit-subset",
        type=int,
        default=None,
        help="continuous skeleton refinement fit subset size for inner refinement (default: config)",
    )
    parser.add_argument(
        "--refine-profile",
        default=None,
        metavar="PROFILE",
        help=f"named continuous skeleton refinement runtime profile ({', '.join(REFINE_PROFILE_NAMES)}; aliases accepted)",
    )
    parser.add_argument(
        "--refine-num-restarts",
        type=int,
        default=None,
        help="continuous skeleton refinement restart count for hard-parameter optimisation (default: config)",
    )
    parser.add_argument(
        "--refine-optimizer",
        choices=list(REFINE_OPTIMIZER_NAMES),
        default=None,
        help="continuous skeleton refinement optimizer policy",
    )
    parser.add_argument(
        "--refine-lbfgs-escalate-improve-factor",
        type=float,
        default=None,
        help="minimum grid improvement factor before grid_then_lbfgs escalates to LBFGS",
    )
    parser.add_argument(
        "--refine-max-variants",
        type=int,
        default=None,
        help="max parameterized variants per expression in continuous skeleton refinement (default: config)",
    )
    parser.add_argument(
        "--refine-max-params",
        type=int,
        default=None,
        help="max number of hard parameters in continuous skeleton refinement (default: config)",
    )
    parser.add_argument(
        "--refine-linear-combo",
        dest="use_refine_linear_combo",
        action="store_true",
        default=None,
        help="enable continuous skeleton refinement additive multi-basis linear elimination (Phase 3)",
    )
    parser.add_argument(
        "--no-refine-linear-combo",
        dest="use_refine_linear_combo",
        action="store_false",
        help="disable continuous skeleton refinement additive multi-basis linear elimination",
    )
    parser.add_argument(
        "--refine-linear-terms-max",
        type=int,
        default=None,
        help="max additive terms for continuous skeleton refinement linear-basis elimination (default: config)",
    )
    parser.add_argument(
        "--refine-linear-prune-rel",
        type=float,
        default=None,
        help="relative contribution prune threshold for continuous skeleton refinement linear terms (default: config)",
    )
    parser.add_argument(
        "--refine-gate-best-factor",
        type=float,
        default=None,
        help="run continuous skeleton refinement only when baseline MSE <= factor * current_best_mse (default: config)",
    )
    parser.add_argument(
        "--refine-max-trials",
        type=int,
        default=None,
        help="global max number of continuous skeleton refinement inner refinements (default: config)",
    )
    parser.add_argument(
        "--refine-trials-per-brute-depth",
        type=int,
        default=None,
        help="max continuous skeleton refinements per brute-force depth (Phase 4 gate, default: config)",
    )
    parser.add_argument(
        "--refine-trials-per-mutation-window",
        type=int,
        default=None,
        help="max continuous skeleton refinements per mutation window (Phase 4 gate, default: config)",
    )
    parser.add_argument(
        "--refine-mutation-window",
        type=int,
        default=None,
        help="mutation-window size for continuous skeleton refinement refine budgeting (default: config)",
    )
    parser.add_argument(
        "--refine-safe-eps",
        type=float,
        default=None,
        help="smooth domain floor epsilon for continuous skeleton refinement safe log/sqrt/div eval (default: config)",
    )
    parser.add_argument(
        "--refine-safe-penalty-weight",
        type=float,
        default=None,
        help="penalty weight for continuous skeleton refinement domain corrections in inner closure (default: config)",
    )
    parser.add_argument(
        "--refine-safe-exp-clip",
        type=float,
        default=None,
        help="soft exp input clip used in continuous skeleton refinement safe eval (default: config)",
    )
    parser.add_argument(
        "--refine-theta-l2",
        type=float,
        default=None,
        help="L2 regularization on continuous skeleton refinement hard-parameter logits (default: config)",
    )
    parser.add_argument(
        "--refine-init-log-min",
        type=float,
        default=None,
        help="min log-scale used for continuous skeleton refinement restarts (default: config)",
    )
    parser.add_argument(
        "--refine-init-log-max",
        type=float,
        default=None,
        help="max log-scale used for continuous skeleton refinement restarts (default: config)",
    )

    # --- Final pruning of small additive terms ---
    parser.add_argument(
        "--no_prune_final",
        dest="prune_final",
        action="store_false",
        default=True,
        help="disable final pruning of small additive terms after Stage B",
    )
    parser.add_argument(
        "--prune_rel_threshold",
        type=float,
        default=None,
        help="RMS(term)/RMS(y_pred) threshold for flagging small additive terms (default: 1e-3)",
    )
    parser.add_argument(
        "--prune_loss_tolerance",
        type=float,
        default=None,
        help="max allowed fractional MSE increase when pruning (default: 0.01)",
    )

    # --- Per-parameter pruning of polynomial coefficients ---
    parser.add_argument(
        "--no_prune_param",
        dest="prune_param",
        action="store_false",
        default=True,
        help="disable per-parameter pruning of polynomial coefficients",
    )
    parser.add_argument(
        "--prune_param_aic_tolerance",
        type=float,
        default=None,
        help="AIC tolerance for per-parameter pruning (default: 2.0)",
    )

    parser.add_argument(
        "--no_stageA_separabilities",
        dest="stageA_separabilities",
        action="store_false",
        help="disable Stage A separability search (use initial NN model as-is)",
    )

    parser.add_argument(
        "--stageB_max_outer_iters",
        type=int,
        default=None,
        help="Maximum number of Stage B refinement iterations (default: 30)",
    )

    parser.add_argument(
        "--max_backtracks",
        type=int,
        default=3,
        help="Max backtrack attempts in Stage B (0 to disable, default: 3)",
    )

    parser.add_argument(
        "--max_ab_iters",
        type=int,
        default=None,
        help="Max Stage A<->B feedback loop iterations (default: 5). Set 1 to disable.",
    )
    parser.add_argument(
        "--stageA_max_passes",
        type=int,
        default=None,
        help=(
            "Maximum Stage-A loop passes before stopping at the restart boundary. "
            "0/unset means unbounded; bounded CoE scouts default to 1."
        ),
    )

    parser.add_argument(
        "--stageB_epochs",
        type=int,
        default=None,
        help="Maximum LM epochs for each Stage B candidate (default: 2000)",
    )

    parser.add_argument(
        "--stageB_score_tol",
        type=float,
        default=None,
        help="Minimum improvement required to accept Stage B rewrite (default: 0.0)",
    )
    parser.add_argument(
        "--stageB_overcap_fallback",
        action="store_true",
        help=(
            "Enable a noisy-data Stage B fallback: when no acceptance noise floor is set "
            "and the current best Stage B loss is still above the Stage B hard cap, "
            "ignore that cap for near-loss-neutral simplifications. Default off so "
            "noiseless runs keep the original behaviour."
        ),
    )
    parser.add_argument(
        "--stageB_polish",
        dest="stageB_polish",
        action="store_true",
        default=True,
        help="Polish fully analytic accepted Stage B states through the normal Stage B policy (default: on).",
    )
    parser.add_argument(
        "--no_stageB_polish",
        dest="stageB_polish",
        action="store_false",
        help="Disable accepted-step Stage B polishing.",
    )
    parser.add_argument(
        "--stageB_polish_max_candidates",
        type=int,
        default=32,
        help="Maximum algebraic cleanup candidates for accepted-step Stage B polish (default: 32).",
    )
    parser.add_argument(
        "--stageB_polish_subtrees",
        dest="stageB_polish_subtrees",
        action="store_true",
        default=True,
        help="Enable accepted-step polishing and commits for newly analytic Stage B subtrees (default: on).",
    )
    parser.add_argument(
        "--no_stageB_polish_subtrees",
        dest="stageB_polish_subtrees",
        action="store_false",
        help="Disable accepted-step polishing for newly analytic Stage B subtrees.",
    )
    parser.add_argument(
        "--stageB_polish_max_subtrees",
        type=int,
        default=8,
        help="Maximum newly analytic subtrees audited per accepted Stage B rewrite (default: 8).",
    )
    parser.add_argument(
        "--stageB_polish_max_seconds",
        type=float,
        default=300.0,
        help="Wall-clock cap for the guarded Stage-B polish worker (default: 300)",
    )
    parser.add_argument(
        "--stageB_polish_mem_fraction",
        type=float,
        default=0.20,
        help="Temporary RAM fraction cap for the guarded Stage-B polish worker (default: 0.20)",
    )
    parser.add_argument(
        "--no_stageB_polish_subprocess",
        dest="stageB_polish_subprocess",
        action="store_false",
        default=True,
        help="Run accepted-step Stage-B polish candidate generation in-process instead of in the guarded worker",
    )
    parser.add_argument(
        "--noise_sigma_frac_y_rms",
        type=float,
        default=None,
        help=(
            "Known homoscedastic additive y-noise, expressed as sigma_y / RMS(y_full). "
            "The RMS is computed from the full loaded dataset, not batches."
        ),
    )
    parser.add_argument(
        "--noise_floor_mc_samples",
        type=int,
        default=8,
        help=(
            "Monte-Carlo samples used to estimate transform-aware irreducible loss floors "
            "from the full raw-y dataset (default: 8)."
        ),
    )

    # -----------------------------------------------------------------
    # DE discovery (make DE a first-class SR output)
    # -----------------------------------------------------------------
    parser.add_argument(
        "--discover_de",
        action="store_true",
        help=(
            "Discover an implicit DE residual from the trained surrogate(s) and run a Stage-B-style fit/simplify pass on it. "
            "Outputs results/<stem>_de.* alongside the usual SR outputs."
        ),
    )
    parser.add_argument(
        "--de_y_space",
        type=str,
        default="identity",
        choices=["identity", "final"],
        help=(
            "Which output space to use for DE discovery: 'identity' uses y, 'final' uses the Stage-A chosen φ(y). "
            "(Default: identity)"
        ),
    )
    parser.add_argument(
        "--de_order_candidates",
        type=str,
        default="1,2",
        help="Comma-separated candidate DE orders to try (e.g. '1' or '1,2').",
    )
    parser.add_argument(
        "--de_x_axis",
        type=int,
        default=None,
        help="Independent variable axis index (time). If omitted, auto-detect when possible, else defaults to 0.",
    )
    parser.add_argument("--de_max_x_power", type=int, default=1, help="Max power of x to include in DE library (default: 1).")
    parser.add_argument("--de_max_u_power", type=int, default=2, help="Max power of u to include in DE library (default: 2).")
    parser.add_argument("--de_no_const", dest="de_include_const", action="store_false", default=True, help="Exclude constant offset term from DE library.")
    parser.add_argument("--de_no_x", dest="de_include_x", action="store_false", default=True, help="Exclude x (and x^p) terms from DE library.")
    parser.add_argument("--de_no_u", dest="de_include_u", action="store_false", default=True, help="Exclude u (and u^q) terms from DE library.")
    parser.add_argument("--de_no_xu", dest="de_include_xu", action="store_false", default=True, help="Exclude x*u cross terms from DE library.")
    parser.add_argument("--de_no_xdu", dest="de_include_xdu", action="store_false", default=True, help="Exclude x*du cross terms from DE library.")
    parser.add_argument("--de_include_du", action="store_true", default=False, help="Allow du terms in the RHS library (if not the anchor).")
    parser.add_argument("--de_include_d2u", action="store_true", default=False, help="Allow d2u terms in the RHS library (if not the anchor).")
    parser.add_argument("--de_include_udu", action="store_true", default=False, help="Allow u*du cross term in the RHS library.")
    parser.add_argument("--de_ridge", type=float, default=1e-10, help="Ridge regularisation for linear solve (default: 1e-10).")
    parser.add_argument("--de_stlsq_lambda", type=float, default=1e-3, help="STLSQ threshold (default: 1e-3).")
    parser.add_argument("--de_stlsq_max_iter", type=int, default=10, help="STLSQ max iterations (default: 10).")
    parser.add_argument("--de_max_batches", type=int, default=32, help="Max batches sampled for DE discovery (default: 32).")
    parser.add_argument("--de_max_points", type=int, default=20000, help="Max points sampled for DE discovery (default: 20000).")
    parser.add_argument("--de_sparsity_penalty", type=float, default=1e-3, help="Penalty per active term for model selection (default: 1e-3).")
    parser.add_argument("--de_ast_simplify", "--ast-simplify", dest="de_ast_simplify", action="store_true", default=False, help="Enable conservative AST canonicalisation/deduplication in first-class DE discovery.")
    parser.add_argument("--de_no_ast_simplify", "--no-ast-simplify", dest="de_ast_simplify", action="store_false", help="Disable DE AST canonicalisation/deduplication.")
    parser.add_argument("--de_ast_simplify_level", "--ast-simplify-level", dest="de_ast_simplify_level", type=str, choices=["safe", "symmetry"], default="safe", help="AST simplification level for DE discovery.")
    parser.add_argument("--de_ast_simplify_domain_policy", "--ast-simplify-domain-policy", dest="de_ast_simplify_domain_policy", type=str, choices=["strict", "common-domain"], default="strict", help="Domain policy for DE AST simplification.")
    parser.add_argument("--de_ast_simplify_max_passes", "--ast-simplify-max-passes", dest="de_ast_simplify_max_passes", type=int, default=12, help="Maximum DE AST simplification passes.")
    parser.add_argument("--de_ast_simplify_validate", "--ast-simplify-validate", dest="de_ast_simplify_validate", action="store_true", default=False, help="Enable numeric validation for DE AST simplification where available.")
    parser.add_argument("--de_ast_simplify_trace", "--ast-simplify-trace", dest="de_ast_simplify_trace", action="store_true", default=False, help="Record detailed DE AST simplification diagnostics.")
    add_expr_ir_cli_args(parser)
    parser.add_argument("--de_coeff_prefix", type=str, default="c", help="Prefix for coefficient tags in the residual AST (default: 'c').")
    parser.add_argument(
        "--de_class_de",
        action="store_true",
        help="In multi-dataset mode: promote low-variance DE coefficients to class-shared constants (Class-DE).",
    )
    parser.add_argument(
        "--de_class_de_cv",
        type=float,
        default=0.05,
        help="Coefficient CV threshold for Class-DE promotion (default: 0.05).",
    )
    parser.add_argument(
        "--no_de_stageB",
        dest="de_stageB",
        action="store_false",
        default=True,
        help="Skip the Stage-B fit/simplify pass on the discovered DE residual.",
    )
    parser.add_argument(
        "--de_stageB_max_outer_iters",
        type=int,
        default=None,
        help="Override Stage B outer iters for DE residual refinement (default: reuse --stageB_max_outer_iters).",
    )
    parser.add_argument(
        "--de_stageB_epochs",
        type=int,
        default=None,
        help="Override Stage B epochs for DE residual refinement (default: reuse --stageB_epochs).",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to save a checkpoint (defaults to results/<stem>.state.pkl)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory for SR outputs (defaults to <repo>/results).",
    )

    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Path to save JSON report (defaults to results/<stem>.report.json)",
    )
    parser.add_argument(
        "--stat_selection",
        "--stat-selection",
        dest="stat_selection",
        action="store_true",
        help=(
            "Make an untouched ordinary-SR audit and simultaneous confidence "
            "Pareto front authoritative. Search heuristics remain proposal-only."
        ),
    )
    parser.add_argument(
        "--stat_audit_filepath",
        "--stat-audit-filepath",
        dest="stat_audit_filepath",
        type=str,
        default=None,
        help=(
            "Optional untouched audit CSV. Without it, the final contiguous tail "
            "of --filepath is physically withheld before search."
        ),
    )
    parser.add_argument(
        "--stat_audit_rows",
        "--stat-audit-rows",
        dest="stat_audit_rows",
        type=int,
        default=0,
        help="Rows reserved from the source tail for statistical audit; 0 uses --stat_audit_fraction.",
    )
    parser.add_argument(
        "--stat_audit_fraction",
        "--stat-audit-fraction",
        dest="stat_audit_fraction",
        type=float,
        default=0.2,
        help="Source-tail fraction reserved for untouched audit when no external audit CSV is supplied.",
    )
    parser.add_argument(
        "--stat_unit_size",
        "--stat-unit-size",
        dest="stat_unit_size",
        type=int,
        default=1,
        help=(
            "Number of contiguous audit rows per declared independent unit. "
            "Use 1 only for genuinely IID rows."
        ),
    )
    parser.add_argument(
        "--stat_alpha",
        "--stat-alpha",
        dest="stat_alpha",
        type=float,
        default=0.05,
        help="Family-wise error level for simultaneous Pareto dominance bounds.",
    )
    parser.add_argument(
        "--stat_delta",
        "--stat-delta",
        dest="stat_delta",
        type=float,
        default=0.0,
        help=(
            "Standardized audit-loss improvement that added complexity must "
            "certify; also the practical Pareto/deployment margin."
        ),
    )
    parser.add_argument(
        "--stat_resamples",
        "--stat-resamples",
        dest="stat_resamples",
        type=int,
        default=4000,
        help="Multiplier max-T draws for simultaneous confidence bounds.",
    )
    parser.add_argument(
        "--stat_seed",
        "--stat-seed",
        dest="stat_seed",
        type=int,
        default=12345,
        help="Random seed for statistical resampling.",
    )
    parser.add_argument(
        "--stat_multiplier",
        "--stat-multiplier",
        dest="stat_multiplier",
        choices=["normal", "rademacher"],
        default="normal",
        help="Multiplier distribution for simultaneous max-T inference.",
    )
    parser.add_argument(
        "--stat_max_candidates",
        "--stat-max-candidates",
        dest="stat_max_candidates",
        type=int,
        default=1024,
        help=(
            "Maximum canonical candidates in the frozen ordinary-SR audit archive. "
            "Primarily a compute bound. The comparison family is counted over "
            "exact pre-audit equivalence classes of the frozen archive (spellings "
            "that canonicalise to the same algebraic form merge; audit-based "
            "near-equivalence is descriptive only and never shrinks the family). "
            "Exceeding the measured calibration envelope is not a validity "
            "failure: select_inference_method reports fallback or beyond_grid and "
            "the pipeline then executes the one-sided Bonferroni-t critical value "
            "in place of the multiplier max-T one, which costs power rather than "
            "correctness. (The old low cap of 100 was NOT the cause of the pb016 "
            "regression once blamed on it; that was a float round-trip in the "
            "audit split, since fixed by byte-exact slicing.)"
        ),
    )
    parser.add_argument(
        "--stat_x_sigma",
        "--stat-x-sigma",
        dest="stat_x_sigma",
        type=str,
        default=None,
        help=(
            "Known Gaussian input standard deviation for Schur-profile audit loss. "
            "Use one scalar or a comma-separated value per x column."
        ),
    )
    parser.add_argument(
        "--stat_x_cov_npz",
        "--stat-x-cov-npz",
        dest="stat_x_cov_npz",
        type=str,
        default=None,
        help=(
            "Optional NPZ containing x_cov with shape (Nx,Nx) or "
            "(audit_rows,Nx,Nx); overrides --stat-x-sigma."
        ),
    )
    parser.add_argument(
        "--stat_x_error_loss",
        "--stat-x-error-loss",
        dest="stat_x_error_loss",
        choices=["marginal_gaussian_nll", "profile_chi2"],
        default="marginal_gaussian_nll",
        help=(
            "Local Gaussian Schur loss used when x errors are declared. "
            "'marginal_gaussian_nll' (default) includes the log-determinant "
            "normalisation, so a candidate cannot lower its loss merely by "
            "being steep and inflating its own effective variance. "
            "'profile_chi2' omits that term and is the profile chi-square: "
            "correct for a goodness-of-fit test at fixed structure, but not "
            "for ranking structures against each other."
        ),
    )
    parser.add_argument(
        "--stat_x_gradient_step",
        "--stat-x-gradient-step",
        dest="stat_x_gradient_step",
        type=float,
        default=1.0e-5,
        help="Relative central-difference step for symbolic candidate input gradients.",
    )
    parser.add_argument(
        "--stat_failure_loss",
        "--stat-failure-loss",
        dest="stat_failure_loss",
        type=float,
        default=1.0e6,
        help=(
            "Common bounded standardized loss assigned to domain, parse, shape, "
            "or nonfinite failures."
        ),
    )
    parser.add_argument(
        "--stat_archive_json",
        "--stat-archive-json",
        dest="stat_archive_json",
        type=str,
        default=None,
        help="Optional output path for the frozen candidate archive JSON.",
    )
    parser.add_argument(
        "--stat_certificate_json",
        "--stat-certificate-json",
        dest="stat_certificate_json",
        type=str,
        default=None,
        help="Optional output path for the confidence Pareto certificate JSON.",
    )

    parser.add_argument(
        "--final_polish",
        dest="final_polish",
        action="store_true",
        default=True,
        help="run the final post-hoc Pareto equation polisher and attach its recommendation to reports (default: on)",
    )
    parser.add_argument(
        "--no_final_polish",
        dest="final_polish",
        action="store_false",
        help="disable the final post-hoc Pareto equation polisher",
    )
    parser.add_argument(
        "--final_polish_out_dir",
        type=str,
        default=None,
        help="Directory for final polisher frontier files (default: results/<stem>_polish)",
    )
    parser.add_argument(
        "--final_polish_max_rows",
        type=int,
        default=10000,
        help="Optional row cap for final polisher scoring (default: 10000; <=0 uses all rows)",
    )
    parser.add_argument(
        "--final_polish_val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction for final polisher scoring split (default: 0.2)",
    )
    parser.add_argument(
        "--final_polish_max_candidates",
        type=int,
        default=256,
        help="Maximum final polisher candidates to score (default: 256)",
    )
    parser.add_argument(
        "--final_polish_max_seconds",
        type=float,
        default=30.0,
        help="Soft time budget for guarded final polisher simplification calls (default: 30)",
    )
    parser.add_argument(
        "--final_polish_full_dataset_snap",
        dest="final_polish_full_dataset_snap",
        action="store_true",
        default=True,
        help=(
            "In noisy runs, rescore final-polish snap candidates on the full CSV "
            "and let statistically tied simpler snaps win (default: on)"
        ),
    )
    parser.add_argument(
        "--no_final_polish_full_dataset_snap",
        dest="final_polish_full_dataset_snap",
        action="store_false",
        help="Disable the noisy-run full-dataset final snap adjudication pass",
    )
    parser.add_argument(
        "--final_polish_drop_addend_refit",
        dest="final_polish_drop_addend_refit",
        action="store_true",
        default=True,
        help=(
            "Propose final-polish candidates that delete a small additive term "
            "and refit+re-snap the surviving coefficients (default: on)"
        ),
    )
    parser.add_argument(
        "--no_final_polish_drop_addend_refit",
        dest="final_polish_drop_addend_refit",
        action="store_false",
        help=(
            "Disable the drop-small-addend refit proposal battery "
            "(restores pre-2026-08 final-polish frontiers exactly)"
        ),
    )
    parser.add_argument(
        "--final_polish_drop_refit_rel_ratio",
        type=float,
        default=5.0e-2,
        help=(
            "Coefficient-ratio ceiling for the large-number fingerprint route "
            "of the drop-addend refit battery (default: 5e-2)"
        ),
    )
    parser.add_argument(
        "--stageC_sympy_max_seconds",
        type=float,
        default=300.0,
        help="Wall-clock cap for the guarded Stage-C SymPy worker (default: 300)",
    )
    parser.add_argument(
        "--stageC_sympy_mem_fraction",
        type=float,
        default=0.20,
        help="Temporary RAM fraction cap for the guarded Stage-C SymPy worker (default: 0.20)",
    )
    parser.add_argument(
        "--no_stageC_sympy_subprocess",
        dest="stageC_sympy_subprocess",
        action="store_false",
        default=True,
        help="Run Stage-C SymPy simplification in-process instead of in the guarded worker",
    )
    parser.add_argument(
        "--final_polish_worker_max_seconds",
        type=float,
        default=300.0,
        help="Wall-clock cap for the guarded final-polish worker (default: 300)",
    )
    parser.add_argument(
        "--final_polish_worker_mem_fraction",
        type=float,
        default=0.20,
        help="Temporary RAM fraction cap for the guarded final-polish worker (default: 0.20)",
    )
    parser.add_argument(
        "--no_final_polish_subprocess",
        dest="final_polish_subprocess",
        action="store_false",
        default=True,
        help="Run final Pareto polish in-process instead of in the guarded worker",
    )
    parser.add_argument(
        "--discovery_enable",
        action="store_true",
        help="Run discovery committee/physics post-processing on the final SR result.",
    )
    parser.add_argument(
        "--discovery_report_json",
        type=str,
        default=None,
        help="Optional path to save discovery JSON report (defaults to results/<stem>.discovery.json)",
    )
    parser.add_argument(
        "--discovery_topk",
        type=int,
        default=8,
        help="Max number of SR candidates to expose to the discovery committee (default: 8).",
    )
    parser.add_argument(
        "--discovery_max_members",
        type=int,
        default=None,
        help="Optional cap on deduplicated committee size.",
    )
    parser.add_argument(
        "--discovery_experiment_manifest",
        type=str,
        default=None,
        help="Optional JSON manifest describing experiment candidates for active design.",
    )
    parser.add_argument(
        "--discovery_research_profile",
        type=str,
        default=None,
        choices=list(RESEARCH_PROFILE_NAMES),
        help="Optional named discovery research profile preset; omit to use the default witness-mode scheduler.",
    )
    parser.add_argument(
        "--discovery_constant_lift_enable",
        action="store_true",
        help="Promote drifting local constants into z->c(z) discovery tasks.",
    )
    parser.add_argument(
        "--discovery_constant_lift_min_regimes",
        type=int,
        default=3,
        help="Minimum number of regimes required before discovery constant lifting is attempted.",
    )
    parser.add_argument(
        "--discovery_constant_lift_trigger_mean_cv",
        type=float,
        default=0.5,
        help="Mean coefficient-of-variation threshold for triggering discovery constant lifting.",
    )
    parser.add_argument(
        "--discovery_constant_lift_apply_enable",
        action="store_true",
        help="Apply top constant-lift proposals back into the discovery committee and rescore them.",
    )
    parser.add_argument(
        "--discovery_constant_lift_apply_topk",
        type=int,
        default=1,
        help="Maximum number of constant-lift proposals to resplice into the committee.",
    )
    parser.add_argument(
        "--discovery_constant_lift_min_rel_gain",
        type=float,
        default=1.01,
        help="Minimum relative gain required before a constant-lift proposal is respliced.",
    )
    parser.add_argument(
        "--discovery_witness_capture_enable",
        action="store_true",
        help="Populate derivative and diagnostic witness predictions for discovery experiments.",
    )
    parser.add_argument(
        "--discovery_witness_hessian_diag_enable",
        action="store_true",
        help="Include Hessian-diagonal witness summaries when discovery witness capture is enabled.",
    )
    parser.add_argument(
        "--discovery_diagnostic_set",
        type=str,
        default="basic",
        choices=["basic", "extended", "physics"],
        help="Diagnostic summary set for discovery witness capture.",
    )
    parser.add_argument(
        "--discovery_beta",
        type=float,
        default=0.0,
        help="Weight on derivative disagreement in discovery active design (default: 0.0).",
    )
    parser.add_argument(
        "--discovery_gamma",
        type=float,
        default=0.0,
        help="Weight on diagnostic disagreement in discovery active design (default: 0.0).",
    )
    parser.add_argument(
        "--discovery_disagreement_mode",
        type=str,
        default="auto",
        choices=["auto", "witness"],
        help="Committee disagreement scorer for discovery active design (default: auto -> witness).",
    )
    parser.add_argument(
        "--discovery_lambda_cost",
        type=float,
        default=1.0,
        help="Penalty weight for experiment cost in discovery active design.",
    )
    parser.add_argument(
        "--discovery_lambda_noise",
        type=float,
        default=1.0,
        help="Penalty weight for experiment noise risk in discovery active design.",
    )
    parser.add_argument(
        "--discovery_lambda_feasibility",
        type=float,
        default=1.0,
        help="Penalty weight for experiment feasibility in discovery active design.",
    )
    parser.add_argument(
        "--discovery_experiment_optimize_enable",
        action="store_true",
        help="Optimize continuous discovery experiment settings before ranking them.",
    )
    parser.add_argument(
        "--discovery_experiment_opt_steps",
        type=int,
        default=32,
        help="Number of gradient steps for discovery experiment optimization.",
    )
    parser.add_argument(
        "--discovery_experiment_opt_lr",
        type=float,
        default=0.05,
        help="Learning rate for discovery experiment optimization.",
    )
    parser.add_argument(
        "--discovery_experiment_project_mode",
        type=str,
        default="nearest_box",
        help="Projection mode for optimized discovery experiments.",
    )
    parser.add_argument(
        "--discovery_theory_benchmark_enable",
        action="store_true",
        help="Emit theory-exploration benchmark metrics in the discovery report.",
    )

    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: reduced epochs/segments for quick testing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override DataHyperparams.batch_size (default from config: 2000).",
    )
    parser.add_argument(
        "--ndata_train",
        type=int,
        default=None,
        help="Override number of training points to sample per dataset (default: 2000).",
    )
    parser.add_argument(
        "--ndata_val",
        type=int,
        default=None,
        help="Override number of validation points to sample per dataset (default: 2000).",
    )
    parser.add_argument(
        "--data_slice",
        type=int,
        default=0,
        help=(
            "Deterministic disjoint data block index. The default 0 preserves "
            "the historical row-0 train/validation split."
        ),
    )
    parser.add_argument(
        "--coe_mode",
        type=str,
        default="off",
        choices=["off", "audit_final", "final_adjudicate", "committee_gated", "reservoir_discovery"],
        help=(
            "CoE slice-committee mode. audit_final is diagnostic; final_adjudicate "
            "may change the final reported expression; committee_gated also applies "
            "the Stage-B risky-accept gate where fixed-expression comparison is supported; "
            "reservoir_discovery additionally exposes Wave-3 proposal-reservoir artifacts."
        ),
    )
    parser.add_argument(
        "--coe_inference",
        type=str,
        default="legacy",
        choices=["legacy", "maxt_observe"],
        help=(
            "Committee decision statistics. legacy keeps the historical "
            "per-slice win/tie/loss vote against the noise_mult/rel_tol "
            "tolerance. maxt_observe additionally computes the calibrated "
            "paired max-T decision over the witness rows (rows are the "
            "statistical units, slices pure compute partitions) and records "
            "it next to every legacy vote WITHOUT changing any decision; the "
            "enforcing mode arrives after the observe-only comparison."
        ),
    )
    parser.add_argument(
        "--coe_maxt_seed",
        type=int,
        default=0,
        help="Seed for the committee max-T multiplier streams (counter-keyed per row).",
    )
    parser.add_argument(
        "--coe_num_slices",
        type=int,
        default=25,
        help="Number of deterministic slices for final CoE committee audit/adjudication.",
    )
    parser.add_argument(
        "--coe_start_slice",
        type=int,
        default=0,
        help="First deterministic slice id for CoE committee evaluation.",
    )
    parser.add_argument(
        "--coe_max_candidates",
        type=int,
        default=16,
        help="Maximum final analytic candidates scored by the CoE committee.",
    )
    parser.add_argument(
        "--coe_reservoir_paths",
        type=str,
        default=None,
        help=(
            "Optional comma- or path-separator-separated report files/directories "
            "whose stageB.coe_proposal_reservoir entries should be merged into "
            "the final CoE candidate pool. Stage-A proposal reservoirs in the "
            "same reports can also materialize y-branch proposals before y-search."
        ),
    )
    parser.add_argument(
        "--coe_noise_mult",
        type=float,
        default=3.0,
        help="Noise multiplier for paired CoE committee vote tolerances.",
    )
    parser.add_argument(
        "--coe_rel_tol",
        type=float,
        default=1.0e-3,
        help="Relative tolerance floor for CoE committee vote comparisons.",
    )
    parser.add_argument(
        "--coe_min_valid_fraction",
        type=float,
        default=0.80,
        help="Minimum finite prediction fraction for CoE committee slice scoring.",
    )
    parser.add_argument(
        "--coe_witness_parallelism",
        type=int,
        default=1,
        help=(
            "Maximum concurrent CoE committee witness evaluations. Default 1 "
            "preserves serial behavior."
        ),
    )
    parser.add_argument(
        "--coe_reservoir_support_bonus",
        type=float,
        default=None,
        help=(
            "Complexity bonus per log2(reservoir support count) when final CoE "
            "candidates are noise-tied. Defaults to 0 outside reservoir_discovery "
            "and 0.5 in reservoir_discovery."
        ),
    )
    parser.add_argument(
        "--coe_stageB_dry_run",
        action="store_true",
        help=(
            "Record CoE Stage-B dry-run risk diagnostics for accepted rewrites. "
            "This is observe-only and never vetoes Stage-B decisions."
        ),
    )
    parser.add_argument(
        "--coe_stageB_gate_slices",
        type=int,
        default=5,
        help=(
            "Maximum number of CoE validation slices used by the Stage-B committee gate "
            "in committee_gated mode. Final CoE audit still uses --coe_num_slices."
        ),
    )
    parser.add_argument(
        "--coe_stageB_initial_gate_slices",
        type=int,
        default=3,
        help=(
            "Initial slice count for adaptive Stage-B CoE gating before expanding "
            "up to --coe_stageB_gate_slices."
        ),
    )
    parser.add_argument(
        "--no_coe_stageB_refit_gate",
        action="store_true",
        help=(
            "Disable the CoE Stage-B short-refit gate for NN-containing incumbent/"
            "candidate decisions. Fixed analytic-expression gates remain enabled."
        ),
    )
    parser.add_argument(
        "--coe_stageB_refit_epochs",
        type=int,
        default=200,
        help=(
            "Epoch cap for committee short-refits of the same incumbent/candidate "
            "ASTs on independent CoE slices."
        ),
    )
    parser.add_argument(
        "--coe_stageB_refit_escalate_epochs",
        type=int,
        default=0,
        help=(
            "Optional Tier-1 CoE refit epoch cap. If greater than "
            "--coe_stageB_refit_epochs, a short-refit veto is retried at this "
            "higher budget before the veto is enforced. Default 0 disables escalation."
        ),
    )
    parser.add_argument(
        "--coe_stageA_dry_run",
        action="store_true",
        help=(
            "Record observe-only CoE Stage-A risk diagnostics. committee_gated "
            "enables this automatically."
        ),
    )
    parser.add_argument(
        "--coe_stageA_fit_tournament",
        action="store_true",
        help=(
            "For each eligible single-model Stage-A proposal, fit the master and "
            "scout-slice lanes concurrently, then adopt the valid identical-model "
            "fit with the lowest MSE on common validation rows."
        ),
    )
    parser.add_argument(
        "--coe_stageA_fit_slices",
        type=str,
        default=None,
        help="Explicit comma/space-separated challenger slices; defaults to scout slices.",
    )
    parser.add_argument(
        "--coe_stageA_fit_alpha",
        type=float,
        default=0.05,
        help="Deprecated compatibility option; best-valid fit selection does not use alpha.",
    )
    parser.add_argument(
        "--coe_stageA_fit_comparison_fraction",
        type=float,
        default=0.5,
        help="Fraction of reference validation rows reserved for common fit comparison.",
    )
    parser.add_argument(
        "--coe_stageA_fit_min_rel_improvement",
        type=float,
        default=0.01,
        help="Deprecated compatibility option; best-valid fit selection uses raw common-row MSE.",
    )
    parser.add_argument(
        "--coe_stageA_compound_shortlist_k",
        type=int,
        default=3,
        help=(
            "Maximum number of already-accepted strict arity-reducing Stage-A "
            "compound candidates to rank on independent CoE witness slices."
        ),
    )
    parser.add_argument(
        "--coe_stageA_split_near_floor_mult",
        type=float,
        default=25.0,
        help=(
            "A Stage-A split is considered near the noise floor for CoE gating when "
            "the parent or candidate validation loss is within this multiplier of "
            "the raw noise floor."
        ),
    )
    parser.add_argument(
        "--coe_scout_count",
        type=int,
        default=0,
        help=(
            "In reservoir_discovery mode, launch this many bounded scout proposer "
            "runs on the next deterministic data slices. Default 0 disables scouts."
        ),
    )
    parser.add_argument(
        "--coe_scout_slices",
        type=str,
        default=None,
        help=(
            "Explicit comma/space-separated scout slice ids for reservoir_discovery. "
            "Overrides --coe_scout_count when supplied."
        ),
    )
    parser.add_argument(
        "--coe_scout_timeout_seconds",
        type=float,
        default=0.0,
        help="Per-scout subprocess timeout in seconds. 0 means no timeout.",
    )
    parser.add_argument(
        "--coe_scout_parallelism",
        type=int,
        default=1,
        help=(
            "Maximum number of CoE scout subprocesses to run concurrently. "
            "Each scout subprocess uses the COE_WORKER_THREADS BLAS/OpenMP "
            "budget (default 1)."
        ),
    )
    parser.add_argument(
        "--coe_scout_stageB_epochs",
        type=int,
        default=800,
        help="Stage-B epoch cap for bounded CoE scout proposer runs.",
    )
    parser.add_argument(
        "--coe_scout_stageB_max_outer_iters",
        type=int,
        default=12,
        help="Stage-B outer-iteration cap for bounded CoE scout proposer runs.",
    )
    parser.add_argument(
        "--coe_scout_max_ab_iters",
        type=int,
        default=1,
        help="Stage A<->B iteration cap for bounded CoE scout proposer runs.",
    )
    parser.add_argument(
        "--coe_scout_stageA_max_passes",
        type=int,
        default=1,
        help=(
            "Stage-A loop-pass cap for bounded CoE scout proposer runs. "
            "1 stops when Stage A would loop back; 0 disables this cap."
        ),
    )
    parser.add_argument(
        "--no_coe_continuation_scouts",
        dest="coe_continuation_scouts",
        action="store_false",
        help=(
            "Disable bounded continuation scouts launched at accepted Stage-A "
            "restart and Stage-B -> Stage-A feedback boundaries. Initial scouts "
            "are unaffected."
        ),
    )
    parser.set_defaults(coe_continuation_scouts=True)
    parser.add_argument(
        "--coe_continuation_scout_count",
        type=int,
        default=None,
        help=(
            "Maximum scout subprocesses launched for each continuation phase. "
            "Continuation phases still draw slice ids from --coe_scout_slices or "
            "--coe_scout_count. Defaults to --coe_scout_count when omitted."
        ),
    )
    parser.add_argument(
        "--coe_continuation_scout_max_phases",
        type=int,
        default=6,
        help=(
            "Maximum accepted-reference continuation scout phases per run. "
            "0 disables the continuation phase cap."
        ),
    )
    parser.add_argument(
        "--coe_scout_final_polish",
        action="store_true",
        help="Allow final Pareto polish inside CoE scout proposer runs. Default off.",
    )
    parser.add_argument(
        "--coe_scout_with_stageB",
        action="store_true",
        help=(
            "Allow CoE scout proposers to run bounded Stage B. By default scouts "
            "are Stage-A-only proposal exporters for PR-A7."
        ),
    )

    parser.add_argument(
        "--disable_stageB_patterns",
        type=str,
        default=None,
        help="Comma-separated list of Stage B patterns to disable (e.g., trapped_sin_ratio,trig_comp)",
    )

    # ── Class SR: multi-dataset with shared constants ────────────────
    parser.add_argument(
        "--class_sr",
        action="store_true",
        help="Enable class SR: auto-detect class (shared) vs experiment constants across datasets",
    )
    parser.add_argument(
        "--class_atoms",
        type=str,
        default=None,
        help="Manually specify class atom tags (comma-separated, e.g., 'leaf0,leaf2')",
    )
    parser.add_argument(
        "--class_cv_threshold",
        type=float,
        default=0.15,
        help="CV threshold for auto-classifying atoms as class vs experiment (default: 0.15)",
    )
    parser.add_argument(
        "--class_auto_include_scales",
        action="store_true",
        help="Include scale/mul_scale leaves in Class-SR auto-classification. "
        "Default excludes them as nuisance constants.",
    )
    parser.add_argument(
        "--class_auto_include_nonfree",
        action="store_true",
        help="Include non-free leaves (e.g. poly/shape leaves) in Class-SR auto-classification. "
        "Default focuses auto-classification on free-constant leaves.",
    )
    parser.add_argument(
        "--class_sr_max_points",
        type=int,
        default=None,
        help="Optional cap on train/val samples per dataset used inside class-SR joint fitting.",
    )
    parser.add_argument(
        "--class_sr_optimizer",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "lm_tie"],
        help="Class-SR joint optimizer backend: lbfgs (default) or lm_tie (LM with exact tie constraints).",
    )
    parser.add_argument(
        "--no_class_param_sr",
        dest="class_param_sr",
        action="store_false",
        help="Disable Parameter-SR derived-invariant discovery/soft constraints in Class-SR.",
    )
    parser.set_defaults(class_param_sr=True)
    parser.add_argument(
        "--class_param_sr_max_invariants",
        type=int,
        default=4,
        help="Max derived invariants kept by Parameter-SR (default: 4).",
    )
    parser.add_argument(
        "--class_param_sr_score_threshold",
        type=float,
        default=0.05,
        help="Invariance score threshold for Parameter-SR candidates (default: 0.05).",
    )
    parser.add_argument(
        "--class_param_sr_penalty_weight",
        type=float,
        default=1.0e-2,
        help="Soft-constraint weight for Parameter-SR invariants during Class-SR joint fit (default: 1e-2).",
    )
    parser.add_argument(
        "--class_param_sr_max_scalars",
        type=int,
        default=16,
        help="Max scalar leaf quantities scanned by Parameter-SR (default: 16).",
    )
    parser.add_argument(
        "--class_param_sr_metadata",
        type=str,
        default=None,
        help=(
            "Optional dataset metadata scalars for Parameter-SR invariant discovery. "
            "Accepted forms: "
            "(1) row-wise list of dicts (len=D), "
            "(2) column-wise dict {meta_name: [v1,...,vD]}, "
            "(3) dataset-keyed dict {dataset_name|stem|path: {meta_name: value}}."
        ),
    )

    parser.add_argument(
        "--force_y_ops",
        type=str,
        default=None,
        help='Force specific y-transforms (comma-separated, e.g., "identity", "square", "log"). '
        "If not set, all transforms are tried.",
    )
    parser.add_argument(
        "--canonical_init",
        action="store_true",
        help="Apply NestyNet canonical initialization to pure Stage-A NN teacher fits before LM.",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Request SR LM construction with NestyNet evidence mode when supported.",
    )
    parser.add_argument(
        "--evidence_disable_residual_whitening",
        action="store_true",
        help=(
            "Disable evidence residual-whitening / patch terms. When combined with "
            "--evidence_disable_segment_priors this collapses back to the legacy LM path."
        ),
    )
    parser.add_argument(
        "--evidence_disable_segment_priors",
        action="store_true",
        help=(
            "Disable evidence segment priors. When combined with "
            "--evidence_disable_residual_whitening this collapses back to the legacy LM path."
        ),
    )
    parser.add_argument(
        "--evidence_lambda_patch",
        type=float,
        default=None,
        help=(
            "Evidence patch-whitening weight. SR now rejects positive residual-whitening "
            "weights; this option is retained to make accidental use fail loudly."
        ),
    )
    parser.add_argument(
        "--evidence_prior_decay_start",
        type=int,
        default=None,
        help=(
            "LM iteration where segment-prior decay starts. Default is automatic: "
            "800, moved earlier if needed so the default decay interval still fits "
            "inside the LM epoch budget, when --evidence is active."
        ),
    )
    parser.add_argument(
        "--evidence_prior_decay_interval",
        type=int,
        default=None,
        help=(
            "Number of LM iterations used to decay the segment prior once decay starts. "
            "Default: 200."
        ),
    )
    parser.add_argument(
        "--evidence_prior_decay_shape",
        type=str,
        choices=["linear", "smoothstep", "cosine"],
        default=None,
        help="Shape of the segment-prior decay ramp.",
    )
    parser.add_argument(
        "--evidence_prior_decay_final_scale",
        type=float,
        default=None,
        help="Final global segment-prior multiplier after decay (default: 0).",
    )
    parser.add_argument(
        "--evidence_prior_cutoff_tol",
        type=float,
        default=None,
        help=(
            "Early-start trigger for segment-prior decay. If plain training selection-loss "
            "loss improvement over an LM report period falls below this threshold before "
            "the scheduled decay start, SR starts the decay immediately. Interpreted on "
            "the plain SR selection-loss scale, so it stays aligned with the visible SR "
            "loss thresholds rather than the augmented evidence objective. "
            "Default: 1e-9."
        ),
    )
    parser.add_argument(
        "--no_evidence_prior_decay_auto",
        dest="evidence_prior_decay_auto",
        action="store_false",
        default=True,
        help="Disable automatic segment-prior decay when --evidence is active.",
    )
    parser.add_argument(
        "--no_evidence_metric_gate",
        dest="evidence_metric_gate",
        action="store_false",
        default=True,
        help="Allow SR validation / stopping metrics before segment-prior decay completes.",
    )

    parser.add_argument(
        "--no_outer_peel_autorun",
        dest="outer_peel_autorun",
        action="store_false",
        help="Deprecated no-op (outer-peel autorun training path has been removed in unified y-search mode).",
    )
    parser.add_argument(
        "--no_ysearch",
        dest="ysearch_enable",
        action="store_false",
        help="Disable virtual y-transform ranking in Stage A (fallback to legacy candidate gating).",
    )
    parser.add_argument(
        "--ysearch_expand_k",
        type=int,
        default=None,
        help="Number of top-ranked virtual y-transforms to full-fit in Stage A fallback (default: config value).",
    )
    parser.add_argument(
        "--ysearch_portfolio_margin_decades",
        type=float,
        default=None,
        help="Keep additional y-transform candidates whose virtual proxy loss is within this many decades of the cutoff.",
    )
    parser.add_argument(
        "--ysearch_portfolio_max_k",
        type=int,
        default=None,
        help="Maximum y-transform candidates retained by portfolio tie expansion.",
    )
    parser.add_argument(
        "--ysearch_depth",
        type=int,
        default=None,
        help="Maximum y-transform stack depth for y-search (default: 1).",
    )
    parser.add_argument(
        "--ysearch_beam",
        type=int,
        default=None,
        help="Beam width for depth-limited y-search frontier pruning (default: config value).",
    )
    parser.add_argument(
        "--ysearch_min_valid_frac",
        type=float,
        default=None,
        help="Minimum valid fraction for y-transform domain checks in virtual ranking (default: config value).",
    )
    parser.add_argument(
        "--ysearch_confirm_improve_ratio",
        type=float,
        default=None,
        help="Branch confirmation ratio for y-search expansion (child_loss <= ratio * parent_loss).",
    )
    parser.add_argument(
        "--ysearch_trigger_trig_affine_conf",
        type=float,
        default=None,
        help="Strong-trigger threshold for trig-affine confidence crossings in y-search (default: config value).",
    )
    parser.add_argument(
        "--ysearch_trigger_sep_min",
        type=float,
        default=None,
        help="Strong-trigger minimum sep score for y-search child states (default: config value).",
    )
    parser.add_argument(
        "--ysearch_trigger_sep_delta",
        type=float,
        default=None,
        help="Strong-trigger minimum sep score improvement over parent state (default: config value).",
    )
    parser.add_argument(
        "--ysearch_trigger_split_score",
        type=float,
        default=None,
        help="Strong-trigger minimum split score for y-search child states (default: config value).",
    )
    parser.add_argument(
        "--ysearch_trigger_split_margin",
        type=float,
        default=None,
        help="Strong-trigger minimum split score improvement over parent state (default: config value).",
    )
    parser.add_argument(
        "--ysearch_max_virtual_deriv",
        type=float,
        default=None,
        help="Absolute clip for virtual transform derivatives during chain-rule probes (default: config value).",
    )
    parser.add_argument(
        "--ysearch_outer_affine_confirm_rms_rel",
        type=float,
        default=None,
        help="Relative RMS threshold for accepting φ(y) ≈ a*z+b as an outer-affine certificate.",
    )
    parser.add_argument(
        "--ysearch_outer_affine_min_domain_frac",
        type=float,
        default=None,
        help="Minimum y-domain coverage for an outer-affine certificate.",
    )
    parser.add_argument(
        "--ysearch_max_state_evals",
        type=int,
        default=None,
        help="Hard cap on Stage-A state evaluations inside y-search (0 means unbounded).",
    )
    parser.add_argument(
        "--ysearch_max_recursive_branches",
        type=int,
        default=None,
        help="Optional split-recursion budget for y-search controller (0 disables recursion).",
    )
    parser.add_argument(
        "--ysearch_max_split_plans_per_state",
        type=int,
        default=None,
        help="Maximum split plans consumed per accepted state when split-recursion is enabled.",
    )

    parser.add_argument(
        "--single_layer",
        action="store_true",
        help="Use single-layer architecture for NN atoms (default is dual-layer)",
    )

    # Units / dimensional analysis (on by default; pass --ignore_units to skip)
    parser.add_argument(
        "--ignore_units",
        action="store_true",
        help="Disable dimensional consistency checks (units are enforced by default when a units spec is provided).",
    )
    parser.add_argument(
        "--units",
        type=str,
        default=None,
        help='Units spec. Either JSON {"y":[...],"x":[[...],...]} or two bracket-lists "[...]" "[[...],...]".',
    )
    parser.add_argument(
        "--y_units", type=str, default=None, help="Units for y as a Python/JSON list of exponents."
    )
    parser.add_argument(
        "--x_units",
        type=str,
        default=None,
        help="Units for x variables as a Python/JSON list-of-lists of exponents.",
    )
    parser.add_argument(
        "--units_basis",
        type=str,
        default=None,
        help='Comma-separated basis names for unit exponents (e.g. "L,T,M,I,Θ"). If omitted, inferred.',
    )
    parser.add_argument(
        "--equations_txt",
        type=str,
        default=None,
        help="Path to equations.txt containing y_units/x_units columns; if set and no units are passed explicitly, load units for this dataset id (CSV stem). For blinded runs, point this at a units-only manifest (see scripts/make_units_manifest.py).",
    )
    parser.add_argument(
        "--blinded",
        action="store_true",
        default=False,
        help="Blinded mode: forbid all in-process access to the ground-truth answer key. Disables the built-in truth evaluation (which would otherwise open aif_canaries.json); score afterwards with scripts/score_blinded_run.py. Pair with a units-only --equations_txt manifest so no file containing a target expression is ever opened by the search.",
    )
    parser.add_argument(
        "--free_consts",
        type=str,
        default=None,
        help="Optional mapping of trainable free constant names to unit vectors, as JSON or Python dict. "
        'Example: {"c":[1,-1,0,0,0]}.',
    )
    parser.add_argument(
        "--local_consts",
        type=str,
        default=None,
        help="Per-dataset trainable constants with unit vectors (scope='experiment'). "
        'JSON dict: {"c0":[1,0], "c1":[0,-1]}.',
    )
    parser.add_argument(
        "--global_consts",
        type=str,
        default=None,
        help="Shared trainable constants with unit vectors (scope='class'). "
        'JSON dict: {"g0":[0,-1]}.',
    )
    parser.add_argument(
        "--fixed_consts",
        type=str,
        default=None,
        help="Optional mapping of fixed physical constant names to (value, unit_vec). "
        "JSON/Python dict accepted. Examples: {\"c\": [299792458.0, [1,-1,0,0,0]]} or "
        "{\"eV\": {\"value\": 1.602176634e-19, \"units\": [2,-2,1,0,0]}}.",
    )
    parser.add_argument(
        "--fixed_consts_mode",
        type=str,
        default="strict",
        choices=["strict", "minimal", "off"],
        help="Declared fixed-constant policy for candidate construction: "
        "'strict' (default) injects fixed-const scalar variants broadly; "
        "'minimal' keeps only explicitly requested template usages; "
        "'off' disables fixed-const variant injection.",
    )

    parser.add_argument(
        "--units_policy",
        type=str,
        default="free_const_only",
        help='Units policy for the checker. "free_const_only" (default) enforces that only declared free constants may carry units.',
    )
    parser.add_argument(
        "--nn_units_semantics",
        type=str,
        default="unknown",
        help='How to treat NN leaves under unit checking: "unknown" (default) = allow any output units; "dimless" = require dimensionless; "span" = output dim must be in the rational span of input dims and declared constants.',
    )

    # Generalized-symmetry Stage-A proposals. Disabled by default. The
    # resulting coordinates still pass through ordinary Stage-A validation.
    parser.add_argument("--gs-stagea", dest="gs_stagea", action="store_true", default=True, help="Enable affine/Lie-style GS audits and Stage-A coordinate proposals (default: on)")
    parser.add_argument("--gs-no-stagea", dest="gs_stagea", action="store_false", help="Disable affine/Lie-style GS Stage-A proposals")
    parser.add_argument("--gs-mode", type=str, choices=["off", "audit", "propose", "auto"], default="propose", help="GS operating mode; audit records witnesses without proposing coordinates")
    parser.add_argument("--gs-policy", type=str, choices=["augment", "replace-shadowed", "gs-only-affine"], default="augment", help="GS proposal policy; augment is the conservative default")
    parser.add_argument("--gs-known-generators", dest="gs_known_generators", action="store_true", default=True, help="Enable named translation/scaling/rotation GS probes")
    parser.add_argument("--gs-no-known-generators", dest="gs_known_generators", action="store_false", help="Disable named GS probes")
    parser.add_argument("--gs-general-affine", dest="gs_general_affine", action="store_true", default=False, help="Enable learned pairwise affine-generator probes")
    parser.add_argument("--gs-no-general-affine", dest="gs_general_affine", action="store_false", help="Disable learned pairwise affine-generator probes")
    parser.add_argument("--gs-charts", type=str, default="identity,log,reciprocal", help="Comma-separated charts for the GS determining operator: identity,log,reciprocal,warp. log exposes monomial invariants (positive inputs); reciprocal exposes sum c_i/x_i invariants (parallel-resistor/lens family, nonzero non-sign-crossing inputs); warp *discovers* the per-axis coordinate warp that linearizes a generalized-additive symmetry (needs the leaf Hessian and >=3 vars). Default identity,log,reciprocal; pass identity to restore the audit-only behavior")
    parser.add_argument("--gs-chart-snap-denominator", type=int, default=4, help="Maximum rational denominator when snapping log-chart exponent rays")
    parser.add_argument("--gs-noise-calibrated-promotion", dest="gs_noise_calibrated_promotion", action="store_true", default=True, help="Promote GS reductions on surrogate-noise-relative evidence (spectral-gap nullity, held-out consistency, bootstrap stability) instead of the oracle-only absolute residual gates (default: on)")
    parser.add_argument("--gs-no-noise-calibrated-promotion", dest="gs_noise_calibrated_promotion", action="store_false", help="Fall back to the oracle-only absolute residual promotion gates")
    parser.add_argument("--gs-nc-min-spectral-gap", type=float, default=10.0, help="Minimum singular-spectrum gap for the noise-calibrated promotion tier")
    parser.add_argument("--gs-pairwise-composition", dest="gs_pairwise_composition", action="store_true", default=True, help="Compose accepted pairwise scaling witnesses into global +/-1 monomial rays (noise-robust route to 3+-variable products/ratios), jointly validated and promoted through the standard proposal contract (default: on)")
    parser.add_argument("--gs-no-pairwise-composition", dest="gs_pairwise_composition", action="store_false", help="Disable pairwise scaling-witness composition")
    parser.add_argument("--gs-recursive-composition", dest="gs_recursive_composition", action="store_true", default=True, help="Recursively compose certified GS carriers in reduced coordinate space; needs --gs-pairwise-composition (default: on)")
    parser.add_argument("--gs-no-recursive-composition", dest="gs_recursive_composition", action="store_false", help="Disable recursive coordinate composition")
    parser.add_argument("--gs-recursive-max-depth", type=int, default=3, help="Maximum GS carrier depth including primitive depth 1 (default: 3, i.e. up to two recursive composition steps)")
    parser.add_argument("--gs-recursive-beam-width", type=int, default=2, help="Maximum newly composed GS carriers retained per recursive depth")
    parser.add_argument("--gs-stagea-proposal-budget", type=int, default=6, help="Hard total cap on GS carrier trials per Stage-A atom, including the protected decisive trial")
    parser.add_argument("--gs-decisive-min-confidence", type=float, default=0.995, help="Minimum confidence for the single protected full-support GS Stage-A trial")
    parser.add_argument("--gs-decisive-max-trials", type=int, default=1, help="Maximum protected decisive GS trials before the ordinary Stage-A compound lane")
    parser.add_argument("--gs-lorentz-boosts", dest="gs_lorentz_boosts", action="store_true", default=False, help="Enable Lorentz-boost invariant probes")
    parser.add_argument("--gs-no-lorentz-boosts", dest="gs_lorentz_boosts", action="store_false", help="Disable Lorentz-boost invariant probes")
    parser.add_argument("--gs-output-equivariance", dest="gs_output_equivariance", action="store_true", default=True, help="Audit affine output actions Xf=alpha+beta*f")
    parser.add_argument("--gs-no-output-equivariance", dest="gs_output_equivariance", action="store_false", help="Disable affine output-action fitting")
    parser.add_argument("--gs-residual-tol", type=float, default=0.03, help="GS proposal residual tolerance")
    parser.add_argument("--gs-audit-residual-tol", type=float, default=0.10, help="GS audit residual tolerance")
    parser.add_argument("--gs-min-confidence", type=float, default=0.65, help="Minimum GS witness confidence")
    parser.add_argument("--gs-max-pair-generators", type=int, default=16, help="Maximum coordinate pairs audited per NN leaf")

    # Generalized-symmetry dimensional analysis. Disabled by default; when
    # explicitly enabled, audit is the conservative default policy.
    parser.add_argument("--gs-unit-torus", dest="gs_unit_torus", action="store_true", default=False, help="Enable unit-torus/Buckingham-pi dimensional GS audit or proposals")
    parser.add_argument("--gs-no-unit-torus", dest="gs_unit_torus", action="store_false", help="Disable unit-torus dimensional GS")
    parser.add_argument("--gs-pi-invariants", dest="gs_pi_invariants", action="store_true", default=False, help="Enable Buckingham-pi invariant GS proposals where units are available")
    parser.add_argument("--gs-no-pi-invariants", dest="gs_pi_invariants", action="store_false", help="Disable Buckingham-pi invariant GS proposals")
    parser.add_argument("--gs-dim-policy", type=str, choices=["baseline", "audit", "augment", "both", "replace-rref", "gs-only"], default="audit", help="Dimensional GS policy; --gs-unit-torus defaults to audit")
    parser.add_argument("--gs-dim-both-rule", type=str, choices=["rref-dominates", "require-both", "either", "gs-dominates"], default="rref-dominates", help="Arbitration rule for --gs-dim-policy both")
    parser.add_argument("--gs-dim-validator", type=str, choices=["local", "nullspace", "linear"], default="nullspace", help="Unit-torus dimensional validator")
    parser.add_argument("--gs-dim-keep-local-gates", dest="gs_dim_keep_local_gates", action="store_true", default=True, help="Keep local dimensional safety gates in replacement modes")
    parser.add_argument("--gs-dim-no-local-gates", dest="gs_dim_keep_local_gates", action="store_false", help="Disable local dimensional safety gates in replacement modes")
    parser.add_argument("--gs-pi-max-exponent", type=int, default=3, help="Maximum absolute exponent in bounded Buckingham-pi enumeration")
    parser.add_argument("--gs-pi-max-l1", type=int, default=6, help="Maximum L1 exponent norm in bounded Buckingham-pi enumeration")
    parser.add_argument("--gs-pi-max-proposals", type=int, default=24, help="Maximum unit-torus pi/prefactor proposals")
    parser.add_argument("--gs-pi-max-basis", type=int, default=8, help="Maximum support size for bounded pi/prefactor enumeration")
    parser.add_argument("--gs-pi-rational-denom", type=int, default=1, help="Maximum rational denominator for pi exponents")
    parser.add_argument("--gs-pi-include-free-consts", dest="gs_pi_include_free_consts", action="store_true", default=True, help="Allow declared free constants in GS dimensional span checks")
    parser.add_argument("--gs-pi-no-free-consts", dest="gs_pi_include_free_consts", action="store_false", help="Ignore free constants in GS dimensional span checks")
    parser.add_argument("--gs-unit-report-json", type=str, default=None, help="Optional path for unit-torus GS report JSON")
    parser.add_argument("--gs-unit-report-md", type=str, default=None, help="Optional path for unit-torus GS report markdown")
    parser.add_argument("--gs-report-dim-disagreements", dest="gs_report_dim_disagreements", action="store_true", default=True, help="Report baseline/GS dimensional disagreements")
    parser.add_argument("--gs-no-report-dim-disagreements", dest="gs_report_dim_disagreements", action="store_false", help="Suppress baseline/GS dimensional disagreement rows")
    parser.add_argument("--gs-report-pi-rejected", dest="gs_report_pi_rejected", action="store_true", default=False, help="Report rejected pi proposals where available")

    # Logging arguments
    parser.add_argument(
        "--log_file", type=str, default=None, help="Path to log file for optimizer output"
    )
    parser.add_argument(
        "--no_log_to_console",
        dest="log_to_console",
        action="store_false",
        help="Disable console logging (default: console logging enabled)",
    )
    parser.set_defaults(log_to_console=True)
    parser.add_argument(
        "--log_level", type=str, default=None, help="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    parser.add_argument(
        "--lm_verbose", action="store_true", default=False,
        help="Enable per-epoch LM optimizer progress output",
    )

    parser.add_argument(
        "--verbose_separabilities",
        action="store_true",
        help="Print detailed cross-derivative diagnostics during separability checks",
    )

    # Compound variable detection arguments
    parser.add_argument(
        "--disable_compound_detection",
        action="store_true",
        default=False,
        help="Disable compound variable detection (e.g., NN(x0*x1), NN(x0/x1))",
    )
    parser.add_argument(
        "--compound_max_batches",
        type=int,
        default=4,
        help=(
            "Maximum number of training batches to use when estimating gradients for "
            "compound detection (default: 4)."
        ),
    )
    args = parser.parse_args()
    # Translate --ignore_units → enforce_units for all internal code
    args.enforce_units = not args.ignore_units
    return args
