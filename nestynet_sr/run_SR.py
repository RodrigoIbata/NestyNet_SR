# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Symbolic Regression via Neural Network Surrogates

This script discovers analytical expressions from data by:
1. Training a neural network surrogate f(x)
2. Detecting separability in the network structure
3. Proposing and refining analytical forms (polynomials, sinusoids, etc.)

Usage Examples:

  # Basic symbolic regression on CSV data
  python run_SR.py --filepath ../data/my_data.csv

  # Fast mode (reduced epochs/segments for quick testing)
  python run_SR.py --filepath ../data/my_data.csv --fast

  # Resume from a saved checkpoint
  python run_SR.py --resume_from results/my_data.state.pkl

  # Load initial expressions from pickle and run Stage B refinement
  python run_SR.py --filepath ../data/my_data.csv --load_expressions results/my_data.expressions.pkl

  # Disable Stage B refinement (Stage A only)
  python run_SR.py --filepath ../data/my_data.csv --no_stageB

  # Force specific y-transforms (e.g., only try identity and square)
  python run_SR.py --filepath ../data/my_data.csv --force_y_ops identity,square

  # Deprecated (no-op in unified y-search mode)
  python run_SR.py --filepath ../data/my_data.csv --no_outer_peel_autorun

  # Disable specific Stage B rewrite patterns
  python run_SR.py --filepath ../data/my_data.csv --disable_stageB_patterns trapped_sin_ratio,trig_comp

  # Control Stage B parameters
  python run_SR.py --filepath ../data/my_data.csv --stageB_max_outer_iters 50 --stageB_epochs 3000

  # Dimensional analysis is on by default when units are provided
  python run_SR.py --filepath ../data/my_data.csv \
      --y_units "[1,0,0]" --x_units "[[1,0,0],[0,1,0]]" --units_basis "L,T,M"

  # Load units from equations.txt file
  python run_SR.py --filepath ../data/my_data.csv --equations_txt ../data/equations.txt

  # Disable dimensional analysis
  python run_SR.py --filepath ../data/my_data.csv --ignore_units

  # Save custom checkpoint and JSON report paths
  python run_SR.py --filepath ../data/my_data.csv \
      --checkpoint_path my_checkpoint.pkl --report_json my_report.json

Output Files:
  - results/<stem>.human              Human-readable discovered expression
  - results/<stem>.state.pkl          Checkpoint for resuming
  - results/<stem>.report.json        Detailed JSON report
  - results/<stem>_polish/            Final Pareto-polisher frontier (if enabled)
  - results/<stem>.expressions.pkl    Stage A expressions (if applicable)

For more details, see the symbolic regression documentation.
"""

import copy
import hashlib
import math
import json
import os
import pathlib
import pickle
import random
import shutil
import socket
import sys
import timeit
import traceback
from typing import Any, Optional

# ------------------------------------------------------------------
# Ensure the local checkout takes precedence when running this file as
# a script (e.g. `python nestynet_sr/run_SR.py`). In that invocation,
# `sys.path[0]` is the *package directory* itself, so `import nestynet_sr`
# would otherwise resolve to a site-packages installation if present.
# ------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import nestynet
import numpy as np
import torch

from nestynet_sr.sr_core import ast_to_human_readable, build_initial_ast
from nestynet_sr.sr_core.bridges import (
    make_stage_a_nn_factory,
)
from nestynet_sr.sr_core.fit_links import canonical_fit_link_name, describe_fit_link
from nestynet_sr.sr_search.config import (
    FactorizedSearchConfig,
    DataHyperparams,
    LMHyperparams,
    ModelHyperparams,
    SearchHyperparams,
)
from nestynet_sr.sr_search.factorized_search.config import (
    apply_refine_profile,
    factorized_config_report,
)
from nestynet_sr.sr_expr_ir.config import apply_expr_ir_args_to_obj
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast
from nestynet_sr.sr_search.search import run_separability_for_transform, stageA_analyze
from nestynet_sr.sr_search.y_transforms import (
    build_default_y_transforms,
    compose_y_stack_ops,
    encode_y_stack_name,
    resolve_y_transform_name,
)
from nestynet_sr.run_sr_args import parse_args
from nestynet_sr.run_sr_coe import (
    _apply_coe_final_adjudication,
    _apply_stageA_provisional_guard,
    _build_coe_stageA_dry_run_records,
    _coe_problem_stem,
    _coe_stageA_materialization_mode_enabled,
    _coe_stageA_ybranch_committee_rank,
    _format_coe_stageA_dry_run_report,
    _format_coe_stageA_exit_audit_report,
    _format_coe_stageA_ybranch_committee_report,
    _format_coe_committee_report,
    _format_coe_stageB_dry_run_report,
    _format_coe_stageB_gate_report,
    _enforce_stageA_provisional_guard_on_report,
    _load_coe_external_stageA_proposal_reservoir,
    _materialize_stageA_y_branch_proposals,
    _merge_coe_expression_reservoir_payload,
    _parse_coe_scout_slice_ids,
    _run_coe_final_committee,
    _run_coe_scout_proposers,
    _run_coe_stageA_exit_audit,
    _stageA_provisional_confirmation_summary,
    _summarize_coe_stageA_dry_run,
    _summarize_coe_stageB_dry_run,
    _summarize_coe_stageB_gate,
    _update_report_with_coe_committee,
    _write_coe_stageA_dry_run_jsonl,
    _write_coe_stageB_dry_run_jsonl,
    _write_coe_stageB_gate_jsonl,
    _write_stageA_provisional_confirmation_jsonl,
)
from nestynet_sr.run_sr_de import (
    _StageAXTransformDataset as _StageAXTransformDataset,
    _parse_int_tuple as _parse_int_tuple,
    _run_firstclass_de_for_sr as _run_firstclass_de_for_sr,
    _train_stageA_models_multi_for_stageB as _train_stageA_models_multi_for_stageB,
)
from nestynet_sr.run_sr_final_polish import _run_final_pareto_polish
from nestynet_sr.stat_selection.sr_pipeline import (
    NoPortableAnalyticCandidatesError,
    format_sr_statistical_selection,
    prepare_sr_audit_plan,
    run_sr_statistical_selection,
    update_report_with_sr_statistical_selection,
)
from nestynet_sr.run_sr_stageb_utils import (
    CompoundCoordVariant as CompoundCoordVariant,
    _compute_compound_is_1d as _compute_compound_is_1d,
    _compound_coordinate_variant_values as _compound_coordinate_variant_values,
    _compound_coordinate_variants as _compound_coordinate_variants,
    _has_stageA_split as _has_stageA_split,
    _identity_outer_affine_units_ok as _identity_outer_affine_units_ok,
    _loss_scale_from_loader_raw_y as _loss_scale_from_loader_raw_y,
    _payload_confirmation_rank as _payload_confirmation_rank,
    _payload_confirmation_status as _payload_confirmation_status,
    _payload_is_confirmed as _payload_is_confirmed,
    _phase_prescan_training_arrays as _phase_prescan_training_arrays,
    _print_compound_outer_affine_entries as _print_compound_outer_affine_entries,
    _probe_compound_outer_affine_variants as _probe_compound_outer_affine_variants,
    _split_success as _split_success,
    _stageB_adjudication_key as _stageB_adjudication_key,
    _stageB_candidate_metrics as _stageB_candidate_metrics,
    _stageB_expression_complexity_score as _stageB_expression_complexity_score,
    _stageB_generic_approximant_signature as _stageB_generic_approximant_signature,
    _stageB_is_plain_rational_approximant_label as _stageB_is_plain_rational_approximant_label,
    _stageB_portfolio_can_stop_early as _stageB_portfolio_can_stop_early,
    _stageB_portfolio_continue_reason as _stageB_portfolio_continue_reason,
    _stageB_portfolio_early_stop_decision as _stageB_portfolio_early_stop_decision,
    _stageB_raw_y_branch_family_signature as _stageB_raw_y_branch_family_signature,
    _stageB_shadow_rescue_reason as _stageB_shadow_rescue_reason,
    _stageB_shortlist_names as _stageB_shortlist_names,
    _stageB_shortlist_source_map as _stageB_shortlist_source_map,
    _stageB_small_integer_rational_expression as _stageB_small_integer_rational_expression,
    _stageB_sparse_rational_raw_y_expression as _stageB_sparse_rational_raw_y_expression,
    _stageB_y_branch_artifact as _stageB_y_branch_artifact,
    _strong_outer_link_hints_for_direct_closure as _strong_outer_link_hints_for_direct_closure,
    _strong_phase_context_hints_for_direct_closure as _strong_phase_context_hints_for_direct_closure,
    _strong_phase_hints_for_direct_closure as _strong_phase_hints_for_direct_closure,
    _trial_adjudication_key as _trial_adjudication_key,
    _try_phase_prescan_direct_closure as _try_phase_prescan_direct_closure,
)
from nestynet_sr.run_sr_input_utils import (
    _infer_units_basis,
    _load_units_from_equations,
    _metadata_linked_invariants as _metadata_linked_invariants,
    _normalize_class_param_sr_metadata as _normalize_class_param_sr_metadata,
    _parse_py_or_json_literal,
    _parse_units_arg,
)
from nestynet_sr.run_sr_reports import (
    _append_final_selection_report,
    _append_final_simplification_path_state,
    _append_truth_eval_summary_to_file,
    _decorate_simplification_path_y_space,
    _final_selection_from_report as _final_selection_from_report,
    _format_final_selection_report as _format_final_selection_report,
    _format_final_polish_report,
    _format_simplification_path,
    _format_truth_eval_summary as _format_truth_eval_summary,
    _make_json_serializable,
    _protect_exact_stageB_seed_in_final_polish as _protect_exact_stageB_seed_in_final_polish,
    _refresh_final_selection_truth_eval,
    _select_final_polish_seed as _select_final_polish_seed,
    _stageA_status_message,
    _update_report_with_final_polish,
    _update_report_with_campaign_outcome,
    has_nn_atoms,
    write_json_report,
)

# ANSI color codes for terminal output
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"




# ──────────────────────────────────────────────────────────────
# Units parsing helpers (Stage-B straightjacket)
# ──────────────────────────────────────────────────────────────


def _parse_gs_charts(raw) -> tuple:
    """Parse and validate the --gs-charts comma list."""

    names = []
    for item in str(raw or "identity").split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in {"identity", "log", "reciprocal", "warp"}:
            raise ValueError(f"--gs-charts: unknown chart {name!r}; valid charts are: identity, log, reciprocal, warp")
        if name not in names:
            names.append(name)
    return tuple(names) if names else ("identity",)




# Structural proposal lanes with provisional acceptance: each lane records a
# proposal marker in gate telemetry under its rule name, and honors the paired
# suppression env var so a failed run can be retried without it.
_STRUCTURAL_LANES = {
    "visible_buckingham_lane": "NNSR_SUPPRESS_VISIBLE_BUCKINGHAM",
}

_STAGEA_LEDGER_ATTRS = {
    "stageA_move_records": "_stageA_move_records",
    "stageA_provisional_commits": "_stageA_provisional_commits",
    "stageA_rejection_records": "_stageA_rejection_records",
}


def _stageA_ledgers_from_model_or_checkpoint(model=None, checkpoint=None) -> dict:
    """Return the Stage-A debt ledgers, preferring persisted resume state."""

    restored = {}
    for key, attr in _STAGEA_LEDGER_ATTRS.items():
        if isinstance(checkpoint, dict) and key in checkpoint:
            raw = checkpoint.get(key)
        else:
            raw = getattr(model, attr, None) if model is not None else None
        if raw is None:
            raw = []
        if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
            raise ValueError(f"invalid {key} in Stage-A fitted-state ledger")
        restored[key] = [copy.deepcopy(row) for row in raw]
    return restored


def _buckingham_retry_trigger(report: dict) -> tuple:
    """Decide whether to retry the run with structural lanes suppressed.

    Trigger requires BOTH (a) the final expression failed internal Stage-C
    verification (uncertified or unresolved NN leaves; truth evaluation is
    never consulted) and (b) at least one registered structural lane was
    active this run. Rationale: these lanes commit structure on Stage-A
    evidence, but their true cost (downstream rules can no longer match the
    original atom) only materializes later, so acceptance must be reversible
    on downstream failure.

    Returns (trigger, reason, suppress_env_vars_for_fired_lanes).
    """
    stagec = report.get("stageC") or {}
    unresolved = bool((stagec.get("unresolved") or {}).get("unresolved"))
    certified = bool(stagec.get("certified", False))
    if certified and not unresolved:
        return False, "certified", []
    tel = report.get("gate_telemetry") or {}
    fired = {}
    for rec in tel.get("records", []):
        rule = rec.get("rule")
        if rule == "stageA_coe_accept" and rec.get("accepted"):
            ctx = rec.get("context") or {}
            if ctx.get("visible_buckingham"):
                fired["visible_buckingham_lane"] = _STRUCTURAL_LANES[
                    "visible_buckingham_lane"
                ]
        elif rule in _STRUCTURAL_LANES:
            # Proposal-tier fallback: acceptance paths other than the CoE
            # committee leave no accept-record, so "the lane proposed at all"
            # plus a failed final expression justifies one suppressed retry
            # (wasted only on runs that already failed).
            fired[rule] = _STRUCTURAL_LANES[rule]
    if not fired:
        return False, "lane_not_fired", []
    return True, ("unresolved_leaves" if unresolved else "uncertified"), sorted(
        set(fired.values())
    )


def _maybe_retry_without_visible_buckingham(*, report_path, results_dir, base_filename):
    """One-shot provisional-acceptance rollback for the visible-Buckingham lane.

    Re-executes this run in a child process with the lane suppressed
    (NNSR_SUPPRESS_VISIBLE_BUCKINGHAM=1). The child overwrites the run's
    output files; the originals are backed up first and restored if the child
    fails to produce a report. Guarded against recursion via
    NNSR_BUCKINGHAM_RETRY.
    """
    import glob as _glob
    import shutil as _shutil
    import subprocess as _subprocess

    if os.environ.get("NNSR_BUCKINGHAM_RETRY") == "1":
        return
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    trigger, reason, suppress_envs = _buckingham_retry_trigger(report)
    suppress_envs = [e for e in suppress_envs if os.environ.get(e) != "1"]
    if not trigger or not suppress_envs:
        return

    print(
        f"\n[Buckingham-retry] Final expression is {reason} and structural "
        f"lane(s) were active this run; re-running once with suppressed: "
        f"{', '.join(suppress_envs)}."
    )

    backup_dir = os.path.join(results_dir, f".buckingham_backup_{base_filename}")
    try:
        os.makedirs(backup_dir, exist_ok=True)
        to_back_up = set(_glob.glob(os.path.join(results_dir, f"{base_filename}*")))
        to_back_up.add(str(report_path))
        for p in sorted(to_back_up):
            if os.path.isfile(p):
                _shutil.copy2(p, os.path.join(backup_dir, os.path.basename(p)))
    except OSError as e:
        print(f"[Buckingham-retry] Backup failed ({e}); skipping retry for safety.")
        return

    argv = [a for a in sys.argv]
    if "--resume_from" in argv:
        i = argv.index("--resume_from")
        del argv[i : i + 2]
    env = dict(os.environ)
    for var in suppress_envs:
        env[var] = "1"
    env["NNSR_BUCKINGHAM_RETRY"] = "1"
    proc = _subprocess.run([sys.executable] + argv, env=env)

    retry_ok = proc.returncode == 0 and os.path.isfile(report_path)
    if not retry_ok:
        print(
            f"[Buckingham-retry] Retry failed (rc={proc.returncode}); "
            "restoring first-attempt outputs."
        )
        try:
            for p in sorted(_glob.glob(os.path.join(backup_dir, "*"))):
                dest_dir = (
                    os.path.dirname(str(report_path))
                    if os.path.basename(p) == os.path.basename(str(report_path))
                    else results_dir
                )
                _shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))
        except OSError as e:
            print(f"[Buckingham-retry] Restore failed: {e}")
        return

    try:
        with open(report_path) as f:
            retry_report = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[Buckingham-retry] Could not read retry report ({e}); restoring first attempt.")
        retry_report = None

    retry_stagec = (retry_report or {}).get("stageC") or {}
    retry_unresolved = bool((retry_stagec.get("unresolved") or {}).get("unresolved"))
    retry_certified = bool(retry_stagec.get("certified", False)) and not retry_unresolved

    if retry_report is not None and retry_certified:
        # The suppressed run succeeded where the peeled run failed: keep it.
        try:
            retry_report.setdefault("metadata", {})["buckingham_retry"] = {
                "applied": True,
                "kept": "retry",
                "first_attempt_reason": reason,
                "first_attempt_backup": backup_dir,
                "suppressed": list(suppress_envs),
            }
            with open(report_path, "w") as f:
                json.dump(retry_report, f, indent=2)
        except (OSError, TypeError) as e:
            print(f"[Buckingham-retry] Could not annotate retry report: {e}")
        print("[Buckingham-retry] Retry certified; retry outputs kept.")
        return

    # Both attempts failed (or the retry report is unreadable): the peeled first
    # attempt is the better-informed default (the lane is presumed helpful when
    # active), so restore it and keep the retry report inside the backup dir.
    print("[Buckingham-retry] Retry did not certify either; restoring first attempt.")
    try:
        if retry_report is not None:
            with open(os.path.join(backup_dir, "retry_attempt.report.json"), "w") as f:
                json.dump(retry_report, f, indent=2)
    except (OSError, TypeError):
        pass
    try:
        for p in sorted(_glob.glob(os.path.join(backup_dir, "*"))):
            base = os.path.basename(p)
            if base == "retry_attempt.report.json":
                continue
            dest_dir = (
                os.path.dirname(str(report_path))
                if base == os.path.basename(str(report_path))
                else results_dir
            )
            _shutil.copy2(p, os.path.join(dest_dir, base))
        with open(report_path) as f:
            first_report = json.load(f)
        first_report.setdefault("metadata", {})["buckingham_retry"] = {
            "applied": True,
            "kept": "first_attempt",
            "first_attempt_reason": reason,
            "suppressed": list(suppress_envs),
            "retry_report": os.path.join(backup_dir, "retry_attempt.report.json"),
        }
        with open(report_path, "w") as f:
            json.dump(first_report, f, indent=2)
    except (OSError, json.JSONDecodeError, TypeError) as e:
        print(f"[Buckingham-retry] Restore after failed retry hit an error: {e}")


def main():
    start_time = timeit.default_timer()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on {} with device {}".format(socket.gethostname(), device))

    args = parse_args()

    # Blinded mode: forbid all in-process access to the ground-truth answer key.
    # Set a process-wide flag so every ground-truth evaluation short-circuits
    # before opening the canary registry (see sr_search/truth_eval._blinded_active),
    # including any subprocesses (e.g. CoE scouts) that inherit the environment.
    if bool(getattr(args, "blinded", False)):
        os.environ["NESTYNET_SR_BLINDED"] = "1"
        print("[Blinded] Ground-truth access disabled; score afterwards with scripts/score_blinded_run.py")

    # Resolve factorized symbolic search default: on when enforce_units (now the default), off with --ignore_units
    if args.use_factorized_search is None:
        args.use_factorized_search = bool(args.enforce_units)
    if args.use_refine_skeleton is None:
        args.use_refine_skeleton = True
    print(
        f"enforce_units={args.enforce_units}, factorized symbolic search={'ON' if args.use_factorized_search else 'OFF'}, "
        f"continuous skeleton refinement={'ON' if args.use_refine_skeleton else 'OFF'}, "
        f"outer-peel-autorun(deprecated)={'ON' if args.outer_peel_autorun else 'OFF'}"
    )

    filepaths = None
    if args.filepaths is not None and len(args.filepaths) > 0:
        filepaths = [str(p) for p in args.filepaths]
    elif args.filepath is not None:
        filepaths = [str(args.filepath)]
    if filepaths is None or len(filepaths) == 0:
        raise ValueError("Provide either --filepath <csv> or --filepaths <csv1> <csv2> ...")

    # Stage A is run on the first dataset; Stage B (if enabled) can optionally
    # fit across all datasets.
    filepath = filepaths[0]
    source_filepaths = list(filepaths)
    source_filepath = str(filepath)
    statistical_split_plan = None

    def _derive_base_filename(paths: list) -> str:
        if len(paths) == 1:
            return pathlib.Path(paths[0]).stem
        stems = [pathlib.Path(p).stem for p in paths]
        common = os.path.commonprefix(stems).rstrip("_-.")
        if common and len(common) >= 3:
            return f"{common}_multi{len(paths)}"
        return f"multi{len(paths)}_{stems[0]}"

    base_filename = _derive_base_filename(filepaths)
    model_base_filename = base_filename

    # Create output directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)  # Parent of nestynet_sr/
    results_dir = (
        os.path.abspath(os.path.expanduser(str(args.results_dir)))
        if args.results_dir is not None
        else os.path.join(repo_root, "results")
    )
    models_dir = os.path.join(repo_root, "models")
    models_sep_dir = os.path.join(repo_root, "models_sep")

    results_ref_dir = os.path.join(repo_root, "results_ref")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(models_sep_dir, exist_ok=True)
    os.makedirs(results_ref_dir, exist_ok=True)

    # Canonical model paths ("current" stage-A model)
    model_output = os.path.join(models_dir, f"{model_base_filename}.mod")
    model_sep_output = os.path.join(models_sep_dir, f"{model_base_filename}.mod")

    # Per-y-transform model archive paths
    def _model_path(y_name: str, *, sep: bool = False) -> str:
        d = models_sep_dir if sep else models_dir
        return os.path.join(d, f"{model_base_filename}.{y_name}.mod")

    model_output_identity = _model_path("identity", sep=False)
    model_sep_output_identity = _model_path("identity", sep=True)

    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = os.path.join(results_dir, f"{base_filename}.state.pkl")

    # Hyper-parameter configuration
    lm_hp = LMHyperparams()
    model_hp = ModelHyperparams()
    data_hp = DataHyperparams()
    search_hp = SearchHyperparams()
    from nestynet_sr.sr_gs import GeneralizedSymmetryConfig

    gs_enabled = bool(
        getattr(args, "gs_stagea", False)
        or getattr(args, "gs_general_affine", False)
        or getattr(args, "gs_unit_torus", False)
        or getattr(args, "gs_pi_invariants", False)
        or getattr(args, "gs_pairwise_composition", False)
        or getattr(args, "gs_recursive_composition", False)
    )
    search_hp.gs_config = GeneralizedSymmetryConfig(
        enabled=gs_enabled,
        mode=str(getattr(args, "gs_mode", "propose") or "propose"),
        policy=str(getattr(args, "gs_policy", "augment") or "augment"),
        known_generators=bool(getattr(args, "gs_known_generators", True)),
        known_lie=bool(getattr(args, "gs_known_generators", True)),
        general_affine=bool(getattr(args, "gs_general_affine", False)),
        affine_dense=bool(getattr(args, "gs_general_affine", False)),
        general_affine_charts=_parse_gs_charts(getattr(args, "gs_charts", "identity")),
        general_affine_chart_snap_denominator=max(1, int(getattr(args, "gs_chart_snap_denominator", 4) or 4)),
        general_affine_promotion_noise_calibrated=bool(getattr(args, "gs_noise_calibrated_promotion", False)),
        noise_calibrated_min_spectral_gap=float(getattr(args, "gs_nc_min_spectral_gap", 10.0) or 10.0),
        pairwise_composition=bool(getattr(args, "gs_pairwise_composition", False)),
        recursive_composition=bool(getattr(args, "gs_recursive_composition", False)),
        recursive_composition_max_depth=max(1, int(getattr(args, "gs_recursive_max_depth", 3) or 3)),
        recursive_composition_beam_width=max(1, int(getattr(args, "gs_recursive_beam_width", 2) or 2)),
        stagea_proposal_budget=max(0, int(getattr(args, "gs_stagea_proposal_budget", 6) or 0)),
        decisive_stagea_min_confidence=float(getattr(args, "gs_decisive_min_confidence", 0.995)),
        decisive_stagea_max_trials=max(0, int(getattr(args, "gs_decisive_max_trials", 1) or 0)),
        lorentz_boosts=bool(getattr(args, "gs_lorentz_boosts", False)),
        output_equivariance=bool(getattr(args, "gs_output_equivariance", True)),
        residual_tol=float(getattr(args, "gs_residual_tol", 0.03)),
        audit_residual_tol=float(getattr(args, "gs_audit_residual_tol", 0.10)),
        min_confidence=float(getattr(args, "gs_min_confidence", 0.65)),
        max_pair_generators=max(1, int(getattr(args, "gs_max_pair_generators", 16))),
        unit_torus=bool(getattr(args, "gs_unit_torus", False) or getattr(args, "gs_pi_invariants", False)),
        pi_invariants=bool(getattr(args, "gs_pi_invariants", False)),
        dim_policy=str(getattr(args, "gs_dim_policy", "audit") or "audit"),
        dim_both_rule=str(getattr(args, "gs_dim_both_rule", "rref-dominates") or "rref-dominates"),
        dim_validator=str(getattr(args, "gs_dim_validator", "nullspace") or "nullspace"),
        dim_keep_local_gates=bool(getattr(args, "gs_dim_keep_local_gates", True)),
        pi_max_exponent=int(getattr(args, "gs_pi_max_exponent", 3)),
        pi_max_l1=int(getattr(args, "gs_pi_max_l1", 6)),
        pi_max_proposals=int(getattr(args, "gs_pi_max_proposals", 24)),
        pi_max_basis=int(getattr(args, "gs_pi_max_basis", 8)),
        pi_rational_denom=int(getattr(args, "gs_pi_rational_denom", 1)),
        pi_include_free_consts=bool(getattr(args, "gs_pi_include_free_consts", True)),
        unit_report_json=getattr(args, "gs_unit_report_json", None),
        unit_report_md=getattr(args, "gs_unit_report_md", None),
        report_dim_disagreements=bool(getattr(args, "gs_report_dim_disagreements", True)),
        report_pi_rejected=bool(getattr(args, "gs_report_pi_rejected", False)),
    )
    if search_hp.gs_config.active():
        print(
            "[GS] Stage-A symmetry layer enabled "
            f"(mode={search_hp.gs_config.mode}, policy={search_hp.gs_config.canonical_policy()}, "
            f"known={search_hp.gs_config.known_active()}, affine={search_hp.gs_config.general_affine_active()}, "
            f"unit_torus={search_hp.gs_config.unit_torus_active()})"
        )
    factorized_search_hp = FactorizedSearchConfig()
    apply_expr_ir_args_to_obj(args, factorized_search_hp)
    factorized_search_hp.refine_enable = bool(args.use_refine_skeleton)
    if getattr(args, "refine_profile", None) is not None:
        apply_refine_profile(factorized_search_hp, args.refine_profile)
    elif bool(factorized_search_hp.refine_enable):
        apply_refine_profile(factorized_search_hp, "rare_slate")
    if bool(getattr(args, "stageA_continuation_seed", False)):
        continuation_fit_link_raw = getattr(args, "stageA_continuation_fit_link", None)
        if continuation_fit_link_raw is not None:
            try:
                continuation_fit_link = canonical_fit_link_name(continuation_fit_link_raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid --stageA_continuation_fit_link={continuation_fit_link_raw!r}: {exc}"
                ) from exc
            lm_hp.fit_y_link = continuation_fit_link
            if continuation_fit_link is not None:
                try:
                    lm_hp.fit_y_link_scale = float(
                        getattr(args, "stageA_continuation_fit_link_scale", 1.0)
                    )
                except Exception:
                    lm_hp.fit_y_link_scale = 1.0
                print(
                    "[CoE continuation scout] Pinned fit-link context: "
                    f"{describe_fit_link(continuation_fit_link, lm_hp.fit_y_link_scale)}"
                )
    if not bool(getattr(args, "stageA_auto_fit_link", True)):
        lm_hp.auto_fit_link_log_dynamic_range_threshold = float("inf")

    # factorized symbolic search scoring head overrides (multi-term linear head on residual)
    if getattr(args, "factorized_score_head", None) is not None:
        factorized_search_hp.score_head_enable = bool(args.factorized_score_head)
    if getattr(args, "factorized_score_head_omp", None) is not None:
        factorized_search_hp.score_head_omp_enable = bool(args.factorized_score_head_omp)
    if args.factorized_score_head_omp_terms is not None:
        factorized_search_hp.score_head_omp_max_terms = int(args.factorized_score_head_omp_terms)
    if args.factorized_score_head_ridge is not None:
        factorized_search_hp.score_head_ridge = float(args.factorized_score_head_ridge)
    if args.factorized_score_head_min_rel_improve is not None:
        factorized_search_hp.score_head_min_rel_improve = float(args.factorized_score_head_min_rel_improve)

    if args.refine_lbfgs_steps is not None:
        factorized_search_hp.refine_lbfgs_steps = int(args.refine_lbfgs_steps)
    if args.refine_fit_subset is not None:
        factorized_search_hp.refine_fit_subset = int(args.refine_fit_subset)
    if args.refine_num_restarts is not None:
        factorized_search_hp.refine_num_restarts = int(args.refine_num_restarts)
    if getattr(args, "refine_optimizer", None) is not None:
        factorized_search_hp.refine_optimizer = str(args.refine_optimizer)
    if getattr(args, "refine_lbfgs_escalate_improve_factor", None) is not None:
        factorized_search_hp.refine_lbfgs_escalate_improve_factor = float(
            args.refine_lbfgs_escalate_improve_factor
        )
    if args.refine_max_variants is not None:
        factorized_search_hp.refine_max_variants = int(args.refine_max_variants)
    if args.refine_max_params is not None:
        factorized_search_hp.refine_max_params = int(args.refine_max_params)
    if args.use_refine_linear_combo is not None:
        factorized_search_hp.refine_linear_combo_enable = bool(args.use_refine_linear_combo)
    if args.refine_linear_terms_max is not None:
        factorized_search_hp.refine_linear_terms_max = int(args.refine_linear_terms_max)
    if args.refine_linear_prune_rel is not None:
        factorized_search_hp.refine_linear_prune_rel = float(args.refine_linear_prune_rel)
    if args.refine_gate_best_factor is not None:
        factorized_search_hp.refine_gate_best_factor = float(args.refine_gate_best_factor)
    if args.refine_max_trials is not None:
        factorized_search_hp.refine_max_trials = int(args.refine_max_trials)
    if args.refine_trials_per_brute_depth is not None:
        factorized_search_hp.refine_trials_per_brute_depth = int(args.refine_trials_per_brute_depth)
    if args.refine_trials_per_mutation_window is not None:
        factorized_search_hp.refine_trials_per_mutation_window = int(args.refine_trials_per_mutation_window)
    if args.refine_mutation_window is not None:
        factorized_search_hp.refine_mutation_window = int(args.refine_mutation_window)
    if args.refine_safe_eps is not None:
        factorized_search_hp.refine_safe_eps = float(args.refine_safe_eps)
    if args.refine_safe_penalty_weight is not None:
        factorized_search_hp.refine_safe_penalty_weight = float(args.refine_safe_penalty_weight)
    if args.refine_safe_exp_clip is not None:
        factorized_search_hp.refine_safe_exp_clip = float(args.refine_safe_exp_clip)
    if args.refine_theta_l2 is not None:
        factorized_search_hp.refine_theta_l2 = float(args.refine_theta_l2)
    if args.refine_init_log_min is not None:
        factorized_search_hp.refine_init_log_min = float(args.refine_init_log_min)
    if args.refine_init_log_max is not None:
        factorized_search_hp.refine_init_log_max = float(args.refine_init_log_max)
    if args.no_brute_force:
        factorized_search_hp.brute_depth = 0

    # Derive dual_layer from --single_layer flag (dual-layer is the default)
    dual_layer = not args.single_layer
    if dual_layer:
        search_hp.force_dual_layer = True

    # Apply compound detection settings (enabled by default)
    search_hp.enable_compound_detection = not args.disable_compound_detection
    search_hp.compound_max_batches = args.compound_max_batches
    search_hp.verbose_separabilities = args.verbose_separabilities
    search_hp.outer_peel_autorun = args.outer_peel_autorun
    search_hp.ysearch_enable = args.ysearch_enable
    if args.ysearch_depth is not None:
        search_hp.ysearch_depth = max(1, int(args.ysearch_depth))
    if args.ysearch_beam is not None:
        search_hp.ysearch_beam = max(1, int(args.ysearch_beam))
    if args.ysearch_expand_k is not None:
        search_hp.ysearch_expand_k = max(1, int(args.ysearch_expand_k))
    if args.ysearch_portfolio_margin_decades is not None:
        search_hp.ysearch_portfolio_margin_decades = max(0.0, float(args.ysearch_portfolio_margin_decades))
    if args.ysearch_portfolio_max_k is not None:
        search_hp.ysearch_portfolio_max_k = max(1, int(args.ysearch_portfolio_max_k))
    if args.ysearch_min_valid_frac is not None:
        search_hp.ysearch_min_valid_frac = float(args.ysearch_min_valid_frac)
    if args.ysearch_confirm_improve_ratio is not None:
        search_hp.ysearch_confirm_improve_ratio = float(args.ysearch_confirm_improve_ratio)
    if args.ysearch_trigger_trig_affine_conf is not None:
        search_hp.ysearch_trigger_trig_affine_conf = float(args.ysearch_trigger_trig_affine_conf)
    if args.ysearch_trigger_sep_min is not None:
        search_hp.ysearch_trigger_sep_min = float(args.ysearch_trigger_sep_min)
    if args.ysearch_trigger_sep_delta is not None:
        search_hp.ysearch_trigger_sep_delta = float(args.ysearch_trigger_sep_delta)
    if args.ysearch_trigger_split_score is not None:
        search_hp.ysearch_trigger_split_score = float(args.ysearch_trigger_split_score)
    if args.ysearch_trigger_split_margin is not None:
        search_hp.ysearch_trigger_split_margin = float(args.ysearch_trigger_split_margin)
    if args.ysearch_max_virtual_deriv is not None:
        search_hp.ysearch_max_virtual_deriv = float(args.ysearch_max_virtual_deriv)
    if args.ysearch_outer_affine_confirm_rms_rel is not None:
        search_hp.ysearch_outer_affine_confirm_rms_rel = max(0.0, float(args.ysearch_outer_affine_confirm_rms_rel))
    if args.ysearch_outer_affine_min_domain_frac is not None:
        search_hp.ysearch_outer_affine_min_domain_frac = max(0.0, min(1.0, float(args.ysearch_outer_affine_min_domain_frac)))
    if args.ysearch_max_state_evals is not None:
        search_hp.ysearch_max_state_evals = max(0, int(args.ysearch_max_state_evals))
    if args.ysearch_max_recursive_branches is not None:
        search_hp.ysearch_max_recursive_branches = max(
            0, int(args.ysearch_max_recursive_branches)
        )
    if args.ysearch_max_split_plans_per_state is not None:
        search_hp.ysearch_max_split_plans_per_state = max(
            0, int(args.ysearch_max_split_plans_per_state)
        )
    if args.max_ab_iters is not None:
        search_hp.max_ab_iters = args.max_ab_iters
    if args.stageA_max_passes is not None:
        search_hp.stageA_max_passes = max(0, int(args.stageA_max_passes))

    # Apply final-pruning settings
    lm_hp.prune_final_enable = args.prune_final
    lm_hp.stageB_overcap_fallback = bool(args.stageB_overcap_fallback)
    lm_hp.stageB_polish = bool(getattr(args, "stageB_polish", True))
    lm_hp.stageB_polish_max_candidates = max(
        1, int(getattr(args, "stageB_polish_max_candidates", 32) or 32)
    )
    lm_hp.stageB_polish_subtrees = bool(getattr(args, "stageB_polish_subtrees", True))
    lm_hp.stageB_polish_max_subtrees = max(
        1, int(getattr(args, "stageB_polish_max_subtrees", 8) or 8)
    )
    lm_hp.stageB_polish_subprocess = bool(
        getattr(args, "stageB_polish_subprocess", True)
    )
    lm_hp.stageB_polish_max_seconds = float(
        getattr(args, "stageB_polish_max_seconds", 300.0) or 300.0
    )
    lm_hp.stageB_polish_mem_fraction = float(
        getattr(args, "stageB_polish_mem_fraction", 0.20) or 0.20
    )
    lm_hp.stagec_sympy_subprocess = bool(getattr(args, "stageC_sympy_subprocess", True))
    lm_hp.stagec_sympy_max_seconds = float(
        getattr(args, "stageC_sympy_max_seconds", 300.0) or 300.0
    )
    lm_hp.stagec_sympy_mem_fraction = float(
        getattr(args, "stageC_sympy_mem_fraction", 0.20) or 0.20
    )
    lm_hp.final_polish_subprocess = bool(getattr(args, "final_polish_subprocess", True))
    lm_hp.final_polish_worker_max_seconds = float(
        getattr(args, "final_polish_worker_max_seconds", 300.0) or 300.0
    )
    lm_hp.final_polish_worker_mem_fraction = float(
        getattr(args, "final_polish_worker_mem_fraction", 0.20) or 0.20
    )
    lm_hp.coe_mode = str(getattr(args, "coe_mode", "off") or "off")
    lm_hp.coe_stageB_dry_run = bool(
        getattr(args, "coe_stageB_dry_run", False)
        or str(getattr(args, "coe_mode", "off") or "off") in {"committee_gated", "reservoir_discovery"}
    )
    lm_hp.canonical_init = bool(getattr(args, "canonical_init", False))
    lm_hp.evidence_enable = bool(getattr(args, "evidence", False))
    lm_hp.evidence_disable_residual_whitening = bool(
        getattr(args, "evidence_disable_residual_whitening", False)
    )
    lm_hp.evidence_disable_segment_priors = bool(
        getattr(args, "evidence_disable_segment_priors", False)
    )
    if getattr(args, "evidence_lambda_patch", None) is not None:
        lm_hp.evidence_lambda_patch = float(args.evidence_lambda_patch)
    lm_hp.evidence_prior_decay_auto = bool(getattr(args, "evidence_prior_decay_auto", True))
    if getattr(args, "evidence_prior_decay_start", None) is not None:
        lm_hp.evidence_prior_decay_start = int(args.evidence_prior_decay_start)
    if getattr(args, "evidence_prior_decay_interval", None) is not None:
        lm_hp.evidence_prior_decay_interval = int(args.evidence_prior_decay_interval)
    if getattr(args, "evidence_prior_decay_shape", None) is not None:
        lm_hp.evidence_prior_decay_shape = str(args.evidence_prior_decay_shape)
    if getattr(args, "evidence_prior_decay_final_scale", None) is not None:
        lm_hp.evidence_prior_decay_final_scale = float(args.evidence_prior_decay_final_scale)
    if getattr(args, "evidence_prior_cutoff_tol", None) is not None:
        lm_hp.evidence_prior_cutoff_tol = float(args.evidence_prior_cutoff_tol)
    lm_hp.evidence_gate_metrics_until_prior_decay = bool(getattr(args, "evidence_metric_gate", True))
    if args.prune_rel_threshold is not None:
        lm_hp.prune_rel_threshold = float(args.prune_rel_threshold)
    if args.prune_loss_tolerance is not None:
        lm_hp.prune_loss_tolerance = float(args.prune_loss_tolerance)

    if lm_hp.canonical_init:
        print("[Canonical Init] Enabled for pure Stage-A NN teacher fits.")

    if lm_hp.evidence_enable:
        from nestynet_sr.sr_search.training import build_sr_evidence_config

        cfg_preview = build_sr_evidence_config(lm_hp, epochs=lm_hp.epochs)
        if cfg_preview is None:
            print(
                "[Evidence] Requested, but SR-side auxiliary evidence terms are fully disabled; "
                "using the legacy LM path for an exact before/after comparison."
            )
        else:
            decay_bits = []
            if getattr(cfg_preview, "prior_decay_start_iter", None) is not None:
                decay_interval = max(
                    0,
                    int(cfg_preview.prior_decay_end_iter - cfg_preview.prior_decay_start_iter),
                )
                decay_bits.append(
                    f"prior decay start={cfg_preview.prior_decay_start_iter}, interval={decay_interval} "
                    f"(end={cfg_preview.prior_decay_end_iter}) "
                    f"({cfg_preview.prior_decay_shape}, final={cfg_preview.prior_decay_final_scale:g})"
                )
            else:
                decay_bits.append("no prior decay schedule")
            print(
                "[Evidence] Requested with active segment-prior guidance; "
                + ", ".join(decay_bits)
                + ". Unsupported composite SR fits will fall back to plain LM."
            )

    # Apply per-parameter pruning settings
    lm_hp.prune_param_enable = args.prune_param
    if args.prune_param_aic_tolerance is not None:
        lm_hp.prune_param_aic_tolerance = float(args.prune_param_aic_tolerance)

    # Apply fast mode overrides if requested
    if args.fast:
        print("[Fast Mode] Reducing epochs and segments for quick testing")
        lm_hp.epochs = 2000
        lm_hp.epochs_min = 500
        lm_hp.epochs_awful_check = 125  # Proportional to fast mode epochs (2.5% of 5000)
        model_hp.num_segments_max = 24
        model_hp.num_segments_min = 8
        model_hp.model_size_target = 500

    # Optional data-size overrides (useful for quick local smoke tests).
    if args.ndata_train is not None:
        data_hp.ndata_select = int(args.ndata_train)
    if args.ndata_val is not None:
        data_hp.ndata_select_val = int(args.ndata_val)
    data_hp.data_slice = int(getattr(args, "data_slice", 0) or 0)
    if args.batch_size is not None:
        data_hp.batch_size = int(args.batch_size)
    elif args.ndata_train is not None or args.ndata_val is not None:
        # If only ndata_* is overridden, clamp batch_size so dataloaders remain valid.
        data_hp.batch_size = int(min(data_hp.batch_size, data_hp.ndata_select, data_hp.ndata_select_val))
    if (
        args.batch_size is not None
        or args.ndata_train is not None
        or args.ndata_val is not None
        or data_hp.data_slice != 0
    ):
        print(
            "[Data] Overrides: "
            f"batch_size={data_hp.batch_size}, "
            f"ndata_train={data_hp.ndata_select}, "
            f"ndata_val={data_hp.ndata_select_val}, "
            f"data_slice={data_hp.data_slice}"
        )

    if bool(getattr(args, "stat_selection", False)):
        if len(filepaths) != 1:
            raise ValueError(
                "installment 2 statistical selection supports one ordinary-SR CSV; "
                "multi-dataset certification is deferred to a later adapter"
            )
        if bool(getattr(args, "discover_de", False)):
            raise ValueError(
                "--stat_selection with --discover_de is deferred to installment 3, "
                "where trajectories become the independent audit units"
            )
        if bool(getattr(args, "discovery_enable", False)):
            raise ValueError(
                "--stat_selection with --discovery_enable is deferred to installment 3; "
                "the ordinary-SR archive does not yet certify discovery-pipeline candidates"
            )
        if not bool(getattr(args, "stageB", True)):
            raise ValueError("--stat_selection requires Stage B to produce portable symbolic candidates")
        if getattr(args, "load_expressions", None):
            raise ValueError(
                "--stat_selection cannot certify an unprovenanced --load_expressions "
                "artifact; rerun the proposal search behind the audit firewall"
            )
        if str(getattr(args, "coe_reservoir_paths", "") or "").strip():
            raise ValueError(
                "--stat_selection cannot ingest unprovenanced external CoE reservoirs; "
                "regenerate them behind the current audit firewall"
            )
        _stat_audit_rows = int(getattr(args, "stat_audit_rows", 0))
        if _stat_audit_rows < 0:
            raise ValueError("--stat-audit-rows must be nonnegative")
        _stat_audit_fraction = float(getattr(args, "stat_audit_fraction", 0.2))
        if getattr(args, "stat_audit_filepath", None) is None and (
            not math.isfinite(_stat_audit_fraction)
            or not 0.0 < _stat_audit_fraction < 1.0
        ):
            raise ValueError("--stat-audit-fraction must lie strictly between zero and one")
        _stat_unit_size = int(getattr(args, "stat_unit_size", 1))
        if _stat_unit_size < 1:
            raise ValueError("--stat-unit-size must be a positive integer")
        _stat_alpha = float(getattr(args, "stat_alpha", 0.05))
        if not math.isfinite(_stat_alpha) or not 0.0 < _stat_alpha < 1.0:
            raise ValueError("--stat-alpha must lie strictly between zero and one")
        _stat_delta = float(getattr(args, "stat_delta", 0.0))
        if not math.isfinite(_stat_delta) or _stat_delta < 0.0:
            raise ValueError("--stat-delta must be finite and nonnegative")
        _stat_resamples = int(getattr(args, "stat_resamples", 4000))
        if _stat_resamples < 1:
            raise ValueError("--stat-resamples must be a positive integer")
        _stat_seed = int(getattr(args, "stat_seed", 12345))
        _stat_multiplier = str(getattr(args, "stat_multiplier", "normal")).strip().lower()
        if _stat_multiplier not in {"normal", "rademacher"}:
            raise ValueError("--stat-multiplier must be 'normal' or 'rademacher'")
        _stat_max_candidates = int(getattr(args, "stat_max_candidates", 1024))
        if _stat_max_candidates < 1:
            raise ValueError("--stat-max-candidates must be a positive integer")
        _stat_failure_loss = float(getattr(args, "stat_failure_loss", 1.0e6))
        if not math.isfinite(_stat_failure_loss) or _stat_failure_loss <= 0.0:
            raise ValueError("--stat-failure-loss must be positive and finite")
        _stat_x_cov_path = getattr(args, "stat_x_cov_npz", None)
        _stat_x_cov_sha256 = None
        if _stat_x_cov_path:
            _stat_x_cov_path = os.path.abspath(os.path.expanduser(str(_stat_x_cov_path)))
            if not os.path.isfile(_stat_x_cov_path):
                raise FileNotFoundError(_stat_x_cov_path)
            _hash = hashlib.sha256()
            with open(_stat_x_cov_path, "rb") as _cov_file:
                for _chunk in iter(lambda: _cov_file.read(1 << 20), b""):
                    _hash.update(_chunk)
            _stat_x_cov_sha256 = _hash.hexdigest()
            args.stat_x_cov_npz = _stat_x_cov_path
        _stat_x_sigma = getattr(args, "stat_x_sigma", None)
        if _stat_x_cov_path and _stat_x_sigma is not None:
            raise ValueError("use either --stat-x-sigma or --stat-x-cov-npz, not both")
        if _stat_x_sigma is not None:
            _sigma_values = [float(v.strip()) for v in str(_stat_x_sigma).split(",") if v.strip()]
            if not _sigma_values or any((not math.isfinite(v) or v < 0.0) for v in _sigma_values):
                raise ValueError("--stat-x-sigma must contain finite nonnegative values")
        _minimum_search_rows = max(
            2,
            (int(getattr(data_hp, "data_slice", 0) or 0) + 1)
            * (
                int(getattr(data_hp, "ndata_select", 0) or 0)
                + int(getattr(data_hp, "ndata_select_val", 0) or 0)
            ),
        )
        statistical_split_plan = prepare_sr_audit_plan(
            source_filepath,
            results_dir=results_dir,
            external_audit_path=getattr(args, "stat_audit_filepath", None),
            audit_rows=_stat_audit_rows,
            audit_fraction=_stat_audit_fraction,
            minimum_search_rows=_minimum_search_rows,
            minimum_audit_rows=2,
            unit_size=_stat_unit_size,
        )
        filepath = statistical_split_plan.search_path
        filepaths = [filepath]
        args.filepath = filepath
        args.filepaths = list(filepaths)
        # Keep the sealed audit path out of downstream search configuration.
        # The private split plan remains the sole handle used after archive freeze.
        args.stat_audit_filepath = None
        model_base_filename = (
            f"{base_filename}.stat-search-n{statistical_split_plan.search_rows}."
            f"{statistical_split_plan.search_sha256[:12]}"
        )
        model_output = os.path.join(models_dir, f"{model_base_filename}.mod")
        model_sep_output = os.path.join(
            models_sep_dir, f"{model_base_filename}.mod"
        )
        model_output_identity = _model_path("identity", sep=False)
        model_sep_output_identity = _model_path("identity", sep=True)
        if args.checkpoint_path is None:
            checkpoint_path = os.path.join(
                results_dir, f"{model_base_filename}.state.pkl"
            )
        os.environ["NESTYNET_SR_STAT_SEARCH_VIEW"] = filepath
        os.environ["NESTYNET_SR_STAT_SOURCE_SHA256"] = statistical_split_plan.source_sha256
        print(
            "[StatSelection] Search/audit firewall established: "
            f"search_rows={statistical_split_plan.search_rows}, "
            f"audit_rows={statistical_split_plan.audit_rows}, "
            f"kind={statistical_split_plan.audit_kind}"
        )
        print(f"[StatSelection] search view: {statistical_split_plan.search_path}")
        print(f"[StatSelection] sealed audit view: {statistical_split_plan.audit_path}")

    # Configure logging from command-line arguments
    if args.log_file is not None:
        lm_hp.log_file = args.log_file
    # log_to_console defaults to True, but can be set to False with --no_log_to_console
    lm_hp.log_to_console = args.log_to_console
    if args.log_level is not None:
        import logging

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        lm_hp.log_level = level_map.get(args.log_level.upper(), logging.INFO)
    lm_hp.LM_verbose = args.lm_verbose

    # Parse ablation switches (disabled patterns)
    disabled_patterns = []
    if args.disable_stageB_patterns:
        disabled_patterns = [p.strip() for p in args.disable_stageB_patterns.split(",")]
        print(f"[Ablation] Disabling Stage B patterns: {disabled_patterns}")

    # Precision choice
    if model_hp.double_precision:
        dtype = torch.float64
        np_dtype = np.float64
    else:
        dtype = torch.float32
        np_dtype = np.float32

    # Reproducibility
    iseed = None
    if model_hp.repeatable_runs:
        iseed = 1234
        random.seed(iseed)
        np.random.seed(iseed)
        prng_global = np.random.default_rng(iseed)
        torch.manual_seed(iseed)

    # Inspect data (Stage-A dataset) and ensure multi-dataset consistency
    x_data_first, y_data_first, Nxvars = nestynet.dataloader.get_csv_data_as_pandas(filepath)
    x_data_all = [x_data_first]
    y_data_all = [y_data_first]
    if len(filepaths) > 1:
        for fp in filepaths[1:]:
            x_data_i, y_data_i, Nx2 = nestynet.dataloader.get_csv_data_as_pandas(fp)
            if int(Nx2) != int(Nxvars):
                raise ValueError(
                    f"Nxvars mismatch: {pathlib.Path(filepath).name} has Nxvars={Nxvars} but {pathlib.Path(fp).name} has Nxvars={Nx2}"
                )
            x_data_all.append(x_data_i)
            y_data_all.append(y_data_i)
    x_data = np.concatenate(x_data_all, axis=0) if len(x_data_all) > 1 else x_data_first
    x_data = np.asarray(x_data, dtype=np.float64)
    y_data = np.concatenate(y_data_all, axis=0) if len(y_data_all) > 1 else y_data_first
    y_data = np.asarray(y_data, dtype=np.float64).reshape(-1)
    y_med = np.median(y_data)
    print("Median of y values: {}".format(y_med))
    y_mad = np.median(np.abs(y_data - y_med))
    print("Median absolute deviation of y values: {}".format(y_mad))
    print("Number of x variables: {}".format(Nxvars))

    print("MAD^2 of y values: {}".format(y_mad**2))
    y_rms = float(np.sqrt(np.mean(y_data * y_data))) if y_data.size > 0 else 0.0
    print("RMS of y values: {}".format(y_rms))
    noise_sigma_y = None
    if args.noise_sigma_frac_y_rms is not None:
        noise_sigma_y = float(args.noise_sigma_frac_y_rms) * y_rms
        _noise_scope = "search-visible" if statistical_split_plan is not None else "full-dataset"
        _noise_rms_label = "RMS(y_search)" if statistical_split_plan is not None else "RMS(y_full)"
        print(
            f"[Noise] Using {_noise_scope} homoscedastic noise model: "
            f"sigma_y = {float(args.noise_sigma_frac_y_rms):.6g} * "
            f"{_noise_rms_label} = {noise_sigma_y:.6g}"
        )

    # CoE mode metadata for Stage-B hooks.  Normal mode keeps these inert.
    lm_hp.coe_mode = str(getattr(args, "coe_mode", "off") or "off")
    lm_hp.coe_stageB_dry_run = bool(
        getattr(args, "coe_stageB_dry_run", False)
        or lm_hp.coe_mode in {"committee_gated", "reservoir_discovery"}
    )
    lm_hp.coe_filepath = str(filepath) if len(filepaths) == 1 else None
    lm_hp.coe_num_slices = max(0, int(getattr(args, "coe_num_slices", 25) or 0))
    lm_hp.coe_start_slice = max(0, int(getattr(args, "coe_start_slice", 0) or 0))
    lm_hp.coe_reference_slice = max(0, int(getattr(data_hp, "data_slice", 0) or 0))
    lm_hp.coe_stageB_gate_slices = max(
        0, int(getattr(args, "coe_stageB_gate_slices", 5) or 0)
    )
    lm_hp.coe_stageB_initial_gate_slices = max(
        1, int(getattr(args, "coe_stageB_initial_gate_slices", 3) or 1)
    )
    lm_hp.coe_stageB_refit_gate = not bool(
        getattr(args, "no_coe_stageB_refit_gate", False)
    )
    lm_hp.coe_stageB_refit_epochs = max(
        1, int(getattr(args, "coe_stageB_refit_epochs", 200) or 1)
    )
    lm_hp.coe_stageB_refit_escalate_epochs = max(
        0, int(getattr(args, "coe_stageB_refit_escalate_epochs", 0) or 0)
    )
    lm_hp.coe_stageA_compound_shortlist_k = max(
        1, int(getattr(args, "coe_stageA_compound_shortlist_k", 3) or 3)
    )
    lm_hp.coe_stageA_split_near_floor_mult = max(
        0.0, float(getattr(args, "coe_stageA_split_near_floor_mult", 25.0) or 25.0)
    )
    lm_hp.coe_stageA_fit_tournament = bool(
        getattr(args, "coe_stageA_fit_tournament", False)
    )
    if lm_hp.coe_stageA_fit_tournament and not lm_hp.canonical_init:
        raise ValueError("--coe_stageA_fit_tournament requires --canonical_init")
    _fit_alpha = float(getattr(args, "coe_stageA_fit_alpha", 0.05))
    _fit_compare_fraction = float(
        getattr(args, "coe_stageA_fit_comparison_fraction", 0.5)
    )
    _fit_min_rel = float(
        getattr(args, "coe_stageA_fit_min_rel_improvement", 0.01)
    )
    if not 0.0 < _fit_alpha < 1.0:
        raise ValueError("--coe_stageA_fit_alpha must lie strictly between 0 and 1")
    if not 0.0 < _fit_compare_fraction < 1.0:
        raise ValueError(
            "--coe_stageA_fit_comparison_fraction must lie strictly between 0 and 1"
        )
    if not math.isfinite(_fit_min_rel) or _fit_min_rel < 0.0:
        raise ValueError("--coe_stageA_fit_min_rel_improvement must be finite and nonnegative")
    _fit_slices_arg = getattr(args, "coe_stageA_fit_slices", None)
    if _fit_slices_arg:
        lm_hp.coe_stageA_fit_slices = str(_fit_slices_arg)
    else:
        lm_hp.coe_stageA_fit_slices = _parse_coe_scout_slice_ids(args)
    if lm_hp.coe_stageA_fit_tournament:
        from nestynet_sr.sr_search.stagea_fit_tournament import (
            validate_stageA_fit_slice_firewall,
        )

        validate_stageA_fit_slice_firewall(
            lm_hp.coe_stageA_fit_slices,
            witness_start=int(getattr(lm_hp, "coe_start_slice", 0) or 0),
            witness_count=int(getattr(lm_hp, "coe_num_slices", 0) or 0),
        )
    lm_hp.coe_stageA_fit_alpha = _fit_alpha
    lm_hp.coe_stageA_fit_comparison_fraction = _fit_compare_fraction
    lm_hp.coe_stageA_fit_min_rel_improvement = _fit_min_rel
    lm_hp.coe_scout_parallelism = max(
        1, int(getattr(args, "coe_scout_parallelism", 1) or 1)
    )
    lm_hp.coe_scout_timeout_seconds = max(
        0.0, float(getattr(args, "coe_scout_timeout_seconds", 0.0) or 0.0)
    )
    lm_hp.coe_stageA_fit_tournament_records = []
    lm_hp.coe_ndata_train = max(1, int(getattr(data_hp, "ndata_select", 2000) or 2000))
    lm_hp.coe_ndata_val = max(1, int(getattr(data_hp, "ndata_select_val", 2000) or 2000))
    _coe_noise_floor_raw = 0.0
    try:
        sigma = float(noise_sigma_y)
        if math.isfinite(sigma) and sigma > 0.0:
            _coe_noise_floor_raw = float(sigma * sigma)
    except Exception:
        _coe_noise_floor_raw = 0.0
    lm_hp.coe_noise_floor_raw = float(_coe_noise_floor_raw)
    lm_hp.coe_noise_mult = float(getattr(args, "coe_noise_mult", 3.0) or 3.0)
    lm_hp.coe_rel_tol = float(getattr(args, "coe_rel_tol", 1.0e-3) or 1.0e-3)
    lm_hp.coe_inference = str(getattr(args, "coe_inference", "legacy") or "legacy")
    lm_hp.coe_maxt_seed = int(getattr(args, "coe_maxt_seed", 0) or 0)
    lm_hp.coe_min_valid_fraction = float(
        getattr(args, "coe_min_valid_fraction", 0.80) or 0.80
    )
    lm_hp.coe_witness_parallelism = max(
        1, int(getattr(args, "coe_witness_parallelism", 1) or 1)
    )

    # Compute log-space dynamic range for auto fit-link detection
    y_abs = np.abs(y_data)
    y_abs_max = np.max(y_abs)
    y_abs_median = np.median(y_abs)
    y_abs_min_nonzero = np.min(y_abs[y_abs > 0]) if np.any(y_abs > 0) else 1e-30
    y_log_dynamic_range = np.log10(y_abs_max) - np.log10(y_abs_min_nonzero)  # In decades
    print(f"Log-space dynamic range of |y|: {y_log_dynamic_range:.2f} decades")

    # Units straightjacket (optional)
    units_payload = None
    if (
        args.units
        or args.y_units
        or args.x_units
        or args.equations_txt
        or args.free_consts
        or args.local_consts
        or args.global_consts
        or args.fixed_consts
    ):
        y_vec = None
        x_mat = None
        basis_hint = None

        if args.units:
            y_vec, x_mat, basis_hint = _parse_units_arg(args.units)

        if args.y_units:
            y_vec = _parse_py_or_json_literal(args.y_units)

        if args.x_units:
            x_mat = _parse_py_or_json_literal(args.x_units)

        if (y_vec is None or x_mat is None) and args.equations_txt:
            y2, x2 = _load_units_from_equations(args.equations_txt, base_filename)
            if y_vec is None:
                y_vec = y2
            if x_mat is None:
                x_mat = x2

        if (y_vec is not None) and (x_mat is not None):
            try:
                y_vec = list(y_vec)
            except Exception:
                raise ValueError(f"y_units must be a sequence, got: {type(y_vec)}")

            try:
                x_mat = [list(u) for u in x_mat]
            except Exception:
                raise ValueError(f"x_units must be a sequence of sequences, got: {type(x_mat)}")

            if len(x_mat) != int(Nxvars):
                raise ValueError(f"x_units has {len(x_mat)} entries but Nxvars={Nxvars}")

            n_basis = len(y_vec)
            if any(len(u) != n_basis for u in x_mat):
                raise ValueError(f"All x_units vectors must have length {n_basis}")

            from nestynet_sr.sr_core.units import UnitSystem

            basis = _infer_units_basis(
                n_basis, args.units_basis if basis_hint is None else basis_hint
            )
            us = UnitSystem(base=basis)
            y_dim = us.dim(y_vec)
            x_dims = tuple(us.dim(u) for u in x_mat)

            free_const_dims = {}
            free_const_scope = {}   # name -> "experiment" | "class"
            sources_by_const_name = {}

            def _record_const_source(name: str, src: str):
                nm = str(name)
                sources_by_const_name.setdefault(nm, []).append(src)

            if args.free_consts:
                fc_map = _parse_py_or_json_literal(args.free_consts)
                if not isinstance(fc_map, dict):
                    raise ValueError("--free_consts must parse to a dict {name: unit_vec}")
                for name in fc_map:
                    _record_const_source(name, "free_consts")
                for name, vec in fc_map.items():
                    free_const_dims[str(name)] = us.dim(vec)

            # --local_consts / --global_consts → merge into free_const_dims with scope
            if args.local_consts:
                lc_map = _parse_py_or_json_literal(args.local_consts)
                if not isinstance(lc_map, dict):
                    raise ValueError("--local_consts must parse to a dict {name: unit_vec}")
                for name in lc_map:
                    _record_const_source(name, "local_consts")
                for name, vec in lc_map.items():
                    nm = str(name)
                    free_const_dims[nm] = us.dim(vec)
                    free_const_scope[nm] = "experiment"
            if args.global_consts:
                gc_map = _parse_py_or_json_literal(args.global_consts)
                if not isinstance(gc_map, dict):
                    raise ValueError("--global_consts must parse to a dict {name: unit_vec}")
                for name in gc_map:
                    _record_const_source(name, "global_consts")
                for name, vec in gc_map.items():
                    nm = str(name)
                    free_const_dims[nm] = us.dim(vec)
                    free_const_scope[nm] = "class"

            fixed_const_dims = {}
            fixed_const_values = {}
            if args.fixed_consts:
                fx_map = _parse_py_or_json_literal(args.fixed_consts)
                if not isinstance(fx_map, dict):
                    raise ValueError("--fixed_consts must parse to a dict {name: (value, unit_vec)}")
                for name in fx_map:
                    _record_const_source(name, "fixed_consts")

                collisions = [
                    (nm, srcs)
                    for nm, srcs in sources_by_const_name.items()
                    if len(srcs) > 1
                ]
                if collisions:
                    collision_str = ", ".join(
                        f"{nm} ({'/'.join(srcs)})" for nm, srcs in sorted(collisions)
                    )
                    raise ValueError(
                        "Constant name collisions across declarations are not allowed: "
                        f"{collision_str}"
                    )

                for name, payload in fx_map.items():
                    nm = str(name)
                    val = None
                    vec = None
                    # Accept either {"c": [value, unit_vec]} or {"c": {"value": ..., "units": [...]}}
                    if isinstance(payload, dict):
                        val = payload.get("value", payload.get("val", None))
                        vec = payload.get("units", payload.get("unit_vec", payload.get("unit", None)))
                    elif isinstance(payload, (list, tuple)) and len(payload) == 2:
                        val, vec = payload[0], payload[1]
                    else:
                        raise ValueError(
                            "--fixed_consts entries must be either [value, unit_vec] or {value:..., units:[...]}"
                        )
                    if val is None or vec is None:
                        raise ValueError(
                            f"--fixed_consts missing value/units for {nm!r}: got {payload}"
                        )
                    fixed_const_values[nm] = float(val)
                    fixed_const_dims[nm] = us.dim(vec)
            else:
                collisions = [
                    (nm, srcs)
                    for nm, srcs in sources_by_const_name.items()
                    if len(srcs) > 1
                ]
                if collisions:
                    collision_str = ", ".join(
                        f"{nm} ({'/'.join(srcs)})" for nm, srcs in sorted(collisions)
                    )
                    raise ValueError(
                        "Constant name collisions across declarations are not allowed: "
                        f"{collision_str}"
                    )
            units_payload = dict(
                unit_system=us,
                y_dim=y_dim,
                x_dims=x_dims,
                free_const_dims=free_const_dims,
                free_const_scope=free_const_scope,
                fixed_const_dims=fixed_const_dims,
                fixed_const_values=fixed_const_values,
                fixed_const_mode=str(args.fixed_consts_mode),
                basis=basis,
                raw_y_units=y_vec,
                raw_x_units=x_mat,
            )
            print(f"[Units] Loaded basis={basis}; y_units={y_vec}; x_units={x_mat}")
        elif args.free_consts or args.local_consts or args.global_consts or args.fixed_consts:
            raise ValueError(
                "Constant unit declarations (--free_consts/--local_consts/--global_consts/"
                "--fixed_consts) require a units spec for y/x. "
                "Provide --units or --y_units/--x_units, or --equations_txt."
            )
        elif args.enforce_units:
            raise ValueError(
                "Units are enforced by default but y_units/x_units could not be loaded. "
                "Provide --units or --y_units/--x_units, or --equations_txt, "
                "or pass --ignore_units to skip dimensional analysis."
            )

    if args.enforce_units and units_payload is None:
        raise ValueError(
            "Units are enforced by default but no units spec was found. "
            "Provide --units or --y_units/--x_units, or --equations_txt, "
            "or pass --ignore_units to skip dimensional analysis."
        )

    phase_hints = []
    phase_context_hints = []
    outer_link_hints = []
    if bool(getattr(search_hp, "phase_prescan_enabled", True)):
        try:
            from nestynet_sr.sr_search.phase_scan import (
                PhaseScanHyperparams,
                format_outer_link_hint,
                format_phase_context_hint,
                format_phase_hint,
                run_phase_context_scan,
                run_outer_inverse_trig_prescan,
                run_phase_prescan,
            )

            phase_hp = PhaseScanHyperparams(
                enabled=bool(getattr(search_hp, "phase_prescan_enabled", True)),
                sample_size=int(getattr(search_hp, "phase_prescan_sample_size", 4096)),
                max_support=int(getattr(search_hp, "phase_prescan_max_support", 3)),
                max_candidates=int(getattr(search_hp, "phase_prescan_max_candidates", 96)),
                max_candidates_per_support=int(
                    getattr(search_hp, "phase_prescan_max_candidates_per_support", 32)
                ),
                max_exp_l1=float(getattr(search_hp, "phase_prescan_max_exp_l1", 4.0)),
                min_domain_frac=float(getattr(search_hp, "phase_prescan_min_domain_frac", 0.98)),
                log_top_k=int(getattr(search_hp, "phase_prescan_log_top_k", 6)),
                context_enabled=bool(getattr(search_hp, "phase_context_enabled", True)),
                context_max_features=int(getattr(search_hp, "phase_context_max_features", 8)),
                context_log_top_k=int(getattr(search_hp, "phase_context_log_top_k", 6)),
            )
            x_phase, y_phase = _phase_prescan_training_arrays(
                filepaths,
                Nxvars=Nxvars,
                np_dtype=np_dtype,
                data_hp=data_hp,
            )
            if x_phase is None or y_phase is None:
                phase_hints = []
                print("[PhaseScan] skipped: could not build identity training split for hint generation.")
            else:
                phase_hints = run_phase_prescan(
                    x_phase,
                    y_phase,
                    Nxvars=int(Nxvars),
                    units_payload=units_payload,
                    ignore_units=not bool(args.enforce_units),
                    hp=phase_hp,
                )
            if phase_hints:
                nlog = max(0, int(getattr(phase_hp, "log_top_k", 6)))
                print(
                    f"[PhaseScan] Found {len(phase_hints)} phase-coordinate hint(s) "
                    "from the identity training split."
                )
                for hint in phase_hints[:nlog]:
                    print(f"[PhaseScan]   {format_phase_hint(hint)}")
            elif x_phase is not None and y_phase is not None:
                print("[PhaseScan] No phase-coordinate hints found.")
            phase_context_hints = []
            if x_phase is not None and y_phase is not None and bool(getattr(phase_hp, "context_enabled", True)):
                phase_context_hints = run_phase_context_scan(
                    x_phase,
                    y_phase,
                    Nxvars=int(Nxvars),
                    units_payload=units_payload,
                    ignore_units=not bool(args.enforce_units),
                    hp=phase_hp,
                )
                if phase_context_hints:
                    nlog_ctx = max(0, int(getattr(phase_hp, "context_log_top_k", 6)))
                    print(
                        f"[PhaseScan Context] Found {len(phase_context_hints)} contextual phase hint(s) "
                        "from the identity training split (proposal evidence only)."
                    )
                    for hint in phase_context_hints[:nlog_ctx]:
                        print(f"[PhaseScan Context]   {format_phase_context_hint(hint)}")
                else:
                    print("[PhaseScan Context] No contextual phase hints found.")
            setattr(search_hp, "phase_context_hints", list(phase_context_hints or []))
            outer_link_hints = []
            if x_phase is not None and y_phase is not None:
                outer_link_hints = run_outer_inverse_trig_prescan(
                    x_phase,
                    y_phase,
                    Nxvars=int(Nxvars),
                    units_payload=units_payload,
                    ignore_units=not bool(args.enforce_units),
                    hp=phase_hp,
                )
                if outer_link_hints:
                    nlog_outer = max(0, int(getattr(phase_hp, "log_top_k", 6)))
                    print(
                        f"[OuterLinkScan] Found {len(outer_link_hints)} inverse-link hint(s) "
                        "from the identity training split (proposal evidence only)."
                    )
                    for hint in outer_link_hints[:nlog_outer]:
                        print(f"[OuterLinkScan]   {format_outer_link_hint(hint)}")
                else:
                    print("[OuterLinkScan] No inverse-link hints found.")
            setattr(search_hp, "outer_link_hints", list(outer_link_hints or []))
        except Exception as e:
            phase_hints = []
            setattr(search_hp, "phase_context_hints", [])
            outer_link_hints = []
            setattr(search_hp, "outer_link_hints", [])
            print(f"[PhaseScan] skipped: {type(e).__name__}: {e}")
    setattr(search_hp, "phase_hints", list(phase_hints or []))
    setattr(search_hp, "outer_link_hints", list(outer_link_hints or []))

    # The LM/search thresholds are now interpreted in units of MAD(φ(y))^2
    # and are scaled per y-transform inside run_separability_for_transform.
    if lm_hp.loss_in_MAD_units:
        print(
            "\nCriteria (base, in units of MAD(φ(y))^2): "
            "loss acceptable: {:e}, loss target: {:e}, chisq_tol: {:e}, "
            "precision derivs d2y: {:e}\n".format(
                lm_hp.loss_acceptable,
                lm_hp.loss_target,
                lm_hp.chisq_tol,
                search_hp.precision_derivs_d2y,
            )
        )
    else:
        print(
            "\nCriteria (base, raw units of y^2): "
            "loss acceptable: {:e}, loss target: {:e}, chisq_tol: {:e}, "
            "precision derivs d2y: {:e}\n".format(
                lm_hp.loss_acceptable,
                lm_hp.loss_target,
                lm_hp.chisq_tol,
                search_hp.precision_derivs_d2y,
            )
        )

    # Load initial expressions if provided
    if args.load_expressions:
        with open(args.load_expressions, "rb") as f:
            loaded_data = pickle.load(f)
            # Expect AST Node format
            if isinstance(loaded_data, (list, tuple)):
                raise ValueError(
                    f"Legacy prefix expression format not supported. "
                    f"File {args.load_expressions} contains old format. "
                    "Please use a file with AST format or build expression directly in code."
                )
            else:
                initial_ast = loaded_data
    else:
        # Default: single NN covering all variables
        initial_ast = build_initial_ast(
            Nxvars, num_segments=model_hp.num_segments_min, dual_layer=dual_layer
        )
        # Examples for custom initial expressions:
        # from sr_core import Mul, AtomNode
        # initial_ast = Mul(Mul(AtomNode('nn', (0,)), AtomNode('nn', (1,))), AtomNode('nn', (2,)))
        # initial_ast = Mul(AtomNode('nn', (0,)), AtomNode('nn', (1, 2, 3, 4)))
        # initial_ast = Mul(AtomNode('nn', (0,)), AtomNode('nn', (1, 2)))

    coe_stageA_external_reservoir = None
    coe_stageA_materialization = None
    coe_stageA_pre_scout_result = None
    coe_stageA_pre_scout_reservoir = None
    coe_stageA_pre_scout_summary = None
    coe_stageA_continuation_scout_summaries: list[dict[str, Any]] = []
    coe_scout_expression_reservoir = None

    # y-transform registry
    y_transform_names = [
        "identity",  # always run first
        # Tier 1: simple monotone, very high payoff
        "square",  # handles sqrt / distance / energy-style outputs: y^2 is polynomial
        "log",  # power laws / exponentials: log(y) ~ linear/additive in x
        "logneg",  # y < 0, product of powers: log(-y) becomes additive
        "reciprocal",  # 1/y for rational relations
        # Tier 2: secondary monotone
        "sqrt",  # y = g(x)^2 ⇒ sqrt(y) = |g|; usually square is better, but keep it
        "sqrt1p",  # y = sqrt(1 + f(x)) ⇒ y² - 1 = f(x); multiplicative inner
        "exp",  # y = log(f(x)) ⇒ exp(y) = f(x); rarer but occasionally useful
        "expneg",  # y = -log(f(x)) ⇒ exp(-y) = f(x)
        # Tier 3: angle-like stuff (inverse trig first, then direct)
        "arctan",
        "arcsin",
        "arccos",
        # Tier 4: "wilder" non-monotone trigs on y itself
        "sin",
        "cos",
        "tan",
    ]
    available_y_transform_names = list(y_transform_names)

    # Allow user to force specific y-transforms via CLI
    if args.force_y_ops is not None:
        requested_ops = [s.strip() for s in args.force_y_ops.split(",")]
        # Validate requested transforms
        invalid = [op for op in requested_ops if op not in y_transform_names]
        if invalid:
            print(f"[WARNING] Invalid y-transform(s): {invalid}")
            print(f"[WARNING] Valid options: {', '.join(y_transform_names)}")
            print("[WARNING] Proceeding with valid transforms only")
        # Filter to requested transforms (preserving order from y_transform_names)
        y_transform_names = [name for name in y_transform_names if name in requested_ops]
        # Always include identity at end for baseline comparison and fallback
        if "identity" not in y_transform_names:
            y_transform_names = y_transform_names + ["identity"]
        if not y_transform_names:
            print("[ERROR] No valid y-transforms selected. Using 'identity' as fallback.")
            y_transform_names = ["identity"]
        print(f"[y-transforms] Forced to: {', '.join(y_transform_names)}")

    coe_stageA_pre_scout_result = _run_coe_scout_proposers(
        args=args,
        filepath=filepath,
        results_dir=results_dir,
        base_filename=base_filename,
        current_stageA_reservoir=None,
        current_reservoir=None,
    )
    if isinstance(coe_stageA_pre_scout_result, dict):
        coe_stageA_pre_scout_summary = coe_stageA_pre_scout_result.get("summary")
        if isinstance(coe_stageA_pre_scout_summary, dict):
            coe_stageA_pre_scout_summary["phase"] = "pre_stageA_materialization"
            print(
                "[CoE scouts] pre-Stage-A completed="
                f"{coe_stageA_pre_scout_summary.get('completed', 0)} "
                f"loaded_stageA_payloads={coe_stageA_pre_scout_summary.get('loaded_stageA_payloads', 0)}"
            )
        coe_stageA_pre_scout_reservoir = coe_stageA_pre_scout_result.get(
            "merged_stageA_reservoir"
        )
        coe_scout_expression_reservoir = _merge_coe_expression_reservoir_payload(
            coe_scout_expression_reservoir,
            coe_stageA_pre_scout_result.get("merged_reservoir"),
            max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 4),
            source="pre_stageA_scout_expression_reservoir",
        )

    coe_stageA_external_reservoir, coe_stageA_materialization = (
        _load_coe_external_stageA_proposal_reservoir(args=args, filepath=source_filepath)
    )
    _materialization_payload = coe_stageA_external_reservoir
    if isinstance(coe_stageA_pre_scout_reservoir, dict):
        try:
            from nestynet_sr.sr_search.coe_committee import (
                merge_stageA_proposal_reservoir_payloads,
            )

            _stageA_materialization_inputs = []
            if isinstance(coe_stageA_external_reservoir, dict):
                _stageA_materialization_inputs.append(coe_stageA_external_reservoir)
            _stageA_materialization_inputs.append(coe_stageA_pre_scout_reservoir)
            _materialization_payload = merge_stageA_proposal_reservoir_payloads(
                _stageA_materialization_inputs,
                max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 4),
            )
            _materialization_payload["source"] = "pre_stageA_materialization_reservoir"
        except Exception as exc:
            if isinstance(coe_stageA_materialization, dict):
                coe_stageA_materialization.setdefault("warnings", []).append(
                    f"could not merge pre-Stage-A scout reservoir: {type(exc).__name__}: {exc}"
                )
            _materialization_payload = coe_stageA_external_reservoir
    if coe_stageA_materialization is None and isinstance(coe_stageA_pre_scout_summary, dict):
        coe_stageA_materialization = {
            "enabled": True,
            "source": "pre_stageA_scouts",
            "mode": str(getattr(args, "coe_mode", "off") or "off"),
        }
    if isinstance(coe_stageA_materialization, dict) and isinstance(
        coe_stageA_pre_scout_summary,
        dict,
    ):
        coe_stageA_materialization["pre_stageA_scouts"] = _make_json_serializable(
            coe_stageA_pre_scout_summary
        )
    y_transform_names, coe_stageA_materialization = _materialize_stageA_y_branch_proposals(
        reservoir_payload=_materialization_payload,
        materialization_summary=coe_stageA_materialization,
        y_transform_names=y_transform_names,
        available_y_transform_names=available_y_transform_names,
    )
    try:
        setattr(search_hp, "coe_problem_id", _coe_problem_stem(source_filepath))
        setattr(search_hp, "coe_stageA_replay_log", [])
        if isinstance(_materialization_payload, dict) and _coe_stageA_materialization_mode_enabled(args):
            setattr(
                search_hp,
                "coe_stageA_replay_reservoir",
                _make_json_serializable(_materialization_payload),
            )
            _replay_count = sum(
                1
                for _row in list(_materialization_payload.get("candidates") or [])
                if isinstance(_row, dict)
                and str(_row.get("kind", "")) == "compound_coordinate_replay"
            )
            if _replay_count:
                if coe_stageA_materialization is None:
                    coe_stageA_materialization = {
                        "enabled": True,
                        "source": "stageA_compound_replay",
                        "mode": str(getattr(args, "coe_mode", "off") or "off"),
                    }
                if isinstance(coe_stageA_materialization, dict):
                    coe_stageA_materialization["compound_replay_candidates"] = int(_replay_count)
        else:
            setattr(search_hp, "coe_stageA_replay_reservoir", None)
    except Exception:
        pass

    # If enforcing units, filter out y-transforms that are dimensionally invalid.
    # Examples: log/exp/trig require dimensionless y; sqrt1p requires dimensionless y.
    if bool(args.enforce_units) and (units_payload is not None):
        try:
            from nestynet_sr.sr_core.units import filter_y_transform_names_by_units, is_dimless

            us = units_payload.get("unit_system")
            y_dim = units_payload.get("y_dim")
            if us is not None and y_dim is not None:
                before = list(y_transform_names)
                y_transform_names = filter_y_transform_names_by_units(
                    y_transform_names, y_dim=y_dim, us=us
                )
                skipped = [n for n in before if n not in y_transform_names]
                if skipped:
                    y_dim_str = us.format_dim(y_dim) if not is_dimless(y_dim) else "1"
                    print(
                        f"[Units] Skipping y-transforms incompatible with y units ({y_dim_str}): {', '.join(skipped)}"
                    )
                if not y_transform_names:
                    print(
                        "[Units] All requested y-transforms were unit-incompatible; falling back to 'identity'."
                    )
                    y_transform_names = ["identity"]
        except Exception as e:
            print(f"[Units] Warning: failed to filter y-transforms by units: {e}")

    if isinstance(coe_stageA_materialization, dict):
        _materialized_y = list(coe_stageA_materialization.get("y_branch_materialized") or [])
        if _materialized_y:
            _active_y = [n for n in _materialized_y if n in y_transform_names]
            _removed_y = [n for n in _materialized_y if n not in y_transform_names]
            coe_stageA_materialization["y_branch_active_after_units"] = _active_y
            if _removed_y:
                coe_stageA_materialization["y_branch_removed_by_units"] = _removed_y

    y_transforms = build_default_y_transforms(y_transform_names)

    # Leaf builder wraps NestyNet + adaptors
    leaf_builder = LeafBuilder(model_hp, device, dtype)

    # Stage‑B fresh NN factory (for newly introduced NN atoms, e.g. local Stage‑A splits)
    fresh_nn_factory = make_stage_a_nn_factory(leaf_builder)

    coe_stageA_continuation_scout_seen: set[str] = set()
    coe_stageA_continuation_scout_counter = 0

    def _maybe_run_reference_state_scouts(
        *,
        phase_kind: str,
        reason: str,
        current_ast,
        y_transform_name: str,
        pass_index: int,
        phase_label: Optional[str] = None,
        x_transform_map: Optional[dict] = None,
    ) -> None:
        """Launch bounded scouts from a confirmed visible reference state."""

        nonlocal coe_stageA_continuation_scout_counter
        nonlocal coe_scout_expression_reservoir

        if not bool(getattr(args, "coe_continuation_scouts", True)):
            return
        continuation_count_arg = getattr(args, "coe_continuation_scout_count", None)
        if continuation_count_arg is None:
            continuation_count_arg = getattr(args, "coe_scout_count", 0)
        if max(0, int(continuation_count_arg or 0)) <= 0:
            return
        if not _parse_coe_scout_slice_ids(args):
            return
        reason_l = str(reason or "").lower()
        if "rollback" in reason_l or "preconditioning" in reason_l:
            return
        max_phases = max(
            0,
            int(getattr(args, "coe_continuation_scout_max_phases", 6) or 0),
        )
        if max_phases > 0 and coe_stageA_continuation_scout_counter >= max_phases:
            return
        try:
            fit_link = canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None))
        except Exception as exc:
            print(
                "[CoE scouts] continuation skipped: invalid active fit-link "
                f"{getattr(lm_hp, 'fit_y_link', None)!r}: {exc}"
            )
            return
        try:
            fit_link_scale = float(getattr(lm_hp, "fit_y_link_scale", 1.0))
        except Exception:
            fit_link_scale = 1.0
        if fit_link is None:
            fit_link_scale = 1.0
        if x_transform_map:
            print(
                "[CoE scouts] continuation skipped: active x-transform "
                "context is not replay-safe yet."
            )
            return
        try:
            if not has_nn_atoms(current_ast):
                return
        except Exception:
            return
        try:
            ast_sig = ast_to_human_readable(current_ast)
        except Exception:
            ast_sig = str(current_ast)
        cont_y_name = str(y_transform_name or "identity")
        seen_key = "|".join(
            [
                str(phase_kind or "continuation"),
                cont_y_name,
                str(fit_link),
                f"{fit_link_scale:.12g}",
                ast_sig,
            ]
        )
        if seen_key in coe_stageA_continuation_scout_seen:
            return
        coe_stageA_continuation_scout_seen.add(seen_key)

        coe_stageA_continuation_scout_counter += 1
        y_safe = "".join(
            ch if (ch.isalnum() or ch in {"_", "-"}) else "_"
            for ch in cont_y_name
        ).strip("_") or "identity"
        phase = str(
            phase_label
            or f"continuation_after_{phase_kind}_{coe_stageA_continuation_scout_counter:03d}_{y_safe}"
        )
        print(
            "[CoE scouts] continuation "
            f"{phase}: kind={phase_kind}, reason={reason}, pass={int(pass_index)}, "
            f"fit_link={describe_fit_link(fit_link, fit_link_scale)}"
        )
        cont_result = _run_coe_scout_proposers(
            args=args,
            filepath=filepath,
            results_dir=results_dir,
            base_filename=base_filename,
            current_stageA_reservoir=getattr(search_hp, "coe_stageA_replay_reservoir", None),
            current_reservoir=coe_scout_expression_reservoir,
            phase=phase,
            continuation_ast=copy.deepcopy(current_ast),
            continuation_y_op_name=cont_y_name,
            continuation_fit_link_name=fit_link,
            continuation_fit_link_scale=fit_link_scale,
        )
        if not isinstance(cont_result, dict):
            return
        cont_summary = cont_result.get("summary")
        if isinstance(cont_summary, dict):
            cont_summary = dict(cont_summary)
            cont_summary["phase_kind"] = str(phase_kind)
            cont_summary["stageA_restart_reason"] = str(reason)
            cont_summary["stageA_pass_index"] = int(pass_index)
            coe_stageA_continuation_scout_summaries.append(
                _make_json_serializable(cont_summary)
            )
            print(
                "[CoE scouts] continuation "
                f"{phase}: completed={cont_summary.get('completed', 0)} "
                f"loaded_payloads={cont_summary.get('loaded_payloads', 0)} "
                f"loaded_stageA_payloads={cont_summary.get('loaded_stageA_payloads', 0)}"
            )
        cont_stageA = cont_result.get("merged_stageA_reservoir")
        if isinstance(cont_stageA, dict):
            setattr(
                search_hp,
                "coe_stageA_replay_reservoir",
                _make_json_serializable(cont_stageA),
            )
        coe_scout_expression_reservoir = _merge_coe_expression_reservoir_payload(
            coe_scout_expression_reservoir,
            cont_result.get("merged_reservoir"),
            max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 4),
            source=f"{phase}_expression_reservoir",
        )

    def _stageA_continuation_scout_callback(
        *,
        reason: str,
        current_ast,
        pass_index: int,
        y_transform_name: str,
        x_transform_map: Optional[dict] = None,
    ) -> None:
        """Launch proposal-only scouts when the reference Stage A is about to restart."""
        _maybe_run_reference_state_scouts(
            phase_kind="stageA_pass",
            reason=reason,
            current_ast=current_ast,
            y_transform_name=y_transform_name,
            pass_index=pass_index,
            x_transform_map=x_transform_map,
        )

    # Optionally load an existing checkpoint to skip Stage A or Stage B
    loaded_checkpoint = None
    if args.resume_from is not None:
        resume_path = args.resume_from
        print(f"Resuming from checkpoint: {resume_path}")
        with open(resume_path, "rb") as f:
            loaded_checkpoint = pickle.load(f)
        if not isinstance(loaded_checkpoint, dict) or "phase" not in loaded_checkpoint:
            raise ValueError(f"Checkpoint {resume_path} does not look like a valid SR checkpoint.")

        _checkpoint_stat_contract = loaded_checkpoint.get("statistical_selection_split")
        if statistical_split_plan is not None:
            statistical_split_plan.assert_checkpoint_compatible(
                _checkpoint_stat_contract
            )
        elif _checkpoint_stat_contract is not None:
            raise ValueError(
                "checkpoint was trained behind a statistical audit firewall; "
                "resume with --stat_selection and the original source/audit contract"
            )

        ckpt_filepaths = loaded_checkpoint.get("filepaths")
        ckpt_filepath = loaded_checkpoint.get("filepath")
        if ckpt_filepaths is not None:
            cur = [str(p) for p in filepaths]
            old = [str(p) for p in ckpt_filepaths]
            if cur != old:
                # Allow mismatch if only basenames match (directory changed)
                cur_basenames = [os.path.basename(p) for p in cur]
                old_basenames = [os.path.basename(p) for p in old]
                if cur_basenames != old_basenames:
                    raise ValueError(
                        f"Checkpoint {resume_path} was created for filepaths={old}, "
                        f"but current filepaths={cur}."
                    )
                else:
                    print("[Resume] Filepath directory changed (allowed)")
        elif ckpt_filepath is not None and ckpt_filepath != filepath:
            # Allow mismatch if only basename matches (directory changed)
            if os.path.basename(ckpt_filepath) != os.path.basename(filepath):
                raise ValueError(
                    f"Checkpoint {resume_path} was created for filepath '{ckpt_filepath}', "
                    f"but current --filepath is '{filepath}'."
                )
            else:
                print("[Resume] Filepath directory changed (allowed)")
        ckpt_Nx = loaded_checkpoint.get("Nxvars")
        if ckpt_Nx is not None and ckpt_Nx != Nxvars:
            raise ValueError(
                f"Checkpoint {resume_path} has Nxvars={ckpt_Nx}, but CSV reports Nxvars={Nxvars}."
            )
        print(f"Loaded checkpoint with phase={loaded_checkpoint['phase']}")

    separability_success = False
    stageA_status = "unresolved"
    stageA_full_compound_solved = False  # Confirmed outer map solved, not mere NN[z] compression
    full_compound_compressed = False  # True if identity found a pure full-variable NN[z(x)]
    compound_is_1d = False  # True if compound is univariate (single atom, arity 1)
    final_model = None
    final_ast = None
    final_y_op = None
    final_y_op_inv = None
    final_y_op_name = None
    rest_add_final = None
    rest_mult_final = None

    # Proposal-only diagnostics (do not affect workflow)
    outer_peel_square_decision = None
    outer_peel_ranked = None
    stageB_virtual_top_names = []
    stageB_virtual_portfolio = []
    stageB_y_shortlist_sources = {}
    stageA_val_loss = None  # Will be set from checkpoint or loaded from model file
    stageA_val_losses = None  # Optional per-dataset Stage A losses (multi-dataset)
    stageA_val_loss_agg_mode = None
    stageA_val_loss_agg_weights = None
    stageA_dataset_ids = None
    fit_y_link_used = None  # Will be set from checkpoint or loaded from model file
    fit_y_link_scale_used = 1.0  # Scale for asinh fit-link (default 1.0)
    fit_link_branch_certificate = None
    fit_link_branch_status = None
    fit_link_original_y_certified = None
    fit_link_original_y_val_loss = None
    fit_link_original_y_allowed_loss = None
    stageA_deferred_fitlink_branches = []
    coe_stageA_ybranch_committee = None
    coe_stageA_compound_shortlist = None

    # ckpt will be (re)used/updated later for saving Stage A/B state
    ckpt = loaded_checkpoint
    stageA_continuation_seed_done = False

    if loaded_checkpoint is None and bool(getattr(args, "stageA_continuation_seed", False)):
        if not has_nn_atoms(initial_ast):
            print("[CoE continuation scout] Seed AST has no NN atoms; no Stage-A continuation needed.")
            final_ast = initial_ast
            stageA_status = "continuation_seed_no_nn"
            stageA_continuation_seed_done = True
        else:
            cont_y_name = str(getattr(args, "stageA_continuation_y_op", None) or "identity")
            try:
                cont_y_op, cont_y_op_inv, cont_y_name = resolve_y_transform_name(
                    cont_y_name,
                    transforms=y_transforms,
                )
            except Exception:
                print(
                    f"[CoE continuation scout] y-transform {cont_y_name!r} unavailable; "
                    "falling back to identity."
                )
                cont_y_op, cont_y_op_inv, cont_y_name = resolve_y_transform_name(
                    "identity",
                    transforms=build_default_y_transforms(["identity"]),
                )
            print(
                "[CoE continuation scout] Running bounded feedback Stage A from "
                f"seed AST in {cont_y_name}(y)-space "
                f"with fit-link {describe_fit_link(getattr(lm_hp, 'fit_y_link', None), getattr(lm_hp, 'fit_y_link_scale', 1.0))}."
            )
            (
                cont_success,
                cont_model,
                cont_rest_add,
                cont_rest_mult,
                _,
                cont_ast,
                _,
                cont_full_compound_solved,
            ) = run_separability_for_transform(
                i_op=0,
                y_op=cont_y_op,
                y_op_inv=cont_y_op_inv,
                candidate_sep_ops=[True],
                y_transform_names=[cont_y_name],
                initial_ast=initial_ast,
                filepath=filepaths if len(filepaths) > 1 else filepath,
                Nxvars=Nxvars,
                y_med=y_med,
                y_mad=y_mad,
                np_dtype=np_dtype,
                dtype=dtype,
                device=device,
                data_hp=data_hp,
                model_hp=model_hp,
                lm_hp=lm_hp,
                search_hp=search_hp,
                leaf_builder=leaf_builder,
                model_output=model_output,
                model_sep_output=model_sep_output,
                mode="full",
                units_payload=units_payload,
                enforce_units=bool(args.enforce_units),
                units_policy=str(args.units_policy),
                nn_units_semantics=str(args.nn_units_semantics),
                y_log_dynamic_range=y_log_dynamic_range,
                y_abs_median=y_abs_median,
                global_best_val_loss_base=None,
                freeze_non_nn=True,
                skip_initial_fit=False,
                y_raw_full=y_data,
                noise_sigma_y=noise_sigma_y,
                noise_floor_mc_samples=args.noise_floor_mc_samples,
            )
            if cont_model is None:
                print("[CoE continuation scout] Continuation Stage A produced no model.")
                final_ast = initial_ast
                stageA_status = "continuation_seed_failed"
            else:
                separability_success = bool(cont_success)
                final_model = cont_model
                final_ast = cont_ast
                final_y_op = cont_y_op
                final_y_op_inv = cont_y_op_inv
                final_y_op_name = cont_y_name
                rest_add_final = cont_rest_add
                rest_mult_final = cont_rest_mult
                stageA_full_compound_solved = bool(cont_full_compound_solved)
                compound_is_1d = _compute_compound_is_1d(final_ast, Nxvars)
                full_compound_compressed = bool(compound_is_1d)
                if cont_success and (cont_rest_add is not None or cont_rest_mult is not None):
                    stageA_status = "continuation_split_confirmed"
                elif full_compound_compressed:
                    stageA_status = "continuation_compound_unresolved"
                else:
                    stageA_status = "continuation_unresolved"
            stageA_continuation_seed_done = True

    if loaded_checkpoint is None and not stageA_continuation_seed_done:
        # ---------------------------------------------------------------
        # Fresh Stage A run (or continuing on hybrid AST)
        # ---------------------------------------------------------------
        phase_direct = _try_phase_prescan_direct_closure(
            phase_hints=phase_hints,
            phase_context_hints=phase_context_hints,
            outer_link_hints=outer_link_hints,
            initial_ast=initial_ast,
            filepath=filepath,
            filepaths=filepaths,
            Nxvars=Nxvars,
            np_dtype=np_dtype,
            data_hp=data_hp,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            fresh_nn_factory=fresh_nn_factory,
            units_payload=units_payload,
            enforce_units=bool(args.enforce_units),
            units_policy=str(args.units_policy),
            nn_units_semantics=str(args.nn_units_semantics),
            model_output=model_output,
            model_sep_output=model_sep_output,
        )

        # Check if AST has any NN atoms
        if phase_direct is not None:
            separability_success = False
            stageA_status = "phase_hint_confirmed"
            final_model = phase_direct["model"]
            final_ast = phase_direct["ast"]
            final_y_op = None
            final_y_op_inv = None
            final_y_op_name = "identity"
            rest_add_final = None
            rest_mult_final = None
            stageA_val_loss = float(phase_direct["val_loss"])
        elif not has_nn_atoms(initial_ast):
            print("[Stage A] No NN atoms in initial AST; skipping separability search.")
            separability_success = False
            final_ast = initial_ast
            # Skip to Stage B - no changes needed to the AST
        elif not args.stageA_separabilities:
            print("[Stage A] Separability search disabled (--no_stageA_separabilities).")
            # NOTE: Even with separability search disabled, we still run a conservative
            # outer-peel *proposal* (identity vs square) after training the identity model.
            # This is important for problems like AIF #028 where no add/mul separability exists
            # under any outer transform, but squaring removes a hard outer nonlinearity.

            name_to_tf = {t.name: t for t in y_transforms}

            def _train_one_transform(y_name: str, y_op, y_op_inv, out_path: str):
                from nestynet_sr.sr_search.data_utils import build_datasets
                from nestynet_sr.sr_search.search import _apply_fit_link_to_model
                from nestynet_sr.sr_search.stagea_fit_tournament import (
                    fit_initial_model_with_tournament,
                )

                dataset_train, dataset_val, datagen_train_noshuffle, datagen_val_noshuffle = (
                    build_datasets(filepath, Nxvars, np_dtype, data_hp, y_op)
                )
                if datagen_train_noshuffle is None:
                    raise RuntimeError("Failed to build datasets")

                model, nparam, _ = build_composite_ast(
                    initial_ast,
                    model_hp.num_segments_min,
                    dual_layer=dual_layer,
                    leaf_builder=leaf_builder,
                    device=device,
                    dtype=dtype,
                )
                _apply_fit_link_to_model(model, lm_hp)
                print(f"[Stage A] Built initial model with {nparam} parameters")
                print(f"[Stage A] Training with LM optimizer (max epochs: {lm_hp.epochs})...")

                best_val_loss, _, best_val_p, lm_opt = fit_initial_model_with_tournament(
                    model=model,
                    train_dl=datagen_train_noshuffle,
                    val_dl=datagen_val_noshuffle,
                    epochs=lm_hp.epochs,
                    LM_strategy=lm_hp.strategy,
                    nval_patience=lm_hp.nval_patience,
                    loss_target=lm_hp.loss_target,
                    epochs_min=lm_hp.epochs_min,
                    chisq_tol=lm_hp.chisq_tol,
                    device=device,
                    epochs_awful_check=lm_hp.epochs_awful_check,
                    awful_threshold=lm_hp.awful_threshold,
                    y_op=y_op,
                    y_op_inv=y_op_inv,
                    lm_hp=lm_hp,
                )
                lm_opt._update_param_groups(best_val_p)

                print(f"[Stage A] Training complete. Final validation loss: {best_val_loss:.6e}")
                print(f"[Stage A] Loss is in units of MAD(φ(y))^2 where φ(y) = {y_name}(y)")

                save_dict = {
                    "y_op": y_op,
                    "y_op_inv": y_op_inv,
                    "Nxvars": Nxvars,
                    "num_segments": model_hp.num_segments_min,
                    "dual_layer": dual_layer,
                    "model_state_dict": model.state_dict(),
                    "ast": initial_ast,
                    "val_loss": best_val_loss,
                    "fit_y_link": getattr(lm_hp, "fit_y_link", None),
                    "fit_y_link_scale": float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                }
                torch.save(save_dict, out_path)
                print(f"[Stage A] Saved trained model to {out_path}")
                return model, best_val_loss, datagen_train_noshuffle, datagen_val_noshuffle

            # If identity is not configured (e.g. user forced a specific y-op),
            # fall back to the previous behaviour: train only the first transform.
            if "identity" not in name_to_tf:
                y_name = y_transform_names[0]
                yt = name_to_tf[y_name]
                print(f"[Stage A] Training initial NN model with y-transform: {y_name}")

                try:
                    model, loss_val, _, _ = _train_one_transform(
                        y_name=y_name,
                        y_op=yt.np_op,
                        y_op_inv=yt.torch_inv,
                        out_path=_model_path(y_name, sep=False),
                    )
                    # Copy to canonical path
                    try:
                        shutil.copyfile(_model_path(y_name, sep=False), model_output)
                    except Exception as e:
                        print(f"Warning: failed to copy model '{y_name}' to canonical path: {e}")

                    separability_success = False
                    final_ast = initial_ast
                    final_model = model
                    final_y_op = yt.np_op
                    final_y_op_inv = yt.torch_inv
                    final_y_op_name = y_name
                    rest_add_final = None
                    rest_mult_final = None
                except Exception as e:
                    print(f"[ERROR] Stage A training failed: {e}")
                    separability_success = False
                    final_ast = initial_ast
            else:
                # 1) Always train identity first and archive it.
                yt_id = name_to_tf["identity"]
                print("[Stage A] Training initial NN model with y-transform: identity")
                try:
                    stageA_model, loss_id, datagen_train_id, datagen_val_id = _train_one_transform(
                        y_name="identity",
                        y_op=yt_id.np_op,
                        y_op_inv=yt_id.torch_inv,
                        out_path=model_output_identity,
                    )
                except Exception as e:
                    print(f"[ERROR] Stage A identity training failed: {e}")
                    separability_success = False
                    final_ast = initial_ast
                else:
                    # Copy identity -> canonical "current"
                    try:
                        shutil.copyfile(model_output_identity, model_output)
                    except Exception as e:
                        print(f"Warning: failed to copy identity model to canonical path: {e}")

                    separability_success = False
                    final_ast = initial_ast
                    final_model = stageA_model
                    final_y_op = yt_id.np_op
                    final_y_op_inv = yt_id.torch_inv
                    final_y_op_name = "identity"
                    rest_add_final = None
                    rest_mult_final = None
                    identity_val_loss = loss_id

                    # ---------------------------------------------------------------
                    # 1a-bis) AUTO ASINH FIT-LINK (bypass path)
                    # ---------------------------------------------------------------
                    auto_fit_link_threshold = getattr(lm_hp, "auto_fit_link_log_dynamic_range_threshold", 4.0)
                    if (
                        identity_val_loss > lm_hp.loss_target * 100
                        and y_log_dynamic_range > auto_fit_link_threshold
                        and getattr(lm_hp, "fit_y_link", None) is None
                    ):
                        print(
                            f"\n{YELLOW}[Stage A] Identity fit borderline "
                            f"(loss {identity_val_loss:.2e} > 100x target {lm_hp.loss_target:.2e}) with "
                            f"high dynamic range ({y_log_dynamic_range:.2f} decades > {auto_fit_link_threshold:.1f}){RESET}"
                        )
                        print(f"{YELLOW}[Stage A] Retrying with asinh fit-link conditioning...{RESET}")

                        orig_fit_y_link = getattr(lm_hp, "fit_y_link", None)
                        orig_fit_y_link_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)

                        lm_hp.fit_y_link = "asinh"
                        lm_hp.fit_y_link_scale = float(y_abs_median) if y_abs_median > 1e-30 else 1.0

                        try:
                            asinh_model, asinh_loss, _, asinh_datagen_val = _train_one_transform(
                                y_name="identity",
                                y_op=yt_id.np_op,
                                y_op_inv=yt_id.torch_inv,
                                out_path=model_output_identity,
                            )

                            # Y-space sanity check
                            from nestynet_sr.sr_search.search import _check_asinh_yspace_sanity

                            asinh_ok, y_mse, y_mse_allowed, D_ref, base_y_mse = _check_asinh_yspace_sanity(
                                model=asinh_model,
                                dl_val=asinh_datagen_val,
                                device=device,
                                asinh_loss=asinh_loss,
                                lm_hp=lm_hp,
                                base_model=stageA_model,
                            )
                            print(
                                f"[Stage A] asinh y-space sanity: ok={asinh_ok}, "
                                f"y_mse={y_mse:.3e}, allowed={y_mse_allowed:.3e}"
                            )

                            if asinh_ok and asinh_loss < identity_val_loss:
                                print(f"{GREEN}[Stage A] asinh-conditioned identity SUCCEEDED (bypass path)!{RESET}")
                                final_model = asinh_model
                                stageA_model = asinh_model
                                identity_val_loss = asinh_loss
                                datagen_val_id = asinh_datagen_val
                                try:
                                    shutil.copyfile(model_output_identity, model_output)
                                except Exception as e:
                                    print(f"Warning: failed to copy asinh model to canonical path: {e}")
                            else:
                                lm_hp.fit_y_link = orig_fit_y_link
                                lm_hp.fit_y_link_scale = orig_fit_y_link_scale
                                print(f"{YELLOW}[Stage A] asinh-conditioned identity did not improve, reverting...{RESET}")
                        except Exception as e:
                            print(f"Warning: asinh fit-link retry failed with exception: {e}")
                            lm_hp.fit_y_link = orig_fit_y_link
                            lm_hp.fit_y_link_scale = orig_fit_y_link_scale

                    # ---------------------------------------------------------------
                    # REFERENCE MODEL SNAPSHOT (bypass path)
                    # ---------------------------------------------------------------
                    if os.path.exists(model_output):
                        try:
                            ref_mod_path = os.path.join(results_ref_dir, f"{model_base_filename}.mod")
                            shutil.copyfile(model_output, ref_mod_path)

                            ref_saved = torch.load(ref_mod_path, weights_only=False)
                            ref_val_loss = ref_saved.get("val_loss", None)
                            ref_fit_y_link = ref_saved.get("fit_y_link", None)
                            ref_fit_y_link_scale = ref_saved.get("fit_y_link_scale", 1.0)

                            ref_ckpt = {
                                "version": 1,
                                "phase": "after_stageA",
                                "filepath": filepath,
                                "filepaths": list(filepaths),
                                "statistical_selection_split": (
                                    statistical_split_plan.checkpoint_contract()
                                    if statistical_split_plan is not None
                                    else None
                                ),
                                "Nxvars": Nxvars,
                                "y_op_name": "identity",
                                "separability_success": False,
                                "stageA_status": "unresolved",
                                "rest_add": None,
                                "rest_mult": None,
                                "ast": ref_saved["ast"],
                                "val_loss": ref_val_loss,
                                "fit_y_link": ref_fit_y_link,
                                "fit_y_link_scale": ref_fit_y_link_scale,
                                "has_remaining_nns": True,
                                "outer_peel_square": None,
                                "outer_peel_ranked": None,
                                "x_transform": {},
                            }
                            ref_ckpt_path = os.path.join(results_ref_dir, f"{model_base_filename}.state.pkl")
                            with open(ref_ckpt_path, "wb") as f:
                                pickle.dump(ref_ckpt, f)

                            fit_link_note = f" [asinh, scale={ref_fit_y_link_scale:.4g}]" if ref_fit_y_link == "asinh" else ""
                            print(f"[Reference] Saved pre-separability model to {ref_mod_path}{fit_link_note}")
                        except Exception as e:
                            print(f"Warning: failed to save reference model: {e}")

                    # Outer-peel proposals skipped in --no_stageA_separabilities mode;
                    # Stage B will run in identity(y)-space directly.
        else:
            # Check if hybrid (has both NN and analytic atoms)
            from nestynet_sr.sr_core import collect_nn_atoms
            from nestynet_sr.sr_core.bridges import (
                AcosNode,
                AddNode,
                AsinNode,
                AtanNode,
                AtomNode,
                CosNode,
                ExpNode,
                LogNode,
                MulNode,
                PowNode,
                SinNode,
            )

            nn_atoms = collect_nn_atoms(initial_ast)

            def _has_non_nn_atoms(node):
                if isinstance(node, AtomNode):
                    return node.kind.lower() != "nn"
                if isinstance(node, (AddNode, MulNode)):
                    return _has_non_nn_atoms(node.left) or _has_non_nn_atoms(node.right)
                if isinstance(node, PowNode):
                    return _has_non_nn_atoms(node.base)
                if isinstance(node, (LogNode, SinNode, CosNode, ExpNode, AsinNode, AcosNode, AtanNode)):
                    return _has_non_nn_atoms(node.arg)
                return False

            has_non_nn_atoms = _has_non_nn_atoms(initial_ast)
            is_hybrid = has_non_nn_atoms and len(nn_atoms) > 0

            # from sr_core import collect_nn_atoms
            # nn_atoms = collect_nn_atoms(initial_ast)
            # is_hybrid = not is_minimal_ast(initial_ast) and len(nn_atoms) > 0

            if is_hybrid:
                # Clean approach: skip Stage A splitting for hybrid ASTs
                # Let Stage B handle all splitting with proper teacher subtree semantics
                print(
                    f"[Stage A] Hybrid AST detected: {len(nn_atoms)} NN atoms in complex expression"
                )
                print(
                    "[Stage A] Skipping Stage A separability search for hybrid AST (avoiding semantic bugs)"
                )
                print(
                    "[Stage A] Stage B will handle all NN splitting with correct teacher-subtree context"
                )
                separability_success = False
                final_ast = initial_ast
            else:
                # Original Stage A logic for pure NN ASTs
                print("[Stage A] Running standard separability search on pure NN expression")
                stageA_filepath_arg = filepaths if len(filepaths) > 1 else filepath

                # ---------------------------------------------------------------
                # 1) ALWAYS run identity baseline first
                # ---------------------------------------------------------------
                # This establishes baseline losses, candidate_sep_ops, and provides
                # a fallback model.
                try:
                    idx_identity = y_transform_names.index("identity")
                except ValueError:
                    idx_identity = None

                # Global candidate flags over the configured y-transform list
                candidate_sep_ops = [True] * len(y_transform_names)
                # Stage-A cache keyed by y-search state signatures.
                stageA_state_cache = {}

                # Keep a stable reference to the initial identity model for
                # outer-peel proposals and for archiving.
                stageA_model = None
                stageA_ast = None
                stageA_rest_add = None
                stageA_rest_mult = None
                stageA_found_separability = False
                stageA_full_compound_solved = False  # Confirmed outer map solved, not mere NN[z] compression
                full_compound_compressed = False
                global_best_val_loss_base = None  # Best val_loss / loss_scale across y-transforms

                if idx_identity is not None:
                    yt = y_transforms[idx_identity]
                    y_op = yt.np_op  # None for identity
                    y_op_inv = yt.torch_inv

                    success, model, rest_add, rest_mult, candidate_sep_ops, current_ast, _, stageA_full_compound_solved = (
                        run_separability_for_transform(
                            i_op=idx_identity,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                            candidate_sep_ops=candidate_sep_ops,
                            y_transform_names=y_transform_names,
                            initial_ast=initial_ast,
                            filepath=stageA_filepath_arg,
                            Nxvars=Nxvars,
                            y_med=y_med,
                            y_mad=y_mad,
                            np_dtype=np_dtype,
                            dtype=dtype,
                            device=device,
                            data_hp=data_hp,
                            model_hp=model_hp,
                            lm_hp=lm_hp,
                            search_hp=search_hp,
                            leaf_builder=leaf_builder,
                            model_output=model_output_identity,
                            model_sep_output=model_sep_output_identity,
                            mode="full",
                            units_payload=units_payload,
                            enforce_units=bool(args.enforce_units),
                            units_policy=str(args.units_policy),
                            nn_units_semantics=str(args.nn_units_semantics),
                            y_log_dynamic_range=y_log_dynamic_range,
                            y_abs_median=y_abs_median,
                            global_best_val_loss_base=global_best_val_loss_base,
                            y_raw_full=y_data,
                            noise_sigma_y=noise_sigma_y,
                            noise_floor_mc_samples=args.noise_floor_mc_samples,
                            stageA_restart_callback=_stageA_continuation_scout_callback,
                        )
                    )

                    # Update global best-loss guard
                    if model is not None:
                        _bvlb = getattr(model, '_best_val_loss_base', None)
                        if _bvlb is not None and (global_best_val_loss_base is None or _bvlb < global_best_val_loss_base):
                            global_best_val_loss_base = _bvlb

                    if model is not None:
                        stageA_model = model
                        stageA_ast = current_ast
                        stageA_rest_add = rest_add
                        stageA_rest_mult = rest_mult
                        compound_is_1d = _compute_compound_is_1d(stageA_ast, Nxvars)
                        full_compound_compressed = bool(compound_is_1d)
                        if full_compound_compressed:
                            stageA_status = "compound_unresolved"

                        # Current/default choice starts as identity
                        final_model = model
                        final_ast = current_ast
                        final_y_op = y_op
                        final_y_op_inv = y_op_inv
                        final_y_op_name = yt.name
                        rest_add_final = rest_add
                        rest_mult_final = rest_mult

                        # Keep a canonical "current" model file as well.
                        try:
                            if os.path.exists(model_output_identity):
                                shutil.copyfile(model_output_identity, model_output)
                            if os.path.exists(model_sep_output_identity):
                                shutil.copyfile(model_sep_output_identity, model_sep_output)
                        except Exception as e:
                            print(f"Warning: failed to copy identity model to canonical path: {e}")

                    # Store whether identity found an actual add/mult split.
                    stageA_found_separability = _split_success(success, rest_add, rest_mult)

                    # Compound detection with extras (e.g. x2*x3*NN[z=(x0*x1)])
                    # creates multiplicative structure without setting rest_mult,
                    # and the function returns separability_success=False because
                    # no formal NN split was accepted.  Recognise this as
                    # separability so we skip redundant y-transforms.
                    if (not stageA_found_separability) and model is not None and current_ast is not None:
                        from nestynet_sr.sr_core import collect_nn_atoms
                        _nn_atoms = collect_nn_atoms(current_ast)
                        if _nn_atoms and all(
                            len(a.var_idxs) < Nxvars for a in _nn_atoms
                        ):
                            stageA_found_separability = True

                # ---------------------------------------------------------------
                # 1a-bis) AUTO ASINH FIT-LINK: If identity failed with high dynamic range
                # ---------------------------------------------------------------
                # When identity fit fails (or barely succeeds) and the data has high dynamic
                # range, retry with asinh fit-link conditioning. This compresses the dynamic
                # range during optimization while keeping the symbolic analysis in y-space.
                auto_fit_link_threshold = getattr(lm_hp, "auto_fit_link_log_dynamic_range_threshold", 4.0)
                asinh_retry_triggered = False
                if (
                    (model is None)
                    and y_log_dynamic_range > auto_fit_link_threshold
                    and getattr(lm_hp, "fit_y_link", None) is None  # Only if not already using a fit-link
                ):
                    print(
                        f"\n{YELLOW}[Stage A] Identity fit {'failed' if model is None else 'borderline'} with "
                        f"high dynamic range ({y_log_dynamic_range:.2f} decades > {auto_fit_link_threshold:.1f}){RESET}"
                    )
                    print(f"{YELLOW}[Stage A] Retrying with asinh fit-link conditioning...{RESET}")

                    # Store original fit_y_link settings
                    orig_fit_y_link = getattr(lm_hp, "fit_y_link", None)
                    orig_fit_y_link_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)

                    # Enable asinh fit-link with scale set to median(|y|)
                    lm_hp.fit_y_link = "asinh"
                    lm_hp.fit_y_link_scale = float(y_abs_median) if y_abs_median > 1e-30 else 1.0
                    asinh_retry_triggered = True

                    try:
                        yt = y_transforms[idx_identity]
                        asinh_success, asinh_model, asinh_rest_add, asinh_rest_mult, _, asinh_ast, _, asinh_full_compound_solved = (
                            run_separability_for_transform(
                                i_op=idx_identity,
                                y_op=yt.np_op,
                                y_op_inv=yt.torch_inv,
                                candidate_sep_ops=candidate_sep_ops,
                                y_transform_names=y_transform_names,
                                initial_ast=initial_ast,
                                filepath=stageA_filepath_arg,
                                Nxvars=Nxvars,
                                y_med=y_med,
                                y_mad=y_mad,
                                np_dtype=np_dtype,
                                dtype=dtype,
                                device=device,
                                data_hp=data_hp,
                                model_hp=model_hp,
                                lm_hp=lm_hp,
                                search_hp=search_hp,
                                leaf_builder=leaf_builder,
                                model_output=model_output_identity,
                                model_sep_output=model_sep_output_identity,
                                mode="full",
                                units_payload=units_payload,
                                enforce_units=bool(args.enforce_units),
                                units_policy=str(args.units_policy),
                                nn_units_semantics=str(args.nn_units_semantics),
                                y_log_dynamic_range=y_log_dynamic_range,
                                y_abs_median=y_abs_median,
                                global_best_val_loss_base=global_best_val_loss_base,
                                y_raw_full=y_data,
                                noise_sigma_y=noise_sigma_y,
                                noise_floor_mc_samples=args.noise_floor_mc_samples,
                                stageA_restart_callback=_stageA_continuation_scout_callback,
                            )
                        )

                        # Update global best-loss guard
                        if asinh_model is not None:
                            _bvlb = getattr(asinh_model, '_best_val_loss_base', None)
                            if _bvlb is not None and (global_best_val_loss_base is None or _bvlb < global_best_val_loss_base):
                                global_best_val_loss_base = _bvlb

                        if asinh_model is not None:
                            print(f"{GREEN}[Stage A] asinh-conditioned identity SUCCEEDED!{RESET}")
                            # Accept the asinh-conditioned result
                            model = asinh_model
                            current_ast = asinh_ast
                            rest_add = asinh_rest_add
                            rest_mult = asinh_rest_mult
                            stageA_full_compound_solved = asinh_full_compound_solved

                            stageA_model = model
                            stageA_ast = current_ast
                            stageA_rest_add = rest_add
                            stageA_rest_mult = rest_mult

                            # Update final choice
                            final_model = model
                            final_ast = current_ast
                            final_y_op = yt.np_op
                            final_y_op_inv = yt.torch_inv
                            final_y_op_name = yt.name
                            rest_add_final = rest_add
                            rest_mult_final = rest_mult
                            stageA_found_separability = _split_success(
                                asinh_success, asinh_rest_add, asinh_rest_mult
                            )
                            if (not stageA_found_separability) and asinh_model is not None and asinh_ast is not None:
                                from nestynet_sr.sr_core import collect_nn_atoms as _cnn
                                _nn_a = _cnn(asinh_ast)
                                if _nn_a and all(len(a.var_idxs) < Nxvars for a in _nn_a):
                                    stageA_found_separability = True
                            compound_is_1d = _compute_compound_is_1d(stageA_ast, Nxvars)
                            full_compound_compressed = bool(compound_is_1d)
                            if full_compound_compressed:
                                stageA_status = "compound_unresolved"

                            # Copy model files
                            try:
                                if os.path.exists(model_output_identity):
                                    shutil.copyfile(model_output_identity, model_output)
                                if os.path.exists(model_sep_output_identity):
                                    shutil.copyfile(model_sep_output_identity, model_sep_output)
                            except Exception as e:
                                print(f"Warning: failed to copy asinh model to canonical path: {e}")
                        else:
                            # Asinh didn't help enough, revert fit-link settings
                            lm_hp.fit_y_link = orig_fit_y_link
                            lm_hp.fit_y_link_scale = orig_fit_y_link_scale
                            print(f"{YELLOW}[Stage A] asinh-conditioned identity did not improve, reverting...{RESET}")

                    except Exception as e:
                        print(f"Warning: asinh fit-link retry failed with exception: {e}")
                        lm_hp.fit_y_link = orig_fit_y_link
                        lm_hp.fit_y_link_scale = orig_fit_y_link_scale

                # ---------------------------------------------------------------
                # REFERENCE MODEL SNAPSHOT: Save pre-separability initial fit
                # ---------------------------------------------------------------
                if model is not None and os.path.exists(model_output):
                    try:
                        ref_mod_path = os.path.join(results_ref_dir, f"{model_base_filename}.mod")
                        shutil.copyfile(model_output, ref_mod_path)

                        # Read metadata from the saved model file
                        ref_saved = torch.load(ref_mod_path, weights_only=False)
                        ref_val_loss = ref_saved.get("val_loss", None)
                        ref_fit_y_link = ref_saved.get("fit_y_link", None)
                        ref_fit_y_link_scale = ref_saved.get("fit_y_link_scale", 1.0)

                        ref_ckpt = {
                            "version": 1,
                            "phase": "after_stageA",
                            "filepath": filepath,
                            "filepaths": list(filepaths),
                            "statistical_selection_split": (
                                statistical_split_plan.checkpoint_contract()
                                if statistical_split_plan is not None
                                else None
                            ),
                            "Nxvars": Nxvars,
                            "y_op_name": "identity",
                            "separability_success": False,
                            "stageA_status": "unresolved",
                            "rest_add": None,
                            "rest_mult": None,
                            "ast": ref_saved["ast"],
                            "val_loss": ref_val_loss,
                            "fit_y_link": ref_fit_y_link,
                            "fit_y_link_scale": ref_fit_y_link_scale,
                            "has_remaining_nns": True,
                            "outer_peel_square": None,
                            "outer_peel_ranked": None,
                            "stageB_virtual_top_names": [],
                            "stageB_virtual_portfolio": [],
                            "x_transform": {},
                        }
                        ref_ckpt_path = os.path.join(results_ref_dir, f"{model_base_filename}.state.pkl")
                        with open(ref_ckpt_path, "wb") as f:
                            pickle.dump(ref_ckpt, f)

                        fit_link_note = f" [asinh, scale={ref_fit_y_link_scale:.4g}]" if ref_fit_y_link == "asinh" else ""
                        print(f"[Reference] Saved pre-separability model to {ref_mod_path}{fit_link_note}")
                    except Exception as e:
                        print(f"Warning: failed to save reference model: {e}")

                # ---------------------------------------------------------------
                # 2) Use identity separability results, then continue with per-transform scans.
                # ---------------------------------------------------------------
                if idx_identity is not None and stageA_found_separability:
                    print("[Stage A] Using identity separability results")
                    separability_success = bool(stageA_found_separability)
                    stageA_status = "split_confirmed"

                # 3) If identity did not find separations, do cheap per-transform Stage A scans.
                #    Also enter this block for pure 1D compounds so the outer-peel probe can run.
                if (
                    ((not separability_success) or compound_is_1d)
                    and (idx_identity is not None)
                    and (stageA_model is not None)
                ):
                    # Always allow a small high-payoff fallback set, even if the
                    # initial separability-op probe is pessimistic.
                    fallback_names = {"square", "log", "reciprocal"}

                    # Include sin/cos as fallback candidates when y-range is
                    # compatible with their respective inverse branches:
                    #   sin(y) paired with arcsin → y ∈ [-π/2, π/2]
                    #   cos(y) paired with arccos → y ∈ [0, π]
                    import math as _math
                    _y_min = float(np.quantile(y_data, 0.005))
                    _y_max = float(np.quantile(y_data, 0.995))
                    if _y_min >= -_math.pi / 2 - 0.1 and _y_max <= _math.pi / 2 + 0.1:
                        fallback_names.add("sin")
                    if _y_min >= -0.1 and _y_max <= _math.pi + 0.1:
                        fallback_names.add("cos")

                    # IMPORTANT: run_separability_for_transform() also gates on candidate_sep_ops[i_op].
                    # If we want fallback_names to actually run, force their flags on here.
                    for j, nm in enumerate(y_transform_names):
                        if nm in fallback_names:
                            candidate_sep_ops[j] = True

                    # A pure NN[z(x)] is a useful compression, not a solved equation.
                    # Keep y-transform alternatives alive until an outer map or Stage B
                    # rewrite confirms a real simplification.
                    if compound_is_1d:
                        print(
                            f"\n{GREEN}[Stage A] Pure 1D compound compressed in identity space; "
                            "continuing y-search for the outer map"
                            f"{RESET}"
                        )

                    best_quick_idx = None

                    # ── Virtual y-transform screening (chain rule) ─────────
                    # Use identity model + chain rule to rank candidate transforms
                    # without retraining each one. Signals are soft priors only.
                    from nestynet_sr.sr_search.chainrule_wrapper import VirtualYModel
                    from nestynet_sr.sr_search.features import (
                        discover_scaling_features,
                        discover_trig_axis,
                        probe_oracle_scaling_groups,
                    )
                    from nestynet_sr.sr_core.separability_math import check_separability
                    from nestynet_sr.sr_search.ysearch_ranker import (
                        VirtualProbeHint,
                        derive_joint_homogeneity_certificate,
                        rank_virtual_hints,
                        select_virtual_portfolio,
                    )

                    ysearch_enable = bool(getattr(search_hp, "ysearch_enable", True))
                    ysearch_expand_k = max(1, int(getattr(search_hp, "ysearch_expand_k", 2)))
                    ysearch_min_valid_frac = float(getattr(search_hp, "ysearch_min_valid_frac", 0.99))
                    ysearch_max_virtual_deriv = float(
                        getattr(search_hp, "ysearch_max_virtual_deriv", 1.0e6)
                    )
                    virtual_hints = []

                    # Build identity-space datagen for chain-rule screening if
                    # not already available (can happen when _train_one_transform
                    # ran inside a try block that we've moved past).
                    _cr_datagen = None
                    try:
                        _cr_datagen = datagen_train_id
                    except (NameError, UnboundLocalError):
                        pass
                    if _cr_datagen is None:
                        from nestynet_sr.sr_search.data_utils import build_datasets
                        _, _, _cr_datagen, _ = build_datasets(
                            filepath, Nxvars, np_dtype, data_hp, y_op=None
                        )
                    _cr_datagens = (
                        [d for d in _cr_datagen if d is not None]
                        if isinstance(_cr_datagen, (list, tuple))
                        else ([_cr_datagen] if _cr_datagen is not None else [])
                    )
                    if not _cr_datagens:
                        print("[Chain-rule screen] No datagen available; skipping virtual ranking.")

                    # Recompute proposal groups on the final identity model,
                    # then certify them on the same static screening data.  The
                    # result is proposal evidence for scheduling one y branch.
                    _joint_scale_certificates = []
                    if _cr_datagens and stageA_model is not None:
                        try:
                            _scale_candidates = discover_scaling_features(
                                model=stageA_model,
                                datagen=_cr_datagens[0],
                                Nxvars=Nxvars,
                                device=device,
                                max_batches=2,
                                max_points=2048,
                                max_group_size=Nxvars,
                            )
                            _joint_candidates = [
                                sp for sp in _scale_candidates if len(sp.indices) >= 2
                            ]
                            _joint_scale_certificates = probe_oracle_scaling_groups(
                                model=stageA_model,
                                datagen=_cr_datagens[0],
                                Nxvars=Nxvars,
                                group_specs=_joint_candidates,
                                device=device,
                                max_batches=2,
                                max_points=2048,
                                rel_std_threshold=float(
                                    getattr(search_hp, "oracle_scaling_rel_std", 0.08)
                                ),
                            )
                            for _cert in _joint_scale_certificates:
                                print(
                                    "[Virtual y-search] Joint homogeneity certificate: "
                                    f"S={list(_cert.indices)}, k={float(_cert.oracle_k):.4g}, "
                                    f"rel_std={float(_cert.oracle_rel_std):.3g}, "
                                    f"n={int(_cert.n_points)}"
                                )
                        except Exception as e:
                            print(
                                "[Virtual y-search] Joint homogeneity verification "
                                f"failed softly: {type(e).__name__}: {e}"
                            )

                    _cr_precision = search_hp.precision_derivs_d2y

                    def _virtual_proxy_mse(wrapper_model, y_transform, datagen, max_batches=2, max_points=2048):
                        n_seen = 0
                        sqerr = 0.0
                        data_iter = datagen() if callable(datagen) else datagen
                        if data_iter is None:
                            return float("inf")
                        for bi, batch in enumerate(data_iter):
                            if bi >= int(max_batches) or n_seen >= int(max_points):
                                break
                            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                                continue
                            x_b = batch[0]
                            y_b = batch[1]
                            try:
                                x_b = x_b.to(device or x_b.device)
                                B = int(x_b.shape[0])
                                x_flat = x_b.reshape(B, -1)
                                y_raw = y_b.to(x_flat.device).reshape(B, -1)[:, 0]
                                with torch.no_grad():
                                    pred_t = wrapper_model.forward(x_flat).reshape(B, -1)[:, 0]
                                    y_tgt = y_transform.torch_op(y_raw.unsqueeze(-1)).reshape(B, -1)[:, 0]
                                mask = torch.isfinite(pred_t) & torch.isfinite(y_tgt)
                                if y_transform.check_fn is not None:
                                    try:
                                        dom = y_transform.check_fn(y_raw)
                                        dom = dom.to(dtype=torch.bool).reshape(-1)
                                        if int(dom.numel()) == int(mask.numel()):
                                            mask = mask & dom
                                    except Exception:
                                        pass
                                if int(mask.sum().item()) <= 0:
                                    continue
                                diff = (pred_t[mask] - y_tgt[mask]).to(dtype=torch.float64)
                                take = min(int(diff.numel()), int(max_points) - int(n_seen))
                                if take <= 0:
                                    break
                                sqerr += float((diff[:take] * diff[:take]).sum().item())
                                n_seen += int(take)
                            except Exception:
                                continue
                        if n_seen <= 0:
                            return float("inf")
                        return float(sqerr / max(1, n_seen))

                    compound_z_affine_by_name = {}
                    compound_z_affine_ranked_early = None
                    identity_outer_affine_confirmed = False

                    def _probe_compound_z_outer_affine_now():
                        nonlocal outer_peel_ranked, compound_z_affine_ranked_early, identity_outer_affine_confirmed
                        if (not compound_is_1d) or stageA_ast is None or not _cr_datagens:
                            return
                        try:
                            from nestynet_sr.sr_core import collect_nn_atoms
                            from nestynet_sr.sr_core.bridges import (
                                compound_input_expr,
                                has_nontrivial_input,
                            )

                            cand_atom = None
                            for at in collect_nn_atoms(stageA_ast):
                                if str(getattr(at, "kind", "")).lower() != "nn":
                                    continue
                                if not has_nontrivial_input(at):
                                    continue
                                raw = getattr(at, "raw_var_idxs", at.var_idxs)
                                if set(int(v) for v in raw) >= set(range(Nxvars)):
                                    cand_atom = at
                                    break
                            if cand_atom is None:
                                return
                            z_expr = compound_input_expr(cand_atom)
                            max_points_aff = int(getattr(search_hp, "ysearch_outer_affine_max_points", 2048))
                            xs, ys = [], []
                            npts = 0
                            src = _cr_datagens[0]
                            data_iter = src() if callable(src) else src
                            for batch in data_iter:
                                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                                    continue
                                xb, yb = batch[0], batch[1]
                                xs.append(xb)
                                ys.append(yb)
                                npts += int(xb.shape[0])
                                if npts >= max_points_aff:
                                    break
                            if not xs:
                                return
                            Xtmp = torch.cat(xs, dim=0)[:max_points_aff].to(device)
                            Ytmp = torch.cat(ys, dim=0)[:max_points_aff].to(device)
                            peel_names = list(dict.fromkeys(["identity"] + list(y_transform_names)))
                            affine_entries, affine_by_name, identity_confirmed = (
                                _probe_compound_outer_affine_variants(
                                    y_values=Ytmp.view(-1),
                                    z_expr=z_expr,
                                    x_values=Xtmp,
                                    Nxvars=Nxvars,
                                    transform_names=peel_names,
                                    units_payload=units_payload,
                                    rms_thr=float(
                                        getattr(
                                            search_hp,
                                            "ysearch_outer_affine_confirm_rms_rel",
                                            1.0e-6,
                                        )
                                    ),
                                    dom_thr=float(
                                        getattr(
                                            search_hp,
                                            "ysearch_outer_affine_min_domain_frac",
                                            0.995,
                                        )
                                    ),
                                    min_points=int(
                                        getattr(
                                            search_hp,
                                            "ysearch_outer_affine_min_points",
                                            256,
                                        )
                                    ),
                                    min_domain_frac=0.20,
                                )
                            )
                            if not affine_entries:
                                return
                            compound_z_affine_ranked_early = affine_entries
                            if outer_peel_ranked is None:
                                outer_peel_ranked = {}
                            compound_z_affine_by_name.update(affine_by_name)
                            if identity_confirmed:
                                identity_outer_affine_confirmed = True
                            for name_i, meta_i in affine_by_name.items():
                                if str(name_i) == "identity" and bool(
                                    meta_i.get("confirmed", False)
                                ):
                                    identity_outer_affine_confirmed = True
                                if bool(meta_i.get("confirmed", False)):
                                    try:
                                        idx = y_transform_names.index(str(name_i))
                                        candidate_sep_ops[idx] = True
                                    except Exception:
                                        pass
                            outer_peel_ranked["compound_z_affine"] = affine_entries[:8]
                            _print_compound_outer_affine_entries(
                                "[Stage A] Early compound-z outer-affine probe "
                                "(φ(y) ≈ a*z + b, z variants: z, 1/z):",
                                affine_entries,
                                domain_digits=3,
                            )
                        except Exception as e:
                            print(f"[Stage A] Early compound-z outer-affine probe skipped: {e}")

                    _probe_compound_z_outer_affine_now()
                    if identity_outer_affine_confirmed:
                        stageA_status = "compound_outer_confirmed"
                        print(
                            "[Stage A] Identity compound-z outer-affine certificate confirmed; "
                            "skipping Stage-A y-transform refits and deferring to Stage B identity "
                            "with normal unit checks."
                        )
                        _cr_datagens = []

                    for i_op, (name, yt) in enumerate(zip(y_transform_names, y_transforms)):
                        if not _cr_datagens:
                            break
                        # A pure 1D compound is provisional; keep scanning y-transforms.
                        if name == "identity":
                            continue

                        if (
                            (not ysearch_enable)
                            and (not candidate_sep_ops[i_op])
                            and (name not in fallback_names)
                        ):
                            continue

                        # Domain check on raw y data
                        _y_np = np.asarray(y_data, dtype=float)
                        _y_tensor = torch.as_tensor(_y_np, dtype=dtype)
                        valid_frac = 1.0
                        if yt.check_fn is not None:
                            try:
                                _dom_mask = yt.check_fn(_y_tensor)
                                valid_frac = float(_dom_mask.to(torch.float32).mean().item())
                            except Exception:
                                valid_frac = 0.0
                        if valid_frac < ysearch_min_valid_frac:
                            print(
                                f"[Chain-rule screen] Skipping '{name}': domain valid frac "
                                f"{valid_frac:.3f} < {ysearch_min_valid_frac:.3f}"
                            )
                            continue

                        wrapper = VirtualYModel(
                            stageA_model,
                            yt,
                            max_abs_deriv=ysearch_max_virtual_deriv,
                        )
                        cr_proposals = []
                        sep_probe_failures = 0
                        for _cr_dl in _cr_datagens:
                            try:
                                cr_props_i, _, _, _, _ = check_separability(
                                    symb=list(range(Nxvars)),
                                    index=0,
                                    model=wrapper,
                                    datagen=_cr_dl,
                                    precision_sum=_cr_precision,
                                    precision_mult=_cr_precision,
                                    device=device,
                                )
                                if isinstance(cr_props_i, (list, tuple)):
                                    cr_proposals.extend(list(cr_props_i))
                            except Exception as e:
                                sep_probe_failures += 1
                                print(f"[Chain-rule screen] '{name}' separability probe failed: {e}")

                        trig_strength = 0.0
                        for _cr_dl in _cr_datagens:
                            try:
                                trig_spec = discover_trig_axis(
                                    wrapper,
                                    _cr_dl,
                                    Nxvars=int(Nxvars),
                                    device=device,
                                    max_batches=2,
                                    max_points=1024,
                                    n_line=96,
                                    strength_threshold=0.0,
                                )
                                if trig_spec is not None:
                                    trig_strength = max(
                                        trig_strength,
                                        float(max(0.0, float(getattr(trig_spec, "strength", 0.0)))),
                                    )
                            except Exception:
                                continue

                        _mse_vals = []
                        for _cr_dl in _cr_datagens:
                            mse_i = _virtual_proxy_mse(
                                wrapper,
                                yt,
                                _cr_dl,
                                max_batches=2,
                                max_points=2048,
                            )
                            if math.isfinite(float(mse_i)):
                                _mse_vals.append(float(mse_i))
                        virtual_mse = (
                            float(sum(_mse_vals) / max(1, len(_mse_vals)))
                            if _mse_vals
                            else float("inf")
                        )

                        cr_has_split = len(cr_proposals) > 0
                        print(
                            f"[Chain-rule screen] '{name}': "
                            f"{'separability FOUND' if cr_has_split else 'no separability'}"
                            f" ({len(cr_proposals)} proposal(s)), "
                            f"trig_strength={trig_strength:.3g}, virtual_mse={virtual_mse:.3g}, "
                            f"sep_probe_failures={int(sep_probe_failures)}"
                        )
                        if cr_has_split:
                            best_quick_idx = i_op if best_quick_idx is None else best_quick_idx

                        _outer_aff = compound_z_affine_by_name.get(str(name), {})
                        _joint_cert = derive_joint_homogeneity_certificate(
                            _joint_scale_certificates,
                            getattr(yt, "homogeneity_power", None),
                        )
                        virtual_hints.append(
                            VirtualProbeHint(
                                idx=int(i_op),
                                name=str(name),
                                domain_ok_frac=float(valid_frac),
                                candidate_flag=bool(candidate_sep_ops[i_op]),
                                sep_has_split=bool(cr_has_split),
                                sep_proposals=int(len(cr_proposals)),
                                trig_strength=float(trig_strength),
                                virtual_mse=float(virtual_mse),
                                outer_affine_confirmed=bool(_outer_aff.get("confirmed", False)),
                                outer_affine_rms_rel=float(_outer_aff.get("rms_rel", float("inf"))),
                                outer_affine_domain_ok_frac=float(_outer_aff.get("domain_ok_frac", 0.0)),
                                joint_homogeneity_verified=bool(_joint_cert),
                                joint_homogeneity_indices=(
                                    tuple(_joint_cert["indices"]) if _joint_cert else ()
                                ),
                                joint_homogeneity_degree=(
                                    float(_joint_cert["degree"])
                                    if _joint_cert
                                    else float("nan")
                                ),
                                joint_homogeneity_rel_std=(
                                    float(_joint_cert["rel_std"])
                                    if _joint_cert
                                    else float("inf")
                                ),
                                joint_homogeneity_n_points=(
                                    int(_joint_cert["n_points"]) if _joint_cert else 0
                                ),
                            )
                        )

                        # Legacy mode keeps first split behavior.
                        if (not ysearch_enable) and cr_has_split:
                            break

                    # If quick scan found a winner, only refit that one.
                    # A pure 1D compound does not suppress y-search; it only seeds
                    # the outer-affine certificate/ranking path above.
                    if identity_outer_affine_confirmed:
                        promising_indices = []
                        stageB_virtual_top_names = []
                    elif ysearch_enable and virtual_hints:
                        ranked_hints = rank_virtual_hints(virtual_hints)
                        print("[Virtual y-search] Ranked transform hints:")
                        for rank_i, h in enumerate(ranked_hints, start=1):
                            print(
                                f"  {rank_i:2d}) {h.name:12s} split={int(h.sep_has_split)} "
                                f"n_props={int(h.sep_proposals):2d} trig={h.trig_strength:.3g} "
                                f"vmse={h.virtual_mse:.3g} dom={h.domain_ok_frac:.3f} "
                                f"outer={int(getattr(h, 'outer_affine_confirmed', False))} "
                                f"joint_h={int(h.joint_homogeneity_verified)} "
                                f"candidate={int(h.candidate_flag)}"
                            )
                        promising_indices, _selection_reasons = select_virtual_portfolio(
                            ranked_hints,
                            ysearch_expand_k,
                            margin_decades=float(getattr(search_hp, "ysearch_portfolio_margin_decades", 0.25)),
                            max_k=int(getattr(search_hp, "ysearch_portfolio_max_k", max(3, ysearch_expand_k))),
                        )
                        _selected_set = set(int(i) for i in promising_indices)
                        stageB_virtual_portfolio = []
                        for _rank, _hint in enumerate(ranked_hints, start=1):
                            _idx = int(_hint.idx)
                            _joint = bool(_hint.joint_homogeneity_verified)
                            stageB_virtual_portfolio.append(
                                {
                                    "rank": int(_rank),
                                    "idx": _idx,
                                    "name": str(_hint.name),
                                    "selected": _idx in _selected_set,
                                    "selection_reason": _selection_reasons.get(
                                        _idx, "not_selected"
                                    ),
                                    "virtual_mse": (
                                        float(_hint.virtual_mse)
                                        if math.isfinite(float(_hint.virtual_mse))
                                        else None
                                    ),
                                    "joint_homogeneity_verified": _joint,
                                    "joint_homogeneity_indices": (
                                        list(_hint.joint_homogeneity_indices)
                                        if _joint
                                        else []
                                    ),
                                    "joint_homogeneity_degree": (
                                        float(_hint.joint_homogeneity_degree)
                                        if _joint
                                        else None
                                    ),
                                    "joint_homogeneity_rel_std": (
                                        float(_hint.joint_homogeneity_rel_std)
                                        if _joint
                                        else None
                                    ),
                                    "joint_homogeneity_n_points": (
                                        int(_hint.joint_homogeneity_n_points)
                                        if _joint
                                        else 0
                                    ),
                                }
                            )
                        print(
                            f"[Virtual y-search] Selected top-{len(promising_indices)} transform(s): "
                            + ", ".join(y_transform_names[i] for i in promising_indices)
                        )
                        stageB_virtual_top_names = [
                            y_transform_names[i] for i in promising_indices
                        ]
                    elif best_quick_idx is not None:
                        promising_indices = [best_quick_idx]
                        stageB_virtual_top_names = [
                            y_transform_names[i] for i in promising_indices
                        ]
                    else:
                        # Otherwise, refit only the transforms predicted as candidates.
                        promising_indices = [
                            i
                            for i, nm in enumerate(y_transform_names)
                            if (nm != "identity") and candidate_sep_ops[i]
                        ]
                        stageB_virtual_top_names = [
                            y_transform_names[i] for i in promising_indices[:3]
                        ]

                    # 3) Refit promising transforms using a depth-1 controller
                    candidate_names = [y_transform_names[i] for i in promising_indices]
                    if candidate_names:
                        from nestynet_sr.sr_search.ysearch_controller import (
                            YSearchControllerConfig,
                            YSearchState,
                            make_stagea_state_key,
                            run_ysearch_beam,
                            run_ysearch_beam_with_split_recursion,
                        )
                        from nestynet_sr.sr_search.ysearch_signals import (
                            format_progress_reasons,
                            normalize_signal_dict,
                            structural_progress,
                        )

                        _parent_loss = getattr(stageA_model, "_best_val_loss_base", None)
                        try:
                            _parent_loss = (
                                float(_parent_loss)
                                if (_parent_loss is not None and math.isfinite(float(_parent_loss)))
                                else float("inf")
                            )
                        except Exception:
                            _parent_loss = float("inf")

                        parent_signals = normalize_signal_dict(
                            getattr(stageA_model, "_stageA_signals", None)
                        )
                        if "sep_score" not in parent_signals:
                            parent_signals["sep_score"] = float(
                                1.0 if bool(stageA_found_separability) else 0.0
                            )
                        if "best_split_score" not in parent_signals:
                            parent_signals["best_split_score"] = float(
                                1.0 if bool(stageA_found_separability) else 0.0
                            )
                        if "trig_affine_conf" not in parent_signals:
                            parent_signals["trig_affine_conf"] = 0.0

                        trig_affine_thr = float(
                            getattr(search_hp, "ysearch_trigger_trig_affine_conf", 0.90)
                        )
                        sep_min_thr = float(getattr(search_hp, "ysearch_trigger_sep_min", 0.80))
                        sep_delta_thr = float(
                            getattr(search_hp, "ysearch_trigger_sep_delta", 0.25)
                        )
                        split_score_thr = float(
                            getattr(search_hp, "ysearch_trigger_split_score", 0.90)
                        )
                        split_margin_thr = float(
                            getattr(search_hp, "ysearch_trigger_split_margin", 0.15)
                        )

                        def _eval_ysearch_state(y_stack):
                            nonlocal global_best_val_loss_base
                            stack_names = tuple(str(n) for n in y_stack if str(n))
                            if not stack_names:
                                return None
                            last_name = stack_names[-1]
                            try:
                                i_cand = y_transform_names.index(last_name)
                            except ValueError:
                                return None
                            try:
                                y_op_stack, y_op_inv_stack, y_stack_name = compose_y_stack_ops(
                                    stack_names, transforms=y_transforms
                                )
                            except Exception as e:
                                print(
                                    f"[ysearch] skipping invalid stack {stack_names}: {type(e).__name__}: {e}"
                                )
                                return None

                            tf_model_output = _model_path(y_stack_name, sep=False)
                            tf_model_sep_output = _model_path(y_stack_name, sep=True)
                            seed_ast = stageA_ast if stageA_ast is not None else initial_ast
                            try:
                                seed_ast = copy.deepcopy(seed_ast)
                            except Exception:
                                pass

                            out = stageA_analyze(
                                i_op=i_cand,
                                y_op=y_op_stack,
                                y_op_inv=y_op_inv_stack,
                                # Proposal mode: do not hard-gate stacked transforms.
                                candidate_sep_ops=[True] * len(y_transform_names),
                                y_transform_names=y_transform_names,
                                initial_ast=seed_ast,
                                filepath=stageA_filepath_arg,
                                Nxvars=Nxvars,
                                y_med=y_med,
                                y_mad=y_mad,
                                np_dtype=np_dtype,
                                dtype=dtype,
                                device=device,
                                data_hp=data_hp,
                                model_hp=model_hp,
                                lm_hp=lm_hp,
                                search_hp=search_hp,
                                leaf_builder=leaf_builder,
                                model_output=tf_model_output,
                                model_sep_output=tf_model_sep_output,
                                mode="full",
                                units_payload=units_payload,
                                enforce_units=bool(args.enforce_units),
                                units_policy=str(args.units_policy),
                                nn_units_semantics=str(args.nn_units_semantics),
                                y_log_dynamic_range=y_log_dynamic_range,
                                y_abs_median=y_abs_median,
                                global_best_val_loss_base=global_best_val_loss_base,
                                y_raw_full=y_data,
                                noise_sigma_y=noise_sigma_y,
                                noise_floor_mc_samples=args.noise_floor_mc_samples,
                                fast=False,
                            )

                            if out.model is not None and math.isfinite(float(out.val_loss_base)):
                                nonlocal_best = float(out.val_loss_base)
                                # Update global best-loss guard
                                if (
                                    global_best_val_loss_base is None
                                    or nonlocal_best < float(global_best_val_loss_base)
                                ):
                                    global_best_val_loss_base = nonlocal_best

                            if out.model is None:
                                return None

                            split_success = _split_success(
                                out.success, out.rest_add, out.rest_mult
                            )
                            parent_state_signals = dict(parent_signals)
                            if len(stack_names) > 1:
                                p_stack = tuple(stack_names[:-1])
                                p_payload = stageA_state_cache.get(_make_stagea_key(p_stack))
                                if p_payload is None:
                                    p_payload = _eval_ysearch_state(p_stack)
                                    if p_payload is not None:
                                        stageA_state_cache[_make_stagea_key(p_stack)] = p_payload
                                if isinstance(p_payload, dict):
                                    parent_state_signals = normalize_signal_dict(
                                        p_payload.get("stagea_signals", {})
                                    )
                            parent_state_signals["split_success"] = float(
                                1.0 if bool(parent_state_signals.get("split_success", 0.0) >= 0.5) else 0.0
                            )
                            # Check principal-branch invertibility for trig transforms.
                            _inv_branch_ok = 1.0
                            _margin = 0.10
                            if last_name == "sin":
                                _inv_branch_ok = float(
                                    _y_min >= -math.pi / 2 - _margin
                                    and _y_max <= math.pi / 2 + _margin
                                )
                            elif last_name == "cos":
                                _inv_branch_ok = float(
                                    _y_min >= 0.0 - _margin
                                    and _y_max <= math.pi + _margin
                                )
                            elif last_name == "tan":
                                _inv_branch_ok = float(
                                    _y_min >= -math.pi / 2 + _margin
                                    and _y_max <= math.pi / 2 - _margin
                                )
                            elif last_name in ("square", "sqrt1p"):
                                _inv_branch_ok = float(_y_min >= -1e-8)

                            _outer_aff = (
                                compound_z_affine_by_name.get(str(last_name), {})
                                if len(stack_names) == 1
                                else {}
                            )
                            _stagea_sig = dict(getattr(out, "signals", {}) or {})
                            if bool(getattr(out, "full_compound_solved", False)):
                                _stagea_sig["full_compound_compressed"] = 1.0
                                _stagea_sig["full_compound_solved"] = 0.0
                            if bool(_outer_aff.get("confirmed", False)) and _inv_branch_ok >= 0.5:
                                _stagea_sig["outer_affine_confirmed"] = 1.0
                                _stagea_sig["outer_affine_rms_rel"] = float(_outer_aff.get("rms_rel", float("inf")))
                                _branch_confirmation = "outer_affine_confirmed"
                            elif bool(split_success):
                                _branch_confirmation = "split_confirmed"
                            elif bool(getattr(out, "full_compound_solved", False)):
                                _branch_confirmation = "provisional"
                            else:
                                _branch_confirmation = "unresolved"

                            return {
                                "name": str(last_name),
                                "i_op": int(i_cand),
                                "out": out,
                                "y_stack": stack_names,
                                "y_stack_name": str(y_stack_name),
                                "y_op": y_op_stack,
                                "y_op_inv": y_op_inv_stack,
                                "model_path": tf_model_output,
                                "model_sep_path": tf_model_sep_output,
                                "val_loss_base": float(out.val_loss_base),
                                "split_success": bool(split_success),
                                "stagea_signals": _stagea_sig,
                                "parent_stagea_signals": dict(parent_state_signals),
                                "branch_confirmation": str(_branch_confirmation),
                                "outer_affine": dict(_outer_aff),
                                "split_plans": list(getattr(out, "split_plans", []) or []),
                                "inv_branch_ok": float(_inv_branch_ok),
                            }

                        def _ysearch_strong_trigger(payload):
                            if not isinstance(payload, dict):
                                return False
                            child_signals = normalize_signal_dict(payload.get("stagea_signals", {}))
                            child_signals["split_success"] = float(
                                1.0 if bool(payload.get("split_success", False)) else 0.0
                            )
                            parent_state_signals = normalize_signal_dict(
                                payload.get("parent_stagea_signals", parent_signals)
                            )
                            ok, reasons = structural_progress(
                                parent_state_signals,
                                child_signals,
                                trig_affine_thr=trig_affine_thr,
                                sep_min_thr=sep_min_thr,
                                sep_delta_thr=sep_delta_thr,
                                split_score_thr=split_score_thr,
                                split_margin_thr=split_margin_thr,
                            )
                            payload["ysearch_trigger_reasons"] = list(reasons)
                            payload["ysearch_trigger_reason_str"] = format_progress_reasons(reasons)
                            return bool(ok)

                        _data_sig = (
                            tuple(str(p) for p in filepaths)
                            if isinstance(stageA_filepath_arg, (list, tuple))
                            else (str(stageA_filepath_arg),)
                        )
                        _model_sig = (
                            int(getattr(model_hp, "num_segments_min", 0)),
                            int(getattr(model_hp, "num_segments_max", 0)),
                            int(getattr(model_hp, "model_size_target", 0)),
                            int(1 if bool(dual_layer) else 0),
                        )
                        _train_sig = (
                            str(getattr(lm_hp, "strategy", "")),
                            int(getattr(lm_hp, "epochs", 0)),
                            int(getattr(lm_hp, "epochs_min", 0)),
                            int(getattr(data_hp, "batch_size", 0)),
                            int(getattr(data_hp, "ndata_select", 0)),
                            int(getattr(data_hp, "ndata_select_val", 0)),
                            float(getattr(search_hp, "precision_derivs_d2y", 0.0)),
                            str(getattr(search_hp, "stageA_move_policy", "")),
                            int(1 if bool(args.enforce_units) else 0),
                            str(args.units_policy),
                            str(args.nn_units_semantics),
                        )
                        try:
                            _seed_sig = int(torch.initial_seed())
                        except Exception:
                            _seed_sig = 0

                        def _make_stagea_key(y_stack):
                            stack_names = tuple(str(n) for n in y_stack if str(n))
                            return make_stagea_state_key(
                                y_stack_sig=stack_names,
                                data_sig=_data_sig,
                                model_sig=_model_sig,
                                train_cfg_sig=_train_sig,
                                seed=_seed_sig,
                                fast=False,
                            )

                        def _split_plan_score(sp):
                            try:
                                if isinstance(sp, dict):
                                    return float(sp.get("score", 0.0))
                                return float(getattr(sp, "score", 0.0))
                            except Exception:
                                return 0.0

                        def _split_plan_sig(sp):
                            kind = "split"
                            part = None
                            try:
                                if isinstance(sp, dict):
                                    kind = str(sp.get("kind", "split"))
                                    part = sp.get("partition", None)
                                else:
                                    kind = str(getattr(sp, "kind", "split"))
                                    part = getattr(sp, "partition", None)
                            except Exception:
                                kind = "split"
                                part = None
                            part_repr = repr(part)
                            if len(part_repr) > 160:
                                part_repr = part_repr[:160]
                            return (kind, part_repr)

                        def _split_plans_from_payload(payload):
                            plans = payload.get("split_plans", []) if isinstance(payload, dict) else []
                            if not isinstance(plans, (list, tuple)):
                                return []
                            ranked = sorted(list(plans), key=_split_plan_score, reverse=True)
                            return ranked

                        def _recurse_split_state(parent_state, split_plan):
                            parent_stack = tuple(
                                str(n) for n in getattr(parent_state, "y_stack", tuple()) if str(n)
                            )
                            if not parent_stack:
                                return None

                            pkey = _make_stagea_key(parent_stack)
                            parent_payload = stageA_state_cache.get(pkey)
                            if parent_payload is None:
                                parent_payload = _eval_ysearch_state(parent_stack)
                                if parent_payload is None:
                                    return None
                                stageA_state_cache[pkey] = parent_payload

                            parent_out = parent_payload.get("out", None) if isinstance(parent_payload, dict) else None
                            seed_ast = getattr(parent_out, "current_ast", None) if parent_out is not None else None
                            if seed_ast is None:
                                return None

                            split_sig = _split_plan_sig(split_plan)
                            print(
                                f"[ysearch] split-recursing from state={parent_stack} "
                                f"plan={split_sig[0]} part={split_sig[1]}"
                            )

                            def _eval_split_state(y_stack):
                                nonlocal global_best_val_loss_base
                                stack_names = tuple(str(n) for n in y_stack if str(n))
                                if len(stack_names) <= len(parent_stack):
                                    return None
                                last_name = stack_names[-1]
                                try:
                                    i_cand = y_transform_names.index(last_name)
                                except ValueError:
                                    return None
                                try:
                                    y_op_stack, y_op_inv_stack, y_stack_name = compose_y_stack_ops(
                                        stack_names, transforms=y_transforms
                                    )
                                except Exception as e:
                                    print(
                                        f"[ysearch] split-recursion skipping invalid stack {stack_names}: "
                                        f"{type(e).__name__}: {e}"
                                    )
                                    return None

                                tf_model_output = _model_path(y_stack_name, sep=False)
                                tf_model_sep_output = _model_path(y_stack_name, sep=True)
                                try:
                                    _seed_ast = copy.deepcopy(seed_ast)
                                except Exception:
                                    _seed_ast = seed_ast

                                out = stageA_analyze(
                                    i_op=i_cand,
                                    y_op=y_op_stack,
                                    y_op_inv=y_op_inv_stack,
                                    candidate_sep_ops=[True] * len(y_transform_names),
                                    y_transform_names=y_transform_names,
                                    initial_ast=_seed_ast,
                                    filepath=stageA_filepath_arg,
                                    Nxvars=Nxvars,
                                    y_med=y_med,
                                    y_mad=y_mad,
                                    np_dtype=np_dtype,
                                    dtype=dtype,
                                    device=device,
                                    data_hp=data_hp,
                                    model_hp=model_hp,
                                    lm_hp=lm_hp,
                                    search_hp=search_hp,
                                    leaf_builder=leaf_builder,
                                    model_output=tf_model_output,
                                    model_sep_output=tf_model_sep_output,
                                    mode="full",
                                    units_payload=units_payload,
                                    enforce_units=bool(args.enforce_units),
                                    units_policy=str(args.units_policy),
                                    nn_units_semantics=str(args.nn_units_semantics),
                                    y_log_dynamic_range=y_log_dynamic_range,
                                    y_abs_median=y_abs_median,
                                    global_best_val_loss_base=global_best_val_loss_base,
                                    y_raw_full=y_data,
                                    noise_sigma_y=noise_sigma_y,
                                    noise_floor_mc_samples=args.noise_floor_mc_samples,
                                    fast=False,
                                )

                                if out.model is not None and math.isfinite(float(out.val_loss_base)):
                                    nonlocal_best = float(out.val_loss_base)
                                    if (
                                        global_best_val_loss_base is None
                                        or nonlocal_best < float(global_best_val_loss_base)
                                    ):
                                        global_best_val_loss_base = nonlocal_best

                                if out.model is None:
                                    return None

                                split_success = _split_success(
                                    out.success, out.rest_add, out.rest_mult
                                )
                                parent_state_signals = normalize_signal_dict(
                                    parent_payload.get("stagea_signals", {})
                                )
                                if len(stack_names) > len(parent_stack) + 1:
                                    p_stack = tuple(stack_names[:-1])
                                    p_payload = stageA_state_cache.get(_make_split_key(p_stack))
                                    if p_payload is None:
                                        p_payload = _eval_split_state(p_stack)
                                        if p_payload is not None:
                                            stageA_state_cache[_make_split_key(p_stack)] = p_payload
                                    if isinstance(p_payload, dict):
                                        parent_state_signals = normalize_signal_dict(
                                            p_payload.get("stagea_signals", {})
                                        )
                                parent_state_signals["split_success"] = float(
                                    1.0 if bool(parent_state_signals.get("split_success", 0.0) >= 0.5) else 0.0
                                )
                                # Check principal-branch invertibility for trig transforms.
                                _inv_branch_ok = 1.0
                                _margin = 0.10
                                if last_name == "sin":
                                    _inv_branch_ok = float(
                                        _y_min >= -math.pi / 2 - _margin
                                        and _y_max <= math.pi / 2 + _margin
                                    )
                                elif last_name == "cos":
                                    _inv_branch_ok = float(
                                        _y_min >= 0.0 - _margin
                                        and _y_max <= math.pi + _margin
                                    )
                                elif last_name == "tan":
                                    _inv_branch_ok = float(
                                        _y_min >= -math.pi / 2 + _margin
                                        and _y_max <= math.pi / 2 - _margin
                                    )
                                elif last_name in ("square", "sqrt1p"):
                                    _margin_sq = max(1e-8, 0.01 * (_y_max - _y_min))
                                    _inv_branch_ok = float(_y_min >= -_margin_sq)

                                _outer_aff = (
                                    compound_z_affine_by_name.get(str(last_name), {})
                                    if len(stack_names) == 1
                                    else {}
                                )
                                _stagea_sig = dict(getattr(out, "signals", {}) or {})
                                if bool(getattr(out, "full_compound_solved", False)):
                                    _stagea_sig["full_compound_compressed"] = 1.0
                                    _stagea_sig["full_compound_solved"] = 0.0
                                if bool(_outer_aff.get("confirmed", False)) and _inv_branch_ok >= 0.5:
                                    _stagea_sig["outer_affine_confirmed"] = 1.0
                                    _stagea_sig["outer_affine_rms_rel"] = float(_outer_aff.get("rms_rel", float("inf")))
                                    _branch_confirmation = "outer_affine_confirmed"
                                elif bool(split_success):
                                    _branch_confirmation = "split_confirmed"
                                elif bool(getattr(out, "full_compound_solved", False)):
                                    _branch_confirmation = "provisional"
                                else:
                                    _branch_confirmation = "unresolved"

                                return {
                                    "name": str(last_name),
                                    "i_op": int(i_cand),
                                    "out": out,
                                    "y_stack": stack_names,
                                    "y_stack_name": str(y_stack_name),
                                    "y_op": y_op_stack,
                                    "y_op_inv": y_op_inv_stack,
                                    "model_path": tf_model_output,
                                    "model_sep_path": tf_model_sep_output,
                                    "val_loss_base": float(out.val_loss_base),
                                    "split_success": bool(split_success),
                                    "stagea_signals": _stagea_sig,
                                    "parent_stagea_signals": dict(parent_state_signals),
                                    "branch_confirmation": str(_branch_confirmation),
                                    "outer_affine": dict(_outer_aff),
                                    "split_plans": list(getattr(out, "split_plans", []) or []),
                                    "split_plan_parent": split_sig,
                                    "inv_branch_ok": float(_inv_branch_ok),
                                }

                            def _make_split_key(y_stack):
                                stack_names = tuple(str(n) for n in y_stack if str(n))
                                return make_stagea_state_key(
                                    y_stack_sig=stack_names + ("__split__", split_sig[0], split_sig[1]),
                                    data_sig=_data_sig,
                                    model_sig=_model_sig,
                                    train_cfg_sig=_train_sig,
                                    seed=_seed_sig,
                                    fast=False,
                                )

                            _parent_loss_local = parent_payload.get(
                                "val_loss_base", float("inf")
                            ) if isinstance(parent_payload, dict) else float("inf")
                            try:
                                _parent_loss_local = (
                                    float(_parent_loss_local)
                                    if math.isfinite(float(_parent_loss_local))
                                    else float("inf")
                                )
                            except Exception:
                                _parent_loss_local = float("inf")

                            _child_eval_budget = max(
                                0, int(getattr(search_hp, "ysearch_max_state_evals", 0))
                            )
                            if _child_eval_budget > 0:
                                _child_eval_budget = max(
                                    1, min(_child_eval_budget, max(1, int(getattr(search_hp, "ysearch_expand_k", 2))))
                                )

                            child_cfg = YSearchControllerConfig(
                                max_depth=max(len(parent_stack) + 1, int(getattr(search_hp, "ysearch_depth", 1))),
                                beam=max(1, int(getattr(search_hp, "ysearch_beam", 3))),
                                expand_k=min(
                                    max(1, int(getattr(search_hp, "ysearch_expand_k", 2))),
                                    max(1, len(candidate_names)),
                                ),
                                confirm_improve_ratio=float(
                                    getattr(search_hp, "ysearch_confirm_improve_ratio", 0.3)
                                ),
                                eps_parent_loss=1.0e-12,
                                max_state_evals=_child_eval_budget,
                                max_recursive_branches=0,
                                max_split_plans_per_state=0,
                            )
                            return run_ysearch_beam(
                                parent_state=YSearchState(y_stack=parent_stack),
                                candidate_names=candidate_names,
                                evaluate_state=_eval_split_state,
                                parent_val_loss_base=_parent_loss_local,
                                cfg=child_cfg,
                                strong_structure_trigger_fn=_ysearch_strong_trigger,
                                log_fn=print,
                                stagea_cache=stageA_state_cache,
                                make_key_fn=_make_split_key,
                            )

                        y_cfg = YSearchControllerConfig(
                            max_depth=max(1, int(getattr(search_hp, "ysearch_depth", 1))),
                            beam=max(1, int(getattr(search_hp, "ysearch_beam", 3))),
                            expand_k=min(
                                max(1, int(getattr(search_hp, "ysearch_expand_k", 2))),
                                max(1, len(candidate_names)),
                            ),
                            confirm_improve_ratio=float(
                                getattr(search_hp, "ysearch_confirm_improve_ratio", 0.3)
                            ),
                            eps_parent_loss=1.0e-12,
                            max_state_evals=max(
                                0, int(getattr(search_hp, "ysearch_max_state_evals", 0))
                            ),
                            max_recursive_branches=max(
                                0,
                                int(getattr(search_hp, "ysearch_max_recursive_branches", 0)),
                            ),
                            max_split_plans_per_state=max(
                                0,
                                int(
                                    getattr(
                                        search_hp,
                                        "ysearch_max_split_plans_per_state",
                                        1,
                                    )
                                ),
                            ),
                        )
                        if int(getattr(y_cfg, "max_recursive_branches", 0)) > 0:
                            y_res = run_ysearch_beam_with_split_recursion(
                                parent_state=YSearchState(y_stack=tuple()),
                                candidate_names=candidate_names,
                                evaluate_state=_eval_ysearch_state,
                                parent_val_loss_base=_parent_loss,
                                cfg=y_cfg,
                                strong_structure_trigger_fn=_ysearch_strong_trigger,
                                log_fn=print,
                                stagea_cache=stageA_state_cache,
                                make_key_fn=_make_stagea_key,
                                split_plans_fn=_split_plans_from_payload,
                                recurse_split_fn=_recurse_split_state,
                            )
                        else:
                            y_res = run_ysearch_beam(
                                parent_state=YSearchState(y_stack=tuple()),
                                candidate_names=candidate_names,
                                evaluate_state=_eval_ysearch_state,
                                parent_val_loss_base=_parent_loss,
                                cfg=y_cfg,
                                strong_structure_trigger_fn=_ysearch_strong_trigger,
                                log_fn=print,
                                stagea_cache=stageA_state_cache,
                                make_key_fn=_make_stagea_key,
                            )
                        if y_res.frontier_trials:
                            _beam_names = ", ".join(
                                encode_y_stack_name(t.state.y_stack)
                                for t in y_res.frontier_trials
                            )
                            print(
                                f"[ysearch] frontier (beam={len(y_res.frontier_trials)}): {_beam_names}"
                            )

                        _commit_trial = None
                        _coe_ybranch_kept_identity = False
                        _confirmed_trials = []
                        if y_res.all_trials:
                            _confirmed_trials = [
                                t for t in y_res.all_trials
                                if _payload_is_confirmed(t.payload if isinstance(t.payload, dict) else {})
                            ]
                            if _confirmed_trials:
                                _commit_trial = min(_confirmed_trials, key=_trial_adjudication_key)
                                if str(getattr(lm_hp, "coe_mode", "off") or "off") in {
                                    "committee_gated",
                                    "reservoir_discovery",
                                }:
                                    _trial_branches = []
                                    _legacy_branch = None
                                    for _trial in _confirmed_trials:
                                        _payload = _trial.payload if isinstance(_trial.payload, dict) else {}
                                        _out = _payload.get("out")
                                        _branch = {
                                            "branch_id": encode_y_stack_name(_trial.state.y_stack),
                                            "name": str(_payload.get("y_stack_name") or encode_y_stack_name(_trial.state.y_stack)),
                                            "model": getattr(_out, "model", None) if _out is not None else None,
                                            "y_op": _payload.get("y_op"),
                                            "y_op_inv": _payload.get("y_op_inv"),
                                            "confirmation": _payload_confirmation_status(_payload),
                                            "rank_key": _trial_adjudication_key(_trial),
                                            "trial": _trial,
                                        }
                                        _trial_branches.append(_branch)
                                        if _trial is _commit_trial:
                                            _legacy_branch = _branch
                                    _identity_branch = {
                                        "branch_id": "identity",
                                        "name": "identity",
                                        "model": stageA_model,
                                        "y_op": yt_id.np_op if "yt_id" in locals() else None,
                                        "y_op_inv": yt_id.torch_inv if "yt_id" in locals() else None,
                                        "confirmation": "identity",
                                        "rank_key": (0, 0, 0, 0, "identity"),
                                    }
                                    _selected_branch, _coe_reason, _coe_summary = _coe_stageA_ybranch_committee_rank(
                                        lm_hp=lm_hp,
                                        filepath=filepath,
                                        identity_branch=_identity_branch,
                                        candidate_branches=_trial_branches,
                                        legacy_selected_branch=_legacy_branch,
                                        dtype=dtype,
                                        device=device,
                                    )
                                    coe_stageA_ybranch_committee = _coe_summary
                                    print("\n" + _format_coe_stageA_ybranch_committee_report(_coe_summary))
                                    if isinstance(_selected_branch, dict):
                                        _commit_trial = _selected_branch.get("trial", _commit_trial)
                                    else:
                                        _commit_trial = None
                                        _coe_ybranch_kept_identity = True

                        if _commit_trial is None and y_res.best_trial is not None:
                            if _coe_ybranch_kept_identity:
                                print(
                                    "[ysearch] CoE y-branch committee kept the identity baseline; "
                                    "confirmed transformed branches remain uncommitted."
                                )
                            else:
                                _bp = y_res.best_trial.payload if isinstance(y_res.best_trial.payload, dict) else {}
                                _status = _payload_confirmation_status(_bp)
                                print(
                                    f"[ysearch] No confirmed y-branch to commit; keeping identity baseline "
                                    f"(best provisional={encode_y_stack_name(y_res.best_trial.state.y_stack)}, "
                                    f"status={_status})."
                                )

                        if _commit_trial is not None:
                            _best_payload = _commit_trial.payload
                            _best_out = _best_payload.get("out")
                            _status = _payload_confirmation_status(_best_payload)
                            print(
                                f"[ysearch] Committing confirmed branch "
                                f"{encode_y_stack_name(_commit_trial.state.y_stack)} ({_status})."
                            )
                            if _best_out is not None and _best_out.model is not None:
                                final_model = _best_out.model
                                final_ast = _best_out.current_ast
                                final_y_op = _best_payload.get("y_op")
                                final_y_op_inv = _best_payload.get("y_op_inv")
                                final_y_op_name = str(_best_payload.get("y_stack_name"))
                                rest_add_final = _best_out.rest_add
                                rest_mult_final = _best_out.rest_mult

                                try:
                                    _best_model_path = _best_payload.get("model_path")
                                    _best_sep_path = _best_payload.get("model_sep_path")
                                    if _best_model_path and os.path.exists(_best_model_path):
                                        shutil.copyfile(_best_model_path, model_output)
                                    if _best_sep_path and os.path.exists(_best_sep_path):
                                        shutil.copyfile(_best_sep_path, model_sep_output)
                                except Exception as e:
                                    print(
                                        f"Warning: failed to copy confirmed ysearch model '{final_y_op_name}' to canonical path: {e}"
                                    )

                            separability_success = bool(
                                separability_success
                                or bool(_best_payload.get("split_success", False))
                            )
                            _commit_status = _payload_confirmation_status(_best_payload)
                            if _commit_status == "split_confirmed":
                                stageA_status = "split_confirmed"
                            elif _commit_status == "outer_affine_confirmed":
                                stageA_status = "compound_outer_confirmed"

                    # 4) If we didn't find separability under any y-transform, try an
                    #    outer-peel *proposal* (identity vs square) without committing.
                    #    Also run when a 1D compound was solved (to discover outer link e.g. sin(y) ≈ z).
                    _should_outer_peel = (
                        (not separability_success)
                        or compound_is_1d
                    )
                    if identity_outer_affine_confirmed:
                        _should_outer_peel = False
                        print(
                            "[Stage A] Outer-peel diagnostics skipped: "
                            "identity compound-z certificate already confirmed."
                        )
                    if (
                        _should_outer_peel
                        and (idx_identity is not None)
                        and (stageA_model is not None)
                    ):
                        try:
                            from nestynet_sr.sr_search.data_utils import build_datasets
                            from nestynet_sr.sr_search.outer_peel import (
                                propose_outer_y_transform,
                                rank_outer_y_transforms,
                            )

                            # Build an identity-space loader for scoring.
                            _, _, datagen_train_noshuffle, _ = build_datasets(
                                filepath, Nxvars, np_dtype, data_hp, y_op=None
                            )

                            if datagen_train_noshuffle is not None:
                                name_to_tf = {t.name: t for t in y_transforms}

                                # Conservative square-vs-identity diagnostic
                                decision = None
                                if "square" in y_transform_names:
                                    decision = propose_outer_y_transform(
                                        identity_model=stageA_model,
                                        identity_datagen=datagen_train_noshuffle,
                                        y_data_np=y_data,
                                        device=device,
                                        min_good_axes=(2 if int(Nxvars) >= 3 else 1),
                                    )

                                    p = decision.proposal
                                    d = decision.diagnostics
                                    outer_peel_square_decision = {
                                        "prefer_square": bool(decision.prefer),
                                        "gain": float(p.improvement),
                                        "score": float(p.score),
                                        "best_axis": p.details.get("best_axis", None),
                                        "frac_negative_y": d.get("frac_negative_y", None),
                                        "reason": d.get("reason", None),
                                        "good_axes": d.get("good_axes", []),
                                        "ignored_axes": d.get("ignored_axes", []),
                                        "num_good_axes": d.get("num_good_axes", 0),
                                        "required_good_axes": d.get("required_good_axes", 0),
                                        "multi_axis_gain": d.get("multi_axis_gain", None),
                                        "multi_axis_gain_min": d.get(
                                            "multi_axis_gain_min", None
                                        ),
                                    }
                                    print(
                                        "[Stage A] Outer-peel proposal (square vs identity): "
                                        f"prefer_square={decision.prefer}, "
                                        f"gain≈{p.improvement:.3g}, score≈{p.score:.3g}, "
                                        f"axis={p.details.get('best_axis', None)}, "
                                        f"frac_neg_y={d.get('frac_negative_y', None):.3g}"
                                    )

                                # Ranked proposals (heuristic, no commit)
                                probe_names = [
                                    n
                                    for n in (
                                        "identity",
                                        "square",
                                        "log",
                                        "exp",
                                        "reciprocal",
                                        "sqrt",
                                        "sin",
                                        "cos",
                                        "tan",
                                        "arcsin",
                                        "arccos",
                                        "arctan",
                                    )
                                    if n in name_to_tf
                                ]
                                if len(probe_names) >= 1:
                                    base_spec, ranked = rank_outer_y_transforms(
                                        identity_model=stageA_model,
                                        identity_datagen=datagen_train_noshuffle,
                                        Nxvars=Nxvars,
                                        transform_names=probe_names,
                                        device=device,
                                        max_points=2048,
                                        min_domain_frac=0.0,
                                    )
                                    outer_peel_ranked = {
                                        "baseline": base_spec.__dict__,
                                        "ranked": [s.__dict__ for s in ranked],
                                    }
                                    print(
                                        "[Stage A] Outer-peel ranked y-transforms (heuristic, no commit):"
                                    )
                                    topk = min(8, len(ranked))
                                    for i in range(topk):
                                        s = ranked[i]
                                        _nls = ""
                                        _struct = ""
                                        try:
                                            _det = getattr(s, "details", {}) or {}
                                            _nls_err = float(
                                                _det.get("nls_subst_err", float("inf"))
                                            )
                                            _nls_tf = _det.get("nls_subst_transform", None)
                                            _nls_col = _det.get("nls_subst_col", None)
                                            if math.isfinite(_nls_err):
                                                _nls = (
                                                    f" nls={_nls_tf}(col {_nls_col})"
                                                    f" err≈{_nls_err:.3g}"
                                                )
                                            _ss = float(
                                                _det.get(
                                                    "structure_screen_score",
                                                    getattr(
                                                        s, "structure_screen_score", 0.0
                                                    ),
                                                )
                                            )
                                            _sp_kind = _det.get("structure_probe_kind", None)
                                            _sp_err = float(
                                                _det.get(
                                                    "structure_probe_err",
                                                    float("inf"),
                                                )
                                            )
                                            if math.isfinite(_ss) and (_ss > 0.0):
                                                _struct = (
                                                    f" screen≈{_ss:.3g}"
                                                    + (
                                                        f" ({_sp_kind} err≈{_sp_err:.3g})"
                                                        if (
                                                            _sp_kind is not None
                                                            and math.isfinite(_sp_err)
                                                        )
                                                        else ""
                                                    )
                                                )
                                        except Exception:
                                            _nls = ""
                                            _struct = ""
                                        print(
                                            f"  {i + 1:2d}) {s.name:10s} score≈{s.score:.3g} "
                                            f"Δ≈{s.score_improvement:.3g} domain={s.domain_ok_frac:.2f} "
                                            f"diag_const≈{s.hess_diag_const_rel_min:.3g} axis={s.hess_diag_const_best_axis}"
                                            f"{_nls}{_struct}"
                                        )

                                # -----------------------------------------------------
                                # (1) Special case: identity found a full-variable compound
                                #     coordinate z(x). Probe outer peels directly in 1D:
                                #         φ(y) ≈ a*z + b
                                # -----------------------------------------------------
                                compound_z_affine_ranked = None
                                _identity_ratpoly_err = float("inf")
                                if compound_is_1d and (stageA_ast is not None):
                                    try:
                                        from nestynet_sr.sr_core import collect_nn_atoms
                                        from nestynet_sr.sr_core.bridges import (
                                            compound_input_expr,
                                            eval_input_expr,
                                            has_nontrivial_input,
                                        )

                                        # Find a compound NN atom that references all variables.
                                        cand_atom = None
                                        for at in collect_nn_atoms(stageA_ast):
                                            if str(getattr(at, "kind", "")).lower() != "nn":
                                                continue
                                            if not has_nontrivial_input(at):
                                                continue
                                            raw = getattr(at, "raw_var_idxs", at.var_idxs)
                                            if set(int(v) for v in raw) >= set(range(Nxvars)):
                                                cand_atom = at
                                                break

                                        if cand_atom is not None:
                                            z_expr = compound_input_expr(cand_atom)

                                            # Collect a modest set of points for the affine probe.
                                            xs, ys = [], []
                                            npts = 0
                                            for batch in datagen_train_noshuffle:
                                                if (
                                                    isinstance(batch, (list, tuple))
                                                    and len(batch) >= 2
                                                ):
                                                    xb, yb = batch[0], batch[1]
                                                else:
                                                    continue
                                                xs.append(xb)
                                                ys.append(yb)
                                                npts += int(xb.shape[0])
                                                if npts >= 2048:
                                                    break

                                            if xs:
                                                Xtmp = torch.cat(xs, dim=0)[:2048].to(device)
                                                Ytmp = torch.cat(ys, dim=0)[:2048].to(device)
                                                Ztmp = eval_input_expr(
                                                    z_expr, Xtmp[:, :Nxvars]
                                                ).view(-1)

                                                peel_names = list(probe_names)
                                                compound_z_affine_ranked, _, _ = (
                                                    _probe_compound_outer_affine_variants(
                                                        y_values=Ytmp.view(-1),
                                                        z_expr=z_expr,
                                                        x_values=Xtmp,
                                                        Nxvars=Nxvars,
                                                        transform_names=peel_names,
                                                        units_payload=units_payload,
                                                        rms_thr=float(
                                                            getattr(
                                                                search_hp,
                                                                "ysearch_outer_affine_confirm_rms_rel",
                                                                1.0e-6,
                                                            )
                                                        ),
                                                        dom_thr=float(
                                                            getattr(
                                                                search_hp,
                                                                "ysearch_outer_affine_min_domain_frac",
                                                                0.995,
                                                            )
                                                        ),
                                                        min_points=int(
                                                            getattr(
                                                                search_hp,
                                                                "ysearch_outer_affine_min_points",
                                                                256,
                                                            )
                                                        ),
                                                        min_domain_frac=0.20,
                                                    )
                                                )

                                                # -- Identity rational-probe gate --
                                                from nestynet_sr.sr_search.fitting_utils import _rational_probe_nd
                                                _rp11 = _rational_probe_nd(
                                                    Ztmp.unsqueeze(-1), Ytmp.view(-1),
                                                    deg_num=1, deg_den=1, min_points=100,
                                                )
                                                _rp22 = _rational_probe_nd(
                                                    Ztmp.unsqueeze(-1), Ytmp.view(-1),
                                                    deg_num=2, deg_den=2, min_points=100,
                                                )
                                                _identity_ratpoly_err = min(
                                                    float(_rp11) if math.isfinite(float(_rp11)) else float("inf"),
                                                    float(_rp22) if math.isfinite(float(_rp22)) else float("inf"),
                                                )

                                                if compound_z_affine_ranked:
                                                    # Persist alongside the general ranking
                                                    if outer_peel_ranked is None:
                                                        outer_peel_ranked = {}
                                                    outer_peel_ranked[
                                                        "compound_z_affine"
                                                    ] = compound_z_affine_ranked[:8]

                                                    _print_compound_outer_affine_entries(
                                                        "[Stage A] Outer-peel 1D compound-z affine tests "
                                                        "(φ(y) ≈ a*z + b, z variants: z, 1/z):",
                                                        compound_z_affine_ranked,
                                                        domain_digits=2,
                                                    )
                                    except Exception as e:
                                        print(
                                            f"[Stage A] Outer-peel compound-z affine test skipped: {e}"
                                        )

                                # Outer-peel is diagnostics-only in unified y-search mode.
                                # Selection/commit of y-transforms is owned by Stage A
                                # y-search; this block intentionally performs no training
                                # and no transform adoption.
                                if bool(getattr(search_hp, "outer_peel_autorun", True)):
                                    print(
                                        "[Stage A] Outer-peel autorun is disabled in unified "
                                        "y-search mode; keeping ranked proposals as diagnostics only."
                                    )
                        except Exception as e:
                            print(f"[Stage A] Outer-peel proposal step failed: {e}")
    elif loaded_checkpoint is not None:
        # ---------------------------------------------------------------
        # Stage A already done: restore state from checkpoint
        # ---------------------------------------------------------------
        separability_success = loaded_checkpoint.get("separability_success", False)
        stageA_status = str(
            loaded_checkpoint.get(
                "stageA_status",
                "split_confirmed" if separability_success else "unresolved",
            )
            or ("split_confirmed" if separability_success else "unresolved")
        )
        final_ast = loaded_checkpoint.get("ast", None)
        rest_add_final = loaded_checkpoint.get("rest_add", None)
        rest_mult_final = loaded_checkpoint.get("rest_mult", None)
        final_y_op_name = loaded_checkpoint.get("y_op_name", None)
        stageA_val_loss = loaded_checkpoint.get(
            "val_loss", None
        )  # May be None for older checkpoints
        stageA_val_losses = loaded_checkpoint.get("val_losses", None)
        stageA_val_loss_agg_mode = loaded_checkpoint.get("val_loss_agg_mode", None)
        stageA_val_loss_agg_weights = loaded_checkpoint.get("val_loss_agg_weights", None)
        stageA_dataset_ids = loaded_checkpoint.get("dataset_ids", None)
        fit_y_link_used = loaded_checkpoint.get(
            "fit_y_link", None
        )  # May be None for older checkpoints
        fit_y_link_scale_used = loaded_checkpoint.get(
            "fit_y_link_scale", 1.0
        )  # Scale for asinh fit-link
        fit_link_branch_certificate = loaded_checkpoint.get("fit_link_branch_certificate", None)
        fit_link_branch_status = loaded_checkpoint.get("fit_link_branch_status", None)
        fit_link_original_y_certified = loaded_checkpoint.get(
            "fit_link_original_y_certified", None
        )
        fit_link_original_y_val_loss = loaded_checkpoint.get(
            "fit_link_original_y_val_loss", None
        )
        fit_link_original_y_allowed_loss = loaded_checkpoint.get(
            "fit_link_original_y_allowed_loss", None
        )
        coe_stageA_ybranch_committee = loaded_checkpoint.get(
            "coe_stageA_ybranch_committee", coe_stageA_ybranch_committee
        )
        coe_stageA_compound_shortlist = loaded_checkpoint.get(
            "coe_stageA_compound_shortlist", coe_stageA_compound_shortlist
        )
        lm_hp.coe_stageA_fit_tournament_records = list(
            loaded_checkpoint.get("coe_stageA_fit_tournament_records", []) or []
        )
        _stageA_ledgers_from_model_or_checkpoint(checkpoint=loaded_checkpoint)

        # Proposal-only diagnostics persisted from Stage A (may be absent for older checkpoints)
        outer_peel_square_decision = loaded_checkpoint.get("outer_peel_square", None)
        outer_peel_ranked = loaded_checkpoint.get("outer_peel_ranked", None)
        stageB_virtual_top_names = list(
            loaded_checkpoint.get("stageB_virtual_top_names", []) or []
        )
        stageB_virtual_portfolio = list(
            loaded_checkpoint.get("stageB_virtual_portfolio", []) or []
        )
        stageB_y_shortlist_sources = dict(
            loaded_checkpoint.get("stageB_y_shortlist_sources", {}) or {}
        )

        # x_transform_map for compound variable display (e.g., cos(x6) instead of x6)
        stageA_x_transform = loaded_checkpoint.get("x_transform", {})

        if stageA_val_loss is not None:
            print(f"[Stage A] Loaded checkpoint with val_loss: {stageA_val_loss:.4e}")
        if fit_y_link_used is not None and fit_link_original_y_certified is False:
            print(
                "[Stage A] Loaded fit-link checkpoint is transformed-space accepted "
                "but not original-y certified."
            )

        # Map y_op_name -> y_op, y_op_inv
        if final_y_op_name is None:
            final_y_op = None
            final_y_op_inv = None
        else:
            try:
                final_y_op, final_y_op_inv, final_y_op_name = resolve_y_transform_name(
                    final_y_op_name, transforms=y_transforms
                )
            except Exception as e:
                raise ValueError(
                    f"Checkpoint refers to y-transform '{final_y_op_name}', "
                    "which is not present in the current y_transform_names."
                ) from e

        # Detect pure 1D full-variable compound from loaded AST
        if final_ast is not None and (final_y_op_name is None or final_y_op_name == "identity"):
            compound_is_1d = _compute_compound_is_1d(final_ast, Nxvars)
            # Keep legacy checkpoint flag only when the loaded AST matches the
            # stricter pure f(z) semantics.
            ckpt_flag = loaded_checkpoint.get("full_compound_solved", False)
            if ckpt_flag and (not compound_is_1d):
                print(
                    "[Stage A] Checkpoint full_compound_solved flag ignored: "
                    "loaded AST is not pure 1D f(z)"
                )
            if compound_is_1d:
                full_compound_compressed = True
                stageA_full_compound_solved = False
                if stageA_status == "unresolved":
                    stageA_status = "compound_unresolved"
                print("[Stage A] Detected full-variable compound compression from checkpoint; outer map remains unresolved")

        # Rebuild a Stage-A model from disk for Stage B to use as teacher.
        # Only use model_sep_output for compound when the final y-transform is
        # identity, to avoid loading an identity-space teacher when Stage B
        # operates in a different φ(y)-space.
        _final_y_name = final_y_op_name or "identity"
        stageA_mod_path = None
        if separability_success and os.path.exists(model_sep_output):
            stageA_mod_path = model_sep_output
        elif (_final_y_name == "identity") and compound_is_1d and os.path.exists(model_sep_output):
            stageA_mod_path = model_sep_output
        elif os.path.exists(model_output):
            stageA_mod_path = model_output
        elif os.path.exists(model_sep_output):
            stageA_mod_path = model_sep_output

        if stageA_mod_path is not None:
            try:
                mod_ckpt = torch.load(stageA_mod_path, map_location=device, weights_only=False)
                ast_mod = mod_ckpt.get("ast", final_ast)
                if ast_mod is None:
                    raise ValueError(
                        f"Model checkpoint {stageA_mod_path} does not contain 'ast' key. "
                        "Legacy 'expressions' format is no longer supported."
                    )

                # Sync AST num_segments from state_dict (fixes common mismatch issue)
                from nestynet_sr.sr_core.bridges import sync_ast_num_segments_from_state_dict

                state_dict = mod_ckpt["model_state_dict"]
                sync_ast_num_segments_from_state_dict(ast_mod, state_dict)

                # Build model with corrected AST (pass None to preserve per-atom kwargs)
                stageA_model, _, _ = build_composite_ast(
                    ast_mod,
                    num_segments=None,  # Preserve per-atom num_segments from AST
                    dual_layer=None,  # Preserve per-atom dual_layer from AST
                    leaf_builder=leaf_builder,
                    device=device,
                    dtype=dtype,
                )
                stageA_model.load_state_dict(state_dict)
                # Preserve any x-preprocessing metadata (used to decide whether Stage-A weights are reusable)
                try:
                    setattr(stageA_model, '_x_transform', mod_ckpt.get('x_transform', {}))
                except Exception:
                    pass
                try:
                    _cert = mod_ckpt.get("fit_link_branch_certificate", None)
                    if _cert is not None:
                        setattr(stageA_model, "_stageA_fit_link_certificate", dict(_cert))
                    if "fit_link_original_y_certified" in mod_ckpt:
                        setattr(
                            stageA_model,
                            "_stageA_original_y_certified",
                            bool(mod_ckpt.get("fit_link_original_y_certified")),
                        )
                    if "fit_link_branch_status" in mod_ckpt:
                        setattr(
                            stageA_model,
                            "_stageA_fit_link_branch_status",
                            str(mod_ckpt.get("fit_link_branch_status")),
                        )
                    if "fit_link_original_y_val_loss" in mod_ckpt:
                        setattr(
                            stageA_model,
                            "_stageA_original_y_val_loss",
                            mod_ckpt.get("fit_link_original_y_val_loss"),
                        )
                    if "fit_link_original_y_allowed_loss" in mod_ckpt:
                        setattr(
                            stageA_model,
                            "_stageA_original_y_allowed_loss",
                            mod_ckpt.get("fit_link_original_y_allowed_loss"),
                        )
                except Exception:
                    pass
                final_model = stageA_model
            except Exception as e:
                print(f"Warning: failed to rebuild Stage-A model from '{stageA_mod_path}': {e}")
        else:
            print(
                "Warning: no Stage-A model file found on disk; Stage B may have to rebuild from AST."
            )

    print()
    _stageA_msg = _stageA_status_message(stageA_status, separability_success)
    if _stageA_msg:
        print(f"{CYAN}{_stageA_msg}{RESET}")

    # IMPORTANT:
    # The chosen y-transform (φ) is independent of whether Stage A found a separability split.
    # We must preserve (y_op, y_op_inv, y_op_name) so Stage B runs in the correct φ(y)-space
    if final_y_op_name is None:
        final_y_op_name = "identity"
    if final_y_op_inv is None:
        final_y_op, final_y_op_inv, final_y_op_name = resolve_y_transform_name(
            final_y_op_name, transforms=y_transforms
        )

    y_op_inv_str = (
        final_y_op_inv.__name__ if hasattr(final_y_op_inv, "__name__") else str(final_y_op_inv)
    )

    # Get x_transform_map from final_model if available, fallback to checkpoint
    final_x_transform_map = getattr(final_model, "_x_transform", None) if final_model else None
    if final_x_transform_map is None and loaded_checkpoint is not None:
        final_x_transform_map = stageA_x_transform if stageA_x_transform else None

    if final_ast is None:
        # Fallback: use initial AST if we never managed to build a model
        final_ast = initial_ast
        expression_human = ast_to_human_readable(final_ast, final_x_transform_map)
    else:
        expression_human = ast_to_human_readable(final_ast, final_x_transform_map)

    rest_add_final = sorted(set(rest_add_final)) if rest_add_final is not None else None
    rest_mult_final = sorted(set(rest_mult_final)) if rest_mult_final is not None else None

    # ---------------------------------------------------------------
    # Save / update checkpoint after Stage A (fresh runs only)
    # ---------------------------------------------------------------
    if loaded_checkpoint is None and final_ast is not None:
        # Check if more NN atoms remain (for outer loop iteration control)
        more_nns_remain = has_nn_atoms(final_ast)

        # Try to load val_loss and fit_y_link from the saved model files (saved by search.py)
        # Check model_sep_output first (used when separability found), then model_output (baseline fit)
        stageA_val_loss = None
        fit_y_link_used = None
        fit_y_link_scale_used = 1.0
        for model_file in [model_sep_output, model_output]:
            try:
                if os.path.exists(model_file):
                    saved_model = torch.load(model_file, weights_only=False)
                    val_loss_candidate = saved_model.get("val_loss", None)
                    if val_loss_candidate is not None:
                        stageA_val_loss = val_loss_candidate
                        stageA_val_losses = saved_model.get("val_losses", stageA_val_losses)
                        stageA_val_loss_agg_mode = saved_model.get("val_loss_agg_mode", stageA_val_loss_agg_mode)
                        stageA_val_loss_agg_weights = saved_model.get(
                            "val_loss_agg_weights", stageA_val_loss_agg_weights
                        )
                        stageA_dataset_ids = saved_model.get("dataset_ids", stageA_dataset_ids)
                        fit_y_link_used = saved_model.get("fit_y_link", None)
                        fit_y_link_scale_used = saved_model.get("fit_y_link_scale", 1.0)
                        fit_link_branch_certificate = saved_model.get(
                            "fit_link_branch_certificate", fit_link_branch_certificate
                        )
                        fit_link_branch_status = saved_model.get(
                            "fit_link_branch_status", fit_link_branch_status
                        )
                        fit_link_original_y_certified = saved_model.get(
                            "fit_link_original_y_certified", fit_link_original_y_certified
                        )
                        fit_link_original_y_val_loss = saved_model.get(
                            "fit_link_original_y_val_loss", fit_link_original_y_val_loss
                        )
                        fit_link_original_y_allowed_loss = saved_model.get(
                            "fit_link_original_y_allowed_loss", fit_link_original_y_allowed_loss
                        )
                        coe_stageA_compound_shortlist = saved_model.get(
                            "coe_stageA_compound_shortlist", coe_stageA_compound_shortlist
                        )
                        break  # Use the first valid val_loss found
            except Exception:
                pass  # Older checkpoints may not have val_loss

        if final_model is not None:
            stageA_val_loss = getattr(final_model, "_stageA_val_loss_agg", stageA_val_loss)
            stageA_val_losses = getattr(final_model, "_stageA_val_losses", stageA_val_losses)
            stageA_val_loss_agg_mode = getattr(final_model, "_stageA_agg_mode", stageA_val_loss_agg_mode)
            stageA_val_loss_agg_weights = getattr(
                final_model, "_stageA_agg_weights", stageA_val_loss_agg_weights
            )
            stageA_dataset_ids = getattr(final_model, "_stageA_dataset_ids", stageA_dataset_ids)
            fit_link_branch_certificate = getattr(
                final_model,
                "_stageA_fit_link_certificate",
                fit_link_branch_certificate,
            )
            fit_link_branch_status = getattr(
                final_model,
                "_stageA_fit_link_branch_status",
                fit_link_branch_status,
            )
            fit_link_original_y_certified = getattr(
                final_model,
                "_stageA_original_y_certified",
                fit_link_original_y_certified,
            )
            fit_link_original_y_val_loss = getattr(
                final_model,
                "_stageA_original_y_val_loss",
                fit_link_original_y_val_loss,
            )
            fit_link_original_y_allowed_loss = getattr(
                final_model,
                "_stageA_original_y_allowed_loss",
                fit_link_original_y_allowed_loss,
            )
            coe_stageA_compound_shortlist = getattr(
                final_model,
                "_stageA_coe_compound_shortlist",
                coe_stageA_compound_shortlist,
            )
            stageA_deferred_fitlink_branches = list(
                getattr(final_model, "_stageA_deferred_fitlink_branches", []) or []
            )
            if stageA_deferred_fitlink_branches:
                labels = ", ".join(
                    str(b.get("name", "fitlink_branch"))
                    for b in stageA_deferred_fitlink_branches
                    if isinstance(b, dict)
                )
                print(
                    "[Stage A] Deferred fit-link branch(es) retained for rescue/adjudication: "
                    f"{labels or len(stageA_deferred_fitlink_branches)}"
                )

        # A pure NN[z(x)] is only a compression; it is not a solved outer relation.
        full_compound_compressed = bool(compound_is_1d)
        stageA_full_compound_solved = False
        if full_compound_compressed and stageA_status == "unresolved":
            stageA_status = "compound_unresolved"

        _checkpoint_stageA_ledgers = _stageA_ledgers_from_model_or_checkpoint(
            model=final_model
        )

        ckpt = {
            "version": 1,
            "phase": "after_stageA",
            "ab_iter": 0,
            "filepath": filepath,
            "filepaths": list(filepaths),
            "statistical_selection_split": (
                statistical_split_plan.checkpoint_contract()
                if statistical_split_plan is not None
                else None
            ),
            "Nxvars": Nxvars,
            "y_op_name": final_y_op_name,
            "separability_success": separability_success,
            "stageA_status": stageA_status,
            "rest_add": rest_add_final,
            "rest_mult": rest_mult_final,
            "ast": final_ast,
            "val_loss": stageA_val_loss,  # Track fit quality for diagnostics
            "val_losses": stageA_val_losses,
            "val_loss_agg_mode": stageA_val_loss_agg_mode,
            "val_loss_agg_weights": stageA_val_loss_agg_weights,
            "dataset_ids": stageA_dataset_ids,
            "fit_y_link": fit_y_link_used,  # Track if asinh fit-link was used
            "fit_y_link_scale": fit_y_link_scale_used,  # Scale factor for asinh fit-link
            "fit_link_branch_certificate": (
                dict(fit_link_branch_certificate)
                if isinstance(fit_link_branch_certificate, dict)
                else fit_link_branch_certificate
            ),
            "fit_link_branch_status": fit_link_branch_status,
            "fit_link_original_y_certified": fit_link_original_y_certified,
            "fit_link_original_y_val_loss": fit_link_original_y_val_loss,
            "fit_link_original_y_allowed_loss": fit_link_original_y_allowed_loss,
            "stageA_deferred_fitlink_branches": list(stageA_deferred_fitlink_branches or []),
            "coe_stageA_ybranch_committee": _make_json_serializable(coe_stageA_ybranch_committee),
            "coe_stageA_compound_shortlist": _make_json_serializable(coe_stageA_compound_shortlist),
            "coe_stageA_fit_tournament_records": _make_json_serializable(
                getattr(lm_hp, "coe_stageA_fit_tournament_records", [])
            ),
            **_make_json_serializable(_checkpoint_stageA_ledgers),
            "has_remaining_nns": more_nns_remain,  # NEW: Track if more NNs remain
            "full_compound_solved": stageA_full_compound_solved,
            "full_compound_compressed": bool(full_compound_compressed),
            # Proposal-only diagnostics (useful for Stage B fallbacks / resume)
            "outer_peel_square": outer_peel_square_decision,
            "outer_peel_ranked": outer_peel_ranked,
            "stageB_virtual_top_names": list(stageB_virtual_top_names or []),
            "stageB_virtual_portfolio": _make_json_serializable(
                stageB_virtual_portfolio
            ),
            "stageB_y_shortlist_sources": dict(stageB_y_shortlist_sources or {}),
            # x_transform_map for compound variable display (e.g., cos(x6) instead of x6)
            "x_transform": final_x_transform_map if final_x_transform_map else {},
        }
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(ckpt, f)
            print(f"Saved Stage-A checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Warning: failed to save Stage-A checkpoint to {checkpoint_path}: {e}")
    else:
        # Resumed run: build ckpt dict from loaded_checkpoint so mid-loop saves work
        ckpt = dict(loaded_checkpoint) if loaded_checkpoint is not None else {}
        stageA_deferred_fitlink_branches = list(
            ckpt.get("stageA_deferred_fitlink_branches", []) or []
        )

    fit_link_suffix = " [loss: asinh]" if fit_y_link_used == "asinh" else ""
    print(
        f"{CYAN}Final model: {y_op_inv_str} ({expression_human}), rest: {rest_add_final}, {rest_mult_final}{fit_link_suffix}{RESET}"
    )

    # Create output paths
    pkl_output = os.path.join(results_dir, f"{base_filename}.pkl")
    human_output = os.path.join(results_dir, f"{base_filename}.human")

    # Save pickle file with AST
    with open(pkl_output, "wb") as f:
        pickle.dump(
            {
                "y_op_inv": final_y_op_inv,
                "ast": final_ast,
                "rest_add": rest_add_final,
                "rest_mult": rest_mult_final,
            },
            f,
        )

    # Save human-readable expression
    with open(human_output, "w") as f:
        f.write(
            "{} ({}), rest: {}, {}{}".format(
                y_op_inv_str, expression_human, rest_add_final, rest_mult_final, fit_link_suffix
            )
        )

    # Stage-B refinement layer
    # -----------------------------------------------------------------
    # Stage B works in the same φ(y)-space selected by Stage A/y-search.
    stageB_state = None  # initialised here so class SR guard sees it even if Stage B is skipped
    _resume_ab_iter = 0  # which A<->B iteration we are resuming from
    _resume_phase = "after_stageA"
    if loaded_checkpoint is not None:
        _resume_ab_iter = loaded_checkpoint.get("ab_iter", 0)
        _resume_phase = loaded_checkpoint.get("phase", "after_stageA")

    if args.stageB and final_model is not None and final_ast is not None:
        from nestynet_sr.sr_core import ast_from_composite
        from nestynet_sr.sr_core.bridges import sync_ast_num_segments_from_state_dict
        from nestynet_sr.sr_search.stageB import run_stageB_from_model
        from nestynet_sr.sr_search.stageB.atom_mapping import _refresh_reuse_from_state
        from nestynet_sr.sr_search.stageB.engine import StageBState

        def _resolve_stageB_y_transform(name):
            nm = "identity" if name in {None, "", "None", "none", "null"} else str(name)
            if nm == "identity":
                return None, None, "identity"
            return resolve_y_transform_name(nm, transforms=y_transforms)

        def _stageB_model_checkpoint_path() -> str:
            return os.path.join(results_dir, f"{model_base_filename}_stageB_model.pt")

        def _stageB_model_checkpoint_candidates(_checkpoint: dict) -> list[str]:
            names: list[str] = []
            raw_values = [
                _checkpoint.get("stageB_model_path"),
                _checkpoint.get("stageB_model_file"),
                f"{model_base_filename}_stageB_model.pt",
            ]
            resume_dir = None
            try:
                if args.resume_from is not None:
                    resume_dir = os.path.dirname(os.path.abspath(str(args.resume_from)))
            except Exception:
                resume_dir = None
            for raw in raw_values:
                if not raw:
                    continue
                p = str(raw)
                if os.path.isabs(p):
                    names.append(p)
                else:
                    names.append(os.path.join(results_dir, p))
                    if resume_dir:
                        names.append(os.path.join(resume_dir, p))
            out: list[str] = []
            seen: set[str] = set()
            for p in names:
                ap = os.path.abspath(p)
                if ap not in seen:
                    seen.add(ap)
                    out.append(p)
            return out

        def _stageB_state_from_checkpoint(_checkpoint: dict):
            model_paths = _stageB_model_checkpoint_candidates(_checkpoint)
            model_path = next((p for p in model_paths if os.path.exists(p)), None)
            if model_path is None:
                tried = ", ".join(model_paths) if model_paths else _stageB_model_checkpoint_path()
                print(f"[Stage B] Stage B model file not found; tried: {tried}")
                return None
            try:
                _sb_mod_ckpt = torch.load(model_path, map_location=device, weights_only=False)
                _sb_ast = _sb_mod_ckpt.get("ast", _checkpoint.get("stageB_ast"))
                if _sb_ast is None:
                    raise ValueError("Stage-B model checkpoint does not contain an AST")
                _sb_sd = _sb_mod_ckpt["model_state_dict"]
                sync_ast_num_segments_from_state_dict(_sb_ast, _sb_sd)
                _sb_model, _, _ = build_composite_ast(
                    _sb_ast,
                    num_segments=None,
                    dual_layer=None,
                    leaf_builder=leaf_builder,
                    device=device,
                    dtype=dtype,
                )
                _sb_model.load_state_dict(_sb_sd)
                _sb_reuse = _refresh_reuse_from_state(_sb_ast, _sb_model)

                _sb_models = None
                _sb_reuses = None
                _sb_model_sds = _sb_mod_ckpt.get("model_state_dicts", None)
                if isinstance(_sb_model_sds, list) and _sb_model_sds:
                    _sb_models = []
                    _sb_reuses = []
                    for _sd in _sb_model_sds:
                        _ast_i = copy.deepcopy(_sb_ast)
                        sync_ast_num_segments_from_state_dict(_ast_i, _sd)
                        _model_i, _, _ = build_composite_ast(
                            _ast_i,
                            num_segments=None,
                            dual_layer=None,
                            leaf_builder=leaf_builder,
                            device=device,
                            dtype=dtype,
                        )
                        _model_i.load_state_dict(_sd)
                        _sb_models.append(_model_i)
                        _sb_reuses.append(_refresh_reuse_from_state(_ast_i, _model_i))

                _state = StageBState(
                    root=_sb_ast,
                    model=_sb_model,
                    reuse=_sb_reuse,
                    val_loss=_checkpoint.get("stageB_val_loss", float("inf")),
                    models=_sb_models,
                    reuses=_sb_reuses,
                    val_losses=_checkpoint.get("stageB_val_losses", None),
                    dataset_ids=_checkpoint.get("stageB_dataset_ids", None),
                    agg_mode=_checkpoint.get("stageB_agg_mode", None),
                    agg_weights=_checkpoint.get("stageB_agg_weights", None),
                    phi_expr_str=_checkpoint.get("stageB_phi_expr_str", None),
                    y_expr_str=_checkpoint.get("stageB_y_expr_str", None),
                    sympy_meta=_checkpoint.get("stageB_sympy_meta", None),
                    coefficient_metadata=_checkpoint.get(
                        "stageB_coefficient_metadata",
                        _sb_mod_ckpt.get("coefficient_metadata", None),
                    ),
                    coefficient_metadata_by_dataset=_checkpoint.get(
                        "stageB_coefficient_metadata_by_dataset",
                        _sb_mod_ckpt.get("coefficient_metadata_by_dataset", None),
                    ),
                    enabled_patterns=_checkpoint.get("stageB_enabled_patterns", None),
                    num_nn_atoms=_checkpoint.get("stageB_num_nn_atoms", None),
                    num_multivar_nn_atoms=_checkpoint.get(
                        "stageB_num_multivar_nn_atoms", None
                    ),
                    max_nn_arity=_checkpoint.get("stageB_max_nn_arity", None),
                    loss_scale=_checkpoint.get("stageB_loss_scale", None),
                    loss_good_enough_eff=_checkpoint.get(
                        "stageB_loss_good_enough_eff", None
                    ),
                    loss_acceptable_eff=_checkpoint.get(
                        "stageB_loss_acceptable_eff", None
                    ),
                    acceptance_noise_floor_raw=_checkpoint.get(
                        "stageB_acceptance_noise_floor_raw", None
                    ),
                    acceptance_noise_n_eff=_checkpoint.get(
                        "stageB_acceptance_noise_n_eff", None
                    ),
                    original_y_val_loss=_checkpoint.get(
                        "stageB_original_y_val_loss", None
                    ),
                    original_y_loss_good_enough_eff=_checkpoint.get(
                        "stageB_original_y_loss_good_enough_eff", None
                    ),
                    original_y_loss_acceptable_eff=_checkpoint.get(
                        "stageB_original_y_loss_acceptable_eff", None
                    ),
                    coe_stageB_dry_run_log=_checkpoint.get(
                        "stageB_coe_stageB_dry_run_log", None
                    ),
                    coe_stageB_gate_log=_checkpoint.get(
                        "stageB_coe_stageB_gate_log", None
                    ),
                    decision_log=_checkpoint.get("stageB_decision_log", None),
                    x_transform_map=_checkpoint.get("stageB_x_transform_map", None),
                    phi_expr_raw_str=_checkpoint.get("stageB_phi_expr_raw_str", None),
                    y_expr_raw_str=_checkpoint.get("stageB_y_expr_raw_str", None),
                    complexity_mapping_cost=float(
                        _checkpoint.get("stageB_complexity_mapping_cost", 0.0) or 0.0
                    ),
                    simplification_path=copy.deepcopy(
                        _checkpoint.get("stageB_simplification_path", []) or []
                    ),
                )
                setattr(
                    _state,
                    "generic_approximant_unpromoted",
                    bool(_checkpoint.get("stageB_generic_approximant_unpromoted", False)),
                )
                _restored_y = _checkpoint.get(
                    "stageB_y_op_name",
                    _checkpoint.get("y_selected", final_y_op_name or "identity"),
                )
                setattr(_state, "_stageB_portfolio_y_name", _restored_y)
                print(f"[Stage B] Restored Stage-B model checkpoint from {model_path}")
                return _state
            except Exception as _e:
                print(
                    "[Stage B] Warning: failed to reconstruct Stage-B state from "
                    f"checkpoint ({_e})."
                )
                return None

        def _stageB_checkpoint_shortlist_names() -> list:
            try:
                return list(stageB_portfolio_names or [])
            except Exception:
                return []

        def _stageB_checkpoint_branch_artifacts() -> list:
            try:
                return list(stageB_y_branch_artifacts or [])
            except Exception:
                return []

        def _save_stageB_resume_checkpoint(*, ab_iter: int, prev_expr_signature, log_prefix: str) -> None:
            nonlocal ckpt
            if stageB_state is None:
                return
            _stageB_model_path = _stageB_model_checkpoint_path()
            _model_payload = {
                "ast": stageB_state.root,
                "model_state_dict": stageB_state.model.state_dict(),
                "stageB_y_op_name": stageB_y_op_name,
                "fit_y_link": getattr(lm_hp, "fit_y_link", None),
                "fit_y_link_scale": getattr(lm_hp, "fit_y_link_scale", 1.0),
                "coefficient_metadata": getattr(
                    stageB_state, "coefficient_metadata", None
                ),
                "coefficient_metadata_by_dataset": getattr(
                    stageB_state, "coefficient_metadata_by_dataset", None
                ),
            }
            _stageB_models = getattr(stageB_state, "models", None)
            if isinstance(_stageB_models, list) and _stageB_models:
                _model_payload["model_state_dicts"] = [
                    _m.state_dict() for _m in _stageB_models
                ]
            torch.save(_model_payload, _stageB_model_path)
            if ckpt is None:
                ckpt = {
                    "version": 1,
                    "filepath": filepath,
                    "filepaths": list(filepaths),
                    "statistical_selection_split": (
                        statistical_split_plan.checkpoint_contract()
                        if statistical_split_plan is not None
                        else None
                    ),
                    "Nxvars": Nxvars,
                    "y_op_name": final_y_op_name,
                    "separability_success": separability_success,
                    "stageA_status": stageA_status,
                    "rest_add": rest_add_final,
                    "rest_mult": rest_mult_final,
                    "ast": final_ast,
                }
            ckpt.update(
                {
                    "phase": "after_stageB",
                    "ab_iter": int(ab_iter),
                    "statistical_selection_split": (
                        statistical_split_plan.checkpoint_contract()
                        if statistical_split_plan is not None
                        else None
                    ),
                    "stageB_model_file": os.path.basename(_stageB_model_path),
                    "stageB_ast": stageB_state.root,
                    "stageB_val_loss": stageB_state.val_loss,
                    "stageB_val_losses": getattr(stageB_state, "val_losses", None),
                    "stageB_dataset_ids": getattr(stageB_state, "dataset_ids", None),
                    "stageB_agg_mode": getattr(stageB_state, "agg_mode", None),
                    "stageB_agg_weights": getattr(stageB_state, "agg_weights", None),
                    "stageB_phi_expr_str": stageB_state.phi_expr_str,
                    "stageB_y_expr_str": stageB_state.y_expr_str,
                    "stageB_phi_expr_raw_str": getattr(stageB_state, "phi_expr_raw_str", None),
                    "stageB_y_expr_raw_str": getattr(stageB_state, "y_expr_raw_str", None),
                    "stageB_sympy_meta": getattr(stageB_state, "sympy_meta", None),
                    "stageB_coefficient_metadata": getattr(
                        stageB_state, "coefficient_metadata", None
                    ),
                    "stageB_coefficient_metadata_by_dataset": getattr(
                        stageB_state, "coefficient_metadata_by_dataset", None
                    ),
                    "stageB_enabled_patterns": getattr(stageB_state, "enabled_patterns", None),
                    "stageB_num_nn_atoms": getattr(stageB_state, "num_nn_atoms", None),
                    "stageB_num_multivar_nn_atoms": getattr(
                        stageB_state, "num_multivar_nn_atoms", None
                    ),
                    "stageB_max_nn_arity": getattr(stageB_state, "max_nn_arity", None),
                    "stageB_loss_scale": getattr(stageB_state, "loss_scale", None),
                    "stageB_loss_good_enough_eff": getattr(
                        stageB_state, "loss_good_enough_eff", None
                    ),
                    "stageB_loss_acceptable_eff": getattr(
                        stageB_state, "loss_acceptable_eff", None
                    ),
                    "stageB_acceptance_noise_floor_raw": getattr(
                        stageB_state, "acceptance_noise_floor_raw", None
                    ),
                    "stageB_acceptance_noise_n_eff": getattr(
                        stageB_state, "acceptance_noise_n_eff", None
                    ),
                    "stageB_original_y_val_loss": getattr(
                        stageB_state, "original_y_val_loss", None
                    ),
                    "stageB_original_y_loss_good_enough_eff": getattr(
                        stageB_state, "original_y_loss_good_enough_eff", None
                    ),
                    "stageB_original_y_loss_acceptable_eff": getattr(
                        stageB_state, "original_y_loss_acceptable_eff", None
                    ),
                    "stageB_coe_stageB_dry_run_log": getattr(
                        stageB_state, "coe_stageB_dry_run_log", None
                    ),
                    "stageB_coe_stageB_gate_log": getattr(
                        stageB_state, "coe_stageB_gate_log", None
                    ),
                    "stageB_decision_log": getattr(stageB_state, "decision_log", None),
                    "stageB_x_transform_map": getattr(stageB_state, "x_transform_map", None),
                    "stageB_complexity_mapping_cost": getattr(
                        stageB_state, "complexity_mapping_cost", 0.0
                    ),
                    "stageB_simplification_path": copy.deepcopy(
                        getattr(stageB_state, "simplification_path", []) or []
                    ),
                    "stageB_generic_approximant_unpromoted": bool(
                        getattr(stageB_state, "generic_approximant_unpromoted", False)
                    ),
                    "stageB_y_op_name": stageB_y_op_name,
                    "stageB_y_shortlist_names": _stageB_checkpoint_shortlist_names(),
                    "stageB_y_shortlist_sources": dict(stageB_y_shortlist_sources or {}),
                    "stageB_y_branch_artifacts": _make_json_serializable(
                        _stageB_checkpoint_branch_artifacts()
                    ),
                    "prev_expr_signature": prev_expr_signature,
                }
            )
            with open(checkpoint_path, "wb") as _f:
                pickle.dump(ckpt, _f)
            print(f"{log_prefix} Saved after-stageB checkpoint to {checkpoint_path}")

        _stageB_resume_restored = False
        if loaded_checkpoint is not None and _resume_phase == "after_stageB":
            _restored_state = _stageB_state_from_checkpoint(loaded_checkpoint)
            if _restored_state is not None:
                stageB_state = _restored_state
                stageB_y_op_name = loaded_checkpoint.get(
                    "stageB_y_op_name", final_y_op_name or "identity"
                )
                stageB_y_op, stageB_y_op_inv, stageB_y_op_name = _resolve_stageB_y_transform(
                    stageB_y_op_name
                )
                stageB_portfolio_names = list(
                    loaded_checkpoint.get("stageB_y_shortlist_names", []) or [stageB_y_op_name]
                )
                stageB_y_shortlist_sources = dict(
                    loaded_checkpoint.get("stageB_y_shortlist_sources", {}) or {}
                )
                stageB_y_branch_artifacts = list(
                    loaded_checkpoint.get("stageB_y_branch_artifacts", []) or []
                )
                prev_expr_signature = loaded_checkpoint.get("prev_expr_signature")
                _stageB_resume_restored = True

        # If resuming a completed single-pass Stage B checkpoint, keep the
        # restored Stage-B state and continue into reporting/final selection.
        if (
            loaded_checkpoint is not None
            and _resume_phase == "after_stageB"
            and _resume_ab_iter == 0
            and _stageB_resume_restored
        ):
            print("Checkpoint indicates Stage B has already completed; restored Stage-B state.")
        else:

            # -----------------------------------------------------------
            # Restore asinh fit-link settings from Stage A if present
            # -----------------------------------------------------------
            if fit_y_link_used is not None:
                lm_hp.fit_y_link = fit_y_link_used
                # Fallback: get scale from model file if not in checkpoint (for old checkpoints)
                if fit_y_link_scale_used == 1.0 and fit_y_link_used == "asinh":
                    for model_file in [model_sep_output, model_output]:
                        try:
                            if os.path.exists(model_file):
                                saved_model = torch.load(model_file, weights_only=False)
                                scale_from_model = saved_model.get("fit_y_link_scale", None)
                                if scale_from_model is not None and scale_from_model != 1.0:
                                    fit_y_link_scale_used = scale_from_model
                                    print(f"[Stage B] Retrieved fit_y_link_scale={scale_from_model:.4g} from model file")
                                    break
                        except Exception:
                            pass
                lm_hp.fit_y_link_scale = fit_y_link_scale_used
                print(f"[Stage B] Restoring fit-link: {fit_y_link_used} (scale={fit_y_link_scale_used:.4g})")
                if fit_link_original_y_certified is False:
                    print(
                        "[Stage B] Stage-A fit-link branch is a transformed-space search scaffold; "
                        "original-y certification is still pending."
                    )

            # -----------------------------------------------------------
            # Stage B y-transform selection
            # -----------------------------------------------------------
            # Stage A may only have proxy evidence for a y-transform.  Keep a
            # small Stage-B adjudication portfolio so proxy-ranked branches can
            # be confirmed by actual rewrites, while identity remains an
            # explicit baseline.
            stageB_virtual_reserved_names = [
                str(row.get("name"))
                for row in stageB_virtual_portfolio
                if isinstance(row, dict)
                and row.get("selection_reason") == "joint_homogeneity_reserve"
                and row.get("name")
            ][:1]
            stageB_portfolio_names = _stageB_shortlist_names(
                final_y_op_name=final_y_op_name or "identity",
                outer_peel_ranked=outer_peel_ranked,
                available_y_names=[t.name for t in y_transforms],
                virtual_top_names=stageB_virtual_top_names,
                virtual_reserved_names=stageB_virtual_reserved_names,
                top_k=3,
                include_identity=True,
            )
            stageB_y_shortlist_sources = _stageB_shortlist_source_map(
                names=stageB_portfolio_names,
                final_y_op_name=final_y_op_name or "identity",
                outer_peel_ranked=outer_peel_ranked,
                virtual_top_names=stageB_virtual_top_names,
                virtual_reserved_names=stageB_virtual_reserved_names,
            )
            if stageB_portfolio_names:
                print("[Stage B] y-transform adjudication shortlist:")
                for _nm in stageB_portfolio_names:
                    _src = ",".join(stageB_y_shortlist_sources.get(_nm, [])) or "unknown"
                    print(f"  - {_nm:12s} source={_src}")
                if ckpt is None:
                    ckpt = {}
                ckpt.update(
                    {
                        "stageB_virtual_top_names": list(stageB_virtual_top_names or []),
                        "stageB_virtual_portfolio": _make_json_serializable(
                            stageB_virtual_portfolio
                        ),
                        "stageB_y_shortlist_names": list(stageB_portfolio_names or []),
                        "stageB_y_shortlist_sources": dict(stageB_y_shortlist_sources or {}),
                    }
                )

            stageB_y_op, stageB_y_op_inv, stageB_y_op_name = _resolve_stageB_y_transform(
                stageB_portfolio_names[0] if stageB_portfolio_names else (final_y_op_name or "identity")
            )
            # Stage B refits the candidate y-space and rejects unacceptable
            # initial fits, so Stage-A leaves are used only as warm starts.
            use_stageA_reuse = True

            # Extract AST with proper tags from the final model
            tagged_ast, _ = ast_from_composite(final_model)
            stageA_models_for_stageB = None
            if len(filepaths) > 1 and use_stageA_reuse:
                stageA_models_attr = getattr(final_model, "_stageA_models", None)
                if (
                    isinstance(stageA_models_attr, (list, tuple))
                    and len(stageA_models_attr) == len(filepaths)
                ):
                    stageA_models_for_stageB = list(stageA_models_attr)
                    print(
                        f"[Stage B] Using per-dataset Stage-A models from Stage A multi-state "
                        f"({len(stageA_models_for_stageB)} datasets)."
                    )
                else:
                    stageA_models_for_stageB = None
            if len(filepaths) > 1 and use_stageA_reuse and stageA_models_for_stageB is None:
                try:
                    stageA_models_for_stageB = _train_stageA_models_multi_for_stageB(
                        base_model=final_model,
                        ast_template=tagged_ast,
                        filepaths=list(filepaths),
                        Nxvars=Nxvars,
                        np_dtype=np_dtype,
                        data_hp=data_hp,
                        y_op=stageB_y_op,
                        leaf_builder=leaf_builder,
                        lm_hp=lm_hp,
                        device=device,
                        dtype=dtype,
                    )
                except Exception as e:
                    print(
                        "[Stage B] Warning: failed to build per-dataset Stage-A teachers; "
                        f"falling back to shared dataset-0 initialisation. ({e})"
                    )
                    stageA_models_for_stageB = None

            # Build kwargs for Stage B configuration
            stageB_kwargs = {
                "stageA_model": final_model,
                "stageA_ast": tagged_ast,  # Use AST with proper tags
                "filepath": filepaths if len(filepaths) > 1 else filepath,
                "Nxvars": Nxvars,
                "data_hp": data_hp,
                "lm_hp": lm_hp,
                "device": device,
                "dtype": dtype,
                "np_dtype": np_dtype,
                "y_op": stageB_y_op,
                "y_op_inv": stageB_y_op_inv,
                "y_transform_name": stageB_y_op_name,
                "y_raw_full": y_data,
                "noise_sigma_y": noise_sigma_y,
                "noise_floor_mc_samples": args.noise_floor_mc_samples,
                "fresh_nn_factory": fresh_nn_factory,
                "disabled_patterns": disabled_patterns,
                "use_stageA_reuse": use_stageA_reuse,
                "verbose_separabilities": bool(getattr(search_hp, 'verbose_separabilities', False)),
                "use_factorized_search": getattr(args, 'use_factorized_search', True),
                "factorized_search_hp": factorized_search_hp,
                "phase_hints": phase_hints,
                "phase_context_hints": phase_context_hints,
                "outer_link_hints": outer_link_hints,
            }
            if stageA_models_for_stageB is not None:
                stageB_kwargs["stageA_models"] = stageA_models_for_stageB

            # Add optional CLI-configurable parameters if provided
            if args.stageB_max_outer_iters is not None:
                stageB_kwargs["max_outer_iters"] = args.stageB_max_outer_iters
            if args.stageB_epochs is not None:
                stageB_kwargs["epochs_stageB"] = args.stageB_epochs
            if args.stageB_score_tol is not None:
                stageB_kwargs["score_tol"] = args.stageB_score_tol
            if args.max_backtracks is not None:
                stageB_kwargs["max_backtracks"] = args.max_backtracks

            # Units straightjacket (optional)
            if units_payload is not None:
                from nestynet_sr.sr_core.units import UnitsSpec

                def _make_units_spec(_y_name):
                    return UnitsSpec(
                        unit_system=units_payload["unit_system"],
                        x_dims=units_payload["x_dims"],
                        y_dim=units_payload["y_dim"],
                        y_transform_name=_y_name,
                        policy=args.units_policy,
                        nn_semantics=args.nn_units_semantics,
                        free_const_dims=units_payload.get("free_const_dims", {}),
                        free_const_scope=units_payload.get("free_const_scope", {}),
                        fixed_const_dims=units_payload.get("fixed_const_dims", {}),
                        fixed_const_values=units_payload.get("fixed_const_values", {}),
                        fixed_const_mode=units_payload.get("fixed_const_mode", "strict"),
                    )
                units_spec = _make_units_spec(stageB_y_op_name)
                stageB_kwargs["units_spec"] = units_spec
                stageB_kwargs["enforce_units"] = bool(args.enforce_units)
                if args.enforce_units:
                    print(
                        f"[Units] Enforcing units in Stage B (policy={args.units_policy}, y_transform={stageB_y_op_name})."
                    )
            # -----------------------------------------------------------
            # Stage A <-> B feedback loop
            # -----------------------------------------------------------
            # Loop A -> B -> A -> B -> ... until convergence (expression
            # stops changing) or the safety cap is hit.  Multi-dataset
            # mode is single-pass for v1.
            max_ab_iters = search_hp.max_ab_iters
            if len(filepaths) > 1:
                max_ab_iters = 1  # single-pass for multi-dataset (v1)

            # Ensure variables needed by feedback Stage A are in scope
            # (always defined for fresh runs, may be absent from checkpoint path).
            stageA_filepath_arg = filepaths if len(filepaths) > 1 else filepath  # noqa: F841
            try:
                idx_identity = y_transform_names.index("identity")
            except ValueError:
                idx_identity = 0

            prev_expr_signature = (
                loaded_checkpoint.get("prev_expr_signature")
                if (_stageB_resume_restored and loaded_checkpoint is not None)
                else None
            )  # tracks expression for convergence

            # --- Mid-loop resume support ---
            _ab_start_iter = 1

            if _resume_ab_iter > 0 and loaded_checkpoint is not None:
                if _resume_phase == "after_stageB":
                    if _stageB_resume_restored:
                        _ab_start_iter = _resume_ab_iter + 1
                        print(
                            f"[A<->B] Resuming after Stage B iter {_resume_ab_iter}; "
                            f"continuing from iter {_ab_start_iter}"
                        )
                    else:
                        print("[A<->B] Stage-B resume artifact unavailable; restarting loop from iter 1.")

                elif _resume_phase == "after_stageA" and _resume_ab_iter > 0:
                    # Resume at this iteration: model on disk is from feedback Stage A,
                    # so we just need to run Stage B (skip feedback Stage A).
                    _ab_start_iter = _resume_ab_iter
                    print(
                        f"[A<->B] Resuming after Stage A iter {_resume_ab_iter}; "
                        f"running Stage B at iter {_ab_start_iter}"
                    )

            stageB_portfolio_done = False

            def _stageB_kwargs_for_y_space(y_name):
                cand_y_op, cand_y_op_inv, cand_y_name = _resolve_stageB_y_transform(y_name)
                cand_kwargs = dict(stageB_kwargs)
                cand_kwargs["stageA_model"] = copy.deepcopy(
                    stageB_kwargs.get("stageA_model", final_model)
                )
                cand_kwargs["stageA_ast"] = copy.deepcopy(
                    stageB_kwargs.get("stageA_ast", tagged_ast)
                )
                if stageB_kwargs.get("stageA_models", None) is not None:
                    cand_kwargs["stageA_models"] = [
                        copy.deepcopy(m) for m in stageB_kwargs.get("stageA_models", [])
                    ]
                cand_kwargs["y_op"] = cand_y_op
                cand_kwargs["y_op_inv"] = cand_y_op_inv
                cand_kwargs["y_transform_name"] = cand_y_name
                cand_kwargs["use_stageA_reuse"] = True
                if units_payload is not None:
                    cand_kwargs["units_spec"] = _make_units_spec(cand_y_name)
                    cand_kwargs["enforce_units"] = bool(args.enforce_units)
                return cand_y_name, cand_y_op, cand_y_op_inv, cand_kwargs

            def _print_stageB_banner(y_name, suffix=""):
                print(f"\n{'='*70}")
                title = f"  [Stage B] ANALYTICAL REWRITING — {y_name}(y)-space"
                if suffix:
                    title += f" {suffix}"
                print(title)
                print(f"{'='*70}")

            for ab_iter in range(_ab_start_iter, max_ab_iters + 1):

                # On iteration 2+: feed Stage B output back to Stage A
                # (skip if this is the first iteration after an after_stageA resume)
                if ab_iter > 1 and not (
                    ab_iter == _ab_start_iter
                    and _resume_phase == "after_stageA"
                    and _resume_ab_iter > 0
                ):
                    # Check: any multivariate NNs left to open?
                    num_mv = getattr(stageB_state, "num_multivar_nn_atoms", None) or 0
                    if num_mv <= 0:
                        print("[A<->B] No multivariate NN atoms remain; converged.")
                        break

                    # Check: did expression change since last pass?
                    cur_sig = stageB_state.phi_expr_str or ast_to_human_readable(stageB_state.root)
                    if cur_sig == prev_expr_signature:
                        print("[A<->B] Expression unchanged; converged.")
                        break
                    prev_expr_signature = cur_sig

                    print(f"\n{'='*60}")
                    print(f"[A<->B] Iteration {ab_iter}: re-running Stage A on Stage B output")
                    print(f"{'='*60}")

                    feedback_ast = stageB_state.root
                    feedback_reuse = stageB_state.reuse
                    _stageB_feedback_xmap = None
                    try:
                        _stageB_feedback_xmap = getattr(final_model, "_x_transform", None)
                    except Exception:
                        _stageB_feedback_xmap = None
                    if not _stageB_feedback_xmap:
                        try:
                            _stageB_feedback_xmap = getattr(stageB_state, "x_transform_map", None)
                        except Exception:
                            _stageB_feedback_xmap = None
                    _maybe_run_reference_state_scouts(
                        phase_kind="stageB_feedback",
                        phase_label=f"continuation_after_stageB_iter_{max(1, int(ab_iter) - 1):03d}",
                        reason=f"stageB feedback iter {max(1, int(ab_iter) - 1)}",
                        current_ast=feedback_ast,
                        y_transform_name=stageB_y_op_name,
                        pass_index=max(1, int(ab_iter) - 1),
                        x_transform_map=_stageB_feedback_xmap,
                    )

                    (
                        _fb_success,
                        _fb_model,
                        _fb_rest_add,
                        _fb_rest_mult,
                        _,
                        _fb_ast,
                        _,
                        _,
                    ) = run_separability_for_transform(
                        i_op=0,
                        y_op=stageB_y_op,
                        y_op_inv=stageB_y_op_inv,
                        candidate_sep_ops=[True],
                        y_transform_names=[stageB_y_op_name],
                        initial_ast=feedback_ast,
                        filepath=stageA_filepath_arg,
                        Nxvars=Nxvars,
                        y_med=y_med,
                        y_mad=y_mad,
                        np_dtype=np_dtype,
                        dtype=dtype,
                        device=device,
                        data_hp=data_hp,
                        model_hp=model_hp,
                        lm_hp=lm_hp,
                        search_hp=search_hp,
                        leaf_builder=leaf_builder,
                        model_output=model_output_identity,
                        model_sep_output=model_sep_output_identity,
                        mode="full",
                        units_payload=units_payload,
                        enforce_units=bool(args.enforce_units),
                        units_policy=str(args.units_policy),
                        nn_units_semantics=str(args.nn_units_semantics),
                        y_log_dynamic_range=y_log_dynamic_range,
                        y_abs_median=y_abs_median,
                        global_best_val_loss_base=global_best_val_loss_base,
                        reuse_leaves_init=feedback_reuse,
                        freeze_non_nn=True,
                        skip_initial_fit=True,
                        y_raw_full=y_data,
                        noise_sigma_y=noise_sigma_y,
                        noise_floor_mc_samples=args.noise_floor_mc_samples,
                        stageA_restart_callback=_stageA_continuation_scout_callback,
                    )

                    if _fb_model is None:
                        print("[A<->B] Stage A re-run produced no model; stopping loop.")
                        break

                    # Update Stage B inputs from feedback Stage A results
                    final_model = _fb_model
                    tagged_ast, _ = ast_from_composite(final_model)
                    stageB_kwargs["stageA_model"] = final_model
                    stageB_kwargs["stageA_ast"] = tagged_ast
                    stageB_kwargs["use_stageA_reuse"] = True

                    # Checkpoint after feedback Stage A
                    try:
                        ckpt.update({
                            "phase": "after_stageA",
                            "ab_iter": ab_iter,
                            "stageA_status": stageA_status,
                            "ast": _fb_ast or stageB_state.root,
                        })
                        with open(checkpoint_path, "wb") as _f:
                            pickle.dump(ckpt, _f)
                        print(f"[A<->B] Saved after-stageA checkpoint (iter {ab_iter})")
                    except Exception as _e:
                        print(f"[A<->B] Warning: failed to save after-stageA checkpoint: {_e}")

                # --- Run Stage B ---
                run_portfolio = (
                    (not stageB_portfolio_done)
                    and ab_iter == 1
                    and _resume_ab_iter == 0
                    and len(stageB_portfolio_names) > 1
                )
                stageB_y_branch_artifacts = []
                if run_portfolio:
                    portfolio_results = []
                    for cand_rank, cand_name0 in enumerate(stageB_portfolio_names):
                        cand_name, cand_y_op, cand_y_op_inv, cand_kwargs = _stageB_kwargs_for_y_space(
                            cand_name0
                        )
                        try:
                            _print_stageB_banner(
                                cand_name,
                                suffix=f"[portfolio {cand_rank + 1}/{len(stageB_portfolio_names)}]",
                            )
                            cand_state = run_stageB_from_model(**cand_kwargs)
                        except Exception as e:
                            print(
                                f"[Stage B] Portfolio candidate '{cand_name}' failed: "
                                f"{type(e).__name__}: {e}"
                            )
                            print(
                                f"[Stage B] Portfolio candidate '{cand_name}' traceback:\n"
                                f"{traceback.format_exc().rstrip()}"
                            )
                            continue
                        try:
                            setattr(cand_state, "_stageB_portfolio_y_name", cand_name)
                        except Exception:
                            pass
                        m = _stageB_candidate_metrics(cand_state, y_name=cand_name)
                        _orig_y = ""
                        if math.isfinite(m.get("original_y_val_loss", float("nan"))):
                            _orig_y = f", original_y={m['original_y_val_loss']:.4e}"
                        print(
                            "[Stage B] Portfolio result "
                            f"{cand_name}: val_loss={m['val_loss']:.4e}, "
                            f"NN(total={m['num_nn']}, multivar={m['num_multivar_nn']}, "
                            f"max_arity={m['max_nn_arity']}), "
                            f"accepted={m['accepted_patterns']}, "
                            f"generic={int(m['generic_approximant'])}, "
                            f"raw_family={m.get('raw_family') or '-'}, "
                            f"raw_protected={int(bool(m.get('raw_protected_family', False)))}, "
                            f"complexity={m['complexity_score']:.3g}, "
                            f"bad_loss={int(m['bad_loss'])}"
                            f"{_orig_y}"
                        )
                        portfolio_results.append(
                            (
                                cand_state,
                                cand_name,
                                cand_y_op,
                                cand_y_op_inv,
                                cand_kwargs,
                                cand_rank,
                            )
                        )
                        _cand_sources = stageB_y_shortlist_sources.get(cand_name, [])
                        _branch_artifact = _stageB_y_branch_artifact(
                            cand_state,
                            y_name=cand_name,
                            rank=cand_rank,
                            y_sources=_cand_sources,
                        )
                        if isinstance(_branch_artifact, dict):
                            stageB_y_branch_artifacts.append(_branch_artifact)
                        _can_stop, _stop_reason = _stageB_portfolio_early_stop_decision(
                            cand_state,
                            y_sources=_cand_sources,
                            y_name=cand_name,
                        )
                        if _can_stop:
                            print(
                                "[Stage B] Portfolio early-stop after confirmed validation-good "
                                f"branch '{cand_name}'."
                            )
                            break
                        if _stop_reason:
                            print(
                                "[Stage B] Portfolio continue after "
                                f"'{cand_name}': {_stop_reason}."
                            )

                    if portfolio_results:
                        (
                            stageB_state,
                            stageB_y_op_name,
                            stageB_y_op,
                            stageB_y_op_inv,
                            stageB_kwargs,
                            _chosen_rank,
                        ) = min(
                            portfolio_results,
                            key=lambda item: _stageB_adjudication_key(
                                item[0],
                                y_name=item[1],
                                rank=item[5],
                                y_sources=stageB_y_shortlist_sources.get(item[1], []),
                            ),
                        )
                        print(
                            "[Stage B] Portfolio selected "
                            f"'{stageB_y_op_name}' "
                            f"(rank={_chosen_rank + 1}/{len(stageB_portfolio_names)})."
                        )
                    else:
                        _print_stageB_banner(stageB_y_op_name)
                        stageB_state = run_stageB_from_model(**stageB_kwargs)
                    stageB_portfolio_done = True
                else:
                    _print_stageB_banner(stageB_y_op_name)
                    stageB_state = run_stageB_from_model(**stageB_kwargs)
                    stageB_portfolio_done = True

                    # Safety fallback: if chosen Stage-B y-space fails badly,
                    # retry in identity.  The portfolio path already includes
                    # identity, so this is mostly for forced/single-candidate runs.
                    stageB_loss_acceptable_eff = getattr(stageB_state, "loss_acceptable_eff", None)
                    if stageB_loss_acceptable_eff is None or (not math.isfinite(float(stageB_loss_acceptable_eff))):
                        stageB_loss_acceptable_eff = lm_hp.loss_acceptable
                    if (
                        stageB_y_op_name != "identity"
                        and stageB_state.val_loss > float(stageB_loss_acceptable_eff)
                    ):
                        print(
                            f"[Stage B] Chosen y-transform '{stageB_y_op_name}' did not achieve acceptable fit "
                            f"(val_loss={stageB_state.val_loss:.4e} > "
                            f"loss_acceptable_eff={float(stageB_loss_acceptable_eff):.4e}). "
                            f"Falling back to identity."
                        )
                        # Reset to identity
                        stageB_y_op_name = "identity"
                        stageB_y_op = None  # identity
                        stageB_y_op_inv = None  # identity
                        use_stageA_reuse = True  # Can reuse Stage-A weights in identity space

                        # Update kwargs and re-run Stage B
                        stageB_kwargs["y_op"] = stageB_y_op
                        stageB_kwargs["y_op_inv"] = stageB_y_op_inv
                        stageB_kwargs["y_transform_name"] = stageB_y_op_name
                        stageB_kwargs["use_stageA_reuse"] = use_stageA_reuse
                        if len(filepaths) > 1 and stageB_kwargs.get("stageA_models") is None:
                            try:
                                stageB_kwargs["stageA_models"] = _train_stageA_models_multi_for_stageB(
                                    base_model=final_model,
                                    ast_template=tagged_ast,
                                    filepaths=list(filepaths),
                                    Nxvars=Nxvars,
                                    np_dtype=np_dtype,
                                    data_hp=data_hp,
                                    y_op=stageB_y_op,
                                    leaf_builder=leaf_builder,
                                    lm_hp=lm_hp,
                                    device=device,
                                    dtype=dtype,
                                )
                            except Exception as e:
                                print(
                                    "[Stage B] Warning: identity fallback could not build per-dataset "
                                    f"Stage-A teachers ({e}); using shared initialisation."
                                )
                        if units_payload is not None:
                            stageB_kwargs["units_spec"] = _make_units_spec("identity")
                        _print_stageB_banner("identity", suffix="[fallback]")
                        stageB_state = run_stageB_from_model(**stageB_kwargs)

                if (
                    ab_iter == 1
                    and _resume_ab_iter == 0
                    and stageA_deferred_fitlink_branches
                ):
                    shadow_reason = _stageB_shadow_rescue_reason(stageB_state)
                    if shadow_reason:
                        print(
                            "[Stage B] Active Stage-A branch did not earn a branch-safe "
                            f"confirmation ({shadow_reason}); activating retained fit-link "
                            "branch(es) for one confirmation window."
                        )
                        branch_results = [
                            (
                                stageB_state,
                                stageB_y_op_name,
                                stageB_y_op,
                                stageB_y_op_inv,
                                stageB_kwargs,
                                0,
                                "active",
                                tuple(stageB_y_shortlist_sources.get(stageB_y_op_name, [])),
                                getattr(lm_hp, "fit_y_link", None),
                                float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                            )
                        ]
                        _orig_fit_link = getattr(lm_hp, "fit_y_link", None)
                        _orig_fit_scale = float(getattr(lm_hp, "fit_y_link_scale", 1.0))

                        for _shadow_i, _branch in enumerate(stageA_deferred_fitlink_branches, start=1):
                            if not isinstance(_branch, dict):
                                continue
                            _fit_link = _branch.get("fit_y_link", None)
                            if not _fit_link:
                                continue
                            _fit_scale = float(_branch.get("fit_y_link_scale", 1.0) or 1.0)
                            _branch_name_raw = str(_branch.get("name", f"fitlink_{_shadow_i}"))
                            _branch_name = "".join(
                                ch if (ch.isalnum() or ch in {"_", "-"}) else "_"
                                for ch in _branch_name_raw
                            ).strip("_") or f"fitlink_{_shadow_i}"
                            print(
                                f"[Stage A] Running retained fit-link branch '{_branch_name_raw}' "
                                f"(fit_link={_fit_link}, scale={_fit_scale:.4g})."
                            )
                            try:
                                lm_hp.fit_y_link = str(_fit_link)
                                lm_hp.fit_y_link_scale = float(_fit_scale)
                                _id_idx = idx_identity if idx_identity is not None else y_transform_names.index("identity")
                                _yt_id = y_transforms[_id_idx]
                                (
                                    _sh_success,
                                    _sh_model,
                                    _sh_rest_add,
                                    _sh_rest_mult,
                                    _,
                                    _sh_ast,
                                    _,
                                    _,
                                ) = run_separability_for_transform(
                                    i_op=_id_idx,
                                    y_op=_yt_id.np_op,
                                    y_op_inv=_yt_id.torch_inv,
                                    candidate_sep_ops=[True] * len(y_transform_names),
                                    y_transform_names=y_transform_names,
                                    initial_ast=copy.deepcopy(initial_ast),
                                    filepath=stageA_filepath_arg,
                                    Nxvars=Nxvars,
                                    y_med=y_med,
                                    y_mad=y_mad,
                                    np_dtype=np_dtype,
                                    dtype=dtype,
                                    device=device,
                                    data_hp=data_hp,
                                    model_hp=model_hp,
                                    lm_hp=lm_hp,
                                    search_hp=search_hp,
                                    leaf_builder=leaf_builder,
                                    model_output=_model_path(_branch_name, sep=False),
                                    model_sep_output=_model_path(_branch_name, sep=True),
                                    mode="full",
                                    units_payload=units_payload,
                                    enforce_units=bool(args.enforce_units),
                                    units_policy=str(args.units_policy),
                                    nn_units_semantics=str(args.nn_units_semantics),
                                    y_log_dynamic_range=y_log_dynamic_range,
                                    y_abs_median=y_abs_median,
                                    global_best_val_loss_base=global_best_val_loss_base,
                                    y_raw_full=y_data,
                                    noise_sigma_y=noise_sigma_y,
                                    noise_floor_mc_samples=args.noise_floor_mc_samples,
                                )
                                if _sh_model is not None:
                                    _bvlb = getattr(_sh_model, "_best_val_loss_base", None)
                                    if (
                                        _bvlb is not None
                                        and math.isfinite(float(_bvlb))
                                        and (
                                            global_best_val_loss_base is None
                                            or float(_bvlb) < float(global_best_val_loss_base)
                                        )
                                    ):
                                        global_best_val_loss_base = float(_bvlb)
                                if _sh_model is None:
                                    print(f"[Stage A] Retained branch '{_branch_name_raw}' produced no model; skipping.")
                                    continue

                                _sh_tagged_ast, _ = ast_from_composite(_sh_model)
                                _sh_kwargs = dict(stageB_kwargs)
                                _sh_kwargs.pop("stageA_models", None)
                                _sh_kwargs.update(
                                    {
                                        "stageA_model": _sh_model,
                                        "stageA_ast": _sh_tagged_ast,
                                        "y_op": None,
                                        "y_op_inv": None,
                                        "y_transform_name": "identity",
                                        "use_stageA_reuse": True,
                                    }
                                )
                                if units_payload is not None:
                                    _sh_kwargs["units_spec"] = _make_units_spec("identity")
                                    _sh_kwargs["enforce_units"] = bool(args.enforce_units)

                                _print_stageB_banner(
                                    "identity",
                                    suffix=f"[retained fit-link {str(_fit_link)}]",
                                )
                                _sh_state = run_stageB_from_model(**_sh_kwargs)
                                _m = _stageB_candidate_metrics(_sh_state)
                                print(
                                    "[Stage B] Retained fit-link result "
                                    f"{_branch_name}: val_loss={_m['val_loss']:.4e}, "
                                    f"NN(total={_m['num_nn']}, multivar={_m['num_multivar_nn']}, "
                                    f"max_arity={_m['max_nn_arity']}), "
                                    f"accepted={_m['accepted_patterns']}, "
                                    f"generic={int(_m['generic_approximant'])}, "
                                    f"complexity={_m['complexity_score']:.3g}, "
                                    f"bad_loss={int(_m['bad_loss'])}"
                                )
                                branch_results.append(
                                    (
                                        _sh_state,
                                        "identity",
                                        None,
                                        None,
                                        _sh_kwargs,
                                        _shadow_i,
                                        _branch_name,
                                        ("baseline", "deferred_fitlink"),
                                        str(_fit_link),
                                        float(_fit_scale),
                                    )
                                )
                            except Exception as _e:
                                print(
                                    f"[Stage B] Retained fit-link branch '{_branch_name_raw}' failed: "
                                    f"{type(_e).__name__}: {_e}"
                                )
                            finally:
                                lm_hp.fit_y_link = _orig_fit_link
                                lm_hp.fit_y_link_scale = _orig_fit_scale

                        if len(branch_results) > 1:
                            _chosen = min(
                                branch_results,
                                key=lambda item: _stageB_adjudication_key(
                                    item[0],
                                    y_name=item[1],
                                    rank=item[5],
                                    y_sources=item[7],
                                ),
                            )
                            if _chosen[6] != "active":
                                (
                                    stageB_state,
                                    stageB_y_op_name,
                                    stageB_y_op,
                                    stageB_y_op_inv,
                                    stageB_kwargs,
                                    _chosen_rank,
                                    _chosen_label,
                                    _chosen_sources,
                                    _chosen_fit_link,
                                    _chosen_fit_scale,
                                ) = _chosen
                                lm_hp.fit_y_link = _chosen_fit_link
                                lm_hp.fit_y_link_scale = float(_chosen_fit_scale)
                                fit_y_link_used = _chosen_fit_link
                                fit_y_link_scale_used = float(_chosen_fit_scale)
                                final_model = stageB_kwargs.get("stageA_model", final_model)
                                final_ast = stageB_kwargs.get("stageA_ast", final_ast)
                                final_y_op = stageB_y_op
                                final_y_op_inv = stageB_y_op_inv
                                final_y_op_name = stageB_y_op_name
                                print(
                                    "[Stage B] Selected retained fit-link branch "
                                    f"'{_chosen_label}' over active branch."
                                )
                            else:
                                lm_hp.fit_y_link = _orig_fit_link
                                lm_hp.fit_y_link_scale = _orig_fit_scale
                                print("[Stage B] Active branch remains preferred over retained fit-link branch(es).")

                # Save a complete resume checkpoint after every Stage-B pass,
                # including the normal single-pass path.
                try:
                    _save_stageB_resume_checkpoint(
                        ab_iter=ab_iter if max_ab_iters > 1 else 0,
                        prev_expr_signature=prev_expr_signature,
                        log_prefix="[A<->B]" if max_ab_iters > 1 else "[Stage B]",
                    )
                except Exception as _e:
                    print(f"[Stage B] Warning: failed to save after-stageB checkpoint: {_e}")

                # Single-pass: no loop needed if max_ab_iters == 1
                if max_ab_iters <= 1:
                    break

            # If we ran in multi-dataset mode, compute per-dataset printable expressions
            phi_expr_strs = None
            try:
                if getattr(stageB_state, "models", None) is not None:
                    from nestynet_sr.sr_search.stageB.representation import (
                        pretty_print_state as _pp,
                    )

                    phi_expr_strs = []
                    for m in stageB_state.models:
                        st_tmp = copy.copy(stageB_state)
                        st_tmp.model = m
                        phi_expr_strs.append(_pp(st_tmp, sig=16))
            except Exception:
                phi_expr_strs = None
            stageB_output = os.path.join(results_dir, f"{base_filename}_stageB.pkl")
            with open(stageB_output, "wb") as f_stageB:
                pickle.dump(
                    {
                        "stageB_ast": stageB_state.root,
                        "stageB_val_loss": stageB_state.val_loss,
                        "stageB_val_losses": getattr(stageB_state, "val_losses", None),
                        "stageB_dataset_ids": getattr(stageB_state, "dataset_ids", None),
                        "stageB_agg_mode": getattr(stageB_state, "agg_mode", None),
                        "phi_expr_strs": phi_expr_strs,
                        "phi_expr_str": stageB_state.phi_expr_str,
                        "y_expr_str": stageB_state.y_expr_str,
                        "phi_expr_raw_str": getattr(stageB_state, "phi_expr_raw_str", None),
                        "y_expr_raw_str": getattr(stageB_state, "y_expr_raw_str", None),
                        "x_transform_map": getattr(stageB_state, "x_transform_map", getattr(final_model, "_x_transform", None)),
                        "sympy_meta": stageB_state.sympy_meta,
                        "coefficient_metadata": getattr(
                            stageB_state, "coefficient_metadata", None
                        ),
                        "coefficient_metadata_by_dataset": getattr(
                            stageB_state, "coefficient_metadata_by_dataset", None
                        ),
                        "y_selected": stageB_y_op_name,
                        "y_shortlist_names": list(stageB_portfolio_names or []),
                        "y_shortlist_sources": dict(stageB_y_shortlist_sources or {}),
                    },
                    f_stageB,
                )

            # Write final human-readable file
            final_human_output = os.path.join(results_dir, f"{base_filename}_final.human")
            with open(final_human_output, "w") as f_final:
                f_final.write(f"Final AST: {stageB_state.root}\n")
                f_final.write(f"Selected y-transform: {stageB_y_op_name}\n")
                f_final.write(f"Expression (φ-space): {stageB_state.phi_expr_str}\n")
                f_final.write(f"Expression (y-space): {stageB_state.y_expr_str}\n")
                if getattr(stageB_state, "phi_expr_raw_str", None) is not None:
                    f_final.write(f"Expression (φ-space, raw x): {stageB_state.phi_expr_raw_str}\n")
                if getattr(stageB_state, "y_expr_raw_str", None) is not None:
                    f_final.write(f"Expression (y-space, raw x): {stageB_state.y_expr_raw_str}\n")

            # Write persistent decision log (Layer 1 of backtracking infrastructure)
            _decision_log = getattr(stageB_state, "decision_log", None)
            if _decision_log:
                decisions_path = os.path.join(results_dir, f"{base_filename}.decisions.json")
                with open(decisions_path, "w") as f_dec:
                    json.dump(_decision_log, f_dec, indent=2, default=str)

    # ── Class SR: joint fitting with shared constants ────────────────
    class_sr_result = None
    class_sr_summary = None
    param_sr_dataset_metadata = None
    if (
        getattr(args, "class_sr", False)
        and len(filepaths) > 1
        and stageB_state is not None
        and getattr(stageB_state, "models", None) is not None
    ):
        print("\n" + "=" * 60)
        print("CLASS SR: joint fitting with shared constants")
        print("=" * 60)
        try:
            from nestynet_sr.sr_search.class_sr import run_class_sr
            from nestynet_sr.sr_search.data_utils import build_datasets_multi

            # Rebuild datasets for all files (same y-space as Stage B)
            _, _, class_sr_train_loaders, class_sr_val_loaders = build_datasets_multi(
                filepaths=list(filepaths),
                Nxvars=Nxvars,
                np_dtype=np_dtype,
                data_hp=data_hp,
                y_op=stageB_y_op,
            )
            # Keep class-SR fitting in the same internal x-coordinate system used by Stage B.
            class_sr_xmap = getattr(stageB_state, "x_transform_map", None) or {}
            if class_sr_xmap:
                try:
                    from nestynet_sr.sr_search.xcoord import XCoordSystem

                    xcoords = XCoordSystem.from_map(class_sr_xmap, Nx_raw=Nxvars)
                    if xcoords is not None and (not xcoords.is_identity()):
                        def _x_op(x, _xc=xcoords):
                            return _xc.apply_torch(x)

                        def _wrap_loader(dl):
                            return torch.utils.data.DataLoader(
                                _StageAXTransformDataset(dl.dataset, _x_op),
                                batch_size=getattr(dl, "batch_size", None) or data_hp.batch_size,
                                shuffle=False,
                                drop_last=bool(getattr(dl, "drop_last", False)),
                            )

                        class_sr_train_loaders = [_wrap_loader(dl) for dl in class_sr_train_loaders]
                        class_sr_val_loaders = [_wrap_loader(dl) for dl in class_sr_val_loaders]
                        print("[Class SR] Applied Stage-A x-coordinate transform map.")
                except Exception as e:
                    print(
                        "[Class SR] Warning: failed to apply Stage-A x-transform map "
                        f"to class-SR loaders ({e})"
                    )

            # Parse manual class atom tags if specified
            manual_class_tags = None
            if args.class_atoms is not None:
                manual_class_tags = [t.strip() for t in args.class_atoms.split(",")]

            # Optional dataset metadata scalars for Parameter-SR.
            if args.class_param_sr_metadata is not None:
                param_sr_dataset_metadata = _normalize_class_param_sr_metadata(
                    args.class_param_sr_metadata,
                    dataset_paths=list(filepaths),
                )
                _meta_keys = sorted(
                    {str(k) for row in param_sr_dataset_metadata for k in row.keys()}
                )
                print(
                    "[Class SR] Parameter-SR metadata enabled: "
                    f"{len(_meta_keys)} scalar(s): {_meta_keys}"
                )

            # Build per-dataset states from Stage B
            per_dataset_states = []
            stageB_reuses = list(getattr(stageB_state, "reuses", None) or [])
            stageB_val_losses = list(getattr(stageB_state, "val_losses", None) or [])
            for i, m in enumerate(stageB_state.models):
                st = copy.copy(stageB_state)
                st.model = m
                st.reuse = (
                    stageB_reuses[i]
                    if i < len(stageB_reuses)
                    else (stageB_state.reuse or {})
                )
                if i < len(stageB_val_losses):
                    st.val_loss = float(stageB_val_losses[i])
                st.models = None
                st.reuses = None
                st.val_losses = None
                per_dataset_states.append(st)

            class_sr_result = run_class_sr(
                root=stageB_state.root,
                states=per_dataset_states,
                train_loaders=class_sr_train_loaders,
                val_loaders=class_sr_val_loaders,
                device=device,
                dtype=dtype,
                cv_threshold=args.class_cv_threshold,
                class_atom_tags=manual_class_tags,
                max_points_per_dataset=args.class_sr_max_points,
                optimizer_backend=args.class_sr_optimizer,
                param_sr_enable=bool(args.class_param_sr),
                param_sr_max_invariants=args.class_param_sr_max_invariants,
                param_sr_score_threshold=args.class_param_sr_score_threshold,
                param_sr_penalty_weight=args.class_param_sr_penalty_weight,
                param_sr_max_scalars=args.class_param_sr_max_scalars,
                param_sr_dataset_metadata=param_sr_dataset_metadata,
                auto_include_scale_leaves=bool(args.class_auto_include_scales),
                auto_focus_free_const_leaves=(not bool(args.class_auto_include_nonfree)),
                lm_verbose=lm_hp.LM_verbose,
            )

            # Report results
            print("\nClass SR results:")
            print(f"  Class atoms (shared):      {class_sr_result.class_tags}")
            print(f"  Experiment atoms (per-ds):  {class_sr_result.experiment_tags}")
            print(f"  CV per atom: {class_sr_result.cv_per_tag}")
            print(
                "  Joint val loss "
                f"({class_sr_result.val_loss_agg_mode}): "
                f"{class_sr_result.val_loss_agg:.6e}"
            )
            print(f"  Per-dataset val losses:    {class_sr_result.val_losses}")
            if class_sr_result.derived_invariants:
                print("  Derived invariants (Parameter-SR):")
                for di in class_sr_result.derived_invariants:
                    _score = float(di.get("score", float("nan")))
                    _cv = float(di.get("cv", float("nan")))
                    print(
                        f"    {di.get('expr')} "
                        f"(score={_score:.4g}, cv={_cv:.4g})"
                    )
            print("\n  Class (shared) parameters:")
            for tag, pv in class_sr_result.class_params.items():
                print(f"    {tag}: {pv.tolist()}")
            print("\n  Experiment (per-dataset) parameters:")
            for i, ep in enumerate(class_sr_result.experiment_params):
                ds_name = stageB_state.dataset_ids[i] if stageB_state.dataset_ids else f"dataset_{i}"
                print(f"    {ds_name}:")
                for tag, pv in ep.items():
                    print(f"      {tag}: {pv.tolist()}")

            # Save class SR results
            class_sr_output = os.path.join(results_dir, f"{base_filename}_classSR.json")
            class_sr_data = {
                "class_tags": class_sr_result.class_tags,
                "experiment_tags": class_sr_result.experiment_tags,
                "cv_per_tag": {k: float(v) for k, v in class_sr_result.cv_per_tag.items()},
                "val_loss_agg": class_sr_result.val_loss_agg,
                "val_loss_agg_mode": class_sr_result.val_loss_agg_mode,
                "val_losses": class_sr_result.val_losses,
                "derived_invariants": class_sr_result.derived_invariants,
                "class_params": {k: v.tolist() for k, v in class_sr_result.class_params.items()},
                "experiment_params": [
                    {k: v.tolist() for k, v in ep.items()}
                    for ep in class_sr_result.experiment_params
                ],
            }
            with open(class_sr_output, "w") as f_cls:
                json.dump(class_sr_data, f_cls, indent=2)
            print(f"\nClass SR results saved to {class_sr_output}")
            class_sr_summary = {
                "enabled": True,
                "results_path": str(class_sr_output),
                "class_tags": list(class_sr_result.class_tags),
                "experiment_tags": list(class_sr_result.experiment_tags),
                "dataset_ids": list(getattr(stageB_state, "dataset_ids", None) or []),
                "val_loss_agg": float(class_sr_result.val_loss_agg),
                "val_loss_agg_mode": str(class_sr_result.val_loss_agg_mode),
            }

            # Human-readable summary
            class_sr_human = os.path.join(results_dir, f"{base_filename}_classSR.human")
            with open(class_sr_human, "w") as f_clsh:
                f_clsh.write("Class SR Joint Fitting Results\n")
                f_clsh.write("=" * 40 + "\n\n")
                f_clsh.write(f"Class (shared) atoms: {class_sr_result.class_tags}\n")
                f_clsh.write(f"Experiment (per-dataset) atoms: {class_sr_result.experiment_tags}\n\n")
                f_clsh.write(
                    f"Joint validation loss ({class_sr_result.val_loss_agg_mode}): "
                    f"{class_sr_result.val_loss_agg:.6e}\n"
                )
                f_clsh.write(f"Per-dataset val losses: {class_sr_result.val_losses}\n\n")
                if class_sr_result.derived_invariants:
                    f_clsh.write("Derived invariants (Parameter-SR):\n")
                    for di in class_sr_result.derived_invariants:
                        f_clsh.write(
                            f"  {di.get('expr')} "
                            f"(score={float(di.get('score', float('nan'))):.4g}, "
                            f"cv={float(di.get('cv', float('nan'))):.4g})\n"
                        )
                    f_clsh.write("\n")
                _meta_linked = _metadata_linked_invariants(class_sr_result.derived_invariants)
                if _meta_linked:
                    f_clsh.write("Metadata-linked invariants (Parameter-SR):\n")
                    for di in _meta_linked:
                        f_clsh.write(
                            f"  {di.get('expr')} "
                            f"(score={float(di.get('score', float('nan'))):.4g}, "
                            f"cv={float(di.get('cv', float('nan'))):.4g})\n"
                        )
                    f_clsh.write("\n")
                elif param_sr_dataset_metadata:
                    f_clsh.write(
                        "Metadata-linked invariants (Parameter-SR): none discovered.\n\n"
                    )
                f_clsh.write("Class parameters:\n")
                for tag, pv in class_sr_result.class_params.items():
                    f_clsh.write(f"  {tag}: {pv.tolist()}\n")
                f_clsh.write("\nExperiment parameters:\n")
                for i, ep in enumerate(class_sr_result.experiment_params):
                    ds_name = stageB_state.dataset_ids[i] if stageB_state.dataset_ids else f"dataset_{i}"
                    f_clsh.write(f"  {ds_name}:\n")
                    for tag, pv in ep.items():
                        f_clsh.write(f"    {tag}: {pv.tolist()}\n")
            print(f"Class SR human-readable saved to {class_sr_human}")

        except Exception as e:
            print(f"Warning: Class SR failed: {e}")
            traceback.print_exc()
            class_sr_summary = {
                "enabled": False,
                "error": str(e),
            }

    time_now = timeit.default_timer()
    elapsed_time = (time_now - start_time) / 3600.0
    print("\nElapsed time: {} hours".format(elapsed_time))

    # Persist GS diagnostics (promotion reasons, policy events) for SR runs,
    # mirroring the run_de.py report path. No-op content when GS is off.
    if search_hp.gs_config is not None and search_hp.gs_config.active():
        try:
            from nestynet_sr.sr_gs.reporting import write_gs_reports

            write_gs_reports(
                json_path=os.path.join(results_dir, f"{base_filename}.gs_report.json"),
                markdown_path=os.path.join(results_dir, f"{base_filename}.gs_report.md"),
                mode=str(search_hp.gs_config.mode),
            )
            print(f"[GS] Wrote GS report to {os.path.join(results_dir, f'{base_filename}.gs_report.json')}")
        except Exception as exc:
            print(f"[GS] Report writing failed: {type(exc).__name__}: {exc}")

    # Generate JSON report if requested or by default
    if args.report_json is not None:
        report_path = args.report_json
    else:
        report_path = os.path.join(results_dir, f"{base_filename}.report.json")
    discovery_report_path = None
    if bool(args.discovery_enable):
        if args.discovery_report_json is not None:
            discovery_report_path = str(args.discovery_report_json)
        else:
            report_path_obj = pathlib.Path(str(report_path))
            if report_path_obj.name.endswith(".report.json"):
                discovery_report_path = str(
                    report_path_obj.with_name(
                        report_path_obj.name[: -len(".report.json")] + ".discovery.json"
                    )
                )
            else:
                discovery_report_path = str(
                    report_path_obj.with_name(f"{report_path_obj.stem}.discovery.json")
                )

    # Prepare Stage A data
    stageA_data = None
    if final_ast is not None:
        # Pure NN info for simplification path step 0
        _nn_val_loss = (
            getattr(final_model, "_stageA_initial_val_loss", None)
            if final_model is not None
            else None
        )
        if _nn_val_loss is None:
            _nn_val_loss = locals().get("identity_val_loss")
        if _nn_val_loss is None:
            _nn_val_loss = stageA_val_loss  # fallback for single-transform path
        _nn_val_losses = (
            getattr(final_model, "_stageA_initial_val_losses", None)
            if final_model is not None
            else None
        )
        _nn_n_params = None
        if final_model is not None:
            _nn_n_params = getattr(final_model, "_stageA_initial_n_params", None)
        if _nn_n_params is None:
            _nn_model = locals().get("stageA_model")
            if _nn_model is not None and hasattr(_nn_model, "num_parameters"):
                _nn_n_params = int(_nn_model.num_parameters())
            elif final_model is not None and hasattr(final_model, "num_parameters"):
                _nn_n_params = int(final_model.num_parameters())
        _stageA_ledgers = _stageA_ledgers_from_model_or_checkpoint(
            model=final_model,
            checkpoint=loaded_checkpoint,
        )
        _stageA_move_records = _stageA_ledgers["stageA_move_records"]
        _stageA_provisional_commits = _stageA_ledgers["stageA_provisional_commits"]
        _stageA_rejection_records = _stageA_ledgers["stageA_rejection_records"]

        stageA_data = {
            "y_op_name": final_y_op_name,
            "stageA_status": stageA_status,
            "ast": final_ast,
            "initial_ast": initial_ast,
            "nn_val_loss": float(_nn_val_loss) if _nn_val_loss is not None else None,
            "nn_val_losses": _make_json_serializable(_nn_val_losses),
            "nn_n_params": _nn_n_params,
            "val_loss": stageA_val_loss,  # Set from checkpoint or loaded from model file
            "val_losses": stageA_val_losses,
            "val_loss_agg_mode": stageA_val_loss_agg_mode,
            "val_loss_agg_weights": stageA_val_loss_agg_weights,
            "dataset_ids": stageA_dataset_ids,
            "fit_y_link": fit_y_link_used,
            "fit_y_link_scale": fit_y_link_scale_used,
            "fit_link_branch_certificate": (
                dict(fit_link_branch_certificate)
                if isinstance(fit_link_branch_certificate, dict)
                else fit_link_branch_certificate
            ),
            "fit_link_branch_status": fit_link_branch_status,
            "fit_link_original_y_certified": fit_link_original_y_certified,
            "fit_link_original_y_val_loss": fit_link_original_y_val_loss,
            "fit_link_original_y_allowed_loss": fit_link_original_y_allowed_loss,
            "rest_add": rest_add_final,
            "rest_mult": rest_mult_final,
            "outer_peel_square": outer_peel_square_decision,
            "outer_peel_ranked": outer_peel_ranked,
            "stageB_virtual_top_names": list(stageB_virtual_top_names or []),
            "stageB_virtual_portfolio": _make_json_serializable(
                stageB_virtual_portfolio
            ),
            "stageB_y_shortlist_sources": dict(stageB_y_shortlist_sources or {}),
            "deferred_fitlink_branches": list(stageA_deferred_fitlink_branches or []),
            "coe_stageA_ybranch_committee": _make_json_serializable(coe_stageA_ybranch_committee),
            "coe_stageA_compound_shortlist": _make_json_serializable(coe_stageA_compound_shortlist),
            "coe_stageA_fit_tournament_records": _make_json_serializable(
                getattr(lm_hp, "coe_stageA_fit_tournament_records", [])
            ),
            "coe_stageA_materialization": _make_json_serializable(coe_stageA_materialization),
            "coe_stageA_continuation_scouts": _make_json_serializable(
                coe_stageA_continuation_scout_summaries
            ),
            "coe_scout_expression_reservoir": _make_json_serializable(
                coe_scout_expression_reservoir
            ),
            "coe_stageA_replay_log": _make_json_serializable(
                getattr(search_hp, "coe_stageA_replay_log", None)
            ),
            "stageA_move_records": _make_json_serializable(_stageA_move_records),
            "stageA_provisional_commits": _make_json_serializable(_stageA_provisional_commits),
            "stageA_rejection_records": _make_json_serializable(_stageA_rejection_records),
            "x_transform_map": getattr(final_model, "_x_transform", None) if final_model else None,
            "phase_prescan_direct_closure": getattr(final_model, "_phase_prescan_direct_closure", None) if final_model else None,
        }
        _coe_stageA_enabled = bool(
            getattr(args, "coe_stageA_dry_run", False)
            or str(getattr(args, "coe_mode", "off") or "off") in {"committee_gated", "reservoir_discovery"}
        )
        if _coe_stageA_enabled:
            _coe_stageA_exit_audit = _run_coe_stageA_exit_audit(
                args=args,
                filepath=filepath,
                results_dir=results_dir,
                base_filename=base_filename,
                stageA_data=stageA_data,
                noise_sigma_y=noise_sigma_y,
                y_op_inv=final_y_op_inv,
                initial_model=locals().get("stageA_model"),
                final_model=final_model,
                units_spec=locals().get("units_spec"),
            )
            if _coe_stageA_exit_audit:
                stageA_data["coe_stageA_exit_audit"] = _coe_stageA_exit_audit
                print("\n" + _format_coe_stageA_exit_audit_report(_coe_stageA_exit_audit))
            _coe_stageA_log = _build_coe_stageA_dry_run_records(
                stageA_data,
                noise_sigma_y=noise_sigma_y,
            )
            _coe_stageA_summary = _summarize_coe_stageA_dry_run(
                _coe_stageA_log,
                enabled=True,
            )
            _coe_stageA_jsonl = _write_coe_stageA_dry_run_jsonl(
                os.path.join(results_dir, f"{base_filename}.coe_stageA_dry_run.jsonl"),
                _coe_stageA_log,
            )
            if _coe_stageA_jsonl:
                _coe_stageA_summary["jsonl_path"] = _coe_stageA_jsonl
            stageA_data["coe_stageA_dry_run_summary"] = _coe_stageA_summary
            if _coe_stageA_log:
                stageA_data["coe_stageA_dry_run_log"] = _coe_stageA_log
            print("\n" + _format_coe_stageA_dry_run_report(_coe_stageA_summary))
        if str(getattr(args, "coe_mode", "off") or "off") != "off":
            try:
                from nestynet_sr.sr_search.coe_committee import (
                    build_stageA_proposal_reservoir,
                    merge_stageA_proposal_reservoir_payloads,
                )

                _stageA_reservoir_max = max(
                    1,
                    int(getattr(args, "coe_max_candidates", 16) or 16) * 4,
                )
                _stageA_reference_reservoir = build_stageA_proposal_reservoir(
                    stageA_data=stageA_data,
                    max_candidates=_stageA_reservoir_max,
                    source="stageA_reference_run",
                )
                _stageA_reservoir_inputs = [_stageA_reference_reservoir]
                if isinstance(coe_stageA_external_reservoir, dict):
                    _stageA_reservoir_inputs.append(coe_stageA_external_reservoir)
                if isinstance(coe_stageA_pre_scout_reservoir, dict):
                    _stageA_reservoir_inputs.append(coe_stageA_pre_scout_reservoir)
                _active_replay_reservoir = getattr(
                    search_hp,
                    "coe_stageA_replay_reservoir",
                    None,
                )
                if isinstance(_active_replay_reservoir, dict):
                    _stageA_reservoir_inputs.append(_active_replay_reservoir)
                if len(_stageA_reservoir_inputs) > 1:
                    _stageA_reference_reservoir = merge_stageA_proposal_reservoir_payloads(
                        _stageA_reservoir_inputs,
                        max_candidates=_stageA_reservoir_max,
                    )
                    _stageA_reference_reservoir["source"] = (
                        "reference_refine_imported_stageA_proposal_reservoir"
                    )
                stageA_data["coe_stageA_proposal_reservoir"] = _make_json_serializable(
                    _stageA_reference_reservoir
                )
            except Exception as e_stagea_reservoir:
                stageA_data["coe_stageA_proposal_reservoir"] = {
                    "enabled": True,
                    "kind": "stageA_proposal_reservoir",
                    "status": "error",
                    "error": str(e_stagea_reservoir),
                }

    # Prepare Stage B data
    stageB_data = None
    if args.stageB and stageB_state is not None:
        try:
            setattr(stageB_state, "_stageB_portfolio_y_name", locals().get("stageB_y_op_name", None))
        except Exception:
            pass
        try:
            _stageB_final_metrics = _stageB_candidate_metrics(
                stageB_state,
                y_name=locals().get("stageB_y_op_name", None),
            )
        except Exception:
            _stageB_final_metrics = {}
        stageB_data = {
            "ast": stageB_state.root,
            "val_loss": stageB_state.val_loss,
            "val_losses": getattr(stageB_state, "val_losses", None),
            "dataset_ids": getattr(stageB_state, "dataset_ids", None),
            "dataset_metadata": param_sr_dataset_metadata,
            "agg_mode": getattr(stageB_state, "agg_mode", None),
            "params": stageB_state.model.num_parameters()
            if hasattr(stageB_state.model, "num_parameters")
            else None,
            "num_nn_atoms": getattr(stageB_state, "num_nn_atoms", None),
            "num_multivar_nn_atoms": getattr(stageB_state, "num_multivar_nn_atoms", None),
            "max_nn_arity": getattr(stageB_state, "max_nn_arity", None),
            "phi_expr_str": stageB_state.phi_expr_str,
            "phi_expr_strs": locals().get("phi_expr_strs", None),
            "y_expr_str": stageB_state.y_expr_str,
            "phi_expr_raw_str": getattr(stageB_state, "phi_expr_raw_str", None),
            "y_expr_raw_str": getattr(stageB_state, "y_expr_raw_str", None),
            "sympy_meta": stageB_state.sympy_meta,
            "coefficient_metadata": getattr(
                stageB_state, "coefficient_metadata", None
            ),
            "coefficient_metadata_by_dataset": getattr(
                stageB_state, "coefficient_metadata_by_dataset", None
            ),
            "enabled_patterns": stageB_state.enabled_patterns
            if hasattr(stageB_state, "enabled_patterns")
            else [],
            "y_shortlist_names": list(locals().get("stageB_portfolio_names", []) or []),
            "y_shortlist_sources": dict(stageB_y_shortlist_sources or {}),
            "y_selected": locals().get("stageB_y_op_name", None),
            "y_branch_artifacts": _make_json_serializable(
                list(locals().get("stageB_y_branch_artifacts", []) or [])
            ),
            "original_y_val_loss": _stageB_final_metrics.get("original_y_val_loss"),
            "candidate_metrics": _stageB_final_metrics,
        }
        _coe_stageB_log = list(getattr(stageB_state, "coe_stageB_dry_run_log", None) or [])
        _coe_stageB_gate_log = list(getattr(stageB_state, "coe_stageB_gate_log", None) or [])
        _coe_stageB_enabled = bool(
            getattr(args, "coe_stageB_dry_run", False)
            or str(getattr(args, "coe_mode", "off") or "off") in {"committee_gated", "reservoir_discovery"}
        )
        if _coe_stageB_enabled:
            _coe_stageB_summary = _summarize_coe_stageB_dry_run(
                _coe_stageB_log,
                enabled=True,
            )
            _coe_stageB_jsonl = _write_coe_stageB_dry_run_jsonl(
                os.path.join(results_dir, f"{base_filename}.coe_stageB_dry_run.jsonl"),
                _coe_stageB_log,
            )
            if _coe_stageB_jsonl:
                _coe_stageB_summary["jsonl_path"] = _coe_stageB_jsonl
            stageB_data["coe_stageB_dry_run_summary"] = _coe_stageB_summary
            print("\n" + _format_coe_stageB_dry_run_report(_coe_stageB_summary))
        if _coe_stageB_log:
            stageB_data["coe_stageB_dry_run_log"] = _coe_stageB_log
        if str(getattr(args, "coe_mode", "off") or "off") in {"committee_gated", "reservoir_discovery"}:
            _coe_mode_name = str(getattr(args, "coe_mode", "off") or "off")
            _coe_stageB_gate_summary = _summarize_coe_stageB_gate(
                _coe_stageB_gate_log,
                enabled=True,
            )
            _coe_stageB_gate_summary["mode"] = _coe_mode_name
            _coe_stageB_gate_jsonl = _write_coe_stageB_gate_jsonl(
                os.path.join(results_dir, f"{base_filename}.coe_stageB_gate.jsonl"),
                _coe_stageB_gate_log,
            )
            if _coe_stageB_gate_jsonl:
                _coe_stageB_gate_summary["jsonl_path"] = _coe_stageB_gate_jsonl
            stageB_data["coe_stageB_gate_summary"] = _coe_stageB_gate_summary
            print("\n" + _format_coe_stageB_gate_report(_coe_stageB_gate_summary))
        if _coe_stageB_gate_log:
            stageB_data["coe_stageB_gate_log"] = _coe_stageB_gate_log

        # Build simplification path: prepend pure NN (step 0) and Stage A (step 1)
        _simp_path = list(getattr(stageB_state, "simplification_path", []))
        _prepend_steps = []
        if stageA_data is not None:
            _phase_direct_label = stageA_data.get("phase_prescan_direct_closure")
            _phase_direct = (
                stageA_data.get("stageA_status") == "phase_hint_confirmed"
                and _phase_direct_label
            )
            # Step 0: pure NN model before separability/compound detection
            _initial_ast = stageA_data.get("initial_ast")
            _nn_val_loss = stageA_data.get("nn_val_loss")
            _nn_n_params = stageA_data.get("nn_n_params")
            if _phase_direct:
                _stageA_ast = stageA_data.get("ast")
                _stageA_val_loss = float(stageA_data.get("val_loss", _nn_val_loss))
                _prepend_steps.append(
                    {
                        "step": 0,
                        "stage": "0",
                        "action": "phase-coordinate direct closure",
                        "expression": ast_to_human_readable(_stageA_ast) if _stageA_ast else "?",
                        "val_loss": _stageA_val_loss,
                        "mse_raw": _stageA_val_loss,
                        "mse_eff": None,
                        "base_loss": None,
                        "threshold": None,
                        "n_params": _nn_n_params,
                        "ast_cost": None,
                        "detail": f"pattern={_phase_direct_label}, y_op={stageA_data.get('y_op_name', 'identity')}",
                    }
                )
            else:
                _nn_step = {
                    "step": 0,
                    "stage": "A",
                    "action": "pure NN surrogate",
                    "expression": ast_to_human_readable(_initial_ast) if _initial_ast else "?",
                    "val_loss": float(_nn_val_loss) if _nn_val_loss is not None else None,
                    "mse_raw": float(_nn_val_loss) if _nn_val_loss is not None else None,
                    "mse_eff": None,
                    "base_loss": None,
                    "threshold": None,
                    "n_params": _nn_n_params,
                    "ast_cost": None,
                    "detail": f"y_op={stageA_data.get('y_op_name', 'identity')}",
                }
                _prepend_steps.append(_nn_step)

            # Step 1: Stage A after separability/compound detection (only if AST changed)
            _stageA_ast = stageA_data.get("ast")
            _stageA_val_loss = float(stageA_data.get("val_loss", float("nan")))
            _stageA_changed = (
                _initial_ast is not None
                and _stageA_ast is not None
                and ast_to_human_readable(_initial_ast) != ast_to_human_readable(_stageA_ast)
            )
            if _stageA_changed and not _phase_direct:
                _stageA_action = "separability/compound detection"
                stageA_step = {
                    "step": 1,
                    "stage": "A",
                    "action": _stageA_action,
                    "expression": ast_to_human_readable(_stageA_ast),
                    "val_loss": _stageA_val_loss,
                    "mse_raw": _stageA_val_loss,
                    "mse_eff": None,
                    "base_loss": float(_nn_val_loss) if _nn_val_loss is not None else None,
                    "threshold": None,
                    "n_params": int(final_model.num_parameters()) if final_model and hasattr(final_model, "num_parameters") else None,
                    "ast_cost": None,
                    "detail": f"y_op={stageA_data.get('y_op_name', 'identity')}",
                }
                _prepend_steps.append(stageA_step)

        # Renumber existing steps and prepend
        _offset = len(_prepend_steps)
        for s in _simp_path:
            s["step"] = s["step"] + _offset
        _simp_path = _prepend_steps + _simp_path
        _simp_path = _decorate_simplification_path_y_space(
            _simp_path,
            y_transform_name=stageB_data.get("y_selected"),
            phi_expr_str=stageB_data.get("phi_expr_raw_str") or stageB_data.get("phi_expr_str"),
            y_expr_str=stageB_data.get("y_expr_raw_str") or stageB_data.get("y_expr_str"),
            original_y_val_loss=stageB_data.get("original_y_val_loss"),
        )
        _stageB_x_transform_for_path = stageB_data.get("x_transform_map")
        if _stageB_x_transform_for_path is None and stageA_data is not None:
            _stageB_x_transform_for_path = stageA_data.get("x_transform_map")
        _stageB_ast_human_for_path = (
            ast_to_human_readable(stageB_data.get("ast"), _stageB_x_transform_for_path)
            if stageB_data.get("ast") is not None
            else None
        )
        if _stageB_ast_human_for_path:
            stageB_data["ast_human"] = _stageB_ast_human_for_path
        try:
            _stageB_num_nn_atoms_for_path = int(stageB_data.get("num_nn_atoms") or 0)
        except Exception:
            _stageB_num_nn_atoms_for_path = 0
        if _stageB_num_nn_atoms_for_path > 0:
            _simp_path = _append_final_simplification_path_state(
                _simp_path,
                final_expr=_stageB_ast_human_for_path,
                val_loss=stageB_data.get("val_loss"),
                n_params=stageB_data.get("params"),
                num_nn_atoms=_stageB_num_nn_atoms_for_path,
            )
        stageB_data["simplification_path"] = _simp_path

        # Add decision log summary for the report
        _dlog = getattr(stageB_state, "decision_log", None) or []
        if _dlog:
            from collections import Counter
            from nestynet_sr.sr_search.model_selection import pareto_front_indices_2d
            _counts = Counter(d.get("outcome", "unknown") for d in _dlog)
            _trackable = []
            for _rec in _dlog:
                if not bool(_rec.get("pareto_trackable", False)):
                    continue
                try:
                    _loss = float(_rec.get("cand_loss"))
                    _cx = float(_rec.get("cand_complexity_total"))
                except Exception:
                    continue
                if not (np.isfinite(_loss) and np.isfinite(_cx)):
                    continue
                _trackable.append((_loss, _cx))
            _pareto_n = 0
            if _trackable:
                try:
                    _pareto_n = len(pareto_front_indices_2d(_trackable))
                except Exception:
                    _pareto_n = 0
            stageB_data["decision_log_summary"] = {
                "total": len(_dlog),
                "accept": _counts.get("accept", 0),
                "reject": _counts.get("reject", 0),
                "precheck_reject": _counts.get("precheck_reject", 0),
                "nonfinite_reject": _counts.get("nonfinite_reject", 0),
                "dedup_skip": _counts.get("dedup_skip", 0),
                "pareto_trackable": len(_trackable),
                "pareto_front": int(_pareto_n),
                "coordinate_variant_accepts": dict(
                    Counter(
                        str(d.get("coordinate_variant_display"))
                        for d in _dlog
                        if d.get("outcome") == "accept"
                        and d.get("coordinate_variant_display")
                    )
                ),
            }
        _coe_mode_for_reservoir = str(getattr(args, "coe_mode", "off") or "off")
        _coe_explicit_reservoir_paths = bool(
            str(getattr(args, "coe_reservoir_paths", "") or "").strip()
        )
        _stat_reservoir_enabled = bool(getattr(args, "stat_selection", False))
        if _stat_reservoir_enabled or (
            _coe_mode_for_reservoir != "off"
            and (
                _coe_mode_for_reservoir == "reservoir_discovery"
                or _coe_explicit_reservoir_paths
            )
        ):
            try:
                from nestynet_sr.sr_search.coe_committee import (
                    build_stageB_proposal_reservoir,
                    load_proposal_reservoir_payloads,
                    merge_proposal_reservoir_payloads,
                    stageA_terminal_proposals_as_expression_reservoir,
                    split_reservoir_path_string,
                )

                _reservoir_max = max(
                    1,
                    int(getattr(args, "coe_max_candidates", 16) or 16) * 2,
                    int(getattr(args, "stat_max_candidates", 1024) or 1024)
                    if _stat_reservoir_enabled
                    else 1,
                )
                _reservoir_payload = build_stageB_proposal_reservoir(
                    decision_log=_dlog,
                    simplification_path=stageB_data.get("simplification_path"),
                    stageB_data=stageB_data,
                    max_candidates=_reservoir_max,
                )
                _external_paths = split_reservoir_path_string(
                    getattr(args, "coe_reservoir_paths", None)
                )
                _reservoir_warnings = []
                if _external_paths:
                    _problem_stem = _coe_problem_stem(source_filepath)
                    _external_payloads, _reservoir_warnings = load_proposal_reservoir_payloads(
                        _external_paths,
                        problem_stem=_problem_stem,
                    )
                    if _external_payloads:
                        _reservoir_payload = merge_proposal_reservoir_payloads(
                            [_reservoir_payload] + list(_external_payloads),
                            max_candidates=_reservoir_max,
                        )
                        _reservoir_payload["external_sources"] = list(_external_paths)
                        _reservoir_payload["external_payload_count"] = len(_external_payloads)
                _stageA_terminal_source = (
                    stageA_data.get("coe_stageA_proposal_reservoir")
                    if isinstance(stageA_data, dict)
                    else None
                )
                if not isinstance(_stageA_terminal_source, dict):
                    _stageA_terminal_source = coe_stageA_external_reservoir
                if isinstance(_stageA_terminal_source, dict):
                    _stageA_terminal_exprs = stageA_terminal_proposals_as_expression_reservoir(
                        _stageA_terminal_source,
                        max_candidates=_reservoir_max,
                    )
                    if _stageA_terminal_exprs.get("candidates"):
                        _reservoir_payload = merge_proposal_reservoir_payloads(
                            [_reservoir_payload, _stageA_terminal_exprs],
                            max_candidates=_reservoir_max,
                        )
                        _reservoir_payload["external_stageA_terminal_payload_count"] = len(
                            list(_stageA_terminal_exprs.get("candidates") or [])
                        )
                if isinstance(coe_scout_expression_reservoir, dict):
                    _reservoir_payload = merge_proposal_reservoir_payloads(
                        [_reservoir_payload, coe_scout_expression_reservoir],
                        max_candidates=_reservoir_max,
                    )
                    _reservoir_payload["scout_expression_payload_count"] = len(
                        list(coe_scout_expression_reservoir.get("candidates") or [])
                    )
                if _reservoir_warnings:
                    _reservoir_payload["warnings"] = list(_reservoir_warnings)
                stageB_data["coe_proposal_reservoir"] = _make_json_serializable(
                    _reservoir_payload
                )
            except Exception as e_reservoir:
                stageB_data["coe_proposal_reservoir"] = {
                    "enabled": True,
                    "status": "error",
                    "error": str(e_reservoir),
                }
        if isinstance(coe_stageA_pre_scout_result, dict):
            _scout_result = {"summary": coe_stageA_pre_scout_summary}
        else:
            _scout_result = _run_coe_scout_proposers(
                args=args,
                filepath=filepath,
                results_dir=results_dir,
                base_filename=base_filename,
                current_stageA_reservoir=(
                    stageA_data.get("coe_stageA_proposal_reservoir")
                    if isinstance(stageA_data, dict)
                    else None
                ),
                current_reservoir=stageB_data.get("coe_proposal_reservoir"),
            )
        if isinstance(_scout_result, dict):
            _scout_summary = _scout_result.get("summary")
            if isinstance(_scout_summary, dict):
                stageB_data["coe_scout_proposers"] = _make_json_serializable(_scout_summary)
                if isinstance(stageA_data, dict):
                    stageA_data["coe_stageA_scout_proposers"] = _make_json_serializable(_scout_summary)
                print(
                    "[CoE scouts] completed="
                    f"{_scout_summary.get('completed', 0)} "
                    f"loaded_payloads={_scout_summary.get('loaded_payloads', 0)} "
                    f"loaded_stageA_payloads={_scout_summary.get('loaded_stageA_payloads', 0)}"
                )
            _merged_reservoir = _scout_result.get("merged_reservoir")
            if isinstance(_merged_reservoir, dict):
                stageB_data["coe_proposal_reservoir"] = _make_json_serializable(
                    _merged_reservoir
                )
            _merged_stageA_reservoir = _scout_result.get("merged_stageA_reservoir")
            if isinstance(_merged_stageA_reservoir, dict) and isinstance(stageA_data, dict):
                stageA_data["coe_stageA_proposal_reservoir"] = _make_json_serializable(
                    _merged_stageA_reservoir
                )

    _provisional_guard = None
    if isinstance(stageA_data, dict) and stageA_data.get("stageA_provisional_commits"):
        _provisional_summary = _stageA_provisional_confirmation_summary(stageA_data, stageB_data)
        _provisional_jsonl = _write_stageA_provisional_confirmation_jsonl(
            os.path.join(results_dir, f"{base_filename}.stageA_provisional.jsonl"),
            _provisional_summary,
        )
        if _provisional_jsonl:
            _provisional_summary["jsonl_path"] = _provisional_jsonl
        stageA_data["stageA_provisional_confirmation"] = _make_json_serializable(_provisional_summary)
        _annotated_commits = (
            _provisional_summary.get("commits")
            if isinstance(_provisional_summary, dict)
            else None
        )
        if isinstance(_annotated_commits, list):
            stageA_data["stageA_provisional_commits"] = _make_json_serializable(_annotated_commits)
            _by_seq = {
                row.get("seq"): row
                for row in _annotated_commits
                if isinstance(row, dict) and row.get("seq") is not None
            }
            _moves = []
            for _move in list(stageA_data.get("stageA_move_records") or []):
                if isinstance(_move, dict) and _move.get("seq") in _by_seq:
                    _merged = dict(_move)
                    _merged.update(
                        {
                            "confirmation_status": _by_seq[_move.get("seq")].get("confirmation_status"),
                            "confirmation_reason": _by_seq[_move.get("seq")].get("confirmation_reason"),
                            "confirmed_by_downstream": _by_seq[_move.get("seq")].get(
                                "confirmed_by_downstream"
                            ),
                        }
                    )
                    _moves.append(_merged)
                else:
                    _moves.append(_move)
            stageA_data["stageA_move_records"] = _make_json_serializable(_moves)
        print(
            "[Stage A provisional] "
            f"{_provisional_summary.get('confirmed', 0)}/"
            f"{_provisional_summary.get('total', 0)} confirmed "
            f"({_provisional_summary.get('status', 'unknown')})"
        )
        _provisional_guard = _apply_stageA_provisional_guard(
            args=args,
            stageA_data=stageA_data,
            stageB_data=stageB_data,
        )
        if _provisional_guard.get("decision") == "mark_uncertified":
            print(f"{YELLOW}[Stage A provisional] {_provisional_guard.get('reason')}{RESET}")

    # Optional: discover an implicit DE residual and treat it as a first-class SR output.
    de_data = None
    if getattr(args, "discover_de", False):
        try:
            de_data = _run_firstclass_de_for_sr(
                args=args,
                filepaths=list(filepaths),
                base_filename=base_filename,
                results_dir=results_dir,
                Nxvars=Nxvars,
                np_dtype=np_dtype,
                data_hp=data_hp,
                lm_hp=lm_hp,
                leaf_builder=leaf_builder,
                device=device,
                dtype=dtype,
                final_model=final_model,
                final_ast=final_ast,
                final_y_op=final_y_op,
                final_y_op_inv=final_y_op_inv,
                final_y_op_name=final_y_op_name,
                model_output_identity=model_output_identity,
                model_sep_output_identity=model_sep_output_identity,
                units_payload=locals().get("units_payload", None),
                factorized_search_hp=factorized_search_hp,
                disabled_patterns=disabled_patterns,
            )
        except Exception as e:
            print(f"[DE/SR] Warning: DE discovery failed: {e}")
            traceback.print_exc()
            de_data = {"enabled": False, "error": str(e)}

    discovery_payload = None
    discovery_summary = None
    if bool(args.discovery_enable):
        from nestynet_sr.discovery.integration import (
            discovery_summary_from_payload,
            run_sr_discovery_integration,
        )

        discovery_payload = run_sr_discovery_integration(
            filepath=str(filepath),
            filepaths=list(filepaths),
            report_path=str(report_path),
            stageA_data=stageA_data,
            stageB_data=stageB_data,
            final_model=final_model,
            final_y_op_inv=final_y_op_inv,
            final_y_op_name=str(final_y_op_name or "identity"),
            stageB_state=stageB_state,
            class_sr_result=class_sr_result,
            units_payload=locals().get("units_payload", None),
            committee_topk=max(1, int(args.discovery_topk)),
            max_members=None if args.discovery_max_members is None else int(args.discovery_max_members),
            experiment_manifest_path=args.discovery_experiment_manifest,
            research_profile=None if args.discovery_research_profile is None else str(args.discovery_research_profile),
            beta=float(args.discovery_beta),
            gamma=float(args.discovery_gamma),
            disagreement_mode=None if args.discovery_disagreement_mode is None else str(args.discovery_disagreement_mode),
            lambda_cost=float(args.discovery_lambda_cost),
            lambda_noise=float(args.discovery_lambda_noise),
            lambda_feasibility=float(args.discovery_lambda_feasibility),
            nvars=int(Nxvars),
            dtype=dtype,
            discovery_constant_lift_enable=bool(args.discovery_constant_lift_enable),
            discovery_constant_lift_min_regimes=max(2, int(args.discovery_constant_lift_min_regimes)),
            discovery_constant_lift_trigger_mean_cv=float(args.discovery_constant_lift_trigger_mean_cv),
            discovery_constant_lift_apply_enable=bool(args.discovery_constant_lift_apply_enable),
            discovery_constant_lift_apply_topk=max(0, int(args.discovery_constant_lift_apply_topk)),
            discovery_constant_lift_min_rel_gain=float(args.discovery_constant_lift_min_rel_gain),
            witness_capture_enable=bool(args.discovery_witness_capture_enable),
            witness_hessian_diag_enable=bool(args.discovery_witness_hessian_diag_enable),
            diagnostic_set=str(args.discovery_diagnostic_set or "basic"),
            experiment_optimize_enable=bool(args.discovery_experiment_optimize_enable),
            experiment_opt_steps=max(1, int(args.discovery_experiment_opt_steps)),
            experiment_opt_lr=float(args.discovery_experiment_opt_lr),
            experiment_project_mode=str(args.discovery_experiment_project_mode or "nearest_box"),
            theory_benchmark_enable=bool(args.discovery_theory_benchmark_enable),
        )
        discovery_summary = discovery_summary_from_payload(
            discovery_payload,
            results_path=str(discovery_report_path),
        )
        discovery_path_obj = pathlib.Path(str(discovery_report_path))
        discovery_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(discovery_path_obj, "w") as f_discovery:
            json.dump(_make_json_serializable(discovery_payload), f_discovery, indent=2)
        print(f"\nWrote discovery report to {discovery_report_path}")
        selected_experiment = (
            dict(discovery_payload.get("experiment_selection", {}) or {}).get("selected", None)
            if isinstance(discovery_payload, dict)
            else None
        )
        if isinstance(selected_experiment, dict):
            print(
                "[Discovery] selected experiment: "
                f"{selected_experiment.get('experiment_id', '')} "
                f"(score={selected_experiment.get('score', None)})"
            )

    # Write the report
    write_json_report(
        filepath=source_filepath if statistical_split_plan is not None else filepath,
        filepaths=source_filepaths if statistical_split_plan is not None else filepaths,
        report_path=report_path,
        device=device,
        dtype=dtype,
        seed=iseed if iseed is not None else -1,
        walltime=elapsed_time,
        stageA_data=stageA_data,
        stageB_data=stageB_data,
        de_data=de_data,
        class_sr_summary=class_sr_summary,
        discovery_summary=discovery_summary,
        fast_mode=args.fast,
        disabled_patterns=disabled_patterns,
        factorized_search_enabled=bool(getattr(args, "use_factorized_search", False)),
        factorized_search_config=factorized_config_report(factorized_search_hp),
        enable_truth_eval=not bool(getattr(args, "blinded", False)),
    )
    if stageB_data is not None and not bool(getattr(args, "blinded", False)):
        _append_truth_eval_summary_to_file(
            report_path=report_path,
            output_path=os.path.join(results_dir, f"{base_filename}_final.human"),
            title="=== Pre-Polish Noiseless Ground Truth Check ===",
        )

    # Write simplification path file
    path_output = None
    if stageB_data is not None:
        _simp_path = stageB_data.get("simplification_path", [])
        if _simp_path:
            path_output = os.path.join(results_dir, f"{base_filename}.path")
            path_text = _format_simplification_path(_simp_path)
            try:
                with open(path_output, "w") as f_path:
                    f_path.write(path_text + "\n")
                print(f"Wrote simplification path to {path_output}")
            except Exception as e_path:
                print(f"Warning: could not write .path file: {e_path}")
            # Also print to console/log
            print("\n" + path_text)
            if stageB_data.get("coe_stageB_dry_run_summary"):
                coe_stageB_text = _format_coe_stageB_dry_run_report(
                    stageB_data["coe_stageB_dry_run_summary"]
                )
                try:
                    with open(path_output, "a") as f_path:
                        f_path.write("\n" + coe_stageB_text + "\n")
                except Exception as e_path:
                    print(f"Warning: could not append CoE Stage-B dry-run to .path file: {e_path}")
            if stageB_data.get("coe_stageB_gate_summary"):
                coe_stageB_gate_text = _format_coe_stageB_gate_report(
                    stageB_data["coe_stageB_gate_summary"]
                )
                try:
                    with open(path_output, "a") as f_path:
                        f_path.write("\n" + coe_stageB_gate_text + "\n")
                except Exception as e_path:
                    print(f"Warning: could not append CoE Stage-B gate to .path file: {e_path}")

    final_polish_summary = _run_final_pareto_polish(
        args=args,
        filepath=filepath,
        filepaths=filepaths,
        report_path=report_path,
        results_dir=results_dir,
        base_filename=base_filename,
        stageB_data=stageB_data,
        seed=iseed if iseed is not None else 1234,
        units_payload=units_payload,
        noise_sigma_y=noise_sigma_y,
    )
    if final_polish_summary:
        _update_report_with_final_polish(report_path, final_polish_summary)
        if bool(final_polish_summary.get("enabled", False)):
            polish_text = _format_final_polish_report(final_polish_summary)
            print("\n" + polish_text)
            if path_output is not None:
                try:
                    with open(path_output, "a") as f_path:
                        f_path.write("\n" + polish_text + "\n")
                except Exception as e_path:
                    print(f"Warning: could not append final polish to .path file: {e_path}")
            final_human_output = os.path.join(results_dir, f"{base_filename}_final.human")
            if os.path.exists(final_human_output):
                try:
                    with open(final_human_output, "a") as f_final:
                        f_final.write("\n" + polish_text + "\n")
                except Exception as e_final:
                    print(
                        f"Warning: could not append final polish to final.human: {e_final}"
                    )

    coe_summary = _run_coe_final_committee(
        args=args,
        filepath=filepath,
        report_path=report_path,
        results_dir=results_dir,
        base_filename=base_filename,
        stageB_data=stageB_data,
        final_polish_summary=final_polish_summary,
        noise_sigma_y=noise_sigma_y,
    )
    if coe_summary:
        if bool(getattr(args, "stat_selection", False)):
            coe_summary["authoritative"] = False
            coe_summary["authority_reason"] = (
                "search committee retained as a proposal/diagnostic layer; "
                "the untouched confidence Pareto audit is authoritative"
            )
            coe_final_selection = None
        else:
            coe_final_selection = _apply_coe_final_adjudication(report_path, coe_summary)
        _update_report_with_coe_committee(report_path, coe_summary)
        if bool(coe_summary.get("enabled", False)):
            coe_text = _format_coe_committee_report(coe_summary)
            if coe_final_selection is not None:
                coe_text += (
                    "\ncoe_final_selection_applied: "
                    + str(coe_final_selection.get("expr"))
                )
            print("\n" + coe_text)
            if path_output is not None:
                try:
                    with open(path_output, "a") as f_path:
                        f_path.write("\n" + coe_text + "\n")
                except Exception as e_path:
                    print(f"Warning: could not append CoE committee to .path file: {e_path}")
            final_human_output = os.path.join(results_dir, f"{base_filename}_final.human")
            if os.path.exists(final_human_output):
                try:
                    with open(final_human_output, "a") as f_final:
                        f_final.write("\n" + coe_text + "\n")
                except Exception as e_final:
                    print(
                        f"Warning: could not append CoE committee to final.human: {e_final}"
                    )

    statistical_selection_summary = None
    if statistical_split_plan is not None:
        _stat_output_dir = os.path.join(
            results_dir,
            f"{base_filename}_stat_selection_{statistical_split_plan.contract_fingerprint[:12]}",
        )
        _archive_path_arg = getattr(args, "stat_archive_json", None)
        _certificate_path_arg = getattr(args, "stat_certificate_json", None)
        _archive_filename = (
            os.path.abspath(os.path.expanduser(str(_archive_path_arg)))
            if _archive_path_arg
            else f"{base_filename}.sr_candidate_archive.json"
        )
        _certificate_filename = (
            os.path.abspath(os.path.expanduser(str(_certificate_path_arg)))
            if _certificate_path_arg
            else f"{base_filename}.sr_pareto_certificate.json"
        )
        try:
            _stat_noise_scale = (
                float(noise_sigma_y)
                if noise_sigma_y is not None
                and math.isfinite(float(noise_sigma_y))
                and float(noise_sigma_y) > 0.0
                else None
            )
            _stat_target_scale = (
                float(y_rms)
                if math.isfinite(float(y_rms)) and float(y_rms) > 0.0
                else 1.0
            )
            _stat_outcome = run_sr_statistical_selection(
                stageB_data=stageB_data,
                final_polish_summary=final_polish_summary,
                split_plan=statistical_split_plan,
                output_dir=_stat_output_dir,
                loss_scale=(
                    _stat_noise_scale
                    if _stat_noise_scale is not None
                    else _stat_target_scale
                ),
                loss_scale_name=(
                    "declared_search_noise_sigma_y"
                    if _stat_noise_scale is not None
                    else "search_target_rms"
                ),
                max_candidates=_stat_max_candidates,
                unit_size=_stat_unit_size,
                failure_loss=_stat_failure_loss,
                alpha=_stat_alpha,
                delta=_stat_delta,
                n_resamples=_stat_resamples,
                seed=_stat_seed,
                multiplier=_stat_multiplier,
                archive_filename=_archive_filename,
                certificate_filename=_certificate_filename,
                x_sigma=getattr(args, "stat_x_sigma", None),
                x_cov_npz=getattr(args, "stat_x_cov_npz", None),
                x_cov_sha256_expected=_stat_x_cov_sha256,
                x_error_loss=getattr(args, "stat_x_error_loss", "marginal_gaussian_nll"),
                x_gradient_step=float(getattr(args, "stat_x_gradient_step", 1.0e-5)),
            )
            statistical_selection_summary = update_report_with_sr_statistical_selection(
                report_path,
                _stat_outcome,
                split_plan=statistical_split_plan,
            )
            _blocked_selection = _enforce_stageA_provisional_guard_on_report(
                report_path,
                _provisional_guard,
            )
            if _blocked_selection is not None:
                statistical_selection_summary = dict(statistical_selection_summary)
                statistical_selection_summary["selection_blocked_by_stageA_provisional_guard"] = True
                statistical_selection_summary["eligible_for_success"] = False
                print(
                    f"{YELLOW}[Stage A provisional] Sealed audit completed, but its "
                    "selection is ineligible because Stage-A debt remains unconfirmed."
                    f"{RESET}"
                )
            elif not bool(getattr(args, "blinded", False)):
                _refresh_final_selection_truth_eval(
                    report_path,
                    source="statistical_selection",
                    preserve_as="truth_eval_pre_statistical_selection",
                    verbose=False,
                )
            _stat_text = format_sr_statistical_selection(statistical_selection_summary)
            if _stat_text:
                print("\n" + _stat_text)
                if path_output is not None:
                    with open(path_output, "a") as _f_path:
                        _f_path.write("\n" + _stat_text + "\n")
                _final_human = os.path.join(results_dir, f"{base_filename}_final.human")
                if os.path.exists(_final_human):
                    with open(_final_human, "a") as _f_final:
                        _f_final.write("\n" + _stat_text + "\n")
        except Exception as _stat_exc:
            try:
                with open(report_path, "r") as _f_report:
                    _report_payload = json.load(_f_report)
                _report_payload["statistical_selection"] = {
                    "enabled": True,
                    "status": "error",
                    "authority": "statistical_selection",
                    "error": str(_stat_exc),
                    "split_plan": statistical_split_plan.to_dict(),
                }
                _legacy = _report_payload.get("final_selection")
                if isinstance(_legacy, dict):
                    _report_payload["legacy_search_selection"] = _legacy
                _report_payload["final_selection"] = {
                    "source": "statistical_selection",
                    "applied": False,
                    "eligible_for_success": False,
                    "status": "audit_failed",
                    "reason": str(_stat_exc),
                    "expr": (_legacy or {}).get("expr") if isinstance(_legacy, dict) else None,
                }
                with open(report_path, "w") as _f_report:
                    json.dump(_report_payload, _f_report, indent=2)
            finally:
                if isinstance(_stat_exc, NoPortableAnalyticCandidatesError):
                    print(
                        "[StatSelection] Certification failed closed: no portable "
                        f"analytic SR candidates were available. Details: {report_path}",
                        file=sys.stderr,
                    )
                    raise SystemExit(1) from None
                raise RuntimeError(
                    "statistical selection was requested but certification failed closed"
                ) from _stat_exc

    _append_final_selection_report(
        report_path=report_path,
        path_output=path_output,
        final_human_output=os.path.join(results_dir, f"{base_filename}_final.human"),
    )

    if statistical_split_plan is None:
        _maybe_retry_without_visible_buckingham(
            report_path=report_path,
            results_dir=results_dir,
            base_filename=base_filename,
        )
    else:
        print(
            "[StatSelection] Buckingham structural retry disabled: reopening the "
            "sealed audit after an adaptive rerun would violate the one-shot audit contract."
        )
    _update_report_with_campaign_outcome(report_path)


if __name__ == "__main__":
    main()
