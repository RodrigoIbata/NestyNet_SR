#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Run the full SR pipeline (Stage A + Stage B + A<->B feedback loop) on AI Feynman problems.

This script:
1. Loads all AI Feynman canaries from aif_canaries.json
2. Finds corresponding CSV data files
3. Runs the full pipeline (Stage A, Stage B, and feedback loop) on each problem
4. Saves results and generates a summary report

Usage:
    python nestynet_sr/run_allstages_suite.py --only pb010
    python nestynet_sr/run_allstages_suite.py --fast --limit 5
    python nestynet_sr/run_allstages_suite.py --all
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Get script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add symbolic_regression to path
sys.path.insert(0, script_dir)


def load_canaries(canaries_path: str) -> Dict:
    """Load AI Feynman canary registry."""
    if not os.path.isabs(canaries_path):
        canaries_path = os.path.join(script_dir, canaries_path)

    try:
        with open(canaries_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Failed to load {canaries_path}: {e}")
        sys.exit(1)


def find_data_file(stem: str, data_dir: str) -> Optional[str]:
    """Find the CSV data file corresponding to a problem stem."""
    data_path = Path(data_dir).resolve()

    patterns = [
        f"{stem}_data.csv",
        f"{stem}.csv",
    ]

    import glob

    glob_pattern = str(data_path / f"{stem}_*.csv")
    matches = glob.glob(glob_pattern)

    if matches:
        data_files = [m for m in matches if m.endswith("_data.csv")]
        if data_files:
            return os.path.abspath(data_files[0])
        return os.path.abspath(matches[0])

    for pattern in patterns:
        filepath = data_path / pattern
        if filepath.exists():
            return str(filepath.resolve())

    return None


def run_allstages_on_problem(
    stem: str,
    filepath: str,
    results_dir: str,
    fast_mode: bool = False,
    stat_selection: bool = False,
    noise_sigma_frac_y_rms: Optional[float] = None,
    force_y_ops: Optional[str] = None,
    single_layer: bool = False,
    disable_compound_detection: bool = False,
    equations_txt: Optional[str] = None,
    ignore_units: bool = False,
    verbose: bool = False,
    verbose_separabilities: bool = False,
    bypass: bool = False,
    max_ab_iters: Optional[int] = None,
    stageB_max_outer_iters: Optional[int] = None,
    stageB_epochs: Optional[int] = None,
    max_backtracks: Optional[int] = None,
    use_factorized_search: Optional[bool] = None,
    use_refine_skeleton: Optional[bool] = None,
    disable_stageB_patterns: Optional[str] = None,
    stageB_overcap_fallback: bool = False,
    stageB_polish: Optional[bool] = None,
    stageB_polish_max_candidates: Optional[int] = None,
    stageB_polish_subtrees: Optional[bool] = None,
    stageB_polish_max_subtrees: Optional[int] = None,
    canonical_init: bool = False,
    evidence: bool = False,
    evidence_disable_residual_whitening: bool = False,
    evidence_disable_segment_priors: bool = False,
    evidence_lambda_patch: Optional[float] = None,
    evidence_prior_decay_start: Optional[int] = None,
    evidence_prior_decay_interval: Optional[int] = None,
    evidence_prior_decay_shape: Optional[str] = None,
    evidence_prior_decay_final_scale: Optional[float] = None,
    evidence_prior_cutoff_tol: Optional[float] = None,
    evidence_prior_decay_auto: bool = True,
    evidence_metric_gate: bool = True,
    ndata_train: Optional[int] = None,
    ndata_val: Optional[int] = None,
    batch_size: Optional[int] = None,
    data_slice: int = 0,
    coe_mode: str = "off",
    coe_num_slices: Optional[int] = None,
    coe_start_slice: Optional[int] = None,
    coe_max_candidates: Optional[int] = None,
    coe_reservoir_paths: Optional[str] = None,
    coe_noise_mult: Optional[float] = None,
    coe_rel_tol: Optional[float] = None,
    coe_min_valid_fraction: Optional[float] = None,
    coe_witness_parallelism: Optional[int] = None,
    coe_reservoir_support_bonus: Optional[float] = None,
    coe_stageB_dry_run: bool = False,
    coe_stageB_gate_slices: Optional[int] = None,
    coe_stageB_initial_gate_slices: Optional[int] = None,
    coe_stageB_refit_gate: bool = True,
    coe_stageB_refit_epochs: Optional[int] = None,
    coe_stageB_refit_escalate_epochs: Optional[int] = None,
    coe_stageA_dry_run: bool = False,
    coe_stageA_fit_tournament: bool = False,
    coe_stageA_fit_slices: Optional[str] = None,
    coe_stageA_fit_alpha: Optional[float] = None,
    coe_stageA_fit_comparison_fraction: Optional[float] = None,
    coe_stageA_fit_min_rel_improvement: Optional[float] = None,
    coe_scout_count: Optional[int] = None,
    coe_scout_slices: Optional[str] = None,
    coe_scout_timeout_seconds: Optional[float] = None,
    coe_scout_parallelism: Optional[int] = None,
    coe_scout_stageB_epochs: Optional[int] = None,
    coe_scout_stageB_max_outer_iters: Optional[int] = None,
    coe_scout_max_ab_iters: Optional[int] = None,
    coe_scout_stageA_max_passes: Optional[int] = None,
    coe_continuation_scouts: bool = True,
    coe_continuation_scout_count: Optional[int] = None,
    coe_continuation_scout_max_phases: Optional[int] = None,
    coe_scout_final_polish: bool = False,
    sr_extra_args: Optional[str] = None,
) -> Dict:
    """
    Run the full SR pipeline on a single AI Feynman problem.

    Parameters
    ----------
    stem : str
        Problem stem (e.g., "pb000")
    filepath : str
        Path to CSV data file
    results_dir : str
        Directory to save results
    fast_mode : bool
        Use fast mode (reduced epochs/segments)
    force_y_ops : str, optional
        Comma-separated list of y-transforms to try
    single_layer : bool
        Use single-layer architecture
    disable_compound_detection : bool
        Disable compound variable detection
    equations_txt : str, optional
        Path to equations.txt file for units information
    ignore_units : bool
        Disable dimensional consistency checking
    verbose : bool
        Print detailed output to console
    verbose_separabilities : bool
        Print detailed separability diagnostics
    bypass : bool
        Skip separability detection; train NN model only
    max_ab_iters : int, optional
        Max Stage A<->B feedback loop iterations
    stageB_max_outer_iters : int, optional
        Maximum number of Stage B refinement iterations
    stageB_epochs : int, optional
        Maximum LM epochs for Stage B fits
    max_backtracks : int, optional
        Max backtrack attempts in Stage B
    use_factorized_search : bool, optional
        Force enable/disable factorized symbolic search explorer in Stage B. None preserves run_SR.py default.
    use_refine_skeleton : bool, optional
        Force enable/disable continuous skeleton refinement inner refinement in Stage B. None preserves run_SR.py default.
    disable_stageB_patterns : str, optional
        Comma-separated list of Stage B pattern names to disable
    stageB_overcap_fallback : bool
        Enable the Stage B noisy-data over-cap fallback in run_SR.py
    stageB_polish : bool, optional
        Pass through accepted-step Stage B polishing toggle to run_SR.py
    stageB_polish_max_candidates : int, optional
        Maximum algebraic cleanup candidates for accepted-step Stage B polish
    stageB_polish_subtrees : bool, optional
        Pass through accepted-step subtree polishing toggle to run_SR.py
    stageB_polish_max_subtrees : int, optional
        Maximum newly analytic subtrees audited per accepted Stage B rewrite
    canonical_init : bool
        Request NestyNet canonical initialization for pure Stage-A NN teacher fits.
    evidence : bool
        Request SR evidence-mode LM construction in run_SR.py
    evidence_disable_residual_whitening : bool
        Disable evidence residual-whitening / patch terms
    evidence_disable_segment_priors : bool
        Disable evidence segment priors
    evidence_lambda_patch : float, optional
        Evidence patch-whitening weight passed through to run_SR.py
    evidence_prior_decay_start/end : int, optional
        Segment-prior decay iteration switches passed through to run_SR.py
    coe_mode : str
        Pass through to run_SR.py: CoE final committee mode.

    Returns
    -------
    dict
        Result with success status and timing information
    """
    result = {
        "stem": stem,
        "filepath": filepath,
        "success": False,
        "walltime_seconds": None,
        "error": None,
    }

    run_sr_path = os.path.join(script_dir, "run_SR.py")

    cmd = [
        sys.executable,
        "-u",
        run_sr_path,
        "--filepath",
        filepath,
        "--log_level",
        "INFO",
        "--results_dir",
        results_dir,
    ]

    if fast_mode:
        cmd.append("--fast")

    if force_y_ops is not None:
        cmd.extend(["--force_y_ops", force_y_ops])

    if stat_selection:
        cmd.append("--stat-selection")

    if noise_sigma_frac_y_rms is not None:
        cmd.extend(["--noise_sigma_frac_y_rms", str(noise_sigma_frac_y_rms)])

    if ndata_train is not None:
        cmd.extend(["--ndata_train", str(ndata_train)])

    if ndata_val is not None:
        cmd.extend(["--ndata_val", str(ndata_val)])

    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])

    if int(data_slice or 0) != 0:
        cmd.extend(["--data_slice", str(int(data_slice))])

    if single_layer:
        cmd.append("--single_layer")

    if disable_compound_detection:
        cmd.append("--disable_compound_detection")

    if equations_txt is not None:
        cmd.extend(["--equations_txt", equations_txt])

    if ignore_units:
        cmd.append("--ignore_units")

    if verbose_separabilities:
        cmd.append("--verbose_separabilities")

    if bypass:
        cmd.append("--no_stageA_separabilities")

    if max_ab_iters is not None:
        cmd.extend(["--max_ab_iters", str(max_ab_iters)])

    if stageB_max_outer_iters is not None:
        cmd.extend(["--stageB_max_outer_iters", str(stageB_max_outer_iters)])

    if stageB_epochs is not None:
        cmd.extend(["--stageB_epochs", str(stageB_epochs)])

    if max_backtracks is not None:
        cmd.extend(["--max_backtracks", str(max_backtracks)])

    if use_factorized_search is False and use_refine_skeleton is True:
        use_refine_skeleton = False

    if use_factorized_search is True:
        cmd.append("--factorized-search")
    elif use_factorized_search is False:
        cmd.append("--no-factorized-search")

    if use_refine_skeleton is True:
        cmd.append("--refine-skeleton")
    elif use_refine_skeleton is False:
        cmd.append("--no-refine-skeleton")

    if disable_stageB_patterns is not None:
        cmd.extend(["--disable_stageB_patterns", disable_stageB_patterns])

    if stageB_overcap_fallback:
        cmd.append("--stageB_overcap_fallback")

    if stageB_polish is True:
        cmd.append("--stageB_polish")
    elif stageB_polish is False:
        cmd.append("--no_stageB_polish")

    if stageB_polish_max_candidates is not None:
        cmd.extend(["--stageB_polish_max_candidates", str(stageB_polish_max_candidates)])

    if stageB_polish_subtrees is True:
        cmd.append("--stageB_polish_subtrees")
    elif stageB_polish_subtrees is False:
        cmd.append("--no_stageB_polish_subtrees")

    if stageB_polish_max_subtrees is not None:
        cmd.extend(["--stageB_polish_max_subtrees", str(stageB_polish_max_subtrees)])

    if canonical_init:
        cmd.append("--canonical_init")

    if evidence:
        cmd.append("--evidence")

    if evidence_disable_residual_whitening:
        cmd.append("--evidence_disable_residual_whitening")

    if evidence_disable_segment_priors:
        cmd.append("--evidence_disable_segment_priors")

    if evidence_lambda_patch is not None:
        cmd.extend(["--evidence_lambda_patch", str(evidence_lambda_patch)])

    if evidence_prior_decay_start is not None:
        cmd.extend(["--evidence_prior_decay_start", str(evidence_prior_decay_start)])

    if evidence_prior_decay_interval is not None:
        cmd.extend(["--evidence_prior_decay_interval", str(evidence_prior_decay_interval)])

    if evidence_prior_decay_shape is not None:
        cmd.extend(["--evidence_prior_decay_shape", str(evidence_prior_decay_shape)])

    if evidence_prior_decay_final_scale is not None:
        cmd.extend(["--evidence_prior_decay_final_scale", str(evidence_prior_decay_final_scale)])

    if evidence_prior_cutoff_tol is not None:
        cmd.extend(["--evidence_prior_cutoff_tol", str(evidence_prior_cutoff_tol)])

    if not evidence_prior_decay_auto:
        cmd.append("--no_evidence_prior_decay_auto")

    if not evidence_metric_gate:
        cmd.append("--no_evidence_metric_gate")

    if coe_mode and coe_mode != "off":
        cmd.extend(["--coe_mode", str(coe_mode)])

    if coe_num_slices is not None:
        cmd.extend(["--coe_num_slices", str(coe_num_slices)])

    if coe_start_slice is not None:
        cmd.extend(["--coe_start_slice", str(coe_start_slice)])

    if coe_max_candidates is not None:
        cmd.extend(["--coe_max_candidates", str(coe_max_candidates)])

    if coe_reservoir_paths:
        cmd.extend(["--coe_reservoir_paths", str(coe_reservoir_paths)])

    if coe_noise_mult is not None:
        cmd.extend(["--coe_noise_mult", str(coe_noise_mult)])

    if coe_rel_tol is not None:
        cmd.extend(["--coe_rel_tol", str(coe_rel_tol)])

    if coe_min_valid_fraction is not None:
        cmd.extend(["--coe_min_valid_fraction", str(coe_min_valid_fraction)])
    if coe_witness_parallelism is not None:
        cmd.extend(["--coe_witness_parallelism", str(coe_witness_parallelism)])
    if coe_reservoir_support_bonus is not None:
        cmd.extend(["--coe_reservoir_support_bonus", str(coe_reservoir_support_bonus)])
    if coe_stageB_dry_run:
        cmd.append("--coe_stageB_dry_run")
    if coe_stageB_gate_slices is not None:
        cmd.extend(["--coe_stageB_gate_slices", str(coe_stageB_gate_slices)])
    if coe_stageB_initial_gate_slices is not None:
        cmd.extend(["--coe_stageB_initial_gate_slices", str(coe_stageB_initial_gate_slices)])
    if not coe_stageB_refit_gate:
        cmd.append("--no_coe_stageB_refit_gate")
    if coe_stageB_refit_epochs is not None:
        cmd.extend(["--coe_stageB_refit_epochs", str(coe_stageB_refit_epochs)])
    if coe_stageB_refit_escalate_epochs is not None:
        cmd.extend(
            [
                "--coe_stageB_refit_escalate_epochs",
                str(coe_stageB_refit_escalate_epochs),
            ]
        )
    if coe_stageA_dry_run:
        cmd.append("--coe_stageA_dry_run")
    if coe_stageA_fit_tournament:
        cmd.append("--coe_stageA_fit_tournament")
    if coe_stageA_fit_slices:
        cmd.extend(["--coe_stageA_fit_slices", str(coe_stageA_fit_slices)])
    if coe_stageA_fit_alpha is not None:
        cmd.extend(["--coe_stageA_fit_alpha", str(coe_stageA_fit_alpha)])
    if coe_stageA_fit_comparison_fraction is not None:
        cmd.extend(
            ["--coe_stageA_fit_comparison_fraction", str(coe_stageA_fit_comparison_fraction)]
        )
    if coe_stageA_fit_min_rel_improvement is not None:
        cmd.extend(
            ["--coe_stageA_fit_min_rel_improvement", str(coe_stageA_fit_min_rel_improvement)]
        )
    if coe_scout_count is not None:
        cmd.extend(["--coe_scout_count", str(coe_scout_count)])
    if coe_scout_slices:
        cmd.extend(["--coe_scout_slices", str(coe_scout_slices)])
    if coe_scout_timeout_seconds is not None:
        cmd.extend(["--coe_scout_timeout_seconds", str(coe_scout_timeout_seconds)])
    if coe_scout_parallelism is not None:
        cmd.extend(["--coe_scout_parallelism", str(coe_scout_parallelism)])
    if coe_scout_stageB_epochs is not None:
        cmd.extend(["--coe_scout_stageB_epochs", str(coe_scout_stageB_epochs)])
    if coe_scout_stageB_max_outer_iters is not None:
        cmd.extend(["--coe_scout_stageB_max_outer_iters", str(coe_scout_stageB_max_outer_iters)])
    if coe_scout_max_ab_iters is not None:
        cmd.extend(["--coe_scout_max_ab_iters", str(coe_scout_max_ab_iters)])
    if coe_scout_stageA_max_passes is not None:
        cmd.extend(["--coe_scout_stageA_max_passes", str(coe_scout_stageA_max_passes)])
    if not coe_continuation_scouts:
        cmd.append("--no_coe_continuation_scouts")
    if coe_continuation_scout_count is not None:
        cmd.extend(["--coe_continuation_scout_count", str(coe_continuation_scout_count)])
    if coe_continuation_scout_max_phases is not None:
        cmd.extend(["--coe_continuation_scout_max_phases", str(coe_continuation_scout_max_phases)])
    if coe_scout_final_polish:
        cmd.append("--coe_scout_final_polish")

    if sr_extra_args:
        # Generic pass-through for run_SR.py flags the suite does not model
        # explicitly (e.g. the --gs-* generalized-symmetry surface).
        import shlex

        cmd.extend(shlex.split(str(sr_extra_args)))

    data_stem = Path(filepath).stem
    log_path = os.path.join(results_dir, f"{data_stem}_allstages.log")
    result["log_file"] = log_path

    print(f"\n{'=' * 60}")
    print(f"Running full pipeline on {stem}")
    print(f"Data: {filepath}")
    print(f"Log: {log_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        with open(log_path, "w") as log_file:
            if verbose:
                process = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    print(line, end="")

                process.wait()
            else:
                process = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                process.wait()

        elapsed = time.time() - start_time

        if process.returncode != 0:
            result["success"] = False
            result["walltime_seconds"] = elapsed
            result["error"] = f"Exit code {process.returncode}"

            print(
                f"✗ {stem} failed after {elapsed / 60:.2f} minutes (exit code {process.returncode})"
            )
            print(f"  Error details logged to: {log_path}")

            with open(log_path, "r") as f:
                lines = f.readlines()
                if lines:
                    print("\nLast output lines:")
                    print("-" * 60)
                    for line in lines[-20:]:
                        print(line.rstrip())
                    print("-" * 60)
        else:
            result["success"] = True
            result["walltime_seconds"] = elapsed

            print(f"✓ {stem} completed in {elapsed / 60:.2f} minutes")
            print(f"  Log saved to: {log_path}")

    except Exception as e:
        elapsed = time.time() - start_time
        result["walltime_seconds"] = elapsed
        result["error"] = str(e)

        print(f"✗ {stem} crashed: {e}")
        if os.path.exists(log_path):
            print(f"  Partial log saved to: {log_path}")
        else:
            print("  (No log file created)")

    return result


def write_summary_report(
    results: List[Dict],
    output_path: str = "results/allstages_suite_summary.json",
):
    """Write summary report of full-pipeline suite run."""
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful

    total_time = sum(r["walltime_seconds"] or 0 for r in results)

    summary = {
        "total_problems": total,
        "successful": successful,
        "failed": failed,
        "total_walltime_hours": total_time / 3600,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ALL-STAGES SUITE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total problems:     {total:4d}")
    print(f"Successful:         {successful:4d} ({100 * successful / total:.1f}%)")
    print(f"Failed:             {failed:4d} ({100 * failed / total:.1f}%)")
    print(f"Total walltime:     {total_time / 3600:.2f} hours")
    if successful > 0:
        avg_time = (total_time / successful) / 60
        print(f"Avg time/problem:   {avg_time:.2f} minutes")
    print(f"\nSummary saved to: {output_path}")
    print(f"{'=' * 60}\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(os.path.dirname(script_dir), "results")

    parser = argparse.ArgumentParser(
        description="Run full SR pipeline (Stage A + B + feedback loop) on AI Feynman problems",
        epilog="Example: python nestynet_sr/run_allstages_suite.py --fast --limit 5",
    )
    default_canaries = os.path.join(script_dir, "aif_canaries.json")
    parser.add_argument(
        "--canaries",
        type=str,
        default=default_canaries,
        help="Path to canaries JSON file (default: nestynet_sr/aif_canaries.json)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing CSV data files (default: data/)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=default_results_dir,
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save summary JSON (default: results/allstages_suite_summary.json)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode (reduced epochs/segments for quick testing)",
    )
    parser.add_argument(
        "--force_y_ops",
        type=str,
        default=None,
        help="Force specific y-transforms (comma-separated, e.g., 'identity,square')",
    )
    parser.add_argument(
        "--canonical_init",
        action="store_true",
        help="Pass through to run_SR.py: apply NestyNet canonical initialization to pure Stage-A NN teachers.",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Pass through to run_SR.py: request SR evidence-mode LM construction.",
    )
    parser.add_argument(
        "--evidence_disable_residual_whitening",
        action="store_true",
        help="Pass through to run_SR.py: disable evidence residual-whitening / patch terms.",
    )
    parser.add_argument(
        "--evidence_disable_segment_priors",
        action="store_true",
        help="Pass through to run_SR.py: disable evidence segment priors.",
    )
    parser.add_argument(
        "--evidence_lambda_patch",
        type=float,
        default=None,
        help="Pass through to run_SR.py: evidence patch-whitening weight.",
    )
    parser.add_argument(
        "--evidence_prior_decay_start",
        type=int,
        default=None,
        help="Pass through to run_SR.py: segment-prior decay start iteration.",
    )
    parser.add_argument(
        "--evidence_prior_decay_interval",
        type=int,
        default=None,
        help="Pass through to run_SR.py: segment-prior decay interval.",
    )
    parser.add_argument(
        "--evidence_prior_decay_shape",
        type=str,
        choices=["linear", "smoothstep", "cosine"],
        default=None,
        help="Pass through to run_SR.py: segment-prior decay ramp shape.",
    )
    parser.add_argument(
        "--evidence_prior_decay_final_scale",
        type=float,
        default=None,
        help="Pass through to run_SR.py: final segment-prior multiplier after decay.",
    )
    parser.add_argument(
        "--evidence_prior_cutoff_tol",
        type=float,
        default=None,
        help=(
            "Pass through to run_SR.py: early-start trigger for segment-prior decay "
            "on plain training selection-loss improvement per LM report period."
        ),
    )
    parser.add_argument(
        "--no_evidence_prior_decay_auto",
        dest="evidence_prior_decay_auto",
        action="store_false",
        default=True,
        help="Pass through to run_SR.py: disable automatic segment-prior decay schedule.",
    )
    parser.add_argument(
        "--no_evidence_metric_gate",
        dest="evidence_metric_gate",
        action="store_false",
        default=True,
        help="Pass through to run_SR.py: allow metrics before prior decay completes.",
    )
    parser.add_argument(
        "--stat_selection",
        "--stat-selection",
        dest="stat_selection",
        action="store_true",
        help=(
            "Run each problem under statistical selection: reserve an untouched "
            "audit partition before the search opens, freeze the candidate "
            "archive, and certify a confidence Pareto front."
        ),
    )
    parser.add_argument(
        "--noise_sigma_frac_y_rms",
        type=float,
        default=None,
        help=(
            "Known homoscedastic additive y-noise, expressed as sigma_y / RMS(y_full). "
            "Passed through to run_SR.py."
        ),
    )
    parser.add_argument(
        "--ndata_train",
        type=int,
        default=None,
        help="Pass through to run_SR.py: number of training points.",
    )
    parser.add_argument(
        "--ndata_val",
        type=int,
        default=None,
        help="Pass through to run_SR.py: number of validation points.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Pass through to run_SR.py: dataloader batch size.",
    )
    parser.add_argument(
        "--data_slice",
        type=int,
        default=0,
        help=(
            "Pass through to run_SR.py: deterministic disjoint data block index. "
            "Default 0 preserves the historical split."
        ),
    )
    parser.add_argument(
        "--coe_mode",
        type=str,
        choices=["off", "audit_final", "final_adjudicate", "committee_gated", "reservoir_discovery"],
        default="off",
        help=(
            "Pass through to run_SR.py: CoE committee mode. "
            "Default off preserves the historical run."
        ),
    )
    parser.add_argument(
        "--coe_num_slices",
        type=int,
        default=None,
        help="Pass through to run_SR.py: number of validation slices for CoE committee audit.",
    )
    parser.add_argument(
        "--coe_start_slice",
        type=int,
        default=None,
        help="Pass through to run_SR.py: first data slice used by the CoE committee.",
    )
    parser.add_argument(
        "--coe_max_candidates",
        type=int,
        default=None,
        help="Pass through to run_SR.py: maximum final candidates audited by CoE.",
    )
    parser.add_argument(
        "--coe_reservoir_paths",
        type=str,
        default=None,
        help="Pass through to run_SR.py: extra CoE proposal-reservoir report files/directories.",
    )
    parser.add_argument(
        "--coe_noise_mult",
        type=float,
        default=None,
        help="Pass through to run_SR.py: CoE paired-vote noise multiplier.",
    )
    parser.add_argument(
        "--coe_rel_tol",
        type=float,
        default=None,
        help="Pass through to run_SR.py: CoE paired-vote relative tolerance.",
    )
    parser.add_argument(
        "--coe_min_valid_fraction",
        type=float,
        default=None,
        help="Pass through to run_SR.py: minimum finite-evaluation fraction for CoE candidates.",
    )
    parser.add_argument(
        "--coe_witness_parallelism",
        type=int,
        default=None,
        help="Pass through to run_SR.py: maximum concurrent CoE committee witness evaluations.",
    )
    parser.add_argument(
        "--coe_reservoir_support_bonus",
        type=float,
        default=None,
        help="Pass through to run_SR.py: reservoir support complexity bonus for noise-tied final CoE candidates.",
    )
    parser.add_argument(
        "--coe_stageB_dry_run",
        action="store_true",
        help=(
            "Pass through to run_SR.py: record observe-only CoE Stage-B "
            "risk diagnostics for accepted rewrites."
        ),
    )
    parser.add_argument(
        "--coe_stageB_gate_slices",
        type=int,
        default=None,
        help="Pass through to run_SR.py: max slices for Stage-B committee gating.",
    )
    parser.add_argument(
        "--coe_stageB_initial_gate_slices",
        type=int,
        default=None,
        help="Pass through to run_SR.py: initial slices for adaptive Stage-B committee gating.",
    )
    parser.add_argument(
        "--no_coe_stageB_refit_gate",
        dest="coe_stageB_refit_gate",
        action="store_false",
        default=True,
        help="Pass through to run_SR.py: disable short-refit gates for NN-containing Stage-B CoE decisions.",
    )
    parser.add_argument(
        "--coe_stageB_refit_epochs",
        type=int,
        default=None,
        help="Pass through to run_SR.py: epoch cap for short-refit Stage-B CoE gates.",
    )
    parser.add_argument(
        "--coe_stageB_refit_escalate_epochs",
        type=int,
        default=None,
        help=(
            "Pass through to run_SR.py: optional Tier-1 epoch cap used before "
            "enforcing a short-refit CoE veto."
        ),
    )
    parser.add_argument(
        "--coe_stageA_dry_run",
        action="store_true",
        help="Pass through to run_SR.py: observe-only CoE Stage-A risk diagnostics.",
    )
    parser.add_argument(
        "--coe_stageA_fit_tournament",
        action="store_true",
        help="Pass through to run_SR.py: parallel canonical slice-fit tournament.",
    )
    parser.add_argument(
        "--coe_stageA_fit_slices",
        type=str,
        default=None,
        help="Pass through to run_SR.py: explicit Stage-A fit challenger slices.",
    )
    parser.add_argument("--coe_stageA_fit_alpha", type=float, default=None)
    parser.add_argument("--coe_stageA_fit_comparison_fraction", type=float, default=None)
    parser.add_argument("--coe_stageA_fit_min_rel_improvement", type=float, default=None)
    parser.add_argument(
        "--coe_scout_count",
        type=int,
        default=None,
        help="Pass through to run_SR.py: number of bounded CoE scout proposer slices.",
    )
    parser.add_argument(
        "--coe_scout_slices",
        type=str,
        default=None,
        help="Pass through to run_SR.py: explicit bounded CoE scout slice ids.",
    )
    parser.add_argument(
        "--coe_scout_timeout_seconds",
        type=float,
        default=None,
        help="Pass through to run_SR.py: timeout per CoE scout proposer run.",
    )
    parser.add_argument(
        "--coe_scout_parallelism",
        type=int,
        default=None,
        help="Pass through to run_SR.py: maximum concurrent CoE scout proposer runs.",
    )
    parser.add_argument(
        "--coe_scout_stageB_epochs",
        type=int,
        default=None,
        help="Pass through to run_SR.py: Stage-B epoch cap for CoE scout proposer runs.",
    )
    parser.add_argument(
        "--coe_scout_stageB_max_outer_iters",
        type=int,
        default=None,
        help="Pass through to run_SR.py: Stage-B outer-iteration cap for CoE scout proposer runs.",
    )
    parser.add_argument(
        "--coe_scout_max_ab_iters",
        type=int,
        default=None,
        help="Pass through to run_SR.py: Stage A<->B iteration cap for CoE scout proposer runs.",
    )
    parser.add_argument(
        "--coe_scout_stageA_max_passes",
        type=int,
        default=None,
        help="Pass through to run_SR.py: Stage-A loop-pass cap for CoE scout proposer runs.",
    )
    parser.add_argument(
        "--no_coe_continuation_scouts",
        dest="coe_continuation_scouts",
        action="store_false",
        help=(
            "Pass through to run_SR.py: disable accepted Stage-A restart and "
            "Stage-B -> Stage-A continuation scouts."
        ),
    )
    parser.set_defaults(coe_continuation_scouts=True)
    parser.add_argument(
        "--coe_continuation_scout_count",
        type=int,
        default=None,
        help="Pass through to run_SR.py: scout count cap for each continuation phase.",
    )
    parser.add_argument(
        "--coe_continuation_scout_max_phases",
        type=int,
        default=None,
        help="Pass through to run_SR.py: maximum continuation scout phases per run.",
    )
    parser.add_argument(
        "--coe_scout_final_polish",
        action="store_true",
        help="Pass through to run_SR.py: allow final polish inside CoE scout proposer runs.",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        default=None,
        help="Start from a specific problem stem (e.g., 'pb050')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems to process",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only specific problems (comma-separated stems, e.g., 'pb000,pb001,pb002')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all problems (default behaviour when no --only/--limit given)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed output from each run (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed console output (log file still captures everything)",
    )
    parser.add_argument(
        "--single_layer",
        action="store_true",
        help="Use single-layer architecture (default is dual-layer)",
    )
    parser.add_argument(
        "--disable_compound_detection",
        action="store_true",
        help="Disable compound variable detection",
    )
    parser.add_argument(
        "--equations_txt",
        type=str,
        default=None,
        help="Path to equations.txt file for units information",
    )
    parser.add_argument(
        "--ignore_units",
        action="store_true",
        help="Disable dimensional consistency checking (units are enforced by default).",
    )
    parser.add_argument(
        "--sr_extra_args",
        type=str,
        default=None,
        help=(
            "Extra run_SR.py arguments appended verbatim (shlex-split) to every "
            "problem invocation, e.g. --sr_extra_args '--gs-stagea --gs-general-affine'"
        ),
    )
    parser.add_argument(
        "--verbose_separabilities",
        action="store_true",
        help="Print detailed separability diagnostics",
    )
    parser.add_argument(
        "--bypass",
        action="store_true",
        help="Skip separability detection; train NN model only",
    )

    # Stage B / feedback loop options
    parser.add_argument(
        "--max_ab_iters",
        type=int,
        default=None,
        help="Max Stage A<->B feedback loop iterations (default: 5). Set 1 to disable.",
    )
    parser.add_argument(
        "--stageB_max_outer_iters",
        type=int,
        default=None,
        help="Maximum number of Stage B refinement iterations (default: 30)",
    )
    parser.add_argument(
        "--stageB_epochs",
        type=int,
        default=None,
        help="Maximum LM epochs for Stage B fits",
    )
    parser.add_argument(
        "--max_backtracks",
        type=int,
        default=None,
        help="Max backtrack attempts in Stage B (0 to disable)",
    )
    parser.add_argument(
        "--factorized-search",
        dest="use_factorized_search",
        action="store_true",
        default=None,
        help="Force enable factorized symbolic search explorer in Stage B",
    )
    parser.add_argument(
        "--no-factorized-search",
        dest="use_factorized_search",
        action="store_false",
        help="Disable factorized symbolic search explorer in Stage B",
    )
    parser.add_argument(
        "--refine-skeleton",
        dest="use_refine_skeleton",
        action="store_true",
        default=None,
        help="Force enable continuous skeleton refinement inner refinement in Stage B",
    )
    parser.add_argument(
        "--no-refine-skeleton",
        dest="use_refine_skeleton",
        action="store_false",
        help="Disable continuous skeleton refinement inner refinement in Stage B",
    )
    parser.add_argument(
        "--disable_stageB_patterns",
        type=str,
        default=None,
        help="Comma-separated Stage B pattern names to disable (passed to run_SR.py)",
    )
    parser.add_argument(
        "--stageB_overcap_fallback",
        action="store_true",
        help=(
            "Pass through to run_SR.py: when no acceptance noise floor is set and "
            "Stage B is still above its hard cap, allow near-loss-neutral "
            "simplifications instead of enforcing the cap. Default off so "
            "noiseless runs preserve the original behaviour."
        ),
    )
    parser.add_argument(
        "--stageB_polish",
        dest="stageB_polish",
        action="store_true",
        default=None,
        help="Pass through to run_SR.py: enable accepted-step Stage B polishing.",
    )
    parser.add_argument(
        "--no_stageB_polish",
        dest="stageB_polish",
        action="store_false",
        help="Pass through to run_SR.py: disable accepted-step Stage B polishing.",
    )
    parser.add_argument(
        "--stageB_polish_max_candidates",
        type=int,
        default=None,
        help=(
            "Pass through to run_SR.py: maximum algebraic cleanup candidates for "
            "accepted-step Stage B polish."
        ),
    )
    parser.add_argument(
        "--stageB_polish_subtrees",
        dest="stageB_polish_subtrees",
        action="store_true",
        default=None,
        help=(
            "Pass through to run_SR.py: enable accepted-step polishing and commits "
            "for newly analytic Stage B subtrees (default inherited: on)."
        ),
    )
    parser.add_argument(
        "--no_stageB_polish_subtrees",
        dest="stageB_polish_subtrees",
        action="store_false",
        help=(
            "Pass through to run_SR.py: disable accepted-step polishing for newly "
            "analytic Stage B subtrees."
        ),
    )
    parser.add_argument(
        "--stageB_polish_max_subtrees",
        type=int,
        default=None,
        help=(
            "Pass through to run_SR.py: maximum newly analytic subtrees audited per "
            "accepted Stage B rewrite."
        ),
    )

    args = parser.parse_args()
    # Translate --ignore_units → enforce_units for all internal code
    args.enforce_units = not args.ignore_units

    # Load canaries
    print(f"Loading canaries from {args.canaries}...")
    canaries = load_canaries(args.canaries)
    print(f"Loaded {len(canaries)} AI Feynman problems")

    # Filter problems if requested
    if args.only is not None:
        only_stems = [s.strip() for s in args.only.split(",")]
        canaries = {k: v for k, v in canaries.items() if k in only_stems}
        print(f"Running only: {', '.join(only_stems)} ({len(canaries)} problems)")

    # Sort by problem ID
    sorted_stems = sorted(canaries.keys(), key=lambda s: canaries[s].get("id", 999))

    # Handle start_from
    if args.start_from is not None:
        try:
            start_idx = sorted_stems.index(args.start_from)
            sorted_stems = sorted_stems[start_idx:]
            print(f"Starting from {args.start_from}")
        except ValueError:
            print(f"Warning: start_from '{args.start_from}' not found in canaries")

    # Handle limit
    if args.limit is not None:
        sorted_stems = sorted_stems[: args.limit]
        print(f"Limited to {args.limit} problems")

    # Resolve results directory
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Set default equations_txt if not provided
    equations_txt = args.equations_txt
    if equations_txt is None:
        default_equations = os.path.join(os.path.dirname(script_dir), "data", "equations.txt")
        if os.path.exists(default_equations):
            equations_txt = default_equations
            print(f"Using equations.txt: {equations_txt}")

    # Run full pipeline on each problem
    results = []
    total_start = time.time()

    for i, stem in enumerate(sorted_stems, 1):
        canary = canaries[stem]
        canary_id = canary.get("id", "?")

        print(f"\n[{i}/{len(sorted_stems)}] Problem {stem} (ID: {canary_id})")
        print(f"Description: {canary.get('description', 'N/A')}")

        filepath = find_data_file(stem, args.data_dir)

        if filepath is None:
            print(f"✗ Data file not found for {stem}")
            results.append(
                {
                    "stem": stem,
                    "id": canary_id,
                    "filepath": None,
                    "success": False,
                    "walltime_seconds": None,
                    "error": "Data file not found",
                    "log_file": None,
                }
            )
            continue

        result = run_allstages_on_problem(
            stem=stem,
            filepath=filepath,
            results_dir=results_dir,
            fast_mode=args.fast,
            stat_selection=args.stat_selection,
            noise_sigma_frac_y_rms=args.noise_sigma_frac_y_rms,
            force_y_ops=args.force_y_ops,
            single_layer=args.single_layer,
            disable_compound_detection=args.disable_compound_detection,
            equations_txt=equations_txt,
            ignore_units=args.ignore_units,
            verbose=args.verbose and not args.quiet,
            verbose_separabilities=args.verbose_separabilities,
            bypass=args.bypass,
            max_ab_iters=args.max_ab_iters,
            stageB_max_outer_iters=args.stageB_max_outer_iters,
            stageB_epochs=args.stageB_epochs,
            max_backtracks=args.max_backtracks,
            use_factorized_search=args.use_factorized_search,
            use_refine_skeleton=args.use_refine_skeleton,
            disable_stageB_patterns=args.disable_stageB_patterns,
            stageB_overcap_fallback=args.stageB_overcap_fallback,
            stageB_polish=args.stageB_polish,
            stageB_polish_max_candidates=args.stageB_polish_max_candidates,
            stageB_polish_subtrees=args.stageB_polish_subtrees,
            stageB_polish_max_subtrees=args.stageB_polish_max_subtrees,
            canonical_init=args.canonical_init,
            evidence=args.evidence,
            ndata_train=args.ndata_train,
            ndata_val=args.ndata_val,
            batch_size=args.batch_size,
            data_slice=args.data_slice,
            coe_mode=args.coe_mode,
            coe_num_slices=args.coe_num_slices,
            coe_start_slice=args.coe_start_slice,
            coe_max_candidates=args.coe_max_candidates,
            coe_reservoir_paths=args.coe_reservoir_paths,
            coe_noise_mult=args.coe_noise_mult,
            coe_rel_tol=args.coe_rel_tol,
            coe_min_valid_fraction=args.coe_min_valid_fraction,
            coe_witness_parallelism=args.coe_witness_parallelism,
            coe_reservoir_support_bonus=args.coe_reservoir_support_bonus,
            coe_stageB_dry_run=args.coe_stageB_dry_run,
            coe_stageB_gate_slices=args.coe_stageB_gate_slices,
            coe_stageB_initial_gate_slices=args.coe_stageB_initial_gate_slices,
            coe_stageB_refit_gate=args.coe_stageB_refit_gate,
            coe_stageB_refit_epochs=args.coe_stageB_refit_epochs,
            coe_stageB_refit_escalate_epochs=args.coe_stageB_refit_escalate_epochs,
            coe_stageA_dry_run=args.coe_stageA_dry_run,
            coe_stageA_fit_tournament=args.coe_stageA_fit_tournament,
            coe_stageA_fit_slices=args.coe_stageA_fit_slices,
            coe_stageA_fit_alpha=args.coe_stageA_fit_alpha,
            coe_stageA_fit_comparison_fraction=args.coe_stageA_fit_comparison_fraction,
            coe_stageA_fit_min_rel_improvement=args.coe_stageA_fit_min_rel_improvement,
            coe_scout_count=args.coe_scout_count,
            coe_scout_slices=args.coe_scout_slices,
            coe_scout_timeout_seconds=args.coe_scout_timeout_seconds,
            coe_scout_parallelism=args.coe_scout_parallelism,
            coe_scout_stageB_epochs=args.coe_scout_stageB_epochs,
            coe_scout_stageB_max_outer_iters=args.coe_scout_stageB_max_outer_iters,
            coe_scout_max_ab_iters=args.coe_scout_max_ab_iters,
            coe_scout_stageA_max_passes=args.coe_scout_stageA_max_passes,
            coe_continuation_scouts=args.coe_continuation_scouts,
            coe_continuation_scout_count=args.coe_continuation_scout_count,
            coe_continuation_scout_max_phases=args.coe_continuation_scout_max_phases,
            coe_scout_final_polish=args.coe_scout_final_polish,
            sr_extra_args=args.sr_extra_args,
            evidence_disable_residual_whitening=args.evidence_disable_residual_whitening,
            evidence_disable_segment_priors=args.evidence_disable_segment_priors,
            evidence_lambda_patch=args.evidence_lambda_patch,
            evidence_prior_decay_start=args.evidence_prior_decay_start,
            evidence_prior_decay_interval=args.evidence_prior_decay_interval,
            evidence_prior_decay_shape=args.evidence_prior_decay_shape,
            evidence_prior_decay_final_scale=args.evidence_prior_decay_final_scale,
            evidence_prior_cutoff_tol=args.evidence_prior_cutoff_tol,
            evidence_prior_decay_auto=args.evidence_prior_decay_auto,
            evidence_metric_gate=args.evidence_metric_gate,
        )
        result["id"] = canary_id
        results.append(result)

    total_elapsed = time.time() - total_start

    # Write summary report
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(results_dir, "allstages_suite_summary.json")
    else:
        output_path = os.path.abspath(output_path)

    write_summary_report(results, output_path)

    print(f"\nTotal suite runtime: {total_elapsed / 3600:.2f} hours")

    failed = sum(1 for r in results if not r["success"])
    if failed > 0:
        print(f"\n⚠️  {failed} problems failed")
        return 1
    else:
        print("\n✓ All problems completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
