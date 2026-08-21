#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Run Stages B and C (refinement + simplification) on all AI Feynman problems that completed Stage A.

This script:
1. Finds all Stage A checkpoint files (*.state.pkl) in results directory
2. Resumes from each checkpoint to run Stages B and C
3. Saves refined results and generates a summary report

Stages:
- Stage A: Neural network fitting + separability detection (already completed)
- Stage B: Analytical expression refinement and rewriting
- Stage C: SymPy simplification (integrated into Stage B)
"""

import argparse
import json
import os
import pickle
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


def find_stageA_checkpoints(
    results_dir: str, only_prefixes: Optional[List[str]] = None
) -> List[str]:
    """
    Find all Stage A checkpoint files in the results directory.

    Parameters
    ----------
    results_dir : str
        Directory containing Stage A checkpoint files
    only_prefixes : list of str, optional
        If provided, only load checkpoints whose stem starts with one of
        these prefixes.  This avoids deserialising every checkpoint on disk.

    Returns
    -------
    list of str
        List of absolute paths to checkpoint files
    """
    results_path = Path(results_dir).resolve()

    # Find all .state.pkl files
    checkpoints = list(results_path.glob("*.state.pkl"))

    # Filter to only Stage A checkpoints (phase='after_stageA')
    stageA_checkpoints = []
    for ckpt_path in checkpoints:
        # Skip files that don't match any requested prefix (before pickle.load)
        if only_prefixes is not None:
            stem = ckpt_path.stem.removesuffix(".state")  # strip .state from .state.pkl
            if not any(stem.startswith(prefix) for prefix in only_prefixes):
                continue

        try:
            with open(ckpt_path, "rb") as f:
                data = pickle.load(f)
                phase = data.get("phase", None)
                # Include checkpoints that are at 'after_stageA' phase
                if phase == "after_stageA":
                    stageA_checkpoints.append(str(ckpt_path.resolve()))
        except Exception as e:
            print(f"Warning: Failed to read {ckpt_path}: {e}")
            continue

    return sorted(stageA_checkpoints)


def extract_stem_from_checkpoint(ckpt_path: str) -> str:
    """
    Extract dataset stem from checkpoint filename.

    Parameters
    ----------
    ckpt_path : str
        Path to checkpoint file

    Returns
    -------
    str
        Dataset stem (e.g., 'pb000_I_6_2a_data')
    """
    filename = Path(ckpt_path).name
    # Remove .state.pkl suffix
    if filename.endswith(".state.pkl"):
        return filename[:-10]
    return filename


def load_stageA_summary(summary_path: str) -> Optional[Dict]:
    """
    Load Stage A summary file to get metadata about problems.

    Parameters
    ----------
    summary_path : str
        Path to Stage A summary JSON file

    Returns
    -------
    dict or None
        Stage A summary data, or None if file doesn't exist
    """
    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load Stage A summary from {summary_path}: {e}")
        return None


def run_stageBC_on_checkpoint(
    ckpt_path: str,
    results_dir: str,
    fast_mode: bool = False,
    verbose: bool = False,
    stageB_max_outer_iters: Optional[int] = None,
    stageB_epochs: Optional[int] = None,
    stageB_score_tol: Optional[float] = None,
    disable_stageB_patterns: Optional[str] = None,
    equations_txt: Optional[str] = None,
    ignore_units: bool = False,
    verbose_separabilities: bool = False,
    use_factorized_search: bool = False,
    no_brute_force: bool = False,
    max_ab_iters: Optional[int] = None,
    max_backtracks: Optional[int] = None,
) -> Dict:
    """
    Run Stages B and C on a single checkpoint.

    Captures stdout/stderr and saves to log file.
    Log files are saved as <results_dir>/<stem>_stageBC.log

    Parameters
    ----------
    ckpt_path : str
        Path to Stage A checkpoint file
    results_dir : str
        Directory to save results
    fast_mode : bool
        Use fast mode (reduced epochs)
    verbose : bool
        Print detailed output to console
    stageB_max_outer_iters : int, optional
        Maximum Stage B iterations
    stageB_epochs : int, optional
        Maximum LM epochs per Stage B candidate
    stageB_score_tol : float, optional
        Minimum improvement for Stage B rewrites
    disable_stageB_patterns : str, optional
        Comma-separated list of Stage B patterns to disable
    equations_txt : str, optional
        Path to equations.txt file for units information
    ignore_units : bool
        Disable dimensional consistency checking

    Returns
    -------
    dict
        Result with success status and timing information
    """
    stem = extract_stem_from_checkpoint(ckpt_path)

    result = {
        "stem": stem,
        "checkpoint": ckpt_path,
        "success": False,
        "walltime_seconds": None,
        "error": None,
    }

    # Load checkpoint to extract filepath(s)
    # run_SR.py requires --filepath or --filepaths even when resuming
    try:
        with open(ckpt_path, "rb") as f:
            ckpt_data = pickle.load(f)

        filepaths = ckpt_data.get("filepaths")
        filepath = ckpt_data.get("filepath")

        def fix_filepath(path):
            """Resolve a checkpoint data path in the current checkout."""
            if path is None:
                return None

            # If path exists, use it as-is
            if os.path.exists(path):
                return path

            # If path doesn't exist, try to find it in current repo's data/ directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            filename = os.path.basename(path)
            candidate = os.path.join(repo_root, "data", filename)
            if os.path.exists(candidate):
                print(f"  Fixed path: {filename} (found in data/)")
                return candidate

            # Give up, return original
            print(f"  Warning: could not fix path: {path}")
            return path

        # Fix filepaths - ensure they are absolute paths
        if filepaths is not None and len(filepaths) > 0:
            filepaths = [os.path.abspath(fix_filepath(fp)) for fp in filepaths]
            filepath_args = ["--filepaths"] + filepaths
        elif filepath is not None:
            filepath = os.path.abspath(fix_filepath(filepath))
            filepath_args = ["--filepath", filepath]
        else:
            result["error"] = "Checkpoint missing filepath/filepaths"
            print(f"✗ {stem} error: checkpoint missing filepath information")
            return result
    except Exception as e:
        result["error"] = f"Failed to read checkpoint: {e}"
        print(f"✗ {stem} error: failed to read checkpoint: {e}")
        return result

    # Construct command - use absolute path to run_SR.py
    run_sr_path = os.path.join(script_dir, "run_SR.py")

    cmd = [
        sys.executable,
        "-u",  # Unbuffered output for real-time logging
        run_sr_path,
        "--resume_from",
        ckpt_path,
        # Note: --stageB is True by default (no --no_stageB flag)
        "--log_level",
        "INFO",  # Enable verbose logging by default
    ]

    # Add filepath argument(s)
    cmd.extend(filepath_args)

    # Add optional arguments
    if fast_mode:
        cmd.append("--fast")

    if stageB_max_outer_iters is not None:
        cmd.extend(["--stageB_max_outer_iters", str(stageB_max_outer_iters)])

    if stageB_epochs is not None:
        cmd.extend(["--stageB_epochs", str(stageB_epochs)])

    if stageB_score_tol is not None:
        cmd.extend(["--stageB_score_tol", str(stageB_score_tol)])

    if disable_stageB_patterns is not None:
        cmd.extend(["--disable_stageB_patterns", disable_stageB_patterns])

    if equations_txt is not None:
        cmd.extend(["--equations_txt", equations_txt])

    if ignore_units:
        cmd.append("--ignore_units")

    if verbose_separabilities:
        cmd.append("--verbose_separabilities")

    if use_factorized_search:
        cmd.append("--factorized-search")
    else:
        cmd.append("--no-factorized-search")

    if no_brute_force:
        cmd.append("--no_brute_force")

    if max_ab_iters is not None:
        cmd.extend(["--max_ab_iters", str(max_ab_iters)])

    if max_backtracks is not None:
        cmd.extend(["--max_backtracks", str(max_backtracks)])

    # Determine log file path (we'll capture stdout/stderr and save it ourselves)
    log_path = os.path.join(results_dir, f"{stem}_stageBC.log")
    result["log_file"] = log_path

    # Run Stages B and C
    print(f"\n{'=' * 60}")
    print(f"Running Stages B & C on {stem}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log: {log_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        # Open log file for real-time streaming
        with open(log_path, "w") as log_file:
            # Run subprocess with output streaming to log file
            if verbose:
                # Verbose mode: stream to both log file and console
                process = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Stream output line by line
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    print(line, end="")

                process.wait()
            else:
                # Non-verbose mode: stream only to log file
                process = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                process.wait()

        elapsed = time.time() - start_time

        # Check exit code
        if process.returncode != 0:
            result["success"] = False
            result["walltime_seconds"] = elapsed
            result["error"] = f"Exit code {process.returncode}"

            print(
                f"✗ {stem} failed after {elapsed / 60:.2f} minutes (exit code {process.returncode})"
            )
            print(f"  Error details logged to: {log_path}")

            # Show last 20 lines of log file
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
    stageA_summary: Optional[Dict],
    output_path: str = "results/stageBC_suite_summary.json",
):
    """Write summary report of Stages B and C suite run."""
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

    # Include Stage A summary if available
    if stageA_summary is not None:
        summary["stageA_summary"] = {
            "total_problems": stageA_summary.get("total_problems"),
            "successful": stageA_summary.get("successful"),
            "total_walltime_hours": stageA_summary.get("total_walltime_hours"),
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("STAGES B & C SUITE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total problems:     {total:4d}")
    if total > 0:
        print(f"Successful:         {successful:4d} ({100 * successful / total:.1f}%)")
        print(f"Failed:             {failed:4d} ({100 * failed / total:.1f}%)")
    else:
        print(f"Successful:         {successful:4d}")
        print(f"Failed:             {failed:4d}")
    print(f"Total walltime:     {total_time / 3600:.2f} hours")
    if successful > 0:
        avg_time = (total_time / successful) / 60
        print(f"Avg time/problem:   {avg_time:.2f} minutes")
    print(f"\nSummary saved to: {output_path}")
    print(f"{'=' * 60}\n")


def main():
    # Default results_dir relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(os.path.dirname(script_dir), "results")

    parser = argparse.ArgumentParser(
        description="Run Stages B and C on all AI Feynman problems that completed Stage A",
        epilog="Example: python nestynet_sr/run_stageBC_suite.py --fast --limit 5",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=default_results_dir,
        help="Directory containing Stage A checkpoints (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save summary JSON (default: results/stageBC_suite_summary.json)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode (reduced epochs for quick testing)",
    )
    parser.add_argument(
        "--stageB_max_outer_iters",
        type=int,
        default=None,
        help="Maximum Stage B refinement iterations (default: 30)",
    )
    parser.add_argument(
        "--stageB_epochs",
        type=int,
        default=None,
        help="Maximum LM epochs per Stage B candidate (default: 2000)",
    )
    parser.add_argument(
        "--stageB_score_tol",
        type=float,
        default=None,
        help="Minimum improvement for Stage B rewrites (default: 0.0)",
    )
    parser.add_argument(
        "--disable_stageB_patterns",
        type=str,
        default=None,
        help="Comma-separated list of Stage B patterns to disable",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        default=None,
        help="Start from a specific problem (prefix match, e.g., 'pb050' or 'pb050_I_50_26_data')",
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
        help="Run only specific problems (comma-separated prefixes, e.g., 'pb048,pb049')",
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
        "--stageA_summary",
        type=str,
        default=None,
        help="Path to Stage A summary JSON (optional, for metadata)",
    )
    parser.add_argument(
        "--equations_txt",
        type=str,
        default=None,
        help="Path to equations.txt file for units information (default: data/equations.txt if it exists)",
    )
    parser.add_argument(
        "--ignore_units",
        action="store_true",
        help="Disable dimensional consistency checking (units are enforced by default).",
    )
    parser.add_argument(
        "--verbose_separabilities",
        action="store_true",
        help="Print detailed separability diagnostics (passed to run_SR.py)",
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
        "--no_brute_force",
        action="store_true",
        help="disable factorized symbolic search brute-force enumeration phase (keep mutation search only)",
    )
    parser.add_argument(
        "--max_ab_iters",
        type=int,
        default=None,
        help="Max Stage A<->B feedback loop iterations (default: 5). Set 1 to disable.",
    )
    parser.add_argument(
        "--max_backtracks",
        type=int,
        default=None,
        help="Max backtrack attempts in Stage B (0 to disable, default: 3)",
    )
    args = parser.parse_args()
    # Translate --ignore_units → enforce_units for all internal code
    args.enforce_units = not args.ignore_units

    # Resolve factorized symbolic search default: on when enforce_units (now the default)
    if args.use_factorized_search is None:
        args.use_factorized_search = bool(args.enforce_units)

    # Resolve results directory (relative to cwd or absolute)
    results_dir = os.path.abspath(args.results_dir)
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    # Set default equations_txt if not provided
    equations_txt = args.equations_txt
    if equations_txt is None:
        # Try to find equations.txt in data/ directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        default_equations = os.path.join(repo_root, "data", "equations.txt")
        if os.path.exists(default_equations):
            equations_txt = default_equations
            print(f"Using equations.txt: {equations_txt}")

    # Load Stage A summary if available
    stageA_summary = None
    if args.stageA_summary is not None:
        stageA_summary_path = args.stageA_summary
    else:
        stageA_summary_path = os.path.join(results_dir, "stageA_suite_summary.json")

    if os.path.exists(stageA_summary_path):
        stageA_summary = load_stageA_summary(stageA_summary_path)
        if stageA_summary is not None:
            print(f"Loaded Stage A summary from {stageA_summary_path}")

    # Parse --only prefixes once, pass into checkpoint finder to skip early
    only_prefixes = None
    if args.only is not None:
        only_prefixes = [s.strip() for s in args.only.split(",")]

    # Find all Stage A checkpoints (filtered by prefix before loading)
    print(f"Searching for Stage A checkpoints in {results_dir}...")
    checkpoints = find_stageA_checkpoints(results_dir, only_prefixes=only_prefixes)
    print(f"Found {len(checkpoints)} Stage A checkpoints")

    if len(checkpoints) == 0:
        print("No Stage A checkpoints found. Make sure Stage A has completed.")
        return 1

    # Extract stems from checkpoint paths
    stems = [extract_stem_from_checkpoint(ckpt) for ckpt in checkpoints]

    if only_prefixes is not None:
        print(f"Running only: {', '.join(only_prefixes)} ({len(checkpoints)} problems)")

    # Handle start_from
    if args.start_from is not None:
        # Find first stem that starts with the provided prefix
        start_idx = None
        for i, stem in enumerate(stems):
            if stem.startswith(args.start_from):
                start_idx = i
                break

        if start_idx is not None:
            checkpoints = checkpoints[start_idx:]
            stems = stems[start_idx:]
            print(f"Starting from {stems[0]} (matched prefix '{args.start_from}')")
        else:
            print(f"Warning: start_from '{args.start_from}' not found in checkpoints")

    # Handle limit
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
        stems = stems[: args.limit]
        print(f"Limited to {args.limit} problems")

    print(f"Will process {len(checkpoints)} problems")

    # Run Stages B and C on each checkpoint
    results = []
    total_start = time.time()

    for i, (ckpt_path, stem) in enumerate(zip(checkpoints, stems), 1):
        print(f"\n[{i}/{len(checkpoints)}] Problem {stem}")

        # Run Stages B and C
        result = run_stageBC_on_checkpoint(
            ckpt_path=ckpt_path,
            results_dir=results_dir,
            fast_mode=args.fast,
            verbose=args.verbose and not args.quiet,
            stageB_max_outer_iters=args.stageB_max_outer_iters,
            stageB_epochs=args.stageB_epochs,
            stageB_score_tol=args.stageB_score_tol,
            disable_stageB_patterns=args.disable_stageB_patterns,
            equations_txt=equations_txt,
            ignore_units=args.ignore_units,
            verbose_separabilities=args.verbose_separabilities,
            use_factorized_search=args.use_factorized_search,
            no_brute_force=args.no_brute_force,
            max_ab_iters=args.max_ab_iters,
            max_backtracks=args.max_backtracks,
        )
        results.append(result)

    total_elapsed = time.time() - total_start

    # Write summary report
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(results_dir, "stageBC_suite_summary.json")
    else:
        output_path = os.path.abspath(output_path)

    write_summary_report(results, stageA_summary, output_path)

    print(f"\nTotal suite runtime: {total_elapsed / 3600:.2f} hours")

    # Return exit code based on results
    failed = sum(1 for r in results if not r["success"])
    if failed > 0:
        print(f"\n⚠️  {failed} problems failed")
        return 1
    else:
        print("\n✓ All problems completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
