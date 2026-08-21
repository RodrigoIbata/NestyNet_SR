#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Run Stage A (neural network fitting + separability detection) on all AI Feynman problems.

This script:
1. Loads all AI Feynman canaries from aif_canaries.json
2. Finds corresponding CSV data files
3. Runs Stage A only (no Stage B refinement) on each problem
4. Saves results and generates a summary report
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
    # If relative path, resolve relative to script directory
    if not os.path.isabs(canaries_path):
        canaries_path = os.path.join(script_dir, canaries_path)

    try:
        with open(canaries_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Failed to load {canaries_path}: {e}")
        sys.exit(1)


def find_data_file(stem: str, data_dir: str) -> Optional[str]:
    """
    Find the CSV data file corresponding to a problem stem.

    Parameters
    ----------
    stem : str
        Problem stem (e.g., "pb000")
    data_dir : str
        Directory containing data files (absolute or relative to cwd)

    Returns
    -------
    str or None
        Absolute path to data file, or None if not found
    """
    # Resolve data_dir to absolute path
    data_path = Path(data_dir).resolve()

    # Try different naming patterns
    patterns = [
        f"{stem}_data.csv",
        f"{stem}.csv",
    ]

    # Also try glob pattern for variants like pb000_I_6_2a_data.csv
    import glob

    glob_pattern = str(data_path / f"{stem}_*.csv")
    matches = glob.glob(glob_pattern)

    if matches:
        # Prefer files ending in _data.csv
        data_files = [m for m in matches if m.endswith("_data.csv")]
        if data_files:
            return os.path.abspath(data_files[0])
        return os.path.abspath(matches[0])

    # Try exact patterns
    for pattern in patterns:
        filepath = data_path / pattern
        if filepath.exists():
            return str(filepath.resolve())

    return None


def run_stageA_on_problem(
    stem: str,
    filepath: str,
    results_dir: str,
    fast_mode: bool = False,
    force_y_ops: Optional[str] = None,
    single_layer: bool = False,
    disable_compound_detection: bool = False,
    equations_txt: Optional[str] = None,
    ignore_units: bool = False,
    verbose: bool = False,
    verbose_separabilities: bool = False,
    bypass: bool = False,
) -> Dict:
    """
    Run Stage A on a single AI Feynman problem.

    Captures stdout/stderr and saves to log file.
    Log files are saved as <results_dir>/<data_stem>_stageA.log

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
        Use single-layer architecture (default is dual-layer)
    disable_compound_detection : bool
        Disable compound variable detection
    equations_txt : str, optional
        Path to equations.txt file for units information
    ignore_units : bool
        Disable dimensional consistency checking
    verbose : bool
        Print detailed output to console

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

    # Construct command - use absolute path to run_SR.py
    run_sr_path = os.path.join(script_dir, "run_SR.py")

    cmd = [
        sys.executable,
        "-u",  # Unbuffered output for real-time logging
        run_sr_path,
        "--filepath",
        filepath,
        "--no_stageB",  # Disable Stage B - only run Stage A
        "--log_level",
        "INFO",  # Enable verbose logging by default
    ]

    # Add optional arguments
    if fast_mode:
        cmd.append("--fast")

    if force_y_ops is not None:
        cmd.extend(["--force_y_ops", force_y_ops])

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

    # Determine log file path (we'll capture stdout/stderr and save it ourselves)
    # Extract stem from filepath for consistent naming
    data_stem = Path(filepath).stem  # e.g., pb000_I_6_2a_data
    log_path = os.path.join(results_dir, f"{data_stem}_stageA.log")
    result["log_file"] = log_path

    # Run Stage A
    print(f"\n{'=' * 60}")
    print(f"Running Stage A on {stem}")
    print(f"Data: {filepath}")
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
    output_path: str = "results/stageA_suite_summary.json",
):
    """Write summary report of Stage A suite run."""
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
    print("STAGE A SUITE SUMMARY")
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
    # Default results_dir relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(os.path.dirname(script_dir), "results")

    parser = argparse.ArgumentParser(
        description="Run Stage A on all AI Feynman problems",
        epilog="Example: python nestynet_sr/run_stageA_suite.py --fast --limit 5",
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
        help="Path to save summary JSON (default: results/stageA_suite_summary.json)",
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
        help="Use single-layer architecture (default is dual-layer, passed to run_SR.py)",
    )
    parser.add_argument(
        "--disable_compound_detection",
        action="store_true",
        help="Disable compound variable detection (passed to run_SR.py)",
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
        "--bypass",
        action="store_true",
        help="Skip separability detection; train NN model only (passed as --no_stageA_separabilities to run_SR.py)",
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

    # Resolve results directory (relative to cwd or absolute)
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

    # Run Stage A on each problem
    results = []
    total_start = time.time()

    for i, stem in enumerate(sorted_stems, 1):
        canary = canaries[stem]
        canary_id = canary.get("id", "?")

        print(f"\n[{i}/{len(sorted_stems)}] Problem {stem} (ID: {canary_id})")
        print(f"Description: {canary.get('description', 'N/A')}")

        # Find data file
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

        # Run Stage A
        result = run_stageA_on_problem(
            stem=stem,
            filepath=filepath,
            results_dir=results_dir,
            fast_mode=args.fast,
            force_y_ops=args.force_y_ops,
            single_layer=args.single_layer,
            disable_compound_detection=args.disable_compound_detection,
            equations_txt=equations_txt,
            ignore_units=args.ignore_units,
            verbose=args.verbose and not args.quiet,
            verbose_separabilities=args.verbose_separabilities,
            bypass=args.bypass,
        )
        result["id"] = canary_id
        results.append(result)

    total_elapsed = time.time() - total_start

    # Write summary report
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(results_dir, "stageA_suite_summary.json")
    else:
        output_path = os.path.abspath(output_path)

    write_summary_report(results, output_path)

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
