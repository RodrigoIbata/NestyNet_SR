#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Run the full SR pipeline (Stage A + Stage B + A<->B feedback loop) on univariate
benchmark problems (u000-u024).

Usage:
    python nestynet_sr/run_allstages_u.py --only u001
    python nestynet_sr/run_allstages_u.py --fast --limit 5
    python nestynet_sr/run_allstages_u.py --all
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

# ---------------------------------------------------------------------------
# Univariate benchmark registry  (u000 – u024)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "u000": {"id": 0,  "desc": "Linear: y = c*x"},
    "u001": {"id": 1,  "desc": "Quadratic monomial: y = c*x^2"},
    "u002": {"id": 2,  "desc": "2-term polynomial: y = c1*x^2 + c2*x"},
    "u003": {"id": 3,  "desc": "Cubic monomial: y = c*x^3"},
    "u004": {"id": 4,  "desc": "Odd polynomial: y = c1*x^3 + c2*x"},
    "u005": {"id": 5,  "desc": "Even polynomial (3 terms): double-well potential"},
    "u006": {"id": 6,  "desc": "Inverse: y = c/x"},
    "u007": {"id": 7,  "desc": "Inverse square: y = c/x^2"},
    "u008": {"id": 8,  "desc": "Square root: y = c*sqrt(x)"},
    "u009": {"id": 9,  "desc": "Power 3/2: y = c*x^(3/2)"},
    "u010": {"id": 10, "desc": "Poly + rational: y = c1*x + c2/x"},
    "u011": {"id": 11, "desc": "Sine: y = c1*sin(c2*x)"},
    "u012": {"id": 12, "desc": "Cosine: y = c1*cos(c2*x)"},
    "u013": {"id": 13, "desc": "Exp decay: y = c1*exp(-c2*x)"},
    "u014": {"id": 14, "desc": "Logarithm: y = c1*ln(c2*x)"},
    "u015": {"id": 15, "desc": "Gaussian: y = c1*exp(-c2*x^2)"},
    "u016": {"id": 16, "desc": "Poly * exp: y = c1*x*exp(-c2*x)"},
    "u017": {"id": 17, "desc": "Trig + poly: y = c1*sin(c2*x) + c3*x"},
    "u018": {"id": 18, "desc": "Poly * trig: y = c1*x*sin(c2*x)"},
    "u019": {"id": 19, "desc": "Quadratic * exp: y = c1*x^2*exp(-c2*x)"},
    "u020": {"id": 20, "desc": "Damped oscillation: y = c1*exp(-c2*x)*sin(c3*x)"},
    "u021": {"id": 21, "desc": "Double harmonic: y = c1*sin(c2*x) + c3*sin(c4*x)"},
    "u022": {"id": 22, "desc": "Sigmoid: y = c1/(1+exp(-c2*x))"},
    "u023": {"id": 23, "desc": "Nested trig: y = c1*sin(c2*x^2)"},
    "u024": {"id": 24, "desc": "Nguyen-5 analog: y = c1*sin(c2*x^2)*cos(c3*x)"},
}


def find_data_file(stem: str, data_dir: str) -> Optional[str]:
    """Find the CSV data file for a univariate benchmark problem."""
    data_path = Path(data_dir).resolve()
    filepath = data_path / f"{stem}.csv"
    if filepath.exists():
        return str(filepath)
    return None


def run_allstages_on_problem(
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
    max_ab_iters: Optional[int] = None,
    stageB_max_outer_iters: Optional[int] = None,
    stageB_epochs: Optional[int] = None,
    max_backtracks: Optional[int] = None,
    no_factorized_search: bool = False,
    factorized_search_plus: bool = False,
) -> Dict:
    """Run the full SR pipeline on a single univariate benchmark problem."""
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
    ]

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

    if max_ab_iters is not None:
        cmd.extend(["--max_ab_iters", str(max_ab_iters)])

    if stageB_max_outer_iters is not None:
        cmd.extend(["--stageB_max_outer_iters", str(stageB_max_outer_iters)])

    if stageB_epochs is not None:
        cmd.extend(["--stageB_epochs", str(stageB_epochs)])

    if max_backtracks is not None:
        cmd.extend(["--max_backtracks", str(max_backtracks)])

    if no_factorized_search:
        cmd.append("--no-factorized-search")

    if factorized_search_plus:
        cmd.append("--refine-skeleton")

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
                f"FAIL {stem} failed after {elapsed / 60:.2f} minutes (exit code {process.returncode})"
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

            print(f"OK {stem} completed in {elapsed / 60:.2f} minutes")
            print(f"  Log saved to: {log_path}")

    except Exception as e:
        elapsed = time.time() - start_time
        result["walltime_seconds"] = elapsed
        result["error"] = str(e)

        print(f"FAIL {stem} crashed: {e}")
        if os.path.exists(log_path):
            print(f"  Partial log saved to: {log_path}")
        else:
            print("  (No log file created)")

    return result


def write_summary_report(
    results: List[Dict],
    output_path: str = "results/univariate_suite_summary.json",
):
    """Write summary report of univariate benchmark suite run."""
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
    print("UNIVARIATE BENCHMARK SUITE SUMMARY")
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
    default_results_dir = os.path.join(project_root, "results")

    parser = argparse.ArgumentParser(
        description="Run full SR pipeline on univariate benchmark problems (u000-u024)",
        epilog="Example: python nestynet_sr/run_allstages_u.py --only u001",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(project_root, "data"),
        help="Directory containing u*.csv data files (default: data/)",
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
        help="Path to save summary JSON (default: results/univariate_suite_summary.json)",
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
        help="Start from a specific problem stem (e.g., 'u010')",
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
        help="Run only specific problems (comma-separated, e.g., 'u000,u001,u002')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 25 problems (default behaviour when no --only/--limit given)",
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
        help="Path to equations file for units (default: data/univariate_benchmark.txt)",
    )
    parser.add_argument(
        "--ignore_units",
        action="store_true",
        help="Disable dimensional consistency checking (units are enforced by default).",
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
        "--no-factorized-search",
        action="store_true",
        help="Disable factorized symbolic search explorer in Stage B",
    )
    parser.add_argument(
        "--refine-skeleton",
        action="store_true",
        help="Enable continuous skeleton refinement inner refinement (LBFGS on trig/log/exp scales)",
    )

    args = parser.parse_args()
    # Translate --ignore_units → enforce_units for all internal code
    args.enforce_units = not args.ignore_units

    # Select problems
    problems = dict(BENCHMARKS)

    if args.only is not None:
        only_stems = [s.strip() for s in args.only.split(",")]
        problems = {k: v for k, v in problems.items() if k in only_stems}
        missing = [s for s in only_stems if s not in BENCHMARKS]
        if missing:
            print(f"Warning: unknown problem stems: {', '.join(missing)}")
        print(f"Running only: {', '.join(sorted(problems.keys()))} ({len(problems)} problems)")

    # Sort by ID
    sorted_stems = sorted(problems.keys(), key=lambda s: problems[s]["id"])

    # Handle start_from
    if args.start_from is not None:
        try:
            start_idx = sorted_stems.index(args.start_from)
            sorted_stems = sorted_stems[start_idx:]
            print(f"Starting from {args.start_from}")
        except ValueError:
            print(f"Warning: start_from '{args.start_from}' not found")

    # Handle limit
    if args.limit is not None:
        sorted_stems = sorted_stems[: args.limit]
        print(f"Limited to {args.limit} problems")

    # Resolve directories
    data_dir = os.path.abspath(args.data_dir)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print(f"Results will be saved to: {results_dir}")

    # Resolve equations_txt for units lookup
    equations_txt = args.equations_txt
    if equations_txt is None:
        default_eq = os.path.join(project_root, "data", "univariate_benchmark.txt")
        if os.path.exists(default_eq):
            equations_txt = default_eq
    if equations_txt is not None:
        print(f"Using equations file: {equations_txt}")

    # Run full pipeline on each problem
    results = []
    total_start = time.time()

    for i, stem in enumerate(sorted_stems, 1):
        info = problems[stem]
        print(f"\n[{i}/{len(sorted_stems)}] Problem {stem} (ID: {info['id']})")
        print(f"  {info['desc']}")

        filepath = find_data_file(stem, data_dir)

        if filepath is None:
            print(f"FAIL Data file not found for {stem} in {data_dir}")
            results.append(
                {
                    "stem": stem,
                    "id": info["id"],
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
            no_factorized_search=args.no_factorized_search,
            factorized_search_plus=args.factorized_search_plus,
        )
        result["id"] = info["id"]
        results.append(result)

    total_elapsed = time.time() - total_start

    # Write summary report
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(results_dir, "univariate_suite_summary.json")
    else:
        output_path = os.path.abspath(output_path)

    write_summary_report(results, output_path)

    print(f"\nTotal suite runtime: {total_elapsed / 3600:.2f} hours")

    failed = sum(1 for r in results if not r["success"])
    if failed > 0:
        print(f"\nWARNING: {failed} problems failed")
        return 1
    else:
        print("\nAll problems completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
