#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Isolated post-run scorer for blinded benchmark runs.

A blinded search (``run_SR.py --blinded``) never opens the ground-truth answer
key, so its ``*.report.json`` files carry the discovered expression but no
truth evaluation.  This script is the separate, post-hoc evaluator: it reads
each report, extracts the discovered expression, and tests numerical recovery
against the answer key (``aif_canaries.json``) via
``truth_eval.evaluate_canary``.

It deliberately runs *un*-blinded (it clears ``NESTYNET_SR_BLINDED``): scoring is
exactly the step that is allowed to see the answer, and keeping it in a separate
process is what makes the blinding auditable.

Usage:
    python scripts/score_blinded_run.py results/            # score all reports
    python scripts/score_blinded_run.py results/ --only pb003 pb005
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


# Scoring is the step that is allowed to read the answer key: make sure we are
# not ourselves in blinded mode, whatever the caller's environment.
os.environ.pop("NESTYNET_SR_BLINDED", None)


# Report fields that may hold the final discovered expression, in preference
# order (post-polish recommendation first, then the Stage-C/Stage-B y-space
# expression). All are sympy-parseable strings.
_EXPR_PATHS = [
    ("final_selection", "expr"),
    ("final_polish", "recommended", "expr"),
    ("stageC", "y_expr_str"),
    ("stageC", "phi_expr_str"),
    ("stageB", "y_expr_str"),
    ("stageB", "phi_expr_str"),
]


def _dig(d: dict, path: tuple):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, str) and cur.strip() else None


def discovered_expression(report: dict) -> str | None:
    from nestynet_sr.run_sr_reports import _report_final_selection_eligibility

    eligible, _reason = _report_final_selection_eligibility(report)
    if not eligible:
        return None
    for path in _EXPR_PATHS:
        v = _dig(report, path)
        if v is not None:
            return v
    return None


def _noise_level(report: dict, results_dir: str = "") -> float | None:
    candidates = (
        str(((report.get("metadata") or {}).get("dataset")) or ""),
        str(results_dir),
    )
    for candidate in candidates:
        for pattern in (r"noise_(\d+(?:\.\d+)?)", r"SRBench_(\d+(?:\.\d+)?)"):
            match = re.search(pattern, candidate)
            if match:
                return float(match.group(1))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="Directory of <stem>.report.json files")
    ap.add_argument("--only", nargs="+", default=None, help="Restrict to these problem ids (e.g. pb003)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--exact-rmse-rel",
        type=float,
        default=1.0e-8,
        help="Maximum relative RMSE classified as exact recovery (default: 1e-8)",
    )
    args = ap.parse_args(argv)

    from nestynet_sr.run_sr_reports import _report_final_selection_eligibility
    from nestynet_sr.sr_core.coefficient_metadata import coefficient_symbol_values
    from nestynet_sr.sr_search.truth_eval import evaluate_canary

    reports = sorted(glob.glob(os.path.join(args.results_dir, "*.report.json")))
    if args.only:
        keep = set(args.only)
        reports = [p for p in reports if any(os.path.basename(p).startswith(k) for k in keep)]
    if not reports:
        print(f"No report.json files found in {args.results_dir}", file=sys.stderr)
        return 2

    n_total = n_scored = n_success = 0
    for path in reports:
        stem = os.path.basename(path)[: -len(".report.json")]
        n_total += 1
        try:
            report = json.load(open(path))
        except Exception as e:
            print(f"  {stem:32s} UNREADABLE ({e})")
            continue
        eligible, ineligible_reason = _report_final_selection_eligibility(report)
        if not eligible:
            n_scored += 1
            print(
                f"  ✗ {stem:32s} INELIGIBLE "
                f"({ineligible_reason or 'no eligible final selection'})"
            )
            continue
        expr = discovered_expression(report)
        if expr is None:
            print(f"  {stem:32s} NO EXPRESSION in report")
            continue
        stagec = report.get("stageC") or {}
        stageb = report.get("stageB") or {}
        final_polish = report.get("final_polish") or {}
        final_selection = report.get("final_selection") or {}
        try:
            coefficient_metadata = next(
                (
                    payload
                    for payload in (
                        final_selection.get("coefficient_metadata"),
                        final_polish.get("coefficient_metadata"),
                        stagec.get("coefficient_metadata"),
                        stageb.get("coefficient_metadata"),
                    )
                    if payload is not None
                ),
                None,
            )
            coefficient_values = coefficient_symbol_values(
                coefficient_metadata
            )
        except Exception as exc:
            n_scored += 1
            print(f"  ✗ {stem:32s} INVALID COEFFICIENT METADATA ({exc})")
            continue
        truth_kwargs = {
            "dataset_stem": stem,
            "discovered_expr_str": expr,
            "verbose": args.verbose,
        }
        if coefficient_values:
            truth_kwargs["symbol_values"] = coefficient_values
        result = evaluate_canary(**truth_kwargs)
        n_scored += 1
        eval_ok = bool((result or {}).get("success"))
        rmse_rel = (result or {}).get("rmse_rel")
        exact = bool(
            eval_ok
            and rmse_rel is not None
            and float(rmse_rel) <= float(args.exact_rmse_rel)
        )
        noise_level = _noise_level(report, args.results_dir)
        failed = bool(not eval_ok or (noise_level == 0.0 and not exact))
        n_success += int(exact)
        mark = "✓" if exact else ("✗" if failed else "?")
        status = (
            "EXACT"
            if exact
            else (
                "evaluation failed"
                if not eval_ok
                else "noiseless not exact"
                if noise_level == 0.0
                else "noisy/unknown exactness indeterminate"
            )
        )
        print(f"  {mark} {stem:32s} {status}")

    print("-" * 60)
    print(f"Scored {n_scored}/{n_total} reports; exact recoveries: {n_success}/{n_scored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
