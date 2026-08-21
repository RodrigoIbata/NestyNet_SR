# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Aggregate Stage-B gate-margin telemetry across benchmark report JSONs.

Usage:
    python scripts/gate_margin_report.py [--band 1.5] [reports_or_dirs ...]

With no arguments, globs results/*.report.json. Prints, per (rule, gate),
the decision counts and margin distribution, then the flip-risk watchlist:
every decision whose statistic sits within --band of its threshold, i.e. the
problems whose benchmark outcome can flip under a small surrogate change.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


def _collect(paths):
    rows = []
    for path in paths:
        try:
            with open(path) as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue
        tel = report.get("gate_telemetry")
        if not tel:
            continue
        stem = os.path.basename(path).replace(".report.json", "")
        for rec in tel.get("records", []):
            rows.append({"problem": stem, **rec})
    return rows


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="report JSONs or directories")
    ap.add_argument("--band", type=float, default=1.5,
                    help="flip-risk band: margin within [1/band, band] (default 1.5)")
    args = ap.parse_args()

    paths = []
    for p in (args.paths or ["results"]):
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.report.json"))))
        else:
            paths.extend(sorted(glob.glob(p)))
    if not paths:
        print("No report JSONs found.", file=sys.stderr)
        return 1

    rows = _collect(paths)
    if not rows:
        print(f"No gate_telemetry sections in {len(paths)} report(s) "
              "(runs predate telemetry, or no instrumented gate fired).")
        return 0

    # Per-gate distribution
    by_gate = defaultdict(list)
    for r in rows:
        by_gate[(r["rule"], r["gate"])].append(r)

    print(f"{len(rows)} gate decisions from {len(paths)} report(s)\n")
    print(f"{'rule':<24} {'gate':<14} {'n':>4} {'acc':>4} "
          f"{'m_q10':>8} {'m_med':>8} {'m_q90':>8}")
    for (rule, gate), recs in sorted(by_gate.items()):
        margins = sorted(r["margin_ratio"] for r in recs
                         if isinstance(r.get("margin_ratio"), (int, float)))
        n_acc = sum(1 for r in recs if r.get("accepted"))
        fmt = lambda v: f"{v:8.3f}" if v is not None else "       -"
        print(f"{rule:<24} {gate:<14} {len(recs):>4} {n_acc:>4} "
              f"{fmt(_quantile(margins, 0.10))} {fmt(_quantile(margins, 0.50))} "
              f"{fmt(_quantile(margins, 0.90))}")

    # Flip-risk watchlist
    band = args.band
    risky = [r for r in rows
             if isinstance(r.get("margin_ratio"), (int, float))
             and r["margin_ratio"] > 0
             and (1.0 / band) <= r["margin_ratio"] <= band]
    risky.sort(key=lambda r: abs(math.log(r["margin_ratio"])))

    print(f"\nFlip-risk watchlist (margin within {band}x of threshold): "
          f"{len(risky)} decision(s)")
    for r in risky:
        ctx = r.get("context") or {}
        ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items() if k != "var_desc")
        print(f"  {r['problem']:<20} {r['rule']:<24} {r['gate']:<14} "
              f"value={r['value']:.4g} thr={r['threshold']:.4g} "
              f"margin={r['margin_ratio']:.3f} "
              f"{'ACCEPT' if r.get('accepted') else 'reject'} {ctx_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
