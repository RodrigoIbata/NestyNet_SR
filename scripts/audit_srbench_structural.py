# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Structural-win audit for SRBench capsules (Paper III, bracketed metric).

A *structural win* is decided operationally, with no symbolic-equivalence
judgment anywhere in the verdict path:

1. take the recovered expression for the problem (the campaign's selected
   solution, as recorded in the capsule's ``summary.csv``),
2. take the canonical NOISELESS benchmark data for the same problem
   (``SRBench_0.000``-style ``pb*_data.csv``),
3. refit every free coefficient on that data,
4. re-snap the coefficients with the pipeline's own arsenal,
5. accept iff the polished expression predicts the noiseless data at the
   noiseless-fit floor (default relative RMSE <= 1e-10; the worst noiseless
   exact fit in the reference campaign is pb116 at 2.4e-13).

Steps 3-5 are exactly ``nestynet_sr.equation_polisher.polish_expression``
run at its shipped defaults, so the criterion is pinned to the code
revision, deterministic, and rerunnable by anyone.

Problems whose capsule row already carries the campaign's exact-recovery
verdict (``truth_exact``) count as ``exact``; the bracketed table entry is
``exact + structural``.

    python3 audit_srbench_structural.py capsules/noise0.100_ndata2k \\
        --noiseless-data /path/to/SRBench_0.000/data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _structural_verdict import (  # noqa: E402
    DEFAULT_TOL,
    load_problem_csv as _load_problem_csv,
    structural_verdict,
)


def _audit_one(expr: str, data_csv: Path, *, tol: float) -> tuple[str, dict]:
    X, y, var_names = _load_problem_csv(data_csv)
    ok, detail = structural_verdict(expr, X, y, variable_names=var_names, tol=tol)
    detail["route"] = "refit_snap_polish"
    return ("structural" if ok else "none"), detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path, help="Capsule dir (or a workspace results dir)")
    parser.add_argument("--noiseless-data", type=Path, required=True,
                        help="Directory with the canonical noiseless pb*_data.csv files")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated pb ids")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.capsule.expanduser().resolve()
    summary_csv = root / "summary.csv"
    if not summary_csv.is_file():
        raise SystemExit(
            f"no summary.csv in {root}; run the workspace's scripts/summarize.sh "
            "before collecting the capsule (the CSV carries the recovered "
            "expressions and the campaign's exact-recovery verdicts)"
        )
    data_dir = args.noiseless_data.expanduser().resolve()
    wanted = None if not args.only else {p.strip() for p in args.only.split(",")}

    per_problem: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    for row in csv.DictReader(summary_csv.open()):
        problem = str(row.get("problem", "")).strip()
        m = re.match(r"pb(\d+)", problem)
        if not m:
            continue
        pid = m.group(1)
        if wanted is not None and pid not in wanted and f"pb{pid}" not in wanted:
            continue
        recovered = (row.get("expression") or "").strip()
        exact = str(row.get("truth_exact", "")).strip().lower() in ("yes", "true", "1")
        if exact:
            verdict, detail = "exact", {"route": "campaign_truth_exact"}
        elif not recovered:
            verdict, detail = "none", {"route": "no_recovered_expression"}
        else:
            data_csv = data_dir / f"{problem}_data.csv"
            if not data_csv.is_file():
                verdict, detail = "none", {"route": "missing_noiseless_data",
                                           "wanted": data_csv.name}
            else:
                try:
                    verdict, detail = _audit_one(recovered, data_csv, tol=args.tol)
                except Exception as exc:
                    verdict, detail = "none", {"route": "audit_error",
                                               "error": str(exc)[:160]}
        counts[verdict] += 1
        per_problem[pid] = {"problem": problem, "verdict": verdict,
                            "expr": recovered, **detail}
        print(f"pb{pid}: {verdict}"
              + (f" (rel={detail.get('polish_rel_rmse'):.2e})"
                 if detail.get("polish_rel_rmse") is not None else ""),
              flush=True)

    structural_wins = counts["exact"] + counts["structural"]
    payload = {
        "capsule": root.name,
        "criterion": "refit + pipeline snap arsenal (equation_polisher defaults) "
                     f"on canonical noiseless data; rel RMSE <= {args.tol:g}",
        "tol": args.tol,
        "counts": dict(counts),
        "exact": counts["exact"],
        "structural_wins_total": structural_wins,
        "per_problem": per_problem,
    }
    out = args.out or (root / "structural_audit.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{root.name}: exact {counts['exact']}, structural-only {counts['structural']}, "
          f"none {counts['none']} -> structural wins {structural_wins}/{sum(counts.values())}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
