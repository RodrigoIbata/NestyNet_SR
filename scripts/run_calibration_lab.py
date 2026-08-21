#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
"""Run the confidence-Pareto calibration laboratory and emit JSON.

No search, no fitting, no symbolic machinery: this exercises the inference
alone on populations whose risks and covariance are known exactly.  It answers
"is the procedure calibrated", which must be settled before asking whether the
whole system stays calibrated on real problems.

    python scripts/run_calibration_lab.py --replicates 2000 --out lab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path


def _bootstrap_paths() -> None:
    """Pin this workspace and its paired NestyNet ahead of any editable install.

    Running a script puts ``scripts/`` on ``sys.path``, not the repository root,
    so a bare ``import nestynet_sr`` resolves to whichever editable install is
    active.  That is the frozen NestyNet_SR, which has no ``stat_selection``, so
    the failure is at least loud.  ``import nestynet`` fails silently instead,
    which is worse.  ``conftest.py`` handles this for pytest; scripts need it
    themselves.
    """
    root = Path(__file__).resolve().parents[1]
    base = Path(os.environ.get("NESTYNET_BASE", root.parent / "NestyNet_plus"))
    if not base.is_absolute():
        base = (root / base).resolve()
    for entry in (str(base), str(root)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


_bootstrap_paths()

import numpy as np  # noqa: E402

from nestynet_sr.stat_selection.calibration_lab import (  # noqa: E402
    cluster_calibration,
    dominance_power,
    make_lab_population,
    multiplicity_sweep,
    null_front_coverage,
    population_front,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--multiplicity-replicates", type=int, default=400)
    parser.add_argument("--units", type=int, default=24)
    parser.add_argument("--resamples", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--archive-sizes", type=int, nargs="+", default=[5, 10, 25, 50, 100]
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--only",
        choices=["null", "multiplicity", "cluster", "power", "all"],
        default="all",
    )
    args = parser.parse_args()

    # Apple Accelerate raises spurious matmul FP flags; see calibration_lab.
    warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

    report: dict = {
        "config": {
            "replicates": args.replicates,
            "units": args.units,
            "resamples": args.resamples,
            "alpha": args.alpha,
            "seed": args.seed,
            "numpy": np.__version__,
        }
    }
    started = time.time()

    if args.only in ("null", "all"):
        population = make_lab_population(seed=args.seed)
        report["population"] = {
            "n_candidates": population.n_candidates,
            "population_front_size": len(population_front(population)),
            "groups": {g: population.group.count(g) for g in sorted(set(population.group))},
            **population.metadata,
        }
        for dist in ("gaussian", "chisq"):
            key = f"null_front_coverage_{dist}"
            report[key] = null_front_coverage(
                population,
                n_units=args.units,
                n_replicates=args.replicates,
                alpha=args.alpha,
                n_resamples=args.resamples,
                seed=args.seed,
                distribution=dist,
            )
            row = report[key]
            print(
                f"[null/{dist:8s}] familywise false-edge {row['familywise_false_edge_rate']:.4f} "
                f"(nominal {args.alpha:.2f})  front coverage {row['front_coverage']:.4f} "
                f"(nominal {1 - args.alpha:.2f})"
            )

    if args.only in ("multiplicity", "all"):
        report["multiplicity_sweep"] = multiplicity_sweep(
            archive_sizes=args.archive_sizes,
            n_units=args.units,
            n_replicates=args.multiplicity_replicates,
            alpha=args.alpha,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        print("\n[multiplicity] familywise false-exclusion rate by archive size")
        print(f"  {'M':>5} {'pairs':>7} {'max-T':>8} {'pointwise':>10} {'marginalSE':>11}")
        for row in report["multiplicity_sweep"]:
            print(
                f"  {row['n_candidates']:>5} {row['n_admissible_pairs']:>7} "
                f"{row['simultaneous_max_t']:>8.4f} {row['pointwise']:>10.4f} "
                f"{row['marginal_se_rule']:>11.4f}"
            )

    if args.only in ("cluster", "all"):
        report["cluster_calibration"] = cluster_calibration(
            n_replicates=args.multiplicity_replicates,
            alpha=args.alpha,
            seed=args.seed,
        )
        print("\n[cluster] interval coverage vs sampling density (nominal "
              f"{1 - args.alpha:.2f})")
        print(f"  {'per group':>10} {'group-level':>12} {'row-level':>10}")
        for row in report["cluster_calibration"]:
            print(
                f"  {row['samples_per_group']:>10} {row['group_level_coverage']:>12.4f} "
                f"{row['row_level_coverage']:>10.4f}"
            )

    if args.only in ("power", "all"):
        report["dominance_power"] = dominance_power(
            n_units=args.units,
            n_replicates=args.multiplicity_replicates,
            alpha=args.alpha,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        print("\n[power] probability of removing a truly dominated candidate")
        for row in report["dominance_power"]:
            print(f"  gap {row['risk_gap']:>5.2f} -> {row['removal_probability']:.4f}")

    report["elapsed_seconds"] = round(time.time() - started, 1)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
