#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Full-pipeline A/B probe: legacy Stage-A detectors vs the GS layer.

For each target ``y = g(z(x))`` this runs the complete pipeline (``run_SR.py``)
twice, with the generalized-symmetry Stage-A layer disabled and enabled, and
reports which arm recovers the expression. It is a *probing tool* for locating
targets where GS adds full-pipeline coverage; it does not presume such targets
exist.

Findings so far (2026-07-07): the baseline detector menu is strong enough that
these synthetic carriers do NOT separate the arms. The Minkowski target is
solved by BOTH arms (the legacy difference-family probes assemble
``x0^2 - x2^2 - (x1^2 + x3^2)`` in a single Stage-A step), and the mixed-power
warp target is solved by NEITHER (a partial radial ``x0^2 + x2^2`` is promoted
first and the atom is reduced before the full coordinate can form). The
demonstrated SR-side GS value therefore lives in the oracle-mode carrier-seed
experiment (``scripts/run_gs_fss_carrier_seed_experiment.py``) and the
deterministic smoke benchmark
(``examples/generalized_symmetries/gs_smoke_benchmark.py``), not in
full-pipeline rescues.

Heavy compute: each cell is a full surrogate-training + Stage-A/B run
(~10-20 min). Intended to be run by the user, not inline.

    python scripts/run_gs_coordinate_discovery_experiment.py [--only minkowski] [--timeout 1800]

By default Stage B runs without factorized search; pass --factorized-search to
put FSS in both arms for the strongest baseline.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Candidate carriers for probing coverage differences between the arms (see the
# module docstring for measured findings on these targets).
# (name, sympy target expr, [(var, lo, hi)])
TARGETS = [
    ("mixed_power_warp", "sin(x0**2 + x1**3 + x2**2)",
     [("x0", 0.6, 1.8), ("x1", 0.6, 1.8), ("x2", 0.6, 1.8)]),
    ("minkowski", "sin(x0**2 - x1**2 - x2**2 - x3**2)",
     [("x0", 0.5, 2.5), ("x1", 0.5, 1.5), ("x2", 0.5, 1.5), ("x3", 0.5, 1.5)]),
    ("euclidean", "sin((x0 - x1)**2 + (x2 - x3)**2)",
     [("x0", 0.5, 2.5), ("x1", 0.5, 2.5), ("x2", 0.5, 2.5), ("x3", 0.5, 2.5)]),
]

# Broad GS Stage-A configuration: all charts + composition + boosts, so one flag
# set discovers whichever coordinate family a target needs.
GS_ON_FLAGS = [
    "--gs-stagea", "--gs-general-affine",
    "--gs-charts", "identity,log,reciprocal,warp",
    "--gs-pairwise-composition", "--gs-lorentz-boosts",
    "--gs-noise-calibrated-promotion",
]

SOLVE_MSE = 1e-8


def make_csv(path, expr, variables, n, seed=0):
    import numpy as np
    import sympy as sp

    rng = np.random.default_rng(seed)
    syms = [sp.Symbol(v, real=True) for v, _, _ in variables]
    fn = sp.lambdify(syms, sp.sympify(expr), modules="numpy")
    cols = [rng.uniform(lo, hi, n) for _, lo, hi in variables]
    y = fn(*cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y"] + [v for v, _, _ in variables])
        for i in range(n):
            w.writerow([y[i]] + [c[i] for c in cols])


def run_arm(csv_path, results_dir, gs_flags, use_fss, timeout):
    os.makedirs(results_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(REPO, "nestynet_sr", "run_SR.py"),
           "--filepath", csv_path, "--ignore_units", "--results_dir", results_dir]
    cmd += (["--factorized-search"] if use_fss else ["--no-factorized-search"])
    cmd += gs_flags
    log = os.path.join(results_dir, "run.log")
    t0 = time.perf_counter()
    with open(log, "w") as f:
        try:
            subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
    return status, time.perf_counter() - t0


def parse_result(results_dir, stem):
    reports = glob.glob(os.path.join(results_dir, f"{stem}*.report.json"))
    if not reports:
        return {"solved": False, "expr": "(no report)", "val_mse": float("nan"), "num_nn": None}
    d = json.load(open(reports[0]))
    fp = (d.get("final_polish") or {}).get("recommended") or {}
    val_mse = fp.get("val_mse")
    if val_mse is None:
        val_mse = ((d.get("stageC") or {}).get("val_mse"))
    expr = fp.get("display_expr") or (d.get("stageC") or {}).get("y_expr_str") or "(?)"
    num_nn = ((d.get("stageB") or {}).get("num_nn_atoms"))
    solved = (val_mse is not None and float(val_mse) < SOLVE_MSE and (num_nn == 0 or num_nn is None))
    return {"solved": bool(solved), "expr": str(expr), "val_mse": val_mse, "num_nn": num_nn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these target names")
    ap.add_argument("--n", type=int, default=5000, help="samples per target")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-run wall cap (s)")
    ap.add_argument("--factorized-search", action="store_true", help="enable FSS in both arms")
    ap.add_argument("--outdir", default="results/gs_coordinate_discovery")
    args = ap.parse_args(argv)

    targets = [t for t in TARGETS if (args.only is None or t[0] in set(args.only))]
    outdir = os.path.join(REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    print(f"{'target':18s} {'legacy (GS off)':>16s} {'GS on':>16s}   verdict")
    print("-" * 74)
    for name, expr, variables in targets:
        csv_path = os.path.join(outdir, f"{name}.csv")
        make_csv(csv_path, expr, variables, args.n)
        row = {"target": name, "expr": expr}
        for arm, flags in (("gsoff", []), ("gson", GS_ON_FLAGS)):
            rd = os.path.join(outdir, f"{name}_{arm}")
            status, secs = run_arm(csv_path, rd, flags, args.factorized_search, args.timeout)
            res = parse_result(rd, name)
            res["status"], res["seconds"] = status, secs
            row[arm] = res
        off, on = row["gsoff"], row["gson"]
        verdict = ("GS RESCUE" if (on["solved"] and not off["solved"])
                   else "both solve" if on["solved"] and off["solved"]
                   else "neither" if not on["solved"] else "check")
        rows.append(row)
        print(f"{name:18s} {('SOLVED' if off['solved'] else 'fail'):>16s} "
              f"{('SOLVED' if on['solved'] else 'fail'):>16s}   {verdict}")
        print(f"{'':18s}   off: {off['expr'][:40]}\n{'':18s}   on:  {on['expr'][:40]}")

    json.dump({"solve_mse": SOLVE_MSE, "rows": rows}, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(outdir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
