#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""FSS alone vs FSS + GS carrier seed (SR GS -> FSS bridge).

Each target ``y = g(z(x))`` has an internal coordinate ``z`` that the
factorized-search skeleton enumeration struggles to assemble, but which the
generalized-symmetry layer (charts / composition / warp) discovers from the
gradient geometry. Given the coordinate, the FSS outer-map battery fits ``g(z)``
in closed form. This script runs both arms at a matched budget and reports the
solve status, mimicking a paper table.

    python scripts/run_gs_fss_carrier_seed_experiment.py [--n_iter 2500] [--wall 120]

A negative control (no clean symmetry) checks that GS declines, so FSS+GS does
not spuriously beat FSS-alone there.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import torch

from nestynet_sr.sr_search.factorized_search.oracle_lab import (
    default_oracle_hyperparams,
    equation_spec_from_dict,
    run_oracle_equation,
)

HEROES = [
    ("minkowski", "sin(x0**2 - x1**2 - x2**2 - x3**2)",
     [("x0", [0.5, 2.5]), ("x1", [0.5, 1.5]), ("x2", [0.5, 1.5]), ("x3", [0.5, 1.5])]),
    ("monomial", "sin(x0*x1*x2/x3)",
     [("x0", [0.7, 2.5]), ("x1", [0.7, 2.5]), ("x2", [0.7, 2.5]), ("x3", [0.7, 2.5])]),
    ("euclidean", "sin((x0-x1)**2 + (x2-x3)**2)",
     [("x0", [0.5, 2.5]), ("x1", [0.5, 2.5]), ("x2", [0.5, 2.5]), ("x3", [0.5, 2.5])]),
    ("mixed_power_warp", "sin(x0**2 + x1**3 + x2**2)",
     [("x0", [0.6, 1.8]), ("x1", [0.6, 1.8]), ("x2", [0.6, 1.8])]),
    # negative control: no single internal coordinate (a product-trig sum), so
    # GS should decline and hand FSS no seed -> no spurious rescue.
    ("control_nonsep", "sin(x0)*x1 + x2*cos(x3)",
     [("x0", [0.6, 1.8]), ("x1", [0.6, 1.8]), ("x2", [0.6, 1.8]), ("x3", [0.6, 1.8])]),
]

SOLVE_MSE = 1e-8


def _spec(name, expr, variables):
    return equation_spec_from_dict({
        "id": name,
        "basis": ["L", "T", "M"],
        "variables": [{"name": n, "bounds": b, "dim": [0, 0, 0]} for n, b in variables],
        "constants": [],
        "target": {"expr": expr, "dim": [0, 0, 0]},
    })


def _run(spec, hp, *, gs):
    t0 = time.perf_counter()
    report = run_oracle_equation(
        spec, factorized_search_hp=hp, dtype=torch.float64,
        enforce_dims=False, verbose=False, gs_carrier_seed=gs,
    )
    dt = time.perf_counter() - t0
    best = report.get("best") or {}
    return {
        "mse": float(best.get("mse", float("nan"))),
        "expr": str(best.get("expr", "")),
        "mapping": str(best.get("mapping_kind", "")),
        "seconds": dt,
        "seeds": [d.get("z_human") for d in report.get("gs_carrier_seed_diagnostics", [])],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_iter", type=int, default=2500)
    ap.add_argument("--wall", type=float, default=120.0, help="per-run wall-time cap (s)")
    ap.add_argument("--n_fit", type=int, default=512)
    ap.add_argument("--output", default="results/gs_fss_carrier_seed_experiment/summary.json")
    args = ap.parse_args(argv)

    hp = default_oracle_hyperparams()
    hp = dataclasses.replace(
        hp, n_iter=int(args.n_iter), n_fit=int(args.n_fit), n_probe=int(args.n_fit),
        n_seeds=1, wall_time_limit_s=float(args.wall),
    )

    rows = []
    print(f"{'target':18s} {'FSS alone':>12s} {'FSS+GS':>12s}   verdict")
    print("-" * 72)
    for name, expr, variables in HEROES:
        spec = _spec(name, expr, variables)
        alone = _run(spec, hp, gs=False)
        withgs = _run(spec, hp, gs=True)
        rows.append({"target": name, "expr": expr, "alone": alone, "with_gs": withgs})
        control = name.startswith("control")
        a_ok, g_ok = alone["mse"] < SOLVE_MSE, withgs["mse"] < SOLVE_MSE
        if control:
            # the honesty property is that GS declines (no seed), so it cannot
            # spuriously rescue a target with no discoverable coordinate.
            verdict = "OK (GS declined, 0 seeds)" if not withgs["seeds"] else f"SPURIOUS SEED {withgs['seeds']}"
        else:
            verdict = "GS RESCUE" if (g_ok and not a_ok) else ("both solve" if g_ok and a_ok else "check")
        print(f"{name:18s} {alone['mse']:>12.2e} {withgs['mse']:>12.2e}   {verdict}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"solve_mse": SOLVE_MSE, "n_iter": int(args.n_iter), "rows": rows},
              open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
