#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Sealed-test rollout of the Paper IV complex-valued ODE benchmark cases.

The complex benchmark scores each case structurally against its ground-truth
term list (see ``examples/feynman_complex/run_benchmark.py::validate_sparse_result``),
fitting on all six generated trajectories.  This script adds an independent
dynamical check for the ODE cases: the discovered system is rolled out, with
the same solver settings the harness uses, on two trajectories (ic6, ic7) that
no discovery step ever saw.  They are obtained by extending the protocol's
seeded initial-condition stream from six to eight draws, so ic0..ic5 are
regenerated identically (verified here) and ic6, ic7 are fresh draws from the
same distribution.  The saved system is rebuilt from the recorded
``(eq, term, coeff)`` triples in the residual convention
``anchor + sum_k c_k phi_k = 0`` with the harness's term vocabulary
(``u``, ``u1``, ... components; ``u_x0``, ``u1_x0`` first derivatives;
``x0`` the independent variable; ``const``; ``sin``/``cos``/``exp``/``log``).

PDE cases are not rolled out; their structural criterion already certifies the
operator support and coefficients, and a PDE rollout would require a solver
per case.  The algebraic case (order 0) is skipped.

Usage::

    python3 scripts/score_paper4_sealed_test_complex.py \
        --results-root ../NestyNet_paper4_recreation/runs/results

Writes ``<results-root>/sealed_test/complex_stlsq/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples" / "feynman_complex"))

PASS_NRMSE = 0.01
PARTIAL_NRMSE = 0.05

_SAFE_FUNCS = {"sin": math.sin, "cos": math.cos, "exp": math.exp, "log": math.log,
               "sqrt": math.sqrt, "abs": abs, "tanh": math.tanh}


def _normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1.0e-12))


def build_rhs(discovered: list[dict], ncomp: int, order: int):
    """Compile the saved (eq, term, coeff) triples into ``rhs(t, state)``."""
    per_eq: dict[int, list[tuple[float, object]]] = {ci: [] for ci in range(ncomp)}
    for row in discovered:
        c = float(row.get("coeff", 0.0))
        if abs(c) <= 1.0e-12:
            continue
        term = str(row["term"])
        code = compile(term, f"<term {term}>", "eval")
        per_eq[int(row["eq"])].append((c, code))

    def rhs(t: float, state) -> list[float]:
        ns = dict(_SAFE_FUNCS)
        ns["x0"] = float(t)
        ns["const"] = 1.0
        for k in range(ncomp):
            ns["u" if k == 0 else f"u{k}"] = float(state[k])
            if order == 2:
                ns["u_x0" if k == 0 else f"u{k}_x0"] = float(state[ncomp + k])
        derivs = []
        for ci in range(ncomp):
            total = 0.0
            for c, code in per_eq[ci]:
                total += c * float(eval(code, {"__builtins__": {}}, ns))
            derivs.append(-total)  # anchor = -sum c_k phi_k
        if order == 2:
            return [float(state[ncomp + k]) for k in range(ncomp)] + derivs
        return derivs

    return rhs


def rollout_status(rhs, trajs: list[dict], ncomp: int) -> tuple[str, str, list[dict]]:
    scores: list[dict] = []
    for k, traj in enumerate(trajs):
        t = np.asarray(traj["t"], dtype=np.float64)
        true = np.asarray(traj["state"], dtype=np.float64)
        try:
            sol = solve_ivp(rhs, [float(t[0]), float(t[-1])], list(map(float, traj["y0"])),
                            t_eval=t, method="RK45", rtol=1e-9, atol=1e-11)
        except Exception as exc:
            return "FAIL", f"Integration error on traj {k}: {exc}", scores
        if not sol.success or sol.y.shape[1] != t.shape[0]:
            return "FAIL", f"Integration failed on traj {k}: {sol.message}", scores
        comp = [_normalized_rmse(true[ci], sol.y[ci]) for ci in range(ncomp)]
        scores.append({"traj_id": f"ic{6 + k}", "nrmse": float(max(comp)), "comp_nrmses": comp,
                       "y0": list(map(float, traj["y0"]))})
    max_e = max(s["nrmse"] for s in scores)
    msg = f"NRMSE max={max_e:.3g} over {len(scores)} sealed trajectories"
    if max_e < PASS_NRMSE:
        return "PASS", msg, scores
    if max_e < PARTIAL_NRMSE:
        return "PARTIAL", msg, scores
    return "FAIL", msg, scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--benchmark-file", type=Path, default=REPO / "data" / "feynman_complex_benchmark.txt")
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-sealed", type=int, default=2)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import run_benchmark as rb
    from problem_defs import (ANCHOR_ORDER, DEFAULT_ICS, DEFAULT_PARAMS, DEFAULT_TMAX,
                              NCOMPONENTS, RHS_REGISTRY, load_complex_problems)

    probs = load_complex_problems(str(a.benchmark_file))
    summary_in = json.loads((a.results_root / "complex_stlsq" / "summary.json").read_text())
    by_id = {str(p["id"]): p for p in summary_in["problems"]}

    rows = []
    for pid in sorted(probs):
        p = probs[pid]
        row = {"id": pid, "description": p.description, "structural_status": by_id.get(pid, {}).get("status")}
        if pid not in RHS_REGISTRY or int(p.order) == 0:
            row.update({"status": "SKIP", "message": "PDE or algebraic case: structural criterion only"})
            rows.append(row)
            continue
        ncomp = NCOMPONENTS.get(pid, 2)
        order = int(p.order)
        params = DEFAULT_PARAMS.get(pid, {})
        seed = rb._problem_seed(int(a.seed), pid)
        common = dict(n_points=int(a.n_points), seed=seed)
        base = rb.generate_ode_multi_traj(pid, params, ncomp, order, DEFAULT_TMAX.get(pid, 10.0),
                                          DEFAULT_ICS.get(pid, {}), n_traj=int(a.n_train), **common)
        ext = rb.generate_ode_multi_traj(pid, params, ncomp, order, DEFAULT_TMAX.get(pid, 10.0),
                                         DEFAULT_ICS.get(pid, {}), n_traj=int(a.n_train + a.n_sealed), **common)
        row["training_trajectories_reproduced"] = [
            bool(np.array_equal(base[k]["state"], ext[k]["state"])) for k in range(int(a.n_train))
        ]
        sealed = ext[int(a.n_train):]
        disc = by_id.get(pid, {}).get("discovered") or []
        if not disc:
            row.update({"status": "MISSING", "message": "no discovered system in summary"})
            rows.append(row)
            continue
        try:
            rhs = build_rhs(disc, ncomp, order)
            status, message, scores = rollout_status(rhs, sealed, ncomp)
        except Exception as exc:
            status, message, scores = "ERROR", f"{type(exc).__name__}: {exc}", []
        row.update({"status": status, "message": message, "traj_scores": scores,
                    "sealed_max_nrmse": max((s["nrmse"] for s in scores), default=None),
                    "anchor_order": ANCHOR_ORDER.get(pid, order)})
        rows.append(row)
        print(f"{pid:5s} {p.description[:40]:40s} structural={row['structural_status']:8s} sealed rollout={status:8s} {message}")

    counts = Counter(r["status"] for r in rows)
    out = {"schema_version": 1, "arm": "complex_stlsq", "criterion": "sealed rollout (ODE cases only)",
           "protocol": {"fit_trajectories": list(range(a.n_train)),
                        "sealed_test_trajectories": list(range(a.n_train, a.n_train + a.n_sealed)),
                        "n_points": a.n_points, "ic_seed": a.seed, "solver": "RK45 rtol 1e-9 atol 1e-11",
                        "nrmse": "full-span rollout, max over components and trajectories",
                        "pass_nrmse": PASS_NRMSE, "partial_nrmse": PARTIAL_NRMSE},
           "status_counts": dict(counts), "problems": rows}
    d = a.results_root / "sealed_test" / "complex_stlsq"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    print("counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
