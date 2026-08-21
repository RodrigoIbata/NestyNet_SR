#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Symmetry-reduction proposals feeding the STLSQ dictionary (R3 experiment).

For each compositional case (900-903) plus a synthetic half-order-kinetics
case, this script:

1. discovers data-supported generators from the trajectory ensemble alone
   (exp(eps*V) flow test — no candidate equation is consulted);
2. rectifies each generator to canonical coordinates, fits the reduced
   univariate law with the factorized-search mapping families, and pulls the
   fitted law back to original coordinates;
3. injects the pulled-back terms into the *standard* STLSQ dictionary — the
   dictionary that fails or aliases on these cases — and re-runs the same
   STLSQ + rollout validation used by the dictionary-isolation audit.

The point is complementarity: the reduction manufactures exactly the
compositional carrier atoms the fixed dictionary lacks, in the same spirit as
(and as seeds for) the factorized symbolic search.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_feynman_de_stlsq_dictionary_baselines import (  # noqa: E402
    BENCHMARK_FILE,
    DATA_DIR,
    LibraryTerm,
    _choice_key,
    _load_xy,
    _mean_max_nrmse,
    build_design,
    build_library_terms,
    canonical_equation,
    fit_stlsq,
    load_existing_runs,
    load_problems,
    make_rhs,
    validate_by_simulation,
)

from nestynet_sr.sr_gs.de_reduction import symmetry_reduction_proposals  # noqa: E402
from nestynet_sr.sr_gs.prolongation import _eval_term_on_jets  # noqa: E402


def ast_library_term(term_ast: Any, name: str) -> LibraryTerm:
    """Wrap a pulled-back AST row as a dictionary term (numpy in, numpy out)."""

    def _fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(x, dtype=np.float64)).reshape(-1, 1)
        ut = torch.as_tensor(np.asarray(u, dtype=np.float64)).reshape(-1, 1)
        zeros = torch.zeros_like(xt)
        with np.errstate(all="ignore"):
            val = _eval_term_on_jets(term_ast, x=xt, u=ut, u1=zeros, u2=zeros, x_axis=0)
        return np.asarray(val.detach().cpu()).reshape(-1)

    return LibraryTerm(name=name, fn=_fn)


def evaluate_library(
    problem: Any,
    fit_runs: list[Any],
    probe_runs: list[Any],
    terms: list[LibraryTerm],
    *,
    lambdas: list[float],
    ridge: float,
    max_iter: int,
    pass_nrmse: float,
    partial_nrmse: float,
) -> dict[str, Any]:
    from run_feynman_de_stlsq_dictionary_baselines import default_param_values

    params = default_param_values(problem)
    Phi_fit, y_fit, _ = build_design(problem, fit_runs, terms, params=params)
    candidates = []
    for lam in lambdas:
        try:
            coeffs, keep = fit_stlsq(Phi_fit, y_fit, lam=float(lam), ridge=ridge, max_iter=max_iter)
            rhs_fn = make_rhs(terms, coeffs, keep)
            status, message, traj_scores = validate_by_simulation(
                probe_runs, rhs_fn=rhs_fn, order=1,
                pass_nrmse=pass_nrmse, partial_nrmse=partial_nrmse,
                traj_time_budget_s=20.0,
            )
            mean_nrmse, max_nrmse = _mean_max_nrmse(traj_scores)
            candidates.append({
                "lambda": float(lam),
                "status": str(status),
                "mean_nrmse": float(mean_nrmse),
                "max_nrmse": float(max_nrmse),
                "fit_rms": float(np.sqrt(np.mean((Phi_fit @ coeffs - y_fit) ** 2))),
                "selected_terms": int(keep.sum()),
                "selected_term_names": [terms[i].name for i in range(len(terms)) if keep[i]],
                "selected_coefficients": [float(coeffs[i]) for i in range(len(terms)) if keep[i]],
                "canonical_equation": canonical_equation(terms, coeffs, keep),
            })
        except Exception as exc:
            candidates.append({
                "lambda": float(lam), "status": "ERROR", "message": str(exc)[:200],
                "mean_nrmse": float("inf"), "max_nrmse": float("inf"),
                "fit_rms": float("inf"), "selected_terms": 0,
                "selected_term_names": [], "selected_coefficients": [],
                "canonical_equation": "",
            })
    best = min(candidates, key=_choice_key)
    return {"selected": best, "candidates": candidates}


def run_benchmark_case(pid: str, args: argparse.Namespace) -> dict[str, Any]:
    problems = load_problems(str(args.benchmark_file))
    problem = problems[pid]
    runs, _src = load_existing_runs(problem, Path(args.data_dir).resolve(), n_traj=int(args.n_traj))
    holdout = int(args.holdout_last_k)
    fit_runs, probe_runs = runs[:-holdout], runs[-holdout:]
    trajectories = []
    for run in fit_runs:
        x, u = _load_xy(run)
        trajectories.append((np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64)))

    reduction = symmetry_reduction_proposals(trajectories)
    rows = list(reduction.get("library_rows", []))

    lambdas = [float(v) for v in str(args.lambdas).split(",")]
    common = dict(lambdas=lambdas, ridge=float(args.ridge), max_iter=int(args.stlsq_max_iter),
                  pass_nrmse=float(args.pass_nrmse), partial_nrmse=float(args.partial_nrmse))
    standard_terms = build_library_terms("standard", pid)
    base = evaluate_library(problem, fit_runs, probe_runs, standard_terms, **common)
    reduction_terms = standard_terms + [
        ast_library_term(term, f"GSRED[{family}]:{term!r}") for term, _s, family in rows
    ]
    augmented = evaluate_library(problem, fit_runs, probe_runs, reduction_terms, **common)

    used = [n for n in augmented["selected"]["selected_term_names"] if n.startswith("GSRED[")]
    return {
        "id": pid,
        "description": str(problem.description),
        "reduction_reports": reduction.get("reports"),
        "n_injected_rows": len(rows),
        "injected_rows": [repr(t) for t, _s, _f in rows],
        "standard": base["selected"],
        "standard_plus_reduction": augmented["selected"],
        "reduction_rows_selected": used,
        "status_change": f"{base['selected']['status']} -> {augmented['selected']['status']}",
    }


def run_half_order_case(args: argparse.Namespace) -> dict[str, Any]:
    """Synthetic u' = -k sqrt(u): a continuous-exponent case the fixed library lacks."""

    k = 0.6
    trajs = []
    for u0 in (1.2, 1.8, 2.6, 3.5):
        x_end = 0.85 * 2.0 * np.sqrt(u0) / k
        x = np.linspace(0.0, x_end, 400)
        trajs.append((x, (np.sqrt(u0) - 0.5 * k * x) ** 2))
    probe_u0 = (1.5, 3.0)

    reduction = symmetry_reduction_proposals(trajs)
    rows = list(reduction.get("library_rows", []))

    standard_terms = build_library_terms("standard", "900")

    def _fit_and_roll(terms: list[LibraryTerm]) -> dict[str, Any]:
        Phi_parts, y_parts = [], []
        for x, u in trajs:
            du = -k * np.sqrt(u)
            cols = [t.fn(x, u).reshape(-1) for t in terms]
            Phi_parts.append(np.stack(cols, axis=1))
            y_parts.append(-du)
        Phi = np.concatenate(Phi_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        finite = np.isfinite(Phi).all(axis=1) & np.isfinite(y)
        Phi, y = Phi[finite], y[finite]
        best = None
        for lam in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
            coeffs, keep = fit_stlsq(Phi, y, lam=lam, ridge=float(args.ridge), max_iter=int(args.stlsq_max_iter))
            rhs_fn = make_rhs(terms, coeffs, keep)
            # probe rollout on held-out initial conditions
            from scipy.integrate import solve_ivp
            nrmses = []
            for u0 in probe_u0:
                x_end = 0.8 * 2.0 * np.sqrt(u0) / k
                x_eval = np.linspace(0.0, x_end, 200)
                truth = (np.sqrt(u0) - 0.5 * k * x_eval) ** 2
                try:
                    sol = solve_ivp(lambda t, s: rhs_fn(t, s), (0.0, x_end), [u0],
                                    t_eval=x_eval, rtol=1e-8, atol=1e-10)
                    if not sol.success or sol.y.shape[1] != x_eval.size:
                        nrmses.append(float("inf"))
                        continue
                    resid = sol.y[0] - truth
                    nrmses.append(float(np.sqrt(np.mean(resid ** 2)) / max(np.std(truth), 1e-12)))
                except Exception:
                    nrmses.append(float("inf"))
            max_nrmse = float(max(nrmses))
            status = "PASS" if max_nrmse < float(args.pass_nrmse) else (
                "PARTIAL" if max_nrmse < float(args.partial_nrmse) else "FAIL")
            row = {
                "lambda": float(lam), "status": status, "max_nrmse": max_nrmse,
                "selected_terms": int(keep.sum()),
                "selected_term_names": [terms[i].name for i in range(len(terms)) if keep[i]],
                "selected_coefficients": [float(coeffs[i]) for i in range(len(terms)) if keep[i]],
                "canonical_equation": canonical_equation(terms, coeffs, keep),
                "mean_nrmse": float(np.mean([v for v in nrmses if np.isfinite(v)] or [np.inf])),
                "fit_rms": float(np.sqrt(np.mean((Phi @ coeffs - y) ** 2))),
            }
            if best is None or _choice_key(row) < _choice_key(best):
                best = row
        return best

    base = _fit_and_roll(standard_terms)
    augmented = _fit_and_roll(standard_terms + [
        ast_library_term(term, f"GSRED[{family}]:{term!r}") for term, _s, family in rows
    ])
    used = [n for n in augmented["selected_term_names"] if n.startswith("GSRED[")]
    return {
        "id": "half_order_synthetic",
        "description": "u' = -0.6*sqrt(u) (continuous exponent, absent from fixed library)",
        "reduction_reports": reduction.get("reports"),
        "n_injected_rows": len(rows),
        "injected_rows": [repr(t) for t, _s, _f in rows],
        "standard": base,
        "standard_plus_reduction": augmented,
        "reduction_rows_selected": used,
        "status_change": f"{base['status']} -> {augmented['status']}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Symmetry-reduction rows feeding the standard STLSQ dictionary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark_file", default=str(BENCHMARK_FILE))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--ids", default="900,901,902,903")
    parser.add_argument("--n_traj", type=int, default=6)
    parser.add_argument("--holdout_last_k", type=int, default=2)
    parser.add_argument("--lambdas", default="1e-8,1e-6,1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--ridge", type=float, default=1.0e-10)
    parser.add_argument("--stlsq_max_iter", type=int, default=10)
    parser.add_argument("--pass_nrmse", type=float, default=1.0e-2)
    parser.add_argument("--partial_nrmse", type=float, default=5.0e-2)
    parser.add_argument("--skip_half_order", action="store_true")
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "gs_reduction_experiment"))
    args = parser.parse_args()

    cases = []
    for pid in [s.strip() for s in str(args.ids).split(",") if s.strip()]:
        case = run_benchmark_case(pid, args)
        cases.append(case)
    if not args.skip_half_order:
        cases.append(run_half_order_case(args))

    print("\n================ summary ================")
    for case in cases:
        std = case["standard"]
        aug = case["standard_plus_reduction"]
        print(f"de{case['id']}: {case['status_change']}"
              f"  (max NRMSE {std['max_nrmse']:.3g} -> {aug['max_nrmse']:.3g})")
        print(f"    injected: {case['n_injected_rows']} rows; selected from reduction: "
              f"{len(case['reduction_rows_selected'])}")
        print(f"    eq: {aug['canonical_equation']}")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gs_reduction_experiment_summary.json"
    out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cases": cases,
    }, indent=2, allow_nan=True), encoding="utf-8")
    print(f"\nSummary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
