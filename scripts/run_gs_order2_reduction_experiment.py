#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Order-2 symmetry reduction feeding the STLSQ dictionary (R3b experiment).

Second-order linear-homogeneous equations with singular coefficients are the
classic fixed-dictionary failures (Lane-Emden n=1, Bessel J0, spherical
acoustic wave): the required carrier u_x/x is a *product* absent from standard
libraries.  All of them admit the scaling symmetry u d/du — discoverable from
the trajectory ensemble alone — under which the order-2 reduction produces a
first-order Riccati law dv/dr = H(r, v) in the invariants (r=x, v=u_x/u),
fitted here by STLSQ over an (r, v) dictionary.  The pullback manufactures the
second-order rows (u, u_x/x, u_x^2/u, ...) that convert the failing standard
dictionary into an exact recovery.

Derivatives u_x, u_xx are supplied exactly by the case generators, standing in
for the analytic derivatives of a trained NestyNet surrogate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nestynet_sr.sr_core.numerics import ridge_lstsq, stlsq  # noqa: E402
from nestynet_sr.sr_gs.de_reduction import symmetry_reduction_proposals  # noqa: E402
from nestynet_sr.sr_gs.prolongation import _eval_term_on_jets  # noqa: E402


@dataclass
class Order2Case:
    name: str
    description: str
    rhs: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]  # u_xx = rhs(x,u,ux)
    trajectories: list[tuple[np.ndarray, np.ndarray]]
    u1_list: list[np.ndarray]
    u2_list: list[np.ndarray]
    probe_ics: list[tuple[float, float, float, np.ndarray]]  # (x0, u0, v0, x_eval)
    truth_fn: Callable[[float, float, float, np.ndarray], np.ndarray]


def _make_singular_case(name: str, desc: str, coef_damp: float, coef_state: float) -> Order2Case:
    """u'' = -(coef_damp/x) u' - coef_state * u with regular solutions a*j0-like."""

    from scipy.integrate import solve_ivp

    def rhs(x: np.ndarray, u: np.ndarray, ux: np.ndarray) -> np.ndarray:
        return -coef_damp * ux / x - coef_state * u

    def _integrate(x0: float, u0: float, v0: float, x_eval: np.ndarray) -> np.ndarray:
        sol = solve_ivp(
            lambda t, s: [s[1], float(rhs(np.asarray([t]), np.asarray([s[0]]), np.asarray([s[1]]))[0])],
            (float(x0), float(x_eval[-1])), [float(u0), float(v0)],
            t_eval=x_eval, rtol=1e-11, atol=1e-13,
        )
        return sol.y[0]

    x = np.linspace(0.05, 2.2, 420)
    trajs, u1s, u2s = [], [], []
    rng = np.random.default_rng(3)
    for amp in (0.6, 1.0, 1.5, 0.85):
        v0 = float(rng.uniform(-0.05, 0.05)) * amp
        u = _integrate(x[0], amp, v0, x)
        u1 = np.gradient(u, x, edge_order=2)
        # refine u1 by integrating the state form for exactness
        sol = None
        from scipy.integrate import solve_ivp as _si
        sol = _si(
            lambda t, s: [s[1], float(rhs(np.asarray([t]), np.asarray([s[0]]), np.asarray([s[1]]))[0])],
            (float(x[0]), float(x[-1])), [amp, v0], t_eval=x, rtol=1e-11, atol=1e-13,
        )
        u, u1 = sol.y[0], sol.y[1]
        u2 = rhs(x, u, u1)
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(u2)
    probe = []
    x_eval = np.linspace(0.05, 2.2, 200)
    for amp in (0.75, 1.25):
        probe.append((0.05, amp, 0.0, x_eval))
    return Order2Case(name, desc, rhs, trajs, u1s, u2s, probe, _integrate)


def _make_damped_oscillator() -> Order2Case:
    from scipy.integrate import solve_ivp

    gamma, omega2 = 0.35, 1.44

    def rhs(x: np.ndarray, u: np.ndarray, ux: np.ndarray) -> np.ndarray:
        return -2.0 * gamma * ux - omega2 * u

    def _integrate(x0: float, u0: float, v0: float, x_eval: np.ndarray) -> np.ndarray:
        sol = solve_ivp(
            lambda t, s: [s[1], -2.0 * gamma * s[1] - omega2 * s[0]],
            (float(x0), float(x_eval[-1])), [float(u0), float(v0)],
            t_eval=x_eval, rtol=1e-11, atol=1e-13,
        )
        return sol.y[0]

    x = np.linspace(0.0, 1.0, 350)
    trajs, u1s, u2s = [], [], []
    for u0, v0 in ((1.0, -0.1), (0.7, 0.25), (1.4, 0.0), (0.9, 0.4)):
        sol = solve_ivp(
            lambda t, s: [s[1], -2.0 * gamma * s[1] - omega2 * s[0]],
            (0.0, float(x[-1])), [u0, v0], t_eval=x, rtol=1e-11, atol=1e-13,
        )
        u, u1 = sol.y[0], sol.y[1]
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(rhs(x, u, u1))
    x_eval = np.linspace(0.0, 1.0, 150)
    probe = [(0.0, 1.1, 0.1, x_eval), (0.0, 0.8, -0.2, x_eval)]
    return Order2Case(
        "damped_oscillator", "u'' = -0.7 u' - 1.44 u (autonomous sanity case)",
        rhs, trajs, u1s, u2s, probe, _integrate,
    )


def build_cases() -> list[Order2Case]:
    return [
        _make_singular_case("lane_emden_n1", "u'' = -(2/x) u' - u (Lane-Emden n=1)", 2.0, 1.0),
        _make_singular_case("bessel_j0", "u'' = -(1/x) u' - u (Bessel J0)", 1.0, 1.0),
        _make_singular_case("spherical_wave", "u'' = -(2/x) u' - 1.69 u (spherical acoustic, k=1.3)", 2.0, 1.69),
        _make_damped_oscillator(),
    ]


# ---------------------------------------------------------------------------
# Order-2 STLSQ over (x, u, u_x) dictionaries
# ---------------------------------------------------------------------------

@dataclass
class Term2:
    name: str
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def standard_order2_terms() -> list[Term2]:
    return [
        Term2("1", lambda x, u, ux: np.ones_like(x)),
        Term2("x", lambda x, u, ux: x),
        Term2("x^2", lambda x, u, ux: x * x),
        Term2("u", lambda x, u, ux: u),
        Term2("u^2", lambda x, u, ux: u * u),
        Term2("u^3", lambda x, u, ux: u**3),
        Term2("x*u", lambda x, u, ux: x * u),
        Term2("u_x", lambda x, u, ux: ux),
        Term2("x*u_x", lambda x, u, ux: x * ux),
        Term2("u*u_x", lambda x, u, ux: u * ux),
    ]


def ast_order2_term(term_ast: Any, name: str) -> Term2:
    def _fn(x: np.ndarray, u: np.ndarray, ux: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(x, dtype=np.float64)).reshape(-1, 1)
        ut = torch.as_tensor(np.asarray(u, dtype=np.float64)).reshape(-1, 1)
        u1t = torch.as_tensor(np.asarray(ux, dtype=np.float64)).reshape(-1, 1)
        u2t = torch.zeros_like(xt)
        val = _eval_term_on_jets(term_ast, x=xt, u=ut, u1=u1t, u2=u2t, x_axis=0)
        return np.asarray(val.detach().cpu()).reshape(-1)

    return Term2(name=name, fn=_fn)


def fit_and_rollout(
    case: Order2Case,
    terms: list[Term2],
    *,
    lambdas=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
    ridge: float = 1.0e-10,
    max_iter: int = 10,
    pass_nrmse: float = 1.0e-2,
    partial_nrmse: float = 5.0e-2,
) -> dict[str, Any]:
    from scipy.integrate import solve_ivp

    cols_parts, y_parts = [], []
    for (x, u), u1, u2 in zip(case.trajectories, case.u1_list, case.u2_list):
        cols = [t.fn(x, u, u1).reshape(-1) for t in terms]
        cols_parts.append(np.stack(cols, axis=1))
        y_parts.append(-u2)
    Phi = np.concatenate(cols_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    finite = np.isfinite(Phi).all(axis=1) & np.isfinite(y)
    Phi, y = Phi[finite], y[finite]
    Phi_t = torch.as_tensor(Phi, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)

    best: dict[str, Any] | None = None
    for lam in lambdas:
        try:
            coeffs_t, keep_t = stlsq(Phi_t, y_t, ridge=float(ridge), lam=float(lam), max_iter=int(max_iter))
            keep = keep_t.detach().cpu().numpy().astype(bool)
            if int(keep.sum()) == 0:
                continue
            refit = ridge_lstsq(Phi_t[:, keep_t], y_t, ridge=0.0).detach().cpu().numpy()
            coeffs = np.zeros(len(terms))
            coeffs[keep] = refit

            def rhs_hat(t: float, s: list[float]) -> list[float]:
                x_a = np.asarray([t]); u_a = np.asarray([s[0]]); v_a = np.asarray([s[1]])
                total = 0.0
                for i, term in enumerate(terms):
                    if keep[i]:
                        total += coeffs[i] * float(term.fn(x_a, u_a, v_a)[0])
                return [s[1], -total]

            nrmses = []
            for (x0, u0, v0, x_eval) in case.probe_ics:
                truth = case.truth_fn(x0, u0, v0, x_eval)
                try:
                    sol = solve_ivp(rhs_hat, (float(x0), float(x_eval[-1])), [u0, v0],
                                    t_eval=x_eval, rtol=1e-9, atol=1e-11)
                    if not sol.success or sol.y.shape[1] != x_eval.size:
                        nrmses.append(float("inf"))
                        continue
                    resid = sol.y[0] - truth
                    nrmses.append(float(np.sqrt(np.mean(resid**2)) / max(np.std(truth), 1e-12)))
                except Exception:
                    nrmses.append(float("inf"))
            max_nrmse = float(max(nrmses))
            status = "PASS" if max_nrmse < pass_nrmse else ("PARTIAL" if max_nrmse < partial_nrmse else "FAIL")
            rank = {"PASS": 0, "PARTIAL": 1, "FAIL": 2}[status]
            cand = {
                "lambda": float(lam), "status": status, "max_nrmse": max_nrmse,
                "selected_terms": int(keep.sum()),
                "selected_term_names": [terms[i].name for i in range(len(terms)) if keep[i]],
                "selected_coefficients": [float(coeffs[i]) for i in range(len(terms)) if keep[i]],
                "equation": "u_xx + " + " + ".join(
                    f"({coeffs[i]:.6g})*{terms[i].name}" for i in range(len(terms)) if keep[i]
                ) + " = 0",
            }
            key = (rank, max_nrmse, int(keep.sum()))
            if best is None or key < best["_key"]:
                cand["_key"] = key
                best = cand
        except Exception:
            continue
    if best is not None:
        best = {k: v for k, v in best.items() if k != "_key"}
    return best or {"status": "ERROR", "max_nrmse": float("inf")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Order-2 symmetry reduction feeding standard 2nd-order STLSQ dictionaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "gs_order2_reduction_experiment"))
    args = parser.parse_args()

    report_cases = []
    for case in build_cases():
        reduction = symmetry_reduction_proposals(
            case.trajectories, u1_list=case.u1_list, u2_list=case.u2_list, order=2
        )
        rows = list(reduction.get("library_rows", []))
        # drop injected rows that duplicate a baseline column numerically
        # (e.g. the Riccati constant term pulls back to u, already present)
        base_terms = standard_order2_terms()
        x_all = np.concatenate([t[0] for t in case.trajectories])
        u_all = np.concatenate([t[1] for t in case.trajectories])
        u1_all = np.concatenate(list(case.u1_list))
        base_cols = [t.fn(x_all, u_all, u1_all).reshape(-1) for t in base_terms]
        fresh_rows = []
        for term, source, family in rows:
            col = ast_order2_term(term, "tmp").fn(x_all, u_all, u1_all)
            finite = np.isfinite(col)
            dup = False
            for bcol in base_cols:
                both = finite & np.isfinite(bcol)
                a, b = col[both], bcol[both]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom > 0 and abs(float(a @ b)) / denom >= 0.9999:
                    dup = True
                    break
            if not dup:
                fresh_rows.append((term, source, family))
        rows = fresh_rows
        reduced_eqs = [
            rep.get("reduced_fit", {}).get("equation")
            for rep in reduction.get("reports", []) if rep.get("status") == "proposed"
        ]
        base = fit_and_rollout(case, standard_order2_terms())
        augmented = fit_and_rollout(case, standard_order2_terms() + [
            ast_order2_term(term, f"GSRED[{family}]:{term!r}") for term, _s, family in rows
        ])
        used = [n for n in augmented.get("selected_term_names", []) if n.startswith("GSRED[")]
        entry = {
            "id": case.name,
            "description": case.description,
            "reduced_equations": reduced_eqs,
            "n_injected_rows": len(rows),
            "injected_rows": [repr(t) for t, _s, _f in rows],
            "standard": base,
            "standard_plus_reduction": augmented,
            "reduction_rows_selected": used,
            "status_change": f"{base.get('status')} -> {augmented.get('status')}",
        }
        report_cases.append(entry)
        print(f"\n=== {case.name}: {case.description} ===")
        for eq in reduced_eqs:
            print(f"  reduced law: {eq}")
        print(f"  {entry['status_change']}  (max NRMSE {base.get('max_nrmse'):.3g} -> "
              f"{augmented.get('max_nrmse'):.3g}); reduction rows selected: {len(used)}")
        print(f"  eq: {augmented.get('equation')}")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gs_order2_reduction_summary.json"
    out.write_text(json.dumps({"cases": report_cases}, indent=2, allow_nan=True), encoding="utf-8")
    print(f"\nSummary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
