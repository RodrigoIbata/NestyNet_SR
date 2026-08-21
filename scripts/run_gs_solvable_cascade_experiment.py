#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Solvable-algebra cascade: reduce the reduced Riccati once more (R3d).

A second-order ODE with a 2-dimensional solvable point-symmetry algebra
integrates by two successive reductions.  The order-2 reduction by V1 = u d/du
sends a constant-coefficient linear equation to an AUTONOMOUS Riccati
dv/dr = -q - p v - v^2; its equilibria are the characteristic roots, so the
second reduction (by V2 = d/dr, discovered from the data-fitted reduced
equation) closes it to quadrature and the closed-form general solution follows
from the discriminant.  Variable-coefficient equations (Bessel, Lane-Emden)
keep explicit r-dependence in the reduced Riccati and the cascade correctly
declines: they are not reducible to quadrature by an affine 2D solvable algebra.

Both derivatives are supplied exactly by the case generators (standing in for
the analytic derivatives of a trained NestyNet surrogate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import j0, j1

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nestynet_sr.sr_gs.de_reduction import solvable_cascade_reduction  # noqa: E402


def _linear_ensemble(
    p: float, q: float, amplitudes=(0.6, 1.0, 1.5, 2.1), x_end: float = 1.6, n: int = 500
) -> tuple[list, list, list]:
    """u'' + p u' + q u = 0, amplitude-varied at fixed phase (v0=0)."""

    def rhs(u, v):
        return -p * v - q * u

    trajs, u1s, u2s = [], [], []
    x = np.linspace(0.02, x_end, n)
    for a in amplitudes:
        sol = solve_ivp(lambda t, s: [s[1], rhs(s[0], s[1])], (0.02, x_end), [a, 0.0],
                        t_eval=x, rtol=1e-11, atol=1e-13)
        u, u1 = sol.y[0], sol.y[1]
        u2 = rhs(u, u1)
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(u2)
    return trajs, u1s, u2s


def _cauchy_euler_ensemble(
    u_fn, up_fn, upp_fn, x0: float, x_end: float, n: int = 600, amplitudes=(0.6, 1.0, 1.5, 2.1)
) -> tuple[list, list, list]:
    """x^2 u'' + a x u' + b u = 0 (equidimensional / scale-invariant).

    A closed-form reference shape supplies exact u, u', u'' (standing in for a
    trained surrogate's analytic derivatives); amplitude scaling keeps the
    shape fixed so V1 = u d/du is an ensemble symmetry.
    """
    x = np.linspace(x0, x_end, n)
    trajs = [(x, c * u_fn(x)) for c in amplitudes]
    u1s = [c * up_fn(x) for c in amplitudes]
    u2s = [c * upp_fn(x) for c in amplitudes]
    return trajs, u1s, u2s


def _cauchy_euler_real():
    # x^2 u'' + 2 x u' - 6 u = 0 -> indicial roots 2, -3
    return _cauchy_euler_ensemble(
        lambda x: x**2 + 0.3 * x**-3, lambda x: 2 * x - 0.9 * x**-4,
        lambda x: 2 + 3.6 * x**-5, 0.3, 3.0)


def _cauchy_euler_complex():
    # x^2 u'' + x u' + u = 0 -> indicial roots +-i
    log = np.log
    return _cauchy_euler_ensemble(
        lambda x: np.cos(log(x)) + 3 * np.sin(log(x)),
        lambda x: (-np.sin(log(x)) + 3 * np.cos(log(x))) / x,
        lambda x: (-4 * np.cos(log(x)) - 2 * np.sin(log(x))) / x**2, 1.2, 10.0)


def _cauchy_euler_double():
    # x^2 u'' - 3 x u' + 4 u = 0 -> double indicial root 2
    log = np.log
    return _cauchy_euler_ensemble(
        lambda x: x**2 * (1 + 0.8 * log(x)), lambda x: x * (2.8 + 1.6 * log(x)),
        lambda x: 4.4 + 1.6 * log(x), 0.6, 4.0)


def _bessel_ensemble(x_end: float = 2.4, n: int = 500) -> tuple[list, list, list]:
    """u'' = -(1/x) u' - u, regular solutions A*J0(x) (variable-coefficient)."""

    trajs, u1s, u2s = [], [], []
    x = np.linspace(0.1, x_end, n)
    for a in (0.6, 1.0, 1.5, 2.1):
        u = a * j0(x)
        u1 = -a * j1(x)
        u2 = -u1 / x - u
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(u2)
    return trajs, u1s, u2s


def _lane_emden_n1_ensemble(x_end: float = 2.8, n: int = 500) -> tuple[list, list, list]:
    """u'' = -(2/x) u' - u, regular solutions A*sin(x)/x (variable-coefficient)."""

    trajs, u1s, u2s = [], [], []
    x = np.linspace(0.05, x_end, n)
    for a in (0.6, 1.0, 1.5, 2.1):
        u = a * np.sin(x) / x
        u1 = a * (np.cos(x) * x - np.sin(x)) / x**2
        u2 = -u - 2.0 * u1 / x
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(u2)
    return trajs, u1s, u2s


CASES: list[tuple[str, str, Callable[[], tuple[list, list, list]], str]] = [
    ("sho", "u'' = -1.69 u  (SHO, omega=1.3)", lambda: _linear_ensemble(0.0, 1.69), "positive"),
    ("exponential", "u'' = 1.44 u  (k=1.2)", lambda: _linear_ensemble(0.0, -1.44), "positive"),
    ("damped_oscillator", "u'' = -1.44 u - 0.7 u'  (underdamped)", lambda: _linear_ensemble(0.7, 1.44), "positive"),
    ("cauchy_euler_real", "x^2 u'' + 2 x u' - 6 u = 0  (Cauchy-Euler, roots 2,-3)", _cauchy_euler_real, "positive"),
    ("cauchy_euler_complex", "x^2 u'' + x u' + u = 0  (Cauchy-Euler, roots +-i)", _cauchy_euler_complex, "positive"),
    ("cauchy_euler_double", "x^2 u'' - 3 x u' + 4 u = 0  (Cauchy-Euler, double root 2)", _cauchy_euler_double, "positive"),
    ("bessel_j0", "u'' = -(1/x) u' - u  (Bessel J0, variable coeff)", _bessel_ensemble, "negative"),
    ("lane_emden_n1", "u'' = -(2/x) u' - u  (Lane-Emden n=1, variable coeff)", _lane_emden_n1_ensemble, "negative"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solvable-algebra cascade: two data-driven reductions to quadrature",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "gs_solvable_cascade_experiment"))
    args = parser.parse_args()

    report_cases = []
    for name, desc, builder, expect in CASES:
        trajs, u1s, u2s = builder()
        rep = solvable_cascade_reduction(trajs, u1_list=u1s, u2_list=u2s)
        best = rep.get("best")
        entry: dict[str, Any] = {
            "id": name,
            "description": desc,
            "expected": expect,
            "cascade_fired": bool(rep["cascade_fired"]),
            "level1_generators": rep["level1_generators"][:4],
        }
        if best:
            entry.update({
                "V1": best["V1"],
                "V1_chart": best["V1_chart"],
                "reduced_equation": best["reduced_equation"],
                "V2": best["V2"],
                "V2_kind": best["V2_kind"],
                "V2_data_confirmed": best["V2_data_confirmed"],
                "algebra_is_abelian": best["algebra_is_abelian"],
                "algebra_is_solvable": best["algebra_is_solvable"],
                "closed_form": best.get("closed_form"),
            })
        else:
            entry["decline_reason"] = (rep["attempts"][0]["reason"] if rep.get("attempts") else "no reducible V1")
            if rep.get("attempts"):
                entry["reduced_equation"] = rep["attempts"][0].get("reduced_equation")
        report_cases.append(entry)

        ok = (entry["cascade_fired"] == (expect == "positive"))
        print(f"\n=== {name}: {desc} ===")
        print(f"  expected {expect}, cascade_fired={entry['cascade_fired']}  [{'OK' if ok else 'MISMATCH'}]")
        if best:
            print(f"  V1={tuple(round(v,3) for v in best['V1'])} ({best['V1_chart']})  "
                  f"reduced: {best['reduced_equation']}")
            print(f"  V2={tuple(round(v,3) for v in best['V2'])} ({best['V2_kind']}, "
                  f"data_confirmed={best['V2_data_confirmed']}); algebra abelian={best['algebra_is_abelian']} "
                  f"solvable={best['algebra_is_solvable']}")
            cf = best.get("closed_form")
            if cf:
                print(f"  closed form: {cf['general_solution']}  [{cf['regime']}]")
        else:
            print(f"  declined: {entry['decline_reason'][:88]}")
            if entry.get("reduced_equation"):
                print(f"  reduced: {entry['reduced_equation']}")

    n_ok = sum(1 for c in report_cases if c["cascade_fired"] == (c["expected"] == "positive"))
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gs_solvable_cascade_summary.json"
    out.write_text(json.dumps({"cases": report_cases}, indent=2, allow_nan=True), encoding="utf-8")
    print(f"\nCascade classification correct in {n_ok}/{len(report_cases)} cases")
    print(f"Summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
