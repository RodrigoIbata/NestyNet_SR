# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FEYNMAN_DE_DIR = REPO_ROOT / "examples" / "feynman_de"
if str(FEYNMAN_DE_DIR) not in sys.path:
    sys.path.insert(0, str(FEYNMAN_DE_DIR))

import problem_defs as pd  # noqa: E402


def _problems():
    return pd.load_problems(REPO_ROOT / "data" / "feynman_de_benchmark.txt")


def test_resolve_rhs_prefers_registry_then_compiles():
    problems = _problems()

    fn0, src0 = pd.resolve_rhs(problems["000"], prefer_manual=True)
    assert src0 == "registry"
    out0 = fn0(0.2, [1.0], pd.default_param_values(problems["000"]))
    assert len(out0) == 1
    assert math.isfinite(float(out0[0]))

    fn14, src14 = pd.resolve_rhs(problems["014"], prefer_manual=True)
    assert src14 == "compiled"
    out14 = fn14(0.2, [1.0], pd.default_param_values(problems["014"]))
    assert len(out14) == 1
    assert math.isfinite(float(out14[0]))


def test_compiled_rhs_handles_symbol_names_I_E_and_lambda():
    problems = _problems()

    # Parameter named "lambda" should compile via dummify=True.
    p000 = problems["000"]
    fn000, src000 = pd.resolve_rhs(p000, prefer_manual=False)
    assert src000 == "compiled"
    params000 = pd.default_param_values(p000)
    out000 = fn000(0.1, [2.0], params000)
    assert len(out000) == 1
    assert math.isclose(float(out000[0]), -params000["lambda"] * 2.0, rel_tol=1e-10, abs_tol=1e-10)

    # Dependent variable name "I" must be treated as a symbol, not imaginary unit.
    p006 = problems["006"]  # dI/dt=-R*I/L
    fn006, src006 = pd.resolve_rhs(p006, prefer_manual=False)
    assert src006 == "compiled"
    params006 = pd.default_param_values(p006)
    out006 = fn006(0.2, [1.5], params006)
    expected006 = -params006["R"] * 1.5 / params006["L"]
    assert math.isclose(float(out006[0]), float(expected006), rel_tol=1e-10, abs_tol=1e-10)

    # Parameter name "E" must be treated as a symbol, not Euler's number.
    p119 = problems["119"]  # ...*(V-E)*psi
    fn119, src119 = pd.resolve_rhs(p119, prefer_manual=False)
    assert src119 == "compiled"
    params119 = pd.default_param_values(p119)
    s = [1.2, -0.4]
    out119 = fn119(0.3, s, params119)
    expected119 = 2.0 * params119["m"] / (params119["hbar"] ** 2) * (params119["V"] - params119["E"]) * s[0]
    assert len(out119) == 2
    assert math.isclose(float(out119[0]), s[1], rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(float(out119[1]), float(expected119), rel_tol=1e-10, abs_tol=1e-10)


def test_compiled_rhs_handles_fractional_power_domain_safely():
    problems = _problems()
    fn, src = pd.resolve_rhs(problems["014"], prefer_manual=False)
    assert src == "compiled"

    # c < 0 can happen numerically near the boundary; compiler should stay finite.
    out = fn(0.3, [-1.0e-9], pd.default_param_values(problems["014"]))
    assert len(out) == 1
    assert math.isfinite(float(out[0]))
    assert abs(float(out[0])) < 1.0e-6


def test_compiled_second_order_rhs_uses_du_term():
    problems = _problems()
    p103 = problems["103"]  # d2x/dt2=-2*gamma*dx/dt-omega0**2*x
    fn, src = pd.resolve_rhs(p103, prefer_manual=False)
    assert src == "compiled"

    params = pd.default_param_values(p103)
    state = [1.3, -0.7]
    out = fn(0.25, state, params)
    assert len(out) == 2
    assert math.isclose(float(out[0]), float(state[1]), rel_tol=0.0, abs_tol=1.0e-12)

    expected = -2.0 * params["gamma"] * state[1] - params["omega0"] ** 2 * state[0]
    assert math.isclose(float(out[1]), float(expected), rel_tol=1.0e-10, abs_tol=1.0e-10)
