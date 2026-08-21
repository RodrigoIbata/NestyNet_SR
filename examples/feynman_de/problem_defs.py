#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Problem definitions for the Feynman DE benchmark.

Parses ``data/feynman_de_benchmark.txt`` into structured ``ProblemDef`` objects
and provides per-problem RHS functions, ground-truth coefficient maps, and
automatic flag inference for ``run_de.py``.
"""

from __future__ import annotations

import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from nestynet_sr.sr_core.problem_dims import CanonicalProblemDims


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProblemDef:
    id: str                                      # e.g. "000"
    order: int                                   # 1 or 2
    indep_var: str                               # "t", "x", "r", "xi"
    dep_var: str                                 # "u", "N", "theta", ...
    equation: str                                # "du/dt=-lambda*u"
    description: str                             # "Radioactive decay"
    feynman_ref: str                             # "I.3", "classical"
    params: list[str] = field(default_factory=list)
    param_ranges: list[tuple[float, float]] = field(default_factory=list)
    ic_type: str = "value"                       # "decay","value","bounded","oscillatory"
    flags: list[str] = field(default_factory=list)  # declared class metadata, e.g. "singular_origin"


# ---------------------------------------------------------------------------
# Benchmark file parser
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"\[([^]]+)\]")


def _parse_ranges(raw: str) -> list[tuple[float, float]]:
    """Parse ``[lo,hi],[lo,hi]`` into a list of (lo, hi) tuples."""
    if raw.strip() == "-":
        return []
    ranges = []
    for m in _RANGE_RE.finditer(raw):
        lo, hi = m.group(1).split(",")
        ranges.append((float(lo), float(hi)))
    return ranges


def _parse_params(raw: str) -> list[str]:
    if raw.strip() == "-":
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def load_problems(benchmark_path: str | Path) -> dict[str, ProblemDef]:
    """Parse ``feynman_de_benchmark.txt`` and return ``{id: ProblemDef}``."""
    path = Path(benchmark_path)
    problems: dict[str, ProblemDef] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Extract the quoted description first, then split the rest.
        m = re.search(r'"([^"]*)"', line)
        if not m:
            continue  # skip malformed lines
        description = m.group(1)
        before = line[: m.start()].strip()
        after = line[m.end() :].strip()

        parts_before = before.split()
        if len(parts_before) < 5:
            continue
        pid = parts_before[0]
        order = int(parts_before[1])
        indep_var = parts_before[2]
        dep_var = parts_before[3]
        equation = parts_before[4]

        parts_after = after.split()
        feynman_ref = parts_after[0] if len(parts_after) > 0 else "-"
        params_raw = parts_after[1] if len(parts_after) > 1 else "-"
        ranges_raw = parts_after[2] if len(parts_after) > 2 else "-"
        ic_type = parts_after[3] if len(parts_after) > 3 else "value"
        flags_raw = parts_after[4] if len(parts_after) > 4 else "-"

        problems[pid] = ProblemDef(
            id=pid,
            order=order,
            indep_var=indep_var,
            dep_var=dep_var,
            equation=equation,
            description=description,
            feynman_ref=feynman_ref,
            params=_parse_params(params_raw),
            param_ranges=_parse_ranges(ranges_raw),
            ic_type=ic_type,
            flags=_parse_params(flags_raw),
        )

    return problems


# ===================================================================
# Automatic flag inference from problem structure
# ===================================================================

def is_autonomous(problem: ProblemDef) -> bool:
    """Return True if the DE RHS has no explicit dependence on the
    independent variable.  Checks whether the independent variable
    name appears in the RHS portion of the equation string."""
    eq = problem.equation
    # Split on '=' to get the RHS
    if "=" not in eq:
        return False
    rhs = eq.split("=", 1)[1]
    iv = problem.indep_var  # e.g. "t", "x", "r", "xi"
    # Match the independent variable as a whole token (not inside
    # parameter names like "omega" containing "o").
    # Use word-boundary matching.
    return re.search(rf'\b{re.escape(iv)}\b', rhs) is None


def needs_u_squared(problem: ProblemDef) -> bool:
    """Return True if the STLSQ library needs u² (or higher) terms.

    Detects two patterns:
    - Explicit dep-var powers:  ``v**2``, ``c**2``, ``theta**3``
    - Implicit nonlinear products: ``N*(1-N/K)`` (dep-var appears >1 time)
    """
    eq = problem.equation
    if "=" not in eq:
        return False
    rhs = eq.split("=", 1)[1]
    dv = re.escape(problem.dep_var)
    # Explicit dep-var power (e.g. v**2, c**2)
    if re.search(rf"\b{dv}\*\*\d", rhs):
        return True
    # Multiple dep-var occurrences → nonlinear product
    return len(re.findall(rf"\b{dv}\b", rhs)) > 1


def _infer_flags_reference(problem: ProblemDef) -> dict:
    """**Reference only** — derives flags from the ground-truth equation.

    The benchmark runner no longer calls this; it uses a uniform superset
    library instead.  Kept for analysis and debugging.

    Rules:
    - Order 2 → ``--include_du``
    - Autonomous → ``--no_x --no_xu``
    - RHS contains ``**2`` or ``**3`` → ``--max_u_power 2``
    """
    flags: dict = {}

    if problem.order >= 2:
        flags["include_du"] = True

    if is_autonomous(problem):
        flags["no_x"] = True
        flags["no_xu"] = True

    if needs_u_squared(problem):
        flags["max_u_power"] = 2

    return flags


# ===================================================================
# RHS function registry
# ===================================================================
# Each entry: rhs(t, state, params) -> list[float]
# ``state`` is [u] for order-1, [u, du/dt] for order-2.
# ``params`` is a dict of parameter name -> value.

RHS_REGISTRY: dict[str, Callable] = {}
COMPILED_RHS_REGISTRY: dict[str, Callable] = {}


def _rhs(pid: str):
    """Decorator to register an RHS function."""
    def decorator(fn: Callable) -> Callable:
        RHS_REGISTRY[pid] = fn
        return fn
    return decorator


_LHS_D1_RE = re.compile(r"^d([A-Za-z_][A-Za-z0-9_]*)/d([A-Za-z_][A-Za-z0-9_]*)$")
_LHS_D2_RE = re.compile(r"^d2([A-Za-z_][A-Za-z0-9_]*)/d([A-Za-z_][A-Za-z0-9_]*)2$")
_D1_PLACEHOLDER = "__de_d1_state__"
_SAFE_EPS = 1.0e-12


def _parse_lhs_derivative(lhs: str) -> tuple[int, str, str]:
    lhs_s = lhs.strip()
    m2 = _LHS_D2_RE.fullmatch(lhs_s)
    if m2 is not None:
        return 2, str(m2.group(1)), str(m2.group(2))
    m1 = _LHS_D1_RE.fullmatch(lhs_s)
    if m1 is not None:
        return 1, str(m1.group(1)), str(m1.group(2))
    raise ValueError(f"Unsupported DE LHS format: {lhs!r}")


def _needs_symbol_alias(name: str) -> bool:
    return (not str(name).isidentifier()) or keyword.iskeyword(str(name))


def _replace_identifier_tokens(expr: str, replacements: dict[str, str]) -> str:
    if not replacements:
        return expr
    out = str(expr)
    for name, alias in sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
        patt = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
        out = re.sub(patt, alias, out)
    return out


def compile_rhs_from_equation(problem: ProblemDef) -> Callable:
    """Compile a numeric RHS callback from ``problem.equation`` using SymPy.

    Returns a function with signature ``rhs(t, state, params) -> list[float]``,
    compatible with ``solve_ivp``.
    """

    try:
        import sympy as sp  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency path
        raise RuntimeError("SymPy is required for auto-compiling RHS from equation strings") from exc

    if "=" not in problem.equation:
        raise ValueError(f"Malformed equation for de{problem.id}: missing '=' ({problem.equation!r})")
    lhs_raw, rhs_raw = problem.equation.split("=", 1)
    lhs = lhs_raw.strip()
    rhs = rhs_raw.strip()
    if rhs == "":
        raise ValueError(f"Malformed equation for de{problem.id}: empty RHS")

    eq_order, dep_sym_name, indep_sym_name = _parse_lhs_derivative(lhs)
    if int(eq_order) != int(problem.order):
        raise ValueError(
            f"de{problem.id}: order mismatch between metadata ({problem.order}) and equation LHS ({eq_order})"
        )

    d1_token = f"d{dep_sym_name}/d{indep_sym_name}"
    rhs_work = rhs.replace(d1_token, _D1_PLACEHOLDER)
    if int(eq_order) == 1 and _D1_PLACEHOLDER in rhs_work:
        raise ValueError(
            f"de{problem.id}: first-order equation has derivative term on RHS ({d1_token}); implicit form unsupported"
        )

    symbol_names = [indep_sym_name, dep_sym_name]
    if int(eq_order) == 2:
        symbol_names.append(_D1_PLACEHOLDER)
    symbol_names.extend(problem.params)
    symbols = {name: sp.Symbol(name, real=True) for name in symbol_names}

    alias_map: dict[str, str] = {}
    alias_idx = 0
    for name in symbol_names:
        if _needs_symbol_alias(name):
            alias = f"__de_sym_{alias_idx}__"
            while alias in symbols or alias in alias_map.values():
                alias_idx += 1
                alias = f"__de_sym_{alias_idx}__"
            alias_map[name] = alias
            alias_idx += 1

    rhs_for_parse = _replace_identifier_tokens(rhs_work, alias_map)

    # Treat I/E/lambda-like names as plain symbols when they appear in equations.
    # This avoids collisions with SymPy's imaginary unit `I` and Euler number `E`.
    parse_symbols = {alias_map.get(name, name): sym for name, sym in symbols.items()}
    locals_: dict[str, object] = dict(parse_symbols)
    if "I" not in locals_:
        locals_["I"] = sp.Symbol("I", real=True)
    if "E" not in locals_:
        locals_["E"] = sp.Symbol("E", real=True)
    locals_.update(
        {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
        }
    )

    try:
        expr = sp.sympify(rhs_for_parse, locals=locals_)
    except Exception as exc:
        raise ValueError(
            f"de{problem.id}: failed to parse RHS expression {rhs!r} from equation {problem.equation!r}"
        ) from exc

    arg_syms = [symbols[indep_sym_name], symbols[dep_sym_name]]
    if int(eq_order) == 2:
        arg_syms.append(symbols[_D1_PLACEHOLDER])
    for pname in problem.params:
        arg_syms.append(symbols[pname])

    def _safe_sqrt(x):
        xa = np.asarray(x)
        return np.sqrt(np.maximum(xa, 0.0))

    def _safe_log(x):
        xa = np.asarray(x)
        return np.log(np.maximum(xa, _SAFE_EPS))

    def _safe_asin(x):
        xa = np.asarray(x)
        return np.arcsin(np.clip(xa, -1.0 + 1.0e-9, 1.0 - 1.0e-9))

    def _safe_acos(x):
        xa = np.asarray(x)
        return np.arccos(np.clip(xa, -1.0 + 1.0e-9, 1.0 - 1.0e-9))

    safe_module = {
        "sqrt": _safe_sqrt,
        "log": _safe_log,
        "asin": _safe_asin,
        "acos": _safe_acos,
    }
    fn = sp.lambdify(
        arg_syms,
        expr,
        modules=[safe_module, "numpy", "math"],
        dummify=True,  # required for params like "lambda"
    )

    def rhs_fn(t: float, s: Sequence[float], p: dict[str, float]) -> list[float]:
        if int(eq_order) == 1:
            u0 = float(s[0])
            args: list[float] = [float(t), u0]
            for pname in problem.params:
                args.append(float(p[pname]))
            val = float(fn(*args))
            if not math.isfinite(val):
                raise FloatingPointError(f"de{problem.id}: non-finite RHS value ({val})")
            return [val]

        u0 = float(s[0])
        du0 = float(s[1])
        args = [float(t), u0, du0]
        for pname in problem.params:
            args.append(float(p[pname]))
        val2 = float(fn(*args))
        if not math.isfinite(val2):
            raise FloatingPointError(f"de{problem.id}: non-finite RHS value ({val2})")
        return [du0, val2]

    return rhs_fn


def resolve_rhs(problem: ProblemDef, *, prefer_manual: bool = True) -> tuple[Callable, str]:
    """Return an RHS function for ``problem`` and its source label.

    Source is one of:
    - ``"registry"``: hand-written RHS from ``RHS_REGISTRY``.
    - ``"compiled"``: auto-compiled from equation string.
    """

    pid = str(problem.id)

    if bool(prefer_manual):
        fn_reg = RHS_REGISTRY.get(pid)
        if fn_reg is not None:
            return fn_reg, "registry"

    fn_cached = COMPILED_RHS_REGISTRY.get(pid)
    if fn_cached is not None:
        return fn_cached, "compiled"

    fn_comp = compile_rhs_from_equation(problem)
    COMPILED_RHS_REGISTRY[pid] = fn_comp
    return fn_comp, "compiled"


# ---- First-order linear ----

@_rhs("000")
def _rhs_000(t, s, p):
    """du/dt = -lambda*u"""
    return [-p["lambda"] * s[0]]

@_rhs("001")
def _rhs_001(t, s, p):
    """dN/dt = r*N"""
    return [p["r"] * s[0]]

@_rhs("003")
def _rhs_003(t, s, p):
    """dT/dt = -k*(T - T_env)"""
    return [-p["k"] * (s[0] - p["T_env"])]

@_rhs("004")
def _rhs_004(t, s, p):
    """dT/dx = -T/L"""
    return [-s[0] / p["L"]]

@_rhs("005")
def _rhs_005(t, s, p):
    """dq/dt = -q/(R*C)"""
    return [-s[0] / (p["R"] * p["C"])]

@_rhs("006")
def _rhs_006(t, s, p):
    """dI/dt = -R*I/L"""
    return [-p["R"] * s[0] / p["L"]]

@_rhs("007")
def _rhs_007(t, s, p):
    """dq/dt = (V0 - q/C)/R"""
    return [(p["V0"] - s[0] / p["C"]) / p["R"]]

@_rhs("008")
def _rhs_008(t, s, p):
    """dv/dt = g - k*v/m"""
    return [p["g"] - p["k"] * s[0] / p["m"]]

@_rhs("011")
def _rhs_011(t, s, p):
    """dp/dx = -rho*g"""
    return [-p["rho"] * p["g"]]

@_rhs("012")
def _rhs_012(t, s, p):
    """dpsi/dx = -kappa*psi"""
    return [-p["kappa"] * s[0]]


# ---- First-order nonlinear (u^2 terms) ----

@_rhs("002")
def _rhs_002(t, s, p):
    """dN/dt = r*N*(1 - N/K)"""
    return [p["r"] * s[0] * (1.0 - s[0] / p["K"])]

@_rhs("009")
def _rhs_009(t, s, p):
    """dv/dt = g - k*v^2/m"""
    return [p["g"] - p["k"] * s[0] ** 2 / p["m"]]

@_rhs("013")
def _rhs_013(t, s, p):
    """dc/dt = -k*c^2"""
    return [-p["k"] * s[0] ** 2]


# ---- Second-order linear ----

@_rhs("100")
def _rhs_100(t, s, p):
    """d2x/dt2 = -omega^2*x"""
    return [s[1], -p["omega"] ** 2 * s[0]]

@_rhs("101")
def _rhs_101(t, s, p):
    """d2x/dt2 = -(k/m)*x"""
    return [s[1], -(p["k"] / p["m"]) * s[0]]

@_rhs("102")
def _rhs_102(t, s, p):
    """d2theta/dt2 = -(g/L)*theta"""
    return [s[1], -(p["g"] / p["L"]) * s[0]]

@_rhs("103")
def _rhs_103(t, s, p):
    """d2x/dt2 = -2*gamma*dx/dt - omega0^2*x"""
    return [s[1], -2.0 * p["gamma"] * s[1] - p["omega0"] ** 2 * s[0]]

@_rhs("104")
def _rhs_104(t, s, p):
    """d2x/dt2 = -(b/m)*dx/dt - (k/m)*x"""
    return [s[1], -(p["b"] / p["m"]) * s[1] - (p["k"] / p["m"]) * s[0]]

@_rhs("105")
def _rhs_105(t, s, p):
    """d2q/dt2 = -(R/L)*dq/dt - q/(L*C)"""
    return [s[1], -(p["R"] / p["L"]) * s[1] - s[0] / (p["L"] * p["C"])]

@_rhs("106")
def _rhs_106(t, s, p):
    """d2x/dt2 = -g"""
    return [s[1], -p["g"]]

@_rhs("107")
def _rhs_107(t, s, p):
    """d2x/dt2 = -g - k*dx/dt"""
    return [s[1], -p["g"] - p["k"] * s[1]]

@_rhs("110")
def _rhs_110(t, s, p):
    """d2phi/dx2 = -rho/epsilon0"""
    return [s[1], -p["rho"] / p["epsilon0"]]

@_rhs("112")
def _rhs_112(t, s, p):
    """d2E/dx2 = -k^2*E"""
    return [s[1], -p["k"] ** 2 * s[0]]

@_rhs("108")
def _rhs_108(t, s, p):
    """d2r/dt2 = -G*M/r^2"""
    r = max(s[0], 1.0e-6)  # guard against singularity
    return [s[1], -p["G"] * p["M"] / r ** 2]

@_rhs("109")
def _rhs_109(t, s, p):
    """d2u/dphi2 = -u + G*M/L^2  (corrected Newtonian Binet)"""
    return [s[1], -s[0] + p["G"] * p["M"] / p["L"] ** 2]

@_rhs("300")
def _rhs_300(t, s, p):
    """d2x/dt2 = -a*x - b*dx/dt + c"""
    return [s[1], -p["a"] * s[0] - p["b"] * s[1] + p["c"]]


# ---- Tier 1: Polynomial STLSQ, straightforward ----

@_rhs("111")
def _rhs_111(t, s, p):
    """d2E/dx2 = k^2*E  (evanescent wave)"""
    return [s[1], p["k"] ** 2 * s[0]]

@_rhs("119")
def _rhs_119(t, s, p):
    """d2psi/dx2 = (2m/hbar^2)*(V-E)*psi"""
    return [s[1], 2.0 * p["m"] / p["hbar"] ** 2 * (p["V"] - p["E"]) * s[0]]

@_rhs("120")
def _rhs_120(t, s, p):
    """d2psi/dx2 = -k^2*psi"""
    return [s[1], -p["k"] ** 2 * s[0]]

@_rhs("121")
def _rhs_121(t, s, p):
    """d2psi/dx2 = (k^2*x^2 - E)*psi"""
    return [s[1], (p["k"] ** 2 * t ** 2 - p["E"]) * s[0]]

@_rhs("124")
def _rhs_124(t, s, p):
    """d2x/dt2 = -omega^2*x + epsilon*x^2"""
    return [s[1], -p["omega"] ** 2 * s[0] + p["epsilon"] * s[0] ** 2]

@_rhs("129")
def _rhs_129(t, s, p):
    """d2y/dx2 = (mu/T)*y  (exponential spatial branch, X'' = (mu s^2/T) X with s = 1)"""
    return [s[1], (p["mu"] / p["T"]) * s[0]]

@_rhs("131")
def _rhs_131(t, s, p):
    """d2y/dx2 = -(omega^2*mu/T)*y"""
    return [s[1], -(p["omega"] ** 2) * p["mu"] / p["T"] * s[0]]

@_rhs("201")
def _rhs_201(t, s, p):
    """dy/dx = y^2 + x"""
    return [s[0] ** 2 + t]

@_rhs("202")
def _rhs_202(t, s, p):
    """dy/dx = y + y^2"""
    return [s[0] + s[0] ** 2]

@_rhs("203")
def _rhs_203(t, s, p):
    """dP/dt = r*P - a*P^2"""
    return [p["r"] * s[0] - p["a"] * s[0] ** 2]

@_rhs("204")
def _rhs_204(t, s, p):
    """dy/dt = k*t*y"""
    return [p["k"] * t * s[0]]


# ---- Tier 2: Singular coefficients, needs inv_xdu ----

@_rhs("114")
def _rhs_114(t, s, p):
    """d2theta/dxi2 = -(2/xi)*dtheta/dxi - theta  (Lane-Emden n=1)"""
    return [s[1], -2.0 / t * s[1] - s[0]]

@_rhs("118")
def _rhs_118(t, s, p):
    """d2y/dx2 = -(1/x)*dy/dx - y  (Bessel J_0)"""
    return [s[1], -1.0 / t * s[1] - s[0]]

@_rhs("128")
def _rhs_128(t, s, p):
    """d2p/dr2 = -(2/r)*dp/dr - k^2*p  (spherical acoustic)"""
    return [s[1], -2.0 / t * s[1] - p["k"] ** 2 * s[0]]


# ---- Tier 3: Cubic nonlinearity ----

@_rhs("115")
def _rhs_115(t, s, p):
    """d2theta/dxi2 = -(2/xi)*dtheta/dxi - theta^3  (Lane-Emden n=3)"""
    return [s[1], -2.0 / t * s[1] - s[0] ** 3]

@_rhs("122")
def _rhs_122(t, s, p):
    """d2x/dt2 = -omega^2*x - alpha*x^3  (Duffing)"""
    return [s[1], -p["omega"] ** 2 * s[0] - p["alpha"] * s[0] ** 3]


# ---- Tier 4: Easy additions ----

@_rhs("200")
def _rhs_200(t, s, p):
    """dy/dx = y^3 + x  (Abel equation)"""
    return [s[0] ** 3 + t]

@_rhs("207")
def _rhs_207(t, s, p):
    """dx/dt = a*x - b*x*y  (prey equation, y=const)"""
    return [(p["a"] - p["b"] * p["y"]) * s[0]]


# ===================================================================
# Ground truth registry
# ===================================================================
# Implicit form: anchor + sum(c_k * term_k) = 0
# For order-1: anchor = u_x0.   For order-2: anchor = u_x0x0.
# ``terms`` maps term repr (e.g. "u") to either:
#   - a param name string (resolved at validation time), or
#   - a float constant, or
#   - a callable(params) -> float for compound coefficients.

@dataclass
class GroundTruth:
    order: int
    terms: dict[str, str | float | Callable]   # expected nonzero terms
    absent: list[str] = field(default_factory=list)  # terms that should be ~0
    coeff_rtol: float = 0.10                   # relative tolerance
    coeff_atol: float = 0.05                   # absolute tolerance (near-zero)
    decoy_atol: float = 0.05                   # tolerance for nuisance terms


GROUND_TRUTH: dict[str, GroundTruth] = {
    # ------------------------------------------------------------------
    # First-order linear
    # ------------------------------------------------------------------

    # du/dt = -lambda*u  =>  u_x0 + lambda*u = 0
    "000": GroundTruth(order=1, terms={"u": "lambda"}),

    # dN/dt = r*N  =>  u_x0 - r*u = 0
    "001": GroundTruth(order=1, terms={"u": lambda p: -p["r"]}),

    # dT/dt = -k*(T-T_env)  =>  u_x0 + k*u - k*T_env = 0
    "003": GroundTruth(
        order=1,
        terms={
            "u": "k",
            "1": lambda p: -p["k"] * p["T_env"],
        },
    ),

    # dT/dx = -T/L  =>  u_x0 + (1/L)*u = 0
    "004": GroundTruth(order=1, terms={"u": lambda p: 1.0 / p["L"]}),

    # dq/dt = -q/(RC)  =>  u_x0 + 1/(RC)*u = 0
    "005": GroundTruth(
        order=1,
        terms={"u": lambda p: 1.0 / (p["R"] * p["C"])},
    ),

    # dI/dt = -(R/L)*I  =>  u_x0 + (R/L)*u = 0
    "006": GroundTruth(
        order=1,
        terms={"u": lambda p: p["R"] / p["L"]},
    ),

    # dq/dt = (V0-q/C)/R  =>  u_x0 + 1/(RC)*u - V0/R = 0
    "007": GroundTruth(
        order=1,
        terms={
            "u": lambda p: 1.0 / (p["R"] * p["C"]),
            "1": lambda p: -p["V0"] / p["R"],
        },
    ),

    # dv/dt = g - k*v/m  =>  u_x0 + (k/m)*u - g = 0
    "008": GroundTruth(
        order=1,
        terms={
            "u": lambda p: p["k"] / p["m"],
            "1": lambda p: -p["g"],
        },
    ),

    # dp/dx = -rho*g  =>  u_x0 + rho*g = 0
    "011": GroundTruth(
        order=1,
        terms={"1": lambda p: p["rho"] * p["g"]},
    ),

    # dpsi/dx = -kappa*psi  =>  u_x0 + kappa*u = 0
    "012": GroundTruth(order=1, terms={"u": "kappa"}),

    # ------------------------------------------------------------------
    # First-order nonlinear (u^2)
    # ------------------------------------------------------------------

    # dN/dt = r*N*(1-N/K) = r*N - (r/K)*N^2  =>  u_x0 - r*u + (r/K)*u^2 = 0
    "002": GroundTruth(
        order=1,
        terms={
            "u": lambda p: -p["r"],
            "(u ** 2)": lambda p: p["r"] / p["K"],
        },
    ),

    # dv/dt = g - k*v^2/m  =>  u_x0 + (k/m)*u^2 - g = 0
    "009": GroundTruth(
        order=1,
        terms={
            "(u ** 2)": lambda p: p["k"] / p["m"],
            "1": lambda p: -p["g"],
        },
    ),

    # dc/dt = -k*c^2  =>  u_x0 + k*u^2 = 0
    "013": GroundTruth(
        order=1,
        terms={"(u ** 2)": "k"},
    ),

    # ------------------------------------------------------------------
    # Second-order linear
    # ------------------------------------------------------------------

    # d2x/dt2 = -omega^2*x  =>  u_x0x0 + omega^2*u = 0
    "100": GroundTruth(
        order=2,
        terms={"u": lambda p: p["omega"] ** 2},
    ),

    # d2x/dt2 = -(k/m)*x  =>  u_x0x0 + (k/m)*u = 0
    "101": GroundTruth(
        order=2,
        terms={"u": lambda p: p["k"] / p["m"]},
    ),

    # d2theta/dt2 = -(g/L)*theta  =>  u_x0x0 + (g/L)*u = 0
    "102": GroundTruth(
        order=2,
        terms={"u": lambda p: p["g"] / p["L"]},
    ),

    # d2x/dt2 = -2*gamma*dx/dt - omega0^2*x
    #   => u_x0x0 + 2*gamma*u_x0 + omega0^2*u = 0
    "103": GroundTruth(
        order=2,
        terms={
            "u": lambda p: p["omega0"] ** 2,
            "u_x0": lambda p: 2.0 * p["gamma"],
        },
    ),

    # d2x/dt2 = -(b/m)*dx/dt - (k/m)*x
    #   => u_x0x0 + (b/m)*u_x0 + (k/m)*u = 0
    "104": GroundTruth(
        order=2,
        terms={
            "u": lambda p: p["k"] / p["m"],
            "u_x0": lambda p: p["b"] / p["m"],
        },
    ),

    # d2q/dt2 = -(R/L)*dq/dt - q/(LC)
    #   => u_x0x0 + (R/L)*u_x0 + 1/(LC)*u = 0
    "105": GroundTruth(
        order=2,
        terms={
            "u": lambda p: 1.0 / (p["L"] * p["C"]),
            "u_x0": lambda p: p["R"] / p["L"],
        },
    ),

    # d2x/dt2 = -g  =>  u_x0x0 + g = 0
    "106": GroundTruth(
        order=2,
        terms={"1": "g"},
    ),

    # d2x/dt2 = -g - k*dx/dt  =>  u_x0x0 + k*u_x0 + g = 0
    "107": GroundTruth(
        order=2,
        terms={
            "u_x0": "k",
            "1": "g",
        },
    ),

    # d2r/dt2 = -G*M/r²  =>  nonlinear (1/r² term), no polynomial ground truth
    # 108: skipped — nonlinear inverse-square term not in polynomial library

    # d2u/dphi2 = -u + G*M/L²  =>  u_x0x0 + u - G*M/L² = 0
    "109": GroundTruth(
        order=2,
        terms={
            "u": 1.0,
            "1": lambda p: -p["G"] * p["M"] / p["L"] ** 2,
        },
    ),

    # d2phi/dx2 = -rho/epsilon0  =>  u_x0x0 + rho/epsilon0 = 0
    "110": GroundTruth(
        order=2,
        terms={"1": lambda p: p["rho"] / p["epsilon0"]},
    ),

    # d2E/dx2 = -k^2*E  =>  u_x0x0 + k^2*u = 0
    "112": GroundTruth(
        order=2,
        terms={"u": lambda p: p["k"] ** 2},
    ),

    # d2x/dt2 = -a*x - b*dx/dt + c  =>  u_x0x0 + a*u + b*u_x0 - c = 0
    "300": GroundTruth(
        order=2,
        terms={
            "u": "a",
            "u_x0": "b",
            "1": lambda p: -p["c"],
        },
    ),

    # ------------------------------------------------------------------
    # Tier 1: Polynomial STLSQ, straightforward
    # ------------------------------------------------------------------

    # d2E/dx2 = k^2*E  =>  u_x0x0 - k^2*u = 0
    "111": GroundTruth(
        order=2,
        terms={"u": lambda p: -p["k"] ** 2},
    ),

    # d2psi/dx2 = (2m/hbar^2)*(V-E)*psi  =>  u_x0x0 - (2m/hbar^2)*(V-E)*u = 0
    "119": GroundTruth(
        order=2,
        terms={"u": lambda p: -2.0 * p["m"] / p["hbar"] ** 2 * (p["V"] - p["E"])},
    ),

    # d2psi/dx2 = -k^2*psi  =>  u_x0x0 + k^2*u = 0
    "120": GroundTruth(
        order=2,
        terms={"u": lambda p: p["k"] ** 2},
    ),

    # d2psi/dx2 = (k^2*x^2 - E)*psi  =>  u_x0x0 - k^2*x^2*u + E*u = 0
    "121": GroundTruth(
        order=2,
        terms={
            "u": "E",
            "((x0 ** 2) * u)": lambda p: -p["k"] ** 2,
        },
    ),

    # d2x/dt2 = -omega^2*x + epsilon*x^2  =>  u_x0x0 + omega^2*u - epsilon*u^2 = 0
    "124": GroundTruth(
        order=2,
        terms={
            "u": lambda p: p["omega"] ** 2,
            "(u ** 2)": lambda p: -p["epsilon"],
        },
    ),

    # d2y/dx2 = (T/mu)*y  =>  u_x0x0 - (T/mu)*u = 0
    "129": GroundTruth(
        order=2,
        terms={"u": lambda p: -p["mu"] / p["T"]},
    ),

    # d2y/dx2 = -(omega^2*mu/T)*y  =>  u_x0x0 + (omega^2*mu/T)*u = 0
    "131": GroundTruth(
        order=2,
        terms={"u": lambda p: (p["omega"] ** 2) * p["mu"] / p["T"]},
    ),

    # dy/dx = y^2 + x  =>  u_x0 - u^2 - x = 0
    "201": GroundTruth(
        order=1,
        terms={
            "(u ** 2)": -1.0,
            "x0": -1.0,
        },
    ),

    # dy/dx = y + y^2  =>  u_x0 - u - u^2 = 0
    "202": GroundTruth(
        order=1,
        terms={
            "u": -1.0,
            "(u ** 2)": -1.0,
        },
    ),

    # dP/dt = r*P - a*P^2  =>  u_x0 - r*u + a*u^2 = 0
    "203": GroundTruth(
        order=1,
        terms={
            "u": lambda p: -p["r"],
            "(u ** 2)": "a",
        },
    ),

    # dy/dt = k*t*y  =>  u_x0 - k*x*u = 0
    "204": GroundTruth(
        order=1,
        terms={"(x0 * u)": lambda p: -p["k"]},
    ),

    # ------------------------------------------------------------------
    # Tier 2: Singular coefficients, needs inv_xdu
    # ------------------------------------------------------------------

    # d2theta/dxi2 = -(2/xi)*dtheta/dxi - theta
    #   =>  u_x0x0 + 2*x^-1*u_x0 + u = 0
    "114": GroundTruth(
        order=2,
        terms={
            "((x0 ** -1) * u_x0)": 2.0,
            "u": 1.0,
        },
    ),

    # d2y/dx2 = -(1/x)*dy/dx - y  =>  u_x0x0 + x^-1*u_x0 + u = 0
    "118": GroundTruth(
        order=2,
        terms={
            "((x0 ** -1) * u_x0)": 1.0,
            "u": 1.0,
        },
    ),

    # d2p/dr2 = -(2/r)*dp/dr - k^2*p
    #   =>  u_x0x0 + 2*x^-1*u_x0 + k^2*u = 0
    "128": GroundTruth(
        order=2,
        terms={
            "((x0 ** -1) * u_x0)": 2.0,
            "u": lambda p: p["k"] ** 2,
        },
    ),

    # ------------------------------------------------------------------
    # Tier 3: Cubic nonlinearity
    # ------------------------------------------------------------------

    # d2theta/dxi2 = -(2/xi)*dtheta/dxi - theta^3
    #   =>  u_x0x0 + 2*x^-1*u_x0 + u^3 = 0
    "115": GroundTruth(
        order=2,
        terms={
            "((x0 ** -1) * u_x0)": 2.0,
            "(u ** 3)": 1.0,
        },
    ),

    # d2x/dt2 = -omega^2*x - alpha*x^3
    #   =>  u_x0x0 + omega^2*u + alpha*u^3 = 0
    "122": GroundTruth(
        order=2,
        terms={
            "u": lambda p: p["omega"] ** 2,
            "(u ** 3)": "alpha",
        },
    ),

    # ------------------------------------------------------------------
    # Tier 4: Easy additions
    # ------------------------------------------------------------------

    # dy/dx = y^3 + x  =>  u_x0 - u^3 - x = 0
    "200": GroundTruth(
        order=1,
        terms={
            "(u ** 3)": -1.0,
            "x0": -1.0,
        },
    ),

    # dx/dt = (a - b*y)*x  =>  u_x0 - (a - b*y)*u = 0
    # Since y is a fixed parameter, the effective coefficient on u is -(a - b*y)
    "207": GroundTruth(
        order=1,
        terms={"u": lambda p: -(p["a"] - p["b"] * p["y"])},
    ),
}


# ===================================================================
# Dimensional analysis registry
# ===================================================================
# Maps problem ID -> dimensional metadata for the oracle lab DE solver.
#
# Convention: 1D basis ("D",) with x_dim=(1,) and u_dim=(0,) for all
# problems.  Each parameter's dimension is derived from the equation
# structure so that every term has consistent dimension.
#
# Order-1 target dim = u_dim - x_dim = (-1,)
# Order-2 target dim = u_dim - 2*x_dim = (-2,)
#
# Problems with dimensionless equations (all terms have dim 0) are
# omitted — dimensional filtering cannot help there.  These include
# Lane-Emden (114,115,116), Bessel (117,118), Van der Pol (125),
# parametric (126), relativistic (127), capillary (130), and bare
# nonlinear ODEs without parameters (200,201,202,205).

@dataclass
class ProblemDims:
    basis: tuple[str, ...]
    x_dim: tuple[float, ...]
    u_dim: tuple[float, ...]
    param_dims: dict[str, tuple[float, ...]]


def to_canonical_problem_dims(dims: ProblemDims) -> CanonicalProblemDims:
    """Convert the legacy scalar benchmark dims into the shared canonical form."""
    return CanonicalProblemDims.scalar(
        basis=tuple(str(v) for v in dims.basis),
        x_dim=tuple(float(v) for v in dims.x_dim),
        u_dim=tuple(float(v) for v in dims.u_dim),
        constant_dims={
            str(name): tuple(float(v) for v in dim)
            for name, dim in dict(dims.param_dims).items()
        },
    )


def get_canonical_problem_dims(pid: str) -> Optional[CanonicalProblemDims]:
    dims = DIMS_REGISTRY.get(str(pid))
    if dims is None:
        return None
    return to_canonical_problem_dims(dims)


DIMS_REGISTRY: dict[str, ProblemDims] = {
    # ------------------------------------------------------------------
    # First-order DEs: target dim = (-1,)
    # ------------------------------------------------------------------

    # du/dt = -lambda*u  →  lambda has dim (-1,) [coeff of u]
    "000": ProblemDims(("D",), (1,), (0,), {"lambda": (-1,)}),

    # dN/dt = r*N  →  r has dim (-1,)
    "001": ProblemDims(("D",), (1,), (0,), {"r": (-1,)}),

    # dN/dt = r*N - (r/K)*N²  →  r: (-1,), K: (0,) [same dim as u]
    "002": ProblemDims(("D",), (1,), (0,), {"r": (-1,), "K": (0,)}),

    # dT/dt = -k*(T-T_env)  →  k: (-1,), T_env: (0,) [same dim as u]
    "003": ProblemDims(("D",), (1,), (0,), {"k": (-1,), "T_env": (0,)}),

    # dT/dx = -T/L  →  L: (1,) [same dim as x]
    "004": ProblemDims(("D",), (1,), (0,), {"L": (1,)}),

    # dq/dt = -q/(R*C)  →  R*C has dim (1,); R: (1,), C: (0,)
    "005": ProblemDims(("D",), (1,), (0,), {"R": (1,), "C": (0,)}),

    # dI/dt = -(R/L)*I  →  R/L has dim (-1,); R: (0,), L: (1,)
    "006": ProblemDims(("D",), (1,), (0,), {"R": (0,), "L": (1,)}),

    # dq/dt = (V0-q/C)/R  →  V0/R: dim (-1,), 1/(RC): dim (-1,)
    # V0: (0,), R: (1,), C: (0,)
    "007": ProblemDims(("D",), (1,), (0,), {"V0": (0,), "R": (1,), "C": (0,)}),

    # dv/dt = g - (k/m)*v  →  2D basis ("T","M") separates gravity from drag
    # g*1: (-1,0) ✓   (k/m)*u: (-1,1)-(0,1)+(0,0) = (-1,0) ✓
    # k*u alone: (-1,1)+(0,0) = (-1,1) ✗ blocked
    "008": ProblemDims(("T", "M"), (1, 0), (0, 0), {
        "g": (-1, 0), "k": (-1, 1), "m": (0, 1),
    }),

    # dv/dt = g - (k/m)*v²  →  same 2D basis as 008
    "009": ProblemDims(("T", "M"), (1, 0), (0, 0), {
        "g": (-1, 0), "k": (-1, 1), "m": (0, 1),
    }),

    # dv/dr = -v/r  →  no parameters, dims on x and u only
    "010": ProblemDims(("D",), (1,), (0,), {}),

    # dp/dx = -rho*g  →  rho*g: (-1,); rho: (0,), g: (-1,)
    "011": ProblemDims(("D",), (1,), (0,), {"rho": (0,), "g": (-1,)}),

    # dpsi/dx = -kappa*psi  →  kappa: (-1,)
    "012": ProblemDims(("D",), (1,), (0,), {"kappa": (-1,)}),

    # dc/dt = -k*c²  →  k: (-1,) [coeff of u²]
    "013": ProblemDims(("D",), (1,), (0,), {"k": (-1,)}),

    # dc/dt = -k*c^(1/2)  →  k: (-1,)
    "014": ProblemDims(("D",), (1,), (0,), {"k": (-1,)}),

    # ------------------------------------------------------------------
    # Second-order DEs: target dim = (-2,)
    # ------------------------------------------------------------------

    # d2x/dt2 = -omega²*x  →  omega: (-1,) [omega²*u has dim (-2,)]
    "100": ProblemDims(("D",), (1,), (0,), {"omega": (-1,)}),

    # d2x/dt2 = -(k/m)*x  →  k/m: (-2,); k: (-2,), m: (0,)
    "101": ProblemDims(("D",), (1,), (0,), {"k": (-2,), "m": (0,)}),

    # d2theta/dt2 = -(g/L)*theta  →  g/L: (-2,); g: (-2,), L: (0,)
    "102": ProblemDims(("D",), (1,), (0,), {"g": (-2,), "L": (0,)}),

    # d2x/dt2 = -2*gamma*dx/dt - omega0²*x
    # gamma*u_x: gamma+(-1,)=(-2,) → gamma: (-1,)
    # omega0²*u: 2*omega0+(0,)=(-2,) → omega0: (-1,)
    "103": ProblemDims(("D",), (1,), (0,), {"gamma": (-1,), "omega0": (-1,)}),

    # d2x/dt2 = -(b/m)*dx/dt - (k/m)*x
    # b/m coeff of u_x: (-1,) → b: (-1,), m: (0,)
    # k/m coeff of u: (-2,) → k: (-2,), m: (0,)  ✓
    "104": ProblemDims(("D",), (1,), (0,), {"b": (-1,), "m": (0,), "k": (-2,)}),

    # d2q/dt2 = -(R/L)*dq/dt - q/(L*C)
    # 2D basis ("T","E") separates inductance from capacitance
    # R/L*du: (0,1)-(1,1)+(-1,0) = (-2,0) ✓
    # u/(LC): (0,0)-(1,1)-(1,-1) = (-2,0) ✓
    # u/L² alone: (0,0)-2*(1,1) = (-2,-2) ✗ blocked
    "105": ProblemDims(("T", "E"), (1, 0), (0, 0), {
        "R": (0, 1), "L": (1, 1), "C": (1, -1),
    }),

    # d2x/dt2 = -g  →  g: (-2,) [constant term]
    "106": ProblemDims(("D",), (1,), (0,), {"g": (-2,)}),

    # d2x/dt2 = -g - k*dx/dt  →  g: (-2,), k: (-1,) [coeff of u_x]
    "107": ProblemDims(("D",), (1,), (0,), {"g": (-2,), "k": (-1,)}),

    # d2r/dt2 = -G*M/r²  →  nonlinear, but G*M has dim (-2,) as constant
    # For the corrected polynomial part: no polynomial terms, just inverse-square
    # Dims still useful for constant pruning: G: (-1,), M: (-1,)  [G*M → (-2,)]
    "108": ProblemDims(("D",), (1,), (0,), {"G": (-1,), "M": (-1,)}),

    # d2u/dphi2 = -u + G*M/L²  →  G*M/L²: (-2,) [constant term]
    # G: (-1,), M: (-1,), L: (0,)  [so G*M/L² = (-1)+(-1)-2*(0) = (-2,)]
    # Alternatively: G: (0,), M: (0,), L: (1,) [so G*M/L² = 0+0-2 = (-2,)]
    "109": ProblemDims(("D",), (1,), (0,), {"G": (0,), "M": (0,), "L": (1,)}),

    # d2phi/dx2 = -rho/epsilon0  →  rho/epsilon0: (-2,)
    # rho: (0,), epsilon0: (2,)
    "110": ProblemDims(("D",), (1,), (0,), {"rho": (0,), "epsilon0": (2,)}),

    # d2E/dx2 = k²*E  →  k: (-1,) [k²*u has dim (-2,)]
    "111": ProblemDims(("D",), (1,), (0,), {"k": (-1,)}),

    # d2E/dx2 = -k²*E  →  k: (-1,)
    "112": ProblemDims(("D",), (1,), (0,), {"k": (-1,)}),

    # d2psi/dx2 = (2m/hbar²)*(V-E)*psi
    # m/hbar²*(V-E)*u: m+(-2*hbar)+V+0 = -2, V=E
    # Choose: m: (-1,), hbar: (1,), V: (1,), E: (1,)
    # Check: -1 + (-2) + 1 + 0 = -2  ✓
    # Note: multi-dim basis not useful here — single param set across
    # trajectories means explorer just fits c*u with numeric constant.
    "119": ProblemDims(("D",), (1,), (0,), {
        "m": (-1,), "hbar": (1,), "V": (1,), "E": (1,),
    }),

    # d2psi/dx2 = -k²*psi  →  k: (-1,)
    "120": ProblemDims(("D",), (1,), (0,), {"k": (-1,)}),

    # d2psi/dx2 = E*psi - k²*x²*psi  →  k and E forced to same dim by equation
    # E*u: (-2,)+(0,) = (-2,) ✓   k²*x²*u: 2*(-1,)+(2,)+(0,) = (-2,+2,) → k: (-1,)
    # In 1D, k and E are both (-2,) from the target perspective.
    "121": ProblemDims(("D",), (1,), (0,), {"k": (-2,), "E": (-2,)}),

    # d2x/dt2 = -omega²*x - alpha*x³
    # omega²*u: (-2,) → omega: (-1,)
    # alpha*u³: (-2,) → alpha: (-2,)
    "122": ProblemDims(("D",), (1,), (0,), {"omega": (-1,), "alpha": (-2,)}),

    # d2x/dt2 = -omega²*x + epsilon*x²
    # omega²*u: (-2,) → omega: (-1,)
    # epsilon*u²: (-2,) → epsilon: (-2,)
    "124": ProblemDims(("D",), (1,), (0,), {"omega": (-1,), "epsilon": (-2,)}),

    # d2p/dr2 = -(2/r)*dp/dr - k²*p  →  k: (-1,)
    # The (2/r)*dp/dr term is built from x and u_x, no param needed.
    "128": ProblemDims(("D",), (1,), (0,), {"k": (-1,)}),

    # d2y/dx2 = (T/mu)*y  →  T/mu: (-2,); T: (-2,), mu: (0,)
    "129": ProblemDims(("D",), (1,), (0,), {"T": (0,), "mu": (-2,)}),

    # d2y/dx2 = -(omega^2*mu/T)*y  →  omega^2*mu/T: L^-2
    "131": ProblemDims(
        ("L", "M", "T", "Y"),
        (1, 0, 0, 0),
        (0, 0, 0, 1),
        {
            "omega": (0, 0, -1, 0),
            "T": (1, 1, -2, 0),
            "mu": (-1, 1, 0, 0),
        },
    ),

    # d2x/dt2 = -a*x - b*dx/dt + c
    # a: (-2,) [coeff of u], b: (-1,) [coeff of u_x], c: (-2,) [const]
    "300": ProblemDims(("D",), (1,), (0,), {"a": (-2,), "b": (-1,), "c": (-2,)}),

    # ------------------------------------------------------------------
    # Special cases (first-order)
    # ------------------------------------------------------------------

    # dP/dt = r*P - a*P²  →  r: (-1,), a: (-1,)
    "203": ProblemDims(("D",), (1,), (0,), {"r": (-1,), "a": (-1,)}),

    # dy/dt = k*t*y  →  k*x*u: k+(1,)+(0,) = (-1,) → k: (-2,)
    "204": ProblemDims(("D",), (1,), (0,), {"k": (-2,)}),

    # dy/dx = -y/x  →  no parameters, purely structural
    "206": ProblemDims(("D",), (1,), (0,), {}),

    # dx/dt = (a-b*y)*x  →  2D basis ("T","P") separates growth rate from predation
    # a*u: (-1,0)+(0,0) = (-1,0) ✓   b*y*u: (-1,-1)+(0,1)+(0,0) = (-1,0) ✓
    # b*u alone: (-1,-1)+(0,0) = (-1,-1) ✗ blocked
    "207": ProblemDims(("T", "P"), (1, 0), (0, 0), {
        "a": (-1, 0), "b": (-1, -1), "y": (0, 1),
    }),
}


# ===================================================================
# Per-problem t_max overrides
# ===================================================================
# These reflect genuine physical constraints (blow-up, large range),
# not tuning.

T_MAX_OVERRIDE: dict[str, float] = {
    "001": 5.0,     # exponential growth → cap before blow-up
    "106": 3.0,     # free fall → parabola gets large fast
    "107": 5.0,     # free fall + drag
    "108": 2.0,     # radial orbit → finite-time collapse (r0=5 collapses ~3.9s)
    "116": 3.0,     # isothermal sphere → blows up at ξ≈3.27
    "110": 3.0,     # Poisson → parabola gets large fast
    "200": 1.0,     # Abel equation → blow-up (u0=0.1 blows at x≈1.57)
    "201": 1.2,     # Riccati → blow-up (u0=0.1 blows at x≈1.90)
    "202": 1.5,     # Bernoulli → blow-up (u0=0.1 blows at x≈2.40)
    "204": 3.0,     # separable product → exponential growth
    "129": 3.0,     # string exponential branch (mu/T>0) → cap before blow-up
}


# ===================================================================
# Initial-condition defaults by ic_type
# ===================================================================

IC_DEFAULTS: dict[str, dict] = {
    "decay":       {"u0": 1.0, "v0": 0.0},
    "value":       {"u0": 1.0, "v0": 0.0},
    "bounded":     {"u0": 0.1, "v0": 0.0},
    "oscillatory": {"u0": 1.0, "v0": 0.0},
}


# ===================================================================
# Per-problem IC overrides (for singular / series-expansion starts)
# ===================================================================
# Each entry: callable(t_min, param_values) -> [u0] or [u0, v0]

IC_OVERRIDE: dict[str, Callable] = {
    # Radial orbit: start at r0=5.0 (far from singularity), v0=0 (free fall)
    "108": lambda t_min, p: [5.0, 0.0],
    # Binet orbit (corrected): start near circular orbit u_circ = G*M/L²
    "109": lambda t_min, p: [p["G"] * p["M"] / p["L"] ** 2, 0.0],
    # Blow-up ODEs: small positive IC to stay well before singularity
    "200": lambda t_min, p: [0.10],   # Abel: blowup at x≈1.57 for u0=0.1
    "201": lambda t_min, p: [0.10],   # Riccati: blowup at x≈1.90 for u0=0.1
    "202": lambda t_min, p: [0.10],   # Bernoulli: blowup at x≈2.40 for u0=0.1
    # Lane-Emden n=1: theta ~ 1 - xi^2/6, theta' ~ -xi/3
    "114": lambda t_min, p: [1.0 - t_min ** 2 / 6.0, -t_min / 3.0],
    # Lane-Emden n=3: theta ~ 1 - xi^2/6, theta' ~ -xi/3
    "115": lambda t_min, p: [1.0 - t_min ** 2 / 6.0, -t_min / 3.0],
    # Isothermal sphere: theta(0)=0, theta'(0)=0; series theta ~ -xi^2/6
    "116": lambda t_min, p: [-t_min ** 2 / 6.0, -t_min / 3.0],
    # Bessel J_0: J_0(x) ~ 1 - x^2/4, J_0'(x) ~ -x/2
    "118": lambda t_min, p: [1.0 - t_min ** 2 / 4.0, -t_min / 2.0],
    # Spherical acoustic: sinc(kr) ~ 1 - (kr)^2/6
    "128": lambda t_min, p: [
        1.0 - (p["k"] * t_min) ** 2 / 6.0,
        -p["k"] ** 2 * t_min / 3.0,
    ],
}


# ===================================================================
# Convenience helpers
# ===================================================================

def default_param_values(problem: ProblemDef) -> dict[str, float]:
    """Return midpoint of each parameter's range."""
    vals: dict[str, float] = {}
    for name, (lo, hi) in zip(problem.params, problem.param_ranges):
        vals[name] = 0.5 * (lo + hi)
    return vals


def default_t_max(problem: ProblemDef, param_values: dict[str, float]) -> float:
    """Heuristic integration endpoint based on IC type and parameters."""
    # Per-problem override takes priority
    if problem.id in T_MAX_OVERRIDE:
        return T_MAX_OVERRIDE[problem.id]

    # Special-case Schrödinger de119:
    #   u'' = c*u,  c = (2m/hbar^2)*(V-E)
    # A fixed long horizon can drive trajectories into an asymptotic
    # near-first-order regime (du ~ sqrt(c)*u for c>0), which hurts DE
    # identifiability despite low simulation error.
    if str(problem.id) == "119":
        m = float(param_values.get("m", float("nan")))
        hbar = float(param_values.get("hbar", float("nan")))
        V = float(param_values.get("V", float("nan")))
        E = float(param_values.get("E", float("nan")))
        if (
            math.isfinite(m)
            and math.isfinite(hbar)
            and math.isfinite(V)
            and math.isfinite(E)
            and hbar != 0.0
        ):
            c = 2.0 * m * (V - E) / (hbar * hbar)
            if math.isfinite(c) and c != 0.0:
                scale = math.sqrt(abs(c))
                if c < 0.0:
                    # Oscillatory branch: keep roughly ~10 periods.
                    t = 10.0 * 2.0 * math.pi / max(scale, 0.1)
                else:
                    # Exponential branch: keep ~5 e-folding lengths.
                    t = 5.0 / max(scale, 0.1)
                return float(min(20.0, max(1.0, t)))

    if str(problem.id) == "131":
        omega = float(param_values.get("omega", float("nan")))
        tension = float(param_values.get("T", float("nan")))
        mu = float(param_values.get("mu", float("nan")))
        if (
            math.isfinite(omega)
            and math.isfinite(tension)
            and math.isfinite(mu)
            and tension > 0.0
            and mu > 0.0
        ):
            k_eff = abs(omega) * math.sqrt(mu / tension)
            if math.isfinite(k_eff) and k_eff > 0.0:
                return float(min(60.0, max(5.0, 10.0 * 2.0 * math.pi / k_eff)))

    if problem.ic_type == "decay":
        # ~5 e-folding times
        rate_params = [abs(v) for v in param_values.values() if v != 0]
        if rate_params:
            return 5.0 / min(rate_params)
        return 10.0
    if problem.ic_type == "oscillatory":
        # look for omega-like parameter
        for name, val in param_values.items():
            if "omega" in name.lower() or name == "k":
                return 10.0 * 2.0 * math.pi / max(abs(val), 0.1)
        return 20.0
    if problem.ic_type == "bounded":
        return 20.0
    return 10.0


def resolve_ground_truth(gt: GroundTruth, param_values: dict[str, float]) -> dict[str, float]:
    """Resolve symbolic ground-truth terms to numeric values."""
    resolved: dict[str, float] = {}
    for term, spec in gt.terms.items():
        if isinstance(spec, str):
            resolved[term] = param_values[spec]
        elif callable(spec):
            resolved[term] = spec(param_values)
        else:
            resolved[term] = float(spec)
    return resolved
