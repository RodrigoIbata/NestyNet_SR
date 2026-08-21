# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Canonical-coordinate reduction for scalar first-order ODE discovery.

A one-parameter point symmetry V = xi(x,u) d/dx + eta(x,u) d/du admitted by an
ODE can be *rectified*: in canonical coordinates (r, s) with V(r)=0, V(s)=1 the
equation loses its dependence on the ignorable coordinate s, so a first-order
law collapses to a univariate problem

    ds/dr = G(r)      (charts whose invariant is the independent variable)
    dr/ds = G(r)      (charts whose invariant mixes u; autonomous in s)

This module compiles closed-form canonical charts for the affine generator
family, transforms trajectory ensembles into the chart, fits G with the same
mapping families the factorized symbolic search uses (power, exponential,
logarithm, rational, ...), and *pulls the fitted law back* to the original
coordinates.  The pullback manufactures exactly the compositional carrier
products (u/(1+x), u*log(1+x), exp(u), u^p, ...) that fixed dictionaries lack
— emitted as source-tagged library rows and whole-law proposals, they are
meant to be *fed to* the STLSQ library and the factorized search as
high-confidence seeds, complementing (not replacing) the structural search.

Scope: scalar first-order ODEs; diagonal-affine generators (shears a2, b1 are
rejected with NotImplementedError).  Generators can be supplied (e.g. from
determining certificates) or discovered from the trajectory ensemble alone via
the exp(eps*V) flow test.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from nestynet_sr.sr_core.bridges import DU, Add, Exp, Log, Mul, Node, Pow, U, Var
from nestynet_sr.sr_core.bridges import ConstNode, PowNode

from .de_certificates import generator_ensemble_support

_EPS = 1.0e-12


def _const(value: float) -> Node:
    return ConstNode(value=float(value))


def _scaled(node: Node, coeff: float) -> Node:
    if abs(float(coeff) - 1.0) <= 1.0e-12:
        return node
    return Mul(_const(coeff), node)


def _shifted(node: Node, shift: float) -> Node:
    if abs(float(shift)) <= 1.0e-12:
        return node
    return Add(_const(shift), node)


def _is_const_one(node: Node | None) -> bool:
    return isinstance(node, ConstNode) and abs(float(node.value) - 1.0) <= 1.0e-12


def _mul2(a: Node | None, b: Node | None) -> Node | None:
    if a is None or b is None:
        return None
    if _is_const_one(a):
        return b
    if _is_const_one(b):
        return a
    return Mul(a, b)


def _add2(a: Node | None, b: Node | None) -> Node | None:
    if a is None:
        return b
    if b is None:
        return a
    return Add(a, b)


def _sub2(a: Node | None, b: Node | None) -> Node | None:
    if b is None:
        return a
    neg = Mul(_const(-1.0), b)
    if a is None:
        return neg
    return Add(a, neg)


def _inv(node: Node) -> Node:
    if _is_const_one(node):
        return node
    if isinstance(node, PowNode) and isinstance(node.exponent, (int, float)):
        neg = -float(node.exponent)
        if abs(neg - 1.0) <= 1.0e-12:
            return node.base
        return Pow(node.base, neg)
    return Pow(node, -1.0)


# ---------------------------------------------------------------------------
# Canonical charts for diagonal-affine generators
# ---------------------------------------------------------------------------

@dataclass
class CanonicalChart:
    """Closed-form canonical coordinates (r, s) for one affine generator."""

    kind: str
    gen_coeffs: tuple[float, ...]
    orientation: str  # "ds_dr" or "dr_ds"
    r_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    s_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    domain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # reduced derivative from chain rule, given (x, u, u_x)
    reduced_derivative_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    human: str = ""
    params: dict[str, float] = field(default_factory=dict)


def compile_canonical_chart(gen_coeffs: Sequence[float], *, x_axis: int = 0) -> CanonicalChart:
    """Dispatch closed-form canonical coordinates for a diagonal-affine generator."""

    a0, a1, a2, b0, b1, b2 = (float(v) for v in gen_coeffs)
    if abs(a2) > 1.0e-10 or abs(b1) > 1.0e-10:
        raise NotImplementedError("shear generators (a2, b1) are not yet rectified")
    xi_zero = abs(a0) <= 1.0e-12 and abs(a1) <= 1.0e-12
    eta_zero = abs(b0) <= 1.0e-12 and abs(b2) <= 1.0e-12
    if xi_zero and eta_zero:
        raise ValueError("zero generator")
    coeffs = (a0, a1, a2, b0, b1, b2)

    if xi_zero:
        # V = (b0 + b2 u) d/du : r = x ; s = integral du/eta
        if abs(b2) <= 1.0e-12:
            s_fn = lambda x, u: u / b0
            dom = lambda x, u: np.isfinite(u)
            human = f"r=x, s=u/{b0:g}"
        else:
            s_fn = lambda x, u: np.log(np.abs(b0 + b2 * u)) / b2
            dom = lambda x, u: (b0 + b2 * u) > _EPS
            human = f"r=x, s=log(b0+b2*u)/b2 [b0={b0:g}, b2={b2:g}]"
        return CanonicalChart(
            kind="u_flow",
            gen_coeffs=coeffs,
            orientation="ds_dr",
            r_fn=lambda x, u: x,
            s_fn=s_fn,
            domain_fn=dom,
            reduced_derivative_fn=lambda x, u, ux: ux / (b0 + b2 * u),
            human=human,
            params={"b0": b0, "b2": b2},
        )

    if eta_zero:
        # V = (a0 + a1 x) d/dx : r = u ; s = integral dx/xi
        if abs(a1) <= 1.0e-12:
            s_fn = lambda x, u: x / a0
            dom = lambda x, u: np.isfinite(x)
            human = f"r=u, s=x/{a0:g}"
        else:
            s_fn = lambda x, u: np.log(np.abs(a0 + a1 * x)) / a1
            dom = lambda x, u: (a0 + a1 * x) > _EPS
            human = f"r=u, s=log(a0+a1*x)/a1 [a0={a0:g}, a1={a1:g}]"
        return CanonicalChart(
            kind="x_flow",
            gen_coeffs=coeffs,
            orientation="dr_ds",
            r_fn=lambda x, u: u,
            s_fn=s_fn,
            domain_fn=dom,
            reduced_derivative_fn=lambda x, u, ux: ux * (a0 + a1 * x),
            human=human,
            params={"a0": a0, "a1": a1},
        )

    if abs(a1) > 1.0e-12 and abs(b2) <= 1.0e-12:
        # V = a1 (x - x*) d/dx + b0 d/du : r = u - (b0/a1) log|x-x*| ; s = log|x-x*|/a1
        x_star = -a0 / a1
        c = b0 / a1
        r_fn = lambda x, u: u - c * np.log(np.abs(x - x_star))
        s_fn = lambda x, u: np.log(np.abs(x - x_star)) / a1
        return CanonicalChart(
            kind="mixed_xscale_utrans",
            gen_coeffs=coeffs,
            orientation="dr_ds",
            r_fn=r_fn,
            s_fn=s_fn,
            domain_fn=lambda x, u: (x - x_star) > _EPS,
            reduced_derivative_fn=lambda x, u, ux: a1 * (x - x_star) * ux - b0,
            human=f"r=u-({c:g})*log(x-{x_star:g}), s=log(x-{x_star:g})/{a1:g}",
            params={"x_star": x_star, "a1": a1, "b0": b0},
        )

    if abs(a1) <= 1.0e-12 and abs(b2) > 1.0e-12:
        # V = a0 d/dx + (b0 + b2 u) d/du : s = x/a0 ; r = (b0 + b2 u) exp(-b2 x / a0)
        k = b2 / a0
        r_fn = lambda x, u: (b0 + b2 * u) * np.exp(-k * x)
        return CanonicalChart(
            kind="xtrans_uflow",
            gen_coeffs=coeffs,
            orientation="dr_ds",
            r_fn=r_fn,
            s_fn=lambda x, u: x / a0,
            domain_fn=lambda x, u: np.isfinite(x),
            reduced_derivative_fn=lambda x, u, ux: (a0 * b2 * ux - b2 * (b0 + b2 * u))
            * np.exp(-k * x) / b2,
            human=f"r=(b0+b2*u)*exp(-{k:g}*x), s=x/{a0:g}",
            params={"a0": a0, "b0": b0, "b2": b2, "k": k},
        )

    # general diagonal case: V = a1 (x - x*) d/dx + b2 (u - u*) d/du
    x_star = -a0 / a1
    u_star = -b0 / b2
    lam = b2 / a1
    r_fn = lambda x, u: (u - u_star) * np.power(np.abs(x - x_star), -lam)
    s_fn = lambda x, u: np.log(np.abs(x - x_star)) / a1

    def _joint_reduced(x: np.ndarray, u: np.ndarray, ux: np.ndarray) -> np.ndarray:
        X = x - x_star
        return a1 * np.power(np.abs(X), -lam) * (X * ux - lam * (u - u_star))

    return CanonicalChart(
        kind="joint_scaling",
        gen_coeffs=coeffs,
        orientation="dr_ds",
        r_fn=r_fn,
        s_fn=s_fn,
        domain_fn=lambda x, u: (x - x_star) > _EPS,
        reduced_derivative_fn=_joint_reduced,
        human=f"r=(u-{u_star:g})*(x-{x_star:g})^(-{lam:g}), s=log(x-{x_star:g})/{a1:g}",
        params={"x_star": x_star, "u_star": u_star, "lam": lam, "a1": a1, "b2": b2},
    )


# ---------------------------------------------------------------------------
# Trajectory reduction
# ---------------------------------------------------------------------------

def reduce_trajectories(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    chart: CanonicalChart,
    *,
    u1_list: Sequence[np.ndarray] | None = None,
    min_survival_fraction: float = 0.5,
) -> dict[str, Any]:
    """Transform an ensemble into the chart and sample (r, G) pairs.

    The reduced derivative is computed by the chain rule when ``u1_list``
    (measured/exact u_x per trajectory) is given, otherwise by finite
    differences of (r, s) along each trajectory.
    """

    z_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    total, kept = 0, 0
    for i, (x_raw, u_raw) in enumerate(trajectories):
        x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        u = np.asarray(u_raw, dtype=np.float64).reshape(-1)
        total += x.size
        ok = chart.domain_fn(x, u) & np.isfinite(x) & np.isfinite(u)
        if u1_list is not None:
            u1 = np.asarray(u1_list[i], dtype=np.float64).reshape(-1)
            ok = ok & np.isfinite(u1)
        if int(ok.sum()) < 8:
            continue
        x, u = x[ok], u[ok]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = chart.r_fn(x, u)
            if u1_list is not None:
                g = chart.reduced_derivative_fn(x, u, u1[ok])
            else:
                s = chart.s_fn(x, u)
                if chart.orientation == "ds_dr":
                    g = np.gradient(s, r, edge_order=2)
                else:
                    g = np.gradient(r, s, edge_order=2)
        finite = np.isfinite(r) & np.isfinite(g)
        kept += int(finite.sum())
        z_parts.append(r[finite])
        g_parts.append(g[finite])
    if not z_parts:
        return {"status": "infeasible", "reason": "no finite reduced samples"}
    survival = kept / max(total, 1)
    if survival < float(min_survival_fraction):
        return {
            "status": "infeasible",
            "reason": f"chart domain keeps only {survival:.1%} of samples",
        }
    return {
        "status": "reduced",
        "z": np.concatenate(z_parts),
        "g": np.concatenate(g_parts),
        "survival_fraction": float(survival),
    }


# ---------------------------------------------------------------------------
# Univariate mapping-family fit (mirrors the FSS family set)
# ---------------------------------------------------------------------------

@dataclass
class UnivariateFit:
    """A fitted mapping family g ~ G(z) with a symbolic builder."""

    family: str
    params: dict[str, float]
    val_rmse_rel: float
    complexity: int
    # builds the AST of G applied to an argument AST; constants included
    build: Callable[[Node], Node]
    human: str = ""


def _rel_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    resid = pred - target
    scale = max(float(np.sqrt(np.mean(np.square(target)))), 1.0e-12)
    return float(np.sqrt(np.mean(np.square(resid))) / scale)


def _snap(value: float, *, max_denominator: int = 3, tol: float = 0.02) -> float:
    from fractions import Fraction

    frac = Fraction(value).limit_denominator(max_denominator)
    snapped = float(frac)
    if abs(snapped - value) <= tol * max(1.0, abs(value)):
        return snapped
    return float(value)


def fit_univariate_families(
    z: np.ndarray,
    g: np.ndarray,
    *,
    holdout_fraction: float = 0.25,
    seed: int = 0,
) -> list[UnivariateFit]:
    """Fit G(z) with the mapping families used by the factorized search.

    Families: constant, affine, shifted power a*(z+c)^p, exponential
    a*exp(k z)+b, logarithm a*log(z+c)+b, rational (a+b z)/(1+c z).
    Returns fits ranked by held-out relative RMSE with complexity tie-break.
    """

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(z.size)
    n_val = max(int(z.size * float(holdout_fraction)), 8)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    zf, gf = z[fit_idx], g[fit_idx]
    zv, gv = z[val_idx], g[val_idx]
    fits: list[UnivariateFit] = []

    def _lin_fit(cols_f: list[np.ndarray], cols_v: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
        A = np.column_stack(cols_f)
        finite = np.isfinite(A).all(axis=1) & np.isfinite(gf)
        if int(finite.sum()) < max(2 * len(cols_f), 8):
            return None
        coef, *_ = np.linalg.lstsq(A[finite], gf[finite], rcond=None)
        Av = np.column_stack(cols_v)
        fin_v = np.isfinite(Av).all(axis=1) & np.isfinite(gv)
        if int(fin_v.sum()) < 4:
            return None
        return coef, _rel_rmse(Av[fin_v] @ coef, gv[fin_v])

    # constant
    out = _lin_fit([np.ones_like(zf)], [np.ones_like(zv)])
    if out is not None:
        (c,), err = out[0], out[1]
        fits.append(UnivariateFit(
            "constant", {"c": float(c)}, err, 1,
            lambda arg, c=float(c): _const(c),
            human=f"{c:.6g}",
        ))

    # affine
    out = _lin_fit([np.ones_like(zf), zf], [np.ones_like(zv), zv])
    if out is not None:
        (c0, c1), err = out[0], out[1]
        def _build_affine(arg: Node, c0=float(c0), c1=float(c1)) -> Node:
            if abs(c0) <= 1.0e-10 * max(1.0, abs(c1)):
                return _scaled(arg, c1)
            return Add(_const(c0), _scaled(arg, c1))

        fits.append(UnivariateFit(
            "affine", {"c0": float(c0), "c1": float(c1)}, err, 2,
            _build_affine,
            human=f"{c0:.6g} + {c1:.6g}*z",
        ))

    # shifted power: a * (z + c)^p  (+ b)
    z_lo = float(np.min(z))
    shift_grid = np.unique(np.concatenate([
        np.asarray([0.0, 1.0, -1.0, 0.5, 2.0]),
        np.linspace(-z_lo + 0.1, -z_lo + 3.0, 7),
    ]))
    best_pow = None
    for c in shift_grid:
        base_f, base_v = zf + c, zv + c
        if np.min(base_f) <= 1.0e-9 or np.min(base_v) <= 1.0e-9:
            continue
        mask = np.abs(gf) > 1.0e-12
        if int(mask.sum()) < 16:
            continue
        sign = float(np.sign(np.median(gf[mask])))
        logg = np.log(np.abs(gf[mask]))
        logz = np.log(base_f[mask])
        A = np.column_stack([np.ones_like(logz), logz])
        coef, *_ = np.linalg.lstsq(A, logg, rcond=None)
        p = _snap(float(coef[1]))
        a = sign * math.exp(float(coef[0]))
        # refit amplitude linearly with snapped exponent
        col_f, col_v = np.power(base_f, p), np.power(base_v, p)
        out = _lin_fit([col_f], [col_v])
        if out is None:
            continue
        (a_lin,), err = out[0], out[1]
        cand = (err, c, p, float(a_lin))
        if best_pow is None or cand[0] < best_pow[0]:
            best_pow = cand
    if best_pow is not None:
        err, c, p, a = best_pow
        c_s = _snap(c)
        if c_s != c:
            col_f, col_v = np.power(zf + c_s, p), np.power(zv + c_s, p)
            if np.min(zf + c_s) > 1.0e-9 and np.min(zv + c_s) > 1.0e-9:
                out = _lin_fit([col_f], [col_v])
                if out is not None and out[1] <= err * 1.5:
                    a, err, c = float(out[0][0]), out[1], c_s

        def _build_pow(arg: Node, a=float(a), c=float(c), p=float(p)) -> Node:
            base = _shifted(arg, c)
            return _scaled(Pow(base, p), a)

        fits.append(UnivariateFit(
            "shifted_power", {"a": float(a), "c": float(c), "p": float(p)}, err, 4 if abs(c) > 1e-12 else 3,
            _build_pow, human=f"{a:.6g}*(z+{c:g})^{p:g}",
        ))

    # exponential: a*exp(k z) + b
    z_span = max(float(np.ptp(z)), 1.0e-6)
    best_exp = None
    for k in np.concatenate([np.linspace(-6.0 / z_span, 6.0 / z_span, 41), [-1.0, 1.0]]):
        if abs(k) < 1.0e-9:
            continue
        col_f, col_v = np.exp(k * zf), np.exp(k * zv)
        if not (np.all(np.isfinite(col_f)) and np.all(np.isfinite(col_v))):
            continue
        out = _lin_fit([col_f, np.ones_like(zf)], [col_v, np.ones_like(zv)])
        if out is None:
            continue
        (a, b), err = out[0], out[1]
        if best_exp is None or err < best_exp[0]:
            best_exp = (err, float(k), float(a), float(b))
    if best_exp is not None:
        err, k, a, b = best_exp
        k_s = _snap(k)
        if k_s != k and abs(k_s) > 1.0e-9:
            col_f, col_v = np.exp(k_s * zf), np.exp(k_s * zv)
            out = _lin_fit([col_f, np.ones_like(zf)], [col_v, np.ones_like(zv)])
            if out is not None and out[1] <= err * 1.5:
                (a, b), err, k = (float(out[0][0]), float(out[0][1])), out[1], k_s
                a, b = float(a), float(b)

        def _build_exp(arg: Node, a=float(a), k=float(k), b=float(b)) -> Node:
            core = _scaled(Exp(_scaled(arg, k)), a)
            return core if abs(b) <= 1.0e-10 else Add(_const(b), core)

        fits.append(UnivariateFit(
            "exponential", {"a": float(a), "k": float(k), "b": float(b)}, err, 4,
            _build_exp, human=f"{a:.6g}*exp({k:g}*z) + {b:.6g}",
        ))

    # logarithm: a*log(z + c) + b
    best_log = None
    for c in shift_grid:
        base_f, base_v = zf + c, zv + c
        if np.min(base_f) <= 1.0e-9 or np.min(base_v) <= 1.0e-9:
            continue
        out = _lin_fit([np.log(base_f), np.ones_like(zf)], [np.log(base_v), np.ones_like(zv)])
        if out is None:
            continue
        (a, b), err = out[0], out[1]
        if best_log is None or err < best_log[0]:
            best_log = (err, float(c), float(a), float(b))
    if best_log is not None:
        err, c, a, b = best_log
        c_s = _snap(c)
        if c_s != c and np.min(zf + c_s) > 1.0e-9 and np.min(zv + c_s) > 1.0e-9:
            out = _lin_fit([np.log(zf + c_s), np.ones_like(zf)], [np.log(zv + c_s), np.ones_like(zv)])
            if out is not None and out[1] <= err * 1.5:
                (a, b), err, c = (float(out[0][0]), float(out[0][1])), out[1], c_s
                a, b = float(a), float(b)

        def _build_log(arg: Node, a=float(a), c=float(c), b=float(b)) -> Node:
            base = _shifted(arg, c)
            core = _scaled(Log(base), a)
            return core if abs(b) <= 1.0e-10 else Add(_const(b), core)

        fits.append(UnivariateFit(
            "logarithm", {"a": float(a), "c": float(c), "b": float(b)}, err, 4,
            _build_log, human=f"{a:.6g}*log(z+{c:g}) + {b:.6g}",
        ))

    # rational: (a + b z) / (1 + c z)
    best_rat = None
    for c in np.linspace(-2.0 / z_span, 2.0 / z_span, 33):
        denom_f, denom_v = 1.0 + c * zf, 1.0 + c * zv
        if np.min(np.abs(denom_f)) < 1.0e-6 or np.min(np.abs(denom_v)) < 1.0e-6:
            continue
        out = _lin_fit(
            [1.0 / denom_f, zf / denom_f], [1.0 / denom_v, zv / denom_v]
        )
        if out is None:
            continue
        (a, b), err = out[0], out[1]
        if best_rat is None or err < best_rat[0]:
            best_rat = (err, float(c), float(a), float(b))
    if best_rat is not None:
        err, c, a, b = best_rat

        def _build_rat(arg: Node, a=float(a), b=float(b), c=float(c)) -> Node:
            num = Add(_const(a), _scaled(arg, b))
            den = Add(_const(1.0), _scaled(arg, c))
            return Mul(num, Pow(den, -1.0))

        fits.append(UnivariateFit(
            "rational", {"a": float(a), "b": float(b), "c": float(c)}, err, 5,
            _build_rat, human=f"({a:.6g} + {b:.6g}*z)/(1 + {c:.6g}*z)",
        ))

    fits.sort(key=lambda f: (round(float(f.val_rmse_rel), 9), int(f.complexity)))
    return fits


def select_univariate_fit(
    fits: Sequence[UnivariateFit],
    *,
    rel_tolerance_factor: float = 3.0,
    min_rmse_floor: float = 1.0e-9,
) -> UnivariateFit | None:
    """Prefer the *simplest* family within tolerance of the best fit."""

    if not fits:
        return None
    best = min(float(f.val_rmse_rel) for f in fits)
    band = max(best * float(rel_tolerance_factor), float(min_rmse_floor))
    eligible = [f for f in fits if float(f.val_rmse_rel) <= band]
    eligible.sort(key=lambda f: (int(f.complexity), float(f.val_rmse_rel)))
    return eligible[0]


# ---------------------------------------------------------------------------
# Pullback to original coordinates
# ---------------------------------------------------------------------------

def _xi_ast(chart: CanonicalChart, x_ast: Node) -> Node:
    a0, a1 = chart.gen_coeffs[0], chart.gen_coeffs[1]
    if abs(a1) <= 1.0e-12:
        return _const(a0)
    if abs(a0) <= 1.0e-12:
        return _scaled(x_ast, a1)
    return Add(_const(a0), _scaled(x_ast, a1))


def _r_ast(chart: CanonicalChart, x_ast: Node, u_ast: Node) -> Node:
    """Symbolic invariant r(x, u) for the chart."""

    kind = chart.kind
    p = chart.params
    if kind == "u_flow":
        return x_ast
    if kind == "x_flow":
        return u_ast
    if kind == "mixed_xscale_utrans":
        shifted = _shifted(x_ast, -p["x_star"])
        return Add(u_ast, _scaled(Log(shifted), -p["b0"] / p["a1"]))
    if kind == "xtrans_uflow":
        eta = _shifted(_scaled(u_ast, p["b2"]), p["b0"])
        return Mul(eta, Exp(_scaled(x_ast, -p["k"])))
    if kind == "joint_scaling":
        shifted_x = _shifted(x_ast, -p["x_star"])
        shifted_u = _shifted(u_ast, -p["u_star"])
        return Mul(shifted_u, Pow(shifted_x, -p["lam"]))
    raise NotImplementedError(f"no symbolic invariant for chart kind {kind!r}")


def pullback_rhs_ast(chart: CanonicalChart, fit: UnivariateFit, *, x_axis: int = 0) -> dict[str, Any]:
    """Reconstruct u_x = RHS(x, u) in original coordinates from the reduced fit.

    Returns the full RHS AST (with fitted constants) plus coefficient-stripped
    library rows suitable for STLSQ dictionaries and factorized-search seeds.
    """

    x_ast = Var(int(x_axis))
    u_ast = U()
    G = fit.build(_r_ast(chart, x_ast, u_ast))
    kind = chart.kind
    p = chart.params
    rows: list[tuple[Node, str, str]] = []

    def _row(term: Node, family: str) -> None:
        rows.append((term, "gs_de_reduction", family))

    if kind == "u_flow":
        b0, b2 = p["b0"], p["b2"]
        # u_x = G(x) * (b0 + b2 u)
        eta = _shifted(_scaled(u_ast, b2), b0)
        rhs = Mul(G, eta)
        G_free = fit.build(x_ast)
        if abs(b2) > 1.0e-12:
            _row(Mul(G_free, u_ast), f"reduction_{fit.family}_times_u")
        if abs(b0) > 1.0e-12:
            _row(G_free, f"reduction_{fit.family}_source")
    elif kind == "x_flow":
        # u_x = G(u) / xi(x)
        xi = _xi_ast(chart, x_ast)
        G_free = fit.build(u_ast)
        if abs(chart.gen_coeffs[1]) <= 1.0e-12:
            rhs = _scaled(G_free, 1.0 / chart.gen_coeffs[0])
            _row(fit.build(u_ast), f"reduction_{fit.family}_state")
        else:
            rhs = Mul(G_free, Pow(xi, -1.0))
            _row(Mul(G_free, Pow(xi, -1.0)), f"reduction_{fit.family}_over_xi")
    elif kind == "mixed_xscale_utrans":
        # u_x = (G(r) + b0) / (a1 (x - x*))
        shifted = _shifted(x_ast, -p["x_star"])
        rhs = Mul(Add(G, _const(p["b0"])), Pow(_scaled(shifted, p["a1"]), -1.0))
        _row(Mul(G, Pow(shifted, -1.0)), f"reduction_{fit.family}_mixed")
        _row(Pow(shifted, -1.0), "reduction_mixed_source")
    elif kind == "xtrans_uflow":
        # u_x = (G(r) + b2 r) * exp(k x) / (a0 b2) with r = (b0+b2 u) exp(-k x)
        a0, b0, b2, k = p["a0"], p["b0"], p["b2"], p["k"]
        r_sym = _r_ast(chart, x_ast, u_ast)
        rhs = _scaled(
            Mul(Add(G, _scaled(r_sym, b2)), Exp(_scaled(x_ast, k))), 1.0 / (a0 * b2)
        )
        _row(Mul(G, Exp(_scaled(x_ast, k))), f"reduction_{fit.family}_xtrans")
    elif kind == "joint_scaling":
        # u_x = (G(r)/a1 + lam*(u-u*) (x-x*)^-lam) * (x-x*)^(lam-1)
        lam, a1 = p["lam"], p["a1"]
        shifted_x = _shifted(x_ast, -p["x_star"])
        shifted_u = _shifted(u_ast, -p["u_star"])
        inner = Add(
            _scaled(G, 1.0 / a1),
            _scaled(Mul(shifted_u, Pow(shifted_x, -lam)), lam),
        )
        rhs = Mul(inner, Pow(shifted_x, lam - 1.0))
        _row(Mul(G, Pow(shifted_x, lam - 1.0)), f"reduction_{fit.family}_scaling")
        _row(Mul(shifted_u, Pow(shifted_x, -1.0)), "reduction_scaling_carrier")
    else:
        raise NotImplementedError(f"no pullback for chart kind {kind!r}")

    return {
        "rhs_ast": rhs,
        "library_rows": rows,
        "chart_kind": kind,
        "chart_human": chart.human,
        "fit_family": fit.family,
        "fit_human": fit.human,
        "fit_val_rmse_rel": float(fit.val_rmse_rel),
    }


# ---------------------------------------------------------------------------
# Ensemble-only generator discovery (no candidate equation required)
# ---------------------------------------------------------------------------

_BASIS_COEFFS = (
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # d_x
    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),  # x d_x
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # d_u
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # u d_u
)


def _ray_key(coeffs: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(coeffs, dtype=float)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs <= 0:
        return tuple(arr)
    arr = arr / max_abs
    first = next((v for v in arr if abs(v) > 1e-9), 1.0)
    if first < 0:
        arr = -arr
    return tuple(round(float(v), 4) for v in arr)


def discover_ensemble_generators(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    support_rel_tol: float = 5.0e-3,
    pair_ratios: Sequence[float] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
    refine: bool = True,
    max_generators: int = 4,
) -> list[dict[str, Any]]:
    """Find affine generators the ensemble itself supports, no equation needed.

    Tests the diagonal-affine singles and two-element integer combinations via
    the exp(eps*V) flow test; promising combination ratios are refined
    continuously and snapped.  This resolves the chicken-and-egg of the
    determining certificate: symmetry evidence is measured from data first,
    and can then steer library construction and factorized-search proposals.
    """

    candidates: list[tuple[float, ...]] = list(_BASIS_COEFFS)
    for (i, base_i), (j, base_j) in itertools.combinations(enumerate(_BASIS_COEFFS), 2):
        for ratio in pair_ratios:
            combo = tuple(
                float(vi + ratio * vj) for vi, vj in zip(base_i, base_j)
            )
            candidates.append(combo)

    seen: set[tuple[float, ...]] = set()
    accepted: list[dict[str, Any]] = []
    scored: list[tuple[float, tuple[float, ...], dict[str, Any]]] = []
    for coeffs in candidates:
        key = _ray_key(coeffs)
        if key in seen:
            continue
        seen.add(key)
        support = generator_ensemble_support(
            trajectories, coeffs, support_rel_tol=float(support_rel_tol)
        )
        if support.get("status") != "tested":
            continue
        scored.append((float(support.get("median_best_rel_rms", math.inf)), coeffs, support))

    scored.sort(key=lambda item: item[0])
    for median_val, coeffs, support in scored:
        if refine and not support.get("supported") and median_val < 20.0 * float(support_rel_tol):
            refined = _refine_combination(
                trajectories, coeffs, support_rel_tol=float(support_rel_tol)
            )
            if refined is not None:
                coeffs, support = refined
                median_val = float(support.get("median_best_rel_rms", math.inf))
        if not support.get("supported"):
            continue
        key = _ray_key(coeffs)
        if any(_ray_key(a["coefficients"]) == key for a in accepted):
            continue
        accepted.append(
            {
                "coefficients": tuple(float(v) for v in coeffs),
                "median_best_rel_rms": median_val,
                "fraction_pairs_supporting": support.get("fraction_pairs_supporting"),
                "source": "ensemble_flow_scan",
            }
        )
        if len(accepted) >= int(max_generators):
            break
    return accepted


def _refine_combination(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    coeffs: Sequence[float],
    *,
    support_rel_tol: float,
) -> tuple[tuple[float, ...], dict[str, Any]] | None:
    """Refine the relative weight of the second active component of a combo."""

    from scipy.optimize import minimize_scalar

    arr = np.asarray(coeffs, dtype=float)
    active = np.flatnonzero(np.abs(arr) > 1e-9)
    if active.size != 2:
        return None
    i, j = int(active[0]), int(active[1])
    base_ratio = arr[j] / arr[i]

    def _objective(log_ratio: float) -> float:
        trial = arr.copy()
        trial[j] = trial[i] * math.copysign(math.exp(log_ratio), base_ratio)
        support = generator_ensemble_support(
            trajectories, trial, support_rel_tol=float(support_rel_tol), n_eps=61
        )
        return float(support.get("median_best_rel_rms", math.inf))

    try:
        opt = minimize_scalar(
            _objective,
            bounds=(math.log(abs(base_ratio)) - 1.2, math.log(abs(base_ratio)) + 1.2),
            method="bounded",
            options={"xatol": 1.0e-3},
        )
    except Exception:
        return None
    if not np.isfinite(float(opt.fun)):
        return None
    ratio = math.copysign(math.exp(float(opt.x)), base_ratio)
    snapped = _snap(ratio, max_denominator=3, tol=0.05)
    trial = arr.copy()
    trial[j] = trial[i] * snapped
    support = generator_ensemble_support(
        trajectories, trial, support_rel_tol=float(support_rel_tol)
    )
    if support.get("supported"):
        return tuple(float(v) for v in trial), support
    if snapped != ratio:
        trial[j] = arr[i] * ratio
        support = generator_ensemble_support(
            trajectories, trial, support_rel_tol=float(support_rel_tol)
        )
        if support.get("supported"):
            return tuple(float(v) for v in trial), support
    return None


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _order2_reduction_pass(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    gen_meta: Sequence[dict[str, Any]],
    gen_list: Sequence[tuple[float, ...]],
    u1_list: Sequence[np.ndarray] | None,
    u2_list: Sequence[np.ndarray] | None,
    x_axis: int,
    fit_accept_rel_rmse: float,
    seed: int,
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    all_rows: list[tuple[Any, str, str]] = []
    reports: list[dict[str, Any]] = []
    for meta, coeffs in zip(gen_meta, gen_list):
        entry: dict[str, Any] = {"generator": dict(meta), "order": 2}
        try:
            chart = compile_canonical_chart(coeffs, x_axis=int(x_axis))
        except NotImplementedError as exc:
            entry["status"] = "chart_unsupported"
            entry["reason"] = str(exc)
            reports.append(entry)
            continue
        entry["chart"] = {"kind": chart.kind, "human": chart.human}
        reduced = reduce_trajectories_order2(
            trajectories, chart, u1_list=u1_list, u2_list=u2_list
        )
        if reduced.get("status") != "reduced":
            entry["status"] = "reduction_infeasible"
            entry["reason"] = reduced.get("reason")
            reports.append(entry)
            continue
        fit = fit_reduced_first_order(
            reduced["r"], reduced["v"], reduced["dvdr"], seed=int(seed)
        )
        if fit is None or float(fit["val_rmse_rel"]) > float(fit_accept_rel_rmse):
            entry["status"] = "no_acceptable_reduced_fit"
            if fit is not None:
                entry["best_fit"] = {"human": fit["human"], "val_rmse_rel": fit["val_rmse_rel"]}
            reports.append(entry)
            continue
        try:
            pulled = pullback_order2(chart, fit, x_axis=int(x_axis))
        except (ValueError, NotImplementedError) as exc:
            entry["status"] = "pullback_failed"
            entry["reason"] = str(exc)
            reports.append(entry)
            continue
        entry["status"] = "proposed"
        entry["reduced_fit"] = {
            "engine": "stlsq_rv",
            "equation": pulled["reduced_equation"],
            "val_rmse_rel": float(fit["val_rmse_rel"]),
        }
        entry["pullback_human"] = repr(pulled["rhs_ast"])
        proposals.append({**pulled, "generator": dict(meta), "order": 2})
        for row in pulled["library_rows"]:
            rep = repr(row[0])
            if not any(repr(r0) == rep for r0, _s, _f in all_rows):
                all_rows.append(row)
        reports.append(entry)

    all_rows = _dedupe_rows_numeric(all_rows, trajectories, u1_list=u1_list)
    return {
        "status": "ok",
        "order": 2,
        "n_generators": len(gen_list),
        "proposals": proposals,
        "library_rows": all_rows,
        "reports": reports,
    }


def symmetry_reduction_proposals(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    u1_list: Sequence[np.ndarray] | None = None,
    u2_list: Sequence[np.ndarray] | None = None,
    generators: Sequence[Sequence[float]] | None = None,
    x_axis: int = 0,
    order: int = 1,
    fit_accept_rel_rmse: float = 2.0e-2,
    support_rel_tol: float = 5.0e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Generator discovery -> chart reduction -> reduced fit -> pullback.

    ``order=1``: the reduced law is univariate (quadrature) and fitted with the
    factorized-search mapping families.  ``order=2``: the reduced law is a
    first-order ODE dv/dr = H(r, v) in the differential invariants, fitted by
    STLSQ over an (r, v) dictionary; the pullback uses the linearity of dv/dr
    in u_xx.  Returns whole-law proposals and library rows intended to augment
    the STLSQ dictionary and seed the factorized search.
    """

    order_i = int(order)
    if order_i not in (1, 2):
        raise ValueError(f"symmetry reduction supports order 1 or 2; got {order_i}")

    if generators is None:
        found = discover_ensemble_generators(
            trajectories, support_rel_tol=float(support_rel_tol)
        )
        gen_list = [g["coefficients"] for g in found]
        gen_meta = found
    else:
        gen_list = [tuple(float(v) for v in g) for g in generators]
        gen_meta = [{"coefficients": g, "source": "provided"} for g in gen_list]

    if order_i == 2:
        return _order2_reduction_pass(
            trajectories,
            gen_meta=gen_meta,
            gen_list=gen_list,
            u1_list=u1_list,
            u2_list=u2_list,
            x_axis=int(x_axis),
            fit_accept_rel_rmse=float(fit_accept_rel_rmse),
            seed=int(seed),
        )

    proposals: list[dict[str, Any]] = []
    all_rows: list[tuple[Any, str, str]] = []
    reports: list[dict[str, Any]] = []
    for meta, coeffs in zip(gen_meta, gen_list):
        entry: dict[str, Any] = {"generator": dict(meta)}
        try:
            chart = compile_canonical_chart(coeffs, x_axis=int(x_axis))
        except NotImplementedError as exc:
            entry["status"] = "chart_unsupported"
            entry["reason"] = str(exc)
            reports.append(entry)
            continue
        entry["chart"] = {"kind": chart.kind, "human": chart.human, "orientation": chart.orientation}
        reduced = reduce_trajectories(trajectories, chart, u1_list=u1_list)
        if reduced.get("status") != "reduced":
            entry["status"] = "reduction_infeasible"
            entry["reason"] = reduced.get("reason")
            reports.append(entry)
            continue
        fits = fit_univariate_families(reduced["z"], reduced["g"], seed=int(seed))
        fit = select_univariate_fit(fits)
        entry["fits"] = [
            {"family": f.family, "val_rmse_rel": f.val_rmse_rel, "human": f.human}
            for f in fits[:4]
        ]
        if fit is None or float(fit.val_rmse_rel) > float(fit_accept_rel_rmse):
            entry["status"] = "no_acceptable_reduced_fit"
            reports.append(entry)
            continue
        pulled = pullback_rhs_ast(chart, fit, x_axis=int(x_axis))
        entry["status"] = "proposed"
        entry["reduced_fit"] = {"family": fit.family, "human": fit.human,
                                "params": dict(fit.params),
                                "val_rmse_rel": float(fit.val_rmse_rel)}
        entry["pullback_human"] = repr(pulled["rhs_ast"])
        proposals.append({**pulled, "generator": dict(meta)})
        for row in pulled["library_rows"]:
            rep = repr(row[0])
            if not any(repr(r0) == rep for r0, _s, _f in all_rows):
                all_rows.append(row)
        reports.append(entry)

    all_rows = _dedupe_rows_numeric(all_rows, trajectories)

    return {
        "status": "ok",
        "n_generators": len(gen_list),
        "proposals": proposals,
        "library_rows": all_rows,
        "reports": reports,
    }




# ---------------------------------------------------------------------------
# Order-2 reduction: a second-order ODE with symmetry V collapses to a
# FIRST-order equation dv/dr = H(r, v) in the differential invariants
# r (zeroth order) and v = ds/dr (first order).  u_xx enters dv/dr linearly,
# so the discovered H pulls back to u_xx = (H - A)/B with closed-form A, B.
# ---------------------------------------------------------------------------

def chart_partials(chart: CanonicalChart) -> Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]]:
    """Numeric first/second partial derivatives of (r, s) for a chart."""

    kind = chart.kind
    p = chart.params

    def _z(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def _o(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    if kind == "u_flow":
        b0, b2 = p["b0"], p["b2"]

        def _parts(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
            eta = b0 + b2 * u
            return {
                "r_x": _o(x), "r_u": _z(x), "r_xx": _z(x), "r_xu": _z(x), "r_uu": _z(x),
                "s_x": _z(x), "s_u": 1.0 / eta, "s_xx": _z(x), "s_xu": _z(x),
                "s_uu": -b2 / eta**2,
            }

        return _parts

    if kind == "x_flow":
        a0, a1 = p["a0"], p["a1"]

        def _parts(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
            xi = a0 + a1 * x
            return {
                "r_x": _z(x), "r_u": _o(x), "r_xx": _z(x), "r_xu": _z(x), "r_uu": _z(x),
                "s_x": 1.0 / xi, "s_u": _z(x), "s_xx": -a1 / xi**2, "s_xu": _z(x),
                "s_uu": _z(x),
            }

        return _parts

    if kind == "mixed_xscale_utrans":
        x_star, a1, b0 = p["x_star"], p["a1"], p["b0"]
        c = b0 / a1

        def _parts(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
            X = x - x_star
            return {
                "r_x": -c / X, "r_u": _o(x), "r_xx": c / X**2, "r_xu": _z(x), "r_uu": _z(x),
                "s_x": 1.0 / (a1 * X), "s_u": _z(x), "s_xx": -1.0 / (a1 * X**2),
                "s_xu": _z(x), "s_uu": _z(x),
            }

        return _parts

    if kind == "xtrans_uflow":
        a0, b0, b2, k = p["a0"], p["b0"], p["b2"], p["k"]

        def _parts(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
            E = np.exp(-k * x)
            eta = b0 + b2 * u
            return {
                "r_x": -k * eta * E, "r_u": b2 * E, "r_xx": k**2 * eta * E,
                "r_xu": -k * b2 * E, "r_uu": _z(x),
                "s_x": np.full_like(x, 1.0 / a0), "s_u": _z(x), "s_xx": _z(x),
                "s_xu": _z(x), "s_uu": _z(x),
            }

        return _parts

    if kind == "joint_scaling":
        x_star, u_star, lam, a1 = p["x_star"], p["u_star"], p["lam"], p["a1"]

        def _parts(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
            X = x - x_star
            Uu = u - u_star
            return {
                "r_x": -lam * Uu * X ** (-lam - 1.0), "r_u": X ** (-lam),
                "r_xx": lam * (lam + 1.0) * Uu * X ** (-lam - 2.0),
                "r_xu": -lam * X ** (-lam - 1.0), "r_uu": _z(x),
                "s_x": 1.0 / (a1 * X), "s_u": _z(x), "s_xx": -1.0 / (a1 * X**2),
                "s_xu": _z(x), "s_uu": _z(x),
            }

        return _parts

    raise NotImplementedError(f"no partials for chart kind {kind!r}")


def _chart_partial_asts(chart: CanonicalChart, x_ast: Node, u_ast: Node) -> dict[str, Node | None]:
    """Symbolic partials of (r, s); ``None`` marks identically-zero entries."""

    kind = chart.kind
    p = chart.params
    if kind == "u_flow":
        b0, b2 = p["b0"], p["b2"]
        eta = _shifted(_scaled(u_ast, b2), b0)
        return {
            "r_x": _const(1.0), "r_u": None, "r_xx": None, "r_xu": None, "r_uu": None,
            "s_x": None, "s_u": _inv(eta), "s_xx": None, "s_xu": None,
            "s_uu": _scaled(Pow(eta, -2.0), -b2) if abs(b2) > 1.0e-12 else None,
        }
    if kind == "x_flow":
        a0, a1 = p["a0"], p["a1"]
        xi = _shifted(_scaled(x_ast, a1), a0)
        return {
            "r_x": None, "r_u": _const(1.0), "r_xx": None, "r_xu": None, "r_uu": None,
            "s_x": _inv(xi), "s_u": None,
            "s_xx": _scaled(Pow(xi, -2.0), -a1) if abs(a1) > 1.0e-12 else None,
            "s_xu": None, "s_uu": None,
        }
    if kind == "mixed_xscale_utrans":
        x_star, a1, b0 = p["x_star"], p["a1"], p["b0"]
        c = b0 / a1
        X = _shifted(x_ast, -x_star)
        return {
            "r_x": _scaled(Pow(X, -1.0), -c), "r_u": _const(1.0),
            "r_xx": _scaled(Pow(X, -2.0), c), "r_xu": None, "r_uu": None,
            "s_x": _scaled(Pow(X, -1.0), 1.0 / a1), "s_u": None,
            "s_xx": _scaled(Pow(X, -2.0), -1.0 / a1), "s_xu": None, "s_uu": None,
        }
    if kind == "xtrans_uflow":
        a0, b0, b2, k = p["a0"], p["b0"], p["b2"], p["k"]
        E = Exp(_scaled(x_ast, -k))
        eta = _shifted(_scaled(u_ast, b2), b0)
        return {
            "r_x": _scaled(Mul(eta, E), -k), "r_u": _scaled(E, b2),
            "r_xx": _scaled(Mul(eta, E), k**2), "r_xu": _scaled(E, -k * b2), "r_uu": None,
            "s_x": _const(1.0 / a0), "s_u": None, "s_xx": None, "s_xu": None, "s_uu": None,
        }
    if kind == "joint_scaling":
        x_star, u_star, lam, a1 = p["x_star"], p["u_star"], p["lam"], p["a1"]
        X = _shifted(x_ast, -x_star)
        Uu = _shifted(u_ast, -u_star)
        return {
            "r_x": _scaled(Mul(Uu, Pow(X, -lam - 1.0)), -lam),
            "r_u": Pow(X, -lam),
            "r_xx": _scaled(Mul(Uu, Pow(X, -lam - 2.0)), lam * (lam + 1.0)),
            "r_xu": _scaled(Pow(X, -lam - 1.0), -lam), "r_uu": None,
            "s_x": _scaled(Pow(X, -1.0), 1.0 / a1), "s_u": None,
            "s_xx": _scaled(Pow(X, -2.0), -1.0 / a1), "s_xu": None, "s_uu": None,
        }
    raise NotImplementedError(f"no symbolic partials for chart kind {kind!r}")


def reduce_trajectories_order2(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    chart: CanonicalChart,
    *,
    u1_list: Sequence[np.ndarray] | None = None,
    u2_list: Sequence[np.ndarray] | None = None,
    min_survival_fraction: float = 0.5,
    return_per_trajectory: bool = False,
) -> dict[str, Any]:
    """Sample the order-2 invariants (r, v = ds/dr, dv/dr) along an ensemble.

    ``u1``/``u2`` per trajectory give exact chain-rule invariants; missing
    derivatives fall back to finite differences along the trajectory.  With
    ``return_per_trajectory`` the reduced curves are also returned separately
    (as ``per_trajectory`` = list of ``(r_i, v_i, dvdr_i)``), so the reduced
    first-order equation can be treated as its own ensemble for a second
    (cascade) symmetry discovery.
    """

    parts_fn = chart_partials(chart)
    r_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    per_traj: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total, kept = 0, 0
    for i, (x_raw, u_raw) in enumerate(trajectories):
        x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        u = np.asarray(u_raw, dtype=np.float64).reshape(-1)
        total += x.size
        if u1_list is not None:
            u1 = np.asarray(u1_list[i], dtype=np.float64).reshape(-1)
        else:
            u1 = np.gradient(u, x, edge_order=2)
        u2 = None
        if u2_list is not None:
            u2 = np.asarray(u2_list[i], dtype=np.float64).reshape(-1)
        ok = chart.domain_fn(x, u) & np.isfinite(x) & np.isfinite(u) & np.isfinite(u1)
        if u2 is not None:
            ok = ok & np.isfinite(u2)
        if int(ok.sum()) < 12:
            continue
        xk, uk, u1k = x[ok], u[ok], u1[ok]
        u2k = u2[ok] if u2 is not None else None
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            parts = parts_fn(xk, uk)
            P = parts["s_x"] + parts["s_u"] * u1k
            Q = parts["r_x"] + parts["r_u"] * u1k
            v = P / Q
            r = chart.r_fn(xk, uk)
            if u2k is not None:
                P0p = parts["s_xx"] + 2.0 * parts["s_xu"] * u1k + parts["s_uu"] * u1k**2
                Q0p = parts["r_xx"] + 2.0 * parts["r_xu"] * u1k + parts["r_uu"] * u1k**2
                dvdx = ((P0p + parts["s_u"] * u2k) * Q - P * (Q0p + parts["r_u"] * u2k)) / Q**2
            else:
                dvdx = np.gradient(v, xk, edge_order=2)
            dvdr = dvdx / Q
        finite = np.isfinite(r) & np.isfinite(v) & np.isfinite(dvdr)
        kept += int(finite.sum())
        r_parts.append(r[finite])
        v_parts.append(v[finite])
        g_parts.append(dvdr[finite])
        if return_per_trajectory and int(finite.sum()) >= 8:
            per_traj.append((r[finite], v[finite], dvdr[finite]))
    if not r_parts:
        return {"status": "infeasible", "reason": "no finite order-2 reduced samples"}
    survival = kept / max(total, 1)
    if survival < float(min_survival_fraction):
        return {
            "status": "infeasible",
            "reason": f"chart domain keeps only {survival:.1%} of samples",
        }
    out = {
        "status": "reduced",
        "r": np.concatenate(r_parts),
        "v": np.concatenate(v_parts),
        "dvdr": np.concatenate(g_parts),
        "survival_fraction": float(survival),
    }
    if return_per_trajectory:
        out["per_trajectory"] = per_traj
    return out


def _reduced_rv_terms() -> list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[Node, Node], Node]]]:
    """Dictionary for the reduced first-order law dv/dr = H(r, v)."""

    def _sd(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return num / den

    return [
        ("1", lambda r, v: np.ones_like(r), lambda ra, va: _const(1.0)),
        ("r", lambda r, v: r, lambda ra, va: ra),
        ("r^2", lambda r, v: r * r, lambda ra, va: Pow(ra, 2.0)),
        ("v", lambda r, v: v, lambda ra, va: va),
        ("v^2", lambda r, v: v * v, lambda ra, va: Pow(va, 2.0)),
        ("v^3", lambda r, v: v**3, lambda ra, va: Pow(va, 3.0)),
        ("r*v", lambda r, v: r * v, lambda ra, va: Mul(ra, va)),
        ("r*v^2", lambda r, v: r * v * v, lambda ra, va: Mul(ra, Pow(va, 2.0))),
        ("v/r", lambda r, v: _sd(v, r), lambda ra, va: Mul(va, Pow(ra, -1.0))),
        ("v/r^2", lambda r, v: _sd(v, r * r), lambda ra, va: Mul(va, Pow(ra, -2.0))),
        ("1/r", lambda r, v: _sd(np.ones_like(r), r), lambda ra, va: Pow(ra, -1.0)),
        ("1/r^2", lambda r, v: _sd(np.ones_like(r), r * r), lambda ra, va: Pow(ra, -2.0)),
    ]


def fit_reduced_first_order(
    r: np.ndarray,
    v: np.ndarray,
    dvdr: np.ndarray,
    *,
    lambdas: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
    ridge: float = 1.0e-10,
    max_iter: int = 10,
    holdout_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, Any] | None:
    """STLSQ discovery of the reduced law dv/dr = H(r, v)."""

    import torch

    from nestynet_sr.sr_core.numerics import ridge_lstsq, stlsq

    terms = _reduced_rv_terms()
    cols = np.column_stack([fn(r, v) for _n, fn, _b in terms])
    y = np.asarray(dvdr, dtype=np.float64)
    finite = np.isfinite(cols).all(axis=1) & np.isfinite(y)
    cols, y = cols[finite], y[finite]
    if cols.shape[0] < 40:
        return None
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(cols.shape[0])
    n_val = max(int(cols.shape[0] * float(holdout_fraction)), 16)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    Phi_f = torch.as_tensor(cols[fit_idx], dtype=torch.float64)
    y_f = torch.as_tensor(y[fit_idx], dtype=torch.float64)
    Phi_v = cols[val_idx]
    y_v = y[val_idx]
    y_scale = max(float(np.sqrt(np.mean(np.square(y_v)))), 1.0e-12)

    candidates: list[dict[str, Any]] = []
    for lam in lambdas:
        try:
            coeffs_t, keep_t = stlsq(Phi_f, y_f, ridge=float(ridge), lam=float(lam), max_iter=int(max_iter))
            keep = keep_t.detach().cpu().numpy().astype(bool)
            if int(keep.sum()) == 0:
                continue
            refit = ridge_lstsq(Phi_f[:, keep_t], y_f, ridge=0.0).detach().cpu().numpy()
            coeffs = np.zeros(len(terms))
            coeffs[keep] = refit
            pred_v = Phi_v @ coeffs
            val_rel = float(np.sqrt(np.mean((pred_v - y_v) ** 2)) / y_scale)
            if not math.isfinite(val_rel):
                continue
            candidates.append({
                "val_rmse_rel": val_rel,
                "lambda": float(lam),
                "selected": [
                    {"name": terms[i][0], "coeff": float(coeffs[i]), "builder": terms[i][2]}
                    for i in range(len(terms)) if keep[i]
                ],
                "human": " + ".join(
                    f"({coeffs[i]:.6g})*{terms[i][0]}" for i in range(len(terms)) if keep[i]
                ),
            })
        except Exception:
            continue
    if not candidates:
        return None
    # simplest model within a tolerance band of the best held-out error
    best_val = min(c["val_rmse_rel"] for c in candidates)
    band = max(best_val * 3.0, 1.0e-9)
    eligible = [c for c in candidates if c["val_rmse_rel"] <= band]
    eligible.sort(key=lambda c: (len(c["selected"]), c["val_rmse_rel"]))
    return eligible[0]


def pullback_order2(
    chart: CanonicalChart,
    reduced_fit: dict[str, Any],
    *,
    x_axis: int = 0,
) -> dict[str, Any]:
    """Reconstruct u_xx = (H(r, v) - A)/B in original coordinates.

    With P = s_x + s_u u_x and Q = r_x + r_u u_x, the invariant derivative is
    dv/dr = A + B u_xx where A = (P0' Q - P Q0')/Q^3 and B = (s_u Q - r_u P)/Q^3
    (primes = total x-derivatives at frozen u_xx).  Rows are the reduced terms
    divided by B, plus the drift term A/B.
    """

    x_ast = Var(int(x_axis))
    u_ast = U()
    du_ast = DU(int(x_axis))
    parts = _chart_partial_asts(chart, x_ast, u_ast)
    P = _add2(parts["s_x"], _mul2(parts["s_u"], du_ast))
    Q = _add2(parts["r_x"], _mul2(parts["r_u"], du_ast))
    if P is None or Q is None:
        raise ValueError("degenerate chart: P or Q identically zero")
    du_sq = Pow(du_ast, 2.0)
    P0p = _add2(_add2(parts["s_xx"], _mul2(_mul2(_const(2.0), parts["s_xu"]), du_ast)),
                _mul2(parts["s_uu"], du_sq))
    Q0p = _add2(_add2(parts["r_xx"], _mul2(_mul2(_const(2.0), parts["r_xu"]), du_ast)),
                _mul2(parts["r_uu"], du_sq))
    A_num = _sub2(_mul2(P0p, Q), _mul2(P, Q0p))
    B_num = _sub2(_mul2(parts["s_u"], Q), _mul2(parts["r_u"], P))
    if B_num is None:
        raise ValueError("chart has B == 0: u_xx does not enter the invariant derivative")
    r_ast = _r_ast(chart, x_ast, u_ast)
    v_ast = _mul2(P, _inv(Q))
    B_inv = _mul2(Pow(Q, 3.0), _inv(B_num)) if not _is_const_one(Q) else _inv(B_num)

    rows: list[tuple[Node, str, str]] = []
    rhs_terms: list[Node] = []
    for item in reduced_fit["selected"]:
        phi = item["builder"](r_ast, v_ast)
        row = _mul2(phi, B_inv)
        rows.append((row, "gs_de_reduction", f"reduction2_{item['name']}"))
        rhs_terms.append(_scaled(row, float(item["coeff"])))
    if A_num is not None:
        # dv/dr = A + B u_xx with A = A_num/Q^3, B = B_num/Q^3: the Q^3 cancels
        # in the drift contribution -A/B = -A_num/B_num.
        drift = _mul2(A_num, _inv(B_num))
        rows.append((drift, "gs_de_reduction", "reduction2_drift"))
        rhs_terms.append(_scaled(drift, -1.0))
    rhs: Node | None = None
    for t in rhs_terms:
        rhs = t if rhs is None else Add(rhs, t)
    return {
        "rhs_ast": rhs,
        "library_rows": rows,
        "chart_kind": chart.kind,
        "chart_human": chart.human,
        "reduced_equation": f"dv/dr = {reduced_fit['human']}",
        "fit_val_rmse_rel": float(reduced_fit["val_rmse_rel"]),
    }


def _dedupe_rows_numeric(
    rows: list[tuple[Any, str, str]],
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    corr_tol: float = 0.999,
    u1_list: Sequence[np.ndarray] | None = None,
) -> list[tuple[Any, str, str]]:
    """Drop pulled-back rows that are numerically proportional to earlier ones.

    Different charts of the same symmetry algebra reconstruct the same law, so
    their rows can be near-duplicates; keeping one avoids splitting the STLSQ
    coefficient across proportional columns.  Order-2 rows contain u_x atoms,
    so measured/finite-difference u_x samples are used when available.
    """

    if len(rows) <= 1:
        return rows
    import torch as _torch

    from .prolongation import _eval_term_on_jets as _eval

    x = np.concatenate([np.asarray(t[0], dtype=np.float64) for t in trajectories])
    u = np.concatenate([np.asarray(t[1], dtype=np.float64) for t in trajectories])
    if u1_list is not None:
        u1 = np.concatenate([np.asarray(a, dtype=np.float64) for a in u1_list])
    else:
        u1 = np.concatenate([
            np.gradient(np.asarray(t[1], dtype=np.float64),
                        np.asarray(t[0], dtype=np.float64), edge_order=2)
            for t in trajectories
        ])
    xt = _torch.as_tensor(x).reshape(-1, 1)
    ut = _torch.as_tensor(u).reshape(-1, 1)
    u1t = _torch.as_tensor(u1).reshape(-1, 1)
    u2t = _torch.zeros_like(xt)
    kept: list[tuple[Any, str, str]] = []
    kept_cols: list[np.ndarray] = []
    for row in rows:
        try:
            col = np.asarray(
                _eval(row[0], x=xt, u=ut, u1=u1t, u2=u2t, x_axis=0).detach().cpu()
            ).reshape(-1)
        except Exception:
            kept.append(row)
            kept_cols.append(np.full(x.shape, np.nan))
            continue
        finite = np.isfinite(col)
        duplicate = False
        for prev in kept_cols:
            both = finite & np.isfinite(prev)
            if int(both.sum()) < 16:
                continue
            a, b = col[both], prev[both]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 0:
                continue
            if abs(float(a @ b) / denom) >= float(corr_tol):
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
            kept_cols.append(col)
    return kept


# ---------------------------------------------------------------------------
# Solvable-algebra cascade: reduce the reduced Riccati once more.
#
# A second-order ODE admitting a 2-dimensional solvable point-symmetry algebra
# integrates by TWO successive reductions.  The order-2 reduction by V1 leaves
# a first-order equation dv/dr = H(r, v); if that reduced equation itself admits
# a symmetry V2 (discovered from the reduced ensemble), a second reduction takes
# it to quadrature.  For the constant-coefficient linear family the first
# reduction by u d/du yields an AUTONOMOUS Riccati dv/dr = c0 + c1 v - v^2 whose
# equilibria are exactly the characteristic roots, so the closed-form general
# solution follows from the discriminant.
# ---------------------------------------------------------------------------

def _affine_bracket(g1: Sequence[float], g2: Sequence[float]) -> tuple[float, ...]:
    """Lie bracket [V1, V2] of two diagonal-affine generators.

    For V = (a0 + a1 x) d/dx + (b0 + b2 u) d/du the x- and u-actions are
    independent, so the bracket is the translation
    ``(a1_1 a0_2 - a1_2 a0_1) d/dx + (b2_1 b0_2 - b2_2 b0_1) d/du``.
    """

    a0_1, a1_1, _a2_1, b0_1, _b1_1, b2_1 = (float(v) for v in g1)
    a0_2, a1_2, _a2_2, b0_2, _b1_2, b2_2 = (float(v) for v in g2)
    dx = a1_1 * a0_2 - a1_2 * a0_1
    du = b2_1 * b0_2 - b2_2 * b0_1
    return (dx, 0.0, 0.0, du, 0.0, 0.0)


def _bracket_in_span(bracket: Sequence[float], g1: Sequence[float], g2: Sequence[float], *, tol: float = 1e-6) -> bool:
    """Is [V1,V2] in span{V1,V2} (2D subalgebra closure => solvable here)?"""

    B = np.asarray(bracket, dtype=float)
    if float(np.max(np.abs(B))) <= tol:
        return True  # abelian: [V1,V2] = 0
    M = np.column_stack([np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)])
    coeffs, *_ = np.linalg.lstsq(M, B, rcond=None)
    return bool(float(np.linalg.norm(M @ coeffs - B)) <= tol * max(1.0, float(np.linalg.norm(B))))


def reduced_equation_symmetry(
    riccati_selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Constructive point symmetry of the reduced first-order equation dv/dr=H(r,v).

    A first-order ODE always admits point symmetries, but only some are found in
    closed form from the affine family.  We read the structure of the data-fitted
    H (the STLSQ-selected terms of the reduced Riccati):

    * autonomous (only ``{1, v, v^2, v^3}`` -- no explicit r) => translation
      V2 = d/dr; the original-level second symmetry is d/dx.  Quadrature
      ``integral dv / H(v) = r + C``.  (constant-coefficient linear family)
    * scale-invariant (only scaling-degree ``-2`` terms ``{v^2, v/r, 1/r^2}``
      under (r,v) -> (lambda r, v/lambda), with an explicit r term present) =>
      reduced-level scaling V2 = r d/dr - v d/dv; the original-level second
      symmetry is x d/dx (equidimensionality).  (Cauchy-Euler / equidimensional
      family)

    ``generator`` acts on the reduced equation; ``algebra_generator`` is the
    original-level V2 used to classify the {V1, V2} algebra.  Returns ``None``
    when no affine symmetry is constructible (the reduced equation is not
    integrable this way).
    """

    coeff = {str(item["name"]): float(item["coeff"]) for item in riccati_selected}
    names = set(coeff)
    autonomous_terms = {"1", "v", "v^2", "v^3"}
    scale_invariant_terms = {"v^2", "v/r", "1/r^2"}  # scaling degree -2

    if names <= autonomous_terms:
        return {
            "generator": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # d/dr (reduced level)
            "algebra_generator": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # d/dx (original level)
            "kind": "translation_r",
            "reason": "reduced equation is autonomous in r",
            "quadrature": "integral dv / H(v) = r + const",
        }
    if names <= scale_invariant_terms and (names & {"v/r", "1/r^2"}):
        return {
            "generator": (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),  # r d/dr - v d/dv (reduced)
            "algebra_generator": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),  # x d/dx (original)
            "kind": "scaling",
            "reason": "reduced equation is scale-invariant under (r,v)->(lam r, v/lam)",
            "quadrature": "reduce by r d/dr - v d/dv (invariant w=r v) to quadrature",
        }
    return None


def recognize_constant_coeff_linear_solution(
    riccati_selected: Sequence[Mapping[str, Any]], *, v2_tol: float = 0.05,
    disc_rel_tol: float = 1.0e-6,
) -> dict[str, Any] | None:
    """Closed-form solution of a constant-coefficient linear 2nd-order ODE.

    The reduction by ``u d/du`` (v = u_x/u) sends ``u'' + p u' + q u = 0`` to the
    autonomous Riccati ``dv/dr = -q - p v - v^2``.  Its equilibria solve
    ``v^2 + p v + q = 0`` -- the characteristic roots -- so the general solution
    follows from the discriminant.  Returns ``None`` unless the Riccati is
    autonomous with a ``-v^2`` leading term.
    """

    coeff = {str(item["name"]): float(item["coeff"]) for item in riccati_selected}
    autonomous_terms = {"1", "v", "v^2"}
    if any(name not in autonomous_terms for name in coeff):
        return None  # r-dependent -> not a constant-coefficient reduction
    c2 = coeff.get("v^2", 0.0)
    if abs(c2 + 1.0) > float(v2_tol):
        return None  # not the u_x/u reduction of a linear equation
    c1 = coeff.get("v", 0.0)
    c0 = coeff.get("1", 0.0)
    p = -c1
    q = -c0
    disc = p * p - 4.0 * q
    tol = float(disc_rel_tol) * (p * p + 4.0 * abs(q) + 1.0)  # scale-aware double-root test
    out: dict[str, Any] = {
        "equation": f"u'' + ({p:.6g}) u' + ({q:.6g}) u = 0",
        "p": float(p), "q": float(q), "discriminant": float(disc),
    }
    if disc > tol:
        rp = (-p + math.sqrt(disc)) / 2.0
        rm = (-p - math.sqrt(disc)) / 2.0
        out["regime"] = "two_real_roots"
        out["roots"] = [float(rp), float(rm)]
        out["general_solution"] = f"u(x) = A*exp({rp:.6g}*x) + B*exp({rm:.6g}*x)"
    elif disc < -tol:
        omega = math.sqrt(-disc) / 2.0
        out["regime"] = "complex_roots"
        out["roots"] = [(-p / 2.0, omega), (-p / 2.0, -omega)]
        decay = "" if abs(p) < 1e-9 else f"exp({-p/2.0:.6g}*x)*"
        out["general_solution"] = f"u(x) = {decay}(A*cos({omega:.6g}*x) + B*sin({omega:.6g}*x))"
    else:
        r0 = -p / 2.0
        out["regime"] = "double_root"
        out["roots"] = [float(r0)]
        out["general_solution"] = f"u(x) = (A + B*x)*exp({r0:.6g}*x)"
    return out


def recognize_equidimensional_solution(
    riccati_selected: Sequence[Mapping[str, Any]], *, v2_tol: float = 0.05,
    disc_rel_tol: float = 1.0e-6,
) -> dict[str, Any] | None:
    """Closed-form solution of a Cauchy-Euler (equidimensional) 2nd-order ODE.

    The reduction by ``u d/du`` (v = u_x/u) sends ``x^2 u'' + a x u' + b u = 0``
    to the scale-invariant Riccati ``dv/dr = -v^2 - (a/r) v - (b/r^2)``.  Its
    scale-invariant equilibria ``v = m/r`` solve the indicial equation
    ``m^2 + (a-1) m + b = 0`` -- so the power-law general solution follows from
    the discriminant, mirroring the characteristic-root case with ``x^m`` in
    place of ``exp(m x)`` and ``ln x`` in place of ``x``.  Returns ``None``
    unless the Riccati is scale-invariant with a ``-v^2`` leading term.
    """

    coeff = {str(item["name"]): float(item["coeff"]) for item in riccati_selected}
    scale_invariant_terms = {"v^2", "v/r", "1/r^2"}
    if any(name not in scale_invariant_terms for name in coeff):
        return None
    if not (set(coeff) & {"v/r", "1/r^2"}):
        return None  # purely v^2 is autonomous, handled elsewhere
    c2 = coeff.get("v^2", 0.0)
    if abs(c2 + 1.0) > float(v2_tol):
        return None  # not the u_x/u reduction of a linear equidimensional equation
    a = -coeff.get("v/r", 0.0)
    b = -coeff.get("1/r^2", 0.0)
    disc = (a - 1.0) ** 2 - 4.0 * b
    tol = float(disc_rel_tol) * ((a - 1.0) ** 2 + 4.0 * abs(b) + 1.0)  # scale-aware
    out: dict[str, Any] = {
        "equation": f"x^2 u'' + ({a:.6g}) x u' + ({b:.6g}) u = 0",
        "a": float(a), "b": float(b), "indicial_discriminant": float(disc),
    }
    if disc > tol:
        mp = (-(a - 1.0) + math.sqrt(disc)) / 2.0
        mm = (-(a - 1.0) - math.sqrt(disc)) / 2.0
        out["regime"] = "two_real_exponents"
        out["exponents"] = [float(mp), float(mm)]
        out["general_solution"] = f"u(x) = A*x^({mp:.6g}) + B*x^({mm:.6g})"
    elif disc < -tol:
        mu = -(a - 1.0) / 2.0
        nu = math.sqrt(-disc) / 2.0
        out["regime"] = "complex_exponents"
        out["exponents"] = [(mu, nu), (mu, -nu)]
        pref = "" if abs(mu) < 1e-9 else f"x^({mu:.6g})*"
        out["general_solution"] = f"u(x) = {pref}(A*cos({nu:.6g}*ln x) + B*sin({nu:.6g}*ln x))"
    else:
        m0 = -(a - 1.0) / 2.0
        out["regime"] = "double_exponent"
        out["exponents"] = [float(m0)]
        out["general_solution"] = f"u(x) = x^({m0:.6g})*(A + B*ln x)"
    return out


def solvable_cascade_reduction(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    u1_list: Sequence[np.ndarray] | None = None,
    u2_list: Sequence[np.ndarray] | None = None,
    x_axis: int = 0,
    support_rel_tol: float = 5.0e-3,
    fit_accept_rel_rmse: float = 2.0e-2,
    seed: int = 0,
) -> dict[str, Any]:
    """Two data-driven reductions of a 2nd-order ODE by a solvable algebra.

    Discover V1 on the trajectories, reduce (order 2) to a first-order equation
    dv/dr = H(r, v), then discover V2 on the *reduced* (r, v) ensemble.  When V2
    exists the equation is integrable by a 2-dimensional solvable point-symmetry
    algebra; the reduced Riccati's structure gives the quadrature (and, for the
    constant-coefficient linear family, the closed-form general solution).

    Returns ``cascade_fired=True`` with the two generators, the algebra
    classification, the reduced Riccati and its closed form on success; otherwise
    the best single reduction and why the cascade declined.
    """

    found1 = discover_ensemble_generators(trajectories, support_rel_tol=float(support_rel_tol))
    # prefer u d/du (pure output scaling) as V1: it yields the canonical Riccati
    found1 = sorted(
        found1,
        key=lambda g: (
            0 if _ray_key(g["coefficients"]) == _ray_key((0, 0, 0, 0, 0, 1)) else 1,
            float(g.get("median_best_rel_rms", 1.0)),
        ),
    )
    report: dict[str, Any] = {
        "status": "ok",
        "n_level1_generators": len(found1),
        "level1_generators": [tuple(float(v) for v in g["coefficients"]) for g in found1],
        "cascade_fired": False,
        "attempts": [],
    }
    for g1 in found1:
        coeffs1 = tuple(float(v) for v in g1["coefficients"])
        try:
            chart1 = compile_canonical_chart(coeffs1, x_axis=int(x_axis))
        except (NotImplementedError, ValueError):
            continue
        reduced = reduce_trajectories_order2(
            trajectories, chart1, u1_list=u1_list, u2_list=u2_list, return_per_trajectory=True
        )
        if reduced.get("status") != "reduced":
            continue
        riccati = fit_reduced_first_order(reduced["r"], reduced["v"], reduced["dvdr"], seed=int(seed))
        if riccati is None or float(riccati["val_rmse_rel"]) > float(fit_accept_rel_rmse):
            continue
        per_traj = reduced.get("per_trajectory") or []
        attempt: dict[str, Any] = {
            "V1": coeffs1,
            "V1_chart": chart1.kind,
            "reduced_equation": f"dv/dr = {riccati['human']}",
            "reduced_val_rmse_rel": float(riccati["val_rmse_rel"]),
        }
        # V2 discovery: read the constructive symmetry of the data-fitted reduced
        # equation (autonomy => d/dr; scale-invariance => x d/dx).  This is robust
        # to a degenerate reduced ensemble (amplitude-varied trajectories collapse
        # to one (r,v) curve).
        sym2 = reduced_equation_symmetry(riccati["selected"])
        if sym2 is None:
            attempt["cascade_fired"] = False
            attempt["reason"] = (
                "reduced equation retains explicit r-dependence with no scale "
                "invariance: no constructible affine second symmetry (not "
                "reducible to quadrature by a 2D solvable point-symmetry algebra)"
            )
            report["attempts"].append(attempt)
            continue
        g2_reduced = tuple(float(v) for v in sym2["generator"])  # acts on dv/dr=H
        g2_original = tuple(float(v) for v in sym2["algebra_generator"])  # original V2
        # optional data confirmation: flow-test the reduced-level V2 on a
        # non-degenerate reduced ensemble (distinct (r,v) curves)
        rv_trajs = [(r, v) for (r, v, _g) in per_traj]
        data_confirmed = None
        if len(rv_trajs) >= 2:
            support = generator_ensemble_support(rv_trajs, g2_reduced, support_rel_tol=float(support_rel_tol))
            if support.get("status") == "tested":
                data_confirmed = bool(support.get("supported"))
        bracket = _affine_bracket(coeffs1, g2_original)
        if sym2["kind"] == "scaling":
            closed = recognize_equidimensional_solution(riccati["selected"])
        else:
            closed = recognize_constant_coeff_linear_solution(riccati["selected"])
        attempt.update(
            cascade_fired=True,
            V2=g2_original,
            V2_reduced_level=g2_reduced,
            V2_kind=sym2["kind"],
            V2_reason=sym2["reason"],
            V2_data_confirmed=data_confirmed,
            quadrature=sym2["quadrature"],
            algebra_bracket=bracket,
            algebra_is_solvable=_bracket_in_span(bracket, coeffs1, g2_original),
            algebra_is_abelian=bool(float(np.max(np.abs(bracket))) <= 1e-6),
            integrable_by_solvable_algebra=True,
            closed_form=closed,
        )
        report["attempts"].append(attempt)
        report.update(cascade_fired=True, best=attempt)
        return report
    return report


__all__ = [
    "CanonicalChart",
    "UnivariateFit",
    "chart_partials",
    "compile_canonical_chart",
    "fit_reduced_first_order",
    "pullback_order2",
    "recognize_constant_coeff_linear_solution",
    "reduce_trajectories_order2",
    "discover_ensemble_generators",
    "fit_univariate_families",
    "pullback_rhs_ast",
    "recognize_equidimensional_solution",
    "reduce_trajectories",
    "reduced_equation_symmetry",
    "select_univariate_fit",
    "solvable_cascade_reduction",
    "symmetry_reduction_proposals",
]
