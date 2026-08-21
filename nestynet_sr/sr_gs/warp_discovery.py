# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Discovered charts: learn the per-axis warp that linearizes a hidden symmetry.

STANDALONE EXPERIMENT — not wired into the Stage-A pipeline.  The GS charts in
:mod:`charts` are three *hand-picked* per-axis warps ``u_i = phi_i(x_i)`` fed as
reweighted gradients to the affine determining operator (identity ``phi'=1``,
log ``phi'=1/x``, reciprocal ``phi'=-1/x^2``).  This module asks the inverse
question: given only sampled ``(f, grad f, Hess f)``, can we *discover* the warp
that makes an otherwise-nonaffine symmetry affine, so the next recursive GS
detector fires?

The signature of "affine after a per-axis warp" — i.e. ``f = g(sum_i phi_i(x_i))``
(generalized-additive; radial ``phi=x^2`` and multiplicative ``phi=log`` are
special cases) — is a **pair-independent normalized off-diagonal Hessian**::

    R_ij(x) := d^2f/dx_i dx_j / (d_i f * d_j f)   ==   g''/g'^2   (same for all i<j).

When that holds, the gradient direction *is* the warp-derivative vector, so
``d_i f = g'(S) * phi_i'(x_i)`` and the log-gradient ratio between two axes is
additively separable::

    log|d_i f| - log|d_r f| = b_i(x_i) - b_r(x_r),   b_i := log|phi_i'|,

which we fit per axis.  For the whole power family (identity/log/reciprocal/
square/power) ``b_i`` is *linear in* ``log|x_i|`` with slope ``a_i - 1`` (warp
exponent ``a_i``; ``a_i -> 0`` is the log limit), so one linear regression snaps
all of them at once.  Non-power axes (e.g. periodic) fall out as ``empirical``.

The recovered coordinate ``z = sum_i s_i * phi_i(x_i)`` is *validated*
pointwise: ``q_i := d_i f / phi_i'(x_i)`` must agree across axes (it equals
``g'(z)``), a strong check that the warp is correct rather than merely fitted.

Two honest boundaries this module is built to respect:

* **Scope.** This finds *per-axis* rectifiers, hence the generalized-additive /
  multiplicative / radial class.  Genuinely oblique or angular symmetries
  (rotations that are not a warped sum, pure-angle dependence) are the affine
  operator's own job and are not per-axis warps.
* **Triviality guard.** ``z = f`` is always a coordinate, so a warp only counts
  when it snaps to a low-complexity dictionary form; an all-identity snap means
  no warp was needed (``warp_is_trivial``).  Empirical (unsnapped) warps are
  reported but flagged, never asserted as discoveries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_EPS = 1.0e-12
# A warp exponent is accepted as "dictionary" only when the power-family linear
# fit is this tight (relative), else the axis is reported as empirical.
_DEFAULT_POWER_FIT_TOL = 5.0e-2
# Pair-consistency below this (robust, relative) certifies warped-additivity.
_DEFAULT_PAIR_TOL = 5.0e-2
# Per-pair deviation above this marks an axis pair as still interacting.
_DEFAULT_INTERACTION_TOL = 1.0e-1


@dataclass(frozen=True)
class AxisWarp:
    """A recovered per-axis warp ``u_i = phi_i(x_i)``."""

    axis: int
    kind: str  # "identity" | "log" | "reciprocal" | "square" | "power" | "empirical"
    exponent: float | None  # warp power a_i (a_i == 0 -> log); None for empirical
    relative_sign: float  # sign of phi_i' relative to the reference axis
    power_fit_residual: float
    deriv_fn: Callable[[np.ndarray], np.ndarray] | None = None  # phi_i'(x_i)
    warp_fn: Callable[[np.ndarray], np.ndarray] | None = None  # phi_i(x_i)

    def human(self) -> str:
        s = "-" if self.relative_sign < 0 else "+"
        if self.kind == "identity":
            return f"{s}x{self.axis}"
        if self.kind == "log":
            return f"{s}log(x{self.axis})"
        if self.kind == "reciprocal":
            return f"{s}1/x{self.axis}"
        if self.kind == "square":
            return f"{s}x{self.axis}^2"
        if self.kind == "power":
            return f"{s}x{self.axis}^{self.exponent:.2f}"
        return f"{s}phi(x{self.axis})[empirical]"


@dataclass(frozen=True)
class WarpCertificate:
    """Outcome of warp discovery on one coordinate set."""

    n_vars: int
    pair_consistency: float  # robust across-pair spread of R_ij (0 == additive)
    is_separable_after_warp: bool
    pair_residuals: dict[tuple[int, int], float]
    interacting_pairs: list[tuple[int, int]]
    warps: list[AxisWarp] | None
    warp_is_trivial: bool  # all recovered warps are identity (no warp needed)
    warp_validation_residual: float | None  # spread of q_i = d_i f / phi_i'
    coordinate_human: str | None
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Certificate: is f generalized-additive after some per-axis warp?
# ---------------------------------------------------------------------------

def normalized_hessian_ratio_map(
    grad: np.ndarray, hess: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    """``R_ij(x) = d^2f/dx_i dx_j / (d_i f * d_j f)`` for every pair ``i<j``."""

    g = np.asarray(grad, dtype=float)
    h = np.asarray(hess, dtype=float)
    n = g.shape[1]
    out: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            denom = g[:, i] * g[:, j]
            safe = np.where(np.abs(denom) < _EPS, np.nan, denom)
            out[(i, j)] = h[:, i, j] / safe
    return out


def certify_warped_additivity(
    x_vals: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    *,
    pair_tol: float = _DEFAULT_PAIR_TOL,
    interaction_tol: float = _DEFAULT_INTERACTION_TOL,
) -> WarpCertificate:
    """Test pair-independence of ``R_ij`` and localize residual coupling.

    Does not recover warps; :func:`discover_warp` calls this first, then
    recovers warps only when the certificate passes.
    """

    g = np.asarray(grad, dtype=float)
    n = g.shape[1]
    if n < 2:
        return WarpCertificate(
            n_vars=n, pair_consistency=float("inf"), is_separable_after_warp=False,
            pair_residuals={}, interacting_pairs=[], warps=None, warp_is_trivial=False,
            warp_validation_residual=None, coordinate_human=None, reason="univariate",
        )

    ratio_map = normalized_hessian_ratio_map(g, hess)
    pairs = list(ratio_map.keys())
    P = np.stack([ratio_map[p] for p in pairs], axis=1)  # (N, npairs)

    # Mask samples where any gradient component is tiny (ratios blow up) or any
    # pair ratio is non-finite.
    grad_scale = np.median(np.abs(g), axis=0) + _EPS
    ok = np.all(np.abs(g) > 1.0e-6 * grad_scale, axis=1) & np.all(np.isfinite(P), axis=1)
    P = P[ok]
    if P.shape[0] < max(8, 2 * n):
        return WarpCertificate(
            n_vars=n, pair_consistency=float("inf"), is_separable_after_warp=False,
            pair_residuals={}, interacting_pairs=[], warps=None, warp_is_trivial=False,
            warp_validation_residual=None, coordinate_human=None,
            reason="too_few_usable_samples",
        )

    # Per-sample consensus R (median over pairs) and robust relative deviation.
    consensus = np.median(P, axis=1, keepdims=True)  # (N,1)
    scale = np.median(np.abs(P), axis=1, keepdims=True) + _EPS
    per_pair = np.median(np.abs(P - consensus) / scale, axis=0)  # (npairs,)
    pair_residuals = {p: float(r) for p, r in zip(pairs, per_pair)}
    pair_consistency = float(np.median(per_pair))
    interacting = sorted(p for p, r in pair_residuals.items() if r > interaction_tol)
    is_sep = pair_consistency <= pair_tol and not interacting

    return WarpCertificate(
        n_vars=n, pair_consistency=pair_consistency, is_separable_after_warp=is_sep,
        pair_residuals=pair_residuals, interacting_pairs=interacting, warps=None,
        warp_is_trivial=False, warp_validation_residual=None, coordinate_human=None,
        reason="certified_warped_additive" if is_sep else "not_pair_consistent",
        evidence={"usable_fraction": float(np.mean(ok)), "n_pairs": len(pairs)},
    )


# ---------------------------------------------------------------------------
# Warp recovery: read phi_i' off the log-gradient ratios and snap the power family
# ---------------------------------------------------------------------------

def _snap_exponent(slope: float) -> tuple[float, str]:
    """Warp exponent ``a = slope + 1``; snap to nearest half-integer, name it."""

    a = slope + 1.0
    a_snapped = round(a * 2.0) / 2.0
    if abs(a_snapped - 0.0) < 1.0e-9:
        return 0.0, "log"
    if abs(a_snapped - 1.0) < 1.0e-9:
        return 1.0, "identity"
    if abs(a_snapped - 2.0) < 1.0e-9:
        return 2.0, "square"
    if abs(a_snapped + 1.0) < 1.0e-9:
        return -1.0, "reciprocal"
    return float(a_snapped), "power"


def _warp_callables(a: float) -> tuple[Callable, Callable]:
    """``(phi', phi)`` for warp exponent ``a`` (``a == 0`` is the log limit)."""

    if abs(a) < 1.0e-9:
        return (lambda x: 1.0 / x, lambda x: np.log(np.abs(x)))
    return (lambda x: a * np.sign(x) * np.abs(x) ** (a - 1.0),
            lambda x: np.sign(x) * np.abs(x) ** a if a % 1 else x ** int(a))


def recover_axis_warps(
    x_vals: np.ndarray,
    grad: np.ndarray,
    *,
    power_fit_tol: float = _DEFAULT_POWER_FIT_TOL,
) -> list[AxisWarp]:
    """Recover per-axis warps from pairwise log-gradient differences.

    ``log|d_i f| - log|d_j f| = b_i(x_i) - b_j(x_j)`` cancels the shared
    ``g'(S)`` factor with *no* per-sample nuisance, so stacking every pair
    identifies all axis slopes jointly (the reference axis is recovered too,
    unlike a ratio-to-reference).  The power-family ansatz
    ``b_i = beta0_i + beta1_i log|x_i|`` snaps identity/log/reciprocal/square/
    power in one solve; ``beta0_0`` is pinned to fix the global-constant gauge.
    """

    x = np.asarray(x_vals, dtype=float)
    g = np.asarray(grad, dtype=float)
    n = x.shape[1]
    logx = np.log(np.abs(x) + _EPS)
    logg = np.log(np.abs(g) + _EPS)
    grad_scale = np.median(np.abs(g), axis=0) + _EPS
    ok = np.all(np.abs(g) > 1.0e-6 * grad_scale, axis=1)
    xo, lgx, lgg = x[ok], logx[ok], logg[ok]
    m = xo.shape[0]

    # Unknowns: beta1_0..beta1_{n-1} (slopes), then beta0_1..beta0_{n-1}.
    n_unk = n + (n - 1)
    rows_A: list[np.ndarray] = []
    rows_b: list[float] = []
    pair_row_index: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                row = np.zeros(n_unk)
                row[i] = lgx[k, i]
                row[j] = -lgx[k, j]
                if i > 0:
                    row[n + (i - 1)] = 1.0
                if j > 0:
                    row[n + (j - 1)] = -1.0
                rows_A.append(row)
                rows_b.append(lgg[k, i] - lgg[k, j])
                pair_row_index.append((i, j))
    A = np.asarray(rows_A)
    b = np.asarray(rows_b)
    finite = np.isfinite(b) & np.all(np.isfinite(A), axis=1)
    A, b = A[finite], b[finite]
    pair_row_index = [p for p, f in zip(pair_row_index, finite) if f]
    # Near-zero gradients (masked/trimmed below) make the log-domain fit
    # underflow subnormals in the matmuls; that is expected and handled, silence.
    with np.errstate(under="ignore", over="ignore", invalid="ignore", divide="ignore"):
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        # Robust refit: a non-power axis (e.g. sin, whose derivative crosses
        # zero in-domain) throws outlier rows that would otherwise inflate the
        # residual of the *clean* axes it shares pairs with.  Trim worst 15%.
        res0 = np.abs(b - A @ coef)
        keep = res0 <= np.quantile(res0, 0.85)
        coef, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        A, b = A[keep], b[keep]
        pair_row_index = [p for p, k in zip(pair_row_index, keep) if k]
        slopes = coef[:n]
        beta0 = np.concatenate([[0.0], coef[n:]])  # beta0_0 pinned by the gauge

        # Per-axis power-fit residual, decoupled from other axes: each axis
        # reconstructs the shared term a_k = log|g'(S_k)| = log|d_i f| - b_i(x_i).
        # For power axes these agree sample-by-sample; the odd (non-power) axis
        # deviates from the consensus (median over axes), so its residual is high
        # without contaminating the clean axes it shares pairs with.
        a = lgg - slopes[None, :] * lgx - beta0[None, :]  # (m, n)
        consensus = np.median(a, axis=1, keepdims=True)

        def _rstd(v: np.ndarray) -> float:
            return 1.4826 * float(np.median(np.abs(v - np.median(v))))

        scale = max(_rstd(consensus.ravel()), 1.0e-6)
        axis_resid = np.array([_rstd(a[:, i] - consensus.ravel()) / scale for i in range(n)])

    ref = int(np.argmax(grad_scale))
    warps: list[AxisWarp] = []
    for i in range(n):
        gi, gr = g[:, i], g[:, ref]
        okr = (np.abs(gi) > _EPS) & (np.abs(gr) > _EPS)
        rel_sign = float(np.sign(np.median((gi / gr)[okr]))) or 1.0
        a, kind = _snap_exponent(float(slopes[i]))
        if float(axis_resid[i]) > power_fit_tol:
            warps.append(AxisWarp(axis=i, kind="empirical", exponent=None,
                                  relative_sign=rel_sign,
                                  power_fit_residual=float(axis_resid[i]),
                                  deriv_fn=None, warp_fn=None))
            continue
        phi_d, phi = _warp_callables(a)
        warps.append(AxisWarp(axis=i, kind=kind, exponent=a, relative_sign=rel_sign,
                              power_fit_residual=float(axis_resid[i]),
                              deriv_fn=phi_d, warp_fn=phi))
    return warps


def _validate_warp(
    x: np.ndarray, g: np.ndarray, warps: list[AxisWarp]
) -> tuple[float, np.ndarray | None]:
    """Rank-1 test of ``q_i = d_i f / phi_i'``; also recovers the covector.

    If ``f = g(sum_i c_i phi_i(x_i))`` then ``q_i(x) = c_i * g'(z)`` — the
    columns are proportional, i.e. ``Q = w(x) c^T`` is rank 1.  The residual is
    ``sigma_2 / sigma_1`` (robust to ``g'`` crossing zero, unlike a ratio), and
    the leading right singular vector recovers the coefficient covector ``c``.
    """

    cols = []
    for w in warps:
        if w.deriv_fn is None:
            return float("inf"), None
        phid = w.deriv_fn(x[:, w.axis])
        cols.append(g[:, w.axis] / np.where(np.abs(phid) < _EPS, np.nan, phid))
    Q = np.stack(cols, axis=1)
    ok = np.all(np.isfinite(Q), axis=1)
    Q = Q[ok]
    if Q.shape[0] < 4:
        return float("inf"), None
    _u, s, vt = np.linalg.svd(Q, full_matrices=False)
    resid = float(s[1] / s[0]) if s[0] > 0 else float("inf")
    c = vt[0]
    c = c / (c[int(np.argmax(np.abs(c)))] + _EPS)  # normalize largest |c_i| -> 1
    return resid, c


def _snap_covector(c: np.ndarray, max_den: int = 4) -> list[float]:
    """Snap a float covector to small rationals (display/handoff only)."""

    from fractions import Fraction

    nz = np.abs(c[np.abs(c) > 1.0e-6])
    base = float(np.min(nz)) if nz.size else 1.0
    out = []
    for v in c / base:
        f = Fraction(float(v)).limit_denominator(max_den)
        out.append(float(f))
    return out


def _render_coordinate(warps: list[AxisWarp], coeffs: list[float]) -> str:
    terms = []
    for w, c in zip(warps, coeffs):
        if abs(c) < 1.0e-6:
            continue
        cs = "" if abs(abs(c) - 1.0) < 1e-6 else f"{abs(c):g}*"
        sign = "-" if c < 0 else "+"
        body = w.human().lstrip("+-")
        terms.append(f"{sign} {cs}{body}")
    return "g( " + " ".join(terms).lstrip("+ ") + " )"


def discover_warp(
    x_vals: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    *,
    pair_tol: float = _DEFAULT_PAIR_TOL,
    power_fit_tol: float = _DEFAULT_POWER_FIT_TOL,
    validation_tol: float = _DEFAULT_PAIR_TOL,
) -> WarpCertificate:
    """Certify warped-additivity, then recover + validate the per-axis warp."""

    cert = certify_warped_additivity(x_vals, grad, hess, pair_tol=pair_tol)
    if not cert.is_separable_after_warp:
        return cert

    x = np.asarray(x_vals, dtype=float)
    g = np.asarray(grad, dtype=float)
    warps = recover_axis_warps(x, g, power_fit_tol=power_fit_tol)
    val, covector = _validate_warp(x, g, warps)
    trivial = all(w.kind == "identity" for w in warps)
    dictionary = all(w.kind != "empirical" for w in warps)
    snapped = _snap_covector(covector) if covector is not None else None
    human = _render_coordinate(warps, snapped) if (dictionary and snapped is not None) else None
    reason = (
        "warp_trivial_identity" if (trivial and val <= validation_tol)
        else "warp_recovered" if (dictionary and val <= validation_tol)
        else "warp_empirical_or_unvalidated"
    )
    return WarpCertificate(
        n_vars=cert.n_vars, pair_consistency=cert.pair_consistency,
        is_separable_after_warp=True, pair_residuals=cert.pair_residuals,
        interacting_pairs=cert.interacting_pairs, warps=warps,
        warp_is_trivial=trivial, warp_validation_residual=val,
        coordinate_human=human, reason=reason,
        evidence={**cert.evidence, "dictionary": dictionary,
                  "validated": val <= validation_tol, "covector": snapped},
    )


def warp_coordinate_ast(
    warps: list[AxisWarp], covector: list[float], cols: Any
) -> tuple[Any, tuple[int, ...], tuple[float, ...]] | None:
    """Build ``z = sum_i c_i * phi_i(x_{cols[i]})`` from recovered warps.

    Returns ``(ast, support, covector)`` for a fully-dictionary recovery, or
    ``None`` if any axis is empirical or fewer than two axes participate.  The
    per-axis warp ``phi_i`` is ``log`` (LogNode), identity (Var), or a power
    (PowNode; covers square/reciprocal/general power).
    """

    from nestynet_sr.sr_core.bridges import (
        AddNode,
        ConstNode,
        LogNode,
        MulNode,
        PowNode,
        Var,
    )

    cols_t = [int(c) for c in cols]
    terms: list[Any] = []
    support: list[int] = []
    for w, c in zip(warps, covector):
        if w.kind == "empirical" or w.deriv_fn is None:
            return None
        c = float(c)
        if abs(c) < 1.0e-6:
            continue
        vi = Var(cols_t[w.axis])
        if w.kind == "log":
            phi = LogNode(vi)
        elif w.kind == "identity":
            phi = vi
        else:  # square / reciprocal / power
            phi = PowNode(vi, float(w.exponent))
        if abs(abs(c) - 1.0) < 1.0e-9:
            term = phi if c > 0 else MulNode(ConstNode(-1.0), phi)
        else:
            term = MulNode(ConstNode(c), phi)
        terms.append(term)
        support.append(cols_t[w.axis])
    if len(support) < 2:
        return None
    ast = terms[0]
    for t in terms[1:]:
        ast = AddNode(ast, t)
    return ast, tuple(sorted(support)), tuple(float(c) for c in covector)


# ---------------------------------------------------------------------------
# Experiment harness (torch autograd grad/Hess + noise injection)
# ---------------------------------------------------------------------------

def numerical_grad_hess(fn: Callable, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact ``(f, grad, Hess)`` via torch autograd — for the experiment only."""

    import torch

    Xt = torch.tensor(np.asarray(X, dtype=float), requires_grad=True)
    y = fn(Xt)
    (g,) = torch.autograd.grad(y.sum(), Xt, create_graph=True)
    n = Xt.shape[1]
    rows = []
    for i in range(n):
        (row,) = torch.autograd.grad(g[:, i].sum(), Xt, create_graph=True)
        rows.append(row)
    H = torch.stack(rows, dim=1)
    return y.detach().numpy(), g.detach().numpy(), H.detach().numpy()


def _inject_noise(g: np.ndarray, h: np.ndarray, rel: float, seed: int):
    rng = np.random.default_rng(seed)
    gn = g + rel * np.std(g) * rng.standard_normal(g.shape)
    hn = h + rel * np.std(h) * rng.standard_normal(h.shape)
    return gn, hn


def run_experiment() -> None:
    """Envelope battery: warp discovery on generalized-additive carriers + controls."""

    import torch

    def carriers():
        return {
            "additive g(sin x0 + x1^3 + log x2)":
                lambda X: torch.sin(torch.sin(X[:, 0]) + X[:, 1] ** 3 + torch.log(X[:, 2]))
                + 0.2 * (torch.sin(X[:, 0]) + X[:, 1] ** 3 + torch.log(X[:, 2])) ** 2,
            "radial g(x0^2 + x1^2 + x2^2)":
                lambda X: torch.sin(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2)
                + 0.3 * (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2),
            "multiplicative g(x0*x1*x2)":
                lambda X: torch.sin(X[:, 0] * X[:, 1] * X[:, 2]) + 0.3 * (X[:, 0] * X[:, 1] * X[:, 2]),
            "reciprocal g(1/x0 + 1/x1 + 1/x2)":
                lambda X: torch.sin(1 / X[:, 0] + 1 / X[:, 1] + 1 / X[:, 2])
                + 0.2 * (1 / X[:, 0] + 1 / X[:, 1] + 1 / X[:, 2]) ** 2,
            "linear g(x0 - 2 x1 + x2) [trivial warp]":
                lambda X: torch.sin(X[:, 0] - 2 * X[:, 1] + X[:, 2]),
            "CONTROL non-sep g(x0*x1 + x0 + x2)":
                lambda X: torch.sin(X[:, 0] * X[:, 1] + X[:, 0] + X[:, 2])
                + 0.2 * (X[:, 0] * X[:, 1] + X[:, 0] + X[:, 2]) ** 2,
        }

    rng = np.random.default_rng(0)
    X = rng.uniform(0.6, 1.8, size=(500, 3))
    for noise in (0.0, 1.0e-3, 1.0e-2):
        print(f"\n=== gradient/Hessian relative noise = {noise:g} ===")
        for name, fn in carriers().items():
            _f, g, h = numerical_grad_hess(fn, X)
            if noise:
                g, h = _inject_noise(g, h, noise, seed=1)
            cert = discover_warp(X, g, h)
            tag = cert.coordinate_human or f"[{cert.reason}]"
            extra = ""
            if not cert.is_separable_after_warp and cert.interacting_pairs:
                extra = f"  interacting={cert.interacting_pairs}"
            val = "" if cert.warp_validation_residual is None else f"  val={cert.warp_validation_residual:.1e}"
            print(f"  {name:44s} R-consist={cert.pair_consistency:.3f}  {tag}{val}{extra}")


if __name__ == "__main__":
    run_experiment()
