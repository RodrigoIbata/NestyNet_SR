# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Shared structural-recovery verdict for the Paper III benchmark audits.

One definition of "solved" is used by both the noisy full-pipeline audit
(Table ``full-noisy``) and the factorized-search oracle audit
(Table ``aif-benchmark``): a recovered expression counts as a structural
recovery when it is algebraically identical to the target up to the VALUES
of its fitted constants.  That is decided operationally, with no
symbolic-equivalence judgment anywhere in the path:

1. take the recovered expression,
2. refit every free numeric constant on the canonical NOISELESS data for
   the same problem, including constants inside transcendentals (a
   frequency or phase within a sine, an exponent within a power), which a
   coefficient-only refit leaves untouched,
3. re-snap the coefficients with the pipeline's own arsenal
   (``equation_polisher.polish_expression`` at its shipped defaults),
4. accept when the polished expression predicts that data at the
   noiseless-fit floor (default relative RMSE 1e-10; the worst noiseless
   exact fit in the reference campaign is pb116 at 2.4e-13).

Keeping this in one module is the point: the two tables cannot drift apart
into "solved" meaning two different things.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np

DEFAULT_TOL = 1.0e-10

# Constants that are structure, not calibration: an exponent fixes the shape
# of a power, so freeing it would let a wrong structure curve-fit its way to
# the floor.  Everything else is a fitted constant.
_EXPONENT_CONTEXT = re.compile(r"\*\*\s*$")


def load_problem_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a benchmark CSV as ``(X, y, variable_names)``."""
    with Path(path).open() as fh:
        header = fh.readline().strip().split(",")
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64))
    y_idx = header.index("y")
    x_cols = [i for i, name in enumerate(header) if name != "y"]
    return data[:, x_cols], data[:, y_idx], [header[i] for i in x_cols]


def find_noiseless_csv(data_dir: Path, problem_id: str) -> Path | None:
    """Locate ``pb<id>_*_data.csv`` for a zero-padded problem id."""
    hits = sorted(Path(data_dir).glob(f"pb{problem_id}_*_data.csv"))
    return hits[0] if hits else None


def _free_constants(expr: str) -> tuple[str, list[float]]:
    """Rewrite numeric literals as ``c0, c1, ...`` placeholders.

    Exponents are left alone: they are structural.  Returns the templated
    expression and the starting values read from the expression itself.
    """
    import sympy as sp

    e = sp.sympify(expr)
    protected: set[int] = set()
    for pw in e.atoms(sp.Pow):
        for atom in sp.sympify(pw.exp).atoms(sp.Number):
            protected.add(id(atom))

    values: list[float] = []
    syms: list[sp.Symbol] = []

    def walk(node):
        if isinstance(node, sp.Number) and not isinstance(node, sp.NumberSymbol):
            if id(node) in protected:
                return node
            sym = sp.Symbol(f"__c{len(values)}")
            syms.append(sym)
            values.append(float(node))
            return sym
        if not node.args:
            return node
        if isinstance(node, sp.Pow):
            return sp.Pow(walk(node.base), node.exp, evaluate=False)
        return node.func(*[walk(a) for a in node.args], evaluate=False)

    return sp.sstr(walk(e)), values


def _nonlinear_refit(
    expr: str, X: np.ndarray, y: np.ndarray, variable_names: list[str]
) -> tuple[str, dict]:
    """Least-squares refit of every free constant, transcendentals included."""
    import sympy as sp
    from scipy.optimize import least_squares

    info: dict = {}
    try:
        template, start = _free_constants(expr)
        if not start or len(start) > 24:
            info["nonlinear_refit"] = "skipped" if not start else "too_many_constants"
            return expr, info
        xs = [sp.Symbol(n) for n in variable_names]
        cs = [sp.Symbol(f"__c{i}") for i in range(len(start))]
        f = sp.lambdify(xs + cs, sp.sympify(template), "numpy")
        n = min(4096, X.shape[0])
        Xf, yf = X[:n], y[:n]
        scale = float(np.sqrt(np.mean(np.square(yf)))) + 1e-300

        def resid(c):
            with np.errstate(all="ignore"):
                pred = f(*[Xf[:, i] for i in range(len(xs))], *c)
            pred = np.broadcast_to(np.asarray(pred, dtype=float), (n,))
            out = (pred - yf) / scale
            return np.where(np.isfinite(out), out, 1e6)

        before = float(np.sqrt(np.mean(np.square(resid(np.asarray(start, dtype=float))))))
        sol = least_squares(resid, np.asarray(start, dtype=float),
                            method="lm", max_nfev=4000)
        after = float(np.sqrt(np.mean(np.square(resid(sol.x)))))
        info.update({"nonlinear_refit": "applied", "n_constants": len(start),
                     "rel_before": before, "rel_after": after})
        if not (after < before):
            return expr, info
        refit = sp.sympify(template).subs(
            {c: sp.Float(v, 17) for c, v in zip(cs, sol.x)}
        )
        return sp.sstr(refit), info
    except Exception as exc:
        info["nonlinear_refit_error"] = str(exc)[:120]
        return expr, info


def structural_verdict(
    expr: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    variable_names: list[str] | None = None,
    tol: float = DEFAULT_TOL,
) -> tuple[bool, dict]:
    """Refit, snap, and score ``expr`` against noiseless ``(X, y)``.

    Returns ``(is_structural, detail)``.  The split is deterministic (even
    rows fit, odd rows score) so both halves cover the sampled domain.
    """
    from nestynet_sr.equation_polisher import polish_expression

    tr, va = slice(0, None, 2), slice(1, None, 2)
    started = time.time()
    names = list(variable_names or [f"x{i}" for i in range(X.shape[1])])
    expr, refit_info = _nonlinear_refit(expr, X[tr], y[tr], names)
    result = polish_expression(
        expr,
        X[tr], y[tr], X[va], y[va],
        variable_names=variable_names,
    )
    y_rms = float(np.sqrt(np.mean(np.square(y[va])))) + 1.0e-300

    best_rel = math.inf
    best: dict = {}
    candidates = list(result.all_candidates)
    if result.recommended is not None:
        candidates.append(result.recommended)
    for cand in candidates:
        if not (cand.val_mse >= 0.0 and math.isfinite(cand.val_mse)):
            continue
        if cand.frac_valid < 1.0:
            continue
        rel = math.sqrt(cand.val_mse) / y_rms
        if rel < best_rel:
            best_rel = rel
            best = {
                "polished_expr": cand.expr,
                "polish_label": cand.label,
                "n_snapped_consts": cand.n_snapped_consts,
            }
    detail = {
        **refit_info,
        "refit_expr": expr[:400],
        "polish_rel_rmse": None if math.isinf(best_rel) else best_rel,
        "polish_seconds": round(time.time() - started, 1),
        "tol": tol,
        **best,
    }
    return (best_rel <= tol), detail
