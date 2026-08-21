# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Pretty-printing utilities for Stage B models.

Converts trained AST-based models into human-readable mathematical expressions.
"""

import math
import os
import signal
import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from nestynet_sr.sr_core.coefficient_metadata import (
    CoefficientMetadataError,
    coefficient_symbol_values,
    named_coefficient_symbol,
    normalize_coefficient_metadata,
)
from nestynet_sr.sr_core.sympy_units import check_sympy_units

# ---- Timeout mechanism for SymPy operations ----


class SympyTimeoutError(Exception):
    """Raised when a SymPy operation times out."""

    pass


# Global budget tracker for SymPy operations within a single simplification pass
_sympy_budget_remaining = 60.0  # seconds
_sympy_budget_start = None


def reset_sympy_budget(total_seconds: float = 60.0):
    """Reset the SymPy time budget for a new simplification pass."""
    global _sympy_budget_remaining, _sympy_budget_start
    _sympy_budget_remaining = total_seconds
    _sympy_budget_start = None


def get_sympy_budget_remaining() -> float:
    """Get remaining SymPy budget in seconds."""
    global _sympy_budget_remaining
    return max(0.0, _sympy_budget_remaining)


@contextmanager
def sympy_timeout(operation_name: str = "sympy", max_seconds: float = None):
    """
    Context manager that enforces a timeout on SymPy operations.

    Uses signal.SIGALRM on Unix systems (including macOS).
    On Windows, this is a no-op (SymPy operations may still hang).

    The timeout is drawn from a shared budget - once the budget is exhausted,
    subsequent operations get minimal time (1 second).

    Parameters
    ----------
    operation_name : str
        Name of the operation (for error messages)
    max_seconds : float, optional
        Maximum seconds for this operation. If None, uses remaining budget.
    """
    global _sympy_budget_remaining, _sympy_budget_start
    import time

    # Determine timeout for this operation
    if max_seconds is None:
        timeout = max(1.0, _sympy_budget_remaining)
    else:
        timeout = min(max_seconds, max(1.0, _sympy_budget_remaining))

    # Round up to integer for signal.alarm
    timeout_int = max(1, int(math.ceil(timeout)))

    # Check if we're on a Unix-like system
    if not hasattr(signal, "SIGALRM"):
        # Windows - no timeout support, just run the operation
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            _sympy_budget_remaining = max(0.0, _sympy_budget_remaining - elapsed)
        return

    def timeout_handler(signum, frame):
        raise SympyTimeoutError(f"SymPy {operation_name} timed out after {timeout_int}s")

    # Save old handler and set new one
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    old_alarm = signal.alarm(timeout_int)

    start = time.time()
    try:
        yield
    finally:
        # Cancel our alarm and restore previous state
        elapsed = time.time() - start
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_alarm > 0:
            # Restore previous alarm with adjusted time
            remaining = max(0, old_alarm - int(elapsed))
            if remaining > 0:
                signal.alarm(remaining)

        # Update budget
        _sympy_budget_remaining = max(0.0, _sympy_budget_remaining - elapsed)


# Terminal colors for logging
GREEN = "\033[32m"
RESET = "\033[0m"

try:
    import sympy as sp

    _HAVE_SYMPY = True
except Exception:
    sp = None
    _HAVE_SYMPY = False

from nestynet_sr.sr_search.polish_utils import (
    canonicalize_trig_phases,
    constant_code_cost as _constant_code_cost,
    final_polish_snap_targets,
    numeric_constant_snap_candidates,
    snap_numeric_constants,
)


def _prune_tiny_additive_constants(expr, *, tol: float):
    """
    Remove near-zero additive constants from an expression.

    Only operates locally within each Add node - does NOT combine
    constants from different parts of the expression tree.
    """
    if (not _HAVE_SYMPY) or tol is None:
        return expr
    try:
        # Recurse first
        if getattr(expr, "args", None):
            expr = expr.func(*[_prune_tiny_additive_constants(a, tol=tol) for a in expr.args])

        # At each Add, check if the constant term is tiny
        if getattr(expr, "is_Add", False):
            # Flatten nested Adds so numeric terms meet each other at this level
            expr = sp.Add(*expr.args)

            c, rest = expr.as_coeff_Add()
            if getattr(c, "is_number", False):
                try:
                    if abs(float(c)) <= float(tol):
                        return rest
                except Exception:
                    pass

        return expr
    except RecursionError:
        return expr
    except Exception:
        return expr


def _fmt(v: float, sig: int = 6) -> str:
    """Format floats for pretty-printing.

    We use significant digits ("g") so that both tiny and large coefficients
    stay readable. Stage C uses this printer to build a SymPy expression; when
    the printed parameter precision is too low, Stage C may incorrectly accept
    an *approximate* symbolic expression (because the baseline printer already
    introduces error).  By allowing higher precision formatting we can make the
    "pretty_print" string numerically faithful to the trained model.
    """
    try:
        vf = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(vf):
        return str(vf)
    # Avoid "-0" artifacts
    if abs(vf) == 0.0:
        vf = 0.0
    return f"{vf:.{int(sig)}g}"


def _needs_parens_as_factor(expr: str) -> bool:
    """Return True when ``expr`` needs grouping as a multiplicative factor."""
    s = str(expr)
    return (
        s.startswith("-")
        or "+" in s
        or "/" in s
        or "*" in s
        or " - " in s
    )


def _needs_parens_as_power_base(expr: str) -> bool:
    """Return True when appending another power would be ambiguous.

    Reciprocal-coordinate candidates often use inputs like ``z = x**-1``.
    Rendering a degree-2 polynomial term as ``z^2`` must produce
    ``(x**-1)^2`` rather than ``x**-1^2``.
    """
    return _needs_parens_as_factor(expr) or "^" in str(expr) or "**" in str(expr)


def _has_outer_parens(expr: str) -> bool:
    """True when one outer parenthesis pair encloses the whole expression."""
    s = str(expr).strip()
    if not (s.startswith("(") and s.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0


def _parenthesize_if_needed(expr: str, *, as_power_base: bool = False) -> str:
    expr = str(expr)
    need = _needs_parens_as_power_base(expr) if as_power_base else _needs_parens_as_factor(expr)
    if need and not _has_outer_parens(expr):
        return f"({expr})"
    return expr


# Common constants used for aggressive SymPy simplification.
if _HAVE_SYMPY:

    def _nsimplify_compat(expr, *, constants=None, rational=True, maxsteps=None, tolerance=None):
        kwargs = {}
        if constants is not None:
            kwargs["constants"] = constants
        if rational is not None:
            kwargs["rational"] = rational
        if tolerance is not None:
            kwargs["tolerance"] = tolerance

        # Try with maxsteps if supported; fall back otherwise.
        if maxsteps is not None:
            try:
                return sp.nsimplify(expr, maxsteps=maxsteps, **kwargs)
            except TypeError:
                # Older SymPy: no 'maxsteps' kwarg
                pass
        return sp.nsimplify(expr, **kwargs)

    COMMON_CONSTS = [
        sp.pi,
        sp.E,
        sp.sqrt(2),
        sp.sqrt(sp.pi),
        sp.sqrt(2 * sp.pi),
        2 * sp.pi,
        4 * sp.pi,
        # Inverses for recognizing constants like 1/√(2π) ≈ 0.3989
        1 / sp.sqrt(2),
        1 / sp.sqrt(sp.pi),
        1 / sp.sqrt(2 * sp.pi),
        1 / (2 * sp.pi),
        1 / (4 * sp.pi),
    ]
    AIF_CONSTS = [
        sp.pi,
        sp.sqrt(2),
        sp.sqrt(sp.pi),
        sp.sqrt(2 * sp.pi),
        2 * sp.pi,
        4 * sp.pi,
        sp.Rational(1, 2),
        sp.Rational(1, 3),
        sp.Rational(1, 5),
        # Inverses for recognizing constants like 1/√(2π) ≈ 0.3989
        1 / sp.sqrt(2),
        1 / sp.sqrt(sp.pi),
        1 / sp.sqrt(2 * sp.pi),
        1 / (2 * sp.pi),
        1 / (4 * sp.pi),
    ]
else:
    COMMON_CONSTS = []
    AIF_CONSTS = []
from nestynet_sr.sr_core.atoms import (
    Expm1Leaf,
    ExpPolyLeaf,
    ExpRationalPolyLeaf,
    FixedConstLeaf,
    FreeConstLeaf,
    InverseMonomialLeaf,
    PolyLogLeaf,
    LogShiftedLeaf,
    PlanckFullLeaf,
    PlanckLeaf,
    PolyLeaf,
    PowerLeaf,
    RationalPolyLeaf,
    RatioPolyLeaf,
    RExpPolyLeaf,
    RInverseMonomialLeaf,
    RPolyLogLeaf,
    RPolyLeaf,
    RRationalPolyLeaf,
    RRatioPolyLeaf,
    SinLinearLeaf,
    TanhLinearLeaf,
    VarLeaf,
)
from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    PowNode,
    SinNode,
    atom_problem_label,
    compound_input_expr,
    extra_input_var_idxs,
    format_const_value,
    get_input_exprs,
)

if _HAVE_SYMPY:

    def _nsimplify_floats(expr, *, constants=None, tolerance=1e-4):
        """
        Recursively apply nsimplify to Float coefficients in an expression.

        This helps recognize constants like 0.3989 → 1/√(2π) before
        rational simplification converts them to rationals like 3989/10000.
        """
        if isinstance(expr, sp.Float):
            # Apply nsimplify to this float
            return sp.nsimplify(expr, constants=constants, tolerance=tolerance)
        elif expr.is_Mul:
            # For products, nsimplify coefficient separately from rest
            coeff, rest = expr.as_coeff_Mul()
            if isinstance(coeff, sp.Float):
                coeff = sp.nsimplify(coeff, constants=constants, tolerance=tolerance)
            rest = _nsimplify_floats(rest, constants=constants, tolerance=tolerance)
            return coeff * rest
        elif expr.is_Add:
            # For sums, apply to each term
            return sp.Add(
                *[
                    _nsimplify_floats(arg, constants=constants, tolerance=tolerance)
                    for arg in expr.args
                ]
            )
        elif expr.is_Pow:
            # For powers, apply to base and exponent
            base = _nsimplify_floats(expr.base, constants=constants, tolerance=tolerance)
            exp = _nsimplify_floats(expr.exp, constants=constants, tolerance=tolerance)
            return sp.Pow(base, exp)
        elif hasattr(expr, "args") and expr.args:
            # For other functions, apply to arguments
            new_args = [
                _nsimplify_floats(arg, constants=constants, tolerance=tolerance)
                for arg in expr.args
            ]
            return expr.func(*new_args)
        else:
            # Leaf node (symbol, integer, etc.)
            return expr


def aggressive_simplify(expr, Nxvars: int, *, verbose: bool = True, budget_seconds: float = 60.0):
    """
    Apply a more aggressive SymPy simplification pass tailored to AIF outputs.

    Parameters
    ----------
    expr : sympy expression
        Expression to simplify
    Nxvars : int
        Number of input variables
    verbose : bool
        Print progress messages
    budget_seconds : float
        Total time budget for all SymPy operations (default 60s)
    """
    if not _HAVE_SYMPY:
        return expr

    # Reset the time budget for this simplification pass
    reset_sympy_budget(budget_seconds)

    # SymPy can recurse deeply in some paths; give it headroom.
    try:
        if sys.getrecursionlimit() < 5000:
            sys.setrecursionlimit(5000)
    except Exception:
        pass

    # 1. First pass: recognize numeric constants from a small symbolic grammar
    # before nsimplify has a chance to turn them into accidental rationals.
    try:
        with sympy_timeout("snap_symbolic_constants", max_seconds=5):
            expr = snap_numeric_constants(
                expr,
                snap_targets=final_polish_snap_targets(),
                snap_rel_tol=1e-4,
            )
            expr = canonicalize_trig_phases(expr, snap_rel_tol=1e-4)
    except RecursionError:
        if verbose:
            print("[SymPy] snap_symbolic_constants hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] snap_symbolic_constants timed out; skipping")

    # 2. Recognize numeric constants like 0.3989 → 1/√(2π)
    try:
        with sympy_timeout("nsimplify_floats", max_seconds=10):
            expr = _nsimplify_floats(expr, constants=COMMON_CONSTS, tolerance=1e-4)
    except RecursionError:
        if verbose:
            print("[SymPy] _nsimplify_floats() hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] _nsimplify_floats() timed out; skipping")

    # Check budget
    if get_sympy_budget_remaining() < 1.0:
        if verbose:
            print("[SymPy] Time budget exhausted; returning early")
        return expr

    # 3. Promote remaining floats to rationals + common constants
    try:
        with sympy_timeout("nsimplify_compat", max_seconds=10):
            expr = _nsimplify_compat(
                expr, constants=COMMON_CONSTS, rational=True, maxsteps=100, tolerance=1e-4
            )
    except RecursionError:
        if verbose:
            print("[SymPy] _nsimplify_compat() hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] _nsimplify_compat() timed out; skipping")

    # Check budget
    if get_sympy_budget_remaining() < 1.0:
        if verbose:
            print("[SymPy] Time budget exhausted; returning early")
        return expr

    try:
        with sympy_timeout("canonicalize_trig_phases", max_seconds=5):
            expr = canonicalize_trig_phases(expr, snap_rel_tol=1e-4)
    except RecursionError:
        if verbose:
            print("[SymPy] canonicalize_trig_phases hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] canonicalize_trig_phases timed out; skipping")

    # 4. State assumptions on x0..x_{Nx-1}
    try:
        with sympy_timeout("xreplace", max_seconds=5):
            xs = sp.symbols(f"x0:{Nxvars}", real=True, finite=True)
            subs = {sp.Symbol(f"x{i}"): xs[i] for i in range(Nxvars)}
            expr = expr.xreplace(subs)
    except RecursionError:
        if verbose:
            print("[SymPy] xreplace() hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] xreplace() timed out; skipping")

    # Optional: if some variables are known to be positive, extend the assumptions here.

    # Check budget
    if get_sympy_budget_remaining() < 1.0:
        if verbose:
            print("[SymPy] Time budget exhausted; returning early")
        return expr

    # 5. Rational / algebraic cleanup
    # Guard: together()+cancel() can expand compound subexpressions (e.g. Padé
    # of a skeleton) into high-degree polynomials.  Only keep the result if
    # it does not increase the operation count.
    expr_pre_tc = expr
    ops_pre_tc = sp.count_ops(expr)

    try:
        with sympy_timeout("together", max_seconds=10):
            expr = sp.together(expr)
    except RecursionError:
        if verbose:
            print("[SymPy] together() hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] together() timed out; skipping")

    # Check budget
    if get_sympy_budget_remaining() < 1.0:
        if verbose:
            print("[SymPy] Time budget exhausted; returning early")
        return expr

    try:
        with sympy_timeout("cancel", max_seconds=10):
            expr = sp.cancel(expr)
    except RecursionError:
        if verbose:
            print("[SymPy] cancel() hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] cancel() timed out; skipping")

    # Revert if together+cancel increased complexity
    ops_post_tc = sp.count_ops(expr)
    if ops_post_tc > ops_pre_tc:
        if verbose:
            print(
                f"[SymPy] together+cancel increased complexity "
                f"({ops_pre_tc}→{ops_post_tc} ops); reverting"
            )
        expr = expr_pre_tc

    # Check budget
    if get_sympy_budget_remaining() < 1.0:
        if verbose:
            print("[SymPy] Time budget exhausted; returning early")
        return expr

    try:
        with sympy_timeout("radsimp", max_seconds=10):
            expr = sp.radsimp(expr)
    except RecursionError:
        if verbose:
            print("[SymPy] radsimp() hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] radsimp() timed out; skipping")

    # Check budget
    if get_sympy_budget_remaining() < 1.0:
        if verbose:
            print("[SymPy] Time budget exhausted; returning early")
        return expr

    # 6. Trig cleanup
    try:
        with sympy_timeout("trigsimp", max_seconds=15):
            expr = canonicalize_trig_phases(sp.trigsimp(expr), snap_rel_tol=1e-4)
    except RecursionError:
        if verbose:
            print("[SymPy] trigsimp() hit recursion limit; skipping")
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] trigsimp() timed out; skipping")

    # Check budget
    budget_left = get_sympy_budget_remaining()
    if verbose:
        print(f"[SymPy] After trigsimp, budget remaining: {budget_left:.1f}s")
    if budget_left < 1.0:
        if verbose:
            print("[SymPy] Time budget exhausted; returning early")
        return expr

    # 7. Final polish
    if verbose:
        print("[SymPy] Starting simplify()...")
    try:
        with sympy_timeout("simplify", max_seconds=15):
            expr = sp.simplify(expr)
        if verbose:
            print("[SymPy] simplify() completed")
    except RecursionError:
        if verbose:
            print(
                "[SymPy] simplify hit recursion limit; returning partially simplified expression."
            )
    except SympyTimeoutError:
        if verbose:
            print("[SymPy] simplify() timed out; returning partially simplified expression.")

    if verbose:
        print("[SymPy] aggressive_simplify result:", expr)

    return expr


# ---- NEW: stability/canonicalization helpers for Stage C ----


def _prefer_stable_half_angle_trig(expr):
    """
    Rewrite (cos(x)±1) forms back into half-angle sin/cos squares.

    This is primarily a NUMERICAL stability preference: many trigsimp
    routes introduce (cos(x)-1) denominators, which are catastrophically
    ill-conditioned near x≈0 and can break the numeric equivalence gate.
    """
    if not _HAVE_SYMPY:
        return expr
    try:
        expr = sp.factor_terms(expr)
        A = sp.Wild("A")
        w = sp.Wild("w")
        # Handle affine forms A*cos(w) ± A (common after together/cancel),
        # which do NOT match cos(w)±1 directly.
        expr = expr.replace(A * sp.cos(w) - A, -2 * A * sp.sin(w / 2) ** 2)
        expr = expr.replace(A - A * sp.cos(w), 2 * A * sp.sin(w / 2) ** 2)
        expr = expr.replace(A * sp.cos(w) + A, 2 * A * sp.cos(w / 2) ** 2)
        expr = expr.replace(A + A * sp.cos(w), 2 * A * sp.cos(w / 2) ** 2)
        # Also handle the pure ±1 forms
        expr = expr.replace(sp.cos(w) - 1, -2 * sp.sin(w / 2) ** 2)
        expr = expr.replace(1 - sp.cos(w), 2 * sp.sin(w / 2) ** 2)
        expr = expr.replace(sp.cos(w) + 1, 2 * sp.cos(w / 2) ** 2)
        expr = expr.replace(1 + sp.cos(w), 2 * sp.cos(w / 2) ** 2)
        expr = sp.together(expr)
        expr = sp.cancel(expr)
    except Exception:
        return expr
    return expr


def _is_minus_one_sympy(v, *, tol: float = 1e-12) -> bool:
    """Return True when a SymPy exponent is numerically -1 (incl. Float(-1.0))."""
    try:
        if v == -1:
            return True
        vf = float(v)
        return math.isfinite(vf) and abs(vf + 1.0) <= float(tol)
    except Exception:
        return False


def _canonicalize_inverse_ratio_powers(expr):
    """
    Canonicalize common inverse-power ratio forms for cleaner Stage-C output.

      - (u**-1)**-1              -> u
      - (a * b**-1)**-1          -> b/a
      - (a**-1 * b)**-1          -> a/b

    This is intentionally conservative (2-factor products only) and does not
    change semantics; it only normalizes display-form expressions.
    """
    if not _HAVE_SYMPY:
        return expr
    try:
        out = expr

        def _is_inv_pow(e):
            return isinstance(e, sp.Pow) and _is_minus_one_sympy(getattr(e, "exp", None))

        def _rw(e):
            if not _is_inv_pow(e):
                return e
            base = e.base
            if _is_inv_pow(base):
                return base.base
            if isinstance(base, sp.Mul) and len(base.args) == 2:
                a0, a1 = base.args
                if _is_inv_pow(a0):
                    return a0.base / a1
                if _is_inv_pow(a1):
                    return a1.base / a0
            return e

        # Fixed-point rewrite: nested inverse patterns often appear in two layers.
        for _ in range(3):
            nxt = out.replace(_is_inv_pow, _rw)
            if nxt == out:
                break
            out = nxt
        return out
    except Exception:
        return expr


def _rat(v, max_den: int = 1000000, zero: float = 1e-14):
    """
    Robust float -> "nice" Rational coercion.
    Uses limit_denominator to avoid huge binary-float rationals and to make
    cancellations (e.g. prefactors) actually happen.
    Returns None for non-finite inputs.
    """
    if not _HAVE_SYMPY:
        return v
    try:
        vf = float(v)
    except Exception:
        try:
            return sp.Float(v)
        except Exception:
            return None
    if not math.isfinite(vf):
        return None
    if abs(vf) < zero:
        return sp.Integer(0)
    try:
        return sp.Rational(vf).limit_denominator(int(max_den))
    except Exception:
        return sp.Float(vf)


def _sparsify_poly_like(expr, *, rel_tol=1e-4, abs_tol=0.0):
    """
    Drop tiny polynomial terms (relative to the largest coefficient) and
    then nsimplify remaining coefficients.

    Used to clean trig arguments like:
        0.5*x1*x2 + 1e-6*x2 + 1e-7*x1^2 + ...
    -> 0.5*x1*x2
    """
    if not _HAVE_SYMPY:
        return expr
    syms = [
        s
        for s in sorted(expr.free_symbols, key=lambda s: s.name)
        if s.name.startswith("x") and s.name[1:].isdigit()
    ]
    if not syms:
        return expr
    try:
        P = sp.Poly(expr, *syms)
    except Exception:
        return expr
    terms = P.terms()
    if not terms:
        return expr
    try:
        max_abs = max(abs(float(c.evalf())) for _, c in terms)
    except Exception:
        return expr
    if (not math.isfinite(max_abs)) or max_abs <= 0.0:
        return expr
    out = sp.Integer(0)
    thr = max(abs_tol, rel_tol * max_abs)
    for mon, c in terms:
        try:
            cv = float(c.evalf())
        except Exception:
            return expr
        if abs(cv) < thr:
            continue
        mon_expr = sp.Integer(1)
        for s, p in zip(syms, mon):
            p = int(p)
            if p:
                mon_expr *= s**p
        out += c * mon_expr
    try:
        out = _nsimplify_compat(
            out, constants=AIF_CONSTS, rational=True, maxsteps=80, tolerance=1e-4
        )
    except Exception:
        pass
    try:
        out = sp.expand(out)
    except Exception:
        pass
    return out


def _cleanup_trig_poly_args(expr, *, rel_tol=1e-4):
    """
    Apply _sparsify_poly_like to the arguments of sin/cos calls.
    """
    if not _HAVE_SYMPY:
        return expr
    try:

        def _is_trig_call(e):
            return (getattr(e, "func", None) in (sp.sin, sp.cos)) and (
                len(getattr(e, "args", ())) == 1
            )

        def _rw(e):
            a = e.args[0]
            a2 = _sparsify_poly_like(a, rel_tol=rel_tol, abs_tol=0.0)
            return e.func(a2) if a2 != a else e

        return expr.replace(_is_trig_call, _rw)
    except Exception:
        return expr


def _collect_val_points_from_loader(model, val_loader, device, *, max_points=2048):
    """
    Collect (xs_np, ys_model_np) from the fitted Stage-B model on a subset of the val set.
    """
    xs_list, ys_list = [], []
    n = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            x = x.to(device)
            y_pred = model(x)
            if y_pred.dim() == 2:
                y_pred = y_pred[:, 0]
            else:
                y_pred = y_pred.view(-1)
            xs_list.append(x.detach().cpu().numpy())
            ys_list.append(y_pred.detach().cpu().numpy().ravel())
            n += int(x.shape[0])
            if n >= int(max_points):
                break
    if not xs_list:
        return None, None
    xs = np.concatenate(xs_list, axis=0)[:max_points]
    ys = np.concatenate(ys_list, axis=0)[:max_points]
    return xs, ys


def _linearize_leaf_calls(expr, *, model, xs_np, device, max_rel_rmse=1e-6, verbose=False):
    """
    If expr contains leaf{i}(xj) calls, try to replace each by a*xj+b when the
    corresponding trained leaf module is numerically (almost) linear over xs_np.

    This is an OPTIONAL post-pass: we still validate the full expression
    numerically against the model, so nothing gets accepted unless it is
    within tolerance.
    """
    if not _HAVE_SYMPY:
        return expr
    leaves = getattr(model, "leaf", None)
    if leaves is None:
        return expr
    leaves = list(leaves)
    calls = []
    try:
        for fcall in expr.atoms(sp.Function):
            fn = getattr(getattr(fcall, "func", None), "__name__", None)
            if not fn or not fn.startswith("leaf"):
                continue
            calls.append(fcall)
    except Exception:
        return expr
    if not calls:
        return expr
    repl = {}
    for fcall in calls:
        fn = getattr(fcall.func, "__name__", "")
        try:
            idx = int(fn.replace("leaf", ""))
        except Exception:
            continue
        if idx < 0 or idx >= len(leaves):
            continue
        if len(getattr(fcall, "args", ())) != 1:
            continue
        arg = fcall.args[0]
        if not (isinstance(arg, sp.Symbol) and arg.name.startswith("x") and arg.name[1:].isdigit()):
            continue
        j = int(arg.name[1:])
        if xs_np.ndim != 2 or j >= xs_np.shape[1]:
            continue
        leaf_mod = leaves[idx]
        xj = xs_np[:, j].reshape(-1, 1)
        try:
            dt = next(leaf_mod.parameters()).dtype
        except Exception:
            dt = torch.get_default_dtype()
        with torch.no_grad():
            yt = leaf_mod(torch.as_tensor(xj, device=device, dtype=dt))
            if yt.dim() == 2:
                yt = yt[:, 0]
            else:
                yt = yt.view(-1)
            y = yt.detach().cpu().numpy().ravel()
        X = np.stack([xj.ravel(), np.ones_like(xj.ravel())], axis=1)
        try:
            a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            continue
        resid = y - (a * xj.ravel() + b)
        rmse = float(np.sqrt(np.mean(resid**2)))
        denom = float(np.median(np.abs(y)))
        denom = denom if (np.isfinite(denom) and denom > 0.0) else 1.0
        rel_rmse = rmse / denom
        if (max_rel_rmse is not None) and (rel_rmse > max_rel_rmse):
            continue

        # Drop tiny intercepts (common when the true mapping is a*x)
        x_med = float(np.median(np.abs(xj.ravel())))
        if abs(b) < 1e-6 * abs(a) * max(1.0, x_med):
            b = 0.0
        a_sp = _rat(a, max_den=1000000)
        b_sp = _rat(b, max_den=1000000)
        if a_sp is None or b_sp is None:
            continue
        r = a_sp * arg
        if b_sp != 0:
            r = r + b_sp
        repl[fcall] = r
        if verbose:
            print(f"[Stage C] linearized {fn}({arg}): a={a_sp}, b={b_sp}, rel_rmse={rel_rmse:.3g}")
    if not repl:
        return expr
    try:
        with sympy_timeout("xreplace_linearize", max_seconds=5):
            expr2 = expr.xreplace(repl)
    except (Exception, SympyTimeoutError):
        return expr
    try:
        with sympy_timeout("simplify_linearize", max_seconds=10):
            expr2 = sp.simplify(expr2)
    except (Exception, SympyTimeoutError):
        pass
    try:
        with sympy_timeout("stable_trig_linearize", max_seconds=5):
            expr2 = _prefer_stable_half_angle_trig(expr2)
    except (Exception, SympyTimeoutError):
        pass
    return expr2


def _monomial_str(var_idxs, exps_row, input_expr=None, extra_var_idxs=None, extra_nodes=None, tol=1e-12):
    """Generate monomial string from exponents.

    Uses input_expr (the first input expression) for dimension 0 and
    extra_var_idxs for remaining dimensions.  When input_expr is a trivial
    Var(i), this produces the same output as the old simple-variable path.

    Parameters
    ----------
    extra_nodes : list[Node], optional
        Actual AST nodes for the extra inputs (inputs[1:]).  When provided,
        nontrivial nodes (e.g. CosNode(Var(2))) are rendered via
        ``_input_expr_to_str`` instead of as plain ``x{idx}``.
    """
    # Build label for each dimension from the input expressions
    from nestynet_sr.sr_core.bridges import is_trivial_input

    labels: list[str] = []
    if input_expr is not None:
        z_str = _input_expr_to_str(input_expr)
        if not is_trivial_input(input_expr):
            z_str = _parenthesize_if_needed(z_str)
        labels.append(z_str)
        ev = extra_var_idxs or []
        en = extra_nodes or []
        for i in range(len(exps_row) - 1):
            if i < len(en) and not is_trivial_input(en[i]):
                # Use the actual AST node (e.g. CosNode(Var(2)))
                lbl = _input_expr_to_str(en[i])
                lbl = _parenthesize_if_needed(lbl)
                labels.append(lbl)
            else:
                idx = ev[i] if i < len(ev) else (var_idxs[i + 1] if (i + 1) < len(var_idxs) else 0)
                labels.append(f"x{idx}")
    else:
        # Legacy fallback (should not happen after unification)
        for vidx in var_idxs:
            labels.append(f"x{vidx}")

    factors = []
    for label, p in zip(labels, exps_row):
        p = int(p)
        if p == 0:
            continue
        if p == 1:
            factors.append(label)
        else:
            factors.append(f"{_parenthesize_if_needed(label, as_power_base=True)}^{p}")
    return "*".join(factors) if factors else "1"


def _poly_str(
    exps,
    coeffs,
    var_idxs,
    input_expr=None,
    extra_var_idxs=None,
    extra_nodes=None,
    tol=1e-10,
    sig: int = 6,
    *,
    preserve_coefficients: bool = False,
):
    exps = exps.detach().cpu()
    coeffs = coeffs.detach().cpu()
    terms = []
    for e, c in zip(exps, coeffs):
        c = float(c)
        if c == 0.0 or (not preserve_coefficients and abs(c) < tol):
            continue
        mon = _monomial_str(var_idxs, e, input_expr=input_expr, extra_var_idxs=extra_var_idxs, extra_nodes=extra_nodes)
        sgn = 1 if c >= 0 else -1
        mag = abs(c)
        if mon == "1":
            core = _fmt(mag, sig)
        else:
            unit_coefficient = (
                mag == 1.0
                if preserve_coefficients
                else abs(mag - 1.0) < 1e-8
            )
            if unit_coefficient:
                core = mon
            else:
                core = f"{_fmt(mag, sig)}*{mon}"
        terms.append((sgn, core))
    if not terms:
        return "0"
    sgn, core = terms[0]
    s = ("-" if sgn < 0 else "") + core
    for sgn, core in terms[1:]:
        s += " - " + core if sgn < 0 else " + " + core
    return s


def _polylog_str(
    exps,
    coeffs,
    var_idxs,
    input_expr=None,
    extra_var_idxs=None,
    extra_nodes=None,
    tol=1e-10,
    sig: int = 6,
):
    exps = exps.detach().cpu()
    coeffs = coeffs.detach().cpu()
    terms = []

    from nestynet_sr.sr_core.bridges import is_trivial_input

    def label_for_dim(local_idx: int) -> str:
        if local_idx == 0 and input_expr is not None:
            label = _input_expr_to_str(input_expr)
        elif local_idx > 0 and extra_nodes is not None and local_idx - 1 < len(extra_nodes):
            node = extra_nodes[local_idx - 1]
            if not is_trivial_input(node):
                label = _input_expr_to_str(node)
            else:
                idx = var_idxs[local_idx] if local_idx < len(var_idxs) else 0
                label = f"x{int(idx)}"
        elif local_idx > 0 and extra_var_idxs is not None and local_idx - 1 < len(extra_var_idxs):
            label = f"x{int(extra_var_idxs[local_idx - 1])}"
        else:
            idx = var_idxs[local_idx] if local_idx < len(var_idxs) else 0
            label = f"x{int(idx)}"

        if any(op in label for op in ("*", "+", "/", "^", " - ")):
            return f"({label})"
        return label

    for e, c_t in zip(exps, coeffs):
        c = float(c_t)
        if abs(c) < tol:
            continue

        # Build a monomial in log(input_j).  For compound atoms, input_0 is
        # the detected coordinate z, not the first raw variable in var_idxs.
        factors = []
        for local_idx, p_t in enumerate(e):
            p = int(p_t)
            if p == 0:
                continue
            base = f"log({label_for_dim(local_idx)})"
            if p == 1:
                factors.append(base)
            else:
                factors.append(f"{base}**{p}")

        mon = "1" if not factors else "*".join(factors)

        sgn = 1 if c >= 0 else -1
        mag = abs(c)
        if mon == "1":
            core = _fmt(mag, sig)
        else:
            if abs(mag - 1.0) < tol:
                core = mon
            else:
                core = f"{_fmt(mag, sig)}*{mon}"
        terms.append((sgn, core))

    if not terms:
        return "0"

    # Assemble with explicit +/-
    s = ""
    for k, (sgn, core) in enumerate(terms):
        if k == 0:
            s += core if sgn > 0 else f"-{core}"
        else:
            s += " + " + core if sgn > 0 else " - " + core
    return s


# ---- new: leaf → (scale, core) helpers ----


def _simplify_coeffs_vector(
    coeffs: torch.Tensor, rel_tol: float = 1e-3, snap_targets=(0.5, 1.0, 2.0)
) -> torch.Tensor:
    """
    Heuristic simplifier for polynomial coefficients, used only for pretty-printing.

    - Drops coefficients that are tiny relative to the largest one.
    - Snaps coefficients that are close to a few 'nice' values like ±0.5, ±1, ±2.

    This doesn't modify the model, only the printed representation.
    """
    coeffs = coeffs.clone()
    max_abs = float(coeffs.abs().max())
    if max_abs == 0.0:
        return coeffs

    for k in range(len(coeffs)):
        v = float(coeffs[k])
        if abs(v) < rel_tol * max_abs:
            coeffs[k] = 0.0
            continue
        # snap near nice values
        for t in snap_targets:
            for s in (+1.0, -1.0):
                cand = s * t
                if abs(v - cand) < rel_tol:
                    coeffs[k] = cand
                    v = cand
                    break
            else:
                continue
            break
    return coeffs


def _poly_leaf_repr(
    core: PolyLeaf, var_idxs, input_expr=None, extra_var_idxs=None, extra_nodes=None, tol=1e-10, rel_offset_tol: float = 1e-3, sig: int = 6
):
    """
    Return (scale, core_str) for a PolyLeaf.

    Special case: 0D poly (no inputs) is a pure scalar parameter; represent it
    as a multiplicative scale (scale=c0, core="1") so it can't get "lost" inside
    a string factor in multiplicative chains.

    Special case: 1D linear poly: c0 + c1*xj ≈ c1 * xj  (dropping tiny offset),
    so scale=c1, core="xj". Otherwise fall back to full polynomial.

    If input_expr is provided (compound variable), use it as the variable
    representation instead of x{j}.
    """
    # Support reduced variants by reconstructing the full coefficient vector
    # for printing purposes (e.g. RExpPolyLeaf).
    if hasattr(core, "exps_full") and hasattr(core, "full_coeffs"):
        exps = core.exps_full.detach().cpu()
        coeffs = core.full_coeffs().detach().cpu()
    else:
        exps = core.exps.detach().cpu()
        coeffs = core.coeffs.detach().cpu()

    # 0D scalar poly leaf: treat as pure scale
    if len(var_idxs) == 0:
        try:
            c0 = float(coeffs.view(-1)[0]) if coeffs.numel() else 0.0
        except Exception:
            c0 = 0.0
        # IMPORTANT: do not magnitude-prune scalar coefficients here.
        # In multiplicative chains (e.g. poly(x0)*poly(x1)*poly()*cos(...)),
        # LM can push this scalar very small while other factors carry
        # compensating large scales (gauge freedom). Hard-zeroing here
        # deletes real structure and leads to "0*cos(...)” printouts.
        return c0, "1"

    # 1D degree-1 special case (also applies to 1D compound variables)
    # For compound variables: core.n_in == 1 even when len(var_idxs) > 1
    n_in = getattr(core, "n_in", len(var_idxs))
    if n_in == 1 and getattr(core, "degree", None) == 1 and exps.shape[0] == 2:
        # Determine variable representation (unified: input_expr always provided)
        var_str = _input_expr_to_str(input_expr)
        needs_parens = _needs_parens_as_factor(var_str)

        # exps should be [(0,), (1,)], but be robust
        idx0 = int((exps[:, 0] == 0).nonzero()[0])
        idx1 = int((exps[:, 0] == 1).nonzero()[0])
        c0 = float(coeffs[idx0])
        c1 = float(coeffs[idx1])

        if abs(c1) < tol:
            # Pure constant; treat as scalar
            return c0, "1"

        scale = c1
        offset = c0 / c1

        # If offset is tiny compared to 1, just drop it
        # Use strict threshold (1e-6) to preserve intercepts down to 1 ppm of scale
        if abs(offset) < 1e-6:
            if needs_parens:
                return scale, f"({var_str})"
            return scale, var_str

        # Otherwise show a shifted linear factor
        if offset > 0:
            core_str = f"({var_str} + {_fmt(offset, sig)})"
        else:
            core_str = f"({var_str} - {_fmt(abs(offset), sig)})"
        return scale, core_str

    # Fallback: full polynomial as string, no separate scale
    return 1.0, _poly_str(exps, coeffs, var_idxs, input_expr=input_expr, extra_var_idxs=extra_var_idxs, extra_nodes=extra_nodes, tol=tol, sig=sig)


def _ratio_poly_leaf_repr(
    core: RatioPolyLeaf, var_idxs, tol=1e-10, sig: int = 6
):
    """
    Return (scale, core_str) for a RatioPolyLeaf.

    RatioPolyLeaf computes P(r) where r = x_num / x_den.
    var_idxs = (num_idx, den_idx) specifies which variables.

    Examples:
        - Constant: c0 -> (c0, "1")
        - Linear: c0 + c1*r -> ((c0 + c1*(x2/x1)))
        - Quadratic: 1 - r^2 -> (1 - (x2/x1)^2)
    """
    if isinstance(core, RRatioPolyLeaf):
        coeffs = core.full_coeffs().detach().cpu()
        degree = core.degree
    else:
        coeffs = core.coeffs.detach().cpu()
        degree = core.degree

    if len(var_idxs) != 2:
        # Unexpected: ratio poly needs exactly 2 variables
        return 1.0, f"ratio_poly({', '.join(f'x{j}' for j in var_idxs)})"

    num_idx, den_idx = int(var_idxs[0]), int(var_idxs[1])
    ratio_str = f"(x{num_idx}/x{den_idx})"

    # Build polynomial string
    terms = []
    for k in range(degree + 1):
        c_k = float(coeffs[k])
        if abs(c_k) < tol:
            continue

        if k == 0:
            terms.append(_fmt(c_k, sig))
        elif k == 1:
            if abs(c_k - 1.0) < tol:
                terms.append(ratio_str)
            elif abs(c_k + 1.0) < tol:
                terms.append(f"-{ratio_str}")
            else:
                terms.append(f"{_fmt(c_k, sig)}*{ratio_str}")
        else:
            power_str = f"{ratio_str}**{k}"
            if abs(c_k - 1.0) < tol:
                terms.append(power_str)
            elif abs(c_k + 1.0) < tol:
                terms.append(f"-{power_str}")
            else:
                terms.append(f"{_fmt(c_k, sig)}*{power_str}")

    if not terms:
        return 0.0, "0"

    # Join terms with proper sign handling
    result = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            result += f" - {t[1:]}"
        else:
            result += f" + {t}"

    # Wrap in parentheses if more than one term
    if len(terms) > 1:
        result = f"({result})"

    return 1.0, result


def _input_expr_to_str(node) -> str:
    """Convert an input_expr AST node to a string representation.

    Handles compound variables like x1*x2, x1+x2, x1^2, etc.
    """
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind == "var":
            idx = node.var_idxs[0]
            return f"x{idx}"
        else:
            # Fallback for unexpected atom types
            return f"x{node.var_idxs[0]}" if node.var_idxs else "?"
    elif isinstance(node, AddNode):
        left = _input_expr_to_str(node.left)
        right = _input_expr_to_str(node.right)
        return f"({left} + {right})"
    elif isinstance(node, MulNode):
        left = _input_expr_to_str(node.left)
        right = _input_expr_to_str(node.right)
        # Avoid redundant parentheses for simple products
        if " " not in left and " " not in right:
            return f"{left}*{right}"
        return f"({left})*({right})"
    elif isinstance(node, PowNode):
        base = _input_expr_to_str(node.base)
        exp_val = node.exponent
        try:
            exp_f = float(exp_val)
            if math.isfinite(exp_f) and abs(exp_f - round(exp_f)) < 1e-12:
                exp_str = str(int(round(exp_f)))
            else:
                exp_str = f"{exp_f:g}"
        except Exception:
            exp_str = str(exp_val)
        return f"({base})^{exp_str}"
    elif isinstance(node, ConstNode):
        return format_const_value(node.value)
    elif isinstance(node, SinNode):
        arg = _input_expr_to_str(node.arg)
        return f"sin({arg})"
    elif isinstance(node, CosNode):
        arg = _input_expr_to_str(node.arg)
        return f"cos({arg})"
    elif isinstance(node, AsinNode):
        arg = _input_expr_to_str(node.arg)
        return f"asin({arg})"
    elif isinstance(node, AcosNode):
        arg = _input_expr_to_str(node.arg)
        return f"acos({arg})"
    elif isinstance(node, AtanNode):
        arg = _input_expr_to_str(node.arg)
        return f"atan({arg})"
    elif isinstance(node, LogNode):
        arg = _input_expr_to_str(node.arg)
        return f"log({arg})"
    elif isinstance(node, ExpNode):
        arg = _input_expr_to_str(node.arg)
        return f"exp({arg})"
    else:
        return "?"


def _sin_leaf_repr(core: SinLinearLeaf, var_idxs, input_expr=None, tol=1e-10, sig: int = 6):
    """
    Return (amp, 'sin(...)') for SinLinearLeaf, factoring amplitude out
    and printing a reasonably clean argument.

    Parameters
    ----------
    input_expr : Node
        The input expression for this atom (always provided).
    """
    w = float(core.weight.detach().cpu().view(-1)[0])
    b = float(core.bias.detach().cpu())
    amp = float(core.amp.detach().cpu())

    if abs(amp) < tol:
        return 0.0, "1"

    var_str = _input_expr_to_str(input_expr)

    # Build argument string
    # If w≈1, just show var_str; otherwise show w*var_str
    if abs(w - 1.0) < 1e-3:
        arg = var_str
    else:
        # Wrap compound expressions in parentheses if needed
        if _needs_parens_as_factor(var_str) or "^" in var_str or "**" in var_str:
            arg = f"{_fmt(w, sig)}*({var_str})"
        else:
            arg = f"{_fmt(w, sig)}*{var_str}"

    # Add phase if it isn't tiny
    if abs(b) > 1e-3:
        sign = "+" if b > 0 else "-"
        arg = f"{arg} {sign} {_fmt(abs(b), sig)}"

    core_str = f"sin({arg})"
    return amp, core_str


def _tanh_leaf_repr(core: TanhLinearLeaf, var_idxs, input_expr=None, tol=1e-10, sig: int = 6):
    """Return (amp, 'tanh(...)') for TanhLinearLeaf.

    Parameters
    ----------
    input_expr : Node, optional
        For compound atoms, the AST representing the compound variable.
    """
    w = float(core.weight.detach().cpu().view(-1)[0])
    b = float(core.bias.detach().cpu())
    amp = float(core.amp.detach().cpu())

    if abs(amp) < tol:
        return 0.0, "1"

    var_str = _input_expr_to_str(input_expr)

    # Build argument string
    if abs(w - 1.0) < 1e-3:
        arg = var_str
    else:
        if _needs_parens_as_factor(var_str) or "^" in var_str or "**" in var_str:
            arg = f"{_fmt(w, sig)}*({var_str})"
        else:
            arg = f"{_fmt(w, sig)}*{var_str}"

    if abs(b) > 1e-3:
        sign = "+" if b > 0 else "-"
        arg = f"{arg} {sign} {_fmt(abs(b), sig)}"

    core_str = f"tanh({arg})"
    return amp, core_str


def _power_leaf_repr(core: PowerLeaf, var_idxs, input_expr=None, tol: float = 1e-10, sig: int = 6):
    """
    Return (amp, 'xj^p' / 'sqrt(xj)' / '1/sqrt(xj)') for PowerLeaf,
    factoring amplitude out.

    Parameters
    ----------
    input_expr : Node, optional
        For compound atoms, the AST representing the compound variable.
        If provided, this is used instead of var_idxs[0].
    """
    z_str = _input_expr_to_str(input_expr)
    var_str = _parenthesize_if_needed(z_str, as_power_base=True)

    amp = float(core.amp.detach().cpu())
    p = float(core.exponent.detach().cpu())

    if abs(amp) < tol:
        return 0.0, "1"

    def close(val, target, thr=1e-3):
        return abs(val - target) < thr

    if close(p, 1.0):
        core_str = var_str
    elif close(p, 2.0):
        core_str = f"{var_str}^2"
    elif close(p, 0.5):
        core_str = f"sqrt({z_str})"
    elif close(p, -0.5):
        core_str = f"1/sqrt({z_str})"
    elif close(p, 0.0):
        core_str = "1"
    else:
        core_str = f"{var_str}^{_fmt(p, sig)}"

    return amp, core_str


def _inv_monomial_leaf_repr(core: InverseMonomialLeaf, var_idxs, input_expr=None, tol: float = 1e-10, sig: int = 6):
    """
    Return (amp, '1/xj' / '1/xj^2' / ...) for InverseMonomialLeaf,
    factoring amplitude out.

    InverseMonomialLeaf computes: amp / x^degree with fixed integer degree.

    Parameters
    ----------
    input_expr : Node, optional
        For compound atoms, the AST representing the compound variable (e.g., x1*x2).
        If provided, this is used instead of var_idxs[0].
    """
    amp = float(core.amp.detach().cpu())
    degree = int(core.degree)

    if abs(amp) < tol:
        return 0.0, "1"

    z_str = _input_expr_to_str(input_expr)
    var_str = _parenthesize_if_needed(z_str, as_power_base=True)

    if degree == 1:
        core_str = f"1/{var_str}"
    elif degree == 2:
        core_str = f"1/{var_str}^2"
    else:
        core_str = f"1/{var_str}^{degree}"

    return amp, core_str


def _rinv_monomial_leaf_repr(core: "RInverseMonomialLeaf", var_idxs, input_expr=None, tol: float = 1e-10):
    """
    Return (1.0, '1/xj^degree') for RInverseMonomialLeaf (parameter-free monic inverse monomial).
    """
    degree = int(core.degree)

    z_str = _input_expr_to_str(input_expr)
    var_str = _parenthesize_if_needed(z_str, as_power_base=True)

    if degree == 1:
        core_str = f"1/{var_str}"
    elif degree == 2:
        core_str = f"1/{var_str}^2"
    else:
        core_str = f"1/{var_str}^{degree}"

    return 1.0, core_str


def _exp_poly_leaf_repr(core: ExpPolyLeaf | RExpPolyLeaf, var_idxs, input_expr=None, extra_var_idxs=None, extra_nodes=None, tol: float = 1e-10, sig: int = 6):
    """
    Return (scale, 'exp(poly(x))') for ExpPolyLeaf, factoring out the
    constant term in the exponent as a multiplicative scale.

    If P(x) = c0 + Q(x), we print  exp(P(x)) = exp(c0) * exp(Q(x)),
    then simplify Q(x) heuristically.
    """
    # Support reduced variants by reconstructing the full coefficient vector
    # for printing purposes (e.g. RExpPolyLeaf).
    if hasattr(core, "exps_full") and hasattr(core, "full_coeffs"):
        exps = core.exps_full.detach().cpu()
        coeffs = core.full_coeffs().detach().cpu()
    else:
        exps = core.exps.detach().cpu()
        coeffs = core.coeffs.detach().cpu()

    # Find constant term in exponent, if present
    const_idx = None
    for k, e in enumerate(exps):
        if int(e.sum().item()) == 0:
            const_idx = k
            break

    c0 = float(coeffs[const_idx]) if const_idx is not None else 0.0
    coeffs_nc = coeffs.clone()
    if const_idx is not None:
        coeffs_nc[const_idx] = 0.0

    # Heuristic simplification: drop tiny terms, snap to nice values
    coeffs_nc = _simplify_coeffs_vector(coeffs_nc, rel_tol=1e-3, snap_targets=(0.5, 1.0, 2.0))

    poly_str = _poly_str(exps, coeffs_nc, var_idxs, input_expr=input_expr, extra_var_idxs=extra_var_idxs, extra_nodes=extra_nodes, tol=tol, sig=sig)

    # If exponent is effectively constant, we just get a scalar
    if poly_str == "0":
        scale = float(math.exp(c0))
        return scale, "1"

    # Factor exp(constant) as an overall scale
    scale = float(math.exp(c0)) if abs(c0) > tol else 1.0
    return scale, f"exp({poly_str})"


def _exp_ratpoly_leaf_repr(core: ExpRationalPolyLeaf, var_idxs, input_expr=None, extra_var_idxs=None, extra_nodes=None, tol: float = 1e-10, sig: int = 6):
    num = _poly_str(core.exps_num, core.coeffs_num, var_idxs, input_expr=input_expr, extra_var_idxs=extra_var_idxs, extra_nodes=extra_nodes, tol=tol, sig=sig)
    den = _poly_str(core.exps_den, core.coeffs_den, var_idxs, input_expr=input_expr, extra_var_idxs=extra_var_idxs, extra_nodes=extra_nodes, tol=tol, sig=sig)
    # A constant could also be factored out from num/den,
    # but this is already quite readable.
    return 1.0, f"exp(({num})/({den}))"


def _ratpoly_leaf_repr(core: RationalPolyLeaf, var_idxs, input_expr=None, extra_var_idxs=None, extra_nodes=None, tol: float = 1e-10, sig: int = 6):
    """
    Return a numerically faithful ``(scale, 'P(x)/Q(x)')`` representation.

    Parameters
    ----------
    input_expr : Node, optional
        For compound atoms, the AST representing the compound variable (e.g., x1*x2).
        If provided, this is used instead of var_idxs[0].
    extra_var_idxs : list[int], optional
        For multivariate compound atoms, the original variable indices for
        dimensions beyond the compound variable (dim 0).
    extra_nodes : list[Node], optional
        Actual AST nodes for inputs[1:], used to render nontrivial extras
        (e.g. CosNode(Var(2))) correctly.

    This is the authoritative serialization path used by Stage C.  It must not
    drop coefficients based on their size relative to another coefficient:
    reduced rational leaves contain a fixed monic numerator term, and small
    denominator terms can dominate on a wide input range.  Cosmetic
    simplification belongs in Stage C, where every candidate is checked against
    this faithful baseline.
    """
    if isinstance(core, RRationalPolyLeaf):
        exps_num = core.exps_num_full.detach().cpu()
        exps_den = core.exps_den.detach().cpu()
        coeffs_num = core.full_coeffs_num().detach().cpu().clone()
        coeffs_den = core.coeffs_den.detach().cpu().clone()
    else:
        exps_num = core.exps_num.detach().cpu()
        exps_den = core.exps_den.detach().cpu()
        coeffs_num = core.coeffs_num.detach().cpu().clone()
        coeffs_den = core.coeffs_den.detach().cpu().clone()

    faithful_sig = max(17, int(sig))
    num_str = _poly_str(
        exps_num,
        coeffs_num,
        var_idxs,
        input_expr=input_expr,
        extra_var_idxs=extra_var_idxs,
        extra_nodes=extra_nodes,
        tol=0.0,
        sig=faithful_sig,
        preserve_coefficients=True,
    )
    den_str = _poly_str(
        exps_den,
        coeffs_den,
        var_idxs,
        input_expr=input_expr,
        extra_var_idxs=extra_var_idxs,
        extra_nodes=extra_nodes,
        tol=0.0,
        sig=faithful_sig,
        preserve_coefficients=True,
    )

    return 1.0, f"({num_str})/({den_str})"


def _planck_leaf_repr(core: PlanckLeaf, var_idxs, input_expr=None, tol: float = 1e-10, sig: int = 6):
    """
    Return (amp, 'x^p/(exp(a*x + b) - 1)') for PlanckLeaf.

    Parameters
    ----------
    input_expr : Node, optional
        For compound atoms, the AST representing the compound variable (e.g., x1*x2).
        If provided, this is used instead of var_idxs[0].
    """
    amp = float(torch.exp(core.log_amp.detach().cpu()))
    p_obj = getattr(core, "p", 1.0)
    if torch.is_tensor(p_obj):
        p = float(p_obj.detach().cpu())
    else:
        p = float(p_obj)
    a = float(torch.exp(core.log_a.detach().cpu()))
    b_obj = getattr(core, "b", None)
    b = float(b_obj.detach().cpu()) if torch.is_tensor(b_obj) else 0.0

    if amp == 0.0:
        return 0.0, "1"

    var_str = _input_expr_to_str(input_expr)
    needs_parens = "*" in var_str or "+" in var_str or "^" in var_str
    faithful_sig = max(17, int(sig))
    reduced_power = isinstance(core, PlanckLeaf) and not isinstance(
        core, PlanckFullLeaf
    )

    # Reduced Planck powers are fixed structural choices and can be rendered
    # canonically.  PlanckFull powers are fitted values and must be preserved
    # literally; Stage C may propose a snap later under its numerical gate.
    if reduced_power and p == 0.0:
        num = "1"
    elif reduced_power and p == 1.0:
        num = f"({var_str})" if needs_parens else var_str
    elif reduced_power and p == 2.0:
        num = f"({var_str})^2" if needs_parens else f"{var_str}^2"
    elif reduced_power and p == 3.0:
        num = f"({var_str})^3" if needs_parens else f"{var_str}^3"
    else:
        p_str = _fmt(p, faithful_sig)
        num = f"({var_str})^{p_str}" if needs_parens else f"{var_str}^{p_str}"

    # Build exp() argument
    a_str = _fmt(a, faithful_sig)
    inner_terms = [
        f"{a_str}*({var_str})" if needs_parens else f"{a_str}*{var_str}"
    ]
    if b != 0.0:
        sign = "+" if b > 0 else "-"
        inner_terms.append(f"{sign} {_fmt(abs(b), faithful_sig)}")
    arg = " ".join(inner_terms)

    core_str = f"{num}/(exp({arg}) - 1)"
    return amp, core_str


def _expm1_leaf_repr(core: Expm1Leaf, var_idxs, input_expr=None, tol: float = 1e-10, sig: int = 6):
    """
    Return (amp, '(exp(a*x + b) - 1)') for Expm1Leaf.
    """
    amp = float(torch.exp(core.log_amp.detach().cpu()))
    a = float(torch.exp(core.log_a.detach().cpu()))
    b = float(core.b.detach().cpu())

    if abs(amp) < tol:
        return 0.0, "1"

    var_str = _input_expr_to_str(input_expr)
    needs_parens = "*" in var_str or "+" in var_str or "^" in var_str

    def close(v, t, thr=1e-3):
        return abs(v - t) < thr

    # Build exp argument: a*x + b
    inner_terms = []
    if close(a, 1.0):
        inner_terms.append(f"({var_str})" if needs_parens else var_str)
    else:
        inner_terms.append(f"{_fmt(a, sig)}*({var_str})" if needs_parens else f"{_fmt(a, sig)}*{var_str}")
    if abs(b) > 1e-3:
        sign = "+" if b > 0 else "-"
        inner_terms.append(f"{sign} {_fmt(abs(b), sig)}")
    arg = " ".join(inner_terms)

    core_str = f"(exp({arg}) - 1)"
    return amp, core_str


def _leaf_to_repr(atom, leaf_mod, tol=1e-10, sig: int = 6):
    """
    Structured version of leaf printing: return (scale, core_str).
    """
    core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
    var_idxs = tuple(int(i) for i in atom.var_idxs)
    kind = str(getattr(atom, "kind", "")).lower()

    # DE/PDE feature atoms are symbolic providers (u, du, d2u, ...).
    # Keep them as explicit symbolic leaves instead of inspecting leaf core types.
    if kind in ("u", "du", "d2u", "field", "state", "d1u", "ddu", "hess_u", "grad_u"):
        return 1.0, repr(atom)

    # Handle 0D scalar leaves early — they have no inputs to extract.  Named
    # free/fixed constants must remain symbols so dimensional identity survives
    # the pretty-print -> SymPy boundary.  Anonymous Scale leaves stay numeric.
    if isinstance(core, (FreeConstLeaf, FixedConstLeaf)):
        if kind in {
            "free_const",
            "freeconst",
            "free_constant",
            "fixed_const",
            "fixedconst",
            "fixed_constant",
        }:
            return 1.0, named_coefficient_symbol(atom)
        val = float(core.value.detach().cpu())
        return val, "1"

    # Unified compound extraction (replaces repeated kwargs.get("input_expr") calls)
    # Guard: 0D atoms have no inputs, so compound_input_expr would crash.
    if var_idxs:
        _ie = compound_input_expr(atom)
        _evi = list(extra_input_var_idxs(atom)) or None
        from nestynet_sr.sr_core.bridges import extra_input_nodes as _ein_fn
        _en = list(_ein_fn(atom)) or None
    else:
        _ie = None
        _evi = None
        _en = None

    if isinstance(core, (PolyLeaf, RPolyLeaf)):
        return _poly_leaf_repr(core, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol, sig=sig)
    if isinstance(core, (PolyLogLeaf, RPolyLogLeaf)):
        # Represent as an explicit polynomial in log(x_j).
        if isinstance(core, RPolyLogLeaf):
            exps = core.exps_full
            coeffs = core.full_coeffs()
        else:
            exps = core.exps
            coeffs = core.coeffs
        core_str = _polylog_str(
            exps,
            coeffs,
            var_idxs,
            input_expr=_ie,
            extra_var_idxs=_evi,
            extra_nodes=_en,
            tol=tol,
            sig=sig,
        )
        return 1.0, core_str
    if isinstance(core, LogShiftedLeaf):
        # Shifted logarithm: amp * log(x - shift) + offset
        amp = float(core.amp.item())
        shift = float(core.shift.item())
        offset = float(core.offset.item())
        var = _input_expr_to_str(_ie)
        parts = []
        if abs(amp - 1.0) > tol:
            parts.append(f"{amp:.{sig}g}*")
        parts.append(f"log({var}")
        if abs(shift) > tol:
            if shift > 0:
                parts.append(f" - {shift:.{sig}g}")
            else:
                parts.append(f" + {-shift:.{sig}g}")
        parts.append(")")
        if abs(offset) > tol:
            if offset > 0:
                parts.append(f" + {offset:.{sig}g}")
            else:
                parts.append(f" - {-offset:.{sig}g}")
        return 1.0, "".join(parts)
    if isinstance(core, (RationalPolyLeaf, RRationalPolyLeaf)):
        return _ratpoly_leaf_repr(core, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol, sig=sig)
    if isinstance(core, SinLinearLeaf):
        return _sin_leaf_repr(core, var_idxs, input_expr=_ie, tol=tol, sig=sig)
    if isinstance(core, TanhLinearLeaf):
        return _tanh_leaf_repr(core, var_idxs, input_expr=_ie, tol=tol, sig=sig)
    if isinstance(core, PowerLeaf):
        return _power_leaf_repr(core, var_idxs, input_expr=_ie, tol=tol, sig=sig)
    if isinstance(core, RInverseMonomialLeaf):
        return _rinv_monomial_leaf_repr(core, var_idxs, input_expr=_ie, tol=tol)
    if isinstance(core, InverseMonomialLeaf):
        return _inv_monomial_leaf_repr(core, var_idxs, input_expr=_ie, tol=tol, sig=sig)
    if isinstance(core, (ExpPolyLeaf, RExpPolyLeaf)):
        return _exp_poly_leaf_repr(core, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol, sig=sig)
    if isinstance(core, ExpRationalPolyLeaf):
        return _exp_ratpoly_leaf_repr(core, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol, sig=sig)
    if isinstance(core, (PlanckLeaf, PlanckFullLeaf)):
        return _planck_leaf_repr(core, var_idxs, input_expr=_ie, tol=tol, sig=sig)
    if isinstance(core, Expm1Leaf):
        return _expm1_leaf_repr(core, var_idxs, input_expr=_ie, tol=tol, sig=sig)
    if isinstance(core, (RatioPolyLeaf, RRatioPolyLeaf)):
        return _ratio_poly_leaf_repr(core, var_idxs, tol, sig=sig)
    if isinstance(core, VarLeaf):
        if _ie is not None:
            return 1.0, _input_expr_to_str(_ie)
        if len(var_idxs) == 1:
            return 1.0, f"x{int(var_idxs[0])}"
        if var_idxs:
            return 1.0, ", ".join(f"x{j}" for j in var_idxs)
        return 1.0, "x"

    # Handle NN atoms (neural network leaves that can't be analytically expanded)
    # Check atom.kind instead of core type since NN atoms are wrapped in adaptors
    if str(atom.kind).lower() == "nn":
        # Return generic representation: NN[tag](x0, x1, ...)
        # This allows SymPy to treat it as a symbolic function
        tag = atom_problem_label(atom) or (atom.tag if atom.tag else "NN")

        # Use input expressions so compound/wrapped inputs render correctly.
        from nestynet_sr.sr_core.bridges import get_input_exprs, has_nontrivial_input
        if has_nontrivial_input(atom):
            inputs = get_input_exprs(atom)
            parts = [_input_expr_to_str(inp) for inp in inputs]
            return 1.0, f"{tag}({', '.join(parts)})"
        vars_str = ", ".join(f"x{j}" for j in var_idxs)
        return 1.0, f"{tag}({vars_str})"

    # Fallback: core type not recognized
    # This can happen when leaf modules are wrapped or have unexpected structure
    tag = atom.tag if atom.tag else atom.kind
    if os.environ.get("NESTYNET_SR_DEBUG_LEAF_REPR"):
        print(
            f"[DEBUG] _leaf_to_repr: unrecognized core type for atom {tag}(vars={var_idxs})",
            file=sys.stderr,
        )
        print(f"[DEBUG]   leaf_mod type: {type(leaf_mod).__name__}", file=sys.stderr)
        print(f"[DEBUG]   core type: {type(core).__name__}", file=sys.stderr)
        print(f"[DEBUG]   atom.kind: {atom.kind}", file=sys.stderr)

    return 1.0, _leaf_to_str(atom, leaf_mod, tol=None, sig=sig)


def _leaf_to_str(atom, leaf_mod, tol=1e-10, sig: int = 6):
    """
    Legacy single-string leaf printer, used as a fallback by _leaf_to_repr.
    """
    core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
    var_idxs = tuple(int(i) for i in atom.var_idxs)
    kind = str(getattr(atom, "kind", "")).lower()

    # Handle 0D scalar leaves early — no inputs to extract
    if isinstance(core, (FreeConstLeaf, FixedConstLeaf)):
        if kind in {
            "free_const",
            "freeconst",
            "free_constant",
            "fixed_const",
            "fixedconst",
            "fixed_constant",
        }:
            return named_coefficient_symbol(atom)
        val = float(core.value.detach().cpu())
        return f"{val:.{sig}g}"

    # Unified compound extraction (guard 0D atoms which have no inputs)
    if var_idxs:
        _ie = compound_input_expr(atom)
        _evi = list(extra_input_var_idxs(atom)) or None
        from nestynet_sr.sr_core.bridges import extra_input_nodes as _ein_fn
        _en = list(_ein_fn(atom)) or None
    else:
        _ie = None
        _evi = None
        _en = None
    if isinstance(core, (PolyLeaf, RPolyLeaf)):
        if isinstance(core, RPolyLeaf):
            exps = core.exps_full
            coeffs = core.full_coeffs()
        else:
            exps = core.exps
            coeffs = core.coeffs
        return _poly_str(exps, coeffs, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol or 1e-10, sig=sig)
    if isinstance(core, (PolyLogLeaf, RPolyLogLeaf)):
        if isinstance(core, RPolyLogLeaf):
            exps = core.exps_full
            coeffs = core.full_coeffs()
        else:
            exps = core.exps
            coeffs = core.coeffs
        return _polylog_str(
            exps,
            coeffs,
            var_idxs,
            input_expr=_ie,
            extra_var_idxs=_evi,
            extra_nodes=_en,
            tol=tol or 1e-10,
            sig=sig,
        )
    if isinstance(core, LogShiftedLeaf):
        amp = float(core.amp.item())
        shift = float(core.shift.item())
        offset = float(core.offset.item())
        var = _input_expr_to_str(_ie)
        tol_use = tol or 1e-10
        parts = []
        if abs(amp - 1.0) > tol_use:
            parts.append(f"{amp:.{sig}g}*")
        parts.append(f"log({var}")
        if abs(shift) > tol_use:
            if shift > 0:
                parts.append(f" - {shift:.{sig}g}")
            else:
                parts.append(f" + {-shift:.{sig}g}")
        parts.append(")")
        if abs(offset) > tol_use:
            if offset > 0:
                parts.append(f" + {offset:.{sig}g}")
            else:
                parts.append(f" - {-offset:.{sig}g}")
        return "".join(parts)
    if isinstance(core, (RationalPolyLeaf, RRationalPolyLeaf)):
        _, core_str = _ratpoly_leaf_repr(core, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol or 1e-10, sig=sig)
        return core_str
    if isinstance(core, (ExpPolyLeaf, RExpPolyLeaf)):
        inner = _poly_str(core.exps, core.coeffs, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol or 1e-10, sig=sig)
        return f"exp({inner})"
    if isinstance(core, ExpRationalPolyLeaf):
        num = _poly_str(core.exps_num, core.coeffs_num, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol or 1e-10, sig=sig)
        den = _poly_str(core.exps_den, core.coeffs_den, var_idxs, input_expr=_ie, extra_var_idxs=_evi, extra_nodes=_en, tol=tol or 1e-10, sig=sig)
        return f"exp(({num})/({den}))"
    # SinLinearLeaf and others: generic fallback
    from nestynet_sr.sr_core.bridges import get_input_exprs, has_nontrivial_input
    if has_nontrivial_input(atom):
        inputs = get_input_exprs(atom)
        parts = [_input_expr_to_str(inp) for inp in inputs]
        vars_str = ", ".join(parts)
    else:
        vars_str = ", ".join(f"x{j}" for j in var_idxs)
    tag = atom.tag if atom.tag is not None else atom.kind
    return f"{tag}({vars_str})"


def _robust_scale_product(scales) -> float:
    """
    Multiply many floats without intermediate under/overflow killing the result.

    This matters because Stage-B/LM can create huge/small per-factor scales
    (multiplicative gauge freedom) whose *total* product is well-behaved.
    Pairwise float multiplication can underflow to 0 (or overflow to inf)
    before later factors would bring it back.
    """
    sign = 1.0
    mant = 1.0
    exp2 = 0
    for s in scales:
        try:
            sf = float(s)
        except Exception:
            return float("nan")
        if sf == 0.0:
            return 0.0
        if not math.isfinite(sf):
            if math.isnan(sf):
                return sf
            # +/-inf: preserve sign and return inf
            sign *= -1.0 if sf < 0 else 1.0
            return sign * float("inf")
        if sf < 0.0:
            sign *= -1.0
            sf = -sf
        m, e = math.frexp(sf)  # sf = m * 2**e, m in [0.5,1)
        mant *= m
        exp2 += int(e)
        # Renormalize mant to keep it in a safe range
        mant, e2 = math.frexp(mant)
        exp2 += int(e2)
    try:
        return sign * math.ldexp(mant, exp2)
    except OverflowError:
        return sign * float("inf")


def _format_scaled(scale: float, core: str, tol_scale: float = 1e-6, sig: int = 6) -> str:
    """
    Turn (scale, core) into a single human-readable string.
    """
    core = core.strip()
    if core in ("", "1"):
        return _fmt(scale, sig)

    # If the extracted scale is *exactly* zero, don't print "0*(...)"
    if scale == 0.0:
        return "0"

    # IMPORTANT: do not prune based solely on a tiny extracted scale when core is nontrivial.
    # Multiplicative gauge freedom can make leaf scales arbitrarily small while other leaves
    # carry compensating large constants, so dropping to "0" here can delete real structure
    # (e.g. trig products after counterterm/trig rewrites).

    # Drop scale if it's ~±1
    if abs(scale - 1.0) < tol_scale:
        return core
    if abs(scale + 1.0) < tol_scale:
        if core.startswith("(") and core.endswith(")"):
            return f"-{core}"
        else:
            return f"-{core}"

    s = _fmt(scale, sig)

    # Special-case exp(...) to avoid redundant parentheses
    if core.startswith("exp("):
        return f"{s}*{core}"

    # If core has + or -, wrap in parentheses
    if (" + " in core) or (" - " in core[1:] and not core.startswith("(")):
        return f"{s}*({core})"
    else:
        return f"{s}*{core}"


def _env_true(name: str) -> bool:
    v = os.getenv(name, "")
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _leaf_tag_of(leaf_mod):
    for attr in ("tag", "leaf_tag", "_tag", "name"):
        v = getattr(leaf_mod, attr, None)
        if isinstance(v, str) and v:
            return v
    core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", None))
    for attr in ("tag", "leaf_tag", "_tag", "name"):
        v = getattr(core, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def dump_state_parameters(state, *, sig: int = 17, max_terms: int = 32, file=None):
    if file is None:
        file = sys.stderr
    from nestynet_sr.sr_search.stageB import _collect_all_atoms

    atoms = _collect_all_atoms(state.root)
    leaves = list(getattr(state.model, "leaf", []))
    tag_to_leaf = {}
    for lf in leaves:
        t = _leaf_tag_of(lf)
        if t and t not in tag_to_leaf:
            tag_to_leaf[t] = lf

    print("==== SR PARAM DUMP BEGIN ====", file=file)
    print(f"n_atoms={len(atoms)} n_leaves={len(leaves)}", file=file)
    for k, atom in enumerate(atoms):
        t = getattr(atom, "tag", None)
        lf = tag_to_leaf.get(t, None) if isinstance(t, str) else None
        if lf is None and k < len(leaves):
            lf = leaves[k]
        core = getattr(lf, "core", getattr(lf, "model", lf)) if lf is not None else None
        print(
            f"[{k}] atom tag={t} kind={getattr(atom, 'kind', None)} var_idxs={tuple(getattr(atom, 'var_idxs', ()))}",
            file=file,
        )
        if lf is None:
            print("     leaf: <missing>", file=file)
            continue
        print(f"     leaf_mod={type(lf).__name__} core={type(core).__name__}", file=file)
        # If the leaf stores its own var indices, print them (catch mismatches!)
        for attr in ("var_idxs", "vars", "idxs"):
            v = getattr(lf, attr, None)
            if v is not None:
                print(f"     leaf_mod.{attr}={v}", file=file)
                break
        for attr in ("var_idxs", "vars", "idxs"):
            v = getattr(core, attr, None)
            if v is not None:
                print(f"     core.{attr}={v}", file=file)
                break

        if isinstance(core, (PolyLeaf, RPolyLeaf)):
            coeffs = core.coeffs.detach().cpu()
            exps = core.exps.detach().cpu()
            dt = coeffs.dtype
            tiny = torch.finfo(dt).tiny if dt.is_floating_point else None
            print(
                f"     PolyLeaf degree={getattr(core, 'degree', None)} coeffs.dtype={dt} tiny={tiny}",
                file=file,
            )
            if coeffs.numel() == 1:
                c0 = coeffs.view(-1)[0].item()
                print(f"     coeff[0]={c0!r}  (fmt={_fmt(c0, sig)})", file=file)
            else:
                n = min(int(coeffs.shape[0]), int(max_terms))
                for i in range(n):
                    c = float(coeffs[i].item())
                    e = tuple(int(x) for x in exps[i].tolist())
                    print(f"     term[{i}] exp={e} coeff={c!r} fmt={_fmt(c, sig)}", file=file)
                if int(coeffs.shape[0]) > n:
                    print(f"     ... ({int(coeffs.shape[0]) - n} more terms)", file=file)
        elif isinstance(core, SinLinearLeaf):
            w = float(core.weight.detach().cpu().view(-1)[0])
            b = float(core.bias.detach().cpu())
            a = float(core.amp.detach().cpu())
            print(f"     SinLinearLeaf amp={a!r} w={w!r} b={b!r}", file=file)
        elif isinstance(core, PowerLeaf):
            a = float(core.amp.detach().cpu())
            p = float(core.exponent.detach().cpu())
            print(f"     PowerLeaf amp={a!r} exponent={p!r}", file=file)
        elif isinstance(core, (ExpPolyLeaf, RExpPolyLeaf)):
            # Print constant term magnitude (common source of hidden scaling)
            coeffs = core.coeffs.detach().cpu()
            exps = core.exps.detach().cpu()
            c0 = None
            for i in range(exps.shape[0]):
                if int(exps[i].sum().item()) == 0:
                    c0 = float(coeffs[i].item())
                    break
            print(f"     ExpPolyLeaf const_in_exponent={c0!r}", file=file)
        print("", file=file)
    print("==== SR PARAM DUMP END ====", file=file)


def pretty_print_state(state, tol=1e-10, sig: int = 6):
    """
    New pretty-printer: factor numeric scales out of simple leaves, flatten
    products, and print a compact expression.

    Note: Requires _collect_all_atoms from stageB module.
    """
    # Import here to avoid circular dependency
    from nestynet_sr.sr_search.stageB import _collect_all_atoms
    from nestynet_sr.sr_search.stageB.atom_mapping import build_atom_to_leaf_map

    if _env_true("SR_DUMP_PARAMS"):
        try:
            dump_state_parameters(state, sig=max(sig, 17), file=sys.stderr)
        except Exception as e:
            print(f"[WARNING] dump_state_parameters failed: {e}", file=sys.stderr)

    atoms = _collect_all_atoms(state.root)
    leaves = list(state.model.leaf)

    # Check for atom/leaf count mismatch
    if len(atoms) != len(leaves):
        print(
            f"[WARNING] Atom/leaf count mismatch: {len(atoms)} atoms vs {len(leaves)} leaves",
            file=sys.stderr,
        )
        print(
            "[WARNING] This may cause some atoms to print as undefined functions", file=sys.stderr
        )
        print(f"[WARNING] Atoms: {[f'{a.kind}({a.tag})' for a in atoms]}", file=sys.stderr)

    # Use authoritative DFS-order mapping (same order as model construction)
    atom_to_leaf = build_atom_to_leaf_map(state.root, state.model)

    # Diagnostic: check for kind/tag mismatches between atoms and their assigned leaves
    for atom in atoms:
        leaf = atom_to_leaf.get(id(atom))
        if leaf is None:
            continue
        core = getattr(leaf, "core", getattr(leaf, "model", leaf))
        atom_kind = str(atom.kind).lower()
        # Expected core types for each atom kind
        if atom_kind == "poly" and not isinstance(core, PolyLeaf):
            print(
                f"[DEBUG] Atom kind mismatch: atom={atom.kind}#{atom.tag}, core type={type(core).__name__}",
                file=sys.stderr,
            )

    def rec(node):
        # Returns (scale, core_str)
        if isinstance(node, AddNode):
            s1, c1 = rec(node.left)
            s2, c2 = rec(node.right)
            t1 = _format_scaled(s1, c1, sig=sig)
            t2 = _format_scaled(s2, c2, sig=sig)
            return 1.0, f"{t1} + {t2}"

        if isinstance(node, MulNode):
            # Flatten the multiplication chain so we can multiply all scale factors robustly.
            scales = []
            factors = []

            def collect(n):
                if isinstance(n, MulNode):
                    collect(n.left)
                    collect(n.right)
                    return
                s, c = rec(n)
                scales.append(s)
                if c not in ("", "1"):
                    cc = c.strip()
                    if (" + " in cc) or (" - " in cc[1:] and not cc.startswith("(")):
                        cc = f"({cc})"
                    factors.append(cc)

            collect(node)
            scale = _robust_scale_product(scales)
            core = " * ".join(factors) if factors else "1"
            # If final scale is exactly zero, nuke the nontrivial core to avoid "0*(cos(...))"
            if scale == 0.0:
                core = "1"
            return scale, core

        if isinstance(node, PowNode):
            # Wrap the entire child expression in a power.
            s, c = rec(node.base)
            inner = _format_scaled(s, c, sig=sig)
            exp = float(node.exponent)
            if abs(exp - 0.5) < 1e-8:
                return 1.0, f"sqrt({inner})"
            if abs(exp + 0.5) < 1e-8:
                return 1.0, f"1/sqrt({inner})"
            if abs(exp - (-1.0)) < 1e-8:
                return 1.0, f"1/({inner})"
            return 1.0, f"({inner})**{_fmt(exp, sig)}"

        if isinstance(node, LogNode):
            s, c = rec(node.arg)
            inner = _format_scaled(s, c, sig=sig)
            return 1.0, f"log({inner})"

        if isinstance(node, ExpNode):
            s, c = rec(node.arg)
            inner = _format_scaled(s, c, sig=sig)
            return 1.0, f"exp({inner})"

        if isinstance(node, SinNode):
            s, c = rec(node.arg)
            inner = _format_scaled(s, c, sig=sig)
            return 1.0, f"sin({inner})"

        if isinstance(node, CosNode):
            s, c = rec(node.arg)
            inner = _format_scaled(s, c, sig=sig)
            return 1.0, f"cos({inner})"

        if isinstance(node, AsinNode):
            s, c = rec(node.arg)
            inner = _format_scaled(s, c, sig=sig)
            return 1.0, f"asin({inner})"

        if isinstance(node, AcosNode):
            s, c = rec(node.arg)
            inner = _format_scaled(s, c, sig=sig)
            return 1.0, f"acos({inner})"

        if isinstance(node, AtanNode):
            s, c = rec(node.arg)
            inner = _format_scaled(s, c, sig=sig)
            return 1.0, f"atan({inner})"

        if isinstance(node, AtomNode):
            # Handle var atoms directly - they're just variable references, not leaves
            kind = str(node.kind).lower()
            if kind in ("var", "x", "input"):
                if len(node.var_idxs) == 1:
                    return 1.0, f"x{int(node.var_idxs[0])}"
                vars_str = ", ".join(f"x{j}" for j in node.var_idxs)
                return 1.0, vars_str

            leaf = atom_to_leaf.get(id(node), None)
            if leaf is None:
                from nestynet_sr.sr_core.bridges import get_input_exprs, has_nontrivial_input
                inputs = get_input_exprs(node)
                if inputs and has_nontrivial_input(node):
                    parts = [_input_expr_to_str(inp) for inp in inputs]
                    return 1.0, f"{node.kind}({', '.join(parts)})"
                vars_str = ", ".join(f"x{j}" for j in node.var_idxs)
                return 1.0, f"{node.kind}({vars_str})"
            return _leaf_to_repr(node, leaf, tol, sig=sig)

        if isinstance(node, ConstNode):
            return 1.0, format_const_value(node.value)

        from nestynet_sr.sr_core.bridges import AbsNode

        if isinstance(node, AbsNode):
            _scale, inner = rec(node.arg)
            inner = _format_scaled(_scale, inner, sig=sig)
            return 1.0, f"Abs({inner})"

        raise TypeError(f"Unexpected node type {type(node)}")

    scale, core = rec(state.root)
    expr = _format_scaled(scale, core, sig=sig)
    return expr


def _infer_sympy_symbol_assumptions_from_samples(x_col: np.ndarray) -> Dict[str, bool]:
    """Infer conservative Symbol assumptions from observed data for one variable."""
    assumptions: Dict[str, bool] = {"real": True}
    try:
        col = np.asarray(x_col, dtype=float).ravel()
    except Exception:
        return assumptions
    if col.size == 0:
        return assumptions
    finite = np.isfinite(col)
    if not bool(np.any(finite)):
        return assumptions
    col = col[finite]
    if col.size == 0:
        return assumptions
    lo = float(np.min(col))
    hi = float(np.max(col))
    scale = max(1.0, abs(lo), abs(hi))
    eps = 1.0e-12 * scale
    if lo > eps:
        assumptions["positive"] = True
    elif lo >= -eps:
        assumptions["nonnegative"] = True
    elif hi < -eps:
        assumptions["negative"] = True
    elif hi <= eps:
        assumptions["nonpositive"] = True
    return assumptions


def _sympy_assumption_label(assumptions: Dict[str, bool]) -> str:
    if assumptions.get("positive", False):
        return ">0"
    if assumptions.get("nonnegative", False):
        return ">=0"
    if assumptions.get("negative", False):
        return "<0"
    if assumptions.get("nonpositive", False):
        return "<=0"
    return "real"


def _build_sympy_input_symbols_from_data(
    xs_np: np.ndarray,
    Nxvars: int,
) -> Tuple[list, Dict[str, Any], Dict[str, str]]:
    """Build x-symbols for SymPy parsing/eval with data-driven sign assumptions."""
    x_syms = []
    locals_map: Dict[str, Any] = {}
    domain_labels: Dict[str, str] = {}
    n_cols = int(xs_np.shape[1]) if getattr(xs_np, "ndim", 0) >= 2 else 0
    for j in range(int(Nxvars)):
        name = f"x{j}"
        if j < n_cols:
            assumptions = _infer_sympy_symbol_assumptions_from_samples(xs_np[:, j])
        else:
            assumptions = {"real": True}
        sym = sp.Symbol(name, **assumptions)
        x_syms.append(sym)
        locals_map[name] = sym
        domain_labels[name] = _sympy_assumption_label(assumptions)
    return x_syms, locals_map, domain_labels


def _abs_node_count(expr_try) -> int:
    """Count Abs nodes with multiplicity."""
    if not _HAVE_SYMPY:
        return 0
    try:
        return int(
            sum(
                1
                for node in sp.preorder_traversal(expr_try)
                if getattr(node, "func", None) == sp.Abs
            )
        )
    except Exception:
        try:
            return int(len(expr_try.atoms(sp.Abs)))
        except Exception:
            return 0


def _unit_consistent_additive_prunings(
    expr,
    variable_names,
    units_spec,
    *,
    requested_count: int = 8,
    max_attempts: int = 128,
    max_depth: int = 3,
    excluded_keys=None,
):
    """Search additive term removals until enough unit-valid repairs are found.

    This is deliberately a small, bounded Stage-C repair operator. It turns a
    dimensional failure into candidate-generation guidance (not merely a veto),
    while the ordinary numeric equivalence gate still decides whether any
    repair is faithful enough to emit as the final expression.
    """

    requested = max(0, int(requested_count))
    attempt_limit = max(0, int(max_attempts))
    depth_limit = max(0, int(max_depth))
    proposals = []
    attempted = 0
    unit_rejected = 0
    deduplicated = 0
    truncated_by_attempt_budget = False
    emission_keys = set(excluded_keys or ())
    queue = [(expr, 0, "")]
    try:
        seen = {sp.srepr(expr)}
    except Exception:
        seen = {str(expr)}

    stop_search = False
    while queue and len(proposals) < requested and not stop_search:
        current, depth, trace = queue.pop(0)
        if depth >= depth_limit:
            continue
        add_nodes = [
            node
            for node in sp.preorder_traversal(current)
            if isinstance(node, sp.Add) and len(node.args) >= 2
        ]
        for add_index, add_node in enumerate(add_nodes):
            args = list(sp.Add.make_args(add_node))
            for drop_index in range(len(args)):
                if len(proposals) >= requested:
                    stop_search = True
                    break
                replacement = sp.Add(*(args[:drop_index] + args[drop_index + 1 :]))
                proposal = current.xreplace({add_node: replacement})
                try:
                    key = sp.srepr(proposal)
                except Exception:
                    key = str(proposal)
                if key in seen:
                    continue
                if attempted >= attempt_limit:
                    # We found a concrete, previously unseen proposal that the
                    # attempt cap prevented us from checking.  This is budget
                    # exhaustion even when the BFS queue itself is empty (for
                    # example, when an unvisited sibling is the next proposal).
                    truncated_by_attempt_budget = True
                    stop_search = True
                    break
                seen.add(key)
                attempted += 1
                step = f"a{add_index}:drop{drop_index}"
                proposal_trace = f"{trace}/{step}" if trace else step
                result = check_sympy_units(
                    proposal,
                    variable_names,
                    units_spec,
                    expression_space="phi",
                )
                if result.checked and result.ok:
                    # Normalize only admissible emissions. Invalid intermediate
                    # states remain cheap to explore, while emitted repairs get
                    # a compact form for downstream complexity ranking.
                    try:
                        with sympy_timeout(
                            "unit_addend_prune_cancel", max_seconds=1
                        ):
                            proposal = sp.cancel(proposal)
                    except (RecursionError, SympyTimeoutError):
                        pass
                    result = check_sympy_units(
                        proposal,
                        variable_names,
                        units_spec,
                        expression_space="phi",
                    )
                    if result.checked and result.ok:
                        try:
                            proposal = _canonicalize_inverse_ratio_powers(proposal)
                        except Exception:
                            pass
                        try:
                            proposal = canonicalize_trig_phases(
                                proposal,
                                snap_rel_tol=1.0e-4,
                            )
                        except Exception:
                            pass
                        result = check_sympy_units(
                            proposal,
                            variable_names,
                            units_spec,
                            expression_space="phi",
                        )
                        try:
                            emission_key = sp.srepr(proposal)
                        except Exception:
                            emission_key = str(proposal)
                        if not (result.checked and result.ok):
                            unit_rejected += 1
                        elif emission_key in emission_keys:
                            deduplicated += 1
                        else:
                            emission_keys.add(emission_key)
                            proposals.append(
                                (proposal_trace, proposal, result.to_dict())
                            )
                    else:
                        unit_rejected += 1
                else:
                    unit_rejected += 1
                if depth + 1 < depth_limit:
                    queue.append((proposal, depth + 1, proposal_trace))
            if stop_search or len(proposals) >= requested:
                break

    if len(proposals) >= requested:
        exhausted = False
        exhaustion_reason = None
    elif truncated_by_attempt_budget:
        exhausted = True
        exhaustion_reason = "attempt_budget_exhausted"
    else:
        exhausted = True
        exhaustion_reason = "candidate_space_exhausted"
    return proposals, {
        "requested_count": requested,
        "raw_attempted": attempted,
        "unit_rejected": unit_rejected,
        "deduplicated": deduplicated,
        "emitted": len(proposals),
        "exhausted": exhausted,
        "exhaustion_reason": exhaustion_reason,
        "truncated_by_attempt_budget": truncated_by_attempt_budget,
        "max_attempts": attempt_limit,
        "max_depth": depth_limit,
    }


def _sympy_simplify_expression(
    expr_str: str,
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    Nxvars: int,
    y_op_inv=None,
    max_points: int = 2048,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-10,
    noise_floor_raw: Optional[float] = None,
    noise_abs_tol_factor: float = 0.25,
    prefer_stable_trig: bool = True,
    prune_trig_poly_args: bool = True,
    linearize_leaves: bool = True,
    units_spec=None,
    coefficient_metadata=None,
    verbose: bool = True,
    precomputed_xs_np: Optional[np.ndarray] = None,
    precomputed_ys_model: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """
    Optional SymPy-based post-simplification.

    Parameters
    ----------
    expr_str : str
        φ(y)-space expression produced by pretty_print_state.
    model : torch.nn.Module
        Trained Stage-B model (in φ(y)-space).
    val_loader :
        Validation dataloader used to numerically check that any symbolic
        simplification leaves the function unchanged to within tolerance.
    device : torch.device
    Nxvars : int
        Number of input variables x0..x{Nxvars-1} used to build lambdified
        SymPy callables.
    y_op_inv :
        Optional inverse y-transform; if provided, we also build a SymPy
        expression in the original y-space.
    units_spec : UnitsSpec, optional
        When supplied, candidates must satisfy the dimensional target in
        φ(y)-space before they may participate in ranking.
    coefficient_metadata : mapping, optional
        Versioned coefficient identity/value metadata. Named coefficients stay
        symbolic in returned expressions and are substituted only while
        checking numerical fidelity.

    Returns
    -------
    (phi_expr_str, y_expr_str, metadata) where:
        - phi_expr_str: simplified SymPy string in φ(y)-space or None
        - y_expr_str: simplified SymPy string in y-space or None
        - metadata: dict with keys 'accepted', 'kind', 'max_err', 'tol', 'parse_success'
    """
    if not _HAVE_SYMPY:
        return None, None, None

    variable_names_expected = tuple(f"x{i}" for i in range(int(Nxvars)))
    try:
        coefficient_metadata_norm = normalize_coefficient_metadata(
            coefficient_metadata,
            variable_names=variable_names_expected,
            require_values=True,
            units_spec=units_spec,
        )
        coefficient_values = coefficient_symbol_values(
            coefficient_metadata_norm,
            variable_names=variable_names_expected,
            units_spec=units_spec,
        )
    except CoefficientMetadataError as exc:
        metadata = {
            "accepted": False,
            "parse_success": False,
            "numeric_fidelity_ok": False,
            "kind": "invalid_coefficient_metadata",
            "reason": exc.reason,
            "coefficient_metadata_error": {
                "code": exc.code,
                "reason": exc.reason,
            },
            "coefficient_metadata": coefficient_metadata,
        }
        return None, None, metadata

    coefficient_symbols_used: list[str] = []

    def _with_coefficient_metadata(metadata):
        out = dict(metadata or {})
        out["coefficient_metadata"] = coefficient_metadata_norm
        out["coefficient_symbols_available"] = sorted(coefficient_values)
        out["coefficient_symbols_used"] = list(coefficient_symbols_used)
        return out

    if verbose:
        print("[Stage C] Attempting SymPy simplification of φ(y)...")

    # 0) Collect reference numeric values from the *fitted* Stage-B model
    #    (used both for equivalence gating and optional leaf-linearization).
    if precomputed_xs_np is not None and precomputed_ys_model is not None:
        xs_np = np.asarray(precomputed_xs_np, dtype=float)
        ys_model = np.asarray(precomputed_ys_model, dtype=float).reshape(-1)
    else:
        xs_np, ys_model = _collect_val_points_from_loader(
            model=model, val_loader=val_loader, device=device, max_points=max_points
        )
    if xs_np is None or ys_model is None:
        return None, None, _with_coefficient_metadata(
            {
                "accepted": False,
                "parse_success": True,
                "kind": "validation_points_unavailable",
                "reason": "could not collect validation points for Stage C",
            }
        )
    x_syms, x_sym_locals, x_domain_labels = _build_sympy_input_symbols_from_data(
        xs_np, Nxvars
    )
    if verbose:
        constrained = [
            f"{name}{domain}"
            for name, domain in x_domain_labels.items()
            if domain in (">0", ">=0", "<0", "<=0")
        ]
        if constrained:
            print(
                "[Stage C] SymPy variable assumptions from data: "
                + ", ".join(constrained)
            )

    # 1) Parse string into a SymPy expression.
    try:
        # '^' is used for powers in pretty_print_state; SymPy understands
        # '**' natively, so we normalise first.
        expr_for_sympy = expr_str.replace("^", "**")
        local_dict = {
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "pi": sp.pi,
            "E": sp.E,
        }
        local_dict.update(x_sym_locals)
        local_dict.update(
            {name: sp.Symbol(name, real=True) for name in coefficient_values}
        )
        try:
            phi_expr = sp.sympify(expr_for_sympy, locals=local_dict)
        except RecursionError:
            # sympify can hit recursion limit on deeply nested expressions
            if verbose:
                print("[Stage C] SymPy sympify hit recursion limit; cannot simplify")
            metadata = {
                "accepted": False,
                "parse_success": False,
                "max_err": None,
                "tol": None,
                "kind": None,
                "error": "recursion",
            }
            return None, None, _with_coefficient_metadata(metadata)

        unexpected_symbols = sorted(
            {
                str(symbol)
                for symbol in phi_expr.free_symbols
                if str(symbol)
                not in set(variable_names_expected) | set(coefficient_values)
            }
        )
        if unexpected_symbols:
            metadata = {
                "accepted": False,
                "parse_success": True,
                "numeric_fidelity_ok": False,
                "kind": "coefficient_value_missing",
                "reason": (
                    "symbolic expression contains symbols with no coefficient "
                    "value metadata: " + ", ".join(unexpected_symbols)
                ),
                "missing_symbols": unexpected_symbols,
            }
            return None, None, _with_coefficient_metadata(metadata)
        coefficient_symbols_used = sorted(
            str(symbol)
            for symbol in phi_expr.free_symbols
            if str(symbol) in coefficient_values
        )

        # Preserve the literal pretty-print parse as the numerical baseline.
        # Constant snapping below is a proposal, not part of that baseline.
        phi_expr_raw = phi_expr

        # Lightweight cleanup: combine cancelling constants and drop tiny DC offsets
        # (tolerance is tied to the numeric equivalence gate)
        scale = max(1.0, float(np.median(np.abs(ys_model))) if ys_model is not None else 1.0)
        try:
            noise_rms = math.sqrt(max(0.0, float(noise_floor_raw or 0.0)))
        except Exception:
            noise_rms = 0.0
        noise_abs_tol = float(max(0.0, float(noise_abs_tol_factor))) * float(noise_rms)
        tol_eff = max(abs_tol * scale, rel_tol * scale, noise_abs_tol)
        snap_rel_tol_eff = max(
            1.0e-4,
            min(2.0e-2, 5.0 * noise_rms / max(float(scale), 1.0e-12)),
        )
        phi_expr = _prune_tiny_additive_constants(phi_expr, tol=tol_eff)
        phi_expr = _canonicalize_inverse_ratio_powers(phi_expr)
        # Trigonometric phases have their own direct rational-pi proposal.
        # Running the general symbolic-constant snap first can turn a numeric
        # phase into an unrelated expression such as pi**3, after which a
        # second phase snap may land on the wrong equivalence class.
        phi_expr = canonicalize_trig_phases(phi_expr, snap_rel_tol=snap_rel_tol_eff)

    except RecursionError as e_rec:
        if verbose:
            print("[Stage C] SymPy processing hit recursion limit:", e_rec)
        metadata = {
            "accepted": False,
            "parse_success": False,
            "max_err": None,
            "tol": None,
            "kind": None,
            "error": "recursion",
        }
        return None, None, _with_coefficient_metadata(metadata)
    except Exception as e:
        if verbose:
            print("[Stage C] SymPy sympify failed:", e)
        metadata = {
            "accepted": False,
            "parse_success": False,
            "max_err": None,
            "tol": None,
            "kind": None,
        }
        return None, None, _with_coefficient_metadata(metadata)

    # Check for undefined function symbols (like leaf0_L_L, leaf0_R, etc.)
    # These indicate that some atoms weren't properly expanded to analytic forms.
    # SymPy's aggressive simplification can hit recursion limits on undefined functions.
    undefined_funcs = set()
    try:
        allowed = {
            sp.sin,
            sp.cos,
            sp.tan,
            sp.tanh,
            sp.exp,
            sp.log,
            sp.sqrt,
            sp.asin,
            sp.acos,
            sp.atan,
            sp.atan2,
            sp.Abs,
        }
        for fcall in phi_expr.atoms(sp.Function):
            f = fcall.func
            if f in allowed:
                continue
            fname = getattr(f, "__name__", None) or str(f)
            if fname.startswith("leaf"):
                continue
            undefined_funcs.add(fname)
    except Exception:
        pass

    if undefined_funcs:
        if verbose:
            funcs_str = ", ".join(sorted(undefined_funcs))
            print(
                f"[Stage C] Cannot simplify: expression contains undefined functions [{funcs_str}]"
            )
            print(
                "[Stage C] Hint: Some polynomial atoms weren't expanded - check _leaf_to_repr logic"
            )
        metadata = {
            "accepted": False,
            "parse_success": True,
            "max_err": None,
            "tol": None,
            "kind": "undefined_functions",
            "undefined": list(undefined_funcs),
        }
        return None, None, _with_coefficient_metadata(metadata)

    # Prepare leaf{i}(...) numeric callables for lambdify.
    def _make_leaf_callable(leaf_mod: torch.nn.Module, atom=None):
        """
        Create a numpy-callable wrapper for a leaf module.

        For compound atoms (with input_expr), evaluates z = input_expr(args)
        before calling the leaf, since compound leaves expect 1D input.
        """
        try:
            p0 = next(leaf_mod.parameters())
            dev = p0.device
            dt = p0.dtype
        except StopIteration:
            dev = device
            try:
                dt = next(model.parameters()).dtype
            except StopIteration:
                dt = torch.get_default_dtype()

        # Get input expression and var_idxs from atom
        input_expr = None
        var_idxs = None
        if atom is not None and atom.var_idxs:
            input_expr = compound_input_expr(atom)  # always non-None
            var_idxs = tuple(int(i) for i in atom.var_idxs)

        def _eval_input_expr_numpy(input_expr_node, args_dict):
            """Recursively evaluate input_expr AST with numpy arrays."""
            from nestynet_sr.sr_core.bridges import (
                AddNode,
                AtomNode,
                ConstNode,
                CosNode,
                ExpNode,
                LogNode,
                MulNode,
                PowNode,
                SinNode,
            )

            if isinstance(input_expr_node, AtomNode):
                kind = str(getattr(input_expr_node, "kind", "")).lower()
                if kind in ("var", "x", "input"):
                    idx = input_expr_node.var_idxs[0]
                    return args_dict[idx]
                else:
                    raise ValueError(f"Unsupported atom kind '{kind}' in input_expr")
            elif isinstance(input_expr_node, AddNode):
                left = _eval_input_expr_numpy(input_expr_node.left, args_dict)
                right = _eval_input_expr_numpy(input_expr_node.right, args_dict)
                return left + right
            elif isinstance(input_expr_node, MulNode):
                left = _eval_input_expr_numpy(input_expr_node.left, args_dict)
                right = _eval_input_expr_numpy(input_expr_node.right, args_dict)
                return left * right
            elif isinstance(input_expr_node, PowNode):
                base = _eval_input_expr_numpy(input_expr_node.base, args_dict)
                exp_val = input_expr_node.exponent
                return base ** exp_val
            elif isinstance(input_expr_node, ConstNode):
                return input_expr_node.value  # Return scalar, numpy will broadcast
            elif isinstance(input_expr_node, SinNode):
                return np.sin(_eval_input_expr_numpy(input_expr_node.arg, args_dict))
            elif isinstance(input_expr_node, CosNode):
                return np.cos(_eval_input_expr_numpy(input_expr_node.arg, args_dict))
            elif isinstance(input_expr_node, ExpNode):
                return np.exp(_eval_input_expr_numpy(input_expr_node.arg, args_dict))
            elif isinstance(input_expr_node, LogNode):
                return np.log(_eval_input_expr_numpy(input_expr_node.arg, args_dict))
            else:
                raise ValueError(f"Unsupported node type in input_expr: {type(input_expr_node)}")

        def _f(*args):
            # Guard against lambdify name collisions.
            # If an atom tag shadows a standard function (sin, cos, sqrt, etc.),
            # lambdify may route e.g. sin(u) to this callable. With broadcasting
            # this can silently return wrong numbers instead of raising.
            len(var_idxs) if var_idxs else len(args)
            if var_idxs is not None and len(args) != len(var_idxs):
                raise ValueError(
                    f"Leaf callable expected {len(var_idxs)} args for vars={var_idxs}, "
                    f"got {len(args)}. Possible lambdify name collision "
                    "(e.g. tag shadowing sin/cos/sqrt/exp/log)."
                )

            arrs = [np.asarray(a) for a in args]
            arrs = np.broadcast_arrays(*arrs)

            if input_expr is not None and var_idxs is not None:
                # Evaluate each input expression via numpy.
                # For trivial Var(i) inputs this returns args_dict[i].
                args_dict = {idx: arrs[i] for i, idx in enumerate(var_idxs)}
                all_inputs = get_input_exprs(atom) if atom is not None else (input_expr,)
                cols = [_eval_input_expr_numpy(inp, args_dict).reshape(-1, 1) for inp in all_inputs]
                x2 = np.column_stack(cols) if len(cols) > 1 else cols[0]
            else:
                # Fallback when atom is None
                x = np.stack(arrs, axis=-1)
                x2 = x.reshape(-1, x.shape[-1])

            xt = torch.as_tensor(x2, device=dev, dtype=dt)
            with torch.no_grad():
                y = leaf_mod(xt)
                if y.dim() == 2:
                    y = y[:, 0]
                else:
                    y = y.view(-1)
            y_np = y.detach().cpu().numpy()
            return y_np.reshape(arrs[0].shape)

        return _f

    custom = {}
    leaves = getattr(model, "leaf", None) if model is not None else None
    if leaves is not None:
        leaves = list(leaves)
        # Try to get corresponding atoms to access their tags
        atoms = None
        if hasattr(model, "_collect_atoms") and hasattr(model, "ast_root"):
            try:
                atoms = model._collect_atoms(model.ast_root)
            except Exception:
                atoms = None
        for i, leaf in enumerate(leaves):
            atom = atoms[i] if atoms is not None and i < len(atoms) else None
            f = _make_leaf_callable(leaf, atom)
            custom[f"leaf{i}"] = f
            # Also register under the atom's tag if available (for pretty_print compatibility)
            if atoms is not None and i < len(atoms):
                tag = getattr(atoms[i], "tag", None)
                if isinstance(tag, str) and tag:
                    custom[tag] = f

    n_samples, n_cols = xs_np.shape
    args = []
    for j in range(Nxvars):
        if j < n_cols:
            args.append(xs_np[:, j])
        else:
            args.append(np.zeros(n_samples, dtype=xs_np.dtype))

    scale = float(np.median(np.abs(ys_model)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    try:
        noise_rms = math.sqrt(max(0.0, float(noise_floor_raw or 0.0)))
    except Exception:
        noise_rms = 0.0
    noise_abs_tol = float(max(0.0, float(noise_abs_tol_factor))) * float(noise_rms)
    tol_eff = max(abs_tol * scale, rel_tol * scale, noise_abs_tol)
    snap_rel_tol_eff = max(
        1.0e-4,
        min(2.0e-2, 5.0 * noise_rms / max(float(scale), 1.0e-12)),
    )

    def _eval_expr(expr_try, debug=False):
        try:
            substitutions = {
                symbol: sp.Float(repr(coefficient_values[str(symbol)]), 17)
                for symbol in expr_try.free_symbols
                if str(symbol) in coefficient_values
            }
            expr_numeric = expr_try.xreplace(substitutions)
            leftover_symbols = {
                str(symbol)
                for symbol in expr_numeric.free_symbols
                if str(symbol) not in set(variable_names_expected)
            }
            if leftover_symbols:
                raise ValueError(
                    "missing coefficient values for symbols: "
                    + ", ".join(sorted(leftover_symbols))
                )
            # NOTE: Put NumPy BEFORE custom mappings.
            # custom contains leaf callables keyed by leaf names/tags. If any tag
            # collides with a standard function name (sin, cos, sqrt, exp, log),
            # putting custom first causes lambdify to shadow NumPy's implementations,
            # producing wildly incorrect evaluations without necessarily raising.
            # NumPy first preserves correct semantics for built-ins while still
            # allowing leaf callables to resolve from custom.
            f = sp.lambdify(x_syms, expr_numeric, modules=["numpy", custom])
            y_try = np.array(f(*args), dtype=float).ravel()
            if y_try.shape != ys_model.shape:
                y_try = y_try.reshape(ys_model.shape)
            if debug:
                print(f"[DEBUG _eval_expr] y_try[:5]={y_try[:5]}", file=sys.stderr)
                print(f"[DEBUG _eval_expr] ys_model[:5]={ys_model[:5]}", file=sys.stderr)
            if not (np.isfinite(y_try).all() and np.isfinite(ys_model).all()):
                return None
            diff = y_try - ys_model
            return float(np.max(np.abs(diff)))
        except Exception as e:
            if debug:
                print(f"[DEBUG _eval_expr] Exception: {e}", file=sys.stderr)
            return None

    # Baseline: how far does the *pretty_print string itself* deviate from the fitted model?
    # (Because pretty_print_state uses ~6 sig figs, this is often ~1e-6.)
    base_err = None
    try:
        base_err = _eval_expr(phi_expr_raw, debug=False)
    except Exception:
        base_err = None
    if (base_err is None) or (not math.isfinite(base_err)):
        base_err = float("inf")
    if verbose and math.isfinite(base_err):
        print(
            f"[Stage C] baseline pretty_print max_err≈{base_err:.3g} vs model, "
            f"tol≈{tol_eff:.3g}"
        )

    if base_err > 1000.0 * tol_eff:
        # Large base_err could be (a) a genuine parse/eval bug, or (b) numerical
        # ill-conditioning from near-cancelling factors.  Try sp.cancel() as a
        # diagnostic: if it reduces the error, the expression is structurally
        # correct but ill-conditioned, and we should proceed with the cancelled
        # form rather than skipping SymPy entirely.
        repaired = False
        try:
            phi_cancelled = sp.cancel(phi_expr_raw)
            if phi_cancelled != phi_expr_raw:
                err2 = _eval_expr(phi_cancelled, debug=verbose)
                if err2 is not None and math.isfinite(err2) and err2 <= 1000.0 * tol_eff:
                    if verbose:
                        print(
                            f"[Stage C] sp.cancel() reduced base_err {base_err:.3g} -> {err2:.3g}; proceeding with cancelled form."
                        )
                    phi_expr_raw = phi_cancelled
                    base_err = err2
                    repaired = True
        except Exception:
            pass
        if not repaired:
            if verbose:
                print(
                    f"[Stage C] Pretty_print parse looks wrong (base_err={base_err:.3g} >> strict={tol_eff:.3g}); skipping SymPy."
                )
            metadata = {
                "accepted": False,
                "parse_success": True,
                "max_err": base_err,
                "tol": tol_eff,
                "kind": "bad_pretty_print",
            }
            return None, None, _with_coefficient_metadata(metadata)

    # Acceptance tolerance: never stricter than the baseline print error.
    tol_accept = 5.0 * tol_eff
    if math.isfinite(base_err):
        tol_accept = max(tol_accept, 2.0 * base_err)

    def _ops_score(expr_try):
        try:
            return int(sp.count_ops(expr_try, visual=False))
        except Exception:
            try:
                return len(sp.sstr(expr_try))
            except Exception:
                return len(str(expr_try))

    def _display_complexity(expr_try):
        try:
            ops = float(sp.count_ops(expr_try, visual=False))
        except Exception:
            try:
                ops = float(len(sp.sstr(expr_try)))
            except Exception:
                ops = float(len(str(expr_try)))
        try:
            const_cost, n_long = _constant_code_cost(
                expr_try,
                snap_targets=final_polish_snap_targets(),
                snap_rel_tol=1e-4,
            )
        except Exception:
            const_cost, n_long = 0.0, 0
        return ops + float(const_cost) + 4.0 * float(n_long)

    abs_penalty_weight = 20.0
    base_abs_nodes = _abs_node_count(phi_expr_raw)

    # 2) Build a small candidate set, then pick the simplest one that passes
    #    the numeric equivalence gate. This avoids "one aggressive pass ruins it".
    candidates = []
    seen = set()

    def _add_candidate(label, ex, *, normalize=True):
        if ex is None:
            return False
        if normalize:
            ex = _canonicalize_inverse_ratio_powers(ex)
            try:
                ex = canonicalize_trig_phases(ex, snap_rel_tol=snap_rel_tol_eff)
            except Exception:
                pass
        try:
            key = sp.srepr(ex)
        except Exception:
            key = str(ex)
        if key in seen:
            return False
        seen.add(key)
        candidates.append((label, ex))
        return True

    # Reset SymPy budget for this simplification pass (60 seconds total)
    reset_sympy_budget(60.0)

    try:
        base = aggressive_simplify(
            phi_expr, Nxvars, verbose=verbose, budget_seconds=get_sympy_budget_remaining()
        )
        if verbose:
            print(
                f"[Stage C] aggressive_simplify completed, budget remaining: {get_sympy_budget_remaining():.1f}s"
            )
    except SympyTimeoutError:
        if verbose:
            print("[Stage C] SymPy aggressive_simplify timed out (continuing with parsed expr)")
        base = phi_expr
    except Exception as e:
        if verbose:
            print("[Stage C] SymPy aggressive_simplify failed (continuing with parsed expr):", e)
        base = phi_expr

    # Always include the parsed pretty_print expression too (so we can fall back
    # to something that is "no worse than what we started with").
    # Apply float preprocessing to recognize constants like 1/√(2π).  The
    # tolerance is widened in noisy runs, but every proposed snap still has to
    # pass the numeric equivalence gate below.
    _add_candidate("pretty_print_raw", phi_expr_raw, normalize=False)
    try:
        with sympy_timeout("nsimplify_floats_parsed", max_seconds=5):
            ex0 = _nsimplify_floats(
                phi_expr,
                constants=AIF_CONSTS,
                tolerance=snap_rel_tol_eff,
            )
    except (RecursionError, SympyTimeoutError):
        ex0 = phi_expr
    if prefer_stable_trig:
        try:
            with sympy_timeout("stable_trig_parsed", max_seconds=5):
                ex0 = _prefer_stable_half_angle_trig(ex0)
        except (RecursionError, SympyTimeoutError):
            pass
    _add_candidate("parsed", ex0)

    # Baseline simplification (+ constants), then stabilise trig forms.
    try:
        with sympy_timeout("nsimplify_compat_base", max_seconds=10):
            base = _nsimplify_compat(
                base,
                constants=AIF_CONSTS,
                rational=True,
                maxsteps=80,
                tolerance=snap_rel_tol_eff,
            )
    except (RecursionError, SympyTimeoutError):
        pass  # Keep base as-is
    except Exception:
        pass
    try:
        with sympy_timeout("simplify_base", max_seconds=10):
            base = sp.simplify(base)
    except (RecursionError, SympyTimeoutError):
        pass  # Keep base as-is
    except Exception:
        pass
    if prefer_stable_trig:
        try:
            with sympy_timeout("stable_trig_base", max_seconds=5):
                base = _prefer_stable_half_angle_trig(base)
        except (RecursionError, SympyTimeoutError):
            pass  # Keep base as-is
    _add_candidate("sympy", base)

    # Generic symbolic-constant snapping: this catches cases such as
    # 0.07957747 -> 1/(4*pi) without relying on benchmark-specific templates.
    try:
        for label, ex in numeric_constant_snap_candidates(
            base,
            snap_targets=final_polish_snap_targets(),
            snap_rel_tol=snap_rel_tol_eff,
            per_number=4,
        ):
            _add_candidate(label, ex)
    except Exception:
        pass

    # Optional: clean tiny polynomial garbage inside trig args (common after LM).
    if prune_trig_poly_args:
        try:
            with sympy_timeout("cleanup_trig_poly_args", max_seconds=10):
                ex = _cleanup_trig_poly_args(base, rel_tol=snap_rel_tol_eff)
        except (RecursionError, SympyTimeoutError):
            ex = base  # Fall back to base
        try:
            with sympy_timeout("simplify_trig_prune", max_seconds=10):
                ex = sp.simplify(ex)
        except (RecursionError, SympyTimeoutError):
            pass  # Keep ex as-is
        except Exception:
            pass
        if prefer_stable_trig:
            try:
                with sympy_timeout("stable_trig_prune", max_seconds=5):
                    ex = _prefer_stable_half_angle_trig(ex)
            except (RecursionError, SympyTimeoutError):
                pass  # Keep ex as-is
        _add_candidate("trig_arg_prune", ex)

    # Recombine scattered exp() factors: exp(a)*exp(b) → exp(a+b)
    try:
        with sympy_timeout("powsimp_exp", max_seconds=5):
            ex_ps = sp.powsimp(base, combine='exp')
        _add_candidate("powsimp_exp", ex_ps)
    except (RecursionError, SympyTimeoutError):
        pass
    except Exception:
        pass

    # Optional: linearise leaf{i}(xj) calls when they're essentially linear.
    if linearize_leaves:
        try:
            with sympy_timeout("linearize_leaf_calls", max_seconds=15):
                ex = _linearize_leaf_calls(
                    base,
                    model=model,
                    xs_np=xs_np,
                    device=device,
                    max_rel_rmse=None,
                    verbose=verbose,
                )
        except (RecursionError, SympyTimeoutError):
            ex = base  # Fall back to base
        _add_candidate("leaf_linear", ex)
        if prune_trig_poly_args:
            try:
                with sympy_timeout("cleanup_trig_poly_args_2", max_seconds=10):
                    ex2 = _cleanup_trig_poly_args(ex, rel_tol=snap_rel_tol_eff)
            except (RecursionError, SympyTimeoutError):
                ex2 = ex
            try:
                with sympy_timeout("simplify_leaf_linear", max_seconds=10):
                    ex2 = sp.simplify(ex2)
            except (RecursionError, SympyTimeoutError):
                pass  # Keep ex2 as-is
            except Exception:
                pass
            if prefer_stable_trig:
                try:
                    with sympy_timeout("stable_trig_leaf_linear", max_seconds=5):
                        ex2 = _prefer_stable_half_angle_trig(ex2)
                except (RecursionError, SympyTimeoutError):
                    pass  # Keep ex2 as-is
            _add_candidate("leaf_linear+trig_arg_prune", ex2)

    variable_names = tuple(str(symbol) for symbol in x_syms)
    candidate_count_before_unit_guidance = len(candidates)
    unit_guided_generation = {
        "enabled": False,
        "requested_count": 0,
        "raw_attempted": 0,
        "unit_rejected": 0,
        "emitted": 0,
        "deduplicated": 0,
        "exhausted": False,
        "exhaustion_reason": "not_needed",
        "max_attempts": 128,
        "max_depth": 3,
    }
    if units_spec is not None:
        invalid_sources = []
        for source_label, source_expr in list(candidates):
            source_units = check_sympy_units(
                source_expr,
                variable_names,
                units_spec,
                expression_space="phi",
            )
            if source_units.checked and not source_units.ok:
                invalid_sources.append((source_label, source_expr))
        if invalid_sources:
            repair_requested = 8
            repair_attempt_limit = 128
            repair_attempted = 0
            repair_rejected = 0
            repair_emitted = 0
            repair_deduplicated = 0
            repair_sources_considered = 0
            repair_attempt_budget_exhausted = False
            for source_label, source_expr in invalid_sources:
                if repair_emitted >= repair_requested:
                    break
                if repair_attempted >= repair_attempt_limit:
                    repair_attempt_budget_exhausted = True
                    break
                repair_sources_considered += 1
                repairs, repair_stats = _unit_consistent_additive_prunings(
                    source_expr,
                    variable_names,
                    units_spec,
                    requested_count=repair_requested - repair_emitted,
                    max_attempts=repair_attempt_limit - repair_attempted,
                    max_depth=3,
                    excluded_keys=seen,
                )
                repair_attempted += int(repair_stats["raw_attempted"])
                repair_rejected += int(repair_stats["unit_rejected"])
                repair_deduplicated += int(repair_stats.get("deduplicated", 0))
                if repair_stats.get("exhaustion_reason") == "attempt_budget_exhausted":
                    repair_attempt_budget_exhausted = True
                for trace, repair_expr, _certificate in repairs:
                    if _add_candidate(
                        f"unit_addend_prune:{source_label}:{trace}", repair_expr
                    ):
                        repair_emitted += 1
                    else:
                        repair_deduplicated += 1
            if (
                repair_attempted >= repair_attempt_limit
                and repair_sources_considered < len(invalid_sources)
            ):
                repair_attempt_budget_exhausted = True
            if repair_emitted >= repair_requested:
                repair_exhausted = False
                repair_exhaustion_reason = None
            elif repair_attempt_budget_exhausted:
                repair_exhausted = True
                repair_exhaustion_reason = "attempt_budget_exhausted"
            else:
                repair_exhausted = True
                repair_exhaustion_reason = "candidate_space_exhausted"
            unit_guided_generation = {
                "enabled": True,
                "requested_count": repair_requested,
                "raw_attempted": repair_attempted,
                "unit_rejected": repair_rejected,
                "emitted": repair_emitted,
                "deduplicated": repair_deduplicated,
                "exhausted": repair_exhausted,
                "exhaustion_reason": repair_exhaustion_reason,
                "max_attempts": repair_attempt_limit,
                "max_depth": 3,
                "invalid_source_count": len(invalid_sources),
                "sources_considered": repair_sources_considered,
                "truncated_by_attempt_budget": repair_attempt_budget_exhausted,
            }

    raw_proposal_attempted = int(candidate_count_before_unit_guidance) + int(
        unit_guided_generation["raw_attempted"]
    )

    best = None
    best_label = None
    best_ops = None
    best_err = None
    best_abs_nodes = None
    best_complexity = None
    min_err = None
    min_err_label = None
    min_unit_valid_err = None
    min_unit_valid_err_label = None
    evaluated_candidate_count = 0
    numeric_pass_count = 0
    unit_reject_count = 0
    unit_valid_numeric_count = 0
    unit_rejections = []
    best_units_result = None
    for label, ex in candidates:
        units_result = check_sympy_units(
            ex,
            variable_names,
            units_spec,
            expression_space="phi",
        )
        if units_result.checked and not units_result.ok:
            unit_reject_count += 1
            if len(unit_rejections) < 12:
                unit_rejections.append(
                    {
                        "label": label,
                        "expr": sp.sstr(ex),
                        "unit_admissibility": units_result.to_dict(),
                    }
                )
        err = _eval_expr(ex)
        if err is None or (not math.isfinite(err)):
            continue
        evaluated_candidate_count += 1
        if min_err is None or err < min_err:
            min_err = err
            min_err_label = label
        if err > tol_accept:
            continue
        numeric_pass_count += 1
        if units_result.checked and not units_result.ok:
            continue
        unit_valid_numeric_count += 1
        if min_unit_valid_err is None or err < min_unit_valid_err:
            min_unit_valid_err = err
            min_unit_valid_err_label = label
        ops = _ops_score(ex)
        abs_nodes = _abs_node_count(ex)
        abs_extra = max(0, int(abs_nodes) - int(base_abs_nodes))
        complexity = _display_complexity(ex) + abs_penalty_weight * float(abs_extra)
        if (
            best is None
            or complexity < best_complexity
            or (
                complexity == best_complexity
                and (
                    ops < best_ops
                    or (
                        ops == best_ops
                        and (
                            err < best_err
                            or (
                                err == best_err
                                and len(sp.sstr(ex)) < len(sp.sstr(best))
                            )
                        )
                    )
                )
            )
        ):
            best = ex
            best_label = label
            best_ops = ops
            best_err = err
            best_abs_nodes = int(abs_nodes)
            best_complexity = float(complexity)
            best_units_result = units_result

    if best is None:
        units_checked = units_spec is not None
        no_unit_valid_candidate = bool(
            units_checked and numeric_pass_count > 0 and unit_valid_numeric_count == 0
        )
        failure_kind = (
            "no_unit_valid_candidate" if no_unit_valid_candidate else min_err_label
        )
        failure_reason = (
            "finite Stage-C candidate pool was exhausted without a candidate "
            "that passed both numeric fidelity and dimensional admissibility"
            if no_unit_valid_candidate
            else "finite Stage-C candidate pool was exhausted without a numerically faithful candidate"
        )
        if verbose:
            if no_unit_valid_candidate:
                print(
                    "[Stage C] SymPy simplification rejected: numerically faithful "
                    "candidates existed, but none was dimensionally admissible."
                )
            elif min_err is None:
                print(
                    "[Stage C] SymPy simplification rejected: no candidate evaluated successfully."
                )
            else:
                print(
                    "[Stage C] SymPy simplified expression rejected: "
                    f"best_max_err={min_err:.3g} (candidate={min_err_label}), tol≈{tol_accept:.3g} (strict≈{tol_eff:.3g})"
                )
        metadata = {
            "accepted": False,
            "parse_success": True,
            "max_err": min_err,
            "best_unit_valid_max_err": min_unit_valid_err,
            "best_unit_valid_error_candidate": min_unit_valid_err_label,
            "tol": tol_accept,
            "kind": failure_kind,
            "reason": failure_reason,
            "numeric_fidelity_ok": bool(numeric_pass_count > 0),
            "units_checked": bool(units_checked),
            "units_ok": False if no_unit_valid_candidate else None,
            "unit_admissibility": {
                "checked": bool(units_checked),
                "valid": False if no_unit_valid_candidate else None,
                "checker": "sympy_units_v1",
                "code": failure_kind,
                "reason": failure_reason,
                "expression_space": "phi",
            },
            "candidate_count": len(candidates),
            "candidate_count_before_unit_guidance": candidate_count_before_unit_guidance,
            "evaluated_candidate_count": evaluated_candidate_count,
            "numeric_pass_count": numeric_pass_count,
            "unit_reject_count": unit_reject_count,
            "unit_valid_numeric_count": unit_valid_numeric_count,
            "unit_rejections": unit_rejections,
            "unit_guided_generation": unit_guided_generation,
            "proposal_budget": {
                "requested_count": 1,
                "raw_attempted": raw_proposal_attempted,
                "unit_rejected": unit_reject_count
                + int(unit_guided_generation["unit_rejected"]),
                "emitted": 0,
                "exhausted": True,
                "exhaustion_reason": (
                    "attempt_budget_exhausted_no_unit_valid_numeric_candidate"
                    if no_unit_valid_candidate
                    and unit_guided_generation.get("exhaustion_reason")
                    == "attempt_budget_exhausted"
                    else "candidate_space_exhausted_no_unit_valid_numeric_candidate"
                    if no_unit_valid_candidate
                    else "attempt_budget_exhausted_no_numeric_candidate"
                    if unit_guided_generation.get("exhaustion_reason")
                    == "attempt_budget_exhausted"
                    else "candidate_space_exhausted_no_numeric_candidate"
                ),
            },
        }
        return None, None, _with_coefficient_metadata(metadata)

    # 3) Canonicalise tiny global scale factors.
    #
    # It's common for Stage-B / pretty_print quantisation to leave a constant-only
    # multiplicative factor extremely close to 1 (e.g. 1.000000...).
    # If the numeric equivalence gate still passes, prefer snapping it to ±1.
    try:
        c, rest = best.as_coeff_Mul()
        if isinstance(c, (sp.Float, sp.Integer, sp.Rational)) and c != 1 and rest != 1:
            for t in (sp.Integer(1), sp.Integer(-1)):
                ex_snap = rest if int(t) == 1 else -rest
                err_snap = _eval_expr(ex_snap)
                if err_snap is None or (not math.isfinite(err_snap)):
                    continue
                if err_snap > tol_accept:
                    continue
                ops_snap = _ops_score(ex_snap)
                abs_snap = _abs_node_count(ex_snap)
                abs_snap_extra = max(0, int(abs_snap) - int(base_abs_nodes))
                complexity_snap = _display_complexity(ex_snap) + abs_penalty_weight * float(abs_snap_extra)
                # Prefer strictly lower penalized complexity; otherwise shorter strings.
                if (complexity_snap < best_complexity) or (
                    (complexity_snap == best_complexity)
                    and (len(sp.sstr(ex_snap)) < len(sp.sstr(best)))
                ):
                    best = ex_snap
                    best_label = f"{best_label}+snap{int(t)}"
                    best_ops = ops_snap
                    best_err = err_snap
                    best_abs_nodes = int(abs_snap)
                    best_complexity = float(complexity_snap)
    except Exception:
        pass

    # Final display normalization for inverse-ratio powers.
    best = _canonicalize_inverse_ratio_powers(best)
    best_units_result = check_sympy_units(
        best,
        variable_names,
        units_spec,
        expression_space="phi",
    )
    if best_units_result.checked and not best_units_result.ok:
        # This should be unreachable because the same check participates in
        # ranking above. Keep the boundary fail-closed if a later display
        # normalization ever changes dimensional structure.
        reason = "selected Stage-C candidate failed the final phi-space units assertion"
        metadata = {
            "accepted": False,
            "parse_success": True,
            "max_err": best_err,
            "tol": tol_accept,
            "kind": "selected_candidate_unit_assertion_failed",
            "reason": reason,
            "numeric_fidelity_ok": True,
            "units_checked": True,
            "units_ok": False,
            "unit_admissibility": best_units_result.to_dict(),
            "candidate_count": len(candidates),
            "candidate_count_before_unit_guidance": candidate_count_before_unit_guidance,
            "evaluated_candidate_count": evaluated_candidate_count,
            "numeric_pass_count": numeric_pass_count,
            "unit_reject_count": unit_reject_count,
            "unit_valid_numeric_count": unit_valid_numeric_count,
            "unit_rejections": unit_rejections,
            "unit_guided_generation": unit_guided_generation,
            "proposal_budget": {
                "requested_count": 1,
                "raw_attempted": raw_proposal_attempted,
                "unit_rejected": unit_reject_count
                + int(unit_guided_generation["unit_rejected"]),
                "emitted": 0,
                "exhausted": True,
                "exhaustion_reason": "selected_candidate_unit_assertion_failed",
            },
        }
        return None, None, _with_coefficient_metadata(metadata)

    if verbose:
        abs_extra = (
            None
            if best_abs_nodes is None
            else int(max(0, int(best_abs_nodes) - int(base_abs_nodes)))
        )
        print(
            f"[Stage C] {GREEN}Accepted{RESET} SymPy candidate: {best_label}, "
            f"ops={best_ops}, abs={best_abs_nodes}, abs_extra={abs_extra}, "
            f"score={best_complexity:.3g}, max_err={best_err:.3g}, tol≈{tol_accept:.3g}"
        )

    try:
        phi_str = sp.sstr(best)
    except Exception:
        phi_str = str(best)

    # 4) Build y-space SymPy expression if requested.
    y_str = None
    y_units_result = None
    if y_op_inv is not None:
        try:
            from .transform_render import wrap_phi_expr_sympy

            y_expr = wrap_phi_expr_sympy(best, y_op_inv, sp=sp)
            try:
                with sympy_timeout("simplify_y_space", max_seconds=10):
                    y_expr = sp.simplify(y_expr)
            except (RecursionError, SympyTimeoutError):
                pass  # Keep y_expr as-is
            except Exception:
                pass
            if prefer_stable_trig:
                try:
                    with sympy_timeout("stable_trig_y_space", max_seconds=5):
                        y_expr = _prefer_stable_half_angle_trig(y_expr)
                except (RecursionError, SympyTimeoutError):
                    pass
                except Exception:
                    pass
            y_expr = _canonicalize_inverse_ratio_powers(y_expr)
            y_str = sp.sstr(y_expr)
            y_units_result = check_sympy_units(
                y_expr,
                variable_names,
                units_spec,
                expression_space="y",
            )
        except Exception as e:
            if verbose:
                print("[Stage C] SymPy y-space simplification failed:", e)
            y_str = None

    phi_certificate = best_units_result.to_dict()
    if y_units_result is None:
        unit_certificate = phi_certificate
    else:
        checks_enabled = bool(best_units_result.checked or y_units_result.checked)
        both_valid = bool(best_units_result.ok and y_units_result.ok)
        unit_certificate = {
            "checked": checks_enabled,
            "valid": both_valid if checks_enabled else None,
            "checker": "sympy_units_v1",
            "code": "units_ok" if both_valid else "raw_y_unit_check_failed",
            "reason": (
                "phi-space and raw-y expressions are dimensionally admissible"
                if both_valid
                else y_units_result.reason
            ),
            "expression_space": "phi_and_y",
            "phi": phi_certificate,
            "y": y_units_result.to_dict(),
        }
    if unit_certificate.get("checked") is True and unit_certificate.get("valid") is not True:
        metadata = {
            "accepted": False,
            "parse_success": True,
            "max_err": best_err,
            "tol": tol_accept,
            "kind": "raw_y_unit_check_failed",
            "reason": unit_certificate.get("reason"),
            "numeric_fidelity_ok": True,
            "units_checked": True,
            "units_ok": False,
            "unit_admissibility": unit_certificate,
            "candidate_count": len(candidates),
            "candidate_count_before_unit_guidance": candidate_count_before_unit_guidance,
            "evaluated_candidate_count": evaluated_candidate_count,
            "numeric_pass_count": numeric_pass_count,
            "unit_reject_count": unit_reject_count,
            "unit_valid_numeric_count": unit_valid_numeric_count,
            "unit_rejections": unit_rejections,
            "unit_guided_generation": unit_guided_generation,
            "proposal_budget": {
                "requested_count": 1,
                "raw_attempted": raw_proposal_attempted,
                "unit_rejected": unit_reject_count
                + int(unit_guided_generation["unit_rejected"]),
                "emitted": 0,
                "exhausted": True,
                "exhaustion_reason": "raw_y_unit_check_failed",
            },
        }
        return None, None, _with_coefficient_metadata(metadata)

    metadata = {
        "accepted": True,
        "parse_success": True,
        "max_err": best_err,
        "best_unit_valid_max_err": min_unit_valid_err,
        "best_unit_valid_error_candidate": min_unit_valid_err_label,
        "tol": tol_accept,
        "kind": best_label,
        "abs_nodes": best_abs_nodes,
        "abs_extra": None
        if best_abs_nodes is None
        else int(max(0, int(best_abs_nodes) - int(base_abs_nodes))),
        "complexity_score": best_complexity,
        "numeric_fidelity_ok": True,
        "units_checked": bool(unit_certificate.get("checked")),
        "units_ok": unit_certificate.get("valid"),
        "units_reason": unit_certificate.get("reason"),
        "unit_admissibility": unit_certificate,
        "candidate_count": len(candidates),
        "candidate_count_before_unit_guidance": candidate_count_before_unit_guidance,
        "evaluated_candidate_count": evaluated_candidate_count,
        "numeric_pass_count": numeric_pass_count,
        "unit_reject_count": unit_reject_count,
        "unit_valid_numeric_count": unit_valid_numeric_count,
        "unit_rejections": unit_rejections,
        "unit_guided_generation": unit_guided_generation,
        "proposal_budget": {
            "requested_count": 1,
            "raw_attempted": raw_proposal_attempted,
            "unit_rejected": unit_reject_count
            + int(unit_guided_generation["unit_rejected"]),
            "emitted": 1,
            "exhausted": False,
            "exhaustion_reason": None,
        },
    }
    return phi_str, y_str, _with_coefficient_metadata(metadata)


def _sympy_simplify_expression_worker_payload(**kwargs):
    """Importable worker entry for guarded Stage-C SymPy simplification."""
    return _sympy_simplify_expression(**kwargs)


def guarded_sympy_simplify_expression(
    expr_str: str,
    *,
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    Nxvars: int,
    y_op_inv=None,
    max_points: int = 2048,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-10,
    noise_floor_raw: Optional[float] = None,
    noise_abs_tol_factor: float = 0.25,
    prefer_stable_trig: bool = True,
    prune_trig_poly_args: bool = True,
    linearize_leaves: bool = True,
    units_spec=None,
    coefficient_metadata=None,
    verbose: bool = True,
    max_seconds: float = 300.0,
    mem_fraction: float = 0.20,
) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """Run Stage-C SymPy simplification in a bounded child process.

    The fitted model is sampled in the parent process; the child receives only
    numeric arrays and the expression string.  This is deliberate: if SymPy is
    killed by a memory limit, the main SR process still has the accepted Stage-B
    AST and can write a conservative report.
    """
    xs_np, ys_model = _collect_val_points_from_loader(
        model=model, val_loader=val_loader, device=device, max_points=max_points
    )
    if xs_np is None or ys_model is None:
        return None, None, {
            "accepted": False,
            "parse_success": True,
            "kind": "guard_input_failed",
            "reason": "could not collect validation points for guarded SymPy",
            "guarded_subprocess": True,
            "coefficient_metadata": coefficient_metadata,
        }

    kwargs = {
        "expr_str": str(expr_str),
        "model": None,
        "val_loader": None,
        "device": torch.device("cpu"),
        "Nxvars": int(Nxvars),
        # Keep y-space wrapping in the parent.  The child only needs to
        # certify an equivalent phi-space expression.
        "y_op_inv": None,
        "max_points": int(max_points),
        "rel_tol": float(rel_tol),
        "abs_tol": float(abs_tol),
        "noise_floor_raw": noise_floor_raw,
        "noise_abs_tol_factor": float(noise_abs_tol_factor),
        "prefer_stable_trig": bool(prefer_stable_trig),
        "prune_trig_poly_args": bool(prune_trig_poly_args),
        # Leaf-linearization needs live torch modules; guarded workers are
        # intended for fully analytic Stage-B states.
        "linearize_leaves": False,
        "units_spec": units_spec,
        "coefficient_metadata": coefficient_metadata,
        "verbose": bool(verbose),
        "precomputed_xs_np": np.asarray(xs_np, dtype=float),
        "precomputed_ys_model": np.asarray(ys_model, dtype=float).reshape(-1),
    }
    try:
        from nestynet_sr.sr_search.postprocess_guard import run_guarded_function

        outcome = run_guarded_function(
            "nestynet_sr.sr_search.representation:_sympy_simplify_expression_worker_payload",
            kwargs=kwargs,
            max_seconds=float(max_seconds),
            mem_fraction=float(mem_fraction),
            label="stageC_sympy",
        )
    except Exception as exc:
        if verbose:
            print("[Stage C] guarded SymPy worker setup failed:", exc)
        return None, None, {
            "accepted": False,
            "parse_success": False,
            "kind": "guard_setup_failed",
            "error": str(exc),
            "coefficient_metadata": coefficient_metadata,
        }

    if outcome.get("ok"):
        result = outcome.get("result")
        if isinstance(result, (tuple, list)) and len(result) == 3:
            phi_str, y_str, meta = result
            if isinstance(meta, dict):
                meta = dict(meta)
                meta["guarded_subprocess"] = True
                meta["guard_memory_limit_bytes"] = outcome.get("memory_limit_bytes")
                meta["guard_max_seconds"] = outcome.get("max_seconds")
            return phi_str, y_str, meta
        return None, None, {
            "accepted": False,
            "parse_success": False,
            "kind": "guard_bad_result",
            "guarded_subprocess": True,
            "result_type": type(result).__name__,
            "coefficient_metadata": coefficient_metadata,
        }

    reason = outcome.get("reason") or outcome.get("error") or outcome.get("status")
    if verbose:
        print(f"[Stage C] guarded SymPy worker failed safely: {reason}")
    return None, None, {
        "accepted": False,
        "parse_success": True,
        "kind": f"guarded_subprocess_{outcome.get('status', 'failed')}",
        "reason": reason,
        "error": outcome.get("error"),
        "error_type": outcome.get("error_type"),
        "returncode": outcome.get("returncode"),
        "guarded_subprocess": True,
        "guard_memory_limit_bytes": outcome.get("memory_limit_bytes"),
        "guard_max_seconds": outcome.get("max_seconds"),
        "coefficient_metadata": coefficient_metadata,
    }
