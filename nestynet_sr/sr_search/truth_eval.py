# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Ground-truth evaluation for symbolic regression canaries.

This module provides robust numeric comparison between discovered expressions
and known ground-truth formulas, with careful handling of singularities and
numerical instabilities.
"""

import math
import warnings
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from nestynet_sr.sr_core.coefficient_metadata import validate_coefficient_symbol

try:
    import sympy as sp

    _HAVE_SYMPY = True
except ImportError:
    _HAVE_SYMPY = False


def _blinded_active() -> bool:
    """True when the process is running in blinded mode.

    Blinded mode (set by ``run_SR.py --blinded`` via the ``NESTYNET_SR_BLINDED``
    environment variable) forbids any access to the ground-truth answer key:
    ground-truth evaluation short-circuits *before* the canary registry is
    opened, so no file containing a target expression is read by the search
    process.  Scoring must then be done afterwards by a separate process
    (``scripts/score_blinded_run.py``).
    """
    import os

    return os.environ.get("NESTYNET_SR_BLINDED", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _sort_var_names(var_names: List[str]) -> List[str]:
    """
    Sort variable names like x0, x1, x2, ... numerically (not lexicographically),
    with a safe fallback for nonconforming names.
    """

    def _key(s: str):
        if s.startswith("x") and s[1:].isdigit():
            return (0, int(s[1:]))
        return (1, s)

    return sorted(var_names, key=_key)


def _detect_singularities(
    expr: "sp.Expr", x_symbols: List["sp.Symbol"], rel_tol: float = 1e-6
) -> Optional["sp.Expr"]:
    """
    Attempt to extract a denominator *magnitude* from a SymPy expression.

    Returns a SymPy expression (typically Abs(denom)) that can be evaluated
    numerically; callers can apply a tolerance threshold to decide what counts
    as "near singular".

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression to analyze
    x_symbols : List[sp.Symbol]
        List of input symbols
    rel_tol : float
        Unused here (kept for API stability); singularity thresholding is
        handled in evaluate_against_truth.

    Returns
    -------
    sp.Expr or None
        A SymPy expression giving |denom(x)|, or None if no denominator detected.
    """
    try:
        # Mild normalization helps expose hidden denominators.
        expr2 = sp.together(expr)
        numer, denom = expr2.as_numer_denom()

        # If denominator is just 1, no singularities
        if denom == 1:
            return None

        # Return denominator magnitude; thresholding happens downstream.
        return sp.Abs(denom)
    except Exception:
        return None


def _as_sample_vector(value, n_samples: int, *, label: str) -> np.ndarray:
    """Normalize scalar or per-sample lambdify output to one flat vector."""
    array = np.asarray(value, dtype=float)
    if array.size == 1:
        return np.full(n_samples, float(array.reshape(-1)[0]), dtype=float)
    if array.size == n_samples:
        return array.reshape(n_samples)
    raise ValueError(
        f"{label} produced {array.size} values for {n_samples} sample points "
        f"(shape={array.shape})"
    )


def evaluate_against_truth(
    discovered_expr_str: str,
    truth_expr_str: str,
    domain_bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 10000,
    min_valid_frac: float = 0.8,
    singularity_tol: float = 1e-6,
    singularity_rel_tol: float = 1e-6,
    stratified: bool = True,
    n_strata_per_dim: int = 5,
    seed: int = 42,
    verbose: bool = False,
    symbol_values: Optional[Mapping[str, float]] = None,
) -> Dict:
    """
    Evaluate a discovered expression against ground truth with robust singularity handling.

    Parameters
    ----------
    discovered_expr_str : str
        String representation of discovered expression (e.g., "0.5*x0**2 + x1")
    truth_expr_str : str
        String representation of ground truth expression
    domain_bounds : Dict[str, Tuple[float, float]]
        Domain bounds for each variable, e.g., {"x0": (0.01, 10), "x1": (-5, 5)}
    n_samples : int
        Total number of points to sample
    min_valid_frac : float
        Minimum fraction of valid points required for successful evaluation
    singularity_tol : float
        Tolerance for detecting singularities (points where |denominator| < tol)
    singularity_rel_tol : float
        Relative tolerance for detecting singularities, scaled by the typical
        magnitude of the (truth) denominator: tol = max(abs_tol, rel_tol * median(|denom|)).
    stratified : bool
        Use stratified sampling to ensure coverage of entire domain
    n_strata_per_dim : int
        Number of strata per dimension for stratified sampling
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print diagnostic information
    symbol_values : mapping, optional
        Numeric values for named coefficients retained in the discovered
        expression. They are substituted for evaluation without changing the
        symbolic expression stored in run artifacts.

    Returns
    -------
    dict
        Dictionary with keys:
        - success: bool, whether evaluation succeeded
        - rmse_abs: float, absolute RMSE
        - rmse_rel: float, relative RMSE (normalized by MAD of truth)
        - max_abs_err: float, maximum absolute error
        - max_rel_err: float, maximum relative error
        - frac_valid: float, fraction of total points valid in both (truth-valid & discovered-finite)
        - frac_truth_valid: float, fraction of total points valid for the truth expression
        - frac_valid_wrt_truth: float, fraction of truth-valid points also valid for discovered
        - n_valid: int, number of valid points
        - n_truth_valid: int, number of truth-valid points
        - n_total: int, total number of points sampled
        - error_message: str, error message if evaluation failed
        - coefficient_symbols_used: list[str], substituted discovered-expression
          coefficient symbols
    """
    result = {
        "success": False,
        "rmse_abs": None,
        "rmse_rel": None,
        "max_abs_err": None,
        "max_rel_err": None,
        "frac_valid": 0.0,
        "frac_truth_valid": 0.0,
        "frac_valid_wrt_truth": 0.0,
        "n_valid": 0,
        "n_truth_valid": 0,
        "n_total": n_samples,
        "error_message": None,
        "coefficient_symbols_used": [],
    }

    if not _HAVE_SYMPY:
        result["error_message"] = "SymPy not available"
        return result

    # Resolve input and coefficient symbols before parsing.  Coefficients stay
    # symbolic in artifacts, but this evaluator is a numerical boundary and
    # therefore requires a finite value for every non-input symbol.
    var_names = _sort_var_names(list(domain_bounds.keys()))
    n_vars = len(var_names)
    variable_set = set(var_names)
    coefficient_values: dict[str, float] = {}
    try:
        for raw_name, raw_value in (symbol_values or {}).items():
            name = validate_coefficient_symbol(raw_name)
            if name in variable_set:
                raise ValueError(
                    f"coefficient symbol {name!r} collides with an input variable"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"coefficient symbol {name!r} has non-finite value {value!r}"
                )
            coefficient_values[name] = value
    except Exception as e:
        result["error_message"] = f"Invalid coefficient values: {e}"
        return result

    try:
        # Prepare local dict for sympify
        local_dict = {
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "asin": sp.asin,
            "acos": sp.acos,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "pi": sp.pi,
            "E": sp.E,
        }
        local_dict.update({name: sp.Symbol(name, real=True) for name in var_names})
        local_dict.update(
            {name: sp.Symbol(name, real=True) for name in coefficient_values}
        )

        # Convert ^ to ** for SymPy
        discovered_expr_str = discovered_expr_str.replace("^", "**")
        truth_expr_str = truth_expr_str.replace("^", "**")
        # Normalize common spellings for inverse trig (AIF canaries use arcsin/arccos)
        discovered_expr_str = discovered_expr_str.replace("arcsin", "asin").replace(
            "arccos", "acos"
        )
        truth_expr_str = truth_expr_str.replace("arcsin", "asin").replace("arccos", "acos")

        discovered_expr = sp.sympify(discovered_expr_str, locals=local_dict)
        truth_expr = sp.sympify(truth_expr_str, locals=local_dict)
    except Exception as e:
        result["error_message"] = f"Failed to parse expressions: {e}"
        return result

    substitutions = {
        symbol: sp.Float(repr(coefficient_values[str(symbol)]), 17)
        for expr in (discovered_expr, truth_expr)
        for symbol in expr.free_symbols
        if str(symbol) in coefficient_values
    }
    discovered_symbols = {str(symbol) for symbol in discovered_expr.free_symbols}
    result["coefficient_symbols_used"] = sorted(
        discovered_symbols & set(coefficient_values)
    )
    discovered_expr_numeric = discovered_expr.xreplace(substitutions)
    truth_expr_numeric = truth_expr.xreplace(substitutions)
    unresolved_symbols = sorted(
        {
            str(symbol)
            for expr in (discovered_expr_numeric, truth_expr_numeric)
            for symbol in expr.free_symbols
            if str(symbol) not in variable_set
        }
    )
    if unresolved_symbols:
        result["error_message"] = (
            "Missing coefficient values for symbols: "
            + ", ".join(unresolved_symbols)
        )
        return result

    x_symbols = [sp.Symbol(name, real=True) for name in var_names]

    # Detect singularities in both expressions
    discovered_denom = _detect_singularities(discovered_expr_numeric, x_symbols)
    truth_denom = _detect_singularities(truth_expr_numeric, x_symbols)

    # Generate sample points
    rng = np.random.default_rng(seed)

    if stratified and n_vars <= 3:
        # Stratified sampling: divide domain into grid, sample from each cell
        # This ensures coverage even near boundaries and singularities
        n_strata = n_strata_per_dim**n_vars
        samples_per_stratum = max(1, n_samples // n_strata)

        points = []
        for i in range(n_strata):
            # Convert linear index to multi-dimensional stratum index
            strat_idx = []
            remainder = i
            for _ in range(n_vars):
                strat_idx.append(remainder % n_strata_per_dim)
                remainder //= n_strata_per_dim

            # Sample within this stratum
            for _ in range(samples_per_stratum):
                point = []
                for j, var_name in enumerate(var_names):
                    lo, hi = domain_bounds[var_name]
                    # Stratum bounds
                    strat_lo = lo + (hi - lo) * strat_idx[j] / n_strata_per_dim
                    strat_hi = lo + (hi - lo) * (strat_idx[j] + 1) / n_strata_per_dim
                    # Sample uniformly within stratum
                    point.append(rng.uniform(strat_lo, strat_hi))
                points.append(point)

        X = np.array(points, dtype=float)
        # Ensure deterministic size exactly n_samples (avoid n_total drifting)
        if X.shape[0] > n_samples:
            idx = rng.choice(X.shape[0], size=n_samples, replace=False)
            X = X[idx]
        elif X.shape[0] < n_samples:
            n_extra = n_samples - X.shape[0]
            X_extra = np.zeros((n_extra, n_vars), dtype=float)
            for j, var_name in enumerate(var_names):
                lo, hi = domain_bounds[var_name]
                X_extra[:, j] = rng.uniform(lo, hi, size=n_extra)
            X = np.vstack([X, X_extra])
    else:
        # Uniform random sampling
        X = np.zeros((n_samples, n_vars))
        for j, var_name in enumerate(var_names):
            lo, hi = domain_bounds[var_name]
            X[:, j] = rng.uniform(lo, hi, size=n_samples)

    n_total = X.shape[0]
    result["n_total"] = n_total

    # Lambdify expressions for fast numeric evaluation
    try:
        discovered_fn = sp.lambdify(
            x_symbols, discovered_expr_numeric, modules=["numpy"]
        )
        truth_fn = sp.lambdify(x_symbols, truth_expr_numeric, modules=["numpy"])

        # Also lambdify singularity masks if present
        if discovered_denom is not None:
            discovered_mask_fn = sp.lambdify(x_symbols, discovered_denom, modules=["numpy"])
        else:
            discovered_mask_fn = None

        if truth_denom is not None:
            truth_mask_fn = sp.lambdify(x_symbols, truth_denom, modules=["numpy"])
        else:
            truth_mask_fn = None

    except Exception as e:
        result["error_message"] = f"Failed to lambdify expressions: {e}"
        return result

    # Evaluate expressions
    try:
        # Prepare arguments (each variable as a separate array)
        args = [X[:, j] for j in range(n_vars)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            y_discovered = _as_sample_vector(
                discovered_fn(*args), n_total, label="discovered expression"
            )
            y_truth = _as_sample_vector(
                truth_fn(*args), n_total, label="truth expression"
            )

            # Evaluate singularity masks
            if discovered_mask_fn is not None:
                _discovered_denom_vals = np.abs(
                    _as_sample_vector(
                        discovered_mask_fn(*args),
                        n_total,
                        label="discovered denominator",
                    )
                )
            else:
                _discovered_denom_vals = None

            if truth_mask_fn is not None:
                truth_denom_vals = np.abs(
                    _as_sample_vector(
                        truth_mask_fn(*args),
                        n_total,
                        label="truth denominator",
                    )
                )
            else:
                truth_denom_vals = None
    except Exception as e:
        result["error_message"] = f"Failed to evaluate expressions: {e}"
        return result

    # ------------------------------------------------------------------
    # Validity policy:
    #   - Define the evaluation region using *truth* validity and truth singularities.
    #   - Require discovered expression to be finite on those truth-valid points.
    # This prevents the discovered expression from "hiding" errors by introducing
    # extra singularities and filtering them away.
    # ------------------------------------------------------------------
    truth_valid_mask = np.isfinite(y_truth)
    if truth_denom_vals is not None:
        finite_den = truth_denom_vals[np.isfinite(truth_denom_vals)]
        denom_scale = float(np.median(finite_den)) if finite_den.size else 0.0
        tol_eff = max(float(singularity_tol), float(singularity_rel_tol) * float(denom_scale))
        if not np.isfinite(tol_eff) or tol_eff <= 0.0:
            tol_eff = float(singularity_tol)
        truth_valid_mask &= truth_denom_vals > tol_eff

    n_truth_valid = int(np.sum(truth_valid_mask))
    frac_truth_valid = n_truth_valid / float(n_total) if n_total > 0 else 0.0
    result["n_truth_valid"] = n_truth_valid
    result["frac_truth_valid"] = float(frac_truth_valid)

    if verbose:
        print(
            f"[Truth Eval] Truth-valid points: {n_truth_valid}/{n_total} ({frac_truth_valid:.1%})"
        )

    if frac_truth_valid < min_valid_frac:
        result["error_message"] = (
            f"Too few truth-valid points: {frac_truth_valid:.1%} < {min_valid_frac:.1%}"
        )
        return result

    both_valid_mask = truth_valid_mask & np.isfinite(y_discovered)
    n_valid = int(np.sum(both_valid_mask))
    frac_valid = n_valid / float(n_total) if n_total > 0 else 0.0
    frac_valid_wrt_truth = n_valid / float(n_truth_valid) if n_truth_valid > 0 else 0.0

    result["n_valid"] = n_valid
    result["frac_valid"] = float(frac_valid)
    result["frac_valid_wrt_truth"] = float(frac_valid_wrt_truth)

    if verbose:
        print(f"[Truth Eval] Both-valid points:  {n_valid}/{n_total} ({frac_valid:.1%})")
        print(f"[Truth Eval] Valid wrt truth:    {frac_valid_wrt_truth:.1%}")

    # Require discovered to be valid on a sufficient fraction of truth-valid points
    if frac_valid_wrt_truth < min_valid_frac:
        result["error_message"] = (
            f"Too few discovered-valid points on truth-valid region: "
            f"{frac_valid_wrt_truth:.1%} < {min_valid_frac:.1%}"
        )
        return result

    # Extract valid points
    y_disc_valid = y_discovered[both_valid_mask]
    y_true_valid = y_truth[both_valid_mask]

    # Compute errors
    abs_err = np.abs(y_disc_valid - y_true_valid)

    # Compute metrics
    rmse_abs = float(np.sqrt(np.mean(abs_err**2)))
    max_abs_err = float(np.max(abs_err))

    # Relative metrics (normalize by MAD of truth)
    truth_mad = float(np.median(np.abs(y_true_valid - np.median(y_true_valid))))
    if truth_mad < 1e-15:
        # If truth is essentially constant, use std or fallback to 1
        truth_mad = float(np.std(y_true_valid))
        if truth_mad < 1e-15:
            truth_mad = 1.0

    rmse_rel = rmse_abs / truth_mad

    # Compute max relative error (avoid division by very small values)
    rel_err = abs_err / np.maximum(np.abs(y_true_valid), truth_mad * 0.1)
    max_rel_err = float(np.max(rel_err))

    result.update(
        {
            "success": True,
            "rmse_abs": rmse_abs,
            "rmse_rel": rmse_rel,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
        }
    )

    if verbose:
        print(f"[Truth Eval] RMSE (abs): {rmse_abs:.3e}")
        print(f"[Truth Eval] RMSE (rel): {rmse_rel:.3e}")
        print(f"[Truth Eval] Max error (abs): {max_abs_err:.3e}")
        print(f"[Truth Eval] Max error (rel): {max_rel_err:.3e}")

    return result


_canary_registry_cache: Optional[Dict[str, Dict]] = None


def load_ground_truth_registry() -> Dict[str, Dict]:
    """
    Load the ground truth registry for known canary problems.

    Attempts to load from aif_canaries.json (comprehensive) or canary_truths.json (legacy)
    in the symbolic_regression directory. Falls back to a minimal default registry if not found.

    Returns a dictionary mapping dataset stems to ground truth metadata:
    {
        "pb000": {  # or "pb000_I_6_2a_data" for legacy format
            "truth_expr": "exp(-x0**2/2)/(sqrt(2*pi))",
            "domain_bounds": {"x0": [1.0, 3.0]},
            "description": "AIF equation 000",
            "reference": "AIF-000"
        },
        ...
    }

    Returns
    -------
    dict
        Registry of ground truth expressions and metadata
    """
    global _canary_registry_cache
    if _canary_registry_cache is not None:
        return _canary_registry_cache

    if _blinded_active():
        # Blinded mode: never open the answer key. Return an empty registry so
        # callers resolve no ground truth.
        return {}

    import json
    import os

    registry_dir = os.path.dirname(os.path.dirname(__file__))

    # Try aif_canaries.json first (comprehensive registry)
    aif_path = os.path.join(registry_dir, "aif_canaries.json")
    if os.path.exists(aif_path):
        try:
            with open(aif_path, "r") as f:
                registry = json.load(f)

            # Convert domain_bounds from lists to tuples for consistency
            for canary_data in registry.values():
                if "domain_bounds" in canary_data:
                    canary_data["domain_bounds"] = {
                        k: tuple(v) if isinstance(v, list) else v
                        for k, v in canary_data["domain_bounds"].items()
                    }

            print(f"[Truth Eval] Loaded {len(registry)} canaries from aif_canaries.json")
            _canary_registry_cache = registry
            return registry
        except Exception as e:
            print(f"Warning: Failed to load aif_canaries.json: {e}")

    # Fallback: minimal default registry
    print("[Truth Eval] Using fallback registry with 1 canary")
    registry = {
        "pb000": {
            "truth_expr": "exp(-x0**2/2)/(sqrt(2*pi))",
            "domain_bounds": {"x0": (1.0, 3.0)},
            "description": "Standard normal PDF (fallback)",
            "reference": "AIF-000",
        },
    }

    _canary_registry_cache = registry
    return registry


def evaluate_canary(
    dataset_stem: str,
    discovered_expr_str: str,
    verbose: bool = False,
    symbol_values: Optional[Mapping[str, float]] = None,
) -> Optional[Dict]:
    """
    Convenience function to evaluate a discovered expression against ground truth
    for a known canary problem.

    Parameters
    ----------
    dataset_stem : str
        Dataset filename stem (e.g., "pb000_I_6_2a_data" or "pb000")
    discovered_expr_str : str
        Discovered expression string
    verbose : bool
        Print diagnostic information
    symbol_values : mapping, optional
        Numeric values for named coefficients retained in the discovered
        expression.

    Returns
    -------
    dict or None
        Evaluation results, or None if canary not in registry
    """
    if _blinded_active():
        return {
            "success": False,
            "skipped": True,
            "blinded": True,
            "reason": (
                "blinded_mode: ground-truth access disabled during the run; "
                "score afterwards with scripts/score_blinded_run.py"
            ),
        }
    registry = load_ground_truth_registry()

    # Try exact match first
    canary = None
    if dataset_stem in registry:
        canary = registry[dataset_stem]
    else:
        # Try extracting just pbXXX prefix (e.g., pb000_I_6_2a -> pb000)
        if canary is None:
            import re

            match = re.match(r"(pb\d{3})", dataset_stem)
            if match:
                short_stem = match.group(1)
                if short_stem in registry:
                    canary = registry[short_stem]

    if canary is None:
        if verbose:
            print(f"[Truth Eval] No ground truth available for {dataset_stem}")
        return None

    if verbose:
        print(f"[Truth Eval] Evaluating against ground truth: {canary['truth_expr']}")
        print(f"[Truth Eval] Description: {canary['description']}")

    return evaluate_against_truth(
        discovered_expr_str=discovered_expr_str,
        truth_expr_str=canary["truth_expr"],
        domain_bounds=canary["domain_bounds"],
        verbose=verbose,
        symbol_values=symbol_values,
    )
