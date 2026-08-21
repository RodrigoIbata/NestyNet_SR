#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Summarize AI Feynman benchmark results from .report.json files.

Usage:
    python scripts/summarize_aifeyn_results.py results --csv results/summary.csv
"""

import argparse
import json
import math
import re
from pathlib import Path


def _legacy_pickle_load(f):
    """Pickle loader that handles old checkpoint files with removed classes."""
    import pickle
    from dataclasses import dataclass
    from typing import Tuple

    @dataclass
    class _CompoundVar:
        """Stub for deserialization of old pickles."""
        expr: object
        extra_var_idxs: Tuple[int, ...] = ()
        def __post_init__(self):
            self.extra_var_idxs = tuple(int(i) for i in (self.extra_var_idxs or ()))

    class _LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "CompoundVar" and "bridges" in module:
                return _CompoundVar
            return super().find_class(module, name)

    return _LegacyUnpickler(f).load()


def load_report(path: Path) -> dict | None:
    """Load a single report JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def load_ground_truth_equations(equations_path: Path) -> dict[str, str]:
    """Load ground truth equations from equations.txt.

    Returns a dict mapping problem ID (e.g., '000') to equation string.
    """
    equations = {}
    if not equations_path.exists():
        return equations

    try:
        with open(equations_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: ID vars xmin xmax eqn y_units x_units
                # Example: 000 ['x0'] [1.] [3.]    exp(-x0**2/2)/(sqrt(2*pi))     [0.0, ...]
                # The equation is after the 3rd ] (xmax) and before the 4th [ (y_units)
                parts = line.split()
                if not parts:
                    continue
                problem_id = parts[0]

                # Find all ] positions and [ positions
                # The equation is between the 3rd ] and 4th [
                close_brackets = [i for i, c in enumerate(line) if c == ']']
                open_brackets = [i for i, c in enumerate(line) if c == '[']

                if len(close_brackets) >= 3 and len(open_brackets) >= 4:
                    # Equation starts after 3rd ] and ends before 4th [
                    start = close_brackets[2] + 1
                    end = open_brackets[3]
                    equation = line[start:end].strip()
                    if equation:
                        equations[problem_id] = equation
    except OSError as e:
        print(f"Warning: Could not load equations file {equations_path}: {e}")

    return equations


def extract_problem_id(problem_name: str) -> str | None:
    """Extract the numeric problem ID from a problem name like 'pb000_I_6_2a'.

    Returns the ID string (e.g., '000') or None if not found.
    """
    match = re.match(r'pb(\d+)', problem_name)
    if match:
        return match.group(1)
    return None


def extract_problem_name(path: Path) -> str:
    """Extract problem name from filename like pb000_I_6_2a_data.report.json."""
    name = path.stem  # pb000_I_6_2a_data.report
    if name.endswith(".report"):
        name = name[:-7]  # pb000_I_6_2a_data
    if name.endswith("_data"):
        name = name[:-5]  # pb000_I_6_2a
    return name


def _infer_noise_level(metadata: dict, results_dir: Path) -> float | None:
    """Infer declared target noise without inventing a tolerance for it."""
    candidates = [str((metadata or {}).get("dataset") or ""), str(results_dir)]
    patterns = (r"noise_(\d+(?:\.\d+)?)", r"SRBench_(\d+(?:\.\d+)?)")
    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
    return None


def load_stageA_val_loss_from_pkl(report_path: Path) -> float | None:
    """Load val_loss from the .state.pkl checkpoint file.

    This is the authoritative source for Stage A val_loss, as it's saved
    directly by run_SR.py after Stage A completes.
    """
    import pickle

    # Derive pkl path from report path: foo.report.json -> foo.state.pkl
    stem = report_path.stem  # foo.report
    if stem.endswith(".report"):
        stem = stem[:-7]  # foo
    pkl_path = report_path.parent / f"{stem}.state.pkl"

    if not pkl_path.exists():
        return None

    try:
        with open(pkl_path, "rb") as f:
            ckpt = _legacy_pickle_load(f)
        return ckpt.get("val_loss")
    except (OSError, pickle.PickleError, KeyError):
        return None


def load_stageA_separability_from_pkl(report_path: Path) -> bool | None:
    """Load separability_success from the .state.pkl checkpoint file."""
    import pickle

    stem = report_path.stem
    if stem.endswith(".report"):
        stem = stem[:-7]
    pkl_path = report_path.parent / f"{stem}.state.pkl"

    if not pkl_path.exists():
        return None

    try:
        with open(pkl_path, "rb") as f:
            ckpt = _legacy_pickle_load(f)
        return ckpt.get("separability_success")
    except (OSError, pickle.PickleError, KeyError):
        return None


def load_stageB_data_from_pkl(report_path: Path) -> dict | None:
    """Load Stage B data from the _stageB.pkl file.

    This is used as a fallback when the JSON report is missing stageB/stageC sections
    but the Stage B output was saved to a separate pkl file.

    Returns dict with: val_loss, phi_expr_str, y_expr_str, or None if unavailable.
    """
    import pickle

    # Derive pkl path from report path: foo.report.json -> foo_stageB.pkl
    stem = report_path.stem  # foo.report
    if stem.endswith(".report"):
        stem = stem[:-7]  # foo
    pkl_path = report_path.parent / f"{stem}_stageB.pkl"

    if not pkl_path.exists():
        return None

    try:
        with open(pkl_path, "rb") as f:
            data = _legacy_pickle_load(f)
        # Extract the relevant fields from the stageB pkl
        # Prefer raw-x expressions (with x-transforms / compound variables expanded)
        # over internal-coordinate expressions
        result = {}
        if "stageB_val_loss" in data:
            result["val_loss"] = data["stageB_val_loss"]
        if "phi_expr_raw_str" in data and data["phi_expr_raw_str"] is not None:
            result["phi_expr_str"] = data["phi_expr_raw_str"]
        elif "phi_expr_str" in data:
            result["phi_expr_str"] = data["phi_expr_str"]
        if "y_expr_raw_str" in data and data["y_expr_raw_str"] is not None:
            result["y_expr_str"] = data["y_expr_raw_str"]
        elif "y_expr_str" in data:
            result["y_expr_str"] = data["y_expr_str"]
        return result if result else None
    except (OSError, pickle.PickleError, KeyError):
        return None


def infer_separability_from_ast(ast_human: str | None) -> bool | None:
    """Infer separability from ast_human string.

    If the AST contains Add/Mul structure (e.g., '(NN[x0] * NN[x1])'),
    separability was found. A single NN atom means no separability.
    """
    if ast_human is None:
        return None
    # Check for additive or multiplicative separability patterns
    if " + " in ast_human or " * " in ast_human:
        return True
    # Single NN atom or unknown structure
    if ast_human.startswith("NN["):
        return False
    return None


def parse_stageA_val_loss_from_log(report_path: Path) -> float | None:
    """Fallback: Parse val-loss from stageA log file when pkl unavailable.

    Looks for the FIRST 'ADOPTING best val-loss X' with the chosen y-transform,
    not the last one (which could be from a different transform trial).
    """
    # Derive log path from report path: foo.report.json -> foo_stageA.log
    stem = report_path.stem  # foo.report
    if stem.endswith(".report"):
        stem = stem[:-7]  # foo
    log_path = report_path.parent / f"{stem}_stageA.log"

    if not log_path.exists():
        return None

    # Pattern for accepted separations and adoptions
    adopting_pattern = re.compile(r"ADOPTING.*val-loss\s+([\d.eE+-]+)")

    first_adopting = None

    try:
        with open(log_path) as f:
            for line in f:
                # Strip ANSI escape codes
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line)

                # Take the FIRST adopting (which is for the chosen y-transform)
                if first_adopting is None:
                    match = adopting_pattern.search(clean)
                    if match:
                        first_adopting = float(match.group(1))
                        break  # Stop at first match
    except OSError:
        return None

    return first_adopting


def format_float(val, precision=3) -> str:
    """Format a float for display, handling None and special values."""
    if val is None:
        return "-"
    if isinstance(val, str):
        return val
    if math.isnan(val) or math.isinf(val):
        return str(val)
    if val == 0:
        return "0"
    if abs(val) < 1e-4 or abs(val) >= 1e4:
        return f"{val:.{precision}e}"
    return f"{val:.{precision}f}"


def _count_inputs(expr_str: str) -> int | None:
    """Count the number of input variables (x0, x1, ...) in an expression."""
    if expr_str is None:
        return None
    # Find all xN patterns and get unique variable indices
    matches = re.findall(r'x(\d+)', expr_str)
    if not matches:
        return None
    # Number of inputs is max index + 1 (since x0-indexed)
    return max(int(m) for m in matches) + 1


def _count_significant_params(expr_str: str, threshold: float = 1e-6) -> int | None:
    """Count numeric constants in expression with |value| > threshold.

    Parameters
    ----------
    expr_str : str
        Expression string to parse with SymPy
    threshold : float
        Minimum absolute value for a constant to be considered significant

    Returns
    -------
    int or None
        Number of significant numeric constants, or None if parsing fails
    """
    if expr_str is None:
        return None
    try:
        import sympy as sp
        # Parse expression (handle ^ as power)
        expr = sp.sympify(expr_str.replace("^", "**"))
        # Extract all numeric atoms (excludes symbolic constants like pi, E)
        numbers = expr.atoms(sp.Number)
        # Count those with |value| > threshold
        return sum(1 for n in numbers if abs(float(n)) > threshold)
    except Exception:
        return None


def _normalize_expression(expr_str: str) -> str | None:
    """Normalize expression string for truth evaluation.

    Converts custom function names to standard forms:
    - negexp(x) -> -exp(x)  (inverse of logneg transform)
    """
    if expr_str is None:
        return None
    # Replace negexp(...) with -exp(...)
    expr_str = re.sub(r'\bnegexp\s*\(', '-exp(', expr_str)
    return expr_str


def _wrap_phi_with_inverse_transform(phi_expr_str: str, y_transform: str, simplify: bool = True) -> str | None:
    """
    Wrap a phi-space expression with the inverse y-transform to produce y-space.

    This is used to fix legacy reports where y_expr_str was not computed.
    When simplify=True, applies SymPy simplification to clean up expressions
    like exp(log(x0) + log(x1)) → x0*x1.
    """
    if phi_expr_str is None or y_transform is None:
        return None

    y_transform = y_transform.lower()

    # Map y-transform name to how to wrap phi to get y
    # y_transform is the forward transform (y -> phi), so we need the inverse
    wrapped_str = None
    if y_transform == "identity":
        wrapped_str = phi_expr_str
    elif y_transform == "log":
        wrapped_str = f"exp({phi_expr_str})"
    elif y_transform == "exp":
        wrapped_str = f"log({phi_expr_str})"
    elif y_transform == "square":
        wrapped_str = f"sqrt({phi_expr_str})"
    elif y_transform == "reciprocal":
        wrapped_str = f"1/({phi_expr_str})"
    elif y_transform == "sin":
        wrapped_str = f"arcsin({phi_expr_str})"
    elif y_transform == "cos":
        wrapped_str = f"arccos({phi_expr_str})"
    elif y_transform == "tan":
        wrapped_str = f"arctan({phi_expr_str})"
    elif y_transform == "arcsin":
        wrapped_str = f"sin({phi_expr_str})"
    elif y_transform == "arccos":
        wrapped_str = f"cos({phi_expr_str})"
    elif y_transform == "arctan":
        wrapped_str = f"tan({phi_expr_str})"
    elif y_transform == "logneg":
        wrapped_str = f"-exp({phi_expr_str})"
    elif y_transform == "expneg":
        wrapped_str = f"-log({phi_expr_str})"
    else:
        # Unknown transform - return None to use original truth_eval
        return None

    if wrapped_str is None:
        return None

    # Apply SymPy simplification if requested
    # This converts expressions like exp(log(x0) + log(x1) + C) → x0*x1*exp(C)
    if simplify:
        try:
            import sympy as sp
            y_expr = sp.sympify(wrapped_str.replace("^", "**"))
            y_expr = sp.simplify(y_expr)
            return sp.sstr(y_expr)
        except Exception:
            pass  # Fall back to unsimplified wrapped string

    return wrapped_str


def summarize_results(results_dir: Path, recompute_truth: bool = False, sig_threshold: float = 1e-6,
                      ground_truth: dict[str, str] | None = None,
                      exact_rmse_rel: float = 1e-8) -> list[dict]:
    """Load all reports and extract summary info.

    Parameters
    ----------
    results_dir : Path
        Directory containing .report.json files
    recompute_truth : bool
        If True, force truth_eval recomputation for the authoritative expression.
    sig_threshold : float
        Threshold for counting significant parameters (|value| > threshold)
    ground_truth : dict or None
        Dict mapping problem ID to ground truth equation string
    exact_rmse_rel : float
        Maximum relative RMSE classified as exact recovery.
    """
    from nestynet_sr.run_sr_reports import _report_final_selection_eligibility
    from nestynet_sr.sr_core.coefficient_metadata import coefficient_symbol_values
    from nestynet_sr.sr_search.truth_eval import evaluate_canary

    if ground_truth is None:
        ground_truth = {}

    reports = []

    for path in sorted(results_dir.glob("pb*.report.json")):
        data = load_report(path)
        if data is None:
            continue

        problem = extract_problem_name(path)

        # Extract fields with safe defaults
        metadata = data.get("metadata", {})
        stageA = data.get("stageA", {})
        stageB = data.get("stageB", {})
        stageC = data.get("stageC", {})
        selection_eligible, selection_reason = _report_final_selection_eligibility(data)
        final_selection = data.get("final_selection")
        selected_expr = (
            final_selection.get("expr")
            if selection_eligible and isinstance(final_selection, dict)
            else None
        )
        top_truth = data.get("truth_eval", {})
        final_truth = (
            final_selection.get("truth_eval", {})
            if isinstance(final_selection, dict)
            else {}
        )
        if selected_expr:
            # Never attach a metric to a different authoritative expression.
            truth = next(
                (
                    candidate
                    for candidate in (final_truth, top_truth)
                    if isinstance(candidate, dict)
                    and candidate.get("expr") == selected_expr
                ),
                {},
            )
        else:
            truth = top_truth if isinstance(top_truth, dict) else {}

        # Fallback: load Stage B data from pkl if JSON is missing these sections
        if not stageB and not stageC:
            stageB_pkl = load_stageB_data_from_pkl(path)
            if stageB_pkl:
                coefficient_metadata = stageB_pkl.get("coefficient_metadata")
                stageB = {
                    "val_loss": stageB_pkl.get("val_loss"),
                    "coefficient_metadata": coefficient_metadata,
                    "coefficient_metadata_by_dataset": stageB_pkl.get(
                        "coefficient_metadata_by_dataset"
                    ),
                }
                stageC = {
                    "phi_expr_str": stageB_pkl.get("phi_expr_str"),
                    "y_expr_str": stageB_pkl.get("y_expr_str"),
                    "coefficient_metadata": coefficient_metadata,
                }

        final_polish = data.get("final_polish") or {}
        try:
            coefficient_metadata = next(
                (
                    payload
                    for payload in (
                        (data.get("final_selection") or {}).get(
                            "coefficient_metadata"
                        ),
                        final_polish.get("coefficient_metadata"),
                        stageC.get("coefficient_metadata"),
                        stageB.get("coefficient_metadata"),
                    )
                    if payload is not None
                ),
                None,
            )
            coefficient_values = coefficient_symbol_values(
                coefficient_metadata
            )
            coefficient_values_error = None
        except Exception as exc:
            coefficient_values = {}
            coefficient_values_error = str(exc)

        def _evaluate_with_coefficients(expression):
            if coefficient_values_error is not None:
                return {
                    "success": False,
                    "error_message": (
                        "Invalid coefficient metadata: "
                        + coefficient_values_error
                    ),
                }
            kwargs = {
                "dataset_stem": problem,
                "discovered_expr_str": expression,
                "verbose": False,
            }
            if coefficient_values:
                kwargs["symbol_values"] = coefficient_values
            result = evaluate_canary(**kwargs)
            if isinstance(result, dict):
                result = dict(result)
                result["source"] = "summary_recompute"
                result["expr"] = expression
            return result

        # Check if we need to re-compute truth_eval
        # This happens when y_expr_str is null but phi_expr_str exists with non-identity y_transform
        y_transform = stageA.get("y_transform", "identity")
        y_expr_str = stageC.get("y_expr_str")
        phi_expr_str = stageC.get("phi_expr_str")

        if (
            selection_eligible
            and recompute_truth
            and selected_expr is None
            and y_expr_str is None
            and phi_expr_str is not None
            and y_transform != "identity"
        ):
            # Apply inverse y-transform to get the correct y-space expression
            corrected_expr = _wrap_phi_with_inverse_transform(phi_expr_str, y_transform)
            if corrected_expr is not None:
                # Re-compute truth_eval with corrected expression
                new_truth = _evaluate_with_coefficients(corrected_expr)
                if new_truth is not None:
                    truth = new_truth

        # Recompute missing/mismatched legacy metrics against the expression
        # that is actually displayed. --recompute-truth forces a fresh score.
        if selection_eligible and selected_expr and (recompute_truth or not truth):
            new_truth = _evaluate_with_coefficients(selected_expr)
            if new_truth is not None:
                truth = new_truth

        # Compute truth_eval if missing but we have an older expression source.
        if selection_eligible and not truth and (
            selected_expr or y_expr_str or phi_expr_str
        ):
            eval_expr = selected_expr or y_expr_str
            if eval_expr is None and phi_expr_str is not None:
                if y_transform != "identity":
                    eval_expr = _wrap_phi_with_inverse_transform(phi_expr_str, y_transform)
                else:
                    eval_expr = phi_expr_str
            if eval_expr:
                new_truth = _evaluate_with_coefficients(eval_expr)
                if new_truth is not None:
                    truth = new_truth

        # Also re-compute if truth_eval failed due to unknown function (e.g., negexp)
        error_msg = truth.get("error_message") or ""
        normalization_expr = selected_expr or y_expr_str
        if (
            selection_eligible
            and "negexp" in error_msg
            and normalization_expr is not None
        ):
            # Normalize expression and re-compute truth_eval
            normalized_expr = _normalize_expression(normalization_expr)
            if normalized_expr != normalization_expr:
                new_truth = _evaluate_with_coefficients(normalized_expr)
                if new_truth is not None:
                    truth = new_truth

        # Compute RMS from val_loss with fallback chain:
        # 1. JSON report (set during Stage A if separability found)
        # 2. pkl checkpoint file (authoritative, saved by run_SR.py)
        # 3. Log file parsing (legacy fallback)
        stageA_val_loss = stageA.get("val_loss")
        if stageA_val_loss is None:
            stageA_val_loss = load_stageA_val_loss_from_pkl(path)
        if stageA_val_loss is None:
            stageA_val_loss = parse_stageA_val_loss_from_log(path)
        stageA_rms = math.sqrt(stageA_val_loss) if stageA_val_loss is not None else None

        stageB_val_loss = stageB.get("val_loss")
        stageB_rms = math.sqrt(stageB_val_loss) if stageB_val_loss is not None else None

        # Compute expression: prefer y_expr_str, else wrap phi_expr_str with inverse transform
        y_expr = selected_expr or stageC.get("y_expr_str")
        if y_expr is None and phi_expr_str is not None:
            if y_transform != "identity":
                y_expr = _wrap_phi_with_inverse_transform(phi_expr_str, y_transform)
            else:
                y_expr = phi_expr_str
        # Normalize expression for display (convert negexp -> -exp, etc.)
        y_expr = _normalize_expression(y_expr)

        # Count number of input variables from expression
        n_inputs = _count_inputs(y_expr or phi_expr_str)

        # Count significant parameters (numeric constants above threshold)
        n_sig_params = _count_significant_params(y_expr or phi_expr_str, sig_threshold)

        # Determine if Stage A found separability
        # 1. Try loading from checkpoint (authoritative)
        # 2. Fall back to inferring from ast_human
        separability = load_stageA_separability_from_pkl(path)
        if separability is None:
            ast_human = stageA.get("ast_human")
            separability = infer_separability_from_ast(ast_human)

        # Look up ground truth equation by problem ID
        problem_id = extract_problem_id(problem)
        gt_expr = ground_truth.get(problem_id, "-") if problem_id else "-"
        noise_level = _infer_noise_level(metadata, results_dir)

        truth_success = truth.get("success") if selection_eligible else False
        truth_rmse_rel = truth.get("rmse_rel")
        if truth_success is False:
            truth_exact = False
        elif truth_success is True and truth_rmse_rel is not None:
            if float(truth_rmse_rel) <= exact_rmse_rel:
                truth_exact = True
            elif noise_level == 0.0:
                truth_exact = False
            else:
                # A strict miss is not a failed recovery when observations are
                # noisy (or their noise level is unknown).
                truth_exact = None
        else:
            truth_exact = None
        row = {
            "problem": problem,
            "n_inputs": n_inputs,
            "separability": separability,
            "y_transform": stageA.get("y_transform", "-"),
            "ground_truth": gt_expr,
            "noise_level": noise_level,
            "expression": y_expr or "-",
            "stageA_rms": stageA_rms,
            "stageB_rms": stageB_rms,
            "stageB_params": stageB.get("params"),
            "stageB_params_sig": n_sig_params,
            "patterns": ", ".join(stageB.get("enabled_patterns") or []),
            "truth_success": truth_success,
            "truth_exact": truth_exact,
            "truth_rmse_abs": truth.get("rmse_abs"),
            "truth_rmse_rel": truth_rmse_rel,
            "final_selection_eligible": selection_eligible,
            "final_selection_ineligible_reason": selection_reason,
            "walltime_hrs": metadata.get("walltime_hours"),
        }
        reports.append(row)

    return reports


def print_table(rows: list[dict], max_expr_len: int = 60, max_gt_len: int = 85):
    """Print a formatted table to stdout."""
    if not rows:
        print("No results found.")
        return

    # Header
    print(f"{'Problem':<40} {'N':<3} {'Sep':<4} {'Y-Tf':<10} {'RMS_A':<12} {'RMS_B':<12} {'Params':<7} {'Sig':<4} "
          f"{'Eval':<5} {'Exact':<6} {'RMSE_abs':<12} {'Ground Truth':<{max_gt_len}} {'Expression'}")
    print("-" * (181 + max_gt_len + 1))

    for row in rows:
        expr = row["expression"] or "-"
        if len(expr) > max_expr_len:
            expr = expr[:max_expr_len-3] + "..."

        gt = row.get("ground_truth") or "-"
        if len(gt) > max_gt_len:
            gt = gt[:max_gt_len-3] + "..."

        truth_ok = "Yes" if row["truth_success"] else ("No" if row["truth_success"] is False else "-")
        exact_ok = "Yes" if row["truth_exact"] else ("No" if row["truth_exact"] is False else "-")
        n_inputs = row.get("n_inputs")
        n_str = str(n_inputs) if n_inputs is not None else "-"
        sep = row.get("separability")
        sep_str = "Yes" if sep else ("No" if sep is False else "-")

        sig_params = row.get("stageB_params_sig")
        sig_str = str(sig_params) if sig_params is not None else "-"

        print(f"{row['problem']:<40} "
              f"{n_str:<3} "
              f"{sep_str:<4} "
              f"{row['y_transform']:<10} "
              f"{format_float(row['stageA_rms']):<12} "
              f"{format_float(row['stageB_rms']):<12} "
              f"{row['stageB_params'] or '-':<7} "
              f"{sig_str:<4} "
              f"{truth_ok:<5} "
              f"{exact_ok:<6} "
              f"{format_float(row['truth_rmse_abs']):<12} "
              f"{gt:<{max_gt_len}} "
              f"{expr}")


def write_csv(rows: list[dict], output_path: Path):
    """Write results to CSV file."""
    import csv

    fieldnames = [
        "problem", "noise_level", "n_inputs", "separability", "y_transform", "stageA_rms", "stageB_rms", "stageB_params", "stageB_params_sig",
        "patterns", "truth_success", "truth_exact", "truth_rmse_abs", "truth_rmse_rel",
        "final_selection_eligible", "final_selection_ineligible_reason",
        "walltime_hrs", "ground_truth", "expression"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize AI Feynman benchmark results from .report.json files."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Directory containing .report.json files (default: results)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--expr-len",
        type=int,
        default=60,
        help="Max expression length in table output (default: 60)"
    )
    parser.add_argument(
        "--recompute-truth",
        action="store_true",
        help="Force truth_eval recomputation for the authoritative final expression."
    )
    parser.add_argument(
        "--sig-threshold",
        type=float,
        default=1e-6,
        help="Threshold for counting significant parameters (|value| > threshold, default: 1e-6)"
    )
    parser.add_argument(
        "--equations",
        type=str,
        default="data/equations.txt",
        help="Path to equations.txt file containing ground truth equations (default: data/equations.txt)"
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Only show cases that are not exact recoveries"
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        return 1

    # Load ground truth equations
    equations_path = Path(args.equations)
    ground_truth = load_ground_truth_equations(equations_path)
    if ground_truth:
        print(f"Loaded {len(ground_truth)} ground truth equations from {equations_path}")
    else:
        print(f"Warning: No ground truth equations loaded (file: {equations_path})")

    rows = summarize_results(results_dir, recompute_truth=args.recompute_truth, sig_threshold=args.sig_threshold,
                             ground_truth=ground_truth)
    print(f"Found {len(rows)} results in {results_dir}")

    # Filter to failed cases if requested
    if args.failed:
        rows = [
            r for r in rows
            if r["truth_success"] is False or r["truth_exact"] is False
        ]
        print(f"Showing {len(rows)} failed cases\n")
    else:
        print()

    print_table(rows, max_expr_len=args.expr_len)

    if args.csv:
        print()
        write_csv(rows, Path(args.csv))

    # Summary statistics
    n_eval_success = sum(1 for r in rows if r["truth_success"] is True)
    n_eval_fail = sum(1 for r in rows if r["truth_success"] is False)
    n_exact = sum(1 for r in rows if r["truth_exact"] is True)
    n_noiseless_miss = sum(
        1 for r in rows
        if r["truth_success"] is True and r["truth_exact"] is False
    )
    n_noise_indeterminate = sum(
        1 for r in rows
        if r["truth_success"] is True and r["truth_exact"] is None
    )
    print(f"\nSummary: {n_exact} exact, {n_noiseless_miss} noiseless not exact, "
          f"{n_noise_indeterminate} noisy/unknown indeterminate, {n_eval_fail} evaluation failed, "
          f"{len(rows) - n_eval_success - n_eval_fail} unknown")

    return 0


if __name__ == "__main__":
    exit(main())
