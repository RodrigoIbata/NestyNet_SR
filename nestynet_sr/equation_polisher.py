# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Post-hoc denoising simplifier for final symbolic-regression expressions.

This module is deliberately small and artifact-aware.  It does not run another
symbolic-regression search.  It reads the final expression plus optional
NestyNet_SR artifacts, uses those artifacts as hints, generates nearby cleaner
forms, scores them on data, and reports a loss-complexity Pareto frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is a declared dependency.
    pd = None

try:
    import sympy as sp
except Exception:  # pragma: no cover - tested environments have sympy.
    sp = None

from nestynet_sr.sr_core.atoms import _enumerate_exponents, _eval_monomials
from nestynet_sr.sr_core.coefficient_metadata import (
    coefficient_symbol_values,
    normalize_coefficient_metadata,
    validate_coefficient_symbol,
)
from nestynet_sr.sr_core.numerics import ridge_lstsq
from nestynet_sr.sr_core.sympy_units import check_sympy_units
from nestynet_sr.sr_search.fitting_utils import _fit_poly_coeffs_1d
from nestynet_sr.sr_search.model_selection import pareto_front_indices_2d
from nestynet_sr.sr_search.rational_sparsify import (
    DEFAULT_RAT_STLSQ_CFG,
    DEFAULT_POLY_STLSQ_CFG,
    stlsq_sparsify_rational_coeffs,
    stlsq_sparsify_poly_coeffs,
)
from nestynet_sr.sr_search.polish_utils import (
    canonicalize_trig_phases,
    constant_code_cost as _shared_constant_code_cost,
    expr_depth as _shared_expr_depth,
    final_polish_snap_targets,
    numeric_constant_snap_candidates,
    rational_snap_targets,
    rationalize_float_exponents,
    snap_numeric_constants,
)

try:
    from nestynet_sr.sr_search.representation import (
        AIF_CONSTS,
        _canonicalize_inverse_ratio_powers,
        _infer_sympy_symbol_assumptions_from_samples,
        _nsimplify_compat,
        _prune_tiny_additive_constants,
        aggressive_simplify,
        sympy_timeout,
    )
except Exception:  # pragma: no cover - fallback keeps the module usable.
    AIF_CONSTS = []
    aggressive_simplify = None
    sympy_timeout = None
    _nsimplify_compat = None
    _canonicalize_inverse_ratio_powers = lambda expr: expr
    _prune_tiny_additive_constants = lambda expr, tol=None: expr

    def _infer_sympy_symbol_assumptions_from_samples(_x_col):
        return {"real": True}


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass
class CoefficientPruneHint:
    param_name: str
    index: int
    target: float = 0.0
    value: Optional[float] = None
    significance: Optional[float] = None
    accepted: bool = False
    source: str = ""


@dataclass
class ArtifactHints:
    dataset_path: Optional[str] = None
    y_transform: Optional[str] = None
    seed_expr: Optional[str] = None
    y_expr: Optional[str] = None
    phi_expr: Optional[str] = None
    stageA_expr: Optional[str] = None
    stageB_expr: Optional[str] = None
    simplification_path: list[dict[str, Any]] = field(default_factory=list)
    accepted_patterns: list[str] = field(default_factory=list)
    compound_exprs: list[str] = field(default_factory=list)
    candidate_family_hints: list[dict[str, Any]] = field(default_factory=list)
    coefficient_prune_hints: list[CoefficientPruneHint] = field(default_factory=list)
    variable_assumptions: dict[str, str] = field(default_factory=dict)
    prior_losses: dict[str, float] = field(default_factory=dict)
    initial_sqrt_poly_coeffs: list[float] = field(default_factory=list)
    coefficient_metadata: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["coefficient_prune_hints"] = [
            asdict(h) for h in self.coefficient_prune_hints
        ]
        return out


@dataclass
class PolishConfig:
    max_candidates: int = 256
    max_beam_width: int = 32
    max_seconds: float = 30.0
    snap_sensitivity_k: float = 2.0
    seed_functional_nrmse_max: float = 0.10
    epsilon_pareto_k: float = 1.0
    loss_equiv_abs_floor: float = 1.0e-24
    noise_floor_raw: float = 0.0
    bootstrap_reps: int = 0
    min_valid_fraction: float = 0.80
    positive_var_detection: bool = True
    use_artifact_hints: bool = True
    enable_noisy_sparse_rational_refit: bool = False
    drop_rel_tol: float = 1.0e-3
    enable_drop_addend_refit: bool = True
    drop_refit_max_sites: int = 8
    drop_refit_max_rounds: int = 3
    drop_refit_max_refits: int = 16
    drop_refit_max_seconds: float = 6.0
    drop_refit_noise_sigma_mult: float = 3.0
    drop_refit_resid_mult: float = 3.0
    drop_refit_site_rel_ratio_max: float = 5.0e-2
    drop_refit_shift_guard_factor: float = 10.0
    drop_refit_max_params: int = 12
    drop_refit_large_number_threshold: int = 10_000
    snap_rel_tol: float = 5.0e-3
    ridge: float = 1.0e-10
    val_fraction: float = 0.2
    seed: int = 1234
    symbol_values: dict[str, float] = field(default_factory=dict)

    def snap_targets(self) -> list[Any]:
        if sp is None:
            return [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
        return final_polish_snap_targets()


@dataclass
class CandidateSpec:
    expr: Any
    label: str
    n_free_params: int = 0
    n_snapped_consts: int = 0
    assumptions: list[str] = field(default_factory=list)
    rewrite_trace: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    # Free-parameter count under the frozen-selection contract.  None means
    # "same as n_free_params".  Proposal batteries that fit literals on search
    # data and then freeze them (like Stage B/C fitted constants) declare 0 so
    # the statistical-selection archive does not charge frozen literals as
    # free parameters for one source while sibling sources declare 0.
    selection_n_free_params: Optional[int] = None


@dataclass
class CandidateRecord:
    expr: str
    display_expr: str
    label: str
    train_mse: float
    val_mse: float
    val_mse_se: float
    complexity: float
    structural_complexity: float
    coefficient_complexity: float
    n_free_params: int
    n_snapped_consts: int
    frac_valid: float
    seed_nrmse: Optional[float]
    assumptions: list[str]
    source_hints: list[str]
    rewrite_trace: list[str]
    distance_from_seed: Optional[float]
    is_strict_pareto: bool = False
    is_epsilon_pareto: bool = False
    is_recommended: bool = False
    full_dataset_mse: Optional[float] = None
    full_dataset_mse_se: Optional[float] = None
    full_dataset_frac_valid: Optional[float] = None
    full_dataset_snap_selected: bool = False
    selection_n_free_params: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolishResult:
    seed_expr: str
    all_candidates: list[CandidateRecord]
    strict_pareto: list[CandidateRecord]
    epsilon_pareto: list[CandidateRecord]
    recommended: Optional[CandidateRecord]
    rewrite_trace: list[str]
    warnings: list[str]
    artifact_hints: Optional[ArtifactHints] = None
    seed_baseline: Optional[CandidateRecord] = None
    seed_units_ok: Optional[bool] = None
    seed_units_reason: Optional[str] = None
    selection_status: str = "selected"
    selection_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_expr": self.seed_expr,
            "all_candidates": [c.to_dict() for c in self.all_candidates],
            "strict_pareto": [c.to_dict() for c in self.strict_pareto],
            "epsilon_pareto": [c.to_dict() for c in self.epsilon_pareto],
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "rewrite_trace": list(self.rewrite_trace),
            "warnings": list(self.warnings),
            "artifact_hints": self.artifact_hints.to_dict()
            if self.artifact_hints is not None
            else None,
            "seed_baseline": (
                self.seed_baseline.to_dict() if self.seed_baseline is not None else None
            ),
            "seed_units_ok": self.seed_units_ok,
            "seed_units_reason": self.seed_units_reason,
            "selection_status": self.selection_status,
            "selection_reason": self.selection_reason,
        }


@dataclass
class InverseSqrtPolyTemplate:
    prefactor: Any
    z_expr: Any
    coeffs: list[float]
    assumptions: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)


@dataclass
class ExpPolyTemplate:
    variable: Any
    coeffs: list[float]
    assumptions: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", str(text))


def _normalize_text_expr(text: str) -> str:
    out = _strip_ansi(str(text)).strip()
    out = out.replace("^", "**")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _clean_expr_field(value: Any) -> Optional[str]:
    """Normalize artifact expression fields and discard textual null sentinels."""
    if value is None:
        return None
    text = _normalize_text_expr(str(value))
    if not text:
        return None
    if text.lower() in {"none", "null", "nan", "<none>", "n/a"}:
        return None
    return text


def _read_text(path: Optional[str | Path]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Optional[str | Path]) -> Any:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _unique_append(seq: list[str], value: Optional[str]) -> None:
    val = _clean_expr_field(value)
    if not val:
        return
    if val not in seq:
        seq.append(val)


def _sympy_locals(variable_names: Sequence[str]) -> dict[str, Any]:
    if sp is None:
        return {}
    loc = {
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "tanh": sp.tanh,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "exp": sp.exp,
        "log": sp.log,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "arcsinh": sp.asinh,
        "arccosh": sp.acosh,
        "arctanh": sp.atanh,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    for name in variable_names:
        loc[str(name)] = sp.Symbol(str(name), real=True)
    return loc


_SUPPORTED_ARTIFACT_CALLS = {
    "sqrt",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sinh",
    "cosh",
    "exp",
    "log",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "arcsin",
    "arccos",
    "arctan",
    "arcsinh",
    "arccosh",
    "arctanh",
    "Abs",
}


def has_unsupported_artifact_call(expr: str) -> bool:
    """Return true for Stage-B internal placeholders that are not equations.

    Simplification-path entries can contain visible internal atoms such as
    ``scale()``, ``poly(...)`` or ``NN[...]``.  They are useful diagnostics but
    are not standalone SymPy equations; parsing ``sqrt(poly(...))`` lets SymPy
    interpret ``poly`` as ``Poly`` and emits a deprecation warning.
    """
    text = _normalize_text_expr(str(expr))
    if re.search(r"\b(?:NN|nn)\s*[\[(]", text):
        return True
    for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", text):
        if name not in _SUPPORTED_ARTIFACT_CALLS:
            return True
    return False


# Private compatibility alias for existing internal callers.
_has_unsupported_artifact_call = has_unsupported_artifact_call


def infer_variable_names(expr: Optional[str], X: Optional[np.ndarray] = None) -> list[str]:
    n = 0
    if X is not None and getattr(X, "ndim", 0) >= 2:
        n = max(n, int(X.shape[1]))
    if expr:
        for m in re.findall(r"\bx(\d+)\b", str(expr)):
            n = max(n, int(m) + 1)
    return [f"x{i}" for i in range(n)]


def parse_sympy_expr(expr: str | Any, variable_names: Sequence[str]):
    if sp is None:
        raise RuntimeError("SymPy is required for equation_polisher")
    if isinstance(expr, sp.Basic):
        return expr
    text = _normalize_text_expr(str(expr))
    return sp.sympify(text, locals=_sympy_locals(variable_names))


def _sstr(expr: Any) -> str:
    if sp is not None and isinstance(expr, sp.Basic):
        return sp.sstr(expr)
    return str(expr)


def _canonical_key(expr: Any) -> str:
    try:
        return sp.srepr(expr)
    except Exception:
        return str(expr)


def _sympy_expr_units_check(
    expr: Any,
    variable_names: Sequence[str],
    units_spec: Any,
) -> tuple[bool, str]:
    """Compatibility wrapper around the shared structured checker."""

    result = check_sympy_units(
        expr,
        variable_names,
        units_spec,
        expression_space="y",
    )
    if result.ok:
        return True, "units-ok" if result.checked else "units-unchecked"
    legacy_code = {
        "add_dimension_mismatch": "add-dim-mismatch",
        "target_dimension_mismatch": "target-dim-mismatch",
        "function_argument_not_dimensionless": "function-arg-not-dimless",
        "unknown_symbol": "unknown-symbol",
        "unsupported_node": "unsupported-units-node",
        "units_spec_error": "units-spec-error",
    }.get(result.code, result.code.replace("_", "-"))
    return False, f"{legacy_code}:{result.reason}"


def _balanced_call_contents(text: str, name: str) -> list[str]:
    out: list[str] = []
    token = f"{name}("
    start = 0
    while True:
        i = text.find(token, start)
        if i < 0:
            break
        j = i + len(token)
        depth = 1
        k = j
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        if depth == 0:
            out.append(text[j : k - 1])
        start = max(k, i + 1)
    return out


def _normalize_compound_expr(expr: str) -> Optional[str]:
    if sp is None:
        return _normalize_text_expr(expr)
    text = _normalize_text_expr(expr)
    while text.startswith("(") and text.endswith(")"):
        try:
            parsed = parse_sympy_expr(text, infer_variable_names(text))
            return sp.sstr(parsed)
        except Exception:
            text = text[1:-1].strip()
    try:
        parsed = parse_sympy_expr(text, infer_variable_names(text))
        return sp.sstr(parsed)
    except Exception:
        return text or None


def _extract_compounds_from_expr_text(text: Optional[str]) -> list[str]:
    if not text:
        return []
    raw = _strip_ansi(text)
    found: list[str] = []
    for content in _balanced_call_contents(raw, "poly"):
        _unique_append(found, _normalize_compound_expr(content))
    for content in _balanced_call_contents(raw, "NN"):
        if "x" in content and any(op in content for op in ("*", "/", "**", "^", "+", "-")):
            _unique_append(found, _normalize_compound_expr(content))
    return found


def _parse_float_list(text: str) -> list[float]:
    vals = []
    for m in FLOAT_RE.findall(text):
        try:
            vals.append(float(m))
        except Exception:
            pass
    return vals


def load_artifact_hints(
    *,
    report_json: Optional[str | Path] = None,
    decisions_json: Optional[str | Path] = None,
    allstages_log: Optional[str | Path] = None,
    path_file: Optional[str | Path] = None,
    final_human: Optional[str | Path] = None,
) -> ArtifactHints:
    """Load post-polishing hints from NestyNet_SR run artifacts."""
    hints = ArtifactHints()

    report = _load_json(report_json)
    if isinstance(report, dict):
        meta = report.get("metadata") or {}
        hints.dataset_path = meta.get("dataset") or hints.dataset_path
        stage_a = report.get("stageA") or {}
        stage_b = report.get("stageB") or {}
        stage_c = report.get("stageC") or {}
        hints.y_transform = stage_a.get("y_transform")
        hints.stageA_expr = stage_a.get("ast_human")
        hints.stageB_expr = stage_b.get("ast_human")
        hints.phi_expr = _clean_expr_field(stage_c.get("phi_expr_str"))
        hints.y_expr = _clean_expr_field(stage_c.get("y_expr_str"))
        hints.coefficient_metadata = stage_c.get("coefficient_metadata")
        if hints.coefficient_metadata is None:
            hints.coefficient_metadata = stage_b.get("coefficient_metadata")
        hints.seed_expr = hints.y_expr or hints.phi_expr
        hints.simplification_path = list(report.get("simplification_path") or [])
        for pat in stage_b.get("enabled_patterns") or []:
            _unique_append(hints.accepted_patterns, str(pat))
        for expr_text in (hints.stageA_expr, hints.stageB_expr, hints.phi_expr, hints.y_expr):
            for c in _extract_compounds_from_expr_text(expr_text):
                _unique_append(hints.compound_exprs, c)
        if stage_a.get("val_loss") is not None:
            hints.prior_losses["stageA_val_loss"] = float(stage_a["val_loss"])
        if stage_b.get("val_loss") is not None:
            hints.prior_losses["stageB_val_loss"] = float(stage_b["val_loss"])

    decisions = _load_json(decisions_json)
    if isinstance(decisions, list):
        for rec in decisions:
            if not isinstance(rec, dict):
                continue
            label = rec.get("label")
            if rec.get("outcome") == "accept" and label:
                _unique_append(hints.accepted_patterns, str(label))
            snapshot = rec.get("ast_snapshot")
            for c in _extract_compounds_from_expr_text(snapshot):
                _unique_append(hints.compound_exprs, c)
            if label and any(k in str(label) for k in ("sqrt_poly", "inv_poly", "ratpoly", "exp", "log")):
                hints.candidate_family_hints.append(
                    {
                        "label": label,
                        "outcome": rec.get("outcome"),
                        "loss": rec.get("cand_loss"),
                        "complexity": rec.get("cand_complexity_total"),
                        "source": "decisions_json",
                    }
                )

    path_text = _read_text(path_file)
    if path_text:
        m = re.search(r"=== Final:\s*(.*?)\s*===", path_text, flags=re.S)
        if m and hints.seed_expr is None:
            hints.seed_expr = _clean_expr_field(m.group(1))
        for c in _extract_compounds_from_expr_text(path_text):
            _unique_append(hints.compound_exprs, c)

    final_text = _read_text(final_human)
    if final_text:
        m = re.search(r"Expression \(y-space\):\s*(.+)", final_text)
        if m and hints.y_expr is None:
            hints.y_expr = _clean_expr_field(m.group(1))
        m = re.search(r"Expression \((?:φ|phi|Phi)-space\):\s*(.+)", final_text)
        if m and hints.phi_expr is None:
            hints.phi_expr = _clean_expr_field(m.group(1))
        if hints.seed_expr is None:
            hints.seed_expr = hints.y_expr or hints.phi_expr
        for c in _extract_compounds_from_expr_text(final_text):
            _unique_append(hints.compound_exprs, c)

    log_text = _read_text(allstages_log)
    if log_text:
        _parse_log_hints(log_text, hints)

    return hints


def _parse_log_hints(log_text: str, hints: ArtifactHints) -> None:
    lines = [_strip_ansi(line) for line in log_text.splitlines()]
    pending_zero_idx: Optional[int] = None
    param_by_idx: dict[int, CoefficientPruneHint] = {}

    for line in lines:
        m = re.search(
            r"\[Compound\].*?z=(.*?)(?:\s+\(pattern=|,\s*extras=|,\s*val-loss|$)",
            line,
        )
        if m:
            _unique_append(hints.compound_exprs, _normalize_compound_expr(m.group(1)))

        m = re.search(r"\[Stage B\]\s+([^\s]+)\s+\(kind=ratio,\s*indices=\((\d+),\s*(\d+)\)", line)
        if m:
            _unique_append(hints.compound_exprs, _normalize_compound_expr(m.group(1)))

        m = re.search(r"Accepted rewrite \(([^,\)]+)", line)
        if m:
            _unique_append(hints.accepted_patterns, m.group(1))

        m = re.search(r"Accepted patterns:\s*(.*)", line)
        if m:
            for pat in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1)):
                if pat not in {"Stage", "B", "Accepted", "patterns"}:
                    _unique_append(hints.accepted_patterns, pat)

        m = re.search(r"leaf\[\d+\]\s+coeffs:\s*\[([^\]]+)\]", line)
        if m:
            coeffs = _parse_float_list(m.group(1))
            if coeffs:
                hints.initial_sqrt_poly_coeffs = coeffs

        m = re.search(r"SymPy variable assumptions from data:\s*(.*)", line)
        if m:
            for part in m.group(1).split(","):
                part = part.strip()
                mm = re.match(r"(x\d+)\s*([<>]=?0)", part)
                if mm:
                    hints.variable_assumptions[mm.group(1)] = mm.group(2)

        m = re.search(
            r"Param\s+(coeffs(?:_[a-z]+)?)\[(\d+)\].*?val=([\-+0-9.eE]+),\s*sig=([\-+0-9.eE]+)",
            line,
        )
        if m:
            idx = int(m.group(2))
            hint = CoefficientPruneHint(
                param_name=m.group(1),
                index=idx,
                value=float(m.group(3)),
                significance=float(m.group(4)),
                source="allstages_log",
            )
            param_by_idx[idx] = hint

        m = re.search(r"Zeroed\s+(coeffs(?:_[a-z]+)?)\[(\d+)\]", line)
        if m:
            pending_zero_idx = int(m.group(2))

        if pending_zero_idx is not None and "ACCEPTED" in line:
            hint = param_by_idx.get(
                pending_zero_idx,
                CoefficientPruneHint(
                    param_name="coeffs",
                    index=pending_zero_idx,
                    source="allstages_log",
                ),
            )
            hint.accepted = True
            if not any(
                h.param_name == hint.param_name and h.index == hint.index
                for h in hints.coefficient_prune_hints
            ):
                hints.coefficient_prune_hints.append(hint)
            pending_zero_idx = None


def load_csv_data(
    filepath: str | Path,
    *,
    target_col: str = "y",
    val_fraction: float = 0.2,
    seed: int = 1234,
    max_rows: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load a CSV and return train/validation arrays."""
    if pd is None:
        raise RuntimeError("pandas is required to load CSV data")
    df = pd.read_csv(filepath)
    if max_rows is not None and int(max_rows) > 0 and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not found in {filepath}")
    var_cols = [c for c in df.columns if c != target_col]
    var_cols.sort(key=lambda s: (0, int(s[1:])) if s.startswith("x") and s[1:].isdigit() else (1, s))
    X = df[var_cols].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64).reshape(-1)
    n = int(len(df))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_val = max(1, int(round(float(val_fraction) * n)))
    n_val = min(n - 1, n_val) if n > 1 else 1
    va = perm[:n_val]
    tr = perm[n_val:] if n > 1 else perm
    return X[tr], y[tr], X[va], y[va], var_cols


def _build_symbol_assumptions(
    X: np.ndarray,
    variable_names: Sequence[str],
    explicit: Optional[Mapping[str, str]] = None,
    *,
    detect_positive: bool = True,
) -> dict[str, str]:
    out: dict[str, str] = {}
    explicit = explicit or {}
    for j, name in enumerate(variable_names):
        if name in explicit:
            out[name] = str(explicit[name])
            continue
        if not detect_positive or j >= X.shape[1]:
            out[name] = "real"
            continue
        ass = _infer_sympy_symbol_assumptions_from_samples(X[:, j])
        if ass.get("positive"):
            out[name] = ">0"
        elif ass.get("nonnegative"):
            out[name] = ">=0"
        elif ass.get("negative"):
            out[name] = "<0"
        elif ass.get("nonpositive"):
            out[name] = "<=0"
        else:
            out[name] = "real"
    return out


def _parse_with_assumptions(expr: str | Any, variable_names: Sequence[str], assumptions: Mapping[str, str]):
    if sp is None:
        raise RuntimeError("SymPy is required for equation_polisher")
    loc = _sympy_locals(variable_names)
    for name in variable_names:
        label = assumptions.get(name, "real")
        kwargs = {"real": True}
        if label == ">0":
            kwargs = {"positive": True}
        elif label == ">=0":
            kwargs = {"nonnegative": True}
        elif label == "<0":
            kwargs = {"negative": True}
        elif label == "<=0":
            kwargs = {"nonpositive": True}
        loc[name] = sp.Symbol(name, **kwargs)
    if isinstance(expr, sp.Basic):
        return expr.xreplace({sp.Symbol(n): loc[n] for n in variable_names})
    return sp.sympify(_normalize_text_expr(str(expr)), locals=loc)


def _eval_expr_array(
    expr: Any,
    X: np.ndarray,
    variable_names: Sequence[str],
    symbol_values: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    if sp is None:
        raise RuntimeError("SymPy is required for equation_polisher")
    variable_set = {str(name) for name in variable_names}
    values: dict[str, float] = {}
    for raw_name, raw_value in (symbol_values or {}).items():
        name = validate_coefficient_symbol(raw_name)
        if name in variable_set:
            raise ValueError(
                f"coefficient symbol {name!r} collides with an input variable"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"coefficient symbol {name!r} has non-finite value")
        values[name] = value
    expr_plain = expr.xreplace({s: sp.Symbol(str(s), real=True) for s in expr.free_symbols})
    syms = [sp.Symbol(name, real=True) for name in variable_names]
    substitutions = {
        symbol: sp.Float(repr(values[str(symbol)]), 17)
        for symbol in expr_plain.free_symbols
        if str(symbol) in values
    }
    expr_numeric = expr_plain.xreplace(substitutions)
    missing = sorted(
        str(symbol)
        for symbol in expr_numeric.free_symbols
        if str(symbol) not in variable_set
    )
    if missing:
        raise ValueError(
            "missing coefficient values for symbols: " + ", ".join(missing)
        )
    fn = sp.lambdify(syms, expr_numeric, modules=["numpy"])
    args = [X[:, j] for j in range(len(variable_names))]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        y = np.asarray(fn(*args), dtype=np.float64)
    if y.shape == ():
        y = np.full(X.shape[0], float(y), dtype=np.float64)
    y = np.ravel(y)
    if y.size == 1 and X.shape[0] != 1:
        y = np.full(X.shape[0], float(y[0]), dtype=np.float64)
    return y


def _mse_and_se(pred: np.ndarray, y: np.ndarray, min_valid_fraction: float) -> tuple[float, float, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(pred.size, y.size)
    pred = pred[:n]
    y = y[:n]
    valid = np.isfinite(pred) & np.isfinite(y)
    frac = float(np.count_nonzero(valid)) / max(1, n)
    if frac < float(min_valid_fraction) or np.count_nonzero(valid) == 0:
        return float("inf"), float("inf"), frac
    losses = (pred[valid] - y[valid]) ** 2
    mse = float(np.mean(losses))
    if losses.size > 1:
        se = float(np.std(losses, ddof=1) / math.sqrt(losses.size))
    else:
        se = 0.0
    return mse, se, frac


def _robust_scale(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 1.0
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    if math.isfinite(mad) and mad > 1e-15:
        return mad
    std = float(np.std(y))
    if math.isfinite(std) and std > 1e-15:
        return std
    return max(1.0, abs(med))


def _seed_distance(pred: np.ndarray, seed_pred: Optional[np.ndarray]) -> Optional[float]:
    if seed_pred is None:
        return None
    n = min(len(pred), len(seed_pred))
    if n == 0:
        return None
    a = np.asarray(pred[:n], dtype=np.float64)
    b = np.asarray(seed_pred[:n], dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) == 0:
        return None
    rms = float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))
    scale = _robust_scale(b[valid])
    return rms / max(scale, 1e-30)


def _expr_depth(expr: Any) -> int:
    return _shared_expr_depth(expr)


def _constant_code_cost(expr: Any, config: PolishConfig) -> tuple[float, int]:
    if sp is None:
        return 0.0, 0
    return _shared_constant_code_cost(
        expr,
        snap_targets=config.snap_targets(),
        snap_rel_tol=float(config.snap_rel_tol),
    )


def _inferred_learned_constant_count(
    expr: Any,
    coefficient_metadata: Optional[Mapping[str, Any]],
) -> int:
    """Infer fitted scalar degrees of freedom visible in ``expr``.

    Named coefficient metadata is authoritative.  Anonymous fitted scales are
    commonly folded into a literal by the Stage-B printer, so any remaining
    SymPy ``Float`` that is not identified as a fixed metadata value is treated
    conservatively as one learned scalar.  Exact integers, rationals, and
    symbolic constants such as ``pi`` contain no ``Float`` atoms and therefore
    carry no inferred fitted-parameter cost.
    """
    if sp is None:
        return 0
    ex = expr if isinstance(expr, sp.Basic) else sp.sympify(str(expr))
    float_atoms = sorted(ex.atoms(sp.Float), key=sp.srepr)
    matched_float_ids: set[int] = set()
    learned_identities: set[str] = set()
    free_symbols = {str(symbol) for symbol in ex.free_symbols}

    records = []
    if isinstance(coefficient_metadata, Mapping):
        records = list(coefficient_metadata.get("records", []) or [])
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        identity = str(record.get("identity") or f"record:{index}")
        trainable = bool(record.get("trainable", False))
        symbol = record.get("symbol")
        if symbol and str(symbol) in free_symbols:
            if trainable:
                learned_identities.add(identity)
            continue

        value = record.get("value")
        if value is None:
            continue
        try:
            value_float = float(value)
        except Exception:
            continue
        for atom_index, atom in enumerate(float_atoms):
            if atom_index in matched_float_ids:
                continue
            if math.isclose(
                float(atom),
                value_float,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                matched_float_ids.add(atom_index)
                if trainable:
                    learned_identities.add(identity)
                break

    anonymous_float_values = {
        sp.srepr(atom)
        for atom_index, atom in enumerate(float_atoms)
        if atom_index not in matched_float_ids
    }
    return int(len(learned_identities) + len(anonymous_float_values))


def expression_cost_components(
    expr: str | Any,
    config: Optional[PolishConfig] = None,
    *,
    n_free_params: int = 0,
) -> tuple[float, float, float]:
    """Return ``(total, structural, coefficient)`` display costs."""
    config = config or PolishConfig()
    if sp is None:
        structural = float(len(str(expr))) + 4.0 * n_free_params
        return structural, structural, 0.0
    ex = expr if isinstance(expr, sp.Basic) else parse_sympy_expr(str(expr), infer_variable_names(str(expr)))
    try:
        ops = float(sp.count_ops(ex, visual=False))
    except Exception:
        ops = float(len(sp.sstr(ex)))
    structural = float(ops + 0.5 * _expr_depth(ex) + 4.0 * int(n_free_params))
    const_cost, n_long = _constant_code_cost(ex, config)
    coefficient = float(const_cost + 4.0 * int(n_long))
    return float(structural + coefficient), structural, coefficient


def expression_complexity(expr: str | Any, config: Optional[PolishConfig] = None, *, n_free_params: int = 0) -> float:
    total, _structural, _coefficient = expression_cost_components(
        expr,
        config,
        n_free_params=n_free_params,
    )
    return float(total)


def _rationalize_float_exponents(expr: Any) -> Any:
    if sp is None:
        return expr
    return rationalize_float_exponents(expr)


def _canonicalize_guarded_candidate_expr(expr: Any, config: PolishConfig) -> Any:
    try:
        expr = _canonicalize_inverse_ratio_powers(expr)
    except Exception:
        pass
    try:
        expr = canonicalize_trig_phases(expr, snap_rel_tol=config.snap_rel_tol)
    except Exception:
        pass
    return expr


def _denominator_coefficient_ratio_snap_specs(
    expr: Any,
    variable_names: Sequence[str],
    config: PolishConfig,
    *,
    max_ops: int = 128,
    max_terms: int = 24,
    max_anchors: int = 6,
    max_candidates: int = 12,
    large_rational_threshold: int = 10_000,
) -> list[CandidateSpec]:
    """Propose small-rational snaps hidden by a polynomial denominator scale.

    Stage-B rational printers can clear fitted denominators and thereby turn a
    nearly simple ratio such as ``2.0000000026`` into two apparently exact large
    integers.  SymPy must preserve those integers, so ordinary ``simplify`` and
    expression-wide ``nsimplify`` cannot recover the intended small constant.

    This proposal pass makes only a bounded, truth-blind normalization:

    1. require a small polynomial denominator with rational coefficients;
    2. divide numerator and denominator by one existing denominator coefficient;
    3. snap only the exposed denominator coefficient ratios to small rationals;
    4. keep only candidates whose symbolic complexity strictly decreases.

    Candidates still pass through the common units, validation-loss, and Pareto
    gates in :func:`polish_expression`.  Denominators containing symbols other
    than the declared input variables are skipped so named/unitful constants are
    never rewritten by this anonymous-literal codec repair.
    """

    if sp is None or int(max_candidates) <= 0:
        return []
    try:
        if int(sp.count_ops(expr, visual=False)) > int(max_ops):
            return []
        # Avoid paying for together/cancel on ordinary expressions.  This
        # repair targets the large exact integers/rationals introduced when a
        # fitted rational leaf clears its coefficient denominators.
        _raw_num, raw_den = sp.fraction(expr)
        exact_rationals = raw_den.atoms(sp.Rational)
        if not any(
            max(
                abs(int(value.p)),
                abs(int(value.q)),
            )
            >= int(large_rational_threshold)
            for value in exact_rationals
        ):
            return []
        num, den = sp.fraction(sp.cancel(sp.together(expr)))
        if den == 1:
            return []

        by_name = {str(sym): sym for sym in getattr(expr, "free_symbols", set())}
        variables = [
            by_name[str(name)]
            for name in variable_names
            if str(name) in by_name and by_name[str(name)] in den.free_symbols
        ]
        if not variables:
            return []
        if set(den.free_symbols) - set(variables):
            return []

        den_poly = sp.Poly(sp.expand(den), *variables)
        terms = den_poly.terms()
        if len(terms) < 2 or len(terms) > int(max_terms):
            return []
        coeffs = list(den_poly.coeffs())
        if not coeffs or any(
            coeff == 0 or not isinstance(coeff, sp.Rational) for coeff in coeffs
        ):
            return []
    except Exception:
        return []

    base_complexity = expression_complexity(expr, config)
    targets = rational_snap_targets(max_denominator=8)
    anchors: list[Any] = []
    seen_anchors: set[str] = set()
    for coeff in coeffs:
        key = sp.srepr(coeff)
        if key in seen_anchors:
            continue
        seen_anchors.add(key)
        anchors.append(coeff)
        if len(anchors) >= int(max_anchors):
            break

    out: list[CandidateSpec] = []
    seen_exprs: set[str] = set()
    base_key = _canonical_key(expr)
    for anchor in anchors:
        try:
            normalized_num = sp.cancel(num / anchor)
            normalized_den = sp.cancel(den / anchor)
            variants = numeric_constant_snap_candidates(
                normalized_den,
                snap_targets=targets,
                snap_rel_tol=float(config.snap_rel_tol),
                per_number=3,
            )
        except Exception:
            continue
        for snap_label, snapped_den in variants:
            try:
                candidate = sp.factor(sp.cancel(normalized_num / snapped_den))
                candidate = _canonicalize_guarded_candidate_expr(candidate, config)
                key = _canonical_key(candidate)
                if key == base_key or key in seen_exprs:
                    continue
                if expression_complexity(candidate, config) >= base_complexity:
                    continue
            except Exception:
                continue
            seen_exprs.add(key)
            out.append(
                CandidateSpec(
                    candidate,
                    label=f"denominator_coefficient_ratio_snap|{snap_label}",
                    n_snapped_consts=1,
                    rewrite_trace=[
                        "normalize polynomial denominator by an existing numeric coefficient",
                        f"small-rational denominator ratio snap: {snap_label}",
                    ],
                    source_hints=["denominator_coefficient_ratio_snap"],
                )
            )
            if len(out) >= int(max_candidates):
                return out
    return out


def _value_position_add_nodes(expr: Any) -> list[Any]:
    """Collect ``Add`` nodes reachable without entering a ``Pow`` exponent.

    Dropping an addend from a units-valid ``Add`` in value position cannot
    change the node's dimension (all addends share one dimension), so these are
    the only sites the drop-addend battery may touch.  Exponent-position Adds,
    ``Piecewise`` and relational subtrees are excluded.  First-visit preorder
    order keeps the site indexing deterministic.
    """

    out: list[Any] = []
    seen: set[str] = set()
    if sp is None or not isinstance(expr, sp.Basic):
        return out

    def walk(node: Any) -> None:
        if isinstance(node, (sp.Piecewise, sp.core.relational.Relational)):
            return
        if isinstance(node, sp.Add):
            key = sp.srepr(node)
            if key not in seen:
                seen.add(key)
                out.append(node)
        if isinstance(node, sp.Pow):
            walk(node.base)
            return
        for arg in node.args:
            walk(arg)

    walk(expr)
    return out


def _addend_numeric_coefficient(term: Any) -> tuple[Optional[float], Any, Any]:
    """Split ``term`` into (coefficient value, coefficient expr, rest).

    The numeric part is the product of every numeric factor in any
    representation (Float, large Integer, ``Rational*sqrt(2)``, ``1/(7*pi**3)``,
    ...), so previously snapped symbolic constants remain reachable.  Returns
    ``(None, 1, term)`` when the numeric part cannot be evaluated.
    """

    numeric_factors: list[Any] = []
    rest_factors: list[Any] = []
    for factor in sp.Mul.make_args(term):
        if bool(getattr(factor, "is_number", False)):
            numeric_factors.append(factor)
        else:
            rest_factors.append(factor)
    try:
        coeff = sp.Mul(*numeric_factors) if numeric_factors else sp.Integer(1)
        coeff_f = float(sp.N(coeff, 18))
    except Exception:
        return None, sp.Integer(1), term
    if not math.isfinite(coeff_f):
        return None, sp.Integer(1), term
    rest = sp.Mul(*rest_factors) if rest_factors else sp.Integer(1)
    return coeff_f, coeff, rest


def _has_large_exact_number(node: Any, threshold: int) -> bool:
    for value in node.atoms(sp.Rational):
        if max(abs(int(value.p)), abs(int(value.q))) >= int(threshold):
            return True
    return False


def _snap_fitted_coefficient(
    value: float,
    sigma: Optional[float],
    config: PolishConfig,
) -> Optional[Any]:
    """Return the simplest snap target within the fit's own tolerance, or None.

    Ranking is simplicity-first among in-tolerance targets (then distance), so
    a ratio fitted at 0.9975 +- 0.003 snaps to ``1`` rather than to a dense
    near-1 imposter such as ``5*sqrt(2)/(4*sqrt(pi))``.  Unlike
    :func:`nearest_symbolic_constant` this accepts exact matches, so a refit
    landing precisely on 1.0 still snaps to ``Integer(1)``.  The sigma-derived
    tolerance is capped at 10% relative so a degenerate fit cannot license
    arbitrary snaps.
    """

    if not math.isfinite(value):
        return None
    if sigma is not None and math.isfinite(sigma) and sigma > 0.0:
        # The fit's own uncertainty is the authority; a loose generic relative
        # tolerance would license statistically forbidden moves (a coefficient
        # constrained to 3e-4 must not jump 2.5e-3 to reach a pretty constant).
        abs_tol = min(3.0 * float(sigma), 0.1 * max(1.0, abs(value)))
        rel_floor = 5.0e-4
    else:
        abs_tol = 0.0
        rel_floor = float(config.snap_rel_tol)
    best = None
    best_key: tuple[int, float] = (1 << 30, float("inf"))
    for target in config.snap_targets():
        try:
            tv = float(target.evalf() if hasattr(target, "evalf") else target)
        except Exception:
            continue
        dist = abs(value - tv)
        tol = max(abs_tol, rel_floor * max(1.0, abs(tv)))
        if dist > tol:
            continue
        # Simplicity by op count, closeness as the tiebreak: string length
        # would prefer 5/pi**3 over the closer 1/(2*pi) on a character quirk.
        try:
            ops = int(sp.count_ops(target, visual=False))
        except Exception:
            ops = 1 << 20
        key = (ops, dist)
        if key < best_key:
            best = target
            best_key = key
    return best


def _pow_chain_for_site(
    base: Any, node: Any
) -> Optional[tuple[Any, Any, Optional[Any]]]:
    """Return ``(outermost_pow, effective_exponent, containing_addend)``.

    Eligible when ``node`` occurs exactly once in ``base``, is the base of
    exactly one ``Pow``, the chain of enclosing Pows (e.g. ``sqrt(1/A) =
    (A**-1)**(1/2)``) ends either in a top-level ``Mul`` factor of ``base``
    (``containing_addend`` is None) or in a ``Mul`` factor of exactly one
    addend of a top-level ``Add`` (``containing_addend`` is that addend, so
    the ``a**p`` compensation folds into that addend's coefficient), and the
    product of exponents along the chain is a Rational other than 1.  The
    identity ``a**(e0*e1) * ((S/a)**e0)**e1 == (S**e0)**e1`` for ``a > 0``
    makes normalizing the site by a positive coefficient exact.
    """

    occurrences = sum(1 for sub in sp.preorder_traversal(base) if sub == node)
    if occurrences != 1:
        return None
    pow_parents = [
        sub
        for sub in sp.preorder_traversal(base)
        if isinstance(sub, sp.Pow) and sub.base == node
    ]
    if len(pow_parents) != 1:
        return None
    outer_pow = pow_parents[0]
    chain_exp = outer_pow.exp
    while True:
        enclosing = [
            sub
            for sub in sp.preorder_traversal(base)
            if isinstance(sub, sp.Pow) and sub.base == outer_pow
        ]
        if len(enclosing) != 1:
            break
        chain_exp = chain_exp * enclosing[0].exp
        outer_pow = enclosing[0]
    if not bool(getattr(chain_exp, "is_Rational", False)) or chain_exp == 1:
        return None
    if outer_pow in list(sp.Mul.make_args(base)):
        return outer_pow, chain_exp, None
    if isinstance(base, sp.Add):
        holders = [
            addend
            for addend in sp.Add.make_args(base)
            if outer_pow in list(sp.Mul.make_args(addend))
        ]
        if len(holders) == 1:
            return outer_pow, chain_exp, holders[0]
    return None


def _exp_rate_parameterize(
    param_expr: Any,
    params: list[Any],
    inits: list[float],
    config: PolishConfig,
    *,
    max_rates: int = 4,
) -> tuple[Any, int]:
    """Parameterize refit-eligible multiplicative rates inside ``exp`` args.

    Only the numeric multiplicative coefficient of an ``exp`` argument is
    touched — a variable-base ``Pow`` exponent itself is never parameterized
    (that would change dimensional structure; an exp atom that happens to sit
    inside such an exponent is still value-safe and re-gated by the units
    check) and trig phases stay with the trig canonicalizer.  A rate that is a small exact rational (``exp(2*z)``,
    ``exp(z/2)``) is structure and stays frozen; Floats, large rationals
    (e.g. ``18361/9171``), and irrational numeric products (previously
    snapped constants like ``2*sqrt(2)*sqrt(pi)/5``) are refittable.  An
    ``exp`` argument must be dimensionless under units checking, so scaling
    it by a dimensionless numeric parameter preserves the units verdict.
    """

    eligible: list[tuple[Any, float, Any]] = []
    for atom in sorted(param_expr.atoms(sp.exp), key=sp.srepr):
        arg = atom.args[0]
        coeff_f, coeff_expr, rest = _addend_numeric_coefficient(arg)
        if coeff_f is None or coeff_f == 0.0 or coeff_expr == 1:
            continue
        if rest == 1:
            # Pure numeric argument: a constant, not a rate.
            continue
        if coeff_expr.free_symbols:
            continue
        structural = bool(getattr(coeff_expr, "is_Rational", False)) and (
            max(abs(int(coeff_expr.p)), abs(int(coeff_expr.q))) < 1000
        )
        if structural:
            continue
        eligible.append((atom, coeff_f, rest))
    # A single top-down xreplace cannot reach an exp nested inside another
    # replaced exp's argument; parameterizing it anyway would create a
    # phantom parameter with a zero Jacobian column.  Keep outermost only.
    eligible = [
        (atom, coeff_f, rest)
        for atom, coeff_f, rest in eligible
        if not any(
            other is not atom and atom in sp.preorder_traversal(other.args[0])
            for other, _c, _r in eligible
        )
    ]
    replacements: dict[Any, Any] = {}
    n_rates = 0
    for atom, coeff_f, rest in eligible:
        if n_rates >= int(max_rates):
            break
        theta = sp.Symbol(f"_dropfit_c{len(params)}", real=True)
        params.append(theta)
        inits.append(coeff_f)
        replacements[atom] = sp.exp(theta * rest)
        n_rates += 1
    if not replacements:
        return param_expr, 0
    return param_expr.xreplace(replacements), n_rates


def _fitted_float_literal(value: float, sigma: Optional[float]) -> Any:
    """Render a refitted coefficient at the precision the fit supports.

    Uses the fewest significant digits whose rounding error stays below
    ``0.3 * sigma``, so candidates are not charged constant-code cost for
    digits the data cannot distinguish.  Falls back to the full 17-digit
    round-trip literal when no uncertainty estimate is available.
    """

    v = float(value)
    if sigma is not None and math.isfinite(sigma) and sigma > 0.0:
        for digits in range(3, 18):
            rounded = float(f"{v:.{digits}g}")
            if abs(rounded - v) <= 0.3 * float(sigma):
                return sp.Float(f"{v:.{digits}g}", 17)
    return sp.Float(repr(v), 17)


def _lm_refit_coefficients(
    param_expr: Any,
    params: Sequence[Any],
    inits: Sequence[float],
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
    *,
    time_budget_s: float = 2.0,
    max_iter: int = 25,
) -> Optional[tuple[list[float], list[Optional[float]], float]]:
    """Damped Gauss-Newton refit of the ``params`` symbols in ``param_expr``.

    Deterministic (no RNG), float64, masked finite residuals.  Returns
    ``(fitted_values, per_parameter_sigmas, final_mse)`` or None on failure or
    when the divergence guard trips (any coefficient moving by more than
    ``drop_refit_shift_guard_factor`` or flipping sign).
    """

    t0 = time.monotonic()
    variable_set = {str(name) for name in variable_names}
    values = {
        validate_coefficient_symbol(name): float(value)
        for name, value in (config.symbol_values or {}).items()
    }
    expr_plain = param_expr.xreplace(
        {s: sp.Symbol(str(s), real=True) for s in param_expr.free_symbols}
    )
    substitutions = {
        symbol: sp.Float(repr(values[str(symbol)]), 17)
        for symbol in expr_plain.free_symbols
        if str(symbol) in values
    }
    expr_numeric = expr_plain.xreplace(substitutions)
    var_syms = [sp.Symbol(name, real=True) for name in variable_names]
    theta_syms = [sp.Symbol(str(p), real=True) for p in params]
    allowed = variable_set | {str(p) for p in params}
    if any(str(s) not in allowed for s in expr_numeric.free_symbols):
        return None
    try:
        fn = sp.lambdify(var_syms + theta_syms, expr_numeric, modules=["numpy"])
        grad_fns = [
            sp.lambdify(
                var_syms + theta_syms,
                sp.diff(expr_numeric, theta),
                modules=["numpy"],
            )
            for theta in theta_syms
        ]
    except Exception:
        return None

    n = X.shape[0]
    args = [X[:, j] for j in range(len(variable_names))]
    y_flat = np.asarray(y, dtype=np.float64).reshape(-1)[:n]

    def _eval(fun: Any, c: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = np.asarray(fun(*args, *c), dtype=np.float64)
        if out.shape == () or out.size == 1:
            out = np.full(n, float(np.ravel(out)[0]), dtype=np.float64)
        return np.ravel(out)[:n]

    c = np.asarray(list(inits), dtype=np.float64)
    try:
        resid = _eval(fn, c) - y_flat
    except Exception:
        return None
    mask = np.isfinite(resid)
    if float(np.count_nonzero(mask)) / max(1, n) < float(config.min_valid_fraction):
        return None
    sse = float(np.sum(resid[mask] ** 2))
    lam = 1.0e-3
    identity = np.eye(len(c))
    jac = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for _ in range(int(max_iter)):
            if time.monotonic() - t0 > float(time_budget_s):
                break
            try:
                jac = np.column_stack([_eval(g, c) for g in grad_fns])
            except Exception:
                return None
            step_mask = mask & np.all(np.isfinite(jac), axis=1)
            if float(np.count_nonzero(step_mask)) / max(1, n) < float(
                config.min_valid_fraction
            ):
                break
            J = jac[step_mask]
            r = resid[step_mask]
            A = J.T @ J
            g_vec = J.T @ r
            if not (np.all(np.isfinite(A)) and np.all(np.isfinite(g_vec))):
                break
            accepted = False
            for _tries in range(8):
                try:
                    step = -np.linalg.solve(
                        A + lam * np.diag(np.diag(A)) + 1.0e-12 * identity, g_vec
                    )
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                c_try = c + step
                try:
                    resid_try = _eval(fn, c_try) - y_flat
                except Exception:
                    lam *= 10.0
                    continue
                mask_try = np.isfinite(resid_try)
                if float(np.count_nonzero(mask_try)) / max(1, n) < float(
                    config.min_valid_fraction
                ):
                    lam *= 10.0
                    continue
                sse_try = float(np.sum(resid_try[mask_try] ** 2))
                if sse_try < sse:
                    rel_improvement = (sse - sse_try) / max(sse, 1.0e-300)
                    c, resid, mask, sse = c_try, resid_try, mask_try, sse_try
                    lam = max(lam / 3.0, 1.0e-12)
                    accepted = True
                    if rel_improvement < 1.0e-12:
                        accepted = False
                    break
                lam *= 10.0
            if not accepted or lam > 1.0e12:
                break

    factor = max(float(config.drop_refit_shift_guard_factor), 1.0)
    for c_j, init_j in zip(c, inits):
        if not math.isfinite(float(c_j)):
            return None
        ref = max(abs(float(init_j)), 1.0e-12)
        if abs(float(c_j)) > factor * ref or abs(float(c_j)) < ref / factor:
            return None
        if abs(init_j) > 1.0e-12 and math.copysign(1.0, float(c_j)) != math.copysign(
            1.0, float(init_j)
        ):
            return None

    n_valid = int(np.count_nonzero(mask))
    mse = sse / max(1, n_valid)
    sigmas: list[Optional[float]] = [None] * len(c)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if jac is None:
                jac = np.column_stack([_eval(g, c) for g in grad_fns])
            sig_mask = mask & np.all(np.isfinite(jac), axis=1)
            J = jac[sig_mask]
            dof = max(1, int(np.count_nonzero(sig_mask)) - len(c))
            cov = (sse / dof) * np.linalg.inv(J.T @ J + 1.0e-12 * identity)
            diag = np.diag(cov)
            sigmas = [
                float(math.sqrt(v)) if math.isfinite(v) and v >= 0.0 else None
                for v in diag
            ]
    except Exception:
        pass
    return [float(v) for v in c], sigmas, float(mse)


def _drop_addend_refit_specs(
    expr: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: Sequence[str],
    units_spec: Any,
    config: PolishConfig,
    *,
    max_rows: int = 4096,
    max_terms: int = 24,
    max_ops: int = 128,
) -> list[CandidateSpec]:
    """Propose drop-small-additive-term candidates with repolished coefficients.

    For each small addend of any value-position ``Add`` (denominators and
    radicands included), emit the expression with the addend deleted and the
    surviving numeric coefficients refitted by damped Gauss-Newton, then
    re-snapped to exact constants when the fit tolerates it.  This makes
    "coefficient -> 0" compete on the same frontier as "coefficient ->
    constant", and rescues truths hidden behind a spurious term that cancels a
    miscalibrated companion coefficient (the coupled move neither plain
    zeroing nor constant snapping can make).

    Candidates only compete: the seed always stays on the frontier and the
    downstream units/loss/Pareto gates adjudicate.  Emission is bounded by
    ``drop_refit_max_refits`` and ``drop_refit_max_seconds``.
    """

    if sp is None or not isinstance(expr, sp.Basic):
        return []
    t0 = time.monotonic()
    variable_set = {str(name) for name in variable_names}
    known_symbols = variable_set | set((config.symbol_values or {}).keys())

    n = X_train.shape[0]
    if n == 0:
        return []
    stride_idx = np.unique(
        np.linspace(0, n - 1, min(n, int(max_rows))).astype(int)
    )
    X_sub = X_train[stride_idx]
    y_sub = np.asarray(y_train, dtype=np.float64).reshape(-1)[stride_idx]

    def _rms(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("inf")
        return float(np.sqrt(np.mean(a**2)))

    def _pred(candidate: Any) -> Optional[np.ndarray]:
        try:
            return _eval_expr_array(
                candidate, X_sub, variable_names, config.symbol_values
            )
        except Exception:
            return None

    noise_floor = float(getattr(config, "noise_floor_raw", 0.0) or 0.0)
    sigma_noise = math.sqrt(noise_floor) if noise_floor > 0.0 else None

    specs: list[CandidateSpec] = []
    refits_used = 0
    large_threshold = int(config.drop_refit_large_number_threshold)

    def _time_left() -> bool:
        return time.monotonic() - t0 <= float(config.drop_refit_max_seconds)

    def _admissible_drops(base: Any, pred_full: np.ndarray) -> list[dict[str, Any]]:
        """Return admitted (node, index) drops with their in-context delta RMS."""
        resid_rms = _rms(pred_full - y_sub)
        if sigma_noise is not None:
            threshold_a = float(config.drop_refit_noise_sigma_mult) * sigma_noise
        else:
            threshold_a = float(config.drop_refit_resid_mult) * resid_rms
        admitted: list[dict[str, Any]] = []
        for site_index, node in enumerate(_value_position_add_nodes(base)):
            terms = list(node.args)
            if len(terms) < 2 or len(terms) > int(max_terms):
                continue
            try:
                if int(sp.count_ops(node, visual=False)) > int(max_ops):
                    continue
            except Exception:
                continue
            has_large = _has_large_exact_number(node, large_threshold)
            coeffs: list[Optional[float]] = []
            for term in terms:
                coeff_f, _coeff_expr, _rest = _addend_numeric_coefficient(term)
                coeffs.append(coeff_f)
            max_coeff = max(
                (abs(c) for c in coeffs if c is not None), default=0.0
            )
            node_rows: list[dict[str, Any]] = []
            deltas: list[float] = []
            for term_index, term in enumerate(terms):
                if not _time_left():
                    break
                extra_symbols = {
                    str(s) for s in term.free_symbols
                } - variable_set
                if extra_symbols:
                    # Named symbols may carry units contracts; never judge or
                    # drop their terms here.
                    continue
                survivors = [
                    t for i, t in enumerate(terms) if i != term_index
                ]
                rebuilt = sp.Add(*survivors)
                if rebuilt == 0:
                    continue
                expr_drop = base.xreplace({node: rebuilt})
                if expr_drop.has(sp.zoo, sp.nan, sp.oo):
                    continue
                if not (
                    {str(s) for s in expr_drop.free_symbols} & variable_set
                ):
                    continue
                unit_ok, _reason = _sympy_expr_units_check(
                    expr_drop, variable_names, units_spec
                )
                if not unit_ok:
                    continue
                pred_drop = _pred(expr_drop)
                if pred_drop is None:
                    continue
                delta_rms = _rms(pred_full - pred_drop)
                if not math.isfinite(delta_rms):
                    continue
                route = None
                if delta_rms <= threshold_a:
                    route = "ablation_rms"
                elif has_large and coeffs[term_index] is not None and max_coeff > 0.0:
                    ratio = abs(coeffs[term_index]) / max_coeff
                    if ratio <= float(config.drop_refit_site_rel_ratio_max):
                        route = "large_number_fingerprint"
                if route is None:
                    continue
                row = {
                    "node": node,
                    "site_index": site_index,
                    "term_index": term_index,
                    "term": term,
                    "survivors": survivors,
                    "delta_rms": delta_rms,
                    "route": route,
                }
                node_rows.append(row)
                deltas.append(delta_rms)
            if not node_rows:
                continue
            # Never drop the node's dominant term.
            if len(node_rows) == len(terms):
                dominant = max(range(len(deltas)), key=lambda i: deltas[i])
                node_rows = [
                    row for i, row in enumerate(node_rows) if i != dominant
                ]
            admitted.extend(node_rows)
        admitted.sort(key=lambda row: (row["delta_rms"], row["site_index"], row["term_index"]))
        return admitted[: int(config.drop_refit_max_sites)]

    def _build_candidate(
        base: Any,
        drops: Sequence[dict[str, Any]],
        label_suffix: str,
        round_tag: str,
    ) -> Optional[tuple[CandidateSpec, float]]:
        """Drop, refit, re-snap; return (spec, refit_train_mse) or None.

        Parameterization is anchor-relative when the drop site carries a large
        exact coefficient (the Stage-B rationalized-integer fingerprint):
        survivor coefficients become ``theta_i * c_anchor`` with ``theta_i``
        initialized at the coefficient ratio, and the expression's top-level
        numeric prefactor becomes ``theta_top * c_anchor``.  Ratios near simple
        constants then snap exactly, and the shared anchor cancels in the final
        ``factor_terms``/``cancel`` cleanup — recovering e.g.
        ``num/(x**(5/2) + ...)`` from a pair of eight-digit integers.  Sites
        without a large anchor use plain absolute coefficients plus one fresh
        global scale.
        """

        nonlocal refits_used
        params: list[Any] = []
        inits: list[float] = []
        scale_param_indices: set[int] = set()
        replacements: dict[Any, Any] = {}
        by_node: dict[str, tuple[Any, list[int]]] = {}
        for row in drops:
            key = sp.srepr(row["node"])
            node, indices = by_node.get(key, (row["node"], []))
            if row["term_index"] is not None:
                indices.append(row["term_index"])
            by_node[key] = (node, indices)
        real_drops = [row for row in drops if row["term_index"] is not None]

        # Choose one anchor: the largest fixed survivor coefficient among
        # sites whose Add carries a large exact number.
        anchor_expr: Any = None
        anchor_value: Optional[float] = None
        site_plans: list[tuple[Any, list[Any], int, list[tuple[Optional[float], Any, Any]]]] = []
        for node, indices in by_node.values():
            survivors = [
                t for i, t in enumerate(node.args) if i not in set(indices)
            ]
            if not survivors:
                return None
            split = [_addend_numeric_coefficient(t) for t in survivors]
            magnitudes = [
                abs(c) if c is not None else -1.0 for c, _ce, _rest in split
            ]
            fixed_index = int(np.argmax(magnitudes)) if magnitudes else 0
            site_plans.append((node, survivors, fixed_index, split))
            fixed_value = split[fixed_index][0]
            if (
                fixed_value is not None
                and abs(fixed_value) >= float(large_threshold)
                and _has_large_exact_number(node, large_threshold)
                and (anchor_value is None or abs(fixed_value) > abs(anchor_value))
            ):
                anchor_value = fixed_value
                anchor_expr = split[fixed_index][1]

        # Radical-gauge normalization: when the (single) drop site is the base
        # of exactly one top-level ``Pow`` factor with rational exponent and
        # carries no large exact anchor, fit the survivor coefficients as
        # ratios to a positive normalizer ``a`` and absorb ``a**p`` into the
        # top scale.  The fitted parameters then live in the gauge where the
        # exact constants sit (e.g. -4903.75/496.8 -> -pi**2, combined scale
        # -> 1), which per-literal snapping cannot reach in the raw gauge
        # (496.8 alone has no snap target).
        gauge_norm_value: Optional[float] = None
        gauge_norm_expr: Any = None
        gauge_norm_index: Optional[int] = None
        gauge_compensation: Optional[float] = None
        gauge_addend: Any = None
        if anchor_value is None and len(site_plans) == 1:
            g_node, _g_survivors, _g_fixed, g_split = site_plans[0]
            chain = _pow_chain_for_site(base, g_node)
            if chain is not None:
                _outer_pow, chain_exp, containing_addend = chain
                positives = [
                    (i, c)
                    for i, (c, _ce, _rest) in enumerate(g_split)
                    if c is not None and c > 0.0
                ]
                if positives:
                    index, value = max(positives, key=lambda pair: pair[1])
                    try:
                        compensation = float(value) ** float(chain_exp)
                    except Exception:
                        compensation = float("nan")
                    if math.isfinite(compensation) and compensation > 0.0:
                        gauge_norm_index = index
                        gauge_norm_value = float(value)
                        gauge_norm_expr = g_split[index][1]
                        gauge_compensation = compensation
                        gauge_addend = containing_addend

        for node, survivors, fixed_index, split in site_plans:
            new_terms: list[Any] = []
            for i, (term, (coeff_f, _coeff_expr, rest)) in enumerate(
                zip(survivors, split)
            ):
                if gauge_norm_index is not None:
                    # Exact division of the whole site by the normalizer.
                    if i == gauge_norm_index:
                        new_terms.append(rest)
                        continue
                    if coeff_f is None or coeff_f == 0.0:
                        new_terms.append(term / gauge_norm_expr)
                        continue
                    theta = sp.Symbol(f"_dropfit_c{len(params)}", real=True)
                    params.append(theta)
                    inits.append(coeff_f / gauge_norm_value)
                    new_terms.append(theta * rest)
                    continue
                if i == fixed_index or coeff_f is None or coeff_f == 0.0:
                    new_terms.append(term)
                    continue
                theta = sp.Symbol(f"_dropfit_c{len(params)}", real=True)
                if anchor_value is not None:
                    params.append(theta)
                    inits.append(coeff_f / anchor_value)
                    new_terms.append(theta * anchor_expr * rest)
                else:
                    params.append(theta)
                    inits.append(coeff_f)
                    new_terms.append(theta * rest)
            replacements[node] = sp.Add(*new_terms)
        if len(params) > int(config.drop_refit_max_params):
            return None

        dropped_expr = base.xreplace(replacements)
        theta_scale = sp.Symbol(f"_dropfit_c{len(params)}", real=True)
        if gauge_addend is None:
            # The per-addend gauge branch appends no overall-scale parameter,
            # so the zero-snap protection must not land on its first addend
            # coefficient by accident.
            scale_param_indices.add(len(params))
        if anchor_value is not None:
            top_value, _top_expr, top_rest = _addend_numeric_coefficient(
                dropped_expr
            )
            if top_value is None or top_value == 0.0:
                params.append(theta_scale)
                inits.append(1.0)
                param_expr = theta_scale * dropped_expr
            else:
                params.append(theta_scale)
                inits.append(top_value / anchor_value)
                param_expr = theta_scale * anchor_expr * top_rest
        elif gauge_compensation is not None and gauge_addend is not None:
            # The gauge site lives inside one addend of a top-level Add: fold
            # a**p into that addend's coefficient and recalibrate the sibling
            # addends' coefficients alongside (pb115-class expressions).
            holder_norm = gauge_addend.xreplace(replacements)
            new_addends: list[Any] = []
            matched_holder = False
            for addend in sp.Add.make_args(dropped_expr):
                coeff_f, _coeff_expr, rest = _addend_numeric_coefficient(addend)
                if not matched_holder and addend == holder_norm:
                    matched_holder = True
                    if coeff_f is None or coeff_f == 0.0:
                        coeff_f, rest = 1.0, addend
                    theta = sp.Symbol(f"_dropfit_c{len(params)}", real=True)
                    params.append(theta)
                    inits.append(float(coeff_f) * float(gauge_compensation))
                    new_addends.append(theta * rest)
                    continue
                if coeff_f is None or coeff_f == 0.0:
                    new_addends.append(addend)
                    continue
                theta = sp.Symbol(f"_dropfit_c{len(params)}", real=True)
                params.append(theta)
                inits.append(float(coeff_f))
                new_addends.append(theta * rest)
            if not matched_holder:
                return None
            if len(params) > int(config.drop_refit_max_params):
                return None
            param_expr = sp.Add(*new_addends)
        elif gauge_compensation is not None:
            # The site was divided by the normalizer, so the top scale must
            # absorb a**p exactly: theta_top init = c_top * a**p.
            top_value, _top_expr, top_rest = _addend_numeric_coefficient(
                dropped_expr
            )
            if top_value is None or top_value == 0.0:
                top_value, top_rest = 1.0, dropped_expr
            params.append(theta_scale)
            inits.append(float(top_value) * float(gauge_compensation))
            param_expr = theta_scale * top_rest
        else:
            # Absorb the top-level numeric prefactor into the scale parameter
            # so the fitted amplitude itself is exposed to the sigma-licensed
            # snap (e.g. 15/937 -> 1/(2*pi**3)); a bare multiplier would leave
            # the literal frozen inside.
            top_value, _top_expr, top_rest = _addend_numeric_coefficient(
                dropped_expr
            )
            if top_value is None or top_value == 0.0:
                top_value, top_rest = 1.0, dropped_expr
            params.append(theta_scale)
            inits.append(float(top_value))
            param_expr = theta_scale * top_rest

        param_expr, n_rates = _exp_rate_parameterize(
            param_expr, params, inits, config
        )
        if len(params) > int(config.drop_refit_max_params):
            return None
        if real_drops:
            dropped_terms = ", ".join(_sstr(row["term"]) for row in real_drops)
            trace = [
                (
                    f"drop small additive term(s): {dropped_terms} "
                    f"[{real_drops[0]['route']}, "
                    f"delta_rms={real_drops[0]['delta_rms']:.4g}]"
                )
            ]
        else:
            trace = [
                "recalibrate site coefficients without dropping terms (d0)"
            ]
        if gauge_compensation is not None:
            trace.append(
                "radical gauge normalization: divide site by "
                f"{gauge_norm_value:.12g}, absorb its power into the top scale"
            )

        refits_used += 1
        fit = _lm_refit_coefficients(
            param_expr,
            params,
            inits,
            X_sub,
            y_sub,
            variable_names,
            config,
        )
        if fit is None:
            if not real_drops:
                # A failed pure recalibration has no fallback: the frozen
                # expression is the base itself.
                return None
            # Cheap fallback: the frozen-coefficient drop still competes.
            frozen = base.xreplace(
                {
                    node: sp.Add(
                        *[
                            t
                            for i, t in enumerate(node.args)
                            if i not in {r["term_index"] for r in drops if sp.srepr(r["node"]) == sp.srepr(node)}
                        ]
                    )
                    for node, _indices in by_node.values()
                }
            )
            spec = CandidateSpec(
                frozen,
                label=f"drop_addend_norefit:{label_suffix}{round_tag}",
                rewrite_trace=trace + ["coefficient refit failed; frozen coefficients kept"],
                source_hints=["drop_addend_refit"],
                selection_n_free_params=0,
            )
            return spec, float("inf")

        fitted, sigmas, refit_mse = fit
        snapped_values: list[Any] = []
        n_snapped = 0
        all_snapped = True
        for index, (value, sigma, init) in enumerate(zip(fitted, sigmas, inits)):
            target = _snap_fitted_coefficient(value, sigma, config)
            if target is not None and index in scale_param_indices and target == 0:
                # Snapping the overall scale to zero would zero the candidate.
                target = None
            if target is not None:
                snapped_values.append(target)
                n_snapped += 1
            else:
                snapped_values.append(_fitted_float_literal(value, sigma))
                all_snapped = False
            shift = abs(value - init) / max(abs(init), 1.0e-12)
            trace.append(
                f"repolish coefficient: {init:.12g} -> {value:.12g} "
                f"(shift {100.0 * shift:.3g}%)"
                + (f", snapped -> {sp.sstr(target)}" if target is not None else "")
            )
        final_expr = param_expr.xreplace(
            dict(zip(params, snapped_values))
        )
        if final_expr.has(sp.zoo, sp.nan, sp.oo):
            return None
        # Belt and braces: if the snapped spelling measurably degrades the
        # refit optimum on the training subsample, keep the float spelling
        # (an over-eager snap must not cost real loss).
        if n_snapped > 0:
            pred_snapped = _pred(final_expr)
            snapped_mse = float("inf")
            if pred_snapped is not None:
                diff = pred_snapped - y_sub
                diff = diff[np.isfinite(diff)]
                if diff.size:
                    snapped_mse = float(np.mean(diff**2))
            if snapped_mse > float(refit_mse) * 1.05 + 1.0e-300:
                float_values = [
                    _fitted_float_literal(v, s)
                    for v, s in zip(fitted, sigmas)
                ]
                # Leave-one-out before full revert: a single bad snap must not
                # discard the good exact constants alongside it.
                snap_indices = [
                    j
                    for j, v in enumerate(snapped_values)
                    if not isinstance(v, sp.Float)
                ]
                rescue = None
                rescue_mse = float("inf")
                if len(snap_indices) >= 2:
                    for j in snap_indices:
                        trial = list(snapped_values)
                        trial[j] = float_values[j]
                        trial_expr = param_expr.xreplace(
                            dict(zip(params, trial))
                        )
                        if trial_expr.has(sp.zoo, sp.nan, sp.oo):
                            continue
                        pred_trial = _pred(trial_expr)
                        if pred_trial is None:
                            continue
                        diff = pred_trial - y_sub
                        diff = diff[np.isfinite(diff)]
                        if not diff.size:
                            continue
                        trial_mse = float(np.mean(diff**2))
                        if (
                            trial_mse <= float(refit_mse) * 1.05 + 1.0e-300
                            and trial_mse < rescue_mse
                        ):
                            rescue = trial
                            rescue_mse = trial_mse
                if rescue is not None:
                    snapped_values = rescue
                    n_snapped -= 1
                    all_snapped = False
                    final_expr = param_expr.xreplace(
                        dict(zip(params, snapped_values))
                    )
                    if final_expr.has(sp.zoo, sp.nan, sp.oo):
                        return None
                    trace.append(
                        "one snapped constant degrades the refit optimum; "
                        "reverted it to a float and kept the others"
                    )
                else:
                    final_expr = param_expr.xreplace(
                        dict(zip(params, float_values))
                    )
                    if final_expr.has(sp.zoo, sp.nan, sp.oo):
                        return None
                    snapped_values = float_values
                    n_snapped = 0
                    all_snapped = False
                    trace.append(
                        "snapped constants degrade the refit optimum; keeping "
                        "float coefficients"
                    )
        # Collapse a shared anchor: factor it out of the surviving Add and
        # cancel it against the top-level prefactor.
        try:
            cleaned = sp.cancel(sp.factor_terms(final_expr))
            if not cleaned.has(sp.zoo, sp.nan, sp.oo) and expression_complexity(
                cleaned, config
            ) <= expression_complexity(final_expr, config):
                final_expr = cleaned
        except Exception:
            pass
        label_kind = "drop_addend_refit_snap" if all_snapped else "drop_addend_refit"
        spec = CandidateSpec(
            final_expr,
            label=f"{label_kind}:{label_suffix}{round_tag}",
            n_free_params=0 if all_snapped else sum(
                1 for v in snapped_values if isinstance(v, sp.Float)
            ),
            n_snapped_consts=n_snapped,
            rewrite_trace=trace,
            source_hints=["drop_addend_refit"],
            # The refitted literals are frozen before the statistical-selection
            # archive is built, exactly like Stage-B/C fitted constants whose
            # rows declare 0 free parameters at selection time.
            selection_n_free_params=0,
        )
        return spec, float(refit_mse)

    base = expr
    round_tag = ""
    for round_index in range(int(config.drop_refit_max_rounds)):
        if not _time_left() or refits_used >= int(config.drop_refit_max_refits):
            break
        pred_full = _pred(base)
        if pred_full is None:
            break
        base_mse = float(np.mean((pred_full - y_sub)[np.isfinite(pred_full - y_sub)] ** 2)) if np.any(np.isfinite(pred_full - y_sub)) else float("inf")
        admitted = _admissible_drops(base, pred_full)
        # d0 recalibration: sites whose fitted-but-frozen coefficients hide
        # simple gauge ratios (rationalized-integer anchors like the Compton
        # 2:1 pair, or Pow-chain gauges) even when there is nothing to drop.
        # Round 0 only, capped at 2 sites, skipping sites the drop candidates
        # already recalibrate.
        recal_rows: list[dict[str, Any]] = []
        if round_index == 0:
            admitted_nodes = {sp.srepr(row["node"]) for row in admitted}
            for site_index, node in enumerate(_value_position_add_nodes(base)):
                if len(recal_rows) >= 2:
                    break
                terms = list(node.args)
                if len(terms) < 2 or len(terms) > int(max_terms):
                    continue
                try:
                    if int(sp.count_ops(node, visual=False)) > int(max_ops):
                        continue
                except Exception:
                    continue
                if sp.srepr(node) in admitted_nodes:
                    continue
                split = [_addend_numeric_coefficient(t) for t in terms]
                magnitudes = [
                    abs(c) if c is not None else -1.0 for c, _ce, _r in split
                ]
                fixed_value = (
                    split[int(np.argmax(magnitudes))][0] if magnitudes else None
                )
                anchor_ok = (
                    fixed_value is not None
                    and abs(fixed_value) >= float(large_threshold)
                    and _has_large_exact_number(node, large_threshold)
                )
                gauge_ok = _pow_chain_for_site(base, node) is not None and any(
                    c is not None and c > 0.0 for c, _ce, _r in split
                )
                if anchor_ok or gauge_ok:
                    recal_rows.append(
                        {
                            "node": node,
                            "site_index": site_index,
                            "term_index": None,
                            "term": None,
                            "delta_rms": 0.0,
                            "route": "recalibration",
                        }
                    )
        if not admitted and not recal_rows:
            break
        round_specs: list[tuple[CandidateSpec, float]] = []
        for row in admitted:
            if not _time_left() or refits_used >= int(config.drop_refit_max_refits):
                break
            built = _build_candidate(
                base,
                [row],
                f"s{row['site_index']}:d1",
                round_tag,
            )
            if built is not None:
                round_specs.append(built)
        if (
            len(admitted) >= 2
            and _time_left()
            and refits_used < int(config.drop_refit_max_refits)
        ):
            built = _build_candidate(
                base,
                admitted,
                f"all{len(admitted)}",
                round_tag,
            )
            if built is not None:
                round_specs.append(built)
        for row in recal_rows:
            if not _time_left() or refits_used >= int(config.drop_refit_max_refits):
                break
            built = _build_candidate(
                base,
                [row],
                f"s{row['site_index']}:d0",
                round_tag,
            )
            if built is not None:
                round_specs.append(built)
        specs.extend(spec for spec, _mse in round_specs)
        # Greedy continuation through the round's best loss-neutral candidate.
        finite_rounds = [
            (spec, mse)
            for spec, mse in round_specs
            if math.isfinite(mse) and mse <= base_mse * (1.0 + 1.0e-6)
        ]
        if not finite_rounds:
            break
        best_spec, _best_mse = min(finite_rounds, key=lambda pair: pair[1])
        extra = {str(s) for s in best_spec.expr.free_symbols} - known_symbols
        if extra:
            break
        base = best_spec.expr
        round_tag = f":r{round_index + 2}"
    return specs


def _guarded_sympy_candidates(expr: Any, variable_names: Sequence[str], config: PolishConfig) -> list[CandidateSpec]:
    out: list[CandidateSpec] = []
    base_cx = expression_complexity(expr, config)
    transforms = []
    try:
        transforms.append(("factor_terms", sp.factor_terms(expr)))
    except Exception:
        pass
    try:
        transforms.append(("cancel", sp.cancel(expr)))
    except Exception:
        pass
    try:
        transforms.append(("together_cancel", sp.cancel(sp.together(expr))))
    except Exception:
        pass
    try:
        transforms.append(("powsimp", sp.powsimp(expr, force=True)))
    except Exception:
        pass
    try:
        transforms.append(
            (
                "canonicalize_trig_phases",
                canonicalize_trig_phases(expr, snap_rel_tol=config.snap_rel_tol),
            )
        )
    except Exception:
        pass
    try:
        transforms.append(
            (
                "snap_symbolic_constants",
                snap_numeric_constants(
                    expr,
                    snap_targets=config.snap_targets(),
                    snap_rel_tol=float(config.snap_rel_tol),
                ),
            )
        )
    except Exception:
        pass
    if aggressive_simplify is not None:
        try:
            if sympy_timeout is not None:
                with sympy_timeout("equation_polisher_aggressive", max_seconds=5):
                    simp = aggressive_simplify(expr, len(variable_names), verbose=False, budget_seconds=5)
            else:
                simp = aggressive_simplify(expr, len(variable_names), verbose=False, budget_seconds=5)
            transforms.append(("aggressive_simplify", simp))
        except Exception:
            pass
    transforms.extend(_log_ratio_candidates(expr))
    transforms.extend(
        numeric_constant_snap_candidates(
            expr,
            snap_targets=config.snap_targets(),
            snap_rel_tol=float(config.snap_rel_tol),
            per_number=4,
        )
    )

    seen: set[str] = set()

    def maybe_add(label: str, ex: Any, rewrite_trace: list[str]) -> bool:
        try:
            ex = _canonicalize_guarded_candidate_expr(ex, config)
        except Exception:
            pass
        if ex == expr:
            return False
        try:
            key = _canonical_key(ex)
        except Exception:
            key = _sstr(ex)
        if key in seen:
            return False
        seen.add(key)
        cx = expression_complexity(ex, config)
        if cx <= base_cx * 1.5:
            out.append(
                CandidateSpec(
                    ex,
                    label=label,
                    rewrite_trace=rewrite_trace,
                )
            )
            return True
        return False

    for label, ex in transforms:
        ex = _canonicalize_guarded_candidate_expr(ex, config)
        if ex == expr:
            continue
        maybe_add(label, ex, [f"guarded SymPy cleanup: {label}"])
        if label.startswith("snap_symbolic"):
            continue
        try:
            snap_variants = numeric_constant_snap_candidates(
                ex,
                snap_targets=config.snap_targets(),
                snap_rel_tol=float(config.snap_rel_tol),
                per_number=2,
            )
        except Exception:
            snap_variants = []
        for snap_label, snap_ex in snap_variants:
            maybe_add(
                f"{label}|{snap_label}",
                snap_ex,
                [
                    f"guarded SymPy cleanup: {label}",
                    f"second-pass symbolic constant snap: {snap_label}",
                ],
            )
    return out


def _sympy_poly_from_terms(
    coeffs: Sequence[float],
    exps: Sequence[Sequence[int]],
    symbols: Sequence[Any],
    *,
    config: PolishConfig,
    eps: float = 1.0e-10,
) -> tuple[Any, int, int]:
    expr = sp.Integer(0)
    n_active = 0
    n_snapped = 0
    for coeff, exp in zip(coeffs, exps):
        c = float(coeff)
        if not math.isfinite(c) or abs(c) <= eps:
            continue
        n_active += 1
        c_expr: Any = sp.Float(c, 15)
        try:
            snap_tol = max(float(config.snap_rel_tol), 2.5e-2)
            nearest_int = int(round(c))
            if abs(c - nearest_int) <= snap_tol * max(1.0, abs(c)):
                c_expr = sp.Integer(nearest_int)
                n_snapped += 1
            else:
                simple_targets = [
                    sp.Integer(0),
                    sp.Integer(1),
                    sp.Integer(-1),
                    sp.Integer(2),
                    sp.Integer(-2),
                    sp.Integer(3),
                    sp.Integer(-3),
                    sp.Rational(1, 2),
                    sp.Rational(-1, 2),
                    sp.Rational(1, 3),
                    sp.Rational(-1, 3),
                    sp.Rational(2, 3),
                    sp.Rational(-2, 3),
                    sp.Rational(3, 2),
                    sp.Rational(-3, 2),
                    sp.Rational(1, 4),
                    sp.Rational(-1, 4),
                    sp.Rational(3, 4),
                    sp.Rational(-3, 4),
                ]
                snapped = snap_numeric_constants(
                    c_expr,
                    snap_targets=simple_targets,
                    snap_rel_tol=snap_tol,
                )
                if snapped != c_expr:
                    c_expr = snapped
                    n_snapped += 1
        except Exception:
            pass
        term = c_expr
        for sym, p in zip(symbols, exp):
            ip = int(p)
            if ip:
                term *= sym ** ip
        expr += term
    return expr, int(n_active), int(n_snapped)


def _symbols_for_expr(expr: Any, variable_names: Sequence[str]) -> list[Any]:
    by_name = {str(sym): sym for sym in getattr(expr, "free_symbols", set())}
    out = []
    for name in variable_names:
        out.append(by_name.get(str(name), sp.Symbol(str(name), positive=True, real=True)))
    return out

def _sparse_rational_seed_support_candidates(
    seed: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
) -> list[CandidateSpec]:
    """Prune a noisy dense rational using its own visible monomial support."""

    if sp is None:
        return []
    try:
        noise_floor = float(config.noise_floor_raw)
    except Exception:
        noise_floor = 0.0
    if not (math.isfinite(noise_floor) and noise_floor > 0.0):
        return []
    try:
        num, den = sp.fraction(sp.together(seed))
        if den == 1:
            return []
        symbols = _symbols_for_expr(seed, variable_names)
        poly_num = sp.Poly(sp.expand(num), *symbols)
        poly_den = sp.Poly(sp.expand(den), *symbols)
        exps_num = list(poly_num.monoms())
        exps_den = list(poly_den.monoms())
        if not exps_num or not exps_den:
            return []
        if len(exps_num) + len(exps_den) > 48:
            return []
        coeffs_num = [float(sp.N(c, 20)) for c in poly_num.coeffs()]
        coeffs_den = [float(sp.N(c, 20)) for c in poly_den.coeffs()]
    except Exception:
        return []

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or X.shape[1] != len(variable_names):
        return []
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(np.count_nonzero(mask)) < max(80, len(exps_num) + len(exps_den) + 8):
        return []
    X = X[mask]
    y = y[mask]
    if X.shape[0] > 5000:
        rng = np.random.default_rng(int(config.seed))
        idx = rng.choice(X.shape[0], size=5000, replace=False)
        X = X[idx]
        y = y[idx]
    try:
        X_t = torch.as_tensor(X, dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64)
        Phi_num = _eval_monomials(X_t, torch.as_tensor(exps_num, dtype=torch.int64))
        Phi_den = _eval_monomials(X_t, torch.as_tensor(exps_den, dtype=torch.int64))
        a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
            Phi_num=Phi_num,
            Phi_den=Phi_den,
            y=y_t,
            coeffs_num=torch.as_tensor(coeffs_num, dtype=torch.float64),
            coeffs_den=torch.as_tensor(coeffs_den, dtype=torch.float64),
            cfg=replace(
                DEFAULT_RAT_STLSQ_CFG,
                proposal_noise_floor=float(noise_floor),
                den_clamp_mode=False,
                proposal_param_gamma=0.75,
                lam_rel=1.0e-4,
                greedy_max_drops=64,
            ),
        )
        if meta.get("accepted", 0.0) < 0.5:
            return []
        num_expr, n_num, s_num = _sympy_poly_from_terms(
            a_sparse.detach().cpu().numpy().tolist(),
            exps_num,
            symbols,
            config=config,
        )
        den_expr, n_den, s_den = _sympy_poly_from_terms(
            b_sparse.detach().cpu().numpy().tolist(),
            exps_den,
            symbols,
            config=config,
        )
        if num_expr == 0 or den_expr == 0:
            return []
        expr = num_expr / den_expr
        try:
            expr = sp.factor(sp.cancel(expr))
        except Exception:
            try:
                expr = sp.cancel(expr)
            except Exception:
                pass
        expr = _canonicalize_guarded_candidate_expr(expr, config)
        if expr == seed:
            return []
        return [
            CandidateSpec(
                expr,
                label="sparse_rational_seed_support",
                n_free_params=max(0, int(n_num + n_den - s_num - s_den)),
                n_snapped_consts=int(s_num + s_den),
                rewrite_trace=[
                    "noisy sparse rational pruning over visible seed support",
                    f"nnz=({n_num},{n_den})",
                ],
                source_hints=["noisy_sparse_rational_seed_support"],
            )
        ]
    except Exception:
        return []


def _sparse_rational_refit_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
) -> list[CandidateSpec]:
    """Fit a tiny sparse rational ballot from data for noisy dense-ratpoly seeds."""

    if sp is None:
        return []
    try:
        noise_floor = float(config.noise_floor_raw)
    except Exception:
        noise_floor = 0.0
    if not (math.isfinite(noise_floor) and noise_floor > 0.0):
        return []
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        return []
    n, dim = X.shape
    if dim <= 0 or dim > 7 or n < 80:
        return []
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(np.count_nonzero(mask)) < 80:
        return []
    X = X[mask]
    y = y[mask]
    if X.shape[0] > 5000:
        rng = np.random.default_rng(int(config.seed))
        idx = rng.choice(X.shape[0], size=5000, replace=False)
        X = X[idx]
        y = y[idx]
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    symbols = [sp.Symbol(str(name), positive=True, real=True) for name in variable_names]
    degree_pairs = [(2, 1), (3, 2), (2, 2), (3, 1)]
    out: list[CandidateSpec] = []
    seen: set[str] = set()
    for deg_num, deg_den in degree_pairs:
        try:
            exps_num = _enumerate_exponents(dim, int(deg_num))
            exps_den = _enumerate_exponents(dim, int(deg_den))
            exps_num_t = torch.as_tensor(exps_num, dtype=torch.int64)
            exps_den_t = torch.as_tensor(exps_den, dtype=torch.int64)
            Phi_num = _eval_monomials(X_t, exps_num_t)
            Phi_den = _eval_monomials(X_t, exps_den_t)
            if int(Phi_num.shape[0]) < int(Phi_num.shape[1] + Phi_den.shape[1] + 5):
                continue
            A = torch.cat([Phi_num, -(y_t.unsqueeze(1) * Phi_den)], dim=1)
            gram = (A.T @ A) / max(1, int(A.shape[0]))
            _evals, vecs = torch.linalg.eigh(gram)
            coeff = vecs[:, 0]
            a = coeff[: int(Phi_num.shape[1])].clone()
            b = coeff[int(Phi_num.shape[1]) :].clone()
            if b.numel() > 0:
                pivot = b[0]
                if abs(float(pivot)) < 1.0e-12:
                    pivot = b[torch.argmax(b.abs())]
                if abs(float(pivot)) >= 1.0e-12:
                    a = a / pivot
                    b = b / pivot
            a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
                Phi_num=Phi_num,
                Phi_den=Phi_den,
                y=y_t,
                coeffs_num=a,
                coeffs_den=b,
                cfg=replace(
                    DEFAULT_RAT_STLSQ_CFG,
                    proposal_noise_floor=float(noise_floor),
                    den_clamp_mode=False,
                    proposal_param_gamma=0.75,
                    lam_rel=5.0e-4,
                ),
            )
            if meta.get("accepted", 0.0) < 0.5:
                continue
            num, n_num, s_num = _sympy_poly_from_terms(
                a_sparse.detach().cpu().numpy().tolist(),
                exps_num,
                symbols,
                config=config,
            )
            den, n_den, s_den = _sympy_poly_from_terms(
                b_sparse.detach().cpu().numpy().tolist(),
                exps_den,
                symbols,
                config=config,
            )
            if num == 0 or den == 0 or int(n_num + n_den) <= 0:
                continue
            expr = num / den
            try:
                expr = sp.factor(sp.cancel(expr))
            except Exception:
                try:
                    expr = sp.cancel(expr)
                except Exception:
                    pass
            try:
                expr = _canonicalize_guarded_candidate_expr(expr, config)
            except Exception:
                pass
            key = _canonical_key(expr)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                CandidateSpec(
                    expr,
                    label=f"sparse_rational_refit_{deg_num}_{deg_den}",
                    n_free_params=max(0, int(n_num + n_den - s_num - s_den)),
                    n_snapped_consts=int(s_num + s_den),
                    rewrite_trace=[
                        "noisy sparse rational refit from final-polish data",
                        f"degrees=({deg_num},{deg_den}), nnz=({n_num},{n_den})",
                    ],
                    source_hints=["noisy_sparse_rational_refit"],
                )
            )
        except Exception:
            continue
    return out


def _is_identity_transform_name(name: Optional[str]) -> bool:
    text = str(name or "identity").strip().lower()
    return text in {"", "identity", "none", "null"}


def _artifact_expression_candidates(
    hints: Optional[ArtifactHints],
    variable_names: Sequence[str],
    assumptions: Mapping[str, str],
) -> list[CandidateSpec]:
    """Add already-discovered analytic artifact forms as scored candidates.

    The final polisher scores against raw dataset y.  Stage A/B and phi-space
    expressions are therefore direct candidates only for identity y-transforms.
    """
    if sp is None or hints is None:
        return []

    out: list[CandidateSpec] = []
    raw: list[tuple[str, str]] = []

    def add(label: str, value: Any) -> None:
        text = _clean_expr_field(value)
        if text is not None:
            raw.append((label, text))

    add("artifact:y_expr", hints.y_expr)
    if _is_identity_transform_name(hints.y_transform):
        add("artifact:phi_expr", hints.phi_expr)
        add("artifact:stageB_expr", hints.stageB_expr)
        for rec in hints.simplification_path:
            if not isinstance(rec, Mapping):
                continue
            stage = str(rec.get("stage") or "").strip().upper()
            if stage not in {"B", "C"}:
                continue
            step = rec.get("step", "?")
            add(f"artifact:path_step_{step}", rec.get("expression"))

    seen_text: set[str] = set()
    for label, text in raw:
        if text in seen_text:
            continue
        seen_text.add(text)
        if _has_unsupported_artifact_call(text):
            continue
        try:
            expr = _parse_with_assumptions(text, variable_names, assumptions)
            expr = _prune_tiny_additive_constants(expr, tol=0.0)
            try:
                expr = _canonicalize_inverse_ratio_powers(expr)
            except Exception:
                pass
            try:
                expr = canonicalize_trig_phases(expr, snap_rel_tol=5.0e-4)
            except Exception:
                pass
        except Exception:
            continue
        out.append(
            CandidateSpec(
                expr,
                label=label,
                rewrite_trace=[f"use analytic expression from {label}"],
                source_hints=[label],
            )
        )
    return out


def _split_single_log_term(term: Any) -> Optional[tuple[Any, Any]]:
    """Return ``(coefficient, log_arg)`` when ``term`` is coeff*log(arg)."""
    if sp is None:
        return None
    logs = [arg for arg in sp.Mul.make_args(term) if getattr(arg, "func", None) == sp.log]
    if len(logs) != 1:
        return None
    log_term = logs[0]
    try:
        coeff = sp.simplify(term / log_term)
    except Exception:
        coeff = term / log_term
    return coeff, log_term.args[0]


def _log_ratio_candidates(expr: Any) -> list[tuple[str, Any]]:
    """Combine matched log differences without pushing coefficients into exponents.

    SymPy's forceful logcombine can turn ``a*log(u) - a*log(v)`` into
    ``log((u/v)**a)``.  That is often numerically fine but dimensionally wrong
    when ``a`` is unitful.  This local rewrite keeps the unitful prefactor
    outside the logarithm.
    """
    if sp is None or not isinstance(expr, sp.Add):
        return []
    terms = list(sp.Add.make_args(expr))
    log_terms: list[tuple[int, Any, Any]] = []
    for i, term in enumerate(terms):
        split = _split_single_log_term(term)
        if split is not None:
            coeff, arg = split
            log_terms.append((i, coeff, arg))

    out: list[tuple[str, Any]] = []
    for a, (i, ci, ui) in enumerate(log_terms):
        for j, cj, uj in log_terms[a + 1:]:
            repl = None
            try:
                if sp.simplify(ci + cj) == 0:
                    ratio = sp.Mul(ui, sp.Pow(uj, -1, evaluate=False), evaluate=False)
                    repl = sp.Mul(ci, sp.log(ratio, evaluate=False), evaluate=False)
                elif sp.simplify(ci - cj) == 0:
                    product = sp.Mul(ui, uj, evaluate=False)
                    repl = sp.Mul(ci, sp.log(product, evaluate=False), evaluate=False)
            except Exception:
                repl = None
            if repl is None:
                continue
            rest = [t for k, t in enumerate(terms) if k not in (i, j)]
            cand = sp.Add(*(rest + [repl]), evaluate=False)
            out.append(("combine_log_ratio", cand))
    return out


def _is_half_exp(exp: Any) -> bool:
    try:
        return abs(float(exp) - 0.5) <= 1e-12 or abs(float(exp) + 0.5) <= 1e-12
    except Exception:
        return False


def _homogeneous_radical_candidates(
    expr: Any,
    variable_names: Sequence[str],
    assumptions: Mapping[str, str],
) -> tuple[list[CandidateSpec], list[InverseSqrtPolyTemplate]]:
    if sp is None:
        return [], []
    candidates: list[CandidateSpec] = []
    templates: list[InverseSqrtPolyTemplate] = []
    args = list(sp.Mul.make_args(expr)) if isinstance(expr, sp.Mul) else [expr]
    for i, arg in enumerate(args):
        if not isinstance(arg, sp.Pow) or not _is_half_exp(arg.exp):
            continue
        alpha = sp.Rational(-1, 2) if float(arg.exp) < 0 else sp.Rational(1, 2)
        P = arg.base
        if not getattr(P, "is_Add", False):
            continue
        syms = sorted(P.free_symbols, key=lambda s: str(s))
        if len(syms) < 2:
            continue
        try:
            poly = sp.Poly(P, *syms)
        except Exception:
            continue
        terms = poly.terms()
        totals = [sum(mon) for mon, _c in terms]
        if not totals or len(set(totals)) != 1:
            continue
        degree = int(totals[0])
        other = sp.Mul(*[a for j, a in enumerate(args) if j != i])
        for scale in syms:
            Q = sp.Integer(0)
            coeff_map: dict[int, Any] = {}
            usable_template = len(syms) == 2 and alpha == sp.Rational(-1, 2)
            other_sym = [s for s in syms if s != scale][0] if usable_template else None
            for mon, coeff in terms:
                term = coeff
                power_other = 0
                for s, p in zip(syms, mon):
                    p = int(p)
                    if s == scale:
                        continue
                    if p:
                        term *= (s / scale) ** p
                    if usable_template and s == other_sym:
                        power_other = p
                Q += term
                if usable_template:
                    coeff_map[power_other] = coeff_map.get(power_other, 0) + coeff
            try:
                new_other = sp.powsimp(other * scale ** (degree * alpha), force=True)
                new_expr = sp.powsimp(new_other * (Q ** alpha), force=True)
                new_expr = sp.simplify(new_expr)
            except Exception:
                continue
            ass = []
            if assumptions.get(str(scale)) == ">0":
                ass.append(f"{scale} > 0")
            else:
                ass.append(f"{scale} > 0 required for radical power cancellation")
            trace = [
                f"homogeneous radical projection: degree={degree}, scale={scale}",
                f"ratio coordinate(s): {', '.join(str(s / scale) for s in syms if s != scale)}",
            ]
            candidates.append(
                CandidateSpec(
                    new_expr,
                    label=f"homogeneous_radical:{scale}",
                    assumptions=ass,
                    rewrite_trace=trace,
                    source_hints=["homogeneous_radical"],
                )
            )
            if usable_template and other_sym is not None:
                max_power = max(coeff_map) if coeff_map else 0
                coeffs = [0.0] * (max_power + 1)
                ok = True
                for power, coeff in coeff_map.items():
                    try:
                        coeffs[int(power)] = float(coeff)
                    except Exception:
                        ok = False
                if ok:
                    templates.append(
                        InverseSqrtPolyTemplate(
                            prefactor=sp.simplify(new_other),
                            z_expr=sp.simplify(other_sym / scale),
                            coeffs=coeffs,
                            assumptions=ass,
                            source_hints=["homogeneous_radical"],
                        )
                    )
    return candidates, templates


def _infer_prefactor_from_stageb(stageb_expr: Optional[str], variable_names: Sequence[str]) -> Optional[Any]:
    if sp is None or not stageb_expr:
        return None
    text = _strip_ansi(stageb_expr)
    m = re.search(r"^\(?\s*(.*?)\s*\*\s*1/sqrt\(poly", text)
    if not m:
        return None
    pref = m.group(1).strip()
    if pref.startswith("(") and pref.endswith(")"):
        pref = pref[1:-1].strip()
    try:
        return parse_sympy_expr(pref, variable_names)
    except Exception:
        return None


def _artifact_templates(
    hints: Optional[ArtifactHints],
    variable_names: Sequence[str],
) -> list[InverseSqrtPolyTemplate]:
    if sp is None or hints is None:
        return []
    if "sqrt_poly" not in set(hints.accepted_patterns):
        return []
    if not hints.compound_exprs or not hints.initial_sqrt_poly_coeffs:
        return []
    pref = _infer_prefactor_from_stageb(hints.stageB_expr, variable_names)
    if pref is None:
        pref = sp.Symbol(variable_names[0], real=True) if variable_names else sp.Integer(1)
    out = []
    for comp in hints.compound_exprs[:3]:
        try:
            z = parse_sympy_expr(comp, variable_names)
        except Exception:
            continue
        out.append(
            InverseSqrtPolyTemplate(
                prefactor=pref,
                z_expr=z,
                coeffs=list(hints.initial_sqrt_poly_coeffs),
                assumptions=[f"{k} {v}" for k, v in hints.variable_assumptions.items()],
                source_hints=["artifact:sqrt_poly", f"artifact:compound:{sp.sstr(z)}"],
            )
        )
    return out


def _poly_from_coeffs(coeffs: Sequence[Any], z: Any) -> Any:
    out = sp.Integer(0)
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        out += c * (z ** int(i))
    return sp.expand(out)


def _candidate_from_template(
    template: InverseSqrtPolyTemplate,
    coeffs: Sequence[Any],
    *,
    label: str,
    n_free_params: int,
    n_snapped_consts: int = 0,
    trace: Optional[list[str]] = None,
    source_hints: Optional[list[str]] = None,
) -> CandidateSpec:
    P = _poly_from_coeffs(coeffs, template.z_expr)
    expr = template.prefactor / sp.sqrt(P)
    try:
        expr = _canonicalize_inverse_ratio_powers(sp.powsimp(expr, force=True))
    except Exception:
        pass
    return CandidateSpec(
        expr=expr,
        label=label,
        n_free_params=n_free_params,
        n_snapped_consts=n_snapped_consts,
        assumptions=list(template.assumptions),
        rewrite_trace=list(trace or [f"inverse-sqrt polynomial template: {label}"]),
        source_hints=list(template.source_hints) + list(source_hints or []),
    )


def _eval_template_part(
    expr: Any,
    X: np.ndarray,
    variable_names: Sequence[str],
    symbol_values: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    return _eval_expr_array(expr, X, variable_names, symbol_values)


def _fit_template_coeffs(
    template: InverseSqrtPolyTemplate,
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
    powers: Sequence[int],
    config: PolishConfig,
) -> Optional[list[float]]:
    pref = _eval_template_part(
        template.prefactor, X, variable_names, config.symbol_values
    )
    z = _eval_template_part(template.z_expr, X, variable_names, config.symbol_values)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pref) & np.isfinite(z) & np.isfinite(y) & (np.abs(y) > 1e-30)
    if np.count_nonzero(valid) < max(10, len(powers) + 5):
        return None
    target = (pref[valid] / y[valid]) ** 2
    valid2 = np.isfinite(target)
    if np.count_nonzero(valid2) < max(10, len(powers) + 5):
        return None
    z_t = torch.as_tensor(z[valid][valid2].reshape(-1, 1), dtype=torch.float64)
    y_t = torch.as_tensor(target[valid2], dtype=torch.float64)
    exps = torch.as_tensor([[int(p)] for p in powers], dtype=torch.int64)
    Phi = _eval_monomials(z_t, exps)
    try:
        coeff = ridge_lstsq(Phi, y_t, float(config.ridge))
    except Exception:
        return None
    try:
        proposal_noise_floor = 0.0
        try:
            sigma2 = float(config.noise_floor_raw)
            if math.isfinite(sigma2) and sigma2 > 0.0:
                pref_v = np.asarray(pref[valid][valid2], dtype=np.float64)
                y_v = np.asarray(y[valid][valid2], dtype=np.float64)
                deriv = -2.0 * (pref_v * pref_v) / np.maximum(np.abs(y_v) ** 3, 1.0e-300)
                deriv = deriv[np.isfinite(deriv)]
                if deriv.size:
                    proposal_noise_floor = float(sigma2 * np.mean(deriv * deriv))
        except Exception:
            proposal_noise_floor = 0.0
        coeff_sparse, meta = stlsq_sparsify_poly_coeffs(
            Phi=Phi,
            y=y_t,
            coeffs=coeff,
            cfg=replace(
                DEFAULT_POLY_STLSQ_CFG,
                proposal_noise_floor=max(0.0, float(proposal_noise_floor)),
            ),
        )
        if meta.get("accepted", 0.0) >= 0.5:
            coeff = coeff_sparse
    except Exception:
        pass
    full_len = max(max(powers) + 1 if powers else 0, len(template.coeffs))
    out = [0.0] * full_len
    for p, c in zip(powers, coeff.detach().cpu().numpy().tolist()):
        out[int(p)] = float(c)
    return out


def _fit_tied_scale(
    template: InverseSqrtPolyTemplate,
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
) -> Optional[float]:
    pref = _eval_template_part(
        template.prefactor, X, variable_names, config.symbol_values
    )
    z = _eval_template_part(template.z_expr, X, variable_names, config.symbol_values)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pref) & np.isfinite(z) & np.isfinite(y) & (np.abs(y) > 1e-30)
    if np.count_nonzero(valid) < 10:
        return None
    target = (pref[valid] / y[valid]) ** 2
    basis = 1.0 - z[valid] ** 2
    ok = np.isfinite(target) & np.isfinite(basis)
    if np.count_nonzero(ok) < 10:
        return None
    denom = float(np.dot(basis[ok], basis[ok]))
    if denom <= 1e-30:
        return None
    return float(np.dot(basis[ok], target[ok]) / denom)


def _nearest_snap(value: float, config: PolishConfig) -> Optional[Any]:
    best = None
    best_dist = float("inf")
    for t in config.snap_targets():
        tv = float(t.evalf()) if hasattr(t, "evalf") else float(t)
        dist = abs(float(value) - tv)
        tol = config.snap_rel_tol * max(1.0, abs(tv))
        if tv == 0.0:
            tol = max(tol, config.drop_rel_tol)
        if dist <= tol and dist < best_dist:
            best = t
            best_dist = dist
    return best


def _ratio_clean_equivalent(template: InverseSqrtPolyTemplate, variable_names: Sequence[str]) -> Optional[Any]:
    z = sp.factor(template.z_expr)
    num, den = sp.fraction(z)
    if not (num in template.z_expr.free_symbols and den in template.z_expr.free_symbols):
        return None
    try:
        expr = template.prefactor * den / sp.sqrt(den**2 - num**2)
        return sp.simplify(expr)
    except Exception:
        return None


def _template_candidates(
    template: InverseSqrtPolyTemplate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
    hints: Optional[ArtifactHints],
) -> list[CandidateSpec]:
    out: list[CandidateSpec] = []
    coeffs = list(template.coeffs)
    if not coeffs:
        return out
    degree = len(coeffs) - 1

    out.append(
        _candidate_from_template(
            template,
            coeffs,
            label="template_seed",
            n_free_params=len(coeffs),
            source_hints=["template_seed"],
        )
    )

    for fit_powers, label in [([0, 1, 2], "refit_full"), ([0, 2], "refit_drop_linear")]:
        if degree >= max(fit_powers):
            fit = _fit_template_coeffs(template, X_train, y_train, variable_names, fit_powers, config)
            if fit is not None:
                out.append(
                    _candidate_from_template(
                        template,
                        fit,
                        label=label,
                        n_free_params=len(fit_powers),
                        trace=[f"local inverse-sqrt polynomial refit: powers={fit_powers}"],
                        source_hints=["local_refit"],
                    )
                )

    if hints is not None:
        for ph in hints.coefficient_prune_hints:
            if ph.param_name == "coeffs" and 0 <= ph.index < len(coeffs):
                c2 = list(coeffs)
                c2[ph.index] = ph.target
                out.append(
                    _candidate_from_template(
                        template,
                        c2,
                        label=f"artifact_prune_coeff_{ph.index}",
                        n_free_params=max(0, len(coeffs) - 1),
                        trace=[f"artifact coefficient prune: coeff[{ph.index}] -> {ph.target:g}"],
                        source_hints=[f"prune:c{ph.index}->0"],
                    )
                )

    max_abs = max(abs(float(c)) for c in coeffs) if coeffs else 0.0
    for i, c in enumerate(coeffs):
        if i == 0:
            continue
        if abs(float(c)) <= config.drop_rel_tol * max(1.0, max_abs):
            c2 = list(coeffs)
            c2[i] = 0.0
            out.append(
                _candidate_from_template(
                    template,
                    c2,
                    label=f"drop_small_coeff_{i}",
                    n_free_params=max(0, len(coeffs) - 1),
                    trace=[f"drop small coefficient: c{i}={float(c):.6g} -> 0"],
                    source_hints=[f"drop:c{i}->0"],
                )
            )

    snapped: list[Any] = []
    n_snap = 0
    for c in coeffs:
        target = _nearest_snap(float(c), config)
        if target is None:
            snapped.append(float(c))
        else:
            snapped.append(target)
            n_snap += 1
    if n_snap:
        out.append(
            _candidate_from_template(
                template,
                snapped,
                label="snap_near_constants",
                n_free_params=max(0, len(coeffs) - n_snap),
                n_snapped_consts=n_snap,
                trace=["snap near constants using configured dictionary"],
                source_hints=["coefficient_snap"],
            )
        )

    if degree >= 2:
        a = _fit_tied_scale(template, X_train, y_train, variable_names, config)
        if a is not None and math.isfinite(a):
            tied = [a, 0.0, -a]
            out.append(
                _candidate_from_template(
                    template,
                    tied,
                    label="refit_tied_a_1_minus_z2",
                    n_free_params=1,
                    trace=["local refit tied denominator: a*(1 - z**2)"],
                    source_hints=["tie:c2=-c0"],
                )
            )
        clean = [sp.Integer(1), sp.Integer(0), sp.Integer(-1)]
        out.append(
            _candidate_from_template(
                template,
                clean,
                label="snap_clean_1_minus_z2",
                n_free_params=0,
                n_snapped_consts=3,
                trace=["fully snapped denominator: 1 - z**2"],
                source_hints=["clean:1-z**2"],
            )
        )
        eq = _ratio_clean_equivalent(template, variable_names)
        if eq is not None:
            out.append(
                CandidateSpec(
                    eq,
                    label="snap_clean_raw_ratio_equivalent",
                    n_free_params=0,
                    n_snapped_consts=3,
                    assumptions=list(template.assumptions),
                    rewrite_trace=["raw-coordinate equivalent of 1 - z**2"],
                    source_hints=list(template.source_hints) + ["clean:raw_ratio_equivalent"],
                )
            )
    return out


def _exp_poly_expr_from_coeffs(var: Any, coeffs: Sequence[Any]) -> Any:
    if len(coeffs) >= 3:
        try:
            c0_ok = sp.simplify(coeffs[0] + sp.log(2 * sp.pi) / 2) == 0
            c1_ok = sp.simplify(coeffs[1]) == 0
            c2_ok = sp.simplify(coeffs[2] + sp.Rational(1, 2)) == 0
            higher_ok = all(sp.simplify(c) == 0 for c in coeffs[3:])
            if c0_ok and c1_ok and c2_ok and higher_ok:
                return _normal_gaussian_expr(var)
        except Exception:
            pass
    exponent = sp.Integer(0)
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        exponent += c * (var ** int(i))
    return sp.exp(sp.expand(exponent))


def _normal_gaussian_expr(var: Any) -> Any:
    denom = sp.sqrt(sp.Mul(2, sp.pi, evaluate=False), evaluate=False)
    return sp.Mul(
        sp.exp(-var**2 / 2),
        sp.Pow(denom, -1, evaluate=False),
        evaluate=False,
    )


def _candidate_from_exp_poly(
    template: ExpPolyTemplate,
    coeffs: Sequence[Any],
    *,
    label: str,
    n_free_params: int,
    n_snapped_consts: int = 0,
    trace: Optional[list[str]] = None,
    source_hints: Optional[list[str]] = None,
) -> CandidateSpec:
    expr = _exp_poly_expr_from_coeffs(template.variable, coeffs)
    return CandidateSpec(
        expr=expr,
        label=label,
        n_free_params=n_free_params,
        n_snapped_consts=n_snapped_consts,
        assumptions=list(template.assumptions),
        rewrite_trace=list(trace or [f"exp-polynomial template: {label}"]),
        source_hints=list(template.source_hints) + list(source_hints or []),
    )


def _fit_exp_poly_templates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: Sequence[str],
    assumptions: Mapping[str, str],
    hints: Optional[ArtifactHints],
) -> list[ExpPolyTemplate]:
    if sp is None:
        return []
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if np.count_nonzero(np.isfinite(y) & (y > 0.0)) < max(50, 3 * len(variable_names)):
        return []
    source = ["exp_poly_projector"]
    if hints is not None:
        labels = {str(row.get("label", "")) for row in hints.candidate_family_hints}
        if any("exp" in label for label in labels):
            source.append("artifact:family:exp")
        if hints.y_transform:
            source.append(f"artifact:y_transform:{hints.y_transform}")
    out: list[ExpPolyTemplate] = []
    for axis, name in enumerate(variable_names):
        if axis >= X_train.shape[1]:
            continue
        valid = np.isfinite(X_train[:, axis]) & np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(valid) < 50:
            continue
        x_t = torch.as_tensor(X_train[valid, axis].reshape(-1, 1), dtype=torch.float64)
        log_y_t = torch.as_tensor(np.log(y[valid]), dtype=torch.float64)
        try:
            coeff = _fit_poly_coeffs_1d(x_t, log_y_t, degree=2, min_points=50)
        except Exception:
            coeff = None
        if coeff is None:
            continue
        coeffs = [float(c) for c in coeff.detach().cpu().numpy().reshape(-1).tolist()]
        if len(coeffs) < 3:
            coeffs = coeffs + [0.0] * (3 - len(coeffs))
        var = sp.Symbol(str(name), real=True)
        ass = []
        if assumptions.get(str(name)) == ">0":
            ass.append(f"{name} > 0")
        out.append(
            ExpPolyTemplate(
                variable=var,
                coeffs=coeffs[:3],
                assumptions=ass,
                source_hints=list(source) + [f"axis:{name}"],
            )
        )
    return out


def _fit_scale_for_basis(
    basis_expr: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: Sequence[str],
    symbol_values: Optional[Mapping[str, float]] = None,
) -> Optional[float]:
    try:
        basis = _eval_expr_array(
            basis_expr, X_train, variable_names, symbol_values
        )
    except Exception:
        return None
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    valid = np.isfinite(basis) & np.isfinite(y)
    if np.count_nonzero(valid) < 10:
        return None
    denom = float(np.dot(basis[valid], basis[valid]))
    if denom <= 1e-30:
        return None
    return float(np.dot(basis[valid], y[valid]) / denom)


def _exp_poly_candidates(
    template: ExpPolyTemplate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
) -> list[CandidateSpec]:
    coeffs = list(template.coeffs)
    if len(coeffs) < 3:
        return []
    out: list[CandidateSpec] = [
        _candidate_from_exp_poly(
            template,
            coeffs,
            label="exp_poly_logfit",
            n_free_params=3,
            source_hints=["log_y_poly_fit"],
        )
    ]

    c0, c1, c2 = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    if abs(c1) <= 5.0 * config.drop_rel_tol * max(1.0, abs(c0), abs(c2)):
        out.append(
            _candidate_from_exp_poly(
                template,
                [c0, 0.0, c2],
                label="exp_poly_drop_linear",
                n_free_params=2,
                trace=[f"drop small log-polynomial linear coefficient: c1={c1:.6g} -> 0"],
                source_hints=["drop:log_c1->0"],
            )
        )

    snapped = []
    n_snap = 0
    for c in coeffs:
        target = _nearest_snap(float(c), config)
        if target is None:
            snapped.append(float(c))
        else:
            snapped.append(target)
            n_snap += 1
    if n_snap:
        out.append(
            _candidate_from_exp_poly(
                template,
                snapped,
                label="exp_poly_snap_exponent_coeffs",
                n_free_params=max(0, 3 - n_snap),
                n_snapped_consts=n_snap,
                trace=["snap exp-polynomial exponent coefficients"],
                source_hints=["coefficient_snap:exp_exponent"],
            )
        )

    if abs(c2 + 0.5) <= 5.0 * config.snap_rel_tol * max(1.0, abs(c2)):
        half_quad_basis = sp.exp(-template.variable**2 / 2)
        scale = _fit_scale_for_basis(
            half_quad_basis,
            X_train,
            y_train,
            variable_names,
            config.symbol_values,
        )
        if scale is not None and math.isfinite(scale):
            out.append(
                CandidateSpec(
                    sp.Float(scale, 16) * half_quad_basis,
                    label="exp_poly_refit_gaussian_scale",
                    n_free_params=1,
                    assumptions=list(template.assumptions),
                    rewrite_trace=["snap exponent to -x**2/2 and refit multiplicative scale"],
                    source_hints=list(template.source_hints) + ["tie:quadratic_gaussian", "local_refit"],
                )
            )
            snap_scale = _nearest_snap(scale, config)
            if snap_scale is not None:
                try:
                    scale_expr = (
                        _normal_gaussian_expr(template.variable)
                        if sp.simplify(snap_scale - 1 / sp.sqrt(2 * sp.pi)) == 0
                        else sp.simplify(snap_scale * half_quad_basis)
                    )
                except Exception:
                    scale_expr = sp.simplify(snap_scale * half_quad_basis)
                out.append(
                    CandidateSpec(
                        scale_expr,
                        label="exp_poly_snap_gaussian_scale",
                        n_free_params=0,
                        n_snapped_consts=2,
                        assumptions=list(template.assumptions),
                        rewrite_trace=["snap exponent to -x**2/2 and scale to normal Gaussian constant"],
                        source_hints=list(template.source_hints)
                        + ["tie:quadratic_gaussian", "coefficient_snap:scale"],
                    )
                )
        if (
            abs(c1) <= 5.0 * config.drop_rel_tol * max(1.0, abs(c0), abs(c2))
            and abs(c0 + 0.5 * math.log(2.0 * math.pi)) <= 5.0 * config.snap_rel_tol * max(1.0, abs(c0))
        ):
            out.append(
                CandidateSpec(
                    _normal_gaussian_expr(template.variable),
                    label="exp_poly_snap_normal_gaussian",
                    n_free_params=0,
                    n_snapped_consts=3,
                    assumptions=list(template.assumptions),
                    rewrite_trace=["fully snapped Gaussian: exp(-x**2/2)/sqrt(2*pi)"],
                    source_hints=list(template.source_hints) + ["clean:normal_gaussian"],
                )
            )
    return out


def _score_candidate(
    spec: CandidateSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
    seed_pred_val: Optional[np.ndarray],
) -> Optional[CandidateRecord]:
    try:
        pred_tr = _eval_expr_array(
            spec.expr, X_train, variable_names, config.symbol_values
        )
        pred_va = _eval_expr_array(
            spec.expr, X_val, variable_names, config.symbol_values
        )
    except Exception:
        return None
    train_mse, _train_se, train_frac = _mse_and_se(pred_tr, y_train, config.min_valid_fraction)
    val_mse, val_se, val_frac = _mse_and_se(pred_va, y_val, config.min_valid_fraction)
    if not math.isfinite(val_mse):
        frac = min(train_frac, val_frac)
    else:
        frac = min(train_frac, val_frac)
    dist = _seed_distance(pred_va, seed_pred_val)
    cx, structural_cx, coefficient_cx = expression_cost_components(
        spec.expr,
        config,
        n_free_params=spec.n_free_params,
    )
    expr_s = _sstr(spec.expr)
    return CandidateRecord(
        expr=expr_s,
        display_expr=expr_s,
        label=spec.label,
        train_mse=float(train_mse),
        val_mse=float(val_mse),
        val_mse_se=float(val_se),
        complexity=float(cx),
        structural_complexity=float(structural_cx),
        coefficient_complexity=float(coefficient_cx),
        n_free_params=int(spec.n_free_params),
        n_snapped_consts=int(spec.n_snapped_consts),
        frac_valid=float(frac),
        seed_nrmse=dist,
        assumptions=list(spec.assumptions),
        source_hints=list(spec.source_hints),
        rewrite_trace=list(spec.rewrite_trace),
        distance_from_seed=dist,
        selection_n_free_params=spec.selection_n_free_params,
    )


def _epsilon_pareto_indices(
    records: Sequence[CandidateRecord],
    k: float,
    loss_equiv_abs_floor: float = 0.0,
) -> list[int]:
    abs_floor = max(float(loss_equiv_abs_floor), 0.0)
    keep = []
    for i, a in enumerate(records):
        if not (math.isfinite(a.val_mse) and math.isfinite(a.complexity)):
            continue
        dominated = False
        for j, b in enumerate(records):
            if i == j or not (math.isfinite(b.val_mse) and math.isfinite(b.complexity)):
                continue
            eps = max(
                float(k) * max(float(a.val_mse_se), float(b.val_mse_se), 0.0),
                abs_floor,
            )
            loss_no_worse = b.val_mse <= a.val_mse + eps
            cx_no_worse = b.complexity <= a.complexity
            strictly_better = b.val_mse < a.val_mse - eps or b.complexity < a.complexity
            if loss_no_worse and cx_no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    keep.sort(key=lambda idx: (records[idx].val_mse, records[idx].complexity, records[idx].expr))
    return keep


def _recommend(
    records: Sequence[CandidateRecord],
    k: float,
    loss_equiv_abs_floor: float = 0.0,
) -> Optional[CandidateRecord]:
    finite = [r for r in records if math.isfinite(r.val_mse) and math.isfinite(r.complexity)]
    if not finite:
        return None
    best = min(finite, key=lambda r: (r.val_mse, r.complexity))
    loss_tol = max(
        float(k) * max(best.val_mse_se, 0.0),
        max(float(loss_equiv_abs_floor), 0.0),
    )
    ceiling = best.val_mse + loss_tol
    eligible = [r for r in finite if r.val_mse <= ceiling]
    return min(
        eligible,
        key=lambda r: (r.complexity, r.n_free_params, len(r.expr), r.val_mse),
    )


def _loss_preserves_incumbent(
    *,
    candidate_mse: float,
    candidate_mse_se: float,
    incumbent_mse: float,
    incumbent_mse_se: float,
    k: float,
    abs_floor: float,
) -> tuple[bool, float]:
    """Return whether a candidate is no worse than a measured incumbent.

    Non-finite losses fail closed.  Non-finite standard errors do not create an
    unbounded acceptance window; only finite, non-negative uncertainty estimates
    contribute to the tolerance.
    """

    try:
        candidate_loss = float(candidate_mse)
        incumbent_loss = float(incumbent_mse)
    except Exception:
        return False, max(float(abs_floor or 0.0), 0.0)
    finite_ses = []
    for value in (candidate_mse_se, incumbent_mse_se):
        try:
            value_f = float(value)
        except Exception:
            continue
        if math.isfinite(value_f) and value_f >= 0.0:
            finite_ses.append(value_f)
    tol = max(
        max(float(abs_floor or 0.0), 0.0),
        max(float(k or 0.0), 0.0) * max(finite_ses, default=0.0),
    )
    if not (math.isfinite(candidate_loss) and math.isfinite(incumbent_loss)):
        return False, float(tol)
    return bool(candidate_loss <= incumbent_loss + tol), float(tol)


def _record_has_symbolic_snap_signal(rec: CandidateRecord) -> bool:
    """Return true for candidates that are plausible symbolic-constant snaps."""
    try:
        if int(getattr(rec, "n_snapped_consts", 0) or 0) > 0:
            return True
    except Exception:
        pass
    chunks = [
        str(getattr(rec, "label", "")),
        str(getattr(rec, "expr", "")),
        " ".join(str(x) for x in getattr(rec, "rewrite_trace", []) or []),
        " ".join(str(x) for x in getattr(rec, "source_hints", []) or []),
    ]
    text = " ".join(chunks).lower()
    if "snap" in text:
        return True
    if "drop_addend_refit" in text:
        return True
    symbolic_tokens = ("pi", "sqrt(2", "sqrt(pi", "sqrt(2*pi", " e ")
    return any(tok in text for tok in symbolic_tokens)


def _full_dataset_snap_targets(config: PolishConfig) -> list[Any]:
    """A slightly wider symbolic target set for the cheap full-data snap pass."""
    targets = list(config.snap_targets())
    if sp is not None:
        try:
            # Late full-data snapping can afford a few extra physics constants
            # without broadening the normal polish frontier.  The common miss is
            # a fitted decimal coefficient that is really a small rational
            # multiple of 1/pi, e.g. 3/(20*pi).
            for q in range(1, 33):
                for p in range(-32, 33):
                    if p:
                        targets.append(sp.Rational(p, q) / sp.pi)
        except Exception:
            pass
    try:
        from nestynet_sr.sr_search.polish_utils import dedupe_sympy_exprs

        return dedupe_sympy_exprs(targets)
    except Exception:
        return targets


def _small_rational_lattice_targets(
    value: float,
    *,
    max_denominator: int = 32,
    max_abs_numerator: int = 512,
) -> list[Any]:
    """Return nearby low-denominator rationals around a fitted coefficient."""
    if sp is None or not math.isfinite(float(value)):
        return []
    out: list[Any] = []
    seen: set[tuple[int, int]] = set()

    def add(frac: Fraction) -> None:
        if frac.denominator <= 0:
            return
        if frac.denominator > int(max_denominator):
            return
        if abs(frac.numerator) > int(max_abs_numerator):
            return
        key = (int(frac.numerator), int(frac.denominator))
        if key in seen:
            return
        seen.add(key)
        out.append(sp.Rational(frac.numerator, frac.denominator))

    try:
        add(Fraction(float(value)).limit_denominator(int(max_denominator)))
    except Exception:
        pass
    for q in range(1, int(max_denominator) + 1):
        center = int(round(float(value) * q))
        for p in (center - 1, center, center + 1):
            if p == 0:
                continue
            try:
                add(Fraction(int(p), int(q)))
            except Exception:
                continue
    out.sort(
        key=lambda r: (
            abs(float(r) - float(value)),
            int(sp.denom(r)),
            abs(int(sp.numer(r))),
        )
    )
    return out


def _global_scale_factorizations(expr: Any) -> list[tuple[Any, Any]]:
    """Find conservative ``coefficient * basis`` views of an expression."""
    if sp is None:
        return []
    variants = [expr]
    for fn in (sp.factor_terms, sp.factor):
        try:
            ex = fn(expr)
            if ex not in variants:
                variants.append(ex)
        except Exception:
            pass

    out: list[tuple[Any, Any]] = []
    seen: set[str] = set()
    for ex in variants:
        try:
            coeff, basis = ex.as_coeff_Mul()
        except Exception:
            continue
        try:
            if basis == 1 or not getattr(basis, "free_symbols", set()):
                continue
            if not bool(getattr(coeff, "is_number", False)):
                continue
            coeff_f = float(coeff)
            if not math.isfinite(coeff_f) or abs(coeff_f) <= 1.0e-300:
                continue
            key = _canonical_key(basis)
            if key in seen:
                continue
            seen.add(key)
            out.append((coeff, basis))
        except Exception:
            continue
    return out


def _full_dataset_coefficient_lattice_snap_specs(
    current_expr: Any,
    X_full: np.ndarray,
    y_full: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
) -> list[CandidateSpec]:
    """Generate full-data small-rational snaps for a fixed global scale.

    This is deliberately narrower than a new model search: only the scalar
    multiplying the current symbolic basis is refit, and only nearby
    low-denominator rationals are placed on the final full-data ballot.
    """
    if sp is None:
        return []
    X_arr = np.asarray(X_full, dtype=np.float64)
    y_arr = np.asarray(y_full, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2 or y_arr.size == 0:
        return []

    specs: list[CandidateSpec] = []
    seen_exprs: set[str] = set()
    for coeff, basis in _global_scale_factorizations(current_expr):
        try:
            basis_pred = _eval_expr_array(
                basis, X_arr, variable_names, config.symbol_values
            )
        except Exception:
            continue
        basis_pred = np.asarray(basis_pred, dtype=np.float64).reshape(-1)
        n = min(basis_pred.size, y_arr.size)
        if n <= 2:
            continue
        b = basis_pred[:n]
        y = y_arr[:n]
        valid = np.isfinite(b) & np.isfinite(y)
        if float(np.count_nonzero(valid)) / float(max(1, n)) < float(config.min_valid_fraction):
            continue
        b = b[valid]
        y = y[valid]
        denom = float(np.dot(b, b))
        if not (math.isfinite(denom) and denom > 1.0e-300):
            continue
        coeff_old = float(coeff)
        coeff_fit = float(np.dot(b, y) / denom)
        if not math.isfinite(coeff_fit):
            continue
        rel_shift = abs(coeff_fit - coeff_old) / max(1.0, abs(coeff_old), abs(coeff_fit))
        if rel_shift > 5.0e-2:
            continue
        targets = _small_rational_lattice_targets(coeff_fit)
        for target in targets[:64]:
            try:
                target_f = float(target)
            except Exception:
                continue
            if not math.isfinite(target_f):
                continue
            if abs(target_f - coeff_old) <= 1.0e-14 * max(1.0, abs(coeff_old)):
                continue
            # Keep the generated ballot local to the full-data refit.
            if abs(target_f - coeff_fit) > max(
                10.0 * abs(coeff_fit - coeff_old),
                2.5e-2 * max(1.0, abs(coeff_fit)),
            ):
                continue
            try:
                cand_expr = _canonicalize_guarded_candidate_expr(sp.simplify(target * basis), config)
                key = _canonical_key(cand_expr)
            except Exception:
                continue
            if key in seen_exprs:
                continue
            seen_exprs.add(key)
            specs.append(
                CandidateSpec(
                    cand_expr,
                    label="full_dataset_coeff_rational_snap",
                    n_free_params=0,
                    n_snapped_consts=1,
                    rewrite_trace=[
                        "full-dataset coefficient lattice snap",
                        f"refit scalar {coeff_old:.12g} -> {coeff_fit:.12g}; snapped to {sp.sstr(target)}",
                    ],
                    source_hints=["full_dataset_snap", "coefficient_lattice_snap"],
                )
            )
            if len(specs) >= 64:
                return specs
    return specs


def _exp_quadratic_factorizations(
    expr: Any,
    variable_names: Sequence[str],
) -> list[tuple[Any, Any, float, float]]:
    """Return ``(var, scale, c1, c2)`` views of ``scale*exp(c1*x+c2*x**2)``.

    This is intentionally narrow.  It is used only as final-polish proposal
    fuel, and the generated expressions are still scored on the data before
    they can win.
    """

    if sp is None:
        return []
    try:
        ex = sp.powsimp(expr, force=True)
    except Exception:
        ex = expr
    try:
        factors = sp.Mul.make_args(ex)
    except Exception:
        factors = (ex,)

    scale = sp.Integer(1)
    exponent = sp.Integer(0)
    found_exp = False
    for factor in factors:
        try:
            if getattr(factor, "func", None) == sp.exp and len(factor.args) == 1:
                exponent += factor.args[0]
                found_exp = True
                continue
            if (
                isinstance(factor, sp.Pow)
                and getattr(factor.base, "func", None) == sp.exp
                and len(factor.base.args) == 1
                and bool(getattr(factor.exp, "is_number", False))
            ):
                exponent += factor.exp * factor.base.args[0]
                found_exp = True
                continue
        except Exception:
            pass
        scale *= factor
    if not found_exp:
        return []

    symbols = _symbols_for_expr(ex, variable_names)
    symbol_set = set(symbols)
    out: list[tuple[Any, Any, float, float]] = []
    for var in symbols:
        try:
            if set(getattr(scale, "free_symbols", set())) & symbol_set:
                continue
            expanded = sp.expand(exponent)
            if set(getattr(expanded, "free_symbols", set())) - {var}:
                continue
            poly = sp.Poly(expanded, var)
            if int(poly.degree()) > 2:
                continue
            c1 = float(sp.N(poly.coeff_monomial(var), 18))
            c2 = float(sp.N(poly.coeff_monomial(var**2), 18))
            scale_f = float(sp.N(scale, 18))
            if not (math.isfinite(c1) and math.isfinite(c2) and math.isfinite(scale_f)):
                continue
            if scale_f <= 0.0:
                continue
        except Exception:
            continue
        out.append((var, sp.simplify(scale), c1, c2))
    return out


def _nearby_full_dataset_targets(
    value: float,
    config: PolishConfig,
    *,
    rel_tol: float,
    abs_tol: float = 0.0,
    limit: int = 6,
) -> list[Any]:
    """Return symbolic targets close enough to a fitted scalar for proposal use."""

    if sp is None or not math.isfinite(float(value)):
        return []
    out: list[tuple[float, float, Any]] = []
    for target in _full_dataset_snap_targets(config):
        try:
            target_f = float(target)
        except Exception:
            continue
        if not math.isfinite(target_f):
            continue
        tol = max(float(abs_tol), float(rel_tol) * max(1.0, abs(float(value)), abs(target_f)))
        if abs(float(value) - target_f) <= tol:
            try:
                cost = float(expression_complexity(target, config))
            except Exception:
                cost = float(len(sp.sstr(target)))
            out.append((cost, abs(float(value) - target_f), target))
    out.sort(key=lambda row: (row[0], row[1], len(sp.sstr(row[2]))))
    return [target for _cost, _dist, target in out[: int(limit)]]


def _sqrt_factorizations(expr: Any) -> list[tuple[Any, Any, Any, int, float]]:
    """Return ``(rest, radicand, sqrt_factor, sign, abs_scale)`` radical views.

    The view is intentionally conservative and handles a single visible radical
    factor with a pure numeric outer scale:

        c * rest * sqrt(P)      or      c * rest / sqrt(P)

    SymPy commonly preserves the latter as ``sqrt(1/P)`` rather than
    ``P**(-1/2)``.  That exact nested reciprocal is exposed as the same
    inverse-square-root view; more general nested powers are left untouched.

    The generated full-dataset ballot later decides whether any snap is useful.
    """

    if sp is None:
        return []
    try:
        variants = [expr, sp.powsimp(expr, force=True), sp.factor_terms(expr)]
    except Exception:
        variants = [expr]
    out: list[tuple[Any, Any, Any, int, float]] = []
    seen: set[str] = set()
    for ex in variants:
        try:
            args = list(sp.Mul.make_args(ex)) if isinstance(ex, sp.Mul) else [ex]
        except Exception:
            args = [ex]
        for idx, factor in enumerate(args):
            if not isinstance(factor, sp.Pow):
                continue
            try:
                exp_f = float(factor.exp)
            except Exception:
                continue
            if abs(abs(exp_f) - 0.5) > 1.0e-12:
                continue
            radicand = factor.base
            effective_factor = factor
            if (
                exp_f > 0.0
                and isinstance(radicand, sp.Pow)
                and radicand.exp == -1
                and getattr(radicand.base, "is_Add", False)
            ):
                radicand = radicand.base
                exp_f = -0.5
                effective_factor = sp.Pow(
                    radicand,
                    sp.Rational(-1, 2),
                    evaluate=False,
                )
            if not getattr(radicand, "is_Add", False):
                continue
            other = sp.Mul(*[a for j, a in enumerate(args) if j != idx])
            try:
                coeff, rest = other.as_coeff_Mul()
                if not bool(getattr(coeff, "is_number", False)):
                    continue
                coeff_f = float(sp.N(coeff, 18))
            except Exception:
                continue
            if not (math.isfinite(coeff_f) and abs(coeff_f) > 1.0e-300):
                continue
            sign = -1 if coeff_f < 0.0 else 1
            abs_scale = abs(coeff_f)
            # No coupled scale to move across the radical.
            if abs(abs_scale - 1.0) <= 1.0e-14:
                continue
            key = _canonical_key((sp.simplify(rest), sp.simplify(radicand), exp_f, sign))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                (
                    sp.simplify(rest),
                    sp.expand(radicand),
                    effective_factor,
                    sign,
                    abs_scale,
                )
            )
    return out


def _radical_coefficient_ratio_snap_specs(
    expr: Any,
    variable_names: Sequence[str],
    config: PolishConfig,
    *,
    max_ops: int = 128,
    max_terms: int = 24,
    max_anchors: int = 6,
    max_candidates: int = 12,
) -> list[CandidateSpec]:
    """Expose and snap a numeric gauge shared across a square-root radicand.

    A fitted radical can be exact while retaining an arbitrary scale split,
    for example ``c*rest/sqrt(a*u + b*v)``.  Dividing the radicand by an
    existing positive coefficient changes the outer scale algebraically and
    can expose both small coefficient ratios and a compact symbolic prefactor.

    This proposal is deliberately narrow and truth-blind:

    - only one of the conservative views from :func:`_sqrt_factorizations`;
    - an additive radicand with bounded size and purely numeric coefficients;
    - no symbols beyond the declared input variables;
    - every normalized coefficient ratio must be close to a small rational;
    - the displaced outer scale must be close to an existing polish target;
    - symbolic complexity must strictly decrease.

    The normal units, validation-loss, incumbent-preservation, and Pareto gates
    in :func:`polish_expression` remain authoritative.
    """

    if sp is None or int(max_candidates) <= 0:
        return []
    try:
        if int(sp.count_ops(expr, visual=False)) > int(max_ops):
            return []
        if not any(
            isinstance(power, sp.Pow) and _is_half_exp(power.exp)
            for power in expr.atoms(sp.Pow)
        ):
            return []
    except Exception:
        return []

    radical_views = _sqrt_factorizations(expr)
    if not radical_views:
        return []

    by_name = {str(sym): sym for sym in getattr(expr, "free_symbols", set())}
    declared_symbols = {
        by_name[str(name)]
        for name in variable_names
        if str(name) in by_name
    }
    ratio_target_rows: list[tuple[float, float, str, Any]] = []
    for target in rational_snap_targets(max_denominator=8):
        try:
            target_text = sp.sstr(target)
            ratio_target_rows.append(
                (
                    float(target),
                    float(len(target_text)),
                    target_text,
                    target,
                )
            )
        except Exception:
            continue
    scale_target_rows: list[tuple[float, float, str, Any]] = []
    for target in config.snap_targets():
        try:
            target_f = float(target)
            if not math.isfinite(target_f):
                continue
            target_text = sp.sstr(target)
            scale_target_rows.append(
                (
                    target_f,
                    float(len(target_text)),
                    target_text,
                    target,
                )
            )
        except Exception:
            continue
    base_complexity = expression_complexity(expr, config)
    base_key = _canonical_key(expr)
    out: list[CandidateSpec] = []
    seen_exprs: set[str] = set()

    for rest, radicand, factor, sign, abs_scale in radical_views:
        try:
            if set(radicand.free_symbols) - declared_symbols:
                continue
            exp_f = float(factor.exp)
            exponent = (
                sp.Rational(1, 2)
                if exp_f > 0.0
                else sp.Rational(-1, 2)
            )
            terms = list(sp.Add.make_args(sp.expand(radicand)))
            if len(terms) < 2 or len(terms) > int(max_terms):
                continue
        except Exception:
            continue

        split_terms: list[tuple[Any, Any, float]] = []
        coefficient_ok = True
        for term in terms:
            try:
                coeff, basis = term.as_coeff_Mul()
                if not bool(getattr(coeff, "is_number", False)):
                    coefficient_ok = False
                    break
                coeff_f = float(sp.N(coeff, 18))
                if not math.isfinite(coeff_f) or coeff_f == 0.0:
                    coefficient_ok = False
                    break
            except Exception:
                coefficient_ok = False
                break
            split_terms.append((coeff, basis, coeff_f))
        if not coefficient_ok:
            continue

        anchors: list[tuple[Any, float]] = []
        seen_anchors: set[str] = set()
        for coeff, _basis, coeff_f in split_terms:
            if coeff_f <= 0.0:
                continue
            key = sp.srepr(coeff)
            if key in seen_anchors:
                continue
            seen_anchors.add(key)
            anchors.append((coeff, coeff_f))
            if len(anchors) >= int(max_anchors):
                break

        for anchor, anchor_f in anchors:
            snapped_terms: list[Any] = []
            ratio_trace: list[str] = []
            ratios_ok = True
            for coeff, basis, coeff_f in split_terms:
                ratio_f = coeff_f / anchor_f
                ranked = sorted(
                    ratio_target_rows,
                    key=lambda row: (
                        abs(row[0] - ratio_f),
                        row[1],
                        row[2],
                    ),
                )
                target_row = ranked[0] if ranked else None
                if target_row is None:
                    ratios_ok = False
                    break
                target_f, _target_cx, _target_text, target = target_row
                tolerance = float(config.snap_rel_tol) * max(
                    1.0,
                    abs(ratio_f),
                    abs(target_f),
                )
                if (
                    target == 0
                    or abs(ratio_f - target_f) > tolerance
                ):
                    ratios_ok = False
                    break
                snapped_terms.append(target * basis)
                ratio_trace.append(
                    f"{coeff_f:.12g}/{anchor_f:.12g}->{sp.sstr(target)}"
                )
            if not ratios_ok:
                continue

            try:
                normalized_radicand = sp.Add(*snapped_terms)
                combined_scale = (
                    float(sign)
                    * float(abs_scale)
                    * float(anchor_f) ** float(exp_f)
                )
                if not math.isfinite(combined_scale):
                    continue
                nearby_scales = [
                    row
                    for row in scale_target_rows
                    if abs(row[0] - combined_scale)
                    <= float(config.snap_rel_tol)
                    * max(1.0, abs(row[0]), abs(combined_scale))
                ]
                nearby_scales.sort(
                    key=lambda row: (
                        abs(row[0] - combined_scale),
                        row[1],
                        row[2],
                    )
                )
            except Exception:
                continue

            seen_scales: set[str] = set()
            for _scale_f, _scale_cx, _scale_text, snapped_scale in nearby_scales[:6]:
                scale_key = _canonical_key(snapped_scale)
                if scale_key in seen_scales:
                    continue
                seen_scales.add(scale_key)
                try:
                    radical = sp.Pow(
                        normalized_radicand,
                        exponent,
                        evaluate=False,
                    )
                    candidate = sp.Mul(
                        snapped_scale,
                        rest,
                        radical,
                        evaluate=False,
                    )
                    candidate = _canonicalize_guarded_candidate_expr(
                        sp.powsimp(candidate, force=True),
                        config,
                    )
                    key = _canonical_key(candidate)
                    if key == base_key or key in seen_exprs:
                        continue
                    if expression_complexity(candidate, config) >= base_complexity:
                        continue
                except Exception:
                    continue
                seen_exprs.add(key)
                out.append(
                    CandidateSpec(
                        candidate,
                        label=(
                            "radical_coefficient_ratio_snap"
                            f"|snap_outer_scale:{combined_scale:.12g}"
                            f"->{sp.sstr(snapped_scale)}"
                        ),
                        n_free_params=0,
                        n_snapped_consts=int(len(split_terms) + 1),
                        rewrite_trace=[
                            "normalize radical by an existing positive numeric coefficient",
                            "small-rational radicand ratios: "
                            + ", ".join(ratio_trace),
                            (
                                f"combine outer scale with radical gauge "
                                f"{combined_scale:.12g}->{sp.sstr(snapped_scale)}"
                            ),
                        ],
                        source_hints=["radical_coefficient_ratio_snap"],
                    )
                )
                if len(out) >= int(max_candidates):
                    return out
    return out


def _snap_additive_numeric_coefficients(
    expr: Any,
    config: PolishConfig,
    *,
    rel_tol: float,
    zero_abs_tol: float,
) -> tuple[Optional[Any], int, list[str]]:
    """Snap numeric coefficients in an additive expression without refitting."""

    if sp is None:
        return None, 0, []
    try:
        expanded = sp.expand(expr)
        terms = list(sp.Add.make_args(expanded)) if isinstance(expanded, sp.Add) else [expanded]
    except Exception:
        return None, 0, []
    snapped_terms: list[Any] = []
    traces: list[str] = []
    n_snap = 0
    max_coeff = 0.0
    split_terms: list[tuple[Any, Any, Optional[float]]] = []
    for term in terms:
        try:
            coeff, basis = term.as_coeff_Mul()
            if bool(getattr(coeff, "is_number", False)):
                coeff_f = float(sp.N(coeff, 18))
            else:
                coeff_f = None
        except Exception:
            coeff, basis, coeff_f = sp.Integer(1), term, None
        if coeff_f is not None and math.isfinite(coeff_f):
            max_coeff = max(max_coeff, abs(coeff_f))
        split_terms.append((coeff, basis, coeff_f))

    zero_tol = max(float(zero_abs_tol), float(config.drop_rel_tol) * max(1.0, max_coeff))
    for coeff, basis, coeff_f in split_terms:
        if coeff_f is None or not math.isfinite(coeff_f):
            snapped_terms.append(coeff * basis)
            continue
        targets = _nearby_full_dataset_targets(
            coeff_f,
            config,
            rel_tol=float(rel_tol),
            abs_tol=zero_tol if abs(coeff_f) <= max(1.0, max_coeff) else 0.0,
            limit=8,
        )
        if targets:
            target = targets[0]
            try:
                target_f = float(sp.N(target, 18))
            except Exception:
                target_f = coeff_f
            if abs(target_f - coeff_f) > 1.0e-14 * max(1.0, abs(coeff_f), abs(target_f)):
                n_snap += 1
                traces.append(f"{coeff_f:.12g}->{sp.sstr(target)}")
                coeff = target
        snapped_terms.append(coeff * basis)
    if n_snap <= 0:
        return None, 0, []
    try:
        snapped_expr = sp.expand(sp.Add(*snapped_terms))
    except Exception:
        snapped_expr = sp.Add(*snapped_terms)
    return snapped_expr, int(n_snap), traces


def _full_dataset_radical_gauge_snap_specs(
    current_expr: Any,
    X_full: np.ndarray,
    y_full: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
) -> list[CandidateSpec]:
    """Generate coupled constant snaps by moving scales across ``sqrt``.

    This catches expressions such as ``A*sqrt(a*z**2 + b*z - 1)`` where the
    independent fitted constants obscure the simpler gauge
    ``sqrt(z**2 - pi**2)``.  It is proposal-only; the full-dataset adjudicator
    still accepts a candidate only when its raw loss is tied with the best.
    """

    del X_full, y_full, variable_names  # scoring happens in the caller
    if sp is None:
        return []
    specs: list[CandidateSpec] = []
    seen: set[str] = set()
    for rest, radicand, factor, sign, abs_scale in _sqrt_factorizations(current_expr):
        try:
            exp_f = float(factor.exp)
        except Exception:
            continue
        scale_sq = float(abs_scale) ** 2
        if not math.isfinite(scale_sq) or scale_sq <= 0.0:
            continue
        if exp_f > 0.0:
            normalized = sp.expand(sp.Float(scale_sq, 16) * radicand)
            exponent = sp.Rational(1, 2)
            trace_head = f"push outer scale {abs_scale:.12g} into sqrt"
        else:
            normalized = sp.expand(radicand / sp.Float(scale_sq, 16))
            exponent = sp.Rational(-1, 2)
            trace_head = f"push outer scale {abs_scale:.12g} into inverse sqrt"
        snapped, n_snap, coeff_trace = _snap_additive_numeric_coefficients(
            normalized,
            config,
            rel_tol=max(5.0e-3, float(config.snap_rel_tol)),
            zero_abs_tol=max(5.0e-3, 5.0 * float(config.drop_rel_tol)),
        )
        if snapped is None or n_snap <= 0:
            continue
        try:
            radical = sp.Pow(snapped, exponent, evaluate=False)
            cand_expr = sp.Mul(sp.Integer(sign), rest, radical, evaluate=False)
            cand_expr = _canonicalize_guarded_candidate_expr(sp.powsimp(cand_expr, force=True), config)
            key = _canonical_key(cand_expr)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            CandidateSpec(
                cand_expr,
                label="full_dataset_radical_gauge_snap",
                n_free_params=0,
                n_snapped_consts=int(n_snap + 1),
                rewrite_trace=[
                    "full-dataset radical gauge coefficient snap",
                    trace_head,
                    "normalized radicand coefficients: " + ", ".join(coeff_trace),
                ],
                source_hints=["full_dataset_snap", "radical_gauge_snap"],
            )
        )
        if len(specs) >= 24:
            return specs
    return specs


def _full_dataset_exp_quadratic_snap_specs(
    current_expr: Any,
    X_full: np.ndarray,
    y_full: np.ndarray,
    variable_names: Sequence[str],
    config: PolishConfig,
) -> list[CandidateSpec]:
    """Generate final-ballot coefficient snaps for ``scale*exp(a*x+b*x**2)``.

    The generator is intentionally family-local: Stage B must already have found
    an exponential quadratic.  This only proposes nearby symbolic coefficients
    and refit/snap scales, then normal full-data scoring decides.
    """

    if sp is None:
        return []
    specs: list[CandidateSpec] = []
    seen: set[str] = set()
    for var, _scale, c1, c2 in _exp_quadratic_factorizations(current_expr, variable_names):
        c1_targets: list[Any] = []
        if abs(c1) <= 0.15:
            c1_targets.append(sp.Integer(0))
        c1_targets.extend(
            _nearby_full_dataset_targets(
                c1,
                config,
                rel_tol=7.5e-2,
                abs_tol=2.0e-2,
                limit=4,
            )
        )
        c2_targets = _nearby_full_dataset_targets(
            c2,
            config,
            rel_tol=7.5e-2,
            abs_tol=2.0e-2,
            limit=6,
        )
        coeff_pairs: list[tuple[Any, Any]] = []
        for c1_t in c1_targets:
            for c2_t in c2_targets:
                try:
                    if abs(float(c1_t) - c1) <= 1.0e-14 and abs(float(c2_t) - c2) <= 1.0e-14:
                        continue
                except Exception:
                    pass
                coeff_pairs.append((c1_t, c2_t))
        for c1_t, c2_t in coeff_pairs[:12]:
            try:
                exponent = sp.expand(c1_t * var + c2_t * var**2)
                basis = sp.exp(exponent)
            except Exception:
                continue
            scale_fit = _fit_scale_for_basis(
                basis,
                X_full,
                y_full,
                variable_names,
                config.symbol_values,
            )
            if scale_fit is not None and math.isfinite(scale_fit):
                fit_expr = sp.Float(scale_fit, 16) * basis
                try:
                    fit_expr = _canonicalize_guarded_candidate_expr(fit_expr, config)
                    key = _canonical_key(fit_expr)
                except Exception:
                    key = str(fit_expr)
                if key not in seen:
                    seen.add(key)
                    specs.append(
                        CandidateSpec(
                            fit_expr,
                            label="full_dataset_exp_quadratic_refit_scale",
                            n_free_params=1,
                            rewrite_trace=[
                                "full-dataset exp-quadratic coefficient snap with refit scale",
                                (
                                    f"c1={c1:.12g}->{sp.sstr(c1_t)}, "
                                    f"c2={c2:.12g}->{sp.sstr(c2_t)}, "
                                    f"scale={scale_fit:.12g}"
                                ),
                            ],
                            source_hints=["full_dataset_snap", "exp_quadratic_coeff_snap"],
                        )
                    )
                for scale_t in _nearby_full_dataset_targets(
                    scale_fit,
                    config,
                    rel_tol=5.0e-2,
                    abs_tol=1.0e-3,
                    limit=4,
                ):
                    try:
                        snapped_expr = _canonicalize_guarded_candidate_expr(
                            sp.simplify(scale_t * basis),
                            config,
                        )
                        key = _canonical_key(snapped_expr)
                    except Exception:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    specs.append(
                        CandidateSpec(
                            snapped_expr,
                            label="full_dataset_exp_quadratic_coeff_snap",
                            n_free_params=0,
                            n_snapped_consts=3,
                            rewrite_trace=[
                                "full-dataset exp-quadratic coefficient and scale snap",
                                (
                                    f"c1={c1:.12g}->{sp.sstr(c1_t)}, "
                                    f"c2={c2:.12g}->{sp.sstr(c2_t)}, "
                                    f"scale={scale_fit:.12g}->{sp.sstr(scale_t)}"
                                ),
                            ],
                            source_hints=["full_dataset_snap", "exp_quadratic_coeff_snap"],
                        )
                    )
            if len(specs) >= 24:
                return specs
    return specs


def apply_full_dataset_snap_adjudication(
    result: PolishResult,
    X_full: np.ndarray,
    y_full: np.ndarray,
    *,
    variable_names: Sequence[str],
    config: Optional[PolishConfig] = None,
    units_spec: Optional[Any] = None,
) -> tuple[PolishResult, dict[str, Any]]:
    """Use all available rows to adjudicate final symbolic-constant snaps.

    The normal final polisher scores a compact candidate frontier on a
    train/validation split that is often capped for speed.  In noisy benchmark
    runs the difference between a learned float constant and a nearby symbolic
    constant can be smaller than that split's sampling noise.  This pass keeps
    the generated frontier fixed, evaluates the current recommendation,
    snap-like candidates, and the raw seed baseline on the full CSV, and lets
    the simpler snap win when it is statistically tied with the best full-data
    candidate.  Retaining the seed in this ballot lets a unit-valid incumbent
    survive when a split-selected snap regresses on the full dataset.
    """

    config = config or PolishConfig()
    coefficient_metadata_norm = None
    artifact_hints = getattr(result, "artifact_hints", None)
    coefficient_metadata = getattr(artifact_hints, "coefficient_metadata", None)
    if coefficient_metadata is not None:
        coefficient_metadata_norm = normalize_coefficient_metadata(
            coefficient_metadata,
            variable_names=variable_names,
            require_values=True,
            units_spec=units_spec,
        )
        artifact_values = coefficient_symbol_values(
            coefficient_metadata_norm,
            variable_names=variable_names,
            units_spec=units_spec,
        )
        merged_values = dict(artifact_values)
        for name, value in config.symbol_values.items():
            if name in merged_values and not math.isclose(
                float(value),
                float(merged_values[name]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    f"coefficient symbol {name!r} has conflicting artifact and "
                    "configuration values"
                )
            merged_values[name] = value
        config = replace(config, symbol_values=merged_values)
    summary: dict[str, Any] = {
        "enabled": True,
        "status": "skipped",
        "reason": None,
    }
    if result is None or not getattr(result, "all_candidates", None):
        summary["reason"] = "no final-polish candidates"
        return result, summary
    if getattr(result, "recommended", None) is None:
        summary["reason"] = "no final-polish recommendation"
        return result, summary

    X_arr = np.asarray(X_full, dtype=np.float64)
    y_arr = np.asarray(y_full, dtype=np.float64).reshape(-1)
    n_full = int(min(X_arr.shape[0], y_arr.size)) if X_arr.ndim >= 2 else 0
    if n_full <= 0:
        summary["reason"] = "empty full dataset"
        return result, summary
    X_arr = X_arr[:n_full]
    y_arr = y_arr[:n_full]

    seed_full_mse = float("inf")
    seed_full_mse_se = float("inf")
    seed_full_frac_valid = 0.0
    seed_full_error = None
    try:
        seed_expr = parse_sympy_expr(str(result.seed_expr), variable_names)
        seed_pred = _eval_expr_array(
            seed_expr, X_arr, variable_names, config.symbol_values
        )
        seed_full_mse, seed_full_mse_se, seed_full_frac_valid = _mse_and_se(
            seed_pred,
            y_arr,
            config.min_valid_fraction,
        )
    except Exception as exc:
        seed_full_error = str(exc)
    summary.update(
        {
            "seed_full_mse": (
                float(seed_full_mse) if math.isfinite(seed_full_mse) else None
            ),
            "seed_full_mse_se": (
                float(seed_full_mse_se) if math.isfinite(seed_full_mse_se) else None
            ),
            "seed_full_frac_valid": float(seed_full_frac_valid),
            "seed_full_error": seed_full_error,
        }
    )

    current = result.recommended
    candidates: list[CandidateRecord] = []
    seen_ids: set[int] = set()
    seen_exprs: set[str] = set()
    for rec in result.all_candidates:
        try:
            seen_exprs.add(_canonical_key(parse_sympy_expr(str(rec.expr), variable_names)))
        except Exception:
            seen_exprs.add(str(rec.expr))
        if rec is current or _record_has_symbolic_snap_signal(rec):
            ident = id(rec)
            if ident not in seen_ids:
                seen_ids.add(ident)
                candidates.append(rec)
    if current is not None and id(current) not in seen_ids:
        seen_ids.add(id(current))
        candidates.append(current)

    # A snap selected on the bounded train/validation split may regress on the
    # full dataset.  Give the raw seed baseline a route back to recommendation
    # instead of turning that rejected snap into a terminal no-recommendation.
    # Do not trust cached seed-unit metadata here: the common unit check in the
    # evaluation loop below must admit or reject the seed just like every other
    # ballot candidate.
    seed_baseline = getattr(result, "seed_baseline", None)
    if seed_baseline is not None and id(seed_baseline) not in seen_ids:
        seen_ids.add(id(seed_baseline))
        candidates.append(seed_baseline)

    generated_records: list[CandidateRecord] = []

    def add_generated_spec(spec: CandidateSpec) -> Optional[CandidateRecord]:
        try:
            snap_expr = _canonicalize_guarded_candidate_expr(spec.expr, config)
            key = _canonical_key(snap_expr)
        except Exception:
            key = str(spec.expr)
            snap_expr = spec.expr
        if key in seen_exprs:
            return None
        seen_exprs.add(key)
        unit_ok, _unit_reason = _sympy_expr_units_check(snap_expr, variable_names, units_spec)
        if not unit_ok:
            return None
        # A full-dataset snap can fix only some of the fitted literals in its
        # parent expression.  Recompute the surviving scalar degrees of
        # freedom after the rewrite; otherwise a partially snapped expression
        # is incorrectly scored as if every coefficient were exact.
        spec = replace(
            spec,
            expr=snap_expr,
            n_free_params=max(
                int(spec.n_free_params),
                _inferred_learned_constant_count(
                    snap_expr,
                    coefficient_metadata_norm,
                ),
            ),
        )
        rec = _score_candidate(
            spec,
            X_arr,
            y_arr,
            X_arr,
            y_arr,
            variable_names,
            config,
            seed_pred_val=None,
        )
        if rec is not None:
            generated_records.append(rec)
            candidates.append(rec)
            return rec
        return None

    def add_symbolic_snap_variants(
        base_rec: CandidateRecord,
        *,
        label_prefix: str,
        per_number: int = 6,
    ) -> list[CandidateRecord]:
        try:
            base_expr = parse_sympy_expr(str(base_rec.expr), variable_names)
            variants = numeric_constant_snap_candidates(
                base_expr,
                snap_targets=_full_dataset_snap_targets(config),
                snap_rel_tol=float(config.snap_rel_tol),
                per_number=per_number,
            )
        except Exception:
            variants = []
        added: list[CandidateRecord] = []
        for snap_label, snap_expr in variants:
            rec = add_generated_spec(
                CandidateSpec(
                    snap_expr,
                    label=f"{label_prefix}{snap_label}",
                    n_snapped_consts=max(1, int(getattr(base_rec, "n_snapped_consts", 0) or 0) + 1),
                    rewrite_trace=[
                        "full-dataset final symbolic-constant snap",
                        f"cascade from {base_rec.label}",
                    ],
                    source_hints=["full_dataset_snap"],
                )
            )
            if rec is not None:
                added.append(rec)
        return added

    current_expr = None
    try:
        current_expr = parse_sympy_expr(str(current.expr), variable_names)
        snap_variants = numeric_constant_snap_candidates(
            current_expr,
            snap_targets=_full_dataset_snap_targets(config),
            snap_rel_tol=float(config.snap_rel_tol),
            per_number=6,
        )
    except Exception:
        snap_variants = []
    for snap_label, snap_expr in snap_variants:
        add_generated_spec(
            CandidateSpec(
                snap_expr,
                label=f"full_dataset_{snap_label}",
                n_snapped_consts=1,
                rewrite_trace=["full-dataset final symbolic-constant snap"],
                source_hints=["full_dataset_snap"],
            )
        )
    coeff_specs = []
    if current_expr is not None:
        try:
            coeff_specs = _full_dataset_coefficient_lattice_snap_specs(
                current_expr,
                X_arr,
                y_arr,
                variable_names,
                config,
            )
        except Exception:
            coeff_specs = []
    for spec in coeff_specs:
        add_generated_spec(spec)
    radical_specs = []
    if current_expr is not None:
        try:
            radical_specs = _full_dataset_radical_gauge_snap_specs(
                current_expr,
                X_arr,
                y_arr,
                variable_names,
                config,
            )
        except Exception:
            radical_specs = []
    for spec in radical_specs:
        add_generated_spec(spec)
    exp_quad_specs = []
    if current_expr is not None:
        try:
            exp_quad_specs = _full_dataset_exp_quadratic_snap_specs(
                current_expr,
                X_arr,
                y_arr,
                variable_names,
                config,
            )
        except Exception:
            exp_quad_specs = []
    for spec in exp_quad_specs:
        add_generated_spec(spec)

    # Some clean noisy expressions require two independent tiny symbolic snaps:
    # e.g. first normalize one coefficient onto the right symbolic gauge, then
    # snap a remaining 1.00006-style coefficient to 1.  Keep this bounded and
    # full-data-scored; it only enlarges the final snap ballot, not the model
    # family being searched.
    cascade_queue = list(generated_records)
    cascade_seen = 0
    cascade_limit = 96
    while cascade_queue and cascade_seen < cascade_limit:
        base = cascade_queue.pop(0)
        cascade_seen += 1
        added = add_symbolic_snap_variants(
            base,
            label_prefix=f"{base.label}|full_dataset_",
            per_number=3,
        )
        for rec in added:
            if len(generated_records) >= cascade_limit:
                break
            cascade_queue.append(rec)
        if len(generated_records) >= cascade_limit:
            break

    evaluated: list[CandidateRecord] = []
    unit_reject_count = 0
    for rec in candidates:
        try:
            expr = parse_sympy_expr(str(rec.expr), variable_names)
        except Exception:
            continue
        unit_ok, _unit_reason = _sympy_expr_units_check(expr, variable_names, units_spec)
        if not unit_ok:
            unit_reject_count += 1
            continue
        try:
            pred = _eval_expr_array(
                expr, X_arr, variable_names, config.symbol_values
            )
        except Exception:
            continue
        mse, se, frac = _mse_and_se(pred, y_arr, config.min_valid_fraction)
        rec.full_dataset_mse = float(mse)
        rec.full_dataset_mse_se = float(se)
        rec.full_dataset_frac_valid = float(frac)
        if math.isfinite(mse):
            evaluated.append(rec)

    summary.update(
        {
            "n_full": n_full,
            "n_considered": len(candidates),
            "n_evaluated": len(evaluated),
            "n_generated": len(generated_records),
            "unit_reject_count": unit_reject_count,
        }
    )
    if not evaluated:
        summary["reason"] = "no full-dataset-evaluable snap candidates"
        return result, summary

    noise_floor = float(getattr(config, "noise_floor_raw", 0.0) or 0.0)
    numerical_floor = float(getattr(PolishConfig(), "loss_equiv_abs_floor", 1.0e-24))
    if math.isfinite(noise_floor) and noise_floor > 0.0:
        abs_floor = max(numerical_floor, noise_floor * math.sqrt(2.0 / float(n_full)))
    else:
        abs_floor = max(numerical_floor, float(getattr(config, "loss_equiv_abs_floor", 0.0) or 0.0))

    best = min(
        evaluated,
        key=lambda r: (
            float(r.full_dataset_mse),
            float(r.complexity),
            int(r.n_free_params),
            len(str(r.expr)),
        ),
    )

    def within_best(rec: CandidateRecord) -> bool:
        tol = max(
            float(getattr(config, "epsilon_pareto_k", 1.0))
            * max(
                float(getattr(best, "full_dataset_mse_se", 0.0) or 0.0),
                float(getattr(rec, "full_dataset_mse_se", 0.0) or 0.0),
            ),
            abs_floor,
        )
        return float(rec.full_dataset_mse) <= float(best.full_dataset_mse) + tol

    eligible = [rec for rec in evaluated if within_best(rec)]
    seed_safe_eligible = [
        rec
        for rec in eligible
        if _loss_preserves_incumbent(
            candidate_mse=rec.full_dataset_mse,
            candidate_mse_se=rec.full_dataset_mse_se,
            incumbent_mse=seed_full_mse,
            incumbent_mse_se=seed_full_mse_se,
            k=config.epsilon_pareto_k,
            abs_floor=abs_floor,
        )[0]
    ]
    selected = min(
        seed_safe_eligible or eligible,
        key=lambda r: (
            float(r.complexity),
            int(r.n_free_params),
            len(str(r.expr)),
            float(r.full_dataset_mse),
        ),
    )

    previous = result.recommended
    summary.update(
        {
            "status": "selected" if selected is not previous else "unchanged",
            "loss_equiv_abs_floor": float(abs_floor),
            "best_expr": str(best.expr),
            "best_label": str(best.label),
            "best_full_mse": float(best.full_dataset_mse),
            "best_full_mse_se": float(best.full_dataset_mse_se or 0.0),
            "selected_expr": str(selected.expr),
            "selected_label": str(selected.label),
            "selected_full_mse": float(selected.full_dataset_mse),
            "selected_full_mse_se": float(selected.full_dataset_mse_se or 0.0),
            "n_seed_safe": len(seed_safe_eligible),
            "previous_expr": str(previous.expr) if previous is not None else None,
            "previous_label": str(previous.label) if previous is not None else None,
            "previous_full_mse": (
                float(previous.full_dataset_mse)
                if previous is not None and previous.full_dataset_mse is not None
                else None
            ),
        }
    )

    preserves_seed, incumbent_tol = _loss_preserves_incumbent(
        candidate_mse=selected.full_dataset_mse,
        candidate_mse_se=selected.full_dataset_mse_se,
        incumbent_mse=seed_full_mse,
        incumbent_mse_se=seed_full_mse_se,
        k=config.epsilon_pareto_k,
        abs_floor=abs_floor,
    )
    summary["seed_loss_tolerance"] = float(incumbent_tol)
    if not preserves_seed:
        reason = (
            "no admissible full-dataset candidate preserves the raw seed loss"
            if math.isfinite(seed_full_mse)
            else "raw seed full-dataset loss is unavailable; refusing promotion"
        )
        summary.update(
            {
                "status": "no_safe_unit_valid_replacement",
                "reason": reason,
                "rejected_expr": str(selected.expr),
                "rejected_full_mse": float(selected.full_dataset_mse),
            }
        )
        for rec in result.all_candidates:
            rec.full_dataset_snap_selected = False
            rec.is_recommended = False
        if current is not None:
            current.is_recommended = False
        result.recommended = None
        result.selection_status = "no_safe_unit_valid_replacement"
        result.selection_reason = reason
        result.rewrite_trace.append(reason)
        result.warnings.append(reason)
        return result, summary

    for rec in result.all_candidates:
        rec.full_dataset_snap_selected = False
    selected.full_dataset_snap_selected = True
    if selected in generated_records and selected not in result.all_candidates:
        result.all_candidates.append(selected)
    if selected is not previous:
        if previous is not None:
            previous.is_recommended = False
        selected.is_recommended = True
        result.recommended = selected
        msg = (
            "full-dataset adjudication selected "
            f"{selected.expr} over {previous.expr if previous is not None else '<none>'}"
        )
        result.rewrite_trace.append(msg)
        result.warnings.append(msg)
    return result, summary


def polish_expression(
    expr: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    variable_names: Optional[Sequence[str]] = None,
    assumptions: Optional[Mapping[str, str]] = None,
    artifact_hints: Optional[ArtifactHints] = None,
    noise_mse: Optional[float] = None,
    units_spec: Optional[Any] = None,
    config: Optional[PolishConfig] = None,
) -> PolishResult:
    """Generate and score nearby simplified expressions."""
    del noise_mse  # Reserved for a later noise-floor-aware acceptance policy.
    if sp is None:
        raise RuntimeError("SymPy is required for equation_polisher")
    config = config or PolishConfig()
    variable_names = list(variable_names or infer_variable_names(expr, X_train))
    coefficient_metadata_norm = None
    if artifact_hints is not None and artifact_hints.coefficient_metadata is not None:
        coefficient_metadata_norm = normalize_coefficient_metadata(
            artifact_hints.coefficient_metadata,
            variable_names=variable_names,
            require_values=True,
            units_spec=units_spec,
        )
        artifact_values = coefficient_symbol_values(
            coefficient_metadata_norm,
            variable_names=variable_names,
            units_spec=units_spec,
        )
        merged_values = dict(artifact_values)
        for name, value in config.symbol_values.items():
            if name in merged_values and not math.isclose(
                float(value),
                float(merged_values[name]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    f"coefficient symbol {name!r} has conflicting artifact and "
                    "configuration values"
                )
            merged_values[name] = value
        config = replace(config, symbol_values=merged_values)
    all_X = np.vstack([X_train, X_val])
    inferred = _build_symbol_assumptions(
        all_X,
        variable_names,
        assumptions,
        detect_positive=config.positive_var_detection,
    )
    if artifact_hints is not None:
        inferred.update({k: v for k, v in artifact_hints.variable_assumptions.items() if k not in (assumptions or {})})

    warnings_out: list[str] = []
    trace: list[str] = ["seed: " + str(expr)]
    seed = _parse_with_assumptions(expr, variable_names, inferred)
    seed = _prune_tiny_additive_constants(seed, tol=0.0)
    seed_raw = seed
    seed_normalized = seed_raw
    try:
        seed_normalized = _canonicalize_inverse_ratio_powers(seed_normalized)
    except Exception:
        pass
    try:
        seed_normalized = canonicalize_trig_phases(
            seed_normalized,
            snap_rel_tol=config.snap_rel_tol,
        )
    except Exception:
        pass
    try:
        seed_pred_val = _eval_expr_array(
            seed_raw, X_val, variable_names, config.symbol_values
        )
    except Exception:
        seed_pred_val = None
        warnings_out.append("failed to evaluate seed expression for seed-distance scoring")

    seed_spec = CandidateSpec(
        seed_raw,
        label="seed",
        n_free_params=_inferred_learned_constant_count(
            seed_raw,
            coefficient_metadata_norm,
        ),
        rewrite_trace=["raw seed expression"],
    )
    seed_baseline = _score_candidate(
        seed_spec,
        X_train,
        y_train,
        X_val,
        y_val,
        variable_names,
        config,
        seed_pred_val,
    )
    if seed_baseline is None:
        warnings_out.append("failed to score raw seed validation-loss baseline")
    seed_units_ok, seed_units_reason = _sympy_expr_units_check(
        seed_raw,
        variable_names,
        units_spec,
    )

    specs: list[CandidateSpec] = [seed_spec]
    if seed_normalized != seed_raw:
        specs.append(
            CandidateSpec(
                seed_normalized,
                label="canonicalize_seed",
                n_free_params=_inferred_learned_constant_count(
                    seed_normalized,
                    coefficient_metadata_norm,
                ),
                rewrite_trace=["canonicalize inverse powers and trig phases"],
            )
        )
    templates: list[InverseSqrtPolyTemplate] = []

    rat_exp = _rationalize_float_exponents(seed_raw)
    if rat_exp != seed_raw:
        specs.append(
            CandidateSpec(
                rat_exp,
                label="rationalize_float_exponents",
                rewrite_trace=["rationalize float exponents"],
            )
        )
        trace.append("rationalized float exponents")

    if config.use_artifact_hints and artifact_hints is not None:
        artifact_specs = _artifact_expression_candidates(
            artifact_hints,
            variable_names,
            inferred,
        )
        specs.extend(artifact_specs)
        for spec in artifact_specs:
            trace.append(f"artifact expression candidate: {spec.label}")

    try:
        if sympy_timeout is not None:
            with sympy_timeout(
                "denominator_coefficient_ratio_snap",
                max_seconds=3,
            ):
                denominator_ratio_specs = (
                    _denominator_coefficient_ratio_snap_specs(
                        rat_exp,
                        variable_names,
                        config,
                    )
                )
        else:
            denominator_ratio_specs = _denominator_coefficient_ratio_snap_specs(
                rat_exp,
                variable_names,
                config,
            )
    except Exception:
        denominator_ratio_specs = []
    specs.extend(denominator_ratio_specs)
    if denominator_ratio_specs:
        trace.append(
            "generated "
            f"{len(denominator_ratio_specs)} denominator coefficient-ratio "
            "snap candidate(s)"
        )

    try:
        if sympy_timeout is not None:
            with sympy_timeout(
                "radical_coefficient_ratio_snap",
                max_seconds=5,
            ):
                radical_ratio_specs = (
                    _radical_coefficient_ratio_snap_specs(
                        rat_exp,
                        variable_names,
                        config,
                    )
                )
        else:
            radical_ratio_specs = _radical_coefficient_ratio_snap_specs(
                rat_exp,
                variable_names,
                config,
            )
    except Exception:
        radical_ratio_specs = []
    specs.extend(radical_ratio_specs)
    if radical_ratio_specs:
        trace.append(
            "generated "
            f"{len(radical_ratio_specs)} radical coefficient-ratio "
            "snap candidate(s)"
        )

    drop_refit_specs: list[CandidateSpec] = []
    if bool(getattr(config, "enable_drop_addend_refit", True)):
        try:
            if sympy_timeout is not None:
                with sympy_timeout(
                    "drop_addend_refit",
                    max_seconds=max(
                        5,
                        int(math.ceil(config.drop_refit_max_seconds)) + 4,
                    ),
                ):
                    drop_refit_specs = _drop_addend_refit_specs(
                        rat_exp,
                        X_train,
                        y_train,
                        variable_names,
                        units_spec,
                        config,
                    )
            else:
                drop_refit_specs = _drop_addend_refit_specs(
                    rat_exp,
                    X_train,
                    y_train,
                    variable_names,
                    units_spec,
                    config,
                )
        except Exception:
            drop_refit_specs = []
    specs.extend(drop_refit_specs)
    if drop_refit_specs:
        trace.append(
            f"generated {len(drop_refit_specs)} drop-addend refit candidate(s)"
        )

    specs.extend(_guarded_sympy_candidates(rat_exp, variable_names, config))
    seed_sparse_rat_specs = _sparse_rational_seed_support_candidates(
        rat_exp,
        X_train,
        y_train,
        variable_names,
        config,
    )
    specs.extend(seed_sparse_rat_specs)
    if seed_sparse_rat_specs:
        trace.append(
            f"generated {len(seed_sparse_rat_specs)} noisy seed-support sparse rational candidate(s)"
        )
    sparse_rat_specs = []
    if bool(getattr(config, "enable_noisy_sparse_rational_refit", False)):
        sparse_rat_specs = _sparse_rational_refit_candidates(
            X_train,
            y_train,
            variable_names,
            config,
        )
    specs.extend(sparse_rat_specs)
    if sparse_rat_specs:
        trace.append(f"generated {len(sparse_rat_specs)} noisy sparse rational candidate(s)")

    if config.use_artifact_hints and artifact_hints is not None:
        art_templates = _artifact_templates(artifact_hints, variable_names)
        templates.extend(art_templates)
        for t in art_templates:
            trace.append(
                f"artifact template: prefactor={_sstr(t.prefactor)}, z={_sstr(t.z_expr)}, coeffs={t.coeffs}"
            )

    proj_specs, proj_templates = _homogeneous_radical_candidates(rat_exp, variable_names, inferred)
    specs.extend(proj_specs)
    templates.extend(proj_templates)
    if proj_specs:
        trace.append(f"generated {len(proj_specs)} homogeneous radical candidate(s)")

    for template in templates:
        specs.extend(
            _template_candidates(
                template,
                X_train,
                y_train,
                variable_names,
                config,
                artifact_hints,
            )
        )

    exp_templates = _fit_exp_poly_templates(
        X_train,
        y_train,
        variable_names,
        inferred,
        artifact_hints,
    )
    for template in exp_templates:
        trace.append(
            f"exp-poly template: variable={_sstr(template.variable)}, coeffs={template.coeffs}"
        )
        specs.extend(
            _exp_poly_candidates(
                template,
                X_train,
                y_train,
                variable_names,
                config,
            )
        )

    seen: set[str] = set()
    scored: list[CandidateRecord] = []
    units_reject_count = 0
    units_reject_examples: list[str] = []
    for spec in specs:
        if len(scored) >= config.max_candidates:
            break
        spec.n_free_params = max(
            int(spec.n_free_params),
            _inferred_learned_constant_count(
                spec.expr,
                coefficient_metadata_norm,
            ),
        )
        key = _canonical_key(spec.expr)
        text_key = _sstr(spec.expr)
        if key in seen or text_key in seen:
            continue
        seen.add(key)
        seen.add(text_key)
        unit_ok, unit_reason = _sympy_expr_units_check(spec.expr, variable_names, units_spec)
        if not unit_ok:
            units_reject_count += 1
            if len(units_reject_examples) < 3:
                units_reject_examples.append(f"{spec.label}: {unit_reason}")
            continue
        rec = (
            seed_baseline
            if spec is seed_spec
            else _score_candidate(
                spec,
                X_train,
                y_train,
                X_val,
                y_val,
                variable_names,
                config,
                seed_pred_val,
            )
        )
        if rec is not None:
            scored.append(rec)
    if units_reject_count:
        msg = f"units filter rejected {units_reject_count} candidate(s)"
        if units_reject_examples:
            msg += "; examples: " + "; ".join(units_reject_examples)
        warnings_out.append(msg)

    strict_idx = pareto_front_indices_2d([(r.val_mse, r.complexity) for r in scored])
    for idx in strict_idx:
        scored[idx].is_strict_pareto = True
    eps_idx = _epsilon_pareto_indices(
        scored,
        config.epsilon_pareto_k,
        config.loss_equiv_abs_floor,
    )
    for idx in eps_idx:
        scored[idx].is_epsilon_pareto = True
    provisional_rec = _recommend(
        scored,
        config.epsilon_pareto_k,
        config.loss_equiv_abs_floor,
    )
    rec = provisional_rec
    selection_status = "selected"
    selection_reason = None
    if seed_baseline is None or not math.isfinite(float(seed_baseline.val_mse)):
        rec = None
        selection_status = "no_safe_unit_valid_replacement"
        selection_reason = (
            "raw seed validation-loss baseline is unavailable; refusing promotion"
        )
    else:
        seed_safe_scored = [
            candidate
            for candidate in scored
            if _loss_preserves_incumbent(
                candidate_mse=candidate.val_mse,
                candidate_mse_se=candidate.val_mse_se,
                incumbent_mse=seed_baseline.val_mse,
                incumbent_mse_se=seed_baseline.val_mse_se,
                k=config.epsilon_pareto_k,
                abs_floor=config.loss_equiv_abs_floor,
            )[0]
        ]
        rec = _recommend(
            seed_safe_scored,
            config.epsilon_pareto_k,
            config.loss_equiv_abs_floor,
        )
        if rec is None:
            selection_status = "no_safe_unit_valid_replacement"
            if provisional_rec is None:
                selection_reason = "no admissible candidate has finite validation loss"
            else:
                _preserves, incumbent_tol = _loss_preserves_incumbent(
                    candidate_mse=provisional_rec.val_mse,
                    candidate_mse_se=provisional_rec.val_mse_se,
                    incumbent_mse=seed_baseline.val_mse,
                    incumbent_mse_se=seed_baseline.val_mse_se,
                    k=config.epsilon_pareto_k,
                    abs_floor=config.loss_equiv_abs_floor,
                )
                selection_reason = (
                    "admissible recommendation worsens raw seed validation loss "
                    f"({provisional_rec.val_mse:.12g} > "
                    f"{seed_baseline.val_mse:.12g} + {incumbent_tol:.12g})"
                )
    if selection_reason:
        warnings_out.append(selection_reason)
    if rec is not None:
        rec.is_recommended = True

    scored.sort(key=lambda r: (r.val_mse, r.complexity, r.expr))
    strict = [r for r in scored if r.is_strict_pareto]
    strict.sort(key=lambda r: (r.val_mse, r.complexity, r.expr))
    epsilon = [r for r in scored if r.is_epsilon_pareto]
    epsilon.sort(key=lambda r: (r.val_mse, r.complexity, r.expr))

    return PolishResult(
        seed_expr=_sstr(seed),
        all_candidates=scored,
        strict_pareto=strict,
        epsilon_pareto=epsilon,
        recommended=rec,
        rewrite_trace=trace,
        warnings=warnings_out + (artifact_hints.warnings if artifact_hints else []),
        artifact_hints=artifact_hints,
        seed_baseline=seed_baseline,
        seed_units_ok=bool(seed_units_ok),
        seed_units_reason=str(seed_units_reason),
        selection_status=selection_status,
        selection_reason=selection_reason,
    )


def write_outputs(result: PolishResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "frontier.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
    with (out / "artifact_hints.json").open("w", encoding="utf-8") as f:
        artifact_payload = (
            result.artifact_hints.to_dict()
            if result.artifact_hints is not None
            else ArtifactHints().to_dict()
        )
        json.dump(artifact_payload, f, indent=2, sort_keys=True)
    with (out / "rewrite_trace.txt").open("w", encoding="utf-8") as f:
        for line in result.rewrite_trace:
            f.write(str(line) + "\n")
        f.write("\nRecommended:\n")
        f.write((result.recommended.expr if result.recommended else "<none>") + "\n")
        f.write("\nEpsilon Pareto:\n")
        for rec in result.epsilon_pareto:
            f.write(f"{rec.val_mse:.12g}\t{rec.complexity:.6g}\t{rec.expr}\n")
    fields = [
        "label",
        "expr",
        "train_mse",
        "val_mse",
        "val_mse_se",
        "complexity",
        "structural_complexity",
        "coefficient_complexity",
        "n_free_params",
        "n_snapped_consts",
        "frac_valid",
        "full_dataset_mse",
        "full_dataset_mse_se",
        "full_dataset_frac_valid",
        "full_dataset_snap_selected",
        "seed_nrmse",
        "is_strict_pareto",
        "is_epsilon_pareto",
        "is_recommended",
        "source_hints",
        "assumptions",
    ]
    with (out / "frontier.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in result.all_candidates:
            row = rec.to_dict()
            row["source_hints"] = ";".join(rec.source_hints)
            row["assumptions"] = ";".join(rec.assumptions)
            writer.writerow({k: row.get(k) for k in fields})


def _resolve_seed_expr(explicit: Optional[str], hints: ArtifactHints) -> Optional[str]:
    explicit_clean = _clean_expr_field(explicit)
    if explicit_clean is not None:
        return explicit_clean
    y_clean = _clean_expr_field(hints.y_expr)
    if y_clean is not None:
        return y_clean
    y_transform = str(hints.y_transform or "identity").strip().lower()
    if y_transform in {"", "identity", "none", "null"}:
        return _clean_expr_field(hints.phi_expr) or _clean_expr_field(hints.seed_expr)
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc equation polisher for NestyNet_SR expressions")
    parser.add_argument("--expr", type=str, default=None, help="Expression string to polish")
    parser.add_argument("--filepath", type=str, default=None, help="CSV data path")
    parser.add_argument("--target-col", type=str, default="y", help="Target column name")
    parser.add_argument("--report-json", type=str, default=None, help="NestyNet report JSON")
    parser.add_argument("--decisions-json", type=str, default=None, help="Stage-B decisions JSON")
    parser.add_argument("--allstages-log", type=str, default=None, help="All-stages log file")
    parser.add_argument("--path-file", type=str, default=None, help="Simplification path file")
    parser.add_argument("--final-human", type=str, default=None, help="Final human-readable file")
    parser.add_argument("--positive", action="append", default=[], help="Variable known positive, e.g. x2")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction for CSV split")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for split/subsample")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row subsample cap")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    hints = load_artifact_hints(
        report_json=args.report_json,
        decisions_json=args.decisions_json,
        allstages_log=args.allstages_log,
        path_file=args.path_file,
        final_human=args.final_human,
    )
    filepath = args.filepath or hints.dataset_path
    if filepath is None:
        parser.error("provide --filepath or a report JSON containing metadata.dataset")
    expr = _resolve_seed_expr(args.expr, hints)
    if expr is None:
        parser.error("provide --expr or artifacts containing a final expression")
    Xtr, ytr, Xva, yva, var_names = load_csv_data(
        filepath,
        target_col=args.target_col,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        max_rows=args.max_rows,
    )
    try:
        symbol_values = coefficient_symbol_values(
            hints.coefficient_metadata,
            variable_names=var_names,
        )
    except Exception as exc:
        parser.error(f"invalid coefficient metadata in report: {exc}")
    config = PolishConfig(
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        symbol_values=symbol_values,
    )
    explicit_assumptions = {str(v): ">0" for v in args.positive}
    result = polish_expression(
        expr,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=var_names,
        assumptions=explicit_assumptions,
        artifact_hints=hints,
        config=config,
    )
    write_outputs(result, args.out_dir)
    if result.recommended is not None:
        print(result.recommended.expr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
