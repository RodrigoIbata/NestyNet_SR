# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Trajectory-aware DE candidate-bank and support scoring.

This module is the first slice of the FSS-for-DE assembly/evaluation plan.  It
normalizes current DE candidate sources into a small term bank, then scores
explicit and implicit/rational supports using per-trajectory evidence.  It is
diagnostic-first: callers can inspect the scored supports without changing the
selected DE engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import Add, ConstNode, Mul, Pow
from nestynet_sr.sr_de.de_search import DESearchResult, DESearchResultMulti
from nestynet_sr.sr_de.factorized_de import (
    DEFeatureGroup,
    FactorizedDEResult,
    FactorizedSearchDEResult,
    _anchor_tensor,
    _eval_ast_on_features,
    evaluate_factorized_search_candidate,
    factorized_search_report_shortlist,
)


@dataclass
class DETerm:
    """One scalar DE candidate column evaluated on every trajectory split."""

    term_id: str
    source: str
    role_hint: str
    ast: Any = None
    payload: Mapping[str, Any] | None = None
    complexity: float = 1.0
    fit_values: tuple[torch.Tensor, ...] = ()
    probe_values: tuple[torch.Tensor, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "source": self.source,
            "role_hint": self.role_hint,
            "display": str(self.metadata.get("display", self.term_id)),
            "complexity": float(self.complexity),
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class DETermBank:
    """A capped, deduplicated term reservoir for one DE anchor order."""

    order: int
    x_axis: int
    terms: list[DETerm]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self, *, max_terms: int = 24) -> dict[str, Any]:
        return {
            "order": int(self.order),
            "x_axis": int(self.x_axis),
            "term_count": int(len(self.terms)),
            "diagnostics": _jsonable(self.diagnostics),
            "terms": [term.summary() for term in self.terms[: max(0, int(max_terms))]],
        }


@dataclass
class RoleShadowOpportunity:
    """Small diagnostic record for a generated DE role-shadow term."""

    opportunity_id: str
    source: str
    input_term_ids: tuple[str, ...]
    emitted_term_ids: tuple[str, ...] = ()
    score_hint: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_id": str(self.opportunity_id),
            "source": str(self.source),
            "input_term_ids": list(self.input_term_ids),
            "emitted_term_ids": list(self.emitted_term_ids),
            "score_hint": None if self.score_hint is None else float(self.score_hint),
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class DESupportCandidate:
    """One fitted explicit support, scored per trajectory."""

    support_id: str
    form: str
    order: int
    x_axis: int
    term_ids: tuple[str, ...]
    coefficients: tuple[float, ...]
    ridge: float
    score: float
    fit_rms_mean: float
    fit_rms_max: float
    probe_rms_mean: float
    probe_rms_max: float
    probe_nrmse_mean: float
    probe_nrmse_max: float
    coefficient_stability: float
    complexity: float
    canonical_equation: str = ""
    validation_candidate: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_id": self.support_id,
            "form": str(self.form),
            "order": int(self.order),
            "x_axis": int(self.x_axis),
            "term_ids": list(self.term_ids),
            "coefficients": [float(c) for c in self.coefficients],
            "ridge": float(self.ridge),
            "score": float(self.score),
            "fit_rms_mean": float(self.fit_rms_mean),
            "fit_rms_max": float(self.fit_rms_max),
            "probe_rms_mean": float(self.probe_rms_mean),
            "probe_rms_max": float(self.probe_rms_max),
            "probe_nrmse_mean": float(self.probe_nrmse_mean),
            "probe_nrmse_max": float(self.probe_nrmse_max),
            "coefficient_stability": float(self.coefficient_stability),
            "complexity": float(self.complexity),
            "canonical_equation": str(self.canonical_equation),
            "validation_candidate": _jsonable(self.validation_candidate),
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class DEImplicitRationalCandidate:
    """One implicit rational/mass-form candidate.

    The represented residual is:

        numerator + anchor * (pivot + denominator) = 0

    When the denominator is numerically safe and all terms are AST-backed, this
    can be rendered explicitly as:

        anchor + numerator / (pivot + denominator) = 0
    """

    support_id: str
    order: int
    x_axis: int
    numerator_term_ids: tuple[str, ...]
    denominator_term_ids: tuple[str, ...]
    numerator_coefficients: tuple[float, ...]
    denominator_coefficients: tuple[float, ...]
    pivot: str
    ridge: float
    score: float
    implicit_fit_rms_mean: float
    implicit_fit_rms_max: float
    implicit_probe_rms_mean: float
    implicit_probe_rms_max: float
    explicit_probe_rms_mean: float
    explicit_probe_rms_max: float
    denominator_safety: dict[str, Any]
    coefficient_stability: float
    complexity: float
    canonical_equation: str = ""
    validation_candidate: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_id": self.support_id,
            "form": "implicit_rational",
            "order": int(self.order),
            "x_axis": int(self.x_axis),
            "numerator_term_ids": list(self.numerator_term_ids),
            "denominator_term_ids": list(self.denominator_term_ids),
            "numerator_coefficients": [float(c) for c in self.numerator_coefficients],
            "denominator_coefficients": [float(c) for c in self.denominator_coefficients],
            "pivot": str(self.pivot),
            "ridge": float(self.ridge),
            "score": float(self.score),
            "implicit_fit_rms_mean": float(self.implicit_fit_rms_mean),
            "implicit_fit_rms_max": float(self.implicit_fit_rms_max),
            "implicit_probe_rms_mean": float(self.implicit_probe_rms_mean),
            "implicit_probe_rms_max": float(self.implicit_probe_rms_max),
            "explicit_probe_rms_mean": float(self.explicit_probe_rms_mean),
            "explicit_probe_rms_max": float(self.explicit_probe_rms_max),
            "denominator_safety": _jsonable(self.denominator_safety),
            "coefficient_stability": float(self.coefficient_stability),
            "complexity": float(self.complexity),
            "canonical_equation": str(self.canonical_equation),
            "validation_candidate": _jsonable(self.validation_candidate),
            "metadata": _jsonable(self.metadata),
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _flatten_tensor(value: Any, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.detach().to(dtype=dtype)
    else:
        out = torch.as_tensor(value, dtype=dtype)
    return out.reshape(-1)


def _source_name(source: str, default: str) -> str:
    out = str(source or "").strip()
    return out if out else default


def _ast_complexity(ast: Any) -> float:
    if ast is None:
        return 1.0
    text = repr(ast)
    return float(max(1, text.count("(") + text.count("[") + 1))


def _candidate_complexity(candidate: Mapping[str, Any]) -> float:
    for key in ("complexity", "size", "mapping_complexity"):
        val = candidate.get(key, None)
        try:
            out = float(val)
        except Exception:
            continue
        if math.isfinite(out) and out > 0.0:
            return out
    return _ast_complexity(candidate.get("expr_ast", candidate.get("expr", None)))


def _term_display_from_ast(ast: Any) -> str:
    return "1" if ast is None else repr(ast)


def _anchor_name(order: int, *, x_axis: int) -> str:
    if int(order) == 1:
        return f"u_x{int(x_axis)}"
    if int(order) == 2:
        return f"u_x{int(x_axis)}x{int(x_axis)}"
    return f"d^{int(order)}u/dx{int(x_axis)}^{int(order)}"


def _scaled_ast(coeff: float, ast: Any):
    c = float(coeff)
    if ast is None:
        return ConstNode(c)
    if abs(c - 1.0) < 1.0e-14:
        return ast
    return Mul(ConstNode(c), ast)


def _sum_scaled_asts(asts: Sequence[Any], coeffs: Sequence[float]):
    out = None
    for ast, coeff in zip(list(asts), list(coeffs)):
        c = float(coeff)
        if abs(c) < 1.0e-14:
            continue
        term = _scaled_ast(c, ast)
        out = term if out is None else Add(out, term)
    return out


def _terms_materializable(terms: Sequence[DETerm]) -> bool:
    for term in list(terms):
        if term.ast is None and not bool(term.metadata.get("allow_constant", False)):
            return False
    return True


def _materialize_explicit_validation_candidate(
    *,
    order: int,
    x_axis: int,
    asts: Sequence[Any],
    coeffs: Sequence[float],
    canonical_equation: str,
    ast_serializer: Any,
) -> dict[str, Any] | None:
    if ast_serializer is None:
        return None
    if any(getattr(ast, "__class__", None) is None for ast in list(asts)):
        return None
    try:
        return {
            "engine": "de_candidate_eval",
            "kind": "assembled_explicit_support",
            "order": int(order),
            "x_axis": int(x_axis),
            "coefficients": [float(c) for c in coeffs],
            "term_asts_json": [ast_serializer(ast) for ast in list(asts)],
            "canonical_equation": str(canonical_equation),
        }
    except Exception:
        return None


def _materialize_rational_validation_candidate(
    *,
    order: int,
    x_axis: int,
    numerator_asts: Sequence[Any],
    numerator_coeffs: Sequence[float],
    denominator_asts: Sequence[Any],
    denominator_coeffs: Sequence[float],
    canonical_equation: str,
    ast_serializer: Any,
    pivot_ast: Any = None,
) -> dict[str, Any] | None:
    if ast_serializer is None:
        return None
    try:
        numerator = _sum_scaled_asts(numerator_asts, numerator_coeffs)
        denominator = ConstNode(1.0) if pivot_ast is None else pivot_ast
        denominator_rest = _sum_scaled_asts(denominator_asts, denominator_coeffs)
        if denominator_rest is not None:
            denominator = Add(denominator, denominator_rest)
        if numerator is None:
            return None
        explicit_term = Mul(numerator, Pow(denominator, -1.0))
        return {
            "engine": "de_candidate_eval",
            "kind": "assembled_implicit_rational",
            "order": int(order),
            "x_axis": int(x_axis),
            "coefficients": [1.0],
            "term_asts_json": [ast_serializer(explicit_term)],
            "canonical_equation": str(canonical_equation),
        }
    except Exception:
        return None


def _eval_ast_term(
    ast: Any,
    groups: Sequence[DEFeatureGroup],
    *,
    x_axis: int,
    split: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    vals = []
    for group in groups:
        col = _eval_ast_on_features(
            ast,
            features=group.features,
            split=str(split),
            x_axis=int(x_axis),
        )
        vals.append(_flatten_tensor(col, dtype=dtype))
    return tuple(vals)


def _split_feature(features: Any, split: str, name: str, *, dtype: torch.dtype) -> torch.Tensor:
    value = getattr(features, f"{name}_{split}")
    return _flatten_tensor(value, dtype=dtype)


def _candidate_feature_names(candidate: Mapping[str, Any], *, order: int) -> list[str]:
    raw = list(candidate.get("feature_names", []) or [])
    if raw:
        return [str(name) for name in raw]
    out = []
    if bool(candidate.get("include_x", True)):
        out.append(f"x{int(candidate.get('x_axis', 0))}")
    if bool(candidate.get("include_u", True)):
        out.append("u")
    if int(order) == 2 and bool(candidate.get("include_du", True)):
        out.append("du")
    for const in list(candidate.get("constants_ordered", []) or []):
        if isinstance(const, Mapping):
            out.append(str(const.get("name", f"c{len(out)}")))
    return out


def _constant_lookup(candidate: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, const in enumerate(list(candidate.get("constants_ordered", []) or [])):
        if isinstance(const, Mapping) and "value" in const:
            out[str(const.get("name", f"c{i}"))] = float(const["value"])
    return out


def _feature_column_by_name(
    group: DEFeatureGroup,
    *,
    split: str,
    name: str,
    x_axis: int,
    constants: Mapping[str, float],
    dtype: torch.dtype,
) -> torch.Tensor:
    key = str(name).strip().lower()
    if key in constants:
        ref = _split_feature(group.features, split, "u", dtype=dtype)
        return torch.full_like(ref, float(constants[key]))
    if key in {"u", "y", "state", "field"}:
        return _split_feature(group.features, split, "u", dtype=dtype)
    if key in {"du", "dudx", "u_dot", "udot", "u_x"}:
        return _split_feature(group.features, split, "du", dtype=dtype)
    if key in {"d2u", "ddu", "u_xx"}:
        return _split_feature(group.features, split, "d2u", dtype=dtype)
    if key in {"x", "var", "input"}:
        x = getattr(group.features, f"x_{split}")
        x_t = torch.as_tensor(x, dtype=dtype)
        return x_t[:, int(x_axis)].reshape(-1)
    if key.startswith("x") and key[1:].isdigit():
        idx = int(key[1:])
        x = getattr(group.features, f"x_{split}")
        x_t = torch.as_tensor(x, dtype=dtype)
        return x_t[:, idx].reshape(-1)
    raise ValueError(f"Unsupported factorized-search DE feature name: {name!r}")


def _candidate_feature_matrix(
    candidate: Mapping[str, Any],
    group: DEFeatureGroup,
    *,
    split: str,
    order: int,
    x_axis: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    names = _candidate_feature_names(candidate, order=int(order))
    constants = {str(k).lower(): float(v) for k, v in _constant_lookup(candidate).items()}
    cols = [
        _feature_column_by_name(
            group,
            split=str(split),
            name=name,
            x_axis=int(x_axis),
            constants=constants,
            dtype=dtype,
        )
        for name in names
    ]
    if not cols:
        n = int(_split_feature(group.features, split, "u", dtype=dtype).numel())
        return torch.empty((n, 0), dtype=dtype)
    return torch.stack(cols, dim=1)


def _eval_factorized_search_term(
    candidate: Mapping[str, Any],
    groups: Sequence[DEFeatureGroup],
    *,
    order: int,
    x_axis: int,
    split: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    vals = []
    payload = dict(candidate)
    payload.setdefault("order", int(order))
    payload.setdefault("x_axis", int(x_axis))
    for group in groups:
        features = _candidate_feature_matrix(
            payload,
            group,
            split=str(split),
            order=int(order),
            x_axis=int(x_axis),
            dtype=dtype,
        )
        arr = evaluate_factorized_search_candidate(payload, features, dtype=dtype)
        vals.append(_flatten_tensor(arr, dtype=dtype))
    return tuple(vals)


def _concat_values(values: Sequence[torch.Tensor]) -> torch.Tensor:
    parts = [_flatten_tensor(v) for v in values if int(_flatten_tensor(v).numel()) > 0]
    if not parts:
        return torch.empty((0,), dtype=torch.float64)
    return torch.cat(parts, dim=0).reshape(-1)


def _finite_fraction(values: Sequence[torch.Tensor]) -> float:
    all_vals = _concat_values(values)
    if int(all_vals.numel()) <= 0:
        return 0.0
    return float(torch.isfinite(all_vals).double().mean().detach().cpu().item())


def _nearly_constant(values: Sequence[torch.Tensor]) -> bool:
    all_vals = _concat_values(values)
    finite = torch.isfinite(all_vals)
    if int(finite.sum()) <= 1:
        return True
    vf = all_vals[finite]
    return bool(float(torch.std(vf).detach().cpu().item()) < 1.0e-14)


def _column_similarity(lhs: Sequence[torch.Tensor], rhs: Sequence[torch.Tensor]) -> float:
    a = _concat_values(lhs)
    b = _concat_values(rhs)
    n = min(int(a.numel()), int(b.numel()))
    if n <= 1:
        return 0.0
    a = a[:n]
    b = b[:n]
    finite = torch.isfinite(a) & torch.isfinite(b)
    if int(finite.sum()) <= 1:
        return 0.0
    a = a[finite].to(dtype=torch.float64)
    b = b[finite].to(dtype=torch.float64)
    ac = a - torch.mean(a)
    bc = b - torch.mean(b)
    an = float(torch.linalg.norm(ac).detach().cpu().item())
    bn = float(torch.linalg.norm(bc).detach().cpu().item())
    if an <= 1.0e-14 or bn <= 1.0e-14:
        au = float(torch.linalg.norm(a).detach().cpu().item())
        bu = float(torch.linalg.norm(b).detach().cpu().item())
        if au <= 1.0e-14 or bu <= 1.0e-14:
            return 1.0 if torch.allclose(a, b, rtol=1.0e-12, atol=1.0e-12) else 0.0
        return abs(float(torch.dot(a, b).detach().cpu().item()) / (au * bu))
    return abs(float(torch.dot(ac, bc).detach().cpu().item()) / (an * bn))


def _raw_column_similarity(lhs: Sequence[torch.Tensor], rhs: Sequence[torch.Tensor]) -> float:
    a = _concat_values(lhs)
    b = _concat_values(rhs)
    n = min(int(a.numel()), int(b.numel()))
    if n <= 1:
        return 0.0
    a = a[:n]
    b = b[:n]
    finite = torch.isfinite(a) & torch.isfinite(b)
    if int(finite.sum()) <= 1:
        return 0.0
    a = a[finite].to(dtype=torch.float64)
    b = b[finite].to(dtype=torch.float64)
    an = float(torch.linalg.norm(a).detach().cpu().item())
    bn = float(torch.linalg.norm(b).detach().cpu().item())
    if an <= 1.0e-14 or bn <= 1.0e-14:
        return 1.0 if torch.allclose(a, b, rtol=1.0e-12, atol=1.0e-12) else 0.0
    return abs(float(torch.dot(a, b).detach().cpu().item()) / (an * bn))


def _append_term(
    terms: list[DETerm],
    diagnostics: dict[str, Any],
    source_counts: dict[str, int],
    term: DETerm,
    *,
    max_terms_total: int,
    max_terms_per_source: int,
    max_duplicate_corr: float,
    min_finite_fraction: float,
) -> bool:
    rejected = diagnostics.setdefault("rejected", [])
    source = str(term.source)
    if len(terms) >= int(max_terms_total):
        rejected.append({"term_id": term.term_id, "source": source, "reason": "max_terms_total"})
        return False
    if int(source_counts.get(source, 0)) >= int(max_terms_per_source):
        rejected.append({"term_id": term.term_id, "source": source, "reason": "max_terms_per_source"})
        return False
    finite_fraction = min(_finite_fraction(term.fit_values), _finite_fraction(term.probe_values))
    if finite_fraction < float(min_finite_fraction):
        rejected.append(
            {
                "term_id": term.term_id,
                "source": source,
                "reason": "nonfinite",
                "finite_fraction": float(finite_fraction),
            }
        )
        return False
    if _nearly_constant(term.fit_values) and not bool(term.metadata.get("allow_constant", False)):
        rejected.append({"term_id": term.term_id, "source": source, "reason": "nearly_constant"})
        return False
    similarity_fn = _raw_column_similarity if bool(term.metadata.get("allow_affine_duplicate", False)) else _column_similarity
    for kept in terms:
        sim = similarity_fn(term.fit_values, kept.fit_values)
        if sim >= float(max_duplicate_corr):
            rejected.append(
                {
                    "term_id": term.term_id,
                    "source": source,
                    "reason": "duplicate_column",
                    "duplicate_of": kept.term_id,
                    "similarity": float(sim),
                }
            )
            return False
    terms.append(term)
    source_counts[source] = int(source_counts.get(source, 0)) + 1
    return True


def _role_shadow_base_terms(terms: Sequence[DETerm], *, max_terms: int) -> list[DETerm]:
    rows = [
        term
        for term in list(terms)
        if term.ast is not None
        and not str(term.source).startswith("role_shadow")
        and not bool(term.metadata.get("allow_constant", False))
    ]
    rows.sort(key=lambda term: (float(term.complexity), str(term.term_id)))
    return rows[: max(0, int(max_terms))]


def _affine_role_shadow_asts(term: DETerm) -> list[tuple[str, Any, str]]:
    ast = term.ast
    display = str(term.metadata.get("display", term.term_id))
    return [
        ("one_minus", Add(ConstNode(1.0), Mul(ConstNode(-1.0), ast)), f"1 - ({display})"),
        ("minus_one", Add(ast, ConstNode(-1.0)), f"({display}) - 1"),
        ("one_plus", Add(ConstNode(1.0), ast), f"1 + ({display})"),
    ]


def _library_terms_from_result(result: Any) -> list[Any]:
    if isinstance(result, (DESearchResult, DESearchResultMulti)):
        return list(getattr(result, "term_asts", []) or [])
    return []


def _factorized_terms_from_result(result: Any) -> list[tuple[Any, dict[str, Any]]]:
    if not isinstance(result, FactorizedDEResult):
        return []
    out: list[tuple[Any, dict[str, Any]]] = []
    for i, block in enumerate(list(getattr(result, "blocks", []) or [])):
        block_ast = getattr(block, "block_ast", None)
        if block_ast is not None:
            out.append(
                (
                    block_ast,
                    {
                        "kind": "block",
                        "block_index": int(i),
                        "role": str(getattr(block, "role", "")),
                        "display": repr(block_ast),
                    },
                )
            )
    nonanchor = getattr(result, "nonanchor_ast", None)
    if nonanchor is not None:
        out.append((nonanchor, {"kind": "whole_rhs", "display": repr(nonanchor)}))
    return out


def _shortlist_from_factorized_search_result(
    result: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(result, FactorizedSearchDEResult):
        return []
    diagnostics = getattr(result, "diagnostics", {}) or {}
    report = diagnostics.get("report", None) if isinstance(diagnostics, Mapping) else None
    if isinstance(report, Mapping):
        try:
            return factorized_search_report_shortlist(dict(report), limit=int(limit))
        except Exception:
            return []
    payload = {
        "engine": "factorized_search",
        "order": int(result.order),
        "x_axis": int(result.x_axis),
        "feature_names": list(getattr(result, "feature_names", []) or []),
        "expr_ast": result.expr_ast,
        "mapping": dict(getattr(result, "mapping", {}) or {}),
        "mapping_kind": str(getattr(result, "mapping_kind", "")),
        "probe_mse": float(getattr(result, "probe_mse", float("inf"))),
        "probe_rms": float(getattr(result, "probe_rms", float("inf"))),
        "canonical_equation": str(getattr(result, "canonical_equation", "")),
        "shortlist_rank": 0,
    }
    return [payload]


def build_de_term_bank(
    groups: Sequence[DEFeatureGroup],
    *,
    order: int,
    x_axis: int = 0,
    library_results: Any = None,
    factorized_results: Any = None,
    factorized_search_results: Any = None,
    factorized_search_candidates: Sequence[Mapping[str, Any]] | None = None,
    extra_ast_terms: Sequence[Any] | None = None,
    enable_role_shadow: bool = False,
    max_role_shadow_base_terms: int = 12,
    max_role_shadow_terms: int = 24,
    max_whole_rhs_candidates: int = 16,
    max_terms_total: int = 128,
    max_terms_per_source: int = 48,
    max_duplicate_corr: float = 0.999999,
    min_finite_fraction: float = 0.98,
    dtype: torch.dtype = torch.float64,
) -> DETermBank:
    """Normalize current DE/FSS candidate sources into a capped term bank."""

    groups = list(groups or [])
    if not groups:
        return DETermBank(
            order=int(order),
            x_axis=int(x_axis),
            terms=[],
            diagnostics={"status": "EMPTY", "reason": "no_feature_groups"},
        )

    diagnostics: dict[str, Any] = {
        "status": "OK",
        "source_inputs": {},
        "rejected": [],
    }
    terms: list[DETerm] = []
    source_counts: dict[str, int] = {}
    role_shadow_opportunities: list[RoleShadowOpportunity] = []
    serial = count()

    def add_ast_term(ast: Any, *, source: str, metadata: dict[str, Any] | None = None) -> str | None:
        source_norm = _source_name(source, "ast")
        idx = next(serial)
        term_id = f"{source_norm}:{idx}"
        fit_values = _eval_ast_term(ast, groups, x_axis=int(x_axis), split="fit", dtype=dtype)
        probe_values = _eval_ast_term(ast, groups, x_axis=int(x_axis), split="probe", dtype=dtype)
        md = dict(metadata or {})
        md.setdefault("display", _term_display_from_ast(ast))
        if ast is None:
            md["allow_constant"] = True
        term = DETerm(
            term_id=term_id,
            source=source_norm,
            role_hint=str(md.get("role_hint", "rhs")),
            ast=ast,
            complexity=_ast_complexity(ast),
            fit_values=fit_values,
            probe_values=probe_values,
            metadata=md,
        )
        accepted = _append_term(
            terms,
            diagnostics,
            source_counts,
            term,
            max_terms_total=int(max_terms_total),
            max_terms_per_source=int(max_terms_per_source),
            max_duplicate_corr=float(max_duplicate_corr),
            min_finite_fraction=float(min_finite_fraction),
        )
        return term_id if accepted else None

    for result in _as_sequence(library_results):
        if result is None or int(getattr(result, "order", order)) != int(order):
            continue
        lib_terms = _library_terms_from_result(result)
        diagnostics["source_inputs"]["stlsq"] = int(diagnostics["source_inputs"].get("stlsq", 0)) + len(lib_terms)
        for i, ast in enumerate(lib_terms):
            add_ast_term(ast, source="stlsq", metadata={"source_index": int(i)})

    for result in _as_sequence(factorized_results):
        if result is None or int(getattr(result, "order", order)) != int(order):
            continue
        fac_terms = _factorized_terms_from_result(result)
        diagnostics["source_inputs"]["factorized"] = int(diagnostics["source_inputs"].get("factorized", 0)) + len(
            fac_terms
        )
        for ast, metadata in fac_terms:
            add_ast_term(ast, source="factorized", metadata=metadata)

    fss_candidates: list[Mapping[str, Any]] = []
    for result in _as_sequence(factorized_search_results):
        fss_candidates.extend(
            _shortlist_from_factorized_search_result(
                result,
                limit=int(max_whole_rhs_candidates),
            )
        )
    fss_candidates.extend(list(factorized_search_candidates or []))
    diagnostics["source_inputs"]["factorized_search"] = int(len(fss_candidates))
    for i, candidate in enumerate(fss_candidates[: max(0, int(max_whole_rhs_candidates))]):
        if not isinstance(candidate, Mapping):
            continue
        cand_order = int(candidate.get("order", order))
        if cand_order != int(order):
            continue
        source_norm = "factorized_search"
        idx = next(serial)
        try:
            fit_values = _eval_factorized_search_term(
                candidate,
                groups,
                order=int(order),
                x_axis=int(x_axis),
                split="fit",
                dtype=dtype,
            )
            probe_values = _eval_factorized_search_term(
                candidate,
                groups,
                order=int(order),
                x_axis=int(x_axis),
                split="probe",
                dtype=dtype,
            )
        except Exception as exc:
            diagnostics.setdefault("rejected", []).append(
                {
                    "term_id": f"{source_norm}:{idx}",
                    "source": source_norm,
                    "reason": "evaluation_error",
                    "message": str(exc),
                }
            )
            continue
        display = str(candidate.get("canonical_equation", "") or candidate.get("expr", "") or candidate.get("expr_ast", ""))
        term = DETerm(
            term_id=f"{source_norm}:{idx}",
            source=source_norm,
            role_hint="rhs",
            payload=dict(candidate),
            complexity=_candidate_complexity(candidate),
            fit_values=fit_values,
            probe_values=probe_values,
            metadata={
                "source_index": int(i),
                "candidate_rank": candidate.get("candidate_rank", candidate.get("shortlist_rank", i)),
                "display": display,
                "mapping_kind": candidate.get("mapping_kind", None),
            },
        )
        _append_term(
            terms,
            diagnostics,
            source_counts,
            term,
            max_terms_total=int(max_terms_total),
            max_terms_per_source=int(max_terms_per_source),
            max_duplicate_corr=float(max_duplicate_corr),
            min_finite_fraction=float(min_finite_fraction),
        )

    for i, ast in enumerate(list(extra_ast_terms or [])):
        diagnostics["source_inputs"]["extra_ast"] = int(diagnostics["source_inputs"].get("extra_ast", 0)) + 1
        add_ast_term(ast, source="extra_ast", metadata={"source_index": int(i)})

    if bool(enable_role_shadow):
        attempted = 0
        base_terms = _role_shadow_base_terms(terms, max_terms=int(max_role_shadow_base_terms))
        for base in base_terms:
            for shadow_kind, shadow_ast, display in _affine_role_shadow_asts(base):
                if attempted >= max(0, int(max_role_shadow_terms)):
                    break
                attempted += 1
                emitted = add_ast_term(
                    shadow_ast,
                    source="role_shadow_affine",
                    metadata={
                        "role_shadow": True,
                        "role_shadow_kind": shadow_kind,
                        "role_shadow_source": "affine_shift",
                        "role_hint": "denominator",
                        "allow_affine_duplicate": True,
                        "source_term_id": base.term_id,
                        "source_display": str(base.metadata.get("display", base.term_id)),
                        "display": display,
                    },
                )
                if emitted is None:
                    continue
                role_shadow_opportunities.append(
                    RoleShadowOpportunity(
                        opportunity_id=f"role_shadow_affine:{len(role_shadow_opportunities)}",
                        source="role_shadow_affine",
                        input_term_ids=(base.term_id,),
                        emitted_term_ids=(emitted,),
                        score_hint=None,
                        metadata={
                            "role_shadow_kind": shadow_kind,
                            "role_hint": "denominator",
                            "display": display,
                        },
                    )
                )
            if attempted >= max(0, int(max_role_shadow_terms)):
                break
        diagnostics["role_shadow"] = {
            "enabled": True,
            "base_term_count": int(len(base_terms)),
            "attempted": int(attempted),
            "kept": int(len(role_shadow_opportunities)),
            "opportunities": [opp.to_dict() for opp in role_shadow_opportunities[:24]],
        }
    else:
        diagnostics["role_shadow"] = {"enabled": False}

    diagnostics["source_counts"] = {str(k): int(v) for k, v in sorted(source_counts.items())}
    diagnostics["term_count"] = int(len(terms))
    diagnostics["rejected_count"] = int(len(diagnostics.get("rejected", []) or []))
    return DETermBank(order=int(order), x_axis=int(x_axis), terms=terms, diagnostics=diagnostics)


def _anchor_values(
    groups: Sequence[DEFeatureGroup],
    *,
    order: int,
    split: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    vals = []
    for group in groups:
        fit_anchor, probe_anchor = _anchor_tensor(group.features, order=int(order))
        vals.append(_flatten_tensor(fit_anchor if str(split) == "fit" else probe_anchor, dtype=dtype))
    return tuple(vals)


def _support_matrix(terms: Sequence[DETerm], support: Sequence[int], *, split: str) -> tuple[torch.Tensor, ...]:
    out = []
    for group_idx in range(len(terms[0].fit_values) if terms else 0):
        cols = []
        for term_idx in support:
            term = terms[int(term_idx)]
            values = term.fit_values if str(split) == "fit" else term.probe_values
            cols.append(_flatten_tensor(values[group_idx]))
        if cols:
            out.append(torch.stack(cols, dim=1))
        else:
            n = int(terms[0].fit_values[group_idx].numel())
            out.append(torch.empty((n, 0), dtype=torch.float64))
    return tuple(out)


def _masked_fit(Phi_parts: Sequence[torch.Tensor], y_parts: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    Xs = []
    ys = []
    for Phi, y in zip(Phi_parts, y_parts):
        yv = _flatten_tensor(y)
        X = torch.as_tensor(Phi, dtype=torch.float64)
        mask = torch.isfinite(yv)
        if int(X.numel()) > 0:
            mask &= torch.isfinite(X).all(dim=1)
        if int(mask.sum()) > 0:
            Xs.append(X[mask])
            ys.append(yv[mask])
    if not Xs:
        return torch.empty((0, 0), dtype=torch.float64), torch.empty((0,), dtype=torch.float64)
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


def _ridge_lstsq(Phi: torch.Tensor, y: torch.Tensor, *, ridge: float) -> torch.Tensor:
    if int(Phi.shape[1]) == 0:
        return torch.empty((0,), dtype=Phi.dtype, device=Phi.device)
    reg = float(ridge)
    A = Phi.T @ Phi
    if reg > 0.0:
        A = A + reg * torch.eye(int(A.shape[0]), dtype=Phi.dtype, device=Phi.device)
    b = Phi.T @ y
    try:
        return torch.linalg.solve(A, b).reshape(-1)
    except Exception:
        return torch.linalg.lstsq(Phi, y).solution.reshape(-1)


def _rms(values: torch.Tensor) -> float:
    v = _flatten_tensor(values)
    finite = torch.isfinite(v)
    if int(finite.sum()) <= 0:
        return float("inf")
    return float(torch.sqrt(torch.mean(v[finite].square())).detach().cpu().item())


def _nrmse(residual: torch.Tensor, anchor: torch.Tensor) -> float:
    r = _flatten_tensor(residual)
    a = _flatten_tensor(anchor)
    n = min(int(r.numel()), int(a.numel()))
    if n <= 0:
        return float("inf")
    r = r[:n]
    a = a[:n]
    finite = torch.isfinite(r) & torch.isfinite(a)
    if int(finite.sum()) <= 0:
        return float("inf")
    num = float(torch.linalg.norm(r[finite]).detach().cpu().item())
    den = float(torch.linalg.norm(a[finite]).detach().cpu().item()) + 1.0e-12
    return float(num / den)


def _mean_max(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("inf"), float("inf")
    return float(sum(finite) / len(finite)), float(max(finite))


def _coefficient_stability(
    Phi_fit_parts: Sequence[torch.Tensor],
    y_fit_parts: Sequence[torch.Tensor],
    coeffs: torch.Tensor,
    *,
    ridge: float,
) -> float:
    if len(Phi_fit_parts) <= 1 or int(coeffs.numel()) <= 0:
        return 0.0
    rows = []
    for Phi, y in zip(Phi_fit_parts, y_fit_parts):
        X, yy = _masked_fit([Phi], [y])
        if int(X.shape[0]) < max(4, int(X.shape[1]) + 1):
            continue
        try:
            rows.append(_ridge_lstsq(X, yy, ridge=float(ridge)).detach().cpu())
        except Exception:
            continue
    if len(rows) <= 1:
        return 0.0
    C = torch.stack(rows, dim=0).to(dtype=torch.float64)
    center = coeffs.detach().cpu().to(dtype=torch.float64).reshape(1, -1)
    scale = torch.clamp(torch.abs(center), min=1.0e-8)
    rel = torch.std(C, dim=0) / scale.reshape(-1)
    signs = torch.sign(C)
    global_sign = torch.sign(center).reshape(-1)
    sign_flip = torch.mean((signs != global_sign.reshape(1, -1)).to(dtype=torch.float64)).item()
    return float(torch.mean(rel).item() + float(sign_flip))


def _support_complexity(terms: Sequence[DETerm], support: Sequence[int]) -> float:
    return float(sum(float(terms[int(i)].complexity) for i in support) + 0.25 * len(support))


def _score_support(
    *,
    probe_rms_mean: float,
    probe_rms_max: float,
    complexity: float,
    coefficient_stability: float,
    complexity_penalty: float,
) -> float:
    eps = 1.0e-18
    if not math.isfinite(probe_rms_max):
        return float("inf")
    return float(
        math.log10(max(float(probe_rms_max), eps))
        + 0.25 * math.log10(max(float(probe_rms_mean), eps))
        + float(complexity_penalty) * float(complexity)
        + 0.05 * min(float(coefficient_stability), 100.0)
    )


def _fit_explicit_support(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    support: Sequence[int],
    *,
    ridge: float,
    complexity_penalty: float,
    ast_serializer: Any = None,
    dtype: torch.dtype,
) -> DESupportCandidate | None:
    support = tuple(sorted(int(i) for i in support))
    if not support:
        return None
    terms = list(bank.terms)
    Phi_fit_parts_all = _support_matrix(terms, support, split="fit")
    Phi_probe_parts_all = _support_matrix(terms, support, split="probe")
    anchor_fit_all = _anchor_values(groups, order=int(bank.order), split="fit", dtype=dtype)
    anchor_probe_all = _anchor_values(groups, order=int(bank.order), split="probe", dtype=dtype)

    Phi_fit_parts = []
    y_fit_parts = []
    for group, Phi, anchor in zip(groups, Phi_fit_parts_all, anchor_fit_all):
        if not bool(getattr(group, "use_for_fit", True)):
            continue
        Phi_fit_parts.append(Phi)
        y_fit_parts.append(-_flatten_tensor(anchor, dtype=dtype))
    X_fit, y_fit = _masked_fit(Phi_fit_parts, y_fit_parts)
    if int(X_fit.shape[0]) < max(10, 2 * int(len(support))):
        return None
    coeffs = _ridge_lstsq(X_fit, y_fit, ridge=float(ridge))
    if int(coeffs.numel()) != int(len(support)) or not torch.isfinite(coeffs).all():
        return None

    fit_rms = []
    probe_rms = []
    probe_nrmse = []
    for group, Phi, anchor in zip(groups, Phi_fit_parts_all, anchor_fit_all):
        if not bool(getattr(group, "use_for_fit", True)):
            continue
        residual = _flatten_tensor(anchor, dtype=dtype) + torch.as_tensor(Phi, dtype=dtype) @ coeffs.to(dtype=dtype)
        fit_rms.append(_rms(residual))
    for group, Phi, anchor in zip(groups, Phi_probe_parts_all, anchor_probe_all):
        if not bool(getattr(group, "use_for_probe", True)):
            continue
        residual = _flatten_tensor(anchor, dtype=dtype) + torch.as_tensor(Phi, dtype=dtype) @ coeffs.to(dtype=dtype)
        probe_rms.append(_rms(residual))
        probe_nrmse.append(_nrmse(residual, anchor))

    fit_rms_mean, fit_rms_max = _mean_max(fit_rms)
    probe_rms_mean, probe_rms_max = _mean_max(probe_rms)
    probe_nrmse_mean, probe_nrmse_max = _mean_max(probe_nrmse)
    complexity = _support_complexity(terms, support)
    stability = _coefficient_stability(Phi_fit_parts, y_fit_parts, coeffs, ridge=float(ridge))
    score = _score_support(
        probe_rms_mean=probe_rms_mean,
        probe_rms_max=probe_rms_max,
        complexity=complexity,
        coefficient_stability=stability,
        complexity_penalty=float(complexity_penalty),
    )
    term_ids = tuple(terms[int(i)].term_id for i in support)
    support_terms = [terms[int(i)] for i in support]
    materializable = _terms_materializable(support_terms)
    support_asts = [term.ast for term in support_terms]
    if materializable:
        nonanchor_ast = _sum_scaled_asts(support_asts, coeffs.detach().cpu().tolist())
        canonical_equation = (
            f"{_anchor_name(int(bank.order), x_axis=int(bank.x_axis))} + {repr(nonanchor_ast)} = 0"
            if nonanchor_ast is not None
            else f"{_anchor_name(int(bank.order), x_axis=int(bank.x_axis))} = 0"
        )
        validation_candidate = _materialize_explicit_validation_candidate(
            order=int(bank.order),
            x_axis=int(bank.x_axis),
            asts=support_asts,
            coeffs=coeffs.detach().cpu().tolist(),
            canonical_equation=canonical_equation,
            ast_serializer=ast_serializer,
        )
    else:
        canonical_equation = "{} + {} = 0".format(
            _anchor_name(int(bank.order), x_axis=int(bank.x_axis)),
            " + ".join(str(term.metadata.get("display", term.term_id)) for term in support_terms),
        )
        validation_candidate = None
    return DESupportCandidate(
        support_id="support:" + ",".join(str(i) for i in support),
        form="explicit_linear",
        order=int(bank.order),
        x_axis=int(bank.x_axis),
        term_ids=term_ids,
        coefficients=tuple(float(v) for v in coeffs.detach().cpu().tolist()),
        ridge=float(ridge),
        score=float(score),
        fit_rms_mean=float(fit_rms_mean),
        fit_rms_max=float(fit_rms_max),
        probe_rms_mean=float(probe_rms_mean),
        probe_rms_max=float(probe_rms_max),
        probe_nrmse_mean=float(probe_nrmse_mean),
        probe_nrmse_max=float(probe_nrmse_max),
        coefficient_stability=float(stability),
        complexity=float(complexity),
        canonical_equation=canonical_equation,
        validation_candidate=validation_candidate,
        metadata={
            "materializable": bool(materializable and validation_candidate is not None),
            "term_summaries": [term.summary() for term in support_terms],
            "fit_rms_by_traj": [float(v) for v in fit_rms],
            "probe_rms_by_traj": [float(v) for v in probe_rms],
            "probe_nrmse_by_traj": [float(v) for v in probe_nrmse],
        },
    )


def _best_ridge_support(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    support: Sequence[int],
    *,
    ridge_grid: Sequence[float],
    complexity_penalty: float,
    ast_serializer: Any = None,
    dtype: torch.dtype,
) -> DESupportCandidate | None:
    best = None
    best_key = None
    for ridge in ridge_grid:
        cand = _fit_explicit_support(
            bank,
            groups,
            support,
            ridge=float(ridge),
            complexity_penalty=float(complexity_penalty),
            ast_serializer=ast_serializer,
            dtype=dtype,
        )
        if cand is None:
            continue
        key = (float(cand.score), float(cand.probe_rms_max), float(cand.complexity), tuple(cand.term_ids))
        if best_key is None or key < best_key:
            best = cand
            best_key = key
    return best


def _fit_residual_for_support(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    support: Sequence[int],
    coeffs: Sequence[float],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    terms = list(bank.terms)
    Phi_fit_parts_all = _support_matrix(terms, support, split="fit")
    anchor_fit_all = _anchor_values(groups, order=int(bank.order), split="fit", dtype=dtype)
    coeff_t = torch.as_tensor(list(coeffs), dtype=dtype)
    residuals = []
    for group, Phi, anchor in zip(groups, Phi_fit_parts_all, anchor_fit_all):
        if not bool(getattr(group, "use_for_fit", True)):
            continue
        residuals.append(_flatten_tensor(-anchor, dtype=dtype) - torch.as_tensor(Phi, dtype=dtype) @ coeff_t)
    return _concat_values(residuals)


def _rank_expansion_terms(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    *,
    used: set[int],
    residual: torch.Tensor,
    dtype: torch.dtype,
) -> list[int]:
    scores = []
    residual = _flatten_tensor(residual, dtype=dtype)
    for idx, term in enumerate(bank.terms):
        if idx in used:
            continue
        vals = []
        for group_idx, group in enumerate(groups):
            if bool(getattr(group, "use_for_fit", True)):
                vals.append(term.fit_values[group_idx])
        col = _concat_values(vals).to(dtype=dtype)
        n = min(int(col.numel()), int(residual.numel()))
        if n <= 1:
            continue
        col = col[:n]
        rr = residual[:n]
        finite = torch.isfinite(col) & torch.isfinite(rr)
        if int(finite.sum()) <= 1:
            continue
        col = col[finite] - torch.mean(col[finite])
        rr = rr[finite] - torch.mean(rr[finite])
        denom = float(torch.linalg.norm(col).detach().cpu().item()) * float(torch.linalg.norm(rr).detach().cpu().item())
        score = 0.0 if denom <= 1.0e-14 else abs(float(torch.dot(col, rr).detach().cpu().item()) / denom)
        scores.append((-score, float(bank.terms[idx].complexity), idx))
    scores.sort()
    return [int(idx) for _, _, idx in scores]


def assemble_explicit_supports(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    *,
    max_support_width: int = 5,
    beam_width: int = 32,
    expansions_per_support: int = 16,
    ridge_grid: Sequence[float] = (0.0, 1.0e-12, 1.0e-10, 1.0e-8),
    complexity_penalty: float = 1.0e-4,
    shortlist_topk: int = 16,
    ast_serializer: Any = None,
    dtype: torch.dtype = torch.float64,
) -> list[DESupportCandidate]:
    """Assemble explicit DE supports and score them by held-out trajectories."""

    if not bank.terms or not groups:
        return []

    seen: set[tuple[int, ...]] = set()
    all_candidates: list[DESupportCandidate] = []

    def add_support(support: Sequence[int]) -> DESupportCandidate | None:
        key = tuple(sorted(int(i) for i in support))
        if key in seen:
            return None
        seen.add(key)
        cand = _best_ridge_support(
            bank,
            groups,
            key,
            ridge_grid=tuple(float(v) for v in ridge_grid),
            complexity_penalty=float(complexity_penalty),
            ast_serializer=ast_serializer,
            dtype=dtype,
        )
        if cand is not None:
            all_candidates.append(cand)
        return cand

    beam = []
    for idx in range(len(bank.terms)):
        cand = add_support((idx,))
        if cand is not None:
            beam.append(cand)
    beam.sort(key=lambda c: (float(c.score), float(c.probe_rms_max), float(c.complexity), tuple(c.term_ids)))
    beam = beam[: max(1, int(beam_width))]

    for _width in range(2, max(1, int(max_support_width)) + 1):
        next_beam: list[DESupportCandidate] = []
        for cand in beam:
            used_ids = set(cand.term_ids)
            used = {idx for idx, term in enumerate(bank.terms) if term.term_id in used_ids}
            residual = _fit_residual_for_support(
                bank,
                groups,
                [idx for idx, term in enumerate(bank.terms) if term.term_id in used_ids],
                cand.coefficients,
                dtype=dtype,
            )
            expansion_terms = _rank_expansion_terms(bank, groups, used=used, residual=residual, dtype=dtype)
            for idx in expansion_terms[: max(1, int(expansions_per_support))]:
                expanded = tuple(sorted((*used, int(idx))))
                new_cand = add_support(expanded)
                if new_cand is not None:
                    next_beam.append(new_cand)
        if not next_beam:
            break
        next_beam.sort(key=lambda c: (float(c.score), float(c.probe_rms_max), float(c.complexity), tuple(c.term_ids)))
        beam = next_beam[: max(1, int(beam_width))]

    all_candidates.sort(key=lambda c: (float(c.score), float(c.probe_rms_max), float(c.complexity), tuple(c.term_ids)))
    return all_candidates[: max(0, int(shortlist_topk))]


def _support_indices_from_term_ids(bank: DETermBank, term_ids: Sequence[str]) -> tuple[int, ...]:
    by_id = {term.term_id: idx for idx, term in enumerate(bank.terms)}
    out = []
    for term_id in list(term_ids):
        idx = by_id.get(str(term_id), None)
        if idx is not None:
            out.append(int(idx))
    return tuple(out)


def _rank_denominator_term_scores(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    numerator_support: Sequence[int],
    numerator_coeffs: Sequence[float],
    *,
    exclude: set[int],
    dtype: torch.dtype,
) -> list[tuple[int, float]]:
    terms = list(bank.terms)
    Phi_num_parts = _support_matrix(terms, numerator_support, split="fit")
    anchor_fit = _anchor_values(groups, order=int(bank.order), split="fit", dtype=dtype)
    coeff_t = torch.as_tensor(list(numerator_coeffs), dtype=dtype)
    residual_parts = []
    anchor_parts = []
    for group, Phi, anchor in zip(groups, Phi_num_parts, anchor_fit):
        if not bool(getattr(group, "use_for_fit", True)):
            continue
        residual_parts.append(_flatten_tensor(anchor, dtype=dtype) + torch.as_tensor(Phi, dtype=dtype) @ coeff_t)
        anchor_parts.append(_flatten_tensor(anchor, dtype=dtype))
    residual = _concat_values(residual_parts).to(dtype=dtype)
    anchor = _concat_values(anchor_parts).to(dtype=dtype)
    scores = []
    for idx, term in enumerate(terms):
        if idx in exclude:
            continue
        vals = []
        for group_idx, group in enumerate(groups):
            if bool(getattr(group, "use_for_fit", True)):
                vals.append(term.fit_values[group_idx])
        dcol = anchor * _concat_values(vals).to(dtype=dtype)
        n = min(int(dcol.numel()), int(residual.numel()))
        if n <= 1:
            continue
        dcol = dcol[:n]
        target = -residual[:n]
        finite = torch.isfinite(dcol) & torch.isfinite(target)
        if int(finite.sum()) <= 1:
            continue
        d0 = dcol[finite] - torch.mean(dcol[finite])
        t0 = target[finite] - torch.mean(target[finite])
        denom = float(torch.linalg.norm(d0).detach().cpu().item()) * float(torch.linalg.norm(t0).detach().cpu().item())
        corr = 0.0 if denom <= 1.0e-14 else abs(float(torch.dot(d0, t0).detach().cpu().item()) / denom)
        role_bonus = 0.01 if str(term.source).startswith("role_shadow") else 0.0
        scores.append((-(corr + role_bonus), float(term.complexity), idx, float(corr)))
    scores.sort()
    return [(int(idx), float(corr)) for _, _, idx, corr in scores]


def _rank_denominator_terms(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    numerator_support: Sequence[int],
    numerator_coeffs: Sequence[float],
    *,
    exclude: set[int],
    dtype: torch.dtype,
) -> list[int]:
    return [
        int(idx)
        for idx, _corr in _rank_denominator_term_scores(
            bank,
            groups,
            numerator_support,
            numerator_coeffs,
            exclude=exclude,
            dtype=dtype,
        )
    ]


def _pivot_candidate_indices(
    bank: DETermBank,
    ranked_den_terms: Sequence[int],
    *,
    exclude: set[int],
    max_pivots: int,
) -> list[int | None]:
    if int(max_pivots) <= 0:
        return [None]
    terms = list(bank.terms)
    out: list[int | None] = [None]
    seen: set[int] = set()

    def add(idx: int) -> None:
        idx_i = int(idx)
        if idx_i in seen or idx_i in exclude:
            return
        if idx_i < 0 or idx_i >= len(terms):
            return
        term = terms[idx_i]
        if term.ast is None:
            return
        seen.add(idx_i)
        out.append(idx_i)

    role_rows = [
        (float(term.complexity), idx)
        for idx, term in enumerate(terms)
        if idx not in exclude
        and term.ast is not None
        and str(term.source).startswith("role_shadow")
        and str(term.role_hint) in {"denominator", "mass", "unknown"}
    ]
    role_rows.sort()
    for _complexity, idx in role_rows:
        add(idx)
        if len(out) >= int(max_pivots) + 1:
            return out

    for idx in list(ranked_den_terms):
        add(int(idx))
        if len(out) >= int(max_pivots) + 1:
            break
    return out


def _denominator_safety(
    numerator_parts: Sequence[torch.Tensor],
    denominator_parts: Sequence[torch.Tensor],
    *,
    min_abs_floor: float = 1.0e-10,
) -> dict[str, Any]:
    den = _concat_values(denominator_parts).to(dtype=torch.float64)
    num = _concat_values(numerator_parts).to(dtype=torch.float64)
    finite = torch.isfinite(den)
    finite_fraction = float(torch.mean(finite.to(dtype=torch.float64)).detach().cpu().item()) if int(den.numel()) else 0.0
    if int(finite.sum()) <= 0:
        return {
            "safe": False,
            "finite_fraction": finite_fraction,
            "reason": "no_finite_denominator_values",
        }
    den_f = den[finite]
    abs_den = torch.abs(den_f)
    median_abs = float(torch.median(abs_den).detach().cpu().item())
    min_abs = float(torch.min(abs_den).detach().cpu().item())
    q01 = float(torch.quantile(abs_den, 0.01).detach().cpu().item()) if int(abs_den.numel()) > 1 else min_abs
    threshold = max(float(min_abs_floor), 1.0e-8 * max(float(median_abs), 1.0))
    near_zero = abs_den <= threshold
    near_zero_fraction = float(torch.mean(near_zero.to(dtype=torch.float64)).detach().cpu().item())
    ratio_finite_fraction = 0.0
    max_abs_explicit = float("inf")
    co_vanish_fraction = None
    if int(num.numel()) == int(den.numel()) and int(num.numel()) > 0:
        finite_ratio = torch.isfinite(num) & torch.isfinite(den) & (torch.abs(den) > threshold)
        ratio_finite_fraction = float(torch.mean(finite_ratio.to(dtype=torch.float64)).detach().cpu().item())
        if int(finite_ratio.sum()) > 0:
            ratio = num[finite_ratio] / den[finite_ratio]
            max_abs_explicit = float(torch.max(torch.abs(ratio)).detach().cpu().item())
        if bool(torch.any(near_zero)):
            num_f = num[finite]
            co_vanish_fraction = float(
                torch.mean((torch.abs(num_f[near_zero]) <= 10.0 * threshold).to(dtype=torch.float64)).detach().cpu().item()
            )
    safe = bool(
        finite_fraction >= 0.98
        and near_zero_fraction <= 0.01
        and q01 > threshold
        and ratio_finite_fraction >= 0.98
        and math.isfinite(max_abs_explicit)
        and max_abs_explicit < 1.0e12
    )
    reason = "ok" if safe else "unsafe_denominator"
    return {
        "safe": safe,
        "reason": reason,
        "finite_fraction": finite_fraction,
        "ratio_finite_fraction": ratio_finite_fraction,
        "near_zero_fraction": near_zero_fraction,
        "min_abs": min_abs,
        "q01_abs": q01,
        "median_abs": median_abs,
        "threshold": threshold,
        "max_abs_explicit_rhs": max_abs_explicit if math.isfinite(max_abs_explicit) else None,
        "co_vanish_fraction": co_vanish_fraction,
    }


def _fit_implicit_rational_support(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    numerator_support: Sequence[int],
    denominator_support: Sequence[int],
    *,
    pivot_support: int | None = None,
    denominator_rank_corr: Mapping[int, float] | None = None,
    ridge: float,
    complexity_penalty: float,
    ast_serializer: Any = None,
    dtype: torch.dtype,
) -> DEImplicitRationalCandidate | None:
    numerator_support = tuple(sorted(int(i) for i in numerator_support))
    denominator_support = tuple(sorted(int(i) for i in denominator_support))
    if not numerator_support or (not denominator_support and pivot_support is None):
        return None
    terms = list(bank.terms)
    pivot_idx = None if pivot_support is None else int(pivot_support)
    if pivot_idx is not None and (pivot_idx < 0 or pivot_idx >= len(terms)):
        return None
    Phi_num_fit_all = _support_matrix(terms, numerator_support, split="fit")
    Phi_den_fit_all = _support_matrix(terms, denominator_support, split="fit")
    Phi_num_probe_all = _support_matrix(terms, numerator_support, split="probe")
    Phi_den_probe_all = _support_matrix(terms, denominator_support, split="probe")
    Phi_pivot_fit_all = _support_matrix(terms, (pivot_idx,), split="fit") if pivot_idx is not None else ()
    Phi_pivot_probe_all = _support_matrix(terms, (pivot_idx,), split="probe") if pivot_idx is not None else ()
    anchor_fit_all = _anchor_values(groups, order=int(bank.order), split="fit", dtype=dtype)
    anchor_probe_all = _anchor_values(groups, order=int(bank.order), split="probe", dtype=dtype)

    X_parts = []
    y_parts = []
    for group_idx, (group, Phi_num, Phi_den, anchor) in enumerate(zip(groups, Phi_num_fit_all, Phi_den_fit_all, anchor_fit_all)):
        if not bool(getattr(group, "use_for_fit", True)):
            continue
        a = _flatten_tensor(anchor, dtype=dtype)
        if pivot_idx is None:
            pivot = torch.ones_like(a, dtype=dtype)
        else:
            pivot = torch.as_tensor(Phi_pivot_fit_all[group_idx], dtype=dtype).reshape(-1)
        X = torch.cat([torch.as_tensor(Phi_num, dtype=dtype), a.reshape(-1, 1) * torch.as_tensor(Phi_den, dtype=dtype)], dim=1)
        y = -a * pivot
        X_parts.append(X)
        y_parts.append(y)
    X_fit, y_fit = _masked_fit(X_parts, y_parts)
    n_coeff = int(len(numerator_support) + len(denominator_support))
    if int(X_fit.shape[0]) < max(10, 2 * n_coeff):
        return None
    coeffs = _ridge_lstsq(X_fit, y_fit, ridge=float(ridge))
    if int(coeffs.numel()) != n_coeff or not torch.isfinite(coeffs).all():
        return None
    num_coeffs = coeffs[: len(numerator_support)]
    den_coeffs = coeffs[len(numerator_support):]

    implicit_fit_rms = []
    implicit_probe_rms = []
    explicit_probe_rms = []
    numerator_probe_parts = []
    for group_idx, (group, Phi_num, Phi_den, anchor) in enumerate(zip(groups, Phi_num_fit_all, Phi_den_fit_all, anchor_fit_all)):
        if not bool(getattr(group, "use_for_fit", True)):
            continue
        a = _flatten_tensor(anchor, dtype=dtype)
        if pivot_idx is None:
            pivot = torch.ones_like(a, dtype=dtype)
        else:
            pivot = torch.as_tensor(Phi_pivot_fit_all[group_idx], dtype=dtype).reshape(-1)
        numerator = torch.as_tensor(Phi_num, dtype=dtype) @ num_coeffs.to(dtype=dtype)
        denominator_rest = torch.as_tensor(Phi_den, dtype=dtype) @ den_coeffs.to(dtype=dtype)
        residual = numerator + a * (pivot + denominator_rest)
        implicit_fit_rms.append(_rms(residual))
    denominator_probe_parts = []
    for group_idx, (group, Phi_num, Phi_den, anchor) in enumerate(zip(groups, Phi_num_probe_all, Phi_den_probe_all, anchor_probe_all)):
        if not bool(getattr(group, "use_for_probe", True)):
            continue
        a = _flatten_tensor(anchor, dtype=dtype)
        if pivot_idx is None:
            pivot = torch.ones_like(a, dtype=dtype)
        else:
            pivot = torch.as_tensor(Phi_pivot_probe_all[group_idx], dtype=dtype).reshape(-1)
        numerator = torch.as_tensor(Phi_num, dtype=dtype) @ num_coeffs.to(dtype=dtype)
        denominator_rest = torch.as_tensor(Phi_den, dtype=dtype) @ den_coeffs.to(dtype=dtype)
        denominator = pivot + denominator_rest
        implicit_residual = numerator + a * denominator
        with torch.no_grad():
            explicit_residual = a + numerator / denominator
        numerator_probe_parts.append(numerator)
        denominator_probe_parts.append(denominator)
        implicit_probe_rms.append(_rms(implicit_residual))
        explicit_probe_rms.append(_rms(explicit_residual))

    implicit_fit_mean, implicit_fit_max = _mean_max(implicit_fit_rms)
    implicit_probe_mean, implicit_probe_max = _mean_max(implicit_probe_rms)
    explicit_probe_mean, explicit_probe_max = _mean_max(explicit_probe_rms)
    safety = _denominator_safety(numerator_probe_parts, denominator_probe_parts)
    stability = _coefficient_stability(X_parts, y_parts, coeffs, ridge=float(ridge))
    complexity = (
        _support_complexity(terms, numerator_support)
        + _support_complexity(terms, denominator_support)
        + (1.0 if pivot_idx is None else float(terms[pivot_idx].complexity) + 0.5)
    )
    safety_penalty = 0.0 if bool(safety.get("safe", False)) else 4.0
    score = float(
        math.log10(max(float(implicit_probe_max), 1.0e-18))
        + 0.25 * math.log10(max(float(implicit_probe_mean), 1.0e-18))
        + float(complexity_penalty) * float(complexity)
        + 0.05 * min(float(stability), 100.0)
        + safety_penalty
    )

    numerator_terms = [terms[int(i)] for i in numerator_support]
    denominator_terms = [terms[int(i)] for i in denominator_support]
    pivot_term = None if pivot_idx is None else terms[pivot_idx]
    pivot_terms = [] if pivot_term is None else [pivot_term]
    materializable = (
        _terms_materializable(numerator_terms)
        and _terms_materializable(denominator_terms)
        and _terms_materializable(pivot_terms)
    )
    num_asts = [term.ast for term in numerator_terms]
    den_asts = [term.ast for term in denominator_terms]
    pivot_ast = None if pivot_term is None else pivot_term.ast
    if materializable:
        numerator_ast = _sum_scaled_asts(num_asts, num_coeffs.detach().cpu().tolist())
        denominator_rest_ast = _sum_scaled_asts(den_asts, den_coeffs.detach().cpu().tolist())
        denominator_ast = ConstNode(1.0) if pivot_ast is None else pivot_ast
        if denominator_rest_ast is not None:
            denominator_ast = Add(denominator_ast, denominator_rest_ast)
        explicit_ast = Mul(numerator_ast, Pow(denominator_ast, -1.0)) if numerator_ast is not None else None
        canonical_equation = (
            f"{_anchor_name(int(bank.order), x_axis=int(bank.x_axis))} + {repr(explicit_ast)} = 0"
            if explicit_ast is not None
            else f"{_anchor_name(int(bank.order), x_axis=int(bank.x_axis))} = 0"
        )
        validation_candidate = (
            _materialize_rational_validation_candidate(
                order=int(bank.order),
                x_axis=int(bank.x_axis),
                numerator_asts=num_asts,
                numerator_coeffs=num_coeffs.detach().cpu().tolist(),
                denominator_asts=den_asts,
                denominator_coeffs=den_coeffs.detach().cpu().tolist(),
                canonical_equation=canonical_equation,
                ast_serializer=ast_serializer,
                pivot_ast=pivot_ast,
            )
            if bool(safety.get("safe", False))
            else None
        )
    else:
        pivot_display = "1" if pivot_term is None else str(pivot_term.metadata.get("display", pivot_term.term_id))
        denominator_display = " + ".join(str(term.metadata.get("display", term.term_id)) for term in denominator_terms)
        canonical_equation = "{} + ({}) / ({} + {}) = 0".format(
            _anchor_name(int(bank.order), x_axis=int(bank.x_axis)),
            " + ".join(str(term.metadata.get("display", term.term_id)) for term in numerator_terms),
            pivot_display,
            denominator_display,
        )
        validation_candidate = None
    pivot_id = "const_one" if pivot_term is None else pivot_term.term_id
    denom_corr = {
        terms[int(idx)].term_id: float(corr)
        for idx, corr in dict(denominator_rank_corr or {}).items()
        if int(idx) in set(denominator_support)
    }

    return DEImplicitRationalCandidate(
        support_id=(
            "implicit:"
            + ",".join(str(i) for i in numerator_support)
            + "|pivot:"
            + str("const" if pivot_idx is None else pivot_idx)
            + "|den:"
            + ",".join(str(i) for i in denominator_support)
        ),
        order=int(bank.order),
        x_axis=int(bank.x_axis),
        numerator_term_ids=tuple(term.term_id for term in numerator_terms),
        denominator_term_ids=tuple(term.term_id for term in denominator_terms),
        numerator_coefficients=tuple(float(v) for v in num_coeffs.detach().cpu().tolist()),
        denominator_coefficients=tuple(float(v) for v in den_coeffs.detach().cpu().tolist()),
        pivot=pivot_id,
        ridge=float(ridge),
        score=float(score),
        implicit_fit_rms_mean=float(implicit_fit_mean),
        implicit_fit_rms_max=float(implicit_fit_max),
        implicit_probe_rms_mean=float(implicit_probe_mean),
        implicit_probe_rms_max=float(implicit_probe_max),
        explicit_probe_rms_mean=float(explicit_probe_mean),
        explicit_probe_rms_max=float(explicit_probe_max),
        denominator_safety=safety,
        coefficient_stability=float(stability),
        complexity=float(complexity),
        canonical_equation=canonical_equation,
        validation_candidate=validation_candidate,
        metadata={
            "materializable": bool(validation_candidate is not None),
            "numerator_terms": [term.summary() for term in numerator_terms],
            "pivot_term": None if pivot_term is None else pivot_term.summary(),
            "denominator_terms": [term.summary() for term in denominator_terms],
            "quotient_residual_corr": denom_corr,
            "implicit_fit_rms_by_traj": [float(v) for v in implicit_fit_rms],
            "implicit_probe_rms_by_traj": [float(v) for v in implicit_probe_rms],
            "explicit_probe_rms_by_traj": [float(v) for v in explicit_probe_rms],
        },
    )


def assemble_implicit_rational_supports(
    bank: DETermBank,
    groups: Sequence[DEFeatureGroup],
    explicit_supports: Sequence[DESupportCandidate],
    *,
    max_numerator_blocks: int = 32,
    max_denominator_terms: int = 32,
    max_denominator_width: int = 2,
    enable_pivoted: bool = False,
    max_pivots: int = 4,
    max_implicit_candidates: int = 64,
    ridge_grid: Sequence[float] = (0.0, 1.0e-12, 1.0e-10, 1.0e-8),
    complexity_penalty: float = 1.0e-4,
    ast_serializer: Any = None,
    dtype: torch.dtype = torch.float64,
) -> list[DEImplicitRationalCandidate]:
    """Assemble implicit rational/mass-form candidates from a term bank."""

    if not bank.terms or not groups or not explicit_supports:
        return []
    seen: set[tuple[tuple[int, ...], int | None, tuple[int, ...]]] = set()
    out: list[DEImplicitRationalCandidate] = []
    for numerator in list(explicit_supports)[: max(1, int(max_numerator_blocks))]:
        numerator_support = _support_indices_from_term_ids(bank, numerator.term_ids)
        if not numerator_support:
            continue
        ranked_den_scores = _rank_denominator_term_scores(
            bank,
            groups,
            numerator_support,
            numerator.coefficients,
            exclude=set(),
            dtype=dtype,
        )[: max(1, int(max_denominator_terms))]
        ranked_den_terms = [int(idx) for idx, _corr in ranked_den_scores]
        denominator_rank_corr = {int(idx): float(corr) for idx, corr in ranked_den_scores}
        denominator_supports: list[tuple[int, ...]] = [(idx,) for idx in ranked_den_terms]
        if int(max_denominator_width) >= 2:
            top = ranked_den_terms[: min(len(ranked_den_terms), 10)]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    denominator_supports.append((int(top[i]), int(top[j])))
        pivot_candidates = (
            _pivot_candidate_indices(
                bank,
                ranked_den_terms,
                exclude=set(),
                max_pivots=int(max_pivots),
            )
            if bool(enable_pivoted)
            else [None]
        )
        for pivot_idx in pivot_candidates:
            candidate_supports = list(denominator_supports)
            if pivot_idx is not None:
                candidate_supports.insert(0, ())
            for denominator_support in candidate_supports:
                if pivot_idx is not None and any(int(i) == int(pivot_idx) for i in denominator_support):
                    continue
                key = (tuple(numerator_support), None if pivot_idx is None else int(pivot_idx), tuple(sorted(denominator_support)))
                if key in seen:
                    continue
                seen.add(key)
                best = None
                best_key = None
                for ridge in ridge_grid:
                    cand = _fit_implicit_rational_support(
                        bank,
                        groups,
                        numerator_support,
                        denominator_support,
                        pivot_support=pivot_idx,
                        denominator_rank_corr=denominator_rank_corr,
                        ridge=float(ridge),
                        complexity_penalty=float(complexity_penalty),
                        ast_serializer=ast_serializer,
                        dtype=dtype,
                    )
                    if cand is None:
                        continue
                    cand_key = (
                        float(cand.score),
                        float(cand.implicit_probe_rms_max),
                        0 if bool(cand.denominator_safety.get("safe", False)) else 1,
                        0 if str(cand.pivot) == "const_one" else 1,
                        float(cand.complexity),
                        cand.numerator_term_ids,
                        cand.pivot,
                        cand.denominator_term_ids,
                    )
                    if best_key is None or cand_key < best_key:
                        best = cand
                        best_key = cand_key
                if best is not None:
                    best.metadata["role_shadow_pivoted"] = bool(str(best.pivot) != "const_one")
                    out.append(best)
    out.sort(
        key=lambda c: (
            float(c.score),
            float(c.implicit_probe_rms_max),
            0 if bool(c.denominator_safety.get("safe", False)) else 1,
            0 if str(c.pivot) == "const_one" else 1,
            float(c.complexity),
            c.numerator_term_ids,
            c.pivot,
            c.denominator_term_ids,
        )
    )
    return out[: max(0, int(max_implicit_candidates))]


def build_de_candidate_eval_report(
    groups: Sequence[DEFeatureGroup],
    *,
    cfg: Any = None,
    order: int | None = None,
    x_axis: int | None = None,
    primary_result: Any = None,
    factorized_result: Any = None,
    factorized_search_result: Any = None,
    max_support_width: int = 5,
    beam_width: int = 32,
    expansions_per_support: int = 16,
    shortlist_topk: int = 16,
    implicit_topk: int = 16,
    enable_role_shadow: bool = True,
    ast_serializer: Any = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Build a diagnostics-only DE candidate-evaluation report."""

    inferred_order = order
    if inferred_order is None:
        for obj in (primary_result, factorized_result, factorized_search_result):
            if obj is not None and getattr(obj, "order", None) is not None:
                inferred_order = int(getattr(obj, "order"))
                break
    if inferred_order is None and cfg is not None:
        orders = tuple(int(o) for o in getattr(cfg, "order_candidates", (1,)) or (1,))
        inferred_order = int(orders[0])
    if inferred_order is None:
        inferred_order = 1

    inferred_x_axis = int(x_axis if x_axis is not None else getattr(cfg, "x_axis", 0) if cfg is not None else 0)
    report: dict[str, Any] = {
        "enabled": True,
        "version": 1,
        "mode": "diagnostics_only",
        "order": int(inferred_order),
        "x_axis": int(inferred_x_axis),
    }
    try:
        bank = build_de_term_bank(
            groups,
            order=int(inferred_order),
            x_axis=int(inferred_x_axis),
            library_results=[primary_result] if primary_result is not None else [],
            factorized_results=[factorized_result] if factorized_result is not None else [],
            factorized_search_results=[factorized_search_result] if factorized_search_result is not None else [],
            enable_role_shadow=bool(enable_role_shadow),
            dtype=dtype,
        )
        supports = assemble_explicit_supports(
            bank,
            list(groups or []),
            max_support_width=int(max_support_width),
            beam_width=int(beam_width),
            expansions_per_support=int(expansions_per_support),
            shortlist_topk=int(shortlist_topk),
            ast_serializer=ast_serializer,
            dtype=dtype,
        )
        implicit_supports = assemble_implicit_rational_supports(
            bank,
            list(groups or []),
            supports,
            max_implicit_candidates=max(int(implicit_topk), 0),
            enable_pivoted=bool(enable_role_shadow),
            ast_serializer=ast_serializer,
            dtype=dtype,
        )
    except Exception as exc:
        report.update({"status": "ERROR", "message": str(exc)})
        return report

    report["status"] = "OK"
    report["term_bank"] = bank.summary()
    report["explicit_supports"] = [cand.to_dict() for cand in supports]
    report["selected_explicit_support"] = supports[0].to_dict() if supports else None
    report["implicit_rational_supports"] = [cand.to_dict() for cand in implicit_supports]
    report["selected_implicit_rational_support"] = implicit_supports[0].to_dict() if implicit_supports else None
    rollout_shortlist = []
    for family, rows in (
        ("explicit", supports),
        ("implicit_rational", implicit_supports),
    ):
        for rank, cand in enumerate(rows):
            payload = cand.validation_candidate
            if not isinstance(payload, Mapping):
                continue
            row = dict(payload)
            row["engine"] = "de_candidate_eval"
            row["candidate_family"] = family
            row["candidate_rank"] = int(len(rollout_shortlist))
            row["source_rank"] = int(rank)
            row["pointwise_score"] = float(cand.score)
            if isinstance(cand, DEImplicitRationalCandidate):
                row["denominator_safety"] = _jsonable(cand.denominator_safety)
                row["probe_rms"] = float(cand.explicit_probe_rms_max)
            else:
                row["probe_rms"] = float(cand.probe_rms_max)
            rollout_shortlist.append(row)
    report["rollout_shortlist"] = _jsonable(rollout_shortlist)
    return report


__all__ = [
    "DETerm",
    "DETermBank",
    "DESupportCandidate",
    "DEImplicitRationalCandidate",
    "RoleShadowOpportunity",
    "assemble_explicit_supports",
    "assemble_implicit_rational_supports",
    "build_de_candidate_eval_report",
    "build_de_term_bank",
]
