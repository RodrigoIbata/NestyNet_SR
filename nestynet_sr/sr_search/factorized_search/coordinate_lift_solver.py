# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Coordinate-lift preview solver for first-class local subproblem specs."""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from .bridge import run_explorer
from .expr_ast import BINARY_OPS, UNARY_OPS, eval_node, node_dims, node_size, node_str
from .expr_mapping import eval_mapping
from .inverse_core import _normalize_inverse_local_score_mode, _weighted_mse_cols
from .inverse_spec_solver import (
    _ScoredLocalCandidate,
    _apply_continuation_frames,
    _candidate_to_preview_row,
    _deserialize_local_problem,
    _proposal_satisfies_hard_constraints,
    _problem_witness_provenance,
    _subproblem_spec_to_local_problem,
)
from .local_teacher_loss import score_local_teacher_prediction_loss
from .lift_route_evidence import build_local_lift_route_context
from .subproblem_active_vars import normalize_active_vars
from .subproblem_spec import (
    deserialize_family_evidence,
    deserialize_subproblem_spec,
    extract_family_regime_metadata,
    serialize_family_evidence,
)
from .subproblem_tests import build_expanded_family_evidence_bundle


def _ensure_col(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _normalize_coordinate_lift_mode(mode: str | None) -> str:
    token = str(mode or "both").strip().lower()
    if token in {"single_index", "invariant", "both"}:
        return token
    return "both"


def _local_problem_from_payload(
    spec_payload: Mapping[str, Any] | None,
) -> tuple[Any, list[dict[str, Any]], Any, str, Any]:
    payload = dict(spec_payload or {})
    subproblem_spec = deserialize_subproblem_spec(payload)
    if subproblem_spec is not None:
        problem, continuation_frames, hole_sub = _subproblem_spec_to_local_problem(subproblem_spec)
        if problem is not None:
            problem_id = str(subproblem_spec.problem_id or "")
            return problem, continuation_frames, hole_sub, problem_id, subproblem_spec
    problem = _deserialize_local_problem(payload.get("problem", {}))
    continuation_frames = [
        dict(frame)
        for frame in list(payload.get("continuation_frames", []) or [])
        if isinstance(frame, Mapping)
    ]
    hole_sub = payload.get("hole_sub", None)
    trace = tuple(str(v) for v in ((payload.get("problem", {}) or {}).get("trace", ()) or ()))
    digest = hashlib.sha1()
    for token in trace:
        digest.update(str(token).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return problem, continuation_frames, hole_sub, digest.hexdigest()[:16], None


def _search_seed(problem_id: str, *, slate_id: str, coord_token: str) -> int:
    token = f"{str(problem_id or '')}:{str(slate_id or '')}:{str(coord_token or '')}"
    digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _mapped_local_mse(
    node,
    *,
    mapping: Mapping[str, Any],
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor | None,
) -> float | None:
    try:
        pred = eval_node(node, x)
        pred = eval_mapping(_ensure_col(pred), mapping)
    except Exception:
        return None
    pred_col = _ensure_col(pred)
    mse = _weighted_mse_cols(_ensure_col(y), pred_col, w)
    return _finite_float(mse)


def _preview_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    probe_mse = _finite_float(row.get("local_probe_mse", None))
    fit_mse = _finite_float(row.get("local_fit_mse", None))
    expr = row.get("expr", None)
    try:
        size = int(node_size(expr)) if isinstance(expr, tuple) else 10**9
    except Exception:
        size = 10**9
    return (
        float("inf") if probe_mse is None else float(probe_mse),
        float("inf") if fit_mse is None else float(fit_mse),
        size,
    )


def _score_mapped_local_candidate(
    node,
    *,
    mapping: Mapping[str, Any],
    problem,
    var_dims,
    nvars: int,
    poly_degree: int,
    generation_kind: str | None,
    witness_loss_enable: bool,
    witness_grad_weight: float,
    witness_d2_weight: float,
    witness_diag_weight: float,
    witness_physics_weight: float,
) -> dict[str, float | None] | None:
    if not _proposal_satisfies_hard_constraints(
        node,
        problem=problem,
        var_dims=var_dims,
        nvars=int(nvars),
        generation_kind=generation_kind,
    ):
        return None
    if bool(witness_loss_enable):
        teacher_loss = score_local_teacher_prediction_loss(
            lambda xx: eval_mapping(_ensure_col(eval_node(node, xx)), mapping),
            x_fit=problem.xf,
            x_probe=problem.xp,
            target_fit=problem.tf,
            target_probe=problem.tp,
            w_fit=problem.wf,
            w_probe=problem.wp,
            target_grad_fit=problem.grad_fit,
            target_grad_probe=problem.grad_probe,
            target_d2_fit=problem.d2_fit,
            target_d2_probe=problem.d2_probe,
            poly_degree=int(poly_degree),
            mode="strict",
            grad_weight=float(witness_grad_weight),
            d2_weight=float(witness_d2_weight),
            diag_weight=float(witness_diag_weight),
            physics_weight=float(witness_physics_weight),
            target_diagnostics=problem.diagnostics,
        )
        if teacher_loss is not None:
            return {
                "local_fit_mse": float(teacher_loss.fit_total),
                "local_probe_mse": float(teacher_loss.probe_total),
                "value_fit_mse": float(teacher_loss.value_fit_loss),
                "value_probe_mse": float(teacher_loss.value_probe_loss),
                "witness_grad_loss": None if teacher_loss.grad_probe_loss is None else float(teacher_loss.grad_probe_loss),
                "witness_d2_loss": None if teacher_loss.d2_probe_loss is None else float(teacher_loss.d2_probe_loss),
                "witness_diag_loss": None if teacher_loss.diag_probe_loss is None else float(teacher_loss.diag_probe_loss),
                "witness_physics_loss": (
                    None if teacher_loss.physics_probe_loss is None else float(teacher_loss.physics_probe_loss)
                ),
                "witness_energy_total": float(teacher_loss.probe_total),
                "witness_fit_jet_source": str(teacher_loss.fit_jet_source or ""),
                "witness_probe_jet_source": str(teacher_loss.probe_jet_source or ""),
                "witness_fit_jet_requested_source": str(teacher_loss.fit_jet_requested_source or ""),
                "witness_probe_jet_requested_source": str(teacher_loss.probe_jet_requested_source or ""),
                "witness_fit_jet_fallback_used": bool(teacher_loss.fit_jet_fallback_used),
                "witness_probe_jet_fallback_used": bool(teacher_loss.probe_jet_fallback_used),
                "witness_numeric_jet_fallback_used": bool(
                    teacher_loss.fit_jet_fallback_used or teacher_loss.probe_jet_fallback_used
                ),
                "witness_exact_jet_used": bool(teacher_loss.exact_jet_used),
                "calibration_gap": max(0.0, float(teacher_loss.probe_total) - float(teacher_loss.value_probe_loss)),
            }
    provenance = _problem_witness_provenance(problem)
    local_fit_mse = _mapped_local_mse(
        node,
        mapping=mapping,
        x=problem.xf,
        y=problem.tf,
        w=problem.wf,
    )
    local_probe_mse = _mapped_local_mse(
        node,
        mapping=mapping,
        x=problem.xp,
        y=problem.tp,
        w=problem.wp,
    )
    if local_fit_mse is None or local_probe_mse is None:
        return None
    return {
        "local_fit_mse": float(local_fit_mse),
        "local_probe_mse": float(local_probe_mse),
        "value_fit_mse": float(local_fit_mse),
        "value_probe_mse": float(local_probe_mse),
        "witness_grad_loss": None,
        "witness_d2_loss": None,
        "witness_diag_loss": None,
        "witness_physics_loss": None,
        "witness_energy_total": float(local_probe_mse),
        "witness_fit_jet_source": str(provenance["witness_fit_jet_source"]),
        "witness_probe_jet_source": str(provenance["witness_probe_jet_source"]),
        "witness_fit_jet_requested_source": str(provenance["witness_fit_jet_requested_source"]),
        "witness_probe_jet_requested_source": str(provenance["witness_probe_jet_requested_source"]),
        "witness_fit_jet_fallback_used": bool(provenance["witness_fit_jet_fallback_used"]),
        "witness_probe_jet_fallback_used": bool(provenance["witness_probe_jet_fallback_used"]),
        "witness_numeric_jet_fallback_used": bool(provenance["witness_numeric_jet_fallback_used"]),
        "witness_exact_jet_used": bool(provenance["witness_exact_jet_used"]),
        "calibration_gap": 0.0,
    }


def _compose_univariate_node(node, *, coord_node):
    if not isinstance(node, tuple) or not node:
        return node
    op = node[0]
    if op == "var":
        local_idx = int(node[1])
        if local_idx != 0:
            raise ValueError(f"coordinate-lift expected univariate node, saw var index {local_idx}")
        return coord_node
    if op in ("const", "hparam"):
        return node
    if op in UNARY_OPS and len(node) >= 2:
        return (op, _compose_univariate_node(node[1], coord_node=coord_node))
    if op in BINARY_OPS and len(node) >= 3:
        return (
            op,
            _compose_univariate_node(node[1], coord_node=coord_node),
            _compose_univariate_node(node[2], coord_node=coord_node),
        )
    return node


def _mean_grad_direction(grad_value: Any, *, active_vars: Sequence[int]) -> torch.Tensor | None:
    if grad_value is None:
        return None
    if not torch.is_tensor(grad_value):
        try:
            grad_value = torch.as_tensor(grad_value, dtype=torch.float64)
        except Exception:
            return None
    else:
        grad_value = grad_value.to(dtype=torch.float64)
    if grad_value.ndim == 1:
        grad_value = grad_value.reshape(1, -1)
    if grad_value.ndim >= 3:
        try:
            grad_value = grad_value.reshape(int(grad_value.shape[0]), -1, int(grad_value.shape[-1])).mean(dim=1)
        except Exception:
            return None
    if grad_value.ndim != 2 or int(grad_value.shape[1]) <= 0:
        return None
    idx = [int(v) for v in tuple(active_vars or ()) if 0 <= int(v) < int(grad_value.shape[1])]
    if not idx:
        return None
    sliced = grad_value[:, idx]
    if not torch.isfinite(sliced).all():
        return None
    direction = sliced.mean(dim=0)
    if int(direction.numel()) <= 0 or float(torch.abs(direction).max().item()) <= 1.0e-12:
        return None
    return direction


def _rounded_dim(dim: Any, *, tol: float = 1.0e-9) -> tuple[float, ...] | None:
    if dim is None:
        return None
    try:
        values = tuple(float(v) for v in tuple(dim))
    except Exception:
        return None
    out: list[float] = []
    for value in values:
        if not math.isfinite(value):
            return None
        if abs(float(value)) <= float(tol):
            value = 0.0
        out.append(round(float(value), 12))
    return tuple(out)


def _node_dim_token(node, *, var_dims) -> tuple[float, ...] | None:
    if var_dims is None:
        return None
    try:
        return _rounded_dim(node_dims(node, var_dims))
    except Exception:
        return None


def _is_dimensionless_dim(dim: Any, *, tol: float = 1.0e-9) -> bool:
    token = _rounded_dim(dim, tol=tol)
    return token is not None and all(abs(float(v)) <= float(tol) for v in token)


def _combine_terms(op: str, terms: Sequence[Any]) -> Any | None:
    nodes = [node for node in list(terms or ()) if isinstance(node, tuple)]
    if not nodes:
        return None
    out = nodes[0]
    for node in nodes[1:]:
        out = (str(op), out, node)
    return out


def _scaled_var_node(coeff: float, var_idx: int) -> Any | None:
    cc = float(coeff)
    if abs(cc) <= 1.0e-12:
        return None
    base = ('var', int(var_idx))
    if abs(cc - 1.0) <= 1.0e-12:
        return base
    if abs(cc + 1.0) <= 1.0e-12:
        return ('neg', base)
    return ('mul', ('const', float(cc)), base)


def _pow_var_node(var_idx: int, exponent: int) -> Any | None:
    exp = int(exponent)
    if exp <= 0:
        return None
    base = ('var', int(var_idx))
    if exp == 1:
        return base
    if exp == 2:
        return ('sqr', base)
    out = base
    for _ in range(1, exp):
        out = ('mul', out, base)
    return out


def _gcd_many(values: Sequence[int]) -> int:
    gg = 0
    for value in tuple(values or ()):
        gg = math.gcd(int(gg), abs(int(value)))
    return int(gg)


def _canonical_exponents(exponents: Sequence[int]) -> tuple[int, ...] | None:
    raw = [int(v) for v in tuple(exponents or ())]
    if not raw or not any(int(v) != 0 for v in raw):
        return None
    gg = _gcd_many(raw)
    if gg > 1:
        raw = [int(v // gg) for v in raw]
    for value in raw:
        if int(value) < 0:
            raw = [-int(v) for v in raw]
            break
        if int(value) > 0:
            break
    return tuple(int(v) for v in raw)


def _build_monomial_node(
    *,
    var_indices: Sequence[int],
    exponents: Sequence[int],
) -> Any | None:
    idx = [int(v) for v in tuple(var_indices or ())]
    exps = [int(v) for v in tuple(exponents or ())]
    if not idx or len(idx) != len(exps):
        return None
    num_terms: list[Any] = []
    den_terms: list[Any] = []
    for var_idx, exponent in zip(idx, exps):
        if int(exponent) == 0:
            continue
        term = _pow_var_node(int(var_idx), abs(int(exponent)))
        if term is None:
            return None
        if int(exponent) > 0:
            num_terms.append(term)
        else:
            den_terms.append(term)
    if not num_terms:
        return None
    numerator = _combine_terms('mul', num_terms)
    if numerator is None:
        return None
    if not den_terms:
        return numerator
    denominator = _combine_terms('mul', den_terms)
    if denominator is None:
        return numerator
    return ('div', numerator, denominator)


def _same_dim_var_groups(
    candidate_vars: Sequence[int],
    *,
    var_dims,
) -> list[tuple[int, ...]]:
    if var_dims is None:
        return []
    groups: dict[tuple[float, ...], list[int]] = {}
    for raw_idx in tuple(candidate_vars or ()):
        var_idx = int(raw_idx)
        if var_idx < 0:
            continue
        dim_token = _node_dim_token(('var', int(var_idx)), var_dims=var_dims)
        if dim_token is None:
            continue
        groups.setdefault(dim_token, []).append(int(var_idx))
    return [tuple(values) for values in groups.values() if len(values) >= 2]


def _quantize_direction_coeff(value: float, *, max_abs: float) -> float:
    scale = float(max_abs)
    if scale <= 1.0e-12 or not math.isfinite(scale):
        return 0.0
    ratio = float(value) / float(scale)
    ar = abs(float(ratio))
    if ar >= 0.80:
        base = 1.0
    elif ar >= 0.45:
        base = 0.5
    elif ar >= 0.20:
        base = 0.25
    else:
        return 0.0
    return float(math.copysign(base, ratio))


def _build_quantized_direction_combo(
    direction: torch.Tensor | None,
    *,
    active_vars: Sequence[int],
    var_dims=None,
    max_terms: int = 4,
) -> tuple[Any, dict[str, Any]] | None:
    if direction is None or not torch.is_tensor(direction):
        return None
    vec = direction.detach().to(dtype=torch.float64).reshape(-1)
    active = [int(v) for v in tuple(active_vars or ())]
    if int(vec.numel()) != len(active) or len(active) < 2:
        return None
    abs_vec = torch.abs(vec)
    if int(abs_vec.numel()) <= 0:
        return None
    ranked = [int(pos) for pos in torch.argsort(abs_vec, descending=True).tolist()]
    if var_dims is not None and ranked:
        anchor_dim = _node_dim_token(('var', int(active[int(ranked[0])])), var_dims=var_dims)
        if anchor_dim is not None:
            ranked = [
                int(pos)
                for pos in ranked
                if _node_dim_token(('var', int(active[int(pos)])), var_dims=var_dims) == anchor_dim
            ]
    if len(ranked) < 2:
        return None
    max_abs = 0.0
    for pos in ranked:
        max_abs = max(max_abs, abs(float(vec[int(pos)].item())))
    if max_abs <= 1.0e-12:
        return None
    used_vars: list[int] = []
    used_coeffs: list[float] = []
    terms: list[Any] = []
    for pos in ranked:
        if len(terms) >= int(max_terms):
            break
        coeff = _quantize_direction_coeff(float(vec[int(pos)].item()), max_abs=max_abs)
        if abs(float(coeff)) <= 1.0e-12:
            continue
        term = _scaled_var_node(float(coeff), int(active[int(pos)]))
        if term is None:
            continue
        used_vars.append(int(active[int(pos)]))
        used_coeffs.append(float(coeff))
        terms.append(term)
    if len(terms) < 2:
        return None
    node = _combine_terms('add', terms)
    if node is None:
        return None
    return node, {
        'vars': [int(v) for v in used_vars],
        'coeffs': [float(v) for v in used_coeffs],
        'support': int(len(terms)),
    }


def _enumerate_dimensionless_group_candidates(
    *,
    candidate_vars: Sequence[int],
    var_dims,
    max_support: int = 4,
    max_abs_power: int = 2,
    max_l1: int = 4,
) -> list[tuple[Any, dict[str, Any]]]:
    if var_dims is None:
        return []
    ranked_vars = [int(v) for v in tuple(candidate_vars or ()) if 0 <= int(v) < int(len(var_dims))]
    ranked_vars = ranked_vars[: min(4, len(ranked_vars))]
    if len(ranked_vars) < 2:
        return []
    out: list[tuple[Any, dict[str, Any]]] = []
    seen: set[str] = set()
    max_support = max(2, min(int(max_support), len(ranked_vars)))
    exponent_choices = tuple(int(v) for v in range(-int(max_abs_power), int(max_abs_power) + 1) if int(v) != 0)
    for support in range(2, max_support + 1):
        for subset in itertools.combinations(ranked_vars, support):
            for raw_exponents in itertools.product(exponent_choices, repeat=int(support)):
                l1 = int(sum(abs(int(v)) for v in raw_exponents))
                if l1 > int(max_l1):
                    continue
                exponents = _canonical_exponents(raw_exponents)
                if exponents is None:
                    continue
                node = _build_monomial_node(var_indices=subset, exponents=exponents)
                if not isinstance(node, tuple):
                    continue
                dim_token = _node_dim_token(node, var_dims=var_dims)
                if not _is_dimensionless_dim(dim_token):
                    continue
                key = str(node_str(node))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    (
                        node,
                        {
                            'vars': [int(v) for v in subset],
                            'exponents': [int(v) for v in exponents],
                            'support': int(sum(1 for v in exponents if int(v) != 0)),
                            'l1': int(l1),
                        },
                    )
                )
    out.sort(
        key=lambda row: (
            int((row[1] or {}).get('support', 99)),
            int((row[1] or {}).get('l1', 99)),
            int(node_size(row[0])),
            str(node_str(row[0])),
        )
    )
    return out


def _enumerate_same_dim_invariant_candidates(
    *,
    candidate_vars: Sequence[int],
    var_dims,
) -> list[tuple[Any, str, dict[str, Any]]]:
    if var_dims is None:
        return []
    out: list[tuple[Any, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for group in _same_dim_var_groups(candidate_vars, var_dims=var_dims):
        limited = tuple(int(v) for v in tuple(group or ())[: min(4, len(group))])
        if len(limited) < 2:
            continue
        max_size = min(3, len(limited))
        for size in range(2, max_size + 1):
            for subset in itertools.combinations(limited, int(size)):
                sq_terms = [('sqr', ('var', int(v))) for v in subset]
                sq_sum = _combine_terms('add', sq_terms)
                if sq_sum is not None:
                    entries = [
                        (sq_sum, 'radial_sq'),
                        (('sqrt', sq_sum), 'radial_norm'),
                    ]
                    for node, kind in entries:
                        key = str(node_str(node))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append((node, str(kind), {'vars': [int(v) for v in subset], 'support': int(size)}))
                if int(size) == 2:
                    prod = ('mul', ('var', int(subset[0])), ('var', int(subset[1])))
                    key = str(node_str(prod))
                    if key not in seen:
                        seen.add(key)
                        out.append((prod, 'pair_product', {'vars': [int(v) for v in subset], 'support': 2}))
                if int(size) >= 3:
                    lin_sum = _combine_terms('add', [('var', int(v)) for v in subset])
                    if lin_sum is not None:
                        key = str(node_str(lin_sum))
                        if key not in seen:
                            seen.add(key)
                            out.append((lin_sum, 'group_sum', {'vars': [int(v) for v in subset], 'support': int(size)}))
    return out


def _coordinate_candidate_family(candidate_kind: str | None) -> str:
    token = str(candidate_kind or '').strip().lower()
    if token.startswith('evidence_seed'):
        return 'evidence_seed'
    if token.startswith('gradient'):
        return 'gradient'
    if token.startswith('dimensionless_group'):
        return 'dimensionless_group'
    if token.startswith('radial'):
        return 'radial'
    if token == 'ratio':
        return 'ratio'
    if token.startswith('group_sum') or token.startswith('pair_sum') or token.startswith('pair_difference'):
        return 'affine'
    if token.startswith('pair_product'):
        return 'product'
    if token in {'dominant_var', 'raw_var'}:
        return 'raw_var'
    return token or 'other'


def _candidate_record(
    node,
    *,
    candidate_kind: str,
    score_hint: float = 0.0,
    coordinate_dim: Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    family = _coordinate_candidate_family(candidate_kind)
    return {
        'node': node,
        'candidate_kind': str(candidate_kind),
        'candidate_family': str(family),
        'score_hint': float(score_hint),
        'node_str': str(node_str(node)),
        'coordinate_dim': None if coordinate_dim is None else [float(v) for v in tuple(coordinate_dim)],
        'dimensionless': bool(coordinate_dim is not None and _is_dimensionless_dim(coordinate_dim)),
        'candidate_metadata': dict(metadata or {}),
    }


def _select_diverse_coordinate_candidates(
    proposals: Sequence[Mapping[str, Any]],
    *,
    topk: int,
) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in list(proposals or ()) if isinstance(row, Mapping)]
    limit = max(1, int(topk))
    if len(ranked) <= limit:
        return ranked
    preferred_families = (
        'evidence_seed',
        'gradient',
        'dimensionless_group',
        'radial',
        'ratio',
        'affine',
        'product',
        'raw_var',
        'other',
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in ranked:
        family = str(row.get('candidate_family', '') or _coordinate_candidate_family(row.get('candidate_kind', '')) or 'other')
        by_family.setdefault(family, []).append(dict(row))
    for family in preferred_families:
        bucket = by_family.get(str(family), [])
        if not bucket or len(selected) >= limit:
            continue
        row = dict(bucket[0])
        key = str(row.get('node_str', '') or '')
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
    if len(selected) < limit:
        for row in ranked:
            key = str(row.get('node_str', '') or '')
            if key in seen:
                continue
            selected.append(dict(row))
            seen.add(key)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _build_coordinate_candidates(
    *,
    problem,
    active_vars: Sequence[int],
    coordinate_mode: str,
    subproblem_spec,
    lift_route_context: Mapping[str, Any] | None = None,
    var_dims=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nvars = int(problem.xf.shape[1]) if getattr(problem.xf, 'ndim', 0) >= 2 else 0
    active = normalize_active_vars(active_vars, nvars=nvars)
    if not active and nvars > 0:
        active = tuple(range(int(nvars)))
    mode_name = _normalize_coordinate_lift_mode(coordinate_mode)
    diagnostics: dict[str, Any] = {
        'coordinate_mode': str(mode_name),
        'active_vars': [int(v) for v in active],
    }
    route_context = dict(lift_route_context or {})
    serialized_bundle = dict(route_context.get('expanded_family_evidence', {}) or {})
    if not serialized_bundle:
        evidence_bundle = build_expanded_family_evidence_bundle(
            x_fit=problem.xf,
            t_fit=problem.tf,
            x_probe=problem.xp,
            t_probe=problem.tp,
            grad_fit=problem.grad_fit,
            grad_probe=problem.grad_probe,
            d2_fit=problem.d2_fit,
            d2_probe=problem.d2_probe,
            target_dim=problem.target_dim,
            active_vars=active,
            wrappers_left=int(getattr(problem, 'wrappers_left', 0) or 0),
            recursion_level=int(getattr(problem, 'recursion_level', 0) or 0),
            direction=(
                ''
                if subproblem_spec is None
                else str(getattr(subproblem_spec, 'direction', '') or '')
            ),
            target_mode=(
                ''
                if subproblem_spec is None
                else str(getattr(subproblem_spec, 'target_mode', '') or '')
            ),
            target_mapping_kind=(
                ''
                if subproblem_spec is None
                else str(getattr(subproblem_spec, 'target_mapping_kind', '') or '')
            ),
            regime_metadata=extract_family_regime_metadata(
                dict(getattr(subproblem_spec, 'metadata', {}) or {}),
                dict(getattr(getattr(subproblem_spec, 'witness', None), 'diagnostics', {}) or {}),
            ),
        )
        serialized_bundle = {
            str(name): serialize_family_evidence(evidence)
            for name, evidence in sorted(dict(evidence_bundle or {}).items())
        }
    diagnostics['expanded_family_evidence'] = serialized_bundle
    diagnostics['lift_route_context'] = route_context
    route_signal = dict(route_context.get('coordinate_lift', {}) or {})
    diagnostics['coordinate_route_status'] = str(route_signal.get('status', '') or '')
    diagnostics['coordinate_route_score'] = _finite_float(route_signal.get('score', None))
    diagnostics['coordinate_route_preferred'] = bool(route_signal.get('preferred', False))
    diagnostics['coordinate_route_reason_family'] = str(route_signal.get('reason_family', '') or '')
    top_var = None
    low_rank_hard = dict(
        (diagnostics['expanded_family_evidence'].get('low_rank_dependence', {}) or {}).get('hard_constraints', {}) or {}
    )
    low_rank_score = _finite_float(
        ((diagnostics['expanded_family_evidence'].get('low_rank_dependence', {}) or {}).get('family_scores', {}) or {}).get(
            'low_rank_dependence',
            None,
        )
    )
    try:
        top_var = int(low_rank_hard.get('top_var', -1))
    except Exception:
        top_var = None
    coordinate_payload = diagnostics['expanded_family_evidence'].get('coordinate_invariant', {}) or {}
    coordinate_evidence = deserialize_family_evidence(coordinate_payload)
    coordinate_hard = dict((coordinate_payload or {}).get('hard_constraints', {}) or {})
    coordinate_score = _finite_float(
        ((coordinate_payload or {}).get('family_scores', {}) or {}).get('coordinate_invariant', None)
    )
    preferred_vars = [
        int(v)
        for v in list(coordinate_hard.get('coordinate_vars', active) or active)
        if 0 <= int(v) < int(nvars)
    ]
    if not preferred_vars:
        preferred_vars = [int(v) for v in active]
    ordered_vars: list[int] = []
    for raw_idx in list(preferred_vars) + [int(v) for v in active]:
        idx = int(raw_idx)
        if idx not in ordered_vars and 0 <= idx < int(nvars):
            ordered_vars.append(int(idx))
    pair_vars = tuple(ordered_vars) if len(ordered_vars) >= 2 else tuple(active)
    branch_score = _finite_float(
        ((diagnostics['expanded_family_evidence'].get('branch_structure', {}) or {}).get('family_scores', {}) or {}).get(
            'branch_structure',
            None,
        )
    )

    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    family_counts: dict[str, int] = {}
    candidate_counts_by_kind: dict[str, int] = {}

    def _add(node, *, candidate_kind: str, score_hint: float = 0.0, metadata: Mapping[str, Any] | None = None) -> None:
        if not isinstance(node, tuple):
            return
        dim_token = _node_dim_token(node, var_dims=var_dims) if var_dims is not None else None
        if var_dims is not None and dim_token is None:
            return
        key = str(node_str(node))
        if key in seen:
            return
        seen.add(key)
        record = _candidate_record(
            node,
            candidate_kind=str(candidate_kind),
            score_hint=float(score_hint),
            coordinate_dim=dim_token,
            metadata=metadata,
        )
        proposals.append(record)
        family = str(record.get('candidate_family', 'other') or 'other')
        kind = str(record.get('candidate_kind', '') or '')
        family_counts[family] = int(family_counts.get(family, 0)) + 1
        candidate_counts_by_kind[kind] = int(candidate_counts_by_kind.get(kind, 0)) + 1

    if mode_name in {'single_index', 'both'} and coordinate_evidence is not None:
        for seed_rank, node in enumerate(tuple(coordinate_evidence.seed_nodes or ())):
            _add(
                node,
                candidate_kind='evidence_seed',
                score_hint=3.0 + float(coordinate_score or 0.0) - 0.05 * float(seed_rank),
                metadata={'seed_rank': int(seed_rank)},
            )

    if mode_name in {'single_index', 'both'} and len(active) >= 2:
        grad_direction = None
        witness = getattr(subproblem_spec, 'witness', None) if subproblem_spec is not None else None
        if witness is not None:
            grad_direction = _mean_grad_direction(getattr(witness, 'grad_fit', None), active_vars=active)
            if grad_direction is None:
                grad_direction = _mean_grad_direction(getattr(witness, 'grad_probe', None), active_vars=active)
        if grad_direction is not None and int(grad_direction.numel()) >= 2:
            combo_payload = _build_quantized_direction_combo(
                grad_direction,
                active_vars=active,
                var_dims=var_dims,
                max_terms=4,
            )
            if combo_payload is not None:
                combo_node, combo_meta = combo_payload
                _add(
                    combo_node,
                    candidate_kind='gradient_quantized_combo',
                    score_hint=2.45 + float(coordinate_score or 0.0) + 0.05 * float(combo_meta.get('support', 0) or 0),
                    metadata=combo_meta,
                )
                diagnostics['gradient_coordinate_vars'] = [int(v) for v in list(combo_meta.get('vars', []) or [])]
                diagnostics['gradient_coordinate_coeffs'] = [float(v) for v in list(combo_meta.get('coeffs', []) or [])]
            abs_dir = torch.abs(grad_direction)
            ranked_local = torch.argsort(abs_dir, descending=True).tolist()
            if len(ranked_local) >= 2:
                i = int(active[int(ranked_local[0])])
                j = int(active[int(ranked_local[1])])
                sign_i = 1.0 if float(grad_direction[int(ranked_local[0])].item()) >= 0.0 else -1.0
                sign_j = 1.0 if float(grad_direction[int(ranked_local[1])].item()) >= 0.0 else -1.0
                combo = ('add', ('var', i), ('var', j)) if sign_i * sign_j >= 0.0 else ('sub', ('var', i), ('var', j))
                _add(
                    combo,
                    candidate_kind='gradient_linear_combo',
                    score_hint=2.0 + float(coordinate_score or 0.0),
                    metadata={'vars': [int(i), int(j)], 'signs': [float(sign_i), float(sign_j)]},
                )
                diagnostics['gradient_coordinate_pair'] = [int(i), int(j)]
                diagnostics['gradient_coordinate_signs'] = [float(sign_i), float(sign_j)]

    if mode_name in {'invariant', 'both'} and len(pair_vars) >= 2:
        dimless_nodes = _enumerate_dimensionless_group_candidates(
            candidate_vars=pair_vars,
            var_dims=var_dims,
            max_support=4,
            max_abs_power=2,
            max_l1=4,
        )
        diagnostics['dimensionless_group_count_raw'] = int(len(dimless_nodes))
        diagnostics['dimensionless_group_node_strs'] = [str(node_str(node)) for node, _meta in dimless_nodes[:16]]
        for rank, (node, meta) in enumerate(dimless_nodes[:16]):
            support = int((meta or {}).get('support', 2) or 2)
            l1 = int((meta or {}).get('l1', support) or support)
            _add(
                node,
                candidate_kind='dimensionless_group',
                score_hint=2.35 + float(coordinate_score or 0.0) + 0.08 * float(support - 2) - 0.06 * float(max(0, l1 - 2)) - 0.02 * float(rank),
                metadata=meta,
            )

        same_dim_nodes = _enumerate_same_dim_invariant_candidates(candidate_vars=pair_vars, var_dims=var_dims)
        diagnostics['same_dim_invariant_count_raw'] = int(len(same_dim_nodes))
        for rank, (node, kind, meta) in enumerate(same_dim_nodes[:16]):
            support = int((meta or {}).get('support', 2) or 2)
            base = 2.05 if str(kind).startswith('radial') else 1.20
            if str(kind) == 'group_sum':
                base = 1.35
            _add(
                node,
                candidate_kind=str(kind),
                score_hint=float(base) + 0.50 * float(coordinate_score or 0.0) + 0.05 * float(support - 2) - 0.02 * float(rank),
                metadata=meta,
            )

    if mode_name in {'invariant', 'both'} and len(pair_vars) >= 2:
        for pos_i, i in enumerate(pair_vars):
            for j in pair_vars[pos_i + 1 :]:
                ratio_hint = 1.50 + 0.50 * float(branch_score or 0.0)
                ratio_meta = {'vars': [int(i), int(j)]}
                _add(('div', ('var', int(i)), ('var', int(j))), candidate_kind='ratio', score_hint=ratio_hint, metadata=ratio_meta)
                _add(('div', ('var', int(j)), ('var', int(i))), candidate_kind='ratio', score_hint=ratio_hint, metadata={'vars': [int(j), int(i)]})

    if len(pair_vars) >= 2:
        for pos_i, i in enumerate(pair_vars):
            for j in pair_vars[pos_i + 1 :]:
                _add(
                    ('mul', ('var', int(i)), ('var', int(j))),
                    candidate_kind='pair_product',
                    score_hint=1.15 + 0.25 * float(coordinate_score or 0.0),
                    metadata={'vars': [int(i), int(j)]},
                )

    if mode_name in {'single_index', 'both'} and len(pair_vars) >= 2:
        for pos_i, i in enumerate(pair_vars):
            for j in pair_vars[pos_i + 1 :]:
                pair_meta = {'vars': [int(i), int(j)]}
                _add(('add', ('var', int(i)), ('var', int(j))), candidate_kind='pair_sum', score_hint=1.25 + 0.5 * float(coordinate_score or 0.0), metadata=pair_meta)
                _add(('sub', ('var', int(i)), ('var', int(j))), candidate_kind='pair_difference', score_hint=1.0 + 0.5 * float(coordinate_score or 0.0), metadata=pair_meta)
                _add(('sub', ('var', int(j)), ('var', int(i))), candidate_kind='pair_difference', score_hint=1.0 + 0.5 * float(coordinate_score or 0.0), metadata={'vars': [int(j), int(i)]})

    if mode_name in {'single_index', 'both'}:
        if top_var is not None and top_var in active:
            _add(('var', int(top_var)), candidate_kind='dominant_var', score_hint=1.25 + float(low_rank_score or 0.0), metadata={'var': int(top_var)})
        for i in active:
            _add(('var', int(i)), candidate_kind='raw_var', score_hint=0.75 + 0.25 * float(low_rank_score or 0.0), metadata={'var': int(i)})

    proposals.sort(
        key=lambda row: (
            -float(row.get('score_hint', 0.0) or 0.0),
            int(node_size(row.get('node', ('const', 0.0)))),
            str(row.get('node_str', '') or ''),
        )
    )
    diagnostics['candidate_count_raw'] = int(len(proposals))
    diagnostics['candidate_node_strs'] = [str(row.get('node_str', '')) for row in proposals]
    diagnostics['candidate_family_counts'] = {str(k): int(v) for k, v in sorted(family_counts.items())}
    diagnostics['candidate_kind_counts'] = {str(k): int(v) for k, v in sorted(candidate_counts_by_kind.items())}
    diagnostics['dimensionally_filtered'] = bool(var_dims is not None)
    return proposals, diagnostics


@torch.no_grad()
def solve_local_coordinate_lift_preview_rows(
    *,
    parent_node,
    spec_payload: Mapping[str, Any],
    path: Sequence[int],
    target_mode: str,
    target_mapping_kind: str,
    beam_rank: int,
    slate_id: str,
    path_gain: float,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    local_score_mode: str = "affine",
    preview_topk: int = 4,
    max_subtree_depth: int | None = None,
    coordinate_topk: int = 4,
    coordinate_mode: str = "both",
    lift_route_context: Mapping[str, Any] | None = None,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 0.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    mode_name = _normalize_inverse_local_score_mode(local_score_mode, default="affine")
    hole_path = tuple(int(v) for v in (path or ()))
    problem, continuation_frames, hole_sub, problem_id, subproblem_spec = _local_problem_from_payload(spec_payload)
    coord_mode_name = _normalize_coordinate_lift_mode(coordinate_mode)
    route_context = dict(lift_route_context or {})
    if not route_context:
        route_context = build_local_lift_route_context(spec_payload)
    route_signal = dict(route_context.get("coordinate_lift", {}) or {})
    solver_meta: dict[str, Any] = {
        "proposal_family": "coordinate_lift",
        "generation_source": "coordinate_lift",
        "path": [int(v) for v in hole_path],
        "target_mode": str(target_mode or ""),
        "target_mapping_kind": str(target_mapping_kind or ""),
        "local_score_mode": str(mode_name),
        "preview_count": 0,
        "candidate_count_scored": 0,
        "child_spec_states": [],
        "child_spec_state_count": 0,
        "wall_seconds": 0.0,
        "status": "started",
        "coordinate_lift_mode": str(coord_mode_name),
        "coordinate_topk": int(max(1, int(coordinate_topk))),
        "route_trigger_status": str(route_signal.get("status", "") or ""),
        "route_trigger_score": _finite_float(route_signal.get("score", None)),
        "route_trigger_preferred": bool(route_signal.get("preferred", False)),
        "route_reason_family": str(route_signal.get("reason_family", "") or ""),
    }
    if problem is None:
        solver_meta["status"] = "missing_spec_payload"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    if int(problem.xf.shape[0]) < 4 or int(problem.xp.shape[0]) < 4:
        solver_meta["status"] = "insufficient_points"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    active_vars = ()
    if subproblem_spec is not None:
        active_vars = normalize_active_vars(
            tuple(subproblem_spec.active_vars or ()),
            nvars=int(problem.xf.shape[1]) if getattr(problem.xf, "ndim", 0) >= 2 else int(nvars),
        )
    proposals, diagnostics = _build_coordinate_candidates(
        problem=problem,
        active_vars=active_vars,
        coordinate_mode=coord_mode_name,
        subproblem_spec=subproblem_spec,
        lift_route_context=route_context,
        var_dims=var_dims,
    )
    coord_limit = max(1, int(coordinate_topk))
    selected_proposals = _select_diverse_coordinate_candidates(proposals, topk=coord_limit)
    solver_meta["coordinate_candidates_raw"] = int(diagnostics.get("candidate_count_raw", 0) or 0)
    solver_meta["coordinate_candidates_selected"] = int(len(selected_proposals))
    solver_meta["coordinate_candidates_tried"] = int(len(selected_proposals))
    solver_meta["coordinate_candidate_node_strs"] = [str(row.get("node_str", "")) for row in selected_proposals]
    solver_meta["coordinate_candidate_families"] = [str(row.get("candidate_family", "")) for row in selected_proposals]
    solver_meta["coordinate_candidate_family_counts"] = dict(diagnostics.get("candidate_family_counts", {}) or {})
    solver_meta["coordinate_candidate_kind_counts"] = dict(diagnostics.get("candidate_kind_counts", {}) or {})
    solver_meta["coordinate_active_vars"] = [int(v) for v in tuple(active_vars or ())]
    solver_meta["expanded_family_evidence"] = dict(diagnostics.get("expanded_family_evidence", {}) or {})
    solver_meta["coordinate_dimensionally_filtered"] = bool(diagnostics.get("dimensionally_filtered", False))
    proposals = selected_proposals
    if not proposals:
        solver_meta["status"] = "no_coordinate_candidates"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    local_max_depth = int(max_subtree_depth if max_subtree_depth is not None else max_depth)
    route_preview_topk = max(1, int(preview_topk))
    per_coord_topk = max(1, min(route_preview_topk, 3))
    search_n_iter = max(384, min(2500, 384 * max(1, per_coord_topk)))
    preview_rows: list[dict[str, Any]] = []

    for coord_rank, proposal in enumerate(proposals):
        coord_node = proposal.get("node", None)
        if not isinstance(coord_node, tuple):
            continue
        try:
            coord_fit = _ensure_col(eval_node(coord_node, problem.xf))
            coord_probe = _ensure_col(eval_node(coord_node, problem.xp))
        except Exception:
            continue
        if (not torch.isfinite(coord_fit).all()) or (not torch.isfinite(coord_probe).all()):
            continue
        if float(coord_fit.std(unbiased=False).item()) <= 1.0e-12:
            continue
        coord_dim = None
        if var_dims is not None:
            try:
                coord_dim = node_dims(coord_node, var_dims)
            except Exception:
                coord_dim = None
        search_seed = _search_seed(problem_id, slate_id=str(slate_id), coord_token=str(node_str(coord_node)))
        raw_results = list(
            run_explorer(
                nvars=1,
                n_iter=int(search_n_iter),
                max_depth=int(max(1, local_max_depth)),
                poly_degree=int(poly_degree),
                seed=int(search_seed),
                var_dims=None if coord_dim is None else [coord_dim],
                y_dims=problem.target_dim,
                return_topk=int(per_coord_topk),
                dtype=problem.xf.dtype,
                x_fit_data=coord_fit,
                y_fit_data=problem.tf,
                x_probe_data=coord_probe,
                y_probe_data=problem.tp,
                simplify_skeletons=False,
                print_every=0,
                verbose=False,
            ) or []
        )
        for result_rank, result in enumerate(raw_results):
            if not isinstance(result, Mapping):
                continue
            node = result.get("toy_ast", None)
            mapping = dict(result.get("mapping", {}) or {})
            if not isinstance(node, tuple) or not node or not mapping:
                continue
            try:
                global_node = _compose_univariate_node(node, coord_node=coord_node)
            except Exception:
                continue
            score = _score_mapped_local_candidate(
                global_node,
                mapping=mapping,
                problem=problem,
                var_dims=var_dims,
                nvars=int(nvars),
                poly_degree=int(poly_degree),
                generation_kind="coordinate_lift",
                witness_loss_enable=bool(witness_loss_enable),
                witness_grad_weight=float(witness_grad_weight),
                witness_d2_weight=float(witness_d2_weight),
                witness_diag_weight=float(witness_diag_weight),
                witness_physics_weight=float(witness_physics_weight),
            )
            if score is None:
                continue
            cand = _ScoredLocalCandidate(
                node=global_node,
                local_probe_mse=float(score["local_probe_mse"]),
                local_fit_mse=float(score["local_fit_mse"]),
                source="coordinate_lift",
                generation_kind="coordinate_lift",
                recursion_depth=int(problem.recursion_level),
                confidence=float(problem.confidence),
                valid_frac=float(problem.valid_frac),
                trace=tuple(problem.trace or ()),
                family="coordinate_lift",
                payload={
                    "mapping": mapping,
                    "mse_raw": _finite_float(result.get("mse_raw", None)),
                    "mse_eff": _finite_float(result.get("mse_eff", None)),
                    "coord_rank": int(coord_rank),
                    "coord_kind": str(proposal.get("candidate_kind", "") or ""),
                    "coord_node": coord_node,
                    "coord_node_str": str(proposal.get("node_str", "") or ""),
                    "result_rank": int(result_rank),
                },
                surrogate_probe_mse=_finite_float(result.get("mse_eff", score["local_probe_mse"])),
                surrogate_fit_mse=float(score["value_fit_mse"]),
                value_probe_mse=float(score["value_probe_mse"]),
                value_fit_mse=float(score["value_fit_mse"]),
                witness_value_loss=float(score["value_probe_mse"]),
                witness_grad_loss=None if score["witness_grad_loss"] is None else float(score["witness_grad_loss"]),
                witness_d2_loss=None if score["witness_d2_loss"] is None else float(score["witness_d2_loss"]),
                witness_diag_loss=None if score["witness_diag_loss"] is None else float(score["witness_diag_loss"]),
                witness_physics_loss=(
                    None if score["witness_physics_loss"] is None else float(score["witness_physics_loss"])
                ),
                witness_energy_total=None if score["witness_energy_total"] is None else float(score["witness_energy_total"]),
                witness_fit_jet_source=str(score.get("witness_fit_jet_source", "") or ""),
                witness_probe_jet_source=str(score.get("witness_probe_jet_source", "") or ""),
                witness_fit_jet_requested_source=str(score.get("witness_fit_jet_requested_source", "") or ""),
                witness_probe_jet_requested_source=str(score.get("witness_probe_jet_requested_source", "") or ""),
                witness_fit_jet_fallback_used=bool(score.get("witness_fit_jet_fallback_used", False)),
                witness_probe_jet_fallback_used=bool(score.get("witness_probe_jet_fallback_used", False)),
                witness_numeric_jet_fallback_used=bool(score.get("witness_numeric_jet_fallback_used", False)),
                witness_exact_jet_used=bool(score.get("witness_exact_jet_used", False)),
                calibration_gap=float(score["calibration_gap"]),
            )
            try:
                wrapped_node = _apply_continuation_frames(cand.node, continuation_frames)
            except Exception:
                continue
            wrapped_cand = replace(
                cand,
                node=wrapped_node,
                generation_kind=f"followup:{str(cand.generation_kind)}",
            )
            row = _candidate_to_preview_row(
                wrapped_cand,
                parent_node=parent_node,
                beam_state={
                    "sub": hole_sub,
                    "target_mode": str(target_mode or ""),
                    "target_mapping_kind": str(target_mapping_kind or ""),
                    "path_gain": float(path_gain),
                    "poly_degree": int(poly_degree),
                },
                beam_rank=int(beam_rank),
                slate_id=str(slate_id),
                path=hole_path,
                xf=problem.xf,
                tf=problem.tf,
                xp=problem.xp,
                tp=problem.tp,
                max_depth=int(max_depth),
                var_dims=var_dims,
                local_score_mode=str(mode_name),
            )
            if row is None:
                continue
            row["proposal_family"] = "coordinate_lift"
            row["generation_source"] = "coordinate_lift"
            row["tuple_provenance"] = "coordinate_lift"
            row["coordinate_lift_coord_rank"] = int(coord_rank)
            row["coordinate_lift_coord_kind"] = str(proposal.get("candidate_kind", "") or "")
            row["coordinate_lift_coord_family"] = str(proposal.get("candidate_family", "") or "")
            row["coordinate_lift_coord_expr"] = coord_node
            row["coordinate_lift_coord_expr_str"] = str(proposal.get("node_str", "") or "")
            row["coordinate_lift_coord_dimensionless"] = bool(proposal.get("dimensionless", False))
            coord_dim = proposal.get("coordinate_dim", None)
            row["coordinate_lift_coord_dim"] = None if coord_dim is None else list(coord_dim)
            coord_meta = dict(proposal.get("candidate_metadata", {}) or {})
            if coord_meta:
                row["coordinate_lift_coord_metadata"] = coord_meta
            row["coordinate_lift_result_rank"] = int(result_rank)
            row["coordinate_lift_mse_eff"] = _finite_float(result.get("mse_eff", None))
            row["coordinate_lift_mse_raw"] = _finite_float(result.get("mse_raw", None))
            row["coordinate_lift_route_status"] = str(route_signal.get("status", "") or "")
            row["coordinate_lift_route_score"] = _finite_float(route_signal.get("score", None))
            row["coordinate_lift_route_reason_family"] = str(route_signal.get("reason_family", "") or "")
            preview_rows.append(row)

    preview_rows.sort(key=_preview_sort_key)
    preview_rows = preview_rows[:route_preview_topk]
    for local_rank, row in enumerate(preview_rows):
        row["local_rank"] = int(local_rank)
        row["local_candidate_count"] = int(len(preview_rows))

    solver_meta["candidate_count_scored"] = int(len(preview_rows))
    solver_meta["preview_count"] = int(len(preview_rows))
    solver_meta["status"] = "ok" if preview_rows else "no_coordinate_lift_candidates"
    solver_meta["wall_seconds"] = float(time.perf_counter() - started)
    return {
        "rows": preview_rows,
        "solver_meta": solver_meta,
    }


__all__ = ["solve_local_coordinate_lift_preview_rows"]
