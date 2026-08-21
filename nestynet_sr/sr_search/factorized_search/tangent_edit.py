# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Residual-guided tangent edit proposals for local subproblem follow-ups."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from .expr_ast import dim_round, dims_eq, eval_node, get_at, node_dims, node_size, node_str, replace_at
from .inverse_core import _normalize_inverse_local_score_mode
from .inverse_spec_solver import (
    _ScoredLocalCandidate,
    _SolverContext,
    _apply_continuation_frames,
    _candidate_to_preview_row,
    _dedup_scored_candidates,
    _deserialize_local_problem,
    _score_node_against_problem,
    _simplify_node,
    _subproblem_spec_to_local_problem,
)
from .subproblem_spec import SubproblemSpec, deserialize_subproblem_spec


def _ensure_matrix(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x


def _ensure_col(x: torch.Tensor) -> torch.Tensor:
    xx = _ensure_matrix(x)
    if xx.ndim != 2 or int(xx.shape[1]) != 1:
        raise ValueError(f"expected [N] or [N,1] tensor, got shape={tuple(xx.shape)}")
    return xx


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _local_problem_from_payload(
    spec_payload: Mapping[str, Any] | None,
) -> tuple[Any, list[dict[str, Any]], Any, SubproblemSpec | None]:
    payload = dict(spec_payload or {})
    subproblem_spec = deserialize_subproblem_spec(payload)
    if subproblem_spec is not None:
        problem, continuation_frames, hole_sub = _subproblem_spec_to_local_problem(subproblem_spec)
        if problem is not None:
            return problem, continuation_frames, hole_sub, subproblem_spec
    problem = _deserialize_local_problem(payload.get("problem", {}))
    continuation_frames = [
        dict(frame)
        for frame in list(payload.get("continuation_frames", []) or [])
        if isinstance(frame, Mapping)
    ]
    hole_sub = payload.get("hole_sub", None)
    return problem, continuation_frames, hole_sub, None


def _normalize_value_tensor(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    else:
        value = value.to(dtype=x.dtype, device=x.device)
    if value.ndim == 0:
        value = value.reshape(1, 1).expand(int(x.shape[0]), 1)
    elif value.ndim == 1:
        value = value.reshape(-1, 1)
    elif value.ndim > 2:
        try:
            value = value.reshape(int(value.shape[0]), -1).mean(dim=1, keepdim=True)
        except Exception:
            return None
    if value.ndim != 2 or int(value.shape[1]) != 1 or int(value.shape[0]) != int(x.shape[0]):
        return None
    if not torch.isfinite(value).all():
        return None
    return value


def _normalize_grad_tensor(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    else:
        value = value.to(dtype=x.dtype, device=x.device)
    if value.ndim == 2 and tuple(value.shape) == (int(x.shape[0]), int(x.shape[1])):
        out = value
    elif value.ndim >= 3 and int(value.shape[0]) == int(x.shape[0]) and int(value.shape[-1]) == int(x.shape[1]):
        try:
            out = value.reshape(int(x.shape[0]), -1, int(x.shape[1])).mean(dim=1)
        except Exception:
            return None
    else:
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _capture_node_value_grad(
    node,
    x: torch.Tensor,
    *,
    capture_gradients: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    xx = _ensure_matrix(x)
    if not capture_gradients:
        try:
            value = _normalize_value_tensor(eval_node(node, xx), x=xx)
        except Exception:
            return None, None
        return value, None
    try:
        with torch.enable_grad():
            x_req = xx.detach().clone().requires_grad_(True)
            value = _normalize_value_tensor(eval_node(node, x_req), x=x_req)
            if value is None:
                return None, None
            grad = torch.autograd.grad(
                value.sum(),
                x_req,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(x_req)
            else:
                grad = grad.detach()
            return value.detach(), grad
    except Exception:
        return None, None


def _subset_rows(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    grad: torch.Tensor | None = None,
    max_rows: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    xx = _ensure_matrix(x)
    yy = _ensure_col(y)
    if int(xx.shape[0]) <= int(max_rows):
        return xx, yy, grad
    idx = torch.linspace(
        0,
        int(xx.shape[0]) - 1,
        steps=int(max_rows),
        dtype=torch.float64,
        device=xx.device,
    ).round().to(dtype=torch.long)
    idx = torch.unique(idx, sorted=True)
    gg = None
    if grad is not None:
        gg = grad.index_select(0, idx)
    return xx.index_select(0, idx), yy.index_select(0, idx), gg


def _alignment_score(target: torch.Tensor, delta: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    tt = target.reshape(-1)
    dd = delta.reshape(-1)
    if tt.numel() == 0 or dd.numel() == 0 or int(tt.numel()) != int(dd.numel()):
        return 0.0
    tt = torch.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0)
    dd = torch.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
    t_norm = torch.sqrt(torch.clamp(torch.mean(tt * tt), min=0.0))
    d_norm = torch.sqrt(torch.clamp(torch.mean(dd * dd), min=0.0))
    denom = torch.clamp(t_norm * d_norm, min=float(eps))
    if float(d_norm.item()) <= float(eps):
        return 0.0
    score = torch.mean(tt * dd) / denom
    try:
        return float(score.item())
    except Exception:
        return 0.0


def _dim_add(d1: Any, d2: Any) -> Any:
    if d1 is None or d2 is None:
        return None
    return dim_round(tuple(float(a) + float(b) for a, b in zip(d1, d2)))


def _dim_sub(d1: Any, d2: Any) -> Any:
    if d1 is None or d2 is None:
        return None
    return dim_round(tuple(float(a) - float(b) for a, b in zip(d1, d2)))


def _dim_scale(d: Any, factor: float) -> Any:
    if d is None:
        return None
    return dim_round(tuple(float(factor) * float(v) for v in d))


def _zero_dim_like(d: Any) -> Any:
    if d is None:
        return None
    return dim_round(tuple(0.0 for _ in d))


def _node_matches_target_dim(node, *, target_dim: Any, var_dims) -> bool:
    if var_dims is None or target_dim is None:
        return True
    try:
        nd = node_dims(node, var_dims)
    except Exception:
        return False
    return bool(nd is not None and dims_eq(nd, target_dim))



def _edit_dim_allowed(
    *,
    edit_kind: str,
    base_dim: Any,
    term_dim: Any = None,
    target_dim: Any = None,
) -> bool:
    if target_dim is None or base_dim is None:
        return True
    candidate_dim = None
    if edit_kind == "wrap:neg":
        candidate_dim = base_dim
    elif edit_kind in ("wrap:sin", "wrap:cos", "wrap:exp", "wrap:log"):
        zero_dim = _zero_dim_like(base_dim)
        if zero_dim is None or not dims_eq(base_dim, zero_dim):
            return False
        candidate_dim = zero_dim
    elif edit_kind == "wrap:sqrt":
        candidate_dim = _dim_scale(base_dim, 0.5)
    elif edit_kind == "wrap:sqr":
        candidate_dim = _dim_scale(base_dim, 2.0)
    elif edit_kind in ("combine:add", "combine:sub"):
        if term_dim is None or not dims_eq(term_dim, base_dim):
            return False
        candidate_dim = base_dim
    elif edit_kind == "combine:mul":
        candidate_dim = _dim_add(base_dim, term_dim)
    elif edit_kind == "combine:div":
        candidate_dim = _dim_sub(base_dim, term_dim)
    elif edit_kind == "replace":
        candidate_dim = term_dim
    if candidate_dim is None:
        return False
    return bool(dims_eq(candidate_dim, target_dim))


def _const_bias_allowed(*, target_dim: Any, var_dims) -> bool:
    if target_dim is None or var_dims is None:
        return True
    zero_dim = _zero_dim_like(target_dim)
    return bool(zero_dim is not None and dims_eq(target_dim, zero_dim))


def _seed_nodes(
    *,
    active_vars: Sequence[int],
    nvars: int,
    pool_nodes: Sequence[Any] | None,
) -> list[tuple]:
    out: list[tuple] = []
    seen: set[str] = set()

    def _add(node) -> None:
        if not isinstance(node, tuple) or not node:
            return
        simp = _simplify_node(node)
        key = node_str(simp)
        if key in seen:
            return
        seen.add(key)
        out.append(simp)

    for value in (1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5):
        _add(("const", float(value)))
    var_ids = [int(v) for v in tuple(active_vars or ()) if int(v) >= 0]
    if not var_ids:
        var_ids = list(range(max(0, int(nvars))))
    for var_idx in var_ids[: max(1, min(4, len(var_ids)))]:
        var_node = ("var", int(var_idx))
        _add(var_node)
        _add(("sqr", var_node))
        _add(("div", ("const", 1.0), var_node))
        _add(("sin", var_node))
        _add(("cos", var_node))
    for node in list(pool_nodes or ())[:12]:
        _add(node)
    return out


def _collect_const_leaf_paths(node, *, path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    if not isinstance(node, tuple) or not node:
        return []
    op = str(node[0])
    if op == "const":
        return [tuple(path)]
    if op in ("var", "hparam"):
        return []
    if len(node) == 2:
        return _collect_const_leaf_paths(node[1], path=tuple(path) + (1,))
    if len(node) == 3:
        return _collect_const_leaf_paths(node[1], path=tuple(path) + (1,)) + _collect_const_leaf_paths(
            node[2],
            path=tuple(path) + (2,),
        )
    return []


def _candidate_family_key(item: Mapping[str, Any]) -> str:
    kind = str(item.get("edit_kind", "") or "")
    if not kind:
        return ""
    parts = kind.split(":")
    if not parts:
        return kind
    head = str(parts[0] or "")
    if head in ("replace", "combine", "promote"):
        if len(parts) >= 2:
            return f"{head}:{parts[1]}"
        return head
    if head == "wrap":
        if len(parts) >= 2:
            return f"{head}:{parts[1]}"
        return head
    if head == "untie_const":
        return "untie_const"
    if head == "power":
        return "power"
    return head


def _select_diverse_candidates(
    ranked_candidates: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    limit = max(0, int(limit))
    if limit <= 0:
        return []
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in ranked_candidates:
        row = dict(item)
        family = _candidate_family_key(row)
        if family not in buckets:
            buckets[family] = []
            order.append(family)
        buckets[family].append(row)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        advanced = False
        for family in order:
            bucket = buckets.get(family, [])
            if depth < len(bucket):
                selected.append(bucket[depth])
                advanced = True
                if len(selected) >= limit:
                    break
        if not advanced:
            break
        depth += 1
    return selected


def _candidate_value_grad(
    candidate: Mapping[str, Any],
    x_rank: torch.Tensor,
    *,
    capture_gradients: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    value = _normalize_value_tensor(candidate.get("value_fit", None), x=x_rank)
    grad = None
    if capture_gradients:
        grad = _normalize_grad_tensor(candidate.get("grad_fit", None), x=x_rank)
    if value is not None and (not capture_gradients or grad is not None):
        return value, grad
    node = candidate.get("node", None)
    if not isinstance(node, tuple) or not node:
        return None, None
    return _capture_node_value_grad(node, x_rank, capture_gradients=capture_gradients)


def _rms_tensor(value: torch.Tensor) -> float:
    if not torch.is_tensor(value):
        return 0.0
    vv = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return float(torch.sqrt(torch.clamp(torch.mean(vv * vv), min=0.0)).item())
    except Exception:
        return 0.0


def _solve_ridge_system(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor | None:
    if not torch.is_tensor(design) or not torch.is_tensor(target):
        return None
    if design.ndim != 2 or target.ndim != 2:
        return None
    if int(design.shape[0]) <= 0 or int(design.shape[1]) <= 0 or int(target.shape[0]) != int(design.shape[0]):
        return None
    ridge = max(0.0, float(ridge))
    try:
        if ridge > 0.0:
            eye = torch.eye(int(design.shape[1]), dtype=design.dtype, device=design.device)
            gram = design.transpose(0, 1) @ design + (ridge * eye)
            rhs = design.transpose(0, 1) @ target
            coeffs = torch.linalg.solve(gram, rhs)
        else:
            coeffs = torch.linalg.lstsq(design, target).solution
    except Exception:
        try:
            coeffs = torch.linalg.lstsq(design, target).solution
        except Exception:
            return None
    if coeffs is None or not torch.isfinite(coeffs).all():
        return None
    return coeffs


def _fit_linear_features(
    *,
    value_cols: Sequence[torch.Tensor],
    target_value: torch.Tensor,
    grad_cols: Sequence[torch.Tensor | None] | None = None,
    target_grad: torch.Tensor | None = None,
    grad_weight: float = 0.5,
    ridge: float = 1.0e-6,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not value_cols:
        return None, None, None
    try:
        value_design = torch.cat([_ensure_col(col) for col in value_cols], dim=1)
    except Exception:
        return None, None, None
    target_value = _ensure_col(target_value)
    design = value_design
    target = target_value
    grad_design = None
    grad_enabled = (
        target_grad is not None
        and grad_cols is not None
        and len(tuple(grad_cols)) == len(tuple(value_cols))
        and float(grad_weight) > 0.0
        and all(torch.is_tensor(col) for col in tuple(grad_cols))
    )
    if grad_enabled:
        try:
            grad_design = torch.cat([col.reshape(-1, 1) for col in tuple(grad_cols)], dim=1)
            target_grad_vec = target_grad.reshape(-1, 1)
            grad_scale = math.sqrt(max(0.0, float(grad_weight)))
            design = torch.cat([value_design, grad_scale * grad_design], dim=0)
            target = torch.cat([target_value, grad_scale * target_grad_vec], dim=0)
        except Exception:
            grad_design = None
            grad_enabled = False
    coeffs = _solve_ridge_system(design, target, ridge=float(ridge))
    if coeffs is None:
        return None, None, None
    pred_value = value_design @ coeffs
    pred_grad = None
    if grad_enabled and grad_design is not None and target_grad is not None:
        try:
            pred_grad = (grad_design @ coeffs).reshape_as(target_grad)
        except Exception:
            pred_grad = None
    return coeffs, pred_value, pred_grad


def _coeff_to_term(node, coeff: float):
    coeff = float(coeff)
    if not math.isfinite(coeff) or abs(coeff) <= 1.0e-12:
        return None
    if abs(coeff - 1.0) <= 1.0e-12:
        return node
    if abs(coeff + 1.0) <= 1.0e-12:
        return ("neg", node)
    return ("mul", ("const", float(coeff)), node)


def _build_linear_combo_expr(
    term_nodes: Sequence[tuple],
    term_coeffs: Sequence[float],
    *,
    bias: float | None = None,
):
    terms = []
    if bias is not None and math.isfinite(float(bias)) and abs(float(bias)) > 1.0e-12:
        terms.append(("const", float(bias)))
    for node, coeff in zip(tuple(term_nodes), tuple(term_coeffs)):
        term = _coeff_to_term(node, float(coeff))
        if term is not None:
            terms.append(term)
    if not terms:
        return ("const", 0.0)
    expr = terms[0]
    for term in terms[1:]:
        expr = ("add", expr, term)
    return _simplify_node(expr)


def _prune_linear_support(
    *,
    term_infos: Sequence[Mapping[str, Any]],
    term_coeffs: Sequence[float],
    feature_value_cols: Sequence[torch.Tensor],
    bias: float | None = None,
    include_bias: bool,
    rel: float = 1.0e-4,
    abs_eps: float = 1.0e-10,
) -> tuple[float | None, list[dict[str, Any]], list[float]]:
    rel = max(0.0, float(rel))
    abs_eps = max(0.0, float(abs_eps))
    contribs: list[float] = []
    bias_contrib = 0.0
    if include_bias and bias is not None and math.isfinite(float(bias)):
        bias_contrib = abs(float(bias))
        contribs.append(bias_contrib)
    for coeff, col in zip(tuple(term_coeffs), tuple(feature_value_cols)):
        contribs.append(abs(float(coeff)) * _rms_tensor(_ensure_col(col)))
    max_contrib = max(contribs) if contribs else 0.0

    keep_bias = False
    if include_bias and bias is not None and math.isfinite(float(bias)) and abs(float(bias)) > abs_eps:
        keep_bias = max_contrib <= 0.0 or bias_contrib >= rel * max_contrib

    kept_infos: list[dict[str, Any]] = []
    kept_coeffs: list[float] = []
    for info, coeff, col in zip(tuple(term_infos), tuple(term_coeffs), tuple(feature_value_cols)):
        coeff = float(coeff)
        if not math.isfinite(coeff) or abs(coeff) <= abs_eps:
            continue
        contrib = abs(coeff) * _rms_tensor(_ensure_col(col))
        if max_contrib > 0.0 and contrib < rel * max_contrib:
            continue
        kept_infos.append(dict(info))
        kept_coeffs.append(coeff)

    if not kept_infos and not keep_bias:
        if term_infos:
            ranked = sorted(
                zip(tuple(term_infos), tuple(term_coeffs), tuple(feature_value_cols)),
                key=lambda item: (
                    -(abs(float(item[1])) * _rms_tensor(_ensure_col(item[2]))),
                    int(node_size(item[0].get("node", ("const", 0.0)))),
                ),
            )
            if ranked:
                best_info, best_coeff, _ = ranked[0]
                if math.isfinite(float(best_coeff)) and abs(float(best_coeff)) > abs_eps:
                    kept_infos = [dict(best_info)]
                    kept_coeffs = [float(best_coeff)]
        elif include_bias and bias is not None and math.isfinite(float(bias)):
            keep_bias = abs(float(bias)) > abs_eps

    return (float(bias) if keep_bias and bias is not None else None), kept_infos, kept_coeffs


def _rank_seed_infos(
    seed_infos: Sequence[Mapping[str, Any]],
    *,
    target_value: torch.Tensor,
    target_grad: torch.Tensor | None,
    mode: str = "direct",
    base_value_fit: torch.Tensor | None = None,
    base_grad_fit: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target_value = _ensure_col(target_value)
    for raw in seed_infos:
        info = dict(raw)
        value_fit = _ensure_col(info.get("value_fit", None))
        grad_fit = info.get("grad_fit", None)
        feature_value = value_fit
        feature_grad = grad_fit if torch.is_tensor(grad_fit) else None
        if str(mode) == "modulation":
            if base_value_fit is None:
                continue
            feature_value = _ensure_col(base_value_fit) * value_fit
            feature_grad = None
            if target_grad is not None and base_grad_fit is not None and torch.is_tensor(grad_fit):
                feature_grad = (value_fit * base_grad_fit) + (_ensure_col(base_value_fit) * grad_fit)
        if not torch.isfinite(feature_value).all():
            continue
        prediction_score = _alignment_score(target_value, feature_value)
        gradient_score = None
        if target_grad is not None and feature_grad is not None:
            gradient_score = _alignment_score(target_grad, feature_grad)
        total_score = float(prediction_score + (0.5 * float(gradient_score) if gradient_score is not None else 0.0))
        info["feature_value"] = feature_value.detach()
        info["feature_grad"] = None if feature_grad is None else feature_grad.detach()
        info["feature_prediction_score"] = float(prediction_score)
        info["feature_gradient_score"] = None if gradient_score is None else float(gradient_score)
        info["feature_score"] = float(total_score)
        out.append(info)
    out.sort(
        key=lambda item: (
            -float(item.get("feature_score", 0.0)),
            -float(item.get("feature_gradient_score", 0.0) or 0.0),
            -float(item.get("feature_prediction_score", 0.0)),
            int(node_size(item.get("node", ("const", 0.0)))),
            node_str(item.get("node", ("const", 0.0))),
        )
    )
    return out


def _fit_replace_or_add_combo_candidate(
    *,
    base_node,
    x_rank: torch.Tensor,
    term_infos: Sequence[Mapping[str, Any]],
    target_value: torch.Tensor,
    target_grad: torch.Tensor | None,
    include_bias: bool,
    target_dim: Any,
    var_dims,
    edit_kind: str,
    add_to_base: bool,
    fit_target_value: torch.Tensor | None = None,
    fit_target_grad: torch.Tensor | None = None,
    grad_weight: float = 0.5,
    ridge: float = 1.0e-6,
) -> dict[str, Any] | None:
    if not term_infos:
        return None
    value_cols: list[torch.Tensor] = []
    grad_cols: list[torch.Tensor | None] = []
    if include_bias:
        value_cols.append(torch.ones_like(_ensure_col(target_value)))
        grad_cols.append(
            torch.zeros((int(x_rank.shape[0]), int(x_rank.shape[1])), dtype=x_rank.dtype, device=x_rank.device)
            if target_grad is not None
            else None
        )
    for info in term_infos:
        value_cols.append(_ensure_col(info["feature_value"]))
        grad_cols.append(info.get("feature_grad", None))
    coeffs, _, _ = _fit_linear_features(
        value_cols=value_cols,
        target_value=_ensure_col(target_value),
        grad_cols=grad_cols,
        target_grad=target_grad,
        grad_weight=float(grad_weight),
        ridge=float(ridge),
    )
    if coeffs is None:
        return None
    coeff_list = [float(v) for v in coeffs.reshape(-1).detach().cpu().tolist()]
    bias = coeff_list[0] if include_bias else None
    term_coeffs = coeff_list[1:] if include_bias else coeff_list
    feature_term_cols = value_cols[1:] if include_bias else value_cols
    pruned_bias, kept_infos, kept_coeffs = _prune_linear_support(
        term_infos=term_infos,
        term_coeffs=term_coeffs,
        feature_value_cols=feature_term_cols,
        bias=bias,
        include_bias=bool(include_bias),
    )
    if not kept_infos and pruned_bias is None:
        return None
    combo_expr = _build_linear_combo_expr(
        [info["node"] for info in kept_infos],
        kept_coeffs,
        bias=pruned_bias,
    )
    node = combo_expr if not add_to_base else _simplify_node(("add", base_node, combo_expr))
    if var_dims is not None and target_dim is not None and not _node_matches_target_dim(
        node,
        target_dim=target_dim,
        var_dims=var_dims,
    ):
        return None
    value_fit, grad_fit = _capture_node_value_grad(node, x_rank, capture_gradients=bool(target_grad is not None or fit_target_grad is not None))
    if value_fit is None:
        return None
    compare_value = _ensure_col(fit_target_value if fit_target_value is not None else target_value)
    compare_grad = fit_target_grad if fit_target_grad is not None else target_grad
    fit_mse = float(torch.mean((_ensure_col(value_fit) - compare_value) ** 2).item())
    fit_grad_mse = None
    if compare_grad is not None and grad_fit is not None:
        fit_grad_mse = float(torch.mean((grad_fit - compare_grad) ** 2).item())
    return {
        "node": node,
        "edit_kind": str(edit_kind),
        "anchor": kept_infos[0]["node"] if kept_infos else None,
        "fit_coefficients": [float(v) for v in ([pruned_bias] if pruned_bias is not None else []) + kept_coeffs],
        "fit_support": [node_str(info["node"]) for info in kept_infos],
        "fit_mse": float(fit_mse),
        "fit_grad_mse": None if fit_grad_mse is None else float(fit_grad_mse),
        "value_fit": value_fit.detach(),
        "grad_fit": None if grad_fit is None else grad_fit.detach(),
    }


def _fit_multiplicative_combo_candidate(
    *,
    base_node,
    x_rank: torch.Tensor,
    base_value_fit: torch.Tensor,
    base_grad_fit: torch.Tensor | None,
    term_infos: Sequence[Mapping[str, Any]],
    target_value: torch.Tensor,
    target_grad: torch.Tensor | None,
    target_dim: Any,
    var_dims,
    edit_kind: str,
    grad_weight: float = 0.5,
    ridge: float = 1.0e-6,
) -> dict[str, Any] | None:
    value_cols: list[torch.Tensor] = [_ensure_col(base_value_fit)]
    grad_cols: list[torch.Tensor | None] = [base_grad_fit]
    term_feature_cols: list[torch.Tensor] = []
    for info in term_infos:
        seed_value = _ensure_col(info["value_fit"])
        feature_value = _ensure_col(base_value_fit) * seed_value
        feature_grad = None
        if target_grad is not None and base_grad_fit is not None and torch.is_tensor(info.get("grad_fit", None)):
            feature_grad = (seed_value * base_grad_fit) + (_ensure_col(base_value_fit) * info["grad_fit"])
        value_cols.append(feature_value)
        grad_cols.append(feature_grad)
        term_feature_cols.append(feature_value)
    coeffs, _, _ = _fit_linear_features(
        value_cols=value_cols,
        target_value=_ensure_col(target_value),
        grad_cols=grad_cols,
        target_grad=target_grad,
        grad_weight=float(grad_weight),
        ridge=float(ridge),
    )
    if coeffs is None:
        return None
    coeff_list = [float(v) for v in coeffs.reshape(-1).detach().cpu().tolist()]
    bias = coeff_list[0]
    term_coeffs = coeff_list[1:]
    pruned_bias, kept_infos, kept_coeffs = _prune_linear_support(
        term_infos=term_infos,
        term_coeffs=term_coeffs,
        feature_value_cols=term_feature_cols,
        bias=bias,
        include_bias=True,
    )
    if pruned_bias is None and not kept_infos:
        return None
    modifier = _build_linear_combo_expr(
        [info["node"] for info in kept_infos],
        kept_coeffs,
        bias=pruned_bias,
    )
    node = _simplify_node(("mul", base_node, modifier))
    if var_dims is not None and target_dim is not None and not _node_matches_target_dim(
        node,
        target_dim=target_dim,
        var_dims=var_dims,
    ):
        return None
    value_fit, grad_fit = _capture_node_value_grad(node, x_rank, capture_gradients=bool(target_grad is not None))
    if value_fit is None:
        return None
    fit_mse = float(torch.mean((_ensure_col(value_fit) - _ensure_col(target_value)) ** 2).item())
    fit_grad_mse = None
    if target_grad is not None and grad_fit is not None:
        fit_grad_mse = float(torch.mean((grad_fit - target_grad) ** 2).item())
    return {
        "node": node,
        "edit_kind": str(edit_kind),
        "anchor": kept_infos[0]["node"] if kept_infos else None,
        "fit_coefficients": [float(v) for v in ([pruned_bias] if pruned_bias is not None else []) + kept_coeffs],
        "fit_support": [node_str(info["node"]) for info in kept_infos],
        "fit_mse": float(fit_mse),
        "fit_grad_mse": None if fit_grad_mse is None else float(fit_grad_mse),
        "value_fit": value_fit.detach(),
        "grad_fit": None if grad_fit is None else grad_fit.detach(),
    }


def _dedup_candidate_dicts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        node = row.get("node", None)
        if not isinstance(node, tuple) or not node:
            continue
        key = node_str(node)
        sort_key = (
            float(row.get("fit_mse", float("inf"))),
            float(row.get("fit_grad_mse", float("inf")) if row.get("fit_grad_mse", None) is not None else float("inf")),
            int(node_size(node)),
            str(row.get("edit_kind", "")),
        )
        current = best.get(key, None)
        if current is None:
            best[key] = dict(row)
            best[key]["_sort_key"] = sort_key
            continue
        if sort_key < tuple(current.get("_sort_key", ()) or (float("inf"), float("inf"), 10**9, "")):
            best[key] = dict(row)
            best[key]["_sort_key"] = sort_key
    out = [dict(row) for row in best.values()]
    for row in out:
        row.pop("_sort_key", None)
    out.sort(
        key=lambda row: (
            float(row.get("fit_mse", float("inf"))),
            float(row.get("fit_grad_mse", float("inf")) if row.get("fit_grad_mse", None) is not None else float("inf")),
            int(node_size(row.get("node", ("const", 0.0)))),
            str(row.get("edit_kind", "")),
        )
    )
    return out


def _fit_sparse_combo_family(
    *,
    ranked_infos: Sequence[Mapping[str, Any]],
    base_node,
    x_rank: torch.Tensor,
    target_value: torch.Tensor,
    target_grad: torch.Tensor | None,
    target_dim: Any,
    var_dims,
    edit_kind_prefix: str,
    add_to_base: bool = False,
    include_bias: bool = False,
    max_seed_pool: int = 6,
    max_terms: int = 3,
    max_return: int = 6,
) -> list[dict[str, Any]]:
    pool = [dict(info) for info in tuple(ranked_infos)[: max(1, int(max_seed_pool))]]
    rows: list[dict[str, Any]] = []
    if not pool:
        return rows
    for width in range(1, min(int(max_terms), len(pool)) + 1):
        for subset in itertools.combinations(pool, width):
            row = _fit_replace_or_add_combo_candidate(
                base_node=base_node,
                x_rank=x_rank,
                term_infos=subset,
                target_value=target_value,
                target_grad=target_grad,
                include_bias=bool(include_bias),
                target_dim=target_dim,
                var_dims=var_dims,
                edit_kind=f"{str(edit_kind_prefix)}:{int(width)}",
                add_to_base=bool(add_to_base),
                fit_target_value=target_value if not add_to_base else None,
                fit_target_grad=target_grad if not add_to_base else None,
            )
            if row is not None:
                rows.append(row)
    rows = _dedup_candidate_dicts(rows)
    return rows[: max(1, int(max_return))]


def _fit_sparse_multiplicative_family(
    *,
    ranked_infos: Sequence[Mapping[str, Any]],
    base_node,
    x_rank: torch.Tensor,
    base_value_fit: torch.Tensor,
    base_grad_fit: torch.Tensor | None,
    target_value: torch.Tensor,
    target_grad: torch.Tensor | None,
    target_dim: Any,
    var_dims,
    edit_kind_prefix: str,
    max_seed_pool: int = 6,
    max_terms: int = 3,
    max_return: int = 6,
) -> list[dict[str, Any]]:
    pool = [dict(info) for info in tuple(ranked_infos)[: max(1, int(max_seed_pool))]]
    rows: list[dict[str, Any]] = []
    if base_value_fit is None or not pool:
        return rows
    for width in range(1, min(int(max_terms), len(pool)) + 1):
        for subset in itertools.combinations(pool, width):
            row = _fit_multiplicative_combo_candidate(
                base_node=base_node,
                x_rank=x_rank,
                base_value_fit=base_value_fit,
                base_grad_fit=base_grad_fit,
                term_infos=subset,
                target_value=target_value,
                target_grad=target_grad,
                target_dim=target_dim,
                var_dims=var_dims,
                edit_kind=f"{str(edit_kind_prefix)}:{int(width)}",
            )
            if row is not None:
                rows.append(row)
    rows = _dedup_candidate_dicts(rows)
    return rows[: max(1, int(max_return))]


def _fit_hparam_template_candidate(
    template_node,
    *,
    x_rank: torch.Tensor,
    target_value: torch.Tensor,
    target_grad: torch.Tensor | None,
    init_values: Sequence[float],
    target_dim: Any,
    var_dims,
    edit_kind: str,
    anchor=None,
    steps: int = 48,
    lr: float = 0.15,
    grad_weight: float = 0.5,
    l2_weight: float = 1.0e-4,
    safe_penalty_weight: float = 1.0e-2,
) -> dict[str, Any] | None:
    try:
        from .explorer import _eval_node_hparam_safe, _materialize_hparams
    except Exception:
        return None
    init = torch.as_tensor(list(init_values or ()), dtype=x_rank.dtype, device=x_rank.device).reshape(-1)
    if init.numel() <= 0:
        return None
    raw = init.detach().clone().requires_grad_(True)
    best_raw = None
    best_loss = None
    opt = torch.optim.Adam([raw], lr=float(max(1.0e-4, lr)))
    patience = 8
    since_improve = 0
    need_grad = bool(target_grad is not None)
    for _ in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x_req = x_rank.detach().clone().requires_grad_(need_grad)
            pred, penalty = _eval_node_hparam_safe(
                template_node,
                x_req,
                raw,
                {"safe_eps": 1.0e-6, "safe_exp_clip": 30.0},
            )
            pred = _normalize_value_tensor(pred, x=x_req)
            if pred is None:
                loss = (raw * raw).sum() + torch.tensor(1.0e6, dtype=raw.dtype, device=raw.device)
            else:
                loss = torch.mean((pred - _ensure_col(target_value)) ** 2)
                loss = loss + (float(l2_weight) * torch.mean((raw - init) ** 2))
                loss = loss + (float(safe_penalty_weight) * penalty)
                if target_grad is not None:
                    grad = torch.autograd.grad(
                        pred.sum(),
                        x_req,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    if grad is None:
                        grad = torch.zeros_like(x_req)
                    loss = loss + (float(grad_weight) * torch.mean((grad - target_grad) ** 2))
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        loss_value = _finite_float(loss.item())
        if loss_value is None:
            continue
        if best_loss is None or float(loss_value) < (best_loss - 1.0e-9):
            best_loss = float(loss_value)
            best_raw = raw.detach().clone()
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break
    if best_raw is None:
        return None
    node = _simplify_node(_materialize_hparams(template_node, [float(v) for v in best_raw.detach().cpu().tolist()]))
    if var_dims is not None and target_dim is not None and not _node_matches_target_dim(
        node,
        target_dim=target_dim,
        var_dims=var_dims,
    ):
        return None
    value_fit, grad_fit = _capture_node_value_grad(node, x_rank, capture_gradients=bool(target_grad is not None))
    if value_fit is None:
        return None
    fit_mse = float(torch.mean((_ensure_col(value_fit) - _ensure_col(target_value)) ** 2).item())
    fit_grad_mse = None
    if target_grad is not None and grad_fit is not None:
        fit_grad_mse = float(torch.mean((grad_fit - target_grad) ** 2).item())
    return {
        "node": node,
        "edit_kind": str(edit_kind),
        "anchor": anchor,
        "fit_coefficients": [float(v) for v in best_raw.detach().cpu().tolist()],
        "fit_support": [],
        "fit_mse": float(fit_mse),
        "fit_grad_mse": None if fit_grad_mse is None else float(fit_grad_mse),
        "value_fit": value_fit.detach(),
        "grad_fit": None if grad_fit is None else grad_fit.detach(),
    }


def _enumerate_tangent_edit_nodes(
    base_node,
    *,
    target_dim: Any,
    nvars: int,
    active_vars: Sequence[int],
    wrappers_left: int | None = None,
    pool_nodes: Sequence[Any] | None,
    var_dims,
    x_rank: torch.Tensor | None = None,
    t_rank: torch.Tensor | None = None,
    target_grad_rank: torch.Tensor | None = None,
    base_value_fit: torch.Tensor | None = None,
    base_grad_fit: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(base_node, tuple) or not base_node:
        return []
    base_node = _simplify_node(base_node)
    base_dim = None
    if var_dims is not None:
        try:
            base_dim = node_dims(base_node, var_dims)
        except Exception:
            base_dim = None
    seed_nodes = _seed_nodes(active_vars=active_vars, nvars=nvars, pool_nodes=pool_nodes)
    seed_dims: dict[str, Any] = {}
    if var_dims is not None:
        for seed in seed_nodes:
            try:
                seed_dims[node_str(seed)] = node_dims(seed, var_dims)
            except Exception:
                seed_dims[node_str(seed)] = None

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_key = node_str(base_node)

    def _add_candidate(candidate: Mapping[str, Any] | None) -> None:
        if not isinstance(candidate, Mapping):
            return
        node = candidate.get("node", None)
        if not isinstance(node, tuple) or not node:
            return
        simp = _simplify_node(node)
        key = node_str(simp)
        if key == base_key or key in seen:
            return
        if var_dims is not None and target_dim is not None and not _node_matches_target_dim(
            simp,
            target_dim=target_dim,
            var_dims=var_dims,
        ):
            return
        row = dict(candidate)
        row["node"] = simp
        seen.add(key)
        out.append(row)

    def _add(node, *, edit_kind: str, anchor=None) -> None:
        _add_candidate(
            {
                "node": node,
                "edit_kind": str(edit_kind),
                "anchor": anchor,
            }
        )

    allow_wrap_edits = wrappers_left is None or int(wrappers_left) > 0
    if bool(allow_wrap_edits):
        for op in ("neg", "sin", "cos", "exp", "log", "sqrt", "sqr"):
            edit_kind = f"wrap:{op}"
            if _edit_dim_allowed(edit_kind=edit_kind, base_dim=base_dim, target_dim=target_dim):
                _add((op, base_node), edit_kind=edit_kind)

    for node, edit_kind in (
        (("div", ("const", 1.0), base_node), "power:-1"),
        (("div", ("const", 1.0), ("sqr", base_node)), "power:-2"),
        (("mul", base_node, ("sqr", base_node)), "power:+3"),
        (("sqr", ("sqr", base_node)), "power:+4"),
    ):
        _add(node, edit_kind=edit_kind)

    for seed in seed_nodes:
        seed_key = node_str(seed)
        term_dim = seed_dims.get(seed_key, None)
        if _edit_dim_allowed(edit_kind="replace", base_dim=base_dim, term_dim=term_dim, target_dim=target_dim):
            _add(seed, edit_kind="replace", anchor=seed)
        if _edit_dim_allowed(edit_kind="combine:add", base_dim=base_dim, term_dim=term_dim, target_dim=target_dim):
            _add(("add", base_node, seed), edit_kind="combine:add", anchor=seed)
        if _edit_dim_allowed(edit_kind="combine:sub", base_dim=base_dim, term_dim=term_dim, target_dim=target_dim):
            _add(("sub", base_node, seed), edit_kind="combine:sub", anchor=seed)
        if _edit_dim_allowed(edit_kind="combine:mul", base_dim=base_dim, term_dim=term_dim, target_dim=target_dim):
            _add(("mul", base_node, seed), edit_kind="combine:mul", anchor=seed)
        if _edit_dim_allowed(edit_kind="combine:div", base_dim=base_dim, term_dim=term_dim, target_dim=target_dim):
            _add(("div", base_node, seed), edit_kind="combine:div", anchor=seed)

    if torch.is_tensor(x_rank) and torch.is_tensor(t_rank):
        need_grad = bool(target_grad_rank is not None)
        if base_value_fit is None:
            base_value_fit, base_grad_fit = _capture_node_value_grad(base_node, x_rank, capture_gradients=need_grad)
        if base_value_fit is not None:
            target_value_rank = _ensure_col(t_rank)
            grad_residual_rank = None
            if target_grad_rank is not None and base_grad_fit is not None:
                grad_residual_rank = target_grad_rank - base_grad_fit
            residual_rank = target_value_rank - _ensure_col(base_value_fit)

            seed_infos: list[dict[str, Any]] = []
            for seed in seed_nodes:
                value_fit, grad_fit = _capture_node_value_grad(seed, x_rank, capture_gradients=need_grad)
                if value_fit is None:
                    continue
                seed_infos.append(
                    {
                        "node": seed,
                        "dim": seed_dims.get(node_str(seed), None),
                        "value_fit": value_fit.detach(),
                        "grad_fit": None if grad_fit is None else grad_fit.detach(),
                    }
                )

            same_dim_infos: list[dict[str, Any]] = []
            dimless_infos: list[dict[str, Any]] = []
            zero_dim = None
            if target_dim is not None:
                zero_dim = _zero_dim_like(target_dim)
            elif base_dim is not None:
                zero_dim = _zero_dim_like(base_dim)
            for info in seed_infos:
                node = info["node"]
                if isinstance(node, tuple) and node and str(node[0]) == "const":
                    continue
                dim = info.get("dim", None)
                if var_dims is None or target_dim is None or (dim is not None and dims_eq(dim, target_dim)):
                    same_dim_infos.append(info)
                if var_dims is None or zero_dim is None or (dim is not None and dims_eq(dim, zero_dim)):
                    dimless_infos.append(info)

            base_info = {
                "node": base_node,
                "dim": base_dim,
                "value_fit": _ensure_col(base_value_fit).detach(),
                "grad_fit": None if base_grad_fit is None else base_grad_fit.detach(),
                "feature_value": _ensure_col(base_value_fit).detach(),
                "feature_grad": None if base_grad_fit is None else base_grad_fit.detach(),
            }
            _add_candidate(
                _fit_replace_or_add_combo_candidate(
                    base_node=base_node,
                    x_rank=x_rank,
                    term_infos=[base_info],
                    target_value=target_value_rank,
                    target_grad=target_grad_rank,
                    include_bias=False,
                    fit_target_value=target_value_rank,
                    fit_target_grad=target_grad_rank,
                    target_dim=target_dim,
                    var_dims=var_dims,
                    edit_kind="promote:scale",
                    add_to_base=False,
                )
            )
            if _const_bias_allowed(target_dim=target_dim, var_dims=var_dims):
                _add_candidate(
                    _fit_replace_or_add_combo_candidate(
                        base_node=base_node,
                        x_rank=x_rank,
                        term_infos=[base_info],
                        target_value=target_value_rank,
                        target_grad=target_grad_rank,
                        include_bias=True,
                        fit_target_value=target_value_rank,
                        fit_target_grad=target_grad_rank,
                        target_dim=target_dim,
                        var_dims=var_dims,
                        edit_kind="promote:scale_shift",
                        add_to_base=False,
                    )
                )

            const_paths = _collect_const_leaf_paths(base_node)
            for const_path in const_paths[:3]:
                try:
                    const_node = get_at(base_node, const_path)
                except Exception:
                    continue
                if not isinstance(const_node, tuple) or not const_node or str(const_node[0]) != "const":
                    continue
                template = replace_at(base_node, const_path, ("hparam", 0))
                _add_candidate(
                    _fit_hparam_template_candidate(
                        template,
                        x_rank=x_rank,
                        target_value=target_value_rank,
                        target_grad=target_grad_rank,
                        init_values=[float(const_node[1])],
                        target_dim=target_dim,
                        var_dims=var_dims,
                        edit_kind="untie_const:leaf",
                        anchor=const_node,
                    )
                )
            if len(const_paths) >= 2:
                template = base_node
                init_values: list[float] = []
                for slot, const_path in enumerate(const_paths[:3]):
                    try:
                        const_node = get_at(base_node, const_path)
                    except Exception:
                        break
                    if not isinstance(const_node, tuple) or not const_node or str(const_node[0]) != "const":
                        break
                    template = replace_at(template, const_path, ("hparam", int(slot)))
                    init_values.append(float(const_node[1]))
                if init_values:
                    _add_candidate(
                        _fit_hparam_template_candidate(
                            template,
                            x_rank=x_rank,
                            target_value=target_value_rank,
                            target_grad=target_grad_rank,
                            init_values=init_values,
                            target_dim=target_dim,
                            var_dims=var_dims,
                            edit_kind="untie_const:joint",
                            anchor=None,
                        )
                    )

            ranked_replace_infos = _rank_seed_infos(
                same_dim_infos,
                target_value=target_value_rank,
                target_grad=target_grad_rank,
                mode="direct",
            )
            for row in _fit_sparse_combo_family(
                ranked_infos=ranked_replace_infos,
                base_node=base_node,
                x_rank=x_rank,
                target_value=target_value_rank,
                target_grad=target_grad_rank,
                target_dim=target_dim,
                var_dims=var_dims,
                edit_kind_prefix="replace:sparse_combo",
                add_to_base=False,
                include_bias=_const_bias_allowed(target_dim=target_dim, var_dims=var_dims),
                max_seed_pool=6,
                max_terms=3,
                max_return=6,
            ):
                _add_candidate(row)

            ranked_add_infos = _rank_seed_infos(
                same_dim_infos,
                target_value=residual_rank,
                target_grad=grad_residual_rank,
                mode="direct",
            )
            add_rows = _fit_sparse_combo_family(
                ranked_infos=ranked_add_infos,
                base_node=base_node,
                x_rank=x_rank,
                target_value=residual_rank,
                target_grad=grad_residual_rank,
                target_dim=target_dim,
                var_dims=var_dims,
                edit_kind_prefix="combine:add_sparse_combo",
                add_to_base=True,
                include_bias=_const_bias_allowed(target_dim=target_dim, var_dims=var_dims),
                max_seed_pool=6,
                max_terms=3,
                max_return=6,
            )
            for row in add_rows:
                compare_value = target_value_rank
                compare_grad = target_grad_rank
                row["fit_mse"] = float(torch.mean((_ensure_col(row["value_fit"]) - compare_value) ** 2).item())
                if compare_grad is not None and row.get("grad_fit", None) is not None:
                    row["fit_grad_mse"] = float(torch.mean((row["grad_fit"] - compare_grad) ** 2).item())
                _add_candidate(row)

            ranked_mod_infos = _rank_seed_infos(
                dimless_infos,
                target_value=target_value_rank,
                target_grad=target_grad_rank,
                mode="modulation",
                base_value_fit=_ensure_col(base_value_fit),
                base_grad_fit=base_grad_fit,
            )
            for row in _fit_sparse_multiplicative_family(
                ranked_infos=ranked_mod_infos,
                base_node=base_node,
                x_rank=x_rank,
                base_value_fit=_ensure_col(base_value_fit),
                base_grad_fit=base_grad_fit,
                target_value=target_value_rank,
                target_grad=target_grad_rank,
                target_dim=target_dim,
                var_dims=var_dims,
                edit_kind_prefix="combine:mul_sparse_combo",
                max_seed_pool=6,
                max_terms=3,
                max_return=6,
            ):
                _add_candidate(row)

    out = _dedup_candidate_dicts(out)
    out.sort(
        key=lambda item: (
            float(item.get("fit_mse", float("inf"))),
            float(item.get("fit_grad_mse", float("inf")) if item.get("fit_grad_mse", None) is not None else float("inf")),
            int(node_size(item["node"])),
            str(item.get("edit_kind", "")),
            node_str(item["node"]),
        )
    )
    return out


def _tangent_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    total_score = _finite_float(item.get("tangent_score", None))
    grad_score = _finite_float(item.get("gradient_score", None))
    pred_score = _finite_float(item.get("prediction_score", None))
    node = item.get("node", None)
    try:
        size = int(node_size(node)) if isinstance(node, tuple) else 10**9
    except Exception:
        size = 10**9
    return (
        -(float("-inf") if total_score is None else float(total_score)),
        -(float("-inf") if grad_score is None else float(grad_score)),
        -(float("-inf") if pred_score is None else float(pred_score)),
        size,
        str(item.get("edit_kind", "")),
    )


def _build_solver_context(
    *,
    parent_node,
    hole_path: tuple[int, ...],
    hole_sub,
    max_depth: int,
    max_subtree_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims,
    pool_nodes,
    pool_dims,
    local_score_mode: str,
    target_mode: str,
    target_mapping_kind: str,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 0.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
) -> _SolverContext:
    return _SolverContext(
        parent_node=parent_node,
        hole_path=hole_path,
        hole_sub=hole_sub,
        max_depth=int(max_depth),
        nvars=int(nvars),
        poly_degree=int(poly_degree),
        var_dims=var_dims,
        pool_nodes=list(pool_nodes or []),
        pool_dims=list(pool_dims or []),
        seed_nodes=[],
        local_score_mode=str(local_score_mode),
        enum_max_depth=1,
        enum_max_trees=1,
        max_subtree_depth=int(max_subtree_depth),
        preview_topk=1,
        complexity_penalty=0.0,
        recursive_enable=False,
        recursive_max_depth=0,
        recursive_trigger_rel_mse=1.0,
        recursive_seed_cap=0,
        recursive_branch_topk=0,
        recursive_child_topk=0,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=1.0,
        confidence_floor=0.0,
        branch_beam_width=1,
        min_valid_frac=0.0,
        min_confidence=0.0,
        allow_legacy_aux=False,
        legacy_aux_kwargs={},
        stats={},
        target_mode=str(target_mode or ""),
        target_mapping_kind=str(target_mapping_kind or ""),
        family_battery_enable=False,
        witness_loss_enable=bool(witness_loss_enable),
        witness_grad_weight=float(witness_grad_weight),
        witness_d2_weight=float(witness_d2_weight),
        witness_diag_weight=float(witness_diag_weight),
        witness_physics_weight=float(witness_physics_weight),
    )


def solve_local_tangent_edit_preview_rows(
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
    pool_nodes=None,
    pool_dims=None,
    local_score_mode: str = "affine",
    preview_topk: int = 8,
    max_subtree_depth: int | None = None,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 0.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    route_preview_topk = max(1, int(preview_topk))
    hole_path = tuple(int(v) for v in tuple(path or ()))
    mode_name = _normalize_inverse_local_score_mode(local_score_mode)
    solver_meta: dict[str, Any] = {
        "status": "started",
        "requested_preview_topk": int(route_preview_topk),
        "tangent_fit_points_used": 0,
        "target_gradient_used": False,
        "candidate_count_generated": 0,
        "candidate_count_ranked": 0,
        "candidate_count_scored": 0,
        "preview_count": 0,
    }

    problem, continuation_frames, hole_sub, subproblem_spec = _local_problem_from_payload(spec_payload)
    if problem is None:
        solver_meta["status"] = "invalid_local_problem"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    base_node = hole_sub
    if not isinstance(base_node, tuple) or not base_node:
        try:
            base_node = get_at(parent_node, hole_path)
        except Exception:
            base_node = None
    if not isinstance(base_node, tuple) or not base_node:
        solver_meta["status"] = "missing_hole_sub"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    base_node = _simplify_node(base_node)

    witness = getattr(subproblem_spec, "witness", None) if subproblem_spec is not None else None
    target_grad_fit = None
    if witness is not None:
        target_grad_fit = _normalize_grad_tensor(getattr(witness, "grad_fit", None), x=problem.xf)
    x_rank, t_rank, target_grad_rank = _subset_rows(
        problem.xf,
        problem.tf,
        grad=target_grad_fit,
        max_rows=32,
    )
    solver_meta["tangent_fit_points_used"] = int(x_rank.shape[0])
    solver_meta["target_gradient_used"] = bool(target_grad_rank is not None)

    need_grad = bool(target_grad_rank is not None)
    base_value_fit, base_grad_fit = _capture_node_value_grad(base_node, x_rank, capture_gradients=need_grad)
    if base_value_fit is None:
        solver_meta["status"] = "invalid_base_prediction"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    if need_grad and base_grad_fit is None:
        solver_meta["target_gradient_used"] = False
        target_grad_rank = None
        need_grad = False

    residual_fit = _ensure_col(t_rank) - _ensure_col(base_value_fit)
    grad_residual_fit = None
    if target_grad_rank is not None and base_grad_fit is not None:
        grad_residual_fit = target_grad_rank - base_grad_fit

    active_vars = ()
    target_dim = problem.target_dim
    if subproblem_spec is not None:
        active_vars = tuple(int(v) for v in tuple(subproblem_spec.active_vars or ()))
        target_dim = subproblem_spec.target_dim

    edit_candidates = _enumerate_tangent_edit_nodes(
        base_node,
        target_dim=target_dim,
        nvars=int(nvars),
        active_vars=active_vars,
        wrappers_left=int(problem.wrappers_left),
        pool_nodes=pool_nodes,
        var_dims=var_dims,
        x_rank=x_rank,
        t_rank=t_rank,
        target_grad_rank=target_grad_rank,
        base_value_fit=base_value_fit,
        base_grad_fit=base_grad_fit,
    )
    solver_meta["candidate_count_generated"] = int(len(edit_candidates))

    ranked_candidates: list[dict[str, Any]] = []
    for candidate in edit_candidates:
        cand_value_fit, cand_grad_fit = _candidate_value_grad(candidate, x_rank, capture_gradients=need_grad)
        if cand_value_fit is None:
            continue
        delta_fit = _ensure_col(cand_value_fit) - _ensure_col(base_value_fit)
        prediction_score = _alignment_score(residual_fit, delta_fit)
        gradient_score = None
        if grad_residual_fit is not None and cand_grad_fit is not None and base_grad_fit is not None:
            gradient_score = _alignment_score(grad_residual_fit, cand_grad_fit - base_grad_fit)
        tangent_score = float(prediction_score + (0.5 * float(gradient_score) if gradient_score is not None else 0.0))
        ranked_candidates.append(
            {
                **candidate,
                "value_fit": cand_value_fit.detach(),
                "grad_fit": None if cand_grad_fit is None else cand_grad_fit.detach(),
                "prediction_score": float(prediction_score),
                "gradient_score": None if gradient_score is None else float(gradient_score),
                "tangent_score": float(tangent_score),
            }
        )

    ranked_candidates.sort(key=_tangent_sort_key)
    solver_meta["candidate_count_ranked"] = int(len(ranked_candidates))

    score_cap = min(len(ranked_candidates), max(12, 3 * route_preview_topk))
    scored_candidates = _select_diverse_candidates(ranked_candidates, score_cap)
    solver_meta["candidate_count_selected_for_scoring"] = int(len(scored_candidates))
    ctx = _build_solver_context(
        parent_node=parent_node,
        hole_path=hole_path,
        hole_sub=base_node,
        max_depth=int(max_depth),
        max_subtree_depth=int(max_subtree_depth if max_subtree_depth is not None else max_depth),
        nvars=int(nvars),
        poly_degree=int(poly_degree),
        var_dims=var_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        local_score_mode=str(mode_name),
        target_mode=str(target_mode or ""),
        target_mapping_kind=str(target_mapping_kind or ""),
        witness_loss_enable=bool(witness_loss_enable),
        witness_grad_weight=float(witness_grad_weight),
        witness_d2_weight=float(witness_d2_weight),
        witness_diag_weight=float(witness_diag_weight),
        witness_physics_weight=float(witness_physics_weight),
    )

    scored_rows: list[_ScoredLocalCandidate] = []
    edit_meta_by_key: dict[str, dict[str, Any]] = {}
    for tangent_rank, candidate in enumerate(scored_candidates):
        scored = _score_node_against_problem(
            candidate["node"],
            problem=problem,
            ctx=ctx,
            source="tangent_edit",
            generation_kind=f"tangent:{str(candidate.get('edit_kind', 'edit'))}",
        )
        if scored is None:
            continue
        payload = dict(scored.payload or {})
        payload["tangent_score"] = float(candidate["tangent_score"])
        payload["prediction_score"] = float(candidate["prediction_score"])
        payload["gradient_score"] = (
            None if candidate.get("gradient_score", None) is None else float(candidate["gradient_score"])
        )
        payload["edit_kind"] = str(candidate.get("edit_kind", "") or "")
        payload["anchor"] = None if candidate.get("anchor", None) is None else node_str(candidate["anchor"])
        scored = replace(
            scored,
            family="tangent_edit",
            payload=payload,
        )
        scored_rows.append(scored)
        edit_meta_by_key[node_str(scored.node)] = {
            "tangent_rank": int(tangent_rank),
            "tangent_score": float(candidate["tangent_score"]),
            "prediction_score": float(candidate["prediction_score"]),
            "gradient_score": (
                None if candidate.get("gradient_score", None) is None else float(candidate["gradient_score"])
            ),
            "edit_kind": str(candidate.get("edit_kind", "") or ""),
            "anchor": None if candidate.get("anchor", None) is None else node_str(candidate["anchor"]),
        }

    deduped = _dedup_scored_candidates(scored_rows, complexity_penalty=0.0)
    solver_meta["candidate_count_scored"] = int(len(deduped))

    preview_rows: list[dict[str, Any]] = []
    for cand in deduped[:route_preview_topk]:
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
        meta = edit_meta_by_key.get(node_str(cand.node), {})
        row["proposal_family"] = "tangent_edit"
        row["generation_source"] = "tangent_edit"
        row["tuple_provenance"] = "tangent_edit"
        row["tangent_edit_rank"] = int(meta.get("tangent_rank", 0))
        row["tangent_edit_score"] = _finite_float(meta.get("tangent_score", None))
        row["tangent_edit_prediction_score"] = _finite_float(meta.get("prediction_score", None))
        row["tangent_edit_gradient_score"] = _finite_float(meta.get("gradient_score", None))
        row["tangent_edit_kind"] = str(meta.get("edit_kind", "") or "")
        row["tangent_edit_anchor"] = meta.get("anchor", None)
        preview_rows.append(row)

    preview_rows.sort(
        key=lambda row: (
            float(row.get("local_probe_mse", float("inf"))),
            float(row.get("local_fit_mse", float("inf"))),
            int(node_size(row.get("expr", ("const", 0.0)))) if isinstance(row.get("expr", None), tuple) else 10**9,
        )
    )
    preview_rows = preview_rows[:route_preview_topk]
    for local_rank, row in enumerate(preview_rows):
        row["local_rank"] = int(local_rank)
        row["local_candidate_count"] = int(len(preview_rows))

    solver_meta["preview_count"] = int(len(preview_rows))
    solver_meta["status"] = "ok" if preview_rows else "no_tangent_edit_candidates"
    solver_meta["wall_seconds"] = float(time.perf_counter() - started)
    return {"rows": preview_rows, "solver_meta": solver_meta}


__all__ = ["solve_local_tangent_edit_preview_rows"]
