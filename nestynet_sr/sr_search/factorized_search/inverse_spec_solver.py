# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Direct-spec local subtree proposal generator for inverse steering."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch

from .expr_ast import (
    BINARY_OPS,
    UNARY_OPS,
    build_pool,
    dim_round,
    dims_eq,
    eval_node,
    node_depth,
    node_dims,
    node_size,
    node_str,
    replace_at,
    simplify,
)
from .expr_enum import enumerate_trees, enumerate_trees_dim
from .expr_mapping import eval_power, fit_best, fit_power
from .inverse_core import (
    _invert_binary_context,
    _invert_unary_context_branches,
    _normalize_inverse_local_score_mode,
    _score_inverse_local_predictions,
    _weighted_mse_cols,
    _weighted_centered_mse,
)
from .inverse_search import _inverse_collect_local_repair_candidates
from .local_teacher_loss import score_local_teacher_loss
from .subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    deserialize_subproblem_spec,
    extract_family_regime_metadata,
    serialize_family_evidence,
    wrap_subproblem_spec_payload,
)
from .subproblem_active_vars import infer_subproblem_active_vars, normalize_active_vars
from .subproblem_teacher import (
    LOCAL_TEACHER_SOURCE_NUMERIC,
    LOCAL_TEACHER_SOURCE_ORACLE,
    LOCAL_TEACHER_SOURCE_RUNTIME,
    LOCAL_TEACHER_SOURCE_SYMBOLIC,
    build_numeric_local_teacher_spec,
    evaluate_local_teacher_jets,
    normalize_local_teacher_source,
    normalize_local_teacher_spec,
)
from .subproblem_tests import (
    build_expanded_family_evidence_bundle,
    build_named_outer_family_evidence,
    default_outer_family_battery_specs,
    family_evidence_should_run,
    family_evidence_status,
    normalize_family_battery_mode,
)


def _ensure_col(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    if x.ndim == 1:
        return x.unsqueeze(-1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"expected [N] or [N,1] tensor, got shape={tuple(x.shape)}")


def _bool_col(mask: torch.Tensor) -> torch.Tensor:
    mm = _ensure_col(mask)
    return mm.to(dtype=torch.bool)


def _mapping_param_count(mapping: Mapping[str, Any] | None) -> int:
    if not isinstance(mapping, Mapping):
        return 0
    count = 0

    def _visit(value: Any) -> None:
        nonlocal count
        if isinstance(value, Mapping):
            for key, inner in value.items():
                if str(key) == "kind":
                    continue
                _visit(inner)
            return
        if isinstance(value, (list, tuple)):
            for inner in value:
                _visit(inner)
            return
        try:
            fv = float(value)
        except Exception:
            return
        if math.isfinite(fv):
            count += 1

    _visit(dict(mapping))
    return int(count)


def _jsonish_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonish_payload(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish_payload(v) for v in value]
    if torch.is_tensor(value):
        if value.ndim == 0:
            try:
                return float(value.item())
            except Exception:
                return None
        return None
    if isinstance(value, bool):
        return bool(value)
    try:
        fv = float(value)
    except Exception:
        return value if value is None or isinstance(value, str) else str(value)
    return float(fv) if math.isfinite(fv) else None


_LOCAL_CONTEXT_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "constant_lift_task",
    "constant_lift_values_by_regime",
    "constant_lift_regime_ids",
    "constant_lift_dataset_metadata",
    "constant_lift_feature_nodes",
    "constant_lift_constant_name",
    "constant_lift_feature_source",
    "constant_lift_min_regimes",
    "constant_lift_trigger_mean_cv",
)


def _is_nonempty_context_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(dict(value))
    if isinstance(value, (list, tuple, set)):
        return bool(list(value))
    if isinstance(value, str):
        return bool(str(value).strip())
    return True


def _merge_local_problem_context(*sources: Any) -> dict[str, Any]:
    merged = extract_family_regime_metadata(*sources)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in _LOCAL_CONTEXT_PASSTHROUGH_KEYS:
            if key not in source:
                continue
            value = source.get(key, None)
            if not _is_nonempty_context_value(value):
                continue
            merged[str(key)] = copy.deepcopy(value)
    return merged


def _stats_add_time(stats: dict[str, Any], key: str, delta_s: float) -> None:
    try:
        delta = float(delta_s)
    except Exception:
        return
    if not math.isfinite(delta) or delta <= 0.0:
        return
    stats[str(key)] = float(stats.get(str(key), 0.0) or 0.0) + delta


def _stats_add_nested_time(stats: dict[str, Any], key: str, subkey: str, delta_s: float) -> None:
    bucket = dict(stats.get(str(key), {}) or {})
    try:
        delta = float(delta_s)
    except Exception:
        delta = None
    if delta is None or not math.isfinite(delta) or delta <= 0.0:
        stats[str(key)] = bucket
        return
    bucket[str(subkey)] = float(bucket.get(str(subkey), 0.0) or 0.0) + delta
    stats[str(key)] = bucket


def _stats_add_nested_count(stats: dict[str, Any], key: str, subkey: str, inc: int = 1) -> None:
    bucket = dict(stats.get(str(key), {}) or {})
    try:
        iv = int(inc)
    except Exception:
        iv = 0
    if iv <= 0:
        stats[str(key)] = bucket
        return
    bucket[str(subkey)] = int(bucket.get(str(subkey), 0) or 0) + iv
    stats[str(key)] = bucket


def _local_mapping_preview(
    node,
    *,
    xf: torch.Tensor,
    tf: torch.Tensor,
    xp: torch.Tensor,
    tp: torch.Tensor,
    poly_degree: int,
    local_score_mode: str,
) -> tuple[str, int]:
    mode_name = _normalize_inverse_local_score_mode(local_score_mode, default="affine")
    if mode_name in ("strict", "direct"):
        return "identity", 0
    if mode_name in ("affine", "lin", "linear"):
        return "affine", 2
    try:
        pred_fit = eval_node(node, xf)
        pred_probe = eval_node(node, xp)
    except Exception:
        return "", 0
    if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
        return "", 0
    fb = fit_best(pred_fit, tf, int(poly_degree))
    if fb is None:
        return "", 0
    _fit_mse, mapping = fb
    return str((mapping or {}).get("kind", "") or ""), int(_mapping_param_count(mapping))


def _preview_sort_key(
    probe_mse: float,
    fit_mse: float,
    node,
    *,
    complexity_penalty: float,
):
    size = int(node_size(node))
    depth = int(node_depth(node))
    penalty = max(0.0, float(complexity_penalty)) * float(size)
    return (
        float(probe_mse) + penalty,
        float(fit_mse) + penalty,
        size,
        depth,
        node_str(node),
    )


def _coerce_legacy_seed_nodes(include_legacy_seed_nodes: Any) -> list[tuple]:
    if include_legacy_seed_nodes is None or include_legacy_seed_nodes is False:
        return []
    if include_legacy_seed_nodes is True:
        return []
    out: list[tuple] = []
    for node in list(include_legacy_seed_nodes or []):
        if isinstance(node, tuple) and node:
            out.append(node)
    return out


def _coerce_weight_col(w: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    rr = _ensure_col(ref)
    if w is None:
        return torch.ones_like(rr)
    ww = _ensure_col(w).to(dtype=rr.dtype, device=rr.device)
    if int(ww.shape[0]) != int(rr.shape[0]):
        return torch.ones_like(rr)
    ww = torch.where(torch.isfinite(ww), ww, torch.zeros_like(ww))
    return torch.clamp(ww, min=0.0)


def _coerce_grad_like(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value.to(dtype=x.dtype, device=x.device)
    else:
        try:
            out = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    if tuple(out.shape) != (int(x.shape[0]), int(x.shape[1])):
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _coerce_hdiag_like(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        raw = value.to(dtype=x.dtype, device=x.device)
    else:
        try:
            raw = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    if raw.ndim == 2 and tuple(raw.shape) == (int(x.shape[0]), int(x.shape[1])):
        out = raw
    elif raw.ndim == 3 and tuple(raw.shape) == (int(x.shape[0]), int(x.shape[1]), int(x.shape[1])):
        out = torch.diagonal(raw, dim1=-2, dim2=-1)
    else:
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _teacher_source_is_exact(source: Any) -> bool:
    token = normalize_local_teacher_source(source, default=LOCAL_TEACHER_SOURCE_NUMERIC)
    return token in {
        LOCAL_TEACHER_SOURCE_ORACLE,
        LOCAL_TEACHER_SOURCE_RUNTIME,
        LOCAL_TEACHER_SOURCE_SYMBOLIC,
    }


def _diagnostic_teacher_source(
    problem: _LocalProblem,
    *,
    split: str,
) -> str:
    diagnostics = dict(problem.diagnostics or {})
    for key in (f"{split}_jet_source", f"witness_{split}_jet_source"):
        token = str(diagnostics.get(key, "") or "").strip()
        if token:
            return normalize_local_teacher_source(token, default=LOCAL_TEACHER_SOURCE_NUMERIC)
    if problem.teacher_spec is not None:
        return normalize_local_teacher_source(
            dict(problem.teacher_spec or {}).get("source", None),
            default=LOCAL_TEACHER_SOURCE_NUMERIC,
        )
    return LOCAL_TEACHER_SOURCE_NUMERIC


def _autograd_node_jets(
    node,
    x: torch.Tensor,
    *,
    capture_d2: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    try:
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            value = _ensure_col(eval_node(node, x_req))
            if not torch.isfinite(value).all():
                return None, None, None
            grad_rows: list[torch.Tensor] = []
            d2_rows: list[torch.Tensor] = []
            n_points = int(x_req.shape[0])
            n_vars = int(x_req.shape[1])
            for idx in range(n_points):
                grad_full = torch.autograd.grad(
                    value[idx, 0],
                    x_req,
                    retain_graph=True,
                    create_graph=bool(capture_d2),
                    allow_unused=True,
                )[0]
                if grad_full is None:
                    grad_i = torch.zeros((n_vars,), dtype=x_req.dtype, device=x_req.device)
                else:
                    grad_i = grad_full[idx]
                grad_rows.append(grad_i)
                if bool(capture_d2):
                    d2_i: list[torch.Tensor] = []
                    for dim in range(n_vars):
                        second_full = torch.autograd.grad(
                            grad_i[dim],
                            x_req,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                        if second_full is None:
                            d2_i.append(torch.zeros((), dtype=x_req.dtype, device=x_req.device))
                        else:
                            d2_i.append(second_full[idx, dim])
                    d2_rows.append(torch.stack(d2_i))
            grad = torch.stack(grad_rows).detach() if grad_rows else None
            d2 = torch.stack(d2_rows).detach() if d2_rows else None
            return value.detach(), grad, d2
    except Exception:
        return None, None, None


def _derived_child_teacher_spec(
    parent_problem: _LocalProblem,
    *,
    transform_kind: str | None,
    transform_op: str | None,
    transform_slot: int | None = None,
) -> Mapping[str, Any] | None:
    if parent_problem.teacher_spec is None:
        return None
    spec = dict(normalize_local_teacher_spec(parent_problem.teacher_spec))
    spec["derivation"] = "recursive_inverse"
    if transform_kind:
        spec["transform_kind"] = str(transform_kind)
    if transform_op:
        spec["transform_op"] = str(transform_op)
    if transform_slot is not None:
        spec["transform_slot"] = int(transform_slot)
    return spec


def _derive_unary_child_jets(
    *,
    parent_target: torch.Tensor,
    parent_grad: torch.Tensor | None,
    parent_d2: torch.Tensor | None,
    child_target: torch.Tensor,
    op: str,
    safe_eps: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    grad = _coerce_grad_like(parent_grad, x=parent_grad) if torch.is_tensor(parent_grad) else None
    if grad is None:
        return None, None
    d2 = _coerce_hdiag_like(parent_d2, x=grad)
    y = _ensure_col(parent_target).to(dtype=grad.dtype, device=grad.device)
    z = _ensure_col(child_target).to(dtype=grad.dtype, device=grad.device)
    eps = float(max(1.0e-12, safe_eps))
    if op == "neg":
        return -grad, None if d2 is None else -d2
    if op == "exp":
        den = torch.clamp(y.abs(), min=eps)
        child_grad = grad / den
        child_d2 = None if d2 is None else (d2 / den) - (grad * grad) / torch.clamp(den * den, min=eps)
        return child_grad, child_d2
    if op == "log":
        child_grad = z * grad
        child_d2 = None if d2 is None else z * (d2 + grad * grad)
        return child_grad, child_d2
    if op == "sqrt":
        child_grad = 2.0 * y * grad
        child_d2 = None if d2 is None else 2.0 * (grad * grad + y * d2)
        return child_grad, child_d2
    if op == "sqr":
        den = torch.clamp((2.0 * z).abs(), min=eps) * torch.sign(torch.where(z.abs() > eps, z, torch.ones_like(z)))
        child_grad = grad / den
        child_d2 = None if d2 is None else (d2 - 2.0 * child_grad * child_grad) / den
        return child_grad, child_d2
    if op == "sin":
        den = torch.cos(z)
        safe = torch.where(den.abs() > eps, den, torch.ones_like(den))
        child_grad = grad / safe
        child_grad = torch.where(den.abs() > eps, child_grad, torch.zeros_like(child_grad))
        child_d2 = None
        if d2 is not None:
            child_d2 = (d2 + torch.sin(z) * child_grad * child_grad) / safe
            child_d2 = torch.where(den.abs() > eps, child_d2, torch.zeros_like(child_d2))
        return child_grad, child_d2
    if op == "cos":
        den = -torch.sin(z)
        safe = torch.where(den.abs() > eps, den, torch.ones_like(den))
        child_grad = grad / safe
        child_grad = torch.where(den.abs() > eps, child_grad, torch.zeros_like(child_grad))
        child_d2 = None
        if d2 is not None:
            child_d2 = (d2 + torch.cos(z) * child_grad * child_grad) / safe
            child_d2 = torch.where(den.abs() > eps, child_d2, torch.zeros_like(child_d2))
        return child_grad, child_d2
    return None, None


def _derive_binary_child_jets(
    *,
    parent_target: torch.Tensor,
    parent_grad: torch.Tensor | None,
    parent_d2: torch.Tensor | None,
    other_target: torch.Tensor,
    other_grad: torch.Tensor | None,
    other_d2: torch.Tensor | None,
    op: str,
    child_slot: int,
    safe_eps: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if parent_grad is None or other_grad is None:
        return None, None
    grad = _coerce_grad_like(parent_grad, x=parent_grad)
    other_g = _coerce_grad_like(other_grad, x=parent_grad)
    if grad is None or other_g is None:
        return None, None
    d2 = _coerce_hdiag_like(parent_d2, x=grad)
    other_d = _coerce_hdiag_like(other_d2, x=grad)
    y = _ensure_col(parent_target).to(dtype=grad.dtype, device=grad.device)
    o = _ensure_col(other_target).to(dtype=grad.dtype, device=grad.device)
    eps = float(max(1.0e-12, safe_eps))
    if op == "add":
        child_grad = grad - other_g
        child_d2 = None if d2 is None or other_d is None else d2 - other_d
        return child_grad, child_d2
    if op == "sub":
        if int(child_slot) == 1:
            child_grad = grad + other_g
            child_d2 = None if d2 is None or other_d is None else d2 + other_d
        else:
            child_grad = other_g - grad
            child_d2 = None if d2 is None or other_d is None else other_d - d2
        return child_grad, child_d2
    if op == "mul":
        den = torch.where(o.abs() > eps, o, torch.ones_like(o))
        child_grad = (grad * den - y * other_g) / torch.clamp(den * den, min=eps)
        child_grad = torch.where(o.abs() > eps, child_grad, torch.zeros_like(child_grad))
        child_d2 = None
        if d2 is not None and other_d is not None:
            child_d2 = (
                d2 / den
                - 2.0 * grad * other_g / torch.clamp(den * den, min=eps)
                - y * other_d / torch.clamp(den * den, min=eps)
                + 2.0 * y * other_g * other_g / torch.clamp(den * den * den, min=eps)
            )
            child_d2 = torch.where(o.abs() > eps, child_d2, torch.zeros_like(child_d2))
        return child_grad, child_d2
    if op == "div":
        if int(child_slot) == 1:
            child_grad = grad * o + y * other_g
            child_d2 = None
            if d2 is not None and other_d is not None:
                child_d2 = d2 * o + 2.0 * grad * other_g + y * other_d
            return child_grad, child_d2
        den = torch.where(y.abs() > eps, y, torch.ones_like(y))
        child_grad = (other_g * den - o * grad) / torch.clamp(den * den, min=eps)
        child_grad = torch.where(y.abs() > eps, child_grad, torch.zeros_like(child_grad))
        child_d2 = None
        if d2 is not None and other_d is not None:
            child_d2 = (
                other_d / den
                - 2.0 * other_g * grad / torch.clamp(den * den, min=eps)
                - o * d2 / torch.clamp(den * den, min=eps)
                + 2.0 * o * grad * grad / torch.clamp(den * den * den, min=eps)
            )
            child_d2 = torch.where(y.abs() > eps, child_d2, torch.zeros_like(child_d2))
        return child_grad, child_d2
    return None, None


def _carry_local_problem_jets(
    problem: _LocalProblem,
    *,
    x: torch.Tensor,
    split: str,
    include_d2: bool,
    teacher_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    grad = _coerce_grad_like(problem.grad_fit if split == "fit" else problem.grad_probe, x=x)
    if grad is None:
        return None
    d2 = _coerce_hdiag_like(problem.d2_fit if split == "fit" else problem.d2_probe, x=x)
    diagnostics = dict(problem.diagnostics or {})
    source = _diagnostic_teacher_source(problem, split=split)
    requested_source = normalize_local_teacher_source(
        diagnostics.get(f"{split}_jet_requested_source", diagnostics.get(f"witness_{split}_jet_requested_source", None)),
        default=normalize_local_teacher_source(
            dict(teacher_spec or {}).get("requested_source", dict(teacher_spec or {}).get("source", None)),
            default=source,
        ),
    )
    status_key = diagnostics.get(
        f"witness_{split}_jet_status",
        diagnostics.get(f"{split}_jet_status", None),
    )
    if not status_key:
        status_key = "carried_forward" if (not include_d2 or d2 is not None) else "carried_forward_missing_d2"
    return {
        "status": str(status_key),
        "source": str(source),
        "requested_source": str(requested_source),
        "teacher_spec": dict(teacher_spec or {}),
        "grad": grad,
        "d2": d2 if bool(include_d2) else None,
        "row_count": int(x.shape[0]),
        "support_count": int(x.shape[0]),
        "neighbor_count": int(x.shape[0]),
        "include_d2": bool(include_d2),
        "failed_rows": 0,
        "fallback_used": bool(diagnostics.get(f"{split}_jet_fallback_used", False)),
    }


def _problem_witness_provenance(problem: _LocalProblem) -> dict[str, Any]:
    diagnostics = dict(problem.diagnostics or {})
    fit_source = _diagnostic_teacher_source(problem, split="fit")
    probe_source = _diagnostic_teacher_source(problem, split="probe")
    fit_requested = normalize_local_teacher_source(
        diagnostics.get("fit_jet_requested_source", diagnostics.get("witness_fit_jet_requested_source", None)),
        default=fit_source,
    )
    probe_requested = normalize_local_teacher_source(
        diagnostics.get("probe_jet_requested_source", diagnostics.get("witness_probe_jet_requested_source", None)),
        default=probe_source,
    )
    fit_fallback = bool(diagnostics.get("fit_jet_fallback_used", False))
    probe_fallback = bool(diagnostics.get("probe_jet_fallback_used", False))
    exact_used = bool(_teacher_source_is_exact(fit_source) or _teacher_source_is_exact(probe_source))
    return {
        "witness_fit_jet_source": str(fit_source),
        "witness_probe_jet_source": str(probe_source),
        "witness_fit_jet_requested_source": str(fit_requested),
        "witness_probe_jet_requested_source": str(probe_requested),
        "witness_fit_jet_fallback_used": bool(fit_fallback),
        "witness_probe_jet_fallback_used": bool(probe_fallback),
        "witness_numeric_jet_fallback_used": bool(fit_fallback or probe_fallback),
        "witness_exact_jet_used": bool(exact_used),
    }


def _dim_add(d1, d2):
    if d1 is None or d2 is None:
        return None
    return dim_round(tuple(float(a) + float(b) for a, b in zip(d1, d2)))


def _dim_sub(d1, d2):
    if d1 is None or d2 is None:
        return None
    return dim_round(tuple(float(a) - float(b) for a, b in zip(d1, d2)))


def _dim_scale(d, factor: float):
    if d is None:
        return None
    return dim_round(tuple(float(factor) * float(v) for v in d))


def _collect_node_var_ids(node, out: set[int]) -> None:
    if not isinstance(node, tuple) or not node:
        return
    op = node[0]
    if op == "var":
        try:
            idx = int(node[1])
        except Exception:
            return
        if idx >= 0:
            out.add(idx)
        return
    if op in ("const", "hparam"):
        return
    if op in UNARY_OPS and len(node) >= 2:
        _collect_node_var_ids(node[1], out)
        return
    if op in BINARY_OPS and len(node) >= 3:
        _collect_node_var_ids(node[1], out)
        _collect_node_var_ids(node[2], out)


def _node_var_ids(node) -> tuple[int, ...]:
    out: set[int] = set()
    _collect_node_var_ids(node, out)
    return tuple(sorted(out))


def _problem_active_vars(problem: _LocalProblem, *, nvars: int) -> tuple[int, ...]:
    diagnostics = dict(problem.diagnostics or {})
    return normalize_active_vars(diagnostics.get("active_vars", ()), nvars=int(max(0, nvars)))


def _generation_consumes_wrapper_budget(generation_kind: str | None) -> bool:
    token = str(generation_kind or "").strip().lower()
    return ":wrap:" in token or token.startswith("wrap:")


def _hard_constraint_violation_reason(
    node,
    *,
    problem: _LocalProblem,
    var_dims,
    nvars: int,
    generation_kind: str | None = None,
) -> str | None:
    if not isinstance(node, tuple) or not node:
        return "invalid_node"
    active_vars = _problem_active_vars(problem, nvars=int(nvars))
    if active_vars:
        allowed = set(int(v) for v in tuple(active_vars or ()))
        used = set(_node_var_ids(node))
        if not used.issubset(allowed):
            return "active_vars"
    if _generation_consumes_wrapper_budget(generation_kind) and int(problem.wrappers_left) <= 0:
        return "wrappers_left"
    if var_dims is not None:
        try:
            nd = node_dims(node, var_dims)
        except Exception:
            nd = None
        if nd is None:
            return "target_dim"
        if problem.target_dim is not None and not dims_eq(nd, problem.target_dim):
            return "target_dim"
    return None


def _proposal_satisfies_hard_constraints(
    node,
    *,
    problem: _LocalProblem,
    var_dims,
    nvars: int,
    generation_kind: str | None = None,
) -> bool:
    return (
        _hard_constraint_violation_reason(
            node,
            problem=problem,
            var_dims=var_dims,
            nvars=int(nvars),
            generation_kind=generation_kind,
        )
        is None
    )


@dataclass(frozen=True)
class _LocalProblem:
    xf: torch.Tensor
    tf: torch.Tensor
    wf: torch.Tensor | None
    xp: torch.Tensor
    tp: torch.Tensor
    wp: torch.Tensor | None
    target_dim: Any
    confidence: float
    valid_frac: float
    wrappers_left: int
    recursion_level: int
    trace: tuple[str, ...]
    grad_fit: torch.Tensor | None = None
    grad_probe: torch.Tensor | None = None
    d2_fit: torch.Tensor | None = None
    d2_probe: torch.Tensor | None = None
    teacher_spec: Mapping[str, Any] | None = None
    teacher_runtime: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ScoredLocalCandidate:
    node: tuple
    local_probe_mse: float
    local_fit_mse: float
    source: str
    generation_kind: str
    recursion_depth: int
    confidence: float
    valid_frac: float
    trace: tuple[str, ...]
    family: str = ""
    payload: Mapping[str, Any] | None = None
    surrogate_probe_mse: float | None = None
    surrogate_fit_mse: float | None = None
    value_probe_mse: float | None = None
    value_fit_mse: float | None = None
    calibration_gap: float | None = None
    witness_value_loss: float | None = None
    witness_grad_loss: float | None = None
    witness_d2_loss: float | None = None
    witness_diag_loss: float | None = None
    witness_physics_loss: float | None = None
    witness_energy_total: float | None = None
    witness_fit_jet_source: str = ""
    witness_probe_jet_source: str = ""
    witness_fit_jet_requested_source: str = ""
    witness_probe_jet_requested_source: str = ""
    witness_fit_jet_fallback_used: bool = False
    witness_probe_jet_fallback_used: bool = False
    witness_numeric_jet_fallback_used: bool = False
    witness_exact_jet_used: bool = False


@dataclass
class _SolverContext:
    parent_node: tuple
    hole_path: tuple[int, ...]
    hole_sub: tuple
    max_depth: int
    nvars: int
    poly_degree: int
    var_dims: Sequence[Sequence[float]] | None
    pool_nodes: list[tuple]
    pool_dims: list[Any]
    seed_nodes: list[tuple]
    local_score_mode: str
    enum_max_depth: int
    enum_max_trees: int
    max_subtree_depth: int
    preview_topk: int
    complexity_penalty: float
    recursive_enable: bool
    recursive_max_depth: int
    recursive_trigger_rel_mse: float
    recursive_seed_cap: int
    recursive_branch_topk: int
    recursive_child_topk: int
    safe_eps: float
    confidence_mode: str
    confidence_target_gain: float
    confidence_floor: float
    branch_beam_width: int
    min_valid_frac: float
    min_confidence: float
    allow_legacy_aux: bool
    legacy_aux_kwargs: dict[str, Any]
    stats: dict[str, Any]
    target_mode: str = ""
    target_mapping_kind: str = ""
    family_battery_enable: bool = False
    family_battery_mode: str = "outer"
    witness_jets_enable: bool = False
    witness_d2_enable: bool = False
    witness_max_rows: int = 64
    witness_loss_enable: bool = False
    witness_grad_weight: float = 0.0
    witness_d2_weight: float = 0.0
    witness_diag_weight: float = 0.0
    witness_physics_weight: float = 0.0
    active_var_screen_enable: bool = False
    active_var_grad_tol: float = 1.0e-3
    active_var_max_count: int = 4
    memo_table: dict[Any, Any] = field(default_factory=dict)

    @property
    def dm(self) -> bool:
        return self.var_dims is not None


@dataclass(frozen=True)
class _RecursiveBranch:
    source: str
    child_problem: _LocalProblem
    child_target_dim: Any
    wrap_kind: str
    op: str
    slot: int
    anchor_node: tuple | None
    priority: tuple[float, float, int, str]

    def wrap(self, child_node):
        if self.wrap_kind == "unary":
            return (self.op, child_node)
        if int(self.slot) == 1:
            return (self.op, child_node, self.anchor_node)
        return (self.op, self.anchor_node, child_node)


@dataclass(frozen=True)
class _OuterFamilySpec:
    name: str
    min_improvement_ratio: float
    precheck_max_seeds: int


@dataclass(frozen=True)
class _OuterFamilyScore:
    fit_mse: float
    probe_mse: float
    payload: Mapping[str, Any] | None = None


def _candidate_depth_limit(problem: _LocalProblem, ctx: _SolverContext) -> int:
    # Subtree depth limit is independent of enum_max_depth.
    # max_subtree_depth is the hard cap on emitted subtrees;
    # enum_max_depth only controls the flat enumerator's reach.
    return max(1, int(ctx.max_subtree_depth))


def _simplify_node(node):
    try:
        return simplify(node)
    except Exception:
        return node


def _final_replacement_ok(node, *, ctx: _SolverContext) -> bool:
    try:
        repaired = _simplify_node(replace_at(ctx.parent_node, ctx.hole_path, node))
    except Exception:
        return False
    if int(node_depth(repaired)) > int(ctx.max_depth):
        return False
    if ctx.dm:
        try:
            repaired_dim = node_dims(repaired, ctx.var_dims)
        except Exception:
            repaired_dim = None
        if repaired_dim is None:
            return False
    return True


def _dedup_scored_candidates(
    rows: Sequence[_ScoredLocalCandidate],
    *,
    complexity_penalty: float,
) -> list[_ScoredLocalCandidate]:
    best: dict[str, tuple[tuple[Any, ...], _ScoredLocalCandidate]] = {}
    for row in rows:
        key = node_str(row.node)
        sort_key = _preview_sort_key(
            float(row.local_probe_mse),
            float(row.local_fit_mse),
            row.node,
            complexity_penalty=float(complexity_penalty),
        ) + (
            int(row.recursion_depth),
            str(row.generation_kind),
            str(row.source),
            "|".join(row.trace),
        )
        cur = best.get(key)
        if cur is None or sort_key < cur[0]:
            best[key] = (sort_key, row)
    out = [item[1] for item in best.values()]
    out.sort(
        key=lambda row: _preview_sort_key(
            float(row.local_probe_mse),
            float(row.local_fit_mse),
            row.node,
            complexity_penalty=float(complexity_penalty),
        ) + (
            int(row.recursion_depth),
            str(row.generation_kind),
            str(row.source),
            "|".join(row.trace),
        )
    )
    return out


def _score_node_against_problem(
    node,
    *,
    problem: _LocalProblem,
    ctx: _SolverContext,
    source: str,
    generation_kind: str,
    confidence: float | None = None,
    valid_frac: float | None = None,
    trace: Sequence[str] | None = None,
) -> _ScoredLocalCandidate | None:
    started = time.perf_counter()
    if not isinstance(node, tuple) or not node:
        return None
    try:
        witness_provenance = _problem_witness_provenance(problem)
        node = _simplify_node(node)
        if int(node_depth(node)) > _candidate_depth_limit(problem, ctx):
            return None
        if not _proposal_satisfies_hard_constraints(
            node,
            problem=problem,
            var_dims=ctx.var_dims if ctx.dm else None,
            nvars=int(ctx.nvars),
            generation_kind=generation_kind,
        ):
            return None
        try:
            pred_fit = eval_node(node, problem.xf)
            pred_probe = eval_node(node, problem.xp)
        except Exception:
            return None
        if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
            return None
        teacher_loss = None
        if bool(ctx.witness_loss_enable):
            teacher_loss = score_local_teacher_loss(
                node,
                pred_fit=pred_fit,
                pred_probe=pred_probe,
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
                poly_degree=int(ctx.poly_degree),
                mode=str(ctx.local_score_mode),
                grad_weight=float(ctx.witness_grad_weight),
                d2_weight=float(ctx.witness_d2_weight),
                diag_weight=float(ctx.witness_diag_weight),
                physics_weight=float(ctx.witness_physics_weight),
                target_diagnostics=problem.diagnostics,
            )
        if teacher_loss is not None:
            local_fit_mse = float(teacher_loss.fit_total)
            local_probe_mse = float(teacher_loss.probe_total)
            value_fit_mse = float(teacher_loss.value_fit_loss)
            value_probe_mse = float(teacher_loss.value_probe_loss)
            witness_grad_loss = (
                None if teacher_loss.grad_probe_loss is None else float(teacher_loss.grad_probe_loss)
            )
            witness_d2_loss = None if teacher_loss.d2_probe_loss is None else float(teacher_loss.d2_probe_loss)
            witness_diag_loss = (
                None if teacher_loss.diag_probe_loss is None else float(teacher_loss.diag_probe_loss)
            )
            witness_physics_loss = (
                None if teacher_loss.physics_probe_loss is None else float(teacher_loss.physics_probe_loss)
            )
            witness_energy_total = float(teacher_loss.probe_total)
            witness_fit_jet_source = str(teacher_loss.fit_jet_source or witness_provenance["witness_fit_jet_source"])
            witness_probe_jet_source = str(teacher_loss.probe_jet_source or witness_provenance["witness_probe_jet_source"])
            witness_fit_jet_requested_source = str(
                teacher_loss.fit_jet_requested_source or witness_provenance["witness_fit_jet_requested_source"]
            )
            witness_probe_jet_requested_source = str(
                teacher_loss.probe_jet_requested_source or witness_provenance["witness_probe_jet_requested_source"]
            )
            witness_fit_jet_fallback_used = bool(teacher_loss.fit_jet_fallback_used)
            witness_probe_jet_fallback_used = bool(teacher_loss.probe_jet_fallback_used)
            witness_numeric_jet_fallback_used = bool(
                witness_fit_jet_fallback_used or witness_probe_jet_fallback_used
            )
            witness_exact_jet_used = bool(teacher_loss.exact_jet_used)
            calibration_gap = max(
                0.0,
                float(teacher_loss.probe_total) - float(teacher_loss.value_probe_loss),
            )
        else:
            local_score = _score_inverse_local_predictions(
                pred_fit,
                pred_probe,
                problem.tf,
                problem.tp,
                w_fit=problem.wf,
                w_probe=problem.wp,
                poly_degree=int(ctx.poly_degree),
                mode=str(ctx.local_score_mode),
            )
            if local_score is None:
                return None
            local_fit_mse, local_probe_mse = local_score
            value_fit_mse = float(local_fit_mse)
            value_probe_mse = float(local_probe_mse)
            witness_grad_loss = None
            witness_d2_loss = None
            witness_diag_loss = None
            witness_physics_loss = None
            witness_energy_total = None
            witness_fit_jet_source = str(witness_provenance["witness_fit_jet_source"])
            witness_probe_jet_source = str(witness_provenance["witness_probe_jet_source"])
            witness_fit_jet_requested_source = str(witness_provenance["witness_fit_jet_requested_source"])
            witness_probe_jet_requested_source = str(witness_provenance["witness_probe_jet_requested_source"])
            witness_fit_jet_fallback_used = bool(witness_provenance["witness_fit_jet_fallback_used"])
            witness_probe_jet_fallback_used = bool(witness_provenance["witness_probe_jet_fallback_used"])
            witness_numeric_jet_fallback_used = bool(witness_provenance["witness_numeric_jet_fallback_used"])
            witness_exact_jet_used = bool(witness_provenance["witness_exact_jet_used"])
            calibration_gap = 0.0
        return _ScoredLocalCandidate(
            node=node,
            local_probe_mse=float(local_probe_mse),
            local_fit_mse=float(local_fit_mse),
            source=str(source),
            generation_kind=str(generation_kind),
            recursion_depth=int(problem.recursion_level),
            confidence=float(problem.confidence if confidence is None else confidence),
            valid_frac=float(problem.valid_frac if valid_frac is None else valid_frac),
            trace=tuple(str(v) for v in (trace or problem.trace)),
            family="",
            payload=None,
            surrogate_probe_mse=float(value_probe_mse),
            surrogate_fit_mse=float(value_fit_mse),
            value_probe_mse=float(value_probe_mse),
            value_fit_mse=float(value_fit_mse),
            calibration_gap=float(calibration_gap),
            witness_value_loss=float(value_probe_mse),
            witness_grad_loss=witness_grad_loss,
            witness_d2_loss=witness_d2_loss,
            witness_diag_loss=witness_diag_loss,
            witness_physics_loss=witness_physics_loss,
            witness_energy_total=witness_energy_total,
            witness_fit_jet_source=str(witness_fit_jet_source),
            witness_probe_jet_source=str(witness_probe_jet_source),
            witness_fit_jet_requested_source=str(witness_fit_jet_requested_source),
            witness_probe_jet_requested_source=str(witness_probe_jet_requested_source),
            witness_fit_jet_fallback_used=bool(witness_fit_jet_fallback_used),
            witness_probe_jet_fallback_used=bool(witness_probe_jet_fallback_used),
            witness_numeric_jet_fallback_used=bool(witness_numeric_jet_fallback_used),
            witness_exact_jet_used=bool(witness_exact_jet_used),
        )
    finally:
        elapsed = time.perf_counter() - started
        _stats_add_time(ctx.stats, "score_node_total_wall_seconds", elapsed)
        _stats_add_nested_time(ctx.stats, "score_node_generation_wall_seconds", str(generation_kind), elapsed)
        _stats_add_nested_count(ctx.stats, "score_node_generation_counts", str(generation_kind), 1)
        _stats_add_nested_time(ctx.stats, "score_node_source_wall_seconds", str(source), elapsed)
        _stats_add_nested_count(ctx.stats, "score_node_source_counts", str(source), 1)


def _collect_flat_candidate_sources(
    problem: _LocalProblem,
    *,
    ctx: _SolverContext,
    include_legacy_aux: bool,
) -> tuple[list[tuple[str, tuple]], dict[str, int], int, int]:
    seen_nodes: set[str] = set()
    source_counts: dict[str, int] = {}
    candidates: list[tuple[str, tuple]] = []

    def _add(node, source: str) -> None:
        if not isinstance(node, tuple) or not node:
            return
        simp = _simplify_node(node)
        if int(node_depth(simp)) > _candidate_depth_limit(problem, ctx):
            return
        if ctx.dm:
            try:
                nd = node_dims(simp, ctx.var_dims)
            except Exception:
                nd = None
            if nd is None or problem.target_dim is None or not dims_eq(nd, problem.target_dim):
                return
        key = node_str(simp)
        if key in seen_nodes:
            return
        seen_nodes.add(key)
        candidates.append((str(source), simp))
        source_counts[str(source)] = int(source_counts.get(str(source), 0) + 1)

    enum_depth_limit = max(1, min(int(ctx.enum_max_depth), _candidate_depth_limit(problem, ctx)))
    enum_tree_limit = max(32, int(ctx.enum_max_trees) // max(1, int(problem.recursion_level) + 1))
    enum_nodes = []
    enum_depth_reached = 0
    try:
        if ctx.dm:
            enum_nodes, enum_depth_reached = enumerate_trees_dim(
                enum_depth_limit,
                int(ctx.nvars),
                ctx.var_dims,
                problem.target_dim,
                max_trees=enum_tree_limit,
            )
        else:
            enum_nodes, enum_depth_reached = enumerate_trees(
                enum_depth_limit,
                int(ctx.nvars),
                max_trees=enum_tree_limit,
            )
    except Exception:
        enum_nodes = []
        enum_depth_reached = 0
    for node in enum_nodes:
        _add(node, "enum")

    for idx, node in enumerate(ctx.pool_nodes):
        if ctx.dm and idx < len(ctx.pool_dims):
            nd = ctx.pool_dims[idx]
            if nd is None or problem.target_dim is None or not dims_eq(nd, problem.target_dim):
                continue
        _add(node, "pool")

    for node in ctx.seed_nodes:
        _add(node, "seed")

    if include_legacy_aux:
        legacy_aux_nodes = []
        try:
            legacy_aux_nodes = _inverse_collect_local_repair_candidates(
                parent_node=ctx.parent_node,
                path=ctx.hole_path,
                sub=ctx.hole_sub,
                target_dim=problem.target_dim,
                xf=problem.xf,
                tf=problem.tf,
                xp=problem.xp,
                tp=problem.tp,
                wf=problem.wf,
                wp=problem.wp,
                mfit=torch.ones((int(problem.xf.shape[0]),), dtype=torch.bool, device=problem.xf.device),
                mprobe=torch.ones((int(problem.xp.shape[0]),), dtype=torch.bool, device=problem.xp.device),
                pool_nodes=ctx.pool_nodes,
                pool_dims=ctx.pool_dims,
                pool_phi_fit=torch.zeros(
                    (int(problem.xf.shape[0]), max(1, len(ctx.pool_nodes))),
                    dtype=problem.xf.dtype,
                    device=problem.xf.device,
                ),
                pool_phi_probe=torch.zeros(
                    (int(problem.xp.shape[0]), max(1, len(ctx.pool_nodes))),
                    dtype=problem.xp.dtype,
                    device=problem.xp.device,
                ),
                idxs=[],
                poly_degree=int(ctx.poly_degree),
                local_mode=str(ctx.local_score_mode),
                topk_terms=1,
                shortlist_mult=1,
                safe_eps=float(ctx.safe_eps),
                var_dims=ctx.var_dims if ctx.dm else None,
                max_depth=int(ctx.max_depth),
                micro_search_enable=False,
            )
        except Exception:
            legacy_aux_nodes = []
        for node in list(legacy_aux_nodes or [])[:4]:
            _add(node, "legacy_aux")

    return candidates, source_counts, int(len(enum_nodes)), int(enum_depth_reached)


def _flat_solve_local_problem(
    problem: _LocalProblem,
    *,
    ctx: _SolverContext,
    include_legacy_aux: bool,
) -> tuple[list[_ScoredLocalCandidate], dict[str, Any]]:
    started = time.perf_counter()
    ctx.stats["flat_call_count"] = int(ctx.stats.get("flat_call_count", 0) or 0) + 1
    collect_t0 = time.perf_counter()
    candidate_sources, source_counts, enum_tree_count, enum_depth_reached = _collect_flat_candidate_sources(
        problem,
        ctx=ctx,
        include_legacy_aux=bool(include_legacy_aux),
    )
    _stats_add_time(ctx.stats, "flat_collect_wall_seconds", time.perf_counter() - collect_t0)
    ctx.stats["enum_tree_count"] = int(ctx.stats.get("enum_tree_count", 0) or 0) + int(enum_tree_count)
    ctx.stats["enum_depth_reached"] = max(
        int(ctx.stats.get("enum_depth_reached", 0) or 0),
        int(enum_depth_reached),
    )
    ctx.stats["candidate_count_raw"] = int(ctx.stats.get("candidate_count_raw", 0) or 0) + int(len(candidate_sources))
    global_sources = dict(ctx.stats.get("candidate_source_counts", {}) or {})
    for key, value in source_counts.items():
        global_sources[str(key)] = int(global_sources.get(str(key), 0) + int(value))
    ctx.stats["candidate_source_counts"] = global_sources

    scored_rows: list[_ScoredLocalCandidate] = []
    for source, cand_sub in candidate_sources:
        scored = _score_node_against_problem(
            cand_sub,
            problem=problem,
            ctx=ctx,
            source=str(source),
            generation_kind="flat",
        )
        if scored is not None:
            scored_rows.append(scored)
    ctx.stats["candidate_count_scored"] = int(ctx.stats.get("candidate_count_scored", 0) or 0) + int(len(scored_rows))

    scored_rows = _dedup_scored_candidates(scored_rows, complexity_penalty=float(ctx.complexity_penalty))
    internal_keep = max(
        int(ctx.preview_topk),
        int(ctx.recursive_seed_cap),
        int(ctx.recursive_branch_topk),
        int(ctx.recursive_child_topk),
    )
    scored_rows = scored_rows[:internal_keep]
    flat_meta = {
        "candidate_count_raw": int(len(candidate_sources)),
        "candidate_count_scored": int(len(scored_rows)),
        "enum_tree_count": int(enum_tree_count),
        "enum_depth_reached": int(enum_depth_reached),
        "candidate_source_counts": dict(sorted(source_counts.items())),
    }
    _stats_add_time(ctx.stats, "flat_solve_wall_seconds", time.perf_counter() - started)
    return scored_rows, flat_meta


def _should_recurse(
    problem: _LocalProblem,
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
) -> tuple[bool, float, float]:
    baseline = max(1.0e-12, float(_weighted_centered_mse(problem.tp, problem.wp)))
    best_probe = float(flat_rows[0].local_probe_mse) if flat_rows else float("inf")
    rel_mse = float(best_probe / baseline) if math.isfinite(best_probe) else float("inf")
    if not bool(ctx.recursive_enable):
        return False, best_probe, rel_mse
    if int(problem.wrappers_left) <= 0:
        return False, best_probe, rel_mse
    if float(problem.confidence) < float(ctx.min_confidence):
        return False, best_probe, rel_mse
    if float(problem.valid_frac) < float(ctx.min_valid_frac):
        return False, best_probe, rel_mse
    return bool(rel_mse > float(ctx.recursive_trigger_rel_mse)), best_probe, rel_mse


def _masked_child_weight(
    parent_weight: torch.Tensor | None,
    step_weight: torch.Tensor | None,
    *,
    ref: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    base = _coerce_weight_col(parent_weight, ref)
    if step_weight is None:
        step = torch.ones_like(base)
    else:
        step = _coerce_weight_col(step_weight, ref)
    mm = _bool_col(mask).squeeze(-1)
    out = torch.clamp(base * step, min=0.0)
    return out[mm]


def _build_child_problem(
    parent_problem: _LocalProblem,
    *,
    child_target_fit: torch.Tensor,
    fit_mask: torch.Tensor,
    fit_weight: torch.Tensor | None,
    fit_conf: float,
    child_target_probe: torch.Tensor,
    probe_mask: torch.Tensor,
    probe_weight: torch.Tensor | None,
    probe_conf: float,
    child_target_dim,
    trace_token: str,
    ctx: _SolverContext,
    transform_kind: str | None = None,
    transform_op: str | None = None,
    transform_slot: int | None = None,
    other_fit: torch.Tensor | None = None,
    other_probe: torch.Tensor | None = None,
    other_grad_fit: torch.Tensor | None = None,
    other_grad_probe: torch.Tensor | None = None,
    other_d2_fit: torch.Tensor | None = None,
    other_d2_probe: torch.Tensor | None = None,
) -> _LocalProblem | None:
    fit_mask_col = _bool_col(fit_mask)
    probe_mask_col = _bool_col(probe_mask)
    nfit = int(fit_mask_col.sum().item())
    nprobe = int(probe_mask_col.sum().item())
    if nfit < 4 or nprobe < 4:
        return None
    fit_valid = float(nfit / max(1, int(parent_problem.xf.shape[0])))
    probe_valid = float(nprobe / max(1, int(parent_problem.xp.shape[0])))
    valid_frac = min(fit_valid, probe_valid)
    step_conf = math.sqrt(
        max(1.0e-12, float(fit_conf)) * max(1.0e-12, float(probe_conf))
    )
    child_conf = min(1.0, max(0.0, float(parent_problem.confidence) * step_conf))
    if valid_frac < float(ctx.min_valid_frac):
        return None
    if child_conf < float(ctx.min_confidence):
        return None
    if ctx.dm and child_target_dim is None:
        return None
    fit_keep = fit_mask_col.squeeze(-1)
    probe_keep = probe_mask_col.squeeze(-1)
    child_xf = parent_problem.xf[fit_keep]
    child_tf = _ensure_col(child_target_fit)[fit_keep]
    child_wf = _masked_child_weight(parent_problem.wf, fit_weight, ref=child_target_fit, mask=fit_mask_col)
    child_xp = parent_problem.xp[probe_keep]
    child_tp = _ensure_col(child_target_probe)[probe_keep]
    child_wp = _masked_child_weight(parent_problem.wp, probe_weight, ref=child_target_probe, mask=probe_mask_col)
    child_grad_fit = None
    child_grad_probe = None
    child_d2_fit = None
    child_d2_probe = None
    fit_source = _diagnostic_teacher_source(parent_problem, split="fit")
    probe_source = _diagnostic_teacher_source(parent_problem, split="probe")
    use_exact_fit = _teacher_source_is_exact(fit_source) and parent_problem.grad_fit is not None
    use_exact_probe = _teacher_source_is_exact(probe_source) and parent_problem.grad_probe is not None
    if str(transform_kind or "") == "unary":
        if use_exact_fit:
            full_grad_fit, full_d2_fit = _derive_unary_child_jets(
                parent_target=parent_problem.tf,
                parent_grad=parent_problem.grad_fit,
                parent_d2=parent_problem.d2_fit,
                child_target=child_target_fit,
                op=str(transform_op or ""),
                safe_eps=float(ctx.safe_eps),
            )
            if full_grad_fit is not None:
                child_grad_fit = full_grad_fit[fit_keep]
            if full_d2_fit is not None:
                child_d2_fit = full_d2_fit[fit_keep]
        if use_exact_probe:
            full_grad_probe, full_d2_probe = _derive_unary_child_jets(
                parent_target=parent_problem.tp,
                parent_grad=parent_problem.grad_probe,
                parent_d2=parent_problem.d2_probe,
                child_target=child_target_probe,
                op=str(transform_op or ""),
                safe_eps=float(ctx.safe_eps),
            )
            if full_grad_probe is not None:
                child_grad_probe = full_grad_probe[probe_keep]
            if full_d2_probe is not None:
                child_d2_probe = full_d2_probe[probe_keep]
    elif str(transform_kind or "") == "binary":
        if use_exact_fit and other_fit is not None and other_grad_fit is not None:
            full_grad_fit, full_d2_fit = _derive_binary_child_jets(
                parent_target=parent_problem.tf,
                parent_grad=parent_problem.grad_fit,
                parent_d2=parent_problem.d2_fit,
                other_target=other_fit,
                other_grad=other_grad_fit,
                other_d2=other_d2_fit,
                op=str(transform_op or ""),
                child_slot=int(transform_slot or 0),
                safe_eps=float(ctx.safe_eps),
            )
            if full_grad_fit is not None:
                child_grad_fit = full_grad_fit[fit_keep]
            if full_d2_fit is not None:
                child_d2_fit = full_d2_fit[fit_keep]
        if use_exact_probe and other_probe is not None and other_grad_probe is not None:
            full_grad_probe, full_d2_probe = _derive_binary_child_jets(
                parent_target=parent_problem.tp,
                parent_grad=parent_problem.grad_probe,
                parent_d2=parent_problem.d2_probe,
                other_target=other_probe,
                other_grad=other_grad_probe,
                other_d2=other_d2_probe,
                op=str(transform_op or ""),
                child_slot=int(transform_slot or 0),
                safe_eps=float(ctx.safe_eps),
            )
            if full_grad_probe is not None:
                child_grad_probe = full_grad_probe[probe_keep]
            if full_d2_probe is not None:
                child_d2_probe = full_d2_probe[probe_keep]
    diagnostics = dict(parent_problem.diagnostics or {})
    if child_grad_fit is not None:
        diagnostics["fit_jet_source"] = str(fit_source)
        diagnostics["fit_jet_fallback_used"] = False
        diagnostics["witness_fit_jet_status"] = "propagated_exact_child"
        diagnostics["witness_fit_jet_source"] = str(fit_source)
    if child_grad_probe is not None:
        diagnostics["probe_jet_source"] = str(probe_source)
        diagnostics["probe_jet_fallback_used"] = False
        diagnostics["witness_probe_jet_status"] = "propagated_exact_child"
        diagnostics["witness_probe_jet_source"] = str(probe_source)
    if child_d2_fit is None and child_grad_fit is not None:
        diagnostics["witness_fit_jet_status"] = "propagated_exact_child_missing_d2"
    if child_d2_probe is None and child_grad_probe is not None:
        diagnostics["witness_probe_jet_status"] = "propagated_exact_child_missing_d2"
    child_teacher_spec = _derived_child_teacher_spec(
        parent_problem,
        transform_kind=transform_kind,
        transform_op=transform_op,
        transform_slot=transform_slot,
    )
    return _LocalProblem(
        xf=child_xf,
        tf=child_tf,
        wf=child_wf,
        xp=child_xp,
        tp=child_tp,
        wp=child_wp,
        target_dim=child_target_dim,
        confidence=float(child_conf),
        valid_frac=float(valid_frac),
        wrappers_left=max(0, int(parent_problem.wrappers_left) - 1),
        recursion_level=int(parent_problem.recursion_level) + 1,
        trace=tuple(list(parent_problem.trace) + [str(trace_token)]),
        grad_fit=child_grad_fit,
        grad_probe=child_grad_probe,
        d2_fit=child_d2_fit,
        d2_probe=child_d2_probe,
        teacher_spec=child_teacher_spec,
        diagnostics=diagnostics,
    )


def _unary_child_dim(op: str, target_dim, *, ctx: _SolverContext):
    if op == "neg":
        return target_dim
    if not ctx.dm:
        return None
    dim0 = (0.0,) * len(ctx.var_dims[0])
    if op in ("sin", "cos", "exp", "log"):
        if target_dim is None or not dims_eq(target_dim, dim0):
            return None
        return dim0
    if op == "sqrt":
        return _dim_scale(target_dim, 2.0)
    if op == "sqr":
        return _dim_scale(target_dim, 0.5)
    return None


def _binary_child_dim(op: str, child_slot: int, target_dim, anchor_dim):
    if op in ("add", "sub"):
        return target_dim
    if op == "mul":
        return _dim_sub(target_dim, anchor_dim)
    if op == "div":
        if int(child_slot) == 1:
            return _dim_add(target_dim, anchor_dim)
        return _dim_sub(anchor_dim, target_dim)
    return None


_PERIODIC_OPS = frozenset({"sin", "cos"})

# Confidence threshold below which a trig unary branch is considered unreliable
# and the periodic forward scorer should be tried instead.
_PERIODIC_CONFIDENCE_THRESHOLD = 0.39
_PERIODIC_PRECHECK_MAX_SEEDS = 6
_PERIODIC_PRECHECK_MIN_IMPROVEMENT_RATIO = 0.90
_OUTER_FAMILY_PRECHECK_DEFAULT_RATIO = 0.95
_OUTER_FAMILY_EXP_B_MAX = 5.0
_OUTER_FAMILY_EXP_N_B = 40
_OUTER_FAMILY_EXP_N_REFINE = 4
_OUTER_FAMILY_RATIONAL_DEN_EPS = 1.0e-5
_OUTER_FAMILY_SPECS: tuple[_OuterFamilySpec, ...] = (
    _OuterFamilySpec(
        name="periodic",
        min_improvement_ratio=float(_PERIODIC_PRECHECK_MIN_IMPROVEMENT_RATIO),
        precheck_max_seeds=int(_PERIODIC_PRECHECK_MAX_SEEDS),
    ),
    _OuterFamilySpec(
        name="exp",
        min_improvement_ratio=float(_OUTER_FAMILY_PRECHECK_DEFAULT_RATIO),
        precheck_max_seeds=4,
    ),
    _OuterFamilySpec(
        name="power",
        min_improvement_ratio=float(_OUTER_FAMILY_PRECHECK_DEFAULT_RATIO),
        precheck_max_seeds=4,
    ),
    _OuterFamilySpec(
        name="rational",
        min_improvement_ratio=float(_OUTER_FAMILY_PRECHECK_DEFAULT_RATIO),
        precheck_max_seeds=4,
    ),
)


def _dim0(ctx: _SolverContext):
    if not ctx.dm:
        return None
    return (0.0,) * len(ctx.var_dims[0])


def _const_node(value: float):
    vv = float(value)
    if not math.isfinite(vv):
        raise ValueError("non-finite const")
    if abs(vv) < 1.0e-12:
        vv = 0.0
    return ("const", vv)


def _mul_const(node, coeff: float):
    cc = float(coeff)
    if not math.isfinite(cc):
        raise ValueError("non-finite coeff")
    if abs(cc) < 1.0e-12:
        return _const_node(0.0)
    if abs(cc - 1.0) < 1.0e-12:
        return node
    if abs(cc + 1.0) < 1.0e-12:
        return _simplify_node(("neg", node))
    return _simplify_node(("mul", _const_node(cc), node))


def _add_const(node, bias: float):
    bb = float(bias)
    if not math.isfinite(bb):
        raise ValueError("non-finite bias")
    if abs(bb) < 1.0e-12:
        return node
    return _simplify_node(("add", node, _const_node(bb)))


def _outer_family_child_target_dim(
    family_name: str,
    problem: _LocalProblem,
    *,
    ctx: _SolverContext,
):
    if not ctx.dm:
        return None
    dim0 = _dim0(ctx)
    if family_name in ("periodic", "exp", "power", "rational"):
        if problem.target_dim is None or not dims_eq(problem.target_dim, dim0):
            return None
        return dim0
    return None


def _outer_family_candidate_nodes(
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    problem: _LocalProblem,
    ctx: _SolverContext,
    max_seeds: int,
    child_target_dim,
) -> list[tuple]:
    seen: set[str] = set()
    out: list[tuple] = []

    def _add(node) -> None:
        if not isinstance(node, tuple) or not node:
            return
        node_s = _simplify_node(node)
        key = node_str(node_s)
        if key in seen:
            return
        if int(node_depth(node_s)) > _candidate_depth_limit(problem, ctx):
            return
        if ctx.dm:
            try:
                nd = node_dims(node_s, ctx.var_dims)
            except Exception:
                nd = None
            if nd is None or child_target_dim is None or not dims_eq(nd, child_target_dim):
                return
        seen.add(key)
        out.append(node_s)

    for row in list(flat_rows or [])[: max(1, int(max_seeds))]:
        _add(row.node)
        if isinstance(row.node, tuple) and len(row.node) == 2 and row.node[0] in _PERIODIC_OPS:
            _add(row.node[1])
    return out


def _record_outer_family_precheck(
    ctx: _SolverContext,
    family_name: str,
    *,
    status: str,
    candidate_count: int = 0,
    best_probe_mse: float | None = None,
    improvement_ratio: float | None = None,
) -> None:
    status_map = dict(ctx.stats.get("outer_family_precheck_status", {}) or {})
    family_key = str(family_name)
    prev_status = str(status_map.get(family_key, "") or "")
    new_status = str(status)
    keep_previous = prev_status == "triggered" and new_status != "triggered"
    if not keep_previous:
        status_map[family_key] = new_status
    ctx.stats["outer_family_precheck_status"] = status_map

    count_map = dict(ctx.stats.get("outer_family_precheck_candidate_count", {}) or {})
    if not keep_previous:
        count_map[family_key] = int(candidate_count)
    ctx.stats["outer_family_precheck_candidate_count"] = count_map

    mse_map = dict(ctx.stats.get("outer_family_precheck_best_probe_mse", {}) or {})
    if not keep_previous:
        mse_map[family_key] = None if best_probe_mse is None or not math.isfinite(float(best_probe_mse)) else float(best_probe_mse)
    ctx.stats["outer_family_precheck_best_probe_mse"] = mse_map

    ratio_map = dict(ctx.stats.get("outer_family_precheck_improvement_ratio", {}) or {})
    if not keep_previous:
        ratio_map[family_key] = None if improvement_ratio is None or not math.isfinite(float(improvement_ratio)) else float(improvement_ratio)
    ctx.stats["outer_family_precheck_improvement_ratio"] = ratio_map


def _record_outer_family_rows(ctx: _SolverContext, family_name: str, n_rows: int) -> None:
    count_map = dict(ctx.stats.get("outer_family_candidate_counts", {}) or {})
    count_map[str(family_name)] = int(count_map.get(str(family_name), 0) or 0) + int(max(0, int(n_rows)))
    ctx.stats["outer_family_candidate_counts"] = count_map
    if int(n_rows) > 0:
        used_map = dict(ctx.stats.get("outer_family_used", {}) or {})
        used_map[str(family_name)] = True
        ctx.stats["outer_family_used"] = used_map
        ctx.stats["outer_family_candidate_count"] = int(ctx.stats.get("outer_family_candidate_count", 0) or 0) + int(n_rows)


def _record_outer_family_wall(ctx: _SolverContext, family_name: str, delta_s: float) -> None:
    _stats_add_time(ctx.stats, "outer_family_wall_seconds", delta_s)
    _stats_add_nested_time(ctx.stats, "outer_family_family_wall_seconds", str(family_name), delta_s)


def _record_outer_family_evidence(ctx: _SolverContext, family_name: str, evidence) -> None:
    evidence_map = dict(ctx.stats.get("outer_family_evidence", {}) or {})
    serialized = serialize_family_evidence(evidence)
    if serialized is None:
        return
    family_key = str(family_name)
    existing = dict(evidence_map.get(family_key, {}) or {})

    def _jet_used(payload: Mapping[str, Any] | None) -> bool:
        data = dict(payload or {})
        meta = dict(data.get("metadata", {}) or {})
        hard = dict(data.get("hard_constraints", {}) or {})
        return bool(meta.get("jet_evidence_used", hard.get("jet_evidence_used", False)))

    if existing and _jet_used(existing) and not _jet_used(serialized):
        return
    evidence_map[family_key] = serialized
    ctx.stats["outer_family_evidence"] = evidence_map


def _expanded_family_signal_summary(payload: Mapping[str, Any] | None, family_name: str) -> dict[str, Any]:
    data = dict(payload or {})
    if not data:
        return {}
    hard = dict(data.get("hard_constraints", {}) or {})
    meta = dict(data.get("metadata", {}) or {})
    scores = dict(data.get("family_scores", {}) or {})
    status = str(meta.get("status", hard.get("status", "")) or "")
    score_value = scores.get(str(family_name), None)
    try:
        score = None if score_value is None else float(score_value)
    except Exception:
        score = None
    if score is not None and not math.isfinite(float(score)):
        score = None
    summary: dict[str, Any] = {}
    if status:
        summary["status"] = str(status)
    if score is not None:
        summary["score"] = float(score)
    key_groups = {
        "asymptotic_monomial": (
            "primary_axis",
            "tail_count",
            "log_fit_r2",
            "exponent_estimate",
        ),
        "branch_structure": (
            "primary_axis",
            "branch_cut_risk",
            "one_sided_support",
            "hazard_severe",
        ),
        "coordinate_invariant": (
            "coordinate_vars",
            "gradient_rank1_fraction",
            "gradient_direction_coherence",
        ),
        "regime_lift": (
            "regime_count",
            "mean_cv",
            "top_constant_name",
            "top_constant_cv",
        ),
    }
    for key in key_groups.get(str(family_name), ()):
        if key in hard and hard.get(key) is not None:
            summary[str(key)] = hard.get(key)
    seed_nodes = tuple(data.get("seed_nodes", ()) or ())
    if str(family_name) == "coordinate_invariant" and seed_nodes:
        summary["seed_node_count"] = int(len(seed_nodes))
    return summary


def _expanded_outer_family_annotations(ctx: _SolverContext) -> dict[str, dict[str, Any]]:
    evidence_map = dict(ctx.stats.get("outer_family_evidence", {}) or {})
    summary: dict[str, dict[str, Any]] = {}
    for family_name in (
        "asymptotic_monomial",
        "branch_structure",
        "coordinate_invariant",
        "regime_lift",
    ):
        item = _expanded_family_signal_summary(evidence_map.get(str(family_name), None), str(family_name))
        if item:
            summary[str(family_name)] = item
    return summary


def _record_expanded_family_evidence(problem: _LocalProblem, *, ctx: _SolverContext) -> None:
    mode_name = normalize_family_battery_mode(getattr(ctx, "family_battery_mode", "outer"))
    if mode_name != "expanded":
        return
    problem_diag = dict(problem.diagnostics or {})
    evidence_bundle = build_expanded_family_evidence_bundle(
        x_fit=problem.xf,
        t_fit=problem.tf,
        x_probe=problem.xp,
        t_probe=problem.tp,
        grad_fit=problem.grad_fit,
        grad_probe=problem.grad_probe,
        d2_fit=problem.d2_fit,
        d2_probe=problem.d2_probe,
        fit_jet_source=_diagnostic_teacher_source(problem, split="fit"),
        probe_jet_source=_diagnostic_teacher_source(problem, split="probe"),
        fit_jet_fallback_used=bool(dict(problem.diagnostics or {}).get("fit_jet_fallback_used", False)),
        probe_jet_fallback_used=bool(dict(problem.diagnostics or {}).get("probe_jet_fallback_used", False)),
        target_dim=problem.target_dim,
        active_vars=problem_diag.get("active_vars", ()),
        wrappers_left=int(problem.wrappers_left),
        recursion_level=int(problem.recursion_level),
        target_mode=str(ctx.target_mode or ""),
        target_mapping_kind=str(ctx.target_mapping_kind or ""),
        regime_metadata=extract_family_regime_metadata(problem_diag),
    )
    for family_name, evidence in sorted(dict(evidence_bundle or {}).items()):
        _record_outer_family_evidence(ctx, str(family_name), evidence)


def _score_sinusoidal_forward(
    g_fit: torch.Tensor,
    g_probe: torch.Tensor,
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
) -> _OuterFamilyScore | None:
    """Fit ``t ≈ a sin(g) + b cos(g) + c`` and return a scored payload.

    For fixed *g*, this is a 3-parameter weighted linear least squares.
    It sidesteps branch ambiguity entirely: all 2πk sheet information is
    absorbed by the ``[sin, cos, 1]`` basis.
    """
    gf = _ensure_col(g_fit).squeeze(-1)
    gp = _ensure_col(g_probe).squeeze(-1)
    tf = _ensure_col(t_fit).squeeze(-1)
    tp = _ensure_col(t_probe).squeeze(-1)

    nf = int(gf.shape[0])
    np_ = int(gp.shape[0])
    if nf < 4 or np_ < 4:
        return None
    if (not torch.isfinite(gf).all()) or (not torch.isfinite(gp).all()):
        return None
    if (not torch.isfinite(tf).all()) or (not torch.isfinite(tp).all()):
        return None

    # Build design matrix [sin(g), cos(g), 1]
    sin_f = torch.sin(gf)
    cos_f = torch.cos(gf)
    ones_f = torch.ones_like(gf)
    A_fit = torch.stack([sin_f, cos_f, ones_f], dim=1)  # [N, 3]

    sin_p = torch.sin(gp)
    cos_p = torch.cos(gp)
    ones_p = torch.ones_like(gp)
    A_probe = torch.stack([sin_p, cos_p, ones_p], dim=1)

    # Apply weights
    if w_fit is not None:
        wf_sq = torch.sqrt(torch.clamp(_ensure_col(w_fit).squeeze(-1), min=0.0))
        A_fit = A_fit * wf_sq.unsqueeze(-1)
        tf_w = tf * wf_sq
    else:
        tf_w = tf

    if w_probe is not None:
        wp_sq = torch.sqrt(torch.clamp(_ensure_col(w_probe).squeeze(-1), min=0.0))
    else:
        wp_sq = None

    # Solve weighted least squares: (A^T A) x = A^T t
    try:
        coeffs = torch.linalg.lstsq(A_fit, tf_w).solution  # [3]
    except Exception:
        return None
    if not torch.isfinite(coeffs).all():
        return None

    pred_fit = A_fit @ coeffs
    if w_fit is not None:
        # Undo weight scaling for residual
        resid_fit = (pred_fit / wf_sq.clamp(min=1e-30)) - tf
        # But MSE should be weighted
        fit_mse = float((wf_sq * resid_fit).pow(2).mean().item())
    else:
        resid_fit = pred_fit - tf
        fit_mse = float(resid_fit.pow(2).mean().item())

    # Probe: use original (unweighted) design matrix
    pred_probe = A_probe @ coeffs
    resid_probe = pred_probe - tp
    if wp_sq is not None:
        probe_mse = float((wp_sq * resid_probe).pow(2).mean().item())
    else:
        probe_mse = float(resid_probe.pow(2).mean().item())

    if not (math.isfinite(fit_mse) and math.isfinite(probe_mse)):
        return None
    return _OuterFamilyScore(
        fit_mse=float(fit_mse),
        probe_mse=float(probe_mse),
        payload={
            "sin_coeff": float(coeffs[0]),
            "cos_coeff": float(coeffs[1]),
            "bias": float(coeffs[2]),
        },
    )


def _exp_raw_sweep_batch(
    g: torch.Tensor,
    y: torch.Tensor,
    bs: torch.Tensor,
    *,
    w: torch.Tensor | None = None,
) -> tuple[float, float | None, torch.Tensor | None]:
    gg = _ensure_col(g).squeeze(-1)
    yy = _ensure_col(y).squeeze(-1)
    M = int(bs.shape[0])
    if M <= 0:
        return float("inf"), None, None
    bz = (bs[:, None] * gg[None, :]).clamp(-20.0, 20.0)
    ebz = torch.exp(bz)
    A = torch.stack([ebz, torch.ones_like(ebz)], dim=2)
    rhs = yy[None, :, None].expand(M, -1, 1)
    if w is not None:
        ww = torch.sqrt(torch.clamp(_ensure_col(w).squeeze(-1), min=0.0))
        A = A * ww[None, :, None]
        rhs = rhs * ww[None, :, None]
    sol = torch.linalg.lstsq(A, rhs).solution
    residual = rhs - A @ sol
    mses = (residual * residual).mean(dim=1).squeeze(-1)
    valid = torch.isfinite(mses) & torch.isfinite(sol.squeeze(-1)).all(dim=1)
    if not valid.any():
        return float("inf"), None, None
    mses[~valid] = float("inf")
    idx = int(mses.argmin())
    return float(mses[idx]), float(bs[idx]), sol[idx, :, 0]


def _score_exp_forward_raw(
    g_fit: torch.Tensor,
    g_probe: torch.Tensor,
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
) -> _OuterFamilyScore | None:
    gf = _ensure_col(g_fit).squeeze(-1)
    gp = _ensure_col(g_probe).squeeze(-1)
    tf = _ensure_col(t_fit)
    tp = _ensure_col(t_probe)
    if int(gf.shape[0]) < 4 or int(gp.shape[0]) < 4:
        return None
    if (not torch.isfinite(gf).all()) or (not torch.isfinite(gp).all()):
        return None
    if (not torch.isfinite(tf).all()) or (not torch.isfinite(tp).all()):
        return None
    b_pos = torch.logspace(
        math.log10(0.1),
        math.log10(float(_OUTER_FAMILY_EXP_B_MAX)),
        int(_OUTER_FAMILY_EXP_N_B),
        dtype=gf.dtype,
        device=gf.device,
    )
    b_cands = torch.cat([-b_pos.flip(0), b_pos], dim=0)
    best_mse, best_b, best_sol = _exp_raw_sweep_batch(gf, tf, b_cands, w=w_fit)
    if best_b is None or best_sol is None:
        return None
    span = max(abs(float(best_b)) * 0.3, 0.2)
    for _ in range(int(_OUTER_FAMILY_EXP_N_REFINE)):
        lo = float(best_b) - float(span)
        hi = float(best_b) + float(span)
        refine_bs = torch.linspace(lo, hi, int(_OUTER_FAMILY_EXP_N_B), dtype=gf.dtype, device=gf.device)
        mse, b, sol = _exp_raw_sweep_batch(gf, tf, refine_bs, w=w_fit)
        if b is not None and mse < best_mse:
            best_mse, best_b, best_sol = mse, b, sol
        span *= 0.35
    if best_sol is None or best_b is None:
        return None
    a_val = float(best_sol[0])
    c_val = float(best_sol[1])
    probe_pred = (a_val * torch.exp((float(best_b) * gp).clamp(-20.0, 20.0)) + c_val).unsqueeze(-1)
    probe_mse = _weighted_mse_cols(tp, probe_pred, w_probe)
    fit_pred = (a_val * torch.exp((float(best_b) * gf).clamp(-20.0, 20.0)) + c_val).unsqueeze(-1)
    fit_mse = _weighted_mse_cols(tf, fit_pred, w_fit)
    if fit_mse is None or probe_mse is None:
        return None
    if not (math.isfinite(float(fit_mse)) and math.isfinite(float(probe_mse))):
        return None
    return _OuterFamilyScore(
        fit_mse=float(fit_mse),
        probe_mse=float(probe_mse),
        payload={"a": float(a_val), "b": float(best_b), "c": float(c_val)},
    )


def _score_power_forward_raw(
    g_fit: torch.Tensor,
    g_probe: torch.Tensor,
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
) -> _OuterFamilyScore | None:
    mapping = fit_power(_ensure_col(g_fit), _ensure_col(t_fit))
    if mapping is None:
        return None
    try:
        fit_pred = eval_power(_ensure_col(g_fit), mapping)
        probe_pred = eval_power(_ensure_col(g_probe), mapping)
    except Exception:
        return None
    fit_mse = _weighted_mse_cols(_ensure_col(t_fit), fit_pred, w_fit)
    probe_mse = _weighted_mse_cols(_ensure_col(t_probe), probe_pred, w_probe)
    if fit_mse is None or probe_mse is None:
        return None
    if not (math.isfinite(float(fit_mse)) and math.isfinite(float(probe_mse))):
        return None
    return _OuterFamilyScore(
        fit_mse=float(fit_mse),
        probe_mse=float(probe_mse),
        payload=dict(mapping),
    )


def _score_rational_forward_raw(
    g_fit: torch.Tensor,
    g_probe: torch.Tensor,
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
) -> _OuterFamilyScore | None:
    gf = _ensure_col(g_fit).squeeze(-1)
    gp = _ensure_col(g_probe).squeeze(-1)
    tf = _ensure_col(t_fit).squeeze(-1)
    tp = _ensure_col(t_probe).squeeze(-1)
    if int(gf.shape[0]) < 4 or int(gp.shape[0]) < 4:
        return None
    if (not torch.isfinite(gf).all()) or (not torch.isfinite(gp).all()):
        return None
    if (not torch.isfinite(tf).all()) or (not torch.isfinite(tp).all()):
        return None
    A = torch.stack([gf, torch.ones_like(gf), -(tf * gf)], dim=1)
    rhs = tf
    if w_fit is not None:
        wf_sq = torch.sqrt(torch.clamp(_ensure_col(w_fit).squeeze(-1), min=0.0))
        A = A * wf_sq.unsqueeze(-1)
        rhs = rhs * wf_sq
    try:
        sol = torch.linalg.lstsq(A, rhs).solution
    except Exception:
        return None
    if not torch.isfinite(sol).all():
        return None
    a_val = float(sol[0])
    b_val = float(sol[1])
    c_val = float(sol[2])
    den_fit = c_val * gf + 1.0
    den_probe = c_val * gp + 1.0
    if (den_fit.abs() < float(_OUTER_FAMILY_RATIONAL_DEN_EPS)).any():
        return None
    if (den_probe.abs() < float(_OUTER_FAMILY_RATIONAL_DEN_EPS)).any():
        return None
    fit_pred = (((a_val * gf) + b_val) / den_fit).unsqueeze(-1)
    probe_pred = (((a_val * gp) + b_val) / den_probe).unsqueeze(-1)
    fit_mse = _weighted_mse_cols(_ensure_col(t_fit), fit_pred, w_fit)
    probe_mse = _weighted_mse_cols(_ensure_col(t_probe), probe_pred, w_probe)
    if fit_mse is None or probe_mse is None:
        return None
    if not (math.isfinite(float(fit_mse)) and math.isfinite(float(probe_mse))):
        return None
    return _OuterFamilyScore(
        fit_mse=float(fit_mse),
        probe_mse=float(probe_mse),
        payload={"a": a_val, "b": b_val, "c": c_val},
    )


def _score_outer_family_forward(
    family_name: str,
    g_fit: torch.Tensor,
    g_probe: torch.Tensor,
    *,
    problem: _LocalProblem,
) -> _OuterFamilyScore | None:
    if str(family_name) == "periodic":
        return _score_sinusoidal_forward(
            g_fit,
            g_probe,
            problem.tf,
            problem.tp,
            problem.wf,
            problem.wp,
        )
    if str(family_name) == "exp":
        return _score_exp_forward_raw(
            g_fit,
            g_probe,
            problem.tf,
            problem.tp,
            problem.wf,
            problem.wp,
        )
    if str(family_name) == "power":
        return _score_power_forward_raw(
            g_fit,
            g_probe,
            problem.tf,
            problem.tp,
            problem.wf,
            problem.wp,
        )
    if str(family_name) == "rational":
        return _score_rational_forward_raw(
            g_fit,
            g_probe,
            problem.tf,
            problem.tp,
            problem.wf,
            problem.wp,
        )
    return None


def _periodic_forward_precheck_candidates(
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    problem: _LocalProblem,
    ctx: _SolverContext,
) -> list[tuple]:
    return _outer_family_candidate_nodes(
        flat_rows,
        problem=problem,
        ctx=ctx,
        max_seeds=max(1, min(int(ctx.recursive_seed_cap), int(_PERIODIC_PRECHECK_MAX_SEEDS))),
        child_target_dim=_outer_family_child_target_dim("periodic", problem, ctx=ctx),
    )


def _set_root_periodic_precheck_stat(problem: _LocalProblem, ctx: _SolverContext, key: str, value) -> None:
    if int(problem.recursion_level) == 0:
        ctx.stats[str(key)] = value


def _estimate_explicit_periodic_inverse_confidence(
    problem: _LocalProblem,
    *,
    ctx: _SolverContext,
) -> tuple[float, float, int]:
    best_confidence = 0.0
    best_supported_confidence = 0.0
    branch_count = 0
    branch_beam_width = max(1, min(int(ctx.branch_beam_width), 2))
    for op in ("sin", "cos"):
        child_target_dim = _unary_child_dim(str(op), problem.target_dim, ctx=ctx)
        if ctx.dm and child_target_dim is None:
            continue
        try:
            fit_branches = _invert_unary_context_branches(
                str(op),
                problem.tf,
                child_pred_ref=None,
                safe_eps=float(ctx.safe_eps),
                confidence_mode=str(ctx.confidence_mode),
                confidence_target_gain=float(ctx.confidence_target_gain),
                confidence_floor=float(ctx.confidence_floor),
                branch_beam_width=int(branch_beam_width),
            )
            probe_branches = _invert_unary_context_branches(
                str(op),
                problem.tp,
                child_pred_ref=None,
                safe_eps=float(ctx.safe_eps),
                confidence_mode=str(ctx.confidence_mode),
                confidence_target_gain=float(ctx.confidence_target_gain),
                confidence_floor=float(ctx.confidence_floor),
                branch_beam_width=int(branch_beam_width),
            )
        except Exception:
            continue
        probe_by_token = {str(token): (t, m, c, pw) for t, m, c, pw, _note, token in probe_branches}
        for fit_t, fit_m, fit_c, fit_pw, _fit_note, token in fit_branches:
            probe_row = probe_by_token.get(str(token), None)
            if probe_row is None:
                continue
            probe_t, probe_m, probe_c, probe_pw = probe_row
            child_problem = _build_child_problem(
                problem,
                child_target_fit=fit_t,
                fit_mask=fit_m,
                fit_weight=fit_pw,
                fit_conf=float(fit_c),
                child_target_probe=probe_t,
                probe_mask=probe_m,
                probe_weight=probe_pw,
                probe_conf=float(probe_c),
                child_target_dim=child_target_dim,
                trace_token=f"periodic_precheck:{op}:{token}",
                ctx=ctx,
            )
            if child_problem is None:
                continue
            branch_count += 1
            child_confidence = float(child_problem.confidence)
            supported_confidence = child_confidence * float(child_problem.valid_frac)
            if child_confidence > best_confidence:
                best_confidence = child_confidence
            if supported_confidence > best_supported_confidence:
                best_supported_confidence = supported_confidence
    return float(best_confidence), float(best_supported_confidence), int(branch_count)


def _should_run_periodic_forward(
    problem: _LocalProblem,
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
) -> bool:
    if bool(ctx.family_battery_enable):
        target_dim_ok = True
        if ctx.dm:
            dim0 = (0.0,) * len(ctx.var_dims[0])
            target_dim_ok = bool(problem.target_dim is not None and dims_eq(problem.target_dim, dim0))
        best_flat_probe = None
        if flat_rows:
            best_flat_probe = float(flat_rows[0].local_probe_mse)
            if not math.isfinite(float(best_flat_probe)):
                best_flat_probe = None
        explicit_inverse_confidence = 0.0
        explicit_inverse_supported_confidence = 0.0
        explicit_inverse_branch_count = 0
        if bool(ctx.recursive_enable) and int(problem.wrappers_left) > 0 and bool(flat_rows) and bool(target_dim_ok) and best_flat_probe is not None:
            explicit_inverse_confidence, explicit_inverse_supported_confidence, explicit_inverse_branch_count = (
                _estimate_explicit_periodic_inverse_confidence(problem, ctx=ctx)
            )
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_explicit_inverse_confidence", float(explicit_inverse_confidence))
        _set_root_periodic_precheck_stat(
            problem,
            ctx,
            "periodic_explicit_inverse_supported_confidence",
            float(explicit_inverse_supported_confidence),
        )
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_explicit_inverse_branch_count", int(explicit_inverse_branch_count))
        if (
            int(explicit_inverse_branch_count) > 0
            and float(explicit_inverse_supported_confidence) >= float(_PERIODIC_CONFIDENCE_THRESHOLD)
        ):
            evidence = build_named_outer_family_evidence(
                "periodic",
                recursive_enable=bool(ctx.recursive_enable),
                wrappers_left=int(problem.wrappers_left),
                flat_rows_present=bool(flat_rows),
                target_dim_ok=bool(target_dim_ok),
                best_flat_probe_mse=best_flat_probe,
                seed_nodes=(),
                min_improvement_ratio=float(_PERIODIC_PRECHECK_MIN_IMPROVEMENT_RATIO),
                best_probe_mse=None,
                status_override="explicit_inverse_confident",
                extra_hard_constraints={
                    "explicit_inverse_supported_confidence": float(explicit_inverse_supported_confidence),
                    "explicit_inverse_branch_count": int(explicit_inverse_branch_count),
                },
                extra_metadata={
                    "explicit_inverse_confidence": float(explicit_inverse_confidence),
                    "explicit_inverse_supported_confidence": float(explicit_inverse_supported_confidence),
                    "explicit_inverse_branch_count": int(explicit_inverse_branch_count),
                },
                target_dim=problem.target_dim,
                active_vars=dict(problem.diagnostics or {}).get("active_vars", ()),
                recursion_level=int(problem.recursion_level),
                target_mode=str(ctx.target_mode or ""),
                target_mapping_kind=str(ctx.target_mapping_kind or ""),
                regime_metadata=extract_family_regime_metadata(dict(problem.diagnostics or {})),
            )
            _record_outer_family_evidence(ctx, "periodic", evidence)
            _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "explicit_inverse_confident")
            _record_outer_family_precheck(
                ctx,
                "periodic",
                status="explicit_inverse_confident",
                candidate_count=0,
            )
            return False
        candidate_nodes: list[tuple] = []
        if bool(flat_rows) and bool(target_dim_ok) and best_flat_probe is not None:
            candidate_nodes = _periodic_forward_precheck_candidates(flat_rows, problem=problem, ctx=ctx)
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_candidate_count", int(len(candidate_nodes)))
        best_periodic_probe = None
        for node in candidate_nodes:
            try:
                g_fit = eval_node(node, problem.xf)
                g_probe = eval_node(node, problem.xp)
            except Exception:
                continue
            if (not torch.isfinite(g_fit).all()) or (not torch.isfinite(g_probe).all()):
                continue
            score = _score_outer_family_forward("periodic", g_fit, g_probe, problem=problem)
            if score is None or not math.isfinite(float(score.probe_mse)):
                continue
            if best_periodic_probe is None:
                best_periodic_probe = float(score.probe_mse)
            else:
                best_periodic_probe = min(float(best_periodic_probe), float(score.probe_mse))
        _set_root_periodic_precheck_stat(
            problem,
            ctx,
            "periodic_precheck_best_probe_mse",
            best_periodic_probe,
        )
        evidence = build_named_outer_family_evidence(
            "periodic",
            recursive_enable=bool(ctx.recursive_enable),
            wrappers_left=int(problem.wrappers_left),
            flat_rows_present=bool(flat_rows),
            target_dim_ok=bool(target_dim_ok),
            best_flat_probe_mse=best_flat_probe,
            seed_nodes=candidate_nodes,
            min_improvement_ratio=float(_PERIODIC_PRECHECK_MIN_IMPROVEMENT_RATIO),
            best_probe_mse=best_periodic_probe,
            extra_hard_constraints={
                "explicit_inverse_supported_confidence": float(explicit_inverse_supported_confidence),
                "explicit_inverse_branch_count": int(explicit_inverse_branch_count),
            },
            extra_metadata={
                "explicit_inverse_confidence": float(explicit_inverse_confidence),
                "explicit_inverse_supported_confidence": float(explicit_inverse_supported_confidence),
                "explicit_inverse_branch_count": int(explicit_inverse_branch_count),
            },
            target_dim=problem.target_dim,
            active_vars=dict(problem.diagnostics or {}).get("active_vars", ()),
            recursion_level=int(problem.recursion_level),
            target_mode=str(ctx.target_mode or ""),
            target_mapping_kind=str(ctx.target_mapping_kind or ""),
            regime_metadata=extract_family_regime_metadata(dict(problem.diagnostics or {})),
        )
        _record_outer_family_evidence(ctx, "periodic", evidence)
        status = family_evidence_status(evidence, "periodic")
        improve_ratio = dict(evidence.metadata or {}).get("improvement_ratio", None)
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_improvement_ratio", improve_ratio)
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", status)
        _record_outer_family_precheck(
            ctx,
            "periodic",
            status=status,
            candidate_count=int(len(candidate_nodes)),
            best_probe_mse=best_periodic_probe,
            improvement_ratio=improve_ratio,
        )
        return bool(family_evidence_should_run(evidence, "periodic"))
    if not bool(ctx.recursive_enable):
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "disabled")
        _record_outer_family_precheck(ctx, "periodic", status="disabled")
        return False
    if int(problem.wrappers_left) <= 0:
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "no_wrappers_left")
        _record_outer_family_precheck(ctx, "periodic", status="no_wrappers_left")
        return False

    # Trig arguments must be dimensionless; trig output is dimensionless.
    if ctx.dm:
        dim0 = (0.0,) * len(ctx.var_dims[0])
        if problem.target_dim is None or not dims_eq(problem.target_dim, dim0):
            _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "nondimensionless_target")
            _record_outer_family_precheck(ctx, "periodic", status="nondimensionless_target")
            return False

    if not flat_rows:
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "no_flat_rows")
        _record_outer_family_precheck(ctx, "periodic", status="no_flat_rows")
        return False

    best_flat_probe = float(flat_rows[0].local_probe_mse)
    if not math.isfinite(best_flat_probe):
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "nonfinite_flat_probe")
        _record_outer_family_precheck(ctx, "periodic", status="nonfinite_flat_probe")
        return False

    explicit_inverse_confidence, explicit_inverse_supported_confidence, explicit_inverse_branch_count = (
        _estimate_explicit_periodic_inverse_confidence(problem, ctx=ctx)
    )
    _set_root_periodic_precheck_stat(problem, ctx, "periodic_explicit_inverse_confidence", float(explicit_inverse_confidence))
    _set_root_periodic_precheck_stat(
        problem,
        ctx,
        "periodic_explicit_inverse_supported_confidence",
        float(explicit_inverse_supported_confidence),
    )
    _set_root_periodic_precheck_stat(problem, ctx, "periodic_explicit_inverse_branch_count", int(explicit_inverse_branch_count))
    if (
        int(explicit_inverse_branch_count) > 0
        and float(explicit_inverse_supported_confidence) >= float(_PERIODIC_CONFIDENCE_THRESHOLD)
    ):
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "explicit_inverse_confident")
        _record_outer_family_precheck(
            ctx,
            "periodic",
            status="explicit_inverse_confident",
            candidate_count=0,
        )
        return False

    candidate_nodes = _periodic_forward_precheck_candidates(flat_rows, problem=problem, ctx=ctx)
    _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_candidate_count", int(len(candidate_nodes)))
    if not candidate_nodes:
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "no_candidate_nodes")
        _record_outer_family_precheck(ctx, "periodic", status="no_candidate_nodes", candidate_count=0)
        return False

    best_periodic_probe = float("inf")
    for node in candidate_nodes:
        try:
            g_fit = eval_node(node, problem.xf)
            g_probe = eval_node(node, problem.xp)
        except Exception:
            continue
        if (not torch.isfinite(g_fit).all()) or (not torch.isfinite(g_probe).all()):
            continue
        score = _score_outer_family_forward("periodic", g_fit, g_probe, problem=problem)
        if score is None:
            continue
        if math.isfinite(float(score.probe_mse)):
            best_periodic_probe = min(best_periodic_probe, float(score.probe_mse))

    _set_root_periodic_precheck_stat(
        problem,
        ctx,
        "periodic_precheck_best_probe_mse",
        None if not math.isfinite(best_periodic_probe) else float(best_periodic_probe),
    )
    if not math.isfinite(best_periodic_probe):
        _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", "no_finite_periodic_fit")
        _record_outer_family_precheck(
            ctx,
            "periodic",
            status="no_finite_periodic_fit",
            candidate_count=int(len(candidate_nodes)),
        )
        return False

    improve_ratio = float(best_periodic_probe / max(best_flat_probe, 1.0e-12))
    _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_improvement_ratio", float(improve_ratio))
    should_run = bool(improve_ratio <= float(_PERIODIC_PRECHECK_MIN_IMPROVEMENT_RATIO))
    root_status = "triggered" if should_run else "insufficient_improvement"
    _set_root_periodic_precheck_stat(problem, ctx, "periodic_precheck_status", root_status)
    _record_outer_family_precheck(
        ctx,
        "periodic",
        status=str(root_status),
        candidate_count=int(len(candidate_nodes)),
        best_probe_mse=float(best_periodic_probe),
        improvement_ratio=float(improve_ratio),
    )
    return should_run


def _score_candidate_sinusoidal(
    g_node,
    *,
    problem: _LocalProblem,
    ctx: _SolverContext,
    wrapper_op: str,
    source: str,
    trace: Sequence[str] | None = None,
) -> list[_ScoredLocalCandidate]:
    started = time.perf_counter()
    """Score a candidate argument *g_node* under both sin and cos wrappers.

    Returns up to 2 scored candidates (one per wrapper) that beat the
    sinusoidal forward objective.
    """
    if not isinstance(g_node, tuple) or not g_node:
        return []
    g_node = _simplify_node(g_node)

    # Dimensional check: argument must match child dim
    if ctx.dm:
        dim0 = (0.0,) * len(ctx.var_dims[0])
        try:
            gd = node_dims(g_node, ctx.var_dims)
        except Exception:
            gd = None
        if gd is None or not dims_eq(gd, dim0):
            return []

    try:
        g_fit = eval_node(g_node, problem.xf)
        g_probe = eval_node(g_node, problem.xp)
    except Exception:
        return []
    if (not torch.isfinite(g_fit).all()) or (not torch.isfinite(g_probe).all()):
        return []

    result = _score_sinusoidal_forward(
        g_fit, g_probe,
        problem.tf, problem.tp,
        problem.wf, problem.wp,
    )
    if result is None:
        return []

    # Emit one or both wrappers
    out: list[_ScoredLocalCandidate] = []
    for op in ("sin", "cos") if wrapper_op == "sincos" else (wrapper_op,):
        wrapped = _simplify_node((op, g_node))
        if int(node_depth(wrapped)) > _candidate_depth_limit(problem, ctx):
            continue
        if ctx.dm:
            try:
                wd = node_dims(wrapped, ctx.var_dims)
            except Exception:
                wd = None
            if wd is None or problem.target_dim is None or not dims_eq(wd, problem.target_dim):
                continue
        # Score the wrapped node against the parent target (value-level check)
        value_scored = _score_node_against_problem(
            wrapped,
            problem=problem,
            ctx=ctx,
            source=source,
            generation_kind="periodic_forward",
            confidence=float(problem.confidence),
            valid_frac=float(problem.valid_frac),
            trace=tuple(str(v) for v in (trace or problem.trace)),
        )
        if value_scored is not None:
            surrogate_probe = float(result.probe_mse)
            surrogate_fit = float(result.fit_mse)
            value_probe = float(
                value_scored.local_probe_mse
                if value_scored.value_probe_mse is None
                else value_scored.value_probe_mse
            )
            value_fit = float(
                value_scored.local_fit_mse
                if value_scored.value_fit_mse is None
                else value_scored.value_fit_mse
            )
            calibration_gap = max(
                abs(surrogate_probe - value_probe),
                abs(surrogate_fit - value_fit),
            )
            out.append(_ScoredLocalCandidate(
                node=wrapped,
                local_probe_mse=float(value_scored.local_probe_mse),
                local_fit_mse=float(value_scored.local_fit_mse),
                source=source,
                generation_kind="periodic_forward",
                recursion_depth=int(problem.recursion_level),
                confidence=float(problem.confidence),
                valid_frac=float(problem.valid_frac),
                trace=tuple(str(v) for v in (trace or problem.trace)),
                family="periodic",
                payload=dict(result.payload or {}),
                surrogate_probe_mse=surrogate_probe,
                surrogate_fit_mse=surrogate_fit,
                value_probe_mse=value_probe,
                value_fit_mse=value_fit,
                calibration_gap=float(calibration_gap),
                witness_value_loss=value_scored.witness_value_loss,
                witness_grad_loss=value_scored.witness_grad_loss,
                witness_d2_loss=value_scored.witness_d2_loss,
                witness_diag_loss=value_scored.witness_diag_loss,
                witness_physics_loss=value_scored.witness_physics_loss,
                witness_energy_total=value_scored.witness_energy_total,
            ))
    _stats_add_time(ctx.stats, "periodic_sinusoidal_wall_seconds", time.perf_counter() - started)
    ctx.stats["periodic_sinusoidal_count"] = int(ctx.stats.get("periodic_sinusoidal_count", 0) or 0) + 1
    return out


def _compose_exp_outer_node(g_node, payload: Mapping[str, Any] | None):
    if not isinstance(payload, Mapping):
        return None
    try:
        a_val = float(payload.get("a", 1.0))
        b_val = float(payload.get("b", 1.0))
        c_val = float(payload.get("c", 0.0))
    except Exception:
        return None
    try:
        exp_arg = _mul_const(g_node, b_val)
        expr = _mul_const(("exp", exp_arg), a_val)
        expr = _add_const(expr, c_val)
        return _simplify_node(expr)
    except Exception:
        return None


def _compose_power_outer_node(g_node, payload: Mapping[str, Any] | None):
    if not isinstance(payload, Mapping):
        return None
    try:
        log_a = float(payload.get("log_a", 0.0))
        b_val = float(payload.get("b", 1.0))
        sgn_f = float(payload.get("sgn_f", 1.0))
        sgn_y = float(payload.get("sgn_y", 1.0))
    except Exception:
        return None
    try:
        base = _mul_const(g_node, sgn_f)
        core = ("log", base)
        core = _mul_const(core, b_val)
        core = _add_const(core, log_a)
        expr = ("exp", core)
        expr = _mul_const(expr, sgn_y)
        return _simplify_node(expr)
    except Exception:
        return None


def _compose_rational_outer_node(g_node, payload: Mapping[str, Any] | None):
    if not isinstance(payload, Mapping):
        return None
    try:
        a_val = float(payload.get("a", 0.0))
        b_val = float(payload.get("b", 0.0))
        c_val = float(payload.get("c", 0.0))
    except Exception:
        return None
    try:
        numer = _add_const(_mul_const(g_node, a_val), b_val)
        denom = _add_const(_mul_const(g_node, c_val), 1.0)
        expr = ("div", numer, denom)
        return _simplify_node(expr)
    except Exception:
        return None


def _score_outer_family_candidate(
    g_node,
    *,
    problem: _LocalProblem,
    ctx: _SolverContext,
    family_name: str,
    source: str,
    trace: Sequence[str] | None = None,
) -> list[_ScoredLocalCandidate]:
    started = time.perf_counter()
    if not isinstance(g_node, tuple) or not g_node:
        return []
    g_node = _simplify_node(g_node)
    child_target_dim = _outer_family_child_target_dim(family_name, problem, ctx=ctx)
    if ctx.dm:
        try:
            gd = node_dims(g_node, ctx.var_dims)
        except Exception:
            gd = None
        if gd is None or child_target_dim is None or not dims_eq(gd, child_target_dim):
            return []
    try:
        g_fit = eval_node(g_node, problem.xf)
        g_probe = eval_node(g_node, problem.xp)
    except Exception:
        return []
    if (not torch.isfinite(g_fit).all()) or (not torch.isfinite(g_probe).all()):
        return []
    score = _score_outer_family_forward(family_name, g_fit, g_probe, problem=problem)
    if score is None:
        return []
    compose_fns = {
        "exp": _compose_exp_outer_node,
        "power": _compose_power_outer_node,
        "rational": _compose_rational_outer_node,
    }
    compose_fn = compose_fns.get(str(family_name), None)
    if compose_fn is None:
        return []
    wrapped = compose_fn(g_node, score.payload)
    if not isinstance(wrapped, tuple) or not wrapped:
        return []
    value_scored = _score_node_against_problem(
        wrapped,
        problem=problem,
        ctx=ctx,
        source=str(source),
        generation_kind=f"outer_family:{family_name}",
        confidence=float(problem.confidence),
        valid_frac=float(problem.valid_frac),
        trace=tuple(str(v) for v in (trace or problem.trace)),
    )
    if value_scored is None:
        return []
    surrogate_probe = float(score.probe_mse)
    surrogate_fit = float(score.fit_mse)
    value_probe = float(
        value_scored.local_probe_mse
        if value_scored.value_probe_mse is None
        else value_scored.value_probe_mse
    )
    value_fit = float(
        value_scored.local_fit_mse
        if value_scored.value_fit_mse is None
        else value_scored.value_fit_mse
    )
    calibration_gap = max(
        abs(surrogate_probe - value_probe),
        abs(surrogate_fit - value_fit),
    )
    row = _ScoredLocalCandidate(
        node=wrapped,
        local_probe_mse=float(value_scored.local_probe_mse),
        local_fit_mse=float(value_scored.local_fit_mse),
        source=str(source),
        generation_kind=f"outer_family:{family_name}",
        recursion_depth=int(problem.recursion_level),
        confidence=float(problem.confidence),
        valid_frac=float(problem.valid_frac),
        trace=tuple(str(v) for v in (trace or problem.trace)),
        family=str(family_name),
        payload=(dict(score.payload) if isinstance(score.payload, Mapping) else score.payload),
        surrogate_probe_mse=surrogate_probe,
        surrogate_fit_mse=surrogate_fit,
        value_probe_mse=value_probe,
        value_fit_mse=value_fit,
        calibration_gap=float(calibration_gap),
        witness_value_loss=value_scored.witness_value_loss,
        witness_grad_loss=value_scored.witness_grad_loss,
        witness_d2_loss=value_scored.witness_d2_loss,
        witness_diag_loss=value_scored.witness_diag_loss,
        witness_physics_loss=value_scored.witness_physics_loss,
        witness_energy_total=value_scored.witness_energy_total,
    )
    _record_outer_family_rows(ctx, str(family_name), 1)
    _record_outer_family_wall(ctx, str(family_name), time.perf_counter() - started)
    return [row]


def _should_run_named_outer_family(
    family_name: str,
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    problem: _LocalProblem,
    ctx: _SolverContext,
    min_improvement_ratio: float,
    max_seeds: int,
) -> bool:
    if bool(ctx.family_battery_enable):
        mode_name = normalize_family_battery_mode(getattr(ctx, "family_battery_mode", "outer"))
        child_target_dim = _outer_family_child_target_dim(family_name, problem, ctx=ctx)
        target_dim_ok = (not ctx.dm) or (child_target_dim is not None)
        best_flat_probe = None
        if flat_rows:
            best_flat_probe = float(flat_rows[0].local_probe_mse)
            if not math.isfinite(float(best_flat_probe)):
                best_flat_probe = None
        candidate_nodes: list[tuple] = []
        if bool(flat_rows) and bool(target_dim_ok):
            candidate_nodes = _outer_family_candidate_nodes(
                flat_rows,
                problem=problem,
                ctx=ctx,
                max_seeds=max(1, int(max_seeds)),
                child_target_dim=child_target_dim,
            )
        best_probe = None
        for node in candidate_nodes:
            try:
                g_fit = eval_node(node, problem.xf)
                g_probe = eval_node(node, problem.xp)
            except Exception:
                continue
            if (not torch.isfinite(g_fit).all()) or (not torch.isfinite(g_probe).all()):
                continue
            score = _score_outer_family_forward(family_name, g_fit, g_probe, problem=problem)
            if score is None or not math.isfinite(float(score.probe_mse)):
                continue
            if best_probe is None:
                best_probe = float(score.probe_mse)
            else:
                best_probe = min(float(best_probe), float(score.probe_mse))
        status_override = None
        extra_hard_constraints: dict[str, Any] = {}
        extra_metadata: dict[str, Any] = {}
        if mode_name == "expanded":
            expanded_signals = _expanded_outer_family_annotations(ctx)
            if expanded_signals:
                extra_hard_constraints["expanded_family_signals"] = expanded_signals
                extra_metadata["expanded_family_signals"] = expanded_signals
                for signal_name, summary in expanded_signals.items():
                    signal_status = str(summary.get("status", "") or "")
                    if signal_status:
                        extra_hard_constraints[f"{signal_name}_status"] = signal_status
                        extra_metadata[f"{signal_name}_status"] = signal_status
                    signal_score = summary.get("score", None)
                    if signal_score is not None:
                        try:
                            scalar = float(signal_score)
                        except Exception:
                            scalar = None
                        if scalar is not None and math.isfinite(float(scalar)):
                            extra_hard_constraints[f"{signal_name}_score"] = float(scalar)
                            extra_metadata[f"{signal_name}_score"] = float(scalar)
            domain_evidence = dict((ctx.stats.get("outer_family_evidence", {}) or {}).get("domain_hazard", {}) or {})
            domain_hard = dict(domain_evidence.get("hard_constraints", {}) or {})
            if bool(domain_hard.get("hazard_severe", False)) and str(family_name) in {"exp", "power"}:
                status_override = "domain_hazard"
                extra_hard_constraints["domain_hazard"] = True
                extra_hard_constraints["domain_hazard_status"] = str(domain_hard.get("status", "") or "")
                extra_hard_constraints["domain_singularity_margin_proxy"] = domain_hard.get("singularity_margin_proxy", None)
                extra_metadata["domain_hazard"] = True
                extra_metadata["domain_hazard_status"] = str(domain_hard.get("status", "") or "")
        evidence = build_named_outer_family_evidence(
            family_name,
            recursive_enable=bool(ctx.recursive_enable),
            wrappers_left=int(problem.wrappers_left),
            flat_rows_present=bool(flat_rows),
            target_dim_ok=bool(target_dim_ok),
            best_flat_probe_mse=best_flat_probe,
            seed_nodes=candidate_nodes,
            min_improvement_ratio=float(min_improvement_ratio),
            best_probe_mse=best_probe,
            status_override=status_override,
            extra_hard_constraints=extra_hard_constraints,
            extra_metadata=extra_metadata,
            target_dim=problem.target_dim,
            active_vars=dict(problem.diagnostics or {}).get("active_vars", ()),
            recursion_level=int(problem.recursion_level),
            target_mode=str(ctx.target_mode or ""),
            target_mapping_kind=str(ctx.target_mapping_kind or ""),
            regime_metadata=extract_family_regime_metadata(dict(problem.diagnostics or {})),
        )
        _record_outer_family_evidence(ctx, family_name, evidence)
        _record_outer_family_precheck(
            ctx,
            family_name,
            status=family_evidence_status(evidence, family_name),
            candidate_count=int(len(candidate_nodes)),
            best_probe_mse=best_probe,
            improvement_ratio=dict(evidence.metadata or {}).get("improvement_ratio", None),
        )
        return bool(family_evidence_should_run(evidence, family_name))
    if not bool(ctx.recursive_enable):
        _record_outer_family_precheck(ctx, family_name, status="disabled")
        return False
    if int(problem.wrappers_left) <= 0:
        _record_outer_family_precheck(ctx, family_name, status="no_wrappers_left")
        return False
    if not flat_rows:
        _record_outer_family_precheck(ctx, family_name, status="no_flat_rows")
        return False
    child_target_dim = _outer_family_child_target_dim(family_name, problem, ctx=ctx)
    if ctx.dm and child_target_dim is None:
        _record_outer_family_precheck(ctx, family_name, status="nondimensionless_target")
        return False
    best_flat_probe = float(flat_rows[0].local_probe_mse)
    if not math.isfinite(best_flat_probe):
        _record_outer_family_precheck(ctx, family_name, status="nonfinite_flat_probe")
        return False
    candidate_nodes = _outer_family_candidate_nodes(
        flat_rows,
        problem=problem,
        ctx=ctx,
        max_seeds=max(1, int(max_seeds)),
        child_target_dim=child_target_dim,
    )
    if not candidate_nodes:
        _record_outer_family_precheck(ctx, family_name, status="no_candidate_nodes")
        return False
    best_probe = float("inf")
    for node in candidate_nodes:
        try:
            g_fit = eval_node(node, problem.xf)
            g_probe = eval_node(node, problem.xp)
        except Exception:
            continue
        if (not torch.isfinite(g_fit).all()) or (not torch.isfinite(g_probe).all()):
            continue
        score = _score_outer_family_forward(family_name, g_fit, g_probe, problem=problem)
        if score is not None and math.isfinite(float(score.probe_mse)):
            best_probe = min(best_probe, float(score.probe_mse))
    if not math.isfinite(best_probe):
        _record_outer_family_precheck(
            ctx,
            family_name,
            status="no_finite_fit",
            candidate_count=int(len(candidate_nodes)),
        )
        return False
    improve_ratio = float(best_probe / max(best_flat_probe, 1.0e-12))
    status = "triggered" if improve_ratio <= float(min_improvement_ratio) else "insufficient_improvement"
    _record_outer_family_precheck(
        ctx,
        family_name,
        status=status,
        candidate_count=int(len(candidate_nodes)),
        best_probe_mse=float(best_probe),
        improvement_ratio=float(improve_ratio),
    )
    return bool(status == "triggered")


def _collect_named_outer_family_candidates(
    family_name: str,
    problem: _LocalProblem,
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
) -> tuple[_LocalProblem, list[tuple[str, tuple]]]:
    child_target_dim = _outer_family_child_target_dim(family_name, problem, ctx=ctx)
    child_problem = _LocalProblem(
        xf=problem.xf,
        tf=problem.tf,
        wf=problem.wf,
        xp=problem.xp,
        tp=problem.tp,
        wp=problem.wp,
        target_dim=child_target_dim,
        confidence=float(problem.confidence),
        valid_frac=float(problem.valid_frac),
        wrappers_left=max(0, int(problem.wrappers_left) - 1),
        recursion_level=int(problem.recursion_level) + 1,
        trace=tuple(list(problem.trace) + [f"{family_name}_arg"]),
        teacher_spec=normalize_local_teacher_spec(problem.teacher_spec) if problem.teacher_spec is not None else None,
    )
    arg_candidates, _source_counts, _enum_count, _enum_depth = _collect_flat_candidate_sources(
        child_problem,
        ctx=ctx,
        include_legacy_aux=False,
    )
    existing_keys = {node_str(node) for _, node in arg_candidates}
    for row in list(flat_rows or [])[: max(1, int(ctx.recursive_seed_cap))]:
        key = node_str(row.node)
        if key in existing_keys:
            continue
        if ctx.dm:
            try:
                nd = node_dims(row.node, ctx.var_dims)
            except Exception:
                nd = None
            if nd is None or child_target_dim is None or not dims_eq(nd, child_target_dim):
                continue
        arg_candidates.append(("flat_seed", row.node))
        existing_keys.add(key)
    return child_problem, arg_candidates


def _build_named_outer_family_branches(
    family_name: str,
    problem: _LocalProblem,
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
    min_improvement_ratio: float,
    max_seeds: int,
) -> list[_ScoredLocalCandidate]:
    started = time.perf_counter()
    if family_name == "periodic":
        rows = _build_periodic_forward_branches(problem, flat_rows, ctx=ctx)
        if rows:
            _record_outer_family_rows(ctx, "periodic", int(len(rows)))
        _record_outer_family_wall(ctx, "periodic", time.perf_counter() - started)
        return rows
    if not _should_run_named_outer_family(
        family_name,
        flat_rows,
        problem=problem,
        ctx=ctx,
        min_improvement_ratio=float(min_improvement_ratio),
        max_seeds=int(max_seeds),
    ):
        return []
    child_problem, arg_candidates = _collect_named_outer_family_candidates(
        family_name,
        problem,
        flat_rows,
        ctx=ctx,
    )
    out: list[_ScoredLocalCandidate] = []
    for source, g_node in arg_candidates:
        out.extend(
            _score_outer_family_candidate(
                g_node,
                problem=problem,
                ctx=ctx,
                family_name=family_name,
                source=f"outer_family:{family_name}:{source}",
                trace=tuple(list(child_problem.trace) + [node_str(g_node)]),
            )
        )
    _record_outer_family_wall(ctx, family_name, time.perf_counter() - started)
    return out


def _build_periodic_forward_branches(
    problem: _LocalProblem,
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
) -> list[_ScoredLocalCandidate]:
    started = time.perf_counter()
    """Search for trig wrappers using the periodic forward objective.

    Instead of inverting through cos/sin (which fails for multi-period
    arguments), enumerate candidate inner arguments *g* and score each by
    fitting ``t ≈ a sin(g) + b cos(g) + c``.  This absorbs all 2πk
    branch ambiguity into the periodic basis.

    Trigger conditions:
    - target dimension is dimensionless (trig wrappers require this)
    - target is bounded (consistent with sinusoidal range)
    - explicit inverse confidence for sin/cos would be low
    """
    if not _should_run_periodic_forward(problem, flat_rows, ctx=ctx):
        return []

    # Check target is bounded (consistent with sinusoidal output)
    t_range = float((problem.tp.max() - problem.tp.min()).item())
    if t_range > 2.5:
        # Target range exceeds what a ≈ ±1 sinusoid can produce;
        # still allow because the fit includes (a, b, c) with free amplitude
        pass

    # Build child problem for argument search (dimensionless target)
    child_target_dim = (0.0,) * len(ctx.var_dims[0]) if ctx.dm else None

    # Collect argument candidates from the same flat enumeration machinery
    child_problem = _LocalProblem(
        xf=problem.xf,
        tf=problem.tf,  # not used for scoring; we score with sinusoidal fit
        wf=problem.wf,
        xp=problem.xp,
        tp=problem.tp,
        wp=problem.wp,
        target_dim=child_target_dim,
        confidence=float(problem.confidence),
        valid_frac=float(problem.valid_frac),
        wrappers_left=max(0, int(problem.wrappers_left) - 1),
        recursion_level=int(problem.recursion_level) + 1,
        trace=tuple(list(problem.trace) + ["periodic_arg"]),
        teacher_spec=normalize_local_teacher_spec(problem.teacher_spec) if problem.teacher_spec is not None else None,
    )

    # Enumerate candidate arguments (dimensionally filtered to dim0)
    arg_candidates, _source_counts, _enum_count, _enum_depth = _collect_flat_candidate_sources(
        child_problem,
        ctx=ctx,
        include_legacy_aux=False,
    )

    # Also include top flat-solve results as argument candidates (they
    # may have the right inner structure even if they score poorly as
    # direct substitutions)
    for row in list(flat_rows or [])[:max(1, int(ctx.recursive_seed_cap))]:
        key = node_str(row.node)
        if not any(node_str(n) == key for _, n in arg_candidates):
            if ctx.dm:
                try:
                    nd = node_dims(row.node, ctx.var_dims)
                except Exception:
                    nd = None
                if nd is not None and dims_eq(nd, child_target_dim):
                    arg_candidates.append(("flat_seed", row.node))
            else:
                arg_candidates.append(("flat_seed", row.node))

    # Score each argument candidate with the sinusoidal forward objective
    periodic_rows: list[_ScoredLocalCandidate] = []
    for source, g_node in arg_candidates:
        scored = _score_candidate_sinusoidal(
            g_node,
            problem=problem,
            ctx=ctx,
            wrapper_op="sincos",
            source=f"periodic_forward:{source}",
            trace=tuple(list(problem.trace) + [f"periodic:{node_str(g_node)}"]),
        )
        periodic_rows.extend(scored)

    # If the recursive solver found any good inner arguments at the current
    # level, also try recursively decomposing the argument further
    if int(child_problem.wrappers_left) > 0 and periodic_rows:
        # Use top periodic arguments as anchors for binary decomposition
        def _extract_arg_key(r):
            if isinstance(r.node, tuple) and r.node[0] in _PERIODIC_OPS and len(r.node) == 2:
                inner = r.node[1]
                if isinstance(inner, tuple):
                    return node_str(inner)
            return ""
        best_args = sorted(
            [(_extract_arg_key(r), r) for r in periodic_rows],
            key=lambda item: item[1].local_probe_mse,
        )
        seen_args: set[str] = set()
        recursive_arg_candidates: list[tuple] = []
        for arg_key, scored_row in best_args[:max(1, int(ctx.recursive_seed_cap))]:
            if arg_key in seen_args:
                continue
            seen_args.add(arg_key)
            # Extract the argument node from the wrapped candidate
            inner = scored_row.node
            if isinstance(inner, tuple) and inner[0] in _PERIODIC_OPS and len(inner) == 2:
                recursive_arg_candidates.append(inner[1])

        # Try binary compositions of top argument candidates
        for anchor_node in recursive_arg_candidates[:3]:
            for bin_op in ("add", "mul", "sub"):
                for slot in (1, 2):
                    for _src, cand in arg_candidates[:max(1, int(ctx.recursive_seed_cap))]:
                        if node_str(cand) == node_str(anchor_node):
                            continue
                        if slot == 1:
                            composed = _simplify_node((bin_op, cand, anchor_node))
                        else:
                            composed = _simplify_node((bin_op, anchor_node, cand))
                        scored = _score_candidate_sinusoidal(
                            composed,
                            problem=problem,
                            ctx=ctx,
                            wrapper_op="sincos",
                            source=f"periodic_forward:recursive_{bin_op}",
                            trace=tuple(list(problem.trace) + [f"periodic_rec:{bin_op}:{node_str(composed)}"]),
                        )
                        periodic_rows.extend(scored)

    ctx.stats["periodic_forward_used"] = bool(len(periodic_rows) > 0) or bool(ctx.stats.get("periodic_forward_used", False))
    ctx.stats["periodic_forward_candidate_count"] = int(ctx.stats.get("periodic_forward_candidate_count", 0) or 0) + int(len(periodic_rows))

    _stats_add_time(ctx.stats, "periodic_forward_wall_seconds", time.perf_counter() - started)
    return periodic_rows


def _build_profiled_outer_family_branches(
    problem: _LocalProblem,
    flat_rows: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
) -> tuple[list[_ScoredLocalCandidate], dict[str, Any]]:
    started = time.perf_counter()
    all_rows: list[_ScoredLocalCandidate] = []
    meta: dict[str, Any] = {"families": {}}
    if bool(ctx.family_battery_enable):
        _record_expanded_family_evidence(problem, ctx=ctx)
    family_specs = (
        default_outer_family_battery_specs(
            periodic_min_improvement_ratio=float(_PERIODIC_PRECHECK_MIN_IMPROVEMENT_RATIO),
            periodic_precheck_max_seeds=int(_PERIODIC_PRECHECK_MAX_SEEDS),
            default_min_improvement_ratio=float(_OUTER_FAMILY_PRECHECK_DEFAULT_RATIO),
        )
        if bool(ctx.family_battery_enable)
        else _OUTER_FAMILY_SPECS
    )
    for spec in family_specs:
        family_rows = _build_named_outer_family_branches(
            spec.name,
            problem,
            flat_rows,
            ctx=ctx,
            min_improvement_ratio=float(spec.min_improvement_ratio),
            max_seeds=int(spec.precheck_max_seeds),
        )
        family_meta = {
            "used": bool(len(family_rows) > 0),
            "candidate_count": int(len(family_rows)),
            "precheck_status": str((ctx.stats.get("outer_family_precheck_status", {}) or {}).get(str(spec.name), "") or ""),
            "precheck_candidate_count": int((ctx.stats.get("outer_family_precheck_candidate_count", {}) or {}).get(str(spec.name), 0) or 0),
            "precheck_best_probe_mse": (ctx.stats.get("outer_family_precheck_best_probe_mse", {}) or {}).get(str(spec.name), None),
            "precheck_improvement_ratio": (ctx.stats.get("outer_family_precheck_improvement_ratio", {}) or {}).get(str(spec.name), None),
        }
        if bool(ctx.family_battery_enable):
            family_meta["evidence"] = dict((ctx.stats.get("outer_family_evidence", {}) or {}).get(str(spec.name), {}) or {})
        meta["families"][str(spec.name)] = family_meta
        all_rows.extend(family_rows)
    _stats_add_time(ctx.stats, "outer_family_dispatch_wall_seconds", time.perf_counter() - started)
    return all_rows, meta


def _build_unary_recursive_branches(
    problem: _LocalProblem,
    *,
    ctx: _SolverContext,
) -> list[_RecursiveBranch]:
    branches: list[_RecursiveBranch] = []
    unary_ops = tuple(op for op in UNARY_OPS if op in ("neg", "sqrt", "sqr", "sin", "cos", "exp", "log"))
    branch_beam_width = max(1, min(int(ctx.branch_beam_width), 2))
    for op in unary_ops:
        child_target_dim = _unary_child_dim(str(op), problem.target_dim, ctx=ctx)
        if ctx.dm and child_target_dim is None:
            continue
        try:
            fit_branches = _invert_unary_context_branches(
                str(op),
                problem.tf,
                child_pred_ref=None,
                safe_eps=float(ctx.safe_eps),
                confidence_mode=str(ctx.confidence_mode),
                confidence_target_gain=float(ctx.confidence_target_gain),
                confidence_floor=float(ctx.confidence_floor),
                branch_beam_width=int(branch_beam_width),
            )
            probe_branches = _invert_unary_context_branches(
                str(op),
                problem.tp,
                child_pred_ref=None,
                safe_eps=float(ctx.safe_eps),
                confidence_mode=str(ctx.confidence_mode),
                confidence_target_gain=float(ctx.confidence_target_gain),
                confidence_floor=float(ctx.confidence_floor),
                branch_beam_width=int(branch_beam_width),
            )
        except Exception:
            continue
        probe_by_token = {str(token): (t, m, c, pw, note) for t, m, c, pw, note, token in probe_branches}
        for fit_t, fit_m, fit_c, fit_pw, fit_note, token in fit_branches:
            probe_row = probe_by_token.get(str(token), None)
            if probe_row is None:
                continue
            probe_t, probe_m, probe_c, probe_pw, _probe_note = probe_row
            trace_token = f"{op}:{token}"
            child_problem = _build_child_problem(
                problem,
                child_target_fit=fit_t,
                fit_mask=fit_m,
                fit_weight=fit_pw,
                fit_conf=float(fit_c),
                child_target_probe=probe_t,
                probe_mask=probe_m,
                probe_weight=probe_pw,
                probe_conf=float(probe_c),
                child_target_dim=child_target_dim,
                trace_token=trace_token,
                ctx=ctx,
                transform_kind="unary",
                transform_op=str(op),
                transform_slot=1,
            )
            if child_problem is None:
                continue
            priority = (
                float(child_problem.confidence) * float(child_problem.valid_frac),
                float(child_problem.confidence),
                -int(problem.recursion_level),
                str(fit_note),
            )
            branches.append(
                _RecursiveBranch(
                    source=f"recursive_unary:{op}",
                    child_problem=child_problem,
                    child_target_dim=child_target_dim,
                    wrap_kind="unary",
                    op=str(op),
                    slot=1,
                    anchor_node=None,
                    priority=priority,
                )
            )
    branches.sort(key=lambda row: row.priority, reverse=True)
    return branches


def _select_binary_recursive_anchors(
    current_level_rows: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
) -> list[_ScoredLocalCandidate]:
    best_by_key: dict[str, _ScoredLocalCandidate] = {}
    for row in list(current_level_rows or []):
        key = node_str(row.node)
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = row
            continue
        prev_gap = float(prev.calibration_gap) if prev.calibration_gap is not None and math.isfinite(float(prev.calibration_gap)) else 0.0
        row_gap = float(row.calibration_gap) if row.calibration_gap is not None and math.isfinite(float(row.calibration_gap)) else 0.0
        prev_key = (
            float(prev.local_probe_mse),
            prev_gap,
            int(node_size(prev.node)),
            str(prev.source),
        )
        row_key = (
            float(row.local_probe_mse),
            row_gap,
            int(node_size(row.node)),
            str(row.source),
        )
        if row_key < prev_key:
            best_by_key[key] = row
    anchors = sorted(
        best_by_key.values(),
        key=lambda row: (
            float(row.local_probe_mse),
            float(row.calibration_gap) if row.calibration_gap is not None and math.isfinite(float(row.calibration_gap)) else 0.0,
            int(node_size(row.node)),
            str(row.source),
        ),
    )
    return anchors[: max(1, int(ctx.recursive_seed_cap))]


def _build_binary_recursive_branches(
    problem: _LocalProblem,
    anchors: Sequence[_ScoredLocalCandidate],
    *,
    ctx: _SolverContext,
) -> list[_RecursiveBranch]:
    branches: list[_RecursiveBranch] = []
    for anchor in list(anchors or [])[: max(1, int(ctx.recursive_seed_cap))]:
        anchor_node = anchor.node
        try:
            other_fit = eval_node(anchor_node, problem.xf)
            other_probe = eval_node(anchor_node, problem.xp)
        except Exception:
            continue
        if (not torch.isfinite(other_fit).all()) or (not torch.isfinite(other_probe).all()):
            continue
        other_grad_fit = None
        other_grad_probe = None
        other_d2_fit = None
        other_d2_probe = None
        if (
            (problem.grad_fit is not None and _teacher_source_is_exact(_diagnostic_teacher_source(problem, split="fit")))
            or (problem.grad_probe is not None and _teacher_source_is_exact(_diagnostic_teacher_source(problem, split="probe")))
        ):
            _fit_value, other_grad_fit, other_d2_fit = _autograd_node_jets(
                anchor_node,
                problem.xf,
                capture_d2=problem.d2_fit is not None,
            )
            _probe_value, other_grad_probe, other_d2_probe = _autograd_node_jets(
                anchor_node,
                problem.xp,
                capture_d2=problem.d2_probe is not None,
            )
        anchor_dim = None
        if ctx.dm:
            try:
                anchor_dim = node_dims(anchor_node, ctx.var_dims)
            except Exception:
                anchor_dim = None
            if anchor_dim is None:
                continue
        for op in BINARY_OPS:
            for child_slot in (1, 2):
                if ctx.dm and op in ("add", "sub") and (problem.target_dim is None or not dims_eq(anchor_dim, problem.target_dim)):
                    continue
                child_target_dim = _binary_child_dim(str(op), int(child_slot), problem.target_dim, anchor_dim)
                if ctx.dm and child_target_dim is None:
                    continue
                try:
                    fit_t, fit_m, fit_c, fit_pw, _fit_note = _invert_binary_context(
                        str(op),
                        problem.tf,
                        child_slot=int(child_slot),
                        other_pred=other_fit,
                        safe_eps=float(ctx.safe_eps),
                        confidence_mode=str(ctx.confidence_mode),
                        confidence_target_gain=float(ctx.confidence_target_gain),
                        confidence_floor=float(ctx.confidence_floor),
                    )
                    probe_t, probe_m, probe_c, probe_pw, _probe_note = _invert_binary_context(
                        str(op),
                        problem.tp,
                        child_slot=int(child_slot),
                        other_pred=other_probe,
                        safe_eps=float(ctx.safe_eps),
                        confidence_mode=str(ctx.confidence_mode),
                        confidence_target_gain=float(ctx.confidence_target_gain),
                        confidence_floor=float(ctx.confidence_floor),
                    )
                except Exception:
                    continue
                trace_token = f"{op}:{child_slot}:{node_str(anchor_node)}"
                child_problem = _build_child_problem(
                    problem,
                    child_target_fit=fit_t,
                    fit_mask=fit_m,
                    fit_weight=fit_pw,
                    fit_conf=float(fit_c),
                    child_target_probe=probe_t,
                    probe_mask=probe_m,
                    probe_weight=probe_pw,
                    probe_conf=float(probe_c),
                    child_target_dim=child_target_dim,
                    trace_token=trace_token,
                    ctx=ctx,
                    transform_kind="binary",
                    transform_op=str(op),
                    transform_slot=int(child_slot),
                    other_fit=other_fit,
                    other_probe=other_probe,
                    other_grad_fit=other_grad_fit,
                    other_grad_probe=other_grad_probe,
                    other_d2_fit=other_d2_fit,
                    other_d2_probe=other_d2_probe,
                )
                if child_problem is None:
                    continue
                priority = (
                    float(child_problem.confidence) * float(child_problem.valid_frac),
                    -float(anchor.local_probe_mse),
                    -int(node_size(anchor_node)),
                    str(trace_token),
                )
                branches.append(
                    _RecursiveBranch(
                        source=f"recursive_binary:{op}:{child_slot}",
                        child_problem=child_problem,
                        child_target_dim=child_target_dim,
                        wrap_kind="binary",
                        op=str(op),
                        slot=int(child_slot),
                        anchor_node=anchor_node,
                        priority=priority,
                    )
                )
    branches.sort(key=lambda row: row.priority, reverse=True)
    return branches


def _tensor_digest(x: torch.Tensor | None) -> tuple[Any, ...]:
    if x is None:
        return ("none",)
    xx = x.detach()
    if xx.device.type != "cpu":
        xx = xx.cpu()
    xx = xx.contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(xx.dtype).encode("utf-8"))
    h.update(str(tuple(int(v) for v in xx.shape)).encode("utf-8"))
    h.update(xx.numpy().tobytes())
    return ("tensor", str(h.hexdigest()))


def _dim_fingerprint(d) -> Any:
    if d is None:
        return None
    if isinstance(d, (tuple, list)):
        out: list[Any] = []
        for v in d:
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(round(float(v), 8))
            else:
                out.append(v)
        return tuple(out)
    return d


def _trace_suffix(trace: Sequence[str], prefix: Sequence[str]) -> tuple[str, ...]:
    tt = tuple(str(v) for v in (trace or ()))
    pp = tuple(str(v) for v in (prefix or ()))
    if len(tt) >= len(pp) and tt[: len(pp)] == pp:
        return tt[len(pp) :]
    return tt


def _normalize_cached_rows(
    rows: Sequence[_ScoredLocalCandidate],
    *,
    problem_trace: Sequence[str],
) -> tuple[_ScoredLocalCandidate, ...]:
    return tuple(
        replace(row, trace=_trace_suffix(row.trace, problem_trace))
        for row in list(rows or [])
    )


def _restore_cached_rows(
    rows: Sequence[_ScoredLocalCandidate],
    *,
    problem_trace: Sequence[str],
) -> list[_ScoredLocalCandidate]:
    prefix = tuple(str(v) for v in (problem_trace or ()))
    return [
        replace(row, trace=prefix + tuple(str(v) for v in (row.trace or ())))
        for row in list(rows or [])
    ]


def _local_problem_memo_key(
    problem: _LocalProblem,
    *,
    include_legacy_aux: bool,
) -> tuple[Any, ...]:
    return (
        "local_problem_v1",
        bool(include_legacy_aux),
        int(problem.wrappers_left),
        int(problem.recursion_level),
        _dim_fingerprint(problem.target_dim),
        round(float(problem.confidence), 8),
        round(float(problem.valid_frac), 8),
        _tensor_digest(problem.xf),
        _tensor_digest(problem.tf),
        _tensor_digest(problem.wf),
        _tensor_digest(problem.xp),
        _tensor_digest(problem.tp),
        _tensor_digest(problem.wp),
    )


def _continuation_frame_payload(
    *,
    wrap_kind: str,
    op: str,
    slot: int,
    anchor_node,
) -> dict[str, Any]:
    return {
        "wrap_kind": str(wrap_kind),
        "op": str(op),
        "slot": int(slot),
        "anchor_node": anchor_node,
    }


def _continuation_frame_token(frame: Mapping[str, Any] | None) -> str:
    if not isinstance(frame, Mapping):
        return ""
    wrap_kind = str(frame.get("wrap_kind", "") or "")
    op = str(frame.get("op", "") or "")
    try:
        slot = int(frame.get("slot", 0) or 0)
    except Exception:
        slot = 0
    anchor_token = ""
    anchor_node = frame.get("anchor_node", None)
    if anchor_node is not None:
        try:
            anchor_token = str(node_str(anchor_node))
        except Exception:
            anchor_token = str(anchor_node)
    parts = [wrap_kind, op, str(slot)]
    if anchor_token:
        parts.append(anchor_token)
    return ":".join(part for part in parts if str(part))


def _continuation_frames_tokens(frames: Sequence[Mapping[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for frame in list(frames or []):
        token = _continuation_frame_token(frame)
        if token:
            out.append(str(token))
    return out


def _subproblem_problem_id(
    *,
    problem_kind: str,
    path: Sequence[int] | tuple[int, ...],
    target_mode: str,
    target_mapping_kind: str,
    target_dim: Any,
    continuation_frames: Sequence[Mapping[str, Any]] | None,
    wrappers_left: int,
    recursion_level: int,
    trace: Sequence[str] | tuple[str, ...],
) -> str:
    digest = hashlib.sha1()
    parts = [
        str(problem_kind or ""),
        ",".join(str(int(v)) for v in tuple(path or ())),
        str(target_mode or ""),
        str(target_mapping_kind or ""),
        repr(_dim_fingerprint(target_dim)),
        "|".join(_continuation_frames_tokens(continuation_frames)),
        str(int(wrappers_left)),
        str(int(recursion_level)),
        "|".join(str(v) for v in tuple(trace or ())),
    ]
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def _local_problem_to_subproblem_spec(
    problem: _LocalProblem,
    *,
    parent_expr,
    path: Sequence[int] | tuple[int, ...],
    target_mode: str,
    target_mapping_kind: str,
    continuation_frames: Sequence[Mapping[str, Any]] | None = None,
    hole_sub=None,
    direction: str = "inside_out",
    metadata: Mapping[str, Any] | None = None,
    witness_jets_enable: bool = False,
    witness_d2_enable: bool = False,
    witness_max_rows: int = 64,
    active_var_screen_enable: bool = False,
    active_var_grad_tol: float = 1.0e-3,
    active_var_max_count: int = 4,
) -> SubproblemSpec:
    diagnostics = dict(problem.diagnostics or {})
    diagnostics.update({
        "confidence": float(problem.confidence),
        "valid_frac": float(problem.valid_frac),
        "trace": tuple(str(v) for v in tuple(problem.trace or ())),
    })
    teacher_spec = normalize_local_teacher_spec(
        problem.teacher_spec,
        default_source=LOCAL_TEACHER_SOURCE_NUMERIC,
    ) if problem.teacher_spec is not None else build_numeric_local_teacher_spec(
        reason="subproblem_spec_default_numeric",
    )
    fit_jets: dict[str, Any] | None = None
    probe_jets: dict[str, Any] | None = None
    grad_fit = None
    grad_probe = None
    d2_fit = None
    d2_probe = None
    if bool(witness_jets_enable) or bool(active_var_screen_enable):
        fit_jets = _carry_local_problem_jets(
            problem,
            x=problem.xf,
            split="fit",
            include_d2=bool(witness_jets_enable and witness_d2_enable),
            teacher_spec=teacher_spec,
        )
        if fit_jets is None:
            fit_jets = evaluate_local_teacher_jets(
                problem.xf,
                problem.tf,
                w=problem.wf,
                include_d2=bool(witness_jets_enable and witness_d2_enable),
                max_rows=max(4, int(witness_max_rows)),
                teacher_spec=teacher_spec,
                teacher_runtime=problem.teacher_runtime,
            )
        probe_jets = _carry_local_problem_jets(
            problem,
            x=problem.xp,
            split="probe",
            include_d2=bool(witness_jets_enable and witness_d2_enable),
            teacher_spec=teacher_spec,
        )
        if probe_jets is None:
            probe_jets = evaluate_local_teacher_jets(
                problem.xp,
                problem.tp,
                w=problem.wp,
                include_d2=bool(witness_jets_enable and witness_d2_enable),
                max_rows=max(4, int(witness_max_rows)),
                teacher_spec=teacher_spec,
                teacher_runtime=problem.teacher_runtime,
            )
        diagnostics.update(
            {
                "fit_jet_source": str(fit_jets.get("source", "") or ""),
                "probe_jet_source": str(probe_jets.get("source", "") or ""),
                "fit_jet_requested_source": str(fit_jets.get("requested_source", "") or ""),
                "probe_jet_requested_source": str(probe_jets.get("requested_source", "") or ""),
                "fit_jet_fallback_used": bool(fit_jets.get("fallback_used", False)),
                "probe_jet_fallback_used": bool(probe_jets.get("fallback_used", False)),
            }
        )
        if bool(witness_jets_enable):
            grad_fit = fit_jets.get("grad", None)
            grad_probe = probe_jets.get("grad", None)
            d2_fit = fit_jets.get("d2", None)
            d2_probe = probe_jets.get("d2", None)
            diagnostics.update(
                {
                    "witness_jets_enabled": True,
                    "witness_hdiag_enabled": bool(witness_d2_enable),
                    "witness_fit_jet_status": str(fit_jets.get("status", "") or ""),
                    "witness_probe_jet_status": str(probe_jets.get("status", "") or ""),
                    "witness_fit_jet_source": str(fit_jets.get("source", "") or ""),
                    "witness_probe_jet_source": str(probe_jets.get("source", "") or ""),
                    "witness_fit_support_count": int(fit_jets.get("support_count", 0) or 0),
                    "witness_probe_support_count": int(probe_jets.get("support_count", 0) or 0),
                    "witness_fit_neighbor_count": int(fit_jets.get("neighbor_count", 0) or 0),
                    "witness_probe_neighbor_count": int(probe_jets.get("neighbor_count", 0) or 0),
                }
            )
    nvars = int(problem.xf.shape[1]) if getattr(problem.xf, "ndim", 0) >= 2 else 1
    active_vars, active_var_diag = infer_subproblem_active_vars(
        hole_sub=hole_sub,
        continuation_frames=continuation_frames,
        grad_fit=(grad_fit if grad_fit is not None else (fit_jets or {}).get("grad", None)),
        grad_probe=(grad_probe if grad_probe is not None else (probe_jets or {}).get("grad", None)),
        nvars=int(nvars),
        screen_enable=bool(active_var_screen_enable),
        grad_tol=float(active_var_grad_tol),
        max_count=int(active_var_max_count),
    )
    diagnostics.update(active_var_diag)
    if bool(active_var_screen_enable):
        diagnostics["active_var_grad_tol"] = float(active_var_grad_tol)
        diagnostics["active_var_max_count"] = int(max(1, int(active_var_max_count)))
        diagnostics["active_var_fit_jet_status"] = str(((fit_jets or {}).get("status", "")) or "")
        diagnostics["active_var_probe_jet_status"] = str(((probe_jets or {}).get("status", "")) or "")
    witness = WitnessBundle(
        x_fit=problem.xf,
        t_fit=problem.tf,
        x_probe=problem.xp,
        t_probe=problem.tp,
        grad_fit=grad_fit,
        grad_probe=grad_probe,
        d2_fit=d2_fit,
        d2_probe=d2_probe,
        masks={
            "w_fit": problem.wf,
            "w_probe": problem.wp,
        },
        diagnostics=diagnostics,
    )
    metadata_out = dict(metadata or {})
    metadata_out["hole_sub"] = hole_sub
    metadata_out["confidence"] = float(problem.confidence)
    metadata_out["valid_frac"] = float(problem.valid_frac)
    metadata_out["teacher_spec"] = dict(teacher_spec)
    metadata_out["trace"] = tuple(str(v) for v in tuple(problem.trace or ()))
    metadata_out.update(active_var_diag)
    return SubproblemSpec(
        problem_id=_subproblem_problem_id(
            problem_kind="local_problem",
            path=path,
            target_mode=str(target_mode or ""),
            target_mapping_kind=str(target_mapping_kind or ""),
            target_dim=problem.target_dim,
            continuation_frames=continuation_frames,
            wrappers_left=int(problem.wrappers_left),
            recursion_level=int(problem.recursion_level),
            trace=tuple(problem.trace or ()),
        ),
        problem_kind="local_problem",
        parent_expr=parent_expr,
        path=tuple(int(v) for v in tuple(path or ())),
        direction=str(direction or "inside_out"),
        target_mode=str(target_mode or ""),
        target_mapping_kind=str(target_mapping_kind or ""),
        target_dim=problem.target_dim,
        continuation_frames=tuple(dict(frame) for frame in list(continuation_frames or [])),
        wrappers_left=int(problem.wrappers_left),
        recursion_level=int(problem.recursion_level),
        active_vars=tuple(int(v) for v in tuple(active_vars or ())),
        witness=witness,
        metadata=metadata_out,
    )


def _subproblem_spec_to_local_problem(spec: SubproblemSpec | None) -> tuple[_LocalProblem | None, list[dict[str, Any]], Any]:
    if spec is None or str(spec.problem_kind or "") != "local_problem" or spec.witness is None:
        return None, [], None
    witness = spec.witness
    xf = witness.x_fit
    tf = witness.t_fit
    xp = witness.x_probe
    tp = witness.t_probe
    if xf is None or tf is None or xp is None or tp is None:
        return None, [], None
    diagnostics = dict(witness.diagnostics or {})
    masks = dict(witness.masks or {})
    if "active_vars" not in diagnostics and tuple(spec.active_vars or ()):
        diagnostics["active_vars"] = [int(v) for v in tuple(spec.active_vars or ())]
    try:
        confidence = float(diagnostics.get("confidence", dict(spec.metadata or {}).get("confidence", 1.0)) or 1.0)
    except Exception:
        confidence = 1.0
    try:
        valid_frac = float(diagnostics.get("valid_frac", dict(spec.metadata or {}).get("valid_frac", 1.0)) or 1.0)
    except Exception:
        valid_frac = 1.0
    teacher_spec = normalize_local_teacher_spec(
        dict(spec.metadata or {}).get("teacher_spec", None),
        default_source=LOCAL_TEACHER_SOURCE_NUMERIC,
    ) if dict(spec.metadata or {}).get("teacher_spec", None) is not None else None
    local_problem = _LocalProblem(
        xf=_ensure_col(xf) if getattr(xf, "ndim", 0) == 1 else xf,
        tf=_ensure_col(tf),
        wf=masks.get("w_fit", None),
        xp=_ensure_col(xp) if getattr(xp, "ndim", 0) == 1 else xp,
        tp=_ensure_col(tp),
        wp=masks.get("w_probe", None),
        target_dim=spec.target_dim,
        confidence=float(min(1.0, max(0.0, confidence))),
        valid_frac=float(min(1.0, max(0.0, valid_frac))),
        wrappers_left=int(max(0, int(spec.wrappers_left))),
        recursion_level=int(max(0, int(spec.recursion_level))),
        trace=tuple(str(v) for v in (diagnostics.get("trace", dict(spec.metadata or {}).get("trace", ())) or ())),
        grad_fit=witness.grad_fit,
        grad_probe=witness.grad_probe,
        d2_fit=witness.d2_fit,
        d2_probe=witness.d2_probe,
        teacher_spec=teacher_spec,
        diagnostics=diagnostics,
    )
    continuation_frames = [dict(frame) for frame in list(spec.continuation_frames or []) if isinstance(frame, Mapping)]
    hole_sub = dict(spec.metadata or {}).get("hole_sub", None)
    return local_problem, continuation_frames, hole_sub


def _serialize_local_problem(problem: _LocalProblem) -> dict[str, Any]:
    return {
        "xf": problem.xf,
        "tf": problem.tf,
        "wf": problem.wf,
        "xp": problem.xp,
        "tp": problem.tp,
        "wp": problem.wp,
        "target_dim": problem.target_dim,
        "confidence": float(problem.confidence),
        "valid_frac": float(problem.valid_frac),
        "wrappers_left": int(problem.wrappers_left),
        "recursion_level": int(problem.recursion_level),
        "trace": [str(v) for v in tuple(problem.trace or ())],
        "grad_fit": problem.grad_fit,
        "grad_probe": problem.grad_probe,
        "d2_fit": problem.d2_fit,
        "d2_probe": problem.d2_probe,
        "teacher_spec": None if problem.teacher_spec is None else dict(normalize_local_teacher_spec(problem.teacher_spec)),
        "diagnostics": dict(problem.diagnostics or {}),
    }


def _deserialize_local_problem(payload: Mapping[str, Any] | None) -> _LocalProblem | None:
    if not isinstance(payload, Mapping):
        return None
    xf = payload.get("xf", None)
    tf = payload.get("tf", None)
    xp = payload.get("xp", None)
    tp = payload.get("tp", None)
    if xf is None or tf is None or xp is None or tp is None:
        return None
    try:
        wrappers_left = int(payload.get("wrappers_left", 0) or 0)
    except Exception:
        wrappers_left = 0
    try:
        recursion_level = int(payload.get("recursion_level", 0) or 0)
    except Exception:
        recursion_level = 0
    try:
        confidence = float(payload.get("confidence", 1.0) or 1.0)
    except Exception:
        confidence = 1.0
    try:
        valid_frac = float(payload.get("valid_frac", 1.0) or 1.0)
    except Exception:
        valid_frac = 1.0
    return _LocalProblem(
        xf=_ensure_col(xf) if getattr(xf, "ndim", 0) == 1 else xf,
        tf=_ensure_col(tf),
        wf=payload.get("wf", None),
        xp=_ensure_col(xp) if getattr(xp, "ndim", 0) == 1 else xp,
        tp=_ensure_col(tp),
        wp=payload.get("wp", None),
        target_dim=payload.get("target_dim", None),
        confidence=float(min(1.0, max(0.0, confidence))),
        valid_frac=float(min(1.0, max(0.0, valid_frac))),
        wrappers_left=int(max(0, wrappers_left)),
        recursion_level=int(max(0, recursion_level)),
        trace=tuple(str(v) for v in (payload.get("trace", ()) or ())),
        grad_fit=payload.get("grad_fit", None),
        grad_probe=payload.get("grad_probe", None),
        d2_fit=payload.get("d2_fit", None),
        d2_probe=payload.get("d2_probe", None),
        teacher_spec=(
            normalize_local_teacher_spec(payload.get("teacher_spec", None), default_source=LOCAL_TEACHER_SOURCE_NUMERIC)
            if payload.get("teacher_spec", None) is not None
            else None
        ),
        diagnostics=dict(payload.get("diagnostics", {}) or {}),
    )


def _apply_continuation_frames(node, frames: Sequence[Mapping[str, Any]] | None):
    out = node
    for frame in list(frames or []):
        if not isinstance(frame, Mapping):
            continue
        wrap_kind = str(frame.get("wrap_kind", "") or "")
        op = str(frame.get("op", "") or "")
        try:
            slot = int(frame.get("slot", 0) or 0)
        except Exception:
            slot = 0
        anchor_node = frame.get("anchor_node", None)
        if wrap_kind == "unary":
            out = (op, out)
        elif slot == 1:
            out = (op, out, anchor_node)
        else:
            out = (op, anchor_node, out)
    return out


def _recursive_branch_child_spec_state(
    branch: _RecursiveBranch,
    *,
    ctx: _SolverContext,
) -> dict[str, Any]:
    anchor_token = ""
    if branch.anchor_node is not None:
        try:
            anchor_token = str(node_str(branch.anchor_node))
        except Exception:
            anchor_token = ""
    frames = [
        _continuation_frame_payload(
            wrap_kind=str(branch.wrap_kind),
            op=str(branch.op),
            slot=int(branch.slot),
            anchor_node=branch.anchor_node,
        )
    ]
    continuation_key = _continuation_frames_tokens(frames)
    branch_id = ":".join(v for v in continuation_key if str(v))
    target_dim_fp = _dim_fingerprint(branch.child_problem.target_dim)
    if isinstance(target_dim_fp, tuple):
        target_dim_key = list(target_dim_fp)
    elif target_dim_fp is None:
        target_dim_key = []
    else:
        target_dim_key = [target_dim_fp]
    subproblem_spec = _local_problem_to_subproblem_spec(
        branch.child_problem,
        parent_expr=ctx.parent_node,
        path=ctx.hole_path,
        target_mode=str(ctx.target_mode or ""),
        target_mapping_kind=str(ctx.target_mapping_kind or ""),
        continuation_frames=frames,
        hole_sub=ctx.hole_sub,
        direction="outside_in",
        metadata={
            "preview_family": str(branch.source),
            "wrap_kind": str(branch.wrap_kind),
            "op": str(branch.op),
            "slot": int(branch.slot),
            "anchor_node": branch.anchor_node,
            "priority": tuple(branch.priority or ()),
        },
        witness_jets_enable=bool(ctx.witness_jets_enable),
        witness_d2_enable=bool(ctx.witness_d2_enable),
        witness_max_rows=int(ctx.witness_max_rows),
        active_var_screen_enable=bool(ctx.active_var_screen_enable),
        active_var_grad_tol=float(ctx.active_var_grad_tol),
        active_var_max_count=int(ctx.active_var_max_count),
    )
    spec_payload = wrap_subproblem_spec_payload(
        subproblem_spec,
        extra_payload={
            "problem": _serialize_local_problem(branch.child_problem),
            "continuation_frames": frames,
            "hole_sub": ctx.hole_sub,
        },
    )
    return {
        "spec_kind": "local_problem",
        "problem_id": str(subproblem_spec.problem_id),
        "path": [int(v) for v in tuple(ctx.hole_path or ())],
        "direction": str(subproblem_spec.direction or ""),
        "branch_id": str(branch_id or str(branch.source)),
        "continuation_key": [str(v) for v in continuation_key],
        "trace": [str(v) for v in tuple(branch.child_problem.trace or ())],
        "target_dim_key": list(target_dim_key),
        "wrappers_left": int(branch.child_problem.wrappers_left),
        "recursion_level": int(branch.child_problem.recursion_level),
        "confidence": float(branch.child_problem.confidence),
        "valid_frac": float(branch.child_problem.valid_frac),
        "preview_family": str(branch.source),
        "wrap_kind": str(branch.wrap_kind),
        "op": str(branch.op),
        "slot": int(branch.slot),
        "anchor_node": str(anchor_token),
        "priority_value": float(branch.priority[0]) if branch.priority else 0.0,
        "priority_gain": float(branch.priority[1]) if len(branch.priority) > 1 else 0.0,
        "spec_payload": spec_payload,
    }


def _solve_local_problem(
    problem: _LocalProblem,
    *,
    ctx: _SolverContext,
    include_legacy_aux: bool,
) -> tuple[list[_ScoredLocalCandidate], dict[str, Any]]:
    started = time.perf_counter()
    memo_key = _local_problem_memo_key(problem, include_legacy_aux=bool(include_legacy_aux))
    cached = ctx.memo_table.get(memo_key, None)
    if cached is not None:
        cached_rows, cached_meta = cached
        ctx.stats["memo_hit_count"] = int(ctx.stats.get("memo_hit_count", 0) or 0) + 1
        ctx.stats["memo_entry_count"] = int(len(ctx.memo_table))
        ctx.stats["memo_hit_row_count"] = int(ctx.stats.get("memo_hit_row_count", 0) or 0) + int(len(cached_rows))
        _stats_add_time(ctx.stats, "memo_hit_wall_seconds", time.perf_counter() - started)
        _stats_add_time(ctx.stats, "solve_local_problem_wall_seconds", time.perf_counter() - started)
        return _restore_cached_rows(cached_rows, problem_trace=problem.trace), copy.deepcopy(cached_meta)
    ctx.stats["memo_miss_count"] = int(ctx.stats.get("memo_miss_count", 0) or 0) + 1
    flat_rows, flat_meta = _flat_solve_local_problem(
        problem,
        ctx=ctx,
        include_legacy_aux=bool(include_legacy_aux),
    )
    recursive_rows: list[_ScoredLocalCandidate] = []
    should_recurse, best_probe, rel_mse = _should_recurse(problem, flat_rows, ctx=ctx)
    if int(problem.recursion_level) == 0:
        ctx.stats["flat_best_probe_mse"] = None if not math.isfinite(best_probe) else float(best_probe)
        ctx.stats["flat_best_probe_rel_mse"] = None if not math.isfinite(rel_mse) else float(rel_mse)
    recursive_meta: dict[str, Any] = {
        "used": False,
        "best_probe_mse": None if not math.isfinite(best_probe) else float(best_probe),
        "best_probe_rel_mse": None if not math.isfinite(rel_mse) else float(rel_mse),
        "branch_count": 0,
        "kept_branch_count": 0,
        "candidate_count": 0,
        "child_spec_states": [],
    }
    # Profiled outer-family stage: periodic/trig plus dimensionless exp,
    # power, and rational forward fits over inner candidates.
    outer_family_rows: list[_ScoredLocalCandidate] = []
    outer_family_meta: dict[str, Any] = {"families": {}}
    if should_recurse or (bool(ctx.recursive_enable) and int(problem.wrappers_left) > 0):
        outer_family_rows, outer_family_meta = _build_profiled_outer_family_branches(problem, flat_rows, ctx=ctx)
    periodic_meta = dict((outer_family_meta.get("families", {}) or {}).get("periodic", {}) or {})
    if not periodic_meta:
        periodic_meta = {"used": False, "candidate_count": 0}

    if should_recurse:
        branch_t0 = time.perf_counter()
        unary_branches = _build_unary_recursive_branches(problem, ctx=ctx)
        binary_anchor_rows = _select_binary_recursive_anchors(
            list(flat_rows) + list(outer_family_rows),
            ctx=ctx,
        )
        ctx.stats["recursive_binary_anchor_count"] = int(len(binary_anchor_rows))
        ctx.stats["recursive_binary_outer_family_anchor_count"] = int(
            sum(1 for row in binary_anchor_rows if str(row.source).startswith("outer_family:") or str(row.family))
        )
        binary_branches = _build_binary_recursive_branches(problem, binary_anchor_rows, ctx=ctx)
        _stats_add_time(ctx.stats, "recursive_branch_build_wall_seconds", time.perf_counter() - branch_t0)
        all_branches = unary_branches + binary_branches
        all_branches.sort(key=lambda row: row.priority, reverse=True)
        recursive_meta["branch_count"] = int(len(all_branches))
        kept_branches = all_branches[: max(1, int(ctx.recursive_branch_topk))]
        recursive_meta["kept_branch_count"] = int(len(kept_branches))
        recursive_meta["child_spec_states"] = [
            _recursive_branch_child_spec_state(branch, ctx=ctx)
            for branch in kept_branches
        ]
        if kept_branches:
            ctx.stats["recursive_used"] = True
            ctx.stats["recursive_expand_count"] = int(ctx.stats.get("recursive_expand_count", 0) or 0) + int(len(kept_branches))
            ctx.stats["recursive_depth_reached"] = max(
                int(ctx.stats.get("recursive_depth_reached", 0) or 0),
                int(problem.recursion_level) + 1,
            )
            recursive_meta["used"] = True
        child_solve_t0 = time.perf_counter()
        for branch in kept_branches:
            child_rows, _child_meta = _solve_local_problem(
                branch.child_problem,
                ctx=ctx,
                include_legacy_aux=False,
            )
            if not child_rows:
                continue
            for child_row in child_rows[: max(1, int(ctx.recursive_child_topk))]:
                try:
                    wrapped = _simplify_node(branch.wrap(child_row.node))
                except Exception:
                    continue
                if int(node_depth(wrapped)) > _candidate_depth_limit(problem, ctx):
                    continue
                if ctx.dm:
                    try:
                        wrapped_dim = node_dims(wrapped, ctx.var_dims)
                    except Exception:
                        wrapped_dim = None
                    if wrapped_dim is None or problem.target_dim is None or not dims_eq(wrapped_dim, problem.target_dim):
                        continue
                scored = _score_node_against_problem(
                    wrapped,
                    problem=problem,
                    ctx=ctx,
                    source=str(branch.source),
                    generation_kind="recursive",
                    confidence=float(child_row.confidence),
                    valid_frac=float(child_row.valid_frac),
                    trace=child_row.trace,
                )
                if scored is not None:
                    recursive_rows.append(scored)
        _stats_add_time(ctx.stats, "recursive_child_solve_wall_seconds", time.perf_counter() - child_solve_t0)
        recursive_meta["candidate_count"] = int(len(recursive_rows))
        ctx.stats["recursive_candidate_count"] = int(ctx.stats.get("recursive_candidate_count", 0) or 0) + int(len(recursive_rows))

    merged_rows = _dedup_scored_candidates(
        list(flat_rows) + list(recursive_rows) + list(outer_family_rows),
        complexity_penalty=float(ctx.complexity_penalty),
    )
    internal_keep = max(
        int(ctx.preview_topk),
        int(ctx.recursive_seed_cap),
        int(ctx.recursive_branch_topk),
        int(ctx.recursive_child_topk),
    )
    merged_rows = merged_rows[:internal_keep]
    solve_meta = {
        "flat": flat_meta,
        "recursive": recursive_meta,
        "periodic": periodic_meta,
        "outer_families": outer_family_meta,
    }
    ctx.memo_table[memo_key] = (
        _normalize_cached_rows(merged_rows, problem_trace=problem.trace),
        copy.deepcopy(solve_meta),
    )
    ctx.stats["memo_store_count"] = int(ctx.stats.get("memo_store_count", 0) or 0) + 1
    ctx.stats["memo_entry_count"] = int(len(ctx.memo_table))
    _stats_add_time(ctx.stats, "solve_local_problem_wall_seconds", time.perf_counter() - started)
    return merged_rows, solve_meta


def _candidate_to_preview_row(
    cand: _ScoredLocalCandidate,
    *,
    parent_node,
    beam_state: Mapping[str, Any],
    beam_rank: int,
    slate_id: str,
    path: tuple[int, ...],
    xf: torch.Tensor,
    tf: torch.Tensor,
    xp: torch.Tensor,
    tp: torch.Tensor,
    max_depth: int,
    var_dims,
    local_score_mode: str,
):
    try:
        child_expr = _simplify_node(replace_at(parent_node, path, cand.node))
    except Exception:
        return None
    if int(node_depth(child_expr)) > int(max_depth):
        return None
    if var_dims is not None:
        try:
            child_dim = node_dims(child_expr, var_dims)
        except Exception:
            child_dim = None
        if child_dim is None:
            return None
    local_mapping_kind, local_mapping_nparams = _local_mapping_preview(
        cand.node,
        xf=xf,
        tf=tf,
        xp=xp,
        tp=tp,
        poly_degree=int(beam_state.get("poly_degree", 0) or 0),
        local_score_mode=str(local_score_mode),
    )
    child_key = str(node_str(child_expr))
    cand_sub_size = int(node_size(cand.node))
    cand_sub_depth = int(node_depth(cand.node))
    child_size = int(node_size(child_expr))
    child_depth = int(node_depth(child_expr))
    parent_sub = beam_state.get("sub", None)
    parent_sub_size = int(node_size(parent_sub)) if parent_sub is not None else 0
    parent_sub_depth = int(node_depth(parent_sub)) if parent_sub is not None else 0
    parent_size = int(node_size(parent_node))
    parent_depth = int(node_depth(parent_node))
    cand_root_op = str(cand.node[0]) if isinstance(cand.node, tuple) and cand.node else ""
    return {
        "slate_id": str(slate_id),
        "expr": child_expr,
        "child_expr": str(child_key),
        "child_key": str(child_key),
        "path": path,
        "target_mode": str(beam_state.get("target_mode", "") or ""),
        "target_mapping_kind": str(beam_state.get("target_mapping_kind", "") or ""),
        "beam_rank": int(beam_rank),
        "local_rank": 0,
        "path_gain": float(beam_state.get("path_gain", 0.0) or 0.0),
        "route": "repair",
        "action": "inv_steer",
        "tuple_provenance": "inverse_spec_sr",
        "proposal_family": "inverse_spec_sr",
        "generation_source": "inverse_spec_solver",
        "inverse_spec_generation_kind": str(cand.generation_kind),
        "inverse_spec_family": str(cand.family or ""),
        "inverse_spec_family_payload": _jsonish_payload(cand.payload),
        "inverse_spec_recursion_depth": int(cand.recursion_depth),
        "inverse_spec_solver_confidence": float(cand.confidence),
        "inverse_spec_solver_valid_frac": float(cand.valid_frac),
        "inverse_spec_trace": [str(v) for v in cand.trace],
        "beam_state": beam_state,
        "local_candidate_count": 0,
        "local_probe_mse": float(cand.local_probe_mse),
        "local_fit_mse": float(cand.local_fit_mse),
        "local_fit_probe_gap": float(max(0.0, float(cand.local_probe_mse) - float(cand.local_fit_mse))),
        "surrogate_local_probe_mse": None if cand.surrogate_probe_mse is None else float(cand.surrogate_probe_mse),
        "surrogate_local_fit_mse": None if cand.surrogate_fit_mse is None else float(cand.surrogate_fit_mse),
        "value_local_probe_mse": None if cand.value_probe_mse is None else float(cand.value_probe_mse),
        "value_local_fit_mse": None if cand.value_fit_mse is None else float(cand.value_fit_mse),
        "local_calibration_gap": None if cand.calibration_gap is None else float(cand.calibration_gap),
        "witness_value_loss": None if cand.witness_value_loss is None else float(cand.witness_value_loss),
        "witness_grad_loss": None if cand.witness_grad_loss is None else float(cand.witness_grad_loss),
        "witness_d2_loss": None if cand.witness_d2_loss is None else float(cand.witness_d2_loss),
        "witness_diag_loss": None if cand.witness_diag_loss is None else float(cand.witness_diag_loss),
        "witness_physics_loss": None if cand.witness_physics_loss is None else float(cand.witness_physics_loss),
        "witness_energy_total": None if cand.witness_energy_total is None else float(cand.witness_energy_total),
        "witness_fit_jet_source": str(cand.witness_fit_jet_source or ""),
        "witness_probe_jet_source": str(cand.witness_probe_jet_source or ""),
        "witness_fit_jet_requested_source": str(cand.witness_fit_jet_requested_source or ""),
        "witness_probe_jet_requested_source": str(cand.witness_probe_jet_requested_source or ""),
        "witness_fit_jet_fallback_used": bool(cand.witness_fit_jet_fallback_used),
        "witness_probe_jet_fallback_used": bool(cand.witness_probe_jet_fallback_used),
        "witness_numeric_jet_fallback_used": bool(cand.witness_numeric_jet_fallback_used),
        "witness_exact_jet_used": bool(cand.witness_exact_jet_used),
        "local_mapping_kind": str(local_mapping_kind),
        "local_mapping_nparams": int(local_mapping_nparams),
        "candidate_subtree_size": int(cand_sub_size),
        "candidate_subtree_depth": int(cand_sub_depth),
        "candidate_subtree_size_delta": int(cand_sub_size - parent_sub_size),
        "candidate_subtree_depth_delta": int(cand_sub_depth - parent_sub_depth),
        "candidate_child_size": int(child_size),
        "candidate_child_depth": int(child_depth),
        "candidate_child_size_delta": int(child_size - parent_size),
        "candidate_child_depth_delta": int(child_depth - parent_depth),
        "candidate_root_op": str(cand_root_op),
        "exact_child_score_observed": False,
        "dedup_kept": False,
        "pre_dedup_rank": 0,
        "post_dedup_rank": 0,
        "raw_mse": None,
        "eff_mse": None,
    }


@torch.no_grad()
def solve_inverse_spec_preview_rows(
    *,
    parent_node,
    beam_state: Mapping[str, Any],
    regime_metadata: Mapping[str, Any] | None = None,
    beam_rank: int,
    slate_id: str,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    pool_nodes=None,
    pool_dims=None,
    include_legacy_seed_nodes=None,
    local_score_mode: str = "affine",
    enum_max_depth: int = 4,
    enum_max_trees: int = 5000,
    max_subtree_depth: int | None = None,
    preview_topk: int = 16,
    complexity_penalty: float = 0.0,
    family_battery_enable: bool = False,
    family_battery_mode: str = "outer",
    recursive_enable: bool = True,
    recursive_max_depth: int = 2,
    recursive_trigger_rel_mse: float = 0.25,
    recursive_seed_cap: int = 6,
    recursive_branch_topk: int = 4,
    recursive_child_topk: int = 2,
    safe_eps: float = 1.0e-12,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 1,
    witness_jets_enable: bool = False,
    witness_d2_enable: bool = False,
    witness_max_rows: int = 64,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 1.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
    active_var_screen_enable: bool = False,
    active_var_grad_tol: float = 1.0e-3,
    active_var_max_count: int = 4,
) -> dict[str, Any]:
    """Generate preview rows by solving the inverse pseudo-target directly."""

    started = time.perf_counter()
    beam_state = dict(beam_state or {})
    path = tuple(int(v) for v in (beam_state.get("path", ()) or ()))
    sub = beam_state.get("sub", None)
    target_dim = beam_state.get("target_dim", None)
    xf = beam_state.get("xf", None)
    tf = beam_state.get("tf", None)
    xp = beam_state.get("xp", None)
    tp = beam_state.get("tp", None)
    wf = beam_state.get("wf", None)
    wp = beam_state.get("wp", None)

    mode_name = _normalize_inverse_local_score_mode(local_score_mode, default="affine")
    try:
        enum_depth_limit = max(1, int(enum_max_depth))
    except Exception:
        enum_depth_limit = 4
    try:
        enum_tree_limit = max(1, int(enum_max_trees))
    except Exception:
        enum_tree_limit = 5000
    try:
        preview_limit = max(1, int(preview_topk))
    except Exception:
        preview_limit = 16
    try:
        recursive_depth_limit = max(0, int(recursive_max_depth))
    except Exception:
        recursive_depth_limit = 2
    try:
        recursive_rel_threshold = max(0.0, float(recursive_trigger_rel_mse))
    except Exception:
        recursive_rel_threshold = 0.25
    try:
        recursive_seed_limit = max(1, int(recursive_seed_cap))
    except Exception:
        recursive_seed_limit = 6
    try:
        recursive_branch_limit = max(1, int(recursive_branch_topk))
    except Exception:
        recursive_branch_limit = 4
    try:
        recursive_child_limit = max(1, int(recursive_child_topk))
    except Exception:
        recursive_child_limit = 2
    dm = var_dims is not None

    solver_meta: dict[str, Any] = {
        "proposal_family": "inverse_spec_sr",
        "generation_source": "inverse_spec_solver",
        "path": [int(v) for v in path],
        "target_mode": str(beam_state.get("target_mode", "") or ""),
        "target_mapping_kind": str(beam_state.get("target_mapping_kind", "") or ""),
        "local_score_mode": str(mode_name),
        "enum_tree_count": 0,
        "enum_depth_limit": int(enum_depth_limit),
        "enum_depth_reached": 0,
        "candidate_count_raw": 0,
        "candidate_count_scored": 0,
        "preview_count": 0,
        "witness_jets_enable": bool(witness_jets_enable),
        "witness_d2_enable": bool(witness_d2_enable),
        "witness_loss_enable": bool(witness_loss_enable),
        "witness_grad_weight": float(witness_grad_weight),
        "witness_d2_weight": float(witness_d2_weight),
        "witness_diag_weight": float(witness_diag_weight),
        "witness_physics_weight": float(witness_physics_weight),
        "top_fit_jet_source": "",
        "top_probe_jet_source": "",
        "top_fit_jet_requested_source": "",
        "top_probe_jet_requested_source": "",
        "top_fit_jet_fallback_used": False,
        "top_probe_jet_fallback_used": False,
        "active_var_screen_enable": bool(active_var_screen_enable),
        "legacy_seed_count": 0,
        "candidate_source_counts": {},
        "flat_call_count": 0,
        "flat_best_probe_mse": None,
        "flat_best_probe_rel_mse": None,
        "recursive_enable": bool(recursive_enable),
        "recursive_max_depth": int(recursive_depth_limit),
        "recursive_trigger_rel_mse": float(recursive_rel_threshold),
        "recursive_seed_cap": int(recursive_seed_limit),
        "recursive_branch_topk": int(recursive_branch_limit),
        "recursive_child_topk": int(recursive_child_limit),
        "family_battery_mode": str(normalize_family_battery_mode(family_battery_mode)),
        "recursive_used": False,
        "recursive_expand_count": 0,
        "recursive_candidate_count": 0,
        "recursive_depth_reached": 0,
        "memo_hit_count": 0,
        "memo_miss_count": 0,
        "memo_store_count": 0,
        "memo_entry_count": 0,
        "memo_hit_row_count": 0,
        "periodic_forward_used": False,
        "periodic_forward_candidate_count": 0,
        "periodic_precheck_status": "",
        "periodic_precheck_candidate_count": 0,
        "periodic_precheck_best_probe_mse": None,
        "periodic_precheck_improvement_ratio": None,
        "outer_family_used": {},
        "outer_family_candidate_counts": {},
        "outer_family_precheck_status": {},
        "outer_family_precheck_candidate_count": {},
        "outer_family_precheck_best_probe_mse": {},
        "outer_family_precheck_improvement_ratio": {},
        "stage_wall_seconds": {},
        "score_node_generation_wall_seconds": {},
        "score_node_generation_counts": {},
        "wall_seconds": 0.0,
        "status": "started",
    }

    if sub is None or xf is None or tf is None or xp is None or tp is None:
        solver_meta["status"] = "missing_beam_state"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
        solver_meta["status"] = "insufficient_points"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    if dm and target_dim is None:
        solver_meta["status"] = "missing_target_dim"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    if pool_nodes is None:
        pool_nodes = build_pool(int(nvars))
    pool_nodes = list(pool_nodes or [])
    if pool_dims is None:
        if dm:
            pool_dims = [node_dims(node, var_dims) for node in pool_nodes]
        else:
            pool_dims = [None] * len(pool_nodes)
    else:
        pool_dims = list(pool_dims or [])

    legacy_seed_nodes = _coerce_legacy_seed_nodes(include_legacy_seed_nodes)
    seed_nodes = [sub] + list(legacy_seed_nodes)
    solver_meta["legacy_seed_count"] = int(len(legacy_seed_nodes))
    stats = {
        "candidate_source_counts": {},
        "enum_tree_count": 0,
        "enum_depth_reached": 0,
        "candidate_count_raw": 0,
        "candidate_count_scored": 0,
        "flat_call_count": 0,
        "flat_best_probe_mse": None,
        "flat_best_probe_rel_mse": None,
        "recursive_used": False,
        "recursive_expand_count": 0,
        "recursive_candidate_count": 0,
        "recursive_depth_reached": 0,
        "periodic_forward_used": False,
        "periodic_forward_candidate_count": 0,
        "periodic_precheck_status": "",
        "periodic_precheck_candidate_count": 0,
        "periodic_precheck_best_probe_mse": None,
        "periodic_precheck_improvement_ratio": None,
        "outer_family_candidate_count": 0,
        "outer_family_used": {},
        "outer_family_candidate_counts": {},
        "outer_family_precheck_status": {},
        "outer_family_precheck_candidate_count": {},
        "outer_family_precheck_best_probe_mse": {},
        "outer_family_precheck_improvement_ratio": {},
        "flat_collect_wall_seconds": 0.0,
        "flat_solve_wall_seconds": 0.0,
        "periodic_forward_wall_seconds": 0.0,
        "periodic_sinusoidal_wall_seconds": 0.0,
        "outer_family_wall_seconds": 0.0,
        "outer_family_dispatch_wall_seconds": 0.0,
        "outer_family_family_wall_seconds": {},
        "memo_hit_wall_seconds": 0.0,
        "periodic_sinusoidal_count": 0,
        "recursive_branch_build_wall_seconds": 0.0,
        "recursive_child_solve_wall_seconds": 0.0,
        "solve_local_problem_wall_seconds": 0.0,
        "preview_row_build_wall_seconds": 0.0,
        "score_node_total_wall_seconds": 0.0,
        "score_node_generation_wall_seconds": {},
        "score_node_generation_counts": {},
        "score_node_source_wall_seconds": {},
        "score_node_source_counts": {},
    }
    ctx = _SolverContext(
        parent_node=parent_node,
        hole_path=path,
        hole_sub=sub,
        max_depth=int(max_depth),
        nvars=int(nvars),
        poly_degree=int(poly_degree),
        var_dims=var_dims if dm else None,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        seed_nodes=seed_nodes,
        local_score_mode=str(mode_name),
        enum_max_depth=int(enum_depth_limit),
        enum_max_trees=int(enum_tree_limit),
        max_subtree_depth=int(max_subtree_depth if max_subtree_depth is not None else max_depth),
        preview_topk=int(preview_limit),
        complexity_penalty=float(complexity_penalty),
        recursive_enable=bool(recursive_enable),
        recursive_max_depth=int(recursive_depth_limit),
        recursive_trigger_rel_mse=float(recursive_rel_threshold),
        recursive_seed_cap=int(recursive_seed_limit),
        recursive_branch_topk=int(recursive_branch_limit),
        recursive_child_topk=int(recursive_child_limit),
        safe_eps=float(safe_eps),
        confidence_mode=str(confidence_mode),
        confidence_target_gain=float(confidence_target_gain),
        confidence_floor=float(confidence_floor),
        branch_beam_width=max(1, int(branch_beam_width)),
        min_valid_frac=float(beam_state.get("min_valid_frac_eff", beam_state.get("valid_frac", 0.25)) or 0.25),
        min_confidence=float(beam_state.get("min_confidence_eff", beam_state.get("confidence", 0.10)) or 0.10),
        allow_legacy_aux=True,
        legacy_aux_kwargs={},
        stats=stats,
        target_mode=str(beam_state.get("target_mode", "") or ""),
        target_mapping_kind=str(beam_state.get("target_mapping_kind", "") or ""),
        family_battery_enable=bool(family_battery_enable),
        family_battery_mode=str(normalize_family_battery_mode(family_battery_mode)),
        witness_jets_enable=bool(witness_jets_enable),
        witness_d2_enable=bool(witness_d2_enable),
        witness_max_rows=max(4, int(witness_max_rows)),
        witness_loss_enable=bool(witness_loss_enable),
        witness_grad_weight=float(witness_grad_weight),
        witness_d2_weight=float(witness_d2_weight),
        witness_diag_weight=float(witness_diag_weight),
        witness_physics_weight=float(witness_physics_weight),
        active_var_screen_enable=bool(active_var_screen_enable),
        active_var_grad_tol=float(active_var_grad_tol),
        active_var_max_count=max(1, int(active_var_max_count)),
    )
    top_grad_fit = None
    top_grad_probe = None
    top_d2_fit = None
    top_d2_probe = None
    top_diagnostics: dict[str, Any] = _merge_local_problem_context(
        beam_state,
        regime_metadata,
    )
    top_teacher_spec = normalize_local_teacher_spec(
        beam_state.get("teacher_spec", beam_state.get("local_teacher_spec", None)),
        default_source=LOCAL_TEACHER_SOURCE_NUMERIC,
    ) if (beam_state.get("teacher_spec", beam_state.get("local_teacher_spec", None)) is not None) else None
    top_teacher_runtime = beam_state.get("teacher_runtime", None)
    if bool(witness_jets_enable):
        fit_jets = evaluate_local_teacher_jets(
            _ensure_col(xf) if xf.ndim == 1 else xf,
            _ensure_col(tf),
            w=wf,
            include_d2=bool(witness_d2_enable),
            max_rows=max(4, int(witness_max_rows)),
            teacher_spec=top_teacher_spec,
            teacher_runtime=top_teacher_runtime,
        )
        probe_jets = evaluate_local_teacher_jets(
            _ensure_col(xp) if xp.ndim == 1 else xp,
            _ensure_col(tp),
            w=wp,
            include_d2=bool(witness_d2_enable),
            max_rows=max(4, int(witness_max_rows)),
            teacher_spec=top_teacher_spec,
            teacher_runtime=top_teacher_runtime,
        )
        top_grad_fit = fit_jets.get("grad", None)
        top_grad_probe = probe_jets.get("grad", None)
        top_d2_fit = fit_jets.get("d2", None)
        top_d2_probe = probe_jets.get("d2", None)
        top_diagnostics.update(
            {
                "witness_jets_enabled": True,
                "witness_hdiag_enabled": bool(witness_d2_enable),
                "witness_fit_jet_status": str(fit_jets.get("status", "") or ""),
                "witness_probe_jet_status": str(probe_jets.get("status", "") or ""),
                "fit_jet_source": str(fit_jets.get("source", "") or ""),
                "probe_jet_source": str(probe_jets.get("source", "") or ""),
                "fit_jet_requested_source": str(fit_jets.get("requested_source", "") or ""),
                "probe_jet_requested_source": str(probe_jets.get("requested_source", "") or ""),
                "fit_jet_fallback_used": bool(fit_jets.get("fallback_used", False)),
                "probe_jet_fallback_used": bool(probe_jets.get("fallback_used", False)),
            }
        )
    top_problem = _LocalProblem(
        xf=_ensure_col(xf) if xf.ndim == 1 else xf,
        tf=_ensure_col(tf),
        wf=wf,
        xp=_ensure_col(xp) if xp.ndim == 1 else xp,
        tp=_ensure_col(tp),
        wp=wp,
        target_dim=target_dim,
        confidence=float(min(1.0, max(0.0, beam_state.get("confidence", 1.0) or 1.0))),
        valid_frac=float(min(1.0, max(0.0, beam_state.get("valid_frac", 1.0) or 1.0))),
        wrappers_left=int(recursive_depth_limit),
        recursion_level=0,
        trace=tuple(),
        grad_fit=top_grad_fit,
        grad_probe=top_grad_probe,
        d2_fit=top_d2_fit,
        d2_probe=top_d2_probe,
        teacher_spec=top_teacher_spec,
        teacher_runtime=top_teacher_runtime,
        diagnostics=top_diagnostics,
    )
    if top_diagnostics:
        solver_meta["top_fit_jet_source"] = str(top_diagnostics.get("fit_jet_source", "") or "")
        solver_meta["top_probe_jet_source"] = str(top_diagnostics.get("probe_jet_source", "") or "")
        solver_meta["top_fit_jet_requested_source"] = str(top_diagnostics.get("fit_jet_requested_source", "") or "")
        solver_meta["top_probe_jet_requested_source"] = str(top_diagnostics.get("probe_jet_requested_source", "") or "")
        solver_meta["top_fit_jet_fallback_used"] = bool(top_diagnostics.get("fit_jet_fallback_used", False))
        solver_meta["top_probe_jet_fallback_used"] = bool(top_diagnostics.get("probe_jet_fallback_used", False))

    scored_nodes, _solve_meta = _solve_local_problem(
        top_problem,
        ctx=ctx,
        include_legacy_aux=True,
    )
    preview_rows: list[dict[str, Any]] = []
    scored_nodes = _dedup_scored_candidates(scored_nodes, complexity_penalty=float(complexity_penalty))
    preview_build_t0 = time.perf_counter()
    for cand in scored_nodes:
        if not _final_replacement_ok(cand.node, ctx=ctx):
            continue
        row = _candidate_to_preview_row(
            cand,
            parent_node=parent_node,
            beam_state={**beam_state, "poly_degree": int(poly_degree)},
            beam_rank=int(beam_rank),
            slate_id=str(slate_id),
            path=path,
            xf=_ensure_col(xf) if xf.ndim == 1 else xf,
            tf=_ensure_col(tf),
            xp=_ensure_col(xp) if xp.ndim == 1 else xp,
            tp=_ensure_col(tp),
            max_depth=int(max_depth),
            var_dims=var_dims if dm else None,
            local_score_mode=str(mode_name),
        )
        if row is not None:
            preview_rows.append(row)
        if len(preview_rows) >= int(preview_limit):
            break
    _stats_add_time(ctx.stats, "preview_row_build_wall_seconds", time.perf_counter() - preview_build_t0)

    solver_meta["candidate_source_counts"] = dict(sorted((ctx.stats.get("candidate_source_counts", {}) or {}).items()))
    solver_meta["enum_tree_count"] = int(ctx.stats.get("enum_tree_count", 0) or 0)
    solver_meta["enum_depth_reached"] = int(ctx.stats.get("enum_depth_reached", 0) or 0)
    solver_meta["candidate_count_raw"] = int(ctx.stats.get("candidate_count_raw", 0) or 0)
    solver_meta["candidate_count_scored"] = int(ctx.stats.get("candidate_count_scored", 0) or 0)
    solver_meta["flat_call_count"] = int(ctx.stats.get("flat_call_count", 0) or 0)
    solver_meta["flat_best_probe_mse"] = ctx.stats.get("flat_best_probe_mse", None)
    solver_meta["flat_best_probe_rel_mse"] = ctx.stats.get("flat_best_probe_rel_mse", None)
    solver_meta["recursive_used"] = bool(ctx.stats.get("recursive_used", False))
    solver_meta["recursive_expand_count"] = int(ctx.stats.get("recursive_expand_count", 0) or 0)
    solver_meta["recursive_candidate_count"] = int(ctx.stats.get("recursive_candidate_count", 0) or 0)
    solver_meta["recursive_depth_reached"] = int(ctx.stats.get("recursive_depth_reached", 0) or 0)
    solver_meta["child_spec_states"] = [
        dict(row)
        for row in list(((_solve_meta.get("recursive", {}) or {}).get("child_spec_states", []) or []))
        if isinstance(row, Mapping)
    ]
    solver_meta["child_spec_state_count"] = int(len(list(solver_meta.get("child_spec_states", []) or [])))
    solver_meta["recursive_binary_anchor_count"] = int(ctx.stats.get("recursive_binary_anchor_count", 0) or 0)
    solver_meta["recursive_binary_outer_family_anchor_count"] = int(
        ctx.stats.get("recursive_binary_outer_family_anchor_count", 0) or 0
    )
    solver_meta["memo_hit_count"] = int(ctx.stats.get("memo_hit_count", 0) or 0)
    solver_meta["memo_miss_count"] = int(ctx.stats.get("memo_miss_count", 0) or 0)
    solver_meta["memo_store_count"] = int(ctx.stats.get("memo_store_count", 0) or 0)
    solver_meta["memo_entry_count"] = int(ctx.stats.get("memo_entry_count", 0) or 0)
    solver_meta["memo_hit_row_count"] = int(ctx.stats.get("memo_hit_row_count", 0) or 0)
    solver_meta["periodic_forward_used"] = bool(ctx.stats.get("periodic_forward_used", False))
    solver_meta["periodic_forward_candidate_count"] = int(ctx.stats.get("periodic_forward_candidate_count", 0) or 0)
    solver_meta["periodic_precheck_status"] = str(ctx.stats.get("periodic_precheck_status", "") or "")
    solver_meta["periodic_precheck_candidate_count"] = int(ctx.stats.get("periodic_precheck_candidate_count", 0) or 0)
    solver_meta["periodic_precheck_best_probe_mse"] = ctx.stats.get("periodic_precheck_best_probe_mse", None)
    solver_meta["periodic_precheck_improvement_ratio"] = ctx.stats.get("periodic_precheck_improvement_ratio", None)
    solver_meta["periodic_explicit_inverse_confidence"] = float(
        ctx.stats.get("periodic_explicit_inverse_confidence", 0.0) or 0.0
    )
    solver_meta["periodic_explicit_inverse_supported_confidence"] = float(
        ctx.stats.get("periodic_explicit_inverse_supported_confidence", 0.0) or 0.0
    )
    solver_meta["periodic_explicit_inverse_branch_count"] = int(
        ctx.stats.get("periodic_explicit_inverse_branch_count", 0) or 0
    )
    solver_meta["outer_family_used"] = dict(sorted((ctx.stats.get("outer_family_used", {}) or {}).items()))
    solver_meta["outer_family_candidate_counts"] = dict(sorted((ctx.stats.get("outer_family_candidate_counts", {}) or {}).items()))
    solver_meta["outer_family_precheck_status"] = dict(sorted((ctx.stats.get("outer_family_precheck_status", {}) or {}).items()))
    solver_meta["outer_family_precheck_candidate_count"] = dict(
        sorted((ctx.stats.get("outer_family_precheck_candidate_count", {}) or {}).items())
    )
    solver_meta["outer_family_precheck_best_probe_mse"] = dict(
        sorted((ctx.stats.get("outer_family_precheck_best_probe_mse", {}) or {}).items())
    )
    solver_meta["outer_family_precheck_improvement_ratio"] = dict(
        sorted((ctx.stats.get("outer_family_precheck_improvement_ratio", {}) or {}).items())
    )
    if bool(ctx.family_battery_enable):
        solver_meta["outer_family_evidence"] = dict(sorted((ctx.stats.get("outer_family_evidence", {}) or {}).items()))
    solver_meta["stage_wall_seconds"] = {
        "flat_collect": float(ctx.stats.get("flat_collect_wall_seconds", 0.0) or 0.0),
        "flat_solve": float(ctx.stats.get("flat_solve_wall_seconds", 0.0) or 0.0),
        "periodic_forward": float(ctx.stats.get("periodic_forward_wall_seconds", 0.0) or 0.0),
        "periodic_sinusoidal": float(ctx.stats.get("periodic_sinusoidal_wall_seconds", 0.0) or 0.0),
        "outer_family": float(ctx.stats.get("outer_family_wall_seconds", 0.0) or 0.0),
        "outer_family_dispatch": float(ctx.stats.get("outer_family_dispatch_wall_seconds", 0.0) or 0.0),
        "memo_hit": float(ctx.stats.get("memo_hit_wall_seconds", 0.0) or 0.0),
        "recursive_branch_build": float(ctx.stats.get("recursive_branch_build_wall_seconds", 0.0) or 0.0),
        "recursive_child_solve": float(ctx.stats.get("recursive_child_solve_wall_seconds", 0.0) or 0.0),
        "solve_local_problem": float(ctx.stats.get("solve_local_problem_wall_seconds", 0.0) or 0.0),
        "preview_row_build": float(ctx.stats.get("preview_row_build_wall_seconds", 0.0) or 0.0),
        "score_node_total": float(ctx.stats.get("score_node_total_wall_seconds", 0.0) or 0.0),
    }
    family_wall = dict(sorted((ctx.stats.get("outer_family_family_wall_seconds", {}) or {}).items()))
    if family_wall:
        solver_meta["stage_wall_seconds"]["outer_family_by_name"] = family_wall
    solver_meta["periodic_sinusoidal_count"] = int(ctx.stats.get("periodic_sinusoidal_count", 0) or 0)
    solver_meta["score_node_generation_wall_seconds"] = dict(
        sorted((ctx.stats.get("score_node_generation_wall_seconds", {}) or {}).items())
    )
    solver_meta["score_node_generation_counts"] = dict(
        sorted((ctx.stats.get("score_node_generation_counts", {}) or {}).items())
    )
    solver_meta["preview_count"] = int(len(preview_rows))
    solver_meta["status"] = "ok" if preview_rows else "no_scored_candidates"
    solver_meta["wall_seconds"] = float(time.perf_counter() - started)

    for local_rank, row in enumerate(preview_rows):
        row["local_rank"] = int(local_rank)
        row["local_candidate_count"] = int(len(preview_rows))

    return {
        "rows": preview_rows,
        "solver_meta": solver_meta,
    }

@torch.no_grad()
def solve_local_problem_spec_preview_rows(
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
    enum_max_depth: int = 4,
    enum_max_trees: int = 5000,
    max_subtree_depth: int | None = None,
    preview_topk: int = 16,
    complexity_penalty: float = 0.0,
    family_battery_enable: bool = False,
    family_battery_mode: str = "outer",
    recursive_enable: bool = True,
    recursive_max_depth: int = 2,
    recursive_trigger_rel_mse: float = 0.25,
    recursive_seed_cap: int = 6,
    recursive_branch_topk: int = 4,
    recursive_child_topk: int = 2,
    safe_eps: float = 1.0e-12,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 1,
    witness_jets_enable: bool = False,
    witness_d2_enable: bool = False,
    witness_max_rows: int = 64,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 1.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
    active_var_screen_enable: bool = False,
    active_var_grad_tol: float = 1.0e-3,
    active_var_max_count: int = 4,
) -> dict[str, Any]:
    started = time.perf_counter()
    mode_name = _normalize_inverse_local_score_mode(local_score_mode, default="affine")
    hole_path = tuple(int(v) for v in (path or ()))
    payload = dict(spec_payload or {})
    subproblem_spec = deserialize_subproblem_spec(payload)
    if subproblem_spec is not None:
        problem, continuation_frames, hole_sub = _subproblem_spec_to_local_problem(subproblem_spec)
    else:
        problem = None
        continuation_frames = []
        hole_sub = None
    if problem is None:
        problem = _deserialize_local_problem(payload.get("problem", {}))
        continuation_frames = list(payload.get("continuation_frames", []) or [])
        hole_sub = payload.get("hole_sub", None)
    solver_meta: dict[str, Any] = {
        "proposal_family": "inverse_spec_local_problem",
        "generation_source": "inverse_spec_solver",
        "path": [int(v) for v in hole_path],
        "target_mode": str(target_mode or ""),
        "target_mapping_kind": str(target_mapping_kind or ""),
        "local_score_mode": str(mode_name),
        "preview_count": 0,
        "witness_jets_enable": bool(witness_jets_enable),
        "witness_d2_enable": bool(witness_d2_enable),
        "witness_loss_enable": bool(witness_loss_enable),
        "witness_grad_weight": float(witness_grad_weight),
        "witness_d2_weight": float(witness_d2_weight),
        "witness_diag_weight": float(witness_diag_weight),
        "witness_physics_weight": float(witness_physics_weight),
        "active_var_screen_enable": bool(active_var_screen_enable),
        "candidate_count_scored": 0,
        "family_battery_mode": str(normalize_family_battery_mode(family_battery_mode)),
        "recursive_used": False,
        "recursive_expand_count": 0,
        "recursive_candidate_count": 0,
        "recursive_depth_reached": 0,
        "child_spec_states": [],
        "child_spec_state_count": 0,
        "wall_seconds": 0.0,
        "status": "started",
    }
    if problem is None:
        solver_meta["status"] = "missing_spec_payload"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    if pool_nodes is None:
        pool_nodes = build_pool(int(nvars))
    pool_nodes = list(pool_nodes or [])
    dm = var_dims is not None
    if pool_dims is None:
        if dm:
            pool_dims = [node_dims(node, var_dims) for node in pool_nodes]
        else:
            pool_dims = [None] * len(pool_nodes)
    else:
        pool_dims = list(pool_dims or [])
    ctx = _SolverContext(
        parent_node=parent_node,
        hole_path=hole_path,
        hole_sub=hole_sub,
        max_depth=int(max_depth),
        nvars=int(nvars),
        poly_degree=int(poly_degree),
        var_dims=var_dims if dm else None,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        seed_nodes=[],
        local_score_mode=str(mode_name),
        enum_max_depth=max(1, int(enum_max_depth)),
        enum_max_trees=max(1, int(enum_max_trees)),
        max_subtree_depth=int(max_subtree_depth if max_subtree_depth is not None else max_depth),
        preview_topk=max(1, int(preview_topk)),
        complexity_penalty=float(complexity_penalty),
        recursive_enable=bool(recursive_enable),
        recursive_max_depth=max(0, int(recursive_max_depth)),
        recursive_trigger_rel_mse=max(0.0, float(recursive_trigger_rel_mse)),
        recursive_seed_cap=max(1, int(recursive_seed_cap)),
        recursive_branch_topk=max(1, int(recursive_branch_topk)),
        recursive_child_topk=max(1, int(recursive_child_topk)),
        safe_eps=float(safe_eps),
        confidence_mode=str(confidence_mode),
        confidence_target_gain=float(confidence_target_gain),
        confidence_floor=float(confidence_floor),
        branch_beam_width=max(1, int(branch_beam_width)),
        min_valid_frac=float(problem.valid_frac),
        min_confidence=float(problem.confidence),
        allow_legacy_aux=False,
        legacy_aux_kwargs={},
        stats={},
        target_mode=str(target_mode or ""),
        target_mapping_kind=str(target_mapping_kind or ""),
        family_battery_enable=bool(family_battery_enable),
        family_battery_mode=str(normalize_family_battery_mode(family_battery_mode)),
        witness_jets_enable=bool(witness_jets_enable),
        witness_d2_enable=bool(witness_d2_enable),
        witness_max_rows=max(4, int(witness_max_rows)),
        witness_loss_enable=bool(witness_loss_enable),
        witness_grad_weight=float(witness_grad_weight),
        witness_d2_weight=float(witness_d2_weight),
        witness_diag_weight=float(witness_diag_weight),
        witness_physics_weight=float(witness_physics_weight),
        active_var_screen_enable=bool(active_var_screen_enable),
        active_var_grad_tol=float(active_var_grad_tol),
        active_var_max_count=max(1, int(active_var_max_count)),
    )
    scored_nodes, solve_meta = _solve_local_problem(
        problem,
        ctx=ctx,
        include_legacy_aux=False,
    )
    scored_nodes = _dedup_scored_candidates(scored_nodes, complexity_penalty=float(complexity_penalty))
    preview_rows: list[dict[str, Any]] = []
    for cand in scored_nodes:
        try:
            wrapped_node = _simplify_node(_apply_continuation_frames(cand.node, continuation_frames))
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
            var_dims=var_dims if dm else None,
            local_score_mode=str(mode_name),
        )
        if row is not None:
            preview_rows.append(row)
        if len(preview_rows) >= max(1, int(preview_topk)):
            break
    child_spec_states: list[dict[str, Any]] = []
    for row in list((solve_meta.get("recursive", {}) or {}).get("child_spec_states", []) or []):
        if not isinstance(row, Mapping):
            continue
        child_row = dict(row)
        child_payload = dict(child_row.get("spec_payload", {}) or {})
        child_frames = list(child_payload.get("continuation_frames", []) or [])
        composed_frames = list(child_frames) + list(continuation_frames)
        child_payload["continuation_frames"] = composed_frames
        child_payload["hole_sub"] = hole_sub
        child_spec = deserialize_subproblem_spec(child_payload)
        if child_spec is not None:
            child_problem, _unused_frames, _unused_hole_sub = _subproblem_spec_to_local_problem(child_spec)
            if child_problem is not None:
                updated_spec = _local_problem_to_subproblem_spec(
                    child_problem,
                    parent_expr=parent_node,
                    path=hole_path,
                    target_mode=str(target_mode or ""),
                    target_mapping_kind=str(target_mapping_kind or ""),
                    continuation_frames=composed_frames,
                    hole_sub=hole_sub,
                    direction=str(child_spec.direction or "inside_out"),
                    metadata=dict(child_spec.metadata or {}),
                    witness_jets_enable=bool(witness_jets_enable),
                    witness_d2_enable=bool(witness_d2_enable),
                    witness_max_rows=max(4, int(witness_max_rows)),
                    active_var_screen_enable=bool(active_var_screen_enable),
                    active_var_grad_tol=float(active_var_grad_tol),
                    active_var_max_count=max(1, int(active_var_max_count)),
                )
                child_payload = wrap_subproblem_spec_payload(
                    updated_spec,
                    extra_payload=child_payload,
                )
        child_row["spec_payload"] = child_payload
        child_row["path"] = [int(v) for v in hole_path]
        final_child_spec = deserialize_subproblem_spec(child_payload)
        child_row["direction"] = str(getattr(final_child_spec, "direction", getattr(child_spec, "direction", "")) or "")
        child_row["continuation_key"] = _continuation_frames_tokens(composed_frames)
        child_row["branch_id"] = str(":".join(str(v) for v in child_row.get("continuation_key", []) if str(v)) or child_row.get("branch_id", ""))
        child_spec_states.append(child_row)
    solver_meta["preview_count"] = int(len(preview_rows))
    solver_meta["candidate_count_scored"] = int(ctx.stats.get("candidate_count_scored", 0) or 0)
    solver_meta["recursive_used"] = bool(ctx.stats.get("recursive_used", False))
    solver_meta["recursive_expand_count"] = int(ctx.stats.get("recursive_expand_count", 0) or 0)
    solver_meta["recursive_candidate_count"] = int(ctx.stats.get("recursive_candidate_count", 0) or 0)
    solver_meta["recursive_depth_reached"] = int(ctx.stats.get("recursive_depth_reached", 0) or 0)
    solver_meta["child_spec_states"] = child_spec_states
    solver_meta["child_spec_state_count"] = int(len(child_spec_states))
    if bool(ctx.family_battery_enable):
        solver_meta["outer_family_evidence"] = dict(sorted((ctx.stats.get("outer_family_evidence", {}) or {}).items()))
    solver_meta["wall_seconds"] = float(time.perf_counter() - started)
    solver_meta["status"] = "ok" if preview_rows else "no_scored_candidates"
    return {
        "rows": preview_rows,
        "solver_meta": solver_meta,
    }


__all__ = ["solve_inverse_spec_preview_rows", "solve_local_problem_spec_preview_rows"]
