# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .expr_ast import BINARY_OPS, UNARY_OPS


def _normalize_tensor_grad(value: Any, *, nvars: int) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value)
        except Exception:
            return None
    if value.ndim == 2 and int(value.shape[1]) == int(nvars):
        out = value
    elif value.ndim >= 3 and int(value.shape[-1]) == int(nvars):
        try:
            out = value.reshape(int(value.shape[0]), -1, int(nvars)).mean(dim=1)
        except Exception:
            return None
    else:
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def normalize_active_vars(active_vars: Sequence[int] | None, *, nvars: int) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for value in list(active_vars or ()):
        try:
            idx = int(value)
        except Exception:
            continue
        if idx < 0 or idx >= int(max(0, nvars)) or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return tuple(out)


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


def _collect_hole_var_ids(hole_sub) -> tuple[int, ...]:
    out: set[int] = set()
    _collect_node_var_ids(hole_sub, out)
    return tuple(sorted(out))


def _collect_anchor_var_ids(continuation_frames: Sequence[Mapping[str, Any]] | None) -> tuple[int, ...]:
    out: set[int] = set()
    for frame in list(continuation_frames or ()):
        if not isinstance(frame, Mapping):
            continue
        _collect_node_var_ids(frame.get("anchor_node", None), out)
    return tuple(sorted(out))


def collect_structural_active_vars(
    *,
    hole_sub,
    continuation_frames: Sequence[Mapping[str, Any]] | None,
    nvars: int,
) -> tuple[int, ...]:
    hole_vars = normalize_active_vars(_collect_hole_var_ids(hole_sub), nvars=int(nvars))
    anchor_vars = normalize_active_vars(_collect_anchor_var_ids(continuation_frames), nvars=int(nvars))
    return normalize_active_vars(tuple(hole_vars) + tuple(anchor_vars), nvars=int(nvars))


def gradient_activity_scores(
    *,
    grad_fit: Any,
    grad_probe: Any,
    nvars: int,
) -> torch.Tensor | None:
    nvars = int(max(0, nvars))
    if nvars <= 0:
        return None
    parts: list[torch.Tensor] = []
    grad_fit_t = _normalize_tensor_grad(grad_fit, nvars=nvars)
    grad_probe_t = _normalize_tensor_grad(grad_probe, nvars=nvars)
    if grad_fit_t is not None and int(grad_fit_t.shape[0]) > 0:
        parts.append(torch.mean(torch.abs(grad_fit_t), dim=0))
    if grad_probe_t is not None and int(grad_probe_t.shape[0]) > 0:
        parts.append(torch.mean(torch.abs(grad_probe_t), dim=0))
    if not parts:
        return None
    scores = torch.stack(parts, dim=0).mean(dim=0)
    if scores.ndim != 1 or int(scores.shape[0]) != nvars:
        return None
    if not torch.isfinite(scores).all():
        return None
    return scores


def select_gradient_active_vars(
    *,
    scores: torch.Tensor | None,
    grad_tol: float,
    max_count: int,
    nvars: int,
) -> tuple[int, ...]:
    if scores is None:
        return ()
    nvars = int(max(0, nvars))
    if nvars <= 0:
        return ()
    try:
        tol = max(0.0, float(grad_tol))
    except Exception:
        tol = 1.0e-3
    try:
        limit = max(1, int(max_count))
    except Exception:
        limit = 4
    max_score = float(torch.max(scores).item()) if int(scores.numel()) > 0 else 0.0
    if max_score <= 0.0:
        return ()
    threshold = tol * max_score
    selected = [idx for idx in range(nvars) if float(scores[idx].item()) >= float(threshold)]
    if not selected:
        top_idx = int(torch.argmax(scores).item())
        selected = [top_idx]
    if len(selected) > limit:
        ranked = sorted(
            selected,
            key=lambda idx: (-float(scores[idx].item()), int(idx)),
        )[:limit]
        selected = sorted(int(idx) for idx in ranked)
    return normalize_active_vars(selected, nvars=nvars)


def infer_subproblem_active_vars(
    *,
    hole_sub,
    continuation_frames: Sequence[Mapping[str, Any]] | None,
    grad_fit: Any,
    grad_probe: Any,
    nvars: int,
    screen_enable: bool,
    grad_tol: float,
    max_count: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    nvars = int(max(0, nvars))
    hole_vars = normalize_active_vars(_collect_hole_var_ids(hole_sub), nvars=nvars)
    anchor_vars = normalize_active_vars(_collect_anchor_var_ids(continuation_frames), nvars=nvars)
    structural = normalize_active_vars(tuple(hole_vars) + tuple(anchor_vars), nvars=nvars)
    scores = gradient_activity_scores(
        grad_fit=grad_fit,
        grad_probe=grad_probe,
        nvars=nvars,
    )
    grad_selected = ()
    if bool(screen_enable):
        grad_selected = select_gradient_active_vars(
            scores=scores,
            grad_tol=float(grad_tol),
            max_count=int(max_count),
            nvars=nvars,
        )
    active_vars = structural
    source = "structural" if structural else "none"
    if grad_selected:
        grad_selected_set = set(grad_selected)
        structural_grad = tuple(v for v in structural if v in grad_selected_set)
        hole_grad = tuple(v for v in hole_vars if v in grad_selected_set)
        if hole_grad:
            active_vars = hole_grad
            source = "hole+gradient"
        elif structural_grad:
            active_vars = structural_grad
            source = "structural+gradient"
        elif not hole_vars:
            active_vars = grad_selected
            source = "gradient"
        elif hole_vars:
            active_vars = hole_vars
            source = "hole"
    diagnostics: dict[str, Any] = {
        "active_var_source": str(source),
        "active_var_hole": [int(v) for v in hole_vars],
        "active_var_anchor": [int(v) for v in anchor_vars],
        "active_var_structural": [int(v) for v in structural],
        "active_var_screen_enabled": bool(screen_enable),
        "active_vars": [int(v) for v in active_vars],
    }
    if scores is not None:
        diagnostics["active_var_gradient_scores"] = [
            float(scores[idx].item()) for idx in range(int(scores.shape[0]))
        ]
    if grad_selected:
        diagnostics["active_var_gradient_selected"] = [int(v) for v in grad_selected]
    return active_vars, diagnostics


def subset_var_tensor(x: torch.Tensor, active_vars: Sequence[int] | None) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    idx = normalize_active_vars(active_vars, nvars=int(x.shape[1]) if x.ndim >= 2 else 0)
    if not idx or x.ndim < 2:
        return x
    return x.index_select(1, torch.as_tensor(idx, dtype=torch.long, device=x.device))


def subset_var_dims(
    var_dims: Sequence[Any] | None,
    *,
    active_vars: Sequence[int] | None,
) -> list[Any] | None:
    if var_dims is None:
        return None
    raw = list(var_dims or [])
    idx = normalize_active_vars(active_vars, nvars=len(raw))
    if not idx:
        return raw
    return [raw[i] for i in idx]


def remap_local_node_vars(node, *, active_vars: Sequence[int] | None):
    idx = tuple(int(v) for v in tuple(active_vars or ()))
    if not idx:
        return node
    if not isinstance(node, tuple) or not node:
        return node
    op = node[0]
    if op == "var":
        local_idx = int(node[1])
        if local_idx < 0 or local_idx >= len(idx):
            raise ValueError(f"local var index {local_idx} outside active var map of size {len(idx)}")
        return ("var", int(idx[local_idx]))
    if op in ("const", "hparam"):
        return node
    if op in UNARY_OPS and len(node) >= 2:
        return (op, remap_local_node_vars(node[1], active_vars=idx))
    if op in BINARY_OPS and len(node) >= 3:
        return (
            op,
            remap_local_node_vars(node[1], active_vars=idx),
            remap_local_node_vars(node[2], active_vars=idx),
        )
    return node


__all__ = [
    "collect_structural_active_vars",
    "gradient_activity_scores",
    "infer_subproblem_active_vars",
    "normalize_active_vars",
    "remap_local_node_vars",
    "select_gradient_active_vars",
    "subset_var_dims",
    "subset_var_tensor",
]
