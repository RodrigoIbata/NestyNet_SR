# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Any, Mapping

import torch

from .subproblem_witness import estimate_pointwise_target_jets


LOCAL_TEACHER_SOURCE_NUMERIC = "numeric_local_quadratic"
LOCAL_TEACHER_SOURCE_ORACLE = "oracle"
LOCAL_TEACHER_SOURCE_SYMBOLIC = "symbolic"
LOCAL_TEACHER_SOURCE_RUNTIME = "runtime_teacher"
LOCAL_TEACHER_SOURCE_UNKNOWN = "unknown"


def normalize_local_teacher_source(
    source: Any,
    *,
    default: str = LOCAL_TEACHER_SOURCE_NUMERIC,
) -> str:
    token = str(source or "").strip().lower()
    if not token:
        return str(default or LOCAL_TEACHER_SOURCE_NUMERIC)
    if token in {"numeric", "numeric_local_quadratic", "numeric-local-quadratic"}:
        return LOCAL_TEACHER_SOURCE_NUMERIC
    if token in {"oracle", "exact_oracle", "exact-oracle"}:
        return LOCAL_TEACHER_SOURCE_ORACLE
    if token in {"symbolic", "exact_symbolic", "exact-symbolic"}:
        return LOCAL_TEACHER_SOURCE_SYMBOLIC
    if token in {"runtime_teacher", "runtime-teacher"}:
        return LOCAL_TEACHER_SOURCE_RUNTIME
    return LOCAL_TEACHER_SOURCE_UNKNOWN


def normalize_local_teacher_spec(
    spec: Mapping[str, Any] | None,
    *,
    default_source: str = LOCAL_TEACHER_SOURCE_NUMERIC,
) -> dict[str, Any]:
    data = dict(spec or {})
    source = normalize_local_teacher_source(data.get("source", None), default=default_source)
    out: dict[str, Any] = {
        "source": str(source),
    }
    requested = data.get("requested_source", None)
    if requested is not None:
        out["requested_source"] = normalize_local_teacher_source(requested, default=source)
    for key, value in data.items():
        if str(key) in {"source", "requested_source"}:
            continue
        out[str(key)] = value
    return out


def build_numeric_local_teacher_spec(
    *,
    requested_source: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    out = {
        "source": LOCAL_TEACHER_SOURCE_NUMERIC,
        "engine": LOCAL_TEACHER_SOURCE_NUMERIC,
    }
    if requested_source:
        out["requested_source"] = normalize_local_teacher_source(
            requested_source,
            default=LOCAL_TEACHER_SOURCE_NUMERIC,
        )
    if reason:
        out["reason"] = str(reason)
    return out


def _ensure_matrix(value: Any, *, like: torch.Tensor | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value
        if like is not None:
            out = out.to(dtype=like.dtype, device=like.device)
        else:
            out = out.to(dtype=torch.float64)
    else:
        try:
            kwargs = {}
            if like is not None:
                kwargs = {"dtype": like.dtype, "device": like.device}
            else:
                kwargs = {"dtype": torch.float64}
            out = torch.as_tensor(value, **kwargs)
        except Exception:
            return None
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    if out.ndim != 2:
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _normalize_grad_tensor(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    out = _ensure_matrix(value, like=x)
    if out is not None and tuple(out.shape) == tuple(x.shape):
        return out
    if torch.is_tensor(value):
        raw = value.to(dtype=x.dtype, device=x.device)
    else:
        try:
            raw = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    if raw.ndim >= 3 and int(raw.shape[0]) == int(x.shape[0]) and int(raw.shape[-1]) == int(x.shape[1]):
        try:
            out = raw.reshape(int(x.shape[0]), -1, int(x.shape[1])).mean(dim=1)
        except Exception:
            return None
        if torch.isfinite(out).all():
            return out
    return None


def _normalize_hdiag_tensor(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        raw = value.to(dtype=x.dtype, device=x.device)
    else:
        try:
            raw = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    if raw.ndim == 2 and tuple(raw.shape) == tuple(x.shape):
        out = raw
    elif raw.ndim == 3 and tuple(raw.shape) == (int(x.shape[0]), int(x.shape[1]), int(x.shape[1])):
        out = torch.diagonal(raw, dim1=-2, dim2=-1)
    elif raw.ndim >= 4 and int(raw.shape[0]) == int(x.shape[0]) and tuple(raw.shape[-2:]) == (
        int(x.shape[1]),
        int(x.shape[1]),
    ):
        try:
            out = torch.diagonal(
                raw.reshape(int(x.shape[0]), -1, int(x.shape[1]), int(x.shape[1])).mean(dim=1),
                dim1=-2,
                dim2=-1,
            )
        except Exception:
            return None
    else:
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _runtime_teacher_jets(
    *,
    x: torch.Tensor,
    include_d2: bool,
    teacher_runtime: Any,
) -> dict[str, Any] | None:
    if teacher_runtime is None or not hasattr(teacher_runtime, "grad"):
        return None
    try:
        grad = _normalize_grad_tensor(teacher_runtime.grad(x), x=x)
    except Exception:
        grad = None
    if grad is None:
        return None
    d2 = None
    status = "ok"
    if bool(include_d2) and hasattr(teacher_runtime, "grad_grad"):
        try:
            d2 = _normalize_hdiag_tensor(teacher_runtime.grad_grad(x), x=x)
        except Exception:
            d2 = None
        if d2 is None:
            status = "ok_missing_d2"
    elif bool(include_d2):
        status = "ok_missing_d2"
    return {
        "status": str(status),
        "source": LOCAL_TEACHER_SOURCE_RUNTIME,
        "grad": grad,
        "d2": d2,
        "row_count": int(x.shape[0]),
        "support_count": int(x.shape[0]),
        "neighbor_count": int(x.shape[0]),
        "include_d2": bool(include_d2),
        "failed_rows": 0,
        "fallback_used": False,
    }


def evaluate_local_teacher_jets(
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    w: torch.Tensor | None = None,
    include_d2: bool = False,
    max_rows: int = 64,
    teacher_spec: Mapping[str, Any] | None = None,
    teacher_runtime: Any = None,
) -> dict[str, Any]:
    spec = normalize_local_teacher_spec(teacher_spec, default_source=LOCAL_TEACHER_SOURCE_NUMERIC)
    requested_source = normalize_local_teacher_source(
        spec.get("requested_source", spec.get("source", None)),
        default=LOCAL_TEACHER_SOURCE_NUMERIC,
    )

    exact = _runtime_teacher_jets(
        x=x,
        include_d2=bool(include_d2),
        teacher_runtime=teacher_runtime,
    )
    if exact is not None:
        effective_source = normalize_local_teacher_source(
            spec.get("source", LOCAL_TEACHER_SOURCE_RUNTIME),
            default=LOCAL_TEACHER_SOURCE_RUNTIME,
        )
        if effective_source == LOCAL_TEACHER_SOURCE_NUMERIC:
            effective_source = LOCAL_TEACHER_SOURCE_RUNTIME
        exact["source"] = str(effective_source)
        exact["requested_source"] = str(requested_source)
        exact["teacher_spec"] = dict(spec)
        return exact

    numeric = estimate_pointwise_target_jets(
        x,
        target,
        w=w,
        include_d2=bool(include_d2),
        max_rows=max_rows,
    )
    numeric["requested_source"] = str(requested_source)
    numeric["teacher_spec"] = dict(spec)
    numeric["fallback_used"] = bool(
        teacher_runtime is not None
        or normalize_local_teacher_source(spec.get("source", None), default=LOCAL_TEACHER_SOURCE_NUMERIC)
        != LOCAL_TEACHER_SOURCE_NUMERIC
    )
    return numeric


__all__ = [
    "LOCAL_TEACHER_SOURCE_NUMERIC",
    "LOCAL_TEACHER_SOURCE_ORACLE",
    "LOCAL_TEACHER_SOURCE_RUNTIME",
    "LOCAL_TEACHER_SOURCE_SYMBOLIC",
    "LOCAL_TEACHER_SOURCE_UNKNOWN",
    "build_numeric_local_teacher_spec",
    "evaluate_local_teacher_jets",
    "normalize_local_teacher_source",
    "normalize_local_teacher_spec",
]
