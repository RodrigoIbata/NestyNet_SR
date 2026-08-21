# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Any

import torch


JET_SOURCE = "numeric_local_quadratic"


def _ensure_matrix(x: Any) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    if x.ndim == 1:
        return x.unsqueeze(-1)
    if x.ndim == 2:
        return x
    raise ValueError(f"expected [N] or [N,D] tensor, got shape={tuple(x.shape)}")


def _ensure_col(y: Any) -> torch.Tensor:
    if not torch.is_tensor(y):
        raise TypeError(f"expected torch.Tensor, got {type(y).__name__}")
    if y.ndim == 1:
        return y.unsqueeze(-1)
    if y.ndim == 2 and int(y.shape[1]) == 1:
        return y
    raise ValueError(f"expected [N] or [N,1] tensor, got shape={tuple(y.shape)}")


def _coerce_weight_col(w: Any, *, ref: torch.Tensor) -> torch.Tensor:
    rr = _ensure_col(ref)
    if w is None:
        return torch.ones_like(rr)
    if not torch.is_tensor(w):
        ww = torch.as_tensor(w, dtype=rr.dtype, device=rr.device)
    else:
        ww = w.to(dtype=rr.dtype, device=rr.device)
    if ww.ndim == 1:
        ww = ww.unsqueeze(-1)
    if ww.ndim != 2 or int(ww.shape[1]) != 1 or int(ww.shape[0]) != int(rr.shape[0]):
        return torch.ones_like(rr)
    ww = torch.where(torch.isfinite(ww), ww, torch.zeros_like(ww))
    return torch.clamp(ww, min=0.0)


def _deterministic_support_indices(n_rows: int, *, max_rows: int, device: torch.device) -> torch.Tensor:
    count = int(max(1, min(int(n_rows), int(max_rows))))
    if count >= int(n_rows):
        return torch.arange(int(n_rows), device=device, dtype=torch.long)
    idx = torch.linspace(
        0,
        int(n_rows) - 1,
        steps=count,
        dtype=torch.float64,
        device=device,
    ).round().to(dtype=torch.long)
    idx = torch.unique(idx, sorted=True)
    if int(idx.numel()) >= count:
        return idx[:count]
    all_idx = torch.arange(int(n_rows), device=device, dtype=torch.long)
    keep = torch.ones(int(n_rows), device=device, dtype=torch.bool)
    keep[idx] = False
    extra = all_idx[keep][: max(0, count - int(idx.numel()))]
    return torch.cat([idx, extra], dim=0)


def _solve_weighted_diag_quadratic(
    delta_scaled: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    include_d2: bool,
) -> torch.Tensor | None:
    if int(delta_scaled.shape[0]) <= 0:
        return None
    parts = [torch.ones((int(delta_scaled.shape[0]), 1), dtype=delta_scaled.dtype, device=delta_scaled.device), delta_scaled]
    if bool(include_d2):
        parts.append(0.5 * (delta_scaled * delta_scaled))
    design = torch.cat(parts, dim=1)
    target_col = _ensure_col(target)
    weight_col = _coerce_weight_col(weights, ref=target_col)
    weight_col = torch.clamp(weight_col, min=1.0e-8)
    sqrt_w = torch.sqrt(weight_col)
    aw = design * sqrt_w
    bw = target_col * sqrt_w
    gram = aw.T @ aw
    if int(gram.shape[0]) != int(gram.shape[1]):
        return None
    rhs = aw.T @ bw
    diag_scale = float(torch.mean(torch.diag(gram)).item()) if int(gram.shape[0]) > 0 else 1.0
    ridge = max(1.0e-8, 1.0e-6 * abs(diag_scale))
    eye = torch.eye(int(gram.shape[0]), dtype=gram.dtype, device=gram.device)
    try:
        sol = torch.linalg.solve(gram + (ridge * eye), rhs)
    except Exception:
        try:
            sol = torch.linalg.lstsq(aw, bw).solution
        except Exception:
            return None
    if not torch.isfinite(sol).all():
        return None
    return sol


def estimate_pointwise_target_jets(
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    w: torch.Tensor | None = None,
    include_d2: bool = False,
    max_rows: int = 64,
) -> dict[str, Any]:
    xx = _ensure_matrix(x)
    yy = _ensure_col(target).to(dtype=xx.dtype, device=xx.device)
    ww = _coerce_weight_col(w, ref=yy).to(dtype=xx.dtype, device=xx.device)
    n_rows = int(xx.shape[0])
    n_dims = int(xx.shape[1])
    if n_rows < max(4, n_dims + 2):
        return {
            "status": "insufficient_points",
            "source": JET_SOURCE,
            "grad": None,
            "d2": None,
            "row_count": int(n_rows),
            "support_count": 0,
            "neighbor_count": 0,
            "include_d2": bool(include_d2),
        }
    if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
        return {
            "status": "nonfinite_input",
            "source": JET_SOURCE,
            "grad": None,
            "d2": None,
            "row_count": int(n_rows),
            "support_count": 0,
            "neighbor_count": 0,
            "include_d2": bool(include_d2),
        }

    support_idx = _deterministic_support_indices(n_rows, max_rows=max(4, int(max_rows)), device=xx.device)
    xs = xx[support_idx]
    ys = yy[support_idx]
    ws = ww[support_idx]
    scale = torch.std(xs, dim=0, unbiased=False)
    scale = torch.where(torch.isfinite(scale) & (scale > 1.0e-8), scale, torch.ones_like(scale))
    xs_scaled = xs / scale
    xx_scaled = xx / scale

    mean_scaled = xs_scaled.mean(dim=0, keepdim=True)
    global_delta = xs_scaled - mean_scaled
    global_sol = _solve_weighted_diag_quadratic(global_delta, ys, ws, include_d2=bool(include_d2))
    global_grad = None
    global_d2 = None
    if global_sol is not None:
        global_grad = (global_sol[1 : 1 + n_dims, 0] / scale).reshape(1, n_dims)
        if bool(include_d2):
            global_d2 = (global_sol[1 + n_dims : 1 + (2 * n_dims), 0] / (scale * scale)).reshape(1, n_dims)

    feature_count = 1 + n_dims + (n_dims if bool(include_d2) else 0)
    target_neighbors = max(feature_count + 2, min(int(xs.shape[0]) + 1, 3 * feature_count))
    grad_rows: list[torch.Tensor] = []
    d2_rows: list[torch.Tensor] = []
    failed_rows = 0

    for row_idx in range(n_rows):
        center = xx_scaled[row_idx : row_idx + 1]
        delta_support = xs_scaled - center
        target_support = ys
        weight_support = ws
        dist_support = torch.sum(delta_support * delta_support, dim=1, keepdim=True)
        if not bool((support_idx == int(row_idx)).any().item()):
            delta_support = torch.cat([delta_support, torch.zeros((1, n_dims), dtype=xx.dtype, device=xx.device)], dim=0)
            target_support = torch.cat([target_support, yy[row_idx : row_idx + 1]], dim=0)
            weight_support = torch.cat([weight_support, ww[row_idx : row_idx + 1]], dim=0)
            dist_support = torch.cat([dist_support, torch.zeros((1, 1), dtype=xx.dtype, device=xx.device)], dim=0)

        neighbor_count = min(int(delta_support.shape[0]), int(target_neighbors))
        try:
            nn = torch.topk(dist_support.squeeze(-1), k=neighbor_count, largest=False).indices
        except Exception:
            nn = torch.arange(int(delta_support.shape[0]), device=xx.device, dtype=torch.long)[:neighbor_count]
        delta_sel = delta_support[nn]
        target_sel = target_support[nn]
        weight_sel = weight_support[nn]
        dist_sel = dist_support[nn]
        band_sq = float(dist_sel.max().item()) if int(dist_sel.numel()) > 0 else 0.0
        band_sq = max(1.0e-6, band_sq)
        kernel = torch.exp(-dist_sel / band_sq)
        local_weights = weight_sel * kernel
        sol = _solve_weighted_diag_quadratic(delta_sel, target_sel, local_weights, include_d2=bool(include_d2))
        if sol is None:
            failed_rows += 1
            if global_grad is None:
                grad_rows.append(torch.zeros((1, n_dims), dtype=xx.dtype, device=xx.device))
                if bool(include_d2):
                    d2_rows.append(torch.zeros((1, n_dims), dtype=xx.dtype, device=xx.device))
                continue
            grad_rows.append(global_grad)
            if bool(include_d2):
                d2_rows.append(global_d2 if global_d2 is not None else torch.zeros((1, n_dims), dtype=xx.dtype, device=xx.device))
            continue
        grad_rows.append((sol[1 : 1 + n_dims, 0] / scale).reshape(1, n_dims))
        if bool(include_d2):
            d2_rows.append((sol[1 + n_dims : 1 + (2 * n_dims), 0] / (scale * scale)).reshape(1, n_dims))

    grad = torch.cat(grad_rows, dim=0) if grad_rows else None
    d2 = torch.cat(d2_rows, dim=0) if d2_rows else None
    status = "ok"
    if failed_rows > 0 and global_grad is not None:
        status = "ok_with_global_fallback"
    elif failed_rows > 0:
        status = "partial_zero_fallback"
    if grad is not None and not torch.isfinite(grad).all():
        grad = None
        d2 = None
        status = "nonfinite_output"
    if d2 is not None and not torch.isfinite(d2).all():
        d2 = None
        status = "nonfinite_d2"
    return {
        "status": str(status),
        "source": JET_SOURCE,
        "grad": grad,
        "d2": d2,
        "row_count": int(n_rows),
        "support_count": int(xs.shape[0]),
        "neighbor_count": int(min(int(xs.shape[0]) + 1, int(target_neighbors))),
        "include_d2": bool(include_d2),
        "failed_rows": int(failed_rows),
    }


__all__ = ["estimate_pointwise_target_jets"]
