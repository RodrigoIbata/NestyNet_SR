# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from typing import Any, Callable

import torch


def _normalize_diagnostic_set(diagnostic_set: str | None) -> str:
    token = str(diagnostic_set or "basic").strip().lower()
    if token == "physics":
        return "physics"
    if token == "extended":
        return "extended"
    return "basic"


def _primary_axis(x: torch.Tensor | None) -> int:
    if x is None or x.ndim != 2 or int(x.shape[1]) <= 1:
        return 0
    try:
        spread = x.to(dtype=torch.float64).std(dim=0, unbiased=False)
    except Exception:
        return 0
    if spread.numel() <= 0 or not torch.isfinite(spread).any():
        return 0
    return int(torch.argmax(torch.nan_to_num(spread, nan=0.0)).item())


def _sorted_axis_view(
    x: torch.Tensor,
    value: torch.Tensor,
    *,
    grad: torch.Tensor | None = None,
    hdiag: torch.Tensor | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    axis = _primary_axis(x)
    coords = x[:, axis]
    order = torch.argsort(coords)
    value_sorted = value.reshape(-1)[order]
    grad_sorted = None if grad is None else grad[order, axis]
    hdiag_sorted = None if hdiag is None else hdiag[order, axis]
    return axis, coords[order], value_sorted, grad_sorted, hdiag_sorted


def _finite_difference_slopes(
    coords: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if int(coords.numel()) < 2 or int(values.numel()) < 2:
        return torch.zeros((0,), dtype=values.dtype, device=values.device)
    dx = coords[1:] - coords[:-1]
    dy = values[1:] - values[:-1]
    mask = dx.abs() > 1.0e-12
    if not bool(mask.any()):
        return torch.zeros((0,), dtype=values.dtype, device=values.device)
    return dy[mask] / dx[mask]


def _sign_change_count(values: torch.Tensor) -> int:
    raw = values.detach().cpu().reshape(-1).tolist()
    last_sign = 0
    out = 0
    for item in raw:
        if not math.isfinite(float(item)):
            continue
        if abs(float(item)) <= 1.0e-12:
            continue
        sign = 1 if float(item) > 0.0 else -1
        if last_sign != 0 and sign != last_sign:
            out += 1
        last_sign = sign
    return int(out)


def _crossing_positions(coords: torch.Tensor, values: torch.Tensor) -> list[float]:
    xs = coords.detach().cpu().reshape(-1).tolist()
    ys = values.detach().cpu().reshape(-1).tolist()
    if len(xs) < 2 or len(ys) < 2:
        return []
    span = max(1.0e-9, abs(float(xs[-1]) - float(xs[0])))
    tol = 1.0e-9 * span
    out: list[float] = []
    for idx in range(len(xs) - 1):
        x0 = float(xs[idx])
        x1 = float(xs[idx + 1])
        y0 = float(ys[idx])
        y1 = float(ys[idx + 1])
        if not (math.isfinite(x0) and math.isfinite(x1) and math.isfinite(y0) and math.isfinite(y1)):
            continue
        position = None
        if abs(y0) <= 1.0e-12 and abs(y1) <= 1.0e-12:
            position = 0.5 * (x0 + x1)
        elif abs(y0) <= 1.0e-12:
            position = x0
        elif abs(y1) <= 1.0e-12:
            position = x1
        elif y0 * y1 < 0.0:
            frac = abs(y0) / max(1.0e-12, abs(y0) + abs(y1))
            position = x0 + frac * (x1 - x0)
        if position is None:
            continue
        if out and abs(float(position) - float(out[-1])) <= tol:
            continue
        out.append(float(position))
    return out


def _tail_mean(values: torch.Tensor, count: int) -> float | None:
    if values.numel() <= 0:
        return None
    width = max(1, min(int(count), int(values.numel())))
    out = float(values[:width].mean().item())
    return out if math.isfinite(out) else None


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
    elif value.ndim == 2 and int(value.shape[1]) == 1:
        pass
    else:
        try:
            value = value.reshape(int(value.shape[0]), -1).mean(dim=1, keepdim=True)
        except Exception:
            return None
    if int(value.shape[0]) != int(x.shape[0]):
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


def _normalize_hdiag_tensor(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    else:
        value = value.to(dtype=x.dtype, device=x.device)
    if value.ndim == 3 and tuple(value.shape) == (int(x.shape[0]), int(x.shape[1]), int(x.shape[1])):
        out = torch.diagonal(value, dim1=-2, dim2=-1)
    elif value.ndim >= 4 and int(value.shape[0]) == int(x.shape[0]) and tuple(value.shape[-2:]) == (
        int(x.shape[1]),
        int(x.shape[1]),
    ):
        try:
            out = torch.diagonal(
                value.reshape(int(x.shape[0]), -1, int(x.shape[1]), int(x.shape[1])).mean(dim=1),
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


def _quantile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    try:
        out = float(torch.quantile(values, float(q)).item())
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _build_diagnostics(
    value: torch.Tensor | None,
    *,
    x: torch.Tensor | None = None,
    grad: torch.Tensor | None = None,
    hdiag: torch.Tensor | None = None,
    diagnostic_set: str = "basic",
) -> dict[str, float]:
    if value is None:
        return {}
    mode = _normalize_diagnostic_set(diagnostic_set)
    flat = value.reshape(-1)
    if flat.numel() == 0:
        return {}
    out: dict[str, float] = {}
    out["finite_frac"] = float(torch.isfinite(flat).to(dtype=torch.float64).mean().item())
    out["value_mean"] = float(flat.mean().item())
    out["value_std"] = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
    out["value_abs_max"] = float(flat.abs().max().item())
    out["value_min"] = float(flat.min().item())
    out["value_max"] = float(flat.max().item())
    out["positive_frac"] = float((flat > 0.0).to(dtype=torch.float64).mean().item())
    if grad is not None and grad.numel() > 0:
        grad_abs = grad.abs()
        grad_norm = torch.linalg.vector_norm(grad, dim=1)
        out["grad_abs_mean"] = float(grad_abs.mean().item())
        out["grad_norm_mean"] = float(grad_norm.mean().item())
        out["grad_norm_max"] = float(grad_norm.max().item())
    if hdiag is not None and hdiag.numel() > 0:
        hdiag_abs = hdiag.abs()
        out["hdiag_abs_mean"] = float(hdiag_abs.mean().item())
        out["hdiag_abs_max"] = float(hdiag_abs.max().item())
    if mode in {"extended", "physics"}:
        q10 = _quantile(flat, 0.10)
        q50 = _quantile(flat, 0.50)
        q90 = _quantile(flat, 0.90)
        if q10 is not None:
            out["value_q10"] = float(q10)
        if q50 is not None:
            out["value_q50"] = float(q50)
        if q90 is not None:
            out["value_q90"] = float(q90)
    if mode == "physics" and x is not None and int(x.shape[0]) > 0:
        axis, coords_sorted, value_sorted, grad_sorted, hdiag_sorted = _sorted_axis_view(
            x,
            value,
            grad=grad,
            hdiag=hdiag,
        )
        span = max(
            1.0e-12,
            float((coords_sorted[-1] - coords_sorted[0]).abs().item()),
        ) if int(coords_sorted.numel()) > 0 else 1.0
        scale = max(1.0e-12, float(value_sorted.abs().mean().item())) if int(value_sorted.numel()) > 0 else 1.0
        out["primary_axis"] = float(axis)
        out["primary_axis_span"] = float(span)
        out["closest_zero_abs_value"] = float(value_sorted.abs().min().item())
        zero_scale = max(1.0e-6, 0.25 * max(1.0e-12, float(flat.std(unbiased=False).item())))
        out["near_zero_frac"] = float((value_sorted.abs() <= float(zero_scale)).to(dtype=torch.float64).mean().item())
        crossing_positions = _crossing_positions(coords_sorted, value_sorted)
        out["zero_crossing_count"] = float(len(crossing_positions))
        out["zero_crossing_frac"] = float(len(crossing_positions) / max(1, int(value_sorted.numel()) - 1))
        if crossing_positions:
            mean_cross = sum(float(item) for item in crossing_positions) / len(crossing_positions)
            out["zero_crossing_location_mean"] = float(mean_cross)
            if len(crossing_positions) > 1:
                var_cross = sum((float(item) - mean_cross) ** 2 for item in crossing_positions) / len(crossing_positions)
                out["zero_crossing_location_std"] = float(math.sqrt(max(0.0, float(var_cross))))
            else:
                out["zero_crossing_location_std"] = 0.0
        y_rev = torch.flip(value_sorted, dims=[0])
        coords_rev = torch.flip(coords_sorted, dims=[0])
        center = 0.5 * (coords_sorted[0] + coords_sorted[-1]) if int(coords_sorted.numel()) > 0 else torch.zeros((), dtype=flat.dtype, device=flat.device)
        out["mirror_axis_mismatch_mean"] = float(((coords_sorted + coords_rev - (2.0 * center)).abs().mean() / max(1.0e-12, span)).item())
        out["mirror_even_residual"] = float((value_sorted - y_rev).abs().mean().item() / scale)
        out["mirror_odd_residual"] = float((value_sorted + y_rev).abs().mean().item() / scale)
        slopes = _finite_difference_slopes(coords_sorted, value_sorted)
        if int(slopes.numel()) > 0:
            left_tail = _tail_mean(slopes, min(3, int(slopes.numel())))
            right_tail = _tail_mean(torch.flip(slopes, dims=[0]), min(3, int(slopes.numel())))
            if left_tail is not None:
                out["left_tail_slope_mean"] = float(left_tail)
            if right_tail is not None:
                out["right_tail_slope_mean"] = float(right_tail)
            if left_tail is not None and right_tail is not None:
                out["tail_slope_gap_abs"] = float(abs(float(right_tail) - float(left_tail)))
            out["slope_sign_change_count"] = float(_sign_change_count(slopes))
        monotonic_source = None
        if grad_sorted is not None and int(grad_sorted.numel()) > 0:
            monotonic_source = grad_sorted
            out["monotonicity_pos_frac"] = float((grad_sorted > 0.0).to(dtype=torch.float64).mean().item())
            out["monotonicity_neg_frac"] = float((grad_sorted < 0.0).to(dtype=torch.float64).mean().item())
            out["monotonicity_abs_mean"] = float(grad_sorted.abs().mean().item())
        elif int(slopes.numel()) > 0:
            monotonic_source = slopes
            out["monotonicity_pos_frac"] = float((slopes > 0.0).to(dtype=torch.float64).mean().item())
            out["monotonicity_neg_frac"] = float((slopes < 0.0).to(dtype=torch.float64).mean().item())
            out["monotonicity_abs_mean"] = float(slopes.abs().mean().item())
        if monotonic_source is not None:
            monotonic_switches = _sign_change_count(monotonic_source)
            out["monotonicity_switch_count"] = float(monotonic_switches)
            out["monotonicity_switch_frac"] = float(monotonic_switches / max(1, int(monotonic_source.numel()) - 1))
        curvature_source = None
        if hdiag_sorted is not None and int(hdiag_sorted.numel()) > 0:
            curvature_source = hdiag_sorted
            out["convexity_pos_frac"] = float((hdiag_sorted > 0.0).to(dtype=torch.float64).mean().item())
            out["convexity_neg_frac"] = float((hdiag_sorted < 0.0).to(dtype=torch.float64).mean().item())
            out["curvature_abs_mean"] = float(hdiag_sorted.abs().mean().item())
        elif int(slopes.numel()) > 1:
            curvature_proxy = slopes[1:] - slopes[:-1]
            curvature_source = curvature_proxy
            out["convexity_pos_frac"] = float((curvature_proxy > 0.0).to(dtype=torch.float64).mean().item())
            out["convexity_neg_frac"] = float((curvature_proxy < 0.0).to(dtype=torch.float64).mean().item())
            out["curvature_abs_mean"] = float(curvature_proxy.abs().mean().item())
        if curvature_source is not None:
            convexity_switches = _sign_change_count(curvature_source)
            out["convexity_switch_count"] = float(convexity_switches)
            out["convexity_switch_frac"] = float(convexity_switches / max(1, int(curvature_source.numel()) - 1))
        grad_spike_ratio = None
        if grad is not None and grad.numel() > 0:
            grad_norm = torch.linalg.vector_norm(grad, dim=1)
            grad_spike_ratio = float(grad_norm.max().item() / max(1.0e-12, float(grad_norm.mean().item())))
            out["gradient_spike_ratio"] = float(grad_spike_ratio)
        elif int(slopes.numel()) > 0:
            grad_spike_ratio = float(slopes.abs().max().item() / max(1.0e-12, float(slopes.abs().mean().item())))
            out["gradient_spike_ratio"] = float(grad_spike_ratio)
        curvature_spike_ratio = None
        if hdiag is not None and hdiag.numel() > 0:
            hdiag_abs = hdiag.abs()
            curvature_spike_ratio = float(hdiag_abs.max().item() / max(1.0e-12, float(hdiag_abs.mean().item())))
            out["curvature_spike_ratio"] = float(curvature_spike_ratio)
        elif curvature_source is not None and int(curvature_source.numel()) > 0:
            curvature_abs = curvature_source.abs()
            curvature_spike_ratio = float(curvature_abs.max().item() / max(1.0e-12, float(curvature_abs.mean().item())))
            out["curvature_spike_ratio"] = float(curvature_spike_ratio)
        singularity_margin = 1.0 / (
            1.0
            + float(grad_spike_ratio or 0.0)
            + float(curvature_spike_ratio or 0.0)
        )
        out["singularity_margin_proxy"] = float(singularity_margin)
        out["domain_stability_proxy"] = float(out["finite_frac"] * singularity_margin)
    return out


def _autograd_value_grad_hdiag(
    forward_fn: Callable[[torch.Tensor], torch.Tensor | None],
    x: torch.Tensor,
    *,
    capture_gradients: bool,
    capture_hessian_diag: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    try:
        x_req = x.detach().clone().requires_grad_(True)
        value = _normalize_value_tensor(forward_fn(x_req), x=x_req)
        if value is None:
            return None, None, None
        if not capture_gradients and not capture_hessian_diag:
            return value.detach(), None, None
        grad_rows: list[torch.Tensor] = []
        hdiag_rows: list[torch.Tensor] = []
        n_points = int(x_req.shape[0])
        n_vars = int(x_req.shape[1])
        for idx in range(n_points):
            grad_full = torch.autograd.grad(
                value[idx, 0],
                x_req,
                retain_graph=True,
                create_graph=bool(capture_hessian_diag),
                allow_unused=True,
            )[0]
            if grad_full is None:
                grad_i = torch.zeros((n_vars,), dtype=x_req.dtype, device=x_req.device)
            else:
                grad_i = grad_full[idx]
            if capture_hessian_diag:
                hdiag_i: list[torch.Tensor] = []
                for dim in range(n_vars):
                    second_full = torch.autograd.grad(
                        grad_i[dim],
                        x_req,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if second_full is None:
                        hdiag_i.append(torch.zeros((), dtype=x_req.dtype, device=x_req.device))
                    else:
                        hdiag_i.append(second_full[idx, dim])
                hdiag_rows.append(torch.stack(hdiag_i))
            grad_rows.append(grad_i)
        grad = torch.stack(grad_rows).detach() if grad_rows else None
        hdiag = torch.stack(hdiag_rows).detach() if hdiag_rows else None
        return value.detach(), grad, hdiag
    except Exception:
        return None, None, None


def capture_runtime_witness(
    candidate: Any,
    x: torch.Tensor,
    *,
    predict_value_fn: Callable[[Any, torch.Tensor], torch.Tensor | None],
    capture_gradients: bool = False,
    capture_hessian_diag: bool = False,
    diagnostic_set: str = "basic",
) -> dict[str, Any]:
    value = _normalize_value_tensor(predict_value_fn(candidate, x), x=x)
    grad = None
    hdiag = None
    model = getattr(candidate, "model", None)
    if value is not None and (capture_gradients or capture_hessian_diag) and model is not None:
        y_inverse = getattr(candidate, "y_inverse", None)
        if y_inverse is None and capture_gradients and callable(getattr(model, "grad", None)):
            try:
                grad = _normalize_grad_tensor(model.grad(x), x=x)
            except Exception:
                grad = None
        if y_inverse is None and capture_hessian_diag and callable(getattr(model, "grad_grad", None)):
            try:
                hdiag = _normalize_hdiag_tensor(model.grad_grad(x), x=x)
            except Exception:
                hdiag = None
        if (capture_gradients and grad is None) or (capture_hessian_diag and hdiag is None):
            auto_value, auto_grad, auto_hdiag = _autograd_value_grad_hdiag(
                lambda xx: _runtime_forward_value(candidate, xx),
                x,
                capture_gradients=bool(capture_gradients),
                capture_hessian_diag=bool(capture_hessian_diag),
            )
            if auto_value is not None:
                value = auto_value
            if grad is None:
                grad = auto_grad
            if hdiag is None:
                hdiag = auto_hdiag
    diagnostics = _build_diagnostics(value, x=x, grad=grad, hdiag=hdiag, diagnostic_set=diagnostic_set)
    return {
        "observable": None if value is None else value[:, 0],
        "derivative": grad,
        "diagnostic": diagnostics,
    }


def _runtime_forward_value(candidate: Any, x: torch.Tensor) -> torch.Tensor | None:
    model = getattr(candidate, "model", None)
    if model is None:
        return None
    try:
        y_hat = model(x)
        if not torch.is_tensor(y_hat):
            return None
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        elif y_hat.ndim > 2:
            y_hat = y_hat.reshape(y_hat.shape[0], -1)
        y_inverse = getattr(candidate, "y_inverse", None)
        if callable(y_inverse):
            y_hat = y_inverse(y_hat)
        return _normalize_value_tensor(y_hat, x=x)
    except Exception:
        return None


def capture_symbolic_witness(
    *,
    expr_ast: Any,
    x: torch.Tensor,
    forward_value_fn: Callable[[Any, torch.Tensor], torch.Tensor | None],
    capture_gradients: bool = False,
    capture_hessian_diag: bool = False,
    diagnostic_set: str = "basic",
) -> dict[str, Any]:
    value = _normalize_value_tensor(forward_value_fn(expr_ast, x), x=x)
    grad = None
    hdiag = None
    if value is not None and (capture_gradients or capture_hessian_diag):
        auto_value, auto_grad, auto_hdiag = _autograd_value_grad_hdiag(
            lambda xx: forward_value_fn(expr_ast, xx),
            x,
            capture_gradients=bool(capture_gradients),
            capture_hessian_diag=bool(capture_hessian_diag),
        )
        if auto_value is not None:
            value = auto_value
            grad = auto_grad
            hdiag = auto_hdiag
    diagnostics = _build_diagnostics(value, x=x, grad=grad, hdiag=hdiag, diagnostic_set=diagnostic_set)
    return {
        "observable": None if value is None else value[:, 0],
        "derivative": grad,
        "diagnostic": diagnostics,
    }


__all__ = [
    "capture_runtime_witness",
    "capture_symbolic_witness",
]
