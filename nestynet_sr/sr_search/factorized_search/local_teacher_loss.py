# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .expr_ast import eval_node
from .inverse_core import (
    _ensure_col,
    _prepare_nonnegative_weights,
    _score_inverse_local_predictions,
)


def _ensure_matrix(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    if value.ndim == 1:
        return value.unsqueeze(-1)
    if value.ndim == 2:
        return value
    raise ValueError(f"expected [N] or [N,D] tensor, got shape={tuple(value.shape)}")


def _weighted_matrix_mse(
    target: torch.Tensor | None,
    pred: torch.Tensor | None,
    w: torch.Tensor | None,
) -> float | None:
    if target is None or pred is None:
        return None
    try:
        yt = _ensure_matrix(target)
        yp = _ensure_matrix(pred).to(dtype=yt.dtype, device=yt.device)
    except Exception:
        return None
    if tuple(yt.shape) != tuple(yp.shape):
        return None
    ww = _prepare_nonnegative_weights(w, yt[:, :1])
    wm = ww.expand_as(yt)
    mask = torch.isfinite(yt) & torch.isfinite(yp) & torch.isfinite(wm) & (wm > 0.0)
    if int(mask.sum().item()) <= 0:
        return None
    err2 = (yt[mask] - yp[mask]).pow(2)
    weights = wm[mask]
    den = float(weights.sum().item())
    if (not math.isfinite(den)) or den <= 1.0e-18:
        return None
    num = float((weights * err2).sum().item())
    if not math.isfinite(num):
        return None
    return float(num / den)


def _weighted_vector_stats(
    value: torch.Tensor | None,
    w: torch.Tensor | None,
) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        vv = _ensure_matrix(value)
    except Exception:
        return None
    vv = vv.reshape(vv.shape[0], -1)
    ww = _prepare_nonnegative_weights(w, vv[:, :1]).expand_as(vv)
    mask = torch.isfinite(vv) & torch.isfinite(ww) & (ww > 0.0)
    if int(mask.sum().item()) <= 0:
        return None
    vals = vv[mask]
    weights = ww[mask]
    scale = max(1.0, float(torch.max(torch.abs(vals)).item()))
    vals = torch.where(torch.abs(vals) <= (1.0e-12 * scale), torch.zeros_like(vals), vals)
    den = float(weights.sum().item())
    if (not math.isfinite(den)) or den <= 1.0e-18:
        return None
    mean = float((weights * vals).sum().item() / den)
    abs_mean = float((weights * torch.abs(vals)).sum().item() / den)
    centered = vals - mean
    var = float((weights * centered.pow(2)).sum().item() / den)
    positive = float((weights * (vals > 0.0).to(vals.dtype)).sum().item() / den)
    negative = float((weights * (vals < 0.0).to(vals.dtype)).sum().item() / den)
    return {
        "mean": float(mean),
        "std": float(math.sqrt(max(0.0, var))),
        "abs_mean": float(abs_mean),
        "positive_fraction": float(positive),
        "negative_fraction": float(negative),
    }


def _ordered_vector(
    value: torch.Tensor | None,
    *,
    x: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if value is None:
        return None
    try:
        vv = _ensure_col(value).reshape(-1)
    except Exception:
        return None
    mask = torch.isfinite(vv)
    if x is not None:
        try:
            xx = _ensure_matrix(x)
        except Exception:
            xx = None
        if xx is not None and int(xx.shape[0]) == int(vv.shape[0]) and int(xx.shape[1]) == 1:
            x0 = xx[:, 0]
            mask = mask & torch.isfinite(x0)
            if int(mask.sum().item()) <= 0:
                return None
            vv = vv[mask]
            x0 = x0[mask]
            order = torch.argsort(x0)
            ordered = vv.index_select(0, order)
            scale = max(1.0, float(torch.max(torch.abs(ordered)).item())) if int(ordered.numel()) > 0 else 1.0
            return torch.where(torch.abs(ordered) <= (1.0e-12 * scale), torch.zeros_like(ordered), ordered)
    if int(mask.sum().item()) <= 0:
        return None
    ordered = vv[mask]
    scale = max(1.0, float(torch.max(torch.abs(ordered)).item())) if int(ordered.numel()) > 0 else 1.0
    return torch.where(torch.abs(ordered) <= (1.0e-12 * scale), torch.zeros_like(ordered), ordered)


def _zero_crossing_count(value: torch.Tensor | None, *, x: torch.Tensor | None = None) -> int | None:
    ordered = _ordered_vector(value, x=x)
    if ordered is None or int(ordered.numel()) <= 1:
        return None
    vals = [float(item) for item in ordered.detach().cpu().tolist()]
    eps = 1.0e-12
    last_sign = 0
    count = 0
    for scalar in vals:
        sign = 0
        if scalar > eps:
            sign = 1
        elif scalar < -eps:
            sign = -1
        if sign == 0:
            continue
        if last_sign != 0 and sign != last_sign:
            count += 1
        last_sign = sign
    return int(count)


def _sign_flag(
    value: torch.Tensor | None,
    *,
    threshold: float = 0.9,
) -> int | None:
    stats = _weighted_vector_stats(value, None)
    if stats is None:
        return None
    pos = float(stats["positive_fraction"])
    neg = float(stats["negative_fraction"])
    if pos >= float(threshold):
        return 1
    if neg >= float(threshold):
        return -1
    return 0


def _monotonic_flag(
    *,
    x: torch.Tensor,
    value: torch.Tensor | None,
    grad: torch.Tensor | None,
    threshold: float = 0.9,
) -> int | None:
    grad_stats = _weighted_vector_stats(grad, None)
    if grad_stats is not None:
        pos = float(grad_stats["positive_fraction"])
        neg = float(grad_stats["negative_fraction"])
        if pos >= float(threshold):
            return 1
        if neg >= float(threshold):
            return -1
        return 0
    ordered = _ordered_vector(value, x=x)
    if ordered is None or int(ordered.numel()) <= 1:
        return None
    diffs = ordered[1:] - ordered[:-1]
    return _sign_flag(diffs, threshold=threshold)


def _convexity_flag(
    *,
    x: torch.Tensor,
    value: torch.Tensor | None,
    d2: torch.Tensor | None,
    threshold: float = 0.9,
) -> int | None:
    d2_stats = _weighted_vector_stats(d2, None)
    if d2_stats is not None:
        pos = float(d2_stats["positive_fraction"])
        neg = float(d2_stats["negative_fraction"])
        if pos >= float(threshold):
            return 1
        if neg >= float(threshold):
            return -1
        return 0
    ordered = _ordered_vector(value, x=x)
    if ordered is None or int(ordered.numel()) <= 2:
        return None
    second = ordered[2:] - (2.0 * ordered[1:-1]) + ordered[:-2]
    return _sign_flag(second, threshold=threshold)


def _diagnostic_summary(
    *,
    x: torch.Tensor,
    value: torch.Tensor | None,
    grad: torch.Tensor | None,
    d2: torch.Tensor | None,
    w: torch.Tensor | None,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    value_stats = _weighted_vector_stats(value, w)
    if value_stats is not None:
        summary["value_abs_mean"] = float(value_stats["abs_mean"])
        summary["value_std"] = float(value_stats["std"])
        summary["value_positive_fraction"] = float(value_stats["positive_fraction"])
        zero_crossing_count = _zero_crossing_count(value, x=x)
        if zero_crossing_count is not None:
            denom = max(1, int(_ensure_col(value).shape[0]) - 1)
            summary["value_zero_crossing_rate"] = float(zero_crossing_count) / float(denom)
    grad_stats = _weighted_vector_stats(grad, w)
    if grad_stats is not None:
        summary["grad_abs_mean"] = float(grad_stats["abs_mean"])
        summary["grad_positive_fraction"] = float(grad_stats["positive_fraction"])
    d2_stats = _weighted_vector_stats(d2, w)
    if d2_stats is not None:
        summary["d2_abs_mean"] = float(d2_stats["abs_mean"])
        summary["d2_positive_fraction"] = float(d2_stats["positive_fraction"])
    monotonic_flag = _monotonic_flag(x=x, value=value, grad=grad)
    if monotonic_flag is not None:
        summary["monotonicity_score"] = 1.0 if int(monotonic_flag) != 0 else 0.0
    convexity_flag = _convexity_flag(x=x, value=value, d2=d2)
    if convexity_flag is not None:
        summary["convexity_score"] = 1.0 if int(convexity_flag) != 0 else 0.0
    return summary


def _physics_summary(
    *,
    x: torch.Tensor,
    value: torch.Tensor | None,
    grad: torch.Tensor | None,
    d2: torch.Tensor | None,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    monotonic_flag = _monotonic_flag(x=x, value=value, grad=grad)
    if monotonic_flag is not None:
        summary["monotonic_flag"] = float(monotonic_flag)
    convexity_flag = _convexity_flag(x=x, value=value, d2=d2)
    if convexity_flag is not None:
        summary["convexity_flag"] = float(convexity_flag)
    sign_flag = _sign_flag(value)
    if sign_flag is not None:
        summary["sign_regime_flag"] = float(sign_flag)
    zero_crossing_count = _zero_crossing_count(value, x=x)
    if zero_crossing_count is not None:
        summary["zero_crossing_count"] = float(zero_crossing_count)
    return summary


def _normalized_scalar_sq_error(key: str, target: float, pred: float) -> float:
    token = str(key or "")
    if any(marker in token for marker in ("fraction", "rate", "score", "flag")):
        scale = 1.0
    elif "count" in token:
        scale = max(1.0, abs(float(target)), abs(float(pred)))
    else:
        scale = max(1.0, abs(float(target)), abs(float(pred)))
    err = (float(pred) - float(target)) / float(scale)
    return float(err * err)


def _mean_summary_loss(
    target_summary: Mapping[str, float] | None,
    pred_summary: Mapping[str, float] | None,
) -> float | None:
    target_map = {str(k): float(v) for k, v in dict(target_summary or {}).items() if math.isfinite(float(v))}
    pred_map = {str(k): float(v) for k, v in dict(pred_summary or {}).items() if math.isfinite(float(v))}
    common = sorted(set(target_map.keys()) & set(pred_map.keys()))
    if not common:
        return None
    terms = [_normalized_scalar_sq_error(key, target_map[key], pred_map[key]) for key in common]
    return float(sum(terms) / max(1, len(terms)))


def _physics_summary_loss(
    target_summary: Mapping[str, float] | None,
    pred_summary: Mapping[str, float] | None,
) -> float | None:
    target_map = {str(k): float(v) for k, v in dict(target_summary or {}).items() if math.isfinite(float(v))}
    pred_map = {str(k): float(v) for k, v in dict(pred_summary or {}).items() if math.isfinite(float(v))}
    terms: list[float] = []
    for key in ("monotonic_flag", "convexity_flag", "sign_regime_flag"):
        if key not in target_map or key not in pred_map:
            continue
        target_flag = int(round(float(target_map[key])))
        pred_flag = int(round(float(pred_map[key])))
        if target_flag == 0 and pred_flag == 0:
            terms.append(0.0)
        elif target_flag == 0 or pred_flag == 0:
            terms.append(0.5)
        else:
            terms.append(0.0 if target_flag == pred_flag else 1.0)
    if "zero_crossing_count" in target_map and "zero_crossing_count" in pred_map:
        terms.append(
            _normalized_scalar_sq_error(
                "zero_crossing_count",
                float(target_map["zero_crossing_count"]),
                float(pred_map["zero_crossing_count"]),
            )
        )
    if not terms:
        return None
    return float(sum(terms) / max(1, len(terms)))


def _diagnostic_confidence_scale(target_diagnostics: Mapping[str, Any] | None) -> float:
    diag = dict(target_diagnostics or {})
    confidence = diag.get("confidence", 1.0)
    valid_frac = diag.get("valid_frac", 1.0)
    try:
        conf = float(confidence)
    except Exception:
        conf = 1.0
    try:
        valid = float(valid_frac)
    except Exception:
        valid = 1.0
    conf = min(1.0, max(0.0, conf))
    valid = min(1.0, max(0.0, valid))
    return float(conf * valid)


def _fit_affine_calibration(
    pred_fit: torch.Tensor,
    target_fit: torch.Tensor,
    *,
    w_fit: torch.Tensor | None,
    mode: str,
) -> tuple[float, float]:
    mode_name = str(mode or "affine").strip().lower()
    if mode_name in ("strict", "direct"):
        return 1.0, 0.0
    pf = _ensure_col(pred_fit)
    tf = _ensure_col(target_fit).to(dtype=pf.dtype, device=pf.device)
    ww = _prepare_nonnegative_weights(w_fit, tf)
    mask = torch.isfinite(pf) & torch.isfinite(tf) & torch.isfinite(ww) & (ww > 0.0)
    if int(mask.sum().item()) <= 0:
        return 1.0, 0.0
    p_sel = pf[mask].reshape(-1, 1)
    t_sel = tf[mask].reshape(-1, 1)
    w_sel = ww[mask].reshape(-1, 1)
    design = torch.cat([p_sel, torch.ones_like(p_sel)], dim=1)
    sqrt_w = torch.sqrt(torch.clamp(w_sel, min=1.0e-12))
    aw = design * sqrt_w
    bw = t_sel * sqrt_w
    try:
        sol = torch.linalg.lstsq(aw, bw).solution
    except Exception:
        return 1.0, 0.0
    if int(sol.shape[0]) < 2 or not torch.isfinite(sol[:2]).all():
        return 1.0, 0.0
    return float(sol[0, 0].item()), float(sol[1, 0].item())


def _autograd_node_jets(
    node,
    x: torch.Tensor,
    *,
    capture_grad: bool,
    capture_d2: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    try:
        with torch.enable_grad():
            x_req = _ensure_matrix(x).detach().clone().requires_grad_(True)
            value = _ensure_col(eval_node(node, x_req))
            if not torch.isfinite(value).all():
                return None, None, None
            if not capture_grad and not capture_d2:
                return value.detach(), None, None
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
                grad_rows.append(grad_i)
            grad = torch.stack(grad_rows).detach() if grad_rows else None
            d2 = torch.stack(d2_rows).detach() if d2_rows else None
            return value.detach(), grad, d2
    except Exception:
        return None, None, None


def _autograd_predict_jets(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    capture_grad: bool,
    capture_d2: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    try:
        with torch.enable_grad():
            x_req = _ensure_matrix(x).detach().clone().requires_grad_(True)
            value = _ensure_col(predict_fn(x_req))
            if not torch.isfinite(value).all():
                return None, None, None
            if not capture_grad and not capture_d2:
                return value.detach(), None, None
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
                grad_rows.append(grad_i)
            grad = torch.stack(grad_rows).detach() if grad_rows else None
            d2 = torch.stack(d2_rows).detach() if d2_rows else None
            return value.detach(), grad, d2
    except Exception:
        return None, None, None


@dataclass(frozen=True)
class LocalTeacherLoss:
    value_fit_loss: float
    value_probe_loss: float
    grad_fit_loss: float | None = None
    grad_probe_loss: float | None = None
    d2_fit_loss: float | None = None
    d2_probe_loss: float | None = None
    diag_fit_loss: float | None = None
    diag_probe_loss: float | None = None
    physics_fit_loss: float | None = None
    physics_probe_loss: float | None = None
    fit_total: float = 0.0
    probe_total: float = 0.0
    calibration_scale: float = 1.0
    calibration_bias: float = 0.0
    used_gradient_loss: bool = False
    used_d2_loss: bool = False
    fit_jet_source: str = ""
    probe_jet_source: str = ""
    fit_jet_requested_source: str = ""
    probe_jet_requested_source: str = ""
    fit_jet_fallback_used: bool = False
    probe_jet_fallback_used: bool = False
    exact_jet_used: bool = False


def _teacher_jet_provenance(target_diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    diagnostics = dict(target_diagnostics or {})
    fit_source = str(diagnostics.get("fit_jet_source", diagnostics.get("witness_fit_jet_source", "")) or "")
    probe_source = str(diagnostics.get("probe_jet_source", diagnostics.get("witness_probe_jet_source", "")) or "")
    fit_requested = str(
        diagnostics.get("fit_jet_requested_source", diagnostics.get("witness_fit_jet_requested_source", fit_source)) or fit_source
    )
    probe_requested = str(
        diagnostics.get("probe_jet_requested_source", diagnostics.get("witness_probe_jet_requested_source", probe_source)) or probe_source
    )
    fit_fallback = bool(diagnostics.get("fit_jet_fallback_used", False))
    probe_fallback = bool(diagnostics.get("probe_jet_fallback_used", False))
    exact_tokens = {"oracle", "symbolic", "runtime_teacher"}
    return {
        "fit_jet_source": str(fit_source),
        "probe_jet_source": str(probe_source),
        "fit_jet_requested_source": str(fit_requested),
        "probe_jet_requested_source": str(probe_requested),
        "fit_jet_fallback_used": bool(fit_fallback),
        "probe_jet_fallback_used": bool(probe_fallback),
        "exact_jet_used": bool(str(fit_source) in exact_tokens or str(probe_source) in exact_tokens),
    }


def _score_local_teacher_from_predictions(
    *,
    pred_fit: torch.Tensor,
    pred_probe: torch.Tensor,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    target_fit: torch.Tensor,
    target_probe: torch.Tensor,
    w_fit: torch.Tensor | None,
    w_probe: torch.Tensor | None,
    target_grad_fit: torch.Tensor | None = None,
    target_grad_probe: torch.Tensor | None = None,
    target_d2_fit: torch.Tensor | None = None,
    target_d2_probe: torch.Tensor | None = None,
    poly_degree: int,
    mode: str = "affine",
    grad_weight: float = 0.0,
    d2_weight: float = 0.0,
    diag_weight: float = 0.0,
    physics_weight: float = 0.0,
    target_diagnostics: Mapping[str, Any] | None = None,
    jet_provider: Callable[[torch.Tensor, bool, bool], tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]
    | None = None,
) -> LocalTeacherLoss | None:
    value_score = _score_inverse_local_predictions(
        pred_fit,
        pred_probe,
        target_fit,
        target_probe,
        w_fit=w_fit,
        w_probe=w_probe,
        poly_degree=int(poly_degree),
        mode=str(mode),
    )
    if value_score is None:
        return None
    value_fit_loss, value_probe_loss = value_score
    need_grad = (
        max(0.0, float(grad_weight)) > 0.0
        or max(0.0, float(diag_weight)) > 0.0
        or max(0.0, float(physics_weight)) > 0.0
    ) and (target_grad_fit is not None or target_grad_probe is not None)
    need_d2 = (
        max(0.0, float(d2_weight)) > 0.0
        or max(0.0, float(diag_weight)) > 0.0
        or max(0.0, float(physics_weight)) > 0.0
    ) and (target_d2_fit is not None or target_d2_probe is not None)
    calibration_scale, calibration_bias = _fit_affine_calibration(
        pred_fit,
        target_fit,
        w_fit=w_fit,
        mode=str(mode),
    )
    grad_fit_loss = None
    grad_probe_loss = None
    d2_fit_loss = None
    d2_probe_loss = None
    diag_fit_loss = None
    diag_probe_loss = None
    physics_fit_loss = None
    physics_probe_loss = None
    pred_grad_fit = None
    pred_grad_probe = None
    pred_d2_fit = None
    pred_d2_probe = None
    if (need_grad or need_d2) and jet_provider is not None:
        _fit_value, pred_grad_fit, pred_d2_fit = jet_provider(x_fit, bool(need_grad), bool(need_d2))
        _probe_value, pred_grad_probe, pred_d2_probe = jet_provider(x_probe, bool(need_grad), bool(need_d2))
        if pred_grad_fit is not None and target_grad_fit is not None:
            grad_fit_loss = _weighted_matrix_mse(
                _ensure_matrix(target_grad_fit),
                calibration_scale * _ensure_matrix(pred_grad_fit),
                w_fit,
            )
        if pred_grad_probe is not None and target_grad_probe is not None:
            grad_probe_loss = _weighted_matrix_mse(
                _ensure_matrix(target_grad_probe),
                calibration_scale * _ensure_matrix(pred_grad_probe),
                w_probe,
            )
        if pred_d2_fit is not None and target_d2_fit is not None:
            d2_fit_loss = _weighted_matrix_mse(
                _ensure_matrix(target_d2_fit),
                calibration_scale * _ensure_matrix(pred_d2_fit),
                w_fit,
            )
        if pred_d2_probe is not None and target_d2_probe is not None:
            d2_probe_loss = _weighted_matrix_mse(
                _ensure_matrix(target_d2_probe),
                calibration_scale * _ensure_matrix(pred_d2_probe),
                w_probe,
            )
    cal_pred_fit = (calibration_scale * _ensure_col(pred_fit)) + calibration_bias
    cal_pred_probe = (calibration_scale * _ensure_col(pred_probe)) + calibration_bias
    cal_pred_grad_fit = None if pred_grad_fit is None else (calibration_scale * _ensure_matrix(pred_grad_fit))
    cal_pred_grad_probe = None if pred_grad_probe is None else (calibration_scale * _ensure_matrix(pred_grad_probe))
    cal_pred_d2_fit = None if pred_d2_fit is None else (calibration_scale * _ensure_matrix(pred_d2_fit))
    cal_pred_d2_probe = None if pred_d2_probe is None else (calibration_scale * _ensure_matrix(pred_d2_probe))
    diag_scale = _diagnostic_confidence_scale(target_diagnostics)
    if max(0.0, float(diag_weight)) > 0.0:
        target_fit_diag = _diagnostic_summary(
            x=x_fit,
            value=target_fit,
            grad=target_grad_fit,
            d2=target_d2_fit,
            w=w_fit,
        )
        pred_fit_diag = _diagnostic_summary(
            x=x_fit,
            value=cal_pred_fit,
            grad=cal_pred_grad_fit,
            d2=cal_pred_d2_fit,
            w=w_fit,
        )
        target_probe_diag = _diagnostic_summary(
            x=x_probe,
            value=target_probe,
            grad=target_grad_probe,
            d2=target_d2_probe,
            w=w_probe,
        )
        pred_probe_diag = _diagnostic_summary(
            x=x_probe,
            value=cal_pred_probe,
            grad=cal_pred_grad_probe,
            d2=cal_pred_d2_probe,
            w=w_probe,
        )
        raw_fit_diag_loss = _mean_summary_loss(target_fit_diag, pred_fit_diag)
        raw_probe_diag_loss = _mean_summary_loss(target_probe_diag, pred_probe_diag)
        if raw_fit_diag_loss is not None:
            diag_fit_loss = float(diag_scale * raw_fit_diag_loss)
        if raw_probe_diag_loss is not None:
            diag_probe_loss = float(diag_scale * raw_probe_diag_loss)
    if max(0.0, float(physics_weight)) > 0.0:
        target_fit_physics = _physics_summary(
            x=x_fit,
            value=target_fit,
            grad=target_grad_fit,
            d2=target_d2_fit,
        )
        pred_fit_physics = _physics_summary(
            x=x_fit,
            value=cal_pred_fit,
            grad=cal_pred_grad_fit,
            d2=cal_pred_d2_fit,
        )
        target_probe_physics = _physics_summary(
            x=x_probe,
            value=target_probe,
            grad=target_grad_probe,
            d2=target_d2_probe,
        )
        pred_probe_physics = _physics_summary(
            x=x_probe,
            value=cal_pred_probe,
            grad=cal_pred_grad_probe,
            d2=cal_pred_d2_probe,
        )
        raw_fit_physics_loss = _physics_summary_loss(target_fit_physics, pred_fit_physics)
        raw_probe_physics_loss = _physics_summary_loss(target_probe_physics, pred_probe_physics)
        if raw_fit_physics_loss is not None:
            physics_fit_loss = float(diag_scale * raw_fit_physics_loss)
        if raw_probe_physics_loss is not None:
            physics_probe_loss = float(diag_scale * raw_probe_physics_loss)

    fit_total = float(value_fit_loss)
    probe_total = float(value_probe_loss)
    if grad_fit_loss is not None:
        fit_total += float(max(0.0, float(grad_weight)) * float(grad_fit_loss))
    if grad_probe_loss is not None:
        probe_total += float(max(0.0, float(grad_weight)) * float(grad_probe_loss))
    if d2_fit_loss is not None:
        fit_total += float(max(0.0, float(d2_weight)) * float(d2_fit_loss))
    if d2_probe_loss is not None:
        probe_total += float(max(0.0, float(d2_weight)) * float(d2_probe_loss))
    if diag_fit_loss is not None:
        fit_total += float(max(0.0, float(diag_weight)) * float(diag_fit_loss))
    if diag_probe_loss is not None:
        probe_total += float(max(0.0, float(diag_weight)) * float(diag_probe_loss))
    if physics_fit_loss is not None:
        fit_total += float(max(0.0, float(physics_weight)) * float(physics_fit_loss))
    if physics_probe_loss is not None:
        probe_total += float(max(0.0, float(physics_weight)) * float(physics_probe_loss))

    provenance = _teacher_jet_provenance(target_diagnostics)

    return LocalTeacherLoss(
        value_fit_loss=float(value_fit_loss),
        value_probe_loss=float(value_probe_loss),
        grad_fit_loss=None if grad_fit_loss is None else float(grad_fit_loss),
        grad_probe_loss=None if grad_probe_loss is None else float(grad_probe_loss),
        d2_fit_loss=None if d2_fit_loss is None else float(d2_fit_loss),
        d2_probe_loss=None if d2_probe_loss is None else float(d2_probe_loss),
        diag_fit_loss=None if diag_fit_loss is None else float(diag_fit_loss),
        diag_probe_loss=None if diag_probe_loss is None else float(diag_probe_loss),
        physics_fit_loss=None if physics_fit_loss is None else float(physics_fit_loss),
        physics_probe_loss=None if physics_probe_loss is None else float(physics_probe_loss),
        fit_total=float(fit_total),
        probe_total=float(probe_total),
        calibration_scale=float(calibration_scale),
        calibration_bias=float(calibration_bias),
        used_gradient_loss=bool(grad_fit_loss is not None or grad_probe_loss is not None),
        used_d2_loss=bool(d2_fit_loss is not None or d2_probe_loss is not None),
        fit_jet_source=str(provenance["fit_jet_source"]),
        probe_jet_source=str(provenance["probe_jet_source"]),
        fit_jet_requested_source=str(provenance["fit_jet_requested_source"]),
        probe_jet_requested_source=str(provenance["probe_jet_requested_source"]),
        fit_jet_fallback_used=bool(provenance["fit_jet_fallback_used"]),
        probe_jet_fallback_used=bool(provenance["probe_jet_fallback_used"]),
        exact_jet_used=bool(provenance["exact_jet_used"]),
    )


def score_local_teacher_loss(
    node,
    *,
    pred_fit: torch.Tensor,
    pred_probe: torch.Tensor,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    target_fit: torch.Tensor,
    target_probe: torch.Tensor,
    w_fit: torch.Tensor | None,
    w_probe: torch.Tensor | None,
    target_grad_fit: torch.Tensor | None = None,
    target_grad_probe: torch.Tensor | None = None,
    target_d2_fit: torch.Tensor | None = None,
    target_d2_probe: torch.Tensor | None = None,
    poly_degree: int = 2,
    mode: str = "affine",
    grad_weight: float = 0.0,
    d2_weight: float = 0.0,
    diag_weight: float = 0.0,
    physics_weight: float = 0.0,
    target_diagnostics: Mapping[str, Any] | None = None,
) -> LocalTeacherLoss | None:
    return _score_local_teacher_from_predictions(
        pred_fit=pred_fit,
        pred_probe=pred_probe,
        x_fit=x_fit,
        x_probe=x_probe,
        target_fit=target_fit,
        target_probe=target_probe,
        w_fit=w_fit,
        w_probe=w_probe,
        target_grad_fit=target_grad_fit,
        target_grad_probe=target_grad_probe,
        target_d2_fit=target_d2_fit,
        target_d2_probe=target_d2_probe,
        poly_degree=int(poly_degree),
        mode=str(mode),
        grad_weight=float(grad_weight),
        d2_weight=float(d2_weight),
        diag_weight=float(diag_weight),
        physics_weight=float(physics_weight),
        target_diagnostics=target_diagnostics,
        jet_provider=lambda x, capture_grad, capture_d2: _autograd_node_jets(
            node,
            x,
            capture_grad=capture_grad,
            capture_d2=capture_d2,
        ),
    )


def score_local_teacher_prediction_loss(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    target_fit: torch.Tensor,
    target_probe: torch.Tensor,
    w_fit: torch.Tensor | None,
    w_probe: torch.Tensor | None,
    target_grad_fit: torch.Tensor | None = None,
    target_grad_probe: torch.Tensor | None = None,
    target_d2_fit: torch.Tensor | None = None,
    target_d2_probe: torch.Tensor | None = None,
    poly_degree: int = 2,
    mode: str = "strict",
    grad_weight: float = 0.0,
    d2_weight: float = 0.0,
    diag_weight: float = 0.0,
    physics_weight: float = 0.0,
    target_diagnostics: Mapping[str, Any] | None = None,
) -> LocalTeacherLoss | None:
    try:
        pred_fit = _ensure_col(predict_fn(x_fit))
        pred_probe = _ensure_col(predict_fn(x_probe))
    except Exception:
        return None
    if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
        return None
    return _score_local_teacher_from_predictions(
        pred_fit=pred_fit,
        pred_probe=pred_probe,
        x_fit=x_fit,
        x_probe=x_probe,
        target_fit=target_fit,
        target_probe=target_probe,
        w_fit=w_fit,
        w_probe=w_probe,
        target_grad_fit=target_grad_fit,
        target_grad_probe=target_grad_probe,
        target_d2_fit=target_d2_fit,
        target_d2_probe=target_d2_probe,
        poly_degree=int(poly_degree),
        mode=str(mode),
        grad_weight=float(grad_weight),
        d2_weight=float(d2_weight),
        diag_weight=float(diag_weight),
        physics_weight=float(physics_weight),
        target_diagnostics=target_diagnostics,
        jet_provider=lambda x, capture_grad, capture_d2: _autograd_predict_jets(
            predict_fn,
            x,
            capture_grad=capture_grad,
            capture_d2=capture_d2,
        ),
    )


__all__ = ["LocalTeacherLoss", "score_local_teacher_loss", "score_local_teacher_prediction_loss"]
