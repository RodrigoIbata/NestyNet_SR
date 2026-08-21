# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

import torch

from .active_design import ExperimentCandidate, resolve_surface_disagreement_mode
from .committee import CommitteeState


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_project_mode(mode: str | None) -> str:
    token = str(mode or "nearest_box").strip().lower()
    if token in {"nearest_box", "clip_box", "clip"}:
        return "nearest_box"
    return "nearest_box"


def _primary_axis(x: torch.Tensor) -> int:
    if x.ndim != 2 or int(x.shape[1]) <= 1:
        return 0
    spread = x.to(dtype=torch.float64).std(dim=0, unbiased=False)
    if spread.numel() <= 0 or not torch.isfinite(spread).any():
        return 0
    return int(torch.argmax(torch.nan_to_num(spread, nan=0.0)).item())


def _ensure_matrix(x: Any, *, dtype: torch.dtype) -> torch.Tensor | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        out = x.to(dtype=dtype)
    else:
        try:
            out = torch.as_tensor(x, dtype=dtype)
        except Exception:
            return None
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    if out.ndim != 2 or int(out.shape[0]) <= 0 or int(out.shape[1]) <= 0:
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _normalize_forward_output(value: Any, *, x: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value.to(dtype=x.dtype, device=x.device)
    else:
        try:
            out = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    if out.ndim == 0:
        out = out.reshape(1, 1).expand(int(x.shape[0]), 1)
    elif out.ndim == 1:
        out = out.reshape(-1, 1)
    elif out.ndim > 2:
        try:
            out = out.reshape(int(out.shape[0]), -1).mean(dim=1, keepdim=True)
        except Exception:
            return None
    if out.ndim != 2 or int(out.shape[0]) != int(x.shape[0]):
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _autograd_value_grad(
    forward_fn: Callable[[torch.Tensor], Any],
    x: torch.Tensor,
    *,
    capture_gradients: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    xx = _ensure_matrix(x, dtype=x.dtype)
    if xx is None:
        return None, None
    try:
        with torch.enable_grad():
            x_req = xx if bool(xx.requires_grad) else xx.detach().clone().requires_grad_(True)
            value = _normalize_forward_output(forward_fn(x_req), x=x_req)
            if value is None:
                return None, None
            if not capture_gradients:
                return value, None
            grad = torch.autograd.grad(
                value.sum(),
                x_req,
                retain_graph=bool(capture_gradients),
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(x_req)
            return value, grad
    except Exception:
        return None, None


def _diagnostic_map(
    value: torch.Tensor,
    *,
    x: torch.Tensor,
    grad: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = {
        "value_mean": value.mean(),
        "value_std": value.std(unbiased=False) if int(value.numel()) > 1 else torch.zeros((), dtype=value.dtype, device=value.device),
    }
    if grad is not None and torch.is_tensor(grad):
        out["grad_norm_mean"] = torch.linalg.vector_norm(grad, dim=1).mean()
    axis = _primary_axis(x)
    coords = x[:, axis]
    order = torch.argsort(coords)
    coords_sorted = coords[order]
    values_sorted = value.reshape(-1)[order]
    scale = values_sorted.abs().mean() + torch.tensor(1.0e-6, dtype=value.dtype, device=value.device)
    if int(values_sorted.numel()) > 0:
        out["closest_zero_abs"] = values_sorted.abs().min()
    if int(values_sorted.numel()) > 1:
        values_rev = torch.flip(values_sorted, dims=[0])
        coords_rev = torch.flip(coords_sorted, dims=[0])
        center = 0.5 * (coords_sorted[0] + coords_sorted[-1])
        span = (coords_sorted[-1] - coords_sorted[0]).abs() + torch.tensor(1.0e-6, dtype=value.dtype, device=value.device)
        out["mirror_even_residual"] = (values_sorted - values_rev).abs().mean() / scale
        out["mirror_odd_residual"] = (values_sorted + values_rev).abs().mean() / scale
        out["mirror_axis_mismatch_mean"] = (coords_sorted + coords_rev - (2.0 * center)).abs().mean() / span
        prod = values_sorted[:-1] * values_sorted[1:]
        out["zero_crossing_soft_count"] = torch.sigmoid((-10.0 * prod) / ((scale * scale) + 1.0e-6)).mean()
        out["zero_soft_margin"] = torch.exp(-5.0 * values_sorted.abs() / scale).mean()
        dx = coords_sorted[1:] - coords_sorted[:-1]
        dy = values_sorted[1:] - values_sorted[:-1]
        slopes = dy / (dx + torch.sign(dx) * 1.0e-6 + (dx == 0.0).to(dtype=value.dtype) * 1.0e-6)
        if int(slopes.numel()) > 0:
            width = max(1, min(3, int(slopes.numel())))
            out["left_tail_slope_mean"] = slopes[:width].mean()
            out["right_tail_slope_mean"] = slopes[-width:].mean()
            out["tail_slope_gap_abs"] = (out["right_tail_slope_mean"] - out["left_tail_slope_mean"]).abs()
            if int(slopes.numel()) > 1:
                second = slopes[1:] - slopes[:-1]
                second_scale = second.abs().mean() + torch.tensor(1.0e-6, dtype=value.dtype, device=value.device)
                out["curvature_proxy_mean"] = second.abs().mean()
                if int(second.numel()) > 1:
                    out["convexity_soft_switch"] = torch.sigmoid(
                        (-10.0 * second[:-1] * second[1:]) / ((second_scale * second_scale) + 1.0e-6)
                    ).mean()
                else:
                    out["convexity_soft_switch"] = torch.zeros((), dtype=value.dtype, device=value.device)
    if grad is not None and torch.is_tensor(grad) and int(grad.shape[0]) == int(x.shape[0]):
        primary_grad = grad[order, axis]
        grad_scale = primary_grad.abs().mean() + torch.tensor(1.0e-6, dtype=value.dtype, device=value.device)
        out["monotonicity_abs_mean"] = primary_grad.abs().mean()
        if int(primary_grad.numel()) > 1:
            out["monotonicity_soft_switch"] = torch.sigmoid(
                (-10.0 * primary_grad[:-1] * primary_grad[1:]) / ((grad_scale * grad_scale) + 1.0e-6)
            ).mean()
    return out


def _prediction_tensor_map(value: Any) -> dict[str, torch.Tensor]:
    if value is None:
        return {}
    if torch.is_tensor(value):
        flat = value.reshape(-1)
        return {f"[{int(idx)}]": flat[idx] for idx in range(int(flat.numel()))}
    if isinstance(value, Mapping):
        out: dict[str, torch.Tensor] = {}
        for key in sorted(dict(value).keys(), key=str):
            prefix = str(key)
            child = _prediction_tensor_map(dict(value)[key])
            for child_key, child_value in child.items():
                key_name = f"{prefix}.{child_key}" if child_key else prefix
                out[key_name] = child_value
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out: dict[str, torch.Tensor] = {}
        for idx, item in enumerate(value):
            child = _prediction_tensor_map(item)
            for child_key, child_value in child.items():
                key_name = f"[{int(idx)}]{child_key}"
                out[key_name] = child_value
        return out
    scalar = _safe_float(value)
    if math.isfinite(scalar):
        return {"value": torch.tensor(float(scalar), dtype=torch.float64)}
    return {}


def _prediction_pairwise_distance_tensor(
    state: CommitteeState,
    prediction_map: Mapping[str, Any],
) -> torch.Tensor | None:
    total = None
    pair_weight_sum = 0.0
    members = list(state.members)
    for idx, member_i in enumerate(members):
        value_i = dict(prediction_map or {}).get(member_i.member_id, None)
        map_i = _prediction_tensor_map(value_i)
        for member_j in members[idx + 1 :]:
            value_j = dict(prediction_map or {}).get(member_j.member_id, None)
            map_j = _prediction_tensor_map(value_j)
            common = sorted(set(map_i.keys()) & set(map_j.keys()))
            if not common:
                continue
            diffs = [(map_i[key] - map_j[key]) ** 2 for key in common]
            if not diffs:
                continue
            distance = torch.stack(diffs).mean()
            pair_weight = float(member_i.committee_weight) * float(member_j.committee_weight)
            if pair_weight <= 0.0:
                continue
            total = distance * pair_weight if total is None else total + (distance * pair_weight)
            pair_weight_sum += pair_weight
    if total is None or pair_weight_sum <= 0.0:
        return None
    return total / pair_weight_sum


def differentiable_committee_disagreement(
    state: CommitteeState,
    *,
    observable_predictions: Mapping[str, Any],
    derivative_predictions: Mapping[str, Any],
    diagnostic_predictions: Mapping[str, Any],
    beta: float,
    gamma: float,
    disagreement_mode: str,
) -> torch.Tensor:
    resolve_surface_disagreement_mode(
        disagreement_mode,
        default_mode="witness",
    )
    observable = _prediction_pairwise_distance_tensor(state, observable_predictions)
    derivative = _prediction_pairwise_distance_tensor(state, derivative_predictions)
    diagnostic = _prediction_pairwise_distance_tensor(state, diagnostic_predictions)
    total = torch.zeros((), dtype=torch.float64)
    for tensor, weight in (
        (observable, 1.0),
        (derivative, float(beta)),
        (diagnostic, float(gamma)),
    ):
        if tensor is None:
            continue
        total = total + (float(weight) * tensor.to(dtype=torch.float64))
    return total


def _candidate_opt_state(candidate: ExperimentCandidate, *, dtype: torch.dtype) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    metadata = dict(candidate.metadata or {})
    state = dict(metadata.get("continuous_optimizer", {}) or {})
    x0 = _ensure_matrix(state.get("points", None), dtype=dtype)
    bounds = _ensure_matrix(state.get("bounds", None), dtype=dtype)
    return x0, bounds, state


def _project_points(x: torch.Tensor, bounds: torch.Tensor | None, *, project_mode: str) -> torch.Tensor:
    if bounds is None or str(project_mode) != "nearest_box":
        return x
    lo = bounds[:, 0].reshape(1, -1)
    hi = bounds[:, 1].reshape(1, -1)
    return torch.max(lo, torch.min(hi, x))


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    scalar = _safe_float(value)
    return None if not math.isfinite(scalar) else float(scalar)


def _evaluate_prediction_maps(
    x: torch.Tensor,
    *,
    state: CommitteeState,
    forward_fns_by_member_id: Mapping[str, Callable[[torch.Tensor], Any]],
    include_gradients: bool,
    include_diagnostics: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observable_predictions: dict[str, Any] = {}
    derivative_predictions: dict[str, Any] = {}
    diagnostic_predictions: dict[str, Any] = {}
    for member in state.members:
        forward_fn = dict(forward_fns_by_member_id or {}).get(str(member.member_id), None)
        if not callable(forward_fn):
            continue
        value, grad = _autograd_value_grad(forward_fn, x, capture_gradients=bool(include_gradients))
        if value is None:
            continue
        observable_predictions[str(member.member_id)] = value.squeeze(-1)
        if include_gradients and grad is not None:
            derivative_predictions[str(member.member_id)] = grad
        if include_diagnostics:
            diagnostic_predictions[str(member.member_id)] = _diagnostic_map(
                value,
                x=x,
                grad=grad if include_gradients else None,
            )
    return observable_predictions, derivative_predictions, diagnostic_predictions


def optimize_continuous_experiment_candidates(
    state: CommitteeState,
    candidates: Sequence[ExperimentCandidate],
    *,
    forward_fns_by_member_id: Mapping[str, Callable[[torch.Tensor], Any]],
    beta: float,
    gamma: float,
    disagreement_mode: str,
    lambda_cost: float,
    lambda_noise: float,
    lambda_feasibility: float,
    opt_steps: int,
    opt_lr: float,
    project_mode: str,
    include_gradients: bool,
    include_diagnostics: bool,
) -> dict[str, Any]:
    optimized_candidates: list[ExperimentCandidate] = []
    summaries: list[dict[str, Any]] = []
    mode_name = _normalize_project_mode(project_mode)
    total_improvement = 0.0
    optimized_count = 0
    for candidate in list(candidates or []):
        x0, bounds, state_payload = _candidate_opt_state(candidate, dtype=torch.float64)
        if x0 is None:
            optimized_candidates.append(candidate)
            summaries.append(
                {
                    "experiment_id": str(candidate.experiment_id),
                    "optimized": False,
                    "status": "missing_optimizer_points",
                }
            )
            continue
        raw = x0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=float(opt_lr))
        best_score = None
        best_x = x0.detach().clone()
        steps_executed = 0
        for step in range(max(1, int(opt_steps))):
            steps_executed = step + 1
            opt.zero_grad(set_to_none=True)
            x_proj = _project_points(raw, bounds, project_mode=mode_name)
            observable, derivative, diagnostic = _evaluate_prediction_maps(
                x_proj,
                state=state,
                forward_fns_by_member_id=forward_fns_by_member_id,
                include_gradients=bool(include_gradients),
                include_diagnostics=bool(include_diagnostics),
            )
            disagreement = differentiable_committee_disagreement(
                state,
                observable_predictions=observable,
                derivative_predictions=derivative,
                diagnostic_predictions=diagnostic,
                beta=float(beta),
                gamma=float(gamma),
                disagreement_mode=disagreement_mode,
            )
            objective = disagreement.to(dtype=torch.float64)
            objective = objective - (
                float(lambda_cost) * float(candidate.cost)
                + float(lambda_noise) * float(candidate.noise_risk)
                + float(lambda_feasibility) * float(candidate.feasibility_penalty)
            )
            loss = -objective
            loss.backward()
            opt.step()
            score_value = _safe_float(objective.detach().item())
            if best_score is None or score_value > best_score:
                best_score = float(score_value)
                best_x = _project_points(raw.detach().clone(), bounds, project_mode=mode_name)
        initial_observable, initial_derivative, initial_diagnostic = _evaluate_prediction_maps(
            x0,
            state=state,
            forward_fns_by_member_id=forward_fns_by_member_id,
            include_gradients=bool(include_gradients),
            include_diagnostics=bool(include_diagnostics),
        )
        initial_score = differentiable_committee_disagreement(
            state,
            observable_predictions=initial_observable,
            derivative_predictions=initial_derivative,
            diagnostic_predictions=initial_diagnostic,
            beta=float(beta),
            gamma=float(gamma),
            disagreement_mode=disagreement_mode,
        )
        initial_score_value = _safe_float(
            (
                initial_score
                - (
                    float(lambda_cost) * float(candidate.cost)
                    + float(lambda_noise) * float(candidate.noise_risk)
                    + float(lambda_feasibility) * float(candidate.feasibility_penalty)
                )
            ).detach().item()
        )
        final_observable, final_derivative, final_diagnostic = _evaluate_prediction_maps(
            best_x,
            state=state,
            forward_fns_by_member_id=forward_fns_by_member_id,
            include_gradients=bool(include_gradients),
            include_diagnostics=bool(include_diagnostics),
        )
        final_score = differentiable_committee_disagreement(
            state,
            observable_predictions=final_observable,
            derivative_predictions=final_derivative,
            diagnostic_predictions=final_diagnostic,
            beta=float(beta),
            gamma=float(gamma),
            disagreement_mode=disagreement_mode,
        )
        final_score_value = _safe_float(
            (
                final_score
                - (
                    float(lambda_cost) * float(candidate.cost)
                    + float(lambda_noise) * float(candidate.noise_risk)
                    + float(lambda_feasibility) * float(candidate.feasibility_penalty)
                )
            ).detach().item()
        )
        total_improvement += max(0.0, float(final_score_value - initial_score_value))
        if float(final_score_value) > float(initial_score_value) + 1.0e-9:
            optimized_count += 1
        optimization_summary = {
            "enabled": True,
            "optimized": bool(float(final_score_value) > float(initial_score_value) + 1.0e-9),
            "status": "ok",
            "project_mode": str(mode_name),
            "steps_requested": int(max(1, int(opt_steps))),
            "steps_executed": int(steps_executed),
            "score_before": float(initial_score_value),
            "score_after": float(final_score_value),
            "score_improvement": float(final_score_value - initial_score_value),
            "condition_delta_l2": float(torch.linalg.vector_norm(best_x - x0).item()),
            "optimized_points_preview": _jsonable(best_x[: min(4, int(best_x.shape[0]))]),
        }
        new_metadata = dict(candidate.metadata or {})
        new_metadata["continuous_optimization"] = optimization_summary
        optimized_candidates.append(
            replace(
                candidate,
                conditions={
                    **dict(candidate.conditions or {}),
                    "optimized": bool(optimization_summary["optimized"]),
                    "project_mode": str(mode_name),
                    "optimized_shape": [int(v) for v in best_x.shape],
                },
                observable_predictions=observable_predictions_to_jsonable(final_observable),
                derivative_predictions=observable_predictions_to_jsonable(final_derivative),
                diagnostic_predictions=observable_predictions_to_jsonable(final_diagnostic),
                metadata=new_metadata,
            )
        )
        summaries.append({"experiment_id": str(candidate.experiment_id), **optimization_summary})
    return {
        "candidates": optimized_candidates,
        "summary": {
            "enabled": True,
            "project_mode": str(mode_name),
            "opt_steps": int(max(1, int(opt_steps))),
            "opt_lr": float(opt_lr),
            "candidate_count": int(len(list(candidates or []))),
            "optimized_candidate_count": int(optimized_count),
            "total_score_improvement": float(total_improvement),
            "candidates": summaries,
        },
    }


def observable_predictions_to_jsonable(predictions: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in dict(predictions or {}).items()}


__all__ = [
    "differentiable_committee_disagreement",
    "observable_predictions_to_jsonable",
    "optimize_continuous_experiment_candidates",
]
