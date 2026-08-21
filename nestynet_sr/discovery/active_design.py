# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .committee import CommitteeState


@dataclass(frozen=True)
class ExperimentCandidate:
    experiment_id: str
    conditions: Any = None
    observable_predictions: Mapping[str, Any] = field(default_factory=dict)
    derivative_predictions: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_predictions: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    noise_risk: float = 0.0
    feasibility_penalty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _prediction_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            flat = value.detach().cpu().reshape(-1).tolist()
        except Exception:
            flat = [_safe_float(value)]
        return [float(item) for item in flat if math.isfinite(_safe_float(item))]
    if isinstance(value, Mapping):
        out: list[float] = []
        for key in sorted(dict(value).keys(), key=str):
            out.extend(_prediction_vector(dict(value)[key]))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out: list[float] = []
        for item in value:
            out.extend(_prediction_vector(item))
        return out
    scalar = _safe_float(value)
    return [] if not math.isfinite(scalar) else [float(scalar)]


def _prediction_scalar_map(value: Any, *, prefix: str = "") -> dict[str, float]:
    if value is None:
        return {}
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            flat = value.detach().cpu().reshape(-1).tolist()
        except Exception:
            flat = [_safe_float(value)]
        out: dict[str, float] = {}
        for idx, item in enumerate(flat):
            scalar = _safe_float(item)
            if math.isfinite(scalar):
                key = f"{prefix}[{int(idx)}]" if prefix else f"[{int(idx)}]"
                out[key] = float(scalar)
        return out
    if isinstance(value, Mapping):
        out: dict[str, float] = {}
        for key in sorted(dict(value).keys(), key=str):
            child_prefix = f"{prefix}.{str(key)}" if prefix else str(key)
            out.update(_prediction_scalar_map(dict(value)[key], prefix=child_prefix))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out: dict[str, float] = {}
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{int(idx)}]" if prefix else f"[{int(idx)}]"
            out.update(_prediction_scalar_map(item, prefix=child_prefix))
        return out
    scalar = _safe_float(value)
    return {} if not math.isfinite(scalar) else {prefix or "value": float(scalar)}


def _prediction_distance(value_a: Any, value_b: Any) -> float | None:
    if isinstance(value_a, Mapping) or isinstance(value_b, Mapping):
        map_a = _prediction_scalar_map(value_a)
        map_b = _prediction_scalar_map(value_b)
        common = sorted(set(map_a.keys()) & set(map_b.keys()))
        if not common:
            return None
        diffs = [(float(map_a[key]) - float(map_b[key])) ** 2 for key in common]
        return float(sum(diffs) / len(diffs))
    vec_a = _prediction_vector(value_a)
    vec_b = _prediction_vector(value_b)
    if not vec_a or not vec_b:
        return None
    if len(vec_a) == len(vec_b):
        diffs = [(float(a) - float(b)) ** 2 for a, b in zip(vec_a, vec_b)]
        return float(sum(diffs) / len(diffs))
    mean_a = float(sum(vec_a) / len(vec_a))
    mean_b = float(sum(vec_b) / len(vec_b))
    return float((mean_a - mean_b) ** 2)


def _prediction_pairwise_distance(
    state: CommitteeState,
    prediction_map: Mapping[str, Any],
) -> float:
    total = 0.0
    pair_weight_sum = 0.0
    members = list(state.members)
    for idx, member_i in enumerate(members):
        value_i = dict(prediction_map or {}).get(member_i.member_id, None)
        for member_j in members[idx + 1 :]:
            value_j = dict(prediction_map or {}).get(member_j.member_id, None)
            distance = _prediction_distance(value_i, value_j)
            if distance is None or not math.isfinite(float(distance)):
                continue
            pair_weight = float(member_i.committee_weight) * float(member_j.committee_weight)
            if pair_weight <= 0.0:
                continue
            total += pair_weight * float(distance)
            pair_weight_sum += pair_weight
    if pair_weight_sum <= 0.0:
        return 0.0
    return float(total / pair_weight_sum)


def _normalize_disagreement_mode(mode: str | None) -> str:
    token = str(mode or "witness").strip().lower()
    if token in {"witness", "scalar"}:
        return "witness"
    raise ValueError(f"unsupported disagreement_mode {mode!r}; expected 'auto' or 'witness'")


def resolve_disagreement_mode(
    disagreement_mode: str | None,
    *,
    default_mode: str = "witness",
) -> str:
    token = str(disagreement_mode or "").strip().lower()
    if token in {"", "auto", "default"}:
        return _normalize_disagreement_mode(default_mode)
    return _normalize_disagreement_mode(token)


def resolve_surface_disagreement_mode(
    disagreement_mode: str | None,
    *,
    default_mode: str = "witness",
) -> str:
    return resolve_disagreement_mode(
        disagreement_mode,
        default_mode=default_mode,
    )


def committee_disagreement(
    state: CommitteeState,
    candidate: ExperimentCandidate,
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    disagreement_mode: str | None = None,
) -> dict[str, Any]:
    return committee_disagreement_with_mode(
        state,
        candidate,
        beta=float(beta),
        gamma=float(gamma),
        disagreement_mode=disagreement_mode,
    )


def committee_disagreement_with_mode(
    state: CommitteeState,
    candidate: ExperimentCandidate,
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    disagreement_mode: str | None = None,
) -> dict[str, Any]:
    mode_name = resolve_disagreement_mode(disagreement_mode)
    observable_component = _prediction_pairwise_distance(state, candidate.observable_predictions)
    derivative_component = _prediction_pairwise_distance(state, candidate.derivative_predictions)
    diagnostic_component = _prediction_pairwise_distance(state, candidate.diagnostic_predictions)
    total = (
        float(observable_component)
        + float(beta) * float(derivative_component)
        + float(gamma) * float(diagnostic_component)
    )
    return {
        "mode": str(mode_name),
        "observable_component": float(observable_component),
        "derivative_component": float(derivative_component),
        "diagnostic_component": float(diagnostic_component),
        "observable_distance": float(observable_component),
        "derivative_distance": float(derivative_component),
        "diagnostic_distance": float(diagnostic_component),
        "observable_variance": float(observable_component),
        "derivative_variance": float(derivative_component),
        "diagnostic_variance": float(diagnostic_component),
        "total_disagreement": float(total),
    }


def score_experiment_candidate(
    state: CommitteeState,
    candidate: ExperimentCandidate,
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    disagreement_mode: str | None = None,
    lambda_cost: float = 1.0,
    lambda_noise: float = 1.0,
    lambda_feasibility: float = 1.0,
) -> dict[str, Any]:
    mode_name = resolve_disagreement_mode(disagreement_mode)
    disagreement = committee_disagreement_with_mode(
        state,
        candidate,
        beta=float(beta),
        gamma=float(gamma),
        disagreement_mode=mode_name,
    )
    score = (
        float(disagreement["total_disagreement"])
        - float(lambda_cost) * float(candidate.cost)
        - float(lambda_noise) * float(candidate.noise_risk)
        - float(lambda_feasibility) * float(candidate.feasibility_penalty)
    )
    return {
        "experiment_id": str(candidate.experiment_id),
        "score": float(score),
        "disagreement": disagreement,
        "disagreement_mode": str(disagreement.get("mode", mode_name)),
        "conditions": candidate.conditions,
        "cost": float(candidate.cost),
        "noise_risk": float(candidate.noise_risk),
        "feasibility_penalty": float(candidate.feasibility_penalty),
    }


def select_next_experiment(
    state: CommitteeState,
    candidates: Sequence[ExperimentCandidate],
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    disagreement_mode: str | None = None,
    lambda_cost: float = 1.0,
    lambda_noise: float = 1.0,
    lambda_feasibility: float = 1.0,
    optimize_continuous: bool = False,
    experiment_optimizer=None,
) -> dict[str, Any]:
    mode_name = resolve_disagreement_mode(disagreement_mode)
    working_candidates = list(candidates or [])
    optimization_summary: dict[str, Any] = {"enabled": bool(optimize_continuous), "applied": False}
    if bool(optimize_continuous):
        if callable(experiment_optimizer):
            try:
                optimized = experiment_optimizer(
                    state,
                    working_candidates,
                    beta=float(beta),
                    gamma=float(gamma),
                    disagreement_mode=mode_name,
                    lambda_cost=float(lambda_cost),
                    lambda_noise=float(lambda_noise),
                    lambda_feasibility=float(lambda_feasibility),
                )
            except Exception as exc:
                optimization_summary = {
                    "enabled": True,
                    "applied": False,
                    "status": f"optimizer_error:{type(exc).__name__}",
                }
            else:
                if isinstance(optimized, Mapping):
                    rows = optimized.get("candidates", None)
                    if isinstance(rows, Sequence):
                        working_candidates = list(rows)
                    optimization_summary = {
                        "enabled": True,
                        "applied": bool(rows is not None),
                        **dict(optimized.get("summary", {}) or {}),
                    }
                else:
                    optimization_summary = {
                        "enabled": True,
                        "applied": False,
                        "status": "invalid_optimizer_result",
                    }
        else:
            optimization_summary = {
                "enabled": True,
                "applied": False,
                "status": "optimizer_unavailable",
            }
    scored = [
        score_experiment_candidate(
            state,
            candidate,
            beta=float(beta),
            gamma=float(gamma),
            disagreement_mode=mode_name,
            lambda_cost=float(lambda_cost),
            lambda_noise=float(lambda_noise),
            lambda_feasibility=float(lambda_feasibility),
        )
        for candidate in working_candidates
    ]
    scored.sort(key=lambda row: (-float(row["score"]), str(row["experiment_id"])))
    selected = None if not scored else dict(scored[0])
    return {
        "selected": selected,
        "ranking": scored,
        "disagreement_mode": mode_name,
        "optimization": optimization_summary,
    }


__all__ = [
    "ExperimentCandidate",
    "committee_disagreement",
    "resolve_disagreement_mode",
    "resolve_surface_disagreement_mode",
    "score_experiment_candidate",
    "select_next_experiment",
]
