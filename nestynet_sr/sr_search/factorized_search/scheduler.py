# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .scheduler_critic import predict_scheduler_plan_slate
from .scheduler_dataset import normalize_scheduler_budget_ladder, scheduler_threshold_token
from .shared_opportunity import shared_opportunity_row_dict


@dataclass(frozen=True)
class PlanCandidate:
    route: str
    method: str
    decision_id: str
    opportunity_id: str
    parent_key: str
    action: str = ""
    path: tuple[int, ...] = ()
    target_mode: str = ""
    exact_budget: int = 1
    widen_budget: int = 0
    micro_budget: int = 0
    features: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanDecision:
    trained: bool
    advisory_only: bool
    candidate_count: int
    fallback_used: bool = False
    fallback_reason: str = ""
    confidence: float = 0.0
    confidence_kind: str = "dominance_gap"
    dominance_prob: float = 0.0
    acquisition_gap: float = 0.0
    acquisition_gap_sigma: float = 0.0
    objective_mode: str = "acquisition"
    objective_gap: float = 0.0
    objective_gap_sigma: float = 0.0
    chosen_candidate: PlanCandidate | None = None
    chosen_route: str = ""
    chosen_opportunity_id: str = ""
    chosen_exact_budget: int = 0
    runner_up_route: str = ""
    runner_up_opportunity_id: str = ""
    runner_up_exact_budget: int = 0
    acquisition_threshold: float = 0.25
    uncertainty_bonus: float = 0.05
    acquisition_weights: dict[str, float] = field(default_factory=dict)
    route_scores: dict[str, float] = field(default_factory=dict)
    rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _witness_energy_fields(row: Mapping[str, Any]) -> dict[str, float | None]:
    value_loss = _safe_float_or_none(row.get("witness_value_loss", None))
    grad_loss = _safe_float_or_none(row.get("witness_grad_loss", None))
    d2_loss = _safe_float_or_none(row.get("witness_d2_loss", None))
    diag_loss = _safe_float_or_none(row.get("witness_diag_loss", None))
    physics_loss = _safe_float_or_none(row.get("witness_physics_loss", None))
    total = _safe_float_or_none(row.get("witness_energy_total", None))
    if total is None:
        parts = [value_loss, grad_loss, d2_loss, diag_loss, physics_loss]
        finite_parts = [float(v) for v in parts if v is not None]
        if finite_parts:
            total = float(sum(finite_parts))
    return {
        "value_loss": value_loss,
        "grad_loss": grad_loss,
        "d2_loss": d2_loss,
        "diag_loss": diag_loss,
        "physics_loss": physics_loss,
        "total": total,
        "delta_estimate": _safe_float_or_none(row.get("witness_energy_delta_estimate", None)),
    }


def _resolved_acquisition_weights(
    bundle: Mapping[str, Any],
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    weights = {
        "break": 1.0,
        "tail": 0.5,
        "route_win": 0.3,
        "new_residual_basin": 0.2,
        "stable": 0.1,
        "fragile": 0.15,
        "cost_exact": 0.05,
        "cost_wall": 0.05,
        "uncertainty": 0.05,
    }
    if isinstance(bundle, Mapping):
        weights.update({
            str(key): float(value)
            for key, value in dict(bundle.get("acquisition_weights", {}) or {}).items()
        })
    if overrides is not None:
        weights.update({
            str(key): float(value)
            for key, value in dict(overrides).items()
        })
    return weights


def _plan_prediction_components(
    pred_row: Mapping[str, Any],
    *,
    budget: int,
    threshold_value: float,
    threshold_token: str,
    weights: Mapping[str, float],
    uncertainty_bonus: float,
) -> dict[str, Any]:
    break_prob = _safe_float(pred_row.get(f"break_prob_{threshold_token}_at_budget_{int(budget)}", 0.0), 0.0)
    tail_gain = _safe_float(pred_row.get(f"tail_gain_{threshold_token}_pred_at_budget_{int(budget)}", 0.0), 0.0)
    route_win_prob = _safe_float(pred_row.get(f"route_win_prob_at_budget_{int(budget)}", 0.0), 0.0)
    new_residual_basin_prob = _safe_float(pred_row.get(f"new_residual_basin_prob_at_budget_{int(budget)}", 0.0), 0.0)
    stable_prob = _safe_float(pred_row.get(f"stable_prob_at_budget_{int(budget)}", 0.0), 0.0)
    fragile_prob = _safe_float(pred_row.get(f"fragile_prob_at_budget_{int(budget)}", 0.0), 0.0)
    cost_exact_pred = _safe_float(pred_row.get(f"cost_exact_pred_at_budget_{int(budget)}", 0.0), 0.0)
    cost_wall_pred = _safe_float(pred_row.get(f"cost_wall_pred_at_budget_{int(budget)}", 0.0), 0.0)
    sigma = _safe_float(pred_row.get(f"acquisition_sigma_at_budget_{int(budget)}", 0.0), 0.0)
    contributions = {
        "break": float(weights.get("break", 1.0)) * break_prob,
        "tail": float(weights.get("tail", 0.5)) * tail_gain,
        "route_win": float(weights.get("route_win", 0.3)) * route_win_prob,
        "new_residual_basin": float(weights.get("new_residual_basin", 0.2)) * new_residual_basin_prob,
        "stable": float(weights.get("stable", 0.1)) * stable_prob,
        "fragile": -float(weights.get("fragile", 0.15)) * fragile_prob,
        "cost_exact": -float(weights.get("cost_exact", 0.05)) * cost_exact_pred,
        "cost_wall": -float(weights.get("cost_wall", 0.05)) * cost_wall_pred,
        "uncertainty": float(uncertainty_bonus) * sigma,
    }
    witness_energy = _witness_energy_fields(pred_row)
    return {
        "threshold": float(threshold_value),
        "threshold_token": str(threshold_token),
        "weights": {str(key): float(value) for key, value in dict(weights).items()},
        "raw": {
            "break_prob": float(break_prob),
            "tail_gain": float(tail_gain),
            "route_win_prob": float(route_win_prob),
            "new_residual_basin_prob": float(new_residual_basin_prob),
            "stable_prob": float(stable_prob),
            "fragile_prob": float(fragile_prob),
            "cost_exact_pred": float(cost_exact_pred),
            "cost_wall_pred": float(cost_wall_pred),
            "sigma": float(sigma),
        },
        "contributions": {
            str(key): float(value)
            for key, value in contributions.items()
        },
        "witness_energy": witness_energy,
        "objective": {
            "mode": str(pred_row.get("objective_mode", "acquisition") or "acquisition"),
            "estimate": _safe_float(
                pred_row.get(
                    f"objective_estimate_at_budget_{int(budget)}",
                    pred_row.get(f"acquisition_estimate_at_budget_{int(budget)}", 0.0),
                ),
                0.0,
            ),
            "sigma": _safe_float(
                pred_row.get(
                    f"objective_sigma_at_budget_{int(budget)}",
                    pred_row.get(f"acquisition_sigma_at_budget_{int(budget)}", 0.0),
                ),
                0.0,
            ),
            "with_uncertainty": _safe_float(
                pred_row.get(
                    f"objective_estimate_at_budget_{int(budget)}",
                    pred_row.get(f"acquisition_estimate_at_budget_{int(budget)}", 0.0),
                ),
                0.0,
            ) + (
                float(uncertainty_bonus)
                * _safe_float(
                    pred_row.get(
                        f"objective_sigma_at_budget_{int(budget)}",
                        pred_row.get(f"acquisition_sigma_at_budget_{int(budget)}", 0.0),
                    ),
                    0.0,
                )
            ),
            "witness_delta_pred": _safe_float_or_none(pred_row.get(f"witness_delta_pred_at_budget_{int(budget)}", None)),
            "witness_rate_pred": _safe_float_or_none(pred_row.get(f"witness_rate_pred_at_budget_{int(budget)}", None)),
            "cost_total_pred": _safe_float_or_none(pred_row.get(f"cost_total_pred_at_budget_{int(budget)}", None)),
        },
        "acquisition_estimate": _safe_float(
            pred_row.get(f"acquisition_estimate_at_budget_{int(budget)}", 0.0),
            0.0,
        ),
        "acquisition_with_uncertainty": _safe_float(
            pred_row.get(f"acquisition_estimate_at_budget_{int(budget)}", 0.0),
            0.0,
        ) + float(contributions["uncertainty"]),
    }


def _dominance_prob_from_gap(gap: float, sigma: float) -> float:
    if not math.isfinite(float(gap)):
        return 1.0
    sigma_f = max(0.0, float(sigma))
    if sigma_f <= 1.0e-12:
        return 1.0 if float(gap) > 0.0 else (0.5 if abs(float(gap)) <= 1.0e-12 else 0.0)
    z_score = float(gap) / sigma_f
    return float(0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))))


def _normalized_dominance_confidence(prob: float) -> float:
    return float(min(1.0, max(0.0, 2.0 * (float(prob) - 0.5))))


def _base_candidate_row(
    row: Mapping[str, Any],
    *,
    route_source: str,
    parent_key: str = "",
    decision_context_id: str = "",
    current_best_route_eff_mse: float | None = None,
) -> dict[str, Any]:
    row_out = shared_opportunity_row_dict(row, route_source=route_source)
    if parent_key and not str(row_out.get("parent_key", "") or ""):
        row_out["parent_key"] = str(parent_key)
    route_decision_id = str(row_out.get("decision_id", "") or "").strip()
    route_decision_context_id = str(
        row_out.get("decision_context_id", "")
        or row_out.get("scheduler_decision_context_id", "")
        or route_decision_id
        or ""
    ).strip()
    if route_decision_id and not str(row_out.get("route_decision_id", "") or "").strip():
        row_out["route_decision_id"] = str(route_decision_id)
    if route_decision_context_id and not str(row_out.get("route_decision_context_id", "") or "").strip():
        row_out["route_decision_context_id"] = str(route_decision_context_id)
    if decision_context_id:
        scheduler_context = str(decision_context_id)
        row_out["decision_context_id"] = scheduler_context
        row_out["scheduler_decision_context_id"] = scheduler_context
    elif route_decision_context_id and not str(row_out.get("scheduler_decision_context_id", "") or "").strip():
        row_out["scheduler_decision_context_id"] = str(route_decision_context_id)
    if current_best_route_eff_mse is not None and row_out.get("current_best_route_eff_mse", None) is None:
        row_out["current_best_route_eff_mse"] = float(current_best_route_eff_mse)
    if current_best_route_eff_mse is not None and row_out.get("global_best_eff_mse", None) is None:
        row_out["global_best_eff_mse"] = float(current_best_route_eff_mse)
    if not str(row_out.get("method_name", "") or ""):
        row_out["method_name"] = str(route_source)
    return row_out


def _row_budget_cap(
    row: Mapping[str, Any],
    *,
    route_source: str,
    default_ladder: Sequence[int],
    explicit_cap: int | None = None,
) -> int:
    if route_source == "build":
        return 1
    budget_remaining = _safe_int(row.get("budget_remaining", 0), 0)
    shadow_total = _safe_int(row.get("shadow_total_exact_available", 0), 0)
    preview_total = _safe_int(row.get("preview_candidate_count_total", 0), 0)
    cap = max(budget_remaining, shadow_total, preview_total, max(int(v) for v in list(default_ladder or (1,))))
    if explicit_cap is not None:
        cap = min(cap, max(1, int(explicit_cap)))
    return max(1, int(cap))


def _budget_choices(
    row: Mapping[str, Any],
    *,
    route_source: str,
    exact_budget_ladder: Sequence[int],
    exact_budget_cap: int | None = None,
) -> list[int]:
    budgets = [int(v) for v in normalize_scheduler_budget_ladder(exact_budget_ladder)]
    if route_source == "build":
        return [1]
    cap = _row_budget_cap(
        row,
        route_source=route_source,
        default_ladder=budgets,
        explicit_cap=exact_budget_cap,
    )
    selected = [int(budget) for budget in budgets if int(budget) <= int(cap)]
    return selected or [min(budgets or [1])]


def build_plan_candidates(
    *,
    parent_key: str,
    decision_context_id: str = "",
    current_best_route_eff_mse: float | None = None,
    build_opportunity_rows: Sequence[Mapping[str, Any]] | None = None,
    repair_opportunity_rows: Sequence[Mapping[str, Any]] | None = None,
    hole_opportunity_rows: Sequence[Mapping[str, Any]] | None = None,
    exact_budget_ladder: Sequence[Any] | None = None,
    hole_exact_budget_cap: int | None = None,
) -> list[PlanCandidate]:
    budgets = normalize_scheduler_budget_ladder(exact_budget_ladder)
    out: list[PlanCandidate] = []
    seen: set[tuple[str, str, int]] = set()
    for route_source, rows, cap in (
        ("build", list(build_opportunity_rows or []), 1),
        ("repair", list(repair_opportunity_rows or []), None),
        ("hole", list(hole_opportunity_rows or []), hole_exact_budget_cap),
    ):
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                continue
            row = _base_candidate_row(
                raw_row,
                route_source=route_source,
                parent_key=parent_key,
                decision_context_id=decision_context_id,
                current_best_route_eff_mse=current_best_route_eff_mse,
            )
            route_budgets = _budget_choices(
                row,
                route_source=route_source,
                exact_budget_ladder=budgets,
                exact_budget_cap=cap,
            )
            for budget in route_budgets:
                key = (
                    str(route_source),
                    str(row.get("opportunity_id", "") or ""),
                    int(budget),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    PlanCandidate(
                        route=str(route_source),
                        method=str(row.get("method_name", route_source) or route_source),
                        decision_id=str(
                            row.get("scheduler_decision_context_id", "")
                            or row.get("decision_context_id", "")
                            or row.get("decision_id", "")
                            or decision_context_id
                        ),
                        opportunity_id=str(row.get("opportunity_id", "") or ""),
                        parent_key=str(row.get("parent_key", parent_key) or parent_key),
                        action=str(row.get("action", "") or ""),
                        path=tuple(int(v) for v in (row.get("path", ()) or ())),
                        target_mode=str(row.get("target_mode", "") or ""),
                        exact_budget=int(budget),
                        features=dict(row),
                        payload={"row": dict(row)},
                    )
                )
    return out


def _row_scheduler_context_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("scheduler_decision_context_id", "")
        or row.get("decision_context_id", "")
        or row.get("decision_id", "")
        or ""
    ).strip()


def build_plan_candidates_from_rows(
    opportunity_rows: Sequence[Mapping[str, Any]],
    *,
    exact_budget_ladder: Sequence[Any] | None = None,
    hole_exact_budget_cap: int | None = None,
    parent_key: str = "",
    decision_context_id: str = "",
    current_best_route_eff_mse: float | None = None,
) -> list[PlanCandidate]:
    rows = [dict(row) for row in list(opportunity_rows or []) if isinstance(row, Mapping)]
    if not rows:
        return []
    parent_key_out = str(parent_key or "")
    if not parent_key_out:
        for row in rows:
            token = str(row.get("parent_key", "") or "")
            if token:
                parent_key_out = token
                break
    decision_context_out = str(decision_context_id or "")
    if not decision_context_out:
        for row in rows:
            token = _row_scheduler_context_id(row)
            if token:
                decision_context_out = token
                break
    current_best_out = current_best_route_eff_mse
    if current_best_out is None:
        current_candidates = [
            _safe_float_or_none(
                row.get(
                    "current_best_route_eff_mse",
                    row.get("global_best_eff_mse", row.get("parent_eff_mse", None)),
                )
            )
            for row in rows
        ]
        finite = [float(value) for value in current_candidates if value is not None]
        current_best_out = min(finite) if finite else None
    build_rows = [row for row in rows if str(row.get("route_source", "") or "") == "build"]
    repair_rows = [row for row in rows if str(row.get("route_source", "") or "") == "repair"]
    hole_rows = [row for row in rows if str(row.get("route_source", "") or "") == "hole"]
    return build_plan_candidates(
        parent_key=parent_key_out,
        decision_context_id=decision_context_out,
        current_best_route_eff_mse=current_best_out,
        build_opportunity_rows=build_rows,
        repair_opportunity_rows=repair_rows,
        hole_opportunity_rows=hole_rows,
        exact_budget_ladder=exact_budget_ladder,
        hole_exact_budget_cap=hole_exact_budget_cap,
    )


def predict_plan_slate(
    bundle: Mapping[str, Any],
    plan_candidates: Sequence[PlanCandidate],
    *,
    acquisition_threshold: float = 0.25,
    uncertainty_bonus: float = 0.05,
    acquisition_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "rows": [],
        "route_scores": {},
        "candidate_count": int(len(list(plan_candidates or []))),
        "objective_mode": str(bundle.get("objective_mode", "acquisition") or "acquisition") if isinstance(bundle, Mapping) else "acquisition",
    }
    candidates = [candidate for candidate in list(plan_candidates or []) if isinstance(candidate, PlanCandidate)]
    if not candidates:
        return out
    base_rows: list[dict[str, Any]] = []
    row_key_to_index: dict[tuple[str, str], int] = {}
    for candidate in candidates:
        row = dict(candidate.payload.get("row", candidate.features) or {})
        row_key = (str(candidate.route), str(candidate.opportunity_id))
        if row_key in row_key_to_index:
            continue
        row_key_to_index[row_key] = int(len(base_rows))
        base_rows.append(shared_opportunity_row_dict(row, route_source=candidate.route))
    pred_kwargs = {
        "acquisition_threshold": float(acquisition_threshold),
    }
    resolved_weights = _resolved_acquisition_weights(bundle, acquisition_weights)
    if acquisition_weights is not None or bool(dict(bundle.get("acquisition_weights", {}) or {})):
        pred_kwargs["acquisition_weights"] = resolved_weights
    pred = predict_scheduler_plan_slate(
        bundle,
        base_rows,
        **pred_kwargs,
    )
    if not bool(pred.get("trained", False)):
        return out
    threshold_token = scheduler_threshold_token(float(acquisition_threshold))
    threshold_value = float(acquisition_threshold)
    pred_rows = [
        dict(row)
        for row in list(pred.get("rows", []) or [])
        if isinstance(row, Mapping)
    ]
    objective_mode = str(pred.get("objective_mode", bundle.get("objective_mode", "acquisition")) or "acquisition")
    pred_map = {
        (
            str(row.get("route_source", "") or ""),
            str(row.get("opportunity_id", "") or ""),
        ): dict(row)
        for row in pred_rows
    }
    rows_out: list[dict[str, Any]] = []
    route_scores: dict[str, float] = {}
    for idx, candidate in enumerate(candidates):
        pred_row = pred_map.get((str(candidate.route), str(candidate.opportunity_id)))
        if pred_row is None:
            continue
        budget = int(candidate.exact_budget)
        acq = _safe_float(
            pred_row.get(f"acquisition_estimate_at_budget_{budget}", float("-inf")),
            float("-inf"),
        )
        sigma = _safe_float(pred_row.get(f"acquisition_sigma_at_budget_{budget}", 0.0), 0.0)
        objective_estimate = _safe_float(
            pred_row.get(
                f"objective_estimate_at_budget_{budget}",
                pred_row.get(f"acquisition_estimate_at_budget_{budget}", float("-inf")),
            ),
            float("-inf"),
        )
        objective_sigma = _safe_float(
            pred_row.get(
                f"objective_sigma_at_budget_{budget}",
                pred_row.get(f"acquisition_sigma_at_budget_{budget}", 0.0),
            ),
            0.0,
        )
        break_prob = _safe_float(pred_row.get(f"break_prob_{threshold_token}_at_budget_{budget}", 0.0), 0.0)
        prediction_components = _plan_prediction_components(
            pred_row,
            budget=budget,
            threshold_value=threshold_value,
            threshold_token=threshold_token,
            weights=resolved_weights,
            uncertainty_bonus=float(uncertainty_bonus),
        )
        row_out = dict(pred_row)
        row_out.update({
            "plan_candidate_index": int(idx),
            "plan_route": str(candidate.route),
            "plan_method": str(candidate.method),
            "plan_exact_budget": int(budget),
            "plan_action": str(candidate.action),
            "plan_path": [int(v) for v in tuple(candidate.path)],
            "plan_target_mode": str(candidate.target_mode),
            "plan_breakthrough_prob": float(break_prob),
            "plan_confidence": float(break_prob),
            "plan_acquisition_estimate": float(acq),
            "plan_acquisition_sigma": float(sigma),
            "plan_acquisition_with_uncertainty": float(acq + float(uncertainty_bonus) * sigma),
            "plan_objective_mode": str(objective_mode),
            "plan_objective_estimate": float(objective_estimate),
            "plan_objective_sigma": float(objective_sigma),
            "plan_objective_with_uncertainty": float(objective_estimate + float(uncertainty_bonus) * objective_sigma),
            "plan_tail_gain": float(prediction_components["raw"]["tail_gain"]),
            "plan_route_win_prob": float(prediction_components["raw"]["route_win_prob"]),
            "plan_new_residual_basin_prob": float(prediction_components["raw"]["new_residual_basin_prob"]),
            "plan_stable_prob": float(prediction_components["raw"]["stable_prob"]),
            "plan_fragile_prob": float(prediction_components["raw"]["fragile_prob"]),
            "plan_cost_exact_pred": float(prediction_components["raw"]["cost_exact_pred"]),
            "plan_cost_wall_pred": float(prediction_components["raw"]["cost_wall_pred"]),
            "plan_cost_total_pred": prediction_components["objective"]["cost_total_pred"],
            "plan_witness_delta_pred": prediction_components["objective"]["witness_delta_pred"],
            "plan_witness_rate_pred": prediction_components["objective"]["witness_rate_pred"],
            "plan_acquisition_threshold": float(threshold_value),
            "plan_prediction_components": prediction_components,
            "plan_witness_value_loss": prediction_components["witness_energy"]["value_loss"],
            "plan_witness_grad_loss": prediction_components["witness_energy"]["grad_loss"],
            "plan_witness_d2_loss": prediction_components["witness_energy"]["d2_loss"],
            "plan_witness_diag_loss": prediction_components["witness_energy"]["diag_loss"],
            "plan_witness_physics_loss": prediction_components["witness_energy"]["physics_loss"],
            "plan_witness_energy_total": prediction_components["witness_energy"]["total"],
            "plan_witness_energy_delta_estimate": prediction_components["witness_energy"]["delta_estimate"],
        })
        route_name = str(candidate.route)
        route_scores[route_name] = max(
            float(route_scores.get(route_name, float("-inf"))),
            float(row_out["plan_objective_with_uncertainty"]),
        )
        rows_out.append(row_out)
    rows_out.sort(
        key=lambda item: (
            float(item.get("plan_objective_with_uncertainty", float("-inf"))),
            float(item.get("plan_confidence", 0.0)),
            str(item.get("plan_route", "")),
            str(item.get("opportunity_id", "")),
            int(item.get("plan_exact_budget", 0)),
        ),
        reverse=True,
    )
    out.update({
        "trained": True,
        "rows": rows_out,
        "route_scores": {str(k): float(v) for k, v in route_scores.items()},
        "acquisition_weights": {str(k): float(v) for k, v in dict(resolved_weights).items()},
        "objective_mode": str(objective_mode),
    })
    return out


def choose_plan(
    bundle: Mapping[str, Any] | None,
    plan_candidates: Sequence[PlanCandidate],
    *,
    advisory_only: bool = True,
    fallback_min_confidence: float = 0.0,
    acquisition_threshold: float = 0.25,
    uncertainty_bonus: float = 0.05,
    acquisition_weights: Mapping[str, float] | None = None,
) -> PlanDecision:
    candidates = [candidate for candidate in list(plan_candidates or []) if isinstance(candidate, PlanCandidate)]
    if not isinstance(bundle, Mapping):
        return PlanDecision(
            trained=False,
            advisory_only=bool(advisory_only),
            candidate_count=int(len(candidates)),
            fallback_used=True,
            fallback_reason="bundle_missing",
        )
    pred = predict_plan_slate(
        bundle,
        candidates,
        acquisition_threshold=float(acquisition_threshold),
        uncertainty_bonus=float(uncertainty_bonus),
        acquisition_weights=acquisition_weights,
    )
    resolved_weights = {
        str(key): float(value)
        for key, value in dict(pred.get("acquisition_weights", _resolved_acquisition_weights(bundle, acquisition_weights)) or {}).items()
    }
    if not bool(pred.get("trained", False)):
        return PlanDecision(
            trained=False,
            advisory_only=bool(advisory_only),
            candidate_count=int(len(candidates)),
            fallback_used=True,
            fallback_reason="scheduler_not_trained",
            acquisition_threshold=float(acquisition_threshold),
            uncertainty_bonus=float(uncertainty_bonus),
            acquisition_weights=resolved_weights,
        )
    rows = [dict(row) for row in list(pred.get("rows", []) or []) if isinstance(row, Mapping)]
    if not rows:
        return PlanDecision(
            trained=True,
            advisory_only=bool(advisory_only),
            candidate_count=int(len(candidates)),
            fallback_used=True,
            fallback_reason="no_scored_candidates",
            acquisition_threshold=float(acquisition_threshold),
            uncertainty_bonus=float(uncertainty_bonus),
            acquisition_weights=resolved_weights,
            route_scores={str(k): float(v) for k, v in dict(pred.get("route_scores", {}) or {}).items()},
        )
    best_row = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    candidate_index = _safe_int(best_row.get("plan_candidate_index", 0), 0)
    chosen_candidate = candidates[candidate_index] if 0 <= candidate_index < len(candidates) else None
    objective_mode = str(pred.get("objective_mode", bundle.get("objective_mode", "acquisition")) or "acquisition")
    best_score = _safe_float(best_row.get("plan_objective_with_uncertainty", best_row.get("plan_acquisition_with_uncertainty", float("-inf"))), float("-inf"))
    runner_score = (
        _safe_float(runner_up.get("plan_objective_with_uncertainty", runner_up.get("plan_acquisition_with_uncertainty", float("-inf"))), float("-inf"))
        if isinstance(runner_up, Mapping)
        else float("-inf")
    )
    acquisition_gap = (
        float(best_score - runner_score)
        if math.isfinite(best_score) and math.isfinite(runner_score)
        else float("inf")
    )
    best_sigma = _safe_float(best_row.get("plan_objective_sigma", best_row.get("plan_acquisition_sigma", 0.0)), 0.0)
    runner_sigma = _safe_float(runner_up.get("plan_objective_sigma", runner_up.get("plan_acquisition_sigma", 0.0)), 0.0) if isinstance(runner_up, Mapping) else 0.0
    acquisition_gap_sigma = (
        float(math.sqrt(max(0.0, float(best_sigma) ** 2 + float(runner_sigma) ** 2)))
        if isinstance(runner_up, Mapping)
        else 0.0
    )
    dominance_prob = (
        _dominance_prob_from_gap(acquisition_gap, acquisition_gap_sigma)
        if isinstance(runner_up, Mapping)
        else 1.0
    )
    confidence = _normalized_dominance_confidence(dominance_prob)
    fallback_used = confidence < float(fallback_min_confidence)
    fallback_reason = "low_dominance_confidence" if fallback_used else ""
    if chosen_candidate is None:
        fallback_used = True
        fallback_reason = "candidate_missing"
    return PlanDecision(
        trained=True,
        advisory_only=bool(advisory_only),
        candidate_count=int(len(candidates)),
        fallback_used=bool(fallback_used),
        fallback_reason=str(fallback_reason),
        confidence=float(confidence),
        confidence_kind="dominance_gap",
        dominance_prob=float(dominance_prob),
        acquisition_gap=0.0 if not math.isfinite(acquisition_gap) else float(acquisition_gap),
        acquisition_gap_sigma=float(acquisition_gap_sigma),
        objective_mode=str(objective_mode),
        objective_gap=0.0 if not math.isfinite(acquisition_gap) else float(acquisition_gap),
        objective_gap_sigma=float(acquisition_gap_sigma),
        chosen_candidate=chosen_candidate,
        chosen_route="" if chosen_candidate is None else str(chosen_candidate.route),
        chosen_opportunity_id="" if chosen_candidate is None else str(chosen_candidate.opportunity_id),
        chosen_exact_budget=0 if chosen_candidate is None else int(chosen_candidate.exact_budget),
        runner_up_route="" if runner_up is None else str(runner_up.get("plan_route", "") or ""),
        runner_up_opportunity_id="" if runner_up is None else str(runner_up.get("opportunity_id", "") or ""),
        runner_up_exact_budget=0 if runner_up is None else int(_safe_int(runner_up.get("plan_exact_budget", 0), 0)),
        acquisition_threshold=float(acquisition_threshold),
        uncertainty_bonus=float(uncertainty_bonus),
        acquisition_weights=resolved_weights,
        route_scores={str(k): float(v) for k, v in dict(pred.get("route_scores", {}) or {}).items()},
        rows=tuple(rows),
    )


__all__ = [
    "PlanCandidate",
    "PlanDecision",
    "build_plan_candidates",
    "build_plan_candidates_from_rows",
    "choose_plan",
    "predict_plan_slate",
]
