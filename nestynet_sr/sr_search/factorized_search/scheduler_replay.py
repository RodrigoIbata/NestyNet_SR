# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from .scheduler_critic import (
    SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS,
)
from .scheduler_dataset import (
    augment_scheduler_shadow_rows,
    normalize_scheduler_budget_ladder,
    scheduler_row_group_id,
    scheduler_threshold_token,
)
from .scheduler import build_plan_candidates_from_rows, choose_plan


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _group_id(row: Mapping[str, Any], idx: int) -> str:
    token = scheduler_row_group_id(row)
    return str(token or f"ungrouped_{int(idx)}")


def _depth_bucket(value: Any) -> str:
    depth = _safe_int(value, -1)
    if depth < 0:
        return "unknown"
    if depth <= 2:
        return "0-2"
    if depth <= 4:
        return "3-4"
    if depth <= 6:
        return "5-6"
    return "7+"


def _binary_brier(probs: Sequence[float], labels: Sequence[float]) -> float | None:
    pairs = [
        (float(prob), float(label))
        for prob, label in zip(list(probs or []), list(labels or []))
        if math.isfinite(float(prob)) and math.isfinite(float(label))
    ]
    if not pairs:
        return None
    return float(sum((prob - label) ** 2 for prob, label in pairs) / len(pairs))


def _mean_or_none(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in list(values or []) if math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _median_or_none(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in list(values or []) if math.isfinite(float(v))]
    if not xs:
        return None
    return float(statistics.median(xs))


def _calibration_bucket(
    buckets: dict[str, dict[str, list[float]]],
    *,
    key: str,
    prob: float | None,
    label: float | None,
) -> None:
    if prob is None or label is None:
        return
    if not math.isfinite(float(prob)) or not math.isfinite(float(label)):
        return
    bucket = buckets.setdefault(str(key), {"probs": [], "labels": []})
    bucket["probs"].append(float(prob))
    bucket["labels"].append(float(label))


def _finalize_calibration(
    buckets: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for key, payload in dict(buckets or {}).items():
        probs = [float(v) for v in list(payload.get("probs", []) or [])]
        labels = [float(v) for v in list(payload.get("labels", []) or [])]
        out[str(key)] = {
            "count": int(len(probs)),
            "mean_prob": _mean_or_none(probs),
            "empirical_rate": _mean_or_none(labels),
            "brier": _binary_brier(probs, labels),
        }
    return out


def _selection_flag(row: Mapping[str, Any]) -> bool:
    truthy_keys = (
        "actual_selected",
        "historical_selected",
        "logged_choice",
        "was_selected",
        "selected",
        "is_selected",
        "is_logged_choice",
        "executed",
        "was_executed",
        "controller_selected",
        "scheduler_applied",
    )
    for key in truthy_keys:
        value = row.get(key, None)
        if isinstance(value, bool) and value:
            return True
        if _safe_int(value, 0) > 0:
            return True
    return False


def _detect_actual_choice_index(group_rows: Sequence[Mapping[str, Any]]) -> int | None:
    flagged = [idx for idx, row in enumerate(group_rows) if _selection_flag(row)]
    if len(flagged) == 1:
        return int(flagged[0])
    positive_exact = [
        idx
        for idx, row in enumerate(group_rows)
        if _safe_int(row.get("budget_exact_spent", 0), 0) > 0
        or _safe_int(row.get("observed_exact_evals", 0), 0) > 0
    ]
    if len(positive_exact) == 1:
        return int(positive_exact[0])
    return None


def _realized_utility(
    row: Mapping[str, Any],
    *,
    budget: int,
    threshold: float,
    weights: Mapping[str, float],
) -> float | None:
    tau = scheduler_threshold_token(float(threshold))
    break_hit = _safe_float_or_none(row.get(f"improve_ge_{tau}_at_budget_{int(budget)}", None))
    tail_gain = _safe_float_or_none(row.get(f"tail_gain_{tau}_at_budget_{int(budget)}", None))
    route_win = _safe_float_or_none(row.get(f"route_win_at_budget_{int(budget)}", None))
    new_residual_basin = _safe_float_or_none(row.get(f"new_residual_basin_at_budget_{int(budget)}", None))
    stable = _safe_float_or_none(row.get(f"stability_at_budget_{int(budget)}", None))
    fragile = _safe_float_or_none(row.get(f"fragility_at_budget_{int(budget)}", None))
    cost_exact = _safe_float_or_none(row.get(f"cost_exact_at_budget_{int(budget)}", None))
    cost_wall = _safe_float_or_none(row.get(f"cost_wall_at_budget_{int(budget)}", None))
    if break_hit is None or tail_gain is None:
        return None
    return float(
        float(weights.get("break", 1.0)) * float(break_hit)
        + float(weights.get("tail", 0.5)) * float(tail_gain)
        + float(weights.get("route_win", 0.3)) * float(route_win or 0.0)
        + float(weights.get("new_residual_basin", 0.2)) * float(new_residual_basin or 0.0)
        + float(weights.get("stable", 0.1)) * float(stable or 0.0)
        - float(weights.get("fragile", 0.15)) * float(fragile or 0.0)
        - float(weights.get("cost_exact", 0.05)) * float(cost_exact or 0.0)
        - float(weights.get("cost_wall", 0.05)) * float(cost_wall or 0.0)
    )


def _realized_witness_delta(
    row: Mapping[str, Any],
    *,
    budget: int,
) -> float | None:
    delta = _safe_float_or_none(row.get(f"witness_energy_delta_at_budget_{int(budget)}", None))
    if delta is not None:
        return float(delta)
    before = _safe_float_or_none(row.get(f"witness_energy_total_before_at_budget_{int(budget)}", None))
    after = _safe_float_or_none(row.get(f"witness_energy_total_after_at_budget_{int(budget)}", None))
    if before is None or after is None:
        return None
    return float(before) - float(after)


def _best_oracle_plan(
    group_rows: Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[int],
    threshold: float,
    weights: Mapping[str, float],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row_idx, row in enumerate(group_rows):
        for budget in budget_ladder:
            utility = _realized_utility(
                row,
                budget=int(budget),
                threshold=float(threshold),
                weights=weights,
            )
            if utility is None:
                continue
            candidate = {
                "row_index": int(row_idx),
                "route": str(row.get("route_source", "") or ""),
                "opportunity_id": str(row.get("opportunity_id", "") or ""),
                "budget": int(budget),
                "utility": float(utility),
                "delta_log_eff": _safe_float_or_none(row.get(f"delta_log_eff_at_budget_{int(budget)}", None)),
            }
            if best is None:
                best = candidate
                continue
            if float(candidate["utility"]) > float(best["utility"]) + 1.0e-12:
                best = candidate
                continue
            if abs(float(candidate["utility"]) - float(best["utility"])) <= 1.0e-12:
                if int(candidate["budget"]) < int(best["budget"]):
                    best = candidate
                    continue
                if (
                    int(candidate["budget"]) == int(best["budget"])
                    and (str(candidate["route"]), str(candidate["opportunity_id"]))
                    < (str(best["route"]), str(best["opportunity_id"]))
                ):
                    best = candidate
    return best


def _best_oracle_witness_plan(
    group_rows: Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[int],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row_idx, row in enumerate(group_rows):
        for budget in budget_ladder:
            witness_delta = _realized_witness_delta(row, budget=int(budget))
            if witness_delta is None:
                continue
            candidate = {
                "row_index": int(row_idx),
                "route": str(row.get("route_source", "") or ""),
                "opportunity_id": str(row.get("opportunity_id", "") or ""),
                "budget": int(budget),
                "witness_delta": float(witness_delta),
            }
            if best is None:
                best = candidate
                continue
            if float(candidate["witness_delta"]) > float(best["witness_delta"]) + 1.0e-12:
                best = candidate
                continue
            if abs(float(candidate["witness_delta"]) - float(best["witness_delta"])) <= 1.0e-12:
                if int(candidate["budget"]) < int(best["budget"]):
                    best = candidate
                    continue
                if (
                    int(candidate["budget"]) == int(best["budget"])
                    and (str(candidate["route"]), str(candidate["opportunity_id"]))
                    < (str(best["route"]), str(best["opportunity_id"]))
                ):
                    best = candidate
    return best


def _minimum_budget_for_utility(
    row: Mapping[str, Any],
    *,
    budget_ladder: Sequence[int],
    threshold: float,
    weights: Mapping[str, float],
    target_utility: float,
) -> int | None:
    for budget in budget_ladder:
        utility = _realized_utility(
            row,
            budget=int(budget),
            threshold=float(threshold),
            weights=weights,
        )
        if utility is None:
            continue
        if float(utility) >= float(target_utility) - 1.0e-9:
            return int(budget)
    return None


def _actual_plan_from_rows(
    group_rows: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    threshold: float,
    weights: Mapping[str, float],
) -> tuple[dict[str, Any] | None, float | None]:
    actual_idx = _detect_actual_choice_index(group_rows)
    if actual_idx is None or not (0 <= int(actual_idx) < len(group_rows)):
        return None, None
    actual_row = dict(group_rows[int(actual_idx)])
    actual_budget = max(
        1,
        _safe_int(
            actual_row.get("budget_exact_spent", actual_row.get("observed_exact_evals", budgets[0])),
            budgets[0],
        ),
    )
    if int(actual_budget) not in budgets:
        actual_budget = min(budgets, key=lambda item: abs(int(item) - int(actual_budget)))
    actual_utility = _realized_utility(
        actual_row,
        budget=int(actual_budget),
        threshold=float(threshold),
        weights=weights,
    )
    return {
        "row_index": int(actual_idx),
        "route": str(actual_row.get("route_source", "") or ""),
        "opportunity_id": str(actual_row.get("opportunity_id", "") or ""),
        "budget": int(actual_budget),
        "utility": actual_utility,
        "row": actual_row,
    }, actual_utility


def _decision_selected_row(
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    route: str,
    opportunity_id: str,
    budget: int,
) -> dict[str, Any] | None:
    for row in list(decision_rows or []):
        if not isinstance(row, Mapping):
            continue
        if (
            str(row.get("plan_route", "") or "") == str(route)
            and str(row.get("opportunity_id", "") or "") == str(opportunity_id)
            and int(_safe_int(row.get("plan_exact_budget", 0), 0)) == int(budget)
        ):
            return dict(row)
    return None


def replay_scheduler_decisions(
    log_rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    *,
    acquisition_threshold: float = 0.25,
    acquisition_weights: Mapping[str, float] | None = None,
    budget_ladder: Sequence[Any] | None = None,
    fallback_min_confidence: float = 0.0,
    uncertainty_bonus: float = 0.05,
    hole_exact_budget_cap: int | None = None,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "n_rows": int(len(list(log_rows or []))),
        "n_groups": 0,
        "groups_replayed": 0,
        "groups_with_actual_choice": 0,
        "groups_with_scheduler_fallback": 0,
        "groups_with_resolved_fallback": 0,
        "groups_with_unresolved_fallback": 0,
        "groups_with_degraded_context": 0,
        "groups_with_witness_labels": 0,
        "grouping_source_counts": {},
        "top1_hit_rate": None,
        "mean_regret": None,
        "median_regret": None,
        "mean_normalized_regret": None,
        "actual_mean_regret": None,
        "actual_mean_normalized_regret": None,
        "mean_witness_regret": None,
        "median_witness_regret": None,
        "mean_normalized_witness_regret": None,
        "actual_mean_witness_regret": None,
        "actual_mean_normalized_witness_regret": None,
        "mean_wasted_budget": None,
        "actual_mean_wasted_budget": None,
        "calibration_by_route": {},
        "calibration_by_depth": {},
        "calibration_by_budget": {},
        "decision_rows": [],
    }
    rows_in = [dict(row) for row in list(log_rows or []) if isinstance(row, Mapping)]
    if not isinstance(bundle, Mapping) or not bool(bundle.get("scheduler_critic_trained", False)):
        return out
    budgets = normalize_scheduler_budget_ladder(
        budget_ladder if budget_ladder is not None else bundle.get("budget_ladder", None)
    )
    rows = augment_scheduler_shadow_rows(
        rows_in,
        budget_ladder=budgets,
        threshold_ladder=bundle.get("threshold_ladder", None),
    )
    if not rows:
        return out
    weights = dict(SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS)
    weights.update(dict(bundle.get("acquisition_weights", {}) or {}))
    if acquisition_weights is not None:
        weights.update({str(k): float(v) for k, v in dict(acquisition_weights).items()})
    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(_group_id(row, idx), []).append(dict(row))
    out["trained"] = True
    out["n_groups"] = int(len(groups))
    route_cal: dict[str, dict[str, list[float]]] = {}
    depth_cal: dict[str, dict[str, list[float]]] = {}
    budget_cal: dict[str, dict[str, list[float]]] = {}
    regrets: list[float] = []
    norm_regrets: list[float] = []
    actual_regrets: list[float] = []
    actual_norm_regrets: list[float] = []
    witness_regrets: list[float] = []
    witness_norm_regrets: list[float] = []
    actual_witness_regrets: list[float] = []
    actual_witness_norm_regrets: list[float] = []
    wasted_budgets: list[float] = []
    actual_wasted_budgets: list[float] = []
    top1_hits = 0
    replayed = 0
    actual_choice_count = 0
    fallback_count = 0
    fallback_resolved = 0
    fallback_unresolved = 0
    degraded_count = 0
    witness_label_count = 0
    grouping_source_counts: dict[str, int] = {}
    decision_rows: list[dict[str, Any]] = []
    for group_name, group_rows in groups.items():
        grouping_source = str(group_rows[0].get("decision_grouping_source", "") or "unknown")
        grouping_source_counts[grouping_source] = grouping_source_counts.get(grouping_source, 0) + 1
        grouping_degraded = bool(group_rows[0].get("decision_group_degraded", False))
        if grouping_degraded:
            degraded_count += 1
        plan_candidates = build_plan_candidates_from_rows(
            group_rows,
            exact_budget_ladder=budgets,
            hole_exact_budget_cap=hole_exact_budget_cap,
        )
        decision = choose_plan(
            bundle,
            plan_candidates,
            advisory_only=False,
            fallback_min_confidence=float(fallback_min_confidence),
            acquisition_threshold=float(acquisition_threshold),
            uncertainty_bonus=float(uncertainty_bonus),
            acquisition_weights=weights if acquisition_weights is not None else None,
        )
        if not bool(getattr(decision, "trained", False)):
            continue
        oracle = _best_oracle_plan(
            group_rows,
            budget_ladder=budgets,
            threshold=float(acquisition_threshold),
            weights=weights,
        )
        if oracle is None:
            continue
        oracle_witness = _best_oracle_witness_plan(
            group_rows,
            budget_ladder=budgets,
        )
        if oracle_witness is not None:
            witness_label_count += 1
        pred_rows = [dict(row) for row in list(getattr(decision, "rows", ()) or []) if isinstance(row, Mapping)]
        if not pred_rows:
            continue
        actual_plan, actual_utility = _actual_plan_from_rows(
            group_rows,
            budgets=budgets,
            threshold=float(acquisition_threshold),
            weights=weights,
        )
        if actual_plan is not None:
            actual_choice_count += 1
        predicted_source = "scheduler_choice"
        predicted = None
        predicted_budget = None
        if bool(getattr(decision, "fallback_used", False)):
            fallback_count += 1
            if actual_plan is None:
                fallback_unresolved += 1
                continue
            fallback_resolved += 1
            predicted_source = "fallback_actual_choice"
            predicted = dict(actual_plan["row"])
            predicted_budget = int(actual_plan["budget"])
        else:
            chosen_candidate = getattr(decision, "chosen_candidate", None)
            if chosen_candidate is None:
                continue
            predicted_budget = int(getattr(chosen_candidate, "exact_budget", budgets[0]) or budgets[0])
            predicted = _decision_selected_row(
                pred_rows,
                route=str(getattr(chosen_candidate, "route", "") or ""),
                opportunity_id=str(getattr(chosen_candidate, "opportunity_id", "") or ""),
                budget=int(predicted_budget),
            )
            if predicted is None:
                continue
        predicted_utility = _realized_utility(
            predicted,
            budget=int(predicted_budget),
            threshold=float(acquisition_threshold),
            weights=weights,
        )
        if predicted_utility is None:
            continue
        oracle_utility = float(oracle["utility"])
        regret = max(0.0, float(oracle_utility) - float(predicted_utility))
        regrets.append(float(regret))
        denom = max(1.0e-9, abs(float(oracle_utility)))
        norm_regrets.append(float(regret) / denom)
        if (
            str(predicted.get("route_source", "") or "") == str(oracle.get("route", "") or "")
            and str(predicted.get("opportunity_id", "") or "") == str(oracle.get("opportunity_id", "") or "")
            and int(predicted_budget) == int(oracle.get("budget", budgets[0]))
        ):
            top1_hits += 1
        min_budget = _minimum_budget_for_utility(
            predicted,
            budget_ladder=budgets,
            threshold=float(acquisition_threshold),
            weights=weights,
            target_utility=float(predicted_utility),
        )
        if min_budget is not None:
            wasted_budgets.append(float(max(0, int(predicted_budget) - int(min_budget))))
        if actual_plan is not None and actual_utility is not None:
            actual_regret = max(0.0, float(oracle_utility) - float(actual_utility))
            actual_regrets.append(float(actual_regret))
            actual_norm_regrets.append(float(actual_regret) / denom)
            min_actual_budget = _minimum_budget_for_utility(
                group_rows[int(actual_plan["row_index"])],
                budget_ladder=budgets,
                threshold=float(acquisition_threshold),
                weights=weights,
                target_utility=float(actual_utility),
            )
            if min_actual_budget is not None:
                actual_wasted_budgets.append(float(max(0, int(actual_plan["budget"]) - int(min_actual_budget))))
        predicted_witness_delta = _realized_witness_delta(predicted, budget=int(predicted_budget))
        actual_witness_delta = (
            None
            if actual_plan is None
            else _realized_witness_delta(
                group_rows[int(actual_plan["row_index"])],
                budget=int(actual_plan["budget"]),
            )
        )
        witness_regret = None
        actual_witness_regret = None
        if oracle_witness is not None and predicted_witness_delta is not None:
            oracle_witness_delta = float(oracle_witness["witness_delta"])
            witness_regret = max(0.0, oracle_witness_delta - float(predicted_witness_delta))
            witness_regrets.append(float(witness_regret))
            witness_norm_regrets.append(float(witness_regret) / max(1.0e-9, abs(oracle_witness_delta)))
        if oracle_witness is not None and actual_witness_delta is not None:
            oracle_witness_delta = float(oracle_witness["witness_delta"])
            actual_witness_regret = max(0.0, oracle_witness_delta - float(actual_witness_delta))
            actual_witness_regrets.append(float(actual_witness_regret))
            actual_witness_norm_regrets.append(
                float(actual_witness_regret) / max(1.0e-9, abs(oracle_witness_delta))
            )
        replayed += 1
        for row in pred_rows:
            for budget in budgets:
                prob = _safe_float_or_none(
                    row.get(
                        f"break_prob_{scheduler_threshold_token(float(acquisition_threshold))}_at_budget_{int(budget)}",
                        None,
                    )
                )
                label = _safe_float_or_none(row.get(f"improve_ge_{scheduler_threshold_token(float(acquisition_threshold))}_at_budget_{int(budget)}", None))
                route_name = str(row.get("route_source", "") or "")
                depth_name = _depth_bucket(row.get("parent_depth", None))
                _calibration_bucket(route_cal, key=route_name, prob=prob, label=label)
                _calibration_bucket(depth_cal, key=depth_name, prob=prob, label=label)
                _calibration_bucket(budget_cal, key=str(int(budget)), prob=prob, label=label)
        decision_rows.append(
            {
                "group_id": str(group_name),
                "grouping_source": str(grouping_source),
                "grouping_degraded": bool(grouping_degraded),
                "grouping_degraded_reason": str(group_rows[0].get("decision_group_degraded_reason", "") or ""),
                "predicted_route": str(predicted.get("route_source", "") or ""),
                "predicted_opportunity_id": str(predicted.get("opportunity_id", "") or ""),
                "predicted_budget": int(predicted_budget),
                "predicted_utility": float(predicted_utility),
                "predicted_source": str(predicted_source),
                "scheduler_fallback_used": bool(getattr(decision, "fallback_used", False)),
                "scheduler_fallback_reason": str(getattr(decision, "fallback_reason", "") or ""),
                "scheduler_confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
                "scheduler_confidence_kind": str(getattr(decision, "confidence_kind", "dominance_gap") or "dominance_gap"),
                "scheduler_dominance_prob": float(getattr(decision, "dominance_prob", 0.0) or 0.0),
                "scheduler_acquisition_gap": float(getattr(decision, "acquisition_gap", 0.0) or 0.0),
                "scheduler_acquisition_gap_sigma": float(getattr(decision, "acquisition_gap_sigma", 0.0) or 0.0),
                "scheduler_candidate_count": int(getattr(decision, "candidate_count", 0) or 0),
                "scheduler_proposed_route": str(getattr(decision, "chosen_route", "") or ""),
                "scheduler_proposed_opportunity_id": str(getattr(decision, "chosen_opportunity_id", "") or ""),
                "scheduler_proposed_budget": int(getattr(decision, "chosen_exact_budget", 0) or 0),
                "scheduler_runner_up_route": str(getattr(decision, "runner_up_route", "") or ""),
                "scheduler_runner_up_opportunity_id": str(getattr(decision, "runner_up_opportunity_id", "") or ""),
                "scheduler_runner_up_budget": int(getattr(decision, "runner_up_exact_budget", 0) or 0),
                "oracle_route": str(oracle.get("route", "") or ""),
                "oracle_opportunity_id": str(oracle.get("opportunity_id", "") or ""),
                "oracle_budget": int(oracle.get("budget", budgets[0])),
                "oracle_utility": float(oracle_utility),
                "regret": float(regret),
                "normalized_regret": float(regret) / denom,
                "oracle_witness_budget": 0 if oracle_witness is None else int(oracle_witness.get("budget", 0) or 0),
                "oracle_witness_delta": None if oracle_witness is None else float(oracle_witness.get("witness_delta", 0.0)),
                "predicted_witness_delta": None if predicted_witness_delta is None else float(predicted_witness_delta),
                "witness_regret": None if witness_regret is None else float(witness_regret),
                "actual_route": "" if actual_plan is None else str(actual_plan.get("route", "") or ""),
                "actual_opportunity_id": "" if actual_plan is None else str(actual_plan.get("opportunity_id", "") or ""),
                "actual_budget": 0 if actual_plan is None else int(actual_plan.get("budget", 0) or 0),
                "actual_utility": None if actual_utility is None else float(actual_utility),
                "actual_witness_delta": None if actual_witness_delta is None else float(actual_witness_delta),
                "actual_witness_regret": None if actual_witness_regret is None else float(actual_witness_regret),
            }
        )
    out.update(
        {
            "groups_replayed": int(replayed),
            "groups_with_actual_choice": int(actual_choice_count),
            "groups_with_scheduler_fallback": int(fallback_count),
            "groups_with_resolved_fallback": int(fallback_resolved),
            "groups_with_unresolved_fallback": int(fallback_unresolved),
            "groups_with_degraded_context": int(degraded_count),
            "groups_with_witness_labels": int(witness_label_count),
            "grouping_source_counts": {
                str(key): int(value) for key, value in grouping_source_counts.items()
            },
            "top1_hit_rate": None if replayed <= 0 else float(top1_hits) / float(replayed),
            "mean_regret": _mean_or_none(regrets),
            "median_regret": _median_or_none(regrets),
            "mean_normalized_regret": _mean_or_none(norm_regrets),
            "actual_mean_regret": _mean_or_none(actual_regrets),
            "actual_mean_normalized_regret": _mean_or_none(actual_norm_regrets),
            "mean_witness_regret": _mean_or_none(witness_regrets),
            "median_witness_regret": _median_or_none(witness_regrets),
            "mean_normalized_witness_regret": _mean_or_none(witness_norm_regrets),
            "actual_mean_witness_regret": _mean_or_none(actual_witness_regrets),
            "actual_mean_normalized_witness_regret": _mean_or_none(actual_witness_norm_regrets),
            "mean_wasted_budget": _mean_or_none(wasted_budgets),
            "actual_mean_wasted_budget": _mean_or_none(actual_wasted_budgets),
            "calibration_by_route": _finalize_calibration(route_cal),
            "calibration_by_depth": _finalize_calibration(depth_cal),
            "calibration_by_budget": _finalize_calibration(budget_cal),
            "decision_rows": decision_rows,
        }
    )
    return out


__all__ = ["replay_scheduler_decisions"]
