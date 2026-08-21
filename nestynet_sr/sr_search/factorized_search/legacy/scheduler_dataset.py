# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .opportunity_dataset import (
    build_opportunity_shadow_dataset,
)
from ..shared_opportunity import (
    normalize_realized_witness_energy_fields,
    normalize_witness_energy_fields,
    shared_opportunity_row_dict,
)


SCHEDULER_DATASET_MODE = "scheduler_shadow_dataset"
SCHEDULER_DEFAULT_BUDGET_LADDER: tuple[int, ...] = (1, 2, 4, 8)
SCHEDULER_DEFAULT_THRESHOLD_LADDER: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)


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


def _safe_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _observed_witness_outcome_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    realized = normalize_realized_witness_energy_fields(row)
    return {
        "witness_value_loss_before": realized.get("realized_witness_value_loss_before", None),
        "witness_grad_loss_before": realized.get("realized_witness_grad_loss_before", None),
        "witness_d2_loss_before": realized.get("realized_witness_d2_loss_before", None),
        "witness_diag_loss_before": realized.get("realized_witness_diag_loss_before", None),
        "witness_physics_loss_before": realized.get("realized_witness_physics_loss_before", None),
        "witness_energy_total_before": realized.get("realized_witness_energy_total_before", None),
        "witness_value_loss_after": realized.get("realized_witness_value_loss_after", None),
        "witness_grad_loss_after": realized.get("realized_witness_grad_loss_after", None),
        "witness_d2_loss_after": realized.get("realized_witness_d2_loss_after", None),
        "witness_diag_loss_after": realized.get("realized_witness_diag_loss_after", None),
        "witness_physics_loss_after": realized.get("realized_witness_physics_loss_after", None),
        "witness_energy_total_after": realized.get("realized_witness_energy_total_after", None),
        "witness_value_delta": realized.get("realized_witness_value_delta", None),
        "witness_grad_delta": realized.get("realized_witness_grad_delta", None),
        "witness_d2_delta": realized.get("realized_witness_d2_delta", None),
        "witness_diag_delta": realized.get("realized_witness_diag_delta", None),
        "witness_physics_delta": realized.get("realized_witness_physics_delta", None),
        "witness_energy_delta": realized.get("realized_witness_energy_delta", None),
    }


def _outcome_has_witness_transition(row: Mapping[str, Any]) -> bool:
    return bool(
        _safe_float_or_none(row.get("witness_energy_total_after", None)) is not None
        or _safe_float_or_none(row.get("witness_energy_delta", None)) is not None
    )


def normalize_scheduler_budget_ladder(values: Sequence[Any] | None = None) -> tuple[int, ...]:
    ladder = values if values is not None else SCHEDULER_DEFAULT_BUDGET_LADDER
    cleaned = sorted({max(1, int(v)) for v in list(ladder or ()) if int(v) > 0})
    return tuple(cleaned or SCHEDULER_DEFAULT_BUDGET_LADDER)


def normalize_scheduler_threshold_ladder(values: Sequence[Any] | None = None) -> tuple[float, ...]:
    ladder = values if values is not None else SCHEDULER_DEFAULT_THRESHOLD_LADDER
    cleaned = sorted({
        round(float(v), 6)
        for v in list(ladder or ())
        if _safe_float_or_none(v) is not None and float(v) >= 0.0
    })
    return tuple(cleaned or SCHEDULER_DEFAULT_THRESHOLD_LADDER)


def scheduler_threshold_token(value: Any) -> str:
    vv = _safe_float(value, 0.0)
    token = f"{vv:.6g}".replace(".", "p").replace("-", "m")
    return token or "0"


def scheduler_context_id(row: Mapping[str, Any]) -> str:
    item = row if isinstance(row, Mapping) else {}
    return str(
        item.get("scheduler_decision_context_id", "")
        or item.get("decision_group_id", "")
        or item.get("decision_context_id", "")
        or item.get("decision_id", "")
        or ""
    ).strip()


def route_local_context_id(row: Mapping[str, Any]) -> str:
    item = row if isinstance(row, Mapping) else {}
    return str(
        item.get("route_decision_context_id", "")
        or item.get("decision_context_id", "")
        or item.get("decision_id", "")
        or ""
    ).strip()


def scheduler_row_group_id(row: Mapping[str, Any]) -> str:
    decision_group = scheduler_context_id(row)
    if decision_group:
        return decision_group
    item = row if isinstance(row, Mapping) else {}
    parts = [
        str(item.get("source_row_family_id", "") or ""),
        str(item.get("shadow_source_row_id", "") or ""),
        str(item.get("opportunity_id", "") or ""),
    ]
    parts = [part for part in parts if part]
    if parts:
        return "::".join(parts)
    return str(item.get("opportunity_id", "") or "ungrouped")


def _row_has_shadow_signal(row: Mapping[str, Any]) -> bool:
    if not isinstance(row, Mapping):
        return False
    if "current_best_route_eff_mse" in row:
        return True
    return any(str(key).startswith("expected_gain_at_budget_") for key in row.keys())


def _coerce_scheduler_source_rows(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[int],
    include_repair: bool,
    include_build: bool,
    include_hole: bool,
    per_opportunity_timeout_s: float | None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    if isinstance(payload, Mapping):
        source_mode = str(payload.get("mode", "") or "")
        rows = [dict(row) for row in list(payload.get("rows", []) or []) if isinstance(row, Mapping)]
    else:
        source_mode = ""
        rows = [dict(row) for row in list(payload or []) if isinstance(row, Mapping)]
    if rows and all(_row_has_shadow_signal(row) for row in rows):
        return rows, source_mode, {
            "source_mode": str(source_mode),
            "n_source_rows": int(len(rows)),
            "derived_from_opportunity_shadow_dataset": bool(source_mode == "opportunity_shadow_dataset"),
        }
    base_dataset = build_opportunity_shadow_dataset(
        payload,
        budget_ladder=(0, *tuple(int(v) for v in budget_ladder)),
        include_repair=bool(include_repair),
        include_build=bool(include_build),
        include_hole=bool(include_hole),
        per_opportunity_timeout_s=per_opportunity_timeout_s,
    )
    return (
        [dict(row) for row in list(base_dataset.get("rows", []) or []) if isinstance(row, Mapping)],
        str(base_dataset.get("source_mode", source_mode) or source_mode),
        {
            "source_mode": str(base_dataset.get("source_mode", source_mode) or source_mode),
            "n_source_rows": int(base_dataset.get("n_source_rows", len(rows)) or len(rows)),
            "base_dataset_meta_rows": [dict(row) for row in list(base_dataset.get("meta_rows", []) or []) if isinstance(row, Mapping)],
            "derived_from_opportunity_shadow_dataset": True,
        },
    )


def _derive_future_best_eff(current_eff: float | None, gain_value: float | None) -> float | None:
    if current_eff is None or gain_value is None:
        return None
    if not math.isfinite(float(current_eff)) or not math.isfinite(float(gain_value)):
        return None
    return float(max(0.0, float(current_eff) - max(0.0, float(gain_value))))


def _delta_log_eff(current_eff: float | None, future_eff: float | None, *, eps: float = 1.0e-30) -> float | None:
    if current_eff is None or future_eff is None:
        return None
    if not math.isfinite(float(current_eff)) or not math.isfinite(float(future_eff)):
        return None
    try:
        return float(max(0.0, math.log(float(current_eff) + eps) - math.log(float(future_eff) + eps)))
    except Exception:
        return None


def _scheduler_outcome_rows(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for row in list(payload.get("scheduler_outcome_log", []) or []):
        if isinstance(row, Mapping):
            out.append(dict(row))
    for row in list(payload.get("scheduler_decision_log", []) or []):
        if not isinstance(row, Mapping):
            continue
        realized = row.get("realized_outcome", None)
        if not isinstance(realized, Mapping):
            continue
        merged = dict(realized)
        if not str(merged.get("decision_context_id", "") or "").strip():
            merged["decision_context_id"] = str(
                row.get("scheduler_decision_context_id", "")
                or row.get("decision_context_id", "")
                or ""
            )
        if not str(merged.get("route", "") or "").strip():
            merged["route"] = str(
                row.get("scheduler_chosen_route", "")
                or row.get("chosen_route", "")
                or ""
            )
        if not str(merged.get("opportunity_id", "") or "").strip():
            merged["opportunity_id"] = str(
                row.get("scheduler_chosen_opportunity_id", "")
                or row.get("opportunity_id", "")
                or ""
            )
        if merged.get("exact_budget", None) is None:
            merged["exact_budget"] = row.get("scheduler_chosen_exact_budget", None)
        chosen_prediction = row.get("chosen_candidate_prediction", None)
        if isinstance(chosen_prediction, Mapping):
            if merged.get("budget_exact_spent", None) is None:
                merged["budget_exact_spent"] = chosen_prediction.get("budget_exact_spent", None)
            if merged.get("budget_remaining", None) is None:
                merged["budget_remaining"] = chosen_prediction.get("budget_remaining", None)
            chosen_witness = normalize_witness_energy_fields(chosen_prediction)
            for src_key, dst_key in (
                ("witness_value_loss", "realized_witness_value_loss_before"),
                ("witness_grad_loss", "realized_witness_grad_loss_before"),
                ("witness_d2_loss", "realized_witness_d2_loss_before"),
                ("witness_diag_loss", "realized_witness_diag_loss_before"),
                ("witness_physics_loss", "realized_witness_physics_loss_before"),
                ("witness_energy_total", "realized_witness_energy_total_before"),
            ):
                if merged.get(dst_key, None) is None:
                    merged[dst_key] = chosen_witness.get(src_key, None)
        if isinstance(realized, Mapping):
            realized_witness = normalize_witness_energy_fields(realized)
            for src_key, dst_key in (
                ("witness_value_loss", "realized_witness_value_loss_after"),
                ("witness_grad_loss", "realized_witness_grad_loss_after"),
                ("witness_d2_loss", "realized_witness_d2_loss_after"),
                ("witness_diag_loss", "realized_witness_diag_loss_after"),
                ("witness_physics_loss", "realized_witness_physics_loss_after"),
                ("witness_energy_total", "realized_witness_energy_total_after"),
            ):
                if merged.get(dst_key, None) is None:
                    merged[dst_key] = realized_witness.get(src_key, None)
        merged.update(normalize_realized_witness_energy_fields(merged))
        out.append(merged)
    return out


def _coerce_observed_scheduler_outcome(
    row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    if not bool(row.get("executed", False)):
        return None
    decision_context_id = str(
        row.get("scheduler_decision_context_id", "")
        or row.get("decision_context_id", "")
        or row.get("decision_id", "")
        or ""
    ).strip()
    route = str(row.get("route", "") or row.get("selected_route", "") or "").strip().lower().replace("-", "_")
    opportunity_id = str(row.get("opportunity_id", "") or "").strip()
    exact_budget = _safe_int_or_none(row.get("exact_budget", None))
    budget_exact_spent = _safe_int_or_none(row.get("budget_exact_spent", None))
    if not decision_context_id or not route or not opportunity_id:
        return None
    if exact_budget is None or exact_budget <= 0:
        return None
    if budget_exact_spent is None or budget_exact_spent < 0:
        return None
    realized_witness = normalize_realized_witness_energy_fields(row)
    return {
        "decision_context_id": str(decision_context_id),
        "route": str(route),
        "opportunity_id": str(opportunity_id),
        "exact_budget": int(exact_budget),
        "budget_exact_spent": int(budget_exact_spent),
        "budget_remaining": _safe_int_or_none(row.get("budget_remaining", None)),
        "realized_wall_seconds": _safe_float_or_none(row.get("realized_wall_seconds", None)),
        "realized_exact_evals": _safe_int_or_none(row.get("realized_exact_evals", None)),
        "realized_preview_evals": _safe_int_or_none(row.get("realized_preview_evals", None)),
        "realized_micro_tokens": _safe_int_or_none(row.get("realized_micro_tokens", None)),
        "realized_widen_tokens": _safe_int_or_none(row.get("realized_widen_tokens", None)),
        "realized_local_delta_log_eff": _safe_float_or_none(row.get("realized_local_delta_log_eff", None)),
        "realized_global_delta_log_eff": _safe_float_or_none(row.get("realized_global_delta_log_eff", None)),
        "realized_new_residual_basin": row.get("realized_new_residual_basin", None),
        "realized_fragility": row.get("realized_fragility", None),
        "realized_stability": row.get("realized_stability", None),
        "status": str(row.get("status", "") or ""),
        **realized_witness,
    }


def _scheduler_outcome_observation_score(row: Mapping[str, Any]) -> int:
    score = 0
    for key in (
        "realized_wall_seconds",
        "realized_exact_evals",
        "realized_preview_evals",
        "realized_micro_tokens",
        "realized_widen_tokens",
        "realized_witness_energy_total_before",
        "realized_witness_energy_total_after",
        "realized_witness_energy_delta",
    ):
        if row.get(key, None) is not None:
            score += 1
    return int(score)


def _scheduler_observed_outcome_index(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, str, int], dict[int, dict[str, Any]]], dict[str, Any]]:
    index: dict[tuple[str, str, str, int], dict[int, dict[str, Any]]] = {}
    n_rows = 0
    n_indexed = 0
    for raw in _scheduler_outcome_rows(payload):
        n_rows += 1
        outcome = _coerce_observed_scheduler_outcome(raw)
        if outcome is None:
            continue
        key = (
            str(outcome["decision_context_id"]),
            str(outcome["route"]),
            str(outcome["opportunity_id"]),
            int(outcome["budget_exact_spent"]),
        )
        budget = int(outcome["exact_budget"])
        budget_map = index.setdefault(key, {})
        prev = budget_map.get(budget, None)
        if prev is None or _scheduler_outcome_observation_score(outcome) >= _scheduler_outcome_observation_score(prev):
            budget_map[budget] = outcome
        n_indexed += 1
    return index, {
        "n_scheduler_outcome_rows": int(n_rows),
        "n_scheduler_outcomes_indexed": int(n_indexed),
    }


def _backfill_rows_with_scheduler_outcomes(
    rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcome_index, meta = _scheduler_observed_outcome_index(payload)
    if not outcome_index:
        return [dict(row) for row in rows], {
            **meta,
            "n_rows_with_observed_cost_backfill": 0,
            "n_budget_backfills": 0,
            "n_rows_with_observed_witness_backfill": 0,
            "n_budget_witness_backfills": 0,
        }
    out: list[dict[str, Any]] = []
    matched_rows = 0
    matched_budgets = 0
    matched_witness_rows = 0
    matched_witness_budgets = 0
    for raw_row in rows:
        row = dict(raw_row)
        key = (
            str(
                row.get("scheduler_decision_context_id", "")
                or row.get("decision_context_id", "")
                or row.get("decision_id", "")
                or ""
            ),
            str(row.get("route_source", "") or row.get("route", "") or "").strip().lower().replace("-", "_"),
            str(row.get("opportunity_id", "") or ""),
            max(0, _safe_int(row.get("budget_exact_spent", row.get("shadow_prefix_index", 0)), 0)),
        )
        observed_by_budget = outcome_index.get(key, None)
        if not observed_by_budget:
            out.append(row)
            continue
        existing = row.get("scheduler_observed_outcomes_by_budget", None)
        merged: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
        row_has_witness = False
        for budget, outcome in sorted(observed_by_budget.items()):
            witness_fields = _observed_witness_outcome_fields(outcome)
            merged[str(int(budget))] = {
                "wall_seconds": outcome.get("realized_wall_seconds", None),
                "exact_evals": outcome.get("realized_exact_evals", None),
                "preview_evals": outcome.get("realized_preview_evals", None),
                "micro_tokens": outcome.get("realized_micro_tokens", None),
                "widen_tokens": outcome.get("realized_widen_tokens", None),
                "local_delta_log_eff": outcome.get("realized_local_delta_log_eff", None),
                "global_delta_log_eff": outcome.get("realized_global_delta_log_eff", None),
                "new_residual_basin": outcome.get("realized_new_residual_basin", None),
                "fragility": outcome.get("realized_fragility", None),
                "stability": outcome.get("realized_stability", None),
                "status": outcome.get("status", ""),
                **witness_fields,
            }
            if _outcome_has_witness_transition(witness_fields):
                row_has_witness = True
                matched_witness_budgets += 1
        row["scheduler_observed_outcomes_by_budget"] = merged
        matched_rows += 1
        matched_budgets += len(observed_by_budget)
        if row_has_witness:
            matched_witness_rows += 1
        out.append(row)
    return out, {
        **meta,
        "n_rows_with_observed_cost_backfill": int(matched_rows),
        "n_budget_backfills": int(matched_budgets),
        "n_rows_with_observed_witness_backfill": int(matched_witness_rows),
        "n_budget_witness_backfills": int(matched_witness_budgets),
    }


def _observed_outcome_for_budget(
    row: Mapping[str, Any],
    budget: int,
) -> Mapping[str, Any] | None:
    observed = row.get("scheduler_observed_outcomes_by_budget", None)
    if not isinstance(observed, Mapping):
        return None
    direct = observed.get(str(int(budget)), observed.get(int(budget), None))
    return direct if isinstance(direct, Mapping) else None


def _derive_exact_cost_label(
    row: Mapping[str, Any],
    budget: int,
) -> tuple[float, str, float]:
    observed_outcome = _observed_outcome_for_budget(row, budget)
    if observed_outcome is not None:
        observed_exact = _safe_int_or_none(observed_outcome.get("exact_evals", None))
        if observed_exact is not None and observed_exact >= 0:
            return float(observed_exact), "observed_outcome", 1.0
    observed_exact = _safe_int_or_none(row.get("observed_exact_evals", None))
    if observed_exact is not None and observed_exact > 0:
        anchor = max(
            1,
            _safe_int(row.get("budget_exact_spent", observed_exact), observed_exact),
        )
        return (
            float(max(0.0, float(observed_exact)) * (float(max(1, int(budget))) / float(anchor))),
            "observed_scaled",
            1.0,
        )
    return float(max(1, int(budget))), "budget_exact_heuristic", 0.0


def _derive_wall_cost_label(
    row: Mapping[str, Any],
    budget: int,
) -> tuple[float | None, str, float]:
    observed_outcome = _observed_outcome_for_budget(row, budget)
    if observed_outcome is not None:
        observed_wall = _safe_float_or_none(observed_outcome.get("wall_seconds", None))
        if observed_wall is not None:
            return float(max(0.0, observed_wall)), "observed_outcome", 1.0
    observed_wall = _safe_float_or_none(row.get("observed_wall_seconds", None))
    observed_exact = _safe_int(row.get("observed_exact_evals", 0), 0)
    if observed_wall is not None and observed_exact > 0:
        return (
            float(max(0.0, observed_wall) * (float(max(1, int(budget))) / float(observed_exact))),
            "observed_scaled",
            1.0,
        )
    cost_estimate = _safe_float_or_none(row.get("cost_estimate", None))
    if cost_estimate is not None:
        anchor = max(1, _safe_int(row.get("budget_exact_spent", 1), 1))
        return (
            float(max(0.0, cost_estimate) * (float(max(1, int(budget))) / float(anchor))),
            "heuristic_estimate_scaled",
            0.0,
        )
    return None, "missing", 0.0


def _derive_total_cost_label(
    row: Mapping[str, Any],
    budget: int,
    *,
    fallback_exact_cost: float,
) -> tuple[float, str, float]:
    observed_outcome = _observed_outcome_for_budget(row, budget)
    if observed_outcome is not None:
        exact = max(0, _safe_int(observed_outcome.get("exact_evals", 0), 0))
        preview = max(0, _safe_int(observed_outcome.get("preview_evals", 0), 0))
        micro = max(0, _safe_int(observed_outcome.get("micro_tokens", 0), 0))
        widen = max(0, _safe_int(observed_outcome.get("widen_tokens", 0), 0))
        return float(exact + preview + micro + widen), "observed_outcome", 1.0
    observed_preview = _safe_int(row.get("observed_preview_evals", row.get("preview_candidate_count_total", 0)), 0)
    observed_micro = _safe_int(row.get("observed_micro_tokens", 0), 0)
    observed_widen = _safe_int(row.get("observed_widen_tokens", 0), 0)
    return (
        float(max(0.0, float(fallback_exact_cost)) + int(observed_preview) + int(observed_micro) + int(observed_widen)),
        "fallback_mixed",
        0.0,
    )


def _derive_witness_transition_label(
    row: Mapping[str, Any],
    budget: int,
) -> dict[str, Any]:
    existing_before = _safe_float_or_none(row.get(f"witness_energy_total_before_at_budget_{int(budget)}", None))
    existing_after = _safe_float_or_none(row.get(f"witness_energy_total_after_at_budget_{int(budget)}", None))
    existing_delta = _safe_float_or_none(row.get(f"witness_energy_delta_at_budget_{int(budget)}", None))
    existing_source = str(row.get(f"witness_energy_label_source_at_budget_{int(budget)}", "") or "")
    existing_mask = _safe_float_or_none(row.get(f"witness_energy_observed_mask_at_budget_{int(budget)}", None))
    observed_outcome = _observed_outcome_for_budget(row, budget)
    if observed_outcome is None:
        return {
            "before": existing_before,
            "after": existing_after,
            "delta": existing_delta,
            "source": existing_source or ("existing_row" if (existing_after is not None or existing_delta is not None) else "missing"),
            "observed_mask": (
                float(existing_mask)
                if existing_mask is not None
                else (1.0 if (existing_after is not None or existing_delta is not None) else 0.0)
            ),
        }
    row_witness = normalize_witness_energy_fields(row)
    before = _safe_float_or_none(
        observed_outcome.get("witness_energy_total_before", row_witness.get("witness_energy_total", None))
    )
    after = _safe_float_or_none(observed_outcome.get("witness_energy_total_after", None))
    delta = _safe_float_or_none(observed_outcome.get("witness_energy_delta", None))
    if delta is None and before is not None and after is not None:
        delta = float(before) - float(after)
    has_observed = after is not None or delta is not None
    return {
        "before": before,
        "after": after,
        "delta": delta,
        "source": "observed_outcome" if has_observed else "observed_before_only",
        "observed_mask": 1.0 if has_observed else 0.0,
    }


def _augment_row_budget_labels(
    row: Mapping[str, Any],
    *,
    budget_ladder: Sequence[int],
    threshold_ladder: Sequence[float],
) -> dict[str, Any]:
    row_out = shared_opportunity_row_dict(
        row,
        route_source=(row.get("route_source", "") if isinstance(row, Mapping) else ""),
    )
    scheduler_decision_context_id = str(row_out.get("scheduler_decision_context_id", "") or "").strip()
    if scheduler_decision_context_id:
        route_context = str(row_out.get("decision_context_id", "") or row_out.get("decision_id", "") or "").strip()
        if route_context and not str(row_out.get("route_decision_context_id", "") or "").strip():
            row_out["route_decision_context_id"] = route_context
        row_out["decision_context_id"] = str(scheduler_decision_context_id)
    current_eff = _safe_float_or_none(
        row_out.get("current_best_route_eff_mse", row_out.get("parent_eff_mse", None))
    )
    decision_group_id = str(
        row_out.get("scheduler_decision_context_id", "")
        or row_out.get("decision_context_id", "")
        or row_out.get("decision_id", "")
        or ""
    )
    source_row_family_id = str(
        row_out.get("shadow_source_row_id", "")
        or row_out.get("source_row_id", "")
        or row_out.get("shadow_state_id", "")
        or row_out.get("opportunity_id", "")
    )
    row_out["decision_group_id"] = str(decision_group_id)
    row_out["source_row_family_id"] = str(source_row_family_id)
    row_out["route_family"] = str(row_out.get("route_source", "") or "")
    row_out["scheduler_budget_ladder"] = [int(v) for v in budget_ladder]
    row_out["scheduler_threshold_ladder"] = [float(v) for v in threshold_ladder]
    row_out.setdefault("global_best_eff_mse", current_eff)
    for budget in budget_ladder:
        exec_gain = _safe_float_or_none(row_out.get(f"expected_gain_at_budget_{int(budget)}_under_executor", None))
        oracle_gain = _safe_float_or_none(row_out.get(f"expected_gain_at_budget_{int(budget)}_under_oracle_executor", None))
        future_exec = _derive_future_best_eff(current_eff, exec_gain)
        future_oracle = _derive_future_best_eff(current_eff, oracle_gain)
        delta_exec = _delta_log_eff(current_eff, future_exec)
        delta_oracle = _delta_log_eff(current_eff, future_oracle)
        row_out[f"future_best_route_eff_mse_at_budget_{int(budget)}_under_executor"] = future_exec
        row_out[f"future_best_route_eff_mse_at_budget_{int(budget)}_under_oracle_executor"] = future_oracle
        row_out[f"delta_log_eff_at_budget_{int(budget)}"] = delta_exec
        row_out[f"delta_log_eff_at_budget_{int(budget)}_under_oracle_executor"] = delta_oracle
        for tau in threshold_ladder:
            tau_token = scheduler_threshold_token(tau)
            row_out[f"improve_ge_{tau_token}_at_budget_{int(budget)}"] = (
                None if delta_exec is None else (1.0 if float(delta_exec) >= float(tau) else 0.0)
            )
            row_out[f"tail_gain_{tau_token}_at_budget_{int(budget)}"] = (
                None if delta_exec is None else float(max(0.0, float(delta_exec) - float(tau)))
            )
        cost_exact, cost_exact_source, cost_exact_observed_mask = _derive_exact_cost_label(
            row_out,
            int(budget),
        )
        cost_wall, cost_wall_source, cost_wall_observed_mask = _derive_wall_cost_label(
            row_out,
            int(budget),
        )
        row_out[f"cost_exact_at_budget_{int(budget)}"] = float(cost_exact)
        row_out[f"cost_exact_label_source_at_budget_{int(budget)}"] = str(cost_exact_source)
        row_out[f"cost_exact_observed_mask_at_budget_{int(budget)}"] = float(cost_exact_observed_mask)
        row_out[f"cost_wall_at_budget_{int(budget)}"] = cost_wall
        row_out[f"cost_wall_label_source_at_budget_{int(budget)}"] = str(cost_wall_source)
        row_out[f"cost_wall_observed_mask_at_budget_{int(budget)}"] = float(cost_wall_observed_mask)
        cost_total, cost_total_source, cost_total_observed_mask = _derive_total_cost_label(
            row_out,
            int(budget),
            fallback_exact_cost=float(cost_exact),
        )
        row_out[f"cost_total_at_budget_{int(budget)}"] = float(cost_total)
        row_out[f"cost_total_label_source_at_budget_{int(budget)}"] = str(cost_total_source)
        row_out[f"cost_total_observed_mask_at_budget_{int(budget)}"] = float(cost_total_observed_mask)
        witness_label = _derive_witness_transition_label(row_out, int(budget))
        row_out[f"witness_energy_total_before_at_budget_{int(budget)}"] = witness_label["before"]
        row_out[f"witness_energy_total_after_at_budget_{int(budget)}"] = witness_label["after"]
        row_out[f"witness_energy_delta_at_budget_{int(budget)}"] = witness_label["delta"]
        row_out[f"witness_energy_label_source_at_budget_{int(budget)}"] = str(witness_label["source"])
        row_out[f"witness_energy_observed_mask_at_budget_{int(budget)}"] = float(witness_label["observed_mask"])
    return row_out


def _annotate_group_route_labels(
    rows: Sequence[dict[str, Any]],
    *,
    budget_ladder: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(scheduler_row_group_id(row), []).append(row)
    for group_rows in grouped.values():
        current_values = [
            _safe_float_or_none(row.get("current_best_route_eff_mse", None))
            for row in group_rows
        ]
        finite_current = [float(value) for value in current_values if value is not None]
        global_best_current = min(finite_current) if finite_current else None
        for row in group_rows:
            row["global_best_eff_mse"] = global_best_current
        for budget in budget_ladder:
            delta_values = [
                _safe_float_or_none(row.get(f"delta_log_eff_at_budget_{int(budget)}", None))
                for row in group_rows
            ]
            finite_pairs = [
                (idx, float(value))
                for idx, value in enumerate(delta_values)
                if value is not None
            ]
            if not finite_pairs:
                for row in group_rows:
                    row[f"route_win_at_budget_{int(budget)}"] = None
                    row[f"route_margin_log_eff_at_budget_{int(budget)}"] = None
                continue
            best_value = max(value for _idx, value in finite_pairs)
            for idx, row in enumerate(group_rows):
                value = delta_values[idx]
                if value is None:
                    row[f"route_win_at_budget_{int(budget)}"] = None
                    row[f"route_margin_log_eff_at_budget_{int(budget)}"] = None
                    continue
                other_values = [candidate for jdx, candidate in finite_pairs if int(jdx) != int(idx)]
                best_alt = max(other_values) if other_values else None
                row[f"route_win_at_budget_{int(budget)}"] = 1.0 if float(value) >= float(best_value) - 1.0e-9 else 0.0
                row[f"route_margin_log_eff_at_budget_{int(budget)}"] = (
                    None if best_alt is None else float(value) - float(best_alt)
                )
                if int(budget) == int(budget_ladder[0]):
                    future_other = [
                        _safe_float_or_none(other.get(f"future_best_route_eff_mse_at_budget_{int(budget)}_under_executor", None))
                        for jdx, other in enumerate(group_rows)
                        if int(jdx) != int(idx)
                    ]
                    finite_other = [float(item) for item in future_other if item is not None]
                    row["best_alt_route_eff_mse"] = min(finite_other) if finite_other else None
    return [dict(row) for row in rows]


def _annotate_group_integrity(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(scheduler_row_group_id(row), []).append(row)
    for group_id, group_rows in grouped.items():
        scheduler_contexts = {
            str(row.get("scheduler_decision_context_id", "") or "").strip()
            for row in group_rows
            if str(row.get("scheduler_decision_context_id", "") or "").strip()
        }
        route_contexts = {
            route_local_context_id(row)
            for row in group_rows
            if route_local_context_id(row)
        }
        if len(scheduler_contexts) > 1:
            grouping_source = "inconsistent_scheduler_context"
            degraded = True
            degraded_reason = "multiple_scheduler_contexts"
        elif scheduler_contexts:
            grouping_source = "scheduler_context"
            degraded = False
            degraded_reason = ""
        else:
            grouping_source = "route_local_fallback"
            degraded = True
            degraded_reason = "missing_scheduler_context"
        route_count = len({str(item.get("route_source", "") or "") for item in group_rows})
        for row in group_rows:
            row["decision_group_id"] = str(group_id)
            row["decision_grouping_source"] = str(grouping_source)
            row["decision_group_degraded"] = bool(degraded)
            row["decision_group_degraded_reason"] = str(degraded_reason)
            row["decision_group_route_count"] = int(route_count)
            row["decision_group_route_context_count"] = int(len(route_contexts))
    return [dict(row) for row in rows]


def summarize_scheduler_group_integrity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in list(rows or []):
        if not isinstance(row, Mapping):
            continue
        grouped.setdefault(scheduler_row_group_id(row), []).append(row)
    source_counts: dict[str, int] = {}
    degraded_groups = 0
    degraded_rows = 0
    for group_rows in grouped.values():
        row0 = group_rows[0] if group_rows else {}
        source = str(row0.get("decision_grouping_source", "") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        if bool(row0.get("decision_group_degraded", False)):
            degraded_groups += 1
            degraded_rows += len(group_rows)
    return {
        "n_decision_groups": int(len(grouped)),
        "n_degraded_decision_groups": int(degraded_groups),
        "n_rows_in_degraded_decision_groups": int(degraded_rows),
        "decision_grouping_source_counts": {
            str(key): int(value) for key, value in source_counts.items()
        },
    }


def augment_scheduler_shadow_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[Any] | None = None,
    threshold_ladder: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    budgets = normalize_scheduler_budget_ladder(budget_ladder)
    thresholds = normalize_scheduler_threshold_ladder(threshold_ladder)
    augmented = [
        _augment_row_budget_labels(row, budget_ladder=budgets, threshold_ladder=thresholds)
        for row in list(rows or [])
        if isinstance(row, Mapping)
    ]
    augmented = _annotate_group_route_labels(augmented, budget_ladder=budgets)
    return _annotate_group_integrity(augmented)


def build_scheduler_shadow_dataset(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[Any] | None = None,
    threshold_ladder: Sequence[Any] | None = None,
    include_repair: bool = True,
    include_build: bool = True,
    include_hole: bool = True,
    per_opportunity_timeout_s: float | None = None,
) -> dict[str, Any]:
    budgets = normalize_scheduler_budget_ladder(budget_ladder)
    thresholds = normalize_scheduler_threshold_ladder(threshold_ladder)
    base_rows, source_mode, source_meta = _coerce_scheduler_source_rows(
        payload,
        budget_ladder=budgets,
        include_repair=bool(include_repair),
        include_build=bool(include_build),
        include_hole=bool(include_hole),
        per_opportunity_timeout_s=per_opportunity_timeout_s,
    )
    base_rows, observed_cost_meta = _backfill_rows_with_scheduler_outcomes(base_rows, payload)
    rows = augment_scheduler_shadow_rows(
        base_rows,
        budget_ladder=budgets,
        threshold_ladder=thresholds,
    )
    return {
        "mode": SCHEDULER_DATASET_MODE,
        "source_mode": str(source_mode),
        "n_source_rows": int(source_meta.get("n_source_rows", len(base_rows)) or len(base_rows)),
        "n_rows": int(len(rows)),
        "budget_ladder": [int(v) for v in budgets],
        "threshold_ladder": [float(v) for v in thresholds],
        "rows": rows,
        "source_meta": {
            **dict(source_meta or {}),
            **dict(observed_cost_meta or {}),
            **summarize_scheduler_group_integrity(rows),
        },
    }


def load_scheduler_dataset_rows(dataset_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in dataset_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("mode", "") or "") == SCHEDULER_DATASET_MODE:
            dataset_rows = [dict(row) for row in list(payload.get("rows", []) or []) if isinstance(row, Mapping)]
            rows.extend(dataset_rows)
            continue
        rows.extend(
            build_scheduler_shadow_dataset(
                payload,
                budget_ladder=payload.get("budget_ladder", None),
                threshold_ladder=payload.get("threshold_ladder", None),
            ).get("rows", [])
        )
    return rows


__all__ = [
    "SCHEDULER_DATASET_MODE",
    "SCHEDULER_DEFAULT_BUDGET_LADDER",
    "SCHEDULER_DEFAULT_THRESHOLD_LADDER",
    "augment_scheduler_shadow_rows",
    "build_scheduler_shadow_dataset",
    "load_scheduler_dataset_rows",
    "normalize_scheduler_budget_ladder",
    "normalize_scheduler_threshold_ladder",
    "route_local_context_id",
    "scheduler_context_id",
    "scheduler_row_group_id",
    "summarize_scheduler_group_integrity",
    "scheduler_threshold_token",
]
