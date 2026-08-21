# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

from ..shared_opportunity import shared_opportunity_row_dict


DEFAULT_SHADOW_BUDGET_LADDER: tuple[int, ...] = (0, 1, 2, 4, 8)


def normalize_shadow_budget_ladder(budget_ladder: Sequence[Any] | None = None) -> tuple[int, ...]:
    values = budget_ladder if budget_ladder is not None else DEFAULT_SHADOW_BUDGET_LADDER
    cleaned = sorted({max(0, int(v)) for v in list(values or ())})
    if 0 not in cleaned:
        cleaned.insert(0, 0)
    return tuple(cleaned or (0,))


def _coerce_source_rows(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if isinstance(payload, Mapping):
        source_mode = str(payload.get("mode", "") or "")
        rows = [dict(row) for row in list(payload.get("rows", []) or []) if isinstance(row, Mapping)]
        return rows, source_mode
    return [dict(row) for row in list(payload or []) if isinstance(row, Mapping)], ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _path_tuple(path_like: Any) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in (path_like or ()))
    except Exception:
        return ()


def _stable_id(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def _shadow_source_row_id(row: Mapping[str, Any], row_index: int) -> str:
    row_map = row if isinstance(row, Mapping) else {}
    return str(
        row_map.get("repair_opportunity_slate_id", "")
        or row_map.get("inverse_repair_slate_id", "")
        or row_map.get("hole_opportunity_slate_id", "")
        or row_map.get("build_opportunity_slate_id", "")
        or row_map.get("controller_build_slate_id", "")
        or row_map.get("spec_id", "")
        or f"shadow_source_{int(row_index)}"
    )


def _child_expr(row: Mapping[str, Any] | None) -> str:
    row = row if isinstance(row, Mapping) else {}
    return str(row.get("child_expr", "") or row.get("child_key", "") or row.get("expr", "") or "")


def _candidate_eff_mse(row: Mapping[str, Any] | None) -> float | None:
    row = row if isinstance(row, Mapping) else {}
    return _safe_float_or_none(row.get("child_eff_mse", row.get("eff_mse", None)))


def _candidate_raw_mse(row: Mapping[str, Any] | None) -> float | None:
    row = row if isinstance(row, Mapping) else {}
    return _safe_float_or_none(row.get("child_raw_mse", row.get("raw_mse", None)))


def _candidate_uid(row: Mapping[str, Any] | None) -> str:
    row = row if isinstance(row, Mapping) else {}
    return str(
        row.get("child_key", "")
        or row.get("child_expr", "")
        or row.get("expr", "")
        or _stable_id(
            row.get("beam_rank", ""),
            row.get("local_rank", ""),
            row.get("action", ""),
            row.get("path", ()),
            row.get("target_mode", ""),
        )
    )


def _best_exact_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, str] | None = None
    for row in rows:
        eff = _candidate_eff_mse(row)
        if eff is None or not math.isfinite(eff):
            continue
        raw = _candidate_raw_mse(row)
        key = (
            float(eff),
            float("inf") if raw is None or not math.isfinite(raw) else float(raw),
            _candidate_uid(row),
        )
        if best_key is None or key < best_key:
            best = dict(row)
            best_key = key
    return best


def _parent_eff_mse(row: Mapping[str, Any] | None) -> float:
    row = row if isinstance(row, Mapping) else {}
    for key in (
        "estimated_parent_eff_mse",
        "parent_eff_mse",
        "parent_best_eff_mse",
        "candidate_probe_mse",
    ):
        value = _safe_float_or_none(row.get(key, None))
        if value is not None and math.isfinite(value):
            return float(value)
    return float("inf")


def _parent_depth(row: Mapping[str, Any] | None) -> int:
    row = row if isinstance(row, Mapping) else {}
    for key in ("truth_depth", "parent_depth"):
        value = row.get(key, None)
        try:
            return int(value)
        except Exception:
            continue
    return 0


def _preview_best_expr(opportunity_row: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, Any]]) -> str:
    explicit = str(opportunity_row.get("current_best_child_expr", "") or opportunity_row.get("best_preview_child_expr", "") or "")
    if explicit:
        return explicit
    best_preview = None
    best_key = None
    for row in candidate_rows:
        local_probe = _safe_float(row.get("local_probe_mse", float("inf")), float("inf"))
        local_fit = _safe_float(row.get("local_fit_mse", float("inf")), float("inf"))
        key = (
            float(row.get("tuple_allocation_estimate", row.get("tuple_combined_estimate", row.get("tuple_utility_estimate", float("-inf"))))),
            -local_probe,
            -local_fit,
            -_safe_int(row.get("local_rank", 0), 0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_preview = row
    return _child_expr(best_preview)


def _bool_hint(value: Any) -> float | None:
    if value is None:
        return None
    return 1.0 if bool(value) else 0.0


def _future_flag(rows: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    if not rows:
        return None
    seen = False
    any_true = False
    for row in rows:
        for key in keys:
            if key in row:
                seen = True
                if bool(row.get(key, False)):
                    any_true = True
    if not seen:
        return None
    return 1.0 if any_true else 0.0


def _route_best_eff(parent_eff: float, rows: Sequence[Mapping[str, Any]]) -> float:
    best_eff = parent_eff
    for row in rows:
        eff = _candidate_eff_mse(row)
        if eff is None or not math.isfinite(eff):
            continue
        if not math.isfinite(best_eff) or float(eff) < float(best_eff):
            best_eff = float(eff)
    return float(best_eff)


def _executor_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
    utility = _safe_float(
        row.get("tuple_allocation_estimate", row.get("tuple_combined_estimate", row.get("tuple_utility_estimate", float("-inf")))),
        float("-inf"),
    )
    regret = _safe_float(row.get("tuple_regret_estimate", float("inf")), float("inf"))
    local_probe = _safe_float(row.get("local_probe_mse", float("inf")), float("inf"))
    local_fit = _safe_float(row.get("local_fit_mse", float("inf")), float("inf"))
    return (
        float(utility),
        _safe_float(row.get("tuple_utility_estimate", utility), utility),
        -float(regret),
        -float(local_probe),
        -float(local_fit),
        -float(_safe_int(row.get("local_rank", 0), 0)),
    )


def _oracle_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
    eff = _candidate_eff_mse(row)
    raw = _candidate_raw_mse(row)
    return (
        float("inf") if eff is None or not math.isfinite(eff) else float(eff),
        float("inf") if raw is None or not math.isfinite(raw) else float(raw),
        _safe_int(row.get("local_rank", 0), 0),
        _candidate_uid(row),
    )


def _derived_repair_opportunity_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    decision_id = str(row.get("repair_opportunity_slate_id", "") or row.get("inverse_repair_slate_id", "") or _stable_id(row.get("parent_expr", "")))
    grouped: dict[tuple[int | None, tuple[int, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for slate_row in list(row.get("inverse_repair_slate", []) or []):
        if not isinstance(slate_row, Mapping):
            continue
        beam_rank = _safe_int(slate_row.get("beam_rank", -1), -1)
        key = (
            None if beam_rank < 0 else beam_rank,
            _path_tuple(slate_row.get("path", None)),
            str(slate_row.get("target_mode", "") or ""),
        )
        grouped[key].append(dict(slate_row))
    out: list[dict[str, Any]] = []
    for ordinal, ((beam_rank, path, target_mode), beam_rows) in enumerate(sorted(grouped.items(), key=lambda item: (999999 if item[0][0] is None else item[0][0], item[0][1], item[0][2]))):
        dedup_rows = [item for item in beam_rows if bool(item.get("dedup_kept", True))]
        preview_rows = dedup_rows if dedup_rows else beam_rows
        preview_best_expr = _preview_best_expr({}, preview_rows)
        out.append(shared_opportunity_row_dict({
            "route_source": "repair",
            "opportunity_type": "repair_beam",
            "decision_id": str(decision_id),
            "beam_id": f"{str(decision_id)}:{int(ordinal if beam_rank is None else beam_rank)}",
            "parent_expr": str(row.get("parent_expr", "") or ""),
            "action": "inv_steer",
            "path": [int(v) for v in path],
            "path_source": "derived_inverse_repair_slate",
            "target_mode": str(target_mode),
            "budget_exact_spent": 0,
            "budget_remaining": int(len([item for item in preview_rows if _candidate_eff_mse(item) is not None])),
            "current_best_child_expr": str(preview_best_expr),
            "current_best_child_eff_mse": None,
            "candidate_count_observed": int(len(preview_rows)),
            "candidate_count_unique": int(len(preview_rows)),
            "beam_rank": int(ordinal if beam_rank is None else beam_rank),
            "best_preview_child_expr": str(preview_best_expr),
            "best_preview_probe_mse": (
                None if not preview_rows else _safe_float_or_none(min(
                    (_safe_float_or_none(item.get("local_probe_mse", None)) for item in preview_rows if _safe_float_or_none(item.get("local_probe_mse", None)) is not None),
                    default=None,
                ))
            ),
            "best_preview_fit_mse": (
                None if not preview_rows else _safe_float_or_none(min(
                    (_safe_float_or_none(item.get("local_fit_mse", None)) for item in preview_rows if _safe_float_or_none(item.get("local_fit_mse", None)) is not None),
                    default=None,
                ))
            ),
            "path_gain": _safe_float(beam_rows[0].get("path_gain", 0.0), 0.0) if beam_rows else 0.0,
        }, route_source="repair"))
    return out


def _repair_opportunity_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = [dict(item) for item in list(row.get("repair_opportunity_slate", []) or []) if isinstance(item, Mapping)]
    return explicit if explicit else _derived_repair_opportunity_rows(row)


def _build_opportunity_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = [dict(item) for item in list(row.get("build_opportunity_slate", []) or []) if isinstance(item, Mapping)]
    if explicit:
        return explicit
    decision_id = str(row.get("build_opportunity_slate_id", "") or row.get("controller_build_slate_id", "") or _stable_id(row.get("parent_expr", ""), "build"))
    out: list[dict[str, Any]] = []
    for ordinal, slate_row in enumerate(list(row.get("controller_build_slate", []) or [])):
        if not isinstance(slate_row, Mapping):
            continue
        status = str(slate_row.get("status", "") or "")
        exact_observed = bool(slate_row.get("exact_child_score_observed", False))
        build_preview_only = bool(slate_row.get("build_preview_only", False))
        candidate_viable = status not in {"proposal_none", "dim_invalid", "dim_mismatch", "score_none"}
        out.append(shared_opportunity_row_dict({
            "route_source": "build",
            "opportunity_type": "build_action",
            "decision_id": str(decision_id),
            "beam_id": f"{str(decision_id)}:{int(ordinal)}",
            "parent_expr": str(row.get("parent_expr", "") or ""),
            "action": str(slate_row.get("action", "") or ""),
            "path": list(slate_row.get("path", []) or []),
            "path_source": str(slate_row.get("path_source", "") or ""),
            "budget_exact_spent": 0,
            "budget_remaining": 0 if exact_observed else (1 if candidate_viable or build_preview_only else 0),
            "current_best_child_expr": str(_child_expr(slate_row)),
            "current_best_child_eff_mse": None,
            "candidate_count_observed": 1 if exact_observed else 0,
            "candidate_count_unique": 1 if exact_observed else 0,
            "preview_candidate_count_total": 1 if candidate_viable or build_preview_only else 0,
            "preview_candidate_count_unique_total": 1 if candidate_viable or build_preview_only else 0,
            "status": status,
            "build_preview_only": bool(build_preview_only),
        }, route_source="build"))
    return out


def _hole_opportunity_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = [dict(item) for item in list(row.get("hole_opportunity_slate", []) or []) if isinstance(item, Mapping)]
    return explicit


def _hole_candidates_for_opportunity(opportunity_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    exact_eff = _safe_float_or_none(
        opportunity_row.get(
            "hole_best_exact_eff_mse",
            opportunity_row.get("current_best_child_eff_mse", None),
        )
    )
    evidence_level = str(opportunity_row.get("evidence_level", "") or "")
    exact_observed = _safe_int(opportunity_row.get("observed_exact_evals", 0), 0)
    if exact_eff is None or exact_observed <= 0 or evidence_level != "exact_known":
        return []
    child_expr = str(
        opportunity_row.get("current_best_child_expr", "")
        or opportunity_row.get("hole_best_child_expr", "")
        or opportunity_row.get("opportunity_id", "")
        or "hole_candidate"
    )
    return [{
        "child_key": str(opportunity_row.get("opportunity_id", "") or child_expr),
        "child_expr": child_expr,
        "child_eff_mse": float(exact_eff),
        "child_raw_mse": float(exact_eff),
        "local_rank": 0,
        "created_new_residual_basin": opportunity_row.get("hole_created_new_residual_basin", None),
        "oracle_mapping_fragile": opportunity_row.get("hole_mapping_fragile", None),
        "oracle_mapping_stable": opportunity_row.get("hole_mapping_stable", None),
    }]


def _repair_group_key_for_opportunity(row: Mapping[str, Any]) -> tuple[int | None, tuple[int, ...], str]:
    beam_rank = row.get("beam_rank", None)
    return (
        None if beam_rank is None else _safe_int(beam_rank, 0),
        _path_tuple(row.get("path", None)),
        str(row.get("target_mode", "") or ""),
    )


def _repair_group_key_for_candidate(row: Mapping[str, Any]) -> tuple[int | None, tuple[int, ...], str]:
    beam_rank = row.get("beam_rank", None)
    return (
        None if beam_rank is None else _safe_int(beam_rank, 0),
        _path_tuple(row.get("path", None)),
        str(row.get("target_mode", "") or ""),
    )


def _repair_candidates_by_key(row: Mapping[str, Any]) -> dict[tuple[int | None, tuple[int, ...], str], list[dict[str, Any]]]:
    grouped: dict[tuple[int | None, tuple[int, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for slate_row in list(row.get("inverse_repair_slate", []) or []):
        if not isinstance(slate_row, Mapping):
            continue
        if "dedup_kept" in slate_row and not bool(slate_row.get("dedup_kept", False)):
            continue
        grouped[_repair_group_key_for_candidate(slate_row)].append(dict(slate_row))
    return grouped


def _build_candidates_by_key(row: Mapping[str, Any]) -> dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    for slate_row in list(row.get("controller_build_slate", []) or []):
        if not isinstance(slate_row, Mapping):
            continue
        key = (str(slate_row.get("action", "") or ""), _path_tuple(slate_row.get("path", None)))
        grouped[key].append(dict(slate_row))
    return grouped


def _current_best_route_snapshot(parent_eff: float, revealed_rows: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    best_row = _best_exact_row(revealed_rows)
    best_route_eff = _route_best_eff(parent_eff, revealed_rows)
    return float(best_route_eff), best_row


def _future_gain_labels(
    *,
    parent_eff: float,
    current_revealed_rows: Sequence[Mapping[str, Any]],
    candidate_order: Sequence[Mapping[str, Any]],
    oracle_order: Sequence[Mapping[str, Any]],
    build_route_eff: float | None,
    budget_ladder: Sequence[int],
) -> dict[str, Any]:
    current_route_eff, _current_best_row = _current_best_route_snapshot(parent_eff, current_revealed_rows)
    current_ids = {_candidate_uid(row) for row in current_revealed_rows}
    unseen_oracle_rows = [row for row in oracle_order if _candidate_uid(row) not in current_ids]
    out: dict[str, Any] = {
        "current_best_route_eff_mse": None if not math.isfinite(current_route_eff) else float(current_route_eff),
    }
    for budget in budget_ladder:
        additional = max(0, int(budget))
        future_exec_rows = list(candidate_order[: min(len(candidate_order), len(current_revealed_rows) + additional)])
        future_exec_eff = _route_best_eff(parent_eff, future_exec_rows)
        exec_gain = 0.0
        if math.isfinite(current_route_eff) and math.isfinite(future_exec_eff):
            exec_gain = max(0.0, float(current_route_eff) - float(future_exec_eff))
        future_oracle_rows = list(current_revealed_rows) + unseen_oracle_rows[:additional]
        future_oracle_eff = _route_best_eff(parent_eff, future_oracle_rows)
        oracle_gain = 0.0
        if math.isfinite(current_route_eff) and math.isfinite(future_oracle_eff):
            oracle_gain = max(0.0, float(current_route_eff) - float(future_oracle_eff))
        best_exec_row = _best_exact_row(future_exec_rows)
        added_exec_rows = future_exec_rows[len(current_revealed_rows) :]
        out[f"coverage_at_budget_{int(additional)}"] = 1.0 if exec_gain > 0.0 else 0.0
        out[f"expected_gain_at_budget_{int(additional)}_under_executor"] = float(exec_gain)
        out[f"expected_gain_at_budget_{int(additional)}_under_oracle_executor"] = float(oracle_gain)
        out[f"cond_gain_at_budget_{int(additional)}_if_covered_under_executor"] = float(exec_gain) if exec_gain > 0.0 else 0.0
        out[f"cond_gain_at_budget_{int(additional)}_if_covered_under_oracle_executor"] = float(oracle_gain) if oracle_gain > 0.0 else 0.0
        if build_route_eff is not None and math.isfinite(build_route_eff) and math.isfinite(current_route_eff) and math.isfinite(future_exec_eff):
            out[f"route_flip_at_budget_{int(additional)}"] = (
                1.0 if current_route_eff >= float(build_route_eff) and future_exec_eff < float(build_route_eff) else 0.0
            )
        else:
            out[f"route_flip_at_budget_{int(additional)}"] = None
        out[f"new_residual_basin_at_budget_{int(additional)}"] = _future_flag(added_exec_rows, "created_new_residual_basin", "new_residual_basin", "oracle_new_residual_basin")
        if best_exec_row is None:
            out[f"fragility_at_budget_{int(additional)}"] = None
            out[f"stability_at_budget_{int(additional)}"] = None
        else:
            out[f"fragility_at_budget_{int(additional)}"] = _bool_hint(best_exec_row.get("oracle_mapping_fragile", None))
            out[f"stability_at_budget_{int(additional)}"] = _bool_hint(best_exec_row.get("oracle_mapping_stable", None))
    next_budget = 1 if 1 in set(int(v) for v in budget_ladder) else min(int(v) for v in budget_ladder if int(v) > 0) if any(int(v) > 0 for v in budget_ladder) else 0
    out["expected_gain_next_under_executor"] = float(out.get(f"expected_gain_at_budget_{int(next_budget)}_under_executor", 0.0) or 0.0)
    out["expected_gain_next_under_oracle_executor"] = float(out.get(f"expected_gain_at_budget_{int(next_budget)}_under_oracle_executor", 0.0) or 0.0)
    out["delta_gain_next_under_executor"] = float(out["expected_gain_next_under_executor"])
    out["delta_gain_next_under_oracle_executor"] = float(out["expected_gain_next_under_oracle_executor"])
    return out


def _opportunity_prefix_rows(
    *,
    source_row: Mapping[str, Any],
    source_row_index: int,
    source_row_id: str,
    route_source: str,
    opportunity_row: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    budget_ladder: Sequence[int],
    build_route_eff: float | None,
    dataset_source_mode: str,
    per_opportunity_timeout_s: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefix_rows: list[dict[str, Any]] = []
    start_time = time.monotonic()
    parent_eff = _parent_eff_mse(source_row)
    parent_depth = _parent_depth(source_row)
    preview_total = int(
        max(
            len(list(candidate_rows)),
            _safe_int(opportunity_row.get("preview_candidate_count_total", 0), 0),
            _safe_int(opportunity_row.get("preview_candidate_count_unique_total", 0), 0),
        )
    )
    exact_rows = [dict(row) for row in candidate_rows if _candidate_eff_mse(row) is not None]
    executor_order = sorted(exact_rows, key=_executor_sort_key, reverse=True)
    oracle_order = sorted(exact_rows, key=_oracle_sort_key)
    preview_best_expr = _preview_best_expr(opportunity_row, candidate_rows)
    existing_evidence = str(opportunity_row.get("evidence_level", "") or "")
    for prefix_index in range(0, len(executor_order) + 1):
        if per_opportunity_timeout_s is not None and per_opportunity_timeout_s > 0.0:
            if (time.monotonic() - start_time) > float(per_opportunity_timeout_s):
                break
        revealed_rows = list(executor_order[:prefix_index])
        revealed_best_route_eff, revealed_best_row = _current_best_route_snapshot(parent_eff, revealed_rows)
        current_best_expr = _child_expr(revealed_best_row) if revealed_best_row is not None else preview_best_expr
        current_best_eff = _candidate_eff_mse(revealed_best_row) if revealed_best_row is not None else None
        label_fields = _future_gain_labels(
            parent_eff=parent_eff,
            current_revealed_rows=revealed_rows,
            candidate_order=executor_order,
            oracle_order=oracle_order,
            build_route_eff=build_route_eff,
            budget_ladder=budget_ladder,
        )
        payload = dict(opportunity_row)
        payload.update({
            "route_source": str(route_source),
            "parent_expr": str(opportunity_row.get("parent_expr", source_row.get("parent_expr", "")) or ""),
            "parent_depth": int(parent_depth),
            "parent_eff_mse": None if not math.isfinite(parent_eff) else float(parent_eff),
            "budget_exact_spent": int(prefix_index),
            "budget_remaining": int(max(0, len(executor_order) - prefix_index)),
            "candidate_count_observed": int(prefix_index),
            "candidate_count_unique": int(prefix_index),
            "preview_candidate_count_total": int(preview_total),
            "preview_candidate_count_unique_total": int(preview_total),
            "current_best_child_expr": str(current_best_expr),
            "current_best_child_eff_mse": None if current_best_eff is None else float(current_best_eff),
            "current_best_route_eff_mse": None if not math.isfinite(revealed_best_route_eff) else float(revealed_best_route_eff),
            "current_exact_best_expr": "" if revealed_best_row is None else str(_child_expr(revealed_best_row)),
            "current_exact_best_eff_mse": None if current_best_eff is None else float(current_best_eff),
            "evidence_level": (
                str(existing_evidence)
                if existing_evidence
                else ("exact_known" if prefix_index > 0 else ("preview_support" if preview_total > 1 else "preview_only"))
            ),
            "shadow_prefix_index": int(prefix_index),
            "shadow_state_id": _stable_id(
                opportunity_row.get("opportunity_id", ""),
                source_row_index,
                route_source,
                prefix_index,
            ),
            "shadow_source_row_id": str(source_row_id),
            "shadow_source_row_index": int(source_row_index),
            "shadow_source_mode": str(dataset_source_mode or source_row.get("mode", "") or ""),
            "shadow_total_exact_available": int(len(executor_order)),
            "shadow_total_preview_available": int(preview_total),
            "shadow_executor_reveals_observed": int(prefix_index),
            "shadow_budget_ladder_max": int(max(budget_ladder or (0,))),
            "label_budget_origin": "additional_exact_tokens",
        })
        payload.update(label_fields)
        if route_source == "repair":
            payload["shadow_executor_order"] = [
                {
                    "child_key": _candidate_uid(item),
                    "child_expr": _child_expr(item),
                    "child_eff_mse": _candidate_eff_mse(item),
                }
                for item in executor_order
            ]
        prefix_rows.append(shared_opportunity_row_dict(payload, route_source=route_source))
    meta = {
        "route_source": str(route_source),
        "opportunity_id": str(opportunity_row.get("opportunity_id", "") or ""),
        "prefix_rows_emitted": int(len(prefix_rows)),
        "preview_candidate_count": int(preview_total),
        "exact_candidate_count": int(len(executor_order)),
        "timeout_hit": bool(per_opportunity_timeout_s is not None and per_opportunity_timeout_s > 0.0 and len(prefix_rows) < (len(executor_order) + 1)),
    }
    return prefix_rows, meta


def build_opportunity_shadow_dataset(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[Any] | None = None,
    include_repair: bool = True,
    include_build: bool = True,
    include_hole: bool = True,
    per_opportunity_timeout_s: float | None = None,
) -> dict[str, Any]:
    rows, source_mode = _coerce_source_rows(payload)
    normalized_budget_ladder = normalize_shadow_budget_ladder(budget_ladder)
    dataset_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for source_row_index, source_row in enumerate(rows):
        parent_eff = _parent_eff_mse(source_row)
        source_row_id = _shadow_source_row_id(source_row, source_row_index)
        build_candidate_groups = _build_candidates_by_key(source_row)
        build_route_rows = [item for group in build_candidate_groups.values() for item in group]
        build_route_eff = _route_best_eff(parent_eff, build_route_rows) if build_route_rows else None
        row_meta = {
            "source_row_index": int(source_row_index),
            "source_mode": str(source_mode),
            "source_row_id": str(source_row_id),
            "truth_depth": int(_parent_depth(source_row)),
            "repair_opportunity_count": 0,
            "build_opportunity_count": 0,
            "hole_opportunity_count": 0,
            "prefix_rows_emitted": 0,
            "timeout_hit": False,
        }
        if include_repair:
            repair_candidate_groups = _repair_candidates_by_key(source_row)
            for opportunity_row in _repair_opportunity_rows(source_row):
                key = _repair_group_key_for_opportunity(opportunity_row)
                candidate_rows = repair_candidate_groups.get(key, [])
                prefix_rows, prefix_meta = _opportunity_prefix_rows(
                    source_row=source_row,
                    source_row_index=source_row_index,
                    source_row_id=source_row_id,
                    route_source="repair",
                    opportunity_row=opportunity_row,
                    candidate_rows=candidate_rows,
                    budget_ladder=normalized_budget_ladder,
                    build_route_eff=build_route_eff,
                    dataset_source_mode=source_mode,
                    per_opportunity_timeout_s=per_opportunity_timeout_s,
                )
                dataset_rows.extend(prefix_rows)
                row_meta["repair_opportunity_count"] = int(row_meta["repair_opportunity_count"]) + 1
                row_meta["prefix_rows_emitted"] = int(row_meta["prefix_rows_emitted"]) + int(prefix_meta.get("prefix_rows_emitted", 0))
                row_meta["timeout_hit"] = bool(row_meta["timeout_hit"]) or bool(prefix_meta.get("timeout_hit", False))
                meta_rows.append({
                    "source_row_index": int(source_row_index),
                    "route_source": "repair",
                    **prefix_meta,
                })
        if include_build:
            build_candidates_by_key = _build_candidates_by_key(source_row)
            for opportunity_row in _build_opportunity_rows(source_row):
                key = (
                    str(opportunity_row.get("action", "") or ""),
                    _path_tuple(opportunity_row.get("path", None)),
                )
                candidate_rows = build_candidates_by_key.get(key, [])
                prefix_rows, prefix_meta = _opportunity_prefix_rows(
                    source_row=source_row,
                    source_row_index=source_row_index,
                    source_row_id=source_row_id,
                    route_source="build",
                    opportunity_row=opportunity_row,
                    candidate_rows=candidate_rows,
                    budget_ladder=normalized_budget_ladder,
                    build_route_eff=None,
                    dataset_source_mode=source_mode,
                    per_opportunity_timeout_s=per_opportunity_timeout_s,
                )
                dataset_rows.extend(prefix_rows)
                row_meta["build_opportunity_count"] = int(row_meta["build_opportunity_count"]) + 1
                row_meta["prefix_rows_emitted"] = int(row_meta["prefix_rows_emitted"]) + int(prefix_meta.get("prefix_rows_emitted", 0))
                row_meta["timeout_hit"] = bool(row_meta["timeout_hit"]) or bool(prefix_meta.get("timeout_hit", False))
                meta_rows.append({
                    "source_row_index": int(source_row_index),
                    "route_source": "build",
                    **prefix_meta,
                })
        if include_hole:
            for opportunity_row in _hole_opportunity_rows(source_row):
                candidate_rows = _hole_candidates_for_opportunity(opportunity_row)
                prefix_rows, prefix_meta = _opportunity_prefix_rows(
                    source_row=source_row,
                    source_row_index=source_row_index,
                    source_row_id=source_row_id,
                    route_source="hole",
                    opportunity_row=opportunity_row,
                    candidate_rows=candidate_rows,
                    budget_ladder=normalized_budget_ladder,
                    build_route_eff=build_route_eff,
                    dataset_source_mode=source_mode,
                    per_opportunity_timeout_s=per_opportunity_timeout_s,
                )
                dataset_rows.extend(prefix_rows)
                row_meta["hole_opportunity_count"] = int(row_meta["hole_opportunity_count"]) + 1
                row_meta["prefix_rows_emitted"] = int(row_meta["prefix_rows_emitted"]) + int(prefix_meta.get("prefix_rows_emitted", 0))
                row_meta["timeout_hit"] = bool(row_meta["timeout_hit"]) or bool(prefix_meta.get("timeout_hit", False))
                meta_rows.append({
                    "source_row_index": int(source_row_index),
                    "route_source": "hole",
                    **prefix_meta,
                })
        meta_rows.append(row_meta)
    return {
        "mode": "opportunity_shadow_dataset",
        "source_mode": str(source_mode),
        "n_source_rows": int(len(rows)),
        "n_rows": int(len(dataset_rows)),
        "budget_ladder": [int(v) for v in normalized_budget_ladder],
        "rows": dataset_rows,
        "meta_rows": meta_rows,
    }


__all__ = [
    "DEFAULT_SHADOW_BUDGET_LADDER",
    "build_opportunity_shadow_dataset",
    "normalize_shadow_budget_ladder",
]
