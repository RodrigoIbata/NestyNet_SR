# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .scheduler_dataset import (
    SCHEDULER_DEFAULT_BUDGET_LADDER,
    SCHEDULER_DEFAULT_THRESHOLD_LADDER,
    augment_scheduler_shadow_rows,
    load_scheduler_dataset_rows,
    normalize_scheduler_budget_ladder,
    normalize_scheduler_threshold_ladder,
    scheduler_row_group_id as dataset_scheduler_row_group_id,
    scheduler_threshold_token,
)
from ..shared_opportunity import (
    SHARED_OPPORTUNITY_MASK_FIELD_NAMES,
    normalize_witness_energy_fields,
    shared_opportunity_row_dict,
)


SCHEDULER_MODEL_KIND = "opportunity_scheduler_v2"
SCHEDULER_FEATURE_SCHEMA_VERSION = 2
SCHEDULER_OBJECTIVE_MODES: tuple[str, ...] = ("acquisition", "witness", "hybrid")
SCHEDULER_DEFAULT_OBJECTIVE_MODE = "acquisition"
SCHEDULER_DEFAULT_HYBRID_OBJECTIVE_MIX = 0.5
SCHEDULER_ROUTE_NAMES: tuple[str, ...] = ("repair", "build", "hole")
SCHEDULER_TYPE_NAMES: tuple[str, ...] = (
    "repair_beam",
    "build_action",
    "hole_frontier",
    "hole_expand",
    "hole_opportunity",
)
SCHEDULER_ACTION_NAMES: tuple[str, ...] = (
    "inv_steer",
    "repair_option",
    "replace",
    "wrap_un",
    "residual",
    "boost",
    "crossover",
    "hole_search",
    "hole_expand",
    "spec_expand",
)
SCHEDULER_MODE_NAMES: tuple[str, ...] = ("identity", "full", "affine")
SCHEDULER_PATH_SOURCE_NAMES: tuple[str, ...] = (
    "inverse",
    "critic",
    "derived_inverse_repair_slate",
    "guided",
    "random",
    "hole_frontier",
)
SCHEDULER_EVIDENCE_NAMES: tuple[str, ...] = ("preview_only", "preview_support", "exact_known")
SCHEDULER_METHOD_NAMES: tuple[str, ...] = (
    "inverse",
    "build",
    "hole",
    "hole_frontier",
    "hole_expand",
)
SCHEDULER_SUBROUTE_NAMES: tuple[str, ...] = (
    "beam",
    "tuple",
    "frontier",
    "recursive",
    "direct",
)
SCHEDULER_NUMERIC_FEATURE_NAMES: tuple[str, ...] = (
    "parent_depth",
    "parent_log_eff_mse",
    "current_best_route_log_eff_mse",
    "current_best_child_log_eff_mse",
    "global_best_log_eff_mse",
    "best_alt_route_log_eff_mse",
    "budget_exact_spent",
    "budget_remaining",
    "budget_widen_spent",
    "budget_micro_spent",
    "candidate_count_observed",
    "candidate_count_unique",
    "preview_candidate_count_total",
    "preview_candidate_count_unique_total",
    "shadow_total_exact_available",
    "shadow_total_preview_available",
    "shadow_executor_reveals_observed",
    "best_preview_log_probe_mse",
    "best_preview_log_fit_mse",
    "best_tuple_utility_estimate",
    "best_tuple_allocation_estimate",
    "path_length",
    "path_gain",
    "path_gain_pre_cut",
    "rel_gain",
    "transport_rel",
    "lin_rel",
    "valid_frac",
    "confidence",
    "effective_n",
    "branch_factor",
    "cut_factor",
    "branch_support",
    "family_scale",
    "observed_wall_seconds_log",
    "observed_exact_evals_log",
    "observed_preview_evals_log",
    "observed_micro_tokens_log",
    "observed_widen_tokens_log",
    "predicted_value",
    "predicted_cost",
    "preview_solvability",
    "preview_recursive_depth",
)
SCHEDULER_WITNESS_FEATURE_NAMES: tuple[str, ...] = (
    "witness_value_log_loss",
    "witness_grad_log_loss",
    "witness_d2_log_loss",
    "witness_diag_log_loss",
    "witness_physics_log_loss",
    "witness_energy_total_log",
    "witness_energy_delta_estimate",
    "witness_grad_present",
    "witness_d2_present",
    "witness_diag_present",
    "witness_physics_present",
)
SCHEDULER_FEATURE_NAMES: tuple[str, ...] = (
    *SCHEDULER_NUMERIC_FEATURE_NAMES,
    *SHARED_OPPORTUNITY_MASK_FIELD_NAMES,
    *tuple(f"route_is_{name}" for name in SCHEDULER_ROUTE_NAMES),
    "route_is_other",
    *tuple(f"type_is_{name}" for name in SCHEDULER_TYPE_NAMES),
    "type_is_other",
    *tuple(f"action_is_{name}" for name in SCHEDULER_ACTION_NAMES),
    "action_is_other",
    *tuple(f"mode_is_{name}" for name in SCHEDULER_MODE_NAMES),
    "mode_is_other",
    *tuple(f"path_source_is_{name}" for name in SCHEDULER_PATH_SOURCE_NAMES),
    "path_source_is_other",
    *tuple(f"evidence_is_{name}" for name in SCHEDULER_EVIDENCE_NAMES),
    "evidence_is_other",
    *tuple(f"method_is_{name}" for name in SCHEDULER_METHOD_NAMES),
    "method_is_other",
    *tuple(f"subroute_is_{name}" for name in SCHEDULER_SUBROUTE_NAMES),
    "subroute_is_other",
)


def scheduler_uses_witness_energy_features(feature_names: Sequence[str] | None) -> bool:
    feature_set = {str(name) for name in list(feature_names or ())}
    return any(name in feature_set for name in SCHEDULER_WITNESS_FEATURE_NAMES)


def scheduler_feature_names(*, witness_energy_feature_enable: bool = False) -> tuple[str, ...]:
    if bool(witness_energy_feature_enable):
        return (*SCHEDULER_FEATURE_NAMES, *SCHEDULER_WITNESS_FEATURE_NAMES)
    return tuple(SCHEDULER_FEATURE_NAMES)
SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS: dict[str, float] = {
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


def _normalize_route_aliases(route_aliases: Mapping[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(route_aliases, Mapping):
        return out
    valid = set(SCHEDULER_ROUTE_NAMES)
    for key, value in dict(route_aliases).items():
        src = str(key or "").strip().lower()
        dst = str(value or "").strip().lower()
        if src in valid and dst in valid and src != dst:
            out[src] = dst
    return out


def _canonical_route_source(route_source: str, route_aliases: Mapping[str, Any] | None = None) -> str:
    route = str(route_source or "").strip().lower()
    return str(_normalize_route_aliases(route_aliases).get(route, route))


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_scheduler_objective_mode(value: Any, *, default: str = SCHEDULER_DEFAULT_OBJECTIVE_MODE) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in SCHEDULER_OBJECTIVE_MODES:
        return str(token)
    return str(default)


def _normalize_hybrid_objective_mix(value: Any, *, default: float = SCHEDULER_DEFAULT_HYBRID_OBJECTIVE_MIX) -> float:
    try:
        mix = float(value)
    except Exception:
        mix = float(default)
    return float(min(1.0, max(0.0, mix)))


def _log1p_nonneg(value: Any) -> float:
    vv = _safe_float_or_none(value)
    if vv is None:
        return 0.0
    return float(math.log1p(max(0.0, float(vv))))


def _one_hot(value: str, allowed: Sequence[str], prefix: str) -> dict[str, float]:
    token = str(value or "")
    out = {f"{prefix}{name}": 0.0 for name in allowed}
    other_name = f"{prefix}other"
    if token in allowed:
        out[f"{prefix}{token}"] = 1.0
    else:
        out[other_name] = 1.0
    return out


def _threshold_token_value(token: str) -> float | None:
    text = str(token or "").strip()
    if not text:
        return None
    if text.startswith("m"):
        text = "-" + text[1:]
    text = text.replace("p", ".")
    try:
        return float(text)
    except Exception:
        return None


def _row_has_scheduler_labels(row: Mapping[str, Any]) -> bool:
    if not isinstance(row, Mapping):
        return False
    if "scheduler_budget_ladder" in row:
        return True
    return any(str(key).startswith("delta_log_eff_at_budget_") for key in row.keys())


def _extract_budget_ladder(
    rows: Sequence[Mapping[str, Any]],
    provided: Sequence[Any] | None = None,
) -> tuple[int, ...]:
    if provided is not None:
        return normalize_scheduler_budget_ladder(provided)
    values: set[int] = set()
    for row in rows:
        for item in list(row.get("scheduler_budget_ladder", []) or []):
            values.add(max(1, int(item)))
        for key in row.keys():
            key_str = str(key)
            if "_at_budget_" not in key_str:
                continue
            try:
                values.add(max(1, int(key_str.rsplit("_", 1)[-1])))
            except Exception:
                continue
    return normalize_scheduler_budget_ladder(sorted(values) if values else None)


def _extract_threshold_ladder(
    rows: Sequence[Mapping[str, Any]],
    provided: Sequence[Any] | None = None,
) -> tuple[float, ...]:
    if provided is not None:
        return normalize_scheduler_threshold_ladder(provided)
    values: set[float] = set()
    for row in rows:
        for item in list(row.get("scheduler_threshold_ladder", []) or []):
            try:
                values.add(float(item))
            except Exception:
                continue
        for key in row.keys():
            key_str = str(key)
            if not key_str.startswith("improve_ge_") or "_at_budget_" not in key_str:
                continue
            token = key_str[len("improve_ge_"):].split("_at_budget_", 1)[0]
            value = _threshold_token_value(token)
            if value is not None:
                values.add(float(value))
    return normalize_scheduler_threshold_ladder(sorted(values) if values else None)


def _coerce_prediction_rows(rows_or_row: Any) -> list[dict[str, Any]]:
    if isinstance(rows_or_row, Mapping):
        row = dict(rows_or_row)
        if "opportunity_id" in row or "shared_opportunity_schema_version" in row:
            return [shared_opportunity_row_dict(row, route_source=row.get("route_source", ""))]
        out: list[dict[str, Any]] = []
        for route_source, key in (
            ("repair", "repair_opportunity_slate"),
            ("build", "build_opportunity_slate"),
            ("hole", "hole_opportunity_slate"),
        ):
            for item in list(row.get(key, []) or []):
                if isinstance(item, Mapping):
                    out.append(shared_opportunity_row_dict(item, route_source=route_source))
        return out
    return [
        shared_opportunity_row_dict(item, route_source=(item.get("route_source", "") if isinstance(item, Mapping) else ""))
        for item in list(rows_or_row or [])
        if isinstance(item, Mapping)
    ]


def _ensure_scheduler_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[Any] | None = None,
    threshold_ladder: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    base_rows = [
        dict(row)
        for row in list(rows or [])
        if isinstance(row, Mapping)
    ]
    if not base_rows:
        return []
    budgets = _extract_budget_ladder(base_rows, budget_ladder)
    thresholds = _extract_threshold_ladder(base_rows, threshold_ladder)
    if all(_row_has_scheduler_labels(row) for row in base_rows):
        return [
            shared_opportunity_row_dict(row, route_source=(row.get("route_source", "") if isinstance(row, Mapping) else ""))
            for row in base_rows
        ]
    return augment_scheduler_shadow_rows(
        base_rows,
        budget_ladder=budgets,
        threshold_ladder=thresholds,
    )


def _route_index(route_source: str) -> int:
    try:
        return SCHEDULER_ROUTE_NAMES.index(str(route_source or ""))
    except ValueError:
        return len(SCHEDULER_ROUTE_NAMES)


def scheduler_feature_vector(
    row: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = SCHEDULER_FEATURE_NAMES,
    route_aliases: Mapping[str, Any] | None = None,
) -> list[float]:
    item = shared_opportunity_row_dict(row, route_source=row.get("route_source", "") if isinstance(row, Mapping) else "")
    route_source = _canonical_route_source(str(item.get("route_source", "") or ""), route_aliases)
    opportunity_type = str(item.get("opportunity_type", "") or "")
    action = str(item.get("action", "") or "")
    target_mode = str(item.get("target_mode", "") or "")
    path_source = str(item.get("path_source", "") or "")
    evidence_level = str(item.get("evidence_level", "") or "")
    method_name = str(item.get("method_name", "") or "")
    subroute = str(item.get("subroute", "") or "")
    path = list(item.get("path", []) or [])
    witness = normalize_witness_energy_fields(item)
    raw_features: dict[str, float] = {
        "parent_depth": float(_safe_int(item.get("parent_depth", 0), 0)),
        "parent_log_eff_mse": _log1p_nonneg(item.get("parent_eff_mse", item.get("estimated_parent_eff_mse", None))),
        "current_best_route_log_eff_mse": _log1p_nonneg(item.get("current_best_route_eff_mse", None)),
        "current_best_child_log_eff_mse": _log1p_nonneg(item.get("current_best_child_eff_mse", None)),
        "global_best_log_eff_mse": _log1p_nonneg(item.get("global_best_eff_mse", None)),
        "best_alt_route_log_eff_mse": _log1p_nonneg(item.get("best_alt_route_eff_mse", None)),
        "budget_exact_spent": float(_safe_int(item.get("budget_exact_spent", 0), 0)),
        "budget_remaining": float(_safe_int(item.get("budget_remaining", 0), 0)),
        "budget_widen_spent": float(_safe_int(item.get("budget_widen_spent", 0), 0)),
        "budget_micro_spent": float(_safe_int(item.get("budget_micro_spent", 0), 0)),
        "candidate_count_observed": float(_safe_int(item.get("candidate_count_observed", 0), 0)),
        "candidate_count_unique": float(_safe_int(item.get("candidate_count_unique", 0), 0)),
        "preview_candidate_count_total": float(_safe_int(item.get("preview_candidate_count_total", 0), 0)),
        "preview_candidate_count_unique_total": float(_safe_int(item.get("preview_candidate_count_unique_total", 0), 0)),
        "shadow_total_exact_available": float(_safe_int(item.get("shadow_total_exact_available", 0), 0)),
        "shadow_total_preview_available": float(_safe_int(item.get("shadow_total_preview_available", 0), 0)),
        "shadow_executor_reveals_observed": float(_safe_int(item.get("shadow_executor_reveals_observed", 0), 0)),
        "best_preview_log_probe_mse": _log1p_nonneg(item.get("best_preview_probe_mse", None)),
        "best_preview_log_fit_mse": _log1p_nonneg(item.get("best_preview_fit_mse", None)),
        "best_tuple_utility_estimate": _safe_float(item.get("best_tuple_utility_estimate", 0.0), 0.0),
        "best_tuple_allocation_estimate": _safe_float(item.get("best_tuple_allocation_estimate", 0.0), 0.0),
        "path_length": float(len(path)),
        "path_gain": _safe_float(item.get("path_gain", 0.0), 0.0),
        "path_gain_pre_cut": _safe_float(item.get("path_gain_pre_cut", 0.0), 0.0),
        "rel_gain": _safe_float(item.get("rel_gain", 0.0), 0.0),
        "transport_rel": _safe_float(item.get("transport_rel", 0.0), 0.0),
        "lin_rel": _safe_float(item.get("lin_rel", 0.0), 0.0),
        "valid_frac": _safe_float(item.get("valid_frac", 0.0), 0.0),
        "confidence": _safe_float(item.get("confidence", 0.0), 0.0),
        "effective_n": _safe_float(item.get("effective_n", 0.0), 0.0),
        "branch_factor": _safe_float(item.get("branch_factor", 0.0), 0.0),
        "cut_factor": _safe_float(item.get("cut_factor", 0.0), 0.0),
        "branch_support": _safe_float(item.get("branch_support", 0.0), 0.0),
        "family_scale": _safe_float(item.get("family_scale", 0.0), 0.0),
        "observed_wall_seconds_log": _log1p_nonneg(item.get("observed_wall_seconds", None)),
        "observed_exact_evals_log": _log1p_nonneg(item.get("observed_exact_evals", None)),
        "observed_preview_evals_log": _log1p_nonneg(item.get("observed_preview_evals", None)),
        "observed_micro_tokens_log": _log1p_nonneg(item.get("observed_micro_tokens", None)),
        "observed_widen_tokens_log": _log1p_nonneg(item.get("observed_widen_tokens", None)),
        "predicted_value": _safe_float(
            item.get("predicted_value", item.get("hole_search_predicted_value", 0.0)),
            0.0,
        ),
        "predicted_cost": _safe_float(
            item.get("predicted_cost", item.get("hole_search_predicted_cost", 0.0)),
            0.0,
        ),
        "preview_solvability": _safe_float(
            item.get("preview_solvability", item.get("hole_search_preview_solvability", 0.0)),
            0.0,
        ),
        "preview_recursive_depth": _safe_float(
            item.get("preview_recursive_depth", item.get("hole_search_recursion_depth", 0.0)),
            0.0,
        ),
        "witness_value_log_loss": _log1p_nonneg(witness.get("witness_value_loss", None)),
        "witness_grad_log_loss": _log1p_nonneg(witness.get("witness_grad_loss", None)),
        "witness_d2_log_loss": _log1p_nonneg(witness.get("witness_d2_loss", None)),
        "witness_diag_log_loss": _log1p_nonneg(witness.get("witness_diag_loss", None)),
        "witness_physics_log_loss": _log1p_nonneg(witness.get("witness_physics_loss", None)),
        "witness_energy_total_log": _log1p_nonneg(witness.get("witness_energy_total", None)),
        "witness_energy_delta_estimate": _safe_float(witness.get("witness_energy_delta_estimate", 0.0), 0.0),
        "witness_grad_present": 1.0 if _safe_float_or_none(witness.get("witness_grad_loss", None)) is not None else 0.0,
        "witness_d2_present": 1.0 if _safe_float_or_none(witness.get("witness_d2_loss", None)) is not None else 0.0,
        "witness_diag_present": 1.0 if _safe_float_or_none(witness.get("witness_diag_loss", None)) is not None else 0.0,
        "witness_physics_present": 1.0 if _safe_float_or_none(witness.get("witness_physics_loss", None)) is not None else 0.0,
    }
    for name in SHARED_OPPORTUNITY_MASK_FIELD_NAMES:
        raw_features[name] = 1.0 if bool(item.get(name, False)) else 0.0
    raw_features.update(_one_hot(route_source, SCHEDULER_ROUTE_NAMES, "route_is_"))
    raw_features.update(_one_hot(opportunity_type, SCHEDULER_TYPE_NAMES, "type_is_"))
    raw_features.update(_one_hot(action, SCHEDULER_ACTION_NAMES, "action_is_"))
    raw_features.update(_one_hot(target_mode, SCHEDULER_MODE_NAMES, "mode_is_"))
    raw_features.update(_one_hot(path_source, SCHEDULER_PATH_SOURCE_NAMES, "path_source_is_"))
    raw_features.update(_one_hot(evidence_level, SCHEDULER_EVIDENCE_NAMES, "evidence_is_"))
    raw_features.update(_one_hot(method_name, SCHEDULER_METHOD_NAMES, "method_is_"))
    raw_features.update(_one_hot(subroute, SCHEDULER_SUBROUTE_NAMES, "subroute_is_"))
    return [float(raw_features.get(name, 0.0)) for name in feature_names]


def scheduler_row_group_id(row: Mapping[str, Any]) -> str:
    return dataset_scheduler_row_group_id(row)


def split_scheduler_rows_grouped(
    rows: Sequence[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(rows)
    if n <= 1:
        return list(rows), []
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(scheduler_row_group_id(row), []).append(int(idx))
    group_items = list(groups.items())
    if len(group_items) <= 1:
        return list(rows), []
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(len(group_items), generator=generator).tolist()
    target_val_rows = max(1, min(n - 1, int(round(float(val_fraction) * float(n)))))
    val_group_positions: set[int] = set()
    val_count = 0
    for position in perm:
        if len(val_group_positions) >= max(1, len(group_items) - 1):
            break
        if val_count >= target_val_rows:
            break
        val_group_positions.add(int(position))
        val_count += len(group_items[int(position)][1])
    if not val_group_positions:
        val_group_positions.add(int(perm[0]))
    if len(val_group_positions) >= len(group_items):
        val_group_positions.remove(int(perm[-1]))
    val_idx = {
        int(idx)
        for position, (_group_id, group_indices) in enumerate(group_items)
        if int(position) in val_group_positions
        for idx in group_indices
    }
    train_rows = [dict(row) for idx, row in enumerate(rows) if idx not in val_idx]
    val_rows = [dict(row) for idx, row in enumerate(rows) if idx in val_idx]
    return train_rows, val_rows


def _target_tensor(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    mask_key: str | None = None,
    allow_finite_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    values: list[float] = []
    mask: list[float] = []
    for row in rows:
        value = row.get(key, None)
        if value is None:
            values.append(0.0)
            mask.append(0.0)
            continue
        try:
            values.append(float(value))
            mask_value = 1.0
            if mask_key:
                raw_mask_value = row.get(mask_key, None)
                if raw_mask_value is None:
                    mask_value = 1.0
                else:
                    try:
                        raw_mask_float = float(raw_mask_value)
                        if raw_mask_float > 0.5:
                            mask_value = 1.0
                        elif allow_finite_fallback and math.isfinite(float(value)):
                            mask_value = 1.0
                        else:
                            mask_value = 0.0
                    except Exception:
                        mask_value = 1.0 if allow_finite_fallback and math.isfinite(float(value)) else 0.0
            mask.append(float(mask_value))
        except Exception:
            values.append(0.0)
            mask.append(0.0)
    return torch.tensor(values, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor, *, weight: torch.Tensor | None = None) -> torch.Tensor:
    effective_mask = mask if weight is None else (mask * weight)
    denom = torch.clamp(effective_mask.sum(), min=1.0)
    return (loss * effective_mask).sum() / denom


def _normalize_inputs(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / torch.clamp(std, min=1.0e-6)


def _feature_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std > 1.0e-6, std, torch.ones_like(std))
    return mean, std


def _route_name(row: Mapping[str, Any]) -> str:
    return str(row.get("route_source", "") or row.get("route_family", "") or "").strip().lower()


def _depth_bucket_label(value: Any) -> str:
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


def _normalize_route_weight_map(weights: Mapping[str, Any] | None) -> dict[str, float]:
    out = {name: 1.0 for name in SCHEDULER_ROUTE_NAMES}
    if not isinstance(weights, Mapping):
        return out
    for key, value in dict(weights).items():
        route = str(key or "").strip().lower()
        if route in out:
            out[route] = max(0.0, _safe_float(value, 1.0))
    return out


def _normalize_budget_weight_map(
    weights: Mapping[Any, Any] | None,
    *,
    budget_ladder: Sequence[int],
) -> dict[int, float]:
    out = {int(budget): 1.0 for budget in list(budget_ladder or [])}
    if not isinstance(weights, Mapping):
        return out
    for key, value in dict(weights).items():
        budget = _safe_int(key, -1)
        if budget in out:
            out[int(budget)] = max(0.0, _safe_float(value, 1.0))
    return out


def _is_deep_repair_row(row: Mapping[str, Any], *, min_depth: int) -> bool:
    return _route_name(row) == "repair" and _safe_int(row.get("parent_depth", -1), -1) >= int(min_depth)


def _row_training_weight(
    row: Mapping[str, Any],
    *,
    route_weights: Mapping[str, float],
    deep_repair_min_depth: int,
    deep_repair_weight: float,
) -> float:
    route = _route_name(row)
    weight = float(route_weights.get(route, 1.0))
    if _is_deep_repair_row(row, min_depth=deep_repair_min_depth):
        weight *= max(0.0, float(deep_repair_weight))
    return float(max(0.0, weight))


def _build_training_weight_tensors(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget_ladder: Sequence[int],
    route_weights: Mapping[str, float] | None = None,
    budget_weights: Mapping[int, float] | None = None,
    deep_repair_min_depth: int = 5,
    deep_repair_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    route_map = _normalize_route_weight_map(route_weights)
    budget_map = _normalize_budget_weight_map(budget_weights, budget_ladder=budget_ladder)
    row_weights: list[float] = []
    budget_matrix: list[list[float]] = []
    for row in list(rows or []):
        base_weight = _row_training_weight(
            row,
            route_weights=route_map,
            deep_repair_min_depth=int(deep_repair_min_depth),
            deep_repair_weight=float(deep_repair_weight),
        )
        row_weights.append(float(base_weight))
        budget_matrix.append([
            float(base_weight) * float(budget_map.get(int(budget), 1.0))
            for budget in budget_ladder
        ])
    row_tensor = torch.tensor(row_weights, dtype=torch.float32)
    budget_tensor = torch.tensor(budget_matrix, dtype=torch.float32) if budget_matrix else torch.zeros((len(rows), 0), dtype=torch.float32)
    return row_tensor, budget_tensor


def _oversample_training_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    hole_oversample_repeat: int = 1,
    deep_repair_oversample_repeat: int = 1,
    deep_repair_min_depth: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    repeat_hist: dict[str, int] = {}
    route_counts_before: dict[str, int] = {}
    route_counts_after: dict[str, int] = {}
    for row in list(rows or []):
        if not isinstance(row, Mapping):
            continue
        route = _route_name(row)
        route_counts_before[route] = route_counts_before.get(route, 0) + 1
        repeat = 1
        if route == "hole":
            repeat = max(repeat, max(1, int(hole_oversample_repeat)))
        if _is_deep_repair_row(row, min_depth=int(deep_repair_min_depth)):
            repeat = max(repeat, max(1, int(deep_repair_oversample_repeat)))
        repeat_hist[str(repeat)] = repeat_hist.get(str(repeat), 0) + 1
        for _ in range(max(1, int(repeat))):
            row_copy = dict(row)
            row_copy["training_oversample_repeat"] = int(repeat)
            out.append(row_copy)
            route_counts_after[route] = route_counts_after.get(route, 0) + 1
    return out, {
        "input_rows": int(len([row for row in list(rows or []) if isinstance(row, Mapping)])),
        "output_rows": int(len(out)),
        "repeat_histogram": {str(k): int(v) for k, v in repeat_hist.items()},
        "route_counts_before": {str(k): int(v) for k, v in route_counts_before.items()},
        "route_counts_after": {str(k): int(v) for k, v in route_counts_after.items()},
    }


def _training_rebalance_summary(
    rows: Sequence[Mapping[str, Any]],
    row_weights: torch.Tensor,
    budget_weights: torch.Tensor,
    *,
    budget_ladder: Sequence[int],
    route_weights: Mapping[str, float] | None,
    budget_weight_map: Mapping[int, float] | None,
    deep_repair_min_depth: int,
    deep_repair_weight: float,
    hole_oversample_repeat: int,
    deep_repair_oversample_repeat: int,
    oversample_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    route_row_weights: dict[str, list[float]] = {}
    route_counts: dict[str, int] = {}
    deep_repair_rows = 0
    for idx, row in enumerate(list(rows or [])):
        route = _route_name(row)
        route_counts[route] = route_counts.get(route, 0) + 1
        route_row_weights.setdefault(route, []).append(float(row_weights[idx].item()))
        if _is_deep_repair_row(row, min_depth=int(deep_repair_min_depth)):
            deep_repair_rows += 1
    return {
        "route_weights": dict(_normalize_route_weight_map(route_weights)),
        "budget_weights": {
            str(int(key)): float(value)
            for key, value in _normalize_budget_weight_map(budget_weight_map, budget_ladder=budget_ladder).items()
        },
        "deep_repair_min_depth": int(deep_repair_min_depth),
        "deep_repair_weight": float(deep_repair_weight),
        "hole_oversample_repeat": int(hole_oversample_repeat),
        "deep_repair_oversample_repeat": int(deep_repair_oversample_repeat),
        "mean_row_weight": 0.0 if row_weights.numel() <= 0 else float(row_weights.mean().item()),
        "max_row_weight": 0.0 if row_weights.numel() <= 0 else float(row_weights.max().item()),
        "mean_budget_weight": 0.0 if budget_weights.numel() <= 0 else float(budget_weights.mean().item()),
        "route_counts": {str(k): int(v) for k, v in route_counts.items()},
        "mean_row_weight_by_route": {
            str(route): float(sum(values) / float(len(values)))
            for route, values in route_row_weights.items()
            if values
        },
        "deep_repair_rows": int(deep_repair_rows),
        "oversample": dict(oversample_meta or {}),
    }


def _binary_brier(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float | None:
    if mask.sum().item() <= 0:
        return None
    value = (((probs - labels) ** 2) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return float(value.item())


def _mean_or_none(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in list(values or []) if math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_bins: int = 10,
) -> float | None:
    if mask.sum().item() <= 0:
        return None
    probs = probs[mask > 0.5]
    labels = labels[mask > 0.5]
    if probs.numel() <= 0:
        return None
    ece = 0.0
    total = float(probs.numel())
    for idx in range(int(n_bins)):
        lo = float(idx) / float(n_bins)
        hi = float(idx + 1) / float(n_bins)
        if idx == int(n_bins) - 1:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs >= lo) & (probs < hi)
        count = int(in_bin.sum().item())
        if count <= 0:
            continue
        conf = float(probs[in_bin].mean().item())
        acc = float(labels[in_bin].mean().item())
        ece += abs(conf - acc) * (float(count) / total)
    return float(ece)


def _regression_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float | None:
    if mask.sum().item() <= 0:
        return None
    value = (torch.abs(pred - target) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return float(value.item())


def _monotonic_violation_rate(values: torch.Tensor) -> float:
    if values.ndim < 2 or values.shape[0] <= 0 or values.shape[1] <= 1:
        return 0.0
    diffs = values[:, 1:] - values[:, :-1]
    if values.ndim > 2:
        leading = values.shape[0]
        diffs = diffs.reshape(leading, -1)
    violated = (diffs < -1.0e-6).any(dim=1)
    return float(violated.float().mean().item())


def _threshold_index(threshold_ladder: Sequence[float], value: float) -> int:
    ladder = [float(v) for v in list(threshold_ladder or SCHEDULER_DEFAULT_THRESHOLD_LADDER)]
    if not ladder:
        return 0
    best_idx = 0
    best_dist = None
    for idx, item in enumerate(ladder):
        dist = abs(float(item) - float(value))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = int(idx)
    return int(best_idx)


def _build_feature_tensors(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    budget_ladder: Sequence[int],
    threshold_ladder: Sequence[float],
    route_aliases: Mapping[str, Any] | None = None,
    route_weights: Mapping[str, float] | None = None,
    budget_weight_map: Mapping[int, float] | None = None,
    deep_repair_min_depth: int = 5,
    deep_repair_weight: float = 1.0,
) -> dict[str, Any]:
    x = torch.tensor(
        [scheduler_feature_vector(row, feature_names=feature_names, route_aliases=route_aliases) for row in rows],
        dtype=torch.float32,
    )
    route_index = torch.tensor(
        [
            _route_index(
                _canonical_route_source(str(row.get("route_source", "") or ""), route_aliases)
            )
            for row in rows
        ],
        dtype=torch.long,
    )
    break_targets: list[torch.Tensor] = []
    break_masks: list[torch.Tensor] = []
    tail_targets: list[torch.Tensor] = []
    tail_masks: list[torch.Tensor] = []
    for budget in budget_ladder:
        break_budget_targets: list[torch.Tensor] = []
        break_budget_masks: list[torch.Tensor] = []
        tail_budget_targets: list[torch.Tensor] = []
        tail_budget_masks: list[torch.Tensor] = []
        for tau in threshold_ladder:
            tau_token = scheduler_threshold_token(tau)
            target, mask = _target_tensor(rows, f"improve_ge_{tau_token}_at_budget_{int(budget)}")
            break_budget_targets.append(target)
            break_budget_masks.append(mask)
            target, mask = _target_tensor(rows, f"tail_gain_{tau_token}_at_budget_{int(budget)}")
            tail_budget_targets.append(target)
            tail_budget_masks.append(mask)
        break_targets.append(torch.stack(break_budget_targets, dim=1))
        break_masks.append(torch.stack(break_budget_masks, dim=1))
        tail_targets.append(torch.stack(tail_budget_targets, dim=1))
        tail_masks.append(torch.stack(tail_budget_masks, dim=1))
    route_win_targets = []
    route_win_masks = []
    new_residual_basin_targets = []
    new_residual_basin_masks = []
    fragile_targets = []
    fragile_masks = []
    stable_targets = []
    stable_masks = []
    cost_wall_targets = []
    cost_wall_masks = []
    cost_exact_targets = []
    cost_exact_masks = []
    cost_total_targets = []
    cost_total_masks = []
    delta_targets = []
    delta_masks = []
    witness_delta_targets = []
    witness_delta_masks = []
    for budget in budget_ladder:
        target, mask = _target_tensor(rows, f"route_win_at_budget_{int(budget)}")
        route_win_targets.append(target)
        route_win_masks.append(mask)
        target, mask = _target_tensor(rows, f"new_residual_basin_at_budget_{int(budget)}")
        new_residual_basin_targets.append(target)
        new_residual_basin_masks.append(mask)
        target, mask = _target_tensor(rows, f"fragility_at_budget_{int(budget)}")
        fragile_targets.append(target)
        fragile_masks.append(mask)
        target, mask = _target_tensor(rows, f"stability_at_budget_{int(budget)}")
        stable_targets.append(target)
        stable_masks.append(mask)
        target, mask = _target_tensor(
            rows,
            f"cost_wall_at_budget_{int(budget)}",
            mask_key=f"cost_wall_observed_mask_at_budget_{int(budget)}",
        )
        cost_wall_targets.append(target)
        cost_wall_masks.append(mask)
        target, mask = _target_tensor(
            rows,
            f"cost_exact_at_budget_{int(budget)}",
            mask_key=f"cost_exact_observed_mask_at_budget_{int(budget)}",
        )
        cost_exact_targets.append(target)
        cost_exact_masks.append(mask)
        target, mask = _target_tensor(
            rows,
            f"cost_total_at_budget_{int(budget)}",
            mask_key=f"cost_total_observed_mask_at_budget_{int(budget)}",
            allow_finite_fallback=True,
        )
        cost_total_targets.append(target)
        cost_total_masks.append(mask)
        target, mask = _target_tensor(rows, f"delta_log_eff_at_budget_{int(budget)}")
        delta_targets.append(target)
        delta_masks.append(mask)
        target, mask = _target_tensor(
            rows,
            f"witness_energy_delta_at_budget_{int(budget)}",
            mask_key=f"witness_energy_observed_mask_at_budget_{int(budget)}",
        )
        witness_delta_targets.append(target)
        witness_delta_masks.append(mask)
    row_weights, budget_weights = _build_training_weight_tensors(
        rows,
        budget_ladder=budget_ladder,
        route_weights=route_weights,
        budget_weights=budget_weight_map,
        deep_repair_min_depth=int(deep_repair_min_depth),
        deep_repair_weight=float(deep_repair_weight),
    )
    threshold_weights = budget_weights.unsqueeze(-1) if budget_weights.ndim == 2 else torch.zeros((len(rows), 0, 0), dtype=torch.float32)
    return {
        "x": x,
        "route_index": route_index,
        "break_targets": torch.stack(break_targets, dim=1) if break_targets else torch.zeros((len(rows), 0, 0), dtype=torch.float32),
        "break_masks": torch.stack(break_masks, dim=1) if break_masks else torch.zeros((len(rows), 0, 0), dtype=torch.float32),
        "tail_targets": torch.stack(tail_targets, dim=1) if tail_targets else torch.zeros((len(rows), 0, 0), dtype=torch.float32),
        "tail_masks": torch.stack(tail_masks, dim=1) if tail_masks else torch.zeros((len(rows), 0, 0), dtype=torch.float32),
        "route_win_targets": torch.stack(route_win_targets, dim=1) if route_win_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "route_win_masks": torch.stack(route_win_masks, dim=1) if route_win_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "new_residual_basin_targets": torch.stack(new_residual_basin_targets, dim=1) if new_residual_basin_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "new_residual_basin_masks": torch.stack(new_residual_basin_masks, dim=1) if new_residual_basin_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "fragile_targets": torch.stack(fragile_targets, dim=1) if fragile_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "fragile_masks": torch.stack(fragile_masks, dim=1) if fragile_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "stable_targets": torch.stack(stable_targets, dim=1) if stable_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "stable_masks": torch.stack(stable_masks, dim=1) if stable_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cost_wall_targets": torch.stack(cost_wall_targets, dim=1) if cost_wall_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cost_wall_masks": torch.stack(cost_wall_masks, dim=1) if cost_wall_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cost_exact_targets": torch.stack(cost_exact_targets, dim=1) if cost_exact_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cost_exact_masks": torch.stack(cost_exact_masks, dim=1) if cost_exact_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cost_total_targets": torch.stack(cost_total_targets, dim=1) if cost_total_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cost_total_masks": torch.stack(cost_total_masks, dim=1) if cost_total_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "delta_targets": torch.stack(delta_targets, dim=1) if delta_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "delta_masks": torch.stack(delta_masks, dim=1) if delta_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "witness_delta_targets": torch.stack(witness_delta_targets, dim=1) if witness_delta_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "witness_delta_masks": torch.stack(witness_delta_masks, dim=1) if witness_delta_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "row_weights": row_weights,
        "budget_weights": budget_weights,
        "threshold_weights": threshold_weights,
        "group_ids": [scheduler_row_group_id(row) for row in rows],
    }


class _SchedulerCriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, budget_count: int, threshold_count: int) -> None:
        super().__init__()
        self.budget_count = int(max(1, budget_count))
        self.threshold_count = int(max(1, threshold_count))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.repair_adapter = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.build_adapter = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.hole_adapter = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.break_base = nn.Linear(hidden_dim, self.threshold_count)
        self.break_delta = nn.Linear(hidden_dim, max(0, self.budget_count - 1) * self.threshold_count)
        self.tail_base = nn.Linear(hidden_dim, self.threshold_count)
        self.tail_delta = nn.Linear(hidden_dim, max(0, self.budget_count - 1) * self.threshold_count)
        self.cost_exact_base = nn.Linear(hidden_dim, 1)
        self.cost_exact_delta = nn.Linear(hidden_dim, max(0, self.budget_count - 1))
        self.cost_wall_base = nn.Linear(hidden_dim, 1)
        self.cost_wall_delta = nn.Linear(hidden_dim, max(0, self.budget_count - 1))
        self.cost_total_base = nn.Linear(hidden_dim, 1)
        self.cost_total_delta = nn.Linear(hidden_dim, max(0, self.budget_count - 1))
        self.witness_delta_base = nn.Linear(hidden_dim, 1)
        self.witness_delta_delta = nn.Linear(hidden_dim, max(0, self.budget_count - 1))
        self.route_win_head = nn.Linear(hidden_dim, self.budget_count)
        self.new_residual_basin_head = nn.Linear(hidden_dim, self.budget_count)
        self.fragile_head = nn.Linear(hidden_dim, self.budget_count)
        self.stable_head = nn.Linear(hidden_dim, self.budget_count)

    def _adapt_hidden(self, hidden: torch.Tensor, route_index: torch.Tensor) -> torch.Tensor:
        repair_hidden = self.repair_adapter(hidden)
        build_hidden = self.build_adapter(hidden)
        hole_hidden = self.hole_adapter(hidden)
        adapted = hidden
        adapted = torch.where(route_index.reshape(-1, 1).eq(0), repair_hidden, adapted)
        adapted = torch.where(route_index.reshape(-1, 1).eq(1), build_hidden, adapted)
        adapted = torch.where(route_index.reshape(-1, 1).eq(2), hole_hidden, adapted)
        return adapted

    def _monotone_logits(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        base = base.unsqueeze(1)
        if self.budget_count <= 1:
            return base
        delta = F.softplus(delta).reshape(-1, self.budget_count - 1, self.threshold_count)
        return torch.cat((base, base + torch.cumsum(delta, dim=1)), dim=1)

    def _monotone_positive_3d(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        base = F.softplus(base).unsqueeze(1)
        if self.budget_count <= 1:
            return base
        delta = F.softplus(delta).reshape(-1, self.budget_count - 1, self.threshold_count)
        return torch.cat((base, base + torch.cumsum(delta, dim=1)), dim=1)

    def _monotone_positive(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        base = F.softplus(base).reshape(-1, 1)
        if self.budget_count <= 1:
            return base
        delta = F.softplus(delta).reshape(-1, self.budget_count - 1)
        return torch.cat((base, base + torch.cumsum(delta, dim=1)), dim=1)

    def forward(self, x: torch.Tensor, route_index: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(x)
        adapted = self._adapt_hidden(hidden, route_index)
        return {
            "break_logits": self._monotone_logits(self.break_base(adapted), self.break_delta(adapted)),
            "tail_gain_pred": self._monotone_positive_3d(self.tail_base(adapted), self.tail_delta(adapted)),
            "route_win_logits": self.route_win_head(adapted),
            "new_residual_basin_logits": self.new_residual_basin_head(adapted),
            "fragile_logits": self.fragile_head(adapted),
            "stable_logits": self.stable_head(adapted),
            "cost_exact_pred": self._monotone_positive(self.cost_exact_base(adapted), self.cost_exact_delta(adapted)),
            "cost_wall_pred": self._monotone_positive(self.cost_wall_base(adapted), self.cost_wall_delta(adapted)),
            "cost_total_pred": self._monotone_positive(self.cost_total_base(adapted), self.cost_total_delta(adapted)),
            "witness_delta_pred": self._monotone_positive(self.witness_delta_base(adapted), self.witness_delta_delta(adapted)),
        }


def _new_scheduler_critic_net(input_dim: int, hidden_dim: int, budget_count: int, threshold_count: int) -> _SchedulerCriticNet:
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = _SchedulerCriticNet(input_dim, hidden_dim, budget_count, threshold_count)
    finally:
        torch.set_default_dtype(prev_dtype)
    return model.to(dtype=torch.float32)


def _default_live_bundle(
    *,
    budget_ladder: Sequence[int],
    threshold_ladder: Sequence[float],
    feature_names: Sequence[str],
    hidden_dim: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    models: Sequence[_SchedulerCriticNet],
    route_aliases: Mapping[str, Any] | None = None,
    objective_mode: str = SCHEDULER_DEFAULT_OBJECTIVE_MODE,
    objective_hybrid_mix: float = SCHEDULER_DEFAULT_HYBRID_OBJECTIVE_MIX,
) -> dict[str, Any]:
    uses_witness = scheduler_uses_witness_energy_features(feature_names)
    return {
        "model_kind": SCHEDULER_MODEL_KIND,
        "feature_schema_version": int(SCHEDULER_FEATURE_SCHEMA_VERSION if uses_witness else 1),
        "witness_energy_feature_enable": bool(uses_witness),
        "scheduler_critic_trained": True,
        "budget_ladder": [int(v) for v in budget_ladder],
        "threshold_ladder": [float(v) for v in threshold_ladder],
        "feature_names": list(feature_names),
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "hidden_dim": int(hidden_dim),
        "ensemble_size": int(len(list(models))),
        "models": list(models),
        "route_names": list(SCHEDULER_ROUTE_NAMES),
        "route_aliases": dict(_normalize_route_aliases(route_aliases)),
        "acquisition_weights": dict(SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS),
        "objective_mode": _normalize_scheduler_objective_mode(objective_mode),
        "objective_hybrid_mix": _normalize_hybrid_objective_mix(objective_hybrid_mix),
    }


def _training_acquisition(
    outputs: Mapping[str, torch.Tensor],
    *,
    threshold_index: int,
    weights: Mapping[str, float],
) -> torch.Tensor:
    break_probs = torch.sigmoid(outputs["break_logits"][:, :, int(threshold_index)])
    tail_gain = outputs["tail_gain_pred"][:, :, int(threshold_index)]
    route_win = torch.sigmoid(outputs["route_win_logits"])
    new_residual_basin = torch.sigmoid(outputs["new_residual_basin_logits"])
    stable = torch.sigmoid(outputs["stable_logits"])
    fragile = torch.sigmoid(outputs["fragile_logits"])
    return (
        float(weights.get("break", 1.0)) * break_probs
        + float(weights.get("tail", 0.5)) * tail_gain
        + float(weights.get("route_win", 0.3)) * route_win
        + float(weights.get("new_residual_basin", 0.2)) * new_residual_basin
        + float(weights.get("stable", 0.1)) * stable
        - float(weights.get("fragile", 0.15)) * fragile
        - float(weights.get("cost_exact", 0.05)) * outputs["cost_exact_pred"]
        - float(weights.get("cost_wall", 0.05)) * outputs["cost_wall_pred"]
    )


def _witness_rate_tensor(
    witness_delta: torch.Tensor,
    cost_total: torch.Tensor,
) -> torch.Tensor:
    return witness_delta / torch.clamp(cost_total, min=1.0e-6)


def _objective_tensor(
    acquisition: torch.Tensor,
    witness_rate: torch.Tensor,
    *,
    objective_mode: str,
    hybrid_mix: float,
) -> torch.Tensor:
    mode = _normalize_scheduler_objective_mode(objective_mode)
    mix = _normalize_hybrid_objective_mix(hybrid_mix)
    if mode == "witness":
        return witness_rate
    if mode == "hybrid":
        return ((1.0 - mix) * acquisition) + (mix * witness_rate)
    return acquisition


def _group_rank_loss(
    acquisition: torch.Tensor,
    delta_targets: torch.Tensor,
    delta_masks: torch.Tensor,
    group_ids: Sequence[str],
    budget_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if acquisition.ndim != 2 or acquisition.shape[0] <= 1:
        return torch.zeros((), dtype=torch.float32)
    groups: dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id or f"ungrouped_{idx}"), []).append(int(idx))
    losses: list[torch.Tensor] = []
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        pred_group = acquisition.index_select(0, idx_tensor)
        delta_group = delta_targets.index_select(0, idx_tensor)
        mask_group = delta_masks.index_select(0, idx_tensor)
        for budget_idx in range(int(pred_group.shape[1])):
            valid = mask_group[:, budget_idx] > 0.5
            if int(valid.sum().item()) <= 1:
                continue
            pred = pred_group[valid, budget_idx]
            target = delta_group[valid, budget_idx]
            diff = target.reshape(-1, 1) - target.reshape(1, -1)
            sign = torch.sign(diff)
            pair_mask = sign.ne(0.0)
            if int(pair_mask.sum().item()) <= 0:
                continue
            pred_diff = pred.reshape(-1, 1) - pred.reshape(1, -1)
            pair_loss = F.softplus(-(sign[pair_mask] * pred_diff[pair_mask]))
            if budget_weights is None:
                losses.append(pair_loss.mean())
                continue
            weight_group = budget_weights.index_select(0, idx_tensor)[valid, budget_idx]
            pair_weight = 0.5 * (weight_group.reshape(-1, 1) + weight_group.reshape(1, -1))
            pair_weight = pair_weight[pair_mask]
            losses.append((pair_loss * pair_weight).sum() / torch.clamp(pair_weight.sum(), min=1.0))
    if not losses:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def _train_single_scheduler_model(
    train_tensors: Mapping[str, Any],
    *,
    input_dim: int,
    hidden_dim: int,
    budget_count: int,
    threshold_count: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    threshold_index: int,
    break_weight: float,
    tail_weight: float,
    route_win_weight: float,
    new_residual_basin_weight: float,
    fragile_weight: float,
    stable_weight: float,
    cost_weight: float,
    rank_weight: float,
    objective_mode: str,
    objective_hybrid_mix: float,
    val_tensors: Mapping[str, Any] | None = None,
    init_state: Mapping[str, Any] | None = None,
) -> tuple[_SchedulerCriticNet, float]:
    torch.manual_seed(int(seed))
    model = _new_scheduler_critic_net(input_dim, int(hidden_dim), int(budget_count), int(threshold_count))
    if isinstance(init_state, Mapping):
        model.load_state_dict(init_state, strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    resolved_objective_mode = _normalize_scheduler_objective_mode(objective_mode)
    resolved_hybrid_mix = _normalize_hybrid_objective_mix(objective_hybrid_mix)

    def _loss_for_tensors(data_tensors: Mapping[str, Any]) -> torch.Tensor:
        x_norm = _normalize_inputs(
            data_tensors["x"],
            feature_mean.reshape(1, -1),
            feature_std.reshape(1, -1),
        )
        outputs = model(x_norm, data_tensors["route_index"])
        break_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["break_logits"], data_tensors["break_targets"], reduction="none"),
            data_tensors["break_masks"],
            weight=data_tensors["threshold_weights"],
        )
        tail_loss = _masked_mean(
            F.smooth_l1_loss(outputs["tail_gain_pred"], data_tensors["tail_targets"], reduction="none"),
            data_tensors["tail_masks"],
            weight=data_tensors["threshold_weights"],
        )
        route_win_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["route_win_logits"], data_tensors["route_win_targets"], reduction="none"),
            data_tensors["route_win_masks"],
            weight=data_tensors["budget_weights"],
        )
        new_residual_basin_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["new_residual_basin_logits"], data_tensors["new_residual_basin_targets"], reduction="none"),
            data_tensors["new_residual_basin_masks"],
            weight=data_tensors["budget_weights"],
        )
        fragile_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["fragile_logits"], data_tensors["fragile_targets"], reduction="none"),
            data_tensors["fragile_masks"],
            weight=data_tensors["budget_weights"],
        )
        stable_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["stable_logits"], data_tensors["stable_targets"], reduction="none"),
            data_tensors["stable_masks"],
            weight=data_tensors["budget_weights"],
        )
        cost_exact_loss = _masked_mean(
            F.smooth_l1_loss(outputs["cost_exact_pred"], data_tensors["cost_exact_targets"], reduction="none"),
            data_tensors["cost_exact_masks"],
            weight=data_tensors["budget_weights"],
        )
        cost_wall_loss = _masked_mean(
            F.smooth_l1_loss(outputs["cost_wall_pred"], data_tensors["cost_wall_targets"], reduction="none"),
            data_tensors["cost_wall_masks"],
            weight=data_tensors["budget_weights"],
        )
        cost_total_loss = _masked_mean(
            F.smooth_l1_loss(outputs["cost_total_pred"], data_tensors["cost_total_targets"], reduction="none"),
            data_tensors["cost_total_masks"],
            weight=data_tensors["budget_weights"],
        )
        witness_loss = _masked_mean(
            F.smooth_l1_loss(outputs["witness_delta_pred"], data_tensors["witness_delta_targets"], reduction="none"),
            data_tensors["witness_delta_masks"],
            weight=data_tensors["budget_weights"],
        )
        witness_rate_loss = _masked_mean(
            F.smooth_l1_loss(
                _witness_rate_tensor(outputs["witness_delta_pred"], outputs["cost_total_pred"]),
                _witness_rate_tensor(data_tensors["witness_delta_targets"], data_tensors["cost_total_targets"]),
                reduction="none",
            ),
            data_tensors["witness_delta_masks"] * data_tensors["cost_total_masks"],
            weight=data_tensors["budget_weights"],
        )
        acquisition_rank_loss = _group_rank_loss(
            _training_acquisition(outputs, threshold_index=threshold_index, weights=SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS),
            data_tensors["delta_targets"],
            data_tensors["delta_masks"],
            data_tensors["group_ids"],
            budget_weights=data_tensors["budget_weights"],
        )
        witness_rank_loss = _group_rank_loss(
            _witness_rate_tensor(outputs["witness_delta_pred"], outputs["cost_total_pred"]),
            _witness_rate_tensor(data_tensors["witness_delta_targets"], data_tensors["cost_total_targets"]),
            data_tensors["witness_delta_masks"] * data_tensors["cost_total_masks"],
            data_tensors["group_ids"],
            budget_weights=data_tensors["budget_weights"],
        )
        if resolved_objective_mode == "witness":
            acquisition_aux_scale = 0.2
            rank_loss = witness_rank_loss
            witness_aux_weight = 1.0
            cost_total_aux_weight = float(cost_weight)
            witness_rate_aux_weight = 1.0
        elif resolved_objective_mode == "hybrid":
            acquisition_aux_scale = 1.0
            rank_loss = ((1.0 - resolved_hybrid_mix) * acquisition_rank_loss) + (resolved_hybrid_mix * witness_rank_loss)
            witness_aux_weight = 1.0
            cost_total_aux_weight = float(cost_weight)
            witness_rate_aux_weight = 1.0
        else:
            acquisition_aux_scale = 1.0
            rank_loss = acquisition_rank_loss
            witness_aux_weight = 0.0
            cost_total_aux_weight = 0.0
            witness_rate_aux_weight = 0.0
        return (
            float(acquisition_aux_scale) * (
                float(break_weight) * break_loss
                + float(tail_weight) * tail_loss
                + float(route_win_weight) * route_win_loss
                + float(new_residual_basin_weight) * new_residual_basin_loss
                + float(fragile_weight) * fragile_loss
                + float(stable_weight) * stable_loss
                + float(cost_weight) * (cost_exact_loss + cost_wall_loss)
            )
            + float(cost_total_aux_weight) * cost_total_loss
            + float(witness_aux_weight) * witness_loss
            + float(witness_rate_aux_weight) * witness_rate_loss
            + float(rank_weight) * rank_loss
        )

    model.train()
    effective_epochs = int(max(1, int(epochs)))
    if resolved_objective_mode != "acquisition":
        effective_epochs = int(max(effective_epochs, 80))
    last_loss = 0.0
    best_loss_value: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    eval_tensors = (
        val_tensors
        if isinstance(val_tensors, Mapping) and int(val_tensors.get("x", torch.zeros((0, 0))).shape[0]) > 0
        else train_tensors
    )
    for _epoch in range(effective_epochs):
        optimizer.zero_grad()
        loss = _loss_for_tensors(train_tensors)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        model.eval()
        with torch.no_grad():
            eval_loss = float(_loss_for_tensors(eval_tensors).item())
        if best_loss_value is None or eval_loss < best_loss_value:
            best_loss_value = float(eval_loss)
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        model.train()
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=True)
    model.eval()
    return model, float(best_loss_value if best_loss_value is not None else last_loss)


@torch.no_grad()
def _predict_ensemble_components(
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    budget_ladder = [int(v) for v in bundle.get("budget_ladder", SCHEDULER_DEFAULT_BUDGET_LADDER)]
    threshold_ladder = [float(v) for v in bundle.get("threshold_ladder", SCHEDULER_DEFAULT_THRESHOLD_LADDER)]
    feature_names = list(bundle.get("feature_names", list(SCHEDULER_FEATURE_NAMES)))
    models = list(bundle.get("models", []) or [])
    if not rows or not models:
        n_rows = len(rows)
        n_budget = len(budget_ladder)
        n_threshold = len(threshold_ladder)
        return {
            "break_probs_mean": torch.zeros((n_rows, n_budget, n_threshold), dtype=torch.float32),
            "tail_gain_mean": torch.zeros((n_rows, n_budget, n_threshold), dtype=torch.float32),
            "route_win_prob_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "new_residual_basin_prob_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "fragile_prob_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "stable_prob_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "cost_exact_pred_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "cost_wall_pred_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "cost_total_pred_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "witness_delta_pred_mean": torch.zeros((n_rows, n_budget), dtype=torch.float32),
            "break_probs_stack": torch.zeros((0, n_rows, n_budget, n_threshold), dtype=torch.float32),
            "tail_gain_stack": torch.zeros((0, n_rows, n_budget, n_threshold), dtype=torch.float32),
            "route_win_prob_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
            "new_residual_basin_prob_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
            "fragile_prob_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
            "stable_prob_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
            "cost_exact_pred_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
            "cost_wall_pred_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
            "cost_total_pred_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
            "witness_delta_pred_stack": torch.zeros((0, n_rows, n_budget), dtype=torch.float32),
        }
    tensors = _build_feature_tensors(
        rows,
        feature_names=feature_names,
        budget_ladder=budget_ladder,
        threshold_ladder=threshold_ladder,
        route_aliases=bundle.get("route_aliases", None),
    )
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    x_norm = _normalize_inputs(tensors["x"], mean, std)
    break_stack = []
    tail_stack = []
    route_win_stack = []
    new_residual_basin_stack = []
    fragile_stack = []
    stable_stack = []
    cost_exact_stack = []
    cost_wall_stack = []
    cost_total_stack = []
    witness_delta_stack = []
    for model in models:
        outputs = model(x_norm, tensors["route_index"])
        break_stack.append(torch.sigmoid(outputs["break_logits"]).detach().cpu())
        tail_stack.append(outputs["tail_gain_pred"].detach().cpu())
        route_win_stack.append(torch.sigmoid(outputs["route_win_logits"]).detach().cpu())
        new_residual_basin_stack.append(torch.sigmoid(outputs["new_residual_basin_logits"]).detach().cpu())
        fragile_stack.append(torch.sigmoid(outputs["fragile_logits"]).detach().cpu())
        stable_stack.append(torch.sigmoid(outputs["stable_logits"]).detach().cpu())
        cost_exact_stack.append(outputs["cost_exact_pred"].detach().cpu())
        cost_wall_stack.append(outputs["cost_wall_pred"].detach().cpu())
        cost_total_stack.append(outputs["cost_total_pred"].detach().cpu())
        witness_delta_stack.append(outputs["witness_delta_pred"].detach().cpu())
    break_probs_stack = torch.stack(break_stack, dim=0)
    tail_gain_stack = torch.stack(tail_stack, dim=0)
    route_win_prob_stack = torch.stack(route_win_stack, dim=0)
    new_residual_basin_prob_stack = torch.stack(new_residual_basin_stack, dim=0)
    fragile_prob_stack = torch.stack(fragile_stack, dim=0)
    stable_prob_stack = torch.stack(stable_stack, dim=0)
    cost_exact_pred_stack = torch.stack(cost_exact_stack, dim=0)
    cost_wall_pred_stack = torch.stack(cost_wall_stack, dim=0)
    cost_total_pred_stack = torch.stack(cost_total_stack, dim=0)
    witness_delta_pred_stack = torch.stack(witness_delta_stack, dim=0)
    return {
        "break_probs_mean": break_probs_stack.mean(dim=0),
        "tail_gain_mean": tail_gain_stack.mean(dim=0),
        "route_win_prob_mean": route_win_prob_stack.mean(dim=0),
        "new_residual_basin_prob_mean": new_residual_basin_prob_stack.mean(dim=0),
        "fragile_prob_mean": fragile_prob_stack.mean(dim=0),
        "stable_prob_mean": stable_prob_stack.mean(dim=0),
        "cost_exact_pred_mean": cost_exact_pred_stack.mean(dim=0),
        "cost_wall_pred_mean": cost_wall_pred_stack.mean(dim=0),
        "cost_total_pred_mean": cost_total_pred_stack.mean(dim=0),
        "witness_delta_pred_mean": witness_delta_pred_stack.mean(dim=0),
        "break_probs_stack": break_probs_stack,
        "tail_gain_stack": tail_gain_stack,
        "route_win_prob_stack": route_win_prob_stack,
        "new_residual_basin_prob_stack": new_residual_basin_prob_stack,
        "fragile_prob_stack": fragile_prob_stack,
        "stable_prob_stack": stable_prob_stack,
        "cost_exact_pred_stack": cost_exact_pred_stack,
        "cost_wall_pred_stack": cost_wall_pred_stack,
        "cost_total_pred_stack": cost_total_pred_stack,
        "witness_delta_pred_stack": witness_delta_pred_stack,
    }


def _acquisition_tensor_from_components(
    *,
    break_probs: torch.Tensor,
    tail_gain: torch.Tensor,
    route_win_prob: torch.Tensor,
    new_residual_basin_prob: torch.Tensor,
    fragile_prob: torch.Tensor,
    stable_prob: torch.Tensor,
    cost_exact_pred: torch.Tensor,
    cost_wall_pred: torch.Tensor,
    weights: Mapping[str, float],
) -> torch.Tensor:
    return (
        float(weights.get("break", 1.0)) * break_probs
        + float(weights.get("tail", 0.5)) * tail_gain
        + float(weights.get("route_win", 0.3)) * route_win_prob
        + float(weights.get("new_residual_basin", 0.2)) * new_residual_basin_prob
        + float(weights.get("stable", 0.1)) * stable_prob
        - float(weights.get("fragile", 0.15)) * fragile_prob
        - float(weights.get("cost_exact", 0.05)) * cost_exact_pred
        - float(weights.get("cost_wall", 0.05)) * cost_wall_pred
    )


def train_scheduler_critic(
    rows: Sequence[Mapping[str, Any]],
    *,
    hidden_dim: int = 64,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    ensemble_size: int = 4,
    budget_ladder: Sequence[Any] | None = None,
    threshold_ladder: Sequence[Any] | None = None,
    break_weight: float = 1.0,
    tail_weight: float = 0.5,
    route_win_weight: float = 0.4,
    new_residual_basin_weight: float = 0.15,
    fragile_weight: float = 0.1,
    stable_weight: float = 0.1,
    cost_weight: float = 0.1,
    rank_weight: float = 0.1,
    route_aliases: Mapping[str, Any] | None = None,
    route_weights: Mapping[str, Any] | None = None,
    budget_weight_map: Mapping[Any, Any] | None = None,
    deep_repair_min_depth: int = 5,
    deep_repair_weight: float = 1.0,
    hole_oversample_repeat: int = 1,
    deep_repair_oversample_repeat: int = 1,
    witness_energy_feature_enable: bool = False,
    objective_mode: str | None = None,
    objective_hybrid_mix: float | None = None,
    init_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_rows = _ensure_scheduler_rows(
        rows,
        budget_ladder=budget_ladder,
        threshold_ladder=threshold_ladder,
    )
    if not dataset_rows:
        raise ValueError("No scheduler rows were provided.")
    budgets = _extract_budget_ladder(dataset_rows, budget_ladder)
    thresholds = _extract_threshold_ladder(dataset_rows, threshold_ladder)
    route_aliases_norm = _normalize_route_aliases(route_aliases)
    train_rows, val_rows = split_scheduler_rows_grouped(dataset_rows, val_fraction=float(val_fraction), seed=int(seed))
    train_rows_effective, oversample_meta = _oversample_training_rows(
        train_rows,
        hole_oversample_repeat=int(hole_oversample_repeat),
        deep_repair_oversample_repeat=int(deep_repair_oversample_repeat),
        deep_repair_min_depth=int(deep_repair_min_depth),
    )
    init_objective_mode = (
        None if not isinstance(init_bundle, Mapping) else init_bundle.get("objective_mode", None)
    )
    init_hybrid_mix = (
        None if not isinstance(init_bundle, Mapping) else init_bundle.get("objective_hybrid_mix", None)
    )
    init_uses_witness_features = bool(
        isinstance(init_bundle, Mapping)
        and scheduler_uses_witness_energy_features(init_bundle.get("feature_names", []) or [])
    )
    resolved_objective_mode = _normalize_scheduler_objective_mode(
        objective_mode if objective_mode is not None else init_objective_mode,
        default="witness" if bool(witness_energy_feature_enable) else SCHEDULER_DEFAULT_OBJECTIVE_MODE,
    )
    resolved_hybrid_mix = _normalize_hybrid_objective_mix(
        objective_hybrid_mix if objective_hybrid_mix is not None else init_hybrid_mix,
    )
    effective_witness_feature_enable = bool(
        witness_energy_feature_enable or init_uses_witness_features
    )
    feature_names = list(
        scheduler_feature_names(
            witness_energy_feature_enable=bool(effective_witness_feature_enable),
        )
    )
    train_tensors = _build_feature_tensors(
        train_rows_effective,
        feature_names=feature_names,
        budget_ladder=budgets,
        threshold_ladder=thresholds,
        route_aliases=route_aliases_norm,
        route_weights=route_weights,
        budget_weight_map=budget_weight_map,
        deep_repair_min_depth=int(deep_repair_min_depth),
        deep_repair_weight=float(deep_repair_weight),
    )
    val_tensors = (
        None
        if not val_rows
        else _build_feature_tensors(
            val_rows,
            feature_names=feature_names,
            budget_ladder=budgets,
            threshold_ladder=thresholds,
            route_aliases=route_aliases_norm,
            route_weights=route_weights,
            budget_weight_map=budget_weight_map,
            deep_repair_min_depth=int(deep_repair_min_depth),
            deep_repair_weight=float(deep_repair_weight),
        )
    )
    feature_mean, feature_std = _feature_stats(train_tensors["x"])
    threshold_index = _threshold_index(thresholds, 0.25)
    models: list[_SchedulerCriticNet] = []
    ensemble_state_dicts: list[dict[str, Any]] = []
    train_member_losses: list[float] = []
    init_state = None
    if isinstance(init_bundle, Mapping):
        init_feature_names = list(init_bundle.get("feature_names", []) or [])
        init_budgets = [int(v) for v in init_bundle.get("budget_ladder", []) or []]
        init_thresholds = [float(v) for v in init_bundle.get("threshold_ladder", []) or []]
        if (
            init_feature_names == feature_names
            and init_budgets == [int(v) for v in budgets]
            and init_thresholds == [float(v) for v in thresholds]
        ):
            init_states = list(init_bundle.get("ensemble_state_dicts", []) or [])
            if init_states:
                init_state = dict(init_states[0])
            elif isinstance(init_bundle.get("model_state_dict", None), Mapping):
                init_state = dict(init_bundle.get("model_state_dict", {}) or {})
    for member_idx in range(max(1, int(ensemble_size))):
        model, loss_value = _train_single_scheduler_model(
            train_tensors,
            input_dim=len(feature_names),
            hidden_dim=int(hidden_dim),
            budget_count=len(budgets),
            threshold_count=len(thresholds),
            feature_mean=feature_mean,
            feature_std=feature_std,
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            seed=int(seed) + int(member_idx),
            threshold_index=threshold_index,
            break_weight=float(break_weight),
            tail_weight=float(tail_weight),
            route_win_weight=float(route_win_weight),
            new_residual_basin_weight=float(new_residual_basin_weight),
            fragile_weight=float(fragile_weight),
            stable_weight=float(stable_weight),
            cost_weight=float(cost_weight),
            rank_weight=float(rank_weight),
            objective_mode=str(resolved_objective_mode),
            objective_hybrid_mix=float(resolved_hybrid_mix),
            val_tensors=val_tensors,
            init_state=init_state if member_idx == 0 else None,
        )
        models.append(model)
        ensemble_state_dicts.append(model.state_dict())
        train_member_losses.append(float(loss_value))
    bundle = _default_live_bundle(
        budget_ladder=budgets,
        threshold_ladder=thresholds,
        feature_names=feature_names,
        hidden_dim=int(hidden_dim),
        feature_mean=feature_mean,
        feature_std=feature_std,
        models=models,
        route_aliases=route_aliases_norm,
        objective_mode=str(resolved_objective_mode),
        objective_hybrid_mix=float(resolved_hybrid_mix),
    )
    bundle["ensemble_state_dicts"] = ensemble_state_dicts
    rebalance_summary = _training_rebalance_summary(
        train_rows_effective,
        train_tensors["row_weights"],
        train_tensors["budget_weights"],
        budget_ladder=budgets,
        route_weights=route_weights,
        budget_weight_map=budget_weight_map,
        deep_repair_min_depth=int(deep_repair_min_depth),
        deep_repair_weight=float(deep_repair_weight),
        hole_oversample_repeat=int(hole_oversample_repeat),
        deep_repair_oversample_repeat=int(deep_repair_oversample_repeat),
        oversample_meta=oversample_meta,
    )
    bundle["training_rebalance"] = dict(rebalance_summary)
    bundle["metrics"] = {
        "train": {
            "n_rows": int(len(train_rows)),
            "effective_n_rows": int(len(train_rows_effective)),
            "member_losses": [float(v) for v in train_member_losses],
            "loss": float(sum(train_member_losses) / float(max(1, len(train_member_losses)))),
            "rebalance": dict(rebalance_summary),
            "route_aliases": dict(route_aliases_norm),
            "objective_mode": str(resolved_objective_mode),
            "objective_hybrid_mix": float(resolved_hybrid_mix),
        },
        "val": evaluate_scheduler_critic(bundle, val_rows) if val_rows else {"n_rows": 0},
    }
    return bundle


def save_scheduler_bundle(bundle: Mapping[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle)
    payload.pop("models", None)
    torch.save(payload, out_path)


def load_scheduler_bundle(path: str | Path) -> dict[str, Any]:
    payload = dict(torch.load(Path(path), map_location="cpu", weights_only=False))
    budgets = [int(v) for v in payload.get("budget_ladder", SCHEDULER_DEFAULT_BUDGET_LADDER)]
    thresholds = [float(v) for v in payload.get("threshold_ladder", SCHEDULER_DEFAULT_THRESHOLD_LADDER)]
    feature_names = list(payload.get("feature_names", list(SCHEDULER_FEATURE_NAMES)))
    hidden_dim = int(payload.get("hidden_dim", 64))
    models: list[_SchedulerCriticNet] = []
    for state_dict in list(payload.get("ensemble_state_dicts", []) or []):
        model = _new_scheduler_critic_net(len(feature_names), hidden_dim, len(budgets), len(thresholds))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        models.append(model)
    payload["model_kind"] = str(payload.get("model_kind", SCHEDULER_MODEL_KIND) or SCHEDULER_MODEL_KIND)
    payload["scheduler_critic_trained"] = bool(payload.get("scheduler_critic_trained", False))
    payload["budget_ladder"] = budgets
    payload["threshold_ladder"] = thresholds
    payload["feature_names"] = feature_names
    payload["feature_mean"] = torch.as_tensor(payload.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32)
    payload["feature_std"] = torch.as_tensor(payload.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32)
    payload["models"] = models
    payload["ensemble_size"] = int(len(models))
    payload["route_aliases"] = dict(_normalize_route_aliases(payload.get("route_aliases", {}) or {}))
    payload["acquisition_weights"] = dict(payload.get("acquisition_weights", {}) or SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS)
    payload["objective_mode"] = _normalize_scheduler_objective_mode(payload.get("objective_mode", None))
    payload["objective_hybrid_mix"] = _normalize_hybrid_objective_mix(payload.get("objective_hybrid_mix", None))
    payload["witness_energy_feature_enable"] = bool(
        payload.get(
            "witness_energy_feature_enable",
            scheduler_uses_witness_energy_features(feature_names),
        )
    )
    payload["feature_schema_version"] = int(
        payload.get(
            "feature_schema_version",
            SCHEDULER_FEATURE_SCHEMA_VERSION if bool(payload["witness_energy_feature_enable"]) else 1,
        )
    )
    return payload


@torch.no_grad()
def predict_scheduler_plan_slate(
    bundle: Mapping[str, Any],
    rows_or_row: Any,
    *,
    acquisition_threshold: float = 0.25,
    acquisition_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "budget_ladder": [],
        "threshold_ladder": [],
        "feature_schema_version": 1,
        "witness_energy_feature_enable": False,
        "objective_mode": SCHEDULER_DEFAULT_OBJECTIVE_MODE,
        "objective_hybrid_mix": SCHEDULER_DEFAULT_HYBRID_OBJECTIVE_MIX,
        "best_index": None,
        "best_route": None,
        "best_opportunity_id": None,
        "best_budget": None,
        "rows": [],
        "route_scores": {},
    }
    if (
        not isinstance(bundle, Mapping)
        or str(bundle.get("model_kind", "")) != SCHEDULER_MODEL_KIND
        or not bool(bundle.get("scheduler_critic_trained", False))
        or not list(bundle.get("models", []) or [])
    ):
        return out
    rows = _ensure_scheduler_rows(
        _coerce_prediction_rows(rows_or_row),
        budget_ladder=bundle.get("budget_ladder", None),
        threshold_ladder=bundle.get("threshold_ladder", None),
    )
    if not rows:
        return out
    budgets = [int(v) for v in bundle.get("budget_ladder", SCHEDULER_DEFAULT_BUDGET_LADDER)]
    thresholds = [float(v) for v in bundle.get("threshold_ladder", SCHEDULER_DEFAULT_THRESHOLD_LADDER)]
    objective_mode = _normalize_scheduler_objective_mode(bundle.get("objective_mode", None))
    objective_hybrid_mix = _normalize_hybrid_objective_mix(bundle.get("objective_hybrid_mix", None))
    weights = dict(SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS)
    weights.update(dict(bundle.get("acquisition_weights", {}) or {}))
    if acquisition_weights is not None:
        weights.update({str(k): float(v) for k, v in dict(acquisition_weights).items()})
    threshold_index = _threshold_index(thresholds, float(acquisition_threshold))
    threshold_value = float(thresholds[threshold_index])
    threshold_token = scheduler_threshold_token(threshold_value)
    preds = _predict_ensemble_components(bundle, rows)
    break_mean = preds["break_probs_mean"][:, :, threshold_index]
    tail_mean = preds["tail_gain_mean"][:, :, threshold_index]
    mean_acquisition = _acquisition_tensor_from_components(
        break_probs=break_mean,
        tail_gain=tail_mean,
        route_win_prob=preds["route_win_prob_mean"],
        new_residual_basin_prob=preds["new_residual_basin_prob_mean"],
        fragile_prob=preds["fragile_prob_mean"],
        stable_prob=preds["stable_prob_mean"],
        cost_exact_pred=preds["cost_exact_pred_mean"],
        cost_wall_pred=preds["cost_wall_pred_mean"],
        weights=weights,
    )
    mean_witness_rate = _witness_rate_tensor(
        preds["witness_delta_pred_mean"],
        preds["cost_total_pred_mean"],
    )
    mean_objective = _objective_tensor(
        mean_acquisition,
        mean_witness_rate,
        objective_mode=objective_mode,
        hybrid_mix=objective_hybrid_mix,
    )
    member_acquisitions: list[torch.Tensor] = []
    member_objectives: list[torch.Tensor] = []
    for member_idx in range(int(preds["break_probs_stack"].shape[0])):
        member_acquisition = _acquisition_tensor_from_components(
            break_probs=preds["break_probs_stack"][member_idx, :, :, threshold_index],
            tail_gain=preds["tail_gain_stack"][member_idx, :, :, threshold_index],
            route_win_prob=preds["route_win_prob_stack"][member_idx],
            new_residual_basin_prob=preds["new_residual_basin_prob_stack"][member_idx],
            fragile_prob=preds["fragile_prob_stack"][member_idx],
            stable_prob=preds["stable_prob_stack"][member_idx],
            cost_exact_pred=preds["cost_exact_pred_stack"][member_idx],
            cost_wall_pred=preds["cost_wall_pred_stack"][member_idx],
            weights=weights,
        )
        member_acquisitions.append(member_acquisition)
        member_objectives.append(
            _objective_tensor(
                member_acquisition,
                _witness_rate_tensor(
                    preds["witness_delta_pred_stack"][member_idx],
                    preds["cost_total_pred_stack"][member_idx],
                ),
                objective_mode=objective_mode,
                hybrid_mix=objective_hybrid_mix,
            )
        )
    acquisition_sigma = torch.stack(member_acquisitions, dim=0).std(dim=0, unbiased=False) if member_acquisitions else torch.zeros_like(mean_acquisition)
    objective_sigma = torch.stack(member_objectives, dim=0).std(dim=0, unbiased=False) if member_objectives else torch.zeros_like(mean_objective)
    rows_out: list[dict[str, Any]] = []
    route_scores: dict[str, float] = {}
    for row_idx, row in enumerate(rows):
        row_out = dict(row)
        row_out["row_index"] = int(row_idx)
        best_budget = None
        best_budget_score = None
        for budget_idx, budget in enumerate(budgets):
            row_out[f"break_prob_{threshold_token}_at_budget_{int(budget)}"] = float(break_mean[row_idx, budget_idx].item())
            row_out[f"tail_gain_{threshold_token}_pred_at_budget_{int(budget)}"] = float(tail_mean[row_idx, budget_idx].item())
            row_out[f"route_win_prob_at_budget_{int(budget)}"] = float(preds["route_win_prob_mean"][row_idx, budget_idx].item())
            row_out[f"new_residual_basin_prob_at_budget_{int(budget)}"] = float(preds["new_residual_basin_prob_mean"][row_idx, budget_idx].item())
            row_out[f"fragile_prob_at_budget_{int(budget)}"] = float(preds["fragile_prob_mean"][row_idx, budget_idx].item())
            row_out[f"stable_prob_at_budget_{int(budget)}"] = float(preds["stable_prob_mean"][row_idx, budget_idx].item())
            row_out[f"cost_exact_pred_at_budget_{int(budget)}"] = float(preds["cost_exact_pred_mean"][row_idx, budget_idx].item())
            row_out[f"cost_wall_pred_at_budget_{int(budget)}"] = float(preds["cost_wall_pred_mean"][row_idx, budget_idx].item())
            row_out[f"cost_total_pred_at_budget_{int(budget)}"] = float(preds["cost_total_pred_mean"][row_idx, budget_idx].item())
            row_out[f"witness_delta_pred_at_budget_{int(budget)}"] = float(preds["witness_delta_pred_mean"][row_idx, budget_idx].item())
            row_out[f"witness_rate_pred_at_budget_{int(budget)}"] = float(mean_witness_rate[row_idx, budget_idx].item())
            row_out[f"acquisition_estimate_at_budget_{int(budget)}"] = float(mean_acquisition[row_idx, budget_idx].item())
            row_out[f"acquisition_sigma_at_budget_{int(budget)}"] = float(acquisition_sigma[row_idx, budget_idx].item())
            row_out[f"objective_estimate_at_budget_{int(budget)}"] = float(mean_objective[row_idx, budget_idx].item())
            row_out[f"objective_sigma_at_budget_{int(budget)}"] = float(objective_sigma[row_idx, budget_idx].item())
            score = float(mean_objective[row_idx, budget_idx].item())
            if best_budget_score is None or score > best_budget_score:
                best_budget_score = score
                best_budget = int(budget)
        row_out["best_budget"] = None if best_budget is None else int(best_budget)
        row_out["best_objective_estimate"] = float(best_budget_score if best_budget_score is not None else float("-inf"))
        row_out["best_acquisition_estimate"] = float(
            max(
                _safe_float(
                    row_out.get(f"acquisition_estimate_at_budget_{int(budget)}", float("-inf")),
                    float("-inf"),
                )
                for budget in budgets
            ) if budgets else float("-inf")
        )
        route_name = str(row_out.get("route_source", "") or "")
        route_scores[route_name] = max(float(route_scores.get(route_name, float("-inf"))), float(row_out["best_objective_estimate"]))
        rows_out.append(row_out)
    rows_out.sort(
        key=lambda item: (
            float(item.get("best_objective_estimate", float("-inf"))),
            float(item.get(f"break_prob_{threshold_token}_at_budget_{int(item.get('best_budget', budgets[0] if budgets else 1))}", float("-inf"))),
            str(item.get("route_source", "")),
            str(item.get("opportunity_id", "")),
        ),
        reverse=True,
    )
    best_row = rows_out[0] if rows_out else None
    best_route = max(route_scores, key=lambda name: (route_scores[name], name)) if route_scores else None
    out.update({
        "trained": True,
        "budget_ladder": budgets,
        "threshold_ladder": thresholds,
        "feature_schema_version": int(bundle.get("feature_schema_version", 1) or 1),
        "witness_energy_feature_enable": bool(bundle.get("witness_energy_feature_enable", False)),
        "objective_mode": str(objective_mode),
        "objective_hybrid_mix": float(objective_hybrid_mix),
        "best_index": None if best_row is None else int(best_row.get("row_index", 0)),
        "best_route": None if best_route is None else str(best_route),
        "best_opportunity_id": None if best_row is None else str(best_row.get("opportunity_id", "")),
        "best_budget": None if best_row is None else int(best_row.get("best_budget", budgets[0] if budgets else 1)),
        "rows": rows_out,
        "route_scores": {str(k): float(v) for k, v in route_scores.items()},
    })
    return out


@torch.no_grad()
def evaluate_scheduler_critic(
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    out = {
        "n_rows": int(len(rows)),
        "break_brier": {},
        "break_ece": {},
        "tail_mae": {},
        "cost_exact_mae": {},
        "cost_wall_mae": {},
        "cost_total_mae": {},
        "witness_delta_mae": {},
        "route_win_brier": {},
        "top1_route_win_accuracy": None,
        "top1_objective_accuracy": None,
        "break_monotonic_violation_rate": 0.0,
        "tail_monotonic_violation_rate": 0.0,
        "cost_exact_monotonic_violation_rate": 0.0,
        "cost_wall_monotonic_violation_rate": 0.0,
        "cost_total_monotonic_violation_rate": 0.0,
        "objective_mode": _normalize_scheduler_objective_mode(bundle.get("objective_mode", None)),
        "route_budget_metrics": {},
    }
    if not rows:
        return out
    dataset_rows = _ensure_scheduler_rows(
        rows,
        budget_ladder=bundle.get("budget_ladder", None),
        threshold_ladder=bundle.get("threshold_ladder", None),
    )
    if not dataset_rows:
        return out
    budgets = [int(v) for v in bundle.get("budget_ladder", SCHEDULER_DEFAULT_BUDGET_LADDER)]
    thresholds = [float(v) for v in bundle.get("threshold_ladder", SCHEDULER_DEFAULT_THRESHOLD_LADDER)]
    feature_names = list(bundle.get("feature_names", list(SCHEDULER_FEATURE_NAMES)))
    tensors = _build_feature_tensors(
        dataset_rows,
        feature_names=feature_names,
        budget_ladder=budgets,
        threshold_ladder=thresholds,
    )
    preds = _predict_ensemble_components(bundle, dataset_rows)
    route_mask_map: dict[str, torch.Tensor] = {}
    for route in SCHEDULER_ROUTE_NAMES:
        route_mask_map[str(route)] = torch.tensor(
            [1.0 if _route_name(row) == str(route) else 0.0 for row in dataset_rows],
            dtype=torch.float32,
        )
    threshold_index = _threshold_index(thresholds, 0.25)
    for budget_idx, budget in enumerate(budgets):
        probs = preds["break_probs_mean"][:, budget_idx, threshold_index]
        labels = tensors["break_targets"][:, budget_idx, threshold_index]
        mask = tensors["break_masks"][:, budget_idx, threshold_index]
        out["break_brier"][str(int(budget))] = _binary_brier(probs, labels, mask)
        out["break_ece"][str(int(budget))] = _expected_calibration_error(probs, labels, mask)
        out["tail_mae"][str(int(budget))] = _regression_mae(
            preds["tail_gain_mean"][:, budget_idx, threshold_index],
            tensors["tail_targets"][:, budget_idx, threshold_index],
            tensors["tail_masks"][:, budget_idx, threshold_index],
        )
        out["cost_exact_mae"][str(int(budget))] = _regression_mae(
            preds["cost_exact_pred_mean"][:, budget_idx],
            tensors["cost_exact_targets"][:, budget_idx],
            tensors["cost_exact_masks"][:, budget_idx],
        )
        out["cost_wall_mae"][str(int(budget))] = _regression_mae(
            preds["cost_wall_pred_mean"][:, budget_idx],
            tensors["cost_wall_targets"][:, budget_idx],
            tensors["cost_wall_masks"][:, budget_idx],
        )
        out["cost_total_mae"][str(int(budget))] = _regression_mae(
            preds["cost_total_pred_mean"][:, budget_idx],
            tensors["cost_total_targets"][:, budget_idx],
            tensors["cost_total_masks"][:, budget_idx],
        )
        out["witness_delta_mae"][str(int(budget))] = _regression_mae(
            preds["witness_delta_pred_mean"][:, budget_idx],
            tensors["witness_delta_targets"][:, budget_idx],
            tensors["witness_delta_masks"][:, budget_idx],
        )
        out["route_win_brier"][str(int(budget))] = _binary_brier(
            preds["route_win_prob_mean"][:, budget_idx],
            tensors["route_win_targets"][:, budget_idx],
            tensors["route_win_masks"][:, budget_idx],
        )
    out["break_monotonic_violation_rate"] = _monotonic_violation_rate(preds["break_probs_mean"][:, :, threshold_index])
    out["tail_monotonic_violation_rate"] = _monotonic_violation_rate(preds["tail_gain_mean"][:, :, threshold_index])
    out["cost_exact_monotonic_violation_rate"] = _monotonic_violation_rate(preds["cost_exact_pred_mean"])
    out["cost_wall_monotonic_violation_rate"] = _monotonic_violation_rate(preds["cost_wall_pred_mean"])
    out["cost_total_monotonic_violation_rate"] = _monotonic_violation_rate(preds["cost_total_pred_mean"])
    threshold_weights = dict(bundle.get("acquisition_weights", {}) or SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS)
    acquisition = _acquisition_tensor_from_components(
        break_probs=preds["break_probs_mean"][:, :, threshold_index],
        tail_gain=preds["tail_gain_mean"][:, :, threshold_index],
        route_win_prob=preds["route_win_prob_mean"],
        new_residual_basin_prob=preds["new_residual_basin_prob_mean"],
        fragile_prob=preds["fragile_prob_mean"],
        stable_prob=preds["stable_prob_mean"],
        cost_exact_pred=preds["cost_exact_pred_mean"],
        cost_wall_pred=preds["cost_wall_pred_mean"],
        weights=threshold_weights,
    )
    objective = _objective_tensor(
        acquisition,
        _witness_rate_tensor(preds["witness_delta_pred_mean"], preds["cost_total_pred_mean"]),
        objective_mode=bundle.get("objective_mode", None),
        hybrid_mix=bundle.get("objective_hybrid_mix", None),
    )
    groups: dict[str, list[int]] = {}
    for idx, group_id in enumerate(tensors["group_ids"]):
        groups.setdefault(str(group_id or f"ungrouped_{idx}"), []).append(int(idx))
    accuracy_hits = 0
    accuracy_total = 0
    objective_hits = 0
    objective_total = 0
    for budget_idx, _budget in enumerate(budgets):
        for indices in groups.values():
            if len(indices) <= 1:
                continue
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            valid = tensors["delta_masks"].index_select(0, idx_tensor)[:, budget_idx] > 0.5
            if int(valid.sum().item()) <= 1:
                continue
            group_indices = idx_tensor[valid]
            acq = acquisition.index_select(0, group_indices)[:, budget_idx]
            target = tensors["delta_targets"].index_select(0, group_indices)[:, budget_idx]
            predicted_best = int(torch.argmax(acq).item())
            target_best = float(target.max().item())
            accuracy_hits += int(float(target[predicted_best].item()) >= target_best - 1.0e-9)
            accuracy_total += 1
            witness_valid = (
                tensors["witness_delta_masks"].index_select(0, idx_tensor)[:, budget_idx] > 0.5
            ) & (
                tensors["cost_total_masks"].index_select(0, idx_tensor)[:, budget_idx] > 0.5
            )
            if int(witness_valid.sum().item()) > 1:
                obj_indices = idx_tensor[witness_valid]
                pred_objective = objective.index_select(0, obj_indices)[:, budget_idx]
                target_objective = _witness_rate_tensor(
                    tensors["witness_delta_targets"].index_select(0, obj_indices)[:, budget_idx],
                    tensors["cost_total_targets"].index_select(0, obj_indices)[:, budget_idx],
                )
                predicted_obj_best = int(torch.argmax(pred_objective).item())
                target_obj_best = float(target_objective.max().item())
                objective_hits += int(float(target_objective[predicted_obj_best].item()) >= target_obj_best - 1.0e-9)
                objective_total += 1
    out["top1_route_win_accuracy"] = None if accuracy_total <= 0 else float(accuracy_hits) / float(accuracy_total)
    out["top1_objective_accuracy"] = None if objective_total <= 0 else float(objective_hits) / float(objective_total)
    threshold_token = scheduler_threshold_token(0.25)
    for route, route_mask in route_mask_map.items():
        route_metrics: dict[str, Any] = {}
        for budget_idx, budget in enumerate(budgets):
            route_break_mask = tensors["break_masks"][:, budget_idx, threshold_index] * route_mask
            route_tail_mask = tensors["tail_masks"][:, budget_idx, threshold_index] * route_mask
            route_route_win_mask = tensors["route_win_masks"][:, budget_idx] * route_mask
            route_cost_exact_mask = tensors["cost_exact_masks"][:, budget_idx] * route_mask
            route_cost_wall_mask = tensors["cost_wall_masks"][:, budget_idx] * route_mask
            route_metrics[str(int(budget))] = {
                "count": int(route_mask.sum().item()),
                "break_brier": _binary_brier(
                    preds["break_probs_mean"][:, budget_idx, threshold_index],
                    tensors["break_targets"][:, budget_idx, threshold_index],
                    route_break_mask,
                ),
                "tail_mae": _regression_mae(
                    preds["tail_gain_mean"][:, budget_idx, threshold_index],
                    tensors["tail_targets"][:, budget_idx, threshold_index],
                    route_tail_mask,
                ),
                "route_win_brier": _binary_brier(
                    preds["route_win_prob_mean"][:, budget_idx],
                    tensors["route_win_targets"][:, budget_idx],
                    route_route_win_mask,
                ),
                "cost_exact_mae": _regression_mae(
                    preds["cost_exact_pred_mean"][:, budget_idx],
                    tensors["cost_exact_targets"][:, budget_idx],
                    route_cost_exact_mask,
                ),
                "cost_wall_mae": _regression_mae(
                    preds["cost_wall_pred_mean"][:, budget_idx],
                    tensors["cost_wall_targets"][:, budget_idx],
                    route_cost_wall_mask,
                ),
                "cost_total_mae": _regression_mae(
                    preds["cost_total_pred_mean"][:, budget_idx],
                    tensors["cost_total_targets"][:, budget_idx],
                    tensors["cost_total_masks"][:, budget_idx] * route_mask,
                ),
                "witness_delta_mae": _regression_mae(
                    preds["witness_delta_pred_mean"][:, budget_idx],
                    tensors["witness_delta_targets"][:, budget_idx],
                    tensors["witness_delta_masks"][:, budget_idx] * route_mask,
                ),
                "mean_break_prob": _mean_or_none([
                    float(value.item())
                    for value, mask_value in zip(
                        preds["break_probs_mean"][:, budget_idx, threshold_index],
                        route_break_mask,
                    )
                    if float(mask_value.item()) > 0.5
                ]),
                "mean_break_label": _mean_or_none([
                    float(value.item())
                    for value, mask_value in zip(
                        tensors["break_targets"][:, budget_idx, threshold_index],
                        route_break_mask,
                    )
                    if float(mask_value.item()) > 0.5
                ]),
                "threshold_token": str(threshold_token),
            }
        out["route_budget_metrics"][str(route)] = route_metrics
    return out


__all__ = [
    "SCHEDULER_DEFAULT_ACQUISITION_WEIGHTS",
    "SCHEDULER_DEFAULT_HYBRID_OBJECTIVE_MIX",
    "SCHEDULER_DEFAULT_OBJECTIVE_MODE",
    "SCHEDULER_FEATURE_NAMES",
    "SCHEDULER_FEATURE_SCHEMA_VERSION",
    "SCHEDULER_MODEL_KIND",
    "SCHEDULER_OBJECTIVE_MODES",
    "SCHEDULER_WITNESS_FEATURE_NAMES",
    "evaluate_scheduler_critic",
    "load_scheduler_bundle",
    "load_scheduler_dataset_rows",
    "predict_scheduler_plan_slate",
    "save_scheduler_bundle",
    "scheduler_feature_names",
    "scheduler_feature_vector",
    "scheduler_row_group_id",
    "_normalize_scheduler_objective_mode",
    "scheduler_uses_witness_energy_features",
    "split_scheduler_rows_grouped",
    "train_scheduler_critic",
]
