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

import torch
from torch import nn
from torch.nn import functional as F

from .opportunity_dataset import normalize_shadow_budget_ladder
from ..shared_opportunity import (
    SHARED_OPPORTUNITY_MASK_FIELD_NAMES,
    normalize_witness_energy_fields,
    shared_opportunity_row_dict,
)


OPPORTUNITY_MODEL_KIND = "opportunity_controller_v1"
OPPORTUNITY_FEATURE_SCHEMA_VERSION = 2
OPPORTUNITY_DEFAULT_BUDGET_LADDER: tuple[int, ...] = (0, 1, 2, 4, 8)
OPPORTUNITY_ROUTE_NAMES: tuple[str, ...] = ("repair", "build")
OPPORTUNITY_TYPE_NAMES: tuple[str, ...] = ("repair_beam", "build_action")
OPPORTUNITY_ACTION_NAMES: tuple[str, ...] = (
    "inv_steer",
    "repair_option",
    "replace",
    "wrap_un",
    "residual",
    "boost",
    "crossover",
)
OPPORTUNITY_MODE_NAMES: tuple[str, ...] = ("identity", "full", "affine")
OPPORTUNITY_PATH_SOURCE_NAMES: tuple[str, ...] = (
    "inverse_beam",
    "critic_path_head",
    "oracle_path_sweep",
    "derived_inverse_repair_slate",
    "guided",
    "random",
)
OPPORTUNITY_EVIDENCE_NAMES: tuple[str, ...] = ("preview_only", "preview_support", "exact_known")
OPPORTUNITY_NUMERIC_FEATURE_NAMES: tuple[str, ...] = (
    "parent_depth",
    "parent_log_eff_mse",
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
    "current_best_child_log_eff_mse",
    "current_best_route_log_eff_mse",
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
)
OPPORTUNITY_WITNESS_FEATURE_NAMES: tuple[str, ...] = (
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
OPPORTUNITY_FEATURE_NAMES: tuple[str, ...] = (
    *OPPORTUNITY_NUMERIC_FEATURE_NAMES,
    *SHARED_OPPORTUNITY_MASK_FIELD_NAMES,
    *tuple(f"route_is_{name}" for name in OPPORTUNITY_ROUTE_NAMES),
    "route_is_other",
    *tuple(f"type_is_{name}" for name in OPPORTUNITY_TYPE_NAMES),
    "type_is_other",
    *tuple(f"action_is_{name}" for name in OPPORTUNITY_ACTION_NAMES),
    "action_is_other",
    *tuple(f"mode_is_{name}" for name in OPPORTUNITY_MODE_NAMES),
    "mode_is_other",
    *tuple(f"path_source_is_{name}" for name in OPPORTUNITY_PATH_SOURCE_NAMES),
    "path_source_is_other",
    *tuple(f"evidence_is_{name}" for name in OPPORTUNITY_EVIDENCE_NAMES),
    "evidence_is_other",
)


def opportunity_uses_witness_energy_features(feature_names: Sequence[str] | None) -> bool:
    feature_set = {str(name) for name in list(feature_names or ())}
    return any(name in feature_set for name in OPPORTUNITY_WITNESS_FEATURE_NAMES)


def opportunity_feature_names(*, witness_energy_feature_enable: bool = False) -> tuple[str, ...]:
    if bool(witness_energy_feature_enable):
        return (*OPPORTUNITY_FEATURE_NAMES, *OPPORTUNITY_WITNESS_FEATURE_NAMES)
    return tuple(OPPORTUNITY_FEATURE_NAMES)


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _log1p_nonneg(value: Any) -> float:
    v = _safe_float_or_none(value)
    if v is None or not math.isfinite(v):
        return 0.0
    return float(math.log1p(max(0.0, float(v))))


def _one_hot(value: str, allowed: Sequence[str], prefix: str) -> dict[str, float]:
    token = str(value or "")
    out = {f"{prefix}{name}": 0.0 for name in allowed}
    other_name = f"{prefix}other"
    if token in allowed:
        out[f"{prefix}{token}"] = 1.0
    else:
        out[other_name] = 1.0
    return out


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


def _load_opportunity_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_opportunity_dataset_rows(dataset_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in dataset_paths:
        payload = _load_opportunity_payload(path)
        rows.extend(
            shared_opportunity_row_dict(row, route_source=(row.get("route_source", "") if isinstance(row, Mapping) else ""))
            for row in list(payload.get("rows", []) or [])
            if isinstance(row, Mapping)
        )
    return rows


def _route_index(route_source: str) -> int:
    try:
        return OPPORTUNITY_ROUTE_NAMES.index(str(route_source or ""))
    except ValueError:
        return 0


def opportunity_feature_vector(row: Mapping[str, Any], *, feature_names: Sequence[str] = OPPORTUNITY_FEATURE_NAMES) -> list[float]:
    item = shared_opportunity_row_dict(row, route_source=row.get("route_source", "") if isinstance(row, Mapping) else "")
    route_source = str(item.get("route_source", "") or "")
    opportunity_type = str(item.get("opportunity_type", "") or "")
    action = str(item.get("action", "") or "")
    target_mode = str(item.get("target_mode", "") or "")
    path_source = str(item.get("path_source", "") or "")
    evidence_level = str(item.get("evidence_level", "") or "")
    path = list(item.get("path", []) or [])
    witness = normalize_witness_energy_fields(item)
    raw_features: dict[str, float] = {
        "parent_depth": float(_safe_int(item.get("parent_depth", 0), 0)),
        "parent_log_eff_mse": _log1p_nonneg(item.get("parent_eff_mse", item.get("estimated_parent_eff_mse", None))),
        "budget_exact_spent": float(_safe_int(item.get("budget_exact_spent", 0), 0)),
        "budget_remaining": float(_safe_int(item.get("budget_remaining", 0), 0)),
        "budget_widen_spent": float(_safe_int(item.get("budget_widen_spent", 0), 0)),
        "budget_micro_spent": float(_safe_int(item.get("budget_micro_spent", 0), 0)),
        "candidate_count_observed": float(_safe_int(item.get("candidate_count_observed", 0), 0)),
        "candidate_count_unique": float(_safe_int(item.get("candidate_count_unique", 0), 0)),
        "preview_candidate_count_total": float(_safe_int(item.get("preview_candidate_count_total", item.get("candidate_count_observed", 0)), 0)),
        "preview_candidate_count_unique_total": float(_safe_int(item.get("preview_candidate_count_unique_total", item.get("candidate_count_unique", 0)), 0)),
        "shadow_total_exact_available": float(_safe_int(item.get("shadow_total_exact_available", 0), 0)),
        "shadow_total_preview_available": float(_safe_int(item.get("shadow_total_preview_available", 0), 0)),
        "shadow_executor_reveals_observed": float(_safe_int(item.get("shadow_executor_reveals_observed", item.get("budget_exact_spent", 0)), 0)),
        "current_best_child_log_eff_mse": _log1p_nonneg(item.get("current_best_child_eff_mse", None)),
        "current_best_route_log_eff_mse": _log1p_nonneg(item.get("current_best_route_eff_mse", None)),
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
    raw_features.update(_one_hot(route_source, OPPORTUNITY_ROUTE_NAMES, "route_is_"))
    raw_features.update(_one_hot(opportunity_type, OPPORTUNITY_TYPE_NAMES, "type_is_"))
    raw_features.update(_one_hot(action, OPPORTUNITY_ACTION_NAMES, "action_is_"))
    raw_features.update(_one_hot(target_mode, OPPORTUNITY_MODE_NAMES, "mode_is_"))
    raw_features.update(_one_hot(path_source, OPPORTUNITY_PATH_SOURCE_NAMES, "path_source_is_"))
    raw_features.update(_one_hot(evidence_level, OPPORTUNITY_EVIDENCE_NAMES, "evidence_is_"))
    return [float(raw_features.get(name, 0.0)) for name in feature_names]


def _target_tensor(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[torch.Tensor, torch.Tensor]:
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
            mask.append(1.0)
        except Exception:
            values.append(0.0)
            mask.append(0.0)
    return torch.tensor(values, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(), min=1.0)
    return (loss * mask).sum() / denom


def opportunity_row_group_id(row: Mapping[str, Any]) -> str:
    item = row if isinstance(row, Mapping) else {}
    group_parts = [
        str(item.get("shadow_source_row_id", "") or ""),
        str(item.get("source_row_id", "") or ""),
        str(item.get("decision_context_id", "") or ""),
        str(item.get("decision_id", "") or ""),
        str(item.get("spec_id", "") or ""),
    ]
    parts = [part for part in group_parts if part]
    if parts:
        return "::".join(parts)
    fallback_parts = [
        str(item.get("opportunity_id", "") or ""),
        str(item.get("route_source", "") or ""),
        str(item.get("parent_key", "") or ""),
        str(item.get("parent_expr", "") or ""),
    ]
    return "::".join([part for part in fallback_parts if part]) or "ungrouped"


def split_opportunity_rows_grouped(
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
        groups.setdefault(opportunity_row_group_id(row), []).append(int(idx))
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


def _split_rows(rows: Sequence[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return split_opportunity_rows_grouped(rows, val_fraction=val_fraction, seed=seed)


class _OpportunityCriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, budget_count: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.repair_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.build_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)
        self.fragility_head = nn.Linear(hidden_dim, 1)
        self.route_flip_head = nn.Linear(hidden_dim, 1)
        self.new_residual_basin_head = nn.Linear(hidden_dim, 1)
        self.cover_head = nn.Linear(hidden_dim, budget_count)
        self.cond_gain_head = nn.Linear(hidden_dim, budget_count)

    def forward(self, x: torch.Tensor, route_index: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(x)
        repair_hidden = self.repair_adapter(hidden)
        build_hidden = self.build_adapter(hidden)
        route_mask = route_index.reshape(-1, 1).eq(0)
        adapted = torch.where(route_mask, repair_hidden, build_hidden)
        return {
            "gain_next": self.gain_head(adapted).reshape(-1),
            "cost_pred": self.cost_head(adapted).reshape(-1),
            "fragility_logit": self.fragility_head(adapted).reshape(-1),
            "route_flip_logit": self.route_flip_head(adapted).reshape(-1),
            "new_residual_basin_logit": self.new_residual_basin_head(adapted).reshape(-1),
            "cover_logits": self.cover_head(adapted),
            "cond_gain_pred": self.cond_gain_head(adapted),
        }


def _cost_target_for_row(row: Mapping[str, Any]) -> tuple[float, float]:
    for key in (
        "cost_total_at_budget_1",
        "observed_total_cost",
        "cost_wall_at_budget_1",
        "observed_wall_seconds",
        "cost_exact_at_budget_1",
        "observed_exact_evals",
        "cost_estimate",
    ):
        value = _safe_float_or_none(row.get(key, None))
        if value is None or not math.isfinite(value):
            continue
        return float(value), 1.0
    return 0.0, 0.0


def _build_feature_tensors(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    budget_ladder: Sequence[int],
) -> dict[str, torch.Tensor]:
    x = torch.tensor(
        [opportunity_feature_vector(row, feature_names=feature_names) for row in rows],
        dtype=torch.float32,
    )
    route_index = torch.tensor(
        [_route_index(str(row.get("route_source", "") or "")) for row in rows],
        dtype=torch.long,
    )
    gain_next, gain_next_mask = _target_tensor(rows, "expected_gain_next_under_executor")
    cost_values, cost_masks = zip(*[_cost_target_for_row(row) for row in rows]) if rows else ((), ())
    cost_target = torch.tensor(cost_values, dtype=torch.float32) if cost_values else torch.zeros((0,), dtype=torch.float32)
    cost_mask = torch.tensor(cost_masks, dtype=torch.float32) if cost_masks else torch.zeros((0,), dtype=torch.float32)
    fragility_target, fragility_mask = _target_tensor(rows, "fragility_at_budget_1")
    route_flip_target, route_flip_mask = _target_tensor(rows, "route_flip_at_budget_1")
    new_residual_basin_target, new_residual_basin_mask = _target_tensor(rows, "new_residual_basin_at_budget_1")
    positive_budgets = [int(v) for v in budget_ladder if int(v) > 0]
    cover_targets = []
    cover_masks = []
    cond_gain_targets = []
    cond_gain_masks = []
    for budget in positive_budgets:
        t, m = _target_tensor(rows, f"coverage_at_budget_{int(budget)}")
        cover_targets.append(t)
        cover_masks.append(m)
        t, m = _target_tensor(rows, f"cond_gain_at_budget_{int(budget)}_if_covered_under_executor")
        cond_gain_targets.append(t)
        cond_gain_masks.append(m)
    return {
        "x": x,
        "route_index": route_index,
        "gain_next": gain_next,
        "gain_next_mask": gain_next_mask,
        "cost_target": cost_target,
        "cost_mask": cost_mask,
        "fragility_target": fragility_target,
        "fragility_mask": fragility_mask,
        "route_flip_target": route_flip_target,
        "route_flip_mask": route_flip_mask,
        "new_residual_basin_target": new_residual_basin_target,
        "new_residual_basin_mask": new_residual_basin_mask,
        "cover_targets": torch.stack(cover_targets, dim=1) if cover_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cover_masks": torch.stack(cover_masks, dim=1) if cover_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cond_gain_targets": torch.stack(cond_gain_targets, dim=1) if cond_gain_targets else torch.zeros((len(rows), 0), dtype=torch.float32),
        "cond_gain_masks": torch.stack(cond_gain_masks, dim=1) if cond_gain_masks else torch.zeros((len(rows), 0), dtype=torch.float32),
    }


def _normalize_inputs(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / torch.clamp(std, min=1.0e-6)


def _feature_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std > 1.0e-6, std, torch.ones_like(std))
    return mean, std


def _binary_brier(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float | None:
    if mask.sum().item() <= 0:
        return None
    value = (((probs - labels) ** 2) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return float(value.item())


def _regression_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float | None:
    if mask.sum().item() <= 0:
        return None
    value = (torch.abs(pred - target) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return float(value.item())


def _regression_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float | None:
    if mask.sum().item() <= 0:
        return None
    value = (((pred - target) ** 2) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return float(torch.sqrt(value).item())


def _expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *, n_bins: int = 10) -> float | None:
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


def _temperature_candidates() -> list[float]:
    return [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() <= 1:
        return 1.0
    best_temp = 1.0
    best_loss = None
    masked_logits = logits[mask > 0.5]
    masked_labels = labels[mask > 0.5]
    for temp in _temperature_candidates():
        loss = F.binary_cross_entropy_with_logits(masked_logits / float(temp), masked_labels)
        loss_value = float(loss.item())
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_temp = float(temp)
    return float(best_temp)


def _enforce_monotone_cover_probs(cover_probs: torch.Tensor) -> torch.Tensor:
    if cover_probs.ndim != 2 or cover_probs.shape[1] <= 1:
        return cover_probs
    return torch.cummax(cover_probs, dim=1)[0]


def _monotonic_violation_rate(cover_probs: torch.Tensor) -> float:
    if cover_probs.ndim != 2 or cover_probs.shape[1] <= 1 or cover_probs.shape[0] <= 0:
        return 0.0
    diffs = cover_probs[:, 1:] - cover_probs[:, :-1]
    violated = (diffs < -1.0e-6).any(dim=1)
    return float(violated.float().mean().item())


@torch.no_grad()
def _predict_tensors(
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    if not rows:
        budget_ladder = [int(v) for v in bundle.get("budget_ladder", OPPORTUNITY_DEFAULT_BUDGET_LADDER)]
        return {
            "gain_next": torch.zeros((0,), dtype=torch.float32),
            "cost_pred": torch.zeros((0,), dtype=torch.float32),
            "fragility_prob": torch.zeros((0,), dtype=torch.float32),
            "route_flip_prob": torch.zeros((0,), dtype=torch.float32),
            "new_residual_basin_prob": torch.zeros((0,), dtype=torch.float32),
            "cover_probs": torch.zeros((0, max(0, len([v for v in budget_ladder if int(v) > 0]))), dtype=torch.float32),
            "cond_gain_pred": torch.zeros((0, max(0, len([v for v in budget_ladder if int(v) > 0]))), dtype=torch.float32),
        }
    feature_names = list(bundle.get("feature_names", list(OPPORTUNITY_FEATURE_NAMES)))
    budget_ladder = [int(v) for v in bundle.get("budget_ladder", OPPORTUNITY_DEFAULT_BUDGET_LADDER)]
    tensors = _build_feature_tensors(rows, feature_names=feature_names, budget_ladder=budget_ladder)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    outputs = bundle["model"](
        _normalize_inputs(tensors["x"], mean, std),
        tensors["route_index"],
    )
    temperatures = {
        int(k): float(v)
        for k, v in dict(bundle.get("coverage_temperature", {}) or {}).items()
    }
    positive_budgets = [int(v) for v in budget_ladder if int(v) > 0]
    calibrated_probs: list[torch.Tensor] = []
    for idx, budget in enumerate(positive_budgets):
        logits = outputs["cover_logits"][:, idx]
        calibrated_probs.append(torch.sigmoid(logits / float(max(1.0e-6, temperatures.get(int(budget), 1.0)))))
    cover_probs = torch.stack(calibrated_probs, dim=1) if calibrated_probs else torch.zeros((len(rows), 0), dtype=torch.float32)
    cover_probs = _enforce_monotone_cover_probs(cover_probs)
    return {
        "gain_next": outputs["gain_next"].detach().cpu(),
        "cost_pred": outputs["cost_pred"].detach().cpu(),
        "fragility_prob": torch.sigmoid(outputs["fragility_logit"]).detach().cpu(),
        "route_flip_prob": torch.sigmoid(outputs["route_flip_logit"]).detach().cpu(),
        "new_residual_basin_prob": torch.sigmoid(outputs["new_residual_basin_logit"]).detach().cpu(),
        "cover_probs": cover_probs.detach().cpu(),
        "cond_gain_pred": outputs["cond_gain_pred"].detach().cpu(),
    }


@torch.no_grad()
def evaluate_opportunity_controller(
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    out = {
        "n_rows": int(len(rows)),
        "gain_mae": None,
        "gain_rmse": None,
        "cost_mae": None,
        "cover_brier": {},
        "cover_ece": {},
        "cover_monotonic_violation_rate": 0.0,
    }
    if not rows:
        return out
    budget_ladder = [int(v) for v in bundle.get("budget_ladder", OPPORTUNITY_DEFAULT_BUDGET_LADDER)]
    feature_names = list(bundle.get("feature_names", list(OPPORTUNITY_FEATURE_NAMES)))
    tensors = _build_feature_tensors(rows, feature_names=feature_names, budget_ladder=budget_ladder)
    preds = _predict_tensors(bundle, rows)
    out["gain_mae"] = _regression_mae(preds["gain_next"], tensors["gain_next"], tensors["gain_next_mask"])
    out["gain_rmse"] = _regression_rmse(preds["gain_next"], tensors["gain_next"], tensors["gain_next_mask"])
    out["cost_mae"] = _regression_mae(preds["cost_pred"], tensors["cost_target"], tensors["cost_mask"])
    positive_budgets = [int(v) for v in budget_ladder if int(v) > 0]
    for idx, budget in enumerate(positive_budgets):
        probs = preds["cover_probs"][:, idx]
        labels = tensors["cover_targets"][:, idx]
        mask = tensors["cover_masks"][:, idx]
        out["cover_brier"][str(int(budget))] = _binary_brier(probs, labels, mask)
        out["cover_ece"][str(int(budget))] = _expected_calibration_error(probs, labels, mask)
    out["cover_monotonic_violation_rate"] = _monotonic_violation_rate(preds["cover_probs"])
    return out


def train_opportunity_controller(
    rows: Sequence[Mapping[str, Any]],
    *,
    hidden_dim: int = 64,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    gain_weight: float = 1.0,
    cover_weight: float = 0.5,
    cond_gain_weight: float = 0.15,
    fragility_weight: float = 0.05,
    cost_weight: float = 0.05,
    route_flip_weight: float = 0.02,
    new_residual_basin_weight: float = 0.02,
    witness_energy_feature_enable: bool = False,
    init_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_rows = [
        shared_opportunity_row_dict(row, route_source=(row.get("route_source", "") if isinstance(row, Mapping) else ""))
        for row in list(rows or [])
        if isinstance(row, Mapping)
        and row.get("expected_gain_next_under_executor", None) is not None
    ]
    if not dataset_rows:
        raise ValueError("No opportunity rows with expected_gain_next_under_executor were provided.")
    budget_ladder = normalize_shadow_budget_ladder(
        tuple(
            sorted(
                {
                    int(key.split("_")[-1])
                    for row in dataset_rows
                    for key in row.keys()
                    if str(key).startswith("coverage_at_budget_")
                }
            )
        )
        or OPPORTUNITY_DEFAULT_BUDGET_LADDER
    )
    positive_budgets = [int(v) for v in budget_ladder if int(v) > 0]
    train_rows, val_rows = _split_rows(dataset_rows, val_fraction=float(val_fraction), seed=int(seed))
    feature_names = list(
        opportunity_feature_names(
            witness_energy_feature_enable=bool(witness_energy_feature_enable),
        )
    )
    train_tensors = _build_feature_tensors(train_rows, feature_names=feature_names, budget_ladder=budget_ladder)
    val_tensors = _build_feature_tensors(val_rows, feature_names=feature_names, budget_ladder=budget_ladder) if val_rows else None
    feature_mean, feature_std = _feature_stats(train_tensors["x"])
    model = _OpportunityCriticNet(len(feature_names), int(hidden_dim), len(positive_budgets)).to(dtype=torch.float32)
    if isinstance(init_bundle, Mapping):
        init_feature_names = list(init_bundle.get("feature_names", []) or [])
        init_budgets = [int(v) for v in init_bundle.get("budget_ladder", []) or []]
        if (
            init_feature_names == feature_names
            and init_budgets == [int(v) for v in budget_ladder]
            and isinstance(init_bundle.get("model_state_dict", None), Mapping)
        ):
            model.load_state_dict(init_bundle.get("model_state_dict", {}), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    torch.manual_seed(int(seed))
    model.train()
    last_train_loss = 0.0
    for _epoch in range(max(1, int(epochs))):
        optimizer.zero_grad()
        outputs = model(
            _normalize_inputs(train_tensors["x"], feature_mean.reshape(1, -1), feature_std.reshape(1, -1)),
            train_tensors["route_index"],
        )
        gain_loss = _masked_mean((outputs["gain_next"] - train_tensors["gain_next"]) ** 2, train_tensors["gain_next_mask"])
        cost_loss = _masked_mean((outputs["cost_pred"] - train_tensors["cost_target"]) ** 2, train_tensors["cost_mask"])
        if len(positive_budgets) > 0:
            cover_loss = _masked_mean(
                F.binary_cross_entropy_with_logits(outputs["cover_logits"], train_tensors["cover_targets"], reduction="none"),
                train_tensors["cover_masks"],
            )
            cond_gain_loss = _masked_mean((outputs["cond_gain_pred"] - train_tensors["cond_gain_targets"]) ** 2, train_tensors["cond_gain_masks"])
        else:
            cover_loss = torch.zeros((), dtype=torch.float32)
            cond_gain_loss = torch.zeros((), dtype=torch.float32)
        fragility_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["fragility_logit"], train_tensors["fragility_target"], reduction="none"),
            train_tensors["fragility_mask"],
        )
        route_flip_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["route_flip_logit"], train_tensors["route_flip_target"], reduction="none"),
            train_tensors["route_flip_mask"],
        )
        new_residual_basin_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(outputs["new_residual_basin_logit"], train_tensors["new_residual_basin_target"], reduction="none"),
            train_tensors["new_residual_basin_mask"],
        )
        loss = (
            float(gain_weight) * gain_loss
            + float(cover_weight) * cover_loss
            + float(cond_gain_weight) * cond_gain_loss
            + float(fragility_weight) * fragility_loss
            + float(cost_weight) * cost_loss
            + float(route_flip_weight) * route_flip_loss
            + float(new_residual_basin_weight) * new_residual_basin_loss
        )
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.item())
    model.eval()

    coverage_temperature: dict[str, float] = {}
    calibration_metrics: dict[str, Any] = {}
    if val_rows and len(positive_budgets) > 0:
        with torch.no_grad():
            raw_outputs = model(
                _normalize_inputs(val_tensors["x"], feature_mean.reshape(1, -1), feature_std.reshape(1, -1)),
                val_tensors["route_index"],
            )
            raw_probs = torch.sigmoid(raw_outputs["cover_logits"]).detach().cpu()
            calibration_metrics["raw_monotonic_violation_rate"] = _monotonic_violation_rate(raw_probs)
            calibrated_probs: list[torch.Tensor] = []
            for idx, budget in enumerate(positive_budgets):
                logits = raw_outputs["cover_logits"][:, idx].detach().cpu()
                labels = val_tensors["cover_targets"][:, idx].detach().cpu()
                mask = val_tensors["cover_masks"][:, idx].detach().cpu()
                temp = _fit_temperature(logits, labels, mask)
                coverage_temperature[str(int(budget))] = float(temp)
                probs = torch.sigmoid(logits / float(temp))
                calibrated_probs.append(probs)
                calibration_metrics[str(int(budget))] = {
                    "temperature": float(temp),
                    "raw_ece": _expected_calibration_error(torch.sigmoid(logits), labels, mask),
                    "calibrated_ece": _expected_calibration_error(probs, labels, mask),
                    "raw_brier": _binary_brier(torch.sigmoid(logits), labels, mask),
                    "calibrated_brier": _binary_brier(probs, labels, mask),
                }
            mono_probs = _enforce_monotone_cover_probs(torch.stack(calibrated_probs, dim=1))
            calibration_metrics["calibrated_monotonic_violation_rate"] = _monotonic_violation_rate(torch.stack(calibrated_probs, dim=1))
            calibration_metrics["monotone_monotonic_violation_rate"] = _monotonic_violation_rate(mono_probs)
    bundle = {
        "model_kind": OPPORTUNITY_MODEL_KIND,
        "feature_schema_version": int(
            OPPORTUNITY_FEATURE_SCHEMA_VERSION if opportunity_uses_witness_energy_features(feature_names) else 1
        ),
        "witness_energy_feature_enable": bool(opportunity_uses_witness_energy_features(feature_names)),
        "opportunity_controller_trained": True,
        "hidden_dim": int(hidden_dim),
        "feature_names": feature_names,
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "budget_ladder": [int(v) for v in budget_ladder],
        "coverage_temperature": coverage_temperature,
        "model_state_dict": model.state_dict(),
        "metrics": {
            "train": {
                "n_rows": int(len(train_rows)),
                "loss": float(last_train_loss),
            },
            "val": evaluate_opportunity_controller(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "feature_mean": feature_mean.detach().cpu(),
                    "feature_std": feature_std.detach().cpu(),
                    "budget_ladder": [int(v) for v in budget_ladder],
                    "coverage_temperature": coverage_temperature,
                },
                val_rows,
            ) if val_rows else {"n_rows": 0},
        },
        "calibration": calibration_metrics,
        "route_names": list(OPPORTUNITY_ROUTE_NAMES),
        "opportunity_action_names": list(OPPORTUNITY_ACTION_NAMES),
        "model": model,
    }
    return bundle


def save_opportunity_bundle(bundle: Mapping[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle)
    payload.pop("model", None)
    torch.save(payload, out_path)


def load_opportunity_bundle(path: str | Path) -> dict[str, Any]:
    payload = dict(torch.load(Path(path), map_location="cpu", weights_only=False))
    budget_ladder = [int(v) for v in payload.get("budget_ladder", OPPORTUNITY_DEFAULT_BUDGET_LADDER)]
    positive_budgets = [int(v) for v in budget_ladder if int(v) > 0]
    feature_names = list(payload.get("feature_names", list(OPPORTUNITY_FEATURE_NAMES)))
    hidden_dim = int(payload.get("hidden_dim", 64))
    model = _OpportunityCriticNet(len(feature_names), hidden_dim, len(positive_budgets)).to(dtype=torch.float32)
    model.load_state_dict(payload.get("model_state_dict", {}), strict=True)
    model.eval()
    payload["model_kind"] = str(payload.get("model_kind", OPPORTUNITY_MODEL_KIND) or OPPORTUNITY_MODEL_KIND)
    payload["opportunity_controller_trained"] = bool(payload.get("opportunity_controller_trained", False))
    payload["feature_names"] = feature_names
    payload["feature_mean"] = torch.as_tensor(payload.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32)
    payload["feature_std"] = torch.as_tensor(payload.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32)
    payload["budget_ladder"] = budget_ladder
    payload["coverage_temperature"] = {
        str(int(k)): float(v)
        for k, v in dict(payload.get("coverage_temperature", {}) or {}).items()
    }
    payload["witness_energy_feature_enable"] = bool(
        payload.get(
            "witness_energy_feature_enable",
            opportunity_uses_witness_energy_features(feature_names),
        )
    )
    payload["feature_schema_version"] = int(
        payload.get(
            "feature_schema_version",
            OPPORTUNITY_FEATURE_SCHEMA_VERSION if bool(payload["witness_energy_feature_enable"]) else 1,
        )
    )
    payload["model"] = model
    return payload


@torch.no_grad()
def predict_opportunity_slate(
    bundle: Mapping[str, Any],
    rows_or_row: Any,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "budget_ladder": [],
        "feature_schema_version": 1,
        "witness_energy_feature_enable": False,
        "best_index": None,
        "best_route": None,
        "best_opportunity_id": None,
        "route_scores": {},
        "rows": [],
    }
    if (
        not isinstance(bundle, Mapping)
        or str(bundle.get("model_kind", "")) != OPPORTUNITY_MODEL_KIND
        or "model" not in bundle
        or not bool(bundle.get("opportunity_controller_trained", False))
    ):
        return out
    rows = _coerce_prediction_rows(rows_or_row)
    if not rows:
        return out
    budget_ladder = [int(v) for v in bundle.get("budget_ladder", OPPORTUNITY_DEFAULT_BUDGET_LADDER)]
    positive_budgets = [int(v) for v in budget_ladder if int(v) > 0]
    preds = _predict_tensors(bundle, rows)
    rows_out: list[dict[str, Any]] = []
    route_scores: dict[str, float] = {}
    for idx, row in enumerate(rows):
        row_out = dict(row)
        row_out["row_index"] = int(idx)
        row_out["expected_gain_next_under_executor"] = float(preds["gain_next"][idx].item())
        row_out["cost_estimate"] = float(preds["cost_pred"][idx].item())
        row_out["fragility_prob"] = float(preds["fragility_prob"][idx].item())
        row_out["route_flip_prob"] = float(preds["route_flip_prob"][idx].item())
        row_out["new_residual_basin_prob"] = float(preds["new_residual_basin_prob"][idx].item())
        row_out["cover_prob_at_0"] = 0.0
        for b_idx, budget in enumerate(positive_budgets):
            row_out[f"cover_prob_at_{int(budget)}"] = float(preds["cover_probs"][idx, b_idx].item())
            row_out[f"cond_gain_pred_at_{int(budget)}_if_covered"] = float(preds["cond_gain_pred"][idx, b_idx].item())
        row_out["acquisition_estimate"] = float(
            row_out["expected_gain_next_under_executor"]
            - row_out["cost_estimate"]
            - (0.25 * row_out["fragility_prob"])
        )
        route_name = str(row_out.get("route_source", "") or "")
        route_scores[route_name] = max(float(route_scores.get(route_name, float("-inf"))), float(row_out["acquisition_estimate"]))
        rows_out.append(row_out)
    rows_out.sort(
        key=lambda item: (
            float(item.get("acquisition_estimate", float("-inf"))),
            float(item.get("expected_gain_next_under_executor", float("-inf"))),
            str(item.get("route_source", "")),
            str(item.get("opportunity_id", "")),
        ),
        reverse=True,
    )
    best_row = rows_out[0] if rows_out else None
    best_route = max(route_scores, key=lambda name: (route_scores[name], name)) if route_scores else None
    out.update({
        "trained": True,
        "budget_ladder": budget_ladder,
        "feature_schema_version": int(bundle.get("feature_schema_version", 1) or 1),
        "witness_energy_feature_enable": bool(bundle.get("witness_energy_feature_enable", False)),
        "best_index": None if best_row is None else int(best_row.get("row_index", 0)),
        "best_route": None if best_route is None else str(best_route),
        "best_opportunity_id": None if best_row is None else str(best_row.get("opportunity_id", "")),
        "route_scores": {str(k): float(v) for k, v in route_scores.items()},
        "rows": rows_out,
    })
    return out


__all__ = [
    "OPPORTUNITY_DEFAULT_BUDGET_LADDER",
    "OPPORTUNITY_FEATURE_NAMES",
    "OPPORTUNITY_FEATURE_SCHEMA_VERSION",
    "OPPORTUNITY_MODEL_KIND",
    "OPPORTUNITY_WITNESS_FEATURE_NAMES",
    "evaluate_opportunity_controller",
    "load_opportunity_bundle",
    "load_opportunity_dataset_rows",
    "opportunity_feature_names",
    "opportunity_feature_vector",
    "predict_opportunity_slate",
    "save_opportunity_bundle",
    "split_opportunity_rows_grouped",
    "train_opportunity_controller",
    "opportunity_row_group_id",
    "opportunity_uses_witness_energy_features",
]
