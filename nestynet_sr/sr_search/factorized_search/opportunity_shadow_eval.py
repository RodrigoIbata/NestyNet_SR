# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .opportunity_dataset import (
    DEFAULT_SHADOW_BUDGET_LADDER,
    build_opportunity_shadow_dataset,
    normalize_shadow_budget_ladder,
)
from .oracle_lab import generate_oracle_shared_candidate_pretrain_dataset


def _clamp01(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return min(1.0, max(0.0, out))


def _stable_fraction(*parts: Any) -> float:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    value = int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)
    return float(value / float(2**64 - 1))


def _shadow_source_row_id(row: Mapping[str, Any], row_index: int) -> str:
    return str(
        row.get("repair_opportunity_slate_id", "")
        or row.get("inverse_repair_slate_id", "")
        or row.get("hole_opportunity_slate_id", "")
        or row.get("build_opportunity_slate_id", "")
        or row.get("controller_build_slate_id", "")
        or row.get("spec_id", "")
        or f"shadow_source_{int(row_index)}"
    )


def shadow_sample_probability_for_depth(
    depth: int,
    *,
    shadow_sample_rate: float,
    depth_oversample_min: int,
    depth_oversample_multiplier: float,
) -> float:
    probability = _clamp01(shadow_sample_rate, 0.0)
    if int(depth) >= int(depth_oversample_min):
        probability *= max(1.0, float(depth_oversample_multiplier))
    return min(1.0, max(0.0, float(probability)))


@dataclass(frozen=True)
class OpportunityShadowEvalConfig:
    shadow_sample_rate: float = 0.10
    budget_ladder: tuple[int, ...] = field(default_factory=lambda: DEFAULT_SHADOW_BUDGET_LADDER)
    depth_oversample_min: int = 6
    depth_oversample_multiplier: float = 2.0
    per_opportunity_timeout_s: float | None = 0.25
    include_repair: bool = True
    include_build: bool = True
    include_hole: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "shadow_sample_rate", _clamp01(self.shadow_sample_rate, 0.10))
        object.__setattr__(self, "budget_ladder", normalize_shadow_budget_ladder(self.budget_ladder))
        object.__setattr__(self, "depth_oversample_min", max(0, int(self.depth_oversample_min)))
        object.__setattr__(self, "depth_oversample_multiplier", max(1.0, float(self.depth_oversample_multiplier)))
        if self.per_opportunity_timeout_s is None:
            return
        object.__setattr__(self, "per_opportunity_timeout_s", max(0.0, float(self.per_opportunity_timeout_s)))


def select_shadow_source_rows(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: OpportunityShadowEvalConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config if isinstance(config, OpportunityShadowEvalConfig) else OpportunityShadowEvalConfig()
    if isinstance(payload, Mapping):
        source_mode = str(payload.get("mode", "") or "")
        rows = [dict(row) for row in list(payload.get("rows", []) or []) if isinstance(row, Mapping)]
    else:
        source_mode = ""
        rows = [dict(row) for row in list(payload or []) if isinstance(row, Mapping)]
    selected_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        depth = int(row.get("truth_depth", row.get("parent_depth", 0)) or 0)
        probability = shadow_sample_probability_for_depth(
            depth,
            shadow_sample_rate=cfg.shadow_sample_rate,
            depth_oversample_min=cfg.depth_oversample_min,
            depth_oversample_multiplier=cfg.depth_oversample_multiplier,
        )
        score = _stable_fraction(source_mode, _shadow_source_row_id(row, row_index), depth, row_index)
        selected = bool(score <= probability)
        decision_rows.append({
            "source_row_index": int(row_index),
            "source_row_id": str(_shadow_source_row_id(row, row_index)),
            "depth": int(depth),
            "selection_probability": float(probability),
            "selection_score": float(score),
            "selected": bool(selected),
        })
        if selected:
            selected_rows.append(row)
    return selected_rows, {
        "source_mode": str(source_mode),
        "n_source_rows_total": int(len(rows)),
        "n_source_rows_sampled": int(len(selected_rows)),
        "config": {
            "shadow_sample_rate": float(cfg.shadow_sample_rate),
            "budget_ladder": [int(v) for v in cfg.budget_ladder],
            "depth_oversample_min": int(cfg.depth_oversample_min),
            "depth_oversample_multiplier": float(cfg.depth_oversample_multiplier),
            "per_opportunity_timeout_s": None if cfg.per_opportunity_timeout_s is None else float(cfg.per_opportunity_timeout_s),
            "include_repair": bool(cfg.include_repair),
            "include_build": bool(cfg.include_build),
            "include_hole": bool(cfg.include_hole),
        },
        "decision_rows": decision_rows,
    }


def build_sampled_opportunity_shadow_dataset(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: OpportunityShadowEvalConfig | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, OpportunityShadowEvalConfig) else OpportunityShadowEvalConfig()
    selected_rows, sampling_meta = select_shadow_source_rows(payload, config=cfg)
    dataset = build_opportunity_shadow_dataset(
        {
            "mode": str(sampling_meta.get("source_mode", "") or ""),
            "rows": selected_rows,
        },
        budget_ladder=cfg.budget_ladder,
        include_repair=bool(cfg.include_repair),
        include_build=bool(cfg.include_build),
        include_hole=bool(cfg.include_hole),
        per_opportunity_timeout_s=cfg.per_opportunity_timeout_s,
    )
    dataset["sampling"] = sampling_meta
    dataset["n_source_rows_total"] = int(sampling_meta.get("n_source_rows_total", 0))
    dataset["n_source_rows_sampled"] = int(sampling_meta.get("n_source_rows_sampled", 0))
    return dataset


def generate_oracle_opportunity_shadow_dataset(
    spec_paths_or_objs,
    *,
    factorized_search_hp=None,
    seeds: Sequence[int] = (0,),
    dtype=None,
    depth_min: int = 1,
    depth_max: int = 99,
    topk: int = 5,
    max_corrupt_paths_per_spec: int | None = 4,
    sweep_all_paths: bool = False,
    sweep_max_paths: int | None = None,
    verbose: bool = False,
    shadow_config: OpportunityShadowEvalConfig | None = None,
) -> dict[str, Any]:
    oracle_payload = generate_oracle_shared_candidate_pretrain_dataset(
        spec_paths_or_objs,
        factorized_search_hp=factorized_search_hp,
        seeds=seeds,
        dtype=dtype,
        depth_min=depth_min,
        depth_max=depth_max,
        topk=topk,
        max_corrupt_paths_per_spec=max_corrupt_paths_per_spec,
        sweep_all_paths=sweep_all_paths,
        sweep_max_paths=sweep_max_paths,
        verbose=verbose,
    )
    dataset = build_sampled_opportunity_shadow_dataset(
        oracle_payload,
        config=shadow_config,
    )
    dataset["oracle_source_mode"] = str(oracle_payload.get("mode", "") or "")
    dataset["oracle_source_n_rows"] = int(oracle_payload.get("n_rows", 0) or 0)
    return dataset


__all__ = [
    "OpportunityShadowEvalConfig",
    "build_sampled_opportunity_shadow_dataset",
    "generate_oracle_opportunity_shadow_dataset",
    "select_shadow_source_rows",
    "shadow_sample_probability_for_depth",
]
