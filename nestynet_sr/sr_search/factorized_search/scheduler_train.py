# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nestynet_sr.sr_search.factorized_search.scheduler_critic import (
    evaluate_scheduler_critic,
    load_scheduler_bundle,
    load_scheduler_dataset_rows,
    predict_scheduler_plan_slate,
    save_scheduler_bundle,
    train_scheduler_critic,
)


def _parse_weight_spec(spec: str, *, int_keys: bool = False) -> dict[Any, float]:
    token = str(spec or "").strip()
    if not token:
        return {}
    out: dict[Any, float] = {}
    for item in token.split(","):
        piece = str(item or "").strip()
        if not piece or "=" not in piece:
            continue
        key_raw, value_raw = piece.split("=", 1)
        key = int(key_raw.strip()) if int_keys else str(key_raw.strip())
        out[key] = float(value_raw.strip())
    return out


def _parse_alias_spec(spec: str) -> dict[str, str]:
    token = str(spec or "").strip()
    if not token:
        return {}
    out: dict[str, str] = {}
    for item in token.split(","):
        piece = str(item or "").strip()
        if not piece or "=" not in piece:
            continue
        key_raw, value_raw = piece.split("=", 1)
        key = str(key_raw.strip())
        value = str(value_raw.strip())
        if key and value:
            out[key] = value
    return out


def run_scheduler_training(
    *,
    dataset_paths: list[str],
    output_path: str,
    init_bundle_path: str = "",
    budget_ladder: list[int] | None = None,
    threshold_ladder: list[float] | None = None,
    route_aliases: dict[str, str] | None = None,
    hidden_dim: int = 64,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    ensemble_size: int = 4,
    break_weight: float = 1.0,
    tail_weight: float = 0.5,
    route_win_weight: float = 0.4,
    new_residual_basin_weight: float = 0.15,
    fragile_weight: float = 0.1,
    stable_weight: float = 0.1,
    cost_weight: float = 0.1,
    rank_weight: float = 0.1,
    route_weights: dict[str, float] | None = None,
    budget_weight_map: dict[int, float] | None = None,
    deep_repair_min_depth: int = 5,
    deep_repair_weight: float = 1.0,
    hole_oversample_repeat: int = 1,
    deep_repair_oversample_repeat: int = 1,
    witness_energy_feature_enable: bool = False,
    objective_mode: str | None = None,
    objective_hybrid_mix: float | None = None,
) -> dict[str, Any]:
    rows = load_scheduler_dataset_rows(dataset_paths)
    if not rows:
        raise ValueError("No scheduler rows were loaded from the provided dataset paths.")
    init_bundle = None
    if str(init_bundle_path or "").strip():
        init_bundle = load_scheduler_bundle(init_bundle_path)
    bundle = train_scheduler_critic(
        rows,
        budget_ladder=budget_ladder,
        threshold_ladder=threshold_ladder,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_fraction=float(val_fraction),
        seed=int(seed),
        ensemble_size=int(ensemble_size),
        break_weight=float(break_weight),
        tail_weight=float(tail_weight),
        route_win_weight=float(route_win_weight),
        new_residual_basin_weight=float(new_residual_basin_weight),
        fragile_weight=float(fragile_weight),
        stable_weight=float(stable_weight),
        cost_weight=float(cost_weight),
        rank_weight=float(rank_weight),
        route_aliases=route_aliases,
        route_weights=route_weights,
        budget_weight_map=budget_weight_map,
        deep_repair_min_depth=int(deep_repair_min_depth),
        deep_repair_weight=float(deep_repair_weight),
        hole_oversample_repeat=int(hole_oversample_repeat),
        deep_repair_oversample_repeat=int(deep_repair_oversample_repeat),
        witness_energy_feature_enable=bool(witness_energy_feature_enable),
        objective_mode=objective_mode,
        objective_hybrid_mix=objective_hybrid_mix,
        init_bundle=init_bundle,
    )
    out_path = Path(output_path)
    save_scheduler_bundle(bundle, out_path)
    loaded = load_scheduler_bundle(out_path)
    sample_prediction = predict_scheduler_plan_slate(loaded, rows[: min(8, len(rows))])
    summary = {
        "dataset_paths": [str(Path(p)) for p in dataset_paths],
        "output_path": str(out_path),
        "init_bundle_path": str(init_bundle_path or ""),
        "budget_ladder": None if budget_ladder is None else [int(v) for v in budget_ladder],
        "threshold_ladder": None if threshold_ladder is None else [float(v) for v in threshold_ladder],
        "route_aliases": dict(route_aliases or {}),
        "witness_energy_feature_enable": bool(bundle.get("witness_energy_feature_enable", False)),
        "objective_mode": str(bundle.get("objective_mode", "acquisition") or "acquisition"),
        "objective_hybrid_mix": float(bundle.get("objective_hybrid_mix", 0.5) or 0.5),
        "metrics": dict(bundle.get("metrics", {}) or {}),
        "training_rebalance": dict(bundle.get("training_rebalance", {}) or {}),
        "full_eval": evaluate_scheduler_critic(loaded, rows),
        "sample_prediction": sample_prediction,
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_paths", nargs="+", required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--init_bundle_path", type=str, default="")
    p.add_argument("--budget_ladder", type=str, default="")
    p.add_argument("--threshold_ladder", type=str, default="")
    p.add_argument("--route_aliases", type=str, default="")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ensemble_size", type=int, default=4)
    p.add_argument("--break_weight", type=float, default=1.0)
    p.add_argument("--tail_weight", type=float, default=0.5)
    p.add_argument("--route_win_weight", type=float, default=0.4)
    p.add_argument("--new_residual_basin_weight", type=float, default=0.15)
    p.add_argument("--fragile_weight", type=float, default=0.1)
    p.add_argument("--stable_weight", type=float, default=0.1)
    p.add_argument("--cost_weight", type=float, default=0.1)
    p.add_argument("--rank_weight", type=float, default=0.1)
    p.add_argument("--route_weights", type=str, default="")
    p.add_argument("--budget_weights", type=str, default="")
    p.add_argument("--deep_repair_min_depth", type=int, default=5)
    p.add_argument("--deep_repair_weight", type=float, default=1.0)
    p.add_argument("--hole_oversample_repeat", type=int, default=1)
    p.add_argument("--deep_repair_oversample_repeat", type=int, default=1)
    p.add_argument("--witness_energy_feature_enable", action="store_true")
    p.add_argument("--objective_mode", type=str, default="")
    p.add_argument("--objective_hybrid_mix", type=float, default=None)
    args = p.parse_args()

    summary = run_scheduler_training(
        dataset_paths=list(args.dataset_paths),
        output_path=str(args.output_path),
        init_bundle_path=str(args.init_bundle_path),
        budget_ladder=[
            int(token.strip())
            for token in str(args.budget_ladder or "").split(",")
            if token.strip()
        ] or None,
        threshold_ladder=[
            float(token.strip())
            for token in str(args.threshold_ladder or "").split(",")
            if token.strip()
        ] or None,
        route_aliases=_parse_alias_spec(str(args.route_aliases or "")),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        ensemble_size=int(args.ensemble_size),
        break_weight=float(args.break_weight),
        tail_weight=float(args.tail_weight),
        route_win_weight=float(args.route_win_weight),
        new_residual_basin_weight=float(args.new_residual_basin_weight),
        fragile_weight=float(args.fragile_weight),
        stable_weight=float(args.stable_weight),
        cost_weight=float(args.cost_weight),
        rank_weight=float(args.rank_weight),
        route_weights=_parse_weight_spec(str(args.route_weights or ""), int_keys=False),
        budget_weight_map=_parse_weight_spec(str(args.budget_weights or ""), int_keys=True),
        deep_repair_min_depth=int(args.deep_repair_min_depth),
        deep_repair_weight=float(args.deep_repair_weight),
        hole_oversample_repeat=int(args.hole_oversample_repeat),
        deep_repair_oversample_repeat=int(args.deep_repair_oversample_repeat),
        witness_energy_feature_enable=bool(args.witness_energy_feature_enable),
        objective_mode=(None if not str(args.objective_mode or "").strip() else str(args.objective_mode)),
        objective_hybrid_mix=args.objective_hybrid_mix,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
