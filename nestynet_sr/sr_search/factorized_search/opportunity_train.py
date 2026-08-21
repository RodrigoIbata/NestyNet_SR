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

from nestynet_sr.sr_search.factorized_search.opportunity_critic import (
    evaluate_opportunity_controller,
    load_opportunity_bundle,
    load_opportunity_dataset_rows,
    predict_opportunity_slate,
    save_opportunity_bundle,
    train_opportunity_controller,
)


def run_opportunity_training(
    *,
    dataset_paths: list[str],
    output_path: str,
    init_bundle_path: str = "",
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
) -> dict[str, Any]:
    rows = load_opportunity_dataset_rows(dataset_paths)
    if not rows:
        raise ValueError("No opportunity rows were loaded from the provided dataset paths.")
    init_bundle = None
    if str(init_bundle_path or "").strip():
        init_bundle = load_opportunity_bundle(init_bundle_path)
    bundle = train_opportunity_controller(
        rows,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_fraction=float(val_fraction),
        seed=int(seed),
        gain_weight=float(gain_weight),
        cover_weight=float(cover_weight),
        cond_gain_weight=float(cond_gain_weight),
        fragility_weight=float(fragility_weight),
        cost_weight=float(cost_weight),
        route_flip_weight=float(route_flip_weight),
        new_residual_basin_weight=float(new_residual_basin_weight),
        witness_energy_feature_enable=bool(witness_energy_feature_enable),
        init_bundle=init_bundle,
    )
    out_path = Path(output_path)
    save_opportunity_bundle(bundle, out_path)
    loaded = load_opportunity_bundle(out_path)
    sample_prediction = predict_opportunity_slate(loaded, rows[: min(8, len(rows))])
    summary = {
        "dataset_paths": [str(Path(p)) for p in dataset_paths],
        "output_path": str(out_path),
        "init_bundle_path": str(init_bundle_path or ""),
        "witness_energy_feature_enable": bool(witness_energy_feature_enable),
        "metrics": dict(bundle.get("metrics", {}) or {}),
        "calibration": dict(bundle.get("calibration", {}) or {}),
        "full_eval": evaluate_opportunity_controller(loaded, rows),
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
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gain_weight", type=float, default=1.0)
    p.add_argument("--cover_weight", type=float, default=0.5)
    p.add_argument("--cond_gain_weight", type=float, default=0.15)
    p.add_argument("--fragility_weight", type=float, default=0.05)
    p.add_argument("--cost_weight", type=float, default=0.05)
    p.add_argument("--route_flip_weight", type=float, default=0.02)
    p.add_argument("--new_residual_basin_weight", type=float, default=0.02)
    p.add_argument("--witness_energy_feature_enable", action="store_true")
    args = p.parse_args()

    summary = run_opportunity_training(
        dataset_paths=list(args.dataset_paths),
        output_path=str(args.output_path),
        init_bundle_path=str(args.init_bundle_path),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        gain_weight=float(args.gain_weight),
        cover_weight=float(args.cover_weight),
        cond_gain_weight=float(args.cond_gain_weight),
        fragility_weight=float(args.fragility_weight),
        cost_weight=float(args.cost_weight),
        route_flip_weight=float(args.route_flip_weight),
        new_residual_basin_weight=float(args.new_residual_basin_weight),
        witness_energy_feature_enable=bool(args.witness_energy_feature_enable),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
