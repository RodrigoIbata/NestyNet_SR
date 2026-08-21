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

from nestynet_sr.sr_search.factorized_search.repair_critic import (
    load_inverse_experiment_rows,
    load_repair_critic_bundle,
    predict_repair_build_route,
    save_repair_critic_bundle,
    train_repair_build_route_comparator,
)


def run_repair_route_training(
    *,
    report_paths: list[str],
    output_path: str,
    init_bundle_path: str = "",
    repair_tuple_bundle_path: str = "",
    build_tuple_bundle_path: str = "",
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    margin_floor: float = 1.0e-3,
    margin_loss_weight: float = 0.25,
) -> dict[str, Any]:
    rows = load_inverse_experiment_rows(report_paths)
    if not rows:
        raise ValueError("No inverse-experiment rows were loaded from the provided report paths.")
    init_bundle = None
    if str(init_bundle_path or "").strip():
        init_bundle = load_repair_critic_bundle(init_bundle_path)
    repair_tuple_bundle = None
    if str(repair_tuple_bundle_path or "").strip():
        repair_tuple_bundle = load_repair_critic_bundle(repair_tuple_bundle_path)
    build_tuple_bundle = None
    if str(build_tuple_bundle_path or "").strip():
        build_tuple_bundle = load_repair_critic_bundle(build_tuple_bundle_path)
    bundle = train_repair_build_route_comparator(
        rows,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_fraction=float(val_fraction),
        seed=int(seed),
        margin_floor=float(margin_floor),
        margin_loss_weight=float(margin_loss_weight),
        repair_tuple_bundle=repair_tuple_bundle,
        build_tuple_bundle=build_tuple_bundle,
        init_bundle=init_bundle,
    )
    out_path = Path(output_path)
    save_repair_critic_bundle(bundle, out_path)
    sample_pred = None
    for row in rows:
        pred = predict_repair_build_route(
            bundle,
            row,
            repair_tuple_bundle=repair_tuple_bundle,
            build_tuple_bundle=build_tuple_bundle,
        )
        if pred.get("trained", False):
            sample_pred = pred
            break
    summary = {
        "report_paths": [str(Path(p)) for p in report_paths],
        "output_path": str(out_path),
        "init_bundle_path": str(init_bundle_path or ""),
        "repair_tuple_bundle_path": str(repair_tuple_bundle_path or ""),
        "build_tuple_bundle_path": str(build_tuple_bundle_path or ""),
        "metrics": dict(bundle.get("metrics", {}) or {}),
        "repair_build_route_compare_target": str(bundle.get("repair_build_route_compare_target", "")),
        "repair_build_route_compare_margin_floor": float(bundle.get("repair_build_route_compare_margin_floor", 0.0) or 0.0),
        "repair_build_route_compare_margin_loss_weight": float(bundle.get("repair_build_route_compare_margin_loss_weight", 0.0) or 0.0),
        "repair_build_route_compare_uses_repair_tuple_features": bool(bundle.get("repair_build_route_compare_uses_repair_tuple_features", False)),
        "repair_build_route_compare_uses_build_tuple_features": bool(bundle.get("repair_build_route_compare_uses_build_tuple_features", False)),
        "sample_prediction": sample_pred,
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report_paths", nargs="+", required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--init_bundle_path", type=str, default="")
    p.add_argument("--repair_tuple_bundle_path", type=str, default="")
    p.add_argument("--build_tuple_bundle_path", type=str, default="")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin_floor", type=float, default=1.0e-3)
    p.add_argument("--margin_loss_weight", type=float, default=0.25)
    args = p.parse_args()

    summary = run_repair_route_training(
        report_paths=list(args.report_paths),
        output_path=str(args.output_path),
        init_bundle_path=str(args.init_bundle_path),
        repair_tuple_bundle_path=str(args.repair_tuple_bundle_path),
        build_tuple_bundle_path=str(args.build_tuple_bundle_path),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        margin_floor=float(args.margin_floor),
        margin_loss_weight=float(args.margin_loss_weight),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
