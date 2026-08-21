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
    predict_unified_candidate_slate,
    save_repair_critic_bundle,
    train_unified_candidate_ranker,
)


def run_unified_candidate_training(
    *,
    report_paths: list[str],
    output_path: str,
    init_bundle_path: str = "",
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    listwise_beta: float = 2.0,
    score_gap_floor: float = 1.0e-3,
    pairwise_gap_scale: float = 0.1,
    path_ce_weight: float = 0.05,
    route_aux_weight: float = 0.0,
    preview_regret_weight: float = 0.10,
    state_value_weight: float = 0.10,
) -> dict[str, Any]:
    rows = load_inverse_experiment_rows(report_paths)
    if not rows:
        raise ValueError("No inverse-experiment rows were loaded from the provided report paths.")
    init_bundle = None
    if str(init_bundle_path or "").strip():
        init_bundle = load_repair_critic_bundle(init_bundle_path)
    bundle = train_unified_candidate_ranker(
        rows,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_fraction=float(val_fraction),
        seed=int(seed),
        listwise_beta=float(listwise_beta),
        score_gap_floor=float(score_gap_floor),
        pairwise_gap_scale=float(pairwise_gap_scale),
        path_ce_weight=float(path_ce_weight),
        route_aux_weight=float(route_aux_weight),
        preview_regret_weight=float(preview_regret_weight),
        state_value_weight=float(state_value_weight),
        init_bundle=init_bundle,
    )
    out_path = Path(output_path)
    save_repair_critic_bundle(bundle, out_path)
    sample_pred = None
    for row in rows:
        pred = predict_unified_candidate_slate(bundle, row)
        if pred.get("trained", False):
            sample_pred = pred
            break
    summary = {
        "report_paths": [str(Path(p)) for p in report_paths],
        "output_path": str(out_path),
        "init_bundle_path": str(init_bundle_path or ""),
        "metrics": dict(bundle.get("metrics", {}) or {}),
        "unified_candidate_ranker_target": str(bundle.get("unified_candidate_ranker_target", "")),
        "unified_candidate_route_tau": float(bundle.get("unified_candidate_route_tau", 1.0) or 1.0),
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
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--listwise_beta", type=float, default=2.0)
    p.add_argument("--score_gap_floor", type=float, default=1.0e-3)
    p.add_argument("--pairwise_gap_scale", type=float, default=0.1)
    p.add_argument("--path_ce_weight", type=float, default=0.05)
    p.add_argument("--route_aux_weight", type=float, default=0.0)
    p.add_argument("--preview_regret_weight", type=float, default=0.10)
    p.add_argument("--state_value_weight", type=float, default=0.10)
    args = p.parse_args()

    summary = run_unified_candidate_training(
        report_paths=list(args.report_paths),
        output_path=str(args.output_path),
        init_bundle_path=str(args.init_bundle_path),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        listwise_beta=float(args.listwise_beta),
        score_gap_floor=float(args.score_gap_floor),
        pairwise_gap_scale=float(args.pairwise_gap_scale),
        path_ce_weight=float(args.path_ce_weight),
        route_aux_weight=float(args.route_aux_weight),
        preview_regret_weight=float(args.preview_regret_weight),
        state_value_weight=float(args.state_value_weight),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
