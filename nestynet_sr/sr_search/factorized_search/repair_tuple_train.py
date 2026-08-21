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
    save_repair_critic_bundle,
    train_repair_controller_tuple_ranker,
)


def run_repair_tuple_training(
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
    path_ce_weight: float = 0.10,
    route_aux_weight: float = 0.05,
    preview_value_weight: float = 0.10,
    preview_regret_weight: float = 0.10,
    state_value_weight: float = 0.10,
    child_value_lambda: float = 0.25,
    reward_target: str = "descendant_preferred",
) -> dict[str, Any]:
    rows = load_inverse_experiment_rows(report_paths)
    if not rows:
        raise ValueError("No inverse-experiment rows were loaded from the provided report paths.")
    init_bundle = None
    if str(init_bundle_path or "").strip():
        init_bundle = load_repair_critic_bundle(init_bundle_path)
    bundle = train_repair_controller_tuple_ranker(
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
        preview_value_weight=float(preview_value_weight),
        preview_regret_weight=float(preview_regret_weight),
        state_value_weight=float(state_value_weight),
        child_value_lambda=float(child_value_lambda),
        reward_target=str(reward_target or "descendant_preferred"),
        init_bundle=init_bundle,
    )
    out_path = Path(output_path)
    save_repair_critic_bundle(bundle, out_path)
    summary = {
        "report_paths": [str(Path(p)) for p in report_paths],
        "output_path": str(out_path),
        "init_bundle_path": str(init_bundle_path or ""),
        "metrics": dict(bundle.get("metrics", {}) or {}),
        "repair_tuple_ranker_target": str(bundle.get("repair_tuple_ranker_target", "")),
        "repair_tuple_preview_value_target": str(bundle.get("repair_tuple_preview_value_target", "")),
        "repair_tuple_regret_target": str(bundle.get("repair_tuple_regret_target", "")),
        "repair_tuple_child_value_lambda": float(bundle.get("repair_tuple_child_value_lambda", 0.0) or 0.0),
        "repair_tuple_regret_weight": float(bundle.get("repair_tuple_regret_weight", 0.0) or 0.0),
        "repair_tuple_action_names": list(bundle.get("repair_tuple_action_names", []) or []),
        "actor_critic_reward_target": str(bundle.get("actor_critic_reward_target", reward_target)),
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
    p.add_argument("--path_ce_weight", type=float, default=0.10)
    p.add_argument("--route_aux_weight", type=float, default=0.05)
    p.add_argument("--preview_value_weight", type=float, default=0.10)
    p.add_argument("--preview_regret_weight", type=float, default=0.10)
    p.add_argument("--state_value_weight", type=float, default=0.10)
    p.add_argument("--child_value_lambda", type=float, default=0.25)
    p.add_argument("--reward_target", type=str, default="descendant_preferred")
    args = p.parse_args()

    summary = run_repair_tuple_training(
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
        preview_value_weight=float(args.preview_value_weight),
        preview_regret_weight=float(args.preview_regret_weight),
        state_value_weight=float(args.state_value_weight),
        child_value_lambda=float(args.child_value_lambda),
        reward_target=str(args.reward_target),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
