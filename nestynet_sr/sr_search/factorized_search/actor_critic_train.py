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
    train_repair_controller_actor_critic,
)


def run_actor_critic_training(
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
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
    policy_ce_weight: float = 0.10,
    advantage_clip: float = 5.0,
    reward_target: str = "descendant_preferred",
) -> dict[str, Any]:
    rows = load_inverse_experiment_rows(report_paths)
    if not rows:
        raise ValueError("No inverse-experiment rows were loaded from the provided report paths.")
    init_bundle = None
    if str(init_bundle_path or "").strip():
        init_bundle = load_repair_critic_bundle(init_bundle_path)
    bundle = train_repair_controller_actor_critic(
        rows,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_fraction=float(val_fraction),
        seed=int(seed),
        entropy_weight=float(entropy_weight),
        value_weight=float(value_weight),
        policy_ce_weight=float(policy_ce_weight),
        advantage_clip=float(advantage_clip),
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
        "actor_critic_reward_target": str(bundle.get("actor_critic_reward_target", reward_target)),
        "actor_critic_reward_mean": float(bundle.get("actor_critic_reward_mean", 0.0)),
        "actor_critic_reward_std": float(bundle.get("actor_critic_reward_std", 1.0)),
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
    p.add_argument("--entropy_weight", type=float, default=0.01)
    p.add_argument("--value_weight", type=float, default=0.5)
    p.add_argument("--policy_ce_weight", type=float, default=0.10)
    p.add_argument("--advantage_clip", type=float, default=5.0)
    p.add_argument("--reward_target", type=str, default="descendant_preferred")
    args = p.parse_args()

    summary = run_actor_critic_training(
        report_paths=list(args.report_paths),
        output_path=str(args.output_path),
        init_bundle_path=str(args.init_bundle_path),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        entropy_weight=float(args.entropy_weight),
        value_weight=float(args.value_weight),
        policy_ce_weight=float(args.policy_ce_weight),
        advantage_clip=float(args.advantage_clip),
        reward_target=str(args.reward_target),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
