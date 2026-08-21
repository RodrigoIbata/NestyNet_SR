#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nestynet_sr.sr_search.factorized_search.repair_critic import (
    load_inverse_experiment_rows,
    save_repair_critic_bundle,
    train_repair_critic,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Train a factorized symbolic search Stage-1 repair critic from inverse experiment logs.")
    p.add_argument("--input", nargs="+", required=True, help="JSON files containing inverse_experiment_log rows or oracle reports.")
    p.add_argument("--output", required=True, help="Output path for the trained critic bundle (.pt).")
    p.add_argument("--metrics_out", default=None, help="Optional JSON path for training metrics.")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1.0e-2)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = load_inverse_experiment_rows(args.input)
    bundle = train_repair_critic(
        rows,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    save_repair_critic_bundle(bundle, args.output)

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bundle.get("metrics", {}), indent=2), encoding="utf-8")

    metrics = dict(bundle.get("metrics", {}))
    print(json.dumps({
        "output": str(args.output),
        "n_examples": int(metrics.get("n_examples", 0)),
        "n_train": int(metrics.get("n_train", 0)),
        "n_val": int(metrics.get("n_val", 0)),
        "best_val_loss": float(metrics.get("best_val_loss", 0.0)),
    }, indent=2))


if __name__ == "__main__":
    main()
