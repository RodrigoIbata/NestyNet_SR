#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
"""Train one Maxwell field-component surrogate in its own process.

Invoked by ``_run_components_subprocess`` for true multi-core parallelism (the
six component fits are independent).  Reads a torch bundle (data + config),
trains the requested component via the shared ``_train_single_component``, and
writes back the trained ``state_dict`` + diagnostics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_benchmark as rb  # noqa: E402
from problem_defs import PROBLEM_REGISTRY  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one Maxwell component surrogate.")
    ap.add_argument("--bundle", required=True, help="torch bundle: data + config")
    ap.add_argument("--out_idx", type=int, required=True, help="component index to train")
    ap.add_argument("--result", required=True, help="output path for state_dict + entry")
    # Optional per-job overrides (used by restart rounds); absent -> use bundle cfg.
    ap.add_argument("--canonical_init", default=None, help="'0'/'1'/'none' override")
    ap.add_argument("--init_seed", type=int, default=None, help="random-init seed override")
    ap.add_argument("--grad_weight_ramp", default=None, help="csv of weights or 'none'")
    ap.add_argument("--epochs", type=int, default=None, help="per-stage epochs override")
    args = ap.parse_args()

    torch.set_num_threads(1)  # single-threaded BLAS: one core per worker, no oversubscription
    blob = torch.load(args.bundle, weights_only=False)
    cfg = dict(blob["cfg"])
    if args.canonical_init is not None:
        cfg["canonical_init"] = None if args.canonical_init == "none" else bool(int(args.canonical_init))
    if args.init_seed is not None:
        cfg["init_seed"] = int(args.init_seed)
    if args.grad_weight_ramp is not None:
        cfg["grad_weight_ramp"] = (None if args.grad_weight_ramp == "none"
                                   else [float(x) for x in args.grad_weight_ramp.split(",") if x.strip()])
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    dtype = torch.float64 if cfg.pop("dtype_str") == "float64" else torch.float32
    problem = PROBLEM_REGISTRY[blob["problem_id"]]

    surrogate_i, best_val, entry = rb._train_single_component(
        int(args.out_idx),
        problem=problem,
        X=blob["X"],
        Y=blob["Y"],
        G_target=blob["G_target"],
        G_exact=blob["G_exact"],
        H_target=blob.get("H_target"),
        device=torch.device("cpu"),
        dtype=dtype,
        data_dir=Path(cfg.pop("data_dir")),
        axes=tuple(cfg.pop("axes")),
        **cfg,
    )
    torch.save(
        {
            "out_idx": int(args.out_idx),
            "state_dict": surrogate_i.state_dict(),
            "best_val": float(best_val),
            "entry": entry,
        },
        args.result,
    )


if __name__ == "__main__":
    main()
