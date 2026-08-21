#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Prototype: canonical-then-random multi-start ("restart until pass") for Ey.

Protocol (deterministic, predeclared -> a globalized initializer, not seed-
hunting; selection is on the surrogate's own Sobolev validation loss, blind to
the downstream discovery):
  restart 0 : value-projected canonical init + grad-weight ramp.
  restart k : random init seeded by k, single-stage gw=1 (the config that gave
              Ey best_val 1.15e-6 in the validated run).
Stop at the first restart whose validation loss is below --threshold; cap at
--max_restarts.  Reference: single-canonical Ey ~1e-3 (1.9% deriv error);
validated random Ey 1.15e-6 (0.05%).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import run_benchmark as rb
from problem_defs import PROBLEM_REGISTRY, build_problem_data
from spectral_derivatives import build_derivative_targets
from noise_sweep import add_relative_noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=1e-4, help="val-loss pass bar")
    ap.add_argument("--max_restarts", type=int, default=5)
    ap.add_argument("--ramp", type=str, default="0,1e-4,1e-3,1e-2,1e-1,1.0")
    a = ap.parse_args()
    ramp = [float(x) for x in a.ramp.split(",")]

    p = PROBLEM_REGISTRY["mw002"]
    X, Yc, G, _ = build_problem_data(p, fast=False)
    Y = add_relative_noise(Yc, float(a.noise), 0)
    Gt = build_derivative_targets("spectral_spatial_exact_time", X, Y, G)

    common = dict(
        problem=p, X=X, Y=Y, G_target=Gt, G_exact=G, data_dir=Path("/tmp/restart_ey_work"),
        num_segments=16, loss_target=1e-8, batch_size=2000, ndata_train=4000, ndata_val=2000,
        device=torch.device("cpu"), dtype=torch.float64, objective="sobolev", axes=(0, 1, 2, 3),
        sobolev_target="spectral_spatial_exact_time", sobolev_value_weight=1.0,
        sobolev_grad_weight=1.0, sobolev_normalize="rms", verbose=True,
    )
    plan = [("canonical+ramp", dict(canonical_init=True, grad_weight_ramp=ramp, init_seed=None, epochs=500))]
    for k in range(a.max_restarts - 1):
        plan.append((f"random_seed{k}", dict(canonical_init=False, grad_weight_ramp=None, init_seed=k, epochs=1500)))

    best = None
    for name, kw in plan:
        t0 = time.time()
        _surro, bv, entry = rb._train_single_component(1, **common, **kw)  # out_idx=1 -> Ey
        gve = entry.get("grad_vs_exact_abs_rms")
        print(f">>> {name}: best_val={bv:.3e}  grad_vs_exact={gve}  ({time.time()-t0:.0f}s)", flush=True)
        if best is None or bv < best[1]:
            best = (name, bv, gve)
        if bv < float(a.threshold):
            print(f">>> PASS at {name}: val {bv:.3e} < {a.threshold:.0e}", flush=True)
            break

    print("=" * 78)
    print(f"RESTART-Ey (noise={a.noise:g}): best = {best[0]}  best_val={best[1]:.3e}  grad_vs_exact={best[2]}")
    print("reference: single-canonical Ey ~1e-3 (1.9% deriv); validated random Ey 1.15e-6 (0.05%)")
    print("=" * 78)


if __name__ == "__main__":
    main()
