#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Prototype: does a grad_weight ramp rescue the divergent Ey component?

Ey (mw002, out_idx=1) diverged under canonical-init Sobolev training
(best_val ~1.6e4 at noise=0, ~1.6e5 at 1e-6), even though it converged with
random init in the validated run (~1.15e-6).  The friend's diagnosis: the
value-projected canonical basin is fine, but full-weight gradient matching kicks
the LM into a ravine.  Fix to test: keep canonical init, then ramp the gradient
weight in over stages (1e-4 -> 1).  This trains Ey alone (cheap) and compares.
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


def run(noise: float, ramp, epochs: int) -> None:
    p = PROBLEM_REGISTRY["mw002"]
    X, Y_clean, G, _ = build_problem_data(p, fast=False)
    Y = add_relative_noise(Y_clean, float(noise), 0)
    # Sobolev targets: spatial FFT of the (noisy) field + exact-generator time.
    G_target = build_derivative_targets("spectral_spatial_exact_time", X, Y, G)
    t0 = time.time()
    _surro, best_val, entry = rb._train_single_component(
        1,  # out_idx = Ey
        problem=p, X=X, Y=Y, G_target=G_target, G_exact=G,
        data_dir=Path("/tmp/ramp_ey_work"),
        num_segments=16, epochs=int(epochs), loss_target=1e-8, batch_size=2000,
        ndata_train=4000, ndata_val=2000, device=torch.device("cpu"), dtype=torch.float64,
        objective="sobolev", axes=(0, 1, 2, 3), sobolev_target="spectral_spatial_exact_time",
        sobolev_value_weight=1.0, sobolev_grad_weight=1.0, sobolev_normalize="rms",
        canonical_init=True, grad_weight_ramp=list(ramp), verbose=True,
    )
    dt = time.time() - t0
    print("=" * 78)
    print(f"RAMP Ey (noise={noise:g}, ramp={list(ramp)}): "
          f"best_val={best_val:.3e}  grad_vs_exact_abs_rms={entry.get('grad_vs_exact_abs_rms')}  ({dt:.0f}s)")
    print("reference: single-stage canonical Ey = 1.586e4 (DIVERGED);  "
          "random-init validated Ey = 1.15e-6 (converged)")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=300, help="epochs per ramp stage (early-stop applies)")
    ap.add_argument("--ramp", type=str, default="1e-4,1e-3,1e-2,1e-1,1.0")
    args = ap.parse_args()
    ramp = [float(x) for x in args.ramp.split(",") if x.strip()]
    run(args.noise, ramp, args.epochs)


if __name__ == "__main__":
    main()
