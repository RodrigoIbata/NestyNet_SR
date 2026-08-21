# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""GS -> FSS carrier-seed handoff on the clean planted control.

The full-pipeline demonstration the vignette could not run at catalog noise:
on the CLEAN planted carrier (exact RAR over z = g_gas + 0.5 g_disk written
onto the real gold-sample rows) the determining operator certifies carriers,
`discover_gs_carrier_seeds` converts them to factorized-search seeds, and the
explorer arbitrates among them by held-out closure. Compared against an
unseeded run at matched budget. Because skeletons carry no free constants,
the oblique real-coefficient carrier is REACHABLE only through the seed, so
a decisive gap is the expected signature of a working handoff.

Success criteria (fixed in advance):
  1. at least one certified seed carries the planted covector (ratio ~ 0.5)
  2. the seeded arm's best held-out closure beats the unseeded arm's clearly
     at matched budget, with the winning expression built on the seeded z
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from nestynet_sr.sr_search.factorized_search.bridge import run_explorer
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node
from nestynet_sr.sr_search.factorized_search.expr_mapping import eval_mapping
from nestynet_sr.sr_search.factorized_search.gs_carrier_seed import (
    default_gs_carrier_cfg, discover_gs_carrier_seeds)

from nuisance_release import load_gold
from run_pilot import fit_surrogate, split_galaxies, warp
from synth_check import UPS_TRUE, rar

HERE = Path(__file__).resolve().parent

BUDGET = dict(n_iter=1500, max_depth=4, poly_degree=3,
              brute_depth=3, brute_max_expressions=2000,
              return_topk=5, simplify_skeletons=False,
              seed=0, verbose=False, print_every=0,
              wall_time_limit_s=300)


def wrms_dex(y_true, pred):
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def score_results(res, xa, y, disc, held, label):
    rows = []
    for r in res:
        try:
            pred = eval_mapping(eval_node(r["toy_ast"], xa), r["mapping"])
            pred = pred.reshape(-1).numpy()
        except Exception:
            continue
        if not np.all(np.isfinite(pred)):
            continue
        rows.append({"expr": r["expr"], "kind": r["mapping"].get("kind"),
                     "size": r["size"],
                     "rms_fit": wrms_dex(y[disc], pred[disc]),
                     "rms_ho": wrms_dex(y[held], pred[held])})
    rows.sort(key=lambda r: r["rms_ho"])
    print(f"\n[{label}] top candidates by held-out rms (dex):")
    for r in rows[:5]:
        print(f"  {r['expr']:<44} {r['kind']:<8} size {r['size']:>3} "
              f"fit {r['rms_fit']:.4f}  held-out {r['rms_ho']:.4f}")
    return rows


def main():
    gal, g_gas, g_disk, g_obs, _e, _m = load_gold()
    scale = float(np.median(g_disk))
    X = np.stack([g_gas / scale, g_disk / scale], axis=1)
    z_phys = np.clip(g_gas + UPS_TRUE * g_disk, 1e-14, None)
    y = np.log10(rar(z_phys))                      # clean planted target

    disc = np.isin(gal, split_galaxies(gal, 0)[0])
    held = ~disc

    # surrogate in warped inputs; expose a physical-coordinate callable so
    # autograd chain-rules the warp exactly
    warp_a = np.array([np.median(np.abs(X[:, 0])), np.median(np.abs(X[:, 1]))])
    U = warp(X, warp_a)
    _m2, leaf, bv = fit_surrogate(U[disc], y[disc], epochs=600, seed=0)
    a_t = torch.tensor(warp_a, dtype=torch.float64)

    def target_fn(Z):
        return leaf(torch.asinh(Z / a_t)).reshape(-1, 1)

    print(f"surrogate on clean planted data: val {bv:.3e}")

    # --- 1. GS carrier seeds from the certified algebra
    t0 = time.time()
    seeds, diag = discover_gs_carrier_seeds(
        target_fn, torch.tensor(X[disc], dtype=torch.float64), n_var=2,
        cfg=default_gs_carrier_cfg(), max_seeds=8)
    print(f"\nGS carrier seeds: {len(seeds)} in {time.time()-t0:.0f}s")
    for s in seeds:
        print(f"  seed: {s}")
    for d in diag[:10]:
        kind = d.get("kind", "?")
        if kind in ("carrier", "gs_carrier", "shadow_reduction"):
            print(f"  diag[{kind}]: "
                  + ", ".join(f"{k}={v}" for k, v in d.items()
                              if k in ("chart", "certified", "covector",
                                       "confidence", "reason")))

    xf = torch.tensor(X[disc], dtype=torch.float64)
    yf = torch.tensor(y[disc], dtype=torch.float64)
    xp = torch.tensor(X[held], dtype=torch.float64)
    yp = torch.tensor(y[held], dtype=torch.float64)
    xa = torch.tensor(X, dtype=torch.float64)

    # --- 2. seeded vs unseeded explorer at matched budget
    t0 = time.time()
    res_seeded = run_explorer(dtype=torch.float64,
                              x_fit_data=xf, y_fit_data=yf,
                              x_probe_data=xp, y_probe_data=yp,
                              var_dims=None, y_dims=None,
                              carrier_seed_exprs=tuple(seeds), **BUDGET)
    t_seed = time.time() - t0
    t0 = time.time()
    res_plain = run_explorer(dtype=torch.float64,
                             x_fit_data=xf, y_fit_data=yf,
                             x_probe_data=xp, y_probe_data=yp,
                             var_dims=None, y_dims=None, **BUDGET)
    t_plain = time.time() - t0

    rows_s = score_results(res_seeded, xa, y, disc, held,
                           f"seeded ({t_seed:.0f}s)")
    rows_p = score_results(res_plain, xa, y, disc, held,
                           f"unseeded ({t_plain:.0f}s)")

    if rows_s and rows_p:
        bs, bp = rows_s[0], rows_p[0]
        print(f"\nverdict: seeded best {bs['rms_ho']:.4f} dex "
              f"({bs['expr']}) vs unseeded best {bp['rms_ho']:.4f} dex "
              f"({bp['expr']})")
        print("handoff " + ("SUCCEEDS" if bs["rms_ho"] < 0.5 * bp["rms_ho"]
                            else "does NOT separate at this budget"))


if __name__ == "__main__":
    main()
