# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""FSS free-search for the outer map F(z) on the gold sample.

Runs the factorized-symbolic-search explorer on (x0, y) =
(log10 z + 13, log10 g_obs) with discovery galaxies as the fit set and
held-out galaxies as the probe set, so the search's own scoring is
held-out-driven. The +13 shift keeps x0 positive (sqrt/log skeletons legal).
Candidates are scored afterwards with the same weighted held-out rms (dex)
as the parametric family in outer_map.py, for a like-for-like comparison.

Budget follows the verified fast recipe: brute_depth=3 dominates cost for
1-input problems (the default of 10 is a >10-minute trap).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from nestynet_sr.sr_search.factorized_search.bridge import run_explorer
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node
from nestynet_sr.sr_search.factorized_search.expr_mapping import eval_mapping

from nuisance_release import LN10, SIG_FLOOR, load_gold
from outer_map import UPS, wrms
from run_pilot import split_galaxies

HERE = Path(__file__).resolve().parent
XSHIFT = 13.0


def main():
    gal, g_gas, g_disk, g_obs, e_frac, _meta = load_gold()
    z = g_gas + UPS * g_disk
    ok = z > 0
    gal = gal[ok]
    x = np.log10(z[ok]) + XSHIFT
    y = np.log10(g_obs[ok])
    sig = np.sqrt((e_frac[ok] / LN10) ** 2 + SIG_FLOOR ** 2)
    w = 1.0 / sig ** 2

    disc_names, _held = split_galaxies(gal, 0)
    disc = np.isin(gal, disc_names)
    held = ~disc

    xf = torch.tensor(x[disc].reshape(-1, 1), dtype=torch.float64)
    yf = torch.tensor(y[disc], dtype=torch.float64)
    xp = torch.tensor(x[held].reshape(-1, 1), dtype=torch.float64)
    yp = torch.tensor(y[held], dtype=torch.float64)

    t0 = time.time()
    res = run_explorer(
        dtype=torch.float64,
        x_fit_data=xf, y_fit_data=yf, x_probe_data=xp, y_probe_data=yp,
        var_dims=None, y_dims=None,
        n_iter=2000, max_depth=4, poly_degree=3,
        brute_depth=3, brute_max_expressions=2000,
        return_topk=8, simplify_skeletons=False,
        seed=0, verbose=False, print_every=0,
        wall_time_limit_s=420,
    )
    print(f"FSS explorer: {len(res)} candidates in {time.time() - t0:.0f}s\n")

    xa = torch.tensor(x.reshape(-1, 1), dtype=torch.float64)
    print(f"{'skeleton':<38} {'mapping':<10} {'size':>4} {'fit wrms':>9} "
          f"{'held-out':>9}")
    rows = []
    for r in res:
        try:
            pred = eval_mapping(eval_node(r["toy_ast"], xa), r["mapping"])
            pred = pred.reshape(-1).numpy()
        except Exception:
            continue
        if not np.all(np.isfinite(pred)):
            continue
        rf = wrms(y[disc] - pred[disc], w[disc])
        rh = wrms(y[held] - pred[held], w[held])
        # fitted constants in the mapping: count float-valued scalars and
        # every element of array/tensor values; skip ints (orders) and strings
        n_map = 0
        for v in r["mapping"].values():
            if hasattr(v, "numel"):
                n_map += int(v.numel())
            elif isinstance(v, np.ndarray):
                n_map += int(v.size)
            elif isinstance(v, (list, tuple)):
                n_map += sum(1 for u in np.ravel(v) if not float(u).is_integer()
                             or isinstance(u, float))
            elif isinstance(v, float):
                n_map += 1
        rows.append({"expr": r["expr"], "kind": r["mapping"].get("kind"),
                     "size": r["size"], "n_map": n_map,
                     "wrms_fit": rf, "wrms_ho": rh, "pred": pred})
        print(f"{r['expr']:<38} {r['mapping'].get('kind'):<10} {r['size']:>4} "
              f"{rf:9.4f} {rh:9.4f}")

    if rows:
        best = min(rows, key=lambda r: r["wrms_ho"])
        print(f"\nbest by held-out: {best['expr']}  ({best['kind']}, "
              f"wrms {best['wrms_ho']:.4f} dex)")
        np.savez(HERE / "results" / "fss_outer.npz",
                 x=x, y=y, held=held,
                 best_pred=best["pred"], best_expr=best["expr"],
                 table=np.array([(r["expr"], r["kind"], r["size"], r["n_map"],
                                  r["wrms_fit"], r["wrms_ho"]) for r in rows],
                                dtype=object))


if __name__ == "__main__":
    main()
