# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Nuisance-release robustness pass (Neil's rung 2, submission-safe version).

Releases galaxy-level distance and inclination as penalized nuisances while
the outer law F and the carrier coefficient Upsilon_d stay class-shared:

    y'_dj = y_dj + 2 log10(sin i_cat,d / sin i_d) - delta_d,
    delta_d = log10(D_d / D_cat,d),

using the standard SPARC scalings (g_obs ~ 1/D and 1/sin^2 i; the component
accelerations are distance-independent and treated as inclination-independent,
a stated approximation). Gaussian priors from catalog errors (Table 1 e_D,
e_Inc); the coherent gauge combination c_d = 2 log10(sin i_cat/sin i_d) -
delta_d is recentered to zero weighted mean each iteration (Neil's gauge
fixing: a constant shift of all galaxies is absorbed by F and must not be
spent by the nuisances).

Alternating fit: (law h + profiled Upsilon on corrected data) <-> (per-galaxy
grid optimization of (delta_d, i_d) under priors). Reports carrier stability,
scatter accounting, nuisance pulls, and a whole-galaxy bootstrap of the
released Upsilon_d.

Row sigma: sigma_dex = sqrt((2 e_V/V / ln10)^2 + floor^2), floor 0.05 dex.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LN10 = math.log(10.0)
SIG_FLOOR = 0.05  # dex, intrinsic-scatter floor so tiny e_V rows don't dominate


def load_gold():
    rows = list(csv.DictReader(open(HERE / "data" / "sparc_carrier_gold.csv")))
    gal = np.array([r["galaxy"] for r in rows])
    g_gas = np.array([float(r["g_gas"]) for r in rows])
    g_disk = np.array([float(r["g_disk"]) for r in rows])
    g_obs = np.array([float(r["g_obs"]) for r in rows])
    e_frac = np.array([float(r["e_gobs_frac"]) for r in rows])
    meta = {}
    for r in csv.DictReader(open(HERE / "data" / "sparc_carrier_galaxies.csv")):
        meta[r["galaxy"]] = {"D": float(r["D_Mpc"]), "e_D": float(r["e_D"]),
                             "i": float(r["Inc_deg"]), "e_i": float(r["e_Inc"])}
    return gal, g_gas, g_disk, g_obs, e_frac, meta


def binned_law(lz, y, w, nbins=25):
    """Weighted-median-free simple law: quantile bins, weighted mean per bin."""
    edges = np.quantile(lz, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(edges, lz, side="right") - 1, 0, nbins - 1)
    ctr, val = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 3:
            continue
        ctr.append(0.5 * (edges[b] + edges[b + 1]))
        val.append(np.sum(w[m] * y[m]) / np.sum(w[m]))
    ctr, val = np.array(ctr), np.array(val)
    return lambda q: np.interp(q, ctr, val)


def profile_upsilon(g_gas, g_disk, y, w, grid):
    best_u, best_s, best_law = None, np.inf, None
    for u in grid:
        z = g_gas + u * g_disk
        ok = z > 0
        lz = np.log10(z[ok])
        law = binned_law(lz, y[ok], w[ok])
        s = np.sqrt(np.sum(w[ok] * (y[ok] - law(lz)) ** 2) / np.sum(w[ok]))
        if s < best_s:
            best_u, best_s, best_law = u, s, law
    return best_u, best_s, best_law


def released_fit(gal, g_gas, g_disk, y0, sig, meta, grid,
                 n_outer=4, n_grid=17, n_sig_range=3.0, verbose=False):
    """Alternating shared-law / per-galaxy-nuisance fit. Returns dict."""
    names = [n for n in dict.fromkeys(gal)]  # keep order, unique
    rows_of = {n: np.flatnonzero(gal == n) for n in names}
    delta = {n: 0.0 for n in names}   # log10 D/Dcat
    inc = {n: meta[n]["i"] for n in names}

    def corrected_y():
        yc = y0.copy()
        for n in names:
            r = rows_of[n]
            corr = (2.0 * (math.log10(math.sin(math.radians(meta[n]["i"])))
                           - math.log10(math.sin(math.radians(inc[n]))))
                    - delta[n])
            yc[r] = y0[r] + corr
        return yc

    ups, scat, law = None, None, None
    for it in range(n_outer):
        yc = corrected_y()
        w = 1.0 / sig ** 2
        ups, scat, law = profile_upsilon(g_gas, g_disk, yc, w, grid)
        z = g_gas + ups * g_disk

        # per-galaxy nuisance update on the prior-penalized objective
        for n in names:
            r = rows_of[n]
            zr = z[r]
            okr = zr > 0
            if okr.sum() < 3:
                continue
            lzr = np.log10(zr[okr])
            pred = law(lzr)
            m = meta[n]
            s_delta = max(m["e_D"] / (m["D"] * LN10), 1e-3)
            s_i = max(m["e_i"], 0.5)
            dgrid = np.linspace(-n_sig_range * s_delta, n_sig_range * s_delta, n_grid)
            igrid = np.clip(np.linspace(m["i"] - n_sig_range * s_i,
                                        m["i"] + n_sig_range * s_i, n_grid),
                            15.0, 89.0)
            base = y0[r][okr]
            wr = 1.0 / sig[r][okr] ** 2
            corr_i = 2.0 * (math.log10(math.sin(math.radians(m["i"])))
                            - np.log10(np.sin(np.radians(igrid))))     # (ni,)
            # objective on the (delta, i) grid, vectorized
            resid = (base[None, None, :] + corr_i[None, :, None]
                     - dgrid[:, None, None] - pred[None, None, :])
            chi = (resid ** 2 * wr[None, None, :]).sum(-1)
            chi += ((dgrid / s_delta) ** 2)[:, None]
            chi += (((igrid - m["i"]) / s_i) ** 2)[None, :]
            kd, ki = np.unravel_index(np.argmin(chi), chi.shape)
            delta[n], inc[n] = float(dgrid[kd]), float(igrid[ki])

        # gauge fixing: remove the coherent shift (absorbed by F otherwise)
        c = np.array([2.0 * (math.log10(math.sin(math.radians(meta[n]["i"])))
                             - math.log10(math.sin(math.radians(inc[n]))))
                      - delta[n] for n in names])
        for n, ci in zip(names, c):
            delta[n] += float(np.mean(c))  # shifts every c_d by -mean(c)
        if verbose:
            print(f"  outer {it}: Upsilon={ups:.3f} scatter={scat:.4f} "
                  f"gauge-shift={np.mean(c):+.4f}")

    pulls_d = np.array([delta[n] / max(meta[n]["e_D"] / (meta[n]["D"] * LN10), 1e-3)
                        for n in names])
    pulls_i = np.array([(inc[n] - meta[n]["i"]) / max(meta[n]["e_i"], 0.5)
                        for n in names])
    return {"upsilon": ups, "scatter": scat, "delta": delta, "inc": inc,
            "pulls_d": pulls_d, "pulls_i": pulls_i, "names": names}


def main():
    gal, g_gas, g_disk, g_obs, e_frac, meta = load_gold()
    y0 = np.log10(g_obs)
    sig = np.sqrt((e_frac / LN10) ** 2 + SIG_FLOOR ** 2)
    grid = np.linspace(0.05, 2.5, 99)
    w = 1.0 / sig ** 2

    ups0, scat0, _ = profile_upsilon(g_gas, g_disk, y0, w, grid)
    print(f"baseline (catalog geometry):  Upsilon_d={ups0:.3f}  "
          f"weighted scatter={scat0:.4f} dex")

    res = released_fit(gal, g_gas, g_disk, y0, sig, meta, grid, verbose=True)
    n_gal = len(res["names"])
    print(f"released (D_d, i_d under catalog priors, gauge-fixed):")
    print(f"  Upsilon_d={res['upsilon']:.3f}  weighted scatter={res['scatter']:.4f} dex"
          f"  ({2 * n_gal} nuisance dof over {len(y0)} rows)")
    print(f"  distance pulls: rms {res['pulls_d'].std():.2f}, "
          f"max |{np.abs(res['pulls_d']).max():.2f}|sigma; "
          f"inclination pulls: rms {res['pulls_i'].std():.2f}, "
          f"max |{np.abs(res['pulls_i']).max():.2f}|sigma")
    railed = [(n, p) for n, p in zip(res["names"], res["pulls_i"])
              if abs(p) > 2.9] + [(n, p) for n, p in zip(res["names"], res["pulls_d"])
                                  if abs(p) > 2.9]
    print(f"  galaxies railed near the 3-sigma prior edge: "
          f"{[n for n, _ in railed] if railed else 'none'}")

    # whole-galaxy bootstrap of the released Upsilon
    rng = np.random.default_rng(7)
    names = np.unique(gal)
    by = {n: np.flatnonzero(gal == n) for n in names}
    boots = []
    for b in range(100):
        pick = rng.choice(names, size=len(names), replace=True)
        idx = np.concatenate([by[n] for n in pick])
        gal_b = np.array([f"{g}#{k}" for k, n in enumerate(pick)
                          for g in [n] * len(by[n])])
        meta_b = {f"{n}#{k}": meta[n] for k, n in enumerate(pick)}
        r = released_fit(gal_b, g_gas[idx], g_disk[idx], y0[idx], sig[idx],
                         meta_b, grid, n_outer=3, n_grid=11)
        boots.append(r["upsilon"])
    lo, med, hi = np.percentile(boots, [16, 50, 84])
    print(f"  released-Upsilon galaxy bootstrap (100): "
          f"{med:.3f} 68% [{lo:.3f}, {hi:.3f}]")
    print(f"\ncarrier stability: baseline {ups0:.3f} -> released {res['upsilon']:.3f} "
          f"(shift {res['upsilon'] - ups0:+.3f}, "
          f"vs bootstrap width {hi - lo:.3f})")


if __name__ == "__main__":
    main()
