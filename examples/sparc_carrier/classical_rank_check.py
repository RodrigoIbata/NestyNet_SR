# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Classical rank-one check: does a 1D carrier capture everything a 2D fit can?

Grouped (by-galaxy) K-fold cross-validation of
  1D: log10 g_obs ~ s(log10 z),  z = g_gas + Upsilon * g_disk (Upsilon fit on train)
  2D: log10 g_obs ~ s2(log10 g_gas', log10 g_disk)  unrestricted smooth
using a distance-weighted k-NN smoother in standardized log coords
(g_gas has 68 mildly negative rows: use asinh scaling for it).

If held-out performance of 1D matches 2D, the data support a rank-one carrier
(Neil's "dimensional reduction" criterion) even where the carrier coefficient
itself is softly identified.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A0 = 1.0e-11  # asinh softening scale for g_gas, m/s^2 (well below data median)


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    gal = np.array([r["galaxy"] for r in rows])
    return (gal,
            np.array([float(r["g_gas"]) for r in rows]),
            np.array([float(r["g_disk"]) for r in rows]),
            np.array([float(r["g_obs"]) for r in rows]))


def knn_fit_predict(Xtr, ytr, Xte, k=25):
    """Distance-weighted k-NN smoother in standardized coordinates."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    d2 = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d2, axis=1)[:, :k]
    dn = np.take_along_axis(d2, nn, axis=1)
    w = 1.0 / (dn + 1e-6)
    return (w * ytr[nn]).sum(1) / w.sum(1)


def fit_ups_1d(g_gas, g_disk, g_obs, grid):
    """Best Upsilon by binned-median collapse scatter (train data only)."""
    best_u, best_s = None, np.inf
    for u in grid:
        z = g_gas + u * g_disk
        ok = z > 0
        lz, lo = np.log10(z[ok]), np.log10(g_obs[ok])
        order = np.argsort(lz)
        lz, lo = lz[order], lo[order]
        edges = np.quantile(lz, np.linspace(0, 1, 26))
        idx = np.clip(np.searchsorted(edges, lz, side="right") - 1, 0, 24)
        med = np.array([np.median(lo[idx == b]) if np.any(idx == b) else np.nan
                        for b in range(25)])
        ctr = 0.5 * (edges[:-1] + edges[1:])
        good = ~np.isnan(med)
        s = np.sqrt(np.mean((lo - np.interp(lz, ctr[good], med[good])) ** 2))
        if s < best_s:
            best_u, best_s = u, s
    return best_u


def main():
    gal, g_gas, g_disk, g_obs = load(HERE / "data" / "sparc_carrier_bulgeless.csv")
    y = np.log10(g_obs)
    x_gas = np.arcsinh(g_gas / A0)          # handles the 68 negative rows
    x_disk = np.log10(g_disk)
    names = np.unique(gal)
    rng = np.random.default_rng(20260808)
    rng.shuffle(names)
    folds = np.array_split(names, 5)
    grid = np.linspace(0.05, 2.0, 40)

    r1, r2, rfix = [], [], []
    for k, held in enumerate(folds):
        te = np.isin(gal, held)
        tr = ~te
        # 1D carrier model, Upsilon fit on train
        u = fit_ups_1d(g_gas[tr], g_disk[tr], g_obs[tr], grid)
        z_tr, z_te = g_gas + u * g_disk, None
        Xtr1 = np.log10(np.clip(z_tr[tr], 1e-15, None))[:, None]
        Xte1 = np.log10(np.clip(z_tr[te], 1e-15, None))[:, None]
        p1 = knn_fit_predict(Xtr1, y[tr], Xte1)
        # 1D carrier with fixed Upsilon = 0.5 (population prior)
        z5 = g_gas + 0.5 * g_disk
        p5 = knn_fit_predict(np.log10(z5[tr])[:, None], y[tr], np.log10(z5[te])[:, None])
        # unrestricted 2D
        X2 = np.stack([x_gas, x_disk], axis=1)
        p2 = knn_fit_predict(X2[tr], y[tr], X2[te])
        r1.append(y[te] - p1)
        rfix.append(y[te] - p5)
        r2.append(y[te] - p2)
        print(f"fold {k}: n_test={te.sum():4d}  Ups_train={u:.2f}  "
              f"rms1D={np.sqrt(np.mean(r1[-1]**2)):.4f}  "
              f"rms1D(0.5)={np.sqrt(np.mean(rfix[-1]**2)):.4f}  "
              f"rms2D={np.sqrt(np.mean(r2[-1]**2)):.4f}")

    def rms(rs):
        r = np.concatenate(rs)
        return np.sqrt(np.mean(r ** 2))

    print(f"\nheld-out-galaxy RMS (dex): 1D carrier {rms(r1):.4f}   "
          f"1D fixed-0.5 {rms(rfix):.4f}   2D unrestricted {rms(r2):.4f}")
    print("rank-one supported" if rms(r2) > 0.95 * rms(r1) else
          "2D beats 1D by >5%: transverse structure present")


if __name__ == "__main__":
    main()
