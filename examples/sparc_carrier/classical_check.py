# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Classical reference for the SPARC carrier pilot (no NestyNet machinery).

Profiles the disk mass-to-light ratio Upsilon_d directly: for each trial
Upsilon, collapse z = g_gas + Upsilon * g_disk, fit a monotone 1D smooth of
log10 g_obs vs log10 z (isotonic-then-binned spline surrogate: here a simple
moving-quantile fit), and record the residual scatter. The minimizing Upsilon
is what the GS carrier detection should reproduce; galaxy bootstrap gives its
classical uncertainty. This is the pilot's oracle cross-check, not the result.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    gal = np.array([r["galaxy"] for r in rows])
    g_gas = np.array([float(r["g_gas"]) for r in rows])
    g_disk = np.array([float(r["g_disk"]) for r in rows])
    g_obs = np.array([float(r["g_obs"]) for r in rows])
    return gal, g_gas, g_disk, g_obs


def collapse_scatter(ups: float, g_gas, g_disk, g_obs, nbins: int = 25) -> float:
    """RMS residual (dex) of log g_obs about a binned-median curve in log z."""
    z = g_gas + ups * g_disk
    ok = z > 0
    lz, lo = np.log10(z[ok]), np.log10(g_obs[ok])
    order = np.argsort(lz)
    lz, lo = lz[order], lo[order]
    edges = np.quantile(lz, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(edges, lz, side="right") - 1, 0, nbins - 1)
    med = np.array([np.median(lo[idx == b]) if np.any(idx == b) else np.nan
                    for b in range(nbins)])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    good = ~np.isnan(med)
    pred = np.interp(lz, ctr[good], med[good])
    return float(np.sqrt(np.mean((lo - pred) ** 2)))


def profile_ups(g_gas, g_disk, g_obs, grid) -> tuple[float, np.ndarray]:
    s = np.array([collapse_scatter(u, g_gas, g_disk, g_obs) for u in grid])
    return float(grid[int(np.argmin(s))]), s


def main():
    gal, g_gas, g_disk, g_obs = load(HERE / "data" / "sparc_carrier_bulgeless.csv")
    grid = np.linspace(0.05, 2.0, 79)

    ups_hat, prof = profile_ups(g_gas, g_disk, g_obs, grid)
    smin = prof.min()
    print(f"profile minimum: Upsilon_d = {ups_hat:.3f}  (collapse scatter {smin:.4f} dex)")
    print(f"scatter at Upsilon=0.5: {collapse_scatter(0.5, g_gas, g_disk, g_obs):.4f} dex")

    # Galaxy bootstrap of the profile minimum.
    rng = np.random.default_rng(20260808)
    names = np.unique(gal)
    by_gal = defaultdict(list)
    for i, g in enumerate(gal):
        by_gal[g].append(i)
    boots = []
    for _ in range(200):
        pick = rng.choice(names, size=len(names), replace=True)
        idx = np.concatenate([by_gal[n] for n in pick])
        u, _ = profile_ups(g_gas[idx], g_disk[idx], g_obs[idx], grid)
        boots.append(u)
    boots = np.array(boots)
    lo_q, hi_q = np.percentile(boots, [16, 84])
    print(f"galaxy bootstrap (200): median {np.median(boots):.3f}, "
          f"68% interval [{lo_q:.3f}, {hi_q:.3f}]")

    # Carrier angle in the (g_gas, g_disk) plane and its bootstrap spread.
    ang = np.degrees(np.arctan2(boots, 1.0))
    ang_hat = math.degrees(math.atan2(ups_hat, 1.0))
    print(f"carrier angle atan(Upsilon/1): {ang_hat:.2f} deg, "
          f"bootstrap sd {ang.std():.2f} deg")


if __name__ == "__main__":
    main()
