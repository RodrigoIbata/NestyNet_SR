# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Outer-map family for F(z) on the gold sample (closure-honesty pass).

Fits a compact family of candidate outer laws g_obs = F(z) on the discovery
galaxies (weighted, log space) and scores them on fully held-out galaxies.
Candidates whose held-out rms differ by less than the galaxy-bootstrap noise
are reported as observationally indistinguishable — the paper's claim is the
FAMILY, not a unique interpolation function.

Candidates (z in m/s^2, gdag = free acceleration scale):
  rar_exp   F = z / (1 - exp(-sqrt(z/gdag)))          (McGaugh+2016)
  simple_nu F = z * (1/2 + sqrt(1/4 + gdag/z))        (simple interpolation)
  superpos  F = z + sqrt(gdag * z)                    (Newton + deep-MOND sum)
  powerlaw  log F = a + b log z                       (scale-free null)
  logparab  log F = a + b u + c u^2, u = log z        (agnostic curvature)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from nuisance_release import LN10, SIG_FLOOR, load_gold, binned_law
from run_pilot import split_galaxies

HERE = Path(__file__).resolve().parent
UPS = 0.55  # carrier coefficient (gold-sample profile / bootstrap median)


def make_candidates():
    def rar_exp(z, p):
        gd = 10.0 ** p[0]
        s = np.sqrt(z / gd)
        return np.log10(z / (1.0 - np.exp(-s)))

    def simple_nu(z, p):
        gd = 10.0 ** p[0]
        return np.log10(z * (0.5 + np.sqrt(0.25 + gd / z)))

    def superpos(z, p):
        gd = 10.0 ** p[0]
        return np.log10(z + np.sqrt(gd * z))

    def powerlaw(z, p):
        return p[0] + p[1] * np.log10(z)

    def logparab(z, p):
        u = np.log10(z) + 10.5
        return p[0] + p[1] * u + p[2] * u ** 2

    return {
        "rar_exp": (rar_exp, [-9.9]),
        "simple_nu": (simple_nu, [-10.1]),
        "superpos": (superpos, [-10.1]),
        "powerlaw": (powerlaw, [-4.0, 0.6]),
        "logparab": (logparab, [-10.2, 0.6, 0.05]),
    }


def wrms(r, w):
    return float(np.sqrt(np.sum(w * r ** 2) / np.sum(w)))


def main():
    gal, g_gas, g_disk, g_obs, e_frac, _meta = load_gold()
    y = np.log10(g_obs)
    sig = np.sqrt((e_frac / LN10) ** 2 + SIG_FLOOR ** 2)
    w = 1.0 / sig ** 2
    z = g_gas + UPS * g_disk
    ok = z > 0
    gal, y, w, z = gal[ok], y[ok], w[ok], z[ok]

    disc_names, held_names = split_galaxies(gal, 0)
    disc = np.isin(gal, disc_names)
    held = ~disc

    # nonparametric reference (same estimator as the pilot)
    law_np = binned_law(np.log10(z[disc]), y[disc], w[disc])
    rms_np = wrms(y[held] - law_np(np.log10(z[held])), w[held])

    # galaxy-bootstrap noise scale on held-out rms (nonparametric law)
    rng = np.random.default_rng(3)
    boots = []
    hn = np.array(sorted(set(gal[held])))
    by = {n: np.flatnonzero(gal == n) for n in hn}
    for _ in range(200):
        idx = np.concatenate([by[n] for n in rng.choice(hn, len(hn), True)])
        boots.append(wrms(y[idx] - law_np(np.log10(z[idx])), w[idx]))
    rms_noise = float(np.std(boots))

    print(f"carrier z = g_gas + {UPS:.2f} g_disk; discovery {disc.sum()} rows, "
          f"held-out {held.sum()} rows")
    print(f"nonparametric law: held-out wrms {rms_np:.4f} dex "
          f"(bootstrap sd {rms_noise:.4f})\n")
    print(f"{'candidate':<12} {'params':<28} {'fit wrms':>9} {'held-out':>9} "
          f"{'vs np':>7}")

    results = {}
    for name, (fn, p0) in make_candidates().items():
        def loss(p):
            r = y[disc] - fn(z[disc], np.atleast_1d(p))
            return np.sum(w[disc] * r ** 2)
        opt = minimize(loss, np.array(p0), method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 4000})
        p = opt.x
        rms_fit = wrms(y[disc] - fn(z[disc], p), w[disc])
        rms_ho = wrms(y[held] - fn(z[held], p), w[held])
        results[name] = {"p": p, "rms_fit": rms_fit, "rms_ho": rms_ho, "fn": fn}
        ptxt = ", ".join(f"{v:.3f}" for v in p)
        flag = (rms_ho - rms_np) / rms_noise
        print(f"{name:<12} [{ptxt:<26}] {rms_fit:9.4f} {rms_ho:9.4f} "
              f"{flag:+6.1f}s")

    best = min(results.values(), key=lambda r: r["rms_ho"])["rms_ho"]
    indist = [n for n, r in results.items() if r["rms_ho"] - best < rms_noise]
    print(f"\nobservationally indistinguishable at 1 bootstrap-sigma: {indist}")
    for name in ("rar_exp", "simple_nu", "superpos"):
        gd = 10.0 ** results[name]["p"][0]
        print(f"  {name}: gdag = {gd:.2e} m/s^2")
    np.save(HERE / "results" / "outer_map_family.npy",
            {n: {"p": r["p"], "rms_fit": r["rms_fit"], "rms_ho": r["rms_ho"]}
             for n, r in results.items()}, allow_pickle=True)


if __name__ == "__main__":
    main()
