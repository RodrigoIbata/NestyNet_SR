# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Planted-carrier control on the real SPARC support.

Generates y = log10 F_RAR(g_gas + 0.5 g_disk) on the REAL bulgeless input
rows (F_RAR = McGaugh interpolation, gdag = 1.2e-10 m/s^2), under three noise
models, then runs the identical surrogate + gradient carrier readouts as the
pilot. This separates method failure from data truth:

  clean   : no noise               -> must recover Upsilon = 0.5, else the
                                      gradient lane is inadequate on this
                                      support geometry, full stop
  iid     : 0.12 dex row noise     -> recovery tests noise robustness
  galaxy  : 0.10 dex galaxy offsets + 0.06 dex row noise -> recovery tests
                                      robustness to the inclination/distance
                                      gauge modes Neil flagged
"""

from __future__ import annotations

import numpy as np

from run_pilot import (HERE, closure_rms, fit_surrogate, gs_certificate, load_rows,
                       orientation_carrier, split_galaxies, surrogate_grad_sample,
                       translation_sector, warp, warp_grad_factor)

GDAG = 1.2e-10  # m/s^2
UPS_TRUE = 0.5


def rar(z):
    """McGaugh+2016 interpolation g_obs = z / (1 - exp(-sqrt(z/gdag)))."""
    s = np.sqrt(np.clip(z, 1e-16, None) / GDAG)
    return z / (1.0 - np.exp(-s))


def main():
    gal, g_gas, g_disk, g_obs = load_rows(HERE / "data" / "sparc_carrier_bulgeless.csv")
    scale = float(np.median(g_disk))
    X = np.stack([g_gas / scale, g_disk / scale], axis=1)
    disc_names, held_names = split_galaxies(gal, 0)
    disc = np.isin(gal, disc_names)
    held = ~disc

    z_phys = np.clip(g_gas + UPS_TRUE * g_disk, 1e-14, None)
    y_true = np.log10(rar(z_phys))
    rng = np.random.default_rng(42)
    gal_off = {n: rng.normal(0.0, 0.10) for n in np.unique(gal)}
    variants = {
        "clean": y_true,
        "iid": y_true + rng.normal(0.0, 0.12, len(y_true)),
        "galaxy": (y_true + np.array([gal_off[n] for n in gal])
                   + rng.normal(0.0, 0.06, len(y_true))),
    }

    warp_a = np.array([np.median(np.abs(X[:, 0])), np.median(np.abs(X[:, 1]))])
    U = warp(X, warp_a)

    for name, y in variants.items():
        _m, leaf, bv = fit_surrogate(U[disc], y[disc], epochs=600, seed=0,
                                     num_segments=32)
        y_hat, G_u = surrogate_grad_sample(leaf, U[disc])
        G = G_u * warp_grad_factor(X[disc], warp_a)
        fit_rms = float(np.sqrt(np.mean((y_hat - y[disc]) ** 2)))
        r1, cov = orientation_carrier(G, gal[disc])
        sec, _ = translation_sector(X[disc], y_hat, G)
        cert, cov_strict = gs_certificate(X[disc], y_hat, G)
        rms_true = closure_rms(np.array([1.0, UPS_TRUE]), X[disc], y[disc],
                               X[held], y[held])
        rms_found = closure_rms(cov, X[disc], y[disc], X[held], y[held])
        print(f"[{name}] val {bv:.3e} fit_rms {fit_rms:.4f} dex")
        print(f"  strict: {'COMPILED ups=%.3f' % cert['upsilon_d'] if cov_strict is not None else 'abstains'}"
              f" ({cert['quotient_policy']}, gap {cert['spectral_gap']})")
        print(f"  orientation: Upsilon {r1['upsilon_d']:+.3f} (true {UPS_TRUE}), "
              f"contrast {r1['contrast']:.1f}")
        print(f"  sector: Upsilon {sec['upsilon_d']:+.3f}, purity {sec['output_purity']:.3f}, "
              f"gap {sec['gap']:.2f}")
        print(f"  closure heldout: found-cov {rms_found:.4f} vs true-cov {rms_true:.4f} dex")


if __name__ == "__main__":
    main()
