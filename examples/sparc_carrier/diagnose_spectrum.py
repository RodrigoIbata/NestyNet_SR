# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Diagnose the GS determining-matrix singular spectrum on SPARC.

For surrogates of varying capacity (segment count), print the tail of the
determining-operator singular spectrum, the best tail gap, and the covector
implied by the SMALLEST singular direction (soft-carrier readout), for the
true data and for the two null controls. The pilot go/no-go hinges on the
CONTRAST between these spectra, not on the strict gate alone.
"""

from __future__ import annotations

import numpy as np

from nestynet_sr.sr_gs.affine_algebra import (
    build_affine_determining_matrix,
    _fit_normalization,  # noqa: PLC2701 - diagnostic uses the module's own scaler
)
from run_pilot import fit_surrogate, surrogate_grad_sample, load_rows, split_galaxies, HERE


def spectrum_report(X, y_hat, G, label):
    norm = _fit_normalization(X, y_hat)
    xs = np.asarray(norm.x_scale, dtype=float)
    u = norm.normalize_x(X)
    yn = norm.normalize_y(y_hat)
    gn = G * (xs / float(norm.y_scale))
    D = build_affine_determining_matrix(u, yn, gn)
    _, s, Vt = np.linalg.svd(D / np.sqrt(len(D)), full_matrices=False)
    tail = s / s[0]
    gaps = s[:-1] / s[1:]
    kbest = int(np.argmax(gaps[-4:])) + len(gaps) - 4
    # smallest singular direction -> (A flat, b, alpha, beta), n=2
    v = Vt[-1]
    A = v[:4].reshape(2, 2)
    b = v[4:6]
    alpha, beta = v[6], v[7]
    # physical-space annihilator direction from b (translation part):
    # normalized-space translation covector complement
    print(f"[{label}] tail sv/s0: " + " ".join(f"{x:.2e}" for x in tail[-5:]))
    print(f"  best tail gap {gaps[kbest]:.2f} at k={kbest+1}/{len(s)}; "
          f"last gap {gaps[-1]:.2f}")
    print(f"  smallest dir: |A|={np.abs(A).max():.3f} b={np.round(b,3).tolist()} "
          f"alpha={alpha:.3f} beta={beta:.3f}")
    # If it is translation-like (|A|,alpha,beta small), the invariant covector in
    # NORMALIZED coords is perpendicular to b; convert to physical: c_phys ~ c_norm/x_scale
    if np.abs(A).max() < 0.3 and abs(alpha) < 0.3 and abs(beta) < 0.3:
        c_norm = np.array([-b[1], b[0]])
        c_phys = c_norm / xs
        if c_phys[0] < 0:
            c_phys = -c_phys
        ups = c_phys[1] / c_phys[0] if abs(c_phys[0]) > 1e-12 else np.inf
        print(f"  translation-like: implied Upsilon_d = {ups:.3f}")
    return s


def main():
    gal, g_gas, g_disk, g_obs = load_rows(HERE / "data" / "sparc_carrier_bulgeless.csv")
    scale = float(np.median(g_disk))
    X = np.stack([g_gas / scale, g_disk / scale], axis=1)
    y = np.log10(g_obs)
    disc_names, _ = split_galaxies(gal, 0)
    disc = np.isin(gal, disc_names)
    rng = np.random.default_rng(7)

    for segs in (8, 16, 32):
        _m, leaf, bv = fit_surrogate(X[disc], y[disc], epochs=800, seed=0,
                                     num_segments=segs)
        y_hat, G = surrogate_grad_sample(leaf, X[disc])
        rms = float(np.sqrt(np.mean((y_hat - y[disc]) ** 2)))
        print(f"\n=== segments={segs}  val={bv:.3e}  fit_rms={rms:.4f} dex ===")
        spectrum_report(X[disc], y_hat, G, "true")

        yc = y[disc][rng.permutation(disc.sum())]
        _m2, lf2, _ = fit_surrogate(X[disc], yc, epochs=400, seed=1, num_segments=segs)
        yh2, G2 = surrogate_grad_sample(lf2, X[disc])
        spectrum_report(X[disc], yh2, G2, "y_shuffle")

        Xc = X[disc].copy()
        Xc[:, 0] = Xc[rng.permutation(disc.sum()), 0]
        _m3, lf3, _ = fit_surrogate(Xc, y[disc], epochs=400, seed=2, num_segments=segs)
        yh3, G3 = surrogate_grad_sample(lf3, Xc)
        spectrum_report(Xc, yh3, G3, "gas_shuffle")


if __name__ == "__main__":
    main()
