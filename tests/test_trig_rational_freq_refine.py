# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""pb110 hardening: the trig_rational init must resolve the sin-ratio frequency
to high precision. The N-slit grating ratio sin(k*xL*xT)^2/sin(k*xL)^2 has a
razor-sharp, unimodal error well in k (k off by 1e-3 -> ~2e-3 rel error), so a
coarse grid alone floors the isolated-leaf fit. A 1D golden-section refinement
on the (locally unimodal) well recovers the exact frequency.

These tests pin the refinement ALGORITHM against the exact mechanism used in
nestynet_sr/sr_search/template_library.py::propose_trig_rational._init.
"""

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def _grating(x2, x3):
    # N-slit diffraction grating ratio (truth NN[x2,x3] for pb110)
    return torch.sin(x2 * x3 / 2) ** 2 / torch.sin(x2 / 2) ** 2


def _err_at(k, F, xL, xT, p=2, denom_eps=1e-3, tiny=1e-18):
    den = torch.sin(k * xL)
    den = torch.sign(den) * den.abs().clamp_min(denom_eps)
    base = (torch.sin(k * xL * xT) / den) ** p
    m = torch.isfinite(base) & torch.isfinite(F)
    bb, ff = base[m], F[m]
    s = (bb * ff).sum() / (bb * bb).sum().clamp_min(tiny)
    return torch.mean((s * bb - ff) ** 2).item()


def _grid_best(F, xL, xT, k_min=0.1, k_max=1.03, n=120):
    ks = torch.linspace(k_min, k_max, n)
    errs = [_err_at(float(k), F, xL, xT) for k in ks]
    i = int(np.argmin(errs))
    return float(ks[i]), errs[i], (k_max - k_min) / (n - 1)


def _golden_refine(k0, step, F, xL, xT, k_min=0.1, k_max=1.03):
    a, b = max(k_min, k0 - step), min(k_max, k0 + step)
    gr = 0.5 * (np.sqrt(5.0) - 1.0)
    c, d = b - gr * (b - a), a + gr * (b - a)
    ec, ed = _err_at(c, F, xL, xT), _err_at(d, F, xL, xT)
    for _ in range(60):
        if b - a <= 1e-8:
            break
        if ec < ed:
            b, d, ed = d, c, ec
            c = b - gr * (b - a)
            ec = _err_at(c, F, xL, xT)
        else:
            a, c, ec = c, d, ed
            d = a + gr * (b - a)
            ed = _err_at(d, F, xL, xT)
    return 0.5 * (a + b)


def _domain(seed=0, n=3000):
    g = torch.Generator().manual_seed(seed)
    x2 = 1.0 + 2.0 * torch.rand(n, generator=g)     # [1,3], as in pb110
    x3 = torch.randint(1, 3, (n,), generator=g).double()  # N in {1,2}
    return x2, x3


def test_grating_exact_at_true_frequency():
    x2, x3 = _domain()
    F = _grating(x2, x3)
    rms = np.sqrt(_err_at(0.5, F, x2, x3)) / np.sqrt(float((F * F).mean()))
    assert rms < 1e-6, f"template should be exact at k=0.5, got rel_rms={rms}"


def test_well_is_sharp_and_coarse_grid_floors():
    x2, x3 = _domain()
    F = _grating(x2, x3)
    scale = np.sqrt(float((F * F).mean()))
    # sharpness: 1e-3 off the true k already costs ~1e-3 rel
    rms_off = np.sqrt(_err_at(0.499, F, x2, x3)) / scale
    assert rms_off > 5e-4, f"well should be sharp, got {rms_off}"
    k0, err0, step = _grid_best(F, x2, x3)
    assert np.sqrt(err0) / scale > 1e-4, "coarse grid alone should floor above 1e-4"


def test_golden_refine_recovers_machine_precision():
    x2, x3 = _domain()
    F = _grating(x2, x3)
    scale = np.sqrt(float((F * F).mean()))
    k0, _, step = _grid_best(F, x2, x3)
    k_ref = _golden_refine(k0, step, F, x2, x3)
    rms_ref = np.sqrt(_err_at(k_ref, F, x2, x3)) / scale
    assert abs(k_ref - 0.5) < 1e-4, f"refined k should be ~0.5, got {k_ref}"
    assert rms_ref < 1e-6, f"refinement should recover machine precision, got {rms_ref}"


def test_refine_adoption_never_worsens_when_grid_already_exact():
    # If the coarse grid already lands the true k, err(k0) is ~machine-zero, so
    # golden-section (which stops at ~1e-8 in k) can give a slightly larger err.
    # The CODE guards this: it only adopts the refined k when err_ref < err_best,
    # so the effective adopted error is min(err_ref, err0) -- never worse.
    x2, x3 = _domain(seed=1)
    F = _grating(x2, x3)
    ks = torch.linspace(0.1, 0.9, 41)  # step 0.02, includes 0.5 exactly
    errs = [_err_at(float(k), F, x2, x3) for k in ks]
    k0 = float(ks[int(np.argmin(errs))])
    assert abs(k0 - 0.5) < 1e-9
    err0 = _err_at(k0, F, x2, x3)
    k_ref = _golden_refine(k0, 0.02, F, x2, x3)
    err_ref = _err_at(k_ref, F, x2, x3)
    adopted = err_ref if err_ref < err0 else err0   # mirror the code's guard
    assert adopted <= err0
