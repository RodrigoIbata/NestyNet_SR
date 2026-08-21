# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""symexp_denom_1d hardening: identifiability statistics, decision helper, and
the comparative-null discrimination property the pb114 fix relies on."""

import math

import torch

from nestynet_sr.sr_search._candidate_builders_univariate import (
    _eval_ratpoly_1d,
    _symexp_gate_decision,
    _symexp_scale_and_u,
)
from nestynet_sr.sr_search.fitting_utils import _fit_rational_coeffs_1d

torch.manual_seed(0)


def _sech_slice(n=2000, scale=2.0, slope=1.2):
    z = torch.linspace(-3.0, 3.0, n, dtype=torch.float64)
    f = scale / (2.0 * torch.cosh(slope * z))
    return z, f


def _monotone_exp_slice(n=2000):
    z = torch.linspace(0.5, 4.0, n, dtype=torch.float64)
    return z, torch.exp(-z)


def test_scale_and_u_on_genuine_sech():
    _, f = _sech_slice()
    scale0, u, doublings = _symexp_scale_and_u(f)
    assert doublings == 0
    # data reach the cosh turnover: u_min ~ 1.05 by construction of scale0
    assert float(u.min()) < 1.3


def test_scale_and_u_quantile_construction_makes_u_min_small_even_on_tails():
    """Documents why u_q01 alone cannot discriminate: scale0 = 2.1*q99(F)
    puts u_min ~ 1.05 on essentially any bounded slice, including a pure
    exponential tail. The comparative null, not the u-statistic, must carry
    the discrimination (see test below)."""
    _, f = _monotone_exp_slice()
    scale0, u, doublings = _symexp_scale_and_u(f)
    assert doublings == 0  # the doubling loop cannot fire for eps-positive F
    assert float(u.min()) < 1.3


def test_gate_decision_thresholds_and_null_margin():
    # fit quality alone
    assert _symexp_gate_decision(0.02, eff_threshold=0.05, null_rms=None, null_margin=0.5)
    assert not _symexp_gate_decision(0.06, eff_threshold=0.05, null_rms=None, null_margin=0.5)
    assert not _symexp_gate_decision(float("nan"), eff_threshold=0.05, null_rms=None, null_margin=0.5)
    # comparative null: symexp must be 2x better than the simple rational
    assert _symexp_gate_decision(0.01, eff_threshold=0.05, null_rms=0.30, null_margin=0.5)
    assert not _symexp_gate_decision(0.034, eff_threshold=0.05, null_rms=0.05, null_margin=0.5)
    # tightened tail threshold rejects the pb114-class values
    assert not _symexp_gate_decision(0.034, eff_threshold=0.01, null_rms=None, null_margin=0.5)


def test_ratpoly_null_discriminates_sech_bump_from_monotone_slice():
    """The load-bearing property: a Mobius (deg 1/1) rational on f cannot fit a
    genuine symmetric sech bump (so real sech candidates beat the null) but
    fits a smooth monotone slice well (so degenerate tail fits lose to it)."""
    z_bump, f_bump = _sech_slice()
    fit = _fit_rational_coeffs_1d(z_bump, f_bump, deg_num=1, deg_den=1, min_points=100)
    assert fit is not None
    a, b = fit
    resid = f_bump - _eval_ratpoly_1d(z_bump, a, b)
    null_rms_bump = float(torch.sqrt(torch.mean(resid**2)) / torch.sqrt(torch.mean(f_bump**2)))

    z_mono, f_mono = _monotone_exp_slice()
    fit2 = _fit_rational_coeffs_1d(z_mono, f_mono, deg_num=1, deg_den=1, min_points=100)
    assert fit2 is not None
    a2, b2 = fit2
    resid2 = f_mono - _eval_ratpoly_1d(z_mono, a2, b2)
    null_rms_mono = float(torch.sqrt(torch.mean(resid2**2)) / torch.sqrt(torch.mean(f_mono**2)))

    # bump: null fails badly; monotone: null is decent
    assert null_rms_bump > 0.15, f"null unexpectedly fits sech bump: {null_rms_bump}"
    assert null_rms_mono < 0.10, f"null unexpectedly poor on monotone slice: {null_rms_mono}"
    assert null_rms_bump > 3.0 * null_rms_mono


def test_eval_ratpoly_guarded_denominator():
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0, 0.0], dtype=torch.float64)  # zero denominator
    out = _eval_ratpoly_1d(x, a, b)
    assert torch.isfinite(out).all()
    assert math.isfinite(float(out.abs().max()))
