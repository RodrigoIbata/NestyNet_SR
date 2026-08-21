# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Tests for RuleUnivariateMulPeel — product decomposition via log-derivative + NN oracle.

The detection pipeline has three stages:
1. _estimate_monomial_power(z, f, f') → k  (monomial exponent from M(z) = z*f'/f)
2. _estimate_exp_rate(f, f', z, k) → mu  (exponential rate from corrected log-derivative)
3. discover_trig_from_data(z, residual) → TrigAxisSpec  (trig in residual)

Tests verify:
- Monomial power estimation (k) from clean data
- Exponential rate estimation (mu) from clean NN-oracle-like data
- Combined pipeline correctly detects mono*trig, mono*exp, mono*exp*trig products
- Combined pipeline correctly detects exp*trig products (original pattern)
- Combined pipeline correctly rejects pure trig, pure exp, and polynomials
"""

import numpy as np
import pytest
import torch

from nestynet_sr.sr_search.features import discover_trig_from_data
from nestynet_sr.sr_search.stageB.rules import RuleUnivariateMulPeel


def _make_uniform_grid(func, dfunc, z_min=-5.0, z_max=5.0, N=512):
    """Generate (z, f, f') on a clean uniform grid (mimics NN oracle)."""
    z = np.linspace(z_min, z_max, N)
    f = func(z)
    fp = dfunc(z)
    return z, f, fp


def _run_full_detection(z, f, fp):
    """Run the two-stage detection pipeline: exp rate + trig in residual.

    Returns (mu, trig_spec) or (None, None) if detection fails.
    Applies the same min-cycles filter as the rule's propose method.
    """
    import math

    mu = RuleUnivariateMulPeel._estimate_exp_rate(f, fp)
    if mu is None or abs(mu) < 0.01:
        return mu, None

    z_t = torch.from_numpy(z).double()
    f_t = torch.from_numpy(f).double()
    residual = f_t * torch.exp(-mu * z_t)

    trig_spec = discover_trig_from_data(
        z_t, residual, strength_threshold=5.0, max_omega=1000.0,
    )
    if trig_spec is not None:
        z_span = float(z_t[-1] - z_t[0])
        n_cycles = trig_spec.omega * z_span / (2.0 * math.pi)
        if n_cycles < 2.0:
            trig_spec = None
    return mu, trig_spec


# ──────────────────────────────────────────────────────────────
# Test _estimate_exp_rate
# ──────────────────────────────────────────────────────────────


def test_exp_rate_pure_exp():
    """exp(-0.5*z): mu should be -0.5."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.exp(-0.5 * z),
        lambda z: -0.5 * np.exp(-0.5 * z),
    )
    mu = RuleUnivariateMulPeel._estimate_exp_rate(f, fp)
    assert mu is not None
    assert abs(mu - (-0.5)) < 0.01, f"mu={mu:.4f}, expected -0.5"


def test_exp_rate_exp_times_cos():
    """exp(-0.5*z)*cos(3*z): mu should still be -0.5 (median of f'/f)."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.exp(-0.5 * z) * np.cos(3.0 * z),
        lambda z: -0.5 * np.exp(-0.5 * z) * np.cos(3.0 * z)
        + np.exp(-0.5 * z) * (-3.0 * np.sin(3.0 * z)),
    )
    mu = RuleUnivariateMulPeel._estimate_exp_rate(f, fp)
    assert mu is not None
    assert abs(mu - (-0.5)) < 0.15, f"mu={mu:.4f}, expected -0.5"


def test_exp_rate_pure_trig_near_zero():
    """cos(3*z): mu should be near zero."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.cos(3.0 * z),
        lambda z: -3.0 * np.sin(3.0 * z),
    )
    mu = RuleUnivariateMulPeel._estimate_exp_rate(f, fp)
    # mu may be None (if all f near zero) or close to 0
    if mu is not None:
        assert abs(mu) < 0.1, f"mu={mu:.4f}, expected ~0"


# ──────────────────────────────────────────────────────────────
# Test combined detection pipeline
# ──────────────────────────────────────────────────────────────


def test_exp_times_cos_detected():
    """exp(-0.5*z) * cos(3*z) should be detected as exp*trig."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.exp(-0.5 * z) * np.cos(3.0 * z),
        lambda z: -0.5 * np.exp(-0.5 * z) * np.cos(3.0 * z)
        + np.exp(-0.5 * z) * (-3.0 * np.sin(3.0 * z)),
    )
    mu, trig_spec = _run_full_detection(z, f, fp)

    assert mu is not None, "Should detect exponential component"
    assert abs(mu - (-0.5)) < 0.15, f"mu={mu:.4f}"
    assert trig_spec is not None, "Should detect trig in residual"
    assert trig_spec.omega > 1.0, f"omega={trig_spec.omega:.2f}, expected >1"


def test_exp_times_sin_detected():
    """exp(0.3*z) * sin(5*z) should be detected as exp*trig."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.exp(0.3 * z) * np.sin(5.0 * z),
        lambda z: 0.3 * np.exp(0.3 * z) * np.sin(5.0 * z)
        + np.exp(0.3 * z) * 5.0 * np.cos(5.0 * z),
    )
    mu, trig_spec = _run_full_detection(z, f, fp)

    assert mu is not None, "Should detect exponential component"
    assert abs(mu - 0.3) < 0.15, f"mu={mu:.4f}"
    assert trig_spec is not None, "Should detect trig in residual"


def test_scaled_exp_trig_detected():
    """2.0 * exp(-0.5*z) * cos(3*z) should still be detected."""
    amp = 2.0
    z, f, fp = _make_uniform_grid(
        lambda z: amp * np.exp(-0.5 * z) * np.cos(3.0 * z),
        lambda z: amp * (-0.5 * np.exp(-0.5 * z) * np.cos(3.0 * z)
                         + np.exp(-0.5 * z) * (-3.0 * np.sin(3.0 * z))),
    )
    mu, trig_spec = _run_full_detection(z, f, fp)

    assert mu is not None, "Should detect exponential component"
    assert abs(mu - (-0.5)) < 0.15, f"mu={mu:.4f}"
    assert trig_spec is not None, "Should detect trig in residual"


# ──────────────────────────────────────────────────────────────
# Test rejection of non-target patterns
# ──────────────────────────────────────────────────────────────


def test_pure_trig_rejected():
    """cos(3*z) alone: mu ~ 0 → rule skips (no exponential component)."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.cos(3.0 * z),
        lambda z: -3.0 * np.sin(3.0 * z),
    )
    mu, trig_spec = _run_full_detection(z, f, fp)

    # mu should be near zero → the |mu| < 0.01 gate rejects it
    if mu is not None:
        assert abs(mu) < 0.1, f"mu={mu:.4f}, expected ~0"
    # trig_spec should be None because pipeline short-circuits on small mu
    assert trig_spec is None, "Pure trig should be rejected (no exp component)"


def test_pure_exp_rejected():
    """exp(-0.5*z) alone: no trig in residual → discover_trig_from_data returns None."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.exp(-0.5 * z),
        lambda z: -0.5 * np.exp(-0.5 * z),
    )
    mu, trig_spec = _run_full_detection(z, f, fp)

    assert mu is not None, "Should detect exponential rate"
    # After dividing out exp, residual ≈ constant → no trig
    assert trig_spec is None, "Pure exp should have no trig in residual"


def test_polynomial_rejected():
    """z^2 + 3*z + 1: log-derivative is smooth, no trig after dividing out exp."""
    z, f, fp = _make_uniform_grid(
        lambda z: z ** 2 + 3 * z + 1,
        lambda z: 2 * z + 3,
        z_min=0.5, z_max=5.0,
    )
    mu, trig_spec = _run_full_detection(z, f, fp)

    # Even if mu is nonzero, the residual should not be trig
    if mu is not None and abs(mu) >= 0.01:
        assert trig_spec is None, (
            f"Polynomial should not have trig in residual: mu={mu:.4f}"
        )


def test_insufficient_data():
    """Too few data points: _estimate_exp_rate should return None."""
    z = np.linspace(-1, 1, 10)
    f = np.exp(-0.5 * z) * np.cos(3 * z)
    fp = -0.5 * f + np.exp(-0.5 * z) * (-3 * np.sin(3 * z))

    mu = RuleUnivariateMulPeel._estimate_exp_rate(f, fp)
    assert mu is None, "Should return None for insufficient data"


# ──────────────────────────────────────────────────────────────
# Full detection helper (mirrors propose() 4-phase algorithm)
# ──────────────────────────────────────────────────────────────


def _run_full_detection_v2(z, f, fp):
    """Run the 4-phase detection: monomial → exp → trig.

    Returns (k, mu, omega) where each may be 0.0/None.
    """
    import math

    k_result = RuleUnivariateMulPeel._estimate_monomial_power(z, f, fp)
    k, mu_envelope = k_result if k_result[0] is not None else (0.0, None)

    # When envelope was used, trust its mu (median-L biased by trig)
    if mu_envelope is not None:
        mu = mu_envelope if abs(mu_envelope) >= 0.05 else 0.0
    else:
        mu = RuleUnivariateMulPeel._estimate_exp_rate(f, fp, z=z, k=k)
        if mu is None or abs(mu) < 0.01:
            mu = 0.0

    # Divide out detected factors
    z_t = torch.from_numpy(z).double()
    f_t = torch.from_numpy(f).double()
    residual = f_t.clone()
    if k != 0.0:
        z_safe = z_t.clamp_min(1e-8)
        residual = residual / z_safe.pow(k)
    if mu != 0.0:
        residual = residual * torch.exp(-mu * z_t)

    trig_spec = discover_trig_from_data(
        z_t, residual, strength_threshold=5.0, max_omega=1000.0,
    )
    omega = None
    if trig_spec is not None:
        z_span = float(z_t[-1] - z_t[0])
        n_cycles = trig_spec.omega * z_span / (2.0 * math.pi)
        if n_cycles >= 2.0:
            omega = float(trig_spec.omega)

    return k, mu, omega


# ──────────────────────────────────────────────────────────────
# Test _estimate_monomial_power
# ──────────────────────────────────────────────────────────────


def test_mono_power_z_squared():
    """z^2 * cos(3*z): k should be ≈ 2."""
    z, f, fp = _make_uniform_grid(
        lambda z: z ** 2 * np.cos(3.0 * z),
        lambda z: 2 * z * np.cos(3.0 * z) + z ** 2 * (-3.0 * np.sin(3.0 * z)),
        z_min=0.5, z_max=10.0,
    )
    k, _mu_env = RuleUnivariateMulPeel._estimate_monomial_power(z, f, fp)
    assert k is not None, "Should detect monomial power"
    assert abs(k - 2.0) < 0.01, f"k={k}, expected 2.0"


def test_mono_power_sqrt():
    """z^0.5 * exp(-z) (z > 0): k should be ≈ 0.5."""
    z, f, fp = _make_uniform_grid(
        lambda z: z ** 0.5 * np.exp(-z),
        lambda z: 0.5 * z ** (-0.5) * np.exp(-z) + z ** 0.5 * (-np.exp(-z)),
        z_min=0.5, z_max=10.0,
    )
    k, _mu_env = RuleUnivariateMulPeel._estimate_monomial_power(z, f, fp)
    assert k is not None, "Should detect monomial power"
    assert abs(k - 0.5) < 0.01, f"k={k}, expected 0.5"


def test_mono_power_negative():
    """z^(-1) * sin(z) (z > 0): k should be ≈ -1."""
    z, f, fp = _make_uniform_grid(
        lambda z: z ** (-1) * np.sin(z),
        lambda z: -z ** (-2) * np.sin(z) + z ** (-1) * np.cos(z),
        z_min=0.5, z_max=10.0,
    )
    k, _mu_env = RuleUnivariateMulPeel._estimate_monomial_power(z, f, fp)
    assert k is not None, "Should detect monomial power"
    assert abs(k - (-1.0)) < 0.01, f"k={k}, expected -1.0"


def test_mono_skipped_for_negative_z():
    """Data with z <= 0 coverage > 20%: monomial detection should be skipped."""
    z, f, fp = _make_uniform_grid(
        lambda z: np.abs(z) * np.cos(3.0 * z),
        lambda z: np.sign(z) * np.cos(3.0 * z) + np.abs(z) * (-3.0 * np.sin(3.0 * z)),
        z_min=-5.0, z_max=5.0,
    )
    k, _mu_env = RuleUnivariateMulPeel._estimate_monomial_power(z, f, fp)
    assert k is None, "Should skip monomial detection when >20% z<=0"


# ──────────────────────────────────────────────────────────────
# Test combined pipeline with monomial factors
# ──────────────────────────────────────────────────────────────


def test_mono_times_trig_detected():
    """z^2 * cos(3*z) should be detected as mono*trig."""
    z, f, fp = _make_uniform_grid(
        lambda z: z ** 2 * np.cos(3.0 * z),
        lambda z: 2 * z * np.cos(3.0 * z) + z ** 2 * (-3.0 * np.sin(3.0 * z)),
        z_min=0.5, z_max=10.0,
    )
    k, mu, omega = _run_full_detection_v2(z, f, fp)

    assert k != 0.0, "Should detect monomial"
    assert abs(k - 2.0) < 0.01, f"k={k}"
    assert mu == 0.0, "Should not detect exponential"
    assert omega is not None, "Should detect trig in residual"


def test_mono_times_exp_detected():
    """z^2 * exp(-0.5*z) (z > 0) should be detected as mono*exp."""
    z, f, fp = _make_uniform_grid(
        lambda z: z ** 2 * np.exp(-0.5 * z),
        lambda z: 2 * z * np.exp(-0.5 * z) + z ** 2 * (-0.5 * np.exp(-0.5 * z)),
        z_min=0.5, z_max=10.0,
    )
    k, mu, omega = _run_full_detection_v2(z, f, fp)

    assert k != 0.0, "Should detect monomial"
    assert abs(k - 2.0) < 0.01, f"k={k}"
    assert mu != 0.0, "Should detect exponential"
    assert abs(mu - (-0.5)) < 0.15, f"mu={mu}"
    assert omega is None, "Should not detect trig"


def test_mono_times_exp_times_trig_detected():
    """z^2 * exp(-0.5*z) * cos(3*z) (z > 0) should be detected as mono*exp*trig."""
    def func(z):
        return z ** 2 * np.exp(-0.5 * z) * np.cos(3.0 * z)

    def dfunc(z):
        return (
            2 * z * np.exp(-0.5 * z) * np.cos(3.0 * z)
            + z ** 2 * (-0.5) * np.exp(-0.5 * z) * np.cos(3.0 * z)
            + z ** 2 * np.exp(-0.5 * z) * (-3.0 * np.sin(3.0 * z))
        )

    z, f, fp = _make_uniform_grid(func, dfunc, z_min=0.5, z_max=10.0)
    k, mu, omega = _run_full_detection_v2(z, f, fp)

    assert k != 0.0, "Should detect monomial"
    assert abs(k - 2.0) < 0.01, f"k={k}"
    assert mu != 0.0, "Should detect exponential"
    assert abs(mu - (-0.5)) < 0.15, f"mu={mu}"
    assert omega is not None, "Should detect trig"


def test_pure_monomial_no_trig_no_exp():
    """z^2 alone: monomial detected but residual is constant — no trig or exp."""
    z, f, fp = _make_uniform_grid(
        lambda z: z ** 2,
        lambda z: 2.0 * z,
        z_min=0.5, z_max=10.0,
    )
    k, mu, omega = _run_full_detection_v2(z, f, fp)

    assert k != 0.0, "Should detect monomial"
    assert abs(k - 2.0) < 0.01, f"k={k}"
    # After dividing out z^2, residual ≈ constant → no trig, no exp
    assert omega is None, "Pure monomial should have no trig"


# ──────────────────────────────────────────────────────────────
# Test corrected exp estimation with monomial
# ──────────────────────────────────────────────────────────────


def test_exp_rate_corrected_for_monomial():
    """z^2 * exp(-0.5*z): without correction, mu estimate is biased.

    With k=2 correction, mu should be ≈ -0.5.
    """
    z, f, fp = _make_uniform_grid(
        lambda z: z ** 2 * np.exp(-0.5 * z),
        lambda z: 2 * z * np.exp(-0.5 * z) + z ** 2 * (-0.5 * np.exp(-0.5 * z)),
        z_min=0.5, z_max=10.0,
    )
    # Without correction
    mu_raw = RuleUnivariateMulPeel._estimate_exp_rate(f, fp)

    # With correction (k=2)
    mu_corr = RuleUnivariateMulPeel._estimate_exp_rate(f, fp, z=z, k=2.0)

    assert mu_corr is not None, "Corrected mu should not be None"
    assert abs(mu_corr - (-0.5)) < 0.1, f"corrected mu={mu_corr:.4f}, expected -0.5"

    # The corrected estimate should be closer to -0.5 than the raw one
    if mu_raw is not None:
        assert abs(mu_corr - (-0.5)) <= abs(mu_raw - (-0.5)) + 0.01, (
            f"Corrected ({mu_corr:.4f}) should be at least as good as raw ({mu_raw:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
