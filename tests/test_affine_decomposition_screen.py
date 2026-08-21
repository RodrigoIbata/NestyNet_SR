# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for the affine decomposition screen.

Tests that _affine_decomposition_screen correctly identifies cases where
g(f(z, w)) = a(z) + b(z) * h(w) for various output and variable transforms.
"""

import math

import torch

from nestynet_sr.sr_search.fitting_utils import _affine_decomposition_screen


def _make_grid(n_z=50, n_w=80, z_range=(1.5, 5.0), w_range=(0.0, 2 * math.pi)):
    """Create a 2D grid of (z, w) values."""
    z_vals = torch.linspace(z_range[0], z_range[1], n_z, dtype=torch.float64)
    w_vals = torch.linspace(w_range[0], w_range[1], n_w, dtype=torch.float64)
    zz, ww = torch.meshgrid(z_vals, w_vals, indexing="ij")
    X = torch.stack([zz.reshape(-1), ww.reshape(-1)], dim=1)
    return X


def test_pb114_leaf():
    """Synthetic pb114 leaf: f(z, x3) = sqrt(z²-1) / (z + cos(x3)).

    Reciprocal is affine in cos(x3): 1/f = z/sqrt(z²-1) + cos(x3)/sqrt(z²-1).
    Expect hit with g=reciprocal, h=cos, R² > 0.999.
    """
    X = _make_grid(n_z=50, n_w=80, z_range=(1.5, 5.0), w_range=(0.0, 2 * math.pi))
    z = X[:, 0]
    w = X[:, 1]
    F = torch.sqrt(z ** 2 - 1.0) / (z + torch.cos(w))

    hits = _affine_decomposition_screen(X, F)

    assert len(hits) > 0, "Expected at least one hit for pb114-like function"
    best = hits[0]
    print(f"pb114 best hit: g={best['g_name']}, h={best['h_name']}, R²={best['median_r2']:.6f}")
    assert best["g_name"] == "reciprocal", f"Expected reciprocal, got {best['g_name']}"
    assert best["h_name"] == "cos", f"Expected cos, got {best['h_name']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_pb114_leaf")


def test_simple_affine():
    """Simple affine: f(z, w) = z² + 3*z*w.

    Already affine in w with identity transforms: f = z² + 3z * w.
    Expect g=identity, h=identity.
    """
    X = _make_grid(n_z=50, n_w=80, z_range=(0.5, 3.0), w_range=(-2.0, 2.0))
    z = X[:, 0]
    w = X[:, 1]
    F = z ** 2 + 3.0 * z * w

    hits = _affine_decomposition_screen(X, F)

    assert len(hits) > 0, "Expected at least one hit for simple affine"
    best = hits[0]
    print(f"simple affine best hit: g={best['g_name']}, h={best['h_name']}, R²={best['median_r2']:.6f}")
    assert best["g_name"] == "identity", f"Expected identity, got {best['g_name']}"
    assert best["h_name"] == "identity", f"Expected identity, got {best['h_name']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_simple_affine")


def test_no_affine():
    """Non-affine: f(z, w) = sin(z * w).

    Not affine in any simple h(w), so expect no hits.
    """
    X = _make_grid(n_z=50, n_w=80, z_range=(0.5, 3.0), w_range=(0.0, 2 * math.pi))
    z = X[:, 0]
    w = X[:, 1]
    F = torch.sin(z * w)

    hits = _affine_decomposition_screen(X, F)

    # Should have no hits (or only very low R²)
    for h in hits:
        assert h["median_r2"] <= 0.999, f"Unexpected high R² hit: {h}"
    print(f"no_affine: {len(hits)} hits (all below threshold)")
    print("PASS: test_no_affine")


def test_reciprocal_guard():
    """f(z, w) that passes through zero — reciprocal should be skipped safely."""
    X = _make_grid(n_z=50, n_w=80, z_range=(-2.0, 2.0), w_range=(-2.0, 2.0))
    z = X[:, 0]
    w = X[:, 1]
    # F passes through zero at z=0
    F = z * (1.0 + w)

    hits = _affine_decomposition_screen(X, F)

    # Should not crash. Reciprocal should be skipped (F has zeros).
    recip_hits = [h for h in hits if h["g_name"] == "reciprocal"]
    assert len(recip_hits) == 0, "Reciprocal should be skipped when F has zeros"
    print(f"reciprocal_guard: reciprocal correctly skipped, {len(hits)} total hits")
    print("PASS: test_reciprocal_guard")


def test_bad_omega_hint():
    """Trig hint with noisy omega=1.15 should snap to 1.0 and still find hit.

    Simulates the pb114 scenario where FFT detects omega≈1.15 but the true
    frequency is 1.0.  The snap logic should correct this.
    """
    from nestynet_sr.sr_search.features import TrigAxisSpec

    X = _make_grid(n_z=50, n_w=80, z_range=(1.5, 5.0), w_range=(0.0, 2 * math.pi))
    z = X[:, 0]
    w = X[:, 1]
    F = torch.sqrt(z ** 2 - 1.0) / (z + torch.cos(w))  # true omega=1.0

    # Pass a bad omega hint (1.15) — should snap to 1.0
    bad_hint = TrigAxisSpec(axis=1, omega=1.15, strength=5.0, n_points=80,
                            tmin=0.0, tmax=2 * math.pi)
    hits = _affine_decomposition_screen(X, F, trig_hints={1: bad_hint})

    assert len(hits) > 0, "Expected at least one hit with snapped omega"
    best = hits[0]
    print(f"bad_omega_hint best: g={best['g_name']}, h={best['h_name']}, "
          f"omega={best['omega']}, R²={best['median_r2']:.6f}")
    assert best["g_name"] == "reciprocal", f"Expected reciprocal, got {best['g_name']}"
    assert best["h_name"] == "cos", f"Expected cos, got {best['h_name']}"
    assert best["omega"] == 1.0, f"Expected omega=1.0 (snapped), got {best['omega']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_bad_omega_hint")


def test_one_minus_cos_hint_is_preserved():
    """A confirmed one-minus-cos trig hint should be usable as affine coordinate."""
    from nestynet_sr.sr_search.features import TrigAxisSpec

    X = _make_grid(n_z=50, n_w=80, z_range=(0.5, 3.0), w_range=(0.0, 2 * math.pi))
    z = X[:, 0]
    w = X[:, 1]
    F = 1.25 * z - 0.75 * (1.0 - torch.cos(2.0 * w))

    hint = TrigAxisSpec(
        axis=1,
        omega=2.0,
        strength=100.0,
        n_points=80,
        tmin=0.0,
        tmax=2 * math.pi,
        basis_fn="one_minus_cos",
    )
    hits = _affine_decomposition_screen(X, F, trig_hints={1: hint})

    assert len(hits) > 0, "Expected one-minus-cos affine hit"
    best = hits[0]
    print(
        f"one_minus_cos best: g={best['g_name']}, h={best['h_name']}, "
        f"omega={best['omega']}, R²={best['median_r2']:.6f}"
    )
    assert best["g_name"] == "identity", f"Expected identity, got {best['g_name']}"
    assert best["h_name"] == "one_minus_cos", f"Expected one_minus_cos, got {best['h_name']}"
    assert best["omega"] == 2.0, f"Expected omega=2.0, got {best['omega']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_one_minus_cos_hint_is_preserved")


def test_swapped_columns():
    """Affine structure in column 0: f(z, w) = w² + 3*w*z.

    Here the affine variable is z (column 0), not w (column 1).
    The swapped-ordering logic should find g=identity, h=identity, col_w=0.
    """
    X = _make_grid(n_z=50, n_w=80, z_range=(-2.0, 2.0), w_range=(0.5, 3.0))
    z = X[:, 0]  # this is the affine variable
    w = X[:, 1]  # this is the binning variable
    F = w ** 2 + 3.0 * w * z

    hits = _affine_decomposition_screen(X, F)

    assert len(hits) > 0, "Expected at least one hit for swapped-column affine"
    # Find the best hit that uses col_w=0 (swapped ordering)
    swapped = [h for h in hits if h["col_w"] == 0]
    assert len(swapped) > 0, f"Expected a col_w=0 hit, got col_w values: {[h['col_w'] for h in hits]}"
    best = swapped[0]
    print(f"swapped best hit: g={best['g_name']}, h={best['h_name']}, "
          f"col_w={best['col_w']}, R²={best['median_r2']:.6f}")
    assert best["g_name"] == "identity", f"Expected identity, got {best['g_name']}"
    assert best["h_name"] == "identity", f"Expected identity, got {best['h_name']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_swapped_columns")


def test_swapped_trig():
    """pb114 with columns swapped: f(z, w) = sqrt(w²-1) / (w + cos(z)).

    Here z is column 0 (trig variable) and w is column 1 (binning variable).
    Reciprocal 1/f = w/sqrt(w²-1) + cos(z)/sqrt(w²-1) is affine in cos(z).
    The original ordering (col_w=1) cannot find this because h(w) doesn't help.
    The swapped ordering (col_w=0) should detect g=reciprocal, h=cos.
    """
    X = _make_grid(n_z=80, n_w=50, z_range=(0.0, 2 * math.pi), w_range=(1.5, 5.0))
    z = X[:, 0]
    w = X[:, 1]
    F = torch.sqrt(w ** 2 - 1.0) / (w + torch.cos(z))

    hits = _affine_decomposition_screen(X, F)

    swapped = [h for h in hits if h["col_w"] == 0 and h["h_name"] == "cos"]
    assert len(swapped) > 0, (
        f"Expected a col_w=0, h=cos hit; got: "
        f"{[(h['col_w'], h['h_name'], h['median_r2']) for h in hits]}"
    )
    best = swapped[0]
    print(f"swapped_trig best: g={best['g_name']}, h={best['h_name']}, "
          f"col_w={best['col_w']}, omega={best['omega']}, R²={best['median_r2']:.6f}")
    assert best["g_name"] == "reciprocal", f"Expected reciprocal, got {best['g_name']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_swapped_trig")


def test_small_probe_budget_still_detects_affine():
    """Compact Stage-B probe clouds should still permit affine detection."""
    X = _make_grid(n_z=8, n_w=8, z_range=(0.5, 3.0), w_range=(-2.0, 2.0))
    z = X[:, 0]
    w = X[:, 1]
    F = z ** 2 + 3.0 * z * w

    hits = _affine_decomposition_screen(X, F)

    assert len(hits) > 0, "Expected at least one hit on a 64-point probe cloud"
    best = hits[0]
    print(
        f"small probe best hit: g={best['g_name']}, h={best['h_name']}, "
        f"R²={best['median_r2']:.6f}"
    )
    assert best["g_name"] == "identity", f"Expected identity, got {best['g_name']}"
    assert best["h_name"] == "identity", f"Expected identity, got {best['h_name']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_small_probe_budget_still_detects_affine")


def test_small_random_cloud_detects_global_affine():
    """Random probe clouds should still expose exactly affine leaves."""
    gen = torch.Generator().manual_seed(0)
    z = 0.5 + 2.5 * torch.rand(64, generator=gen, dtype=torch.float64)
    w = -2.0 + 4.0 * torch.rand(64, generator=gen, dtype=torch.float64)
    X = torch.stack([z, w], dim=1)
    F = 1.25 * z - 0.75 * w

    hits = _affine_decomposition_screen(X, F)

    assert len(hits) > 0, "Expected at least one hit on a random affine cloud"
    best = hits[0]
    print(
        f"small random best hit: g={best['g_name']}, h={best['h_name']}, "
        f"R²={best['median_r2']:.6f}"
    )
    assert best["g_name"] == "identity", f"Expected identity, got {best['g_name']}"
    assert best["h_name"] == "identity", f"Expected identity, got {best['h_name']}"
    assert best["median_r2"] > 0.999, f"R² too low: {best['median_r2']}"
    print("PASS: test_small_random_cloud_detects_global_affine")


if __name__ == "__main__":
    test_pb114_leaf()
    test_simple_affine()
    test_no_affine()
    test_reciprocal_guard()
    test_bad_omega_hint()
    test_one_minus_cos_hint_is_preserved()
    test_swapped_columns()
    test_swapped_trig()
    test_small_probe_budget_still_detects_affine()
    test_small_random_cloud_detects_global_affine()
    print("\nAll tests passed!")
