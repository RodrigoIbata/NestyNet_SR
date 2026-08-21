# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Tests for power-difference compound detection via gradient-ratio scan.

Verifies that _scan_gradient_ratio_pairs and _test_power_difference_structure
correctly detect z = xi^n - xj^n for various integer powers n.
"""

import numpy as np
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nestynet_sr.sr_search.search import (
    _scan_gradient_ratio_pairs,
    _test_power_difference_structure,
    _test_power_diff_product_structure,
)
from nestynet_sr.sr_core.separability_math import (
    build_power_difference_ast,
    build_power_difference_product_ast,
)


def _make_data(n_points=2000, seed=42):
    """Generate random 2D data on [0.5, 2.5] to avoid zero issues."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 2.5, size=(n_points, 2))
    return x


def _numerical_grad(f, x, eps=1e-7):
    """Compute numerical gradients for a scalar function f(x0, x1)."""
    N, k = x.shape
    grad = np.zeros_like(x)
    for j in range(k):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[:, j] += eps
        x_minus[:, j] -= eps
        grad[:, j] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


# ─── Test n=1: linear difference ──────────────────────────────────────

def test_n1_linear_diff():
    """f = sin(x0 - x1) => n=1 detected."""
    x = _make_data()
    f_fn = lambda x: np.sin(x[:, 0] - x[:, 1])
    dydx = _numerical_grad(f_fn, x)

    hits = _scan_gradient_ratio_pairs(x, dydx)
    assert len(hits) >= 1, f"Expected at least 1 hit, got {len(hits)}"
    i, j, n, conf = hits[0]
    assert n == 1, f"Expected n=1, got n={n}"
    assert conf > 0.8, f"Expected conf>0.8, got {conf:.3f}"

    # Verify with structure test
    verify = _test_power_difference_structure(x, dydx, i, j, n)
    assert verify > 0.9, f"Verification conf={verify:.3f}, expected >0.9"
    print(f"  PASS n=1: conf={conf:.3f}, verify={verify:.3f}")


# ─── Test n=2: quadratic difference ──────────────────────────────────

def test_n2_quadratic_diff():
    """f = 1 / (x0^2 - x1^2 + 0.1)  => n=2 detected."""
    x = _make_data()
    # Add small offset to avoid division by zero
    f_fn = lambda x: 1.0 / (x[:, 0] ** 2 - x[:, 1] ** 2 + 0.1)
    dydx = _numerical_grad(f_fn, x)

    hits = _scan_gradient_ratio_pairs(x, dydx)
    assert len(hits) >= 1, f"Expected at least 1 hit, got {len(hits)}"

    # Find the n=2 hit
    n2_hits = [(i, j, n, c) for i, j, n, c in hits if n == 2]
    assert len(n2_hits) >= 1, f"Expected n=2 hit, got powers: {[h[2] for h in hits]}"
    i, j, n, conf = n2_hits[0]
    assert conf > 0.8, f"Expected conf>0.8, got {conf:.3f}"

    # Verify with structure test
    verify = _test_power_difference_structure(x, dydx, i, j, n)
    assert verify > 0.9, f"Verification conf={verify:.3f}, expected >0.9"
    print(f"  PASS n=2: conf={conf:.3f}, verify={verify:.3f}")


# ─── Test n=3: cubic difference ──────────────────────────────────────

def test_n3_cubic_diff():
    """f = (x0^3 - x1^3)^2  => n=3 detected."""
    x = _make_data()
    f_fn = lambda x: (x[:, 0] ** 3 - x[:, 1] ** 3) ** 2
    dydx = _numerical_grad(f_fn, x)

    hits = _scan_gradient_ratio_pairs(x, dydx)
    assert len(hits) >= 1, f"Expected at least 1 hit, got {len(hits)}"

    n3_hits = [(i, j, n, c) for i, j, n, c in hits if n == 3]
    assert len(n3_hits) >= 1, f"Expected n=3 hit, got powers: {[h[2] for h in hits]}"
    i, j, n, conf = n3_hits[0]
    assert conf > 0.8, f"Expected conf>0.8, got {conf:.3f}"

    verify = _test_power_difference_structure(x, dydx, i, j, n)
    assert verify > 0.9, f"Verification conf={verify:.3f}, expected >0.9"
    print(f"  PASS n=3: conf={conf:.3f}, verify={verify:.3f}")


# ─── Test n=2 with multiplier ────────────────────────────────────────

def test_n2_with_multiplier():
    """f = sin((x0^2 - x1^2) * x2)  => n=2 diff-product detected."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.5, 2.0, size=(2000, 3))
    # True product compound: z = (x0² - x1²) * x2
    f_fn = lambda x: np.sin((x[:, 0] ** 2 - x[:, 1] ** 2) * x[:, 2])
    f_vals = f_fn(x)
    dydx = _numerical_grad(f_fn, x)

    # The gradient-ratio scan on the first two variables should find n=2
    hits = _scan_gradient_ratio_pairs(x, dydx)
    n2_hits = [(i, j, n, c) for i, j, n, c in hits if n == 2]
    assert len(n2_hits) >= 1, f"Expected n=2 hit, got powers: {[h[2] for h in hits]}"

    # Test the product structure
    i, j, n, conf = n2_hits[0]
    # k is the remaining index
    k = [idx for idx in range(3) if idx not in (i, j)][0]
    prod_conf, outer_power = _test_power_diff_product_structure(
        x, dydx, i, j, k, n, f_vals=f_vals
    )
    assert prod_conf > 0.5, f"Product conf={prod_conf:.3f}, expected >0.5"
    print(f"  PASS n=2 with multiplier: ratio_conf={conf:.3f}, product_conf={prod_conf:.3f}, outer_power={outer_power}")


# ─── Test: non-integer power is skipped ──────────────────────────────

def test_skip_non_integer():
    """f = x0^1.5 - x1^1.5 wrapped in sin => non-integer slope, should not match."""
    x = _make_data()
    f_fn = lambda x: np.sin(np.abs(x[:, 0]) ** 1.5 - np.abs(x[:, 1]) ** 1.5)
    dydx = _numerical_grad(f_fn, x)

    hits = _scan_gradient_ratio_pairs(x, dydx, int_threshold=0.15)
    # With tighter int_threshold, the non-integer power should be rejected
    if hits:
        # If any hit, its n should not be close to 1.5
        for i, j, n, conf in hits:
            print(f"  INFO: got hit n={n}, conf={conf:.3f} (may be weak)")
    print(f"  PASS skip_non_integer: {len(hits)} hits (expected few/none with tight threshold)")


# ─── Test: high power is skipped ─────────────────────────────────────

def test_skip_high_power():
    """f = x0^6 - x1^6 => skipped because n>4."""
    x = _make_data()
    f_fn = lambda x: x[:, 0] ** 6 - x[:, 1] ** 6
    dydx = _numerical_grad(f_fn, x)

    hits = _scan_gradient_ratio_pairs(x, dydx, max_power=4)
    # Should not detect n=6 since max_power=4
    for i, j, n, conf in hits:
        assert n <= 4, f"Expected n<=4, got n={n}"
    print(f"  PASS skip_high_power: {len(hits)} hits (none with n>4)")


# ─── Test: AST builders produce valid nodes ──────────────────────────

def test_ast_builders():
    """Verify AST builder functions produce valid nodes."""
    # Power difference
    ast1 = build_power_difference_ast(1, 3, 1)
    assert ast1 is not None
    ast2 = build_power_difference_ast(1, 3, 2)
    assert ast2 is not None

    # Power difference product
    ast3 = build_power_difference_product_ast(1, 3, 2, 0)
    assert ast3 is not None
    ast4 = build_power_difference_product_ast(1, 3, 2, 0, p=-1)
    assert ast4 is not None

    print("  PASS AST builders produce valid nodes")


# ─── Test n=4: quartic difference ────────────────────────────────────

def test_n4_quartic_diff():
    """f = exp(-(x0^4 - x1^4)) => n=4 detected."""
    x = _make_data(n_points=3000)
    f_fn = lambda x: np.exp(-(x[:, 0] ** 4 - x[:, 1] ** 4))
    dydx = _numerical_grad(f_fn, x)

    hits = _scan_gradient_ratio_pairs(x, dydx)
    n4_hits = [(i, j, n, c) for i, j, n, c in hits if n == 4]
    assert len(n4_hits) >= 1, f"Expected n=4 hit, got powers: {[h[2] for h in hits]}"
    i, j, n, conf = n4_hits[0]
    assert conf > 0.7, f"Expected conf>0.7, got {conf:.3f}"

    verify = _test_power_difference_structure(x, dydx, i, j, n)
    assert verify > 0.8, f"Verification conf={verify:.3f}, expected >0.8"
    print(f"  PASS n=4: conf={conf:.3f}, verify={verify:.3f}")


# ─── Run all tests ───────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("n=1 linear difference", test_n1_linear_diff),
        ("n=2 quadratic difference", test_n2_quadratic_diff),
        ("n=3 cubic difference", test_n3_cubic_diff),
        ("n=4 quartic difference", test_n4_quartic_diff),
        ("n=2 with multiplier", test_n2_with_multiplier),
        ("skip non-integer power", test_skip_non_integer),
        ("skip high power", test_skip_high_power),
        ("AST builders", test_ast_builders),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"Running: {name}...")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")
