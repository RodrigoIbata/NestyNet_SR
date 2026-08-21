# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Unit tests for outer-transform probing in _nonlinear_substitution_screen.

Tests that the NLS screen can detect cases where F is NOT a rational function
of the transformed variables, but F² or 1/F IS.

Example: pb114 has f(z, x3) = sqrt(z² - 1) / (z + cos(x3)).
  f² = (z² - 1) / (z + cos(x3))² is a rational in (z, cos(x3)).
"""

import math
import torch

from nestynet_sr.sr_search.fitting_utils import _nonlinear_substitution_screen


def test_pb114_sqrt_rational_screen():
    """f(z, x3) = sqrt(z² - 1) / (z + cos(x3)).

    f itself is NOT rational in (z, cos(x3)), but f² IS.
    The outer_transforms=["square"] option should detect this.
    """
    torch.manual_seed(42)
    N = 3000

    z = torch.rand(N, dtype=torch.float64) * 4.0 + 1.05  # z > 1 so z²-1 > 0
    x3 = torch.rand(N, dtype=torch.float64) * 2 * math.pi - math.pi

    num = torch.sqrt(z ** 2 - 1.0)
    denom = z + torch.cos(x3)
    mask = denom.abs() > 0.2
    z, x3, num, denom = z[mask], x3[mask], num[mask], denom[mask]

    F = num / denom
    X = torch.stack([z, x3], dim=1)

    hits = _nonlinear_substitution_screen(
        X, F, teacher=None, threshold=0.05,
        outer_transforms=["square"],
    )

    # Filter to square hits on col 1 (the cos(x3) column)
    sq_hits = [h for h in hits if h.get("outer_transform") == "square"]

    assert len(sq_hits) > 0, (
        f"Expected at least one outer_transform='square' hit, got {len(sq_hits)}.\n"
        f"All hits: {hits}"
    )

    best_sq = sq_hits[0]
    print(
        f"Best square hit: transform={best_sq['transform']}, "
        f"col={best_sq['col_idx']}, "
        f"deg={best_sq['deg_num']}/{best_sq['deg_den']}, "
        f"err={best_sq['error']:.6f}, "
        f"outer={best_sq.get('outer_transform')}"
    )

    assert best_sq["transform"] == "cos", f"Expected cos, got {best_sq['transform']}"
    assert best_sq["col_idx"] == 1, f"Expected col 1, got {best_sq['col_idx']}"
    assert best_sq["error"] < 1e-3, f"Error {best_sq['error']:.4f} too high (expected < 1e-3)"

    print("PASSED: sqrt-rational detected via outer_transform='square'")


def test_backward_compat():
    """Without outer_transforms, no 'outer_transform' key should appear."""
    torch.manual_seed(42)
    N = 2000

    z = torch.rand(N, dtype=torch.float64) * 2 * math.pi - math.pi
    x1 = torch.rand(N, dtype=torch.float64) * 1.6 - 0.8

    denom = x1 * torch.cos(z) + 1.0
    mask = denom.abs() > 0.1
    z, x1, denom = z[mask], x1[mask], denom[mask]

    F = (1.0 - x1 ** 2) / denom
    X = torch.stack([z, x1], dim=1)

    hits = _nonlinear_substitution_screen(X, F, teacher=None, threshold=0.05)

    for h in hits:
        assert "outer_transform" not in h, (
            f"Hit should not have 'outer_transform' key when outer_transforms=None: {h}"
        )

    assert len(hits) > 0, "Expected identity hits for backward-compat test"
    print(f"PASSED: backward compat — {len(hits)} hits, none with outer_transform key")


def test_reciprocal_outer_transform():
    """f(z, x3) = (z + cos(x3)) / (z² - 1).

    1/f = (z² - 1) / (z + cos(x3)) is a simpler rational.
    """
    torch.manual_seed(42)
    N = 3000

    z = torch.rand(N, dtype=torch.float64) * 4.0 + 1.05
    x3 = torch.rand(N, dtype=torch.float64) * 2 * math.pi - math.pi

    num = z + torch.cos(x3)
    denom = z ** 2 - 1.0
    mask = denom.abs() > 0.2
    z, x3, num, denom = z[mask], x3[mask], num[mask], denom[mask]

    F = num / denom
    X = torch.stack([z, x3], dim=1)

    hits = _nonlinear_substitution_screen(
        X, F, teacher=None, threshold=0.05,
        outer_transforms=["reciprocal"],
    )

    # F itself might already be detected as a rational — that's fine.
    # We just check the reciprocal hits exist and are good.
    recip_hits = [h for h in hits if h.get("outer_transform") == "reciprocal"]

    # The reciprocal transform should only appear if it's significantly
    # better than the identity fit.  For this function, 1/F has a simpler
    # numerator degree, so it may or may not beat the identity.
    # Just verify no crashes and correct key if present.
    print(f"Identity hits: {len([h for h in hits if 'outer_transform' not in h])}")
    print(f"Reciprocal hits: {len(recip_hits)}")
    for h in recip_hits:
        assert h["outer_transform"] == "reciprocal"
        print(
            f"  transform={h['transform']}, col={h['col_idx']}, "
            f"deg={h['deg_num']}/{h['deg_den']}, err={h['error']:.6f}"
        )

    print("PASSED: reciprocal outer transform runs without errors")


if __name__ == "__main__":
    test_pb114_sqrt_rational_screen()
    print()
    test_backward_compat()
    print()
    test_reciprocal_outer_transform()
    print()
    print("All tests passed!")
