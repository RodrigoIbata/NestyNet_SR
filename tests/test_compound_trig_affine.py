# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""
Test _parse_pure_difference_expr and compound atom handling in
_build_trig_diff_affine_envelope_candidate.
"""

import sys
sys.path.insert(0, ".")

from nestynet_sr.sr_core.bridges import (
    AddNode,
    ConstNode,
    MulNode,
)
from nestynet_sr.sr_core.separability_math import build_linear_ast
from nestynet_sr.sr_search.candidate_builders import _parse_pure_difference_expr


def test_parse_pure_difference_standard_form():
    """Test parsing z = x4 - x5 (standard form from build_linear_ast)."""
    # Build z = x4 - x5 using the same function that Stage A uses
    z_ast = build_linear_ast((4, 5), (1, -1))

    result = _parse_pure_difference_expr(z_ast)
    assert result is not None, "Should parse standard pure difference"
    i, j = result
    assert (i, j) == (4, 5), f"Expected (4, 5), got {(i, j)}"
    print(f"[PASS] Standard form z = x4 - x5 parsed correctly: {result}")


def test_parse_pure_difference_reversed():
    """Test parsing z = x5 - x4 (reversed order)."""
    z_ast = build_linear_ast((5, 4), (1, -1))

    result = _parse_pure_difference_expr(z_ast)
    assert result is not None, "Should parse reversed pure difference"
    i, j = result
    assert (i, j) == (5, 4), f"Expected (5, 4), got {(i, j)}"
    print(f"[PASS] Reversed form z = x5 - x4 parsed correctly: {result}")


def test_parse_pure_difference_different_indices():
    """Test parsing z = x0 - x1."""
    z_ast = build_linear_ast((0, 1), (1, -1))

    result = _parse_pure_difference_expr(z_ast)
    assert result is not None, "Should parse different indices"
    i, j = result
    assert (i, j) == (0, 1), f"Expected (0, 1), got {(i, j)}"
    print(f"[PASS] Different indices z = x0 - x1 parsed correctly: {result}")


def test_parse_not_pure_difference_sum():
    """Test that z = x4 + x5 (sum, not difference) returns None."""
    z_ast = build_linear_ast((4, 5), (1, 1))

    result = _parse_pure_difference_expr(z_ast)
    assert result is None, f"Sum should not be parsed as difference, got {result}"
    print("[PASS] Sum z = x4 + x5 correctly rejected")


def test_parse_not_pure_difference_scaled():
    """Test that z = 2*x4 - x5 (scaled, not pure difference) returns None."""
    z_ast = build_linear_ast((4, 5), (2, -1))

    result = _parse_pure_difference_expr(z_ast)
    assert result is None, f"Scaled difference should not be parsed, got {result}"
    print("[PASS] Scaled z = 2*x4 - x5 correctly rejected")


def test_parse_not_pure_difference_product():
    """Test that a product (not Add) returns None."""
    # Var(4) * Var(5)
    from nestynet_sr.sr_core.bridges import Var
    z_ast = MulNode(Var(4), Var(5))

    result = _parse_pure_difference_expr(z_ast)
    assert result is None, f"Product should not be parsed, got {result}"
    print("[PASS] Product correctly rejected")


def test_parse_not_pure_difference_single_var():
    """Test that a single variable returns None."""
    from nestynet_sr.sr_core.bridges import Var
    z_ast = Var(4)

    result = _parse_pure_difference_expr(z_ast)
    assert result is None, f"Single var should not be parsed, got {result}"
    print("[PASS] Single variable correctly rejected")


def test_parse_manual_construction():
    """Test manually constructed pure difference AST."""
    from nestynet_sr.sr_core.bridges import Var

    # Manually build: Add(Var(4), Mul(ConstNode(-1.0), Var(5)))
    neg_one = ConstNode(-1.0)
    neg_x5 = MulNode(neg_one, Var(5))
    z_ast = AddNode(Var(4), neg_x5)

    result = _parse_pure_difference_expr(z_ast)
    assert result is not None, "Should parse manually constructed difference"
    i, j = result
    assert (i, j) == (4, 5), f"Expected (4, 5), got {(i, j)}"
    print(f"[PASS] Manual construction parsed correctly: {result}")


def test_parse_manual_reversed_order():
    """Test manually constructed with reversed order in Add."""
    from nestynet_sr.sr_core.bridges import Var

    # Manually build: Add(Mul(ConstNode(-1.0), Var(5)), Var(4))
    neg_one = ConstNode(-1.0)
    neg_x5 = MulNode(neg_one, Var(5))
    z_ast = AddNode(neg_x5, Var(4))

    result = _parse_pure_difference_expr(z_ast)
    assert result is not None, "Should parse reversed order"
    i, j = result
    assert (i, j) == (4, 5), f"Expected (4, 5), got {(i, j)}"
    print(f"[PASS] Reversed order in Add parsed correctly: {result}")


if __name__ == "__main__":
    test_parse_pure_difference_standard_form()
    test_parse_pure_difference_reversed()
    test_parse_pure_difference_different_indices()
    test_parse_not_pure_difference_sum()
    test_parse_not_pure_difference_scaled()
    test_parse_not_pure_difference_product()
    test_parse_not_pure_difference_single_var()
    test_parse_manual_construction()
    test_parse_manual_reversed_order()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
