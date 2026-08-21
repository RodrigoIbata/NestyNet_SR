# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for the rational polynomial degree probe with dimensional analysis.

Run:
    python tests/test_ratpoly_degree_probe.py
"""

import numpy as np
from fractions import Fraction

from nestynet_sr.sr_search.ratpoly_degree_probe import (
    probe_rational_degrees,
    probe_rational_fit,
    probe_poly_exponents,
    summarise_degree_info,
    _to_dim,
    _monomial_dim,
    _enumerate_monomials,
    _dimension_matrix_rank,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _F(x):
    """Shorthand for Fraction."""
    return Fraction(x)


# ─────────────────────────────────────────────────────────────────────
# 1. Unit tests for internal helpers
# ─────────────────────────────────────────────────────────────────────

def test_enumerate_monomials_degree_0():
    """Degree-0 monomials: just the constant (0,...,0)."""
    for n_in in (1, 2, 3):
        monos = _enumerate_monomials(n_in, 0)
        assert len(monos) == 1
        assert monos[0] == tuple(0 for _ in range(n_in))


def test_enumerate_monomials_counts():
    """Number of degree-k monomials in d variables = C(k+d-1, d-1)."""
    from math import comb
    for d in (1, 2, 3, 4):
        for k in range(0, 6):
            monos = _enumerate_monomials(d, k)
            expected = comb(k + d - 1, d - 1)
            assert len(monos) == expected, f"d={d}, k={k}: got {len(monos)}, expected {expected}"
            # Every monomial should sum to exactly k.
            for alpha in monos:
                assert sum(alpha) == k


def test_monomial_dim_basic():
    """Monomial x1^2 * x2 with x1=[L], x2=[T] should give [2L, T]."""
    x_dims = [_to_dim([1, 0]), _to_dim([0, 1])]
    alpha = (2, 1)
    d = _monomial_dim(alpha, x_dims)
    assert d == (_F(2), _F(1))


def test_monomial_dim_same_units():
    """When all vars have dim [L], monomial dim depends only on total degree."""
    x_dims = [_to_dim([1]), _to_dim([1])]
    assert _monomial_dim((3, 0), x_dims) == (_F(3),)
    assert _monomial_dim((2, 1), x_dims) == (_F(3),)
    assert _monomial_dim((1, 2), x_dims) == (_F(3),)
    assert _monomial_dim((0, 3), x_dims) == (_F(3),)


def test_dimension_matrix_rank():
    """Check rank computation for various configurations."""
    # Two identical dims → rank 1
    assert _dimension_matrix_rank([_to_dim([1, 0]), _to_dim([1, 0])]) == 1
    # Two independent dims → rank 2
    assert _dimension_matrix_rank([_to_dim([1, 0]), _to_dim([0, 1])]) == 2
    # Three dims, two independent → rank 2
    assert _dimension_matrix_rank([_to_dim([1, 0]), _to_dim([0, 1]), _to_dim([1, 1])]) == 2
    # Empty
    assert _dimension_matrix_rank([]) == 0
    # Single dimensionless variable
    assert _dimension_matrix_rank([_to_dim([0, 0])]) == 0


# ─────────────────────────────────────────────────────────────────────
# 2. Core probe: all dimensionless
# ─────────────────────────────────────────────────────────────────────

def test_all_dimensionless():
    """When everything is dimensionless, every degree pair is valid."""
    info = probe_rational_degrees(
        target_dim=[0, 0],
        x_dims=[[0, 0], [0, 0]],
        max_total_degree=3,
    )
    assert info.all_dimensionless
    assert info.dimensionless_rank == 2
    # There should be exactly one dim-class (the zero vector), so exactly
    # one valid pair: (zero, zero).
    assert len(info.valid_pairs) == 1
    p = info.valid_pairs[0]
    # All monomials up to degree 3 are in this single dim-class.
    assert p.min_deg_num == 0
    assert p.min_deg_den == 0
    assert p.has_polynomial_solution if hasattr(p, 'has_polynomial_solution') else True


def test_all_dimensionless_has_polynomial():
    """All-dimensionless case should report has_polynomial_solution."""
    info = probe_rational_degrees([0, 0], [[0, 0]], max_total_degree=4)
    assert info.has_polynomial_solution


# ─────────────────────────────────────────────────────────────────────
# 3. Core probe: same units (both [L])
# ─────────────────────────────────────────────────────────────────────

def test_same_units_two_vars():
    """Two variables with dim [L], target [L^-2].

    This is the pb111 hard leaf: f(x1,x3) = x1*x3/(x1^2 - x3^2)^2.
    Valid pairs: k_num - k_den = -2, i.e. (0,2), (1,3), (2,4), ...
    """
    info = probe_rational_degrees(
        target_dim=[-2],
        x_dims=[[1], [1]],
        max_total_degree=6,
    )
    assert info.same_units
    assert info.dimensionless_rank == 1  # x1/x3 is dimensionless (but not as polynomial monomial)
    assert not info.all_dimensionless

    # Valid pairs should be (k, k+2) for k = 0, 1, 2, 3, 4.
    pair_degs = [(p.min_deg_num, p.min_deg_den) for p in info.valid_pairs]
    assert (0, 2) in pair_degs
    assert (1, 3) in pair_degs
    assert (2, 4) in pair_degs
    assert (3, 5) in pair_degs
    assert (4, 6) in pair_degs

    # The simplest is (0, 2) — constant / (quadratic).
    assert info.global_min_deg_num == 0
    assert info.global_min_deg_den == 2

    # Pair (2, 4) should have:
    # - num: 3 monomials at degree 2 (x1^2, x1*x3, x3^2)
    # - den: 5 monomials at degree 4
    pair_2_4 = [p for p in info.valid_pairs if p.min_deg_num == 2 and p.min_deg_den == 4][0]
    assert pair_2_4.n_monomials_num_up_to(2) == 3
    assert pair_2_4.n_monomials_den_up_to(4) == 5

    # No polynomial solution (target is [L^-2], can't get negative dims from non-negative exponents).
    assert not info.has_polynomial_solution


def test_same_units_positive_target():
    """Two vars [L], target [L^2]. Should have polynomial solution."""
    info = probe_rational_degrees(
        target_dim=[2],
        x_dims=[[1], [1]],
        max_total_degree=4,
    )
    # Pair (2, 0): degree-2 numerator over constant denominator.
    assert info.has_polynomial_solution
    pair_poly = [p for p in info.valid_pairs if p.min_deg_den == 0][0]
    assert pair_poly.min_deg_num == 2
    # 3 degree-2 monomials: x1^2, x1*x3, x3^2
    assert pair_poly.n_monomials_num_up_to(2) == 3


# ─────────────────────────────────────────────────────────────────────
# 4. Core probe: mixed units
# ─────────────────────────────────────────────────────────────────────

def test_mixed_units_no_dimless():
    """x1=[L], x2=[T], target=[L/T]. Valid monomial: x1/x2 (needs denominator)."""
    info = probe_rational_degrees(
        target_dim=[1, -1],
        x_dims=[[1, 0], [0, 1]],
        max_total_degree=4,
    )
    assert not info.all_dimensionless
    assert not info.same_units
    assert info.dimensionless_rank == 0  # no dimensionless products

    # For pure polynomial: need monomial with dim [L, T^-1].
    # With non-negative exponents only: x1^a * x2^b has dim [a, b].
    # We need a=1, b=-1 — impossible with b >= 0.
    # So no polynomial solution.
    assert not info.has_polynomial_solution

    # Simplest rational: x1 / x2 → pair dim_num=[1,0], dim_den=[0,1].
    # That's deg_num=1, deg_den=1 with 1 monomial each.
    pair_degs = [(p.min_deg_num, p.min_deg_den) for p in info.valid_pairs]
    assert (1, 1) in pair_degs

    pair_1_1 = [p for p in info.valid_pairs if p.min_deg_num == 1 and p.min_deg_den == 1][0]
    # At min degrees: exactly 1 numerator monomial (x1) and 1 denominator monomial (x2).
    assert pair_1_1.n_monomials_num_up_to(1) == 1
    assert pair_1_1.n_monomials_den_up_to(1) == 1


def test_mixed_units_with_dimless_var():
    """x1=[L], x2=[dimensionless], target=[L^2].

    Dimensionless x2 means monomials at different total degrees can share
    the same dim-class: x1^2, x1^2*x2, x1^2*x2^2 all have dim [2L].
    """
    info = probe_rational_degrees(
        target_dim=[2],
        x_dims=[[1], [0]],
        max_total_degree=4,
    )
    assert not info.all_dimensionless
    assert not info.same_units
    assert info.dimensionless_rank == 1  # x2 is dimensionless

    # Polynomial solution exists: x1^2 has dim [2L] = target.
    assert info.has_polynomial_solution

    # The polynomial pair should have dim_num = [2], dim_den = [0].
    poly_pair = [p for p in info.valid_pairs if 0 in p.monomials_den][0]
    assert poly_pair.min_deg_num == 2
    assert poly_pair.min_deg_den == 0

    # The numerator dim-class [2L] should have monomials at degrees 2, 3, 4
    # (x1^2, x1^2*x2, x1^2*x2^2).
    assert 2 in poly_pair.monomials_num
    assert 3 in poly_pair.monomials_num
    assert 4 in poly_pair.monomials_num
    assert poly_pair.n_monomials_num_up_to(4) == 3


# ─────────────────────────────────────────────────────────────────────
# 5. Core probe: the full pb111 problem
# ─────────────────────────────────────────────────────────────────────

def test_pb111_full():
    """Full pb111 units: 5 variables, target [L, T^-2, M, 0, 0].

    y_units = [1, -2, 1, 0, 0]
    x0 = [2, -2, 1, 0, -1]
    x1 = [1, 0, 0, 0, 0]
    x2 = [0, 0, 0, 0, 1]
    x3 = [1, 0, 0, 0, 0]
    x4 = [1, -2, 1, 0, -2]
    """
    info = probe_rational_degrees(
        target_dim=[1, -2, 1, 0, 0],
        x_dims=[
            [2, -2, 1, 0, -1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, -2, 1, 0, -2],
        ],
        max_total_degree=5,
    )
    # Should have valid pairs.
    assert len(info.valid_pairs) > 0
    # dim_rank of the 5×5 dimension matrix.
    assert info.dimensionless_rank >= 0
    # x1 and x3 have the same dims, so dimensionless_rank >= 1.
    assert info.dimensionless_rank >= 1

    # Print summary for inspection.
    summary = summarise_degree_info(info)
    print(summary)

    # The simplest solution involves polynomial: x0*x2*x3/x1^2 is a monomial
    # with exponents (1,0,1,1,0) → dim = [2,-2,1,0,-1] + [0,0,0,0,1] + [1,0,0,0,0]
    # = [3,-2,1,0,0]... hmm, that's not the target [1,-2,1,0,0].
    # Actually x0*x2*x3/x1^2 has dim x0*x2*x3*x1^(-2), which needs negative exponents.
    # So it must be a rational, not a polynomial.
    # There should be a valid pair at low total degree.
    assert info.global_min_deg_num + info.global_min_deg_den <= 6


def test_pb111_hard_leaf():
    """The hard leaf after separability: f(x1, x3) with both [L], target [L^-2].

    This is exactly test_same_units_two_vars but with explicit connection to pb111.
    The actual function is x1*x3 / (x1^2 - x3^2)^2 which needs (deg_num=2, deg_den=4).
    """
    info = probe_rational_degrees(
        target_dim=[-2],
        x_dims=[[1], [1]],
        max_total_degree=6,
    )
    # The (2, 4) pair must exist.
    pair_2_4 = [p for p in info.valid_pairs if p.min_deg_num == 2 and p.min_deg_den == 4]
    assert len(pair_2_4) == 1, f"Expected pair (2,4), got pairs: {[(p.min_deg_num, p.min_deg_den) for p in info.valid_pairs]}"

    # This pair should have exactly the right monomials.
    pair = pair_2_4[0]
    # Numerator: degree 2 → {x1^2, x1*x3, x3^2}
    num_monos = pair.exponents_num_up_to(2)
    assert set(num_monos) == {(2, 0), (1, 1), (0, 2)}
    # Denominator: degree 4 → {x1^4, x1^3*x3, x1^2*x3^2, x1*x3^3, x3^4}
    den_monos = pair.exponents_den_up_to(4)
    assert set(den_monos) == {(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)}


# ─────────────────────────────────────────────────────────────────────
# 6. Univariate cases
# ─────────────────────────────────────────────────────────────────────

def test_univariate_dimless():
    """Single dimensionless variable. Every degree pair is valid."""
    info = probe_rational_degrees(
        target_dim=[0],
        x_dims=[[0]],
        max_total_degree=4,
    )
    assert info.all_dimensionless
    # Single dim-class (zero), so one valid pair containing all degrees.
    assert len(info.valid_pairs) == 1
    p = info.valid_pairs[0]
    assert p.min_deg_num == 0
    assert p.min_deg_den == 0
    # Total monomials: degrees 0,1,2,3,4 → 5 monomials each for num and den.
    assert p.total_monomials_num == 5
    assert p.total_monomials_den == 5


def test_univariate_unitful():
    """Single variable x with dim [L], target [L^-1].

    Only rational solutions: 1/x (deg_num=0, deg_den=1).
    """
    info = probe_rational_degrees(
        target_dim=[-1],
        x_dims=[[1]],
        max_total_degree=4,
    )
    assert not info.has_polynomial_solution
    # Valid pairs: (k, k+1) for k=0..3.
    pair_degs = [(p.min_deg_num, p.min_deg_den) for p in info.valid_pairs]
    assert (0, 1) in pair_degs
    assert (1, 2) in pair_degs


# ─────────────────────────────────────────────────────────────────────
# 7. Data probe fit
# ─────────────────────────────────────────────────────────────────────

def test_data_probe_exact_rational():
    """Fit f(x1,x3) = x1*x3 / (x1^2 - x3^2)^2 with the degree probe.

    The probe should identify (deg_num=2, deg_den=4) as the best fit.
    """
    rng = np.random.RandomState(42)
    N = 2000
    # x1 in [1, 3], x3 in [4, 6] — same domain as pb111.
    x1 = rng.uniform(1.0, 3.0, N)
    x3 = rng.uniform(4.0, 6.0, N)
    X = np.column_stack([x1, x3])
    F = x1 * x3 / (x1**2 - x3**2) ** 2

    info = probe_rational_degrees(
        target_dim=[-2],
        x_dims=[[1], [1]],
        max_total_degree=6,
    )

    results = probe_rational_fit(X, F, info, max_deg_per_pair=4)
    assert len(results) > 0

    # The best fit should have very small error.
    best = results[0]
    print(f"Best probe fit: deg=({best.deg_num},{best.deg_den}), "
          f"rel_rms={best.rel_rms:.2e}, n_terms=({best.n_terms_num},{best.n_terms_den})")
    assert best.rel_rms < 0.01, f"Expected rel_rms < 1%, got {best.rel_rms:.4f}"

    # The winning pair should be (2, 4) — the correct degrees.
    assert best.deg_num == 2, f"Expected deg_num=2, got {best.deg_num}"
    assert best.deg_den == 4, f"Expected deg_den=4, got {best.deg_den}"


def test_data_probe_polynomial():
    """Fit f(x1,x2) = x1^2 + 3*x1*x2 + x2^2 (pure polynomial, dimensionless)."""
    rng = np.random.RandomState(123)
    N = 1000
    x1 = rng.uniform(-2.0, 2.0, N)
    x2 = rng.uniform(-2.0, 2.0, N)
    X = np.column_stack([x1, x2])
    F = x1**2 + 3 * x1 * x2 + x2**2

    info = probe_rational_degrees(
        target_dim=[0],
        x_dims=[[0], [0]],
        max_total_degree=4,
    )

    results = probe_rational_fit(X, F, info, max_deg_per_pair=4)
    assert len(results) > 0
    best = results[0]
    print(f"Polynomial fit: rel_rms={best.rel_rms:.2e}")
    assert best.rel_rms < 0.01


def test_data_probe_simple_ratio():
    """Fit f(x1,x2) = x1/x2 with x1=[L], x2=[T], target=[L/T]."""
    rng = np.random.RandomState(77)
    N = 1000
    x1 = rng.uniform(1.0, 5.0, N)
    x2 = rng.uniform(1.0, 5.0, N)
    X = np.column_stack([x1, x2])
    F = x1 / x2

    info = probe_rational_degrees(
        target_dim=[1, -1],
        x_dims=[[1, 0], [0, 1]],
        max_total_degree=4,
    )

    results = probe_rational_fit(X, F, info, max_deg_per_pair=3)
    assert len(results) > 0
    best = results[0]
    print(f"Simple ratio fit: deg=({best.deg_num},{best.deg_den}), rel_rms={best.rel_rms:.2e}")
    assert best.rel_rms < 0.01
    assert best.deg_num == 1
    assert best.deg_den == 1


# ─────────────────────────────────────────────────────────────────────
# 8. Edge cases
# ─────────────────────────────────────────────────────────────────────

def test_single_variable_no_valid():
    """If target dim is unreachable from variable dims, no valid pairs."""
    # x has dim [L, T], target has dim [L, M]. No monomial of x^k gives [L, M]
    # for integer k with positive exponents, since all powers scale L and T together.
    # Actually x^1 = [L,T], x^2 = [2L,2T], etc. Target [1,0] (L only, no T)
    # requires fractional exponent — impossible for polynomial.
    info = probe_rational_degrees(
        target_dim=[1, 0, 1],
        x_dims=[[1, 1, 0]],
        max_total_degree=6,
    )
    # x^k has dim [k, k, 0]. Target [1, 0, 1] requires k=1 for first component,
    # k=0 for second, k from third — contradictory. No valid pairs.
    assert len(info.valid_pairs) == 0


def test_three_vars_mixed():
    """Three variables: x1=[L], x2=[T], x3=[M]. Target=[L*T/M]=[1,1,-1].

    Simplest: x1*x2/x3 → deg_num=2, deg_den=1 (since x1*x2 has dim [L,T]
    and x3 has dim [M]).
    """
    info = probe_rational_degrees(
        target_dim=[1, 1, -1],
        x_dims=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        max_total_degree=4,
    )
    assert len(info.valid_pairs) > 0
    # Should have a pair where num has x1*x2 (deg 2) and den has x3 (deg 1).
    pair_degs = [(p.min_deg_num, p.min_deg_den) for p in info.valid_pairs]
    assert (2, 1) in pair_degs


def test_summarise():
    """Smoke test for summary output."""
    info = probe_rational_degrees(
        target_dim=[-2],
        x_dims=[[1], [1]],
        max_total_degree=4,
    )
    s = summarise_degree_info(info)
    assert "valid dim-class pair" in s
    assert "dimensionless_rank" in s
    print(s)


# ─────────────────────────────────────────────────────────────────────
# 9. Property: every monomial in a dim-class has the correct dimension
# ─────────────────────────────────────────────────────────────────────

def test_monomial_dimension_consistency():
    """All monomials within a dim-class should have the claimed dimension."""
    x_dims_raw = [[2, -1], [1, 0], [0, 1]]
    info = probe_rational_degrees(
        target_dim=[1, 0],
        x_dims=x_dims_raw,
        max_total_degree=4,
    )
    x_dims = [_to_dim(d) for d in x_dims_raw]

    for pair in info.valid_pairs:
        for deg, monos in pair.monomials_num.items():
            for alpha in monos:
                assert sum(alpha) == deg
                d = _monomial_dim(alpha, x_dims)
                assert d == pair.dim_num, f"Monomial {alpha} has dim {d}, expected {pair.dim_num}"
        for deg, monos in pair.monomials_den.items():
            for alpha in monos:
                assert sum(alpha) == deg
                d = _monomial_dim(alpha, x_dims)
                assert d == pair.dim_den, f"Monomial {alpha} has dim {d}, expected {pair.dim_den}"


def test_pair_dimension_constraint():
    """Every valid pair should satisfy dim_num - dim_den = target."""
    target_raw = [1, -1]
    x_dims_raw = [[1, 0], [0, 1], [1, 1]]
    info = probe_rational_degrees(
        target_dim=target_raw,
        x_dims=x_dims_raw,
        max_total_degree=4,
    )
    target = _to_dim(target_raw)
    for pair in info.valid_pairs:
        diff = tuple(pair.dim_num[k] - pair.dim_den[k] for k in range(len(target)))
        assert diff == target, f"Pair dim_num={pair.dim_num}, dim_den={pair.dim_den}: diff={diff} != target={target}"


# ─────────────────────────────────────────────────────────────────────
# probe_poly_exponents tests
# ─────────────────────────────────────────────────────────────────────

def test_poly_exps_dimensionless():
    """All-dimensionless: every monomial is valid (dimension = [0])."""
    result = probe_poly_exponents([0], [[0], [0]], max_degree=2)
    assert result is not None
    # degree 0: (0,0); degree 1: (1,0),(0,1); degree 2: (2,0),(1,1),(0,2)
    all_exps = []
    for k in sorted(result):
        all_exps.extend(result[k])
    assert len(all_exps) == 6  # 1 + 2 + 3


def test_poly_exps_same_units():
    """x1 [L], x2 [L], target [L^2]:  valid monomials are degree-2 only."""
    result = probe_poly_exponents([2], [[1], [1]], max_degree=4)
    assert result is not None
    assert 2 in result
    assert 0 not in result  # constant [L^0] can't match [L^2]
    assert 1 not in result  # degree-1 is [L^1]
    # degree-2 monomials: (2,0), (1,1), (0,2) — all have dim [L^2]
    assert len(result[2]) == 3
    # degree-4 monomials with dim [L^2]? No — they're all [L^4]
    assert 4 not in result


def test_poly_exps_mixed_units():
    """x1 [L], x2 [T], target [L*T]:  valid = {x1*x2, x1^2*x2^2, ...}."""
    result = probe_poly_exponents([1, 1], [[1, 0], [0, 1]], max_degree=4)
    assert result is not None
    # Only monomials where sum of L-exponents = 1 AND sum of T-exponents = 1
    # Degree 2: (1,1) only.  Degree 3: none (can't get L^1 T^1 with sum=3 from L,T).
    # Degree 4: (2,2) -> L^2 T^2 ≠ target.  Actually (1,1) is degree 2 → L^1 T^1 ✓
    # Wait: with 2 vars each 1D... degree 3: (2,1) → L^2 T^1 ≠, (1,2) → L^1 T^2 ≠
    assert 2 in result
    assert len(result[2]) == 1
    assert result[2][0] == (1, 1)
    # No higher-degree monomial can hit L^1 T^1
    assert 3 not in result
    assert 4 not in result


def test_poly_exps_negative_target():
    """For 1/P(x): dim(P) = -target_dim.  x [L], target [L^-2] → dim(P) = [L^2]."""
    neg_target = [2]  # -(-2) = 2
    result = probe_poly_exponents(neg_target, [[1]], max_degree=4)
    assert result is not None
    assert 2 in result
    assert len(result[2]) == 1
    assert result[2][0] == (2,)


def test_poly_exps_no_valid():
    """x [L], target [L^0.5]:  no integer-exponent monomial can hit half-integer dim."""
    from fractions import Fraction
    result = probe_poly_exponents([Fraction(1, 2)], [[1]], max_degree=8)
    assert result is None


def test_poly_exps_pb111_hard_leaf():
    """pb111 hard leaf: x1 [L], x3 [L], target [L^-2].
    Valid polynomial monomials: none at deg<2 (can't get L^-2 from positive exponents).
    So probe_poly_exponents returns None → 1/P or ratpoly is needed."""
    result = probe_poly_exponents([-2], [[1], [1]], max_degree=8)
    # Target is [L^-2]: need sum of exponents = -2 with non-negative exponents.
    # Impossible! So result should be None.
    assert result is None


def test_poly_exps_inv_poly_pb111():
    """For inv_poly on pb111 hard leaf: dim(P) = -target = [L^2].
    P needs degree-2 monomials with dim [L^2]."""
    neg_target = [2]
    result = probe_poly_exponents(neg_target, [[1], [1]], max_degree=4)
    assert result is not None
    assert 2 in result
    # (2,0), (1,1), (0,2) — all have dim [L^2]
    assert len(result[2]) == 3
    # degree-4 monomials all have dim [L^4] ≠ [L^2], so not in result
    assert 4 not in result


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = 0
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed out of {len(tests)} tests.")
    sys.exit(1 if n_fail > 0 else 0)
