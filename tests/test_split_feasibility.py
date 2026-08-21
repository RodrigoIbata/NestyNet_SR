# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for the dimensional feasibility gate (check_split_feasibility).

Run:
    python tests/test_split_feasibility.py
"""


from nestynet_sr.sr_core.units import (
    UnitSystem,
    check_split_feasibility,
    _dim_in_rational_span,
)


def _us():
    """L, T, M basis."""
    return UnitSystem(("L", "T", "M"))


def test_dim_in_rational_span_basic():
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    LT2 = us.dim({"L": 1, "T": -2})  # acceleration

    # L^a * T^b can reach L*T^-2
    assert _dim_in_rational_span(LT2, [L, T])
    # L alone cannot reach L*T^-2
    assert not _dim_in_rational_span(LT2, [L])
    # T alone cannot reach L*T^-2
    assert not _dim_in_rational_span(LT2, [T])
    # Dimensionless target: always reachable
    assert _dim_in_rational_span(us.dimless(), [])
    assert _dim_in_rational_span(us.dimless(), [L])
    # Empty basis, non-dimless target: unreachable
    assert not _dim_in_rational_span(L, [])


def test_additive_feasible():
    """Both children can independently reach y_dim."""
    us = _us()
    L = us.dim({"L": 1})
    # y = NN1(x0) + NN2(x1), both x0 and x1 are length => y dim L is reachable
    ok, reason = check_split_feasibility("add", [0], [1], L, (L, L), us)
    assert ok, reason


def test_additive_infeasible():
    """One child's input dim can't reach y_dim."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    LT2 = us.dim({"L": 1, "T": -2})
    # y dim L*T^-2, child1 has only L, child2 has only T
    # child1 can only produce L^p, not L*T^-2
    ok, reason = check_split_feasibility("add", [0], [1], LT2, (L, T), us)
    assert not ok
    assert "child1" in reason


def test_multiplicative_feasible():
    """Combined inputs span y_dim."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    LT2 = us.dim({"L": 1, "T": -2})
    # y dim L*T^-2, combined inputs have L and T => reachable
    ok, reason = check_split_feasibility("mul", [0], [1], LT2, (L, T), us)
    assert ok, reason


def test_multiplicative_infeasible():
    """y_dim has a base component absent from all inputs."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    MLT = us.dim({"M": 1, "L": 1, "T": -1})  # needs mass
    # Inputs only have L and T, but y needs M
    ok, reason = check_split_feasibility("mul", [0], [1], MLT, (L, T), us)
    assert not ok
    assert "multiplicative" in reason


def test_multiplicative_with_free_const():
    """Free constant extends reachability."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    M = us.dim({"M": 1})
    MLT = us.dim({"M": 1, "L": 1, "T": -1})
    # Inputs L, T can't reach M*L*T^-1 alone, but a free constant with dim M can
    ok, reason = check_split_feasibility(
        "mul", [0], [1], MLT, (L, T), us,
        free_const_dims={"mass_const": M},
    )
    assert ok, reason


def test_dimensionless_target_always_feasible():
    """Dimensionless target is always feasible regardless of inputs."""
    us = _us()
    L = us.dim({"L": 1})
    dimless = us.dimless()
    ok1, _ = check_split_feasibility("add", [0], [1], dimless, (L, L), us)
    ok2, _ = check_split_feasibility("mul", [0], [1], dimless, (L, L), us)
    assert ok1
    assert ok2


def test_all_dimensionless_inputs():
    """All-dimensionless inputs: always feasible for dimensionless target."""
    us = _us()
    dimless = us.dimless()
    ok, _ = check_split_feasibility("add", [0], [1], dimless, (dimless, dimless), us)
    assert ok


def test_all_dimensionless_inputs_unitful_target():
    """All-dimensionless inputs cannot produce unitful target without free consts."""
    us = _us()
    dimless = us.dimless()
    L = us.dim({"L": 1})
    ok, _ = check_split_feasibility("add", [0], [1], L, (dimless, dimless), us)
    assert not ok


def test_additive_free_const_rescues_child():
    """A free constant with matching dim makes an otherwise infeasible additive child feasible."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    LT2 = us.dim({"L": 1, "T": -2})
    # child1 has only L, can't reach L*T^-2 alone
    # but a free constant with dim T^-2 allows it
    T_inv2 = us.dim({"T": -2})
    ok, reason = check_split_feasibility(
        "add", [0], [1], LT2, (L, T), us,
        free_const_dims={"omega2": T_inv2},
    )
    # child1 basis: [L, T^-2] => L * T^-2 reachable; child2 basis: [T, T^-2] => L*T^-2 not reachable from T alone
    # Actually child2 has [T, T^-2] which cannot produce L. So still infeasible.
    assert not ok
    assert "child2" in reason


def test_unknown_op_passes():
    """Unknown op type is not blocked."""
    us = _us()
    L = us.dim({"L": 1})
    ok, _ = check_split_feasibility("unknown_op", [0], [1], L, (L, L), us)
    assert ok


def test_context_dims_mul_sibling():
    """context_dims from a multiplicative sibling relaxes the target.

    Scenario: AST = NN[x0] * NN[x1,x3,x4,x5], target = L^2*T^-2*M.
    NN[x0] provides M, so NN[x1,x3,x4,x5] only needs L^2*T^-2.
    Without context_dims, splitting [1]/[3,4,5] as mul fails because
    {T^-1,L,L,1} can't reach L^2*T^-2*M. With context_dims=[M], it succeeds.
    """
    us = _us()
    M = us.dim({"M": 1})
    L = us.dim({"L": 1})
    T_inv = us.dim({"T": -1})
    dimless = us.dimless()
    target = us.dim({"L": 2, "T": -2, "M": 1})
    x_dims = (M, T_inv, dimless, L, L, dimless)  # x0=M, x1=T^-1, x2=1, x3=L, x4=L, x5=1

    # Without context_dims: mul split [1]/[3,4,5] should fail
    ok_no_ctx, reason = check_split_feasibility(
        "mul", [1], [3, 4, 5], target, x_dims, us,
    )
    assert not ok_no_ctx, "Should fail without context_dims"

    # With context_dims=[M] from sibling NN[x0]: should pass
    ok_ctx, reason = check_split_feasibility(
        "mul", [1], [3, 4, 5], target, x_dims, us,
        context_dims=[M],
    )
    assert ok_ctx, f"Should pass with context_dims=[M], but: {reason}"


def test_context_dims_additive_child():
    """context_dims helps additive children too.

    If atom is inside a MulNode, an additive split of the atom
    benefits from context_dims on each child.
    """
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    target = us.dim({"L": 1, "T": -1})
    # child1 has [L], child2 has [L] — neither can reach L*T^-1 alone
    ok_no_ctx, _ = check_split_feasibility("add", [0], [1], target, (L, L), us)
    assert not ok_no_ctx

    # With context_dims=[T], each child basis becomes [L, T] which can reach L*T^-1
    ok_ctx, reason = check_split_feasibility(
        "add", [0], [1], target, (L, L), us, context_dims=[T],
    )
    assert ok_ctx, f"Should pass with context_dims=[T], but: {reason}"


if __name__ == "__main__":
    tests = [
        test_dim_in_rational_span_basic,
        test_additive_feasible,
        test_additive_infeasible,
        test_multiplicative_feasible,
        test_multiplicative_infeasible,
        test_multiplicative_with_free_const,
        test_dimensionless_target_always_feasible,
        test_all_dimensionless_inputs,
        test_all_dimensionless_inputs_unitful_target,
        test_additive_free_const_rescues_child,
        test_unknown_op_passes,
        test_context_dims_mul_sibling,
        test_context_dims_additive_child,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("Done.")
