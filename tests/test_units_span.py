# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for nn_semantics="span" in the units system.

Run:
    python tests/test_units_span.py
"""


from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, PowNode
from nestynet_sr.sr_core.units import (
    UnitSystem,
    UnitsSpec,
    check_units_ast,
    infer_atom_output_dim,
)


def _us():
    """L, T, M basis."""
    return UnitSystem(("L", "T", "M"))


def _nn(var_idxs, tag=None):
    """Create an NN atom."""
    return AtomNode(kind="nn", var_idxs=tuple(var_idxs), tag=tag)


def _var(idx):
    """Create a Var leaf atom."""
    return AtomNode(kind="var", var_idxs=(idx,))


def _free_const(name, tag=None):
    """Create a free_const leaf atom."""
    return AtomNode(kind="free_const", var_idxs=(), kwargs={"name": name}, tag=tag or name)


def _spec(us, x_dims, y_dim, nn_semantics="span", free_const_dims=None, fixed_const_dims=None):
    return UnitsSpec(
        unit_system=us,
        x_dims=tuple(x_dims),
        y_dim=y_dim,
        nn_semantics=nn_semantics,
        free_const_dims=free_const_dims or {},
        fixed_const_dims=fixed_const_dims or {},
    )


# ── Test: NN(L, T) can produce L·T⁻² (in span) ──


def test_span_nn_produces_combination():
    """NN(L,T) should be able to produce L·T⁻² under span semantics."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    y_dim = us.dim({"L": 1, "T": -2})  # acceleration

    nn = _nn([0, 1], tag="nn0")
    spec = _spec(us, [L, T], y_dim, nn_semantics="span")
    result = check_units_ast(nn, spec)
    assert result.ok, f"Should pass: {result.reason}"


# ── Test: NN(L, T) cannot produce M (not in span) ──


def test_span_nn_rejects_outside_span():
    """NN(L,T) should NOT produce M under span semantics."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    M = us.dim({"M": 1})

    nn = _nn([0, 1], tag="nn0")
    spec = _spec(us, [L, T], M, nn_semantics="span")
    result = check_units_ast(nn, spec)
    assert not result.ok, "Should reject: M is not in span(L, T)"


# ── Test: NN(L) can produce L² ──


def test_span_nn_produces_powers():
    """NN(L) can produce L² (rational scaling of the single basis vector)."""
    us = _us()
    L = us.dim({"L": 1})
    L2 = us.dim({"L": 2})

    nn = _nn([0], tag="nn0")
    spec = _spec(us, [L], L2, nn_semantics="span")
    result = check_units_ast(nn, spec)
    assert result.ok, f"Should pass: {result.reason}"


# ── Test: NN(dimless inputs) → forced dimless ──


def test_span_all_dimless_degenerates():
    """NN with all-dimless inputs degenerates to dimensionless output."""
    us = _us()
    dimless = us.dimless()
    L = us.dim({"L": 1})

    nn = _nn([0, 1], tag="nn0")
    # Target is L, but NN is dimless due to dimless inputs → should fail
    spec = _spec(us, [dimless, dimless], L, nn_semantics="span")
    result = check_units_ast(nn, spec)
    assert not result.ok, "Should fail: dimless NN can't produce L"

    # Target is dimless → should pass
    spec2 = _spec(us, [dimless, dimless], dimless, nn_semantics="span")
    result2 = check_units_ast(nn, spec2)
    assert result2.ok, f"Should pass for dimless target: {result2.reason}"


# ── Test: free_const with dim T extends span ──


def test_span_free_const_extends():
    """A declared free_const with dim T extends the span basis."""
    us = _us()
    L = us.dim({"L": 1})
    M = us.dim({"M": 1})

    # NN sees only x0 (L), but free_const_dims declares a constant with dim M.
    # So the span should be {L, M} and NN can produce L·M but not L·T.
    nn = _nn([0], tag="nn0")

    # AST: nn + free_const  (additive requires same dim)
    # We'll test just the NN alone with a target in span(L, M).
    y_dim = us.dim({"L": 1, "M": -1})
    spec = _spec(us, [L], y_dim, nn_semantics="span", free_const_dims={"mass_scale": M})
    result = check_units_ast(nn, spec)
    assert result.ok, f"Should pass with free_const extending span: {result.reason}"

    # Target in T direction should still fail.
    y_dim_t = us.dim({"T": 1})
    spec2 = _spec(us, [L], y_dim_t, nn_semantics="span", free_const_dims={"mass_scale": M})
    result2 = check_units_ast(nn, spec2)
    assert not result2.ok, "Should fail: T not in span(L, M)"


# ── Test: Two NNs with different spans in one AST ──


def test_span_multiple_nn():
    """Two NN atoms with different var_idxs → different span bases."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})

    # AST: NN1(x0=L) * NN2(x1=T)
    # Under mul: dim(product) = dim(NN1) + dim(NN2) in exponent space.
    # dim(NN1) ∈ span(L), dim(NN2) ∈ span(T).
    # Target dim = L * T = L^1 * T^1 → dim(NN1)=L, dim(NN2)=T → OK.
    nn1 = _nn([0], tag="nn1")
    nn2 = _nn([1], tag="nn2")
    ast = MulNode(nn1, nn2)

    y_dim = us.dim({"L": 1, "T": 1})
    spec = _spec(us, [L, T], y_dim, nn_semantics="span")
    result = check_units_ast(ast, spec)
    assert result.ok, f"Should pass: {result.reason}"

    # Target L * M: NN2(T) can't produce M → fail.
    y_dim2 = us.dim({"L": 1, "M": 1})
    spec2 = _spec(us, [L, T], y_dim2, nn_semantics="span")
    result2 = check_units_ast(ast, spec2)
    assert not result2.ok, "Should fail: M not in span of either NN"


# ── Test: "unknown" still works identically ──


def test_span_backward_compat():
    """nn_semantics='unknown' should behave as before (NN can produce any dim)."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    M = us.dim({"M": 1})

    nn = _nn([0, 1], tag="nn0")
    # Under "unknown", NN(L,T) can produce M (free unknown).
    spec = _spec(us, [L, T], M, nn_semantics="unknown")
    result = check_units_ast(nn, spec)
    assert result.ok, f"Backward compat: unknown should pass: {result.reason}"


# ── Test: infer_atom_output_dim returns correct dim under span ──


def test_span_infer_atom_dim():
    """infer_atom_output_dim should return the correct dim for a span atom."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    y_dim = us.dim({"L": 1, "T": -2})

    nn = _nn([0, 1], tag="nn0")
    spec = _spec(us, [L, T], y_dim, nn_semantics="span")

    inferred = infer_atom_output_dim(nn, nn, spec)
    assert inferred is not None, "Should infer a concrete dim"
    assert inferred == y_dim, f"Expected {y_dim}, got {inferred}"


# ── Test: AST with both free and span atoms ──


def test_span_mixed_free_and_span():
    """AST containing both a span NN and a non-span (poly) atom."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})

    # AST: NN(x0=L, x1=T) + poly(x0, deg=1)
    # poly(x0, deg=1) → dim = L
    # Add constraint: dim(NN) == dim(poly) == L
    # NN span is {L, T}, so L is reachable → OK
    nn = _nn([0, 1], tag="nn0")
    poly = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1, "min_total": 1})
    ast = AddNode(nn, poly)

    y_dim = L
    spec = _spec(us, [L, T], y_dim, nn_semantics="span")
    result = check_units_ast(ast, spec)
    assert result.ok, f"Should pass: {result.reason}"

    # Now change target to M → poly can't produce M, so fails before span matters.
    M = us.dim({"M": 1})
    spec2 = _spec(us, [L, T], M, nn_semantics="span")
    result2 = check_units_ast(ast, spec2)
    assert not result2.ok, "Should fail: poly(x0=L) can't produce M"


# ── Test: compound input z=x0/x1 (dimless ratio) forces dimless NN ──


def test_span_compound_dimless_ratio():
    """NN with compound input z = x0/x1 where x0=L, x1=L → z is dimensionless.

    The span basis should reflect the *effective* input dims, not the raw
    var_idxs.  Since z is dimensionless, the NN is forced dimless and cannot
    produce a unitful target.
    """
    us = _us()
    L = us.dim({"L": 1})
    dimless = us.dimless()

    # Build compound input expression: z = x0 / x1 = x0 * x1^(-1)
    var0 = AtomNode(kind="var", var_idxs=(0,))
    var1 = AtomNode(kind="var", var_idxs=(1,))
    z_expr = MulNode(var0, PowNode(var1, -1))

    # NN atom with compound input z = x0/x1
    nn = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="nn_ratio",
        inputs=(z_expr,),
    )

    # Target = L: NN sees only a dimensionless channel, so should FAIL
    spec_L = _spec(us, [L, L], L, nn_semantics="span")
    result = check_units_ast(nn, spec_L)
    assert not result.ok, "Should fail: compound dimless input can't produce L"

    # Target = dimless: should PASS
    spec_dl = _spec(us, [L, L], dimless, nn_semantics="span")
    result2 = check_units_ast(nn, spec_dl)
    assert result2.ok, f"Should pass for dimless target: {result2.reason}"


# ── Test: compound input z=x0*x1 (unitful product) retains units ──


def test_span_compound_unitful_product():
    """NN with compound input z = x0 * x1 where x0=L, x1=T → z = L*T.

    The NN sees a single channel with dim L*T, so span = {L*T} and the
    NN can produce (L*T)^n for any rational n.
    """
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})

    var0 = AtomNode(kind="var", var_idxs=(0,))
    var1 = AtomNode(kind="var", var_idxs=(1,))
    z_expr = MulNode(var0, var1)

    nn = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="nn_prod",
        inputs=(z_expr,),
    )

    # Target = L*T: NN sees L*T, so it can produce L*T → OK
    y_dim = us.dim({"L": 1, "T": 1})
    spec = _spec(us, [L, T], y_dim, nn_semantics="span")
    result = check_units_ast(nn, spec)
    assert result.ok, f"Should pass: {result.reason}"

    # Target = L²*T²: NN can produce (L*T)^2 = L²T² → OK
    y_dim2 = us.dim({"L": 2, "T": 2})
    spec2 = _spec(us, [L, T], y_dim2, nn_semantics="span")
    result2 = check_units_ast(nn, spec2)
    assert result2.ok, f"Should pass for (L*T)^2: {result2.reason}"

    # Target = L: requires L = (L*T)^a, i.e. a=1 but then T^1 ≠ 0 → FAIL
    spec3 = _spec(us, [L, T], L, nn_semantics="span")
    result3 = check_units_ast(nn, spec3)
    assert not result3.ok, "Should fail: L is not in span(L*T)"


# ── Test: linearly dependent basis vectors are reduced ──


def test_span_dependent_basis_reduced():
    """NN with inputs x0=L, x1=L² (linearly dependent in exponent space).

    L² = 2·L, so with both in the basis _build_combined_system would assign
    two columns (c₁, c₂) with constraint c₁ + 2·c₂ = k.  That has infinitely
    many solutions, so infer_atom_output_dim returns None.

    After _reduce_to_independent filters to {L} alone, the single column gives
    a unique solution and infer_atom_output_dim returns the correct dim.
    """
    us = _us()
    L = us.dim({"L": 1})
    L2 = us.dim({"L": 2})
    L3 = us.dim({"L": 3})

    nn = _nn([0, 1], tag="nn_dep")
    spec = _spec(us, [L, L2], L3, nn_semantics="span")

    # check_units_ast should still pass (L³ is in span(L))
    result = check_units_ast(nn, spec)
    assert result.ok, f"Should pass: {result.reason}"

    # infer_atom_output_dim should return L³ (not None)
    inferred = infer_atom_output_dim(nn, nn, spec)
    assert inferred is not None, (
        "infer_atom_output_dim returned None — dependent basis was not reduced"
    )
    assert inferred == L3, f"Expected {L3}, got {inferred}"


if __name__ == "__main__":
    tests = [
        test_span_nn_produces_combination,
        test_span_nn_rejects_outside_span,
        test_span_nn_produces_powers,
        test_span_all_dimless_degenerates,
        test_span_free_const_extends,
        test_span_multiple_nn,
        test_span_backward_compat,
        test_span_infer_atom_dim,
        test_span_mixed_free_and_span,
        test_span_compound_dimless_ratio,
        test_span_compound_unitful_product,
        test_span_dependent_basis_reduced,
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
