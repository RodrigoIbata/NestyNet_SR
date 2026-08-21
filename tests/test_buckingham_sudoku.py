# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for Buckingham-Sudoku (Level 2) global constraint propagation.

Run:
    python tests/test_buckingham_sudoku.py
"""

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, PowNode
from nestynet_sr.sr_core.units import (
    DimSubspace,
    UnitsSpec,
    UnitSystem,
    check_units_ast,
    compute_node_domains,
    propose_split,
)

# ── Helpers ──


def _us():
    """L, T, M basis."""
    return UnitSystem(("L", "T", "M"))


def _nn(var_idxs, tag=None):
    """Create an NN atom."""
    return AtomNode(kind="nn", var_idxs=tuple(var_idxs), tag=tag)


def _var(idx):
    """Create a Var leaf atom."""
    return AtomNode(kind="var", var_idxs=(idx,))


def _spec(us, x_dims, y_dim, nn_semantics="unknown", **kwargs):
    return UnitsSpec(
        unit_system=us,
        x_dims=tuple(x_dims),
        y_dim=y_dim,
        nn_semantics=nn_semantics,
        **kwargs,
    )


# ── Test 1: Additive children with incompatible dimensions ──


def test_additive_incompatible_children():
    """AddNode(NN[x0=[L]], NN[x2=[M]]) with y=[L] is infeasible.

    Under additive constraint, both children must have the same dim.
    NN0 must produce L (to match y), but NN1 must also produce L.
    With nn_semantics='span', NN1's span is {M} and L is not in span(M).
    """
    us = _us()
    L = us.dim({"L": 1})
    M = us.dim({"M": 1})

    nn0 = _nn([0], tag="nn0")
    nn1 = _nn([1], tag="nn1")
    ast = AddNode(nn0, nn1)

    spec = _spec(us, [L, M], L, nn_semantics="span")
    domains = compute_node_domains(ast, spec)
    # Should be None (inconsistent) because nn1 can't produce L
    assert domains is None, f"Expected None (infeasible), got {domains}"


# ── Test 2: Level 1 passes but Level 2 catches ──


def test_level1_passes_level2_catches():
    """MulNode(AddNode(NN[x0=[L]], NN[x1=[M]]), NN[x2=[T]]) with y=[L*T].

    Level 1 (check_units_ast): passes because unknowns can be assigned
    freely (nn_semantics='unknown').

    Level 2 (compute_node_domains) with span semantics: the AddNode forces
    NN0 and NN1 to have the same dim. But NN0 span={L}, NN1 span={M},
    so they can't both produce the same dim → infeasible.
    """
    us = _us()
    L = us.dim({"L": 1})
    M = us.dim({"M": 1})
    T = us.dim({"T": 1})

    nn0 = _nn([0], tag="nn0")
    nn1 = _nn([1], tag="nn1")
    nn2 = _nn([2], tag="nn2")
    ast = MulNode(AddNode(nn0, nn1), nn2)

    # Under 'span', this should be infeasible
    spec_span = _spec(us, [L, M, T], us.dim({"L": 1, "T": 1}), nn_semantics="span")
    domains = compute_node_domains(ast, spec_span)
    assert domains is None, "Expected infeasible under span semantics"

    # Under 'unknown', this should be feasible (Level 1 style)
    spec_unknown = _spec(us, [L, M, T], us.dim({"L": 1, "T": 1}), nn_semantics="unknown")
    domains_unk = compute_node_domains(ast, spec_unknown)
    assert domains_unk is not None, "Should be feasible under unknown semantics"


# ── Test 3: Compound that works ──


def test_compound_that_works():
    """NN[x0=[L], x1=[L]] with compound z=x0*x1 and y=[L^2] → feasible.

    The compound creates a single channel with dim L^2, and the target is L^2.
    """
    us = _us()
    L = us.dim({"L": 1})
    L2 = us.dim({"L": 2})

    # Build compound input z = x0 * x1
    var0 = _var(0)
    var1 = _var(1)
    z_expr = MulNode(var0, var1)

    nn = AtomNode(
        kind="nn", var_idxs=(0, 1), tag="nn_compound",
        inputs=(z_expr,),
    )

    spec = _spec(us, [L, L], L2, nn_semantics="span")
    domains = compute_node_domains(nn, spec)
    assert domains is not None, "Should be feasible"
    # The NN node should be pinned to L^2
    nn_domain = domains[id(nn)]
    assert nn_domain.is_pinned(), f"NN should be pinned, got rank={nn_domain.rank()}"
    assert nn_domain.offset == L2, f"Expected L^2, got {nn_domain.offset}"


# ── Test 4: Pinned nodes ──


def test_pinned_nodes():
    """Simple AST where all nodes get pinned dimensions.

    AST: NN0(x0=[L]) * NN1(x1=[T]) with y = L*T.
    Both NNs are uniquely determined: NN0→L, NN1→T.
    """
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})

    nn0 = _nn([0], tag="nn0")
    nn1 = _nn([1], tag="nn1")
    ast = MulNode(nn0, nn1)

    spec = _spec(us, [L, T], us.dim({"L": 1, "T": 1}), nn_semantics="span")
    domains = compute_node_domains(ast, spec)
    assert domains is not None, "Should be feasible"

    d0 = domains[id(nn0)]
    d1 = domains[id(nn1)]
    assert d0.is_pinned(), f"NN0 should be pinned, got rank={d0.rank()}"
    assert d1.is_pinned(), f"NN1 should be pinned, got rank={d1.rank()}"
    assert d0.offset == L, f"NN0 should be L, got {d0.offset}"
    assert d1.offset == T, f"NN1 should be T, got {d1.offset}"


# ── Test 5: Unconstrained node ──


def test_unconstrained_node():
    """Single NN with unknown semantics and target y.

    With nn_semantics='unknown', the NN is a free unknown, but the
    root constraint pins it to y_dim exactly.
    """
    us = _us()
    L = us.dim({"L": 1})

    nn = _nn([0], tag="nn0")
    spec = _spec(us, [L], L, nn_semantics="unknown")
    domains = compute_node_domains(nn, spec)
    assert domains is not None
    d = domains[id(nn)]
    # Single unknown pinned by y constraint
    assert d.is_pinned(), f"Expected pinned, got rank={d.rank()}"
    assert d.offset == L


# ── Test 6: Soundness vs oracle ──


def test_equivalence_with_check_units_ast():
    """compute_node_domains and check_units_ast must agree (same system, same answer).

    Both solve the same linear constraint system over dimensions — one via
    exact Fraction arithmetic (Sudoku), the other via the existing oracle.
    Generate a variety of small ASTs and verify bidirectional equivalence:
    oracle ok ⟹ domains non-None, and oracle reject ⟹ domains None.
    """
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    dimless = us.dimless()

    test_cases = []

    # Single NN atoms
    for x_dims, y_dim in [
        ([L], L),
        ([L, T], us.dim({"L": 1, "T": -1})),
        ([dimless], dimless),
        ([L], dimless),
    ]:
        nn = _nn(list(range(len(x_dims))), tag="nn0")
        test_cases.append((nn, x_dims, y_dim))

    # Add nodes
    for x_dims, y_dim in [
        ([L, T], L),
        ([L, L], L),
    ]:
        nn0 = _nn([0], tag="a0")
        nn1 = _nn([1], tag="a1")
        test_cases.append((AddNode(nn0, nn1), x_dims, y_dim))

    # Mul nodes
    for x_dims, y_dim in [
        ([L, T], us.dim({"L": 1, "T": 1})),
        ([L, T], us.dim({"L": 2, "T": -1})),
    ]:
        nn0 = _nn([0], tag="m0")
        nn1 = _nn([1], tag="m1")
        test_cases.append((MulNode(nn0, nn1), x_dims, y_dim))

    # Pow nodes
    nn_pow = _nn([0], tag="p0")
    test_cases.append((PowNode(nn_pow, 2), [L], us.dim({"L": 2})))

    for ast, x_dims, y_dim in test_cases:
        for sem in ("unknown", "span"):
            spec = _spec(us, x_dims, y_dim, nn_semantics=sem)
            oracle = check_units_ast(ast, spec)
            domains = compute_node_domains(ast, spec)

            if oracle.ok:
                assert domains is not None, (
                    f"Equivalence violation: check_units_ast ok but compute_node_domains "
                    f"returned None for sem={sem}, y={y_dim}"
                )
            else:
                assert domains is None, (
                    f"Equivalence violation: check_units_ast rejected but "
                    f"compute_node_domains returned feasible for sem={sem}, y={y_dim}"
                )


# ── Test 7: propose_split basic ──


def test_propose_split_feasible():
    """propose_split for a feasible additive split."""
    us = _us()
    L = us.dim({"L": 1})

    nn = _nn([0, 1], tag="nn_parent")
    spec = _spec(us, [L, L], L, nn_semantics="span")

    result = propose_split(nn, spec, nn, "add", [0], [1])
    assert result is not None, "Feasible split should return domains"


def test_propose_split_infeasible():
    """propose_split for an infeasible additive split."""
    us = _us()
    L = us.dim({"L": 1})
    M = us.dim({"M": 1})

    nn = _nn([0, 1], tag="nn_parent")
    spec = _spec(us, [L, M], L, nn_semantics="span")

    result = propose_split(nn, spec, nn, "add", [0], [1])
    # Should be None: child2 sees only M but needs L for additive consistency
    assert result is None, f"Expected infeasible, got {result}"


# ── Test 8: DimSubspace dataclass methods ──


def test_dimsubspace_methods():
    """Basic DimSubspace predicate tests."""
    us = _us()
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    M = us.dim({"M": 1})

    pinned = DimSubspace(offset=L)
    assert pinned.is_pinned()
    assert not pinned.is_unconstrained()
    assert pinned.rank() == 0
    assert pinned.n_base() == 3

    free = DimSubspace(offset=L, basis=(L, T, M))
    assert free.is_unconstrained()
    assert not free.is_pinned()
    assert free.rank() == 3
    assert free.n_base() == 3


if __name__ == "__main__":
    tests = [
        test_additive_incompatible_children,
        test_level1_passes_level2_catches,
        test_compound_that_works,
        test_pinned_nodes,
        test_unconstrained_node,
        test_equivalence_with_check_units_ast,
        test_propose_split_feasible,
        test_propose_split_infeasible,
        test_dimsubspace_methods,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            import traceback
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("Done.")
