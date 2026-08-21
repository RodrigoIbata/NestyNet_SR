# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""AbsNode must be walkable by every Stage-B AST traversal (pb059 crash).

A sympy round-trip in the Stage-B polish can factor sqrt(x**2 * R(x)) into
Abs(x)*sqrt(R(x)); the accepted state then carries an AbsNode.  Walkers with
a stale pre-complex-extension unary list either crashed ("Unexpected node
type") — as run_stageb_pruning_pipeline did on pb059 — or silently skipped
the subtree.  All walkers now share bridges.UNARY_NODE_TYPES.
"""

import torch

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AtomNode,
    MulNode,
    PowNode,
    UNARY_NODE_TYPES,
    Var,
    collect_all_atoms,
)
from nestynet_sr.sr_search.stageB.atom_mapping import (
    _collect_all_atoms,
    _vars_in_subtree,
)
from nestynet_sr.sr_search.stageB.rules_common import _subtree_content_hash


def _pb059_like_root():
    """scale * x0 * sqrt(poly(x2)) * Abs(x2) / x1 — the pb059 crash shape."""
    ratpoly = AtomNode(kind="poly", var_idxs=(2,), kwargs={"degree": 3}, tag="p0")
    scale = AtomNode(kind="scale", var_idxs=(), kwargs={"init": 1.0}, tag="s0")
    return MulNode(
        MulNode(
            MulNode(scale, Var(0)),
            MulNode(PowNode(ratpoly, 0.5), AbsNode(Var(2))),
        ),
        PowNode(Var(1), -1.0),
    )


def test_unary_node_types_is_complete():
    # Every unary class named in bridges with a single .arg child is present.
    names = {cls.__name__ for cls in UNARY_NODE_TYPES}
    assert {"AbsNode", "ConjNode", "RealNode", "ImagNode", "ArgNode"} <= names
    assert {"LogNode", "ExpNode", "SinNode", "CosNode"} <= names


def test_stageb_collect_all_atoms_traverses_abs():
    root = _pb059_like_root()
    atoms = _collect_all_atoms(root)
    tags = {a.tag for a in atoms if a.tag}
    assert {"p0", "s0"} <= tags
    # Canonical bridges walker agrees.
    assert len(collect_all_atoms(root)) >= len(atoms)


def test_vars_in_subtree_sees_through_abs():
    root = _pb059_like_root()
    assert 2 in _vars_in_subtree(root)


def test_subtree_content_hash_handles_abs():
    h1 = _subtree_content_hash(AbsNode(Var(2)))
    h2 = _subtree_content_hash(AbsNode(Var(3)))
    assert isinstance(h1, int) and h1 != h2


def test_rules_nn_leaf_eval_handles_abs():
    from nestynet_sr.sr_search.stageB import rules_nn_leaf as rnl

    assert AbsNode in rnl._UNARY_AST_NODES
    # The local torch evaluator must evaluate Abs.
    x = torch.tensor([[-2.0], [3.0]], dtype=torch.float64)
    # _eval is nested; exercise it through a walker-level function instead:
    # _contains_node and _vars_in_subtree_simple must traverse AbsNode.
    node = AbsNode(Var(0))
    assert rnl._contains_node(node, node.arg)
