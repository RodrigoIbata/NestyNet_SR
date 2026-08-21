# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for DE-feature leaf rendering in Stage C representation."""

from nestynet_sr.sr_core.atoms import VarLeaf
from nestynet_sr.sr_core.bridges import AtomNode, D2U, DU, U
from nestynet_sr.sr_search.representation import _leaf_to_repr


def test_leaf_to_repr_handles_de_feature_atoms_by_kind():
    dummy_leaf = object()
    atoms = [
        U(),
        DU(0),
        D2U(0, 0),
        AtomNode(kind="grad_u", var_idxs=(), kwargs={"axis": 0}),
        AtomNode(kind="hess_u", var_idxs=(), kwargs={"axis0": 0, "axis1": 0}),
        AtomNode(kind="field", var_idxs=(), kwargs={"field": "E", "comp_name": "x"}),
        AtomNode(kind="state", var_idxs=()),
    ]

    for atom in atoms:
        scale, core_str = _leaf_to_repr(atom, dummy_leaf)
        assert scale == 1.0
        assert core_str == repr(atom)


def test_leaf_to_repr_handles_varleaf_fallback():
    atom = AtomNode(kind="poly", var_idxs=(2,), kwargs={}, tag="p2")
    scale, core_str = _leaf_to_repr(atom, VarLeaf())
    assert scale == 1.0
    assert core_str == "x2"
