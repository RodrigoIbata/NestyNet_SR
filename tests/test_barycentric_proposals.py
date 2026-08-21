# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

from nestynet_sr.sr_core.bridges import AtomNode, Var
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search.compound_proposals import (
    build_barycentric_compound_proposals,
    proposal_signature,
)
from nestynet_sr.sr_search.stageB.rules import RuleBarycentricCompound


def _spec(us, x_dims, y_dim):
    return UnitsSpec(unit_system=us, x_dims=tuple(x_dims), y_dim=y_dim)


def test_barycentric_proposal_emits_weighted_average_with_units():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    M = us.dim({"M": 1})
    spec = _spec(us, [M, M, L, L], L)
    inputs = (Var(0), Var(1), Var(2), Var(3))

    props = build_barycentric_compound_proposals(inputs, units_spec=spec, wrappers=("z",), max_proposals=12)
    direct = [
        p
        for p in props
        if p.family == "weighted_avg_direct_plus"
        and p.meta.get("weights") == (0, 1)
        and p.meta.get("values") == (2, 3)
    ]

    assert direct
    assert direct[0].kind == "barycentric"
    assert direct[0].z_dim == tuple(L)
    assert direct[0].base_dim == tuple(L)
    assert direct[0].consumed_pattern == (1, 1, 1, 1)
    assert check_units_ast(direct[0].z_ast, spec).ok


def test_barycentric_proposal_rejects_unit_incompatible_pairs():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    M = us.dim({"M": 1})
    spec = _spec(us, [M, T, L, T], L)
    inputs = (Var(0), Var(1), Var(2), Var(3))

    props = build_barycentric_compound_proposals(inputs, units_spec=spec, wrappers=("z",), max_proposals=12)

    assert props == []


def test_barycentric_proposal_filters_wrapper_alias_duplicates():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    M = us.dim({"M": 1})
    spec = _spec(us, [M, M, L, L], L)
    inputs = (Var(0), Var(1), Var(2), Var(3))

    props = build_barycentric_compound_proposals(
        inputs,
        units_spec=spec,
        wrappers=("z", "identity"),
        max_proposals=12,
    )
    signatures = [proposal_signature(p) for p in props]

    assert props
    assert len(signatures) == len(set(signatures))
    assert all(p.wrapper == "z" for p in props)


def test_stageB_barycentric_rule_proposes_unit_valid_terminal_closure():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    M = us.dim({"M": 1})
    spec = _spec(us, [M, M, L, L], L)
    root = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf")
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=root),
        enforce_units=True,
        units_spec=spec,
        verbose=False,
        infer_target_dim=lambda _target: L,
    )

    cands = RuleBarycentricCompound().propose(ctx, root)
    direct = [c for c in cands if c.meta.get("barycentric_family") == "weighted_avg_direct_plus"]

    assert direct
    assert direct[0].meta["pattern_family"] == "weighted_avg_direct_plus"
    units = check_units_ast(direct[0].root, spec)
    assert units.ok, units.reason
