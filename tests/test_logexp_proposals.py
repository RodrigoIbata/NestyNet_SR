# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

from nestynet_sr.sr_core import collect_nn_atoms
from nestynet_sr.sr_core.bridges import AtomNode, LogNode, Var
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search.compound_proposals import build_logexp_compound_proposals, proposal_signature
from nestynet_sr.sr_search.stageB.rules import RuleLogExpCompound


def _spec(us, x_dims, y_dim):
    return UnitsSpec(unit_system=us, x_dims=tuple(x_dims), y_dim=y_dim)


def _contains_node_type(node, typ) -> bool:
    if isinstance(node, typ):
        return True
    for attr in ("left", "right", "arg", "base"):
        if hasattr(node, attr) and _contains_node_type(getattr(node, attr), typ):
            return True
    return False


def test_logexp_ratio_proposals_are_symmetric_and_dimensionless():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L], dimless)
    inputs = (Var(0), Var(1))

    props = build_logexp_compound_proposals(inputs, units_spec=spec, wrappers=("log",), max_proposals=8)
    ratios = [p for p in props if p.family == "dimless_ratio" and p.wrapper == "log"]

    assert len(ratios) == 2
    assert {p.meta["indices"] for p in ratios} == {(0, 1), (1, 0)}
    assert all(p.z_dim == tuple(dimless) for p in ratios)
    assert all(check_units_ast(p.z_ast, spec).ok for p in ratios)


def test_logexp_rejects_unitful_raw_log_argument():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L], dimless)

    props = build_logexp_compound_proposals((Var(0),), units_spec=spec, wrappers=("log", "exp"))

    assert props == []


def test_logexp_product_zero_net_dimension_is_allowed():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    inv_L = us.dim({"L": -1})
    dimless = us.dimless()
    spec = _spec(us, [L, inv_L], dimless)

    props = build_logexp_compound_proposals((Var(0), Var(1)), units_spec=spec, wrappers=("log",), max_proposals=8)
    products = [p for p in props if p.family == "dimless_product"]

    assert products
    assert products[0].z_dim == tuple(dimless)
    assert check_units_ast(products[0].z_ast, spec).ok


def test_logexp_filters_wrapper_alias_duplicates():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L], dimless)

    props = build_logexp_compound_proposals((Var(0), Var(1)), units_spec=spec, wrappers=("log", "log_z"))
    signatures = [proposal_signature(p) for p in props]

    assert props
    assert len(signatures) == len(set(signatures))
    assert all(p.wrapper == "log" for p in props)


def test_stageB_logexp_rule_proposes_unit_valid_terminal_closure():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L], dimless)
    root = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf")
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=root),
        enforce_units=True,
        units_spec=spec,
        verbose=False,
        infer_target_dim=lambda _target: dimless,
    )

    cands = RuleLogExpCompound().propose(ctx, root)
    logs = [c for c in cands if c.meta.get("logexp_family") == "dimless_ratio" and c.meta.get("logexp_wrapper") == "log"]

    assert logs
    assert logs[0].meta["pattern_family"] == "dimless_ratio"
    units = check_units_ast(logs[0].root, spec)
    assert units.ok, units.reason
    assert collect_nn_atoms(logs[0].root) == []
    assert _contains_node_type(logs[0].root, LogNode)
