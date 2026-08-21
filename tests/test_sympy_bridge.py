# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import sympy as sp

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    FixedConst,
    FreeConst,
    MulNode,
    PowNode,
    ast_equals,
)
from nestynet_sr.sr_core.sympy_bridge import (
    coefficient_symbol_nodes_from_ast,
    sympy_to_nestynet,
)
from nestynet_sr.sr_search.factorized_search.bridge import sympy_to_nestynet as legacy_sympy_to_nestynet


def test_sympy_bridge_converts_basic_expression():
    expr = sp.sympify("x0 + x1*x2")
    node = sympy_to_nestynet(expr, 3)

    assert isinstance(node, AddNode)
    assert isinstance(node.left, AtomNode)
    assert node.left.kind == "var"
    assert node.left.var_idxs == (0,)
    assert isinstance(node.right, MulNode)


def test_sympy_bridge_handles_numeric_power_and_legacy_reexport():
    expr = sp.sympify("x0**2")
    node = sympy_to_nestynet(expr, 1)
    legacy = legacy_sympy_to_nestynet(expr, 1)

    assert isinstance(node, PowNode)
    assert node.exponent == 2.0
    assert type(legacy) is type(node)


def test_sympy_bridge_rebuilds_named_coefficient_identity_and_scope():
    root = AddNode(
        FreeConst("c", tag="shared_c", init=2.0, scope="class"),
        FixedConst("pi", tag="declared_pi", value=3.0),
    )
    templates = coefficient_symbol_nodes_from_ast(root)

    rebuilt = sympy_to_nestynet(
        sp.sympify("2*c*x0 + coef_pi", evaluate=False),
        1,
        symbol_nodes=templates,
    )
    rebuilt_templates = coefficient_symbol_nodes_from_ast(rebuilt)

    assert set(rebuilt_templates) == {"c", "coef_pi"}
    assert ast_equals(rebuilt_templates["c"], root.left)
    assert ast_equals(rebuilt_templates["coef_pi"], root.right)
    assert rebuilt_templates["c"] is not root.left
