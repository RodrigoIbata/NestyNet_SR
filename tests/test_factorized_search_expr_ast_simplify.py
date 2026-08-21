# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.factorized_search.expr_ast import node_str, simplify


def test_simplify_log_mul_exp_rewrites_to_add():
    expr = ("log", ("mul", ("mul", ("var", 0), ("var", 1)), ("exp", ("var", 2))))
    simp = simplify(expr)

    assert simp[0] == "add"
    child_strs = {node_str(simp[1]), node_str(simp[2])}
    assert child_strs == {"log((x0*x1))", "x2"}


def test_simplify_log_div_exp_rewrites_to_sub():
    expr = ("log", ("div", ("mul", ("var", 0), ("var", 1)), ("exp", ("var", 2))))
    simp = simplify(expr)

    assert simp[0] == "sub"
    assert node_str(simp[1]) == "log((x0*x1))"
    assert node_str(simp[2]) == "x2"
