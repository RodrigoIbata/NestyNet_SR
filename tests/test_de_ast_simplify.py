# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from nestynet_sr.sr_core.ast_simplify import ast_node_count, simplify_ast
from nestynet_sr.sr_core.bridges import Add, ConstNode, DU, Mul, Pow, U


def test_simplify_ast_collects_duplicate_state_terms():
    u = U()
    expr = Add(
        Add(Mul(ConstNode(0.624996), u), Mul(ConstNode(0.624998), u)),
        Mul(ConstNode(-0.416665), Pow(u, 2.0)),
    )

    simplified = simplify_ast(expr, snap_tol=1.0e-10)
    text = repr(simplified)

    assert "1.24999" in text
    assert text.count("u") == 2
    assert "(u ** 2)" in text


def test_simplify_ast_collapses_affine_scaffold_and_preserves_residual_anchor():
    u = U()
    scaffold = Mul(ConstNode(2.0), Add(u, ConstNode(-1.0)))
    residual = Add(DU(0), Mul(ConstNode(-1.0), scaffold))

    simplified = simplify_ast(residual)
    text = repr(simplified)

    assert "u_x0" in text
    assert "-2" in text
    assert text.count("u") == 2
    assert ast_node_count(simplified) < ast_node_count(residual)


def test_simplify_ast_drops_near_zero_and_near_one_factors():
    u = U()
    expr = Add(Mul(ConstNode(1.0 + 1.0e-12), u), Mul(ConstNode(1.0e-12), u))

    simplified = simplify_ast(expr, coeff_tol=1.0e-10, snap_tol=1.0e-10)

    assert repr(simplified) == "u"
