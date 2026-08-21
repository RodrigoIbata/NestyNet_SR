# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared SymPy <-> NestyNet conversion helpers.

This module owns generic SymPy-to-NestyNet AST conversion that is used outside
of factorized symbolic search, notably by Stage B pruning.
"""

from __future__ import annotations

from typing import Mapping, Optional

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    PowNode,
    SinNode,
    Var,
    ast_equals,
    clone_ast,
    collect_all_atoms,
)
from nestynet_sr.sr_core.coefficient_metadata import named_coefficient_symbol


def coefficient_symbol_nodes_from_ast(root) -> dict[str, object]:
    """Return reusable named-coefficient AST templates keyed by printed symbol."""

    templates: dict[str, object] = {}
    for atom in collect_all_atoms(root):
        symbol = named_coefficient_symbol(atom)
        if symbol is None:
            continue
        prior = templates.get(symbol)
        if prior is not None and not ast_equals(prior, atom):
            raise ValueError(
                f"coefficient symbol {symbol!r} maps to conflicting AST atoms"
            )
        templates[symbol] = clone_ast(atom)
    return templates


def sympy_to_nestynet(
    expr,
    nvars,
    *,
    symbol_nodes: Optional[Mapping[str, object]] = None,
):
    """Convert a SymPy expression to a NestyNet_SR node tree.

    Handles:
    - `Symbol` -> `Var`
    - numeric literals -> `ConstNode`
    - n-ary `Add`/`Mul` -> left-folded binary trees
    - `Pow`
    - `sin`, `cos`, inverse trig, `exp`, `log`, `Abs`

    Non-numeric exponents are converted via `exp(exponent * log(base))`.
    """
    import sympy as sp

    if expr.is_number:
        return ConstNode(float(expr))

    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name.startswith("x") and name[1:].isdigit():
            return Var(int(name[1:]))
        if symbol_nodes is not None and name in symbol_nodes:
            return clone_ast(symbol_nodes[name])
        raise ValueError(f"Unknown symbol: {name}")

    if isinstance(expr, sp.Add):
        args = list(expr.args)
        result = sympy_to_nestynet(args[0], nvars, symbol_nodes=symbol_nodes)
        for arg in args[1:]:
            result = AddNode(
                result,
                sympy_to_nestynet(arg, nvars, symbol_nodes=symbol_nodes),
            )
        return result

    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        result = sympy_to_nestynet(args[0], nvars, symbol_nodes=symbol_nodes)
        for arg in args[1:]:
            result = MulNode(
                result,
                sympy_to_nestynet(arg, nvars, symbol_nodes=symbol_nodes),
            )
        return result

    if isinstance(expr, sp.Pow):
        base_node = sympy_to_nestynet(
            expr.args[0], nvars, symbol_nodes=symbol_nodes
        )
        exp_val = expr.args[1]
        if exp_val.is_number:
            return PowNode(base_node, float(exp_val))
        exp_node = sympy_to_nestynet(
            exp_val, nvars, symbol_nodes=symbol_nodes
        )
        return ExpNode(MulNode(exp_node, LogNode(base_node)))

    if isinstance(expr, sp.sin):
        return SinNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )
    if isinstance(expr, sp.cos):
        return CosNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )
    if expr.func == sp.asin:
        return AsinNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )
    if expr.func == sp.acos:
        return AcosNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )
    if expr.func == sp.atan:
        return AtanNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )
    if isinstance(expr, sp.exp):
        return ExpNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )
    if isinstance(expr, sp.log):
        return LogNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )
    if isinstance(expr, sp.Abs):
        return AbsNode(
            sympy_to_nestynet(expr.args[0], nvars, symbol_nodes=symbol_nodes)
        )

    raise ValueError(f"Unsupported sympy type: {type(expr).__name__}: {expr}")


__all__ = ["coefficient_symbol_nodes_from_ast", "sympy_to_nestynet"]
