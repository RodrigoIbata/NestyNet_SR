# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from typing import Any, Sequence

from .expr_ast import INVERSE_TRIG_OPS, is_valid_node, node_str, simplify


def _valid_node(node: Any) -> tuple | None:
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


def _flatten_commutative(node: tuple, op: str) -> list[tuple]:
    out: list[tuple] = []

    def _visit(cur: tuple) -> None:
        if isinstance(cur, tuple) and len(cur) == 3 and str(cur[0]) == str(op):
            _visit(cur[1])
            _visit(cur[2])
            return
        out.append(cur)

    _visit(node)
    return out


def _rebuild_commutative(op: str, children: list[tuple]) -> tuple:
    if not children:
        return ("const", 0.0) if op == "add" else ("const", 1.0)
    cur = children[0]
    for child in children[1:]:
        cur = (str(op), cur, child)
    return cur


def canonicalize_basis_expr(node: Any) -> tuple | None:
    cur = _valid_node(node)
    if cur is None:
        return None
    cur = simplify(cur)
    op = str(cur[0])
    if op in ("var", "const", "hparam"):
        return cur
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg", *INVERSE_TRIG_OPS):
        child = canonicalize_basis_expr(cur[1])
        return simplify((op, child)) if child is not None else cur
    if op == "add":
        parts = [canonicalize_basis_expr(part) for part in _flatten_commutative(cur, "add")]
        valid_parts = [part for part in parts if part is not None and not (part[0] == "const" and float(part[1]) == 0.0)]
        if not valid_parts:
            return ("const", 0.0)
        valid_parts.sort(key=lambda item: str(node_str(item)))
        return simplify(_rebuild_commutative("add", valid_parts))
    if op == "mul":
        parts = [canonicalize_basis_expr(part) for part in _flatten_commutative(cur, "mul")]
        valid_parts = [part for part in parts if part is not None]
        for part in valid_parts:
            if part[0] == "const" and float(part[1]) == 0.0:
                return ("const", 0.0)
        valid_parts = [part for part in valid_parts if not (part[0] == "const" and float(part[1]) == 1.0)]
        if not valid_parts:
            return ("const", 1.0)
        valid_parts.sort(key=lambda item: str(node_str(item)))
        return simplify(_rebuild_commutative("mul", valid_parts))
    if op in ("sub", "div"):
        left = canonicalize_basis_expr(cur[1])
        right = canonicalize_basis_expr(cur[2])
        return simplify((op, left, right)) if left is not None and right is not None else cur
    return cur


def basis_expr_key(node: Any) -> str:
    cur = canonicalize_basis_expr(node)
    if cur is None:
        return ""
    try:
        return str(node_str(cur))
    except Exception:
        return ""


def _structure_signature(node: tuple, *, collapse_scalar_consts: bool) -> Any:
    op = str(node[0])
    if op == "var":
        try:
            return ("var", int(node[1]))
        except Exception:
            return ("var", str(node[1]))
    if op in ("const", "hparam"):
        if collapse_scalar_consts:
            return ("const",)
        return (op, node[1] if len(node) > 1 else None)
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg", *INVERSE_TRIG_OPS):
        return (op, _structure_signature(node[1], collapse_scalar_consts=collapse_scalar_consts))
    if op == "add":
        parts = [
            _structure_signature(part, collapse_scalar_consts=collapse_scalar_consts)
            for part in _flatten_commutative(node, "add")
        ]
        parts.sort(key=repr)
        return ("add", tuple(parts))
    if op == "mul":
        parts = [
            _structure_signature(part, collapse_scalar_consts=collapse_scalar_consts)
            for part in _flatten_commutative(node, "mul")
        ]
        if collapse_scalar_consts:
            parts = [part for part in parts if part != ("const",)]
            if not parts:
                return ("const",)
        parts.sort(key=repr)
        return ("mul", tuple(parts))
    if op in ("sub", "div"):
        return (
            op,
            _structure_signature(node[1], collapse_scalar_consts=collapse_scalar_consts),
            _structure_signature(node[2], collapse_scalar_consts=collapse_scalar_consts),
        )
    return (op,)


def basis_structure_signature(node: Any, *, collapse_scalar_consts: bool = True) -> Any:
    cur = canonicalize_basis_expr(node)
    if cur is None:
        return None
    return _structure_signature(cur, collapse_scalar_consts=bool(collapse_scalar_consts))


def basis_structure_signature_key(node: Any, *, collapse_scalar_consts: bool = True) -> str:
    sig = basis_structure_signature(node, collapse_scalar_consts=collapse_scalar_consts)
    return "" if sig is None else repr(sig)


def compile_basis_linear_combo(
    nodes: Sequence[tuple],
    coeffs: Sequence[float],
    intercept: float,
) -> tuple | None:
    terms: list[tuple] = []
    for node, raw_coeff in zip(list(nodes or ()), list(coeffs or ())):
        valid = _valid_node(node)
        if valid is None:
            continue
        try:
            coeff = float(raw_coeff)
        except Exception:
            continue
        if not math.isfinite(coeff) or abs(coeff) <= 1.0e-12:
            continue
        if abs(coeff - 1.0) <= 1.0e-12:
            term = valid
        elif abs(coeff + 1.0) <= 1.0e-12:
            term = ("neg", valid)
        else:
            term = ("mul", ("const", float(coeff)), valid)
        terms.append(simplify(term))
    if math.isfinite(float(intercept)) and abs(float(intercept)) > 1.0e-12:
        terms.append(("const", float(intercept)))
    if not terms:
        return None
    expr = terms[0]
    for term in terms[1:]:
        expr = simplify(("add", expr, term))
    return canonicalize_basis_expr(expr)


__all__ = [
    "basis_expr_key",
    "basis_structure_signature",
    "basis_structure_signature_key",
    "canonicalize_basis_expr",
    "compile_basis_linear_combo",
]
