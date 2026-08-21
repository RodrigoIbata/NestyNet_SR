# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Core AST bridge for the opt-in expression IR."""

from __future__ import annotations

from typing import Any

from .config import coerce_expr_ir_config, expr_ir_active
from .qdag import QArena
from .stats import ExpressionIRStats


def _atom_payload(node: Any) -> tuple:
    kind = str(getattr(node, "kind", "atom"))
    var_idxs = tuple(int(v) for v in tuple(getattr(node, "var_idxs", ()) or ()))
    tag = getattr(node, "tag", None)
    kwargs = getattr(node, "kwargs", {}) or {}
    kwargs_key = tuple(sorted((str(k), repr(v)) for k, v in dict(kwargs).items()))
    inputs = tuple(repr(v) for v in tuple(getattr(node, "inputs", ()) or ()))
    return kind, var_idxs, kwargs_key, None if tag is None else str(tag), inputs


def core_ast_to_qdag(node: Any, arena: QArena) -> int:
    from nestynet_sr.sr_core.bridges import (
        AcosNode,
        AddNode,
        ArgNode,
        AsinNode,
        AtanNode,
        AtomNode,
        AbsNode,
        ConjNode,
        ConstNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        PowNode,
        RealNode,
        SinNode,
    )

    if isinstance(node, ConstNode):
        return arena.const(node.value)
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        var_idxs = tuple(getattr(node, "var_idxs", ()) or ())
        if kind in {"var", "x", "input"} and len(var_idxs) == 1 and not getattr(node, "tag", None):
            return arena.var(int(var_idxs[0]))
        return arena.atom(_atom_payload(node), original=node)
    if isinstance(node, AddNode):
        return arena.add(core_ast_to_qdag(node.left, arena), core_ast_to_qdag(node.right, arena))
    if isinstance(node, MulNode):
        return arena.mul(core_ast_to_qdag(node.left, arena), core_ast_to_qdag(node.right, arena))
    if isinstance(node, PowNode):
        return arena.pow(core_ast_to_qdag(node.base, arena), float(node.exponent))
    unary = {
        LogNode: "log",
        ExpNode: "exp",
        SinNode: "sin",
        CosNode: "cos",
        AsinNode: "asin",
        AcosNode: "acos",
        AtanNode: "atan",
        ConjNode: "conj",
        RealNode: "real",
        ImagNode: "imag",
        AbsNode: "abs",
        ArgNode: "arg",
    }
    for cls, op in unary.items():
        if isinstance(node, cls):
            return arena.unary(op, core_ast_to_qdag(node.arg, arena))
    raise TypeError(f"unsupported core AST node {type(node).__name__}")



def _is_close(value: Any, target: float, tol: float = 1.0e-15) -> bool:
    try:
        return bool(abs(value - target) <= tol)
    except Exception:
        return False

def _chain(op: str, children: list[Any]) -> Any:
    from nestynet_sr.sr_core.bridges import Add, ConstNode, Mul

    if not children:
        return ConstNode(0.0 if op == "add" else 1.0)
    out = children[0]
    for child in children[1:]:
        out = Add(out, child) if op == "add" else Mul(out, child)
    return out


def qdag_to_core_ast(arena: QArena, node_id: int):
    from nestynet_sr.sr_core.bridges import (
        Acos,
        Arg,
        Asin,
        Atan,
        Abs,
        Conj,
        ConstNode,
        Cos,
        Exp,
        Imag,
        Log,
        Mul,
        Pow,
        Real,
        Sin,
        Var,
        clone_ast,
    )

    node = arena.nodes[int(node_id)]
    if node.op == "const":
        return ConstNode(node.value)
    if node.op == "var":
        return Var(int(node.value))
    if node.op == "hparam":
        return ConstNode(1.0)
    if node.op == "atom":
        if node.value is None:
            raise ValueError("cannot lower opaque atom without original node")
        return clone_ast(node.value)
    if node.op == "pow":
        base_id, exp = node.value
        return Pow(qdag_to_core_ast(arena, base_id), float(exp))
    if node.op == "unary":
        child = qdag_to_core_ast(arena, node.args[0])
        op = str(node.value)
        table = {
            "log": Log,
            "exp": Exp,
            "sin": Sin,
            "cos": Cos,
            "asin": Asin,
            "acos": Acos,
            "atan": Atan,
            "conj": Conj,
            "real": Real,
            "imag": Imag,
            "abs": Abs,
            "arg": Arg,
            "sqrt": lambda x: Pow(x, 0.5),
        }
        if op not in table:
            raise ValueError(f"unsupported core unary op {op!r}")
        return table[op](child)
    if node.op == "mul":
        const_val, factors = node.value
        pieces = []
        if not _is_close(const_val, 1.0):
            pieces.append(ConstNode(const_val))
        for base_id, exp in factors:
            base = qdag_to_core_ast(arena, base_id)
            pieces.append(base if abs(float(exp) - 1.0) <= 1.0e-12 else Pow(base, float(exp)))
        return _chain("mul", pieces)
    if node.op == "add":
        pieces = []
        for coeff, child_id in node.value:
            child = qdag_to_core_ast(arena, child_id)
            if _is_close(coeff, 1.0):
                pieces.append(child)
            elif _is_close(coeff, -1.0):
                pieces.append(Mul(ConstNode(-1.0), child))
            elif isinstance(child, ConstNode) and _is_close(child.value, 1.0):
                pieces.append(ConstNode(coeff))
            else:
                pieces.append(Mul(ConstNode(coeff), child))
        return _chain("add", pieces)
    if node.op in {"add_raw", "mul_raw"}:
        return _chain(str(node.value), [qdag_to_core_ast(arena, arg) for arg in node.args])
    raise ValueError(f"cannot lower QDAG op {node.op!r} to core AST")


def canonicalize_core_ast(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None):
    c = coerce_expr_ir_config(cfg)
    arena = QArena(c, stats=stats, signature_context=signature_context)
    root = core_ast_to_qdag(node, arena)
    if c.qdag_max_cost is not None and arena.nodes[root].cost > float(c.qdag_max_cost):
        raise RuntimeError("QDAG cost limit exceeded")
    out = qdag_to_core_ast(arena, root)
    if isinstance(stats, ExpressionIRStats):
        stats.canonicalized_candidates += 1
        stats.record_example({"before": repr(node), "after": repr(out), "key": str(arena.key(root))}, limit=int(c.debug_dump_examples))
    return out


def canonical_key_core_ast(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None) -> tuple:
    c = coerce_expr_ir_config(cfg)
    if not expr_ir_active(c):
        return ("legacy", repr(node))
    arena = QArena(c, stats=stats, signature_context=signature_context)
    root = core_ast_to_qdag(node, arena)
    return arena.key(root)


def maybe_canonicalize_core_ast(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None):
    c = coerce_expr_ir_config(cfg)
    if not expr_ir_active(c):
        return node
    try:
        if c.expr_ir == "qdag-egraph" or bool(c.egraph_enable):
            from .egraph_normalizer import normalize_with_egraph_core

            return normalize_with_egraph_core(node, c, stats=stats, signature_context=signature_context)
        return canonicalize_core_ast(node, c, stats=stats, signature_context=signature_context)
    except Exception:
        if isinstance(stats, ExpressionIRStats):
            stats.fallback_count += 1
        if c.strict_errors or not c.fallback_on_error:
            raise
        return node


__all__ = [
    "canonical_key_core_ast",
    "canonicalize_core_ast",
    "core_ast_to_qdag",
    "maybe_canonicalize_core_ast",
    "qdag_to_core_ast",
]
