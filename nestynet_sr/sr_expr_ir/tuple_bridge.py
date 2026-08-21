# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tuple-AST bridge for the opt-in expression IR."""

from __future__ import annotations

from typing import Any

from .config import coerce_expr_ir_config, expr_ir_active
from .qdag import QArena
from .stats import ExpressionIRStats


_UNARY = {"sin", "cos", "exp", "log", "sqrt", "sqr", "neg", "asin", "acos"}


def tuple_ast_to_qdag(node: Any, arena: QArena) -> int:
    if not isinstance(node, (tuple, list)) or not node:
        raise TypeError(f"invalid tuple AST node {node!r}")
    op = str(node[0])
    if op == "var":
        return arena.var(int(node[1]))
    if op == "hparam":
        return arena.hparam(int(node[1]))
    if op == "const":
        return arena.const(node[1])
    if op in _UNARY:
        return arena.unary(op, tuple_ast_to_qdag(node[1], arena))
    if op == "add":
        return arena.add(tuple_ast_to_qdag(node[1], arena), tuple_ast_to_qdag(node[2], arena))
    if op == "sub":
        return arena.sub(tuple_ast_to_qdag(node[1], arena), tuple_ast_to_qdag(node[2], arena))
    if op == "mul":
        return arena.mul(tuple_ast_to_qdag(node[1], arena), tuple_ast_to_qdag(node[2], arena))
    if op == "div":
        return arena.div(tuple_ast_to_qdag(node[1], arena), tuple_ast_to_qdag(node[2], arena))
    raise ValueError(f"unsupported tuple AST op {op!r}")


def _chain(op: str, children: list[Any]) -> Any:
    if not children:
        return ("const", 0.0 if op == "add" else 1.0)
    if len(children) == 1:
        return children[0]
    out = children[0]
    for child in children[1:]:
        out = (op, out, child)
    return out


def _pow_tuple(base: Any, exp: float) -> Any:
    if abs(exp - 1.0) <= 1.0e-12:
        return base
    if abs(exp - 2.0) <= 1.0e-12:
        return ("sqr", base)
    if abs(exp - 0.5) <= 1.0e-12:
        return ("sqrt", base)
    if abs(exp + 1.0) <= 1.0e-12:
        return ("div", ("const", 1.0), base)
    if abs(exp - round(exp)) <= 1.0e-12:
        n = int(round(exp))
        if n > 1:
            return _chain("mul", [base for _ in range(n)])
        if n < -1:
            return ("div", ("const", 1.0), _chain("mul", [base for _ in range(abs(n))]))
    raise ValueError(f"cannot lower tuple power exponent {exp!r}")


def _is_one_const(node: Any) -> bool:
    return isinstance(node, tuple) and len(node) == 2 and node[0] == "const" and abs(float(node[1]) - 1.0) <= 1.0e-15


def _is_minus_one_const(node: Any) -> bool:
    return isinstance(node, tuple) and len(node) == 2 and node[0] == "const" and abs(float(node[1]) + 1.0) <= 1.0e-15


def qdag_to_tuple_ast(arena: QArena, node_id: int):
    node = arena.nodes[int(node_id)]
    if node.op == "const":
        value = node.value
        if isinstance(value, complex):
            raise ValueError("tuple AST does not support complex constants")
        return ("const", float(value))
    if node.op == "var":
        return ("var", int(node.value))
    if node.op == "hparam":
        return ("hparam", int(node.value))
    if node.op == "unary":
        return (str(node.value), qdag_to_tuple_ast(arena, node.args[0]))
    if node.op == "pow":
        base_id, exp = node.value
        return _pow_tuple(qdag_to_tuple_ast(arena, base_id), float(exp))
    if node.op == "mul":
        const_val, factors = node.value
        pieces = []
        if abs(float(const_val) - 1.0) > 1.0e-15:
            pieces.append(("const", float(const_val)))
        for base_id, exp in factors:
            pieces.append(_pow_tuple(qdag_to_tuple_ast(arena, base_id), float(exp)))
        if len(pieces) == 2 and _is_minus_one_const(pieces[0]):
            return ("neg", pieces[1])
        return _chain("mul", pieces)
    if node.op == "add":
        pieces = []
        for coeff, child_id in node.value:
            child = qdag_to_tuple_ast(arena, child_id)
            if abs(float(coeff) - 1.0) <= 1.0e-15:
                pieces.append(child)
            elif abs(float(coeff) + 1.0) <= 1.0e-15:
                pieces.append(("neg", child))
            elif _is_one_const(child):
                pieces.append(("const", float(coeff)))
            else:
                pieces.append(("mul", ("const", float(coeff)), child))
        return _chain("add", pieces)
    if node.op in {"add_raw", "mul_raw"}:
        return _chain(str(node.value), [qdag_to_tuple_ast(arena, arg) for arg in node.args])
    raise ValueError(f"cannot lower QDAG op {node.op!r} to tuple AST")


def _lowered_limits_ok(node: Any, cfg: object | None, stats: ExpressionIRStats | None) -> bool:
    c = coerce_expr_ir_config(cfg)
    try:
        from nestynet_sr.sr_search.factorized_search.expr_ast import node_depth, node_size

        depth = int(node_depth(node))
        size = int(node_size(node))
        if isinstance(stats, ExpressionIRStats):
            stats.lowered_depth_max = max(int(stats.lowered_depth_max), depth)
            stats.lowered_size_max = max(int(stats.lowered_size_max), size)
        if c.max_lowered_depth is not None and depth > int(c.max_lowered_depth):
            return False
        if c.max_lowered_size is not None and size > int(c.max_lowered_size):
            return False
    except Exception:
        return True
    return True


def canonicalize_tuple_ast(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None):
    c = coerce_expr_ir_config(cfg)
    arena = QArena(c, stats=stats, signature_context=signature_context)
    root = tuple_ast_to_qdag(node, arena)
    if c.qdag_max_cost is not None and arena.nodes[root].cost > float(c.qdag_max_cost):
        raise RuntimeError("QDAG cost limit exceeded")
    lowered = qdag_to_tuple_ast(arena, root)
    if not _lowered_limits_ok(lowered, c, stats):
        raise RuntimeError("lowered tuple AST limit exceeded")
    if isinstance(stats, ExpressionIRStats):
        stats.canonicalized_candidates += 1
        try:
            from nestynet_sr.sr_search.factorized_search.expr_ast import node_str

            stats.record_example(
                {"before": node_str(node), "after": node_str(lowered), "key": str(arena.key(root))},
                limit=int(c.debug_dump_examples),
            )
        except Exception:
            pass
    return lowered


def canonical_key_tuple_ast(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None) -> tuple:
    c = coerce_expr_ir_config(cfg)
    if not expr_ir_active(c):
        try:
            from nestynet_sr.sr_search.factorized_search.expr_ast import node_str, simplify

            return ("legacy", node_str(simplify(node)))
        except Exception:
            return ("legacy", repr(node))
    node = canonicalize_tuple_ast(node, c, stats=stats, signature_context=signature_context)
    arena = QArena(c, stats=stats, signature_context=signature_context)
    root = tuple_ast_to_qdag(node, arena)
    return arena.key(root)


def maybe_canonicalize_tuple_ast(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None):
    c = coerce_expr_ir_config(cfg)
    if not expr_ir_active(c):
        return node
    try:
        if c.expr_ir == "qdag-egraph" or bool(c.egraph_enable):
            from .egraph_normalizer import normalize_with_egraph_tuple

            return normalize_with_egraph_tuple(node, c, stats=stats, signature_context=signature_context)
        return canonicalize_tuple_ast(node, c, stats=stats, signature_context=signature_context)
    except Exception:
        if isinstance(stats, ExpressionIRStats):
            stats.fallback_count += 1
        if c.strict_errors or not c.fallback_on_error:
            raise
        return node


def maybe_prepare_candidate_for_scoring(
    node: Any,
    cfg: object | None = None,
    stats: ExpressionIRStats | None = None,
    signature_context: Any = None,
    seen_keys: set[tuple] | None = None,
):
    """Canonicalize and optionally duplicate-filter one tuple candidate."""

    c = coerce_expr_ir_config(cfg)
    if isinstance(stats, ExpressionIRStats):
        stats.raw_candidates_seen += 1
    if not expr_ir_active(c):
        return node, None, ""
    prepared = maybe_canonicalize_tuple_ast(node, c, stats=stats, signature_context=signature_context)
    key = canonical_key_tuple_ast(prepared, c, stats=stats, signature_context=signature_context)
    if seen_keys is not None:
        if key in seen_keys:
            if isinstance(stats, ExpressionIRStats):
                stats.canonical_key_hits += 1
                stats.duplicate_candidates_dropped += 1
                stats.evals_skipped_duplicate += 1
            return None, key, "duplicate"
        seen_keys.add(key)
    return prepared, key, ""


__all__ = [
    "canonical_key_tuple_ast",
    "canonicalize_tuple_ast",
    "maybe_canonicalize_tuple_ast",
    "maybe_prepare_candidate_for_scoring",
    "qdag_to_tuple_ast",
    "tuple_ast_to_qdag",
]
