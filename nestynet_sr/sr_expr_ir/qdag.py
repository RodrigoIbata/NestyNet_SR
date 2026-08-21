# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Bounded quotient-DAG canonicalization primitives.

The QDAG is an internal quotient representation.  It never reaches current
evaluators directly; bridges lower it back to ordinary tuple/core AST trees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import ExpressionIRConfig, coerce_expr_ir_config
from .stats import ExpressionIRStats


@dataclass(frozen=True)
class QKey:
    value: tuple


@dataclass
class QNode:
    op: str
    args: tuple[int, ...] = ()
    value: Any = None
    key: tuple = ()
    cost: float = 0.0
    signature: Any = None
    guard: Any = None


def _is_int_like(v: float) -> bool:
    return math.isfinite(float(v)) and abs(float(v) - round(float(v))) <= 1.0e-12


def _clean_float(value: Any) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"non-finite constant {value!r}")
    if v == 0.0:
        return 0.0
    if _is_int_like(v):
        return float(int(round(v)))
    return v


def _const_key(value: Any) -> tuple:
    if isinstance(value, complex):
        re = _clean_float(value.real)
        im = _clean_float(value.imag)
        return ("const", ("complex", re, im))
    return ("const", _clean_float(value))


def _const_value_from_key(key: tuple) -> Any:
    v = key[1]
    if isinstance(v, tuple) and len(v) == 3 and v[0] == "complex":
        return complex(float(v[1]), float(v[2]))
    return float(v)


def _const_mul(a: Any, b: Any) -> Any:
    out = a * b
    if isinstance(out, complex):
        return complex(_clean_float(out.real), _clean_float(out.imag))
    return _clean_float(out)


def _const_add(a: Any, b: Any) -> Any:
    out = a + b
    if isinstance(out, complex):
        return complex(_clean_float(out.real), _clean_float(out.imag))
    return _clean_float(out)


def _const_is_zero(v: Any) -> bool:
    try:
        return abs(v) <= 1.0e-15
    except Exception:
        return False


def _const_is_one(v: Any) -> bool:
    try:
        return abs(v - 1.0) <= 1.0e-15
    except Exception:
        return False


def _const_is_minus_one(v: Any) -> bool:
    try:
        return abs(v + 1.0) <= 1.0e-15
    except Exception:
        return False


class QArena:
    def __init__(
        self,
        cfg: ExpressionIRConfig | object | None = None,
        stats: ExpressionIRStats | None = None,
        signature_context: Any = None,
    ):
        self.cfg = coerce_expr_ir_config(cfg)
        self.stats = stats if isinstance(stats, ExpressionIRStats) else ExpressionIRStats()
        self.signature_context = signature_context
        self.nodes: list[QNode] = []
        self.intern: dict[tuple, int] = {}

    def _intern(self, op: str, key: tuple, args: tuple[int, ...] = (), value: Any = None, cost: float | None = None) -> int:
        if self.cfg.qdag_hash_cons and key in self.intern:
            self.stats.qdag_intern_hits += 1
            return self.intern[key]
        if len(self.nodes) >= int(self.cfg.qdag_max_nodes):
            raise RuntimeError(f"QDAG node limit exceeded: {self.cfg.qdag_max_nodes}")
        node_cost = float(cost) if cost is not None else 1.0 + sum(self.nodes[i].cost for i in args)
        idx = len(self.nodes)
        self.nodes.append(QNode(op=op, args=tuple(args), value=value, key=key, cost=node_cost))
        if self.cfg.qdag_hash_cons:
            self.intern[key] = idx
        self.stats.qdag_nodes_created += 1
        return idx

    def const(self, value: Any) -> int:
        key = _const_key(value)
        val = _const_value_from_key(key)
        return self._intern("const", key, value=val, cost=0.2)

    def var(self, idx: int) -> int:
        return self._intern("var", ("var", int(idx)), value=int(idx), cost=0.0)

    def hparam(self, idx: int) -> int:
        return self._intern("hparam", ("hparam", int(idx)), value=int(idx), cost=0.25)

    def atom(self, payload: tuple, original: Any = None) -> int:
        return self._intern("atom", ("atom", payload), value=original, cost=1.0)

    def key(self, node_id: int) -> tuple:
        return self.nodes[int(node_id)].key

    def node_size(self, node_id: int) -> int:
        seen: set[int] = set()

        def rec(i: int) -> int:
            i = int(i)
            if i in seen:
                return 0
            seen.add(i)
            return 1 + sum(rec(j) for j in self.nodes[i].args)

        return rec(int(node_id))

    def add(self, *children: int) -> int:
        if not children:
            return self.const(0.0)
        if self.cfg.canonicalize not in {"safe", "common-domain", "aggressive"}:
            return self._binary_chain("add_raw", "add", tuple(children))

        terms: dict[tuple, tuple[Any, int]] = {}

        def add_term(child_id: int, coeff: Any = 1.0) -> None:
            node = self.nodes[int(child_id)]
            if self.cfg.qdag_flatten_ac and node.op == "add":
                self.stats.add_flattened += 1
                for c, cid in tuple(node.value or ()):
                    add_term(int(cid), _const_mul(coeff, c))
                return
            coeff2, term_id = self._split_add_coeff(int(child_id))
            coeff = _const_mul(coeff, coeff2)
            if _const_is_zero(coeff):
                return
            key = self.key(term_id)
            old = terms.get(key)
            if old is None:
                terms[key] = (coeff, term_id)
                return
            new_coeff = _const_add(old[0], coeff)
            if _const_is_zero(new_coeff):
                terms.pop(key, None)
            else:
                terms[key] = (new_coeff, old[1])
                self.stats.like_terms_combined += 1

        for child in children:
            add_term(int(child), 1.0)

        rows = tuple((coeff, cid) for _key, (coeff, cid) in sorted(terms.items(), key=lambda item: item[0]))
        if not rows:
            return self.const(0.0)
        if len(rows) == 1:
            coeff, cid = rows[0]
            if _const_is_one(coeff):
                return int(cid)
            return self.mul(self.const(coeff), int(cid))
        if len(rows) > int(self.cfg.qdag_max_terms_per_add):
            raise RuntimeError("QDAG add term limit exceeded")
        key = ("add", tuple((_const_key(coeff), self.key(cid)) for coeff, cid in rows))
        return self._intern("add", key, args=tuple(int(cid) for _, cid in rows), value=rows)

    def sub(self, a: int, b: int) -> int:
        return self.add(int(a), self.mul(self.const(-1.0), int(b)))

    def mul(self, *children: int) -> int:
        if not children:
            return self.const(1.0)
        if self.cfg.canonicalize not in {"safe", "common-domain", "aggressive"}:
            return self._binary_chain("mul_raw", "mul", tuple(children))

        const_val: Any = 1.0
        factors: list[tuple[int, float]] = []

        def add_factor(child_id: int, exp: float = 1.0) -> None:
            nonlocal const_val
            node = self.nodes[int(child_id)]
            if node.op == "const":
                const_val = _const_mul(const_val, _const_value_from_key(node.key) ** exp)
                self.stats.constants_folded += 1
                return
            if self.cfg.qdag_flatten_ac and node.op == "mul":
                self.stats.mul_flattened += 1
                c, rows = node.value
                const_val = _const_mul(const_val, c ** exp)
                for base_id, e in rows:
                    add_factor(int(base_id), float(e) * float(exp))
                return
            if self.cfg.qdag_combine_powers and node.op == "pow":
                base_id, e = node.value
                add_factor(int(base_id), float(e) * float(exp))
                return
            factors.append((int(child_id), float(exp)))

        for child in children:
            add_factor(int(child), 1.0)

        if _const_is_zero(const_val) and not (self.cfg.domain_mode == "strict" and factors):
            return self.const(0.0)

        factors = self._combine_factors_without_unsafe_cancellation(factors)
        if not factors:
            return self.const(const_val)
        if len(factors) > int(self.cfg.qdag_max_factors_per_mul):
            raise RuntimeError("QDAG mul factor limit exceeded")
        if _const_is_one(const_val) and len(factors) == 1 and abs(factors[0][1] - 1.0) <= 1.0e-12:
            return int(factors[0][0])
        key = (
            "mul",
            _const_key(const_val),
            tuple((self.key(base_id), float(exp)) for base_id, exp in factors),
        )
        return self._intern(
            "mul",
            key,
            args=tuple(int(base_id) for base_id, _ in factors),
            value=(const_val, tuple(factors)),
        )

    def div(self, a: int, b: int) -> int:
        return self.mul(int(a), self.pow(int(b), -1.0))

    def pow(self, base: int, exponent: int | float) -> int:
        exp = _clean_float(exponent)
        if abs(exp - 1.0) <= 1.0e-12:
            return int(base)
        if abs(exp) <= 1.0e-12 and self.cfg.domain_mode != "strict":
            return self.const(1.0)
        node = self.nodes[int(base)]
        if self.cfg.qdag_combine_powers and node.op == "pow" and _is_int_like(exp) and _is_int_like(node.value[1]):
            inner, inner_exp = node.value
            return self.pow(int(inner), float(inner_exp) * float(exp))
        key = ("pow", self.key(int(base)), float(exp))
        return self._intern("pow", key, args=(int(base),), value=(int(base), float(exp)))

    def unary(self, op: str, child: int) -> int:
        op = str(op)
        child = int(child)
        if op == "neg":
            return self.mul(self.const(-1.0), child)
        if op == "sqr":
            return self.pow(child, 2.0)
        if op == "sqrt":
            return self._unary_with_rules("sqrt", child)
        if op in {"sin", "cos"}:
            neg_child = self._negative_child(child)
            if neg_child is not None:
                if op == "sin":
                    return self.mul(self.const(-1.0), self.unary("sin", neg_child))
                return self.unary("cos", neg_child)
        if op == "exp" and self.cfg.domain_mode != "strict":
            n = self.nodes[child]
            if n.op == "unary" and n.value == "log":
                return n.args[0]
        if op == "log" and self.cfg.domain_mode != "strict":
            n = self.nodes[child]
            if n.op == "unary" and n.value == "exp":
                return n.args[0]
        return self._unary_with_rules(op, child)

    def _unary_with_rules(self, op: str, child: int) -> int:
        node = self.nodes[int(child)]
        if node.op == "const" and self.cfg.qdag_constant_fold:
            value = _const_value_from_key(node.key)
            try:
                folded = None
                if not isinstance(value, complex):
                    if op == "sin":
                        folded = math.sin(float(value))
                    elif op == "cos":
                        folded = math.cos(float(value))
                    elif op == "exp":
                        folded = math.exp(float(value))
                    elif op == "log" and float(value) > 0.0:
                        folded = math.log(float(value))
                    elif op == "sqrt" and float(value) >= 0.0:
                        folded = math.sqrt(float(value))
                    elif op == "asin" and -1.0 <= float(value) <= 1.0:
                        folded = math.asin(float(value))
                    elif op == "acos" and -1.0 <= float(value) <= 1.0:
                        folded = math.acos(float(value))
                if folded is not None and math.isfinite(float(folded)):
                    self.stats.constants_folded += 1
                    return self.const(folded)
            except Exception:
                pass
        key = ("unary", str(op), self.key(int(child)))
        return self._intern("unary", key, args=(int(child),), value=str(op))

    def _negative_child(self, child: int) -> int | None:
        node = self.nodes[int(child)]
        if node.op != "mul":
            return None
        const_val, factors = node.value
        if not _const_is_minus_one(const_val):
            return None
        if len(factors) == 1 and abs(float(factors[0][1]) - 1.0) <= 1.0e-12:
            return int(factors[0][0])
        return self._mul_from_factor_rows(1.0, tuple(factors))

    def _split_add_coeff(self, node_id: int) -> tuple[Any, int]:
        node = self.nodes[int(node_id)]
        if node.op == "const":
            return _const_value_from_key(node.key), self.const(1.0)
        if node.op == "mul":
            const_val, factors = node.value
            if not _const_is_one(const_val):
                return const_val, self._mul_from_factor_rows(1.0, tuple(factors))
        return 1.0, int(node_id)

    def _combine_factors_without_unsafe_cancellation(self, factors: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not self.cfg.qdag_combine_powers:
            return sorted(factors, key=lambda row: (self.key(row[0]), row[1]))
        by_bucket: dict[tuple, tuple[int, float]] = {}
        strict = self.cfg.domain_mode == "strict"
        for idx, (base_id, exp) in enumerate(factors):
            if abs(float(exp)) <= 1.0e-12:
                continue
            sign_bucket = "pos" if exp > 0 else "neg"
            is_integer_exp = abs(float(exp) - round(float(exp))) <= 1.0e-12
            if strict and not is_integer_exp:
                bucket = (self.key(base_id), sign_bucket, int(idx))
            else:
                bucket = (self.key(base_id), sign_bucket if strict else "any")
            old = by_bucket.get(bucket)
            if old is None:
                by_bucket[bucket] = (int(base_id), float(exp))
            else:
                new_exp = float(old[1]) + float(exp)
                if abs(new_exp) <= 1.0e-12 and not strict:
                    by_bucket.pop(bucket, None)
                else:
                    by_bucket[bucket] = (old[0], new_exp)
                    self.stats.powers_combined += 1
        return sorted(by_bucket.values(), key=lambda row: (self.key(row[0]), row[1]))

    def _mul_from_factor_rows(self, const_val: Any, rows: tuple[tuple[int, float], ...]) -> int:
        children: list[int] = []
        if not _const_is_one(const_val):
            children.append(self.const(const_val))
        for base_id, exp in rows:
            if abs(float(exp) - 1.0) <= 1.0e-12:
                children.append(int(base_id))
            else:
                children.append(self.pow(int(base_id), float(exp)))
        return self.mul(*children) if children else self.const(1.0)

    def _binary_chain(self, raw_op: str, display_op: str, children: tuple[int, ...]) -> int:
        if len(children) == 1:
            return int(children[0])
        left = int(children[0])
        for right in children[1:]:
            key = (display_op, self.key(left), self.key(int(right)))
            left = self._intern(raw_op, key, args=(left, int(right)), value=display_op)
        return left


__all__ = ["QArena", "QKey", "QNode"]
