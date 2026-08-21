# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Conservative structural simplification for the main NestyNet AST.

This module is intentionally small: it provides deterministic canonical keys
and guarded local rewrites for the dataclass AST in :mod:`sr_core.bridges`.
It is not a CAS, and it avoids domain-changing rewrites in strict mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any, Iterable

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
    Node,
    PowNode,
    RealNode,
    SinNode,
    clone_ast,
    format_const_value,
    get_input_exprs,
    has_nontrivial_input,
)


@dataclass(frozen=True)
class SimplifyOptions:
    enabled: bool = False
    level: str = "safe"
    domain_policy: str = "strict"
    context: str = "generic"
    max_passes: int = 12
    canonicalize_ac: bool = True
    combine_monomials: bool = True
    combine_like_terms: bool = False
    fold_constants: bool = True
    trig_identities: bool = False
    validate_numeric: bool = False
    validation_samples: int = 256
    trace: bool = False
    fail_closed: bool = True


@dataclass
class SimplifyStats:
    enabled: bool
    changed: bool = False
    before_nodes: int = 0
    after_nodes: int = 0
    before_cost: float | None = None
    after_cost: float | None = None
    passes: int = 0
    rules_fired: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def fire(self, rule: str) -> None:
        self.rules_fired[str(rule)] = int(self.rules_fired.get(str(rule), 0)) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "changed": bool(self.changed),
            "before_nodes": int(self.before_nodes),
            "after_nodes": int(self.after_nodes),
            "before_cost": self.before_cost,
            "after_cost": self.after_cost,
            "passes": int(self.passes),
            "rules_fired": dict(self.rules_fired),
            "warnings": list(self.warnings),
        }


_UNARY_TYPES = (
    LogNode,
    ExpNode,
    SinNode,
    CosNode,
    AsinNode,
    AcosNode,
    AtanNode,
    ConjNode,
    RealNode,
    ImagNode,
    AbsNode,
    ArgNode,
)

_UNARY_NAMES = {
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


def node_count(node: Node) -> int:
    if isinstance(node, AtomNode):
        n = 1
        if has_nontrivial_input(node):
            for inp in get_input_exprs(node):
                n += node_count(inp)
        return n
    if isinstance(node, ConstNode):
        return 1
    if isinstance(node, (AddNode, MulNode)):
        return 1 + node_count(node.left) + node_count(node.right)
    if isinstance(node, PowNode):
        return 1 + node_count(node.base)
    if isinstance(node, _UNARY_TYPES):
        return 1 + node_count(node.arg)
    return 1


def _const_key(value: float | complex) -> tuple:
    if isinstance(value, complex):
        return ("complex", format_const_value(value), float(value.real), float(value.imag))
    return ("real", format_const_value(float(value)), float(value))


def _jsonish_key(value: Any, *, ignore_tags: bool, context: str) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return _const_key(value)
    if isinstance(value, ConstNode):
        return stable_ast_key(value, ignore_tags=ignore_tags, context=context)
    if isinstance(value, (AtomNode, AddNode, MulNode, PowNode) + _UNARY_TYPES):
        return stable_ast_key(value, ignore_tags=ignore_tags, context=context)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _jsonish_key(v, ignore_tags=ignore_tags, context=context)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_jsonish_key(v, ignore_tags=ignore_tags, context=context) for v in value)
    return repr(value)


def _normal_exponent(exp: Any) -> Any:
    if isinstance(exp, (int, float)):
        f = float(exp)
        if math.isfinite(f):
            for v in (-3, -2, -1, 0, 0.5, 1, 2, 3, 4):
                if abs(f - float(v)) <= 1.0e-12:
                    return int(v) if float(v).is_integer() else float(v)
        return f
    return repr(exp)


def stable_ast_key(node: Node, *, ignore_tags: bool = False, context: str = "generic") -> tuple:
    """Return a deterministic structural key for the main AST."""

    if isinstance(node, ConstNode):
        return ("const", _const_key(node.value))
    if isinstance(node, AtomNode):
        kwargs = dict(getattr(node, "kwargs", {}) or {})
        tag = None if ignore_tags else getattr(node, "tag", None)
        kind = str(getattr(node, "kind", ""))
        if kind.lower() in ("var", "x", "input"):
            inputs = ()
        else:
            inputs = tuple(stable_ast_key(inp, ignore_tags=ignore_tags, context=context) for inp in get_input_exprs(node))
        return (
            "atom",
            kind,
            tuple(int(i) for i in getattr(node, "var_idxs", ()) or ()),
            str(getattr(node, "scope", "")),
            tag,
            _jsonish_key(kwargs, ignore_tags=ignore_tags, context=context),
            inputs,
        )
    if isinstance(node, AddNode):
        return ("add", stable_ast_key(node.left, ignore_tags=ignore_tags, context=context), stable_ast_key(node.right, ignore_tags=ignore_tags, context=context))
    if isinstance(node, MulNode):
        return ("mul", stable_ast_key(node.left, ignore_tags=ignore_tags, context=context), stable_ast_key(node.right, ignore_tags=ignore_tags, context=context))
    if isinstance(node, PowNode):
        return ("pow", stable_ast_key(node.base, ignore_tags=ignore_tags, context=context), _normal_exponent(node.exponent))
    if isinstance(node, _UNARY_TYPES):
        return (_UNARY_NAMES.get(type(node), type(node).__name__), stable_ast_key(node.arg, ignore_tags=ignore_tags, context=context))
    return (type(node).__name__, repr(node))


def _is_const(node: Node, value: float | complex | None = None) -> bool:
    if not isinstance(node, ConstNode):
        return False
    if value is None:
        return True
    return node.value == value


def _is_zero(node: Node) -> bool:
    return isinstance(node, ConstNode) and node.value == 0


def _is_one(node: Node) -> bool:
    return isinstance(node, ConstNode) and node.value == 1


def _finite_const(value: Any) -> bool:
    try:
        if isinstance(value, complex):
            return math.isfinite(float(value.real)) and math.isfinite(float(value.imag))
        return math.isfinite(float(value))
    except Exception:
        return False


def _clone_atom(atom: AtomNode, inputs: tuple[Node, ...] | None = None) -> AtomNode:
    if inputs is None:
        inputs = tuple(clone_ast(inp) for inp in get_input_exprs(atom)) if has_nontrivial_input(atom) else None
    return AtomNode(
        kind=str(atom.kind),
        var_idxs=tuple(int(i) for i in atom.var_idxs),
        kwargs=dict(atom.kwargs or {}),
        tag=atom.tag,
        inputs=inputs,
        scope=str(getattr(atom, "scope", "experiment")),
    )


def _clone_with_simplified_children(node: Node, opts: SimplifyOptions, stats: SimplifyStats) -> Node:
    if isinstance(node, AtomNode):
        if has_nontrivial_input(node):
            return _clone_atom(node, tuple(_simplify_once(inp, opts, stats) for inp in get_input_exprs(node)))
        return _clone_atom(node, None)
    if isinstance(node, ConstNode):
        return ConstNode(node.value)
    if isinstance(node, AddNode):
        return AddNode(_simplify_once(node.left, opts, stats), _simplify_once(node.right, opts, stats))
    if isinstance(node, MulNode):
        return MulNode(_simplify_once(node.left, opts, stats), _simplify_once(node.right, opts, stats))
    if isinstance(node, PowNode):
        base = _simplify_once(node.base, opts, stats)
        exp = _normal_exponent(node.exponent)
        if opts.fold_constants and exp != node.exponent:
            stats.fire("pow_exponent_canonical")
        if exp == 1:
            stats.fire("pow_one")
            return base
        if isinstance(base, ConstNode) and opts.fold_constants:
            try:
                v = base.value ** exp
                if _finite_const(v):
                    stats.fire("const_pow")
                    return ConstNode(v)
            except Exception:
                pass
        return PowNode(base, exp)
    if isinstance(node, _UNARY_TYPES):
        arg = _simplify_once(node.arg, opts, stats)
        return type(node)(arg)
    return clone_ast(node)


def _flatten_binary(node: Node, cls: type) -> list[Node]:
    if isinstance(node, cls):
        return _flatten_binary(node.left, cls) + _flatten_binary(node.right, cls)
    return [node]


def _rebuild_binary(cls: type, children: Iterable[Node]) -> Node:
    vals = list(children)
    if not vals:
        raise ValueError("Cannot rebuild binary AST with no children")
    root = vals[0]
    for child in vals[1:]:
        root = cls(root, child)
    return root


def _const_add(a: float | complex, b: float | complex) -> float | complex:
    if isinstance(a, complex) or isinstance(b, complex):
        return complex(a) + complex(b)
    return float(a) + float(b)


def _const_mul(a: float | complex, b: float | complex) -> float | complex:
    if isinstance(a, complex) or isinstance(b, complex):
        return complex(a) * complex(b)
    return float(a) * float(b)


def _fraction_from_exp(exp: Any) -> Fraction | None:
    exp = _normal_exponent(exp)
    if isinstance(exp, int):
        return Fraction(exp, 1)
    if isinstance(exp, float) and math.isfinite(exp):
        for denom in (1, 2, 3, 4, 6, 8):
            f = Fraction(round(exp * denom), denom)
            if abs(float(f) - exp) <= 1.0e-12:
                return f
    return None


def _split_factor(node: Node) -> tuple[tuple, Node, Fraction] | None:
    if isinstance(node, PowNode):
        frac = _fraction_from_exp(node.exponent)
        if frac is None:
            return None
        return stable_ast_key(node.base, ignore_tags=False), node.base, frac
    return stable_ast_key(node, ignore_tags=False), node, Fraction(1, 1)


def _combine_monomial_factors(factors: list[Node], opts: SimplifyOptions, stats: SimplifyStats) -> list[Node]:
    if not opts.combine_monomials:
        return factors
    grouped: dict[tuple, tuple[Node, Fraction, int]] = {}
    residual: list[Node] = []
    for factor in factors:
        split = _split_factor(factor)
        if split is None:
            residual.append(factor)
            continue
        key, base, exp = split
        if opts.domain_policy == "strict" and exp.denominator != 1:
            residual.append(factor)
            continue
        old = grouped.get(key)
        if old is None:
            grouped[key] = (base, exp, 1)
        else:
            grouped[key] = (old[0], old[1] + exp, old[2] + 1)
    rebuilt: list[Node] = []
    for key in sorted(grouped):
        base, exp, count = grouped[key]
        if exp == 0:
            if opts.domain_policy == "common-domain":
                stats.fire("mul_cancel_common_domain")
                continue
            # Strict mode preserves a visible x*x^-1 style singular witness.
            rebuilt.extend([clone_ast(base), PowNode(clone_ast(base), -1.0)])
            continue
        if exp == 1:
            rebuilt.append(base)
        else:
            value: int | float
            value = int(exp) if exp.denominator == 1 else float(exp)
            rebuilt.append(PowNode(base, value))
        if count > 1:
            stats.fire("mul_collect_powers")
    rebuilt.extend(residual)
    return rebuilt


def _simplify_add(node: AddNode, opts: SimplifyOptions, stats: SimplifyStats) -> Node:
    terms = _flatten_binary(node, AddNode) if opts.canonicalize_ac else [node.left, node.right]
    terms = [_simplify_once(t, opts, stats) for t in terms]
    const_value: float | complex | None = None
    nonconst: list[Node] = []
    for term in terms:
        if isinstance(term, ConstNode) and opts.fold_constants:
            const_value = term.value if const_value is None else _const_add(const_value, term.value)
            continue
        nonconst.append(term)
    if const_value is not None and _finite_const(const_value) and const_value != 0:
        nonconst.append(ConstNode(const_value))
        if len(terms) != len(nonconst):
            stats.fire("add_const_fold")
    elif const_value is not None and const_value == 0:
        stats.fire("add_zero")
    if not nonconst:
        return ConstNode(0)
    if opts.canonicalize_ac:
        nonconst.sort(key=lambda n: stable_ast_key(n, context=opts.context))
        if len(terms) > 1:
            stats.fire("add_flatten_sort")
    if len(nonconst) == 1:
        return nonconst[0]
    return _rebuild_binary(AddNode, nonconst)


def _simplify_mul(node: MulNode, opts: SimplifyOptions, stats: SimplifyStats) -> Node:
    factors = _flatten_binary(node, MulNode) if opts.canonicalize_ac else [node.left, node.right]
    factors = [_simplify_once(f, opts, stats) for f in factors]
    const_value: float | complex | None = None
    nonconst: list[Node] = []
    for factor in factors:
        if isinstance(factor, ConstNode) and opts.fold_constants:
            const_value = factor.value if const_value is None else _const_mul(const_value, factor.value)
            continue
        nonconst.append(factor)
    if const_value is not None:
        if const_value == 0 and not nonconst:
            stats.fire("mul_const_fold")
            return ConstNode(const_value)
        if const_value == 1:
            stats.fire("mul_one")
        elif const_value == 0 and opts.domain_policy == "common-domain":
            stats.fire("mul_zero_common_domain")
            return ConstNode(const_value)
        else:
            nonconst.append(ConstNode(const_value))
            stats.fire("mul_const_fold")
    if opts.combine_monomials:
        nonconst = _combine_monomial_factors(nonconst, opts, stats)
    if not nonconst:
        return ConstNode(1)
    if opts.canonicalize_ac:
        nonconst.sort(key=lambda n: stable_ast_key(n, context=opts.context))
        if len(factors) > 1:
            stats.fire("mul_flatten_sort")
    if len(nonconst) == 1:
        return nonconst[0]
    return _rebuild_binary(MulNode, nonconst)


def _simplify_once(node: Node, opts: SimplifyOptions, stats: SimplifyStats) -> Node:
    if isinstance(node, AddNode):
        return _simplify_add(node, opts, stats)
    if isinstance(node, MulNode):
        return _simplify_mul(node, opts, stats)
    return _clone_with_simplified_children(node, opts, stats)


def _simplify_ast_with_options(
    node: Node,
    options: SimplifyOptions | None = None,
    *,
    units_spec: Any = None,
    expected_dim: Any = None,
    eval_check: Any = None,
) -> tuple[Node, SimplifyStats]:
    """Simplify ``node`` according to guarded local rewrite options."""

    opts = options or SimplifyOptions()
    stats = SimplifyStats(enabled=bool(opts.enabled), before_nodes=node_count(node))
    if not opts.enabled:
        stats.after_nodes = stats.before_nodes
        return node, stats
    try:
        before_key = stable_ast_key(node, context=opts.context)
        current = clone_ast(node)
        max_passes = max(1, int(opts.max_passes))
        for p in range(max_passes):
            old_key = stable_ast_key(current, context=opts.context)
            current = _simplify_once(current, opts, stats)
            stats.passes = p + 1
            new_key = stable_ast_key(current, context=opts.context)
            if new_key == old_key:
                break
        if eval_check is not None:
            try:
                if not bool(eval_check(node, current)):
                    stats.warnings.append("eval_check_rejected")
                    current = clone_ast(node)
            except Exception as exc:
                stats.warnings.append(f"eval_check_failed:{type(exc).__name__}")
                current = clone_ast(node)
        after_key = stable_ast_key(current, context=opts.context)
        stats.changed = after_key != before_key
        stats.after_nodes = node_count(current)
        return current, stats
    except Exception as exc:
        stats.warnings.append(f"simplify_failed:{type(exc).__name__}:{exc}")
        if opts.fail_closed:
            out = clone_ast(node)
            stats.after_nodes = node_count(out)
            return out, stats
        raise


def _legacy_is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _legacy_const_value(node: Any) -> float | None:
    if not isinstance(node, ConstNode):
        return None
    value = getattr(node, "value", None)
    if isinstance(value, complex):
        if abs(float(value.imag)) > 0.0:
            return None
        value = float(value.real)
    if not _legacy_is_real_number(value):
        return None
    return float(value)


def _legacy_const_node(value: float, *, snap_tol: float) -> ConstNode:
    value_f = float(value)
    for target in (-1.0, 0.0, 1.0):
        if abs(value_f - target) <= float(snap_tol):
            value_f = float(target)
            break
    return ConstNode(value_f)


def _legacy_is_near(value: float, target: float, *, tol: float) -> bool:
    return abs(float(value) - float(target)) <= float(tol)


def _legacy_flatten_add(node: Any) -> list[Any]:
    if isinstance(node, AddNode):
        return [*_legacy_flatten_add(node.left), *_legacy_flatten_add(node.right)]
    return [node]


def _legacy_flatten_mul(node: Any) -> list[Any]:
    if isinstance(node, MulNode):
        return [*_legacy_flatten_mul(node.left), *_legacy_flatten_mul(node.right)]
    return [node]


def _legacy_rebuild_add(terms: Iterable[Any], *, snap_tol: float) -> Any:
    out: Any | None = None
    const_total = 0.0
    kept: list[Any] = []
    for term in terms:
        c = _legacy_const_value(term)
        if c is None:
            kept.append(term)
        else:
            const_total += float(c)
    kept.sort(key=repr)
    if not _legacy_is_near(const_total, 0.0, tol=snap_tol):
        kept.append(_legacy_const_node(const_total, snap_tol=snap_tol))
    for term in kept:
        out = term if out is None else AddNode(out, term)
    return _legacy_const_node(0.0, snap_tol=snap_tol) if out is None else out


def _legacy_rebuild_mul(factors: Iterable[Any], *, snap_tol: float) -> Any:
    const_prod = 1.0
    kept: list[Any] = []
    for factor in factors:
        c = _legacy_const_value(factor)
        if c is None:
            kept.append(factor)
        else:
            const_prod *= float(c)
    if _legacy_is_near(const_prod, 0.0, tol=snap_tol):
        return _legacy_const_node(0.0, snap_tol=snap_tol)
    if _legacy_is_near(const_prod, 1.0, tol=snap_tol) and kept:
        const_prod = 1.0
    elif _legacy_is_near(const_prod, -1.0, tol=snap_tol):
        const_prod = -1.0
    kept.sort(key=repr)
    factors_out: list[Any] = []
    if (not kept) or (not _legacy_is_near(const_prod, 1.0, tol=snap_tol)):
        factors_out.append(_legacy_const_node(const_prod, snap_tol=snap_tol))
    factors_out.extend(kept)
    out: Any | None = None
    for factor in factors_out:
        out = factor if out is None else MulNode(out, factor)
    return _legacy_const_node(1.0, snap_tol=snap_tol) if out is None else out


def _legacy_mul_term_count(node: Any) -> int:
    if isinstance(node, AddNode):
        return _legacy_mul_term_count(node.left) + _legacy_mul_term_count(node.right)
    return 1


def _legacy_distribute_mul_once(factors: list[Any], *, max_terms: int, snap_tol: float) -> Any | None:
    for i, factor in enumerate(factors):
        if not isinstance(factor, AddNode):
            continue
        left_factors = [*factors[:i], factor.left, *factors[i + 1 :]]
        right_factors = [*factors[:i], factor.right, *factors[i + 1 :]]
        if _legacy_mul_term_count(factor.left) + _legacy_mul_term_count(factor.right) > int(max_terms):
            return None
        left = _legacy_simplify_node(
            _legacy_rebuild_mul(left_factors, snap_tol=snap_tol),
            snap_tol=snap_tol,
            max_terms=max_terms,
        )
        right = _legacy_simplify_node(
            _legacy_rebuild_mul(right_factors, snap_tol=snap_tol),
            snap_tol=snap_tol,
            max_terms=max_terms,
        )
        return _legacy_simplify_add(AddNode(left, right), snap_tol=snap_tol, max_terms=max_terms)
    return None


def _legacy_split_coeff(term: Any, *, snap_tol: float) -> tuple[float, Any | None]:
    c = _legacy_const_value(term)
    if c is not None:
        return float(c), None
    coeff = 1.0
    rest: list[Any] = []
    for factor in _legacy_flatten_mul(term):
        c = _legacy_const_value(factor)
        if c is None:
            rest.append(factor)
        else:
            coeff *= float(c)
    if not rest:
        return float(coeff), None
    base = _legacy_rebuild_mul(rest, snap_tol=snap_tol)
    return float(coeff), base


def _legacy_scaled_term(coeff: float, base: Any | None, *, snap_tol: float) -> Any:
    if base is None:
        return _legacy_const_node(coeff, snap_tol=snap_tol)
    coeff_f = float(coeff)
    if _legacy_is_near(coeff_f, 1.0, tol=snap_tol):
        return base
    if _legacy_is_near(coeff_f, -1.0, tol=snap_tol):
        return MulNode(_legacy_const_node(-1.0, snap_tol=snap_tol), base)
    return MulNode(_legacy_const_node(coeff_f, snap_tol=snap_tol), base)


def _legacy_simplify_add(node: AddNode, *, snap_tol: float, max_terms: int) -> Any:
    buckets: dict[str, tuple[float, Any | None]] = {}
    const_total = 0.0
    for raw_term in _legacy_flatten_add(node):
        term = _legacy_simplify_node(raw_term, snap_tol=snap_tol, max_terms=max_terms)
        coeff, base = _legacy_split_coeff(term, snap_tol=snap_tol)
        if _legacy_is_near(coeff, 0.0, tol=snap_tol):
            continue
        if base is None:
            const_total += float(coeff)
            continue
        key = repr(base)
        prev_coeff, prev_base = buckets.get(key, (0.0, base))
        buckets[key] = (float(prev_coeff) + float(coeff), prev_base)

    terms: list[Any] = []
    for key in sorted(buckets):
        coeff, base = buckets[key]
        if _legacy_is_near(coeff, 0.0, tol=snap_tol):
            continue
        terms.append(_legacy_scaled_term(coeff, base, snap_tol=snap_tol))
    if not _legacy_is_near(const_total, 0.0, tol=snap_tol):
        terms.append(_legacy_const_node(const_total, snap_tol=snap_tol))
    return _legacy_rebuild_add(terms, snap_tol=snap_tol)


def _legacy_simplify_mul(node: MulNode, *, snap_tol: float, max_terms: int) -> Any:
    factors = [_legacy_simplify_node(f, snap_tol=snap_tol, max_terms=max_terms) for f in _legacy_flatten_mul(node)]
    if any(isinstance(f, AddNode) for f in factors):
        expanded = _legacy_distribute_mul_once(factors, max_terms=max_terms, snap_tol=snap_tol)
        if expanded is not None:
            return expanded
    return _legacy_rebuild_mul(factors, snap_tol=snap_tol)


def _legacy_simplify_pow(node: PowNode, *, snap_tol: float, max_terms: int) -> Any:
    base = _legacy_simplify_node(node.base, snap_tol=snap_tol, max_terms=max_terms)
    exponent = float(node.exponent)
    if _legacy_is_near(exponent, 0.0, tol=snap_tol):
        return _legacy_const_node(1.0, snap_tol=snap_tol)
    if _legacy_is_near(exponent, 1.0, tol=snap_tol):
        return base
    c = _legacy_const_value(base)
    if c is not None:
        try:
            value = float(c) ** exponent
            if math.isfinite(value):
                return _legacy_const_node(value, snap_tol=snap_tol)
        except Exception:
            pass
    if isinstance(base, PowNode):
        try:
            return _legacy_simplify_pow(PowNode(base.base, float(base.exponent) * exponent), snap_tol=snap_tol, max_terms=max_terms)
        except Exception:
            pass
    return PowNode(base, exponent)


def _legacy_simplify_unary(node: Any, *, snap_tol: float, max_terms: int) -> Any:
    arg = _legacy_simplify_node(node.arg, snap_tol=snap_tol, max_terms=max_terms)
    cls = type(node)
    c = _legacy_const_value(arg)
    if c is not None:
        try:
            value: float | None = None
            if cls is ExpNode:
                value = math.exp(float(c))
            elif cls is SinNode:
                value = math.sin(float(c))
            elif cls is CosNode:
                value = math.cos(float(c))
            elif cls is LogNode and float(c) > 0.0:
                value = math.log(float(c))
            if value is not None and math.isfinite(value):
                return _legacy_const_node(value, snap_tol=snap_tol)
        except Exception:
            pass
    return cls(arg)


def _legacy_simplify_node(node: Any, *, snap_tol: float, max_terms: int) -> Any:
    if isinstance(node, ConstNode):
        c = _legacy_const_value(node)
        return clone_ast(node) if c is None else _legacy_const_node(c, snap_tol=snap_tol)
    if isinstance(node, AtomNode):
        return clone_ast(node)
    if isinstance(node, AddNode):
        return _legacy_simplify_add(node, snap_tol=snap_tol, max_terms=max_terms)
    if isinstance(node, MulNode):
        return _legacy_simplify_mul(node, snap_tol=snap_tol, max_terms=max_terms)
    if isinstance(node, PowNode):
        return _legacy_simplify_pow(node, snap_tol=snap_tol, max_terms=max_terms)
    if isinstance(node, _UNARY_TYPES):
        return _legacy_simplify_unary(node, snap_tol=snap_tol, max_terms=max_terms)
    return clone_ast(node)


def _legacy_simplify_ast(node: Any, *, coeff_tol: float = 1.0e-10, snap_tol: float = 1.0e-8, max_terms: int = 16) -> Any:
    tol = max(float(coeff_tol), float(snap_tol))
    out = _legacy_simplify_node(node, snap_tol=tol, max_terms=int(max_terms))
    return _legacy_simplify_node(out, snap_tol=tol, max_terms=int(max_terms))


def simplify_ast(
    node: Node,
    options: SimplifyOptions | None = None,
    *,
    units_spec: Any = None,
    expected_dim: Any = None,
    eval_check: Any = None,
    coeff_tol: float = 1.0e-10,
    snap_tol: float = 1.0e-8,
    max_terms: int = 16,
) -> Any:
    """Simplify an AST.

    Without ``options`` this preserves the current mainline API and returns a
    simplified node.  With ``SimplifyOptions`` it uses the GS canonicalization
    API and returns ``(node, SimplifyStats)``.
    """

    if isinstance(options, SimplifyOptions):
        return _simplify_ast_with_options(
            node,
            options,
            units_spec=units_spec,
            expected_dim=expected_dim,
            eval_check=eval_check,
        )
    return _legacy_simplify_ast(node, coeff_tol=coeff_tol, snap_tol=snap_tol, max_terms=max_terms)


ast_node_count = node_count


__all__ = [
    "SimplifyOptions",
    "SimplifyStats",
    "ast_node_count",
    "node_count",
    "simplify_ast",
    "stable_ast_key",
]
