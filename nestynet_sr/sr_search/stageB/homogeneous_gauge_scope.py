# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Transient homogeneous-gauge scope analysis for Stage B.

This sidecar handles multiplicative representative ambiguity such as

    x_i**k * NN(x_j/x_i)  <->  x_j**k * NN(x_i/x_j)

The AST remains the source of truth.  The index only identifies current
ratio-leaf/product scopes where a leaf-local univariate rewrite may be
committing to one homogeneous representative before the reciprocal
representative has had a chance to close analytically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AcosNode,
    AddNode,
    ArgNode,
    AsinNode,
    AtanNode,
    AtomNode,
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
    Var,
    ast_to_human_readable,
    clone_ast,
    effective_arity,
    get_input_exprs,
    is_trivial_input,
)


@dataclass(frozen=True)
class RatioMonomial:
    """A parsed monomial ratio with exactly two variables, powers +1 and -1."""

    numerator_var: int
    denominator_var: int
    expr: Node

    @property
    def inverse_expr(self) -> Node:
        return MulNode(Var(self.denominator_var), PowNode(Var(self.numerator_var), -1.0))


@dataclass(frozen=True)
class HomogeneousPowerFactor:
    """A visible monomial power factor in the same product as the ratio NN."""

    node: Node
    var_idx: int
    degree: float
    literal: bool


@dataclass(frozen=True)
class HomogeneousGaugeScope:
    """One product scope containing a ratio NN and a matched monomial factor."""

    uid: str
    mul_node: MulNode
    ratio_factor: Node
    ratio_atom: AtomNode
    ratio_power: float
    ratio: RatioMonomial
    power_factor: HomogeneousPowerFactor
    factors: Tuple[Node, ...]
    other_factors: Tuple[Node, ...]
    analytic_factor_count: int
    analytic_complexity: int

    @property
    def unresolved(self) -> bool:
        return True

    @property
    def alternate_power_var(self) -> int:
        if self.power_factor.var_idx == self.ratio.denominator_var:
            return self.ratio.numerator_var
        return self.ratio.denominator_var

    @property
    def alternate_ratio_expr(self) -> Node:
        return self.ratio.inverse_expr

    @property
    def direction(self) -> str:
        try:
            z = ast_to_human_readable(self.ratio.expr)
        except Exception:
            z = f"x{self.ratio.numerator_var}/x{self.ratio.denominator_var}"
        nn_part = f"NN({z})"
        if not _float_close(self.ratio_power, 1.0):
            nn_part = f"{nn_part}^{self.ratio_power:g}"
        return f"x{self.power_factor.var_idx}^{self.power_factor.degree:g} * {nn_part}"


@dataclass(frozen=True, order=True)
class HomogeneousGaugeGlobalScore:
    """Lexicographic score; smaller means less unresolved homogeneous gauge."""

    total_unresolved_scopes: int
    total_ratio_nn_atoms_inside_unresolved_scopes: int
    sum_effective_arity_sq_inside_unresolved_scopes: int
    max_effective_arity_inside_unresolved_scopes: int
    analytic_factor_count_inside_unresolved_scopes: int
    analytic_complexity_inside_unresolved_scopes: int


class HomogeneousGaugeScopeIndex:
    """Sidecar index for homogeneous ratio/product gauges in one AST root."""

    def __init__(self, root: Node):
        self.root = root
        self.scopes: Tuple[HomogeneousGaugeScope, ...] = tuple(_discover_scopes(root))
        self.unresolved_scopes: Tuple[HomogeneousGaugeScope, ...] = self.scopes
        target_map: Dict[int, HomogeneousGaugeScope] = {}
        for scope in self.unresolved_scopes:
            target_map.setdefault(id(scope.ratio_atom), scope)
        self._target_map = target_map

    def scope_for_target(self, target: Node) -> Optional[HomogeneousGaugeScope]:
        return self._target_map.get(id(target))

    def contains_target(self, target: Node) -> bool:
        return id(target) in self._target_map

    def global_score(self) -> HomogeneousGaugeGlobalScore:
        return homogeneous_gauge_global_score(self.root, index=self)


def homogeneous_gauge_global_score(
    root: Node,
    *,
    index: Optional[HomogeneousGaugeScopeIndex] = None,
) -> HomogeneousGaugeGlobalScore:
    """Compute a global unresolved homogeneous-gauge score for *root*."""
    idx = index if index is not None else HomogeneousGaugeScopeIndex(root)
    scopes = idx.unresolved_scopes
    nn_ids = set()
    sum_arity_sq = 0
    max_arity = 0
    analytic_factor_count = 0
    analytic_complexity = 0

    for scope in scopes:
        analytic_factor_count += int(scope.analytic_factor_count)
        analytic_complexity += int(scope.analytic_complexity)
        atom = scope.ratio_atom
        aid = id(atom)
        if aid in nn_ids:
            continue
        nn_ids.add(aid)
        arity = int(effective_arity(atom))
        sum_arity_sq += arity * arity
        max_arity = max(max_arity, arity)

    return HomogeneousGaugeGlobalScore(
        total_unresolved_scopes=len(scopes),
        total_ratio_nn_atoms_inside_unresolved_scopes=len(nn_ids),
        sum_effective_arity_sq_inside_unresolved_scopes=sum_arity_sq,
        max_effective_arity_inside_unresolved_scopes=max_arity,
        analytic_factor_count_inside_unresolved_scopes=analytic_factor_count,
        analytic_complexity_inside_unresolved_scopes=analytic_complexity,
    )


def parse_ratio_monomial(expr: Node) -> Optional[RatioMonomial]:
    """Return a two-variable +1/-1 ratio if *expr* is monomial-like."""
    powers = _monomial_powers(expr)
    if powers is None or len(powers) != 2:
        return None
    pos = [j for j, p in powers.items() if _float_close(p, 1.0)]
    neg = [j for j, p in powers.items() if _float_close(p, -1.0)]
    if len(pos) != 1 or len(neg) != 1:
        return None
    return RatioMonomial(int(pos[0]), int(neg[0]), clone_ast(expr))


def parse_power_factor(node: Node) -> Optional[HomogeneousPowerFactor]:
    """Return a monomial power factor if *node* is visibly x_i**k."""
    powers = _monomial_powers(node)
    if powers is not None and len(powers) == 1:
        var_idx, degree = next(iter(powers.items()))
        if _valid_degree(degree):
            return HomogeneousPowerFactor(
                node=node,
                var_idx=int(var_idx),
                degree=float(degree),
                literal=True,
            )

    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("rpoly", "rpolynomial", "r_polynomial"):
            var_idx = _single_effective_var(node)
            if var_idx is None:
                return None
            kwargs = getattr(node, "kwargs", {}) or {}
            try:
                degree = float(kwargs.get("degree", 1))
                min_total = float(kwargs.get("min_total", degree))
            except Exception:
                return None
            if _float_close(degree, min_total) and _valid_degree(degree):
                return HomogeneousPowerFactor(
                    node=node,
                    var_idx=int(var_idx),
                    degree=float(degree),
                    literal=False,
                )

    return None


def _discover_scopes(root: Node) -> Iterable[HomogeneousGaugeScope]:
    scopes: List[HomogeneousGaugeScope] = []
    counter = 0

    def visit(node: Node, in_mul_chain: bool = False) -> None:
        nonlocal counter
        if isinstance(node, MulNode):
            if not in_mul_chain:
                factors = tuple(_flatten_mul_factors(node))
                for scope in _make_scopes(node, factors, counter):
                    scopes.append(scope)
                    counter += 1
                for factor in factors:
                    visit(factor, in_mul_chain=False)
            else:
                visit(node.left, in_mul_chain=True)
                visit(node.right, in_mul_chain=True)
            return
        if isinstance(node, AddNode):
            visit(node.left, in_mul_chain=False)
            visit(node.right, in_mul_chain=False)
        elif isinstance(node, PowNode):
            visit(node.base, in_mul_chain=False)
        elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            visit(node.arg, in_mul_chain=False)

    visit(root)
    return scopes


def _make_scopes(
    mul_node: MulNode,
    factors: Tuple[Node, ...],
    counter_start: int,
) -> Iterable[HomogeneousGaugeScope]:
    power_factors = [
        pf for pf in (parse_power_factor(factor) for factor in factors)
        if pf is not None
    ]
    if not power_factors:
        return []

    out: List[HomogeneousGaugeScope] = []
    local_counter = int(counter_start)
    for factor in factors:
        ratio_info = _parse_ratio_nn_factor(factor)
        if ratio_info is None:
            continue
        atom, ratio, ratio_power = ratio_info
        matched = [
            pf for pf in power_factors
            if int(pf.var_idx) in (int(ratio.numerator_var), int(ratio.denominator_var))
            and pf.node is not factor
            and pf.node is not atom
        ]
        if not matched:
            continue
        matched.sort(key=lambda pf: (0 if pf.var_idx == ratio.denominator_var else 1, abs(float(pf.degree) - 1.0)))
        pf = matched[0]
        other_factors = tuple(f for f in factors if f is not factor and f is not pf.node)
        out.append(
            HomogeneousGaugeScope(
                uid=f"hom:{local_counter}",
                mul_node=mul_node,
                ratio_factor=factor,
                ratio_atom=atom,
                ratio_power=float(ratio_power),
                ratio=ratio,
                power_factor=pf,
                factors=factors,
                other_factors=other_factors,
                analytic_factor_count=sum(1 for f in factors if _is_analytic_factor(f)),
                analytic_complexity=sum(_analytic_cost(f) for f in factors),
            )
        )
        local_counter += 1
    return out


def _parse_ratio_nn_factor(factor: Node) -> Optional[Tuple[AtomNode, RatioMonomial, float]]:
    """Return ``(atom, ratio, power)`` for ``NN(ratio)**power`` factors."""
    atom = factor
    power = 1.0
    if isinstance(factor, PowNode):
        atom = factor.base
        try:
            power = float(factor.exponent)
        except Exception:
            return None
        if not _valid_degree(power):
            return None
    if not isinstance(atom, AtomNode):
        return None
    if str(getattr(atom, "kind", "")).lower() != "nn":
        return None
    if int(effective_arity(atom)) != 1:
        return None
    inputs = get_input_exprs(atom)
    if len(inputs) != 1 or is_trivial_input(inputs[0]):
        return None
    ratio = parse_ratio_monomial(inputs[0])
    if ratio is None:
        return None
    return atom, ratio, float(power)


def _flatten_mul_factors(node: Node) -> Iterable[Node]:
    if isinstance(node, MulNode):
        yield from _flatten_mul_factors(node.left)
        yield from _flatten_mul_factors(node.right)
        return
    yield node


def _monomial_powers(node: Node) -> Optional[Dict[int, float]]:
    if isinstance(node, ConstNode):
        return {}
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input") and len(getattr(node, "var_idxs", ()) or ()) == 1:
            return {int(node.var_idxs[0]): 1.0}
        return None
    if isinstance(node, MulNode):
        left = _monomial_powers(node.left)
        right = _monomial_powers(node.right)
        if left is None or right is None:
            return None
        out = dict(left)
        for j, p in right.items():
            out[int(j)] = out.get(int(j), 0.0) + float(p)
        return {j: p for j, p in out.items() if not _float_close(p, 0.0)}
    if isinstance(node, PowNode):
        base = _monomial_powers(node.base)
        if base is None:
            return None
        try:
            exponent = float(node.exponent)
        except Exception:
            return None
        if not math.isfinite(exponent):
            return None
        return {
            int(j): float(p) * exponent
            for j, p in base.items()
            if not _float_close(float(p) * exponent, 0.0)
        }
    return None


def _single_effective_var(node: AtomNode) -> Optional[int]:
    if not isinstance(node, AtomNode):
        return None
    kind = str(getattr(node, "kind", "")).lower()
    if kind in ("var", "x", "input") and len(getattr(node, "var_idxs", ()) or ()) == 1:
        return int(node.var_idxs[0])
    if int(effective_arity(node)) != 1:
        return None
    inputs = get_input_exprs(node)
    if len(inputs) == 1 and is_trivial_input(inputs[0]):
        try:
            return int(inputs[0].var_idxs[0])
        except Exception:
            return None
    return None


def _is_analytic_factor(node: Node) -> bool:
    if isinstance(node, AtomNode):
        return str(getattr(node, "kind", "")).lower() != "nn"
    return True


def _analytic_cost(node: Node) -> int:
    """Small structural cost used only as a gauge-score tie breaker."""
    if isinstance(node, AtomNode):
        return 0 if str(node.kind).lower() == "nn" else 1
    if isinstance(node, ConstNode):
        return 1
    if isinstance(node, (AddNode, MulNode)):
        return 1 + _analytic_cost(node.left) + _analytic_cost(node.right)
    if isinstance(node, PowNode):
        return 1 + _analytic_cost(node.base)
    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return 1 + _analytic_cost(node.arg)
    return 1


def _valid_degree(value: float) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    return math.isfinite(v) and abs(v) > 1.0e-12 and abs(v) <= 8.0


def _float_close(a: float, b: float, *, tol: float = 1.0e-9) -> bool:
    return abs(float(a) - float(b)) <= tol
