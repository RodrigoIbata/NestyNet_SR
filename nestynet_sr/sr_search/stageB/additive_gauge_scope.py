# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Transient additive-gauge scope analysis for Stage B.

The AST is kept free of persistent gauge metadata.  This module derives a
sidecar view of the current tree: flattened additive scopes whose NN-bearing
terms share variables in a non-unique decomposition.  Stage B can then treat
leaf-local rewrites inside those scopes as provisional unless the fitted
candidate improves the whole scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

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
    collect_all_atoms,
    effective_arity,
)


@dataclass(frozen=True)
class AdditiveGaugeTerm:
    """One signed term in a flattened additive scope."""

    node: Node
    sign: float
    vars: FrozenSet[int]
    nn_atoms: Tuple[AtomNode, ...]
    analytic_cost: int


@dataclass(frozen=True)
class AdditiveGaugeScope:
    """A flattened additive scope with unresolved NN-sharing pairs."""

    uid: str
    add_node: AddNode
    terms: Tuple[AdditiveGaugeTerm, ...]
    shared_vars: FrozenSet[int]
    unresolved_pairs: Tuple[Tuple[int, int], ...]

    @property
    def unresolved(self) -> bool:
        return bool(self.unresolved_pairs)


@dataclass(frozen=True, order=True)
class AdditiveGaugeGlobalScore:
    """Lexicographic score; smaller means less unresolved additive gauge."""

    total_unresolved_scopes: int
    total_nn_atoms_inside_unresolved_scopes: int
    sum_effective_arity_sq_inside_unresolved_scopes: int
    max_effective_arity_inside_unresolved_scopes: int
    unresolved_overlap_pair_count: int
    analytic_complexity_inside_unresolved_scopes: int


class AdditiveGaugeScopeIndex:
    """Sidecar index for unresolved additive gauges in one AST root."""

    def __init__(self, root: Node):
        self.root = root
        self.scopes: Tuple[AdditiveGaugeScope, ...] = tuple(_discover_scopes(root))
        self.unresolved_scopes: Tuple[AdditiveGaugeScope, ...] = tuple(
            scope for scope in self.scopes if scope.unresolved
        )
        target_map: Dict[int, AdditiveGaugeScope] = {}
        for scope in self.unresolved_scopes:
            for term in scope.terms:
                for atom in term.nn_atoms:
                    target_map.setdefault(id(atom), scope)
        self._target_map = target_map

    def scope_for_target(self, target: Node) -> Optional[AdditiveGaugeScope]:
        return self._target_map.get(id(target))

    def contains_target(self, target: Node) -> bool:
        return id(target) in self._target_map

    def global_score(self) -> AdditiveGaugeGlobalScore:
        return additive_gauge_global_score(self.root, index=self)


def additive_gauge_global_score(root: Node, *, index: Optional[AdditiveGaugeScopeIndex] = None) -> AdditiveGaugeGlobalScore:
    """Compute a global unresolved-gauge score for *root*."""
    idx = index if index is not None else AdditiveGaugeScopeIndex(root)
    scopes = idx.unresolved_scopes
    nn_ids = set()
    sum_arity_sq = 0
    max_arity = 0
    analytic_complexity = 0
    pair_count = 0

    for scope in scopes:
        pair_count += len(scope.unresolved_pairs)
        analytic_complexity += sum(term.analytic_cost for term in scope.terms)
        for term in scope.terms:
            for atom in term.nn_atoms:
                aid = id(atom)
                if aid in nn_ids:
                    continue
                nn_ids.add(aid)
                arity = int(effective_arity(atom))
                sum_arity_sq += arity * arity
                max_arity = max(max_arity, arity)

    return AdditiveGaugeGlobalScore(
        total_unresolved_scopes=len(scopes),
        total_nn_atoms_inside_unresolved_scopes=len(nn_ids),
        sum_effective_arity_sq_inside_unresolved_scopes=sum_arity_sq,
        max_effective_arity_inside_unresolved_scopes=max_arity,
        unresolved_overlap_pair_count=pair_count,
        analytic_complexity_inside_unresolved_scopes=analytic_complexity,
    )


def _discover_scopes(root: Node) -> Iterable[AdditiveGaugeScope]:
    scopes: List[AdditiveGaugeScope] = []
    counter = 0

    def visit(node: Node, in_add_chain: bool = False) -> None:
        nonlocal counter
        if isinstance(node, AddNode):
            if not in_add_chain:
                terms = tuple(_flatten_add_terms(node))
                scope = _make_scope(node, terms, uid=f"add:{counter}")
                counter += 1
                if len(scope.terms) >= 2:
                    scopes.append(scope)
                for term in terms:
                    visit(term.node, in_add_chain=False)
            else:
                visit(node.left, in_add_chain=True)
                visit(node.right, in_add_chain=True)
            return
        if isinstance(node, MulNode):
            visit(node.left, in_add_chain=False)
            visit(node.right, in_add_chain=False)
        elif isinstance(node, PowNode):
            visit(node.base, in_add_chain=False)
        elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            visit(node.arg, in_add_chain=False)

    visit(root)
    return scopes


def _flatten_add_terms(node: Node, sign: float = 1.0) -> Iterable[AdditiveGaugeTerm]:
    if isinstance(node, AddNode):
        yield from _flatten_add_terms(node.left, sign)
        yield from _flatten_add_terms(node.right, sign)
        return
    yield _make_term(node, sign)


def _make_scope(add_node: AddNode, terms: Tuple[AdditiveGaugeTerm, ...], *, uid: str) -> AdditiveGaugeScope:
    pairs: List[Tuple[int, int]] = []
    shared_all = set()
    for i in range(len(terms)):
        if not terms[i].nn_atoms:
            continue
        for j in range(i + 1, len(terms)):
            if not terms[j].nn_atoms:
                continue
            shared = set(terms[i].vars) & set(terms[j].vars)
            if not shared:
                continue
            private_i = set(terms[i].vars) - set(terms[j].vars)
            private_j = set(terms[j].vars) - set(terms[i].vars)
            if private_i or private_j:
                pairs.append((i, j))
                shared_all.update(shared)
    return AdditiveGaugeScope(
        uid=uid,
        add_node=add_node,
        terms=terms,
        shared_vars=frozenset(shared_all),
        unresolved_pairs=tuple(pairs),
    )


def _make_term(node: Node, sign: float) -> AdditiveGaugeTerm:
    atoms = tuple(
        atom for atom in collect_all_atoms(node)
        if isinstance(atom, AtomNode) and str(atom.kind).lower() == "nn"
    )
    return AdditiveGaugeTerm(
        node=node,
        sign=float(sign),
        vars=frozenset(_node_vars(node)),
        nn_atoms=atoms,
        analytic_cost=_analytic_cost(node),
    )


def _node_vars(node: Node) -> FrozenSet[int]:
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input"):
            return frozenset(int(i) for i in node.var_idxs)
        try:
            return frozenset(int(i) for i in node.raw_var_idxs)
        except Exception:
            return frozenset(int(i) for i in getattr(node, "var_idxs", ()))
    if isinstance(node, (AddNode, MulNode)):
        return _node_vars(node.left) | _node_vars(node.right)
    if isinstance(node, PowNode):
        return _node_vars(node.base)
    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return _node_vars(node.arg)
    if isinstance(node, ConstNode):
        return frozenset()
    return frozenset()


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
