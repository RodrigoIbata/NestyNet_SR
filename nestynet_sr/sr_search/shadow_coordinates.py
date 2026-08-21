# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License.

"""Sidecar records for uncommitted coordinate evidence.

Shadow coordinates are deliberately not AST rewrites.  They remember that a
coordinate such as sin(x4) or log(x4/x3) looked promising for a particular NN
leaf, so later rules may build visible analytic candidates from that evidence.
They must not mutate the committed model by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from nestynet_sr.sr_core.bridges import AtomNode, Node, _collect_var_idxs_from_node, ast_to_human_readable, clone_ast
from nestynet_sr.sr_search.feature_grammar import ast_key

try:
    from nestynet_sr.sr_gs.quotient import compose_reduction_plan_with_inputs
except Exception:  # pragma: no cover - keep shadow registry importable during partial builds
    compose_reduction_plan_with_inputs = None  # type: ignore


def shadow_parent_key(atom: AtomNode | None) -> Tuple[str, Any]:
    """Return a stable-enough key for a leaf-local shadow scope."""
    tag = getattr(atom, "tag", None)
    if tag is not None:
        return ("tag", str(tag))
    if atom is None:
        return ("none", None)
    return ("id", id(atom))


@dataclass(frozen=True)
class ShadowCoordinate:
    """Evidence for a candidate coordinate that has not been committed."""

    parent_key: Tuple[str, Any]
    parent_atom_tag: str | None
    base_ast: Node
    shadow_ast: Node
    transform_kind: str
    source: str
    confidence: float
    unit_status: str = "unchecked"
    domain_ok_frac: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shadow_key(self) -> Tuple[Any, ...]:
        return ast_key(self.shadow_ast)

    @property
    def base_key(self) -> Tuple[Any, ...]:
        return ast_key(self.base_ast)


@dataclass(frozen=True)
class ShadowReduction:
    """Audit/shadow record for a whole geometric reduction plan.

    A reduction can contain several invariant coordinates, orbit coordinates,
    output-action metadata, and later normal-form evidence.  It is deliberately
    not a Stage-A proposal and must not mutate the active candidate slate.
    """

    parent_key: Tuple[str, Any]
    parent_atom_tag: str | None
    reduction_plan: Any
    source: str
    confidence: float
    status: str = "shadow"
    raw_var_idxs: Tuple[int, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reduction_key(self) -> Tuple[Any, ...]:
        keys: list[Any] = [self.source, self.status]
        for coord in getattr(self.reduction_plan, "invariant_coordinates", ()) or ():
            ast = getattr(coord, "ast", None)
            keys.append(ast_key(ast) if ast is not None else (getattr(coord, "name", ""), getattr(coord, "kind", "")))
        for coord in getattr(self.reduction_plan, "orbit_coordinates", ()) or ():
            ast = getattr(coord, "ast", None)
            keys.append(("orbit", ast_key(ast) if ast is not None else (getattr(coord, "name", ""), getattr(coord, "kind", ""))))
        return tuple(keys)

    def to_report(self) -> Dict[str, Any]:
        coords = []
        for kind, items in (
            ("invariant", getattr(self.reduction_plan, "invariant_coordinates", ()) or ()),
            ("orbit", getattr(self.reduction_plan, "orbit_coordinates", ()) or ()),
        ):
            for coord in items:
                ast = getattr(coord, "ast", None)
                coords.append(
                    {
                        "role": kind,
                        "name": getattr(coord, "name", ""),
                        "kind": getattr(coord, "kind", ""),
                        "human": ast_to_human_readable(ast) if ast is not None and not isinstance(ast, str) else ast,
                        "raw_support": [int(v) for v in tuple(getattr(coord, "raw_support", ()) or ())],
                        "provenance": dict(getattr(coord, "provenance", {}) or {}),
                    }
                )
        output_action = getattr(self.reduction_plan, "output_action", None)
        normal_form = getattr(self.reduction_plan, "normal_form", None)
        return {
            "parent_key": list(self.parent_key),
            "parent_atom_tag": self.parent_atom_tag,
            "source": self.source,
            "confidence": float(self.confidence),
            "status": self.status,
            "raw_var_idxs": [int(v) for v in self.raw_var_idxs],
            "reduction_status": str(getattr(self.reduction_plan, "status", "")),
            "reduction_reason": str(getattr(self.reduction_plan, "reason", "")),
            "coordinates": coords,
            "output_action": output_action.to_report() if hasattr(output_action, "to_report") else None,
            "normal_form": normal_form.to_report() if hasattr(normal_form, "to_report") else None,
            "provenance": dict(self.provenance or {}),
            "evidence": dict(self.evidence or {}),
        }


def shadow_reduction_from_plan(
    *,
    parent_atom: AtomNode | None,
    reduction_plan: Any,
    local_inputs: Sequence[Node] | None = None,
    source: str = "generalized_symmetry",
    confidence: float = 1.0,
    status: str = "shadow",
    evidence: Mapping[str, Any] | None = None,
) -> ShadowReduction:
    """Create an audit-only shadow reduction, composing local coordinates first."""

    plan = reduction_plan
    if local_inputs is not None:
        if compose_reduction_plan_with_inputs is None:
            raise RuntimeError("compose_reduction_plan_with_inputs is unavailable")
        plan = compose_reduction_plan_with_inputs(reduction_plan, tuple(local_inputs))
    raw_vars: set[int] = set()
    for coord in list(getattr(plan, "invariant_coordinates", ()) or ()) + list(getattr(plan, "orbit_coordinates", ()) or ()):
        raw_vars.update(int(v) for v in tuple(getattr(coord, "raw_support", ()) or ()))
        ast = getattr(coord, "ast", None)
        if ast is not None and not isinstance(ast, str):
            raw_vars.update(int(v) for v in _collect_var_idxs_from_node(ast))
    provenance = dict(getattr(plan, "provenance", {}) or {})
    provenance.update({"shadow_only": True, "active_candidate": False})
    return ShadowReduction(
        parent_key=shadow_parent_key(parent_atom),
        parent_atom_tag=getattr(parent_atom, "tag", None) if parent_atom is not None else None,
        reduction_plan=plan,
        source=str(source),
        confidence=float(confidence),
        status=str(status),
        raw_var_idxs=tuple(sorted(raw_vars)),
        provenance=provenance,
        evidence=dict(evidence or {}),
    )


class ShadowRegistry:
    """Leaf-local shadow store plus a global AST-key index."""

    def __init__(self) -> None:
        self._by_parent: Dict[Tuple[str, Any], Dict[Tuple[Any, ...], ShadowCoordinate]] = {}
        self._global: Dict[Tuple[Any, ...], ShadowCoordinate] = {}
        self._reductions_by_parent: Dict[Tuple[str, Any], Dict[Tuple[Any, ...], ShadowReduction]] = {}

    def add(self, shadow: ShadowCoordinate) -> tuple[ShadowCoordinate, bool]:
        """Insert or merge a shadow.

        Returns ``(stored_shadow, created)``.  Duplicate shadows are merged by
        keeping the higher confidence and combining evidence dictionaries.
        """
        parent_bucket = self._by_parent.setdefault(tuple(shadow.parent_key), {})
        key = tuple(shadow.shadow_key)
        old = parent_bucket.get(key)
        created = old is None
        stored = shadow
        if old is not None:
            evidence = dict(old.evidence or {})
            evidence.update(dict(shadow.evidence or {}))
            if float(shadow.confidence) <= float(old.confidence):
                stored = ShadowCoordinate(
                    parent_key=old.parent_key,
                    parent_atom_tag=old.parent_atom_tag,
                    base_ast=clone_ast(old.base_ast),
                    shadow_ast=clone_ast(old.shadow_ast),
                    transform_kind=old.transform_kind,
                    source=old.source,
                    confidence=float(old.confidence),
                    unit_status=old.unit_status,
                    domain_ok_frac=old.domain_ok_frac,
                    evidence=evidence,
                )
            else:
                stored = ShadowCoordinate(
                    parent_key=shadow.parent_key,
                    parent_atom_tag=shadow.parent_atom_tag,
                    base_ast=clone_ast(shadow.base_ast),
                    shadow_ast=clone_ast(shadow.shadow_ast),
                    transform_kind=shadow.transform_kind,
                    source=shadow.source,
                    confidence=float(shadow.confidence),
                    unit_status=shadow.unit_status,
                    domain_ok_frac=shadow.domain_ok_frac,
                    evidence=evidence,
                )
        parent_bucket[key] = stored

        global_old = self._global.get(key)
        if global_old is None or float(stored.confidence) > float(global_old.confidence):
            self._global[key] = stored
        return stored, created

    def local_for(self, parent_key: Tuple[str, Any]) -> List[ShadowCoordinate]:
        return list(self._by_parent.get(tuple(parent_key), {}).values())

    def global_for_ast(self, expr: Node) -> ShadowCoordinate | None:
        return self._global.get(tuple(ast_key(expr)))

    def all(self) -> List[ShadowCoordinate]:
        out: List[ShadowCoordinate] = []
        for bucket in self._by_parent.values():
            out.extend(bucket.values())
        return out

    def add_reduction(self, shadow: ShadowReduction) -> tuple[ShadowReduction, bool]:
        parent_bucket = self._reductions_by_parent.setdefault(tuple(shadow.parent_key), {})
        key = tuple(shadow.reduction_key)
        old = parent_bucket.get(key)
        created = old is None
        stored = shadow
        if old is not None:
            evidence = dict(old.evidence or {})
            evidence.update(dict(shadow.evidence or {}))
            provenance = dict(old.provenance or {})
            provenance.update(dict(shadow.provenance or {}))
            if float(shadow.confidence) <= float(old.confidence):
                stored = ShadowReduction(
                    parent_key=old.parent_key,
                    parent_atom_tag=old.parent_atom_tag,
                    reduction_plan=old.reduction_plan,
                    source=old.source,
                    confidence=float(old.confidence),
                    status=old.status,
                    raw_var_idxs=old.raw_var_idxs,
                    provenance=provenance,
                    evidence=evidence,
                )
            else:
                stored = ShadowReduction(
                    parent_key=shadow.parent_key,
                    parent_atom_tag=shadow.parent_atom_tag,
                    reduction_plan=shadow.reduction_plan,
                    source=shadow.source,
                    confidence=float(shadow.confidence),
                    status=shadow.status,
                    raw_var_idxs=shadow.raw_var_idxs,
                    provenance=provenance,
                    evidence=evidence,
                )
        parent_bucket[key] = stored
        return stored, created

    def reductions_local_for(self, parent_key: Tuple[str, Any]) -> List[ShadowReduction]:
        return list(self._reductions_by_parent.get(tuple(parent_key), {}).values())

    def reductions_all(self) -> List[ShadowReduction]:
        out: List[ShadowReduction] = []
        for bucket in self._reductions_by_parent.values():
            out.extend(bucket.values())
        return out

    def reduction_count(self) -> int:
        return sum(len(bucket) for bucket in self._reductions_by_parent.values())

    def count(self) -> int:
        return sum(len(bucket) for bucket in self._by_parent.values())

    def _rebuild_global(self) -> None:
        self._global.clear()
        for bucket in self._by_parent.values():
            for shadow in bucket.values():
                old = self._global.get(tuple(shadow.shadow_key))
                if old is None or float(shadow.confidence) > float(old.confidence):
                    self._global[tuple(shadow.shadow_key)] = shadow

    def clear_missing_parent_keys(self, live_parent_keys: Iterable[Tuple[str, Any]]) -> None:
        live = {tuple(k) for k in live_parent_keys}
        for key in list(self._by_parent.keys()):
            if key not in live:
                self._by_parent.pop(key, None)
        for key in list(self._reductions_by_parent.keys()):
            if key not in live:
                self._reductions_by_parent.pop(key, None)
        self._rebuild_global()

    def prune_for_shadow_keys(
        self,
        consumed_shadow_keys: Mapping[Tuple[str, Any], Iterable[Tuple[Any, ...]]],
    ) -> int:
        """Drop selected shadows by parent/key and rebuild the global index."""
        removed = 0
        for parent_key, keys in consumed_shadow_keys.items():
            bucket = self._by_parent.get(tuple(parent_key), None)
            if not bucket:
                continue
            for key in keys or ():
                if bucket.pop(tuple(key), None) is not None:
                    removed += 1
            if not bucket:
                self._by_parent.pop(tuple(parent_key), None)
        if removed:
            self._rebuild_global()
        return int(removed)

    def prune_for_live_parent_vars(
        self,
        live_parent_vars: Mapping[Tuple[str, Any], Sequence[int]],
    ) -> tuple[int, int]:
        """Prune shadows whose parent leaf disappeared or changed support.

        Parent tags are intentionally reused by Stage A candidate rewrites.  A
        shadow may survive such a rewrite only if every raw variable referenced
        by the shadow is still in the live NN atom's raw support.

        Returns ``(removed_parent_buckets, removed_shadow_records)``.
        """
        live = {tuple(k): {int(v) for v in (vals or ())} for k, vals in live_parent_vars.items()}
        removed_parents = 0
        removed_shadows = 0

        for key in sorted(set(self._by_parent.keys()) | set(self._reductions_by_parent.keys())):
            if key not in live:
                removed_shadows += len(self._by_parent.get(key, {}) or {})
                removed_shadows += len(self._reductions_by_parent.get(key, {}) or {})
                self._by_parent.pop(key, None)
                self._reductions_by_parent.pop(key, None)
                removed_parents += 1
                continue

            allowed = live.get(key, set())
            bucket = self._by_parent.get(key, {})
            for shadow_key, shadow in list(bucket.items()):
                try:
                    shadow_vars = {int(v) for v in _collect_var_idxs_from_node(shadow.shadow_ast)}
                except Exception:
                    shadow_vars = set()
                if shadow_vars and not shadow_vars.issubset(allowed):
                    bucket.pop(shadow_key, None)
                    removed_shadows += 1
            if not bucket:
                self._by_parent.pop(key, None)
                removed_parents += 1

            reduction_bucket = self._reductions_by_parent.get(key, {})
            for reduction_key, reduction in list(reduction_bucket.items()):
                raw_vars = {int(v) for v in tuple(getattr(reduction, "raw_var_idxs", ()) or ())}
                if raw_vars and not raw_vars.issubset(allowed):
                    reduction_bucket.pop(reduction_key, None)
                    removed_shadows += 1
            if not reduction_bucket and key in self._reductions_by_parent:
                self._reductions_by_parent.pop(key, None)

        if removed_parents or removed_shadows:
            self._rebuild_global()
        return int(removed_parents), int(removed_shadows)
