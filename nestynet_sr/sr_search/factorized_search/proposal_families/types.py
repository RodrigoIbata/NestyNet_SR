# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ..closures import BoundClosure
from ..expr_ast import is_valid_node

if TYPE_CHECKING:
    from .compat import OuterScaffoldSpec


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _snapshot_value(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _valid_node(raw: Any) -> tuple | None:
    node = getattr(raw, "node", raw)
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


@dataclass(frozen=True)
class OperatorCompatState:
    scaffold_id: str
    parent_node: tuple
    hole_path: tuple[int, ...]
    anchor_node: tuple | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scaffold_id": str(self.scaffold_id),
            "parent_node": _snapshot_value(self.parent_node),
            "hole_path": [int(v) for v in tuple(self.hole_path or ())],
            "anchor_node": _snapshot_value(self.anchor_node),
        }


@dataclass(frozen=True, init=False)
class OperatorApplication:
    family: str
    operator_id: str
    target_mode: str
    bindings: Mapping[str, Any]
    bound_closure: BoundClosure | None
    metadata: dict[str, Any]
    compat_state: OperatorCompatState | None

    def __init__(
        self,
        *,
        family: str,
        operator_id: str,
        target_mode: str = "robust",
        bindings: Mapping[str, Any] | None = None,
        bound_closure: BoundClosure | None = None,
        metadata: Mapping[str, Any] | None = None,
        compat_state: OperatorCompatState | None = None,
        scaffold_id: str | None = None,
        parent_node: tuple | None = None,
        hole_path: tuple[int, ...] | None = None,
        anchor_node: tuple | None = None,
    ) -> None:
        meta = dict(metadata or {})
        compat = compat_state
        if compat is None and (
            scaffold_id is not None
            or isinstance(parent_node, tuple)
            or hole_path is not None
            or isinstance(anchor_node, tuple)
        ):
            compat = OperatorCompatState(
                scaffold_id=str(scaffold_id or meta.get("scaffold_id", operator_id) or operator_id),
                parent_node=parent_node if isinstance(parent_node, tuple) and is_valid_node(parent_node) else ("const", 1.0),
                hole_path=tuple(int(v) for v in tuple(hole_path or ())),
                anchor_node=anchor_node if isinstance(anchor_node, tuple) and is_valid_node(anchor_node) else None,
            )
        object.__setattr__(self, "family", str(family))
        object.__setattr__(self, "operator_id", str(operator_id))
        object.__setattr__(self, "target_mode", str(target_mode))
        object.__setattr__(self, "bindings", dict(bindings or {}))
        object.__setattr__(self, "bound_closure", bound_closure if isinstance(bound_closure, BoundClosure) else None)
        object.__setattr__(self, "metadata", meta)
        object.__setattr__(self, "compat_state", compat if isinstance(compat, OperatorCompatState) else None)

    @property
    def scaffold_id(self) -> str:
        if isinstance(self.compat_state, OperatorCompatState):
            return str(self.compat_state.scaffold_id)
        bound_meta = dict(getattr(self.bound_closure, "metadata", {}) or {})
        return str(self.metadata.get("scaffold_id", bound_meta.get("scaffold_id", self.operator_id)) or self.operator_id)

    @property
    def parent_node(self) -> tuple:
        if isinstance(self.compat_state, OperatorCompatState):
            return self.compat_state.parent_node
        meta_parent = self.metadata.get("parent_node", None)
        meta_parent_node = _valid_node(meta_parent)
        if meta_parent_node is not None:
            return meta_parent_node
        bound_bindings = dict(getattr(self.bound_closure, "bindings", {}) or {})
        expr_node = _valid_node(bound_bindings.get("expr"))
        if expr_node is not None:
            return expr_node
        feature_node = _valid_node(bound_bindings.get("feature"))
        if feature_node is not None:
            return feature_node
        carrier_node = _valid_node(bound_bindings.get("carrier"))
        if carrier_node is not None:
            return carrier_node
        numerator = _valid_node(bound_bindings.get("numerator"))
        denominator = _valid_node(bound_bindings.get("denominator"))
        if numerator is not None and denominator is not None:
            return ("div", numerator, denominator)
        bases = tuple(
            node
            for node in (_valid_node(raw) for raw in tuple(bound_bindings.get("bases", ()) or ()))
            if node is not None
        )
        if bases:
            return bases[0]
        return ("const", 1.0)

    @property
    def hole_path(self) -> tuple[int, ...]:
        if isinstance(self.compat_state, OperatorCompatState):
            return tuple(int(v) for v in tuple(self.compat_state.hole_path or ()))
        raw = self.metadata.get("hole_path", ())
        return tuple(int(v) for v in tuple(raw or ()))

    @property
    def anchor_node(self) -> tuple | None:
        if isinstance(self.compat_state, OperatorCompatState):
            return self.compat_state.anchor_node
        meta_anchor = _valid_node(self.metadata.get("anchor_node", None))
        if meta_anchor is not None:
            return meta_anchor
        bound_bindings = dict(getattr(self.bound_closure, "bindings", {}) or {})
        return _valid_node(bound_bindings.get("anchor")) or _valid_node(bound_bindings.get("companion"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "operator_id": str(self.operator_id),
            "scaffold_id": str(self.scaffold_id),
            "parent_node": _snapshot_value(self.parent_node),
            "hole_path": [int(v) for v in tuple(self.hole_path or ())],
            "target_mode": str(self.target_mode),
            "anchor_node": _snapshot_value(self.anchor_node),
            "bindings": _snapshot_value(self.bindings),
            "bound_closure": (
                self.bound_closure.to_dict() if isinstance(self.bound_closure, BoundClosure) else None
            ),
            "metadata": _snapshot_value(self.metadata),
            "compat_state": (
                self.compat_state.to_dict() if isinstance(self.compat_state, OperatorCompatState) else None
            ),
        }

    def to_scaffold_spec(self, *, include_slot_bindings: bool = True) -> OuterScaffoldSpec:
        from .compat import render_operator_as_scaffold

        return render_operator_as_scaffold(self, include_slot_bindings=include_slot_bindings)


@dataclass(frozen=True)
class ScaffoldPreviewCandidate:
    expr: tuple
    family: str
    scaffold_id: str
    preview_probe_mse: float
    preview_fit_mse: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "OperatorApplication",
    "OperatorCompatState",
    "ScaffoldPreviewCandidate",
]
