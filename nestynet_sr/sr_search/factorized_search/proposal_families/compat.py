# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Legacy scaffold adapter surface.

This module exists only for compatibility with historical scaffold-shaped call
sites and tests. The live proposal/runtime path should use native
``OperatorApplication`` objects and only cross this boundary when an explicit
legacy adapter seam is requested.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..closures import BoundClosure
from .scaffold_enum import enumerate_operator_applications
from .slot_binding import binding_snapshot
from .types import OperatorApplication, OperatorCompatState


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _snapshot_value(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class OuterScaffoldSpec:
    family: str
    scaffold_id: str
    parent_node: tuple
    hole_path: tuple[int, ...]
    target_mode: str = "robust"
    anchor_node: tuple | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "scaffold_id": str(self.scaffold_id),
            "parent_node": _snapshot_value(self.parent_node),
            "hole_path": [int(v) for v in tuple(self.hole_path or ())],
            "target_mode": str(self.target_mode),
            "anchor_node": _snapshot_value(self.anchor_node),
            "metadata": _snapshot_value(self.metadata),
        }


def is_legacy_scaffold_spec(value: Any) -> bool:
    return isinstance(value, OuterScaffoldSpec)


def render_operator_as_scaffold(
    app: OperatorApplication,
    *,
    include_slot_bindings: bool = True,
) -> OuterScaffoldSpec:
    meta = dict(app.metadata or {})
    if include_slot_bindings and app.bindings:
        meta.setdefault("slot_bindings", _snapshot_value(app.bindings))
    compat = app.compat_state
    return OuterScaffoldSpec(
        family=str(app.family),
        scaffold_id=str(app.scaffold_id),
        parent_node=compat.parent_node if isinstance(compat, OperatorCompatState) else app.parent_node,
        hole_path=tuple(int(v) for v in tuple((compat.hole_path if isinstance(compat, OperatorCompatState) else app.hole_path) or ())),
        target_mode=str(app.target_mode),
        anchor_node=(
            compat.anchor_node
            if isinstance(compat, OperatorCompatState) and isinstance(compat.anchor_node, tuple)
            else app.anchor_node if isinstance(app.anchor_node, tuple) else None
        ),
        metadata=meta,
    )


def operator_application_from_scaffold(
    spec: OuterScaffoldSpec,
    *,
    operator_id: str | None = None,
    bindings: Mapping[str, Any] | None = None,
    bound_closure: BoundClosure | None = None,
) -> OperatorApplication:
    meta = dict(spec.metadata or {})
    meta.setdefault("compat_from_legacy_scaffold", True)
    return OperatorApplication(
        family=str(spec.family),
        operator_id=str(operator_id or meta.get("operator", spec.scaffold_id) or spec.scaffold_id),
        target_mode=str(spec.target_mode),
        bindings=dict(bindings or {}),
        bound_closure=bound_closure if isinstance(bound_closure, BoundClosure) else None,
        metadata=meta,
        compat_state=OperatorCompatState(
            scaffold_id=str(spec.scaffold_id),
            parent_node=spec.parent_node,
            hole_path=tuple(int(v) for v in tuple(spec.hole_path or ())),
            anchor_node=spec.anchor_node if isinstance(spec.anchor_node, tuple) else None,
        ),
    )


def enumerate_closure_search_specs(
    *,
    families: Sequence[str] | None,
    nvars: int,
    y_dims,
    var_dims,
    pool_nodes: Sequence[tuple] | None,
    pool_dims: Sequence[Any] | None,
    anchors_per_family: int,
    max_scaffolds: int,
    basis_state=None,
    basis_state_beam: Sequence[Any] | None = None,
) -> list[OuterScaffoldSpec]:
    apps = enumerate_operator_applications(
        families=families,
        nvars=nvars,
        y_dims=y_dims,
        var_dims=var_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        anchors_per_family=anchors_per_family,
        max_scaffolds=max_scaffolds,
        basis_state=basis_state,
        basis_state_beam=basis_state_beam,
    )
    out: list[OuterScaffoldSpec] = []
    for app in list(apps or ()):
        compat_app = app
        if app.bindings:
            compat_app = OperatorApplication(
                family=str(app.family),
                operator_id=str(app.operator_id),
                target_mode=str(app.target_mode),
                bindings=binding_snapshot(app.bindings),
                bound_closure=app.bound_closure,
                metadata=dict(app.metadata or {}),
                compat_state=app.compat_state,
            )
        out.append(render_operator_as_scaffold(compat_app, include_slot_bindings=bool(app.bindings)))
    return out[: max(0, int(max_scaffolds))]


__all__ = [
    "OuterScaffoldSpec",
    "enumerate_closure_search_specs",
    "is_legacy_scaffold_spec",
    "operator_application_from_scaffold",
    "render_operator_as_scaffold",
]
