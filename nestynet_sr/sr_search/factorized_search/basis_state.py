# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Thin basis/block abstractions for iterative symbolic basis construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .basis_compile import basis_expr_key, canonicalize_basis_expr
from .closures import (
    BoundClosure,
    bound_closure_from_closure_candidate,
    bound_closure_identity_key,
)
from .expr_ast import is_valid_node, node_depth, node_size, node_str, simplify


def _valid_node(node: Any) -> tuple | None:
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


def _snapshot_value(value: Any) -> Any:
    node = _valid_node(value)
    if node is not None:
        return str(node_str(node))
    if isinstance(value, Mapping):
        return {str(k): _snapshot_value(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collect_active_vars(nodes: Sequence[Any]) -> tuple[int, ...]:
    seen: set[int] = set()

    def _visit(cur: Any) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        op = str(cur[0])
        if op == "var":
            try:
                seen.add(int(cur[1]))
            except Exception:
                pass
            return
        for child in cur[1:]:
            if isinstance(child, tuple):
                _visit(child)

    for node in nodes:
        _visit(node)
    return tuple(sorted(seen))


def _block_complexity(nodes: Sequence[Any]) -> float:
    seen: set[str] = set()
    total = 0.0
    for raw_node in nodes:
        node = _valid_node(raw_node)
        if node is None:
            continue
        key = str(node_str(node))
        if key in seen:
            continue
        seen.add(key)
        total += float(node_size(node))
    return float(total)


def _bundle_nodes(block: "FeatureBlock") -> tuple[tuple, ...]:
    rows = []
    for node in tuple(getattr(block, "latent_bundle_nodes", ()) or ()):
        valid = _valid_node(node)
        if valid is not None:
            rows.append(valid)
    return tuple(rows)


def _head_bundle_nodes(block: "FeatureBlock") -> tuple[tuple, ...]:
    rows = []
    for node in tuple(getattr(block, "head_bundle_nodes", ()) or ()):
        valid = _valid_node(node)
        if valid is not None:
            rows.append(valid)
    return tuple(rows)


def _head_type_from_mapping(mapping_kind: str) -> str:
    kind = str(mapping_kind or "").strip().lower()
    if "rational" in kind or "varpro" in kind:
        return "varpro"
    if "exact" in kind:
        return "exact"
    if kind:
        return "linear"
    return "unknown"


@dataclass(frozen=True)
class FeatureBlock:
    family: str
    atoms: tuple[tuple, ...]
    head_type: str
    block_id: str = ""
    parent_block_ids: tuple[str, ...] = ()
    latent_bundle_nodes: tuple[tuple, ...] = ()
    latent_bundle_roles: tuple[str, ...] = ()
    head_bundle_nodes: tuple[tuple, ...] = ()
    head_bundle_roles: tuple[str, ...] = ()
    dim_signature: Any = None
    active_vars: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def complexity(self) -> float:
        return _block_complexity((*self.atoms, *_bundle_nodes(self), *_head_bundle_nodes(self)))

    def to_dict(self) -> dict[str, Any]:
        atom_exprs = [str(node_str(node)) for node in self.atoms if _valid_node(node) is not None]
        bundle_rows = []
        for role, node in zip(
            tuple(getattr(self, "latent_bundle_roles", ()) or ()),
            tuple(getattr(self, "latent_bundle_nodes", ()) or ()),
        ):
            valid = _valid_node(node)
            if valid is None:
                continue
            bundle_rows.append(
                {
                    "role": str(role),
                    "expr": str(node_str(valid)),
                    "size": int(node_size(valid)),
                    "depth": int(node_depth(valid)),
                }
            )
        head_bundle_rows = []
        for role, node in zip(
            tuple(getattr(self, "head_bundle_roles", ()) or ()),
            tuple(getattr(self, "head_bundle_nodes", ()) or ()),
        ):
            valid = _valid_node(node)
            if valid is None:
                continue
            head_bundle_rows.append(
                {
                    "role": str(role),
                    "expr": str(node_str(valid)),
                    "size": int(node_size(valid)),
                    "depth": int(node_depth(valid)),
                }
            )
        return {
            "family": str(self.family),
            "head_type": str(self.head_type),
            "block_id": str(feature_block_id(self)),
            "parent_block_ids": [str(v) for v in feature_block_parent_ids(self)],
            "atom_exprs": atom_exprs,
            "atom_sizes": [int(node_size(node)) for node in self.atoms if _valid_node(node) is not None],
            "atom_depths": [int(node_depth(node)) for node in self.atoms if _valid_node(node) is not None],
            "latent_bundle": bundle_rows,
            "latent_bundle_exprs": [str(row["expr"]) for row in bundle_rows],
            "latent_bundle_roles": [str(row["role"]) for row in bundle_rows],
            "head_bundle": head_bundle_rows,
            "head_bundle_exprs": [str(row["expr"]) for row in head_bundle_rows],
            "head_bundle_roles": [str(row["role"]) for row in head_bundle_rows],
            "active_vars": [int(v) for v in self.active_vars],
            "complexity": float(self.complexity()),
            "dim_signature": _snapshot_value(self.dim_signature),
            "metadata": _snapshot_value(self.metadata),
        }


@dataclass(frozen=True)
class BasisState:
    blocks: tuple[FeatureBlock, ...]
    fit_bundle: Mapping[str, Any] = field(default_factory=dict)
    fit_loss: float = math.inf
    probe_loss: float = math.inf
    complexity: float = 0.0
    residual_fit: Any = None
    residual_probe: Any = None
    residual_witness: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[Any, ...] = ()
    compiled_expr: tuple | None = None

    def to_dict(self) -> dict[str, Any]:
        compiled_canon = canonicalize_basis_expr(self.compiled_expr)
        return {
            "block_count": int(len(self.blocks)),
            "block_families": [str(block.family) for block in self.blocks],
            "fit_loss": float(self.fit_loss),
            "probe_loss": float(self.probe_loss),
            "complexity": float(self.complexity),
            "compiled_expr": (
                str(node_str(compiled_canon))
                if _valid_node(compiled_canon) is not None
                else ""
            ),
            "fit_bundle": _snapshot_value(self.fit_bundle),
            "diagnostics": _snapshot_value(self.diagnostics),
            "provenance": _snapshot_value(self.provenance),
            "residual_witness": _snapshot_value(self.residual_witness),
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class ProposalContext:
    basis_state: BasisState | None = None
    basis_state_beam: tuple[BasisState, ...] = ()
    aux_seed_blocks: tuple[Any, ...] = ()
    atom_library: Any = None
    residual_witness: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    family_hints: Mapping[str, float] = field(default_factory=dict)
    total_budget: int | None = None
    wall_time_remaining_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis_state": self.basis_state.to_dict() if isinstance(self.basis_state, BasisState) else None,
            "basis_state_beam": [
                state.to_dict() for state in tuple(self.basis_state_beam or ()) if isinstance(state, BasisState)
            ],
            "aux_seed_blocks": [
                block.to_dict() if hasattr(block, "to_dict") else _snapshot_value(block)
                for block in tuple(self.aux_seed_blocks or ())
            ],
            "atom_library": (
                self.atom_library.to_dict() if hasattr(self.atom_library, "to_dict") else _snapshot_value(self.atom_library)
            ),
            "residual_witness": _snapshot_value(self.residual_witness),
            "diagnostics": _snapshot_value(self.diagnostics),
            "family_hints": {
                str(k): float(v)
                for k, v in dict(self.family_hints or {}).items()
                if isinstance(v, (int, float))
            },
            "total_budget": None if self.total_budget is None else int(self.total_budget),
            "wall_time_remaining_s": (
                None if self.wall_time_remaining_s is None else float(self.wall_time_remaining_s)
            ),
        }


@dataclass(frozen=True)
class ProposalCandidate:
    family: str
    rendered_expr: tuple | None = None
    scaffold_id: str = ""
    identity_key: str = ""
    feature_block: FeatureBlock | None = None
    basis_state: BasisState | None = None
    bound_closure: BoundClosure | None = None
    local_fit_loss: float = math.inf
    local_probe_loss: float = math.inf
    complexity: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expr(self) -> tuple | None:
        return self.rendered_expr

    def to_dict(self) -> dict[str, Any]:
        expr_node = _valid_node(self.rendered_expr)
        return {
            "family": str(self.family),
            "scaffold_id": str(self.scaffold_id),
            "identity_key": str(self.identity_key),
            "expr": str(node_str(expr_node)) if expr_node is not None else "",
            "rendered_expr": str(node_str(expr_node)) if expr_node is not None else "",
            "local_fit_loss": float(self.local_fit_loss),
            "local_probe_loss": float(self.local_probe_loss),
            "complexity": float(self.complexity),
            "feature_block": (
                self.feature_block.to_dict() if isinstance(self.feature_block, FeatureBlock) else None
            ),
            "basis_state": (
                self.basis_state.to_dict() if isinstance(self.basis_state, BasisState) else None
            ),
            "bound_closure": (
                self.bound_closure.to_dict() if isinstance(self.bound_closure, BoundClosure) else None
            ),
            "metadata": _snapshot_value(self.metadata),
        }


class ProposalFamily(Protocol):
    """Minimal expert interface for proposing feature blocks."""

    def propose(self, context: ProposalContext, budget: int) -> list[ProposalCandidate]:
        ...


def _block_identity(block: FeatureBlock) -> str:
    parent_ids = feature_block_parent_ids(block)
    parent_suffix = f"@@parents:{','.join(sorted(parent_ids))}" if parent_ids else ""
    head_bundle_nodes = tuple(getattr(block, "head_bundle_nodes", ()) or ())
    head_bundle_roles = tuple(getattr(block, "head_bundle_roles", ()) or ())
    if head_bundle_nodes:
        entries = []
        for role, node in zip(head_bundle_roles, head_bundle_nodes):
            valid = _valid_node(node)
            if valid is None:
                continue
            entries.append(f"head:{str(role)}:{str(node_str(valid))}")
        if entries:
            return "|".join(entries) + parent_suffix
    bundle_nodes = tuple(getattr(block, "latent_bundle_nodes", ()) or ())
    bundle_roles = tuple(getattr(block, "latent_bundle_roles", ()) or ())
    if bundle_nodes:
        entries = []
        for role, node in zip(bundle_roles, bundle_nodes):
            valid = _valid_node(node)
            if valid is None:
                continue
            entries.append(f"{str(role)}:{str(node_str(valid))}")
        if entries:
            return "|".join(entries) + parent_suffix
    atom_keys = [
        str(node_str(atom))
        for atom in tuple(getattr(block, "atoms", ()) or ())
        if _valid_node(atom) is not None
    ]
    return "|".join(atom_keys) + parent_suffix


def _normalize_block_id(raw: Any) -> str:
    token = str(raw or "").strip()
    return token


def _normalize_parent_block_ids(raw: Any) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in tuple(raw or ()):
        token = _normalize_block_id(value)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def feature_block_id(block: FeatureBlock | None) -> str:
    if not isinstance(block, FeatureBlock):
        return ""
    direct = _normalize_block_id(getattr(block, "block_id", ""))
    if direct:
        return direct
    metadata = dict(getattr(block, "metadata", {}) or {})
    meta_id = _normalize_block_id(metadata.get("block_id", ""))
    if meta_id:
        return meta_id
    return _block_identity(block)


def feature_block_parent_ids(block: FeatureBlock | None) -> tuple[str, ...]:
    if not isinstance(block, FeatureBlock):
        return ()
    direct = _normalize_parent_block_ids(getattr(block, "parent_block_ids", ()))
    if direct:
        return direct
    metadata = dict(getattr(block, "metadata", {}) or {})
    return _normalize_parent_block_ids(metadata.get("parent_block_ids", ()))


def _normalize_feature_block_graph(block: FeatureBlock | None) -> FeatureBlock | None:
    block = ensure_feature_block_head_bundle(block)
    if not isinstance(block, FeatureBlock):
        return None
    block_id = feature_block_id(block)
    parent_ids = tuple(pid for pid in feature_block_parent_ids(block) if pid and pid != block_id)
    metadata = dict(getattr(block, "metadata", {}) or {})
    metadata["block_id"] = str(block_id)
    metadata["parent_block_ids"] = [str(pid) for pid in parent_ids]
    if (
        str(getattr(block, "block_id", "") or "") == str(block_id)
        and tuple(getattr(block, "parent_block_ids", ()) or ()) == tuple(parent_ids)
        and metadata == dict(getattr(block, "metadata", {}) or {})
    ):
        return block
    return FeatureBlock(
        family=str(block.family),
        atoms=tuple(block.atoms),
        head_type=str(block.head_type),
        block_id=str(block_id),
        parent_block_ids=tuple(parent_ids),
        latent_bundle_nodes=tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
        latent_bundle_roles=tuple(getattr(block, "latent_bundle_roles", ()) or ()),
        head_bundle_nodes=tuple(getattr(block, "head_bundle_nodes", ()) or ()),
        head_bundle_roles=tuple(getattr(block, "head_bundle_roles", ()) or ()),
        dim_signature=block.dim_signature,
        active_vars=tuple(getattr(block, "active_vars", ()) or ()),
        metadata=metadata,
    )


def topologically_order_feature_blocks(
    blocks: Sequence[FeatureBlock] | None,
    *,
    drop_orphans: bool = True,
) -> tuple[FeatureBlock, ...]:
    normalized = [
        block
        for block in (
            _normalize_feature_block_graph(block)
            for block in tuple(blocks or ())
        )
        if isinstance(block, FeatureBlock)
    ]
    if not normalized:
        return ()
    by_id: dict[str, FeatureBlock] = {}
    order_index: dict[str, int] = {}
    for idx, block in enumerate(normalized):
        block_id = feature_block_id(block)
        if not block_id:
            continue
        prev = by_id.get(block_id, None)
        if prev is None:
            by_id[block_id] = block
            order_index[block_id] = int(idx)
    existing_ids = set(by_id)
    parent_map = {
        block_id: tuple(pid for pid in feature_block_parent_ids(block) if pid in existing_ids)
        for block_id, block in by_id.items()
    }
    unresolved = {
        block_id
        for block_id, block in by_id.items()
        if any(pid not in existing_ids for pid in feature_block_parent_ids(block))
    }
    if unresolved and not drop_orphans:
        for block_id in sorted(unresolved, key=lambda key: order_index.get(key, 0)):
            parent_map[block_id] = ()
    elif unresolved and drop_orphans:
        for block_id in unresolved:
            by_id.pop(block_id, None)
            parent_map.pop(block_id, None)
            order_index.pop(block_id, None)
    ordered: list[FeatureBlock] = []
    emitted: set[str] = set()
    while True:
        progress = False
        for block_id, block in sorted(by_id.items(), key=lambda item: order_index.get(item[0], 0)):
            if block_id in emitted:
                continue
            parents = tuple(pid for pid in parent_map.get(block_id, ()) if pid in by_id)
            if all(pid in emitted for pid in parents):
                ordered.append(block)
                emitted.add(block_id)
                progress = True
        if len(emitted) == len(by_id):
            return tuple(ordered)
        if not progress:
            if drop_orphans:
                break
            for block_id, block in sorted(by_id.items(), key=lambda item: order_index.get(item[0], 0)):
                if block_id not in emitted:
                    ordered.append(block)
                    emitted.add(block_id)
            return tuple(ordered)
    return tuple(block for block in ordered if isinstance(block, FeatureBlock))


def closure_keep_feature_blocks(
    blocks: Sequence[FeatureBlock] | None,
    keep_block_ids: Sequence[str] | None,
) -> tuple[FeatureBlock, ...]:
    ordered = topologically_order_feature_blocks(blocks, drop_orphans=True)
    by_id = {feature_block_id(block): block for block in ordered if feature_block_id(block)}
    if not by_id:
        return ()
    keep: set[str] = {
        token
        for token in (_normalize_block_id(value) for value in tuple(keep_block_ids or ()))
        if token in by_id
    }
    if not keep:
        return ()
    stack = list(keep)
    while stack:
        block_id = stack.pop()
        block = by_id.get(block_id, None)
        if not isinstance(block, FeatureBlock):
            continue
        for parent_id in feature_block_parent_ids(block):
            if parent_id in by_id and parent_id not in keep:
                keep.add(parent_id)
                stack.append(parent_id)
    return tuple(block for block in ordered if feature_block_id(block) in keep)


def drop_feature_block_with_dependents(
    blocks: Sequence[FeatureBlock] | None,
    drop_block_id: str,
) -> tuple[FeatureBlock, ...]:
    ordered = topologically_order_feature_blocks(blocks, drop_orphans=True)
    by_id = {feature_block_id(block): block for block in ordered if feature_block_id(block)}
    target = _normalize_block_id(drop_block_id)
    if not target or target not in by_id:
        return ordered
    reverse_edges: dict[str, set[str]] = {block_id: set() for block_id in by_id}
    for block_id, block in by_id.items():
        for parent_id in feature_block_parent_ids(block):
            if parent_id in reverse_edges:
                reverse_edges[parent_id].add(block_id)
    removed: set[str] = {target}
    stack = [target]
    while stack:
        current = stack.pop()
        for child_id in tuple(reverse_edges.get(current, ())):
            if child_id in removed:
                continue
            removed.add(child_id)
            stack.append(child_id)
    survivors = [block for block in ordered if feature_block_id(block) not in removed]
    return topologically_order_feature_blocks(survivors, drop_orphans=True)


def _basis_state_identity(state: BasisState) -> str:
    block_keys = [
        _block_identity(block)
        for block in tuple(getattr(state, "blocks", ()) or ())
        if isinstance(block, FeatureBlock)
    ]
    compiled_key = basis_expr_key(getattr(state, "compiled_expr", None))
    return "||".join(sorted(block_keys)) + f"##{compiled_key}"


def _block_node_keys(block: FeatureBlock | None) -> set[str]:
    if not isinstance(block, FeatureBlock):
        return set()
    nodes = [*_head_bundle_nodes(block), *_bundle_nodes(block), *tuple(getattr(block, "atoms", ()) or ())]
    return {
        str(node_str(node))
        for node in nodes
        if _valid_node(node) is not None
    }


def _role_node_keys(roles: Sequence[Any] | None, nodes: Sequence[Any] | None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for role, node in zip(tuple(roles or ()), tuple(nodes or ())):
        valid = _valid_node(node)
        if valid is None:
            continue
        out.add((str(role), str(node_str(valid))))
    return out


def _basis_state_block_keys(state: BasisState) -> frozenset[str]:
    return frozenset(
        _block_identity(block)
        for block in tuple(getattr(state, "blocks", ()) or ())
        if isinstance(block, FeatureBlock)
    )


def _basis_state_sort_key(state: BasisState) -> tuple[float, float, int, str]:
    compiled_key = basis_expr_key(getattr(state, "compiled_expr", None))
    return (
        float(getattr(state, "probe_loss", math.inf)),
        float(getattr(state, "complexity", math.inf)),
        int(len(tuple(getattr(state, "blocks", ()) or ()))),
        compiled_key,
    )


def basis_state_retarget(
    state: BasisState | None,
    *,
    fit_loss: float | None = None,
    probe_loss: float | None = None,
    compiled_expr: tuple | None = None,
    route_name: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    provenance_tag: Any | None = None,
) -> BasisState | None:
    if not isinstance(state, BasisState):
        return None
    merged_diag = dict(getattr(state, "diagnostics", {}) or {})
    if isinstance(diagnostics, Mapping):
        merged_diag.update(dict(diagnostics or {}))
    if route_name:
        merged_diag["route"] = str(route_name)
    provenance = list(tuple(getattr(state, "provenance", ()) or ()))
    if provenance_tag is not None:
        provenance.append(provenance_tag)
    return BasisState(
        blocks=tuple(state.blocks),
        fit_bundle=dict(state.fit_bundle or {}),
        fit_loss=float(state.fit_loss if fit_loss is None else fit_loss),
        probe_loss=float(state.probe_loss if probe_loss is None else probe_loss),
        complexity=float(state.complexity),
        residual_fit=state.residual_fit,
        residual_probe=state.residual_probe,
        residual_witness=state.residual_witness,
        diagnostics=merged_diag,
        provenance=tuple(provenance),
        compiled_expr=canonicalize_basis_expr(compiled_expr) or canonicalize_basis_expr(state.compiled_expr),
    )


def prepare_basis_state_candidate(
    current_basis_state: BasisState | None,
    preview_state: BasisState | None,
    *,
    route_name: str = "basis_prepare",
    compiled_expr: tuple | None = None,
) -> BasisState | None:
    if isinstance(current_basis_state, BasisState) and isinstance(preview_state, BasisState):
        return basis_state_extend(
            current_basis_state,
            preview_state,
            route_name=str(route_name),
            fit_loss=float(preview_state.fit_loss),
            probe_loss=float(preview_state.probe_loss),
            compiled_expr=compiled_expr if isinstance(compiled_expr, tuple) else preview_state.compiled_expr,
        )
    if isinstance(preview_state, BasisState):
        return basis_state_retarget(
            preview_state,
            compiled_expr=compiled_expr if isinstance(compiled_expr, tuple) else preview_state.compiled_expr,
            route_name=str(route_name),
            provenance_tag=f"{str(route_name)}:preview",
        )
    if isinstance(current_basis_state, BasisState):
        return basis_state_retarget(
            current_basis_state,
            compiled_expr=compiled_expr if isinstance(compiled_expr, tuple) else current_basis_state.compiled_expr,
            route_name=str(route_name),
            provenance_tag=f"{str(route_name)}:carry",
        )
    return None


def _basis_state_dominates(lhs: BasisState, rhs: BasisState) -> bool:
    lhs_keys = _basis_state_block_keys(lhs)
    rhs_keys = _basis_state_block_keys(rhs)
    if not lhs_keys or not rhs_keys:
        return False
    if not lhs_keys.issubset(rhs_keys):
        return False
    probe_tol = max(1.0e-12, abs(float(getattr(rhs, "probe_loss", math.inf))) * 1.0e-9)
    fit_tol = max(1.0e-12, abs(float(getattr(rhs, "fit_loss", math.inf))) * 1.0e-9)
    if float(getattr(lhs, "probe_loss", math.inf)) > float(getattr(rhs, "probe_loss", math.inf)) + probe_tol:
        return False
    if float(getattr(lhs, "fit_loss", math.inf)) > float(getattr(rhs, "fit_loss", math.inf)) + fit_tol:
        return False
    if float(getattr(lhs, "complexity", math.inf)) > float(getattr(rhs, "complexity", math.inf)) + 1.0e-9:
        return False
    return True


def prune_basis_state_beam(
    beam: Sequence[BasisState] | None,
    *,
    beam_width: int = 3,
) -> tuple[BasisState, ...]:
    rows = [row for row in list(beam or ()) if isinstance(row, BasisState)]
    if not rows or int(beam_width) <= 0:
        return ()
    best_by_id: dict[str, BasisState] = {}
    for row in rows:
        key = _basis_state_identity(row)
        prev = best_by_id.get(key, None)
        if prev is None or _basis_state_sort_key(row) < _basis_state_sort_key(prev):
            best_by_id[key] = row
    ranked = sorted(best_by_id.values(), key=_basis_state_sort_key)
    kept: list[BasisState] = []
    for row in ranked:
        if any(_basis_state_dominates(existing, row) for existing in kept):
            continue
        kept = [existing for existing in kept if not _basis_state_dominates(row, existing)]
        kept.append(row)
        kept.sort(key=_basis_state_sort_key)
        if len(kept) > int(beam_width):
            kept = kept[: int(beam_width)]
    return tuple(kept)


def _feature_block_from_node(
    *,
    family: str,
    node: tuple | None,
    head_type: str,
    metadata: Mapping[str, Any] | None = None,
    parent_block_ids: Sequence[str] | None = None,
) -> FeatureBlock | None:
    node_valid = _valid_node(node)
    if node_valid is None:
        return None
    block = FeatureBlock(
        family=str(family),
        atoms=(node_valid,),
        head_type=str(head_type),
        block_id="",
        parent_block_ids=tuple(_normalize_parent_block_ids(parent_block_ids)),
        latent_bundle_nodes=(node_valid,),
        latent_bundle_roles=("primary",),
        head_bundle_nodes=(node_valid,),
        head_bundle_roles=("primary",),
        active_vars=_collect_active_vars((node_valid,)),
        metadata={
            "block_expr_obj": node_valid,
            "block_expr": str(node_str(node_valid)),
            "head_bundle_exprs": [str(node_str(node_valid))],
            "head_bundle_roles": ["primary"],
            **dict(metadata or {}),
        },
    )
    return _normalize_feature_block_graph(block)


def _atoms_from_bound_closure(bound_closure: BoundClosure | None) -> tuple[tuple, ...]:
    if not isinstance(bound_closure, BoundClosure):
        return ()
    atoms: list[tuple] = []

    def _visit(value: Any) -> None:
        node = _valid_node(value)
        if node is not None:
            atoms.append(node)
            return
        if isinstance(value, Mapping):
            for child in dict(value).values():
                _visit(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                _visit(child)

    _visit(bound_closure.bindings)
    dedup: list[tuple] = []
    seen: set[str] = set()
    for node in atoms:
        key = str(node_str(node))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(node)
    return tuple(dedup)


def _binding_parent_block_ids(value: Any) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()

    def _visit(cur: Any) -> None:
        if isinstance(cur, Mapping):
            for child in dict(cur).values():
                _visit(child)
            return
        if isinstance(cur, (list, tuple)) and not _valid_node(cur):
            for child in tuple(cur or ()):
                _visit(child)
            return
        metadata = dict(getattr(cur, "metadata", {}) or {})
        block_id = _normalize_block_id(metadata.get("basis_block_id", ""))
        if block_id and block_id not in seen:
            seen.add(block_id)
            out.append(block_id)

    _visit(value)
    return tuple(out)


def _append_bundle_entry(
    out: list[tuple[str, tuple]],
    seen: set[tuple[str, str]],
    *,
    role: str,
    value: Any,
) -> None:
    valid = _valid_node(value)
    if valid is None:
        return
    key = (str(role), str(node_str(valid)))
    if key in seen:
        return
    seen.add(key)
    out.append((str(role), valid))


def _append_bundle_entries(
    out: list[tuple[str, tuple]],
    seen: set[tuple[str, str]],
    *,
    role: str,
    values: Sequence[Any] | None,
) -> None:
    for value in list(values or ()):
        _append_bundle_entry(out, seen, role=str(role), value=value)


def _latent_bundle_from_candidate(
    *,
    bound_closure: BoundClosure | None,
    direct_metadata: Mapping[str, Any] | None,
    anchor_node: tuple | None,
    scaffold_form: str,
    expr: tuple | None,
) -> tuple[tuple[str, tuple], ...]:
    direct_meta = dict(direct_metadata or {})
    rows: list[tuple[str, tuple]] = []
    seen: set[tuple[str, str]] = set()

    if isinstance(bound_closure, BoundClosure):
        bindings = dict(bound_closure.bindings or {})
        _append_bundle_entries(rows, seen, role="term", values=bindings.get("terms", None))
        _append_bundle_entry(rows, seen, role="carrier", value=bindings.get("carrier", None))
        _append_bundle_entry(rows, seen, role="feature", value=bindings.get("feature", None))
        _append_bundle_entries(rows, seen, role="harmonic_feature", values=bindings.get("harmonic_features", None))
        _append_bundle_entry(rows, seen, role="envelope", value=bindings.get("envelope", None))
        _append_bundle_entry(rows, seen, role="anchor", value=bindings.get("anchor", None))
        _append_bundle_entry(rows, seen, role="companion", value=bindings.get("companion", None))
        _append_bundle_entries(rows, seen, role="companion", values=bindings.get("companions", None))
        _append_bundle_entry(rows, seen, role="numerator", value=bindings.get("numerator", None))
        _append_bundle_entry(rows, seen, role="denominator", value=bindings.get("denominator", None))
        _append_bundle_entries(rows, seen, role="base", values=bindings.get("bases", None))

    _append_bundle_entry(rows, seen, role="feature", value=direct_meta.get("feature_node", None))
    _append_bundle_entries(rows, seen, role="term", values=direct_meta.get("term_nodes", None))
    _append_bundle_entry(rows, seen, role="carrier", value=direct_meta.get("hole_node", None))
    _append_bundle_entry(rows, seen, role="carrier", value=direct_meta.get("power_inner_node", None))
    _append_bundle_entries(rows, seen, role="harmonic_feature", values=direct_meta.get("harmonic_feature_nodes", None))
    _append_bundle_entry(rows, seen, role="envelope", value=direct_meta.get("envelope_node", None))
    _append_bundle_entries(rows, seen, role="companion", values=direct_meta.get("companion_nodes", None))
    _append_bundle_entry(rows, seen, role="numerator", value=direct_meta.get("u_node", None))
    _append_bundle_entry(rows, seen, role="denominator", value=direct_meta.get("v_node", None))
    _append_bundle_entries(rows, seen, role="base", values=direct_meta.get("quadratic_base_nodes", None))
    _append_bundle_entry(rows, seen, role="latent", value=direct_meta.get("quadratic_latent_node", None))
    _append_bundle_entry(rows, seen, role="envelope", value=direct_meta.get("anchor_lift_node", None))

    anchor = _valid_node(anchor_node)
    form = str(scaffold_form or "").strip().lower()
    if anchor is not None:
        if form.endswith("_add"):
            _append_bundle_entry(rows, seen, role="companion", value=anchor)
        elif form.endswith("_mul"):
            _append_bundle_entry(rows, seen, role="envelope", value=anchor)
        else:
            _append_bundle_entry(rows, seen, role="anchor", value=anchor)

    if not rows:
        _append_bundle_entry(rows, seen, role="expr", value=expr)
    return tuple(rows)


def _bundle_first(latent_bundle: Sequence[tuple[str, tuple]], role: str) -> tuple | None:
    token = str(role or "")
    for entry_role, node in tuple(latent_bundle or ()):
        if str(entry_role) == token:
            return _valid_node(node)
    return None


def _bundle_all(latent_bundle: Sequence[tuple[str, tuple]], role: str) -> tuple[tuple, ...]:
    token = str(role or "")
    out: list[tuple] = []
    for entry_role, node in tuple(latent_bundle or ()):
        if str(entry_role) != token:
            continue
        valid = _valid_node(node)
        if valid is not None:
            out.append(valid)
    return tuple(out)


def _interaction_node_from_bundle(
    *,
    family: str,
    latent_bundle: Sequence[tuple[str, tuple]],
    direct_metadata: Mapping[str, Any] | None,
    expr: tuple | None,
) -> tuple | None:
    family_token = str(family or "").strip().lower()
    direct_meta = dict(direct_metadata or {})
    preferred_roles = {
        "periodic": ("harmonic_feature", "feature", "carrier"),
        "power": ("carrier", "feature", "harmonic_feature"),
        "exp": ("feature", "carrier"),
        "log": ("feature", "carrier"),
    }.get(family_token, ("harmonic_feature", "feature", "carrier"))
    for role in preferred_roles:
        node = _bundle_first(latent_bundle, role)
        if node is not None:
            return node
    for key in ("feature_node", "power_inner_node", "hole_node"):
        node = _valid_node(direct_meta.get(key, None))
        if node is not None:
            return node
    return _valid_node(expr)


def _anchor_diversity_node_from_bundle(
    *,
    latent_bundle: Sequence[tuple[str, tuple]],
    direct_metadata: Mapping[str, Any] | None,
    anchor_node: tuple | None,
) -> tuple | None:
    direct_meta = dict(direct_metadata or {})
    for key in ("anchor_node", "envelope_node", "anchor_lift_node"):
        node = _valid_node(direct_meta.get(key, None))
        if node is not None:
            return node
    for role in ("anchor", "envelope", "companion"):
        node = _bundle_first(latent_bundle, role)
        if node is not None:
            return node
    return _valid_node(anchor_node)


def _head_bundle_from_candidate(
    *,
    family: str,
    latent_bundle: Sequence[tuple[str, tuple]],
    local_mapping_kind: str,
    scaffold_metadata: Mapping[str, Any] | None,
    direct_metadata: Mapping[str, Any] | None,
    expr: tuple | None,
) -> tuple[tuple[str, tuple], ...]:
    family_token = str(family or "").strip().lower()
    scaffold_meta = dict(scaffold_metadata or {})
    direct_meta = dict(direct_metadata or {})
    form = str(scaffold_meta.get("form", "") or "").strip().lower()
    rows: list[tuple[str, tuple]] = []
    seen: set[tuple[str, str]] = set()

    def _add(role: str, node: Any) -> None:
        _append_bundle_entry(rows, seen, role=str(role), value=node)

    def _add_many(role: str, nodes: Sequence[Any] | None) -> None:
        _append_bundle_entries(rows, seen, role=str(role), values=nodes)

    if family_token == "periodic":
        envelope = _bundle_first(latent_bundle, "envelope")
        harmonic_nodes = list(_bundle_all(latent_bundle, "harmonic_feature"))
        companion_nodes = list(_bundle_all(latent_bundle, "companion"))
        if harmonic_nodes:
            for node in harmonic_nodes:
                if envelope is not None:
                    _add("harmonic_term", ("mul", envelope, node))
                else:
                    _add("harmonic_term", node)
        else:
            feature = _bundle_first(latent_bundle, "feature")
            if feature is not None:
                if envelope is not None:
                    _add("wrapped_feature", ("mul", envelope, feature))
                else:
                    _add("wrapped_feature", feature)
        _add_many("companion_term", companion_nodes)
    elif family_token in {"exp", "log"}:
        feature = _bundle_first(latent_bundle, "feature")
        anchor = _bundle_first(latent_bundle, "anchor")
        kind_key = f"{family_token}_kind"
        kind = str(direct_meta.get(kind_key, scaffold_meta.get(kind_key, "")) or "").strip().lower()
        if not kind:
            if form == f"{family_token}_add":
                kind = "add"
            elif form == f"{family_token}_mul":
                kind = "mul"
            else:
                kind = "base"
        if kind == "mul" and feature is not None and anchor is not None:
            _add("wrapped_feature", ("mul", feature, anchor))
        elif kind == "add":
            _add("wrapped_feature", feature)
            _add("companion_term", anchor)
        else:
            _add("wrapped_feature", feature)
    elif family_token == "affine":
        _add_many("affine_term", _bundle_all(latent_bundle, "term"))
    elif family_token == "rational":
        expr_node = _valid_node(expr)
        if expr_node is not None:
            _add("fractional_term", expr_node)
    elif family_token == "quadratic":
        expr_node = _valid_node(expr)
        if expr_node is not None:
            _add("quadratic_term", expr_node)
    elif family_token == "power":
        exponent = float(direct_meta.get("power_exponent", 0.0) or 0.0)
        variant_token = str(direct_meta.get("power_variant", "") or "").strip().lower()
        carrier = _bundle_first(latent_bundle, "carrier")
        anchor = _bundle_first(latent_bundle, "anchor")
        if anchor is None:
            anchor = _bundle_first(latent_bundle, "envelope")
        if exponent == 2.0 and carrier is not None:
            layouts: dict[str, tuple[str, ...]] = {
                "square_only": ("square",),
                "bias_square": ("bias", "square"),
                "linear_square": ("linear", "square"),
                "full_quadratic": ("bias", "linear", "square"),
            }
            active_terms = layouts.get(variant_token, layouts["full_quadratic"])
            if "bias" in active_terms and anchor is not None:
                _add("power_bias_term", anchor)
            if "linear" in active_terms:
                linear_term = simplify(("mul", anchor, carrier)) if anchor is not None else carrier
                _add("power_linear_term", linear_term)
            if "square" in active_terms:
                square_term = simplify(("sqr", carrier))
                if anchor is not None:
                    square_term = simplify(("mul", anchor, square_term))
                _add("power_square_term", square_term)
        else:
            expr_node = _valid_node(expr)
            if expr_node is not None:
                _add("power_term", expr_node)
    else:
        expr_node = _valid_node(expr)
        if expr_node is not None:
            _add("primary", expr_node)

    if not rows:
        expr_node = _valid_node(expr)
        if expr_node is not None:
            _add("primary", expr_node)
    return tuple(rows)


def ensure_feature_block_head_bundle(block: FeatureBlock | None) -> FeatureBlock | None:
    if not isinstance(block, FeatureBlock):
        return None
    explicit_nodes = tuple(_valid_node(node) for node in tuple(getattr(block, "head_bundle_nodes", ()) or ()))
    explicit_nodes = tuple(node for node in explicit_nodes if node is not None)
    explicit_roles = tuple(str(role) for role in tuple(getattr(block, "head_bundle_roles", ()) or ()))
    if explicit_nodes and len(explicit_nodes) == len(explicit_roles):
        return block

    rows: list[tuple[str, tuple]] = []
    seen: set[tuple[str, str]] = set()
    for role, node in zip(
        tuple(getattr(block, "latent_bundle_roles", ()) or ()),
        tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
    ):
        valid = _valid_node(node)
        if valid is None:
            continue
        key = (str(role), str(node_str(valid)))
        if key in seen:
            continue
        seen.add(key)
        rows.append((str(role), valid))

    if not rows:
        expr_node = _valid_node(dict(getattr(block, "metadata", {}) or {}).get("block_expr_obj", None))
        if expr_node is None:
            expr_node = _valid_node(dict(getattr(block, "metadata", {}) or {}).get("expr_obj", None))
        if expr_node is None:
            expr_node = _valid_node(tuple(getattr(block, "atoms", ()) or ())[0]) if tuple(getattr(block, "atoms", ()) or ()) else None
        if expr_node is not None:
            rows.append(("primary", expr_node))

    if not rows:
        return block

    head_bundle_roles = tuple(role for role, _ in rows)
    head_bundle_nodes = tuple(node for _, node in rows)
    metadata = dict(getattr(block, "metadata", {}) or {})
    metadata["head_bundle_roles"] = [str(role) for role in head_bundle_roles]
    metadata["head_bundle_exprs"] = [str(node_str(node)) for node in head_bundle_nodes]
    return _normalize_feature_block_graph(FeatureBlock(
        family=str(block.family),
        atoms=tuple(block.atoms),
        head_type=str(block.head_type),
        block_id=str(getattr(block, "block_id", "") or ""),
        parent_block_ids=tuple(getattr(block, "parent_block_ids", ()) or ()),
        latent_bundle_nodes=tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
        latent_bundle_roles=tuple(getattr(block, "latent_bundle_roles", ()) or ()),
        head_bundle_nodes=head_bundle_nodes,
        head_bundle_roles=head_bundle_roles,
        dim_signature=block.dim_signature,
        active_vars=tuple(getattr(block, "active_vars", ()) or ()),
        metadata=metadata,
    ))


def ensure_basis_state_head_bundles(state: BasisState | None) -> BasisState | None:
    if not isinstance(state, BasisState):
        return None
    blocks = topologically_order_feature_blocks(
        [
            normalized
            for normalized in (
                ensure_feature_block_head_bundle(block)
                for block in tuple(getattr(state, "blocks", ()) or ())
            )
            if isinstance(normalized, FeatureBlock)
        ],
        drop_orphans=True,
    )
    if blocks == tuple(getattr(state, "blocks", ()) or ()):
        return state
    return BasisState(
        blocks=blocks,
        fit_bundle=dict(getattr(state, "fit_bundle", {}) or {}),
        fit_loss=float(getattr(state, "fit_loss", math.inf)),
        probe_loss=float(getattr(state, "probe_loss", math.inf)),
        complexity=float(getattr(state, "complexity", math.inf)),
        residual_fit=getattr(state, "residual_fit", None),
        residual_probe=getattr(state, "residual_probe", None),
        residual_witness=getattr(state, "residual_witness", None),
        diagnostics=dict(getattr(state, "diagnostics", {}) or {}),
        provenance=tuple(getattr(state, "provenance", ()) or ()),
        compiled_expr=canonicalize_basis_expr(getattr(state, "compiled_expr", None)),
    )


def feature_block_from_closure_candidate(
    *,
    family: str,
    scaffold_id: str,
    expr: tuple | None,
    anchor_node: tuple | None,
    scaffold_metadata: Mapping[str, Any] | None,
    local_mapping_kind: str,
    local_mapping_coeffs: Sequence[float] | None = None,
    direct_metadata: Mapping[str, Any] | None = None,
    bound_closure: BoundClosure | None = None,
    binding_values: Mapping[str, Any] | None = None,
) -> FeatureBlock:
    direct_meta = dict(direct_metadata or {})
    scaffold_meta = dict(scaffold_metadata or {})
    atoms: list[tuple] = list(_atoms_from_bound_closure(bound_closure))

    if not atoms:
        for key in (
            "feature_node",
            "u_node",
            "v_node",
            "anchor_lift_node",
            "envelope_node",
            "quadratic_latent_node",
            "power_inner_node",
            "hole_node",
        ):
            node = _valid_node(direct_meta.get(key, None))
            if node is not None:
                atoms.append(node)
    if not atoms:
        for list_key in ("term_nodes", "harmonic_feature_nodes", "companion_nodes", "quadratic_base_nodes"):
            for raw_node in list(direct_meta.get(list_key, []) or ()):
                node = _valid_node(raw_node)
                if node is not None:
                    atoms.append(node)
    else:
        for list_key in ("term_nodes", "harmonic_feature_nodes", "companion_nodes", "quadratic_base_nodes"):
            for raw_node in list(direct_meta.get(list_key, []) or ()):
                node = _valid_node(raw_node)
                if node is not None:
                    atoms.append(node)

    if not atoms:
        for raw_node in list(_atoms_from_bound_closure(bound_closure) or ()):
            node = _valid_node(raw_node)
            if node is not None:
                atoms.append(node)

    anchor = _valid_node(anchor_node)
    form = str(scaffold_meta.get("form", "") or "").strip().lower()
    if anchor is not None and form.endswith(("_add", "_mul")):
        atoms.append(anchor)

    if not atoms:
        expr_node = _valid_node(expr)
        if expr_node is not None:
            atoms.append(expr_node)

    dedup_atoms: list[tuple] = []
    seen_atom_keys: set[str] = set()
    for node in atoms:
        key = str(node_str(node))
        if key in seen_atom_keys:
            continue
        seen_atom_keys.add(key)
        dedup_atoms.append(node)

    latent_bundle = _latent_bundle_from_candidate(
        bound_closure=bound_closure,
        direct_metadata=direct_meta,
        anchor_node=anchor_node,
        scaffold_form=form,
        expr=expr,
    )
    head_bundle = _head_bundle_from_candidate(
        family=family,
        latent_bundle=latent_bundle,
        local_mapping_kind=local_mapping_kind,
        scaffold_metadata=scaffold_meta,
        direct_metadata=direct_meta,
        expr=expr,
    )
    latent_bundle_nodes = tuple(node for _, node in latent_bundle)
    latent_bundle_roles = tuple(role for role, _ in latent_bundle)
    head_bundle_nodes = tuple(node for _, node in head_bundle)
    head_bundle_roles = tuple(role for role, _ in head_bundle)
    active_vars = _collect_active_vars((*dedup_atoms, *latent_bundle_nodes, *head_bundle_nodes))
    expr_node = _valid_node(expr)
    interaction_node = _interaction_node_from_bundle(
        family=family,
        latent_bundle=latent_bundle,
        direct_metadata=direct_meta,
        expr=expr,
    )
    anchor_diversity_node = _anchor_diversity_node_from_bundle(
        latent_bundle=latent_bundle,
        direct_metadata=direct_meta,
        anchor_node=anchor_node,
    )
    metadata = {
        "scaffold_id": str(scaffold_id),
        "scaffold_form": str(form),
        "local_mapping_kind": str(local_mapping_kind),
        "local_mapping_coeffs": [float(v) for v in list(local_mapping_coeffs or ())],
        "block_expr_obj": expr_node,
        "block_expr": str(node_str(expr_node)) if expr_node is not None else "",
        "head_bundle_exprs": [str(node_str(node)) for node in head_bundle_nodes if _valid_node(node) is not None],
        "head_bundle_roles": [str(role) for role in head_bundle_roles],
        "direct_metadata": dict(direct_meta),
        "bound_closure": bound_closure.to_dict() if isinstance(bound_closure, BoundClosure) else None,
        "bound_closure_obj": bound_closure if isinstance(bound_closure, BoundClosure) else None,
        "interaction_key": str(node_str(interaction_node)) if interaction_node is not None else "",
        "interaction_expr": str(node_str(interaction_node)) if interaction_node is not None else "",
        "anchor_diversity_key": (
            str(node_str(anchor_diversity_node)) if anchor_diversity_node is not None else ""
        ),
        "anchor_diversity_expr": (
            str(node_str(anchor_diversity_node)) if anchor_diversity_node is not None else ""
        ),
    }
    support_id = bound_closure_identity_key(bound_closure) or str(scaffold_id)
    basis_id = str(support_id)
    if str(family or "").strip().lower() == "power" and float(direct_meta.get("power_exponent", 0.0) or 0.0) == 2.0:
        head_mask_entries = [
            f"{str(role)}:{str(node_str(node))}"
            for role, node in head_bundle
            if _valid_node(node) is not None
        ]
        if head_mask_entries:
            basis_id = f"{support_id}::head::{ '|'.join(head_mask_entries) }"
    metadata["support_id"] = str(support_id)
    metadata["basis_id"] = str(basis_id)
    parent_block_ids = _binding_parent_block_ids(binding_values)
    return _normalize_feature_block_graph(FeatureBlock(
        family=str(family),
        atoms=tuple(dedup_atoms),
        head_type=_head_type_from_mapping(str(local_mapping_kind)),
        block_id=str(basis_id),
        parent_block_ids=tuple(parent_block_ids),
        latent_bundle_nodes=latent_bundle_nodes,
        latent_bundle_roles=latent_bundle_roles,
        head_bundle_nodes=head_bundle_nodes,
        head_bundle_roles=head_bundle_roles,
        active_vars=active_vars,
        metadata=metadata,
    ))


def basis_state_from_closure_candidate(
    *,
    family: str,
    scaffold_id: str,
    expr: tuple | None,
    anchor_node: tuple | None,
    scaffold_metadata: Mapping[str, Any] | None,
    local_fit_mse: float,
    local_probe_mse: float,
    local_mapping_kind: str,
    local_mapping_coeffs: Sequence[float] | None = None,
    direct_metadata: Mapping[str, Any] | None = None,
    bound_closure: BoundClosure | None = None,
    binding_values: Mapping[str, Any] | None = None,
) -> BasisState:
    block = feature_block_from_closure_candidate(
        family=family,
        scaffold_id=scaffold_id,
        expr=expr,
        anchor_node=anchor_node,
        scaffold_metadata=scaffold_metadata,
        local_mapping_kind=local_mapping_kind,
        local_mapping_coeffs=local_mapping_coeffs,
        direct_metadata=direct_metadata,
        bound_closure=bound_closure,
        binding_values=binding_values,
    )
    fit_bundle = {
        "mapping_kind": str(local_mapping_kind),
        "mapping_coeffs": [float(v) for v in list(local_mapping_coeffs or ())],
    }
    diagnostics = {
        "route": "closure_search",
        "scaffold_id": str(scaffold_id),
        "family": str(family),
    }
    return BasisState(
        blocks=topologically_order_feature_blocks((block,), drop_orphans=True),
        fit_bundle=fit_bundle,
        fit_loss=float(local_fit_mse),
        probe_loss=float(local_probe_mse),
        complexity=float(block.complexity()),
        diagnostics=diagnostics,
        provenance=(f"closure_search:{str(scaffold_id)}",),
        compiled_expr=canonicalize_basis_expr(expr),
    )


def basis_state_from_additive_transition(
    transition: Mapping[str, Any],
    *,
    base_state: BasisState | None = None,
    family: str = "basis",
    route_name: str = "score_head_direct_combo",
    fit_loss: float = math.inf,
    probe_loss: float = math.inf,
    compiled_expr: tuple | None = None,
) -> BasisState | None:
    if not isinstance(transition, Mapping):
        return None
    if str(transition.get("kind", "") or "") != "additive_basis_admission":
        return None

    core_expr = _valid_node(transition.get("core_expr", None))
    term_nodes = [
        node
        for node in list(transition.get("term_nodes", []) or [])
        if _valid_node(node) is not None
    ]
    coeffs = [float(v) for v in list(transition.get("coeffs", []) or [])]
    compiled_node = canonicalize_basis_expr(compiled_expr) or canonicalize_basis_expr(transition.get("compiled_expr", None))

    blocks: list[FeatureBlock] = list(getattr(base_state, "blocks", ()) or ())
    seen_atoms = {
        str(node_str(atom))
        for block in blocks
        for atom in tuple(getattr(block, "atoms", ()) or ())
        if _valid_node(atom) is not None
    }

    def _add_block(node: tuple | None, *, role: str, coeff: float) -> None:
        if _valid_node(node) is None:
            return
        key = str(node_str(node))
        if key in seen_atoms:
            return
        if math.isfinite(float(coeff)) and abs(float(coeff)) <= 1.0e-12:
            return
        block = _feature_block_from_node(
            family=str(family),
            node=node,
            head_type="linear",
            metadata={
                "route": str(route_name),
                "role": str(role),
                "coefficient": float(coeff),
            },
        )
        if block is None:
            return
        blocks.append(block)
        seen_atoms.add(key)

    if not blocks:
        coeff0 = float(coeffs[0]) if coeffs else 1.0
        _add_block(core_expr, role="core", coeff=coeff0)

    for idx, node in enumerate(term_nodes):
        coeff_idx = int(idx) + 1
        coeff = float(coeffs[coeff_idx]) if coeff_idx < len(coeffs) else 0.0
        _add_block(node, role="companion", coeff=coeff)

    bias_idx = int(len(term_nodes)) + 1
    intercept = float(coeffs[bias_idx]) if bias_idx < len(coeffs) else 0.0
    total_complexity = sum(float(block.complexity()) for block in blocks)
    fit_bundle = dict(getattr(base_state, "fit_bundle", {}) or {})
    fit_bundle["basis_transition"] = {
        "kind": str(transition.get("kind", "")),
        "coeffs": coeffs,
        "intercept": float(intercept),
        "ridge": float(transition.get("ridge", 0.0) or 0.0),
        "prune_rel": float(transition.get("prune_rel", 0.0) or 0.0),
    }
    diagnostics = dict(getattr(base_state, "diagnostics", {}) or {})
    diagnostics["route"] = str(route_name)
    diagnostics["basis_transition_kind"] = str(transition.get("kind", ""))
    provenance = tuple(getattr(base_state, "provenance", ()) or ())
    provenance = (*provenance, f"{str(route_name)}:additive_basis")
    return BasisState(
        blocks=topologically_order_feature_blocks(tuple(blocks), drop_orphans=True),
        fit_bundle=fit_bundle,
        fit_loss=float(fit_loss),
        probe_loss=float(probe_loss),
        complexity=float(total_complexity),
        residual_fit=getattr(base_state, "residual_fit", None),
        residual_probe=getattr(base_state, "residual_probe", None),
        residual_witness=getattr(base_state, "residual_witness", None),
        diagnostics=diagnostics,
        provenance=provenance,
        compiled_expr=compiled_node,
    )


def basis_state_extend(
    base_state: BasisState | None,
    added_state: BasisState | None,
    *,
    route_name: str = "basis_extend",
    fit_loss: float | None = None,
    probe_loss: float | None = None,
    compiled_expr: tuple | None = None,
) -> BasisState | None:
    if not isinstance(base_state, BasisState):
        return added_state if isinstance(added_state, BasisState) else None
    if not isinstance(added_state, BasisState):
        if fit_loss is None and probe_loss is None and compiled_expr is None:
            return base_state
        return BasisState(
            blocks=tuple(base_state.blocks),
            fit_bundle=dict(base_state.fit_bundle or {}),
            fit_loss=float(base_state.fit_loss if fit_loss is None else fit_loss),
            probe_loss=float(base_state.probe_loss if probe_loss is None else probe_loss),
            complexity=float(base_state.complexity),
            residual_fit=base_state.residual_fit,
            residual_probe=base_state.residual_probe,
            residual_witness=base_state.residual_witness,
            diagnostics=dict(base_state.diagnostics or {}),
            provenance=tuple(base_state.provenance or ()),
            compiled_expr=canonicalize_basis_expr(compiled_expr) or canonicalize_basis_expr(base_state.compiled_expr),
        )

    merged_blocks: list[FeatureBlock] = []
    seen_blocks: set[str] = set()
    for block in [*tuple(base_state.blocks or ()), *tuple(added_state.blocks or ())]:
        normalized = _normalize_feature_block_graph(block)
        if not isinstance(normalized, FeatureBlock):
            continue
        key = _block_identity(normalized)
        if not key or key in seen_blocks:
            continue
        seen_blocks.add(key)
        merged_blocks.append(normalized)

    merged_blocks_tuple = topologically_order_feature_blocks(tuple(merged_blocks), drop_orphans=True)

    total_complexity = sum(float(block.complexity()) for block in merged_blocks_tuple)
    diagnostics = dict(base_state.diagnostics or {})
    diagnostics.update(dict(added_state.diagnostics or {}))
    diagnostics["route"] = str(route_name)
    fit_bundle = dict(base_state.fit_bundle or {})
    fit_bundle.update(dict(added_state.fit_bundle or {}))
    provenance = (*tuple(base_state.provenance or ()), *tuple(added_state.provenance or ()), f"{str(route_name)}:merge")
    return BasisState(
        blocks=tuple(merged_blocks_tuple),
        fit_bundle=fit_bundle,
        fit_loss=float(added_state.fit_loss if fit_loss is None else fit_loss),
        probe_loss=float(added_state.probe_loss if probe_loss is None else probe_loss),
        complexity=float(total_complexity),
        residual_fit=added_state.residual_fit if added_state.residual_fit is not None else base_state.residual_fit,
        residual_probe=added_state.residual_probe if added_state.residual_probe is not None else base_state.residual_probe,
        residual_witness=(
            added_state.residual_witness if added_state.residual_witness is not None else base_state.residual_witness
        ),
        diagnostics=diagnostics,
        provenance=provenance,
        compiled_expr=(
            canonicalize_basis_expr(compiled_expr)
            or canonicalize_basis_expr(added_state.compiled_expr)
            or canonicalize_basis_expr(base_state.compiled_expr)
        ),
    )


def basis_state_covers_feature_block(state: BasisState | None, block: FeatureBlock | None) -> bool:
    if not isinstance(state, BasisState) or not isinstance(block, FeatureBlock):
        return False
    candidate = _normalize_feature_block_graph(block)
    if not isinstance(candidate, FeatureBlock):
        return False
    candidate_id = feature_block_id(candidate)
    candidate_family = str(getattr(candidate, "family", "") or "")
    candidate_head_type = str(getattr(candidate, "head_type", "") or "")
    candidate_atoms = _block_node_keys(candidate)
    candidate_latent = _role_node_keys(
        getattr(candidate, "latent_bundle_roles", ()),
        getattr(candidate, "latent_bundle_nodes", ()),
    )
    candidate_head = _role_node_keys(
        getattr(candidate, "head_bundle_roles", ()),
        getattr(candidate, "head_bundle_nodes", ()),
    )
    candidate_parents = set(feature_block_parent_ids(candidate))
    if not candidate_atoms and not candidate_latent and not candidate_head:
        return False

    for existing in tuple(state.blocks or ()):
        existing_norm = _normalize_feature_block_graph(existing)
        if not isinstance(existing_norm, FeatureBlock):
            continue
        if candidate_id and feature_block_id(existing_norm) == candidate_id:
            return True
        if candidate_family and str(getattr(existing_norm, "family", "") or "") != candidate_family:
            continue
        if candidate_head_type and str(getattr(existing_norm, "head_type", "") or "") != candidate_head_type:
            continue
        existing_atoms = _block_node_keys(existing_norm)
        existing_latent = _role_node_keys(
            getattr(existing_norm, "latent_bundle_roles", ()),
            getattr(existing_norm, "latent_bundle_nodes", ()),
        )
        existing_head = _role_node_keys(
            getattr(existing_norm, "head_bundle_roles", ()),
            getattr(existing_norm, "head_bundle_nodes", ()),
        )
        existing_parents = set(feature_block_parent_ids(existing_norm))
        if candidate_head or candidate_latent:
            if candidate_head and not candidate_head.issubset(existing_head):
                continue
            if candidate_latent and not candidate_latent.issubset(existing_latent):
                continue
            if candidate_atoms and not candidate_atoms.issubset(existing_atoms):
                continue
            if candidate_parents and not candidate_parents.issubset(existing_parents):
                continue
            return True
        if candidate_atoms == existing_atoms and (not candidate_parents or candidate_parents == existing_parents):
            return True
    return False


def admit_basis_state_to_beam(
    beam: Sequence[BasisState] | None,
    state: BasisState | None,
    *,
    beam_width: int = 3,
) -> tuple[BasisState, ...]:
    rows = [row for row in list(beam or ()) if isinstance(row, BasisState)]
    if isinstance(state, BasisState):
        rows.append(state)
    return prune_basis_state_beam(rows, beam_width=beam_width)


def enrich_closure_candidate_row(
    row: Mapping[str, Any],
    *,
    family: str,
    scaffold_id: str,
    anchor_node: tuple | None,
    scaffold_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(row)
    expr = _valid_node(enriched.get("expr", None))
    bound_closure = enriched.get("bound_closure_obj", None)
    operator_application = enriched.get("operator_application_obj", None)
    if not isinstance(bound_closure, BoundClosure) and operator_application is not None:
        candidate_closure = getattr(operator_application, "bound_closure", None)
        if isinstance(candidate_closure, BoundClosure):
            bound_closure = candidate_closure
    if not isinstance(bound_closure, BoundClosure):
        bound_closure = None
    block = feature_block_from_closure_candidate(
        family=family,
        scaffold_id=scaffold_id,
        expr=expr,
        anchor_node=anchor_node,
        scaffold_metadata=scaffold_metadata,
        local_mapping_kind=str(enriched.get("local_mapping_kind", "") or ""),
        local_mapping_coeffs=list(enriched.get("local_mapping_coeffs", []) or []),
        direct_metadata=dict(enriched.get("direct_metadata", {}) or {}),
        bound_closure=bound_closure,
        binding_values=getattr(operator_application, "bindings", None),
    )
    state = basis_state_from_closure_candidate(
        family=family,
        scaffold_id=scaffold_id,
        expr=expr,
        anchor_node=anchor_node,
        scaffold_metadata=scaffold_metadata,
        local_fit_mse=float(enriched.get("local_fit_mse", math.inf) or math.inf),
        local_probe_mse=float(enriched.get("local_probe_mse", math.inf) or math.inf),
        local_mapping_kind=str(enriched.get("local_mapping_kind", "") or ""),
        local_mapping_coeffs=list(enriched.get("local_mapping_coeffs", []) or []),
        direct_metadata=dict(enriched.get("direct_metadata", {}) or {}),
        bound_closure=bound_closure,
        binding_values=getattr(operator_application, "bindings", None),
    )
    enriched["feature_block_obj"] = block
    enriched["feature_block_dict"] = block.to_dict()
    enriched["basis_state_obj"] = state
    enriched["basis_state_dict"] = state.to_dict()
    if not isinstance(bound_closure, BoundClosure):
        bound_closure = bound_closure_from_closure_candidate(
            family=family,
            scaffold_id=scaffold_id,
            expr=expr,
            anchor_node=anchor_node,
            scaffold_metadata=scaffold_metadata,
            direct_metadata=dict(enriched.get("direct_metadata", {}) or {}),
        )
    enriched["bound_closure_obj"] = bound_closure
    enriched["bound_closure_dict"] = bound_closure.to_dict()
    proposal_key = str(
        enriched.get("proposal_key", "")
        or bound_closure_identity_key(bound_closure)
        or enriched.get("child_key", "")
    )
    enriched["proposal_key"] = proposal_key
    candidate = ProposalCandidate(
        family=str(family),
        rendered_expr=expr,
        scaffold_id=str(scaffold_id),
        identity_key=str(proposal_key),
        feature_block=block,
        basis_state=state,
        bound_closure=bound_closure,
        local_fit_loss=float(enriched.get("local_fit_mse", math.inf) or math.inf),
        local_probe_loss=float(enriched.get("local_probe_mse", math.inf) or math.inf),
        complexity=float(block.complexity()),
        metadata={
            "route": "closure_search",
            "scaffold_metadata": dict(scaffold_metadata or {}),
            "local_mapping_kind": str(enriched.get("local_mapping_kind", "") or ""),
            "proposal_key": str(proposal_key),
            "rendered_expr": str(node_str(expr)) if isinstance(expr, tuple) else "",
            "child_key": str(enriched.get("child_key", "") or ""),
        },
    )
    enriched["proposal_candidate_obj"] = candidate
    enriched["proposal_candidate_dict"] = candidate.to_dict()
    return enriched


__all__ = [
    "BasisState",
    "FeatureBlock",
    "ProposalCandidate",
    "ProposalContext",
    "ProposalFamily",
    "admit_basis_state_to_beam",
    "closure_keep_feature_blocks",
    "drop_feature_block_with_dependents",
    "ensure_basis_state_head_bundles",
    "ensure_feature_block_head_bundle",
    "basis_state_retarget",
    "basis_state_covers_feature_block",
    "basis_state_from_additive_transition",
    "basis_state_from_closure_candidate",
    "basis_state_extend",
    "feature_block_id",
    "feature_block_parent_ids",
    "prepare_basis_state_candidate",
    "prune_basis_state_beam",
    "enrich_closure_candidate_row",
    "feature_block_from_closure_candidate",
    "topologically_order_feature_blocks",
]
