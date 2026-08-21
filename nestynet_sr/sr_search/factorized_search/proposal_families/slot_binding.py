# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..closures import BoundClosure, SlotSpec
from ..expr_ast import dims_eq, is_valid_node, node_str
from .common import dim0
from .seed_blocks import SeedBlock, make_seed_block


def _dim_rule_matches(dim_rule: Any, block_dim: Any, *, dim0_value: Any) -> bool:
    if dim_rule is None:
        return True
    if isinstance(dim_rule, str):
        token = str(dim_rule).strip().lower()
        if token in {"scalar", "dimless", "dimensionless"}:
            if dim0_value is None:
                return True
            return block_dim is not None and dims_eq(block_dim, dim0_value)
        return True
    if block_dim is None:
        return False
    try:
        return dims_eq(block_dim, dim_rule)
    except Exception:
        return False


def _domain_rule_matches(domain_rule: Any, block: SeedBlock) -> bool:
    if domain_rule is None:
        return True
    tags = set(str(v) for v in tuple(block.domain_tags or ()))
    if not tags:
        return True
    if isinstance(domain_rule, str):
        return str(domain_rule) in tags
    if isinstance(domain_rule, (list, tuple, set)):
        needed = {str(v) for v in domain_rule}
        return needed.issubset(tags)
    return True


def _slot_metadata_tokens(metadata: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    for key in keys:
        raw = metadata.get(str(key), None)
        if raw is None:
            continue
        if isinstance(raw, str):
            token = str(raw).strip()
            return (token,) if token else ()
        return tuple(str(v).strip() for v in tuple(raw or ()) if str(v).strip())
    return ()


def _slot_metadata_int(metadata: Mapping[str, Any], key: str) -> int | None:
    raw = metadata.get(str(key), None)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _block_metadata_int(block: SeedBlock, key: str, default: int = 0) -> int:
    try:
        raw = dict(block.metadata or {}).get(str(key), default)
    except Exception:
        raw = default
    try:
        return int(raw)
    except Exception:
        return int(default)


def _slot_metadata_matches(slot_spec: SlotSpec, block: SeedBlock) -> bool:
    meta = dict(getattr(slot_spec, "metadata", {}) or {})
    if not meta:
        return True
    builder = str(getattr(block, "builder", "") or "")
    allowed_builders = set(_slot_metadata_tokens(meta, "allowed_builders"))
    if allowed_builders and builder not in allowed_builders:
        return False
    blocked_builders = set(_slot_metadata_tokens(meta, "disallow_builders", "forbidden_builders"))
    if blocked_builders and builder in blocked_builders:
        return False
    max_builder_depth = _slot_metadata_int(meta, "max_builder_depth")
    if max_builder_depth is not None and _block_metadata_int(block, "builder_depth") > int(max_builder_depth):
        return False
    max_nonlinear_depth = _slot_metadata_int(meta, "max_nonlinear_depth")
    if max_nonlinear_depth is not None and _block_metadata_int(block, "nonlinear_depth") > int(max_nonlinear_depth):
        return False
    max_product_arity = _slot_metadata_int(meta, "max_product_arity")
    if max_product_arity is not None and _block_metadata_int(block, "product_arity", default=1) > int(max_product_arity):
        return False
    if bool(meta.get("require_recursive", False)) and _block_metadata_int(block, "builder_depth") <= 0:
        return False
    allowed_origin_prefixes = _slot_metadata_tokens(meta, "allowed_origin_prefixes")
    if allowed_origin_prefixes:
        origin = str(dict(block.metadata or {}).get("origin", "") or block.source or "")
        if not any(origin.startswith(prefix) for prefix in allowed_origin_prefixes):
            return False
    basis_role_prefixes = _slot_metadata_tokens(meta, "allowed_basis_role_prefixes")
    if basis_role_prefixes:
        basis_role = str(dict(block.metadata or {}).get("basis_role", "") or "")
        if not any(basis_role.startswith(prefix) for prefix in basis_role_prefixes):
            return False
    return True


def _binding_blocks(value: Any, *, var_dims: Sequence[Sequence[float]] | None = None) -> list[SeedBlock]:
    if isinstance(value, SeedBlock):
        return [value]
    if isinstance(value, tuple) and is_valid_node(value):
        dim = None
        if var_dims is not None:
            try:
                from ..expr_ast import node_dims

                dim = node_dims(value, var_dims)
            except Exception:
                dim = None
        return [make_seed_block(value, dim=dim, source="binding")]
    out: list[SeedBlock] = []
    for raw in list(value or ()):
        out.extend(_binding_blocks(raw, var_dims=var_dims))
    return out


def _binding_node_keys(value: Any, *, var_dims: Sequence[Sequence[float]] | None = None) -> tuple[str, ...]:
    return tuple(str(node_str(block.node)) for block in _binding_blocks(value, var_dims=var_dims))


def _reuse_policy_matches(
    reuse_policy: str | None,
    value: Any,
    *,
    existing_bindings: Mapping[str, Any] | None = None,
    var_dims: Sequence[Sequence[float]] | None = None,
) -> bool:
    policy = str(reuse_policy or "allow").strip().lower()
    if policy in {"", "allow"}:
        return True
    value_keys = _binding_node_keys(value, var_dims=var_dims)
    if not value_keys:
        return True
    value_key_set = set(value_keys)
    if policy in {"pairwise_distinct", "distinct"} and len(value_key_set) != len(value_keys):
        return False
    if policy == "pairwise_distinct":
        return True
    refs: list[str] = []
    if policy == "distinct":
        refs = [str(key) for key in dict(existing_bindings or {}).keys()]
    elif policy.startswith("distinct_from:"):
        refs = [str(token).strip() for token in policy.split(":", 1)[1].split(",") if str(token).strip()]
    else:
        return True
    for ref in refs:
        ref_value = dict(existing_bindings or {}).get(ref, None)
        if ref_value is None:
            continue
        if value_key_set.intersection(_binding_node_keys(ref_value, var_dims=var_dims)):
            return False
    return True


def bind_slot_candidates(
    slot_spec: SlotSpec,
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    var_dims: Sequence[Sequence[float]] | None = None,
    limit: int | None = None,
    existing_bindings: Mapping[str, Any] | None = None,
) -> list[SeedBlock]:
    rows = list(seed_blocks or ())
    dim0_value = dim0(var_dims)
    out: list[SeedBlock] = []
    for block in rows:
        if not _dim_rule_matches(slot_spec.dim_rule, block.dim, dim0_value=dim0_value):
            continue
        if not _domain_rule_matches(slot_spec.domain_rule, block):
            continue
        if not _slot_metadata_matches(slot_spec, block):
            continue
        if slot_spec.arity_cap is not None and len(tuple(block.active_vars or ())) > int(slot_spec.arity_cap):
            continue
        if not _reuse_policy_matches(
            slot_spec.reuse_policy,
            block,
            existing_bindings=existing_bindings,
            var_dims=var_dims,
        ):
            continue
        out.append(block)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def pick_placeholder_block(
    *,
    desired_dim: Any,
    seed_blocks: Sequence[SeedBlock] | None,
    var_dims: Sequence[Sequence[float]] | None = None,
) -> SeedBlock | None:
    slot_spec = SlotSpec(name="placeholder", role="carrier", dim_rule=desired_dim)
    rows = bind_slot_candidates(
        slot_spec,
        seed_blocks,
        var_dims=var_dims,
        limit=1,
    )
    return rows[0] if rows else None


def family_anchor_blocks(
    family: str,
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    anchor_cap: int,
) -> list[SeedBlock]:
    rows = list(seed_blocks or ())
    if int(anchor_cap) <= 0:
        return []
    family_token = str(family or "").strip().lower()
    if family_token not in {"exp", "log", "quadratic", "power"}:
        return rows[: max(0, int(anchor_cap))]

    out: list[SeedBlock] = []
    seen: set[str] = set()

    def _add(block: SeedBlock) -> None:
        if len(out) >= int(anchor_cap):
            return
        key = str(node_str(block.node))
        if key in seen:
            return
        out.append(block)
        seen.add(key)

    for block in rows:
        if isinstance(block.node, tuple) and block.node and str(block.node[0]) == "var":
            _add(block)
    for block in rows:
        if block.node == ("const", 1.0):
            _add(block)
    for block in rows:
        _add(block)
    return out[: max(0, int(anchor_cap))]


def binding_snapshot(bindings: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(bindings or {}).items():
        if isinstance(value, SeedBlock):
            out[str(key)] = value.to_dict()
        elif isinstance(value, (list, tuple)):
            rows = []
            for raw in list(value):
                if isinstance(raw, SeedBlock):
                    rows.append(raw.to_dict())
                else:
                    rows.append(raw)
            out[str(key)] = rows
        else:
            out[str(key)] = value
    return out


def validate_bound_closure_bindings(
    bound_closure: BoundClosure | None,
    *,
    var_dims: Sequence[Sequence[float]] | None = None,
    binding_values: Mapping[str, Any] | None = None,
) -> bool:
    if not isinstance(bound_closure, BoundClosure):
        return False
    bindings = dict(binding_values or bound_closure.bindings or {})
    accepted: dict[str, Any] = {}
    for slot_spec in tuple(getattr(bound_closure.spec, "slot_specs", ()) or ()):
        slot_name = str(getattr(slot_spec, "name", "") or "")
        if not slot_name:
            continue
        if slot_name not in bindings:
            continue
        value = bindings.get(slot_name, None)
        blocks = _binding_blocks(value, var_dims=var_dims)
        if isinstance(value, (list, tuple)) and not (isinstance(value, tuple) and is_valid_node(value)):
            if slot_spec.arity_cap is not None and len(blocks) > int(slot_spec.arity_cap):
                return False
        for block in blocks:
            if not _dim_rule_matches(slot_spec.dim_rule, block.dim, dim0_value=dim0(var_dims)):
                return False
            if not _domain_rule_matches(slot_spec.domain_rule, block):
                return False
            if not _slot_metadata_matches(slot_spec, block):
                return False
        if not _reuse_policy_matches(
            slot_spec.reuse_policy,
            value,
            existing_bindings=accepted,
            var_dims=var_dims,
        ):
            return False
        accepted[slot_name] = value
    return True


__all__ = [
    "bind_slot_candidates",
    "binding_snapshot",
    "family_anchor_blocks",
    "pick_placeholder_block",
    "validate_bound_closure_bindings",
]
