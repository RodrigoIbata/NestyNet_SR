# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

from ..basis_head import basis_block_head_terms
from ..basis_state import (
    BasisState,
    FeatureBlock,
    ensure_feature_block_head_bundle,
    feature_block_id,
    feature_block_parent_ids,
    topologically_order_feature_blocks,
)
from ..expr_ast import dims_eq, is_valid_node, node_dims, node_size, node_str, simplify
from .common import anchor_priority, dim0, dim_scale, node_var_count


def _collect_active_vars(node: tuple | None) -> tuple[int, ...]:
    if not isinstance(node, tuple) or not node:
        return ()
    seen: set[int] = set()

    def _visit(cur: Any) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        if str(cur[0]) == "var":
            try:
                seen.add(int(cur[1]))
            except Exception:
                pass
            return
        for child in cur[1:]:
            if isinstance(child, tuple):
                _visit(child)

    _visit(node)
    return tuple(sorted(seen))


def _domain_tags_for_node(node: tuple) -> tuple[str, ...]:
    try:
        op = str(node[0])
    except Exception:
        return ()
    if op == "const":
        try:
            value = float(node[1])
        except Exception:
            return ()
        if value > 0.0:
            return ("positive_output", "nonnegative_output")
        if value == 0.0:
            return ("nonnegative_output",)
        return ()
    if op == "exp":
        return ("positive_output", "nonnegative_output")
    if op in {"sqrt", "sqr"}:
        return ("nonnegative_output",)
    if op in {"add", "mul"}:
        child_rows = [
            set(_domain_tags_for_node(child))
            for child in node[1:]
            if isinstance(child, tuple) and child
        ]
        if not child_rows:
            return ()
        if all("positive_output" in row for row in child_rows):
            return ("positive_output", "nonnegative_output")
        if all(("positive_output" in row) or ("nonnegative_output" in row) for row in child_rows):
            return ("nonnegative_output",)
    return ()


def _merge_domain_tags(*tag_rows: Sequence[str] | None) -> tuple[str, ...]:
    tags: set[str] = set()
    for row in tag_rows:
        for token in tuple(row or ()):
            token_str = str(token or "").strip()
            if token_str:
                tags.add(token_str)
    return tuple(sorted(tags))


def _product_domain_tags(blocks: Sequence["SeedBlock"]) -> tuple[str, ...]:
    rows = [set(str(v) for v in tuple(getattr(block, "domain_tags", ()) or ())) for block in blocks]
    if not rows:
        return ()
    tags: set[str] = set()
    if all("positive_output" in row for row in rows):
        tags.update({"positive_output", "nonnegative_output"})
    elif all(("positive_output" in row) or ("nonnegative_output" in row) for row in rows):
        tags.add("nonnegative_output")
    return tuple(sorted(tags))


def _monomial_domain_tags(block: "SeedBlock", exponent: float) -> tuple[str, ...] | None:
    tags = set(str(v) for v in tuple(getattr(block, "domain_tags", ()) or ()))
    exponent_f = float(exponent)
    if exponent_f == 2.0:
        return ("nonnegative_output",)
    if exponent_f == 0.5:
        if tags and "nonnegative_output" not in tags and "positive_output" not in tags:
            return None
        return ("nonnegative_output",)
    if exponent_f in {-1.0, -0.5, -2.0}:
        if tags and "positive_output" not in tags:
            return None
        return ("positive_output", "nonnegative_output")
    return tuple(sorted(tags))


def _quadratic_domain_tags(_blocks: Sequence["SeedBlock"]) -> tuple[str, ...]:
    return ("nonnegative_output",)


def _matching_dim_index(rows: Sequence[Any], dim: Any) -> int | None:
    for idx, existing in enumerate(list(rows or ())):
        if existing is None and dim is None:
            return int(idx)
        if existing is not None and dim is not None and dims_eq(existing, dim):
            return int(idx)
    return None


def _append_unique_dim(out: list[Any], dim: Any) -> None:
    if dim is None:
        return
    if _matching_dim_index(out, dim) is None:
        out.append(dim)


def _infer_nonlinear_depth(node: Any) -> int:
    if not isinstance(node, tuple) or not node:
        return 0
    op = str(node[0])
    child_depth = 0
    for child in node[1:]:
        child_depth = max(child_depth, _infer_nonlinear_depth(child))
    if op in {"sqrt", "sqr", "exp", "log", "sin", "cos"}:
        return child_depth + 1
    if op == "div" and len(node) >= 3:
        numerator = node[1]
        if isinstance(numerator, tuple) and numerator and str(numerator[0]) == "const":
            try:
                if float(numerator[1]) == 1.0:
                    return child_depth + 1
            except Exception:
                pass
    return child_depth


def _infer_product_arity(node: Any) -> int:
    if not isinstance(node, tuple) or not node:
        return 0
    op = str(node[0])
    if op == "const":
        return 0
    if op == "mul":
        return sum(max(1, _infer_product_arity(child)) for child in node[1:] if isinstance(child, tuple))
    return 1


def _seed_builder_depth(block: SeedBlock) -> int:
    try:
        return max(0, int(dict(block.metadata or {}).get("builder_depth", 0)))
    except Exception:
        return 0


def _seed_nonlinear_depth(block: SeedBlock) -> int:
    meta = dict(block.metadata or {})
    try:
        return max(0, int(meta.get("nonlinear_depth", _infer_nonlinear_depth(block.node))))
    except Exception:
        return max(0, _infer_nonlinear_depth(block.node))


def _seed_product_arity(block: SeedBlock) -> int:
    meta = dict(block.metadata or {})
    try:
        return max(0, int(meta.get("product_arity", _infer_product_arity(block.node))))
    except Exception:
        return max(0, _infer_product_arity(block.node))


@dataclass(frozen=True)
class SeedBlock:
    node: tuple
    dim: Any = None
    source: str = ""
    builder: str = "identity"
    active_vars: tuple[int, ...] = ()
    domain_tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr": str(node_str(self.node)),
            "dim": self.dim,
            "source": str(self.source),
            "builder": str(self.builder),
            "active_vars": [int(v) for v in self.active_vars],
            "domain_tags": [str(v) for v in self.domain_tags],
            "metadata": dict(self.metadata or {}),
        }


def _seed_block_dim(block: SeedBlock, var_dims) -> Any:
    if getattr(block, "dim", None) is not None:
        return block.dim
    if var_dims is None:
        return None
    try:
        return node_dims(block.node, var_dims)
    except Exception:
        return None


def _quadratic_base_sort_key(block: SeedBlock) -> tuple[int, int, int, int, str]:
    node = block.node
    op = str(node[0]) if isinstance(node, tuple) and node else ""
    builder = str(getattr(block, "builder", "") or "")
    source = str(getattr(block, "source", "") or "")
    try:
        size = max(1, int(node_size(node)))
    except Exception:
        size = 99
    try:
        uniq_vars = max(0, int(node_var_count(node)))
    except Exception:
        uniq_vars = 0
    builder_depth = _seed_builder_depth(block)
    key = str(node_str(node))
    if op == "var":
        rank = 0
    elif builder == "identity" and op not in {"sqrt", "sqr", "exp", "log", "sin", "cos"}:
        rank = 1
    elif source == "var":
        rank = 2
    else:
        rank = 3
    return (rank, builder_depth, size, -uniq_vars, key)


def _required_quadratic_base_dims(required_expr_dims: Sequence[Any] | None) -> list[Any]:
    out: list[Any] = []
    for expr_dim in list(required_expr_dims or ()):
        _append_unique_dim(out, dim_scale(expr_dim, 0.5))
    return out


def make_seed_block(
    node: tuple,
    *,
    dim: Any = None,
    source: str = "",
    builder: str = "identity",
    metadata: Mapping[str, Any] | None = None,
) -> SeedBlock:
    simp = simplify(node)
    meta = dict(metadata or {})
    meta.setdefault("builder_depth", 0)
    meta.setdefault("nonlinear_depth", _infer_nonlinear_depth(simp))
    meta.setdefault("product_arity", _infer_product_arity(simp))
    domain_tags = _merge_domain_tags(_domain_tags_for_node(simp), meta.get("domain_tags", ()))
    return SeedBlock(
        node=simp,
        dim=dim,
        source=str(source),
        builder=str(builder),
        active_vars=_collect_active_vars(simp),
        domain_tags=domain_tags,
        metadata=meta,
    )


def _basis_block_seed_rows(block: FeatureBlock | None) -> list[tuple[str, tuple, str]]:
    block = ensure_feature_block_head_bundle(block)
    if not isinstance(block, FeatureBlock):
        return []
    out: list[tuple[str, tuple, str]] = []
    seen: set[str] = set()

    def _add(role: str, node: Any, builder: str) -> None:
        if not isinstance(node, tuple) or not is_valid_node(node):
            return
        simp = simplify(node)
        key = str(node_str(simp))
        if key in seen:
            return
        seen.add(key)
        out.append((str(role), simp, str(builder)))

    for role, expr in basis_block_head_terms(block):
        _add(f"head:{str(role)}", expr, "basis_head")
    for role, node in zip(
        tuple(getattr(block, "latent_bundle_roles", ()) or ()),
        tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
    ):
        _add(f"bundle:{str(role)}", node, "basis_latent")
    for atom in tuple(getattr(block, "atoms", ()) or ()):
        _add("atom", atom, "basis_atom")
    return out


def seed_blocks_from_basis_state(
    state: BasisState | None,
    *,
    var_dims: Sequence[Sequence[float]] | None = None,
    limit: int | None = None,
    source_prefix: str = "basis:active",
) -> list[SeedBlock]:
    if not isinstance(state, BasisState):
        return []
    out: list[SeedBlock] = []
    seen: set[str] = set()
    for block_idx, block in enumerate(
        topologically_order_feature_blocks(tuple(getattr(state, "blocks", ()) or ()), drop_orphans=True)
    ):
        if not isinstance(block, FeatureBlock):
            continue
        block_family = str(getattr(block, "family", "") or "")
        block_id = feature_block_id(block)
        parent_block_ids = feature_block_parent_ids(block)
        for role, node, builder in _basis_block_seed_rows(block):
            key = str(node_str(node))
            if key in seen:
                continue
            dim = None
            if var_dims is not None:
                try:
                    dim = node_dims(node, var_dims)
                except Exception:
                    dim = None
            out.append(
                make_seed_block(
                    node,
                    dim=dim,
                    source=f"{str(source_prefix)}:{block_family}:{role}",
                    builder=builder,
                    metadata={
                        "origin": str(source_prefix),
                        "basis_family": block_family,
                        "basis_block_index": int(block_idx),
                        "basis_block_id": str(block_id),
                        "basis_parent_block_ids": [str(v) for v in parent_block_ids],
                        "basis_role": str(role),
                        "basis_head_type": str(getattr(block, "head_type", "") or ""),
                    },
                )
            )
            seen.add(key)
            if limit is not None and len(out) >= int(limit):
                return out
    return out


def extend_seed_blocks_with_basis(
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    basis_state: BasisState | None = None,
    basis_state_beam: Sequence[BasisState] | None = None,
    var_dims: Sequence[Sequence[float]] | None = None,
    limit: int | None = None,
    basis_limit: int | None = None,
    append_basis: bool = True,
) -> list[SeedBlock]:
    core_rows = [block for block in list(seed_blocks or ()) if isinstance(block, SeedBlock)]
    basis_rows: list[SeedBlock] = []
    basis_cap = basis_limit
    if basis_cap is None:
        basis_cap = limit

    def _remaining_basis_cap() -> int | None:
        if basis_cap is None:
            return None
        return max(0, int(basis_cap) - len(basis_rows))

    active_limit = _remaining_basis_cap()
    if active_limit is None or active_limit > 0:
        basis_rows.extend(
            seed_blocks_from_basis_state(
                basis_state,
                var_dims=var_dims,
                limit=active_limit,
                source_prefix="basis:active",
            )
        )
    if basis_cap is None or len(basis_rows) < int(basis_cap):
        for beam_idx, state in enumerate(tuple(basis_state_beam or ())):
            if state is basis_state or not isinstance(state, BasisState):
                continue
            remaining = _remaining_basis_cap()
            if remaining == 0:
                break
            basis_rows.extend(
                seed_blocks_from_basis_state(
                    state,
                    var_dims=var_dims,
                    limit=remaining,
                    source_prefix=f"basis:beam{int(beam_idx)}",
                )
            )
            if basis_cap is not None and len(basis_rows) >= int(basis_cap):
                break

    ordered_rows = [*core_rows, *basis_rows] if append_basis else [*basis_rows, *core_rows]
    out: list[SeedBlock] = []
    seen: set[str] = set()
    for block in ordered_rows:
        if not isinstance(block, SeedBlock):
            continue
        key = str(node_str(block.node))
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def seed_anchor_blocks(
    *,
    nvars: int,
    pool_nodes: Sequence[tuple] | None,
    pool_dims: Sequence[Any] | None,
    var_dims: Sequence[Sequence[float]] | None,
    max_count: int,
) -> list[SeedBlock]:
    out: list[SeedBlock] = []
    seen: set[str] = set()
    dm = bool(var_dims)
    dim0_value = dim0(var_dims)

    def _add(node, *, dim=None, source: str, builder: str = "identity", metadata=None) -> None:
        if not isinstance(node, tuple) or not node:
            return
        if not is_valid_node(node):
            return
        simp = simplify(node)
        if not isinstance(simp, tuple) or not is_valid_node(simp):
            return
        key = node_str(simp)
        if key in seen:
            return
        local_dim = dim
        if local_dim is None and dm:
            try:
                local_dim = node_dims(simp, var_dims)
            except Exception:
                local_dim = None
        out.append(
            make_seed_block(
                simp,
                dim=local_dim,
                source=source,
                builder=builder,
                metadata=metadata,
            )
        )
        seen.add(key)

    _add(("const", 1.0), dim=dim0_value, source="const")
    for idx in range(max(0, int(nvars))):
        local_dim = None
        if dm and idx < len(var_dims):
            local_dim = tuple(var_dims[idx])
        _add(("var", idx), dim=local_dim, source="var")

    pool_nodes_list = list(pool_nodes or ())
    pool_dims_list = list(pool_dims or ())
    pool_cap = max(0, int(max_count)) * 16
    if pool_cap <= 0:
        pool_cap = 32
    for idx, node in enumerate(pool_nodes_list):
        if idx >= pool_cap:
            break
        try:
            if int(node_size(node)) > 3:
                continue
        except Exception:
            continue
        dim = pool_dims_list[idx] if idx < len(pool_dims_list) else None
        _add(node, dim=dim, source="pool")

    out.sort(key=lambda row: anchor_priority(row.node))
    return out


def generate_product_seed_blocks(
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    max_arity: int = 2,
    limit: int = 32,
    max_builder_depth: int = 2,
    max_nonlinear_depth: int = 1,
) -> list[SeedBlock]:
    if max_arity < 2:
        return []
    rows = list(seed_blocks or ())[: max(4, int(limit) * 2)]
    out: list[SeedBlock] = []
    seen: set[str] = set()
    for arity in range(2, max(2, int(max_arity)) + 1):
        if len(out) >= int(limit):
            break
        for combo in combinations(rows, arity):
            if len(out) >= int(limit):
                break
            combo_product_arity = sum(_seed_product_arity(block) for block in combo)
            if combo_product_arity < 2 or combo_product_arity > int(max_arity):
                continue
            combo_builder_depth = max((_seed_builder_depth(block) for block in combo), default=0) + 1
            combo_nonlinear_depth = max((_seed_nonlinear_depth(block) for block in combo), default=0)
            if combo_builder_depth > int(max_builder_depth):
                continue
            if combo_nonlinear_depth > int(max_nonlinear_depth):
                continue
            expr = combo[0].node
            for block in combo[1:]:
                expr = simplify(("mul", expr, block.node))
            key = str(node_str(expr))
            if key in seen or not is_valid_node(expr):
                continue
            seen.add(key)
            out.append(
                make_seed_block(
                    expr,
                    source="*".join(str(block.source or block.builder or "seed") for block in combo),
                    builder="product",
                    metadata={
                        "children": [block.to_dict() for block in combo],
                        "builder_depth": combo_builder_depth,
                        "nonlinear_depth": combo_nonlinear_depth,
                        "product_arity": combo_product_arity,
                        "domain_tags": _product_domain_tags(combo),
                    },
                )
            )
    return out


def generate_monomial_seed_blocks(
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    exponents: Sequence[float] = (-1.0, -0.5, 0.5, 1.0, 2.0),
    limit: int = 64,
    max_builder_depth: int = 2,
    max_nonlinear_depth: int = 1,
) -> list[SeedBlock]:
    rows = list(seed_blocks or ())

    def _is_ratio_like_node(node: tuple | None) -> bool:
        if not (isinstance(node, tuple) and node):
            return False
        if str(node[0]) != "div" or len(node) < 3:
            return False
        left, right = node[1], node[2]
        if not (isinstance(left, tuple) and left and isinstance(right, tuple) and right):
            return False
        if str(left[0]) == "const" or str(right[0]) == "const":
            return False
        return True

    def _monomial_input_sort_key(block: SeedBlock) -> tuple[int, int, int, int, str]:
        node = block.node
        builder = str(getattr(block, "builder", "") or "")
        try:
            size = max(1, int(node_size(node)))
        except Exception:
            size = 99
        try:
            uniq_vars = max(0, int(node_var_count(node)))
        except Exception:
            uniq_vars = 0
        if _is_ratio_like_node(node):
            ratio_rank = 0
        else:
            ratio_rank = 1
        builder_rank = {
            "identity": 0,
            "basis_head": 1,
            "basis_atom": 1,
            "product": 2,
            "basis_latent": 2,
            "monomial": 3,
            "quadratic": 4,
            "affine": 5,
        }.get(builder, 6)
        return (ratio_rank, builder_rank, size, -uniq_vars, str(node_str(node)))

    # Stable sort: ratio-like nodes float to the front while preserving
    # relative order within each partition so that the existing seed-pool
    # ordering (important for periodic/quadratic searches) is respected.
    ratio_rows = [b for b in rows if _is_ratio_like_node(b.node)]
    other_rows = [b for b in rows if not _is_ratio_like_node(b.node)]
    rows = (ratio_rows + other_rows)[: max(4, int(limit) * 2)]
    out: list[SeedBlock] = []
    seen: set[str] = set()
    for block in rows:
        if len(out) >= int(limit):
            break
        for exponent in list(exponents or ()):
            if len(out) >= int(limit):
                break
            if float(exponent) == 1.0:
                continue
            builder_depth = _seed_builder_depth(block) + 1
            nonlinear_depth = _seed_nonlinear_depth(block) + 1
            if builder_depth > int(max_builder_depth):
                continue
            if nonlinear_depth > int(max_nonlinear_depth):
                continue
            expr = None
            if float(exponent) == 2.0:
                expr = simplify(("sqr", block.node))
            elif float(exponent) == 0.5:
                expr = simplify(("sqrt", block.node))
            elif float(exponent) == -1.0:
                expr = simplify(("div", ("const", 1.0), block.node))
            elif float(exponent) == -0.5:
                expr = simplify(("div", ("const", 1.0), ("sqrt", block.node)))
            if not (isinstance(expr, tuple) and is_valid_node(expr)):
                continue
            domain_tags = _monomial_domain_tags(block, float(exponent))
            if domain_tags is None:
                continue
            key = str(node_str(expr))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                make_seed_block(
                    expr,
                    source=block.source,
                    builder="monomial",
                    metadata={
                        "base": block.to_dict(),
                        "exponent": float(exponent),
                        "builder_depth": builder_depth,
                        "nonlinear_depth": nonlinear_depth,
                        "product_arity": _seed_product_arity(block),
                        "domain_tags": domain_tags,
                    },
                )
            )
    return out


def generate_quadratic_seed_blocks(
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    limit: int = 16,
    max_arity: int = 3,
    max_builder_depth: int = 2,
    max_nonlinear_depth: int = 1,
    var_dims: Sequence[Sequence[float]] | None = None,
    required_expr_dims: Sequence[Any] | None = None,
    bucket_row_cap: int | None = None,
    per_dim_limit: int | None = None,
) -> list[SeedBlock]:
    rows = [block for block in list(seed_blocks or ()) if isinstance(block, SeedBlock)]
    limit_i = max(0, int(limit))
    if limit_i <= 0 or not rows:
        return []

    required_expr_dims_norm: list[Any] = []
    for expr_dim in list(required_expr_dims or ()):
        _append_unique_dim(required_expr_dims_norm, expr_dim)
    required_base_dims = _required_quadratic_base_dims(required_expr_dims_norm)

    bucket_dims: list[Any] = []
    bucket_rows: list[list[SeedBlock]] = []
    for block in rows:
        if isinstance(block.node, tuple) and block.node and str(block.node[0]) == "const":
            continue
        if max(0, node_var_count(block.node)) > 2:
            continue
        block_dim = _seed_block_dim(block, var_dims)
        if required_base_dims:
            if block_dim is None or _matching_dim_index(required_base_dims, block_dim) is None:
                continue
        bucket_idx = _matching_dim_index(bucket_dims, block_dim)
        if bucket_idx is None:
            bucket_dims.append(block_dim)
            bucket_rows.append([block])
        else:
            bucket_rows[int(bucket_idx)].append(block)

    if not bucket_rows:
        return []

    cap_i = max(4, int(bucket_row_cap or max(8, limit_i * 3)))
    per_bucket_limit = max(1, int(per_dim_limit or limit_i))

    bucket_items: list[tuple[Any, list[SeedBlock]]] = []
    for block_dim, blocks in zip(bucket_dims, bucket_rows):
        blocks_sorted = sorted(blocks, key=_quadratic_base_sort_key)
        bucket_items.append((block_dim, blocks_sorted[: max(1, cap_i)]))

    def _bucket_rank(block_dim: Any) -> tuple[int, int]:
        if required_base_dims:
            idx = _matching_dim_index(required_base_dims, block_dim)
            return (0 if idx is not None else 1, int(idx) if idx is not None else len(required_base_dims))
        return (0 if block_dim is not None else 1, 0)

    bucket_items.sort(
        key=lambda item: (
            _bucket_rank(item[0]),
            _quadratic_base_sort_key(item[1][0]) if item[1] else (99, 99, 99, 99, ""),
        )
    )

    out: list[SeedBlock] = []
    seen: set[str] = set()
    bucket_counts = [0 for _ in bucket_items]
    max_arity_i = max(1, int(max_arity))

    def _is_affine_difference_block(block: SeedBlock) -> bool:
        return (
            str(getattr(block, "builder", "") or "") == "affine"
            and str(dict(getattr(block, "metadata", {}) or {}).get("affine_kind", "") or "") == "difference"
        )

    def _difference_combo_sort_key(combo: Sequence[SeedBlock]) -> tuple[Any, ...]:
        support: set[int] = set()
        for block in list(combo or ()):
            for value in tuple(getattr(block, "active_vars", ()) or ()):
                try:
                    support.add(int(value))
                except Exception:
                    continue
        try:
            size = sum(max(1, int(node_size(block.node))) for block in combo)
        except Exception:
            size = 999
        return (
            -int(len(support)),
            int(size),
            tuple(str(node_str(block.node)) for block in combo),
        )

    def _emit_combo(
        combo: Sequence[SeedBlock],
        *,
        bucket_idx: int,
        block_dim: Any,
    ) -> bool:
        if len(out) >= limit_i or bucket_counts[bucket_idx] >= per_bucket_limit:
            return False
        builder_depth = max((_seed_builder_depth(block) for block in combo), default=0) + 1
        nonlinear_depth = max((_seed_nonlinear_depth(block) for block in combo), default=0) + 1
        if builder_depth > int(max_builder_depth):
            return False
        if nonlinear_depth > int(max_nonlinear_depth):
            return False
        quad_terms = [simplify(("sqr", block.node)) for block in combo]
        if not quad_terms or not all(is_valid_node(term) for term in quad_terms):
            return False
        expr = quad_terms[0]
        for term in quad_terms[1:]:
            expr = simplify(("add", expr, term))
        key = str(node_str(expr))
        if key in seen or not is_valid_node(expr):
            return False
        expr_dim = dim_scale(block_dim, 2.0) if block_dim is not None else None
        if expr_dim is None and var_dims is not None:
            try:
                expr_dim = node_dims(expr, var_dims)
            except Exception:
                expr_dim = None
        if required_expr_dims_norm and (
            expr_dim is None or _matching_dim_index(required_expr_dims_norm, expr_dim) is None
        ):
            return False
        seen.add(key)
        out.append(
            make_seed_block(
                expr,
                dim=expr_dim,
                source="+".join(str(block.source or block.builder or "seed") for block in combo),
                builder="quadratic",
                metadata={
                    "bases": [block.to_dict() for block in combo],
                    "base_nodes": [block.node for block in combo],
                    "base_dims": [_seed_block_dim(block, var_dims) for block in combo],
                    "base_builders": [str(getattr(block, "builder", "") or "") for block in combo],
                    "base_affine_kinds": [
                        str(dict(getattr(block, "metadata", {}) or {}).get("affine_kind", "") or "")
                        for block in combo
                    ],
                    "quadratic_base_dim": block_dim,
                    "quadratic_expr_dim": expr_dim,
                    "builder_depth": builder_depth,
                    "nonlinear_depth": nonlinear_depth,
                    "product_arity": max((_seed_product_arity(block) for block in combo), default=1),
                    "domain_tags": _quadratic_domain_tags(combo),
                },
            )
        )
        bucket_counts[bucket_idx] += 1
        return True

    for bucket_idx, (block_dim, blocks) in enumerate(bucket_items):
        if len(out) >= limit_i or bucket_counts[bucket_idx] >= per_bucket_limit:
            break
        diff_blocks = [block for block in blocks if _is_affine_difference_block(block)]
        if len(diff_blocks) < 2:
            continue
        max_diff_arity = min(max_arity_i, len(diff_blocks))
        for arity in range(2, max_diff_arity + 1):
            if len(out) >= limit_i or bucket_counts[bucket_idx] >= per_bucket_limit:
                break
            diff_combos = sorted(combinations(diff_blocks, arity), key=_difference_combo_sort_key)
            for combo in diff_combos:
                if len(out) >= limit_i or bucket_counts[bucket_idx] >= per_bucket_limit:
                    break
                _emit_combo(combo, bucket_idx=bucket_idx, block_dim=block_dim)

    arity_order = list(range(max_arity_i, 1, -1)) + [1]
    for arity in arity_order:
        if len(out) >= limit_i:
            break
        iterators: list[tuple[int, Any, Any]] = []
        for bucket_idx, (block_dim, blocks) in enumerate(bucket_items):
            if bucket_counts[bucket_idx] >= per_bucket_limit or len(blocks) < arity:
                continue
            iterators.append((bucket_idx, block_dim, iter(combinations(blocks, arity))))
        if not iterators:
            continue
        progressed = True
        while len(out) < limit_i and progressed:
            progressed = False
            for bucket_idx, block_dim, combo_iter in iterators:
                if len(out) >= limit_i:
                    break
                if bucket_counts[bucket_idx] >= per_bucket_limit:
                    continue
                while True:
                    try:
                        combo = next(combo_iter)
                    except StopIteration:
                        combo = None
                    if combo is None:
                        break
                    if _emit_combo(combo, bucket_idx=bucket_idx, block_dim=block_dim):
                        progressed = True
                        break
    return out


def generate_affine_seed_blocks(
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    limit: int = 16,
    max_arity: int = 3,
    max_builder_depth: int = 2,
    max_nonlinear_depth: int = 1,
) -> list[SeedBlock]:
    rows = [block for block in list(seed_blocks or ()) if isinstance(block, SeedBlock)]
    out: list[SeedBlock] = []
    seen: set[str] = set()
    limit_i = max(0, int(limit))

    def _append_affine_expr(
        expr: tuple | None,
        *,
        dim,
        combo: Sequence[SeedBlock],
        source: str,
        affine_kind: str,
        builder_depth: int,
        nonlinear_depth: int,
    ) -> None:
        if len(out) >= limit_i:
            return
        if not (isinstance(expr, tuple) and is_valid_node(expr)):
            return
        key = str(node_str(expr))
        if key in seen:
            return
        seen.add(key)
        out.append(
            make_seed_block(
                expr,
                dim=dim,
                source=source,
                builder="affine",
                metadata={
                    "terms": [block.to_dict() for block in combo],
                    "term_nodes": [block.node for block in combo],
                    "affine_kind": str(affine_kind),
                    "builder_depth": builder_depth,
                    "nonlinear_depth": nonlinear_depth,
                    "product_arity": max((_seed_product_arity(block) for block in combo), default=1),
                },
            )
        )

    # Preserve simple coordinate-offset latents even when the live pool is
    # crowded with higher-priority product/unary anchors.
    raw_var_rows = [
        block
        for block in rows
        if isinstance(block.node, tuple)
        and len(block.node) >= 2
        and str(block.node[0]) == "var"
        and str(block.builder or "") in {"", "identity"}
    ]
    raw_var_rows.sort(key=lambda block: int(block.node[1]) if isinstance(block.node[1], int) else 999)
    for left, right in combinations(raw_var_rows, 2):
        if len(out) >= limit_i:
            break
        left_dim = getattr(left, "dim", None)
        right_dim = getattr(right, "dim", None)
        if left_dim is not None and right_dim is not None and not dims_eq(left_dim, right_dim):
            continue
        builder_depth = max((_seed_builder_depth(block) for block in (left, right)), default=0) + 1
        nonlinear_depth = max((_seed_nonlinear_depth(block) for block in (left, right)), default=0)
        if builder_depth > int(max_builder_depth):
            continue
        if nonlinear_depth > int(max_nonlinear_depth):
            continue
        _append_affine_expr(
            simplify(("sub", right.node, left.node)),
            dim=left_dim if left_dim is not None else right_dim,
            combo=(left, right),
            source="-".join(
                [
                    str(right.source or right.builder or "seed"),
                    str(left.source or left.builder or "seed"),
                ]
            ),
            affine_kind="difference",
            builder_depth=builder_depth,
            nonlinear_depth=nonlinear_depth,
        )

    for arity in range(2, max(2, int(max_arity)) + 1):
        if len(out) >= limit_i:
            break
        for combo in combinations(rows, arity):
            if len(out) >= limit_i:
                break
            if any(block.node == ("const", 1.0) for block in combo):
                continue
            dims = [block.dim for block in combo if getattr(block, "dim", None) is not None]
            if dims and any(not dims_eq(dim, dims[0]) for dim in dims[1:]):
                continue
            builder_depth = max((_seed_builder_depth(block) for block in combo), default=0) + 1
            nonlinear_depth = max((_seed_nonlinear_depth(block) for block in combo), default=0)
            if builder_depth > int(max_builder_depth):
                continue
            if nonlinear_depth > int(max_nonlinear_depth):
                continue
            expr = combo[0].node
            for block in combo[1:]:
                expr = simplify(("add", expr, block.node))
            _append_affine_expr(
                expr,
                dim=dims[0] if dims else None,
                combo=combo,
                source="+".join(str(block.source or block.builder or "seed") for block in combo),
                affine_kind="sum",
                builder_depth=builder_depth,
                nonlinear_depth=nonlinear_depth,
            )
            if len(out) >= limit_i:
                break
            if arity == 2:
                diff_expr = simplify(("sub", combo[1].node, combo[0].node))
                _append_affine_expr(
                    diff_expr,
                    dim=dims[0] if dims else None,
                    combo=combo,
                    source="-".join(
                        [
                            str(combo[1].source or combo[1].builder or "seed"),
                            str(combo[0].source or combo[0].builder or "seed"),
                        ]
                    ),
                    affine_kind="difference",
                    builder_depth=builder_depth,
                    nonlinear_depth=nonlinear_depth,
                )
    return out


def build_recursive_seed_pool(
    seed_blocks: Sequence[SeedBlock] | None,
    *,
    rounds: int = 2,
    include_product: bool = True,
    include_monomial: bool = True,
    include_quadratic: bool = False,
    include_affine: bool = False,
    product_max_arity: int = 3,
    product_limit: int = 16,
    monomial_exponents: Sequence[float] = (-1.0, -0.5, 0.5, 1.0, 2.0),
    monomial_limit: int = 24,
    quadratic_max_arity: int = 3,
    quadratic_limit: int = 12,
    quadratic_required_dims: Sequence[Any] | None = None,
    quadratic_bucket_row_cap: int | None = None,
    quadratic_per_dim_limit: int | None = None,
    affine_max_arity: int = 3,
    affine_limit: int = 12,
    max_builder_depth: int = 2,
    max_nonlinear_depth: int = 1,
    var_dims: Sequence[Sequence[float]] | None = None,
) -> list[SeedBlock]:
    pool: list[SeedBlock] = list(seed_blocks or ())
    seen = {str(node_str(block.node)) for block in pool if isinstance(block, SeedBlock)}
    rounds_count = max(0, int(rounds))
    for _ in range(rounds_count):
        current = list(pool)
        fresh: list[SeedBlock] = []
        if include_product:
            fresh.extend(
                generate_product_seed_blocks(
                    current,
                    max_arity=product_max_arity,
                    limit=product_limit,
                    max_builder_depth=max_builder_depth,
                    max_nonlinear_depth=max_nonlinear_depth,
                )
            )
        if include_monomial:
            fresh.extend(
                generate_monomial_seed_blocks(
                    current,
                    exponents=monomial_exponents,
                    limit=monomial_limit,
                    max_builder_depth=max_builder_depth,
                    max_nonlinear_depth=max_nonlinear_depth,
                )
            )
        if include_quadratic:
            fresh.extend(
                generate_quadratic_seed_blocks(
                    current,
                    limit=quadratic_limit,
                    max_arity=quadratic_max_arity,
                    max_builder_depth=max_builder_depth,
                    max_nonlinear_depth=max_nonlinear_depth,
                    var_dims=var_dims,
                    required_expr_dims=quadratic_required_dims,
                    bucket_row_cap=quadratic_bucket_row_cap,
                    per_dim_limit=quadratic_per_dim_limit,
                )
            )
        if include_affine:
            fresh.extend(
                generate_affine_seed_blocks(
                    current,
                    limit=affine_limit,
                    max_arity=affine_max_arity,
                    max_builder_depth=max_builder_depth,
                    max_nonlinear_depth=max_nonlinear_depth,
                )
            )
        added = False
        for block in fresh:
            key = str(node_str(block.node))
            if key in seen:
                continue
            pool.append(block)
            seen.add(key)
            added = True
        if not added:
            break
    return pool


__all__ = [
    "SeedBlock",
    "build_recursive_seed_pool",
    "extend_seed_blocks_with_basis",
    "generate_affine_seed_blocks",
    "generate_monomial_seed_blocks",
    "generate_product_seed_blocks",
    "generate_quadratic_seed_blocks",
    "make_seed_block",
    "seed_anchor_blocks",
    "seed_blocks_from_basis_state",
]
