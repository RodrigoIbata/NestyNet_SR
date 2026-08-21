# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import torch

from ..expr_ast import dims_eq, eval_node, node_dims, node_size, node_str
from .common import deadline_exceeded, node_var_count, shortlist_direct_candidate_nodes
from .seed_blocks import SeedBlock


def seed_block_dim(block: SeedBlock, var_dims) -> Any:
    if block.dim is not None:
        return block.dim
    if var_dims is None:
        return None
    try:
        return node_dims(block.node, var_dims)
    except Exception:
        return None


def dedup_seed_blocks(blocks: Sequence[SeedBlock]) -> list[SeedBlock]:
    out: list[SeedBlock] = []
    seen: set[str] = set()
    for block in list(blocks or ()):
        if not isinstance(block, SeedBlock):
            continue
        key = str(node_str(block.node))
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def filter_seed_blocks_for_dim(
    blocks: Sequence[SeedBlock],
    *,
    target_dim,
    var_dims,
    drop_const: bool = False,
) -> list[SeedBlock]:
    out: list[SeedBlock] = []
    for block in list(blocks or ()):
        if drop_const and str(block.node[0]) == "const":
            continue
        block_dim = seed_block_dim(block, var_dims)
        if target_dim is not None:
            if block_dim is None or not dims_eq(block_dim, target_dim):
                continue
        out.append(
            SeedBlock(
                node=block.node,
                dim=block_dim,
                source=block.source,
                builder=block.builder,
                active_vars=block.active_vars,
                domain_tags=block.domain_tags,
                metadata=dict(block.metadata or {}),
            )
        )
    return dedup_seed_blocks(out)


def quadratic_base_priority(block: SeedBlock) -> tuple[int, int, int, str]:
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
    key = str(node_str(node))
    if op == "var":
        return (0, size, -uniq_vars, key)
    if builder == "identity" and op not in {"sqrt", "sqr", "exp", "log", "sin", "cos"}:
        return (1, size, -uniq_vars, key)
    if source == "var":
        return (2, size, -uniq_vars, key)
    return (3, size, -uniq_vars, key)


def evaluate_seed_blocks(
    blocks: Sequence[SeedBlock],
    *,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    deadline_s: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in list(blocks or ()):
        if deadline_exceeded(deadline_s):
            break
        try:
            fit_t = eval_node(block.node, x_fit)
            probe_t = eval_node(block.node, x_probe)
        except Exception:
            continue
        if (not torch.is_tensor(fit_t)) or (not torch.is_tensor(probe_t)):
            continue
        if (not torch.isfinite(fit_t).all()) or (not torch.isfinite(probe_t).all()):
            continue
        rows.append(
            {
                "block": block,
                "fit": fit_t.squeeze(-1),
                "probe": probe_t.squeeze(-1),
            }
        )
    return rows


def evaluate_candidate_node_map(
    rows: Sequence[tuple[str, tuple]],
    *,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    deadline_s: float | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, str, tuple]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor, str, tuple]] = {}
    for source, node in list(rows or ()):
        if deadline_exceeded(deadline_s):
            break
        key = str(node_str(node))
        if key in out:
            continue
        try:
            fit_t = eval_node(node, x_fit)
            probe_t = eval_node(node, x_probe)
        except Exception:
            continue
        if (not torch.is_tensor(fit_t)) or (not torch.is_tensor(probe_t)):
            continue
        if (not torch.isfinite(fit_t).all()) or (not torch.isfinite(probe_t).all()):
            continue
        out[key] = (fit_t, probe_t, str(source), node)
    return out


def collect_shortlisted_hole_candidates(
    *,
    collect_direct_hole_candidates_fn: Callable[..., tuple[list[tuple[str, tuple]], dict[str, Any]]],
    nvars: int,
    enum_max_depth: int,
    enum_max_trees: int,
    var_dims,
    target_dim,
    pool_nodes,
    pool_dims,
    shortlist_k: int,
    deadline_s: float | None,
    pin_predicate: Callable[[tuple], bool] | None = None,
) -> tuple[list[tuple[str, tuple]], dict[str, Any]]:
    started = time.perf_counter()
    candidate_nodes, meta = collect_direct_hole_candidates_fn(
        nvars=int(nvars),
        enum_max_depth=int(enum_max_depth),
        enum_max_trees=int(enum_max_trees),
        var_dims=var_dims,
        target_dim=target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        deadline_s=deadline_s,
    )
    collect_s = float(time.perf_counter() - started)
    shortlist_started = time.perf_counter()
    meta_out = dict(meta)
    meta_out["candidate_count_collected"] = int(len(candidate_nodes or ()))
    meta_out["shortlist_k"] = int(shortlist_k)
    meta_out["timing_collect_call_s"] = float(collect_s)
    if not candidate_nodes:
        meta_out["candidate_count_shortlisted"] = 0
        meta_out["candidate_count_pinned"] = 0
        meta_out["timing_shortlist_s"] = float(time.perf_counter() - shortlist_started)
        meta_out["timing_collect_shortlist_total_s"] = float(time.perf_counter() - started)
        return [], meta_out
    if pin_predicate is None:
        shortlisted = shortlist_direct_candidate_nodes(candidate_nodes, max_count=max(1, int(shortlist_k)))
        meta_out["candidate_count_shortlisted"] = int(len(shortlisted))
        meta_out["candidate_count_pinned"] = 0
        meta_out["timing_shortlist_s"] = float(time.perf_counter() - shortlist_started)
        meta_out["timing_collect_shortlist_total_s"] = float(time.perf_counter() - started)
        return shortlisted, meta_out

    pinned_candidates: list[tuple[str, tuple]] = []
    pinned_keys: set[str] = set()
    for source, node in list(candidate_nodes or ()):
        if not isinstance(node, tuple) or not node:
            continue
        try:
            if not bool(pin_predicate(node)):
                continue
        except Exception:
            continue
        key = str(node_str(node))
        if key in pinned_keys:
            continue
        pinned_keys.add(key)
        pinned_candidates.append((source, node))
    remaining_candidates = [
        row for row in list(candidate_nodes or ()) if str(node_str(row[1])) not in pinned_keys
    ]
    shortlisted = pinned_candidates + shortlist_direct_candidate_nodes(
        remaining_candidates,
        max_count=max(1, int(shortlist_k) - len(pinned_candidates)),
    )
    shortlisted = shortlisted[: max(int(shortlist_k), len(pinned_candidates))]
    meta_out["candidate_count_shortlisted"] = int(len(shortlisted))
    meta_out["candidate_count_pinned"] = int(len(pinned_candidates))
    meta_out["timing_shortlist_s"] = float(time.perf_counter() - shortlist_started)
    meta_out["timing_collect_shortlist_total_s"] = float(time.perf_counter() - started)
    return shortlisted, meta_out


def pin_single_var_square(node: tuple) -> bool:
    return (
        isinstance(node, tuple)
        and len(node) >= 2
        and str(node[0]) == "sqr"
        and max(0, node_var_count(node)) <= 1
    )


def pin_small_trig_carrier(node: tuple) -> bool:
    if not (isinstance(node, tuple) and len(node) >= 2):
        return False
    if str(node[0]) not in {"sin", "cos"}:
        return False
    try:
        uniq_vars = max(0, node_var_count(node))
    except Exception:
        uniq_vars = 99
    if uniq_vars <= 0 or uniq_vars > 2:
        return False
    try:
        size = int(node_size(node))
    except Exception:
        size = 999
    return size <= 6


def pin_ratio_square(node: tuple) -> bool:
    if not (isinstance(node, tuple) and len(node) >= 2 and str(node[0]) == "sqr"):
        return False
    inner = node[1]
    if not (isinstance(inner, tuple) and inner):
        return False
    if str(inner[0]) != "div":
        return False
    try:
        return max(0, node_var_count(node)) >= 2
    except Exception:
        return True


def pin_dimensionless_ratio_square(block, *, var_dims=None) -> bool:
    node = getattr(block, "node", block)
    if not pin_ratio_square(node):
        return False
    if var_dims is None:
        return False
    dim0_val = tuple(0.0 for _ in var_dims[0]) if var_dims else None
    if dim0_val is None:
        return False
    block_dim = getattr(block, "dim", None)
    if block_dim is None:
        try:
            block_dim = node_dims(node, var_dims)
        except Exception:
            return False
    return block_dim is not None and dims_eq(block_dim, dim0_val)


def pin_wrapped_rational_term(node: tuple) -> bool:
    if not (isinstance(node, tuple) and node):
        return False
    op = str(node[0])
    if op in {"exp", "sqr", "sqrt", "sin", "cos", "log"}:
        return True
    if op in {"mul", "div"}:
        try:
            return max(0, node_var_count(node)) >= 2
        except Exception:
            return True
    return False


def prepend_pinned_candidate(
    candidate_nodes: Sequence[tuple[str, tuple]],
    *,
    pinned_row: tuple[str, tuple] | None,
    shortlist_count: int,
) -> list[tuple[str, tuple]]:
    if pinned_row is None:
        return shortlist_direct_candidate_nodes(candidate_nodes, max_count=int(shortlist_count))
    return [
        pinned_row,
        *shortlist_direct_candidate_nodes(
            candidate_nodes,
            max_count=max(1, int(shortlist_count) - 1),
        ),
    ]


__all__ = [
    "collect_shortlisted_hole_candidates",
    "dedup_seed_blocks",
    "evaluate_candidate_node_map",
    "evaluate_seed_blocks",
    "filter_seed_blocks_for_dim",
    "pin_dimensionless_ratio_square",
    "pin_ratio_square",
    "pin_small_trig_carrier",
    "pin_single_var_square",
    "pin_wrapped_rational_term",
    "prepend_pinned_candidate",
    "quadratic_base_priority",
    "seed_block_dim",
]
