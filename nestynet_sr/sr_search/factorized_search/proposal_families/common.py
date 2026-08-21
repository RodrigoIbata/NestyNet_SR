# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import torch

from ..basis_scoring import (
    add_terms,
    fit_direct_linear_design,
    fit_direct_rational_design,
    materialize_direct_linear_combo,
    materialize_direct_rational_expr,
    scaled_node,
    snap_direct_coeff,
)
from ..expr_ast import (
    dim_round,
    dims_eq,
    eval_node,
    node_depth,
    node_size,
    node_str,
    simplify,
)


def dim0(var_dims: Sequence[Sequence[float]] | None) -> tuple[float, ...] | None:
    if not var_dims:
        return None
    return (0.0,) * len(var_dims[0])


def dim_add(a: Any, b: Any) -> tuple[float, ...] | None:
    if a is None or b is None:
        return None
    try:
        return dim_round(tuple(float(x) + float(y) for x, y in zip(tuple(a), tuple(b))))
    except Exception:
        return None


def dim_sub(a: Any, b: Any) -> tuple[float, ...] | None:
    if a is None or b is None:
        return None
    try:
        return dim_round(tuple(float(x) - float(y) for x, y in zip(tuple(a), tuple(b))))
    except Exception:
        return None


def dim_scale(a: Any, scale: float) -> tuple[float, ...] | None:
    if a is None:
        return None
    try:
        return dim_round(tuple(float(scale) * float(x) for x in tuple(a)))
    except Exception:
        return None


def record_status(stats: dict[str, Any], status: object) -> None:
    key = str(status or "")
    if not key:
        return
    counts = stats.get("status_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        stats["status_counts"] = counts
    counts[key] = int(counts.get(key, 0) or 0) + 1


def deadline_exceeded(deadline_s: float | None) -> bool:
    if deadline_s is None:
        return False
    try:
        return bool(time.perf_counter() >= float(deadline_s))
    except Exception:
        return False


def node_var_count(node: tuple | None) -> int:
    if not isinstance(node, tuple) or not node:
        return 0
    op = str(node[0])
    if op == "var":
        return 1
    seen: set[int] = set()

    def _visit(cur) -> None:
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
    return int(len(seen))


def positive_scalar_only_value(node, x_ref: torch.Tensor | None) -> float | None:
    if not isinstance(node, tuple) or not node or node_var_count(node) != 0:
        return None
    if not torch.is_tensor(x_ref) or int(x_ref.shape[0]) <= 0:
        return None
    try:
        value = eval_node(node, x_ref[:1])
    except Exception:
        return None
    if not torch.is_tensor(value):
        return None
    try:
        scalar = float(value.reshape(-1)[0].item())
    except Exception:
        return None
    if not math.isfinite(scalar) or scalar <= 0.0:
        return None
    return float(scalar)


def strip_log_scalar_factors(node, x_ref: torch.Tensor | None):
    cur = simplify(node)
    for _ in range(8):
        if not isinstance(cur, tuple) or not cur:
            break
        op = str(cur[0])
        if op == "mul" and len(cur) >= 3:
            left, right = cur[1], cur[2]
            if positive_scalar_only_value(left, x_ref) is not None and node_var_count(right) > 0:
                cur = simplify(right)
                continue
            if positive_scalar_only_value(right, x_ref) is not None and node_var_count(left) > 0:
                cur = simplify(left)
                continue
            break
        if op == "div" and len(cur) >= 3:
            left, right = cur[1], cur[2]
            if positive_scalar_only_value(right, x_ref) is not None and node_var_count(left) > 0:
                cur = simplify(left)
                continue
            if positive_scalar_only_value(left, x_ref) is not None and node_var_count(right) > 0:
                cur = simplify(("div", ("const", 1.0), right))
                continue
            break
        break
    return cur


def anchor_priority(node: tuple) -> tuple[int, int, int, str]:
    op = str(node[0]) if isinstance(node, tuple) and node else ""
    try:
        size = max(1, int(node_size(node)))
    except Exception:
        size = 99
    key = str(node_str(node))
    uniq_vars = max(0, node_var_count(node))
    if node == ("const", 1.0):
        return (0, size, 0, key)
    if op == "mul":
        return (1, size, -uniq_vars, key)
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr"):
        return (2, size, -uniq_vars, key)
    if op == "var":
        return (3, size, -uniq_vars, key)
    if op in ("add", "sub"):
        return (4, size, -uniq_vars, key)
    if op == "div":
        return (5, size, -uniq_vars, key)
    if op == "const":
        return (6, size, 0, key)
    return (7, size, -uniq_vars, key)


def pick_placeholder_node(
    *,
    desired_dim: Any,
    seed_nodes: Sequence[tuple[tuple, Any]],
    dim0_value: Any,
) -> tuple | None:
    if desired_dim is None:
        return ("const", 1.0)
    if dim0_value is not None and dims_eq(desired_dim, dim0_value):
        return ("const", 1.0)
    for node, node_dim in list(seed_nodes or ()):
        if node_dim is None:
            continue
        try:
            if dims_eq(node_dim, desired_dim):
                return node
        except Exception:
            continue
    return None


def shortlist_direct_candidate_nodes(
    rows: Sequence[tuple[str, tuple]] | None,
    *,
    max_count: int,
) -> list[tuple[str, tuple]]:
    ranked: list[tuple[tuple[int, int, int, str, str], tuple[str, tuple]]] = []
    for source, node in list(rows or ()):
        if not isinstance(node, tuple) or not node:
            continue
        try:
            size = int(node_size(node))
        except Exception:
            size = 999
        try:
            depth = int(node_depth(node))
        except Exception:
            depth = 999
        uniq_vars = max(0, node_var_count(node))
        key = str(node_str(node))
        ranked.append(((size, depth, -uniq_vars, str(source), key), (str(source), node)))
    ranked.sort(key=lambda item: item[0])
    return [row for _key, row in ranked[: max(1, int(max_count))]]


__all__ = [
    "add_terms",
    "anchor_priority",
    "deadline_exceeded",
    "dim_add",
    "dim_scale",
    "dim_sub",
    "dim0",
    "fit_direct_linear_design",
    "fit_direct_rational_design",
    "materialize_direct_linear_combo",
    "materialize_direct_rational_expr",
    "node_var_count",
    "pick_placeholder_node",
    "positive_scalar_only_value",
    "record_status",
    "scaled_node",
    "shortlist_direct_candidate_nodes",
    "snap_direct_coeff",
    "strip_log_scalar_factors",
]
