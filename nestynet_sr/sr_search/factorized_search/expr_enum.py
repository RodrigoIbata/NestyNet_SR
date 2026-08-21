# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Cycle-free expression enumeration helpers shared across factorized symbolic search modules."""

from __future__ import annotations

from .expr_ast import (
    BINARY_OPS,
    UNARY_OPS,
    dim_round,
    node_str,
    simplify,
)


def _has_const_zero(node):
    op = node[0]
    if op == "const":
        return node[1] == 0.0
    if op == "var":
        return False
    if op in UNARY_OPS:
        return _has_const_zero(node[1])
    if op in BINARY_OPS:
        return _has_const_zero(node[1]) or _has_const_zero(node[2])
    return False


def enumerate_trees(max_depth, nvars, max_trees=None):
    """Return all expression trees up to *max_depth* with *nvars* variables."""

    n_un = len(UNARY_OPS)
    n_bin = len(BINARY_OPS)
    up_to = [("var", i) for i in range(nvars)]
    depth_reached = 1
    for _depth in range(2, max_depth + 1):
        n_prev = len(up_to)
        projected = n_prev + n_un * n_prev + n_bin * n_prev * n_prev
        if max_trees is not None and projected > max_trees:
            print(
                f"[brute]  adaptive: depth {_depth} would produce ~{projected:,} "
                f"raw trees (budget {max_trees:,}), stopping at depth {depth_reached}"
            )
            break
        new = []
        for op in UNARY_OPS:
            for tree in up_to:
                new.append((op, tree))
        for op in BINARY_OPS:
            for left in up_to:
                for right in up_to:
                    new.append((op, left, right))
        up_to = up_to + new
        depth_reached = _depth
    return up_to, depth_reached


def enumerate_trees_dim(max_depth, nvars, var_dims, y_dims, max_trees=None):
    """Enumerate only dimensionally valid trees up to *max_depth*."""

    ndim = len(var_dims[0])
    dim0 = (0.0,) * ndim
    by_dim: dict[tuple, dict[str, tuple]] = {}

    def _add(tree, dim):
        simp = simplify(tree)
        if _has_const_zero(simp):
            return
        key = node_str(simp)
        bucket = by_dim.setdefault(dim, {})
        if key not in bucket:
            bucket[key] = simp

    def _total():
        return sum(len(bucket) for bucket in by_dim.values())

    for i in range(nvars):
        _add(("var", i), dim_round(var_dims[i]))

    prev_total = _total()
    depth_reached = 1

    for _depth in range(2, max_depth + 1):
        cur_total = _total()
        if max_trees is not None and _depth > 2:
            ratio = cur_total / max(prev_total, 1)
            projected = int(cur_total * ratio)
            if projected > max_trees:
                print(
                    f"[brute]  adaptive: depth {_depth} would produce ~{projected:,} "
                    f"trees (growth x{ratio:.1f}, budget {max_trees:,}), "
                    f"stopping at depth {depth_reached}"
                )
                break
        prev_total = cur_total

        all_trees = []
        for dim, bucket in by_dim.items():
            for tree in bucket.values():
                all_trees.append((dim, tree))

        new_entries = []
        for dim, tree in all_trees:
            new_entries.append((dim, ("neg", tree)))
            if dim == dim0:
                for op in ("sin", "cos", "exp", "log"):
                    new_entries.append((dim0, (op, tree)))
            new_entries.append((dim_round(tuple(x / 2 for x in dim)), ("sqrt", tree)))
            new_entries.append((dim_round(tuple(x * 2 for x in dim)), ("sqr", tree)))

        dim_list = list(by_dim.keys())
        for d1 in dim_list:
            bucket1 = list(by_dim[d1].values())
            for d2 in dim_list:
                bucket2 = list(by_dim[d2].values())
                same_dim = d1 == d2
                d_mul = dim_round(tuple(a + b for a, b in zip(d1, d2)))
                d_div = dim_round(tuple(a - b for a, b in zip(d1, d2)))
                for left in bucket1:
                    for right in bucket2:
                        if same_dim:
                            new_entries.append((d1, ("add", left, right)))
                            new_entries.append((d1, ("sub", left, right)))
                        new_entries.append((d_mul, ("mul", left, right)))
                        new_entries.append((d_div, ("div", left, right)))

        for dim, tree in new_entries:
            _add(tree, dim)

        depth_reached = _depth

    y_key = dim_round(tuple(y_dims))
    bucket = by_dim.get(y_key, {})
    return list(bucket.values()), depth_reached


__all__ = [
    "enumerate_trees",
    "enumerate_trees_dim",
]
