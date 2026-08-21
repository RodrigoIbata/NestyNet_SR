# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tuple-AST primitives shared across factorized symbolic search explorer subsystems."""

from __future__ import annotations

import math
import operator

import torch


UNARY_OPS = ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg")
INVERSE_TRIG_OPS = ("asin", "acos")
UNARY_NODE_OPS = UNARY_OPS + INVERSE_TRIG_OPS
BINARY_OPS = ("add", "sub", "mul", "div")


def _is_index_token(value):
    try:
        operator.index(value)
    except Exception:
        return False
    return not isinstance(value, bool)


def _is_finite_scalar(value):
    try:
        out = float(value)
    except Exception:
        return False
    return math.isfinite(out)


def is_valid_node(node):
    """Return True when ``node`` is a structurally valid tuple-AST."""
    if not isinstance(node, (tuple, list)) or len(node) == 0:
        return False
    op = node[0]
    if not isinstance(op, str):
        return False
    if op in ("var", "hparam"):
        return len(node) == 2 and _is_index_token(node[1])
    if op == "const":
        return len(node) == 2 and _is_finite_scalar(node[1])
    if op in UNARY_NODE_OPS:
        return len(node) == 2 and is_valid_node(node[1])
    if op in BINARY_OPS:
        return len(node) == 3 and is_valid_node(node[1]) and is_valid_node(node[2])
    return False


def sample_box(n, d, lo, hi, *, dtype, g):
    # lo/hi can be scalars or per-dimension vectors (length d)
    lo_t = torch.as_tensor(lo, dtype=dtype)
    hi_t = torch.as_tensor(hi, dtype=dtype)
    if lo_t.ndim == 0:
        lo_t = lo_t.repeat(d)
    if hi_t.ndim == 0:
        hi_t = hi_t.repeat(d)
    lo_t = lo_t.reshape(1, d)
    hi_t = hi_t.reshape(1, d)
    u = torch.rand((n, d), generator=g, dtype=dtype)
    return lo_t + (hi_t - lo_t) * u


def eval_node(node, x):
    op = node[0]
    if op == "var":
        i = node[1]
        return x[:, i : i + 1]
    if op == "const":
        return torch.full((x.shape[0], 1), node[1], dtype=x.dtype, device=x.device)
    if op == "sin":
        return torch.sin(eval_node(node[1], x))
    if op == "cos":
        return torch.cos(eval_node(node[1], x))
    if op == "exp":
        return torch.exp(eval_node(node[1], x))
    if op == "log":
        return torch.log(eval_node(node[1], x))
    if op == "sqrt":
        return torch.sqrt(eval_node(node[1], x))
    if op == "sqr":
        c = eval_node(node[1], x)
        return c * c
    if op == "neg":
        return -eval_node(node[1], x)
    if op == "asin":
        return torch.asin(eval_node(node[1], x))
    if op == "acos":
        return torch.acos(eval_node(node[1], x))
    if op == "add":
        return eval_node(node[1], x) + eval_node(node[2], x)
    if op == "sub":
        return eval_node(node[1], x) - eval_node(node[2], x)
    if op == "mul":
        return eval_node(node[1], x) * eval_node(node[2], x)
    if op == "div":
        a = eval_node(node[1], x)
        b = eval_node(node[2], x)
        return a / b
    raise ValueError(op)


def node_str(node):
    op = node[0]
    if op == "var":
        return f"x{node[1]}"
    if op == "const":
        v = node[1]
        return f"{v:g}" if v != int(v) else str(int(v))
    if op == "hparam":
        return f"hp{int(node[1])}"
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "asin", "acos"):
        return f"{op}({node_str(node[1])})"
    if op == "neg":
        return f"(-{node_str(node[1])})"
    if op == "add":
        return f"({node_str(node[1])}+{node_str(node[2])})"
    if op == "sub":
        return f"({node_str(node[1])}-{node_str(node[2])})"
    if op == "mul":
        return f"({node_str(node[1])}*{node_str(node[2])})"
    if op == "div":
        return f"({node_str(node[1])}/{node_str(node[2])})"
    return "?"


def node_size(node):
    op = node[0]
    if op in ("var", "const", "hparam"):
        return 1
    if op in UNARY_NODE_OPS:
        return 1 + node_size(node[1])
    return 1 + node_size(node[1]) + node_size(node[2])


def node_cost_physics_prior(node):
    """Weighted tuple-AST complexity with a lightweight physics prior."""
    op = node[0]
    if op == "var":
        return 0.0
    if op == "const":
        return 0.2
    if op == "hparam":
        return 0.25
    if op == "neg":
        return 0.5 + node_cost_physics_prior(node[1])
    if op == "add":
        return 1.0 + node_cost_physics_prior(node[1]) + node_cost_physics_prior(node[2])
    if op == "sub":
        return 1.0 + node_cost_physics_prior(node[1]) + node_cost_physics_prior(node[2])
    if op == "mul":
        return 1.0 + node_cost_physics_prior(node[1]) + node_cost_physics_prior(node[2])
    if op == "div":
        return 2.0 + node_cost_physics_prior(node[1]) + node_cost_physics_prior(node[2])
    if op == "sqr":
        return 1.5 + node_cost_physics_prior(node[1])
    if op == "sqrt":
        return 2.5 + node_cost_physics_prior(node[1])
    if op in ("sin", "cos"):
        return 4.0 + node_cost_physics_prior(node[1])
    if op in ("asin", "acos"):
        return 5.0 + node_cost_physics_prior(node[1])
    if op in ("exp", "log"):
        return 4.5 + node_cost_physics_prior(node[1])
    if op in UNARY_OPS:
        return 5.0 + node_cost_physics_prior(node[1])
    if op in BINARY_OPS:
        return 5.0 + node_cost_physics_prior(node[1]) + node_cost_physics_prior(node[2])
    return 6.0


def node_depth(node):
    op = node[0]
    if op in ("var", "const", "hparam"):
        return 1
    if op in UNARY_NODE_OPS:
        return 1 + node_depth(node[1])
    return 1 + max(node_depth(node[1]), node_depth(node[2]))


def rand_node(rng, max_depth, nvars):
    if max_depth <= 1:
        return ("var", rng.randrange(nvars))
    r = rng.random()
    if r < 0.25:
        return ("var", rng.randrange(nvars))
    if r < 0.55:
        op = rng.choice(UNARY_OPS)
        return (op, rand_node(rng, max_depth - 1, nvars))
    op = rng.choice(BINARY_OPS)
    return (op, rand_node(rng, max_depth - 1, nvars), rand_node(rng, max_depth - 1, nvars))


def collect_paths(node, path=()):
    out = [path]
    op = node[0]
    if op in UNARY_NODE_OPS:
        out += collect_paths(node[1], path + (1,))
    elif op in BINARY_OPS:
        out += collect_paths(node[1], path + (1,))
        out += collect_paths(node[2], path + (2,))
    return out


def get_at(node, path):
    if not path:
        return node
    op = node[0]
    idx = path[0]
    if op in UNARY_NODE_OPS and idx == 1:
        return get_at(node[1], path[1:])
    if op in BINARY_OPS:
        if idx == 1:
            return get_at(node[1], path[1:])
        if idx == 2:
            return get_at(node[2], path[1:])
    raise ValueError("bad path")


def replace_at(node, path, new_sub):
    if not path:
        return new_sub
    op = node[0]
    idx = path[0]
    if op in UNARY_NODE_OPS and idx == 1:
        return (op, replace_at(node[1], path[1:], new_sub))
    if op in BINARY_OPS:
        if idx == 1:
            return (op, replace_at(node[1], path[1:], new_sub), node[2])
        if idx == 2:
            return (op, node[1], replace_at(node[2], path[1:], new_sub))
    raise ValueError("bad path")


def cap_depth(node, rng, max_depth, nvars):
    return node if node_depth(node) <= max_depth else rand_node(rng, max_depth, nvars)


def build_pool(nvars, *, ir_cfg=None, ir_stats=None, signature_context=None):
    """Pool of simple candidate terms for residual-guided search."""
    pool = []
    seen = set()

    def _add(node):
        candidate = node
        try:
            from nestynet_sr.sr_expr_ir.config import expr_ir_active
            from nestynet_sr.sr_expr_ir.tuple_bridge import (
                canonical_key_tuple_ast,
                maybe_canonicalize_tuple_ast,
            )

            if expr_ir_active(ir_cfg):
                candidate = maybe_canonicalize_tuple_ast(
                    node,
                    ir_cfg,
                    stats=ir_stats,
                    signature_context=signature_context,
                )
                key = canonical_key_tuple_ast(
                    candidate,
                    ir_cfg,
                    stats=ir_stats,
                    signature_context=signature_context,
                )
            else:
                key = ("legacy", node_str(candidate))
        except Exception:
            candidate = node
            key = ("legacy", node_str(node))
        if key in seen:
            return
        seen.add(key)
        pool.append(candidate)

    _add(("const", 1.0))
    _add(("const", -1.0))

    for i in range(nvars):
        _add(("var", i))
        _add(("div", ("const", 1.0), ("var", i)))

    for i in range(nvars):
        for j in range(i, nvars):
            _add(("mul", ("var", i), ("var", j)))

    if int(nvars) <= 10:
        for i in range(nvars):
            for j in range(i, nvars):
                for k in range(j, nvars):
                    _add(("mul", ("mul", ("var", i), ("var", j)), ("var", k)))

    for i in range(nvars):
        _add(("sin", ("var", i)))
        _add(("cos", ("var", i)))
        _add(("exp", ("var", i)))
        _add(("log", ("var", i)))
        _add(("sqrt", ("var", i)))
        _add(("sqr", ("var", i)))

    if int(nvars) <= 12:
        for i in range(nvars):
            for j in range(i, nvars):
                ij = ("mul", ("var", i), ("var", j))
                _add(("sin", ij))
                _add(("cos", ij))

    for i in range(nvars):
        for j in range(nvars):
            if i == j:
                continue
            _add(("mul", ("var", i), ("sin", ("var", j))))
            _add(("mul", ("var", i), ("cos", ("var", j))))
            _add(("div", ("var", i), ("var", j)))
            _add(("div", ("var", i), ("sqr", ("var", j))))

    if int(nvars) <= 8:
        for i in range(nvars):
            for j in range(nvars):
                for k in range(j + 1, nvars):
                    jk = ("mul", ("var", j), ("var", k))
                    _add(("mul", ("var", i), ("cos", jk)))
                    _add(("mul", ("var", i), ("sin", jk)))

    return pool


_DIM_PREC = 16


def set_dim_precision(max_depth, *, min_prec=16):
    """Set the global dimension rounding precision for this run."""
    global _DIM_PREC
    try:
        md = int(max_depth)
        if md < 0:
            md = 0
    except Exception:
        md = 0
    _DIM_PREC = max(int(min_prec), int(2 ** md))


def dims_eq(d1, d2, tol=1e-9):
    if d1 is None or d2 is None:
        return False
    if len(d1) != len(d2):
        return False
    return all(abs(a - b) < tol for a, b in zip(d1, d2))


def dim_round(d, prec=None):
    p = _DIM_PREC if prec is None else int(prec)
    return tuple(round(float(x) * p) / p for x in d)


def node_dims(node, var_dims):
    """Dimension vector of expression, or None if internally inconsistent."""
    op = node[0]
    ndim = len(var_dims[0])
    dim0 = (0.0,) * ndim
    if op == "var":
        return var_dims[node[1]]
    if op == "const":
        return dim0
    if op == "hparam":
        return dim0
    if op == "neg":
        return node_dims(node[1], var_dims)
    if op in ("sin", "cos", "exp", "log", "asin", "acos"):
        d = node_dims(node[1], var_dims)
        if d is None or not dims_eq(d, dim0):
            return None
        return dim0
    if op == "sqrt":
        d = node_dims(node[1], var_dims)
        if d is None:
            return None
        return dim_round(tuple(x / 2 for x in d))
    if op == "sqr":
        d = node_dims(node[1], var_dims)
        if d is None:
            return None
        return dim_round(tuple(x * 2 for x in d))
    if op in ("add", "sub"):
        d1 = node_dims(node[1], var_dims)
        d2 = node_dims(node[2], var_dims)
        if d1 is None or d2 is None or not dims_eq(d1, d2):
            return None
        return d1
    if op == "mul":
        d1 = node_dims(node[1], var_dims)
        d2 = node_dims(node[2], var_dims)
        if d1 is None or d2 is None:
            return None
        return dim_round(tuple(a + b for a, b in zip(d1, d2)))
    if op == "div":
        d1 = node_dims(node[1], var_dims)
        d2 = node_dims(node[2], var_dims)
        if d1 is None or d2 is None:
            return None
        return dim_round(tuple(a - b for a, b in zip(d1, d2)))
    return None


def _dim_l1(d, anchors):
    best = None
    for a in anchors:
        dist = 0.0
        for x, y in zip(d, a):
            dist += abs(float(x) - float(y))
        if best is None or dist < best:
            best = dist
    return 0.0 if best is None else best


def compute_reachable(var_dims, max_depth, target_dim=None, max_set=500):
    """Approximate set of buildable dimensions at each tree depth."""
    ndim = len(var_dims[0])
    dim0 = (0.0,) * ndim
    var_set = set(dim_round(tuple(d)) for d in var_dims)

    tgt = dim_round(tuple(target_dim)) if target_dim is not None else None
    anchors = [dim0]
    must_keep = set(var_set)
    must_keep.add(dim0)
    if tgt is not None:
        anchors.append(tgt)
        anchors.append(dim_round(tuple(2 * x for x in tgt)))
        must_keep.add(tgt)
        must_keep.add(dim_round(tuple(2 * x for x in tgt)))
        must_keep.add(dim_round(tuple(0.5 * x for x in tgt)))

    reach = [frozenset(), frozenset(var_set)]
    for depth in range(2, int(max_depth) + 1):
        prev = reach[depth - 1]
        curr = set(prev)

        for d in prev:
            curr.add(dim_round(tuple(x / 2 for x in d)))

        plist = list(prev)
        for d1 in plist:
            for d2 in plist:
                curr.add(dim_round(tuple(a + b for a, b in zip(d1, d2))))
                curr.add(dim_round(tuple(a - b for a, b in zip(d1, d2))))

        if len(curr) > int(max_set):
            keep = set(d for d in must_keep if d in curr)
            if len(keep) > int(max_set):
                keep = set(sorted(keep, key=lambda d: _dim_l1(d, anchors))[: int(max_set)])
            else:
                budget = int(max_set) - len(keep)
                rest = [d for d in curr if d not in keep]
                rest.sort(key=lambda d: _dim_l1(d, anchors))
                keep.update(rest[:budget])
            curr = keep

        reach.append(frozenset(curr))
    return reach


def rand_node_dim(rng, max_depth, var_dims, target_dim, reach):
    """Generate random expression with given target dimension."""
    ndim = len(target_dim)
    dim0 = (0.0,) * ndim
    nvars = len(var_dims)
    target_dim = dim_round(target_dim)

    matching = [i for i in range(nvars) if dims_eq(var_dims[i], target_dim)]

    if max_depth <= 1:
        return ("var", rng.choice(matching)) if matching else None

    if matching and rng.random() < 0.20:
        return ("var", rng.choice(matching))

    prev = None
    if reach is not None and max_depth >= 2:
        prev = reach[min(max_depth - 1, len(reach) - 1)]

    split_cands = set(dim_round(tuple(d)) for d in var_dims)
    split_cands.add(dim0)
    split_cands.add(target_dim)
    if prev is not None:
        split_cands.update(prev)

    base_list = list(split_cands)
    for d in base_list:
        split_cands.add(dim_round(tuple(2 * x for x in d)))
        split_cands.add(dim_round(tuple(0.5 * x for x in d)))
    for i in range(nvars):
        for j in range(nvars):
            split_cands.add(dim_round(tuple(a + b for a, b in zip(var_dims[i], var_dims[j]))))
            split_cands.add(dim_round(tuple(a - b for a, b in zip(var_dims[i], var_dims[j]))))

    split_cands = list(split_cands)

    mul_splits = []
    for da in split_cands:
        db = dim_round(tuple(t - a for t, a in zip(target_dim, da)))
        if db in split_cands:
            mul_splits.append((da, db))

    div_splits = []
    for db in split_cands:
        da = dim_round(tuple(t + b for t, b in zip(target_dim, db)))
        if da in split_cands:
            div_splits.append((da, db))

    if prev is not None:
        def _split_score(pair):
            a, b = pair
            ina = a in prev
            inb = b in prev
            return 0 if (ina and inb) else (1 if (ina or inb) else 2)

        mul_splits.sort(key=_split_score)
        div_splits.sort(key=_split_score)

    ops = []
    ops.extend(["add", "sub", "neg"])
    if mul_splits:
        ops.extend(["mul", "mul"])
    if div_splits:
        ops.append("div")

    double = dim_round(tuple(2 * x for x in target_dim))
    half = dim_round(tuple(x / 2 for x in target_dim))
    ops.append("sqrt")
    ops.append("sqr")

    if dims_eq(target_dim, dim0):
        ops.extend(["sin", "cos"])
        if rng.random() < 0.3:
            ops.extend(["exp", "log"])

    rng.shuffle(ops)

    for op in ops[:10]:
        if op == "neg":
            c = rand_node_dim(rng, max_depth - 1, var_dims, target_dim, reach)
            if c is not None:
                return ("neg", c)

        elif op in ("add", "sub"):
            a = rand_node_dim(rng, max_depth - 1, var_dims, target_dim, reach)
            if a is None:
                continue
            b = rand_node_dim(rng, max_depth - 1, var_dims, target_dim, reach)
            if b is None:
                continue
            return (op, a, b)

        elif op == "mul" and mul_splits:
            da, db = rng.choice(mul_splits[: max(1, min(len(mul_splits), 64))])
            a = rand_node_dim(rng, max_depth - 1, var_dims, da, reach)
            if a is None:
                continue
            b = rand_node_dim(rng, max_depth - 1, var_dims, db, reach)
            if b is None:
                continue
            return ("mul", a, b)

        elif op == "div" and div_splits:
            da, db = rng.choice(div_splits[: max(1, min(len(div_splits), 64))])
            a = rand_node_dim(rng, max_depth - 1, var_dims, da, reach)
            if a is None:
                continue
            b = rand_node_dim(rng, max_depth - 1, var_dims, db, reach)
            if b is None:
                continue
            return ("div", a, b)

        elif op == "sqrt":
            c = rand_node_dim(rng, max_depth - 1, var_dims, double, reach)
            if c is not None:
                return ("sqrt", c)

        elif op == "sqr":
            c = rand_node_dim(rng, max_depth - 1, var_dims, half, reach)
            if c is not None:
                return ("sqr", c)

        elif op in ("sin", "cos", "exp", "log"):
            c = rand_node_dim(rng, max_depth - 1, var_dims, dim0, reach)
            if c is not None:
                return (op, c)

    return ("var", rng.choice(matching)) if matching else None


def _simplify_once(node):
    """One bottom-up pass of peephole simplification."""
    op = node[0]

    if op in ("var", "const"):
        return node

    if op in UNARY_OPS or op in ("asin", "acos"):
        child = _simplify_once(node[1])
        if op == "neg" and child[0] == "neg":
            return child[1]
        if op == "neg" and child[0] == "const":
            return ("const", -child[1])
        if op == "exp" and child[0] == "log":
            return child[1]
        if op == "log" and child[0] == "exp":
            return child[1]
        if op == "log" and child[0] == "mul":
            left, right = child[1], child[2]
            if left[0] == "exp":
                return simplify(("add", right if right[0] == "log" else ("log", right), left[1]))
            if right[0] == "exp":
                return simplify(("add", left if left[0] == "log" else ("log", left), right[1]))
        if op == "log" and child[0] == "div":
            left, right = child[1], child[2]
            if right[0] == "exp":
                return simplify(("sub", left if left[0] == "log" else ("log", left), right[1]))
            if left[0] == "exp":
                return simplify(("sub", left[1], right if right[0] == "log" else ("log", right)))
        if op == "sqr" and child[0] == "neg":
            return ("sqr", child[1])
        if op == "sin" and child[0] == "neg":
            return ("neg", ("sin", child[1]))
        if op == "cos" and child[0] == "neg":
            return ("cos", child[1])
        if child[0] == "const":
            v = child[1]
            try:
                if op == "sqrt" and v >= 0:
                    return ("const", v ** 0.5)
                if op == "sqr":
                    return ("const", v * v)
                if op == "sin":
                    return ("const", math.sin(v))
                if op == "cos":
                    return ("const", math.cos(v))
                if op == "exp" and -20 <= v <= 20:
                    return ("const", math.exp(v))
                if op == "log" and v > 0:
                    return ("const", math.log(v))
                if op == "asin" and -1 <= v <= 1:
                    return ("const", math.asin(v))
                if op == "acos" and -1 <= v <= 1:
                    return ("const", math.acos(v))
            except (ValueError, OverflowError):
                pass
        return (op, child) if child is not node[1] else node

    if op in BINARY_OPS:
        left = _simplify_once(node[1])
        right = _simplify_once(node[2])

        if op == "sub" and left == right:
            return ("const", 0.0)
        if op == "div" and left == right:
            return ("const", 1.0)

        if left[0] == "const" and right[0] == "const":
            a, b = left[1], right[1]
            try:
                if op == "add":
                    return ("const", a + b)
                if op == "sub":
                    return ("const", a - b)
                if op == "mul":
                    return ("const", a * b)
                if op == "div" and abs(b) > 1e-30:
                    return ("const", a / b)
            except (ValueError, OverflowError):
                pass

        if op == "add" and right[0] == "const" and right[1] == 0.0:
            return left
        if op == "add" and left[0] == "const" and left[1] == 0.0:
            return right
        if op == "sub" and right[0] == "const" and right[1] == 0.0:
            return left
        if op == "sub" and left[0] == "const" and left[1] == 0.0:
            return ("neg", right)

        if op == "mul" and right[0] == "const" and right[1] == 1.0:
            return left
        if op == "mul" and left[0] == "const" and left[1] == 1.0:
            return right
        if op == "div" and right[0] == "const" and right[1] == 1.0:
            return left

        if op == "mul" and right[0] == "const" and right[1] == 0.0:
            return ("const", 0.0)
        if op == "mul" and left[0] == "const" and left[1] == 0.0:
            return ("const", 0.0)
        if op == "div" and left[0] == "const" and left[1] == 0.0:
            return ("const", 0.0)

        if op == "sub" and right[0] == "neg":
            return ("add", left, right[1])
        if op == "add" and right[0] == "neg":
            return ("sub", left, right[1])
        if op == "add" and left[0] == "neg":
            return ("sub", right, left[1])
        if op == "sub" and left[0] == "neg":
            return ("neg", ("add", left[1], right))
        if op in ("mul", "div") and left[0] == "neg":
            return ("neg", (op, left[1], right))
        if op in ("mul", "div") and right[0] == "neg":
            return ("neg", (op, left, right[1]))

        if op == "mul" and left == right:
            return ("sqr", left)

        if op in ("add", "mul") and node_str(left) > node_str(right):
            return (op, right, left)

        return (op, left, right) if (left is not node[1] or right is not node[2]) else node

    return node


def simplify(node, max_passes=10):
    """Iteratively apply peephole simplifications until fixed point."""
    for _ in range(max_passes):
        new = _simplify_once(node)
        if new == node:
            return node
        node = new
    return node
