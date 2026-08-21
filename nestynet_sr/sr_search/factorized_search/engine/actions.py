# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Engine-owned factorized symbolic search mutation and crossover actions."""

from __future__ import annotations

import math

import torch

from nestynet_sr.sr_search.model_selection import mapping_cost
from nestynet_sr.sr_search.factorized_search.expr_mapping import mean_squared_error_same_shape

A_REPLACE = 0
A_WRAP_UNARY = 1
A_ADD_RAND = 2
A_MUL_RAND = 3
A_RESIDUAL = 4
A_PRUNE = 5
A_CROSSOVER = 6
A_BOOST = 7
A_INVSTEER = 8
A_REPAIR = 9
A_HOLESEARCH = 10

ACTIONS = [
    A_REPLACE,
    A_WRAP_UNARY,
    A_ADD_RAND,
    A_MUL_RAND,
    A_RESIDUAL,
    A_INVSTEER,
    A_REPAIR,
    A_BOOST,
    A_PRUNE,
    A_CROSSOVER,
    A_HOLESEARCH,
]
ACTION_NAME = {
    A_REPLACE: "replace",
    A_WRAP_UNARY: "wrap_un",
    A_ADD_RAND: "add_rand",
    A_MUL_RAND: "mul_rand",
    A_RESIDUAL: "residual",
    A_INVSTEER: "inv_steer",
    A_REPAIR: "repair_option",
    A_BOOST: "boost",
    A_PRUNE: "prune",
    A_CROSSOVER: "crossover",
    A_HOLESEARCH: "hole_search",
}
ACTION_ID_BY_NAME = {v: k for k, v in ACTION_NAME.items()}


def apply_action_impl(
    node,
    action,
    rng,
    max_depth,
    nvars,
    *,
    var_dims=None,
    reach=None,
    path=None,
    replace_action_id: int,
    wrap_unary_action_id: int,
    add_rand_action_id: int,
    mul_rand_action_id: int,
    prune_action_id: int,
    unary_ops,
    select_action_path_fn,
    action_candidate_paths_fn,
    coerce_guided_path_fn,
    node_dims_fn,
    dims_eq_fn,
    get_at_fn,
    replace_at_fn,
    rand_node_dim_fn,
    rand_node_fn,
    node_depth_fn,
):
    dm = var_dims is not None
    dim0 = (0.0,) * len(var_dims[0]) if dm else None

    if action == replace_action_id:
        path = select_action_path_fn(node, action, rng, path=path)
        if path is None:
            return None
        if dm:
            sub_dim = node_dims_fn(get_at_fn(node, path), var_dims)
            if sub_dim is None:
                return None
            budget = max(1, max_depth - len(path))
            new_sub = rand_node_dim_fn(rng, budget, var_dims, sub_dim, reach)
            if new_sub is None:
                return None
        else:
            new_sub = rand_node_fn(rng, rng.randrange(1, max_depth + 1), nvars)
        result = replace_at_fn(node, path, new_sub)
        if node_depth_fn(result) > max_depth:
            return None if dm else rand_node_fn(rng, max_depth, nvars)
        return result

    if action == wrap_unary_action_id:
        path = select_action_path_fn(node, action, rng, path=path)
        if path is None:
            return None
        sub = get_at_fn(node, path)
        if dm:
            sub_dim = node_dims_fn(sub, var_dims)
            if sub_dim is None:
                return None
            if dims_eq_fn(sub_dim, dim0):
                op = rng.choice(unary_ops)
            else:
                op = "neg"
        else:
            op = rng.choice(unary_ops)
        result = replace_at_fn(node, path, (op, sub))
        if node_depth_fn(result) > max_depth:
            return None if dm else rand_node_fn(rng, max_depth, nvars)
        return result

    if action == add_rand_action_id:
        path = select_action_path_fn(node, action, rng, path=path)
        if path is None:
            return None
        sub = get_at_fn(node, path)
        if dm:
            sub_dim = node_dims_fn(sub, var_dims)
            if sub_dim is None:
                return None
            budget = max(1, max_depth - len(path) - 1)
            other = rand_node_dim_fn(rng, budget, var_dims, sub_dim, reach)
            if other is None:
                return None
        else:
            other = rand_node_fn(rng, rng.randrange(1, max_depth + 1), nvars)
        op = rng.choice(["add", "sub"])
        new = (op, sub, other) if rng.random() < 0.5 else (op, other, sub)
        result = replace_at_fn(node, path, new)
        if node_depth_fn(result) > max_depth:
            return None if dm else rand_node_fn(rng, max_depth, nvars)
        return result

    if action == mul_rand_action_id:
        path = select_action_path_fn(node, action, rng, path=path)
        if path is None:
            return None
        sub = get_at_fn(node, path)
        if dm:
            budget = max(1, max_depth - len(path) - 1)
            other = rand_node_dim_fn(rng, budget, var_dims, dim0, reach)
            if other is None:
                return None
        else:
            other = rand_node_fn(rng, rng.randrange(1, max_depth + 1), nvars)
        op = rng.choice(["mul", "div"])
        if dm and op == "div":
            new = ("div", sub, other)
        else:
            new = (op, sub, other) if rng.random() < 0.5 else (op, other, sub)
        result = replace_at_fn(node, path, new)
        if node_depth_fn(result) > max_depth:
            return None if dm else rand_node_fn(rng, max_depth, nvars)
        return result

    if action == prune_action_id:
        internal = action_candidate_paths_fn(node, action)
        if not internal:
            return None
        guided = coerce_guided_path_fn(path)
        path = guided if guided is not None and guided in set(internal) else rng.choice(internal)
        sub = get_at_fn(node, path)
        if sub[0] in unary_ops:
            child = sub[1]
        else:
            child = sub[1] if rng.random() < 0.5 else sub[2]
        return replace_at_fn(node, path, child)

    raise ValueError(action)


def apply_crossover_action_impl(
    recipient,
    arch,
    parent_key,
    rng,
    max_depth,
    nvars,
    *,
    var_dims=None,
    exploit_frac=0.35,
    exploit_topk=50,
    path=None,
    crossover_action_id: int,
    select_action_path_fn,
    node_dims_fn,
    get_at_fn,
    choose_parent_fn,
    collect_paths_fn,
    replace_at_fn,
    node_depth_fn,
    dims_eq_fn,
):
    _ = nvars
    if len(arch.d) < 2:
        return None
    dm = var_dims is not None

    path = select_action_path_fn(recipient, crossover_action_id, rng, path=path)
    if path is None:
        return None
    sub = get_at_fn(recipient, path)
    budget = max(1, max_depth - len(path))
    sub_dim = node_dims_fn(sub, var_dims) if dm else None

    remaining = [(k, r) for k, r in arch.items() if k != parent_key]
    if not remaining:
        return None
    donor_key, donor_rec = choose_parent_fn(arch, rng, exploit_frac, exploit_topk)
    if donor_key == parent_key or donor_key is None or donor_rec is None:
        donor_key, donor_rec = rng.choice(remaining)

    donor_expr = donor_rec.best_expr
    if getattr(donor_rec, "elites", None):
        elites = sorted(donor_rec.elites, key=lambda e: (e.mse, e.size))
        kk = min(3, len(elites))
        if kk > 0:
            donor_expr = rng.choice(elites[:kk]).expr
    donor_paths = collect_paths_fn(donor_expr)

    rng.shuffle(donor_paths)
    for dp in donor_paths[:5]:
        donor_sub = get_at_fn(donor_expr, dp)
        if node_depth_fn(donor_sub) > budget:
            continue
        if dm:
            donor_dim = node_dims_fn(donor_sub, var_dims)
            if donor_dim is None or sub_dim is None:
                continue
            if not dims_eq_fn(donor_dim, sub_dim):
                continue
        result = replace_at_fn(recipient, path, donor_sub)
        if node_depth_fn(result) <= max_depth:
            return result

    return None


@torch.no_grad()
def apply_residual_action_impl(
    parent_node,
    parent_mapping,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    pool_nodes,
    pool_phi,
    pool_norms,
    pool_dims,
    rng,
    max_depth,
    nvars,
    poly_degree,
    *,
    var_dims=None,
    topk=3,
    complexity_penalty=0.0,
    eval_node_fn,
    eval_mapping_total_fn,
    node_dims_fn,
    dims_eq_fn,
    collect_paths_fn,
    get_at_fn,
    replace_at_fn,
    fit_best_fn,
    eval_mapping_fn,
    node_depth_fn,
    node_size_fn,
):
    _ = nvars
    dm = var_dims is not None
    p = eval_node_fn(parent_node, x_probe)
    y_hat = eval_mapping_total_fn(p, parent_mapping, x_probe)
    r = (y_probe - y_hat).squeeze(-1)

    dots = r @ pool_phi
    scores = dots ** 2 / (pool_norms + 1e-12)

    if dm:
        parent_dim = node_dims_fn(parent_node, var_dims)
        if parent_dim is not None:
            valid = torch.tensor([
                pool_dims[i] is not None and dims_eq_fn(pool_dims[i], parent_dim)
                for i in range(len(pool_nodes))
            ], device=scores.device)
            if valid.any():
                scores = scores.clone()
                scores[~valid] = -float("inf")

    k = min(topk, len(pool_nodes))
    topk_idx = scores.topk(k).indices.tolist()

    # Also compute the base residual (without the linear head).  When the
    # score head has absorbed a significant additive term (e.g. a raw
    # variable), the total residual becomes noise and the correct pool term
    # is invisible.  The base residual preserves the structural gap signal.
    if isinstance(parent_mapping, dict) and parent_mapping.get("_lin_head") is not None:
        y_hat_base = eval_mapping_fn(p, parent_mapping)
        r_base = (y_probe - y_hat_base).squeeze(-1)
        dots_base = r_base @ pool_phi
        scores_base = dots_base ** 2 / (pool_norms + 1e-12)
        if dm and parent_dim is not None and valid.any():
            scores_base = scores_base.clone()
            scores_base[~valid] = -float("inf")
        topk_base = scores_base.topk(k).indices.tolist()
        # Merge: add any base-residual top-k terms not already selected.
        seen = set(topk_idx)
        for idx in topk_base:
            if idx not in seen:
                topk_idx.append(idx)
                seen.add(idx)

    paths = collect_paths_fn(parent_node)
    leaf_paths = [pp for pp in paths if get_at_fn(parent_node, pp)[0] == "var"]

    best_expr = None
    best_mse = float("inf")
    best_score = float("inf")

    for idx in topk_idx:
        phi_node = pool_nodes[idx]
        phi_dim = pool_dims[idx] if dm else None
        candidates = []

        if not dm or (parent_dim is not None and phi_dim is not None and dims_eq_fn(phi_dim, parent_dim)):
            cand = ("add", parent_node, phi_node)
            if node_depth_fn(cand) <= max_depth:
                candidates.append(cand)

        if leaf_paths:
            lp = rng.choice(leaf_paths)
            leaf_dim = node_dims_fn(get_at_fn(parent_node, lp), var_dims) if dm else None
            if not dm or (phi_dim is not None and leaf_dim is not None and dims_eq_fn(phi_dim, leaf_dim)):
                cand = replace_at_fn(parent_node, lp, phi_node)
                if node_depth_fn(cand) <= max_depth:
                    candidates.append(cand)

        if len(paths) > 1:
            sp = rng.choice(paths[1:])
            sub = get_at_fn(parent_node, sp)
            sub_dim = node_dims_fn(sub, var_dims) if dm else None
            if not dm or (phi_dim is not None and sub_dim is not None and dims_eq_fn(phi_dim, sub_dim)):
                cand = replace_at_fn(parent_node, sp, ("add", sub, phi_node))
                if node_depth_fn(cand) <= max_depth:
                    candidates.append(cand)

        for cand in candidates:
            if dm and node_dims_fn(cand, var_dims) is None:
                continue
            try:
                pf = eval_node_fn(cand, x_fit)
            except Exception:
                continue
            if not torch.isfinite(pf).all():
                continue
            fb = fit_best_fn(pf, y_fit, poly_degree)
            if fb is None:
                continue
            cand_mse, cand_map = fb
            try:
                pp = eval_node_fn(cand, x_probe)
            except Exception:
                continue
            if not torch.isfinite(pp).all():
                continue
            yh = eval_mapping_fn(pp, cand_map)
            mse = mean_squared_error_same_shape(y_probe, yh)
            if not math.isfinite(mse):
                continue
            mse_eff = float(
                mse
                + float(complexity_penalty)
                * float(node_size_fn(cand) + mapping_cost(cand_map))
            )
            if (mse_eff < best_score) or (mse_eff == best_score and mse < best_mse):
                best_score = mse_eff
                best_mse = mse
                best_expr = cand

    if best_expr is None:
        return None
    return best_expr


__all__ = [
    "ACTION_ID_BY_NAME",
    "ACTION_NAME",
    "ACTIONS",
    "A_ADD_RAND",
    "A_BOOST",
    "A_CROSSOVER",
    "A_HOLESEARCH",
    "A_INVSTEER",
    "A_MUL_RAND",
    "A_PRUNE",
    "A_REPAIR",
    "A_REPLACE",
    "A_RESIDUAL",
    "A_WRAP_UNARY",
    "apply_action_impl",
    "apply_crossover_action_impl",
    "apply_residual_action_impl",
]
