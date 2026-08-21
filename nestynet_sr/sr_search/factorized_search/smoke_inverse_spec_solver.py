# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Focused regression checks for inverse direct-spec proposal generation.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_inverse_spec_solver.py
"""

from __future__ import annotations

import importlib
import random
import sys

import torch

from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node, node_dims, node_str
from nestynet_sr.sr_search.factorized_search.inverse_action import (
    _estimate_inverse_action_transport,
    _group_inverse_action_preview_rows,
    _inverse_action_path_mode_beam_states,
)
from nestynet_sr.sr_search.factorized_search.inverse_spec_solver import solve_inverse_spec_preview_rows


n_pass = 0
n_fail = 0


def check(name, ok, detail=""):
    global n_pass, n_fail
    if ok:
        n_pass += 1
        print(f"  PASS  {name}  {detail}")
    else:
        n_fail += 1
        print(f"  FAIL  {name}  {detail}")


def _safe_pool_phi(pool_nodes, x):
    cols = []
    for node in pool_nodes:
        try:
            vals = eval_node(node, x).squeeze(-1)
        except Exception:
            vals = torch.zeros((int(x.shape[0]),), dtype=x.dtype)
        if not torch.isfinite(vals).all():
            vals = torch.zeros_like(vals)
        cols.append(vals)
    return torch.stack(cols, dim=1)


def _make_problem(truth_expr, candidate_expr, *, nvars, var_dims=None, seed=0):
    g_fit = torch.Generator().manual_seed(int(seed))
    g_probe = torch.Generator().manual_seed(int(seed) + 17)
    x_fit = 0.5 + 1.5 * torch.rand((96, int(nvars)), generator=g_fit, dtype=torch.float64)
    x_probe = 0.5 + 1.5 * torch.rand((192, int(nvars)), generator=g_probe, dtype=torch.float64)
    y_fit = eval_node(truth_expr, x_fit)
    y_probe = eval_node(truth_expr, x_probe)
    fb = explorer.fit_best(eval_node(candidate_expr, x_fit), y_fit, 4)
    if fb is None:
        raise RuntimeError("candidate mapping fit failed")
    _fit_mse, mapping = fb
    pool_nodes = explorer.build_pool(int(nvars))
    pool_dims = [node_dims(node, var_dims) for node in pool_nodes] if var_dims is not None else [None] * len(pool_nodes)
    pool_phi_fit = _safe_pool_phi(pool_nodes, x_fit)
    pool_phi_probe = _safe_pool_phi(pool_nodes, x_probe)
    return {
        "truth_expr": truth_expr,
        "candidate_expr": candidate_expr,
        "mapping": mapping,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_probe": x_probe,
        "y_probe": y_probe,
        "pool_nodes": pool_nodes,
        "pool_dims": pool_dims,
        "pool_phi_fit": pool_phi_fit,
        "pool_phi_probe": pool_phi_probe,
    }


def _run_inverse_action(problem, *, candidate_paths, inverse_spec_enable=False, **kwargs):
    return explorer.apply_inverse_steering_action(
        problem["candidate_expr"],
        problem["mapping"],
        problem["x_fit"],
        problem["y_fit"],
        problem["x_probe"],
        problem["y_probe"],
        problem["pool_nodes"],
        problem["pool_phi_fit"],
        problem["pool_phi_probe"],
        problem["pool_dims"],
        random.Random(0),
        4,
        int(problem["x_fit"].shape[1]),
        4,
        var_dims=kwargs.get("var_dims", None),
        max_paths=1,
        topk_terms=6,
        shortlist_mult=4,
        inverse_spec_enable=bool(inverse_spec_enable),
        candidate_paths=candidate_paths,
        return_meta=True,
        **{k: v for k, v in kwargs.items() if k != "var_dims"},
    )


def _build_beam_state(problem, *, path, var_dims=None):
    transport_ctx = _estimate_inverse_action_transport(
        problem["candidate_expr"],
        problem["mapping"],
        problem["x_fit"],
        problem["y_fit"],
        problem["x_probe"],
        problem["y_probe"],
        [path],
        safe_eps=1.0e-12,
    )
    beam_states = _inverse_action_path_mode_beam_states(
        parent_node=problem["candidate_expr"],
        parent_mapping=problem["mapping"],
        x_fit=problem["x_fit"],
        y_fit=problem["y_fit"],
        x_probe=problem["x_probe"],
        y_probe=problem["y_probe"],
        pool_nodes=problem["pool_nodes"],
        pool_phi_fit=problem["pool_phi_fit"],
        pool_phi_probe=problem["pool_phi_probe"],
        pool_dims=problem["pool_dims"],
        all_paths=[path],
        path_target_modes=None,
        transport_ctx=transport_ctx,
        cfg={
            "max_paths": 1,
            "dm": bool(var_dims is not None),
            "var_dims": var_dims,
            "max_depth": 4,
            "poly_degree": 4,
            "topk_terms": 6,
            "shortlist_mult": 4,
            "local_mode": "affine",
            "min_valid_frac": 0.25,
            "min_confidence": 0.10,
            "safe_eps": 1.0e-12,
            "confidence_mode": "conditioning",
            "confidence_target_gain": 4.0,
            "confidence_floor": 0.05,
            "branch_beam_width": 1,
            "micro_search_enable": False,
            "micro_search_max_depth": 3,
            "micro_search_beam_width": 24,
            "micro_search_topk": 16,
            "micro_search_seed_terms": 8,
            "target_mode": "robust",
            "full_mapping_penalty": 0.75,
            "exact_simple_target_bonus": 0.10,
            "additive_descend_penalty": 0.15,
            "nonadditive_leaf_penalty": 0.20,
            "exact_path_eta": 0.98,
            "exact_transport_min_lin_rel": 0.0,
            "periodic_min_valid_scale": 1.25,
            "periodic_min_confidence_scale": 1.35,
            "periodic_path_penalty": 0.65,
            "nonperiodic_muldiv_bonus": 0.10,
            "nonperiodic_explogsqrt_bonus": 0.05,
            "branch_ambiguity_penalty": 0.50,
            "transport_min_lin_rel": 0.02,
            "transport_min_effective_n": 8.0,
        },
        beam_width=1,
    )
    if not beam_states:
        raise RuntimeError("failed to build inverse beam state")
    return beam_states[0]


print("\n=== Test: inverse_spec flag-off parity ===")
truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
candidate_expr = ("add", ("var", 0), ("var", 2))
problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=3)
path = (1,)
expr_a, meta_a = _run_inverse_action(problem, candidate_paths=[path])
expr_b, meta_b = _run_inverse_action(
    problem,
    candidate_paths=[path],
    inverse_spec_enable=False,
    inverse_spec_enum_max_depth=2,
    inverse_spec_enum_max_trees=128,
    inverse_spec_preview_topk=6,
)
expr_a_str = "" if expr_a is None else node_str(expr_a)
expr_b_str = "" if expr_b is None else node_str(expr_b)
check("flag-off best expr parity", expr_a_str == expr_b_str, f"a={expr_a_str} b={expr_b_str}")
check(
    "flag-off slate parity",
    [row.get("child_key", "") for row in meta_a.get("inverse_repair_slate", [])]
    == [row.get("child_key", "") for row in meta_b.get("inverse_repair_slate", [])],
)


print("\n=== Test: inverse_spec simple corrupt-hole recovery ===")
expr_c, meta_c = _run_inverse_action(
    problem,
    candidate_paths=[path],
    inverse_spec_enable=True,
    inverse_spec_enum_max_depth=2,
    inverse_spec_enum_max_trees=256,
    inverse_spec_preview_topk=8,
)
truth_key = node_str(truth_expr)
slate_keys = [row.get("child_key", "") for row in meta_c.get("inverse_repair_slate", [])]
check("inverse_spec used", bool(meta_c.get("inverse_spec_used", False)))
check("truth candidate reaches slate", truth_key in slate_keys, f"truth_key={truth_key}")
expr_c_str = "" if expr_c is None else node_str(expr_c)
# Best may be a semantically equivalent expression (e.g. x0/(1/x1) == x0*x1),
# so check the truth is in the slate and the best achieves very low MSE.
best_mse = float(meta_c.get("inverse_repair_slate", [{}])[0].get("local_probe_mse", float("inf"))) if meta_c.get("inverse_repair_slate") else float("inf")
check("best repair has near-zero MSE", best_mse < 1e-4, f"best={expr_c_str} mse={best_mse:.2e}")


print("\n=== Test: inverse_spec recursive decomposition extends depth reach ===")
truth_recursive_expr = (
    "add",
    ("add", ("mul", ("var", 0), ("var", 1)), ("sin", ("var", 2))),
    ("var", 3),
)
candidate_recursive_expr = ("add", ("var", 0), ("var", 3))
recursive_problem = _make_problem(truth_recursive_expr, candidate_recursive_expr, nvars=4, seed=11)
recursive_beam_state = _build_beam_state(recursive_problem, path=path, var_dims=None)
flat_spec = solve_inverse_spec_preview_rows(
    parent_node=recursive_problem["candidate_expr"],
    beam_state=recursive_beam_state,
    beam_rank=0,
    slate_id="recursive-flat",
    max_depth=4,
    nvars=4,
    poly_degree=4,
    pool_nodes=recursive_problem["pool_nodes"],
    pool_dims=recursive_problem["pool_dims"],
    enum_max_depth=2,
    enum_max_trees=256,
    preview_topk=8,
    recursive_enable=False,
)
recursive_spec = solve_inverse_spec_preview_rows(
    parent_node=recursive_problem["candidate_expr"],
    beam_state=recursive_beam_state,
    beam_rank=0,
    slate_id="recursive-on",
    max_depth=4,
    nvars=4,
    poly_degree=4,
    pool_nodes=recursive_problem["pool_nodes"],
    pool_dims=recursive_problem["pool_dims"],
    enum_max_depth=2,
    enum_max_trees=256,
    preview_topk=8,
    recursive_enable=True,
    recursive_max_depth=2,
    recursive_trigger_rel_mse=0.0,
    recursive_seed_cap=6,
    recursive_branch_topk=4,
    recursive_child_topk=2,
)
recursive_truth_key = node_str(truth_recursive_expr)
flat_recursive_keys = [row.get("child_key", "") for row in flat_spec["rows"]]
recursive_keys = [row.get("child_key", "") for row in recursive_spec["rows"]]
check("flat direct-spec misses depth-3 truth", recursive_truth_key not in flat_recursive_keys, f"flat={flat_recursive_keys}")
check("recursive direct-spec uses recursion", bool(recursive_spec["solver_meta"].get("recursive_used", False)))
check("recursive direct-spec finds truth", recursive_truth_key in recursive_keys, f"rec={recursive_keys}")


print("\n=== Test: inverse_spec dimensional validity ===")
truth_dim_expr = ("mul", ("sin", ("var", 1)), ("var", 0))
candidate_dim_expr = ("mul", ("cos", ("var", 1)), ("var", 0))
var_dims = ((1.0,), (0.0,))
dim_problem = _make_problem(truth_dim_expr, candidate_dim_expr, nvars=2, var_dims=var_dims, seed=7)
beam_state_dim = _build_beam_state(dim_problem, path=path, var_dims=var_dims)
spec_rows = solve_inverse_spec_preview_rows(
    parent_node=dim_problem["candidate_expr"],
    beam_state=beam_state_dim,
    beam_rank=0,
    slate_id="dim-smoke",
    max_depth=4,
    nvars=2,
    poly_degree=4,
    var_dims=var_dims,
    pool_nodes=dim_problem["pool_nodes"],
    pool_dims=dim_problem["pool_dims"],
    enum_max_depth=2,
    enum_max_trees=128,
    preview_topk=6,
)["rows"]
all_dim_valid = True
for row in spec_rows:
    all_dim_valid = all_dim_valid and (node_dims(row["expr"], var_dims) is not None)
check("dimensional direct-spec emits rows", len(spec_rows) > 0, f"n_rows={len(spec_rows)}")
check("all direct-spec full expressions keep valid dimensions", all_dim_valid, f"n_rows={len(spec_rows)}")


print("\n=== Test: dedup keeps best evidence and provenance ===")
row_a = {
    "expr": truth_expr,
    "child_key": truth_key,
    "child_expr": truth_key,
    "path": path,
    "target_mode": "full",
    "local_mapping_kind": "affine",
    "local_probe_mse": 0.5,
    "local_fit_mse": 0.4,
    "candidate_subtree_size": 3,
    "beam_rank": 1,
    "local_rank": 2,
    "tuple_provenance": "beam_local_repair",
}
row_b = {
    "expr": truth_expr,
    "child_key": truth_key,
    "child_expr": truth_key,
    "path": path,
    "target_mode": "full",
    "local_mapping_kind": "affine",
    "local_probe_mse": 0.1,
    "local_fit_mse": 0.08,
    "candidate_subtree_size": 2,
    "beam_rank": 0,
    "local_rank": 1,
    "tuple_provenance": "inverse_spec_sr",
    "proposal_family": "inverse_spec_sr",
    "generation_source": "inverse_spec_solver",
}
row_c = {
    "expr": candidate_expr,
    "child_key": node_str(candidate_expr),
    "child_expr": node_str(candidate_expr),
    "path": path,
    "target_mode": "full",
    "local_mapping_kind": "affine",
    "local_probe_mse": 1.0,
    "local_fit_mse": 0.9,
    "candidate_subtree_size": 1,
    "beam_rank": 0,
    "local_rank": 0,
    "tuple_provenance": "beam_local_repair",
}
grouped_rows, dup_rows = _group_inverse_action_preview_rows([row_a, row_b, row_c])
best_truth_row = next(row for row in grouped_rows if row.get("child_key", "") == truth_key)
check("dedup keeps better preview row", best_truth_row.get("tuple_provenance", "") == "inverse_spec_sr")
check("dedup preserves provenance_count", int(best_truth_row.get("provenance_count", 0) or 0) == 2)
check("dedup preserves provenance_rows", len(list(best_truth_row.get("provenance_rows", []) or [])) == 2)
check("duplicate map groups both rows", len(dup_rows.get(truth_key, [])) == 2)


print("\n=== Test: periodic forward solver recovers trig wrapper ===")
# Truth has cos((x0*x1)*x2) at depth 4.  Flat and recursive inverse-spec
# cannot find it because arccos inversion fails on multi-period arguments.
# The periodic forward solver should find it by fitting t ≈ a sin(g) + b cos(g) + c.
truth_trig_expr = ("add", ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 3)), ("cos", ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 2))))
candidate_trig_expr = ("add", ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 3)), ("const", 1.0))
trig_var_dims = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
trig_problem = _make_problem(truth_trig_expr, candidate_trig_expr, nvars=4, var_dims=trig_var_dims, seed=42)
trig_beam_state = _build_beam_state(trig_problem, path=(2,), var_dims=trig_var_dims)
trig_truth_key = node_str(truth_trig_expr)

# Flat solver should NOT find it (depth-4 truth, depth-2 enum)
flat_trig = solve_inverse_spec_preview_rows(
    parent_node=trig_problem["candidate_expr"],
    beam_state=trig_beam_state, beam_rank=0, slate_id="trig-flat",
    max_depth=6, nvars=4, poly_degree=4,
    var_dims=trig_var_dims, pool_nodes=trig_problem["pool_nodes"], pool_dims=trig_problem["pool_dims"],
    enum_max_depth=2, enum_max_trees=256, preview_topk=32,
    recursive_enable=False,
)
flat_trig_keys = [r.get("child_key", "") for r in flat_trig["rows"]]
check("flat solver misses trig truth", trig_truth_key not in flat_trig_keys)

# Periodic forward solver SHOULD find it
periodic_trig = solve_inverse_spec_preview_rows(
    parent_node=trig_problem["candidate_expr"],
    beam_state=trig_beam_state, beam_rank=0, slate_id="trig-periodic",
    max_depth=6, nvars=4, poly_degree=4,
    var_dims=trig_var_dims, pool_nodes=trig_problem["pool_nodes"], pool_dims=trig_problem["pool_dims"],
    enum_max_depth=2, enum_max_trees=256, preview_topk=32, complexity_penalty=1e-6,
    recursive_enable=True, recursive_max_depth=2, recursive_trigger_rel_mse=0.0,
    recursive_seed_cap=6, recursive_branch_topk=4, recursive_child_topk=2,
    safe_eps=1e-12, confidence_mode="conditioning",
    confidence_target_gain=4.0, confidence_floor=0.05,
    branch_beam_width=1,
)
periodic_trig_keys = [r.get("child_key", "") for r in periodic_trig["rows"]]
check("periodic forward solver uses periodic mode", bool(periodic_trig["solver_meta"].get("periodic_forward_used", False)))
check("periodic forward solver finds trig truth", trig_truth_key in periodic_trig_keys, f"truth={trig_truth_key}")
if trig_truth_key in periodic_trig_keys:
    truth_rank = periodic_trig_keys.index(trig_truth_key) + 1
    check("periodic forward truth ranks in top-3", truth_rank <= 3, f"rank={truth_rank}")


print("\n=== Test: import smoke ===")
mod_solver = importlib.import_module("nestynet_sr.sr_search.factorized_search.inverse_spec_solver")
mod_action = importlib.import_module("nestynet_sr.sr_search.factorized_search.inverse_action")
mod_explorer = importlib.import_module("nestynet_sr.sr_search.factorized_search.explorer")
check("inverse_spec_solver import", hasattr(mod_solver, "solve_inverse_spec_preview_rows"))
check("inverse_action import", hasattr(mod_action, "run_inverse_steering_action"))
check("explorer import", hasattr(mod_explorer, "apply_inverse_steering_action"))


print(f"\n{'=' * 50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
