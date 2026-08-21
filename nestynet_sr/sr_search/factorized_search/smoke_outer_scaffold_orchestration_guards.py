# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Regression checks for outer scaffold orchestration guards.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_outer_scaffold_orchestration_guards.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestynet_sr.sr_search.factorized_search.basis_state import BasisState, FeatureBlock, ProposalContext, basis_state_covers_feature_block
from nestynet_sr.sr_search.factorized_search.engine.proposal_execution import run_outer_scaffold_pass
from nestynet_sr.sr_search.factorized_search.expr_ast import node_str, simplify
from nestynet_sr.sr_search.factorized_search.proposal_families.runner import run_outer_scaffold_pass_impl
from nestynet_sr.sr_search.factorized_search.proposal_families.types import OperatorApplication

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


x_fit = torch.zeros((8, 4), dtype=torch.float32)
y_fit = torch.zeros((8, 1), dtype=torch.float32)
x_probe = torch.zeros((8, 4), dtype=torch.float32)
y_probe = torch.zeros((8, 1), dtype=torch.float32)


print("\n=== Test: runner splits canonical and basis-augmented lanes ===")
lane_calls = []


def fake_enumerator(*, basis_seed_mode="merged", pool_nodes=None, basis_state=None, basis_state_beam=None, **kwargs):
    lane_calls.append(
        {
            "basis_seed_mode": str(basis_seed_mode),
            "pool_nodes": tuple(pool_nodes or ()),
            "basis_state": basis_state,
            "basis_state_beam": tuple(basis_state_beam or ()),
        }
    )
    if str(basis_seed_mode) == "core_only":
        expr = ("mul", ("var", 0), ("sqrt", ("sqr", ("var", 1))))
        return [
            OperatorApplication(
                family="quadratic",
                operator_id="quadratic:sqrt_mul",
                scaffold_id="core:q_norm",
                parent_node=expr,
                hole_path=(2, 1),
            )
        ]
    if str(basis_seed_mode) == "basis_augmented":
        expr = ("mul", ("var", 0), ("sqrt", ("sqr", ("mul", ("var", 0), ("var", 1)))))
        return [
            OperatorApplication(
                family="quadratic",
                operator_id="quadratic:sqrt_mul",
                scaffold_id="aug:q_times_e",
                parent_node=expr,
                hole_path=(2, 1),
            )
        ]
    return []


def fake_direct_solver(spec, **kwargs):
    expr_key = str(node_str(spec.parent_node))
    probe_mse = 1.0e-12 if expr_key == str(node_str(("mul", ("var", 0), ("sqrt", ("sqr", ("var", 1)))))) else 1.0
    return (
        [
            {
                "expr": spec.parent_node,
                "local_probe_mse": float(probe_mse),
                "local_fit_mse": float(probe_mse),
                "candidate_child_size": 3,
                "proposal_key": expr_key,
                "child_key": expr_key,
            }
        ],
        "direct_ok",
        {},
    )


proposal_context = ProposalContext(
    basis_state=BasisState(blocks=()),
    basis_state_beam=(),
    diagnostics={"route": "test"},
    family_hints={},
)
ret = run_outer_scaffold_pass_impl(
    families=["quadratic"],
    nvars=4,
    max_scaffolds=8,
    anchors_per_family=4,
    max_depth=4,
    poly_degree=3,
    x_fit=x_fit,
    y_fit=y_fit,
    x_probe=x_probe,
    y_probe=y_probe,
    var_dims=None,
    y_dims=None,
    pool_nodes=[("mul", ("var", 0), ("var", 1))],
    pool_phi_fit=torch.zeros((8, 1), dtype=torch.float32),
    pool_phi_probe=torch.zeros((8, 1), dtype=torch.float32),
    pool_dims=[None],
    safe_eps=1.0e-6,
    preview_topk=4,
    beam_cfg={},
    solver_kwargs={},
    proposal_context=proposal_context,
    enumerate_operator_applications_fn=fake_enumerator,
    solve_direct_operator_preview_rows_fn=fake_direct_solver,
)
rows = list(ret.get("candidate_rows", []) or [])
row_lanes = {str(row.get("proposal_lane", "")) for row in rows}
check("two lane calls issued", len(lane_calls) == 2, f"calls={lane_calls}")
if len(lane_calls) >= 2:
    core_pool_nodes = lane_calls[0]["pool_nodes"]
    check(
        "core lane uses canonical raw pool",
        lane_calls[0]["basis_seed_mode"] == "core_only"
        and bool(core_pool_nodes)
        and core_pool_nodes != (("mul", ("var", 0), ("var", 1)),)
        and ("var", 0) in core_pool_nodes,
        f"core={lane_calls[0]}",
    )
    check("core lane does not inject basis state", lane_calls[0]["basis_state"] is None and not lane_calls[0]["basis_state_beam"], f"core={lane_calls[0]}")
    check("aug lane sees polluted pool", lane_calls[1]["basis_seed_mode"] == "basis_augmented" and bool(lane_calls[1]["pool_nodes"]), f"aug={lane_calls[1]}")
check("candidate rows keep both lanes", row_lanes == {"core", "basis_augmented"}, f"lanes={sorted(row_lanes)}")


print("\n=== Test: refined basis coverage requires matching structural bundles ===")
state_block = FeatureBlock(
    family="quadratic",
    atoms=(("var", 1), ("var", 2)),
    head_type="linear",
    head_bundle_nodes=(("sqrt", simplify(("add", ("sqr", ("var", 1)), ("sqr", ("var", 2))))),),
    head_bundle_roles=("wrapper",),
    block_id="quadratic:2norm",
)
candidate_block = FeatureBlock(
    family="quadratic",
    atoms=(("var", 1), ("var", 2)),
    head_type="linear",
    head_bundle_nodes=(("sqrt", simplify(("add", simplify(("add", ("sqr", ("var", 1)), ("sqr", ("var", 2)))), ("sqr", ("var", 3))))),),
    head_bundle_roles=("wrapper",),
    block_id="quadratic:3norm",
)
state = BasisState(blocks=(state_block,))
check("different wrapper bundles are not marked covered", not basis_state_covers_feature_block(state, candidate_block))
check("exact block id still counts as covered", basis_state_covers_feature_block(state, state_block))


print("\n=== Test: proposal execution reserves one core row per operator family/spec ===")


def _run_execution(candidate_rows, *, exact_topk):
    scored = []
    stats = {}

    def fake_pass_impl(**kwargs):
        return {
            "candidate_rows": [dict(row) for row in candidate_rows],
            "stats": {
                "families_considered": 1,
                "scaffolds_enumerated": len(candidate_rows),
                "scaffolds_considered": len(candidate_rows),
                "preview_calls": 1,
                "preview_candidates": len(candidate_rows),
                "direct_calls": 1,
                "direct_candidates": len(candidate_rows),
            },
        }

    def fake_score_expr(expr, *, parent_raw_mse=None, stats=None, route_name=None, candidate_meta=None):
        scored.append(str(node_str(expr)))
        return {"expr": expr, "raw_mse": float(candidate_meta.get("local_probe_mse", 0.0) or 0.0)}

    run_outer_scaffold_pass(
        outer_scaffold_enable=True,
        outer_scaffold_stats=stats,
        outer_scaffold_families=["quadratic"],
        outer_scaffold_max_scaffolds=4,
        outer_scaffold_anchors_per_family=2,
        outer_scaffold_preview_topk=4,
        outer_scaffold_exact_topk=int(exact_topk),
        outer_scaffold_min_valid_frac=0.0,
        outer_scaffold_min_confidence=0.0,
        outer_scaffold_periodic_min_valid_scale=1.0,
        outer_scaffold_periodic_min_confidence_scale=1.0,
        outer_scaffold_transport_min_lin_rel=0.0,
        inverse_periodic_path_penalty=0.0,
        inverse_nonperiodic_muldiv_bonus=0.0,
        inverse_nonperiodic_explogsqrt_bonus=0.0,
        inverse_branch_beam_width=1,
        inverse_topk_terms=1,
        inverse_shortlist_mult=1,
        inverse_local_score_mode="probe",
        inverse_micro_search_enable=False,
        inverse_micro_search_max_depth=1,
        inverse_micro_search_beam_width=1,
        inverse_micro_search_topk=1,
        inverse_micro_search_seed_terms=1,
        inverse_target_mode="robust",
        inverse_safe_eps=1.0e-6,
        inverse_confidence_mode="none",
        inverse_confidence_target_gain=0.0,
        inverse_confidence_floor=0.0,
        inverse_full_mapping_penalty=0.0,
        inverse_exact_simple_target_bonus=0.0,
        inverse_additive_descend_penalty=0.0,
        inverse_nonadditive_leaf_penalty=0.0,
        inverse_exact_path_eta=0.0,
        inverse_branch_ambiguity_penalty=0.0,
        inverse_transport_min_effective_n=0.0,
        inverse_spec_regime_metadata={},
        inverse_spec_local_score_mode="probe",
        inverse_spec_enum_max_depth=1,
        inverse_spec_enum_max_trees=1,
        inverse_spec_max_subtree_depth=1,
        inverse_spec_complexity_penalty=0.0,
        inverse_spec_family_battery_enable=False,
        inverse_spec_family_battery_mode="outer",
        inverse_spec_recursive_enable=False,
        inverse_spec_recursive_max_depth=1,
        inverse_spec_recursive_trigger_rel_mse=0.0,
        inverse_spec_recursive_seed_cap=1,
        inverse_spec_recursive_branch_topk=1,
        inverse_spec_recursive_child_topk=1,
        inverse_spec_witness_jets_enable=False,
        inverse_spec_witness_d2_enable=False,
        inverse_spec_witness_max_rows=0,
        inverse_spec_witness_loss_enable=False,
        inverse_spec_witness_grad_weight=0.0,
        inverse_spec_witness_d2_weight=0.0,
        inverse_spec_witness_diag_weight=0.0,
        inverse_spec_witness_physics_weight=0.0,
        inverse_spec_active_var_screen_enable=False,
        inverse_spec_active_var_grad_tol=0.0,
        inverse_spec_active_var_max_count=0,
        wall_time_deadline=None,
        wall_time_limit_s=10.0,
        max_depth=2,
        poly_degree=2,
        nvars=4,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=None,
        y_dims=None,
        boost_pool_nodes=(),
        boost_pool_phi_fit=torch.zeros((8, 1), dtype=torch.float32),
        boost_pool_phi=torch.zeros((8, 1), dtype=torch.float32),
        boost_pool_dims=(),
        dm=False,
        wall_time_exceeded_fn=lambda: False,
        run_outer_scaffold_pass_impl=fake_pass_impl,
        score_external_candidate_expr_fn=fake_score_expr,
        score_native_candidate_basis_state_fn=None,
        node_str_fn=node_str,
        proposal_context=None,
        family_allocator_fn=None,
    )
    return scored, stats


core_expr = ("var", 0)
aug_expr = ("var", 1)
scored_rows, scored_stats = _run_execution(
    [
        {
            "expr": aug_expr,
            "local_probe_mse": 1.0e-3,
            "local_fit_mse": 1.0e-3,
            "candidate_child_size": 1,
            "proposal_key": "aug",
            "child_key": "aug",
            "proposal_lane": "basis_augmented",
            "operator_id": "quadratic:sqrt_mul",
            "scaffold_family": "quadratic",
            "scaffold_id": "aug",
        },
        {
            "expr": core_expr,
            "local_probe_mse": 2.0e-3,
            "local_fit_mse": 2.0e-3,
            "candidate_child_size": 1,
            "proposal_key": "core",
            "child_key": "core",
            "proposal_lane": "core",
            "operator_id": "quadratic:sqrt_mul",
            "scaffold_family": "quadratic",
            "scaffold_id": "core",
        },
    ],
    exact_topk=1,
)
check("core reservation scores canonical row before better aug row", scored_rows == [str(node_str(core_expr))], f"scored={scored_rows}")
check("core reservation counter increments", int(scored_stats.get("basis_state_core_lane_reservations", 0) or 0) >= 1, f"stats={scored_stats.get('basis_state_core_lane_reservations', None)}")


print("\n=== Test: fast-track rows bypass the exact_topk competition ===")
fast_expr = ("var", 2)
regular_expr = ("var", 3)
fast_rows, fast_stats = _run_execution(
    [
        {
            "expr": regular_expr,
            "local_probe_mse": 1.0e-3,
            "local_fit_mse": 1.0e-3,
            "candidate_child_size": 1,
            "proposal_key": "regular",
            "child_key": "regular",
            "proposal_lane": "basis_augmented",
            "operator_id": "quadratic:sqrt",
            "scaffold_family": "quadratic",
            "scaffold_id": "regular",
        },
        {
            "expr": fast_expr,
            "local_probe_mse": 1.0e-12,
            "local_fit_mse": 1.0e-12,
            "candidate_child_size": 1,
            "proposal_key": "fast",
            "child_key": "fast",
            "proposal_lane": "basis_augmented",
            "operator_id": "quadratic:sqrt_mul",
            "scaffold_family": "quadratic",
            "scaffold_id": "fast",
            "preview_fasttrack": True,
        },
    ],
    exact_topk=1,
)
check("fast-track row is scored first", fast_rows[:1] == [str(node_str(fast_expr))], f"scored={fast_rows}")
check("fast-track row does not consume exact budget", len(fast_rows) == 2, f"scored={fast_rows}")
check("fast-track score counter increments", int(fast_stats.get("basis_state_fasttrack_scored", 0) or 0) == 1, f"stats={fast_stats.get('basis_state_fasttrack_scored', None)}")


print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
