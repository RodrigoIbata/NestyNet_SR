# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.sr_search.factorized_search.basis_head import fit_basis_state_head
from nestynet_sr.sr_search.factorized_search.basis_state import (
    BasisState,
    FeatureBlock,
    ProposalContext,
    basis_state_from_closure_candidate,
)
from nestynet_sr.sr_search.factorized_search.engine.proposal_execution import run_closure_search_pass


def _make_var_block_state(var_idx: int) -> BasisState:
    expr = ("var", int(var_idx))
    return basis_state_from_closure_candidate(
        family="basis",
        scaffold_id=f"basis:var:{int(var_idx)}",
        expr=expr,
        anchor_node=None,
        scaffold_metadata={"form": "var_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        local_mapping_coeffs=[1.0, 0.0],
        direct_metadata={"feature_node": expr, "hole_node": expr},
    )


def _make_expr_block_state(expr: tuple, *, scaffold_id: str) -> BasisState:
    return basis_state_from_closure_candidate(
        family="basis",
        scaffold_id=str(scaffold_id),
        expr=expr,
        anchor_node=None,
        scaffold_metadata={"form": "expr_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        local_mapping_coeffs=[1.0, 0.0],
        direct_metadata={"feature_node": expr, "hole_node": expr},
    )


def _base_closure_search_kwargs():
    return {
        "closure_search_enable": True,
        "closure_search_stats": {},
        "closure_search_families": ["periodic", "exp"],
        "closure_search_max_proposals": 4,
        "closure_search_anchors_per_family": 2,
        "closure_search_preview_topk": 2,
        "closure_search_exact_topk": 1,
        "closure_search_min_valid_frac": 0.1,
        "closure_search_min_confidence": 0.1,
        "closure_search_periodic_min_valid_scale": 1.0,
        "closure_search_periodic_min_confidence_scale": 1.0,
        "closure_search_transport_min_lin_rel": 0.0,
        "inverse_periodic_path_penalty": 0.0,
        "inverse_nonperiodic_muldiv_bonus": 0.0,
        "inverse_nonperiodic_explogsqrt_bonus": 0.0,
        "inverse_branch_beam_width": 4,
        "inverse_topk_terms": 4,
        "inverse_shortlist_mult": 1,
        "inverse_local_score_mode": "mse",
        "inverse_micro_search_enable": False,
        "inverse_micro_search_max_depth": 2,
        "inverse_micro_search_beam_width": 2,
        "inverse_micro_search_topk": 2,
        "inverse_micro_search_seed_terms": 2,
        "inverse_target_mode": "full",
        "inverse_safe_eps": 1.0e-8,
        "inverse_confidence_mode": "gain",
        "inverse_confidence_target_gain": 0.0,
        "inverse_confidence_floor": 0.0,
        "inverse_full_mapping_penalty": 0.0,
        "inverse_exact_simple_target_bonus": 0.0,
        "inverse_additive_descend_penalty": 0.0,
        "inverse_nonadditive_leaf_penalty": 0.0,
        "inverse_exact_path_eta": 0.0,
        "inverse_branch_ambiguity_penalty": 0.0,
        "inverse_transport_min_effective_n": 0.0,
        "inverse_spec_regime_metadata": None,
        "inverse_spec_local_score_mode": "mse",
        "inverse_spec_enum_max_depth": 2,
        "inverse_spec_enum_max_trees": 16,
        "inverse_spec_max_subtree_depth": None,
        "inverse_spec_complexity_penalty": 0.0,
        "inverse_spec_family_battery_enable": False,
        "inverse_spec_family_battery_mode": "outer",
        "inverse_spec_recursive_enable": False,
        "inverse_spec_recursive_max_depth": 2,
        "inverse_spec_recursive_trigger_rel_mse": 0.0,
        "inverse_spec_recursive_seed_cap": 2,
        "inverse_spec_recursive_branch_topk": 2,
        "inverse_spec_recursive_child_topk": 2,
        "inverse_spec_witness_jets_enable": False,
        "inverse_spec_witness_d2_enable": False,
        "inverse_spec_witness_max_rows": 16,
        "inverse_spec_witness_loss_enable": False,
        "inverse_spec_witness_grad_weight": 0.0,
        "inverse_spec_witness_d2_weight": 0.0,
        "inverse_spec_witness_diag_weight": 0.0,
        "inverse_spec_witness_physics_weight": 0.0,
        "inverse_spec_active_var_screen_enable": False,
        "inverse_spec_active_var_grad_tol": 0.0,
        "inverse_spec_active_var_max_count": 0,
        "wall_time_deadline": None,
        "wall_time_limit_s": None,
        "max_depth": 5,
        "poly_degree": 2,
        "nvars": 2,
        "x_fit": None,
        "y_fit": None,
        "x_probe": None,
        "y_probe": None,
        "var_dims": None,
        "y_dims": None,
        "boost_pool_nodes": [],
        "boost_pool_phi_fit": None,
        "boost_pool_phi": None,
        "boost_pool_dims": [],
        "dm": False,
        "wall_time_exceeded_fn": lambda: False,
        "node_str_fn": lambda expr: str(expr),
    }


def test_fit_basis_state_head_refits_global_linear_head_from_blocks():
    state_x0 = _make_var_block_state(0)
    state_x1 = _make_var_block_state(1)
    state = BasisState(
        blocks=(state_x0.blocks[0], state_x1.blocks[0]),
        fit_loss=1.0,
        probe_loss=1.0,
        complexity=state_x0.blocks[0].complexity() + state_x1.blocks[0].complexity(),
        compiled_expr=("var", 0),
    )
    x = torch.tensor(
        [
            [0.2, 0.7],
            [0.4, 0.5],
            [0.8, 0.1],
            [1.2, 0.3],
            [1.7, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.0).unsqueeze(-1)

    refit = fit_basis_state_head(
        state,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        route_name="test_basis_head",
    )

    assert refit is not None
    assert float(refit.fit_loss) < 1.0e-20
    assert float(refit.probe_loss) < 1.0e-20
    basis_head = dict(refit.fit_bundle or {}).get("basis_head", {})
    coeffs = list(basis_head.get("coeffs", []))
    assert len(coeffs) == 2
    assert abs(float(coeffs[0]) - 2.0) < 1.0e-8
    assert abs(float(coeffs[1]) + 0.5) < 1.0e-8
    assert abs(float(basis_head.get("intercept", 0.0)) - 1.0) < 1.0e-8
    assert refit.residual_probe is not None
    assert "x1" in refit.to_dict()["compiled_expr"]


def test_fit_basis_state_head_backward_prunes_zero_weight_blocks():
    state_x0 = _make_var_block_state(0)
    state_x1 = _make_var_block_state(1)
    state_x2 = _make_var_block_state(2)
    state = BasisState(
        blocks=(state_x0.blocks[0], state_x1.blocks[0], state_x2.blocks[0]),
        fit_loss=1.0,
        probe_loss=1.0,
        complexity=sum(block.complexity() for block in (state_x0.blocks[0], state_x1.blocks[0], state_x2.blocks[0])),
        compiled_expr=("var", 0),
    )
    x = torch.tensor(
        [
            [0.2, 0.7, 1.3],
            [0.4, 0.5, 0.1],
            [0.8, 0.1, 2.0],
            [1.2, 0.3, 0.8],
            [1.7, 0.9, 1.6],
        ],
        dtype=torch.float64,
    )
    y = (2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.0).unsqueeze(-1)

    refit = fit_basis_state_head(
        state,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        route_name="test_basis_head_prune",
    )

    assert refit is not None
    assert len(refit.blocks) == 2
    block_exprs = [block.to_dict()["atom_exprs"][0] for block in refit.blocks]
    assert "x0" in block_exprs
    assert "x1" in block_exprs
    assert "x2" not in block_exprs
    basis_head = dict(refit.fit_bundle or {}).get("basis_head", {})
    assert int(basis_head.get("pruned_block_count", 0)) == 1
    assert bool(refit.diagnostics.get("basis_head_pruned", False)) is True
    assert "x2" not in refit.to_dict()["compiled_expr"]


def test_fit_basis_state_head_can_prune_to_constant_only_state():
    state_x0 = _make_var_block_state(0)
    state = BasisState(
        blocks=(state_x0.blocks[0],),
        fit_loss=1.0,
        probe_loss=1.0,
        complexity=float(state_x0.blocks[0].complexity()),
        compiled_expr=("var", 0),
    )
    x = torch.tensor(
        [
            [0.2],
            [0.4],
            [0.8],
            [1.2],
            [1.7],
        ],
        dtype=torch.float64,
    )
    y = torch.full((int(x.shape[0]), 1), 2.5, dtype=torch.float64)

    refit = fit_basis_state_head(
        state,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        route_name="test_basis_head_const_prune",
    )

    assert refit is not None
    assert len(refit.blocks) == 0
    assert float(refit.fit_loss) < 1.0e-20
    basis_head = dict(refit.fit_bundle or {}).get("basis_head", {})
    assert int(basis_head.get("pruned_block_count", 0)) == 1
    assert abs(float(basis_head.get("intercept", 0.0)) - 2.5) < 1.0e-8
    assert refit.to_dict()["compiled_expr"] == "2.5"


def test_fit_basis_state_head_backward_prunes_redundant_collinear_blocks_after_refit():
    state_x0 = _make_var_block_state(0)
    state_2x0 = _make_expr_block_state(("mul", ("const", 2.0), ("var", 0)), scaffold_id="basis:2x0")
    state = BasisState(
        blocks=(state_x0.blocks[0], state_2x0.blocks[0]),
        fit_loss=1.0,
        probe_loss=1.0,
        complexity=state_x0.blocks[0].complexity() + state_2x0.blocks[0].complexity(),
        compiled_expr=("add", ("var", 0), ("mul", ("const", 2.0), ("var", 0))),
    )
    x = torch.tensor(
        [
            [0.2],
            [0.4],
            [0.8],
            [1.2],
            [1.7],
        ],
        dtype=torch.float64,
    )
    y = x[:, [0]]

    refit = fit_basis_state_head(
        state,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        route_name="test_basis_head_subset_prune",
    )

    assert refit is not None
    assert float(refit.fit_loss) < 1.0e-20
    assert float(refit.probe_loss) < 1.0e-20
    assert len(refit.blocks) == 1
    basis_head = dict(refit.fit_bundle or {}).get("basis_head", {})
    assert int(basis_head.get("subset_pruned_block_count", 0)) >= 1
    assert bool(refit.diagnostics.get("basis_head_subset_pruned", False)) is True


def test_fit_basis_state_head_keeps_parent_block_needed_by_retained_child():
    x0 = ("var", 0)
    x1 = ("var", 1)
    sqrt_shift = ("sqrt", ("add", ("const", 1.0), ("sqr", x0)))
    parent_block = FeatureBlock(
        family="basis",
        atoms=(x0,),
        head_type="linear",
        block_id="basis:block:x0",
        head_bundle_nodes=(x0,),
        head_bundle_roles=("primary",),
        metadata={"block_expr_obj": x0},
    )
    child_block = FeatureBlock(
        family="basis",
        atoms=(sqrt_shift,),
        head_type="linear",
        block_id="basis:block:sqrt_shift",
        parent_block_ids=("basis:block:x0",),
        head_bundle_nodes=(sqrt_shift,),
        head_bundle_roles=("primary",),
        metadata={"block_expr_obj": sqrt_shift},
    )
    companion_block = FeatureBlock(
        family="basis",
        atoms=(x1,),
        head_type="linear",
        block_id="basis:block:x1",
        head_bundle_nodes=(x1,),
        head_bundle_roles=("primary",),
        metadata={"block_expr_obj": x1},
    )
    state = BasisState(
        blocks=(parent_block, child_block, companion_block),
        fit_loss=1.0,
        probe_loss=1.0,
        complexity=sum(block.complexity() for block in (parent_block, child_block, companion_block)),
        compiled_expr=("add", sqrt_shift, x1),
    )
    x = torch.tensor(
        [
            [0.2, 0.7],
            [0.4, 0.5],
            [0.8, 0.1],
            [1.2, 0.3],
            [1.7, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (torch.sqrt(1.0 + x[:, 0] ** 2) + x[:, 1]).unsqueeze(-1)

    refit = fit_basis_state_head(
        state,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        route_name="test_basis_head_dependency_keep",
    )

    assert refit is not None
    assert float(refit.fit_loss) < 1.0e-20
    assert [getattr(block, "block_id", "") for block in refit.blocks] == [
        "basis:block:x0",
        "basis:block:sqrt_shift",
        "basis:block:x1",
    ]
    basis_head = dict(refit.fit_bundle or {}).get("basis_head", {})
    assert "sqrt((1+sqr(x0)))" in list(basis_head.get("block_exprs", []))


def test_fit_basis_state_head_uses_periodic_bundle_terms_not_only_primary_expr():
    hole = ("var", 0)
    anchor = ("var", 1)
    expr = ("add", ("cos", hole), anchor)
    state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos_add:x1",
        expr=expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "cos_add"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_harmonic_head",
        local_mapping_coeffs=[1.0, 0.0, 1.0, 0.0],
        direct_metadata={
            "feature_kind": "cos",
            "hole_node": hole,
            "feature_node": ("cos", hole),
            "harmonic_feature_nodes": [("cos", hole), ("sin", hole)],
            "companion_nodes": [anchor],
        },
    )

    x = torch.tensor(
        [
            [0.1, 0.7],
            [0.4, 0.5],
            [0.8, 0.1],
            [1.2, 0.3],
            [1.7, 0.9],
            [2.0, 1.1],
        ],
        dtype=torch.float64,
    )
    y = (
        2.0 * torch.cos(x[:, 0])
        - 0.75 * torch.sin(x[:, 0])
        + 1.5 * x[:, 1]
        + 0.25
    ).unsqueeze(-1)

    refit = fit_basis_state_head(
        state,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        route_name="test_bundle_basis_head",
    )

    assert refit is not None
    assert float(refit.fit_loss) < 1.0e-20
    basis_head = dict(refit.fit_bundle or {}).get("basis_head", {})
    term_exprs = list(basis_head.get("term_exprs", []))
    term_roles = list(basis_head.get("term_roles", []))
    assert "cos(x0)" in term_exprs
    assert "sin(x0)" in term_exprs
    assert "x1" in term_exprs
    assert "harmonic_term" in term_roles
    assert "companion_term" in term_roles
    compiled = refit.to_dict()["compiled_expr"]
    assert "sin(x0)" in compiled
    assert "x1" in compiled


def test_run_closure_search_pass_builds_residual_context_from_basis_head_not_stale_expr():
    state_x0 = _make_var_block_state(0)
    state_x1 = _make_var_block_state(1)
    basis_state = BasisState(
        blocks=(state_x0.blocks[0], state_x1.blocks[0]),
        fit_loss=1.0,
        probe_loss=1.0,
        complexity=state_x0.blocks[0].complexity() + state_x1.blocks[0].complexity(),
        compiled_expr=("var", 0),
    )
    x = torch.tensor(
        [
            [0.2, 0.7],
            [0.4, 0.5],
            [0.8, 0.1],
            [1.2, 0.3],
            [1.7, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.0).unsqueeze(-1)
    seen = {}

    def _run_closure_search_pass_impl(**kwargs):
        seen["proposal_context"] = kwargs.get("proposal_context")
        return {"stats": {}, "candidate_rows": []}

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "x_fit": x,
            "y_fit": y,
            "x_probe": x,
            "y_probe": y,
            "boost_pool_nodes": [("var", 0), ("var", 1)],
            "proposal_context": ProposalContext(
                basis_state=basis_state,
                basis_state_beam=(basis_state,),
                total_budget=4,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": lambda *args, **kwargs: None,
        }
    )

    run_closure_search_pass(**kwargs)

    proposal_context = seen["proposal_context"]
    assert proposal_context is not None
    assert proposal_context.basis_state is not None
    assert float(proposal_context.basis_state.probe_loss) < 1.0e-20
    assert float(proposal_context.residual_witness["residual_probe_rms"]) < 1.0e-10
    assert "basis_linear_head" == proposal_context.basis_state.fit_bundle["basis_head"]["kind"]
