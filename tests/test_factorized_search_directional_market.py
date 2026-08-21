# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod
from nestynet_sr.sr_search.factorized_search.hole_search import (
    HoleOpportunity,
    _build_spec_preview_route_calls,
    _normalize_followup_spec_rows,
    run_hole_search_action,
)
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


def _make_local_problem_opportunity(*, direction: str) -> HoleOpportunity:
    x_fit = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-1.5, 1.5, 11, dtype=torch.float64).unsqueeze(-1)
    spec = SubproblemSpec(
        problem_id=f"directional_{direction}",
        problem_kind="local_problem",
        parent_expr=("add", ("const", 1.0), ("var", 0)),
        path=(1,),
        direction=direction,
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=({"wrap_kind": "unary", "op": "sin", "slot": 0, "anchor_node": None},),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(0,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=x_fit.clone(),
            x_probe=x_probe,
            t_probe=x_probe.clone(),
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("inner",)},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    return HoleOpportunity(
        parent_key=f"parent_{direction}",
        parent_expr_str=f"expr_{direction}",
        path=(1,),
        target_mode="identity",
        beam_rank=0,
        spec_kind="local_problem",
        direction=direction,
        branch_id=f"branch_{direction}",
        path_gain=0.5,
        confidence=0.8,
        valid_frac=0.9,
        target_mapping_kind="affine",
        spec_payload=wrap_subproblem_spec_payload(spec),
    )


def _make_coordinate_problem_opportunity(*, direction: str) -> HoleOpportunity:
    x0_fit = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    x1_fit = torch.linspace(1.0, -1.0, 17, dtype=torch.float64)
    x_fit = torch.stack([x0_fit, x1_fit + 0.5 * x0_fit], dim=1)
    x0_probe = torch.linspace(-1.25, 1.25, 19, dtype=torch.float64)
    x1_probe = torch.linspace(1.25, -1.25, 19, dtype=torch.float64)
    x_probe = torch.stack([x0_probe, x1_probe + 0.5 * x0_probe], dim=1)
    z_fit = x_fit[:, 0:1] + x_fit[:, 1:2]
    z_probe = x_probe[:, 0:1] + x_probe[:, 1:2]
    grad_fit = torch.ones_like(x_fit)
    grad_probe = torch.ones_like(x_probe)
    spec = SubproblemSpec(
        problem_id=f"directional_coord_{direction}",
        problem_kind="local_problem",
        parent_expr=("add", ("const", 1.0), ("var", 0)),
        path=(1,),
        direction=direction,
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(0, 1),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=z_fit,
            x_probe=x_probe,
            t_probe=z_probe,
            grad_fit=grad_fit,
            grad_probe=grad_probe,
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("inner_coord",)},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    return HoleOpportunity(
        parent_key=f"parent_coord_{direction}",
        parent_expr_str=f"expr_coord_{direction}",
        path=(1,),
        target_mode="identity",
        beam_rank=0,
        spec_kind="local_problem",
        direction=direction,
        branch_id=f"branch_coord_{direction}",
        path_gain=0.5,
        confidence=0.8,
        valid_frac=0.9,
        target_mapping_kind="affine",
        spec_payload=wrap_subproblem_spec_payload(spec),
    )


def _make_soft_edit_problem_opportunity(*, direction: str, wrappers_left: int = 0) -> HoleOpportunity:
    x_fit = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-1.5, 1.5, 25, dtype=torch.float64).unsqueeze(-1)
    spec = SubproblemSpec(
        problem_id=f"directional_soft_{direction}",
        problem_kind="local_problem",
        parent_expr=("var", 0),
        path=(),
        direction=direction,
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=int(wrappers_left),
        recursion_level=1,
        active_vars=(0,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=torch.sin(x_fit),
            x_probe=x_probe,
            t_probe=torch.sin(x_probe),
            grad_fit=torch.cos(x_fit),
            grad_probe=torch.cos(x_probe),
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("soft",)},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    return HoleOpportunity(
        parent_key=f"parent_soft_{direction}",
        parent_expr_str=f"expr_soft_{direction}",
        path=(),
        target_mode="identity",
        beam_rank=0,
        spec_kind="local_problem",
        direction=direction,
        branch_id=f"branch_soft_{direction}",
        path_gain=0.5,
        confidence=0.8,
        valid_frac=0.9,
        target_mapping_kind="affine",
        spec_payload=wrap_subproblem_spec_payload(spec),
    )


def _route_names(route_calls):
    return [call.route_name for call in route_calls]


def test_build_spec_preview_route_calls_reorders_inside_out_directional_market():
    opp = _make_local_problem_opportunity(direction="inside_out")
    x = torch.randn(8, 1, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    route_calls, status = _build_spec_preview_route_calls(
        opp,
        parent_node=("add", ("const", 1.0), ("var", 0)),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        enum_max_depth=2,
        enum_max_trees=64,
        preview_topk=4,
        max_subtree_depth=3,
        complexity_penalty=0.0,
        family_battery_enable=False,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=4,
        recursive_branch_topk=2,
        recursive_child_topk=1,
        recursive_sr_enable=True,
        recursive_sr_preview_topk=4,
        recursive_sr_exact_budget=2,
        coordinate_lift_enable=False,
        coordinate_lift_topk=4,
        coordinate_lift_mode="both",
        tangent_edit_enable=True,
        tangent_edit_topk=8,
        soft_edit_enable=True,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
        witness_jets_enable=False,
        witness_d2_enable=False,
        witness_max_rows=64,
        active_var_screen_enable=False,
        active_var_grad_tol=1.0e-3,
        active_var_max_count=4,
        directional_market_enable=True,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        beam_cfg={},
        slate_prefix="test",
    )

    assert status == "ok"
    assert _route_names(route_calls) == [
        "recursive_local_sr",
        "tangent_edit",
        "soft_edit_search",
        "inverse_spec_followup",
    ]


def test_build_spec_preview_route_calls_reorders_outside_in_directional_market():
    opp = _make_local_problem_opportunity(direction="outside_in")
    x = torch.randn(8, 1, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    route_calls, status = _build_spec_preview_route_calls(
        opp,
        parent_node=("add", ("const", 1.0), ("var", 0)),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        enum_max_depth=2,
        enum_max_trees=64,
        preview_topk=4,
        max_subtree_depth=3,
        complexity_penalty=0.0,
        family_battery_enable=False,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=4,
        recursive_branch_topk=2,
        recursive_child_topk=1,
        recursive_sr_enable=True,
        recursive_sr_preview_topk=4,
        recursive_sr_exact_budget=2,
        coordinate_lift_enable=False,
        coordinate_lift_topk=4,
        coordinate_lift_mode="both",
        tangent_edit_enable=True,
        tangent_edit_topk=8,
        soft_edit_enable=True,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
        witness_jets_enable=False,
        witness_d2_enable=False,
        witness_max_rows=64,
        active_var_screen_enable=False,
        active_var_grad_tol=1.0e-3,
        active_var_max_count=4,
        directional_market_enable=True,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        beam_cfg={},
        slate_prefix="test",
    )

    assert status == "ok"
    assert _route_names(route_calls) == [
        "inverse_spec_followup",
        "recursive_local_sr",
        "tangent_edit",
        "soft_edit_search",
    ]


def test_build_spec_preview_route_calls_places_coordinate_lift_first_for_inside_out():
    opp = _make_local_problem_opportunity(direction="inside_out")
    x = torch.randn(8, 1, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    route_calls, status = _build_spec_preview_route_calls(
        opp,
        parent_node=("add", ("const", 1.0), ("var", 0)),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        enum_max_depth=2,
        enum_max_trees=64,
        preview_topk=4,
        max_subtree_depth=3,
        complexity_penalty=0.0,
        family_battery_enable=False,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=4,
        recursive_branch_topk=2,
        recursive_child_topk=1,
        recursive_sr_enable=True,
        recursive_sr_preview_topk=4,
        recursive_sr_exact_budget=2,
        coordinate_lift_enable=True,
        coordinate_lift_topk=3,
        coordinate_lift_mode="both",
        tangent_edit_enable=True,
        tangent_edit_topk=8,
        soft_edit_enable=True,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
        witness_jets_enable=False,
        witness_d2_enable=False,
        witness_max_rows=64,
        active_var_screen_enable=False,
        active_var_grad_tol=1.0e-3,
        active_var_max_count=4,
        directional_market_enable=True,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        beam_cfg={},
        slate_prefix="test",
    )

    assert status == "ok"
    assert _route_names(route_calls) == [
        "coordinate_lift",
        "recursive_local_sr",
        "tangent_edit",
        "soft_edit_search",
        "inverse_spec_followup",
    ]


def test_build_spec_preview_route_calls_uses_coordinate_evidence_to_reorder_lifts():
    opp = _make_coordinate_problem_opportunity(direction="inside_out")
    x = torch.randn(8, 2, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    route_calls, status = _build_spec_preview_route_calls(
        opp,
        parent_node=("add", ("const", 1.0), ("var", 0)),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=2,
        poly_degree=2,
        var_dims=None,
        enum_max_depth=2,
        enum_max_trees=64,
        preview_topk=4,
        max_subtree_depth=3,
        complexity_penalty=0.0,
        family_battery_enable=False,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=4,
        recursive_branch_topk=2,
        recursive_child_topk=1,
        recursive_sr_enable=True,
        recursive_sr_preview_topk=4,
        recursive_sr_exact_budget=2,
        constant_lift_route_enable=True,
        constant_lift_route_topk=2,
        coordinate_lift_enable=True,
        coordinate_lift_topk=3,
        coordinate_lift_mode="both",
        tangent_edit_enable=True,
        tangent_edit_topk=8,
        soft_edit_enable=True,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
        witness_jets_enable=False,
        witness_d2_enable=False,
        witness_max_rows=64,
        active_var_screen_enable=False,
        active_var_grad_tol=1.0e-3,
        active_var_max_count=4,
        directional_market_enable=True,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        beam_cfg={},
        slate_prefix="test",
    )

    assert status == "ok"
    assert _route_names(route_calls) == [
        "coordinate_lift",
        "constant_lift_route",
        "recursive_local_sr",
        "tangent_edit",
        "soft_edit_search",
        "inverse_spec_followup",
    ]


def test_build_spec_preview_route_calls_keeps_coordinate_lift_for_unknown_direction():
    opp = _make_local_problem_opportunity(direction="")
    x = torch.randn(8, 1, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    route_calls, status = _build_spec_preview_route_calls(
        opp,
        parent_node=("add", ("const", 1.0), ("var", 0)),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        enum_max_depth=2,
        enum_max_trees=64,
        preview_topk=4,
        max_subtree_depth=3,
        complexity_penalty=0.0,
        family_battery_enable=False,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=4,
        recursive_branch_topk=2,
        recursive_child_topk=1,
        recursive_sr_enable=True,
        recursive_sr_preview_topk=4,
        recursive_sr_exact_budget=2,
        coordinate_lift_enable=True,
        coordinate_lift_topk=3,
        coordinate_lift_mode="both",
        tangent_edit_enable=True,
        tangent_edit_topk=8,
        soft_edit_enable=True,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
        witness_jets_enable=False,
        witness_d2_enable=False,
        witness_max_rows=64,
        active_var_screen_enable=False,
        active_var_grad_tol=1.0e-3,
        active_var_max_count=4,
        directional_market_enable=True,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        beam_cfg={},
        slate_prefix="test",
    )

    assert status == "ok"
    assert _route_names(route_calls) == [
        "inverse_spec_followup",
        "coordinate_lift",
        "recursive_local_sr",
        "tangent_edit",
        "soft_edit_search",
    ]


def test_normalize_followup_spec_rows_preserves_subproblem_direction():
    opp = _make_local_problem_opportunity(direction="inside_out")
    rows = _normalize_followup_spec_rows(
        [
            {
                "spec_kind": "local_problem",
                "spec_payload": dict(opp.spec_payload or {}),
            }
        ],
        opportunity=opp,
    )

    assert rows[0]["direction"] == "inside_out"


def test_run_hole_search_action_reports_directional_route_order(monkeypatch):
    opp = _make_local_problem_opportunity(direction="inside_out")
    x = torch.randn(8, 1, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    def fake_market(route_calls, *, preview_topk, exact_topk):
        route_names = [call.route_name for call in route_calls]
        return {
            "rows": [],
            "solver_meta": {
                "status": "no_market_candidates",
                "solver_market_routes": [
                    {
                        "route_rank": idx,
                        "route_name": name,
                        "method_name": name,
                        "subroute": name,
                        "status": "no_rows",
                        "row_count": 0,
                        "child_spec_state_count": 0,
                        "preview_best_probe_mse": None,
                        "error": "",
                    }
                    for idx, name in enumerate(route_names)
                ],
                "solver_market_route_count": len(route_names),
                "solver_market_candidate_count_raw": 0,
                "solver_market_candidate_count_unique": 0,
                "solver_market_selected_route": "",
                "solver_market_selected_method_name": "",
                "solver_market_selected_subroute": "",
                "preview_count": 0,
                "child_spec_states": [],
                "child_spec_state_count": 0,
            },
        }

    monkeypatch.setattr(hs_mod, "run_preview_solver_market", fake_market)

    expr, meta = run_hole_search_action(
        opp,
        parent_node=("add", ("const", 1.0), ("var", 0)),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        rng=None,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        solver_market_enable=True,
        solver_market_preview_topk=4,
        solver_market_exact_topk=2,
        inverse_spec_recursive_sr_enable=True,
        inverse_spec_tangent_edit_enable=True,
        inverse_spec_soft_edit_enable=True,
        inverse_spec_directional_market_enable=True,
        return_meta=True,
    )

    assert expr is None
    assert meta["hole_search_direction"] == "inside_out"
    assert meta["hole_search_solver_market_directional_order"] == [
        "recursive_local_sr",
        "tangent_edit",
        "soft_edit_search",
        "inverse_spec_followup",
    ]


def test_run_hole_search_action_can_execute_soft_edit_route_under_no_grad(monkeypatch):
    opp = _make_soft_edit_problem_opportunity(direction="inside_out", wrappers_left=1)
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
    y = torch.sin(x).squeeze(-1)

    def fake_inverse_followup(**kwargs):
        return {"rows": [], "solver_meta": {"status": "disabled_for_test"}}

    monkeypatch.setattr(hs_mod, "solve_local_problem_spec_preview_rows", fake_inverse_followup)

    expr, meta = run_hole_search_action(
        opp,
        parent_node=("var", 0),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        rng=None,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        solver_market_enable=True,
        solver_market_preview_topk=4,
        solver_market_exact_topk=2,
        inverse_spec_coordinate_lift_enable=False,
        inverse_spec_constant_lift_route_enable=False,
        inverse_spec_tangent_edit_enable=False,
        inverse_spec_soft_edit_enable=True,
        inverse_spec_soft_edit_steps=24,
        inverse_spec_witness_loss_enable=True,
        inverse_spec_witness_grad_weight=0.5,
        inverse_spec_witness_diag_weight=0.25,
        inverse_spec_directional_market_enable=True,
        return_meta=True,
    )

    assert isinstance(expr, tuple)
    assert meta["status"] == "ok"
    assert meta["hole_search_solver_market_selected_route"] == "soft_edit_search"
    route_meta = {
        str(row.get("route_name", "") or ""): dict(row)
        for row in list(meta.get("hole_search_solver_market_routes", []) or [])
    }
    assert route_meta["soft_edit_search"]["status"] == "ok"
    assert route_meta["soft_edit_search"]["error"] == ""
    assert route_meta["soft_edit_search"]["row_count"] > 0


def test_run_hole_search_action_path_hole_forwards_regime_metadata(monkeypatch):
    opp = HoleOpportunity(
        parent_key="parent_path",
        parent_expr_str="expr_path",
        path=(1,),
        target_mode="identity",
        beam_rank=0,
        spec_kind="path_hole",
        direction="inside_out",
        path_gain=0.5,
        confidence=0.8,
        valid_frac=0.9,
        target_mapping_kind="affine",
    )
    x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1)
    beam_state = {
        "path": (1,),
        "sub": ("var", 0),
        "target_mode": "identity",
        "target_mapping_kind": "affine",
        "path_gain": 0.5,
        "confidence": 0.8,
        "valid_frac": 0.9,
        "target_dim": None,
        "xf": x,
        "tf": x.clone(),
        "xp": x,
        "tp": x.clone(),
        "wf": None,
        "wp": None,
    }
    expected_regime_metadata = {
        "dataset_ids": ["d0", "d1"],
        "local_constants_by_experiment": {
            "d0": {"local_leaf": 1.0},
            "d1": {"local_leaf": 2.0},
        },
    }

    def fake_path_beam_state(*args, **kwargs):
        return dict(beam_state), "ok"

    captured = {}

    def fake_inverse_preview(**kwargs):
        captured["regime_metadata"] = kwargs.get("regime_metadata", None)
        return {"rows": [], "solver_meta": {}}

    monkeypatch.setattr(hs_mod, "_build_path_hole_beam_state", fake_path_beam_state)
    monkeypatch.setattr(hs_mod, "solve_inverse_spec_preview_rows", fake_inverse_preview)

    expr, meta = run_hole_search_action(
        opp,
        parent_node=("add", ("const", 1.0), ("var", 0)),
        parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        x_fit=x,
        y_fit=x.squeeze(-1),
        x_probe=x,
        y_probe=x.squeeze(-1),
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        rng=None,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        inverse_spec_regime_metadata=expected_regime_metadata,
        return_meta=True,
    )

    assert expr is None
    assert meta["status"] == "no_preview_candidates"
    assert captured["regime_metadata"] == expected_regime_metadata
