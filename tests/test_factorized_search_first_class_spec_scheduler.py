# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

import nestynet_sr.sr_search.factorized_search.engine.search as engine_search
import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod
import nestynet_sr.sr_search.factorized_search.hole_search as hole_search_mod
from nestynet_sr.sr_search.factorized_search.expr_ast import node_str
from nestynet_sr.sr_search.factorized_search.hole_search import HoleOpportunity


def test_first_class_scheduler_executes_spec_route_without_macro_action_count(monkeypatch):
    init_expr = ("add", ("var", 0), ("const", 0.0))
    spec_expr = ("sin", ("var", 0))
    captured_kwargs = {}

    def _fake_rand_node(rng, max_depth, nvars):
        return init_expr

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        score_map = {
            init_expr: 1.0,
            spec_expr: 0.25,
        }
        mse = float(score_map.get(node, 1.5))
        key = ("expr", node_str(node))
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return mse, key, z, mapping, node
        return mse, key, z, mapping

    def _fake_route_select(self, rng, available_routes, route_scores=None):
        if "opportunity_expand" in list(available_routes or []):
            return "opportunity_expand", "forced_test"
        return "expression_expand", "forced_test"

    def _fake_mine_frontier_from_archive(frontier, archive_records, **kwargs):
        rec = list(archive_records or [])[0]
        snapshot_id = kwargs["snapshot_parent_fn"](
            residual_basin_key=str(getattr(rec, "residual_basin_key", "seed")),
            elite_id=str(getattr(rec, "best_elite_id", "") or ""),
            expr=rec.best_expr,
            mapping=rec.mapping,
            eff_mse=float(rec.best_mse),
            raw_mse=float(getattr(rec, "best_raw_mse", rec.best_mse)),
            current_iter=int(kwargs.get("current_iter", 0)),
            expr_str=node_str(rec.best_expr),
        )
        frontier.enqueue_spec_state(
            HoleOpportunity(
                parent_key=str(getattr(rec, "residual_basin_key", "seed")),
                parent_expr_str=node_str(rec.best_expr),
                path=(1,),
                target_mode="identity",
                beam_rank=0,
                parent_elite_id=str(getattr(rec, "best_elite_id", "") or ""),
                parent_snapshot_id=str(snapshot_id),
                source="archive_mine",
                spec_kind="path_hole",
                path_gain=0.8,
                confidence=0.9,
                valid_frac=0.9,
                target_mapping_kind="affine",
                parent_eff_mse_at_emit=float(rec.best_mse),
            ),
            current_iter=int(kwargs.get("current_iter", 0)),
        )
        return 1

    def _fake_run_hole_search_action(opportunity, **kwargs):
        captured_kwargs.update(kwargs)
        return spec_expr, {
            "status": "ok",
            "hole_search_wall_seconds": 0.0,
            "hole_search_best_eff_mse": 0.3,
            "hole_search_followup_spec_states": [],
            "hole_search_followup_spec_state_count": 0,
        }

    monkeypatch.setattr(explorer_mod, "rand_node", _fake_rand_node)
    monkeypatch.setattr(engine_search._RouteScheduler, "select", _fake_route_select)
    monkeypatch.setattr(hole_search_mod, "mine_frontier_from_archive", _fake_mine_frontier_from_archive)
    monkeypatch.setattr(hole_search_mod, "run_hole_search_action", _fake_run_hole_search_action)

    def _target_fn(x):
        return x[:, :1]

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        n_iter=2,
        max_depth=3,
        poly_degree=2,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        no_residual=True,
        no_crossover=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
        inverse_steering_enable=True,
        inverse_spec_enable=True,
        hole_search_enable=True,
        hole_search_first_class_scheduler_enable=True,
        hole_search_route_scheduler_enable=True,
        hole_search_solver_market_enable=True,
        hole_search_solver_market_preview_topk=5,
        hole_search_solver_market_exact_topk=3,
        hole_search_solver_market_proposal_objects_enable=True,
        inverse_spec_recursive_sr_enable=True,
        inverse_spec_recursive_sr_preview_topk=6,
        inverse_spec_recursive_sr_exact_budget=2,
        inverse_spec_constant_lift_route_enable=True,
        inverse_spec_constant_lift_route_topk=3,
        inverse_spec_coordinate_lift_enable=True,
        inverse_spec_coordinate_lift_topk=5,
        inverse_spec_coordinate_lift_mode="single_index",
        inverse_spec_tangent_edit_enable=True,
        inverse_spec_tangent_edit_topk=7,
        inverse_spec_soft_edit_enable=True,
        inverse_spec_soft_edit_steps=48,
        inverse_spec_soft_edit_l1=2.0e-3,
        inverse_spec_witness_jets_enable=True,
        inverse_spec_witness_d2_enable=True,
        inverse_spec_witness_max_rows=40,
        inverse_spec_active_var_screen_enable=True,
        inverse_spec_active_var_grad_tol=2.0e-3,
        inverse_spec_active_var_max_count=3,
        inverse_spec_directional_market_enable=True,
        inverse_experiment_log_enable=True,
        _score_expr_fn=_fake_score_expr,
    )

    hs = getattr(arch, "hole_search_stats", {})
    rs = getattr(arch, "route_scheduler_stats", {})
    ad = getattr(arch, "action_distribution", {})
    counts = dict(ad.get("counts", {}) or {})
    log_rows = list(getattr(arch, "inverse_experiment_log", []) or [])
    route_rows = [
        row
        for row in log_rows
        if str(row.get("route_scheduler_selected_route", "") or "") == "opportunity_expand"
    ]

    assert int(hs.get("first_class_scheduler_selected", 0)) >= 1
    assert int(hs.get("run_hole_search_action_called", 0)) >= 1
    assert int(rs.get("selected_opportunity_expand", 0)) >= 1
    assert str(rs.get("mode", "")) == "first_class_agenda"
    assert int(rs.get("diagnostic_count", 0)) >= 1
    assert int(counts.get("hole_search", 0)) == 0
    assert captured_kwargs["solver_market_enable"] is True
    assert captured_kwargs["solver_market_preview_topk"] == 5
    assert captured_kwargs["solver_market_exact_topk"] == 3
    assert captured_kwargs["solver_market_proposal_objects_enable"] is True
    assert captured_kwargs["inverse_spec_recursive_sr_enable"] is True
    assert captured_kwargs["inverse_spec_recursive_sr_preview_topk"] == 6
    assert captured_kwargs["inverse_spec_recursive_sr_exact_budget"] == 2
    assert captured_kwargs["inverse_spec_constant_lift_route_enable"] is True
    assert captured_kwargs["inverse_spec_constant_lift_route_topk"] == 3
    assert captured_kwargs["inverse_spec_coordinate_lift_enable"] is True
    assert captured_kwargs["inverse_spec_coordinate_lift_topk"] == 5
    assert captured_kwargs["inverse_spec_coordinate_lift_mode"] == "single_index"
    assert captured_kwargs["inverse_spec_tangent_edit_enable"] is True
    assert captured_kwargs["inverse_spec_tangent_edit_topk"] == 7
    assert captured_kwargs["inverse_spec_soft_edit_enable"] is True
    assert captured_kwargs["inverse_spec_soft_edit_steps"] == 48
    assert captured_kwargs["inverse_spec_soft_edit_l1"] == 2.0e-3
    assert captured_kwargs["inverse_spec_witness_jets_enable"] is True
    assert captured_kwargs["inverse_spec_witness_d2_enable"] is True
    assert captured_kwargs["inverse_spec_witness_max_rows"] == 40
    assert captured_kwargs["inverse_spec_active_var_screen_enable"] is True
    assert captured_kwargs["inverse_spec_active_var_grad_tol"] == 2.0e-3
    assert captured_kwargs["inverse_spec_active_var_max_count"] == 3
    assert captured_kwargs["inverse_spec_directional_market_enable"] is True
    assert arch.best(1)[0].best_expr == spec_expr
    assert route_rows
    assert route_rows[0]["route_scheduler_best_available_route"] == "opportunity_expand"
    assert route_rows[0]["route_scheduler_selected_best_preview_route"] is True
    assert route_rows[0]["route_scheduler_realized_adjusted_reward"] is not None
