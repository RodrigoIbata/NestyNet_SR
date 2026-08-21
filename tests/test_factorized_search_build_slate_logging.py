# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch

from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.policy import build_slate as build_slate_mod


def test_explorer_build_slate_wrappers_exist():
    assert callable(build_slate_mod.collect_controller_build_slate)
    assert callable(build_slate_mod.controller_selected_action_path)
    assert callable(explorer._collect_controller_build_slate)
    assert callable(explorer._controller_selected_action_path)


def test_collect_controller_build_slate_logs_same_parent_alternatives(monkeypatch):
    parent_expr = ("add", ("var", 0), ("var", 1))
    parent_rec = SimpleNamespace(
        best_expr=parent_expr,
        mapping={"kind": "poly"},
        best_mse=1.0,
    )

    monkeypatch.setattr(
        explorer,
        "_controller_selected_action_path",
        lambda *args, **kwargs: ((1,), "critic_path_head"),
    )

    def _fake_apply_action(node, action, rng, max_depth, nvars, var_dims=None, reach=None, path=None):
        if int(action) == int(explorer.A_WRAP_UNARY):
            return ("sin", ("var", 0))
        return ("add", ("var", 0), ("var", 1))

    monkeypatch.setattr(explorer, "apply_action", _fake_apply_action)
    monkeypatch.setattr(
        explorer,
        "apply_residual_action",
        lambda *args, **kwargs: ("mul", ("var", 0), ("const", 1.0)),
    )

    def _fake_score_expr(expr, *args, **kwargs):
        label = explorer.node_str(expr)
        if isinstance(expr, tuple) and expr and expr[0] == "mul":
            mse = 0.30
        elif isinstance(expr, tuple) and expr and expr[0] == "sin":
            mse = 0.40
        else:
            mse = 0.20
        return mse, f"k_{label}", torch.zeros((1,), dtype=torch.float32), {"kind": "poly"}, expr

    monkeypatch.setattr(explorer, "score_expr", _fake_score_expr)

    out = explorer._collect_controller_build_slate(
        parent_key="parentA",
        parent_rec=parent_rec,
        n_evaluated=17,
        seed_search=3,
        active_actions=(explorer.A_REPLACE, explorer.A_WRAP_UNARY, explorer.A_RESIDUAL),
        action_names=("replace", "wrap_un", "residual"),
        max_actions=3,
        controller_policy_guidance=None,
        macro_decision=None,
        macro_state=None,
        inverse_gate_diag=None,
        x_fit=torch.zeros((4, 2), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 2), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        proj=torch.zeros((4, 4), dtype=torch.float64),
        fp_mode="bits",
        q_scale=2.0,
        q_clip=6.0,
        poly_degree=2,
        refine_enable=False,
        refine_cfg={},
        refine_state={},
        best_raw_mse_struct=float("inf"),
        best_raw_mse=1.0,
        early_stop_mse=1.0e-10,
        complexity_penalty=0.0,
        boost_enable=False,
        boost_pool_nodes=[],
        boost_pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        boost_pool_norms_fit=torch.zeros((0,), dtype=torch.float64),
        boost_pool_phi=torch.zeros((4, 0), dtype=torch.float64),
        boost_pool_norms=torch.zeros((0,), dtype=torch.float64),
        boost_pool_dims=None,
        boost_selection_split="fit",
        boost_ridge=None,
        boost_include_parent=True,
        boost_from_scratch_prob=0.0,
        boost_prune_rel=1.0e-10,
        boost_max_terms=4,
        boost_topk_try=4,
        boost_min_rel_improve=1.0e-3,
        max_depth=4,
        nvars=2,
        var_dims=None,
        y_dims=None,
        reach=None,
    )

    assert out["controller_build_slate_id"]
    assert out["controller_build_slate_count"] == 3
    assert out["controller_build_slate_exact_observed_count"] == 3
    assert out["build_opportunity_slate_id"]
    assert out["build_opportunity_slate_count"] == 3
    rows = out["controller_build_slate"]
    opp_rows = out["build_opportunity_slate"]
    assert [row["action"] for row in rows] == ["replace", "wrap_un", "residual"]
    assert [row["action"] for row in opp_rows] == ["replace", "wrap_un", "residual"]
    assert all(bool(row["exact_child_score_observed"]) for row in rows)
    assert all(row["tuple_provenance"] == "build_slate" for row in rows)
    assert all("candidate_child_size" in row for row in rows)
    assert all("candidate_child_depth" in row for row in rows)
    assert all("candidate_root_op" in row for row in rows)
    assert all("path_length" in row for row in rows)
    assert all(row["route_source"] == "build" for row in opp_rows)
    assert all(row["opportunity_type"] == "build_action" for row in opp_rows)
    assert all(row["decision_id"] == out["build_opportunity_slate_id"] for row in opp_rows)
    assert all(row["budget_exact_spent"] == 1 for row in opp_rows)
    assert all(row["candidate_count_observed"] == 1 for row in opp_rows)
    assert rows[0]["child_eff_mse"] <= rows[1]["child_eff_mse"]


def test_collect_controller_build_slate_preview_only_skips_exact_scoring(monkeypatch):
    parent_expr = ("add", ("var", 0), ("var", 1))
    parent_rec = SimpleNamespace(
        best_expr=parent_expr,
        mapping={"kind": "poly"},
        best_mse=1.0,
    )
    score_calls = {"count": 0}

    monkeypatch.setattr(
        explorer,
        "_controller_selected_action_path",
        lambda *args, **kwargs: ((1,), "critic_path_head"),
    )
    monkeypatch.setattr(
        explorer,
        "apply_action",
        lambda *args, **kwargs: ("sin", ("var", 0)),
    )

    def _fake_score_expr(expr, *args, **kwargs):
        score_calls["count"] += 1
        return 0.4, "k_preview", torch.zeros((1,), dtype=torch.float32), {"kind": "poly"}, expr

    monkeypatch.setattr(explorer, "score_expr", _fake_score_expr)

    out = explorer._collect_controller_build_slate(
        parent_key="parentB",
        parent_rec=parent_rec,
        n_evaluated=4,
        seed_search=1,
        active_actions=(explorer.A_WRAP_UNARY,),
        action_names=("wrap_un",),
        max_actions=1,
        controller_policy_guidance=None,
        macro_decision=None,
        macro_state=None,
        inverse_gate_diag=None,
        x_fit=torch.zeros((4, 2), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 2), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        proj=torch.zeros((4, 4), dtype=torch.float64),
        fp_mode="bits",
        q_scale=2.0,
        q_clip=6.0,
        poly_degree=2,
        refine_enable=False,
        refine_cfg={},
        refine_state={},
        best_raw_mse_struct=float("inf"),
        best_raw_mse=1.0,
        early_stop_mse=1.0e-10,
        complexity_penalty=0.0,
        boost_enable=False,
        boost_pool_nodes=[],
        boost_pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        boost_pool_norms_fit=torch.zeros((0,), dtype=torch.float64),
        boost_pool_phi=torch.zeros((4, 0), dtype=torch.float64),
        boost_pool_norms=torch.zeros((0,), dtype=torch.float64),
        boost_pool_dims=None,
        boost_selection_split="fit",
        boost_ridge=None,
        boost_include_parent=True,
        boost_from_scratch_prob=0.0,
        boost_prune_rel=1.0e-10,
        boost_max_terms=4,
        boost_topk_try=4,
        boost_min_rel_improve=1.0e-3,
        max_depth=4,
        nvars=2,
        var_dims=None,
        y_dims=None,
        reach=None,
        preview_only=True,
    )

    assert score_calls["count"] == 0
    assert out["controller_build_slate_preview_only"] is True
    assert out["controller_build_slate_exact_observed_count"] == 0
    assert out["controller_build_slate"][0]["status"] == "preview_only"
    assert out["controller_build_slate"][0]["exact_child_score_observed"] is False
    assert out["build_opportunity_slate"][0]["budget_exact_spent"] == 0
    assert out["build_opportunity_slate"][0]["build_preview_only"] is True
