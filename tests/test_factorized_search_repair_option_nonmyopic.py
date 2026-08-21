# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_search.factorized_search.expr_ast import node_str
from nestynet_sr.sr_search.factorized_search.repair_action import run_repair_option_action


def _score_expr_factory(raw_by_expr):
    def _score_expr(
        expr,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        fp_mode,
        q_scale,
        q_clip,
        poly_degree,
        *,
        refine_enable=False,
        refine_cfg=None,
        return_expr=False,
    ):
        raw = float(raw_by_expr[node_str(expr)])
        return raw, None, None, {"kind": "identity"}, expr

    return _score_expr


def test_repair_option_low_gain_blocks_without_nonmyopic_setup():
    parent = ("add", ("var", 0), ("var", 1))
    setup_expr = ("add", ("var", 0), ("const", 1.0))
    raw_by_expr = {
        node_str(parent): 1.0,
        node_str(setup_expr): 0.99,
    }

    out = run_repair_option_action(
        parent,
        {"kind": "identity"},
        None,
        None,
        None,
        None,
        [],
        None,
        None,
        None,
        None,
        5,
        2,
        2,
        max_steps=2,
        min_step_rel_improve=0.05,
        first_step_expr=setup_expr,
        first_step_meta={
            "status": "ok",
            "selected_path": [1],
            "estimated_child_eff_mse": 0.99,
            "tuple_value_estimate": 0.40,
            "tuple_regret_estimate": 0.05,
            "tuple_allocation_estimate": 0.35,
        },
        current_eff_mse=1.0,
        return_meta=True,
        score_expr_fn=_score_expr_factory(raw_by_expr),
        inverse_action_fn=lambda *args, **kwargs: (None, None),
    )

    expr, meta = out
    assert expr is None
    assert meta["status"] == "repair_option_low_step_gain"
    assert meta["repair_option_steps_attempted"] == 1
    assert meta["repair_option_steps_accepted"] == 0
    assert meta["repair_option_setup_steps_used"] == 0
    assert meta["repair_option_step_nonmyopic_continue"] == [False]


def test_repair_option_allows_one_setup_step_with_good_continuation_signal():
    parent = ("add", ("var", 0), ("var", 1))
    setup_expr = ("add", ("var", 0), ("const", 1.0))
    final_expr = ("mul", ("var", 0), ("var", 1))
    raw_by_expr = {
        node_str(parent): 1.0,
        node_str(setup_expr): 0.99,
        node_str(final_expr): 0.60,
    }

    calls = {"n": 0}

    def _inverse_action_fn(*args, **kwargs):
        calls["n"] += 1
        return final_expr, {
            "status": "ok",
            "selected_path": [1, 1],
            "estimated_child_eff_mse": 0.60,
            "tuple_value_estimate": 0.15,
            "tuple_regret_estimate": 0.05,
            "tuple_allocation_estimate": 0.20,
        }

    expr, meta = run_repair_option_action(
        parent,
        {"kind": "identity"},
        None,
        None,
        None,
        None,
        [],
        None,
        None,
        None,
        None,
        5,
        2,
        2,
        max_steps=2,
        min_step_rel_improve=0.05,
        max_setup_steps=1,
        setup_step_value_min=0.20,
        setup_step_regret_max=0.10,
        setup_step_max_worsen=0.05,
        initial_path=[1],
        first_step_expr=setup_expr,
        first_step_meta={
            "status": "ok",
            "selected_path": [1],
            "estimated_child_eff_mse": 0.99,
            "tuple_value_estimate": 0.35,
            "tuple_regret_estimate": 0.05,
            "tuple_allocation_estimate": 0.30,
        },
        current_eff_mse=1.0,
        return_meta=True,
        score_expr_fn=_score_expr_factory(raw_by_expr),
        inverse_action_fn=_inverse_action_fn,
    )

    assert calls["n"] == 1
    assert expr == final_expr
    assert meta["status"] == "ok"
    assert meta["repair_option_steps_attempted"] == 2
    assert meta["repair_option_steps_accepted"] == 1
    assert meta["repair_option_setup_steps_used"] == 1
    assert meta["repair_option_step_nonmyopic_continue"] == [True, False]
    assert meta["repair_option_step_statuses"] == ["ok", "ok"]


def test_repair_option_blocks_setup_step_when_worsening_too_large():
    parent = ("add", ("var", 0), ("var", 1))
    bad_setup_expr = ("mul", ("var", 0), ("const", 3.0))
    raw_by_expr = {
        node_str(parent): 1.0,
        node_str(bad_setup_expr): 1.20,
    }

    expr, meta = run_repair_option_action(
        parent,
        {"kind": "identity"},
        None,
        None,
        None,
        None,
        [],
        None,
        None,
        None,
        None,
        5,
        2,
        2,
        max_steps=2,
        min_step_rel_improve=0.05,
        max_setup_steps=1,
        setup_step_value_min=0.20,
        setup_step_regret_max=0.10,
        setup_step_max_worsen=0.05,
        first_step_expr=bad_setup_expr,
        first_step_meta={
            "status": "ok",
            "selected_path": [1],
            "estimated_child_eff_mse": 1.20,
            "tuple_value_estimate": 0.50,
            "tuple_regret_estimate": 0.01,
            "tuple_allocation_estimate": 0.40,
        },
        current_eff_mse=1.0,
        return_meta=True,
        score_expr_fn=_score_expr_factory(raw_by_expr),
        inverse_action_fn=lambda *args, **kwargs: (None, None),
    )

    assert expr is None
    assert meta["status"] == "repair_option_low_step_gain"
    assert meta["repair_option_setup_steps_used"] == 0
    assert meta["repair_option_step_nonmyopic_continue"] == [False]


def test_repair_option_setup_controller_allows_continue_on_positive_acquisition(monkeypatch):
    parent = ("add", ("var", 0), ("var", 1))
    setup_expr = ("add", ("var", 0), ("const", 1.0))
    final_expr = ("mul", ("var", 0), ("var", 1))
    raw_by_expr = {
        node_str(parent): 1.0,
        node_str(setup_expr): 0.99,
        node_str(final_expr): 0.60,
    }

    calls = {"n": 0}

    def _fake_predict(bundle, rows):
        row = dict(list(rows)[0])
        return {
            "trained": True,
            "rows": [{
                **row,
                "expected_gain_next_under_executor": 0.20,
                "cost_estimate": 0.03,
                "fragility_prob": 0.04,
                "route_flip_prob": 0.01,
                "new_residual_basin_prob": 0.0,
                "acquisition_estimate": 0.12,
            }],
        }

    def _inverse_action_fn(*args, **kwargs):
        calls["n"] += 1
        return final_expr, {
            "status": "ok",
            "selected_path": [1, 1],
            "estimated_child_eff_mse": 0.60,
            "tuple_value_estimate": 0.15,
            "tuple_regret_estimate": 0.05,
            "tuple_allocation_estimate": 0.20,
        }

    monkeypatch.setitem(run_repair_option_action.__globals__, "predict_opportunity_slate", _fake_predict)

    expr, meta = run_repair_option_action(
        parent,
        {"kind": "identity"},
        None,
        None,
        None,
        None,
        [],
        None,
        None,
        None,
        None,
        5,
        2,
        2,
        max_steps=2,
        min_step_rel_improve=0.05,
        max_setup_steps=1,
        setup_step_value_min=0.20,
        setup_step_regret_max=0.10,
        setup_step_max_worsen=0.05,
        initial_path=[1],
        first_step_expr=setup_expr,
        first_step_meta={
            "status": "ok",
            "selected_path": [1],
            "selected_target_mode": "identity",
            "estimated_child_eff_mse": 0.99,
            "tuple_value_estimate": 0.05,
            "tuple_regret_estimate": 0.90,
            "tuple_allocation_estimate": 0.05,
        },
        current_eff_mse=1.0,
        return_meta=True,
        score_expr_fn=_score_expr_factory(raw_by_expr),
        inverse_action_fn=_inverse_action_fn,
        repair_opportunity_controller_enable=True,
        repair_opportunity_bundle={"opportunity_controller_trained": True},
    )

    assert calls["n"] == 1
    assert expr == final_expr
    assert meta["status"] == "ok"
    assert meta["repair_option_setup_controller_requested"] is True
    assert meta["repair_option_setup_controller_used"] is True
    assert meta["repair_option_step_nonmyopic_continue"] == [True, False]
    assert meta["repair_option_step_continue_source"] == ["opportunity_controller", "accept_step"]
    assert meta["repair_option_reveal_trace_count"] == 2
    first_trace = meta["repair_option_reveal_trace"][0]
    assert first_trace["reveal_type"] == "continue_setup_step"
    assert first_trace["decision_source"] == "opportunity_controller"
    assert first_trace["allow_continue"] is True
    assert first_trace["acquisition_estimate"] == 0.12


def test_repair_option_setup_controller_blocks_continue_on_negative_acquisition(monkeypatch):
    parent = ("add", ("var", 0), ("var", 1))
    setup_expr = ("add", ("var", 0), ("const", 1.0))
    raw_by_expr = {
        node_str(parent): 1.0,
        node_str(setup_expr): 0.99,
    }

    calls = {"n": 0}

    def _fake_predict(bundle, rows):
        row = dict(list(rows)[0])
        return {
            "trained": True,
            "rows": [{
                **row,
                "expected_gain_next_under_executor": 0.01,
                "cost_estimate": 0.03,
                "fragility_prob": 0.02,
                "route_flip_prob": 0.0,
                "new_residual_basin_prob": 0.0,
                "acquisition_estimate": -0.02,
            }],
        }

    def _inverse_action_fn(*args, **kwargs):
        calls["n"] += 1
        return None, None

    monkeypatch.setitem(run_repair_option_action.__globals__, "predict_opportunity_slate", _fake_predict)

    expr, meta = run_repair_option_action(
        parent,
        {"kind": "identity"},
        None,
        None,
        None,
        None,
        [],
        None,
        None,
        None,
        None,
        5,
        2,
        2,
        max_steps=2,
        min_step_rel_improve=0.05,
        max_setup_steps=1,
        setup_step_value_min=0.20,
        setup_step_regret_max=0.10,
        setup_step_max_worsen=0.05,
        first_step_expr=setup_expr,
        first_step_meta={
            "status": "ok",
            "selected_path": [1],
            "selected_target_mode": "identity",
            "estimated_child_eff_mse": 0.99,
            "tuple_value_estimate": 0.50,
            "tuple_regret_estimate": 0.01,
            "tuple_allocation_estimate": 0.40,
        },
        current_eff_mse=1.0,
        return_meta=True,
        score_expr_fn=_score_expr_factory(raw_by_expr),
        inverse_action_fn=_inverse_action_fn,
        repair_opportunity_controller_enable=True,
        repair_opportunity_bundle={"opportunity_controller_trained": True},
    )

    assert calls["n"] == 0
    assert expr is None
    assert meta["status"] == "repair_option_low_step_gain"
    assert meta["repair_option_setup_controller_requested"] is True
    assert meta["repair_option_setup_controller_used"] is True
    assert meta["repair_option_step_nonmyopic_continue"] == [False]
    assert meta["repair_option_step_continue_source"] == ["opportunity_controller"]
    assert meta["repair_option_reveal_trace_count"] == 1
    first_trace = meta["repair_option_reveal_trace"][0]
    assert first_trace["reveal_type"] == "continue_setup_step"
    assert first_trace["decision_source"] == "opportunity_controller"
    assert first_trace["allow_continue"] is False
    assert first_trace["acquisition_estimate"] == -0.02
