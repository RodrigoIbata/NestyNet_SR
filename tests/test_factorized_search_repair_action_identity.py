# SPDX-License-Identifier: MPL-2.0

import random

import torch

import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod
from nestynet_sr.sr_search.factorized_search.engine.signals import (
    InverseSteeringPotential,
    ModeStateFeatures,
    PathStateFeatures,
)
from nestynet_sr.sr_search.factorized_search.policy.guidance import _choose_repair_execution_preview


def test_explorer_reexports_repair_preview_guidance_helper():
    assert explorer_mod._choose_repair_execution_preview is _choose_repair_execution_preview


def _base_diag():
    return InverseSteeringPotential(
        allowed=True,
        reason="ok",
        best_path=(1,),
        best_rel_gain=0.55,
        best_weighted_rel_gain=0.65,
        candidate_paths=((1,),),
        path_rows=(
            PathStateFeatures(
                path=(1,),
                target_mode="identity",
                weighted_rel_gain=0.65,
                rel_gain=0.55,
                valid_frac=0.90,
                confidence=0.88,
                mode_rows=(
                    ModeStateFeatures(target_mode="identity", best_alt_probe_mse=0.06),
                    ModeStateFeatures(target_mode="full", best_alt_probe_mse=0.02),
                ),
            ),
        ),
    )


def test_apply_action_replace_honors_requested_path(monkeypatch):
    expr = ("add", ("var", 0), ("var", 1))

    monkeypatch.setattr(
        explorer_mod,
        "rand_node",
        lambda rng, max_depth, nvars: ("const", 7.0),
    )

    out = explorer_mod.apply_action(
        expr,
        explorer_mod.A_REPLACE,
        random.Random(0),
        max_depth=4,
        nvars=2,
        path=(2,),
    )

    assert out == ("add", ("var", 0), ("const", 7.0))


def test_choose_repair_execution_preview_prefers_analytic_without_clear_gain():
    choice = explorer_mod._choose_repair_execution_preview(
        analytic_preview_expr=("var", 0),
        analytic_preview_meta={"estimated_child_eff_mse": 1.00},
        analytic_preview_rng=None,
        analytic_anchor_path=(1,),
        analytic_preview_paths=[(1,), (1, 1)],
        analytic_preview_path_target_modes=None,
        learned_preview_expr=("var", 1),
        learned_preview_meta={"estimated_child_eff_mse": 0.98},
        learned_preview_rng=None,
        learned_anchor_path=(2,),
        learned_preview_paths=[(2,), (2, 1)],
        learned_preview_path_target_modes={(2,): "identity"},
        learned_preview_source="critic_path_action",
        min_rel_gain=0.05,
    )

    assert choice["source"] == "analytic"
    assert choice["expr"] == ("var", 0)
    assert choice["relative_gain_vs_analytic"] == 0.0


def test_choose_repair_execution_preview_accepts_learned_when_margin_is_real():
    choice = explorer_mod._choose_repair_execution_preview(
        analytic_preview_expr=("var", 0),
        analytic_preview_meta={"estimated_child_eff_mse": 1.00},
        analytic_preview_rng=None,
        analytic_anchor_path=(1,),
        analytic_preview_paths=[(1,), (1, 1)],
        analytic_preview_path_target_modes=None,
        learned_preview_expr=("var", 1),
        learned_preview_meta={"estimated_child_eff_mse": 0.80},
        learned_preview_rng=None,
        learned_anchor_path=(2,),
        learned_preview_paths=[(2,), (2, 1)],
        learned_preview_path_target_modes={(2,): "identity"},
        learned_preview_source="critic_path_action",
        min_rel_gain=0.05,
    )

    assert choice["source"] == "critic_path_action"
    assert choice["expr"] == ("var", 1)
    assert choice["relative_gain_vs_analytic"] > 0.15


def _run_identity_scenario(
    monkeypatch,
    *,
    controller_score: float,
    controller_threshold: float,
    select_action: int | None,
    run_repair_option_result,
    diag_factory=None,
    predict_controller_heads=None,
    preview_observer=None,
    extra_run_kwargs=None,
):
    init_expr = ("sqr", ("var", 0))
    preview_expr = ("add", ("var", 0), ("var", 0))
    repair_expr = ("var", 0)

    def _fake_rand_node(rng, max_depth, nvars):
        return init_expr

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        score_map = {
            init_expr: 1.0,
            preview_expr: 0.8,
            repair_expr: 0.4,
        }
        mse = float(score_map.get(node, 1.2))
        key = ("expr", str(node))
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return mse, key, z, mapping, node
        return mse, key, z, mapping

    def _fake_choose_parent_repair_aware(arch, *args, **kwargs):
        key = next(iter(arch.d.keys()))
        return key, arch.d[key]

    def _fake_apply_inverse_steering_action(*args, **kwargs):
        if callable(preview_observer):
            preview_observer(kwargs)
        meta = {
            "status": "ok",
            "selected_path": [1],
            "selected_target_mode": "identity",
            "selected_path_gain": 0.4,
            "selected_path_gain_pre_cut": 0.45,
            "selected_rel_gain": 0.3,
            "selected_transport_rel": 0.2,
            "selected_lin_rel": 0.1,
            "selected_branch_factor": 1.0,
            "selected_cut_factor": 0.95,
            "selected_effective_n": 12.0,
            "local_candidate_count": 3,
            "estimated_child_raw_mse": 0.7,
            "estimated_child_eff_mse": 0.8,
            "estimated_parent_raw_mse": 0.95,
            "estimated_parent_eff_mse": 1.0,
            "estimated_one_hole_rel_improve_raw": 0.25,
            "estimated_one_hole_rel_improve_eff": 0.2,
            "inverse_opportunity_controller_requested": bool(kwargs.get("repair_opportunity_controller_enable", False)),
            "inverse_opportunity_controller_used": bool(kwargs.get("repair_opportunity_controller_enable", False)),
        }
        if kwargs.get("return_meta", False):
            return preview_expr, meta
        return preview_expr

    def _fake_run_repair_option(*args, **kwargs):
        if callable(run_repair_option_result):
            return run_repair_option_result(*args, **kwargs)
        return run_repair_option_result

    monkeypatch.setattr(explorer_mod, "rand_node", _fake_rand_node)
    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)
    monkeypatch.setattr(explorer_mod, "choose_parent_repair_aware", _fake_choose_parent_repair_aware)
    monkeypatch.setattr(explorer_mod, "estimate_inverse_steering_potential", lambda *a, **k: diag_factory() if callable(diag_factory) else _base_diag())
    monkeypatch.setattr(explorer_mod, "_repair_parent_retry_gate", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(
        explorer_mod,
        "_repair_controller_stagnation_state",
        lambda *a, **k: {
            "visits": 3.0,
            "visits_since_improve": 1.0,
            "stagnation_score": 0.1,
            "stagnation_ratio": 0.1,
        },
    )
    monkeypatch.setattr(
        explorer_mod,
        "_analytic_repair_controller_score",
        lambda row, stats: (
            float(controller_score),
            {
                "potential": 0.6,
                "concentration": 0.5,
                "contrast": 0.4,
                "cost": 0.1,
                "stagnation": 0.1,
            },
        ),
    )
    monkeypatch.setattr(explorer_mod, "_repair_controller_component_gate", lambda *a, **k: (True, []))
    monkeypatch.setattr(explorer_mod, "_repair_controller_threshold", lambda *a, **k: float(controller_threshold))
    monkeypatch.setattr(explorer_mod, "apply_inverse_steering_action", _fake_apply_inverse_steering_action)
    monkeypatch.setattr(explorer_mod, "run_repair_option", _fake_run_repair_option)
    if callable(predict_controller_heads):
        monkeypatch.setattr(explorer_mod, "load_repair_critic_bundle", lambda _path: {"stub": True})
        monkeypatch.setattr(explorer_mod, "predict_repair_controller_heads", predict_controller_heads)
    if select_action is not None:
        monkeypatch.setattr(explorer_mod.Explorer, "select_action", lambda self, s_key, rng, allowed_actions=None: select_action)

    def _target_fn(x):
        return x[:, :1]

    run_kwargs = {
        "repair_controller_critic_enable": bool(callable(predict_controller_heads)),
        "repair_controller_critic_path": "/tmp/stub_repair_critic.pt" if callable(predict_controller_heads) else "",
    }
    if isinstance(extra_run_kwargs, dict):
        run_kwargs.update(extra_run_kwargs)

    return explorer_mod.run_explorer_core(
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
        inverse_gate_min_depth=1,
        inverse_gate_min_size=1,
        inverse_experiment_log_enable=True,
        repair_controller_enable=True,
        repair_controller_min_score=float(controller_threshold),
        **run_kwargs,
    )


def test_repair_option_has_distinct_macro_action_and_keeps_preview_features(monkeypatch):
    repair_expr = ("var", 0)
    repair_meta = {
        "status": "ok",
        "selected_path": [1, 1],
        "selected_target_mode": "full",
        "selected_path_gain": 0.9,
        "selected_path_gain_pre_cut": 1.0,
        "selected_rel_gain": 0.7,
        "selected_transport_rel": 0.6,
        "selected_lin_rel": 0.5,
        "selected_branch_factor": 1.1,
        "selected_cut_factor": 0.98,
        "selected_effective_n": 20.0,
        "local_candidate_count": 7,
        "estimated_child_raw_mse": 0.35,
        "estimated_child_eff_mse": 0.4,
        "estimated_parent_raw_mse": 0.9,
        "estimated_parent_eff_mse": 1.0,
        "estimated_one_hole_rel_improve_raw": 0.61,
        "estimated_one_hole_rel_improve_eff": 0.60,
        "repair_option_anchor_path": [1],
        "repair_option_steps_attempted": 2,
        "repair_option_steps_accepted": 1,
        "repair_option_step_statuses": ["ok", "ok"],
        "repair_option_step_paths": [[1], [1, 1]],
        "repair_option_step_rel_improve": [0.2, 0.6],
    }
    arch = _run_identity_scenario(
        monkeypatch,
        controller_score=0.9,
        controller_threshold=0.3,
        select_action=None,
        run_repair_option_result=(repair_expr, repair_meta),
    )

    ad = getattr(arch, "action_distribution", {})
    counts = ad.get("counts", {})
    assert int(counts.get("repair_option", 0)) == 1
    assert int(counts.get("inv_steer", 0)) == 0

    rows = list(getattr(arch, "inverse_experiment_log", []) or [])
    assert len(rows) == 1
    row = rows[0]
    assert row["macro_action"] == "repair_option"
    assert row["proposal_generator_action"] == "inv_steer"
    assert row["controller_preview_status"] == "ok"
    assert row["repair_option_status"] == "ok"
    assert row["selected_target_mode"] == "identity"
    assert row["selected_path_gain"] == 0.4
    assert row["repair_option_final_selected_target_mode"] == "full"
    assert row["repair_option_final_selected_path_gain"] == 0.9
    assert row["repair_option_steps_attempted"] == 2
    assert "actor_critic_reward" in row
    assert abs(
        float(row["actor_critic_reward"])
        - (
            float(row["actor_critic_reward_log_gain"])
            + float(row["actor_critic_reward_novelty_bonus"])
            + float(row["actor_critic_reward_best_bonus"])
            - float(row["actor_critic_reward_time_penalty"])
        )
    ) < 1.0e-12
    assert row["status"] == "scored"


def test_route_compare_preview_uses_allocator_off_meta(monkeypatch):
    preview_flags: list[bool] = []
    route_compare_rows: list[dict] = []

    def _fake_load_bundle(_path):
        return {"stub": True}

    def _fake_predict_repair_build_route(_bundle, row, **kwargs):
        route_compare_rows.append(dict(row))
        return {
            "trained": True,
            "best_route": "repair",
            "repair_prob": 0.8,
            "build_prob": 0.2,
            "margin_estimate": 0.1,
            "exact_margin": 0.1,
            "repair_summary": {"rows": [{"child_eff_mse": 0.2}]},
            "build_summary": {"rows": [{"child_eff_mse": 0.3}]},
        }

    monkeypatch.setattr(explorer_mod, "load_repair_critic_bundle", _fake_load_bundle)
    monkeypatch.setattr(explorer_mod, "predict_repair_build_route", _fake_predict_repair_build_route)

    _run_identity_scenario(
        monkeypatch,
        controller_score=0.9,
        controller_threshold=0.3,
        select_action=explorer_mod.A_REPLACE,
        run_repair_option_result=(("var", 0), {"status": "ok"}),
        preview_observer=lambda kwargs: preview_flags.append(bool(kwargs.get("repair_opportunity_controller_enable", False))),
        extra_run_kwargs={
            "repair_controller_route_compare_enable": True,
            "repair_controller_route_compare_path": "/tmp/stub_route_compare.pt",
            "repair_opportunity_controller_enable": True,
        },
    )

    assert route_compare_rows
    assert route_compare_rows[0]["inverse_opportunity_controller_used"] is False
    assert True in preview_flags
    assert False in preview_flags


def test_repair_option_and_inv_steer_log_as_distinct_options(monkeypatch):
    def _unexpected_repair(*args, **kwargs):
        raise AssertionError("repair option should not run when controller score is below threshold")

    arch = _run_identity_scenario(
        monkeypatch,
        controller_score=0.1,
        controller_threshold=0.5,
        select_action=explorer_mod.A_INVSTEER,
        run_repair_option_result=_unexpected_repair,
    )

    ad = getattr(arch, "action_distribution", {})
    counts = ad.get("counts", {})
    assert int(counts.get("repair_option", 0)) == 0
    assert int(counts.get("inv_steer", 0)) == 1

    rows = list(getattr(arch, "inverse_experiment_log", []) or [])
    assert len(rows) == 2
    repair_row, inv_row = rows

    assert repair_row["macro_action"] == "repair_option"
    assert repair_row["status"] == "controller_blocked_low_score"
    assert repair_row["controller_preview_status"] == "ok"

    assert inv_row["macro_action"] == "inv_steer"
    assert inv_row["proposal_status"] == "ok"
    assert inv_row["controller_preview_reused"] is True
    assert inv_row["selected_target_mode"] == "identity"
    assert inv_row["status"] == "scored"


def test_repair_controller_uses_learned_path_and_mode_guidance(monkeypatch):
    preview_calls = []

    def _diag_with_alternative_path():
        diag = _base_diag()
        return InverseSteeringPotential(
            allowed=diag.allowed,
            reason=diag.reason,
            best_path=diag.best_path,
            best_rel_gain=diag.best_rel_gain,
            best_weighted_rel_gain=diag.best_weighted_rel_gain,
            candidate_paths=((1,), (1, 1)),
            path_rows=(
                diag.path_rows[0],
                PathStateFeatures(
                    path=(1, 1),
                    target_mode="identity",
                    weighted_rel_gain=0.40,
                    rel_gain=0.35,
                    valid_frac=0.85,
                    confidence=0.82,
                ),
            ),
        )

    def _predict_controller_heads(_bundle, _row):
        return {
            "auxiliary": {
                "utility_score": 0.90,
                "accept_prob": 0.85,
                "positive_reward_prob": 0.88,
                "new_residual_basin_prob": 0.10,
                "new_best_prob": 0.05,
                "reward_per_s_score": 0.80,
            },
            "macro_action": {"trained": False, "best_action": None, "probs": {}},
            "path": {
                "trained": True,
                "best_path": [1, 1],
                "best_target_mode": "affine",
                "rows": [
                    {
                        "path": [1, 1],
                        "target_mode": "identity",
                        "prob": 0.92,
                        "weighted_rel_gain": 0.40,
                        "best_relation": "same",
                        "relation_probs": {"same": 0.94, "ancestor": 0.04, "descendant": 0.02},
                        "best_mode": "affine",
                        "mode_probs": {"identity": 0.05, "affine": 0.90, "full": 0.05},
                        "improvement_estimate": 0.83,
                    },
                    {
                        "path": [1],
                        "target_mode": "identity",
                        "prob": 0.12,
                        "weighted_rel_gain": 0.65,
                        "best_relation": "ancestor",
                        "relation_probs": {"same": 0.05, "ancestor": 0.80, "descendant": 0.10, "disjoint": 0.05},
                        "best_mode": "full",
                        "mode_probs": {"identity": 0.10, "affine": 0.10, "full": 0.80},
                        "improvement_estimate": 0.20,
                    },
                ],
            },
        }

    arch = _run_identity_scenario(
        monkeypatch,
        controller_score=0.9,
        controller_threshold=0.3,
        select_action=None,
        run_repair_option_result=(("var", 0), {
            "status": "ok",
            "selected_path": [1, 1],
            "selected_target_mode": "affine",
            "estimated_child_eff_mse": 0.4,
            "estimated_one_hole_rel_improve_eff": 0.6,
        }),
        diag_factory=_diag_with_alternative_path,
        predict_controller_heads=_predict_controller_heads,
        preview_observer=lambda kwargs: preview_calls.append({
            "candidate_paths": [list(path) for path in list(kwargs.get("candidate_paths") or [])],
            "path_target_modes": {
                tuple(int(v) for v in path): str(mode)
                for path, mode in dict(kwargs.get("path_target_modes") or {}).items()
            },
        }),
    )

    assert len(preview_calls) >= 2
    assert preview_calls[0]["candidate_paths"][0] == [1]
    assert preview_calls[1]["candidate_paths"][0] == [1, 1]
    assert preview_calls[1]["path_target_modes"][(1, 1)] == "affine"

    rows = list(getattr(arch, "inverse_experiment_log", []) or [])
    assert len(rows) == 1
    row = rows[0]
    assert row["controller_policy_path_trained"] is True
    assert row["controller_policy_gate_preview_source"] == "analytic"
    assert row["controller_policy_best_path"] == [1, 1]


def test_macro_build_action_uses_guided_path(monkeypatch):
    build_call = {}

    def _predict_controller_heads(_bundle, _row):
        return {
            "auxiliary": {
                "utility_score": 0.10,
                "accept_prob": 0.10,
                "positive_reward_prob": 0.10,
                "new_residual_basin_prob": 0.05,
                "new_best_prob": 0.01,
                "reward_per_s_score": 0.10,
            },
            "macro_action": {"trained": False, "best_action": None, "probs": {}},
            "path": {
                "trained": True,
                "best_path": [1],
                "best_target_mode": "identity",
                "rows": [
                    {
                        "path": [1],
                        "target_mode": "identity",
                        "prob": 0.95,
                        "weighted_rel_gain": 0.65,
                        "best_relation": "same",
                        "relation_probs": {"same": 0.95, "ancestor": 0.05},
                        "best_mode": "identity",
                        "mode_probs": {"identity": 0.95, "affine": 0.03, "full": 0.02},
                        "improvement_estimate": 0.60,
                    },
                ],
            },
            "value": {"trained": False},
        }

    class _Decision:
        action_name = "replace"
        policy_source = "unit_test"
        learned_confidence = 0.9
        learned_value_estimate = None
        learned_value_normalized = None
        learned_scores = {}
        learned_action_probs = {}

    monkeypatch.setattr(
        explorer_mod.MacroController,
        "select_action",
        lambda self, state, rng, policy_guidance=None: _Decision(),
    )

    def _fake_apply_action(node, action, rng, max_depth, nvars, var_dims=None, reach=None, path=None):
        build_call["action"] = int(action)
        build_call["path"] = None if path is None else tuple(int(v) for v in path)
        return ("var", 0)

    monkeypatch.setattr(explorer_mod, "apply_action", _fake_apply_action)

    arch = _run_identity_scenario(
        monkeypatch,
        controller_score=0.1,
        controller_threshold=0.5,
        select_action=None,
        run_repair_option_result=(("var", 0), {"status": "ok"}),
        predict_controller_heads=_predict_controller_heads,
        extra_run_kwargs={
            "macro_controller_enable": True,
        },
    )

    assert build_call["action"] == explorer_mod.A_REPLACE
    assert build_call["path"] == (1,)

    rows = list(getattr(arch, "inverse_experiment_log", []) or [])
    assert len(rows) == 1
    row = rows[0]
    assert row["macro_action"] == "replace"
    assert row["controller_action_path"] == [1]
    assert row["controller_action_path_source"] == "critic_path_head"
    assert row["controller_action_path_guided"] is True


def test_repair_preview_prefers_slate_ranker_path_action(monkeypatch):
    preview_calls = []

    def _diag_with_alternative_path():
        return explorer_mod.InverseSteeringPotential(
            allowed=True,
            reason="ok",
            best_path=(1,),
            best_rel_gain=0.8,
            best_weighted_rel_gain=0.85,
            candidate_paths=((1,), (1, 1)),
            path_rows=(
                explorer_mod.PathStateFeatures(
                    path=(1,),
                    weighted_rel_gain=0.85,
                    rel_gain=0.8,
                    valid_frac=0.95,
                    confidence=0.90,
                    static_score=1.10,
                    transport_rel=0.60,
                    target_mode="identity",
                ),
                explorer_mod.PathStateFeatures(
                    path=(1, 1),
                    weighted_rel_gain=0.40,
                    rel_gain=0.35,
                    valid_frac=0.70,
                    confidence=0.55,
                    static_score=0.80,
                    transport_rel=0.20,
                    target_mode="affine",
                ),
            ),
        )

    def _predict_controller_heads(_bundle, _row):
        return {
            "auxiliary": {
                "utility_score": 0.05,
                "accept_prob": 0.05,
                "positive_reward_prob": 0.05,
                "new_residual_basin_prob": 0.01,
                "new_best_prob": 0.0,
                "reward_per_s_score": 0.05,
            },
            "path": {
                "trained": True,
                "best_path": [1],
                "best_target_mode": "identity",
                "rows": [
                    {
                        "path": [1],
                        "target_mode": "identity",
                        "prob": 0.95,
                        "weighted_rel_gain": 0.80,
                        "best_relation": "same",
                        "best_mode": "identity",
                        "improvement_estimate": 0.85,
                    },
                    {
                        "path": [1, 1],
                        "target_mode": "affine",
                        "prob": 0.25,
                        "weighted_rel_gain": 0.35,
                        "best_relation": "ancestor",
                        "best_mode": "affine",
                        "improvement_estimate": 0.20,
                    },
                ],
            },
            "path_action": {
                "trained": True,
                "rows": [
                    {
                        "path": [1],
                        "target_mode": "identity",
                        "path_prob": 0.30,
                        "q_estimates": {"inv_steer": 0.20, "repair_option": 0.10},
                    },
                    {
                        "path": [1, 1],
                        "target_mode": "affine",
                        "path_prob": 0.55,
                        "q_estimates": {"inv_steer": 1.40, "repair_option": 1.25},
                    },
                ],
            },
            "value": {"trained": False},
        }

    arch = _run_identity_scenario(
        monkeypatch,
        controller_score=0.9,
        controller_threshold=0.3,
        select_action=None,
        run_repair_option_result=(("var", 0), {
            "status": "ok",
            "selected_path": [1, 1],
            "selected_target_mode": "affine",
            "estimated_child_eff_mse": 0.4,
            "estimated_one_hole_rel_improve_eff": 0.6,
        }),
        diag_factory=_diag_with_alternative_path,
        predict_controller_heads=_predict_controller_heads,
        preview_observer=lambda kwargs: preview_calls.append({
            "candidate_paths": [list(path) for path in list(kwargs.get("candidate_paths") or [])],
            "path_target_modes": {
                tuple(int(v) for v in path): str(mode)
                for path, mode in dict(kwargs.get("path_target_modes") or {}).items()
            },
        }),
    )

    assert len(preview_calls) >= 2
    assert preview_calls[0]["candidate_paths"][0] == [1]
    assert preview_calls[1]["candidate_paths"][0] == [1, 1]
    assert preview_calls[1]["path_target_modes"][(1, 1)] == "affine"

    rows = list(getattr(arch, "inverse_experiment_log", []) or [])
    assert len(rows) == 1
    row = rows[0]
    assert row["controller_policy_gate_preview_source"] == "analytic"
    assert row["controller_policy_best_path"] == [1, 1]
