# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_search.factorized_search.config import (
    InverseSteeringConfig,
    coerce_inverse_steering_config,
)
import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod
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


def test_coerce_inverse_steering_config_reads_prefixed_mapping():
    cfg = coerce_inverse_steering_config({
        "inverse_max_paths": 7,
        "inverse_topk_terms": 5,
        "inverse_local_score_mode": "affine",
        "inverse_spec_enable": True,
        "inverse_spec_enum_max_depth": 3,
        "inverse_target_mode": "identity",
    })

    assert cfg.max_paths == 7
    assert cfg.topk_terms == 5
    assert cfg.inverse_spec_enable is True
    assert cfg.inverse_spec_enum_max_depth == 3
    assert cfg.target_mode == "identity"


def test_run_repair_option_wraps_inverse_kwargs_into_shared_config(monkeypatch):
    captured = {}

    def _fake_run_repair_option_action(*args, **kwargs):
        captured.update(kwargs)
        return None, {"status": "stub"}

    monkeypatch.setattr(explorer_mod, "run_repair_option_action", _fake_run_repair_option_action)

    expr, meta = explorer_mod.run_repair_option(
        ("add", ("var", 0), ("var", 1)),
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
        inverse_spec_enable=True,
        inverse_spec_enum_max_depth=7,
        inverse_spec_recursive_enable=False,
        target_mode="identity",
        return_meta=True,
    )

    assert expr is None
    assert meta["status"] == "stub"
    cfg = captured["inverse_action_config"]
    assert isinstance(cfg, InverseSteeringConfig)
    assert cfg.inverse_spec_enable is True
    assert cfg.inverse_spec_enum_max_depth == 7
    assert cfg.inverse_spec_recursive_enable is False
    assert cfg.target_mode == "identity"


def test_run_repair_option_action_honors_shared_inverse_config():
    parent = ("add", ("var", 0), ("var", 1))
    seen = {}

    def _inverse_action_fn(*args, **kwargs):
        seen.update(kwargs)
        return None, {"status": "repair_option_none"}

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
        max_steps=1,
        initial_path=[1],
        current_eff_mse=1.0,
        return_meta=True,
        score_expr_fn=_score_expr_factory({node_str(parent): 1.0}),
        inverse_action_fn=_inverse_action_fn,
        inverse_action_config=InverseSteeringConfig(
            inverse_spec_enable=True,
            inverse_spec_enum_max_depth=9,
            target_mode="identity",
        ),
    )

    assert expr is None
    assert meta["status"] == "repair_option_none"
    assert seen["inverse_spec_enable"] is True
    assert seen["inverse_spec_enum_max_depth"] == 9
    assert seen["target_mode"] == "identity"


def test_run_repair_option_action_filters_stale_shared_inverse_kwargs():
    parent = ("add", ("var", 0), ("var", 1))
    seen = {}

    def _strict_inverse_action_fn(
        *args,
        var_dims=None,
        inverse_spec_enable=False,
        inverse_spec_enum_max_depth=0,
        inverse_spec_recursive_enable=True,
        complexity_penalty=0.0,
        candidate_paths=None,
        proj=None,
        fp_mode="bits",
        q_scale=2.0,
        q_clip=8.0,
        score_expr_cfg=None,
        target_mode="robust",
        return_meta=False,
    ):
        seen.update(
            {
                "var_dims": var_dims,
                "inverse_spec_enable": inverse_spec_enable,
                "inverse_spec_enum_max_depth": inverse_spec_enum_max_depth,
                "inverse_spec_recursive_enable": inverse_spec_recursive_enable,
                "complexity_penalty": complexity_penalty,
                "candidate_paths": candidate_paths,
                "proj": proj,
                "fp_mode": fp_mode,
                "q_scale": q_scale,
                "q_clip": q_clip,
                "score_expr_cfg": score_expr_cfg,
                "target_mode": target_mode,
                "return_meta": return_meta,
            }
        )
        return None, {"status": "repair_option_none"}

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
        max_steps=1,
        initial_path=[1],
        current_eff_mse=1.0,
        return_meta=True,
        score_expr_fn=_score_expr_factory({node_str(parent): 1.0}),
        inverse_action_fn=_strict_inverse_action_fn,
        inverse_action_config=InverseSteeringConfig(
            inverse_spec_enable=True,
            inverse_spec_enum_max_depth=9,
            inverse_spec_recursive_enable=False,
            inverse_spec_recursive_sr_enable=True,
            target_mode="identity",
        ),
    )

    assert expr is None
    assert meta["status"] == "repair_option_none"
    assert seen["inverse_spec_enable"] is True
    assert seen["inverse_spec_enum_max_depth"] == 9
    assert seen["inverse_spec_recursive_enable"] is False
    assert seen["var_dims"] is None
    assert seen["target_mode"] == "identity"
    assert seen["return_meta"] is True
