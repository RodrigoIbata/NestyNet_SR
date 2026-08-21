# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sympy")

from nestynet_sr.sr_search.factorized_search.engine.archive import ResidualBasinArchive
from nestynet_sr.sr_search.factorized_search.expr_ast import collect_paths, eval_node, node_str
import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod
import nestynet_sr.sr_search.factorized_search.oracle_lab as oracle_lab_mod
from nestynet_sr.sr_search.factorized_search.oracle_lab import (
    build_oracle_dataset,
    compile_target_ast,
    compile_target_expression,
    equation_spec_from_dict,
    generate_oracle_opportunity_shadow_dataset,
    generate_oracle_policy_pretrain_dataset,
    generate_oracle_shared_candidate_pretrain_dataset,
    run_oracle_equation,
)
from nestynet_sr.sr_search.factorized_search.opportunity_shadow_eval import OpportunityShadowEvalConfig
from nestynet_sr.sr_search.config import FactorizedSearchConfig


def _payload_with_constants():
    return {
        "id": "demo_constants",
        "basis": ["L", "T", "M"],
        "variables": [
            {"name": "x", "bounds": [0.4, 3.5], "dim": [1, 0, 0]},
            {"name": "v", "bounds": [0.4, 3.5], "dim": [-1, 0, 0]},
        ],
        "constants": [
            {"name": "w", "value": 2.0, "dim": [0, 0, 0]},
            {"name": "a", "value": 1.5, "dim": [0, 0, 0]},
        ],
        "target": {
            "expr": "cos(w*x*v) + a*x*v",
            "dim": [0, 0, 0],
        },
    }


def _payload_simple_linear():
    return {
        "id": "simple_linear",
        "basis": ["L"],
        "variables": [
            {"name": "x", "bounds": [0.3, 3.2], "dim": [1]},
        ],
        "constants": [],
        "target": {
            "expr": "x",
            "dim": [1],
        },
    }


def _payload_oracle_pretrain_demo():
    return {
        "id": "oracle_pretrain_demo",
        "basis": ["L"],
        "variables": [
            {"name": "x", "bounds": [0.3, 3.2], "dim": [1]},
        ],
        "constants": [],
        "target": {
            "expr": "sin(x) + x*(x + x)",
            "dim": [1],
        },
    }


def _payload_inverse_trig_demo():
    return {
        "id": "inverse_trig_demo",
        "basis": ["L"],
        "variables": [
            {"name": "a", "bounds": [0.1, 0.9], "dim": [0.0]},
            {"name": "b", "bounds": [0.1, 1.0], "dim": [0.0]},
            {"name": "c", "bounds": [0.2, 1.0], "dim": [0.0]},
        ],
        "constants": [],
        "target": {
            "expr": "asin(a*sin(b)) + acos(a/c)",
            "dim": [0.0],
        },
    }


def test_compile_target_expression_and_dataset_with_constants():
    spec = equation_spec_from_dict(_payload_with_constants(), source="unit-test")
    target_fn = compile_target_expression(spec)

    x = torch.tensor(
        [
            [0.8, 0.9, 2.0, 1.5],
            [1.1, 0.7, 2.0, 1.5],
        ],
        dtype=torch.float64,
    )
    y = target_fn(x)
    expected = torch.cos(2.0 * x[:, 0:1] * x[:, 1:2]) + 1.5 * x[:, 0:1] * x[:, 1:2]
    assert torch.allclose(y, expected, atol=1.0e-12)

    ds = build_oracle_dataset(spec, target_fn, n_fit=32, n_probe=48, seed=0, dtype=torch.float64)
    assert tuple(ds["x_fit"].shape) == (32, 4)
    assert tuple(ds["x_probe"].shape) == (48, 4)
    assert torch.allclose(ds["x_fit"][:, 2], torch.full((32,), 2.0, dtype=torch.float64))
    assert torch.allclose(ds["x_fit"][:, 3], torch.full((32,), 1.5, dtype=torch.float64))
    assert ds["var_dims"][0] == (1.0, 0.0, 0.0)
    assert ds["var_dims"][2] == (0.0, 0.0, 0.0)
    assert ds["y_dims"] == (0.0, 0.0, 0.0)


def test_equation_spec_validation_rejects_bad_dimension_length():
    bad = _payload_simple_linear()
    bad["target"] = dict(bad["target"])
    bad["target"]["dim"] = [1, 0]
    with pytest.raises(ValueError):
        equation_spec_from_dict(bad, source="bad-dims")


def test_run_oracle_equation_smoke():
    spec = equation_spec_from_dict(_payload_simple_linear(), source="smoke")

    hp = FactorizedSearchConfig()
    hp.n_iter = 150
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 3
    hp.n_fit = 96
    hp.n_probe = 128
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.brute_max_expressions = 200
    hp.refine_enable = False

    report = run_oracle_equation(spec, factorized_search_hp=hp, seed=7, dtype=torch.float64, verbose=False)
    assert report["best"] is not None
    assert math.isfinite(float(report["best"]["mse"]))
    assert float(report["best"]["mse"]) >= 0.0
    assert len(report["results"]) >= 1
    ad = report.get("action_distribution")
    assert isinstance(ad, dict)
    assert isinstance(ad.get("counts"), dict)
    assert isinstance(ad.get("fractions"), dict)
    assert int(ad.get("total_selected", 0)) >= 0
    ps = report.get("score_prescreen_stats")
    assert isinstance(ps, dict)
    assert isinstance(ps.get("full_score_calls_by_action"), dict)
    rs = report.get("route_scheduler_stats")
    assert isinstance(rs, dict)
    assert isinstance(rs.get("route_summary"), dict)
    route_summary = rs.get("route_summary", {})
    selected_expression = int(rs.get("selected_expression_expand", 0))
    selected_opportunity = int(rs.get("selected_opportunity_expand", 0))
    assert int((route_summary.get("expression_expand") or {}).get("count", 0)) == selected_expression
    assert int((route_summary.get("opportunity_expand") or {}).get("count", 0)) == selected_opportunity


def test_compile_target_ast_supports_inverse_trig_and_traverses_children():
    spec = equation_spec_from_dict(_payload_inverse_trig_demo(), source="invtrig")

    truth_ast = compile_target_ast(spec)

    assert node_str(truth_ast) in {
        "(acos((x0*(1/x2)))+asin((sin(x1)*x0)))",
        "(acos(((1/x2)*x0))+asin((sin(x1)*x0)))",
    }
    paths = set(collect_paths(truth_ast))
    assert (1,) in paths
    assert (1, 1) in paths
    assert (2, 1) in paths


def test_invtrig_peel_presearch_recovers_asin_wrapped_inner_expr(monkeypatch):
    gen = torch.Generator().manual_seed(0)
    x_fit = torch.empty((96, 2), dtype=torch.float64)
    x_probe = torch.empty((128, 2), dtype=torch.float64)
    x_fit[:, 0] = 0.1 + 0.8 * torch.rand(96, generator=gen, dtype=torch.float64)
    x_fit[:, 1] = 0.1 + 1.0 * torch.rand(96, generator=gen, dtype=torch.float64)
    x_probe[:, 0] = 0.1 + 0.8 * torch.rand(128, generator=gen, dtype=torch.float64)
    x_probe[:, 1] = 0.1 + 1.0 * torch.rand(128, generator=gen, dtype=torch.float64)
    y_fit = torch.asin(x_fit[:, 0] * torch.sin(x_fit[:, 1])).unsqueeze(-1)
    y_probe = torch.asin(x_probe[:, 0] * torch.sin(x_probe[:, 1])).unsqueeze(-1)

    def _fake_score_expr(
        tree,
        x_fit_local,
        y_fit_local,
        x_probe_local,
        y_probe_local,
        *_args,
        return_expr=False,
        **_kwargs,
    ):
        try:
            pred_probe = eval_node(tree, x_probe_local)
        except Exception:
            return None
        mse = float(torch.mean((pred_probe - y_probe_local) ** 2).item())
        result = (mse, node_str(tree), torch.ones(1, dtype=torch.float64), {"kind": "identity"}, tree)
        return result if return_expr else result[:-1]

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    arch = ResidualBasinArchive()
    solved = explorer_mod._invtrig_peel_presearch(
        arch,
        2,
        ((0.0,), (0.0,)),
        (0.0,),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj=None,
        fp_mode="f32",
        q_scale=1.0,
        q_clip=8.0,
        poly_degree=1,
        brute_depth=3,
        brute_budget=5000,
        refine_enable=False,
        refine_cfg=None,
        refine_state=None,
        early_stop_mse=1.0e-12,
        verbose=False,
    )

    assert solved is True
    best_rows = arch.best(10)
    assert best_rows
    assert any(
        float(row.best_mse) <= 1.0e-12
        and node_str(row.best_expr) in {"asin((x0*sin(x1)))", "asin((sin(x1)*x0))"}
        for row in best_rows
    )


def test_invtrig_peel_presearch_accepts_nonmonomial_dimensionless_carrier(monkeypatch):
    gen = torch.Generator().manual_seed(3)
    x_fit = torch.empty((64, 3), dtype=torch.float64)
    x_probe = torch.empty((96, 3), dtype=torch.float64)
    x_fit[:, 0] = 1.2 + 1.0 * torch.rand(64, generator=gen, dtype=torch.float64)
    x_fit[:, 1] = 0.1 + 0.6 * torch.rand(64, generator=gen, dtype=torch.float64)
    x_fit[:, 2] = 0.2 + 1.1 * torch.rand(64, generator=gen, dtype=torch.float64)
    x_probe[:, 0] = 1.2 + 1.0 * torch.rand(96, generator=gen, dtype=torch.float64)
    x_probe[:, 1] = 0.1 + 0.6 * torch.rand(96, generator=gen, dtype=torch.float64)
    x_probe[:, 2] = 0.2 + 1.1 * torch.rand(96, generator=gen, dtype=torch.float64)

    ratio_fit = x_fit[:, 1] / x_fit[:, 0]
    ratio_probe = x_probe[:, 1] / x_probe[:, 0]
    carrier = (
        "div",
        ("sub", ("cos", ("var", 2)), ("div", ("var", 1), ("var", 0))),
        ("sub", ("const", 1.0), ("mul", ("div", ("var", 1), ("var", 0)), ("cos", ("var", 2)))),
    )
    inner_fit = (torch.cos(x_fit[:, 2]) - ratio_fit) / (1.0 - ratio_fit * torch.cos(x_fit[:, 2]))
    inner_probe = (torch.cos(x_probe[:, 2]) - ratio_probe) / (1.0 - ratio_probe * torch.cos(x_probe[:, 2]))
    y_fit = torch.acos(inner_fit).unsqueeze(-1)
    y_probe = torch.acos(inner_probe).unsqueeze(-1)

    def _fake_score_expr(
        tree,
        x_fit_local,
        y_fit_local,
        x_probe_local,
        y_probe_local,
        *_args,
        return_expr=False,
        **_kwargs,
    ):
        try:
            pred_probe = eval_node(tree, x_probe_local)
        except Exception:
            return None
        mse = float(torch.mean((pred_probe - y_probe_local) ** 2).item())
        result = (mse, node_str(tree), pred_probe.squeeze(-1), {"kind": "identity"}, tree)
        return result if return_expr else result[:-1]

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    arch = ResidualBasinArchive()
    solved = explorer_mod._invtrig_peel_presearch(
        arch,
        3,
        ((0.0,), (0.0,), (0.0,)),
        (0.0,),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj=None,
        fp_mode="f32",
        q_scale=1.0,
        q_clip=8.0,
        poly_degree=1,
        brute_depth=2,
        brute_budget=5000,
        carrier_nodes=[carrier],
        refine_enable=False,
        refine_cfg=None,
        refine_state=None,
        early_stop_mse=1.0e-12,
        verbose=False,
    )

    assert solved is True
    best_rows = arch.best(1)
    assert best_rows
    assert float(best_rows[0].best_mse) < 1.0e-12
    assert node_str(best_rows[0].best_expr) == node_str(("acos", carrier))


def test_gaussian_peel_presearch_positive_target_does_not_raise_dm_nameerror(monkeypatch):
    gen = torch.Generator().manual_seed(1)
    x_fit = torch.empty((24, 2), dtype=torch.float64)
    x_probe = torch.empty((32, 2), dtype=torch.float64)
    x_fit[:, 0] = 0.2 + 0.8 * torch.rand(24, generator=gen, dtype=torch.float64)
    x_fit[:, 1] = 0.2 + 0.8 * torch.rand(24, generator=gen, dtype=torch.float64)
    x_probe[:, 0] = 0.2 + 0.8 * torch.rand(32, generator=gen, dtype=torch.float64)
    x_probe[:, 1] = 0.2 + 0.8 * torch.rand(32, generator=gen, dtype=torch.float64)
    y_fit = (0.2 + 0.1 * x_fit[:, 0]).unsqueeze(-1)
    y_probe = (0.2 + 0.1 * x_probe[:, 0]).unsqueeze(-1)

    monkeypatch.setattr(explorer_mod, "score_expr", lambda *args, **kwargs: None)
    monkeypatch.setattr(explorer_mod, "_enumerate_dim_incremental", lambda *args, **kwargs: [])
    monkeypatch.setattr(explorer_mod, "_enumerate_incremental", lambda *args, **kwargs: [])

    solved = explorer_mod._gaussian_peel_presearch(
        ResidualBasinArchive(),
        2,
        ((0.0,), (0.0,)),
        (0.0,),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj=None,
        fp_mode="f32",
        q_scale=1.0,
        q_clip=8.0,
        poly_degree=1,
        refine_enable=False,
        refine_cfg=None,
        refine_state=None,
        early_stop_mse=1.0e-12,
        verbose=False,
    )

    assert solved is False


def test_gaussian_peel_presearch_accepts_nonmonomial_dimensionless_carrier(monkeypatch):
    gen = torch.Generator().manual_seed(2)
    x_fit = torch.empty((64, 3), dtype=torch.float64)
    x_probe = torch.empty((96, 3), dtype=torch.float64)
    x_fit[:, 0] = 1.0 + 1.0 * torch.rand(64, generator=gen, dtype=torch.float64)
    x_fit[:, 1] = 1.0 + 2.0 * torch.rand(64, generator=gen, dtype=torch.float64)
    x_fit[:, 2] = 1.0 + 2.0 * torch.rand(64, generator=gen, dtype=torch.float64)
    x_probe[:, 0] = 1.0 + 1.0 * torch.rand(96, generator=gen, dtype=torch.float64)
    x_probe[:, 1] = 1.0 + 2.0 * torch.rand(96, generator=gen, dtype=torch.float64)
    x_probe[:, 2] = 1.0 + 2.0 * torch.rand(96, generator=gen, dtype=torch.float64)

    carrier = ("div", ("sub", ("var", 1), ("var", 2)), ("var", 0))
    z_fit = eval_node(carrier, x_fit).squeeze(-1)
    z_probe = eval_node(carrier, x_probe).squeeze(-1)
    y_fit = (torch.reciprocal(x_fit[:, 0]) * torch.exp(-0.5 * torch.square(z_fit))).unsqueeze(-1)
    y_probe = (torch.reciprocal(x_probe[:, 0]) * torch.exp(-0.5 * torch.square(z_probe))).unsqueeze(-1)

    def _fake_score_expr(
        tree,
        x_fit_local,
        y_fit_local,
        x_probe_local,
        y_probe_local,
        *_args,
        return_expr=False,
        **_kwargs,
    ):
        try:
            pred_probe = eval_node(tree, x_probe_local)
        except Exception:
            return None
        mse = float(torch.mean((pred_probe - y_probe_local) ** 2).item())
        result = (mse, node_str(tree), pred_probe.squeeze(-1), {"kind": "identity"}, tree)
        return result if return_expr else result[:-1]

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    arch = ResidualBasinArchive()
    solved = explorer_mod._gaussian_peel_presearch(
        arch,
        3,
        ((0.0,), (0.0,), (0.0,)),
        (0.0,),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj=None,
        fp_mode="f32",
        q_scale=1.0,
        q_clip=8.0,
        poly_degree=1,
        refine_enable=False,
        refine_cfg=None,
        refine_state=None,
        early_stop_mse=1.0e-12,
        carrier_nodes=[carrier],
        verbose=False,
    )

    assert solved is True
    best_rows = arch.best(1)
    assert best_rows
    assert float(best_rows[0].best_mse) < 1.0e-12
    expr_text = node_str(best_rows[0].best_expr)
    assert "exp(" in expr_text
    assert "(1/x0)" in expr_text
    assert "((x1-x2)/x0)" in expr_text


def test_brute_phase_quarantines_gaussian_shadow_and_promotes_structural_solve(monkeypatch):
    x = torch.linspace(0.2, 0.8, 16, dtype=torch.float64).unsqueeze(-1)
    y = (0.5 + 0.1 * x).clone()
    main_arch = ResidualBasinArchive()
    gaussian_arches = []

    monkeypatch.setattr(explorer_mod, "_monomial_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_lorentz_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_planck_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_hyperbolic_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_invtrig_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_enumerate_dim_incremental", lambda *args, **kwargs: [])
    monkeypatch.setattr(explorer_mod, "_enumerate_incremental", lambda *args, **kwargs: [])

    def _fake_gaussian(arch, *args, **kwargs):
        gaussian_arches.append(arch)
        arch.update(
            ("gauss",),
            1.0e-13,
            ("var", 0),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            {"kind": "poly", "coeffs": [0.0, 1.0, 0.0, 0.0], "mu": 0.0, "std": 1.0},
            raw_mse=1.0e-13,
        )
        arch.update(
            ("gauss",),
            2.0e-13,
            ("cos", ("var", 0)),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            {"kind": "identity"},
            raw_mse=2.0e-13,
        )
        return True

    monkeypatch.setattr(explorer_mod, "_gaussian_peel_presearch", _fake_gaussian)

    solved = explorer_mod._run_brute_phase(
        arch=main_arch,
        nvars=1,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=None,
        fp_mode="f32",
        q_scale=1.0,
        q_clip=8.0,
        poly_degree=1,
        var_dims=((0.0,),),
        y_dims=(0.0,),
        brute_depth=0,
        refine_enable=False,
        refine_cfg=None,
        refine_state=None,
        early_stop_mse=1.0e-12,
        verbose=False,
    )

    assert solved is True
    assert gaussian_arches and gaussian_arches[0] is not main_arch
    best_rows = main_arch.best(1)
    assert best_rows
    assert best_rows[0].best_expr == ("cos", ("var", 0))


def test_brute_phase_retries_gaussian_peel_with_archive_carriers(monkeypatch):
    target_carrier = ("div", ("sub", ("var", 1), ("var", 2)), ("var", 0))
    x = torch.tensor(
        [
            [1.2, 1.6, 1.1],
            [1.5, 2.1, 1.2],
            [1.8, 2.4, 1.4],
            [1.4, 1.9, 1.3],
        ],
        dtype=torch.float64,
    )
    y = torch.ones((4, 1), dtype=torch.float64)
    gaussian_call_nodes = []

    monkeypatch.setattr(explorer_mod, "_monomial_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_lorentz_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_planck_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_hyperbolic_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_invtrig_peel_presearch", lambda *args, **kwargs: False)

    def _fake_gaussian(arch, *args, carrier_nodes=None, **kwargs):
        gaussian_call_nodes.append(list(carrier_nodes or ()))
        if carrier_nodes:
            arch.update(
                ("gaussian-retry",),
                1.0e-13,
                ("exp", ("neg", ("sqr", target_carrier))),
                torch.ones(4, dtype=torch.float64),
                {"kind": "identity"},
                raw_mse=1.0e-13,
            )
            return True
        return False

    def _fake_score_expr(
        tree,
        x_fit_local,
        y_fit_local,
        x_probe_local,
        y_probe_local,
        *_args,
        return_expr=False,
        **_kwargs,
    ):
        mse = 1.0e-3 if tree == target_carrier else 1.0
        z = torch.ones(x_probe_local.shape[0], dtype=torch.float64)
        result = (mse, node_str(tree), z, {"kind": "identity"}, tree)
        return result if return_expr else result[:-1]

    monkeypatch.setattr(explorer_mod, "_gaussian_peel_presearch", _fake_gaussian)
    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)
    monkeypatch.setattr(
        explorer_mod,
        "_enumerate_dim_incremental",
        lambda *args, **kwargs: [(2, [target_carrier])],
    )

    arch = ResidualBasinArchive()
    solved = explorer_mod._run_brute_phase(
        arch=arch,
        nvars=3,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=None,
        fp_mode="f32",
        q_scale=1.0,
        q_clip=8.0,
        poly_degree=1,
        var_dims=((0.0,), (0.0,), (0.0,)),
        y_dims=(0.0,),
        brute_depth=2,
        refine_enable=False,
        refine_cfg=None,
        refine_state=None,
        early_stop_mse=1.0e-12,
        verbose=False,
    )

    assert solved is True
    assert len(gaussian_call_nodes) == 2
    assert gaussian_call_nodes[0] == []
    assert any(node_str(node) == node_str(target_carrier) for node in gaussian_call_nodes[1])


def test_brute_phase_retries_invtrig_peel_with_archive_carriers(monkeypatch):
    target_carrier = (
        "div",
        ("sub", ("cos", ("var", 2)), ("div", ("var", 1), ("var", 0))),
        ("sub", ("const", 1.0), ("mul", ("div", ("var", 1), ("var", 0)), ("cos", ("var", 2)))),
    )
    x = torch.tensor(
        [
            [1.2, 0.3, 0.4],
            [1.4, 0.4, 0.6],
            [1.6, 0.5, 0.8],
            [1.8, 0.6, 1.0],
        ],
        dtype=torch.float64,
    )
    y = torch.ones((4, 1), dtype=torch.float64)
    invtrig_call_nodes = []

    monkeypatch.setattr(explorer_mod, "_monomial_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_lorentz_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_planck_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_hyperbolic_peel_presearch", lambda *args, **kwargs: False)
    monkeypatch.setattr(explorer_mod, "_gaussian_peel_presearch", lambda *args, **kwargs: False)

    def _fake_invtrig(arch, *args, carrier_nodes=None, **kwargs):
        invtrig_call_nodes.append(list(carrier_nodes or ()))
        if carrier_nodes:
            arch.update(
                ("invtrig-retry",),
                1.0e-13,
                ("acos", target_carrier),
                torch.ones(4, dtype=torch.float64),
                {"kind": "identity"},
                raw_mse=1.0e-13,
            )
            return True
        return False

    def _fake_score_expr(
        tree,
        x_fit_local,
        y_fit_local,
        x_probe_local,
        y_probe_local,
        *_args,
        return_expr=False,
        **_kwargs,
    ):
        mse = 1.0e-3 if tree == target_carrier else 1.0
        z = torch.ones(x_probe_local.shape[0], dtype=torch.float64)
        result = (mse, node_str(tree), z, {"kind": "identity"}, tree)
        return result if return_expr else result[:-1]

    monkeypatch.setattr(explorer_mod, "_invtrig_peel_presearch", _fake_invtrig)
    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)
    monkeypatch.setattr(
        explorer_mod,
        "_enumerate_dim_incremental",
        lambda *args, **kwargs: [(2, [target_carrier])],
    )

    arch = ResidualBasinArchive()
    solved = explorer_mod._run_brute_phase(
        arch=arch,
        nvars=3,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=None,
        fp_mode="f32",
        q_scale=1.0,
        q_clip=8.0,
        poly_degree=1,
        var_dims=((0.0,), (0.0,), (0.0,)),
        y_dims=(0.0,),
        brute_depth=2,
        refine_enable=False,
        refine_cfg=None,
        refine_state=None,
        early_stop_mse=1.0e-12,
        verbose=False,
    )

    assert solved is True
    assert len(invtrig_call_nodes) == 2
    assert invtrig_call_nodes[0] == []
    assert any(node_str(node) == node_str(target_carrier) for node in invtrig_call_nodes[1])


def test_oracle_fast_benchmark_override_disables_brute_and_gates_mapping():
    hp = FactorizedSearchConfig()
    hp.brute_depth = 3
    hp.score_mapping_family_mode = "full"

    args = oracle_lab_mod._parse_args(["--spec", "dummy.json", "--fast_benchmark"])
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.brute_depth == 0
    assert hp2.score_mapping_family_mode == "full"


def test_oracle_fast_benchmark_preserves_explicit_mapping_mode_override():
    hp = FactorizedSearchConfig()
    hp.brute_depth = 3
    hp.score_mapping_family_mode = "full"

    args = oracle_lab_mod._parse_args(
        ["--spec", "dummy.json", "--fast_benchmark", "--score_mapping_family_mode", "cheap"]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.brute_depth == 0
    assert hp2.score_mapping_family_mode == "cheap"


def test_oracle_wall_time_limit_cli_override_updates_hp():
    hp = FactorizedSearchConfig()
    hp.wall_time_limit_s = None

    args = oracle_lab_mod._parse_args(
        [
            "--spec",
            "dummy.json",
            "--wall_time_limit_s",
            "12.5",
        ]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.wall_time_limit_s == pytest.approx(12.5)


def test_oracle_refine_profile_cli_applies_before_explicit_refine_overrides():
    hp = FactorizedSearchConfig()

    args = oracle_lab_mod._parse_args(
        [
            "--spec",
            "dummy.json",
            "--refine_profile",
            "rare_slate",
            "--refine_mode",
            "inline",
            "--refine_max_trials",
            "7",
            "--refine_optimizer",
            "grid_then_lbfgs",
            "--refine_lbfgs_escalate_improve_factor",
            "3.5",
        ]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.refine_profile == "rare_slate"
    assert hp2.refine_mode == "inline"
    assert hp2.refine_during_brute is True
    assert hp2.refine_during_mutation is True
    assert hp2.refine_during_slate is False
    assert hp2.refine_optimizer == "grid_then_lbfgs"
    assert hp2.refine_lbfgs_escalate_improve_factor == 3.5
    assert hp2.refine_max_trials == 7
    assert hp2.refine_max_variants == 1


def test_oracle_plus_uses_scheduled_refinement_default():
    hp = FactorizedSearchConfig()

    args = oracle_lab_mod._parse_args(["--spec", "dummy.json", "--plus"])
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.refine_enable is True
    assert hp2.refine_profile == "default"
    assert hp2.refine_mode == "slate"
    assert hp2.refine_during_brute is False
    assert hp2.refine_during_mutation is False
    assert hp2.refine_during_slate is True


def test_oracle_refine_profile_accepts_aliases():
    hp = FactorizedSearchConfig()

    args = oracle_lab_mod._parse_args(["--spec", "dummy.json", "--refine_profile", "slate"])
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.refine_profile == "rare_slate"
    assert hp2.refine_mode == "slate"


def test_oracle_inline_profile_remains_cli_compatibility_path():
    hp = FactorizedSearchConfig()

    args = oracle_lab_mod._parse_args(
        ["--spec", "dummy.json", "--plus", "--refine_profile", "inline"]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.refine_enable is True
    assert hp2.refine_profile == "inline"
    assert hp2.refine_mode == "inline"
    assert hp2.refine_during_brute is True
    assert hp2.refine_during_mutation is True
    assert hp2.refine_during_slate is False


def test_oracle_prescreen_cli_override_updates_hp():
    hp = FactorizedSearchConfig()
    hp.score_prescreen_enable = True
    hp.score_prescreen_family_mode = "cheap"

    args = oracle_lab_mod._parse_args(
        [
            "--spec",
            "dummy.json",
            "--no_score_prescreen",
            "--score_prescreen_family_mode",
            "gated",
            "--score_prescreen_residual_family_mode",
            "cheap",
            "--score_prescreen_residual_allow_hint",
            "--score_prescreen_residual_use_global_best",
            "--score_prescreen_residual_parent_best_factor",
            "1.2",
            "--score_prescreen_residual_global_best_factor",
            "1.7",
            "--score_prescreen_parent_best_factor",
            "2.0",
        ]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.score_prescreen_enable is False
    assert hp2.score_prescreen_family_mode == "gated"
    assert hp2.score_prescreen_residual_family_mode == "cheap"
    assert hp2.score_prescreen_residual_allow_hint is True
    assert hp2.score_prescreen_residual_use_global_best is True
    assert hp2.score_prescreen_parent_best_factor == pytest.approx(2.0)
    assert hp2.score_prescreen_residual_parent_best_factor == pytest.approx(1.2)
    assert hp2.score_prescreen_residual_global_best_factor == pytest.approx(1.7)


def test_oracle_route_scheduler_cost_cli_override_updates_hp():
    hp = FactorizedSearchConfig()
    hp.hole_search_route_scheduler_enable = True
    hp.hole_search_route_reward_mode = "penalized"
    hp.hole_search_route_time_penalty = 0.01
    hp.hole_search_route_time_floor = 1.0

    args = oracle_lab_mod._parse_args(
        [
            "--spec",
            "dummy.json",
            "--no_hole_search_route_scheduler",
            "--hole_search_route_reward_mode",
            "per_second",
            "--hole_search_route_time_penalty",
            "0.125",
            "--hole_search_route_time_floor",
            "2.5",
        ]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.hole_search_route_scheduler_enable is False
    assert hp2.hole_search_route_reward_mode == "per_second"
    assert hp2.hole_search_route_time_penalty == pytest.approx(0.125)
    assert hp2.hole_search_route_time_floor == pytest.approx(2.5)


def test_oracle_hole_abstraction_cli_override_updates_hp():
    hp = FactorizedSearchConfig()
    hp.hole_search_abstraction_enable = True
    hp.hole_search_abstraction_on_improve = True
    hp.hole_search_abstraction_on_stall = True
    hp.hole_search_abstraction_stage_enable = True

    args = oracle_lab_mod._parse_args(
        [
            "--spec",
            "dummy.json",
            "--no_hole_search_abstraction",
            "--no_hole_search_abstraction_on_improve",
            "--no_hole_search_abstraction_on_stall",
            "--hole_search_abstraction_cooldown_iters",
            "17",
            "--hole_search_abstraction_max_parents",
            "3",
            "--hole_search_abstraction_max_paths_per_parent",
            "4",
            "--hole_search_abstraction_improve_min_delta_log_mse",
            "0.2",
            "--no_hole_search_abstraction_stage",
            "--hole_search_abstraction_stage_max_entries",
            "41",
            "--hole_search_abstraction_promote_topk",
            "3",
            "--hole_search_abstraction_promote_frontier_floor",
            "5",
        ]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.hole_search_abstraction_enable is False
    assert hp2.hole_search_abstraction_on_improve is False
    assert hp2.hole_search_abstraction_on_stall is False
    assert hp2.hole_search_abstraction_cooldown_iters == 17
    assert hp2.hole_search_abstraction_max_parents == 3
    assert hp2.hole_search_abstraction_max_paths_per_parent == 4
    assert hp2.hole_search_abstraction_improve_min_delta_log_mse == pytest.approx(0.2)
    assert hp2.hole_search_abstraction_stage_enable is False
    assert hp2.hole_search_abstraction_stage_max_entries == 41
    assert hp2.hole_search_abstraction_promote_topk == 3
    assert hp2.hole_search_abstraction_promote_frontier_floor == 5


def test_oracle_stall_cli_override_updates_hp():
    hp = FactorizedSearchConfig()
    hp.stall_window = 500
    hp.stall_patience = 3
    hp.stall_delta = 1.0e-4

    args = oracle_lab_mod._parse_args(
        [
            "--spec",
            "dummy.json",
            "--stall_window",
            "12",
            "--stall_patience",
            "2",
            "--stall_delta",
            "0.05",
        ]
    )
    hp2 = oracle_lab_mod._apply_cli_overrides(hp, args)

    assert hp2.stall_window == 12
    assert hp2.stall_patience == 2
    assert hp2.stall_delta == pytest.approx(0.05)


def test_generate_oracle_policy_pretrain_dataset_smoke():
    spec = equation_spec_from_dict(_payload_oracle_pretrain_demo(), source="oracle-pretrain")

    hp = FactorizedSearchConfig()
    hp.n_fit = 32
    hp.n_probe = 48
    hp.poly_degree = 3
    hp.inverse_max_paths = 4
    hp.inverse_topk_terms = 3
    hp.inverse_shortlist_mult = 2

    payload = generate_oracle_policy_pretrain_dataset(
        [spec],
        factorized_search_hp=hp,
        seeds=[0],
        dtype=torch.float64,
        depth_min=2,
        depth_max=8,
        topk=4,
        max_corrupt_paths_per_spec=2,
        sweep_max_paths=4,
        verbose=False,
    )

    assert payload["mode"] == "oracle_policy_pretrain_dataset"
    assert int(payload["n_rows"]) >= 1
    row = payload["rows"][0]
    assert row["spec_id"] == "oracle_pretrain_demo"
    assert isinstance(row["controller_row"], dict)
    assert isinstance(row["path_labels"], list)
    assert row["controller_row"]["path_summaries"]


def test_generate_oracle_shared_candidate_pretrain_dataset_smoke():
    spec = equation_spec_from_dict(_payload_oracle_pretrain_demo(), source="oracle-shared-candidate")

    hp = FactorizedSearchConfig()
    hp.n_fit = 32
    hp.n_probe = 48
    hp.poly_degree = 3
    hp.inverse_max_paths = 4
    hp.inverse_topk_terms = 3
    hp.inverse_shortlist_mult = 2

    payload = generate_oracle_shared_candidate_pretrain_dataset(
        [spec],
        factorized_search_hp=hp,
        seeds=[0],
        dtype=torch.float64,
        depth_min=2,
        depth_max=8,
        topk=3,
        max_corrupt_paths_per_spec=2,
        sweep_max_paths=4,
        verbose=False,
    )

    assert payload["mode"] == "oracle_shared_candidate_pretrain_dataset"
    assert int(payload["n_rows"]) >= 1
    row = payload["rows"][0]
    assert row["spec_id"] == "oracle_pretrain_demo"
    assert isinstance(row["inverse_repair_slate"], list)
    assert len(row["inverse_repair_slate"]) >= 2
    assert isinstance(row["controller_build_slate"], list)
    assert int(row["controller_build_slate_count"]) == len(row["controller_build_slate"])
    assert "oracle_truth_in_slate" in row
    assert "oracle_truth_path_index" in row
    assert isinstance(row["path_summaries"], list)
    assert row["path_summaries"]
    assert "oracle_relation_to_reference" in row["path_summaries"][0]
    assert "oracle_best_mode" in row["path_summaries"][0]
    cand = row["inverse_repair_slate"][0]
    assert cand["route_source"] == "repair"
    assert cand["exact_child_score_observed"] is True
    assert "local_mapping_kind" in cand
    assert "oracle_path_is_correct" in cand
    assert "oracle_truth_rank_score" in cand
    assert "candidate_evidence_exact_known" in cand
    build_cand = row["controller_build_slate"][0]
    assert build_cand["route_source"] == "build"
    assert build_cand["exact_child_score_observed"] is True
    assert "oracle_truth_rank_score" in build_cand


def test_generate_oracle_opportunity_shadow_dataset_smoke():
    spec = equation_spec_from_dict(_payload_oracle_pretrain_demo(), source="oracle-opportunity-shadow")

    hp = FactorizedSearchConfig()
    hp.n_fit = 32
    hp.n_probe = 48
    hp.poly_degree = 3
    hp.inverse_max_paths = 4
    hp.inverse_topk_terms = 3
    hp.inverse_shortlist_mult = 2

    payload = generate_oracle_opportunity_shadow_dataset(
        [spec],
        factorized_search_hp=hp,
        seeds=[0],
        dtype=torch.float64,
        depth_min=2,
        depth_max=8,
        topk=3,
        max_corrupt_paths_per_spec=2,
        sweep_max_paths=4,
        verbose=False,
        shadow_config=OpportunityShadowEvalConfig(
            shadow_sample_rate=1.0,
            budget_ladder=(0, 1, 2),
            include_repair=True,
            include_build=True,
        ),
    )

    assert payload["mode"] == "opportunity_shadow_dataset"
    assert payload["n_source_rows_total"] >= 1
    assert payload["n_source_rows_sampled"] >= 1
    assert int(payload["n_rows"]) >= 1
    row = payload["rows"][0]
    assert row["route_source"] in {"repair", "build"}
    assert "shadow_prefix_index" in row
    assert "budget_remaining" in row
    assert "coverage_at_budget_0" in row
    assert "expected_gain_next_under_executor" in row
    assert row["label_budget_origin"] == "additional_exact_tokens"


def test_run_oracle_equation_stops_seed_sweep_on_early_stop(monkeypatch):
    spec = equation_spec_from_dict(_payload_simple_linear(), source="seed-early-stop")

    calls = {"n": 0}

    class _FakeArch:
        refine_diagnostics = {
            "score_calls": 4,
            "refinement_attempts": 2,
            "accepted_refinements": 1,
            "attempt_cache_hits": 1,
            "attempt_cache_misses": 3,
            "grid_evals": 8,
            "lbfgs_closures": 6,
            "linear_solves": 5,
            "attempt_cache_size": 2,
        }
        refine_slate_stats = {"total_passes": 1, "total_trials_used": 2}

        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-12,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    def _fake_run_explorer_core(**kwargs):
        calls["n"] += 1
        return _FakeArch()

    monkeypatch.setattr(oracle_lab_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = FactorizedSearchConfig()
    hp.n_iter = 200
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 1
    hp.n_fit = 32
    hp.n_probe = 48
    hp.n_seeds = 5
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.brute_max_expressions = 200
    hp.early_stop_mse = 1.0e-8
    hp.refine_enable = False

    report = run_oracle_equation(spec, factorized_search_hp=hp, seed=7, dtype=torch.float64, verbose=False)
    assert calls["n"] == 1
    assert int(report["n_seeds"]) == 5
    assert int(report["n_seeds_ran"]) == 1
    ad = report.get("action_distribution")
    assert isinstance(ad, dict)
    assert report["refine_diagnostics"]["refinement_attempts"] == 2
    assert report["refine_cost_summary"]["accepted_per_attempt"] == pytest.approx(0.5)
    assert report["refine_cost_summary"]["attempt_cache_hit_rate"] == pytest.approx(0.25)
    assert report["refine_diagnostics_by_seed"][0]["seed_search"] == 7
    assert report["refine_slate_stats_by_seed"][0]["total_trials_used"] == 2


def test_run_oracle_equation_forwards_inverse_knobs(monkeypatch):
    spec = equation_spec_from_dict(_payload_simple_linear(), source="inverse-forward")
    seen = {}

    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-6,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    def _fake_run_explorer_core(**kwargs):
        seen.update(kwargs)
        return _FakeArch()

    monkeypatch.setattr(oracle_lab_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = FactorizedSearchConfig()
    hp.n_iter = 16
    hp.n_fit = 32
    hp.n_probe = 48
    hp.n_seeds = 1
    hp.inverse_steering_enable = True
    hp.inverse_max_paths = 7
    hp.inverse_topk_terms = 5
    hp.inverse_shortlist_mult = 3
    hp.inverse_min_valid_frac = 0.2
    hp.inverse_min_confidence = 0.15
    hp.inverse_confidence_mode = "heuristic"
    hp.inverse_confidence_target_gain = 2.5
    hp.inverse_confidence_floor = 0.2
    hp.inverse_branch_beam_width = 2
    hp.inverse_micro_search_enable = True
    hp.inverse_micro_search_max_depth = 2
    hp.inverse_micro_search_beam_width = 10
    hp.inverse_micro_search_topk = 6
    hp.inverse_micro_search_seed_terms = 4
    hp.inverse_local_score_mode = "strict"
    hp.inverse_gate_enable = False
    hp.repair_controller_enable = True
    hp.repair_controller_min_score = 0.75
    hp.repair_controller_steps = 2
    hp.repair_controller_ancestor_hops = 2
    hp.repair_controller_min_step_rel_improve = 5.0e-4
    hp.repair_controller_adaptive = True
    hp.repair_controller_adapt_quantile = 0.6
    hp.repair_controller_adapt_window = 32
    hp.repair_controller_adapt_min_samples = 6
    hp.repair_controller_min_concentration = 0.42
    hp.repair_controller_potential_weight = 1.2
    hp.repair_controller_concentration_weight = 0.55
    hp.repair_controller_contrast_weight = 0.25
    hp.repair_controller_cost_weight = 0.08
    hp.repair_controller_stagnation_weight = 0.3
    hp.repair_controller_frontier_topk = 11
    hp.repair_controller_stagnation_visits = 5
    hp.repair_controller_focus_prob = 0.4
    hp.repair_controller_parent_max_repeats = 3
    hp.repair_controller_parent_min_eval_gap = 17
    hp.repair_controller_parent_reset_rel_improve = 0.07
    hp.repair_controller_critic_enable = True
    hp.repair_controller_critic_path = "/tmp/demo_repair_critic.pt"
    hp.repair_controller_critic_blend = 0.8
    hp.hole_search_enable = True
    hp.hole_search_mine_cooldown_iters = 73

    run_oracle_equation(spec, factorized_search_hp=hp, seed=3, dtype=torch.float64, verbose=False)
    assert bool(seen.get("inverse_steering_enable", False)) is True
    assert int(seen.get("inverse_max_paths", 0)) == 7
    assert int(seen.get("inverse_topk_terms", 0)) == 5
    assert int(seen.get("inverse_shortlist_mult", 0)) == 3
    assert float(seen.get("inverse_min_valid_frac", 0.0)) == pytest.approx(0.2)
    assert float(seen.get("inverse_min_confidence", 0.0)) == pytest.approx(0.15)
    assert str(seen.get("inverse_confidence_mode", "")) == "heuristic"
    assert float(seen.get("inverse_confidence_target_gain", 0.0)) == pytest.approx(2.5)
    assert float(seen.get("inverse_confidence_floor", 0.0)) == pytest.approx(0.2)
    assert int(seen.get("inverse_branch_beam_width", 0)) == 2
    assert bool(seen.get("inverse_micro_search_enable", False)) is True
    assert int(seen.get("inverse_micro_search_max_depth", 0)) == 2
    assert int(seen.get("inverse_micro_search_beam_width", 0)) == 10
    assert int(seen.get("inverse_micro_search_topk", 0)) == 6
    assert int(seen.get("inverse_micro_search_seed_terms", 0)) == 4
    assert str(seen.get("inverse_local_score_mode", "")) == "strict"
    assert bool(seen.get("inverse_gate_enable", True)) is False
    assert bool(seen.get("repair_controller_enable", False)) is True
    assert float(seen.get("repair_controller_min_score", 0.0)) == pytest.approx(0.75)
    assert int(seen.get("repair_controller_steps", 0)) == 2
    assert int(seen.get("repair_controller_ancestor_hops", 0)) == 2
    assert float(seen.get("repair_controller_min_step_rel_improve", 0.0)) == pytest.approx(5.0e-4)
    assert bool(seen.get("repair_controller_adaptive", False)) is True
    assert float(seen.get("repair_controller_adapt_quantile", 0.0)) == pytest.approx(0.6)
    assert int(seen.get("repair_controller_adapt_window", 0)) == 32
    assert int(seen.get("repair_controller_adapt_min_samples", 0)) == 6
    assert float(seen.get("repair_controller_min_concentration", 0.0)) == pytest.approx(0.42)
    assert float(seen.get("repair_controller_potential_weight", 0.0)) == pytest.approx(1.2)
    assert float(seen.get("repair_controller_concentration_weight", 0.0)) == pytest.approx(0.55)
    assert float(seen.get("repair_controller_contrast_weight", 0.0)) == pytest.approx(0.25)
    assert float(seen.get("repair_controller_cost_weight", 0.0)) == pytest.approx(0.08)
    assert float(seen.get("repair_controller_stagnation_weight", 0.0)) == pytest.approx(0.3)
    assert int(seen.get("repair_controller_frontier_topk", 0)) == 11
    assert int(seen.get("repair_controller_stagnation_visits", 0)) == 5
    assert float(seen.get("repair_controller_focus_prob", 0.0)) == pytest.approx(0.4)
    assert int(seen.get("repair_controller_parent_max_repeats", 0)) == 3
    assert int(seen.get("repair_controller_parent_min_eval_gap", 0)) == 17
    assert float(seen.get("repair_controller_parent_reset_rel_improve", 0.0)) == pytest.approx(0.07)
    assert bool(seen.get("repair_controller_critic_enable", False)) is True
    assert str(seen.get("repair_controller_critic_path", "")) == "/tmp/demo_repair_critic.pt"
    assert float(seen.get("repair_controller_critic_blend", 0.0)) == pytest.approx(0.8)
    assert bool(seen.get("hole_search_enable", False)) is True
    assert int(seen.get("hole_search_mine_cooldown_iters", 0)) == 73


def test_run_oracle_equation_forwards_closure_controller_knobs(monkeypatch):
    spec = equation_spec_from_dict(_payload_simple_linear(), source="closure-forward")
    seen = {}

    class _FakeArch:
        closure_search_stats = {
            "scaffolds_considered": 9,
            "preview_candidates": 12,
            "scored": 3,
            "new_residual_basins": 2,
            "global_best_updates": 1,
            "closure_search_rounds": 2,
            "basis_state_controller_stop_reason": "no_basis_update",
            "basis_state_round_commit_scored": 4,
            "basis_state_round_commit_accepted": 2,
            "basis_state_round_commit_selected": 1,
            "basis_state_round_commit_selected_singleton": 0,
            "basis_state_round_commit_selected_pair": 1,
            "basis_state_pair_precommit_scored": 1,
            "basis_state_pair_precommit_accepted": 1,
            "basis_state_pair_normal_scored": 1,
            "basis_state_pair_normal_accepted": 0,
            "basis_state_pair_rescue_scored": 1,
            "basis_state_pair_rescue_accepted": 0,
            "accepted_pair_events": [
                {
                    "round": 1,
                    "profile": "seed_precommit",
                    "proposal_key": "pair_precommit::p1++q1",
                    "pair_members": ["p1", "q1"],
                    "pair_member_sources": ["exact_scored_singleton", "exact_scored_singleton"],
                    "relation_tags": ["same_interaction", "different_support", "cross_family"],
                    "pair_gain_over_best_singleton": 0.25,
                }
            ],
            "selected_pair_events": [
                {
                    "round": 1,
                    "profile": "seed_precommit",
                    "proposal_key": "pair_precommit::p1++q1",
                    "pair_members": ["p1", "q1"],
                    "pair_member_sources": ["exact_scored_singleton", "exact_scored_singleton"],
                    "relation_tags": ["same_interaction", "different_support", "cross_family"],
                    "pair_gain_over_best_singleton": 0.25,
                }
            ],
            "debug_round_summaries": [
                {
                    "round": 1,
                    "seed_mode": True,
                    "exact_budget": 3,
                    "raw_family_counts": {"periodic": 2, "power": 1},
                    "raw_native_family_counts": {"periodic": 2, "power": 1},
                    "expr_family_counts": {"periodic": 2, "power": 1},
                    "expr_native_family_counts": {"periodic": 2, "power": 1},
                    "dedup_family_counts": {"periodic": 1, "power": 1},
                    "dedup_native_family_counts": {"periodic": 1, "power": 1},
                    "prioritized_family_counts": {"periodic": 1, "power": 1},
                    "prioritized_native_family_counts": {"periodic": 1, "power": 1},
                }
            ],
            "debug_preview_rows": [{"proposal_key": "p1"}],
            "debug_exact_rows": [{"proposal_key": "p1", "accepted": True}],
            "debug_pair_pool": [{"proposal_key": "p1"}],
            "debug_pair_attempts": [{"proposal_key": "pair_rescue::p1++q1"}],
        }

        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-6,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    def _fake_run_explorer_core(**kwargs):
        seen.update(kwargs)
        return _FakeArch()

    monkeypatch.setattr(oracle_lab_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = FactorizedSearchConfig()
    hp.n_iter = 16
    hp.n_fit = 32
    hp.n_probe = 48
    hp.n_seeds = 1
    hp.closure_search_enable = True
    hp.closure_search_families = ["periodic", "power"]
    hp.closure_search_max_proposals = 48
    hp.closure_search_anchors_per_family = 8
    hp.closure_search_preview_topk = 12
    hp.closure_search_exact_topk = 8
    hp.closure_search_beam_width = 6
    hp.closure_search_seed_exact_topk = 5
    hp.closure_search_seed_beam_width = 3
    hp.closure_search_seed_scaffold_reserve = 4
    hp.closure_search_seed_family_cap = 1
    hp.closure_search_seed_exact_bound_bonus = 0.4
    hp.closure_search_pair_normal_enable = True
    hp.closure_search_pair_normal_topk = 4
    hp.closure_search_pair_normal_max_pairs = 2
    hp.closure_search_pair_rescue_enable = True
    hp.closure_search_pair_rescue_topk = 5
    hp.closure_search_pair_rescue_max_pairs = 7
    hp.closure_search_debug_topk = 3

    report = run_oracle_equation(spec, factorized_search_hp=hp, seed=3, dtype=torch.float64, verbose=False)

    assert bool(seen.get("closure_search_enable", False)) is True
    assert list(seen.get("closure_search_families", [])) == ["periodic", "power"]
    assert int(seen.get("closure_search_max_proposals", 0)) == 48
    assert int(seen.get("closure_search_anchors_per_family", 0)) == 8
    assert int(seen.get("closure_search_preview_topk", 0)) == 12
    assert int(seen.get("closure_search_exact_topk", 0)) == 8
    assert int(seen.get("closure_search_beam_width", 0)) == 6
    assert int(seen.get("closure_search_seed_exact_topk", 0)) == 5
    assert int(seen.get("closure_search_seed_beam_width", 0)) == 3
    assert int(seen.get("closure_search_seed_scaffold_reserve", 0)) == 4
    assert int(seen.get("closure_search_seed_family_cap", 0)) == 1
    assert float(seen.get("closure_search_seed_exact_bound_bonus", 0.0)) == pytest.approx(0.4)
    assert bool(seen.get("closure_search_pair_normal_enable", False)) is True
    assert int(seen.get("closure_search_pair_normal_topk", 0)) == 4
    assert int(seen.get("closure_search_pair_normal_max_pairs", 0)) == 2
    assert bool(seen.get("closure_search_pair_rescue_enable", False)) is True
    assert int(seen.get("closure_search_pair_rescue_topk", 0)) == 5
    assert int(seen.get("closure_search_pair_rescue_max_pairs", 0)) == 7
    assert int(seen.get("closure_search_debug_topk", 0)) == 3

    hp_report = report.get("hp", {})
    assert int(hp_report.get("closure_search_beam_width", 0)) == 6
    assert int(hp_report.get("closure_search_seed_exact_topk", 0)) == 5
    assert int(hp_report.get("closure_search_seed_beam_width", 0)) == 3
    assert int(hp_report.get("closure_search_seed_scaffold_reserve", 0)) == 4
    assert int(hp_report.get("closure_search_seed_family_cap", 0)) == 1
    assert float(hp_report.get("closure_search_seed_exact_bound_bonus", 0.0)) == pytest.approx(0.4)
    assert bool(hp_report.get("closure_search_pair_normal_enable", False)) is True
    assert int(hp_report.get("closure_search_pair_normal_topk", 0)) == 4
    assert int(hp_report.get("closure_search_pair_normal_max_pairs", 0)) == 2
    assert bool(hp_report.get("closure_search_pair_rescue_enable", False)) is True
    assert int(hp_report.get("closure_search_pair_rescue_topk", 0)) == 5
    assert int(hp_report.get("closure_search_pair_rescue_max_pairs", 0)) == 7
    assert int(hp_report.get("closure_search_debug_topk", 0)) == 3
    closure_summary = report.get("closure_search_summary", {})
    assert int(closure_summary.get("scaffolds_considered", 0)) == 9
    assert int(closure_summary.get("round_commit_scored", 0)) == 4
    assert int(closure_summary.get("round_commit_accepted", 0)) == 2
    assert int(closure_summary.get("round_commit_selected", 0)) == 1
    assert int(closure_summary.get("round_commit_selected_singleton", 0)) == 0
    assert int(closure_summary.get("round_commit_selected_pair", 0)) == 1
    assert int(closure_summary.get("pair_precommit_scored", 0)) == 1
    assert int(closure_summary.get("pair_normal_scored", 0)) == 1
    assert int(closure_summary.get("pair_rescue_scored", 0)) == 1
    assert list(closure_summary.get("accepted_pair_events", [])) == [
        {
            "round": 1,
            "profile": "seed_precommit",
            "proposal_key": "pair_precommit::p1++q1",
            "pair_members": ["p1", "q1"],
            "pair_member_sources": ["exact_scored_singleton", "exact_scored_singleton"],
            "relation_tags": ["same_interaction", "different_support", "cross_family"],
            "pair_gain_over_best_singleton": 0.25,
        }
    ]
    assert list(closure_summary.get("selected_pair_events", [])) == [
        {
            "round": 1,
            "profile": "seed_precommit",
            "proposal_key": "pair_precommit::p1++q1",
            "pair_members": ["p1", "q1"],
            "pair_member_sources": ["exact_scored_singleton", "exact_scored_singleton"],
            "relation_tags": ["same_interaction", "different_support", "cross_family"],
            "pair_gain_over_best_singleton": 0.25,
        }
    ]
    closure_debug = report.get("closure_search_debug", {})
    assert int((closure_debug.get("summary") or {}).get("round_commit_scored", 0)) == 4
    assert int((closure_debug.get("summary") or {}).get("round_commit_selected_pair", 0)) == 1
    assert list(closure_debug.get("round_summaries", [])) == [
        {
            "round": 1,
            "seed_mode": True,
            "exact_budget": 3,
            "raw_family_counts": {"periodic": 2, "power": 1},
            "raw_native_family_counts": {"periodic": 2, "power": 1},
            "expr_family_counts": {"periodic": 2, "power": 1},
            "expr_native_family_counts": {"periodic": 2, "power": 1},
            "dedup_family_counts": {"periodic": 1, "power": 1},
            "dedup_native_family_counts": {"periodic": 1, "power": 1},
            "prioritized_family_counts": {"periodic": 1, "power": 1},
            "prioritized_native_family_counts": {"periodic": 1, "power": 1},
        }
    ]
    assert list(closure_debug.get("preview_rows", [])) == [{"proposal_key": "p1"}]
    assert list(closure_debug.get("exact_rows", [])) == [{"proposal_key": "p1", "accepted": True}]
    assert list(closure_debug.get("accepted_pair_events", [])) == [
        {
            "round": 1,
            "profile": "seed_precommit",
            "proposal_key": "pair_precommit::p1++q1",
            "pair_members": ["p1", "q1"],
            "pair_member_sources": ["exact_scored_singleton", "exact_scored_singleton"],
            "relation_tags": ["same_interaction", "different_support", "cross_family"],
            "pair_gain_over_best_singleton": 0.25,
        }
    ]
    assert list(closure_debug.get("selected_pair_events", [])) == [
        {
            "round": 1,
            "profile": "seed_precommit",
            "proposal_key": "pair_precommit::p1++q1",
            "pair_members": ["p1", "q1"],
            "pair_member_sources": ["exact_scored_singleton", "exact_scored_singleton"],
            "relation_tags": ["same_interaction", "different_support", "cross_family"],
            "pair_gain_over_best_singleton": 0.25,
        }
    ]


def test_run_oracle_equation_aggregates_hole_search_stats(monkeypatch):
    spec = equation_spec_from_dict(_payload_simple_linear(), source="hole-stats")
    calls = {"n": 0}
    stats_rows = [
        {
            "prepare_calls": 7,
            "prepared_executable_checks": 11,
            "prepared_resolution_live_archive": 2,
            "prepared_resolution_snapshot": 5,
            "prepared_resolution_missing": 4,
            "prepare_prune_wall_seconds": 0.3,
            "prepare_mine_wall_seconds": 1.2,
            "prepare_select_wall_seconds": 0.8,
            "prepare_wall_seconds": 2.5,
            "selected": 3,
            "fired": 1,
            "ingested": 2,
            "mined": 1,
            "selected_with_any_frontier_entries": 3,
            "selected_with_nonempty_frontier": 2,
            "selected_with_opportunity": 2,
            "selected_with_opportunity_inverse_slate": 1,
            "selected_with_opportunity_archive_mine": 1,
            "selected_with_opportunity_other": 0,
            "selected_with_resolved_parent": 1,
            "invalidated_parent": 1,
            "invalidated_parent_inverse_slate": 1,
            "invalidated_parent_archive_mine": 0,
            "invalidated_parent_other": 0,
            "run_hole_search_action_called": 1,
            "child_expr_none": 1,
            "frontier_size": 4,
            "last_mined_iter": 17,
            "best_eff_mse": 0.4,
        },
        {
            "prepare_calls": 9,
            "prepared_executable_checks": 13,
            "prepared_resolution_live_archive": 1,
            "prepared_resolution_snapshot": 7,
            "prepared_resolution_missing": 5,
            "prepare_prune_wall_seconds": 0.4,
            "prepare_mine_wall_seconds": 1.8,
            "prepare_select_wall_seconds": 1.1,
            "prepare_wall_seconds": 3.7,
            "selected": 5,
            "fired": 2,
            "ingested": 4,
            "mined": 2,
            "selected_with_any_frontier_entries": 4,
            "selected_with_nonempty_frontier": 3,
            "selected_with_opportunity": 3,
            "selected_with_opportunity_inverse_slate": 1,
            "selected_with_opportunity_archive_mine": 2,
            "selected_with_opportunity_other": 0,
            "selected_with_resolved_parent": 2,
            "invalidated_parent": 1,
            "invalidated_parent_inverse_slate": 0,
            "invalidated_parent_archive_mine": 1,
            "invalidated_parent_other": 0,
            "run_hole_search_action_called": 2,
            "child_expr_none": 0,
            "frontier_size": 7,
            "last_mined_iter": 23,
            "best_eff_mse": 0.2,
        },
    ]

    class _FakeArch:
        def __init__(self, hole_stats):
            self.hole_search_stats = hole_stats
            self.inverse_experiment_log = []
            self.repair_controller_stats = None

        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-4,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    def _fake_run_explorer_core(**kwargs):
        idx = calls["n"]
        calls["n"] += 1
        return _FakeArch(stats_rows[idx])

    monkeypatch.setattr(oracle_lab_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = FactorizedSearchConfig()
    hp.n_iter = 20
    hp.n_fit = 32
    hp.n_probe = 48
    hp.n_seeds = 2
    hp.split_iter_across_seeds = True
    hp.return_topk = 1
    hp.refine_enable = False

    report = run_oracle_equation(spec, factorized_search_hp=hp, seed=5, dtype=torch.float64, verbose=False)
    hs = report["hole_search_stats"]
    assert hs["prepare_calls"] == 16
    assert hs["prepared_executable_checks"] == 24
    assert hs["prepared_resolution_live_archive"] == 3
    assert hs["prepared_resolution_snapshot"] == 12
    assert hs["prepared_resolution_missing"] == 9
    assert hs["prepare_prune_wall_seconds"] == pytest.approx(0.7)
    assert hs["prepare_mine_wall_seconds"] == pytest.approx(3.0)
    assert hs["prepare_select_wall_seconds"] == pytest.approx(1.9)
    assert hs["prepare_wall_seconds"] == pytest.approx(6.2)
    assert hs["selected"] == 8
    assert hs["fired"] == 3
    assert hs["ingested"] == 6
    assert hs["mined"] == 3
    assert hs["selected_with_any_frontier_entries"] == 7
    assert hs["selected_with_nonempty_frontier"] == 5
    assert hs["selected_with_opportunity"] == 5
    assert hs["selected_with_opportunity_inverse_slate"] == 2
    assert hs["selected_with_opportunity_archive_mine"] == 3
    assert hs["selected_with_opportunity_other"] == 0
    assert hs["selected_with_resolved_parent"] == 3
    assert hs["invalidated_parent"] == 2
    assert hs["invalidated_parent_inverse_slate"] == 1
    assert hs["invalidated_parent_archive_mine"] == 1
    assert hs["invalidated_parent_other"] == 0
    assert hs["run_hole_search_action_called"] == 3
    assert hs["child_expr_none"] == 1
    assert hs["frontier_size"] == 7
    assert hs["last_mined_iter"] == 23
    assert hs["best_eff_mse"] == pytest.approx(0.2)
