# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
import math

import pytest
import torch

import nestynet_sr.sr_de.factorized_de as factorized_search_de_mod
import nestynet_sr.sr_search.factorized_search.oracle_lab_de as oracle_lab_de_mod
from nestynet_sr.sr_search.factorized_search.domain_projection import (
    domain_projection_is_acceptable,
    eval_node_with_domain_projection,
)
from nestynet_sr.sr_search.factorized_search.engine.scoring import score_expr
from nestynet_sr.sr_search.factorized_search.expr_mapping import fit_best
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_de import (
    FactorizedSearchDERescueConfig,
    FactorizedSearchDEResult,
    DEFeatureGroup,
    DESearchConfig,
    factorized_search_candidate_to_feature_predictor,
    de_lab_spec_from_de_cfg,
    default_physics_rescue_hp,
    evaluate_factorized_search_candidate,
    factorized_search_report_to_de_result,
    factorized_search_report_to_rhs_callable,
    normalized_rmse,
    run_direct_residual_fss_from_feature_groups,
    run_regularized_implicit_residual_fss_from_feature_groups,
    run_factorized_search_de_from_surrogate,
    run_factorized_search_de_from_surrogates,
    run_factorized_de_from_feature_groups,
    validate_order2_generator_witness,
)
from nestynet_sr.sr_search.factorized_search.oracle_lab_de import DEFeatureTensors, DELabSpec


class _FakeArch:
    def best(self, k):
        rec = type(
            "_Rec",
            (),
            {
                "best_mse": 1.0e-12,
                "best_expr": ("var", 1),  # u
                "mapping": {"kind": "poly", "coeffs": [0.0, -0.5], "mu": 0.0, "std": 1.0},
            },
        )
        return [rec]


class _ExpDecaySurrogate(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x[:, :1])

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((int(x.shape[0]), int(x.shape[1])), dtype=x.dtype, device=x.device)
        out[:, 0] = -0.5 * torch.exp(-0.5 * x[:, 0])
        return out

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            (int(x.shape[0]), int(x.shape[1]), int(x.shape[1])),
            dtype=x.dtype,
            device=x.device,
        )
        out[:, 0, 0] = 0.25 * torch.exp(-0.5 * x[:, 0])
        return out


def _make_feature_group(
    group_id: str,
    *,
    decay: float,
    use_for_fit: bool = True,
    use_for_probe: bool = True,
) -> DEFeatureGroup:
    x_fit = torch.linspace(0.1, 1.0, 16, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 16, dtype=torch.float64).reshape(-1, 1)
    u_fit = torch.exp(-float(decay) * x_fit)
    u_probe = torch.exp(-float(decay) * x_probe)
    return DEFeatureGroup(
        id=group_id,
        features=DEFeatureTensors(
            x_fit=x_fit,
            u_fit=u_fit,
            du_fit=-float(decay) * u_fit,
            d2u_fit=(float(decay) ** 2) * u_fit,
            x_probe=x_probe,
            u_probe=u_probe,
            du_probe=-float(decay) * u_probe,
            d2u_probe=(float(decay) ** 2) * u_probe,
        ),
        use_for_fit=bool(use_for_fit),
        use_for_probe=bool(use_for_probe),
    )


def _sqrt_decay_group() -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 32, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 32, dtype=torch.float64).reshape(-1, 1)
    u_fit = torch.square(2.0 - 0.5 * x_fit)
    u_probe = torch.square(2.0 - 0.5 * x_probe)
    du_fit = -torch.sqrt(u_fit)
    du_probe = -torch.sqrt(u_probe)
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=torch.zeros_like(u_fit),
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=torch.zeros_like(u_probe),
    )
    return DEFeatureGroup(id="sqrt_decay", features=features)


def _de010_like_group() -> DEFeatureGroup:
    x = torch.linspace(0.001, 8.0, 512, dtype=torch.float64).reshape(-1, 1)
    u = 0.73 / x
    du_clean = -u / x
    du_observed = 0.52 * du_clean
    fit = torch.arange(0, 512, 2, dtype=torch.long)
    probe = torch.arange(1, 512, 2, dtype=torch.long)
    features = DEFeatureTensors(
        x_fit=x.index_select(0, fit),
        u_fit=u.index_select(0, fit),
        du_fit=du_observed.index_select(0, fit),
        d2u_fit=torch.zeros_like(u.index_select(0, fit)),
        x_probe=x.index_select(0, probe),
        u_probe=u.index_select(0, probe),
        du_probe=du_observed.index_select(0, probe),
        d2u_probe=torch.zeros_like(u.index_select(0, probe)),
    )
    return DEFeatureGroup(id="de010_like", features=features)


def _sqrt_decay_group_with_boundary_dent(
    group_id: str,
    *,
    shift: float,
    dent: float = 1.0e-6,
) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 96, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.05, 2.0, 96, dtype=torch.float64).reshape(-1, 1)
    a = 1.0 + float(shift)
    b = 0.45
    u_fit = torch.square(a - b * x_fit)
    u_probe = torch.square(a - b * x_probe)
    u_fit = u_fit.clone()
    u_probe = u_probe.clone()
    u_fit[-2:] -= float(dent)
    u_probe[-4:] -= float(dent)
    du_fit = -2.0 * b * (a - b * x_fit)
    du_probe = -2.0 * b * (a - b * x_probe)
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=torch.zeros_like(u_fit),
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=torch.zeros_like(u_probe),
    )
    return DEFeatureGroup(id=str(group_id), features=features)


def _sho_group(
    group_id: str,
    *,
    omega: float = 3.0,
    phase: float = 0.0,
    target_sign: float = -1.0,
    noise_rel: float = 0.03,
) -> DEFeatureGroup:
    x = torch.linspace(0.0, 2.5, 96, dtype=torch.float64).reshape(-1, 1)
    arg = float(omega) * x + float(phase)
    u = torch.cos(arg)
    du = -float(omega) * torch.sin(arg)
    d2u_clean = float(target_sign) * (float(omega) ** 2) * u
    noise = float(noise_rel) * (float(omega) ** 2) * torch.sin(5.0 * x + 0.7 + float(phase))
    d2u = d2u_clean + noise
    fit = torch.arange(0, 64, dtype=torch.long)
    probe = torch.arange(32, 96, dtype=torch.long)
    return DEFeatureGroup(
        id=str(group_id),
        features=DEFeatureTensors(
            x_fit=x.index_select(0, fit),
            u_fit=u.index_select(0, fit),
            du_fit=du.index_select(0, fit),
            d2u_fit=d2u.index_select(0, fit),
            x_probe=x.index_select(0, probe),
            u_probe=u.index_select(0, probe),
            du_probe=du.index_select(0, probe),
            d2u_probe=d2u.index_select(0, probe),
        ),
    )


def test_de_lab_spec_from_de_cfg_bridges_units_and_fixed_constants():
    us = UnitSystem(base=("L", "T"))
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        fixed_const_dims={"omega": us.dim([0, -1])},
        fixed_const_values={"omega": 2.0},
    )
    cfg = DESearchConfig(
        x_axis=0,
        order_candidates=(1, 2),
        include_x=False,
        units_spec=units_spec,
        enforce_units=True,
    )

    spec = de_lab_spec_from_de_cfg(cfg)

    assert spec.id == "surrogate_de"
    assert spec.order_candidates == (1, 2)
    assert spec.include_x is False
    assert spec.include_u is True
    assert spec.include_du is False
    assert spec.traj_metric == "max"
    assert spec.split_mode == "traj_holdout"
    assert spec.dims is not None
    assert spec.dims.basis == ("L", "T")
    assert spec.dims.x_dim == (0.0, 1.0)
    assert spec.dims.u_dim == (1.0, 0.0)
    assert len(spec.constants) == 1
    assert spec.constants[0].name == "omega"
    assert float(spec.constants[0].value) == 2.0
    assert spec.constants[0].dim == (0.0, -1.0)


def test_factorized_search_report_to_rhs_callable_respects_feature_switches(monkeypatch):
    seen: list[tuple[float, ...]] = []

    def _fake_predictor(candidate, *, dtype=torch.float64):
        def _predict(features_1d):
            seen.append(tuple(float(v) for v in features_1d))
            return 0.0

        return _predict

    monkeypatch.setattr(
        factorized_search_de_mod,
        "factorized_search_candidate_to_feature_predictor",
        _fake_predictor,
    )

    cases = [
        (
            {"order": 1, "include_x": True, "include_u": False, "feature_names": ["x0"]},
            (2.0,),
        ),
        (
            {"order": 1, "include_x": False, "include_u": True, "feature_names": ["u"]},
            (3.0,),
        ),
        (
            {"order": 1, "feature_names": ["x0", "u"]},
            (2.0, 3.0),
        ),
        (
            {"order": 2, "feature_names": ["x0", "u", "du"]},
            (2.0, 3.0, 4.0),
        ),
        (
            {
                "order": 1,
                "feature_names": ["x0", "omega"],
                "constants_ordered": [{"name": "omega", "value": 5.0}],
            },
            (2.0, 5.0),
        ),
    ]
    for report, expected in cases:
        seen.clear()
        order, rhs_fn = factorized_search_report_to_rhs_callable(report)
        if int(order) == 1:
            rhs_fn(2.0, [3.0])
        else:
            rhs_fn(2.0, [3.0, 4.0])
        assert seen == [expected]


def test_factorized_search_report_to_de_result_handles_empty_report():
    report = {
        "best": None,
        "x_axis": 0,
        "include_x": True,
        "include_u": True,
        "include_du": False,
        "per_order": [
            {
                "order": 1,
                "feature_names": ["x0", "u"],
                "best_mse": None,
                "top_candidates": [],
            }
        ],
    }

    result = factorized_search_report_to_de_result(report)

    assert isinstance(result, FactorizedSearchDEResult)
    assert result.order == 1
    assert result.feature_names == ["x0", "u"]
    assert math.isinf(result.probe_rms)
    assert result.rhs_ast is None
    assert result.diagnostics["status"] == "NO_CANDIDATE"
    assert result.diagnostics["failure_kind"] == "no_factorized_search_candidate"
    assert result.diagnostics["domain_ok"] is False


def test_de_lab_spec_from_de_cfg_respects_feature_flags():
    cfg = DESearchConfig(
        x_axis=0,
        order_candidates=(2,),
        include_x=True,
        include_u=False,
        include_du=False,
    )

    spec = de_lab_spec_from_de_cfg(cfg)

    assert spec.order_candidates == (2,)
    assert spec.include_x is True
    assert spec.include_u is False
    assert spec.include_du is False


def test_default_physics_rescue_hp_uses_poly_only_mapping_families():
    hp = default_physics_rescue_hp(preset="fast")

    assert hp.score_mapping_family_mode == "poly_only"
    assert hp.brute_score_mapping_family_mode == "poly_only"
    assert hp.score_head_enable is False
    assert hp.de_score_head_policy == "proposal_only"
    assert hp.de_accept_hidden_score_head is False
    assert hp.de_score_head_untyped_enable is False
    assert hp.score_pade_structural_enable is True
    assert hp.score_finite_mask_enable is True
    assert hp.score_finite_mask_min_probe_frac == pytest.approx(0.98)
    assert hp.score_domain_projection_enable is True
    assert hp.score_domain_projection_rel_tol == pytest.approx(1.0e-2)


def test_direct_residual_attempt_hp_enables_additive_composer():
    hp = default_physics_rescue_hp(preset="fast")
    hp.return_topk = 4
    hp.de_sparse_combo_pool_topk = 2
    hp.de_sparse_combo_max_terms = 2
    hp.de_sparse_combo_beam = 4

    out_auto = factorized_search_de_mod._direct_residual_attempt_hp(hp, autonomous=True)
    out_full = factorized_search_de_mod._direct_residual_attempt_hp(hp, autonomous=False)

    assert out_auto.de_sparse_combo_enable is True
    assert out_auto.de_sparse_combo_pre_mutation_enable is True
    assert out_auto.return_topk >= 16
    assert out_auto.de_sparse_combo_pool_topk >= 16
    assert out_auto.de_sparse_combo_max_terms >= 4
    assert out_auto.de_sparse_combo_beam >= 32
    assert out_auto.de_sparse_combo_backward_prune is True
    assert out_auto.score_prescreen_enable is False
    assert out_auto.plateau_stop_enable is True
    assert out_auto.plateau_stop_max_soft_restarts == 2
    assert out_auto.plateau_stop_min_evals == 2000
    assert out_full.de_sparse_combo_enable is True
    assert out_full.de_sparse_combo_pool_topk >= 32
    assert out_full.score_prescreen_enable is False
    assert out_full.plateau_stop_enable is True
    assert out_full.plateau_stop_max_soft_restarts == 2
    assert out_full.plateau_stop_min_evals == 2000


def test_factorized_search_report_to_de_result_exposes_simplified_equation_fields():
    report = {
        "x_axis": 0,
        "include_x": False,
        "include_u": True,
        "include_du": False,
        "constants_ordered": [],
        "per_order": [{"order": 1, "feature_names": ["u"]}],
        "best": {
            "order": 1,
            "expr": "u",
            "expr_ast": ("var", 0),
            "mse": 0.0,
            "score": 0.0,
            "score_raw": 0.0,
            "size": 1,
            "mapping_complexity": 1,
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "mapping_kind": "poly",
        },
    }

    res = factorized_search_report_to_de_result(report)

    assert res.expr_ast == ("var", 0)
    assert res.mapping["kind"] == "poly"
    assert res.residual_ast_raw is not None
    assert res.residual_ast_simplified is not None
    assert res.canonical_equation_raw
    assert res.canonical_equation_simplified
    assert res.canonical_equation == res.canonical_equation_simplified
    assert "symbolic_size_simplified" in res.diagnostics


def test_broad_structural_gate_rejects_log_zero_even_with_projection():
    row = {
        "order": 1,
        "expr_ast": ("log", ("const", 0.0)),
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        "mse": 0.0,
        "score": 0.0,
        "score_raw": 0.0,
        "domain_projection": {"enabled": True, "ok": True},
    }

    diag = factorized_search_de_mod._broad_row_structural_safety(
        row,
        expr_ast=row["expr_ast"],
        mapping=row["mapping"],
    )

    assert diag["structural_ok"] is False
    assert diag["structural_hard_reject"] is True
    assert "log_nonpositive_constant" in diag["structural_reasons"]


def test_factorized_search_report_to_de_result_skips_structurally_rejected_best():
    unsafe = {
        "order": 1,
        "expr": "log(0)",
        "expr_ast": ("log", ("const", 0.0)),
        "mse": 0.0,
        "score": 0.0,
        "score_raw": 0.0,
        "size": 1,
        "mapping_complexity": 1,
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        "mapping_kind": "poly",
        "structural_ok": False,
        "structural_hard_reject": True,
        "structural_reasons": ["log_nonpositive_constant"],
    }
    safe = {
        "order": 1,
        "expr": "u",
        "expr_ast": ("var", 0),
        "mse": 1.0e-12,
        "score": 1.0e-12,
        "score_raw": 1.0e-12,
        "size": 1,
        "mapping_complexity": 1,
        "mapping": {"kind": "poly", "coeffs": [0.0, -0.5], "mu": 0.0, "std": 1.0},
        "mapping_kind": "poly",
        "structural_ok": True,
        "structural_hard_reject": False,
        "structural_reasons": [],
    }
    report = {
        "x_axis": 0,
        "include_x": False,
        "include_u": True,
        "include_du": False,
        "constants_ordered": [],
        "per_order": [{"order": 1, "feature_names": ["u"], "results": [unsafe, safe]}],
        "best": unsafe,
    }

    res = factorized_search_report_to_de_result(report)

    assert res.expr_ast == ("var", 0)
    assert res.diagnostics["structural_ok"] is True
    assert res.diagnostics["structural_hard_reject"] is False
    assert res.diagnostics["report"]["best"]["expr_ast"] == ("var", 0)
    assert res.diagnostics["report"]["best_replaced_due_to_structural_reject"] is True
    assert res.probe_rms == pytest.approx(1.0e-6)


def test_typed_state_family_does_not_predeclare_sqrt():
    with pytest.raises(ValueError, match="Unsupported cheap factorized family"):
        factorized_search_de_mod._family_basis_asts(("var", 0), "sqrt")


def test_run_factorized_de_from_feature_groups_keeps_strict_dimensions_by_default(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_order_search(**kwargs):
        captured["var_dims"] = kwargs["var_dims"]
        captured["y_dims"] = kwargs["y_dims"]
        expr = ("sqrt", ("var", 0))
        return (
            [
                {
                    "order": int(kwargs["order"]),
                    "seed_search": 0,
                    "expr": "sqrt(u)",
                    "expr_ast": expr,
                    "mse": 0.0,
                    "score_raw": 0.0,
                    "score": 0.0,
                    "size": 2,
                    "mapping_complexity": 1,
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "mapping_kind": "poly",
                    "_expr_obj": expr,
                    "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0]},
                }
            ],
            1,
            1,
            1,
        )

    monkeypatch.setattr(oracle_lab_de_mod, "_run_order_search", _fake_run_order_search)

    spec = DELabSpec(
        id="strict_dims",
        csv_paths=(),
        order_candidates=(1,),
        include_x=False,
        include_u=True,
        u_col="u",
        dims=oracle_lab_de_mod.DimensionSpec(basis=("D",), x_dim=(1.0,), u_dim=(0.0,)),
    )
    hp = default_physics_rescue_hp(preset="fast")
    hp.n_fit = 16
    hp.n_probe = 16

    report = run_factorized_de_from_feature_groups(
        spec,
        [_sqrt_decay_group()],
        factorized_search_hp=hp,
        seed=3,
        dtype=torch.float64,
        enforce_dims=True,
        verbose=False,
    )

    assert captured["var_dims"] == [(0.0,)]
    assert captured["y_dims"] == (-1.0,)
    assert report["coefficient_dim_mode"] == "strict_expression"
    assert report["per_order"][0]["y_dims"] == [-1.0]
    assert report["best"]["probe_rms"] == pytest.approx(0.0)
    assert report["best"]["probe_rel_rms"] == pytest.approx(0.0)


def test_run_factorized_de_from_feature_groups_inferred_outer_relaxes_target_dim(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_order_search(**kwargs):
        captured["var_dims"] = kwargs["var_dims"]
        captured["y_dims"] = kwargs["y_dims"]
        expr = ("sqrt", ("var", 0))
        return (
            [
                {
                    "order": int(kwargs["order"]),
                    "seed_search": 0,
                    "expr": "sqrt(u)",
                    "expr_ast": expr,
                    "mse": 0.0,
                    "score_raw": 0.0,
                    "score": 0.0,
                    "size": 2,
                    "mapping_complexity": 1,
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "mapping_kind": "poly",
                    "_expr_obj": expr,
                    "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0]},
                }
            ],
            1,
            1,
            1,
        )

    monkeypatch.setattr(oracle_lab_de_mod, "_run_order_search", _fake_run_order_search)

    spec = DELabSpec(
        id="inferred_outer_dims",
        csv_paths=(),
        order_candidates=(1,),
        include_x=False,
        include_u=True,
        u_col="u",
        dims=oracle_lab_de_mod.DimensionSpec(basis=("D",), x_dim=(1.0,), u_dim=(0.0,)),
    )
    hp = default_physics_rescue_hp(preset="fast")
    hp.n_fit = 16
    hp.n_probe = 16

    report = run_factorized_de_from_feature_groups(
        spec,
        [_sqrt_decay_group()],
        factorized_search_hp=hp,
        seed=3,
        dtype=torch.float64,
        enforce_dims=True,
        verbose=False,
        coefficient_dim_mode="inferred_outer",
    )

    best = report["best"]
    assert captured["var_dims"] == [(0.0,)]
    assert captured["y_dims"] is None
    assert report["coefficient_dim_mode"] == "inferred_outer"
    assert report["per_order"][0]["target_y_dims"] == [-1.0]
    assert report["per_order"][0]["y_dims"] is None
    assert best["expr_dim"] == [0.0]
    assert best["target_dim"] == [-1.0]
    assert best["coefficient_dim"] == [-1.0]


def test_direct_residual_fss_uses_inferred_outer_dimensions(monkeypatch):
    calls: list[dict[str, object]] = []

    def _fake_run_factorized_de_from_feature_groups(spec, groups, **kwargs):
        hp_seen = kwargs["factorized_search_hp"]
        calls.append(
            {
                "coefficient_dim_mode": kwargs["coefficient_dim_mode"],
                "include_x": bool(spec.include_x),
                "include_u": bool(spec.include_u),
                "order_candidates": tuple(spec.order_candidates),
                "max_depth": int(hp_seen.max_depth),
                "brute_depth": int(hp_seen.brute_depth),
                "trigger_val_rms": float(getattr(hp_seen, "_de_early_stop_val_rms")),
                "trigger_rel_rms": float(getattr(hp_seen, "_de_early_stop_rel_rms")),
            }
        )
        expr = ("sqrt", ("var", 0))
        return {
            "x_axis": 0,
            "include_x": bool(spec.include_x),
            "include_u": bool(spec.include_u),
            "include_du": bool(spec.include_du),
            "constants_ordered": [],
            "trajectories": [{"id": "sqrt_decay"}],
            "fit_trajectories": [{"id": "sqrt_decay"}],
            "probe_trajectories": [{"id": "sqrt_decay"}],
            "factorized_de_diagnostics": {},
            "per_order": [{"order": 1, "feature_names": ["u"], "results": []}],
            "best": {
                "order": 1,
                "expr": "sqrt(u)",
                "expr_ast": expr,
                "mse": 0.0,
                "score_raw": 0.0,
                "score": 0.0,
                "size": 2,
                "mapping_complexity": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                "mapping_kind": "poly",
            },
        }

    monkeypatch.setattr(
        factorized_search_de_mod,
        "run_factorized_de_from_feature_groups",
        _fake_run_factorized_de_from_feature_groups,
    )

    hp = default_physics_rescue_hp(preset="fast")
    hp.n_fit = 16
    hp.n_probe = 16
    res = run_direct_residual_fss_from_feature_groups(
        [_sqrt_decay_group()],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedSearchDERescueConfig(hp=hp),
        dtype=torch.float64,
        verbose=False,
    )

    assert res is not None
    assert res.residual_ast is not None
    assert calls
    assert calls[0]["coefficient_dim_mode"] == "inferred_outer"
    assert calls[0]["include_x"] is False
    assert calls[0]["include_u"] is True
    assert calls[0]["order_candidates"] == (1,)
    assert calls[0]["max_depth"] == 3
    assert calls[0]["brute_depth"] == 3
    assert calls[0]["trigger_val_rms"] == pytest.approx(1.0e-3)
    assert calls[0]["trigger_rel_rms"] == pytest.approx(1.0e-3)
    assert res.diagnostics["direct_residual_fss"]["coefficient_dim_mode"] == "inferred_outer"


def test_order2_generator_witness_accepts_noisy_sho_carrier():
    groups = [
        _sho_group("ic0", phase=0.0),
        _sho_group("ic1", phase=0.4),
        _sho_group("ic2", phase=0.9),
    ]
    cfg = DESearchConfig(x_axis=0, order_candidates=(2,), include_x=False, include_u=True, include_du=True)
    spec = de_lab_spec_from_de_cfg(cfg)
    result = FactorizedSearchDEResult(
        order=2,
        x_axis=0,
        rhs_ast="rhs",
        residual_ast="resid",
        canonical_equation="u_xx - (a*u) = 0",
        probe_mse=0.13**2,
        probe_rms=0.13,
        expr_ast=("var", 0),
        mapping={"kind": "poly", "coeffs": [0.0, -9.0]},
        mapping_kind="poly",
        feature_names=["u", "du"],
        diagnostics={"probe_rel_rms": 0.03},
    )

    witness = validate_order2_generator_witness(
        result,
        groups,
        spec=spec,
        rescue_cfg=FactorizedSearchDERescueConfig(),
        dtype=torch.float64,
    )

    assert witness["generator_status"] in {"EXACT_STRUCTURAL_GENERATOR", "DYNAMICALLY_COMPATIBLE"}
    assert witness["generator_status_legacy"] in {"EXACT_GENERATOR", "VIABLE_WITH_MODELLING_ERROR"}
    assert witness["evidence_tier"] == "generator_witness"
    assert witness["theta_shared"]["a"] == pytest.approx(-9.0, rel=0.08)
    assert witness["theta_spread_rel"] < 0.2
    assert witness["local_rms_z"] < 2.0
    assert witness["rollout_ok"] is True
    assert witness["rollout_u_nrmse"] < 5.0e-2


def test_order2_generator_witness_rejects_inconsistent_trajectory_coefficients():
    groups = [
        _sho_group("ic0", phase=0.0, target_sign=-1.0),
        _sho_group("ic1", phase=0.4, target_sign=1.0),
    ]
    cfg = DESearchConfig(x_axis=0, order_candidates=(2,), include_x=False, include_u=True, include_du=True)
    spec = de_lab_spec_from_de_cfg(cfg)
    result = FactorizedSearchDEResult(
        order=2,
        x_axis=0,
        rhs_ast="rhs",
        residual_ast="resid",
        canonical_equation="u_xx - (a*u) = 0",
        probe_mse=0.13**2,
        probe_rms=0.13,
        expr_ast=("var", 0),
        mapping={"kind": "poly", "coeffs": [0.0, -9.0]},
        mapping_kind="poly",
        feature_names=["u", "du"],
        diagnostics={"probe_rel_rms": 0.03},
    )

    witness = validate_order2_generator_witness(
        result,
        groups,
        spec=spec,
        rescue_cfg=FactorizedSearchDERescueConfig(),
        dtype=torch.float64,
    )

    assert witness["generator_status"] in {"AMBIGUOUS_ROLE", "NOT_VIABLE"}
    assert witness["coeff_ok"] is False


def test_direct_residual_fss_exits_after_order2_generator_witness(monkeypatch):
    groups = [
        _sho_group("ic0", phase=0.0),
        _sho_group("ic1", phase=0.35),
        _sho_group("ic2", phase=0.8),
    ]
    calls: list[dict[str, object]] = []

    def _fake_run_factorized_de_from_feature_groups(spec, groups_seen, **kwargs):
        calls.append(
            {
                "include_x": bool(spec.include_x),
                "include_u": bool(spec.include_u),
                "include_du": bool(spec.include_du),
                "order_candidates": tuple(spec.order_candidates),
            }
        )
        return {
            "x_axis": 0,
            "include_x": bool(spec.include_x),
            "include_u": bool(spec.include_u),
            "include_du": bool(spec.include_du),
            "constants_ordered": [],
            "trajectories": [{"id": group.id} for group in groups],
            "fit_trajectories": [{"id": group.id} for group in groups],
            "probe_trajectories": [{"id": group.id} for group in groups],
            "factorized_de_diagnostics": {
                "orders": [
                    {
                        "order": 2,
                        "target_scale": 4.7,
                        "effective_early_stop_mse": 1.0e-8,
                    }
                ]
            },
            "per_order": [{"order": 2, "feature_names": ["u", "du"], "results": []}],
            "best": {
                "order": 2,
                "expr": "u",
                "expr_ast": ("var", 0),
                "mse": 0.13**2,
                "score_raw": 0.13**2,
                "score": 0.13**2,
                "size": 1,
                "mapping_complexity": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, -9.0]},
                "mapping_kind": "poly",
            },
        }

    monkeypatch.setattr(
        factorized_search_de_mod,
        "run_factorized_de_from_feature_groups",
        _fake_run_factorized_de_from_feature_groups,
    )

    hp = default_physics_rescue_hp(preset="fast")
    hp.n_fit = 64
    hp.n_probe = 64
    res = run_direct_residual_fss_from_feature_groups(
        groups,
        cfg=DESearchConfig(x_axis=0, order_candidates=(2,), include_x=True, include_u=True, include_du=True),
        rescue_cfg=FactorizedSearchDERescueConfig(hp=hp, trigger_val_rms=1.0e-8),
        dtype=torch.float64,
        verbose=False,
    )

    assert res is not None
    assert len(calls) == 1
    assert calls[0]["include_x"] is False
    assert res.diagnostics["generator_status"] == "EXACT_STRUCTURAL_GENERATOR"
    assert res.diagnostics["early_exit_reason"] == "generator_witness_pass"
    assert res.diagnostics["direct_residual_fss"]["early_exit_reason"] == "generator_witness_pass"
    assert res.mapping["coeffs"][1] == pytest.approx(-9.0, rel=0.08)


def test_direct_residual_fss_materializes_dynamic_generator_witness(monkeypatch):
    groups = [_sho_group("ic0", phase=0.0), _sho_group("ic1", phase=0.4)]
    calls: list[dict[str, object]] = []

    def _fake_run_factorized_de_from_feature_groups(spec, groups_seen, **kwargs):
        calls.append({"include_x": bool(spec.include_x), "include_du": bool(spec.include_du)})
        return {
            "x_axis": 0,
            "include_x": bool(spec.include_x),
            "include_u": bool(spec.include_u),
            "include_du": bool(spec.include_du),
            "constants_ordered": [],
            "trajectories": [{"id": group.id} for group in groups],
            "fit_trajectories": [{"id": group.id} for group in groups],
            "probe_trajectories": [{"id": group.id} for group in groups],
            "factorized_de_diagnostics": {
                "orders": [{"order": 2, "target_scale": 4.7, "effective_early_stop_mse": 1.0e-8}]
            },
            "per_order": [{"order": 2, "feature_names": ["u", "du"], "results": []}],
            "best": {
                "order": 2,
                "expr": "u",
                "expr_ast": ("var", 0),
                "mse": 0.12**2,
                "score_raw": 0.12**2,
                "score": 0.12**2,
                "size": 1,
                "mapping_complexity": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, -8.0]},
                "mapping_kind": "poly",
            },
        }

    def _fake_validate_order2_generator_witness(result, groups_seen, *, spec, rescue_cfg, dtype):
        return {
            "enabled": True,
            "evidence_tier": "generator_witness",
            "generator_status": "DYNAMICALLY_COMPATIBLE",
            "rollout_u_nrmse": 0.014,
            "witness_candidate": {
                "engine": "factorized_search",
                "kind": "factorized",
                "order": 2,
                "x_axis": 0,
                "include_x": bool(spec.include_x),
                "include_u": bool(spec.include_u),
                "include_du": bool(spec.include_du),
                "constants_ordered": [],
                "feature_names": ["u", "du"],
                "expr_ast": ("var", 0),
                "mapping": {"kind": "poly", "coeffs": [0.1, -8.9], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        }

    monkeypatch.setattr(
        factorized_search_de_mod,
        "run_factorized_de_from_feature_groups",
        _fake_run_factorized_de_from_feature_groups,
    )
    monkeypatch.setattr(
        factorized_search_de_mod,
        "validate_order2_generator_witness",
        _fake_validate_order2_generator_witness,
    )

    hp = default_physics_rescue_hp(preset="fast")
    hp.n_fit = 64
    hp.n_probe = 64
    res = run_direct_residual_fss_from_feature_groups(
        groups,
        cfg=DESearchConfig(x_axis=0, order_candidates=(2,), include_x=True, include_u=True, include_du=True),
        rescue_cfg=FactorizedSearchDERescueConfig(hp=hp, trigger_val_rms=1.0e-8),
        dtype=torch.float64,
        verbose=False,
        attempt_phase="autonomous",
    )

    assert res is not None
    assert len(calls) == 1
    assert res.diagnostics["generator_status"] == "DYNAMICALLY_COMPATIBLE"
    assert res.diagnostics["witness_materialized"] is True
    assert "early_exit_reason" not in res.diagnostics
    assert res.diagnostics["direct_residual_fss"]["early_exit_reason"] is None
    assert res.mapping["coeffs"] == pytest.approx([0.1, -8.9])
    shortlist = res.diagnostics["shortlist_union"]
    assert shortlist[0]["source_lane"] == "direct_residual_fss"
    assert shortlist[0]["mapping"]["coeffs"] == pytest.approx([0.1, -8.9])


def test_score_expr_finite_mask_keeps_sqrt_with_tiny_domain_leakage():
    x = torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(-1, 1)
    x[0, 0] = -1.0e-12
    y = torch.sqrt(torch.clamp(x, min=0.0))
    proj = torch.randn((100, 8), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    expr = ("sqrt", ("var", 0))

    strict = score_expr(
        expr,
        x,
        y,
        x,
        y,
        proj,
        "quant",
        10.0,
        15,
        1,
        refine_cfg={"score_mapping_family_mode": "poly_only"},
    )
    masked = score_expr(
        expr,
        x,
        y,
        x,
        y,
        proj,
        "quant",
        10.0,
        15,
        1,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "score_finite_mask_enable": True,
            "score_finite_mask_min_fit_frac": 0.98,
            "score_finite_mask_min_probe_frac": 0.98,
            "score_finite_mask_min_points": 8,
        },
    )

    assert strict is None
    assert masked is not None
    assert masked[0] == pytest.approx(0.0, abs=1.0e-20)
    diag = masked[3]["_finite_mask"]
    assert diag["fit_valid_frac"] == pytest.approx(0.99)
    assert diag["probe_valid_frac"] == pytest.approx(0.99)


def test_score_expr_finite_mask_rejects_broad_sqrt_domain_violation():
    x = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64).reshape(-1, 1)
    y = torch.sqrt(torch.clamp(x, min=0.0))
    proj = torch.randn((100, 8), dtype=torch.float64, generator=torch.Generator().manual_seed(1))

    masked = score_expr(
        ("sqrt", ("var", 0)),
        x,
        y,
        x,
        y,
        proj,
        "quant",
        10.0,
        15,
        1,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "score_finite_mask_enable": True,
            "score_finite_mask_min_fit_frac": 0.98,
            "score_finite_mask_min_probe_frac": 0.98,
            "score_finite_mask_min_points": 8,
        },
    )

    assert masked is None


def test_score_expr_domain_projection_keeps_boundary_sqrt_plateau():
    positive = torch.linspace(1.0, 0.0, 30, dtype=torch.float64)
    boundary = torch.full((70,), -7.5e-10, dtype=torch.float64)
    x = torch.cat([positive, boundary]).reshape(-1, 1)
    y = torch.sqrt(torch.clamp(x, min=0.0))
    proj = torch.randn((100, 8), dtype=torch.float64, generator=torch.Generator().manual_seed(2))

    strict = score_expr(
        ("sqrt", ("var", 0)),
        x,
        y,
        x,
        y,
        proj,
        "quant",
        10.0,
        15,
        1,
        refine_cfg={"score_mapping_family_mode": "poly_only"},
    )
    projected = score_expr(
        ("sqrt", ("var", 0)),
        x,
        y,
        x,
        y,
        proj,
        "quant",
        10.0,
        15,
        1,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "score_domain_projection_enable": True,
            "score_domain_projection_abs_tol": 1.0e-8,
            "score_domain_projection_rel_tol": 1.0e-4,
            "score_domain_projection_max_frac": 1.0,
        },
    )

    assert strict is None
    assert projected is not None
    assert projected[0] == pytest.approx(0.0, abs=1.0e-20)
    diag = projected[3]["_domain_projection"]
    assert diag["status"] == "projected_within_tube"
    assert diag["projected_frac"] == pytest.approx(0.70)
    assert diag["max_violation"] == pytest.approx(7.5e-10)


def test_score_expr_domain_projection_rejects_large_sqrt_violation():
    x = torch.cat(
        [
            torch.linspace(1.0, 0.0, 30, dtype=torch.float64),
            torch.full((70,), -1.0e-2, dtype=torch.float64),
        ]
    ).reshape(-1, 1)
    y = torch.sqrt(torch.clamp(x, min=0.0))
    proj = torch.randn((100, 8), dtype=torch.float64, generator=torch.Generator().manual_seed(3))

    projected = score_expr(
        ("sqrt", ("var", 0)),
        x,
        y,
        x,
        y,
        proj,
        "quant",
        10.0,
        15,
        1,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "score_domain_projection_enable": True,
            "score_domain_projection_abs_tol": 1.0e-8,
            "score_domain_projection_rel_tol": 1.0e-4,
            "score_domain_projection_max_frac": 1.0,
        },
    )

    assert projected is None


def test_domain_projection_reference_scale_keeps_scalar_rollout_tube():
    x = torch.tensor([[-1.0e-5]], dtype=torch.float64)
    base_cfg = {
        "score_domain_projection_enable": True,
        "score_domain_projection_abs_tol": 1.0e-8,
        "score_domain_projection_rel_tol": 1.0e-2,
        "score_domain_projection_max_frac": 1.0,
    }

    _, tiny_scale_diag = eval_node_with_domain_projection(("sqrt", ("var", 0)), x, base_cfg)
    _, ref_scale_diag = eval_node_with_domain_projection(
        ("sqrt", ("var", 0)),
        x,
        {**base_cfg, "score_domain_projection_reference_scale": 1.0},
    )

    assert not domain_projection_is_acceptable(tiny_scale_diag)
    assert domain_projection_is_acceptable(ref_scale_diag)
    assert ref_scale_diag["ops"][0]["argument_scale"] == pytest.approx(1.0)


def test_direct_residual_fss_recovers_sqrt_atom_on_synthetic_decay():
    hp = default_physics_rescue_hp(preset="fast")
    hp.n_iter = 1500
    hp.n_fit = 64
    hp.n_probe = 64
    hp.return_topk = 8
    hp.max_depth = 3

    res = run_direct_residual_fss_from_feature_groups(
        [_sqrt_decay_group()],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedSearchDERescueConfig(hp=hp, trigger_val_rms=1.0e-8),
        dtype=torch.float64,
        verbose=False,
    )

    assert res is not None
    expr_ast = res.expr_ast
    if isinstance(expr_ast, tuple) and len(expr_ast) == 3 and expr_ast[0] == "mul":
        assert expr_ast[1][0] == "const"
        assert abs(abs(float(expr_ast[1][1])) - 1.0) < 1.0e-8
        expr_ast = expr_ast[2]
    assert expr_ast == ("sqrt", ("var", 0))
    assert res.probe_rms < 1.0e-8
    assert res.diagnostics["direct_residual_fss"]["coefficient_dim_mode"] == "inferred_outer"


def test_regularized_implicit_residual_recovers_de010_gauge():
    hp = default_physics_rescue_hp(preset="fast")
    hp.n_fit = 256
    hp.n_probe = 256

    res = run_regularized_implicit_residual_fss_from_feature_groups(
        [_de010_like_group()],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedSearchDERescueConfig(
            hp=hp,
            regularized_implicit_enable=True,
            regularized_implicit_max_b_terms=1,
        ),
        dtype=torch.float64,
        verbose=False,
    )

    assert res is not None
    implicit = res.diagnostics["implicit_residual"]
    assert implicit["a_expr"] == "x0"
    assert implicit["b_exprs"] == ["u"]
    assert implicit["b_coeff_source"] == "separable_invariant_refit"
    assert implicit["derivative_b_coeffs"][0] == pytest.approx(0.52, rel=1.0e-2)
    assert implicit["b_coeffs"][0] == pytest.approx(1.0, rel=2.0e-2)
    assert implicit["normalized_probe_score"] < 1.0e-8
    assert res.probe_rms < 1.0e-4
    assert "u_x0" in res.canonical_equation
    assert res.diagnostics["regularized_implicit_residual"]["selected"]["a_expr"] == "x0"


def test_direct_residual_fss_keeps_projected_sqrt_ahead_of_worse_linear_rows():
    hp = default_physics_rescue_hp(preset="fast")
    hp.n_iter = 1500
    hp.n_fit = 128
    hp.n_probe = 128
    hp.return_topk = 8
    hp.max_depth = 3

    groups = [
        _sqrt_decay_group_with_boundary_dent("traj0", shift=0.0),
        _sqrt_decay_group_with_boundary_dent("traj1", shift=0.15),
        _sqrt_decay_group_with_boundary_dent("traj2", shift=0.30),
        _sqrt_decay_group_with_boundary_dent("traj3", shift=0.45),
    ]
    res = run_direct_residual_fss_from_feature_groups(
        groups,
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedSearchDERescueConfig(hp=hp, trigger_val_rms=1.0e-8),
        dtype=torch.float64,
        verbose=False,
    )

    assert res is not None
    assert res.expr_ast == ("sqrt", ("var", 0))
    assert res.probe_rms < 1.0e-5


def test_fit_best_poly_only_skips_expensive_mapping_families():
    z = torch.linspace(0.0, 1.0, 128, dtype=torch.float64).reshape(-1, 1)
    y = torch.sin(2.0 * z)

    out = fit_best(z, y, poly_degree=1, family_mode="poly_only")

    assert out is not None
    assert out[1]["kind"] == "poly"


def test_default_physics_rescue_hp_sets_shared_rhs_defaults():
    hp = default_physics_rescue_hp(preset="fast")

    assert hp.refine_enable is True
    assert hp.refine_profile == "rare_slate"
    assert hp.refine_mode == "slate"
    assert hp.refine_during_brute is False
    assert hp.refine_during_mutation is False
    assert hp.refine_during_slate is True
    assert hp.refine_final_polish is True
    assert hp.refine_max_trials == 50
    assert hp.refine_max_variants == 1
    assert hp.refine_max_params == 1
    assert hp.refine_num_restarts == 1
    assert hp.refine_optimizer == "lbfgs"
    assert hp.refine_lbfgs_escalate_improve_factor == 2.0
    assert hp.refine_lbfgs_steps == 4
    assert hp.refine_fit_subset == 64
    assert hp.refine_joint_score_enable is False
    assert hp.refine_joint_enable is False
    assert hp.refine_joint_terms_enable is False
    assert hp.refine_fit_subset_mode == "stratified"
    assert int(hp.poly_degree) == 1
    assert int(hp.brute_max_expressions) == 50_000
    assert float(hp.complexity_penalty) == pytest.approx(1.0e-4)
    assert float(hp.mapping_complexity_penalty) == pytest.approx(0.01)
    assert hp.score_domain_projection_enable is True
    assert float(hp.score_domain_projection_abs_tol) == pytest.approx(1.0e-8)
    assert float(hp.score_domain_projection_rel_tol) == pytest.approx(1.0e-2)
    assert float(hp.score_domain_projection_max_frac) == pytest.approx(1.0)
    assert int(hp.n_iter) == 15_000
    assert int(hp.n_fit) == 2_000
    assert int(hp.n_probe) == 2_000
    assert int(hp.return_topk) == 16


def test_shared_factorized_search_candidate_eval_helpers_support_row_and_batch():
    candidate = {
        "expr_ast": ["var", 1],
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
    }

    predictor = factorized_search_candidate_to_feature_predictor(candidate)
    assert predictor([3.0, 2.0]) == pytest.approx(2.0)

    vals = evaluate_factorized_search_candidate(
        candidate,
        [[1.0, 4.0], [2.0, 5.0]],
        dtype=torch.float64,
    )
    assert vals.tolist() == pytest.approx([4.0, 5.0])


def test_factorized_search_candidate_eval_uses_projection_tube_for_sqrt_boundary():
    strict_candidate = {
        "expr_ast": ["sqrt", ["var", 0]],
        "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
    }
    projected_candidate = {
        **strict_candidate,
        "domain_projection": {
            "enabled": True,
            "abs_tol": 1.0e-8,
            "rel_tol": 1.0e-2,
            "max_frac": 1.0,
        },
    }

    with pytest.raises(FloatingPointError):
        evaluate_factorized_search_candidate(strict_candidate, [[-1.0e-10]], dtype=torch.float64)

    vals = evaluate_factorized_search_candidate(
        projected_candidate,
        [[-1.0e-10], [4.0]],
        dtype=torch.float64,
    )

    assert vals.tolist() == pytest.approx([-0.0, -2.0])


def test_factorized_search_rhs_callable_rollout_uses_projection_tube():
    candidate = {
        "order": 1,
        "include_x": False,
        "include_u": True,
        "feature_names": ["u"],
        "expr_ast": ["sqrt", ["var", 0]],
        "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
        "domain_projection": {
            "enabled": True,
            "abs_tol": 1.0e-8,
            "rel_tol": 1.0e-2,
            "max_frac": 1.0,
        },
    }

    order, rhs_fn = factorized_search_report_to_rhs_callable(candidate)

    assert order == 1
    assert rhs_fn(0.0, [-1.0e-10]) == pytest.approx([-0.0])
    assert rhs_fn(0.0, [4.0]) == pytest.approx([-2.0])


def test_factorized_search_rerank_prefers_score_before_size():
    small_bad = {
        "score": 1.0e-2,
        "size": 1,
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
    }
    larger_good = {
        "score": 1.0e-7,
        "size": 2,
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
    }

    assert factorized_search_de_mod._row_rerank_key(larger_good) < factorized_search_de_mod._row_rerank_key(small_bad)


def test_factorized_search_rerank_prefers_size_within_score_decade():
    simple_same_decade = {
        "score": 8.0e-7,
        "size": 1,
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
    }
    complex_same_decade = {
        "score": 1.0e-7,
        "size": 12,
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
    }

    assert (
        factorized_search_de_mod._row_rerank_key(simple_same_decade)
        < factorized_search_de_mod._row_rerank_key(complex_same_decade)
    )


def test_oracle_de_export_sort_prefers_size_within_score_decade():
    rows = [
        {"score": 1.0e-7, "size": 12, "original_rank": 0},
        {"score": 8.0e-7, "size": 1, "original_rank": 1},
    ]

    ordered = sorted(rows, key=oracle_lab_de_mod._de_complexity_order_key)

    assert ordered[0]["size"] == 1


def test_oracle_de_export_sort_prefers_validation_decade_before_size():
    rows = [
        {"mse": 1.0e-12, "score": 2.0e-2, "symbolic_size_simplified": 20, "original_rank": 0},
        {"mse": 1.0e-4, "score": 2.1e-2, "symbolic_size_simplified": 3, "original_rank": 1},
    ]

    ordered = sorted(rows, key=oracle_lab_de_mod._de_complexity_order_key)

    assert ordered[0]["mse"] == pytest.approx(1.0e-12)


def test_factorized_search_rerank_puts_structural_rejects_after_safe_rows():
    unsafe_better_score = {
        "score": 1.0e-12,
        "size": 1,
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
        "structural_ok": False,
        "structural_hard_reject": True,
        "structural_reasons": ["log_nonpositive_constant"],
    }
    safe_worse_score = {
        "score": 1.0e-6,
        "size": 2,
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
        "structural_ok": True,
        "structural_hard_reject": False,
        "structural_reasons": [],
    }

    assert (
        factorized_search_de_mod._row_rerank_key(safe_worse_score)
        < factorized_search_de_mod._row_rerank_key(unsafe_better_score)
    )


def test_factorized_search_candidate_compiler_prefers_expr_ast_over_display_expr():
    candidate = {
        "expr": "sin(x0)",
        "expr_ast": ["var", 1],
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
    }

    predictor = factorized_search_candidate_to_feature_predictor(candidate)

    assert predictor([3.0, 2.0]) == pytest.approx(2.0)


def test_normalized_rmse_matches_expected_ratio():
    score = normalized_rmse([1.0, 2.0], [1.0, 1.0])
    expected = math.sqrt(1.0) / math.sqrt(5.0)
    assert score == pytest.approx(expected)


def test_run_factorized_de_from_feature_groups_preserves_group_ids(monkeypatch):
    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    spec = DELabSpec(
        id="grouped_api",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
    )
    hp = default_physics_rescue_hp(preset="fast")
    groups = [
        _make_feature_group("traj0", decay=0.5),
        _make_feature_group("traj1", decay=0.5),
    ]

    report = run_factorized_de_from_feature_groups(
        spec,
        groups,
        factorized_search_hp=hp,
        seed=11,
        dtype=torch.float64,
        verbose=False,
    )

    assert report["best"] is not None
    assert report["split_mode"] == "prebuilt_feature_groups"
    assert [row["id"] for row in report["trajectories"]] == ["traj0", "traj1"]
    assert report["per_order"][0]["fit_traj_ids"] == ["traj0", "traj1"]
    assert report["per_order"][0]["probe_traj_ids"] == ["traj0", "traj1"]
    assert int(report["per_order"][0]["n_traj_total"]) == 2
    assert report["per_order"][0]["feature_names"] == ["x0", "u"]
    assert float(report["best"]["mse"]) < 1.0e-10
    diag = report["factorized_de_diagnostics"]
    assert diag["mode"] == "broad_whole_rhs"
    assert diag["attempts"] == 1
    assert diag["n_groups_total"] == 2
    assert diag["effective_fit_rows"] == report["per_order"][0]["n_points_fit"]
    assert diag["effective_probe_rows"] == report["per_order"][0]["n_points_probe"]
    assert diag["budget_scope"] == "per_group"
    assert [
        (row["id"], row["original_rows"], row["rows"], row["target_rows"], row["budget"])
        for row in diag["orders"][0]["fit_row_counts"]
    ] == [
        ("traj0", 16, 16, 16, None),
        ("traj1", 16, 16, 16, None),
    ]
    assert diag["orders"][0]["n_results"] >= 1
    assert diag["orders"][0]["search_wall_seconds"] >= 0.0
    json.dumps(diag)

    result = factorized_search_report_to_de_result(report)
    assert result.diagnostics["factorized_de_diagnostics"]["effective_total_rows"] == diag["effective_total_rows"]


def test_run_factorized_de_from_feature_groups_supports_global_budget_scope(monkeypatch):
    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    spec = DELabSpec(
        id="grouped_api_global_budget",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
    )
    hp = default_physics_rescue_hp(preset="fast")
    hp.n_fit = 10
    hp.n_probe = 8
    groups = [
        _make_feature_group("traj0", decay=0.5),
        _make_feature_group("traj1", decay=0.5),
    ]

    report = run_factorized_de_from_feature_groups(
        spec,
        groups,
        factorized_search_hp=hp,
        seed=19,
        dtype=torch.float64,
        verbose=False,
        budget_scope="global",
    )

    diag = report["factorized_de_diagnostics"]
    order_diag = diag["orders"][0]
    assert diag["budget_scope"] == "global"
    assert report["budget_scope"] == "global"
    assert report["per_order"][0]["n_points_fit"] == 10
    assert report["per_order"][0]["n_points_probe"] == 8
    assert diag["effective_fit_rows"] == 10
    assert diag["effective_probe_rows"] == 8
    assert [(row["id"], row["original_rows"], row["rows"], row["budget"]) for row in order_diag["fit_row_counts"]] == [
        ("traj0", 16, 5, 5),
        ("traj1", 16, 5, 5),
    ]
    assert [(row["id"], row["original_rows"], row["rows"], row["budget"]) for row in order_diag["probe_row_counts"]] == [
        ("traj0", 16, 4, 4),
        ("traj1", 16, 4, 4),
    ]
    assert report["best"] is not None


def test_run_factorized_de_from_feature_groups_supports_integrate_validation(monkeypatch):
    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    spec = DELabSpec(
        id="grouped_api_validate",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
        validate_integrate_topk=1,
    )
    hp = default_physics_rescue_hp(preset="fast")
    report = run_factorized_de_from_feature_groups(
        spec,
        [_make_feature_group("traj0", decay=0.5)],
        factorized_search_hp=hp,
        seed=13,
        dtype=torch.float64,
        verbose=False,
    )

    row = report["per_order"][0]["results"][0]
    assert row["integrate_ok"] is True
    assert float(row["integrate_mse"]) < 1.0e-6


def test_run_factorized_de_from_feature_groups_respects_fit_probe_roles(monkeypatch):
    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    spec = DELabSpec(
        id="grouped_api_roles",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
    )
    hp = default_physics_rescue_hp(preset="fast")
    report = run_factorized_de_from_feature_groups(
        spec,
        [
            _make_feature_group("fit0", decay=0.5, use_for_fit=True, use_for_probe=False),
            _make_feature_group("probe0", decay=0.5, use_for_fit=False, use_for_probe=True),
        ],
        factorized_search_hp=hp,
        seed=17,
        dtype=torch.float64,
        verbose=False,
    )

    assert [row["id"] for row in report["trajectories"]] == ["fit0", "probe0"]
    assert [row["id"] for row in report["fit_trajectories"]] == ["fit0"]
    assert [row["id"] for row in report["probe_trajectories"]] == ["probe0"]
    assert report["probe_fallback_to_fit"] is False
    assert report["per_order"][0]["fit_traj_ids"] == ["fit0"]
    assert report["per_order"][0]["probe_traj_ids"] == ["probe0"]
    assert int(report["per_order"][0]["n_traj_fit"]) == 1
    assert int(report["per_order"][0]["n_traj_probe"]) == 1


def test_run_factorized_de_from_feature_groups_uses_probe_roles_for_integration(monkeypatch):
    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())
    seen: dict[str, list[str]] = {}

    orig_validate = oracle_lab_de_mod._validate_candidate_by_integration

    def _wrapped_validate(*args, trajectories, **kwargs):
        seen["traj_ids"] = [str(getattr(tr, "traj_id", "?")) for tr in trajectories]
        return orig_validate(*args, trajectories=trajectories, **kwargs)

    monkeypatch.setattr(oracle_lab_de_mod, "_validate_candidate_by_integration", _wrapped_validate)

    spec = DELabSpec(
        id="grouped_api_roles_validate",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
        validate_integrate_topk=1,
    )
    hp = default_physics_rescue_hp(preset="fast")
    run_factorized_de_from_feature_groups(
        spec,
        [
            _make_feature_group("fit0", decay=0.5, use_for_fit=True, use_for_probe=False),
            _make_feature_group("probe0", decay=0.5, use_for_fit=False, use_for_probe=True),
        ],
        factorized_search_hp=hp,
        seed=18,
        dtype=torch.float64,
        verbose=False,
    )

    assert seen["traj_ids"] == ["probe0"]


def test_run_factorized_de_from_feature_groups_reranks_within_order_by_integration(monkeypatch):
    def _fake_order_search(**kwargs):
        order_i = int(kwargs["order"])
        rows = [
            {
                "order": order_i,
                "score": 1.0e-8,
                "size": 9,
                "expr_ast": ("bad",),
                "_expr_obj": ("bad",),
                "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
            },
            {
                "order": order_i,
                "score": 1.0e-4,
                "size": 2,
                "expr_ast": ("good",),
                "_expr_obj": ("good",),
                "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
            },
        ]
        return rows, 1, 1, 10

    def _fake_integrate(*, expr_ast, **kwargs):
        if expr_ast == ("bad",):
            return float("inf")
        return 1.0e-4

    monkeypatch.setattr(oracle_lab_de_mod, "_run_order_search", _fake_order_search)
    monkeypatch.setattr(oracle_lab_de_mod, "_validate_candidate_by_integration", _fake_integrate)
    monkeypatch.setattr(
        factorized_search_de_mod,
        "_score_candidate_domain_fragility",
        lambda *args, **kwargs: {"domain_ok": True, "domain_fragility_penalty": 0.0},
    )

    spec = DELabSpec(
        id="grouped_api_rerank",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
        validate_integrate_topk=2,
    )
    hp = default_physics_rescue_hp(preset="fast")
    report = run_factorized_de_from_feature_groups(
        spec,
        [_make_feature_group("traj0", decay=0.5)],
        factorized_search_hp=hp,
        seed=19,
        dtype=torch.float64,
        verbose=False,
    )

    rows = report["per_order"][0]["results"]
    assert rows[0]["expr_ast"] == ("good",)
    assert rows[0]["original_rank"] == 1
    assert rows[0]["rerank_rank"] == 0
    assert rows[1]["expr_ast"] == ("bad",)
    assert rows[1]["original_rank"] == 0
    assert rows[1]["rerank_rank"] == 1
    assert report["per_order"][0]["best"]["expr_ast"] == ("good",)
    assert report["per_order"][0]["best_selection_mode"] == "integrate_rerank"


def test_run_factorized_de_from_feature_groups_uses_integration_for_global_best(monkeypatch):
    def _fake_order_search(**kwargs):
        order_i = int(kwargs["order"])
        if order_i == 1:
            rows = [
                {
                    "order": 1,
                    "score": 1.0e-2,
                    "size": 2,
                    "expr_ast": ("ord1_good",),
                    "_expr_obj": ("ord1_good",),
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                    "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                }
            ]
        else:
            rows = [
                {
                    "order": 2,
                    "score": 1.0e-8,
                    "size": 1,
                    "expr_ast": ("ord2_bad",),
                    "_expr_obj": ("ord2_bad",),
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                    "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                }
            ]
        return rows, 1, 1, 10

    def _fake_integrate(*, expr_ast, **kwargs):
        if expr_ast == ("ord2_bad",):
            return float("inf")
        return 1.0e-3

    monkeypatch.setattr(oracle_lab_de_mod, "_run_order_search", _fake_order_search)
    monkeypatch.setattr(oracle_lab_de_mod, "_validate_candidate_by_integration", _fake_integrate)
    monkeypatch.setattr(
        factorized_search_de_mod,
        "_score_candidate_domain_fragility",
        lambda *args, **kwargs: {"domain_ok": True, "domain_fragility_penalty": 0.0},
    )

    spec = DELabSpec(
        id="grouped_api_global_rerank",
        csv_paths=(),
        order_candidates=(1, 2),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
        validate_integrate_topk=1,
    )
    hp = default_physics_rescue_hp(preset="fast")
    report = run_factorized_de_from_feature_groups(
        spec,
        [_make_feature_group("traj0", decay=0.5)],
        factorized_search_hp=hp,
        seed=23,
        dtype=torch.float64,
        verbose=False,
    )

    assert report["best"]["expr_ast"] == ("ord1_good",)
    assert report["best"]["order"] == 1


def test_run_factorized_de_from_feature_groups_prefers_domain_stable_candidates(monkeypatch):
    def _fake_order_search(**kwargs):
        order_i = int(kwargs["order"])
        rows = [
            {
                "order": order_i,
                "score": 1.0e-8,
                "size": 1,
                "expr_ast": ("fragile",),
                "_expr_obj": ("fragile",),
                "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
            },
            {
                "order": order_i,
                "score": 1.0e-4,
                "size": 2,
                "expr_ast": ("stable",),
                "_expr_obj": ("stable",),
                "mapping": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
                "_mapping_obj": {"kind": "poly", "coeffs": [0.0, -1.0], "mu": 0.0, "std": 1.0},
            },
        ]
        return rows, 1, 1, 10

    def _fake_domain(expr_ast, mapping, **kwargs):
        if expr_ast == ("fragile",):
            return {
                "domain_ok": False,
                "domain_failure_reason": "nonfinite_perturbed",
                "domain_fragility_penalty": float("inf"),
            }
        return {
            "domain_ok": True,
            "domain_failure_reason": None,
            "domain_fragility_penalty": 0.0,
        }

    monkeypatch.setattr(oracle_lab_de_mod, "_run_order_search", _fake_order_search)
    monkeypatch.setattr(
        factorized_search_de_mod,
        "_score_candidate_domain_fragility",
        _fake_domain,
    )

    spec = DELabSpec(
        id="grouped_api_domain_prefilter",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
        validate_integrate_topk=0,
    )
    hp = default_physics_rescue_hp(preset="fast")
    report = run_factorized_de_from_feature_groups(
        spec,
        [_make_feature_group("traj0", decay=0.5)],
        factorized_search_hp=hp,
        seed=29,
        dtype=torch.float64,
        verbose=False,
    )

    rows = report["per_order"][0]["results"]
    assert rows[0]["expr_ast"] == ("stable",)
    assert rows[0]["domain_ok"] is True
    assert rows[1]["expr_ast"] == ("fragile",)
    assert rows[1]["domain_ok"] is False
    assert rows[1]["domain_failure_reason"] == "nonfinite_perturbed"
    assert report["per_order"][0]["best"]["expr_ast"] == ("stable",)


def test_run_factorized_search_de_from_surrogate_returns_public_result(monkeypatch):
    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    x = torch.linspace(0.1, 2.0, 64, dtype=torch.float64).reshape(-1, 1)
    y = torch.zeros_like(x)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=16,
        shuffle=False,
    )
    hp = default_physics_rescue_hp(preset="fast")
    hp.seed = 7

    result = run_factorized_search_de_from_surrogate(
        _ExpDecaySurrogate(),
        loader,
        loader,
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True),
        rescue_cfg=FactorizedSearchDERescueConfig(hp=hp),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert isinstance(result, FactorizedSearchDEResult)
    assert int(result.order) == 1
    assert result.mapping_kind == "poly"
    assert result.feature_names == ["x0", "u"]
    assert result.rhs_ast is not None
    assert result.residual_ast is not None
    assert math.isfinite(float(result.probe_rms))
    assert "dataset0" in result.diagnostics["fit_traj_ids"]


def test_run_factorized_search_de_from_surrogates_preserves_dataset_ids(monkeypatch):
    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    x = torch.linspace(0.1, 2.0, 64, dtype=torch.float64).reshape(-1, 1)
    y = torch.zeros_like(x)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=16,
        shuffle=False,
    )
    hp = default_physics_rescue_hp(preset="fast")
    hp.seed = 3

    result = run_factorized_search_de_from_surrogates(
        [_ExpDecaySurrogate(), _ExpDecaySurrogate()],
        [loader, loader],
        [loader, loader],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True),
        rescue_cfg=FactorizedSearchDERescueConfig(hp=hp),
        device=torch.device("cpu"),
        dataset_ids=["ic0", "ic1"],
        dtype=torch.float64,
    )

    assert isinstance(result, FactorizedSearchDEResult)
    assert result.diagnostics["fit_traj_ids"] == ["ic0", "ic1"]
    assert result.diagnostics["probe_traj_ids"] == ["ic0", "ic1"]
    assert int(result.diagnostics["n_traj_total"]) == 2
