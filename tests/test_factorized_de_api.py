# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from nestynet_sr.sr_core.bridges import Abs, Acos, Add, Asin, Atan, ConstNode, Exp, Log, Mul, Pow, Sin, U, Var
from nestynet_sr.sr_de import DEFeatureGroup, DESearchConfig, DESearchResult
from nestynet_sr.sr_de.factorized_de import (
    FactorizedDERescueConfig,
    _active_first_order_typed_lanes,
    _candidate_preferred,
    _carrier_pool,
    _choose_preferred_zero_lane,
    _coord_pool,
    _distill_state_lane_explorer_candidate,
    _distill_x_lane_explorer_candidate,
    _eval_ast_on_features,
    _eval_univariate_ast_on_values,
    _fit_family_lane_candidate,
    _fit_original_scale_affine_explorer_head,
    _family_basis_asts,
    _family_design_from_coord_values,
    _lane_allows_linear_projection_variants,
    _lane_consistency_stats,
    _lane_witness_stats,
    _masked_original_scale_probe_mse,
    _nestynet_static_const_value,
    _pooled_same_coord_coeff_target,
    _record_explorer_launch_diagnostics,
    _residual_ratio_collapse_diagnostics,
    _select_lane_representative,
    _select_state_lane_candidates,
    _select_x_lane_candidates,
    _typed_explorer_caps_for_order,
    _typed_explorer_caps_from_hp,
    run_factorized_coeff_rescue_from_feature_groups,
)
from nestynet_sr.sr_search.factorized_search.oracle_lab_de import DEFeatureTensors


def test_record_explorer_launch_diagnostics_includes_process_memory():
    diag: dict[str, object] = {}

    _record_explorer_launch_diagnostics(
        diag,
        lane="second_order_state_nonlinearity",
        base_mode="zero",
        order=2,
        carrier_ast=ConstNode(1.0),
        coord_ast=U(),
        fit_rows_full=24000,
        probe_rows_full=24000,
        fit_rows_search=2048,
        probe_rows_search=4096,
        seed=0,
        sample_seed=123,
        wall_seconds=1.5,
        rows=[{"explorer_diagnostics": {"score_calls": 7, "search_stop_reason": "plateau"}}],
        memory_before={"rss_mb": 100.0, "maxrss_mb": 110.0},
        memory_after={"rss_mb": 125.5, "maxrss_mb": 130.0},
    )

    report = diag["typed_explorer_launch_reports"][0]
    assert report["process_rss_mb_before"] == pytest.approx(100.0)
    assert report["process_rss_mb_after"] == pytest.approx(125.5)
    assert report["process_rss_delta_mb"] == pytest.approx(25.5)
    assert report["process_maxrss_delta_mb"] == pytest.approx(20.0)


def _problem_900_like_group() -> DEFeatureGroup:
    return _problem_900_like_group_scaled(2.0)


def test_poly3_family_basis_and_design_include_cubic_term():
    basis = _family_basis_asts(U(), "poly3")
    assert len(basis) == 4
    assert basis[0] is None
    assert repr(basis[-1]) == repr(Pow(U(), 3.0))

    z = torch.tensor([[2.0], [-3.0]], dtype=torch.float64)
    Phi = _family_design_from_coord_values(z, "poly3")
    expected = torch.tensor(
        [
            [1.0, 2.0, 4.0, 8.0],
            [1.0, -3.0, 9.0, -27.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(Phi, expected)


def _problem_900_like_group_scaled(scale: float) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = float(scale) / (1.0 + x_fit)
    u_probe = float(scale) / (1.0 + x_probe)
    du_fit = -float(scale) / torch.pow(1.0 + x_fit, 2.0)
    du_probe = -float(scale) / torch.pow(1.0 + x_probe, 2.0)
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
    return DEFeatureGroup(id="de900_like", features=features)


def _problem_900_like_group_biased(
    *,
    scale: float,
    coeff_bias: float,
    surrogate_val_loss: float,
    group_id: str,
) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    coeff_fit = torch.pow(1.0 + x_fit, -1.0) + float(coeff_bias)
    coeff_probe = torch.pow(1.0 + x_probe, -1.0) + float(coeff_bias)
    u_fit = float(scale) / (1.0 + x_fit)
    u_probe = float(scale) / (1.0 + x_probe)
    du_fit = -coeff_fit * u_fit
    du_probe = -coeff_probe * u_probe
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
    return DEFeatureGroup(
        id=group_id,
        features=features,
        surrogate_val_loss=float(surrogate_val_loss),
    )


def _problem_906_like_group() -> DEFeatureGroup:
    x_fit = torch.linspace(0.1, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    t_fit = 1.0 + x_fit
    t_probe = 1.0 + x_probe
    u_fit = torch.sin(t_fit) / t_fit
    u_probe = torch.sin(t_probe) / t_probe
    du_fit = (t_fit * torch.cos(t_fit) - torch.sin(t_fit)) / torch.pow(t_fit, 2.0)
    du_probe = (t_probe * torch.cos(t_probe) - torch.sin(t_probe)) / torch.pow(t_probe, 2.0)
    d2u_fit = -(2.0 / t_fit) * du_fit - u_fit
    d2u_probe = -(2.0 / t_probe) * du_probe - u_probe
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=d2u_fit,
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=d2u_probe,
    )
    return DEFeatureGroup(id="de906_like", features=features)


def _mapping_only_group() -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = torch.exp(-x_fit + torch.cos(1.0 + x_fit))
    u_probe = torch.exp(-x_probe + torch.cos(1.0 + x_probe))
    du_fit = -(1.0 + torch.sin(1.0 + x_fit)) * u_fit
    du_probe = -(1.0 + torch.sin(1.0 + x_probe)) * u_probe
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
    return DEFeatureGroup(id="de_mapping_like", features=features)


def _problem_903_like_group() -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = -torch.log(1.0 + x_fit)
    u_probe = -torch.log(1.0 + x_probe)
    du_fit = -torch.pow(1.0 + x_fit, -1.0)
    du_probe = -torch.pow(1.0 + x_probe, -1.0)
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
    return DEFeatureGroup(id="de903_like", features=features)


def _problem_903_like_group_scaled_rhs(
    *,
    rhs_scale: float,
    surrogate_val_loss: float,
    group_id: str,
) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = -torch.log(1.0 + x_fit)
    u_probe = -torch.log(1.0 + x_probe)
    du_fit = -float(rhs_scale) * torch.exp(u_fit)
    du_probe = -float(rhs_scale) * torch.exp(u_probe)
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
    return DEFeatureGroup(
        id=group_id,
        features=features,
        surrogate_val_loss=float(surrogate_val_loss),
    )


def _autonomous_u_squared_group(*, shift: float, group_id: str) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = 1.0 / (x_fit + float(shift))
    u_probe = 1.0 / (x_probe + float(shift))
    du_fit = -torch.pow(u_fit, 2.0)
    du_probe = -torch.pow(u_probe, 2.0)
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
    return DEFeatureGroup(id=group_id, features=features)


def _harmonic_state_force_group(*, scale: float, group_id: str) -> DEFeatureGroup:
    x_fit = torch.linspace(0.1, 0.9, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.0, 1.4, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = float(scale) * torch.sin(x_fit)
    u_probe = float(scale) * torch.sin(x_probe)
    du_fit = float(scale) * torch.cos(x_fit)
    du_probe = float(scale) * torch.cos(x_probe)
    d2u_fit = -u_fit
    d2u_probe = -u_probe
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=d2u_fit,
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=d2u_probe,
    )
    return DEFeatureGroup(id=group_id, features=features)


def _second_order_x_coeff_group(*, scale: float, group_id: str) -> DEFeatureGroup:
    return _harmonic_state_force_group(scale=float(scale), group_id=group_id)


def _state_damping_group(*, shift: float, group_id: str) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = torch.log(x_fit + float(shift))
    u_probe = torch.log(x_probe + float(shift))
    du_fit = 1.0 / (x_fit + float(shift))
    du_probe = 1.0 / (x_probe + float(shift))
    d2u_fit = -torch.pow(x_fit + float(shift), -2.0)
    d2u_probe = -torch.pow(x_probe + float(shift), -2.0)
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=d2u_fit,
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=d2u_probe,
    )
    return DEFeatureGroup(id=group_id, features=features)


def _x_damping_group(*, scale: float, group_id: str) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = float(scale) * torch.exp(-x_fit)
    u_probe = float(scale) * torch.exp(-x_probe)
    du_fit = -u_fit
    du_probe = -u_probe
    d2u_fit = u_fit
    d2u_probe = u_probe
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=d2u_fit,
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=d2u_probe,
    )
    return DEFeatureGroup(id=group_id, features=features)


def _velocity_coeff_on_u_group(*, phase: float, group_id: str) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    u_fit = 1.4 + 0.25 * torch.sin(2.3 * x_fit + float(phase))
    u_probe = 1.4 + 0.25 * torch.sin(2.3 * x_probe + float(phase))
    du_fit = -1.2 + 2.7 * x_fit + 0.15 * torch.cos(5.0 * x_fit + float(phase))
    du_probe = -1.2 + 2.7 * x_probe + 0.15 * torch.cos(5.0 * x_probe + float(phase))
    d2u_fit = -(2.0 + 0.75 * du_fit.square()) * u_fit
    d2u_probe = -(2.0 + 0.75 * du_probe.square()) * u_probe
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=d2u_fit,
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=d2u_probe,
    )
    return DEFeatureGroup(id=group_id, features=features)


def _two_block_xu_u2_group(*, group_id: str, u_mode: str) -> DEFeatureGroup:
    x_fit = torch.linspace(0.1, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    if str(u_mode) == "rising":
        u_fit = 1.0 + x_fit
        u_probe = 1.0 + x_probe
        du_fit = torch.ones_like(u_fit)
        du_probe = torch.ones_like(u_probe)
    else:
        u_fit = 2.5 - 0.5 * x_fit
        u_probe = 2.5 - 0.5 * x_probe
        du_fit = -0.5 * torch.ones_like(u_fit)
        du_probe = -0.5 * torch.ones_like(u_probe)
    d2u_fit = -(x_fit * u_fit + torch.pow(u_fit, 2.0))
    d2u_probe = -(x_probe * u_probe + torch.pow(u_probe, 2.0))
    features = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=d2u_fit,
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=d2u_probe,
    )
    return DEFeatureGroup(id=group_id, features=features)


def _problem_902_like_group_scaled(scale: float, *, group_id: str) -> DEFeatureGroup:
    x_fit = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 256, dtype=torch.float64).reshape(-1, 1)
    phase_fit = -(1.0 + x_fit) * torch.log(1.0 + x_fit) + x_fit
    phase_probe = -(1.0 + x_probe) * torch.log(1.0 + x_probe) + x_probe
    u_fit = float(scale) * torch.exp(phase_fit)
    u_probe = float(scale) * torch.exp(phase_probe)
    du_fit = -torch.log(1.0 + x_fit) * u_fit
    du_probe = -torch.log(1.0 + x_probe) * u_probe
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
    return DEFeatureGroup(id=group_id, features=features)


def _piecewise_x2_group(
    *,
    scale: float,
    fit_lo: float,
    fit_hi: float,
    probe_lo: float,
    probe_hi: float,
    group_id: str,
) -> DEFeatureGroup:
    x_fit = torch.linspace(float(fit_lo), float(fit_hi), 192, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(float(probe_lo), float(probe_hi), 192, dtype=torch.float64).reshape(-1, 1)
    u_fit = float(scale) * torch.exp(-torch.pow(x_fit, 3.0) / 3.0)
    u_probe = float(scale) * torch.exp(-torch.pow(x_probe, 3.0) / 3.0)
    du_fit = -torch.pow(x_fit, 2.0) * u_fit
    du_probe = -torch.pow(x_probe, 2.0) * u_probe
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
    return DEFeatureGroup(id=group_id, features=features)


def test_factorized_x_coeff_on_u_prefers_symbolic_explorer_when_competitive(monkeypatch):
    def _fake_run_explorer(**kwargs):
        return [
            {
                "expr": "1/z",
                "toy_ast": ("div", ("const", 1.0), ("var", 0)),
                "nestynet_ast": Pow(Var(0), -1.0),
                "mse": 0.0,
                "mse_eff": 0.0,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
                "size": 1,
            }
        ]

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [_problem_900_like_group()],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=1.0e-6,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 1
    assert res.engine == "factorized"
    assert float(res.probe_rms) < 1.0e-10
    assert "u" in repr(res.nonanchor_ast)
    assert len(res.blocks) == 1
    assert res.diagnostics["lane"] == "x_coeff_on_u"
    assert res.diagnostics["family"] == "explorer"
    assert res.diagnostics["base_mode"] == "zero"
    measure = res.diagnostics["factorized_de_diagnostics"]
    assert measure["mode"] == "operator_factorized"
    assert measure["n_groups_total"] == 1
    assert measure["family_fit_attempts"] >= 8
    assert measure["family_candidates"] >= 1
    assert measure["explorer_launches"] >= 1
    assert measure["typed_explorer_launches"] >= 1
    assert measure["selected_lane"] == "x_coeff_on_u"
    assert measure["selected_family"] == "explorer"
    assert any(row["lane"] == "x_coeff_on_u" for row in measure["explorer_lane_calls"])
    json.dumps(measure)
    shortlist = list(res.diagnostics.get("shortlist_rows", []) or [])
    assert shortlist
    assert shortlist[0]["collapse_reason"] == "single_dataset_low_confidence"
    assert shortlist[0]["collapse_confidence"] == "low"
    assert shortlist[0]["collapse_pairs"] == 0
    zero_diag = res.diagnostics.get("zero_base_x_lane_diagnostics")
    assert isinstance(zero_diag, dict)
    assert isinstance(zero_diag.get("coord_reports"), list)
    refine_x_reports = [row for row in zero_diag["coord_reports"] if row.get("coord_ast") == repr(Add(ConstNode(1.0), Var(0)))]
    assert refine_x_reports
    assert refine_x_reports[0]["fit_target"] is not None
    assert refine_x_reports[0]["probe_target"] is not None
    assert refine_x_reports[0]["fit_target"]["samples"]
    assert any(score.get("family") == "explorer" for score in refine_x_reports[0]["candidate_scores"])
    assert any(score.get("collapse_reason") == "single_dataset_low_confidence" for score in refine_x_reports[0]["candidate_scores"])


def test_typed_explorer_caps_default_to_full_rows_for_correctness():
    hp = SimpleNamespace(n_fit=6000, n_probe=6000, refine_fit_subset=512)

    assert _typed_explorer_caps_from_hp(hp) == (None, None)
    assert _typed_explorer_caps_for_order(1, *_typed_explorer_caps_from_hp(hp)) == (
        None,
        None,
        "disabled_correctness_first",
    )
    assert _typed_explorer_caps_for_order(2, *_typed_explorer_caps_from_hp(hp)) == (
        2048,
        4096,
        "second_order_resource_guard",
    )
    assert _typed_explorer_caps_for_order(2, 512, 1024) == (512, 1024, "explicit")


def test_family_first_gate_skips_x_explorer_when_reciprocal_family_certifies(monkeypatch):
    def _unexpected_run_explorer(**kwargs):
        raise AssertionError("x coefficient family gate should skip explorer")

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _unexpected_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [_problem_900_like_group_scaled(2.0), _problem_900_like_group_scaled(2.5)],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(1,),
            include_const=False,
            include_x=True,
            include_u=True,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=0.98,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.diagnostics["lane"] == "x_coeff_on_u"
    assert res.diagnostics["family"] == "reciprocal"
    measure = res.diagnostics["factorized_de_diagnostics"]
    assert int(measure["explorer_skipped"]) >= 1
    assert int(measure["typed_explorer_launches"]) == 0
    gate_reports = list(measure.get("family_gate_reports", []) or [])
    assert any(row["lane"] == "x_coeff_on_u" and row["skip_explorer"] for row in gate_reports)


def test_family_first_gate_skips_state_explorer_when_poly_family_certifies(monkeypatch):
    def _unexpected_run_explorer(**kwargs):
        raise AssertionError("state family gate should skip explorer")

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _unexpected_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _autonomous_u_squared_group(shift=1.0, group_id="u2_a"),
            _autonomous_u_squared_group(shift=1.5, group_id="u2_b"),
        ],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(1,),
            include_const=True,
            include_x=False,
            include_u=True,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=0.98,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.diagnostics["lane"] == "state_nonlinearity"
    assert res.diagnostics["family"] == "poly2"
    measure = res.diagnostics["factorized_de_diagnostics"]
    assert int(measure["explorer_skipped"]) >= 1
    assert int(measure["typed_explorer_launches"]) == 0
    gate_reports = list(measure.get("family_gate_reports", []) or [])
    assert any(row["lane"] == "state_nonlinearity" and row["skip_explorer"] for row in gate_reports)


def test_collapse_scheduler_keeps_small_high_confidence_x_coord_pool(monkeypatch):
    calls = []

    def _fake_run_explorer(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [_problem_900_like_group_scaled(2.0), _problem_900_like_group_scaled(2.5)],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(1,),
            include_const=False,
            include_x=True,
            include_u=True,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=1.0e-20,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert len(calls) == 2
    measure = res.diagnostics["factorized_de_diagnostics"]
    assert int(measure["scheduler_coord_candidates_considered"]) == 2
    assert int(measure["scheduler_coord_candidates_skipped"]) == 0
    scheduler_reports = list(measure.get("lane_scheduler_reports", []) or [])
    assert scheduler_reports[0]["reason"] == "high_confidence_keep_all_small_pool"


def test_collapse_scheduler_keeps_all_x_coords_for_single_trajectory(monkeypatch):
    calls = []

    def _fake_run_explorer(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [_problem_900_like_group_scaled(2.0)],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(1,),
            include_const=False,
            include_x=True,
            include_u=True,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=1.0e-20,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert len(calls) == 2
    measure = res.diagnostics["factorized_de_diagnostics"]
    assert int(measure["scheduler_coord_candidates_considered"]) == 2
    assert int(measure["scheduler_coord_candidates_skipped"]) == 0
    scheduler_reports = list(measure.get("lane_scheduler_reports", []) or [])
    assert scheduler_reports[0]["reason"] == "inconclusive_keep_all"


def test_second_order_state_nonlinearity_lane_solves_harmonic_force_without_explorer(monkeypatch):
    def _unexpected_run_explorer(**kwargs):
        raise AssertionError("second-order state family gate should skip explorer")

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _unexpected_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _harmonic_state_force_group(scale=1.0, group_id="harm_a"),
            _harmonic_state_force_group(scale=1.05, group_id="harm_b"),
        ],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(2,),
            include_const=True,
            include_x=False,
            include_u=True,
            include_du=False,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=0.98,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 2
    assert float(res.probe_rms) < 1.0e-10
    assert res.diagnostics["lane"] == "second_order_state_nonlinearity"
    assert res.diagnostics["family"] == "poly2"
    measure = res.diagnostics["factorized_de_diagnostics"]
    assert int(measure["typed_explorer_launches"]) == 0
    assert any(
        row["lane"] == "second_order_state_nonlinearity" and row["skip_explorer"]
        for row in list(measure.get("family_gate_reports", []) or [])
    )


def test_second_order_x_coeff_on_u_lane_solves_harmonic_x_coeff_without_explorer(monkeypatch):
    def _unexpected_run_explorer(**kwargs):
        raise AssertionError("second-order x coefficient family gate should skip explorer")

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _unexpected_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _second_order_x_coeff_group(scale=1.0, group_id="xcoef_a"),
            _second_order_x_coeff_group(scale=1.3, group_id="xcoef_b"),
        ],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(2,),
            include_const=False,
            include_x=True,
            include_u=True,
            include_du=False,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=0.98,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 2
    assert float(res.probe_rms) < 1.0e-10
    assert res.diagnostics["lane"] == "second_order_x_coeff_on_u"
    assert res.diagnostics["family"] in {"log", "poly2", "reciprocal", "inv_square"}
    assert int(res.diagnostics["factorized_de_diagnostics"]["typed_explorer_launches"]) == 0


def test_second_order_velocity_coeff_on_u_lane_solves_du2_correction(monkeypatch):
    calls = []

    def _fake_run_explorer(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _velocity_coeff_on_u_group(phase=0.0, group_id="velcoef_a"),
            _velocity_coeff_on_u_group(phase=0.8, group_id="velcoef_b"),
        ],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(2,),
            include_const=False,
            include_x=False,
            include_u=True,
            include_du=True,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=0.98,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 2
    assert float(res.probe_rms) < 1.0e-10
    assert res.diagnostics["lane"] == "second_order_velocity_coeff_on_u"
    assert res.diagnostics["family"] == "poly2"
    measure = res.diagnostics["factorized_de_diagnostics"]
    gate_reports = list(measure.get("family_gate_reports", []) or [])
    assert any(
        row["lane"] == "second_order_velocity_coeff_on_u" and row["skip_explorer"]
        for row in gate_reports
    )
    vel_diag = measure["best_velocity_coeff_on_u_diagnostic"]
    assert vel_diag["probe_improved"]
    assert vel_diag["u_du2_coeff"] == pytest.approx(0.75, abs=1.0e-8)
    assert vel_diag["probe_rms_ratio"] < 1.0e-8


def test_second_order_state_damping_lane_solves_exp_damping(monkeypatch):
    def _empty_run_explorer(**kwargs):
        return []

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _empty_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _state_damping_group(shift=1.0, group_id="damp_u_a"),
            _state_damping_group(shift=1.5, group_id="damp_u_b"),
        ],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(2,),
            include_const=False,
            include_x=False,
            include_u=True,
            include_du=True,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=0.98,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 2
    assert float(res.probe_rms) < 1.0e-10
    assert res.diagnostics["lane"] == "second_order_state_damping_on_du"
    assert res.diagnostics["family"] == "exp"
    assert repr(res.blocks[0].coord_ast) == repr(Mul(ConstNode(-1.0), U()))
    gate_reports = list(res.diagnostics["factorized_de_diagnostics"].get("family_gate_reports", []) or [])
    assert any(
        row["lane"] == "second_order_state_damping_on_du" and row["skip_explorer"]
        for row in gate_reports
    )


def test_second_order_x_damping_lane_solves_constant_damping_without_explorer(monkeypatch):
    def _unexpected_run_explorer(**kwargs):
        raise AssertionError("second-order x damping family gate should skip explorer")

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _unexpected_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _x_damping_group(scale=1.0, group_id="damp_x_a"),
            _x_damping_group(scale=1.4, group_id="damp_x_b"),
        ],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(2,),
            include_const=False,
            include_x=True,
            include_u=False,
            include_du=True,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=0.98,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 2
    assert float(res.probe_rms) < 1.0e-10
    assert res.diagnostics["lane"] == "second_order_x_damping_on_du"
    assert res.diagnostics["family"] in {"log", "poly2", "reciprocal", "inv_square"}
    assert int(res.diagnostics["factorized_de_diagnostics"]["typed_explorer_launches"]) == 0


def test_two_block_typed_assembly_combines_xu_and_u2_blocks(monkeypatch):
    def _fake_run_explorer(**kwargs):
        return [
            {
                "expr": "z",
                "toy_ast": ("var", 0),
                "nestynet_ast": Var(0),
                "mse": 0.0,
                "mse_eff": 0.0,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
                "size": 1,
            },
            {
                "expr": "z^2",
                "toy_ast": ("pow", ("var", 0), 2.0),
                "nestynet_ast": Pow(Var(0), 2.0),
                "mse": 0.0,
                "mse_eff": 0.0,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
                "size": 2,
            },
        ]

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _two_block_xu_u2_group(group_id="twoblock_a", u_mode="rising"),
            _two_block_xu_u2_group(group_id="twoblock_b", u_mode="falling"),
        ],
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(2,),
            include_const=True,
            include_x=True,
            include_u=True,
            include_du=False,
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=1.0e-20,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=8,
            two_block_shared_coord_mode="always",
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 2
    assert float(res.probe_rms) < 1.0e-10
    assert res.diagnostics["lane"] == "two_block_typed_assembly"
    assert len(res.blocks) == 2
    roles = {block.role for block in res.blocks}
    assert roles == {"two_block_typed_assembly"}
    lanes = set(res.diagnostics["shortlist_rows"][0].get("assembly_lanes", []))
    assert {"second_order_x_coeff_on_u", "second_order_state_nonlinearity"} == lanes
    measure = res.diagnostics["factorized_de_diagnostics"]
    assert int(measure["two_block_typed_candidates"]) >= 1


def test_pooled_same_x_target_downweights_bad_surrogate():
    good = _problem_900_like_group_biased(
        scale=2.0,
        coeff_bias=0.0,
        surrogate_val_loss=1.0e-4,
        group_id="good",
    )
    bad = _problem_900_like_group_biased(
        scale=2.0,
        coeff_bias=2.0,
        surrogate_val_loss=1.0,
        group_id="bad",
    )
    pooled = _pooled_same_coord_coeff_target(
        groups=[good, bad],
        resid_parts=[good.features.du_fit.reshape(-1), bad.features.du_fit.reshape(-1)],
        carrier_ast=U(),
        coord_ast=Add(ConstNode(1.0), Var(0)),
        split="fit",
        x_axis=0,
        rel_eps=1.0e-6,
        min_rows=64,
    )

    assert pooled is not None
    coord_fit, coeff_fit, sample_weight = pooled
    good_target = torch.pow(coord_fit.reshape(-1), -1.0)
    bad_target = good_target + 2.0
    pooled_target = coeff_fit.reshape(-1)

    good_err = float(torch.mean(torch.abs(pooled_target - good_target)).item())
    bad_err = float(torch.mean(torch.abs(pooled_target - bad_target)).item())

    assert int(coord_fit.shape[0]) >= 64
    assert torch.all(sample_weight.reshape(-1) > 0.0)
    assert good_err < 0.25
    assert bad_err > 1.0


def test_pooled_same_x_target_is_robust_to_one_outlier_curve():
    good_a = _problem_900_like_group_biased(
        scale=2.0,
        coeff_bias=0.0,
        surrogate_val_loss=0.1,
        group_id="good_a",
    )
    good_b = _problem_900_like_group_biased(
        scale=3.0,
        coeff_bias=0.0,
        surrogate_val_loss=0.1,
        group_id="good_b",
    )
    bad = _problem_900_like_group_biased(
        scale=2.5,
        coeff_bias=2.0,
        surrogate_val_loss=0.1,
        group_id="bad",
    )
    pooled = _pooled_same_coord_coeff_target(
        groups=[good_a, good_b, bad],
        resid_parts=[
            good_a.features.du_fit.reshape(-1),
            good_b.features.du_fit.reshape(-1),
            bad.features.du_fit.reshape(-1),
        ],
        carrier_ast=U(),
        coord_ast=Add(ConstNode(1.0), Var(0)),
        split="fit",
        x_axis=0,
        rel_eps=1.0e-6,
        min_rows=64,
    )

    assert pooled is not None
    coord_fit, coeff_fit, sample_weight = pooled
    good_target = torch.pow(coord_fit.reshape(-1), -1.0)
    bad_target = good_target + 2.0
    pooled_target = coeff_fit.reshape(-1)

    good_err = float(torch.mean(torch.abs(pooled_target - good_target)).item())
    bad_err = float(torch.mean(torch.abs(pooled_target - bad_target)).item())

    assert int(coord_fit.shape[0]) >= 64
    assert torch.all(sample_weight.reshape(-1) > 0.0)
    assert good_err < 0.1
    assert bad_err > 1.0


def test_x_lane_robust_probe_score_downweights_corrupted_probe_group():
    good0 = _problem_902_like_group_scaled(1.5, group_id="g0")
    good1 = _problem_902_like_group_scaled(2.5, group_id="g1")
    bad = _problem_902_like_group_scaled(2.0, group_id="bad")
    du_probe_bad = bad.features.du_probe.clone()
    du_probe_bad[:32] = du_probe_bad[:32] + 50.0
    bad = DEFeatureGroup(
        id="bad",
        features=DEFeatureTensors(
            x_fit=bad.features.x_fit,
            u_fit=bad.features.u_fit,
            du_fit=bad.features.du_fit,
            d2u_fit=bad.features.d2u_fit,
            x_probe=bad.features.x_probe,
            u_probe=bad.features.u_probe,
            du_probe=du_probe_bad,
            d2u_probe=bad.features.d2u_probe,
        ),
    )

    groups = [good0, good1, bad]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]
    coeff_ast = Log(Add(ConstNode(1.0), Var(0)))

    raw_mse = _masked_original_scale_probe_mse(
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=U(),
        coeff_ast=coeff_ast,
        x_axis=0,
        rel_eps=1.0e-6,
        robust=False,
    )
    robust_mse = _masked_original_scale_probe_mse(
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=U(),
        coeff_ast=coeff_ast,
        x_axis=0,
        rel_eps=1.0e-6,
        robust=True,
    )

    assert math.isfinite(raw_mse)
    assert math.isfinite(robust_mse)
    assert robust_mse < 0.2 * raw_mse


def test_lane_consistency_prefers_shared_x_coeff_over_state_alias():
    groups = [_problem_900_like_group_scaled(2.0), _problem_900_like_group_scaled(2.5)]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]

    x_score, x_pairs, x_total = _lane_consistency_stats(
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=U(),
        coord_ast=Add(ConstNode(1.0), Var(0)),
        x_axis=0,
        rel_eps=1.0e-6,
    )
    state_score, state_pairs, state_total = _lane_consistency_stats(
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=ConstNode(1.0),
        coord_ast=U(),
        x_axis=0,
        rel_eps=1.0e-6,
    )

    assert x_pairs == x_total == 1
    assert state_pairs == state_total == 1
    assert float(x_score) < 1.0e-10
    assert float(state_score) > 1.0e-2


def test_residual_ratio_collapse_prefers_x_coeff_lane_for_shared_x_coeff_case():
    groups = [_problem_900_like_group_scaled(2.0), _problem_900_like_group_scaled(2.5)]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]

    x_diag = _residual_ratio_collapse_diagnostics(
        groups=groups,
        resid_parts=resid_probe_parts,
        carrier_ast=U(),
        coord_ast=Add(ConstNode(1.0), Var(0)),
        split="probe",
        x_axis=0,
        rel_eps=1.0e-6,
    )
    state_diag = _residual_ratio_collapse_diagnostics(
        groups=groups,
        resid_parts=resid_probe_parts,
        carrier_ast=ConstNode(1.0),
        coord_ast=U(),
        split="probe",
        x_axis=0,
        rel_eps=1.0e-6,
    )

    assert x_diag["collapse_reason"] == "ok"
    assert x_diag["collapse_pairs"] == x_diag["collapse_total_pairs"] == 1
    assert x_diag["collapse_coverage"] == pytest.approx(1.0)
    assert float(x_diag["collapse_score"]) < 1.0e-10
    assert float(state_diag["collapse_score"]) > 1.0e-2


def test_residual_ratio_collapse_prefers_state_lane_for_autonomous_u_squared_case():
    groups = [
        _autonomous_u_squared_group(shift=1.0, group_id="u2_a"),
        _autonomous_u_squared_group(shift=1.5, group_id="u2_b"),
    ]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]

    state_diag = _residual_ratio_collapse_diagnostics(
        groups=groups,
        resid_parts=resid_probe_parts,
        carrier_ast=ConstNode(1.0),
        coord_ast=U(),
        split="probe",
        x_axis=0,
        rel_eps=1.0e-6,
    )
    x_diag = _residual_ratio_collapse_diagnostics(
        groups=groups,
        resid_parts=resid_probe_parts,
        carrier_ast=U(),
        coord_ast=Var(0),
        split="probe",
        x_axis=0,
        rel_eps=1.0e-6,
    )

    assert state_diag["collapse_reason"] == "ok"
    assert state_diag["collapse_pairs"] == state_diag["collapse_total_pairs"] == 1
    assert state_diag["collapse_coverage"] == pytest.approx(1.0)
    assert float(state_diag["collapse_score"]) < 1.0e-10
    assert float(x_diag["collapse_score"]) > 1.0e-2


def test_residual_ratio_collapse_single_trajectory_is_low_confidence():
    group = _problem_900_like_group_scaled(2.0)
    diag = _residual_ratio_collapse_diagnostics(
        groups=[group],
        resid_parts=[group.features.du_probe.reshape(-1)],
        carrier_ast=U(),
        coord_ast=Var(0),
        split="probe",
        x_axis=0,
        rel_eps=1.0e-6,
    )

    assert diag["collapse_reason"] == "single_dataset_low_confidence"
    assert diag["collapse_confidence"] == "low"
    assert diag["collapse_pairs"] == 0
    assert diag["collapse_total_pairs"] == 0
    assert diag["collapse_group_coverage"] == pytest.approx(1.0)


def test_lane_native_witness_prefers_same_x_for_x_coeff_case():
    groups = [_problem_900_like_group_scaled(2.0), _problem_900_like_group_scaled(2.5)]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]

    x_kind, x_score, x_pairs, x_total = _lane_witness_stats(
        lane="x_coeff_on_u",
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=U(),
        coord_ast=Var(0),
        x_axis=0,
        rel_eps=1.0e-6,
    )
    state_kind, state_score, state_pairs, state_total = _lane_witness_stats(
        lane="state_nonlinearity",
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=ConstNode(1.0),
        coord_ast=U(),
        x_axis=0,
        rel_eps=1.0e-6,
    )

    assert x_kind == "same_x_witness"
    assert state_kind == "matched_u_witness"
    assert x_pairs == x_total == 1
    assert state_pairs == state_total == 1
    assert float(x_score) < float(state_score)


def test_zero_lane_choice_uses_structural_witness_before_probe():
    state_row = {
        "probe_rms": 0.12,
        "consistency_score": 0.86,
        "consistency_pairs": 5,
        "consistency_total_pairs": 6,
        "evidence_tier": "weakly_verified",
        "witness_kind": "matched_u_witness",
        "base_mode": "zero",
        "family": "explorer",
        "size": 1,
        "ratio_probe_mse": 0.01,
    }
    x_row = {
        "probe_rms": 0.36,
        "consistency_score": 0.71,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "evidence_tier": "verified",
        "witness_kind": "same_x_witness",
        "base_mode": "zero",
        "family": "explorer",
        "size": 1,
        "ratio_probe_mse": 0.02,
    }

    assert _choose_preferred_zero_lane(
        state_rows=[state_row],
        x_coeff_rows=[x_row],
    ) == "x_coeff_on_u"


def test_x_lane_representative_prefers_simple_family_when_competitive():
    family_row = {
        "lane": "x_coeff_on_u",
        "family": "log",
        "probe_rms": 0.307,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "shape_score": 0.02,
        "base_mode": "zero",
        "size": 1,
        "ratio_probe_mse": 0.08,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
    }
    explorer_row = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.28,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "shape_score": 2.4,
        "base_mode": "zero",
        "size": 1,
        "ratio_probe_mse": 0.07,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
    }

    choice = _select_lane_representative([family_row, explorer_row])
    assert choice is family_row


def test_x_lane_representative_does_not_replace_explorer_with_other_coordinate_family():
    family_row = {
        "lane": "x_coeff_on_u",
        "family": "log",
        "probe_rms": 0.29,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "shape_score": 0.02,
        "base_mode": "zero",
        "size": 1,
        "ratio_probe_mse": 0.08,
        "coord_ast": Var(0),
    }
    explorer_row = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.28,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "shape_score": 2.4,
        "base_mode": "zero",
        "size": 1,
        "ratio_probe_mse": 0.07,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
    }

    choice = _select_lane_representative([family_row, explorer_row])
    assert choice is explorer_row


def test_x_lane_selection_anchors_on_lowest_probe_explorer_before_family_replacement():
    x0_explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.080,
        "shape_score": 0.0,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 3,
        "ratio_probe_mse": 0.05,
        "fit_target_mse": 5.0e-3,
        "coord_ast": Var(0),
        "carrier_ast": U(),
        "coeff_ast": Add(Var(0), Sin(Var(0))),
        "block_ast": None,
    }
    shifted_explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.074,
        "shape_score": 2.5,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 4,
        "ratio_probe_mse": 0.051,
        "fit_target_mse": 1.0e-2,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Sin(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }
    shifted_family = {
        "lane": "x_coeff_on_u",
        "family": "reciprocal",
        "probe_rms": 0.091,
        "shape_score": 0.0,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.008,
        "fit_target_mse": 1.0e-7,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Pow(Add(ConstNode(1.0), Var(0)), -1.0),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([x0_explorer, shifted_explorer, shifted_family])

    assert representative is not None
    assert representative["family"] == "reciprocal"
    assert representative["coord_ast"] == Add(ConstNode(1.0), Var(0))
    assert any(row["family"] == "explorer" and row["coord_ast"] == Add(ConstNode(1.0), Var(0)) for row in kept)


def test_x_lane_selection_allows_stronger_shifted_family_to_replace_wrong_coord_explorer():
    wrong_coord_explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.131,
        "shape_score": 0.5,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 3,
        "ratio_probe_mse": 0.23,
        "fit_target_mse": 2.8e-2,
        "coord_ast": Var(0),
        "carrier_ast": U(),
        "coeff_ast": Add(Var(0), Sin(Var(0))),
        "block_ast": None,
    }
    shifted_reciprocal = {
        "lane": "x_coeff_on_u",
        "family": "reciprocal",
        "probe_rms": 0.207,
        "shape_score": 0.0,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.043,
        "fit_target_mse": 1.0e-7,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Pow(Add(ConstNode(1.0), Var(0)), -1.0),
        "block_ast": None,
    }
    wrong_coord_log = {
        "lane": "x_coeff_on_u",
        "family": "log",
        "probe_rms": 0.213,
        "shape_score": 0.0,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.044,
        "fit_target_mse": 2.2e-3,
        "coord_ast": Var(0),
        "carrier_ast": U(),
        "coeff_ast": Log(Var(0)),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([wrong_coord_explorer, shifted_reciprocal, wrong_coord_log])

    assert representative is not None
    assert representative["family"] == "reciprocal"
    assert representative["coord_ast"] == Add(ConstNode(1.0), Var(0))
    assert {str(row["family"]) for row in kept} == {"explorer", "reciprocal"}


def test_x_lane_selection_does_not_let_same_coord_poly2_block_stronger_shifted_family():
    wrong_coord_explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.131,
        "shape_score": 0.5,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 3,
        "ratio_probe_mse": 0.23,
        "fit_target_mse": 2.8e-2,
        "coord_ast": Var(0),
        "carrier_ast": U(),
        "coeff_ast": Add(Var(0), Sin(Var(0))),
        "block_ast": None,
    }
    same_coord_poly2 = {
        "lane": "x_coeff_on_u",
        "family": "poly2",
        "probe_rms": 0.169,
        "shape_score": 0.0,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 3,
        "ratio_probe_mse": 0.09,
        "fit_target_mse": 2.7e-3,
        "coord_ast": Var(0),
        "carrier_ast": U(),
        "coeff_ast": Add(ConstNode(1.0), Var(0)),
        "block_ast": None,
    }
    shifted_reciprocal = {
        "lane": "x_coeff_on_u",
        "family": "reciprocal",
        "probe_rms": 0.207,
        "shape_score": 0.0,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.043,
        "fit_target_mse": 1.0e-7,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Pow(Add(ConstNode(1.0), Var(0)), -1.0),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([wrong_coord_explorer, same_coord_poly2, shifted_reciprocal])

    assert representative is not None
    assert representative["family"] == "reciprocal"
    assert representative["coord_ast"] == Add(ConstNode(1.0), Var(0))
    assert {str(row["family"]) for row in kept} == {"explorer", "reciprocal"}


def test_primary_typed_refinement_only_runs_preferred_zero_lane():
    assert _active_first_order_typed_lanes(
        base_mode="zero",
        preferred_zero_lane=None,
        allow_state_lane=True,
        allow_x_coeff_lane=True,
    ) == (True, True)
    assert _active_first_order_typed_lanes(
        base_mode="primary",
        preferred_zero_lane="x_coeff_on_u",
        allow_state_lane=True,
        allow_x_coeff_lane=True,
    ) == (False, True)
    assert _active_first_order_typed_lanes(
        base_mode="primary",
        preferred_zero_lane="state_nonlinearity",
        allow_state_lane=True,
        allow_x_coeff_lane=True,
    ) == (True, False)
    assert _active_first_order_typed_lanes(
        base_mode="primary",
        preferred_zero_lane=None,
        allow_state_lane=True,
        allow_x_coeff_lane=True,
    ) == (False, False)


def test_x_lane_explorer_refit_uses_original_scale_objective():
    groups = [
        _piecewise_x2_group(scale=12.0, fit_lo=0.0, fit_hi=0.45, probe_lo=0.50, probe_hi=0.75, group_id="g0"),
        _piecewise_x2_group(scale=1.0, fit_lo=0.55, fit_hi=1.00, probe_lo=1.05, probe_hi=1.30, group_id="g1"),
    ]
    resid_fit_parts = [g.features.du_fit.reshape(-1) for g in groups]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]

    coeff_ast, mapping, probe_mse, coeff_ast_local = _fit_original_scale_affine_explorer_head(
        row={
            "expr": "z",
            "toy_ast": ("var", 0),
            "nestynet_ast": Var(0),
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
        },
        groups=groups,
        order=1,
        x_axis=0,
        resid_fit_parts=resid_fit_parts,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=U(),
        coord_ast=Var(0),
        rel_eps=1.0e-6,
        min_ratio_rows=64,
    )

    assert coeff_ast is not None
    assert coeff_ast_local is not None
    assert mapping.get("_factorized_refit") == "original_scale_affine"
    assert float(probe_mse) >= 0.0

    A_parts = []
    y_parts = []
    for group in groups:
        x_fit = group.features.x_fit.reshape(-1)
        u_fit = group.features.u_fit.reshape(-1)
        du_fit = group.features.du_fit.reshape(-1)
        scale = torch.median(torch.abs(u_fit)).clamp_min(1.0e-8)
        mask = torch.abs(u_fit) > 1.0e-6 * scale
        A_parts.append(torch.stack([u_fit[mask], u_fit[mask] * x_fit[mask]], dim=1))
        y_parts.append((-du_fit[mask]).reshape(-1, 1))

    sol = torch.linalg.lstsq(torch.cat(A_parts, dim=0), torch.cat(y_parts, dim=0)).solution.reshape(-1)
    pred = _eval_ast_on_features(coeff_ast, features=groups[0].features, split="fit", x_axis=0).reshape(-1)
    pred_local = _eval_univariate_ast_on_values(coeff_ast_local, groups[0].features.x_fit.reshape(-1))
    expected = sol[0] + sol[1] * groups[0].features.x_fit.reshape(-1)
    assert torch.max(torch.abs(pred - expected)).item() < 1.0e-8
    assert torch.max(torch.abs(pred_local - expected)).item() < 1.0e-8


def test_factorized_de_evaluators_support_abs_node():
    group = _problem_900_like_group()

    pred = _eval_ast_on_features(
        Abs(Add(Var(0), ConstNode(-0.5))),
        features=group.features,
        split="fit",
        x_axis=0,
    ).reshape(-1)
    expected = torch.abs(group.features.x_fit.reshape(-1) - 0.5)
    assert torch.allclose(pred, expected)

    z = torch.tensor([[-2.0], [0.0], [3.0]], dtype=torch.float64)
    pred_local = _eval_univariate_ast_on_values(Abs(Var(0)), z)
    assert torch.allclose(pred_local, torch.abs(z.reshape(-1)))
    assert _nestynet_static_const_value(Abs(ConstNode(-3.0))) == 3.0


def test_factorized_de_evaluators_support_inverse_trig_nodes():
    group = _problem_900_like_group()
    x = group.features.x_fit.reshape(-1)

    asin_expr = Asin(Sin(Mul(ConstNode(0.1), Var(0))))
    acos_expr = Acos(Sin(Mul(ConstNode(0.1), Var(0))))
    atan_expr = Atan(Var(0))

    pred_asin = _eval_ast_on_features(asin_expr, features=group.features, split="fit", x_axis=0).reshape(-1)
    pred_acos = _eval_ast_on_features(acos_expr, features=group.features, split="fit", x_axis=0).reshape(-1)
    pred_atan = _eval_ast_on_features(atan_expr, features=group.features, split="fit", x_axis=0).reshape(-1)
    expected_arg = torch.sin(0.1 * x)
    assert torch.allclose(pred_asin, torch.asin(expected_arg))
    assert torch.allclose(pred_acos, torch.acos(expected_arg))
    assert torch.allclose(pred_atan, torch.atan(x))

    z = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float64)
    assert torch.allclose(_eval_univariate_ast_on_values(Asin(Var(0)), z), torch.asin(z.reshape(-1)))
    assert torch.allclose(_eval_univariate_ast_on_values(Acos(Var(0)), z), torch.acos(z.reshape(-1)))
    assert torch.allclose(_eval_univariate_ast_on_values(Atan(Var(0)), z), torch.atan(z.reshape(-1)))
    assert _nestynet_static_const_value(Asin(ConstNode(0.5))) == pytest.approx(math.asin(0.5))
    assert _nestynet_static_const_value(Acos(ConstNode(0.5))) == pytest.approx(math.acos(0.5))
    assert _nestynet_static_const_value(Atan(ConstNode(0.5))) == pytest.approx(math.atan(0.5))
    assert _nestynet_static_const_value(Asin(ConstNode(2.0))) is None
    assert _nestynet_static_const_value(Acos(ConstNode(2.0))) is None


def test_x_lane_distillation_projects_explorer_curve_to_log_family():
    groups = [
        _problem_902_like_group_scaled(1.5, group_id="g0"),
        _problem_902_like_group_scaled(2.5, group_id="g1"),
    ]
    resid_fit_parts = [g.features.du_fit.reshape(-1) for g in groups]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]
    explorer_candidate = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "base_mode": "zero",
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Add(ConstNode(0.0), Log(Add(ConstNode(1.0), Var(0)))),
        "coeff_local_ast": Add(ConstNode(0.0), Log(Var(0))),
    }

    distilled = _distill_x_lane_explorer_candidate(
        explorer_candidate=explorer_candidate,
        groups=groups,
        order=1,
        x_axis=0,
        base_ast=None,
        base_mode="zero",
        resid_fit_parts=resid_fit_parts,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=U(),
        coord_ast=Add(ConstNode(1.0), Var(0)),
        rel_eps=1.0e-6,
        min_ratio_rows=64,
    )

    assert distilled
    log_rows = [row for row in distilled if row.get("family") == "log"]
    assert log_rows
    best = min(log_rows, key=lambda row: float(row.get("probe_rms", float("inf"))))
    assert float(best["probe_rms"]) < 1.0e-10


def test_state_lane_distillation_projects_explorer_curve_to_exp_family():
    groups = [
        _problem_903_like_group_scaled_rhs(
            rhs_scale=1.0,
            surrogate_val_loss=1.0e-6,
            group_id="g0",
        ),
        _problem_903_like_group_scaled_rhs(
            rhs_scale=1.0,
            surrogate_val_loss=5.0e-6,
            group_id="g1",
        ),
    ]
    resid_fit_parts = [g.features.du_fit.reshape(-1) for g in groups]
    resid_probe_parts = [g.features.du_probe.reshape(-1) for g in groups]
    explorer_candidate = {
        "lane": "state_nonlinearity",
        "family": "explorer",
        "base_mode": "zero",
        "coord_ast": Add(ConstNode(1.0), U()),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Exp(U()),
    }

    distilled = _distill_state_lane_explorer_candidate(
        explorer_candidate=explorer_candidate,
        groups=groups,
        order=1,
        x_axis=0,
        base_ast=None,
        base_mode="zero",
        resid_fit_parts=resid_fit_parts,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=ConstNode(1.0),
        coord_asts=[U(), Add(ConstNode(1.0), U())],
        rel_eps=1.0e-6,
        min_ratio_rows=64,
    )

    assert distilled
    exp_rows = [
        row
        for row in distilled
        if row.get("family") == "exp" and repr(row.get("coord_ast", None)) == repr(U())
    ]
    assert exp_rows
    best = min(exp_rows, key=lambda row: float(row.get("probe_rms", float("inf"))))
    assert float(best["probe_rms"]) < 1.0e-10


def test_candidate_preference_uses_consistency_only_for_meaningful_tradeoff():
    x_lane = {
        "probe_rms": 1.20,
        "consistency_score": 0.60,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 1,
        "ratio_probe_mse": 1.0,
    }
    state_lane = {
        "probe_rms": 0.74,
        "consistency_score": 0.77,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 1,
        "ratio_probe_mse": 1.0,
    }
    assert _candidate_preferred(x_lane, state_lane) is True

    weak_consistency_win = dict(x_lane, probe_rms=2.10, consistency_score=0.53)
    better_probe = dict(state_lane, probe_rms=0.91, consistency_score=0.62)
    assert _candidate_preferred(weak_consistency_win, better_probe) is False


def test_x_lane_selection_keeps_explorer_alive_alongside_worse_distilled_family():
    explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.41,
        "shape_score": 0.8,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 3,
        "ratio_probe_mse": 0.2,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Sin(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }
    distilled = {
        "lane": "x_coeff_on_u",
        "family": "log",
        "probe_rms": 0.66,
        "shape_score": 0.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.3,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Log(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([distilled, explorer])

    assert representative is not None
    assert representative["family"] == "explorer"
    assert {str(row["family"]) for row in kept} == {"explorer", "log"}


def test_x_lane_selection_allows_near_lossless_family_to_replace_explorer():
    explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.40,
        "shape_score": 1.2,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 4,
        "ratio_probe_mse": 0.2,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Sin(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }
    distilled = {
        "lane": "x_coeff_on_u",
        "family": "log",
        "probe_rms": 0.418,
        "shape_score": 0.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.22,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Log(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([distilled, explorer])

    assert representative is not None
    assert representative["family"] == "log"
    assert {str(row["family"]) for row in kept} == {"explorer", "log"}


def test_x_lane_selection_keeps_explorer_when_poly2_is_not_close_enough():
    explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.40,
        "shape_score": 1.2,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 4,
        "ratio_probe_mse": 0.2,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Sin(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }
    poly2 = {
        "lane": "x_coeff_on_u",
        "family": "poly2",
        "probe_rms": 0.445,
        "shape_score": 0.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 3,
        "ratio_probe_mse": 0.21,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Add(ConstNode(1.0), Var(0)),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([poly2, explorer])

    assert representative is not None
    assert representative["family"] == "explorer"
    assert {str(row["family"]) for row in kept} == {"explorer", "poly2"}


def test_x_lane_selection_prefers_log_when_fit_target_advantage_is_strong():
    explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.023,
        "shape_score": 1.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 4,
        "ratio_probe_mse": 0.008,
        "fit_target_mse": 1.0e-2,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Sin(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }
    log_family = {
        "lane": "x_coeff_on_u",
        "family": "log",
        "probe_rms": 0.042,
        "shape_score": 0.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.0017,
        "fit_target_mse": 1.0e-6,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Log(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }
    poly2 = {
        "lane": "x_coeff_on_u",
        "family": "poly2",
        "probe_rms": 0.033,
        "shape_score": 0.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 3,
        "ratio_probe_mse": 0.0010,
        "fit_target_mse": 5.0e-3,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Add(ConstNode(1.0), Var(0)),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([explorer, log_family, poly2])

    assert representative is not None
    assert representative["family"] == "log"
    assert {str(row["family"]) for row in kept} == {"explorer", "log"}


def test_x_lane_selection_checks_fit_target_strong_family_before_lower_probe_family():
    explorer = {
        "lane": "x_coeff_on_u",
        "family": "explorer",
        "probe_rms": 0.024,
        "shape_score": 1.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 4,
        "ratio_probe_mse": 0.008,
        "fit_target_mse": 1.0e-2,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Sin(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }
    inv_square = {
        "lane": "x_coeff_on_u",
        "family": "inv_square",
        "probe_rms": 0.039,
        "shape_score": 0.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.0015,
        "fit_target_mse": 2.0e-2,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Pow(Add(ConstNode(1.0), Var(0)), -2.0),
        "block_ast": None,
    }
    log_family = {
        "lane": "x_coeff_on_u",
        "family": "log",
        "probe_rms": 0.043,
        "shape_score": 0.0,
        "consistency_score": 0.7,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.0018,
        "fit_target_mse": 1.0e-4,
        "coord_ast": Add(ConstNode(1.0), Var(0)),
        "carrier_ast": U(),
        "coeff_ast": Log(Add(ConstNode(1.0), Var(0))),
        "block_ast": None,
    }

    representative, kept = _select_x_lane_candidates([explorer, inv_square, log_family])

    assert representative is not None
    assert representative["family"] == "log"
    assert {str(row["family"]) for row in kept} == {"explorer", "log"}


def test_state_lane_selection_keeps_best_explorer_and_family():
    explorer = {
        "lane": "state_nonlinearity",
        "family": "explorer",
        "probe_rms": 0.040,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 6,
        "ratio_probe_mse": 0.26,
        "fit_target_mse": 2.6e-1,
        "probe_target_mse": 2.6e-1,
        "coord_ast": Add(ConstNode(1.0), U()),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Sin(Add(ConstNode(1.0), U())),
        "block_ast": None,
    }
    exp_family = {
        "lane": "state_nonlinearity",
        "family": "exp",
        "probe_rms": 0.055,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.08,
        "fit_target_mse": 1.0e-5,
        "probe_target_mse": 1.0e-5,
        "coord_ast": U(),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Add(ConstNode(0.0), ConstNode(1.0)),
        "block_ast": None,
    }

    representative, kept = _select_state_lane_candidates([explorer, exp_family])

    assert representative is not None
    assert {str(row["family"]) for row in kept} == {"explorer", "exp"}


def test_state_lane_selection_preserves_diverse_closed_families():
    explorer = {
        "lane": "state_nonlinearity",
        "family": "explorer",
        "probe_rms": 0.18,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 6,
        "ratio_probe_mse": 0.26,
        "fit_target_mse": 2.6e-1,
        "probe_target_mse": 2.6e-1,
        "coord_ast": Add(ConstNode(1.0), U()),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Sin(Add(ConstNode(1.0), U())),
        "block_ast": None,
    }
    cos_family = {
        "lane": "state_nonlinearity",
        "family": "cos",
        "probe_rms": 0.62,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.10,
        "fit_target_mse": 8.0e-2,
        "probe_target_mse": 9.0e-2,
        "coord_ast": Add(ConstNode(1.0), U()),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Sin(Add(ConstNode(1.0), U())),
        "block_ast": None,
    }
    exp_family = {
        "lane": "state_nonlinearity",
        "family": "exp",
        "probe_rms": 0.95,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.15,
        "fit_target_mse": 1.1e-1,
        "probe_target_mse": 1.2e-1,
        "coord_ast": U(),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Exp(U()),
        "block_ast": None,
    }

    representative, kept = _select_state_lane_candidates([explorer, cos_family, exp_family])

    assert representative is not None
    assert {str(row["family"]) for row in kept} == {"explorer", "cos", "exp"}


def test_state_lane_selection_keeps_direct_exp_separate_from_distilled_exp():
    explorer = {
        "lane": "state_nonlinearity",
        "family": "explorer",
        "probe_rms": 0.18,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 6,
        "ratio_probe_mse": 0.26,
        "fit_target_mse": 2.6e-1,
        "probe_target_mse": 2.6e-1,
        "coord_ast": Add(ConstNode(1.0), U()),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Sin(Add(ConstNode(1.0), U())),
        "block_ast": None,
    }
    cos_family = {
        "lane": "state_nonlinearity",
        "family": "cos",
        "probe_rms": 0.62,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 0.10,
        "fit_target_mse": 8.0e-2,
        "probe_target_mse": 9.0e-2,
        "coord_ast": Add(ConstNode(1.0), U()),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Sin(Add(ConstNode(1.0), U())),
        "block_ast": None,
        "coeff_expr": "cos",
    }
    distilled_exp = {
        "lane": "state_nonlinearity",
        "family": "exp",
        "probe_rms": 60.0,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 60.0,
        "fit_target_mse": 1.0e-1,
        "probe_target_mse": 1.1e-1,
        "coord_ast": U(),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Add(ConstNode(0.4), Exp(U())),
        "block_ast": None,
        "coeff_expr": "exp[distilled]",
    }
    direct_exp = {
        "lane": "state_nonlinearity",
        "family": "exp",
        "probe_rms": 70.0,
        "consistency_score": 0.54,
        "consistency_pairs": 3,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 2,
        "ratio_probe_mse": 70.0,
        "fit_target_mse": 2.0e-1,
        "probe_target_mse": 2.1e-1,
        "coord_ast": U(),
        "carrier_ast": ConstNode(1.0),
        "coeff_ast": Exp(U()),
        "block_ast": None,
        "coeff_expr": "exp",
    }

    representative, kept = _select_state_lane_candidates([explorer, cos_family, distilled_exp, direct_exp])

    assert representative is not None
    exp_exprs = {str(row.get("coeff_expr", "")) for row in kept if str(row.get("family")) == "exp"}
    assert {"exp", "exp[distilled]"} <= exp_exprs


def test_candidate_preference_allows_verified_zero_x_lane_to_beat_unverified_primary_state():
    verified_x_lane = {
        "lane": "x_coeff_on_u",
        "probe_rms": 0.50,
        "consistency_score": 0.70,
        "consistency_pairs": 6,
        "consistency_total_pairs": 6,
        "base_mode": "zero",
        "size": 1,
        "ratio_probe_mse": 1.0,
    }
    unverified_primary_state = {
        "probe_rms": 0.22,
        "consistency_score": None,
        "consistency_pairs": 0,
        "consistency_total_pairs": 6,
        "base_mode": "primary",
        "size": 1,
        "ratio_probe_mse": 1.0,
    }
    assert _candidate_preferred(verified_x_lane, unverified_primary_state) is True


def test_factorized_state_nonlinearity_can_fit_exp_family_directly():
    group = _problem_903_like_group()
    cand = _fit_family_lane_candidate(
        lane="state_nonlinearity",
        family="exp",
        base_mode="zero",
        groups=[group],
        order=1,
        x_axis=0,
        base_ast=None,
        resid_fit_parts=[group.features.du_fit.reshape(-1)],
        resid_probe_parts=[group.features.du_probe.reshape(-1)],
        carrier_ast=ConstNode(1.0),
        coord_ast=U(),
        rel_eps=1.0e-6,
        min_ratio_rows=64,
    )

    assert cand is not None
    assert cand["family"] == "exp"
    assert float(cand["probe_rms"]) < 1.0e-10


def test_state_family_exp_fit_downweights_bad_surrogate_group():
    good = _problem_903_like_group_scaled_rhs(
        rhs_scale=1.0,
        surrogate_val_loss=1.0e-6,
        group_id="good",
    )
    bad = _problem_903_like_group_scaled_rhs(
        rhs_scale=5.0,
        surrogate_val_loss=1.0e6,
        group_id="bad",
    )

    cand = _fit_family_lane_candidate(
        lane="state_nonlinearity",
        family="exp",
        base_mode="zero",
        groups=[good, bad],
        order=1,
        x_axis=0,
        base_ast=None,
        resid_fit_parts=[good.features.du_fit.reshape(-1), bad.features.du_fit.reshape(-1)],
        resid_probe_parts=[good.features.du_probe.reshape(-1), bad.features.du_probe.reshape(-1)],
        carrier_ast=ConstNode(1.0),
        coord_ast=U(),
        rel_eps=1.0e-6,
        min_ratio_rows=64,
    )

    assert cand is not None
    coeff_probe = _eval_ast_on_features(cand["coeff_ast"], features=good.features, split="probe", x_axis=0).reshape(-1)
    target_probe = (-good.features.du_probe).reshape(-1)
    mse = float(torch.mean((coeff_probe - target_probe) ** 2).item())

    assert cand["family"] == "exp"
    assert mse < 5.0e-2
    assert float(cand["probe_rms"]) < 2.0e-1


def test_factorized_x_lane_can_recover_log_shift_from_family(monkeypatch):
    def _fake_run_explorer(**kwargs):
        return [
            {
                "expr": "0",
                "toy_ast": ("const", 0.0),
                "nestynet_ast": ConstNode(0.0),
                "mse": 1.0,
                "mse_eff": 1.0,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
                "size": 1,
            }
        ]

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [
            _problem_902_like_group_scaled(1.5, group_id="g0"),
            _problem_902_like_group_scaled(2.5, group_id="g1"),
        ],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert float(res.probe_rms) < 1.0e-10
    assert res.diagnostics["lane"] == "x_coeff_on_u"
    assert res.diagnostics["family"] == "log"
    assert res.diagnostics["base_mode"] == "zero"


def test_factorized_coeff_on_carrier_embeds_mapping_head(monkeypatch):
    def _fake_run_explorer(**kwargs):
        return [
            {
                "expr": "sin(z)",
                "toy_ast": ("sin", ("var", 0)),
                "nestynet_ast": None,
                "mse": 0.0,
                "mse_eff": 0.0,
                "mapping": {"kind": "poly", "coeffs": [1.0, 1.0]},
                "size": 1,
            }
        ]

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [_mapping_only_group()],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            replace_rel_factor=1.0e-6,
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert float(res.probe_rms) < 1.0e-10
    assert repr(res.blocks[0].coord_ast) == repr(Add(ConstNode(1.0), Var(0)))
    assert repr(res.blocks[0].coeff_ast) != repr(Sin(Add(ConstNode(1.0), Var(0))))
    assert res.diagnostics["family"] == "explorer"


def test_factorized_zero_base_mode_can_override_bad_primary(monkeypatch):
    def _fake_run_explorer(**kwargs):
        return [
            {
                "expr": "0",
                "toy_ast": ("const", 0.0),
                "nestynet_ast": ConstNode(0.0),
                "mse": 1.0,
                "mse_eff": 1.0,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
                "size": 1,
            }
        ]

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    bad_primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([10.0], dtype=torch.float64),
        rms_train=1.0,
        rms_val=1.0,
    )

    res = run_factorized_coeff_rescue_from_feature_groups(
        [_problem_900_like_group()],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=4,
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=bad_primary,
        dtype=torch.float64,
    )

    assert res is not None
    assert float(res.probe_rms) < 1.0e-10
    # On this single-trajectory fixture u = 1/(1+x), so u' = -u/(1+x) and
    # u' = -u^2 are the same law; either exact representation may win the
    # float-noise tie between them.
    assert (res.diagnostics["lane"], res.diagnostics["family"]) in {
        ("x_coeff_on_u", "reciprocal"),
        ("state_nonlinearity", "poly2"),
    }
    assert res.diagnostics["base_mode"] == "zero"


def test_factorized_two_block_shared_coord_finds_du_refine_u_structure(monkeypatch):
    def _fake_run_explorer(**kwargs):
        return [
            {
                "expr": "1",
                "toy_ast": ("const", 1.0),
                "nestynet_ast": None,
                "mse": 0.0,
                "mse_eff": 0.0,
                "mapping": {"kind": "poly", "coeffs": [1.0]},
                "size": 1,
            },
            {
                "expr": "1/z",
                "toy_ast": ("div", ("const", 1.0), ("var", 0)),
                "nestynet_ast": None,
                "mse": 0.0,
                "mse_eff": 0.0,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
                "size": 2,
            },
        ]

    monkeypatch.setattr("nestynet_sr.sr_de.factorized_de.run_explorer", _fake_run_explorer)

    res = run_factorized_coeff_rescue_from_feature_groups(
        [_problem_906_like_group()],
        cfg=DESearchConfig(x_axis=0, order_candidates=(2,), include_x=True, include_u=True, include_du=True),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            ratio_rel_eps=1.0e-6,
            min_ratio_rows=64,
            shortlist_topk=8,
            two_block_shared_coord_mode="always",
            hp=SimpleNamespace(n_iter=4000, max_depth=4, seed=0),
        ),
        primary=None,
        dtype=torch.float64,
    )

    assert res is not None
    assert res.order == 2
    assert res.engine == "factorized"
    assert float(res.probe_rms) < 5.0e-10
    assert len(res.blocks) == 2
    # Both two-block assembly routes can now reach this exact du+u structure
    # (the typed assembly gained the x-coeff x x-damping pairing for de117).
    assert res.diagnostics["lane"] in {"two_block_shared_coord", "two_block_typed_assembly"}


def test_factorized_pools_respect_cfg_flags():
    cfg = DESearchConfig(
        x_axis=0,
        order_candidates=(2,),
        include_const=False,
        include_x=False,
        include_u=True,
        include_du=False,
    )

    coords = _coord_pool(cfg=cfg, order=2, x_axis=0)
    carriers = _carrier_pool(cfg=cfg, order=2, x_axis=0)

    coord_reprs = {repr(node) for node in coords}
    carrier_reprs = {repr(node) for node in carriers}

    assert all("Var" not in text for text in coord_reprs)
    assert carrier_reprs == {repr(U())}
    assert coord_reprs == {repr(U()), repr(Add(ConstNode(1.0), U()))}


# ---------------------------------------------------------------------------
# Robust trimmed refit + shortlist diversity (de115 Lane-Emden regression)
# ---------------------------------------------------------------------------


def test_trimmed_ridge_lstsq_matches_plain_on_clean_data():
    from nestynet_sr.sr_core.numerics import ridge_lstsq
    from nestynet_sr.sr_de.factorized_de import _trimmed_ridge_lstsq

    gen = torch.Generator().manual_seed(0)
    Phi = torch.randn(500, 3, dtype=torch.float64, generator=gen)
    truth = torch.tensor([1.5, -0.7, 0.2], dtype=torch.float64)
    y = Phi @ truth + 1.0e-3 * torch.randn(500, dtype=torch.float64, generator=gen)

    plain = ridge_lstsq(Phi, y, ridge=0.0).reshape(-1)
    trimmed = _trimmed_ridge_lstsq(Phi, y, ridge=0.0).reshape(-1)
    assert torch.allclose(plain, trimmed)


def test_trimmed_ridge_lstsq_recovers_truth_under_leverage_outliers():
    from nestynet_sr.sr_core.numerics import ridge_lstsq
    from nestynet_sr.sr_de.factorized_de import _trimmed_ridge_lstsq

    gen = torch.Generator().manual_seed(1)
    n = 2000
    Phi = torch.randn(n, 2, dtype=torch.float64, generator=gen)
    truth = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = Phi @ truth

    # A handful of extreme-leverage corrupted rows (unresolved boundary layer).
    Phi[:5, 0] = 40.0
    y[:5] = -300.0

    plain = ridge_lstsq(Phi, y, ridge=0.0).reshape(-1)
    trimmed = _trimmed_ridge_lstsq(Phi, y, ridge=0.0).reshape(-1)
    assert float((plain - truth).abs().max()) > 0.05
    assert float((trimmed - truth).abs().max()) < 1.0e-8


def test_trimmed_mean_sq_ignores_rare_spikes_only():
    from nestynet_sr.sr_de.factorized_de import _trimmed_mean_sq

    gen = torch.Generator().manual_seed(2)
    r = 0.01 * torch.randn(1000, dtype=torch.float64, generator=gen)
    base = _trimmed_mean_sq(r)
    assert abs(base - float(torch.mean(r**2))) < 0.5 * float(torch.mean(r**2))

    r_spiked = r.clone()
    r_spiked[:4] = 50.0
    spiked = _trimmed_mean_sq(r_spiked)
    # Without trimming the spikes would dominate (mean sq ~ 10); with the
    # trim the score stays at the noise floor.
    assert spiked < 10.0 * base


def _lane_emden_like_group(*, n_corrupt: int, group_id: str) -> DEFeatureGroup:
    """Synthetic two-block features: d2u = -(2*du/x + u^3), boundary spikes.

    The corruption hits the du FEATURE near x_min, where the 1/x coordinate
    amplifies it into extreme-leverage rows. Such rows pull the least-squares
    fit through themselves (small residual at the outlier), so residual-only
    trimming cannot see them — the leverage pre-trim must (de115 mechanism).
    """
    x = torch.linspace(0.05, 3.0, 1200, dtype=torch.float64)
    u = torch.cos(x)
    du = -torch.sin(x)
    d2u = -(2.0 * du / x + u**3)
    if n_corrupt > 0:
        du = du.clone()
        du[:n_corrupt] += 5.0  # unresolved boundary-layer derivative error
        d2u = d2u.clone()
        d2u[:n_corrupt] += 30.0
    features = DEFeatureTensors(
        x_fit=x[::2].reshape(-1, 1),
        u_fit=u[::2].reshape(-1, 1),
        du_fit=du[::2].reshape(-1, 1),
        d2u_fit=d2u[::2].reshape(-1, 1),
        x_probe=x[1::2].reshape(-1, 1),
        u_probe=u[1::2].reshape(-1, 1),
        du_probe=du[1::2].reshape(-1, 1),
        d2u_probe=d2u[1::2].reshape(-1, 1),
    )
    return DEFeatureGroup(id=group_id, features=features)


def test_two_block_refit_survives_boundary_layer_spikes():
    from nestynet_sr.sr_core.bridges import DU
    from nestynet_sr.sr_de.factorized_de import _refit_shared_block_combo

    groups = [
        _lane_emden_like_group(n_corrupt=5, group_id="lane_emden_like_0"),
        _lane_emden_like_group(n_corrupt=5, group_id="lane_emden_like_1"),
    ]
    zeros = [torch.zeros(600, dtype=torch.float64) for _ in groups]

    def _row(carrier_ast, coord_ast, coeff_ast, expr):
        return {
            "block_ast": Mul(coeff_ast, carrier_ast),
            "carrier_ast": carrier_ast,
            "coord_ast": coord_ast,
            "coeff_ast": coeff_ast,
            "coeff_expr": expr,
            "mapping": {},
            "size": 2,
            "ratio_probe_mse": 0.1,
            "consistency_score": 0.1,
            "consistency_pairs": 2,
            "consistency_total_pairs": 2,
            "probe_rms": 0.1,
        }

    du_row = _row(DU(0), Var(0), Pow(Var(0), -1.0), "1/x")
    u_row = _row(U(), U(), Pow(U(), 2.0), "u^2")

    cand = _refit_shared_block_combo(
        [du_row, u_row],
        base_mode="zero",
        groups=groups,
        base_fit_parts=zeros,
        base_probe_parts=zeros,
        base_ast=None,
        order=2,
        x_axis=0,
        dtype=torch.float64,
    )

    assert cand is not None
    weights = [float(b.diagnostics["top_level_weight"]) for b in cand["blocks"]]
    # True law: u'' + 2*(du/x) + 1*(u^3) = 0 — recovered despite the spikes.
    assert abs(weights[0] - 2.0) < 1.0e-6
    assert abs(weights[1] - 1.0) < 1.0e-6
    assert float(cand["probe_rms"]) < 1.0e-6


def test_diverse_candidate_shortlist_breaks_single_basin_monopoly():
    from nestynet_sr.sr_de.factorized_de import _diverse_candidate_shortlist

    def _mk(lane, family, rms):
        return {"lane": lane, "family": family, "base_mode": "zero", "probe_rms": rms}

    rows = [
        _mk("second_order_state_nonlinearity", "poly3", 0.0028),
        _mk("second_order_state_nonlinearity", "poly3", 0.0028),
        _mk("second_order_state_nonlinearity", "poly3", 0.0029),
        _mk("second_order_state_nonlinearity", "poly3", 0.0029),
        _mk("two_block_typed_assembly", "shared_refit", 0.0030),
        _mk("second_order_x_damping_on_du", "reciprocal", 0.0460),
        _mk("second_order_state_nonlinearity", "poly3", 0.0031),
        _mk("second_order_state_nonlinearity", "poly3", 0.0031),
    ]

    shortlist = _diverse_candidate_shortlist(rows, 4)
    assert len(shortlist) == 4
    # Best overall stays first; every distinct structure is represented.
    assert shortlist[0] is rows[0]
    lanes = {(r["lane"], r["family"]) for r in shortlist}
    assert ("two_block_typed_assembly", "shared_refit") in lanes
    assert ("second_order_x_damping_on_du", "reciprocal") in lanes
    # Remaining slot back-filled by rank with a clone.
    assert sum(r["lane"] == "second_order_state_nonlinearity" for r in shortlist) == 2


def _bessel_like_group(*, group_id: str) -> DEFeatureGroup:
    """Synthetic Bessel-type features: d2u = -(du/x) - (1 - nu^2/x^2)*u, nu=1.5."""
    nu_sq = 2.25
    x = torch.linspace(0.5, 6.0, 1200, dtype=torch.float64)
    u = torch.cos(1.3 * x) / torch.sqrt(x)
    du = -1.3 * torch.sin(1.3 * x) / torch.sqrt(x) - 0.5 * u / x
    d2u = -(du / x) - (1.0 - nu_sq / x**2) * u
    features = DEFeatureTensors(
        x_fit=x[::2].reshape(-1, 1),
        u_fit=u[::2].reshape(-1, 1),
        du_fit=du[::2].reshape(-1, 1),
        d2u_fit=d2u[::2].reshape(-1, 1),
        x_probe=x[1::2].reshape(-1, 1),
        u_probe=u[1::2].reshape(-1, 1),
        du_probe=du[1::2].reshape(-1, 1),
        d2u_probe=d2u[1::2].reshape(-1, 1),
    )
    return DEFeatureGroup(id=group_id, features=features)


def test_two_block_assembly_allows_bessel_x_coeff_x_damping_pair():
    """Regression (de117): u'' + g(x)*u' + h(x)*u needs the x-coeff x x-damping
    pair, which the assembly previously forbade."""
    from nestynet_sr.sr_core.bridges import DU
    from nestynet_sr.sr_de.factorized_de import (
        _build_two_block_typed_candidates,
        _typed_two_block_pair_allowed,
    )

    groups = [_bessel_like_group(group_id="bessel_like_0")]
    zeros = [torch.zeros(600, dtype=torch.float64) for _ in groups]

    def _row(lane, family, carrier_ast, coord_ast, coeff_ast, expr):
        return {
            "lane": lane,
            "family": family,
            "base_mode": "zero",
            "block_ast": Mul(coeff_ast, carrier_ast),
            "carrier_ast": carrier_ast,
            "coord_ast": coord_ast,
            "coeff_ast": coeff_ast,
            "coeff_expr": expr,
            "mapping": {},
            "size": 2,
            "ratio_probe_mse": 0.1,
            "consistency_score": 0.1,
            "consistency_pairs": 2,
            "consistency_total_pairs": 2,
            "probe_rms": 0.1,
        }

    xcoeff_row = _row(
        "second_order_x_coeff_on_u", "inv_square", U(), Var(0), Pow(Var(0), -2.0), "1/x^2"
    )
    xdamp_row = _row(
        "second_order_x_damping_on_du", "reciprocal", DU(0), Var(0), Pow(Var(0), -1.0), "1/x"
    )
    assert _typed_two_block_pair_allowed(xcoeff_row, xdamp_row)

    cands = _build_two_block_typed_candidates(
        [xdamp_row, xcoeff_row],
        base_mode="zero",
        groups=groups,
        base_fit_parts=zeros,
        base_probe_parts=zeros,
        base_ast=None,
        order=2,
        x_axis=0,
        dtype=torch.float64,
        replace_rel_factor=0.98,
    )
    assert cands
    best = min(cands, key=lambda c: float(c.get("probe_rms", float("inf"))))
    # Joint inner refit recovers the exact Bessel law from the pair.
    assert float(best["probe_rms"]) < 1.0e-8
    eq = str(best["canonical_equation"])
    assert "u_x0" in eq and "x0 ** -2" in eq


def test_target_scale_row_weights_gate():
    from nestynet_sr.sr_de.factorized_de import _target_scale_row_weights

    # modest dynamic range: no weighting (absolute fitting preserved)
    y_mod = torch.linspace(0.5, 2.0, 200, dtype=torch.float64)
    assert _target_scale_row_weights(y_mod) is None

    # decades of dynamic range: relative weights, floored away from zero
    x = torch.linspace(0.001, 10.0, 2000, dtype=torch.float64)
    y_wide = 1.0 / x**2
    w = _target_scale_row_weights(y_wide)
    assert w is not None
    assert torch.isfinite(w).all()
    assert float((w * y_wide.abs()).max()) <= 1.0 + 1.0e-12


def test_first_order_x_coeff_lane_survives_dynamic_range(monkeypatch):
    """Regression (de206): u' = -u/x spans ~4 decades; absolute least squares
    concentrates all weight in the steep noisy band and splits the coefficient
    between collinear columns. Scale-weighted fitting recovers the exact law."""
    monkeypatch.setattr(
        "nestynet_sr.sr_de.factorized_de.run_explorer", lambda **kwargs: []
    )

    groups = []
    for j, c in enumerate((1.2e-3, 2.0e-3)):
        x = torch.linspace(0.001, 10.0, 2400, dtype=torch.float64)
        u = c / x
        du = -c / x**2
        du = du.clone()
        du[:4] *= 1.5  # unresolved boundary-layer derivative error
        feats = DEFeatureTensors(
            x_fit=x[::2].reshape(-1, 1),
            u_fit=u[::2].reshape(-1, 1),
            du_fit=du[::2].reshape(-1, 1),
            d2u_fit=torch.zeros(1200, 1, dtype=torch.float64),
            x_probe=x[1::2].reshape(-1, 1),
            u_probe=u[1::2].reshape(-1, 1),
            du_probe=du[1::2].reshape(-1, 1),
            d2u_probe=torch.zeros(1200, 1, dtype=torch.float64),
        )
        groups.append(DEFeatureGroup(id=f"inv_decay_{j}", features=feats))

    res = run_factorized_coeff_rescue_from_feature_groups(
        groups,
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=True, include_u=True),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            hp=SimpleNamespace(n_iter=2000, max_depth=4, seed=0),
        ),
        dtype=torch.float64,
    )

    assert res is not None
    assert float(res.probe_rms) < 1.0e-6
    eq = str(res.canonical_equation)
    assert "x0 ** -1" in eq


def test_x_coeff_family_generates_support_and_snap_variants():
    x = torch.linspace(0.5, 5.0, 512, dtype=torch.float64)
    u = 2.0 / x
    du = -(0.02 + 0.964 / x) * u
    feats = DEFeatureTensors(
        x_fit=x[::2].reshape(-1, 1),
        u_fit=u[::2].reshape(-1, 1),
        du_fit=du[::2].reshape(-1, 1),
        d2u_fit=torch.zeros(256, 1, dtype=torch.float64),
        x_probe=x[1::2].reshape(-1, 1),
        u_probe=u[1::2].reshape(-1, 1),
        du_probe=du[1::2].reshape(-1, 1),
        d2u_probe=torch.zeros(256, 1, dtype=torch.float64),
    )
    group = DEFeatureGroup(id="snap_decay", features=feats)

    rows = _fit_family_lane_candidate(
        lane="x_coeff_on_u",
        family="reciprocal",
        base_mode="zero",
        groups=[group],
        order=1,
        x_axis=0,
        base_ast=None,
        resid_fit_parts=[group.features.du_fit.reshape(-1)],
        resid_probe_parts=[group.features.du_probe.reshape(-1)],
        carrier_ast=U(),
        coord_ast=Var(0),
        rel_eps=1.0e-6,
        min_ratio_rows=64,
        return_variants=True,
    )

    assert isinstance(rows, list)
    assert any(row.get("projection_kind") == "support" and row.get("projection_support") == [1] for row in rows)
    snapped = [
        row
        for row in rows
        if row.get("projection_kind") == "snap"
        and row.get("projection_support") == [1]
        and row.get("projection_coeffs") == [1.0]
    ]
    assert snapped
    assert "x0 ** -1" in str(snapped[0]["canonical_equation"])


def test_projection_variants_are_not_generated_for_second_order_lanes():
    assert _lane_allows_linear_projection_variants("x_coeff_on_u")
    assert not _lane_allows_linear_projection_variants("second_order_x_coeff_on_u")
    assert not _lane_allows_linear_projection_variants("second_order_x_damping_on_du")


# ---------------------------------------------------------------------------
# Periodic seeding (de301/de126: forcing and parametric frequency discovery)
# ---------------------------------------------------------------------------


def test_periodogram_frequency_hints_finds_tone_and_skips_decay():
    from nestynet_sr.sr_search.factorized_search.engine.search import (
        _periodogram_frequency_hints,
    )

    x = torch.linspace(0.0, 31.4, 2000, dtype=torch.float64).reshape(-1, 1)
    y_tone = 3.0 * torch.cos(2.7 * x[:, 0]) + 0.4
    hints = _periodogram_frequency_hints(x, y_tone)
    assert hints
    var_idx, omega = hints[0]
    assert var_idx == 0
    # one rfft bin over span 31.4 is 2*pi/31.4 = 0.2 rad
    assert abs(omega - 2.7) < 0.25

    y_decay = torch.exp(-0.3 * x[:, 0])
    assert _periodogram_frequency_hints(x, y_decay) == []

    for dtype in (torch.float32, torch.float64):
        x_linear = x.to(dtype=dtype)
        y_linear = 3.7 * x_linear[:, 0] - 2.1
        assert _periodogram_frequency_hints(x_linear, y_linear) == []

    tiny_hints = _periodogram_frequency_hints(x, 1.0e-200 * torch.cos(2.7 * x[:, 0]))
    assert tiny_hints
    assert abs(tiny_hints[0][1] - 2.7) < 0.25


def test_direct_lane_recovers_periodic_forcing_via_seeded_combo():
    """Regression (de301-class): u'' = -omega^2*u + F*cos(gamma*x). The
    forcing frequency is invisible to correlation-guided search; the
    periodogram seeds pin trig atoms into the additive-combo pool, whose
    joint linear solve recovers the law in closed form."""
    from nestynet_sr.sr_de.factorized_de import (
        FactorizedSearchDERescueConfig,
        default_physics_rescue_hp,
        run_direct_residual_fss_from_feature_groups,
    )

    # Two trajectories with different tone mixes: on a single trajectory the
    # forcing is expressible through u' and the natural-frequency seed (an
    # exact alias); pooling breaks the degeneracy, as in the real benchmark.
    x = torch.linspace(0.0, 31.4, 2400, dtype=torch.float64)
    groups = []
    for gid, (a, b) in enumerate(((1.0, 0.3), (-0.6, 0.55))):
        u = a * torch.cos(1.4 * x) + b * torch.sin(2.7 * x)
        du = -1.4 * a * torch.sin(1.4 * x) + 2.7 * b * torch.cos(2.7 * x)
        d2u = -4.0 * u + 3.0 * torch.cos(2.7 * x)  # the law to discover
        feats = DEFeatureTensors(
            x_fit=x[::2].reshape(-1, 1),
            u_fit=u[::2].reshape(-1, 1),
            du_fit=du[::2].reshape(-1, 1),
            d2u_fit=d2u[::2].reshape(-1, 1),
            x_probe=x[1::2].reshape(-1, 1),
            u_probe=u[1::2].reshape(-1, 1),
            du_probe=du[1::2].reshape(-1, 1),
            d2u_probe=d2u[1::2].reshape(-1, 1),
        )
        groups.append(DEFeatureGroup(id=f"forced_{gid}", features=feats))

    cfg = DESearchConfig(
        x_axis=0, order_candidates=(2,), include_x=True, include_u=True, include_du=True
    )
    hp = default_physics_rescue_hp(preset="fast")
    hp.n_iter = 300
    hp.max_depth = 4
    rescue = FactorizedSearchDERescueConfig(
        mode="always",
        validate_integrate_topk=0,
        direct_generator_witness_topk=0,
        hp=hp,
    )
    res = run_direct_residual_fss_from_feature_groups(
        groups, cfg=cfg, rescue_cfg=rescue, verbose=False, attempt_phase="full"
    )

    assert res is not None
    # The seeded combo recovers the structure with the periodogram frequency
    # (<0.1% off); the exact polish is continuous skeleton refinement's job
    # at final_polish placement, not exercised in this small budget.
    assert float(res.probe_rms) < 0.1
    eq = str(res.canonical_equation)
    assert "cos((2.7" in eq and "u" in eq


def test_typed_x_lanes_get_frequency_hinted_coords(monkeypatch):
    """Regression (de126-class): a parametric drive coefficient cos(gamma*x)
    is invisible to the canonical frequency-1 sin/cos families; periodogram
    hints add gamma*x coords so the closed-form family fit recovers it."""
    monkeypatch.setattr(
        "nestynet_sr.sr_de.factorized_de.run_explorer", lambda **kwargs: []
    )

    x = torch.linspace(0.0, 25.0, 2400, dtype=torch.float64)
    u = torch.cos(1.3 * x) * torch.exp(-0.02 * x) + 1.5
    du = -1.3 * torch.sin(1.3 * x) * torch.exp(-0.02 * x) - 0.02 * (u - 1.5)
    d2u = -(4.0 + 2.0 * torch.cos(3.0 * x)) * u  # u'' + (4 + 2cos(3x))u = 0
    feats = DEFeatureTensors(
        x_fit=x[::2].reshape(-1, 1),
        u_fit=u[::2].reshape(-1, 1),
        du_fit=du[::2].reshape(-1, 1),
        d2u_fit=d2u[::2].reshape(-1, 1),
        x_probe=x[1::2].reshape(-1, 1),
        u_probe=u[1::2].reshape(-1, 1),
        du_probe=du[1::2].reshape(-1, 1),
        d2u_probe=d2u[1::2].reshape(-1, 1),
    )
    groups = [DEFeatureGroup(id="parametric_0", features=feats)]

    res = run_factorized_coeff_rescue_from_feature_groups(
        groups,
        cfg=DESearchConfig(
            x_axis=0, order_candidates=(2,), include_x=True, include_u=True, include_du=True
        ),
        rescue_cfg=FactorizedDERescueConfig(
            mode="always",
            hp=SimpleNamespace(n_iter=2000, max_depth=4, seed=0),
        ),
        dtype=torch.float64,
    )

    assert res is not None
    # The model-based Gauss-Newton polish converges the drive frequency well
    # below periodogram-bin resolution, so the family fit is near-exact.
    assert float(res.probe_rms) < 1.0e-4
    eq = str(res.canonical_equation)
    assert ("cos((3 * x0))" in eq or "cos((3.000" in eq) and "u" in eq
