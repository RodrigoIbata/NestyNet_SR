# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.sr_search.factorized_search import expr_mapping as expr_mapping_mod
from nestynet_sr.sr_search.factorized_search.engine import scoring as scoring_mod
from nestynet_sr.sr_search.factorized_search.engine.scoring import score_expr


def _pade_unit_problem():
    x_fit = torch.linspace(0.1, 0.7, 32, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(0.15, 0.75, 48, dtype=torch.float64).unsqueeze(-1)
    y_fit = 1.0 / (1.0 + x_fit)
    y_probe = 1.0 / (1.0 + x_probe)
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)
    return x_fit, y_fit, x_probe, y_probe, proj


def test_score_expr_leaves_pade_nonstructural_by_default(monkeypatch):
    def _fit_best_pade_stub(pred, y, poly_degree, **kwargs):
        mapping = {"kind": "pade", "numer": [1.0], "denom": [1.0, 1.0], "mu": 0.0, "std": 1.0}
        return None, mapping

    monkeypatch.setattr(scoring_mod, "fit_best", _fit_best_pade_stub)
    x_fit, y_fit, x_probe, y_probe, proj = _pade_unit_problem()

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        3,
        refine_enable=False,
        refine_cfg={"score_mapping_family_mode": "cheap"},
        return_expr=True,
    )

    assert sc is not None
    _, _, _, mapping, expr = sc
    assert expr == ("var", 0)
    assert mapping["kind"] == "pade"
    assert mapping["_acceptance_basis"] == "mapped"
    assert expr_mapping_mod.mapping_is_structural(mapping) is False


def test_score_expr_de_flag_compiles_low_order_pade_to_structural_rational(monkeypatch):
    def _fit_best_pade_stub(pred, y, poly_degree, **kwargs):
        mapping = {"kind": "pade", "numer": [1.0], "denom": [1.0, 1.0], "mu": 0.0, "std": 1.0}
        return None, mapping

    monkeypatch.setattr(scoring_mod, "fit_best", _fit_best_pade_stub)
    x_fit, y_fit, x_probe, y_probe, proj = _pade_unit_problem()

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        3,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "cheap",
            "score_pade_structural_enable": True,
        },
        return_expr=True,
    )

    assert sc is not None
    mse, _, _, mapping, expr = sc
    assert float(mse) < 1.0e-24
    assert expr[0] == "div"
    assert mapping["kind"] == "poly"
    assert mapping["_acceptance_basis"] == "typed_structural_rational"
    assert mapping["_compiled_from_mapping"]["kind"] == "pade"
    ladder = mapping["_score_ladder"]
    assert ladder["compiled_structural"]["accepted"] is True
    assert ladder["compiled_structural"]["source"] == "compiled_pade_mapping"


def test_score_expr_hidden_head_records_score_decomposition():
    x0_fit = torch.linspace(0.1, 1.0, 32, dtype=torch.float64)
    x0_probe = torch.linspace(0.15, 1.05, 48, dtype=torch.float64)
    x_fit = torch.stack(
        [
            x0_fit,
            x0_fit.square() + 0.2,
        ],
        dim=1,
    )
    x_probe = torch.stack(
        [
            x0_probe,
            x0_probe.square() + 0.2,
        ],
        dim=1,
    )
    y_fit = x_fit[:, :1] + x_fit[:, 1:2]
    y_probe = x_probe[:, :1] + x_probe[:, 1:2]
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        1,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "score_head_enable": True,
            "score_head_vars_enable": True,
            "score_head_var_terms": [("var", 1)],
            "score_head_direct_combo_enable": False,
        },
        return_expr=True,
    )

    assert sc is not None
    mapping = sc[3]
    assert isinstance(mapping.get("_lin_head"), dict)
    decomp = mapping.get("_score_decomp")
    assert isinstance(decomp, dict)
    assert decomp["mse_core"] > decomp["mse_with_head"]
    assert decomp["head_rel_gain"] > 0.0
    assert decomp["n_head_terms"] == 1
    assert decomp["head_energy_frac"] is not None


def test_score_expr_skips_negated_equiv_for_degree1_poly_only():
    x_fit = torch.linspace(-1.0, 0.5, 32, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-0.9, 0.7, 48, dtype=torch.float64).unsqueeze(-1)
    y_fit = 0.5 + 1.25 * x_fit
    y_probe = 0.5 + 1.25 * x_probe
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)
    diagnostics: dict[str, float] = {}

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        1,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "diagnostics": diagnostics,
        },
        return_expr=True,
    )

    assert sc is not None
    assert int(diagnostics.get("negated_variant_scores", 0)) == 0
    assert int(diagnostics.get("negated_variant_skipped_affine_poly_only", 0)) == 1


def test_score_expr_keeps_negated_equiv_outside_poly_only():
    x_fit = torch.linspace(-1.0, 0.5, 32, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-0.9, 0.7, 48, dtype=torch.float64).unsqueeze(-1)
    y_fit = 0.5 + 1.25 * x_fit
    y_probe = 0.5 + 1.25 * x_probe
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)
    diagnostics: dict[str, float] = {}

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        1,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "cheap",
            "diagnostics": diagnostics,
        },
        return_expr=True,
    )

    assert sc is not None
    assert int(diagnostics.get("negated_variant_scores", 0)) == 1
    assert int(diagnostics.get("negated_variant_skipped_affine_poly_only", 0)) == 0


def test_fit_best_cheap_mode_skips_expensive_families(monkeypatch):
    calls: list[str] = []

    def _fit_sine_stub(pred, y):
        calls.append("sine")
        return None

    def _fit_exp_stub(pred, y):
        calls.append("exp")
        return None

    monkeypatch.setattr(expr_mapping_mod, "fit_sine", _fit_sine_stub)
    monkeypatch.setattr(expr_mapping_mod, "fit_exp_mapping", _fit_exp_stub)

    pred = torch.linspace(0.1, 1.3, 32, dtype=torch.float64).unsqueeze(-1)
    y = pred + 0.2 * pred.square()

    fb = expr_mapping_mod.fit_best(pred, y, 4, family_mode="cheap")
    assert fb is not None
    assert calls == []


def test_fit_best_gated_mode_uses_periodic_hint(monkeypatch):
    calls: list[str] = []

    def _fit_sine_stub(pred, y):
        calls.append("sine")
        return None

    monkeypatch.setattr(expr_mapping_mod, "fit_sine", _fit_sine_stub)

    pred = torch.linspace(0.1, 1.7, 32, dtype=torch.float64).unsqueeze(-1)
    y = torch.sin(pred)

    expr_mapping_mod.fit_best(
        pred,
        y,
        4,
        family_mode="gated",
        family_hint="periodic",
        expensive_gate_best_mse=None,
        expensive_gate_rel_y=0.0,
    )
    assert "sine" in calls


def test_score_expr_prescreen_drops_candidate_before_full_fit(monkeypatch):
    calls: list[str] = []

    def _fit_best_stub(pred, y, poly_degree, **kwargs):
        calls.append(str(kwargs.get("family_mode", "")))
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        return None, mapping

    monkeypatch.setattr(scoring_mod, "fit_best", _fit_best_stub)

    x_fit = torch.linspace(0.1, 1.1, 32, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(0.2, 1.2, 48, dtype=torch.float64).unsqueeze(-1)
    y_fit = 50.0 * x_fit
    y_probe = 50.0 * x_probe
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        3,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "full",
            "score_prescreen_enable": True,
            "score_prescreen_family_mode": "cheap",
            "score_prescreen_parent_mse": 1.0e-6,
            "score_prescreen_global_best_mse": 1.0e-6,
            "score_prescreen_parent_best_factor": 1.0,
            "score_prescreen_global_best_factor": 1.0,
        },
    )

    assert sc is None
    assert calls
    assert set(calls) == {"cheap"}


def test_score_expr_force_full_bypasses_prescreen(monkeypatch):
    calls: list[str] = []

    def _fit_best_stub(pred, y, poly_degree, **kwargs):
        calls.append(str(kwargs.get("family_mode", "")))
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        return None, mapping

    monkeypatch.setattr(scoring_mod, "fit_best", _fit_best_stub)

    x_fit = torch.linspace(0.1, 1.1, 24, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(0.2, 1.2, 32, dtype=torch.float64).unsqueeze(-1)
    y_fit = x_fit
    y_probe = x_probe
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        3,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "full",
            "score_prescreen_enable": True,
            "score_prescreen_force_full": True,
            "score_prescreen_family_mode": "cheap",
            "score_prescreen_parent_mse": 1.0e-6,
            "score_prescreen_global_best_mse": 1.0e-6,
        },
    )

    assert sc is not None
    assert calls
    assert "full" in calls
    assert "cheap" not in calls


def test_score_expr_poly_only_mode_reaches_engine_scoring_fit_path(monkeypatch):
    calls: list[str] = []

    def _fit_best_stub(pred, y, poly_degree, **kwargs):
        calls.append(str(kwargs.get("family_mode", "")))
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        return None, mapping

    monkeypatch.setattr(scoring_mod, "fit_best", _fit_best_stub)

    x_fit = torch.linspace(0.1, 1.1, 32, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(0.2, 1.2, 48, dtype=torch.float64).unsqueeze(-1)
    y_fit = x_fit
    y_probe = x_probe
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)

    sc = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        3,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "score_prescreen_enable": True,
        },
    )

    assert sc is not None
    assert calls
    assert set(calls) == {"poly_only"}
