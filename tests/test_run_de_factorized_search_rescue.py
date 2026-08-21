# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nestynet_sr.run_de as run_de_mod
from nestynet_sr.run_de import (
    _build_factorized_search_only_attempts,
    _build_library_validation_candidate,
    _build_factorized_search_rescue_attempts,
    _prepare_factorized_search_feature_groups,
    _result_is_forced_only,
    _run_factorized_search_only_with_heuristics,
    _run_factorized_search_rescue_with_heuristics,
    _factorized_de_preferred,
    _should_fallback_from_full_factorized_search_only,
    _should_widen_from_restricted_rescue,
    build_factorized_search_rescue_config_from_args,
    build_factorized_rescue_config_from_args,
    candidate_probe_rms,
    choose_best_de_candidate,
    serialize_de_candidate,
    should_escalate_to_factorized_search,
    write_de_json_report,
)
from nestynet_sr.sr_core.bridges import Add, ConstNode, DU, U, Var
from nestynet_sr.sr_de import FactorizedSearchDEResult, DESearchConfig, FactorizedDEBlock, FactorizedDEResult
from nestynet_sr.sr_de.de_search import DESearchResult, DESearchResultMulti


def _make_args(**overrides):
    base = dict(
        device=None,
        num_segments=8,
        epochs=100,
        loss_target=1.0e-8,
        order_candidates="1,2",
        max_x_power=1,
        max_u_power=1,
        max_xu_total_degree=0,
        include_xdu=False,
        include_inv_xdu=False,
        include_inv_xu=False,
        include_inv_x2u=False,
        include_du=False,
        include_d2u=False,
        include_udu=False,
        stlsq_lambda=1.0e-3,
        sparsity_penalty=1.0e-3,
        enforce_units=False,
        units_policy=None,
        nn_units_semantics=None,
        stageb_refine_residual=False,
        stageb_epochs=0,
        factorized_rescue="never",
        factorized_two_block_shared_coord="never",
        factorized_search_rescue="never",
        factorized_search_trigger_val_rms=1.0e-3,
        factorized_search_trigger_rel_rms=1.0e-3,
        factorized_search_trigger_cond=1.0e8,
        factorized_search_replace_rel_factor=0.98,
        factorized_search_preset="default",
        factorized_search_n_iter=None,
        factorized_search_max_depth=None,
        factorized_search_n_fit=None,
        factorized_search_n_probe=None,
        factorized_search_return_topk=None,
        factorized_search_max_attempts=None,
        factorized_search_integrate_topk=None,
        factorized_search_direct_generator_witness_topk=1,
        factorized_de_whole_rhs="auto",
        factorized_de_typed_lane_workers=1,
        factorized_search_de_refine_mode="rare_final_polish",
        de_coe_mode="off",
        de_coe_csr_on_ties=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_primary_result(*, rms_train=1.0e-2, rms_val=2.0e-2, condition_number=1.0):
    return DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[None],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=float(rms_train),
        rms_val=float(rms_val) if rms_val is not None else None,
        residual_ast="lib_residual",
        condition_number=float(condition_number) if condition_number is not None else None,
    )


def _make_rescue_result(*, probe_rms=5.0e-3):
    return FactorizedSearchDEResult(
        order=1,
        x_axis=0,
        rhs_ast="rhs_ast",
        residual_ast="factorized_search_residual",
        canonical_equation="u_x - (rhs) = 0",
        probe_mse=float(probe_rms) ** 2,
        probe_rms=float(probe_rms),
        expr_ast=("var", 1),
        mapping={"kind": "poly", "coeffs": [0.0, -0.5]},
        mapping_kind="poly",
        feature_names=["x0", "u"],
        diagnostics={"score": float(probe_rms) ** 2},
    )


def _make_factorized_result(*, probe_rms=4.0e-3):
    nonanchor_ast = U()
    residual_ast = Add(DU(0), nonanchor_ast)
    shortlist_row = {
        "order": 1,
        "nonanchor_ast": nonanchor_ast,
        "residual_ast": residual_ast,
        "canonical_equation": "u_x + U() = 0",
        "probe_mse": float(probe_rms) ** 2,
        "probe_rms": float(probe_rms),
        "lane": "x_coeff_on_u",
        "family": "reciprocal",
        "base_mode": "zero",
        "evidence_tier": "verified",
        "witness_kind": "same_x_witness",
        "consistency_score": 1.0e-6,
        "consistency_pairs": 3,
        "consistency_total_pairs": 3,
        "collapse_score": 2.0e-6,
        "collapse_coverage": 1.0,
        "collapse_group_coverage": 1.0,
        "collapse_pairs": 3,
        "collapse_total_pairs": 3,
        "collapse_reason": "ok",
        "collapse_confidence": "high",
        "collapse_safe_rows": 12,
        "collapse_total_rows": 12,
        "collapse_domain_safe_fraction": 1.0,
        "collapse_within_bin_variance": 1.0e-8,
        "collapse_cross_trajectory_variance": 2.0e-8,
        "collapse_monotonic_support": 1.0,
        "collapse_sign_changes_mean": 0.0,
        "collapse_mixed_sign_fraction": 0.0,
        "carrier_ast": U(),
        "coord_ast": Var(0),
        "coeff_ast": Add(ConstNode(1.0), Var(0)),
        "coeff_expr": "1",
    }
    return FactorizedDEResult(
        order=1,
        x_axis=0,
        nonanchor_ast=nonanchor_ast,
        residual_ast=residual_ast,
        canonical_equation="u_x + U() = 0",
        probe_mse=float(probe_rms) ** 2,
        probe_rms=float(probe_rms),
        blocks=[
            FactorizedDEBlock(
                role="x_coeff_on_u",
                carrier_ast=U(),
                coord_ast=Var(0),
                coeff_ast=Var(0),
                block_ast=nonanchor_ast,
                diagnostics={},
            )
        ],
        diagnostics={
            "shortlist_rows": [shortlist_row],
            "selected_shortlist_rank": 0,
            "lane": "x_coeff_on_u",
            "family": "reciprocal",
            "base_mode": "zero",
            "evidence_tier": "verified",
            "witness_kind": "same_x_witness",
            "consistency_score": 1.0e-6,
            "consistency_pairs": 3,
            "consistency_total_pairs": 3,
            "collapse_score": 2.0e-6,
            "collapse_coverage": 1.0,
            "collapse_group_coverage": 1.0,
            "collapse_pairs": 3,
            "collapse_total_pairs": 3,
            "collapse_reason": "ok",
            "collapse_confidence": "high",
            "collapse_safe_rows": 12,
            "collapse_total_rows": 12,
            "collapse_domain_safe_fraction": 1.0,
            "collapse_within_bin_variance": 1.0e-8,
            "collapse_cross_trajectory_variance": 2.0e-8,
            "collapse_monotonic_support": 1.0,
            "collapse_sign_changes_mean": 0.0,
            "collapse_mixed_sign_fraction": 0.0,
            "zero_base_x_lane_diagnostics": {
                "best_by_ratio_probe_mse": {"family": "reciprocal", "ratio_probe_mse": 1.0e-6},
                "coord_reports": [
                    {
                        "coord_ast": "(1 + x0)",
                        "fit_target": {"n_rows": 8, "samples": [{"z": 1.0, "y": 1.0, "weight": 1.0}]},
                        "probe_target": {"n_rows": 8, "samples": [{"z": 2.0, "y": 0.5, "weight": 1.0}]},
                        "candidate_scores": [{"family": "reciprocal", "ratio_probe_mse": 1.0e-6}],
                    }
                ],
            },
        },
    )


def _disable_direct_residual_lane(monkeypatch):
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        lambda *args, **kwargs: None,
    )


def _attach_shortlist_report(rescue: FactorizedSearchDEResult) -> FactorizedSearchDEResult:
    best_row = {
        "order": 1,
        "score": 1.0e-6,
        "score_raw": 1.0e-6,
        "mse": float(rescue.probe_mse),
        "size": 1,
        "mapping_complexity": 1,
        "mapping_kind": "poly",
        "expr_ast": ("var", 1),
        "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
        "residual_ast": "u_x - (u) = 0",
        "expr": "u",
        "original_rank": 0,
        "score_rank": 0,
        "rerank_rank": 0,
        "integrate_ok": True,
        "integrate_mse": float(rescue.probe_mse),
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
    }
    alt_row = {
        "order": 1,
        "score": 1.0e-4,
        "score_raw": 1.0e-4,
        "mse": 2.0 * float(rescue.probe_mse),
        "size": 2,
        "mapping_complexity": 2,
        "mapping_kind": "poly",
        "expr_ast": ("var", 0),
        "mapping": {"kind": "poly", "coeffs": [0.0, -0.5]},
        "residual_ast": "u_x - (x) = 0",
        "expr": "x",
        "original_rank": 1,
        "score_rank": 1,
        "rerank_rank": 1,
        "integrate_ok": True,
        "integrate_mse": 2.0 * float(rescue.probe_mse),
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
    }
    rescue.diagnostics["report"] = {
        "x_axis": 0,
        "include_x": True,
        "include_u": True,
        "include_du": True,
        "constants_ordered": [],
        "extra": {"order_preference_factor": 1.0},
        "hp": {"return_topk": 2},
        "per_order": [
            {
                "order": 1,
                "feature_names": ["x0", "u"],
                "results": [best_row, alt_row],
            }
        ],
        "best": best_row,
        "trajectories": [{"id": "dataset0"}],
        "fit_trajectories": [{"id": "dataset0"}],
        "probe_trajectories": [{"id": "dataset0"}],
    }
    return rescue


class _ExpDecaySurrogate(torch.nn.Module):
    def __init__(self, decay: float):
        super().__init__()
        self.decay = float(decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.decay * x[:, :1])

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((int(x.shape[0]), int(x.shape[1])), dtype=x.dtype, device=x.device)
        out[:, 0] = -self.decay * torch.exp(-self.decay * x[:, 0])
        return out

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            (int(x.shape[0]), int(x.shape[1]), int(x.shape[1])),
            dtype=x.dtype,
            device=x.device,
        )
        out[:, 0, 0] = (self.decay ** 2) * torch.exp(-self.decay * x[:, 0])
        return out


def test_build_factorized_search_rescue_config_from_args_applies_overrides():
    args = _make_args(
        factorized_search_rescue="auto",
        factorized_search_preset="fast",
        factorized_search_trigger_rel_rms=2.0e-4,
        factorized_search_n_iter=123,
        factorized_search_max_depth=7,
        factorized_search_n_fit=456,
        factorized_search_n_probe=789,
        factorized_search_return_topk=4,
    )

    cfg = build_factorized_search_rescue_config_from_args(args)

    assert cfg.mode == "auto"
    assert float(cfg.trigger_val_rms) == pytest.approx(1.0e-3)
    assert float(cfg.trigger_rel_rms) == pytest.approx(2.0e-4)
    assert float(cfg.trigger_cond) == pytest.approx(1.0e8)
    assert int(cfg.hp.n_iter) == 123
    assert int(cfg.hp.max_depth) == 7
    assert int(cfg.hp.n_fit) == 456
    assert int(cfg.hp.n_probe) == 789
    assert int(cfg.hp.return_topk) == 4
    assert int(cfg.validate_integrate_topk) == 4
    assert cfg.budget_scope == "per_group"


def test_build_factorized_search_rescue_config_from_args_applies_budget_scope_override():
    cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="auto", factorized_search_budget_scope="global")
    )

    assert cfg.budget_scope == "global"


def test_build_factorized_search_rescue_config_from_args_applies_integrate_topk_override():
    off = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_return_topk=8, factorized_search_integrate_topk=0)
    )
    explicit = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_return_topk=8, factorized_search_integrate_topk=2)
    )

    assert int(off.validate_integrate_topk) == 0
    assert int(explicit.validate_integrate_topk) == 2


def test_build_factorized_search_rescue_config_from_args_applies_max_attempts_override():
    cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="auto", factorized_search_max_attempts=1)
    )

    assert cfg.max_attempts == 1


def test_build_factorized_rescue_config_from_args_reuses_shared_thresholds():
    args = _make_args(
        factorized_rescue="auto",
        factorized_two_block_shared_coord="always",
        factorized_search_preset="fast",
        factorized_search_trigger_val_rms=2.5e-3,
        factorized_search_trigger_cond=9.0e7,
        factorized_search_replace_rel_factor=0.95,
        factorized_search_n_iter=123,
        factorized_search_max_depth=7,
        factorized_search_n_fit=456,
        factorized_search_n_probe=789,
        factorized_search_return_topk=4,
        factorized_de_typed_lane_workers=3,
    )

    cfg = build_factorized_rescue_config_from_args(args)

    assert cfg.mode == "auto"
    assert float(cfg.trigger_val_rms) == pytest.approx(2.5e-3)
    assert float(cfg.trigger_cond) == pytest.approx(9.0e7)
    assert float(cfg.replace_rel_factor) == pytest.approx(0.95)
    assert cfg.two_block_shared_coord_mode == "always"
    assert int(cfg.typed_lane_workers) == 3
    assert int(cfg.hp.n_iter) == 123
    assert int(cfg.hp.max_depth) == 7
    assert int(cfg.hp.n_fit) == 456
    assert int(cfg.hp.n_probe) == 789
    assert int(cfg.hp.return_topk) == 4
    assert int(cfg.hp.n_iter) == int(build_factorized_search_rescue_config_from_args(args).hp.n_iter)


def test_factorized_de_config_preserves_explicit_two_block_never():
    cfg = build_factorized_rescue_config_from_args(
        _make_args(factorized_rescue="always", factorized_two_block_shared_coord="never")
    )

    out = run_de_mod._as_factorized_de_factorized_config(cfg)

    assert out.mode == "always"
    assert out.base_modes == ("zero",)
    assert out.two_block_shared_coord_mode == "never"


def test_build_factorized_search_rescue_config_from_args_compositional_preset_enables_sparse_combo():
    cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="auto", factorized_search_preset="compositional")
    )

    assert bool(getattr(cfg.hp, "de_sparse_combo_enable", False)) is True
    assert int(getattr(cfg.hp, "de_sparse_combo_pool_topk", 0)) >= 2
    assert int(getattr(cfg.hp, "de_sparse_combo_max_terms", 0)) == 4
    assert bool(getattr(cfg.hp, "score_head_enable", True)) is False
    assert bool(getattr(cfg.hp, "de_score_head_untyped_enable", True)) is False
    assert bool(getattr(cfg.hp, "de_accept_hidden_score_head", True)) is False


def test_build_factorized_search_rescue_config_from_args_compositional_fast_keeps_fast_budget():
    cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="auto", factorized_search_preset="compositional_fast")
    )

    assert bool(getattr(cfg.hp, "de_sparse_combo_enable", False)) is True
    assert bool(getattr(cfg.hp, "score_head_enable", True)) is False
    assert bool(getattr(cfg.hp, "de_score_head_untyped_enable", True)) is False
    assert int(cfg.hp.n_iter) == 15_000
    assert int(cfg.hp.n_fit) == 2_000
    assert int(cfg.hp.n_probe) == 2_000
    assert int(cfg.hp.return_topk) == 16
    assert int(getattr(cfg.hp, "de_sparse_combo_pool_topk", 0)) == 8


def test_build_factorized_search_rescue_config_applies_de_refine_mode():
    cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="auto", factorized_search_de_refine_mode="off")
    )

    assert cfg.hp.refine_profile == "off"
    assert bool(cfg.hp.refine_enable) is False


def test_factorized_de_whole_rhs_never_skips_broad_fss(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    factorized = _make_factorized_result(probe_rms=5.0e-4)
    calls: list[str] = []

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: factorized,
    )

    def _unexpected_fss(**kwargs):
        calls.append("fss")
        return _make_rescue_result(probe_rms=1.0e-4)

    monkeypatch.setattr(run_de_mod, "_run_factorized_search_only_with_heuristics", _unexpected_fss)

    selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
        typed_lanes_policy="always",
    )

    assert calls == []
    assert selected is factorized
    assert factorized_res is factorized
    assert rescue_res is None
    assert selected_engine == "factorized"
    assert whole_rhs_diag["run"] is False
    assert whole_rhs_diag["reason"] == "policy_never"


def test_factorized_de_whole_rhs_always_preserves_broad_fss(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    factorized = _make_factorized_result(probe_rms=5.0e-4)
    rescue = _make_rescue_result(probe_rms=1.0e-4)
    calls: list[str] = []

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: factorized,
    )

    def _fake_fss(**kwargs):
        assert kwargs["rescue_cfg"].max_attempts is None
        calls.append("fss")
        return rescue

    monkeypatch.setattr(run_de_mod, "_run_factorized_search_only_with_heuristics", _fake_fss)

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="always",
        typed_lanes_policy="always",
    )

    assert calls == ["fss"]
    assert selected is rescue
    assert rescue_res is rescue
    assert selected_engine == "factorized_search"
    assert whole_rhs_diag["run"] is True
    assert whole_rhs_diag["reason"] == "policy_always"
    assert whole_rhs_diag["max_attempts"] is None


def test_factorized_de_auto_skips_when_typed_lane_is_good(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    factorized = _make_factorized_result(probe_rms=5.0e-4)
    calls: list[str] = []

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: factorized,
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: calls.append("fss") or _make_rescue_result(probe_rms=1.0e-4),
    )

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="auto",
        typed_lanes_policy="always",
    )

    assert calls == []
    assert selected is factorized
    assert rescue_res is None
    assert selected_engine == "factorized"
    assert whole_rhs_diag["reason"] == "typed_probe_rms_pass"
    assert whole_rhs_diag["max_attempts"] == 1


def test_factorized_de_auto_skips_when_relative_rms_is_good(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    factorized = _make_factorized_result(probe_rms=1.4e-3)
    factorized.diagnostics["probe_rel_rms"] = 3.0e-5
    calls: list[str] = []

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: factorized,
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: calls.append("fss") or _make_rescue_result(probe_rms=1.0e-4),
    )

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="auto",
        typed_lanes_policy="always",
    )

    assert calls == []
    assert selected is factorized
    assert rescue_res is None
    assert selected_engine == "factorized"
    assert whole_rhs_diag["reason"] == "typed_probe_rel_rms_pass"
    assert whole_rhs_diag["typed_probe_rel_rms"] == pytest.approx(3.0e-5)


def test_factorized_de_auto_skips_when_direct_generator_witness_passes(monkeypatch):
    cfg = DESearchConfig(order_candidates=(2,), include_x=True, include_u=True, include_du=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-8)
    )
    direct = _make_rescue_result(probe_rms=1.3e-1)
    direct.order = 2
    direct.feature_names = ["u", "du"]
    direct.diagnostics["probe_rel_rms"] = 2.8e-2
    direct.diagnostics["generator_status"] = "EXACT_STRUCTURAL_GENERATOR"
    direct.diagnostics["evidence_tier"] = "generator_witness"
    direct.diagnostics["generator_witness"] = {
        "generator_status": "EXACT_STRUCTURAL_GENERATOR",
        "rollout_u_nrmse": 1.0e-3,
    }
    direct_calls: list[str] = []

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])

    def _fake_direct(*args, **kwargs):
        direct_calls.append(str(kwargs.get("attempt_phase", "")))
        return direct

    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        _fake_direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_regularized_implicit_residual_fss_from_feature_groups",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: pytest.fail("typed lanes should be skipped after exact generator witness"),
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: pytest.fail("whole-RHS fallback should be skipped after generator witness"),
    )

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="auto",
        typed_lanes_policy="auto",
    )

    assert selected is direct
    assert direct_calls == ["autonomous"]
    assert rescue_res is None
    assert selected_engine == "factorized_search"
    assert direct.diagnostics["factorized_de"]["typed_lanes_attempted"] is False
    assert direct.diagnostics["factorized_de"]["typed_lanes_decision"]["reason"] == "policy_auto_first_line_clean"
    assert whole_rhs_diag["run"] is False
    assert whole_rhs_diag["reason"] == "generator_witness_pass"
    assert whole_rhs_diag["typed_generator_status"] == "EXACT_STRUCTURAL_GENERATOR"


def test_factorized_de_typed_lanes_always_runs_despite_clean_first_line(monkeypatch):
    """'always' means always: the cleanliness gate only applies to 'auto'."""
    cfg = DESearchConfig(order_candidates=(2,), include_x=True, include_u=True, include_du=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-8)
    )
    direct = _make_rescue_result(probe_rms=1.3e-1)
    direct.order = 2
    direct.feature_names = ["u", "du"]
    direct.diagnostics["probe_rel_rms"] = 2.8e-2
    direct.diagnostics["generator_status"] = "EXACT_STRUCTURAL_GENERATOR"
    direct.diagnostics["evidence_tier"] = "generator_witness"
    direct.diagnostics["generator_witness"] = {
        "generator_status": "EXACT_STRUCTURAL_GENERATOR",
        "rollout_u_nrmse": 1.0e-3,
    }
    typed_calls: list[str] = []

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        lambda *args, **kwargs: direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_regularized_implicit_residual_fss_from_feature_groups",
        lambda *args, **kwargs: None,
    )

    def _fake_typed(*args, **kwargs):
        typed_calls.append("typed")
        return None

    monkeypatch.setattr(run_de_mod, "run_factorized_coeff_rescue_from_feature_groups", _fake_typed)

    selected, _, _, selected_engine, _ = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
        typed_lanes_policy="always",
    )

    assert typed_calls == ["typed"]
    assert selected is direct
    assert selected_engine == "factorized_search"
    assert direct.diagnostics["factorized_de"]["typed_lanes_attempted"] is True
    assert direct.diagnostics["factorized_de"]["typed_lanes_decision"]["reason"] == "policy_always"


def test_factorized_de_auto_challenges_dynamically_compatible_direct_witness(monkeypatch):
    cfg = DESearchConfig(order_candidates=(2,), include_x=True, include_u=True, include_du=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-8)
    )
    direct = _make_rescue_result(probe_rms=1.3e-1)
    direct.order = 2
    direct.feature_names = ["u", "du"]
    direct.diagnostics["probe_rel_rms"] = 2.8e-2
    direct.diagnostics["generator_status"] = "DYNAMICALLY_COMPATIBLE"
    direct.diagnostics["evidence_tier"] = "generator_witness"
    direct.diagnostics["generator_witness"] = {
        "generator_status": "DYNAMICALLY_COMPATIBLE",
        "rollout_u_nrmse": 1.0e-3,
    }
    typed = _make_factorized_result(probe_rms=5.0e-1)
    rescue = _make_rescue_result(probe_rms=1.0e-1)
    calls: list[str] = []

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        lambda *args, **kwargs: direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: calls.append("typed") or typed,
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: calls.append("fss") or rescue,
    )

    selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="auto",
        typed_lanes_policy="auto",
    )

    assert calls == ["typed", "fss"]
    assert factorized_res is typed
    assert rescue_res is rescue
    assert selected is direct
    assert selected_engine == "factorized_search"
    assert whole_rhs_diag["run"] is True
    assert whole_rhs_diag["reason"] != "generator_witness_pass"


def test_factorized_de_auto_runs_when_typed_lane_is_ambiguous(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    factorized = _make_factorized_result(probe_rms=5.0e-2)
    factorized.diagnostics["evidence_tier"] = ""
    rescue = _make_rescue_result(probe_rms=1.0e-4)
    calls: list[str] = []

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: factorized,
    )

    def _fake_fss(**kwargs):
        assert kwargs["rescue_cfg"].max_attempts == 1
        calls.append("fss")
        return rescue

    monkeypatch.setattr(run_de_mod, "_run_factorized_search_only_with_heuristics", _fake_fss)

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="auto",
        typed_lanes_policy="always",
    )

    assert calls == ["fss"]
    assert selected is rescue
    assert rescue_res is rescue
    assert selected_engine == "factorized_search"
    assert whole_rhs_diag["reason"] == "typed_candidate_ambiguous"
    assert whole_rhs_diag["max_attempts"] == 1


def test_factorized_de_auto_does_not_treat_unverified_as_verified(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    factorized = _make_factorized_result(probe_rms=5.0e-3)
    factorized.diagnostics["evidence_tier"] = "unverified"
    factorized.diagnostics["consistency_score"] = 1.0
    rescue = _make_rescue_result(probe_rms=1.0e-4)
    calls: list[str] = []

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: factorized,
    )

    def _fake_fss(**kwargs):
        calls.append("fss")
        return rescue

    monkeypatch.setattr(run_de_mod, "_run_factorized_search_only_with_heuristics", _fake_fss)

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="auto",
        typed_lanes_policy="always",
    )

    assert calls == ["fss"]
    assert selected is rescue
    assert rescue_res is rescue
    assert selected_engine == "factorized_search"
    assert whole_rhs_diag["reason"] == "typed_candidate_ambiguous"


def test_factorized_de_whole_rhs_max_attempts_zero_skips_broad_fss(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_max_attempts=0)
    )
    factorized = _make_factorized_result(probe_rms=5.0e-2)
    factorized.diagnostics["evidence_tier"] = ""
    calls: list[str] = []

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: factorized,
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: calls.append("fss") or _make_rescue_result(probe_rms=1.0e-4),
    )

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="always",
        typed_lanes_policy="always",
    )

    assert calls == []
    assert selected is factorized
    assert rescue_res is None
    assert selected_engine == "factorized"
    assert whole_rhs_diag["run"] is False
    assert whole_rhs_diag["reason"] == "max_attempts_zero"
    assert whole_rhs_diag["max_attempts"] == 0


def test_factorized_de_prefers_direct_residual_lane_by_default(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    direct = _make_rescue_result(probe_rms=1.0e-4)
    typed_calls: list[str] = []

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        lambda *args, **kwargs: direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: typed_calls.append("typed") or _make_factorized_result(probe_rms=5.0e-4),
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: _make_rescue_result(probe_rms=5.0e-5),
    )

    selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
    )

    assert typed_calls == []
    assert selected is direct
    assert factorized_res is None
    assert rescue_res is None
    assert selected_engine == "factorized_search"
    assert whole_rhs_diag["reason"] == "policy_never"
    assert selected.diagnostics["factorized_de"]["selected_lane"] == "direct_residual_fss"
    assert selected.diagnostics["factorized_de"]["typed_lanes_policy"] == "never"


def test_factorized_de_implicit_first_line_suppresses_typed_fallback(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-3)
    )
    direct = _make_rescue_result(probe_rms=4.0e-2)
    direct.diagnostics["probe_rel_rms"] = 5.0e-2
    direct.diagnostics["factorized_de_diagnostics"] = {
        "orders": [
            {
                "order": 2,
                "search_diagnostics": {
                    "additive_fss": {
                        "enabled": True,
                        "pre_mutation_context_rows": 11,
                        "post_mutation_context_rows": 7,
                        "pre_mutation_contextual_atom_diagnostics": {
                            "promoted_rows": 11,
                            "trace": [{"atom": "cos((1.75*x0))", "promoted": True}],
                        },
                    }
                },
            }
        ],
        "search_diagnostics_summary": {
            "n_orders": 1,
            "additive_fss_orders": 1,
            "pre_mutation_context_rows": 11,
            "post_mutation_context_rows": 7,
        },
    }
    implicit = _make_rescue_result(probe_rms=5.9e-3)
    implicit.diagnostics["candidate_source"] = "regularized_implicit_residual"
    implicit.diagnostics["probe_rel_rms"] = 5.9e-3
    implicit.diagnostics["implicit_residual"] = {
        "a_expr": "x0",
        "b_exprs": ["u"],
        "b_coeff_source": "separable_invariant_refit",
        "b_coeffs": [1.0016],
        "normalized_probe_score": 5.8e-3,
        "multiplier": {"ok": True, "sign_ok": True, "nonzero_frac": 1.0},
        "invariant_refit": {
            "kind": "separable_invariant",
            "coeffs": [1.0016],
            "probe_score": 5.8e-3,
            "fit_coeff_spread_rel": 0.01,
            "probe_coeff_spread_rel": 0.02,
            "probe_traj": [{"id": "ic4", "score": 0.006}, {"id": "ic5", "score": 0.007}],
        },
    }
    calls: list[str] = []
    direct_phases: list[str] = []

    def _direct(*args, **kwargs):
        direct_phases.append(str(kwargs.get("attempt_phase", "")))
        return direct

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        _direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_regularized_implicit_residual_fss_from_feature_groups",
        lambda *args, **kwargs: implicit,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: calls.append("typed") or _make_factorized_result(probe_rms=3.0e-2),
    )

    selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
        typed_lanes_policy="auto",
    )

    assert calls == []
    assert direct_phases == ["autonomous"]
    assert selected is implicit
    assert factorized_res is None
    assert rescue_res is None
    assert selected_engine == "factorized_search"
    assert whole_rhs_diag["reason"] == "policy_never"
    lane_diag = selected.diagnostics["factorized_de"]
    assert lane_diag["selected_lane"] == "regularized_implicit_residual"
    assert lane_diag["typed_lanes_decision"]["direct_needs_typed"] is True
    assert lane_diag["typed_lanes_decision"]["first_line_needs_typed"] is False
    assert lane_diag["typed_lanes_decision"]["full_direct_attempted"] is False
    assert lane_diag["full_direct_residual_attempted"] is False
    assert lane_diag["typed_lanes_attempted"] is False
    assert lane_diag["direct_residual_search_diagnostics_summary"]["pre_mutation_context_rows"] == 11
    direct_search_diag = lane_diag["direct_residual_search_diagnostics"]
    additive_diag = direct_search_diag["orders"][0]["search_diagnostics"]["additive_fss"]
    assert additive_diag["pre_mutation_contextual_atom_diagnostics"]["trace"][0]["atom"] == "cos((1.75*x0))"


def test_factorized_de_always_typed_promotes_implicit_to_rollout_slate(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-3)
    )
    direct = _make_rescue_result(probe_rms=4.0e-2)
    direct.diagnostics["probe_rel_rms"] = 5.0e-2
    implicit = _make_rescue_result(probe_rms=5.9e-3)
    implicit.diagnostics["candidate_source"] = "regularized_implicit_residual"
    implicit.diagnostics["probe_rel_rms"] = 5.9e-3
    implicit.diagnostics["implicit_residual"] = {
        "a_expr": "x0",
        "b_exprs": ["u"],
        "b_coeff_source": "separable_invariant_refit",
        "b_coeffs": [1.0016],
        "normalized_probe_score": 5.8e-3,
        "multiplier": {"ok": True, "sign_ok": True, "nonzero_frac": 1.0},
        "invariant_refit": {
            "kind": "separable_invariant",
            "coeffs": [1.0016],
            "probe_score": 5.8e-3,
            "fit_coeff_spread_rel": 0.01,
            "probe_coeff_spread_rel": 0.02,
            "probe_traj": [{"id": "ic4", "score": 0.006}, {"id": "ic5", "score": 0.007}],
        },
    }
    typed = _make_factorized_result(probe_rms=5.0e-4)

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        lambda *args, **kwargs: direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_regularized_implicit_residual_fss_from_feature_groups",
        lambda *args, **kwargs: implicit,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: typed,
    )

    selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
        typed_lanes_policy="always",
    )

    assert selected is typed
    assert factorized_res is typed
    assert rescue_res is None
    assert selected_engine == "factorized"
    assert whole_rhs_diag["reason"] == "policy_never"
    aux = typed.diagnostics["auxiliary_rollout_candidates"]
    assert len(aux) == 1
    assert aux[0]["source_lane"] == "regularized_implicit_residual"
    assert aux[0]["auxiliary_first_line"] is True
    assert aux[0]["first_line_certified"] is True

    payload = serialize_de_candidate(typed)
    assert payload["auxiliary_rollout_candidates"][0]["source_lane"] == "regularized_implicit_residual"


def test_factorized_de_always_typed_promotes_dynamic_direct_witness_to_rollout_slate(monkeypatch):
    cfg = DESearchConfig(order_candidates=(2,), include_x=True, include_u=True, include_du=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-3)
    )
    direct = _make_rescue_result(probe_rms=1.6e-1)
    direct.order = 2
    direct.feature_names = ["u", "du"]
    direct.mapping = {"kind": "poly", "coeffs": [0.1, -8.9], "mu": 0.0, "std": 1.0}
    direct.diagnostics["probe_rel_rms"] = 2.7e-2
    direct.diagnostics["generator_status"] = "DYNAMICALLY_COMPATIBLE"
    direct.diagnostics["evidence_tier"] = "generator_witness"
    direct.diagnostics["witness_materialized"] = True
    direct.diagnostics["generator_witness"] = {
        "generator_status": "DYNAMICALLY_COMPATIBLE",
        "rollout_u_nrmse": 0.014,
    }
    direct.diagnostics["shortlist_union"] = [
        {
            "engine": "factorized_search",
            "kind": "factorized",
            "order": 2,
            "x_axis": 0,
            "expr_ast": ("var", 0),
            "mapping": {"kind": "poly", "coeffs": [0.1, -8.9], "mu": 0.0, "std": 1.0},
            "mapping_kind": "poly",
            "canonical_equation": "direct_dynamic",
            "source_lane": "direct_residual_fss",
            "generator_status": "DYNAMICALLY_COMPATIBLE",
        }
    ]
    implicit = _make_rescue_result(probe_rms=6.0e-3)
    implicit.diagnostics["candidate_source"] = "regularized_implicit_residual"
    implicit.diagnostics["implicit_residual"] = {
        "a_expr": "x0",
        "b_exprs": ["u"],
        "b_coeff_source": "separable_invariant_refit",
        "b_coeffs": [1.0],
        "normalized_probe_score": 5.0e-3,
        "multiplier": {"ok": True, "sign_ok": True, "nonzero_frac": 1.0},
        "invariant_refit": {
            "kind": "separable_invariant",
            "coeffs": [1.0],
            "probe_score": 5.0e-3,
            "fit_coeff_spread_rel": 0.01,
            "probe_coeff_spread_rel": 0.02,
            "probe_traj": [{"id": "ic4", "score": 0.005}],
        },
    }
    typed = _make_factorized_result(probe_rms=5.0e-4)

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        lambda *args, **kwargs: direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_regularized_implicit_residual_fss_from_feature_groups",
        lambda *args, **kwargs: implicit,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: typed,
    )

    selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
        typed_lanes_policy="always",
    )

    assert selected is typed
    assert factorized_res is typed
    assert rescue_res is None
    assert selected_engine == "factorized"
    assert whole_rhs_diag["reason"] == "policy_never"
    aux = typed.diagnostics["auxiliary_rollout_candidates"]
    assert [row["source_lane"] for row in aux] == [
        "direct_residual_fss",
        "regularized_implicit_residual",
    ]
    assert aux[0]["generator_status"] == "DYNAMICALLY_COMPATIBLE"
    assert aux[0]["generator_witness_promoted"] is True
    assert aux[0]["first_line_certified"] is False

    payload = serialize_de_candidate(typed)
    aux_payload = payload["auxiliary_rollout_candidates"]
    assert [row["source_lane"] for row in aux_payload] == [
        "direct_residual_fss",
        "regularized_implicit_residual",
    ]
    assert aux_payload[0]["shortlist"][0]["mapping"]["coeffs"] == pytest.approx([0.1, -8.9])


def test_factorized_de_invariant_gate_rejects_unstable_coefficients(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-3)
    )
    direct = _make_rescue_result(probe_rms=4.0e-2)
    direct.diagnostics["probe_rel_rms"] = 5.0e-2
    direct_full = _make_rescue_result(probe_rms=3.0e-3)
    implicit = _make_rescue_result(probe_rms=5.9e-3)
    implicit.diagnostics["candidate_source"] = "regularized_implicit_residual"
    implicit.diagnostics["probe_rel_rms"] = 5.9e-3
    implicit.diagnostics["implicit_residual"] = {
        "a_expr": "x0",
        "b_exprs": ["u"],
        "b_coeff_source": "separable_invariant_refit",
        "b_coeffs": [1.0],
        "normalized_probe_score": 5.0e-3,
        "multiplier": {"ok": True, "sign_ok": True, "nonzero_frac": 1.0},
        "invariant_refit": {
            "kind": "separable_invariant",
            "coeffs": [1.0],
            "probe_score": 5.0e-3,
            "fit_coeff_spread_rel": 0.5,
            "probe_coeff_spread_rel": 0.5,
            "probe_traj": [{"id": "ic4", "score": 0.006}, {"id": "ic5", "score": 0.007}],
        },
    }
    direct_phases: list[str] = []

    def _direct(*args, **kwargs):
        phase = str(kwargs.get("attempt_phase", ""))
        direct_phases.append(phase)
        return direct_full if phase == "full" else direct

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        _direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_regularized_implicit_residual_fss_from_feature_groups",
        lambda *args, **kwargs: implicit,
    )

    selected, _, _, _, _ = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
        typed_lanes_policy="never",
    )

    assert direct_phases == ["autonomous", "full"]
    assert selected is direct_full
    assert selected.diagnostics["factorized_de"]["full_direct_residual_attempted"] is True


def test_factorized_de_auto_runs_typed_lane_for_finite_dubious_direct(monkeypatch):
    cfg = DESearchConfig(order_candidates=(2,), include_x=True, include_u=True, include_du=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_trigger_val_rms=1.0e-3)
    )
    direct = _make_rescue_result(probe_rms=4.0e-2)
    direct.order = 2
    direct.diagnostics["probe_rel_rms"] = 5.0e-2
    typed = _make_factorized_result(probe_rms=3.0e-2)
    calls: list[str] = []

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_direct_residual_fss_from_feature_groups",
        lambda *args, **kwargs: direct,
    )
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: calls.append("typed") or typed,
    )

    selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="never",
        typed_lanes_policy="auto",
    )

    assert calls == ["typed"]
    assert selected is typed
    assert factorized_res is typed
    assert rescue_res is None
    assert selected_engine == "factorized"
    assert whole_rhs_diag["reason"] == "policy_never"


def test_factorized_de_prefers_clean_typed_over_comparable_broad_fss():
    typed = _make_factorized_result(probe_rms=2.0e-2)
    broad = _make_rescue_result(probe_rms=1.2e-2)

    assert _factorized_de_preferred(broad, "factorized_search", typed, "factorized") is False
    assert _factorized_de_preferred(typed, "factorized", broad, "factorized_search") is True

    much_better_broad = _make_rescue_result(probe_rms=8.0e-3)
    assert _factorized_de_preferred(much_better_broad, "factorized_search", typed, "factorized") is True


def test_factorized_de_rejects_domain_failed_whole_rhs_candidate(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    typed = _make_factorized_result(probe_rms=5.0e-2)
    rescue = _make_rescue_result(probe_rms=1.0e-6)
    rescue.diagnostics["domain_ok"] = False

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: typed,
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: rescue,
    )

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="always",
        typed_lanes_policy="always",
    )

    assert selected is typed
    assert rescue_res is rescue
    assert selected_engine == "factorized"
    assert whole_rhs_diag["reason"] == "policy_always"


def test_factorized_de_rejects_structurally_failed_whole_rhs_candidate(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    factorized_cfg = build_factorized_rescue_config_from_args(_make_args(factorized_rescue="always"))
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    typed = _make_factorized_result(probe_rms=5.0e-2)
    rescue = _make_rescue_result(probe_rms=1.0e-6)
    rescue.diagnostics["structural_ok"] = False
    rescue.diagnostics["structural_hard_reject"] = True
    rescue.diagnostics["structural_reasons"] = ["log_nonpositive_constant"]

    _disable_direct_residual_lane(monkeypatch)
    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["fg"])
    monkeypatch.setattr(
        run_de_mod,
        "run_factorized_coeff_rescue_from_feature_groups",
        lambda *args, **kwargs: typed,
    )
    monkeypatch.setattr(
        run_de_mod,
        "_run_factorized_search_only_with_heuristics",
        lambda **kwargs: rescue,
    )

    selected, _, rescue_res, selected_engine, whole_rhs_diag = run_de_mod._run_factorized_de(
        cfg=cfg,
        factorized_cfg=factorized_cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        surrogate_val_losses=[1.0e-6],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        whole_rhs_policy="always",
        typed_lanes_policy="always",
    )

    assert selected is typed
    assert rescue_res is rescue
    assert selected_engine == "factorized"
    assert whole_rhs_diag["reason"] == "policy_always"


def test_should_escalate_to_factorized_search_auto_triggers_on_rms_and_condition_number():
    cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="auto"))

    bad_rms = _make_primary_result(rms_val=2.0e-3, condition_number=1.0)
    bad_cond = _make_primary_result(rms_val=1.0e-6, condition_number=2.0e8)
    good = _make_primary_result(rms_val=1.0e-6, condition_number=5.0)

    assert should_escalate_to_factorized_search(bad_rms, cfg) is True
    assert should_escalate_to_factorized_search(bad_cond, cfg) is True
    assert should_escalate_to_factorized_search(good, cfg) is False


def test_build_factorized_search_rescue_attempts_default_is_not_stlsq_conditioned():
    primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )
    cfg = DESearchConfig(order_candidates=(1, 2), include_x=True, include_u=True, include_du=True)

    attempts = _build_factorized_search_rescue_attempts(primary, cfg)

    assert attempts[0]["name"] == "full"
    assert attempts[0]["cfg"].order_candidates == (1, 2)
    assert attempts[0]["cfg"].include_x is True
    assert attempts[0]["cfg"].include_u is True
    assert attempts[0]["conditioned_on_primary"] is False
    assert all("no_x" not in attempt["constraints"] for attempt in attempts)


def test_build_factorized_search_rescue_attempts_legacy_restricts_order_and_autonomous_no_x():
    primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )
    cfg = DESearchConfig(order_candidates=(1, 2), include_x=True)

    attempts = _build_factorized_search_rescue_attempts(primary, cfg, use_primary_constraints=True)

    assert len(attempts) == 2
    assert attempts[0]["name"] == "restricted"
    assert attempts[0]["cfg"].order_candidates == (1,)
    assert attempts[0]["cfg"].include_x is False
    assert attempts[0]["constraints"] == ["order=1", "no_x"]
    assert attempts[1]["name"] == "full"
    assert attempts[1]["cfg"].order_candidates == (1, 2)
    assert attempts[1]["cfg"].include_x is True


def test_build_factorized_search_rescue_attempts_legacy_keeps_x_for_nonautonomous_result():
    primary = DESearchResult(
        order=2,
        x_axis=0,
        term_asts=[Var(0)],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )
    cfg = DESearchConfig(order_candidates=(1, 2), include_x=True)

    attempts = _build_factorized_search_rescue_attempts(primary, cfg, use_primary_constraints=True)

    assert len(attempts) == 2
    assert attempts[0]["cfg"].order_candidates == (2,)
    assert attempts[0]["cfg"].include_x is True
    assert attempts[0]["cfg"].include_u is False
    assert attempts[0]["constraints"] == ["order=2", "no_u"]


def test_result_is_forced_only_detects_x_only_support():
    primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[Var(0)],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )

    assert _result_is_forced_only(primary) is True


def test_build_factorized_search_rescue_attempts_legacy_restricts_for_forced_only_result():
    primary = DESearchResult(
        order=2,
        x_axis=0,
        term_asts=[Var(0)],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )
    cfg = DESearchConfig(order_candidates=(1, 2), include_x=True, include_u=True, include_du=True)

    attempts = _build_factorized_search_rescue_attempts(primary, cfg, use_primary_constraints=True)

    assert len(attempts) == 2
    assert attempts[0]["cfg"].order_candidates == (2,)
    assert attempts[0]["cfg"].include_x is True
    assert attempts[0]["cfg"].include_u is False
    assert attempts[0]["cfg"].include_du is False
    assert attempts[0]["constraints"] == ["order=2", "no_u", "no_du"]


def test_build_factorized_search_only_attempts_tries_full_before_singletons():
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)

    attempts = _build_factorized_search_only_attempts(cfg)

    assert [attempt["name"] for attempt in attempts] == ["full", "order1_x", "order1_u"]
    assert attempts[0]["cfg"].include_x is True
    assert attempts[0]["cfg"].include_u is True
    assert attempts[1]["cfg"].include_x is True
    assert attempts[1]["cfg"].include_u is False
    assert attempts[2]["cfg"].include_x is False
    assert attempts[2]["cfg"].include_u is True


def test_should_fallback_from_full_factorized_search_only_only_when_full_attempt_is_bad():
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))

    good = _make_rescue_result(probe_rms=5.0e-3)
    good.diagnostics = {"domain_ok": True, "integrate_ok": True, "integrate_mse": 1.0e-8}

    bad = _make_rescue_result(probe_rms=2.0e-1)
    bad.diagnostics = {"domain_ok": True, "integrate_ok": False, "integrate_mse": 1.0e-2}

    assert _should_fallback_from_full_factorized_search_only(good, rescue_cfg) is False
    assert _should_fallback_from_full_factorized_search_only(bad, rescue_cfg) is True


def test_should_widen_from_restricted_rescue_only_on_bad_failure():
    cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="auto"))
    primary = _make_primary_result(rms_val=1.0e-2)

    restricted_good = _make_rescue_result(probe_rms=5.0e-3)
    restricted_meh = _make_rescue_result(probe_rms=1.5e-2)
    restricted_bad = _make_rescue_result(probe_rms=5.0e-1)
    restricted_inf = _make_rescue_result(probe_rms=float("inf"))

    assert _should_widen_from_restricted_rescue(restricted_good, primary, cfg) is False
    assert _should_widen_from_restricted_rescue(restricted_meh, primary, cfg) is False
    assert _should_widen_from_restricted_rescue(restricted_bad, primary, cfg) is True
    assert _should_widen_from_restricted_rescue(restricted_inf, primary, cfg) is True


def test_run_factorized_search_rescue_with_heuristics_keeps_full_pass(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1, 2), include_x=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="auto"))
    primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )

    seen: list[tuple[tuple[int, ...], bool]] = []

    def _fake_run(**kwargs):
        local_cfg = kwargs["cfg_attempt"]
        seen.append((tuple(local_cfg.order_candidates), bool(local_cfg.include_x)))
        out = _make_rescue_result(probe_rms=5.0e-3)
        out.diagnostics = {}
        return out

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["cached"])
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_rescue_with_heuristics(
        primary_res=primary,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    assert seen == [((1, 2), True)]
    assert out.diagnostics["selected_attempt"] == "full"
    assert len(out.diagnostics["rescue_attempts"]) == 1
    assert out.diagnostics["rescue_attempts"][0]["conditioned_on_primary"] is False


def test_run_factorized_search_rescue_with_heuristics_falls_back_after_bad_full_pass(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1, 2), include_x=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="auto"))
    primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )

    seen: list[tuple[tuple[int, ...], bool]] = []

    def _fake_run(**kwargs):
        local_cfg = kwargs["cfg_attempt"]
        seen.append((tuple(local_cfg.order_candidates), bool(local_cfg.include_x)))
        if len(seen) == 1:
            out = _make_rescue_result(probe_rms=5.0e-1)
        else:
            out = _make_rescue_result(probe_rms=5.0e-3)
        out.diagnostics = {}
        return out

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["cached"])
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_rescue_with_heuristics(
        primary_res=primary,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    assert seen == [((1, 2), True), ((1,), True)]
    assert out.diagnostics["selected_attempt"] == "order1_full"
    assert len(out.diagnostics["rescue_attempts"]) == 2
    assert out.diagnostics["rescue_attempts"][0]["fallback_triggered"] is True


def test_run_factorized_search_rescue_with_heuristics_returns_best_not_last_attempt(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1, 2), include_x=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="auto"))
    primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
    )

    seen: list[tuple[tuple[int, ...], bool]] = []

    def _fake_run(**kwargs):
        local_cfg = kwargs["cfg_attempt"]
        seen.append((tuple(local_cfg.order_candidates), bool(local_cfg.include_x)))
        if len(seen) == 1:
            out = _make_rescue_result(probe_rms=5.0e-1)
        else:
            out = _make_rescue_result(probe_rms=9.0e-1)
        out.diagnostics = {}
        return out

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", lambda **kwargs: ["cached"])
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_rescue_with_heuristics(
        primary_res=primary,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    assert seen == [((1, 2), True), ((1,), True)]
    assert out.probe_rms == pytest.approx(5.0e-1)
    assert out.diagnostics["selected_attempt"] == "full"
    assert out.diagnostics["rescue_attempts"][-1]["name"] == "order1_full"


def test_run_factorized_search_rescue_with_heuristics_accepts_cached_feature_groups(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    primary = _make_primary_result(rms_val=5.0e-1)
    seen: list[object] = []

    def _fake_prepare(**kwargs):
        raise AssertionError("feature groups should be reused when provided")

    def _fake_run(**kwargs):
        seen.append(kwargs["feature_groups"])
        out = _make_rescue_result(probe_rms=1.0e-3)
        out.diagnostics = {}
        return out

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", _fake_prepare)
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_rescue_with_heuristics(
        primary_res=primary,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
        feature_groups=["typed-cached"],
    )

    assert out.probe_rms == pytest.approx(1.0e-3)
    assert seen == [["typed-cached"]]


def test_prepare_factorized_search_feature_groups_uses_multi_surrogate_builder(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    calls = []

    def _fake_build_multi(*args, **kwargs):
        calls.append(kwargs)
        return ["cached-groups"]

    monkeypatch.setattr(run_de_mod, "build_factorized_search_de_feature_groups_from_surrogates", _fake_build_multi)

    out = _prepare_factorized_search_feature_groups(
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        surrogates=[object(), object()],
        dl_tr_list=[object(), object()],
        dl_va_list=[object(), object()],
        dataset_ids=["d0", "d1"],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert out == ["cached-groups"]
    assert len(calls) == 1
    assert calls[0]["dataset_ids"] == ["d0", "d1"]


def test_run_factorized_search_only_with_heuristics_stops_after_good_full_attempt(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))

    seen = []

    def _fake_prepare(**kwargs):
        return ["cached"]

    def _fake_run(**kwargs):
        seen.append((kwargs["cfg_attempt"], kwargs["feature_groups"]))
        out = _make_rescue_result(probe_rms=5.0e-3)
        out.diagnostics = {"domain_ok": True, "integrate_ok": True, "integrate_mse": 1.0e-8, "size": 1}
        return out

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", _fake_prepare)
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_only_with_heuristics(
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    assert len(seen) == 1
    assert seen[0][0].include_x is True
    assert seen[0][0].include_u is True
    assert seen[0][1] == ["cached"]
    assert out.diagnostics["selected_attempt"] == "full"
    assert out.diagnostics["rescue_attempts"][0]["fallback_triggered"] is False


def test_run_factorized_search_only_with_heuristics_falls_back_and_prefers_best_restricted_attempt(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))

    seen: list[tuple[tuple[int, ...], bool, bool, bool]] = []

    def _fake_prepare(**kwargs):
        return ["cached"]

    def _fake_run(**kwargs):
        local_cfg = kwargs["cfg_attempt"]
        assert kwargs["feature_groups"] == ["cached"]
        sig = (
            tuple(local_cfg.order_candidates),
            bool(local_cfg.include_x),
            bool(local_cfg.include_u),
            bool(local_cfg.include_du),
        )
        seen.append(sig)
        if bool(local_cfg.include_u) and not bool(local_cfg.include_x):
            out = _make_rescue_result(probe_rms=1.0e-4)
            out.diagnostics = {"domain_ok": True, "integrate_ok": True, "integrate_mse": 1.0e-8, "size": 1}
            return out
        out = _make_rescue_result(probe_rms=1.0e-1)
        out.diagnostics = {"domain_ok": True, "integrate_ok": True, "integrate_mse": 1.0e-2, "size": 2}
        return out

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", _fake_prepare)
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_only_with_heuristics(
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    assert seen == [
        ((1,), True, True, False),
        ((1,), True, False, False),
        ((1,), False, True, False),
    ]
    assert out.diagnostics["selected_attempt"] == "order1_u"
    assert len(out.diagnostics["rescue_attempts"]) == 3
    assert out.diagnostics["rescue_attempts"][0]["fallback_triggered"] is True


def test_run_factorized_search_only_with_heuristics_respects_max_attempts(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(
        _make_args(factorized_search_rescue="always", factorized_search_max_attempts=1)
    )

    seen: list[str] = []

    def _fake_prepare(**kwargs):
        return ["cached"]

    def _fake_run(**kwargs):
        seen.append(str(kwargs["cfg_attempt"].order_candidates))
        out = _make_rescue_result(probe_rms=1.0e-1)
        out.diagnostics = {"domain_ok": True, "integrate_ok": True, "integrate_mse": 1.0e-2, "size": 2}
        return out

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", _fake_prepare)
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_only_with_heuristics(
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    assert len(seen) == 1
    assert out.diagnostics["rescue_attempts_available"] == 3
    assert out.diagnostics["rescue_attempts_run"] == 1
    assert out.diagnostics["rescue_max_attempts"] == 1
    assert out.diagnostics["rescue_attempts_capped"] is True
    assert out.diagnostics["rescue_attempts"][0]["fallback_triggered"] is True


def test_run_factorized_search_only_with_heuristics_merges_shortlists_across_attempts(monkeypatch):
    cfg = DESearchConfig(order_candidates=(1,), include_x=True, include_u=True)
    rescue_cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))

    def _fake_prepare(**kwargs):
        return ["cached"]

    def _fake_result(*, probe_rms: float, tag: str) -> FactorizedSearchDEResult:
        out = _make_rescue_result(probe_rms=probe_rms)
        out.expr_ast = ("var", 1 if tag == "u" else 0)
        out.mapping = {"kind": "poly", "coeffs": [0.0, -1.0 if tag == "u" else -0.5]}
        out.canonical_equation = f"eq_{tag}"
        out.diagnostics = {
            "domain_ok": True,
            "integrate_ok": True,
            "integrate_mse": 1.0e-8 if tag == "u" else 1.0e-2,
            "size": 1 if tag == "u" else 2,
            "shortlist_union": [
                {
                    "engine": "factorized_search",
                    "kind": "factorized",
                    "order": 1,
                    "x_axis": 0,
                    "expr_ast": ["var", 1 if tag == "u" else 0],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0 if tag == "u" else -0.5]},
                    "mapping_kind": "poly",
                    "canonical_equation": f"eq_{tag}",
                    "shortlist_rank": 0,
                }
            ],
        }
        return out

    def _fake_run(**kwargs):
        local_cfg = kwargs["cfg_attempt"]
        if bool(local_cfg.include_u) and not bool(local_cfg.include_x):
            return _fake_result(probe_rms=1.0e-4, tag="u")
        return _fake_result(probe_rms=1.0e-1, tag="x")

    monkeypatch.setattr(run_de_mod, "_prepare_factorized_search_feature_groups", _fake_prepare)
    monkeypatch.setattr(run_de_mod, "_run_factorized_search_rescue_attempt", _fake_run)

    out = _run_factorized_search_only_with_heuristics(
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        filepaths=["dummy.csv"],
        surrogates=[object()],
        dl_tr_list=[object()],
        dl_va_list=[object()],
        dataset_ids=["d0"],
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    merged = out.diagnostics["shortlist_union"]
    assert len(merged) == 2
    assert [row["candidate_rank"] for row in merged] == [0, 1]

    payload = serialize_de_candidate(out)
    assert len(payload["shortlist"]) == 2
    assert payload["internal_selected_shortlist_rank"] == 1


def test_candidate_probe_rms_uses_worst_dataset_for_multi_result():
    multi = DESearchResultMulti(
        order=2,
        x_axis=0,
        term_asts=[None],
        coeffs=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        rms_train=[1.0e-3, 2.0e-3],
        rms_val=[3.0e-3, 7.0e-3],
        dataset_ids=["d0", "d1"],
        residual_asts=["r0", "r1"],
    )

    assert candidate_probe_rms(multi) == pytest.approx(7.0e-3)


def test_candidate_probe_rms_supports_factorized_result():
    factorized = _make_factorized_result(probe_rms=4.5e-4)
    assert candidate_probe_rms(factorized) == pytest.approx(4.5e-4)


def test_serialize_de_candidate_factorized_includes_validation_candidate():
    factorized = _make_factorized_result(probe_rms=3.5e-4)

    payload = serialize_de_candidate(factorized)

    assert payload["engine"] == "factorized"
    assert payload["kind"] == "factorized_blocks"
    assert float(payload["probe_rms"]) == pytest.approx(3.5e-4)
    assert payload["validation_candidate"]["coefficients"] == [1.0]
    assert len(payload["validation_candidate"]["term_asts_json"]) == 1
    assert len(payload["shortlist"]) == 1
    assert payload["shortlist"][0]["candidate_rank"] == 0
    assert payload["shortlist"][0]["validation_candidate"]["coefficients"] == [1.0]
    assert payload["shortlist"][0]["lane"] == "x_coeff_on_u"
    assert payload["shortlist"][0]["family"] == "reciprocal"
    assert payload["shortlist"][0]["base_mode"] == "zero"
    assert payload["shortlist"][0]["evidence_tier"] == "verified"
    assert payload["shortlist"][0]["witness_kind"] == "same_x_witness"
    assert payload["shortlist"][0]["consistency_score"] == pytest.approx(1.0e-6)
    assert payload["shortlist"][0]["consistency_pairs"] == 3
    assert payload["shortlist"][0]["consistency_total_pairs"] == 3
    assert payload["shortlist"][0]["collapse_confidence"] == "high"
    assert payload["shortlist"][0]["collapse_reason"] == "ok"
    assert payload["shortlist"][0]["collapse_score"] == pytest.approx(2.0e-6)
    assert payload["shortlist"][0]["typed_metadata"]["lane"] == "x_coeff_on_u"
    assert payload["shortlist"][0]["typed_metadata"]["coord_ast"] == repr(Var(0))
    assert payload["shortlist"][0]["typed_metadata"]["collapse_confidence"] == "high"
    assert payload["shortlist"][0]["typed_metadata"]["collapse_pairs"] == 3
    assert payload["shortlist"][0]["coeff_ast"] == repr(Add(ConstNode(1.0), Var(0)))
    assert payload["lane"] == "x_coeff_on_u"
    assert payload["family"] == "reciprocal"
    assert payload["base_mode"] == "zero"
    assert payload["evidence_tier"] == "verified"
    assert payload["witness_kind"] == "same_x_witness"
    assert payload["consistency_score"] == pytest.approx(1.0e-6)
    assert payload["collapse_confidence"] == "high"
    assert payload["typed_metadata"]["collapse_confidence"] == "high"
    assert payload["typed_metadata"]["carrier_ast"] == repr(U())
    assert payload["typed_metadata"]["coeff_ast"] == repr(Add(ConstNode(1.0), Var(0)))
    assert payload["diagnostics"]["zero_base_x_lane_diagnostics"]["coord_reports"][0]["coord_ast"] == "(1 + x0)"


def test_choose_best_de_candidate_requires_material_improvement():
    cfg = build_factorized_search_rescue_config_from_args(_make_args(factorized_search_rescue="always"))
    primary = _make_primary_result(rms_val=1.0e-2)
    weak_rescue = _make_rescue_result(probe_rms=9.9e-3)
    strong_rescue = _make_rescue_result(probe_rms=5.0e-3)

    selected0, engine0 = choose_best_de_candidate(primary, weak_rescue, cfg)
    selected1, engine1 = choose_best_de_candidate(primary, strong_rescue, cfg)

    assert selected0 is primary
    assert engine0 == "stlsq"
    assert selected1 is strong_rescue
    assert engine1 == "factorized_search"


def test_write_de_json_report_includes_structured_rescue_fields(tmp_path: Path):
    primary = _make_primary_result(rms_val=2.0e-2, condition_number=3.0e8)
    rescue = _attach_shortlist_report(_make_rescue_result(probe_rms=5.0e-4))
    args = _make_args(factorized_search_rescue="auto", factorized_search_preset="paper")
    report_path = tmp_path / "report.json"
    rescue_cfg = build_factorized_search_rescue_config_from_args(args)

    write_de_json_report(
        ["dummy.csv"],
        str(report_path),
        [1.0e-4],
        rescue,
        args,
        walltime=0.0,
        primary_result=primary,
        rescue_result=rescue,
        selected_engine="factorized_search",
        rescue_cfg=rescue_cfg,
        rescue_triggered=True,
        rescue_trigger_reason="high_val_rms",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    de_payload = payload["de_discovery"]

    assert de_payload["selected_engine"] == "factorized_search"
    assert de_payload["rescue_attempted"] is True
    assert de_payload["rescue_triggered"] is True
    assert de_payload["rescue_reason"]["trigger"] == "high_val_rms"
    assert de_payload["first_line"]["engine"] == "stlsq"
    assert float(de_payload["first_line"]["condition_number"]) == pytest.approx(3.0e8)
    assert de_payload["factorized_search_rescue"]["engine"] == "factorized_search"
    assert len(de_payload["factorized_search_rescue"]["shortlist"]) == 2
    assert de_payload["factorized_search_rescue"]["shortlist"][0]["shortlist_rank"] == 0
    assert de_payload["selected"]["engine"] == "factorized_search"
    assert isinstance(de_payload["proposal_slate"], list)
    assert any(p["engine"] == "stlsq" for p in de_payload["proposal_slate"])
    assert any(
        p["engine"] == "factorized_search"
        and "factorized_search_rescue" in p["support"]["sources"]
        for p in de_payload["proposal_slate"]
    )
    assert any(
        "selected" in p["support"]["sources"] and p["support"]["selected"] is True
        for p in de_payload["proposal_slate"]
    )
    assert len(de_payload["selected"]["shortlist"]) == 2
    assert de_payload["canonical_equation"] == "u_x - (rhs) = 0"
    assert float(de_payload["probe_rms"]) == pytest.approx(5.0e-4)
    assert payload["config"]["factorized_search_rescue"] == "auto"
    assert payload["config"]["factorized_search_preset"] == "paper"
    assert payload["config"]["factorized_search_max_attempts"] is None
    assert payload["config"]["factorized_search_effective_max_attempts"] is None


def test_write_de_json_report_adjudicate_selects_committee_candidate(tmp_path: Path):
    primary = _make_primary_result(rms_val=2.0e-2)
    factorized = _make_factorized_result(probe_rms=5.0e-4)
    args = _make_args(de_coe_mode="adjudicate", factorized_search_rescue="auto")
    report_path = tmp_path / "report_adjudicate.json"

    write_de_json_report(
        ["dummy.csv"],
        str(report_path),
        [1.0e-4],
        primary,
        args,
        walltime=0.0,
        primary_result=primary,
        factorized_result=factorized,
        selected_engine="stlsq",
        rescue_cfg=build_factorized_search_rescue_config_from_args(args),
    )

    de_payload = json.loads(report_path.read_text(encoding="utf-8"))["de_discovery"]

    assert de_payload["selected_engine"] == "factorized"
    assert de_payload["internal_selected_engine"] == "stlsq"
    assert de_payload["internal_selected"]["engine"] == "stlsq"
    assert de_payload["selected"]["engine"] == "factorized"
    assert de_payload["committee_adjudicated"] is True
    assert de_payload["committee_adjudication_fallback"] is False
    assert de_payload["committee_decision"]["config"]["mode"] == "adjudicate"
    assert de_payload["probe_rms"] == pytest.approx(5.0e-4)
    assert de_payload["factorized_rescue"]["typed_metadata"]["collapse_confidence"] == "high"
    assert de_payload["selected"]["typed_metadata"]["collapse_confidence"] == "high"
    assert any(
        proposal["engine"] == "factorized"
        and proposal["rhs_payload"]["typed_metadata"]["collapse_confidence"] == "high"
        for proposal in de_payload["proposal_slate"]
    )


def test_write_de_json_report_adjudicate_falls_back_when_committee_has_no_valid_candidate(tmp_path: Path):
    primary = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[DU(0)],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-2,
        rms_val=2.0e-2,
        residual_ast="invalid_first_order_du",
        condition_number=1.0,
    )
    args = _make_args(de_coe_mode="adjudicate")
    report_path = tmp_path / "report_adjudicate_fallback.json"

    write_de_json_report(
        ["dummy.csv"],
        str(report_path),
        [1.0e-4],
        primary,
        args,
        walltime=0.0,
        primary_result=primary,
        selected_engine="stlsq",
    )

    de_payload = json.loads(report_path.read_text(encoding="utf-8"))["de_discovery"]

    assert de_payload["selected_engine"] == "stlsq"
    assert de_payload["selected"]["engine"] == "stlsq"
    assert de_payload["internal_selected_engine"] == "stlsq"
    assert de_payload["committee_adjudicated"] is False
    assert de_payload["committee_adjudication_fallback"] is True
    assert de_payload["committee_decision"]["status"] == "no_valid_candidates"
    assert any("kept legacy internal selection" in warning for warning in de_payload["committee_decision"]["warnings"])


def test_write_de_json_report_factorized_search_only_keeps_first_line_null(tmp_path: Path):
    rescue = _make_rescue_result(probe_rms=2.5e-4)
    args = _make_args(factorized_search_rescue="never", factorized_search_preset="fast")
    report_path = tmp_path / "report_factorized_search_only.json"
    rescue_cfg = build_factorized_search_rescue_config_from_args(args)

    write_de_json_report(
        ["dummy.csv"],
        str(report_path),
        [1.0e-4],
        rescue,
        args,
        walltime=0.0,
        primary_result=None,
        rescue_result=rescue,
        selected_engine="factorized_search",
        rescue_cfg=rescue_cfg,
        rescue_triggered=True,
        rescue_trigger_reason="mode_factorized_search_only",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    de_payload = payload["de_discovery"]

    assert de_payload["first_line"] is None
    assert de_payload["selected"]["engine"] == "factorized_search"
    assert len(de_payload["proposal_slate"]) == 1
    assert de_payload["proposal_slate"][0]["support"]["sources"] == [
        "factorized_search_rescue",
        "selected",
    ]
    assert de_payload["rescue_reason"]["trigger"] == "mode_factorized_search_only"
    assert de_payload["rescue_reason"]["primary_rms"] is None
    assert de_payload["rescue_reason"]["primary_condition_number"] is None


def test_build_library_validation_candidate_refits_shared_coefficients():
    x = torch.linspace(0.1, 2.0, 64, dtype=torch.float64).reshape(-1, 1)
    y = torch.zeros_like(x)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=16,
        shuffle=False,
    )

    multi = DESearchResultMulti(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([[0.1], [2.0]], dtype=torch.float64),
        rms_train=[1.0, 1.0],
        rms_val=[1.0, 1.0],
        dataset_ids=["d0", "d1"],
    )

    candidate = _build_library_validation_candidate(
        multi,
        surrogates=[_ExpDecaySurrogate(0.5), _ExpDecaySurrogate(0.5)],
        train_dataloaders=[loader, loader],
        val_dataloaders=[loader, loader],
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,), include_x=False, include_u=True),
        device=torch.device("cpu"),
    )

    assert candidate is not None
    assert candidate["coefficient_mode"] == "shared_pooled_refit"
    assert candidate["coefficients"] == pytest.approx([0.5], rel=1.0e-2, abs=1.0e-2)
    assert candidate["dataset_coefficients"] == [[0.1], [2.0]]
    assert candidate["term_asts_json"][0]["type"] == "atom"
    assert candidate["term_asts_json"][0]["kind"] == "u"
