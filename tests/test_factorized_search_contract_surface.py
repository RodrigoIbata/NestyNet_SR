# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.factorized_search.adapters.nestynet.api import run_explorer
from nestynet_sr.sr_search.factorized_search import explorer as explorer_mod
from nestynet_sr.sr_search.factorized_search.engine.actions import (
    apply_action_impl,
    apply_crossover_action_impl,
    apply_residual_action_impl,
)
from nestynet_sr.sr_search.factorized_search.engine.scoring import (
    _eval_node_hparam_safe,
    _harvest_pool_from_archive,
    _mapping_equiv_root,
    fingerprint,
    score_expr,
)
from nestynet_sr.sr_search.factorized_search.engine.search import Explorer, run_explorer_core
from nestynet_sr.sr_search.factorized_search.config import (
    FactorizedSearchConfig,
    InverseSteeringConfig,
    REFINE_OPTIMIZER_NAMES,
    REFINE_PROFILE_NAMES,
    apply_refine_mode_placement_defaults,
    apply_refine_profile,
    factorized_config_report,
    resolve_refine_profile,
)
from nestynet_sr.sr_search.factorized_search.policy.parent_selection import choose_parent, choose_parent_repair_aware
from nestynet_sr.sr_search.factorized_search.engine.api import ArchiveRecord, EngineRequest, EngineResult
from nestynet_sr.sr_search.config import FactorizedSearchConfig as LegacyFactorizedSearchConfig
from nestynet_sr.sr_search.factorized_search.aif_closure_benchmark import _attach_full_validation


def test_factorized_search_config_legacy_reexport_matches_new_home():
    assert LegacyFactorizedSearchConfig is FactorizedSearchConfig

    hp = FactorizedSearchConfig()
    assert hp.n_iter > 0
    assert hp.wall_time_limit_s is None
    assert hp.outer_wrapper_transforms[:2] == ["log", "reciprocal"]
    assert hp.hole_search_mine_cooldown_iters == 50
    assert hp.hole_search_route_scheduler_enable is True
    assert hp.hole_search_route_reward_mode == "penalized"
    assert hp.hole_search_route_time_penalty == 0.01
    assert hp.hole_search_route_time_floor == 1.0
    assert hp.hole_search_abstraction_enable is True
    assert hp.hole_search_abstraction_on_improve is True
    assert hp.hole_search_abstraction_on_stall is True
    assert hp.hole_search_abstraction_improve_min_delta_log_mse == 0.15
    assert hp.hole_search_abstraction_stage_enable is True
    assert hp.hole_search_abstraction_stage_max_entries == 64
    assert hp.hole_search_abstraction_promote_topk == 2
    assert hp.hole_search_abstraction_promote_frontier_floor == 3
    assert hp.hole_search_solver_market_enable is False
    assert hp.hole_search_solver_market_preview_topk == 4
    assert hp.hole_search_solver_market_exact_topk == 2
    assert hp.hole_search_solver_market_proposal_objects_enable is False
    assert hp.scheduler_witness_energy_enable is False
    assert hp.inverse_spec_recursive_sr_enable is False
    assert hp.inverse_spec_recursive_sr_preview_topk == 4
    assert hp.inverse_spec_recursive_sr_exact_budget == 2
    assert hp.inverse_spec_constant_lift_route_enable is False
    assert hp.inverse_spec_constant_lift_route_topk == 2
    assert hp.inverse_spec_coordinate_lift_enable is False
    assert hp.inverse_spec_coordinate_lift_topk == 4
    assert hp.inverse_spec_coordinate_lift_mode == "both"
    assert hp.inverse_spec_tangent_edit_enable is False
    assert hp.inverse_spec_tangent_edit_topk == 8
    assert hp.inverse_spec_soft_edit_enable is False
    assert hp.inverse_spec_soft_edit_steps == 64
    assert hp.inverse_spec_soft_edit_l1 == 1.0e-3
    assert hp.inverse_spec_witness_jets_enable is False


def test_factorized_config_report_serializes_full_resolved_surface_and_dynamic_fields():
    hp = FactorizedSearchConfig()
    hp.search_profile = "unit_test"
    hp.closure_search_enable = False
    hp.de_sparse_combo_enable = False
    hp.de_sparse_combo_pool_topk = 8

    report = factorized_config_report(hp)
    resolved = report["resolved"]

    assert report["schema_version"] == 1
    assert report["includes_disabled_lanes"] is True
    assert resolved["search_profile"] == "unit_test"
    assert resolved["closure_search_enable"] is False
    assert resolved["repair_controller_enable"] is False
    assert resolved["de_sparse_combo_enable"] is False
    assert resolved["de_sparse_combo_pool_topk"] == 8
    assert resolved["de_sparse_combo_mapping_mode"] == "affine_only"
    assert resolved["de_sparse_combo_max_condition"] == 1.0e10
    assert report["field_count"] == len(resolved)
    assert report["diff_from_config_default"]["search_profile"] == "unit_test"
    assert "de_sparse_combo_enable" not in report["diff_from_config_default"]
    assert hp.inverse_spec_witness_d2_enable is False
    assert hp.inverse_spec_witness_max_rows == 64
    assert hp.inverse_spec_witness_loss_enable is False
    assert hp.inverse_spec_witness_grad_weight == 1.0
    assert hp.inverse_spec_witness_d2_weight == 0.0
    assert hp.inverse_spec_witness_diag_weight == 0.0
    assert hp.inverse_spec_witness_physics_weight == 0.0
    assert hp.inverse_spec_active_var_screen_enable is False
    assert hp.inverse_spec_active_var_grad_tol == 1.0e-3
    assert hp.inverse_spec_active_var_max_count == 4
    assert hp.inverse_spec_directional_market_enable is False
    assert hp.brute_score_mapping_family_mode == "gated"
    assert hp.refine_profile == "default"
    assert hp.refine_mode == "slate"
    assert hp.refine_during_brute is False
    assert hp.refine_during_mutation is False
    assert hp.refine_during_controller_slate is False
    assert hp.refine_during_slate is True
    assert hp.refine_slate_after_brute is True
    assert hp.refine_slate_period == 0
    assert hp.refine_final_polish is True
    assert hp.refine_slate_k == 16
    assert hp.refine_slate_diverse_k == 8
    assert hp.refine_slate_budget == 32
    assert hp.refine_optimizer == "lbfgs"
    assert hp.refine_lbfgs_escalate_improve_factor == 2.0
    assert hp.refine_prune_mapping_equiv_root_slots is True
    assert hp.refine_attempt_cache_enable is True
    assert hp.refine_attempt_cache_max_entries == 4096
    assert "grid" in REFINE_OPTIMIZER_NAMES
    assert "grid_then_lbfgs" in REFINE_OPTIMIZER_NAMES
    assert hp.score_prescreen_enable is True
    assert hp.score_prescreen_family_mode == "cheap"
    assert hp.score_prescreen_residual_family_mode == "gated"
    assert hp.score_prescreen_residual_allow_hint is False
    assert hp.score_prescreen_residual_use_global_best is False
    assert hp.score_prescreen_residual_parent_best_factor == 1.1
    assert hp.score_prescreen_residual_global_best_factor == 1.5
    assert hp.inverse_spec_family_battery_enable is False
    assert hp.inverse_spec_family_battery_mode == "outer"

    inverse_cfg = InverseSteeringConfig()
    assert inverse_cfg.inverse_spec_family_battery_enable is False
    assert inverse_cfg.inverse_spec_recursive_sr_enable is False
    assert inverse_cfg.inverse_spec_recursive_sr_preview_topk == 4
    assert inverse_cfg.inverse_spec_recursive_sr_exact_budget == 2
    assert inverse_cfg.inverse_spec_constant_lift_route_enable is False
    assert inverse_cfg.inverse_spec_constant_lift_route_topk == 2
    assert inverse_cfg.inverse_spec_coordinate_lift_enable is False
    assert inverse_cfg.inverse_spec_coordinate_lift_topk == 4
    assert inverse_cfg.inverse_spec_coordinate_lift_mode == "both"
    assert inverse_cfg.inverse_spec_tangent_edit_enable is False
    assert inverse_cfg.inverse_spec_tangent_edit_topk == 8
    assert inverse_cfg.inverse_spec_soft_edit_enable is False
    assert inverse_cfg.inverse_spec_soft_edit_steps == 64
    assert inverse_cfg.inverse_spec_soft_edit_l1 == 1.0e-3
    assert inverse_cfg.inverse_spec_witness_jets_enable is False
    assert inverse_cfg.inverse_spec_witness_d2_enable is False
    assert inverse_cfg.inverse_spec_witness_max_rows == 64
    assert inverse_cfg.inverse_spec_witness_loss_enable is False
    assert inverse_cfg.inverse_spec_witness_grad_weight == 1.0
    assert inverse_cfg.inverse_spec_witness_d2_weight == 0.0
    assert inverse_cfg.inverse_spec_witness_diag_weight == 0.0
    assert inverse_cfg.inverse_spec_witness_physics_weight == 0.0
    assert inverse_cfg.inverse_spec_active_var_screen_enable is False
    assert inverse_cfg.inverse_spec_active_var_grad_tol == 1.0e-3
    assert inverse_cfg.inverse_spec_active_var_max_count == 4
    assert inverse_cfg.inverse_spec_directional_market_enable is False
    assert inverse_cfg.inverse_spec_family_battery_mode == "outer"


def test_refine_profile_inline_preserves_compatibility_placement():
    hp = FactorizedSearchConfig()

    apply_refine_profile(hp, "legacy")

    assert "inline" in REFINE_PROFILE_NAMES
    assert resolve_refine_profile("inline")[0] == "inline"
    assert resolve_refine_profile("compat-inline")[0] == "inline"
    assert hp.refine_profile == "inline"
    assert hp.refine_mode == "inline"
    assert hp.refine_during_brute is True
    assert hp.refine_during_mutation is True
    assert hp.refine_during_controller_slate is False
    assert hp.refine_during_slate is False


def test_refine_mode_defaults_reset_cli_placement():
    hp = FactorizedSearchConfig()
    apply_refine_profile(hp, "rare_slate")

    apply_refine_mode_placement_defaults(hp, "inline")

    assert hp.refine_mode == "inline"
    assert hp.refine_during_brute is True
    assert hp.refine_during_mutation is True
    assert hp.refine_during_controller_slate is False
    assert hp.refine_during_slate is False
    assert hp.refine_max_trials == 50


def test_refine_profile_rare_sets_stingy_runtime_knobs():
    hp = FactorizedSearchConfig()

    apply_refine_profile(hp, "rare")

    assert hp.refine_profile == "rare"
    assert hp.refine_mode == "inline"
    assert hp.refine_optimizer == "lbfgs"
    assert hp.refine_max_trials == 50
    assert hp.refine_max_variants == 1
    assert hp.refine_max_params == 1
    assert hp.refine_num_restarts == 1
    assert hp.refine_lbfgs_steps == 4
    assert hp.refine_fit_subset == 64
    assert hp.refine_grid_size == 17
    assert hp.refine_grid_size_2d == 5
    assert hp.refine_grid_passes == 1
    assert hp.refine_grid_max_evals == 32
    assert hp.refine_gate_best_factor == 2.0
    assert hp.refine_gate_max_evals == 16
    assert hp.refine_stall_gate_relax_factor == 1.0


def test_refine_profile_rare_slate_sets_scheduled_placement():
    hp = FactorizedSearchConfig()

    apply_refine_profile(hp, "rare-slate")

    assert resolve_refine_profile("slate")[0] == "rare_slate"
    assert hp.refine_profile == "rare_slate"
    assert hp.refine_mode == "slate"
    assert hp.refine_during_brute is False
    assert hp.refine_during_mutation is False
    assert hp.refine_during_controller_slate is False
    assert hp.refine_during_slate is True
    assert hp.refine_slate_after_brute is True
    assert hp.refine_final_polish is True
    assert hp.refine_slate_budget == 32


def test_factorized_search_engine_contracts_construct():
    req = EngineRequest(nvars=2, metadata={"source": "unit_test"})
    rec = ArchiveRecord(
        expr=("var", 0),
        mse_raw=1.0,
        mse_eff=1.0,
        size=1,
        depth=0,
        meta={"tag": "toy"},
    )
    result = EngineResult(records=(rec,), best=rec, diagnostics={"request": req.metadata["source"]})

    assert req.nvars == 2
    assert result.best is rec
    assert result.records[0].meta["tag"] == "toy"
    assert result.diagnostics["request"] == "unit_test"


def test_nestynet_adapter_api_reexports_bridge_surface():
    assert callable(run_explorer)


def test_explorer_reexports_policy_parent_selection():
    assert explorer_mod.choose_parent is choose_parent
    assert explorer_mod.choose_parent_repair_aware is choose_parent_repair_aware


def test_engine_action_impls_exist_alongside_explorer_wrappers():
    assert callable(apply_action_impl)
    assert callable(apply_crossover_action_impl)
    assert callable(apply_residual_action_impl)
    assert callable(explorer_mod.apply_action)
    assert callable(explorer_mod.apply_crossover_action)
    assert callable(explorer_mod.apply_residual_action)


def test_engine_search_surface_reexports_explorer_entries():
    assert Explorer is explorer_mod.Explorer
    assert callable(run_explorer_core)
    assert callable(explorer_mod.run_explorer_core)


def test_apply_inverse_steering_action_does_not_forward_hole_search_only_knobs(monkeypatch):
    captured = {}

    def _fake_run_inverse_steering_action(*args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return ("var", 0)

    monkeypatch.setattr(explorer_mod, "run_inverse_steering_action", _fake_run_inverse_steering_action)

    out = explorer_mod.apply_inverse_steering_action(
        ("var", 0),
        {"kind": "affine"},
        None,
        None,
        None,
        None,
        [],
        None,
        None,
        [],
        None,
        4,
        1,
        2,
        inverse_spec_coordinate_lift_enable=True,
        inverse_spec_coordinate_lift_topk=5,
        inverse_spec_coordinate_lift_mode="single_index",
        inverse_spec_constant_lift_route_enable=True,
        inverse_spec_constant_lift_route_topk=3,
        inverse_spec_directional_market_enable=True,
    )

    assert out == ("var", 0)
    forwarded = captured["kwargs"]
    assert "inverse_spec_coordinate_lift_enable" not in forwarded
    assert "inverse_spec_coordinate_lift_topk" not in forwarded
    assert "inverse_spec_coordinate_lift_mode" not in forwarded
    assert "inverse_spec_constant_lift_route_enable" not in forwarded
    assert "inverse_spec_constant_lift_route_topk" not in forwarded
    assert "inverse_spec_directional_market_enable" not in forwarded


def test_engine_scoring_surface_reexports_explorer_entries():
    assert callable(score_expr)
    assert callable(explorer_mod.score_expr)
    assert callable(fingerprint)
    assert callable(explorer_mod.fingerprint)
    assert callable(_mapping_equiv_root)
    assert callable(explorer_mod._mapping_equiv_root)
    assert callable(_harvest_pool_from_archive)
    assert callable(explorer_mod._harvest_pool_from_archive)
    assert callable(_eval_node_hparam_safe)
    assert callable(explorer_mod._eval_node_hparam_safe)


def test_engine_score_expr_records_refine_diagnostics_and_uses_cache():
    dtype = torch.float64
    x_fit = torch.linspace(0.2, 2.0, 80, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.25, 1.95, 96, dtype=dtype).unsqueeze(-1)
    y_fit = torch.sin(2.0 * x_fit)
    y_probe = torch.sin(2.0 * x_probe)
    proj = torch.randn(
        (x_probe.shape[0], 8),
        generator=torch.Generator(device="cpu").manual_seed(3),
        dtype=dtype,
    )
    cfg = {
        "optimizer": "grid",
        "fit_subset": 64,
        "fit_subset_mode": "stride",
        "max_variants": 1,
        "max_params": 1,
        "max_refines": 4,
        "linear_combo_enable": False,
        "gate_best_factor": 100.0,
        "num_restarts": 1,
        "init_log_min": -2.0,
        "init_log_max": 2.0,
        "refine_grid_enable": True,
        "refine_grid_size": 9,
        "refine_grid_size_2d": 5,
        "refine_grid_passes": 0,
        "refine_grid_topk": 1,
        "refine_grid_max_evals": 16,
        "safe_eps": 1.0e-6,
        "diagnostics": {},
        "attempt_cache": {},
        "attempt_cache_enable": True,
        "refine_context": "brute",
        "_legacy_refinement_hooks": explorer_mod.make_engine_refinement_hooks(),
    }

    state1 = {"trials_done": 0}
    sc1 = score_expr(
        ("sin", ("var", 0)),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        4,
        refine_enable=True,
        refine_cfg=cfg,
        refine_state=state1,
        return_expr=True,
    )
    state2 = {"trials_done": 0}
    sc2 = score_expr(
        ("sin", ("var", 0)),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        4,
        refine_enable=True,
        refine_cfg=cfg,
        refine_state=state2,
        return_expr=True,
    )

    diag = cfg["diagnostics"]
    assert sc1 is not None
    assert sc2 is not None
    assert state1["trials_done"] == 1
    assert state2["trials_done"] == 0
    assert diag["score_calls"] == 2
    assert diag["refinement_attempts"] == 1
    assert diag["brute_refinement_attempts"] == 1
    assert diag["attempt_cache_hits"] >= 1
    assert diag["materialized_rescores"] == 2


def test_engine_score_expr_respects_window_budget_inside_variant_loop(monkeypatch):
    dtype = torch.float64
    x_fit = torch.linspace(0.2, 2.0, 32, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.25, 1.95, 40, dtype=dtype).unsqueeze(-1)
    y_fit = torch.sin(2.0 * x_fit)
    y_probe = torch.sin(2.0 * x_probe)
    proj = torch.randn(
        (x_probe.shape[0], 8),
        generator=torch.Generator(device="cpu").manual_seed(4),
        dtype=dtype,
    )
    variants = [
        (("sin", ("mul", ("hparam", 0), ("var", 0))), 1, frozenset()),
        (("cos", ("mul", ("hparam", 0), ("var", 0))), 1, frozenset()),
    ]
    calls = {"refine_hparams": 0}

    def fake_refine_hparams(*args, **kwargs):
        calls["refine_hparams"] += 1
        return None

    monkeypatch.setattr(explorer_mod, "_decorate_refine_variants", lambda *args, **kwargs: list(variants))
    monkeypatch.setattr(explorer_mod, "_refine_hparams", fake_refine_hparams)

    cfg = {
        "max_variants": 2,
        "max_params": 1,
        "max_refines": 10,
        "linear_combo_enable": False,
        "gate_best_factor": 100.0,
        "attempt_cache_enable": False,
        "_legacy_refinement_hooks": explorer_mod.make_engine_refinement_hooks(),
    }
    state = {"trials_done": 0, "window_trials_left": 1}
    sc = score_expr(
        ("sin", ("var", 0)),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        4,
        refine_enable=True,
        refine_cfg=cfg,
        refine_state=state,
        return_expr=True,
    )

    assert sc is not None
    assert calls["refine_hparams"] == 1
    assert state["trials_done"] == 1
    assert state["window_trials_left"] == 0


def test_score_expr_attaches_score_ladder_provenance():
    dtype = torch.float64
    x_fit = torch.linspace(-1.0, 1.0, 32, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(-0.9, 0.9, 40, dtype=dtype).unsqueeze(-1)
    y_fit = 2.0 * x_fit + 1.0
    y_probe = 2.0 * x_probe + 1.0
    proj = torch.randn(
        (x_probe.shape[0], 8),
        generator=torch.Generator(device="cpu").manual_seed(5),
        dtype=dtype,
    )

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
        refine_cfg={},
        return_expr=True,
    )

    assert sc is not None
    mapping = sc[3]
    ladder = mapping["_score_ladder"]
    assert mapping["_acceptance_basis"] == "mapped_structural"
    assert ladder["schema_version"] == 1
    assert ladder["carrier"]["expr"] == "x0"
    assert ladder["carrier"]["probe_mse_identity"] > 0.0
    assert ladder["mapped"]["available"] is True
    assert ladder["mapped"]["mapping_kind"] == "poly"
    assert ladder["mapped"]["mapping_structural"] is True
    assert ladder["mapped"]["probe_mse"] < 1.0e-20
    assert ladder["head_augmented"]["accepted"] is False
    assert ladder["compiled_structural"]["accepted"] is False
    assert ladder["final_validation"]["available"] is False


def test_full_validation_extends_score_ladder_without_reranking():
    dtype = torch.float64
    x_fit = torch.linspace(-1.0, 1.0, 8, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(-0.9, 0.9, 10, dtype=dtype).unsqueeze(-1)
    dataset = {
        "x_fit": x_fit,
        "y_fit": 2.0 * x_fit + 1.0,
        "x_probe": x_probe,
        "y_probe": 2.0 * x_probe + 1.0,
        "metadata": {"source": "unit_test"},
    }
    report = {
        "results": [
            {
                "expr_ast": ["var", 0],
                "mapping": {"kind": "poly", "coeffs": [1.0, 2.0], "mu": 0.0, "std": 1.0},
                "score_ladder": {"schema_version": 1, "mapped": {"probe_mse": 0.0}},
            }
        ]
    }

    _attach_full_validation(report, dataset, allow_rerank=False)

    row = report["results"][0]
    assert row["full_validation"]["probe_mse"] < 1.0e-24
    assert row["score_ladder"]["final_validation"]["available"] is True
    assert row["score_ladder"]["final_validation"]["source"] == "full_split_no_refit"
    assert row["final_acceptance_basis"] == "full_validation_audit"
    assert report["full_validation"]["best_index"] == 0
    assert "best_full" not in report
