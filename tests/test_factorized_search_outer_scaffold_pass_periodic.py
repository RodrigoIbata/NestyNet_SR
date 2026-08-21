# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch
import time

import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod
import nestynet_sr.sr_search.factorized_search.engine.search as search_mod
import nestynet_sr.sr_search.factorized_search.closure_search_compat as closure_compat_mod
import nestynet_sr.sr_search.factorized_search.outer_scaffold_search as scaffold_mod
import nestynet_sr.sr_search.factorized_search.proposal_families.direct as direct_mod
import nestynet_sr.sr_search.factorized_search.proposal_families.runner as runner_mod
from nestynet_sr.sr_search.factorized_search.basis_state import BasisState, ProposalContext
from nestynet_sr.sr_search.factorized_search.closures import (
    make_direct_linear_wrap_closure,
    make_direct_periodic_closure,
    make_direct_power_closure,
    make_direct_quadratic_closure,
    make_direct_rational_closure,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.compat import OuterScaffoldSpec, operator_application_from_scaffold
from nestynet_sr.sr_search.factorized_search.proposal_families.closure_runners import PreparedCandidatesSearchPlan
from nestynet_sr.sr_search.factorized_search.proposal_families.runner import (
    run_outer_scaffold_pass_impl as run_outer_scaffold_pass_impl_native,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.types import OperatorApplication


def test_outer_scaffold_pass_periodic_inserts_completed_candidate(monkeypatch):
    scaffold_parent = ("mul", ("sin", ("const", 1.0)), ("var", 2))
    completed_expr = ("mul", ("sin", ("mul", ("var", 0), ("var", 1))), ("var", 2))

    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="periodic",
        scaffold_id="periodic:sin_mul:x2",
        parent_node=scaffold_parent,
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 2),
        metadata={"form": "sin_mul"},
    )

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        score_map = {
            completed_expr: 0.05,
        }
        mse = float(score_map.get(node, 1.5))
        key = ("expr", str(node))
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return mse, key, z, mapping, node
        return mse, key, z, mapping

    def _fake_enumerate_outer_scaffold_specs(**kwargs):
        return [scaffold_spec]

    def _fake_direct_periodic(*args, **kwargs):
        return (
            [
                {
                    "expr": completed_expr,
                    "child_key": str(completed_expr),
                    "local_probe_mse": 1.0e-6,
                    "local_fit_mse": 1.0e-6,
                    "candidate_child_size": 6,
                    "local_mapping_kind": "direct_harmonic_head",
                    "local_mapping_coeffs": [1.0, 0.0, 0.0],
                    "direct_metadata": {
                        "feature_kind": "sin",
                        "hole_node": ("mul", ("var", 0), ("var", 1)),
                        "feature_node": ("sin", ("mul", ("var", 0), ("var", 1))),
                        "envelope_node": ("var", 2),
                    },
                }
            ],
            "direct_ok",
            {},
        )

    op_app = operator_application_from_scaffold(scaffold_spec)

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    monkeypatch.setattr(runner_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)
    monkeypatch.setattr(runner_mod, "solve_direct_operator_preview_rows", _fake_direct_periodic)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, 2:3],
        nvars=3,
        n_iter=0,
        max_depth=4,
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
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["periodic"],
        closure_search_max_proposals=1,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=1,
        closure_search_exact_topk=1,
        _score_expr_fn=_fake_score_expr,
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert best.best_expr == completed_expr
    assert float(best.best_mse) == 0.05
    assert bool(stats.get("enabled", False)) is True
    assert int(stats.get("families_considered", 0)) == 1
    assert int(stats.get("scaffolds_enumerated", 0)) == 1
    assert int(stats.get("scaffolds_considered", 0)) == 1
    assert int(stats.get("preview_calls", 0)) == 1
    assert int(stats.get("preview_candidates", 0)) == 1
    assert int(stats.get("scored", 0)) == 1
    assert int(stats.get("new_residual_basins", 0)) == 1
    assert int(stats.get("global_best_updates", 0)) == 1


def test_outer_scaffold_pass_records_direct_not_supported_diagnostics(monkeypatch):
    """When the direct operator route returns 'direct_not_supported', the status is recorded."""
    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="periodic",
        scaffold_id="periodic:sin",
        parent_node=("sin", ("const", 1.0)),
        hole_path=(1,),
        target_mode="robust",
        anchor_node=None,
        metadata={},
    )
    op_app = operator_application_from_scaffold(scaffold_spec)

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    def _fake_direct_not_supported(*args, **kwargs):
        return [], "direct_not_supported", {}

    monkeypatch.setattr(runner_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)
    monkeypatch.setattr(runner_mod, "solve_direct_operator_preview_rows", _fake_direct_not_supported)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, :1],
        nvars=1,
        n_iter=0,
        max_depth=3,
        poly_degree=1,
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
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["periodic"],
        closure_search_max_proposals=1,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=1,
        closure_search_exact_topk=1,
    )

    stats = getattr(arch, "closure_search_stats", {})
    status_counts = dict(stats.get("status_counts", {}) or {})

    assert int(stats.get("scaffolds_considered", 0)) == 1
    assert int(status_counts.get("direct_not_supported", 0)) == 1
    assert int(stats.get("preview_candidates", 0)) == 0


def test_enumerate_outer_scaffold_specs_adds_periodic_add_forms():
    specs = scaffold_mod.enumerate_outer_scaffold_specs(
        families=["periodic"],
        nvars=3,
        y_dims=None,
        var_dims=None,
        pool_nodes=[
            ("const", 1.0),
            ("mul", ("var", 0), ("var", 1)),
            ("mul", ("var", 0), ("var", 2)),
        ],
        pool_dims=[],
        anchors_per_family=3,
        max_scaffolds=16,
    )

    scaffold_ids = {str(spec.scaffold_id) for spec in specs}

    assert any(sid.startswith("periodic:sin_add:(x0*x1)") for sid in scaffold_ids)
    assert any(sid.startswith("periodic:cos_add:(x0*x1)") for sid in scaffold_ids)


def test_enumerate_outer_scaffold_specs_adds_rational_affine_form():
    specs = scaffold_mod.enumerate_outer_scaffold_specs(
        families=["rational"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[
            ("mul", ("var", 0), ("var", 1)),
        ],
        pool_dims=[(0.0,)],
        anchors_per_family=2,
        max_scaffolds=8,
    )

    scaffold_ids = {str(spec.scaffold_id) for spec in specs}

    assert any(sid.startswith("rational:affine") for sid in scaffold_ids)


def test_enumerate_outer_scaffold_specs_adds_quadratic_sqrt_forms():
    specs = scaffold_mod.enumerate_outer_scaffold_specs(
        families=["quadratic"],
        nvars=4,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=2,
        max_scaffolds=8,
    )

    scaffold_ids = {str(spec.scaffold_id) for spec in specs}

    assert any(sid.startswith("quadratic:sqrt") and not sid.startswith("quadratic:sqrt_mul") for sid in scaffold_ids)
    assert any(sid.startswith("quadratic:sqrt_mul:") for sid in scaffold_ids)


def test_enumerate_outer_scaffold_specs_adds_power_inverse_forms():
    specs = scaffold_mod.enumerate_outer_scaffold_specs(
        families=["power"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=2,
        max_scaffolds=12,
    )

    scaffold_ids = {str(spec.scaffold_id) for spec in specs}

    assert any(sid.startswith("power:invsqrt") and not sid.startswith("power:invsqrt_mul") for sid in scaffold_ids)
    assert any(sid.startswith("power:neg2") and not sid.startswith("power:neg2_mul") for sid in scaffold_ids)
    assert any(sid.startswith("power:invsqrt_mul:") for sid in scaffold_ids)


def test_enumerate_outer_scaffold_specs_keeps_var_anchor_for_exp_family():
    specs = scaffold_mod.enumerate_outer_scaffold_specs(
        families=["exp"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[
            ("mul", ("var", 0), ("var", 1)),
            ("mul", ("var", 0), ("const", 1.0)),
        ],
        pool_dims=[(0.0,), (0.0,)],
        anchors_per_family=2,
        max_scaffolds=8,
    )

    scaffold_ids = {str(spec.scaffold_id) for spec in specs}

    assert any(sid.startswith("exp:add:") and ":x0" in sid for sid in scaffold_ids) or \
           any(sid.startswith("exp:mul:") and ":x0" in sid for sid in scaffold_ids)
    assert any(sid.startswith("exp:add:") and ":x1" in sid for sid in scaffold_ids) or \
           any(sid.startswith("exp:mul:") and ":x1" in sid for sid in scaffold_ids)


def test_enumerate_outer_scaffold_specs_keeps_var_anchor_for_log_family():
    specs = scaffold_mod.enumerate_outer_scaffold_specs(
        families=["log"],
        nvars=3,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,)],
        pool_nodes=[
            ("mul", ("var", 0), ("var", 1)),
            ("mul", ("var", 1), ("var", 2)),
        ],
        pool_dims=[(0.0,), (0.0,)],
        anchors_per_family=3,
        max_scaffolds=8,
    )

    scaffold_ids = {str(spec.scaffold_id) for spec in specs}

    assert any(sid.startswith("log:add:") and "x2" in sid for sid in scaffold_ids)


def test_direct_exp_add_can_recover_affine_exponential(monkeypatch):
    anchor = ("var", 1)
    spec = scaffold_mod.OuterScaffoldSpec(
        family="exp",
        scaffold_id="exp:add:x1",
        parent_node=("add", ("exp", ("const", 1.0)), anchor),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=anchor,
        metadata={"form": "exp_add"},
    )

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [
            ("enum", ("var", 0)),
            ("enum", ("mul", ("var", 0), ("var", 3))),
        ], {"candidate_source_counts": {"enum": 2}}

    monkeypatch.setattr(closure_compat_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    n = 32
    x0 = torch.linspace(0.2, 1.1, steps=n, dtype=torch.float64)
    x1 = torch.linspace(0.3, 1.2, steps=n, dtype=torch.float64)
    x2 = torch.full((n,), 1.3, dtype=torch.float64)
    x3 = torch.full((n,), 1.1, dtype=torch.float64)
    x = torch.stack([x0, x1, x2, x3], dim=1)
    y = (x2 * torch.exp(x3 * x0) + x1).unsqueeze(-1)

    rows, status, meta = scaffold_mod._solve_direct_exp_preview_rows(
        spec,
        nvars=4,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"enum_max_depth": 4, "enum_max_trees": 64},
    )

    assert status == "direct_ok"
    assert int(meta.get("candidate_count_raw", 0)) >= 2
    assert len(rows) >= 1
    assert rows[0]["local_mapping_kind"] == "direct_exp_add_head"
    # Materialized expression is structural — the 1.3 coefficient lives in
    # the linear head mapping, not embedded as a const node in the AST.
    expr_text = scaffold_mod.node_str(rows[0]["expr"])
    assert "exp(" in expr_text
    assert "x1" in expr_text
    assert float(rows[0]["local_probe_mse"]) < 1.0e-12

    rows_generic, status_generic, meta_generic = scaffold_mod._solve_direct_operator_preview_rows(
        spec,
        nvars=4,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"enum_max_depth": 4, "enum_max_trees": 64},
    )

    assert status_generic == "direct_ok"
    assert int(meta_generic.get("candidate_count_raw", 0)) >= 2
    assert len(rows_generic) >= 1
    assert rows_generic[0]["local_mapping_kind"] == "direct_exp_add_head"


def test_direct_log_add_can_recover_log_refine_anchor(monkeypatch):
    anchor = ("var", 2)
    spec = scaffold_mod.OuterScaffoldSpec(
        family="log",
        scaffold_id="log:add:x2",
        parent_node=("add", ("log", ("const", 1.0)), anchor),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=anchor,
        metadata={"form": "log_add"},
    )

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [
            ("enum", ("var", 0)),
            ("enum", ("mul", ("var", 0), ("var", 1))),
        ], {"candidate_source_counts": {"enum": 2}}

    monkeypatch.setattr(closure_compat_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    n = 32
    x0 = torch.linspace(0.5, 1.7, steps=n, dtype=torch.float64)
    x1 = torch.linspace(0.6, 1.8, steps=n, dtype=torch.float64)
    x2 = torch.linspace(0.4, 1.2, steps=n, dtype=torch.float64)
    x = torch.stack([x0, x1, x2], dim=1)
    y = (torch.log(x0 * x1) + x2).unsqueeze(-1)

    rows, status, meta = scaffold_mod._solve_direct_log_preview_rows(
        spec,
        nvars=3,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"enum_max_depth": 4, "enum_max_trees": 64},
    )

    assert status == "direct_ok"
    assert int(meta.get("candidate_count_raw", 0)) >= 2
    assert len(rows) >= 1
    assert rows[0]["local_mapping_kind"] == "direct_log_add_head"
    expr_text = scaffold_mod.node_str(rows[0]["expr"])
    assert "log(" in expr_text
    assert "x2" in expr_text
    assert float(rows[0]["local_probe_mse"]) < 1.0e-12


def test_direct_log_add_strips_scalar_factor_inside_log(monkeypatch):
    anchor = ("var", 2)
    spec = scaffold_mod.OuterScaffoldSpec(
        family="log",
        scaffold_id="log:add:x2",
        parent_node=("add", ("log", ("const", 1.0)), anchor),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=anchor,
        metadata={"form": "log_add"},
    )

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [
            ("enum", ("mul", ("exp", ("const", 1.0)), ("mul", ("var", 0), ("var", 1)))),
        ], {"candidate_source_counts": {"enum": 1}}

    monkeypatch.setattr(closure_compat_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    n = 24
    x0 = torch.linspace(0.5, 1.7, steps=n, dtype=torch.float64)
    x1 = torch.linspace(0.6, 1.8, steps=n, dtype=torch.float64)
    x2 = torch.linspace(0.4, 1.2, steps=n, dtype=torch.float64)
    x = torch.stack([x0, x1, x2], dim=1)
    y = (torch.log(x0 * x1) + x2).unsqueeze(-1)

    rows, status, _meta = scaffold_mod._solve_direct_log_preview_rows(
        spec,
        nvars=3,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"enum_max_depth": 4, "enum_max_trees": 64},
    )

    assert status == "direct_ok"
    assert rows
    assert any(scaffold_mod.node_str(row["expr"]) in {
        "(log((x0*x1))+x2)",
        "(x2+log((x0*x1)))",
    } for row in rows)
    assert float(min(row["local_probe_mse"] for row in rows)) < 1.0e-12


def test_outer_scaffold_pass_uses_route_specific_beam_limits(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_outer_scaffold_pass(**kwargs):
        captured["beam_cfg"] = dict(kwargs.get("beam_cfg", {}) or {})
        captured["deadline_s"] = kwargs.get("deadline_s", None)
        return {"candidate_rows": [], "stats": {}}

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_outer_scaffold_pass)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, :1],
        nvars=1,
        n_iter=0,
        max_depth=3,
        poly_degree=1,
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
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["periodic"],
        closure_search_max_proposals=1,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=1,
        closure_search_exact_topk=1,
        closure_search_min_valid_frac=0.07,
        closure_search_min_confidence=0.03,
        closure_search_periodic_min_valid_scale=1.1,
        closure_search_periodic_min_confidence_scale=1.2,
        closure_search_transport_min_lin_rel=0.01,
        wall_time_limit_s=20.0,
    )

    beam_cfg = dict(captured.get("beam_cfg", {}) or {})
    stats = getattr(arch, "closure_search_stats", {})

    assert float(beam_cfg.get("min_valid_frac", -1.0)) == 0.07
    assert float(beam_cfg.get("min_confidence", -1.0)) == 0.03
    assert float(beam_cfg.get("periodic_min_valid_scale", -1.0)) == 1.1
    assert float(beam_cfg.get("periodic_min_confidence_scale", -1.0)) == 1.2
    assert float(beam_cfg.get("transport_min_lin_rel", -1.0)) == 0.01
    assert captured.get("deadline_s", None) is not None
    assert float(stats.get("beam_min_valid_frac", -1.0)) == 0.07
    assert float(stats.get("beam_min_confidence", -1.0)) == 0.03
    assert 4.0 <= float(stats.get("wall_time_budget_s", 0.0)) <= 6.0
    assert float(stats.get("wall_time_budget_fraction", 0.0)) == 0.25


def test_outer_scaffold_pass_allocates_exact_budget_per_scaffold(monkeypatch):
    expr_a1 = ("sin", ("var", 0))
    expr_a2 = ("cos", ("var", 0))
    expr_b1 = ("sqr", ("var", 0))
    captured_scored: list[tuple] = []

    def _fake_run_outer_scaffold_pass(**kwargs):
        return {
            "candidate_rows": [
                {
                    "expr": expr_a1,
                    "scaffold_id": "A",
                    "child_key": str(expr_a1),
                    "local_probe_mse": 1.0e-6,
                    "local_fit_mse": 1.0e-6,
                    "candidate_child_size": 2,
                },
                {
                    "expr": expr_a2,
                    "scaffold_id": "A",
                    "child_key": str(expr_a2),
                    "local_probe_mse": 2.0e-6,
                    "local_fit_mse": 2.0e-6,
                    "candidate_child_size": 2,
                },
                {
                    "expr": expr_b1,
                    "scaffold_id": "B",
                    "child_key": str(expr_b1),
                    "local_probe_mse": 1.0e-2,
                    "local_fit_mse": 1.0e-2,
                    "candidate_child_size": 3,
                },
            ],
            "stats": {
                "families_considered": 1,
                "scaffolds_enumerated": 2,
                "scaffolds_considered": 2,
                "preview_calls": 2,
                "preview_candidates": 3,
                "status_counts": {"ok": 2},
            },
        }

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        captured_scored.append(node)
        score_map = {
            expr_a1: 0.4,
            expr_a2: 0.5,
            expr_b1: 0.1,
        }
        mse = float(score_map.get(node, 1.5))
        key = ("expr", str(node))
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return mse, key, z, mapping, node
        return mse, key, z, mapping

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_outer_scaffold_pass)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, :1],
        nvars=1,
        n_iter=0,
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
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["periodic"],
        closure_search_max_proposals=2,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=2,
        closure_search_exact_topk=2,
        _score_expr_fn=_fake_score_expr,
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    # With proposal-key grouping (not scaffold-id grouping), all three
    # candidates are distinct.  exact_topk=2 means only the two best by
    # preview MSE get scored: expr_a1 (1e-6) and expr_a2 (2e-6).
    assert expr_a1 in captured_scored
    assert expr_a2 in captured_scored
    assert expr_b1 not in captured_scored
    assert best.best_expr == expr_a1
    assert float(best.best_mse) == 0.4
    assert int(stats.get("scored", 0)) == 2


def test_outer_scaffold_exact_scoring_can_use_anchor_head(monkeypatch):
    expr = ("add", ("cos", ("var", 0)), ("var", 0))

    def _fake_run_outer_scaffold_pass(**kwargs):
        return {
            "candidate_rows": [
                {
                    "expr": expr,
                    "scaffold_id": "periodic:cos_add:x0",
                    "child_key": str(expr),
                    "local_probe_mse": 1.0e-6,
                    "local_fit_mse": 1.0e-6,
                    "candidate_child_size": 4,
                    "scaffold_anchor_node": ("var", 0),
                    "scaffold_metadata": {"form": "cos_add"},
                }
            ],
            "stats": {
                "families_considered": 1,
                "scaffolds_enumerated": 1,
                "scaffolds_considered": 1,
                "preview_calls": 1,
                "preview_candidates": 1,
                "status_counts": {"ok": 1},
            },
        }

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_outer_scaffold_pass)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: torch.cos(x[:, :1]) + 2.0 * x[:, :1],
        nvars=1,
        n_iter=0,
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
        refine_enable=False,
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["periodic"],
        closure_search_max_proposals=1,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=1,
        closure_search_exact_topk=1,
        score_mapping_family_mode="cheap",
        var_dims=[(0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert int(stats.get("anchor_head_attempts", 0)) == 1
    assert float(best.best_raw_mse) < 1.0e-8


def test_outer_scaffold_periodic_add_uses_direct_fill_without_inverse(monkeypatch):
    anchor = ("mul", ("var", 0), ("var", 1))
    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="periodic",
        scaffold_id="periodic:cos_add:(x0*x1)",
        parent_node=("add", ("cos", ("const", 1.0)), anchor),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=anchor,
        metadata={"form": "cos_add"},
    )
    op_app = operator_application_from_scaffold(scaffold_spec)

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    monkeypatch.setattr(runner_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: torch.cos(x[:, :1] * x[:, 1:2]) + 1.2 * (x[:, :1] * x[:, 1:2]),
        nvars=2,
        n_iter=0,
        max_depth=4,
        poly_degree=1,
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
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["periodic"],
        closure_search_max_proposals=1,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=4,
        closure_search_exact_topk=1,
        score_mapping_family_mode="cheap",
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert int(stats.get("preview_calls", 0)) >= 1
    assert int(stats.get("direct_calls", 0)) >= 1
    assert int(stats.get("preview_candidates", 0)) >= 1
    assert int(stats.get("scored", 0)) >= 1
    assert int(dict(stats.get("status_counts", {}) or {}).get("direct_ok", 0)) >= 1
    best_expr_str = scaffold_mod.node_str(best.best_expr)
    assert "cos((x0*x1))" in best_expr_str
    assert "x0" in best_expr_str and "x1" in best_expr_str
    assert float(best.best_raw_mse) < 1.0e-8


def test_outer_scaffold_exp_add_uses_direct_fill_without_inverse(monkeypatch):
    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="exp",
        scaffold_id="exp:add:x1",
        parent_node=("add", ("exp", ("const", 1.0)), ("var", 1)),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 1),
        metadata={"form": "exp_add"},
    )
    op_app = operator_application_from_scaffold(scaffold_spec)

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", ("var", 0))], {"candidate_source_counts": {"enum": 1}}

    monkeypatch.setattr(runner_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)
    monkeypatch.setattr(runner_mod, "collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: torch.exp(x[:, :1]) + x[:, 1:2],
        nvars=2,
        n_iter=0,
        max_depth=4,
        poly_degree=1,
        lo=0.2,
        hi=1.2,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        no_residual=True,
        no_crossover=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
        refine_enable=False,
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["exp"],
        closure_search_max_proposals=2,
        closure_search_anchors_per_family=2,
        closure_search_preview_topk=4,
        closure_search_exact_topk=1,
        score_mapping_family_mode="cheap",
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert int(stats.get("preview_calls", 0)) >= 1
    assert int(stats.get("direct_calls", 0)) >= 1
    assert int(stats.get("preview_candidates", 0)) >= 1
    assert int(stats.get("scored", 0)) >= 1
    assert int(dict(stats.get("status_counts", {}) or {}).get("direct_ok", 0)) >= 1
    assert float(best.best_mse) < 1.0e-20
    assert float(best.best_raw_mse) < 1.0e-8


def test_outer_scaffold_log_add_uses_direct_fill_without_inverse(monkeypatch):
    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="log",
        scaffold_id="log:add:x2",
        parent_node=("add", ("log", ("const", 1.0)), ("var", 2)),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 2),
        metadata={"form": "log_add"},
    )
    op_app = operator_application_from_scaffold(scaffold_spec)

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", ("mul", ("var", 0), ("var", 1)))], {"candidate_source_counts": {"enum": 1}}

    monkeypatch.setattr(runner_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)
    monkeypatch.setattr(runner_mod, "collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: torch.log(x[:, :1] * x[:, 1:2]) + x[:, 2:3],
        nvars=3,
        n_iter=0,
        max_depth=5,
        poly_degree=1,
        lo=0.5,
        hi=1.8,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        no_residual=True,
        no_crossover=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
        refine_enable=False,
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["log"],
        closure_search_max_proposals=2,
        closure_search_anchors_per_family=3,
        closure_search_preview_topk=4,
        closure_search_exact_topk=1,
        score_mapping_family_mode="cheap",
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert int(stats.get("preview_calls", 0)) >= 1
    assert int(stats.get("direct_calls", 0)) >= 1
    assert int(stats.get("preview_candidates", 0)) >= 1
    assert int(stats.get("scored", 0)) >= 1
    assert int(dict(stats.get("status_counts", {}) or {}).get("direct_ok", 0)) >= 1
    assert float(best.best_mse) < 1.0e-20
    assert float(best.best_raw_mse) < 1.0e-8


def test_outer_scaffold_quadratic_sqrt_mul_uses_direct_fill_without_inverse(monkeypatch):
    bound_closure = make_direct_quadratic_closure(
        scaffold_id="quadratic:sqrt_mul:x0",
        quadratic_kind="sqrt_mul",
        base_nodes=(("var", 1), ("var", 2), ("var", 3)),
        anchor_node=("var", 0),
    )
    op_app = OperatorApplication(
        family="quadratic",
        operator_id="quadratic:sqrt_mul",
        target_mode="robust",
        bound_closure=bound_closure,
        metadata={"form": "quadratic_sqrt_mul"},
    )

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    monkeypatch.setattr(runner_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, :1] * torch.sqrt(torch.sum(x[:, 1:] ** 2, dim=1, keepdim=True)),
        nvars=4,
        n_iter=0,
        max_depth=8,
        poly_degree=1,
        lo=0.5,
        hi=1.5,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        no_residual=True,
        no_crossover=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
        refine_enable=False,
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["quadratic"],
        closure_search_max_proposals=2,
        closure_search_anchors_per_family=2,
        closure_search_preview_topk=4,
        closure_search_exact_topk=1,
        score_mapping_family_mode="cheap",
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert int(stats.get("preview_calls", 0)) >= 1
    assert int(stats.get("direct_calls", 0)) >= 1
    assert int(stats.get("preview_candidates", 0)) >= 1
    assert int(stats.get("scored", 0)) >= 1
    assert int(dict(stats.get("status_counts", {}) or {}).get("direct_ok", 0)) >= 1
    best_expr_str = scaffold_mod.node_str(best.best_expr)
    assert "sqrt(" in best_expr_str
    assert "x0" in best_expr_str
    assert "sqr(x1)" in best_expr_str
    assert float(best.best_raw_mse) < 1.0e-10


def test_outer_scaffold_power_invsqrt_mul_uses_direct_fill_without_inverse(monkeypatch):
    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="power",
        scaffold_id="power:invsqrt_mul:x0",
        parent_node=("mul", ("var", 0), ("div", ("const", 1.0), ("sqrt", ("const", 1.0)))),
        hole_path=(2, 2, 1),
        target_mode="robust",
        anchor_node=("var", 0),
        metadata={"form": "power_invsqrt_mul"},
    )
    op_app = operator_application_from_scaffold(scaffold_spec)

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    monkeypatch.setattr(runner_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, :1] / torch.sqrt(1.0 + x[:, 1:2] ** 2),
        nvars=2,
        n_iter=0,
        max_depth=8,
        poly_degree=1,
        lo=0.4,
        hi=2.0,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        no_residual=True,
        no_crossover=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
        refine_enable=False,
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["power"],
        closure_search_max_proposals=2,
        closure_search_anchors_per_family=2,
        closure_search_preview_topk=4,
        closure_search_exact_topk=1,
        score_mapping_family_mode="cheap",
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert int(stats.get("preview_calls", 0)) >= 1
    assert int(stats.get("direct_calls", 0)) >= 1
    assert int(stats.get("preview_candidates", 0)) >= 1
    assert int(stats.get("scored", 0)) >= 1
    assert int(dict(stats.get("status_counts", {}) or {}).get("direct_ok", 0)) >= 1
    best_expr_str = scaffold_mod.node_str(best.best_expr)
    assert "sqrt(" in best_expr_str
    assert "x0" in best_expr_str
    assert "sqr(x1)" in best_expr_str
    assert float(best.best_raw_mse) < 1.0e-10


def test_direct_periodic_add_can_lift_anchor_with_unused_variable(monkeypatch):
    anchor = ("mul", ("var", 0), ("var", 1))
    hole = ("mul", ("var", 0), ("var", 1))
    spec = scaffold_mod.OuterScaffoldSpec(
        family="periodic",
        scaffold_id="periodic:cos_add:(x0*x1)",
        parent_node=("add", ("cos", ("const", 1.0)), anchor),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=anchor,
        metadata={"form": "cos_add"},
    )

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", hole)], {"candidate_source_counts": {"enum": 1}}

    monkeypatch.setattr(closure_compat_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    n = 32
    x0 = torch.linspace(0.2, 0.8, steps=n, dtype=torch.float64)
    x1 = torch.linspace(0.3, 0.9, steps=n, dtype=torch.float64)
    x2 = torch.linspace(1.5, 2.7, steps=n, dtype=torch.float64)
    x3 = torch.full((n,), 1.2, dtype=torch.float64)
    x = torch.stack([x0, x1, x2, x3], dim=1)
    y = (torch.cos((x0 * x1)).unsqueeze(-1) + (x3 * x0 * x1).unsqueeze(-1))

    rows, status, meta = scaffold_mod._solve_direct_periodic_add_preview_rows(
        spec,
        nvars=4,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"enum_max_depth": 4, "enum_max_trees": 64},
    )

    assert status == "direct_ok"
    assert int(meta.get("anchor_lift_attempts", 0)) >= 1
    assert int(meta.get("anchor_lift_applied", 0)) >= 1
    assert rows
    assert rows[0]["local_mapping_kind"] == "direct_anchor_lift"
    assert int(rows[0]["direct_metadata"]["anchor_lift_var_idx"]) == 3
    assert "x3" in scaffold_mod.node_str(rows[0]["expr"])
    assert float(rows[0]["local_probe_mse"]) < 1.0e-12


def test_direct_periodic_harmonic_closure_can_fit_envelope_and_companions(monkeypatch):
    spec = scaffold_mod.OuterScaffoldSpec(
        family="periodic",
        scaffold_id="periodic:cos_add:x0",
        parent_node=("add", ("cos", ("const", 1.0)), ("var", 0)),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 0),
        metadata={"form": "cos_add"},
    )

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", ("var", 2))], {"candidate_source_counts": {"enum": 1}}

    monkeypatch.setattr(closure_compat_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    x0 = torch.tensor([0.27, 0.44, 0.63, 0.88, 1.11, 0.39, 0.72, 1.24, 0.58, 0.95, 1.31, 0.81], dtype=torch.float64)
    x1 = torch.tensor([0.41, 0.92, 0.57, 1.36, 0.74, 1.12, 0.48, 1.27, 0.83, 0.66, 1.43, 1.05], dtype=torch.float64)
    x2 = torch.tensor([0.19, 0.54, 1.11, 0.73, 1.37, 0.28, 0.96, 1.22, 0.47, 1.04, 0.82, 1.29], dtype=torch.float64)
    x = torch.stack([x0, x1, x2], dim=1)
    y = (2.0 * torch.sqrt(x0 * x1) * torch.cos(x2) + x0 + x1).unsqueeze(-1)

    rows, status, meta = scaffold_mod._solve_direct_periodic_add_preview_rows(
        spec,
        nvars=3,
        max_depth=7,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(0.0,)],
        preview_topk=4,
        solver_kwargs={
            "enum_max_depth": 3,
            "enum_max_trees": 16,
            "direct_periodic_seed_topk": 5,
            "direct_periodic_envelope_topk": 4,
            "direct_periodic_companion_topk": 2,
        },
    )

    assert status == "direct_ok"
    assert int(meta.get("harmonic_candidate_count_scored", 0)) >= 1
    assert len(rows) >= 1
    assert rows[0]["local_mapping_kind"] == "direct_harmonic_head"
    assert float(rows[0]["local_probe_mse"]) < 1.0e-12
    expr_text = scaffold_mod.node_str(rows[0]["expr"])
    assert "sqrt((x0*x1))" in expr_text
    assert "cos(x2)" in expr_text
    assert "x0" in expr_text
    assert "x1" in expr_text


def test_outer_scaffold_anchor_head_compare_logs_delta(monkeypatch):
    anchor = ("mul", ("var", 0), ("var", 1))
    expr = ("add", ("cos", ("var", 0)), anchor)

    def _fake_run_outer_scaffold_pass(**kwargs):
        return {
            "candidate_rows": [
                {
                    "expr": expr,
                    "scaffold_id": "periodic:cos_add:x0",
                    "child_key": str(expr),
                    "local_probe_mse": 1.0e-6,
                    "local_fit_mse": 1.0e-6,
                    "candidate_child_size": 4,
                    "scaffold_anchor_node": anchor,
                    "scaffold_anchor_expr": "(x0*x1)",
                    "scaffold_metadata": {"form": "cos_add"},
                }
            ],
            "stats": {
                "families_considered": 1,
                "scaffolds_enumerated": 1,
                "scaffolds_considered": 1,
                "preview_calls": 1,
                "preview_candidates": 1,
                "status_counts": {"ok": 1},
            },
        }

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        refine_cfg = dict(kwargs.get("refine_cfg", {}) or {})
        head_terms = list(refine_cfg.get("score_head_var_terms", []) or [])
        has_anchor = any(term == anchor for term in head_terms)
        mse = 0.1 if has_anchor else 0.4
        key = ("expr", str(node), "anchor" if has_anchor else "base")
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return mse, key, z, mapping, node
        return mse, key, z, mapping

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_outer_scaffold_pass)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: torch.cos(x[:, :1]) + 2.0 * (x[:, :1] * x[:, 1:2]),
        nvars=2,
        n_iter=0,
        max_depth=3,
        poly_degree=1,
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
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["periodic"],
        closure_search_max_proposals=1,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=1,
        closure_search_exact_topk=1,
        closure_search_anchor_head_compare_enable=True,
        _score_expr_fn=_fake_score_expr,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    examples = list(stats.get("anchor_head_compare_examples", []) or [])
    best = arch.best(1)[0]

    assert int(stats.get("anchor_head_compare_attempts", 0)) == 1
    assert int(stats.get("anchor_head_compare_improved", 0)) == 1
    assert float(stats.get("anchor_head_compare_delta_sum", 0.0)) > 0.0
    assert len(examples) == 1
    assert float(examples[0]["base_raw_mse"]) == 0.4
    assert float(examples[0]["anchor_head_raw_mse"]) == 0.1
    assert float(best.best_raw_mse) == 0.1


def test_run_outer_scaffold_pass_skips_enumeration_after_deadline(monkeypatch):
    def _should_not_run(**kwargs):
        raise AssertionError("enumeration should be skipped once the scaffold deadline is exceeded")

    monkeypatch.setattr(scaffold_mod, "enumerate_outer_scaffold_specs", _should_not_run)

    ret = scaffold_mod.run_outer_scaffold_pass(
        families=["periodic"],
        nvars=1,
        max_scaffolds=4,
        anchors_per_family=1,
        max_depth=3,
        poly_degree=1,
        x_fit=torch.zeros((4, 1), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 1), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        var_dims=None,
        y_dims=None,
        pool_nodes=[],
        pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 0), dtype=torch.float64),
        pool_dims=[],
        safe_eps=1.0e-6,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        deadline_s=time.perf_counter() - 1.0,
    )

    stats = dict(ret.get("stats", {}) or {})
    assert ret.get("candidate_rows", []) == []
    assert bool(stats.get("deadline_exceeded", False)) is True
    assert int(dict(stats.get("status_counts", {}) or {}).get("deadline_exceeded", 0)) == 1


def test_run_outer_scaffold_pass_impl_does_not_silently_fallback_to_legacy_enum():
    def _operator_enum(**_kwargs):
        return []

    def _legacy_enum(**_kwargs):
        raise AssertionError("legacy scaffold enumeration should not run without explicit opt-in")

    ret = run_outer_scaffold_pass_impl_native(
        families=["periodic"],
        nvars=1,
        max_scaffolds=4,
        anchors_per_family=1,
        max_depth=3,
        poly_degree=1,
        x_fit=torch.zeros((4, 1), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 1), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        var_dims=None,
        y_dims=None,
        pool_nodes=[],
        pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 0), dtype=torch.float64),
        pool_dims=[],
        safe_eps=1.0e-6,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        enumerate_operator_applications_fn=_operator_enum,
        enumerate_closure_search_specs_fn=_legacy_enum,
        prefer_legacy_scaffold_enumeration=False,
    )

    stats = dict(ret.get("stats", {}) or {})
    assert ret.get("candidate_rows", []) == []
    assert int(stats.get("scaffolds_enumerated", 0) or 0) == 0
    assert bool(stats.get("compat_legacy_enumeration_used", False)) is False


def test_run_outer_scaffold_pass_impl_uses_native_enum():
    """Native operator enumeration produces candidates via direct route."""
    spec = OuterScaffoldSpec(
        family="periodic",
        scaffold_id="periodic:cos",
        parent_node=("cos", ("var", 0)),
        hole_path=(1,),
        target_mode="robust",
        metadata={"form": "cos_base", "operator": "periodic:cos_base"},
    )
    op_app = operator_application_from_scaffold(spec)

    def _native_enum(**_kwargs):
        return [op_app]

    def _score_direct(spec, **_kwargs):
        return (
            [
                {
                    "expr": ("cos", ("var", 0)),
                    "local_probe_mse": 0.0,
                    "local_fit_mse": 0.0,
                    "candidate_child_size": 2,
                    "child_key": "cos(x0)",
                }
            ],
            "direct_ok",
            {},
        )

    ret = run_outer_scaffold_pass_impl_native(
        families=["periodic"],
        nvars=1,
        max_scaffolds=4,
        anchors_per_family=1,
        max_depth=3,
        poly_degree=1,
        x_fit=torch.zeros((4, 1), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 1), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        var_dims=None,
        y_dims=None,
        pool_nodes=[],
        pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 0), dtype=torch.float64),
        pool_dims=[],
        safe_eps=1.0e-6,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        enumerate_operator_applications_fn=_native_enum,
        solve_direct_operator_preview_rows_fn=_score_direct,
    )

    stats = dict(ret.get("stats", {}) or {})
    rows = list(ret.get("candidate_rows", []) or [])
    assert stats.get("proposal_object_mode") == "operator_native"
    assert int(stats.get("scaffolds_enumerated", 0) or 0) == 1
    assert len(rows) == 1
    assert str(rows[0].get("scaffold_id", "")) == "periodic:cos"


def test_run_outer_scaffold_pass_impl_core_lane_uses_canonical_pool():
    lane_calls = []

    def _native_enum(*, basis_seed_mode="merged", pool_nodes=None, basis_state=None, basis_state_beam=None, **_kwargs):
        lane_calls.append(
            {
                "basis_seed_mode": str(basis_seed_mode),
                "pool_nodes": tuple(pool_nodes or ()),
                "basis_state": basis_state,
                "basis_state_beam": tuple(basis_state_beam or ()),
            }
        )
        expr = ("var", 0) if str(basis_seed_mode) == "core_only" else ("var", 1)
        return [
            OperatorApplication(
                family="quadratic",
                operator_id="quadratic:sqrt_mul",
                scaffold_id=f"{str(basis_seed_mode)}:demo",
                parent_node=expr,
                hole_path=(),
            )
        ]

    def _score_direct(spec, **_kwargs):
        expr = spec.parent_node
        return (
            [
                {
                    "expr": expr,
                    "local_probe_mse": 0.0 if expr == ("var", 0) else 1.0,
                    "local_fit_mse": 0.0 if expr == ("var", 0) else 1.0,
                    "candidate_child_size": 1,
                    "child_key": str(scaffold_mod.node_str(expr)),
                }
            ],
            "direct_ok",
            {},
        )

    ret = run_outer_scaffold_pass_impl_native(
        families=["quadratic"],
        nvars=2,
        max_scaffolds=4,
        anchors_per_family=2,
        max_depth=3,
        poly_degree=1,
        x_fit=torch.zeros((4, 2), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 2), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        var_dims=None,
        y_dims=None,
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_phi_fit=torch.zeros((4, 1), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 1), dtype=torch.float64),
        pool_dims=[None],
        safe_eps=1.0e-6,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        proposal_context=ProposalContext(
            basis_state=BasisState(blocks=()),
            basis_state_beam=(),
            diagnostics={"route": "test"},
            family_hints={},
        ),
        enumerate_operator_applications_fn=_native_enum,
        solve_direct_operator_preview_rows_fn=_score_direct,
    )

    rows = list(ret.get("candidate_rows", []) or [])
    assert len(lane_calls) == 2
    core_call, aug_call = lane_calls
    assert core_call["basis_seed_mode"] == "core_only"
    assert core_call["basis_state"] is None
    assert core_call["basis_state_beam"] == ()
    assert ("var", 0) in core_call["pool_nodes"]
    assert core_call["pool_nodes"] != (("mul", ("var", 0), ("var", 1)),)
    assert aug_call["basis_seed_mode"] == "basis_augmented"
    assert aug_call["pool_nodes"] == (("mul", ("var", 0), ("var", 1)),)
    assert {str(row.get("proposal_lane", "")) for row in rows} == {"core", "basis_augmented"}


def test_run_outer_scaffold_pass_impl_applies_family_budget_plan_per_family():
    periodic_spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_base",
        scaffold_id="periodic:demo",
        parent_node=("cos", ("var", 0)),
        hole_path=(1,),
    )
    quadratic_spec = OperatorApplication(
        family="quadratic",
        operator_id="quadratic:sqrt_mul",
        scaffold_id="quadratic:demo",
        parent_node=("mul", ("var", 0), ("sqrt", ("sqr", ("var", 1)))),
        hole_path=(2, 1),
    )
    enum_calls = []

    def _native_enum(*, families=None, max_scaffolds=0, basis_seed_mode="merged", **_kwargs):
        enum_calls.append((tuple(families or ()), int(max_scaffolds), str(basis_seed_mode)))
        fams = tuple(families or ())
        if fams == ("periodic",):
            return [periodic_spec]
        if fams == ("quadratic",):
            return [quadratic_spec]
        return [periodic_spec]

    def _family_allocator_fn(*, families=None, max_scaffolds=0, anchors_per_family=0, context=None):
        assert tuple(families or ()) == ("periodic", "quadratic")
        assert int(max_scaffolds) == 2
        return {
            "steered": True,
            "scores": {"periodic": 1.0, "quadratic": 1.0},
            "entries": [
                {
                    "family": "periodic",
                    "max_scaffolds": 1,
                    "anchors_per_family": int(anchors_per_family),
                    "priority_score": 1.0,
                    "reason": "test",
                },
                {
                    "family": "quadratic",
                    "max_scaffolds": 1,
                    "anchors_per_family": int(anchors_per_family),
                    "priority_score": 1.0,
                    "reason": "test",
                },
            ],
        }

    def _score_direct(spec, **_kwargs):
        expr = spec.parent_node
        return (
            [
                {
                    "expr": expr,
                    "local_probe_mse": 0.0,
                    "local_fit_mse": 0.0,
                    "candidate_child_size": 1,
                    "child_key": str(scaffold_mod.node_str(expr)),
                }
            ],
            "direct_ok",
            {},
        )

    ret = run_outer_scaffold_pass_impl_native(
        families=["periodic", "quadratic"],
        nvars=2,
        max_scaffolds=2,
        anchors_per_family=1,
        max_depth=3,
        poly_degree=1,
        x_fit=torch.zeros((4, 2), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 2), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        var_dims=None,
        y_dims=None,
        pool_nodes=[],
        pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 0), dtype=torch.float64),
        pool_dims=[],
        safe_eps=1.0e-6,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        enumerate_operator_applications_fn=_native_enum,
        solve_direct_operator_preview_rows_fn=_score_direct,
        family_allocator_fn=_family_allocator_fn,
    )

    rows = list(ret.get("candidate_rows", []) or [])
    assert [call[0] for call in enum_calls] == [("periodic",), ("quadratic",)]
    assert {str(row.get("scaffold_family", "")) for row in rows} == {"periodic", "quadratic"}


def test_run_outer_scaffold_pass_impl_keeps_same_expr_from_distinct_operator_families():
    shared_expr = ("mul", ("var", 0), ("sqrt", ("sqr", ("var", 1))))
    power_spec = OperatorApplication(
        family="power",
        operator_id="power:sqrt_mul",
        scaffold_id="power:demo",
        parent_node=shared_expr,
        hole_path=(2, 1),
    )
    quadratic_spec = OperatorApplication(
        family="quadratic",
        operator_id="quadratic:sqrt_mul",
        scaffold_id="quadratic:demo",
        parent_node=shared_expr,
        hole_path=(2, 1),
    )

    def _native_enum(*, families=None, **_kwargs):
        fams = tuple(families or ())
        if fams == ("power",):
            return [power_spec]
        if fams == ("quadratic",):
            return [quadratic_spec]
        return []

    def _family_allocator_fn(*, families=None, max_scaffolds=0, anchors_per_family=0, context=None):
        assert tuple(families or ()) == ("power", "quadratic")
        assert int(max_scaffolds) == 2
        return {
            "steered": True,
            "scores": {"power": 1.0, "quadratic": 1.0},
            "entries": [
                {
                    "family": "power",
                    "max_scaffolds": 1,
                    "anchors_per_family": int(anchors_per_family),
                    "priority_score": 1.0,
                    "reason": "test",
                },
                {
                    "family": "quadratic",
                    "max_scaffolds": 1,
                    "anchors_per_family": int(anchors_per_family),
                    "priority_score": 1.0,
                    "reason": "test",
                },
            ],
        }

    def _score_direct(spec, **_kwargs):
        return (
            [
                {
                    "expr": spec.parent_node,
                    "local_probe_mse": 0.0,
                    "local_fit_mse": 0.0,
                    "candidate_child_size": 1,
                    "proposal_key": f"{spec.family}:{scaffold_mod.node_str(spec.parent_node)}",
                    "child_key": f"{spec.family}:{scaffold_mod.node_str(spec.parent_node)}",
                }
            ],
            "direct_ok",
            {},
        )

    ret = run_outer_scaffold_pass_impl_native(
        families=["power", "quadratic"],
        nvars=2,
        max_scaffolds=2,
        anchors_per_family=1,
        max_depth=3,
        poly_degree=1,
        x_fit=torch.zeros((4, 2), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 2), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        var_dims=None,
        y_dims=None,
        pool_nodes=[],
        pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 0), dtype=torch.float64),
        pool_dims=[],
        safe_eps=1.0e-6,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        enumerate_operator_applications_fn=_native_enum,
        solve_direct_operator_preview_rows_fn=_score_direct,
        family_allocator_fn=_family_allocator_fn,
    )

    rows = list(ret.get("candidate_rows", []) or [])
    assert {str(row.get("scaffold_id", "")) for row in rows} == {"power:demo", "quadratic:demo"}


def test_direct_rational_affine_can_recover_shifted_ratio(monkeypatch):
    spec = scaffold_mod.OuterScaffoldSpec(
        family="rational",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"form": "rational_affine"},
    )

    def _fake_collect_direct_hole_candidates(**kwargs):
        target_dim = kwargs.get("target_dim", None)
        if target_dim == (0.0,):
            return [
                ("enum", ("var", 0)),
                ("enum", ("mul", ("var", 0), ("var", 1))),
            ], {"candidate_source_counts": {"enum": 2}}
        return [], {"candidate_source_counts": {}}

    monkeypatch.setattr(closure_compat_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

    n = 32
    x0 = torch.linspace(0.3, 1.8, steps=n, dtype=torch.float64)
    x1 = torch.linspace(0.4, 1.9, steps=n, dtype=torch.float64)
    x = torch.stack([x0, x1], dim=1)
    y = ((1.0 + x0) / (1.0 + x0 * x1)).unsqueeze(-1)

    rows, status, meta = scaffold_mod._solve_direct_rational_affine_preview_rows(
        spec,
        nvars=2,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"enum_max_depth": 4, "enum_max_trees": 64, "safe_eps": 1.0e-6},
    )

    assert status == "direct_ok"
    assert int(meta.get("u_shortlist_count", 0)) >= 2
    assert int(meta.get("v_shortlist_count", 0)) >= 2
    assert len(rows) >= 1
    assert rows[0]["local_mapping_kind"] == "direct_rational_head"
    assert float(rows[0]["local_probe_mse"]) < 1.0e-12


def test_direct_operator_affine_latent_recovers_two_term_linear_combo():
    apps = scaffold_mod.enumerate_operator_applications(
        families=["affine"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=2,
        max_scaffolds=16,
    )
    spec = next(
        app
        for app in apps
        if {getattr(term, "node", None) for term in tuple(app.bindings.get("terms", ()) or ())}
        == {("var", 0), ("var", 1)}
    )

    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (1.5 * x[:, 0] - 2.0 * x[:, 1] + 0.25).unsqueeze(-1)

    rows, status, meta = scaffold_mod._solve_direct_operator_preview_rows(
        spec,
        nvars=2,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert int(meta.get("term_count", 0) or 0) == 2
    assert rows
    assert rows[0]["local_mapping_kind"] == "direct_affine_head"
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    expr_text = scaffold_mod.node_str(rows[0]["expr"])
    assert "x0" in expr_text
    assert "x1" in expr_text


def test_direct_operator_wrapper_routes_affine_via_native_planner():
    """An OperatorApplication with affine family is handled by the unified direct impl."""
    apps = scaffold_mod.enumerate_operator_applications(
        families=["affine"],
        nvars=1,
        y_dims=(0.0,),
        var_dims=[(0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=1,
        max_scaffolds=8,
    )
    spec = next(app for app in apps if app.family == "affine")

    x = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.float64)
    y = (1.5 * x[:, 0] + 0.25).unsqueeze(-1)

    rows, status, meta = scaffold_mod._solve_direct_operator_preview_rows(
        spec,
        nvars=1,
        max_depth=3,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=1,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_wrapper_keeps_native_operator_on_native_executor(monkeypatch):
    apps = scaffold_mod.enumerate_operator_applications(
        families=["affine"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=2,
        max_scaffolds=16,
    )
    spec = next(
        app
        for app in apps
        if {getattr(term, "node", None) for term in tuple(app.bindings.get("terms", ()) or ())}
        == {("var", 0), ("var", 1)}
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("native OperatorApplication should not route through legacy affine wrapper")

    monkeypatch.setattr(scaffold_mod, "_solve_direct_affine_preview_rows", _should_not_run)

    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (1.5 * x[:, 0] - 2.0 * x[:, 1] + 0.25).unsqueeze(-1)

    rows, status, meta = scaffold_mod._solve_direct_operator_preview_rows(
        spec,
        nvars=2,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert int(meta.get("term_count", 0) or 0) == 2


def test_direct_operator_native_planner_does_not_call_family_solver_wrapper(monkeypatch):
    apps = scaffold_mod.enumerate_operator_applications(
        families=["affine"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=2,
        max_scaffolds=16,
    )
    spec = next(
        app
        for app in apps
        if {getattr(term, "node", None) for term in tuple(app.bindings.get("terms", ()) or ())}
        == {("var", 0), ("var", 1)}
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("native planner should not call family solve_direct_affine_preview_rows wrapper")

    monkeypatch.setattr(direct_mod, "solve_direct_affine_preview_rows", _should_not_run)

    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (1.5 * x[:, 0] - 2.0 * x[:, 1] + 0.25).unsqueeze(-1)

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=2,
        max_depth=5,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert int(meta.get("term_count", 0) or 0) == 2


def test_direct_operator_exact_bound_power_preserves_bound_carrier():
    carrier = ("mul", ("var", 0), ("var", 1))
    spec = OperatorApplication(
        family="power",
        operator_id="power:sqrt",
        scaffold_id="power:sqrt:x0x1",
        parent_node=("sqrt", carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_power_closure(
            scaffold_id="power:sqrt:x0x1",
            power_kind="sqrt",
            exponent=0.5,
            hole_node=carrier,
        ),
        metadata={"operator_kind": "power_wrap", "power_kind": "sqrt"},
    )
    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = torch.sqrt(x[:, 0] * x[:, 1]).unsqueeze(-1)

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=2,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "exact_bound"
    assert rows[0]["direct_metadata"]["hole_node"] == carrier
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_power_rebinding_requires_explicit_opt_in():
    bound_carrier = ("var", 1)
    rebound_carrier = ("var", 0)
    spec = OperatorApplication(
        family="power",
        operator_id="power:sqrt",
        scaffold_id="power:sqrt:rebind",
        parent_node=("sqrt", bound_carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_power_closure(
            scaffold_id="power:sqrt:rebind",
            power_kind="sqrt",
            exponent=0.5,
            hole_node=bound_carrier,
        ),
        metadata={"operator_kind": "power_wrap", "power_kind": "sqrt"},
    )
    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = torch.sqrt(x[:, 0]).unsqueeze(-1)

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", rebound_carrier)], {"candidate_source_counts": {"enum": 1}}

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=2,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"allow_slot_rebinding": True},
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")).startswith("slot_search")
    assert rows[0]["direct_metadata"]["hole_node"] == rebound_carrier
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_exact_bound_periodic_preserves_bound_carrier_and_envelope():
    carrier = ("var", 2)
    envelope = ("sqrt", ("mul", ("var", 0), ("var", 1)))
    companions = (("var", 0), ("var", 1))
    spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_add",
        scaffold_id="periodic:cos_add:bound",
        parent_node=("add", ("cos", carrier), ("var", 0)),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos_add:bound",
            periodic_kind="cos",
            hole_node=carrier,
            feature_node=("cos", carrier),
            anchor_node=None,
            envelope_node=envelope,
            companion_nodes=companions,
        ),
        metadata={"operator_kind": "harmonic_wrap", "periodic_kind": "cos", "form": "cos_add"},
    )
    x = torch.tensor(
        [
            [0.2, 0.3, 0.4],
            [0.4, 0.5, 0.6],
            [0.6, 0.7, 0.8],
            [0.8, 0.9, 1.0],
        ],
        dtype=torch.float64,
    )
    y = (2.0 * torch.sqrt(x[:, 0] * x[:, 1]) * torch.cos(x[:, 2]) + x[:, 0] + x[:, 1]).unsqueeze(-1)

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=3,
        max_depth=8,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "exact_bound"
    assert rows[0]["direct_metadata"]["hole_node"] == carrier
    assert rows[0]["direct_metadata"]["envelope_node"] == envelope
    assert tuple(rows[0]["direct_metadata"]["companion_nodes"]) == companions
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_periodic_rebinding_requires_explicit_opt_in():
    bound_carrier = ("var", 2)
    rebound_carrier = ("var", 0)
    spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_base",
        scaffold_id="periodic:cos_base:rebind",
        parent_node=("cos", bound_carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos_base:rebind",
            periodic_kind="cos",
            hole_node=bound_carrier,
            feature_node=("cos", bound_carrier),
            anchor_node=None,
        ),
        metadata={"operator_kind": "harmonic_wrap", "periodic_kind": "cos", "form": "cos_base"},
    )
    # The harmonic design is (envelope*cos, envelope*sin, companions..., 1), so it needs
    # at least as many rows as columns: fit_direct_linear_design rejects an
    # underdetermined head, and no slot candidate can score.
    grid = torch.linspace(0.2, 3.2, 16, dtype=torch.float64)
    x = torch.stack([grid, grid + 0.1, grid + 0.2], dim=1)
    y = torch.cos(x[:, 0]).unsqueeze(-1)

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", rebound_carrier)], {"candidate_source_counts": {"enum": 1}}

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=3,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"allow_slot_rebinding": True},
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")).startswith("slot_search")
    assert rows[0]["direct_metadata"]["hole_node"] == rebound_carrier
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_exact_bound_quadratic_preserves_bound_bases():
    base_nodes = (("var", 1), ("var", 2), ("var", 3))
    quad_parent = (
        "sqrt",
        (
            "add",
            ("sqr", ("var", 1)),
            ("add", ("sqr", ("var", 2)), ("sqr", ("var", 3))),
        ),
    )
    spec = OperatorApplication(
        family="quadratic",
        operator_id="quadratic:sqrt",
        scaffold_id="quadratic:sqrt:norm",
        parent_node=quad_parent,
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_quadratic_closure(
            scaffold_id="quadratic:sqrt:norm",
            quadratic_kind="sqrt",
            base_nodes=base_nodes,
        ),
        metadata={"operator_kind": "quadratic_wrap", "quadratic_kind": "sqrt"},
    )
    x = torch.tensor(
        [
            [0.4, 0.2, 0.3, 0.5],
            [0.5, 0.4, 0.6, 0.7],
            [0.8, 0.5, 0.7, 0.9],
            [1.0, 0.6, 0.8, 1.1],
        ],
        dtype=torch.float64,
    )
    y = torch.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2).unsqueeze(-1)

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=4,
        max_depth=8,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "exact_bound"
    assert tuple(rows[0]["direct_metadata"]["quadratic_base_nodes"]) == base_nodes
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_exact_bound_rational_preserves_bound_slots():
    numerator = ("var", 0)
    denominator = ("var", 1)
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", numerator, denominator),
        hole_path=(),
        target_mode="full",
        bound_closure=make_direct_rational_closure(
            scaffold_id="rational:affine",
            u_node=numerator,
            v_node=denominator,
        ),
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )
    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = ((1.0 + x[:, 0]) / (1.0 + x[:, 1])).unsqueeze(-1)

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=2,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "exact_bound"
    assert rows[0]["direct_metadata"]["u_node"] == numerator
    assert rows[0]["direct_metadata"]["v_node"] == denominator
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_rational_completion_fills_missing_slot_without_overwriting_bound_one():
    numerator = ("var", 0)
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine:partial",
        parent_node=("div", numerator, ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        bindings={"numerator": numerator},
        bound_closure=make_direct_rational_closure(
            scaffold_id="rational:affine:partial",
            u_node=numerator,
            v_node=("const", 0.0),
        ),
        metadata={"operator_kind": "fractional_head", "composition_mode": "fractional", "form": "rational_affine"},
    )
    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = ((1.0 + x[:, 0]) / (1.0 + x[:, 1])).unsqueeze(-1)

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", ("var", 1))], {"candidate_source_counts": {"enum": 1}}

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=2,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "slot_search"
    assert str(meta.get("completion_mode", "")) == "fill_denominator"
    assert str(meta.get("preserved_slot", "")) == "numerator"
    assert rows[0]["direct_metadata"]["u_node"] == numerator
    assert rows[0]["direct_metadata"]["v_node"] == ("var", 1)
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_unary_builder_returns_exact_bound_plan_when_carrier_is_bound():
    carrier = ("var", 0)
    spec = OperatorApplication(
        family="exp",
        operator_id="exp:base",
        scaffold_id="exp:base:x0",
        parent_node=("exp", carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_linear_wrap_closure(
            scaffold_id="exp:base:x0",
            family="exp",
            wrap_kind="base",
            wrap_op="exp",
            hole_node=carrier,
            feature_node=("exp", carrier),
        ),
        metadata={"operator_kind": "unary_wrap", "wrap_op": "exp", "exp_kind": "base"},
    )
    x = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.float64)

    plan = direct_mod.build_direct_unary_linear_search_plan(
        spec,
        family="exp",
        kind="base",
        wrap_op="exp",
        hole_transform_fn=lambda node, _x: node,
        nvars=1,
        max_depth=4,
        x_fit=x,
        y_fit=torch.exp(x[:, :1]),
        x_probe=x,
        y_probe=torch.exp(x[:, :1]),
        var_dims=[(0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert isinstance(plan, PreparedCandidatesSearchPlan)
    assert str(plan.meta.get("execution_mode", "")) == "exact_bound"


def test_direct_power_builder_returns_exact_bound_plan_when_carrier_is_bound():
    carrier = ("mul", ("var", 0), ("var", 1))
    spec = OperatorApplication(
        family="power",
        operator_id="power:sqrt",
        scaffold_id="power:sqrt:x0x1",
        parent_node=("sqrt", carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_power_closure(
            scaffold_id="power:sqrt:x0x1",
            power_kind="sqrt",
            exponent=0.5,
            hole_node=carrier,
        ),
        metadata={"operator_kind": "power_wrap", "power_kind": "sqrt"},
    )
    x = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], dtype=torch.float64)

    plan = direct_mod.build_direct_power_search_plan(
        spec,
        nvars=2,
        max_depth=6,
        x_fit=x,
        y_fit=torch.sqrt(x[:, 0] * x[:, 1]).unsqueeze(-1),
        x_probe=x,
        y_probe=torch.sqrt(x[:, 0] * x[:, 1]).unsqueeze(-1),
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert isinstance(plan, PreparedCandidatesSearchPlan)
    assert str(plan.meta.get("execution_mode", "")) == "exact_bound"


def test_direct_quadratic_builder_returns_exact_bound_plan_when_bases_are_bound():
    base_nodes = (("var", 1), ("var", 2), ("var", 3))
    spec = OperatorApplication(
        family="quadratic",
        operator_id="quadratic:sqrt",
        scaffold_id="quadratic:sqrt:norm",
        parent_node=("sqrt", ("const", 1.0)),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_quadratic_closure(
            scaffold_id="quadratic:sqrt:norm",
            quadratic_kind="sqrt",
            base_nodes=base_nodes,
        ),
        metadata={"operator_kind": "quadratic_wrap", "quadratic_kind": "sqrt"},
    )
    x = torch.tensor(
        [
            [0.4, 0.2, 0.3, 0.5],
            [0.5, 0.4, 0.6, 0.7],
            [0.8, 0.5, 0.7, 0.9],
            [1.0, 0.6, 0.8, 1.1],
        ],
        dtype=torch.float64,
    )

    plan = direct_mod.build_direct_quadratic_search_plan(
        spec,
        nvars=4,
        max_depth=8,
        x_fit=x,
        y_fit=torch.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2).unsqueeze(-1),
        x_probe=x,
        y_probe=torch.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2).unsqueeze(-1),
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={},
    )

    assert isinstance(plan, PreparedCandidatesSearchPlan)
    assert str(plan.meta.get("execution_mode", "")) == "exact_bound"


def test_outer_scaffold_rational_affine_uses_direct_fill_without_inverse(monkeypatch):
    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="rational",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"form": "rational_affine"},
    )

    def _fake_enumerate_outer_scaffold_specs(**kwargs):
        return [scaffold_spec]

    def _raise_inverse_path(*args, **kwargs):
        raise AssertionError("inverse scaffold path should not run for direct rational scaffolds")

    monkeypatch.setattr(scaffold_mod, "LEGACY_SCAFFOLD_ADAPTER_ENABLED", True)
    monkeypatch.setattr(scaffold_mod, "enumerate_outer_scaffold_specs", _fake_enumerate_outer_scaffold_specs)
    monkeypatch.setattr(scaffold_mod, "_fit_scaffold_mapping", _raise_inverse_path)
    monkeypatch.setattr(scaffold_mod, "_build_scaffold_beam_state", _raise_inverse_path)
    monkeypatch.setattr(scaffold_mod, "solve_inverse_spec_preview_rows", _raise_inverse_path)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: ((1.0 + x[:, :1]) / (1.0 + (x[:, :1] * x[:, 1:2]))),
        nvars=2,
        n_iter=0,
        max_depth=5,
        poly_degree=1,
        lo=0.3,
        hi=1.9,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        no_residual=True,
        no_crossover=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
        inverse_spec_enable=True,
        closure_search_enable=True,
        closure_search_families=["rational"],
        closure_search_max_proposals=1,
        closure_search_anchors_per_family=1,
        closure_search_preview_topk=4,
        closure_search_exact_topk=1,
        score_mapping_family_mode="cheap",
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    stats = getattr(arch, "closure_search_stats", {})
    best = arch.best(1)[0]

    assert int(stats.get("preview_calls", 0)) >= 1
    assert int(stats.get("direct_calls", 0)) >= 1
    assert int(stats.get("preview_candidates", 0)) >= 1
    assert int(stats.get("scored", 0)) >= 1
    assert int(dict(stats.get("status_counts", {}) or {}).get("direct_ok", 0)) >= 1
    assert float(best.best_mse) < 1.0e-20
    assert float(best.best_raw_mse) < 1.0e-8
