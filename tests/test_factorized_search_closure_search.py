# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch
import time

import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod
import nestynet_sr.sr_search.factorized_search.engine.search as search_mod
import nestynet_sr.sr_search.factorized_search.closure_search_compat as scaffold_mod
import nestynet_sr.sr_search.factorized_search.proposal_families.direct as direct_mod
import nestynet_sr.sr_search.factorized_search.proposal_families.periodic_search as periodic_search_mod
import nestynet_sr.sr_search.factorized_search.proposal_families.runner as runner_mod
from nestynet_sr.sr_search.factorized_search.basis_state import (
    BasisState,
    ProposalContext,
    basis_state_from_closure_candidate,
)
from nestynet_sr.sr_search.factorized_search.basis_scoring import (
    direct_power_depth_slack_from_coeffs,
    fit_direct_power_design,
    materialize_direct_power_expr,
    materialize_multi_term_rational_expr,
)
from nestynet_sr.sr_search.factorized_search.expr_ast import build_pool, node_depth, node_dims, node_str
from nestynet_sr.sr_search.factorized_search.closures import (
    bound_closure_from_closure_candidate,
    make_direct_linear_wrap_closure,
    make_direct_periodic_closure,
    make_direct_power_closure,
    make_direct_quadratic_closure,
    make_direct_rational_closure,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.compat import OuterScaffoldSpec, operator_application_from_scaffold
from nestynet_sr.sr_search.factorized_search.proposal_families.closure_runners import (
    PreparedCandidatesSearchPlan,
    SeedSubsetSearchPlan,
    execute_direct_search_plan,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.closure_builders import (
    build_affine_power_candidate,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.closure_eval import (
    score_direct_closure_candidate,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.binding_search import pin_small_trig_carrier
from nestynet_sr.sr_search.factorized_search.proposal_families.scaffold_enum import enumerate_operator_applications
from nestynet_sr.sr_search.factorized_search.proposal_families.runner import (
    run_closure_search_pass_impl as run_closure_search_pass_impl_native,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.types import OperatorApplication


def test_closure_search_pass_periodic_inserts_completed_candidate(monkeypatch):
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

    def _fake_enumerate_closure_search_specs(**kwargs):
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


def test_closure_search_pass_records_direct_not_supported_diagnostics(monkeypatch):
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


def test_enumerate_closure_search_specs_adds_periodic_add_forms():
    specs = scaffold_mod.enumerate_closure_search_specs(
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


def test_enumerate_closure_search_specs_adds_rational_affine_form():
    specs = scaffold_mod.enumerate_closure_search_specs(
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


def test_enumerate_closure_search_specs_adds_quadratic_sqrt_forms():
    specs = scaffold_mod.enumerate_closure_search_specs(
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


def test_enumerate_closure_search_specs_adds_power_inverse_forms():
    specs = scaffold_mod.enumerate_closure_search_specs(
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


def test_enumerate_operator_applications_power_budget_surfaces_square_forms():
    nvars = 4
    var_dims = [(0.0,), (0.0,), (0.0,), (0.0,)]
    y_dims = (0.0,)
    pool_nodes = build_pool(nvars)
    pool_dims = []
    for node in pool_nodes:
        try:
            pool_dims.append(node_dims(node, var_dims))
        except Exception:
            pool_dims.append(None)

    apps = enumerate_operator_applications(
        families=["power"],
        nvars=nvars,
        y_dims=y_dims,
        var_dims=var_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        anchors_per_family=8,
        max_scaffolds=8,
    )

    operator_ids = [str(app.operator_id) for app in apps]

    assert "power:invsqrt_mul" in operator_ids
    assert "power:sqrt_mul" in operator_ids
    assert "power:sqr_mul" in operator_ids


def test_enumerate_operator_applications_periodic_budget_surfaces_prefactor_phase_products():
    nvars = 4
    var_dims = [(0.0,), (0.0,), (0.0,), (0.0,)]
    y_dims = (0.0,)
    pool_nodes = build_pool(nvars)
    pool_dims = []
    for node in pool_nodes:
        try:
            pool_dims.append(node_dims(node, var_dims))
        except Exception:
            pool_dims.append(None)

    apps = enumerate_operator_applications(
        families=["periodic"],
        nvars=nvars,
        y_dims=y_dims,
        var_dims=var_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        anchors_per_family=8,
        max_scaffolds=8,
    )

    periodic_parents = {str(app.scaffold_id): app.parent_node for app in apps}

    assert "periodic:cos_mul:x0:(x1*x2)" in periodic_parents
    assert periodic_parents["periodic:cos_mul:x0:(x1*x2)"] == (
        "mul",
        ("cos", ("mul", ("var", 1), ("var", 2))),
        ("var", 0),
    )


def test_enumerate_closure_search_specs_keeps_var_anchor_for_exp_family():
    specs = scaffold_mod.enumerate_closure_search_specs(
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


def test_enumerate_closure_search_specs_keeps_var_anchor_for_log_family():
    specs = scaffold_mod.enumerate_closure_search_specs(
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

    monkeypatch.setattr(scaffold_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

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

    monkeypatch.setattr(scaffold_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

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
    assert scaffold_mod.node_str(rows[0]["expr"]) in {
        "(log((x0*x1))+x2)",
        "(x2+log((x0*x1)))",
    }
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

    monkeypatch.setattr(scaffold_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

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
    expr_text = scaffold_mod.node_str(rows[0]["expr"])
    assert "log(" in expr_text
    assert "x2" in expr_text
    assert float(rows[0]["local_probe_mse"]) < 1.0e-12


def test_closure_search_pass_uses_route_specific_beam_limits(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_closure_search_pass(**kwargs):
        captured["beam_cfg"] = dict(kwargs.get("beam_cfg", {}) or {})
        captured["deadline_s"] = kwargs.get("deadline_s", None)
        return {"candidate_rows": [], "stats": {}}

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_closure_search_pass)

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


def test_closure_search_pass_allocates_exact_budget_per_scaffold(monkeypatch):
    expr_a1 = ("sin", ("var", 0))
    expr_a2 = ("cos", ("var", 0))
    expr_b1 = ("sqr", ("var", 0))
    captured_scored: list[tuple] = []

    def _fake_run_closure_search_pass(**kwargs):
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

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_closure_search_pass)

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


def test_closure_search_exact_scoring_can_use_anchor_head(monkeypatch):
    expr = ("add", ("cos", ("var", 0)), ("var", 0))

    def _fake_run_closure_search_pass(**kwargs):
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

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_closure_search_pass)

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


def test_closure_search_periodic_add_uses_direct_fill_without_inverse(monkeypatch):
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


def test_closure_search_exp_add_uses_direct_fill_without_inverse(monkeypatch):
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


def test_closure_search_log_add_uses_direct_fill_without_inverse(monkeypatch):
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


def test_closure_search_quadratic_sqrt_mul_uses_direct_fill_without_inverse(monkeypatch):
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


def test_closure_search_power_invsqrt_mul_uses_direct_fill_without_inverse(monkeypatch):
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

    monkeypatch.setattr(scaffold_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

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
    assert int(meta.get("anchor_lift_attempts", 0)) == 1
    assert int(meta.get("anchor_lift_applied", 0)) == 1
    assert len(rows) == 1
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

    monkeypatch.setattr(scaffold_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

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


def test_closure_search_anchor_head_compare_logs_delta(monkeypatch):
    anchor = ("mul", ("var", 0), ("var", 1))
    expr = ("add", ("cos", ("var", 0)), anchor)

    def _fake_run_closure_search_pass(**kwargs):
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

    monkeypatch.setattr(search_mod, "_run_closure_search_pass_impl", _fake_run_closure_search_pass)

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


def test_run_closure_search_pass_skips_enumeration_after_deadline(monkeypatch):
    def _should_not_run(**kwargs):
        raise AssertionError("enumeration should be skipped once the scaffold deadline is exceeded")

    monkeypatch.setattr(scaffold_mod, "enumerate_closure_search_specs", _should_not_run)

    ret = scaffold_mod.run_closure_search_pass(
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


def test_run_closure_search_pass_impl_does_not_silently_fallback_to_legacy_enum():
    def _operator_enum(**_kwargs):
        return []

    def _legacy_enum(**_kwargs):
        raise AssertionError("legacy scaffold enumeration should not run without explicit opt-in")

    ret = run_closure_search_pass_impl_native(
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


def test_run_closure_search_pass_impl_uses_native_enum():
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

    ret = run_closure_search_pass_impl_native(
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


def test_run_closure_search_pass_impl_preserves_rebound_row_closure_for_power_head_bundle():
    stale_carrier = ("add", ("sqr", ("mul", ("var", 1), ("var", 2))), ("add", ("sqr", ("var", 3)), ("var", 3)))
    stale_anchor = ("var", 0)
    rebound_anchor = ("mul", ("var", 0), ("var", 3))
    rebound_carrier = ("cos", ("mul", ("var", 1), ("var", 2)))
    rebound_expr = ("mul", rebound_anchor, ("sqr", rebound_carrier))
    spec = OperatorApplication(
        family="power",
        operator_id="power:sqr_mul",
        scaffold_id="power:sqr_mul:seed",
        parent_node=("mul", stale_anchor, ("sqr", stale_carrier)),
        hole_path=(2, 1),
        target_mode="robust",
        anchor_node=stale_anchor,
        bound_closure=make_direct_power_closure(
            scaffold_id="power:sqr_mul:seed",
            power_kind="sqr_mul",
            exponent=2.0,
            hole_node=stale_carrier,
            anchor_node=stale_anchor,
        ),
        metadata={"form": "sqr_mul", "power_kind": "sqr_mul"},
    )
    rebound_closure = make_direct_power_closure(
        scaffold_id="power:sqr_mul:seed",
        power_kind="sqr_mul",
        exponent=2.0,
        hole_node=rebound_carrier,
        anchor_node=rebound_anchor,
    )

    def _fake_enumerate_operator_applications(**_kwargs):
        return [spec]

    def _fake_direct_power(_spec, **_kwargs):
        return (
            [
                {
                    "expr": rebound_expr,
                    "child_key": str(rebound_expr),
                    "proposal_key": str(rebound_expr),
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 8,
                    "local_mapping_kind": "direct_power_head",
                    "local_mapping_coeffs": [0.0, 0.0, 1.0],
                    "direct_metadata": {
                        "anchor_node": rebound_anchor,
                        "hole_node": rebound_carrier,
                        "power_inner_node": rebound_carrier,
                        "power_exponent": 2.0,
                        "power_variant": "square_only",
                    },
                    "bound_closure_obj": rebound_closure,
                    "bound_closure_dict": rebound_closure.to_dict(),
                }
            ],
            "direct_ok",
            {},
        )

    ret = run_closure_search_pass_impl_native(
        families=["power"],
        nvars=4,
        max_scaffolds=4,
        anchors_per_family=1,
        max_depth=5,
        poly_degree=2,
        x_fit=torch.zeros((4, 4), dtype=torch.float64),
        y_fit=torch.zeros((4, 1), dtype=torch.float64),
        x_probe=torch.zeros((4, 4), dtype=torch.float64),
        y_probe=torch.zeros((4, 1), dtype=torch.float64),
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_phi_fit=torch.zeros((4, 0), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 0), dtype=torch.float64),
        pool_dims=[],
        safe_eps=1.0e-8,
        preview_topk=4,
        beam_cfg={},
        solver_kwargs={},
        enumerate_operator_applications_fn=_fake_enumerate_operator_applications,
        solve_direct_operator_preview_rows_fn=_fake_direct_power,
    )

    rows = list(ret.get("candidate_rows", []) or [])
    assert rows
    row = rows[0]
    assert row["bound_closure_obj"].bindings["carrier"] == rebound_carrier
    assert row["bound_closure_obj"].bindings["anchor"] == rebound_anchor
    latent_bundle_exprs = list(row["feature_block_dict"].get("latent_bundle_exprs", []) or [])
    assert "(x0*x3)" in latent_bundle_exprs
    head_bundle_exprs = list(row["feature_block_obj"].metadata.get("head_bundle_exprs", []) or [])
    assert "((x0*x3)*sqr(cos((x1*x2))))" in head_bundle_exprs
    assert all("sqr((x1*x2))" not in expr for expr in head_bundle_exprs)


def test_run_closure_search_pass_impl_core_lane_uses_canonical_pool():
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

    ret = run_closure_search_pass_impl_native(
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


def test_run_closure_search_pass_impl_seed_mode_uses_core_lane_only():
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
        return [
            OperatorApplication(
                family="power",
                operator_id="power:sqr_mul",
                scaffold_id=f"{str(basis_seed_mode)}:power",
                parent_node=("mul", ("var", 0), ("sqr", ("var", 1))),
                hole_path=(2, 1),
            )
        ]

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

    ret = run_closure_search_pass_impl_native(
        families=["periodic", "power"],
        nvars=2,
        max_scaffolds=6,
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
            basis_state=None,
            basis_state_beam=(),
            diagnostics={"route": "seed-round"},
            family_hints={},
        ),
        enumerate_operator_applications_fn=_native_enum,
        solve_direct_operator_preview_rows_fn=_score_direct,
    )

    stats = dict(ret.get("stats", {}) or {})
    rows = list(ret.get("candidate_rows", []) or [])
    assert len(lane_calls) == 2
    assert {call["basis_seed_mode"] for call in lane_calls} == {"core_only"}
    assert stats.get("proposal_lane_budgets", {}).get("core") == 6
    assert stats.get("proposal_lane_budgets", {}).get("basis_augmented") == 0
    assert {str(row.get("proposal_lane", "")) for row in rows} == {"core"}


def test_run_closure_search_pass_impl_applies_family_budget_plan_per_family():
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

    ret = run_closure_search_pass_impl_native(
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


def test_run_closure_search_pass_impl_keeps_same_expr_from_distinct_operator_families():
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

    ret = run_closure_search_pass_impl_native(
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

    monkeypatch.setattr(scaffold_mod, "_collect_direct_hole_candidates", _fake_collect_direct_hole_candidates)

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
    assert scaffold_mod.node_str(rows[0]["expr"]) in {
        "((1+x0)/(1+(x0*x1)))",
        "((1+x0)/((x0*x1)+1))",
    }
    assert float(rows[0]["local_probe_mse"]) < 1.0e-12


def test_direct_rational_plan_uses_normalized_gate_for_multi_term(monkeypatch):
    spec = OuterScaffoldSpec(
        family="rational",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"form": "rational_affine"},
    )

    single_rows = [
        {
            "local_probe_mse": 1.0e-10,
            "local_fit_mse": 1.0e-10,
            "direct_metadata": {},
        }
    ]
    fallback_calls: list[dict[str, float]] = []

    def _fake_single_term_plan(*_args, **_kwargs):
        return direct_mod.CustomDirectSearchPlan(
            run_fn=lambda **__kwargs: (list(single_rows), "direct_ok", {}),
            kwargs={},
        )

    def _fake_multi_term_fallback(*_args, **kwargs):
        fallback_calls.append(
            {
                "single_term_best_mse": float(kwargs.get("single_term_best_mse", float("inf"))),
            }
        )
        return [], "multi_term_rational_no_improvement", {}

    monkeypatch.setattr(direct_mod, "build_direct_rational_affine_search_plan", _fake_single_term_plan)
    monkeypatch.setattr(direct_mod, "_run_multi_term_rational_fallback", _fake_multi_term_fallback)

    y_probe = torch.full((8, 1), 1.0e-12, dtype=torch.float64)
    plan = direct_mod._build_direct_rational_plan(
        spec,
        nvars=2,
        max_depth=4,
        x_fit=torch.ones((8, 2), dtype=torch.float64),
        y_fit=y_probe.clone(),
        x_probe=torch.ones((8, 2), dtype=torch.float64),
        y_probe=y_probe,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"multi_term_rational_threshold": 1.0e-8},
        deadline_s=None,
    )

    rows, status, meta = plan.run_fn()

    assert status == "direct_ok"
    assert rows == single_rows
    assert len(fallback_calls) == 1
    assert float(meta.get("multi_term_rational_single_best_mse", 0.0)) == 1.0e-10
    assert float(meta.get("multi_term_rational_single_best_norm", 0.0)) > 1.0e-8
    assert str(meta.get("multi_term_rational_gate_metric", "")) == "normalized_probe_mse"


def test_multi_term_rational_fallback_seeds_from_single_term_metadata():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    x = torch.tensor(
        [
            [0.2, 0.3, 0.1],
            [0.4, 0.5, 0.2],
            [0.6, 0.7, 0.3],
            [0.8, 0.9, 0.4],
            [1.0, 1.1, 0.5],
            [1.2, 1.3, 0.6],
        ],
        dtype=torch.float64,
    )
    y = ((x[:, 0] + x[:, 1]) / (1.0 + x[:, 2])).unsqueeze(-1)

    single_term_best_rows = [
        {"direct_metadata": {"u_node": ("var", 0), "v_node": ("var", 2)}},
        {"direct_metadata": {"u_node": ("var", 1), "v_node": ("var", 2)}},
    ]

    def _empty_candidates(**_kwargs):
        return [], {"candidate_source_counts": {}}

    rows, status, meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=1.0,
        single_term_best_rows=single_term_best_rows,
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
        solver_kwargs={"multi_term_rational_role_shadow_enable": False},
        collect_direct_hole_candidates_fn=_empty_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-6
    assert tuple(rows[0]["direct_metadata"]["u_nodes"]) == (("var", 0), ("var", 1))
    assert tuple(rows[0]["direct_metadata"]["v_nodes"]) == (("var", 2),)
    assert all(isinstance(node, tuple) for node in rows[0]["direct_metadata"]["u_nodes"])
    assert all(isinstance(node, tuple) for node in rows[0]["direct_metadata"]["v_nodes"])
    rebuilt = bound_closure_from_closure_candidate(
        family="rational",
        scaffold_id="rational:affine",
        expr=rows[0]["expr"],
        anchor_node=None,
        scaffold_metadata={"form": "multi_term_rational"},
        direct_metadata=rows[0]["direct_metadata"],
    )
    assert rebuilt.bindings["numerator_terms"] == (("var", 0), ("var", 1))
    assert rebuilt.bindings["denominator_terms"] == (("var", 2),)
    assert int(meta.get("candidate_count_scored", 0)) >= 1


def test_multi_term_rational_fallback_keeps_raw_vars_and_structural_denominator_seeds():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    x = torch.tensor(
        [
            [1.2, 1.3, 1.1],
            [1.6, 1.9, 1.2],
            [2.0, 2.5, 1.4],
            [2.4, 2.1, 1.7],
            [2.8, 3.0, 1.9],
            [3.2, 2.7, 2.2],
            [3.6, 3.4, 2.5],
            [4.0, 3.8, 2.9],
        ],
        dtype=torch.float64,
    )
    y = ((x[:, 1] + x[:, 2]) / (1.0 + (x[:, 1] * x[:, 2]) / (x[:, 0] ** 2))).unsqueeze(-1)

    numerator_nodes = [
        ("var", 0),
        ("add", ("var", 0), ("var", 0)),
        ("sub", ("add", ("var", 0), ("var", 1)), ("var", 1)),
        ("sub", ("var", 1), ("sub", ("var", 1), ("var", 0))),
        ("add", ("sub", ("var", 0), ("var", 1)), ("var", 1)),
        ("sub", ("var", 2), ("sub", ("var", 2), ("var", 0))),
        ("mul", ("div", ("var", 0), ("var", 1)), ("var", 1)),
        ("mul", ("div", ("var", 0), ("var", 2)), ("var", 2)),
        ("div", ("var", 1), ("div", ("var", 1), ("var", 0))),
        ("div", ("var", 2), ("div", ("var", 2), ("var", 0))),
        ("sub", ("add", ("var", 0), ("var", 2)), ("var", 2)),
        ("sub", ("add", ("var", 0), ("var", 0)), ("var", 0)),
        ("sub", ("add", ("var", 0), ("var", 1)), ("var", 1)),
        ("sub", ("add", ("var", 0), ("var", 2)), ("var", 2)),
        ("var", 1),
        ("var", 2),
    ]
    denominator_nodes = [
        ("const", 1.0),
        ("const", -1.0),
        ("div", ("var", 0), ("add", ("var", 0), ("var", 0))),
        ("div", ("var", 1), ("add", ("var", 1), ("var", 1))),
        ("div", ("var", 2), ("add", ("var", 2), ("var", 2))),
        ("div", ("add", ("var", 0), ("var", 0)), ("var", 0)),
        ("div", ("add", ("var", 1), ("var", 1)), ("var", 1)),
        ("div", ("add", ("var", 2), ("var", 2)), ("var", 2)),
        ("div", ("var", 0), ("var", 1)),
        ("div", ("var", 0), ("var", 2)),
        ("div", ("var", 1), ("var", 0)),
        ("div", ("var", 1), ("var", 2)),
        ("div", ("var", 2), ("var", 0)),
        ("div", ("var", 2), ("var", 1)),
        ("cos", ("div", ("var", 0), ("var", 1))),
        ("sqrt", ("div", ("var", 0), ("var", 1))),
        ("div", ("mul", ("var", 1), ("var", 2)), ("sqr", ("var", 0))),
    ]

    def _collector(**kwargs):
        target_dim = tuple(kwargs.get("target_dim", ()) or ())
        rows = numerator_nodes if target_dim == (1.0,) else denominator_nodes
        return [("enum", node) for node in rows], {"candidate_source_counts": {"enum": len(rows)}}

    rows, status, meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=0.2,
        single_term_best_rows=[],
        nvars=3,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(1.0,), (1.0,), (1.0,)],
        y_dims=(1.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={"multi_term_rational_role_shadow_enable": False},
        collect_direct_hole_candidates_fn=_collector,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert set(tuple(node) for node in rows[0]["direct_metadata"]["u_nodes"]) == {
        ("var", 1),
        ("var", 2),
    }
    assert ("div", ("mul", ("var", 1), ("var", 2)), ("sqr", ("var", 0))) in tuple(
        rows[0]["direct_metadata"]["v_nodes"]
    )
    assert int(meta.get("u_screen_count", 0)) >= 2
    assert int(meta.get("v_screen_count", 0)) >= 1


def test_multi_term_rational_role_shadows_promote_product_prefactor_and_unary_denominator():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    generator = torch.Generator().manual_seed(29)
    x_all = 1.0 + 4.0 * torch.rand((240, 3), generator=generator, dtype=torch.float64)
    y_all = (
        x_all[:, 0]
        * torch.sin(0.5 * x_all[:, 1] * x_all[:, 2]) ** 2
        / (torch.sin(0.5 * x_all[:, 1]) ** 2)
    ).unsqueeze(-1)
    x_fit, x_probe = x_all[:160], x_all[160:]
    y_fit, y_probe = y_all[:160], y_all[160:]

    cos_x1 = ("cos", ("var", 1))
    cos_x1x2 = ("cos", ("mul", ("var", 1), ("var", 2)))
    near_miss = (
        "sub",
        ("mul", cos_x1x2, ("var", 0)),
        ("mul", cos_x1, ("var", 0)),
    )
    target_dim = (0.0, -3.0, 1.0, 0.0, 0.0)
    dimless = (0.0, 0.0, 0.0, 0.0, 0.0)

    def _collector(**kwargs):
        requested_dim = tuple(kwargs.get("target_dim", ()) or ())
        if requested_dim == target_dim:
            return [("near_miss", near_miss)], {"candidate_source_counts": {"near_miss": 1}}
        return [], {"candidate_source_counts": {}}

    rows, status, meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=1.0e9,
        single_term_best_rows=[],
        nvars=3,
        max_depth=5,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=[target_dim, dimless, dimless],
        y_dims=target_dim,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={
            "multi_term_rational_budget": 4,
            "multi_term_rational_max_u": 4,
            "multi_term_rational_max_v": 4,
            "multi_term_rational_role_shadow_budget": 4,
            "multi_term_rational_role_shadow_max_supports": 8,
            "safe_eps": 1.0e-8,
        },
        collect_direct_hole_candidates_fn=_collector,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert int(meta.get("role_shadow_product_den_count", 0)) >= 2
    assert int(meta.get("role_shadow_low_cost_v_count", 0)) == 0
    assert int(meta.get("role_shadow_fit_improvement_count", 0)) >= 1
    assert int(meta.get("role_shadow_hit_count", 0)) >= 1
    direct_meta = rows[0]["direct_metadata"]
    assert direct_meta["support_source"] == "role_shadow"
    assert direct_meta["role_shadow_support"] is True
    assert ("var", 0) in tuple(direct_meta["u_nodes"])
    assert any("cos((x1*x2))" in str(expr) and "x0" in str(expr) for expr in direct_meta["u_node_exprs"])
    assert tuple(direct_meta["v_nodes"]) == (cos_x1,)


def test_multi_term_rational_auto_role_shadow_budget_reserves_generic_lane():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    generator = torch.Generator().manual_seed(29)
    x_all = 1.0 + 4.0 * torch.rand((180, 3), generator=generator, dtype=torch.float64)
    y_all = (
        x_all[:, 0]
        * torch.sin(0.5 * x_all[:, 1] * x_all[:, 2]) ** 2
        / (torch.sin(0.5 * x_all[:, 1]) ** 2)
    ).unsqueeze(-1)
    x_fit, x_probe = x_all[:120], x_all[120:]
    y_fit, y_probe = y_all[:120], y_all[120:]

    cos_x1 = ("cos", ("var", 1))
    cos_x1x2 = ("cos", ("mul", ("var", 1), ("var", 2)))
    near_miss = (
        "sub",
        ("mul", cos_x1x2, ("var", 0)),
        ("mul", cos_x1, ("var", 0)),
    )
    target_dim = (0.0, -3.0, 1.0, 0.0, 0.0)
    dimless = (0.0, 0.0, 0.0, 0.0, 0.0)

    def _collector(**kwargs):
        requested_dim = tuple(kwargs.get("target_dim", ()) or ())
        if requested_dim == target_dim:
            return [("near_miss", near_miss)], {"candidate_source_counts": {"near_miss": 1}}
        return [], {"candidate_source_counts": {}}

    rows, status, meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=1.0e9,
        single_term_best_rows=[],
        nvars=3,
        max_depth=5,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=[target_dim, dimless, dimless],
        y_dims=target_dim,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={
            "multi_term_rational_budget": 8,
            "multi_term_rational_max_u": 4,
            "multi_term_rational_max_v": 4,
            "multi_term_rational_role_shadow_max_supports": 8,
            "safe_eps": 1.0e-8,
        },
        collect_direct_hole_candidates_fn=_collector,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert meta.get("role_shadow_budget_explicit") is False
    assert int(meta.get("role_shadow_generic_budget_reserve", 0)) > 0
    assert int(meta.get("role_shadow_budget_cap", 0)) <= 4
    assert int(meta.get("role_shadow_budget_used", 0)) <= int(meta.get("role_shadow_budget_cap", 0))


def test_multi_term_rational_materializer_allows_fitted_scale_depth_slack():
    x0 = ("var", 0)
    cos_x1 = ("cos", ("var", 1))
    cos_x1x2 = ("cos", ("mul", ("var", 1), ("var", 2)))

    expr = materialize_multi_term_rational_expr(
        u_nodes=[x0, ("mul", cos_x1x2, x0)],
        v_nodes=[cos_x1],
        coeffs=[0.0, 1.0, -1.0, -1.0],
        max_depth=5,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    assert expr is not None
    assert int(node_depth(expr)) == 7


def test_multi_term_rational_role_shadows_synthesize_prefactor_carrier_products():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    generator = torch.Generator().manual_seed(29)
    x_all = 1.0 + 4.0 * torch.rand((240, 3), generator=generator, dtype=torch.float64)
    y_all = (
        x_all[:, 0]
        * torch.sin(0.5 * x_all[:, 1] * x_all[:, 2]) ** 2
        / (torch.sin(0.5 * x_all[:, 1]) ** 2)
    ).unsqueeze(-1)
    x_fit, x_probe = x_all[:160], x_all[160:]
    y_fit, y_probe = y_all[:160], y_all[160:]

    x0 = ("var", 0)
    cos_x1 = ("cos", ("var", 1))
    cos_x1x2 = ("cos", ("mul", ("var", 1), ("var", 2)))
    dimless = (0.0,)

    def _collector(**_kwargs):
        return [
            ("prefactor", x0),
            ("carrier_den", cos_x1),
            ("carrier_num", cos_x1x2),
        ], {"candidate_source_counts": {"unit": 3}}

    rows, status, meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=1.0e9,
        single_term_best_rows=[],
        nvars=3,
        max_depth=5,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=[dimless, dimless, dimless],
        y_dims=dimless,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={
            "multi_term_rational_budget": 64,
            "multi_term_rational_max_u": 4,
            "multi_term_rational_max_v": 4,
            "multi_term_rational_role_shadow_budget": 64,
            "multi_term_rational_role_shadow_max_supports": 64,
            "multi_term_rational_role_shadow_max_den": 8,
            "multi_term_rational_role_shadow_max_low_cost_den": 8,
            "multi_term_rational_role_shadow_max_synthetic_prefactors": 2,
            "multi_term_rational_role_shadow_max_synthetic_carriers": 1,
            "multi_term_rational_role_shadow_max_synthetic_num_blocks": 8,
            "safe_eps": 1.0e-8,
        },
        collect_direct_hole_candidates_fn=_collector,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert int(meta.get("role_shadow_synthetic_num_block_count", 0)) >= 1
    direct_meta = rows[0]["direct_metadata"]
    assert direct_meta["support_source"] == "role_shadow"
    assert tuple(direct_meta["v_nodes"]) == (cos_x1,)
    assert any("cos((x1*x2))" in str(expr) and "x0" in str(expr) for expr in direct_meta["u_node_exprs"])


def test_multi_term_rational_role_shadows_synthesize_nontrig_transfer_product():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    generator = torch.Generator().manual_seed(113)
    x_all = 0.2 + 2.8 * torch.rand((180, 3), generator=generator, dtype=torch.float64)
    y_all = (x_all[:, 0] * (1.0 - torch.exp(-x_all[:, 1])) / (1.0 + x_all[:, 2])).unsqueeze(-1)
    x_fit, x_probe = x_all[:120], x_all[120:]
    y_fit, y_probe = y_all[:120], y_all[120:]

    x0 = ("var", 0)
    x2 = ("var", 2)
    exp_neg_x1 = ("exp", ("neg", ("var", 1)))
    dimless = (0.0,)

    def _collector(**_kwargs):
        return [
            ("prefactor", x0),
            ("carrier_num", exp_neg_x1),
            ("carrier_den", x2),
        ], {"candidate_source_counts": {"unit": 3}}

    rows, status, meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=1.0e9,
        single_term_best_rows=[],
        nvars=3,
        max_depth=5,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=[dimless, dimless, dimless],
        y_dims=dimless,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={
            "multi_term_rational_budget": 64,
            "multi_term_rational_max_u": 4,
            "multi_term_rational_max_v": 4,
            "multi_term_rational_role_shadow_budget": 64,
            "multi_term_rational_role_shadow_max_supports": 64,
            "multi_term_rational_role_shadow_max_den": 4,
            "multi_term_rational_role_shadow_max_low_cost_den": 4,
            "multi_term_rational_role_shadow_max_synthetic_prefactors": 1,
            "multi_term_rational_role_shadow_max_synthetic_carriers": 2,
            "multi_term_rational_role_shadow_max_synthetic_num_blocks": 4,
            "safe_eps": 1.0e-8,
        },
        collect_direct_hole_candidates_fn=_collector,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert int(meta.get("role_shadow_synthetic_num_block_count", 0)) >= 1
    direct_meta = rows[0]["direct_metadata"]
    assert direct_meta["support_source"] == "role_shadow"
    assert tuple(direct_meta["v_nodes"]) == (x2,)
    assert any("exp((-x1))" in str(expr) and "x0" in str(expr) for expr in direct_meta["u_node_exprs"])


def test_multi_term_rational_role_shadows_allow_singleton_affine_numerator_kernel():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    generator = torch.Generator().manual_seed(131)
    x_all = 1.0 + 4.0 * torch.rand((220, 2), generator=generator, dtype=torch.float64)
    y_all = ((1.0 - torch.cos(x_all[:, 0] * x_all[:, 1])) / (1.0 - torch.cos(x_all[:, 0]))).unsqueeze(-1)
    x_fit, x_probe = x_all[:150], x_all[150:]
    y_fit, y_probe = y_all[:150], y_all[150:]

    cos_x0 = ("cos", ("var", 0))
    cos_x0x1 = ("cos", ("mul", ("var", 0), ("var", 1)))
    dimless = (0.0,)

    def _collector(**_kwargs):
        return [
            ("carrier_den", cos_x0),
            ("carrier_num", cos_x0x1),
        ], {"candidate_source_counts": {"unit": 2}}

    rows, status, meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=1.0e9,
        single_term_best_rows=[],
        nvars=2,
        max_depth=5,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=[dimless, dimless],
        y_dims=dimless,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={
            "multi_term_rational_budget": 32,
            "multi_term_rational_max_u": 4,
            "multi_term_rational_max_v": 4,
            "multi_term_rational_role_shadow_budget": 32,
            "multi_term_rational_role_shadow_max_supports": 32,
            "multi_term_rational_role_shadow_max_den": 4,
            "multi_term_rational_role_shadow_max_low_cost_den": 4,
            "multi_term_rational_role_shadow_max_synthetic_prefactors": 0,
            "multi_term_rational_role_shadow_max_synthetic_carriers": 4,
            "multi_term_rational_role_shadow_max_synthetic_num_blocks": 4,
            "safe_eps": 1.0e-8,
        },
        collect_direct_hole_candidates_fn=_collector,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert int(meta.get("role_shadow_synthetic_num_block_count", 0)) >= 1
    assert int(meta.get("role_shadow_hit_count", 0)) >= 1
    matched = [
        row for row in rows
        if row["direct_metadata"].get("support_source") == "role_shadow"
        and tuple(row["direct_metadata"].get("u_nodes", ())) == (cos_x0x1,)
        and tuple(row["direct_metadata"].get("v_nodes", ())) == (cos_x0,)
    ]
    assert matched
    assert float(matched[0]["local_probe_mse"]) < 1.0e-20


def test_multi_term_rational_fallback_includes_denominator_only_support():
    spec = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    x = torch.tensor(
        [
            [0.2],
            [0.4],
            [0.7],
            [1.1],
            [1.7],
            [2.3],
        ],
        dtype=torch.float64,
    )
    y = (1.0 / (1.0 + x[:, 0])).unsqueeze(-1)
    x0 = ("var", 0)

    def _collector(**_kwargs):
        return [("den", x0)], {"candidate_source_counts": {"unit": 1}}

    rows, status, _meta = direct_mod._run_multi_term_rational_fallback(
        spec,
        single_term_best_mse=1.0e9,
        single_term_best_rows=[],
        nvars=1,
        max_depth=4,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,)],
        y_dims=(0.0,),
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        solver_kwargs={
            "multi_term_rational_role_shadow_enable": False,
            "multi_term_rational_budget": 8,
            "multi_term_rational_max_u": 4,
            "multi_term_rational_max_v": 4,
            "safe_eps": 1.0e-8,
        },
        collect_direct_hole_candidates_fn=_collector,
    )

    assert status == "direct_ok"
    assert rows
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20
    assert any(
        row["direct_metadata"].get("support_size") == (0, 1)
        and float(row["local_probe_mse"]) < 1.0e-20
        for row in rows
    )


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
    assert str(meta.get("execution_mode", "")) in {"slot_search", "slot_search+exact_bound"}
    assert rows[0]["direct_metadata"]["hole_node"] == rebound_carrier
    assert float(rows[0]["local_probe_mse"]) < 1.0e-6


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
            [0.15, 0.31, 0.47],
            [0.28, 0.44, 0.63],
            [0.39, 0.58, 0.79],
            [0.51, 0.72, 0.94],
            [0.66, 0.83, 1.08],
            [0.79, 0.97, 1.21],
            [0.91, 1.12, 1.37],
            [1.03, 1.24, 1.49],
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


def test_direct_operator_exact_bound_periodic_mul_uses_anchor_as_envelope_only():
    carrier = ("mul", ("var", 1), ("var", 2))
    spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_mul",
        scaffold_id="periodic:cos_mul:x0:(x1*x2)",
        parent_node=("mul", ("cos", carrier), ("var", 0)),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 0),
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos_mul:x0:(x1*x2)",
            periodic_kind="cos",
            hole_node=carrier,
            feature_node=("mul", ("cos", carrier), ("var", 0)),
            anchor_node=("var", 0),
        ),
        metadata={"operator_kind": "harmonic_wrap", "periodic_kind": "cos", "form": "cos_mul"},
    )
    x = torch.tensor(
        [
            [0.25, 0.35, 0.45, 0.55],
            [0.42, 0.51, 0.63, 0.74],
            [0.58, 0.69, 0.77, 0.88],
            [0.73, 0.82, 0.91, 1.02],
            [0.89, 0.97, 1.08, 1.16],
            [1.03, 1.12, 1.21, 1.29],
        ],
        dtype=torch.float64,
    )
    y = (x[:, 0] * torch.cos(x[:, 1] * x[:, 2])).unsqueeze(-1)

    plan = periodic_search_mod.build_exact_bound_periodic_search_plan(
        spec,
        max_depth=7,
        x_fit=x,
        x_probe=x,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
    )
    rows, status, meta = execute_direct_search_plan(
        plan,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        max_depth=7,
        y_dims=(0.0,),
        preview_topk=4,
        deadline_s=None,
        collect_direct_hole_candidates_fn=None,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "exact_bound"
    assert rows[0]["expr"] == ("mul", ("cos", carrier), ("var", 0))
    assert rows[0]["direct_metadata"]["envelope_node"] == ("var", 0)
    assert tuple(rows[0]["direct_metadata"]["companion_nodes"]) == ()
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_periodic_auto_rebinding_on_poor_exact_bound():
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
    x = torch.tensor(
        [
            [0.15, 0.31, 0.47],
            [0.28, 0.44, 0.63],
            [0.39, 0.58, 0.79],
            [0.51, 0.72, 0.94],
            [0.66, 0.83, 1.08],
            [0.79, 0.97, 1.21],
            [0.91, 1.12, 1.37],
            [1.03, 1.24, 1.49],
        ],
        dtype=torch.float64,
    )
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
        solver_kwargs={},
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "slot_search+exact_bound"
    assert rows[0]["direct_metadata"]["hole_node"] == rebound_carrier
    assert float(rows[0]["local_probe_mse"]) < 1.0e-6


def test_direct_operator_periodic_slot_search_keeps_bound_carrier_for_envelope_repair():
    bound_carrier = ("mul", ("var", 1), ("var", 2))
    spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_base",
        scaffold_id="periodic:cos_base:bound-carrier",
        parent_node=("cos", bound_carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos_base:bound-carrier",
            periodic_kind="cos",
            hole_node=bound_carrier,
            feature_node=("cos", bound_carrier),
            anchor_node=None,
        ),
        metadata={"operator_kind": "harmonic_wrap", "periodic_kind": "cos", "form": "cos_base"},
    )
    x = torch.tensor(
        [
            [0.25, 0.35, 0.45, 0.55],
            [0.42, 0.51, 0.63, 0.74],
            [0.58, 0.69, 0.77, 0.88],
            [0.73, 0.82, 0.91, 1.02],
            [0.89, 0.97, 1.08, 1.16],
            [1.03, 1.12, 1.21, 1.29],
        ],
        dtype=torch.float64,
    )
    y = (x[:, 0] * torch.cos(x[:, 1] * x[:, 2])).unsqueeze(-1)

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", ("var", 3))], {"candidate_source_counts": {"enum": 1}}

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=4,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[("var", 0)],
        pool_dims=[(0.0,)],
        preview_topk=4,
        solver_kwargs={},
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "slot_search+exact_bound"
    assert rows[0]["direct_metadata"]["hole_node"] == bound_carrier
    assert rows[0]["direct_metadata"]["envelope_node"] == ("var", 0)
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_periodic_slot_search_normalizes_trig_rebindings_to_phase():
    bound_carrier = ("var", 2)
    spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_base",
        scaffold_id="periodic:cos_base:normalize-trig-rebind",
        parent_node=("cos", bound_carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos_base:normalize-trig-rebind",
            periodic_kind="cos",
            hole_node=bound_carrier,
            feature_node=("cos", bound_carrier),
            anchor_node=None,
        ),
        metadata={"operator_kind": "harmonic_wrap", "periodic_kind": "cos", "form": "cos_base"},
    )
    x = torch.tensor(
        [
            [0.15, 0.31, 0.47],
            [0.28, 0.44, 0.63],
            [0.39, 0.58, 0.79],
            [0.51, 0.72, 0.94],
            [0.66, 0.83, 1.08],
            [0.79, 0.97, 1.21],
            [0.91, 1.12, 1.37],
            [1.03, 1.24, 1.49],
        ],
        dtype=torch.float64,
    )
    y = torch.cos(x[:, 2]).unsqueeze(-1)

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [
            ("enum", ("cos", ("var", 2))),
            ("enum", ("sin", ("var", 2))),
        ], {"candidate_source_counts": {"enum": 2}}

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
        preview_topk=8,
        solver_kwargs={},
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) in {"exact_bound", "slot_search+exact_bound"}
    hole_nodes = [dict(row.get("direct_metadata", {})).get("hole_node", None) for row in rows]
    assert all(isinstance(node, tuple) and node for node in hole_nodes)
    assert all(str(node[0]) not in {"sin", "cos"} for node in hole_nodes)
    assert all(node == bound_carrier for node in hole_nodes)
    assert float(rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_operator_periodic_rebinding_seeds_carrier_from_basis_state():
    bound_carrier = ("var", 2)
    rebound_phase = ("mul", ("var", 0), ("var", 1))
    rebound_carrier = ("cos", rebound_phase)
    spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_base",
        scaffold_id="periodic:cos_base:basis-seed",
        parent_node=("cos", bound_carrier),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos_base:basis-seed",
            periodic_kind="cos",
            hole_node=bound_carrier,
            feature_node=("cos", bound_carrier),
            anchor_node=None,
        ),
        metadata={"operator_kind": "harmonic_wrap", "periodic_kind": "cos", "form": "cos_base"},
    )
    seed_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqr_mul:seed",
        expr=("mul", ("var", 0), ("sqr", rebound_carrier)),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "sqr_mul"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "power_inner_node": rebound_carrier,
            "hole_node": rebound_carrier,
        },
    )
    x = torch.tensor(
        [
            [0.18, 0.27, 0.33],
            [0.29, 0.41, 0.46],
            [0.37, 0.52, 0.61],
            [0.48, 0.66, 0.73],
            [0.59, 0.78, 0.87],
            [0.71, 0.89, 1.02],
            [0.84, 1.03, 1.15],
            [0.95, 1.16, 1.29],
        ],
        dtype=torch.float64,
    )
    y = torch.cos(x[:, 0] * x[:, 1]).unsqueeze(-1)

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("enum", ("var", 2)), ("enum", ("var", 1))], {"candidate_source_counts": {"enum": 2}}

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
        proposal_context=ProposalContext(
            basis_state=seed_state,
            basis_state_beam=(),
            diagnostics={"route": "test"},
            family_hints={},
        ),
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) in {"slot_search", "slot_search+exact_bound"}
    assert int(meta.get("basis_seed_candidates", 0) or 0) >= 1
    assert rows[0]["direct_metadata"]["hole_node"] == rebound_phase
    assert float(rows[0]["local_probe_mse"]) < 1.0e-6


def test_direct_operator_periodic_merges_exact_bound_atom_with_slot_search_rows():
    carrier = ("mul", ("var", 1), ("var", 2))
    spec = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_mul",
        scaffold_id="periodic:cos_mul:x0:(x1*x2)",
        parent_node=("mul", ("cos", carrier), ("var", 0)),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 0),
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos_mul:x0:(x1*x2)",
            periodic_kind="cos",
            hole_node=carrier,
            feature_node=("mul", ("cos", carrier), ("var", 0)),
            anchor_node=("var", 0),
        ),
        metadata={"operator_kind": "harmonic_wrap", "periodic_kind": "cos", "form": "cos_mul"},
    )
    x = torch.tensor(
        [
            [0.25, 0.35, 0.45, 0.55],
            [0.42, 0.51, 0.63, 0.74],
            [0.58, 0.69, 0.77, 0.88],
            [0.73, 0.82, 0.91, 1.02],
            [0.89, 0.97, 1.08, 1.16],
            [1.03, 1.12, 1.21, 1.29],
            [1.14, 1.24, 1.33, 1.41],
            [1.27, 1.35, 1.44, 1.53],
        ],
        dtype=torch.float64,
    )
    h = torch.cos(x[:, 1] * x[:, 2])
    y = (x[:, 0] * (x[:, 3] * h * h + h)).unsqueeze(-1)

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=4,
        max_depth=7,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[
            ("var", 0),
            ("mul", ("var", 0), ("var", 3)),
        ],
        pool_dims=[(0.0,), (0.0,)],
        preview_topk=32,
        solver_kwargs={},
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "slot_search+exact_bound"
    assert any(row["expr"] == ("mul", ("cos", carrier), ("var", 0)) for row in rows)


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


def test_pin_small_trig_carrier_prefers_compact_periodic_atoms():
    assert pin_small_trig_carrier(("cos", ("mul", ("var", 1), ("var", 2)))) is True
    assert pin_small_trig_carrier(("sin", ("var", 0))) is True
    assert pin_small_trig_carrier(("cos", ("add", ("var", 0), ("var", 1), ("var", 2)))) is False
    assert pin_small_trig_carrier(("var", 0)) is False


def test_direct_power_sqr_mul_plan_pins_trig_carriers():
    scaffold_spec = OuterScaffoldSpec(
        family="power",
        scaffold_id="power:sqr_mul:(x0*x3)",
        parent_node=("mul", ("mul", ("var", 0), ("var", 3)), ("sqr", ("const", 1.0))),
        hole_path=(2, 1),
        target_mode="full",
        anchor_node=("mul", ("var", 0), ("var", 3)),
        metadata={"form": "power_sqr_mul"},
    )
    spec = operator_application_from_scaffold(scaffold_spec)
    x = torch.tensor(
        [
            [0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4],
            [1.5, 1.6, 1.7, 1.8],
        ],
        dtype=torch.float64,
    )
    h = torch.cos(x[:, 1] * x[:, 2])
    y = (x[:, 0] * x[:, 3] * h * h).unsqueeze(-1)

    plan = direct_mod.build_direct_power_search_plan(
        spec,
        nvars=4,
        max_depth=7,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[
            ("cos", ("mul", ("var", 1), ("var", 2))),
            ("var", 1),
        ],
        pool_dims=[(0.0,), (0.0,)],
        preview_topk=4,
        solver_kwargs={},
    )

    assert isinstance(plan, PreparedCandidatesSearchPlan)
    seen_carriers = {
        cand.built.direct_metadata.get("hole_node")
        for cand in list(plan.candidates or ())
    }
    seen_variants = {
        str(cand.built.direct_metadata.get("power_variant", "") or "")
        for cand in list(plan.candidates or ())
        if cand.built.direct_metadata.get("hole_node") == ("cos", ("mul", ("var", 1), ("var", 2)))
    }
    assert ("cos", ("mul", ("var", 1), ("var", 2))) in seen_carriers
    assert seen_variants == {"square_only", "bias_square", "linear_square", "full_quadratic"}


def test_direct_power_sqr_mul_rebinding_searches_anchor_and_carrier():
    scaffold_spec = OuterScaffoldSpec(
        family="power",
        scaffold_id="power:sqr_mul:x0",
        parent_node=("mul", ("var", 0), ("sqr", ("const", 1.0))),
        hole_path=(2, 1),
        target_mode="full",
        anchor_node=("var", 0),
        metadata={"form": "power_sqr_mul"},
    )
    spec = operator_application_from_scaffold(scaffold_spec)
    x = torch.tensor(
        [
            [0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4],
            [1.5, 1.6, 1.7, 1.8],
        ],
        dtype=torch.float64,
    )
    h = torch.cos(x[:, 1] * x[:, 2])
    y = (x[:, 0] * x[:, 3] * h * h).unsqueeze(-1)
    anchor_node = ("mul", ("var", 0), ("var", 3))
    carrier_node = ("cos", ("mul", ("var", 1), ("var", 2)))

    plan = direct_mod.build_direct_power_search_plan(
        spec,
        nvars=4,
        max_depth=7,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[
            anchor_node,
            carrier_node,
            ("var", 0),
            ("var", 1),
        ],
        pool_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        preview_topk=4,
        solver_kwargs={"allow_slot_rebinding": True},
    )

    assert isinstance(plan, PreparedCandidatesSearchPlan)
    assert bool(plan.meta.get("anchor_rebinding", False)) is True
    seen_pairs = {
        (
            cand.built.direct_metadata.get("anchor_node"),
            cand.built.direct_metadata.get("hole_node"),
        )
        for cand in list(plan.candidates or ())
    }
    assert (anchor_node, carrier_node) in seen_pairs


def test_direct_power_sqr_mul_mixed_quadratic_head_gets_depth_slack():
    x = torch.tensor(
        [
            [1.10, 1.15, 1.20, 1.25],
            [1.20, 1.25, 1.30, 1.35],
            [1.30, 1.35, 1.40, 1.45],
            [1.40, 1.45, 1.50, 1.55],
            [1.50, 1.55, 1.60, 1.65],
            [1.60, 1.65, 1.70, 1.75],
        ],
        dtype=torch.float64,
    )
    carrier = ("cos", ("mul", ("var", 1), ("var", 2)))
    anchor = ("mul", ("var", 0), ("var", 3))
    h = torch.cos(x[:, 1] * x[:, 2])
    y = (x[:, 0] * x[:, 3] * (0.5 * h + h * h)).unsqueeze(-1)

    built = build_affine_power_candidate(
        scaffold_id="power:sqr_mul:debug",
        power_kind="sqr_mul",
        exponent=2.0,
        hole_node=carrier,
        anchor_node=anchor,
        h_fit=explorer_mod.eval_node(carrier, x),
        h_probe=explorer_mod.eval_node(carrier, x),
        anchor_fit=explorer_mod.eval_node(anchor, x),
        anchor_probe=explorer_mod.eval_node(anchor, x),
        max_depth=5,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        safe_eps=1.0e-8,
        source="unit",
    )

    row = score_direct_closure_candidate(
        bound_closure=built.bound_closure,
        design=built.design,
        y_fit=y,
        y_probe=y,
        max_depth=5,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        candidate_subtree_node=carrier,
        parent_sub_size=1,
        parent_sub_depth=1,
        parent_size=1,
        parent_depth=1,
        generation_source="unit",
        tuple_provenance="unit",
        proposal_family="closure_search_direct_power",
        local_mapping_kind="direct_power_head",
        direct_metadata=built.direct_metadata,
        seen_child_keys=set(),
        local_mapping_coeffs=None,
        local_mapping_nparams=built.local_mapping_nparams,
    )

    assert row is not None
    assert row["proposal_family"] == "closure_search_direct_power"
    assert row["candidate_child_depth"] == 6


def test_direct_power_square_only_nonunit_coeff_survives_depth_check():
    x = torch.tensor(
        [
            [1.10, 1.15, 1.20, 1.25],
            [1.20, 1.25, 1.30, 1.35],
            [1.30, 1.35, 1.40, 1.45],
            [1.40, 1.45, 1.50, 1.55],
            [1.50, 1.55, 1.60, 1.65],
            [1.60, 1.65, 1.70, 1.75],
        ],
        dtype=torch.float64,
    )
    carrier = ("cos", ("mul", ("var", 1), ("var", 2)))
    anchor = ("mul", ("var", 0), ("var", 3))
    h_fit = explorer_mod.eval_node(carrier, x)
    anchor_fit = explorer_mod.eval_node(anchor, x)
    h = torch.cos(x[:, 1] * x[:, 2])
    y = (0.8 * x[:, 0] * x[:, 3] * h * h).unsqueeze(-1)

    fit_ret = fit_direct_power_design(
        h_fit=h_fit,
        h_probe=h_fit,
        y_fit=y,
        y_probe=y,
        exponent=2.0,
        variant="square_only",
        anchor_fit=anchor_fit,
        anchor_probe=anchor_fit,
        safe_eps=1.0e-8,
    )

    assert fit_ret is not None
    _, _probe_mse, coeffs = fit_ret
    assert abs(float(coeffs[2]) - 1.0) > 1.0e-3
    assert direct_power_depth_slack_from_coeffs(coeffs, exponent=2.0) >= 1

    expr = materialize_direct_power_expr(
        hole_node=carrier,
        coeffs=coeffs,
        exponent=2.0,
        anchor_node=anchor,
        max_depth=5,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
    )

    assert expr is not None
    assert int(node_depth(expr)) == 6


def test_direct_operator_power_empty_exact_bound_falls_through_to_rebinding():
    rebound_anchor = ("mul", ("var", 0), ("var", 3))
    rebound_carrier = ("cos", ("mul", ("var", 1), ("var", 2)))
    x = torch.tensor(
        [
            [0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4],
            [1.5, 1.6, 1.7, 1.8],
        ],
        dtype=torch.float64,
    )
    h = torch.cos(x[:, 1] * x[:, 2])
    y = (x[:, 0] * x[:, 3] * h * h).unsqueeze(-1)
    var_dims = [(0.0,), (0.0,), (0.0,), (0.0,)]
    y_dims = (0.0,)
    pool_nodes = build_pool(4)
    pool_dims = [node_dims(node, var_dims) for node in pool_nodes]
    apps = enumerate_operator_applications(
        families=["power"],
        nvars=4,
        y_dims=y_dims,
        var_dims=var_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        anchors_per_family=8,
        max_scaffolds=8,
    )
    spec = next(app for app in apps if str(getattr(app, "scaffold_id", "")).startswith("power:sqr_mul"))

    def _fake_collect_direct_hole_candidates(**kwargs):
        return [("pool", rebound_carrier)], {"candidate_source_counts": {"pool": 1}}

    rows, status, meta = direct_mod.solve_direct_operator_preview_rows(
        spec,
        nvars=4,
        max_depth=6,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=var_dims,
        y_dims=y_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        preview_topk=8,
        solver_kwargs={},
        collect_direct_hole_candidates_fn=_fake_collect_direct_hole_candidates,
    )

    assert status == "direct_ok"
    assert rows
    assert str(meta.get("execution_mode", "")) == "slot_search+exact_bound"
    rebound_rows = [
        row for row in rows
        if row["direct_metadata"].get("anchor_node") == rebound_anchor
        and row["direct_metadata"].get("hole_node") == rebound_carrier
    ]
    assert rebound_rows
    assert float(rebound_rows[0]["local_probe_mse"]) < 1.0e-20


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


def test_direct_quadratic_builder_keeps_ratio_and_inverse_seed_blocks_for_sqrt():
    ratio_node = ("div", ("var", 0), ("var", 1))
    inv_node = ("div", ("const", 1.0), ("var", 2))
    spec = OperatorApplication(
        family="quadratic",
        operator_id="quadratic:sqrt",
        scaffold_id="quadratic:sqrt:ratio_diffsq",
        parent_node=("sqrt", ("const", 1.0)),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_quadratic_closure(
            scaffold_id="quadratic:sqrt:ratio_diffsq",
            quadratic_kind="sqrt",
            base_nodes=(("var", 0), ("var", 1)),
        ),
        metadata={"operator_kind": "quadratic_wrap", "quadratic_kind": "sqrt"},
    )
    x = torch.tensor(
        [
            [2.5, 1.0, 4.0],
            [3.1, 1.2, 5.0],
            [3.8, 1.4, 5.5],
            [4.4, 1.5, 6.2],
        ],
        dtype=torch.float64,
    )
    ratio = x[:, 0] / x[:, 1]
    inv = 1.0 / x[:, 2]
    y = torch.sqrt(torch.square(ratio) - 0.5 * torch.square(inv)).unsqueeze(-1)

    plan = direct_mod.build_direct_quadratic_search_plan(
        spec,
        nvars=3,
        max_depth=8,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[ratio_node, inv_node],
        pool_dims=[(0.0,), (0.0,)],
        preview_topk=4,
        solver_kwargs={
            "allow_slot_rebinding": True,
            "direct_quadratic_base_topk": 4,
        },
    )

    assert isinstance(plan, SeedSubsetSearchPlan)
    seed_texts = {node_str(block.node) for block in plan.seed_blocks}
    assert node_str(ratio_node) in seed_texts
    assert node_str(inv_node) in seed_texts

    rows, status, _meta = execute_direct_search_plan(
        plan,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        max_depth=8,
        y_dims=(0.0,),
        preview_topk=4,
        deadline_s=None,
        collect_direct_hole_candidates_fn=None,
    )

    assert status == "direct_ok"
    matched_rows = [
        row
        for row in rows
        if {node_str(node) for node in row["direct_metadata"].get("quadratic_base_nodes", [])}
        == {node_str(ratio_node), node_str(inv_node)}
    ]
    assert matched_rows
    assert float(matched_rows[0]["local_probe_mse"]) < 1.0e-20


def test_direct_quadratic_builder_keeps_affine_difference_seed_blocks_for_distance_norm():
    spec = OperatorApplication(
        family="quadratic",
        operator_id="quadratic:sqrt",
        scaffold_id="quadratic:sqrt:distance_norm",
        parent_node=("sqrt", ("const", 1.0)),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_quadratic_closure(
            scaffold_id="quadratic:sqrt:distance_norm",
            quadratic_kind="sqrt",
            base_nodes=(("var", 0), ("var", 1)),
        ),
        metadata={"operator_kind": "quadratic_wrap", "quadratic_kind": "sqrt"},
    )
    x = torch.tensor(
        [
            [0.2, 0.7, 0.1, 0.8],
            [0.4, 1.0, 0.3, 0.9],
            [0.8, 1.5, 0.5, 1.4],
            [1.1, 1.9, 0.9, 1.8],
        ],
        dtype=torch.float64,
    )
    diff01 = x[:, 1] - x[:, 0]
    diff23 = x[:, 3] - x[:, 2]
    y = torch.sqrt(torch.square(diff01) + torch.square(diff23)).unsqueeze(-1)

    plan = direct_mod.build_direct_quadratic_search_plan(
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
        preview_topk=6,
        solver_kwargs={
            "allow_slot_rebinding": True,
            "direct_quadratic_base_topk": 4,
            "direct_quadratic_affine_diff_topk": 2,
        },
    )

    assert isinstance(plan, SeedSubsetSearchPlan)
    diff_pairs = {
        frozenset((int(node[1][1]), int(node[2][1])))
        for block in plan.seed_blocks
        for node in [block.node]
        if isinstance(node, tuple)
        and len(node) >= 3
        and str(node[0]) == "sub"
        and isinstance(node[1], tuple)
        and isinstance(node[2], tuple)
        and str(node[1][0]) == "var"
        and str(node[2][0]) == "var"
    }
    assert frozenset((0, 1)) in diff_pairs
    assert frozenset((2, 3)) in diff_pairs

    rows, status, _meta = execute_direct_search_plan(
        plan,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        max_depth=8,
        y_dims=(0.0,),
        preview_topk=6,
        deadline_s=None,
        collect_direct_hole_candidates_fn=None,
    )

    assert status == "direct_ok"
    matched_rows = [
        row
        for row in rows
        if {
            frozenset((int(node[1][1]), int(node[2][1])))
            for node in row["direct_metadata"].get("quadratic_base_nodes", [])
            if isinstance(node, tuple)
            and len(node) >= 3
            and str(node[0]) == "sub"
            and isinstance(node[1], tuple)
            and isinstance(node[2], tuple)
            and str(node[1][0]) == "var"
            and str(node[2][0]) == "var"
        }
        == {frozenset((0, 1)), frozenset((2, 3))}
    ]
    assert matched_rows
    assert float(matched_rows[0]["local_probe_mse"]) < 1.0e-20


def test_closure_search_rational_affine_uses_direct_fill_without_inverse(monkeypatch):
    scaffold_spec = scaffold_mod.OuterScaffoldSpec(
        family="rational",
        scaffold_id="rational:affine",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        metadata={"form": "rational_affine"},
    )

    def _fake_enumerate_closure_search_specs(**kwargs):
        return [scaffold_spec]

    def _raise_inverse_path(*args, **kwargs):
        raise AssertionError("inverse scaffold path should not run for direct rational scaffolds")

    monkeypatch.setattr(scaffold_mod, "LEGACY_SCAFFOLD_ADAPTER_ENABLED", True)
    monkeypatch.setattr(scaffold_mod, "enumerate_closure_search_specs", _fake_enumerate_closure_search_specs)
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
