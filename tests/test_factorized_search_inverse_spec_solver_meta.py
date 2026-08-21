# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

import nestynet_sr.sr_search.factorized_search.tangent_edit as tangent_mod
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node, node_dims, node_str
from nestynet_sr.sr_search.factorized_search.expr_ast import build_pool as build_pool_nodes
from nestynet_sr.sr_search.factorized_search.inverse_action import (
    _estimate_inverse_action_transport,
    _inverse_action_path_mode_beam_states,
)
from nestynet_sr.sr_search.factorized_search.lift_route_evidence import build_local_lift_route_context
from nestynet_sr.sr_search.factorized_search.inverse_spec_solver import solve_inverse_spec_preview_rows
from nestynet_sr.sr_search.factorized_search.inverse_spec_solver import (
    _PERIODIC_CONFIDENCE_THRESHOLD,
    _LocalProblem,
    _ScoredLocalCandidate,
    _SolverContext,
    _flat_solve_local_problem,
    _select_binary_recursive_anchors,
    _solve_local_problem,
    _score_candidate_sinusoidal,
    _should_run_periodic_forward,
    solve_local_problem_spec_preview_rows,
)
from nestynet_sr.sr_search.factorized_search.subproblem_spec import deserialize_subproblem_spec


def _safe_pool_phi(pool_nodes, x):
    cols = []
    for node in pool_nodes:
        try:
            vals = eval_node(node, x).squeeze(-1)
        except Exception:
            vals = torch.zeros((int(x.shape[0]),), dtype=x.dtype)
        if not torch.isfinite(vals).all():
            vals = torch.zeros_like(vals)
        cols.append(vals)
    return torch.stack(cols, dim=1)


def _make_problem(truth_expr, candidate_expr, *, nvars, var_dims=None, seed=0):
    g_fit = torch.Generator().manual_seed(int(seed))
    g_probe = torch.Generator().manual_seed(int(seed) + 17)
    x_fit = 0.5 + 1.5 * torch.rand((64, int(nvars)), generator=g_fit, dtype=torch.float64)
    x_probe = 0.5 + 1.5 * torch.rand((96, int(nvars)), generator=g_probe, dtype=torch.float64)
    y_fit = eval_node(truth_expr, x_fit)
    y_probe = eval_node(truth_expr, x_probe)
    pool_nodes = build_pool_nodes(int(nvars))
    pool_dims = [node_dims(node, var_dims) for node in pool_nodes] if var_dims is not None else [None] * len(pool_nodes)
    return {
        "candidate_expr": candidate_expr,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_probe": x_probe,
        "y_probe": y_probe,
        "pool_nodes": pool_nodes,
        "pool_dims": pool_dims,
        "pool_phi_fit": _safe_pool_phi(pool_nodes, x_fit),
        "pool_phi_probe": _safe_pool_phi(pool_nodes, x_probe),
    }


def _make_problem_with_samples(truth_expr, candidate_expr, *, x_fit, x_probe, var_dims=None):
    nvars = int(x_fit.shape[1])
    y_fit = eval_node(truth_expr, x_fit)
    y_probe = eval_node(truth_expr, x_probe)
    pool_nodes = build_pool_nodes(int(nvars))
    pool_dims = [node_dims(node, var_dims) for node in pool_nodes] if var_dims is not None else [None] * len(pool_nodes)
    return {
        "candidate_expr": candidate_expr,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_probe": x_probe,
        "y_probe": y_probe,
        "pool_nodes": pool_nodes,
        "pool_dims": pool_dims,
        "pool_phi_fit": _safe_pool_phi(pool_nodes, x_fit),
        "pool_phi_probe": _safe_pool_phi(pool_nodes, x_probe),
    }


def _build_beam_state(problem, *, path, var_dims=None):
    from nestynet_sr.sr_search.factorized_search import explorer

    fit_result = explorer.fit_best(
        eval_node(problem["candidate_expr"], problem["x_fit"]),
        problem["y_fit"],
        4,
    )
    assert fit_result is not None
    _fit_mse, mapping = fit_result
    transport_ctx = _estimate_inverse_action_transport(
        problem["candidate_expr"],
        mapping,
        problem["x_fit"],
        problem["y_fit"],
        problem["x_probe"],
        problem["y_probe"],
        [path],
        safe_eps=1.0e-12,
    )
    beam_states = _inverse_action_path_mode_beam_states(
        parent_node=problem["candidate_expr"],
        parent_mapping=mapping,
        x_fit=problem["x_fit"],
        y_fit=problem["y_fit"],
        x_probe=problem["x_probe"],
        y_probe=problem["y_probe"],
        pool_nodes=problem["pool_nodes"],
        pool_phi_fit=problem["pool_phi_fit"],
        pool_phi_probe=problem["pool_phi_probe"],
        pool_dims=problem["pool_dims"],
        all_paths=[path],
        path_target_modes=None,
        transport_ctx=transport_ctx,
        cfg={
            "max_paths": 1,
            "dm": bool(var_dims is not None),
            "var_dims": var_dims,
            "max_depth": 4,
            "poly_degree": 4,
            "topk_terms": 6,
            "shortlist_mult": 4,
            "local_mode": "affine",
            "min_valid_frac": 0.25,
            "min_confidence": 0.10,
            "safe_eps": 1.0e-12,
            "confidence_mode": "conditioning",
            "confidence_target_gain": 4.0,
            "confidence_floor": 0.05,
            "branch_beam_width": 1,
            "micro_search_enable": False,
            "micro_search_max_depth": 3,
            "micro_search_beam_width": 24,
            "micro_search_topk": 16,
            "micro_search_seed_terms": 8,
            "target_mode": "robust",
            "full_mapping_penalty": 0.75,
            "exact_simple_target_bonus": 0.10,
            "additive_descend_penalty": 0.15,
            "nonadditive_leaf_penalty": 0.20,
            "exact_path_eta": 0.98,
            "exact_transport_min_lin_rel": 0.0,
            "periodic_min_valid_scale": 1.25,
            "periodic_min_confidence_scale": 1.35,
            "periodic_path_penalty": 0.65,
            "nonperiodic_muldiv_bonus": 0.10,
            "nonperiodic_explogsqrt_bonus": 0.05,
            "branch_ambiguity_penalty": 0.50,
            "transport_min_lin_rel": 0.02,
            "transport_min_effective_n": 8.0,
        },
        beam_width=1,
    )
    assert beam_states
    return beam_states[0]


class _MulTeacher:
    def grad(self, x):
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 1]
        out[:, 1] = x[:, 0]
        return out

    def grad_grad(self, x):
        out = torch.zeros((int(x.shape[0]), int(x.shape[1]), int(x.shape[1])), dtype=x.dtype, device=x.device)
        return out


def test_inverse_spec_solver_reports_stage_timing_metadata():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("var", 0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=3)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-flat",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=2,
        enum_max_trees=128,
        preview_topk=6,
        recursive_enable=True,
        recursive_max_depth=2,
    )
    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["preview_count"] > 0
    assert meta["wall_seconds"] >= 0.0
    assert meta["periodic_forward_used"] is False
    assert meta["periodic_precheck_status"] in {
        "explicit_inverse_confident",
        "insufficient_improvement",
        "no_candidate_nodes",
        "no_finite_periodic_fit",
    }
    assert meta["stage_wall_seconds"]["flat_collect"] >= 0.0
    assert meta["stage_wall_seconds"]["flat_solve"] >= 0.0
    assert meta["stage_wall_seconds"]["solve_local_problem"] >= 0.0
    assert meta["stage_wall_seconds"]["preview_row_build"] >= 0.0
    assert meta["score_node_generation_counts"]
    assert meta["score_node_generation_wall_seconds"]
    assert all(v >= 0 for v in meta["score_node_generation_counts"].values())
    assert all(v >= 0.0 for v in meta["score_node_generation_wall_seconds"].values())
    assert set(meta["outer_family_precheck_status"].keys()) >= {"periodic", "exp", "power", "rational"}
    assert meta["stage_wall_seconds"]["outer_family"] >= 0.0
    assert meta["stage_wall_seconds"]["outer_family_dispatch"] >= 0.0


def test_inverse_spec_solver_exports_followup_spec_states_with_continuations():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("const", 1.0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=11)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-followup",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=6,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_branch_topk=4,
    )
    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["child_spec_state_count"] >= 1
    child = meta["child_spec_states"][0]
    assert child["spec_kind"] == "local_problem"
    assert child["path"] == [1]
    assert child["continuation_key"]
    assert isinstance(child["trace"], list)
    assert int(child["recursion_level"]) >= 1
    payload = child["spec_payload"]
    assert isinstance(payload, dict)
    assert isinstance(payload.get("problem"), dict)
    assert payload["schema_name"] == "factorized_search.subproblem_spec"
    assert payload["schema_version"] == 1
    spec = deserialize_subproblem_spec(payload)
    assert spec is not None
    assert spec.problem_kind == "local_problem"
    assert spec.path == (1,)
    assert spec.direction == "outside_in"
    assert spec.witness is not None
    assert spec.metadata["hole_sub"] == beam_state["sub"]
    frames = payload.get("continuation_frames")
    assert isinstance(frames, list) and frames
    assert isinstance(frames[0], dict)
    assert frames[0]["op"]


def test_inverse_spec_solver_followup_specs_populate_active_vars_from_gradient_screen():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("const", 1.0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=19)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-active-vars",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=6,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_branch_topk=4,
        active_var_screen_enable=True,
        active_var_grad_tol=0.05,
        active_var_max_count=2,
    )

    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["child_spec_state_count"] >= 1
    payload = meta["child_spec_states"][0]["spec_payload"]
    spec = deserialize_subproblem_spec(payload)
    assert spec is not None
    assert spec.active_vars == (0, 1)
    assert spec.metadata["active_var_source"] == "gradient"


def test_inverse_spec_solver_followup_specs_preserve_regime_context():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("const", 1.0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=29)
    beam_state = _build_beam_state(problem, path=(1,))
    regime_metadata = {
        "dataset_ids": ["d0", "d1", "d2"],
        "dataset_metadata": {
            "d0": {"temperature": 0.0},
            "d1": {"temperature": 1.0},
            "d2": {"temperature": 2.0},
        },
        "local_constants_by_experiment": {
            "d0": {"local_leaf": 1.0, "stable_leaf": 5.0},
            "d1": {"local_leaf": 2.0, "stable_leaf": 5.05},
            "d2": {"local_leaf": 4.0, "stable_leaf": 4.95},
        },
        "constant_lift_feature_nodes": [("var", 1)],
    }
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        regime_metadata=regime_metadata,
        beam_rank=0,
        slate_id="meta-regime-context",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=6,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_branch_topk=4,
    )

    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["child_spec_state_count"] >= 1
    payload = meta["child_spec_states"][0]["spec_payload"]
    spec = deserialize_subproblem_spec(payload)
    assert spec is not None
    diagnostics = dict(spec.witness.diagnostics or {})
    assert diagnostics["dataset_ids"] == ["d0", "d1", "d2"]
    assert diagnostics["dataset_metadata"]["d1"]["temperature"] == 1.0
    assert diagnostics["local_constants_by_experiment"]["d2"]["local_leaf"] == 4.0
    assert diagnostics["constant_lift_feature_nodes"] == [("var", 1)]

    route_context = build_local_lift_route_context(payload)
    constant_lift = dict(route_context.get("constant_lift", {}) or {})
    assert constant_lift["preferred"] is True
    assert constant_lift["status"] == "drifting_constants"
    assert constant_lift["top_constant_name"] == "local_leaf"


def test_solve_local_problem_spec_preview_rows_accepts_legacy_and_canonical_payloads():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("const", 1.0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=13)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-followup-compat",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=6,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_branch_topk=4,
    )
    child = result["solver_meta"]["child_spec_states"][0]
    canonical_payload = dict(child["spec_payload"])
    legacy_payload = {
        "problem": canonical_payload["problem"],
        "continuation_frames": canonical_payload["continuation_frames"],
        "hole_sub": canonical_payload["hole_sub"],
    }

    canonical = solve_local_problem_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        spec_payload=canonical_payload,
        path=(1,),
        target_mode=str(beam_state.get("target_mode", "") or ""),
        target_mapping_kind=str(beam_state.get("target_mapping_kind", "") or ""),
        beam_rank=0,
        slate_id="canonical-followup",
        path_gain=float(beam_state.get("path_gain", 0.0) or 0.0),
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=4,
        recursive_enable=True,
        recursive_max_depth=2,
    )
    legacy = solve_local_problem_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        spec_payload=legacy_payload,
        path=(1,),
        target_mode=str(beam_state.get("target_mode", "") or ""),
        target_mapping_kind=str(beam_state.get("target_mapping_kind", "") or ""),
        beam_rank=0,
        slate_id="legacy-followup",
        path_gain=float(beam_state.get("path_gain", 0.0) or 0.0),
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=4,
        recursive_enable=True,
        recursive_max_depth=2,
    )

    assert canonical["solver_meta"]["status"] == "ok"
    assert legacy["solver_meta"]["status"] == "ok"
    assert canonical["rows"]
    assert legacy["rows"]
    assert canonical["rows"][0]["child_key"] == legacy["rows"][0]["child_key"]


def test_inverse_spec_followup_specs_capture_witness_jets_when_enabled():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("var", 0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=19)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-followup-witness",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=6,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_branch_topk=4,
        witness_jets_enable=True,
        witness_d2_enable=True,
        witness_max_rows=40,
    )
    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["witness_jets_enable"] is True
    assert meta["witness_d2_enable"] is True
    assert meta["child_spec_state_count"] >= 1

    child = meta["child_spec_states"][0]
    spec = deserialize_subproblem_spec(child["spec_payload"])
    assert spec is not None
    assert spec.witness is not None
    assert spec.witness.grad_fit is not None
    assert spec.witness.grad_probe is not None
    assert spec.witness.d2_fit is not None
    assert spec.witness.d2_probe is not None
    assert tuple(spec.witness.grad_fit.shape) == tuple(spec.witness.x_fit.shape)
    assert tuple(spec.witness.grad_probe.shape) == tuple(spec.witness.x_probe.shape)
    assert spec.witness.diagnostics["witness_jets_enabled"] is True
    assert spec.witness.diagnostics["fit_jet_source"] == "numeric_local_quadratic"
    assert spec.witness.diagnostics["probe_jet_source"] == "numeric_local_quadratic"
    assert spec.metadata["teacher_spec"]["source"] == "numeric_local_quadratic"

    tangent = tangent_mod.solve_local_tangent_edit_preview_rows(
        parent_node=problem["candidate_expr"],
        spec_payload=child["spec_payload"],
        path=(1,),
        target_mode=str(beam_state.get("target_mode", "") or ""),
        target_mapping_kind=str(beam_state.get("target_mapping_kind", "") or ""),
        beam_rank=0,
        slate_id="meta-followup-tangent",
        path_gain=float(beam_state.get("path_gain", 0.0) or 0.0),
        max_depth=4,
        nvars=3,
        poly_degree=4,
        var_dims=None,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        preview_topk=4,
        max_subtree_depth=3,
    )
    assert tangent["solver_meta"]["status"] == "ok"
    assert tangent["solver_meta"]["target_gradient_used"] is True


def test_inverse_spec_solver_uses_oracle_teacher_runtime_for_direct_top_problem_jets():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("var", 0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=23)
    beam_state = dict(_build_beam_state(problem, path=(1,)))
    beam_state["teacher_spec"] = {"source": "oracle"}
    beam_state["teacher_runtime"] = _MulTeacher()

    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-direct-oracle-jets",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=4,
        witness_jets_enable=True,
        witness_d2_enable=True,
        witness_loss_enable=True,
        witness_grad_weight=1.0,
        recursive_enable=False,
        recursive_max_depth=0,
    )

    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["top_fit_jet_source"] == "oracle"
    assert meta["top_probe_jet_source"] == "oracle"
    assert meta["top_fit_jet_requested_source"] == "oracle"
    assert meta["top_probe_jet_requested_source"] == "oracle"
    assert meta["top_fit_jet_fallback_used"] is False
    assert meta["top_probe_jet_fallback_used"] is False
    assert result["rows"]
    assert result["rows"][0]["witness_grad_loss"] is not None


def test_inverse_spec_followup_specs_preserve_oracle_teacher_jets_across_serialization():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("var", 0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=29)
    beam_state = dict(_build_beam_state(problem, path=(1,)))
    beam_state["teacher_spec"] = {"source": "oracle"}
    beam_state["teacher_runtime"] = _MulTeacher()

    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-followup-oracle-jets",
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=6,
        witness_jets_enable=True,
        witness_d2_enable=True,
        witness_loss_enable=True,
        witness_grad_weight=1.0,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_branch_topk=4,
    )

    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["child_spec_state_count"] >= 1
    child_spec = deserialize_subproblem_spec(meta["child_spec_states"][0]["spec_payload"])
    assert child_spec is not None
    assert child_spec.witness is not None
    assert child_spec.witness.grad_fit is not None
    assert child_spec.witness.grad_probe is not None
    assert child_spec.witness.diagnostics["fit_jet_source"] == "oracle"
    assert child_spec.witness.diagnostics["probe_jet_source"] == "oracle"
    assert child_spec.witness.diagnostics["fit_jet_fallback_used"] is False
    assert child_spec.witness.diagnostics["probe_jet_fallback_used"] is False
    assert child_spec.metadata["teacher_spec"]["source"] == "oracle"
    assert child_spec.metadata["teacher_spec"]["derivation"] == "recursive_inverse"

    followup = solve_local_problem_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        spec_payload=meta["child_spec_states"][0]["spec_payload"],
        path=(1,),
        target_mode=str(beam_state.get("target_mode", "") or ""),
        target_mapping_kind=str(beam_state.get("target_mapping_kind", "") or ""),
        beam_rank=0,
        slate_id="meta-followup-oracle-jets-replay",
        path_gain=float(beam_state.get("path_gain", 0.0) or 0.0),
        max_depth=4,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=4,
        recursive_enable=True,
        recursive_max_depth=1,
        recursive_trigger_rel_mse=0.0,
        recursive_branch_topk=4,
        witness_jets_enable=True,
        witness_d2_enable=True,
        witness_loss_enable=True,
        witness_grad_weight=1.0,
    )
    assert followup["solver_meta"]["status"] == "ok"
    assert followup["solver_meta"]["child_spec_state_count"] >= 1
    grandchild_spec = deserialize_subproblem_spec(followup["solver_meta"]["child_spec_states"][0]["spec_payload"])
    assert grandchild_spec is not None
    assert grandchild_spec.witness is not None
    assert grandchild_spec.witness.grad_fit is not None
    assert grandchild_spec.witness.grad_probe is not None
    assert grandchild_spec.witness.diagnostics["fit_jet_source"] == "oracle"
    assert grandchild_spec.witness.diagnostics["probe_jet_source"] == "oracle"
    assert grandchild_spec.witness.diagnostics["fit_jet_fallback_used"] is False
    assert grandchild_spec.witness.diagnostics["probe_jet_fallback_used"] is False


def test_inverse_spec_solver_reports_periodic_timing_when_triggered():
    truth_expr = (
        "add",
        ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 3)),
        ("cos", ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 2))),
    )
    candidate_expr = (
        "add",
        ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 3)),
        ("const", 1.0),
    )
    problem = _make_problem(truth_expr, candidate_expr, nvars=4, seed=23)
    beam_state = _build_beam_state(problem, path=(2,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-trig",
        max_depth=5,
        nvars=4,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=2,
        enum_max_trees=256,
        preview_topk=8,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
    )
    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["periodic_forward_used"] is True
    assert meta["periodic_forward_candidate_count"] > 0
    assert meta["periodic_precheck_status"] == "triggered"
    assert meta["periodic_sinusoidal_count"] > 0
    assert meta["periodic_explicit_inverse_branch_count"] >= 0
    assert meta["periodic_explicit_inverse_supported_confidence"] <= float(_PERIODIC_CONFIDENCE_THRESHOLD) + 1.0e-9
    assert meta["stage_wall_seconds"]["periodic_forward"] > 0.0
    assert meta["stage_wall_seconds"]["periodic_sinusoidal"] > 0.0
    assert meta["score_node_generation_counts"].get("periodic_forward", 0) > 0
    assert meta["outer_family_used"].get("periodic", False) is True
    assert meta["outer_family_candidate_counts"].get("periodic", 0) > 0


def test_inverse_spec_solver_exports_outer_family_evidence_when_enabled():
    truth_expr = (
        "add",
        ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 3)),
        ("cos", ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 2))),
    )
    candidate_expr = (
        "add",
        ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 3)),
        ("const", 1.0),
    )
    problem = _make_problem(truth_expr, candidate_expr, nvars=4, seed=29)
    beam_state = _build_beam_state(problem, path=(2,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-family-evidence",
        max_depth=5,
        nvars=4,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=2,
        enum_max_trees=256,
        preview_topk=8,
        family_battery_enable=True,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
    )

    meta = result["solver_meta"]
    evidence = meta["outer_family_evidence"]
    periodic = evidence["periodic"]
    status = periodic["hard_constraints"]["status"]

    assert meta["status"] == "ok"
    assert set(evidence.keys()) >= {"periodic", "exp", "power", "rational"}
    assert status in {
        "triggered",
        "insufficient_improvement",
        "explicit_inverse_confident",
        "no_candidate_nodes",
        "no_finite_fit",
    }
    assert periodic["hard_constraints"]["should_run"] is (status == "triggered")
    assert periodic["metadata"]["candidate_count"] > 0
    assert meta["outer_family_precheck_status"]["periodic"] in {
        "triggered",
        "insufficient_improvement",
        "explicit_inverse_confident",
        "no_candidate_nodes",
        "no_finite_periodic_fit",
    }


def test_inverse_spec_solver_exports_expanded_family_evidence_when_enabled():
    truth_expr = (
        "add",
        ("mul", ("add", ("var", 0), ("var", 1)), ("add", ("var", 0), ("var", 1))),
        ("var", 2),
    )
    candidate_expr = ("add", ("const", 1.0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=31)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-family-evidence-expanded",
        max_depth=5,
        nvars=3,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=2,
        enum_max_trees=256,
        preview_topk=8,
        family_battery_enable=True,
        family_battery_mode="expanded",
        witness_jets_enable=True,
        witness_d2_enable=True,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
    )

    meta = result["solver_meta"]
    evidence = meta["outer_family_evidence"]

    assert meta["status"] == "ok"
    assert meta["family_battery_mode"] == "expanded"
    assert set(evidence.keys()) >= {
        "periodic",
        "exp",
        "power",
        "rational",
        "symmetry",
        "separability",
        "low_rank_dependence",
        "domain_hazard",
        "asymptotic_monomial",
        "branch_structure",
        "coordinate_invariant",
        "regime_lift",
    }
    assert evidence["symmetry"]["metadata"]["status"] in {"even_like", "odd_like", "asymmetric"}
    assert evidence["symmetry"]["metadata"]["jet_evidence_used"] is True
    assert evidence["low_rank_dependence"]["metadata"]["jet_evidence_used"] is True
    assert evidence["separability"]["hard_constraints"]["should_run"] is False
    assert evidence["domain_hazard"]["metadata"]["advisory_only"] is True
    power = evidence["power"]
    assert set(power["metadata"]["expanded_family_signals"].keys()) >= {
        "asymptotic_monomial",
        "branch_structure",
        "coordinate_invariant",
        "regime_lift",
    }
    assert power["hard_constraints"]["asymptotic_monomial_status"] in {
        "monomial_like",
        "non_monomial",
        "insufficient_tail",
    }
    assert power["hard_constraints"]["coordinate_invariant_status"] in {
        "single_index_like",
        "mixed_coordinate",
        "insufficient_gradient_evidence",
    }


def test_inverse_spec_solver_expanded_family_battery_blocks_domain_hazard_families():
    truth_expr = (
        "add",
        ("div", ("const", 1.0), ("var", 0)),
        ("var", 1),
    )
    candidate_expr = ("add", ("const", 1.0), ("var", 1))
    x0_fit = torch.linspace(0.05, 1.00, 64, dtype=torch.float64).unsqueeze(-1)
    x1_fit = torch.linspace(-0.25, 0.25, 64, dtype=torch.float64).unsqueeze(-1)
    x_fit = torch.cat([x0_fit, x1_fit], dim=1)
    x0_probe = torch.linspace(0.055, 1.05, 96, dtype=torch.float64).unsqueeze(-1)
    x1_probe = torch.linspace(-0.30, 0.30, 96, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.cat([x0_probe, x1_probe], dim=1)
    problem = _make_problem_with_samples(truth_expr, candidate_expr, x_fit=x_fit, x_probe=x_probe)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-family-hazard-expanded",
        max_depth=5,
        nvars=2,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=2,
        enum_max_trees=128,
        preview_topk=6,
        family_battery_enable=True,
        family_battery_mode="expanded",
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
    )

    meta = result["solver_meta"]
    hazard = meta["outer_family_evidence"]["domain_hazard"]

    assert hazard["hard_constraints"]["hazard_severe"] is True
    assert meta["outer_family_precheck_status"]["exp"] == "domain_hazard"
    assert meta["outer_family_precheck_status"]["power"] == "domain_hazard"


def test_inverse_spec_solver_blocks_periodic_when_explicit_trig_inverse_is_confident():
    truth_expr = ("add", ("sin", ("var", 0)), ("var", 1))
    candidate_expr = ("add", ("const", 1.0), ("var", 1))
    x0_fit = torch.linspace(0.05, 0.45, 64, dtype=torch.float64).unsqueeze(-1)
    x1_fit = torch.linspace(0.50, 1.50, 64, dtype=torch.float64).unsqueeze(-1)
    x_fit = torch.cat([x0_fit, x1_fit], dim=1)
    x0_probe = torch.linspace(0.08, 0.48, 96, dtype=torch.float64).unsqueeze(-1)
    x1_probe = torch.linspace(0.55, 1.55, 96, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.cat([x0_probe, x1_probe], dim=1)
    problem = _make_problem_with_samples(truth_expr, candidate_expr, x_fit=x_fit, x_probe=x_probe)
    beam_state = _build_beam_state(problem, path=(1,))
    ctx = _SolverContext(
        parent_node=problem["candidate_expr"],
        hole_path=(1,),
        hole_sub=beam_state["sub"],
        max_depth=5,
        nvars=2,
        poly_degree=4,
        var_dims=None,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        seed_nodes=[beam_state["sub"]],
        local_score_mode="affine",
        enum_max_depth=2,
        enum_max_trees=128,
        max_subtree_depth=5,
        preview_topk=6,
        complexity_penalty=0.0,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=6,
        recursive_branch_topk=4,
        recursive_child_topk=2,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        min_valid_frac=0.10,
        min_confidence=0.10,
        allow_legacy_aux=True,
        legacy_aux_kwargs={},
        stats={},
    )
    local_problem = _LocalProblem(
        xf=beam_state["xf"],
        tf=beam_state["tf"],
        wf=beam_state.get("wf"),
        xp=beam_state["xp"],
        tp=beam_state["tp"],
        wp=beam_state.get("wp"),
        target_dim=beam_state.get("target_dim"),
        confidence=float(beam_state.get("confidence", 1.0) or 1.0),
        valid_frac=float(beam_state.get("valid_frac", 1.0) or 1.0),
        wrappers_left=2,
        recursion_level=0,
        trace=tuple(),
    )
    flat_rows, _flat_meta = _flat_solve_local_problem(
        local_problem,
        ctx=ctx,
        include_legacy_aux=True,
    )
    should_run = _should_run_periodic_forward(local_problem, flat_rows, ctx=ctx)
    assert should_run is False
    assert ctx.stats["periodic_precheck_status"] == "explicit_inverse_confident"
    assert ctx.stats["periodic_explicit_inverse_branch_count"] > 0
    assert ctx.stats["periodic_explicit_inverse_supported_confidence"] >= float(_PERIODIC_CONFIDENCE_THRESHOLD)


def test_inverse_spec_solver_registry_triggers_exp_family():
    truth_expr = (
        "add",
        ("add", ("mul", ("const", 1.5), ("exp", ("mul", ("const", 2.0), ("var", 0)))), ("const", 0.3)),
        ("var", 1),
    )
    candidate_expr = ("add", ("const", 1.0), ("var", 1))
    problem = _make_problem(truth_expr, candidate_expr, nvars=2, seed=11)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-exp",
        max_depth=10,
        nvars=2,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=10,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
    )
    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["outer_family_precheck_status"].get("exp", "") == "triggered"
    assert meta["outer_family_used"].get("exp", False) is True
    assert meta["outer_family_candidate_counts"].get("exp", 0) > 0


def test_inverse_spec_solver_registry_triggers_power_family():
    truth_expr = (
        "add",
        ("exp", ("mul", ("const", 2.0), ("log", ("var", 0)))),
        ("var", 1),
    )
    candidate_expr = ("add", ("const", 1.0), ("var", 1))
    problem = _make_problem(truth_expr, candidate_expr, nvars=2, seed=7)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-power",
        max_depth=8,
        nvars=2,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=10,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
    )
    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["outer_family_precheck_status"].get("power", "") == "triggered"
    assert meta["outer_family_used"].get("power", False) is True
    assert meta["outer_family_candidate_counts"].get("power", 0) > 0


def test_inverse_spec_solver_registry_triggers_rational_family():
    truth_expr = (
        "add",
        ("div", ("const", 1.0), ("var", 0)),
        ("var", 1),
    )
    candidate_expr = ("add", ("const", 1.0), ("var", 1))
    problem = _make_problem(truth_expr, candidate_expr, nvars=2, seed=9)
    beam_state = _build_beam_state(problem, path=(1,))
    result = solve_inverse_spec_preview_rows(
        parent_node=problem["candidate_expr"],
        beam_state=beam_state,
        beam_rank=0,
        slate_id="meta-rational",
        max_depth=8,
        nvars=2,
        poly_degree=4,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        enum_max_depth=1,
        enum_max_trees=128,
        preview_topk=10,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
    )
    meta = result["solver_meta"]
    assert meta["status"] == "ok"
    assert meta["outer_family_precheck_status"].get("rational", "") == "triggered"
    assert meta["outer_family_used"].get("rational", False) is True
    assert meta["outer_family_candidate_counts"].get("rational", 0) > 0


def test_inverse_spec_periodic_rows_preserve_calibration_gap_for_sin_cos_realizations():
    periodic_combo = (
        "add",
        (
            "add",
            ("mul", ("const", 0.7), ("sin", ("var", 0))),
            ("mul", ("const", 0.7), ("cos", ("var", 0))),
        ),
        ("const", 0.1),
    )
    truth_expr = ("add", periodic_combo, ("var", 1))
    candidate_expr = ("add", ("const", 1.0), ("var", 1))
    problem = _make_problem(truth_expr, candidate_expr, nvars=2, seed=19)
    beam_state = _build_beam_state(problem, path=(1,))
    ctx = _SolverContext(
        parent_node=problem["candidate_expr"],
        hole_path=(1,),
        hole_sub=beam_state["sub"],
        max_depth=8,
        nvars=2,
        poly_degree=4,
        var_dims=None,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        seed_nodes=[beam_state["sub"]],
        local_score_mode="affine",
        enum_max_depth=1,
        enum_max_trees=128,
        max_subtree_depth=8,
        preview_topk=64,
        complexity_penalty=0.0,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=6,
        recursive_branch_topk=4,
        recursive_child_topk=2,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        min_valid_frac=0.25,
        min_confidence=0.10,
        allow_legacy_aux=True,
        legacy_aux_kwargs={},
        stats={},
    )
    local_problem = _LocalProblem(
        xf=beam_state["xf"],
        tf=beam_state["tf"],
        wf=beam_state.get("wf"),
        xp=beam_state["xp"],
        tp=beam_state["tp"],
        wp=beam_state.get("wp"),
        target_dim=beam_state.get("target_dim"),
        confidence=float(beam_state.get("confidence", 1.0) or 1.0),
        valid_frac=float(beam_state.get("valid_frac", 1.0) or 1.0),
        wrappers_left=2,
        recursion_level=0,
        trace=tuple(),
    )
    rows = _score_candidate_sinusoidal(
        ("var", 0),
        problem=local_problem,
        ctx=ctx,
        wrapper_op="sincos",
        source="test_periodic_gap",
        trace=tuple(),
    )
    rows_by_key = {node_str(row.node): row for row in rows}
    sin_key = node_str(("sin", ("var", 0)))
    cos_key = node_str(("cos", ("var", 0)))
    sin_row = rows_by_key.get(sin_key)
    cos_row = rows_by_key.get(cos_key)
    assert sin_row is not None
    assert cos_row is not None
    assert sin_row.family == "periodic"
    assert cos_row.family == "periodic"
    assert sin_row.surrogate_probe_mse is not None and sin_row.surrogate_probe_mse < 1.0e-6
    assert cos_row.surrogate_probe_mse is not None and cos_row.surrogate_probe_mse < 1.0e-6
    assert sin_row.local_probe_mse > 1.0e-2
    assert cos_row.local_probe_mse > 1.0e-3
    assert sin_row.calibration_gap is not None and sin_row.calibration_gap > 1.0e-2
    assert cos_row.calibration_gap is not None and cos_row.calibration_gap > 1.0e-3


def test_binary_recursive_anchor_selection_includes_outer_family_rows():
    ctx = _SolverContext(
        parent_node=("var", 0),
        hole_path=(1,),
        hole_sub=("var", 0),
        max_depth=5,
        nvars=2,
        poly_degree=4,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        seed_nodes=[],
        local_score_mode="affine",
        enum_max_depth=2,
        enum_max_trees=64,
        max_subtree_depth=5,
        preview_topk=6,
        complexity_penalty=0.0,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=3,
        recursive_branch_topk=4,
        recursive_child_topk=2,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        min_valid_frac=0.10,
        min_confidence=0.10,
        allow_legacy_aux=True,
        legacy_aux_kwargs={},
        stats={},
    )
    flat_bad = _ScoredLocalCandidate(
        node=("var", 0),
        local_probe_mse=0.30,
        local_fit_mse=0.30,
        source="flat",
        generation_kind="flat",
        recursion_depth=0,
        confidence=1.0,
        valid_frac=1.0,
        trace=tuple(),
    )
    outer_better_same_node = _ScoredLocalCandidate(
        node=("var", 0),
        local_probe_mse=0.10,
        local_fit_mse=0.10,
        source="outer_family:exp:flat_seed",
        generation_kind="outer_family",
        recursion_depth=0,
        confidence=1.0,
        valid_frac=1.0,
        trace=tuple(),
        family="exp",
        payload={"a": 1.0, "b": 0.0},
    )
    outer_unique = _ScoredLocalCandidate(
        node=("mul", ("var", 0), ("var", 1)),
        local_probe_mse=0.12,
        local_fit_mse=0.12,
        source="outer_family:rational:enum",
        generation_kind="outer_family",
        recursion_depth=0,
        confidence=1.0,
        valid_frac=1.0,
        trace=tuple(),
        family="rational",
        payload={"a": 1.0, "b": 0.0, "c": 0.0},
    )
    flat_other = _ScoredLocalCandidate(
        node=("var", 1),
        local_probe_mse=0.20,
        local_fit_mse=0.20,
        source="flat",
        generation_kind="flat",
        recursion_depth=0,
        confidence=1.0,
        valid_frac=1.0,
        trace=tuple(),
    )
    anchors = _select_binary_recursive_anchors(
        [flat_bad, outer_better_same_node, outer_unique, flat_other],
        ctx=ctx,
    )
    anchor_keys = [node_str(row.node) for row in anchors]
    anchor_sources = {node_str(row.node): row.source for row in anchors}
    assert node_str(("var", 0)) in anchor_keys
    assert anchor_sources[node_str(("var", 0))] == "outer_family:exp:flat_seed"
    assert node_str(("mul", ("var", 0), ("var", 1))) in anchor_keys


def test_solve_local_problem_memoizes_repeated_spec_solves():
    truth_expr = ("add", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    candidate_expr = ("add", ("var", 0), ("var", 2))
    problem = _make_problem(truth_expr, candidate_expr, nvars=3, seed=31)
    beam_state = _build_beam_state(problem, path=(1,))
    ctx = _SolverContext(
        parent_node=problem["candidate_expr"],
        hole_path=(1,),
        hole_sub=beam_state["sub"],
        max_depth=5,
        nvars=3,
        poly_degree=4,
        var_dims=None,
        pool_nodes=problem["pool_nodes"],
        pool_dims=problem["pool_dims"],
        seed_nodes=[beam_state["sub"]],
        local_score_mode="affine",
        enum_max_depth=2,
        enum_max_trees=128,
        max_subtree_depth=5,
        preview_topk=8,
        complexity_penalty=0.0,
        recursive_enable=True,
        recursive_max_depth=2,
        recursive_trigger_rel_mse=0.0,
        recursive_seed_cap=6,
        recursive_branch_topk=4,
        recursive_child_topk=2,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        min_valid_frac=0.10,
        min_confidence=0.10,
        allow_legacy_aux=True,
        legacy_aux_kwargs={},
        stats={},
    )
    local_problem = _LocalProblem(
        xf=beam_state["xf"],
        tf=beam_state["tf"],
        wf=beam_state.get("wf"),
        xp=beam_state["xp"],
        tp=beam_state["tp"],
        wp=beam_state.get("wp"),
        target_dim=beam_state.get("target_dim"),
        confidence=float(beam_state.get("confidence", 1.0) or 1.0),
        valid_frac=float(beam_state.get("valid_frac", 1.0) or 1.0),
        wrappers_left=2,
        recursion_level=0,
        trace=("memo-root",),
    )
    rows1, meta1 = _solve_local_problem(local_problem, ctx=ctx, include_legacy_aux=True)
    rows2, meta2 = _solve_local_problem(local_problem, ctx=ctx, include_legacy_aux=True)
    assert ctx.stats.get("memo_miss_count", 0) >= 1
    assert ctx.stats.get("memo_hit_count", 0) >= 1
    assert len(ctx.memo_table) >= 1
    assert [node_str(row.node) for row in rows1] == [node_str(row.node) for row in rows2]
    assert [tuple(row.trace) for row in rows1] == [tuple(row.trace) for row in rows2]
    assert meta1["flat"]["candidate_count_scored"] == meta2["flat"]["candidate_count_scored"]
