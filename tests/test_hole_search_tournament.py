# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for the risk-seeking hole search tournament.

Verifies that:
1. select_n_executable returns multiple eligible opportunities
2. run_hole_tournament cheap-previews candidates, updates losers' preview
   scores in-place, and returns elite_k winners sorted by preview MSE
"""

import random

import pytest

from nestynet_sr.sr_search.factorized_search.hole_search import (
    export_hole_opportunity_rows,
    HoleFrontier,
    HoleOpportunity,
)


# ---------------------------------------------------------------------------
# select_n_executable
# ---------------------------------------------------------------------------

def _make_opp(parent_key, path, confidence=0.8, valid_frac=0.9, path_gain=0.5,
              preview_solvability=None, parent_elite_id="e0"):
    return HoleOpportunity(
        parent_key=parent_key,
        parent_expr_str=f"expr_{parent_key}",
        path=tuple(path),
        target_mode="identity",
        beam_rank=0,
        parent_elite_id=parent_elite_id,
        path_gain=path_gain,
        confidence=confidence,
        valid_frac=valid_frac,
        preview_solvability=preview_solvability,
    )


def test_select_n_executable_returns_multiple():
    frontier = HoleFrontier(cooldown_iters=100)
    for i in range(5):
        opp = _make_opp(f"k{i}", [1, i], path_gain=0.1 * (i + 1))
        frontier._entries[opp.frontier_key] = opp

    rng = random.Random(42)
    result = frontier.select_n_executable(
        current_iter=0,
        rng=rng,
        is_executable_fn=lambda opp: True,
        n=3,
    )
    assert len(result) == 3
    # Should be sorted by score descending (highest path_gain first)
    scores = [frontier._score(o) for o in result]
    assert scores == sorted(scores, reverse=True)


def test_select_n_executable_respects_cooldown():
    frontier = HoleFrontier(cooldown_iters=100)
    opp_ok = _make_opp("ok", [1])
    opp_cool = _make_opp("cool", [2])
    opp_cool.cooldown_until = 999
    frontier._entries[opp_ok.frontier_key] = opp_ok
    frontier._entries[opp_cool.frontier_key] = opp_cool

    rng = random.Random(0)
    result = frontier.select_n_executable(0, rng, lambda o: True, n=10)
    assert len(result) == 1
    assert result[0].parent_key == "ok"


def test_select_n_executable_filters_non_executable():
    frontier = HoleFrontier()
    for i in range(4):
        opp = _make_opp(f"k{i}", [1, i])
        frontier._entries[opp.frontier_key] = opp

    rng = random.Random(0)
    result = frontier.select_n_executable(
        0, rng,
        is_executable_fn=lambda o: o.parent_key in ("k0", "k2"),
        n=10,
    )
    assert len(result) == 2
    keys = {o.parent_key for o in result}
    assert keys == {"k0", "k2"}


def test_export_hole_opportunity_rows_emits_shared_rows():
    frontier = HoleFrontier()
    opp = _make_opp("parentA", [1, 2], preview_solvability=0.2)
    opp.spec_kind = "path_hole"
    opp.source = "archive_mine"
    opp.preview_candidate_count = 4
    opp.predicted_value = 0.7
    opp.predicted_cost = 1.3
    opp.witness_value_loss = 0.05
    opp.witness_grad_loss = 0.02
    opp.witness_energy_total = 0.07
    frontier._entries[opp.frontier_key] = opp

    rows = export_hole_opportunity_rows(
        frontier,
        current_iter=0,
        decision_id="hole_decision",
        decision_context_id="hole_ctx",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["route_source"] == "hole"
    assert row["decision_id"] == "hole_decision"
    assert row["decision_context_id"] == "hole_ctx"
    assert row["action"] == "hole_search"
    assert row["method_name"] == "archive_mine"
    assert row["subroute"] == "path_hole"
    assert row["cost_estimate"] == pytest.approx(1.3)
    assert row["preview_candidate_count_total"] == 4
    assert row["opportunity_route_valid_hole"] == pytest.approx(1.0)
    assert row["witness_value_loss"] == pytest.approx(0.05)
    assert row["witness_grad_loss"] == pytest.approx(0.02)
    assert row["witness_energy_total"] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# run_hole_tournament
# ---------------------------------------------------------------------------

def test_tournament_returns_elite_k_sorted_by_preview(monkeypatch):
    """Mock the inverse solver so we can test the tournament selection logic."""
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    # Create 4 opportunities with varying quality
    opps = [_make_opp(f"k{i}", [1, i]) for i in range(4)]

    # Mock parent_resolver to return a fake parent
    fake_parent = ("add", ("var", 0), ("var", 1))
    fake_mapping = {"kind": "poly", "coeffs": [1.0, 0.0], "mu": 0.0, "std": 1.0}

    def mock_resolver(opp):
        return {"parent_node": fake_parent, "parent_mapping": fake_mapping}

    # Mock the inverse action functions to return minimal beam states
    def mock_transport(*a, **kw):
        return {"paths": {}}

    def mock_beam_states(*a, **kw):
        return [{"path": kw.get("all_paths", [[1]])[0],
                 "target_mode": "identity",
                 "confidence": 0.8,
                 "valid_frac": 0.9,
                 "path_gain": 0.5}]

    # Mock the solver to return different MSEs per opportunity
    # Use path to determine MSE: path (1,0) -> 0.001, (1,1) -> 0.1, etc.
    solver_mses = {0: 0.001, 1: 0.1, 2: 0.05, 3: 0.5}

    def mock_solver(*a, **kw):
        slate_id = kw.get("slate_id", "")
        # Extract the path index from slate_id like "tournament:k2:1/2"
        for idx, mse in solver_mses.items():
            if f"k{idx}" in slate_id:
                return {
                    "rows": [{"local_probe_mse": mse, "expr": ("var", 0)}],
                    "solver_meta": {},
                }
        return {"rows": [], "solver_meta": {}}

    # The tournament does a lazy `from .inverse_action import ...` so patching
    # the source module works for those.  But solve_inverse_spec_preview_rows
    # is imported at module level in hole_search.py, so we must patch it on
    # the hole_search module directly.
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._estimate_inverse_action_transport",
        mock_transport,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._inverse_action_path_mode_beam_states",
        mock_beam_states,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_inverse_spec_preview_rows",
        mock_solver,
    )

    import torch
    N = 64
    x = torch.randn(N, 2, dtype=torch.float64)
    y = torch.randn(N, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        opps,
        parent_resolver=mock_resolver,
        x_fit=x, y_fit=y,
        x_probe=x, y_probe=y,
        pool_nodes=[], pool_phi_fit=x[:, :0], pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=5, nvars=2, poly_degree=4,
        preview_budget=32,
        preview_topk=4,
        elite_k=2,
    )

    # Should return 2 elites
    assert len(elites) == 2
    # Best elite should be k0 (MSE 0.001), second should be k2 (MSE 0.05)
    assert elites[0][0].parent_key == "k0"
    assert elites[0][1] == pytest.approx(0.001)
    assert elites[1][0].parent_key == "k2"
    assert elites[1][1] == pytest.approx(0.05)

    # Losers should have preview_solvability updated in-place
    loser_k1 = [o for o in opps if o.parent_key == "k1"][0]
    loser_k3 = [o for o in opps if o.parent_key == "k3"][0]
    assert loser_k1.preview_solvability == pytest.approx(0.1)
    assert loser_k3.preview_solvability == pytest.approx(0.5)


def test_tournament_updates_existing_preview_only_if_better(monkeypatch):
    """If a hole already has a better preview score, don't overwrite it."""
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    opp = _make_opp("k0", [1, 0], preview_solvability=0.001)

    def mock_resolver(o):
        return {"parent_node": ("var", 0), "parent_mapping": {"kind": "poly", "coeffs": [1.0], "mu": 0.0, "std": 1.0}}

    def mock_transport(*a, **kw):
        return {}

    def mock_beam_states(*a, **kw):
        return [{"path": (1, 0), "target_mode": "identity",
                 "confidence": 0.8, "valid_frac": 0.9, "path_gain": 0.5}]

    # Solver returns worse MSE than existing
    def mock_solver(*a, **kw):
        return {"rows": [{"local_probe_mse": 0.1, "expr": ("var", 0)}], "solver_meta": {}}

    # The tournament does a lazy `from .inverse_action import ...` so patching
    # the source module works for those.  But solve_inverse_spec_preview_rows
    # is imported at module level in hole_search.py, so we must patch it on
    # the hole_search module directly.
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._estimate_inverse_action_transport",
        mock_transport,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._inverse_action_path_mode_beam_states",
        mock_beam_states,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_inverse_spec_preview_rows",
        mock_solver,
    )

    import torch
    N = 32
    x = torch.randn(N, 2, dtype=torch.float64)
    y = torch.randn(N, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        [opp],
        parent_resolver=mock_resolver,
        x_fit=x, y_fit=y, x_probe=x, y_probe=y,
        pool_nodes=[], pool_phi_fit=x[:, :0], pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=5, nvars=2, poly_degree=4,
        preview_budget=16, preview_topk=2, elite_k=1,
    )

    # Should return the opp as elite (only candidate)
    assert len(elites) == 1
    # preview_solvability should NOT be overwritten with worse value
    assert opp.preview_solvability == pytest.approx(0.001)


def test_tournament_solver_market_supports_local_problem_followups(monkeypatch):
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    opp = _make_opp("follow", [1, 0], preview_solvability=None)
    opp.spec_kind = "local_problem"
    opp.spec_payload = {"problem": {"xf": None}}

    def mock_resolver(o):
        return {
            "parent_node": ("var", 0),
            "parent_mapping": {"kind": "poly", "coeffs": [1.0], "mu": 0.0, "std": 1.0},
        }

    def mock_local_solver(*a, **kw):
        assert kw["path"] == (1, 0)
        return {
            "rows": [{"local_probe_mse": 0.02, "expr": ("var", 0), "child_key": "x0"}],
            "solver_meta": {"status": "ok"},
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_problem_spec_preview_rows",
        mock_local_solver,
    )

    import torch
    x = torch.randn(24, 1, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        [opp],
        parent_resolver=mock_resolver,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        preview_budget=16,
        preview_topk=2,
        elite_k=1,
        solver_market_enable=True,
        solver_market_preview_topk=2,
        solver_market_exact_topk=1,
    )

    assert len(elites) == 1
    assert elites[0][0].parent_key == "follow"
    assert elites[0][1] == pytest.approx(0.02)
    assert opp.preview_solvability == pytest.approx(0.02)


def test_tournament_solver_market_can_prefer_recursive_local_sr(monkeypatch):
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    opp = _make_opp("follow_sr", [1, 0], preview_solvability=None)
    opp.spec_kind = "local_problem"
    opp.spec_payload = {"problem": {"xf": None}}

    def mock_resolver(o):
        return {
            "parent_node": ("var", 0),
            "parent_mapping": {"kind": "poly", "coeffs": [1.0], "mu": 0.0, "std": 1.0},
        }

    def mock_local_solver(*a, **kw):
        return {
            "rows": [{"local_probe_mse": 0.20, "expr": ("var", 0), "child_key": "legacy"}],
            "solver_meta": {"status": "ok"},
        }

    def mock_recursive_sr_solver(*a, **kw):
        assert kw["preview_topk"] == 3
        assert kw["exact_budget"] == 2
        assert kw["witness_loss_enable"] is True
        assert kw["witness_grad_weight"] == pytest.approx(0.5)
        return {
            "rows": [{"local_probe_mse": 0.02, "expr": ("var", 0), "child_key": "recursive"}],
            "solver_meta": {"status": "ok"},
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_problem_spec_preview_rows",
        mock_local_solver,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_recursive_sr_preview_rows",
        mock_recursive_sr_solver,
    )

    import torch
    x = torch.randn(24, 1, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        [opp],
        parent_resolver=mock_resolver,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        preview_budget=16,
        preview_topk=2,
        elite_k=1,
        solver_market_enable=True,
        solver_market_preview_topk=3,
        solver_market_exact_topk=2,
        inverse_spec_recursive_sr_enable=True,
        inverse_spec_recursive_sr_preview_topk=3,
        inverse_spec_recursive_sr_exact_budget=2,
        inverse_spec_witness_loss_enable=True,
        inverse_spec_witness_grad_weight=0.5,
    )

    assert len(elites) == 1
    assert elites[0][0].parent_key == "follow_sr"
    assert elites[0][1] == pytest.approx(0.02)
    assert opp.preview_solvability == pytest.approx(0.02)


def test_tournament_solver_market_can_prefer_tangent_edit(monkeypatch):
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    opp = _make_opp("follow_tangent", [1, 0], preview_solvability=None)
    opp.spec_kind = "local_problem"
    opp.spec_payload = {"problem": {"xf": None}}

    def mock_resolver(o):
        return {
            "parent_node": ("var", 0),
            "parent_mapping": {"kind": "poly", "coeffs": [1.0], "mu": 0.0, "std": 1.0},
        }

    def mock_local_solver(*a, **kw):
        return {
            "rows": [{"local_probe_mse": 0.20, "expr": ("var", 0), "child_key": "legacy"}],
            "solver_meta": {"status": "ok"},
        }

    def mock_tangent_solver(*a, **kw):
        assert kw["preview_topk"] == 5
        return {
            "rows": [{"local_probe_mse": 0.02, "expr": ("sin", ("var", 0)), "child_key": "tangent"}],
            "solver_meta": {"status": "ok"},
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_problem_spec_preview_rows",
        mock_local_solver,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_tangent_edit_preview_rows",
        mock_tangent_solver,
    )

    import torch
    x = torch.randn(24, 1, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        [opp],
        parent_resolver=mock_resolver,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        preview_budget=16,
        preview_topk=2,
        elite_k=1,
        solver_market_enable=True,
        solver_market_preview_topk=4,
        solver_market_exact_topk=2,
        inverse_spec_tangent_edit_enable=True,
        inverse_spec_tangent_edit_topk=5,
    )

    assert len(elites) == 1
    assert elites[0][0].parent_key == "follow_tangent"
    assert elites[0][1] == pytest.approx(0.02)
    assert opp.preview_solvability == pytest.approx(0.02)


def test_tournament_solver_market_can_prefer_coordinate_lift(monkeypatch):
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    opp = _make_opp("follow_coord", [1, 0], preview_solvability=None)
    opp.spec_kind = "local_problem"
    opp.spec_payload = {"problem": {"xf": None}}

    def mock_resolver(o):
        return {
            "parent_node": ("var", 0),
            "parent_mapping": {"kind": "poly", "coeffs": [1.0], "mu": 0.0, "std": 1.0},
        }

    def mock_local_solver(*a, **kw):
        return {
            "rows": [{"local_probe_mse": 0.20, "expr": ("var", 0), "child_key": "legacy"}],
            "solver_meta": {"status": "ok"},
        }

    def mock_coordinate_solver(*a, **kw):
        assert kw["coordinate_topk"] == 3
        assert kw["coordinate_mode"] == "both"
        return {
            "rows": [{"local_probe_mse": 0.015, "expr": ("add", ("var", 0), ("var", 1)), "child_key": "coord"}],
            "solver_meta": {"status": "ok"},
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_problem_spec_preview_rows",
        mock_local_solver,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_coordinate_lift_preview_rows",
        mock_coordinate_solver,
    )

    import torch
    x = torch.randn(24, 2, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        [opp],
        parent_resolver=mock_resolver,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=2,
        poly_degree=2,
        preview_budget=16,
        preview_topk=2,
        elite_k=1,
        solver_market_enable=True,
        solver_market_preview_topk=3,
        solver_market_exact_topk=2,
        inverse_spec_coordinate_lift_enable=True,
        inverse_spec_coordinate_lift_topk=3,
        inverse_spec_coordinate_lift_mode="both",
    )

    assert len(elites) == 1
    assert elites[0][0].parent_key == "follow_coord"
    assert elites[0][1] == pytest.approx(0.015)
    assert opp.preview_solvability == pytest.approx(0.015)


def test_tournament_solver_market_can_prefer_constant_lift_route(monkeypatch):
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    opp = _make_opp("follow_constlift", [1, 0], preview_solvability=None)
    opp.spec_kind = "local_problem"
    opp.spec_payload = {"problem": {"xf": None}, "subproblem_spec": {"metadata": {"constant_lift_task": {}}}}

    def mock_resolver(o):
        return {
            "parent_node": ("add", ("var", 0), ("const", 0.0)),
            "parent_mapping": {"kind": "poly", "coeffs": [1.0], "mu": 0.0, "std": 1.0},
        }

    def mock_local_solver(*a, **kw):
        return {
            "rows": [{"local_probe_mse": 0.20, "expr": ("var", 0), "child_key": "legacy"}],
            "solver_meta": {"status": "ok"},
        }

    def mock_constant_lift_solver(*a, **kw):
        assert kw["constant_lift_topk"] == 2
        assert kw["preview_topk"] == 2
        return {
            "rows": [{"local_probe_mse": 0.005, "expr": ("add", ("var", 0), ("var", 1)), "child_key": "constlift"}],
            "solver_meta": {"status": "ok"},
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_problem_spec_preview_rows",
        mock_local_solver,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_constant_lift_preview_rows",
        mock_constant_lift_solver,
    )

    import torch
    x = torch.randn(24, 2, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        [opp],
        parent_resolver=mock_resolver,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=2,
        poly_degree=2,
        preview_budget=16,
        preview_topk=2,
        elite_k=1,
        solver_market_enable=True,
        solver_market_preview_topk=3,
        solver_market_exact_topk=2,
        inverse_spec_constant_lift_route_enable=True,
        inverse_spec_constant_lift_route_topk=2,
    )

    assert len(elites) == 1
    assert elites[0][0].parent_key == "follow_constlift"
    assert elites[0][1] == pytest.approx(0.005)
    assert opp.preview_solvability == pytest.approx(0.005)


def test_tournament_solver_market_can_prefer_soft_edit(monkeypatch):
    import nestynet_sr.sr_search.factorized_search.hole_search as hs_mod

    opp = _make_opp("follow_soft", [1, 0], preview_solvability=None)
    opp.spec_kind = "local_problem"
    opp.spec_payload = {"problem": {"xf": None}}

    def mock_resolver(o):
        return {
            "parent_node": ("var", 0),
            "parent_mapping": {"kind": "poly", "coeffs": [1.0], "mu": 0.0, "std": 1.0},
        }

    def mock_local_solver(*a, **kw):
        return {
            "rows": [{"local_probe_mse": 0.20, "expr": ("var", 0), "child_key": "legacy"}],
            "solver_meta": {"status": "ok"},
        }

    def mock_soft_solver(*a, **kw):
        assert kw["soft_edit_steps"] == 48
        assert kw["soft_edit_l1"] == pytest.approx(2.0e-3)
        return {
            "rows": [{"local_probe_mse": 0.01, "expr": ("sin", ("var", 0)), "child_key": "soft"}],
            "solver_meta": {"status": "ok"},
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_problem_spec_preview_rows",
        mock_local_solver,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_local_soft_edit_preview_rows",
        mock_soft_solver,
    )

    import torch
    x = torch.randn(24, 1, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64)

    elites = hs_mod.run_hole_tournament(
        [opp],
        parent_resolver=mock_resolver,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=4,
        nvars=1,
        poly_degree=2,
        preview_budget=16,
        preview_topk=2,
        elite_k=1,
        solver_market_enable=True,
        solver_market_preview_topk=4,
        solver_market_exact_topk=2,
        inverse_spec_soft_edit_enable=True,
        inverse_spec_soft_edit_steps=48,
        inverse_spec_soft_edit_l1=2.0e-3,
    )

    assert len(elites) == 1
    assert elites[0][0].parent_key == "follow_soft"
    assert elites[0][1] == pytest.approx(0.01)
    assert opp.preview_solvability == pytest.approx(0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
