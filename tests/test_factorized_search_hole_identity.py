# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import pytest
import torch

from nestynet_sr.sr_search.factorized_search.engine.archive import ResidualBasinArchive
import nestynet_sr.sr_search.factorized_search.hole_search as hole_search_mod
from nestynet_sr.sr_search.factorized_search.hole_search import HoleFrontier, _score_hole_search_expr


def test_archive_resolve_elite_persists_old_elites_with_stable_handles():
    arch = ResidualBasinArchive()
    arch.update("kb", 1.0, ("var", 0), torch.tensor([1.0, 0.0]), {}, raw_mse=1.0)
    rec = arch.d["kb"]
    first_elite_id = rec.best_elite_id

    arch.update("kb", 0.5, ("sin", ("var", 0)), torch.tensor([0.0, 1.0]), {}, raw_mse=0.5)
    rec = arch.d["kb"]

    assert rec.residual_basin_key == "kb"
    assert rec.best_elite_id != first_elite_id

    resolved_rec, resolved_elite = arch.resolve_elite("kb", elite_id=first_elite_id)
    assert resolved_rec is rec
    assert resolved_elite is not None
    assert resolved_elite.elite_id == first_elite_id
    assert resolved_elite.expr == ("var", 0)


def test_archive_best_surface_keeps_residual_basin_and_elite_identity():
    arch = ResidualBasinArchive()
    arch.update("ka", 0.25, ("var", 0), torch.tensor([1.0, 0.0]), {}, raw_mse=0.25)

    best = arch.best(1)[0]

    assert best.residual_basin_key == "ka"
    assert best.best_elite_id == arch.d["ka"].best_elite_id


def test_hole_frontier_distinguishes_elites_within_one_residual_basin():
    frontier = HoleFrontier()
    slate = [{
        "path": [1],
        "target_mode": "identity",
        "beam_rank": 0,
        "path_gain": 0.5,
        "confidence": 0.8,
        "valid_frac": 0.9,
    }]

    frontier.ingest_opportunity_slate("kb", "expr_a", slate, 0, parent_elite_id="elite_a")
    frontier.ingest_opportunity_slate("kb", "expr_b", slate, 0, parent_elite_id="elite_b")

    assert len(frontier) == 2

    frontier.record_attempt(
        "kb",
        (1,),
        "identity",
        10,
        parent_elite_id="elite_a",
    )

    opp_a = next(opp for opp in frontier._entries.values() if opp.parent_elite_id == "elite_a")
    opp_b = next(opp for opp in frontier._entries.values() if opp.parent_elite_id == "elite_b")
    assert opp_a.attempts == 1
    assert opp_b.attempts == 0

    frontier.invalidate_parent("kb", parent_elite_id="elite_a")

    assert len(frontier) == 1
    remaining = next(iter(frontier._entries.values()))
    assert remaining.parent_elite_id == "elite_b"


def test_hole_frontier_records_source_on_ingest():
    frontier = HoleFrontier()
    slate = [{
        "path": [1],
        "target_mode": "identity",
        "beam_rank": 0,
        "path_gain": 0.5,
        "confidence": 0.8,
        "valid_frac": 0.9,
    }]

    frontier.ingest_opportunity_slate(
        "kb",
        "expr_a",
        slate,
        0,
        parent_elite_id="elite_a",
        parent_snapshot_id="snap_a",
        source="inverse_slate",
    )

    opp = next(iter(frontier._entries.values()))
    assert opp.source == "inverse_slate"
    assert opp.parent_snapshot_id == "snap_a"
    assert frontier.active_snapshot_ids() == {"snap_a"}


def test_hole_frontier_populates_inverse_slate_preview_solvability():
    frontier = HoleFrontier()
    slate = [{
        "path": [1],
        "target_mode": "identity",
        "beam_rank": 2,
        "path_gain": 0.5,
        "confidence": 0.8,
        "valid_frac": 0.9,
        "best_preview_probe_mse": 0.125,
        "candidate_count_observed": 7,
        "inverse_spec_recursion_depth": 3,
        "inverse_spec_generation_kind": "periodic_forward",
    }]

    frontier.ingest_opportunity_slate(
        "kb",
        "expr_a",
        slate,
        5,
        parent_elite_id="elite_a",
        parent_snapshot_id="snap_a",
        source="inverse_slate",
    )

    opp = next(iter(frontier._entries.values()))
    assert opp.created_at_iter == 5
    assert opp.preview_solvability == pytest.approx(0.125)
    assert opp.preview_candidate_count == 7
    assert opp.preview_recursive_depth == 3
    assert opp.preview_periodic_fired is True


def test_hole_frontier_keeps_distinct_spec_siblings_and_updates_exact_one():
    frontier = HoleFrontier()
    base_row = {
        "path": [1],
        "target_mode": "identity",
        "beam_rank": 0,
        "path_gain": 0.5,
        "confidence": 0.8,
        "valid_frac": 0.9,
    }
    frontier.ingest_opportunity_slate(
        "kb",
        "expr_a",
        [{**base_row, "branch_id": "left", "continuation_key": ["wrap:left"], "trace": ["left"]}],
        0,
        parent_elite_id="elite_a",
        parent_snapshot_id="snap_a",
    )
    frontier.ingest_opportunity_slate(
        "kb",
        "expr_a",
        [{**base_row, "branch_id": "right", "continuation_key": ["wrap:right"], "trace": ["right"]}],
        0,
        parent_elite_id="elite_a",
        parent_snapshot_id="snap_a",
    )

    assert len(frontier) == 2
    opp_left = next(opp for opp in frontier._entries.values() if opp.branch_id == "left")
    opp_right = next(opp for opp in frontier._entries.values() if opp.branch_id == "right")

    frontier.record_attempt(current_iter=7, opportunity=opp_left, child_eff_mse=0.25)
    frontier.record_exact_outcome(
        opp_left,
        current_iter=7,
        exact_eff_mse=0.125,
        shortlist_eff_mse=0.25,
        reward=0.5,
        parent_eff_mse=1.0,
        accepted=True,
        status="accepted",
    )

    assert opp_left.attempts == 1
    assert opp_left.best_shortlist_eff_mse == pytest.approx(0.25)
    assert opp_left.best_exact_eff_mse == pytest.approx(0.125)
    assert opp_right.attempts == 0
    assert opp_right.best_shortlist_eff_mse is None
    assert opp_right.best_exact_eff_mse is None


def test_hole_search_shortlist_scorer_does_not_call_generic_score_fn():
    x_fit = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    y_fit = 1.0 + 2.0 * x_fit
    x_probe = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float64)
    y_probe = 1.0 + 2.0 * x_probe

    def _boom(*args, **kwargs):
        raise AssertionError("generic score_expr_fn should not be called from shortlist scorer")

    scored = _score_hole_search_expr(
        ("var", 0),
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        poly_degree=4,
        mapping_kind_hint="affine",
        score_expr_fn=_boom,
    )

    assert scored is not None
    assert scored["mapping"]["kind"] == "poly"
    assert float(scored["raw_mse"]) == pytest.approx(0.0)


def test_abstract_frontier_from_parent_handles_empty_preview_rows(monkeypatch):
    def _mock_transport(*args, **kwargs):
        return {}

    def _mock_beam_states(*args, **kwargs):
        return [{
            "path": (1,),
            "target_mode": "identity",
            "confidence": 0.8,
            "valid_frac": 0.9,
            "path_gain": 0.5,
        }]

    def _mock_solver(*args, **kwargs):
        return {"rows": [], "solver_meta": {}}

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._estimate_inverse_action_transport",
        _mock_transport,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._inverse_action_path_mode_beam_states",
        _mock_beam_states,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_inverse_spec_preview_rows",
        _mock_solver,
    )

    frontier = HoleFrontier()
    x = torch.randn(8, 2, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    added = hole_search_mod.abstract_frontier_from_parent(
        frontier,
        parent_key="kb",
        parent_elite_id="elite_a",
        parent_expr=("add", ("mul", ("var", 0), ("var", 1)), ("var", 1)),
        parent_mapping={"kind": "identity"},
        parent_eff_mse=1.0,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=5,
        nvars=2,
        poly_degree=4,
        current_iter=0,
    )

    assert added == 1
    opp = next(iter(frontier._entries.values()))
    assert opp.preview_solvability is None
    assert opp.preview_candidate_count == 0


def test_abstract_frontier_from_parent_forwards_regime_metadata(monkeypatch):
    expected_regime_metadata = {
        "dataset_ids": ["d0", "d1"],
        "local_constants_by_experiment": {
            "d0": {"local_leaf": 1.0},
            "d1": {"local_leaf": 2.0},
        },
    }

    def _mock_transport(*args, **kwargs):
        return {}

    def _mock_beam_states(*args, **kwargs):
        return [{
            "path": (1,),
            "target_mode": "identity",
            "confidence": 0.8,
            "valid_frac": 0.9,
            "path_gain": 0.5,
        }]

    captured = {}

    def _mock_solver(*args, **kwargs):
        captured["regime_metadata"] = kwargs.get("regime_metadata", None)
        return {"rows": [], "solver_meta": {}}

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._estimate_inverse_action_transport",
        _mock_transport,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.inverse_action._inverse_action_path_mode_beam_states",
        _mock_beam_states,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.hole_search.solve_inverse_spec_preview_rows",
        _mock_solver,
    )

    frontier = HoleFrontier()
    x = torch.randn(8, 2, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    added = hole_search_mod.abstract_frontier_from_parent(
        frontier,
        parent_key="kb",
        parent_elite_id="elite_a",
        parent_expr=("add", ("mul", ("var", 0), ("var", 1)), ("var", 1)),
        parent_mapping={"kind": "identity"},
        parent_eff_mse=1.0,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        pool_nodes=[],
        pool_phi_fit=x[:, :0],
        pool_phi_probe=x[:, :0],
        pool_dims=[],
        max_depth=5,
        nvars=2,
        poly_degree=4,
        current_iter=0,
        regime_metadata=expected_regime_metadata,
    )

    assert added == 1
    assert captured["regime_metadata"] == expected_regime_metadata
