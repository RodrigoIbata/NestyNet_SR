# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import random

import torch

from nestynet_sr.sr_search.factorized_search import inverse_action as inverse_action_mod
from nestynet_sr.sr_search.factorized_search import repair_critic as repair_critic_mod


def test_inverse_action_opportunity_row_preserves_best_preview_witness_fields():
    row = inverse_action_mod._serialize_inverse_action_opportunity_row(
        parent_node=("var", 0),
        decision_id="repair_decision",
        beam_rank=0,
        beam_state={"path": (1,), "target_mode": "identity"},
        beam_rows=[
            {
                "dedup_kept": True,
                "child_key": "x0",
                "child_expr": "x0",
                "local_probe_mse": 0.2,
                "local_fit_mse": 0.1,
                "witness_value_loss": 0.05,
                "witness_grad_loss": 0.02,
                "witness_energy_total": 0.07,
            }
        ],
        budget_remaining=1,
        local_limit=2,
    )

    assert row["witness_value_loss"] == 0.05
    assert row["witness_grad_loss"] == 0.02
    assert row["witness_energy_total"] == 0.07


def test_inverse_allocator_rows_preserve_explicit_witness_fields_for_observed_child():
    rows = inverse_action_mod._build_inverse_allocator_opportunity_rows(
        parent_node=("var", 0),
        decision_id="repair_decision",
        beam_states=[{"path": (1,), "target_mode": "identity"}],
        candidate_rows_by_beam={
            0: [
                {
                    "dedup_kept": True,
                    "child_key": "x0",
                    "child_expr": "x0",
                    "local_rank": 0,
                    "witness_value_loss": 0.05,
                    "witness_grad_loss": 0.02,
                    "witness_energy_total": 0.07,
                }
            ]
        },
        local_limit=2,
        selected_keys={"x0"},
        selected_counts_by_beam={0: 1},
        observed_by_child_key={"x0": {"eff_mse": 0.1, "raw_mse": 0.11}},
        parent_eff_mse=0.5,
    )

    assert len(rows) == 1
    assert rows[0]["evidence_level"] == "exact_known"
    assert rows[0]["witness_value_loss"] == 0.05
    assert rows[0]["witness_grad_loss"] == 0.02
    assert rows[0]["witness_energy_total"] == 0.07


def test_inverse_action_beam_global_rerank_and_slate_logging(monkeypatch):
    expr = ("add", ("add", ("var", 0), ("var", 1)), ("var", 2))
    x_fit = torch.zeros((4, 3), dtype=torch.float64)
    y_fit = torch.zeros((4, 1), dtype=torch.float64)
    x_probe = torch.zeros((4, 3), dtype=torch.float64)
    y_probe = torch.zeros((4, 1), dtype=torch.float64)

    monkeypatch.setattr(
        inverse_action_mod,
        "_deterministic_row_subset",
        lambda cap, *rows: rows,
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_estimate_inverse_action_transport",
        lambda *args, **kwargs: {
            "ranked_paths": [
                (1.0, 0.8, 0, 1, -1, (1,)),
                (0.9, 0.7, 0, 1, -1, (2,)),
            ],
            "path_transport_rel": {(1,): 0.8, (2,): 0.7},
        },
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_action_path_mode_beam_states",
        lambda **kwargs: [
            {
                "path": (1,),
                "sub": ("var", 0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [0],
                "path_gain": 1.20,
                "path_gain_pre_cut": 1.30,
                "rel_gain": 0.40,
                "transport_rel": 0.80,
                "lin_rel": 0.20,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 12.0,
                "target_mode": "identity",
            },
            {
                "path": (2,),
                "sub": ("var", 1),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [1],
                "path_gain": 0.90,
                "path_gain_pre_cut": 0.95,
                "rel_gain": 0.30,
                "transport_rel": 0.70,
                "lin_rel": 0.15,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 10.0,
                "target_mode": "full",
            },
        ],
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_collect_local_repair_candidates",
        lambda **kwargs: [
            (
                "add",
                ("const", float(tuple(kwargs["path"])[0])),
                ("const", float(idx)),
            )
            for idx in (0, 1)
        ],
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_rank_local_repair_candidates",
        lambda cand_subtrees, **kwargs: [
            (float(idx), float(idx), cand)
            for idx, cand in enumerate(cand_subtrees)
        ],
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_transport_aligned_local_rows",
        lambda local_rows, **kwargs: list(local_rows),
    )

    def _fake_score_expr(cand, **kwargs):
        child_key = inverse_action_mod.node_str(cand)
        if ("x0" in child_key) and ("2" in child_key):
            eff_mse = 0.10
        elif ("x1" in child_key) and ("2" in child_key):
            eff_mse = 0.28
        elif "1" in child_key:
            eff_mse = 0.35
        elif "0" in child_key:
            eff_mse = 0.13
        else:
            eff_mse = 0.40
        return {
            "expr": cand,
            "raw_mse": eff_mse + 0.01,
            "eff_mse": eff_mse,
            "mapping": {"kind": "poly"},
        }

    monkeypatch.setattr(
        inverse_action_mod,
        "_score_inverse_action_expr",
        _fake_score_expr,
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_score_inverse_action_parent",
        lambda **kwargs: {"raw_mse": 1.0, "eff_mse": 1.2},
    )

    out_expr, meta = inverse_action_mod.run_inverse_steering_action(
        expr,
        {"kind": "poly"},
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes=[("var", 0), ("var", 1), ("var", 2)],
        pool_phi_fit=torch.zeros((4, 3), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 3), dtype=torch.float64),
        pool_dims=None,
        rng=random.Random(0),
        max_depth=4,
        nvars=3,
        poly_degree=2,
        return_meta=True,
        score_expr_fn=lambda *args, **kwargs: None,
    )

    assert out_expr is not None
    assert meta["status"] == "ok"
    assert meta["selected_path"] == [2]
    assert meta["selected_target_mode"] == "full"
    assert meta["inverse_path_mode_beam_count"] == 2
    assert len(meta["inverse_path_mode_beam"]) == 2
    beam_row = meta["inverse_path_mode_beam"][0]
    assert "weighted_rel_gain" in beam_row
    assert "weighted_rel_gain_pre_cut" in beam_row
    assert "cut_factor" in beam_row
    assert "valid_frac" in beam_row
    assert "confidence" in beam_row
    assert "branch_support" in beam_row
    assert "family_scale" in beam_row
    assert "mode_rows" in beam_row
    assert meta["inverse_exact_score_budget"] == 4
    assert meta["inverse_exact_support_floor_beams"] == 2
    assert meta["inverse_exact_support_floor_selected"] == 2
    assert meta["inverse_exact_global_allocated"] == 2
    assert meta["inverse_exact_score_observed_count"] == 4
    assert meta["inverse_repair_slate_id"]
    assert meta["inverse_repair_slate_count"] == 4
    assert len(meta["inverse_repair_slate"]) == 4
    assert meta["repair_opportunity_slate_id"]
    assert meta["repair_opportunity_slate_count"] == 2
    assert len(meta["repair_opportunity_slate"]) == 2
    assert meta["inverse_repair_slate"][0]["slate_id"] == meta["inverse_repair_slate_id"]
    assert meta["inverse_repair_slate"][0]["tuple_provenance"] == "beam_local_repair"
    assert meta["inverse_repair_slate"][0]["exact_child_score_observed"] is True
    assert meta["inverse_repair_slate"][0]["path"] == [2]
    assert meta["inverse_repair_slate"][0]["child_eff_mse"] == 0.10
    assert "local_probe_mse" in meta["inverse_repair_slate"][0]
    assert "local_fit_mse" in meta["inverse_repair_slate"][0]
    assert "local_fit_probe_gap" in meta["inverse_repair_slate"][0]
    assert "target_mapping_kind" in meta["inverse_repair_slate"][0]
    assert "local_mapping_kind" in meta["inverse_repair_slate"][0]
    assert "local_mapping_nparams" in meta["inverse_repair_slate"][0]
    assert "candidate_subtree_size" in meta["inverse_repair_slate"][0]
    assert "candidate_child_size" in meta["inverse_repair_slate"][0]
    assert "candidate_root_op" in meta["inverse_repair_slate"][0]
    assert meta["repair_opportunity_slate"][0]["route_source"] == "repair"
    assert meta["repair_opportunity_slate"][0]["opportunity_type"] == "repair_beam"
    assert meta["repair_opportunity_slate"][0]["decision_id"] == meta["repair_opportunity_slate_id"]
    assert meta["repair_opportunity_slate"][0]["budget_exact_spent"] == 0
    assert meta["repair_opportunity_slate"][0]["budget_remaining"] == meta["inverse_exact_score_budget"]
    assert meta["repair_opportunity_slate"][0]["candidate_count_observed"] == 2
    assert meta["repair_opportunity_slate"][0]["candidate_count_unique"] == 2
    assert "path_gain" in meta["repair_opportunity_slate"][0]
    assert "best_preview_probe_mse" in meta["repair_opportunity_slate"][0]
    assert meta["estimated_one_hole_rel_improve_eff"] > 0.0


def test_inverse_action_beam_dedupes_children_and_keeps_fixed_exact_budget(monkeypatch):
    expr = ("add", ("const", 0.0), ("const", 0.0))
    x_fit = torch.zeros((4, 1), dtype=torch.float64)
    y_fit = torch.zeros((4, 1), dtype=torch.float64)
    x_probe = torch.zeros((4, 1), dtype=torch.float64)
    y_probe = torch.zeros((4, 1), dtype=torch.float64)
    score_calls: list[str] = []

    monkeypatch.setattr(
        inverse_action_mod,
        "_deterministic_row_subset",
        lambda cap, *rows: rows,
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_estimate_inverse_action_transport",
        lambda *args, **kwargs: {
            "ranked_paths": [
                (1.0, 0.8, 0, 1, -1, (1,)),
                (0.9, 0.7, 0, 1, -1, (2,)),
                (0.8, 0.6, 0, 1, -1, (3,)),
            ],
            "path_transport_rel": {(1,): 0.8, (2,): 0.7, (3,): 0.6},
        },
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_action_path_mode_beam_states",
        lambda **kwargs: [
            {
                "path": (1,),
                "sub": ("const", 0.0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [0],
                "path_gain": 1.20,
                "path_gain_pre_cut": 1.30,
                "rel_gain": 0.40,
                "transport_rel": 0.80,
                "lin_rel": 0.20,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 12.0,
                "target_mode": "identity",
            },
            {
                "path": (2,),
                "sub": ("const", 0.0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [1],
                "path_gain": 1.10,
                "path_gain_pre_cut": 1.15,
                "rel_gain": 0.35,
                "transport_rel": 0.70,
                "lin_rel": 0.18,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 11.0,
                "target_mode": "identity",
            },
            {
                "path": (3,),
                "sub": ("const", 0.0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [2],
                "path_gain": 1.00,
                "path_gain_pre_cut": 1.05,
                "rel_gain": 0.30,
                "transport_rel": 0.60,
                "lin_rel": 0.15,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 10.0,
                "target_mode": "full",
            },
        ],
    )

    def _collect(**kwargs):
        path = tuple(kwargs["path"])
        if path == (1,):
            return [("const", 9.0), ("const", 1.0)]
        if path == (2,):
            return [("const", 9.0), ("const", 2.0)]
        return [("const", 3.0), ("const", 4.0)]

    monkeypatch.setattr(inverse_action_mod, "_inverse_collect_local_repair_candidates", _collect)
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_rank_local_repair_candidates",
        lambda cand_subtrees, **kwargs: [(float(idx), float(idx), cand) for idx, cand in enumerate(cand_subtrees)],
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_transport_aligned_local_rows",
        lambda local_rows, **kwargs: list(local_rows),
    )

    def _fake_score_expr(cand, **kwargs):
        child_key = inverse_action_mod.node_str(cand)
        score_calls.append(child_key)
        eff_map = {
            "9": 0.40,
            "1": 0.25,
            "2": 0.20,
            "3": 0.15,
        }
        hit = next((v for k, v in eff_map.items() if k in child_key), 0.50)
        return {
            "expr": cand,
            "raw_mse": hit + 0.01,
            "eff_mse": hit,
            "mapping": {"kind": "poly"},
        }

    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_expr", _fake_score_expr)
    monkeypatch.setattr(
        inverse_action_mod,
        "_score_inverse_action_parent",
        lambda **kwargs: {"raw_mse": 1.0, "eff_mse": 1.0},
    )

    out_expr, meta = inverse_action_mod.run_inverse_steering_action(
        expr,
        {"kind": "poly"},
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes=[("var", 0), ("var", 1), ("var", 2)],
        pool_phi_fit=torch.zeros((4, 3), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 3), dtype=torch.float64),
        pool_dims=None,
        rng=random.Random(0),
        max_depth=4,
        nvars=3,
        poly_degree=2,
        topk_terms=2,
        return_meta=True,
        score_expr_fn=lambda *args, **kwargs: None,
    )

    assert out_expr is not None
    assert meta["inverse_exact_score_budget"] == 2
    assert meta["inverse_exact_score_observed_count"] == 2
    assert len(score_calls) == 2
    assert len(set(score_calls)) == 2

    dup_rows = [row for row in meta["inverse_repair_slate"] if row["child_key"] == "9"]
    assert len(dup_rows) >= 2
    assert sum(1 for row in dup_rows if row["dedup_kept"]) == 1
    assert all(row["exact_child_score_observed"] is True for row in dup_rows)
    assert all(row["child_eff_mse"] == dup_rows[0]["child_eff_mse"] for row in dup_rows)


def test_inverse_action_beam_tuple_ranker_reorders_exact_budget(monkeypatch):
    expr = ("add", ("add", ("var", 0), ("var", 1)), ("var", 2))
    x_fit = torch.zeros((4, 3), dtype=torch.float64)
    y_fit = torch.zeros((4, 1), dtype=torch.float64)
    x_probe = torch.zeros((4, 3), dtype=torch.float64)
    y_probe = torch.zeros((4, 1), dtype=torch.float64)

    monkeypatch.setattr(inverse_action_mod, "_deterministic_row_subset", lambda cap, *rows: rows)
    monkeypatch.setattr(
        inverse_action_mod,
        "_estimate_inverse_action_transport",
        lambda *args, **kwargs: {
            "ranked_paths": [
                (1.0, 0.8, 0, 1, -1, (1,)),
                (0.9, 0.7, 0, 1, -1, (2,)),
            ],
            "path_transport_rel": {(1,): 0.8, (2,): 0.7},
        },
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_action_path_mode_beam_states",
        lambda **kwargs: [
            {
                "path": (1,),
                "sub": ("var", 0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [0],
                "path_gain": 1.20,
                "path_gain_pre_cut": 1.30,
                "rel_gain": 0.40,
                "transport_rel": 0.80,
                "lin_rel": 0.20,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 12.0,
                "target_mode": "identity",
            },
            {
                "path": (2,),
                "sub": ("var", 1),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [1],
                "path_gain": 0.90,
                "path_gain_pre_cut": 0.95,
                "rel_gain": 0.30,
                "transport_rel": 0.70,
                "lin_rel": 0.15,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 10.0,
                "target_mode": "full",
            },
        ],
    )

    def _collect(**kwargs):
        path = tuple(kwargs["path"])
        if path == (1,):
            return [("const", 10.0), ("const", 11.0)]
        return [("const", 20.0), ("const", 21.0)]

    monkeypatch.setattr(inverse_action_mod, "_inverse_collect_local_repair_candidates", _collect)
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_rank_local_repair_candidates",
        lambda cand_subtrees, **kwargs: [(float(idx), float(idx), cand) for idx, cand in enumerate(cand_subtrees)],
    )
    monkeypatch.setattr(inverse_action_mod, "_transport_aligned_local_rows", lambda local_rows, **kwargs: list(local_rows))

    def _fake_score_expr(cand, **kwargs):
        child_key = inverse_action_mod.node_str(cand)
        if child_key.startswith("(10"):
            eff = 0.45
        elif child_key.startswith("(11"):
            eff = 0.05
        elif child_key.startswith("(20"):
            eff = 0.22
        elif child_key.startswith("(21"):
            eff = 0.31
        else:
            eff = 1.0
        return {
            "expr": cand,
            "raw_mse": eff + 0.01,
            "eff_mse": eff,
            "mapping": {"kind": "poly"},
        }

    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_expr", _fake_score_expr)
    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_parent", lambda **kwargs: {"raw_mse": 1.0, "eff_mse": 1.0})

    def _fake_tuple_predict(bundle, row, *, path_rows=None, preview_rows=None, repair_action_names=None):
        rows = []
        for preview_row in list(preview_rows or []):
            child_key = str(preview_row.get("child_key", ""))
            utility = 5.0 if child_key.startswith("(11") else (2.0 if child_key.startswith("(20") else 1.0)
            rows.append({
                **dict(preview_row),
                "matched_path": list(preview_row.get("path", [])),
                "matched_target_mode": str(preview_row.get("target_mode", "")),
                "utility_estimate": float(utility),
                "value_estimate": float(utility * 0.5),
                "path_prob": 0.5,
            })
        rows.sort(key=lambda row: row["utility_estimate"], reverse=True)
        best = rows[0]
        return {
            "trained": True,
            "best_index": 0,
            "best_path": list(best["matched_path"]),
            "best_target_mode": best["matched_target_mode"],
            "best_action": best.get("action", "inv_steer"),
            "best_child_key": best.get("child_key", ""),
            "state_value_estimate": 0.5,
            "rows": rows,
        }

    monkeypatch.setattr(repair_critic_mod, "predict_repair_tuple_slate", _fake_tuple_predict)

    _, meta = inverse_action_mod.run_inverse_steering_action(
        expr,
        {"kind": "poly"},
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes=[("var", 0), ("var", 1)],
        pool_phi_fit=torch.zeros((4, 2), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 2), dtype=torch.float64),
        pool_dims=None,
        rng=random.Random(0),
        max_depth=4,
        nvars=1,
        poly_degree=2,
        topk_terms=2,
        return_meta=True,
        score_expr_fn=lambda *args, **kwargs: None,
        repair_tuple_bundle={"repair_tuple_ranker_trained": True},
        repair_tuple_controller_row={},
    )

    assert meta["inverse_exact_score_budget"] == 2
    assert meta["inverse_tuple_ranker_used"] is True
    assert meta["inverse_tuple_ranker_regret_weight"] == 1.0
    assert str(meta["inverse_tuple_ranker_best_child_key"]).startswith("(11")
    assert meta["estimated_child_eff_mse"] == 0.05
    assert meta["inverse_repair_slate"][0]["tuple_utility_estimate"] is not None


def test_inverse_action_beam_support_floor_then_global_allocation(monkeypatch):
    expr = ("add", ("add", ("var", 0), ("var", 1)), ("var", 2))
    x_fit = torch.zeros((4, 3), dtype=torch.float64)
    y_fit = torch.zeros((4, 1), dtype=torch.float64)
    x_probe = torch.zeros((4, 3), dtype=torch.float64)
    y_probe = torch.zeros((4, 1), dtype=torch.float64)
    score_calls: list[str] = []

    monkeypatch.setattr(inverse_action_mod, "_deterministic_row_subset", lambda cap, *rows: rows)
    monkeypatch.setattr(
        inverse_action_mod,
        "_estimate_inverse_action_transport",
        lambda *args, **kwargs: {
                "ranked_paths": [
                    (1.0, 0.9, 0, 1, -1, (1,)),
                    (0.9, 0.8, 0, 1, -1, (2,)),
                    (0.8, 0.7, 0, 1, -1, (1, 1)),
                ],
                "path_transport_rel": {(1,): 0.9, (2,): 0.8, (1, 1): 0.7},
            },
        )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_action_path_mode_beam_states",
        lambda **kwargs: [
            {
                "path": (1,),
                "sub": ("var", 0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [0],
                "path_gain": 1.2,
                "path_gain_pre_cut": 1.3,
                "rel_gain": 0.4,
                "transport_rel": 0.9,
                "lin_rel": 0.2,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 12.0,
                "target_mode": "identity",
            },
            {
                "path": (2,),
                "sub": ("var", 1),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [1],
                "path_gain": 1.1,
                "path_gain_pre_cut": 1.2,
                "rel_gain": 0.35,
                "transport_rel": 0.8,
                "lin_rel": 0.18,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 11.0,
                "target_mode": "identity",
            },
            {
                "path": (1, 1),
                "sub": ("var", 0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [1],
                "path_gain": 1.0,
                "path_gain_pre_cut": 1.1,
                "rel_gain": 0.3,
                "transport_rel": 0.7,
                "lin_rel": 0.15,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 10.0,
                "target_mode": "full",
            },
        ],
    )

    def _collect(**kwargs):
        path = tuple(kwargs["path"])
        if path == (1,):
            return [("const", 10.0), ("const", 11.0)]
        if path == (2,):
            return [("const", 20.0), ("const", 21.0)]
        if path == (1, 1):
            return [("const", 30.0), ("const", 31.0)]
        return [("const", 30.0), ("const", 31.0)]

    monkeypatch.setattr(inverse_action_mod, "_inverse_collect_local_repair_candidates", _collect)
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_rank_local_repair_candidates",
        lambda cand_subtrees, **kwargs: [(float(idx), float(idx), cand) for idx, cand in enumerate(cand_subtrees)],
    )
    monkeypatch.setattr(inverse_action_mod, "_transport_aligned_local_rows", lambda local_rows, **kwargs: list(local_rows))

    def _fake_score_expr(cand, **kwargs):
        child_key = inverse_action_mod.node_str(cand)
        score_calls.append(child_key)
        eff = 0.5
        if child_key.startswith("(11"):
            eff = 0.40
        elif child_key.startswith("(20"):
            eff = 0.32
        elif child_key.startswith("(30"):
            eff = 0.08
        elif child_key.startswith("(31"):
            eff = 0.12
        return {"expr": cand, "raw_mse": eff + 0.01, "eff_mse": eff, "mapping": {"kind": "poly"}}

    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_expr", _fake_score_expr)
    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_parent", lambda **kwargs: {"raw_mse": 1.0, "eff_mse": 1.0})

    def _fake_tuple_predict(bundle, row, *, path_rows=None, preview_rows=None, repair_action_names=None):
        rows = []
        for preview_row in list(preview_rows or []):
            path = tuple(int(v) for v in (preview_row.get("path", []) or ()))
            local_rank = int(preview_row.get("local_rank", 0) or 0)
            alloc = 0.0
            if path == (1, 1) and local_rank == 0:
                alloc = 9.0
            elif path == (1, 1) and local_rank == 1:
                alloc = 8.0
            elif path == (2,) and local_rank == 0:
                alloc = 2.0
            elif path == (1,) and local_rank == 1:
                alloc = 1.0
            rows.append({
                **dict(preview_row),
                "matched_path": list(preview_row.get("path", [])),
                "matched_target_mode": str(preview_row.get("target_mode", "")),
                "utility_estimate": alloc,
                "value_estimate": 0.0,
                "regret_estimate": 0.0,
                "combined_estimate": alloc,
                "allocation_estimate": alloc,
                "path_prob": 0.5,
            })
        rows.sort(key=lambda row: row["allocation_estimate"], reverse=True)
        best = rows[0]
        return {
            "trained": True,
            "best_index": 0,
            "best_path": list(best["matched_path"]),
            "best_target_mode": best["matched_target_mode"],
            "best_action": best.get("action", "inv_steer"),
            "best_child_key": best.get("child_key", ""),
            "state_value_estimate": 0.0,
            "child_value_lambda": 0.0,
            "regret_weight": 1.0,
            "rows": rows,
        }

    monkeypatch.setattr(repair_critic_mod, "predict_repair_tuple_slate", _fake_tuple_predict)

    _, meta = inverse_action_mod.run_inverse_steering_action(
        expr,
        {"kind": "poly"},
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes=[("var", 0), ("var", 1), ("var", 2)],
        pool_phi_fit=torch.zeros((4, 3), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 3), dtype=torch.float64),
        pool_dims=None,
        rng=random.Random(0),
        max_depth=4,
        nvars=3,
        poly_degree=2,
        topk_terms=4,
        return_meta=True,
        score_expr_fn=lambda *args, **kwargs: None,
        repair_tuple_bundle={"repair_tuple_ranker_trained": True},
        repair_tuple_controller_row={},
    )

    assert meta["inverse_exact_score_budget"] == 4
    assert meta["inverse_exact_support_floor_beams"] == 2
    assert meta["inverse_exact_support_floor_selected"] == 2
    assert meta["inverse_exact_global_allocated"] == 2
    assert len(score_calls) == 4
    assert any("11" in call for call in score_calls)
    assert any("20" in call for call in score_calls)
    assert any("30" in call for call in score_calls)
    assert any("31" in call for call in score_calls)


def test_inverse_action_beam_opportunity_controller_updates_prefix_state(monkeypatch):
    expr = ("add", ("var", 0), ("var", 1))
    x_fit = torch.zeros((4, 2), dtype=torch.float64)
    y_fit = torch.zeros((4, 1), dtype=torch.float64)
    x_probe = torch.zeros((4, 2), dtype=torch.float64)
    y_probe = torch.zeros((4, 1), dtype=torch.float64)
    prediction_snapshots: list[list[dict[str, object]]] = []

    monkeypatch.setattr(inverse_action_mod, "_deterministic_row_subset", lambda cap, *rows: rows)
    monkeypatch.setattr(
        inverse_action_mod,
        "_estimate_inverse_action_transport",
        lambda *args, **kwargs: {
            "ranked_paths": [
                (1.0, 0.9, 0, 1, -1, (1,)),
                (0.9, 0.8, 0, 1, -1, (2,)),
            ],
            "path_transport_rel": {(1,): 0.9, (2,): 0.8},
        },
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_action_path_mode_beam_states",
        lambda **kwargs: [
            {
                "path": (1,),
                "sub": ("var", 0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [0],
                "path_gain": 1.2,
                "path_gain_pre_cut": 1.3,
                "rel_gain": 0.4,
                "transport_rel": 0.9,
                "lin_rel": 0.2,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 12.0,
                "target_mode": "identity",
            },
            {
                "path": (2,),
                "sub": ("var", 1),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [1],
                "path_gain": 1.1,
                "path_gain_pre_cut": 1.2,
                "rel_gain": 0.35,
                "transport_rel": 0.8,
                "lin_rel": 0.18,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 11.0,
                "target_mode": "full",
            },
        ],
    )

    def _collect(**kwargs):
        path = tuple(kwargs["path"])
        if path == (1,):
            return [("const", 10.0), ("const", 11.0)]
        return [("const", 20.0), ("const", 21.0)]

    monkeypatch.setattr(inverse_action_mod, "_inverse_collect_local_repair_candidates", _collect)
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_rank_local_repair_candidates",
        lambda cand_subtrees, **kwargs: [(float(idx), float(idx), cand) for idx, cand in enumerate(cand_subtrees)],
    )
    monkeypatch.setattr(inverse_action_mod, "_transport_aligned_local_rows", lambda local_rows, **kwargs: list(local_rows))

    def _fake_score_expr(cand, **kwargs):
        child_key = inverse_action_mod.node_str(cand)
        if "10" in child_key:
            eff = 0.08
        elif "11" in child_key:
            eff = 0.18
        elif "20" in child_key:
            eff = 0.25
        elif "21" in child_key:
            eff = 0.30
        else:
            eff = 1.0
        return {"expr": cand, "raw_mse": eff + 0.01, "eff_mse": eff, "mapping": {"kind": "poly"}}

    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_expr", _fake_score_expr)
    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_parent", lambda **kwargs: {"raw_mse": 1.0, "eff_mse": 0.9})

    def _fake_predict(bundle, rows):
        snapshot = []
        by_path = {}
        for row in list(rows or []):
            path = tuple(int(v) for v in (row.get("path", []) or ()))
            snap_row = {
                "path": list(path),
                "budget_exact_spent": int(row.get("budget_exact_spent", 0) or 0),
                "budget_remaining": int(row.get("budget_remaining", 0) or 0),
                "current_best_child_eff_mse": row.get("current_best_child_eff_mse", None),
                "current_best_route_eff_mse": row.get("current_best_route_eff_mse", None),
            }
            snapshot.append(snap_row)
            by_path[path] = dict(row)
        prediction_snapshots.append(snapshot)

        if len(prediction_snapshots) == 1:
            assert by_path[(1,)]["budget_exact_spent"] == 0
            assert by_path[(2,)]["budget_exact_spent"] == 0
            ranked_paths = [(2,), (1,)]
        else:
            assert by_path[(2,)]["budget_exact_spent"] == 1
            assert by_path[(2,)]["budget_remaining"] == 1
            assert by_path[(2,)]["current_best_child_eff_mse"] == 0.25
            assert by_path[(2,)]["current_best_route_eff_mse"] == 0.25
            assert by_path[(1,)]["budget_exact_spent"] == 0
            ranked_paths = [(1,), (2,)]

        rows_out = []
        for rank, path in enumerate(ranked_paths):
            src = by_path[path]
            gain = 3.0 - float(rank)
            rows_out.append({
                **src,
                "expected_gain_next_under_executor": gain,
                "cost_estimate": 0.1,
                "fragility_prob": 0.0,
                "route_flip_prob": 0.0,
                "new_residual_basin_prob": 0.0,
                "acquisition_estimate": gain - 0.1,
            })
        return {"trained": True, "rows": rows_out}

    monkeypatch.setattr(inverse_action_mod, "predict_opportunity_slate", _fake_predict)

    _, meta = inverse_action_mod.run_inverse_steering_action(
        expr,
        {"kind": "poly"},
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes=[("var", 0), ("var", 1)],
        pool_phi_fit=torch.zeros((4, 2), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 2), dtype=torch.float64),
        pool_dims=None,
        rng=random.Random(0),
        max_depth=4,
        nvars=2,
        poly_degree=2,
        topk_terms=2,
        return_meta=True,
        score_expr_fn=lambda *args, **kwargs: None,
        repair_opportunity_controller_enable=True,
        repair_opportunity_bundle={"opportunity_controller_trained": True},
    )

    assert meta["status"] == "ok"
    assert meta["inverse_opportunity_controller_requested"] is True
    assert meta["inverse_opportunity_controller_used"] is True
    assert meta["inverse_exact_allocator_mode"] == "opportunity_controller"
    assert len(prediction_snapshots) == 2
    assert meta["inverse_exact_budget_trace_count"] == 2
    assert [tuple(item["path"]) for item in meta["inverse_exact_budget_trace"]] == [(2,), (1,)]
    assert meta["selected_path"] == [1]
    assert meta["selected_target_mode"] == "identity"

    final_rows = {
        tuple(int(v) for v in (row.get("path", []) or ())): row
        for row in meta["repair_opportunity_slate_final"]
    }
    assert meta["repair_opportunity_slate_final_count"] == 2
    assert final_rows[(2,)]["budget_exact_spent"] == 1
    assert final_rows[(2,)]["budget_remaining"] == 1
    assert final_rows[(2,)]["current_best_child_eff_mse"] == 0.25
    assert final_rows[(2,)]["evidence_level"] == "exact_known"
    assert final_rows[(1,)]["budget_exact_spent"] == 1
    assert final_rows[(1,)]["budget_remaining"] == 1
    assert final_rows[(1,)]["current_best_child_eff_mse"] == 0.08
    assert final_rows[(1,)]["evidence_level"] == "exact_known"


def test_inverse_action_beam_opportunity_controller_falls_back_to_legacy(monkeypatch):
    expr = ("add", ("var", 0), ("var", 1))
    x_fit = torch.zeros((4, 2), dtype=torch.float64)
    y_fit = torch.zeros((4, 1), dtype=torch.float64)
    x_probe = torch.zeros((4, 2), dtype=torch.float64)
    y_probe = torch.zeros((4, 1), dtype=torch.float64)
    score_calls: list[str] = []

    monkeypatch.setattr(inverse_action_mod, "_deterministic_row_subset", lambda cap, *rows: rows)
    monkeypatch.setattr(
        inverse_action_mod,
        "_estimate_inverse_action_transport",
        lambda *args, **kwargs: {
            "ranked_paths": [
                (1.0, 0.9, 0, 1, -1, (1,)),
                (0.9, 0.8, 0, 1, -1, (2,)),
                (0.8, 0.7, 0, 1, -1, (3,)),
            ],
            "path_transport_rel": {(1,): 0.9, (2,): 0.8, (3,): 0.7},
        },
    )
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_action_path_mode_beam_states",
        lambda **kwargs: [
            {
                "path": (1,),
                "sub": ("var", 0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [0],
                "path_gain": 1.2,
                "path_gain_pre_cut": 1.3,
                "rel_gain": 0.4,
                "transport_rel": 0.9,
                "lin_rel": 0.2,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 12.0,
                "target_mode": "identity",
            },
            {
                "path": (2,),
                "sub": ("var", 1),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [1],
                "path_gain": 1.1,
                "path_gain_pre_cut": 1.2,
                "rel_gain": 0.35,
                "transport_rel": 0.8,
                "lin_rel": 0.18,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 11.0,
                "target_mode": "identity",
            },
            {
                "path": (3,),
                "sub": ("var", 0),
                "target_dim": None,
                "xf": x_fit,
                "tf": y_fit,
                "xp": x_probe,
                "tp": y_probe,
                "wf": torch.ones((4, 1), dtype=torch.float64),
                "wp": torch.ones((4, 1), dtype=torch.float64),
                "mfit": torch.ones(4, dtype=torch.bool),
                "mprobe": torch.ones(4, dtype=torch.bool),
                "pool_idx": [0],
                "path_gain": 1.0,
                "path_gain_pre_cut": 1.1,
                "rel_gain": 0.3,
                "transport_rel": 0.7,
                "lin_rel": 0.15,
                "branch_factor": 1.0,
                "path_cut_factor": 1.0,
                "effective_n": 10.0,
                "target_mode": "full",
            },
        ],
    )

    def _collect(**kwargs):
        path = tuple(kwargs["path"])
        if path == (1,):
            return [("const", 10.0), ("const", 11.0)]
        if path == (2,):
            return [("const", 20.0), ("const", 21.0)]
        return [("const", 30.0), ("const", 31.0)]

    monkeypatch.setattr(inverse_action_mod, "_inverse_collect_local_repair_candidates", _collect)
    monkeypatch.setattr(
        inverse_action_mod,
        "_inverse_rank_local_repair_candidates",
        lambda cand_subtrees, **kwargs: [(float(idx), float(idx), cand) for idx, cand in enumerate(cand_subtrees)],
    )
    monkeypatch.setattr(inverse_action_mod, "_transport_aligned_local_rows", lambda local_rows, **kwargs: list(local_rows))

    def _fake_score_expr(cand, **kwargs):
        child_key = inverse_action_mod.node_str(cand)
        score_calls.append(child_key)
        if "10" in child_key:
            eff = 0.30
        elif "20" in child_key:
            eff = 0.12
        elif "30" in child_key:
            eff = 0.05
        else:
            eff = 0.40
        return {"expr": cand, "raw_mse": eff + 0.01, "eff_mse": eff, "mapping": {"kind": "poly"}}

    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_expr", _fake_score_expr)
    monkeypatch.setattr(inverse_action_mod, "_score_inverse_action_parent", lambda **kwargs: {"raw_mse": 1.0, "eff_mse": 0.9})
    monkeypatch.setattr(inverse_action_mod, "predict_opportunity_slate", lambda bundle, rows: {"trained": False, "rows": []})

    _, meta = inverse_action_mod.run_inverse_steering_action(
        expr,
        {"kind": "poly"},
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes=[("var", 0), ("var", 1)],
        pool_phi_fit=torch.zeros((4, 2), dtype=torch.float64),
        pool_phi_probe=torch.zeros((4, 2), dtype=torch.float64),
        pool_dims=None,
        rng=random.Random(0),
        max_depth=4,
        nvars=2,
        poly_degree=2,
        topk_terms=2,
        return_meta=True,
        score_expr_fn=lambda *args, **kwargs: None,
        repair_opportunity_controller_enable=True,
        repair_opportunity_bundle={"opportunity_controller_trained": True},
    )

    assert meta["status"] == "ok"
    assert meta["inverse_opportunity_controller_requested"] is True
    assert meta["inverse_opportunity_controller_used"] is False
    assert meta["inverse_exact_allocator_mode"] == "legacy"
    assert "prediction unavailable" in meta["inverse_opportunity_controller_error"]
    assert meta["inverse_exact_support_floor_beams"] == 1
    assert meta["inverse_exact_support_floor_selected"] == 1
    assert meta["inverse_exact_global_allocated"] == 1
    assert meta["inverse_exact_budget_trace_count"] == 2
    assert [tuple(item["path"]) for item in meta["inverse_exact_budget_trace"]] == [(1,), (2,)]
    assert all(item["allocator_mode"] == "legacy" for item in meta["inverse_exact_budget_trace"])
    assert len(score_calls) == 2
    assert meta["selected_path"] == [2]

    final_rows = {
        tuple(int(v) for v in (row.get("path", []) or ())): row
        for row in meta["repair_opportunity_slate_final"]
    }
    assert meta["repair_opportunity_slate_final_count"] == 3
    assert final_rows[(1,)]["budget_exact_spent"] == 1
    assert final_rows[(2,)]["budget_exact_spent"] == 1
    assert final_rows[(3,)]["budget_exact_spent"] == 0
    assert final_rows[(3,)]["candidate_count_observed"] == 0
    assert final_rows[(3,)]["evidence_level"] == "preview_only"
