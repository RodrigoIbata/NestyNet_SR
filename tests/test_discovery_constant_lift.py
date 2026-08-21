# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.discovery.committee import CommitteeMember
import nestynet_sr.discovery.constant_lift as lift_mod
from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, Var
import nestynet_sr.sr_search.factorized_search.constant_lift_solver as solver_mod


def test_solve_constant_lift_task_uses_regime_index_when_metadata_missing(monkeypatch):
    captured = {}

    def fake_run_explorer(**kwargs):
        captured.update(kwargs)
        return [
            {
                "expr": "x0",
                "toy_ast": ("var", 0),
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mse_raw": 1.0e-4,
                "mse_eff": 1.0e-4,
            }
        ]

    monkeypatch.setattr(solver_mod, "run_explorer", fake_run_explorer)

    result = solver_mod.solve_constant_lift_task(
        regime_ids=["d0", "d1", "d2"],
        values_by_regime={"d0": 1.0, "d1": 2.0, "d2": 4.0},
        dtype=torch.float64,
    )

    assert result is not None
    assert result["solver"] == "factorized_search"
    assert result["feature_source"] == "regime_index"
    assert result["feature_names"] == ["regime_index"]
    assert captured["x_fit_data"].shape == (3, 1)
    assert result["improvement_ratio"] > 1.0


def test_discover_constant_lifts_builds_substitution_preview(monkeypatch):
    def fake_solve_constant_lift_task(**kwargs):
        assert kwargs["regime_ids"] == ["d0", "d1", "d2"]
        return {
            "solver": "factorized_search",
            "expr": "x0",
            "expr_ast": ["var", 0],
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "fit_mse": 1.0e-4,
            "probe_mse": 1.0e-4,
            "baseline_mse": 1.0,
            "improvement_ratio": 10.0,
            "regime_ids": ["d0", "d1", "d2"],
            "regime_points": [[1.0], [2.0], [3.0]],
            "regime_values": [1.0, 2.0, 4.0],
            "feature_names": ["temperature"],
            "feature_source": "dataset_metadata",
        }

    monkeypatch.setattr(lift_mod, "solve_constant_lift_task", fake_solve_constant_lift_task)

    member = CommitteeMember(
        member_id="m0",
        symbolic_structure="shared_leaf*x0 + local_leaf",
        local_constants_by_experiment={
            "d0": {"local_leaf": 1.0},
            "d1": {"local_leaf": 2.0},
            "d2": {"local_leaf": 4.0},
        },
        display_expr="shared_leaf*x0 + local_leaf",
    )

    summary = lift_mod.discover_constant_lifts(
        [member],
        dataset_ids=["d0", "d1", "d2"],
        dataset_metadata={
            "d0": {"temperature": 1.0},
            "d1": {"temperature": 2.0},
            "d2": {"temperature": 3.0},
        },
        min_regimes=3,
        trigger_mean_cv=0.2,
        dtype=torch.float64,
    )

    assert summary["triggered_member_count"] == 1
    assert summary["proposal_count"] == 1
    proposal = summary["members"][0]["proposals"][0]
    assert proposal["constant_name"] == "local_leaf"
    assert proposal["substitution_preview"]["lift_expr"] == "temperature(x0)"
    assert proposal["lifted_display_expr"] == "shared_leaf*x0 + (temperature(x0))"


def test_apply_constant_lift_proposals_materializes_lifted_member():
    member = CommitteeMember(
        member_id="m0",
        symbolic_structure="shared_leaf*x0 + local_leaf",
        fitted_constants={"shared_leaf": 2.0, "local_leaf": 1.0},
        shared_constants={"shared_leaf": 2.0},
        local_constants_by_experiment={
            "d0": {"local_leaf": 1.0},
            "d1": {"local_leaf": 2.0},
        },
        validation_error=0.1,
        train_error=0.1,
        simplicity_score=0.5,
        physics_consistency_score=0.7,
        display_expr="shared_leaf*x0 + local_leaf",
    )

    result = lift_mod.apply_constant_lift_proposals(
        [member],
        {
            "enabled": True,
            "proposal_count": 1,
            "members": [
                {
                    "member_id": "m0",
                    "proposals": [
                        {
                            "constant_name": "local_leaf",
                            "mean_cv": 0.8,
                            "improvement_ratio": 10.0,
                            "lifted_display_expr": "shared_leaf*x0 + (x0)",
                            "feature_source": "dataset_metadata",
                            "substitution_preview": {
                                "constant_name": "local_leaf",
                                "lift_expr": "x0",
                            },
                        }
                    ],
                }
            ],
        },
        apply_topk=1,
        min_rel_gain=2.0,
    )

    summary = result["summary"]
    assert summary["apply_enabled"] is True
    assert summary["applied_member_count"] == 1
    proposal = summary["members"][0]["proposals"][0]
    assert proposal["applied"] is True
    applied_member = result["applied_members"][0]
    assert applied_member.display_expr == "shared_leaf*x0 + (x0)"
    assert applied_member.local_constants_by_experiment == {}
    assert applied_member.metadata["constant_lift_parent_member_id"] == "m0"
    assert applied_member.metadata["constant_lift_applied"] is True


def test_apply_constant_lift_proposals_uses_ast_reinsertion_when_parent_is_structured():
    parent_ast = AddNode(
        MulNode(
            AtomNode(kind="free_const", var_idxs=(), kwargs={"name": "shared_leaf"}, tag="shared_leaf"),
            Var(0),
        ),
        AtomNode(kind="free_const", var_idxs=(), kwargs={"name": "local_leaf"}, tag="local_leaf"),
    )
    member = CommitteeMember(
        member_id="m_ast",
        symbolic_structure=parent_ast,
        fitted_constants={"shared_leaf": 2.0, "local_leaf": 1.0},
        shared_constants={"shared_leaf": 2.0},
        local_constants_by_experiment={
            "d0": {"local_leaf": 1.0},
            "d1": {"local_leaf": 2.0},
        },
        validation_error=0.1,
        train_error=0.1,
        simplicity_score=0.5,
        physics_consistency_score=0.7,
        display_expr="shared_leaf*x0 + local_leaf",
    )

    result = lift_mod.apply_constant_lift_proposals(
        [member],
        {
            "enabled": True,
            "proposal_count": 1,
            "members": [
                {
                    "member_id": "m_ast",
                    "proposals": [
                        {
                            "constant_name": "local_leaf",
                            "mean_cv": 0.8,
                            "improvement_ratio": 10.0,
                            "lifted_display_expr": "shared_leaf*x0 + (temperature(x0))",
                            "feature_source": "dataset_metadata",
                            "feature_names": ["temperature"],
                            "expr_ast": ["var", 0],
                            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                            "substitution_preview": {
                                "constant_name": "local_leaf",
                                "lift_expr": "temperature(x0)",
                            },
                        }
                    ],
                }
            ],
        },
        apply_topk=1,
        min_rel_gain=2.0,
    )

    applied_member = result["applied_members"][0]
    assert not isinstance(applied_member.symbolic_structure, str)
    assert applied_member.display_expr == "shared_leaf*x0 + (temperature(x0))"
    assert applied_member.metadata["constant_lift_symbolic_structure_mode"] == "ast"
