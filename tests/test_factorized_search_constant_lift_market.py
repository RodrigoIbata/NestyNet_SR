# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

import nestynet_sr.sr_search.factorized_search.constant_lift_solver as lift_mod
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


def _build_constant_lift_payload():
    x_fit = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    x_probe = torch.tensor(
        [
            [0.0, 0.5],
            [0.0, 1.5],
            [0.0, 2.5],
        ],
        dtype=torch.float64,
    )
    t_fit = x_fit[:, 1:2].clone()
    t_probe = x_probe[:, 1:2].clone()
    spec = SubproblemSpec(
        problem_id="toy_constant_lift",
        problem_kind="local_problem",
        parent_expr=("add", ("var", 0), ("const", 0.0)),
        path=(2,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(1,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=t_fit,
            x_probe=x_probe,
            t_probe=t_probe,
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("constant_lift",)},
        ),
        metadata={
            "hole_sub": ("const", 0.0),
            "constant_lift_task": {
                "constant_name": "local_leaf",
                "regime_ids": ["d0", "d1", "d2"],
                "values_by_regime": {"d0": 0.0, "d1": 1.0, "d2": 2.0},
                "dataset_metadata": {
                    "d0": {"temperature": 0.0},
                    "d1": {"temperature": 1.0},
                    "d2": {"temperature": 2.0},
                },
                "feature_nodes": [("var", 1)],
            },
        },
    )
    return wrap_subproblem_spec_payload(spec)


def _build_auto_constant_lift_payload():
    x_fit = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    x_probe = torch.tensor(
        [
            [0.0, 0.5],
            [0.0, 1.5],
            [0.0, 2.5],
        ],
        dtype=torch.float64,
    )
    t_fit = x_fit[:, 1:2].clone()
    t_probe = x_probe[:, 1:2].clone()
    spec = SubproblemSpec(
        problem_id="toy_constant_lift_auto",
        problem_kind="local_problem",
        parent_expr=("add", ("var", 0), ("const", 0.0)),
        path=(2,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(1,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=t_fit,
            x_probe=x_probe,
            t_probe=t_probe,
            masks={},
            diagnostics={
                "confidence": 0.9,
                "valid_frac": 0.95,
                "trace": ("constant_lift",),
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
            },
        ),
        metadata={
            "hole_sub": ("const", 0.0),
        },
    )
    return wrap_subproblem_spec_payload(spec)


def test_local_constant_lift_solver_builds_preview_row(monkeypatch):
    payload = _build_constant_lift_payload()

    def fake_solve_constant_lift_task(**kwargs):
        assert kwargs["regime_ids"] == ["d0", "d1", "d2"]
        return {
            "solver": "factorized_search",
            "expr": "x0",
            "expr_ast": ["var", 0],
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "fit_mse": 1.0e-6,
            "probe_mse": 1.0e-6,
            "baseline_mse": 1.0,
            "improvement_ratio": 100.0,
            "regime_ids": ["d0", "d1", "d2"],
            "feature_names": ["temperature"],
            "feature_source": "dataset_metadata",
        }

    monkeypatch.setattr(lift_mod, "solve_constant_lift_task", fake_solve_constant_lift_task)

    result = lift_mod.solve_local_constant_lift_preview_rows(
        parent_node=("add", ("var", 0), ("const", 0.0)),
        spec_payload=payload,
        path=(2,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_constant_lift",
        path_gain=0.5,
        max_depth=4,
        nvars=2,
        poly_degree=2,
        var_dims=None,
        preview_topk=2,
        max_subtree_depth=3,
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["route_reason_family"] == "regime_lift"
    assert result["solver_meta"]["route_trigger_preferred"] is True
    rows = result["rows"]
    assert len(rows) == 1
    assert rows[0]["proposal_family"] == "constant_lift_route"
    assert rows[0]["constant_lift_constant_name"] == "local_leaf"
    assert rows[0]["constant_lift_feature_source"] == "dataset_metadata"
    assert rows[0]["constant_lift_route_reason_family"] == "regime_lift"
    assert rows[0]["expr"] == ("add", ("var", 0), ("var", 1))
    assert rows[0]["local_probe_mse"] <= 1.0e-12


def test_local_constant_lift_solver_auto_synthesizes_task_from_parameter_stability(monkeypatch):
    payload = _build_auto_constant_lift_payload()

    def fake_solve_constant_lift_task(**kwargs):
        assert kwargs["regime_ids"] == ["d0", "d1", "d2"]
        assert kwargs["values_by_regime"] == {"d0": 1.0, "d1": 2.0, "d2": 4.0}
        assert kwargs["dataset_metadata"] == {
            "d0": {"temperature": 0.0},
            "d1": {"temperature": 1.0},
            "d2": {"temperature": 2.0},
        }
        return {
            "solver": "factorized_search",
            "expr": "x0",
            "expr_ast": ["var", 0],
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "fit_mse": 1.0e-6,
            "probe_mse": 1.0e-6,
            "baseline_mse": 1.0,
            "improvement_ratio": 100.0,
            "regime_ids": ["d0", "d1", "d2"],
            "feature_names": ["temperature"],
            "feature_source": "dataset_metadata",
        }

    monkeypatch.setattr(lift_mod, "solve_constant_lift_task", fake_solve_constant_lift_task)

    result = lift_mod.solve_local_constant_lift_preview_rows(
        parent_node=("add", ("var", 0), ("const", 0.0)),
        spec_payload=payload,
        path=(2,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_constant_lift_auto",
        path_gain=0.5,
        max_depth=4,
        nvars=2,
        poly_degree=2,
        var_dims=None,
        preview_topk=2,
        max_subtree_depth=3,
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["task_source"] == "parameter_stability_auto"
    assert result["solver_meta"]["route_trigger_status"] == "drifting_constants"
    assert result["solver_meta"]["route_trigger_preferred"] is True
    assert result["solver_meta"]["mean_cv"] > result["solver_meta"]["trigger_mean_cv"]
    rows = result["rows"]
    assert len(rows) == 1
    assert rows[0]["constant_lift_constant_name"] == "local_leaf"
    assert rows[0]["constant_lift_feature_source"] == "dataset_metadata"
