# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

import nestynet_sr.sr_search.factorized_search.coordinate_lift_solver as coord_mod
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


def _build_sum_payload():
    x0_fit = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    x1_fit = torch.linspace(1.0, -1.0, 17, dtype=torch.float64)
    x_fit = torch.stack([x0_fit, x1_fit + 0.5 * x0_fit], dim=1)
    x0_probe = torch.linspace(-1.25, 1.25, 19, dtype=torch.float64)
    x1_probe = torch.linspace(1.25, -1.25, 19, dtype=torch.float64)
    x_probe = torch.stack([x0_probe, x1_probe + 0.5 * x0_probe], dim=1)
    z_fit = (x_fit[:, 0:1] + x_fit[:, 1:2])
    z_probe = (x_probe[:, 0:1] + x_probe[:, 1:2])
    y_fit = torch.sin(z_fit)
    y_probe = torch.sin(z_probe)
    grad_fit = torch.cos(z_fit).repeat(1, 2)
    grad_probe = torch.cos(z_probe).repeat(1, 2)
    spec = SubproblemSpec(
        problem_id="toy_coordinate_sum",
        problem_kind="local_problem",
        parent_expr=("add", ("const", 1.0), ("var", 0)),
        path=(1,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(0, 1),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=y_fit,
            x_probe=x_probe,
            t_probe=y_probe,
            grad_fit=grad_fit,
            grad_probe=grad_probe,
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("coord",)},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    return wrap_subproblem_spec_payload(spec), z_fit, z_probe


def _build_ratio_payload():
    x0_fit = torch.linspace(0.5, 2.5, 15, dtype=torch.float64)
    x1_fit = torch.linspace(1.5, 3.0, 15, dtype=torch.float64)
    x_fit = torch.stack([x0_fit, x1_fit], dim=1)
    x0_probe = torch.linspace(0.55, 2.75, 17, dtype=torch.float64)
    x1_probe = torch.linspace(1.4, 3.2, 17, dtype=torch.float64)
    x_probe = torch.stack([x0_probe, x1_probe], dim=1)
    z_fit = (x_fit[:, 0:1] / x_fit[:, 1:2])
    z_probe = (x_probe[:, 0:1] / x_probe[:, 1:2])
    spec = SubproblemSpec(
        problem_id="toy_coordinate_ratio",
        problem_kind="local_problem",
        parent_expr=("add", ("const", 1.0), ("var", 0)),
        path=(1,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(0, 1),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=z_fit,
            x_probe=x_probe,
            t_probe=z_probe,
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("coord",)},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    return wrap_subproblem_spec_payload(spec), z_fit, z_probe


def test_coordinate_lift_solver_builds_single_index_coordinate(monkeypatch):
    payload, z_fit, z_probe = _build_sum_payload()
    captured = []

    def fake_run_explorer(**kwargs):
        captured.append(kwargs)
        if torch.allclose(kwargs["x_fit_data"], z_fit) and torch.allclose(kwargs["x_probe_data"], z_probe):
            return [
                {
                    "toy_ast": ("sin", ("var", 0)),
                    "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                    "mse_raw": 1.0e-4,
                    "mse_eff": 1.0e-4,
                }
            ]
        return []

    monkeypatch.setattr(coord_mod, "run_explorer", fake_run_explorer)

    result = coord_mod.solve_local_coordinate_lift_preview_rows(
        parent_node=("add", ("const", 1.0), ("var", 0)),
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="coord_sum",
        path_gain=0.5,
        max_depth=4,
        nvars=2,
        poly_degree=2,
        var_dims=None,
        preview_topk=3,
        max_subtree_depth=3,
        coordinate_topk=4,
        coordinate_mode="single_index",
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["coordinate_lift_mode"] == "single_index"
    assert result["solver_meta"]["route_trigger_status"] == "single_index_like"
    assert result["solver_meta"]["route_reason_family"] == "coordinate_invariant"
    assert result["solver_meta"]["route_trigger_preferred"] is True
    assert any(torch.allclose(call["x_fit_data"], z_fit) for call in captured)
    rows = result["rows"]
    assert len(rows) == 1
    assert rows[0]["coordinate_lift_coord_expr"] == ("add", ("var", 0), ("var", 1))
    assert rows[0]["coordinate_lift_route_reason_family"] == "coordinate_invariant"
    assert rows[0]["expr"] == ("add", ("sin", ("add", ("var", 0), ("var", 1))), ("var", 0))
    assert rows[0]["witness_value_loss"] == rows[0]["local_probe_mse"]
    assert rows[0]["witness_energy_total"] == rows[0]["local_probe_mse"]


def test_coordinate_lift_solver_builds_ratio_coordinate(monkeypatch):
    payload, z_fit, z_probe = _build_ratio_payload()

    def fake_run_explorer(**kwargs):
        if torch.allclose(kwargs["x_fit_data"], z_fit) and torch.allclose(kwargs["x_probe_data"], z_probe):
            return [
                {
                    "toy_ast": ("var", 0),
                    "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                    "mse_raw": 2.0e-4,
                    "mse_eff": 2.0e-4,
                }
            ]
        return []

    monkeypatch.setattr(coord_mod, "run_explorer", fake_run_explorer)

    result = coord_mod.solve_local_coordinate_lift_preview_rows(
        parent_node=("add", ("const", 1.0), ("var", 0)),
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="coord_ratio",
        path_gain=0.5,
        max_depth=4,
        nvars=2,
        poly_degree=2,
        var_dims=None,
        preview_topk=3,
        max_subtree_depth=3,
        coordinate_topk=3,
        coordinate_mode="invariant",
    )

    assert result["solver_meta"]["status"] == "ok"
    rows = result["rows"]
    assert len(rows) == 1
    assert rows[0]["coordinate_lift_coord_kind"] == "ratio"
    assert rows[0]["coordinate_lift_coord_expr"] == ("div", ("var", 0), ("var", 1))
    assert rows[0]["expr"] == ("add", ("div", ("var", 0), ("var", 1)), ("var", 0))
    assert rows[0]["witness_value_loss"] == rows[0]["local_probe_mse"]


def test_coordinate_lift_solver_uses_witness_teacher_loss_when_enabled(monkeypatch):
    payload, z_fit, z_probe = _build_sum_payload()

    def fake_run_explorer(**kwargs):
        if torch.allclose(kwargs["x_fit_data"], z_fit) and torch.allclose(kwargs["x_probe_data"], z_probe):
            return [
                {
                    "toy_ast": ("sin", ("var", 0)),
                    "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                    "mse_raw": 1.0e-4,
                    "mse_eff": 1.0e-4,
                }
            ]
        return []

    monkeypatch.setattr(coord_mod, "run_explorer", fake_run_explorer)

    result = coord_mod.solve_local_coordinate_lift_preview_rows(
        parent_node=("add", ("const", 1.0), ("var", 0)),
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="coord_sum_witness",
        path_gain=0.5,
        max_depth=4,
        nvars=2,
        poly_degree=2,
        var_dims=None,
        preview_topk=3,
        max_subtree_depth=3,
        coordinate_topk=4,
        coordinate_mode="single_index",
        witness_loss_enable=True,
        witness_grad_weight=0.5,
    )

    row = result["rows"][0]
    assert row["witness_grad_loss"] is not None
    assert row["witness_energy_total"] == row["local_probe_mse"]
    assert row["local_probe_mse"] >= row["witness_value_loss"]
