# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

import nestynet_sr.sr_search.factorized_search.local_sr_solver as local_sr_mod
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


def _build_spec_payload(*, continuation_frames=None, include_grad: bool = False):
    x_fit = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-1.5, 1.5, 11, dtype=torch.float64).unsqueeze(-1)
    y_fit = x_fit.clone()
    y_probe = x_probe.clone()
    grad_fit = torch.ones_like(x_fit) if include_grad else None
    grad_probe = torch.ones_like(x_probe) if include_grad else None
    spec = SubproblemSpec(
        problem_id="toy_local_problem",
        problem_kind="local_problem",
        parent_expr=("add", ("const", 1.0), ("var", 0)),
        path=(1,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=tuple(dict(frame) for frame in list(continuation_frames or [])),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(0,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=y_fit,
            x_probe=x_probe,
            t_probe=y_probe,
            grad_fit=grad_fit,
            grad_probe=grad_probe,
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("inner",)},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    return wrap_subproblem_spec_payload(spec)


def test_local_recursive_sr_solver_builds_wrapped_preview_rows(monkeypatch):
    payload = _build_spec_payload(continuation_frames=[{"wrap_kind": "unary", "op": "sin", "slot": 0}])
    captured = {}

    def fake_run_explorer(**kwargs):
        captured.update(kwargs)
        return [
            {
                "toy_ast": ("var", 0),
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mse_raw": 1.0e-4,
                "mse_eff": 2.0e-4,
                "expr": "x0",
            }
        ]

    monkeypatch.setattr(local_sr_mod, "run_explorer", fake_run_explorer)

    result = local_sr_mod.solve_local_recursive_sr_preview_rows(
        parent_node=("add", ("const", 1.0), ("var", 0)),
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        preview_topk=3,
        exact_budget=2,
        max_subtree_depth=3,
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["search_max_depth"] == 3
    assert result["solver_meta"]["requested_preview_topk"] == 3
    assert captured["return_topk"] == 3
    assert captured["y_fit_data"].shape == (9, 1)
    rows = result["rows"]
    assert len(rows) == 1
    assert rows[0]["proposal_family"] == "local_recursive_sr"
    assert rows[0]["expr"] == ("add", ("sin", ("var", 0)), ("var", 0))
    assert rows[0]["local_recursive_sr_result_rank"] == 0
    assert rows[0]["witness_value_loss"] == rows[0]["local_probe_mse"]
    assert rows[0]["witness_energy_total"] == rows[0]["local_probe_mse"]
    assert rows[0]["witness_grad_loss"] is None


def test_local_recursive_sr_solver_respects_route_budget(monkeypatch):
    payload = _build_spec_payload()

    def fake_run_explorer(**kwargs):
        return [
            {
                "toy_ast": ("var", 0),
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mse_raw": 1.0e-3,
                "mse_eff": 1.0e-3,
                "expr": "x0",
            },
            {
                "toy_ast": ("neg", ("var", 0)),
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mse_raw": 2.0e-3,
                "mse_eff": 2.0e-3,
                "expr": "-x0",
            },
        ]

    monkeypatch.setattr(local_sr_mod, "run_explorer", fake_run_explorer)

    result = local_sr_mod.solve_local_recursive_sr_preview_rows(
        parent_node=("add", ("const", 1.0), ("var", 0)),
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_budget",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        preview_topk=4,
        exact_budget=1,
        max_subtree_depth=3,
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["preview_count"] == 1
    assert len(result["rows"]) == 1


def test_local_recursive_sr_solver_subsets_active_vars_and_remaps_back(monkeypatch):
    x_fit = torch.stack(
        [
            torch.linspace(-1.0, 1.0, 9, dtype=torch.float64),
            torch.linspace(2.0, 4.0, 9, dtype=torch.float64),
        ],
        dim=1,
    )
    x_probe = torch.stack(
        [
            torch.linspace(-1.5, 1.5, 11, dtype=torch.float64),
            torch.linspace(1.5, 4.5, 11, dtype=torch.float64),
        ],
        dim=1,
    )
    y_fit = x_fit[:, 1:2].clone()
    y_probe = x_probe[:, 1:2].clone()
    spec = SubproblemSpec(
        problem_id="toy_local_problem_active_vars",
        problem_kind="local_problem",
        parent_expr=("add", ("const", 1.0), ("var", 1)),
        path=(1,),
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
            t_fit=y_fit,
            x_probe=x_probe,
            t_probe=y_probe,
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("inner",)},
        ),
        metadata={"hole_sub": ("var", 1)},
    )
    payload = wrap_subproblem_spec_payload(spec)
    captured = {}

    def fake_run_explorer(**kwargs):
        captured.update(kwargs)
        return [
            {
                "toy_ast": ("var", 0),
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mse_raw": 1.0e-4,
                "mse_eff": 1.0e-4,
                "expr": "x0",
            }
        ]

    monkeypatch.setattr(local_sr_mod, "run_explorer", fake_run_explorer)

    result = local_sr_mod.solve_local_recursive_sr_preview_rows(
        parent_node=("add", ("const", 1.0), ("var", 1)),
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_active_vars",
        path_gain=0.5,
        max_depth=4,
        nvars=2,
        poly_degree=2,
        var_dims=None,
        preview_topk=3,
        exact_budget=2,
        max_subtree_depth=3,
    )

    assert captured["nvars"] == 1
    assert captured["x_fit_data"].shape == (9, 1)
    assert captured["x_probe_data"].shape == (11, 1)
    assert result["solver_meta"]["active_vars"] == [1]
    assert result["solver_meta"]["active_var_subsetting_used"] is True
    assert result["rows"][0]["expr"] == ("add", ("var", 1), ("var", 1))
    assert result["rows"][0]["witness_value_loss"] == result["rows"][0]["local_probe_mse"]


def test_local_recursive_sr_solver_uses_witness_teacher_loss_when_enabled(monkeypatch):
    payload = _build_spec_payload(include_grad=True)

    def fake_run_explorer(**kwargs):
        return [
            {
                "toy_ast": ("var", 0),
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mse_raw": 1.0e-4,
                "mse_eff": 1.0e-4,
                "expr": "x0",
            }
        ]

    monkeypatch.setattr(local_sr_mod, "run_explorer", fake_run_explorer)

    result = local_sr_mod.solve_local_recursive_sr_preview_rows(
        parent_node=("add", ("const", 1.0), ("var", 0)),
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_witness",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        preview_topk=3,
        exact_budget=2,
        max_subtree_depth=3,
        witness_loss_enable=True,
        witness_grad_weight=0.5,
    )

    row = result["rows"][0]
    assert row["witness_grad_loss"] is not None
    assert row["witness_energy_total"] == row["local_probe_mse"]
    assert row["local_probe_mse"] >= row["witness_value_loss"]
