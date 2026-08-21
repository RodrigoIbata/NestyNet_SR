# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

import nestynet_sr.sr_search.factorized_search.tangent_edit as tangent_mod
from nestynet_sr.sr_search.factorized_search.expr_ast import node_str
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


def _build_spec_payload(*, hole_sub=("var", 0), target_dim=None, wrappers_left=0):
    x_fit = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-1.5, 1.5, 25, dtype=torch.float64).unsqueeze(-1)
    t_fit = torch.sin(x_fit)
    t_probe = torch.sin(x_probe)
    spec = SubproblemSpec(
        problem_id="toy_tangent_problem",
        problem_kind="local_problem",
        parent_expr=("var", 0),
        path=(),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=target_dim,
        wrappers_left=wrappers_left,
        recursion_level=1,
        active_vars=(0, 1),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=t_fit,
            x_probe=x_probe,
            t_probe=t_probe,
            grad_fit=torch.cos(x_fit),
            grad_probe=torch.cos(x_probe),
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("tangent",)},
            masks={},
        ),
        metadata={"hole_sub": hole_sub},
    )
    return wrap_subproblem_spec_payload(spec)


def test_local_tangent_edit_solver_prefers_residual_aligned_wrap():
    payload = _build_spec_payload(wrappers_left=1)

    result = tangent_mod.solve_local_tangent_edit_preview_rows(
        parent_node=("var", 0),
        spec_payload=payload,
        path=(),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_tangent",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        max_subtree_depth=3,
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["target_gradient_used"] is True
    rows = result["rows"]
    assert rows
    assert rows[0]["expr"] == ("sin", ("var", 0))
    assert rows[0]["proposal_family"] == "tangent_edit"
    assert rows[0]["tangent_edit_kind"] == "wrap:sin"
    assert rows[0]["tangent_edit_rank"] == 0


def test_local_tangent_edit_solver_threads_witness_teacher_loss():
    payload = _build_spec_payload(wrappers_left=1)

    result = tangent_mod.solve_local_tangent_edit_preview_rows(
        parent_node=("var", 0),
        spec_payload=payload,
        path=(),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_tangent_witness",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        max_subtree_depth=3,
        witness_loss_enable=True,
        witness_grad_weight=0.5,
        witness_diag_weight=0.25,
    )

    row = result["rows"][0]
    assert row["witness_grad_loss"] is not None
    assert row["witness_diag_loss"] is not None
    assert row["witness_energy_total"] == row["local_probe_mse"]


def test_tangent_edit_menu_filters_dimension_incompatible_additions():
    candidates = tangent_mod._enumerate_tangent_edit_nodes(
        ("var", 0),
        target_dim=(1.0,),
        nvars=2,
        active_vars=(0, 1),
        wrappers_left=1,
        pool_nodes=[("var", 1)],
        var_dims=[(1.0,), (0.0,)],
    )

    node_keys = {node_str(item["node"]) for item in candidates}
    assert node_str(("mul", ("var", 0), ("var", 1))) in node_keys
    assert node_str(("add", ("var", 0), ("var", 1))) not in node_keys
    assert node_str(("sin", ("var", 0))) not in node_keys


def test_local_tangent_edit_solver_blocks_wrap_edits_without_budget():
    payload = _build_spec_payload(wrappers_left=0)

    result = tangent_mod.solve_local_tangent_edit_preview_rows(
        parent_node=("var", 0),
        spec_payload=payload,
        path=(),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_tangent_no_wrap_budget",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        max_subtree_depth=3,
    )

    assert result["rows"]
    assert all(not str(row["tangent_edit_kind"]).startswith("wrap:") for row in result["rows"])


def test_tangent_edit_menu_respects_wrapper_budget():
    candidates = tangent_mod._enumerate_tangent_edit_nodes(
        ("var", 0),
        target_dim=None,
        nvars=1,
        active_vars=(0,),
        wrappers_left=0,
        pool_nodes=[],
        var_dims=None,
    )

    edit_kinds = {str(item["edit_kind"]) for item in candidates}
    assert all(not kind.startswith("wrap:") for kind in edit_kinds)
