# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

import nestynet_sr.sr_search.factorized_search.soft_edit_search as soft_mod
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


def _build_spec_payload(*, hole_sub=("var", 0), wrappers_left=0):
    x_fit = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-1.5, 1.5, 25, dtype=torch.float64).unsqueeze(-1)
    t_fit = torch.sin(x_fit)
    t_probe = torch.sin(x_probe)
    spec = SubproblemSpec(
        problem_id="toy_soft_problem",
        problem_kind="local_problem",
        parent_expr=("var", 0),
        path=(),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        wrappers_left=wrappers_left,
        recursion_level=1,
        active_vars=(0,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=t_fit,
            x_probe=x_probe,
            t_probe=t_probe,
            grad_fit=torch.cos(x_fit),
            grad_probe=torch.cos(x_probe),
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("soft",)},
            masks={},
        ),
        metadata={"hole_sub": hole_sub},
    )
    return wrap_subproblem_spec_payload(spec)


def test_local_soft_edit_search_prefers_snapped_discrete_edit():
    payload = _build_spec_payload(wrappers_left=1)

    result = soft_mod.solve_local_soft_edit_preview_rows(
        parent_node=("var", 0),
        spec_payload=payload,
        path=(),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_soft",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=3,
        max_subtree_depth=3,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["soft_edit_steps_executed"] <= 32
    assert result["solver_meta"]["best_soft_loss"] is not None
    rows = result["rows"]
    assert rows
    assert rows[0]["expr"] == ("sin", ("var", 0))
    assert rows[0]["proposal_family"] == "soft_edit_search"
    assert rows[0]["soft_edit_kind"] == "wrap:sin"
    assert rows[0]["soft_edit_weight"] is not None


def test_soft_edit_search_does_not_mutate_parent_ast_and_returns_tuple_ast():
    payload = _build_spec_payload()
    parent_node = ("add", ("var", 0), ("const", 1.0))
    original_parent = parent_node

    result = soft_mod.solve_local_soft_edit_preview_rows(
        parent_node=parent_node,
        spec_payload=payload,
        path=(1,),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_soft_parent",
        path_gain=0.5,
        max_depth=5,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=2,
        max_subtree_depth=3,
        soft_edit_steps=24,
        soft_edit_l1=1.0e-3,
    )

    assert parent_node == original_parent
    assert result["rows"]
    assert isinstance(result["rows"][0]["expr"], tuple)
    assert result["rows"][0]["expr"][0] == "add"


def test_local_soft_edit_search_threads_witness_teacher_loss():
    payload = _build_spec_payload(wrappers_left=1)

    result = soft_mod.solve_local_soft_edit_preview_rows(
        parent_node=("var", 0),
        spec_payload=payload,
        path=(),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_soft_witness",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=3,
        max_subtree_depth=3,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
        witness_loss_enable=True,
        witness_grad_weight=0.5,
        witness_diag_weight=0.25,
    )

    row = result["rows"][0]
    assert row["witness_grad_loss"] is not None
    assert row["witness_diag_loss"] is not None
    assert row["witness_energy_total"] == row["local_probe_mse"]


def test_local_soft_edit_search_blocks_wrap_edits_without_budget():
    payload = _build_spec_payload(wrappers_left=0)

    result = soft_mod.solve_local_soft_edit_preview_rows(
        parent_node=("var", 0),
        spec_payload=payload,
        path=(),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="toy_soft_no_wrap_budget",
        path_gain=0.5,
        max_depth=4,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=3,
        max_subtree_depth=3,
        soft_edit_steps=32,
        soft_edit_l1=1.0e-3,
    )

    assert result["rows"]
    assert all(not str(row["soft_edit_kind"]).startswith("wrap:") for row in result["rows"])
