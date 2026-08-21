# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import pytest
import torch

from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node
from nestynet_sr.sr_search.factorized_search.inverse_spec_solver import (
    _LocalProblem,
    _SolverContext,
    _score_node_against_problem,
    solve_local_problem_spec_preview_rows,
)
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


def _make_problem_with_teacher() -> _LocalProblem:
    x = torch.tensor([[-1.0], [0.0], [0.0], [1.0]], dtype=torch.float64)
    target = eval_node(("sqr", ("var", 0)), x)
    grad = 2.0 * x
    d2 = 2.0 * torch.ones_like(x)
    return _LocalProblem(
        xf=x,
        tf=target,
        wf=None,
        xp=x.clone(),
        tp=target.clone(),
        wp=None,
        target_dim=None,
        confidence=1.0,
        valid_frac=1.0,
        wrappers_left=0,
        recursion_level=0,
        trace=(),
        grad_fit=grad,
        grad_probe=grad.clone(),
        d2_fit=d2,
        d2_probe=d2.clone(),
        diagnostics={
            "fit_jet_source": "oracle",
            "probe_jet_source": "oracle",
            "fit_jet_requested_source": "oracle",
            "probe_jet_requested_source": "oracle",
            "fit_jet_fallback_used": False,
            "probe_jet_fallback_used": False,
        },
    )


def _make_ctx(*, witness_loss_enable: bool) -> _SolverContext:
    return _SolverContext(
        parent_node=("var", 0),
        hole_path=(),
        hole_sub=("var", 0),
        max_depth=6,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        seed_nodes=[],
        local_score_mode="affine",
        enum_max_depth=3,
        enum_max_trees=64,
        max_subtree_depth=4,
        preview_topk=8,
        complexity_penalty=0.0,
        recursive_enable=False,
        recursive_max_depth=0,
        recursive_trigger_rel_mse=0.25,
        recursive_seed_cap=4,
        recursive_branch_topk=2,
        recursive_child_topk=2,
        safe_eps=1.0e-12,
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.05,
        branch_beam_width=1,
        min_valid_frac=0.25,
        min_confidence=0.10,
        allow_legacy_aux=False,
        legacy_aux_kwargs={},
        stats={},
        target_mode="identity",
        target_mapping_kind="affine",
        family_battery_enable=False,
        family_battery_mode="outer",
        witness_jets_enable=True,
        witness_d2_enable=True,
        witness_max_rows=64,
        witness_loss_enable=bool(witness_loss_enable),
        witness_grad_weight=1.0,
        witness_d2_weight=0.0,
        witness_diag_weight=0.0,
        witness_physics_weight=0.0,
        active_var_screen_enable=False,
        active_var_grad_tol=1.0e-3,
        active_var_max_count=4,
    )


def test_score_node_against_problem_uses_gradient_teacher_loss_to_break_value_tie():
    problem = _make_problem_with_teacher()
    truth_node = ("sqr", ("var", 0))
    tied_value_node = ("sqr", ("sqr", ("var", 0)))

    score_truth_off = _score_node_against_problem(
        truth_node,
        problem=problem,
        ctx=_make_ctx(witness_loss_enable=False),
        source="unit",
        generation_kind="unit",
    )
    score_tied_off = _score_node_against_problem(
        tied_value_node,
        problem=problem,
        ctx=_make_ctx(witness_loss_enable=False),
        source="unit",
        generation_kind="unit",
    )

    assert score_truth_off is not None
    assert score_tied_off is not None
    assert score_truth_off.local_probe_mse == pytest.approx(score_tied_off.local_probe_mse)
    assert score_truth_off.value_probe_mse == pytest.approx(0.0)
    assert score_tied_off.value_probe_mse == pytest.approx(0.0)

    score_truth_on = _score_node_against_problem(
        truth_node,
        problem=problem,
        ctx=_make_ctx(witness_loss_enable=True),
        source="unit",
        generation_kind="unit",
    )
    score_tied_on = _score_node_against_problem(
        tied_value_node,
        problem=problem,
        ctx=_make_ctx(witness_loss_enable=True),
        source="unit",
        generation_kind="unit",
    )

    assert score_truth_on is not None
    assert score_tied_on is not None
    assert score_truth_on.local_probe_mse < score_tied_on.local_probe_mse
    assert score_truth_on.witness_grad_loss == pytest.approx(0.0)
    assert score_tied_on.witness_grad_loss is not None
    assert score_tied_on.witness_grad_loss > 0.0
    assert score_truth_on.witness_energy_total == pytest.approx(score_truth_on.local_probe_mse)
    assert score_tied_on.witness_energy_total == pytest.approx(score_tied_on.local_probe_mse)
    assert score_truth_on.witness_exact_jet_used is True
    assert score_truth_on.witness_fit_jet_source == "oracle"
    assert score_truth_on.witness_probe_jet_source == "oracle"


def test_local_problem_preview_rows_emit_teacher_loss_fields_when_enabled():
    problem = _make_problem_with_teacher()
    payload = wrap_subproblem_spec_payload(
        SubproblemSpec(
            problem_id="teacher-loss-preview",
            problem_kind="local_problem",
            parent_expr=("var", 0),
            path=(),
            direction="inside_out",
            target_mode="identity",
            target_mapping_kind="affine",
            target_dim=None,
            recursion_level=1,
            active_vars=(0,),
            witness=WitnessBundle(
                x_fit=problem.xf,
                t_fit=problem.tf,
                x_probe=problem.xp,
                t_probe=problem.tp,
                grad_fit=problem.grad_fit,
                grad_probe=problem.grad_probe,
                d2_fit=problem.d2_fit,
                d2_probe=problem.d2_probe,
                diagnostics={
                    "confidence": 1.0,
                    "valid_frac": 1.0,
                    "trace": ("teacher",),
                    "fit_jet_source": "oracle",
                    "probe_jet_source": "oracle",
                    "fit_jet_requested_source": "oracle",
                    "probe_jet_requested_source": "oracle",
                    "fit_jet_fallback_used": False,
                    "probe_jet_fallback_used": False,
                },
                masks={},
            ),
            metadata={"hole_sub": ("var", 0)},
        )
    )

    result = solve_local_problem_spec_preview_rows(
        parent_node=("var", 0),
        spec_payload=payload,
        path=(),
        target_mode="identity",
        target_mapping_kind="affine",
        beam_rank=0,
        slate_id="teacher-preview",
        path_gain=0.0,
        max_depth=6,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        pool_nodes=[],
        pool_dims=[],
        preview_topk=4,
        witness_loss_enable=True,
        witness_grad_weight=1.0,
        witness_diag_weight=1.0,
        witness_physics_weight=1.0,
    )

    assert result["solver_meta"]["status"] == "ok"
    assert result["solver_meta"]["witness_loss_enable"] is True
    rows = result["rows"]
    assert rows
    assert "witness_value_loss" in rows[0]
    assert "witness_grad_loss" in rows[0]
    assert "witness_diag_loss" in rows[0]
    assert "witness_physics_loss" in rows[0]
    assert "witness_energy_total" in rows[0]
    assert rows[0]["witness_fit_jet_source"] == "oracle"
    assert rows[0]["witness_probe_jet_source"] == "oracle"
    assert rows[0]["witness_exact_jet_used"] is True
    assert rows[0]["witness_value_loss"] == pytest.approx(0.0)
    assert rows[0]["witness_diag_loss"] == pytest.approx(0.0)
    assert rows[0]["witness_physics_loss"] == pytest.approx(0.0)
