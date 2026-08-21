# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import pytest
import torch

from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node
from nestynet_sr.sr_search.factorized_search.local_teacher_loss import score_local_teacher_loss


def _score_teacher(
    node,
    *,
    x: torch.Tensor,
    target_value: torch.Tensor,
    target_grad: torch.Tensor | None = None,
    target_d2: torch.Tensor | None = None,
    diag_weight: float = 0.0,
    physics_weight: float = 0.0,
    target_diagnostics: dict | None = None,
):
    pred = eval_node(node, x)
    return score_local_teacher_loss(
        node,
        pred_fit=pred,
        pred_probe=pred,
        x_fit=x,
        x_probe=x,
        target_fit=target_value,
        target_probe=target_value,
        w_fit=None,
        w_probe=None,
        target_grad_fit=target_grad,
        target_grad_probe=target_grad,
        target_d2_fit=target_d2,
        target_d2_probe=target_d2,
        poly_degree=2,
        mode="affine",
        grad_weight=0.0,
        d2_weight=0.0,
        diag_weight=float(diag_weight),
        physics_weight=float(physics_weight),
        target_diagnostics=dict(target_diagnostics or {"confidence": 1.0, "valid_frac": 1.0}),
    )


def test_score_local_teacher_loss_emits_diag_loss_from_shape_summaries():
    x = torch.tensor([[-1.0], [0.0], [0.0], [1.0]], dtype=torch.float64)
    target_node = ("sqr", ("var", 0))
    target_value = eval_node(target_node, x)
    target_grad = 2.0 * x
    target_d2 = 2.0 * torch.ones_like(x)

    truth = _score_teacher(
        target_node,
        x=x,
        target_value=target_value,
        target_grad=target_grad,
        target_d2=target_d2,
        diag_weight=1.0,
    )
    tied_value = _score_teacher(
        ("sqr", ("sqr", ("var", 0))),
        x=x,
        target_value=target_value,
        target_grad=target_grad,
        target_d2=target_d2,
        diag_weight=1.0,
    )

    assert truth is not None
    assert tied_value is not None
    assert truth.value_probe_loss == pytest.approx(0.0)
    assert tied_value.value_probe_loss == pytest.approx(0.0)
    assert truth.diag_probe_loss == pytest.approx(0.0)
    assert tied_value.diag_probe_loss is not None
    assert tied_value.diag_probe_loss > 0.0
    assert truth.probe_total < tied_value.probe_total


def test_score_local_teacher_loss_emits_physics_loss_from_constraint_flags():
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    target_node = ("sqr", ("var", 0))
    target_value = eval_node(target_node, x)
    target_grad = 2.0 * x
    target_d2 = 2.0 * torch.ones_like(x)

    truth = _score_teacher(
        target_node,
        x=x,
        target_value=target_value,
        target_grad=target_grad,
        target_d2=target_d2,
        physics_weight=1.0,
    )
    reversed_node = _score_teacher(
        ("var", 0),
        x=x,
        target_value=target_value,
        target_grad=target_grad,
        target_d2=target_d2,
        physics_weight=1.0,
    )

    assert truth is not None
    assert reversed_node is not None
    assert truth.physics_probe_loss == pytest.approx(0.0)
    assert reversed_node.physics_probe_loss is not None
    assert reversed_node.physics_probe_loss > 0.0
    assert truth.probe_total < reversed_node.probe_total


def test_score_local_teacher_loss_scales_diag_and_physics_terms_by_target_confidence():
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    target_value = eval_node(("sqr", ("var", 0)), x)
    target_grad = 2.0 * x
    target_d2 = 2.0 * torch.ones_like(x)

    high_conf = _score_teacher(
        ("var", 0),
        x=x,
        target_value=target_value,
        target_grad=target_grad,
        target_d2=target_d2,
        diag_weight=1.0,
        physics_weight=1.0,
        target_diagnostics={"confidence": 1.0, "valid_frac": 1.0},
    )
    low_conf = _score_teacher(
        ("var", 0),
        x=x,
        target_value=target_value,
        target_grad=target_grad,
        target_d2=target_d2,
        diag_weight=1.0,
        physics_weight=1.0,
        target_diagnostics={"confidence": 0.5, "valid_frac": 0.5},
    )

    assert high_conf is not None
    assert low_conf is not None
    assert high_conf.diag_probe_loss is not None
    assert low_conf.diag_probe_loss is not None
    assert high_conf.physics_probe_loss is not None
    assert low_conf.physics_probe_loss is not None
    assert low_conf.diag_probe_loss < high_conf.diag_probe_loss
    assert low_conf.physics_probe_loss < high_conf.physics_probe_loss
