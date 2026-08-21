# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import torch

from nestynet_sr.sr_search.factorized_search.explorer import (
    eval_node,
    invert_context_target,
    invert_context_target_beam,
)


def test_inverse_conditioning_confidence_penalizes_small_divisor():
    ast = ("mul", ("var", 0), ("var", 1))

    x_good = torch.zeros((64, 2), dtype=torch.float64)
    x_good[:, 0] = torch.linspace(0.5, 2.5, 64, dtype=torch.float64)
    x_good[:, 1] = 2.0
    y_good = eval_node(ast, x_good)

    x_bad = x_good.clone()
    x_bad[:, 1] = 1.0e-4
    y_bad = eval_node(ast, x_bad)

    inv_good = invert_context_target(
        ast,
        (1,),
        x_good,
        y_good,
        mapping={"kind": "identity"},
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.01,
    )
    inv_bad = invert_context_target(
        ast,
        (1,),
        x_bad,
        y_bad,
        mapping={"kind": "identity"},
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.01,
    )

    assert float(inv_good.confidence) > float(inv_bad.confidence)
    assert float(inv_bad.confidence) < 0.2

    mg = inv_good.valid_mask.squeeze(-1)
    mb = inv_bad.valid_mask.squeeze(-1)
    pw_good = float(inv_good.point_weight[mg].mean().item()) if int(mg.sum().item()) > 0 else 0.0
    pw_bad = float(inv_bad.point_weight[mb].mean().item()) if int(mb.sum().item()) > 0 else 0.0
    assert pw_good > pw_bad
    assert pw_bad < 0.2


def test_inverse_branch_beam_emits_alternate_square_branches():
    ast = ("sqr", ("var", 0))
    x = torch.linspace(0.2, 2.0, 32, dtype=torch.float64).unsqueeze(-1)
    y = eval_node(ast, x)

    targets = invert_context_target_beam(
        ast,
        (1,),
        x,
        y,
        mapping={"kind": "identity"},
        confidence_mode="conditioning",
        confidence_target_gain=4.0,
        confidence_floor=0.01,
        branch_beam_width=2,
    )

    assert len(targets) >= 2
    branch_ids = {str(t.branch_id) for t in targets}
    assert "main" in branch_ids
    assert any(b != "main" for b in branch_ids)
