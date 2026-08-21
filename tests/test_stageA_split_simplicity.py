# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_core.bridges import MulNode, PowNode, SinNode, Var
from nestynet_sr.sr_search.search import (
    _stageA_has_meaningful_loss_improvement,
    _stageA_split_simplicity_score,
)


def test_stageA_split_simplicity_prefers_clean_visible_split_over_retained_axis_rewrite():
    """pb011-style: NN[q, x1] -> NN[q]+NN[x1] beats NN[q/x1]*NN[x1]."""
    q = MulNode(MulNode(Var(2), Var(3)), SinNode(Var(4)))
    baseline_score = _stageA_split_simplicity_score(
        sep_cands=[(torch.add, ("z",), (1,), None, None)],
        z_expr=q,
        extra_var_idxs=[1],
        retained_axis_wrapper=False,
        same_arity_coordinate=False,
    )

    q_over_x1 = MulNode(q, PowNode(Var(1), -1.0))
    candidate_score = _stageA_split_simplicity_score(
        sep_cands=[(torch.multiply, ("z",), (1,), None, None)],
        z_expr=q_over_x1,
        extra_var_idxs=[1],
        retained_axis_wrapper=True,
        same_arity_coordinate=True,
    )

    assert baseline_score is not None
    assert candidate_score is not None
    assert baseline_score < candidate_score


def test_stageA_split_simplicity_allows_strictly_lower_overlap_coordinate():
    """The score is structural, not an unconditional ban on rewritten coordinates."""
    q_with_overlap = MulNode(MulNode(Var(0), Var(1)), Var(2))
    baseline_score = _stageA_split_simplicity_score(
        sep_cands=[(torch.multiply, ("z",), (1,), None, None)],
        z_expr=q_with_overlap,
        extra_var_idxs=[1],
        retained_axis_wrapper=True,
        same_arity_coordinate=True,
    )

    q_clean = MulNode(Var(0), Var(2))
    candidate_score = _stageA_split_simplicity_score(
        sep_cands=[(torch.add, ("z",), (1,), None, None)],
        z_expr=q_clean,
        extra_var_idxs=[1],
        retained_axis_wrapper=False,
        same_arity_coordinate=False,
    )

    assert baseline_score is not None
    assert candidate_score is not None
    assert candidate_score < baseline_score


def test_stageA_loss_improvement_respects_existing_floor_semantics():
    assert not _stageA_has_meaningful_loss_improvement(
        cand_loss=1.0e-9,
        reference_loss=5.0e-9,
        loss_floor=1.0e-7,
        noise_floor=0.0,
    )
    assert _stageA_has_meaningful_loss_improvement(
        cand_loss=1.0e-5,
        reference_loss=1.0e-3,
        loss_floor=1.0e-7,
        noise_floor=0.0,
    )
    assert not _stageA_has_meaningful_loss_improvement(
        cand_loss=1.0e-3,
        reference_loss=1.0e-5,
        loss_floor=1.0e-7,
        noise_floor=0.0,
    )


def test_stageA_loss_improvement_uses_noise_floor_excess_space():
    assert not _stageA_has_meaningful_loss_improvement(
        cand_loss=1.01,
        reference_loss=1.02,
        loss_floor=0.1,
        noise_floor=2.0,
    )
    assert _stageA_has_meaningful_loss_improvement(
        cand_loss=2.1,
        reference_loss=3.0,
        loss_floor=0.1,
        noise_floor=2.0,
    )
