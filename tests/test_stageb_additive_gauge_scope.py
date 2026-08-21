# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, ConstNode, MulNode, Var
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBContext, StageBEngine, StageBState
from nestynet_sr.sr_search.stageB.additive_gauge_scope import AdditiveGaugeScopeIndex, additive_gauge_global_score


def _nn(tag, *vars_):
    return AtomNode(kind="nn", var_idxs=tuple(vars_), tag=tag)


def _ctx(root):
    state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=1.0e-8,
    )
    return StageBContext(
        state=state,
        train_loader=None,
        val_loader=None,
        lm_hp=SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-8,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )


def test_additive_gauge_scope_index_detects_shared_additive_nn_scope():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))

    idx = AdditiveGaugeScopeIndex(root)
    score = idx.global_score()

    assert len(idx.unresolved_scopes) == 1
    scope = idx.unresolved_scopes[0]
    assert scope.shared_vars == frozenset({1})
    assert scope.unresolved_pairs == ((0, 1),)
    assert score.total_unresolved_scopes == 1
    assert score.total_nn_atoms_inside_unresolved_scopes == 2
    assert score.sum_effective_arity_sq_inside_unresolved_scopes == 8


def test_additive_gauge_scope_score_improves_when_scope_is_resolved():
    base = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    resolved = MulNode(Var(0), _nn("right", 1, 2))

    assert additive_gauge_global_score(resolved) < additive_gauge_global_score(base)


def test_gauge_acceptance_gate_rejects_marked_candidate_without_scope_improvement():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    ctx = _ctx(root)
    cand = Candidate(
        "homogeneity_peel",
        root,
        meta={
            "additive_gauge_requires_scope_improvement": True,
            "additive_gauge_score_before": ctx.additive_gauge_global_score(),
        },
    )
    cand_state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=1.0e-12,
    )

    ok, reason = ctx.gauge_acceptance_gate(cand, cand_state, "better-loss")

    assert not ok
    assert reason == "reject-unresolved-additive-gauge-local-compression"


def test_gauge_acceptance_gate_allows_marked_candidate_that_removes_all_nn():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    ctx = _ctx(root)
    cand = Candidate(
        "compound_fn_macros",
        ConstNode(0.0),
        meta={
            "additive_gauge_requires_scope_improvement": True,
            "additive_gauge_score_before": ctx.additive_gauge_global_score(),
        },
    )
    cand_state = StageBState(
        root=ConstNode(0.0),
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=1.0e-12,
    )

    ok, reason = ctx.gauge_acceptance_gate(cand, cand_state, "better-loss")

    assert ok
    assert reason == "better-loss"


def test_gauge_acceptance_gate_rejects_hidden_gauge_only_candidate():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    ctx = _ctx(root)
    cand = Candidate("additive_gauge_transfer", root, meta={"hidden_gauge_only": True})
    cand_state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=1.0e-12,
    )

    ok, reason = ctx.gauge_acceptance_gate(cand, cand_state, "better-loss")

    assert not ok
    assert reason == "reject-hidden-gauge-only"


def test_engine_marks_gauge_sensitive_candidates_inside_unresolved_scope():
    left = _nn("left", 0, 1)
    root = AddNode(left, _nn("right", 1, 2))
    ctx = _ctx(root)
    cand = Candidate("homogeneity_peel", left)

    marked = StageBEngine([])._mark_gauge_tainted_candidates(
        ctx,
        "homogeneity_peel",
        left,
        [cand],
    )

    assert marked[0].meta["additive_gauge_sensitive"] is True
    assert marked[0].meta["additive_gauge_requires_scope_improvement"] is True
    assert marked[0].meta["additive_gauge_scope_uid"] == "add:0"
    assert marked[0].meta["additive_gauge_score_before"] == ctx.additive_gauge_global_score()


def test_engine_does_not_mark_scope_aware_overlap_rules():
    left = _nn("left", 0, 1)
    root = AddNode(left, _nn("right", 1, 2))
    ctx = _ctx(root)
    cand = Candidate("common_prefactor", root)

    marked = StageBEngine([])._mark_gauge_tainted_candidates(
        ctx,
        "common_prefactor",
        left,
        [cand],
    )

    assert "additive_gauge_requires_scope_improvement" not in marked[0].meta
