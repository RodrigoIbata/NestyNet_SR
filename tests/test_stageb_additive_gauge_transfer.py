# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, Var, collect_all_atoms, get_input_exprs
from nestynet_sr.sr_search.stageB.engine import StageBContext, StageBEngine, StageBState
from nestynet_sr.sr_search.stageB.additive_gauge_scope import additive_gauge_global_score
from nestynet_sr.sr_search.stageB.rules import RuleAdditiveGaugeTransfer


def _nn(tag, *vars_):
    return AtomNode(kind="nn", var_idxs=tuple(vars_), tag=tag)


def _ctx(root, *, train_loader=None, max_features=3):
    state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=1.0e-8,
    )
    return StageBContext(
        state=state,
        train_loader=train_loader,
        val_loader=None,
        lm_hp=SimpleNamespace(
            stageB_gauge_transfer_max_features=max_features,
            stageB_gauge_transfer_min_domain_ok_frac=0.98,
        ),
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


def test_additive_gauge_transfer_targets_unresolved_direct_nn_scope():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    ctx = _ctx(root)
    rule = RuleAdditiveGaugeTransfer()

    targets = rule.iter_targets(ctx)

    assert targets == [root]


def test_additive_gauge_transfer_emits_visible_scope_simplifying_candidates():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    ctx = _ctx(root)
    rule = RuleAdditiveGaugeTransfer()

    cands = rule.propose(ctx, root)

    assert cands
    assert all(c.meta["pattern"] == "additive_gauge_transfer" for c in cands)
    assert all(c.meta["additive_gauge_confirmed"] is True for c in cands)
    assert all(c.meta["additive_gauge_requires_scope_improvement"] is True for c in cands)
    assert all(not c.meta.get("hidden_gauge_only", False) for c in cands)
    assert all(additive_gauge_global_score(c.root) < additive_gauge_global_score(root) for c in cands)


def test_additive_gauge_transfer_drops_shared_axis_from_one_representative():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    ctx = _ctx(root)
    cands = RuleAdditiveGaugeTransfer().propose(ctx, root)

    reduced_inputs = []
    for cand in cands:
        for atom in collect_all_atoms(cand.root):
            if isinstance(atom, AtomNode) and str(atom.kind).lower() == "nn":
                reduced_inputs.append(tuple(inp.var_idxs[0] for inp in get_input_exprs(atom) if inp.var_idxs))

    assert (0,) in reduced_inputs
    assert (2,) in reduced_inputs


def test_additive_gauge_transfer_skips_non_gauge_scope():
    root = AddNode(_nn("left", 0), _nn("right", 1))
    ctx = _ctx(root)

    assert RuleAdditiveGaugeTransfer().iter_targets(ctx) == []


def test_additive_gauge_transfer_is_scope_aware_for_engine_marking():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    ctx = _ctx(root)
    cand = RuleAdditiveGaugeTransfer().propose(ctx, root)[0]

    marked = StageBEngine([])._mark_gauge_tainted_candidates(
        ctx,
        "additive_gauge_transfer",
        root.left,
        [cand],
    )

    assert marked[0].meta["pattern"] == "additive_gauge_transfer"
    assert marked[0].meta["additive_gauge_confirmed"] is True


def test_additive_gauge_transfer_filters_bad_transfer_domains():
    root = AddNode(_nn("left", 0, 1), _nn("right", 1, 2))
    X = torch.tensor(
        [
            [1.0, -0.5, 2.0],
            [1.2, -0.25, 2.2],
            [1.4, -0.75, 2.4],
        ],
        dtype=torch.float64,
    )
    loader = DataLoader(TensorDataset(X, torch.zeros(X.shape[0], 1, dtype=torch.float64)), batch_size=3)
    ctx = _ctx(root, train_loader=loader, max_features=16)

    cands = RuleAdditiveGaugeTransfer().propose(ctx, root)

    assert cands
    assert all(c.meta["additive_gauge_transfer_domain_ok_frac"] >= 0.98 for c in cands)
    assert not any(c.meta["additive_gauge_transfer_basis"] == "C*log(x1)" for c in cands)


def test_additive_gauge_transfer_handles_same_analytic_prefactor_terms():
    root = AddNode(
        MulNode(Var(3), _nn("left", 0, 1)),
        MulNode(Var(3), _nn("right", 1, 2)),
    )
    ctx = _ctx(root)

    cands = RuleAdditiveGaugeTransfer().propose(ctx, root)

    assert cands
    assert all(c.meta["additive_gauge_transfer_prefactor"] == "same" for c in cands)
    assert all(c.meta["additive_gauge_transfer_shared_vars"] == (1,) for c in cands)
    assert all(additive_gauge_global_score(c.root) < additive_gauge_global_score(root) for c in cands)


def test_additive_gauge_transfer_skips_mismatched_prefactor_terms():
    root = AddNode(
        MulNode(Var(3), _nn("left", 0, 1)),
        MulNode(Var(4), _nn("right", 1, 2)),
    )
    ctx = _ctx(root)

    assert RuleAdditiveGaugeTransfer().propose(ctx, root) == []
