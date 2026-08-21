# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Stall-retry policy for Stage-B candidate fits (pb119-class rescue).

A candidate whose family provably contains the teacher (strong substitution
screen) can still lose its single LM fit to a bad basin.  The opt-in
``_candidate_retry_stall_loss`` attribute lets ``_fit_candidate_root`` restart
such fits with jittered inits, keeping the best state.  Candidates without the
attribute must keep the historical single-start behavior.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

import nestynet_sr.sr_search.stageB.fitting as fitting
from nestynet_sr.sr_core.bridges import AtomNode, AddNode


def _root():
    return AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")


def _run_fit(monkeypatch, losses, custom_init_fn):
    calls = {"count": 0, "start_idxs": []}

    def fake_once(**kwargs):
        idx = calls["count"]
        calls["count"] += 1
        calls["start_idxs"].append(
            int(getattr(kwargs["custom_init_fn"], "_candidate_start_idx", -1))
        )
        loss = losses[min(idx, len(losses) - 1)]
        return SimpleNamespace(val_loss=loss)

    monkeypatch.setattr(fitting, "_fit_candidate_root_once", fake_once)
    state = fitting._fit_candidate_root(
        root=_root(),
        reuse={},
        train_loader=None,
        val_loader=None,
        lm_hp=SimpleNamespace(candidate_seed_base=1234),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        custom_init_fn=custom_init_fn,
    )
    return state, calls


def test_stall_retry_runs_extra_starts_and_keeps_best(monkeypatch):
    def init_fn(root, model):
        pass

    init_fn._candidate_max_starts = 3
    init_fn._candidate_retry_nonfinite = True
    init_fn._candidate_retry_stall_loss = 1.0e-7

    state, calls = _run_fit(monkeypatch, [1.0e-3, 1.0e-5, 1.0e-9], init_fn)
    assert calls["count"] == 3
    assert calls["start_idxs"] == [0, 1, 2]
    assert state.val_loss == 1.0e-9


def test_no_retry_without_opt_in(monkeypatch):
    def init_fn(root, model):
        pass

    state, calls = _run_fit(monkeypatch, [1.0e-3], init_fn)
    assert calls["count"] == 1
    assert state.val_loss == 1.0e-3


def test_stops_immediately_when_below_threshold(monkeypatch):
    def init_fn(root, model):
        pass

    init_fn._candidate_max_starts = 3
    init_fn._candidate_retry_stall_loss = 1.0e-7

    state, calls = _run_fit(monkeypatch, [1.0e-9, 1.0e-3], init_fn)
    assert calls["count"] == 1
    assert state.val_loss == 1.0e-9


def test_stall_retry_keeps_best_even_if_later_starts_worse(monkeypatch):
    def init_fn(root, model):
        pass

    init_fn._candidate_max_starts = 3
    init_fn._candidate_retry_stall_loss = 1.0e-7

    state, calls = _run_fit(monkeypatch, [1.0e-4, 1.0e-2, 5.0e-3], init_fn)
    assert calls["count"] == 3
    assert state.val_loss == 1.0e-4


def test_jitter_new_leaves_targets_only_new_leaves():
    atom_reused = AtomNode(kind="poly", var_idxs=(0,), kwargs={}, tag="kept")
    atom_new = AtomNode(kind="poly", var_idxs=(1,), kwargs={}, tag="fresh")
    root = AddNode(atom_reused, atom_new)

    leaf_reused = nn.Linear(1, 1)
    leaf_new = nn.Linear(1, 1)
    before_reused = leaf_reused.weight.detach().clone()
    before_new = leaf_new.weight.detach().clone()

    atom_to_leaf = {id(atom_reused): leaf_reused, id(atom_new): leaf_new}
    reuse = {"kept": leaf_reused}

    torch.manual_seed(0)
    n = fitting._jitter_new_leaves(root, atom_to_leaf, reuse, scale=0.1)
    assert n == 1
    assert torch.equal(leaf_reused.weight, before_reused)
    assert not torch.equal(leaf_new.weight, before_new)
