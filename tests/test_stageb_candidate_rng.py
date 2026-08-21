# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import random
from types import SimpleNamespace

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import AtomNode
from nestynet_sr.sr_search.stageB import fitting
from nestynet_sr.sr_search.stageB.engine import StageBState


def _call_fit(root, init_fn):
    return fitting._fit_candidate_root(
        root=root,
        reuse={},
        train_loader=(),
        val_loader=(),
        lm_hp=SimpleNamespace(candidate_seed_base=1234),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        custom_init_fn=init_fn,
    )


def test_candidate_seed_is_semantic_and_start_specific():
    root = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf0")
    init_fn = lambda *_args: None
    init_fn._candidate_seed_key = "sqrt:add:(0,)|(1,)"

    seed0 = fitting._stageB_candidate_local_seed(
        root, init_fn, start_idx=0, base_seed=1234
    )
    seed0_again = fitting._stageB_candidate_local_seed(
        root, init_fn, start_idx=0, base_seed=1234
    )
    seed1 = fitting._stageB_candidate_local_seed(
        root, init_fn, start_idx=1, base_seed=1234
    )

    assert seed0 == seed0_again
    assert seed0 != seed1


def test_nonfinite_candidate_gets_bounded_stable_retry_without_rng_leak(monkeypatch):
    root = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf0")
    init_fn = lambda *_args: None
    init_fn._candidate_seed_key = "pb042-sqrt-split"
    init_fn._candidate_max_starts = 3
    init_fn._candidate_retry_nonfinite = True
    draws = []

    def fake_fit_once(**kwargs):
        del kwargs
        draws.append((random.random(), np.random.random(), float(torch.rand(()))))
        loss = float("inf") if len(draws) == 1 else 0.125
        return StageBState(root=root, model=None, reuse={}, val_loss=loss)

    monkeypatch.setattr(fitting, "_fit_candidate_root_once", fake_fit_once)

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    state = _call_fit(root, init_fn)

    assert state.val_loss == 0.125
    assert len(draws) == 2
    assert draws[0] != draws[1]
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_finite_candidate_does_not_pay_for_extra_starts(monkeypatch):
    root = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    init_fn = lambda *_args: None
    init_fn._candidate_seed_key = "finite-first-start"
    init_fn._candidate_max_starts = 3
    init_fn._candidate_retry_nonfinite = True
    calls = []

    def fake_fit_once(**kwargs):
        del kwargs
        calls.append(1)
        return StageBState(root=root, model=None, reuse={}, val_loss=0.25)

    monkeypatch.setattr(fitting, "_fit_candidate_root_once", fake_fit_once)

    state = _call_fit(root, init_fn)

    assert state.val_loss == 0.25
    assert len(calls) == 1
