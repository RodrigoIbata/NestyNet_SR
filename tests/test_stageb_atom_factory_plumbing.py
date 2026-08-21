# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Regression tests for Stage-B atom_factory plumbing."""

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core import bridges as core_bridges
from nestynet_sr.sr_core.bridges import FreeConst
from nestynet_sr.sr_search.stageB import atom_mapping as stageb_atom_mapping
from nestynet_sr.sr_search.stageB import fitting as stageb_fitting
from nestynet_sr.sr_search.stageB.engine import (
    Candidate,
    StageBContext,
    StageBState,
    _Checkpoint,
    _restore_from_checkpoint,
)


class _DummyModel(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([float(value)], dtype=torch.float64))
        self.leaf = torch.nn.ModuleList([])

    def forward(self, x):
        return x[:, :1] * 0.0 + self.weight

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def _make_state(*, root=None) -> StageBState:
    if root is None:
        root = FreeConst("c", tag="c", init=1.0)
    return StageBState(
        root=root,
        model=_DummyModel(1.0),
        reuse={},
        val_loss=1.0e-3,
    )


def _make_ctx(*, train_loader, val_loader, atom_factory=None, state=None) -> StageBContext:
    if state is None:
        state = _make_state()
    lm_hp = SimpleNamespace(
        fit_y_link=None,
        fit_y_link_scale=1.0,
        loss_acceptable=1.0,
        select_stageB_max_decades_over_floor=1.0,
    )
    return StageBContext(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=5,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-3,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        atom_factory=atom_factory,
    )


def test_stageb_context_fit_candidate_forwards_atom_factory_single(monkeypatch):
    captured = {}

    def _fake_fit_candidate_root(**kwargs):
        captured["atom_factory"] = kwargs.get("atom_factory", None)
        return _make_state(root=kwargs["root"])

    monkeypatch.setattr(stageb_fitting, "_fit_candidate_root", _fake_fit_candidate_root)

    atom_factory = lambda atom, existing: None
    ctx = _make_ctx(
        train_loader=object(),
        val_loader=object(),
        atom_factory=atom_factory,
    )
    cand = Candidate(label="noop", root=ctx.state.root)

    _ = ctx.fit_candidate(cand)
    assert captured["atom_factory"] is atom_factory


def test_stageb_context_fit_candidate_forwards_atom_factory_multi(monkeypatch):
    captured = {}

    def _fake_fit_candidate_root_multi(**kwargs):
        captured["atom_factory"] = kwargs.get("atom_factory", None)
        st = _make_state(root=kwargs["root"])
        st.models = [st.model, _DummyModel(2.0)]
        st.reuses = [{}, {}]
        st.val_losses = [1.0e-3, 2.0e-3]
        return st

    monkeypatch.setattr(stageb_fitting, "_fit_candidate_root_multi", _fake_fit_candidate_root_multi)

    atom_factory_0 = lambda atom, existing: None
    atom_factory_1 = lambda atom, existing: None
    ctx = _make_ctx(
        train_loader=[object(), object()],
        val_loader=[object(), object()],
        atom_factory=[atom_factory_0, atom_factory_1],
    )
    cand = Candidate(label="noop", root=ctx.state.root)

    _ = ctx.fit_candidate(cand)
    assert captured["atom_factory"] == [atom_factory_0, atom_factory_1]


def test_restore_from_checkpoint_uses_per_dataset_atom_factory(monkeypatch):
    calls = []

    def _fake_build_composite_from_ast(root, *, atom_factory=None, **_kwargs):
        calls.append(atom_factory)
        return _DummyModel(1.0), {}

    monkeypatch.setattr(core_bridges, "build_composite_from_ast", _fake_build_composite_from_ast)
    monkeypatch.setattr(stageb_atom_mapping, "_refresh_reuse_from_state", lambda _root, _model: {})

    root = FreeConst("c", tag="c", init=1.0)
    atom_factory_0 = lambda atom, existing: None
    atom_factory_1 = lambda atom, existing: None
    ctx = _make_ctx(
        train_loader=object(),
        val_loader=object(),
        atom_factory=[atom_factory_0, atom_factory_1],
        state=_make_state(root=root),
    )
    ckpt = _Checkpoint(
        root=root,
        val_loss=1.0e-3,
        model_state_dict=_DummyModel(1.0).state_dict(),
        reuse_state_dicts=[_DummyModel(2.0).state_dict(), _DummyModel(3.0).state_dict()],
        enabled_patterns=[],
        best_val_loss=1.0e-3,
        has_structural=False,
        decision_log_len=0,
        decision_step=0,
        attempted_transformations={},
        accept_step=0,
        accept_rule="",
        accept_label="",
        accept_target="",
    )

    state = _restore_from_checkpoint(ctx, ckpt)

    # One primary rebuild (index 0), then one per reuse_state_dict entry.
    assert calls == [atom_factory_0, atom_factory_0, atom_factory_1]
    assert state.models is not None
    assert len(state.models) == 2
