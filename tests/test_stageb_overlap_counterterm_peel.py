#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for the Stage-B overlap-counterterm peel proposal."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode
from nestynet_sr.sr_search.stageB import RuleOverlapCountertermPeelNN
from nestynet_sr.sr_search.stageB.splits import _build_overlap_counterterm_peel_candidates
import nestynet_sr.sr_search.stageB.splits as stageb_splits


class _MockComposite(nn.Module):
    def __init__(self, *leaves):
        super().__init__()
        self.leaf = nn.ModuleList(list(leaves))


class _AnalyticLeaf(nn.Module):
    """Analytic leaf exposing the grad/grad_grad API expected by Stage B."""

    def __init__(self, fn, grad_fn, hess_fn, n_in: int):
        super().__init__()
        self._fn = fn
        self._grad_fn = grad_fn
        self._hess_fn = hess_fn
        self.n_in = int(n_in)

    def forward(self, x):
        y = self._fn(x)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        return y

    def grad(self, cache, allow_unused=True):
        x = cache["x"]
        g = self._grad_fn(x)
        if g.ndim == 2:
            g = g.unsqueeze(1)
        return g

    def grad_grad(self, cache):
        x = cache["x"]
        h = self._hess_fn(x)
        if h.ndim == 3:
            h = h.unsqueeze(1)
        return h


def _make_loader(x: torch.Tensor):
    y = torch.zeros(x.shape[0], 1, dtype=x.dtype)
    return DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)


def _make_left_positive_problem(dtype: torch.dtype):
    left_atom = AtomNode("nn", (0, 1), tag="L")
    right_atom = AtomNode("nn", (0, 2), tag="R")
    root = MulNode(left_atom, right_atom)

    zero = lambda t: torch.zeros_like(t[:, 0])
    left_leaf = _AnalyticLeaf(
        fn=lambda t: t[:, 0] + t[:, 1],
        grad_fn=lambda t: torch.stack([torch.ones_like(t[:, 0]), torch.ones_like(t[:, 0])], dim=-1),
        hess_fn=lambda t: torch.stack(
            [
                torch.stack([zero(t), zero(t)], dim=-1),
                torch.stack([zero(t), zero(t)], dim=-1),
            ],
            dim=1,
        ),
        n_in=2,
    )
    right_leaf = _AnalyticLeaf(
        fn=lambda t: t[:, 0] * (1.0 + t[:, 1]),
        grad_fn=lambda t: torch.stack([1.0 + t[:, 1], t[:, 0]], dim=-1),
        hess_fn=lambda t: torch.stack(
            [
                torch.stack([zero(t), torch.ones_like(t[:, 0])], dim=-1),
                torch.stack([torch.ones_like(t[:, 0]), zero(t)], dim=-1),
            ],
            dim=1,
        ),
        n_in=2,
    )
    model = _MockComposite(left_leaf, right_leaf)
    x = 0.5 + torch.rand(512, 3, dtype=dtype)
    return root, model, x


def _make_bidirectional_problem(dtype: torch.dtype):
    left_atom = AtomNode("nn", (0, 1), tag="L")
    right_atom = AtomNode("nn", (0, 2), tag="R")
    root = MulNode(left_atom, right_atom)

    zero = lambda t: torch.zeros_like(t[:, 0])
    left_leaf = _AnalyticLeaf(
        fn=lambda t: t[:, 0] + t[:, 1],
        grad_fn=lambda t: torch.stack([torch.ones_like(t[:, 0]), torch.ones_like(t[:, 0])], dim=-1),
        hess_fn=lambda t: torch.stack(
            [
                torch.stack([zero(t), zero(t)], dim=-1),
                torch.stack([zero(t), zero(t)], dim=-1),
            ],
            dim=1,
        ),
        n_in=2,
    )
    right_leaf = _AnalyticLeaf(
        fn=lambda t: t[:, 0] + 2.0 * t[:, 1],
        grad_fn=lambda t: torch.stack(
            [torch.ones_like(t[:, 0]), 2.0 * torch.ones_like(t[:, 0])], dim=-1
        ),
        hess_fn=lambda t: torch.stack(
            [
                torch.stack([zero(t), zero(t)], dim=-1),
                torch.stack([zero(t), zero(t)], dim=-1),
            ],
            dim=1,
        ),
        n_in=2,
    )
    model = _MockComposite(left_leaf, right_leaf)
    x = 0.5 + torch.rand(512, 3, dtype=dtype)
    return root, model, x


def test_overlap_counterterm_peel_positive_singleton():
    assert RuleOverlapCountertermPeelNN is not None

    dtype = torch.float64
    device = torch.device("cpu")

    root, model, x = _make_left_positive_problem(dtype)
    dl = _make_loader(x)

    candidates = _build_overlap_counterterm_peel_candidates(
        root=root,
        target=root,
        model=model,
        train_loader=dl,
        device=device,
        dtype=dtype,
        max_points=512,
    )

    assert candidates, "expected a left counterterm peel candidate"
    cand_root, _init_fn, meta = next(
        item for item in candidates if item[2]["direction"] == "left" and item[2]["peeled_var"] == 0
    )

    assert isinstance(cand_root, MulNode)
    assert isinstance(cand_root.left, AddNode)
    assert isinstance(cand_root.left.left, AtomNode)
    assert isinstance(cand_root.left.right, AtomNode)
    assert tuple(int(v) for v in cand_root.left.left.var_idxs) == (0,)
    assert tuple(int(v) for v in cand_root.left.right.var_idxs) == (1,)
    assert isinstance(cand_root.right, AtomNode)
    assert tuple(int(v) for v in cand_root.right.var_idxs) == (0, 2)
    assert meta["pattern_family"] == "overlap_counterterm_peel"


def test_overlap_counterterm_peel_right_direction_and_init_binding(monkeypatch, capsys):
    dtype = torch.float64
    device = torch.device("cpu")

    root, model, x = _make_bidirectional_problem(dtype)
    dl = _make_loader(x)
    candidates = _build_overlap_counterterm_peel_candidates(
        root=root,
        target=root,
        model=model,
        train_loader=dl,
        device=device,
        dtype=dtype,
        max_points=512,
    )

    left_item = next(
        item for item in candidates if item[2]["direction"] == "left" and item[2]["peeled_var"] == 0
    )
    right_item = next(
        item for item in candidates if item[2]["direction"] == "right" and item[2]["peeled_var"] == 0
    )

    def _fake_atom_to_leaf_map(root_new, _model_new):
        mapping = {}
        for atom in stageb_splits._collect_all_atoms(root_new):
            if isinstance(atom, AtomNode) and str(getattr(atom, "kind", "")).lower() == "nn":
                mapping[id(atom)] = nn.Linear(atom.n_in, 1, dtype=dtype)
        return mapping

    monkeypatch.setattr(stageb_splits, "build_atom_to_leaf_map", _fake_atom_to_leaf_map)

    left_root, left_init, _left_meta = left_item
    right_root, right_init, _right_meta = right_item
    left_init(left_root, object())
    right_init(right_root, object())

    out = capsys.readouterr().out
    assert "missing new leaves" not in out
    assert "overlap_counterterm_peel init (left, x0)" in out
    assert "overlap_counterterm_peel init (right, x0)" in out


def test_overlap_counterterm_peel_assigns_stay_tag_when_missing(monkeypatch, capsys):
    dtype = torch.float64
    device = torch.device("cpu")

    root, model, x = _make_left_positive_problem(dtype)
    root.right.tag = None
    dl = _make_loader(x)

    candidates = _build_overlap_counterterm_peel_candidates(
        root=root,
        target=root,
        model=model,
        train_loader=dl,
        device=device,
        dtype=dtype,
        max_points=512,
    )

    cand_root, init_fn, meta = next(
        item for item in candidates if item[2]["direction"] == "left" and item[2]["peeled_var"] == 0
    )

    assert isinstance(cand_root, MulNode)
    assert isinstance(cand_root.right, AtomNode)
    assert getattr(cand_root.right, "tag", None)
    assert meta["direction"] == "left"

    def _fake_atom_to_leaf_map(root_new, _model_new):
        mapping = {}
        for atom in stageb_splits._collect_all_atoms(root_new):
            if isinstance(atom, AtomNode) and str(getattr(atom, "kind", "")).lower() == "nn":
                mapping[id(atom)] = nn.Linear(atom.n_in, 1, dtype=dtype)
        return mapping

    monkeypatch.setattr(stageb_splits, "build_atom_to_leaf_map", _fake_atom_to_leaf_map)
    init_fn(cand_root, object())

    out = capsys.readouterr().out
    assert "missing stay leaf" not in out
    assert "stay=nan" not in out


def test_overlap_counterterm_peel_rejects_nonadditive_overlap():
    dtype = torch.float64
    device = torch.device("cpu")

    left_atom = AtomNode("nn", (0, 1), tag="L")
    right_atom = AtomNode("nn", (0, 2), tag="R")
    root = MulNode(left_atom, right_atom)

    zero = lambda t: torch.zeros_like(t[:, 0])
    left_leaf = _AnalyticLeaf(
        fn=lambda t: t[:, 0] * (1.0 + t[:, 1]),
        grad_fn=lambda t: torch.stack([1.0 + t[:, 1], t[:, 0]], dim=-1),
        hess_fn=lambda t: torch.stack(
            [
                torch.stack([zero(t), torch.ones_like(t[:, 0])], dim=-1),
                torch.stack([torch.ones_like(t[:, 0]), zero(t)], dim=-1),
            ],
            dim=1,
        ),
        n_in=2,
    )
    right_leaf = _AnalyticLeaf(
        fn=lambda t: t[:, 0] * (2.0 + t[:, 1]),
        grad_fn=lambda t: torch.stack([2.0 + t[:, 1], t[:, 0]], dim=-1),
        hess_fn=lambda t: torch.stack(
            [
                torch.stack([zero(t), torch.ones_like(t[:, 0])], dim=-1),
                torch.stack([torch.ones_like(t[:, 0]), zero(t)], dim=-1),
            ],
            dim=1,
        ),
        n_in=2,
    )
    model = _MockComposite(left_leaf, right_leaf)

    x = 0.5 + torch.rand(512, 3, dtype=dtype)
    dl = _make_loader(x)

    candidates = _build_overlap_counterterm_peel_candidates(
        root=root,
        target=root,
        model=model,
        train_loader=dl,
        device=device,
        dtype=dtype,
        max_points=512,
    )

    assert candidates == []
