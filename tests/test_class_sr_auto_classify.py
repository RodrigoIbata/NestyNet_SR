# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, FreeConst, Scale
from nestynet_sr.sr_search.class_sr import auto_classify_atoms


class _ScalarLeaf(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([float(value)], dtype=torch.float64))


class _DummyComposite(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.leaf = nn.ModuleList([_ScalarLeaf(v) for v in values])


def _mk_states(*leaf_vectors):
    return [SimpleNamespace(model=_DummyComposite(v)) for v in leaf_vectors]


def test_auto_classify_excludes_scale_leaves_by_default():
    # leaf0: nuisance scale, leaf1: analytic parameter
    root = AddNode(
        Scale(name="s", tag="s"),
        AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p"),
    )
    states = _mk_states(
        [1.00, 2.00],
        [1.01, 2.02],
        [0.99, 1.98],
    )

    _, class_tags, experiment_tags, _ = auto_classify_atoms(
        root,
        states,
        cv_threshold=0.2,
        focus_free_const_leaves=False,
    )

    assert "s" in experiment_tags
    assert "s" not in class_tags
    assert "p" in class_tags


def test_auto_classify_can_include_scale_leaves():
    root = AddNode(
        Scale(name="s", tag="s"),
        AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p"),
    )
    states = _mk_states(
        [1.00, 2.00],
        [1.01, 2.02],
        [0.99, 1.98],
    )

    _, class_tags, experiment_tags, _ = auto_classify_atoms(
        root,
        states,
        cv_threshold=0.2,
        exclude_scale_leaves=False,
        focus_free_const_leaves=False,
    )

    assert "s" in class_tags
    assert "s" not in experiment_tags


def test_auto_classify_uses_robust_cv_near_zero_mean():
    root = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p")
    # Mean cancels near zero across datasets; robust denominator should avoid CV blow-up.
    states = _mk_states(
        [1.0e-6],
        [-1.0e-6],
    )

    _, class_tags, experiment_tags, cv_per_tag = auto_classify_atoms(
        root,
        states,
        cv_threshold=2.0,
        focus_free_const_leaves=False,
    )

    assert torch.isfinite(torch.tensor(cv_per_tag["p"]))
    assert cv_per_tag["p"] < 10.0
    assert "p" in class_tags
    assert "p" not in experiment_tags


def test_auto_classify_default_focuses_on_free_consts():
    root = AddNode(
        FreeConst("k", tag="k", init=1.0, scope="class"),
        AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p"),
    )
    states = _mk_states(
        [2.0, 1.0],
        [2.01, 1.02],
        [1.99, 0.98],
    )

    _, class_tags, experiment_tags, _ = auto_classify_atoms(
        root,
        states,
        cv_threshold=0.2,
    )

    assert "k" in class_tags
    assert "k" not in experiment_tags
    assert "p" in experiment_tags
    assert "p" not in class_tags
