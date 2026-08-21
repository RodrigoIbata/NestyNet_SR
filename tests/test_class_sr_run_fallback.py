# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, Scale
from nestynet_sr.sr_search import class_sr as class_sr_mod
from nestynet_sr.sr_search.class_sr import ClassSRResult, run_class_sr


class _ScalarLeaf(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([float(value)], dtype=torch.float64))


class _DummyComposite(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.leaf = nn.ModuleList([_ScalarLeaf(v) for v in values])


def _mk_states(*leaf_vectors):
    return [
        SimpleNamespace(
            model=_DummyComposite(v),
            val_loss=1.0e-6,
            reuse={},
        )
        for v in leaf_vectors
    ]


def test_run_class_sr_relaxed_auto_retry_recovers_shared_tags(monkeypatch):
    # No free_const leaves: strict auto mode (free_const-only + no scales)
    # initially classifies nothing as shared.
    root = AddNode(
        Scale(name="s", tag="s"),
        AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p"),
    )
    states = _mk_states(
        [1.00, 2.00],
        [1.01, 2.02],
        [0.99, 1.98],
    )

    def _fake_eval_models(**kwargs):
        d = len(kwargs["models"])
        return [1.0e-6] * d, 1.0e-6, "mean"

    fit_calls = []

    def _fake_fit_class_sr_joint(*, root, states, class_tags, experiment_tags, **kwargs):
        fit_calls.append(list(class_tags))
        d = len(states)
        return ClassSRResult(
            root=root,
            composites=[st.model for st in states],
            val_losses=[1.0e-6] * d,
            val_loss_agg=1.0e-6,
            val_loss_agg_mode="mean",
            class_tags=list(class_tags),
            experiment_tags=list(experiment_tags),
            class_params={},
            experiment_params=[{} for _ in range(d)],
            cv_per_tag=dict(kwargs.get("cv_per_tag", {})),
            derived_invariants=[],
        )

    monkeypatch.setattr(class_sr_mod, "_evaluate_models_on_loaders", _fake_eval_models)
    monkeypatch.setattr(class_sr_mod, "fit_class_sr_joint", _fake_fit_class_sr_joint)

    result = run_class_sr(
        root=root,
        states=states,
        train_loaders=[object(), object(), object()],
        val_loaders=[object(), object(), object()],
        device=torch.device("cpu"),
        dtype=torch.float64,
        cv_threshold=0.2,
        param_sr_enable=False,
        max_epochs=10,
    )

    assert set(result.class_tags) == {"s", "p"}
    assert fit_calls
    assert any(("p" in tags) for tags in fit_calls)
