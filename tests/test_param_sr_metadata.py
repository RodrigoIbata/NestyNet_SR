# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import pytest
import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import FreeConst
from nestynet_sr.sr_search.param_sr import (
    ParamInvariant,
    ParamScalarRef,
    discover_param_invariants,
    evaluate_invariant_on_composite,
)


class _ScalarLeaf(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([float(value)], dtype=torch.float64))


class _DummyComposite(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.leaf = nn.ModuleList([_ScalarLeaf(value)])


def test_discover_param_invariants_includes_metadata_scalars():
    root = FreeConst("k", tag="k", init=1.0, scope="experiment")
    models = [_DummyComposite(2.0), _DummyComposite(4.0), _DummyComposite(6.0)]
    dataset_metadata = [
        {"temp": 1.0},
        {"temp": 2.0},
        {"temp": 3.0},
    ]

    refs, invariants = discover_param_invariants(
        root=root,
        models=models,
        dataset_metadata=dataset_metadata,
        candidate_tags=["k"],
        max_scalars=8,
        max_invariants=8,
        score_threshold=1.0e-6,
    )

    assert any(r.source == "metadata" and r.meta_key == "temp" for r in refs)
    assert any(inv.expr == "k#p0/meta:temp" for inv in invariants)


def test_evaluate_invariant_on_composite_with_metadata():
    comp = _DummyComposite(6.0)
    refs = [
        ParamScalarRef(key="k#p0", tag="k", source="param", flat_index=0),
        ParamScalarRef(
            key="meta:temp",
            tag="__meta__",
            source="metadata",
            flat_index=0,
            meta_key="temp",
        ),
    ]
    inv = ParamInvariant(expr="k#p0/meta:temp", op="div", a=0, b=1)
    tag_to_leafidx = {"k": 0}

    v = evaluate_invariant_on_composite(
        comp=comp,
        tag_to_leafidx=tag_to_leafidx,
        refs=refs,
        invariant=inv,
        dataset_metadata={"temp": 3.0},
    )
    assert v is not None
    assert abs(float(v.detach().cpu().item()) - 2.0) < 1.0e-12


def test_discover_param_invariants_metadata_length_mismatch_raises():
    root = FreeConst("k", tag="k", init=1.0, scope="experiment")
    models = [_DummyComposite(2.0), _DummyComposite(4.0), _DummyComposite(6.0)]

    with pytest.raises(ValueError):
        discover_param_invariants(
            root=root,
            models=models,
            dataset_metadata=[{"temp": 1.0}, {"temp": 2.0}],
            candidate_tags=["k"],
            max_scalars=8,
            max_invariants=8,
            score_threshold=1.0e-6,
        )
