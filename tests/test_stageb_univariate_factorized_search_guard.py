# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AtomNode
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.stageB.rules import RuleUniNN


class _DummyFactorizedSearch:
    return_topk = 1

    def propose(self, ctx, target):
        raise AssertionError("Lazy factorized symbolic search builder should not be invoked in this unit test")


def _make_ctx(root, *, units_spec, target_dim):
    logs = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=root, reuse=False),
        scaling_by_axis={},
        trig_by_axis={},
        stageA_x_transforms={},
        xcoords_applied=False,
        train_loader_probe=None,
        device=torch.device("cpu"),
        dtype=torch.float64,
        units_spec=units_spec,
        enforce_units=True,
        lm_hp=None,
        infer_target_dim=lambda _target: target_dim,
        log=logs.append,
        _logs=logs,
    )
    return ctx


def test_univariate_factorized_search_skipped_when_target_units_unreachable():
    us = UnitSystem(("L", "T"))
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dim({"L": 2, "T": -4}),
    )
    ctx = _make_ctx(root, units_spec=units_spec, target_dim=units_spec.y_dim)

    cands = RuleUniNN(factorized_search_rule=_DummyFactorizedSearch()).propose(ctx, root)

    assert [c.label for c in cands] == []
    assert any("Skipping factorized symbolic search fallback" in msg for msg in ctx._logs)


def test_univariate_factorized_search_kept_when_declared_constants_extend_span():
    us = UnitSystem(("L", "T"))
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    target_dim = us.dim({"L": 1, "T": -1})
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=target_dim,
        free_const_dims={"time_scale": us.dim({"T": 1})},
    )
    ctx = _make_ctx(root, units_spec=units_spec, target_dim=target_dim)

    cands = RuleUniNN(factorized_search_rule=_DummyFactorizedSearch()).propose(ctx, root)
    labels = [c.label for c in cands]

    assert labels == ["factorized_search[0]", "factorized_search[1]"]
    assert not any("Skipping factorized symbolic search fallback" in msg for msg in ctx._logs)
