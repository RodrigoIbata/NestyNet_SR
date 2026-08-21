# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import math
from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    MulNode,
    PowNode,
    Var,
    ast_to_human_readable,
    effective_arity,
    get_input_exprs,
)
from nestynet_sr.sr_search import candidate_builders
from nestynet_sr.sr_search.stageB import rules as stageb_rules


def _compound_bivariate_target():
    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf0",
        inputs=(z_expr, Var(2)),
    )


def _compound_multivariate_target():
    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf0",
        inputs=(z_expr, Var(2), Var(3)),
    )


class _BivariateRatioLeaf(torch.nn.Module):
    def forward(self, x):
        return (x[:, 1] / x[:, 0]).unsqueeze(1)

    def grad(self, cache_or_x, allow_unused=True):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        x0 = x[:, 0]
        x1 = x[:, 1]
        g = torch.stack((-x1 / (x0 ** 2), 1.0 / x0), dim=1)
        return g.unsqueeze(1)


class _MultivariateRatioLeaf(torch.nn.Module):
    def forward(self, x):
        return (x[:, 2] / x[:, 1]).unsqueeze(1)

    def grad(self, cache_or_x, allow_unused=True):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        x1 = x[:, 1]
        x2 = x[:, 2]
        g0 = torch.zeros_like(x1)
        g1 = -x2 / (x1 ** 2)
        g2 = 1.0 / x1
        g = torch.stack((g0, g1, g2), dim=1)
        return g.unsqueeze(1)


def test_build_ratio_invariance_candidate_uses_compound_effective_inputs(monkeypatch):
    target = _compound_bivariate_target()
    X = torch.tensor(
        [[2.0, 4.5], [3.0, 5.0], [4.0, 7.0], [5.0, 11.0]],
        dtype=torch.float64,
    )
    F = 1.0 + X[:, 1] / X[:, 0]

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    cand_root, init_fn = candidate_builders._build_ratio_invariance_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        xi_local_idx=0,
        xj_local_idx=1,
        degree=1,
        exponent=1.0,
        min_points=1,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    assert isinstance(cand_root, PowNode)
    assert cand_root.base.kind == "ratio_poly"
    ratio_inputs = get_input_exprs(cand_root.base)
    assert len(ratio_inputs) == 2
    assert isinstance(ratio_inputs[0], AtomNode)
    assert tuple(ratio_inputs[0].var_idxs) == (2,)
    assert ast_to_human_readable(ratio_inputs[1]) == ast_to_human_readable(get_input_exprs(target)[0])


def test_rule_ratio_invariance_uses_effective_arity_for_compound_bivariate_target(monkeypatch):
    target = _compound_bivariate_target()
    x0 = torch.linspace(1.0, 10.0, 600, dtype=torch.float64)
    x1 = torch.ones_like(x0)
    x2 = torch.linspace(2.0, 20.0, 600, dtype=torch.float64) + 0.3 * x0
    X_full = torch.stack((x0, x1, x2), dim=1)
    calls = []

    monkeypatch.setattr(
        stageb_rules,
        "build_atom_to_leaf_map",
        lambda _root, _model: {id(target): _BivariateRatioLeaf()},
    )

    def _capture_builder(**kwargs):
        calls.append((kwargs["xi_local_idx"], kwargs["xj_local_idx"]))
        return target, None

    monkeypatch.setattr(stageb_rules, "_build_ratio_invariance_candidate", _capture_builder)

    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, model=object(), reuse={"leaf0": object()}),
        train_loader_probe=[X_full],
        device=torch.device("cpu"),
        dtype=torch.float64,
        log=lambda *_a, **_k: None,
    )

    cands = stageb_rules.RuleRatioInvariance().propose(ctx, target)

    assert cands
    assert calls
    assert set(calls).issubset({(0, 1), (1, 0)})


def test_rule_ratio_invariance_collapses_compound_multivar_in_local_coordinates(monkeypatch):
    target = _compound_multivariate_target()
    x0 = torch.linspace(1.0, 10.0, 600, dtype=torch.float64)
    x1 = torch.ones_like(x0)
    x2 = torch.linspace(2.0, 20.0, 600, dtype=torch.float64)
    x3 = 1.5 * x2 + 0.25 * x0
    X_full = torch.stack((x0, x1, x2, x3), dim=1)

    monkeypatch.setattr(
        stageb_rules,
        "build_atom_to_leaf_map",
        lambda _root, _model: {id(target): _MultivariateRatioLeaf()},
    )

    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, model=object(), reuse={"leaf0": object()}),
        train_loader_probe=[X_full],
        device=torch.device("cpu"),
        dtype=torch.float64,
        log=lambda *_a, **_k: None,
    )

    cands = stageb_rules.RuleRatioInvariance().propose(ctx, target)

    assert cands
    new_atom = cands[0].root
    assert new_atom.kind == "nn"
    assert effective_arity(new_atom) == 2
    new_inputs = get_input_exprs(new_atom)
    assert len(new_inputs) == 2
    ratio_expr = new_inputs[0]
    assert isinstance(ratio_expr, MulNode)
    assert ast_to_human_readable(ratio_expr.left) == "x3"
    assert isinstance(ratio_expr.right, PowNode)
    assert ast_to_human_readable(ratio_expr.right.base) == "x2"
    assert math.isclose(float(ratio_expr.right.exponent), -1.0)
    assert ast_to_human_readable(new_inputs[1]) == ast_to_human_readable(get_input_exprs(target)[0])
