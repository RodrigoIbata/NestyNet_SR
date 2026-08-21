# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    MulNode,
    Var,
    ast_to_human_readable,
    effective_arity,
    get_input_exprs,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.stageB import rules as stageb_rules
from nestynet_sr.sr_search.stageB import rules_gauge_homogeneity as homogeneity_rules


class _GaussianScaleLeaf(torch.nn.Module):
    """f(z, q) = exp(-0.5*(z/q)^2) / q."""

    def forward(self, x):
        z = x[:, 0]
        q = x[:, 1]
        u = z / q
        return (torch.exp(-0.5 * u * u) / q).unsqueeze(1)

    def grad(self, cache_or_x, allow_unused=True):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        z = x[:, 0]
        q = x[:, 1]
        f = torch.exp(-0.5 * (z / q) ** 2) / q
        df_dz = f * (-z / (q * q))
        df_dq = f * (((z / q) ** 2 - 1.0) / q)
        return torch.stack((df_dz, df_dq), dim=1).unsqueeze(1)


class _Pb001GaussianLeaf(torch.nn.Module):
    """f(q, z) = exp(-0.5*(z/q)^2) / q in pb001 variable order."""

    def forward(self, x):
        q = x[:, 0]
        z = x[:, 1]
        return (torch.exp(-0.5 * (z / q) ** 2) / q).unsqueeze(1)

    def grad(self, cache_or_x, allow_unused=True):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        q = x[:, 0]
        z = x[:, 1]
        f = torch.exp(-0.5 * (z / q) ** 2) / q
        df_dq = f * (((z / q) ** 2 - 1.0) / q)
        df_dz = f * (-z / (q * q))
        return torch.stack((df_dq, df_dz), dim=1).unsqueeze(1)


class _TrainableGaussianResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(0.25))
        self.rate = torch.nn.Parameter(torch.tensor(0.20))

    def forward(self, x):
        return self.amplitude * torch.exp(-self.rate.square() * x.square())


class _CompoundProductScaleLeaf(torch.nn.Module):
    """f(q, a, b) = q * exp(-0.5*(a*b/q)^2)."""

    def forward(self, x):
        q = x[:, 0]
        a = x[:, 1]
        b = x[:, 2]
        u = (a * b) / q
        return (q * torch.exp(-0.5 * u * u)).unsqueeze(1)


def _pb002_style_target():
    z_expr = AddNode(Var(1), MulNode(ConstNode(-1.0), Var(2)))
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf0",
        inputs=(z_expr, Var(0)),
    )


def _pb086_style_target():
    z_expr = MulNode(Var(2), Var(3))
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf0",
        inputs=(z_expr, Var(0), Var(1)),
    )


def _ctx(target, x_full, *, units_spec=None):
    return SimpleNamespace(
        state=SimpleNamespace(root=target, model=object(), reuse={"leaf0": object()}),
        train_loader_probe=[x_full],
        device=torch.device("cpu"),
        dtype=torch.float64,
        lm_hp=SimpleNamespace(stageB_homogeneity_pretrain_epochs=50),
        enforce_units=units_spec is not None,
        units_spec=units_spec,
        verbose=False,
        log=lambda *_a, **_k: None,
        infer_target_dim=lambda _target: None,
    )


def test_homogeneity_peel_certifies_prefactor_normalized_ratio(monkeypatch):
    target = _pb002_style_target()
    x0 = torch.linspace(1.0, 3.0, 700, dtype=torch.float64)
    x1 = torch.linspace(1.2, 2.9, 700, dtype=torch.float64)
    x2 = 2.0 + 0.35 * torch.sin(torch.linspace(0.0, 8.0, 700, dtype=torch.float64))
    x_full = torch.stack((x0, x1, x2), dim=1)

    monkeypatch.setattr(
        stageb_rules,
        "build_atom_to_leaf_map",
        lambda _root, _model: {id(target): _GaussianScaleLeaf()},
    )

    cands = stageb_rules.RuleHomogeneityPeel().propose(_ctx(target, x_full), target)

    assert cands
    best = min(cands, key=lambda c: c.meta["collapse_score"])
    assert best.init_fn is not None
    assert best.meta["degree"] == -1.0
    assert best.meta["separability_like"] is True
    assert best.meta["collapse_score"] < 5.0e-2

    assert isinstance(best.root, MulNode)
    residual = best.root.right if isinstance(best.root.right, AtomNode) else best.root.left
    assert isinstance(residual, AtomNode)
    assert effective_arity(residual) == 1
    ratio_expr = get_input_exprs(residual)[0]
    assert "x0" in ast_to_human_readable(ratio_expr)
    assert "x1" in ast_to_human_readable(ratio_expr)
    assert "x2" in ast_to_human_readable(ratio_expr)


def test_pb001_homogeneity_peel_builds_inverse_prefactor_ratio(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf0")
    t = torch.linspace(0.0, 1.0, 900, dtype=torch.float64)
    x0 = 1.1 + 1.8 * t
    x1 = 1.2 + 1.6 * torch.sin(9.0 * t + 0.2) ** 2
    x_full = torch.stack((x0, x1), dim=1)

    monkeypatch.setattr(
        stageb_rules,
        "build_atom_to_leaf_map",
        lambda _root, _model: {id(target): _Pb001GaussianLeaf()},
    )

    cands = stageb_rules.RuleHomogeneityPeel().propose(_ctx(target, x_full), target)

    assert cands
    best = min(cands, key=lambda c: c.meta["collapse_score"])
    assert best.init_fn is not None
    assert best.meta["degree"] == -1.0
    assert best.meta["power_dim"] == 0
    assert best.meta["ratio_dim"] == 1
    assert best.meta["collapse_score"] < 5.0e-2
    assert ast_to_human_readable(best.root) == "((x0)**-1 * NN[(x1 * (x0)**-1)])"
    residual = best.root.right
    assert isinstance(residual, AtomNode)
    assert effective_arity(residual) == 1

    residual_model = _TrainableGaussianResidual()
    ratio = (x1 / x0).reshape(-1, 1)
    target_values = torch.exp(-0.5 * ratio.square())
    with torch.no_grad():
        loss_before = torch.mean((residual_model(ratio) - target_values) ** 2)
    monkeypatch.setattr(
        homogeneity_rules,
        "build_atom_to_leaf_map",
        lambda _root, _model: {id(residual): residual_model},
    )
    best.init_fn(best.root, object())
    with torch.no_grad():
        loss_after = torch.mean((residual_model(ratio) - target_values) ** 2)
    assert loss_after < 0.30 * loss_before


def test_homogeneity_peel_synthesizes_product_ratio_for_compound_extras(monkeypatch):
    target = _pb086_style_target()
    t = torch.linspace(0.0, 1.0, 900, dtype=torch.float64)
    x0 = 1.0 + 2.0 * t
    x1 = 1.3 + 0.7 * torch.sin(7.0 * t) ** 2
    x2 = 1.2 + 2.4 * torch.cos(5.0 * t + 0.1) ** 2
    x3 = 1.1 + 1.8 * torch.sin(11.0 * t + 0.3) ** 2
    x_full = torch.stack((x0, x1, x2, x3), dim=1)

    monkeypatch.setattr(
        stageb_rules,
        "build_atom_to_leaf_map",
        lambda _root, _model: {id(target): _CompoundProductScaleLeaf()},
    )

    cands = stageb_rules.RuleHomogeneityPeel().propose(_ctx(target, x_full), target)

    assert cands
    best = min(cands, key=lambda c: c.meta["collapse_score"])
    assert best.init_fn is not None
    assert best.meta["mode"] == "compound_product_ratio"
    assert best.meta["degree"] == 1.0
    assert best.meta["power_dim"] == 0
    assert best.meta["numerator_dims"] == (1, 2)
    assert best.meta["collapse_score"] < 5.0e-2

    assert isinstance(best.root, MulNode)
    residual = best.root.right if isinstance(best.root.right, AtomNode) else best.root.left
    assert isinstance(residual, AtomNode)
    assert effective_arity(residual) == 1
    ratio_expr = get_input_exprs(residual)[0]
    ratio_hr = ast_to_human_readable(ratio_expr)
    for token in ("x0", "x1", "x2", "x3"):
        assert token in ratio_hr


def test_homogeneity_ratio_units_require_compatible_inputs():
    target = _pb002_style_target()
    us = UnitSystem(("L", "T"))
    ok_spec = UnitsSpec(
        unit_system=us,
        y_dim=us.dim({}),
        x_dims=(us.dim({"L": 1}), us.dim({"L": 1}), us.dim({"L": 1})),
    )
    bad_spec = UnitsSpec(
        unit_system=us,
        y_dim=us.dim({}),
        x_dims=(us.dim({"T": 1}), us.dim({"L": 1}), us.dim({"L": 1})),
    )

    assert stageb_rules._homogeneity_ratio_units_ok(
        _ctx(target, torch.empty(0, 3, dtype=torch.float64), units_spec=ok_spec),
        target,
        power_dim=1,
        ratio_dim=0,
    )
    assert not stageb_rules._homogeneity_ratio_units_ok(
        _ctx(target, torch.empty(0, 3, dtype=torch.float64), units_spec=bad_spec),
        target,
        power_dim=1,
        ratio_dim=0,
    )


def test_homogeneity_product_ratio_units_require_dimensionless_ratio():
    target = _pb086_style_target()
    us = UnitSystem(("L", "T"))
    ok_spec = UnitsSpec(
        unit_system=us,
        y_dim=us.dim({"L": 1, "T": 1}),
        x_dims=(
            us.dim({"L": 1}),
            us.dim({"T": 1}),
            us.dim({"L": 1}),
            us.dim({"T": 1}),
        ),
    )
    bad_spec = UnitsSpec(
        unit_system=us,
        y_dim=us.dim({"L": 1, "T": 1}),
        x_dims=(
            us.dim({"L": 1}),
            us.dim({"L": 1}),
            us.dim({"L": 1}),
            us.dim({"T": 1}),
        ),
    )

    assert stageb_rules._homogeneity_product_ratio_units_ok(
        _ctx(target, torch.empty(0, 4, dtype=torch.float64), units_spec=ok_spec),
        target,
        power_dim=0,
        numerator_dims=(1, 2),
    )
    assert not stageb_rules._homogeneity_product_ratio_units_ok(
        _ctx(target, torch.empty(0, 4, dtype=torch.float64), units_spec=bad_spec),
        target,
        power_dim=0,
        numerator_dims=(1, 2),
    )
