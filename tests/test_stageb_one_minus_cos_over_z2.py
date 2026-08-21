# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import math
from types import SimpleNamespace

import torch

from nestynet_sr.sr_core import collect_nn_atoms
from nestynet_sr.sr_core.bridges import AtomNode, ast_to_human_readable
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.stageB import rules as stageb_rules


def _ctx(root, *, enforce_units=False, units_spec=None, y_dim=None):
    return SimpleNamespace(
        state=SimpleNamespace(root=root, model=torch.nn.Identity(), reuse={}),
        train_loader_probe=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        lm_hp=SimpleNamespace(),
        enforce_units=bool(enforce_units),
        units_spec=units_spec,
        infer_target_dim=lambda _target: y_dim,
    )


def test_one_minus_cos_over_z2_exact_candidate_from_trig_hint(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(1.0, 3.0, 512, dtype=torch.float64).unsqueeze(1)
    y = 2.0 * (1.0 - torch.cos(x[:, 0])) / (x[:, 0] ** 2)

    monkeypatch.setattr(
        stageb_rules,
        "_stageB_target_raw_teacher_data",
        lambda *_args, **_kwargs: (x, y),
    )

    trig = SimpleNamespace(axis=0, omega=1.0, rel_std=0.01)
    cands = stageb_rules._build_one_minus_cos_over_z2_candidates(
        _ctx(target),
        target,
        trig,
    )

    assert cands
    assert cands[0].label == "one_minus_cos_over_z2"
    assert cands[0].meta["precheck_rel_rms"] < 1.0e-10
    rendered = ast_to_human_readable(cands[0].root)
    assert "cos(x0)" in rendered
    assert "(x0)**-2" in rendered


def test_one_minus_cos_over_z2_rejects_unitful_phase(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    units_spec = UnitsSpec(unit_system=us, x_dims=(L,), y_dim=dimless)

    monkeypatch.setattr(
        stageb_rules,
        "_stageB_target_raw_teacher_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data screen should not run")),
    )

    trig = SimpleNamespace(axis=0, omega=1.0, rel_std=0.01)
    cands = stageb_rules._build_one_minus_cos_over_z2_candidates(
        _ctx(target, enforce_units=True, units_spec=units_spec, y_dim=dimless),
        target,
        trig,
    )

    assert cands == []


class _CosTeacher(torch.nn.Module):
    def forward(self, x):
        return torch.cos(x[:, :1])


class _OneMinusCosTeacher(torch.nn.Module):
    def forward(self, x):
        return 2.0 * (1.0 - torch.cos(x[:, :1]))


def _fixed_trig_ctx(root, teacher):
    x = torch.linspace(0.2, 2.8, 512, dtype=torch.float64).unsqueeze(1)
    y = torch.zeros_like(x)
    ctx = _ctx(root)
    ctx.state.reuse = {"leaf0": teacher}
    ctx.train_loader_probe = [(x, y)]
    return ctx


def test_fixed_trig_factor_candidate_closes_cos_leaf_before_generic_rational():
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    trig = SimpleNamespace(axis=0, omega=1.0, trig_fn="cos", basis_fn="cos", rel_std=0.001)

    cands = stageb_rules._build_fixed_trig_factor_candidates(
        _fixed_trig_ctx(target, _CosTeacher()),
        target,
        trig,
    )

    labels = [str(c.label) for c in cands]
    assert "fixed_trig_factor_cos" in labels
    cand = cands[labels.index("fixed_trig_factor_cos")]
    assert cand.meta["min_free_params"] == 1
    assert cand.meta["exact_non_generic"] is True
    assert cand.meta["screen_rel_rms"] < 1.0e-12
    assert collect_nn_atoms(cand.root) == []
    assert "cos(x0)" in ast_to_human_readable(cand.root)


def test_fixed_trig_factor_candidate_supports_one_minus_cos_leaf():
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    trig = SimpleNamespace(axis=0, omega=1.0, trig_fn="cos", basis_fn="one_minus_cos", rel_std=0.001)

    cands = stageb_rules._build_fixed_trig_factor_candidates(
        _fixed_trig_ctx(target, _OneMinusCosTeacher()),
        target,
        trig,
    )

    labels = [str(c.label) for c in cands]
    assert "fixed_trig_factor_one_minus_cos" in labels
    cand = cands[labels.index("fixed_trig_factor_one_minus_cos")]
    assert cand.meta["screen_rel_rms"] < 1.0e-12
    rendered = ast_to_human_readable(cand.root)
    assert "1" in rendered
    assert "cos(x0)" in rendered


def test_univariate_rule_offers_fixed_trig_factor_before_ratpoly(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    trig = SimpleNamespace(axis=0, omega=1.0, trig_fn="cos", basis_fn="cos", rel_std=0.001)
    ctx = _fixed_trig_ctx(target, _CosTeacher())
    ctx.scaling_by_axis = {}
    ctx.trig_by_axis = {0: trig}
    ctx.stageA_x_transforms = {}
    ctx.xcoords_applied = False
    ctx.is_pattern_disabled = lambda _pattern: False
    ctx.verbose = False
    ctx.log = lambda _msg: None
    ctx.phase_hints = []
    ctx.state.reuses = None
    ctx.lm_hp.stageB_leaf_transforms_enable = False

    monkeypatch.setattr(
        stageb_rules,
        "_build_ratpoly_1d_candidates",
        lambda **_kwargs: [(target, None, {"log": "ratpoly sentinel"})],
    )

    cands = stageb_rules.RuleUniNN().propose(ctx, target)
    labels = [str(c.label) for c in cands]

    assert "fixed_trig_factor_cos" in labels
    assert "ratpoly_1d" in labels
    assert labels.index("fixed_trig_factor_cos") < labels.index("ratpoly_1d")


def test_sparse_factor_1d_exact_candidate_before_generic_ratpoly(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(0.2, 0.8, 900, dtype=torch.float64).unsqueeze(1)
    z = x[:, 0]
    y = -z.pow(4) / (4.0 * math.pi * (1.0 - z.square()).square())

    monkeypatch.setattr(
        stageb_rules,
        "_stageB_target_raw_teacher_data",
        lambda *_args, **_kwargs: (x, y),
    )
    monkeypatch.setattr(
        stageb_rules,
        "_build_ratpoly_1d_candidates",
        lambda **_kwargs: [(target, None, {"log": "ratpoly sentinel"})],
    )

    ctx = _ctx(target)
    ctx.scaling_by_axis = {}
    ctx.trig_by_axis = {}
    ctx.stageA_x_transforms = {}
    ctx.xcoords_applied = False
    ctx.is_pattern_disabled = lambda _pattern: False
    ctx.verbose = False
    ctx.log = lambda _msg: None
    ctx.phase_hints = []
    ctx.state.reuses = None
    ctx.lm_hp.stageB_leaf_transforms_enable = False

    cands = stageb_rules.RuleUniNN().propose(ctx, target)
    labels = [str(c.label) for c in cands]

    assert "sparse_factor_1d" in labels
    assert "ratpoly_1d" in labels
    assert labels.index("sparse_factor_1d") < labels.index("ratpoly_1d")
    cand = cands[labels.index("sparse_factor_1d")]
    assert cand.meta["pattern_family"] == "sparse_factor_1d"
    assert cand.meta["sparse_factor_1d_exponents"] == {"z": 4, "1-z^2": -2}
    assert cand.meta["precheck_rel_rms"] < 1.0e-10
    assert abs(cand.meta["scale_init"] + 1.0 / (4.0 * math.pi)) < 1.0e-12


def test_sparse_factor_1d_small_shifted_denominator(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(0.05, 0.95, 700, dtype=torch.float64).unsqueeze(1)
    z = x[:, 0]
    y = 3.5 * (1.0 - z) / (z + 2.0)

    monkeypatch.setattr(
        stageb_rules,
        "_stageB_target_raw_teacher_data",
        lambda *_args, **_kwargs: (x, y),
    )

    cands = stageb_rules._build_sparse_factor_1d_candidates(_ctx(target), target)

    assert cands
    cand = cands[0]
    assert cand.meta["pattern_family"] == "sparse_factor_1d"
    assert cand.meta["sparse_factor_1d_exponents"] == {"1-z": 1, "z+2": -1}
    assert cand.meta["precheck_rel_rms"] < 1.0e-10
    assert abs(cand.meta["scale_init"] - 3.5) < 1.0e-12


def test_sparse_factor_1d_filters_pure_monomials(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(0.2, 0.8, 600, dtype=torch.float64).unsqueeze(1)
    y = 3.0 * x[:, 0].pow(4)

    monkeypatch.setattr(
        stageb_rules,
        "_stageB_target_raw_teacher_data",
        lambda *_args, **_kwargs: (x, y),
    )

    cands = stageb_rules._build_sparse_factor_1d_candidates(_ctx(target), target)

    assert cands == []


def test_sparse_factor_1d_rejects_unitful_coordinate(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    units_spec = UnitsSpec(unit_system=us, x_dims=(L,), y_dim=dimless)

    monkeypatch.setattr(
        stageb_rules,
        "_stageB_target_raw_teacher_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data screen should not run")),
    )

    cands = stageb_rules._build_sparse_factor_1d_candidates(
        _ctx(target, enforce_units=True, units_spec=units_spec, y_dim=dimless),
        target,
    )

    assert cands == []


def test_sparse_factor_1d_rejects_undeclared_unitful_scale(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    units_spec = UnitsSpec(unit_system=us, x_dims=(dimless,), y_dim=L)
    x = torch.linspace(0.2, 0.8, 600, dtype=torch.float64).unsqueeze(1)
    y = x[:, 0].pow(4) / (1.0 - x[:, 0].square()).square()

    monkeypatch.setattr(
        stageb_rules,
        "_stageB_target_raw_teacher_data",
        lambda *_args, **_kwargs: (x, y),
    )

    cands = stageb_rules._build_sparse_factor_1d_candidates(
        _ctx(target, enforce_units=True, units_spec=units_spec, y_dim=L),
        target,
    )

    assert cands == []
