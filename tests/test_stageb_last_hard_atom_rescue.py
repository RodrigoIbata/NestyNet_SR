# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, PowNode, Var, get_input_exprs
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search import candidate_builders
from nestynet_sr.sr_search.stageB import rules as stageb_rules


def _ctx(root, target_dim, x_dims, *, enforce_units=True):
    us = UnitSystem(("L", "T", "M"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=tuple(x_dims),
        y_dim=target_dim,
    )
    return SimpleNamespace(
        state=SimpleNamespace(root=root, reuse={"leaf0": object()}, model=object()),
        enforce_units=bool(enforce_units),
        units_spec=spec,
        infer_target_dim=lambda _target: target_dim,
        train_loader_probe=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        is_pattern_disabled=lambda _name: False,
        log=lambda *_a, **_k: None,
    )


def test_last_hard_atom_rescue_requires_single_unit_checked_known_dim():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    L = us.dim({"L": 1})
    target = AtomNode("nn", (0, 1), tag="leaf0")
    other = AtomNode("nn", (0,), tag="leaf1")
    rule = stageb_rules.RuleLastHardAtomRescue()

    assert rule.iter_targets(_ctx(target, dimless, [L, L])) == [target]
    assert rule.iter_targets(_ctx(target, dimless, [L, L], enforce_units=False)) == []

    multi_root = AddNode(target, other)
    assert rule.iter_targets(_ctx(multi_root, dimless, [L, L])) == []

    problem = AtomNode(
        "nn",
        (0,),
        kwargs={"_problem_label": "nonsense_units"},
        tag="leaf_problem",
    )
    assert rule.iter_targets(_ctx(AddNode(target, problem), dimless, [L, L])) == []

    ctx_unknown = _ctx(target, dimless, [L, L])
    ctx_unknown.infer_target_dim = lambda _target: None
    assert rule.iter_targets(ctx_unknown) == []


def test_last_hard_atom_rescue_calls_capped_degree4_ratpoly(monkeypatch):
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    L = us.dim({"L": 1})
    target = AtomNode("nn", (0, 1), tag="leaf0")
    captured = {}

    monkeypatch.setattr(stageb_rules, "_build_last_hard_ratio_candidates", lambda **_kwargs: [])

    def _capture_ratpoly(**kwargs):
        captured.update(kwargs)
        rat = AtomNode(
            "ratpoly",
            (0, 1),
            kwargs={"deg_num": 1, "deg_den": 1},
            tag="leaf0",
        )
        return [(rat, None, {"signature": (101,), "pattern_family": "ratpoly"})]

    monkeypatch.setattr(stageb_rules, "_build_ratpoly_candidates", _capture_ratpoly)

    ctx = _ctx(target, dimless, [L, L])
    cands = stageb_rules.RuleLastHardAtomRescue().propose(ctx, target)

    assert cands
    assert cands[0].label == "last_ratpoly"
    assert captured["max_deg_num"] == 4
    assert captured["max_deg_den"] == 4
    assert captured["max_terms_total"] == 90
    assert captured["target_dim"] == tuple(dimless)
    assert captured["x_dims"] == [tuple(L), tuple(L)]
    assert captured["enforce_units"] is True


def test_last_hard_ratio_builder_finds_inv_one_minus_z2_squared(monkeypatch):
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    L = us.dim({"L": 1})
    target = AtomNode("nn", (0, 1), tag="leaf0")

    x0 = torch.linspace(2.0, 5.0, 600, dtype=torch.float64)
    z = torch.linspace(0.05, 0.60, 600, dtype=torch.float64)
    x1 = x0 * z
    X = torch.stack((x0, x1), dim=1)
    F = 1.0 / ((1.0 - z * z) ** 2)

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    results = candidate_builders._build_last_hard_ratio_candidates(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        target_dim=tuple(dimless),
        x_dims=[tuple(L), tuple(L)],
        min_points=50,
        rel_rms_threshold=1e-6,
    )

    matches = [item for item in results if item[0] == "last_ratio_z1_over_z0_p0_q2"]
    assert matches
    _label, cand_root, _init, meta = matches[0]
    assert cand_root.kind == "ratpoly"
    assert meta["ratio_p"] == 0
    assert meta["ratio_q"] == 2
    ratio_expr = get_input_exprs(cand_root)[0]
    assert isinstance(ratio_expr, MulNode)
    assert getattr(ratio_expr.left, "var_idxs", ()) == (1,)
    assert isinstance(ratio_expr.right, PowNode)
    assert getattr(ratio_expr.right.base, "var_idxs", ()) == (0,)
    assert float(ratio_expr.right.exponent) == -1.0


def test_last_hard_ratio_builder_requires_dimensionless_output(monkeypatch):
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    target = AtomNode("nn", (0, 1), tag="leaf0")

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (
            torch.ones(100, 2, dtype=torch.float64),
            torch.ones(100, dtype=torch.float64),
        ),
    )

    results = candidate_builders._build_last_hard_ratio_candidates(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        target_dim=tuple(L),
        x_dims=[tuple(L), tuple(L)],
        min_points=10,
    )
    assert results == []


class _InvSinFourthTeacher(torch.nn.Module):
    def forward(self, x):
        return torch.sin(0.5 * x[:, :1]).pow(-4)


class _SinSquareTeacher(torch.nn.Module):
    def forward(self, x):
        return torch.sin(2.0 * torch.pi * x[:, :1]).pow(2)


class _SinSquareReciprocalTeacher(torch.nn.Module):
    def forward(self, x):
        return torch.sin(2.0 * torch.pi / x[:, :1]).pow(2)


def test_last_hard_trig_square_finds_sin_squared():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    target = AtomNode("nn", (0,), tag="leaf0")

    x = torch.linspace(0.05, 1.95, 800, dtype=torch.float64).unsqueeze(1)
    y = torch.zeros(x.shape[0], 1, dtype=torch.float64)
    ctx = _ctx(target, dimless, [dimless])
    ctx.state.reuse = {"leaf0": _SinSquareTeacher()}
    ctx.train_loader_probe = [(x, y)]
    ctx.lm_hp = SimpleNamespace(macro_domain_ok_frac=0.98)

    rule = stageb_rules.RuleLastHardTrigSquare1D()
    assert rule.iter_targets(ctx) == [target]

    cands = rule.propose(ctx, target)

    assert cands
    first = cands[0]
    assert first.label == "last_trig_square_sin"
    assert first.meta["pattern_family"] == "last_hard_trig_square"
    assert abs(first.meta["omega"] - 2.0 * torch.pi) < 1.0e-12
    assert abs(first.meta["harmonic_omega"] - 4.0 * torch.pi) < 1.0e-12
    assert first.meta["screen_rel_rms"] < 1.0e-10
    assert all(
        not (isinstance(atom, AtomNode) and str(atom.kind).lower() == "nn")
        for atom in stageb_rules._collect_all_atoms(first.root)
    )


def test_last_hard_trig_square_works_on_reciprocal_coordinate_alias():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    target = AtomNode("nn", (0,), tag="leaf0", inputs=(Var(0),))

    x = torch.linspace(1.0, 4.0, 800, dtype=torch.float64).unsqueeze(1)
    y = torch.zeros(x.shape[0], 1, dtype=torch.float64)
    ctx = _ctx(target, dimless, [dimless])
    ctx.state.reuse = {"leaf0": _SinSquareReciprocalTeacher()}
    ctx.train_loader_probe = [(x, y)]
    ctx.lm_hp = SimpleNamespace(macro_domain_ok_frac=0.98)

    cands = stageb_rules.RuleLastHardTrigSquare1D().propose(ctx, target)

    alias = [c for c in cands if str(c.label).endswith("[z_inv]")]
    assert alias
    assert alias[0].meta["coordinate_variant"] == "z_inv"
    assert alias[0].meta["pattern_family"] == "last_hard_trig_square"


def test_last_hard_trig_power_finds_half_angle_inverse_fourth_power():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    target = AtomNode("nn", (0,), tag="leaf0")

    x = torch.linspace(1.0, 5.0, 600, dtype=torch.float64).unsqueeze(1)
    y = torch.zeros(x.shape[0], 1, dtype=torch.float64)
    ctx = _ctx(target, dimless, [dimless])
    ctx.state.reuse = {"leaf0": _InvSinFourthTeacher()}
    ctx.train_loader_probe = [(x, y)]
    ctx.lm_hp = SimpleNamespace(macro_domain_ok_frac=0.98)

    rule = stageb_rules.RuleLastHardTrigPower1D()
    assert rule.iter_targets(ctx) == [target]

    cands = rule.propose(ctx, target)

    assert cands
    first = cands[0]
    assert first.label == "last_trig_power_sin_p-4"
    assert first.meta["pattern_family"] == "last_hard_trig_power"
    assert abs(first.meta["omega"] - 0.5) < 1.0e-12
    assert first.meta["screen_rel_rms"] < 1.0e-10
    assert all(
        not (isinstance(atom, AtomNode) and str(atom.kind).lower() == "nn")
        for atom in stageb_rules._collect_all_atoms(first.root)
    )


def test_last_hard_trig_power_rejects_dimensionful_argument():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    L = us.dim({"L": 1})
    target = AtomNode("nn", (0,), tag="leaf0")
    ctx = _ctx(target, dimless, [L])

    assert stageb_rules.RuleLastHardTrigPower1D().iter_targets(ctx) == []
