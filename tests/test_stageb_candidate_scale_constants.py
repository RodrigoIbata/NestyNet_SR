# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    MulNode,
    Var,
    ast_to_human_readable,
    get_input_exprs,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search import candidate_builders as cb
from nestynet_sr.sr_search.features import TrigAxisSpec
from nestynet_sr.sr_search.stageB import rules as stageb_rules
from nestynet_sr.sr_search.stageB.atom_mapping import _collect_all_atoms


def _trig_spec(axis: int = 0, omega: float = 1.0) -> TrigAxisSpec:
    return TrigAxisSpec(
        axis=axis,
        omega=omega,
        strength=1.0,
        n_points=512,
        tmin=-1.0,
        tmax=1.0,
        phase=0.0,
        rel_std=0.0,
    )


def test_make_affine_trig_rewrite_uses_scale_offset():
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    root = target

    out = cb._make_affine_trig_rewrite(root, target, _trig_spec(axis=0, omega=1.0))
    assert isinstance(out, AddNode)
    assert isinstance(out.left, AtomNode)
    assert str(out.left.kind).lower() == "scale"
    assert out.left.tag == "leaf0_c"


def test_trig_diff_affine_envelope_uses_scale_for_constant_terms(monkeypatch):
    import nestynet_sr.sr_search.stageB.splits as splits
    import nestynet_sr.sr_search.stageB.subtree_utils as subtree_utils

    omega = 2.0
    spec = _trig_spec(axis=0, omega=omega)

    def _fake_gather_nn_atom_value_grad_hess(**kwargs):
        n = 600
        z = torch.linspace(-2.0, 2.0, n)
        x = z.view(-1, 1)
        arg = omega * z
        offset = 1.25
        amp = 0.8
        u = offset + amp * torch.cos(arg)

        du = torch.zeros(n, 1, dtype=u.dtype)
        du[:, 0] = -amp * omega * torch.sin(arg)

        h = torch.zeros(n, 1, 1, dtype=u.dtype)
        h[:, 0, 0] = -amp * (omega**2) * torch.cos(arg)
        return x, None, u, du, h

    monkeypatch.setattr(
        splits, "_gather_nn_atom_value_grad_hess", _fake_gather_nn_atom_value_grad_hess
    )
    monkeypatch.setattr(subtree_utils, "_infer_nn_hyperparams_from_root", lambda _root: (4, False))

    z_expr = AddNode(Var(0), MulNode(ConstNode(-1.0), Var(1)))
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nn01", inputs=(z_expr,))
    root = target
    cand_root, _init_fn = cb._build_trig_diff_affine_envelope_candidate(
        root=root,
        target=target,
        trig_spec=spec,
        model=nn.Identity(),
        train_loader=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        partner_axis=1,
        min_points=200,
    )

    assert isinstance(cand_root, AddNode)
    assert isinstance(cand_root.left, AtomNode)
    assert isinstance(cand_root.right, MulNode)
    assert isinstance(cand_root.right.left, AtomNode)
    assert str(cand_root.left.kind).lower() == "scale"
    assert str(cand_root.right.left.kind).lower() == "scale"


def test_trig_diff_affine_envelope_finds_second_compound_input(monkeypatch):
    import nestynet_sr.sr_search.stageB.splits as splits
    import nestynet_sr.sr_search.stageB.subtree_utils as subtree_utils

    omega = 1.7
    spec = _trig_spec(axis=2, omega=omega)

    def _fake_gather_nn_atom_value_grad_hess(**kwargs):
        n = 700
        p = torch.linspace(1.0, 2.0, n)
        z = torch.linspace(-2.0, 2.0, n)
        x = torch.stack((p, z), dim=1)
        offset = p
        amp = 0.5 * p
        arg = omega * z
        u = offset + amp * torch.cos(arg)

        du = torch.zeros(n, 2, dtype=u.dtype)
        du[:, 1] = -amp * omega * torch.sin(arg)

        h = torch.zeros(n, 2, 2, dtype=u.dtype)
        h[:, 1, 1] = -amp * (omega**2) * torch.cos(arg)
        return x, None, u, du, h

    monkeypatch.setattr(
        splits, "_gather_nn_atom_value_grad_hess", _fake_gather_nn_atom_value_grad_hess
    )
    monkeypatch.setattr(subtree_utils, "_infer_nn_hyperparams_from_root", lambda _root: (4, False))

    p_expr = MulNode(Var(0), Var(1))
    diff_expr = AddNode(Var(2), MulNode(ConstNode(-1.0), Var(3)))
    target = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={},
        tag="nn_multi",
        inputs=(p_expr, diff_expr),
    )
    cand_root, _init_fn = cb._build_trig_diff_affine_envelope_candidate(
        root=target,
        target=target,
        trig_spec=spec,
        model=nn.Identity(),
        train_loader=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        partner_axis=3,
        min_points=200,
    )

    assert isinstance(cand_root, AddNode)
    assert isinstance(cand_root.left, AtomNode)
    assert ast_to_human_readable(get_input_exprs(cand_root.left)[0]) == ast_to_human_readable(p_expr)
    arg_atom = cand_root.right.right.arg
    assert isinstance(arg_atom, AtomNode)
    assert ast_to_human_readable(get_input_exprs(arg_atom)[0]) == ast_to_human_readable(diff_expr)


def test_trig_affine_envelope_uses_per_leaf_homogeneous_units(monkeypatch):
    import nestynet_sr.sr_search.stageB.splits as splits

    us = UnitSystem(("U",))
    unit = us.dim({"U": 1})
    dimless = us.dimless()
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(unit, unit, dimless),
        y_dim=unit,
    )

    # This is the pb037 shape.  The full NN atom has a dimensionless trig axis,
    # so a whole-atom homogeneity flag is false; the offset/amplitude leaves
    # only see x0,x1 and must still use homogeneous polynomial bases.
    assert not stageb_rules._poly_leaf_homogeneous_for_raw_var_idxs([0, 1, 2], units_spec)
    assert stageb_rules._poly_leaf_homogeneous_for_raw_var_idxs([0, 1], units_spec)

    def _fake_gather_nn_atom_value_grad_hess(**kwargs):
        x0_vals = torch.linspace(1.0, 5.0, 10, dtype=torch.float64)
        x1_vals = torch.linspace(1.2, 4.8, 10, dtype=torch.float64)
        x2_vals = torch.linspace(-2.0, 2.0, 8, dtype=torch.float64)
        x0, x1, x2 = torch.meshgrid(x0_vals, x1_vals, x2_vals, indexing="ij")
        x0 = x0.reshape(-1)
        x1 = x1.reshape(-1)
        x2 = x2.reshape(-1)
        X = torch.stack([x0, x1, x2], dim=1)

        amp = 2.0 * torch.sqrt(x0 * x1)
        u = x0 + x1 + amp * torch.cos(x2)

        du = torch.zeros(X.shape[0], 3, dtype=torch.float64)
        du[:, 2] = -amp * torch.sin(x2)

        hess = torch.zeros(X.shape[0], 3, 3, dtype=torch.float64)
        hess[:, 2, 2] = -amp * torch.cos(x2)
        return X, None, u, du, hess

    monkeypatch.setattr(
        splits, "_gather_nn_atom_value_grad_hess", _fake_gather_nn_atom_value_grad_hess
    )

    target = AtomNode(kind="nn", var_idxs=(0, 1, 2), kwargs={}, tag="leaf0")
    cand_root, _init_fn = cb._build_trig_affine_envelope_candidate(
        root=target,
        target=target,
        trig_spec=_trig_spec(axis=2, omega=1.0),
        model=nn.Identity(),
        train_loader=None,
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=200,
        homogeneous=stageb_rules._poly_leaf_homogeneous_for_raw_var_idxs([0, 1], units_spec),
    )

    assert cand_root is not None
    units = check_units_ast(cand_root, units_spec)
    assert units.ok, units.reason

    tagged = {
        atom.tag: atom
        for atom in _collect_all_atoms(cand_root)
        if isinstance(atom, AtomNode) and atom.tag is not None
    }
    assert tagged["leaf0_trig_off"].kwargs["min_total"] == tagged["leaf0_trig_off"].kwargs["degree"]
    assert tagged["leaf0_trig_amp2"].kwargs["min_total"] == tagged["leaf0_trig_amp2"].kwargs["degree"]
    assert tagged["leaf0_trig_arg"].kwargs["min_total"] == 0


def test_symexp_denom_candidate_uses_scale_constants(monkeypatch):
    import nestynet_sr.sr_search.fitting_utils as fitting_utils

    slope = 0.7

    def _fake_teacher_data(*args, **kwargs):
        x = torch.linspace(0.25, 2.5, 700)
        scale = 1.8
        f = scale / (torch.exp(slope * x) + torch.exp(-slope * x))
        return x, f

    def _fake_fit_rational_coeffs_1d(x, r, deg_num, deg_den, min_points):
        return (
            torch.tensor([0.0, slope], dtype=x.dtype, device=x.device),
            torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device),
        )

    monkeypatch.setattr(cb, "_gather_teacher_data_1d", _fake_teacher_data)
    monkeypatch.setattr(fitting_utils, "_fit_rational_coeffs_1d", _fake_fit_rational_coeffs_1d)

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="nn0")
    root = target
    cands = cb._build_symexp_denom_1d_candidate(
        root=root,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        min_points=200,
        rel_rms_threshold=0.2,
    )

    assert cands, "Expected at least one symexp_denom candidate"
    cand_root, _init_fn, _label = cands[0]

    matched = 0
    for atom in _collect_all_atoms(cand_root):
        if not isinstance(atom, AtomNode) or atom.tag is None:
            continue
        if not atom.tag.startswith("symexp1d_"):
            continue
        if ("_cp_" in atom.tag) or ("_cm_" in atom.tag) or ("_m1_" in atom.tag) or ("_scale_" in atom.tag):
            matched += 1
            assert str(atom.kind).lower() == "scale"

    assert matched >= 4


def test_symexp_denom_candidate_emits_fixed_const_scale_variant(monkeypatch):
    import nestynet_sr.sr_search.fitting_utils as fitting_utils

    slope = 0.7

    def _fake_teacher_data(*args, **kwargs):
        x = torch.linspace(0.25, 2.5, 700)
        scale = 1.8
        f = scale / (torch.exp(slope * x) + torch.exp(-slope * x))
        return x, f

    def _fake_fit_rational_coeffs_1d(x, r, deg_num, deg_den, min_points):
        return (
            torch.tensor([0.0, slope], dtype=x.dtype, device=x.device),
            torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device),
        )

    monkeypatch.setattr(cb, "_gather_teacher_data_1d", _fake_teacher_data)
    monkeypatch.setattr(fitting_utils, "_fit_rational_coeffs_1d", _fake_fit_rational_coeffs_1d)

    us = UnitSystem(base=("L", "T"))
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        fixed_const_dims={"k_fixed": us.dimless()},
        fixed_const_values={"k_fixed": 1.8},
    )

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="nn0")
    root = target
    cands = cb._build_symexp_denom_1d_candidate(
        root=root,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        min_points=200,
        rel_rms_threshold=0.2,
        units_spec=units_spec,
    )

    assert cands, "Expected at least one symexp_denom candidate"

    found_fixed_scale = False
    for cand_root, _init_fn, _label in cands:
        for atom in _collect_all_atoms(cand_root):
            if not isinstance(atom, AtomNode) or atom.tag is None:
                continue
            if not atom.tag.startswith("symexp1d_scale_"):
                continue
            if str(atom.kind).lower() == "fixed_const":
                found_fixed_scale = True
                break
        if found_fixed_scale:
            break

    assert found_fixed_scale, "Expected a fixed-const variant for symexp scale"
