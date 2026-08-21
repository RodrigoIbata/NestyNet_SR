# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import warnings
from types import SimpleNamespace

import torch
import sympy as sp
from sympy.utilities.exceptions import SymPyDeprecationWarning
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    FreeConst,
    MulNode,
    Var,
    build_composite_from_ast,
    collect_all_atoms,
)
from nestynet_sr.sr_core.sympy_bridge import sympy_to_nestynet
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.representation import pretty_print_state
from nestynet_sr.sr_search.stageB.engine import StageBContext, StageBState
from nestynet_sr.sr_search.stageB.fitting import _fit_candidate_root
from nestynet_sr.sr_search.stageB.polish import _candidate_exprs, _parse_expression


def _loader(x, y):
    return DataLoader(TensorDataset(x, y), batch_size=len(x), shuffle=False)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = torch.nn.ModuleList([])

    def forward(self, x):
        return x[:, :1] * 0.0

    def num_parameters(self) -> int:
        return 0

    def build_cache(self, x, y=None):
        return {"x": x, "y": y}

    def jvp(self, *args, **kwargs):
        return torch.zeros((0,), dtype=torch.float64)

    def vjp(self, *args, **kwargs):
        return torch.zeros((0,), dtype=torch.float64)

    def blocks(self):
        return []


def _hp(**overrides):
    base = dict(
        epochs=1,
        strategy="explicit",
        nval_patience=1,
        epochs_min=0,
        chisq_tol=1.0e-12,
        epochs_awful_check=0,
        awful_threshold=1.0e30,
        LM_verbose=False,
        fit_y_link=None,
        fit_y_link_scale=1.0,
        loss_in_MAD_units=False,
        loss_target=1.0e-9,
        loss_acceptable=1.0e-6,
        acceptance_noise_floor=None,
        acceptance_noise_floor_raw=None,
        stageB_overcap_fallback=False,
        stageB_polish=True,
        stageB_polish_max_candidates=16,
        stageB_polish_subtrees=False,
        stageB_polish_max_subtrees=8,
        select_stageB_max_decades_over_floor=3.0,
        select_count_weight=1.0,
        select_struct_gamma=0.05,
        select_param_gamma=0.30,
        select_sep_bonus_decades=0.05,
        select_partial_sep_bonus_decades=0.02,
        select_base_bonus_decades=0.0,
        select_floor_guard_decades=2.0,
        select_below_floor_max_regress_decades=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_stageb_polish_parser_keeps_sr_poly_as_inert_function():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expr = _parse_expression("((scale() * rpoly(x0)) * poly(x1))", 2)
        sp.simplify(expr)

    assert "poly(x1)" in str(expr)
    assert not any(isinstance(w.message, SymPyDeprecationWarning) for w in caught)


def test_stageb_polish_does_not_tie_sqrt_exponent_as_coefficient():
    x0, x1, x2 = sp.symbols("x0 x1 x2")
    expr = sp.sqrt(-9.869604401089354 + (x0 * x1**-1 * x2) ** 2) * x2**-1

    cands = _candidate_exprs(expr, max_candidates=32)
    labels = [label for label, _ in cands]

    assert not any(label.startswith("tie_opposite_coefficients") for label in labels)
    assert any(label == "snap_symbolic_constants" for label in labels)


def test_zero_parameter_candidate_uses_fit_link_loss():
    dtype = torch.float64
    device = torch.device("cpu")
    x = torch.tensor([[1.0], [2.0], [4.0], [8.0]], dtype=dtype)
    y = 2.0 * x
    loader = _loader(x, y)
    hp = _hp(fit_y_link="asinh", fit_y_link_scale=1.0)

    state = _fit_candidate_root(
        root=sympy_to_nestynet(sp.sympify("x0"), 1),
        reuse={},
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
    )

    expected = ((torch.asinh(x) - torch.asinh(y)) ** 2).mean().item()
    raw = ((x - y) ** 2).mean().item()
    assert abs(state.val_loss - expected) < 1.0e-15
    assert abs(state.val_loss - raw) > 1.0


def test_fully_analytic_accepted_polish_commits_simpler_equivalent_state():
    dtype = torch.float64
    device = torch.device("cpu")
    x0 = torch.linspace(1.0, 2.0, 32, dtype=dtype)
    x1 = torch.linspace(2.0, 3.0, 32, dtype=dtype)
    x2 = torch.linspace(3.0, 4.0, 32, dtype=dtype)
    x3 = torch.linspace(4.0, 5.0, 32, dtype=dtype)
    x = torch.stack([x0, x1, x2, x3], dim=1)
    y = ((x0 * x2 + x1 * x3) / (x0 + x1)).reshape(-1, 1)
    loader = _loader(x, y)
    hp = _hp(loss_target=1.0e-9, loss_acceptable=1.0e-6)
    dirty = "((x0*x2 + x1*x3)*(x0+x1))/(x0+x1)**2"

    state = _fit_candidate_root(
        root=sympy_to_nestynet(sp.sympify(dirty, evaluate=False), 4),
        reuse={},
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )

    changed = ctx.maybe_polish_after_accept()

    assert changed is True
    assert ctx.state.val_loss <= state.val_loss + 1.0e-30
    polished = pretty_print_state(ctx.state, sig=16)
    assert "(x0 + x1)**2" not in polished
    assert "x0 * x2 + x1 * x3" in polished
    assert "1/(x0 + x1)" in polished
    assert any(
        rec.get("rule") == "stageB_polish" and rec.get("outcome") == "accept"
        for rec in ctx.decision_log
    )


def test_fully_analytic_polish_snaps_scale_and_drops_small_term():
    dtype = torch.float64
    device = torch.device("cpu")
    x0 = torch.linspace(1.0, 2.0, 40, dtype=dtype)
    x1 = torch.linspace(2.0, 4.0, 40, dtype=dtype)
    x = torch.stack([x0, x1], dim=1)
    y = x0.reshape(-1, 1)
    loader = _loader(x, y)
    hp = _hp(loss_target=1.0e-9, loss_acceptable=1.0e-6)
    dirty = "1.000001*x0 + 1e-6*x1"

    state = _fit_candidate_root(
        root=sympy_to_nestynet(sp.sympify(dirty, evaluate=False), 2),
        reuse={},
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )

    changed = ctx.maybe_polish_after_accept()

    assert changed is True
    assert ctx.state.val_loss < state.val_loss
    assert pretty_print_state(ctx.state, sig=16) == "x0"
    assert any(
        str(rec.get("label", "")).startswith("stageB_polish:snap_all_numeric_nearest")
        for rec in ctx.decision_log
        if rec.get("outcome") == "accept"
    )


def test_fully_analytic_polish_rebuilds_named_coefficient_symbol():
    dtype = torch.float64
    device = torch.device("cpu")
    x_train = torch.linspace(1.0, 2.0, 32, dtype=dtype).reshape(-1, 1)
    x_val = torch.linspace(2.1, 3.0, 32, dtype=dtype).reshape(-1, 1)
    train_loader = _loader(x_train, 4.0 * x_train)
    val_loader = _loader(x_val, 4.0 * x_val)
    hp = _hp(loss_target=1.0e-9, loss_acceptable=1.0e-6)
    root = AddNode(
        MulNode(FreeConst("c", tag="shared_c", init=2.0), Var(0)),
        MulNode(FreeConst("c", tag="shared_c", init=2.0), Var(0)),
    )
    state = StageBState(
        root=root,
        model=build_composite_from_ast(root, dtype=dtype, device=device),
        reuse={},
        val_loss=0.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )

    changed = ctx.maybe_polish_after_accept()

    assert changed is True
    coefficient_atoms = [
        atom
        for atom in collect_all_atoms(ctx.state.root)
        if str(atom.kind).lower() == "free_const"
    ]
    assert len(coefficient_atoms) == 1
    assert coefficient_atoms[0].kwargs["name"] == "c"
    assert coefficient_atoms[0].tag == "shared_c"
    assert "c" in pretty_print_state(ctx.state, sig=16)


def test_subtree_polish_shadow_selects_only_new_maximal_analytic_subtree():
    dtype = torch.float64
    device = torch.device("cpu")
    x = torch.stack(
        [
            torch.linspace(1.0, 2.0, 24, dtype=dtype),
            torch.linspace(2.0, 3.0, 24, dtype=dtype),
        ],
        dim=1,
    )
    y = x[:, :1].clone()
    loader = _loader(x, y)
    previous = AddNode(
        AtomNode(kind="nn", var_idxs=(0, 1), tag="old"),
        AtomNode(kind="nn", var_idxs=(1,), tag="keep"),
    )
    current_left = sympy_to_nestynet(sp.sympify("1.000001*x0 + 1e-6*x1", evaluate=False), 2)
    current = AddNode(current_left, AtomNode(kind="nn", var_idxs=(1,), tag="keep"))
    hp = _hp(stageB_polish_subtrees=True)
    state = _fit_candidate_root(
        root=current,
        reuse={"keep": _DummyModel()},
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )

    changed = ctx.maybe_shadow_polish_subtrees_after_accept(
        previous_root=previous,
        accepted_label="unit_test",
    )

    assert changed is True
    subtree_records = [
        rec for rec in ctx.decision_log if rec.get("rule") == "stageB_polish_subtree"
    ]
    assert len(subtree_records) == 1
    assert subtree_records[0]["outcome"] == "accept"
    assert subtree_records[0]["target"] == "root.left"
    assert subtree_records[0]["cand_loss"] < subtree_records[0]["base_loss"]
    assert "x0" in subtree_records[0]["ast_snapshot"]
    assert "nn0" in subtree_records[0]["ast_snapshot"]
    polished = pretty_print_state(ctx.state, sig=16)
    assert "1.000001" not in polished
    assert "1e-6" not in polished
    assert "keep(x1)" in polished
    assert ctx.state.simplification_path[-1]["action"].startswith("rewrite stageB_polish_subtree:")
    assert "target=root.left" in ctx.state.simplification_path[-1]["detail"]


def test_subtree_polish_shadow_skips_unit_inconsistent_subtree():
    dtype = torch.float64
    device = torch.device("cpu")
    x = torch.stack(
        [
            torch.linspace(1.0, 2.0, 12, dtype=dtype),
            torch.linspace(2.0, 3.0, 12, dtype=dtype),
        ],
        dim=1,
    )
    y = torch.zeros((x.shape[0], 1), dtype=dtype)
    loader = _loader(x, y)
    previous = AddNode(
        AtomNode(kind="nn", var_idxs=(0, 1), tag="old"),
        AtomNode(kind="nn", var_idxs=(1,), tag="keep"),
    )
    current = AddNode(
        AddNode(Var(0), Var(1)),
        AtomNode(kind="nn", var_idxs=(1,), tag="keep"),
    )
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}), us.dim({"T": 1})),
        y_dim=us.dim({"L": 1}),
    )
    hp = _hp(stageB_polish_subtrees=True)
    ctx = StageBContext(
        state=StageBState(root=current, model=_DummyModel(), reuse={}, val_loss=1.0e-8),
        train_loader=loader,
        val_loader=loader,
        lm_hp=hp,
        device=device,
        dtype=dtype,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
        enforce_units=True,
        units_spec=spec,
    )

    changed = ctx.maybe_shadow_polish_subtrees_after_accept(
        previous_root=previous,
        accepted_label="unit_test",
    )

    assert changed is False
    subtree_records = [
        rec for rec in ctx.decision_log if rec.get("rule") == "stageB_polish_subtree"
    ]
    assert len(subtree_records) == 1
    assert subtree_records[0]["outcome"] == "shadow_skip"
    assert "units-domain-inconsistent" in subtree_records[0]["reason"]
