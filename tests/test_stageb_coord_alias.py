# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import pytest
import torch

from nestynet_sr.sr_core import ast_to_human_readable
from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, PowNode, Var, build_composite_from_ast
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search.representation import pretty_print_state
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBState
from nestynet_sr.sr_search.stageB.rules import RuleUniNN


def _make_compound_ctx(root):
    logs = []
    x = torch.tensor(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        dtype=torch.float64,
    )
    y = torch.zeros((x.shape[0], 1), dtype=torch.float64)
    return SimpleNamespace(
        state=SimpleNamespace(
            root=root,
            reuse={"leaf0": torch.nn.Identity()},
            reuses=None,
        ),
        scaling_by_axis={},
        trig_by_axis={},
        stageA_x_transforms={},
        xcoords_applied=False,
        train_loader_probe=[(x, y)],
        device=torch.device("cpu"),
        dtype=torch.float64,
        units_spec=None,
        enforce_units=False,
        lm_hp=SimpleNamespace(),
        infer_target_dim=lambda _target: None,
        is_pattern_disabled=lambda _pattern: False,
        verbose=False,
        log=logs.append,
        _logs=logs,
    )


class _SqrtTeacher(torch.nn.Module):
    def __init__(self, mode="sqrt"):
        super().__init__()
        self.mode = str(mode)

    def forward(self, x):
        z = x[:, :1].to(dtype=torch.float64)
        if self.mode == "inv_sqrt":
            return torch.rsqrt(torch.clamp(z, min=1.0e-12))
        if self.mode == "sqrt1p":
            return torch.sqrt(torch.clamp(1.0 + z, min=1.0e-12))
        return torch.sqrt(torch.clamp(z, min=1.0e-12))


class _PowerTeacher(torch.nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = float(power)

    def forward(self, x):
        z = x[:, :1].to(dtype=torch.float64)
        return torch.pow(torch.clamp(z, min=1.0e-12), self.power)


def _make_unit_ctx(root, *, teacher, x_dims, y_dim, x):
    logs = []
    y = torch.zeros((x.shape[0], 1), dtype=torch.float64)
    us = UnitSystem(("L", "T", "M"))
    spec = UnitsSpec(unit_system=us, x_dims=tuple(x_dims), y_dim=y_dim)
    return SimpleNamespace(
        state=SimpleNamespace(
            root=root,
            reuse={"leaf0": teacher},
            reuses=None,
        ),
        scaling_by_axis={},
        trig_by_axis={},
        stageA_x_transforms={},
        xcoords_applied=False,
        train_loader_probe=[(x.to(dtype=torch.float64), y)],
        device=torch.device("cpu"),
        dtype=torch.float64,
        units_spec=spec,
        enforce_units=True,
        lm_hp=SimpleNamespace(stageB_leaf_transforms_enable=False),
        infer_target_dim=lambda _target: y_dim,
        is_pattern_disabled=lambda _pattern: False,
        verbose=False,
        log=logs.append,
        _logs=logs,
    )


def _candidate_by_label(cands, label):
    for cand in cands:
        if str(cand.label) == str(label):
            return cand
    return None


def _sqrt_poly_base(cand):
    root = cand.root
    assert isinstance(root, PowNode)
    assert isinstance(root.base, AtomNode)
    assert str(root.base.kind).lower() == "poly"
    return root.base, float(root.exponent)


def test_univariate_rule_keeps_only_nonrepeat_reciprocal_coordinate_candidates():
    z_expr = MulNode(Var(0), Var(1))
    root = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0", inputs=(z_expr,))
    ctx = _make_compound_ctx(root)

    cands = RuleUniNN().propose(ctx, root)
    labels = [str(c.label) for c in cands]

    assert "monomial_deg1[z_inv]" in labels
    assert "monomial_deg2[z_inv]" in labels
    assert "monomial_deg3[z_inv]" in labels
    assert "poly[z_inv]" in labels
    assert "polylog[z_inv]" not in labels
    assert all(not label.startswith("inv_monomial_deg") or not label.endswith("[z_inv]") for label in labels)

    cand = cands[labels.index("poly[z_inv]")]
    text = ast_to_human_readable(cand.root)
    compact = text.replace(" ", "")
    assert "x0*x1" in compact
    assert (
        "**(-1.0)" in compact
        or "**-1.0" in compact
        or "**(-1)" in compact
        or "**-1" in compact
    )
    assert cand.meta["coordinate_variant"] == "z_inv"
    assert cand.meta["coordinate_variant_display"] == "1/z"

    wrapped = cand.meta["_reuse_override"]["leaf0"]
    out = wrapped(torch.tensor([[0.25]], dtype=torch.float64))
    assert torch.allclose(out, torch.tensor([[4.0]], dtype=torch.float64))


def test_univariate_compound_leaf_reuses_compound_macro_library(monkeypatch):
    from nestynet_sr.sr_search import compound_functions

    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    root = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0", inputs=(z_expr,))
    ctx = _make_compound_ctx(root)
    calls = []

    def _fake_macro_proposer(_ctx, target):
        calls.append(tuple(getattr(target, "var_idxs", ())))
        return [
            Candidate(
                "cf_inv1m",
                root=Var(0),
                meta={"log": "[Stage B]  Trying macro inv1m on NN leaf"},
            )
        ]

    monkeypatch.setattr(compound_functions, "propose_compound_function_macros", _fake_macro_proposer)

    cands = RuleUniNN().propose(ctx, root)
    macro_cands = [c for c in cands if str(c.label) == "cf_inv1m"]

    assert calls
    assert macro_cands
    assert macro_cands[0].meta["compound_1d_macro"] is True
    assert "1D compound macro" in macro_cands[0].meta["log"]


def test_univariate_trig_hint_also_proposes_affine_trig_for_offset_forms():
    z_expr = MulNode(Var(0), Var(1))
    root = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0", inputs=(z_expr,))
    ctx = _make_compound_ctx(root)
    ctx.trig_by_axis = {0: SimpleNamespace(axis=0, omega=1.0, strength=100.0)}

    cands = RuleUniNN().propose(ctx, root)
    labels = [str(c.label) for c in cands]

    assert "trig" in labels
    assert "affine_trig" in labels
    affine = cands[labels.index("affine_trig")]
    assert "c + A*sin" in affine.meta["log"]


def test_unitful_sqrt_poly_allows_sqrt_of_unitful_compound():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    z = AddNode(PowNode(Var(0), 2.0), PowNode(Var(1), 2.0))
    root = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0", inputs=(z,))
    t = torch.linspace(0.3, 2.0, 600, dtype=torch.float64)
    x = torch.stack((t, t + 0.4), dim=1)
    ctx = _make_unit_ctx(root, teacher=_SqrtTeacher(), x_dims=(L, L), y_dim=L, x=x)

    cands = RuleUniNN().propose(ctx, root)
    cand = _candidate_by_label(cands, "sqrt_poly")

    assert cand is not None
    poly, exponent = _sqrt_poly_base(cand)
    assert exponent == 0.5
    assert poly.kwargs["degree"] == 1
    assert poly.kwargs["min_total"] == 1
    assert check_units_ast(cand.root, ctx.units_spec).ok


def test_unitful_sqrt_poly_allows_inverse_sqrt_of_unitful_compound():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    inv_L = us.dim({"L": -1})
    z = AddNode(PowNode(Var(0), 2.0), PowNode(Var(1), 2.0))
    root = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0", inputs=(z,))
    t = torch.linspace(0.5, 2.5, 600, dtype=torch.float64)
    x = torch.stack((t, t + 0.3), dim=1)
    ctx = _make_unit_ctx(root, teacher=_SqrtTeacher("inv_sqrt"), x_dims=(L, L), y_dim=inv_L, x=x)

    cands = RuleUniNN().propose(ctx, root)
    cand = _candidate_by_label(cands, "sqrt_poly")

    assert cand is not None
    poly, exponent = _sqrt_poly_base(cand)
    assert exponent == -0.5
    assert poly.kwargs["degree"] == 1
    assert poly.kwargs["min_total"] == 1
    assert check_units_ast(cand.root, ctx.units_spec).ok


def test_sqrt_poly_reciprocal_coordinate_alias_handles_sqrt_one_over_z():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    inv_L2 = us.dim({"L": -2})
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(0.2, 2.0, 600, dtype=torch.float64).unsqueeze(1)
    ctx = _make_unit_ctx(root, teacher=_SqrtTeacher("inv_sqrt"), x_dims=(inv_L2,), y_dim=L, x=x)

    cands = RuleUniNN().propose(ctx, root)
    cand = _candidate_by_label(cands, "sqrt_poly[z_inv]")

    assert cand is not None
    assert cand.meta["coordinate_variant"] == "z_inv"
    assert cand.meta["coordinate_variant_display"] == "1/z"
    poly, exponent = _sqrt_poly_base(cand)
    assert exponent == 0.5
    assert poly.kwargs["degree"] == 1
    assert poly.kwargs["min_total"] == 1
    assert check_units_ast(cand.root, ctx.units_spec).ok


def test_dimensionless_sqrt_one_refine_z_remains_allowed():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(0.1, 2.0, 600, dtype=torch.float64).unsqueeze(1)
    ctx = _make_unit_ctx(root, teacher=_SqrtTeacher("sqrt1p"), x_dims=(dimless,), y_dim=dimless, x=x)

    cands = RuleUniNN().propose(ctx, root)
    cand = _candidate_by_label(cands, "sqrt_poly")

    assert cand is not None
    poly, exponent = _sqrt_poly_base(cand)
    assert exponent == 0.5
    assert poly.kwargs["min_total"] == 0
    assert check_units_ast(cand.root, ctx.units_spec).ok


def test_unitful_sqrt_one_refine_z_is_not_a_strict_units_sqrt_poly():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    z = AddNode(PowNode(Var(0), 2.0), PowNode(Var(1), 2.0))
    root = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0", inputs=(z,))
    t = torch.linspace(0.05, 1.1, 600, dtype=torch.float64)
    x = torch.stack((t, t + 0.05), dim=1)
    ctx = _make_unit_ctx(root, teacher=_SqrtTeacher("sqrt1p"), x_dims=(L, L), y_dim=L, x=x)

    cands = RuleUniNN().propose(ctx, root)
    labels = [str(c.label) for c in cands]

    assert "sqrt_poly" not in labels


def test_monomial_only_rule_uses_reciprocal_coordinate_for_inverse_degrees():
    z_expr = MulNode(Var(0), Var(1))
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0", inputs=(z_expr,))
    root = MulNode(Var(0), target)
    ctx = _make_compound_ctx(root)

    cands = RuleUniNN(monomial_only=True).propose(ctx, target)
    labels = [str(c.label) for c in cands]

    assert "inv_monomial_deg1" not in labels
    assert "inv_monomial_deg2" not in labels
    assert "inv_monomial_deg3" not in labels
    assert "monomial_deg1[z_inv]" in labels
    assert "monomial_deg2[z_inv]" in labels
    assert "monomial_deg3[z_inv]" in labels
    assert labels.index("monomial_deg1[z_inv]") == labels.index("monomial_deg1") + 1
    assert labels.index("monomial_deg2[z_inv]") == labels.index("monomial_deg2") + 1
    assert labels.index("monomial_deg3[z_inv]") == labels.index("monomial_deg3") + 1


def test_univariate_rule_adds_reciprocal_coordinate_for_plain_leaf():
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    ctx = _make_compound_ctx(root)

    cands = RuleUniNN(monomial_only=True).propose(ctx, root)
    labels = [str(c.label) for c in cands]

    assert "monomial_deg1[z_inv]" in labels
    assert "monomial_deg2[z_inv]" in labels
    assert "monomial_deg3[z_inv]" in labels
    assert "inv_monomial_deg1" not in labels

    cand = cands[labels.index("monomial_deg1[z_inv]")]
    compact = ast_to_human_readable(cand.root).replace(" ", "")
    assert "x0" in compact
    assert (
        "**(-1.0)" in compact
        or "**-1.0" in compact
        or "**(-1)" in compact
        or "**-1" in compact
    )
    wrapped = cand.meta["_reuse_override"]["leaf0"]
    out = wrapped(torch.tensor([[0.25]], dtype=torch.float64))
    assert torch.allclose(out, torch.tensor([[4.0]], dtype=torch.float64))


def test_clean_integer_monomial_screen_adds_degree4_candidate_with_scale():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    L4 = us.dim({"L": 4})
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(1.0, 5.0, 600, dtype=torch.float64).unsqueeze(1)
    ctx = _make_unit_ctx(root, teacher=_PowerTeacher(4), x_dims=(L,), y_dim=L4, x=x)

    cands = RuleUniNN().propose(ctx, root)
    labels = [str(c.label) for c in cands]
    cand = _candidate_by_label(cands, "monomial_deg4")

    assert cand is not None
    assert labels.index("monomial_deg4") == 0
    assert cand.meta["monomial_fixed_power"] == 4.0
    assert cand.meta["monomial_screen_rel_rms"] < 1.0e-3
    assert check_units_ast(cand.root, ctx.units_spec).ok
    compact = ast_to_human_readable(cand.root).replace(" ", "")
    assert "scale" in compact
    assert "x0" in compact
    assert "**4.0" in compact or "**4" in compact


def test_clean_integer_monomial_screen_adds_reciprocal_degree4_candidate():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    inv_L4 = us.dim({"L": -4})
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x = torch.linspace(1.0, 5.0, 600, dtype=torch.float64).unsqueeze(1)
    ctx = _make_unit_ctx(root, teacher=_PowerTeacher(-4), x_dims=(L,), y_dim=inv_L4, x=x)

    cands = RuleUniNN().propose(ctx, root)
    cand = _candidate_by_label(cands, "monomial_deg4[z_inv]")

    assert cand is not None
    assert cand.meta["coordinate_variant"] == "z_inv"
    assert cand.meta["monomial_fixed_power"] == 4.0
    assert check_units_ast(cand.root, ctx.units_spec).ok


def _render_analytic_root(root):
    model = build_composite_from_ast(root, dtype=torch.float64, device=torch.device("cpu"))
    state = StageBState(root=root, model=model, reuse={}, val_loss=0.0)
    return pretty_print_state(state, sig=16)


def test_pretty_print_parenthesizes_reciprocal_coordinate_polynomial_power():
    sp = pytest.importorskip("sympy")
    z_inv = PowNode(Var(4), -1.0)
    root = AtomNode(
        kind="rpoly",
        var_idxs=(4,),
        kwargs={"degree": 2, "min_total": 2},
        tag="leaf0",
        inputs=(z_inv,),
    )

    expr = _render_analytic_root(root)

    assert "^-1^2" not in expr
    x4 = sp.Symbol("x4")
    parsed = sp.sympify(expr.replace("^", "**"), locals={"x4": x4})
    assert sp.simplify(parsed - x4**-2) == 0


def test_pretty_print_parenthesizes_reciprocal_coordinate_power_leaf():
    sp = pytest.importorskip("sympy")
    z_inv = PowNode(Var(4), -1.0)
    root = AtomNode(
        kind="power",
        var_idxs=(4,),
        kwargs={"exponent_init": 2.0},
        tag="leaf0",
        inputs=(z_inv,),
    )

    expr = _render_analytic_root(root)

    assert "^-1^2" not in expr
    x4 = sp.Symbol("x4")
    parsed = sp.sympify(expr.replace("^", "**"), locals={"x4": x4})
    assert sp.simplify(parsed - x4**-2) == 0
