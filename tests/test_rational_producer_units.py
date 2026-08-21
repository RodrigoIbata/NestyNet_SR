# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    FixedConst,
    FreeConst,
    LogNode,
    MulNode,
    PowNode,
    Var,
    collect_all_atoms,
    eval_inputs,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search._candidate_builders_multivariate import (
    _build_log_ratpoly_candidate,
    _build_ratpoly_candidates,
    _build_sqrt_ratpoly_candidate,
)
from nestynet_sr.sr_search._candidate_builders_univariate import (
    _build_ratpoly_1d_candidates,
    _make_scaling_based_rewrite,
)
from nestynet_sr.sr_search.fitting_utils import _fit_rational_coeffs_1d
from nestynet_sr.sr_search.stageB.rules import propose_rational_linear
from nestynet_sr.sr_search.rational_supports import (
    plan_unit_consistent_rational_supports,
)
from nestynet_sr.sr_search.template_library import (
    propose_exp_of_quadratic,
    propose_symexp_denom_family,
    propose_tanh_family,
)


def _units():
    unit_system = UnitSystem(("L", "T"))
    zero = unit_system.dimless()
    length = unit_system.dim({"L": 1})
    return unit_system, zero, length


def _rational_atoms(root):
    return [
        atom
        for atom in collect_all_atoms(root)
        if str(getattr(atom, "kind", "")).lower() in {"ratpoly", "rratpoly"}
    ]


def _assert_explicit_rational_supports(root):
    atoms = _rational_atoms(root)
    assert atoms
    for atom in atoms:
        assert atom.kwargs["exps_num_override"]
        assert atom.kwargs["exps_den_override"]


def test_missing_rational_support_fails_closed_even_when_dimensionless():
    unit_system, zero, _length = _units()
    atom = AtomNode(
        kind="ratpoly",
        var_idxs=(0,),
        kwargs={"deg_num": 2, "deg_den": 2},
    )

    result = check_units_ast(
        atom,
        UnitsSpec(unit_system=unit_system, x_dims=(zero,), y_dim=zero),
    )

    assert result.ok is False
    assert "explicit numerator and denominator supports" in result.reason


def test_unitless_support_planner_keeps_constant_only_numerator_recipe():
    _unit_system, zero, _length = _units()

    plan = plan_unit_consistent_rational_supports(
        target_dim=zero,
        input_dims=(zero,),
        max_deg_num=1,
        max_deg_den=1,
    )

    assert any(
        support.numerator_exponents == ((0,),)
        and support.denominator_exponents == ((0,), (1,))
        for support in plan.supports
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ([[0.5], [1]], "exact integer"),
        ([[False], [1]], "exact integer"),
        ([[-1], [1]], "nonnegative"),
        ([[0], [0]], "repeats exponent"),
        ([[0], [2]], "outside"),
    ],
)
def test_rational_1d_exact_support_override_rejects_invalid_rows(override, message):
    x = torch.linspace(0.25, 2.0, 32, dtype=torch.float64)
    y = 1.0 / (1.0 + x)

    with pytest.raises(ValueError, match=message):
        _fit_rational_coeffs_1d(
            x,
            y,
            deg_num=1,
            deg_den=1,
            min_points=1,
            exps_num_override=override,
            exps_den_override=[[0], [1]],
        )


def test_fixed_constant_can_participate_in_effective_input_evaluation():
    expr = MulNode(Var(0), FixedConst("twice", value=2.0))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0,),
        kwargs={},
        inputs=(expr,),
    )
    x = torch.tensor([[1.0], [3.0]], dtype=torch.float64)

    values, grad, hess = eval_inputs(
        atom,
        x,
        need_grad=True,
        need_hess=True,
    )

    assert torch.equal(values, torch.tensor([[2.0], [6.0]], dtype=torch.float64))
    assert torch.equal(grad, torch.full((2, 1, 1), 2.0, dtype=torch.float64))
    assert torch.equal(hess, torch.zeros((2, 1, 1, 1), dtype=torch.float64))


def test_trainable_constant_effective_input_fails_with_explicit_reason():
    atom = AtomNode(
        kind="nn",
        var_idxs=(0,),
        kwargs={},
        inputs=(FreeConst("scale", init=2.0),),
    )

    with pytest.raises(ValueError, match="trainable free constant"):
        eval_inputs(atom, torch.ones((2, 1), dtype=torch.float64))


def test_negative_scaling_recipe_declares_its_exact_inverse_support():
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="scale_target")

    rewritten = _make_scaling_based_rewrite(
        target,
        target,
        SimpleNamespace(k_hat=-1.0),
    )

    assert isinstance(rewritten, AtomNode)
    assert rewritten.kind == "ratpoly"
    assert rewritten.kwargs["exps_num_override"] == [[0]]
    assert rewritten.kwargs["exps_den_override"] == [[1]]


def test_rational_linear_template_declares_its_dense_support():
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="template_target")
    ctx = SimpleNamespace(state=SimpleNamespace(root=target))
    hint = SimpleNamespace(ok=True, best_name="recip")

    candidate = propose_rational_linear(ctx, target, hint)

    assert candidate is not None
    atoms = _rational_atoms(candidate.root)
    assert len(atoms) == 1
    assert atoms[0].kwargs["exps_num_override"] == [[0, 0]]
    assert atoms[0].kwargs["exps_den_override"] == [[0, 0], [0, 1], [1, 0]]


def test_every_static_rational_template_declares_explicit_supports():
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="template_target")
    ctx = SimpleNamespace(state=SimpleNamespace(root=target))

    candidates = []
    candidates.extend(
        propose_exp_of_quadratic(
            ctx,
            target,
            SimpleNamespace(ok=True),
        )
    )
    candidates.extend(
        propose_tanh_family(
            ctx,
            target,
            transform_hint=SimpleNamespace(ok=True, best_name="atanh"),
        )
    )
    candidates.extend(
        propose_symexp_denom_family(
            ctx,
            target,
            transform_hint=SimpleNamespace(ok=True, best_name="recip"),
        )
    )

    rational_roots = [candidate.root for candidate in candidates if _rational_atoms(candidate.root)]
    assert len(rational_roots) == 3
    for root in rational_roots:
        _assert_explicit_rational_supports(root)


def test_unitful_multivariate_builder_certifies_every_final_support():
    unit_system, zero, length = _units()
    generator = torch.Generator().manual_seed(611)
    theta = torch.rand(900, generator=generator, dtype=torch.float64) + 0.2
    radius = torch.rand(900, generator=generator, dtype=torch.float64) + 0.5
    x = torch.stack((theta, radius), dim=1)
    output = radius * (1.0 + theta) / (2.0 + theta)

    class Teacher(torch.nn.Module):
        def forward(self, values):
            return (
                values[:, 1]
                * (1.0 + values[:, 0])
                / (2.0 + values[:, 0])
            ).unsqueeze(1)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="unitful_nd")
    results = _build_ratpoly_candidates(
        root=target,
        target=target,
        reuse={"unitful_nd": Teacher()},
        train_loader=DataLoader(
            TensorDataset(x, output.unsqueeze(1)),
            batch_size=len(x),
            shuffle=False,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=2,
        max_deg_den=2,
        min_points=200,
        rel_rms_threshold=1.0e-6,
        enforce_units=True,
        target_dim=length,
        x_dims=[zero, length],
    )

    assert results
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(zero, length),
        y_dim=length,
    )
    for root, _init, metadata in results:
        assert metadata["coefficient_unit_certificate"]["valid"] is True
        assert metadata["unit_support_planned"] is True
        assert check_units_ast(root, spec).ok is True
        for atom in _rational_atoms(root):
            assert atom.kwargs["exps_num_override"]
            assert atom.kwargs["exps_den_override"]


def test_unitful_univariate_builder_certifies_every_final_support():
    unit_system, _zero, length = _units()
    length_squared = tuple(Fraction(2) * value for value in length)
    x = torch.linspace(0.4, 2.0, 900, dtype=torch.float64).unsqueeze(1)
    output = 3.0 * x[:, 0].square()

    class Teacher(torch.nn.Module):
        def forward(self, values):
            return 3.0 * values[:, :1].square()

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="unitful_1d")
    results = _build_ratpoly_1d_candidates(
        root=target,
        target=target,
        reuse={"unitful_1d": Teacher()},
        train_loader=DataLoader(
            TensorDataset(x, output.unsqueeze(1)),
            batch_size=len(x),
            shuffle=False,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=2,
        max_deg_den=2,
        min_points=200,
        rel_rms_threshold=1.0e-6,
        enforce_units=True,
        target_dim=length_squared,
        x_dims=[length],
    )

    assert results
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(length,),
        y_dim=length_squared,
    )
    for root, _init, metadata in results:
        assert metadata["coefficient_unit_certificate"]["valid"] is True
        assert metadata["unit_support_planned"] is True
        assert check_units_ast(root, spec).ok is True


def test_unitful_univariate_builder_auto_raises_to_first_viable_degree():
    unit_system, _zero, length = _units()
    length_sixth = tuple(Fraction(6) * value for value in length)
    x = torch.linspace(0.4, 2.0, 900, dtype=torch.float64).unsqueeze(1)
    output = 2.0 * x[:, 0].pow(6)

    class Teacher(torch.nn.Module):
        def forward(self, values):
            return 2.0 * values[:, :1].pow(6)

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="unitful_degree6")
    results = _build_ratpoly_1d_candidates(
        root=target,
        target=target,
        reuse={"unitful_degree6": Teacher()},
        train_loader=DataLoader(
            TensorDataset(x, output.unsqueeze(1)),
            batch_size=len(x),
            shuffle=False,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=200,
        rel_rms_threshold=1.0e-7,
        enforce_units=True,
        target_dim=length_sixth,
        x_dims=[length],
    )

    assert results
    assert any(metadata["deg_num"] >= 6 for _root, _init, metadata in results)
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(length,),
        y_dim=length_sixth,
    )
    for root, _init, metadata in results:
        assert metadata["coefficient_unit_certificate"]["valid"] is True
        assert check_units_ast(root, spec).ok is True


def test_unitful_sqrt_rational_builder_plans_the_lifted_support_first():
    unit_system, zero, length = _units()
    generator = torch.Generator().manual_seed(612)
    theta = torch.rand(900, generator=generator, dtype=torch.float64) + 0.2
    radius = torch.rand(900, generator=generator, dtype=torch.float64) + 0.5
    x = torch.stack((theta, radius), dim=1)
    output = radius * (1.0 + theta) / (2.0 + theta)

    class Teacher(torch.nn.Module):
        def forward(self, values):
            return (
                values[:, 1]
                * (1.0 + values[:, 0])
                / (2.0 + values[:, 0])
            ).unsqueeze(1)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="sqrt_unitful")
    root, _init, metadata = _build_sqrt_ratpoly_candidate(
        root=target,
        target=target,
        reuse={"sqrt_unitful": Teacher()},
        train_loader=DataLoader(
            TensorDataset(x, output.unsqueeze(1)),
            batch_size=len(x),
            shuffle=False,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=200,
        rel_rms_threshold=1.0e-5,
        enforce_units=True,
        target_dim=length,
        x_dims=[zero, length],
    )

    assert isinstance(root, PowNode)
    assert metadata["unit_support_planned"] is True
    assert metadata["coefficient_unit_certificate"]["valid"] is True
    _assert_explicit_rational_supports(root)
    assert check_units_ast(
        root,
        UnitsSpec(
            unit_system=unit_system,
            x_dims=(zero, length),
            y_dim=length,
        ),
    ).ok is True


def test_unit_aware_log_rational_builder_uses_dimensionless_support_class():
    unit_system, zero, length = _units()
    generator = torch.Generator().manual_seed(613)
    theta = torch.rand(900, generator=generator, dtype=torch.float64) + 0.2
    radius = torch.rand(900, generator=generator, dtype=torch.float64) + 0.5
    x = torch.stack((theta, radius), dim=1)
    output = torch.log((1.0 + theta) / (2.0 + theta))

    class Teacher(torch.nn.Module):
        def forward(self, values):
            return torch.log(
                (1.0 + values[:, :1]) / (2.0 + values[:, :1])
            )

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="log_unitful_inputs")
    root, _init = _build_log_ratpoly_candidate(
        root=target,
        target=target,
        reuse={"log_unitful_inputs": Teacher()},
        train_loader=DataLoader(
            TensorDataset(x, output.unsqueeze(1)),
            batch_size=len(x),
            shuffle=False,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=200,
        rel_rms_threshold=1.0e-5,
        enforce_units=True,
        target_dim=zero,
        x_dims=[zero, length],
    )

    assert isinstance(root, LogNode)
    _assert_explicit_rational_supports(root)
    rational_atom = _rational_atoms(root)[0]
    assert all(row[1] == 0 for row in rational_atom.kwargs["exps_num_override"])
    assert all(row[1] == 0 for row in rational_atom.kwargs["exps_den_override"])
    assert check_units_ast(
        root,
        UnitsSpec(
            unit_system=unit_system,
            x_dims=(zero, length),
            y_dim=zero,
        ),
    ).ok is True
