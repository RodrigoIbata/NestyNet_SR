# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import math

import torch

from nestynet_sr.sr_core.bridges import Add, ConstNode, Mul, Pow, Var, _eval_single_input
from nestynet_sr.sr_gs.de_invariant_compiler import (
    InvariantCompilerConfig,
    SymbolicInvariantObjective,
    compile_orbit_coordinate,
    compile_point_invariants,
    default_point_candidate_vocabulary,
)
from nestynet_sr.sr_gs.de_upgrades import PolynomialPointGenerator


DTYPE = torch.float64


def _points(n: int, *, phase: float = 0.0) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, n, dtype=DTYPE)
    x = 0.7 + 1.6 * t
    u = 0.35 + 0.6 * torch.sin(2.3 * t + phase) + 0.4 * t
    return torch.stack((x, u), dim=1)


def _projective_generator() -> PolynomialPointGenerator:
    # X = x^2 d_x + x*u d_u.  Its invariant is u/x and -1/x is
    # a rectifying orbit coordinate because X(-1/x)=1.
    return PolynomialPointGenerator(
        "projective",
        "quadratic_point",
        xi_terms=((1.0, 2, 0),),
        eta_terms=((1.0, 1, 1),),
    )


def test_default_vocabulary_is_bounded_and_domain_safe_on_finite_points() -> None:
    vocabulary = default_point_candidate_vocabulary()
    assert 10 <= len(vocabulary) <= 32
    points = _points(48)
    for ast in vocabulary:
        value, gradient, _ = _eval_single_input(ast, points, need_grad=True)
        assert torch.isfinite(value).all()
        assert gradient is not None and torch.isfinite(gradient).all()

    degree_two = {repr(ast) for ast in default_point_candidate_vocabulary(max_polynomial_degree=2)}
    degree_three = {repr(ast) for ast in default_point_candidate_vocabulary(max_polynomial_degree=3)}
    assert repr(Pow(Var(0), 2)) in degree_two
    assert repr(Pow(Var(0), 3)) not in degree_two
    assert repr(Pow(Var(0), 3)) in degree_three


def test_recovers_projective_mobius_invariant_and_orbit_coordinate() -> None:
    generator = _projective_generator()
    x = Var(0)
    u = Var(1)
    ratio = Mul(u, Pow(x, -1))
    vocabulary = (x, u, Pow(x, 2), Mul(x, u), Pow(u, 2), ratio, Pow(x, -1))
    cfg = InvariantCompilerConfig(max_invariants=1, action_rtol=1.0e-9, action_atol=1.0e-10)

    result = compile_point_invariants(
        (generator,),
        _points(96),
        _points(73, phase=0.27),
        vocabulary,
        cfg,
    )
    assert result.status == "recovered"
    assert result.determining_nullity >= 1
    assert len(result.invariants) == 1
    invariant = result.invariants[0]
    assert invariant.accepted
    assert invariant.validation_action_rms < 1.0e-10
    assert invariant.validation_variance > cfg.min_variance

    orbit = compile_orbit_coordinate(
        generator,
        _points(96),
        _points(73, phase=0.27),
        vocabulary,
        cfg,
    )
    assert orbit.accepted
    assert orbit.ast is not None
    assert orbit.validation_residual_rms < 1.0e-10
    value, _gradient, _ = _eval_single_input(orbit.ast, _points(31), need_grad=True)
    expected = -1.0 / _points(31)[:, 0]
    assert torch.allclose(value.reshape(-1), expected, atol=1.0e-10, rtol=1.0e-10)


def test_sparse_nullspace_rotation_compiles_quadratic_rotation_invariant() -> None:
    # X = -u d_x + x d_u, with invariant x^2 + u^2.  Neither polynomial
    # term is invariant alone, so this exercises coupled nullspace recovery.
    generator = PolynomialPointGenerator(
        "rotation",
        "quadratic_carrier",
        xi_terms=((-1.0, 0, 1),),
        eta_terms=((1.0, 1, 0),),
    )
    x = Var(0)
    u = Var(1)
    vocabulary = (x, u, Pow(x, 2), Mul(x, u), Pow(u, 2))
    result = compile_point_invariants(
        (generator,),
        _points(101),
        _points(87, phase=0.31),
        vocabulary,
        InvariantCompilerConfig(max_invariants=1),
    )
    assert result.status == "recovered"
    invariant = result.invariants[0]
    assert len(invariant.support) == 2
    assert invariant.validation_action_rms < 1.0e-9

    values, _gradient, _ = _eval_single_input(invariant.ast, _points(57), need_grad=True)
    points = _points(57)
    expected = points[:, 0].square() + points[:, 1].square()
    # Invariants have arbitrary overall scale/sign.  The compiler fixes the
    # largest coefficient to one, so this representative matches directly.
    assert torch.allclose(values.reshape(-1), expected, atol=1.0e-8, rtol=1.0e-8)


def test_constants_collapsed_aliases_and_dependent_invariants_are_rejected() -> None:
    x = Var(0)
    u = Var(1)
    collapsed = Add(x, Mul(ConstNode(-1.0), x))
    translations = (
        PolynomialPointGenerator("d_x", "translation", xi_terms=((1.0, 0, 0),)),
        PolynomialPointGenerator("d_u", "translation", eta_terms=((1.0, 0, 0),)),
    )
    rejected = compile_point_invariants(
        translations,
        _points(80),
        _points(67, phase=0.2),
        (ConstNode(1.0), collapsed, x, u),
    )
    assert rejected.status == "rejected"
    reasons = {row["reason"] for row in rejected.discarded_terms}
    assert "constant_or_collapsed_value" in reasons or "constant_or_collapsed_gradient" in reasons
    assert not rejected.invariants

    ratio = Mul(u, Pow(x, -1))
    dependent = Pow(ratio, 2)
    scaling = PolynomialPointGenerator(
        "common_scaling",
        "scaling",
        xi_terms=((1.0, 1, 0),),
        eta_terms=((1.0, 0, 1),),
    )
    result = compile_point_invariants(
        (scaling,),
        _points(93),
        _points(79, phase=0.19),
        (ratio, dependent),
        InvariantCompilerConfig(max_invariants=2),
    )
    assert len(result.invariants) == 1
    assert any(row.reason == "functionally_dependent_on_selected_invariants" for row in result.candidates)


def test_multiple_invariants_must_have_pointwise_independent_gradients() -> None:
    # A zero point field is intentionally used as a compiler-level test: its
    # invariant space is all functions, so the only remaining gate is local
    # functional independence.  The compiler should select x and u, not two
    # functions of x.
    zero = PolynomialPointGenerator("zero", "test_zero")
    x = Var(0)
    u = Var(1)
    result = compile_point_invariants(
        (zero,),
        _points(90),
        _points(75, phase=0.24),
        (Pow(x, 2), x, u),
        InvariantCompilerConfig(max_invariants=2),
    )
    assert result.status == "recovered"
    assert len(result.invariants) == 2
    assert result.invariants[1].independent_rank == 2
    assert result.invariants[1].independence_fraction >= 0.99


def test_symbolic_objective_scores_action_collapse_and_independence() -> None:
    generator = _projective_generator()
    points = _points(83)
    x = Var(0)
    u = Var(1)
    ratio = Mul(u, Pow(x, -1))
    objective = SymbolicInvariantObjective((generator,), points)
    invariant_score = objective.evaluate(ratio)
    noninvariant_score = objective.evaluate(x)
    collapsed_score = objective.evaluate(ConstNode(1.0))
    assert invariant_score.total < 1.0e-12
    assert noninvariant_score.total > invariant_score.total + 0.1
    assert collapsed_score.variance_penalty > 0.99
    assert math.isfinite(float(objective(ratio)))

    independent_objective = SymbolicInvariantObjective(
        (generator,),
        points,
        reference_invariants=(ratio,),
    )
    dependent_score = independent_objective.evaluate(Pow(ratio, 2))
    assert dependent_score.action_loss < 1.0e-12
    assert dependent_score.independence_penalty > 0.99
