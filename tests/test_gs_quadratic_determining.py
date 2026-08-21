import numpy as np
import pytest

from nestynet_sr.sr_gs.jet_bundle import JetSpaceSpec
from nestynet_sr.sr_gs.nonlinear_de_symmetry import (
    PolynomialDESymmetryConfig,
    pair_major_generator,
    project_generator_direction,
    recover_polynomial_de_symmetries,
)


def _order1_samples(seed=910, n=320):
    rng = np.random.default_rng(seed)
    return rng, rng.uniform(-0.9, 0.9, n), rng.uniform(0.35, 1.4, n)


def _order2_free_particle(seed=911, n=360):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    u = rng.uniform(-1.0, 1.0, n)
    u1 = rng.uniform(-1.2, 1.2, n)
    return rng, x, u, u1


def _direction(result, terms=None, **named_terms):
    values = np.zeros(len(result.coefficient_labels), dtype=float)
    all_terms = dict(terms or {})
    all_terms.update(named_terms)
    for label, value in all_terms.items():
        values[result.coefficient_labels.index(label.replace("_", ":", 1))] = float(value)
    return values


def _closest_candidate(result, direction):
    unit = direction / np.linalg.norm(direction)
    return max(result.candidates, key=lambda row: abs(np.dot(row.coefficients, unit)))


def test_pair_major_coefficients_are_exact_and_interoperate_with_polynomial_generator():
    monomials = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    # xi=x^2 and eta=x*u occupy adjacent pair-major slots for their monomials.
    coeffs = np.zeros(12)
    coeffs[6] = 1.0
    coeffs[9] = 1.0
    gen = pair_major_generator(coeffs, monomials, name="projective")

    import torch

    x = torch.tensor([[2.0]], dtype=torch.float64)
    u = torch.tensor([[3.0]], dtype=torch.float64)
    zero = torch.zeros_like(x)
    xi, eta, _eta1, _eta2 = gen.fields(x, u, zero, zero)
    assert xi.item() == pytest.approx(4.0)
    assert eta.item() == pytest.approx(6.0)
    assert gen.xi_terms == ((1.0, 2, 0),)
    assert gen.eta_terms == ((1.0, 1, 1),)


def test_quadratic_free_particle_recovers_projective_generator_missed_by_affine_lane():
    rng, x, u, u1 = _order2_free_particle()
    jet = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2)
    on = {"x": x, "u": u, "u_x": u1, "u_xx": np.zeros_like(x)}
    off = {"x": x, "u": u, "u_x": u1, "u_xx": rng.uniform(-1.1, 1.1, x.size)}
    common = dict(
        heldout_fraction=0.2,
        bootstrap=3,
        rank_rtol=1.0e-10,
        rank_atol=1.0e-11,
        on_shell_tol=1.0e-9,
        off_shell_tol=1.0e-9,
    )
    affine = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual="u_xx",
        on_shell_samples=on,
        off_shell_samples=off,
        config=PolynomialDESymmetryConfig(generator_degree=1, **common),
    )
    quadratic = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual="u_xx",
        on_shell_samples=on,
        off_shell_samples=off,
        config=PolynomialDESymmetryConfig(generator_degree=2, **common),
    )

    assert affine.certified_nullity == 6
    assert quadratic.certified_nullity == 8
    projective = _direction(quadratic, {"xi:x^2": 1.0, "eta:x*u": 1.0})
    _projection, residual = project_generator_direction(quadratic, projective)
    assert residual < 1.0e-9
    assert max(quadratic.bootstrap_principal_angles) < 1.0e-7
    assert quadratic.on_shell_heldout_residual_rel < 1.0e-10
    assert quadratic.status == "recovered"


def test_functional_multiplier_recovers_x_multiplier_that_constant_lane_rejects():
    rng, x, u = _order1_samples()
    u1_off = rng.uniform(-1.2, 1.6, x.size)
    jet = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1)
    on = {"x": x, "u": u, "u_x": u}
    off = {"x": x, "u": u, "u_x": u1_off}
    residual = "exp(0.5*x*x)*(u_x-u)"
    base = dict(generator_degree=1, heldout_fraction=0.2, on_shell_tol=1.0e-9, off_shell_tol=1.0e-9)
    constant = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual=residual,
        on_shell_samples=on,
        off_shell_samples=off,
        config=PolynomialDESymmetryConfig(multiplier_degree=0, **base),
    )
    functional = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual=residual,
        on_shell_samples=on,
        off_shell_samples=off,
        config=PolynomialDESymmetryConfig(multiplier_degree=1, **base),
    )

    d_x = _direction(functional, xi_1=1.0)
    _projection, functional_residual = project_generator_direction(functional, d_x)
    _projection, constant_residual = project_generator_direction(constant, d_x)
    assert functional_residual < 1.0e-9
    assert constant_residual > 0.1
    candidate = _closest_candidate(functional, d_x)
    x_multiplier_index = functional.multiplier_monomials.index((1, 0, 0))
    assert candidate.multiplier_coefficients[x_multiplier_index] == pytest.approx(1.0, abs=1.0e-8)
    assert candidate.off_shell_relative_residual_rel < 1.0e-10


def test_degree_one_constant_multiplier_regresses_existing_affine_certificate():
    rng, x, u = _order1_samples(seed=912)
    jet = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1)
    result = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual="u_x-u",
        on_shell_samples={"x": x, "u": u, "u_x": u},
        off_shell_samples={"x": x, "u": u, "u_x": rng.uniform(-1.5, 1.5, x.size)},
        config=PolynomialDESymmetryConfig(
            generator_degree=1,
            multiplier_degree=0,
            on_shell_tol=1.0e-9,
            off_shell_tol=1.0e-9,
        ),
    )

    u_d_u = _direction(result, eta_u=1.0)
    _projection, residual = project_generator_direction(result, u_d_u)
    assert residual < 1.0e-9
    candidate = _closest_candidate(result, u_d_u)
    assert candidate.multiplier_coefficients == pytest.approx((1.0,), abs=1.0e-8)
    assert candidate.support_size == 1
    assert result.to_report()["coefficient_convention"] == "pair_major_xi_eta_per_monomial"


def test_sparse_rotation_and_evaluated_bracket_closure_for_free_particle_algebra():
    rng, x, u, u1 = _order2_free_particle(seed=913)
    jet = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2)
    result = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual="u_xx",
        on_shell_samples={"x": x, "u": u, "u_x": u1, "u_xx": np.zeros_like(x)},
        off_shell_samples={"x": x, "u": u, "u_x": u1, "u_xx": rng.uniform(-1.0, 1.0, x.size)},
        config=PolynomialDESymmetryConfig(
            generator_degree=2,
            multiplier_degree=2,
            sparse_threshold=0.05,
            sparse_iterations=32,
            on_shell_tol=1.0e-9,
            off_shell_tol=1.0e-9,
            bracket_closure_tol=1.0e-8,
        ),
    )

    supports = sorted(row.support_size for row in result.candidates)
    assert supports[0] == 1
    assert 2 in supports
    assert result.bracket_certificates
    assert result.bracket_closure_residual < 1.0e-9
    assert all(row.accepted for row in result.bracket_certificates)


def test_generic_nonpolynomial_first_order_control_has_no_bounded_polynomial_symmetry():
    rng, x, u = _order1_samples(seed=914, n=420)
    on_u1 = np.exp(x * u + 0.2 * x**2)
    jet = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1)
    result = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual="u_x-exp(x*u+0.2*x*x)",
        on_shell_samples={"x": x, "u": u, "u_x": on_u1},
        off_shell_samples={"x": x, "u": u, "u_x": rng.uniform(-0.5, 3.0, x.size)},
        config=PolynomialDESymmetryConfig(
            generator_degree=2,
            multiplier_degree=2,
            rank_rtol=1.0e-11,
            rank_atol=1.0e-12,
            on_shell_tol=1.0e-9,
            off_shell_tol=1.0e-9,
        ),
    )

    assert result.status == "rejected"
    assert result.on_shell_nullity == 0
    assert result.candidates == ()


def test_order_two_validation_and_wide_structural_nullspace_are_retained():
    rng, x, u, u1 = _order2_free_particle(seed=915, n=18)
    jet = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2)
    result = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual="u_xx",
        on_shell_samples={"x": x, "u": u, "u_x": u1, "u_xx": np.zeros_like(x)},
        off_shell_samples={"x": x, "u": u, "u_x": u1, "u_xx": rng.uniform(-1.0, 1.0, x.size)},
        config=PolynomialDESymmetryConfig(
            generator_degree=2,
            multiplier_degree=2,
            heldout_fraction=0.0,
            min_samples=16,
            on_shell_tol=1.0e-8,
            off_shell_tol=1.0e-8,
        ),
    )
    assert result.on_shell_basis.shape == (12, 8)
    assert result.certified_nullity == 8

    with pytest.raises(KeyError, match="u_xx"):
        recover_polynomial_de_symmetries(
            jet_space=jet,
            residual="u_xx",
            on_shell_samples={"x": x, "u": u, "u_x": u1},
            off_shell_samples={"x": x, "u": u, "u_x": u1, "u_xx": np.ones_like(x)},
        )


def test_bootstrap_instability_blocks_promotion_but_retains_candidate_audit(monkeypatch):
    import nestynet_sr.sr_gs.nonlinear_de_symmetry as nonlinear

    rng, x, u = _order1_samples(seed=916)
    monkeypatch.setattr(
        nonlinear,
        "_bootstrap_angles",
        lambda *_args, **_kwargs: [0.5 * np.pi],
    )
    result = recover_polynomial_de_symmetries(
        jet_space=JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1),
        residual="u_x-u",
        on_shell_samples={"x": x, "u": u, "u_x": u},
        off_shell_samples={"x": x, "u": u, "u_x": rng.uniform(-1.5, 1.5, x.size)},
        config=PolynomialDESymmetryConfig(
            generator_degree=2,
            multiplier_degree=2,
            bootstrap=2,
            bootstrap_angle_tol=0.2,
        ),
    )

    assert any(row.accepted for row in result.candidates)
    assert result.bootstrap_stable is False
    assert result.promotable_generators is False
    assert result.status == "rejected"
    assert result.reason == "bootstrap_unstable_generator_subspace"


def test_nonclosed_truncation_keeps_individual_generators_but_not_full_algebra(monkeypatch):
    import nestynet_sr.sr_gs.nonlinear_de_symmetry as nonlinear

    rng, x, u = _order1_samples(seed=917)
    monkeypatch.setattr(
        nonlinear,
        "_evaluated_bracket_certificates",
        lambda *_args, **_kwargs: ([], 1.0),
    )
    result = recover_polynomial_de_symmetries(
        jet_space=JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1),
        residual="u_x-u",
        on_shell_samples={"x": x, "u": u, "u_x": u},
        off_shell_samples={"x": x, "u": u, "u_x": rng.uniform(-1.5, 1.5, x.size)},
        config=PolynomialDESymmetryConfig(generator_degree=2, multiplier_degree=2),
    )

    assert result.individual_generators_accepted is True
    assert result.promotable_generators is True
    assert result.closed_truncated_algebra is False
    assert result.promotable_full_algebra is False
    assert result.status == "recovered_generators_nonclosed"
