import numpy as np
import torch

from nestynet_sr.sr_core.bridges import Add, Exp, Mul, Pow, Var
from nestynet_sr.sr_de.poisson_ast import (
    ASTScalarBasis,
    ast_triangular_darboux_map,
    polynomial_bivector_to_ast,
)
from nestynet_sr.sr_de.poisson_basis import BivectorBasis, PolynomialScalarBasis
from nestynet_sr.sr_de.poisson_core import VectorField
from nestynet_sr.sr_de.poisson_darboux import canonical_degenerate_poisson, certify_darboux_map
from nestynet_sr.sr_de.poisson_rollout import rollout_vector_field


def test_ast_scalar_basis_returns_analytic_gradients():
    z = torch.tensor([[2.0, 3.0], [-1.0, 4.0]], dtype=torch.float64)
    basis = ASTScalarBasis(
        2,
        [Add(Var(0), Var(1)), Mul(Pow(Var(0), 2.0), Var(1))],
    )
    evaluated = basis.evaluate(z)

    expected_values = torch.stack((z[:, 0] + z[:, 1], z[:, 0].square() * z[:, 1]), dim=1)
    expected_gradients = torch.stack(
        (
            torch.ones_like(z),
            torch.stack((2.0 * z[:, 0] * z[:, 1], z[:, 0].square()), dim=1),
        ),
        dim=1,
    )
    torch.testing.assert_close(evaluated.values, expected_values)
    torch.testing.assert_close(evaluated.gradients, expected_gradients)


def test_quadratic_polynomial_tensor_round_trips_through_ast():
    torch.manual_seed(3)
    z = torch.randn(19, 3, dtype=torch.float64)
    scalar_basis = PolynomialScalarBasis(3, max_degree=2, include_constant=True)
    bivector_basis = BivectorBasis(3, scalar_basis)
    coefficients = torch.randn(bivector_basis.size, dtype=torch.float64)

    expected = bivector_basis.assemble(coefficients, z)
    ast_tensor = polynomial_bivector_to_ast(coefficients, scalar_basis)
    actual = ast_tensor.evaluate(z)

    torch.testing.assert_close(actual.tensor, expected.tensor, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(actual.derivatives, expected.derivatives, rtol=1.0e-12, atol=1.0e-12)


def test_general_vector_rollout_matches_harmonic_oscillator():
    field = VectorField(
        lambda z: torch.stack((z[:, 1], -z[:, 0]), dim=1),
        state_dim=2,
    )
    times = np.linspace(0.0, 2.0 * np.pi, 121)
    reference = np.stack((np.cos(times), -np.sin(times)), axis=1)
    result = rollout_vector_field(
        field,
        np.array([1.0, 0.0]),
        times,
        reference_states=reference,
    )

    assert result.success
    assert result.relative_rms_error is not None
    assert result.relative_rms_error < 1.0e-8


def test_ast_triangular_darboux_map_uses_analytic_ast_jacobian():
    chart = ast_triangular_darboux_map(
        [
            Exp(Var(0)),
            Add(Var(1), Pow(Var(0), 2.0)),
            Var(2),
        ],
        name="symbolic_chart",
    )
    rng = np.random.default_rng(7)
    states = rng.uniform(low=[-0.4, -1.0, -1.0], high=[0.4, 1.0, 1.0], size=(64, 3))
    certificate = certify_darboux_map(
        states,
        chart,
        canonical_degenerate_poisson(3, 1),
    )

    assert certificate.accepted
    assert certificate.local_diffeomorphism
    assert certificate.sampled_jacobi_relative_residual < 1.0e-8
