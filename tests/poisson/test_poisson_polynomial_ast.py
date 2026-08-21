import torch

from nestynet_sr.sr_de.poisson_ast import polynomial_bivector_to_ast
from nestynet_sr.sr_de.poisson_basis import BivectorBasis, PolynomialScalarBasis
from nestynet_sr.sr_de.poisson_certificates import (
    jacobi_residual,
    polynomial_jacobi_certificate,
)


def _set_term(coefficients, basis, pair, exponent, value):
    pair_index = basis.pairs.index(pair)
    exponent_index = basis.scalar_basis.exponents.index(exponent)
    coefficients[pair_index, exponent_index] = value


def test_exact_linear_certificate_accepts_euler_and_rejects_almost_poisson_tensor():
    scalar = PolynomialScalarBasis(3, max_degree=1, include_constant=True)
    basis = BivectorBasis(3, scalar)

    euler = torch.zeros((len(basis.pairs), scalar.size), dtype=torch.float64)
    _set_term(euler, basis, (0, 1), (0, 0, 1), -1.0)
    _set_term(euler, basis, (0, 2), (0, 1, 0), 1.0)
    _set_term(euler, basis, (1, 2), (1, 0, 0), -1.0)
    assert polynomial_jacobi_certificate(euler, scalar.exponents).passed

    non_poisson = euler.clone()
    _set_term(non_poisson, basis, (1, 2), (1, 0, 0), 0.0)
    _set_term(non_poisson, basis, (1, 2), (0, 1, 0), 1.0)
    rejected = polynomial_jacobi_certificate(non_poisson, scalar.exponents)
    assert not rejected.passed
    assert rejected.max_abs > 0.5


def test_quadratic_rank_two_tensor_has_global_and_ast_jacobi_certificates():
    # J=(1+x)*(x,y,z), Pi v=J cross v. This is J=mu*grad(C), so Jacobi
    # holds by construction while the tensor contains genuine quadratic terms.
    scalar = PolynomialScalarBasis(3, max_degree=2, include_constant=True)
    basis = BivectorBasis(3, scalar)
    coefficients = torch.zeros((len(basis.pairs), scalar.size), dtype=torch.float64)
    _set_term(coefficients, basis, (0, 1), (0, 0, 1), -1.0)
    _set_term(coefficients, basis, (0, 1), (1, 0, 1), -1.0)
    _set_term(coefficients, basis, (0, 2), (0, 1, 0), 1.0)
    _set_term(coefficients, basis, (0, 2), (1, 1, 0), 1.0)
    _set_term(coefficients, basis, (1, 2), (1, 0, 0), -1.0)
    _set_term(coefficients, basis, (1, 2), (2, 0, 0), -1.0)

    exact = polynomial_jacobi_certificate(coefficients, scalar.exponents)
    assert exact.passed
    assert exact.max_abs < 1.0e-14

    z = torch.randn(101, 3, dtype=torch.float64)
    ast_tensor = polynomial_bivector_to_ast(coefficients.reshape(-1), scalar)
    evaluated = ast_tensor.evaluate(z)
    sampled = jacobi_residual(evaluated.tensor, evaluated.derivatives)
    assert sampled.relative < 1.0e-12
