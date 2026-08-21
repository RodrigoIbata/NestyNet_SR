from dataclasses import replace
import math

import torch

from nestynet_sr.sr_de.poisson_certificates import (
    jacobiator,
    jacobi_residual,
    polynomial_jacobi_certificate,
    rank_profile,
)
from nestynet_sr.sr_de.poisson_basis import BivectorBasis, PolynomialScalarBasis
from nestynet_sr.sr_de.poisson_core import StableNullspaceConfig, VectorField
from nestynet_sr.sr_de.poisson_invariants import discover_casimirs
from nestynet_sr.sr_de.poisson_invariants import (
    HamiltonianFitConfig,
    fit_hamiltonian_given_poisson,
)
from nestynet_sr.sr_de.poisson_search import (
    PoissonSearchConfig,
    discover_poisson_structure,
    discover_poisson_structure_multi,
)


DTYPE = torch.float64


def _search_config(*lanes, hamiltonian_mode="independent"):
    return PoissonSearchConfig(
        lanes=tuple(lanes),
        validation_fraction=0.25,
        random_seed=41,
        max_representatives=24,
        sparse_rotation_steps=8,
        nullspace=StableNullspaceConfig(rank_rtol=1e-9, rank_atol=1e-11),
        hamiltonian=HamiltonianFitConfig(
            solver="stlsq",
            ridge=1e-12,
            stlsq_lambda=1e-7,
            relative_residual_tolerance=1e-7,
            absolute_residual_tolerance=1e-10,
        ),
        hamiltonian_mode=hamiltonian_mode,
        invariance_relative_tolerance=1e-8,
        jacobi_relative_tolerance=1e-8,
        polynomial_jacobi_tolerance=1e-9,
    )


def _linear_field(A):
    return VectorField(
        lambda z, A=A: z @ A.mT,
        lambda z, A=A: A,
        state_dim=A.shape[0],
    )


def test_shared_constant_lane_recovers_randomly_mixed_symplectic_geometry():
    generator = torch.Generator().manual_seed(3)
    J = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]],
        dtype=DTYPE,
    )
    mixing = torch.randn(4, 4, generator=generator, dtype=DTYPE)
    mixing = mixing + 2.0 * torch.eye(4, dtype=DTYPE)
    Pi = mixing @ J @ mixing.mT
    stiffnesses = []
    for _ in range(3):
        raw = torch.randn(4, 4, generator=generator, dtype=DTYPE)
        stiffnesses.append(raw.mT @ raw + 0.5 * torch.eye(4, dtype=DTYPE))
    fields = [_linear_field(Pi @ K) for K in stiffnesses]
    points = [torch.randn(180, 4, generator=generator, dtype=DTYPE) for _ in fields]

    result = discover_poisson_structure_multi(
        fields,
        points,
        _search_config("constant"),
    )

    assert result.accepted
    assert result.best is not None
    assert result.best.lane == "constant"
    expected = torch.stack([Pi[i, j] for i in range(4) for j in range(i + 1, 4)])
    expected = expected / torch.linalg.vector_norm(expected)
    recovered = result.best.coefficients / torch.linalg.vector_norm(result.best.coefficients)
    assert float(torch.abs(torch.dot(expected, recovered))) > 1.0 - 1e-9
    assert result.best.rank.generic_rank == 4
    assert all(value < 1e-8 for value in result.best.hamiltonian_validation_relative)


def _euler_field(inverse_inertia):
    inverse_inertia = torch.as_tensor(inverse_inertia, dtype=DTYPE)

    def value(z):
        return torch.linalg.cross(z, z * inverse_inertia)

    def jacobian(z):
        eye = torch.eye(3, dtype=z.dtype, device=z.device)
        omega = z * inverse_inertia
        columns = []
        for axis in range(3):
            perturbation = eye[axis].expand_as(z)
            columns.append(
                torch.linalg.cross(perturbation, omega)
                + torch.linalg.cross(z, perturbation * inverse_inertia)
            )
        return torch.stack(columns, dim=-1)

    return VectorField(value, jacobian, state_dim=3)


def test_shared_linear_lane_recovers_euler_lie_poisson_tensor_and_casimir():
    generator = torch.Generator().manual_seed(9)
    fields = [
        _euler_field([1.0, 0.7, 0.2]),
        _euler_field([0.9, 0.45, 0.15]),
        _euler_field([1.3, 0.8, 0.4]),
    ]
    points = [torch.randn(220, 3, generator=generator, dtype=DTYPE) for _ in fields]

    result = discover_poisson_structure_multi(
        fields,
        points,
        _search_config("linear", hamiltonian_mode="shared_support"),
    )

    assert result.accepted
    assert result.best is not None
    best = result.best
    expected = torch.zeros(9, dtype=DTYPE)
    # scalar basis is (z0,z1,z2), pair order is (01),(02),(12)
    expected[2] = -1.0
    expected[3 + 1] = 1.0
    expected[6 + 0] = -1.0
    expected = expected / torch.linalg.vector_norm(expected)
    recovered = best.coefficients / torch.linalg.vector_norm(best.coefficients)
    assert float(torch.abs(torch.dot(expected, recovered))) > 1.0 - 1e-9
    assert best.polynomial_jacobi.passed
    assert best.rank.generic_rank == 2
    assert best.casimirs is not None
    assert any(candidate.poisson_residual_rms < 1e-9 for candidate in best.casimirs.candidates)


def test_damped_oscillator_is_rejected_by_constant_poisson_lane():
    A = torch.tensor([[0.0, 1.0], [-1.0, -0.2]], dtype=DTYPE)
    generator = torch.Generator().manual_seed(12)
    points = torch.randn(160, 2, generator=generator, dtype=DTYPE)
    result = discover_poisson_structure(
        _linear_field(A),
        points,
        _search_config("constant"),
    )
    assert not result.accepted
    assert result.best is None
    assert result.lanes[0].nullspace.nullity == 0


def test_quadratic_two_dimensional_lane_and_callable_hamiltonian_terms():
    # In two dimensions every smooth skew bivector is Poisson.  This example
    # still tests that the quadratic determining lane recovers mu=1+x^2.
    def pi(z):
        mu = 1.0 + z[:, 0].square()
        out = z.new_zeros((z.shape[0], 2, 2))
        out[:, 0, 1] = mu
        out[:, 1, 0] = -mu
        return out

    def field(z):
        grad_h = torch.stack((z[:, 0], 2.0 * z[:, 1]), dim=1)
        return torch.einsum("nij,nj->ni", pi(z), grad_h)

    generator = torch.Generator().manual_seed(22)
    points = 0.7 * torch.randn(240, 2, generator=generator, dtype=DTYPE)
    result = discover_poisson_structure(
        field,
        points,
        _search_config("quadratic"),
    )
    assert result.accepted
    assert result.best is not None
    assert result.best.degree == 2
    # Exponent ordering is 1,x,y,x^2,xy,y^2 for the sole pair.
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=DTYPE)
    recovered = result.best.coefficients
    assert float(torch.abs(torch.dot(expected, recovered)) / (
        torch.linalg.vector_norm(expected) * torch.linalg.vector_norm(recovered)
    )) > 1.0 - 1e-7

    fit = fit_hamiltonian_given_poisson(
        pi,
        field(points),
        points,
        [lambda z: z[:, 0].square(), lambda z: z[:, 1].square()],
        HamiltonianFitConfig(solver="least_squares", ridge=0.0, stlsq_lambda=1e-10),
    )
    assert fit.accepted
    torch.testing.assert_close(fit.coeffs, torch.tensor([0.5, 1.0], dtype=DTYPE))


def test_polynomial_and_sampled_jacobi_certificates_agree_on_linear_tensors():
    # Euler's cross(M) tensor is Poisson.
    euler = torch.zeros(3, 3, dtype=DTYPE)
    euler[0, 2] = -1.0
    euler[1, 1] = 1.0
    euler[2, 0] = -1.0
    exact = polynomial_jacobi_certificate(
        euler,
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        tolerance=1e-12,
    )
    assert exact.passed

    # A generic linear skew tensor need not obey the Lie-algebra Jacobi rules.
    bad = torch.tensor(
        [[0.3, -0.7, 0.2], [0.4, 0.1, 0.8], [-0.5, 0.9, 0.6]],
        dtype=DTYPE,
    )
    assert not polynomial_jacobi_certificate(
        bad,
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        tolerance=1e-12,
    ).passed

    z = torch.randn(30, 3, generator=torch.Generator().manual_seed(5), dtype=DTYPE)
    # Build Pi and dPi from the bad coefficient table explicitly.
    Pi = z.new_zeros((z.shape[0], 3, 3))
    dPi = z.new_zeros((z.shape[0], 3, 3, 3))
    pairs = ((0, 1), (0, 2), (1, 2))
    for pair_index, (i, j) in enumerate(pairs):
        Pi[:, i, j] = z @ bad[pair_index]
        Pi[:, j, i] = -Pi[:, i, j]
        dPi[:, i, j, :] = bad[pair_index]
        dPi[:, j, i, :] = -bad[pair_index]
    assert not jacobi_residual(Pi, dPi, absolute_tolerance=1e-12).passed

    profile = rank_profile(Pi)
    assert profile.generic_rank == 2
    assert profile.even


def test_exact_quadratic_jacobi_coefficients_reproduce_sampled_jacobiator():
    generator = torch.Generator().manual_seed(101)
    scalar = PolynomialScalarBasis(3, max_degree=2, include_constant=True)
    bivectors = BivectorBasis(3, scalar)
    coefficient_table = torch.randn(
        3, scalar.size, generator=generator, dtype=DTYPE
    )
    certificate = polynomial_jacobi_certificate(
        coefficient_table,
        scalar.exponents,
        tolerance=0.0,
    )
    assert not certificate.passed

    z = 0.5 * torch.randn(37, 3, generator=generator, dtype=DTYPE)
    represented = bivectors.assemble(coefficient_table.reshape(-1), z)
    sampled = jacobiator(represented.tensor, represented.derivatives)[:, 0]
    reconstructed = torch.zeros_like(sampled)
    for (i, j, k, exponent), coefficient in certificate.coefficients.items():
        assert (i, j, k) == (0, 1, 2)
        monomial = torch.ones_like(sampled)
        for axis, power in enumerate(exponent):
            if power:
                monomial = monomial * z[:, axis].pow(power)
        reconstructed = reconstructed + coefficient * monomial
    torch.testing.assert_close(reconstructed, sampled, atol=2e-12, rtol=2e-12)


def test_sparsified_near_casimir_is_refit_and_rejected_on_residual():
    generator = torch.Generator().manual_seed(102)
    points = torch.randn(160, 3, generator=generator, dtype=DTYPE)
    tensor = torch.zeros((points.shape[0], 3, 3), dtype=DTYPE)
    tensor[:, 0, 1] = 1.0
    tensor[:, 1, 0] = -1.0
    epsilon = 1.0e-3
    terms = (
        lambda z: z[:, 2] + epsilon * z[:, 0],
        lambda z: z[:, 0],
    )

    result = discover_casimirs(
        tensor,
        points,
        terms,
        coefficient_threshold=1.0e-2,
        relative_residual_tolerance=1.0e-6,
        absolute_residual_tolerance=1.0e-12,
    )

    assert result.candidates == ()
    assert len(result.rejected_candidates) == 1
    rejected = result.rejected_candidates[0]
    assert rejected.accepted is False
    assert "poisson_residual" in rejected.failure_reasons
    assert rejected.poisson_residual_relative > 1.0e-6


def test_unstable_determining_subspace_blocks_poisson_candidate_promotion(monkeypatch):
    import nestynet_sr.sr_de.poisson_search as poisson_search

    original = poisson_search.stable_nullspace

    def unstable(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, heldout_principal_angle=0.5 * math.pi)

    monkeypatch.setattr(poisson_search, "stable_nullspace", unstable)
    points = torch.randn(
        180, 2, generator=torch.Generator().manual_seed(103), dtype=DTYPE
    )
    field = lambda z: torch.stack((z[:, 1], -z[:, 0]), dim=1)

    result = discover_poisson_structure(field, points, _search_config("constant"))

    assert result.accepted is False
    assert result.candidates
    assert all("nullspace_unstable" in row.failure_reasons for row in result.candidates)
