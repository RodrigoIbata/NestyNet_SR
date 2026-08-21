import math

import torch

from nestynet_sr.sr_de.poisson_basis import (
    BivectorBasis,
    CallableScalarBasis,
    PolynomialScalarBasis,
    assemble_skew_tensor,
    build_poisson_determining_matrix,
    extract_upper_triangle,
    upper_triangle_pairs,
)
from nestynet_sr.sr_de.poisson_core import (
    StableNullspaceConfig,
    VectorField,
    stable_nullspace,
)


DTYPE = torch.float64


def test_vector_field_value_and_jacobian_explicit_and_autograd_agree():
    def value(z):
        return torch.stack(
            [z[:, 0] * z[:, 1], torch.sin(z[:, 0]) + z[:, 2] ** 2, 0.5 * z[:, 1]],
            dim=1,
        )

    def jacobian(z):
        out = torch.zeros(z.shape[0], 3, 3, dtype=z.dtype, device=z.device)
        out[:, 0, 0] = z[:, 1]
        out[:, 0, 1] = z[:, 0]
        out[:, 1, 0] = torch.cos(z[:, 0])
        out[:, 1, 2] = 2.0 * z[:, 2]
        out[:, 2, 1] = 0.5
        return out

    z = torch.tensor([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]], dtype=DTYPE)
    automatic = VectorField(value, state_dim=3)
    explicit = VectorField.from_callable(value, jacobian, state_dim=3)

    f_auto, df_auto = automatic.value_and_jacobian(z)
    f_explicit, df_explicit = explicit.value_and_jacobian(z)
    torch.testing.assert_close(f_auto, f_explicit)
    torch.testing.assert_close(df_auto, df_explicit)
    assert f_auto.dtype == DTYPE
    assert df_auto.shape == (2, 3, 3)


def test_polynomial_basis_through_quadratic_has_analytic_gradients_at_zero():
    basis = PolynomialScalarBasis(2, max_degree=2)
    assert basis.exponents == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    z = torch.tensor([[0.0, 0.0], [2.0, -3.0]], dtype=DTYPE)
    evaluated = basis.evaluate(z)

    expected_values = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, -3.0, 4.0, -6.0, 9.0]],
        dtype=DTYPE,
    )
    torch.testing.assert_close(evaluated.values, expected_values)
    torch.testing.assert_close(
        evaluated.gradients[1],
        torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [4.0, 0.0], [-3.0, 2.0], [0.0, -6.0]],
            dtype=DTYPE,
        ),
    )
    assert torch.isfinite(evaluated.gradients).all()


def test_callable_scalar_basis_autograd_adapter():
    basis = CallableScalarBasis(
        2,
        2,
        lambda z: torch.stack([torch.exp(z[:, 0]), z[:, 0] * z[:, 1]], dim=1),
        names=("exp(z0)", "z0*z1"),
    )
    z = torch.tensor([[0.0, 2.0], [1.0, -3.0]], dtype=DTYPE)
    evaluated = basis(z)
    torch.testing.assert_close(
        evaluated.gradients,
        torch.tensor([[[1.0, 0.0], [2.0, 0.0]], [[math.e, 0.0], [-3.0, 1.0]]], dtype=DTYPE),
    )


def test_skew_assembly_and_pair_major_bivector_evaluation():
    pairs = upper_triangle_pairs(3)
    assert pairs == ((0, 1), (0, 2), (1, 2))
    upper = torch.tensor([[2.0, -1.0, 4.0]], dtype=DTYPE)
    tensor = assemble_skew_tensor(upper, state_dim=3)
    torch.testing.assert_close(tensor + tensor.mT, torch.zeros_like(tensor))
    torch.testing.assert_close(extract_upper_triangle(tensor), upper)

    scalar_basis = PolynomialScalarBasis(3, max_degree=1)
    bivectors = BivectorBasis(3, scalar_basis)
    assert bivectors.size == 12
    # Pair (0,1), scalar z2 is flat index 3.
    coefficients = torch.zeros(12, dtype=DTYPE)
    coefficients[bivectors.flat_index(0, 3)] = 1.0
    z = torch.tensor([[1.0, 2.0, 5.0]], dtype=DTYPE)
    represented = bivectors.assemble(coefficients, z)
    assert represented.tensor[0, 0, 1] == 5.0
    assert represented.tensor[0, 1, 0] == -5.0
    assert represented.derivatives[0, 0, 1, 2] == 1.0
    assert represented.derivatives[0, 1, 0, 2] == -1.0


def _euler_field_and_jacobian(z: torch.Tensor, inverse_inertia: torch.Tensor):
    # Pi(M) v = M x v, H = 0.5 sum_i inverse_inertia_i M_i^2.
    omega = z * inverse_inertia
    f = torch.linalg.cross(z, omega)
    e = torch.eye(3, dtype=z.dtype, device=z.device)
    rows = []
    for coordinate in range(3):
        perturbation = e[coordinate].expand_as(z)
        rows.append(
            torch.linalg.cross(perturbation, omega)
            + torch.linalg.cross(z, perturbation * inverse_inertia)
        )
    jacobian = torch.stack(rows, dim=-1)
    return f, jacobian


def test_lie_derivative_determining_matrix_annihilates_euler_lie_poisson_tensor():
    generator = torch.Generator().manual_seed(7)
    z = torch.randn(48, 3, generator=generator, dtype=DTYPE)
    f, df = _euler_field_and_jacobian(z, torch.tensor([1.0, 0.7, 0.3], dtype=DTYPE))
    # Linear-only basis: terms z0,z1,z2.  Pi is cross(M), hence
    # Pi01=-z2, Pi02=z1, Pi12=-z0.
    bivectors = BivectorBasis(3, PolynomialScalarBasis(3, 1, include_constant=False))
    coefficients = torch.zeros(bivectors.size, dtype=DTYPE)
    coefficients[bivectors.flat_index(0, 2)] = -1.0
    coefficients[bivectors.flat_index(1, 1)] = 1.0
    coefficients[bivectors.flat_index(2, 0)] = -1.0
    determining = build_poisson_determining_matrix(f, df, bivectors.evaluate(z))
    efficient_determining = bivectors.determining_matrix(z, f, df)
    torch.testing.assert_close(efficient_determining, determining)

    residual = torch.linalg.vector_norm(determining @ coefficients)
    scale = torch.linalg.matrix_norm(determining) * torch.linalg.vector_norm(coefficients)
    assert float(residual / scale) < 1.0e-14


def test_stable_nullspace_recovers_projector_and_subspace_diagnostics():
    generator = torch.Generator().manual_seed(29)
    left_train = torch.randn(80, 3, generator=generator, dtype=DTYPE)
    left_heldout = torch.randn(40, 3, generator=generator, dtype=DTYPE)

    def determining(left):
        # Column 3 = column 0 + 2 column 1, giving null vector (-1,-2,0,1).
        return torch.column_stack((left, left[:, 0] + 2.0 * left[:, 1]))

    result = stable_nullspace(
        determining(left_train),
        determining(left_heldout),
        config=StableNullspaceConfig(bootstrap=5, random_seed=11),
    )
    expected = torch.tensor([-1.0, -2.0, 0.0, 1.0], dtype=DTYPE)
    expected = expected / torch.linalg.vector_norm(expected)

    assert result.nullity == 1
    torch.testing.assert_close(result.projector, expected[:, None] @ expected[None, :], atol=1e-12, rtol=1e-12)
    assert result.train_residual_relative < 1.0e-15
    assert result.heldout_residual_relative is not None
    assert result.heldout_residual_relative < 1.0e-15
    assert result.heldout_principal_angle is not None
    assert result.heldout_principal_angle < 1.0e-7
    assert len(result.bootstrap_principal_angles) == 5
    assert result.max_bootstrap_principal_angle is not None
    assert result.max_bootstrap_principal_angle < 1.0e-7


def test_nullspace_bootstrap_blocks_must_partition_rows():
    matrix = torch.eye(3, dtype=DTYPE)
    try:
        stable_nullspace(
            matrix,
            config=StableNullspaceConfig(bootstrap=1, bootstrap_block_size=2),
        )
    except ValueError as exc:
        assert "not divisible" in str(exc)
    else:
        raise AssertionError("expected a block-size validation error")


def test_spectral_gap_strategy_retains_exact_all_zero_nullspace():
    result = stable_nullspace(
        torch.zeros(4, 3, dtype=DTYPE),
        config=StableNullspaceConfig(nullity_strategy="spectral_gap"),
    )
    assert result.rank == 0
    assert result.nullity == 3
    torch.testing.assert_close(result.projector, torch.eye(3, dtype=DTYPE))


def test_stable_nullspace_uses_economy_svd_for_tall_matrix(monkeypatch):
    calls = []
    original = torch.linalg.svd

    def recording_svd(matrix, *, full_matrices=True):
        calls.append(bool(full_matrices))
        return original(matrix, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", recording_svd)
    matrix = torch.randn(200, 7, dtype=DTYPE)
    stable_nullspace(matrix, config=StableNullspaceConfig(bootstrap=0))

    assert calls == [False]


def test_stable_nullspace_retains_structural_nullity_for_wide_matrix():
    matrix = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    result = stable_nullspace(matrix, config=StableNullspaceConfig(bootstrap=0))

    assert result.nullity == 2
    torch.testing.assert_close(
        result.projector,
        torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=DTYPE)),
    )


def test_noise_calibrated_near_null_tier_is_distinct_from_exact_nullspace():
    matrix = torch.full((80, 1), 1.0e-6, dtype=DTYPE)

    exact = stable_nullspace(matrix, config=StableNullspaceConfig())
    calibrated = stable_nullspace(
        matrix,
        config=StableNullspaceConfig(near_null_max_vectors=1),
    )

    assert exact.nullity == 0
    assert exact.exact_nullity == 0
    assert exact.tier == "none"
    assert calibrated.nullity == 1
    assert calibrated.exact_nullity == 0
    assert calibrated.tier == "noise_calibrated"

    no_gap = stable_nullspace(
        torch.eye(4, dtype=DTYPE),
        config=StableNullspaceConfig(
            near_null_max_vectors=2,
            near_null_min_spectral_gap=10.0,
        ),
    )
    assert no_gap.nullity == 0
    assert no_gap.tier == "none"
