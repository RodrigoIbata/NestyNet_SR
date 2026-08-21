import numpy as np


def _translation_fixture(n=64):
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    z = X[:, 0] - X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z, -grad_z], axis=1)
    return X, y, grad


def test_translation_fixture_has_three_dimensional_affine_tangent_space():
    X, _y, grad = _translation_fixture()
    basis_fields = np.stack(
        [
            np.column_stack([np.ones(len(X)), np.ones(len(X))]),
            np.column_stack([X[:, 0], X[:, 0]]),
            np.column_stack([X[:, 1], X[:, 1]]),
        ],
        axis=0,
    )

    tangency = np.einsum("ni,gni->gn", grad, basis_fields)
    assert np.max(np.abs(tangency)) < 1.0e-12

    flattened_basis = basis_fields.reshape(3, -1)
    algebra_rank = np.linalg.matrix_rank(flattened_basis, tol=1.0e-10)
    assert algebra_rank == 3

    pointwise_ranks = [
        np.linalg.matrix_rank(basis_fields[:, i, :], tol=1.0e-10)
        for i in range(len(X))
    ]
    assert max(pointwise_ranks) == 1


def test_output_link_log_fixture_satisfies_implicit_residual_without_division():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.0, 1.0, size=(96, 2))
    a = X[:, 0] ** 2
    b = np.sin(X[:, 1])
    f = np.exp(a + b)
    f_x = f * (2.0 * X[:, 0])
    f_y = f * np.cos(X[:, 1])
    f_xy = f * (2.0 * X[:, 0]) * np.cos(X[:, 1])

    # For psi(f)=log(f), r(f)=psi''/psi'=-1/f.
    residual = f_xy + (-1.0 / f) * f_x * f_y
    assert np.max(np.abs(residual)) < 1.0e-12


def test_de_relative_invariance_fixture_distinguishes_off_shell_certificate():
    rng = np.random.default_rng(13)
    u = rng.uniform(0.3, 2.0, size=80)
    u_x = rng.uniform(-1.0, 1.0, size=80)

    # F = u_x - u. For V = u partial_u, pr V(F) = u_x - u = F.
    F = u_x - u
    prolongation_action = u_x - u
    lambda_multiplier = 1.0
    assert np.max(np.abs(prolongation_action - lambda_multiplier * F)) < 1.0e-12

    # Off shell matters: this certificate is not just checking solution samples
    # where F is already zero.
    assert np.any(np.abs(F) > 1.0e-3)
