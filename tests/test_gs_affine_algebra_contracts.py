import numpy as np


def _translation_fixture(n=128):
    rng = np.random.default_rng(101)
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    z = X[:, 0] - X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z, -grad_z], axis=1)
    return X, y, grad


def _generic_no_symmetry_fixture(n=160):
    rng = np.random.default_rng(102)
    X = rng.uniform(-1.2, 1.2, size=(n, 2))
    y = (
        np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2)
        + 0.17 * X[:, 0] * X[:, 1]
        + 0.11 * X[:, 0] ** 3
    )
    dy_dx0 = (
        np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2)
        + 0.17 * X[:, 1]
        + 0.33 * X[:, 0] ** 2
    )
    dy_dx1 = (
        0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2)
        + 0.17 * X[:, 0]
    )
    grad = np.stack([dy_dx0, dy_dx1], axis=1)
    return X, y, grad


def test_global_affine_algebra_solves_output_action_jointly():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra

    rng = np.random.default_rng(103)
    X = rng.uniform(-1.0, 1.0, size=(192, 2))
    beta = 0.7
    H = np.sin(X[:, 1]) + 1.3
    y = np.exp(beta * X[:, 0]) * H
    grad = np.stack([beta * y, np.exp(beta * X[:, 0]) * np.cos(X[:, 1])], axis=1)

    algebra = discover_affine_algebra(X, y, grad, heldout_fraction=0.25, bootstrap=4)

    assert algebra.nullity == 1
    assert algebra.promotable
    gen = algebra.basis_generators[0]
    b0 = gen.b_physical[0]
    assert abs(b0) > 1.0e-8
    assert abs(gen.A_physical[0][0]) < 1.0e-8
    assert abs(gen.A_physical[0][1]) < 1.0e-8
    assert abs(gen.A_physical[1][0]) < 1.0e-8
    assert abs(gen.A_physical[1][1]) < 1.0e-8
    assert abs(gen.b_physical[1]) < 1.0e-8
    np.testing.assert_allclose(gen.beta_physical / b0, beta, rtol=1.0e-8, atol=1.0e-8)


def test_global_affine_algebra_recovers_translation_subspace_distribution_and_quotient():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra

    X, y, grad = _translation_fixture()
    algebra = discover_affine_algebra(X, y, grad, heldout_fraction=0.25, bootstrap=8)

    assert algebra.nullity == 3
    assert algebra.nullspace_basis.shape == (algebra.unknown_count, 3)
    assert algebra.nullspace_projector.shape == (algebra.unknown_count, algebra.unknown_count)
    assert algebra.distribution_rank == 1
    assert algebra.train_residual_rel < 1.0e-8
    assert algebra.heldout_residual_rel < 1.0e-8
    assert max(algebra.bootstrap_principal_angles) < 1.0e-4
    assert algebra.bracket_closure_residual < 1.0e-8
    assert algebra.promotable

    assert algebra.linear_invariant_covectors.shape == (1, 2)
    z = X @ algebra.linear_invariant_covectors[0]
    target = X[:, 0] - X[:, 1]
    corr = abs(float(np.corrcoef(z, target)[0, 1]))
    assert corr > 1.0 - 1.0e-10


def test_global_affine_algebra_reports_too_few_rows_separately_from_nullity():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra

    X, y, grad = _translation_fixture(n=4)
    algebra = discover_affine_algebra(X, y, grad)

    assert algebra.unknown_count == 8
    assert algebra.independent_row_count < algebra.unknown_count
    assert algebra.structurally_underdetermined
    assert not algebra.promotable


def test_global_affine_algebra_rejects_generic_no_symmetry_fixture_for_promotion():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra

    X, y, grad = _generic_no_symmetry_fixture()
    algebra = discover_affine_algebra(X, y, grad, heldout_fraction=0.25, bootstrap=8)

    assert algebra.discovered_nullity == 0 or not algebra.promotable
    assert algebra.heldout_residual_rel > algebra.acceptance_residual_tol
