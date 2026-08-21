import numpy as np
import pytest

from nestynet_sr.sr_de.poisson_3d import (
    ScalarIntegralSamples,
    cross_product_matrices,
    reconstruct_nambu_3d,
)
from nestynet_sr.sr_de.poisson_darboux import (
    AffineDarbouxMap,
    ScalarMapComponent,
    TriangularDarbouxMap,
    canonical_degenerate_poisson,
    certify_darboux_map,
    poisson_jacobiator,
    pullback_poisson_tensor,
    rank_darboux_candidates,
)
from nestynet_sr.sr_de.poisson_noether import (
    canonical_affine_momentum_map,
    canonical_symplectic_matrix,
    classify_noether_symmetry,
)


def test_exact_3d_euler_nambu_reconstruction_masks_singular_origin():
    rng = np.random.default_rng(145)
    inertia = np.array([1.7, 2.3, 4.1])
    state = rng.normal(size=(256, 3))
    state[0] = 0.0

    grad_c = state.copy()
    grad_h = state / inertia
    casimir = ScalarIntegralSamples(
        values=0.5 * np.sum(state**2, axis=1),
        gradients=grad_c,
        name="C=|M|^2/2",
    )
    hamiltonian = ScalarIntegralSamples(
        values=0.5 * np.sum(state**2 / inertia, axis=1),
        gradients=grad_h,
        name="H",
    )
    true_pi = cross_product_matrices(state)
    field = np.einsum("nij,nj->ni", true_pi, grad_h)

    report = reconstruct_nambu_3d(
        field,
        casimir,
        hamiltonian,
        multiplier_features=np.ones((state.shape[0], 1)),
        multiplier_feature_names=["1"],
        multiplier_features_are_differentiable=True,
    )

    assert report.accepted
    assert report.jacobi_by_construction
    assert report.generic_rank == 2
    assert report.singular_mask[0]
    assert np.count_nonzero(report.singular_mask) == 1
    assert report.multiplier.mode == "linear_features"
    np.testing.assert_allclose(report.multiplier.coefficients, [1.0], atol=2e-14)
    np.testing.assert_allclose(
        report.poisson_tensor[report.regular_mask],
        true_pi[report.regular_mask],
        atol=2e-14,
    )
    assert report.reconstruction_relative_residual_regular < 2e-14
    assert max(report.first_integral_relative_residuals) < 2e-14


def test_rank_two_nambu_lane_rejects_zero_structure():
    rng = np.random.default_rng(146)
    state = rng.normal(size=(128, 3))
    casimir = ScalarIntegralSamples(
        values=0.5 * np.sum(state**2, axis=1),
        gradients=state,
        name="C",
    )
    grad_h = np.column_stack((state[:, 1], state[:, 0], np.ones(state.shape[0])))
    hamiltonian = ScalarIntegralSamples(
        values=state[:, 0] * state[:, 1] + state[:, 2],
        gradients=grad_h,
        name="H",
    )

    report = reconstruct_nambu_3d(
        np.zeros_like(state),
        casimir,
        hamiltonian,
        multiplier_features=np.ones((state.shape[0], 1)),
        multiplier_features_are_differentiable=True,
    )

    assert report.generic_rank == 0
    assert report.rank_two_fraction == 0.0
    assert report.accepted is False


def test_pointwise_nambu_multiplier_is_a_pseudotarget_not_jacobi_certificate():
    rng = np.random.default_rng(147)
    state = rng.normal(size=(128, 3))
    grad_c = state
    grad_h = np.column_stack((state[:, 1], state[:, 0], np.ones(state.shape[0])))
    carrier = np.cross(grad_c, grad_h)
    casimir = ScalarIntegralSamples(np.zeros(state.shape[0]), grad_c, name="C")
    hamiltonian = ScalarIntegralSamples(np.zeros(state.shape[0]), grad_h, name="H")

    report = reconstruct_nambu_3d(carrier, casimir, hamiltonian)

    assert report.multiplier.mode == "pointwise"
    assert report.reconstruction_relative_residual_regular < 1.0e-14
    assert report.jacobi_by_construction is False
    assert report.jacobi_certification == "pointwise_multiplier_pseudotarget_only"
    assert report.accepted is False

    sampled_features = reconstruct_nambu_3d(
        carrier,
        casimir,
        hamiltonian,
        multiplier_features=np.ones((state.shape[0], 1)),
    )
    assert sampled_features.multiplier.mode == "linear_features"
    assert sampled_features.jacobi_by_construction is False
    assert (
        sampled_features.jacobi_certification
        == "sampled_feature_multiplier_without_differentiability_contract"
    )
    assert sampled_features.accepted is False


def test_jacobiator_vanishes_for_linear_rigid_body_tensor():
    rng = np.random.default_rng(231)
    state = rng.normal(size=(64, 3))
    pi = cross_product_matrices(state)
    derivatives = np.zeros((state.shape[0], 3, 3, 3))
    basis = np.eye(3)
    for coordinate in range(3):
        derivatives[..., coordinate] = cross_product_matrices(basis[coordinate])
    jacobiator = poisson_jacobiator(pi, derivatives)
    np.testing.assert_allclose(jacobiator, 0.0, atol=1e-14)


def test_general_noether_classification_and_strict_canonical_fast_path():
    rng = np.random.default_rng(341)
    state = rng.normal(size=(200, 2))
    j = canonical_symplectic_matrix(1)
    pi = np.broadcast_to(j, (state.shape[0], 2, 2)).copy()
    d_pi = np.zeros((state.shape[0], 2, 2, 2))

    # Harmonic-oscillator phase rotation Y=J grad(H), H=(q^2+p^2)/2.
    grad_h = state.copy()
    generator = state @ j.T
    d_generator = np.broadcast_to(j, (state.shape[0], 2, 2)).copy()
    charge_gradients = grad_h[:, None, :]
    report = classify_noether_symmetry(
        generator,
        d_generator,
        pi,
        d_pi,
        grad_h,
        charge_gradients=charge_gradients,
        charge_term_names=["(q^2+p^2)/2"],
    )

    assert report.poisson_symmetry
    assert report.preserves_hamiltonian
    assert report.hamiltonian_generator
    assert report.hamiltonian_symmetry
    assert report.classification == "hamiltonian_symmetry_with_sampled_local_charge"
    assert report.charge_fit is not None
    np.testing.assert_allclose(report.charge_fit.coefficients, [1.0], atol=2e-14)
    assert report.charge_fit.local_charge_found
    assert not report.global_charge_proven
    assert not report.charge_fit.global_charge_proven

    charge = canonical_affine_momentum_map(j, np.zeros(2), state)
    np.testing.assert_allclose(charge, 0.5 * np.sum(state**2, axis=1), atol=2e-14)
    with pytest.raises(ValueError, match="not canonical symplectic"):
        canonical_affine_momentum_map(np.eye(2), np.zeros(2), state)


def test_poisson_symmetry_is_not_automatically_noether_symmetry():
    rng = np.random.default_rng(817)
    state = rng.normal(size=(128, 2))
    j = canonical_symplectic_matrix(1)
    pi = np.broadcast_to(j, (state.shape[0], 2, 2)).copy()
    d_pi = np.zeros((state.shape[0], 2, 2, 2))
    translation = np.broadcast_to(np.array([1.0, 0.0]), state.shape).copy()
    d_translation = np.zeros((state.shape[0], 2, 2))

    report = classify_noether_symmetry(
        translation,
        d_translation,
        pi,
        d_pi,
        state,
    )

    assert report.poisson_symmetry
    assert not report.preserves_hamiltonian
    assert not report.hamiltonian_symmetry
    assert report.classification == "poisson_symmetry_not_preserving_H"


def test_affine_darboux_pullback_certificate_and_candidate_ranking():
    rng = np.random.default_rng(901)
    state = rng.normal(size=(80, 3))
    j0 = canonical_degenerate_poisson(3, 1)
    matrix = np.diag([2.0, 3.0, 4.0])
    true_chart = AffineDarbouxMap(matrix, np.array([0.2, -0.1, 0.4]), name="scaled")
    identity_chart = AffineDarbouxMap(np.eye(3), np.zeros(3), name="identity")

    target = pullback_poisson_tensor(state, true_chart, j0)
    certificate = certify_darboux_map(state, true_chart, j0)
    assert certificate.accepted
    assert certificate.jacobi_by_construction
    assert certificate.rank_stable
    assert certificate.expected_rank == 2
    assert set(certificate.sampled_ranks) == {2}
    assert certificate.pushforward_relative_residual < 2e-14
    assert certificate.sampled_jacobi_relative_residual == 0.0

    ranked = rank_darboux_candidates(
        state,
        [identity_chart, true_chart],
        j0,
        target_poisson=target,
    )
    assert [candidate.map_name for candidate in ranked] == ["scaled", "identity"]
    assert ranked[0].tensor_relative_residual is not None
    assert ranked[0].tensor_relative_residual < 2e-14


def test_nonlinear_triangular_map_adapter_has_jacobi_by_construction():
    rng = np.random.default_rng(1111)
    state = rng.uniform(low=[-0.5, -1.0, -1.0], high=[0.5, 1.0, 1.0], size=(96, 3))

    def exp_value(z):
        return np.exp(z[:, 0])

    def exp_gradient(z):
        out = np.zeros_like(z)
        out[:, 0] = np.exp(z[:, 0])
        return out

    def coordinate_component(index):
        def value(z):
            return z[:, index]

        def gradient(z):
            out = np.zeros_like(z)
            out[:, index] = 1.0
            return out

        return value, gradient

    y_value, y_gradient = coordinate_component(1)
    c_value, c_gradient = coordinate_component(2)
    chart = TriangularDarbouxMap(
        components=(
            ScalarMapComponent(exp_value, exp_gradient, "exp(x)", 0, complexity=2.0),
            ScalarMapComponent(y_value, y_gradient, "y", 1),
            ScalarMapComponent(c_value, c_gradient, "c", 2),
        ),
        name="exp_triangular",
    )
    j0 = canonical_degenerate_poisson(3, 1)
    certificate = certify_darboux_map(state, chart, j0)

    assert certificate.accepted
    assert certificate.local_diffeomorphism
    assert certificate.rank_stable
    assert certificate.sampled_jacobi_relative_residual < 1e-9
    expected = np.exp(-state[:, 0])
    np.testing.assert_allclose(certificate.poisson_tensor[:, 0, 1], expected, rtol=2e-14)
