import json

import numpy as np

from nestynet_sr.sr_gs.algebra_certificates import (
    affine_graph_bracket_coeffs,
    certify_affine_algebra,
    classify_affine_generator,
)
from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra


def _orthonormal_projector(raw_basis):
    q, _r = np.linalg.qr(np.asarray(raw_basis, dtype=float))
    return q, q @ q.T


def _translation_fixture(n=128):
    rng = np.random.default_rng(501)
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    z = X[:, 0] - X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z, -grad_z], axis=1)
    return X, y, grad


def test_discovered_translation_algebra_has_serializable_quotient_ready_certificate():
    X, y, grad = _translation_fixture()
    algebra = discover_affine_algebra(X, y, grad, heldout_fraction=0.25, bootstrap=4)
    cert = algebra.certificate

    assert cert is not None
    assert cert.is_closed
    assert cert.quotient_ready
    assert cert.quotient_policy == "quotient_ready"
    assert cert.distribution_rank == 1
    assert cert.orbit_dimension == 1
    assert cert.quotient_codimension == 1
    assert cert.heldout_verified
    assert cert.subspace_stable
    assert cert.dimensionally_consistent
    assert cert.bracket_closure_residual < 1.0e-8

    payload = algebra.to_report()
    json.dumps(payload)
    assert payload["certificate"]["quotient_policy"] == "quotient_ready"


def test_full_graph_bracket_includes_output_action():
    # n=1 coefficient order: [A00, b0, alpha, beta].
    # V1 = d_x + y d_y, V2 = d_y. Their bracket is -d_y in this convention.
    v1 = np.asarray([0.0, 1.0, 0.0, 1.0])
    v2 = np.asarray([0.0, 0.0, 1.0, 0.0])
    bracket = affine_graph_bracket_coeffs(v1, v2, input_dim=1)

    np.testing.assert_allclose(bracket, np.asarray([0.0, 0.0, -1.0, 0.0]))
    assert classify_affine_generator(v1, 1) == "translation+output_action"
    assert classify_affine_generator(v2, 1) == "output_translation"

    basis, projector = _orthonormal_projector(np.column_stack([v1, v2]))
    cert = certify_affine_algebra(
        basis=basis,
        projector=projector,
        input_dim=1,
        singular_values=(2.0, 1.0, 1.0e-12, 0.0),
        independent_row_count=2,
        unknown_count=4,
        train_residual_rel=0.0,
        heldout_residual_rel=0.0,
        acceptance_residual_tol=1.0e-8,
        structurally_underdetermined=False,
        distribution_rank=1,
        pointwise_distribution_ranks=(1, 1, 1),
    )

    assert cert.is_closed
    assert cert.quotient_ready
    assert any(abs(record.output_alpha) > 0.0 for record in cert.bracket_records)


def test_nonclosed_affine_span_is_retained_for_audit_but_rejected_for_quotients():
    # n=2 coefficient order: [A00,A01,A10,A11,b0,b1,alpha,beta].
    # Span{d_x, x d_y} is not closed because [d_x, x d_y] = d_y.
    dx = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    x_dy = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dy = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    bracket = affine_graph_bracket_coeffs(dx, x_dy, input_dim=2)
    np.testing.assert_allclose(bracket, dy)

    basis, projector = _orthonormal_projector(np.column_stack([dx, x_dy]))
    cert = certify_affine_algebra(
        basis=basis,
        projector=projector,
        input_dim=2,
        singular_values=(3.0, 2.0, 1.0, 1.0e-12, 0.0, 0.0, 0.0, 0.0),
        independent_row_count=2,
        unknown_count=8,
        train_residual_rel=0.0,
        heldout_residual_rel=0.0,
        acceptance_residual_tol=1.0e-8,
        structurally_underdetermined=False,
        distribution_rank=2,
        pointwise_distribution_ranks=(2, 2, 2),
    )

    assert not cert.is_closed
    assert not cert.quotient_ready
    assert cert.quotient_policy == "reject_for_quotient_nonclosed"
    assert cert.bracket_closure_residual > 0.9
    assert cert.bracket_records[0].outside_span_norm > 0.9


def test_certificate_reports_dimensional_warnings_as_audit_only():
    dx = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    basis, projector = _orthonormal_projector(dx.reshape(-1, 1))
    cert = certify_affine_algebra(
        basis=basis,
        projector=projector,
        input_dim=2,
        singular_values=(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        independent_row_count=2,
        unknown_count=8,
        train_residual_rel=0.0,
        heldout_residual_rel=0.0,
        acceptance_residual_tol=1.0e-8,
        structurally_underdetermined=False,
        distribution_rank=1,
        pointwise_distribution_ranks=(1, 1),
        dimensional_warnings=("A[0,1] mixes seconds into meters",),
    )

    assert cert.is_closed
    assert not cert.dimensionally_consistent
    assert not cert.quotient_ready
    assert cert.quotient_policy == "audit_only_dimensionally_questionable"
