import numpy as np
import pytest


def _scaling_fixture(n=128):
    rng = np.random.default_rng(201)
    X = rng.uniform(0.4, 2.0, size=(n, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z / X[:, 1], -grad_z * X[:, 0] / X[:, 1] ** 2], axis=1)
    return X, y, grad


def _rotation_fixture(n=160):
    rng = np.random.default_rng(202)
    X = rng.uniform(-1.4, 1.4, size=(n, 2))
    X = X[np.linalg.norm(X, axis=1) > 0.15]
    z = X[:, 0] ** 2 + X[:, 1] ** 2
    y = np.sin(z) + 0.17 * z
    grad_z = np.cos(z) + 0.17
    grad = np.stack([2.0 * X[:, 0] * grad_z, 2.0 * X[:, 1] * grad_z], axis=1)
    return X, y, grad


def _translation_fixture(n=128):
    rng = np.random.default_rng(204)
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    z = X[:, 0] - X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z, -grad_z], axis=1)
    return X, y, grad


def test_translation_reduction_plan_uses_distribution_annihilator_not_basis_vector():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
    from nestynet_sr.sr_gs.quotient import compile_reduction_plan

    X, y, grad = _translation_fixture()
    algebra = discover_affine_algebra(X, y, grad)
    reduction = compile_reduction_plan(algebra)

    assert algebra.nullity == 3
    assert reduction.status == "compiled"
    assert reduction.reason == "linear_distribution_annihilator"
    assert reduction.generic_orbit_rank == 1
    assert reduction.invariant_coordinates
    assert reduction.orbit_coordinates
    z = reduction.invariant_coordinates[0].evaluate(X)
    target = X[:, 0] - X[:, 1]
    corr = abs(float(np.corrcoef(z, target)[0, 1]))
    assert corr > 1.0 - 1.0e-10


def test_scaling_reduction_plan_contains_ratio_chart_domain_and_provenance():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
    from nestynet_sr.sr_gs.quotient import compile_reduction_plan

    X, y, grad = _scaling_fixture()
    algebra = discover_affine_algebra(X, y, grad)
    reduction = compile_reduction_plan(algebra)

    assert reduction.status == "compiled"
    assert reduction.reason == "common_diagonal_scaling"
    assert reduction.generic_orbit_rank == 1
    assert reduction.invariant_coordinates
    assert reduction.orbit_coordinates
    assert reduction.invariant_coordinates[0].domain.excludes("x1 == 0")
    assert reduction.invariant_coordinates[0].ast is not None
    assert reduction.invariant_coordinates[0].coordinate_map is not None
    assert reduction.invariant_coordinates[0].provenance["source"] == "data"
    z = reduction.invariant_coordinates[0].evaluate(X)
    target = np.log(X[:, 0] / X[:, 1])
    corr = abs(float(np.corrcoef(z, target)[0, 1]))
    assert corr > 1.0 - 1.0e-10


def test_rotation_reduction_plan_reports_radial_chart_and_origin_singularity():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
    from nestynet_sr.sr_gs.quotient import compile_reduction_plan

    X, y, grad = _rotation_fixture()
    algebra = discover_affine_algebra(X, y, grad)
    reduction = compile_reduction_plan(algebra)

    assert reduction.status == "compiled"
    assert reduction.reason == "rotation_quadratic_radius"
    assert reduction.generic_orbit_rank == 1
    assert reduction.orbit_coordinates
    assert any("origin" in str(s).lower() for s in reduction.singular_strata)
    z = reduction.invariant_coordinates[0].evaluate(X)
    target = X[:, 0] ** 2 + X[:, 1] ** 2
    corr = abs(float(np.corrcoef(z, target)[0, 1]))
    assert corr > 1.0 - 1.0e-10


def test_output_equivariant_reduction_contains_orbit_coordinate_and_normal_form():
    from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
    from nestynet_sr.sr_gs.quotient import compile_reduction_plan

    rng = np.random.default_rng(203)
    X = rng.uniform(-1.0, 1.0, size=(160, 2))
    s = X[:, 0]
    z = X[:, 1]
    beta = 0.7
    H = np.sin(z) + 1.3
    y = np.exp(beta * s) * H
    grad = np.stack([beta * y, np.exp(beta * s) * np.cos(z)], axis=1)

    algebra = discover_affine_algebra(X, y, grad)
    reduction = compile_reduction_plan(algebra)

    assert reduction.output_action.beta == pytest.approx(beta, rel=1.0e-6)
    assert reduction.orbit_coordinates[0].satisfies_unit_speed(algebra)
    assert reduction.normal_form.kind == "multiplicative_prefactor"
    assert reduction.normal_form.reduced_target_name == "H"
    reduced = reduction.normal_form.reduce_target(y, reduction.orbit_coordinates[0].evaluate(X))
    np.testing.assert_allclose(reduced, H, rtol=1.0e-8, atol=1.0e-8)
    reconstructed = reduction.normal_form.reconstruct_target(reduced, reduction.orbit_coordinates[0].evaluate(X))
    np.testing.assert_allclose(reconstructed, y, rtol=1.0e-8, atol=1.0e-8)


def test_output_equivariant_normal_form_is_reported_in_stagea_shadow_diagnostics():
    from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
    from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals

    rng = np.random.default_rng(205)
    X = rng.uniform(-1.0, 1.0, size=(144, 2))
    s = X[:, 0]
    z = X[:, 1]
    beta = 0.7
    H = np.sin(z) + 1.3
    y = np.exp(beta * s) * H
    grad = np.stack([beta * y, np.exp(beta * s) * np.cos(z)], axis=1)
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="audit",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
    )

    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=cfg,
    )

    assert proposals == []
    shadow_rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(shadow_rows) == 1
    reduction_report = shadow_rows[0]["reduction"]
    assert reduction_report["normal_form"]["kind"] == "multiplicative_prefactor"
    assert reduction_report["output_action"]["beta"] == pytest.approx(beta, rel=1.0e-6)
    assert shadow_rows[0]["shadow_only"]
    assert not shadow_rows[0]["used_for_proposal"]
