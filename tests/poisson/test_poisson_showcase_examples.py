import pytest

from examples.poisson_geometry.casimir_taxonomy import (
    run_showcase as run_casimir_taxonomy,
)
from examples.poisson_geometry.cyclic_lotka_volterra import (
    run_showcase as run_lotka_volterra,
)


def test_casimir_taxonomy_distinguishes_physical_and_algebra_casimirs():
    report = run_casimir_taxonomy(sample_count=320, seed=53)

    assert report.poisson_casimirs.complete
    assert report.poisson_casimirs.expected_corank == 1
    assert report.poisson_casimirs.discovered_corank == 1
    assert report.poisson_casimirs.candidates[0].coordinate.kind == "casimir"
    assert report.poisson_casimir_alignment == pytest.approx(1.0, abs=1.0e-9)

    assert report.hamiltonian.accepted
    assert report.hamiltonian.gauge is not None
    assert report.hamiltonian.gauge.nullity >= 2
    assert report.hamiltonian.gauge.representative_mode == "sparsest"
    assert report.hamiltonian.gauge.flow_equivalence_relative < 1.0e-12

    assert report.canonical_casimirs.complete
    assert report.canonical_casimirs.expected_corank == 0
    assert report.canonical_casimirs.status == "full_rank_no_nonconstant_casimir"
    assert report.canonical_casimirs.candidates == ()

    assert report.algebra_casimir.accepted
    assert report.algebra_casimir.expected_corank == 1
    assert report.charge_brackets.accepted
    assert not report.charge_brackets.global_equivariance_proven
    assert report.algebra_pullback_relative_rms < 1.0e-12
from examples.poisson_geometry.translated_euler_top import (
    run_showcase as run_translated_euler,
)


def test_cyclic_lotka_volterra_recovers_quadratic_bracket_and_cubic_casimir():
    report = run_lotka_volterra(sample_count=320, seed=43)

    assert report.search.accepted
    assert report.automatic.status == "accepted"
    assert report.best.lane == "quadratic"
    assert report.best.nullspace.nullity == 1
    assert report.tensor_alignment == pytest.approx(1.0, abs=1.0e-8)
    assert min(report.hamiltonian_alignments) == pytest.approx(1.0, abs=1.0e-8)
    assert report.best.rank.generic_rank == 2
    assert report.best.polynomial_jacobi.passed
    assert report.best.polynomial_jacobi.max_abs < 1.0e-12
    assert max(report.best.hamiltonian_validation_relative) < 1.0e-10
    assert report.cubic_casimir_alignment == pytest.approx(1.0, abs=1.0e-8)
    assert report.cubic_casimir.poisson_residual_rms < 1.0e-10
    assert report.cubic_casimir.flow_residual_rms is not None
    assert report.cubic_casimir.flow_residual_rms < 1.0e-10


def test_translated_euler_requires_affine_lane_and_exposes_rank_drop():
    report = run_translated_euler(sample_count=320, seed=47)

    assert report.search.accepted
    assert report.automatic.status == "accepted"
    assert tuple(lane.lane for lane in report.search.lanes) == (
        "constant",
        "linear",
        "affine",
    )
    assert report.lower_lane_nullities == (("constant", 0), ("linear", 0))
    assert report.best.lane == "affine"
    assert report.best.nullspace.nullity == 1
    assert report.tensor_alignment == pytest.approx(1.0, abs=1.0e-8)
    assert report.best.rank.generic_rank == 2
    assert report.singular_point_tensor_norm < 1.0e-10
    assert report.best.polynomial_jacobi.passed
    assert report.best.polynomial_jacobi.max_abs < 1.0e-12
    assert max(report.best.hamiltonian_validation_relative) < 1.0e-10
    # H is identifiable only modulo the translated quadratic Casimir; the
    # gauge-aware representative removes one redundant diagonal square.
    assert report.quadratic_hamiltonian_terms >= 2
    assert all(fit.gauge is not None and fit.gauge.nullity >= 2 for fit in report.best.hamiltonian.fits)
    assert report.casimir_alignment == pytest.approx(1.0, abs=1.0e-8)
    assert report.translated_casimir.poisson_residual_rms < 1.0e-10
    automatic_payload = report.automatic.to_report()
    automatic_casimirs = automatic_payload["casimirs"]
    assert automatic_casimirs is not None
    assert automatic_casimirs["complete"]
    assert automatic_casimirs["expected_corank"] == 1
    assert len(automatic_casimirs["expressions"]) == 1
    assert automatic_payload["hamiltonian"] is not None
    assert all(
        fit["gauge"]["nullity"] >= 2
        for fit in automatic_payload["hamiltonian"]["fits"]
    )
