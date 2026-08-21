import torch

from nestynet_sr.sr_core.bridges import U, Var
from nestynet_sr.sr_de.poisson_auto import (
    AutoPoissonConfig,
    _bounded_search_config,
    auto_discover_poisson_from_system_result,
    auto_discover_poisson_structure,
    vector_field_from_system_result,
)
from nestynet_sr.sr_de.poisson_core import StableNullspaceConfig
from nestynet_sr.sr_de.poisson_invariants import HamiltonianFitConfig
from nestynet_sr.sr_de.poisson_search import PoissonSearchConfig
from nestynet_sr.sr_de.system_de_search import SystemDESearchResult


DTYPE = torch.float64


def _oscillator_result(*, nonautonomous=False, perturbation=0.0, residual=0.0):
    terms = [U(out_idx=0), U(out_idx=1)]
    if nonautonomous:
        terms.append(Var(0))
    coeffs = torch.tensor(
        [[0.0, -1.0], [1.0, 0.0]], dtype=DTYPE
    )
    coeffs[0, 0] = -float(perturbation)
    if nonautonomous:
        coeffs = torch.cat((coeffs, torch.zeros((2, 1), dtype=DTYPE)), dim=1)
    return SystemDESearchResult(
        order=1,
        x_axis=0,
        out_idxs=(0, 1),
        term_asts=terms,
        coeffs=coeffs,
        rms_train=[float(residual), float(residual)],
    )


def test_recovered_autonomous_system_automatically_enters_constant_poisson_lane():
    result = _oscillator_result()
    field = vector_field_from_system_result(result)
    generator = torch.Generator().manual_seed(712)
    points = torch.randn(240, 2, generator=generator, dtype=DTYPE)
    torch.testing.assert_close(
        field.value(points),
        torch.stack((points[:, 1], -points[:, 0]), dim=1),
    )

    report = auto_discover_poisson_from_system_result(result, points)

    assert report.status == "accepted"
    assert report.selected_lane == "constant"
    assert report.search is not None
    assert report.search.best is not None
    assert report.search.best.rank.generic_rank == 2
    assert report.search.best.polynomial_jacobi.passed
    assert max(report.search.best.hamiltonian_validation_relative) < 1.0e-10


def test_automatic_poisson_branch_fails_closed_for_thin_or_disabled_support():
    field = lambda z: torch.stack((z[:, 1], -z[:, 0]), dim=1)
    collapsed = torch.ones((160, 2), dtype=DTYPE)
    skipped = auto_discover_poisson_structure(field, collapsed)
    assert skipped.status == "skipped"
    assert skipped.reason == "insufficient_state_space_coverage"
    assert skipped.search is None

    points = torch.randn(
        160, 2, generator=torch.Generator().manual_seed(713), dtype=DTYPE
    )
    disabled = auto_discover_poisson_structure(
        field, points, AutoPoissonConfig(enabled=False)
    )
    assert disabled.status == "skipped"
    assert disabled.reason == "disabled"


def test_nonautonomous_recovered_system_is_skipped_without_affecting_baseline():
    result = _oscillator_result(nonautonomous=True)
    points = torch.randn(
        180, 2, generator=torch.Generator().manual_seed(714), dtype=DTYPE
    )

    report = auto_discover_poisson_from_system_result(result, points)

    assert report.status == "skipped"
    assert report.reason.startswith("nonautonomous_or_unsupported_system")
    assert report.search is None


def test_manual_search_config_cannot_widen_automatic_lane_or_runtime_budgets():
    points = torch.randn(
        160, 2, generator=torch.Generator().manual_seed(715), dtype=DTYPE
    )
    field = lambda z: torch.stack((z[:, 1], -z[:, 0]), dim=1)
    outside_lane = auto_discover_poisson_structure(
        field,
        points,
        AutoPoissonConfig(lanes=("constant",), max_tensor_coefficients=1),
        search_config=PoissonSearchConfig(
            lanes=("quadratic",),
            max_representatives=1,
        ),
    )
    assert outside_lane.status == "skipped"
    assert outside_lane.reason == "search_config_lane_outside_auto_policy"
    assert outside_lane.search is None

    bounded = auto_discover_poisson_structure(
        field,
        points,
        AutoPoissonConfig(
            lanes=("constant",),
            max_representatives=3,
            sparse_rotation_steps=2,
            bootstrap=1,
        ),
        search_config=PoissonSearchConfig(
            lanes=("constant",),
            max_representatives=99,
            sparse_rotation_steps=99,
            nullspace=StableNullspaceConfig(bootstrap=9),
        ),
    )
    assert bounded.status == "accepted"
    assert bounded.search is not None
    lane = bounded.search.lanes[0]
    assert len(lane.candidates) <= 3
    assert len(lane.nullspace.bootstrap_principal_angles) <= 1


def test_nested_search_override_can_only_tighten_trusted_automatic_policy():
    auto = AutoPoissonConfig(
        lanes=("constant", "affine"),
        bootstrap=2,
        max_hamiltonian_stlsq_iterations=17,
    )
    requested = PoissonSearchConfig(
        lanes=("affine", "constant", "affine"),
        validation_fraction=0.9,
        stop_at_first_accepted_lane=False,
        normalize_dataset_blocks=False,
        nullspace=StableNullspaceConfig(
            rank_rtol=1.0,
            rank_atol=1.0,
            bootstrap=0,
        ),
        invariance_relative_tolerance=1.0,
        jacobi_relative_tolerance=1.0,
        minimum_rank_stable_fraction=0.0,
        require_nonzero_rank=False,
        require_hamiltonian=False,
        discover_casimirs=False,
        require_complete_casimirs=False,
        hamiltonian=HamiltonianFitConfig(
            stlsq_max_iter=10_000_000,
            relative_residual_tolerance=1.0,
            absolute_residual_tolerance=1.0,
        ),
    )

    bounded, reason = _bounded_search_config(auto, requested)

    assert reason is None
    assert bounded is not None
    assert bounded.lanes == ("constant", "affine")
    assert bounded.validation_fraction == 0.5
    assert bounded.stop_at_first_accepted_lane is True
    assert bounded.normalize_dataset_blocks is True
    assert bounded.nullspace.bootstrap == 2
    assert bounded.nullspace.rank_rtol <= 1.0e-9
    assert bounded.nullspace.rank_atol <= 1.0e-11
    assert bounded.hamiltonian.stlsq_max_iter == 17
    assert bounded.invariance_relative_tolerance <= 1.0e-8
    assert bounded.jacobi_relative_tolerance <= 1.0e-8
    assert bounded.minimum_rank_stable_fraction >= 0.9
    assert bounded.require_nonzero_rank is True
    assert bounded.require_hamiltonian is True
    assert bounded.discover_casimirs is True
    assert bounded.require_complete_casimirs is True
    assert bounded.hamiltonian.relative_residual_tolerance <= 1.0e-7


def test_recovered_system_uses_separate_noise_calibrated_near_null_tier():
    epsilon = 1.0e-5
    result = _oscillator_result(perturbation=epsilon, residual=epsilon)
    points = torch.randn(
        320, 2, generator=torch.Generator().manual_seed(716), dtype=DTYPE
    )

    exact_only = auto_discover_poisson_from_system_result(
        result,
        points,
        AutoPoissonConfig(lanes=("constant",), noise_calibrated=False),
    )
    calibrated = auto_discover_poisson_from_system_result(
        result,
        points,
        AutoPoissonConfig(lanes=("constant",), noise_calibrated=True),
    )

    assert exact_only.status == "rejected"
    assert calibrated.status == "accepted"
    assert calibrated.promotion_tier == "noise_calibrated"
    assert calibrated.search is not None
    assert calibrated.search.best is not None
    assert calibrated.search.best.nullspace.exact_nullity == 0
    assert calibrated.noise_calibration is not None
    assert calibrated.noise_calibration["source"] == "system_de_residual"
    assert calibrated.noise_calibration["accepted"] is True
    assert calibrated.noise_calibration["relative_tolerance"] <= 1.0e-3


def test_recovered_system_abstains_when_noise_exceeds_calibration_ceiling():
    result = _oscillator_result(perturbation=0.0, residual=0.1)
    points = torch.randn(
        320, 2, generator=torch.Generator().manual_seed(717), dtype=DTYPE
    )

    report = auto_discover_poisson_from_system_result(
        result,
        points,
        AutoPoissonConfig(
            lanes=("constant",),
            noise_calibrated=True,
            max_noise_relative_tolerance=1.0e-3,
        ),
    )

    assert report.status == "skipped"
    assert report.reason == "system_residual_exceeds_noise_calibration_limit"
    assert report.promotion_tier is None
    assert report.noise_calibration is not None
    assert report.noise_calibration["accepted"] is False
    assert (
        report.noise_calibration["reason"]
        == "measured_noise_exceeds_automatic_limit"
    )

    explicit = auto_discover_poisson_from_system_result(
        result,
        points,
        AutoPoissonConfig(
            lanes=("constant",),
            noise_calibrated=True,
            noise_calibrated_relative_tolerance=1.0e-6,
            max_noise_relative_tolerance=1.0e-3,
        ),
    )
    assert explicit.status == "skipped"
    assert explicit.reason == "system_residual_exceeds_noise_calibration_limit"
