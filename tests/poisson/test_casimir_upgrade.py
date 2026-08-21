# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from dataclasses import replace

import numpy as np
import pytest
import torch

from nestynet_sr.sr_de.poisson_invariants import (
    HamiltonianFitConfig,
    discover_casimirs,
    fit_hamiltonian_given_poisson,
    fit_hamiltonians_given_poisson,
    score_casimir_carrier,
)
from nestynet_sr.sr_de.poisson_core import StableNullspaceConfig
from nestynet_sr.sr_de.poisson_search import (
    PoissonSearchConfig,
    discover_poisson_structure,
)


DTYPE = torch.float64


def _so3_poisson(z: torch.Tensor) -> torch.Tensor:
    out = z.new_zeros((z.shape[0], 3, 3))
    out[:, 0, 1] = -z[:, 2]
    out[:, 0, 2] = z[:, 1]
    out[:, 1, 2] = -z[:, 0]
    return out - out.transpose(1, 2)


def _canonical_poisson(z: torch.Tensor) -> torch.Tensor:
    out = z.new_zeros((z.shape[0], 4, 4))
    out[:, 0, 2] = 1.0
    out[:, 1, 3] = 1.0
    return out - out.transpose(1, 2)


def _heavy_top_poisson(z: torch.Tensor) -> torch.Tensor:
    """Lie--Poisson tensor on se(3)* in coordinates (M, Gamma)."""

    m = z[:, :3]
    gamma = z[:, 3:]

    def cross_matrix(v: torch.Tensor) -> torch.Tensor:
        out = v.new_zeros((v.shape[0], 3, 3))
        out[:, 0, 1] = -v[:, 2]
        out[:, 0, 2] = v[:, 1]
        out[:, 1, 2] = -v[:, 0]
        return out - out.transpose(1, 2)

    cm = cross_matrix(m)
    cg = cross_matrix(gamma)
    out = z.new_zeros((z.shape[0], 6, 6))
    out[:, :3, :3] = cm
    out[:, :3, 3:] = cg
    out[:, 3:, :3] = cg
    return out


def _monomials(state_dim: int, degree: int) -> tuple[tuple[int, ...], ...]:
    from nestynet_sr.sr_de.poisson_basis import PolynomialScalarBasis

    return PolynomialScalarBasis(
        state_dim, max_degree=degree, include_constant=True
    ).exponents


def test_euler_casimir_is_complete_rendered_and_reusable_as_gs_coordinate():
    points = torch.randn(
        320, 3, generator=torch.Generator().manual_seed(401), dtype=DTYPE
    )
    result = discover_casimirs(
        _so3_poisson,
        points,
        _monomials(3, 2),
        validation_fraction=0.25,
        max_representatives=48,
    )

    assert result.expected_corank == 1
    assert result.discovered_corank == 1
    assert result.complete
    assert result.status == "complete"
    assert result.gradient_projector.shape == (points.shape[0], 3, 3)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.selected
    assert candidate.gradient_rank_gain == 1
    assert candidate.expression
    assert candidate.ast is not None
    assert candidate.coordinate is not None
    values = candidate.coordinate.evaluate(points.numpy())
    assert np.var(values) > 1.0e-4
    assert candidate.coordinate.kind == "casimir"


def test_heavy_top_recovers_two_functionally_independent_casimirs():
    points = torch.randn(
        700, 6, generator=torch.Generator().manual_seed(402), dtype=DTYPE
    )
    result = discover_casimirs(
        _heavy_top_poisson,
        points,
        _monomials(6, 2),
        validation_fraction=0.2,
        max_representatives=96,
        sparse_rotation_steps=20,
    )

    assert result.expected_corank == 2
    assert result.discovered_corank == 2
    assert result.complete
    assert len(result.candidates) == 2
    assert all(row.gradient_rank_gain == 1 for row in result.candidates)
    assert all(row.poisson_residual_relative < 1.0e-9 for row in result.candidates)


def test_casimir_foliation_is_invariant_to_nullspace_basis_rotation(monkeypatch):
    import nestynet_sr.sr_de.poisson_invariants as invariants

    points = torch.randn(
        520, 6, generator=torch.Generator().manual_seed(417), dtype=DTYPE
    )
    terms = _monomials(6, 2)
    baseline = discover_casimirs(
        _heavy_top_poisson,
        points,
        terms,
        max_representatives=96,
    )
    original = invariants.stable_nullspace

    def rotated_nullspace(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.nullity != 2:
            return result
        angle = torch.tensor(0.731, dtype=result.basis.dtype, device=result.basis.device)
        rotation = torch.stack(
            (
                torch.stack((torch.cos(angle), -torch.sin(angle))),
                torch.stack((torch.sin(angle), torch.cos(angle))),
            )
        )
        return replace(result, basis=result.basis @ rotation)

    monkeypatch.setattr(invariants, "stable_nullspace", rotated_nullspace)
    rotated = discover_casimirs(
        _heavy_top_poisson,
        points,
        terms,
        max_representatives=96,
    )

    assert baseline.complete and rotated.complete
    torch.testing.assert_close(
        rotated.gradient_projector,
        baseline.gradient_projector,
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_full_rank_canonical_tensor_reports_no_nonconstant_casimir():
    points = torch.randn(
        240, 4, generator=torch.Generator().manual_seed(403), dtype=DTYPE
    )
    result = discover_casimirs(
        _canonical_poisson,
        points,
        _monomials(4, 2),
    )

    assert result.expected_corank == 0
    assert result.discovered_corank == 0
    assert result.complete
    assert result.status == "full_rank_no_nonconstant_casimir"
    assert result.candidates == ()
    assert result.coordinates == ()


def test_deficient_tensor_reports_when_library_cannot_express_the_casimir():
    points = torch.randn(
        220, 3, generator=torch.Generator().manual_seed(409), dtype=DTYPE
    )
    result = discover_casimirs(
        _so3_poisson,
        points,
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    assert result.expected_corank == 1
    assert result.discovered_corank == 0
    assert not result.complete
    assert result.status == "incomplete_library"


def test_incomplete_required_casimir_blocks_poisson_candidate_promotion():
    inverse_inertia = torch.tensor([1.0, 0.7, 0.2], dtype=DTYPE)

    def euler_field(z: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cross(z, z * inverse_inertia)

    points = torch.randn(
        260, 3, generator=torch.Generator().manual_seed(410), dtype=DTYPE
    )
    config = PoissonSearchConfig(
        lanes=("linear",),
        stop_at_first_accepted_lane=False,
        nullspace=StableNullspaceConfig(rank_rtol=1.0e-10, rank_atol=1.0e-12),
        hamiltonian=HamiltonianFitConfig(
            solver="least_squares", ridge=0.0, stlsq_lambda=1.0e-10
        ),
        require_complete_casimirs=True,
    )
    result = discover_poisson_structure(
        euler_field,
        points,
        config,
        hamiltonian_terms=_monomials(3, 2),
        casimir_terms=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    assert not result.accepted
    assert result.candidates
    assert all("casimir_incomplete" in row.failure_reasons for row in result.candidates)


def test_heldout_residual_blocks_training_only_casimir():
    train = torch.randn(
        180, 3, generator=torch.Generator().manual_seed(404), dtype=DTYPE
    )
    train[:, 0] = -train[:, 0].abs() - 0.1
    heldout = torch.randn(
        90, 3, generator=torch.Generator().manual_seed(405), dtype=DTYPE
    )
    heldout[:, 0] = heldout[:, 0].abs() + 0.1

    def patchwise_pi(z: torch.Tensor) -> torch.Tensor:
        tensor = _so3_poisson(z)
        alternate = z.new_zeros((z.shape[0], 3, 3))
        alternate[:, 0, 1] = 1.0
        alternate[:, 1, 0] = -1.0
        mask = z[:, 0] > 0.0
        tensor[mask] = alternate[mask]
        return tensor

    result = discover_casimirs(
        patchwise_pi,
        train,
        ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        validation_points=heldout,
        relative_residual_tolerance=1.0e-9,
        absolute_residual_tolerance=1.0e-12,
    )

    assert not result.complete
    assert result.status == "unstable_nullspace"
    assert result.candidates == ()
    assert result.rejected_candidates
    assert any(
        "heldout_poisson_residual" in row.failure_reasons
        for row in result.rejected_candidates
    )


def test_heldout_gradient_collapse_blocks_false_completeness():
    train = torch.randn(
        180, 2, generator=torch.Generator().manual_seed(412), dtype=DTYPE
    )
    train[:, 1] = -train[:, 1].abs() - 0.2
    heldout = torch.randn(
        90, 2, generator=torch.Generator().manual_seed(413), dtype=DTYPE
    )
    heldout[:, 1] = heldout[:, 1].abs() + 0.2
    zero_poisson = torch.zeros((2, 2), dtype=DTYPE)

    def first(z: torch.Tensor) -> torch.Tensor:
        return z[:, 0]

    def support_dependent(z: torch.Tensor) -> torch.Tensor:
        return torch.where(z[:, 1] < 0.0, z[:, 1], z[:, 0])

    result = discover_casimirs(
        zero_poisson,
        train,
        (first, support_dependent),
        validation_points=heldout,
        minimum_independence_fraction=0.9,
    )

    assert result.expected_corank == 2
    assert result.discovered_corank == 1
    assert not result.complete
    assert result.status == "incomplete_library"
    assert len(result.candidates) == 1
    dependent = next(
        row
        for row in result.rejected_candidates
        if "functionally_dependent" in row.failure_reasons
    )
    assert dependent.independence_fraction == 1.0
    assert dependent.heldout_independence_fraction == 0.0


def test_explicit_sampled_poisson_accepts_separate_heldout_tensor():
    train = torch.randn(
        160, 3, generator=torch.Generator().manual_seed(414), dtype=DTYPE
    )
    heldout = torch.randn(
        80, 3, generator=torch.Generator().manual_seed(415), dtype=DTYPE
    )
    train_tensor = _so3_poisson(train)
    heldout_tensor = _so3_poisson(heldout)
    terms = _monomials(3, 2)

    with pytest.raises(ValueError, match="validation_poisson is required"):
        discover_casimirs(
            train_tensor,
            train,
            terms,
            validation_points=heldout,
        )

    result = discover_casimirs(
        train_tensor,
        train,
        terms,
        validation_points=heldout,
        validation_poisson=heldout_tensor,
    )
    assert result.complete
    assert result.expected_corank == 1


def test_degenerate_hamiltonian_fit_reports_and_fixes_casimir_gauge():
    points = torch.randn(
        260, 3, generator=torch.Generator().manual_seed(406), dtype=DTYPE
    )
    terms = ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2))
    physical = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=DTYPE)
    gradients = torch.stack(
        (
            torch.zeros_like(points),
            torch.stack((2.0 * points[:, 0], torch.zeros_like(points[:, 0]), torch.zeros_like(points[:, 0])), dim=1),
            torch.stack((torch.zeros_like(points[:, 0]), 2.0 * points[:, 1], torch.zeros_like(points[:, 0])), dim=1),
            torch.stack((torch.zeros_like(points[:, 0]), torch.zeros_like(points[:, 0]), 2.0 * points[:, 2]), dim=1),
        ),
        dim=1,
    )
    grad_h = torch.einsum("k,nkd->nd", physical, gradients)
    field = torch.einsum("nij,nj->ni", _so3_poisson(points), grad_h)

    fit = fit_hamiltonian_given_poisson(
        _so3_poisson,
        field,
        points,
        terms,
        HamiltonianFitConfig(
            solver="least_squares",
            ridge=0.0,
            stlsq_lambda=1.0e-10,
            gauge_mode="sparsest",
        ),
    )

    assert fit.accepted
    assert fit.gauge is not None
    assert fit.gauge.nullity == 2  # additive constant and quadratic Casimir
    assert fit.gauge.representative_mode == "sparsest"
    assert fit.gauge.flow_equivalence_relative < 1.0e-11
    assert fit.complexity == 2
    torch.testing.assert_close(
        fit.prediction.reshape_as(field), field, atol=1.0e-10, rtol=1.0e-10
    )


def test_multi_hamiltonian_heads_apply_the_same_explicit_gauge_policy():
    points = torch.randn(
        230, 3, generator=torch.Generator().manual_seed(411), dtype=DTYPE
    )
    terms = ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2))
    gradients = torch.stack(
        (
            torch.zeros_like(points),
            torch.stack((2.0 * points[:, 0], torch.zeros_like(points[:, 0]), torch.zeros_like(points[:, 0])), dim=1),
            torch.stack((torch.zeros_like(points[:, 0]), 2.0 * points[:, 1], torch.zeros_like(points[:, 0])), dim=1),
            torch.stack((torch.zeros_like(points[:, 0]), torch.zeros_like(points[:, 0]), 2.0 * points[:, 2]), dim=1),
        ),
        dim=1,
    )
    coefficients = (
        torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=DTYPE),
        torch.tensor([0.0, 2.0, 1.0, 4.0], dtype=DTYPE),
    )
    fields = [
        torch.einsum(
            "nij,nj->ni",
            _so3_poisson(points),
            torch.einsum("k,nkd->nd", row, gradients),
        )
        for row in coefficients
    ]
    result = fit_hamiltonians_given_poisson(
        _so3_poisson,
        fields,
        (points, points),
        terms,
        mode="independent",
        config=HamiltonianFitConfig(
            solver="least_squares", ridge=0.0, gauge_mode="sparsest"
        ),
    )

    assert result.accepted
    assert all(fit.gauge is not None and fit.gauge.nullity == 2 for fit in result.fits)
    assert all(fit.complexity == 2 for fit in result.fits)
    assert all(fit.gauge.flow_equivalence_relative < 1.0e-11 for fit in result.fits)

    shared = fit_hamiltonians_given_poisson(
        _so3_poisson,
        fields,
        (points, points),
        terms,
        mode="shared_support",
        config=HamiltonianFitConfig(
            solver="least_squares", ridge=0.0, gauge_mode="sparsest"
        ),
    )
    assert shared.accepted
    assert shared.support.sum() == 2
    assert all(torch.equal(fit.support, shared.support) for fit in shared.fits)
    assert all(fit.complexity == 2 for fit in shared.fits)
    assert all(
        fit.gauge is not None
        and fit.gauge.representative_mode == "sparsest_shared_support"
        and fit.gauge.flow_equivalence_relative < 1.0e-11
        for fit in shared.fits
    )


def test_casimir_carrier_objective_rejects_constants_and_scores_true_carrier():
    from nestynet_sr.sr_core.bridges import AddNode, PowNode, Var

    points = torch.randn(
        180, 3, generator=torch.Generator().manual_seed(407), dtype=DTYPE
    )
    carrier = AddNode(
        AddNode(PowNode(Var(0), 2.0), PowNode(Var(1), 2.0)),
        PowNode(Var(2), 2.0),
    )
    accepted = score_casimir_carrier(_so3_poisson, points, carrier)
    collapsed = score_casimir_carrier(
        _so3_poisson, points, (0, 0, 0)
    )
    heldout = torch.randn(
        80, 3, generator=torch.Generator().manual_seed(416), dtype=DTYPE
    )
    alternate = torch.zeros((3, 3), dtype=DTYPE)
    alternate[0, 1] = 1.0
    alternate[1, 0] = -1.0
    heldout_failure = score_casimir_carrier(
        _so3_poisson,
        points,
        carrier,
        validation_points=heldout,
        validation_poisson=alternate,
    )

    assert accepted.accepted
    assert accepted.poisson_residual_relative < 1.0e-12
    assert not collapsed.accepted
    assert "collapsed_gradient" in collapsed.failure_reasons
    assert "collapsed_variation" in collapsed.failure_reasons
    assert not heldout_failure.accepted
    assert "heldout_poisson_residual" in heldout_failure.failure_reasons


def test_certified_casimirs_complete_a_darboux_chart_candidate():
    from nestynet_sr.sr_de.poisson_darboux import (
        ScalarMapComponent,
        darboux_map_with_casimirs,
    )

    points = torch.randn(
        220, 3, generator=torch.Generator().manual_seed(408), dtype=DTYPE
    )
    points[:, 2] = points[:, 2].abs() + 0.4
    result = discover_casimirs(
        _so3_poisson,
        points,
        _monomials(3, 2),
    )
    regular = (
        ScalarMapComponent(
            value_function=lambda z: z[:, 0],
            gradient_function=lambda z: np.broadcast_to(
                np.array([1.0, 0.0, 0.0]), z.shape
            ),
            name="q",
            max_input_index=0,
        ),
        ScalarMapComponent(
            value_function=lambda z: z[:, 1],
            gradient_function=lambda z: np.broadcast_to(
                np.array([0.0, 1.0, 0.0]), z.shape
            ),
            name="p",
            max_input_index=1,
        ),
    )
    chart = darboux_map_with_casimirs(regular, result)
    values = chart.value(points.numpy())
    jacobian = chart.jacobian(points.numpy())

    assert values.shape == points.shape
    assert jacobian.shape == (points.shape[0], 3, 3)
    np.testing.assert_allclose(values[:, 2], result.coordinates[0].evaluate(points.numpy()))
    assert np.min(np.linalg.svd(jacobian, compute_uv=False)[:, -1]) > 1.0e-6
