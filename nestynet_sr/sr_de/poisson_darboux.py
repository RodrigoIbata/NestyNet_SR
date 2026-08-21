# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Interpretable Darboux-map candidates for regular Poisson patches.

For an explicit local diffeomorphism ``g: z -> y`` with Jacobian ``A=Dg`` and
a constant canonical/degenerate tensor ``J0`` in ``y`` coordinates, the tensor
in the original coordinates is

``Pi_z = A^{-1} J0 A^{-T}``.

Jacobi is inherited from ``J0``.  This module deliberately searches only a
finite slate of caller-supplied, low-complexity maps.  It does not train an
unconstrained invertible neural network.  Affine maps are first-class; callable
and triangular adapters allow symbolic/AST expressions to be connected by
providing analytic component gradients.

All certificates are local to the sampled constant-rank patch.  Singular and
rank-changing Poisson structures require multiple patches or another lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np


Array = np.ndarray


@runtime_checkable
class DifferentiableMap(Protocol):
    """Minimal protocol required by the Darboux pullback lane."""

    name: str
    complexity: float

    def value(self, state_points: Array) -> Array:
        """Evaluate ``g(z)`` with shape ``(N, d)``."""

    def jacobian(self, state_points: Array) -> Array:
        """Evaluate ``Dg(z)`` with shape ``(N, d, d)``."""


@dataclass(frozen=True)
class AffineDarbouxMap:
    """Explicit affine chart ``g(z)=matrix @ z + offset``."""

    matrix: Array
    offset: Array
    name: str = "affine"
    complexity: float = 1.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if offset.shape != (matrix.shape[0],):
            raise ValueError("offset dimension must match matrix")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
            raise ValueError("affine map contains non-finite values")
        if np.linalg.matrix_rank(matrix) != matrix.shape[0]:
            raise ValueError("affine Darboux map must be invertible")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    def value(self, state_points: Array) -> Array:
        z = _state_array(state_points)
        if z.shape[1] != self.matrix.shape[1]:
            raise ValueError("state dimension does not match affine map")
        return z @ self.matrix.T + self.offset

    def jacobian(self, state_points: Array) -> Array:
        z = _state_array(state_points)
        if z.shape[1] != self.matrix.shape[1]:
            raise ValueError("state dimension does not match affine map")
        return np.broadcast_to(self.matrix, (z.shape[0],) + self.matrix.shape).copy()


@dataclass(frozen=True)
class CallableDarbouxMap:
    """Adapter for an explicit map and analytic Jacobian callbacks."""

    value_function: Callable[[Array], Array]
    jacobian_function: Callable[[Array], Array]
    name: str = "callable_map"
    complexity: float = 1.0

    def value(self, state_points: Array) -> Array:
        z = _state_array(state_points)
        value = np.asarray(self.value_function(z), dtype=np.float64)
        if value.shape != z.shape or not np.all(np.isfinite(value)):
            raise ValueError("value_function must return finite values with shape (N, d)")
        return value

    def jacobian(self, state_points: Array) -> Array:
        z = _state_array(state_points)
        jacobian = np.asarray(self.jacobian_function(z), dtype=np.float64)
        expected = (z.shape[0], z.shape[1], z.shape[1])
        if jacobian.shape != expected or not np.all(np.isfinite(jacobian)):
            raise ValueError(
                f"jacobian_function must return finite values with shape {expected}"
            )
        return jacobian


@dataclass(frozen=True)
class ScalarMapComponent:
    """One scalar component for :class:`TriangularDarbouxMap`.

    ``gradient_function`` returns the full ambient gradient.  This accommodates
    an AST expression evaluator without importing a particular AST class.
    """

    value_function: Callable[[Array], Array]
    gradient_function: Callable[[Array], Array]
    name: str
    max_input_index: int
    complexity: float = 1.0


def darboux_map_with_casimirs(
    regular_components: Sequence[ScalarMapComponent],
    casimir_result: object,
    *,
    name: str = "casimir_completed_map",
) -> CallableDarbouxMap:
    """Append certified Casimirs to caller-supplied ``(q,p)`` chart components.

    The regular components remain an explicit caller responsibility; this
    helper supplies the silent ``c`` coordinates in canonical ordering
    ``(q,p,c)`` and preserves analytic gradients from the scalar library.
    """

    from nestynet_sr.sr_de.poisson_invariants import evaluate_scalar_library
    import torch

    regular = tuple(regular_components)
    candidates = tuple(getattr(casimir_result, "candidates", ()))
    expected_corank = int(getattr(casimir_result, "expected_corank", -1))
    if not bool(getattr(casimir_result, "complete", False)):
        raise ValueError("Casimir discovery must be complete before Darboux completion")
    if len(candidates) != expected_corank:
        raise ValueError("selected Casimir count does not match the expected corank")
    if len(regular) % 2:
        raise ValueError("regular Darboux components must contain paired (q,p) coordinates")
    terms = tuple(getattr(casimir_result, "terms", ()))

    def value_function(state_points: Array) -> Array:
        state = _state_array(state_points)
        if len(regular) + len(candidates) != state.shape[1]:
            raise ValueError("regular plus Casimir coordinates must equal state dimension")
        regular_values = [
            np.asarray(component.value_function(state), dtype=np.float64)
            for component in regular
        ]
        points = torch.as_tensor(state, dtype=torch.float64)
        library = evaluate_scalar_library(points, terms)
        casimir_values = [
            (library.values @ candidate.coeffs.detach().cpu().to(points))
            .detach()
            .cpu()
            .numpy()
            for candidate in candidates
        ]
        return np.column_stack((*regular_values, *casimir_values))

    def jacobian_function(state_points: Array) -> Array:
        state = _state_array(state_points)
        regular_gradients = [
            np.asarray(component.gradient_function(state), dtype=np.float64)
            for component in regular
        ]
        points = torch.as_tensor(state, dtype=torch.float64)
        library = evaluate_scalar_library(points, terms)
        casimir_gradients = [
            torch.einsum(
                "k,nkd->nd",
                candidate.coeffs.detach().cpu().to(points),
                library.gradients,
            )
            .detach()
            .cpu()
            .numpy()
            for candidate in candidates
        ]
        return np.stack((*regular_gradients, *casimir_gradients), axis=1)

    complexity = float(
        sum(component.complexity for component in regular)
        + sum(candidate.complexity for candidate in candidates)
    )
    return CallableDarbouxMap(
        value_function=value_function,
        jacobian_function=jacobian_function,
        name=name,
        complexity=complexity,
    )


@dataclass(frozen=True)
class TriangularDarbouxMap:
    """Adapter for symbolic/AST triangular maps.

    Component ``i`` may depend only on coordinates ``0..max_input_index`` and
    must declare ``max_input_index <= i``.  The analytic gradients are checked
    against that structure at evaluation time, catching accidental dense ASTs.
    """

    components: tuple[ScalarMapComponent, ...]
    name: str = "triangular_map"
    complexity: float | None = None
    triangular_tol: float = 1.0e-12

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("triangular map needs at least one component")
        for i, component in enumerate(self.components):
            if component.max_input_index < 0 or component.max_input_index > i:
                raise ValueError(
                    f"component {i} must declare 0 <= max_input_index <= {i}"
                )
        if self.complexity is None:
            object.__setattr__(
                self, "complexity", float(sum(c.complexity for c in self.components))
            )

    def value(self, state_points: Array) -> Array:
        z = _state_array(state_points)
        if z.shape[1] != len(self.components):
            raise ValueError("component count must equal state dimension")
        columns = []
        for component in self.components:
            values = np.asarray(component.value_function(z), dtype=np.float64)
            if values.shape != (z.shape[0],) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"component {component.name} must return finite values with shape (N,)"
                )
            columns.append(values)
        return np.column_stack(columns)

    def jacobian(self, state_points: Array) -> Array:
        z = _state_array(state_points)
        d = len(self.components)
        if z.shape[1] != d:
            raise ValueError("component count must equal state dimension")
        rows = []
        for i, component in enumerate(self.components):
            gradient = np.asarray(component.gradient_function(z), dtype=np.float64)
            if gradient.shape != z.shape or not np.all(np.isfinite(gradient)):
                raise ValueError(
                    f"gradient for {component.name} must be finite with shape (N, d)"
                )
            forbidden = gradient[:, component.max_input_index + 1 :]
            if forbidden.size and np.max(np.abs(forbidden)) > float(self.triangular_tol):
                raise ValueError(
                    f"component {component.name} violates its declared triangular support"
                )
            rows.append(gradient)
        return np.stack(rows, axis=1)


@dataclass(frozen=True)
class DarbouxCertificate:
    """Sampled chart-geometry certificate, not a data-agreement claim."""

    map_name: str
    poisson_tensor: Array
    pushforward_relative_residual: float
    map_jacobian_relative_residual: float
    skew_relative_residual: float
    sampled_jacobi_relative_residual: float
    jacobi_by_construction: bool
    jacobi_statement: str
    expected_rank: int
    sampled_ranks: tuple[int, ...]
    rank_stable: bool
    minimum_map_singular_value: float
    maximum_map_condition_number: float
    local_diffeomorphism: bool
    chart_geometry_accepted: bool

    @property
    def accepted(self) -> bool:
        """Compatibility alias for chart geometry, not model/data acceptance."""

        return bool(self.chart_geometry_accepted)


@dataclass(frozen=True)
class DarbouxCandidateReport:
    """Ranked explicit-map candidate."""

    map_name: str
    complexity: float
    certificate: DarbouxCertificate
    tensor_relative_residual: float | None
    flow_relative_residual: float | None
    score: float


def _state_array(state_points: Array) -> Array:
    z = np.asarray(state_points, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] == 0:
        raise ValueError(f"state_points must have shape (N, d), got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("state_points contains non-finite values")
    return z


def _rms(value: Array) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def canonical_degenerate_poisson(dimension: int, n_pairs: int) -> Array:
    """Return ``diag(J_{2r}, 0)`` in coordinate order ``(q, p, c)``.

    The first ``n_pairs`` coordinates are ``q``, the next ``n_pairs`` are
    ``p``, and the remaining coordinates are Casimir coordinates ``c``.
    """

    d = int(dimension)
    r = int(n_pairs)
    if d <= 0 or r < 0 or 2 * r > d:
        raise ValueError("require dimension > 0 and 0 <= 2*n_pairs <= dimension")
    j0 = np.zeros((d, d), dtype=np.float64)
    if r:
        j0[:r, r : 2 * r] = np.eye(r)
        j0[r : 2 * r, :r] = -np.eye(r)
    return j0


def _map_jacobians(chart: DifferentiableMap, state_points: Array) -> Array:
    z = _state_array(state_points)
    jacobians = np.asarray(chart.jacobian(z), dtype=np.float64)
    expected = (z.shape[0], z.shape[1], z.shape[1])
    if jacobians.shape != expected:
        raise ValueError(f"chart.jacobian must have shape {expected}, got {jacobians.shape}")
    if not np.all(np.isfinite(jacobians)):
        raise ValueError("chart.jacobian returned non-finite values")
    return jacobians


def _map_values(chart: DifferentiableMap, state_points: Array) -> Array:
    z = _state_array(state_points)
    values = np.asarray(chart.value(z), dtype=np.float64)
    if values.shape != z.shape or not np.all(np.isfinite(values)):
        raise ValueError("chart.value must return finite values with shape (N, d)")
    return values


def _finite_difference_map_jacobian(
    state_points: Array,
    chart: DifferentiableMap,
    *,
    relative_step: float,
) -> Array:
    z = _state_array(state_points)
    n, d = z.shape
    derivative = np.empty((n, d, d), dtype=np.float64)
    for k in range(d):
        step = float(relative_step) * np.maximum(1.0, np.abs(z[:, k]))
        plus = z.copy()
        minus = z.copy()
        plus[:, k] += step
        minus[:, k] -= step
        value_plus = _map_values(chart, plus)
        value_minus = _map_values(chart, minus)
        derivative[..., k] = (value_plus - value_minus) / (2.0 * step[:, None])
    return derivative


def pullback_poisson_tensor(
    state_points: Array,
    chart: DifferentiableMap,
    canonical_tensor: Array,
) -> Array:
    """Evaluate ``Dg^{-1} J0 Dg^{-T}`` for an explicit local chart."""

    z = _state_array(state_points)
    j0 = np.asarray(canonical_tensor, dtype=np.float64)
    d = z.shape[1]
    if j0.shape != (d, d) or not np.all(np.isfinite(j0)):
        raise ValueError(f"canonical_tensor must be finite with shape {(d, d)}")
    if not np.allclose(j0, -j0.T, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("canonical_tensor must be skew-symmetric")
    jacobians = _map_jacobians(chart, z)
    tensors = np.empty((z.shape[0], d, d), dtype=np.float64)
    for n, jacobian in enumerate(jacobians):
        try:
            left = np.linalg.solve(jacobian, j0)
            tensors[n] = np.linalg.solve(jacobian, left.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"chart Jacobian is singular at sample {n}") from exc
    # Remove roundoff-level symmetric leakage without changing the construction.
    return 0.5 * (tensors - tensors.transpose(0, 2, 1))


def _finite_difference_tensor_derivatives(
    state_points: Array,
    chart: DifferentiableMap,
    canonical_tensor: Array,
    *,
    relative_step: float,
) -> Array:
    z = _state_array(state_points)
    n, d = z.shape
    derivative = np.empty((n, d, d, d), dtype=np.float64)
    for k in range(d):
        step = float(relative_step) * np.maximum(1.0, np.abs(z[:, k]))
        plus = z.copy()
        minus = z.copy()
        plus[:, k] += step
        minus[:, k] -= step
        pi_plus = pullback_poisson_tensor(plus, chart, canonical_tensor)
        pi_minus = pullback_poisson_tensor(minus, chart, canonical_tensor)
        derivative[..., k] = (pi_plus - pi_minus) / (2.0 * step[:, None, None])
    return derivative


def poisson_jacobiator(poisson_values: Array, poisson_derivatives: Array) -> Array:
    """Return independent Jacobiator components ``(i,j,k)``, ``i<j<k``."""

    pi = np.asarray(poisson_values, dtype=np.float64)
    dpi = np.asarray(poisson_derivatives, dtype=np.float64)
    if pi.ndim != 3 or pi.shape[1] != pi.shape[2]:
        raise ValueError("poisson_values must have shape (N, d, d)")
    n, d, _ = pi.shape
    if dpi.shape != (n, d, d, d):
        raise ValueError(f"poisson_derivatives must have shape {(n, d, d, d)}")
    triples = [(i, j, k) for i in range(d) for j in range(i + 1, d) for k in range(j + 1, d)]
    if not triples:
        return np.zeros((n, 0), dtype=np.float64)
    result = np.empty((n, len(triples)), dtype=np.float64)
    for column, (i, j, k) in enumerate(triples):
        result[:, column] = (
            np.einsum("nl,nl->n", pi[:, i, :], dpi[:, j, k, :])
            + np.einsum("nl,nl->n", pi[:, j, :], dpi[:, k, i, :])
            + np.einsum("nl,nl->n", pi[:, k, :], dpi[:, i, j, :])
        )
    return result


def certify_darboux_map(
    state_points: Array,
    chart: DifferentiableMap,
    canonical_tensor: Array,
    *,
    pushforward_tol: float = 1.0e-9,
    map_jacobian_tol: float = 1.0e-6,
    skew_tol: float = 1.0e-10,
    jacobi_tol: float = 1.0e-6,
    map_singular_value_tol: float = 1.0e-10,
    rank_rtol: float = 1.0e-9,
    rank_atol: float = 1.0e-12,
    finite_difference_step: float = 1.0e-5,
) -> DarbouxCertificate:
    """Certify an explicit Darboux pullback on sampled regular points."""

    z = _state_array(state_points)
    j0 = np.asarray(canonical_tensor, dtype=np.float64)
    pi = pullback_poisson_tensor(z, chart, j0)
    _map_values(chart, z)
    jacobians = _map_jacobians(chart, z)
    numerical_jacobians = _finite_difference_map_jacobian(
        z,
        chart,
        relative_step=float(finite_difference_step),
    )
    map_jacobian_residual = _rms(jacobians - numerical_jacobians) / max(
        _rms(jacobians), np.finfo(np.float64).tiny
    )
    pushed = np.einsum("nij,njk,nlk->nil", jacobians, pi, jacobians)
    push_residual = _rms(pushed - j0[None, :, :]) / max(
        _rms(j0), np.finfo(np.float64).tiny
    )
    skew_residual = _rms(pi + pi.transpose(0, 2, 1)) / max(
        _rms(pi), np.finfo(np.float64).tiny
    )

    d_pi = _finite_difference_tensor_derivatives(
        z,
        chart,
        j0,
        relative_step=float(finite_difference_step),
    )
    jacobiator = poisson_jacobiator(pi, d_pi)
    if jacobiator.shape[1] == 0 or _rms(d_pi) == 0.0:
        jacobi_relative = 0.0
    else:
        jacobi_relative = _rms(jacobiator) / max(
            _rms(pi) * _rms(d_pi), np.finfo(np.float64).tiny
        )

    map_singular_values = np.linalg.svd(jacobians, compute_uv=False)
    minimum_map_sv = float(np.min(map_singular_values[:, -1]))
    maximum_condition = float(
        np.max(map_singular_values[:, 0] / map_singular_values[:, -1])
    )
    local_diffeomorphism = bool(minimum_map_sv > float(map_singular_value_tol))

    j0_singular_values = np.linalg.svd(j0, compute_uv=False)
    expected_rank = int(
        np.sum(j0_singular_values > max(float(rank_atol), rank_rtol * j0_singular_values[0]))
    )
    sampled_ranks: list[int] = []
    for tensor in pi:
        singular_values = np.linalg.svd(tensor, compute_uv=False)
        tolerance = max(float(rank_atol), float(rank_rtol) * float(singular_values[0]))
        sampled_ranks.append(int(np.sum(singular_values > tolerance)))
    rank_stable = bool(sampled_ranks and all(rank == expected_rank for rank in sampled_ranks))
    accepted = bool(
        local_diffeomorphism
        and rank_stable
        and push_residual <= float(pushforward_tol)
        and map_jacobian_residual <= float(map_jacobian_tol)
        and skew_residual <= float(skew_tol)
        and jacobi_relative <= float(jacobi_tol)
    )
    return DarbouxCertificate(
        map_name=str(chart.name),
        poisson_tensor=pi,
        pushforward_relative_residual=float(push_residual),
        map_jacobian_relative_residual=float(map_jacobian_residual),
        skew_relative_residual=float(skew_residual),
        sampled_jacobi_relative_residual=float(jacobi_relative),
        jacobi_by_construction=True,
        jacobi_statement=(
            "Pi=Dg^{-1}J0Dg^{-T} inherits Jacobi from constant J0 on a C2 "
            "local-diffeomorphism patch; the reported finite-difference residual "
            "is an additional sampled implementation check."
        ),
        expected_rank=expected_rank,
        sampled_ranks=tuple(sampled_ranks),
        rank_stable=rank_stable,
        minimum_map_singular_value=minimum_map_sv,
        maximum_map_condition_number=maximum_condition,
        local_diffeomorphism=local_diffeomorphism,
        chart_geometry_accepted=accepted,
    )


def _relative_residual(target: Array, prediction: Array) -> float:
    return float(
        np.linalg.norm(target - prediction)
        / max(np.linalg.norm(target), np.finfo(np.float64).tiny)
    )


def rank_darboux_candidates(
    state_points: Array,
    charts: Sequence[DifferentiableMap],
    canonical_tensor: Array,
    *,
    target_poisson: Array | None = None,
    target_vector_field: Array | None = None,
    hamiltonian_gradients: Array | None = None,
    complexity_weight: float = 1.0e-3,
    tensor_weight: float = 1.0,
    flow_weight: float = 1.0,
    invalid_penalty: float = 1.0e6,
) -> list[DarbouxCandidateReport]:
    """Certify and rank a finite slate of explicit Darboux charts.

    At least one data-fit target should normally be supplied.  With no target,
    the function ranks valid charts only by complexity and numerical
    certification; many gauge-equivalent charts will then be indistinguishable.
    """

    z = _state_array(state_points)
    if not charts:
        return []
    d = z.shape[1]
    target_pi: Array | None = None
    if target_poisson is not None:
        target_pi = np.asarray(target_poisson, dtype=np.float64)
        if target_pi.shape == (d, d):
            target_pi = np.broadcast_to(target_pi, (z.shape[0], d, d))
        if target_pi.shape != (z.shape[0], d, d) or not np.all(np.isfinite(target_pi)):
            raise ValueError("target_poisson must have shape (d,d) or (N,d,d)")
    target_f: Array | None = None
    grad_h: Array | None = None
    if (target_vector_field is None) != (hamiltonian_gradients is None):
        raise ValueError(
            "target_vector_field and hamiltonian_gradients must be supplied together"
        )
    if target_vector_field is not None:
        target_f = np.asarray(target_vector_field, dtype=np.float64)
        grad_h = np.asarray(hamiltonian_gradients, dtype=np.float64)
        if target_f.shape != (z.shape[0], d) or grad_h.shape != target_f.shape:
            raise ValueError("flow target and Hamiltonian gradients must have shape (N,d)")

    reports: list[DarbouxCandidateReport] = []
    for chart in charts:
        certificate = certify_darboux_map(z, chart, canonical_tensor)
        pi = certificate.poisson_tensor
        tensor_residual = None if target_pi is None else _relative_residual(target_pi, pi)
        flow_residual = None
        if target_f is not None and grad_h is not None:
            prediction = np.einsum("nij,nj->ni", pi, grad_h)
            flow_residual = _relative_residual(target_f, prediction)
        score = float(complexity_weight) * float(chart.complexity)
        if tensor_residual is not None:
            score += float(tensor_weight) * tensor_residual
        if flow_residual is not None:
            score += float(flow_weight) * flow_residual
        if not certificate.accepted:
            score += float(invalid_penalty)
        reports.append(
            DarbouxCandidateReport(
                map_name=str(chart.name),
                complexity=float(chart.complexity),
                certificate=certificate,
                tensor_relative_residual=tensor_residual,
                flow_relative_residual=flow_residual,
                score=float(score),
            )
        )
    reports.sort(key=lambda report: (report.score, report.complexity, report.map_name))
    return reports


__all__ = [
    "AffineDarbouxMap",
    "CallableDarbouxMap",
    "DarbouxCandidateReport",
    "DarbouxCertificate",
    "DifferentiableMap",
    "ScalarMapComponent",
    "TriangularDarbouxMap",
    "canonical_degenerate_poisson",
    "certify_darboux_map",
    "darboux_map_with_casimirs",
    "poisson_jacobiator",
    "pullback_poisson_tensor",
    "rank_darboux_candidates",
]
