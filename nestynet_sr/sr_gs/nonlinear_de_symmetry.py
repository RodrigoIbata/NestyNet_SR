# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Bounded polynomial point-symmetry recovery for scalar ODEs.

The generator dictionary is feature-linear even though its vector fields are
nonlinear.  For monomials ``m_j(x, u)`` the coefficient convention is exactly
pair-major::

    (xi_m0, eta_m0, xi_m1, eta_m1, ...).

The recovery funnel is deliberately subspace first:

1. solve ``A_on c = 0`` for the on-shell determining nullspace ``N``;
2. solve the reduced off-shell system
   ``[A_off N, -F Chi] [a, q] = 0``;
3. project the certified generator directions back to coefficient space;
4. rotate that subspace towards sparse representatives and independently
   refit each functional multiplier.

Only scalar ODEs of order one or two are supported.  The implementation uses
economy SVD for tall matrices, but requests a full right factor for wide
matrices so structural null directions are not discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .de_determining import _pr_action_and_residual
from .de_upgrades import PolynomialPointGenerator, polynomial_generator_basis
from .jet_bundle import JetSpaceSpec, ScalarODEJetInputs

_EPS = 1.0e-12


@dataclass(frozen=True)
class PolynomialDESymmetryConfig:
    """Configuration for bounded scalar-ODE point-symmetry recovery."""

    generator_degree: int = 2
    multiplier_degree: int = 2
    heldout_fraction: float = 0.25
    bootstrap: int = 0
    random_seed: int = 1729
    rank_rtol: float = 1.0e-9
    rank_atol: float = 1.0e-11
    on_shell_tol: float = 1.0e-8
    off_shell_tol: float = 1.0e-8
    sparse_threshold: float = 0.04
    sparse_iterations: int = 24
    sparse_zero_tol: float = 1.0e-8
    direction_dedup_tol: float = 1.0e-7
    max_candidates: int = 32
    min_samples: int = 16
    bootstrap_angle_tol: float = 0.35
    evaluate_bracket_closure: bool = True
    bracket_closure_tol: float = 1.0e-7

    def validated(self) -> "PolynomialDESymmetryConfig":
        if int(self.generator_degree) not in (1, 2):
            raise ValueError(
                "generator_degree must be 1 (affine regression) or 2 "
                "(bounded nonlinear lane); cubic and higher are deferred"
            )
        if int(self.multiplier_degree) < 0:
            raise ValueError("multiplier_degree must be nonnegative")
        if not 0.0 <= float(self.heldout_fraction) < 1.0:
            raise ValueError("heldout_fraction must lie in [0, 1)")
        if int(self.bootstrap) < 0:
            raise ValueError("bootstrap must be nonnegative")
        for name in ("rank_rtol", "rank_atol", "on_shell_tol", "off_shell_tol"):
            if float(getattr(self, name)) < 0.0 or not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 <= float(self.sparse_threshold) < 1.0:
            raise ValueError("sparse_threshold must lie in [0, 1)")
        if int(self.sparse_iterations) < 0:
            raise ValueError("sparse_iterations must be nonnegative")
        if float(self.sparse_zero_tol) < 0.0:
            raise ValueError("sparse_zero_tol must be nonnegative")
        if not 0.0 <= float(self.direction_dedup_tol) < 1.0:
            raise ValueError("direction_dedup_tol must lie in [0, 1)")
        if int(self.max_candidates) < 1:
            raise ValueError("max_candidates must be positive")
        if int(self.min_samples) < 2:
            raise ValueError("min_samples must be at least two")
        if not 0.0 <= float(self.bootstrap_angle_tol) <= 0.5 * math.pi:
            raise ValueError("bootstrap_angle_tol must lie in [0, pi/2]")
        if float(self.bracket_closure_tol) < 0.0:
            raise ValueError("bracket_closure_tol must be nonnegative")
        return self

    def to_report(self) -> dict[str, Any]:
        return {
            "generator_degree": int(self.generator_degree),
            "multiplier_degree": int(self.multiplier_degree),
            "heldout_fraction": float(self.heldout_fraction),
            "bootstrap": int(self.bootstrap),
            "random_seed": int(self.random_seed),
            "rank_rtol": float(self.rank_rtol),
            "rank_atol": float(self.rank_atol),
            "on_shell_tol": float(self.on_shell_tol),
            "off_shell_tol": float(self.off_shell_tol),
            "sparse_threshold": float(self.sparse_threshold),
            "sparse_iterations": int(self.sparse_iterations),
            "sparse_zero_tol": float(self.sparse_zero_tol),
            "direction_dedup_tol": float(self.direction_dedup_tol),
            "max_candidates": int(self.max_candidates),
            "min_samples": int(self.min_samples),
            "bootstrap_angle_tol": float(self.bootstrap_angle_tol),
            "evaluate_bracket_closure": bool(self.evaluate_bracket_closure),
            "bracket_closure_tol": float(self.bracket_closure_tol),
        }


@dataclass(frozen=True)
class PolynomialDESymmetryCandidate:
    """One sparse representative of the certified generator subspace."""

    name: str
    coefficients: tuple[float, ...]
    multiplier_coefficients: tuple[float, ...]
    generator: PolynomialPointGenerator
    on_shell_residual_rel: float
    off_shell_relative_residual_rel: float
    support_size: int
    accepted: bool
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "polynomial_point_symmetry",
            "coefficient_convention": "pair_major_xi_eta_per_monomial",
            "coefficients": [float(v) for v in self.coefficients],
            "multiplier_coefficients": [float(v) for v in self.multiplier_coefficients],
            "generator": self.generator.to_report(),
            "on_shell_residual_rel": float(self.on_shell_residual_rel),
            "off_shell_relative_residual_rel": float(self.off_shell_relative_residual_rel),
            "support_size": int(self.support_size),
            "accepted": bool(self.accepted),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class EvaluatedBracketCertificate:
    """Function-space closure check for one pair of recovered generators."""

    left: int
    right: int
    bracket_norm: float
    outside_span_norm: float
    residual_rel: float
    accepted: bool

    def to_report(self) -> dict[str, Any]:
        return {
            "left": int(self.left),
            "right": int(self.right),
            "bracket_norm": float(self.bracket_norm),
            "outside_span_norm": float(self.outside_span_norm),
            "residual_rel": float(self.residual_rel),
            "accepted": bool(self.accepted),
        }


@dataclass(frozen=True)
class PolynomialDESymmetryResult:
    """Subspace-first result for bounded polynomial point symmetries."""

    status: str
    residual: str
    jet_space: JetSpaceSpec
    config: PolynomialDESymmetryConfig
    generator_monomials: tuple[tuple[int, int], ...]
    multiplier_coordinates: tuple[str, ...]
    multiplier_monomials: tuple[tuple[int, ...], ...]
    coefficient_labels: tuple[str, ...]
    on_shell_basis: np.ndarray
    on_shell_projector: np.ndarray
    on_shell_singular_values: tuple[float, ...]
    on_shell_rank: int
    on_shell_nullity: int
    on_shell_train_residual_rel: float
    on_shell_heldout_residual_rel: float
    bootstrap_principal_angles: tuple[float, ...]
    bootstrap_stable: bool
    joint_singular_values: tuple[float, ...]
    joint_rank: int
    joint_nullity: int
    certified_basis: np.ndarray
    certified_projector: np.ndarray
    candidates: tuple[PolynomialDESymmetryCandidate, ...]
    bracket_certificates: tuple[EvaluatedBracketCertificate, ...]
    bracket_closure_residual: float
    bracket_closure_evaluated: bool
    individual_generators_accepted: bool
    closed_truncated_algebra: bool
    promotable_generators: bool
    promotable_full_algebra: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def certified_nullity(self) -> int:
        return int(self.certified_basis.shape[1])

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "residual": self.residual,
            "jet_space": self.jet_space.to_report(),
            "config": self.config.to_report(),
            "generator_monomials": [list(v) for v in self.generator_monomials],
            "multiplier_coordinates": list(self.multiplier_coordinates),
            "multiplier_monomials": [list(v) for v in self.multiplier_monomials],
            "coefficient_convention": "pair_major_xi_eta_per_monomial",
            "coefficient_labels": list(self.coefficient_labels),
            "on_shell_basis": self.on_shell_basis.tolist(),
            "on_shell_projector": self.on_shell_projector.tolist(),
            "on_shell_singular_values": [float(v) for v in self.on_shell_singular_values],
            "on_shell_rank": int(self.on_shell_rank),
            "on_shell_nullity": int(self.on_shell_nullity),
            "on_shell_train_residual_rel": float(self.on_shell_train_residual_rel),
            "on_shell_heldout_residual_rel": float(self.on_shell_heldout_residual_rel),
            "bootstrap_principal_angles": [float(v) for v in self.bootstrap_principal_angles],
            "bootstrap_stable": bool(self.bootstrap_stable),
            "joint_singular_values": [float(v) for v in self.joint_singular_values],
            "joint_rank": int(self.joint_rank),
            "joint_nullity": int(self.joint_nullity),
            "certified_nullity": int(self.certified_nullity),
            "certified_basis": self.certified_basis.tolist(),
            "certified_projector": self.certified_projector.tolist(),
            "candidates": [row.to_report() for row in self.candidates],
            "bracket_closure_residual": float(self.bracket_closure_residual),
            "bracket_closure_evaluated": bool(self.bracket_closure_evaluated),
            "bracket_certificates": [row.to_report() for row in self.bracket_certificates],
            "individual_generators_accepted": bool(self.individual_generators_accepted),
            "closed_truncated_algebra": bool(self.closed_truncated_algebra),
            "promotable_generators": bool(self.promotable_generators),
            "promotable_full_algebra": bool(self.promotable_full_algebra),
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class _NullspaceSolve:
    basis: np.ndarray
    projector: np.ndarray
    singular_values: np.ndarray
    rank: int
    rank_tol: float

    @property
    def nullity(self) -> int:
        return int(self.basis.shape[1])


def recover_polynomial_de_symmetries(
    *,
    jet_space: JetSpaceSpec,
    residual: Any,
    on_shell_samples: Mapping[str, Any],
    off_shell_samples: Mapping[str, Any],
    config: PolynomialDESymmetryConfig | None = None,
) -> PolynomialDESymmetryResult:
    """Recover and certify bounded polynomial scalar-ODE point symmetries."""

    cfg = (config or PolynomialDESymmetryConfig()).validated()
    jet_space.require_scalar_ode_phase_one()
    order = int(jet_space.max_order)
    on_inputs = jet_space.materialize_scalar_ode_inputs(on_shell_samples, order=order)
    off_inputs = jet_space.materialize_scalar_ode_inputs(off_shell_samples, order=order)
    _validate_inputs(on_inputs, min_samples=int(cfg.min_samples), label="on_shell_samples")
    _validate_inputs(off_inputs, min_samples=int(cfg.min_samples), label="off_shell_samples")

    basis_generators = tuple(polynomial_generator_basis(max_degree=int(cfg.generator_degree)))
    generator_monomials = tuple(_monomials_2d(int(cfg.generator_degree)))
    expected = 2 * len(generator_monomials)
    if len(basis_generators) != expected:
        raise RuntimeError("polynomial generator basis violates pair-major convention")
    labels = tuple(_coefficient_labels(generator_monomials))

    A_on, F_on = _determining_operator(residual, on_inputs, basis_generators)
    A_off, F_off = _determining_operator(residual, off_inputs, basis_generators)
    _require_finite_matrix(A_on, "on-shell determining matrix")
    _require_finite_matrix(A_off, "off-shell determining matrix")
    _require_finite_matrix(F_on.reshape(-1, 1), "on-shell residual")
    _require_finite_matrix(F_off.reshape(-1, 1), "off-shell residual")

    rng = np.random.default_rng(int(cfg.random_seed))
    train_idx, heldout_idx = _train_heldout_indices(
        A_on.shape[0], fraction=float(cfg.heldout_fraction), rng=rng
    )
    A_train = A_on[train_idx]
    A_heldout = A_on[heldout_idx] if heldout_idx.size else A_train
    on_solve = _solve_nullspace(
        A_train, rank_rtol=float(cfg.rank_rtol), rank_atol=float(cfg.rank_atol)
    )
    on_train_rel = _subspace_residual(A_train, on_solve.basis)
    on_heldout_rel = _subspace_residual(A_heldout, on_solve.basis)
    bootstrap_angles = _bootstrap_angles(
        A_train,
        on_solve,
        count=int(cfg.bootstrap),
        rng=rng,
        rank_rtol=float(cfg.rank_rtol),
        rank_atol=float(cfg.rank_atol),
    )

    multiplier_coordinates = ("x", "u", "u_x") + (("u_xx",) if order == 2 else ())
    multiplier_monomials = tuple(
        _total_degree_monomials(len(multiplier_coordinates), int(cfg.multiplier_degree))
    )
    Chi = _multiplier_design(off_inputs, multiplier_monomials)

    joint_solve = _empty_nullspace(on_solve.nullity + Chi.shape[1])
    certified_basis = np.zeros((expected, 0), dtype=float)
    certified_projector = np.zeros((expected, expected), dtype=float)
    if on_solve.nullity:
        reduced_action = _matrix_matrix(A_off, on_solve.basis)
        multiplier_action = F_off[:, None] * Chi
        joint = np.column_stack((reduced_action, -multiplier_action))
        joint_solve = _solve_nullspace(
            joint, rank_rtol=float(cfg.rank_rtol), rank_atol=float(cfg.rank_atol)
        )
        if joint_solve.nullity:
            reduced_coeffs = joint_solve.basis[: on_solve.nullity]
            raw_certified = _matrix_matrix(on_solve.basis, reduced_coeffs)
            certified_basis = _orthonormal_column_space(
                raw_certified,
                rtol=float(cfg.rank_rtol),
                atol=float(cfg.rank_atol),
            )
            certified_projector = _matrix_matrix(certified_basis, certified_basis.T)

    candidates = _candidate_rows(
        residual=residual,
        on_inputs=on_inputs,
        off_inputs=off_inputs,
        A_on=A_on,
        A_off=A_off,
        F_off=F_off,
        Chi=Chi,
        certified_basis=certified_basis,
        certified_projector=certified_projector,
        monomials=generator_monomials,
        multiplier_monomials=multiplier_monomials,
        cfg=cfg,
    )
    bracket_closure_evaluated = bool(cfg.evaluate_bracket_closure)
    if bracket_closure_evaluated:
        bracket_records, closure_residual = _evaluated_bracket_certificates(
            certified_basis,
            generator_monomials,
            off_inputs,
            tol=float(cfg.bracket_closure_tol),
        )
    else:
        bracket_records = []
        closure_residual = 0.0 if certified_basis.shape[1] <= 1 else math.inf
    accepted = tuple(row for row in candidates if row.accepted)
    bootstrap_stable = bool(
        not bootstrap_angles
        or max(bootstrap_angles) <= float(cfg.bootstrap_angle_tol)
    )
    individual_generators_accepted = bool(accepted)
    closed_truncated_algebra = bool(
        closure_residual <= float(cfg.bracket_closure_tol)
    )
    promotable_generators = bool(
        individual_generators_accepted and bootstrap_stable
    )
    promotable_full_algebra = bool(
        promotable_generators and closed_truncated_algebra
    )
    if promotable_full_algebra:
        status = "recovered"
    elif promotable_generators:
        status = "recovered_generators_nonclosed"
    else:
        status = "rejected"
    if not on_solve.nullity:
        reason = "no_on_shell_nullspace"
    elif not certified_basis.shape[1]:
        reason = "no_functionally_relative_off_shell_subspace"
    elif not accepted:
        reason = "no_sparse_representative_passed_certificates"
    elif not bootstrap_stable:
        reason = "bootstrap_unstable_generator_subspace"
    elif not closed_truncated_algebra:
        reason = "individual_symmetries_passed_but_truncated_algebra_nonclosed"
    else:
        reason = "accepted_polynomial_point_symmetries"

    return PolynomialDESymmetryResult(
        status=status,
        residual=_residual_label(residual),
        jet_space=jet_space,
        config=cfg,
        generator_monomials=generator_monomials,
        multiplier_coordinates=multiplier_coordinates,
        multiplier_monomials=multiplier_monomials,
        coefficient_labels=labels,
        on_shell_basis=on_solve.basis,
        on_shell_projector=on_solve.projector,
        on_shell_singular_values=tuple(float(v) for v in on_solve.singular_values),
        on_shell_rank=int(on_solve.rank),
        on_shell_nullity=int(on_solve.nullity),
        on_shell_train_residual_rel=float(on_train_rel),
        on_shell_heldout_residual_rel=float(on_heldout_rel),
        bootstrap_principal_angles=tuple(float(v) for v in bootstrap_angles),
        bootstrap_stable=bootstrap_stable,
        joint_singular_values=tuple(float(v) for v in joint_solve.singular_values),
        joint_rank=int(joint_solve.rank),
        joint_nullity=int(joint_solve.nullity),
        certified_basis=certified_basis,
        certified_projector=certified_projector,
        candidates=candidates,
        bracket_certificates=tuple(bracket_records),
        bracket_closure_residual=float(closure_residual),
        bracket_closure_evaluated=bracket_closure_evaluated,
        individual_generators_accepted=individual_generators_accepted,
        closed_truncated_algebra=closed_truncated_algebra,
        promotable_generators=promotable_generators,
        promotable_full_algebra=promotable_full_algebra,
        reason=reason,
        evidence={
            "on_shell_operator": "Pr(F)c=0",
            "off_shell_operator": "[Pr(F)N, -F*Chi] [a,q]=0",
            "coefficient_convention": "pair_major_xi_eta_per_monomial",
            "on_shell_rows": int(A_on.shape[0]),
            "on_shell_train_rows": int(A_train.shape[0]),
            "on_shell_heldout_rows": int(A_heldout.shape[0]),
            "off_shell_rows": int(A_off.shape[0]),
            "generator_unknowns": int(expected),
            "multiplier_unknowns": int(Chi.shape[1]),
            "on_shell_residual_rms": float(_rms(F_on)),
            "off_shell_residual_rms": float(_rms(F_off)),
            "bracket_space": "evaluated_point_vector_fields",
        },
    )


def pair_major_generator(
    coefficients: Sequence[float],
    monomials: Sequence[tuple[int, int]],
    *,
    name: str = "polynomial_point_generator",
) -> PolynomialPointGenerator:
    """Build a polynomial generator from exact pair-major coefficients."""

    c = np.asarray(tuple(coefficients), dtype=float).reshape(-1)
    monoms = tuple((int(px), int(pu)) for px, pu in monomials)
    if c.size != 2 * len(monoms):
        raise ValueError(
            f"pair-major coefficient vector has length {c.size}; expected {2 * len(monoms)}"
        )
    if not np.isfinite(c).all():
        raise ValueError("generator coefficients must be finite")
    xi_terms = tuple(
        (float(c[2 * j]), px, pu)
        for j, (px, pu) in enumerate(monoms)
        if abs(float(c[2 * j])) > 0.0
    )
    eta_terms = tuple(
        (float(c[2 * j + 1]), px, pu)
        for j, (px, pu) in enumerate(monoms)
        if abs(float(c[2 * j + 1])) > 0.0
    )
    return PolynomialPointGenerator(
        name=name,
        family="determining_polynomial",
        xi_terms=xi_terms,
        eta_terms=eta_terms,
        source="coupled_polynomial_determining",
        description="pair-major sparse nullspace representative",
    )


def project_generator_direction(
    result: PolynomialDESymmetryResult, coefficients: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Project a proposed coefficient direction onto the certified subspace."""

    c = np.asarray(tuple(coefficients), dtype=float).reshape(-1)
    if c.size != result.certified_projector.shape[0]:
        raise ValueError("coefficient direction has the wrong length")
    projected = _matrix_vector(result.certified_projector, c)
    denom = max(float(np.linalg.norm(c)), _EPS)
    residual = float(np.linalg.norm(c - projected) / denom)
    return projected, residual


def _residual_label(residual: Any) -> str:
    if isinstance(residual, str):
        return residual
    description = getattr(residual, "description", None)
    return str(description) if description else repr(residual)


def _validate_inputs(inputs: ScalarODEJetInputs, *, min_samples: int, label: str) -> None:
    arrays = [inputs.x, inputs.u, inputs.u1]
    if int(inputs.order) == 2:
        if inputs.u2 is None:
            raise ValueError(f"{label} is missing u_xx for a second-order ODE")
        arrays.append(inputs.u2)
    n = int(inputs.x.shape[0])
    if n < int(min_samples):
        raise ValueError(f"{label} has {n} rows; at least {min_samples} are required")
    for value in arrays:
        if value.dtype != torch.float64:
            raise TypeError(f"{label} must materialize as float64")
        if value.ndim != 2 or value.shape[1] != 1 or int(value.shape[0]) != n:
            raise ValueError(f"{label} columns must all have shape ({n}, 1)")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{label} contains non-finite jet coordinates")


def _determining_operator(
    residual: Any,
    inputs: ScalarODEJetInputs,
    generators: Sequence[PolynomialPointGenerator],
) -> tuple[np.ndarray, np.ndarray]:
    columns: list[np.ndarray] = []
    F_ref: np.ndarray | None = None
    for generator in generators:
        action, F = _pr_action_and_residual(residual, inputs, generator)
        columns.append(action.detach().cpu().numpy().reshape(-1).astype(float, copy=False))
        current = F.detach().cpu().numpy().reshape(-1).astype(float, copy=False)
        if F_ref is None:
            F_ref = current
        elif not np.allclose(F_ref, current, rtol=0.0, atol=0.0):
            raise RuntimeError("residual evaluation changed between generator basis columns")
    n = int(inputs.x.shape[0])
    matrix = np.column_stack(columns) if columns else np.zeros((n, 0), dtype=float)
    return matrix, F_ref if F_ref is not None else np.zeros(n, dtype=float)


def _solve_nullspace(D: np.ndarray, *, rank_rtol: float, rank_atol: float) -> _NullspaceSolve:
    matrix = np.asarray(D, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("determining matrix must be two-dimensional")
    _require_finite_matrix(matrix, "determining matrix")
    rows, cols = matrix.shape
    if cols == 0:
        return _empty_nullspace(0)
    if rows == 0:
        basis = np.eye(cols, dtype=float)
        return _NullspaceSolve(basis, basis.copy(), np.asarray([], dtype=float), 0, float(rank_atol))
    # Economy SVD is sufficient for tall/square matrices.  A wide matrix needs
    # the full Vt so the p-rank structural right-nullspace directions survive.
    full = bool(rows < cols)
    _U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=full)
    s0 = float(singular_values[0]) if singular_values.size else 0.0
    tol = max(float(rank_atol), float(rank_rtol) * max(1.0, s0))
    rank = int(np.sum(singular_values > tol))
    raw = Vt[rank:].T if rank < cols else np.zeros((cols, 0), dtype=float)
    basis = _orthonormal_column_space(raw, rtol=rank_rtol, atol=rank_atol)
    projector = _matrix_matrix(basis, basis.T)
    return _NullspaceSolve(basis, projector, singular_values, rank, tol)


def _empty_nullspace(size: int) -> _NullspaceSolve:
    n = int(size)
    return _NullspaceSolve(
        np.zeros((n, 0), dtype=float),
        np.zeros((n, n), dtype=float),
        np.asarray([], dtype=float),
        0,
        0.0,
    )


def _orthonormal_column_space(M: np.ndarray, *, rtol: float, atol: float) -> np.ndarray:
    matrix = np.asarray(M, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("column-space matrix must be two-dimensional")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    U, singular_values, _Vt = np.linalg.svd(matrix, full_matrices=False)
    s0 = float(singular_values[0]) if singular_values.size else 0.0
    tol = max(float(atol), float(rtol) * max(1.0, s0))
    rank = int(np.sum(singular_values > tol))
    return U[:, :rank]


def _train_heldout_indices(
    n: int, *, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    count = int(n)
    indices = rng.permutation(count)
    heldout_count = int(round(float(fraction) * count))
    if fraction > 0.0:
        heldout_count = max(1, heldout_count)
    heldout_count = min(max(0, heldout_count), count - 1)
    return indices[heldout_count:], indices[:heldout_count]


def _subspace_residual(D: np.ndarray, basis: np.ndarray) -> float:
    if basis.shape[1] == 0:
        return math.inf
    action = _matrix_matrix(D, basis)
    denom = max(float(np.linalg.norm(D, ord="fro")) * math.sqrt(basis.shape[1]), _EPS)
    return float(np.linalg.norm(action, ord="fro") / denom)


def _direction_residual(D: np.ndarray, c: np.ndarray) -> float:
    vector = np.asarray(c, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(D, ord="fro")) * float(np.linalg.norm(vector)), _EPS)
    return float(np.linalg.norm(_matrix_vector(D, vector)) / denom)


def _bootstrap_angles(
    D: np.ndarray,
    reference: _NullspaceSolve,
    *,
    count: int,
    rng: np.random.Generator,
    rank_rtol: float,
    rank_atol: float,
) -> list[float]:
    if count <= 0 or reference.nullity == 0 or D.shape[0] < 2:
        return []
    angles: list[float] = []
    for _ in range(int(count)):
        idx = rng.integers(0, D.shape[0], size=D.shape[0])
        solved = _solve_nullspace(D[idx], rank_rtol=rank_rtol, rank_atol=rank_atol)
        if solved.nullity != reference.nullity:
            angles.append(float(math.pi / 2.0))
            continue
        norm = float(np.linalg.norm(reference.projector - solved.projector, ord=2))
        angles.append(float(math.asin(min(1.0, max(0.0, norm)))))
    return angles


def _multiplier_design(
    inputs: ScalarODEJetInputs, monomials: Sequence[tuple[int, ...]]
) -> np.ndarray:
    coords = [
        inputs.x.detach().cpu().numpy().reshape(-1),
        inputs.u.detach().cpu().numpy().reshape(-1),
        inputs.u1.detach().cpu().numpy().reshape(-1),
    ]
    if int(inputs.order) == 2:
        if inputs.u2 is None:
            raise ValueError("second-order multiplier basis requires u_xx")
        coords.append(inputs.u2.detach().cpu().numpy().reshape(-1))
    columns = []
    for powers in monomials:
        value = np.ones_like(coords[0], dtype=float)
        for coord, power in zip(coords, powers):
            if int(power):
                value = value * np.power(coord, int(power))
        columns.append(value)
    design = np.column_stack(columns) if columns else np.zeros((coords[0].size, 0), dtype=float)
    _require_finite_matrix(design, "multiplier design")
    return design


def _candidate_rows(
    *,
    residual: Any,
    on_inputs: ScalarODEJetInputs,
    off_inputs: ScalarODEJetInputs,
    A_on: np.ndarray,
    A_off: np.ndarray,
    F_off: np.ndarray,
    Chi: np.ndarray,
    certified_basis: np.ndarray,
    certified_projector: np.ndarray,
    monomials: Sequence[tuple[int, int]],
    multiplier_monomials: Sequence[tuple[int, ...]],
    cfg: PolynomialDESymmetryConfig,
) -> tuple[PolynomialDESymmetryCandidate, ...]:
    if certified_basis.shape[1] == 0:
        return ()
    directions = _sparse_subspace_representatives(certified_basis, certified_projector, cfg=cfg)
    rows: list[PolynomialDESymmetryCandidate] = []
    multiplier_matrix = F_off[:, None] * Chi
    for index, c in enumerate(directions):
        action = _matrix_vector(A_off, c)
        q, _residuals, _rank, _s = np.linalg.lstsq(multiplier_matrix, action, rcond=None)
        fitted = _matrix_vector(multiplier_matrix, q)
        relative, absolute = _relative_multiplier_residual(action, fitted, F_off)
        on_relative = _direction_residual(A_on, c)
        support = int(np.sum(np.abs(c) > float(cfg.sparse_zero_tol)))
        accepted = bool(
            on_relative <= float(cfg.on_shell_tol)
            and relative <= float(cfg.off_shell_tol)
        )
        generator = pair_major_generator(c, monomials, name=f"poly_sparse_{index}")
        row = PolynomialDESymmetryCandidate(
            name=f"poly_sparse_{index}",
            coefficients=tuple(float(v) for v in c),
            multiplier_coefficients=tuple(float(v) for v in q),
            generator=generator,
            on_shell_residual_rel=float(on_relative),
            off_shell_relative_residual_rel=float(relative),
            support_size=int(support),
            accepted=accepted,
            evidence={
                "off_shell_abs_rms": float(absolute),
                "multiplier_monomials": [list(v) for v in multiplier_monomials],
                "projector_membership_residual": float(
                    np.linalg.norm(c - _matrix_vector(certified_projector, c))
                    / max(np.linalg.norm(c), _EPS)
                ),
                "on_shell_sample_count": int(on_inputs.x.shape[0]),
                "off_shell_sample_count": int(off_inputs.x.shape[0]),
                "residual_label": _residual_label(residual),
            },
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            0 if row.accepted else 1,
            int(row.support_size),
            float(row.off_shell_relative_residual_rel),
            float(row.on_shell_residual_rel),
            row.name,
        )
    )
    return tuple(rows[: int(cfg.max_candidates)])


def _sparse_subspace_representatives(
    basis: np.ndarray,
    projector: np.ndarray,
    *,
    cfg: PolynomialDESymmetryConfig,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = [basis[:, j] for j in range(basis.shape[1])]
    for j in range(projector.shape[0]):
        projected = projector[:, j]
        if float(np.linalg.norm(projected)) > _EPS:
            starts.append(projected)
    out: list[np.ndarray] = []
    for start in starts:
        direction = np.asarray(start, dtype=float).copy()
        norm = float(np.linalg.norm(direction))
        if norm <= _EPS:
            continue
        direction /= norm
        for _ in range(int(cfg.sparse_iterations)):
            threshold = float(cfg.sparse_threshold) * max(float(np.max(np.abs(direction))), _EPS)
            sparse = np.sign(direction) * np.maximum(np.abs(direction) - threshold, 0.0)
            projected = _matrix_vector(projector, sparse)
            projected_norm = float(np.linalg.norm(projected))
            if projected_norm <= _EPS:
                break
            updated = projected / projected_norm
            if min(float(np.linalg.norm(updated - direction)), float(np.linalg.norm(updated + direction))) <= 1.0e-12:
                direction = updated
                break
            direction = updated
        direction = _canonical_direction(direction)
        if any(abs(float(np.dot(direction, previous))) >= 1.0 - float(cfg.direction_dedup_tol) for previous in out):
            continue
        out.append(direction)
    out.sort(
        key=lambda c: (
            int(np.sum(np.abs(c) > float(cfg.sparse_zero_tol))),
            tuple(float(v) for v in np.round(c, 12)),
        )
    )
    return out[: int(cfg.max_candidates)]


def _canonical_direction(c: np.ndarray) -> np.ndarray:
    direction = np.asarray(c, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if norm <= _EPS or not math.isfinite(norm):
        raise ValueError("cannot normalize a zero or non-finite generator direction")
    direction = direction / norm
    first = next((float(v) for v in direction if abs(float(v)) > 1.0e-12), 1.0)
    return -direction if first < 0.0 else direction


def _relative_multiplier_residual(
    action: np.ndarray, fitted: np.ndarray, F: np.ndarray
) -> tuple[float, float]:
    residual = np.asarray(action) - np.asarray(fitted)
    absolute = _rms(residual)
    scale = max(_rms(action), _rms(fitted), _rms(F), _EPS)
    return float(absolute / scale), float(absolute)


def _evaluated_bracket_certificates(
    basis: np.ndarray,
    monomials: Sequence[tuple[int, int]],
    inputs: ScalarODEJetInputs,
    *,
    tol: float,
) -> tuple[list[EvaluatedBracketCertificate], float]:
    if basis.shape[1] <= 1:
        return [], 0.0
    x = inputs.x.detach().cpu().numpy().reshape(-1)
    u = inputs.u.detach().cpu().numpy().reshape(-1)
    fields = [_polynomial_field_and_partials(basis[:, j], monomials, x, u) for j in range(basis.shape[1])]
    evaluated = np.column_stack(
        [np.concatenate((field[0], field[1])) for field in fields]
    )
    records: list[EvaluatedBracketCertificate] = []
    worst = 0.0
    for left in range(len(fields)):
        for right in range(left + 1, len(fields)):
            a = fields[left]
            b = fields[right]
            bracket_xi = a[0] * b[2] + a[1] * b[3] - b[0] * a[2] - b[1] * a[3]
            bracket_eta = a[0] * b[4] + a[1] * b[5] - b[0] * a[4] - b[1] * a[5]
            bracket = np.concatenate((bracket_xi, bracket_eta))
            bracket_norm = float(np.linalg.norm(bracket))
            if bracket_norm <= _EPS:
                outside = np.zeros_like(bracket)
                relative = 0.0
            else:
                alpha, _residuals, _rank, _s = np.linalg.lstsq(evaluated, bracket, rcond=None)
                outside = bracket - _matrix_vector(evaluated, alpha)
                relative = float(np.linalg.norm(outside) / bracket_norm)
            worst = max(worst, relative)
            records.append(
                EvaluatedBracketCertificate(
                    left=left,
                    right=right,
                    bracket_norm=bracket_norm,
                    outside_span_norm=float(np.linalg.norm(outside)),
                    residual_rel=relative,
                    accepted=bool(relative <= float(tol)),
                )
            )
    return records, float(worst)


def _polynomial_field_and_partials(
    coefficients: np.ndarray,
    monomials: Sequence[tuple[int, int]],
    x: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(coefficients, dtype=float).reshape(-1)
    xi = np.zeros_like(x)
    eta = np.zeros_like(x)
    xi_x = np.zeros_like(x)
    xi_u = np.zeros_like(x)
    eta_x = np.zeros_like(x)
    eta_u = np.zeros_like(x)
    for j, (px, pu) in enumerate(monomials):
        value = np.power(x, px) * np.power(u, pu)
        dx = px * np.power(x, px - 1) * np.power(u, pu) if px else np.zeros_like(x)
        du = pu * np.power(x, px) * np.power(u, pu - 1) if pu else np.zeros_like(x)
        xi += c[2 * j] * value
        eta += c[2 * j + 1] * value
        xi_x += c[2 * j] * dx
        xi_u += c[2 * j] * du
        eta_x += c[2 * j + 1] * dx
        eta_u += c[2 * j + 1] * du
    return xi, eta, xi_x, xi_u, eta_x, eta_u


def _coefficient_labels(monomials: Sequence[tuple[int, int]]) -> list[str]:
    labels: list[str] = []
    for px, pu in monomials:
        name = _monomial_name_2d(px, pu)
        labels.extend((f"xi:{name}", f"eta:{name}"))
    return labels


def _monomial_name_2d(px: int, pu: int) -> str:
    pieces = []
    if px:
        pieces.append("x" if px == 1 else f"x^{px}")
    if pu:
        pieces.append("u" if pu == 1 else f"u^{pu}")
    return "*".join(pieces) if pieces else "1"


def _monomials_2d(max_degree: int) -> list[tuple[int, int]]:
    return [(powers[0], powers[1]) for powers in _total_degree_monomials(2, max_degree)]


def _total_degree_monomials(dimension: int, max_degree: int) -> list[tuple[int, ...]]:
    if int(dimension) < 1:
        raise ValueError("monomial dimension must be positive")
    if int(max_degree) < 0:
        raise ValueError("monomial degree must be nonnegative")
    out: list[tuple[int, ...]] = []
    for total in range(int(max_degree) + 1):
        out.extend(_compositions_descending(total, int(dimension)))
    return out


def _compositions_descending(total: int, length: int) -> list[tuple[int, ...]]:
    if length == 1:
        return [(int(total),)]
    out: list[tuple[int, ...]] = []
    for first in range(int(total), -1, -1):
        for tail in _compositions_descending(int(total) - first, length - 1):
            out.append((first,) + tail)
    return out


def _require_finite_matrix(matrix: np.ndarray, label: str) -> None:
    if not np.isfinite(np.asarray(matrix)).all():
        raise ValueError(f"{label} contains non-finite values")


def _matrix_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Small dense product without platform BLAS matmul overflow warnings.

    Some macOS Accelerate/NumPy combinations have emitted spurious overflow
    warnings for finite, order-ten determining matrices.  ``einsum`` is just
    as clear for these small coefficient spaces and avoids that backend path.
    """

    return np.einsum("ij,jk->ik", np.asarray(left, dtype=float), np.asarray(right, dtype=float), optimize=False)


def _matrix_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", np.asarray(matrix, dtype=float), np.asarray(vector, dtype=float), optimize=False)


def _rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return math.inf
    return float(np.sqrt(np.mean(np.square(array))))


__all__ = [
    "EvaluatedBracketCertificate",
    "PolynomialDESymmetryCandidate",
    "PolynomialDESymmetryConfig",
    "PolynomialDESymmetryResult",
    "pair_major_generator",
    "project_generator_direction",
    "recover_polynomial_de_symmetries",
]
