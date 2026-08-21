# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Shared numerical foundations for finite-dimensional Poisson discovery.

The module deliberately contains no Hamiltonian or Jacobi model-selection
policy.  It supplies the two pieces shared by every Poisson hypothesis class:

* a batched autonomous vector-field interface exposing values and Jacobians;
* a basis-independent, diagnostic-rich nullspace solve.

All tensor conventions are explicit.  A vector-field Jacobian has entries
``jacobian[n, i, k] = partial_k f^i(z_n)``.  Nullspace bases are stored as
orthonormal *columns*, so their projector is independent of the arbitrary SVD
basis used to represent the subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import torch


TensorCallable = Callable[[torch.Tensor], torch.Tensor]


def validate_state_points(z: torch.Tensor, state_dim: int | None = None) -> None:
    """Validate the common batched, real-valued state-point contract."""

    if not isinstance(z, torch.Tensor):
        raise TypeError("state points must be a torch.Tensor")
    if z.ndim != 2:
        raise ValueError(f"state points must have shape (n_samples, state_dim), got {tuple(z.shape)}")
    if z.shape[0] < 1:
        raise ValueError("at least one state point is required")
    if z.shape[1] < 1:
        raise ValueError("state_dim must be positive")
    if state_dim is not None and z.shape[1] != int(state_dim):
        raise ValueError(f"state dimension {z.shape[1]} != expected {int(state_dim)}")
    if not z.dtype.is_floating_point:
        raise TypeError("state points must use a real floating dtype")
    if not torch.isfinite(z).all():
        raise ValueError("state points contain non-finite values")


# Backward-compatible private spelling for sibling modules developed alongside
# this foundation layer.  New code should use the public function above.
_validate_state_points = validate_state_points


def _coerce_like(value: torch.Tensor, reference: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must return a torch.Tensor")
    if value.device != reference.device or value.dtype != reference.dtype:
        value = value.to(device=reference.device, dtype=reference.dtype)
    return value


class VectorField:
    """Batched autonomous vector field with analytic or automatic Jacobians.

    Parameters
    ----------
    value_fn:
        Callable mapping ``Z`` with shape ``(N, d)`` to ``F`` with shape
        ``(N, d)``.  The callable is assumed to act independently on samples.
    jacobian_fn:
        Optional callable returning ``(N, d, d)`` with the convention
        ``J[n, i, k] = partial_k f^i(Z[n])``.  A constant ``(d, d)`` Jacobian
        is accepted and broadcast over samples.  When omitted, exact PyTorch
        derivatives are used.
    state_dim:
        Optional dimension contract checked on every call.
    create_graph:
        Retain a derivative graph for automatically evaluated Jacobians.

    Notes
    -----
    The autograd fallback differentiates sums over the batch.  It is exact for
    the documented pointwise/batched callable contract and avoids constructing
    the much larger full ``(N*d) x (N*d)`` batch Jacobian.
    """

    def __init__(
        self,
        value_fn: TensorCallable,
        jacobian_fn: TensorCallable | None = None,
        *,
        state_dim: int | None = None,
        create_graph: bool = False,
    ) -> None:
        if not callable(value_fn):
            raise TypeError("value_fn must be callable")
        if jacobian_fn is not None and not callable(jacobian_fn):
            raise TypeError("jacobian_fn must be callable or None")
        if state_dim is not None and int(state_dim) < 1:
            raise ValueError("state_dim must be positive")
        self._value_fn = value_fn
        self._jacobian_fn = jacobian_fn
        self.state_dim = None if state_dim is None else int(state_dim)
        self.create_graph = bool(create_graph)

    @classmethod
    def from_callable(
        cls,
        value_fn: TensorCallable,
        jacobian_fn: TensorCallable | None = None,
        *,
        state_dim: int | None = None,
        create_graph: bool = False,
    ) -> "VectorField":
        """Named constructor mirroring the public callable-based API."""

        return cls(
            value_fn,
            jacobian_fn,
            state_dim=state_dim,
            create_graph=create_graph,
        )

    def _evaluate_value(self, z: torch.Tensor) -> torch.Tensor:
        values = _coerce_like(self._value_fn(z), z, name="value_fn")
        expected = (z.shape[0], z.shape[1])
        if tuple(values.shape) != expected:
            raise ValueError(f"value_fn returned shape {tuple(values.shape)}, expected {expected}")
        return values

    def value(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate ``f(Z)`` while preserving input dtype and device."""

        validate_state_points(z, self.state_dim)
        return self._evaluate_value(z)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate ``Df(Z)`` using the explicit callable or autograd."""

        return self.value_and_jacobian(z)[1]

    def value_and_jacobian(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate values and pointwise state Jacobians in one public call."""

        validate_state_points(z, self.state_dim)
        n_samples, dimension = z.shape

        if self._jacobian_fn is not None:
            values = self._evaluate_value(z)
            jacobian = _coerce_like(self._jacobian_fn(z), z, name="jacobian_fn")
            if jacobian.ndim == 2 and tuple(jacobian.shape) == (dimension, dimension):
                jacobian = jacobian.unsqueeze(0).expand(n_samples, -1, -1)
            expected = (n_samples, dimension, dimension)
            if tuple(jacobian.shape) != expected:
                raise ValueError(
                    f"jacobian_fn returned shape {tuple(jacobian.shape)}, expected {expected}"
                )
            return values, jacobian

        # Evaluation must remain differentiable even inside an outer no_grad
        # block.  Reuse the caller's leaf when possible; otherwise create a
        # local leaf without mutating the input's requires_grad flag.
        with torch.enable_grad():
            work = z if z.requires_grad else z.detach().clone().requires_grad_(True)
            values = self._evaluate_value(work)
            rows: list[torch.Tensor] = []
            for component in range(dimension):
                scalar = values[:, component].sum()
                if scalar.requires_grad:
                    grad = torch.autograd.grad(
                        scalar,
                        work,
                        retain_graph=component + 1 < dimension,
                        create_graph=self.create_graph,
                        allow_unused=True,
                    )[0]
                else:
                    grad = None
                if grad is None:
                    grad = torch.zeros_like(work)
                rows.append(grad)
            jacobian = torch.stack(rows, dim=1)
        return values, jacobian


@dataclass(frozen=True)
class StableNullspaceConfig:
    """Numerical and resampling policy for :func:`stable_nullspace`."""

    rank_rtol: float = 1.0e-9
    rank_atol: float = 1.0e-11
    nullity_strategy: Literal["rank_tol", "spectral_gap"] = "rank_tol"
    min_spectral_gap: float = 10.0
    max_gap_nullity: int | None = None
    bootstrap: int = 0
    bootstrap_block_size: int = 1
    random_seed: int = 0
    near_null_max_vectors: int = 0
    near_null_min_spectral_gap: float = 10.0

    def __post_init__(self) -> None:
        if self.rank_rtol < 0.0 or self.rank_atol < 0.0:
            raise ValueError("rank tolerances must be non-negative")
        if self.nullity_strategy not in {"rank_tol", "spectral_gap"}:
            raise ValueError("nullity_strategy must be 'rank_tol' or 'spectral_gap'")
        if self.min_spectral_gap <= 1.0:
            raise ValueError("min_spectral_gap must be greater than one")
        if self.max_gap_nullity is not None and self.max_gap_nullity < 1:
            raise ValueError("max_gap_nullity must be positive when provided")
        if self.bootstrap < 0:
            raise ValueError("bootstrap must be non-negative")
        if self.bootstrap_block_size < 1:
            raise ValueError("bootstrap_block_size must be positive")
        if self.near_null_max_vectors < 0:
            raise ValueError("near_null_max_vectors must be non-negative")
        if self.near_null_min_spectral_gap <= 1.0:
            raise ValueError("near_null_min_spectral_gap must be greater than one")


@dataclass(frozen=True)
class StableNullspaceResult:
    """Basis-independent determining-nullspace result.

    ``heldout_principal_angle`` and every element of
    ``bootstrap_principal_angles`` are maximum principal angles in radians.
    A value of ``pi/2`` records a nullity mismatch rather than hiding it in a
    comparison between unequal-dimensional subspaces.
    """

    basis: torch.Tensor
    projector: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    exact_rank: int
    rank_tolerance: float
    nullity: int
    exact_nullity: int
    tier: Literal["exact", "noise_calibrated", "none"]
    best_vector: torch.Tensor
    train_residual_relative: float
    heldout_residual_relative: float | None
    heldout_principal_angle: float | None
    bootstrap_principal_angles: tuple[float, ...]

    @property
    def max_bootstrap_principal_angle(self) -> float | None:
        if not self.bootstrap_principal_angles:
            return None
        return max(self.bootstrap_principal_angles)


def _validate_matrix(matrix: torch.Tensor, *, name: str, columns: int | None = None) -> None:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must have at least one column")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(f"{name} has {matrix.shape[1]} columns, expected {columns}")
    if not matrix.dtype.is_floating_point:
        raise TypeError(f"{name} must use a real floating dtype")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")


def _spectral_gap_rank(
    singular_values: torch.Tensor,
    *,
    unknown_count: int,
    min_spectral_gap: float,
    max_gap_nullity: int | None,
) -> int:
    """Choose a rank cut at the largest accepted tail singular-value gap."""

    value_count = int(singular_values.numel())
    if value_count < 2:
        return value_count
    maximum_nullity = unknown_count if max_gap_nullity is None else int(max_gap_nullity)
    # rank=k implies nullity=unknown_count-k, including structural zeros that
    # are absent from the compact singular-value array when rows < columns.
    lower_rank = max(1, unknown_count - maximum_nullity)
    lower_rank = min(lower_rank, value_count - 1)
    best_rank = value_count
    best_gap = 0.0
    tiny = torch.finfo(singular_values.dtype).tiny
    for rank in range(lower_rank, value_count):
        gap = float((singular_values[rank - 1] / singular_values[rank].clamp_min(tiny)).item())
        if gap > best_gap:
            best_gap = gap
            best_rank = rank
    return best_rank if best_gap >= min_spectral_gap else value_count


def _solve_nullspace(
    matrix: torch.Tensor,
    config: StableNullspaceConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    float,
    torch.Tensor,
    int,
    Literal["exact", "noise_calibrated", "none"],
]:
    unknown_count = int(matrix.shape[1])
    if matrix.shape[0] == 0:
        basis = torch.eye(unknown_count, device=matrix.device, dtype=matrix.dtype)
        return (
            basis,
            basis,
            torch.empty(0, device=matrix.device, dtype=matrix.dtype),
            0,
            float(config.rank_atol),
            basis[:, 0],
            0,
            "exact",
        )

    # Tall determining systems are the normal discovery case.  A full SVD
    # would materialize an unused (rows x rows) left-singular matrix.  Reduced
    # SVD still returns the complete square Vh when rows >= columns.  Only a
    # genuinely wide system needs the full Vh to retain its structural right
    # nullspace beyond the number of rows.
    full_matrices = matrix.shape[0] < matrix.shape[1]
    _u, singular_values, vh = torch.linalg.svd(
        matrix,
        full_matrices=full_matrices,
    )
    leading = float(singular_values[0].item()) if singular_values.numel() else 0.0
    rank_tolerance = max(float(config.rank_atol), float(config.rank_rtol) * max(1.0, leading))
    tolerance_rank = int(torch.count_nonzero(singular_values > rank_tolerance).item())
    if config.nullity_strategy == "spectral_gap":
        # Exact or clearly numerical zeros should not be promoted to signal
        # merely because a one-element/all-zero spectrum has no definable gap.
        if tolerance_rank < singular_values.numel():
            rank = tolerance_rank
        else:
            rank = _spectral_gap_rank(
                singular_values,
                unknown_count=unknown_count,
                min_spectral_gap=float(config.min_spectral_gap),
                max_gap_nullity=config.max_gap_nullity,
            )
    else:
        rank = tolerance_rank
    rank = min(rank, unknown_count)
    exact_rank = rank
    tier: Literal["exact", "noise_calibrated", "none"] = (
        "exact" if exact_rank < unknown_count else "none"
    )
    if exact_rank == unknown_count and int(config.near_null_max_vectors) > 0:
        if unknown_count == 1:
            # A one-column determining operator has no internal spectral gap.
            # Retain its sole direction only as a proposal; downstream
            # invariance, held-out, bootstrap, Jacobi, and Hamiltonian gates
            # decide whether the noise-calibrated tier can promote it.
            rank = 0
            tier = "noise_calibrated"
        else:
            proposed_rank = _spectral_gap_rank(
                singular_values,
                unknown_count=unknown_count,
                min_spectral_gap=float(config.near_null_min_spectral_gap),
                max_gap_nullity=int(config.near_null_max_vectors),
            )
            if proposed_rank < exact_rank:
                rank = proposed_rank
                tier = "noise_calibrated"
    basis = vh[rank:].mT.contiguous()
    projector = basis @ basis.mT
    best_vector = vh[-1].conj().clone()
    best_norm = torch.linalg.vector_norm(best_vector)
    if float(best_norm.item()) > 0.0:
        best_vector = best_vector / best_norm
    return (
        basis,
        projector,
        singular_values,
        rank,
        rank_tolerance,
        best_vector,
        exact_rank,
        tier,
    )


def _relative_residual(matrix: torch.Tensor, directions: torch.Tensor) -> float:
    if directions.numel() == 0:
        return math.inf
    numerator = torch.linalg.matrix_norm(matrix @ directions)
    denominator = torch.linalg.matrix_norm(matrix) * torch.linalg.matrix_norm(directions)
    floor = torch.finfo(matrix.dtype).eps
    return float((numerator / denominator.clamp_min(floor)).item())


def _maximum_principal_angle(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape[1] != candidate.shape[1]:
        return math.pi / 2.0
    if reference.shape[1] == 0:
        return 0.0
    cosines = torch.linalg.svdvals(reference.mT @ candidate)
    smallest_cosine = cosines.min().clamp(0.0, 1.0)
    return float(torch.acos(smallest_cosine).item())


def _canonicalise_nullspace_direction(
    direction: torch.Tensor,
) -> torch.Tensor | None:
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm.item()) <= 1.0e-14:
        return None
    canonical = direction / norm
    pivot = int(torch.argmax(canonical.abs()).item())
    if float(canonical[pivot].item()) < 0.0:
        canonical = -canonical
    return canonical


def sparse_nullspace_representatives(
    nullspace: StableNullspaceResult,
    *,
    max_representatives: int = 48,
    sparse_rotation_steps: int = 16,
    random_seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Return deterministic sparse views of a nullspace without privileging its SVD basis.

    The intrinsic object is ``nullspace.projector``.  Coordinate projections
    and projected soft-threshold iterations therefore seed the representative
    search before reproducible random rotations are considered.  Every emitted
    direction is reprojected into the certified subspace.
    """

    limit = int(max_representatives)
    steps = int(sparse_rotation_steps)
    if limit < 1:
        raise ValueError("max_representatives must be positive")
    if steps < 0:
        raise ValueError("sparse_rotation_steps must be non-negative")
    if nullspace.nullity == 0:
        return ()
    basis = nullspace.basis
    projector = nullspace.projector
    seeds: list[torch.Tensor] = []

    coordinate_norms = torch.linalg.vector_norm(projector, dim=0)
    for index in torch.argsort(coordinate_norms, descending=True).tolist():
        if float(coordinate_norms[index].item()) <= 1.0e-12:
            continue
        seeds.append(projector[:, index])
        if len(seeds) >= 2 * limit:
            break
    seeds.extend(basis[:, index] for index in range(basis.shape[1]))

    if nullspace.nullity > 1:
        generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        random_count = min(limit, 8 * int(nullspace.nullity))
        weights = torch.randn(
            random_count,
            nullspace.nullity,
            generator=generator,
            dtype=basis.dtype,
        ).to(device=basis.device)
        seeds.extend(basis @ weights[row] for row in range(random_count))

    directions: list[torch.Tensor] = []

    def add(direction: torch.Tensor) -> None:
        canonical = _canonicalise_nullspace_direction(projector @ direction)
        if canonical is None:
            return
        if any(
            float(torch.abs(torch.dot(canonical, old)).item()) > 1.0 - 1.0e-7
            for old in directions
        ):
            return
        directions.append(canonical)

    for seed in seeds:
        add(seed)
        work = _canonicalise_nullspace_direction(projector @ seed)
        if work is not None and steps > 0:
            tau = 0.05 * float(work.abs().max().item())
            for _ in range(steps):
                shrunk = torch.sign(work) * torch.clamp(work.abs() - tau, min=0.0)
                work = _canonicalise_nullspace_direction(projector @ shrunk)
                if work is None:
                    break
            if work is not None:
                add(work)
        if len(directions) >= limit:
            break
    return tuple(directions[:limit])


def _bootstrap_indices(
    row_count: int,
    block_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if row_count % block_size:
        raise ValueError(
            f"matrix row count {row_count} is not divisible by bootstrap_block_size {block_size}"
        )
    block_count = row_count // block_size
    sampled_blocks = torch.randint(block_count, (block_count,), generator=generator)
    offsets = torch.arange(block_size)
    indices = (sampled_blocks[:, None] * block_size + offsets[None, :]).reshape(-1)
    return indices.to(device=device)


def stable_nullspace(
    matrix: torch.Tensor,
    heldout_matrix: torch.Tensor | None = None,
    *,
    config: StableNullspaceConfig | None = None,
) -> StableNullspaceResult:
    """Solve and diagnose the right nullspace of a determining matrix.

    The routine uses a direct SVD rather than a Gram matrix, avoiding the
    squared condition number that is particularly harmful near a determining
    nullspace.  Bootstrap resampling is block-aware: for a Poisson determining
    matrix, set ``bootstrap_block_size = d*(d-1)//2`` so every state sample's
    tensor equations are resampled together.
    """

    policy = StableNullspaceConfig() if config is None else config
    _validate_matrix(matrix, name="matrix")
    if heldout_matrix is not None:
        _validate_matrix(heldout_matrix, name="heldout_matrix", columns=matrix.shape[1])
        if heldout_matrix.device != matrix.device or heldout_matrix.dtype != matrix.dtype:
            raise ValueError("heldout_matrix must share matrix dtype and device")

    (
        basis,
        projector,
        singular_values,
        rank,
        rank_tolerance,
        best_vector,
        exact_rank,
        tier,
    ) = _solve_nullspace(matrix, policy)
    nullity = int(basis.shape[1])
    directions = basis if nullity else best_vector[:, None]
    train_residual = _relative_residual(matrix, directions)

    heldout_residual: float | None = None
    heldout_angle: float | None = None
    if heldout_matrix is not None and heldout_matrix.shape[0] > 0:
        heldout_residual = _relative_residual(heldout_matrix, directions)
        heldout_basis, _p, _s, _r, _t, _b, _er, _tier = _solve_nullspace(
            heldout_matrix, policy
        )
        heldout_angle = _maximum_principal_angle(basis, heldout_basis)

    bootstrap_angles: list[float] = []
    if policy.bootstrap > 0 and matrix.shape[0] > 1:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(policy.random_seed))
        for _ in range(int(policy.bootstrap)):
            indices = _bootstrap_indices(
                matrix.shape[0],
                int(policy.bootstrap_block_size),
                generator=generator,
                device=matrix.device,
            )
            replicate_basis, _p, _s, _r, _t, _b, _er, _tier = _solve_nullspace(
                matrix[indices], policy
            )
            bootstrap_angles.append(_maximum_principal_angle(basis, replicate_basis))

    return StableNullspaceResult(
        basis=basis,
        projector=projector,
        singular_values=singular_values,
        rank=rank,
        exact_rank=exact_rank,
        rank_tolerance=rank_tolerance,
        nullity=nullity,
        exact_nullity=max(0, int(matrix.shape[1]) - int(exact_rank)),
        tier=tier,
        best_vector=best_vector,
        train_residual_relative=train_residual,
        heldout_residual_relative=heldout_residual,
        heldout_principal_angle=heldout_angle,
        bootstrap_principal_angles=tuple(bootstrap_angles),
    )


__all__ = [
    "StableNullspaceConfig",
    "StableNullspaceResult",
    "VectorField",
    "sparse_nullspace_representatives",
    "stable_nullspace",
    "validate_state_points",
]
