# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Scalar and bivector bases for Poisson determining equations.

A bivector coefficient vector is ordered pair-major.  For upper-triangle pair
``pairs[p] = (i, j)`` and scalar term ``a``, its flat index is
``p * scalar_basis.size + a``.  The represented basis element is

``psi_a(z) * (e_i wedge e_j)``.

This makes antisymmetry exact by construction and gives each state sample
``d*(d-1)//2`` independent Lie-derivative equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Callable, Protocol, Sequence, runtime_checkable

import torch

from .poisson_core import validate_state_points


@dataclass(frozen=True)
class ScalarBasisEvaluation:
    """Values ``(N,K)`` and analytic gradients ``(N,K,d)``."""

    values: torch.Tensor
    gradients: torch.Tensor


@runtime_checkable
class ScalarBasis(Protocol):
    """Runtime-checkable scalar-basis contract used by :class:`BivectorBasis`."""

    state_dim: int
    size: int

    def evaluate(self, z: torch.Tensor) -> ScalarBasisEvaluation: ...


def polynomial_exponents(
    state_dim: int,
    max_degree: int,
    *,
    include_constant: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate total-degree monomial exponents in deterministic order."""

    dimension = int(state_dim)
    degree_limit = int(max_degree)
    if dimension < 1:
        raise ValueError("state_dim must be positive")
    if degree_limit < 0:
        raise ValueError("max_degree must be non-negative")
    out: list[tuple[int, ...]] = []
    if include_constant:
        out.append((0,) * dimension)
    for degree in range(1, degree_limit + 1):
        for variable_indices in combinations_with_replacement(range(dimension), degree):
            exponent = [0] * dimension
            for index in variable_indices:
                exponent[index] += 1
            out.append(tuple(exponent))
    if not out:
        raise ValueError("the requested polynomial basis is empty")
    return tuple(out)


class PolynomialScalarBasis:
    """Constant, linear, or quadratic total-degree monomial basis.

    Higher non-negative degrees are accepted as a convenience, although the
    initial Poisson lanes use only ``max_degree`` 0, 1, and 2.
    """

    def __init__(
        self,
        state_dim: int,
        max_degree: int = 1,
        *,
        include_constant: bool = True,
    ) -> None:
        self.state_dim = int(state_dim)
        self.max_degree = int(max_degree)
        self.include_constant = bool(include_constant)
        self.exponents = polynomial_exponents(
            self.state_dim,
            self.max_degree,
            include_constant=self.include_constant,
        )
        self.size = len(self.exponents)

    def evaluate(self, z: torch.Tensor) -> ScalarBasisEvaluation:
        """Evaluate monomials and first derivatives without division by ``z``."""

        validate_state_points(z, self.state_dim)
        values: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for exponent in self.exponents:
            value = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
            for coordinate, power in enumerate(exponent):
                if power:
                    value = value * z[:, coordinate].pow(power)
            values.append(value)

            coordinate_gradients: list[torch.Tensor] = []
            for derivative_coordinate, derivative_power in enumerate(exponent):
                if derivative_power == 0:
                    coordinate_gradients.append(torch.zeros_like(value))
                    continue
                derivative = torch.full_like(value, float(derivative_power))
                for coordinate, power in enumerate(exponent):
                    effective_power = power - 1 if coordinate == derivative_coordinate else power
                    if effective_power:
                        derivative = derivative * z[:, coordinate].pow(effective_power)
                coordinate_gradients.append(derivative)
            gradients.append(torch.stack(coordinate_gradients, dim=-1))
        return ScalarBasisEvaluation(
            values=torch.stack(values, dim=1),
            gradients=torch.stack(gradients, dim=1),
        )

    __call__ = evaluate


class CallableScalarBasis:
    """Adapter for callable, AST-backed, or learned scalar term libraries.

    ``value_fn`` must return all basis values as ``(N,K)``.  ``gradient_fn``
    may return analytic gradients as ``(N,K,d)``; otherwise pointwise PyTorch
    autograd supplies them.  AST libraries can therefore participate without
    coupling this low-level module to a particular AST node hierarchy.
    """

    def __init__(
        self,
        state_dim: int,
        size: int,
        value_fn: Callable[[torch.Tensor], torch.Tensor],
        gradient_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        names: Sequence[str] | None = None,
        create_graph: bool = False,
    ) -> None:
        self.state_dim = int(state_dim)
        self.size = int(size)
        if self.state_dim < 1 or self.size < 1:
            raise ValueError("state_dim and size must be positive")
        if not callable(value_fn):
            raise TypeError("value_fn must be callable")
        if gradient_fn is not None and not callable(gradient_fn):
            raise TypeError("gradient_fn must be callable or None")
        if names is not None and len(names) != self.size:
            raise ValueError("names length must equal basis size")
        self.names = None if names is None else tuple(str(name) for name in names)
        self._value_fn = value_fn
        self._gradient_fn = gradient_fn
        self.create_graph = bool(create_graph)

    def _values(self, z: torch.Tensor) -> torch.Tensor:
        values = self._value_fn(z)
        if not isinstance(values, torch.Tensor):
            raise TypeError("value_fn must return a torch.Tensor")
        values = values.to(device=z.device, dtype=z.dtype)
        expected = (z.shape[0], self.size)
        if tuple(values.shape) != expected:
            raise ValueError(f"value_fn returned shape {tuple(values.shape)}, expected {expected}")
        return values

    def evaluate(self, z: torch.Tensor) -> ScalarBasisEvaluation:
        validate_state_points(z, self.state_dim)
        if self._gradient_fn is not None:
            values = self._values(z)
            gradients = self._gradient_fn(z)
            if not isinstance(gradients, torch.Tensor):
                raise TypeError("gradient_fn must return a torch.Tensor")
            gradients = gradients.to(device=z.device, dtype=z.dtype)
            expected = (z.shape[0], self.size, self.state_dim)
            if tuple(gradients.shape) != expected:
                raise ValueError(
                    f"gradient_fn returned shape {tuple(gradients.shape)}, expected {expected}"
                )
            return ScalarBasisEvaluation(values=values, gradients=gradients)

        with torch.enable_grad():
            work = z if z.requires_grad else z.detach().clone().requires_grad_(True)
            values = self._values(work)
            gradients_list: list[torch.Tensor] = []
            for term in range(self.size):
                scalar = values[:, term].sum()
                if scalar.requires_grad:
                    gradient = torch.autograd.grad(
                        scalar,
                        work,
                        retain_graph=term + 1 < self.size,
                        create_graph=self.create_graph,
                        allow_unused=True,
                    )[0]
                else:
                    gradient = None
                if gradient is None:
                    gradient = torch.zeros_like(work)
                gradients_list.append(gradient)
        return ScalarBasisEvaluation(values=values, gradients=torch.stack(gradients_list, dim=1))

    __call__ = evaluate


def upper_triangle_pairs(state_dim: int) -> tuple[tuple[int, int], ...]:
    """Return the independent skew-tensor index pairs in lexicographic order."""

    dimension = int(state_dim)
    if dimension < 2:
        raise ValueError("a bivector requires state_dim >= 2")
    return tuple((i, j) for i in range(dimension) for j in range(i + 1, dimension))


def _wedge_matrices(
    pairs: Sequence[tuple[int, int]],
    state_dim: int,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    wedges = torch.zeros(
        len(pairs),
        state_dim,
        state_dim,
        device=reference.device,
        dtype=reference.dtype,
    )
    for index, (i, j) in enumerate(pairs):
        wedges[index, i, j] = 1.0
        wedges[index, j, i] = -1.0
    return wedges


def assemble_skew_tensor(
    upper_values: torch.Tensor,
    *,
    state_dim: int,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Assemble skew matrices from values whose final axis indexes ``i<j``."""

    if not isinstance(upper_values, torch.Tensor):
        raise TypeError("upper_values must be a torch.Tensor")
    pair_order = upper_triangle_pairs(state_dim) if pairs is None else tuple(pairs)
    if upper_values.ndim < 1 or upper_values.shape[-1] != len(pair_order):
        raise ValueError(
            f"upper_values final dimension must be {len(pair_order)}, got {tuple(upper_values.shape)}"
        )
    wedges = _wedge_matrices(pair_order, int(state_dim), reference=upper_values)
    return torch.einsum("...p,pij->...ij", upper_values, wedges)


def extract_upper_triangle(
    tensors: torch.Tensor,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Extract lexicographically ordered upper-triangle entries."""

    if not isinstance(tensors, torch.Tensor):
        raise TypeError("tensors must be a torch.Tensor")
    if tensors.ndim < 2 or tensors.shape[-1] != tensors.shape[-2]:
        raise ValueError("tensors must end in square matrix dimensions")
    dimension = int(tensors.shape[-1])
    pair_order = upper_triangle_pairs(dimension) if pairs is None else tuple(pairs)
    return torch.stack([tensors[..., i, j] for i, j in pair_order], dim=-1)


@dataclass(frozen=True)
class BivectorBasisEvaluation:
    """Bivector basis values and coordinate derivatives.

    Shapes are ``values=(N,M,d,d)`` and ``derivatives=(N,M,d,d,d)``, with
    the final derivative axis satisfying
    ``derivatives[n,m,i,j,k] = partial_k B_m^{ij}(z_n)``.
    """

    values: torch.Tensor
    derivatives: torch.Tensor
    pairs: tuple[tuple[int, int], ...]
    scalar_size: int


@dataclass(frozen=True)
class BivectorEvaluation:
    """A represented bivector ``Pi`` and its first coordinate derivatives."""

    tensor: torch.Tensor
    derivatives: torch.Tensor

    @property
    def Pi(self) -> torch.Tensor:
        return self.tensor

    @property
    def dPi(self) -> torch.Tensor:
        return self.derivatives


class BivectorBasis:
    """Antisymmetric tensor basis induced by any scalar basis."""

    def __init__(self, state_dim: int, scalar_basis: ScalarBasis) -> None:
        self.state_dim = int(state_dim)
        if self.state_dim < 2:
            raise ValueError("a bivector requires state_dim >= 2")
        if not isinstance(scalar_basis, ScalarBasis):
            raise TypeError("scalar_basis does not satisfy the ScalarBasis protocol")
        if int(scalar_basis.state_dim) != self.state_dim:
            raise ValueError("scalar basis and bivector state dimensions differ")
        self.scalar_basis = scalar_basis
        self.scalar_size = int(scalar_basis.size)
        self.pairs = upper_triangle_pairs(self.state_dim)
        self.size = len(self.pairs) * self.scalar_size

    def flat_index(self, pair_index: int, scalar_index: int) -> int:
        if not 0 <= int(pair_index) < len(self.pairs):
            raise IndexError("pair_index out of range")
        if not 0 <= int(scalar_index) < self.scalar_size:
            raise IndexError("scalar_index out of range")
        return int(pair_index) * self.scalar_size + int(scalar_index)

    def evaluate(self, z: torch.Tensor) -> BivectorBasisEvaluation:
        scalar = self.scalar_basis.evaluate(z)
        wedges = _wedge_matrices(self.pairs, self.state_dim, reference=z)
        values = torch.einsum("na,pij->npaij", scalar.values, wedges)
        derivatives = torch.einsum("nak,pij->npaijk", scalar.gradients, wedges)
        return BivectorBasisEvaluation(
            values=values.reshape(z.shape[0], self.size, self.state_dim, self.state_dim),
            derivatives=derivatives.reshape(
                z.shape[0],
                self.size,
                self.state_dim,
                self.state_dim,
                self.state_dim,
            ),
            pairs=self.pairs,
            scalar_size=self.scalar_size,
        )

    __call__ = evaluate

    def assemble(self, coefficients: torch.Tensor, z: torch.Tensor) -> BivectorEvaluation:
        """Evaluate one coefficient vector without materializing every basis tensor."""

        validate_state_points(z, self.state_dim)
        if not isinstance(coefficients, torch.Tensor):
            raise TypeError("coefficients must be a torch.Tensor")
        if coefficients.ndim != 1 or coefficients.numel() != self.size:
            raise ValueError(f"coefficients must have shape ({self.size},)")
        coefficients = coefficients.to(device=z.device, dtype=z.dtype)
        scalar = self.scalar_basis.evaluate(z)
        coefficient_table = coefficients.reshape(len(self.pairs), self.scalar_size)
        upper = torch.einsum("pa,na->np", coefficient_table, scalar.values)
        upper_derivatives = torch.einsum("pa,nak->npk", coefficient_table, scalar.gradients)
        tensor = assemble_skew_tensor(upper, state_dim=self.state_dim, pairs=self.pairs)
        # Move the derivative coordinate before the pair axis for assembly, then
        # restore the public (N,i,j,k) derivative convention.
        derivatives = assemble_skew_tensor(
            upper_derivatives.movedim(-1, -2),
            state_dim=self.state_dim,
            pairs=self.pairs,
        ).movedim(-3, -1)
        return BivectorEvaluation(tensor=tensor, derivatives=derivatives)

    def determining_matrix(
        self,
        z: torch.Tensor,
        field_values: torch.Tensor,
        field_jacobians: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``L_f`` directly, without materializing tensor-basis derivatives."""

        validate_state_points(z, self.state_dim)
        return build_poisson_determining_matrix_from_scalar(
            field_values,
            field_jacobians,
            self.scalar_basis.evaluate(z),
            state_dim=self.state_dim,
            pairs=self.pairs,
        )


def build_poisson_determining_matrix(
    field_values: torch.Tensor,
    field_jacobians: torch.Tensor,
    basis: BivectorBasisEvaluation,
) -> torch.Tensor:
    """Build columns of the bivector Lie-derivative operator ``L_f``.

    The implemented coordinate formula is

    ``(L_f B)^ij = f^k partial_k B^ij
                    - (partial_k f^i) B^kj
                    - (partial_k f^j) B^ik``.

    Rows are sample-major and then upper-triangle-pair-major; columns follow
    the pair-major/scalar-major convention of :class:`BivectorBasis`.
    """

    if not isinstance(field_values, torch.Tensor) or not isinstance(field_jacobians, torch.Tensor):
        raise TypeError("field values and Jacobians must be torch tensors")
    if field_values.ndim != 2:
        raise ValueError("field_values must have shape (N,d)")
    n_samples, dimension = field_values.shape
    if tuple(field_jacobians.shape) != (n_samples, dimension, dimension):
        raise ValueError("field_jacobians must have shape (N,d,d)")
    if basis.values.ndim != 4 or tuple(basis.values.shape[:1] + basis.values.shape[-2:]) != (
        n_samples,
        dimension,
        dimension,
    ):
        raise ValueError("bivector basis values must have shape (N,M,d,d)")
    expected_derivative_tail = (dimension, dimension, dimension)
    if basis.derivatives.ndim != 5 or tuple(basis.derivatives.shape[0:1]) != (n_samples,):
        raise ValueError("bivector basis derivatives must have shape (N,M,d,d,d)")
    if tuple(basis.derivatives.shape[-3:]) != expected_derivative_tail:
        raise ValueError("bivector basis derivative coordinate dimensions are inconsistent")
    if basis.derivatives.shape[1] != basis.values.shape[1]:
        raise ValueError("bivector basis value and derivative counts differ")
    tensors = (field_values, field_jacobians, basis.values, basis.derivatives)
    if len({tensor.device for tensor in tensors}) != 1 or len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all determining-operator tensors must share dtype and device")

    transport = torch.einsum("nk,nmijk->nmij", field_values, basis.derivatives)
    deform_first = torch.einsum("nik,nmkj->nmij", field_jacobians, basis.values)
    deform_second = torch.einsum("njk,nmik->nmij", field_jacobians, basis.values)
    lie_derivatives = transport - deform_first - deform_second
    independent = torch.stack(
        [lie_derivatives[:, :, i, j] for i, j in basis.pairs],
        dim=1,
    )
    return independent.reshape(n_samples * len(basis.pairs), basis.values.shape[1])


def build_poisson_determining_matrix_from_scalar(
    field_values: torch.Tensor,
    field_jacobians: torch.Tensor,
    scalar_basis: ScalarBasisEvaluation,
    *,
    state_dim: int,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Memory-efficient Poisson determining matrix from scalar basis values.

    This is algebraically identical to :func:`build_poisson_determining_matrix`
    but avoids arrays of shape ``(N,M,d,d,d)``.  Its largest result is the
    determining matrix itself, ``(N*P, P*K)``, where ``P=d*(d-1)//2``.
    """

    dimension = int(state_dim)
    pair_order = upper_triangle_pairs(dimension) if pairs is None else tuple(pairs)
    if not isinstance(field_values, torch.Tensor) or not isinstance(field_jacobians, torch.Tensor):
        raise TypeError("field values and Jacobians must be torch tensors")
    if field_values.ndim != 2 or tuple(field_values.shape[1:]) != (dimension,):
        raise ValueError("field_values must have shape (N,state_dim)")
    n_samples = int(field_values.shape[0])
    if tuple(field_jacobians.shape) != (n_samples, dimension, dimension):
        raise ValueError("field_jacobians must have shape (N,state_dim,state_dim)")
    if scalar_basis.values.ndim != 2 or scalar_basis.values.shape[0] != n_samples:
        raise ValueError("scalar basis values must have shape (N,K)")
    scalar_size = int(scalar_basis.values.shape[1])
    if tuple(scalar_basis.gradients.shape) != (n_samples, scalar_size, dimension):
        raise ValueError("scalar basis gradients must have shape (N,K,state_dim)")
    tensors = (
        field_values,
        field_jacobians,
        scalar_basis.values,
        scalar_basis.gradients,
    )
    if len({tensor.device for tensor in tensors}) != 1 or len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all determining-operator tensors must share dtype and device")

    wedges = _wedge_matrices(pair_order, dimension, reference=field_values)
    transport_scalar = torch.einsum("nk,nak->na", field_values, scalar_basis.gradients)
    deformation_rows: list[torch.Tensor] = []
    for i, j in pair_order:
        first = torch.einsum("nk,pk->np", field_jacobians[:, i, :], wedges[:, :, j])
        second = torch.einsum("nk,pk->np", field_jacobians[:, j, :], wedges[:, i, :])
        deformation_rows.append(first + second)
    deformation = torch.stack(deformation_rows, dim=1)  # (N, output_pair, input_pair)
    pair_identity = torch.eye(
        len(pair_order),
        device=field_values.device,
        dtype=field_values.dtype,
    )
    columns = (
        transport_scalar[:, None, None, :] * pair_identity[None, :, :, None]
        - deformation[:, :, :, None] * scalar_basis.values[:, None, None, :]
    )
    return columns.reshape(n_samples * len(pair_order), len(pair_order) * scalar_size)


__all__ = [
    "BivectorBasis",
    "BivectorBasisEvaluation",
    "BivectorEvaluation",
    "CallableScalarBasis",
    "PolynomialScalarBasis",
    "ScalarBasis",
    "ScalarBasisEvaluation",
    "assemble_skew_tensor",
    "build_poisson_determining_matrix",
    "build_poisson_determining_matrix_from_scalar",
    "extract_upper_triangle",
    "polynomial_exponents",
    "upper_triangle_pairs",
]
