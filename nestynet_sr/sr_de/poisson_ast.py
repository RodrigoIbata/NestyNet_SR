# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

"""AST adapters for symbolic Poisson tensor bases.

The numerical Poisson search only needs scalar basis values and first
derivatives.  This module connects that small interface to NestyNet_SR's AST
without compiling trainable leaves or invoking autograd.  It also converts a
polynomial bivector coefficient vector back to explicit upper-triangular AST
entries for reporting and downstream symbolic manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import (
    Add,
    ConstNode,
    Mul,
    Node,
    Pow,
    Var,
    _collect_var_idxs_from_node,
    _eval_single_input,
)

from .poisson_basis import (
    BivectorEvaluation,
    PolynomialScalarBasis,
    ScalarBasisEvaluation,
    upper_triangle_pairs,
)
from .poisson_darboux import ScalarMapComponent, TriangularDarbouxMap


@dataclass(frozen=True)
class ASTScalarBasis:
    """A finite scalar basis represented by pure input AST expressions."""

    state_dim: int
    terms: tuple[Node, ...]

    def __init__(self, state_dim: int, terms: Sequence[Node]):
        object.__setattr__(self, "state_dim", int(state_dim))
        object.__setattr__(self, "terms", tuple(terms))
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not self.terms:
            raise ValueError("ASTScalarBasis needs at least one term")

    @property
    def size(self) -> int:
        return len(self.terms)

    def evaluate(self, z: torch.Tensor) -> ScalarBasisEvaluation:
        """Evaluate all terms and analytic state gradients at ``z``."""

        _validate_state_points(z, self.state_dim)
        values = []
        gradients = []
        for term in self.terms:
            value, grad, _ = _eval_single_input(term, z, need_grad=True, need_hess=False)
            if grad is None:
                raise RuntimeError("AST evaluator did not return a requested gradient")
            values.append(value.reshape(z.shape[0], 1))
            gradients.append(grad.reshape(z.shape[0], 1, self.state_dim))
        return ScalarBasisEvaluation(
            values=torch.cat(values, dim=1),
            gradients=torch.cat(gradients, dim=1),
        )


@dataclass(frozen=True)
class ASTPoissonTensor:
    """Explicit skew tensor whose independent entries are scalar ASTs.

    Missing upper-triangular entries are interpreted as zero.  Entries supplied
    with lower-triangular keys are rejected so that antisymmetry has one clear
    source of truth.
    """

    state_dim: int
    entries: Mapping[tuple[int, int], Node]

    def __post_init__(self) -> None:
        d = int(self.state_dim)
        if d <= 0:
            raise ValueError("state_dim must be positive")
        valid = set(upper_triangle_pairs(d))
        bad = [pair for pair in self.entries if pair not in valid]
        if bad:
            raise ValueError(f"Poisson AST entries must use i < j; invalid keys: {bad}")

    def evaluate(self, z: torch.Tensor) -> BivectorEvaluation:
        """Return ``Pi`` and its analytic coordinate derivatives."""

        d = int(self.state_dim)
        _validate_state_points(z, d)
        n = int(z.shape[0])
        tensor = z.new_zeros((n, d, d))
        derivatives = z.new_zeros((n, d, d, d))
        for (i, j), term in self.entries.items():
            value, grad, _ = _eval_single_input(term, z, need_grad=True, need_hess=False)
            if grad is None:
                raise RuntimeError("AST evaluator did not return a requested gradient")
            scalar = value[:, 0]
            scalar_grad = grad[:, 0, :]
            tensor[:, i, j] = scalar
            tensor[:, j, i] = -scalar
            derivatives[:, i, j, :] = scalar_grad
            derivatives[:, j, i, :] = -scalar_grad
        return BivectorEvaluation(tensor=tensor, derivatives=derivatives)


def polynomial_bivector_to_ast(
    coefficients: torch.Tensor | Sequence[float],
    basis: PolynomialScalarBasis,
    *,
    zero_tolerance: float = 0.0,
) -> ASTPoissonTensor:
    """Compile pair-major polynomial coefficients into explicit AST entries."""

    coeff = torch.as_tensor(coefficients, dtype=torch.float64).detach().cpu().reshape(-1)
    pairs = upper_triangle_pairs(int(basis.state_dim))
    expected = len(pairs) * len(basis.exponents)
    if int(coeff.numel()) != expected:
        raise ValueError(f"expected {expected} coefficients, got {int(coeff.numel())}")

    entries: dict[tuple[int, int], Node] = {}
    k = 0
    for pair in pairs:
        entry: Node | None = None
        for exponent in basis.exponents:
            value = float(coeff[k])
            k += 1
            if abs(value) <= float(zero_tolerance):
                continue
            term = _monomial_ast(exponent)
            if value != 1.0:
                term = Mul(ConstNode(value), term)
            entry = term if entry is None else Add(entry, term)
        if entry is not None:
            entries[pair] = entry
    return ASTPoissonTensor(state_dim=int(basis.state_dim), entries=entries)


def ast_triangular_darboux_map(
    components: Sequence[Node],
    *,
    name: str = "ast_triangular_map",
    complexities: Sequence[float] | None = None,
) -> TriangularDarbouxMap:
    """Adapt pure scalar AST components to a certified triangular chart.

    Variable support is read from each AST.  Component ``i`` is rejected unless
    it depends only on coordinates ``0..i``; the returned Darboux adapter checks
    the same condition again on its analytic gradients.
    """

    terms = tuple(components)
    if not terms:
        raise ValueError("a Darboux map needs at least one AST component")
    if complexities is None:
        costs = tuple(1.0 for _ in terms)
    else:
        costs = tuple(float(value) for value in complexities)
        if len(costs) != len(terms):
            raise ValueError("complexities must match the AST component count")

    adapted: list[ScalarMapComponent] = []
    for index, (term, complexity) in enumerate(zip(terms, costs)):
        support = _collect_var_idxs_from_node(term)
        max_input_index = max(support, default=0)
        if max_input_index > index:
            raise ValueError(
                f"AST component {index} depends on coordinate {max_input_index}; "
                "the map is not lower triangular"
            )

        def evaluate_value(z: np.ndarray, expression: Node = term) -> np.ndarray:
            points = torch.as_tensor(z, dtype=torch.float64)
            value, _, _ = _eval_single_input(
                expression, points, need_grad=False, need_hess=False
            )
            return value[:, 0].detach().cpu().numpy()

        def evaluate_gradient(z: np.ndarray, expression: Node = term) -> np.ndarray:
            points = torch.as_tensor(z, dtype=torch.float64)
            _, gradient, _ = _eval_single_input(
                expression, points, need_grad=True, need_hess=False
            )
            if gradient is None:
                raise RuntimeError("AST evaluator did not return a requested gradient")
            return gradient[:, 0, :].detach().cpu().numpy()

        adapted.append(
            ScalarMapComponent(
                value_function=evaluate_value,
                gradient_function=evaluate_gradient,
                name=repr(term),
                max_input_index=max_input_index,
                complexity=complexity,
            )
        )
    return TriangularDarbouxMap(components=tuple(adapted), name=str(name))


def _monomial_ast(exponent: Sequence[int]) -> Node:
    factors: list[Node] = []
    for axis, power_raw in enumerate(exponent):
        power = int(power_raw)
        if power < 0:
            raise ValueError("polynomial exponents must be non-negative")
        if power == 0:
            continue
        factor: Node = Var(axis)
        if power != 1:
            factor = Pow(factor, float(power))
        factors.append(factor)
    if not factors:
        return ConstNode(1.0)
    term = factors[0]
    for factor in factors[1:]:
        term = Mul(term, factor)
    return term


def _validate_state_points(z: torch.Tensor, state_dim: int) -> None:
    if not isinstance(z, torch.Tensor):
        raise TypeError("state points must be a torch.Tensor")
    if z.ndim != 2 or int(z.shape[1]) != int(state_dim):
        raise ValueError(f"expected state points of shape (N, {state_dim}), got {tuple(z.shape)}")
    if not torch.is_floating_point(z):
        raise TypeError("state points must use a floating dtype")


__all__ = [
    "ASTPoissonTensor",
    "ASTScalarBasis",
    "ast_triangular_darboux_map",
    "polynomial_bivector_to_ast",
]
