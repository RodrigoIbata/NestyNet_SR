# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Bounded symbolic invariant compiler for scalar point generators.

The compiler deliberately solves a finite feature problem.  Given point
generators ``X_a = xi_a(x,u) d_x + eta_a(x,u) d_u`` and a caller-provided AST
vocabulary ``phi_j``, it constructs the joint action matrix

    A[(a,n), j] = X_a phi_j(x_n, u_n)

and searches the nullspace of ``A`` for sparse, certified invariants.  It does
not launch a neural or unconstrained symbolic search.  The same analytic AST
evaluator is exposed through :class:`SymbolicInvariantObjective`, allowing an
external factorized symbolic search to score an arbitrary candidate AST.

The implementation is intentionally restricted to point coordinates ``(x,u)``.
Jet-space differential invariants use the same idea with prolonged generators,
but belong in a separate compiler once their feature and certification APIs are
stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from numbers import Number
from typing import Any, Mapping, Sequence

import torch

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    Add,
    AddNode,
    AcosNode,
    ArgNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    Exp,
    ExpNode,
    ImagNode,
    Log,
    LogNode,
    Mul,
    MulNode,
    Pow,
    PowNode,
    RealNode,
    SinNode,
    U,
    Var,
    _eval_single_input,
    ast_to_human_readable,
)


NodeLike = Any
_EPS = 1.0e-15


def point_coordinate_ast_to_de_ast(ast: NodeLike, *, x_axis: int = 0) -> NodeLike:
    """Translate a carrier AST from coordinate ``(x,u)`` to DE AST atoms.

    The invariant compiler evaluates point coordinates as a two-column tensor,
    so ``Var(0)`` means ``x`` and ``Var(1)`` means ``u``.  DE libraries instead
    represent the dependent field with ``U()``.  Keeping this conversion at
    the bridge boundary prevents an invariant such as ``u/x`` from becoming
    an invalid request for a second independent input axis.
    """

    if isinstance(ast, AtomNode):
        kind = str(getattr(ast, "kind", "")).lower()
        if kind not in {"var", "x", "input"} or len(ast.var_idxs) != 1:
            return ast
        coordinate = int(ast.var_idxs[0])
        if coordinate == 0:
            return Var(int(x_axis))
        if coordinate == 1:
            return U()
        raise ValueError(f"point carrier references unsupported coordinate Var({coordinate})")
    if isinstance(ast, ConstNode):
        return ast
    if isinstance(ast, AddNode):
        return Add(
            point_coordinate_ast_to_de_ast(ast.left, x_axis=x_axis),
            point_coordinate_ast_to_de_ast(ast.right, x_axis=x_axis),
        )
    if isinstance(ast, MulNode):
        return Mul(
            point_coordinate_ast_to_de_ast(ast.left, x_axis=x_axis),
            point_coordinate_ast_to_de_ast(ast.right, x_axis=x_axis),
        )
    if isinstance(ast, PowNode):
        exponent = ast.exponent
        if not isinstance(exponent, Number):
            exponent = point_coordinate_ast_to_de_ast(exponent, x_axis=x_axis)
        return Pow(point_coordinate_ast_to_de_ast(ast.base, x_axis=x_axis), exponent)
    unary = (
        LogNode,
        ExpNode,
        SinNode,
        CosNode,
        AsinNode,
        AcosNode,
        AtanNode,
        ConjNode,
        RealNode,
        ImagNode,
        AbsNode,
        ArgNode,
    )
    if isinstance(ast, unary):
        return type(ast)(point_coordinate_ast_to_de_ast(ast.arg, x_axis=x_axis))
    raise TypeError(f"unsupported point-carrier AST node {type(ast).__name__}")


@dataclass(frozen=True)
class InvariantCompilerConfig:
    """Numerical and complexity gates for bounded invariant compilation."""

    rank_rtol: float = 1.0e-9
    rank_atol: float = 1.0e-11
    action_rtol: float = 1.0e-7
    action_atol: float = 1.0e-9
    orbit_rtol: float = 1.0e-7
    orbit_atol: float = 1.0e-9
    min_variance: float = 1.0e-10
    min_gradient_rms: float = 1.0e-8
    independence_rank_rtol: float = 1.0e-7
    independence_rank_atol: float = 1.0e-9
    min_independent_fraction: float = 0.8
    coefficient_tol: float = 1.0e-9
    sparse_threshold: float = 0.08
    sparse_iterations: int = 12
    max_sparse_seeds: int = 64
    max_candidates: int = 32
    max_invariants: int = 2
    max_orbit_support: int = 4
    max_polynomial_degree: int = 3
    objective_action_weight: float = 1.0
    objective_variance_weight: float = 1.0
    objective_independence_weight: float = 1.0
    objective_domain_penalty: float = 1.0e6


@dataclass(frozen=True)
class InvariantCandidateResult:
    """One explicit AST invariant and its train/held-out certificates."""

    ast: NodeLike
    coefficients: tuple[float, ...]
    support: tuple[int, ...]
    support_terms: tuple[str, ...]
    train_action_rms: float
    train_action_relative: float
    validation_action_rms: float
    validation_action_relative: float
    train_variance: float
    validation_variance: float
    train_gradient_rms: float
    validation_gradient_rms: float
    finite_train_fraction: float
    finite_validation_fraction: float
    independence_fraction: float
    independent_rank: int
    accepted: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "ast": repr(self.ast),
            "human": _human(self.ast),
            "coefficients": [float(v) for v in self.coefficients],
            "support": [int(v) for v in self.support],
            "support_terms": list(self.support_terms),
            "train_action_rms": float(self.train_action_rms),
            "train_action_relative": float(self.train_action_relative),
            "validation_action_rms": float(self.validation_action_rms),
            "validation_action_relative": float(self.validation_action_relative),
            "train_variance": float(self.train_variance),
            "validation_variance": float(self.validation_variance),
            "train_gradient_rms": float(self.train_gradient_rms),
            "validation_gradient_rms": float(self.validation_gradient_rms),
            "finite_train_fraction": float(self.finite_train_fraction),
            "finite_validation_fraction": float(self.finite_validation_fraction),
            "independence_fraction": float(self.independence_fraction),
            "independent_rank": int(self.independent_rank),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class InvariantCompilationResult:
    """Certified sparse representatives of a joint invariant subspace."""

    status: str
    invariants: tuple[InvariantCandidateResult, ...]
    candidates: tuple[InvariantCandidateResult, ...]
    vocabulary: tuple[NodeLike, ...]
    active_vocabulary_indices: tuple[int, ...]
    discarded_terms: tuple[dict[str, Any], ...]
    determining_rank: int
    determining_nullity: int
    singular_values: tuple[float, ...]
    nullspace_basis: torch.Tensor
    nullspace_projector: torch.Tensor
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "invariants": [row.to_report() for row in self.invariants],
            "candidates": [row.to_report() for row in self.candidates],
            "vocabulary": [repr(term) for term in self.vocabulary],
            "active_vocabulary_indices": [int(v) for v in self.active_vocabulary_indices],
            "discarded_terms": [dict(row) for row in self.discarded_terms],
            "determining_rank": int(self.determining_rank),
            "determining_nullity": int(self.determining_nullity),
            "singular_values": [float(v) for v in self.singular_values],
            "nullspace_basis_shape": list(self.nullspace_basis.shape),
            "nullspace_projector_shape": list(self.nullspace_projector.shape),
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class OrbitCoordinateResult:
    """A bounded-vocabulary coordinate satisfying ``X s ~= 1``."""

    status: str
    ast: NodeLike | None
    coefficients: tuple[float, ...]
    support: tuple[int, ...]
    train_residual_rms: float
    train_residual_relative: float
    validation_residual_rms: float
    validation_residual_relative: float
    train_variance: float
    validation_variance: float
    finite_train_fraction: float
    finite_validation_fraction: float
    accepted: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ast": None if self.ast is None else repr(self.ast),
            "human": None if self.ast is None else _human(self.ast),
            "coefficients": [float(v) for v in self.coefficients],
            "support": [int(v) for v in self.support],
            "train_residual_rms": float(self.train_residual_rms),
            "train_residual_relative": float(self.train_residual_relative),
            "validation_residual_rms": float(self.validation_residual_rms),
            "validation_residual_relative": float(self.validation_residual_relative),
            "train_variance": float(self.train_variance),
            "validation_variance": float(self.validation_variance),
            "finite_train_fraction": float(self.finite_train_fraction),
            "finite_validation_fraction": float(self.finite_validation_fraction),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class SubalgebraInvariantCompilation:
    """Certified carriers collected from simple recovered subalgebras.

    A full Lie algebra can have no nonconstant common point invariant even
    when one of its one-generator subalgebras exposes a very useful carrier.
    The automatic GS escalation therefore compiles every accepted singleton
    and, for comparison, the full recovered algebra.  Deduplicated accepted
    invariants and rectifying coordinates are exposed through the same small
    attribute contract consumed by the DE-library and FSS bridges.
    """

    invariants: tuple[InvariantCandidateResult, ...]
    orbit_coordinates: tuple[OrbitCoordinateResult, ...]
    subalgebra_results: tuple[InvariantCompilationResult, ...]
    subalgebra_generator_indices: tuple[tuple[int, ...], ...]
    reason: str

    @property
    def status(self) -> str:
        return "recovered" if self.invariants or self.orbit_coordinates else "rejected"

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "invariants": [row.to_report() for row in self.invariants],
            "orbit_coordinates": [row.to_report() for row in self.orbit_coordinates],
            "subalgebras": [
                {
                    "generator_indices": [int(v) for v in indices],
                    "compilation": result.to_report(),
                }
                for indices, result in zip(
                    self.subalgebra_generator_indices, self.subalgebra_results
                )
            ],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InvariantObjectiveReport:
    """Score components returned by :class:`SymbolicInvariantObjective`."""

    total: float
    action_loss: float
    variance_penalty: float
    independence_penalty: float
    domain_penalty: float
    variance: float
    gradient_rms: float
    independence_fraction: float
    finite_fraction: float

    def to_report(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def default_point_candidate_vocabulary(max_polynomial_degree: int = 3) -> tuple[NodeLike, ...]:
    """Return a compact point-coordinate vocabulary for ``(x,u)``.

    The rational ratio terms expose projective/Mobius carriers.  Logarithmic
    and exponential terms use globally real, overflow-resistant forms instead
    of bare ``log(x)`` or ``exp(x)``.  Reciprocal terms remain patch-local and
    are removed automatically when either sample set touches their singularity.
    """

    x = Var(0)
    u = Var(1)
    one = ConstNode(1.0)
    minus_one = ConstNode(-1.0)
    x2 = Pow(x, 2)
    u2 = Pow(u, 2)
    polynomial: list[NodeLike] = []
    for total in range(1, max(0, int(max_polynomial_degree)) + 1):
        for px in range(total, -1, -1):
            pu = total - px
            x_factor = None if px == 0 else (x if px == 1 else Pow(x, px))
            u_factor = None if pu == 0 else (u if pu == 1 else Pow(u, pu))
            if x_factor is None:
                term = u_factor
            elif u_factor is None:
                term = x_factor
            else:
                term = Mul(x_factor, u_factor)
            assert term is not None
            polynomial.append(term)
    typed = (
        Pow(x, -1),
        Pow(u, -1),
        Mul(u, Pow(x, -1)),
        Mul(x, Pow(u, -1)),
        Mul(x, Pow(Add(one, x2), -1)),
        Mul(u, Pow(Add(one, u2), -1)),
        Log(Add(one, x2)),
        Log(Add(one, u2)),
        Exp(Mul(minus_one, x2)),
        Exp(Mul(minus_one, u2)),
    )
    return _unique_asts((*polynomial, *typed))


def compile_point_invariants(
    generators: Sequence[Any],
    train_points: Any,
    validation_points: Any,
    candidate_asts: Sequence[NodeLike] | None = None,
    config: InvariantCompilerConfig | None = None,
) -> InvariantCompilationResult:
    """Recover sparse AST invariants common to all supplied point generators."""

    cfg = config or InvariantCompilerConfig()
    gens = tuple(generators)
    if not gens:
        raise ValueError("at least one point generator is required")
    train = _coerce_points(train_points)
    validation = _coerce_points(validation_points, like=train)
    vocabulary = _unique_asts(candidate_asts or default_point_candidate_vocabulary(cfg.max_polynomial_degree))
    if not vocabulary:
        raise ValueError("candidate_asts must contain at least one AST")

    active: list[int] = []
    discarded: list[dict[str, Any]] = []
    action_columns: list[torch.Tensor] = []
    for idx, term in enumerate(vocabulary):
        train_eval = _evaluate_ast(term, train)
        validation_eval = _evaluate_ast(term, validation)
        reason = _individual_term_rejection(train_eval, validation_eval, cfg)
        if reason:
            discarded.append({"index": int(idx), "ast": repr(term), "reason": reason})
            continue
        active.append(idx)
        action_columns.append(_joint_actions(gens, train, train_eval.gradient).reshape(-1))

    if not active:
        empty = torch.empty((0, 0), dtype=train.dtype, device=train.device)
        return InvariantCompilationResult(
            status="rejected",
            invariants=(),
            candidates=(),
            vocabulary=vocabulary,
            active_vocabulary_indices=(),
            discarded_terms=tuple(discarded),
            determining_rank=0,
            determining_nullity=0,
            singular_values=(),
            nullspace_basis=empty,
            nullspace_projector=empty,
            reason="no_finite_noncollapsed_candidate_terms",
        )

    action_matrix = torch.stack(action_columns, dim=1)
    basis, projector, singular_values, rank = _stable_nullspace(action_matrix, cfg)
    nullity = int(basis.shape[1])
    if nullity == 0:
        return InvariantCompilationResult(
            status="rejected",
            invariants=(),
            candidates=(),
            vocabulary=vocabulary,
            active_vocabulary_indices=tuple(active),
            discarded_terms=tuple(discarded),
            determining_rank=int(rank),
            determining_nullity=0,
            singular_values=tuple(float(v) for v in singular_values),
            nullspace_basis=basis,
            nullspace_projector=projector,
            reason="joint_action_operator_has_trivial_nullspace",
        )

    active_terms = tuple(vocabulary[j] for j in active)
    directions = _sparse_nullspace_representatives(basis, projector, cfg)
    candidates: list[InvariantCandidateResult] = []
    for coeffs in directions[: max(0, int(cfg.max_candidates))]:
        ast, pruned, support = _compile_linear_combination(active_terms, coeffs, cfg)
        if ast is None:
            continue
        candidates.append(
            _certify_invariant(
                ast,
                pruned,
                support,
                active_terms,
                gens,
                train,
                validation,
                cfg,
            )
        )

    candidates.sort(key=_candidate_sort_key)
    accepted: list[InvariantCandidateResult] = []
    accepted_gradients: list[torch.Tensor] = []
    certified_rows: list[InvariantCandidateResult] = []
    for row in candidates:
        if not row.accepted:
            certified_rows.append(row)
            continue
        gradient = _evaluate_ast(row.ast, validation).gradient
        independence_fraction, independent_rank = _independence_certificate(
            accepted_gradients,
            gradient,
            cfg,
        )
        independent = independence_fraction >= float(cfg.min_independent_fraction)
        within_limit = len(accepted) < max(0, int(cfg.max_invariants))
        selected = bool(independent and within_limit)
        if selected:
            reason = "accepted"
        elif not independent:
            reason = "functionally_dependent_on_selected_invariants"
        else:
            reason = "invariant_selection_limit_reached"
        updated = replace(
            row,
            independence_fraction=float(independence_fraction),
            independent_rank=int(independent_rank),
            accepted=selected,
            reason=reason,
        )
        certified_rows.append(updated)
        if selected:
            accepted.append(updated)
            accepted_gradients.append(gradient)

    reason = "accepted_certified_invariants" if accepted else "no_candidate_passed_heldout_and_independence_gates"
    return InvariantCompilationResult(
        status="recovered" if accepted else "rejected",
        invariants=tuple(accepted),
        candidates=tuple(certified_rows),
        vocabulary=vocabulary,
        active_vocabulary_indices=tuple(active),
        discarded_terms=tuple(discarded),
        determining_rank=int(rank),
        determining_nullity=int(nullity),
        singular_values=tuple(float(v) for v in singular_values),
        nullspace_basis=basis,
        nullspace_projector=projector,
        reason=reason,
        evidence={
            "operator": "stack_a X_a(phi_j)",
            "train_points": int(train.shape[0]),
            "validation_points": int(validation.shape[0]),
            "generator_count": len(gens),
            "subspace_first": True,
        },
    )


def compile_orbit_coordinate(
    generator: Any,
    train_points: Any,
    validation_points: Any,
    candidate_asts: Sequence[NodeLike] | None = None,
    config: InvariantCompilerConfig | None = None,
) -> OrbitCoordinateResult:
    """Find a sparse bounded-vocabulary coordinate satisfying ``X s ~= 1``."""

    cfg = config or InvariantCompilerConfig()
    train = _coerce_points(train_points)
    validation = _coerce_points(validation_points, like=train)
    vocabulary = _unique_asts(candidate_asts or default_point_candidate_vocabulary(cfg.max_polynomial_degree))
    active_terms: list[NodeLike] = []
    train_columns: list[torch.Tensor] = []
    validation_columns: list[torch.Tensor] = []
    original_indices: list[int] = []
    for idx, term in enumerate(vocabulary):
        train_eval = _evaluate_ast(term, train)
        validation_eval = _evaluate_ast(term, validation)
        if _individual_term_rejection(train_eval, validation_eval, cfg):
            continue
        active_terms.append(term)
        original_indices.append(idx)
        train_columns.append(_joint_actions((generator,), train, train_eval.gradient).reshape(-1))
        validation_columns.append(_joint_actions((generator,), validation, validation_eval.gradient).reshape(-1))

    if not active_terms:
        return _empty_orbit("no_finite_noncollapsed_candidate_terms")
    matrix = torch.stack(train_columns, dim=1)
    validation_matrix = torch.stack(validation_columns, dim=1)
    target = torch.ones(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    validation_target = torch.ones(validation_matrix.shape[0], dtype=validation_matrix.dtype, device=validation_matrix.device)
    directions = _orbit_candidate_directions(matrix, target, cfg)
    rows: list[tuple[tuple[Any, ...], OrbitCoordinateResult]] = []
    for coeffs in directions:
        ast, pruned, support = _compile_linear_combination(tuple(active_terms), coeffs, cfg, normalize=False)
        if ast is None:
            continue
        train_residual = matrix @ pruned - target
        validation_residual = validation_matrix @ pruned - validation_target
        train_rms = _rms(train_residual)
        validation_rms = _rms(validation_residual)
        train_rel = train_rms  # target RMS is exactly one
        validation_rel = validation_rms
        train_eval = _evaluate_ast(ast, train)
        validation_eval = _evaluate_ast(ast, validation)
        finite_train = train_eval.finite_fraction
        finite_validation = validation_eval.finite_fraction
        train_variance = _variance(train_eval.value)
        validation_variance = _variance(validation_eval.value)
        finite = finite_train == 1.0 and finite_validation == 1.0
        noncollapsed = min(train_variance, validation_variance) >= float(cfg.min_variance)
        fit_ok = (train_rms <= cfg.orbit_atol or train_rel <= cfg.orbit_rtol) and (
            validation_rms <= cfg.orbit_atol or validation_rel <= cfg.orbit_rtol
        )
        accepted = bool(finite and noncollapsed and fit_ok)
        if not finite:
            reason = "nonfinite_domain"
        elif not noncollapsed:
            reason = "collapsed_coordinate"
        elif not fit_ok:
            reason = "heldout_orbit_equation_failed"
        else:
            reason = "accepted"
        mapped_support = tuple(original_indices[j] for j in support)
        expanded_coefficients = torch.zeros(len(vocabulary), dtype=pruned.dtype, device=pruned.device)
        expanded_coefficients[torch.as_tensor(original_indices, dtype=torch.long, device=pruned.device)] = pruned
        result = OrbitCoordinateResult(
            status="recovered" if accepted else "rejected",
            ast=ast,
            coefficients=tuple(float(v) for v in expanded_coefficients),
            support=mapped_support,
            train_residual_rms=float(train_rms),
            train_residual_relative=float(train_rel),
            validation_residual_rms=float(validation_rms),
            validation_residual_relative=float(validation_rel),
            train_variance=float(train_variance),
            validation_variance=float(validation_variance),
            finite_train_fraction=float(finite_train),
            finite_validation_fraction=float(finite_validation),
            accepted=accepted,
            reason=reason,
            evidence={"equation": "X(s)=1", "support_terms": [repr(active_terms[j]) for j in support]},
        )
        key = (0 if accepted else 1, len(support), float(validation_rel), float(train_rel))
        rows.append((key, result))
    if not rows:
        return _empty_orbit("orbit_least_squares_produced_no_nonzero_candidate")
    rows.sort(key=lambda item: item[0])
    return rows[0][1]


def compile_subalgebra_invariants(
    generators: Sequence[Any],
    train_points: Any,
    validation_points: Any,
    candidate_asts: Sequence[NodeLike] | None = None,
    config: InvariantCompilerConfig | None = None,
    *,
    include_full_algebra: bool = True,
    include_orbit_coordinates: bool = True,
) -> SubalgebraInvariantCompilation:
    """Compile carriers from singleton generators and the full algebra.

    Singleton compilation is the bounded alternative to enumerating every
    subalgebra.  It captures the most useful projective/scaling reductions,
    keeps runtime linear in the recovered nullity, and avoids the common
    failure mode where intersecting all generator invariants leaves only
    constants.
    """

    gens = tuple(generators)
    if not gens:
        return SubalgebraInvariantCompilation(
            invariants=(),
            orbit_coordinates=(),
            subalgebra_results=(),
            subalgebra_generator_indices=(),
            reason="no_accepted_generators",
        )
    cfg = config or InvariantCompilerConfig()
    index_sets: list[tuple[int, ...]] = [(index,) for index in range(len(gens))]
    full = tuple(range(len(gens)))
    if include_full_algebra and len(gens) > 1:
        index_sets.append(full)

    results: list[InvariantCompilationResult] = []
    invariant_rows: list[InvariantCandidateResult] = []
    orbit_rows: list[OrbitCoordinateResult] = []
    seen_invariants: set[str] = set()
    seen_orbits: set[str] = set()
    for indices in index_sets:
        selected = tuple(gens[index] for index in indices)
        result = compile_point_invariants(
            selected,
            train_points,
            validation_points,
            candidate_asts=candidate_asts,
            config=cfg,
        )
        results.append(result)
        for row in result.invariants:
            key = repr(row.ast)
            if key not in seen_invariants:
                seen_invariants.add(key)
                invariant_rows.append(row)
        if include_orbit_coordinates and len(indices) == 1:
            orbit = compile_orbit_coordinate(
                selected[0],
                train_points,
                validation_points,
                candidate_asts=candidate_asts,
                config=cfg,
            )
            if orbit.accepted and orbit.ast is not None:
                key = repr(orbit.ast)
                if key not in seen_orbits:
                    seen_orbits.add(key)
                    orbit_rows.append(orbit)

    limit = max(0, int(cfg.max_invariants))
    invariant_rows.sort(key=_candidate_sort_key)
    orbit_rows.sort(
        key=lambda row: (
            len(row.support),
            float(row.validation_residual_relative),
            repr(row.ast),
        )
    )
    return SubalgebraInvariantCompilation(
        invariants=tuple(invariant_rows[:limit]),
        orbit_coordinates=tuple(orbit_rows[: max(0, int(cfg.max_candidates))]),
        subalgebra_results=tuple(results),
        subalgebra_generator_indices=tuple(index_sets),
        reason=(
            "accepted_subalgebra_carriers"
            if invariant_rows or orbit_rows
            else "no_subalgebra_carrier_passed_certification"
        ),
    )


class SymbolicInvariantObjective:
    """Callable bounded objective for external factorized symbolic search.

    ``__call__`` returns a scalar tensor so callers can use this instance as a
    loss adapter.  ``evaluate`` returns all diagnostic components.  Candidate
    AST construction remains the responsibility of the caller.
    """

    def __init__(
        self,
        generators: Sequence[Any],
        points: Any,
        reference_invariants: Sequence[NodeLike] = (),
        config: InvariantCompilerConfig | None = None,
    ) -> None:
        self.generators = tuple(generators)
        if not self.generators:
            raise ValueError("at least one point generator is required")
        self.points = _coerce_points(points)
        self.reference_invariants = tuple(reference_invariants)
        self.config = config or InvariantCompilerConfig()
        self._reference_gradients = tuple(_evaluate_ast(ast, self.points).gradient for ast in self.reference_invariants)

    def __call__(self, candidate_ast: NodeLike) -> torch.Tensor:
        report = self.evaluate(candidate_ast)
        return torch.as_tensor(report.total, dtype=self.points.dtype, device=self.points.device)

    def evaluate(self, candidate_ast: NodeLike) -> InvariantObjectiveReport:
        cfg = self.config
        evaluated = _evaluate_ast(candidate_ast, self.points)
        if evaluated.finite_fraction < 1.0:
            return InvariantObjectiveReport(
                total=float(cfg.objective_domain_penalty),
                action_loss=float(cfg.objective_domain_penalty),
                variance_penalty=1.0,
                independence_penalty=1.0,
                domain_penalty=float(cfg.objective_domain_penalty),
                variance=0.0,
                gradient_rms=0.0,
                independence_fraction=0.0,
                finite_fraction=float(evaluated.finite_fraction),
            )
        actions = _joint_actions(self.generators, self.points, evaluated.gradient)
        action_rms, action_relative = _action_metrics(actions, self.generators, self.points, evaluated.gradient)
        action_loss = min(float(action_relative), float(action_rms / max(cfg.action_atol, _EPS))) ** 2
        variance = _variance(evaluated.value)
        gradient_rms = _gradient_rms(evaluated.gradient)
        variance_ratio = min(1.0, variance / max(float(cfg.min_variance), _EPS))
        gradient_ratio = min(1.0, gradient_rms / max(float(cfg.min_gradient_rms), _EPS))
        variance_penalty = 1.0 - min(variance_ratio, gradient_ratio)
        independence_fraction, _rank = _independence_certificate(
            list(self._reference_gradients),
            evaluated.gradient,
            cfg,
        )
        independence_penalty = max(0.0, 1.0 - independence_fraction)
        total = (
            float(cfg.objective_action_weight) * action_loss
            + float(cfg.objective_variance_weight) * variance_penalty
            + float(cfg.objective_independence_weight) * independence_penalty
        )
        return InvariantObjectiveReport(
            total=float(total),
            action_loss=float(action_loss),
            variance_penalty=float(variance_penalty),
            independence_penalty=float(independence_penalty),
            domain_penalty=0.0,
            variance=float(variance),
            gradient_rms=float(gradient_rms),
            independence_fraction=float(independence_fraction),
            finite_fraction=float(evaluated.finite_fraction),
        )


@dataclass(frozen=True)
class _ASTEvaluation:
    value: torch.Tensor
    gradient: torch.Tensor
    finite_fraction: float


def _coerce_points(points: Any, *, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(points, Mapping):
        x = torch.as_tensor(points["x"])
        u = torch.as_tensor(points["u"])
        tensor = torch.stack((x.reshape(-1), u.reshape(-1)), dim=1)
    elif isinstance(points, (tuple, list)) and len(points) == 2 and not isinstance(points, torch.Tensor):
        x = torch.as_tensor(points[0])
        u = torch.as_tensor(points[1])
        tensor = torch.stack((x.reshape(-1), u.reshape(-1)), dim=1)
    else:
        tensor = torch.as_tensor(points)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(f"point samples must have shape (N,2); got {tuple(tensor.shape)}")
    if tensor.shape[0] < 2:
        raise ValueError("at least two point samples are required")
    if like is not None:
        tensor = tensor.to(dtype=like.dtype, device=like.device)
    elif not tensor.dtype.is_floating_point:
        tensor = tensor.to(dtype=torch.float64)
    else:
        tensor = tensor.to(dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("point samples must be finite")
    return tensor


def _evaluate_ast(ast: NodeLike, points: torch.Tensor) -> _ASTEvaluation:
    try:
        value, gradient, _hessian = _eval_single_input(ast, points, need_grad=True, need_hess=False)
    except Exception:
        nan_value = torch.full((points.shape[0],), math.nan, dtype=points.dtype, device=points.device)
        nan_gradient = torch.full((points.shape[0], 2), math.nan, dtype=points.dtype, device=points.device)
        return _ASTEvaluation(nan_value, nan_gradient, 0.0)
    assert gradient is not None
    value = value.reshape(-1)
    gradient = gradient.reshape(points.shape[0], 2)
    if value.is_complex() or gradient.is_complex():
        nan_value = torch.full((points.shape[0],), math.nan, dtype=points.dtype, device=points.device)
        nan_gradient = torch.full((points.shape[0], 2), math.nan, dtype=points.dtype, device=points.device)
        return _ASTEvaluation(nan_value, nan_gradient, 0.0)
    finite = torch.isfinite(value) & torch.isfinite(gradient).all(dim=1)
    return _ASTEvaluation(value, gradient, float(finite.to(dtype=points.dtype).mean()))


def _generator_fields(generator: Any, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = points[:, 0]
    u = points[:, 1]
    zero = torch.zeros_like(x)
    raw = generator.fields(x, u, zero, zero)
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        raise TypeError("generator.fields(x,u,u1,u2) must return at least (xi, eta)")
    xi = torch.as_tensor(raw[0], dtype=points.dtype, device=points.device).reshape(-1)
    eta = torch.as_tensor(raw[1], dtype=points.dtype, device=points.device).reshape(-1)
    if xi.numel() == 1:
        xi = xi.expand(points.shape[0])
    if eta.numel() == 1:
        eta = eta.expand(points.shape[0])
    if xi.shape[0] != points.shape[0] or eta.shape[0] != points.shape[0]:
        raise ValueError("generator point fields must match the point-sample length")
    if not torch.isfinite(xi).all() or not torch.isfinite(eta).all():
        raise ValueError("generator point fields must be finite on the supported domain")
    return xi, eta


def _joint_actions(generators: Sequence[Any], points: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    rows = []
    for generator in generators:
        xi, eta = _generator_fields(generator, points)
        rows.append(xi * gradient[:, 0] + eta * gradient[:, 1])
    return torch.stack(rows, dim=0)


def _individual_term_rejection(
    train: _ASTEvaluation,
    validation: _ASTEvaluation,
    cfg: InvariantCompilerConfig,
) -> str:
    if train.finite_fraction < 1.0 or validation.finite_fraction < 1.0:
        return "nonfinite_on_supported_domain"
    if min(_variance(train.value), _variance(validation.value)) < float(cfg.min_variance):
        return "constant_or_collapsed_value"
    if min(_gradient_rms(train.gradient), _gradient_rms(validation.gradient)) < float(cfg.min_gradient_rms):
        return "constant_or_collapsed_gradient"
    return ""


def _stable_nullspace(
    matrix: torch.Tensor,
    cfg: InvariantCompilerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    scales = torch.linalg.vector_norm(matrix, dim=0)
    # Do not amplify a column that is already zero at the determining
    # tolerance.  Analytic cancellations commonly leave roundoff-level
    # columns (rather than bitwise zeros); unit-normalizing those columns would
    # incorrectly turn exact invariants into full-rank directions.
    scale_floor = max(float(cfg.rank_atol), float(cfg.action_atol), _EPS) * math.sqrt(max(1, matrix.shape[0]))
    safe_scales = torch.where(scales > scale_floor, scales, torch.ones_like(scales))
    normalized = matrix / safe_scales.unsqueeze(0)
    full = bool(normalized.shape[0] < normalized.shape[1])
    _u, singular_values, vh = torch.linalg.svd(normalized, full_matrices=full)
    s0 = float(singular_values[0]) if singular_values.numel() else 0.0
    threshold = max(float(cfg.rank_atol), float(cfg.rank_rtol) * s0)
    rank = int(torch.count_nonzero(singular_values > threshold))
    raw = vh[rank:, :].T / safe_scales.unsqueeze(1)
    if raw.numel() == 0:
        basis = torch.empty((matrix.shape[1], 0), dtype=matrix.dtype, device=matrix.device)
        projector = torch.zeros((matrix.shape[1], matrix.shape[1]), dtype=matrix.dtype, device=matrix.device)
        return basis, projector, singular_values, rank
    basis, _r = torch.linalg.qr(raw, mode="reduced")
    projector = basis @ basis.T
    return basis, projector, singular_values, rank


def _sparse_nullspace_representatives(
    basis: torch.Tensor,
    projector: torch.Tensor,
    cfg: InvariantCompilerConfig,
) -> list[torch.Tensor]:
    seeds: list[torch.Tensor] = [basis[:, j] for j in range(basis.shape[1])]
    coordinate_scores = torch.diag(projector)
    order = torch.argsort(coordinate_scores, descending=True)
    for index in order[: max(0, int(cfg.max_sparse_seeds))]:
        seeds.append(projector[:, int(index)])
    directions: list[torch.Tensor] = []
    keys: set[tuple[int, ...]] = set()
    for seed in seeds:
        norm = torch.linalg.vector_norm(seed)
        if float(norm) <= _EPS:
            continue
        direction = seed / norm
        for _ in range(max(0, int(cfg.sparse_iterations))):
            cutoff = float(cfg.sparse_threshold) * float(torch.max(torch.abs(direction)))
            sparse = torch.sign(direction) * torch.relu(torch.abs(direction) - cutoff)
            projected = projector @ sparse
            projected_norm = torch.linalg.vector_norm(projected)
            if float(projected_norm) <= _EPS:
                break
            direction = projected / projected_norm
        direction = _canonical_direction(direction)
        key = tuple(int(round(float(v) * 1.0e7)) for v in direction)
        if key in keys:
            continue
        keys.add(key)
        directions.append(direction)
    directions.sort(key=lambda row: (int(torch.count_nonzero(torch.abs(row) > cfg.coefficient_tol)), -float(torch.max(torch.abs(row)))))
    return directions


def _canonical_direction(direction: torch.Tensor) -> torch.Tensor:
    max_abs = float(torch.max(torch.abs(direction)))
    if max_abs <= _EPS:
        return direction
    out = direction / max_abs
    nonzero = torch.nonzero(torch.abs(out) > 1.0e-12)
    if nonzero.numel() and float(out[int(nonzero[0])]) < 0.0:
        out = -out
    return out


def _compile_linear_combination(
    terms: Sequence[NodeLike],
    coefficients: torch.Tensor,
    cfg: InvariantCompilerConfig,
    *,
    normalize: bool = True,
) -> tuple[NodeLike | None, torch.Tensor, tuple[int, ...]]:
    coefficients = coefficients.clone()
    max_abs = float(torch.max(torch.abs(coefficients))) if coefficients.numel() else 0.0
    if max_abs <= _EPS:
        return None, coefficients, ()
    if normalize:
        coefficients = coefficients / max_abs
        cutoff = float(cfg.coefficient_tol)
    else:
        cutoff = float(cfg.coefficient_tol) * max_abs
    coefficients[torch.abs(coefficients) <= cutoff] = 0.0
    support = tuple(int(v) for v in torch.nonzero(coefficients, as_tuple=False).reshape(-1))
    ast: NodeLike | None = None
    for index in support:
        coefficient = float(coefficients[index])
        term = terms[index]
        piece = term if abs(coefficient - 1.0) <= 1.0e-12 else Mul(ConstNode(coefficient), term)
        ast = piece if ast is None else Add(ast, piece)
    return ast, coefficients, support


def _certify_invariant(
    ast: NodeLike,
    coefficients: torch.Tensor,
    support: tuple[int, ...],
    terms: Sequence[NodeLike],
    generators: Sequence[Any],
    train: torch.Tensor,
    validation: torch.Tensor,
    cfg: InvariantCompilerConfig,
) -> InvariantCandidateResult:
    train_eval = _evaluate_ast(ast, train)
    validation_eval = _evaluate_ast(ast, validation)
    train_actions = _joint_actions(generators, train, train_eval.gradient)
    validation_actions = _joint_actions(generators, validation, validation_eval.gradient)
    train_rms, train_rel = _action_metrics(train_actions, generators, train, train_eval.gradient)
    validation_rms, validation_rel = _action_metrics(validation_actions, generators, validation, validation_eval.gradient)
    train_variance = _variance(train_eval.value)
    validation_variance = _variance(validation_eval.value)
    train_gradient_rms = _gradient_rms(train_eval.gradient)
    validation_gradient_rms = _gradient_rms(validation_eval.gradient)
    finite = train_eval.finite_fraction == 1.0 and validation_eval.finite_fraction == 1.0
    noncollapsed = (
        min(train_variance, validation_variance) >= float(cfg.min_variance)
        and min(train_gradient_rms, validation_gradient_rms) >= float(cfg.min_gradient_rms)
    )
    action_ok = (train_rms <= cfg.action_atol or train_rel <= cfg.action_rtol) and (
        validation_rms <= cfg.action_atol or validation_rel <= cfg.action_rtol
    )
    accepted = bool(finite and noncollapsed and action_ok)
    if not finite:
        reason = "nonfinite_domain"
    elif not noncollapsed:
        reason = "constant_or_collapsed_combination"
    elif not action_ok:
        reason = "heldout_generator_action_failed"
    else:
        reason = "pending_independence_gate"
    return InvariantCandidateResult(
        ast=ast,
        coefficients=tuple(float(v) for v in coefficients),
        support=support,
        support_terms=tuple(repr(terms[j]) for j in support),
        train_action_rms=float(train_rms),
        train_action_relative=float(train_rel),
        validation_action_rms=float(validation_rms),
        validation_action_relative=float(validation_rel),
        train_variance=float(train_variance),
        validation_variance=float(validation_variance),
        train_gradient_rms=float(train_gradient_rms),
        validation_gradient_rms=float(validation_gradient_rms),
        finite_train_fraction=float(train_eval.finite_fraction),
        finite_validation_fraction=float(validation_eval.finite_fraction),
        independence_fraction=0.0,
        independent_rank=0,
        accepted=accepted,
        reason=reason,
    )


def _action_metrics(
    actions: torch.Tensor,
    generators: Sequence[Any],
    points: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[float, float]:
    absolute = _rms(actions)
    scales = []
    for generator in generators:
        xi, eta = _generator_fields(generator, points)
        scales.append(torch.sqrt((xi * gradient[:, 0]).square() + (eta * gradient[:, 1]).square()))
    scale = _rms(torch.stack(scales, dim=0))
    relative = absolute / max(scale, _EPS)
    return float(absolute), float(relative)


def _independence_certificate(
    selected_gradients: Sequence[torch.Tensor],
    candidate_gradient: torch.Tensor,
    cfg: InvariantCompilerConfig,
) -> tuple[float, int]:
    gradients = [*selected_gradients, candidate_gradient]
    count = len(gradients)
    dimension = int(candidate_gradient.shape[1])
    if count > dimension:
        return 0.0, dimension
    matrix = torch.stack(gradients, dim=2)  # N x d x q
    singular_values = torch.linalg.svdvals(matrix)
    s0 = singular_values[:, 0]
    thresholds = torch.maximum(
        torch.full_like(s0, float(cfg.independence_rank_atol)),
        s0 * float(cfg.independence_rank_rtol),
    )
    ranks = torch.sum(singular_values > thresholds.unsqueeze(1), dim=1)
    fraction = torch.mean((ranks >= count).to(dtype=matrix.dtype))
    generic_rank = int(torch.quantile(ranks.to(dtype=matrix.dtype), 0.5).item())
    return float(fraction), generic_rank


def _orbit_candidate_directions(
    matrix: torch.Tensor,
    target: torch.Tensor,
    cfg: InvariantCompilerConfig,
) -> list[torch.Tensor]:
    directions: list[torch.Tensor] = []
    # Exact or nearly exact one-term charts are preferred and remain legible.
    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        denom = torch.dot(column, column)
        if float(denom) <= _EPS:
            continue
        coeffs = torch.zeros(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
        coeffs[j] = torch.dot(column, target) / denom
        directions.append(coeffs)

    # Orthogonal matching pursuit provides bounded sparse combinations.
    selected: list[int] = []
    residual = target.clone()
    for _ in range(min(int(cfg.max_orbit_support), matrix.shape[1])):
        correlations = torch.abs(matrix.T @ residual)
        if selected:
            correlations[selected] = -1.0
        index = int(torch.argmax(correlations))
        if float(correlations[index]) <= _EPS:
            break
        selected.append(index)
        sub = matrix[:, selected]
        fit = torch.linalg.lstsq(sub, target).solution
        coeffs = torch.zeros(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
        coeffs[selected] = fit
        directions.append(coeffs)
        residual = target - matrix @ coeffs

    full = torch.linalg.lstsq(matrix, target).solution
    directions.append(full)
    unique: list[torch.Tensor] = []
    keys: set[tuple[int, ...]] = set()
    for row in directions:
        key = tuple(int(round(float(v) * 1.0e10)) for v in row)
        if key not in keys:
            keys.add(key)
            unique.append(row)
    return unique


def _empty_orbit(reason: str) -> OrbitCoordinateResult:
    return OrbitCoordinateResult(
        status="rejected",
        ast=None,
        coefficients=(),
        support=(),
        train_residual_rms=math.inf,
        train_residual_relative=math.inf,
        validation_residual_rms=math.inf,
        validation_residual_relative=math.inf,
        train_variance=0.0,
        validation_variance=0.0,
        finite_train_fraction=0.0,
        finite_validation_fraction=0.0,
        accepted=False,
        reason=reason,
    )


def _candidate_sort_key(row: InvariantCandidateResult) -> tuple[Any, ...]:
    return (
        0 if row.accepted else 1,
        len(row.support),
        float(row.validation_action_relative),
        float(row.train_action_relative),
        -float(row.validation_variance),
        repr(row.ast),
    )


def _unique_asts(asts: Sequence[NodeLike]) -> tuple[NodeLike, ...]:
    out: list[NodeLike] = []
    seen: set[str] = set()
    for ast in asts:
        key = repr(ast)
        if key in seen:
            continue
        seen.add(key)
        out.append(ast)
    return tuple(out)


def _variance(values: torch.Tensor) -> float:
    if not torch.isfinite(values).all() or values.numel() < 2:
        return 0.0
    return float(torch.var(values, unbiased=False))


def _gradient_rms(gradient: torch.Tensor) -> float:
    if not torch.isfinite(gradient).all():
        return 0.0
    return _rms(gradient)


def _rms(values: torch.Tensor) -> float:
    if values.numel() == 0 or not torch.isfinite(values).all():
        return math.inf
    return float(torch.sqrt(torch.mean(values.square())))


def _human(ast: NodeLike) -> str:
    try:
        return ast_to_human_readable(ast)
    except Exception:
        return repr(ast)


__all__ = [
    "InvariantCandidateResult",
    "InvariantCompilationResult",
    "InvariantCompilerConfig",
    "InvariantObjectiveReport",
    "OrbitCoordinateResult",
    "SubalgebraInvariantCompilation",
    "SymbolicInvariantObjective",
    "compile_orbit_coordinate",
    "compile_point_invariants",
    "compile_subalgebra_invariants",
    "default_point_candidate_vocabulary",
    "point_coordinate_ast_to_de_ast",
]
