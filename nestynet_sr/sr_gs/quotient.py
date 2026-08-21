# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Quotient-chart compiler for affine symmetry algebras.

PR5 keeps this deliberately conservative: it compiles only exact, recognizable
affine reductions into auditable ``ReductionPlan`` objects.  Active Stage A/FSS
wiring and compound-coordinate substitution are later PRs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from nestynet.coords import LinearAffineMap, LogMonomialMap, QuadraticFormMap

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AddNode,
    AcosNode,
    AtanNode,
    AtomNode,
    AsinNode,
    ArgNode,
    ConstNode,
    ConjNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    RealNode,
    SinNode,
    Var,
    _collect_var_idxs_from_node,
    ast_to_human_readable,
    clone_ast,
    eval_input_expr,
)

from .affine_algebra import SymmetryAlgebraSpec
from .unit_torus import build_monomial_ast

_EPS = 1.0e-12
# Relative alpha/beta cut for spectral-gap (noise-calibrated) solves: noisy
# pure-invariance generators leak |alpha|,|beta| <~ 1e-2 of their input-action
# scale, while genuine homogeneity sits at O(0.4).
_CALIBRATED_OUTPUT_ACTION_REL_TOL = 0.05


@dataclass(frozen=True)
class DomainSpec:
    """Domain and branch assumptions for a compiled coordinate."""

    assumptions: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    branches: tuple[str, ...] = ()

    def excludes(self, condition: str) -> bool:
        target = _compact(condition)
        return any(target in _compact(item) or _compact(item) in target for item in self.exclusions)

    def to_report(self) -> dict[str, Any]:
        return {
            "assumptions": list(self.assumptions),
            "exclusions": list(self.exclusions),
            "branches": list(self.branches),
        }


@dataclass(frozen=True)
class OutputActionSpec:
    """Affine output action attached to a one-generator reduction."""

    alpha: float = 0.0
    beta: float = 0.0
    generator_scale: float = 1.0

    @property
    def is_equivariant(self) -> bool:
        return abs(float(self.alpha)) > _EPS or abs(float(self.beta)) > _EPS

    def to_report(self) -> dict[str, float | bool]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "generator_scale": float(self.generator_scale),
            "is_equivariant": bool(self.is_equivariant),
        }


@dataclass(frozen=True)
class OutputNormalFormSpec:
    """Executable scalar normal form for a one-dimensional orbit reduction.

    The graph tangency equation is ``X f = alpha + beta f`` once the orbit
    coordinate ``s`` is gauged so that ``X s = 1``.  This spec records the
    corresponding target transform between raw target values ``f`` and the
    reduced target ``H(z)``.
    """

    kind: str
    reduced_target_name: str = "H"
    orbit_coordinate: str | None = None
    invariant_coordinates: tuple[str, ...] = ()
    alpha: float = 0.0
    beta: float = 0.0
    target_formula: str = ""
    reduced_target_formula: str = ""
    assumptions: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def reduce_target(self, y: Any, orbit_values: Any | None = None) -> np.ndarray:
        """Transform raw target values ``f`` into reduced target values ``H``."""

        y_arr = np.asarray(y, dtype=float).reshape(-1)
        alpha = float(self.alpha)
        beta = float(self.beta)
        if abs(beta) > _EPS:
            s = _require_orbit_values(orbit_values)
            return np.exp(-beta * s) * (y_arr + alpha / beta)
        if abs(alpha) > _EPS:
            s = _require_orbit_values(orbit_values)
            return y_arr - alpha * s
        return y_arr.copy()

    def reconstruct_target(self, reduced_y: Any, orbit_values: Any | None = None) -> np.ndarray:
        """Reconstruct raw target values ``f`` from reduced target values ``H``."""

        h_arr = np.asarray(reduced_y, dtype=float).reshape(-1)
        alpha = float(self.alpha)
        beta = float(self.beta)
        if abs(beta) > _EPS:
            s = _require_orbit_values(orbit_values)
            return -alpha / beta + np.exp(beta * s) * h_arr
        if abs(alpha) > _EPS:
            s = _require_orbit_values(orbit_values)
            return alpha * s + h_arr
        return h_arr.copy()

    def to_report(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reduced_target_name": self.reduced_target_name,
            "orbit_coordinate": self.orbit_coordinate,
            "invariant_coordinates": list(self.invariant_coordinates),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "target_formula": self.target_formula,
            "reduced_target_formula": self.reduced_target_formula,
            "assumptions": list(self.assumptions),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class CoordinateSpec:
    """Executable coordinate plus exact symbolic/provenance metadata."""

    name: str
    kind: str
    ast: Node | str | None
    coordinate_map: Any | None
    domain: DomainSpec = field(default_factory=DomainSpec)
    provenance: dict[str, Any] = field(default_factory=dict)
    raw_support: tuple[int, ...] = ()
    gauge: str | None = None
    evaluator: Callable[[np.ndarray], np.ndarray] | None = None
    unit_speed_generator: np.ndarray | None = None

    @property
    def raw_var_idxs(self) -> tuple[int, ...]:
        if self.raw_support:
            return tuple(int(v) for v in self.raw_support)
        if self.ast is not None and not isinstance(self.ast, str):
            return tuple(int(v) for v in _collect_var_idxs_from_node(self.ast))
        return ()

    def evaluate(self, x: Any) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if self.evaluator is not None:
            value = self.evaluator(x_arr)
            return np.asarray(value, dtype=float).reshape(x_arr.shape[0])
        if self.coordinate_map is not None:
            xt = torch.as_tensor(x_arr, dtype=torch.float64)
            with torch.no_grad():
                z = self.coordinate_map.forward(xt)
            return z.detach().cpu().numpy().reshape(x_arr.shape[0], -1)[:, 0]
        if self.ast is not None and not isinstance(self.ast, str):
            xt = torch.as_tensor(x_arr, dtype=torch.float64)
            return eval_input_expr(self.ast, xt).detach().cpu().numpy().reshape(x_arr.shape[0])
        raise ValueError(f"CoordinateSpec {self.name!r} has no executable representation")

    def satisfies_unit_speed(self, algebra: SymmetryAlgebraSpec, *, sample: np.ndarray | None = None, tol: float = 1.0e-6) -> bool:
        if self.unit_speed_generator is None:
            return False
        if sample is None:
            sample = _default_sample(algebra.input_dim)
        fields = algebra.input_fields(sample, basis=self.unit_speed_generator.reshape(-1, 1), physical=True)[0]
        x = torch.as_tensor(sample, dtype=torch.float64).requires_grad_(True)
        y = self._evaluate_torch(x).reshape(sample.shape[0])
        grad = torch.autograd.grad(y.sum(), x, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None:
            return False
        speed = torch.sum(grad * torch.as_tensor(fields, dtype=torch.float64), dim=1)
        return bool(torch.allclose(speed, torch.ones_like(speed), atol=tol, rtol=tol))

    def _evaluate_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.coordinate_map is not None:
            return self.coordinate_map.forward(x)
        if self.ast is not None and not isinstance(self.ast, str):
            return eval_input_expr(self.ast, x)
        if self.evaluator is not None:
            value = self.evaluator(x.detach().cpu().numpy())
            return torch.as_tensor(value, dtype=x.dtype, device=x.device).reshape(x.shape[0], 1)
        raise ValueError(f"CoordinateSpec {self.name!r} has no differentiable representation")

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "ast": repr(self.ast) if self.ast is not None else None,
            "human": ast_to_human_readable(self.ast) if self.ast is not None and not isinstance(self.ast, str) else self.ast,
            "coordinate_map": type(self.coordinate_map).__name__ if self.coordinate_map is not None else None,
            "domain": self.domain.to_report(),
            "provenance": dict(self.provenance),
            "raw_support": [int(v) for v in self.raw_support],
            "gauge": self.gauge,
        }


@dataclass(frozen=True)
class ReductionPlan:
    """Compiled quotient/equivariant coordinate plan."""

    algebra: SymmetryAlgebraSpec
    invariant_coordinates: tuple[CoordinateSpec, ...] = ()
    orbit_coordinates: tuple[CoordinateSpec, ...] = ()
    generic_orbit_rank: int = 0
    singular_strata: tuple[str, ...] = ()
    domain: DomainSpec = field(default_factory=DomainSpec)
    output_action: OutputActionSpec = field(default_factory=OutputActionSpec)
    normal_form: OutputNormalFormSpec | None = None
    status: str = "audit"
    reason: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def quotient_codimension(self) -> int:
        return max(0, int(self.algebra.input_dim) - int(self.generic_orbit_rank))

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "generic_orbit_rank": int(self.generic_orbit_rank),
            "quotient_codimension": int(self.quotient_codimension),
            "singular_strata": list(self.singular_strata),
            "domain": self.domain.to_report(),
            "output_action": self.output_action.to_report(),
            "normal_form": self.normal_form.to_report() if self.normal_form is not None else None,
            "invariant_coordinates": [c.to_report() for c in self.invariant_coordinates],
            "orbit_coordinates": [c.to_report() for c in self.orbit_coordinates],
            "provenance": dict(self.provenance),
        }


def substitute_local_coordinate_ast(expr: Node, local_inputs: Sequence[Node]) -> Node:
    """Substitute local ``z_i`` variables in ``expr`` with raw-input ASTs.

    The quotient compiler may work in an atom-local namespace where ``Var(0)``
    means local coordinate ``z_0``.  Before a reduction can be stored globally
    as shadow evidence, those local variables must be composed with the atom's
    actual ``AtomNode.inputs``.
    """

    inputs = tuple(clone_ast(inp) for inp in local_inputs)

    def rec(node: Node) -> Node:
        if isinstance(node, AtomNode):
            kind = str(getattr(node, "kind", "")).lower()
            if kind in ("var", "x", "input") and len(node.var_idxs) == 1:
                idx = int(node.var_idxs[0])
                if idx < 0 or idx >= len(inputs):
                    raise IndexError(f"local coordinate z_{idx} has no input expression")
                return clone_ast(inputs[idx])
            return clone_ast(node)
        if isinstance(node, AddNode):
            return AddNode(rec(node.left), rec(node.right))
        if isinstance(node, MulNode):
            return MulNode(rec(node.left), rec(node.right))
        if isinstance(node, PowNode):
            return PowNode(rec(node.base), node.exponent)
        if isinstance(node, LogNode):
            return LogNode(rec(node.arg))
        if isinstance(node, ExpNode):
            return ExpNode(rec(node.arg))
        if isinstance(node, SinNode):
            return SinNode(rec(node.arg))
        if isinstance(node, CosNode):
            return CosNode(rec(node.arg))
        if isinstance(node, AsinNode):
            return AsinNode(rec(node.arg))
        if isinstance(node, AcosNode):
            return AcosNode(rec(node.arg))
        if isinstance(node, AtanNode):
            return AtanNode(rec(node.arg))
        if isinstance(node, ConjNode):
            return ConjNode(rec(node.arg))
        if isinstance(node, RealNode):
            return RealNode(rec(node.arg))
        if isinstance(node, ImagNode):
            return ImagNode(rec(node.arg))
        if isinstance(node, AbsNode):
            return AbsNode(rec(node.arg))
        if isinstance(node, ArgNode):
            return ArgNode(rec(node.arg))
        if isinstance(node, ConstNode):
            return ConstNode(node.value)
        return clone_ast(node)

    return rec(expr)


def compose_coordinate_spec_with_inputs(coord: CoordinateSpec, local_inputs: Sequence[Node]) -> CoordinateSpec:
    """Return a coordinate whose AST/provenance refer to raw input variables."""

    if coord.ast is None or isinstance(coord.ast, str):
        raw_ast = coord.ast
        raw_support = tuple(coord.raw_support)
        human = raw_ast
    else:
        raw_ast = substitute_local_coordinate_ast(coord.ast, local_inputs)
        raw_support = tuple(int(v) for v in _collect_var_idxs_from_node(raw_ast))
        human = ast_to_human_readable(raw_ast)
    provenance = dict(coord.provenance or {})
    provenance.update(
        {
            "local_coordinate_namespace": True,
            "local_input_count": int(len(tuple(local_inputs))),
            "substituted_to_raw_ast": raw_ast is not coord.ast,
            "raw_human": human,
        }
    )
    return CoordinateSpec(
        name=coord.name,
        kind=coord.kind,
        ast=raw_ast,
        coordinate_map=None,
        domain=_compose_domain_with_inputs(coord.domain, local_inputs),
        provenance=provenance,
        raw_support=raw_support,
        gauge=coord.gauge,
        evaluator=None,
        unit_speed_generator=None,
    )


def compose_reduction_plan_with_inputs(plan: ReductionPlan, local_inputs: Sequence[Node]) -> ReductionPlan:
    """Compose all symbolic coordinates in a reduction plan with local inputs."""

    inputs = tuple(local_inputs)
    invariants = tuple(compose_coordinate_spec_with_inputs(c, inputs) for c in plan.invariant_coordinates)
    orbits = tuple(compose_coordinate_spec_with_inputs(c, inputs) for c in plan.orbit_coordinates)
    provenance = dict(plan.provenance or {})
    provenance.update(
        {
            "local_coordinate_namespace": True,
            "local_input_count": int(len(inputs)),
            "substituted_to_raw_ast": True,
        }
    )
    return ReductionPlan(
        algebra=plan.algebra,
        invariant_coordinates=invariants,
        orbit_coordinates=orbits,
        generic_orbit_rank=plan.generic_orbit_rank,
        singular_strata=tuple(plan.singular_strata),
        domain=_compose_domain_with_inputs(plan.domain, inputs),
        output_action=plan.output_action,
        normal_form=plan.normal_form,
        status=plan.status,
        reason=plan.reason,
        provenance=provenance,
    )


def compile_reduction_plan(algebra: SymmetryAlgebraSpec) -> ReductionPlan:
    """Compile an accepted affine algebra into a conservative reduction plan."""

    cert = algebra.certificate
    if cert is not None and not cert.quotient_ready:
        return ReductionPlan(
            algebra=algebra,
            generic_orbit_rank=int(algebra.distribution_rank),
            status="audit",
            reason=str(cert.quotient_policy),
            provenance={"source": "data", "compiler": "affine_quotient_v1"},
        )
    if algebra.nullity <= 0:
        return ReductionPlan(algebra=algebra, status="audit", reason="no_nullspace")

    # Chart dispatch must precede the identity-chart branches below: in a
    # transformed chart both the 2-D one-generator compilers and the
    # linear-projection plan would render semantically wrong coordinates
    # (e.g. linear-in-log ASTs).
    chart = getattr(algebra, "chart", "identity")
    if chart == "log":
        return _compile_log_monomial_plan(algebra)
    if chart == "reciprocal":
        return _compile_reciprocal_linear_plan(algebra)

    if algebra.distribution_rank == 1 and algebra.input_dim == 2 and algebra.nullity == 1:
        plan = _compile_one_generator_2d(algebra, algebra.nullspace_basis[:, 0])
        if plan is not None:
            return plan

    if algebra.linear_invariant_covectors.size:
        return _compile_linear_projection_plan(algebra)

    return ReductionPlan(
        algebra=algebra,
        generic_orbit_rank=int(algebra.distribution_rank),
        status="audit",
        reason="no_supported_phase_one_chart",
        provenance={"source": "data", "compiler": "affine_quotient_v1"},
    )


def _compile_one_generator_2d(algebra: SymmetryAlgebraSpec, coeffs: np.ndarray) -> ReductionPlan | None:
    gen = algebra.basis_generators[0]
    A = np.asarray(gen.A_physical, dtype=float)
    b = np.asarray(gen.b_physical, dtype=float)
    alpha = float(gen.alpha_physical)
    beta = float(gen.beta_physical)
    center = _fixed_point(A, b)
    if center is None:
        center = np.zeros(2, dtype=float)
    if _is_common_diagonal_scaling(A, b, center):
        return _compile_common_scaling_plan(algebra, coeffs, A, center, alpha, beta)
    if _is_rotation(A, b, center):
        return _compile_rotation_plan(algebra, coeffs, A, center, alpha, beta)
    if np.linalg.norm(A) <= 1.0e-8 and np.linalg.norm(b) > 1.0e-8:
        return _compile_translation_plan(algebra, coeffs, b, alpha, beta)
    return None


def _compile_common_scaling_plan(
    algebra: SymmetryAlgebraSpec,
    coeffs: np.ndarray,
    A: np.ndarray,
    center: np.ndarray,
    alpha: float,
    beta: float,
) -> ReductionPlan:
    lam = float(np.trace(A) / 2.0)
    powers = np.asarray([[1.0, -1.0]], dtype=float)
    invariant = CoordinateSpec(
        name="log_ratio_x0_x1",
        kind="log_ratio",
        ast=LogNode(MulNode(_shifted_var(0, center[0]), PowNode(_shifted_var(1, center[1]), -1.0))),
        coordinate_map=LogMonomialMap(powers, center=torch.as_tensor(center, dtype=torch.float64), domain="abs", eps=1.0e-12),
        domain=DomainSpec(
            assumptions=("same scaling weight on x0 and x1", "log-ratio chart uses absolute branch"),
            exclusions=(_zero_condition(0, center[0]), _zero_condition(1, center[1])),
            branches=("fixed sign component for x0/x1",),
        ),
        provenance={"source": "data", "family": "common_diagonal_scaling"},
        raw_support=(0, 1),
        gauge="log(abs((x0-c0)/(x1-c1)))",
    )
    orbit_power = np.asarray([[1.0 / lam, 0.0]], dtype=float) if abs(lam) > _EPS else np.asarray([[1.0, 0.0]], dtype=float)
    orbit = CoordinateSpec(
        name="scaling_orbit_log_x0",
        kind="orbit_log",
        ast=MulNode(ConstNode(float(1.0 / lam)), LogNode(_shifted_var(0, center[0]))) if abs(lam) > _EPS else LogNode(_shifted_var(0, center[0])),
        coordinate_map=LogMonomialMap(orbit_power, center=torch.as_tensor(center, dtype=torch.float64), domain="abs", eps=1.0e-12),
        domain=invariant.domain,
        provenance={"source": "data", "family": "common_diagonal_scaling", "unit_speed": abs(lam) > _EPS},
        raw_support=(0,),
        gauge="X s = 1 away from chart singularities",
        unit_speed_generator=np.asarray(coeffs, dtype=float),
    )
    output_action = OutputActionSpec(alpha=alpha, beta=beta)
    return ReductionPlan(
        algebra=algebra,
        invariant_coordinates=(invariant,),
        orbit_coordinates=(orbit,),
        generic_orbit_rank=1,
        singular_strata=tuple(invariant.domain.exclusions),
        domain=invariant.domain,
        output_action=output_action,
        normal_form=_normal_form_for_output_action(output_action, (invariant,), (orbit,)),
        status="compiled",
        reason="common_diagonal_scaling",
        provenance={"source": "data", "compiler": "affine_quotient_v1"},
    )


def _compile_rotation_plan(
    algebra: SymmetryAlgebraSpec,
    coeffs: np.ndarray,
    A: np.ndarray,
    center: np.ndarray,
    alpha: float,
    beta: float,
) -> ReductionPlan:
    Q = np.eye(2, dtype=float)
    invariant = CoordinateSpec(
        name="radius_squared",
        kind="quadratic_radius",
        ast=AddNode(PowNode(_shifted_var(0, center[0]), 2.0), PowNode(_shifted_var(1, center[1]), 2.0)),
        coordinate_map=QuadraticFormMap(torch.as_tensor(Q, dtype=torch.float64), center=torch.as_tensor(center, dtype=torch.float64)),
        domain=DomainSpec(
            assumptions=("Euclidean radial chart for skew affine generator",),
            exclusions=(_center_condition(center),),
        ),
        provenance={"source": "data", "family": "rotation"},
        raw_support=(0, 1),
        gauge="r^2",
    )
    omega = float(0.5 * (A[1, 0] - A[0, 1]))
    orbit = CoordinateSpec(
        name="rotation_angle",
        kind="orbit_angle",
        ast=AtanNode(MulNode(_shifted_var(1, center[1]), PowNode(_shifted_var(0, center[0]), -1.0))),
        coordinate_map=None,
        domain=DomainSpec(
            assumptions=("local atan(y/x) angle chart",),
            exclusions=(_center_condition(center), _zero_condition(0, center[0])),
            branches=("atan branch; use only as orbit metadata in PR5",),
        ),
        provenance={"source": "data", "family": "rotation", "omega": omega, "unit_speed": abs(omega) > _EPS},
        raw_support=(0, 1),
        gauge="theta / omega locally satisfies X s = 1",
        evaluator=lambda x: np.arctan2(x[:, 1] - center[1], x[:, 0] - center[0]) / omega if abs(omega) > _EPS else np.arctan2(x[:, 1] - center[1], x[:, 0] - center[0]),
        unit_speed_generator=np.asarray(coeffs, dtype=float),
    )
    output_action = OutputActionSpec(alpha=alpha, beta=beta)
    return ReductionPlan(
        algebra=algebra,
        invariant_coordinates=(invariant,),
        orbit_coordinates=(orbit,),
        generic_orbit_rank=1,
        singular_strata=(_center_condition(center),),
        domain=invariant.domain,
        output_action=output_action,
        normal_form=_normal_form_for_output_action(output_action, (invariant,), (orbit,)),
        status="compiled",
        reason="rotation_quadratic_radius",
        provenance={"source": "data", "compiler": "affine_quotient_v1"},
    )


def _compile_translation_plan(
    algebra: SymmetryAlgebraSpec,
    coeffs: np.ndarray,
    b: np.ndarray,
    alpha: float,
    beta: float,
) -> ReductionPlan:
    b_unit, orbit_vec, coeffs_unit, generator_scale = _translation_orbit_gauge(b, coeffs)
    alpha_unit = float(alpha) / float(generator_scale)
    beta_unit = float(beta) / float(generator_scale)
    covectors = _orthogonal_complement(b_unit.reshape(1, -1), 2)
    invariant_vec = covectors[0]
    invariant = CoordinateSpec(
        name="translation_invariant_linear",
        kind="linear_projection",
        ast=_linear_ast(invariant_vec),
        coordinate_map=LinearAffineMap(torch.as_tensor(invariant_vec.reshape(1, -1), dtype=torch.float64)),
        domain=DomainSpec(),
        provenance={"source": "data", "family": "translation", "covector": [float(v) for v in invariant_vec]},
        raw_support=(0, 1),
        gauge="orthogonal projection to translation direction",
    )
    orbit = CoordinateSpec(
        name="translation_orbit_linear",
        kind="orbit_linear",
        ast=_linear_ast(orbit_vec),
        coordinate_map=LinearAffineMap(torch.as_tensor(orbit_vec.reshape(1, -1), dtype=torch.float64)),
        domain=DomainSpec(),
        provenance={"source": "data", "family": "translation", "unit_speed": True},
        raw_support=(0, 1),
        gauge="X s = 1",
        unit_speed_generator=np.asarray(coeffs_unit, dtype=float),
    )
    output_action = OutputActionSpec(alpha=alpha_unit, beta=beta_unit, generator_scale=float(generator_scale))
    return ReductionPlan(
        algebra=algebra,
        invariant_coordinates=(invariant,),
        orbit_coordinates=(orbit,),
        generic_orbit_rank=1,
        output_action=output_action,
        normal_form=_normal_form_for_output_action(output_action, (invariant,), (orbit,)),
        status="compiled",
        reason="translation_linear_projection",
        provenance={"source": "data", "compiler": "affine_quotient_v1"},
    )


def _compile_log_monomial_plan(algebra: SymmetryAlgebraSpec) -> ReductionPlan:
    """Compile a snapped log-chart invariant covector into a monomial chart.

    In the log chart ``u = log(x)`` the single invariant covector ``e`` is the
    exponent vector of the monomial invariant ``z = prod_i x_i**e_i``.  Only a
    snapped single-covector algebra is compiled; anything else stays audit.
    """

    base_provenance = {"source": "data", "compiler": "affine_quotient_v1", "chart": "log"}
    covectors = np.asarray(algebra.linear_invariant_covectors, dtype=float)
    n_covectors = int(covectors.shape[0]) if covectors.ndim == 2 else 0
    if n_covectors == 0:
        return ReductionPlan(
            algebra=algebra,
            generic_orbit_rank=int(algebra.distribution_rank),
            status="audit",
            reason="log_chart_no_invariant_covectors",
            provenance=base_provenance,
        )
    if n_covectors > 1:
        # An orthonormal multi-covector mixture is basis-arbitrary and need not
        # contain rational rays; lattice reduction is future work.
        return ReductionPlan(
            algebra=algebra,
            generic_orbit_rank=int(algebra.distribution_rank),
            status="audit",
            reason="log_chart_multi_invariant_unsupported",
            provenance=base_provenance,
        )
    snap_report = dict((algebra.evidence or {}).get("chart_snap") or {})
    if str(snap_report.get("status", "")) != "snapped":
        return ReductionPlan(
            algebra=algebra,
            generic_orbit_rank=int(algebra.distribution_rank),
            status="audit",
            reason="log_chart_covector_not_snapped",
            provenance=base_provenance,
        )

    exponents = tuple(int(round(float(v))) for v in covectors[0])
    support = tuple(i for i, e in enumerate(exponents) if e != 0)
    exps = np.asarray([float(v) for v in exponents], dtype=float)

    def _monomial_evaluator(x: Any, _e: np.ndarray = exps) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.prod(np.power(x_arr, _e.reshape(1, -1)), axis=1)

    domain = DomainSpec(
        assumptions=("log chart: all inputs strictly positive",),
        exclusions=tuple(f"x{i} <= 0" for i in support),
    )
    invariant = CoordinateSpec(
        name="log_monomial_invariant_0",
        kind="monomial",
        ast=build_monomial_ast(exponents),
        coordinate_map=LogMonomialMap(np.asarray([exps]), domain="signed", eps=1.0e-12),
        domain=domain,
        provenance={
            "source": "data",
            "family": "log_monomial",
            "chart": "log",
            "exponents": [int(v) for v in exponents],
            "covector": [float(v) for v in covectors[0]],
            "snap": snap_report,
            "coordinate_map_semantics": "log-chart linearization; ast/evaluator are the monomial",
        },
        raw_support=support,
        gauge="monomial invariant of log-chart translation subalgebra",
        evaluator=_monomial_evaluator,
    )
    output_action = _thresholded_output_action(algebra)
    return ReductionPlan(
        algebra=algebra,
        invariant_coordinates=(invariant,),
        orbit_coordinates=(),
        generic_orbit_rank=int(algebra.distribution_rank),
        singular_strata=tuple(domain.exclusions),
        domain=domain,
        output_action=output_action,
        normal_form=_normal_form_for_output_action(output_action, (invariant,), ()),
        status="compiled",
        reason="log_monomial_invariant",
        provenance=base_provenance,
    )


def _thresholded_output_action(algebra: SymmetryAlgebraSpec) -> OutputActionSpec:
    """Largest generator output action, with noise components thresholded off.

    Pure invariance is reported exactly, while genuine equivariance
    (homogeneity) survives to be judged by the promotion gate.  An absolute
    (oracle) solve has roundoff components ~1e-15 so an absolute cut suffices;
    a spectral-gap (noise-calibrated) solve leaks noise-level alpha/beta into
    the recovered generators, so the cut is relative to each generator's
    input-action scale (measured leakage <~1e-2 vs 0.45 for genuine
    homogeneity).
    """

    alpha = 0.0
    beta = 0.0
    alpha_rel = 0.0
    beta_rel = 0.0
    for gen in algebra.basis_generators:
        a = float(getattr(gen, "alpha_physical", 0.0))
        b = float(getattr(gen, "beta_physical", 0.0))
        A_phys = np.asarray(getattr(gen, "A_physical", ()), dtype=float)
        b_phys = np.asarray(getattr(gen, "b_physical", ()), dtype=float)
        scale = max(float(np.linalg.norm(A_phys)) + float(np.linalg.norm(b_phys)), 1.0e-12)
        if abs(a) > abs(alpha):
            alpha, alpha_rel = a, abs(a) / scale
        if abs(b) > abs(beta):
            beta, beta_rel = b, abs(b) / scale
    if str((algebra.evidence or {}).get("nullity_strategy", "")) == "spectral_gap":
        if alpha_rel <= _CALIBRATED_OUTPUT_ACTION_REL_TOL:
            alpha = 0.0
        if beta_rel <= _CALIBRATED_OUTPUT_ACTION_REL_TOL:
            beta = 0.0
    else:
        if abs(alpha) <= 1.0e-10:
            alpha = 0.0
        if abs(beta) <= 1.0e-10:
            beta = 0.0
    return OutputActionSpec(alpha=alpha, beta=beta)


def _reciprocal_linear_ast(coeffs: Sequence[float]) -> Node:
    """AST for ``sum_i c_i / x_i`` (linear in the reciprocal chart)."""

    terms: list[Node] = []
    for i, coeff in enumerate(coeffs):
        c = float(coeff)
        if abs(c) <= 1.0e-12:
            continue
        recip = PowNode(Var(int(i)), -1.0)
        terms.append(recip if abs(c - 1.0) <= 1.0e-12 else MulNode(ConstNode(c), recip))
    if not terms:
        return ConstNode(0.0)
    expr = terms[0]
    for term in terms[1:]:
        expr = AddNode(expr, term)
    return expr


def _compile_reciprocal_linear_plan(algebra: SymmetryAlgebraSpec) -> ReductionPlan:
    """Compile a snapped reciprocal-chart covector into a ``sum c_i/x_i`` form.

    In the reciprocal chart ``u = 1/x`` the single invariant covector ``c`` is
    the coefficient vector of the invariant ``z = sum_i c_i / x_i`` (the
    parallel-resistor / reduced-mass / lens family).  Only a snapped
    single-covector algebra is compiled; anything else stays audit.
    """

    base_provenance = {"source": "data", "compiler": "affine_quotient_v1", "chart": "reciprocal"}
    covectors = np.asarray(algebra.linear_invariant_covectors, dtype=float)
    n_covectors = int(covectors.shape[0]) if covectors.ndim == 2 else 0
    if n_covectors == 0:
        return ReductionPlan(algebra=algebra, generic_orbit_rank=int(algebra.distribution_rank),
                             status="audit", reason="reciprocal_chart_no_invariant_covectors", provenance=base_provenance)
    if n_covectors > 1:
        return ReductionPlan(algebra=algebra, generic_orbit_rank=int(algebra.distribution_rank),
                             status="audit", reason="reciprocal_chart_multi_invariant_unsupported", provenance=base_provenance)
    snap_report = dict((algebra.evidence or {}).get("chart_snap") or {})
    if str(snap_report.get("status", "")) != "snapped":
        return ReductionPlan(algebra=algebra, generic_orbit_rank=int(algebra.distribution_rank),
                             status="audit", reason="reciprocal_chart_covector_not_snapped", provenance=base_provenance)

    coeffs = tuple(int(round(float(v))) for v in covectors[0])
    support = tuple(i for i, c in enumerate(coeffs) if c != 0)
    coeff_arr = np.asarray([float(v) for v in coeffs], dtype=float)

    def _reciprocal_evaluator(x: Any, _c: np.ndarray = coeff_arr) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.sum(_c.reshape(1, -1) / x_arr, axis=1)

    domain = DomainSpec(
        assumptions=("reciprocal chart: all inputs nonzero, no sign crossing",),
        exclusions=tuple(f"x{i} == 0" for i in support),
    )
    invariant = CoordinateSpec(
        name="reciprocal_linear_invariant_0",
        kind="reciprocal_linear",
        ast=_reciprocal_linear_ast(coeffs),
        coordinate_map=None,
        domain=domain,
        provenance={
            "source": "data",
            "family": "reciprocal_linear",
            "chart": "reciprocal",
            "coefficients": [int(v) for v in coeffs],
            "covector": [float(v) for v in covectors[0]],
            "snap": snap_report,
        },
        raw_support=support,
        gauge="linear invariant of reciprocal-chart translation subalgebra",
        evaluator=_reciprocal_evaluator,
    )
    output_action = _thresholded_output_action(algebra)
    return ReductionPlan(
        algebra=algebra,
        invariant_coordinates=(invariant,),
        orbit_coordinates=(),
        generic_orbit_rank=int(algebra.distribution_rank),
        singular_strata=tuple(domain.exclusions),
        domain=domain,
        output_action=output_action,
        normal_form=_normal_form_for_output_action(output_action, (invariant,), ()),
        status="compiled",
        reason="reciprocal_linear_invariant",
        provenance=base_provenance,
    )


def _compile_linear_projection_plan(algebra: SymmetryAlgebraSpec) -> ReductionPlan:
    covectors = np.asarray(algebra.linear_invariant_covectors, dtype=float)
    invariants = []
    for i, row in enumerate(covectors):
        invariants.append(
            CoordinateSpec(
                name=f"linear_invariant_{i}",
                kind="linear_projection",
                ast=_linear_ast(row),
                coordinate_map=LinearAffineMap(torch.as_tensor(row.reshape(1, -1), dtype=torch.float64)),
                domain=DomainSpec(),
                provenance={"source": "data", "family": "linear_distribution_annihilator", "covector": [float(v) for v in row]},
                raw_support=tuple(range(algebra.input_dim)),
                gauge="distribution annihilator",
            )
        )
    orbit_coordinates: tuple[CoordinateSpec, ...] = ()
    if algebra.distribution_basis.size:
        direction = np.asarray(algebra.distribution_basis[0], dtype=float)
        direction = direction / max(float(np.linalg.norm(direction)), _EPS)
        orbit_coordinates = (
            CoordinateSpec(
                name="linear_orbit_coordinate",
                kind="orbit_linear_distribution",
                ast=_linear_ast(direction),
                coordinate_map=LinearAffineMap(torch.as_tensor(direction.reshape(1, -1), dtype=torch.float64)),
                domain=DomainSpec(),
                provenance={"source": "data", "family": "linear_distribution", "unit_direction": True},
                raw_support=tuple(range(algebra.input_dim)),
                gauge="unit projection along distribution basis",
            ),
        )
    return ReductionPlan(
        algebra=algebra,
        invariant_coordinates=tuple(invariants),
        orbit_coordinates=orbit_coordinates,
        generic_orbit_rank=int(algebra.distribution_rank),
        output_action=OutputActionSpec(),
        normal_form=_normal_form_for_output_action(OutputActionSpec(), tuple(invariants), orbit_coordinates),
        status="compiled",
        reason="linear_distribution_annihilator",
        provenance={"source": "data", "compiler": "affine_quotient_v1"},
    )


def _translation_orbit_gauge(b: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    coeffs_arr = np.asarray(coeffs, dtype=float).reshape(-1)
    idx = int(np.argmax(np.abs(b_arr)))
    dominant = float(b_arr[idx]) if b_arr.size else 0.0
    rest = np.delete(b_arr, idx) if b_arr.size else np.asarray([], dtype=float)
    if abs(dominant) > _EPS and float(np.linalg.norm(rest)) <= 1.0e-8 * max(1.0, abs(dominant)):
        orbit_vec = np.zeros_like(b_arr)
        orbit_vec[idx] = 1.0
        scale = dominant
        return b_arr / scale, orbit_vec, coeffs_arr / scale, scale

    norm = max(float(np.linalg.norm(b_arr)), _EPS)
    b_unit = b_arr / norm
    return b_unit, b_unit.copy(), coeffs_arr / norm, norm


def _normal_form_for_output_action(
    output_action: OutputActionSpec,
    invariants: Sequence[CoordinateSpec],
    orbits: Sequence[CoordinateSpec],
) -> OutputNormalFormSpec:
    alpha = float(output_action.alpha)
    beta = float(output_action.beta)
    orbit_name = str(orbits[0].name) if orbits else None
    invariant_names = tuple(str(coord.name) for coord in invariants)
    z_label = ", ".join(invariant_names) if invariant_names else "z"
    s_label = orbit_name or "s"
    h_label = "H"
    assumptions = ("orbit coordinate is gauged so X s = 1",)
    if abs(beta) > _EPS:
        shift = alpha / beta
        return OutputNormalFormSpec(
            kind="multiplicative_prefactor",
            reduced_target_name=h_label,
            orbit_coordinate=orbit_name,
            invariant_coordinates=invariant_names,
            alpha=alpha,
            beta=beta,
            target_formula=f"f = {-shift:.12g} + exp({beta:.12g} * {s_label}) * {h_label}({z_label})",
            reduced_target_formula=f"{h_label} = exp(-{beta:.12g} * {s_label}) * (f + {shift:.12g})",
            assumptions=assumptions,
            provenance={"source": "output_action", "equation": "X f = alpha + beta f"},
        )
    if abs(alpha) > _EPS:
        return OutputNormalFormSpec(
            kind="additive_cocycle",
            reduced_target_name=h_label,
            orbit_coordinate=orbit_name,
            invariant_coordinates=invariant_names,
            alpha=alpha,
            beta=beta,
            target_formula=f"f = {alpha:.12g} * {s_label} + {h_label}({z_label})",
            reduced_target_formula=f"{h_label} = f - {alpha:.12g} * {s_label}",
            assumptions=assumptions,
            provenance={"source": "output_action", "equation": "X f = alpha"},
        )
    return OutputNormalFormSpec(
        kind="invariant_residual",
        reduced_target_name=h_label,
        orbit_coordinate=orbit_name,
        invariant_coordinates=invariant_names,
        alpha=0.0,
        beta=0.0,
        target_formula=f"f = {h_label}({z_label})",
        reduced_target_formula=f"{h_label} = f",
        assumptions=("output is invariant along the compiled orbit",),
        provenance={"source": "output_action", "equation": "X f = 0"},
    )


def _is_common_diagonal_scaling(A: np.ndarray, b: np.ndarray, center: np.ndarray) -> bool:
    del center
    scale = max(1.0, float(np.linalg.norm(A)))
    offdiag = A - np.diag(np.diag(A))
    return (
        np.linalg.norm(b) <= 1.0e-7 * scale
        and np.linalg.norm(offdiag) <= 1.0e-7 * scale
        and abs(float(A[0, 0] - A[1, 1])) <= 1.0e-7 * scale
        and abs(float(A[0, 0])) > 1.0e-10
    )


def _is_rotation(A: np.ndarray, b: np.ndarray, center: np.ndarray) -> bool:
    del center
    scale = max(1.0, float(np.linalg.norm(A)))
    return (
        np.linalg.norm(b) <= 1.0e-7 * scale
        and np.linalg.norm(A + A.T) <= 1.0e-7 * scale
        and abs(float(A[1, 0] - A[0, 1])) > 1.0e-10
    )


def _fixed_point(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    try:
        if np.linalg.matrix_rank(A, tol=1.0e-10) == A.shape[0]:
            return -np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return None


def _orthogonal_complement(row_basis: np.ndarray, n: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(np.asarray(row_basis, dtype=float), full_matrices=True)
    del U
    tol = max(1.0e-11, 1.0e-9 * max(1.0, float(s[0]) if s.size else 0.0))
    rank = int(np.sum(s > tol))
    return Vt[rank:].reshape(-1, n)


def _shifted_var(idx: int, center_value: float) -> Node:
    if abs(float(center_value)) <= 1.0e-12:
        return Var(int(idx))
    return AddNode(Var(int(idx)), ConstNode(-float(center_value)))


def _linear_ast(coeffs: Sequence[float]) -> Node:
    terms: list[Node] = []
    for i, coeff in enumerate(coeffs):
        c = float(coeff)
        if abs(c) <= 1.0e-12:
            continue
        var = Var(i)
        terms.append(var if abs(c - 1.0) <= 1.0e-12 else MulNode(ConstNode(c), var))
    if not terms:
        return ConstNode(0.0)
    expr = terms[0]
    for term in terms[1:]:
        expr = AddNode(expr, term)
    return expr


def _zero_condition(idx: int, center_value: float) -> str:
    if abs(float(center_value)) <= 1.0e-12:
        return f"x{idx} == 0"
    return f"x{idx} - ({float(center_value):.12g}) == 0"


def _center_condition(center: np.ndarray) -> str:
    if np.linalg.norm(center) <= 1.0e-12:
        return "origin"
    pieces = ", ".join(f"x{i}={float(v):.12g}" for i, v in enumerate(center))
    return f"affine center ({pieces})"


def _compose_domain_with_inputs(domain: DomainSpec, local_inputs: Sequence[Node]) -> DomainSpec:
    input_labels = tuple(ast_to_human_readable(inp) for inp in local_inputs)
    assumptions = tuple(domain.assumptions) + (f"local_inputs={input_labels}",)
    return DomainSpec(
        assumptions=assumptions,
        exclusions=tuple(domain.exclusions),
        branches=tuple(domain.branches),
    )


def _compact(s: str) -> str:
    return "".join(str(s).lower().split())


def _default_sample(input_dim: int) -> np.ndarray:
    if input_dim == 1:
        return np.asarray([[0.25], [1.0]], dtype=float)
    base = np.eye(int(input_dim), dtype=float)
    return np.concatenate([base, 0.5 * np.ones((1, int(input_dim)), dtype=float)], axis=0)


def _require_orbit_values(orbit_values: Any | None) -> np.ndarray:
    if orbit_values is None:
        raise ValueError("orbit_values are required for an output-equivariant normal form")
    return np.asarray(orbit_values, dtype=float).reshape(-1)
