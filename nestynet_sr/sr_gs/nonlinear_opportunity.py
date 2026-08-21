# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Matched opportunity accounting for nonlinear DE symmetries.

This module deliberately does *not* discover generators.  It consumes generic
report mappings from an affine solver, a neutral carrier baseline, and a
nonlinear-symmetry solver and asks the narrower scientific question: did the
symmetry labels add value beyond access to the same carrier vocabulary?

The three arms are:

``A``
    Current affine-in-adapted-charts evidence and carriers.
``B``
    A neutral baseline given the extra carrier vocabulary, without symmetry
    labels.
``C``
    Quadratic generalized symmetries and their certified generated carriers.

Arm C is credited only after (1) B/C vocabulary identity, (2) absolute
determining/off-shell/stability/invariant gates, and (3) configured gains over
the matched B arm in invariant recovery, problem reduction, and held-out
performance.  A small deterministic scalar-ODE opportunity registry is
included for tests and downstream experiment drivers; this module does not run
benchmarks or write results to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


_EPS = 1.0e-12
_QUALITY_METRICS = (
    "determining_residual",
    "subspace_stability",
    "false_generator_rate",
)
_DOWNSTREAM_METRICS = (
    "simple_invariant_recovery",
    "arity_reduction",
    "order_reduction",
    "heldout_equation_gain",
    "heldout_rollout_gain",
)
_ALL_METRICS = _QUALITY_METRICS + _DOWNSTREAM_METRICS


@runtime_checkable
class OpportunityReport(Protocol):
    """Structural protocol accepted by :meth:`OpportunityArmMetrics.from_report`."""

    def to_report(self) -> Mapping[str, Any]:
        """Return a JSON-like report mapping."""


def _finite_optional(value: Any, *, name: str, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite or None")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _bounded_rate(value: Any, *, name: str) -> float | None:
    out = _finite_optional(value, name=name)
    if out is not None and not (0.0 <= out <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return out


def _canonical_vocabulary(values: Sequence[Any] | None) -> tuple[str, ...]:
    raw = tuple(str(value).strip() for value in (values or ()))
    if any(not value for value in raw):
        raise ValueError("carrier vocabulary IDs must be non-empty")
    if len(set(raw)) != len(raw):
        raise ValueError("carrier vocabulary IDs must be unique")
    return tuple(sorted(raw))


def _as_mapping(report: Mapping[str, Any] | OpportunityReport) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    method = getattr(report, "to_report", None)
    if not callable(method):
        raise TypeError("arm report must be a mapping or expose to_report()")
    payload = method()
    if not isinstance(payload, Mapping):
        raise TypeError("to_report() must return a mapping")
    return payload


def _nested_get(payload: Mapping[str, Any], name: str, default: Any = None) -> Any:
    if name in payload:
        return payload[name]
    for container_name in ("metrics", "scores", "evidence"):
        container = payload.get(container_name)
        if isinstance(container, Mapping) and name in container:
            return container[name]
    return default


@dataclass(frozen=True)
class OpportunityArmMetrics:
    """Normalized metrics for one arm of a matched opportunity experiment.

    Reductions and gains use a higher-is-better convention.  They may be
    negative when an arm makes the discovery problem worse.  The three
    generator-quality metrics are optional because a neutral baseline need not
    expose a determining operator.
    """

    arm: str
    extra_carrier_vocabulary: tuple[str, ...] = ()
    determining_residual: float | None = None
    subspace_stability: float | None = None
    false_generator_rate: float | None = None
    simple_invariant_recovery: float = 0.0
    arity_reduction: float = 0.0
    order_reduction: float = 0.0
    heldout_equation_gain: float = 0.0
    heldout_rollout_gain: float = 0.0
    generator_discovered: bool | None = None
    certificates: Mapping[str, bool] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arm = str(self.arm).strip().upper()
        if arm not in {"A", "B", "C"}:
            raise ValueError("arm must be one of A, B, or C")
        object.__setattr__(self, "arm", arm)
        object.__setattr__(
            self,
            "extra_carrier_vocabulary",
            _canonical_vocabulary(self.extra_carrier_vocabulary),
        )
        object.__setattr__(
            self,
            "determining_residual",
            _finite_optional(
                self.determining_residual,
                name="determining_residual",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "subspace_stability",
            _bounded_rate(self.subspace_stability, name="subspace_stability"),
        )
        object.__setattr__(
            self,
            "false_generator_rate",
            _bounded_rate(self.false_generator_rate, name="false_generator_rate"),
        )
        object.__setattr__(
            self,
            "simple_invariant_recovery",
            float(
                _bounded_rate(
                    self.simple_invariant_recovery,
                    name="simple_invariant_recovery",
                )
            ),
        )
        for name in (
            "arity_reduction",
            "order_reduction",
            "heldout_equation_gain",
            "heldout_rollout_gain",
        ):
            object.__setattr__(self, name, float(_finite_optional(getattr(self, name), name=name)))
        object.__setattr__(
            self,
            "certificates",
            {str(key): bool(value) for key, value in dict(self.certificates).items()},
        )
        object.__setattr__(self, "evidence", dict(self.evidence))

    @classmethod
    def from_report(
        cls,
        report: "OpportunityArmMetrics | Mapping[str, Any] | OpportunityReport",
        *,
        expected_arm: str | None = None,
    ) -> "OpportunityArmMetrics":
        """Coerce a solver-independent flat or nested report mapping.

        Metrics may live at the top level or under ``metrics``, ``scores``, or
        ``evidence``.  ``carrier_vocabulary`` is accepted as an alias for
        ``extra_carrier_vocabulary``.
        """

        if isinstance(report, cls):
            out = report
        else:
            payload = _as_mapping(report)
            arm = _nested_get(payload, "arm", expected_arm)
            if arm is None:
                raise ValueError("arm report is missing its arm label")
            vocabulary = _nested_get(payload, "extra_carrier_vocabulary", None)
            if vocabulary is None:
                vocabulary = _nested_get(payload, "carrier_vocabulary", ())
            certificates = payload.get("certificates", {})
            if not isinstance(certificates, Mapping):
                raise TypeError("certificates must be a mapping")
            generator_discovered = _nested_get(payload, "generator_discovered", None)
            if generator_discovered is None:
                accepted_count = _nested_get(payload, "accepted_generator_count", None)
                if accepted_count is not None:
                    generator_discovered = int(accepted_count) > 0
                else:
                    accepted = _nested_get(payload, "accepted_generators", None)
                    if isinstance(accepted, Sequence) and not isinstance(accepted, (str, bytes)):
                        generator_discovered = len(accepted) > 0
            out = cls(
                arm=str(arm),
                extra_carrier_vocabulary=tuple(vocabulary or ()),
                determining_residual=_nested_get(payload, "determining_residual", None),
                subspace_stability=_nested_get(payload, "subspace_stability", None),
                false_generator_rate=_nested_get(payload, "false_generator_rate", None),
                simple_invariant_recovery=_nested_get(
                    payload,
                    "simple_invariant_recovery",
                    0.0,
                ),
                arity_reduction=_nested_get(payload, "arity_reduction", 0.0),
                order_reduction=_nested_get(payload, "order_reduction", 0.0),
                heldout_equation_gain=_nested_get(payload, "heldout_equation_gain", 0.0),
                heldout_rollout_gain=_nested_get(payload, "heldout_rollout_gain", 0.0),
                generator_discovered=(
                    None if generator_discovered is None else bool(generator_discovered)
                ),
                certificates=certificates,
                evidence=(
                    payload.get("evidence", {})
                    if isinstance(payload.get("evidence", {}), Mapping)
                    else {}
                ),
            )
        if expected_arm is not None and out.arm != str(expected_arm).strip().upper():
            raise ValueError(f"expected arm {expected_arm!r}, received arm {out.arm!r}")
        return out

    def to_report(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "extra_carrier_vocabulary": list(self.extra_carrier_vocabulary),
            "metrics": {name: getattr(self, name) for name in _ALL_METRICS},
            "generator_discovered": self.generator_discovered,
            "certificates": dict(self.certificates),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class OpportunityAttributionConfig:
    """Certificate thresholds and matched-B attribution margins."""

    required_certificates: tuple[str, ...] = (
        "determining",
        "off_shell",
        "bootstrap",
        "flow",
        "invariant",
        "support",
    )
    require_generator_detection: bool = True
    require_extra_vocabulary: bool = True
    max_determining_residual: float = 1.0e-6
    min_subspace_stability: float = 0.90
    max_false_generator_rate: float = 0.05
    min_simple_invariant_recovery: float = 0.50
    min_invariant_gain_vs_b: float = 0.05
    min_reduction_gain_vs_b: float = 0.50
    min_heldout_gain_vs_b: float = 0.01
    max_downstream_regression: float = 0.0
    min_determining_residual_reduction_vs_b: float | None = None
    min_subspace_stability_gain_vs_b: float | None = None
    min_false_generator_rate_reduction_vs_b: float | None = None

    def __post_init__(self) -> None:
        if self.max_determining_residual < 0.0:
            raise ValueError("max_determining_residual must be nonnegative")
        for name in (
            "min_subspace_stability",
            "max_false_generator_rate",
            "min_simple_invariant_recovery",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "min_invariant_gain_vs_b",
            "min_reduction_gain_vs_b",
            "min_heldout_gain_vs_b",
            "max_downstream_regression",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class OpportunitySampleBatch:
    """Deterministic synthetic scalar-jet samples for one opportunity case."""

    case_id: str
    on_shell: Mapping[str, np.ndarray]
    off_shell: Mapping[str, np.ndarray]
    noise_std: float = 0.0
    support_fraction: float = 1.0
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.noise_std) < 0.0:
            raise ValueError("noise_std must be nonnegative")
        if not (0.0 < float(self.support_fraction) <= 1.0):
            raise ValueError("support_fraction must lie in (0, 1]")
        on_shell = {str(k): np.asarray(v, dtype=float).copy() for k, v in self.on_shell.items()}
        off_shell = {str(k): np.asarray(v, dtype=float).copy() for k, v in self.off_shell.items()}
        if not on_shell or not off_shell:
            raise ValueError("both on-shell and off-shell samples are required")
        if len({value.shape[0] for value in on_shell.values()}) != 1:
            raise ValueError("on-shell sample arrays must have a common first dimension")
        if len({value.shape[0] for value in off_shell.values()}) != 1:
            raise ValueError("off-shell sample arrays must have a common first dimension")
        object.__setattr__(self, "on_shell", on_shell)
        object.__setattr__(self, "off_shell", off_shell)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_on_shell(self) -> int:
        return int(next(iter(self.on_shell.values())).shape[0])

    @property
    def n_off_shell(self) -> int:
        return int(next(iter(self.off_shell.values())).shape[0])

    def to_report(self, *, include_samples: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {
            "case_id": self.case_id,
            "n_on_shell": self.n_on_shell,
            "n_off_shell": self.n_off_shell,
            "noise_std": float(self.noise_std),
            "support_fraction": float(self.support_fraction),
            "seed": int(self.seed),
            "metadata": dict(self.metadata),
        }
        if include_samples:
            report["on_shell"] = {key: value.tolist() for key, value in self.on_shell.items()}
            report["off_shell"] = {key: value.tolist() for key, value in self.off_shell.items()}
        return report


OpportunitySampler = Callable[[int, int, float, float], OpportunitySampleBatch]


@dataclass(frozen=True)
class OpportunityCaseSpec:
    """One bounded opportunity case, independent of any solver implementation."""

    case_id: str
    title: str
    scope: str
    ode_order: int | None
    expected_opportunity: bool
    expected_structure: str
    sampler: OpportunitySampler | None
    tags: tuple[str, ...] = ()
    deferred_reason: str = ""

    @property
    def deferred(self) -> bool:
        return self.sampler is None

    def sample(
        self,
        *,
        n_samples: int = 128,
        seed: int = 0,
        noise_std: float = 0.0,
        support_fraction: float = 1.0,
    ) -> OpportunitySampleBatch:
        if self.sampler is None:
            raise ValueError(f"case {self.case_id!r} is deferred: {self.deferred_reason}")
        if int(n_samples) < 8:
            raise ValueError("n_samples must be at least 8")
        return self.sampler(
            int(n_samples),
            int(seed),
            float(noise_std),
            float(support_fraction),
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "scope": self.scope,
            "ode_order": self.ode_order,
            "expected_opportunity": bool(self.expected_opportunity),
            "expected_structure": self.expected_structure,
            "tags": list(self.tags),
            "deferred": self.deferred,
            "deferred_reason": self.deferred_reason,
        }


def _noise_jet(
    rng: np.random.Generator,
    jet: Mapping[str, np.ndarray],
    noise_std: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in jet.items():
        arr = np.asarray(value, dtype=float)
        if noise_std <= 0.0 or key == "x":
            out[key] = arr.copy()
            continue
        scale = max(float(np.sqrt(np.mean(arr * arr))), 1.0)
        out[key] = arr + float(noise_std) * scale * rng.standard_normal(arr.shape)
    return out


def _free_particle_samples(
    n: int,
    seed: int,
    noise_std: float,
    support_fraction: float,
) -> OpportunitySampleBatch:
    rng = np.random.default_rng(seed)
    width = 2.0 * support_fraction
    x = rng.uniform(-width, width, n)
    slope = rng.uniform(-1.5, 1.5, n)
    intercept = rng.uniform(-1.0, 1.0, n)
    on = {"x": x, "u": slope * x + intercept, "u_x": slope, "u_xx": np.zeros(n)}
    off = {
        "x": rng.uniform(-2.0, 2.0, n),
        "u": rng.uniform(-2.0, 2.0, n),
        "u_x": rng.uniform(-1.5, 1.5, n),
        "u_xx": rng.uniform(-2.0, 2.0, n),
    }
    return OpportunitySampleBatch(
        case_id="free_particle_projective",
        on_shell=_noise_jet(rng, on, noise_std),
        off_shell=off,
        noise_std=noise_std,
        support_fraction=support_fraction,
        seed=seed,
        metadata={"residual": "u_xx", "family": "projective"},
    )


def _riccati_samples(
    n: int,
    seed: int,
    noise_std: float,
    support_fraction: float,
) -> OpportunitySampleBatch:
    rng = np.random.default_rng(seed)
    width = 0.8 * support_fraction
    x = rng.uniform(-width, width, n)
    u = rng.uniform(-1.25, 1.25, n)
    on = {"x": x, "u": u, "u_x": u * u}
    off = {
        "x": rng.uniform(-0.8, 0.8, n),
        "u": rng.uniform(-1.25, 1.25, n),
        "u_x": rng.uniform(-2.0, 2.0, n),
    }
    return OpportunitySampleBatch(
        case_id="riccati_mobius",
        on_shell=_noise_jet(rng, on, noise_std),
        off_shell=off,
        noise_std=noise_std,
        support_fraction=support_fraction,
        seed=seed,
        metadata={"residual": "u_x-u^2", "family": "mobius"},
    )


def _generic_control_samples(
    n: int,
    seed: int,
    noise_std: float,
    support_fraction: float,
) -> OpportunitySampleBatch:
    rng = np.random.default_rng(seed)
    width = 1.5 * support_fraction
    x = rng.uniform(-width, width, n)
    u = rng.uniform(-1.25, 1.25, n)
    u_x = rng.uniform(-1.0, 1.0, n)
    u_xx = np.sin(x) + u**3 + x * u_x + 0.2 * np.sin(u_x)
    on = {"x": x, "u": u, "u_x": u_x, "u_xx": u_xx}
    off = {
        "x": rng.uniform(-1.5, 1.5, n),
        "u": rng.uniform(-1.25, 1.25, n),
        "u_x": rng.uniform(-1.0, 1.0, n),
        "u_xx": rng.uniform(-3.0, 3.0, n),
    }
    return OpportunitySampleBatch(
        case_id="generic_negative_control",
        on_shell=_noise_jet(rng, on, noise_std),
        off_shell=off,
        noise_std=noise_std,
        support_fraction=support_fraction,
        seed=seed,
        metadata={
            "residual": "u_xx-sin(x)-u^3-x*u_x-0.2*sin(u_x)",
            "family": "generic_control",
        },
    )


OPPORTUNITY_CASE_REGISTRY: Mapping[str, OpportunityCaseSpec] = {
    "free_particle_projective": OpportunityCaseSpec(
        case_id="free_particle_projective",
        title="Free particle projective opportunity",
        scope="scalar_ode",
        ode_order=2,
        expected_opportunity=True,
        expected_structure="quadratic/projective point generators",
        sampler=_free_particle_samples,
        tags=("projective", "quadratic", "positive"),
    ),
    "riccati_mobius": OpportunityCaseSpec(
        case_id="riccati_mobius",
        title="Riccati Möbius opportunity",
        scope="scalar_ode",
        ode_order=1,
        expected_opportunity=True,
        expected_structure="fractional-linear/Mobius carrier geometry",
        sampler=_riccati_samples,
        tags=("riccati", "mobius", "positive"),
    ),
    "generic_negative_control": OpportunityCaseSpec(
        case_id="generic_negative_control",
        title="Generic nonlinear scalar-ODE control",
        scope="scalar_ode",
        ode_order=2,
        expected_opportunity=False,
        expected_structure="no stable low-complexity quadratic generator",
        sampler=_generic_control_samples,
        tags=("negative_control",),
    ),
    "wave_conformal_deferred": OpportunityCaseSpec(
        case_id="wave_conformal_deferred",
        title="Wave-equation conformal opportunity",
        scope="coupled_pde",
        ode_order=None,
        expected_opportunity=True,
        expected_structure="special conformal PDE generators",
        sampler=None,
        tags=("conformal", "pde", "deferred"),
        deferred_reason="coupled/PDE prolongation is outside the scalar-ODE prototype",
    ),
    "schrodinger_conformal_deferred": OpportunityCaseSpec(
        case_id="schrodinger_conformal_deferred",
        title="Schrödinger conformal opportunity",
        scope="coupled_pde",
        ode_order=None,
        expected_opportunity=True,
        expected_structure="Schrodinger-group differential invariants",
        sampler=None,
        tags=("schrodinger", "pde", "deferred"),
        deferred_reason="coupled/PDE prolongation is outside the scalar-ODE prototype",
    ),
}


def get_opportunity_case(case_id: str) -> OpportunityCaseSpec:
    """Return one registered opportunity case or raise a descriptive error."""

    key = str(case_id).strip()
    try:
        return OPPORTUNITY_CASE_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(OPPORTUNITY_CASE_REGISTRY))
        raise KeyError(f"unknown opportunity case {key!r}; known cases: {known}") from exc


def list_opportunity_cases(*, include_deferred: bool = False) -> tuple[OpportunityCaseSpec, ...]:
    """List deterministic registry specs without running any experiment."""

    cases = tuple(OPPORTUNITY_CASE_REGISTRY[key] for key in sorted(OPPORTUNITY_CASE_REGISTRY))
    if include_deferred:
        return cases
    return tuple(case for case in cases if not case.deferred)


def sample_opportunity_case(
    case_id: str,
    *,
    n_samples: int = 128,
    seed: int = 0,
    noise_std: float = 0.0,
    support_fraction: float = 1.0,
) -> OpportunitySampleBatch:
    """Sample a registered case deterministically, without invoking a solver."""

    return get_opportunity_case(case_id).sample(
        n_samples=n_samples,
        seed=seed,
        noise_std=noise_std,
        support_fraction=support_fraction,
    )


def _case_spec(case: str | OpportunityCaseSpec) -> OpportunityCaseSpec:
    return get_opportunity_case(case) if isinstance(case, str) else case


def _positive_delta(name: str, reference: OpportunityArmMetrics, candidate: OpportunityArmMetrics) -> float | None:
    a = getattr(reference, name)
    c = getattr(candidate, name)
    if a is None or c is None:
        return None
    if name in {"determining_residual", "false_generator_rate"}:
        return float(a) - float(c)
    return float(c) - float(a)


def _metric_deltas(reference: OpportunityArmMetrics, candidate: OpportunityArmMetrics) -> dict[str, float | None]:
    return {name: _positive_delta(name, reference, candidate) for name in _ALL_METRICS}


@dataclass(frozen=True)
class OpportunityAttributionResult:
    """One matched A/B/C attribution decision and its complete gate trace."""

    case_id: str
    condition_id: str
    expected_opportunity: bool
    status: str
    credited: bool
    vocabulary_matched: bool
    certificate_gates_passed: bool
    comparison_gates_passed: bool
    negative_control_passed: bool | None
    noise_level: float
    support_fraction: float
    arm_a: OpportunityArmMetrics
    arm_b: OpportunityArmMetrics
    arm_c: OpportunityArmMetrics
    deltas_vs_b: Mapping[str, float | None]
    deltas_vs_a: Mapping[str, float | None]
    certificate_gates: Mapping[str, bool]
    comparison_gates: Mapping[str, bool]
    reasons: tuple[str, ...] = ()

    def to_report(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "condition_id": self.condition_id,
            "expected_opportunity": bool(self.expected_opportunity),
            "status": self.status,
            "credited": bool(self.credited),
            "vocabulary_matched": bool(self.vocabulary_matched),
            "certificate_gates_passed": bool(self.certificate_gates_passed),
            "comparison_gates_passed": bool(self.comparison_gates_passed),
            "negative_control_passed": self.negative_control_passed,
            "noise_level": float(self.noise_level),
            "support_fraction": float(self.support_fraction),
            "arms": {
                "A": self.arm_a.to_report(),
                "B": self.arm_b.to_report(),
                "C": self.arm_c.to_report(),
            },
            "deltas_vs_b": dict(self.deltas_vs_b),
            "deltas_vs_a": dict(self.deltas_vs_a),
            "certificate_gates": dict(self.certificate_gates),
            "comparison_gates": dict(self.comparison_gates),
            "reasons": list(self.reasons),
        }


def evaluate_matched_opportunity(
    case: str | OpportunityCaseSpec,
    arm_a: OpportunityArmMetrics | Mapping[str, Any] | OpportunityReport,
    arm_b: OpportunityArmMetrics | Mapping[str, Any] | OpportunityReport,
    arm_c: OpportunityArmMetrics | Mapping[str, Any] | OpportunityReport,
    *,
    config: OpportunityAttributionConfig | None = None,
    condition_id: str = "clean",
    noise_level: float = 0.0,
    support_fraction: float = 1.0,
) -> OpportunityAttributionResult:
    """Evaluate one matched experiment without running any search engine.

    All deltas use a positive-means-C-is-better convention.  ``B`` and ``C``
    must declare exactly the same extra carrier IDs; a mismatch produces an
    invalid, never-credited result rather than a misleading comparison.
    """

    spec = _case_spec(case)
    cfg = config or OpportunityAttributionConfig()
    a = OpportunityArmMetrics.from_report(arm_a, expected_arm="A")
    b = OpportunityArmMetrics.from_report(arm_b, expected_arm="B")
    c = OpportunityArmMetrics.from_report(arm_c, expected_arm="C")
    noise_level = float(noise_level)
    support_fraction = float(support_fraction)
    if noise_level < 0.0:
        raise ValueError("noise_level must be nonnegative")
    if not (0.0 < support_fraction <= 1.0):
        raise ValueError("support_fraction must lie in (0, 1]")

    deltas_b = _metric_deltas(b, c)
    deltas_a = _metric_deltas(a, c)
    vocabulary_matched = b.extra_carrier_vocabulary == c.extra_carrier_vocabulary
    vocabulary_nonempty = bool(c.extra_carrier_vocabulary) or not cfg.require_extra_vocabulary
    reasons: list[str] = []
    if not vocabulary_matched:
        reasons.append("unmatched_extra_carrier_vocabulary")
    if not vocabulary_nonempty:
        reasons.append("missing_extra_carrier_vocabulary")

    if spec.deferred:
        reasons.append(f"deferred:{spec.deferred_reason}")
        return OpportunityAttributionResult(
            case_id=spec.case_id,
            condition_id=str(condition_id),
            expected_opportunity=bool(spec.expected_opportunity),
            status="deferred",
            credited=False,
            vocabulary_matched=vocabulary_matched,
            certificate_gates_passed=False,
            comparison_gates_passed=False,
            negative_control_passed=None,
            noise_level=noise_level,
            support_fraction=support_fraction,
            arm_a=a,
            arm_b=b,
            arm_c=c,
            deltas_vs_b=deltas_b,
            deltas_vs_a=deltas_a,
            certificate_gates={},
            comparison_gates={},
            reasons=tuple(reasons),
        )

    if not vocabulary_matched or not vocabulary_nonempty:
        return OpportunityAttributionResult(
            case_id=spec.case_id,
            condition_id=str(condition_id),
            expected_opportunity=bool(spec.expected_opportunity),
            status="invalid",
            credited=False,
            vocabulary_matched=vocabulary_matched,
            certificate_gates_passed=False,
            comparison_gates_passed=False,
            negative_control_passed=None,
            noise_level=noise_level,
            support_fraction=support_fraction,
            arm_a=a,
            arm_b=b,
            arm_c=c,
            deltas_vs_b=deltas_b,
            deltas_vs_a=deltas_a,
            certificate_gates={"vocabulary_matched": vocabulary_matched},
            comparison_gates={},
            reasons=tuple(reasons),
        )

    if not spec.expected_opportunity:
        false_rate_ok = (
            c.false_generator_rate is not None
            and c.false_generator_rate <= cfg.max_false_generator_rate
        )
        rejection_ok = c.generator_discovered is False
        negative_passed = bool(false_rate_ok and rejection_ok)
        if not rejection_ok:
            reasons.append("negative_control_generator_not_rejected")
        if not false_rate_ok:
            reasons.append("negative_control_false_generator_rate_failed")
        gates = {
            "vocabulary_matched": True,
            "generator_rejected": rejection_ok,
            "false_generator_rate": false_rate_ok,
        }
        return OpportunityAttributionResult(
            case_id=spec.case_id,
            condition_id=str(condition_id),
            expected_opportunity=False,
            status=("negative_control_passed" if negative_passed else "negative_control_failed"),
            credited=False,
            vocabulary_matched=True,
            certificate_gates_passed=negative_passed,
            comparison_gates_passed=True,
            negative_control_passed=negative_passed,
            noise_level=noise_level,
            support_fraction=support_fraction,
            arm_a=a,
            arm_b=b,
            arm_c=c,
            deltas_vs_b=deltas_b,
            deltas_vs_a=deltas_a,
            certificate_gates=gates,
            comparison_gates={},
            reasons=tuple(reasons),
        )

    certificate_gates: dict[str, bool] = {
        "vocabulary_matched": True,
        "generator_discovered": (
            c.generator_discovered is True if cfg.require_generator_detection else True
        ),
        "determining_residual": (
            c.determining_residual is not None
            and c.determining_residual <= cfg.max_determining_residual
        ),
        "subspace_stability": (
            c.subspace_stability is not None
            and c.subspace_stability >= cfg.min_subspace_stability
        ),
        "false_generator_rate": (
            c.false_generator_rate is not None
            and c.false_generator_rate <= cfg.max_false_generator_rate
        ),
        "simple_invariant_recovery": (
            c.simple_invariant_recovery >= cfg.min_simple_invariant_recovery
        ),
    }
    for name in cfg.required_certificates:
        certificate_gates[f"certificate:{name}"] = bool(c.certificates.get(name, False))

    invariant_delta = float(deltas_b["simple_invariant_recovery"] or 0.0)
    reduction_delta = max(
        float(deltas_b["arity_reduction"] or 0.0),
        float(deltas_b["order_reduction"] or 0.0),
    )
    heldout_delta = max(
        float(deltas_b["heldout_equation_gain"] or 0.0),
        float(deltas_b["heldout_rollout_gain"] or 0.0),
    )
    comparison_gates: dict[str, bool] = {
        "invariant_gain_vs_b": invariant_delta >= cfg.min_invariant_gain_vs_b,
        "reduction_gain_vs_b": reduction_delta >= cfg.min_reduction_gain_vs_b,
        "heldout_gain_vs_b": heldout_delta >= cfg.min_heldout_gain_vs_b,
        "no_downstream_regression": all(
            float(deltas_b[name] or 0.0) >= -cfg.max_downstream_regression
            for name in _DOWNSTREAM_METRICS
        ),
    }

    optional_pairwise = (
        (
            "determining_residual_reduction_vs_b",
            "determining_residual",
            cfg.min_determining_residual_reduction_vs_b,
        ),
        (
            "subspace_stability_gain_vs_b",
            "subspace_stability",
            cfg.min_subspace_stability_gain_vs_b,
        ),
        (
            "false_generator_rate_reduction_vs_b",
            "false_generator_rate",
            cfg.min_false_generator_rate_reduction_vs_b,
        ),
    )
    for gate_name, metric_name, margin in optional_pairwise:
        if margin is None:
            continue
        delta = deltas_b[metric_name]
        comparison_gates[gate_name] = delta is not None and float(delta) >= float(margin)

    for name, passed in certificate_gates.items():
        if not passed:
            reasons.append(f"certificate_gate_failed:{name}")
    for name, passed in comparison_gates.items():
        if not passed:
            reasons.append(f"comparison_gate_failed:{name}")
    certificate_passed = all(certificate_gates.values())
    comparison_passed = all(comparison_gates.values())
    credited = bool(certificate_passed and comparison_passed)
    return OpportunityAttributionResult(
        case_id=spec.case_id,
        condition_id=str(condition_id),
        expected_opportunity=True,
        status="credited" if credited else "not_credited",
        credited=credited,
        vocabulary_matched=True,
        certificate_gates_passed=certificate_passed,
        comparison_gates_passed=comparison_passed,
        negative_control_passed=None,
        noise_level=noise_level,
        support_fraction=support_fraction,
        arm_a=a,
        arm_b=b,
        arm_c=c,
        deltas_vs_b=deltas_b,
        deltas_vs_a=deltas_a,
        certificate_gates=certificate_gates,
        comparison_gates=comparison_gates,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class OpportunityAggregateConfig:
    """Robustness thresholds for clean, noisy, and restricted-support trials."""

    min_overall_credit_rate: float = 0.75
    min_clean_credit_rate: float = 1.0
    min_noisy_credit_rate: float = 0.50
    min_restricted_credit_rate: float = 0.50
    require_clean_condition: bool = True
    require_noisy_condition: bool = False
    require_restricted_condition: bool = False
    require_negative_control: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_overall_credit_rate",
            "min_clean_credit_rate",
            "min_noisy_credit_rate",
            "min_restricted_credit_rate",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")


def _credit_rate(rows: Sequence[OpportunityAttributionResult]) -> float | None:
    if not rows:
        return None
    return float(sum(bool(row.credited) for row in rows) / len(rows))


@dataclass(frozen=True)
class OpportunityAggregateResult:
    """Aggregate opportunity evidence across support and noise conditions."""

    robustly_attributed: bool
    n_results: int
    n_positive_trials: int
    n_negative_controls: int
    overall_credit_rate: float | None
    clean_credit_rate: float | None
    noisy_credit_rate: float | None
    restricted_credit_rate: float | None
    negative_control_pass_rate: float | None
    gates: Mapping[str, bool]
    case_credit_rates: Mapping[str, float | None]
    reasons: tuple[str, ...] = ()

    def to_report(self) -> dict[str, Any]:
        return {
            "robustly_attributed": bool(self.robustly_attributed),
            "n_results": int(self.n_results),
            "n_positive_trials": int(self.n_positive_trials),
            "n_negative_controls": int(self.n_negative_controls),
            "overall_credit_rate": self.overall_credit_rate,
            "clean_credit_rate": self.clean_credit_rate,
            "noisy_credit_rate": self.noisy_credit_rate,
            "restricted_credit_rate": self.restricted_credit_rate,
            "negative_control_pass_rate": self.negative_control_pass_rate,
            "gates": dict(self.gates),
            "case_credit_rates": dict(self.case_credit_rates),
            "reasons": list(self.reasons),
        }


def aggregate_opportunity_results(
    results: Sequence[OpportunityAttributionResult],
    *,
    config: OpportunityAggregateConfig | None = None,
) -> OpportunityAggregateResult:
    """Aggregate matched decisions across clean/noisy/restricted conditions."""

    cfg = config or OpportunityAggregateConfig()
    rows = tuple(results)
    positive = tuple(
        row for row in rows if row.expected_opportunity and row.status not in {"invalid", "deferred"}
    )
    negative = tuple(row for row in rows if not row.expected_opportunity)
    clean = tuple(
        row
        for row in positive
        if row.noise_level <= _EPS and row.support_fraction >= 1.0 - _EPS
    )
    noisy = tuple(row for row in positive if row.noise_level > _EPS)
    restricted = tuple(row for row in positive if row.support_fraction < 1.0 - _EPS)

    overall_rate = _credit_rate(positive)
    clean_rate = _credit_rate(clean)
    noisy_rate = _credit_rate(noisy)
    restricted_rate = _credit_rate(restricted)
    negative_rate = (
        None
        if not negative
        else float(sum(row.negative_control_passed is True for row in negative) / len(negative))
    )
    gates = {
        "positive_trials_present": bool(positive),
        "overall_credit_rate": (
            overall_rate is not None and overall_rate >= cfg.min_overall_credit_rate
        ),
        "clean_credit_rate": (
            (clean_rate is not None and clean_rate >= cfg.min_clean_credit_rate)
            if (clean or cfg.require_clean_condition)
            else True
        ),
        "noisy_credit_rate": (
            (noisy_rate is not None and noisy_rate >= cfg.min_noisy_credit_rate)
            if (noisy or cfg.require_noisy_condition)
            else True
        ),
        "restricted_credit_rate": (
            (
                restricted_rate is not None
                and restricted_rate >= cfg.min_restricted_credit_rate
            )
            if (restricted or cfg.require_restricted_condition)
            else True
        ),
        "negative_controls": (
            negative_rate == 1.0 if (negative or cfg.require_negative_control) else True
        ),
    }
    reasons = tuple(f"aggregate_gate_failed:{name}" for name, passed in gates.items() if not passed)
    by_case: dict[str, float | None] = {}
    for case_id in sorted({row.case_id for row in positive}):
        by_case[case_id] = _credit_rate(tuple(row for row in positive if row.case_id == case_id))
    return OpportunityAggregateResult(
        robustly_attributed=all(gates.values()),
        n_results=len(rows),
        n_positive_trials=len(positive),
        n_negative_controls=len(negative),
        overall_credit_rate=overall_rate,
        clean_credit_rate=clean_rate,
        noisy_credit_rate=noisy_rate,
        restricted_credit_rate=restricted_rate,
        negative_control_pass_rate=negative_rate,
        gates=gates,
        case_credit_rates=by_case,
        reasons=reasons,
    )


__all__ = [
    "OpportunityAggregateConfig",
    "OpportunityAggregateResult",
    "OpportunityArmMetrics",
    "OpportunityAttributionConfig",
    "OpportunityAttributionResult",
    "OpportunityCaseSpec",
    "OpportunityReport",
    "OpportunitySampleBatch",
    "OPPORTUNITY_CASE_REGISTRY",
    "aggregate_opportunity_results",
    "evaluate_matched_opportunity",
    "get_opportunity_case",
    "list_opportunity_cases",
    "sample_opportunity_case",
]
