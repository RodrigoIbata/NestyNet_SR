# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Paired, multiplicity-aware confidence Pareto fronts.

The inferential primitive is the unit-level paired loss difference
``loss(challenger) - loss(incumbent)``.  A Gaussian or Rademacher multiplier
max-T bootstrap preserves dependence among all candidate comparisons and gives
one simultaneous upper confidence bound for every complexity-admissible edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .archive import CandidateArchive
from .audit import LossAudit
from .complexity import ComplexityVector, validate_complexity_collection


@dataclass(frozen=True)
class PairwiseRiskComparison:
    """One archive-conditional paired risk comparison."""

    challenger_id: str
    incumbent_id: str
    risk_difference: float
    standard_error: float
    upper_confidence_bound: float
    estimable: bool
    complexity_strictly_better: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenger_id": self.challenger_id,
            "incumbent_id": self.incumbent_id,
            "risk_difference": float(self.risk_difference),
            "standard_error": float(self.standard_error),
            "upper_confidence_bound": (
                float(self.upper_confidence_bound)
                if math.isfinite(self.upper_confidence_bound)
                else None
            ),
            "estimable": bool(self.estimable),
            "complexity_strictly_better": bool(self.complexity_strictly_better),
        }


@dataclass(frozen=True)
class ConfidenceParetoResult:
    """Point, strict-confidence, and practical-noninferiority fronts."""

    candidate_ids: tuple[str, ...]
    complexity_names: tuple[str, ...]
    risks: tuple[float, ...]
    eligible_candidate_ids: tuple[str, ...]
    ineligible_candidate_ids: tuple[str, ...]
    point_front: tuple[str, ...]
    confidence_front: tuple[str, ...]
    practical_front: tuple[str, ...]
    strict_dominance_edges: tuple[tuple[str, str], ...]
    practical_dominance_edges: tuple[tuple[str, str], ...]
    comparisons: tuple[PairwiseRiskComparison, ...]
    alpha: float
    delta: float
    critical_value: float
    n_resamples: int
    seed: int
    multiplier: str
    effective_unit_count: float
    archive_fingerprint: Optional[str]
    audit_fingerprint: str
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    # Which rule produced ``critical_value``: "multiplier_max_t" or
    # "bonferroni_t".  Defaults for backward compatibility of positional
    # construction; the certificate uses it as proof of dispatch.
    critical_value_method: str = "multiplier_max_t"

    @property
    def n_comparisons(self) -> int:
        return int(len(self.comparisons))

    @property
    def n_estimable_comparisons(self) -> int:
        return int(sum(comparison.estimable for comparison in self.comparisons))

    def risk_by_id(self) -> dict[str, float]:
        return {
            candidate_id: float(self.risks[i])
            for i, candidate_id in enumerate(self.candidate_ids)
        }

    def dominators_of(self, candidate_id: str, *, practical: bool = False) -> tuple[str, ...]:
        target = str(candidate_id)
        edges = self.practical_dominance_edges if practical else self.strict_dominance_edges
        return tuple(challenger for challenger, incumbent in edges if incumbent == target)

    def to_dict(self, *, include_comparisons: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_ids": list(self.candidate_ids),
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "ineligible_candidate_ids": list(self.ineligible_candidate_ids),
            "complexity_names": list(self.complexity_names),
            "risks": self.risk_by_id(),
            "point_front": list(self.point_front),
            "confidence_front": list(self.confidence_front),
            "practical_front": list(self.practical_front),
            "strict_dominance_edges": [list(edge) for edge in self.strict_dominance_edges],
            "practical_dominance_edges": [list(edge) for edge in self.practical_dominance_edges],
            "alpha": float(self.alpha),
            "delta": float(self.delta),
            "critical_value": float(self.critical_value),
            "n_comparisons": self.n_comparisons,
            "n_estimable_comparisons": self.n_estimable_comparisons,
            "n_resamples": int(self.n_resamples),
            "seed": int(self.seed),
            "multiplier": self.multiplier,
            "critical_value_method": self.critical_value_method,
            "effective_unit_count": float(self.effective_unit_count),
            "archive_fingerprint": self.archive_fingerprint,
            "audit_fingerprint": self.audit_fingerprint,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
        }
        if include_comparisons:
            payload["comparisons"] = [comparison.to_dict() for comparison in self.comparisons]
        return payload

    def write_json(self, path: str | Path, *, include_comparisons: bool = True) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_dict(include_comparisons=include_comparisons),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output


def point_pareto_front(
    candidate_ids: Sequence[str],
    risks: Sequence[float],
    complexities: Mapping[str, ComplexityVector] | Sequence[ComplexityVector],
    *,
    delta: float = 0.0,
    complexity_atol: float = 0.0,
    risk_atol: float = 0.0,
) -> tuple[str, ...]:
    """Return a point-estimate Pareto/noninferiority front.

    A no-more-complex challenger dominates when it has strictly lower risk, or
    when it is strictly simpler and its risk is no more than ``delta`` worse.
    ``delta=0`` is the ordinary risk/complexity Pareto front.
    """

    ids = tuple(str(item) for item in candidate_ids)
    risk_array = np.asarray(risks, dtype=np.float64)
    if risk_array.shape != (len(ids),):
        raise ValueError(f"risks must have shape {(len(ids),)!r}, got {risk_array.shape!r}")
    if not np.all(np.isfinite(risk_array)):
        raise ValueError("risks must be finite")
    complexity_tuple = _resolve_complexities(ids, complexities)
    delta_f = _nonnegative_finite(delta, "delta")
    complexity_tol = _nonnegative_finite(complexity_atol, "complexity_atol")
    risk_tol = _nonnegative_finite(risk_atol, "risk_atol")

    dominated: set[int] = set()
    for incumbent in range(len(ids)):
        for challenger in range(len(ids)):
            if challenger == incumbent:
                continue
            challenger_complexity = complexity_tuple[challenger]
            incumbent_complexity = complexity_tuple[incumbent]
            if not challenger_complexity.no_worse_than(
                incumbent_complexity,
                atol=complexity_tol,
            ):
                continue
            strictly_simpler = challenger_complexity.strictly_better_than(
                incumbent_complexity,
                atol=complexity_tol,
            )
            risk_better = risk_array[challenger] < risk_array[incumbent] - risk_tol
            noninferior_and_simpler = strictly_simpler and (
                risk_array[challenger] <= risk_array[incumbent] + delta_f + risk_tol
            )
            if risk_better or noninferior_and_simpler:
                dominated.add(incumbent)
                break

    front = [i for i in range(len(ids)) if i not in dominated]
    front.sort(key=lambda i: (float(risk_array[i]), complexity_tuple[i].values, ids[i]))
    return tuple(ids[i] for i in front)


def confidence_pareto(
    audit: LossAudit,
    complexities: CandidateArchive | Mapping[str, ComplexityVector] | Sequence[ComplexityVector],
    *,
    alpha: float = 0.05,
    delta: float = 0.0,
    n_resamples: int = 4000,
    seed: int = 12345,
    multiplier: str = "normal",
    resample_batch_size: int = 256,
    pair_batch_size: int = 4096,
    complexity_atol: float = 0.0,
    risk_atol: float = 0.0,
    eligible_candidate_ids: Optional[Sequence[str]] = None,
    method: str = "multiplier_max_t",
    bonferroni_comparisons: Optional[int] = None,
) -> ConfidenceParetoResult:
    """Construct an archive-conditional simultaneous confidence Pareto graph.

    The multiplier max-T calculation treats audit rows as independent units.
    It is therefore essential that callers aggregate correlated samples into
    trajectories/experiments before constructing ``LossAudit``.

    ``method`` selects the simultaneous critical value: ``"multiplier_max_t"``
    is the calibrated bootstrap default; ``"bonferroni_t"`` replaces it with
    the closed-form one-sided ``t_{1-alpha/K, G-1}`` for use outside the
    calibrated envelope, where ``K`` is ``bonferroni_comparisons`` (the
    declared pre-audit comparison family, never smaller than the pairs
    actually formed).  The studentised pair statistics are identical in both
    modes; only the critical value differs.
    """

    if not isinstance(audit, LossAudit):
        raise TypeError("audit must be a LossAudit")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    delta_f = _nonnegative_finite(delta, "delta")
    complexity_tol = _nonnegative_finite(complexity_atol, "complexity_atol")
    risk_tol = _nonnegative_finite(risk_atol, "risk_atol")
    n_boot = int(n_resamples)
    if n_boot < 1:
        raise ValueError("n_resamples must be positive")
    seed_i = int(seed)
    resample_batch = max(1, int(resample_batch_size))
    pair_batch = max(1, int(pair_batch_size))
    multiplier_name = str(multiplier).strip().lower()
    if multiplier_name not in {"normal", "rademacher"}:
        raise ValueError("multiplier must be 'normal' or 'rademacher'")
    method_name = str(method).strip().lower()
    if method_name not in {"multiplier_max_t", "bonferroni_t"}:
        raise ValueError("method must be 'multiplier_max_t' or 'bonferroni_t'")
    if bonferroni_comparisons is not None and int(bonferroni_comparisons) < 0:
        raise ValueError("bonferroni_comparisons must be non-negative when given")

    archive_fingerprint: Optional[str] = audit.archive_fingerprint
    if isinstance(complexities, CandidateArchive):
        archive = complexities
        audit.assert_archive(archive)
        archive_fingerprint = archive.fingerprint
        complexity_tuple = _resolve_complexities(
            audit.candidate_ids,
            archive.complexity_by_id(),
        )
    else:
        complexity_tuple = _resolve_complexities(audit.candidate_ids, complexities)

    candidate_ids = audit.candidate_ids
    eligible_ids = _resolve_eligible_candidate_ids(
        candidate_ids,
        eligible_candidate_ids,
    )
    eligible_set = set(eligible_ids)
    eligible_indices = tuple(
        index
        for index, candidate_id in enumerate(candidate_ids)
        if candidate_id in eligible_set
    )
    ineligible_ids = tuple(
        candidate_id for candidate_id in candidate_ids if candidate_id not in eligible_set
    )
    risks = np.asarray(audit.risks, dtype=np.float64)
    pairs: list[tuple[int, int, bool]] = []
    for incumbent in eligible_indices:
        for challenger in eligible_indices:
            if challenger == incumbent:
                continue
            challenger_complexity = complexity_tuple[challenger]
            incumbent_complexity = complexity_tuple[incumbent]
            if not challenger_complexity.no_worse_than(
                incumbent_complexity,
                atol=complexity_tol,
            ):
                continue
            pairs.append(
                (
                    challenger,
                    incumbent,
                    challenger_complexity.strictly_better_than(
                        incumbent_complexity,
                        atol=complexity_tol,
                    ),
                )
            )

    if pairs:
        challenger_idx = np.fromiter((item[0] for item in pairs), dtype=np.int64)
        incumbent_idx = np.fromiter((item[1] for item in pairs), dtype=np.int64)
        strictly_simpler = np.fromiter((item[2] for item in pairs), dtype=np.bool_)
        differences = risks[challenger_idx] - risks[incumbent_idx]
        centered_candidates = audit.losses - risks[None, :]
        denominators = np.empty(len(pairs), dtype=np.float64)
        pair_scales = np.empty(len(pairs), dtype=np.float64)
        for start in range(0, len(pairs), pair_batch):
            selected = slice(start, min(start + pair_batch, len(pairs)))
            pair_centered = (
                centered_candidates[:, challenger_idx[selected]]
                - centered_candidates[:, incumbent_idx[selected]]
            )
            weighted_pair_centered = audit.unit_weights[:, None] * pair_centered
            denominators[selected] = np.sqrt(
                np.sum(weighted_pair_centered * weighted_pair_centered, axis=0)
            )
            pair_scales[selected] = np.maximum(
                1.0,
                np.maximum(
                    np.abs(differences[selected]),
                    np.max(np.abs(pair_centered), axis=0),
                ),
            )
        correction = 1.0 - float(np.sum(audit.unit_weights * audit.unit_weights))
        standard_errors = denominators / math.sqrt(correction)
        denominator_floor = 64.0 * np.finfo(np.float64).eps * pair_scales
        estimable = denominators > denominator_floor
        if method_name == "bonferroni_t":
            # Never divide alpha by fewer comparisons than were actually
            # formed: a caller-declared family may only enlarge the burden.
            family = max(len(pairs), int(bonferroni_comparisons or 0))
            critical_value = _bonferroni_t_critical_value(
                alpha=alpha_f,
                n_comparisons=family,
                effective_unit_count=audit.effective_unit_count,
            )
        else:
            critical_value = _max_t_critical_value(
                audit=audit,
                challenger_idx=challenger_idx,
                incumbent_idx=incumbent_idx,
                denominators=denominators,
                estimable=estimable,
                alpha=alpha_f,
                n_resamples=n_boot,
                seed=seed_i,
                multiplier=multiplier_name,
                resample_batch_size=resample_batch,
                pair_batch_size=pair_batch,
            )
        upper_bounds = np.full(len(pairs), np.inf, dtype=np.float64)
        upper_bounds[estimable] = (
            differences[estimable] + critical_value * standard_errors[estimable]
        )
    else:
        challenger_idx = np.empty(0, dtype=np.int64)
        incumbent_idx = np.empty(0, dtype=np.int64)
        strictly_simpler = np.empty(0, dtype=np.bool_)
        differences = np.empty(0, dtype=np.float64)
        standard_errors = np.empty(0, dtype=np.float64)
        estimable = np.empty(0, dtype=np.bool_)
        upper_bounds = np.empty(0, dtype=np.float64)
        critical_value = 0.0

    comparisons: list[PairwiseRiskComparison] = []
    strict_edges: list[tuple[str, str]] = []
    practical_edges: list[tuple[str, str]] = []
    for k in range(len(pairs)):
        challenger_id = candidate_ids[int(challenger_idx[k])]
        incumbent_id = candidate_ids[int(incumbent_idx[k])]
        comparison = PairwiseRiskComparison(
            challenger_id=challenger_id,
            incumbent_id=incumbent_id,
            risk_difference=float(differences[k]),
            standard_error=float(standard_errors[k]),
            upper_confidence_bound=float(upper_bounds[k]),
            estimable=bool(estimable[k]),
            complexity_strictly_better=bool(strictly_simpler[k]),
        )
        comparisons.append(comparison)
        if not comparison.estimable:
            continue
        confidently_better = comparison.upper_confidence_bound < -risk_tol
        practically_dominant = confidently_better or (
            comparison.complexity_strictly_better
            and comparison.upper_confidence_bound <= delta_f + risk_tol
        )
        if confidently_better:
            strict_edges.append((challenger_id, incumbent_id))
        if practically_dominant:
            practical_edges.append((challenger_id, incumbent_id))

    strict_edges = _sorted_unique_edges(strict_edges)
    practical_edges = _sorted_unique_edges(practical_edges)
    eligible_risks = np.asarray(
        [risks[index] for index in eligible_indices],
        dtype=np.float64,
    )
    eligible_complexities = tuple(complexity_tuple[index] for index in eligible_indices)
    confidence_front = _front_from_edges(
        eligible_ids,
        strict_edges,
        eligible_risks,
        eligible_complexities,
    )
    practical_front = _front_from_edges(
        eligible_ids,
        practical_edges,
        eligible_risks,
        eligible_complexities,
    )
    point_front = point_pareto_front(
        eligible_ids,
        eligible_risks,
        eligible_complexities,
        delta=0.0,
        complexity_atol=complexity_tol,
        risk_atol=risk_tol,
    )

    if method_name == "bonferroni_t":
        method_assumption = (
            "Studentised paired risk differences are adequately t-distributed "
            "with G-1 degrees of freedom; the Bonferroni correction over the "
            "declared pre-audit comparison family is level-conservative."
        )
    else:
        method_assumption = (
            "The multiplier max-T approximation is adequate for the number and "
            "distribution of independent units."
        )
    assumptions = (
        "The candidate archive was frozen before the audit losses were inspected.",
        f"Audit rows are declared independent units of kind {audit.design.unit_kind!r}.",
        "Every candidate was evaluated on the same declared units and domain.",
        f"Declared candidate-fitting protocol: {audit.design.fit_protocol}.",
        method_assumption,
    ) + tuple(audit.design.sampling_assumptions)
    warnings: list[str] = []
    if audit.effective_unit_count < 20.0:
        warnings.append(
            "Fewer than 20 effective independent units were supplied; "
            + (
                "small-sample t-approximation"
                if method_name == "bonferroni_t"
                else "multiplier-bootstrap"
            )
            + " coverage should be checked by problem-specific simulation."
        )
    if any(not comparison.estimable for comparison in comparisons):
        warnings.append(
            "At least one paired comparison had empirically degenerate variance and was "
            "retained on the front conservatively."
        )
    if np.any(audit.failure_mask):
        warnings.append(
            "One or more candidate/unit failures were scored with the declared common penalty."
        )
    if ineligible_ids:
        warnings.append(
            "One or more frozen candidates were excluded from inferential fronts by the "
            "predeclared feasibility rule; they remain recorded in the audit and certificate."
        )
    if n_boot < 2000 and method_name != "bonferroni_t":
        warnings.append(
            "Fewer than 2000 multiplier draws were requested; tail-quantile Monte Carlo "
            "error may be visible."
        )
    return ConfidenceParetoResult(
        candidate_ids=candidate_ids,
        complexity_names=complexity_tuple[0].names,
        risks=tuple(float(value) for value in risks),
        eligible_candidate_ids=eligible_ids,
        ineligible_candidate_ids=ineligible_ids,
        point_front=point_front,
        confidence_front=confidence_front,
        practical_front=practical_front,
        strict_dominance_edges=tuple(strict_edges),
        practical_dominance_edges=tuple(practical_edges),
        comparisons=tuple(comparisons),
        alpha=alpha_f,
        delta=delta_f,
        critical_value=float(critical_value),
        n_resamples=n_boot,
        seed=seed_i,
        multiplier=multiplier_name,
        effective_unit_count=audit.effective_unit_count,
        archive_fingerprint=archive_fingerprint,
        audit_fingerprint=audit.fingerprint,
        assumptions=assumptions,
        warnings=tuple(warnings),
        critical_value_method=method_name,
    )


def bootstrap_front_inclusion_frequencies(
    audit: LossAudit,
    complexities: CandidateArchive | Mapping[str, ComplexityVector] | Sequence[ComplexityVector],
    *,
    n_resamples: int = 2000,
    seed: int = 12345,
    delta: float = 0.0,
    complexity_atol: float = 0.0,
    risk_atol: float = 0.0,
    eligible_candidate_ids: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """Return ordinary unit-bootstrap front-inclusion frequencies.

    This is a stability diagnostic, not a confidence level.  The implementation
    deliberately requires equal unit weights so that the resampling target is
    unambiguous.
    """

    n_boot = int(n_resamples)
    if n_boot < 1:
        raise ValueError("n_resamples must be positive")
    uniform = np.full(audit.n_units, 1.0 / float(audit.n_units), dtype=np.float64)
    if not np.allclose(audit.unit_weights, uniform, rtol=0.0, atol=1.0e-14):
        raise ValueError("front-inclusion bootstrap currently requires equal unit weights")
    if isinstance(complexities, CandidateArchive):
        audit.assert_archive(complexities)
        complexity_tuple = _resolve_complexities(
            audit.candidate_ids,
            complexities.complexity_by_id(),
        )
    else:
        complexity_tuple = _resolve_complexities(audit.candidate_ids, complexities)

    eligible_ids = _resolve_eligible_candidate_ids(
        audit.candidate_ids,
        eligible_candidate_ids,
    )
    eligible_index = {
        candidate_id: index for index, candidate_id in enumerate(audit.candidate_ids)
    }
    selected_indices = np.asarray(
        [eligible_index[candidate_id] for candidate_id in eligible_ids],
        dtype=np.int64,
    )
    selected_complexities = tuple(
        complexity_tuple[int(index)] for index in selected_indices
    )

    rng = np.random.default_rng(int(seed))
    counts = np.zeros(audit.n_candidates, dtype=np.int64)
    index_by_id = {candidate_id: i for i, candidate_id in enumerate(audit.candidate_ids)}
    for _ in range(n_boot):
        sample = rng.integers(0, audit.n_units, size=audit.n_units)
        risks = np.mean(audit.losses[sample, :][:, selected_indices], axis=0)
        front = point_pareto_front(
            eligible_ids,
            risks,
            selected_complexities,
            delta=delta,
            complexity_atol=complexity_atol,
            risk_atol=risk_atol,
        )
        for candidate_id in front:
            counts[index_by_id[candidate_id]] += 1
    return {
        candidate_id: float(counts[i]) / float(n_boot)
        for i, candidate_id in enumerate(audit.candidate_ids)
    }


def _resolve_eligible_candidate_ids(
    candidate_ids: Sequence[str],
    eligible_candidate_ids: Optional[Sequence[str]],
) -> tuple[str, ...]:
    ids = tuple(str(item) for item in candidate_ids)
    if eligible_candidate_ids is None:
        return ids
    requested = tuple(str(item) for item in eligible_candidate_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("eligible_candidate_ids must not contain duplicates")
    unknown = sorted(set(requested) - set(ids))
    if unknown:
        raise ValueError(f"eligible_candidate_ids contains unknown candidates: {unknown!r}")
    selected = set(requested)
    resolved = tuple(candidate_id for candidate_id in ids if candidate_id in selected)
    if not resolved:
        raise ValueError("at least one eligible candidate is required")
    return resolved


def _resolve_complexities(
    candidate_ids: Sequence[str],
    complexities: Mapping[str, ComplexityVector] | Sequence[ComplexityVector],
) -> tuple[ComplexityVector, ...]:
    ids = tuple(str(item) for item in candidate_ids)
    if isinstance(complexities, Mapping):
        missing = [candidate_id for candidate_id in ids if candidate_id not in complexities]
        extra = sorted(set(str(key) for key in complexities) - set(ids))
        if missing or extra:
            raise ValueError(
                "complexity records do not match candidate_ids; "
                f"missing={missing!r}, extra={extra!r}"
            )
        out = tuple(complexities[candidate_id] for candidate_id in ids)
    else:
        out = tuple(complexities)
        if len(out) != len(ids):
            raise ValueError(
                f"expected {len(ids)} complexity vectors, received {len(out)}"
            )
    return validate_complexity_collection(out)


def _bonferroni_t_critical_value(
    *,
    alpha: float,
    n_comparisons: int,
    effective_unit_count: float,
) -> float:
    """One-sided ``t_{1-alpha/K, G-1}`` over the declared comparison family.

    The closed form makes no use of the multiplier approximation at small
    ``G``, which is exactly why it may be used outside the calibrated
    envelope; it pays for that validity in power, never in level.  Degrees of
    freedom are floored from the effective unit count, the conservative
    direction under unequal weights.
    """
    from scipy.stats import t as student_t

    family = max(1, int(n_comparisons))
    # The epsilon absorbs float round-off in the Kish count (e.g. 79.9999...),
    # which would otherwise cost a whole degree of freedom for nothing.
    dof = max(1, int(math.floor(float(effective_unit_count) + 1.0e-9)) - 1)
    return float(student_t.ppf(1.0 - float(alpha) / family, dof))


def _max_t_critical_value(
    *,
    audit: LossAudit,
    challenger_idx: np.ndarray,
    incumbent_idx: np.ndarray,
    denominators: np.ndarray,
    estimable: np.ndarray,
    alpha: float,
    n_resamples: int,
    seed: int,
    multiplier: str,
    resample_batch_size: int,
    pair_batch_size: int,
) -> float:
    valid = np.flatnonzero(estimable)
    if valid.size == 0:
        return 0.0

    risks = np.asarray(audit.risks, dtype=np.float64)
    centered = audit.losses - risks[None, :]
    weighted_centered = audit.unit_weights[:, None] * centered
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_resamples, dtype=np.float64)

    written = 0
    while written < n_resamples:
        batch = min(resample_batch_size, n_resamples - written)
        if multiplier == "normal":
            multipliers = rng.standard_normal((batch, audit.n_units))
        else:
            multipliers = rng.integers(0, 2, size=(batch, audit.n_units), dtype=np.int8)
            multipliers = multipliers.astype(np.float64) * 2.0 - 1.0
        candidate_perturbations = multipliers @ weighted_centered
        batch_max = np.full(batch, -np.inf, dtype=np.float64)
        for start in range(0, valid.size, pair_batch_size):
            selected = valid[start : start + pair_batch_size]
            numerators = (
                candidate_perturbations[:, challenger_idx[selected]]
                - candidate_perturbations[:, incumbent_idx[selected]]
            )
            t_values = numerators / denominators[selected][None, :]
            batch_max = np.maximum(batch_max, np.max(t_values, axis=1))
        maxima[written : written + batch] = batch_max
        written += batch

    quantile = _higher_quantile(maxima, 1.0 - alpha)
    return float(max(0.0, quantile))


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy < 1.22 compatibility
        return float(np.quantile(values, probability, interpolation="higher"))


def _front_from_edges(
    candidate_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    risks: np.ndarray,
    complexities: Sequence[ComplexityVector],
) -> tuple[str, ...]:
    dominated = {incumbent for _, incumbent in edges}
    index = {candidate_id: i for i, candidate_id in enumerate(candidate_ids)}
    front = [candidate_id for candidate_id in candidate_ids if candidate_id not in dominated]
    front.sort(
        key=lambda candidate_id: (
            float(risks[index[candidate_id]]),
            complexities[index[candidate_id]].values,
            candidate_id,
        )
    )
    return tuple(front)


def _sorted_unique_edges(edges: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted({(str(challenger), str(incumbent)) for challenger, incumbent in edges})


def _nonnegative_finite(value: float, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return out
