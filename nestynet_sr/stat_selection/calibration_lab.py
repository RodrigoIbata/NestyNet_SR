# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
"""A controlled laboratory for the confidence-Pareto machinery.

``calibration_smoke_test`` in :mod:`.uncertainty` checks the arithmetic of a
one-sample t interval.  It does not exercise the multiplier max-T, the pairwise
dominance rule, or the front.  This module does, on synthetic populations whose
risks and loss covariance are known exactly, with no search, no fitting and no
symbolic machinery in the way.

The separation is deliberate.  A laboratory experiment answers *is the
inferential procedure calibrated*; a benchmark campaign answers *does the whole
system stay calibrated once parsing, fitting, archive generation, aliases and
domain failures are involved*.  Confounding the two makes a failure impossible
to localise.

The central claim under test is familywise, not pointwise:

    Pr(any dominance edge asserts an ordering the population does not have)
        <= alpha

and its user-facing consequence, that every genuinely non-dominated candidate
survives on the confidence front with probability at least ``1 - alpha``.  Both
are measured, because the first is what the max-T controls and the second is
what a reader of the certificate believes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .audit import AuditDesign, LossAudit
from .complexity import ComplexityVector
from .pareto import confidence_pareto, point_pareto_front

__all__ = [
    "LabPopulation",
    "make_lab_population",
    "draw_loss_matrix",
    "population_front",
    "null_front_coverage",
    "multiplicity_sweep",
    "cluster_calibration",
    "dominance_power",
    "critical_value_strategies",
    "calibration_envelope",
    "candidates_for_comparison_count",
]


# --------------------------------------------------------------------------- #
# population construction                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabPopulation:
    """A synthetic candidate population with exactly known statistics.

    ``risks`` are population risks, ``cov`` the candidate-by-candidate
    covariance of the per-unit loss vector, and ``complexities`` the declared
    complexity vectors.  ``group`` labels each candidate's role so that
    diagnostics can be read by construction rather than inferred.
    """

    candidate_ids: tuple[str, ...]
    risks: np.ndarray                      # (M,)
    cov: np.ndarray                        # (M, M)
    complexities: tuple[ComplexityVector, ...]
    group: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return int(self.risks.size)

    def complexity_map(self) -> dict[str, ComplexityVector]:
        return dict(zip(self.candidate_ids, self.complexities))


def _complexity(value: float) -> ComplexityVector:
    return ComplexityVector(components=(("size", float(value)),))


def make_lab_population(
    *,
    n_null: int = 6,
    n_dominated: int = 4,
    n_near: int = 3,
    n_decoy: int = 3,
    base_risk: float = 1.0,
    dominated_gap: float = 0.60,
    near_gap: float = 0.02,
    rho: float = 0.85,
    variance_spread: float = 8.0,
    noise_fraction: float = 0.5,
    seed: int = 0,
) -> LabPopulation:
    """Build a population exercising the pathologies the front must survive.

    ``n_null`` candidates share an identical population risk *and* an identical
    complexity.  Neither axis separates them, so all of them are genuinely
    non-dominated and every dominance edge among them is false by construction.
    They are the coverage target, and they supply ``n_null * (n_null - 1)``
    ordered comparisons, which is what makes multiplicity bite.

    ``n_dominated`` sit at ``base_risk * (1 + dominated_gap)`` with the same
    complexity, so they are truly dominated and drive the power measurement.
    ``n_near`` sit a hair above the null risk, the regime where a procedure
    that is merely conservative looks identical to one that is calibrated.
    ``n_decoy`` occupy a complexity ladder so the front has real structure
    rather than a single indifference class.

    ``rho`` sets the correlation between candidate losses.  Real candidates see
    the same data and are strongly correlated, which is precisely the structure
    a paired analysis exploits and a marginal one throws away.
    ``variance_spread`` is the ratio between the largest and smallest marginal
    loss variance, so that an unstudentized statistic can be seen failing.
    ``noise_fraction`` caps the largest marginal variance at that fraction of
    the smallest population risk, which the chi-square construction requires to
    keep the declared risks exact.
    """
    if min(n_null, n_dominated, n_near, n_decoy) < 0:
        raise ValueError("group sizes must be non-negative")
    if n_null < 2:
        raise ValueError("n_null must be at least 2 to create a null comparison")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if variance_spread <= 0.0:
        raise ValueError("variance_spread must be positive")
    if not 0.0 < noise_fraction < 1.0:
        raise ValueError("noise_fraction must lie strictly in (0, 1)")

    risks: list[float] = []
    complexities: list[ComplexityVector] = []
    groups: list[str] = []

    for _ in range(n_null):
        risks.append(base_risk)
        complexities.append(_complexity(10.0))
        groups.append("null")
    for _ in range(n_dominated):
        risks.append(base_risk * (1.0 + dominated_gap))
        complexities.append(_complexity(10.0))
        groups.append("dominated")
    for _ in range(n_near):
        risks.append(base_risk * (1.0 + near_gap))
        complexities.append(_complexity(10.0))
        groups.append("near")
    for k in range(n_decoy):
        # Simpler but worse: a genuine trade-off, so the front is not a single
        # indifference class.
        risks.append(base_risk * (1.0 + 0.25 * (k + 1)))
        complexities.append(_complexity(10.0 - (k + 1)))
        groups.append("decoy")

    n = len(risks)
    rng = np.random.default_rng(int(seed))
    # Heteroscedastic marginal variances spanning `variance_spread`.
    scale = np.exp(np.linspace(0.0, np.log(variance_spread), n) * 0.5)
    rng.shuffle(scale)
    corr = np.full((n, n), float(rho))
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(scale, scale)
    # Symmetrise and nudge onto the PSD cone against rounding.
    cov = 0.5 * (cov + cov.T)
    smallest = float(np.linalg.eigvalsh(cov).min())
    if smallest <= 0.0:
        cov = cov + (abs(smallest) + 1e-12) * np.eye(n)

    # Cap the marginal variances below the smallest population risk.  The
    # chi-square construction represents a loss as ``(shift + e)^2`` with
    # ``shift^2 = risk - Var(e)``, so a variance exceeding the risk would clamp
    # the shift to zero and silently change the population risk to the variance.
    # The declared risks would then be wrong and every null group would cease to
    # be equal-risk, which reads as a catastrophic coverage failure rather than
    # as the setup error it is.
    variance_ceiling = float(noise_fraction) * float(np.min(risks))
    largest = float(np.max(np.diag(cov)))
    if largest > variance_ceiling:
        cov = cov * (variance_ceiling / largest)

    return LabPopulation(
        candidate_ids=tuple(f"c{i:03d}" for i in range(n)),
        risks=np.asarray(risks, dtype=np.float64),
        cov=cov,
        complexities=tuple(complexities),
        group=tuple(groups),
        metadata={
            "rho": float(rho),
            "variance_spread": float(variance_spread),
            "base_risk": float(base_risk),
            "dominated_gap": float(dominated_gap),
            "near_gap": float(near_gap),
            "noise_fraction": float(noise_fraction),
        },
    )


# --------------------------------------------------------------------------- #
# data generation                                                             #
# --------------------------------------------------------------------------- #
def draw_loss_matrix(
    population: LabPopulation,
    n_units: int,
    rng: np.random.Generator,
    *,
    distribution: str = "gaussian",
) -> np.ndarray:
    """Return a ``(n_units, M)`` per-unit loss matrix from the population.

    ``distribution='gaussian'`` is the easy case: the multiplier bootstrap is
    being asked to reproduce a Gaussian it already matches, so a failure here
    is a coding error rather than a modelling one.

    ``distribution='chisq'`` squares correlated normals, which is what a
    per-unit mean squared residual actually is: non-negative, right-skewed, and
    correlated across candidates.  The population risk is preserved exactly
    because ``E[(mu + e)^2] = mu^2 + Var(e)``, so ``mu`` is chosen to absorb the
    variance.  This is the case that can genuinely break a bootstrap.
    """
    mode = str(distribution).strip().lower()
    if mode not in {"gaussian", "chisq"}:
        raise ValueError("distribution must be 'gaussian' or 'chisq'")
    if int(n_units) < 2:
        raise ValueError("n_units must be at least 2")

    chol = np.linalg.cholesky(population.cov)
    # numpy built against Apple Accelerate raises spurious divide/overflow/invalid
    # FP flags from matmul even on small well-conditioned finite operands; a bare
    # `standard_normal((24,16)) @ standard_normal((16,16))` reproduces it.  The
    # result is verified finite below rather than trusted.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        noise = rng.standard_normal((int(n_units), population.n_candidates)) @ chol.T
    if not np.isfinite(noise).all():
        raise FloatingPointError("synthetic loss draw produced non-finite values")

    if mode == "gaussian":
        return population.risks[None, :] + noise

    variances = np.diag(population.cov)
    shift = np.sqrt(np.maximum(population.risks - variances, 0.0))
    return (shift[None, :] + noise) ** 2


def _audit_from_matrix(population: LabPopulation, losses: np.ndarray) -> LossAudit:
    return LossAudit.from_matrix(
        candidate_ids=list(population.candidate_ids),
        unit_ids=[f"u{g:04d}" for g in range(losses.shape[0])],
        design=AuditDesign(
            loss_name="synthetic_lab_loss",
            unit_kind="synthetic_independent_unit",
            fit_protocol="frozen population; no fitting",
            evaluation_domain={"source": "calibration_lab"},
        ),
        losses=losses,
    )


def population_front(population: LabPopulation) -> tuple[str, ...]:
    """Return the exact population Pareto front, computed from known risks."""
    return point_pareto_front(
        population.candidate_ids,
        population.risks,
        population.complexity_map(),
        delta=0.0,
    )


# --------------------------------------------------------------------------- #
# comparators                                                                 #
# --------------------------------------------------------------------------- #
def _pair_stats(losses: np.ndarray, j: int, i: int) -> tuple[float, float]:
    """Return the paired risk difference ``R_j - R_i`` and its standard error."""
    d = losses[:, j] - losses[:, i]
    n = d.size
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))


def _edges_pointwise(losses: np.ndarray, pairs, alpha: float) -> set[tuple[int, int]]:
    """Per-pair one-sided bound with no multiplicity correction."""
    from scipy.stats import norm  # local import: only the comparators need scipy

    z = float(norm.ppf(1.0 - alpha))
    out = set()
    for j, i in pairs:
        diff, se = _pair_stats(losses, j, i)
        if se <= 0.0:
            continue
        if diff + z * se < 0.0:
            out.add((j, i))
    return out


def _edges_marginal_se(losses: np.ndarray, pairs, alpha: float, k: float = 1.0) -> set[tuple[int, int]]:
    """The superseded ``max(SE_a, SE_b)`` rule on *marginal* risk standard errors.

    This is the comparator that motivated the whole exercise.  It ignores the
    covariance between two candidates evaluated on the same units, so it is
    calibrated for independent candidates and wrong for the correlated ones
    that actually arise.
    """
    n = losses.shape[0]
    marginal_se = losses.std(axis=0, ddof=1) / np.sqrt(n)
    risks = losses.mean(axis=0)
    out = set()
    for j, i in pairs:
        eps = k * max(float(marginal_se[j]), float(marginal_se[i]))
        if risks[j] < risks[i] - eps:
            out.add((j, i))
    return out


def _admissible_pairs(population: LabPopulation) -> list[tuple[int, int]]:
    """Ordered pairs the confidence machinery would consider (complexity-filtered)."""
    comps = population.complexities
    pairs = []
    for i in range(population.n_candidates):
        for j in range(population.n_candidates):
            if i == j:
                continue
            if comps[j].no_worse_than(comps[i], atol=0.0):
                pairs.append((j, i))
    return pairs


def _false_edges(population: LabPopulation, edges) -> int:
    """Count edges asserting an ordering the population does not have."""
    risks = population.risks
    return sum(1 for j, i in edges if not (risks[j] < risks[i]))


# --------------------------------------------------------------------------- #
# experiments                                                                 #
# --------------------------------------------------------------------------- #
def null_front_coverage(
    population: LabPopulation,
    *,
    n_units: int = 24,
    n_replicates: int = 2000,
    alpha: float = 0.05,
    n_resamples: int = 400,
    seed: int = 12345,
    distribution: str = "gaussian",
) -> dict[str, Any]:
    """Measure familywise false-edge rate and front coverage.

    Two quantities are reported.  ``familywise_false_edge_rate`` is the
    probability that *any* dominance edge asserts an ordering the population
    does not have; this is what the max-T is supposed to control at ``alpha``.
    ``front_coverage`` is the probability that every genuinely non-dominated
    candidate survives on the confidence front; this is what a reader of the
    certificate actually believes, and it is implied by the first.
    """
    truth = set(population_front(population))
    complexity_map = population.complexity_map()
    rng = np.random.default_rng(int(seed))

    any_false = 0
    covered = 0
    front_sizes: list[int] = []
    false_edge_counts: list[int] = []
    index_of = {cid: k for k, cid in enumerate(population.candidate_ids)}

    for _ in range(int(n_replicates)):
        losses = draw_loss_matrix(population, n_units, rng, distribution=distribution)
        audit = _audit_from_matrix(population, losses)
        result = confidence_pareto(
            audit,
            complexity_map,
            alpha=alpha,
            delta=0.0,
            n_resamples=int(n_resamples),
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        edges = [
            (index_of[a], index_of[b]) for a, b in result.strict_dominance_edges
        ]
        n_false = _false_edges(population, edges)
        false_edge_counts.append(n_false)
        any_false += int(n_false > 0)
        covered += int(truth.issubset(set(result.confidence_front)))
        front_sizes.append(len(result.confidence_front))

    reps = int(n_replicates)
    return {
        "familywise_false_edge_rate": any_false / reps,
        "front_coverage": covered / reps,
        "nominal_coverage": 1.0 - float(alpha),
        "alpha": float(alpha),
        "mean_false_edges": float(np.mean(false_edge_counts)),
        "mean_front_size": float(np.mean(front_sizes)),
        "population_front_size": len(truth),
        "n_candidates": population.n_candidates,
        "n_units": int(n_units),
        "n_replicates": reps,
        "distribution": str(distribution),
    }


def multiplicity_sweep(
    *,
    archive_sizes: Sequence[int] = (5, 10, 25, 50, 100),
    n_units: int = 24,
    n_replicates: int = 400,
    alpha: float = 0.05,
    n_resamples: int = 400,
    rho: float = 0.85,
    variance_spread: float = 8.0,
    seed: int = 7,
    distribution: str = "gaussian",
) -> list[dict[str, Any]]:
    """Hold units fixed, grow the archive, and compare the three rules.

    The claim the package makes is archive-level: false exclusions stay
    controlled as the number of correlated comparisons grows.  Pointwise
    intervals and the marginal-SE rule should degrade visibly with ``M`` while
    the simultaneous procedure stays near nominal.  This is the sweep that
    shows *why* ordinary selection logic is unsafe rather than asserting it.
    """
    rows: list[dict[str, Any]] = []
    for size in archive_sizes:
        n_null = max(2, int(size))
        population = make_lab_population(
            n_null=n_null, n_dominated=0, n_near=0, n_decoy=0,
            rho=rho, variance_spread=variance_spread, seed=int(seed),
        )
        pairs = _admissible_pairs(population)
        complexity_map = population.complexity_map()
        rng = np.random.default_rng(int(seed) + size)

        simultaneous = pointwise = marginal = 0
        index_of = {cid: k for k, cid in enumerate(population.candidate_ids)}

        for _ in range(int(n_replicates)):
            losses = draw_loss_matrix(population, n_units, rng, distribution=distribution)
            audit = _audit_from_matrix(population, losses)
            result = confidence_pareto(
                audit, complexity_map, alpha=alpha, delta=0.0,
                n_resamples=int(n_resamples),
                seed=int(rng.integers(1, 2**31 - 1)),
            )
            edges = [(index_of[a], index_of[b]) for a, b in result.strict_dominance_edges]
            simultaneous += int(_false_edges(population, edges) > 0)
            pointwise += int(_false_edges(population, _edges_pointwise(losses, pairs, alpha)) > 0)
            marginal += int(_false_edges(population, _edges_marginal_se(losses, pairs, alpha)) > 0)

        reps = int(n_replicates)
        rows.append({
            "n_candidates": population.n_candidates,
            "n_admissible_pairs": len(pairs),
            "n_units": int(n_units),
            "alpha": float(alpha),
            "simultaneous_max_t": simultaneous / reps,
            "pointwise": pointwise / reps,
            "marginal_se_rule": marginal / reps,
            "n_replicates": reps,
        })
    return rows


def cluster_calibration(
    *,
    n_groups: int = 12,
    samples_per_group: Sequence[int] = (1, 4, 16, 64, 256),
    n_replicates: int = 400,
    alpha: float = 0.05,
    n_resamples: int = 400,
    seed: int = 11,
) -> list[dict[str, Any]]:
    """Show that dense sampling within a group is not extra independent evidence.

    Two candidates have equal population risk.  Each group carries a shared
    group-level offset plus within-group noise, so observations inside a group
    are dependent.  Treating rows as units makes the apparent standard error
    shrink like ``1/sqrt(n_groups * n_per_group)`` when the truth only supports
    ``1/sqrt(n_groups)``, so its coverage falls away as sampling density grows
    while the group-level analysis stays flat.
    """
    rows: list[dict[str, Any]] = []
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else None
    if z is None:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))

    for per_group in samples_per_group:
        rng = np.random.default_rng(int(seed) + int(per_group))
        group_covered = 0
        row_covered = 0
        for _ in range(int(n_replicates)):
            # Shared group effect dominates; within-group noise is independent.
            group_effect = rng.normal(0.0, 1.0, size=int(n_groups))
            within = rng.normal(0.0, 0.5, size=(int(n_groups), int(per_group)))
            paired = group_effect[:, None] + within        # paired loss differences

            group_means = paired.mean(axis=1)
            g_mean = float(group_means.mean())
            g_se = float(group_means.std(ddof=1) / np.sqrt(n_groups))
            group_covered += int(abs(g_mean) <= z * g_se)

            flat = paired.reshape(-1)
            r_mean = float(flat.mean())
            r_se = float(flat.std(ddof=1) / np.sqrt(flat.size))
            row_covered += int(abs(r_mean) <= z * r_se)

        reps = int(n_replicates)
        rows.append({
            "samples_per_group": int(per_group),
            "n_groups": int(n_groups),
            "group_level_coverage": group_covered / reps,
            "row_level_coverage": row_covered / reps,
            "nominal_coverage": 1.0 - float(alpha),
            "n_replicates": reps,
        })
    return rows


def _wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    """Wilson upper confidence bound for a binomial rate.

    The calibration experiment has Monte Carlo error of its own, so a cell is
    certified on the upper bound rather than the point estimate.  Certifying on
    the point estimate would let a cell whose true rate is 0.07 pass whenever
    the draw happened to land below the threshold.
    """
    from scipy.stats import norm

    if trials <= 0:
        return 1.0
    z = float(norm.ppf(confidence))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    spread = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float(min(1.0, (centre + spread) / denominator))


def candidates_for_comparison_count(target_pairs: int) -> int:
    """Smallest equal-complexity candidate count giving at least ``target_pairs``.

    With one complexity class every ordered pair is admissible, so
    ``K_adm = M (M - 1)``.  This inverts that so a grid can be specified in the
    coordinate that actually drives the difficulty, the number of simultaneous
    comparisons, rather than in raw archive size.
    """
    target = max(2, int(target_pairs))
    n = 2
    while n * (n - 1) < target:
        n += 1
    return n


def calibration_envelope(
    *,
    unit_grid: Sequence[int] = (12, 24, 60, 150),
    comparison_grid: Sequence[int] = (10, 100, 1000),
    n_replicates: int = 800,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    validated_bound: float = 0.06,
    transitional_bound: float = 0.08,
    seed: int = 101,
    distribution: str = "gaussian",
) -> list[dict[str, Any]]:
    """Map the joint ``(G, K_adm)`` operating envelope of the multiplier max-T.

    The failure mode is a high-dimensional finite-sample approximation, so the
    governing coordinates are the number of independent units and the number of
    *admissible simultaneous comparisons*, not raw archive size.  Two archives
    of equal size can present very different families once complexity filtering
    is applied, which is why ``K_adm`` is the transparent first coordinate.

    Each cell is classified on the Wilson upper bound of the observed false-edge
    rate, never on the point estimate:

    ``validated``     upper bound at or below ``validated_bound``
    ``transitional``  upper bound between the two bounds
    ``outside``       upper bound above ``transitional_bound``

    ``transitional`` is descriptive metadata, not permission to spend type-I
    error; production inference should use the conservative fallback there too.

    ``log_k_over_g`` is recorded as a candidate collapse coordinate.  If cells
    fall onto a smooth curve in it, the envelope can be interpolated rather than
    tabulated, and it hints at the finite-sample scaling.  It is a summary of
    the grid, not a substitute for it.
    """
    rows: list[dict[str, Any]] = []
    for target_pairs in comparison_grid:
        n_null = candidates_for_comparison_count(target_pairs)
        population = make_lab_population(
            n_null=n_null, n_dominated=0, n_near=0, n_decoy=0, seed=int(seed)
        )
        complexity_map = population.complexity_map()
        k_adm = len(_admissible_pairs(population))
        index_of = {cid: k for k, cid in enumerate(population.candidate_ids)}

        for n_units in unit_grid:
            rng = np.random.default_rng(int(seed) + 1000 * int(n_units) + int(k_adm))
            false_runs = 0
            for _ in range(int(n_replicates)):
                losses = draw_loss_matrix(
                    population, n_units, rng, distribution=distribution
                )
                audit = _audit_from_matrix(population, losses)
                result = confidence_pareto(
                    audit, complexity_map, alpha=alpha, delta=0.0,
                    n_resamples=int(n_resamples),
                    seed=int(rng.integers(1, 2**31 - 1)),
                )
                edges = [
                    (index_of[a], index_of[b]) for a, b in result.strict_dominance_edges
                ]
                false_runs += int(_false_edges(population, edges) > 0)

            reps = int(n_replicates)
            rate = false_runs / reps
            upper = _wilson_upper(false_runs, reps)
            if upper <= validated_bound:
                status = "validated"
            elif upper <= transitional_bound:
                status = "transitional"
            else:
                status = "outside"
            rows.append({
                "n_units": int(n_units),
                "n_candidates": population.n_candidates,
                "k_admissible": int(k_adm),
                "false_edge_rate": rate,
                "false_edge_upper_95": upper,
                "status": status,
                "log_k_over_g": float(math.log(max(k_adm, 2)) / float(n_units)),
                "n_replicates": reps,
                "alpha": float(alpha),
            })
    return rows


def critical_value_strategies(
    *,
    unit_grid: Sequence[int] = (24, 60, 150),
    n_null: int = 6,
    n_replicates: int = 600,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 31,
    distribution: str = "gaussian",
) -> list[dict[str, Any]]:
    """Ask whether small-unit anti-conservatism is repairable by the critical value.

    The multiplier bootstrap is asymptotically calibrated but runs hot at small
    ``G``.  Two possibilities: the multiplier approximation itself is optimistic
    there, in which case a different multiplier will not help and a conservative
    fallback is required; or the Gaussian multiplier specifically is at fault,
    in which case a cheaper repair exists.

    Four rules are evaluated on the *same* replicates, so the comparison is
    paired and free of between-run noise.  The pairwise risk differences and
    standard errors do not depend on the multiplier, only the critical value
    does, so one fit supports every rule except the Rademacher draw itself:

    ``multiplier_normal``      Gaussian multiplier max-T, the current default.
    ``multiplier_rademacher``  Rademacher multiplier max-T.
    ``bonferroni_t``           ``t_{1-alpha/K, G-1}`` over the K estimable
                               admissible pairs, a conservative closed form.
    ``hybrid``                 ``max(bootstrap, bonferroni)``, which cannot be
                               optimistic relative to either component.
    """
    from scipy.stats import t as student_t

    rows: list[dict[str, Any]] = []
    population = make_lab_population(
        n_null=int(n_null), n_dominated=0, n_near=0, n_decoy=0, seed=int(seed)
    )
    complexity_map = population.complexity_map()
    index_of = {cid: k for k, cid in enumerate(population.candidate_ids)}

    def _edges_at(result, critical: float) -> set[tuple[int, int]]:
        out = set()
        for comparison in result.comparisons:
            if not comparison.estimable:
                continue
            bound = comparison.risk_difference + critical * comparison.standard_error
            if bound < 0.0:
                out.add((
                    index_of[comparison.challenger_id],
                    index_of[comparison.incumbent_id],
                ))
        return out

    for n_units in unit_grid:
        rng = np.random.default_rng(int(seed) + int(n_units))
        counters = {k: 0 for k in
                    ("multiplier_normal", "multiplier_rademacher", "bonferroni_t", "hybrid")}
        critical_means = {k: 0.0 for k in ("multiplier_normal", "multiplier_rademacher",
                                           "bonferroni_t", "hybrid")}

        for _ in range(int(n_replicates)):
            losses = draw_loss_matrix(population, n_units, rng, distribution=distribution)
            audit = _audit_from_matrix(population, losses)
            draw_seed = int(rng.integers(1, 2**31 - 1))

            normal = confidence_pareto(
                audit, complexity_map, alpha=alpha, delta=0.0,
                n_resamples=int(n_resamples), seed=draw_seed, multiplier="normal",
            )
            rademacher = confidence_pareto(
                audit, complexity_map, alpha=alpha, delta=0.0,
                n_resamples=int(n_resamples), seed=draw_seed, multiplier="rademacher",
            )

            n_estimable = max(1, sum(1 for c in normal.comparisons if c.estimable))
            bonferroni = float(
                student_t.ppf(1.0 - alpha / n_estimable, max(1, n_units - 1))
            )
            hybrid = max(float(normal.critical_value), bonferroni)

            values = {
                "multiplier_normal": float(normal.critical_value),
                "multiplier_rademacher": float(rademacher.critical_value),
                "bonferroni_t": bonferroni,
                "hybrid": hybrid,
            }
            edge_sets = {
                "multiplier_normal": [
                    (index_of[a], index_of[b]) for a, b in normal.strict_dominance_edges
                ],
                "multiplier_rademacher": [
                    (index_of[a], index_of[b]) for a, b in rademacher.strict_dominance_edges
                ],
                "bonferroni_t": _edges_at(normal, bonferroni),
                "hybrid": _edges_at(normal, hybrid),
            }
            for name, edges in edge_sets.items():
                counters[name] += int(_false_edges(population, edges) > 0)
                critical_means[name] += values[name]

        reps = int(n_replicates)
        row: dict[str, Any] = {
            "n_units": int(n_units),
            "n_candidates": population.n_candidates,
            "alpha": float(alpha),
            "n_replicates": reps,
        }
        for name in counters:
            row[f"{name}_false_edge_rate"] = counters[name] / reps
            row[f"{name}_mean_critical_value"] = critical_means[name] / reps
        rows.append(row)
    return rows


def dominance_power(
    *,
    gaps: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    n_units: int = 24,
    n_replicates: int = 400,
    alpha: float = 0.05,
    n_resamples: int = 400,
    rho: float = 0.85,
    seed: int = 23,
    distribution: str = "gaussian",
) -> list[dict[str, Any]]:
    """Probability of correctly removing a truly dominated candidate.

    Confirms the procedure is not merely conservative: as the risk gap grows,
    the dominated candidate should be excluded with increasing probability.
    """
    rows: list[dict[str, Any]] = []
    for gap in gaps:
        population = make_lab_population(
            n_null=2, n_dominated=1, n_near=0, n_decoy=0,
            dominated_gap=float(gap), rho=rho, variance_spread=1.0, seed=int(seed),
        )
        complexity_map = population.complexity_map()
        dominated_ids = {
            cid for cid, grp in zip(population.candidate_ids, population.group)
            if grp == "dominated"
        }
        rng = np.random.default_rng(int(seed) + int(1000 * gap))
        removed = 0
        for _ in range(int(n_replicates)):
            losses = draw_loss_matrix(population, n_units, rng, distribution=distribution)
            audit = _audit_from_matrix(population, losses)
            result = confidence_pareto(
                audit, complexity_map, alpha=alpha, delta=0.0,
                n_resamples=int(n_resamples),
                seed=int(rng.integers(1, 2**31 - 1)),
            )
            removed += int(dominated_ids.isdisjoint(set(result.confidence_front)))
        reps = int(n_replicates)
        rows.append({
            "risk_gap": float(gap),
            "removal_probability": removed / reps,
            "n_units": int(n_units),
            "n_replicates": reps,
        })
    return rows
