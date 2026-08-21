# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Calibrated committee decisions: paired max-T over rows, sharded by design.

This module replaces the hand-crafted CoE vote gate (per-slice win/tie/loss
against ``_committee_tolerance``) with the same multiplier max-T machinery the
sealed audit uses.  The statistical framing is the one fixed in design review:

* the **rows** are the independent units (G large, exactly as in the non-CoE
  statistical layer);
* the **committee members** are the comparison family (M small), each compared
  against a common baseline (identity) on the same rows;
* witness **slices are compute partitions, not statistical units**.

Sharding is exact, not approximate.  Bootstrap multipliers are keyed by
``SeedSequence([seed, unit_key])`` per unit (counter-based, stream-position
free), so any worker regenerates the multipliers for its own rows with zero
communication.  A shard contributes only additive sufficient statistics
(``B x M`` weighted sums plus first/second moments), and the reduce is an
associative add: in exact arithmetic any partition of the rows reproduces the
single-shot decision identically.  In floating point, shard boundaries
reassociate the sums, so critical values agree to ~1 ulp relative rather than
bit for bit; verdicts are unaffected except on measure-zero knife edges.

Cluster keying: passing ``cluster_ids`` keys the multiplier by cluster rather
than by row, giving the cluster (block) bootstrap for rows that share a fit
(e.g. all rows scored under one refit of the ladder).  Per-row keying is the
default and matches the fixed-coefficient scope of the sealed audit.

Decision rule (simultaneous two-sided family at level ``alpha``): with
``d_m`` the mean paired delta (member minus baseline; negative is better),
``se_m`` its standard error and ``c`` the (1-alpha) quantile of the
bootstrapped ``max_m |T*_m|``,

* ``better``            if ``d_m + c * se_m < 0``
* ``worse``             if ``d_m - c * se_m > 0``
* ``indistinguishable`` otherwise.

Every decision carries the calibration-envelope licensing verdict from
:func:`select_inference_method` at ``(n_units=G, k_pre=M)``, and the verdict
is **dispatched, not merely recorded**: outside the validated envelope the
two-sided Bonferroni ``t_{1-alpha/(2M), G-1}`` critical value replaces the
bootstrap quantile over the same studentised statistics (the family is
two-sided, hence the ``2M`` split).  Small-M, large-G committee decisions sit
inside the measured grid via the monotone witness rule and keep the bootstrap.

Cluster-keyed decisions are the exception: there the row count overstates the
independent units and the per-row standard errors are too small, which the
cluster bootstrap's critical value compensates for and a t quantile cannot.
The Bonferroni fallback is therefore undefined under cluster keying; the
bootstrap is retained and the decision is explicitly marked as carrying no
envelope license.  The in-search gates that consume these decisions are
sequential, so each decision is locally calibrated; the sealed audit remains
the only end-to-end certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from nestynet_sr.stat_selection.calibration_profile import select_inference_method

__all__ = [
    "CommitteeShardStats",
    "CommitteeMemberVerdict",
    "CommitteeMaxTDecision",
    "committee_shard_stats",
    "reduce_committee_decision",
    "committee_maxt_decision",
    "maxt_decision_from_slice_rows",
]

DEFAULT_N_RESAMPLES = 2000
DEFAULT_ALPHA = 0.05


def _unit_multipliers(seed: int, unit_key: int, n_resamples: int) -> np.ndarray:
    """The B multipliers for one unit, independent of shard or stream order.

    Keyed by ``SeedSequence([seed, unit_key])`` so any worker holding any
    subset of units regenerates exactly the same draws.  Normal multipliers,
    matching the measured calibration profile.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(unit_key)]))
    return rng.standard_normal(int(n_resamples))


@dataclass(frozen=True)
class CommitteeShardStats:
    """Additive sufficient statistics from one shard of paired rows.

    All arrays are per-member (columns ordered as ``member_ids``).  Two shard
    objects with identical ``member_ids``, ``seed`` and ``n_resamples`` may be
    added; the reduce over any partition of the rows is exact.
    """

    member_ids: tuple[str, ...]
    seed: int
    n_resamples: int
    n_rows: int                      # complete-case paired rows in this shard
    n_dropped: int                   # rows dropped for non-finite entries
    delta_sum: np.ndarray            # (M,)   sum_i delta_i^m
    delta_sq_sum: np.ndarray         # (M,)   sum_i (delta_i^m)^2
    g_delta_sum: np.ndarray          # (B, M) sum_i g_i^(b) * delta_i^m
    g_sum: np.ndarray                # (B,)   sum_i g_i^(b)
    # Whether multipliers were keyed by cluster rather than by row.  Carried so
    # the reduce knows the row count overstates the independent units, which
    # forbids the Bonferroni-t fallback there.
    clustered: bool = False

    def __add__(self, other: "CommitteeShardStats") -> "CommitteeShardStats":
        if self.member_ids != other.member_ids:
            raise ValueError("shards disagree on member_ids; cannot reduce")
        if int(self.seed) != int(other.seed) or int(self.n_resamples) != int(other.n_resamples):
            raise ValueError("shards disagree on (seed, n_resamples); cannot reduce")
        if bool(self.clustered) != bool(other.clustered):
            raise ValueError("shards disagree on cluster keying; cannot reduce")
        return CommitteeShardStats(
            member_ids=self.member_ids,
            seed=self.seed,
            n_resamples=self.n_resamples,
            n_rows=int(self.n_rows) + int(other.n_rows),
            n_dropped=int(self.n_dropped) + int(other.n_dropped),
            delta_sum=self.delta_sum + other.delta_sum,
            delta_sq_sum=self.delta_sq_sum + other.delta_sq_sum,
            g_delta_sum=self.g_delta_sum + other.g_delta_sum,
            g_sum=self.g_sum + other.g_sum,
            clustered=bool(self.clustered),
        )


def committee_shard_stats(
    *,
    baseline_losses: Sequence[float],
    member_losses: Mapping[str, Sequence[float]],
    unit_keys: Sequence[int],
    seed: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    cluster_ids: Optional[Sequence[int]] = None,
) -> CommitteeShardStats:
    """Compute one shard's sufficient statistics from per-row losses.

    ``unit_keys`` are the global unit identities (e.g. absolute row numbers in
    the source CSV); they key the multipliers, so the same row always draws the
    same multiplier no matter which shard evaluates it.  ``cluster_ids``, when
    given, replace the multiplier key (block bootstrap): every row of a cluster
    shares the cluster's draw, while first/second moments still accumulate per
    row.

    Rows are complete-case: a row enters only if the baseline and every member
    loss is finite there, which keeps the comparison family on one common set
    of units.  Dropped rows are counted and reported.
    """
    member_ids = tuple(str(k) for k in member_losses.keys())
    if not member_ids:
        raise ValueError("at least one committee member is required")
    base = np.asarray(baseline_losses, dtype=np.float64).reshape(-1)
    n = base.size
    keys = np.asarray(unit_keys, dtype=np.int64).reshape(-1)
    if keys.size != n:
        raise ValueError("unit_keys must align with baseline_losses")
    if np.unique(keys).size != n:
        raise ValueError("unit_keys must be unique within a shard")
    losses = np.empty((n, len(member_ids)), dtype=np.float64)
    for j, mid in enumerate(member_ids):
        col = np.asarray(member_losses[mid], dtype=np.float64).reshape(-1)
        if col.size != n:
            raise ValueError(f"member {mid!r} losses must align with baseline_losses")
        losses[:, j] = col
    if cluster_ids is not None:
        clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
        if clusters.size != n:
            raise ValueError("cluster_ids must align with baseline_losses")
        mult_keys = clusters
    else:
        mult_keys = keys

    valid = np.isfinite(base) & np.all(np.isfinite(losses), axis=1)
    n_valid = int(np.count_nonzero(valid))
    B = int(n_resamples)
    M = len(member_ids)
    deltas = losses[valid] - base[valid, None]          # (n_valid, M)

    # Multipliers, one draw vector per distinct key.  Rows sharing a cluster
    # key share the draw; keys are counter-based, so a cluster split across
    # shards still sees the identical draw vector on both sides.
    valid_keys = mult_keys[valid]
    unique_keys, inverse = np.unique(valid_keys, return_inverse=True)
    draws = np.stack(
        [_unit_multipliers(seed, int(k), B) for k in unique_keys], axis=1
    ) if unique_keys.size else np.empty((B, 0), dtype=np.float64)
    g_matrix = draws[:, inverse] if unique_keys.size else np.empty((B, 0))

    # Inputs are complete-case finite by construction (filtered above).  The
    # errstate guard silences spurious FP-status flags some BLAS backends
    # (notably macOS Accelerate) raise on thin matmuls despite exact results.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return CommitteeShardStats(
            member_ids=member_ids,
            seed=int(seed),
            n_resamples=B,
            n_rows=n_valid,
            n_dropped=int(n - n_valid),
            delta_sum=deltas.sum(axis=0) if n_valid else np.zeros(M),
            delta_sq_sum=(deltas**2).sum(axis=0) if n_valid else np.zeros(M),
            g_delta_sum=g_matrix @ deltas if n_valid else np.zeros((B, M)),
            g_sum=g_matrix.sum(axis=1) if n_valid else np.zeros(B),
            clustered=cluster_ids is not None,
        )


@dataclass(frozen=True)
class CommitteeMemberVerdict:
    member_id: str
    verdict: str                     # "better" | "worse" | "indistinguishable"
    mean_delta: float                # member risk minus baseline risk
    se: float
    t_statistic: float
    ci_lower: float                  # simultaneous CI at the max-T critical value
    ci_upper: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "verdict": self.verdict,
            "mean_delta": self.mean_delta,
            "se": self.se,
            "t_statistic": self.t_statistic,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


@dataclass(frozen=True)
class CommitteeMaxTDecision:
    """One calibrated committee decision over G paired rows.

    ``best_member_id`` is the member with the most negative mean delta among
    those judged ``better``; ``None`` when no member significantly beats the
    baseline.  The decision is locally calibrated at level ``alpha``; the
    sealed audit remains the end-to-end certificate.
    """

    member_verdicts: tuple[CommitteeMemberVerdict, ...]
    best_member_id: Optional[str]
    n_units: int
    n_dropped: int
    alpha: float
    critical_value: float
    n_resamples: int
    seed: int
    inference_regime: Mapping[str, Any] = field(default_factory=dict)
    # Which rule produced ``critical_value``; proof of dispatch, mirroring
    # ConfidenceParetoResult.critical_value_method.
    critical_value_method: str = "multiplier_max_t"

    def verdict_for(self, member_id: str) -> str:
        for row in self.member_verdicts:
            if row.member_id == str(member_id):
                return row.verdict
        raise KeyError(member_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "committee_paired_maxt",
            "member_verdicts": [row.to_dict() for row in self.member_verdicts],
            "best_member_id": self.best_member_id,
            "n_units": int(self.n_units),
            "n_dropped": int(self.n_dropped),
            "alpha": float(self.alpha),
            "critical_value": float(self.critical_value),
            "n_resamples": int(self.n_resamples),
            "seed": int(self.seed),
            "critical_value_method": self.critical_value_method,
            "inference_regime": dict(self.inference_regime),
        }


def _bonferroni_two_sided_t_critical_value(
    *, alpha: float, n_members: int, n_units: int
) -> float:
    """Two-sided ``t_{1-alpha/(2M), G-1}`` over the M-member family.

    The committee family is two-sided (each member may be judged better or
    worse), so the Bonferroni split is ``alpha/(2M)``.  The closed form makes
    no small-G use of the multiplier approximation, which is why it may be
    used outside the calibrated envelope; it pays in power, never in level.
    """
    from scipy.stats import t as student_t

    members = max(1, int(n_members))
    dof = max(1, int(n_units) - 1)
    return float(student_t.ppf(1.0 - float(alpha) / (2.0 * members), dof))


def reduce_committee_decision(
    shards: Sequence[CommitteeShardStats],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> CommitteeMaxTDecision:
    """Reduce shard statistics to one calibrated committee decision.

    The reduce is associative and exact: any partition of the rows into
    shards, in any order, yields the identical decision.
    """
    if not shards:
        raise ValueError("at least one shard is required")
    total = shards[0]
    for shard in shards[1:]:
        total = total + shard
    G = int(total.n_rows)
    M = len(total.member_ids)
    if G < 2:
        raise ValueError(
            f"committee inference needs at least 2 complete-case rows, got {G}"
        )

    mean_delta = total.delta_sum / G
    # Unbiased per-member variance of the paired deltas from the moments.
    var = (total.delta_sq_sum - G * mean_delta**2) / (G - 1)
    var = np.clip(var, 0.0, None)
    se = np.sqrt(var / G)

    # Centered bootstrap perturbations of the mean delta:
    #   P[b, m] = (1/G) sum_i g_i^(b) (delta_i^m - mean_delta_m)
    perturb = (total.g_delta_sum - np.outer(total.g_sum, mean_delta)) / G
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(se[None, :] > 0.0, perturb / se[None, :], 0.0)
        t_obs = np.where(se > 0.0, mean_delta / se, np.where(mean_delta != 0.0, np.inf * np.sign(mean_delta), 0.0))

    # Decide the licensed method before computing the critical value, then
    # execute it.  Cluster keying forbids the t fallback: G counts rows, not
    # independent units, and per-row standard errors are too small there in
    # exactly the way the cluster bootstrap's critical value compensates for
    # and a t quantile cannot.
    regime = dict(select_inference_method(n_units=G, k_pre=M))
    if total.clustered:
        method_executed = "multiplier_max_t"
        regime["dispatch_note"] = (
            "cluster-keyed decision: the row count overstates the independent "
            "units, so the envelope lookup is advisory and the Bonferroni-t "
            "fallback is undefined; the cluster-bootstrap critical value is "
            "retained and this decision carries no envelope license"
        )
    elif str(regime.get("method")) == "bonferroni_t":
        method_executed = "bonferroni_t"
    else:
        method_executed = "multiplier_max_t"

    if method_executed == "bonferroni_t":
        critical = _bonferroni_two_sided_t_critical_value(
            alpha=float(alpha), n_members=M, n_units=G
        )
    else:
        family_max = np.max(np.abs(t_star), axis=1)
        try:
            critical = float(np.quantile(family_max, 1.0 - float(alpha), method="higher"))
        except TypeError:  # pragma: no cover - NumPy < 1.22 compatibility
            critical = float(np.quantile(family_max, 1.0 - float(alpha), interpolation="higher"))
    critical = max(0.0, critical)
    regime["method_executed"] = method_executed
    regime["critical_value"] = float(critical)

    verdicts: list[CommitteeMemberVerdict] = []
    for j, mid in enumerate(total.member_ids):
        lo = float(mean_delta[j] - critical * se[j])
        hi = float(mean_delta[j] + critical * se[j])
        if se[j] == 0.0:
            # Degenerate: deltas identical on every row.  Any nonzero mean is
            # deterministic, not sampled.
            verdict = (
                "better" if mean_delta[j] < 0.0
                else "worse" if mean_delta[j] > 0.0
                else "indistinguishable"
            )
        elif hi < 0.0:
            verdict = "better"
        elif lo > 0.0:
            verdict = "worse"
        else:
            verdict = "indistinguishable"
        verdicts.append(
            CommitteeMemberVerdict(
                member_id=mid,
                verdict=verdict,
                mean_delta=float(mean_delta[j]),
                se=float(se[j]),
                t_statistic=float(t_obs[j]),
                ci_lower=lo,
                ci_upper=hi,
            )
        )

    better = [row for row in verdicts if row.verdict == "better"]
    best = min(better, key=lambda row: row.mean_delta).member_id if better else None

    return CommitteeMaxTDecision(
        member_verdicts=tuple(verdicts),
        best_member_id=best,
        n_units=G,
        n_dropped=int(total.n_dropped),
        alpha=float(alpha),
        critical_value=critical,
        n_resamples=int(total.n_resamples),
        seed=int(total.seed),
        inference_regime=regime,
        critical_value_method=method_executed,
    )


def committee_maxt_decision(
    *,
    baseline_losses: Sequence[float],
    member_losses: Mapping[str, Sequence[float]],
    unit_keys: Sequence[int],
    seed: int,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    cluster_ids: Optional[Sequence[int]] = None,
) -> CommitteeMaxTDecision:
    """One-shot committee decision (single shard).

    Equivalent by construction to any sharded evaluation of the same rows:
    this simply builds one shard and reduces it.
    """
    shard = committee_shard_stats(
        baseline_losses=baseline_losses,
        member_losses=member_losses,
        unit_keys=unit_keys,
        seed=seed,
        n_resamples=n_resamples,
        cluster_ids=cluster_ids,
    )
    return reduce_committee_decision([shard], alpha=alpha)


def maxt_decision_from_slice_rows(
    *,
    baseline_rows: Mapping[int, tuple[int, Sequence[float]]],
    member_rows: Mapping[str, Mapping[int, tuple[int, Sequence[float]]]],
    seed: int,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    cluster_by_slice: bool = False,
) -> CommitteeMaxTDecision:
    """Bridge from per-slice witness row losses to one committee decision.

    ``baseline_rows`` maps ``slice_id -> (val_start, row_losses)`` where
    ``val_start`` is the slice's absolute first row in the source CSV, so
    ``val_start + i`` is a globally stable unit key; the CoE witness slices
    are disjoint by construction, which keeps the keys unique.  Only slices
    present for the baseline AND every member enter the decision (the paired
    family must share its units), and row counts must agree per slice.

    ``cluster_by_slice=True`` keys the bootstrap multipliers by slice id
    instead of by row: the block bootstrap for gates whose per-slice losses
    share a fit (the short-refit ladder).  Fixed-expression gates keep the
    default per-row keying, matching the fixed-coefficient scope of the
    sealed audit.
    """
    member_ids = tuple(str(k) for k in member_rows.keys())
    if not member_ids:
        raise ValueError("at least one committee member is required")
    common = sorted(
        set(baseline_rows.keys()).intersection(
            *[set(member_rows[m].keys()) for m in member_ids]
        )
    )
    if not common:
        raise ValueError("baseline and members share no witness slices")
    if cluster_by_slice and len(common) < 2:
        # With one cluster the centered cluster-bootstrap perturbation is
        # identically zero, so the critical value collapses to 0 and any
        # nonzero delta would look significant.  Refuse rather than be
        # silently anti-conservative.
        raise ValueError(
            "cluster bootstrap needs at least 2 shared witness slices; "
            f"got {len(common)}"
        )

    base_parts: list[np.ndarray] = []
    member_parts: dict[str, list[np.ndarray]] = {m: [] for m in member_ids}
    key_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    for slice_id in common:
        val_start, base_losses = baseline_rows[slice_id]
        base_arr = np.asarray(base_losses, dtype=np.float64).reshape(-1)
        for m in member_ids:
            m_start, m_losses = member_rows[m][slice_id]
            m_arr = np.asarray(m_losses, dtype=np.float64).reshape(-1)
            if int(m_start) != int(val_start) or m_arr.size != base_arr.size:
                raise ValueError(
                    f"slice {slice_id}: member {m!r} rows are not paired with the "
                    f"baseline (val_start {m_start} vs {val_start}, "
                    f"n {m_arr.size} vs {base_arr.size})"
                )
            member_parts[m].append(m_arr)
        base_parts.append(base_arr)
        key_parts.append(int(val_start) + np.arange(base_arr.size, dtype=np.int64))
        cluster_parts.append(np.full(base_arr.size, int(slice_id), dtype=np.int64))

    return committee_maxt_decision(
        baseline_losses=np.concatenate(base_parts),
        member_losses={m: np.concatenate(member_parts[m]) for m in member_ids},
        unit_keys=np.concatenate(key_parts),
        seed=seed,
        alpha=alpha,
        n_resamples=n_resamples,
        cluster_ids=np.concatenate(cluster_parts) if cluster_by_slice else None,
    )
