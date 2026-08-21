# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import itertools
import re

import torch

from nestynet_sr.sr_search.y_transforms import get_separability_y_ops

# Factor by which to relax derivative tolerances when hunting for complete
# (non-overlapping) separations. Overlapping splits are still allowed, but
# only after we have given disjoint splits a chance at this looser scale.
COMPLETE_TOL_FACTOR = 10.0

# Relaxed tolerance for generalized-additivity Lx/Ly 1D polynomial fits.
# Expressed as median(|residual|) / median(|signal|) along each axis.
# We treat GenAdd as a *hint*; Stage B will still reject bad candidates
# by validation loss, so this can be fairly lenient.
GENADD_REL_RES_TOL = 0.7
TRAPPED_REL_RES_TOL = 0.4


import math
from dataclasses import dataclass


@dataclass
class MultiplicativityOffset:
    """Result of offset detection in multiplicative separability.

    For expressions of the form y = c + f(x_A) * g(x_B), the offset c is
    estimated from b_vals = y - (u_i * u_j) / u_ij during the multiplicativity
    check. This dataclass captures the quality of that estimate.
    """
    b_hat: float           # Robust offset estimate (normalized scale)
    b_rel_mad: float       # Relative scatter across pairs: MAD(b) / |b_hat|
    support_frac: float    # Fraction of data points surviving the mask
    strong: bool           # True if evidence is strong enough to use


# Thresholds for offset detection (configurable)
OFFSET_MIN_ABS = 0.05       # |b_hat| must exceed this (normalized scale)
OFFSET_MAX_REL_MAD = 0.2    # Scatter must be below this
OFFSET_MIN_SUPPORT = 0.5    # At least 50% of points must contribute


@dataclass
class GenAddPairStats:
    i: int
    j: int
    rel_res_x: float
    rel_res_y: float
    ok: bool


def _fit_poly_1d(x, y, deg=2, eps=1e-12):
    x = x.view(-1)
    y = y.view(-1)
    X = torch.stack([x**d for d in range(deg + 1)], dim=-1)
    sol = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution
    return sol.view(-1)


def _eval_poly_1d(x, coeffs):
    powers = torch.stack([x**d for d in range(coeffs.numel())], dim=-1)
    return (powers * coeffs).sum(-1)


@dataclass
class GeneralizedAdditivityResult:
    X: tuple
    Y: tuple
    best_pair: GenAddPairStats
    ok: bool


@dataclass
class TrappedVariableResult:
    trapped_idx: int
    leaky_idx: int
    kind: str
    candidate_P: str
    rel_res: float
    ok: bool


@dataclass
class TrappedFactorization:
    ok: bool
    trapped_idx: int
    leaky_idx: int
    candidate_P: str
    P: torch.Tensor
    logB: torch.Tensor
    logA: torch.Tensor


def check_additivity(symb, derivatives, precision=0.0001, very_verbose=False):
    def map_to_symbols(indices):
        return [symb[i] for i in indices]

    def compute_max_metric(assignment, derivatives):
        """Compute max median cross-derivative for an assignment (worst pair)."""
        exclusive1 = {i for i, a in enumerate(assignment) if a == 0}
        exclusive2 = {i for i, a in enumerate(assignment) if a == 1}
        max_metric = 0.0
        for i in exclusive1:
            for j in exclusive2:
                med = torch.abs(derivatives[:, i, j]).median().item()
                max_metric = max(max_metric, med)
        return max_metric

    n_variables = len(symb)
    rest = None
    solutions = []
    # assign: 0 - exclusive to func1, 1 - exclusive to func2, 2 - in both
    for assignment in itertools.product([0, 1, 2], repeat=n_variables):
        group1 = {i for i, a in enumerate(assignment) if a in (0, 2)}
        group2 = {i for i, a in enumerate(assignment) if a in (1, 2)}
        exclusive1 = {i for i, a in enumerate(assignment) if a == 0}
        exclusive2 = {i for i, a in enumerate(assignment) if a == 1}
        if not exclusive1 or not exclusive2:
            continue
        valid = True
        for i in exclusive1:
            for j in exclusive2:
                if torch.abs(derivatives[:, i, j]).median() >= precision:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            if very_verbose:
                print(
                    f"Additivity between x{map_to_symbols(sorted(group1))} and x{map_to_symbols(sorted(group2))}"
                )
            non_intersecting = 2 not in assignment
            rank_val = min(len(group1), len(group2))
            solutions.append((non_intersecting, rank_val, assignment, group1, group2))
    if not solutions:
        if very_verbose:
            for i in range(n_variables):
                for j in range(n_variables):
                    if i != j:
                        print(
                            f"Derivs between x{map_to_symbols([i])} and x{map_to_symbols([j])}: {torch.abs(derivatives[:, i, j]).median().item()}"
                        )
            print(f"There is no additivity (in NN{map_to_symbols(_ for _ in range(n_variables))})")
        return False, -1, -1, False, rest, None, None
    # Sort: non-intersecting solutions come first (using not non_intersecting as key component), then by the size of the smallest group.
    solutions.sort(key=lambda x: (not x[0], x[1]))
    solution_best = solutions[0]
    group1 = sorted(solution_best[3])
    group2 = sorted(solution_best[4])
    complete_add = set(group1).isdisjoint(set(group2))
    rest = list(
        set(group1) & set(group2)
    )  # elements that are in both groups, i.e. not uniquely assigned.

    # Compute critical_metric using MAX (worst pair) instead of min
    # This better represents the quality of the split - lower max = better separation
    assignment = solution_best[2]
    critical_metric = compute_max_metric(assignment, derivatives)

    # Find qualifying overlapping solutions — return a short list rather than
    # just the single "best", because the lowest max-metric trivially favours
    # maximally-overlapping splits (fewer exclusive pairs → lower max-metric).
    overlapping_solutions = [s for s in solutions if not s[0]]  # non_intersecting=False
    best_overlapping_info = None
    if overlapping_solutions:
        overlapping_with_metrics = [
            (s, compute_max_metric(s[2], derivatives)) for s in overlapping_solutions
        ]

        # Deduplicate by (frozenset(group1), frozenset(group2)) — mirror
        # assignments produce the same split.
        seen = set()
        deduped = []
        for s, m in overlapping_with_metrics:
            key = (frozenset(s[3]), frozenset(s[4]))
            mirror = (frozenset(s[4]), frozenset(s[3]))
            if key not in seen and mirror not in seen:
                seen.add(key)
                deduped.append((s, m))

        best_overlapping_list = []

        # 1) Best by minimal overlap (fewest shared vars), then max-metric as tiebreak
        deduped_by_overlap = sorted(
            deduped, key=lambda x: (sum(1 for a in x[0][2] if a == 2), x[1])
        )
        if deduped_by_overlap:
            best_overlapping_list.append(deduped_by_overlap[0])

        # 2) Best by max-metric (may have higher overlap), added only if distinct
        deduped_by_metric = sorted(deduped, key=lambda x: x[1])
        for item in deduped_by_metric:
            if item not in best_overlapping_list:
                best_overlapping_list.append(item)
                if len(best_overlapping_list) >= 3:
                    break

        # Convert to list of (group1, group2, metric) tuples
        if best_overlapping_list:
            best_overlapping_info = [
                (sorted(s[3]), sorted(s[4]), m)
                for s, m in best_overlapping_list
            ]

    if very_verbose:
        for i in range(n_variables):
            for j in range(n_variables):
                if i != j:
                    print(
                        f"Derivs between x{map_to_symbols([i])} and x{map_to_symbols([j])}: {torch.abs(derivatives[:, i, j]).median().item()}"
                    )
    if not rest:
        print(
            f"All additive for x{map_to_symbols(group1)}, complement x{map_to_symbols(group2)} (in NN{map_to_symbols(_ for _ in range(n_variables))})"
        )
    else:
        print(
            f"There is additivity between x{map_to_symbols(group1)} and x{map_to_symbols(group2)} (in NN{map_to_symbols(_ for _ in range(n_variables))}), rest: {map_to_symbols(rest)}"
        )
    if best_overlapping_info:
        for g1o, g2o, mo in best_overlapping_info:
            n_shared = len(set(g1o) & set(g2o))
            print(
                f"  overlapping candidate: x{map_to_symbols(g1o)} + x{map_to_symbols(g2o)}, "
                f"shared={n_shared}, max-metric={mo:.2e}"
            )

    return True, group1, group2, complete_add, rest, critical_metric, best_overlapping_info


def _compute_offset_info(passing_b_medians, N_total, eps=1e-10):
    """
    Aggregate offset estimates from multiple passing (i,j) pairs.

    Parameters
    ----------
    passing_b_medians : list of (b_median, support_count)
        Offset estimates from each passing multiplicative pair.
    N_total : int
        Total number of data points.

    Returns
    -------
    MultiplicativityOffset or None
        Aggregated offset statistics, or None if no passing pairs.
    """
    if not passing_b_medians:
        return None

    b_values = [b for b, _ in passing_b_medians]
    support_counts = [s for _, s in passing_b_medians]

    # Robust aggregate: median of all b estimates
    b_hat = float(sorted(b_values)[len(b_values) // 2])

    # Scatter: MAD of b estimates relative to b_hat
    abs_deviations = [abs(b - b_hat) for b in b_values]
    b_mad = float(sorted(abs_deviations)[len(abs_deviations) // 2])
    b_rel_mad = b_mad / (abs(b_hat) + eps) if abs(b_hat) > eps else float('inf')

    # Support fraction: minimum support across all pairs
    min_support = min(support_counts)
    support_frac = float(min_support) / float(N_total) if N_total > 0 else 0.0

    # Determine if offset evidence is strong
    strong = (
        abs(b_hat) > OFFSET_MIN_ABS and
        b_rel_mad < OFFSET_MAX_REL_MAD and
        support_frac > OFFSET_MIN_SUPPORT
    )

    return MultiplicativityOffset(
        b_hat=b_hat,
        b_rel_mad=b_rel_mad,
        support_frac=support_frac,
        strong=strong,
    )


def _test_multiplicativity_with_offset(
    d2ydx2_vals, dydx_vals, y_vals, b_offset, precision=0.0001, eps=1e-10, min_support_frac=0.5
):
    """
    Test if all variable pairs pass the multiplicativity derivative criterion
    when using a fixed offset b_offset.

    Uses masking instead of clamping to avoid artificially regularizing points
    where y ≈ b_offset (near singularity). Requires minimum support fraction.

    Returns (all_pass, n_passing_pairs, total_pairs, worst_metric).
    worst_metric is the highest med_derivs among passing pairs (quality indicator).
    """
    y = y_vals.squeeze()
    n_variables = dydx_vals.shape[1]
    N = y.numel()
    min_support = int(min_support_frac * N)

    # Compute denominator with fixed offset
    denom = y - b_offset

    # Mask for valid points (not near singularity)
    denom_mask = torch.abs(denom) > eps

    # Mask out extreme-|y| points: NN approximation is unreliable near
    # poles/singularities, which produce the largest function values.
    if N > 40:
        y_abs = torch.abs(y)
        q95 = torch.quantile(y_abs, 0.95)
        inlier_mask = y_abs <= q95
        denom_mask = denom_mask & inlier_mask

    n_pass = 0
    n_total = 0
    worst_metric = 0.0  # Track worst (highest) med_derivs among passing pairs
    for i in range(n_variables):
        for j in range(n_variables):
            if j == i:
                continue
            n_total += 1

            # Also mask where d2y is near-zero (avoid 0/denom artifacts)
            d2_mask = torch.abs(d2ydx2_vals[:, i, j]) > eps
            mask = denom_mask & d2_mask

            if mask.sum() < min_support:
                # Not enough support for this pair - fail this pair
                continue

            d2lnydx2_vals = (
                d2ydx2_vals[:, i, j][mask] / denom[mask]
                - dydx_vals[:, i][mask] * dydx_vals[:, j][mask] / denom[mask] ** 2
            )
            med_derivs = torch.median(torch.abs(d2lnydx2_vals))

            if med_derivs < precision:
                n_pass += 1
                worst_metric = max(worst_metric, med_derivs.item())

    all_pass = (n_pass == n_total) if n_total > 0 else False
    return all_pass, n_pass, n_total, worst_metric


def check_multiplicativity(
    symb,
    d2ydx2_vals,
    dydx_vals,
    y_vals,
    precision=0.0001,
    mad_b_precision=0.1,
    big=1e10,
    eps=1e-10,
    very_verbose=False,
):
    # The multiplicative separability condition is less trivial than the additive condition.

    # Say we have
    # y = f(x1) * g(x2) = f * g
    # Diff wrt x1:
    # =>  dy/dx1 = df/dx1 * g
    # => (dy/dx1) / y = (df/dx1) / f

    # Diff wrt x2:
    # (d^2y/dx1 dx2) / y -  (dy/dx1)/y^2 * (dy/dx2) = 0

    # So our multiplicative separability constraint is:
    # y * d^2y/dx1 dx2 = (dy/dx1) * (dy/dx2)

    # For the case where an additive constant is present, say y = f(x1) * g(x2) + b
    # all the above stays the same, except that “naked” versions of y become (y-b).

    # So we want to check if there is a "b" such that:
    # (y-b) * d^2y/dx1 dx2 = (dy/dx1) * (dy/dx2)
    # => b = y - (dy/dx1) * (dy/dx2) / (d^2y/dx1 dx2)    has small scatter.

    # Since we're building up the network with additive and multiplicative parts, we may end up with, e.g.:
    # F = h(x0) + f(x1)*g(x2) + c

    # where h(x0=0) isn’t zero. However, we can always imagine that we have fitted a H(x0) which does obey H(x0=0)=0. Then

    # F = H(x0) + f(x1)*g(x2) + b

    # We can then define

    # y = F - H(x0) = f(x1)*g(x2) + b

    # Just like above, this means we have separability if

    # (y-b) * d^2y/dx1 dx2 = (dy/dx1) * (dy/dx2)
    # =>
    # b = y - (dy/dx1) * (dy/dx2) / (d^2y/dx1 dx2)

    # i.e. we can find a value of b for each set of y, and its derivatives.
    # So we have multiplicative separability if there is a small scatter in the values of b derived from the sample.

    def map_to_symbols(indices):  # Helper function to map indices to symbols
        return [symb[i] for i in indices]

    n_variables = len(symb)
    best_sep = big
    found_any = False
    best_g1, best_g2 = [], []
    rest = None

    y = y_vals.squeeze()
    y_scale = torch.median(torch.abs(y))
    if y_scale == 0:
        y_scale = y_scale + eps

    # === PRE-CHECK: Try candidate offsets to detect offset-multiplicativity ===
    # This handles the case where plain multiplicativity fails due to offset.

    # Generate candidates from data distribution
    y_sorted = torch.sort(y)[0]
    n = y.numel()
    candidates = [
        0.0,                                    # pure multiplicative (no offset)
        torch.median(y).item(),                 # median
        y_sorted[max(0, n//10)].item(),         # 10th percentile
        y_sorted[min(n-1, 9*n//10)].item(),     # 90th percentile
    ]
    # Remove duplicates and sort by absolute value (prefer smaller offsets)
    candidates = sorted(set(candidates), key=abs)

    best_offset_candidate = None
    best_offset_metric = None

    for b_cand in candidates:
        all_pass, n_pass, n_total, metric = _test_multiplicativity_with_offset(
            d2ydx2_vals, dydx_vals, y_vals, b_cand, precision=precision, eps=eps
        )

        if all_pass and abs(b_cand) > OFFSET_MIN_ABS:
            # Non-trivial offset makes all pairs pass
            best_offset_candidate = b_cand
            best_offset_metric = metric
            break

    # If a non-zero offset candidate makes all pairs pass, return early
    if best_offset_candidate is not None:
        # Determine grouping: all variables are multiplicatively related
        group1 = [0]
        group2 = list(range(1, n_variables))

        # Build offset_info with the discovered offset
        offset_info = MultiplicativityOffset(
            b_hat=float(best_offset_candidate),  # same scale as y_vals
            b_rel_mad=0.0,
            support_frac=1.0,
            strong=True,
        )

        if very_verbose:
            print(f"[Offset pre-check] Found offset-multiplicativity with b={best_offset_candidate:.4f}")

        # Return actual quality metric from offset-multiplicative detection
        return True, group1, group2, True, None, offset_info, best_offset_metric

    # === END PRE-CHECK - Fall through to existing pair-by-pair logic ===

    # Mask out extreme-|y| points: NN approximation is unreliable near
    # poles/singularities, which produce the largest function values.
    # Applied to both b-value and d²ln metric computations.
    N = y.numel()
    if N > 40:
        y_abs = torch.abs(y)
        q95 = torch.quantile(y_abs, 0.95)
        inlier_mask = y_abs <= q95
    else:
        inlier_mask = torch.ones(N, dtype=torch.bool, device=y.device)

    # Avoid division by (near-)zero second derivatives when computing b_vals.
    mask_d2 = torch.abs(d2ydx2_vals) > eps
    # Combine with inlier mask (broadcast: inlier_mask is [N], mask_d2 is [N, V, V])
    mask_d2 = mask_d2 & inlier_mask.unsqueeze(-1).unsqueeze(-1)
    sign_d2 = torch.sign(d2ydx2_vals)
    sign_d2 = torch.where(sign_d2 == 0, torch.ones_like(sign_d2), sign_d2)
    safe_d2 = torch.where(mask_d2, d2ydx2_vals, sign_d2 * eps)

    b_vals = (
        y.unsqueeze(-1).unsqueeze(-1)
        - (dydx_vals.unsqueeze(-1) * dydx_vals.unsqueeze(-2)) / safe_d2
    )

    multiplicativity = torch.zeros(n_variables, n_variables, dtype=torch.bool, device=y_vals.device)

    # Collect offset estimates from passing pairs for aggregate analysis.
    # Also track a conservative per-pair "quality" score so Stage A can safely
    # rank non-singleton disjoint multiplicative splits.
    N_total = y.numel()
    passing_b_medians = []  # List of (b_median, support_count) for passing pairs

    # Legacy diagnostic: smallest passing |d2 ln(y-b)| median across any pair.
    min_passing_metric = float("inf")

    # Conservative pair-quality matrix (same scale as `precision`):
    #   quality(i,j) = max( med|d2 ln(y-b)|, (mad_b_scaled / mad_b_precision) * precision )
    # so that pairs with noisy inferred offsets are penalized even if the
    # derivative metric alone is small.
    pair_quality = [[float("inf")] * n_variables for _ in range(n_variables)]

    def _group_cross_max_metric(g1, g2):
        """Worst (max) conservative metric across cross-pairs between g1 and g2.

        We symmetrize by taking max(metric[i][j], metric[j][i]) for each pair.
        """
        if not g1 or not g2:
            return None
        m = 0.0
        for a in g1:
            for b in g2:
                if a == b:
                    continue
                q1 = pair_quality[a][b]
                q2 = pair_quality[b][a]
                q = q1 if q1 >= q2 else q2
                if q > m:
                    m = q
        return m

    for i in range(n_variables):
        all_multiplicative_for_i = True
        for j in range(n_variables):
            if j == i:
                continue

            mask_ij = mask_d2[:, i, j]
            if not mask_ij.any():
                if very_verbose:
                    print(
                        f"Skipping multiplicativity check for pair (x{i}, x{j}) because all d2≈0."
                    )
                all_multiplicative_for_i = False
                continue

            b_list = b_vals[:, i, j][mask_ij]  # only where d2 is safely non-zero
            b_median = torch.median(b_list.clone().detach())

            mad_b = torch.median(torch.abs(b_list.clone().detach() - b_median))
            mad_b_scaled = mad_b / (y_scale + eps)

            # Mask out points where y≈b_median to avoid huge spikes in d2ln
            # (consistent with _test_multiplicativity_with_offset).
            # Also exclude extreme-|y| inlier points (already excluded from b_vals).
            denom = y - b_median
            denom_mask = (torch.abs(denom) > eps) & inlier_mask
            min_support = int(0.5 * y.numel())
            if denom_mask.sum() < min_support:
                all_multiplicative_for_i = False
                continue

            d2lnydx2_vals = (
                d2ydx2_vals[:, i, j][denom_mask] / denom[denom_mask]
                - dydx_vals[:, i][denom_mask] * dydx_vals[:, j][denom_mask] / denom[denom_mask] ** 2
            )
            med_derivs_ij = torch.median(torch.abs(d2lnydx2_vals))

            # Conservative per-pair score on the same scale as `precision`.
            med_val = float(med_derivs_ij.item())
            mad_val = float(mad_b_scaled.item())
            if mad_b_precision > 0:
                quality = max(med_val, (mad_val / mad_b_precision) * float(precision))
            else:
                quality = med_val
            pair_quality[i][j] = quality

            # print(f"i={i}, j={j}, med_derivs_ij={med_derivs_ij}, mad_b_scaled={mad_b_scaled}, b_median={b_median}, mad_b={mad_b}")
            if med_derivs_ij < precision and mad_b_scaled < mad_b_precision:
                if very_verbose:
                    print(
                        f"There is     multiplicativity between x{i} and x{j}: "
                        f"derivs {med_derivs_ij.item()}, mad_b {mad_b_scaled.item()}"
                    )
                multiplicativity[i, j] = True
                if med_derivs_ij < best_sep:
                    best_sep = med_derivs_ij
                    best_g1, best_g2 = [i], [j]
                found_any = True

                # Track min metric among passing pairs (this is the "best" pair)
                if med_val < min_passing_metric:
                    min_passing_metric = med_val

                # Collect offset estimate for this passing pair
                support_count = int(mask_ij.sum().item())
                passing_b_medians.append((float(b_median.item()), support_count))
            else:
                if very_verbose:
                    print(
                        f"There is NOT multiplicativity between x{i} and x{j}: "
                        f"derivs {med_derivs_ij.item()}, mad_b {mad_b_scaled.item()}"
                    )
                all_multiplicative_for_i = False

        # If x_i is multiplicative with all others, return that group
        if all_multiplicative_for_i:
            group1 = [i]
            group2 = [k for k in range(n_variables) if k != i]
            print(
                f"All multiplicative for x{map_to_symbols(group1)}, complement "
                f"x{map_to_symbols(group2)} "
                f"(in NN{map_to_symbols(_ for _ in range(n_variables))})"
            )
            # Compute offset info from collected passing pairs
            offset_info = _compute_offset_info(passing_b_medians, N_total)

            # Conservative group metric: worst cross-pair quality between the two groups.
            group_metric = _group_cross_max_metric(group1, group2)
            if group_metric is None:
                group_metric = (
                    min_passing_metric if min_passing_metric != float("inf") else None
                )

            return True, group1, group2, True, rest, offset_info, group_metric

    # If we found at least one pair, produce a grouping
    if found_any:
        rest = [x for x in range(n_variables) if x not in best_g1 and x not in best_g2]

        rest1, rest2, rest_both = [], [], []
        for x in rest:
            if not multiplicativity[best_g1[0], x] and not multiplicativity[best_g2[0], x]:
                rest1.append(x)
                rest2.append(x)
            elif multiplicativity[best_g1[0], x] and not multiplicativity[best_g2[0], x]:
                rest2.append(x)
            elif not multiplicativity[best_g1[0], x] and multiplicativity[best_g2[0], x]:
                rest1.append(x)
            elif multiplicativity[best_g1[0], x] and multiplicativity[best_g2[0], x]:
                # rest1.append(x)
                # rest2.append(x)
                rest_both.append(x)
        # if an element of rest_both is mutiplicative with all elements of rest1, then it should be in rest2 and vice versa. Otherwise, it should be in both.
        for x in rest_both:
            if all([multiplicativity[x, y] for y in rest1]):
                rest2.append(x)
                # print(f"appending x{map_to_symbols([x])} to rest2")
            elif all([multiplicativity[x, y] for y in rest2]):
                rest1.append(x)
                # print(f"appending x{map_to_symbols([x])} to rest1")
            else:
                rest1.append(x)
                rest2.append(x)

        # print(f"Initial multiplicativity between x{map_to_symbols(best_g1)} and x{map_to_symbols(best_g2)}, rest: {map_to_symbols(rest)} (in NN{map_to_symbols(_ for _ in range(n_variables))})")
        group1 = sorted(best_g1 + rest1)
        group2 = sorted(best_g2 + rest2)
        print(
            f"There is multiplicativity between x{map_to_symbols(group1)} and "
            f"x{map_to_symbols(group2)} "
            f"(in NN{map_to_symbols(_ for _ in range(n_variables))}), "
            f"rest: {map_to_symbols(rest)}"
        )
        # Compute offset info from collected passing pairs
        offset_info = _compute_offset_info(passing_b_medians, N_total)

        # Conservative group metric: worst cross-pair quality between the two groups.
        group_metric = _group_cross_max_metric(group1, group2)
        if group_metric is None:
            group_metric = min_passing_metric if min_passing_metric != float("inf") else None

        return True, group1, group2, False, rest, offset_info, group_metric

    print(f"There is no multiplicativity (in NN{map_to_symbols(_ for _ in range(n_variables))})")
    return False, -1, -1, False, rest, None, None


def generalized_additivity_pair_from_derivs(
    x_batch,
    f,
    grad,
    hess,
    i,
    j,
    min_points=200,
    poly_deg=2,
    fx_cut=1e-6,
    fy_cut=1e-6,
    eps=1e-12,
):
    f = f.view(-1)
    gi = grad[:, i].view(-1)
    gj = grad[:, j].view(-1)
    hii = hess[:, i, i].view(-1)
    hij = hess[:, i, j].view(-1)
    hjj = hess[:, j, j].view(-1)
    mask = (
        torch.isfinite(gi)
        & torch.isfinite(gj)
        & torch.isfinite(hii)
        & torch.isfinite(hij)
        & torch.isfinite(hjj)
    )
    mask &= gi.abs() > fx_cut
    mask &= gj.abs() > fy_cut
    if mask.sum() < min_points:
        return GenAddPairStats(i=i, j=j, rel_res_x=math.inf, rel_res_y=math.inf, ok=False)
    x_i = x_batch[:, i][mask]
    x_j = x_batch[:, j][mask]
    gi = gi[mask]
    gj = gj[mask]
    hii = hii[mask]
    hij = hij[mask]
    hjj = hjj[mask]

    Lx = hii / gi - hij / gj
    Ly = hij / gi - hjj / gj

    mask_Lx = torch.isfinite(Lx)
    mask_Ly = torch.isfinite(Ly)
    Lx = Lx[mask_Lx]
    Ly = Ly[mask_Ly]
    x_i_L = x_i[mask_Lx]
    x_j_L = x_j[mask_Ly]

    if Lx.numel() < min_points or Ly.numel() < min_points:
        return GenAddPairStats(i=i, j=j, rel_res_x=math.inf, rel_res_y=math.inf, ok=False)

    coeff_x = _fit_poly_1d(x_i_L, Lx, deg=poly_deg)
    pred_Lx = _eval_poly_1d(x_i_L, coeff_x)
    rx = (Lx - pred_Lx).abs()
    scale_x = Lx.abs().median()
    if not torch.isfinite(scale_x) or scale_x < eps:
        scale_x = rx.abs().median().clamp_min(eps)
    rel_res_x = float((rx.median() / scale_x).item())

    coeff_y = _fit_poly_1d(x_j_L, Ly, deg=poly_deg)
    pred_Ly = _eval_poly_1d(x_j_L, coeff_y)
    ry = (Ly - pred_Ly).abs()
    scale_y = Ly.abs().median()
    if not torch.isfinite(scale_y) or scale_y < eps:
        scale_y = ry.abs().median().clamp_min(eps)
    rel_res_y = float((ry.median() / scale_y).item())

    # Treat the *better* of (Lx, Ly) as the score. In practice, for true
    # generalized-additive structure G(h(x) + k(y)) it's quite common that
    # one of {Lx, Ly} is estimated well while the other is noisy because of
    # derivative noise / NN approximation error.
    score = min(rel_res_x, rel_res_y)
    ok = score < GENADD_REL_RES_TOL

    return GenAddPairStats(
        i=i,
        j=j,
        rel_res_x=rel_res_x,
        rel_res_y=rel_res_y,
        ok=ok,
    )


def check_generalized_additivity_from_derivs(
    x_batch,
    f,
    grad,
    hess,
    X_group,
    Y_group,
    poly_deg: int = 2,
    min_points: int = 200,
):
    best = GenAddPairStats(
        i=-1,
        j=-1,
        rel_res_x=math.inf,
        rel_res_y=math.inf,
        ok=False,
    )
    for i in X_group:
        for j in Y_group:
            stats = generalized_additivity_pair_from_derivs(
                x_batch=x_batch,
                f=f,
                grad=grad,
                hess=hess,
                i=i,
                j=j,
                min_points=min_points,
                poly_deg=poly_deg,
            )
            # Score pairs by the cleaner of the two directions; we only
            # need one axis where the Lx/Ly field looks 1D-like to build
            # a reasonable g(h(x) + k(y)) candidate and let Stage B's LM
            # refine/reject it.
            score = min(stats.rel_res_x, stats.rel_res_y)
            best_score = min(best.rel_res_x, best.rel_res_y)
            if score < best_score:
                best = stats
    ok = best.ok
    return GeneralizedAdditivityResult(
        X=tuple(int(i) for i in X_group),
        Y=tuple(int(j) for j in Y_group),
        best_pair=best,
        ok=ok,
    )


def check_separability(
    symb,
    index,
    model,
    datagen,
    precision_sum=0.0001,
    precision_mult=0.0001,
    device=None,
    eps=1.0e-12,
    very_verbose=False,
):
    var_x_all = []
    for batch_data in datagen:
        x, _ = batch_data
        var_x_all.append(x)
    var_x = torch.cat(var_x_all, dim=0)

    # When we're checking separability over a subset of variables `symb`,
    # it's important that the *other* coordinates do not inject a
    # sample‑dependent offset into y. Otherwise terms like log(x0) will
    # make the inferred "b" in the multiplicativity test look noisy and
    # we incorrectly reject true multiplicative structure inside the
    # active subset.  We therefore freeze all non‑symb coordinates to a
    # representative constant (their median over the dataset) before
    # evaluating the model and its derivatives.
    if var_x.dim() != 2:
        raise ValueError(f"Expected 2D input [N, d] in check_separability, got shape {var_x.shape}")
    n_total_vars = var_x.size(1)
    all_idx = torch.arange(n_total_vars, dtype=torch.long)
    symb_idx = torch.as_tensor(symb, dtype=torch.long)
    mask_rest = torch.ones_like(all_idx, dtype=torch.bool)
    mask_rest[symb_idx] = False
    rest_idx = all_idx[mask_rest]
    if rest_idx.numel() > 0:
        # Compute medians on CPU; we'll move to `device` afterwards.
        med_rest = var_x[:, rest_idx].median(dim=0).values
        var_x = var_x.clone()
        var_x[:, rest_idx] = med_rest.unsqueeze(0).expand(var_x.size(0), -1)

    var_x = var_x.to(device)
    y_vals = model(var_x)
    dydx_vals = model.grad(var_x)
    d2ydx2_vals = model.grad_grad(var_x)

    print(
        f"symb: {symb}, index: {index}, var_x: {var_x.shape}, y_vals: {y_vals.shape}, dydx_vals: {dydx_vals.shape}, d2ydx2_vals: {d2ydx2_vals.shape}"
    )

    # Our separability checks are currently only possible for scalar-valued functions, so we're slicing out the first element here.
    # We're currently ignoring all the other output dimensions of the model. ********BEWARE********
    # This could be generalized to multiple output dimensions, but it would require a more complex separability check.

    y_med = torch.median(y_vals[:, 0, ...])
    y_mad = torch.median(torch.abs(y_vals[:, 0, ...] - y_med))
    if y_mad == 0:
        y_mad = y_mad + eps
    print(f"Median of sub-model: {y_med}, median absolute deviation: {y_mad}")

    # Check for degenerate (near-constant) output.
    # If y_mad is tiny, the network didn't learn anything meaningful and
    # separability tests trivially pass with metric=0.0 (false positive).
    DEGENERATE_THRESHOLD = 1e-10  # Absolute threshold for y_mad
    if y_mad < DEGENERATE_THRESHOLD:
        print(f"[check_separability] Degenerate output detected: y_mad={y_mad:.2e} < {DEGENERATE_THRESHOLD:.0e}")
        # Return empty proposals and None metric to indicate unreliable test
        return [], None, None, float(y_mad.item()), None

    # Also check for degenerate gradients (zero first derivatives = no structure).
    # A truly separable function has non-zero gradients in at least some direction.
    # Without this check, networks that learn near-constant outputs with tiny
    # variation (above DEGENERATE_THRESHOLD but with ~zero gradients) would
    # trivially pass separability tests because cross-derivatives are also ~zero.
    dydx_raw = dydx_vals[:, 0, ...]  # [N, n_total_vars] - raw before normalization
    grad_mads = []
    for i in range(dydx_raw.shape[1]):
        grad_i = dydx_raw[:, i]
        grad_med = torch.median(grad_i)
        grad_mad = torch.median(torch.abs(grad_i - grad_med))
        grad_mads.append(grad_mad.item())
    max_grad_mad = max(grad_mads) if grad_mads else 0.0

    GRAD_DEGENERATE_THRESHOLD = 1e-10
    if max_grad_mad < GRAD_DEGENERATE_THRESHOLD:
        print(f"[check_separability] Degenerate gradients: max_grad_mad={max_grad_mad:.2e} < {GRAD_DEGENERATE_THRESHOLD:.0e}")
        return [], None, None, float(y_mad.item()), None

    y_med = torch.abs(y_med) + eps

    # Normalize by the median absolute deviation of the *sub-model* (the idea being that the precision of the separability relates to the precision of the sub-model)
    y_vals = y_vals[:, 0, ...] / y_mad  # [N]
    dydx_vals = dydx_vals[:, 0, ...] / y_mad  # [N, n_total_vars]
    d2ydx2_vals = d2ydx2_vals[:, 0, ...] / y_mad  # [N, n_total_vars, n_total_vars]

    # Restrict derivatives to the current symbol subset; downstream
    # routines interpret indices 0..len(symb)-1 as local coordinates.
    idx = torch.as_tensor(symb, dtype=torch.long, device=dydx_vals.device)
    dydx_sub = dydx_vals[:, idx]  # [N, n_symb]
    d2ydx2_sub = d2ydx2_vals[:, idx][:, :, idx]  # [N, n_symb, n_symb]

    rest_add = None
    rest_mult = None

    # --- STRICT pass (current base-precision behaviour) ---
    add_strict, g1_add_strict, g2_add_strict, complete_add_strict, resta_strict, add_metric_strict, best_overlap_strict = check_additivity(
        symb, d2ydx2_sub, precision_sum, very_verbose=very_verbose
    )
    mult_strict, g1_mult_strict, g2_mult_strict, complete_mult_strict, restm_strict, offset_strict, mult_metric_strict = (
        check_multiplicativity(symb, d2ydx2_sub, dydx_sub, y_vals, precision_mult, very_verbose=very_verbose)
    )

    # --- LOOSE pass: only used to find *complete* (non-overlapping) splits ---
    add_loose = mult_loose = False
    complete_add_loose = complete_mult_loose = False
    g1_add_loose = g2_add_loose = None
    g1_mult_loose = g2_mult_loose = None
    offset_loose = None
    add_metric_loose = mult_metric_loose = None

    if not (complete_add_strict or complete_mult_strict):
        prec_sum_loose = precision_sum * COMPLETE_TOL_FACTOR
        prec_mult_loose = precision_mult * COMPLETE_TOL_FACTOR

        add_loose, g1_add_loose, g2_add_loose, complete_add_loose, _, add_metric_loose, _ = check_additivity(
            symb, d2ydx2_sub, prec_sum_loose, very_verbose=very_verbose
        )
        mult_loose, g1_mult_loose, g2_mult_loose, complete_mult_loose, _, offset_loose, mult_metric_loose = check_multiplicativity(
            symb, d2ydx2_sub, dydx_sub, y_vals, prec_mult_loose, very_verbose=very_verbose
        )

    def map_to_symbols(indices):  # Helper function to map indices to symbols
        return [symb[i] for i in indices]

    # Convert "rest" indices from local (0..len(symb)-1) to global x-indices,
    # but only from the STRICT pass. If we end up accepting a loose complete
    # split, we conceptually do not have overlapping "rest" variables at this
    # level.
    if resta_strict:
        resta_global = [symb[i] for i in resta_strict]
        rest_add = resta_global if rest_add is None else rest_add + resta_global
    if restm_strict:
        restm_global = [symb[i] for i in restm_strict]
        rest_mult = restm_global if rest_mult is None else rest_mult + restm_global

    proposed_separabily_list = []
    winning_metric = None  # Track the critical_metric from the first (winning) split

    # 1) STRICT complete splits first (cleanest, non-overlapping)
    # Candidate format is:
    #   [op, group1, group2, offset_info, metric]
    # where metric is the critical diagnostic (lower is better) for that split.
    # This is appended as an optional 5th element; callers that only need
    # the first 4 fields can safely ignore it.
    if complete_mult_strict:
        proposed_separabily_list.append(
            [
                torch.multiply,
                map_to_symbols(g1_mult_strict),
                map_to_symbols(g2_mult_strict),
                offset_strict,
                mult_metric_strict,
            ]
        )
        if winning_metric is None:
            winning_metric = mult_metric_strict
    if complete_add_strict:
        proposed_separabily_list.append(
            [
                torch.add,
                map_to_symbols(g1_add_strict),
                map_to_symbols(g2_add_strict),
                None,
                add_metric_strict,
            ]
        )
        if winning_metric is None:
            winning_metric = add_metric_strict

        # Also add qualifying overlapping solutions (best_overlap_strict is now
        # a list of (group1, group2, metric) tuples, or None).
        if best_overlap_strict is not None:
            for g1_overlap, g2_overlap, overlap_metric in best_overlap_strict:
                if overlap_metric < add_metric_strict * 0.5:
                    proposed_separabily_list.append(
                        [
                            torch.add,
                            map_to_symbols(g1_overlap),
                            map_to_symbols(g2_overlap),
                            None,
                            overlap_metric,
                        ]
                    )
                    n_shared = len(set(g1_overlap) & set(g2_overlap))
                    print(
                        f"  [Overlapping additive split proposed: "
                        f"x{map_to_symbols(g1_overlap)} + x{map_to_symbols(g2_overlap)}, "
                        f"shared={n_shared}, max-metric={overlap_metric:.2e} "
                        f"vs disjoint={add_metric_strict:.2e}]"
                    )

    # 2) If there was no strict complete split, try LOOSE complete splits
    if not complete_mult_strict and complete_mult_loose:
        proposed_separabily_list.append(
            [
                torch.multiply,
                map_to_symbols(g1_mult_loose),
                map_to_symbols(g2_mult_loose),
                offset_loose,
                mult_metric_loose,
            ]
        )
        if winning_metric is None:
            winning_metric = mult_metric_loose
        # For a loose complete split we treat this as purely disjoint at this level
        rest_mult = None
    if not complete_add_strict and complete_add_loose:
        proposed_separabily_list.append(
            [
                torch.add,
                map_to_symbols(g1_add_loose),
                map_to_symbols(g2_add_loose),
                None,
                add_metric_loose,
            ]
        )
        if winning_metric is None:
            winning_metric = add_metric_loose
        rest_add = None

    # 3) STRICT partial (overlapping) splits as a last resort
    if mult_strict and not complete_mult_strict:
        proposed_separabily_list.append(
            [
                torch.multiply,
                map_to_symbols(g1_mult_strict),
                map_to_symbols(g2_mult_strict),
                offset_strict,
                mult_metric_strict,
            ]
        )
        if winning_metric is None:
            winning_metric = mult_metric_strict
    if add_strict and not complete_add_strict:
        proposed_separabily_list.append(
            [
                torch.add,
                map_to_symbols(g1_add_strict),
                map_to_symbols(g2_add_strict),
                None,
                add_metric_strict,
            ]
        )
        if winning_metric is None:
            winning_metric = add_metric_strict

    # If no separability was found, compute the min median cross-derivative
    # as a diagnostic metric (this is the "best" pair that still failed)
    if winning_metric is None:
        n_symb = len(symb)
        min_median_cross = float("inf")
        for i in range(n_symb):
            for j in range(n_symb):
                if i != j:
                    med_val = torch.abs(d2ydx2_sub[:, i, j]).median().item()
                    if med_val < min_median_cross:
                        min_median_cross = med_val
        if min_median_cross != float("inf"):
            winning_metric = min_median_cross

    # Return y_mad (in original scale) for denormalization of offset
    y_mad_scalar = float(y_mad.item())
    return proposed_separabily_list, rest_add, rest_mult, y_mad_scalar, winning_metric


def _sample_plane_for_pair(datagen, i: int, j: int, n_points: int = 2048):
    xs = []
    iterator = datagen() if callable(datagen) else datagen
    for bi, batch in enumerate(iterator):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.view(x.size(0), -1).detach().cpu()
        xs.append(x)
        if sum(t.size(0) for t in xs) >= n_points:
            break
    if not xs:
        raise RuntimeError("No data available in _sample_plane_for_pair")
    X_all = torch.cat(xs, dim=0)
    x_ref = X_all.median(dim=0).values
    t_i = X_all[:, i]
    t_j = X_all[:, j]
    ti_min, ti_max = float(t_i.min().item()), float(t_i.max().item())
    tj_min, tj_max = float(t_j.min().item()), float(t_j.max().item())
    m = max(4, int(math.sqrt(n_points)))
    ti = torch.linspace(ti_min, ti_max, m)
    tj = torch.linspace(tj_min, tj_max, m)
    grid_i, grid_j = torch.meshgrid(ti, tj, indexing="ij")
    X_plane = x_ref.repeat(m * m, 1)
    X_plane[:, i] = grid_i.reshape(-1)
    X_plane[:, j] = grid_j.reshape(-1)
    return X_plane


def check_generalized_additivity_ops(
    model,
    datagen,
    X_group,
    Y_group,
    device,
    dtype,
    n_points=2048,
    poly_deg=2,
):
    i = int(X_group[0])
    j = int(Y_group[0])
    X_plane = _sample_plane_for_pair(datagen, i=i, j=j, n_points=n_points).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        f = model.forward(X_plane).view(-1)
        grad = model.grad(X_plane).view(-1, X_plane.shape[1])
        hess = model.grad_grad(X_plane).view(-1, X_plane.shape[1], X_plane.shape[1])
    res = check_generalized_additivity_from_derivs(
        x_batch=X_plane,
        f=f,
        grad=grad,
        hess=hess,
        X_group=X_group,
        Y_group=Y_group,
        poly_deg=poly_deg,
        min_points=n_points // 4,
    )
    bp = res.best_pair
    print(
        f"[GenAdd] X_group={res.X}, Y_group={res.Y}, "
        f"best_pair=(i={bp.i}, j={bp.j}), "
        f"rel_res_x={bp.rel_res_x:.3g}, rel_res_y={bp.rel_res_y:.3g}, ok={res.ok}"
    )
    return res


def trapped_variable_ops(
    model,
    datagen,
    trapped_idx: int,
    leaky_idx: int,
    device,
    dtype,
    n_points: int = 2048,
    kind: str = "multiplicative",
    candidate_P: str = "product",
    min_points: int = 200,
):
    T = int(trapped_idx)
    L = int(leaky_idx)
    X = _sample_plane_for_pair(datagen, i=T, j=L, n_points=n_points)
    X = X.to(device=device, dtype=dtype)
    with torch.no_grad():
        y = model.forward(X).view(-1)
        g = model.grad(X).view(-1, X.shape[1])
    dyT = g[:, T]
    if kind == "multiplicative":
        y_abs = y.abs()
        m_y = torch.isfinite(y_abs)
        if not m_y.any():
            return TrappedVariableResult(
                trapped_idx=T,
                leaky_idx=L,
                kind=kind,
                candidate_P=candidate_P,
                rel_res=float("inf"),
                ok=False,
            )
        y_scale = y_abs[m_y].median().clamp_min(1e-12)
        eps_y = 1e-6 * y_scale + 1e-12
        sign_y = torch.sign(y)
        sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
        y_safe = sign_y * y_abs.clamp_min(eps_y)
        gT = dyT / y_safe
        y_mask = y_abs > eps_y
    else:
        gT = dyT
        y_mask = torch.ones_like(gT, dtype=torch.bool)
    xT = X[:, T]
    xL = X[:, L]
    if candidate_P == "product":
        P = xT * xL
        dP = xL
    elif candidate_P == "sum":
        P = xT + xL
        dP = torch.ones_like(xT)
    else:
        raise ValueError(f"Unknown candidate_P {candidate_P}")
    mask = torch.isfinite(P) & torch.isfinite(gT) & torch.isfinite(dP)
    mask &= dP.abs() > 1e-8
    mask &= y_mask
    P = P[mask]
    gT = gT[mask]
    dP = dP[mask]
    if P.numel() < min_points:
        return TrappedVariableResult(
            trapped_idx=T,
            leaky_idx=L,
            kind=kind,
            candidate_P=candidate_P,
            rel_res=float("inf"),
            ok=False,
        )
    Q = gT / dP
    maskQ = torch.isfinite(Q)
    P = P[maskQ]
    Q = Q[maskQ]
    N = P.numel()
    if N < min_points:
        return TrappedVariableResult(
            trapped_idx=T,
            leaky_idx=L,
            kind=kind,
            candidate_P=candidate_P,
            rel_res=float("inf"),
            ok=False,
        )
    order = torch.argsort(P)
    _P_sorted = P[order]
    Q_sorted = Q[order]
    bins = max(4, min(32, int(math.sqrt(N))))
    edges = torch.linspace(0, float(N), steps=bins + 1, device=P.device)
    edges = edges.round().long()
    Q_pred_sorted = torch.empty_like(Q_sorted)
    Q_pred_sorted.copy_(Q_sorted)
    for b in range(bins):
        start = int(edges[b].item())
        end = int(edges[b + 1].item())
        if end <= start:
            continue
        q_seg = Q_sorted[start:end]
        q_med = q_seg.median()
        Q_pred_sorted[start:end] = q_med
    inv = torch.empty_like(order)
    inv[order] = torch.arange(N, device=order.device)
    Q_pred = Q_pred_sorted[inv]
    r = (Q - Q_pred).abs()
    scale = Q.abs().median()
    eps = 1e-12
    if (not torch.isfinite(scale)) or scale < eps:
        scale = r.abs().median().clamp_min(eps)
    rel_res = float((r.median() / scale).item())
    ok = rel_res < TRAPPED_REL_RES_TOL
    return TrappedVariableResult(
        trapped_idx=T,
        leaky_idx=L,
        kind=kind,
        candidate_P=candidate_P,
        rel_res=rel_res,
        ok=ok,
    )


def factor_trapped_multiplicative(
    model,
    datagen,
    trapped_idx: int,
    leaky_idx: int,
    device,
    dtype,
    candidate_P: str = "product",
    n_points: int = 4096,
    min_points: int = 400,
):
    T = int(trapped_idx)
    L = int(leaky_idx)
    X = _sample_plane_for_pair(datagen, i=T, j=L, n_points=n_points)
    X = X.to(device=device, dtype=dtype)
    with torch.no_grad():
        y = model.forward(X).view(-1)
        g = model.grad(X).view(-1, X.shape[1])
    dyT = g[:, T]
    y_abs = y.abs()
    m_y = torch.isfinite(y_abs)
    if not m_y.any():
        return TrappedFactorization(False, T, L, candidate_P, y.new_empty((0,)), None, None)
    y_scale = y_abs[m_y].median().clamp_min(1e-12)
    eps_y = 1e-6 * y_scale + 1e-12
    sign_y = torch.sign(y)
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
    y_safe = sign_y * y_abs.clamp_min(eps_y)
    Lvals = torch.log(y_abs.clamp_min(eps_y))
    xT = X[:, T]
    xL = X[:, L]
    if candidate_P == "product":
        P = xT * xL
        dP = xL
    elif candidate_P == "sum":
        P = xT + xL
        dP = torch.ones_like(xT)
    else:
        raise ValueError
    g_log = dyT / y_safe
    mask = torch.isfinite(P) & torch.isfinite(dP) & torch.isfinite(g_log)
    mask &= dP.abs() > 1e-8
    mask &= y_abs > eps_y
    P = P[mask]
    dP = dP[mask]
    g_log = g_log[mask]
    Lvals = Lvals[mask]
    X = X[mask]
    if P.numel() < min_points:
        return TrappedFactorization(False, T, L, candidate_P, P, None, None)
    Q = g_log / dP
    maskQ = torch.isfinite(Q)
    P = P[maskQ]
    Q = Q[maskQ]
    Lvals = Lvals[maskQ]
    X = X[maskQ]
    if P.numel() < min_points:
        return TrappedFactorization(False, T, L, candidate_P, P, None, None)
    N = P.numel()
    order = torch.argsort(P)
    P_s = P[order]
    Q_s = Q[order]
    dP_s = P_s[1:] - P_s[:-1]
    H_mid = 0.5 * (Q_s[1:] + Q_s[:-1])
    dP_s_clamped = torch.where(dP_s.abs() < 1e-6, torch.sign(dP_s) * 1e-6, dP_s)
    dlogB = H_mid * dP_s_clamped
    logB_s = torch.zeros_like(P_s)
    logB_s[1:] = torch.cumsum(dlogB, dim=0)
    logB_s = logB_s - logB_s.median()
    inv = torch.empty_like(order)
    inv[order] = torch.arange(N, device=order.device)
    logB = logB_s[inv]
    logA = Lvals - logB
    return TrappedFactorization(True, T, L, candidate_P, P, logB, logA)


def check_separability_ops(
    symb,
    index,
    model,
    datagen,
    precision_sum=0.0001,
    precision_mult=0.0001,
    device=None,
    eps=1.0e-8,
    y_transform_names=None,
    very_verbose=False,
):
    var_x_all = []
    var_y_all = []
    for batch_data in datagen:
        x, y = batch_data
        var_x_all.append(x)
        var_y_all.append(y)
    var_x = torch.cat(var_x_all, dim=0)
    var_y = torch.cat(var_y_all, dim=0)

    # Same logic as in check_separability: when probing separability
    # under alternative y‑ops for a given symbol subset, freeze all
    # non‑symb coordinates so that they behave as a genuine constant
    # offset rather than a noisy, sample‑dependent "b".
    if var_x.dim() != 2:
        raise ValueError(
            f"Expected 2D input [N, d] in check_separability_ops, got shape {var_x.shape}"
        )
    n_total_vars = var_x.size(1)
    all_idx = torch.arange(n_total_vars, dtype=torch.long)
    symb_idx = torch.as_tensor(symb, dtype=torch.long)
    mask_rest = torch.ones_like(all_idx, dtype=torch.bool)
    mask_rest[symb_idx] = False
    rest_idx = all_idx[mask_rest]
    if rest_idx.numel() > 0:
        med_rest = var_x[:, rest_idx].median(dim=0).values
        var_x = var_x.clone()
        var_x[:, rest_idx] = med_rest.unsqueeze(0).expand(var_x.size(0), -1)

    var_x = var_x.to(device)
    var_y = var_y.to(device)
    y_vals = model(var_x)
    dydx_vals = model.grad(var_x)
    d2ydx2_vals = model.grad_grad(var_x)

    print(
        f"symb: {symb}, index: {index}, var_x: {var_x.shape}, y_vals: {y_vals.shape}, dydx_vals: {dydx_vals.shape}, d2ydx2_vals: {d2ydx2_vals.shape}"
    )

    # Our separability checks are currently only possible for scalar-valued functions, so we're slicing out the first element here.
    # We're currently ignoring all the other output dimensions of the model. ********BEWARE********
    # This could be generalized to multiple output dimensions, but it would require a more complex separability check.
    y_vals = y_vals[:, 0, ...]
    dydx_vals = dydx_vals[:, 0, ...]
    d2ydx2_vals = d2ydx2_vals[:, 0, ...]

    transform_specs, y_ops, dy_ops, d2y_ops = get_separability_y_ops(names=y_transform_names)

    candidate_sep_ops = [False] * len(y_ops)
    for i, (spec, op, dop, d2op) in enumerate(zip(transform_specs, y_ops, dy_ops, d2y_ops)):
        op_str = spec.name
        print(f"Trying transformation {op_str}")
        try:
            transformed_var_y = op(var_y)
            if torch.isnan(transformed_var_y).any() or torch.isinf(transformed_var_y).any():
                raise ValueError("Invalid y DATA values from op")
            transformed_y_vals = op(y_vals)
            if torch.isnan(transformed_y_vals).any() or torch.isinf(transformed_y_vals).any():
                raise ValueError("Invalid values from op")
            tmp_dop = dop(y_vals)
            if torch.isnan(tmp_dop).any() or torch.isinf(tmp_dop).any():
                raise ValueError("Invalid values from dop")
            tmp_d2op = d2op(y_vals)
            if torch.isnan(tmp_d2op).any() or torch.isinf(tmp_d2op).any():
                raise ValueError("Invalid values from d2op")
            transformed_dydx_vals = torch.einsum("b,bn->bn", tmp_dop, dydx_vals)
            outer_dydx = torch.einsum("bi,bj->bij", dydx_vals, dydx_vals)
            transformed_d2ydx2_vals = torch.einsum(
                "b,bij->bij", tmp_d2op, outer_dydx
            ) + torch.einsum("b,bij->bij", tmp_dop, d2ydx2_vals)
        except Exception as e:
            print("Skipping transformation", op_str, "due to domain error:", e)
            continue

        y_med = torch.median(transformed_var_y)
        y_mad = torch.median(torch.abs(transformed_var_y - y_med))
        if y_mad == 0:
            y_mad = y_mad + eps
        print(
            f"Median of model: {y_med}, median absolute deviation: {y_mad}, precision: {precision_sum}, {precision_mult}"
        )

        transformed_y_vals = transformed_y_vals / y_mad  # [N]
        transformed_dydx_vals = transformed_dydx_vals / y_mad  # [N, n_total_vars]
        transformed_d2ydx2_vals = transformed_d2ydx2_vals / y_mad  # [N, n_total_vars, n_total_vars]

        # Again, restrict to current symbol subset so that the math in
        # check_additivity / check_multiplicativity matches the local indexing.
        idx = torch.as_tensor(symb, dtype=torch.long, device=transformed_dydx_vals.device)
        t_dydx_sub = transformed_dydx_vals[:, idx]  # [N, n_symb]
        t_d2_sub = transformed_d2ydx2_vals[:, idx][:, :, idx]  # [N, n_symb, n_symb]

        add, g1_add, g2_add, complete_add, resta_new, _, _ = check_additivity(
            symb, t_d2_sub, precision_sum, very_verbose=very_verbose
        )
        mult, g1_mult, g2_mult, complete_mult, restm_new, _, _ = check_multiplicativity(
            symb, t_d2_sub, t_dydx_sub, transformed_y_vals, precision_mult, very_verbose=very_verbose
        )

        if add or mult:
            print(f"Found separability candidate separability for op {op_str}, {i}\n")
            candidate_sep_ops[i] = True
        else:
            print(f"No candidate separability for op {op_str}, {i}\n")
            candidate_sep_ops[i] = False

    print("Recap:")
    for spec, possible in zip(transform_specs, candidate_sep_ops):
        if possible:
            print(f"Transformation {spec.name} is a candidate for separability: {possible}")
    print()

    return candidate_sep_ops


# Recursively parse a "prefix" expression (operator first, then subexpressions)
def parse_prefix(tokens, idx=0):
    op_map = {
        torch.add: "+",
        torch.multiply: "*",
        torch.sub: "-",
        torch.div: "/",
    }

    if idx >= len(tokens):
        return "", idx

    current = tokens[idx]

    # If current is an operator, parse the next two subexpressions
    if callable(current) and current in op_map:
        left_expr, next_idx = parse_prefix(tokens, idx + 1)
        right_expr, next_idx = parse_prefix(tokens, next_idx)
        return f"({left_expr} {op_map[current]} {right_expr})", next_idx

    # If current is a list, parse that entire sub-list
    elif isinstance(current, list):
        return f"NN{current}", idx + 1

    # If current is an integer, interpret it as a neural-network block
    elif isinstance(current, int):
        return f"NN[{current}]", idx + 1

    # Fallback for anything else (e.g. literal number)
    else:
        return str(current), idx + 1


class Var:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Var({self.name})"


class BinOp:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinOp({self.op}, {self.left}, {self.right})"


def tokenize(expr_str):
    token_pattern = r"\(|\)|\+|\*|NN\[[^\]]*\]"
    tokens = re.findall(token_pattern, expr_str)
    return tokens


def parse_expression(tokens, i=0):
    (left_node, i) = parse_term(tokens, i)
    while i < len(tokens) and tokens[i] == "+":
        op = tokens[i]
        i += 1
        (right_node, i) = parse_term(tokens, i)
        left_node = BinOp(op, left_node, right_node)
    return (left_node, i)


def parse_term(tokens, i=0):
    (left_node, i) = parse_factor(tokens, i)
    while i < len(tokens) and tokens[i] == "*":
        op = tokens[i]
        i += 1
        (right_node, i) = parse_factor(tokens, i)
        left_node = BinOp(op, left_node, right_node)
    return (left_node, i)


def parse_factor(tokens, i=0):
    t = tokens[i]
    if t == "(":
        i += 1
        (node, i) = parse_expression(tokens, i)
        if i >= len(tokens) or tokens[i] != ")":
            raise ValueError("Missing closing parenthesis")
        i += 1
        return (node, i)
    else:
        # Must be something like NN[...] (including possible spaces)
        if not (t.startswith("NN[") and t.endswith("]")):
            raise ValueError(f"Unexpected token: {t}")
        node = Var(t)
        i += 1
        return (node, i)


def parse(expr_str):
    ts = tokenize(expr_str)
    (node, idx) = parse_expression(ts, 0)
    if idx != len(ts):
        raise ValueError("Extra tokens after valid expression.")
    return node


def precedence(op):
    if op == "+":
        return 1
    elif op == "*":
        return 2
    return 0


def to_string(ast, parent_op=None):
    if isinstance(ast, Var):
        return ast.name

    if isinstance(ast, BinOp):
        left_str = to_string(ast.left, ast.op)
        right_str = to_string(ast.right, ast.op)
        s = f"{left_str} {ast.op} {right_str}"

        if parent_op is not None:
            if precedence(ast.op) < precedence(parent_op):
                s = f"({s})"
        return s

    raise ValueError("Unknown AST node type.")


def simplify_expression_string(expr_str):
    ast = parse(expr_str)
    return to_string(ast)


# ──────────────────────────────────────────────────────────────────────────────
# Ratio-Invariance and Coupled-Leaf Detection
# ──────────────────────────────────────────────────────────────────────────────

# Threshold for Euler operator test (homogeneous degree 0)
RATIO_INVARIANCE_EULER_TOL = 0.05

# Threshold for coupled-leaf ratio polynomial fit
COUPLED_RATIO_REL_RMS_TOL = 0.02


@dataclass
class RatioInvarianceResult:
    """Result of ratio-invariance test for homogeneous degree-0 functions."""

    ok: bool
    xi_idx: int
    xj_idx: int
    euler_score: float  # median(|Euler|) / scale(f)
    ratio_direction: str  # "xj/xi" or "xi/xj"


def check_ratio_invariance(
    model,
    datagen,
    xi_idx: int,
    xj_idx: int,
    device,
    dtype,
    threshold: float = RATIO_INVARIANCE_EULER_TOL,
    n_points: int = 2048,
) -> RatioInvarianceResult:
    """
    Test if f(xi, xj) ≈ h(xj/xi) using the Euler derivative criterion.

    For a function f that is homogeneous of degree 0 in (xi, xj):
        xi * ∂f/∂xi + xj * ∂f/∂xj = 0

    This test checks if the Euler operator is approximately zero.

    Parameters
    ----------
    model : torch.nn.Module
        Model that provides .forward(X) and .grad(X) methods.
    datagen : iterable
        Data generator yielding (x, y) batches.
    xi_idx, xj_idx : int
        Indices of the two variables to test.
    device, dtype : torch device and dtype
    threshold : float
        Maximum relative Euler score for acceptance.
    n_points : int
        Number of points to sample.

    Returns
    -------
    RatioInvarianceResult
        Result including ok status, score, and ratio direction.
    """
    # Collect data
    X_list = []
    for batch in datagen:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        X_list.append(x.to(device=device, dtype=dtype))
        if sum(t.size(0) for t in X_list) >= n_points:
            break

    if not X_list:
        return RatioInvarianceResult(
            ok=False, xi_idx=xi_idx, xj_idx=xj_idx, euler_score=float("inf"), ratio_direction=""
        )

    X = torch.cat(X_list, dim=0)[:n_points]

    # Evaluate model and gradients
    with torch.no_grad():
        f = model.forward(X)
        grad = model.grad(X)

    # Handle output dimension
    if f.ndim == 2:
        f = f[:, 0]
    if grad.ndim == 3:
        grad = grad[:, 0, :]

    xi = X[:, xi_idx]
    xj = X[:, xj_idx]
    df_dxi = grad[:, xi_idx]
    df_dxj = grad[:, xj_idx]

    # Euler operator: E = xi * ∂f/∂xi + xj * ∂f/∂xj
    # For homogeneous degree-0: E = 0
    Euler = xi * df_dxi + xj * df_dxj

    # Compute relative score
    f_scale = f.abs().median().clamp_min(1e-12)
    euler_score = float((Euler.abs().median() / f_scale).item())

    ok = euler_score < threshold

    # Determine ratio direction (which one makes sense for the domain)
    ratio_xj_xi = xj / xi.clamp_min(1e-12)
    ratio_xi_xj = xi / xj.clamp_min(1e-12)

    # Check which ratio has better bounded range
    range_xj_xi = ratio_xj_xi.max() - ratio_xj_xi.min()
    range_xi_xj = ratio_xi_xj.max() - ratio_xi_xj.min()

    ratio_direction = "xj/xi" if range_xj_xi < range_xi_xj else "xi/xj"

    return RatioInvarianceResult(
        ok=ok,
        xi_idx=xi_idx,
        xj_idx=xj_idx,
        euler_score=euler_score,
        ratio_direction=ratio_direction,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Homogeneous Degree Detection (non-zero degree)
# ──────────────────────────────────────────────────────────────────────────────

# Candidate degrees to test for homogeneity peel
HOMOGENEOUS_DEGREE_CANDIDATES = (-4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0)

# Threshold for Euler operator test (non-zero homogeneous degree)
HOMOGENEOUS_DEGREE_EULER_TOL = 0.05


@dataclass
class HomogeneousDegreeResult:
    """Result of homogeneous degree-k test for non-zero k.

    When ok=True, the function f(xi, xj) is approximately homogeneous of degree k:
        xi * ∂f/∂xi + xj * ∂f/∂xj = k * f

    This means f can be rewritten as:
        f(xi, xj) = xi^k * h(xj/xi)

    where h is a univariate function of the ratio.
    """

    ok: bool
    degree: float  # The detected homogeneous degree k
    euler_residual: float  # median(|Euler - k*f|) / scale(f)
    ratio_direction: str  # "xj/xi" or "xi/xj" - preferred ratio variable
    power_var_idx: int  # Index of the variable to raise to power k (denominator)
    ratio_var_idx: int  # Index of the numerator variable in the ratio


def check_homogeneous_degree(
    model,
    datagen,
    xi_idx: int,
    xj_idx: int,
    device,
    dtype,
    degree_candidates: tuple = HOMOGENEOUS_DEGREE_CANDIDATES,
    threshold: float = HOMOGENEOUS_DEGREE_EULER_TOL,
    n_points: int = 2048,
) -> HomogeneousDegreeResult:
    """
    Test if f(xi, xj) is homogeneous of non-zero degree k in (xi, xj).

    For a function f that is homogeneous of degree k:
        xi * ∂f/∂xi + xj * ∂f/∂xj = k * f

    This generalizes the Euler criterion used for ratio-invariance (k=0).
    When degree k is detected, f can be written as:
        f(xi, xj) = xi^k * h(xj/xi)

    Parameters
    ----------
    model : torch.nn.Module
        Model that provides .forward(X) and .grad(X) methods.
    datagen : iterable
        Data generator yielding (x, y) batches.
    xi_idx, xj_idx : int
        Indices of the two variables to test.
    device, dtype : torch device and dtype
    degree_candidates : tuple
        Candidate degrees to test (default: -2, -1, -0.5, 0.5, 1, 2).
    threshold : float
        Maximum relative Euler score for acceptance.
    n_points : int
        Number of points to sample.

    Returns
    -------
    HomogeneousDegreeResult
        Result including ok status, degree, score, and ratio direction.
    """
    # Collect data
    X_list = []
    for batch in datagen:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        X_list.append(x.to(device=device, dtype=dtype))
        if sum(t.size(0) for t in X_list) >= n_points:
            break

    if not X_list:
        return HomogeneousDegreeResult(
            ok=False,
            degree=0.0,
            euler_residual=float("inf"),
            ratio_direction="",
            power_var_idx=xi_idx,
            ratio_var_idx=xj_idx,
        )

    X = torch.cat(X_list, dim=0)[:n_points]

    # Evaluate model and gradients
    with torch.no_grad():
        f = model.forward(X)
        grad = model.grad(X)

    # Handle output dimension
    if f.ndim == 2:
        f = f[:, 0]
    if grad.ndim == 3:
        grad = grad[:, 0, :]

    xi = X[:, xi_idx]
    xj = X[:, xj_idx]
    df_dxi = grad[:, xi_idx]
    df_dxj = grad[:, xj_idx]

    # Euler operator (without degree factor): E = xi * ∂f/∂xi + xj * ∂f/∂xj
    Euler_raw = xi * df_dxi + xj * df_dxj

    # Scale for normalization
    f_scale = f.abs().median().clamp_min(1e-12)

    # Test each candidate degree
    best_score = float("inf")
    best_degree = 0.0

    for k in degree_candidates:
        # For homogeneous degree k: Euler = k * f
        # Residual = Euler - k*f
        residual = Euler_raw - k * f
        score = float((residual.abs().median() / f_scale).item())

        if score < best_score:
            best_score = score
            best_degree = k

    ok = best_score < threshold

    # Determine ratio direction (which one makes sense for the domain)
    # For f = xi^k * h(xj/xi), we use ratio r = xj/xi
    # Use sign-preserving epsilon: x_safe = sign(x) * max(|x|, eps), with sign(0)=1
    eps = 1e-12
    xi_sign = xi.sign()
    xi_sign = torch.where(xi_sign == 0, torch.ones_like(xi_sign), xi_sign)
    xi_safe = xi_sign * xi.abs().clamp_min(eps)
    xj_sign = xj.sign()
    xj_sign = torch.where(xj_sign == 0, torch.ones_like(xj_sign), xj_sign)
    xj_safe = xj_sign * xj.abs().clamp_min(eps)
    ratio_xj_xi = xj / xi_safe
    ratio_xi_xj = xi / xj_safe

    # Check which ratio has better bounded range (more stable for NN)
    range_xj_xi = ratio_xj_xi.max() - ratio_xj_xi.min()
    range_xi_xj = ratio_xi_xj.max() - ratio_xi_xj.min()

    if range_xj_xi < range_xi_xj:
        ratio_direction = "xj/xi"
        power_var_idx = xi_idx  # xi^k
        ratio_var_idx = xj_idx  # xj/xi
    else:
        ratio_direction = "xi/xj"
        power_var_idx = xj_idx  # xj^k
        ratio_var_idx = xi_idx  # xi/xj

    return HomogeneousDegreeResult(
        ok=ok,
        degree=best_degree,
        euler_residual=best_score,
        ratio_direction=ratio_direction,
        power_var_idx=power_var_idx,
        ratio_var_idx=ratio_var_idx,
    )


@dataclass
class CoupledLeafRatioResult:
    """Result of coupled-leaf ratio detection."""

    ok: bool
    ratio_rel_rms: float
    poly_form: str  # Human-readable polynomial form
    poly_coeffs: dict  # Coefficients for reconstruction


def check_coupled_leaf_ratio_from_derivs(
    model,
    datagen,
    affine_idx_F: int,
    affine_idx_G: int,
    shared_var_idxs: tuple,
    device,
    dtype,
    threshold: float = COUPLED_RATIO_REL_RMS_TOL,
    n_points: int = 2048,
) -> CoupledLeafRatioResult:
    """
    Check if the ratio F/G of two coupled leaves is a simple polynomial.

    For an expression like:
        y = poly_F(x_a) * F(x_shared) + poly_G(x_b) * G(x_shared)

    where poly_F and poly_G are linear, we have:
        ∂y/∂x_a = c_F * F(x_shared)
        ∂y/∂x_b = c_G * G(x_shared)

    So the ratio R = (∂y/∂x_a) / (∂y/∂x_b) = (c_F/c_G) * F/G

    We test if this ratio can be fit by simple polynomial forms.

    Parameters
    ----------
    model : torch.nn.Module
        Full model with .forward() and .grad() methods.
    datagen : iterable
        Data generator.
    affine_idx_F, affine_idx_G : int
        Indices of the affine (linear) variables that multiply F and G.
    shared_var_idxs : tuple
        Indices of shared variables in both F and G.
    device, dtype : torch device and dtype
    threshold : float
        Maximum relative RMS for acceptance.
    n_points : int
        Number of points to sample.

    Returns
    -------
    CoupledLeafRatioResult
        Result with ok status, relative RMS, and polynomial form.
    """
    # Collect data
    X_list = []
    for batch in datagen:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        X_list.append(x.to(device=device, dtype=dtype))
        if sum(t.size(0) for t in X_list) >= n_points:
            break

    if not X_list:
        return CoupledLeafRatioResult(
            ok=False, ratio_rel_rms=float("inf"), poly_form="", poly_coeffs={}
        )

    X = torch.cat(X_list, dim=0)[:n_points]

    # Compute gradients
    with torch.no_grad():
        grad = model.grad(X)

    # Handle output dimension
    if grad.ndim == 3:
        grad = grad[:, 0, :]

    # Extract derivatives
    dydx_F = grad[:, affine_idx_F]
    dydx_G = grad[:, affine_idx_G]

    # Filter out near-zero denominators
    mask = dydx_G.abs() > 1e-10
    if mask.sum() < 100:
        return CoupledLeafRatioResult(
            ok=False, ratio_rel_rms=float("inf"), poly_form="", poly_coeffs={}
        )

    R = dydx_F[mask] / dydx_G[mask]
    X_masked = X[mask]

    # Extract shared variables
    if len(shared_var_idxs) != 2:
        return CoupledLeafRatioResult(
            ok=False, ratio_rel_rms=float("inf"), poly_form="not_bivariate", poly_coeffs={}
        )

    i, j = shared_var_idxs
    xi = X_masked[:, i]
    xj = X_masked[:, j]

    # Try various simple polynomial forms for the ratio
    # Form 1: R = a (constant)
    # Form 2: R = a * xj / xi^2 (the Lorentz case)
    # Form 3: R = a + b * xj / xi^2
    # Form 4: R = a * xj / xi
    # Form 5: R = a + b * (xj/xi)

    best_rel_rms = float("inf")
    best_form = ""
    best_coeffs = {}

    R_scale = R.abs().median().clamp_min(1e-12)

    # Form 1: constant
    a_const = R.median()
    res_const = (R - a_const).abs()
    rel_rms_const = float((res_const.median() / R_scale).item())
    if rel_rms_const < best_rel_rms:
        best_rel_rms = rel_rms_const
        best_form = "constant"
        best_coeffs = {"a": float(a_const.item())}

    # Form 2: a * xj / xi^2
    xi_safe = xi.abs().clamp_min(1e-12) * xi.sign()
    xi_safe = torch.where(xi_safe == 0, torch.ones_like(xi_safe) * 1e-12, xi_safe)
    feature_xj_xi2 = xj / (xi_safe**2)
    mask_finite = torch.isfinite(feature_xj_xi2)
    if mask_finite.sum() > 100:
        R_f = R[mask_finite]
        feat_f = feature_xj_xi2[mask_finite]
        # Least squares: R = a * feat => a = sum(R*feat) / sum(feat^2)
        a_xj_xi2 = (R_f * feat_f).sum() / (feat_f**2).sum().clamp_min(1e-12)
        res_xj_xi2 = (R_f - a_xj_xi2 * feat_f).abs()
        rel_rms_xj_xi2 = float((res_xj_xi2.median() / R_scale).item())
        if rel_rms_xj_xi2 < best_rel_rms:
            best_rel_rms = rel_rms_xj_xi2
            best_form = f"a * x{j} / x{i}^2"
            best_coeffs = {"a": float(a_xj_xi2.item()), "form": "xj/xi^2"}

    # Form 3: a + b * xj / xi^2
    if mask_finite.sum() > 100:
        R_f = R[mask_finite]
        feat_f = feature_xj_xi2[mask_finite]
        # Least squares with intercept
        ones = torch.ones_like(feat_f)
        A = torch.stack([ones, feat_f], dim=1)
        try:
            sol = torch.linalg.lstsq(A, R_f.unsqueeze(-1)).solution
            a_int, b_int = sol[0, 0], sol[1, 0]
            pred = a_int + b_int * feat_f
            res_int = (R_f - pred).abs()
            rel_rms_int = float((res_int.median() / R_scale).item())
            if rel_rms_int < best_rel_rms:
                best_rel_rms = rel_rms_int
                best_form = f"a + b * x{j} / x{i}^2"
                best_coeffs = {
                    "a": float(a_int.item()),
                    "b": float(b_int.item()),
                    "form": "a+b*xj/xi^2",
                }
        except Exception:
            pass

    # Form 4: a * xj / xi
    feature_xj_xi = xj / xi_safe
    mask_finite2 = torch.isfinite(feature_xj_xi)
    if mask_finite2.sum() > 100:
        R_f = R[mask_finite2]
        feat_f = feature_xj_xi[mask_finite2]
        a_xj_xi = (R_f * feat_f).sum() / (feat_f**2).sum().clamp_min(1e-12)
        res_xj_xi = (R_f - a_xj_xi * feat_f).abs()
        rel_rms_xj_xi = float((res_xj_xi.median() / R_scale).item())
        if rel_rms_xj_xi < best_rel_rms:
            best_rel_rms = rel_rms_xj_xi
            best_form = f"a * x{j} / x{i}"
            best_coeffs = {"a": float(a_xj_xi.item()), "form": "xj/xi"}

    # Form 5: a + b * (xj/xi)
    if mask_finite2.sum() > 100:
        R_f = R[mask_finite2]
        feat_f = feature_xj_xi[mask_finite2]
        ones = torch.ones_like(feat_f)
        A = torch.stack([ones, feat_f], dim=1)
        try:
            sol = torch.linalg.lstsq(A, R_f.unsqueeze(-1)).solution
            a_lin, b_lin = sol[0, 0], sol[1, 0]
            pred = a_lin + b_lin * feat_f
            res_lin = (R_f - pred).abs()
            rel_rms_lin = float((res_lin.median() / R_scale).item())
            if rel_rms_lin < best_rel_rms:
                best_rel_rms = rel_rms_lin
                best_form = f"a + b * x{j} / x{i}"
                best_coeffs = {
                    "a": float(a_lin.item()),
                    "b": float(b_lin.item()),
                    "form": "a+b*xj/xi",
                }
        except Exception:
            pass

    ok = best_rel_rms < threshold

    return CoupledLeafRatioResult(
        ok=ok,
        ratio_rel_rms=best_rel_rms,
        poly_form=best_form,
        poly_coeffs=best_coeffs,
    )


def check_inv_sqrt_ratio_fit(
    model,
    datagen,
    var_idxs: tuple,
    device,
    dtype,
    threshold: float = 0.02,
    n_points: int = 2048,
):
    """
    Check if a bivariate leaf G(xi, xj) fits the form G = 1/sqrt(1 - (xj/xi)^2).

    This is done by fitting G^(-2) ≈ 1 - (xj/xi)^2, which is a simple polynomial
    in r = xj/xi, avoiding sqrt domain issues.

    Parameters
    ----------
    model : torch.nn.Module
        Model (typically _SubtreeModel) for the leaf.
    datagen : iterable
        Data generator.
    var_idxs : tuple
        (xi_idx, xj_idx) variable indices.
    device, dtype : torch device and dtype
    threshold : float
        Maximum relative RMS for acceptance.
    n_points : int
        Number of points to sample.

    Returns
    -------
    dict
        {'ok': bool, 'rel_rms': float, 'form': str, 'coeffs': dict}
    """
    # Collect data
    X_list = []
    for batch in datagen:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        X_list.append(x.to(device=device, dtype=dtype))
        if sum(t.size(0) for t in X_list) >= n_points:
            break

    if not X_list:
        return {"ok": False, "rel_rms": float("inf"), "form": "", "coeffs": {}}

    X = torch.cat(X_list, dim=0)[:n_points]

    # Evaluate model
    with torch.no_grad():
        G = model.forward(X)

    if G.ndim == 2:
        G = G[:, 0]

    # Compute G^(-2)
    G_safe = G.abs().clamp_min(1e-12)
    G_inv_sq = 1.0 / (G_safe**2)

    # Extract variables
    i, j = var_idxs
    xi = X[:, i]
    xj = X[:, j]

    # Compute ratio r = xj / xi
    xi_safe = xi.abs().clamp_min(1e-12) * xi.sign()
    xi_safe = torch.where(xi_safe == 0, torch.ones_like(xi_safe) * 1e-12, xi_safe)
    r = xj / xi_safe
    r_sq = r**2

    # Filter valid points
    mask = torch.isfinite(r_sq) & torch.isfinite(G_inv_sq) & (G.abs() > 1e-10)
    if mask.sum() < 100:
        return {"ok": False, "rel_rms": float("inf"), "form": "", "coeffs": {}}

    G_inv_sq_m = G_inv_sq[mask]
    r_sq_m = r_sq[mask]

    # Fit G^(-2) = a + b * r^2
    # For Lorentz factor: a = 1, b = -1
    ones = torch.ones_like(r_sq_m)
    A = torch.stack([ones, r_sq_m], dim=1)
    try:
        sol = torch.linalg.lstsq(A, G_inv_sq_m.unsqueeze(-1)).solution
        a, b = sol[0, 0], sol[1, 0]
    except Exception:
        return {"ok": False, "rel_rms": float("inf"), "form": "", "coeffs": {}}

    pred = a + b * r_sq_m
    res = (G_inv_sq_m - pred).abs()
    scale = G_inv_sq_m.abs().median().clamp_min(1e-12)
    rel_rms = float((res.median() / scale).item())

    ok = rel_rms < threshold

    # Check if it's close to the Lorentz form (a≈1, b≈-1)
    is_lorentz_like = abs(float(a.item()) - 1.0) < 0.1 and abs(float(b.item()) + 1.0) < 0.1

    form = "1/sqrt(a + b*r^2)"
    if is_lorentz_like:
        form = "1/sqrt(1 - (xj/xi)^2)"

    return {
        "ok": ok,
        "rel_rms": rel_rms,
        "form": form,
        "coeffs": {"a": float(a.item()), "b": float(b.item())},
        "is_lorentz_like": is_lorentz_like,
    }


def conv_expression(expressions):
    expression = parse_prefix(expressions, 0)[0]
    expression = simplify_expression_string(expression)
    expression = re.sub(
        r"\[(\d+(?:,\s*\d+)*)\]", lambda m: "[x" + m.group(1).replace(", ", ", x") + "]", expression
    )
    return expression


# ──────────────────────────────────────────────────────────────
# Monomial Compound Variable Detection
# ──────────────────────────────────────────────────────────────


def check_monomial_compound(
    var_idxs,
    x_vals,
    dydx_vals,
    max_exponent: int = 2,
    precision: float = 0.05,
    min_grad_weight: float = 0.1,
    min_samples_fraction: float = 0.5,
):
    """
    Detect if a function depends on a monomial compound variable z = ∏xᵢ^aᵢ.

    Uses the rank-1 test in log-coordinates: if f(x_S) = g(z) with z = ∏xᵢ^aᵢ,
    then uᵢ ≡ xᵢ · ∂f/∂xᵢ = g'(z) · aᵢ · z is collinear with the exponent vector a.

    Parameters
    ----------
    var_idxs : tuple of int
        Indices of the variables to test for compound structure.
    x_vals : array-like, shape (N, Nx)
        Input data points.
    dydx_vals : array-like, shape (N, Nx) or (N, 1, Nx)
        Gradients ∂f/∂xᵢ at each data point.
    max_exponent : int
        Maximum absolute value of exponents to consider (e.g., 2 means -2 to +2).
    precision : float
        Threshold for σ₂/σ₁ ratio to declare rank-1 structure.
    min_grad_weight : float
        Minimum gradient magnitude (relative to median) to include a sample.
    min_samples_fraction : float
        Minimum fraction of samples that must pass filtering.

    Returns
    -------
    tuple (proposals, sigma_ratio)
        proposals : list of (exponents, confidence)
            List of detected compound structures, sorted by confidence (highest first).
            exponents is a tuple of integers, e.g., (1, -1) for x0/x1.
            confidence is 1 - σ₂/σ₁ (higher is better).
        sigma_ratio : float or None
            The σ₂/σ₁ ratio from SVD (lower is better), or None if SVD failed.
    """
    import numpy as np

    # Convert to numpy if needed
    if hasattr(x_vals, "detach"):
        x_vals = x_vals.detach().cpu().numpy()
    if hasattr(dydx_vals, "detach"):
        dydx_vals = dydx_vals.detach().cpu().numpy()

    x_vals = np.asarray(x_vals)
    dydx_vals = np.asarray(dydx_vals)

    # Handle shape (N, 1, Nx) -> (N, Nx)
    if dydx_vals.ndim == 3 and dydx_vals.shape[1] == 1:
        dydx_vals = dydx_vals[:, 0, :]

    var_idxs = tuple(var_idxs)
    n_vars = len(var_idxs)
    if n_vars < 2:
        return [], None

    N = x_vals.shape[0]

    # Extract relevant columns
    x_sub = x_vals[:, list(var_idxs)]  # (N, n_vars)
    dydx_sub = dydx_vals[:, list(var_idxs)]  # (N, n_vars)

    # Form weighted gradient matrix: U[n, i] = x[n, i] * ∂f/∂x_i
    U = x_sub * dydx_sub  # (N, n_vars)

    # Filter samples:
    # 1. All x values must be safely away from 0 (for potential negative exponents)
    # 2. Gradient magnitude should be significant
    x_min_abs = np.abs(x_sub).min(axis=1)
    grad_mag = np.linalg.norm(dydx_sub, axis=1)
    grad_median = np.median(grad_mag)

    # Safe threshold for x values (avoid division by very small numbers)
    x_safe_threshold = 1e-6

    # Weight by gradient magnitude (samples with larger gradients are more informative)
    weights = grad_mag / (grad_median + 1e-12)
    valid_mask = (x_min_abs > x_safe_threshold) & (weights > min_grad_weight)

    n_valid = np.sum(valid_mask)
    if n_valid < min_samples_fraction * N or n_valid < n_vars + 1:
        return [], None

    U_valid = U[valid_mask]
    weights_valid = weights[valid_mask]

    # Weighted SVD: weight rows by sqrt(weights)
    sqrt_w = np.sqrt(weights_valid)[:, None]

    # Center the data (subtract weighted mean)
    weighted_mean = np.average(U_valid, axis=0, weights=weights_valid)
    U_centered = (U_valid - weighted_mean) * sqrt_w

    # SVD to find dominant direction
    try:
        _, s, Vt = np.linalg.svd(U_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return [], None

    if len(s) < 2:
        return [], None

    # Check rank-1 score
    sigma_ratio = s[1] / (s[0] + 1e-12)

    # Return sigma_ratio even if threshold not met (caller uses best-of-subsets)
    if sigma_ratio > precision:
        return [], sigma_ratio

    confidence = 1.0 - sigma_ratio

    # Get the dominant direction (exponent vector estimate)
    a_hat = Vt[0]  # First right singular vector

    # Normalize so largest absolute component is 1
    max_abs = np.max(np.abs(a_hat))
    if max_abs < 1e-12:
        return [], sigma_ratio
    a_hat = a_hat / max_abs

    # Try to integerize to small integers
    results = []

    # Try different normalizations (by each component)
    for norm_idx in range(n_vars):
        if np.abs(a_hat[norm_idx]) < 0.1:
            continue

        a_scaled = a_hat / a_hat[norm_idx]

        # Round to integers in [-max_exponent, max_exponent]
        a_int = np.round(a_scaled).astype(int)
        a_int = np.clip(a_int, -max_exponent, max_exponent)

        # Skip if all zeros or all same sign with magnitude 1 (trivial)
        if np.all(a_int == 0):
            continue

        # Check how well the integer approximation matches
        a_int_normalized = a_int / (np.linalg.norm(a_int) + 1e-12)
        a_hat_normalized = a_hat / (np.linalg.norm(a_hat) + 1e-12)
        match_score = np.abs(np.dot(a_int_normalized, a_hat_normalized))

        if match_score > 0.95:  # Good integer match
            exponents = tuple(int(e) for e in a_int)

            # Verify this exponent combination makes sense
            # (at least one positive and one negative for ratio, or multiple positives for product)
            if _is_valid_exponent_pattern(exponents):
                # Check if we already have this pattern (up to sign flip)
                is_duplicate = False
                for prev_exp, _ in results:
                    if prev_exp == exponents or prev_exp == tuple(-e for e in exponents):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    results.append((exponents, confidence * match_score))

    # Sort by confidence (highest first)
    results.sort(key=lambda x: -x[1])

    return results, sigma_ratio


def check_linear_compound(
    var_idxs,
    dydx_vals,
    max_coeff: int = 2,
    precision: float = 0.05,
    min_grad_weight: float = 0.1,
    min_samples_fraction: float = 0.5,
):
    r"""Detect if a function depends on a *linear* compound variable.

    We test for the existence of a scalar z = \sum_i a_i x_i such that

        f(x_S) = g(z)

    In that case the gradient vectors are collinear:

        ∂f/∂x_i = g'(z) a_i

    so the sample-by-variable gradient matrix has (approximately) rank 1.

    This is the additive analogue of :func:`check_monomial_compound` and is
    especially effective for discovering coupled arguments like:

      - z = x_i - x_j
      - z = x_i + x_j
      - z = x0 + x1 - x2

    Parameters
    ----------
    var_idxs : tuple[int]
        Variable indices (columns) to consider.
    dydx_vals : array-like, shape (N, Nx) or (N, 1, Nx)
        Gradients ∂f/∂xᵢ at each sample.
    max_coeff : int
        Maximum absolute integer coefficient to consider.
    precision : float
        Threshold for σ₂/σ₁ to declare rank-1 structure.
    min_grad_weight : float
        Minimum gradient magnitude (relative to median) to include a sample.
    min_samples_fraction : float
        Minimum fraction of samples that must pass filtering.

    Returns
    -------
    tuple (results, sigma_ratio)
        results : list[(coeffs, confidence)]
            coeffs is a tuple of ints (same length as var_idxs).
            confidence is a heuristic score in [0,1].
        sigma_ratio : float or None
            σ₂/σ₁ from SVD (lower is better), or None if SVD failed.
    """
    import numpy as np

    if hasattr(dydx_vals, "detach"):
        dydx_vals = dydx_vals.detach().cpu().numpy()
    dydx_vals = np.asarray(dydx_vals)

    if dydx_vals.ndim == 3 and dydx_vals.shape[1] == 1:
        dydx_vals = dydx_vals[:, 0, :]

    var_idxs = tuple(var_idxs)
    n_vars = len(var_idxs)
    if n_vars < 2:
        return [], None

    # Extract relevant columns
    dydx_sub = dydx_vals[:, list(var_idxs)]

    # Weight by gradient magnitude
    grad_mag = np.linalg.norm(dydx_sub, axis=1)
    grad_median = np.median(grad_mag)
    weights = grad_mag / (grad_median + 1e-12)

    finite_mask = np.isfinite(weights) & np.isfinite(dydx_sub).all(axis=1)
    valid_mask = finite_mask & (weights > float(min_grad_weight))

    N = int(dydx_sub.shape[0])
    n_valid = int(np.sum(valid_mask))
    if n_valid < min_samples_fraction * N or n_valid < n_vars + 1:
        return [], None

    G_valid = dydx_sub[valid_mask]
    w_valid = weights[valid_mask]

    sqrt_w = np.sqrt(w_valid)[:, None]
    weighted_mean = np.average(G_valid, axis=0, weights=w_valid)
    G_centered = (G_valid - weighted_mean) * sqrt_w

    try:
        _, s, Vt = np.linalg.svd(G_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return [], None

    if len(s) < 2:
        return [], None

    sigma_ratio = float(s[1] / (s[0] + 1e-12))
    if sigma_ratio > float(precision):
        return [], sigma_ratio

    confidence_base = 1.0 - sigma_ratio
    a_hat = Vt[0]
    max_abs = float(np.max(np.abs(a_hat)))
    if not np.isfinite(max_abs) or max_abs < 1e-12:
        return [], sigma_ratio
    a_hat = a_hat / max_abs

    results = []
    for norm_idx in range(n_vars):
        if abs(float(a_hat[norm_idx])) < 0.1:
            continue

        a_scaled = a_hat / a_hat[norm_idx]
        a_int = np.round(a_scaled).astype(int)
        a_int = np.clip(a_int, -int(max_coeff), int(max_coeff))

        if np.all(a_int == 0):
            continue

        # Require at least two non-zero coefficients to be a non-trivial compound
        if int(np.sum(a_int != 0)) < 2:
            continue

        a_int_norm = a_int / (np.linalg.norm(a_int) + 1e-12)
        a_hat_norm = a_hat / (np.linalg.norm(a_hat) + 1e-12)
        match_score = float(np.abs(np.dot(a_int_norm, a_hat_norm)))
        if match_score <= 0.95:
            continue

        coeffs = tuple(int(e) for e in a_int)

        # Deduplicate up to global sign flips (z and -z are equivalent for g)
        is_dup = False
        for prev, _ in results:
            if prev == coeffs or prev == tuple(-c for c in coeffs):
                is_dup = True
                break
        if is_dup:
            continue

        results.append((coeffs, confidence_base * match_score))

    results.sort(key=lambda t: -t[1])
    return results, sigma_ratio


def _stable_weighted_orthogonal_mean(values, direction, weights):
    """Weighted mean projected off ``direction`` without overflow.

    Projection and averaging are both linear, so project the scale-normalized
    weighted mean instead of every potentially huge row.  Non-finite or
    unrepresentable inputs fail closed with ``None``.
    """

    import numpy as np

    matrix = np.asarray(values, dtype=float)
    axis = np.asarray(direction, dtype=float).reshape(-1)
    sample_weights = np.asarray(weights, dtype=float).reshape(-1)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != sample_weights.size
        or matrix.shape[1] != axis.size
        or matrix.size == 0
        or axis.size == 0
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(axis))
        or not np.all(np.isfinite(sample_weights))
        or np.any(sample_weights < 0.0)
    ):
        return None

    weight_scale = float(np.max(sample_weights))
    if weight_scale <= 0.0:
        return None
    denominator = float(np.dot(axis, axis) + 1.0e-12)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        coefficients = (matrix @ axis) / denominator
        residuals = matrix - coefficients[:, None] * axis[None, :]
        direct = np.average(residuals, axis=0, weights=sample_weights)
    if np.all(np.isfinite(direct)):
        return direct

    weights_scaled = sample_weights / weight_scale
    weight_sum = float(np.sum(weights_scaled))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return None

    value_scale = float(np.max(np.abs(matrix)))
    if not np.isfinite(value_scale):
        return None
    if value_scale <= 0.0:
        return np.zeros(matrix.shape[1], dtype=float)

    matrix_scaled = matrix / value_scale
    mean_scaled = np.sum(matrix_scaled * weights_scaled[:, None], axis=0) / weight_sum
    coefficient = float(np.dot(mean_scaled, axis) / denominator)
    projected_scaled = mean_scaled - coefficient * axis
    with np.errstate(over="ignore", invalid="ignore"):
        projected = projected_scaled * value_scale
    if not np.all(np.isfinite(projected)):
        return None
    return projected



def check_monomial_compound_logderiv(
    var_idxs,
    x_vals,
    y_vals,
    dydx_vals,
    max_exponent: int = 2,
    precision: float = 0.05,
    min_grad_weight: float = 0.1,
    min_samples_fraction: float = 0.5,
    min_y_abs_weight: float = 1e-3,
    verbose: bool = False,
):
    """
    Detect monomial compound structure z = ∏xᵢ^aᵢ in the *presence of a monomial prefactor*.

    Many AIF-style targets are of the form:

        f(x) = m(x) * g(z),     z = ∏ xᵢ^aᵢ,     m(x) = ∏ xᵢ^bᵢ

    where some variables can appear in BOTH m and z. In that case the classic
    rank-1 test on uᵢ = xᵢ ∂f/∂xᵢ generally becomes rank-2 and may fail.

    This variant works in *log-derivative space*:

        vᵢ = (xᵢ ∂f/∂xᵢ) / f = ∂ ln f / ∂ ln xᵢ

    For f = m(x) g(z), one has v = b + s(z) a where s(z) is a scalar function.
    Therefore the centered matrix (v - mean(v)) is rank-1 with dominant direction a,
    regardless of overlaps between m and z (as long as f doesn't cross 0 too often).

    Parameters
    ----------
    var_idxs : tuple[int]
        Indices of variables to consider (columns in x_vals / dydx_vals).
    x_vals : array-like, shape (N, Nx)
    y_vals : array-like, shape (N,) or (N,1)
        Function values f(x) at the same samples.
    dydx_vals : array-like, shape (N, Nx) or (N, 1, Nx)
    max_exponent : int
        Max |aᵢ| to consider when integerising the SVD direction.
    precision : float
        Threshold for σ₂/σ₁ in the centered log-derivative matrix.
    min_grad_weight : float
        Minimum gradient magnitude (relative to median) to include a sample.
    min_samples_fraction : float
        Minimum fraction of samples that must pass filtering.
    min_y_abs_weight : float
        Minimum |f| relative to median(|f|) to include a sample (avoid division blow-ups).

    Returns
    -------
    tuple (results, sigma_ratio, b_perp)
        results : list[(exponents, confidence)]
            Candidate exponent patterns for z, sorted by confidence.
        sigma_ratio : float or None
            σ₂/σ₁ from SVD, or None if SVD failed.
        b_perp : np.ndarray | None
            Estimate of the constant log-derivative offset orthogonal to a:
                b_perp = b - proj_a(b)
            Useful as a heuristic to decide which raw variables also act as outer monomial
            factors and should be kept as extra inputs alongside z.
    """
    import numpy as np

    # Convert to numpy if needed
    if hasattr(x_vals, "detach"):
        x_vals = x_vals.detach().cpu().numpy()
    if hasattr(y_vals, "detach"):
        y_vals = y_vals.detach().cpu().numpy()
    if hasattr(dydx_vals, "detach"):
        dydx_vals = dydx_vals.detach().cpu().numpy()

    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals).reshape(-1)
    dydx_vals = np.asarray(dydx_vals)

    # Handle shape (N, 1, Nx) -> (N, Nx)
    if dydx_vals.ndim == 3 and dydx_vals.shape[1] == 1:
        dydx_vals = dydx_vals[:, 0, :]

    var_idxs = tuple(var_idxs)
    n_vars = len(var_idxs)
    if n_vars < 2:
        return [], None, None

    N = x_vals.shape[0]
    if y_vals.shape[0] != N:
        # best-effort alignment
        N = min(N, y_vals.shape[0])
        x_vals = x_vals[:N]
        y_vals = y_vals[:N]
        dydx_vals = dydx_vals[:N]

    # Extract relevant columns
    x_sub = x_vals[:, list(var_idxs)]        # (N, n_vars)
    dydx_sub = dydx_vals[:, list(var_idxs)]  # (N, n_vars)

    # uᵢ = xᵢ ∂f/∂xᵢ
    U = x_sub * dydx_sub  # (N, n_vars)

    # Filter samples:
    #  1) avoid x ~ 0 (for negative exponents)
    #  2) require significant gradient magnitude
    #  3) avoid f ~ 0 (division instability)
    x_min_abs = np.abs(x_sub).min(axis=1)
    grad_mag = np.linalg.norm(dydx_sub, axis=1)
    grad_median = np.median(grad_mag)

    x_safe_threshold = 1e-6
    weights = grad_mag / (grad_median + 1e-12)

    y_abs = np.abs(y_vals)
    y_abs_median = np.median(y_abs)
    y_safe = max(1e-12, float(min_y_abs_weight) * float(y_abs_median + 1e-12))

    finite_mask = np.isfinite(y_vals) & np.isfinite(weights) & np.isfinite(x_min_abs)
    valid_mask = finite_mask & (x_min_abs > x_safe_threshold) & (weights > min_grad_weight) & (y_abs > y_safe)

    n_valid = int(np.sum(valid_mask))
    if n_valid < min_samples_fraction * N or n_valid < n_vars + 1:
        if verbose:
            print(
                f"[Compound LogDeriv] Rejected: n_valid={n_valid}/{N} "
                f"(need {min_samples_fraction*N:.0f} or {n_vars+1}), vars={var_idxs}"
            )
        return [], None, None

    U_valid = U[valid_mask]
    y_valid = y_vals[valid_mask]
    weights_valid = weights[valid_mask]

    # vᵢ = uᵢ / f
    V = U_valid / y_valid[:, None]  # (n_valid, n_vars)

    # Filter out rows with non-finite log-derivatives (can occur near singularities)
    finite_rows = np.isfinite(V).all(axis=1)
    n_dropped = int(np.sum(~finite_rows))
    if n_dropped > 0:
        if verbose:
            print(f"[Compound LogDeriv] Dropped {n_dropped}/{len(V)} rows with non-finite log-derivatives")
        V = V[finite_rows]
        weights_valid = weights_valid[finite_rows]
        n_valid = len(V)
        if n_valid < n_vars + 1:
            if verbose:
                print(f"[Compound LogDeriv] Rejected: insufficient finite rows ({n_valid}), vars={var_idxs}")
            return [], None, None

    # Weighted SVD on centered V
    sqrt_w = np.sqrt(weights_valid)[:, None]
    weighted_mean = np.average(V, axis=0, weights=weights_valid)
    V_centered = (V - weighted_mean) * sqrt_w

    try:
        _, s, Vt = np.linalg.svd(V_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return [], None, None

    if len(s) < 2:
        return [], None, None

    sigma_ratio = float(s[1] / (s[0] + 1e-12))
    if verbose:
        print(
            f"[Compound LogDeriv] n_valid={n_valid}/{N}, sigma_ratio={sigma_ratio:.4g}, "
            f"threshold={precision}, vars={var_idxs}"
        )

    if sigma_ratio > precision:
        return [], sigma_ratio, None

    confidence = 1.0 - sigma_ratio

    a_hat = Vt[0]
    max_abs = np.max(np.abs(a_hat))
    if max_abs < 1e-12:
        return [], sigma_ratio, None
    a_hat = a_hat / max_abs
    if verbose:
        print(f"[Compound LogDeriv] Dominant direction a_hat={a_hat}, vars={var_idxs}")

    results = []
    for norm_idx in range(n_vars):
        if np.abs(a_hat[norm_idx]) < 0.1:
            continue

        a_scaled = a_hat / a_hat[norm_idx]
        a_int = np.round(a_scaled).astype(int)
        a_int = np.clip(a_int, -max_exponent, max_exponent)

        if np.all(a_int == 0):
            continue

        a_int_normalized = a_int / (np.linalg.norm(a_int) + 1e-12)
        a_hat_normalized = a_hat / (np.linalg.norm(a_hat) + 1e-12)
        match_score = float(np.abs(np.dot(a_int_normalized, a_hat_normalized)))

        if verbose:
            print(
                f"[Compound LogDeriv] norm_idx={norm_idx}: a_int={tuple(a_int)}, "
                f"match_score={match_score:.4f}"
            )

        if match_score > 0.95:
            exponents = tuple(int(e) for e in a_int)
            if _is_valid_exponent_pattern(exponents):
                is_duplicate = False
                for prev_exp, _ in results:
                    if prev_exp == exponents or prev_exp == tuple(-e for e in exponents):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    results.append((exponents, confidence * match_score))

    results.sort(key=lambda x: -x[1])

    # b_perp estimate (constant log-derivative offset orthogonal to the dominant a-direction)
    b_perp = None
    if results:
        a_vec = np.asarray(results[0][0], dtype=float)
        b_try = _stable_weighted_orthogonal_mean(V, a_vec, weights_valid)
        if b_try is not None:
            b_perp = b_try
        else:
            # keep the diagnostic; just don't export poison
            if verbose:
                print("[Compound LogDeriv] b_perp non-finite; dropping prefactor estimate")
            b_perp = None

    return results, sigma_ratio, b_perp



def _is_valid_exponent_pattern(exponents):
    """Check if exponent pattern represents a meaningful compound variable."""
    # Filter out trivial cases
    non_zero = [e for e in exponents if e != 0]
    if len(non_zero) < 2:
        return False

    # All same value with magnitude 1 is a product/ratio which IS meaningful
    # e.g., (1, 1) for x0*x1, (1, -1) for x0/x1
    # Only reject if all exponents are identical AND there's just one variable
    # (which is already handled by len(non_zero) < 2)

    return True


def build_monomial_ast(var_idxs, exponents):
    """
    Build an AST for the monomial z = ∏ Var(i)^exp_i.

    Parameters
    ----------
    var_idxs : tuple of int
        Variable indices.
    exponents : tuple of int or float
        Exponents for each variable.

    Returns
    -------
    Node
        AST representing the monomial product/ratio.

    Examples
    --------
    >>> build_monomial_ast((0, 1), (1, 1))   # x0 * x1
    >>> build_monomial_ast((0, 1), (1, -1))  # x0 / x1 = x0 * x1^(-1)
    """
    from .bridges import Mul, Pow, Var

    if len(var_idxs) != len(exponents):
        raise ValueError("var_idxs and exponents must have same length")

    terms = []
    for idx, exp in zip(var_idxs, exponents):
        if exp == 0:
            continue
        elif exp == 1:
            terms.append(Var(idx))
        else:
            # Use the exponent directly (int or float) - avoid Fraction for simpler formatting
            terms.append(Pow(Var(idx), int(exp) if float(exp).is_integer() else exp))

    if len(terms) == 0:
        raise ValueError("All exponents are zero - no valid monomial")

    result = terms[0]
    for t in terms[1:]:
        result = Mul(result, t)

    return result


def build_linear_ast(var_idxs, coeffs):
    r"""Build an AST for the linear form z = \sum_i c_i * Var(var_i).

    Parameters
    ----------
    var_idxs : tuple[int]
        Global variable indices.
    coeffs : tuple[int|float]
        Coefficients for each variable index (same length as var_idxs).

    Returns
    -------
    Node
        AST representing the linear form.
    """
    from .bridges import Add, ConstNode, Mul, Var

    if len(var_idxs) != len(coeffs):
        raise ValueError("var_idxs and coeffs must have same length")

    terms = []
    for idx, c in zip(var_idxs, coeffs):
        if c == 0:
            continue
        if c == 1:
            terms.append(Var(int(idx)))
        elif c == -1:
            terms.append(Mul(ConstNode(-1.0), Var(int(idx))))
        else:
            terms.append(Mul(ConstNode(float(c)), Var(int(idx))))

    if not terms:
        raise ValueError("All coefficients are zero - no valid linear form")

    expr = terms[0]
    for t in terms[1:]:
        expr = Add(expr, t)
    return expr


def build_radial_r2_ast(var_idxs):
    r"""Build an AST for r^2 = \sum_i Var(i)^2 over the given indices."""
    from .bridges import Add, Pow, Var

    idxs = list(int(i) for i in (var_idxs or ()))
    if len(idxs) < 2:
        raise ValueError("Need at least two variables for a radial compound")

    terms = [Pow(Var(i), 2) for i in idxs]
    expr = terms[0]
    for t in terms[1:]:
        expr = Add(expr, t)
    return expr


def build_radial_r_ast(var_idxs):
    r"""Build an AST for r = sqrt(\sum_i Var(i)^2)."""
    from .bridges import Pow

    r2 = build_radial_r2_ast(var_idxs)
    return Pow(r2, 0.5)


def build_difference_product_ast(i, j, k):
    """
    Build AST for z = (x_i - x_j) * x_k.

    Parameters
    ----------
    i, j, k : int
        Global variable indices.

    Returns
    -------
    Node
        AST node representing (Var(i) - Var(j)) * Var(k).

    Examples
    --------
    >>> build_difference_product_ast(4, 5, 2)  # (x4 - x5) * x2
    """
    from .bridges import Add, ConstNode, Mul, Var

    # Build x_i - x_j = x_i + (-1)*x_j
    neg_one = ConstNode(-1.0)
    diff = Add(Var(i), Mul(neg_one, Var(j)))

    # Multiply by x_k
    return Mul(diff, Var(k))


def build_difference_product_power_ast(i, j, k, p: int = 1):
    """
    Build AST for z = (x_i - x_j) * x_k^p.

    This generalises :func:`build_difference_product_ast` to allow the multiplier
    variable to appear with a small integer power (including negative powers,
    i.e. division).

    Parameters
    ----------
    i, j, k : int
        Global variable indices.
    p : int
        Power on x_k (e.g. -1 for division, 2 for square).

    Returns
    -------
    Node
        AST node representing (Var(i) - Var(j)) * Pow(Var(k), p).

    Examples
    --------
    >>> build_difference_product_power_ast(1, 2, 0, -1)  # (x1 - x2) / x0
    """
    from .bridges import Add, ConstNode, Mul, Pow, Var

    # Build x_i - x_j = x_i + (-1)*x_j
    neg_one = ConstNode(-1.0)
    diff = Add(Var(i), Mul(neg_one, Var(j)))

    # Multiply by x_k^p
    try:
        pp = int(p)
    except Exception:
        pp = 1
    if pp == 1:
        return Mul(diff, Var(k))
    return Mul(diff, Pow(Var(k), float(pp)))


def build_power_difference_ast(i, j, n):
    """Build AST for z = x_i^n - x_j^n.  n=1 gives the linear difference.

    Parameters
    ----------
    i, j : int
        Global variable indices.
    n : int
        Power on each variable (must be positive integer).

    Returns
    -------
    Node
        AST node representing Var(i)^n - Var(j)^n.

    Examples
    --------
    >>> build_power_difference_ast(1, 3, 2)  # x1² - x3²
    """
    from .bridges import Add, ConstNode, Mul, Pow, Var

    n = int(n)
    if n == 1:
        return Add(Var(i), Mul(ConstNode(-1.0), Var(j)))
    return Add(Pow(Var(i), float(n)), Mul(ConstNode(-1.0), Pow(Var(j), float(n))))


def build_power_difference_product_ast(i, j, n, k, p=1):
    """Build AST for z = (x_i^n - x_j^n) * x_k^p.

    Parameters
    ----------
    i, j : int
        Global variable indices for the power-difference pair.
    n : int
        Power on each variable in the difference.
    k : int
        Global variable index for the multiplier.
    p : int
        Power on x_k (default 1).

    Returns
    -------
    Node
        AST node representing (Var(i)^n - Var(j)^n) * Var(k)^p.

    Examples
    --------
    >>> build_power_difference_product_ast(1, 3, 2, 0)  # (x1² - x3²) * x0
    """
    from .bridges import Mul, Pow, Var

    diff = build_power_difference_ast(i, j, n)
    p = int(p)
    if p == 1:
        return Mul(diff, Var(k))
    return Mul(diff, Pow(Var(k), float(p)))


# ──────────────────────────────────────────────────────────────
# Mixed Compound Variable Detection (monomial * trig product)
# ──────────────────────────────────────────────────────────────


def _phase_to_trig_kind(phase: float, tol: float = 0.25 * math.pi):
    """
    Determine trig function (cos/sin) from FFT phase.

    Parameters
    ----------
    phase : float
        Phase angle in radians from FFT coefficient (complex argument).
    tol : float
        Tolerance for detecting pure cos (phase~0) or sin (phase~±π/2).

    Returns
    -------
    tuple (kind, adjusted_phase)
        kind : str
            "cos" or "sin"
        adjusted_phase : float
            Residual phase after selecting cos/sin (0.0 for pure cos/sin).
    """
    # Normalize phase to [-π, π]
    phase = math.atan2(math.sin(phase), math.cos(phase))

    # Check for pure cosine (phase ~ 0 or ±π)
    if abs(phase) < tol:
        return "cos", 0.0
    if abs(abs(phase) - math.pi) < tol:
        return "cos", 0.0  # cos(x + π) = -cos(x), absorbed by amplitude

    # Check for pure sine (phase ~ ±π/2)
    if abs(phase - math.pi / 2) < tol:
        return "sin", 0.0
    if abs(phase + math.pi / 2) < tol:
        return "sin", 0.0  # sin(x - π/2) = -cos(x), but we prefer sin

    # Non-standard phase: use cos with explicit phase offset
    return "cos", phase


def _extract_phase_from_fft(signal, dt: float, omega: float):
    """
    Extract the phase of a signal at a given frequency using FFT.

    Parameters
    ----------
    signal : array-like
        1D signal sampled at uniform intervals.
    dt : float
        Sampling interval.
    omega : float
        Target angular frequency.

    Returns
    -------
    float
        Phase angle in radians.
    """
    import numpy as np

    signal = np.asarray(signal)
    n = len(signal)
    if n < 4:
        return 0.0

    # Center the signal
    signal_centered = signal - np.mean(signal)

    # FFT
    fft_result = np.fft.rfft(signal_centered)
    freqs = np.fft.rfftfreq(n, dt) * 2 * np.pi  # angular frequencies

    # Find closest frequency bin
    k = np.argmin(np.abs(freqs - omega))
    if k == 0:
        k = 1  # Skip DC

    # Extract phase from complex FFT coefficient
    # FFT gives: A * exp(i*phi) for cos(omega*t + phi)
    phase = np.angle(fft_result[k])

    return float(phase)


def check_mixed_compound(
    var_idxs,
    x_vals,
    dydx_vals,
    trig_axis_specs,
    max_exponent: int = 2,
    precision: float = 0.05,
    min_overall_confidence: float = 0.5,
):
    """
    Detect mixed compound variables of the form z = monomial * trig_product.

    For functions like f(x) = g(x0 * x1 * cos(x2)), this partitions variables:
    - Linear (monomial) variables: show rank-1 structure in weighted gradients
    - Trig variables: show periodic FFT signature on axis scans

    Parameters
    ----------
    var_idxs : tuple of int
        Global variable indices to consider.
    x_vals : array-like, shape (N, Nx)
        Input data points.
    dydx_vals : array-like, shape (N, Nx) or (N, 1, Nx)
        Gradients ∂f/∂xᵢ at each data point.
    trig_axis_specs : dict[int, TrigAxisSpec] or list[TrigAxisSpec]
        Trig axis information from `discover_trig_axes`. Maps axis index to spec.
    max_exponent : int
        Maximum absolute value of monomial exponents.
    precision : float
        Threshold for σ₂/σ₁ ratio in monomial detection.
    min_overall_confidence : float
        Minimum overall confidence to return a proposal.

    Returns
    -------
    list of MixedCompoundProposal
        Proposals for mixed compound variables, sorted by confidence.
    """
    import numpy as np

    # Import here to avoid circular dependency
    from nestynet_sr.sr_search.features import MixedCompoundProposal

    # Convert to numpy
    if hasattr(x_vals, "detach"):
        x_vals = x_vals.detach().cpu().numpy()
    if hasattr(dydx_vals, "detach"):
        dydx_vals = dydx_vals.detach().cpu().numpy()

    x_vals = np.asarray(x_vals)
    dydx_vals = np.asarray(dydx_vals)

    if dydx_vals.ndim == 3 and dydx_vals.shape[1] == 1:
        dydx_vals = dydx_vals[:, 0, :]

    var_idxs = tuple(int(v) for v in var_idxs)
    if len(var_idxs) < 2:
        return []

    # Convert trig_axis_specs to dict if it's a list
    if isinstance(trig_axis_specs, (list, tuple)):
        trig_specs_dict = {spec.axis: spec for spec in trig_axis_specs}
    else:
        trig_specs_dict = dict(trig_axis_specs) if trig_axis_specs else {}

    # 1. Partition variables: identify which are trig vs linear
    trig_var_set = set()
    trig_info = {}  # axis -> (omega, strength)

    for var_idx in var_idxs:
        spec = trig_specs_dict.get(var_idx)
        if spec is not None:  # oracle presence is sufficient
            trig_var_set.add(var_idx)
            trig_info[var_idx] = (spec.omega, spec.strength, spec.tmin, spec.tmax)

    # If no trig axes or all axes are trig, not a "mixed" compound
    linear_vars = [v for v in var_idxs if v not in trig_var_set]
    trig_vars = [v for v in var_idxs if v in trig_var_set]

    if len(trig_vars) == 0:
        # No trig detected - fall back to pure monomial detection
        return []

    if len(linear_vars) == 0:
        # All axes are trig - not a mixed compound (pure trig product)
        return []

    # 2. Run monomial detection on the linear (non-trig) subset only
    x_linear = x_vals[:, linear_vars]
    dydx_linear = dydx_vals[:, linear_vars]

    monomial_proposals = []
    sigma_ratio = None

    if len(linear_vars) >= 2:
        # Use the existing monomial compound detection on linear subset
        mono_results, sigma_ratio = check_monomial_compound(
            var_idxs=tuple(range(len(linear_vars))),  # local indices
            x_vals=x_linear,
            dydx_vals=dydx_linear,
            max_exponent=max_exponent,
            precision=precision,
        )
        monomial_proposals = mono_results
    elif len(linear_vars) == 1:
        # Single linear variable: treat as exponent 1
        monomial_proposals = [((1,), 0.9)]  # high confidence for single var
        sigma_ratio = 0.0

    if not monomial_proposals:
        # Monomial detection failed even on linear subset
        # Try with exponent (1,1,...) as fallback for simple products
        if len(linear_vars) >= 1:
            fallback_exps = tuple(1 for _ in linear_vars)
            monomial_proposals = [(fallback_exps, 0.5)]  # lower confidence
            sigma_ratio = 0.1

    if not monomial_proposals:
        return []

    # 3. For each trig axis: determine cos vs sin from FFT phase
    trig_kinds = []
    trig_phases = []
    trig_omegas = []

    for var_idx in trig_vars:
        omega, strength, tmin, tmax = trig_info[var_idx]
        trig_omegas.append(omega)

        # Extract phase from the data along this axis
        # Use a simple linear scan through the data
        axis_vals = x_vals[:, var_idx]
        grad_vals = dydx_vals[:, var_idx]

        # Sort by axis value for cleaner phase extraction
        sort_idx = np.argsort(axis_vals)
        axis_sorted = axis_vals[sort_idx]
        grad_sorted = grad_vals[sort_idx]

        # Estimate dt from sorted values
        if len(axis_sorted) > 1:
            dt = np.median(np.diff(axis_sorted))
            dt = max(dt, 1e-10)
        else:
            dt = 0.01

        # Extract phase from gradient profile (derivative has same frequency)
        phase = _extract_phase_from_fft(grad_sorted, dt, omega)
        kind, adj_phase = _phase_to_trig_kind(phase)

        trig_kinds.append(kind)
        trig_phases.append(adj_phase)

    # 4. Build proposals
    proposals = []

    for mono_exps, mono_conf in monomial_proposals:
        # Map local exponents back to global variable indices
        global_exps = tuple(int(e) for e in mono_exps)

        # Calculate overall confidence
        trig_conf = 1.0  # oracle-verified
        overall_conf = mono_conf * 0.7 + trig_conf * 0.3

        if overall_conf < min_overall_confidence:
            continue

        # Build the AST for z = monomial * trig_product
        try:
            z_ast = build_mixed_compound_ast(
                linear_var_idxs=tuple(linear_vars),
                linear_exponents=global_exps,
                trig_var_idxs=tuple(trig_vars),
                trig_omegas=tuple(trig_omegas),
                trig_kinds=tuple(trig_kinds),
                trig_phases=tuple(trig_phases),
            )
        except ValueError:
            continue

        proposal = MixedCompoundProposal(
            linear_var_idxs=tuple(linear_vars),
            linear_exponents=global_exps,
            trig_var_idxs=tuple(trig_vars),
            trig_omegas=tuple(trig_omegas),
            trig_kinds=tuple(trig_kinds),
            trig_phases=tuple(trig_phases),
            monomial_sigma_ratio=sigma_ratio if sigma_ratio is not None else 0.0,
            trig_strengths=(100.0,) * len(trig_vars),  # synthetic; oracle-verified
            overall_confidence=overall_conf,
            z_ast=z_ast,
        )
        proposals.append(proposal)

    # Sort by confidence (highest first)
    proposals.sort(key=lambda p: -p.overall_confidence)

    return proposals


def build_mixed_compound_ast(
    linear_var_idxs,
    linear_exponents,
    trig_var_idxs,
    trig_omegas,
    trig_kinds,
    trig_phases,
):
    """
    Build an AST for a mixed compound variable.

    Creates: z = (x0^a0 * x1^a1 * ...) * cos(ω1*xk + φ1) * sin(ω2*xl + φ2) * ...

    Parameters
    ----------
    linear_var_idxs : tuple of int
        Variable indices for the monomial part.
    linear_exponents : tuple of int
        Exponents for each linear variable.
    trig_var_idxs : tuple of int
        Variable indices for trig terms.
    trig_omegas : tuple of float
        Angular frequencies for each trig variable.
    trig_kinds : tuple of str
        "cos" or "sin" for each trig variable.
    trig_phases : tuple of float
        Phase offsets for each trig variable (0.0 for pure cos/sin).

    Returns
    -------
    Node
        AST representing the mixed compound variable.

    Examples
    --------
    >>> build_mixed_compound_ast((0, 1), (1, 1), (2,), (2.0,), ("cos",), (0.0,))
    # Returns AST for: x0 * x1 * cos(2.0 * x2)
    """
    from .bridges import Add, ConstNode, Cos, Mul, Pow, Sin, Var

    terms = []

    # 1. Build monomial part
    for idx, exp in zip(linear_var_idxs, linear_exponents):
        if exp == 0:
            continue
        elif exp == 1:
            terms.append(Var(int(idx)))
        else:
            terms.append(Pow(Var(int(idx)), int(exp)))

    # 2. Build trig part
    for idx, omega, kind, phase in zip(trig_var_idxs, trig_omegas, trig_kinds, trig_phases):
        # Build argument: omega * x_idx + phase
        if abs(omega - 1.0) < 1e-9:
            arg = Var(int(idx))
        else:
            arg = Mul(ConstNode(float(omega)), Var(int(idx)))

        if abs(phase) > 1e-9:
            arg = Add(arg, ConstNode(float(phase)))

        # Apply trig function
        if kind == "sin":
            terms.append(Sin(arg))
        else:
            terms.append(Cos(arg))

    if len(terms) == 0:
        raise ValueError("No valid terms in mixed compound AST")

    # Multiply all terms together
    result = terms[0]
    for t in terms[1:]:
        result = Mul(result, t)

    return result
