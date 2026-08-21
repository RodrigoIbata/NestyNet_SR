# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
"""Measured finite-sample calibration envelope for simultaneous Pareto inference.

The multiplier max-T is asymptotically calibrated and anti-conservative in
finite samples.  The deficiency is governed jointly by the number of
independent audit units ``G`` and the *pre-audit* admissible comparison count
``K_pre``, and at the far corner it is not a degradation but a loss of control:
at ``G=12`` with ``K=10100`` the familywise false-edge rate is 0.971 against a
nominal 0.05.

This module holds the measured grid and the lookup that decides, for a given
audit, whether the multiplier procedure is licensed or the conservative
Bonferroni-t fallback governs.

Three rules make the lookup trustworthy.

**The table is the authority.**  There is no fitted collapse law.  A 15-cell
pilot suggested ``log(K)^1.5/G`` collapsed the grid, with a worst
near-neighbour gap of 0.026; at 63 cells the best candidate leaves 0.124, which
is 2.5 times the nominal rate.  That was small-sample optimism, and a switching
policy built on it would have rested on noise.

**Licensing is monotone and conservative.**  A query is licensed only when some
*measured* cell that is strictly harder in both coordinates (no more units, no
fewer comparisons) is itself validated.  Rate falls with ``G`` and rises with
``K`` throughout the measured grid, so a harder validated cell bounds the query.

**Escaping the grid is reported, never guessed.**  ``K_pre`` above the measured
maximum cannot be bounded by any cell, so it returns ``beyond_grid`` rather
than silently taking the fallback.  A silent fallback there would look like the
policy working while discarding the power the max-T exists to provide.

Cells are classified on the Wilson upper bound of the observed rate, never the
point estimate, because the calibration experiment carries Monte Carlo error of
its own.  At 800 replicates a cell whose true rate is exactly 0.05 yields an
upper bound near 0.066, so it could not certify at a 0.06 threshold; the
replicate count is therefore part of the method specification and is recorded
here rather than treated as provenance.

Regenerate with ``scripts/run_calibration_lab.py`` and
``nestynet_sr.stat_selection.calibration_lab.calibration_envelope``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "CalibrationCell",
    "CalibrationProfile",
    "MAXT_PROFILE_V1",
    "select_inference_method",
]


@dataclass(frozen=True)
class CalibrationCell:
    """One measured grid point."""

    n_units: int
    k_admissible: int
    false_edge_rate: float
    wilson_upper_bound: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_units": int(self.n_units),
            "k_admissible": int(self.k_admissible),
            "false_edge_rate": float(self.false_edge_rate),
            "wilson_upper_bound": float(self.wilson_upper_bound),
            "status": str(self.status),
        }


# (G, K_adm, false_edge_rate, wilson_upper_95, status)
_CELLS: tuple[tuple[int, int, float, float, str], ...] = (
    (  12,     12, 0.1144, 0.1253, "outside"),
    (  18,     12, 0.0892, 0.0990, "outside"),
    (  24,     12, 0.0736, 0.0827, "outside"),
    (  36,     12, 0.0664, 0.0751, "transitional"),
    (  60,     12, 0.0584, 0.0666, "transitional"),
    ( 100,     12, 0.0604, 0.0687, "transitional"),
    ( 150,     12, 0.0528, 0.0607, "transitional"),
    ( 250,     12, 0.0508, 0.0585, "validated"),
    ( 400,     12, 0.0480, 0.0555, "validated"),
    (  12,     30, 0.1604, 0.1728, "outside"),
    (  18,     30, 0.1180, 0.1290, "outside"),
    (  24,     30, 0.1048, 0.1153, "outside"),
    (  36,     30, 0.0756, 0.0848, "outside"),
    (  60,     30, 0.0736, 0.0827, "outside"),
    ( 100,     30, 0.0552, 0.0632, "transitional"),
    ( 150,     30, 0.0484, 0.0560, "validated"),
    ( 250,     30, 0.0480, 0.0555, "validated"),
    ( 400,     30, 0.0452, 0.0525, "validated"),
    (  12,    110, 0.2644, 0.2792, "outside"),
    (  18,    110, 0.1628, 0.1753, "outside"),
    (  24,    110, 0.1408, 0.1526, "outside"),
    (  36,    110, 0.1020, 0.1124, "outside"),
    (  60,    110, 0.0796, 0.0890, "outside"),
    ( 100,    110, 0.0668, 0.0755, "transitional"),
    ( 150,    110, 0.0644, 0.0730, "transitional"),
    ( 250,    110, 0.0576, 0.0658, "transitional"),
    ( 400,    110, 0.0484, 0.0560, "validated"),
    (  12,    306, 0.3980, 0.4142, "outside"),
    (  18,    306, 0.2504, 0.2649, "outside"),
    (  24,    306, 0.1836, 0.1967, "outside"),
    (  36,    306, 0.1364, 0.1481, "outside"),
    (  60,    306, 0.0896, 0.0994, "outside"),
    ( 100,    306, 0.0652, 0.0738, "transitional"),
    ( 150,    306, 0.0668, 0.0755, "transitional"),
    ( 250,    306, 0.0644, 0.0730, "transitional"),
    ( 400,    306, 0.0624, 0.0708, "transitional"),
    (  12,   1056, 0.6360, 0.6517, "outside"),
    (  18,   1056, 0.3928, 0.4090, "outside"),
    (  24,   1056, 0.2752, 0.2901, "outside"),
    (  36,   1056, 0.1768, 0.1897, "outside"),
    (  60,   1056, 0.1080, 0.1186, "outside"),
    ( 100,   1056, 0.0908, 0.1007, "outside"),
    ( 150,   1056, 0.0748, 0.0839, "outside"),
    ( 250,   1056, 0.0688, 0.0776, "transitional"),
    ( 400,   1056, 0.0592, 0.0675, "transitional"),
    (  12,   3080, 0.8376, 0.8494, "outside"),
    (  18,   3080, 0.5408, 0.5571, "outside"),
    (  24,   3080, 0.3800, 0.3961, "outside"),
    (  36,   3080, 0.2296, 0.2437, "outside"),
    (  60,   3080, 0.1388, 0.1506, "outside"),
    ( 100,   3080, 0.0968, 0.1070, "outside"),
    ( 150,   3080, 0.0800, 0.0894, "outside"),
    ( 250,   3080, 0.0672, 0.0759, "transitional"),
    ( 400,   3080, 0.0624, 0.0708, "transitional"),
    (  12,  10100, 0.9712, 0.9762, "outside"),
    (  18,  10100, 0.7604, 0.7742, "outside"),
    (  24,  10100, 0.5316, 0.5480, "outside"),
    (  36,  10100, 0.3040, 0.3193, "outside"),
    (  60,  10100, 0.1816, 0.1946, "outside"),
    ( 100,  10100, 0.1140, 0.1249, "outside"),
    ( 150,  10100, 0.0836, 0.0932, "outside"),
    ( 250,  10100, 0.0676, 0.0763, "transitional"),
    ( 400,  10100, 0.0616, 0.0700, "transitional"),    # High-G extension: ordinary SR runs at G in the thousands, and the
    # K=10100 row was still falling at the original grid edge.
    (1000,   1056, 0.0592, 0.0675, "transitional"),
    (2000,   1056, 0.0556, 0.0636, "transitional"),
    (5000,   1056, 0.0504, 0.0581, "validated"),
    (1000,   3080, 0.0652, 0.0738, "transitional"),
    (2000,   3080, 0.0568, 0.0649, "transitional"),
    (5000,   3080, 0.0496, 0.0572, "validated"),
    (1000,  10100, 0.0632, 0.0717, "transitional"),
    (2000,  10100, 0.0504, 0.0581, "validated"),
    (5000,  10100, 0.0540, 0.0619, "transitional"),
)


@dataclass(frozen=True)
class CalibrationProfile:
    """A versioned, measured operating envelope plus its licensing rule.

    Every field that changes what the numbers mean is recorded, because a
    profile measured at a different alpha, multiplier or replicate count is a
    different profile and must not be silently reused.
    """

    profile_version: str
    alpha: float
    replicates_per_cell: int
    n_resamples: int
    multiplier: str
    studentization_rule: str
    validated_threshold: float
    transitional_threshold: float
    cells: tuple[CalibrationCell, ...]
    notes: str = ""

    @property
    def grid_units(self) -> tuple[int, ...]:
        return tuple(sorted({cell.n_units for cell in self.cells}))

    @property
    def grid_comparisons(self) -> tuple[int, ...]:
        return tuple(sorted({cell.k_admissible for cell in self.cells}))

    def lookup(self, *, n_units: int, k_pre: int) -> dict[str, Any]:
        """Decide which inference method is licensed for ``(G, K_pre)``.

        ``k_pre`` must be the **pre-audit** admissible comparison count, derived
        only from frozen candidate structure and declared complexity.  Keying on
        the estimable subset instead would let a candidate shrink the
        multiplicity burden by failing on the audit, which is a post-audit leak.

        Returns ``decision`` in ``{"licensed", "fallback", "beyond_grid"}``.
        """
        units = int(n_units)
        comparisons = int(k_pre)
        if units < 1:
            raise ValueError("n_units must be positive")
        if comparisons < 0:
            raise ValueError("k_pre must be non-negative")

        max_measured_k = max(self.grid_comparisons)
        base: dict[str, Any] = {
            "profile_version": self.profile_version,
            "alpha": float(self.alpha),
            "n_units": units,
            "k_pre": comparisons,
            "calibration_lookup_key": [units, comparisons],
            "replicates_per_cell": int(self.replicates_per_cell),
            "multiplier": self.multiplier,
            "studentization_rule": self.studentization_rule,
            "validated_threshold": float(self.validated_threshold),
        }

        if comparisons > max_measured_k:
            # No measured cell is at least this hard, so nothing bounds the
            # query.  Refuse rather than extrapolate; see the module docstring.
            base.update({
                "decision": "beyond_grid",
                "method": "bonferroni_t",
                "witness_cell": None,
                "escaped_coordinate": "k_pre",
                "reason": (
                    f"pre-audit comparison count {comparisons} exceeds the largest "
                    f"measured cell ({max_measured_k}); the envelope does not cover "
                    "this configuration and is not extrapolated"
                ),
            })
            return base

        # Licensed when some measured cell that is harder in BOTH coordinates
        # (no more units, no fewer comparisons) is itself validated.
        witnesses = [
            cell for cell in self.cells
            if cell.status == "validated"
            and cell.n_units <= units
            and cell.k_admissible >= comparisons
        ]
        if witnesses:
            # Tightest witness: fewest units, then most comparisons.
            witness = min(witnesses, key=lambda c: (c.n_units, -c.k_admissible))
            base.update({
                "decision": "licensed",
                "method": "multiplier_max_t",
                "witness_cell": witness.to_dict(),
                "escaped_coordinate": None,
                "reason": (
                    f"validated measured cell at G={witness.n_units}, "
                    f"K={witness.k_admissible} is harder in both coordinates"
                ),
            })
            return base

        nearest = min(
            self.cells,
            key=lambda c: (abs(c.n_units - units), abs(c.k_admissible - comparisons)),
        )
        base.update({
            "decision": "fallback",
            "method": "bonferroni_t",
            "witness_cell": None,
            "escaped_coordinate": None,
            "reason": (
                f"no validated measured cell is harder in both coordinates; nearest "
                f"measured cell G={nearest.n_units}, K={nearest.k_admissible} has "
                f"false-edge rate {nearest.false_edge_rate:.3f} "
                f"(upper bound {nearest.wilson_upper_bound:.3f})"
            ),
        })
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "alpha": float(self.alpha),
            "replicates_per_cell": int(self.replicates_per_cell),
            "n_resamples": int(self.n_resamples),
            "multiplier": self.multiplier,
            "studentization_rule": self.studentization_rule,
            "validated_threshold": float(self.validated_threshold),
            "transitional_threshold": float(self.transitional_threshold),
            "grid_units": list(self.grid_units),
            "grid_comparisons": list(self.grid_comparisons),
            "cells": [cell.to_dict() for cell in self.cells],
            "notes": self.notes,
        }


MAXT_PROFILE_V1 = CalibrationProfile(
    profile_version="maxt-envelope-2026-07-30-v2",
    alpha=0.05,
    replicates_per_cell=2500,
    n_resamples=1000,
    multiplier="normal",
    studentization_rule="paired_difference_ddof1_over_sqrt_units",
    validated_threshold=0.06,
    transitional_threshold=0.08,
    cells=tuple(
        CalibrationCell(
            n_units=g, k_admissible=k, false_edge_rate=rate,
            wilson_upper_bound=upper, status=status,
        )
        for g, k, rate, upper, status in _CELLS
    ),
    notes=(
        "Equal-risk equal-complexity null candidates, correlated losses "
        "(rho=0.85), heteroscedastic marginal variances (spread 8). Cells "
        "classified on the Wilson upper bound, not the point estimate. "
        "Transitional cells route to the fallback: transitional is descriptive "
        "metadata, not permission to spend type-I error."
    ),
)


def select_inference_method(
    *,
    n_units: int,
    k_pre: int,
    profile: Optional[CalibrationProfile] = None,
) -> dict[str, Any]:
    """Table-driven method selection for a given audit configuration."""
    return (profile or MAXT_PROFILE_V1).lookup(n_units=n_units, k_pre=k_pre)
