# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

"""Shared planning helpers for monomial subset and partial peel proposals.

These helpers are intentionally proposal-only.  They split monomial evidence
into visible clean integer factors and residual coordinates, but Stage A/B
validation remains responsible for accepting or rejecting the resulting AST.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CleanPowerSplitPlan:
    """Clean integer prefactor powers plus residual input positions."""

    full_powers: Tuple[Fraction, ...]
    clean_powers: Tuple[int, ...]
    residual_indices: Tuple[int, ...]

    @property
    def clean_support(self) -> Tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.clean_powers) if int(p) != 0)

    @property
    def residual_powers(self) -> Tuple[Fraction, ...]:
        return tuple(self.full_powers[i] for i in self.residual_indices)


def as_fraction(value, *, max_denominator: int = 64) -> Fraction:
    """Convert a numeric exponent to a small rational exponent."""

    if isinstance(value, Fraction):
        return value
    try:
        return Fraction(value).limit_denominator(int(max_denominator))
    except Exception:
        try:
            fv = float(value)
        except Exception:
            return Fraction(0)
        if not math.isfinite(fv):
            return Fraction(0)
        return Fraction(fv).limit_denominator(int(max_denominator))


def expand_forced_power_vector(
    *,
    pattern: Sequence[int],
    basis_powers: Sequence[Fraction | int | float],
    extra_local_indices: Sequence[int] = (),
) -> Optional[Tuple[Fraction, ...]]:
    """Expand powers of ``z`` and extras into powers over local atom inputs.

    ``basis_powers`` follows the Stage-A forced-monomial basis convention:
    first the proposed monomial coordinate ``z`` and then any preserved extras.
    ``pattern`` is the exponent vector defining ``z`` over local atom inputs.
    """

    if not basis_powers:
        return None
    try:
        pat = tuple(int(v) for v in pattern)
    except Exception:
        return None
    if not pat:
        return None

    powers = [Fraction(0) for _ in pat]
    z_power = as_fraction(basis_powers[0])
    for i, exp in enumerate(pat):
        if int(exp) != 0:
            powers[i] += z_power * Fraction(int(exp))

    for power, local_idx in zip(tuple(basis_powers)[1:], tuple(extra_local_indices)):
        try:
            li = int(local_idx)
        except Exception:
            continue
        if 0 <= li < len(powers):
            powers[li] += as_fraction(power)
    return tuple(powers)


def split_clean_integer_powers(
    powers: Sequence[Fraction | int | float],
    *,
    max_abs_clean_power: int = 8,
    min_clean_support: int = 1,
    min_residual_support: int = 1,
) -> Optional[CleanPowerSplitPlan]:
    """Split exact integer powers into a visible prefactor and residual inputs.

    Non-integer nonzero powers are treated as ambiguous residual structure and
    are left inside the NN.  We deliberately do not peel integer floors from
    fractional powers: ``x**(5/2)`` remains entirely residual, not ``x**2``
    times ``sqrt(x)``.
    """

    full = tuple(as_fraction(p) for p in powers)
    clean = []
    residual = []
    for i, p in enumerate(full):
        if p == 0:
            clean.append(0)
            continue
        if p.denominator == 1 and abs(int(p)) <= int(max_abs_clean_power):
            clean.append(int(p))
            continue
        clean.append(0)
        residual.append(int(i))

    if sum(1 for p in clean if int(p) != 0) < int(min_clean_support):
        return None
    if len(residual) < int(min_residual_support):
        return None
    return CleanPowerSplitPlan(
        full_powers=full,
        clean_powers=tuple(int(v) for v in clean),
        residual_indices=tuple(residual),
    )


def clean_subset_patterns(
    pattern: Sequence[int],
    *,
    max_subsets: int = 6,
    min_support: int = 2,
) -> Tuple[Tuple[int, ...], ...]:
    """Return a small set of integer subproduct patterns for a monomial z.

    This is coordinate-compression evidence only.  The generated subsets are
    biased toward high-magnitude exponent groups, then conservative pairs.
    """

    try:
        pat = tuple(int(v) for v in pattern)
    except Exception:
        return ()
    support = tuple(i for i, v in enumerate(pat) if int(v) != 0)
    if len(support) <= int(min_support):
        return ()

    out = []
    seen = set()

    def _add(indices: Iterable[int]) -> None:
        if len(out) >= int(max_subsets):
            return
        idxs = tuple(sorted(set(int(i) for i in indices)))
        if len(idxs) < int(min_support) or len(idxs) >= len(support):
            return
        sub = tuple(int(pat[i]) if i in idxs else 0 for i in range(len(pat)))
        if sub in seen:
            return
        seen.add(sub)
        out.append(sub)

    abs_levels = sorted({abs(int(pat[i])) for i in support}, reverse=True)
    for level in abs_levels:
        if len(out) >= int(max_subsets):
            break
        _add(i for i in support if abs(int(pat[i])) >= int(level))

    pos = [i for i in support if int(pat[i]) > 0]
    neg = [i for i in support if int(pat[i]) < 0]
    _add(pos)
    _add(neg)

    ranked_pairs = sorted(
        itertools.combinations(support, 2),
        key=lambda ij: (
            -(abs(int(pat[ij[0]])) + abs(int(pat[ij[1]]))),
            ij,
        ),
    )
    for pair in ranked_pairs:
        if len(out) >= int(max_subsets):
            break
        _add(pair)

    return tuple(out)
