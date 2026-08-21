# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from fractions import Fraction

from nestynet_sr.sr_search.monomial_peel_plan import (
    clean_subset_patterns,
    expand_forced_power_vector,
    split_clean_integer_powers,
)


def test_forced_power_split_leaves_fractional_residual_axes():
    full = expand_forced_power_vector(
        pattern=(2, -2, 1, 1, -2),
        basis_powers=(Fraction(5, 2), Fraction(-1, 1)),
        extra_local_indices=(0,),
    )

    assert full == (
        Fraction(4, 1),
        Fraction(-5, 1),
        Fraction(5, 2),
        Fraction(5, 2),
        Fraction(-5, 1),
    )

    plan = split_clean_integer_powers(full)

    assert plan is not None
    assert plan.clean_powers == (4, -5, 0, 0, -5)
    assert plan.residual_indices == (2, 3)


def test_clean_subset_patterns_prefers_high_magnitude_subproduct():
    subsets = clean_subset_patterns((2, -2, 1, 1, -2), max_subsets=4)

    assert (2, -2, 0, 0, -2) in subsets
    assert all(sum(1 for v in pat if v) >= 2 for pat in subsets)
