# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_core.bridges import MulNode, PowNode, Var
from nestynet_sr.sr_core.units import UnitSystem, eval_analytic_expr_dim
from nestynet_sr.sr_search.compound_proposals import apply_compound_wrapper, build_compound_wrappers


def test_unitful_sqrt_wrapper_is_allowed_when_dimensionally_meaningful():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    L2 = us.dim({"L": 2})

    wrapped = apply_compound_wrapper(Var(0), tuple(L2), "sqrt", strict_units=True)

    assert wrapped is not None
    assert wrapped.dim == tuple(L)
    assert isinstance(wrapped.expr, PowNode)
    assert wrapped.expr.exponent == 0.5


def test_unitful_inverse_sqrt_wrapper_is_allowed_when_dimensionally_meaningful():
    us = UnitSystem(("L", "T", "M"))
    inv_L = us.dim({"L": -1})
    L2 = us.dim({"L": 2})

    wrapped = apply_compound_wrapper(Var(0), tuple(L2), "inv_sqrt", strict_units=True)

    assert wrapped is not None
    assert wrapped.dim == tuple(inv_L)
    assert isinstance(wrapped.expr, PowNode)
    assert wrapped.expr.exponent == -0.5


def test_unitful_sqrt1p_wrapper_is_rejected_in_strict_units():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})

    wrapped = apply_compound_wrapper(Var(0), tuple(L), "sqrt1p", strict_units=True)

    assert wrapped is None


def test_dimensionless_sqrt1p_wrapper_is_allowed():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()

    wrapped = apply_compound_wrapper(Var(0), tuple(dimless), "sqrt1p", strict_units=True)

    assert wrapped is not None
    assert wrapped.dim == tuple(dimless)
    assert isinstance(wrapped.expr, PowNode)


def test_log_ratio_wrapper_requires_dimensionless_argument():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    ratio = MulNode(Var(0), PowNode(Var(1), -1.0))
    ratio_dim = eval_analytic_expr_dim(ratio, (L, L))

    wrapped = apply_compound_wrapper(ratio, tuple(ratio_dim), "log", strict_units=True)
    rejected = apply_compound_wrapper(Var(0), tuple(L), "log", strict_units=True)

    assert ratio_dim == dimless
    assert wrapped is not None
    assert wrapped.dim == tuple(dimless)
    assert rejected is None


def test_trig_wrappers_reject_unitful_arguments():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})

    assert apply_compound_wrapper(Var(0), tuple(L), "sin", strict_units=True) is None
    assert apply_compound_wrapper(Var(0), tuple(L), "cos", strict_units=True) is None
    assert apply_compound_wrapper(Var(0), tuple(L), "sin_half_sq", strict_units=True) is None
    assert apply_compound_wrapper(Var(0), tuple(L), "cos_half_sq", strict_units=True) is None
    assert apply_compound_wrapper(Var(0), tuple(L), "sinc_sq", strict_units=True) is None


def test_periodic_wrappers_accept_dimensionless_half_angle_and_sinc():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()

    wrappers = build_compound_wrappers(
        Var(0),
        tuple(dimless),
        ("sin_half_sq", "cos_half_sq", "inv_sin_half_sq", "inv_sin_half4", "sinc_sq"),
        strict_units=True,
    )

    assert [w.name for w in wrappers] == [
        "sin_half_sq",
        "cos_half_sq",
        "inv_sin_half_sq",
        "inv_sin_half4",
        "sinc_sq",
    ]
    assert all(w.dim == tuple(dimless) for w in wrappers)


def test_build_compound_wrappers_filters_invalid_unitful_wrappers():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})

    wrapped = build_compound_wrappers(
        Var(0),
        tuple(L),
        ("z", "sqrt", "sin", "inv_q"),
        strict_units=True,
    )

    assert [w.name for w in wrapped] == ["z", "sqrt", "inv_q"]
