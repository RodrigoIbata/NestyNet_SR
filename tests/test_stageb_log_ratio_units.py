# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from fractions import Fraction

from nestynet_sr.sr_core.bridges import AtomNode, LogNode, MulNode, Var
from nestynet_sr.sr_core.units import (
    UnitSystem,
    UnitsSpec,
    check_units_ast,
    eval_analytic_expr_dim,
)
from nestynet_sr.sr_search.stageB.rules import (
    _dimensionless_log_monomial_powers,
    _make_log_monomial_expr,
)


def test_log_ratio_units_prefers_dimensionless_compound_for_pb047_shape():
    us = UnitSystem(("L", "T", "M", "I", "Theta"))
    spec = UnitsSpec(
        unit_system=us,
        y_dim=us.dim([2, -2, 1, 0, 0]),
        x_dims=tuple(
            us.dim(v)
            for v in [
                [0, 0, 0, 0, 0],
                [2, -2, 1, -1, 0],
                [0, 0, 0, 1, 0],
                [3, 0, 0, 0, 0],
                [3, 0, 0, 0, 0],
            ]
        ),
    )
    prefactor = MulNode(MulNode(Var(0), Var(1)), Var(2))
    powers = _dimensionless_log_monomial_powers(3, 4, spec)

    assert powers == (Fraction(1), Fraction(-1))

    z_expr = _make_log_monomial_expr(3, powers[0], 4, powers[1])
    compound_log = AtomNode(
        kind="polylog",
        var_idxs=(3, 4),
        kwargs={"degree": 1},
        inputs=(z_expr,),
        tag="logratio_compound",
    )
    raw_logs = AtomNode(
        kind="polylog",
        var_idxs=(3, 4),
        kwargs={"degree": 1},
        tag="logratio_raw",
    )

    assert check_units_ast(MulNode(prefactor, compound_log), spec).ok
    assert not check_units_ast(MulNode(prefactor, raw_logs), spec).ok


def test_log_monomial_units_generalize_beyond_equal_dimensions():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        y_dim=us.dim({}),
        x_dims=(us.dim({"L": 2}), us.dim({"L": 1}), us.dim({"T": 1})),
    )

    powers = _dimensionless_log_monomial_powers(0, 1, spec)
    assert powers == (Fraction(1), Fraction(-2))

    z_expr = _make_log_monomial_expr(0, powers[0], 1, powers[1])
    assert eval_analytic_expr_dim(LogNode(z_expr), spec.x_dims) == us.dim({})
    assert _dimensionless_log_monomial_powers(0, 2, spec) is None
