# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

from nestynet_sr.sr_core.bridges import Var
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, eval_analytic_expr_dim
from nestynet_sr.sr_search.compound_functions import _default_macros, _macro_expr_from, _macro_units_ok
from nestynet_sr.sr_search.feature_grammar import FeatureExpr


def _spec(us, x_dims, y_dim):
    return UnitsSpec(unit_system=us, x_dims=tuple(x_dims), y_dim=y_dim)


def _macro(name):
    matches = [m for m in _default_macros() if m.name == name]
    assert matches, name
    return matches[0]


def _arg(i=0):
    return FeatureExpr(expr=Var(i), kind="var", cost=0, desc=f"x{i}")


def test_periodic_macro_registry_contains_half_angle_motifs():
    names = {m.name for m in _default_macros()}

    assert {
        "sin_half_sq",
        "cos_half_sq",
        "one_minus_cos",
        "inv_sin_half_sq",
        "inv_sin_half4",
        "sinc_sq",
    }.issubset(names)


def test_periodic_macro_unit_prefilter_rejects_unitful_arguments():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    macro = _macro("sin_half_sq")
    expr = _macro_expr_from(macro, (_arg(0),))

    dimless_ctx = SimpleNamespace(enforce_units=True, units_spec=_spec(us, [dimless], dimless))
    unitful_ctx = SimpleNamespace(enforce_units=True, units_spec=_spec(us, [L], dimless))

    assert _macro_units_ok(dimless_ctx, expr)
    assert not _macro_units_ok(unitful_ctx, expr)


def test_macro_unit_prefilter_allows_unitful_hypot_internal_dimension():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    macro = _macro("hypot")
    expr = _macro_expr_from(macro, (_arg(0), _arg(1)))
    ctx = SimpleNamespace(enforce_units=True, units_spec=_spec(us, [L, L], L))

    assert _macro_units_ok(ctx, expr)
    assert eval_analytic_expr_dim(expr, (L, L)) == L
