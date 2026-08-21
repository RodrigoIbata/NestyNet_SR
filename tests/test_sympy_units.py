# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from fractions import Fraction

import sympy as sp

from nestynet_sr.sr_core.sympy_units import check_sympy_units
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec


def _spec(*, x_dims, y_dim, y_transform_name="identity", **kwargs):
    unit_system = UnitSystem(("L", "T"))
    return UnitsSpec(
        unit_system=unit_system,
        x_dims=tuple(unit_system.dim(dim) for dim in x_dims),
        y_dim=unit_system.dim(y_dim),
        y_transform_name=y_transform_name,
        **kwargs,
    )


def test_log_of_ratio_with_matching_units_is_valid():
    spec = _spec(x_dims=({"L": 1}, {"T": 1}, {"T": 1}), y_dim={"L": 1})
    x0, x1, x2 = sp.symbols("x0 x1 x2")

    result = check_sympy_units(x0 * sp.log(x1 / x2), ("x0", "x1", "x2"), spec)

    assert result.ok is True
    assert result.checked is True
    assert result.to_dict()["valid"] is True


def test_additive_mismatch_reports_structured_failure_location():
    spec = _spec(x_dims=({"L": 1}, {"T": 1}), y_dim={"L": 1})
    x0, x1 = sp.symbols("x0 x1")

    result = check_sympy_units(x0 + x1, ("x0", "x1"), spec)

    assert result.ok is False
    assert result.code == "add_dimension_mismatch"
    assert result.failure_path == "$expr"
    assert result.to_dict()["valid"] is False


def test_dimensionful_transcendental_argument_is_invalid():
    spec = _spec(x_dims=({"L": 1},), y_dim={})
    x0 = sp.Symbol("x0")

    result = check_sympy_units(sp.exp(x0), ("x0",), spec)

    assert result.ok is False
    assert result.code == "function_argument_not_dimensionless"
    assert result.failure_expr == "x0"


def test_string_parser_normalizes_inverse_trig_aliases_used_by_y_wrappers():
    spec = _spec(x_dims=({},), y_dim={})

    result = check_sympy_units("arcsin(x0)", ("x0",), spec)

    assert result.ok is True


def test_declared_constant_uses_same_symbol_path_for_unitful_and_unitless_cases():
    unit_system = UnitSystem(("L", "T"))
    speed = unit_system.dim({"L": 1, "T": -1})
    time = unit_system.dim({"T": 1})
    length = unit_system.dim({"L": 1})
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(time,),
        y_dim=length,
        free_const_dims={"c": speed},
    )
    c, x0 = sp.symbols("c x0")

    unitful = check_sympy_units(c * x0, ("x0",), spec)
    unitless = check_sympy_units(
        c * x0,
        ("x0",),
        UnitsSpec(
            unit_system=unit_system,
            x_dims=(length,),
            y_dim=length,
            free_const_dims={"c": unit_system.dimless()},
        ),
    )

    assert unitful.ok is True
    assert unitless.ok is True


def test_fixed_unitful_constant_is_recognized_by_name():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"T": 2}),),
        y_dim=unit_system.dim({"L": 1}),
        fixed_const_dims={"g": unit_system.dim({"L": 1, "T": -2})},
        fixed_const_values={"g": 9.81},
    )
    g, x0 = sp.symbols("g x0")

    result = check_sympy_units(g * x0, ("x0",), spec)

    assert result.ok is True


def test_phi_and_raw_y_targets_are_checked_separately():
    spec = _spec(
        x_dims=({"L": 1},),
        y_dim={"L": 2},
        y_transform_name="sqrt",
    )
    x0 = sp.Symbol("x0")

    phi_result = check_sympy_units(x0, ("x0",), spec, expression_space="phi")
    y_result = check_sympy_units(x0, ("x0",), spec, expression_space="y")

    assert phi_result.ok is True
    assert phi_result.expected_dim == (Fraction(1), Fraction(0))
    assert y_result.ok is False
    assert y_result.code == "target_dimension_mismatch"


def test_fitted_float_power_is_rationalized_for_dimension_comparison():
    spec = _spec(x_dims=({"L": 1},), y_dim={"L": 2})
    x0 = sp.Symbol("x0")

    result = check_sympy_units(x0 ** sp.Float("1.999999999"), ("x0",), spec)

    assert result.ok is True
    assert result.actual_dim == (Fraction(2), Fraction(0))


def test_missing_spec_is_explicitly_unchecked_not_invalid():
    result = check_sympy_units(sp.Symbol("x0"), ("x0",), None)

    assert result.ok is True
    assert result.checked is False
    assert result.to_dict()["valid"] is None


def test_unknown_symbol_fails_closed():
    spec = _spec(x_dims=({"L": 1},), y_dim={"L": 1})

    result = check_sympy_units(sp.Symbol("mystery"), ("x0",), spec)

    assert result.ok is False
    assert result.code == "unknown_symbol"


def test_declared_constant_matching_input_namespace_uses_safe_alias():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"L": 1}),),
        y_dim=unit_system.dim({"T": 1}),
        free_const_dims={"x0": unit_system.dim({"T": 1})},
    )

    result = check_sympy_units("coef_x0", ("x0",), spec)

    assert result.ok is True
    assert result.code == "units_ok"


def test_declared_constant_reserved_sympy_name_uses_safe_alias():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"L": 1}),),
        y_dim=unit_system.dim({"L": 1}),
        fixed_const_dims={"E": unit_system.dimless()},
        fixed_const_values={"E": float(sp.E)},
    )

    result = check_sympy_units("coef_E*x0", ("x0",), spec)

    assert result.ok is True
    assert result.code == "units_ok"
