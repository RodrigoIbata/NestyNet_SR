# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Structured dimensional checks for fully symbolic SymPy expressions.

The AST checker in :mod:`nestynet_sr.sr_core.units` constrains expressions while
Stage B is searching.  This module covers the other important boundary: strings
and SymPy trees produced by Stage C and the final equation polisher.  It keeps
numeric fidelity and dimensional admissibility as separate certificates so a
caller can rank only candidates that satisfy both.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Optional, Sequence

try:
    import sympy as sp
except Exception:  # pragma: no cover - supported installs declare SymPy.
    sp = None

from .coefficient_metadata import coefficient_symbol_for_name
from .units import Dim, UnitsSpec, add_dim, scale_dim


def _fraction(value: Any, *, max_denominator: int = 128) -> Fraction:
    """Convert a rational-looking SymPy/Python number without float drift."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float) or getattr(value, "is_Float", False):
        return Fraction.from_float(float(value)).limit_denominator(max_denominator)
    numerator = getattr(value, "p", None)
    denominator = getattr(value, "q", None)
    if numerator is not None and denominator is not None:
        return Fraction(int(numerator), int(denominator))
    try:
        return Fraction(str(value))
    except Exception:
        return Fraction(float(value)).limit_denominator(max_denominator)


def _coerce_dim(dim: Any) -> Dim:
    return tuple(_fraction(value) for value in tuple(dim))


def _dim_payload(dim: Optional[Dim]) -> Optional[list[str]]:
    if dim is None:
        return None
    return [str(value) for value in dim]


@dataclass(frozen=True)
class SympyUnitsCheckResult:
    """Machine-readable result of a SymPy dimensional check."""

    ok: bool
    checked: bool
    code: str
    reason: str
    expression_space: str
    actual_dim: Optional[Dim] = None
    expected_dim: Optional[Dim] = None
    failure_path: Optional[str] = None
    failure_expr: Optional[str] = None
    checker: str = "sympy_units_v1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly admissibility certificate."""

        return {
            "checked": bool(self.checked),
            "valid": bool(self.ok) if self.checked else None,
            "checker": self.checker,
            "code": self.code,
            "reason": self.reason,
            "expression_space": self.expression_space,
            "actual_dim": _dim_payload(self.actual_dim),
            "expected_dim": _dim_payload(self.expected_dim),
            "failure_path": self.failure_path,
            "failure_expr": self.failure_expr,
        }


class _UnitCheckFailure(ValueError):
    def __init__(self, code: str, reason: str, *, path: str, node: Any):
        super().__init__(reason)
        self.code = str(code)
        self.reason = str(reason)
        self.path = str(path)
        try:
            self.expr = sp.sstr(node) if sp is not None else str(node)
        except Exception:
            self.expr = str(node)


def _target_dim(units_spec: UnitsSpec, expression_space: str) -> Dim:
    if expression_space == "phi":
        return _coerce_dim(units_spec.y_phi_dim)
    if expression_space == "y":
        return _coerce_dim(units_spec.y_dim)
    raise ValueError("expression_space must be 'phi' or 'y'")


def _symbol_dimensions(
    variable_names: Sequence[str], units_spec: UnitsSpec, rank: int
) -> dict[str, Dim]:
    x_dims = tuple(_coerce_dim(dim) for dim in units_spec.x_dims)
    mapping: dict[str, Dim] = {}
    owners: dict[str, str] = {}
    reserved_names = {
        "pi",
        "E",
        "I",
        "oo",
        "zoo",
        "nan",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "arcsin",
        "arccos",
        "arctan",
        "exp",
        "log",
        "ln",
        "Abs",
        "abs",
    }

    def add_symbol(
        name: Any,
        dim: Any,
        owner: str,
        *,
        coefficient: bool = False,
    ) -> None:
        symbol_name = (
            coefficient_symbol_for_name(name) if coefficient else str(name)
        )
        if symbol_name in reserved_names:
            raise ValueError(
                f"symbol {symbol_name!r} from {owner} collides with a reserved SymPy name"
            )
        if symbol_name in mapping:
            raise ValueError(
                f"symbol {symbol_name!r} is declared both as {owners[symbol_name]} and {owner}"
            )
        mapping[symbol_name] = _coerce_dim(dim)
        owners[symbol_name] = owner

    for index, name in enumerate(variable_names):
        if index < len(x_dims):
            add_symbol(name, x_dims[index], f"input[{index}]")
    for attr in ("free_const_dims", "fixed_const_dims"):
        declared: Mapping[str, Any] = getattr(units_spec, attr, {}) or {}
        for name, dim in declared.items():
            add_symbol(name, dim, attr, coefficient=True)
    bad = {name: len(dim) for name, dim in mapping.items() if len(dim) != rank}
    if bad:
        raise ValueError(f"dimension-rank mismatch for symbols: {bad}; expected {rank}")
    return mapping


def check_sympy_units(
    expr: Any,
    variable_names: Sequence[str],
    units_spec: Optional[UnitsSpec],
    *,
    expression_space: str = "y",
    target_dim: Optional[Dim] = None,
) -> SympyUnitsCheckResult:
    """Check a SymPy expression against declared variable/constant dimensions.

    Plain numeric literals are dimensionless.  A coefficient that carries
    units must therefore remain a named symbol declared in ``free_const_dims``
    or ``fixed_const_dims``; this is the same rule for unitless and unitful
    constants, with only the declared dimension changing.
    """

    space = str(expression_space).strip().lower()
    if units_spec is None:
        return SympyUnitsCheckResult(
            ok=True,
            checked=False,
            code="units_unchecked",
            reason="no units specification was supplied",
            expression_space=space,
        )
    if sp is None:
        return SympyUnitsCheckResult(
            ok=False,
            checked=True,
            code="checker_unavailable",
            reason="SymPy is unavailable, so dimensional admissibility cannot be certified",
            expression_space=space,
        )

    expected: Optional[Dim] = None
    try:
        expected = _coerce_dim(target_dim) if target_dim is not None else _target_dim(units_spec, space)
        if not expected:
            raise ValueError("target dimension is empty")
        rank = len(expected)
        zero: Dim = tuple(Fraction(0) for _ in range(rank))
        symbols = _symbol_dimensions(variable_names, units_spec, rank)
        if not isinstance(expr, sp.Basic):
            local_names = {name: sp.Symbol(name) for name in symbols}
            local_names.update(
                {
                    "sqrt": sp.sqrt,
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "sinh": sp.sinh,
                    "cosh": sp.cosh,
                    "tanh": sp.tanh,
                    "asin": sp.asin,
                    "acos": sp.acos,
                    "atan": sp.atan,
                    "asinh": sp.asinh,
                    "acosh": sp.acosh,
                    "atanh": sp.atanh,
                    "arcsin": sp.asin,
                    "arccos": sp.acos,
                    "arctan": sp.atan,
                    "exp": sp.exp,
                    "log": sp.log,
                    "ln": sp.log,
                    "Abs": sp.Abs,
                    "abs": sp.Abs,
                    "pi": sp.pi,
                    "E": sp.E,
                }
            )
            expr = sp.sympify(str(expr), locals=local_names)
    except Exception as exc:
        return SympyUnitsCheckResult(
            ok=False,
            checked=True,
            code="units_spec_error",
            reason=f"invalid units context: {exc}",
            expression_space=space,
            expected_dim=expected,
        )

    same_dimension_functions = {
        fn
        for fn in (
            getattr(sp, "Abs", None),
            getattr(sp, "re", None),
            getattr(sp, "im", None),
            getattr(sp, "conjugate", None),
        )
        if fn is not None
    }
    dimensionless_argument_functions = {
        fn
        for fn in (
            getattr(sp, "log", None),
            getattr(sp, "exp", None),
            getattr(sp, "sin", None),
            getattr(sp, "cos", None),
            getattr(sp, "tan", None),
            getattr(sp, "sinh", None),
            getattr(sp, "cosh", None),
            getattr(sp, "tanh", None),
            getattr(sp, "asin", None),
            getattr(sp, "acos", None),
            getattr(sp, "atan", None),
            getattr(sp, "asinh", None),
            getattr(sp, "acosh", None),
            getattr(sp, "atanh", None),
        )
        if fn is not None
    }

    def fail(code: str, reason: str, node: Any, path: str) -> None:
        raise _UnitCheckFailure(code, reason, path=path, node=node)

    def rec(node: Any, path: str) -> Dim:
        if getattr(node, "is_number", False):
            return zero
        if isinstance(node, sp.Symbol):
            name = str(node)
            if name in symbols:
                return symbols[name]
            fail("unknown_symbol", f"symbol {name!r} has no declared dimension", node, path)
        if isinstance(node, sp.Add):
            args = list(sp.Add.make_args(node))
            if not args:
                return zero
            first = rec(args[0], f"{path}.args[0]")
            for index, arg in enumerate(args[1:], start=1):
                other = rec(arg, f"{path}.args[{index}]")
                if other != first:
                    fail(
                        "add_dimension_mismatch",
                        f"addends have dimensions {first} and {other}",
                        node,
                        path,
                    )
            return first
        if isinstance(node, sp.Mul):
            result = zero
            for index, arg in enumerate(sp.Mul.make_args(node)):
                result = add_dim(result, rec(arg, f"{path}.args[{index}]"))
            return result
        if isinstance(node, sp.Pow):
            base_dim = rec(node.base, f"{path}.base")
            exponent = node.exp
            if getattr(exponent, "is_number", False):
                if base_dim == zero:
                    return zero
                try:
                    return scale_dim(base_dim, _fraction(exponent))
                except Exception:
                    fail(
                        "invalid_dimension_power",
                        f"unitful base cannot be raised to exponent {exponent}",
                        node,
                        path,
                    )
            exponent_dim = rec(exponent, f"{path}.exponent")
            if base_dim != zero:
                fail(
                    "symbolic_power_unitful_base",
                    f"symbolic exponent requires a dimensionless base, got {base_dim}",
                    node,
                    path,
                )
            if exponent_dim != zero:
                fail(
                    "power_exponent_not_dimensionless",
                    f"power exponent must be dimensionless, got {exponent_dim}",
                    exponent,
                    f"{path}.exponent",
                )
            return zero
        func = getattr(node, "func", None)
        if func in dimensionless_argument_functions:
            arg_dim = rec(node.args[0], f"{path}.args[0]")
            if arg_dim != zero:
                fail(
                    "function_argument_not_dimensionless",
                    f"{func.__name__} argument must be dimensionless, got {arg_dim}",
                    node.args[0],
                    f"{path}.args[0]",
                )
            return zero
        if func in same_dimension_functions:
            return rec(node.args[0], f"{path}.args[0]")
        if func in {getattr(sp, "Min", None), getattr(sp, "Max", None)}:
            args = list(node.args)
            first = rec(args[0], f"{path}.args[0]")
            for index, arg in enumerate(args[1:], start=1):
                other = rec(arg, f"{path}.args[{index}]")
                if other != first:
                    fail(
                        "ordered_argument_dimension_mismatch",
                        f"{func.__name__} arguments have dimensions {first} and {other}",
                        node,
                        path,
                    )
            return first
        if func == getattr(sp, "atan2", None):
            first = rec(node.args[0], f"{path}.args[0]")
            second = rec(node.args[1], f"{path}.args[1]")
            if first != second:
                fail(
                    "atan2_argument_dimension_mismatch",
                    f"atan2 arguments have dimensions {first} and {second}",
                    node,
                    path,
                )
            return zero
        if func in {getattr(sp, "arg", None), getattr(sp, "sign", None)}:
            rec(node.args[0], f"{path}.args[0]")
            return zero
        fail(
            "unsupported_node",
            f"unsupported dimensional-analysis node {type(node).__name__}",
            node,
            path,
        )

    try:
        actual = rec(expr, "$expr")
    except _UnitCheckFailure as exc:
        return SympyUnitsCheckResult(
            ok=False,
            checked=True,
            code=exc.code,
            reason=exc.reason,
            expression_space=space,
            expected_dim=expected,
            failure_path=exc.path,
            failure_expr=exc.expr,
        )
    except Exception as exc:  # Defensive: dimensional checks must fail closed.
        return SympyUnitsCheckResult(
            ok=False,
            checked=True,
            code="checker_error",
            reason=f"dimensional checker failed: {exc}",
            expression_space=space,
            expected_dim=expected,
        )

    if actual != expected:
        return SympyUnitsCheckResult(
            ok=False,
            checked=True,
            code="target_dimension_mismatch",
            reason=f"expression dimension {actual} does not match target {expected}",
            expression_space=space,
            actual_dim=actual,
            expected_dim=expected,
            failure_path="$expr",
            failure_expr=sp.sstr(expr),
        )
    return SympyUnitsCheckResult(
        ok=True,
        checked=True,
        code="units_ok",
        reason="expression is dimensionally admissible",
        expression_space=space,
        actual_dim=actual,
        expected_dim=expected,
    )


__all__ = ["SympyUnitsCheckResult", "check_sympy_units"]
