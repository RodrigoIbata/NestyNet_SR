# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Small units-aware analytic transfer basis for additive gauge probes.

This module does not perform gauge transfer by itself.  It only builds a
bounded set of visible expressions of the form ``C * phi(z)`` that future
scope-aware rules can use as candidate null moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import (
    AddNode,
    ConstNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    Var,
    ast_to_human_readable,
    clone_ast,
)
from nestynet_sr.sr_core.constants import make_unit_aware_scalar_atom, safe_const_token
from nestynet_sr.sr_core.units import eval_analytic_expr_dim, is_dimless, scale_dim, sub_dim
from nestynet_sr.sr_search.feature_grammar import ast_key


@dataclass(frozen=True)
class TransferFeature:
    """One analytic transfer term h(shared) = C * phi(z)."""

    expr: Node
    basis_expr: Node
    desc: str
    cost: int
    dim: Any | None
    coeff_dim: Any | None
    domain: str = "all"
    local_eval_factory: Optional[Callable[..., Any]] = None


def build_transfer_basis(
    *,
    shared_vars: Sequence[int],
    ctx: Any = None,
    units_spec: Any = None,
    required_dim: Any = None,
    shared_inputs: Optional[Sequence[Node]] = None,
    max_features: int = 32,
    allow_logs: bool = True,
    allow_sqrt1m: bool = True,
    allow_rational: bool = True,
    strict_units: bool = True,
) -> List[TransferFeature]:
    """Build a tiny units-aware transfer basis over shared variables.

    The conservative v1 form is ``C * phi(z)``.  Under strict units, nonlinear
    functions such as ``log(z)`` and ``sqrt(1-z**2)`` require dimensionless
    ``z``.  Unitful coefficients require a declared matching free constant.
    """
    if units_spec is None and ctx is not None:
        units_spec = getattr(ctx, "units_spec", None)

    shared = tuple(sorted({int(v) for v in shared_vars}))
    max_features = max(0, int(max_features))
    if max_features <= 0:
        return []

    z_pool = _build_z_pool(shared, shared_inputs=shared_inputs, units_spec=units_spec)
    out: List[TransferFeature] = []
    seen = set()

    def add_feature(phi: Node, phi_dim: Any, desc: str, cost: int, *, domain: str = "all") -> None:
        coeff_dim = _coefficient_dim(required_dim, phi_dim, units_spec=units_spec, strict_units=strict_units)
        if coeff_dim is _REJECT:
            return
        key = (ast_key(phi), tuple(coeff_dim) if coeff_dim is not None else None)
        if key in seen:
            return
        coeff = _make_coeff_atom(coeff_dim, units_spec, len(out), desc, strict_units=strict_units)
        if coeff is None:
            return
        seen.add(key)
        phi_c = _clone(phi)
        if _is_one(phi_c):
            expr = coeff
        else:
            expr = MulNode(coeff, phi_c)
        out.append(
            TransferFeature(
                expr=expr,
                basis_expr=_clone(phi),
                desc=f"C*{desc}" if not _is_one(phi) else "C",
                cost=int(cost) + 1,
                dim=required_dim,
                coeff_dim=coeff_dim,
                domain=str(domain),
            )
        )

    # Constant transfer.
    add_feature(ConstNode(1.0), _dimless(units_spec), "1", 0)

    for z, z_dim, z_desc, z_cost in z_pool:
        add_feature(z, z_dim, z_desc, z_cost)

        if allow_rational:
            add_feature(PowNode(_clone(z), -1.0), _scale_dim_safe(z_dim, -1.0), f"1/({z_desc})", z_cost + 1)

        add_feature(PowNode(_clone(z), 2.0), _scale_dim_safe(z_dim, 2.0), f"({z_desc})^2", z_cost + 1)

        if _nonlinear_arg_allowed(z_dim, units_spec):
            one_minus_z = _sub(ConstNode(1.0), _clone(z))
            add_feature(one_minus_z, _dimless(units_spec), f"1-({z_desc})", z_cost + 1)

            z2 = PowNode(_clone(z), 2.0)
            one_minus_z2 = _sub(ConstNode(1.0), z2)
            add_feature(one_minus_z2, _dimless(units_spec), f"1-({z_desc})^2", z_cost + 2)

            if allow_sqrt1m:
                sqrt1m = PowNode(_clone(one_minus_z2), 0.5)
                add_feature(sqrt1m, _dimless(units_spec), f"sqrt(1-({z_desc})^2)", z_cost + 3, domain="sqrt1m")
                if allow_rational:
                    inv_sqrt1m = PowNode(_clone(one_minus_z2), -0.5)
                    add_feature(inv_sqrt1m, _dimless(units_spec), f"1/sqrt(1-({z_desc})^2)", z_cost + 3, domain="sqrt1m")

            if allow_logs:
                add_feature(LogNode(_clone(z)), _dimless(units_spec), f"log({z_desc})", z_cost + 2, domain="positive")

    out.sort(key=lambda f: (int(f.cost), len(f.desc), f.desc))
    return out[:max_features]


_REJECT = object()


def _build_z_pool(
    shared_vars: Tuple[int, ...],
    *,
    shared_inputs: Optional[Sequence[Node]],
    units_spec: Any,
) -> List[Tuple[Node, Any, str, int]]:
    out: List[Tuple[Node, Any, str, int]] = []
    seen = set()

    def add(expr: Node, desc: str, cost: int, *, require_valid_dim: bool = False) -> None:
        dim = _expr_dim(expr, units_spec)
        if require_valid_dim and units_spec is not None and dim is None:
            return
        key = ast_key(expr)
        if key in seen:
            return
        seen.add(key)
        out.append((_clone(expr), dim, desc, int(cost)))

    if shared_inputs:
        for k, expr in enumerate(shared_inputs):
            add(_clone(expr), f"arg{k}", 0)

    for i in shared_vars:
        add(Var(i), f"x{i}", 0)

    for a, i in enumerate(shared_vars):
        for j in shared_vars[a + 1 :]:
            xi = Var(i)
            xj = Var(j)
            if _dims_compatible(_expr_dim(xi, units_spec), _expr_dim(xj, units_spec), units_spec):
                add(_sub(_clone(xi), _clone(xj)), f"x{i}-x{j}", 1)
                add(_sub(_clone(xj), _clone(xi)), f"x{j}-x{i}", 1)
            add(MulNode(_clone(xi), _clone(xj)), f"x{i}*x{j}", 1)
            add(MulNode(_clone(xi), PowNode(_clone(xj), -1.0)), f"x{i}/x{j}", 1)
            add(MulNode(_clone(xj), PowNode(_clone(xi), -1.0)), f"x{j}/x{i}", 1)

    def sort_key(item):
        _expr, dim, desc, cost = item
        dim_pri = 0 if _dim_is_known_dimless(dim) else 1
        return (dim_pri, int(cost), len(desc), desc)

    out.sort(key=sort_key)
    return out


def _coefficient_dim(required_dim: Any, phi_dim: Any, *, units_spec: Any, strict_units: bool):
    if required_dim is None:
        return None
    if phi_dim is None:
        return _REJECT if strict_units and units_spec is not None else None
    try:
        return sub_dim(tuple(required_dim), tuple(phi_dim))
    except Exception:
        return _REJECT if strict_units else None


def _make_coeff_atom(coeff_dim: Any, units_spec: Any, index: int, desc: str, *, strict_units: bool):
    tag = f"gauge_h_{index}_{safe_const_token(desc)[:24]}"
    try:
        return make_unit_aware_scalar_atom(
            coeff_dim,
            units_spec,
            base_tag=tag,
            init=1.0,
            strict=bool(strict_units and units_spec is not None),
        )
    except Exception:
        return None


def _expr_dim(expr: Node, units_spec: Any):
    if units_spec is None:
        return None
    try:
        return eval_analytic_expr_dim(expr, units_spec.x_dims)
    except Exception:
        return None


def _dimless(units_spec: Any):
    if units_spec is None:
        return None
    try:
        return units_spec.unit_system.dimless()
    except Exception:
        return None


def _dim_is_known_dimless(dim: Any) -> bool:
    if dim is None:
        return False
    try:
        return bool(is_dimless(dim))
    except Exception:
        return False


def _nonlinear_arg_allowed(dim: Any, units_spec: Any) -> bool:
    if units_spec is None:
        return True
    return _dim_is_known_dimless(dim)


def _scale_dim_safe(dim: Any, value: float):
    if dim is None:
        return None
    try:
        from fractions import Fraction

        return scale_dim(tuple(dim), Fraction.from_float(float(value)).limit_denominator(64))
    except Exception:
        return None


def _dims_compatible(a: Any, b: Any, units_spec: Any) -> bool:
    if units_spec is None:
        return True
    if a is None or b is None:
        return False
    try:
        return tuple(a) == tuple(b)
    except Exception:
        return False


def _clone(expr: Node) -> Node:
    try:
        return clone_ast(expr)
    except Exception:
        return expr


def _is_one(expr: Node) -> bool:
    return isinstance(expr, ConstNode) and float(expr.value) == 1.0


def _sub(a: Node, b: Node) -> Node:
    return AddNode(a, MulNode(ConstNode(-1.0), b))


def describe_transfer_basis(features: Iterable[TransferFeature]) -> List[str]:
    """Return compact descriptions useful for logging/tests."""
    out = []
    for feature in features:
        try:
            expr_s = ast_to_human_readable(feature.expr)
        except Exception:
            expr_s = repr(feature.expr)
        out.append(f"{feature.desc}: {expr_s}")
    return out
