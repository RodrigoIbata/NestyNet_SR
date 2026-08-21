# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared wrapper policy for compound proposals.

Wrappers are proposal evidence only.  A flexible NN can fit many contorted
coordinates, so successful training on ``op(z)`` must not be interpreted as
proof that ``op`` belongs in the final formula.  Consumers still need visible
simplification and their normal validation/acceptance checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from nestynet_sr.sr_core.bridges import (
    AddNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    clone_ast,
)
from nestynet_sr.sr_search.compound_proposals.units import DimTuple, is_dimless_dim, scale_dim


@dataclass(frozen=True)
class WrappedExpression:
    """A wrapped expression and its dimension metadata."""

    name: str
    expr: Node
    dim: Optional[DimTuple]
    domain: str = "real"
    requires_dimless_arg: bool = False
    reason: str = ""


_ALIASES = {
    "identity": "z",
    "q": "z",
    "raw": "z",
    "z_inv": "inv_z",
    "1/z": "inv_z",
    "inv": "inv_z",
    "inv_q": "inv_z",
    "reciprocal": "inv_z",
    "sq": "z2",
    "square": "z2",
    "q2": "z2",
    "inv_q2": "inv_z2",
    "1/z^2": "inv_z2",
    "sqrt": "sqrt_z",
    "sqrt_q": "sqrt_z",
    "inv_sqrt": "inv_sqrt_z",
    "inv_sqrt_q": "inv_sqrt_z",
    "1/sqrt": "inv_sqrt_z",
    "1/(1+z)": "inv1p",
    "rat_inv_zp1": "inv1p",
    "1/(1-z)": "inv1m",
    "rat_inv_1mz": "inv1m",
    "sqrt(1+z)": "sqrt1p",
    "sqrt1p": "sqrt1p",
    "sqrt(1-z)": "sqrt1m",
    "sqrt1m": "sqrt1m",
    "1/sqrt(1+z)": "inv_sqrt1p",
    "inv_sqrt1p": "inv_sqrt1p",
    "1/sqrt(1-z)": "inv_sqrt1m",
    "inv_sqrt1m": "inv_sqrt1m",
    "z/(1+z)": "z_over_1p",
    "rat_z_over_zp1": "z_over_1p",
    "z/(1-z)": "z_over_1m",
    "log_z": "log",
    "exp_z": "exp",
    "sin_z": "sin",
    "cos_z": "cos",
    "sin^2": "sin2",
    "sin_sq": "sin2",
    "cos^2": "cos2",
    "cos_sq": "cos2",
    "1-cos": "one_minus_cos",
    "one_minus_cos_z": "one_minus_cos",
    "sin(z/2)^2": "sin_half_sq",
    "sin_half2": "sin_half_sq",
    "cos(z/2)^2": "cos_half_sq",
    "cos_half2": "cos_half_sq",
    "1/sin(z/2)^2": "inv_sin_half_sq",
    "inv_sin_half2": "inv_sin_half_sq",
    "1/sin(z/2)^4": "inv_sin_half_4",
    "inv_sin_half4": "inv_sin_half_4",
    "sinc(z)^2": "sinc_sq",
    "sinc_sq": "sinc_sq",
}


def canonical_wrapper_name(name: str) -> str:
    key = str(name).strip()
    return _ALIASES.get(key, key)


def _sub(a: Node, b: Node) -> Node:
    return AddNode(a, MulNode(ConstNode(-1.0), b))


def _half_arg(expr: Node) -> Node:
    return MulNode(ConstNode(0.5), clone_ast(expr))


def _dimless_required(name: str, dim: Optional[DimTuple]) -> Optional[str]:
    if is_dimless_dim(dim):
        return None
    return f"{name} requires a dimensionless argument"


def apply_compound_wrapper(
    expr: Node,
    dim: Optional[DimTuple],
    wrapper: str,
    *,
    strict_units: bool = True,
) -> Optional[WrappedExpression]:
    """Apply one named wrapper if unit policy permits it.

    Unknown dimensions are permissive because some proposal sources run before
    units are available.  When dimensions are known and ``strict_units`` is
    true, transcendental and ``1 +/- z`` wrappers require dimensionless ``z``.
    """

    name = canonical_wrapper_name(wrapper)
    z = clone_ast(expr)

    if name == "z":
        return WrappedExpression(str(wrapper), z, dim)
    if name == "inv_z":
        return WrappedExpression(str(wrapper), PowNode(z, -1.0), scale_dim(dim, -1.0))
    if name == "z2":
        return WrappedExpression(str(wrapper), PowNode(z, 2.0), scale_dim(dim, 2.0))
    if name == "inv_z2":
        return WrappedExpression(str(wrapper), PowNode(z, -2.0), scale_dim(dim, -2.0))
    if name == "sqrt_z":
        return WrappedExpression(str(wrapper), PowNode(z, 0.5), scale_dim(dim, 0.5), domain="nonnegative")
    if name == "inv_sqrt_z":
        return WrappedExpression(
            str(wrapper),
            PowNode(z, -0.5),
            scale_dim(dim, -0.5),
            domain="positive",
        )

    if name in {
        "inv1p",
        "inv1m",
        "sqrt1p",
        "sqrt1m",
        "inv_sqrt1p",
        "inv_sqrt1m",
        "z_over_1p",
        "z_over_1m",
        "log",
        "exp",
        "sin",
        "cos",
        "sin2",
        "cos2",
        "one_minus_cos",
        "sin_half_sq",
        "cos_half_sq",
        "inv_sin_half_sq",
        "inv_sin_half_4",
        "sinc_sq",
    }:
        if bool(strict_units):
            reason = _dimless_required(name, dim)
            if reason is not None:
                return None
        dimless = scale_dim(dim, 0.0)
        one = ConstNode(1.0)
        if name == "inv1p":
            return WrappedExpression(str(wrapper), PowNode(AddNode(one, z), -1.0), dimless, requires_dimless_arg=True)
        if name == "inv1m":
            return WrappedExpression(str(wrapper), PowNode(_sub(one, z), -1.0), dimless, requires_dimless_arg=True)
        if name == "sqrt1p":
            return WrappedExpression(
                str(wrapper),
                PowNode(AddNode(one, z), 0.5),
                dimless,
                domain="nonnegative",
                requires_dimless_arg=True,
            )
        if name == "sqrt1m":
            return WrappedExpression(
                str(wrapper),
                PowNode(_sub(one, z), 0.5),
                dimless,
                domain="nonnegative",
                requires_dimless_arg=True,
            )
        if name == "inv_sqrt1p":
            return WrappedExpression(
                str(wrapper),
                PowNode(AddNode(one, z), -0.5),
                dimless,
                domain="positive",
                requires_dimless_arg=True,
            )
        if name == "inv_sqrt1m":
            return WrappedExpression(
                str(wrapper),
                PowNode(_sub(one, z), -0.5),
                dimless,
                domain="positive",
                requires_dimless_arg=True,
            )
        if name == "z_over_1p":
            return WrappedExpression(
                str(wrapper),
                MulNode(z, PowNode(AddNode(ConstNode(1.0), clone_ast(expr)), -1.0)),
                dimless,
                requires_dimless_arg=True,
            )
        if name == "z_over_1m":
            return WrappedExpression(
                str(wrapper),
                MulNode(z, PowNode(_sub(ConstNode(1.0), clone_ast(expr)), -1.0)),
                dimless,
                requires_dimless_arg=True,
            )
        if name == "log":
            return WrappedExpression(str(wrapper), LogNode(z), dimless, domain="positive", requires_dimless_arg=True)
        if name == "exp":
            return WrappedExpression(str(wrapper), ExpNode(z), dimless, requires_dimless_arg=True)
        if name == "sin":
            return WrappedExpression(str(wrapper), SinNode(z), dimless, requires_dimless_arg=True)
        if name == "cos":
            return WrappedExpression(str(wrapper), CosNode(z), dimless, requires_dimless_arg=True)
        if name == "sin2":
            return WrappedExpression(str(wrapper), PowNode(SinNode(z), 2.0), dimless, requires_dimless_arg=True)
        if name == "cos2":
            return WrappedExpression(str(wrapper), PowNode(CosNode(z), 2.0), dimless, requires_dimless_arg=True)
        if name == "one_minus_cos":
            return WrappedExpression(str(wrapper), _sub(one, CosNode(z)), dimless, requires_dimless_arg=True)
        if name == "sin_half_sq":
            return WrappedExpression(
                str(wrapper),
                PowNode(SinNode(_half_arg(z)), 2.0),
                dimless,
                requires_dimless_arg=True,
            )
        if name == "cos_half_sq":
            return WrappedExpression(
                str(wrapper),
                PowNode(CosNode(_half_arg(z)), 2.0),
                dimless,
                requires_dimless_arg=True,
            )
        if name == "inv_sin_half_sq":
            return WrappedExpression(
                str(wrapper),
                PowNode(SinNode(_half_arg(z)), -2.0),
                dimless,
                domain="nonzero",
                requires_dimless_arg=True,
            )
        if name == "inv_sin_half_4":
            return WrappedExpression(
                str(wrapper),
                PowNode(SinNode(_half_arg(z)), -4.0),
                dimless,
                domain="nonzero",
                requires_dimless_arg=True,
            )
        if name == "sinc_sq":
            arg = clone_ast(expr)
            sinc = MulNode(SinNode(clone_ast(arg)), PowNode(clone_ast(arg), -1.0))
            return WrappedExpression(
                str(wrapper),
                PowNode(sinc, 2.0),
                dimless,
                domain="nonzero",
                requires_dimless_arg=True,
            )

    raise ValueError(f"unknown compound wrapper {wrapper!r}")


def build_compound_wrappers(
    expr: Node,
    dim: Optional[DimTuple],
    wrappers: Sequence[str],
    *,
    strict_units: bool = True,
) -> list[WrappedExpression]:
    out: list[WrappedExpression] = []
    for wrapper in wrappers:
        wrapped = apply_compound_wrapper(expr, dim, str(wrapper), strict_units=strict_units)
        if wrapped is not None:
            out.append(wrapped)
    return out
