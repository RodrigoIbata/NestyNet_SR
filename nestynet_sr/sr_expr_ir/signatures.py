# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Conservative dimensional/signature helpers for expression IR pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainGuard:
    mode: str = "strict"
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymmetrySignature:
    dim: tuple[float, ...] | None = None
    lie_weights: tuple[float, ...] | None = None
    invariant: bool | None = None
    support: tuple[int, ...] = ()
    known: bool = True
    guard: DomainGuard = DomainGuard()


def _dim(ctx: Any, idx: int) -> tuple[float, ...] | None:
    if isinstance(ctx, dict):
        var_dims = ctx.get("var_dims")
    else:
        var_dims = getattr(ctx, "var_dims", None)
    try:
        return tuple(float(v) for v in var_dims[int(idx)])
    except Exception:
        return None


def _add_dims(a: tuple[float, ...] | None, b: tuple[float, ...] | None, sign: float = 1.0) -> tuple[float, ...] | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return tuple(float(x) + sign * float(y) for x, y in zip(a, b))


def _scale_dim(a: tuple[float, ...] | None, exp: float) -> tuple[float, ...] | None:
    if a is None:
        return None
    return tuple(float(exp) * float(x) for x in a)


def sig_const(ctx: Any = None) -> SymmetrySignature:
    _ = ctx
    return SymmetrySignature(dim=None, invariant=True, support=(), known=True)


def sig_var(idx: int, ctx: Any = None) -> SymmetrySignature:
    return SymmetrySignature(dim=_dim(ctx, int(idx)), invariant=None, support=(int(idx),), known=True)


def sig_add(a: SymmetrySignature, b: SymmetrySignature, ctx: Any = None) -> SymmetrySignature | None:
    _ = ctx
    if not a.known or not b.known:
        return SymmetrySignature(known=False)
    if a.dim is not None and b.dim is not None and tuple(a.dim) != tuple(b.dim):
        return None
    return SymmetrySignature(dim=a.dim if a.dim is not None else b.dim, support=tuple(sorted(set(a.support) | set(b.support))))


def sig_mul(a: SymmetrySignature, b: SymmetrySignature, ctx: Any = None) -> SymmetrySignature | None:
    _ = ctx
    if not a.known or not b.known:
        return SymmetrySignature(known=False)
    return SymmetrySignature(dim=_add_dims(a.dim, b.dim), support=tuple(sorted(set(a.support) | set(b.support))))


def sig_pow(a: SymmetrySignature, exponent: float, ctx: Any = None) -> SymmetrySignature | None:
    _ = ctx
    if not a.known:
        return SymmetrySignature(known=False)
    return SymmetrySignature(dim=_scale_dim(a.dim, float(exponent)), support=tuple(a.support))


def sig_unary(op: str, a: SymmetrySignature, ctx: Any = None) -> SymmetrySignature | None:
    _ = ctx
    if not a.known:
        return SymmetrySignature(known=False)
    op = str(op)
    if op in {"sin", "cos", "exp", "log", "asin", "acos"}:
        if a.dim is not None and any(abs(float(v)) > 1.0e-12 for v in a.dim):
            return None
        return SymmetrySignature(dim=None, support=tuple(a.support))
    if op == "sqrt":
        return SymmetrySignature(dim=_scale_dim(a.dim, 0.5), support=tuple(a.support))
    if op in {"sqr"}:
        return sig_pow(a, 2.0, ctx)
    return SymmetrySignature(dim=a.dim, support=tuple(a.support), known=True)


def collect_invariant_tuple_seeds(context: Any = None, cfg: Any = None) -> list[tuple]:
    """Return generic invariant tuple seeds.

    This first implementation intentionally does not add solution-shaped physics
    motifs.  It only exposes a stable extension point and returns no seeds unless
    future context explicitly provides neutral, data-derived seed nodes.
    """

    _ = context, cfg
    return []


def collect_invariant_core_seeds(context: Any = None, cfg: Any = None) -> list[Any]:
    _ = context, cfg
    return []


__all__ = [
    "DomainGuard",
    "SymmetrySignature",
    "collect_invariant_core_seeds",
    "collect_invariant_tuple_seeds",
    "sig_add",
    "sig_const",
    "sig_mul",
    "sig_pow",
    "sig_unary",
    "sig_var",
]
