# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Registry of y -> φ(y) transforms and their inverses.

Conventions
-----------
For a given transform name:

  * np_op(y):      numpy implementation of φ(y) (used by PhysDataset)
  * torch_op(y):   torch implementation of φ(y) (used by separability math)
  * torch_inv(t):  inverse mapping y = φ^{-1}(t) applied to network output

This file is intended to be the single source of truth for transform
semantics (including numerically-safe inverses).
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class YTransform:
    np_op: object  # callable | None
    torch_inv: object  # inverse used on network output
    name: str
    torch_op: object = None  # forward op in torch-space (for separability)
    d1: object = None  # first derivative wrt y
    d2: object = None  # second derivative wrt y
    # Optional helpers (used by quickscan/outer-selection and validation)
    check_fn: object = None  # torch: y -> bool mask of valid samples
    requires_dimless: bool = False  # units gate
    # Exact exponent p when phi(c*y) == c**p * phi(y), else None.
    homogeneity_power: float | None = None


_STACK_PREFIX = "stack__"


# ----------------------------------------------------------------------------
# Numerically-safe inverse wrappers (named functions for pickling/stable display)
# ----------------------------------------------------------------------------

_EPS_POS = 1.0e-12
_EPS_TRIG = 1.0e-6


def _torch_sqrt_clamped(t, eps: float = 0.0):
    return torch.sqrt(torch.clamp(t, min=float(eps)))


def _torch_log_clamped(t, eps: float = _EPS_POS):
    return torch.log(torch.clamp(t, min=float(eps)))


def _torch_arcsin_clamped(t, eps: float = _EPS_TRIG):
    return torch.asin(torch.clamp(t, -1.0 + float(eps), 1.0 - float(eps)))


def _torch_arccos_clamped(t, eps: float = _EPS_TRIG):
    return torch.acos(torch.clamp(t, -1.0 + float(eps), 1.0 - float(eps)))


# Keep familiar names so string/sympy wrappers stay stable.
_torch_sqrt_clamped.__name__ = "sqrt"
_torch_log_clamped.__name__ = "log"
_torch_arcsin_clamped.__name__ = "arcsin"
_torch_arccos_clamped.__name__ = "arccos"


# ----------------------------------------------------------------------------
# Named functions for logneg/expneg (lambdas can't be pickled by torch.save)
# ----------------------------------------------------------------------------


def _np_logneg(y):
    """numpy: log(-y)"""
    return np.log(-y)


def _torch_logneg(x):
    """torch forward: log(-x)"""
    return torch.log(-x)


def _torch_logneg_inv(t):
    """torch inverse for logneg: -exp(t)"""
    return -torch.exp(t)


def _logneg_d1(x):
    """d/dy[log(-y)] = 1/y"""
    return 1 / x


def _logneg_d2(x):
    """d2/dy2[log(-y)] = -1/y^2"""
    return -1 / x**2


def _np_expneg(y):
    """numpy: exp(-y)"""
    return np.exp(-y)


def _torch_expneg(x):
    """torch forward: exp(-x)"""
    return torch.exp(-x)


def _torch_expneg_inv(t):
    """torch inverse for expneg: -log(t)"""
    return -torch.log(torch.clamp(t, min=_EPS_POS))


def _expneg_d1(x):
    """d/dy[exp(-y)] = -exp(-y)"""
    return -torch.exp(-x)


def _expneg_d2(x):
    """d2/dy2[exp(-y)] = exp(-y)"""
    return torch.exp(-x)


# Set nice display names for printing
_np_logneg.__name__ = "logneg"
_torch_logneg.__name__ = "logneg"
_torch_logneg_inv.__name__ = "negexp"
_np_expneg.__name__ = "expneg"
_torch_expneg.__name__ = "expneg"
_torch_expneg_inv.__name__ = "neglog"


# ----------------------------------------------------------------------------
# Named functions for sqrt1p
#   sqrt1p: t = y^2 - 1,  y = sqrt(1 + t)
# Useful when y = sqrt(1 + f(x)) where f(x) is multiplicative
# ----------------------------------------------------------------------------


def _np_sqrt1p(y):
    """numpy: y^2 - 1"""
    return y**2 - 1


def _torch_sqrt1p(x):
    """torch forward: x^2 - 1"""
    return x**2 - 1


def _torch_sqrt1p_inv(t):
    """torch inverse for sqrt1p: sqrt(1 + t) (clamped for numerical safety)"""
    return torch.sqrt(torch.clamp(1 + t, min=_EPS_POS))


def _sqrt1p_d1(x):
    """d/dy[y^2 - 1] = 2y"""
    return 2 * x


def _sqrt1p_d2(x):
    """d2/dy2[y^2 - 1] = 2"""
    return 2 * torch.ones_like(x)


_np_sqrt1p.__name__ = "sqrt1p"
_torch_sqrt1p.__name__ = "sqrt1p"
_torch_sqrt1p_inv.__name__ = "sqrt1p_inv"


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


def _build_registry():
    T = []

    T.append(
        YTransform(
            np_op=None,
            torch_inv=torch.nn.Identity(),
            name="identity",
            torch_op=lambda x: x,
            d1=lambda x: torch.ones_like(x),
            d2=lambda x: torch.zeros_like(x),
            check_fn=lambda y: torch.ones_like(y, dtype=torch.bool),
            requires_dimless=False,
            homogeneity_power=1.0,
        )
    )

    T.append(
        YTransform(
            np_op=np.reciprocal,
            torch_inv=torch.reciprocal,
            name="reciprocal",
            torch_op=torch.reciprocal,
            d1=lambda x: -1 / x**2,
            d2=lambda x: 2 / x**3,
            check_fn=lambda y: torch.isfinite(y) & (y.abs() > _EPS_POS),
            requires_dimless=False,
            homogeneity_power=-1.0,
        )
    )

    T.append(
        YTransform(
            np_op=np.sqrt,
            torch_inv=torch.square,
            name="sqrt",
            torch_op=torch.sqrt,
            d1=lambda x: 1 / (2 * torch.sqrt(x)),
            d2=lambda x: -1 / (4 * x**1.5),
            check_fn=lambda y: torch.isfinite(y) & (y >= -_EPS_POS),
            requires_dimless=False,
            homogeneity_power=0.5,
        )
    )

    # sqrt1p: t = y^2 - 1, y = sqrt(1 + t)
    T.append(
        YTransform(
            np_op=_np_sqrt1p,
            torch_inv=_torch_sqrt1p_inv,
            name="sqrt1p",
            torch_op=_torch_sqrt1p,
            d1=_sqrt1p_d1,
            d2=_sqrt1p_d2,
            check_fn=lambda y: torch.isfinite(y) & (y >= -_EPS_POS),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.square,
            torch_inv=_torch_sqrt_clamped,
            name="square",
            torch_op=torch.square,
            d1=lambda x: 2 * x,
            d2=lambda x: 2 * torch.ones_like(x),
            check_fn=lambda y: torch.isfinite(y),
            requires_dimless=False,
            homogeneity_power=2.0,
        )
    )

    T.append(
        YTransform(
            np_op=np.sin,
            torch_inv=_torch_arcsin_clamped,
            name="sin",
            torch_op=torch.sin,
            d1=lambda x: torch.cos(x),
            d2=lambda x: -torch.sin(x),
            check_fn=lambda y: torch.isfinite(y),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.arcsin,
            torch_inv=torch.sin,
            name="arcsin",
            torch_op=torch.arcsin,
            d1=lambda x: 1 / torch.sqrt(1 - x**2),
            d2=lambda x: x / ((1 - x**2) ** 1.5),
            check_fn=lambda y: torch.isfinite(y) & (y.abs() <= 1.0 + 1e-6),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.cos,
            torch_inv=_torch_arccos_clamped,
            name="cos",
            torch_op=torch.cos,
            d1=lambda x: -torch.sin(x),
            d2=lambda x: -torch.cos(x),
            check_fn=lambda y: torch.isfinite(y),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.arccos,
            torch_inv=torch.cos,
            name="arccos",
            torch_op=torch.arccos,
            d1=lambda x: -1 / torch.sqrt(1 - x**2),
            d2=lambda x: -x / ((1 - x**2) ** 1.5),
            check_fn=lambda y: torch.isfinite(y) & (y.abs() <= 1.0 + 1e-6),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.tan,
            torch_inv=torch.arctan,
            name="tan",
            torch_op=torch.tan,
            d1=lambda x: 1 / torch.cos(x) ** 2,
            d2=lambda x: 2 * torch.sin(x) / torch.cos(x) ** 3,
            check_fn=lambda y: torch.isfinite(y),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.arctan,
            torch_inv=torch.tan,
            name="arctan",
            torch_op=torch.arctan,
            d1=lambda x: 1 / (1 + x**2),
            d2=lambda x: -2 * x / (1 + x**2) ** 2,
            check_fn=lambda y: torch.isfinite(y),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.exp,
            torch_inv=_torch_log_clamped,
            name="exp",
            torch_op=torch.exp,
            d1=lambda x: torch.exp(x),
            d2=lambda x: torch.exp(x),
            check_fn=lambda y: torch.isfinite(y),
            requires_dimless=True,
        )
    )

    T.append(
        YTransform(
            np_op=np.log,
            torch_inv=torch.exp,
            name="log",
            torch_op=torch.log,
            d1=lambda x: 1 / x,
            d2=lambda x: -1 / x**2,
            check_fn=lambda y: torch.isfinite(y) & (y > _EPS_POS),
            requires_dimless=True,
        )
    )

    # logneg: t = log(-y), y = -exp(t)
    T.append(
        YTransform(
            np_op=_np_logneg,
            torch_inv=_torch_logneg_inv,
            name="logneg",
            torch_op=_torch_logneg,
            d1=_logneg_d1,
            d2=_logneg_d2,
            check_fn=lambda y: torch.isfinite(y) & (y < -_EPS_POS),
            requires_dimless=True,
        )
    )

    # expneg: t = exp(-y), y = -log(t)
    T.append(
        YTransform(
            np_op=_np_expneg,
            torch_inv=_torch_expneg_inv,
            name="expneg",
            torch_op=_torch_expneg,
            d1=_expneg_d1,
            d2=_expneg_d2,
            check_fn=lambda y: torch.isfinite(y),
            requires_dimless=True,
        )
    )

    return T


def get_y_transform_registry() -> list[YTransform]:
    """Return the full default transform registry."""
    return _build_registry()


def encode_y_stack_name(names) -> str:
    nn = [str(n) for n in names if str(n)]
    if not nn:
        return "identity"
    if len(nn) == 1:
        return nn[0]
    return _STACK_PREFIX + "__".join(nn)


def decode_y_stack_name(name: str):
    s = str(name or "").strip()
    if not s:
        return ["identity"]
    if s.startswith(_STACK_PREFIX):
        rem = s[len(_STACK_PREFIX) :]
        parts = [p for p in rem.split("__") if p]
        return parts if parts else ["identity"]
    return [s]


class _StackedNPOp:
    """Picklable numpy y-op composition over a stack of transform names."""

    def __init__(self, names):
        self.names = tuple(str(n) for n in names)
        self.__name__ = encode_y_stack_name(self.names)

    def __call__(self, y):
        lookup = {t.name: t for t in _build_registry()}
        out = y
        for n in self.names:
            yt = lookup[n]
            if yt.np_op is not None:
                out = yt.np_op(out)
        return out

    def __repr__(self):
        return f"_StackedNPOp(names={self.names})"


class _StackedTorchInv:
    """Picklable torch inverse composition for stacked y-ops."""

    def __init__(self, names):
        self.names = tuple(str(n) for n in names)
        self.__name__ = "inv__" + encode_y_stack_name(self.names)

    def __call__(self, t):
        lookup = {yt.name: yt for yt in _build_registry()}
        out = t
        for n in reversed(self.names):
            yt = lookup[n]
            if yt.torch_inv is not None:
                out = yt.torch_inv(out)
        return out

    def __repr__(self):
        return f"_StackedTorchInv(names={self.names})"


class _StackedTorchOp:
    def __init__(self, names):
        self.names = tuple(str(n) for n in names)
        self.__name__ = "op__" + encode_y_stack_name(self.names)

    def __call__(self, y):
        lookup = {yt.name: yt for yt in _build_registry()}
        out = y
        for n in self.names:
            out = lookup[n].torch_op(out)
        return out


class _StackedD1:
    def __init__(self, names):
        self.names = tuple(str(n) for n in names)
        self.__name__ = "d1__" + encode_y_stack_name(self.names)

    def __call__(self, y):
        lookup = {yt.name: yt for yt in _build_registry()}
        cur = y
        d1_tot = torch.ones_like(y)
        for n in self.names:
            yt = lookup[n]
            d1_cur = yt.d1(cur)
            d1_tot = d1_cur * d1_tot
            cur = yt.torch_op(cur)
        return d1_tot


class _StackedD2:
    def __init__(self, names):
        self.names = tuple(str(n) for n in names)
        self.__name__ = "d2__" + encode_y_stack_name(self.names)

    def __call__(self, y):
        lookup = {yt.name: yt for yt in _build_registry()}
        cur = y
        d1_tot = torch.ones_like(y)
        d2_tot = torch.zeros_like(y)
        for n in self.names:
            yt = lookup[n]
            d1_cur = yt.d1(cur)
            d2_cur = yt.d2(cur)
            d2_tot = d2_cur * (d1_tot**2) + d1_cur * d2_tot
            d1_tot = d1_cur * d1_tot
            cur = yt.torch_op(cur)
        return d2_tot


class _StackedCheckFn:
    def __init__(self, names):
        self.names = tuple(str(n) for n in names)
        self.__name__ = "check__" + encode_y_stack_name(self.names)

    def __call__(self, y):
        lookup = {yt.name: yt for yt in _build_registry()}
        mask = torch.isfinite(y)
        cur = y
        for n in self.names:
            yt = lookup[n]
            chk = getattr(yt, "check_fn", None)
            if chk is not None:
                mask = mask & chk(cur)
            cur_safe = torch.where(mask, cur, torch.zeros_like(cur))
            cur = yt.torch_op(cur_safe)
            mask = mask & torch.isfinite(cur)
        return mask


def build_default_y_transforms(names=None):
    """Used in run_SR.py to decide which transforms to loop over.

    names=None -> default ["identity"].
    """
    registry = _build_registry()
    if names is None:
        names = ["identity"]
    lookup = {t.name: t for t in registry}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"Unknown y-transform names: {missing}")
    return [lookup[n] for n in names]


def compose_y_stack_ops(stack_names, transforms=None):
    """Compose y-transform stack into (np_op, torch_inv, stack_name)."""
    if transforms is None:
        transforms = _build_registry()
    lookup = {t.name: t for t in transforms}
    names = [str(n) for n in stack_names if str(n)]
    if not names:
        return None, torch.nn.Identity(), "identity"
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"Unknown y-transform names in stack: {missing}")
    if len(names) == 1:
        yt = lookup[names[0]]
        return yt.np_op, yt.torch_inv, yt.name
    return _StackedNPOp(names), _StackedTorchInv(names), encode_y_stack_name(names)


def resolve_y_transform_name(y_name: str, transforms=None):
    """Resolve single or stacked y-transform name into callable ops."""
    if transforms is None:
        transforms = _build_registry()
    lookup = {t.name: t for t in transforms}
    if y_name in lookup:
        yt = lookup[y_name]
        return yt.np_op, yt.torch_inv, yt.name
    stack = decode_y_stack_name(y_name)
    return compose_y_stack_ops(stack, transforms=transforms)


def get_separability_y_ops(names=None):
    """Used in separability_math.check_separability_ops.

    names=None -> use all transforms from the registry.
    names=[...] -> use only that subset, in the given order.
    """
    registry = _build_registry()
    if names is None:
        selected = registry
    else:
        lookup = {t.name: t for t in registry}
        selected = []
        missing = []
        for n in names:
            if n in lookup:
                selected.append(lookup[n])
                continue
            stack = decode_y_stack_name(str(n))
            if len(stack) > 1 and all(s in lookup for s in stack):
                np_op, torch_inv, stack_name = compose_y_stack_ops(stack, transforms=registry)
                selected.append(
                    YTransform(
                        np_op=np_op,
                        torch_inv=torch_inv,
                        name=stack_name,
                        torch_op=_StackedTorchOp(stack),
                        d1=_StackedD1(stack),
                        d2=_StackedD2(stack),
                        check_fn=_StackedCheckFn(stack),
                        requires_dimless=False,
                    )
                )
            else:
                missing.append(n)
        if missing:
            raise ValueError(f"Unknown y-transform names for separability: {missing}")

    y_ops = [t.torch_op for t in selected]
    dy_ops = [t.d1 for t in selected]
    d2y_ops = [t.d2 for t in selected]
    return selected, y_ops, dy_ops, d2y_ops


def precision_for_transform(y_op, y_med, y_mad, base_precision):
    """Map the base second-derivative precision to the value actually used
    inside the separability checks for a given y-transform.

    Inside the separability math (check_separability / check_separability_ops)
    the model output f(x) is always normalised by the MAD of the *current*
    sub-model output:

        f̃(x) = f(x) / MAD(f)

    The derivative tests run on f̃, so they are already dimensionless and
    automatically adapt to the scale of φ(y). That means the natural thing
    is to keep a single dimensionless tolerance, independent of y_op.

    For now we simply return base_precision.
    """
    return base_precision
