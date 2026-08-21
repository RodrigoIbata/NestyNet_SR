# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Lightweight, parameter-free leaves for native DE/PDE discovery.

These leaves expose the *fitted field* u(x) and its derivatives as AtomNodes
inside an ASTCompositeAdaptor, so that DE residuals can be represented as a
symbolic expression tree.

Design goals
------------
* No trainable parameters (num_parameters()==0).
* Implements the minimal per-leaf interface used by ASTCompositeAdaptor:
    - build_cache
    - jvp / vjp (both return empty contributions)
    - blocks (yields nothing)
    - optional grad / grad_grad for input-derivative queries.
* Reuses a shared cache so u/du/d2u leaves don't recompute expensive
  surrogate derivatives repeatedly for the same x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _as_BY(t: torch.Tensor) -> torch.Tensor:
    """Ensure a (B,Ny) tensor.

    Accepts:
      - (B,) -> (B,1)
      - (B,Ny) -> (B,Ny)
    """
    if t.ndim == 1:
        return t.unsqueeze(1)
    if t.ndim == 2:
        return t
    raise ValueError(f"Expected 1D or 2D tensor, got shape={tuple(t.shape)}")


def _as_BYN(t: torch.Tensor) -> torch.Tensor:
    """Ensure a (B,Ny,Nx) tensor.

    Accepts:
      - (B,Nx) -> (B,1,Nx)
      - (B,Ny,Nx) -> (B,Ny,Nx)
    """
    if t.ndim == 2:
        return t.unsqueeze(1)
    if t.ndim == 3:
        return t
    raise ValueError(f"Expected 2D or 3D tensor, got shape={tuple(t.shape)}")


def _as_BYNN(t: torch.Tensor) -> torch.Tensor:
    """Ensure a (B,Ny,Nx,Nx) tensor.

    Accepts:
      - (B,Nx,Nx) -> (B,1,Nx,Nx)
      - (B,Ny,Nx,Nx) -> (B,Ny,Nx,Nx)
    """
    if t.ndim == 3:
        return t.unsqueeze(1)
    if t.ndim == 4:
        return t
    raise ValueError(f"Expected 3D or 4D tensor, got shape={tuple(t.shape)}")


@dataclass
class UFeatureCache:
    """Cache u(x), ∇u(x), and H_u(x) for a frozen surrogate model."""

    surrogate: any
    _x_id: int = -1
    u: Optional[torch.Tensor] = None  # (B,Ny)
    g: Optional[torch.Tensor] = None  # (B,Ny,Nx)
    H: Optional[torch.Tensor] = None  # (B,Ny,Nx,Nx)

    def reset(self):
        self._x_id = -1
        self.u = None
        self.g = None
        self.H = None

    def ensure(self, x: torch.Tensor, *, need_grad: bool = False, need_hess: bool = False):
        xid = id(x)
        if xid != self._x_id:
            self.reset()
            self._x_id = xid

        # Always compute u.
        if self.u is None:
            with torch.no_grad():
                self.u = _as_BY(self.surrogate(x).detach())

        Ny = int(self.u.shape[1])

        if need_grad and self.g is None:
            with torch.no_grad():
                g_raw = self.surrogate.grad(x).detach()
                g = _as_BYN(g_raw)
                # For multi-output surrogates, gradients must carry an output axis.
                if Ny > 1 and g_raw.ndim == 2:
                    raise ValueError(
                        "UFeatureCache.ensure: surrogate.grad(x) returned shape "
                        f"{tuple(g_raw.shape)} but u(x) has Ny={Ny} outputs; "
                        "expected (B,Ny,Nx)."
                    )
                self.g = g

        if need_hess and self.H is None:
            with torch.no_grad():
                H_raw = self.surrogate.grad_grad(x).detach()
                H = _as_BYNN(H_raw)
                if Ny > 1 and H_raw.ndim == 3:
                    raise ValueError(
                        "UFeatureCache.ensure: surrogate.grad_grad(x) returned shape "
                        f"{tuple(H_raw.shape)} but u(x) has Ny={Ny} outputs; "
                        "expected (B,Ny,Nx,Nx)."
                    )
                self.H = H


class UFeatureLeaf(torch.nn.Module):
    """A parameter-free AST leaf exposing u / du / d2u from a frozen surrogate."""

    def __init__(
        self,
        cache: UFeatureCache,
        kind: str,
        *,
        out_idx: int = 0,
        axis: int = 0,
        axis0: int = 0,
        axis1: int = 0,
    ):
        super().__init__()
        self.cache = cache
        self.kind = str(kind).lower()
        self.out_idx = int(out_idx)
        self.axis = int(axis)
        self.axis0 = int(axis0)
        self.axis1 = int(axis1)
        # Give the leaf a device/dtype anchor so ASTCompositeAdaptor can infer device.
        self.register_buffer("_dummy", torch.empty(0))

    def _out_slice(self) -> slice:
        # Always return a slice so we preserve an explicit (B,1,...) output axis.
        i = int(self.out_idx)
        return slice(i, i + 1)

    # ──────────────────────────────────────────────────────────────
    # Minimal per-leaf interface (parameterless)
    # ──────────────────────────────────────────────────────────────

    def num_parameters(self):
        return 0

    def build_cache(self, data, **_):
        x, y = data[0], data[1]
        f = self.forward(x).detach()
        B = int(f.shape[0])
        O = int(f.shape[1]) if f.ndim >= 2 else 1
        jac = f.new_zeros(B, O, 0)
        return {"x": x, "y": y, "f": f, "jac": jac}

    def jvp(self, cache, v, out_dim=None):
        f = cache.get("f", None)
        if f is None:
            # Fall back to x to infer batch size
            x = cache["x"]
            f = x.new_zeros(x.shape[0], 1)
        return f.new_zeros(f.shape)

    def vjp(self, cache, v, out_dim=None):
        # Return shape (O, P_leaf) = (O, 0)
        if v.ndim == 1:
            O = 1
        else:
            O = int(v.shape[1])
        return v.new_zeros(O, 0)

    def jacobian(self, cache):
        return cache["jac"]

    def blocks(self, *_, **__):
        if False:
            yield None

    # ──────────────────────────────────────────────────────────────
    # Forward and optional input-derivative interface
    # ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kind
        if k in ("u", "field", "state"):
            self.cache.ensure(x, need_grad=False, need_hess=False)
            u = self.cache.u[:, self._out_slice()]
            return u
        if k in ("du", "d1u", "grad_u"):
            self.cache.ensure(x, need_grad=True, need_hess=False)
            g = self.cache.g[:, self._out_slice(), :]
            # Return (B,1) for the selected component derivative.
            return g[:, :, self.axis]
        if k in ("d2u", "ddu", "hess_u"):
            self.cache.ensure(x, need_grad=False, need_hess=True)
            H = self.cache.H[:, self._out_slice(), :, :]
            return H[:, :, self.axis0, self.axis1]
        raise ValueError(f"Unknown UFeatureLeaf kind={self.kind!r}")

    def grad(self, cache_or_x, out_dim=None):
        # Provide input gradients where possible.
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        k = self.kind
        if k in ("u", "field", "state"):
            self.cache.ensure(x, need_grad=True, need_hess=False)
            return self.cache.g[:, self._out_slice(), :]
        if k in ("du", "d1u", "grad_u"):
            # ∇(u_xi) is the i-th row of the Hessian.
            self.cache.ensure(x, need_grad=False, need_hess=True)
            H = self.cache.H[:, self._out_slice(), :, :]
            return H[:, :, self.axis, :]
        raise NotImplementedError(f"grad() not implemented for kind={self.kind!r}")

    def grad_grad(self, cache_or_x, out_dim=None):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        k = self.kind
        if k in ("u", "field", "state"):
            self.cache.ensure(x, need_grad=False, need_hess=True)
            return self.cache.H[:, self._out_slice(), :, :]
        raise NotImplementedError(f"grad_grad() not implemented for kind={self.kind!r}")
