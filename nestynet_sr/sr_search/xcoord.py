# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""xcoord.py

First-class x-coordinate transforms.

This module introduces :class:`XCoordSystem`, a small abstraction that treats
"x-transforms" as an explicit coordinate system mapping

    x_raw  ->  x_internal

and provides:
  * Torch + NumPy forward transforms (for dataset wrapping or model preprocessing)
  * SymPy rewrites from internal-coordinates back to raw coordinates
  * Unit-aware dimension propagation and parameter unit declarations
  * Chain-rule helpers for converting derivatives w.r.t. internal coords back
    to derivatives w.r.t. raw coords (for DE/PDE discovery).

The API is intentionally JSON-serialisable so it can be stored in checkpoints
and attached to models (e.g. model._x_transform) without requiring pickling.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
import torch

from ..sr_core.units import Dim, UnitError, UnitSystem, is_dimless, scale_dim, sub_dim

# ──────────────────────────────────────────────────────────────
# Primitive transform registry
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class XPrimBackend:
    kind: str
    torch_apply: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
    torch_d1: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
    torch_d2: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
    np_apply: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
    sympy_apply: Callable[[sp.Expr, Dict[str, Any], Callable[[str, float], sp.Expr]], sp.Expr]
    requires_dimless_input: bool = False
    forces_dimless_output: bool = False


_PRIMS: Dict[str, XPrimBackend] = {}


def register_x_primitive(backend: XPrimBackend):
    k = str(backend.kind).lower().strip()
    if not k:
        raise ValueError("Primitive kind must be non-empty")
    _PRIMS[k] = backend


def get_x_primitive(kind: str) -> XPrimBackend:
    k = str(kind).lower().strip()
    if k not in _PRIMS:
        raise ValueError(f"Unknown x-transform primitive kind: {kind!r}. Available: {sorted(_PRIMS)}")
    return _PRIMS[k]


def available_x_primitives() -> Tuple[str, ...]:
    return tuple(sorted(_PRIMS))


def _build_default_primitives():
    def _as_t(x, v):
        return x.new_tensor(float(v))

    # identity
    register_x_primitive(
        XPrimBackend(
            kind="identity",
            torch_apply=lambda x, step: x,
            torch_d1=lambda x, step: torch.ones_like(x),
            torch_d2=lambda x, step: torch.zeros_like(x),
            np_apply=lambda x, step: x,
            sympy_apply=lambda x, step, const: x,
        )
    )

    # shift: x -> x - shift
    register_x_primitive(
        XPrimBackend(
            kind="shift",
            torch_apply=lambda x, step: x - _as_t(x, step.get("shift", step.get("value", 0.0))),
            torch_d1=lambda x, step: torch.ones_like(x),
            torch_d2=lambda x, step: torch.zeros_like(x),
            np_apply=lambda x, step: x - float(step.get("shift", step.get("value", 0.0))),
            sympy_apply=lambda x, step, const: x - const(step.get("name", "shift"), float(step.get("shift", step.get("value", 0.0)))),
        )
    )

    # scale: x -> scale * x
    register_x_primitive(
        XPrimBackend(
            kind="scale",
            torch_apply=lambda x, step: _as_t(x, step.get("scale", step.get("value", 1.0))) * x,
            torch_d1=lambda x, step: _as_t(x, step.get("scale", step.get("value", 1.0))).expand_as(x),
            torch_d2=lambda x, step: torch.zeros_like(x),
            np_apply=lambda x, step: float(step.get("scale", step.get("value", 1.0))) * x,
            sympy_apply=lambda x, step, const: const(step.get("name", "scale"), float(step.get("scale", step.get("value", 1.0)))) * x,
        )
    )

    # square: x -> x^2
    register_x_primitive(
        XPrimBackend(
            kind="square",
            torch_apply=lambda x, step: x * x,
            torch_d1=lambda x, step: 2 * x,
            torch_d2=lambda x, step: 2 * torch.ones_like(x),
            np_apply=lambda x, step: x * x,
            sympy_apply=lambda x, step, const: x ** 2,
        )
    )

    # recip: x -> 1/x
    register_x_primitive(
        XPrimBackend(
            kind="recip",
            torch_apply=lambda x, step: torch.reciprocal(x),
            torch_d1=lambda x, step: -torch.reciprocal(x * x),
            torch_d2=lambda x, step: 2 * torch.reciprocal(x * x * x),
            np_apply=lambda x, step: 1.0 / x,
            sympy_apply=lambda x, step, const: 1 / x,
        )
    )
    # Alias
    register_x_primitive(
        XPrimBackend(
            kind="reciprocal",
            torch_apply=lambda x, step: torch.reciprocal(x),
            torch_d1=lambda x, step: -torch.reciprocal(x * x),
            torch_d2=lambda x, step: 2 * torch.reciprocal(x * x * x),
            np_apply=lambda x, step: 1.0 / x,
            sympy_apply=lambda x, step, const: 1 / x,
        )
    )

    # sqrt: x -> sqrt(x)
    register_x_primitive(
        XPrimBackend(
            kind="sqrt",
            torch_apply=lambda x, step: torch.sqrt(x),
            torch_d1=lambda x, step: 0.5 * torch.reciprocal(torch.sqrt(x)),
            torch_d2=lambda x, step: -0.25 * torch.reciprocal(x * torch.sqrt(x)),
            np_apply=lambda x, step: np.sqrt(x),
            sympy_apply=lambda x, step, const: sp.sqrt(x),
        )
    )

    # log: x -> log(x)
    register_x_primitive(
        XPrimBackend(
            kind="log",
            torch_apply=lambda x, step: torch.log(x),
            torch_d1=lambda x, step: torch.reciprocal(x),
            torch_d2=lambda x, step: -torch.reciprocal(x * x),
            np_apply=lambda x, step: np.log(x),
            sympy_apply=lambda x, step, const: sp.log(x),
            requires_dimless_input=True,
            forces_dimless_output=True,
        )
    )

    # sin
    register_x_primitive(
        XPrimBackend(
            kind="sin",
            torch_apply=lambda x, step: torch.sin(x),
            torch_d1=lambda x, step: torch.cos(x),
            torch_d2=lambda x, step: -torch.sin(x),
            np_apply=lambda x, step: np.sin(x),
            sympy_apply=lambda x, step, const: sp.sin(x),
            requires_dimless_input=True,
            forces_dimless_output=True,
        )
    )

    # cos
    register_x_primitive(
        XPrimBackend(
            kind="cos",
            torch_apply=lambda x, step: torch.cos(x),
            torch_d1=lambda x, step: -torch.sin(x),
            torch_d2=lambda x, step: -torch.cos(x),
            np_apply=lambda x, step: np.cos(x),
            sympy_apply=lambda x, step, const: sp.cos(x),
            requires_dimless_input=True,
            forces_dimless_output=True,
        )
    )


_build_default_primitives()


# ──────────────────────────────────────────────────────────────
# XCoordSystem
# ──────────────────────────────────────────────────────────────


def _default_var_names(N: int) -> Tuple[str, ...]:
    return tuple(f"x{i}" for i in range(int(N)))


def _norm_kind(k: Any) -> str:
    return str(k).lower().strip()


def _is_trig_kind(k: str) -> bool:
    return _norm_kind(k) in ("sin", "cos")


@dataclass(frozen=True)
class XCoordSystem:
    """A shared coordinate system mapping raw x -> internal x."""

    Nx_raw: int
    axis_map: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    raw_var_names: Tuple[str, ...] = ()
    internal_var_names: Tuple[str, ...] = ()
    name_prefix: str = "xtr"

    # ──────────────────────────────────────────────────────────
    # Construction / serialization
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def identity(Nx_raw: int, *, raw_var_names: Optional[Sequence[str]] = None) -> "XCoordSystem":
        return XCoordSystem.from_map(None, Nx_raw=Nx_raw, raw_var_names=raw_var_names)

    @staticmethod
    def from_map(
        x_transform_map: Optional[Mapping[int, Any]],
        Nx_raw: int,
        *,
        raw_var_names: Optional[Sequence[str]] = None,
        internal_var_names: Optional[Sequence[str]] = None,
        name_prefix: str = "xtr",
    ) -> "XCoordSystem":
        Nx = int(Nx_raw)
        if Nx <= 0:
            raise ValueError(f"Nx_raw must be positive; got {Nx_raw}")

        internal_names = tuple(internal_var_names) if internal_var_names is not None else _default_var_names(Nx)
        raw_names = tuple(raw_var_names) if raw_var_names is not None else tuple(internal_names)
        if len(internal_names) != Nx or len(raw_names) != Nx:
            raise ValueError(f"Expected {Nx} x-names; got internal={len(internal_names)}, raw={len(raw_names)}")

        axis_map: Dict[int, Dict[str, Any]] = {}
        if x_transform_map:
            for ax_k, spec in dict(x_transform_map).items():
                try:
                    axis = int(ax_k)
                except Exception:
                    continue
                if axis < 0 or axis >= Nx:
                    continue
                canon = _canonicalize_axis_spec(spec, axis=axis, name_prefix=str(name_prefix))
                if canon is not None:
                    axis_map[int(axis)] = canon

        return XCoordSystem(
            Nx_raw=Nx,
            axis_map=axis_map,
            raw_var_names=raw_names,
            internal_var_names=internal_names,
            name_prefix=str(name_prefix),
        )

    def to_map(self) -> Dict[int, Any]:
        return {int(k): dict(v) for k, v in (self.axis_map or {}).items()}

    def is_identity(self) -> bool:
        return not bool(self.axis_map)

    # ──────────────────────────────────────────────────────────
    # Data-space transforms
    # ──────────────────────────────────────────────────────────

    def apply_torch(self, x_raw: torch.Tensor) -> torch.Tensor:
        if self.is_identity():
            return x_raw
        if not torch.is_tensor(x_raw):
            x_raw = torch.as_tensor(x_raw)
        x2 = x_raw.clone()
        for axis, spec in (self.axis_map or {}).items():
            mode = str(spec.get("mode", "replace")).lower().strip()
            out_axis = int(spec.get("out_axis", axis))
            if mode != "replace" or out_axis != int(axis):
                raise NotImplementedError("Only mode='replace' with out_axis==axis is currently supported")
            v = x2[..., axis]
            for step in (spec.get("pipeline", []) or []):
                kind = _norm_kind(step.get("kind", ""))
                if kind == "":
                    continue
                backend = get_x_primitive(kind)
                v = backend.torch_apply(v, step)
            x2[..., axis] = v
        return x2

    def apply_np(self, x_raw: np.ndarray) -> np.ndarray:
        if self.is_identity():
            return x_raw
        x = np.asarray(x_raw)
        x2 = np.array(x, copy=True)
        for axis, spec in (self.axis_map or {}).items():
            mode = str(spec.get("mode", "replace")).lower().strip()
            out_axis = int(spec.get("out_axis", axis))
            if mode != "replace" or out_axis != int(axis):
                raise NotImplementedError("Only mode='replace' with out_axis==axis is currently supported")
            v = x2[..., axis]
            for step in (spec.get("pipeline", []) or []):
                kind = _norm_kind(step.get("kind", ""))
                if kind == "":
                    continue
                backend = get_x_primitive(kind)
                v = backend.np_apply(v, step)
            x2[..., axis] = v
        return x2

    def make_x_op(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x: self.apply_torch(x)

    # ──────────────────────────────────────────────────────────
    # SymPy view
    # ──────────────────────────────────────────────────────────

    def internal_symbols(self) -> Tuple[sp.Symbol, ...]:
        names = self.internal_var_names or _default_var_names(self.Nx_raw)
        return tuple(sp.Symbol(str(n)) for n in names)

    def raw_symbols_unique(self) -> Tuple[sp.Symbol, ...]:
        # Make raw symbols distinct from internal symbols, even if they share display names.
        names = self.raw_var_names or _default_var_names(self.Nx_raw)
        return tuple(sp.Symbol(f"{str(n)}__raw") for n in names)

    def raw_symbols_display(self) -> Tuple[sp.Symbol, ...]:
        names = self.raw_var_names or _default_var_names(self.Nx_raw)
        return tuple(sp.Symbol(str(n)) for n in names)

    def sympy_internal_to_raw_subs(
        self,
        *,
        const_mode: str = "number",
    ) -> Dict[sp.Symbol, sp.Expr]:
        const_mode = str(const_mode).lower().strip()
        if const_mode not in ("number", "symbol"):
            raise ValueError("const_mode must be 'number' or 'symbol'")

        x_int = self.internal_symbols()
        x_raw_u = self.raw_symbols_unique()

        def _const(name: str, value: float) -> sp.Expr:
            if const_mode == "symbol":
                return sp.Symbol(str(name))
            return sp.Float(float(value))

        out: Dict[sp.Symbol, sp.Expr] = {}
        for i in range(self.Nx_raw):
            expr = x_raw_u[i]
            spec = (self.axis_map or {}).get(i, None)
            if spec is not None:
                pipe = list(spec.get("pipeline", []) or [])
                for step in pipe:
                    kind = _norm_kind(step.get("kind", ""))
                    if kind == "":
                        continue
                    backend = get_x_primitive(kind)
                    expr = backend.sympy_apply(expr, step, _const)
            out[x_int[i]] = expr
        return out

    def sympy_rewrite_internal_expr_to_raw(self, expr: sp.Expr, *, const_mode: str = "number") -> sp.Expr:
        subs = self.sympy_internal_to_raw_subs(const_mode=const_mode)
        e = sp.sympify(expr).xreplace(subs)

        # Rename unique raw symbols to display names.
        x_raw_u = self.raw_symbols_unique()
        x_raw_d = self.raw_symbols_display()
        ren = {a: b for a, b in zip(x_raw_u, x_raw_d)}
        return e.xreplace(ren)

    # ──────────────────────────────────────────────────────────
    # Units integration
    # ──────────────────────────────────────────────────────────

    def _infer_axis_units(
        self, us: UnitSystem, raw_dim: Dim, axis: int
    ) -> Tuple[Dim, Dict[str, Dim]]:
        """Return (dim_out, param_dims) for one axis."""
        spec = (self.axis_map or {}).get(axis, None)
        if spec is None:
            return raw_dim, {}

        dimless = us.dimless()
        dim_cur = raw_dim

        pipe = list(spec.get("pipeline", []) or [])
        if not pipe:
            return raw_dim, {}

        # Lookahead: where do we need dimensionless input?
        needs_dimless_after_step = [False] * len(pipe)
        future_need = False
        for k in range(len(pipe) - 1, -1, -1):
            kind = _norm_kind(pipe[k].get("kind", ""))
            backend = get_x_primitive(kind) if kind else None
            if backend is not None and backend.requires_dimless_input:
                future_need = True
            needs_dimless_after_step[k] = future_need

        param_dims: Dict[str, Dim] = {}
        for k, step in enumerate(pipe):
            kind = _norm_kind(step.get("kind", ""))
            if kind in ("", "identity"):
                continue

            if kind == "shift":
                nm = str(step.get("name", "shift"))
                param_dims[nm] = dim_cur
                # dim unchanged
                continue

            if kind == "scale":
                nm = str(step.get("name", "scale"))
                # If any downstream primitive requires dimless input, choose scale units to make dimless.
                if needs_dimless_after_step[k] and not is_dimless(dim_cur):
                    param_dims[nm] = sub_dim(dimless, dim_cur)  # -dim_cur
                    dim_cur = dimless
                else:
                    param_dims[nm] = dimless
                continue

            if kind == "square":
                dim_cur = scale_dim(dim_cur, Fraction(2))
                continue

            if kind in ("recip", "reciprocal"):
                dim_cur = scale_dim(dim_cur, Fraction(-1))
                continue

            if kind == "sqrt":
                dim_cur = scale_dim(dim_cur, Fraction(1, 2))
                continue

            backend = get_x_primitive(kind)
            if backend.requires_dimless_input and not is_dimless(dim_cur):
                raise UnitError(
                    f"x-transform axis {axis}: primitive {kind} requires dimensionless input, "
                    f"but got dim={us.format_dim(dim_cur)}"
                )
            if backend.forces_dimless_output:
                dim_cur = dimless
        return dim_cur, param_dims

    def internal_x_dims(self, us: UnitSystem, raw_x_dims: Sequence[Dim]) -> Tuple[Dim, ...]:
        if len(raw_x_dims) != int(self.Nx_raw):
            raise ValueError(f"Expected {self.Nx_raw} raw x dims; got {len(raw_x_dims)}")
        out = []
        for i in range(self.Nx_raw):
            d_out, _ = self._infer_axis_units(us, raw_x_dims[i], i)
            out.append(d_out)
        return tuple(out)

    def transform_free_const_dims(self, us: UnitSystem, raw_x_dims: Sequence[Dim]) -> Dict[str, Dim]:
        if len(raw_x_dims) != int(self.Nx_raw):
            raise ValueError(f"Expected {self.Nx_raw} raw x dims; got {len(raw_x_dims)}")
        out: Dict[str, Dim] = {}
        for i in range(self.Nx_raw):
            _, pd = self._infer_axis_units(us, raw_x_dims[i], i)
            for k, v in pd.items():
                if k in out and out[k] != v:
                    raise UnitError(
                        f"x-transform free-const name collision with incompatible dims: {k} "
                        f"({us.format_dim(out[k])} vs {us.format_dim(v)})"
                    )
                out[k] = v
        return out

    def validate_units(
        self,
        us: UnitSystem,
        raw_x_dims: Sequence[Dim],
        *,
        strict: bool = True,
    ) -> Tuple[bool, str]:
        if not strict:
            return True, ""
        try:
            _ = self.internal_x_dims(us, raw_x_dims)
            _ = self.transform_free_const_dims(us, raw_x_dims)
        except Exception as e:
            return False, str(e)
        return True, ""

    # ──────────────────────────────────────────────────────────
    # Derivative semantics (raw <-> internal)
    # ──────────────────────────────────────────────────────────

    def is_separable(self) -> bool:
        # Current implementation only supports per-axis pipelines.
        return True

    def dzdx_diag(self, x_raw: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x_raw):
            x_raw = torch.as_tensor(x_raw)
        d1_all = torch.ones_like(x_raw)
        if self.is_identity():
            return d1_all
        for axis, spec in (self.axis_map or {}).items():
            v = x_raw[..., axis]
            d1 = torch.ones_like(v)
            d2 = torch.zeros_like(v)
            for step in (spec.get("pipeline", []) or []):
                kind = _norm_kind(step.get("kind", ""))
                if kind in ("", "identity"):
                    continue
                backend = get_x_primitive(kind)
                f1 = backend.torch_d1(v, step)
                f2 = backend.torch_d2(v, step)
                d2 = f2 * (d1 * d1) + f1 * d2
                d1 = f1 * d1
                v = backend.torch_apply(v, step)
            d1_all[..., axis] = d1
        return d1_all

    def d2zdx2_diag(self, x_raw: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x_raw):
            x_raw = torch.as_tensor(x_raw)
        d2_all = torch.zeros_like(x_raw)
        if self.is_identity():
            return d2_all
        for axis, spec in (self.axis_map or {}).items():
            v = x_raw[..., axis]
            d1 = torch.ones_like(v)
            d2 = torch.zeros_like(v)
            for step in (spec.get("pipeline", []) or []):
                kind = _norm_kind(step.get("kind", ""))
                if kind in ("", "identity"):
                    continue
                backend = get_x_primitive(kind)
                f1 = backend.torch_d1(v, step)
                f2 = backend.torch_d2(v, step)
                d2 = f2 * (d1 * d1) + f1 * d2
                d1 = f1 * d1
                v = backend.torch_apply(v, step)
            d2_all[..., axis] = d2
        return d2_all

    def chain_rule_grad(self, grad_u_wrt_z: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        dzdx = self.dzdx_diag(x_raw)
        g = grad_u_wrt_z
        z = dzdx
        while z.dim() < g.dim():
            z = z.unsqueeze(1)
        return g * z

    def chain_rule_hess(
        self,
        hess_u_wrt_z: torch.Tensor,
        grad_u_wrt_z: torch.Tensor,
        x_raw: torch.Tensor,
    ) -> torch.Tensor:
        dzdx = self.dzdx_diag(x_raw)  # [B, Nx]
        d2z = self.d2zdx2_diag(x_raw)  # [B, Nx]

        H = hess_u_wrt_z
        if H.dim() != 4:
            raise ValueError(f"Expected hess shape [B,1,Nx,Nx], got {tuple(H.shape)}")

        # Broadcast dzdx across the Hessian.
        dz_i = dzdx.unsqueeze(1).unsqueeze(-1)  # [B,1,Nx,1]
        dz_j = dzdx.unsqueeze(1).unsqueeze(-2)  # [B,1,1,Nx]
        Hx = H * dz_i * dz_j

        # Add diagonal correction term: g_z[i] * d2z_i/dx_i^2.
        extra = grad_u_wrt_z
        if extra.dim() == 3:
            extra = extra * d2z.unsqueeze(1)
        elif extra.dim() == 2:
            extra = extra.unsqueeze(1) * d2z.unsqueeze(1)
        else:
            raise ValueError(f"Expected grad shape [B,1,Nx] or [B,Nx], got {tuple(grad_u_wrt_z.shape)}")

        Hx.diagonal(dim1=-2, dim2=-1).add_(extra)
        return Hx


# ──────────────────────────────────────────────────────────────
# Canonicalisation helpers
# ──────────────────────────────────────────────────────────────


def _canonicalize_step(step: Any, *, axis: int, idx: int, name_prefix: str, pipe: Sequence[Any]) -> Dict[str, Any]:
    if isinstance(step, str):
        step = {"kind": step}
    if not isinstance(step, Mapping):
        raise ValueError(f"Invalid x-transform step for axis {axis}: {step!r}")
    d = dict(step)
    kind = _norm_kind(d.get("kind", ""))
    if kind == "":
        return {"kind": "identity"}

    # Normalise common aliases.
    if kind in ("sinlin", "sin_linear"):
        kind = "sin"
    if kind in ("coslin", "cos_linear"):
        kind = "cos"
    d["kind"] = kind

    if kind == "shift":
        if "shift" not in d:
            if "offset" in d:
                d["shift"] = d.pop("offset")
            elif "value" in d:
                d["shift"] = d["value"]
            else:
                d["shift"] = 0.0
        d["shift"] = float(d.get("shift", 0.0))
        if "name" not in d or not str(d.get("name") or "").strip():
            d["name"] = f"shift_x{axis}" if idx == 0 else f"shift{idx}_x{axis}"
        return d

    if kind == "scale":
        if "scale" not in d:
            if "omega" in d:
                d["scale"] = d.pop("omega")
            elif "value" in d:
                d["scale"] = d["value"]
            else:
                d["scale"] = 1.0
        d["scale"] = float(d.get("scale", 1.0))
        if "name" not in d or not str(d.get("name") or "").strip():
            # Heuristic naming: if a trig primitive appears later in the pipeline, call it omega.
            def _kind_of(s):
                if isinstance(s, Mapping):
                    return _norm_kind(s.get("kind", ""))
                if isinstance(s, str):
                    return _norm_kind(s)
                return ""

            trig_later = any(_is_trig_kind(_kind_of(s)) for s in pipe[idx + 1 :])
            base = "omega" if trig_later else "scale"
            d["name"] = f"{base}_x{axis}" if idx == 0 else f"{base}{idx}_x{axis}"
        return d

    # Everything else: ensure it's registered.
    _ = get_x_primitive(kind)
    return d


def _canonicalize_axis_spec(spec: Any, *, axis: int, name_prefix: str) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None

    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray)):
        pipe_raw = list(spec)
        pipe = [_canonicalize_step(s, axis=axis, idx=i, name_prefix=name_prefix, pipe=pipe_raw) for i, s in enumerate(pipe_raw)]
        return {"pipeline": pipe, "mode": "replace", "out_axis": int(axis), "meta": {"source": "list"}}

    if isinstance(spec, Mapping):
        mode = str(spec.get("mode", "replace")).lower().strip() or "replace"
        out_axis = int(spec.get("out_axis", axis))
        pipe_raw = list(spec.get("pipeline", []) or [])
        pipe = [_canonicalize_step(s, axis=axis, idx=i, name_prefix=name_prefix, pipe=pipe_raw) for i, s in enumerate(pipe_raw)]
        meta = dict(spec.get("meta", {}) or {})
        if not meta:
            meta = {"source": "canonical"}
        return {"pipeline": pipe, "mode": mode, "out_axis": out_axis, "meta": meta}

    raise ValueError(f"Unrecognised x-transform spec for axis {axis}: {spec!r}")
