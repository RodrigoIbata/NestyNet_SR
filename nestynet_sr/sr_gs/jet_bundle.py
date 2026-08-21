# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Minimal jet-space representation for generalized-symmetry DE work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class JetCoordinateSpec:
    """One coordinate on a finite jet bundle chart."""

    name: str
    kind: str
    component: str | None = None
    multi_index: tuple[int, ...] = ()
    order: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "component": self.component,
            "multi_index": [int(v) for v in self.multi_index],
            "order": int(self.order),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ScalarODEJetInputs:
    """Existing scalar prolongation input shape materialized from a jet table."""

    order: int
    x_axis: int
    x: torch.Tensor
    u: torch.Tensor
    u1: torch.Tensor
    u2: torch.Tensor | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "order": int(self.order),
            "x_axis": int(self.x_axis),
            "x": self.x,
            "u": self.u,
            "u1": self.u1,
            "u2": self.u2,
        }


@dataclass(frozen=True)
class JetSampleTable:
    """Validated finite-jet samples for scalar, vector, ODE, or PDE charts."""

    jet_space: "JetSpaceSpec"
    columns: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        cols = {str(name): _as_col_tensor_preserve(value, name=str(name)) for name, value in self.columns.items()}
        _check_same_rows(cols)
        unknown = sorted(set(cols) - set(self.jet_space.coordinate_names()))
        if unknown:
            raise KeyError(f"unknown jet coordinate(s): {unknown}")
        object.__setattr__(self, "columns", cols)

    @property
    def num_samples(self) -> int:
        if not self.columns:
            return 0
        first = next(iter(self.columns.values()))
        return int(first.shape[0])

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return tuple(self.columns.keys())

    def tensor(self, name: str) -> torch.Tensor:
        key = str(name)
        if key not in self.columns:
            raise KeyError(f"missing jet sample {key!r}")
        return self.columns[key]

    def component_tensor(self, component: str) -> torch.Tensor:
        return self.tensor(str(component))

    def derivative_tensor(self, component: str, multi_index: Sequence[int]) -> torch.Tensor:
        return self.tensor(self.jet_space.derivative_name(component, multi_index))

    def as_matrix(self, names: Sequence[str] | None = None) -> torch.Tensor:
        selected = tuple(self.coordinate_names if names is None else tuple(str(name) for name in names))
        if not selected:
            return torch.empty((self.num_samples, 0), dtype=torch.float64)
        return torch.cat([self.tensor(name) for name in selected], dim=1)

    def to_report(self) -> dict[str, Any]:
        dtypes = sorted({str(value.dtype).replace("torch.", "") for value in self.columns.values()})
        devices = sorted({str(value.device) for value in self.columns.values()})
        return {
            "type": "jet_sample_table",
            "jet_scope": self.jet_space.jet_scope,
            "num_samples": int(self.num_samples),
            "coordinate_names": list(self.coordinate_names),
            "num_coordinates": len(self.coordinate_names),
            "dtypes": dtypes,
            "devices": devices,
        }


@dataclass(frozen=True)
class JetSpaceSpec:
    """Finite-order jet-space chart metadata.

    Phase one supports scalar ODE charts for one independent variable and one
    dependent component.  Vector and PDE charts are represented, but callers
    must explicitly opt into future prolongation support before using them.
    """

    independent: tuple[str, ...]
    dependent: tuple[str, ...]
    max_order: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        independent = _name_tuple(self.independent, "independent")
        dependent = _name_tuple(self.dependent, "dependent")
        max_order = int(self.max_order)
        if not independent:
            raise ValueError("JetSpaceSpec requires at least one independent variable")
        if not dependent:
            raise ValueError("JetSpaceSpec requires at least one dependent component")
        if max_order < 0:
            raise ValueError("JetSpaceSpec max_order must be nonnegative")
        if len(set(independent)) != len(independent):
            raise ValueError("independent variable names must be unique")
        if len(set(dependent)) != len(dependent):
            raise ValueError("dependent component names must be unique")
        overlap = set(independent) & set(dependent)
        if overlap:
            raise ValueError(f"independent/dependent names overlap: {sorted(overlap)}")
        object.__setattr__(self, "independent", independent)
        object.__setattr__(self, "dependent", dependent)
        object.__setattr__(self, "max_order", max_order)

    @property
    def input_dim(self) -> int:
        return len(self.independent)

    @property
    def output_dim(self) -> int:
        return len(self.dependent)

    @property
    def is_scalar_ode_phase_one(self) -> bool:
        return self.input_dim == 1 and self.output_dim == 1 and self.max_order in (1, 2)

    @property
    def is_pde(self) -> bool:
        return self.input_dim > 1

    @property
    def is_vector_system(self) -> bool:
        return self.output_dim > 1

    @property
    def jet_scope(self) -> str:
        if self.input_dim == 1 and self.output_dim == 1:
            return "scalar_ode"
        if self.input_dim == 1:
            return "vector_ode"
        if self.output_dim == 1:
            return "scalar_pde"
        return "vector_pde"

    def unsupported_scope_message(self, feature: str = "prolongation") -> str:
        return (
            f"vector/PDE {feature} is not implemented in phase one; "
            f"got jet_scope={self.jet_scope}, independent={self.independent}, dependent={self.dependent}, max_order={self.max_order}"
        )

    def require_scalar_ode_phase_one(self) -> "JetSpaceSpec":
        if self.input_dim != 1 or self.output_dim != 1:
            raise NotImplementedError(self.unsupported_scope_message("prolongation"))
        if self.max_order not in (1, 2):
            raise NotImplementedError(
                "scalar ODE prolongation currently supports max_order 1 or 2; "
                f"got max_order={self.max_order}"
            )
        return self

    def multi_indices(self, *, include_zero: bool = False, max_order: int | None = None) -> tuple[tuple[int, ...], ...]:
        items: list[tuple[int, ...]] = []
        start = 0 if include_zero else 1
        order_max = self._checked_order(max_order)
        for order in range(start, order_max + 1):
            items.extend(_multi_indices_of_order(self.input_dim, order))
        return tuple(items)

    def _checked_order(self, order: int | None) -> int:
        order_i = int(self.max_order if order is None else order)
        if order_i < 0:
            raise ValueError(f"jet order must be nonnegative; got {order_i}")
        if order_i > int(self.max_order):
            raise ValueError(f"jet order {order_i} exceeds max_order={self.max_order}")
        return order_i

    def derivative_name(self, component: str, multi_index: Sequence[int]) -> str:
        comp = str(component)
        if comp not in self.dependent:
            raise ValueError(f"unknown dependent component {comp!r}")
        mi = tuple(int(v) for v in multi_index)
        if len(mi) != self.input_dim:
            raise ValueError(f"multi-index {mi} has length {len(mi)}; expected {self.input_dim}")
        order = int(sum(mi))
        if order == 0:
            return comp
        if order > int(self.max_order):
            raise ValueError(f"multi-index order {order} exceeds max_order={self.max_order}")
        suffix = "".join(str(axis) * count for axis, count in zip(self.independent, mi))
        return f"{comp}_{suffix}"

    def coordinate_specs(self, *, max_order: int | None = None) -> tuple[JetCoordinateSpec, ...]:
        order_max = self._checked_order(max_order)
        coords: list[JetCoordinateSpec] = []
        coords.extend(
            JetCoordinateSpec(
                name=name,
                kind="independent",
                component=None,
                multi_index=(0,) * self.input_dim,
                order=0,
            )
            for name in self.independent
        )
        for dep in self.dependent:
            coords.append(
                JetCoordinateSpec(
                    name=dep,
                    kind="dependent",
                    component=dep,
                    multi_index=(0,) * self.input_dim,
                    order=0,
                )
            )
            for mi in self.multi_indices(include_zero=False, max_order=order_max):
                coords.append(
                    JetCoordinateSpec(
                        name=self.derivative_name(dep, mi),
                        kind="derivative",
                        component=dep,
                        multi_index=mi,
                        order=int(sum(mi)),
                    )
                )
        return tuple(coords)

    def coordinate_names(self, *, max_order: int | None = None) -> tuple[str, ...]:
        return tuple(coord.name for coord in self.coordinate_specs(max_order=max_order))

    def scalar_ode_coordinate_names(self, *, order: int | None = None) -> tuple[str, ...]:
        self.require_scalar_ode_phase_one()
        order_i = int(self.max_order if order is None else order)
        if order_i not in (1, 2) or order_i > int(self.max_order):
            raise ValueError(f"scalar ODE order must be 1 or 2 and <= max_order; got {order_i}")
        x_name = self.independent[0]
        u_name = self.dependent[0]
        names = [x_name, u_name, self.derivative_name(u_name, (1,))]
        if order_i >= 2:
            names.append(self.derivative_name(u_name, (2,)))
        return tuple(names)

    def materialize_scalar_ode_inputs(
        self,
        samples: Mapping[str, Any],
        *,
        order: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: Any | None = None,
    ) -> ScalarODEJetInputs:
        """Convert named jet samples into existing scalar prolongation inputs."""

        self.require_scalar_ode_phase_one()
        order_i = int(self.max_order if order is None else order)
        if order_i not in (1, 2) or order_i > int(self.max_order):
            raise ValueError(f"scalar ODE order must be 1 or 2 and <= max_order; got {order_i}")
        x_name = self.independent[0]
        u_name = self.dependent[0]
        u1_name = self.derivative_name(u_name, (1,))
        u2_name = self.derivative_name(u_name, (2,)) if order_i >= 2 else None
        x = _as_col_tensor(_require_sample(samples, x_name), name=x_name, dtype=dtype, device=device)
        u = _as_col_tensor(_require_sample(samples, u_name), name=u_name, dtype=dtype, device=device)
        u1 = _as_col_tensor(_require_sample(samples, u1_name), name=u1_name, dtype=dtype, device=device)
        u2 = None
        if u2_name is not None:
            u2 = _as_col_tensor(_require_sample(samples, u2_name), name=u2_name, dtype=dtype, device=device)
        _check_same_rows({"x": x, "u": u, "u1": u1, "u2": u2})
        return ScalarODEJetInputs(order=order_i, x_axis=0, x=x, u=u, u1=u1, u2=u2)

    def materialize_jet_samples(
        self,
        samples: Mapping[str, Any],
        *,
        order: int | None = None,
        names: Sequence[str] | None = None,
        dtype: torch.dtype = torch.float64,
        device: Any | None = None,
    ) -> JetSampleTable:
        """Convert named finite-jet samples into a validated general jet table."""

        order_i = self._checked_order(order)
        selected = tuple(self.coordinate_names(max_order=order_i) if names is None else tuple(str(name) for name in names))
        valid = set(self.coordinate_names())
        unknown = sorted(set(selected) - valid)
        if unknown:
            raise KeyError(f"unknown jet coordinate(s): {unknown}")
        columns = {
            name: _as_col_tensor(_require_sample(samples, name), name=name, dtype=dtype, device=device)
            for name in selected
        }
        _check_same_rows(columns)
        return JetSampleTable(self, columns)

    def to_report(self) -> dict[str, Any]:
        return {
            "type": "jet_space",
            "jet_scope": self.jet_scope,
            "independent": list(self.independent),
            "dependent": list(self.dependent),
            "max_order": int(self.max_order),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "is_pde": bool(self.is_pde),
            "is_vector_system": bool(self.is_vector_system),
            "is_scalar_ode_phase_one": bool(self.is_scalar_ode_phase_one),
            "coordinate_count": len(self.coordinate_specs()),
            "coordinates": [coord.to_report() for coord in self.coordinate_specs()],
            "provenance": dict(self.provenance),
        }


def _multi_indices_of_order(dim: int, order: int) -> tuple[tuple[int, ...], ...]:
    dim_i = int(dim)
    order_i = int(order)
    if dim_i <= 0:
        return ()
    if dim_i == 1:
        return ((order_i,),)
    out: list[tuple[int, ...]] = []
    for head in range(order_i, -1, -1):
        for tail in _multi_indices_of_order(dim_i - 1, order_i - head):
            out.append((head,) + tail)
    return tuple(out)


def _name_tuple(value: Sequence[str] | str, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        items = (value,)
    else:
        items = tuple(value)
    names = tuple(str(v) for v in items)
    if any(not name for name in names):
        raise ValueError(f"{label} names must be nonempty")
    return names


def _require_sample(samples: Mapping[str, Any], name: str) -> Any:
    if name not in samples:
        raise KeyError(f"missing jet sample {name!r}")
    return samples[name]


def _as_col_tensor(value: Any, *, name: str, dtype: torch.dtype, device: Any | None) -> torch.Tensor:
    arr = value.detach() if hasattr(value, "detach") else value
    tensor = torch.as_tensor(arr, dtype=dtype, device=device)
    if tensor.ndim == 0:
        raise ValueError(f"jet sample {name!r} must have at least one row")
    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    elif tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.reshape(-1, 1)
    else:
        raise ValueError(f"jet sample {name!r} must be a vector or single-column matrix; got shape {tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0:
        raise ValueError(f"jet sample {name!r} must have at least one row")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"jet sample {name!r} contains nonfinite values")
    return tensor


def _as_col_tensor_preserve(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        dtype = value.dtype if value.is_floating_point() or value.is_complex() else torch.float64
        return _as_col_tensor(value, name=name, dtype=dtype, device=value.device)
    return _as_col_tensor(value, name=name, dtype=torch.float64, device=None)


def _check_same_rows(values: Mapping[str, torch.Tensor | None]) -> None:
    rows = {name: int(value.shape[0]) for name, value in values.items() if value is not None}
    if len(set(rows.values())) > 1:
        raise ValueError(f"jet samples must share row count; got {rows}")


__all__ = [
    "JetCoordinateSpec",
    "JetSampleTable",
    "JetSpaceSpec",
    "ScalarODEJetInputs",
]
