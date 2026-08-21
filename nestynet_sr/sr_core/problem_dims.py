# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Canonical benchmark dimension metadata and shared dim-algebra helpers.

This module is the common source-of-truth layer for benchmark-family metadata.
Solver-facing adapters are intentionally left separate:

* ``sr_de`` consumes :class:`nestynet_sr.sr_core.units.UnitsSpec`
* factorized symbolic search-style table search consumes ``var_dims`` / ``y_dims``

The goal here is to share the benchmark representation and the low-level
dimension algebra, not to force every solver to use the same runtime object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Tuple

DimVec = Tuple[float, ...]


def _as_dim(dim: Sequence[float], *, where: str) -> DimVec:
    out = tuple(float(v) for v in dim)
    if len(out) == 0:
        raise ValueError(f"{where}: expected non-empty dimension vector")
    return out


def _same_rank(d1: Sequence[float], d2: Sequence[float], *, where: str) -> None:
    if len(d1) != len(d2):
        raise ValueError(f"{where}: dimension vector lengths do not match ({len(d1)} != {len(d2)})")


def _payload_number(v: float) -> int | float:
    fv = float(v)
    iv = int(round(fv))
    if abs(fv - float(iv)) <= 1.0e-12:
        return iv
    return fv


def _payload_dim(dim: Sequence[float]) -> list[int | float]:
    return [_payload_number(v) for v in dim]


def dim_add(d1: Sequence[float], d2: Sequence[float]) -> DimVec:
    _same_rank(d1, d2, where="dim_add")
    return tuple(float(a) + float(b) for a, b in zip(d1, d2))


def dim_sub(d1: Sequence[float], d2: Sequence[float]) -> DimVec:
    _same_rank(d1, d2, where="dim_sub")
    return tuple(float(a) - float(b) for a, b in zip(d1, d2))


def dim_scale(dim: Sequence[float], scale: float) -> DimVec:
    return tuple(float(v) * float(scale) for v in dim)


def derivative_dim(base_dim: Sequence[float], axis_dim: Sequence[float], *, order: int = 1) -> DimVec:
    order_i = int(order)
    if order_i < 0:
        raise ValueError(f"derivative_dim: expected non-negative order, got {order_i}")
    return dim_sub(base_dim, dim_scale(axis_dim, float(order_i)))


def product_dim(dims: Iterable[Sequence[float]]) -> DimVec:
    dims_list = [tuple(float(v) for v in dim) for dim in dims]
    if not dims_list:
        raise ValueError("product_dim: expected at least one dimension vector")
    out = dims_list[0]
    for dim in dims_list[1:]:
        out = dim_add(out, dim)
    return out


def is_dimless(dim: Sequence[float], *, atol: float = 1.0e-12) -> bool:
    return all(abs(float(v)) <= float(atol) for v in dim)


def require_dimensionless(dim: Sequence[float], *, where: str) -> None:
    if not is_dimless(dim):
        raise ValueError(f"{where}: expected dimensionless argument, got {tuple(float(v) for v in dim)!r}")


def dim_eq(d1: Sequence[float], d2: Sequence[float], *, tol: float = 1.0e-12) -> bool:
    return len(tuple(d1)) == len(tuple(d2)) and all(
        abs(float(a) - float(b)) <= float(tol)
        for a, b in zip(d1, d2)
    )


def dimless_dim(rank: int) -> DimVec:
    rank_i = int(rank)
    if rank_i <= 0:
        raise ValueError(f"dimless_dim: expected positive rank, got {rank_i}")
    return tuple(0.0 for _ in range(rank_i))


@dataclass(frozen=True)
class CanonicalProblemDims:
    """Canonical benchmark-family dimension metadata.

    Parameters
    ----------
    basis
        Base-dimension labels, e.g. ``("T", "M")``.
    axis_dims
        Dimension vector for each independent axis.
    component_dims
        Dimension vector for each discovered field/component.
    constant_dims
        Declared physical dimensions for named constants/parameters.
    complex_pairs
        Optional real/imag paired components. Each pair must share units.
    target_dims
        Optional per-component discovery target dimensions when these are fixed
        by the benchmark definition rather than inferred later from the anchor.
    """

    basis: Tuple[str, ...]
    axis_dims: Tuple[DimVec, ...]
    component_dims: Tuple[DimVec, ...]
    constant_dims: dict[str, DimVec] = field(default_factory=dict)
    complex_pairs: Tuple[Tuple[int, int], ...] = ()
    target_dims: Tuple[DimVec, ...] | None = None

    def __post_init__(self):
        basis = tuple(str(b) for b in self.basis)
        if len(basis) == 0:
            raise ValueError("CanonicalProblemDims.basis must be non-empty")
        if any(str(b).strip() == "" for b in basis):
            raise ValueError("CanonicalProblemDims.basis entries must be non-empty strings")
        object.__setattr__(self, "basis", basis)

        rank = len(basis)

        axis_dims = tuple(_as_dim(dim, where=f"axis_dims[{i}]") for i, dim in enumerate(self.axis_dims))
        if len(axis_dims) == 0:
            raise ValueError("CanonicalProblemDims.axis_dims must be non-empty")
        for i, dim in enumerate(axis_dims):
            if len(dim) != rank:
                raise ValueError(
                    f"axis_dims[{i}] has rank {len(dim)} but basis has rank {rank}"
                )
        object.__setattr__(self, "axis_dims", axis_dims)

        component_dims = tuple(
            _as_dim(dim, where=f"component_dims[{i}]") for i, dim in enumerate(self.component_dims)
        )
        if len(component_dims) == 0:
            raise ValueError("CanonicalProblemDims.component_dims must be non-empty")
        for i, dim in enumerate(component_dims):
            if len(dim) != rank:
                raise ValueError(
                    f"component_dims[{i}] has rank {len(dim)} but basis has rank {rank}"
                )
        object.__setattr__(self, "component_dims", component_dims)

        const_dims = {
            str(name): _as_dim(dim, where=f"constant_dims[{name!r}]")
            for name, dim in dict(self.constant_dims).items()
        }
        for name, dim in const_dims.items():
            if len(dim) != rank:
                raise ValueError(
                    f"constant_dims[{name!r}] has rank {len(dim)} but basis has rank {rank}"
                )
        object.__setattr__(self, "constant_dims", const_dims)

        pairs = tuple((int(a), int(b)) for a, b in tuple(self.complex_pairs))
        for pair in pairs:
            a, b = pair
            if a < 0 or a >= len(component_dims) or b < 0 or b >= len(component_dims):
                raise ValueError(
                    f"complex pair {pair!r} out of range for {len(component_dims)} components"
                )
            if tuple(component_dims[a]) != tuple(component_dims[b]):
                raise ValueError(
                    f"complex pair {pair!r} must share units, got "
                    f"{component_dims[a]!r} vs {component_dims[b]!r}"
                )
        object.__setattr__(self, "complex_pairs", pairs)

        tgt = getattr(self, "target_dims", None)
        if tgt is not None:
            tgt_norm = tuple(_as_dim(dim, where=f"target_dims[{i}]") for i, dim in enumerate(tgt))
            if len(tgt_norm) != len(component_dims):
                raise ValueError(
                    f"target_dims has len {len(tgt_norm)} but component_dims has len {len(component_dims)}"
                )
            for i, dim in enumerate(tgt_norm):
                if len(dim) != rank:
                    raise ValueError(
                        f"target_dims[{i}] has rank {len(dim)} but basis has rank {rank}"
                    )
            object.__setattr__(self, "target_dims", tgt_norm)

    @property
    def rank(self) -> int:
        return len(self.basis)

    @property
    def n_axes(self) -> int:
        return len(self.axis_dims)

    @property
    def n_components(self) -> int:
        return len(self.component_dims)

    def axis_dim(self, axis_idx: int) -> DimVec:
        return tuple(self.axis_dims[int(axis_idx)])

    def component_dim(self, component_idx: int) -> DimVec:
        return tuple(self.component_dims[int(component_idx)])

    def target_dim(self, component_idx: int) -> DimVec | None:
        if self.target_dims is None:
            return None
        return tuple(self.target_dims[int(component_idx)])

    @classmethod
    def scalar(
        cls,
        *,
        basis: Sequence[str],
        x_dim: Sequence[float],
        u_dim: Sequence[float],
        constant_dims: Mapping[str, Sequence[float]] | None = None,
        target_dim: Sequence[float] | None = None,
    ) -> "CanonicalProblemDims":
        return cls(
            basis=tuple(str(v) for v in basis),
            axis_dims=(tuple(float(v) for v in x_dim),),
            component_dims=(tuple(float(v) for v in u_dim),),
            constant_dims={
                str(name): tuple(float(v) for v in dim)
                for name, dim in dict(constant_dims or {}).items()
            },
            target_dims=None if target_dim is None else (tuple(float(v) for v in target_dim),),
        )


def canonical_target_dim_for_order(
    problem_dims: CanonicalProblemDims,
    *,
    order: int,
    x_axis: int = 0,
    component_idx: int = 0,
) -> DimVec:
    """Return the target dimension for one discovered equation/component.

    If the benchmark metadata pins explicit target dimensions, those win.
    Otherwise this uses the standard DE anchor rule:

    * order 0: ``[u]``
    * order 1: ``[u] - [x_axis]``
    * order 2: ``[u] - 2[x_axis]``
    """

    comp_idx = int(component_idx)
    tgt = problem_dims.target_dim(comp_idx)
    if tgt is not None:
        return tuple(tgt)

    order_i = int(order)
    if order_i < 0:
        raise ValueError(f"canonical_target_dim_for_order: expected non-negative order, got {order_i}")
    if order_i == 0:
        return problem_dims.component_dim(comp_idx)
    return derivative_dim(
        problem_dims.component_dim(comp_idx),
        problem_dims.axis_dim(int(x_axis)),
        order=order_i,
    )


def canonical_component_target_dims(
    problem_dims: CanonicalProblemDims | None,
    *,
    anchor_order: int,
    anchor_axis: int = 0,
    n_components: int | None = None,
) -> list[DimVec] | None:
    """Return target dims for the first ``n_components`` discovered equations."""
    if problem_dims is None:
        return None
    n_take = problem_dims.n_components if n_components is None else min(int(n_components), problem_dims.n_components)
    return [
        canonical_target_dim_for_order(
            problem_dims,
            order=int(anchor_order),
            x_axis=int(anchor_axis),
            component_idx=i,
        )
        for i in range(n_take)
    ]


def default_complex_pairs(n_components: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(0, max(0, int(n_components) - 1), 2))


def complex_pairs_for_problem_dims(
    problem_dims: CanonicalProblemDims | None,
    *,
    n_components: int,
) -> tuple[tuple[int, int], ...]:
    """Return the declared complex pairs, or the default adjacent pairing when dims are absent."""
    if problem_dims is None:
        return default_complex_pairs(int(n_components))
    return tuple(problem_dims.complex_pairs)


def canonical_constant_dims(
    problem_dims: CanonicalProblemDims | None,
    constant_names: Sequence[str],
    *,
    default_dimless: bool = True,
) -> list[tuple[str, DimVec]]:
    """Return named constant dimensions in a stable caller-specified order."""
    if problem_dims is None:
        return []
    dim0 = dimless_dim(problem_dims.rank)
    out: list[tuple[str, DimVec]] = []
    for raw_name in constant_names:
        name = str(raw_name)
        dim = problem_dims.constant_dims.get(name, None)
        if dim is None:
            if not bool(default_dimless):
                raise KeyError(f"Constant {name!r} has no declared dimension")
            dim = dim0
        out.append((name, tuple(dim)))
    return out


def parameterized_trig_specs(
    axis_dim: Sequence[float] | None,
    *,
    constant_values: Mapping[str, float] | None = None,
    constant_dims: Mapping[str, Sequence[float]] | None = None,
    x_label: str = "x",
) -> list[tuple[str | None, float, str, str]]:
    """Return legal bare/parameterized sin/cos carriers for one axis dimension."""
    specs: list[tuple[str | None, float, str, str]] = []
    if axis_dim is None:
        specs.append((None, 1.0, "sin", f"sin({x_label})"))
        specs.append((None, 1.0, "cos", f"cos({x_label})"))
        return specs

    axis_dim_t = tuple(float(v) for v in axis_dim)
    if is_dimless(axis_dim_t):
        specs.append((None, 1.0, "sin", f"sin({x_label})"))
        specs.append((None, 1.0, "cos", f"cos({x_label})"))

    const_vals = dict(constant_values or {})
    const_dims = dict(constant_dims or {})
    inv_axis = dim_scale(axis_dim_t, -1.0)
    for raw_name, raw_val in const_vals.items():
        name = str(raw_name)
        cdim = const_dims.get(name, None)
        if cdim is None or not dim_eq(tuple(cdim), inv_axis):
            continue
        scale = float(raw_val)
        specs.append((name, scale, "sin", f"sin({name}*{x_label})"))
        specs.append((name, scale, "cos", f"cos({name}*{x_label})"))
    return specs


def canonical_to_factorized_search_dims(
    problem_dims: CanonicalProblemDims,
    *,
    order: int,
    x_axis: int = 0,
    component_idx: int = 0,
    include_x: bool = True,
    include_u: bool = True,
    include_du: bool = True,
    constant_names: Sequence[str] = (),
    constant_dims_override: Mapping[str, Sequence[float] | None] | None = None,
    default_constant_dimless: bool = False,
) -> tuple[list[DimVec], DimVec]:
    """Build factorized symbolic search-style ``(var_dims, y_dims)`` from canonical metadata."""

    order_i = int(order)
    comp_idx = int(component_idx)
    x_axis_i = int(x_axis)
    dim0 = dimless_dim(problem_dims.rank)
    const_override = dict(constant_dims_override or {})

    var_dims: list[DimVec] = []
    if bool(include_x):
        var_dims.append(problem_dims.axis_dim(x_axis_i))
    if bool(include_u):
        var_dims.append(problem_dims.component_dim(comp_idx))
    if order_i == 2 and bool(include_du):
        var_dims.append(
            derivative_dim(
                problem_dims.component_dim(comp_idx),
                problem_dims.axis_dim(x_axis_i),
                order=1,
            )
        )

    for raw_name in constant_names:
        name = str(raw_name)
        dim = const_override.get(name, problem_dims.constant_dims.get(name, None))
        if dim is None:
            if not bool(default_constant_dimless):
                raise KeyError(
                    f"Constant {name!r} has no declared dimension in canonical metadata"
                )
            dim = dim0
        var_dims.append(tuple(float(v) for v in dim))

    y_dims = canonical_target_dim_for_order(
        problem_dims,
        order=order_i,
        x_axis=x_axis_i,
        component_idx=comp_idx,
    )
    return var_dims, y_dims


def units_spec_from_dim_vectors(
    *,
    basis: Sequence[str],
    x_dims: Sequence[Sequence[float]],
    y_dim: Sequence[float],
    output_dims: Sequence[Sequence[float]] | None = None,
    y_transform_name: str = "identity",
    atom_kind_fixed: Mapping[str, Sequence[float]] | None = None,
    atom_tag_fixed: Mapping[str, Sequence[float]] | None = None,
    free_const_dims: Mapping[str, Sequence[float]] | None = None,
    free_const_scope: Mapping[str, str] | None = None,
    fixed_const_dims: Mapping[str, Sequence[float]] | None = None,
    fixed_const_values: Mapping[str, float] | None = None,
    fixed_const_mode: str = "strict",
    policy: str = "free_const_only",
    nn_semantics: str = "unknown",
):
    """Build a :class:`UnitsSpec` from explicit dimension vectors."""

    from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec

    basis_t = tuple(str(b) for b in basis)
    us = UnitSystem(base=basis_t)

    def _dim_map(raw: Mapping[str, Sequence[float]] | None) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for name, dim in dict(raw or {}).items():
            out[str(name)] = us.dim(dim)
        return out

    return UnitsSpec(
        unit_system=us,
        x_dims=tuple(us.dim(dim) for dim in x_dims),
        y_dim=us.dim(y_dim),
        output_dims=None if output_dims is None else tuple(us.dim(dim) for dim in output_dims),
        y_transform_name=str(y_transform_name),
        atom_kind_fixed=_dim_map(atom_kind_fixed),
        atom_tag_fixed=_dim_map(atom_tag_fixed),
        free_const_dims=_dim_map(free_const_dims),
        free_const_scope=dict(free_const_scope or {}),
        fixed_const_dims=_dim_map(fixed_const_dims),
        fixed_const_values={str(name): float(val) for name, val in dict(fixed_const_values or {}).items()},
        fixed_const_mode=str(fixed_const_mode),
        policy=str(policy),
        nn_semantics=str(nn_semantics),
    )


def canonical_to_units_spec(
    problem_dims: CanonicalProblemDims,
    *,
    y_component: int = 0,
    output_component_idxs: Sequence[int] | None = None,
    y_transform_name: str = "identity",
    free_constant_names: Sequence[str] = (),
    free_const_scope: Mapping[str, str] | None = None,
    fixed_constant_values: Mapping[str, float] | None = None,
    fixed_const_mode: str = "strict",
    policy: str = "free_const_only",
    nn_semantics: str = "unknown",
):
    """Build a :class:`UnitsSpec` from canonical benchmark metadata."""

    y_comp = int(y_component)
    out_idxs = (
        tuple(range(problem_dims.n_components))
        if output_component_idxs is None
        else tuple(int(i) for i in output_component_idxs)
    )
    free_names = tuple(str(name) for name in free_constant_names)
    fixed_vals = {str(name): float(val) for name, val in dict(fixed_constant_values or {}).items()}

    overlap = sorted(set(free_names).intersection(fixed_vals))
    if overlap:
        raise ValueError(
            f"canonical_to_units_spec: constants cannot be both free and fixed: {overlap}"
        )

    free_dims: dict[str, DimVec] = {}
    for name in free_names:
        if name not in problem_dims.constant_dims:
            raise KeyError(f"canonical_to_units_spec: unknown free constant {name!r}")
        free_dims[name] = tuple(problem_dims.constant_dims[name])

    fixed_dims: dict[str, DimVec] = {}
    for name in fixed_vals:
        if name not in problem_dims.constant_dims:
            raise KeyError(f"canonical_to_units_spec: unknown fixed constant {name!r}")
        fixed_dims[name] = tuple(problem_dims.constant_dims[name])

    return units_spec_from_dim_vectors(
        basis=problem_dims.basis,
        x_dims=problem_dims.axis_dims,
        y_dim=problem_dims.component_dim(y_comp),
        output_dims=tuple(problem_dims.component_dim(i) for i in out_idxs),
        y_transform_name=y_transform_name,
        free_const_dims=free_dims,
        free_const_scope={
            str(name): str(scope)
            for name, scope in dict(free_const_scope or {}).items()
            if str(name) in free_dims
        },
        fixed_const_dims=fixed_dims,
        fixed_const_values=fixed_vals,
        fixed_const_mode=fixed_const_mode,
        policy=policy,
        nn_semantics=nn_semantics,
    )


def canonical_scalar_dims_payload(
    problem_dims: CanonicalProblemDims,
    *,
    x_axis: int = 0,
    component_idx: int = 0,
) -> dict[str, list[float] | list[str]]:
    """Serialize one scalar view of canonical metadata for DE benchmark payloads."""
    return {
        "basis": [str(v) for v in problem_dims.basis],
        "x": _payload_dim(problem_dims.axis_dim(int(x_axis))),
        "u": _payload_dim(problem_dims.component_dim(int(component_idx))),
    }


def canonical_constant_payload(
    problem_dims: CanonicalProblemDims | None,
    values: Mapping[str, float],
    *,
    names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Serialize named constants into benchmark/oracle payload rows.

    When ``problem_dims`` is provided, declared constant dimensions are emitted
    under ``"dim"``. Undeclared constants keep the old behavior and are left
    without a dimension entry.
    """

    order = [str(name) for name in (names if names is not None else tuple(values.keys()))]
    out: list[dict[str, object]] = []
    const_dims = {} if problem_dims is None else dict(problem_dims.constant_dims)
    for name in order:
        if name not in values:
            continue
        row: dict[str, object] = {
            "name": str(name),
            "value": float(values[name]),
        }
        dim = const_dims.get(str(name), None)
        if dim is not None:
            row["dim"] = _payload_dim(dim)
        out.append(row)
    return out


__all__ = [
    "CanonicalProblemDims",
    "DimVec",
    "canonical_constant_payload",
    "canonical_constant_dims",
    "canonical_component_target_dims",
    "canonical_scalar_dims_payload",
    "canonical_target_dim_for_order",
    "canonical_to_factorized_search_dims",
    "canonical_to_units_spec",
    "complex_pairs_for_problem_dims",
    "default_complex_pairs",
    "derivative_dim",
    "dim_eq",
    "dimless_dim",
    "dim_add",
    "dim_scale",
    "dim_sub",
    "is_dimless",
    "parameterized_trig_specs",
    "product_dim",
    "require_dimensionless",
    "units_spec_from_dim_vectors",
]
