# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Vector-calculus macro-terms for system DE/PDE discovery.

These utilities build *scalar AST nodes* that represent common vector-calculus
operators applied to named (vector) fields. The intent is to provide a small
set of reusable building blocks that are:

* **More structured than raw partials** (better inductive bias than just DU/D2U)
* **Not hard-coded to Maxwell** (useful for fluid/MHD/elasticity PDEs too)
* Still compatible with the current scalar AST machinery (no vector-typed AST
  nodes are introduced here)

All functions return either:

* a scalar :class:`~nestynet_sr.sr_core.bridges.Node`, or
* a tuple of scalar Nodes (vector components), which you can flatten into a
  term list for :func:`nestynet_sr.sr_de.system_de_search.discover_system_de_from_surrogate`.

Important limitation
--------------------
These macros only differentiate *surrogate output components* (i.e. Field/DField
atoms). They do not symbolically differentiate arbitrary ASTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import (
    Add,
    ConstNode,
    Mul,
    Node,
    VField,
)

# ──────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────


def _neg(a: Node) -> Node:
    return Mul(ConstNode(-1.0), a)


def _sub(a: Node, b: Node) -> Node:
    return Add(a, _neg(b))


def _sum(nodes: Sequence[Node]) -> Node:
    if not nodes:
        raise ValueError("_sum() requires at least one term")
    out = nodes[0]
    for t in nodes[1:]:
        out = Add(out, t)
    return out


def _dedup(nodes: Iterable[Node]) -> List[Node]:
    uniq: Dict[str, Node] = {}
    for n in nodes:
        if n is None:
            continue
        uniq[repr(n)] = n
    return list(uniq.values())


def _resolve_comps(field: VField, comps: Optional[Tuple[object, ...]], dim: int) -> Tuple[object, ...]:
    """Resolve component labels/indices for a field.

    Parameters
    ----------
    field : VField
        Named vector field wrapper.
    comps : tuple | None
        Component identifiers passed by the user. If None, try to infer from
        the registered field spec.
    dim : int
        Desired number of components.
    """

    if comps is not None:
        if len(comps) != dim:
            raise ValueError(f"Expected comps length {dim}, got {len(comps)}")
        return tuple(comps)

    spec = field.spec()
    if int(spec.n_comp) < dim:
        raise ValueError(f"Field {spec.name!r} has n_comp={spec.n_comp}, need at least {dim}")

    # Prefer readable names if available and non-empty.
    cn = tuple(spec.comp_names)
    if len(cn) >= dim and all(str(cn[i]) != "" for i in range(dim)):
        return cn[:dim]

    # Fall back to numeric component indices.
    return tuple(int(i) for i in range(dim))




# ──────────────────────────────────────────────────────────────
# Vector expression helper (Python-level vector algebra)
# ──────────────────────────────────────────────────────────────


def _as_scalar_node(v: object) -> Node:
    """Coerce a Python scalar into a scalar AST Node."""

    if isinstance(v, (int, float)):
        return ConstNode(float(v))
    if isinstance(v, Vec):
        raise TypeError("Expected scalar, got Vec")
    # Assume it's already a scalar AST node (AtomNode/AddNode/...)
    return v  # type: ignore


class Vec(tuple):
    """A lightweight vector expression: tuple-of-Nodes + basic vector algebra.

    This is deliberately **not** a vector-typed AST node. It is a Python-level
    wrapper that keeps the core SR machinery scalar while providing ergonomic
    vector-calculus / vector-algebra composition.

    Compatibility note: Vec is a **tuple subclass**, so existing code that
    expects tuples of components (e.g. ``list(curl(E))``) will still work.
    """

    def __new__(cls, comps, name: str | None = None):
        obj = super().__new__(cls, tuple(comps))
        obj._name = str(name) if name is not None else None
        return obj

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def dim(self) -> int:
        return len(self)

    def label(self) -> str:
        if self._name is not None:
            return self._name
        return f"Vec({', '.join(repr(c) for c in self)})"

    def _as_vec(self, other) -> "Vec":
        if isinstance(other, Vec):
            return other
        if isinstance(other, (tuple, list)):
            return Vec(other)
        raise TypeError(f"Expected Vec/tuple/list, got {type(other).__name__}")

    def __repr__(self):
        return self.label()

    def __neg__(self):
        return Vec((_neg(a) for a in self), name=self._name)

    def __add__(self, other):
        o = self._as_vec(other)
        if len(o) != len(self):
            raise ValueError(f"Vec dim mismatch: {len(self)} vs {len(o)}")
        return Vec((Add(a, b) for a, b in zip(self, o)))

    def __sub__(self, other):
        o = self._as_vec(other)
        if len(o) != len(self):
            raise ValueError(f"Vec dim mismatch: {len(self)} vs {len(o)}")
        return Vec((Add(a, _neg(b)) for a, b in zip(self, o)))

    def __mul__(self, scalar):
        s = _as_scalar_node(scalar)
        return Vec((Mul(s, a) for a in self))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def dot(self, other) -> Node:
        o = self._as_vec(other)
        if len(o) != len(self):
            raise ValueError(f"Vec dim mismatch: {len(self)} vs {len(o)}")
        return _sum([Mul(a, b) for a, b in zip(self, o)])

    def cross(self, other) -> "Vec":
        o = self._as_vec(other)
        if len(self) != 3 or len(o) != 3:
            raise ValueError("cross() requires 3D vectors")
        ax, ay, az = self
        bx, by, bz = o
        return Vec((_sub(Mul(ay, bz), Mul(az, by)), _sub(Mul(az, bx), Mul(ax, bz)), _sub(Mul(ax, by), Mul(ay, bx))))

# ──────────────────────────────────────────────────────────────
# Vector-calculus primitives (AST macros)
# ──────────────────────────────────────────────────────────────


def div(field: VField, *, spatial_axes: Tuple[int, ...], comps: Optional[Tuple[object, ...]] = None) -> Node:
    """Divergence of a vector field: ∇·F.

    Notes
    -----
    `spatial_axes` and `comps` must have the same length (dimension).
    """

    dim = len(tuple(spatial_axes))
    if dim < 1:
        raise ValueError("spatial_axes must be non-empty")
    comps2 = _resolve_comps(field, comps, dim)
    terms = [field.d(int(ax), comps2[i]) for i, ax in enumerate(tuple(spatial_axes))]
    return _sum(terms)


def curl(
    field: VField,
    *,
    spatial_axes: Tuple[int, int, int],
    comps: Optional[Tuple[object, object, object]] = None,
) -> Vec:
    """Curl of a 3D vector field: ∇×F.

    Returns (curl_x, curl_y, curl_z) as scalar Nodes.
    """

    ax, ay, az = (int(spatial_axes[0]), int(spatial_axes[1]), int(spatial_axes[2]))
    cx, cy, cz = _resolve_comps(field, comps, 3)

    # (∂Fz/∂y - ∂Fy/∂z,
    #  ∂Fx/∂z - ∂Fz/∂x,
    #  ∂Fy/∂x - ∂Fx/∂y)
    curl_x = _sub(field.d(ay, cz), field.d(az, cy))
    curl_y = _sub(field.d(az, cx), field.d(ax, cz))
    curl_z = _sub(field.d(ax, cy), field.d(ay, cx))
    return Vec((curl_x, curl_y, curl_z), name=f"curl({field.name})")


def grad(scalar_field: VField, *, spatial_axes: Tuple[int, int, int]) -> Vec:
    """Gradient of a scalar field: ∇φ.

    The scalar field is expected to have n_comp=1.
    """

    spec = scalar_field.spec()
    if int(spec.n_comp) != 1:
        raise ValueError(f"grad() expects a scalar field (n_comp=1), got n_comp={spec.n_comp}")

    ax, ay, az = (int(spatial_axes[0]), int(spatial_axes[1]), int(spatial_axes[2]))
    dphidx = scalar_field.d(ax, 0)
    dphidy = scalar_field.d(ay, 0)
    dphidz = scalar_field.d(az, 0)
    return Vec((dphidx, dphidy, dphidz), name=f"grad({scalar_field.name})")


def laplacian(field: VField, *, spatial_axes: Tuple[int, ...], comps: Optional[Tuple[object, ...]] = None):
    """Laplacian of a field: ∇²φ or component-wise ∇²F.

    Returns
    -------
    Node
        If field is scalar.
    tuple[Node,...]
        If field is vector (one scalar Node per component).
    """

    axes = tuple(int(a) for a in spatial_axes)
    if len(axes) < 1:
        raise ValueError("spatial_axes must be non-empty")

    spec = field.spec()
    if int(spec.n_comp) == 1:
        terms = [field.d2(a, a, 0) for a in axes]
        return _sum(terms)

    dim = len(axes)
    comps2 = _resolve_comps(field, comps, dim)
    outs: List[Node] = []
    for i in range(dim):
        terms = [field.d2(a, a, comps2[i]) for a in axes]
        outs.append(_sum(terms))
    return Vec(outs, name=f"laplacian({field.name})")


def dot(a: VField, b: VField, *, comps: Optional[Tuple[object, ...]] = None) -> Node:
    """Dot product of two vector fields: a·b."""

    spec_a = a.spec()
    spec_b = b.spec()
    if int(spec_a.n_comp) != int(spec_b.n_comp):
        raise ValueError(f"dot() requires same n_comp; got {spec_a.n_comp} vs {spec_b.n_comp}")

    dim = int(spec_a.n_comp)
    comps2 = _resolve_comps(a, comps, dim)
    terms = [Mul(a(comps2[i]), b(comps2[i])) for i in range(dim)]
    return _sum(terms)


def cross(
    a: VField,
    b: VField,
    *,
    comps: Optional[Tuple[object, object, object]] = None,
) -> Vec:
    """Cross product of two 3D vector fields: a×b."""

    spec_a = a.spec()
    spec_b = b.spec()
    if int(spec_a.n_comp) != 3 or int(spec_b.n_comp) != 3:
        raise ValueError("cross() currently supports only 3D fields")

    cx, cy, cz = _resolve_comps(a, comps, 3)

    ax = a(cx)
    ay = a(cy)
    az = a(cz)
    bx = b(cx)
    by = b(cy)
    bz = b(cz)

    # (ay*bz - az*by,
    #  az*bx - ax*bz,
    #  ax*by - ay*bx)
    out_x = _sub(Mul(ay, bz), Mul(az, by))
    out_y = _sub(Mul(az, bx), Mul(ax, bz))
    out_z = _sub(Mul(ax, by), Mul(ay, bx))
    return Vec((out_x, out_y, out_z), name=f"cross({a.name},{b.name})")


def advect(
    v: VField,
    u: VField,
    *,
    spatial_axes: Tuple[int, ...],
    comps: Optional[Tuple[object, ...]] = None,
):
    """Advection operator (v·∇)u.

    Returns a tuple of scalar Nodes for each component of u.

    Notes
    -----
    `spatial_axes` defines the derivative axes used in the dot with v.
    `comps` defines how u/v components correspond to those axes.
    """

    axes = tuple(int(a) for a in spatial_axes)
    dim = len(axes)
    if dim < 1:
        raise ValueError("spatial_axes must be non-empty")

    spec_u = u.spec()
    spec_v = v.spec()
    if int(spec_u.n_comp) < dim or int(spec_v.n_comp) < dim:
        raise ValueError(
            f"advect requires fields with at least dim={dim} components; got u:{spec_u.n_comp}, v:{spec_v.n_comp}"
        )

    comps2 = _resolve_comps(u, comps, dim)

    outs: List[Node] = []
    for i in range(dim):
        # component i of (v·∇)u is Σ_j v_j * ∂u_i/∂x_j
        terms = [Mul(v(comps2[j]), u.d(axes[j], comps2[i])) for j in range(dim)]
        outs.append(_sum(terms))
    return Vec(outs, name=f"advect({v.name},{u.name})")


# ──────────────────────────────────────────────────────────────
# Convenience library builders (flatten to scalar terms)
# ──────────────────────────────────────────────────────────────


@dataclass
class VectorOpsLibraryConfig:
    """Convenience configuration for building a vector-calculus term library."""

    spatial_axes: Tuple[int, int, int] = (1, 2, 3)
    comps: Tuple[object, object, object] | None = ("x", "y", "z")

    include_components: bool = False
    include_div: bool = False
    include_curl: bool = True
    include_laplacian: bool = False

    include_first_partials: bool = False
    include_second_partials: bool = False

    include_advection: bool = False


def build_vector_calculus_terms(fields: Sequence[VField], *, cfg: Optional[VectorOpsLibraryConfig] = None) -> List[Node]:
    """Build a flattened scalar-term library from vector-calculus operators.

    This is intentionally modest (defaults to curls only) and aims to provide
    useful, reusable operator-shaped features for many PDE systems.

    Parameters
    ----------
    fields : sequence[VField]
        Fields to include in the library.
    cfg : VectorOpsLibraryConfig | None
        Operator selection + axis mapping.

    Returns
    -------
    list[Node]
        Scalar AST terms suitable to pass as `extra_terms=` into
        :func:`~nestynet_sr.sr_de.system_de_search.discover_system_de_from_surrogate`.
    """

    if cfg is None:
        cfg = VectorOpsLibraryConfig()

    axes = tuple(int(a) for a in cfg.spatial_axes)
    comps = cfg.comps

    terms: List[Node] = []

    for F in fields:
        spec = F.spec()
        dim = min(int(spec.n_comp), len(axes))

        # Raw components
        if cfg.include_components:
            cc = _resolve_comps(F, comps, dim) if comps is not None else tuple(range(dim))
            for i in range(dim):
                terms.append(F(cc[i]))

        # Divergence (scalar)
        if cfg.include_div:
            cc = _resolve_comps(F, comps, dim) if comps is not None else tuple(range(dim))
            terms.append(div(F, spatial_axes=axes[:dim], comps=cc))

        # Curl components (3D)
        if cfg.include_curl and dim >= 3:
            cc3 = _resolve_comps(F, comps, 3) if comps is not None else (0, 1, 2)
            terms.extend(list(curl(F, spatial_axes=(axes[0], axes[1], axes[2]), comps=cc3)))

        # Laplacian
        if cfg.include_laplacian:
            cc = _resolve_comps(F, comps, dim) if comps is not None else tuple(range(dim))
            L = laplacian(F, spatial_axes=axes[:dim], comps=cc)
            if isinstance(L, tuple):
                terms.extend(list(L))
            else:
                terms.append(L)

        # Optionally include raw partials (can explode in size)
        if cfg.include_first_partials:
            cc = _resolve_comps(F, comps, dim) if comps is not None else tuple(range(dim))
            for i in range(dim):
                for a in axes[:dim]:
                    terms.append(F.d(int(a), cc[i]))

        if cfg.include_second_partials:
            cc = _resolve_comps(F, comps, dim) if comps is not None else tuple(range(dim))
            for i in range(dim):
                for a in axes[:dim]:
                    terms.append(F.d2(int(a), int(a), cc[i]))

    # Optional: self-advection for each field (useful for fluids)
    if cfg.include_advection:
        for F in fields:
            spec = F.spec()
            dim = min(int(spec.n_comp), len(axes))
            if dim >= 2:
                cc = _resolve_comps(F, comps, dim) if comps is not None else tuple(range(dim))
                adv = advect(F, F, spatial_axes=axes[:dim], comps=cc)
                if isinstance(adv, tuple):
                    terms.extend(list(adv))

    return _dedup(terms)


def build_maxwell_candidate_terms(
    E: VField,
    B: VField,
    *,
    spatial_axes: Tuple[int, int, int] = (1, 2, 3),
    comps: Tuple[object, object, object] | None = ("x", "y", "z"),
    include_div: bool = False,
) -> List[Node]:
    """Return a small Maxwell-shaped candidate term set (still scalar AST terms).

    By default this returns only curl(E) and curl(B) components, which is the
    "dynamic" heart of Maxwell (Faraday + Ampère–Maxwell). Optionally include
    divergences for Gauss constraints.

    This function is deliberately thin: it is here mainly as a fun example
    of the more general vector-calculus operators in this module.
    """

    axes = tuple(int(a) for a in spatial_axes)
    cc3 = _resolve_comps(E, comps, 3) if comps is not None else (0, 1, 2)

    terms: List[Node] = []
    terms.extend(list(curl(E, spatial_axes=(axes[0], axes[1], axes[2]), comps=cc3)))
    terms.extend(list(curl(B, spatial_axes=(axes[0], axes[1], axes[2]), comps=cc3)))

    if include_div:
        terms.append(div(E, spatial_axes=axes, comps=cc3))
        terms.append(div(B, spatial_axes=axes, comps=cc3))

    return _dedup(terms)
